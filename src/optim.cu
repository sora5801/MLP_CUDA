// ============================================================================
//  src/optim.cu                                               (added in push 0002)
// ----------------------------------------------------------------------------
//  ROLE IN THE PROJECT
//  Implements the optimizer abstraction declared in include/optim.cuh: the two
//  stateful update kernels (Momentum and Adam) plus the host-side glue that
//  allocates per-parameter state, advances the timestep, and applies one update
//  to every weight and bias in the network. SGD reuses the existing
//  launch_sgd_update from kernels.cu, so it needs no kernel here.
//
//  WHY OPTIMIZERS MATTER (the lesson)
//  Plain SGD takes a fixed step downhill along the raw gradient. That is simple
//  but brittle: it oscillates across steep, narrow valleys and crawls along
//  shallow ones, and a single learning rate must suit every parameter at once.
//    * Momentum accumulates a velocity (an exponentially-decayed running sum of
//      gradients). Consistent directions build up speed; oscillating directions
//      cancel out — like a heavy ball rolling through the loss surface.
//    * Adam additionally divides each parameter's step by the running RMS of its
//      own gradients, giving every parameter an ADAPTIVE per-element step size,
//      and adds bias correction so the early steps (when the running averages are
//      still near their zero initialization) are not artificially tiny.
//
//  CUDA ANGLE
//  Every update is a 1-D element-wise kernel (one thread per parameter element),
//  the same launch idiom as sgd_update — the interesting part is the persistent
//  device STATE buffers that the kernels read-modify-write each step.
// ============================================================================

#include "optim.cuh"     // OptConfig / Optimizer / the kernels we define here
#include "kernels.cuh"   // launch_sgd_update (SGD reuses it); ceil_div via common
#include "common.cuh"    // CUDA_CHECK / CUDA_CHECK_LAST, kBlockSize, ceil_div
#include "matrix.cuh"    // matrix_alloc / matrix_zero / matrix_free for state

#include <cmath>         // std::pow for the Adam bias-correction terms
#include <new>           // (implicit) array new for the state pointer arrays

// ----------------------------------------------------------------------------
//  Config constructors (textbook defaults live here, documented once).
// ----------------------------------------------------------------------------
OptConfig opt_sgd(float lr) {
    // Stateless: only lr is meaningful; the rest are filled with harmless values.
    return OptConfig{ OptType::SGD, lr, /*momentum*/0.0f,
                      /*beta1*/0.0f, /*beta2*/0.0f, /*eps*/0.0f };
}
OptConfig opt_momentum(float lr, float momentum) {
    return OptConfig{ OptType::Momentum, lr, momentum,
                      /*beta1*/0.0f, /*beta2*/0.0f, /*eps*/0.0f };
}
OptConfig opt_adam(float lr, float beta1, float beta2, float eps) {
    return OptConfig{ OptType::Adam, lr, /*momentum*/0.0f, beta1, beta2, eps };
}

const char* opt_name(OptType type) {
    switch (type) {
        case OptType::SGD:      return "SGD";
        case OptType::Momentum: return "Momentum";
        case OptType::Adam:     return "Adam";
    }
    return "Unknown";
}

// ============================================================================
//  momentum_update  —  velocity = mu*velocity + grad ;  param -= lr*velocity.
// ============================================================================
//
// SHAPES: param, grad, velocity are flat arrays of length n (param & velocity
// updated in place). mu is the velocity decay (e.g. 0.9), lr the step size.
//
// Each thread owns one parameter element: it reads the old velocity, blends in
// the new gradient, writes the velocity back (this is the persistent state), and
// then steps the parameter along the velocity. Because velocity starts at 0, the
// very first step equals lr*grad (plain SGD); momentum builds up over time.
__global__ void momentum_update(float* param, const float* grad, float* velocity,
                                float lr, float mu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // flat parameter index
    if (i >= n) return;                               // guard padding threads
    float v = mu * velocity[i] + grad[i];             // decayed running gradient
    velocity[i] = v;                                  // persist the new velocity
    param[i] -= lr * v;                               // step along the velocity
}

void launch_momentum_update(float* param, const float* grad, float* velocity,
                            float lr, float mu, int n) {
    int block = kBlockSize;
    int grid  = ceil_div(n, block);
    momentum_update<<<grid, block>>>(param, grad, velocity, lr, mu, n);
    CUDA_CHECK_LAST();
}

// ============================================================================
//  adam_update  —  adaptive per-parameter step with bias-corrected moments.
// ============================================================================
//
// The Adam rule for each parameter element i (Kingma & Ba, 2015):
//   m[i] = beta1*m[i] + (1-beta1)*grad[i]            // 1st moment: mean of grads
//   v[i] = beta2*v[i] + (1-beta2)*grad[i]^2          // 2nd moment: mean of grad^2
//   mhat = m[i] / (1 - beta1^t)                      // bias correction (1st)
//   vhat = v[i] / (1 - beta2^t)                      // bias correction (2nd)
//   param[i] -= lr * mhat / (sqrt(vhat) + eps)
//
// WHY BIAS CORRECTION: m and v start at 0, so for small t the running averages are
// biased toward 0 (too small). Dividing by (1 - beta^t) — which is small early and
// → 1 as t grows — rescales them to be unbiased estimates. The host computes the
// two correction DENOMINATORS (1 - beta1^t) and (1 - beta2^t) once per step (they
// are the same for every element) and passes them in, so the kernel stays branch-
// and pow-free.
//
// INTUITION FOR THE STEP: dividing by sqrt(vhat) normalizes each parameter's step
// by the typical magnitude of its own recent gradients — large-gradient params get
// damped, small-gradient params get amplified — so a single lr works across very
// differently-scaled parameters. eps avoids division by zero when vhat ≈ 0.
//
// SHAPES: param, grad, m, v are flat length-n arrays (param, m, v updated in place).
__global__ void adam_update(float* param, const float* grad, float* m, float* v,
                            float lr, float beta1, float beta2, float eps,
                            float bias_correction1, float bias_correction2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // flat parameter index
    if (i >= n) return;                               // guard padding threads

    float g = grad[i];
    // Update the running 1st and 2nd raw moments (persistent state).
    float m_i = beta1 * m[i] + (1.0f - beta1) * g;
    float v_i = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_i;
    v[i] = v_i;

    // Bias-corrected estimates (host supplied the denominators 1 - beta^t).
    float m_hat = m_i / bias_correction1;
    float v_hat = v_i / bias_correction2;

    // Adaptive step. sqrtf is the single-precision device intrinsic.
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void launch_adam_update(float* param, const float* grad, float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        float bias_correction1, float bias_correction2, int n) {
    int block = kBlockSize;
    int grid  = ceil_div(n, block);
    adam_update<<<grid, block>>>(param, grad, m, v, lr, beta1, beta2, eps,
                                 bias_correction1, bias_correction2, n);
    CUDA_CHECK_LAST();
}

// ============================================================================
//  optim_create  —  allocate + zero the state buffers this rule needs.
// ============================================================================
//
// We mirror the network's shape: for each layer l we may allocate state buffers
// matching W[in,out] and b[1,out]. SGD allocates nothing (all pointers stay null);
// Momentum allocates the velocity (vW/vb); Adam allocates both moments
// (mW/vW for weights, mb/vb for biases). Every buffer is zeroed, per the standard
// "moments/velocity start at 0" convention. The host arrays of Matrix are plain
// `new[]` allocations (host bookkeeping); the Matrix.data inside each lives on the
// device.
Optimizer optim_create(const MLP& net, OptConfig cfg) {
    Optimizer opt;
    opt.cfg        = cfg;
    opt.t          = 0;
    opt.num_layers = net.num_layers;

    // Default everything to null; we only allocate what the rule needs below.
    opt.vW = opt.vb = opt.mW = opt.mb = nullptr;

    const bool need_v = (cfg.type == OptType::Momentum || cfg.type == OptType::Adam);
    const bool need_m = (cfg.type == OptType::Adam);

    if (need_v) {
        opt.vW = new Matrix[opt.num_layers];
        opt.vb = new Matrix[opt.num_layers];
    }
    if (need_m) {
        opt.mW = new Matrix[opt.num_layers];
        opt.mb = new Matrix[opt.num_layers];
    }

    for (int l = 0; l < opt.num_layers; ++l) {
        const Layer& L = net.layers[l];
        const int in  = L.in_features;
        const int out = L.out_features;
        if (need_v) {
            opt.vW[l] = matrix_alloc(in, out); matrix_zero(opt.vW[l]); // velocity / v
            opt.vb[l] = matrix_alloc(1,  out); matrix_zero(opt.vb[l]);
        }
        if (need_m) {
            opt.mW[l] = matrix_alloc(in, out); matrix_zero(opt.mW[l]); // 1st moment m
            opt.mb[l] = matrix_alloc(1,  out); matrix_zero(opt.mb[l]);
        }
    }
    return opt;
}

// ============================================================================
//  optim_free  —  release all state buffers and the host arrays.
// ============================================================================
void optim_free(Optimizer& opt) {
    for (int l = 0; l < opt.num_layers; ++l) {
        if (opt.vW) { matrix_free(opt.vW[l]); matrix_free(opt.vb[l]); }
        if (opt.mW) { matrix_free(opt.mW[l]); matrix_free(opt.mb[l]); }
    }
    delete[] opt.vW; delete[] opt.vb;   // delete[] nullptr is safe (no-op)
    delete[] opt.mW; delete[] opt.mb;
    opt.vW = opt.vb = opt.mW = opt.mb = nullptr;
    opt.num_layers = 0;
    opt.t = 0;
}

// ============================================================================
//  optim_step  —  apply ONE update to every weight and bias of `net`.
// ============================================================================
//
// Precondition: mlp_backward just filled every layer's dW/db. We bump the
// timestep, precompute Adam's bias-correction denominators on the host (the same
// scalars for every element this step), and per layer launch the chosen update on
// both W (with dW + its state) and b (with db + its state).
void optim_step(Optimizer& opt, MLP& net) {
    opt.t += 1;   // advance the global timestep (Adam's bias correction uses it)
    const OptConfig& c = opt.cfg;

    // Adam-only: 1 - beta^t. Computed once per step in double for accuracy, then
    // passed to every kernel launch this step. (Harmless to compute even for the
    // other rules; we just don't use it there.)
    const float bc1 = static_cast<float>(1.0 - std::pow((double)c.beta1, (double)opt.t));
    const float bc2 = static_cast<float>(1.0 - std::pow((double)c.beta2, (double)opt.t));

    for (int l = 0; l < net.num_layers; ++l) {
        Layer& L = net.layers[l];
        const int nW = L.in_features * L.out_features;  // #weight elements
        const int nB = L.out_features;                  // #bias elements

        switch (c.type) {
            case OptType::SGD:
                // Stateless: reuse the push-0001 kernel. Equivalent to mlp_sgd_step.
                launch_sgd_update(L.W.data, L.dW.data, c.lr, nW);
                launch_sgd_update(L.b.data, L.db.data, c.lr, nB);
                break;

            case OptType::Momentum:
                launch_momentum_update(L.W.data, L.dW.data, opt.vW[l].data,
                                       c.lr, c.momentum, nW);
                launch_momentum_update(L.b.data, L.db.data, opt.vb[l].data,
                                       c.lr, c.momentum, nB);
                break;

            case OptType::Adam:
                launch_adam_update(L.W.data, L.dW.data, opt.mW[l].data, opt.vW[l].data,
                                   c.lr, c.beta1, c.beta2, c.eps, bc1, bc2, nW);
                launch_adam_update(L.b.data, L.db.data, opt.mb[l].data, opt.vb[l].data,
                                   c.lr, c.beta1, c.beta2, c.eps, bc1, bc2, nB);
                break;
        }
    }
}
