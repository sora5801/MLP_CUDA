// ============================================================================
//  include/optim.cuh                                          (added in push 0002)
// ----------------------------------------------------------------------------
//  ROLE IN THE PROJECT
//  Declares pluggable gradient-descent OPTIMIZERS that consume the gradients
//  produced by mlp_backward and update the network's weights and biases. Push
//  0001 only had plain SGD (param -= lr*grad, implemented by mlp_sgd_step). This
//  header generalizes that into three choices:
//
//      SGD       : param -= lr * grad                         (no state)
//      Momentum  : v = mu*v + grad;  param -= lr * v          (one state buffer)
//      Adam      : per-parameter adaptive step from 1st & 2nd
//                  moment estimates with bias correction       (two state buffers)
//
//  THE CENTRAL NEW IDEA: OPTIMIZER STATE.
//  SGD is stateless — given the gradient it knows the whole update. Momentum and
//  Adam are STATEFUL: they keep running averages (a "velocity" for Momentum; first
//  and second moment estimates m, v for Adam) that PERSIST across training steps.
//  That state has exactly the same shape as the parameters, so for every weight
//  matrix W[in,out] and bias b[1,out] in the network we allocate matching device
//  buffers. The Optimizer struct below owns those buffers (parallel arrays indexed
//  by layer), allocates+zeroes them in optim_create, and frees them in optim_free.
//
//  This file only DECLARES the API and the update kernels; src/optim.cu defines
//  them, and src/kernels.cu owns the actual __global__ update kernels' twins for
//  SGD (we reuse launch_sgd_update). Everything operates on device pointers.
// ============================================================================
#pragma once

#include "matrix.cuh"   // Matrix (device buffer + dims) for the state arrays
#include "mlp.cuh"       // MLP / Layer: optim_create mirrors the layer shapes

// ----------------------------------------------------------------------------
//  enum class OptType — which update rule to use.
// ----------------------------------------------------------------------------
enum class OptType { SGD, Momentum, Adam };

// ----------------------------------------------------------------------------
//  struct OptConfig — the chosen rule plus its hyperparameters.
// ----------------------------------------------------------------------------
//  Not every field is used by every rule (e.g. SGD ignores momentum/beta/eps);
//  the helper constructors below fill in sensible, conventional defaults so call
//  sites read clearly, e.g. `opt_adam(1e-3f)`.
struct OptConfig {
    OptType type;     // which update rule
    float   lr;       // learning rate (step size); used by all three
    float   momentum; // Momentum: the velocity decay mu (e.g. 0.9). Unused by SGD/Adam.
    float   beta1;    // Adam: decay for the 1st moment (mean of grads), e.g. 0.9
    float   beta2;    // Adam: decay for the 2nd moment (mean of grad^2), e.g. 0.999
    float   eps;      // Adam: small constant in the denominator for stability, e.g. 1e-8
};

// Convenience constructors with the textbook default hyperparameters. They keep
// main.cu readable and put the "magic numbers" in exactly one documented place.
OptConfig opt_sgd(float lr);
OptConfig opt_momentum(float lr, float momentum = 0.9f);
OptConfig opt_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                   float eps = 1e-8f);

// Human-readable name ("SGD" / "Momentum" / "Adam") for logging.
const char* opt_name(OptType type);

// ----------------------------------------------------------------------------
//  struct Optimizer — the configured rule plus its per-parameter STATE.
// ----------------------------------------------------------------------------
//  The state arrays are parallel to the network's layers: index l holds the state
//  for layer l's weight (…W) and bias (…b). The buffers are:
//    Momentum : vW[l], vb[l]            — the velocity (running grad accumulator)
//    Adam     : mW[l], vW[l], mb[l], vb[l]
//               m* = 1st moment (mean of grads), v* = 2nd moment (mean of grad^2)
//    SGD      : no buffers are allocated (all pointers stay null).
//  Each buffer has the SAME shape as the parameter it tracks and is zero-init'd,
//  matching the standard "start moments/velocity at 0" convention.
struct Optimizer {
    OptConfig cfg;        // the rule + hyperparameters
    long long t;          // timestep counter (1,2,3,…); Adam uses it for bias correction
    int       num_layers; // == MLP::num_layers; length of every array below

    // State arrays (length num_layers each). Allocated only for the rules that
    // need them; otherwise the pointer is nullptr and the entries are unused.
    Matrix* vW;  // velocity (Momentum) OR 2nd-moment v (Adam) for each W
    Matrix* vb;  // …same, for each bias b
    Matrix* mW;  // 1st-moment m (Adam only) for each W; nullptr otherwise
    Matrix* mb;  // …same, for each bias b
};

// ----------------------------------------------------------------------------
//  optim_create — allocate + zero the state needed by `cfg.type`, sized to `net`.
// ----------------------------------------------------------------------------
//  Walks the network's layers and, per layer, allocates state buffers matching
//  each W[in,out] and b[1,out]. SGD allocates nothing. Momentum allocates vW/vb.
//  Adam allocates mW/vW/mb/vb. All buffers are zeroed (matrix_zero). t starts 0.
//  The returned Optimizer must be released with optim_free.
Optimizer optim_create(const MLP& net, OptConfig cfg);

// Free every allocated state buffer and the parallel arrays; reset fields.
void optim_free(Optimizer& opt);

// ----------------------------------------------------------------------------
//  optim_step — apply ONE update to every weight and bias of `net`.
// ----------------------------------------------------------------------------
//  Precondition: mlp_backward has just filled every layer's dW/db. This advances
//  opt.t and, per layer, launches the update kernel for opt.cfg.type on both W
//  (using dW + its state) and b (using db + its state). After it returns, the
//  network's parameters have taken one optimizer step and the state buffers hold
//  the updated running averages. This is the drop-in replacement for the push
//  0001 mlp_sgd_step (which still exists and equals optim_step with an SGD config).
void optim_step(Optimizer& opt, MLP& net);

// ----------------------------------------------------------------------------
//  Update kernels + launchers (defined in src/optim.cu).
// ----------------------------------------------------------------------------
//  (SGD reuses sgd_update / launch_sgd_update from kernels.cuh, so it has no twin
//   here.) Both kernels are 1-D element-wise: one thread per parameter element.

// Momentum ("heavy ball"): velocity[i] = mu*velocity[i] + grad[i];
//                          param[i]   -= lr * velocity[i].
// The velocity is an exponentially-decayed running sum of gradients, which
// smooths the path and accelerates along consistent directions.
__global__ void momentum_update(float* param, const float* grad, float* velocity,
                                float lr, float mu, int n);
void launch_momentum_update(float* param, const float* grad, float* velocity,
                            float lr, float mu, int n);

// Adam: maintains m (1st moment) and v (2nd moment) per parameter, with
// bias-corrected estimates, giving each parameter its own adaptive step size:
//   m[i] = beta1*m[i] + (1-beta1)*grad[i]
//   v[i] = beta2*v[i] + (1-beta2)*grad[i]^2
//   mhat = m[i]/bias_correction1 ;  vhat = v[i]/bias_correction2
//   param[i] -= lr * mhat / (sqrt(vhat) + eps)
// The host passes the precomputed bias-correction terms (1 - beta^t) so the
// kernel does not need the timestep t. See src/optim.cu for the full derivation.
__global__ void adam_update(float* param, const float* grad, float* m, float* v,
                            float lr, float beta1, float beta2, float eps,
                            float bias_correction1, float bias_correction2, int n);
void launch_adam_update(float* param, const float* grad, float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        float bias_correction1, float bias_correction2, int n);
