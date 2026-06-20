// ============================================================================
//  src/mlp.cu
// ----------------------------------------------------------------------------
//  ROLE IN THE PROJECT
//  This file is the "brain" of the repository: it ties the individual CUDA
//  kernels (declared in kernels.cuh) and the device-memory helpers (matrix.cuh)
//  together into a working Multi-Layer Perceptron with a full forward pass, a
//  full backward pass (backpropagation), an SGD parameter update, loss and
//  accuracy reporting, and — crucially for a *study* repo — a finite-difference
//  gradient check that proves the analytic gradients are correct without any
//  reference framework like PyTorch.
//
//  Nothing in here launches a kernel directly; we only ever call the `launch_*`
//  wrappers from kernels.cuh. Each wrapper internally computes its own grid/block
//  configuration and calls CUDA_CHECK_LAST() to surface launch errors. That keeps
//  this file focused on the *algorithm* (linear algebra + chain rule) rather than
//  on CUDA launch bookkeeping.
//
//  MATH/MEMORY CONVENTIONS (from the build spec — never deviate):
//    * All matrices are row-major flat float arrays in GPU global memory.
//    * A weight matrix W for a layer with `in` inputs / `out` outputs has shape
//      [in, out]; element W[i*out + o].
//    * A batch of activations A has shape [batch, features]; element A[r*feat + c].
//    * Bias b has shape [1, out]; element b[o].
//    * Forward linear layer:  Z = A_prev · W + b  (NO transpose in forward).
//    * Loss is the mean (over the batch) softmax cross-entropy. The 1/batch
//      factor is folded into the output-layer gradient by cross_entropy_grad, so
//      every downstream gradient already carries the 1/batch scale and the SGD
//      update is simply  param -= lr * grad.
// ============================================================================

#include "mlp.cuh"      // Layer/MLP structs + the public API we implement here
#include "matrix.cuh"   // Matrix struct + matrix_alloc/free/zero/copy helpers
#include "kernels.cuh"  // launch_gemm / launch_add_bias / launch_relu_* / ...
#include "common.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST, ceil_div, constants

#include <random>       // std::mt19937_64, std::normal_distribution (He init)
#include <cmath>        // std::sqrt, std::log, std::fabs
#include <cstdio>       // std::printf for grad-check reporting
#include <vector>       // std::vector for host-side scratch buffers

// Negative-side slope used by the LeakyReLU activation (push 0002). A small
// constant (0.01 is the conventional default) so dead-unit gradients stay
// nonzero. Kept here as a single named knob rather than a magic number sprinkled
// through the dispatch code below.
static constexpr float kLeakyAlpha = 0.01f;

// ----------------------------------------------------------------------------
//  mlp_create
// ----------------------------------------------------------------------------
//  Build an MLP from a list of layer widths.
//
//  Parameters:
//    layer_sizes : host int array, length `num_sizes`. e.g. {2, 64, 32, 3} means
//                  2 input features -> hidden 64 -> hidden 32 -> 3 output classes.
//                  Units: number of neurons / features per stage.
//    num_sizes   : length of layer_sizes (>= 2). We create (num_sizes - 1) layers.
//    batch_size  : number of samples processed at once. Activation caches (Z, A,
//                  dZ, dA) are all sized [batch_size, out_features] because every
//                  forward/backward op runs on a whole batch.
//    seed        : seed for std::mt19937_64 so weight initialization is fully
//                  reproducible (the spec requires deterministic randomness).
//
//  Returns: a fully-allocated MLP. All device buffers are allocated here; weights
//  are He-initialized on the host then copied H2D; biases are zeroed on device.
//
//  WHY He initialization (N(0, sqrt(2/in)))? ReLU zeroes out roughly half its
//  inputs, halving the variance of the signal as it flows forward. Scaling the
//  initial weights by sqrt(2/in) compensates for that so the activation variance
//  stays ~constant across layers, which keeps gradients from vanishing/exploding
//  early in training. `in` is the fan-in (number of inputs to the neuron).
MLP mlp_create(const int* layer_sizes, int num_sizes, int batch_size,
               unsigned long long seed, Activation hidden_act) {
    MLP net;
    net.num_layers      = num_sizes - 1;          // edges between the size nodes
    net.batch_size      = batch_size;
    net.input_features  = layer_sizes[0];         // width of the input stage
    net.num_classes     = layer_sizes[num_sizes - 1]; // width of the final stage

    // Allocate the array of Layer structs on the HOST. The Matrix members inside
    // each Layer hold *device* pointers, but the Layer structs themselves are
    // plain host-side bookkeeping (dimensions + the device pointers).
    net.layers = new Layer[net.num_layers];

    // One deterministic RNG drives ALL weight draws so the whole network init is
    // reproducible from `seed` alone.
    std::mt19937_64 rng(seed);

    for (int l = 0; l < net.num_layers; ++l) {
        Layer& L = net.layers[l];
        L.in_features  = layer_sizes[l];          // fan-in  of this layer
        L.out_features = layer_sizes[l + 1];      // fan-out of this layer
        // The LAST layer is the softmax output layer; all others use the chosen
        // hidden activation. We still record `activation` on the output layer for
        // completeness, but is_output makes the forward/backward code use softmax
        // there regardless of this field.
        L.is_output    = (l == net.num_layers - 1);
        L.activation   = hidden_act;

        const int in  = L.in_features;
        const int out = L.out_features;

        // --- Allocate all per-layer device buffers (uninitialized) -----------
        // Parameters:
        L.W  = matrix_alloc(in, out);     // weights      [in,  out]
        L.b  = matrix_alloc(1, out);      // bias         [1,   out]
        // Forward caches (needed again during backprop):
        L.Z  = matrix_alloc(batch_size, out); // pre-activation  [batch, out]
        L.A  = matrix_alloc(batch_size, out); // post-activation [batch, out]
        // Gradients:
        L.dW = matrix_alloc(in, out);         // dL/dW   [in,  out]
        L.db = matrix_alloc(1, out);          // dL/db   [1,   out]
        L.dZ = matrix_alloc(batch_size, out); // dL/dZ   [batch, out]
        L.dA = matrix_alloc(batch_size, out); // dL/dA   [batch, out]

        // --- He-initialize the weights on the host, then copy to the device --
        // stddev = sqrt(2 / fan_in). Mean 0. (See WHY note above.)
        const float stddev = std::sqrt(2.0f / static_cast<float>(in));
        std::normal_distribution<float> dist(0.0f, stddev);

        // Host scratch in the SAME row-major [in, out] layout the device expects.
        std::vector<float> h_W(static_cast<size_t>(in) * out);
        for (size_t i = 0; i < h_W.size(); ++i) {
            h_W[i] = dist(rng);           // one Gaussian sample per weight
        }
        matrix_copy_to_device(L.W, h_W.data()); // H2D: in*out floats

        // Biases start at exactly zero (standard practice; the He scheme only
        // governs the weights). matrix_zero issues a cudaMemset(0) on device.
        matrix_zero(L.b);
    }

    return net; // returned by value: Matrix members are just {ptr,rows,cols}.
}

// ----------------------------------------------------------------------------
//  mlp_free
// ----------------------------------------------------------------------------
//  Release every device buffer owned by the network, then free the host-side
//  Layer array. Mirror of mlp_create. After this the MLP must not be used.
//
//  matrix_free both cudaFree's the device pointer and resets the struct fields
//  to {nullptr,0,0}, so double-free is harmless.
void mlp_free(MLP& net) {
    if (net.layers != nullptr) {
        for (int l = 0; l < net.num_layers; ++l) {
            Layer& L = net.layers[l];
            matrix_free(L.W);
            matrix_free(L.b);
            matrix_free(L.Z);
            matrix_free(L.A);
            matrix_free(L.dW);
            matrix_free(L.db);
            matrix_free(L.dZ);
            matrix_free(L.dA);
        }
        delete[] net.layers;   // host array allocated with new[] in mlp_create
        net.layers = nullptr;
    }
    net.num_layers = 0;
}

// ----------------------------------------------------------------------------
//  mlp_forward
// ----------------------------------------------------------------------------
//  Run the forward pass over a device batch.
//
//  Parameters:
//    net          : the network (its layer caches Z/A get overwritten).
//    batch_input  : device Matrix [batch, input_features] — one mini-batch.
//
//  Postcondition: net.layers[num_layers-1].A holds the class probabilities,
//  shape [batch, num_classes] (each row sums to 1 via softmax).
//
//  Algorithm (verbatim from the build spec):
//    prev = batch_input
//    for l in 0..num_layers-1:
//        Z = prev · W + b
//        A = is_output ? softmax(Z) : relu(Z)
//        prev = A
//
//  GEMM shape bookkeeping for the linear step `Z = prev · W`:
//    prev is [batch, in], W is [in, out], so Z is [batch, out]. In launch_gemm's
//    (M, N, K) terms: M=batch (rows of C), N=out (cols of C), K=in (shared dim).
//    No transposes — both operands are already stored in the natural orientation.
void mlp_forward(MLP& net, const Matrix& batch_input) {
    const int batch = net.batch_size;

    // `prev` is the activation feeding the current layer. For layer 0 that is the
    // raw input batch; thereafter it is the previous layer's post-activation A.
    const Matrix* prev = &batch_input;

    for (int l = 0; l < net.num_layers; ++l) {
        Layer& L = net.layers[l];
        const int in  = L.in_features;
        const int out = L.out_features;

        // Z = prev · W            C[M,N]=A[M,K]·B[K,N] with M=batch,N=out,K=in.
        // transA=false (prev stored [batch,in]), transB=false (W stored [in,out]).
        launch_gemm(prev->data, L.W.data, L.Z.data,
                    /*M=*/batch, /*N=*/out, /*K=*/in,
                    /*transA=*/false, /*transB=*/false);

        // Z[r,o] += b[o]  — broadcast the length-`out` bias across all batch rows.
        launch_add_bias(L.Z.data, L.b.data, batch, out);

        // Nonlinearity. The output layer turns logits into a probability
        // distribution per row (softmax); hidden layers apply elementwise ReLU.
        if (L.is_output) {
            launch_softmax_rows(L.Z.data, L.A.data, batch, out);
        } else {
            // Hidden layer: dispatch on the configured activation (push 0002).
            // All of these are purely elementwise, so we treat Z/A as flat arrays
            // of batch*out elements (the 2-D shape is irrelevant to act(x)).
            const int m = batch * out;
            switch (L.activation) {
                case Activation::ReLU:
                    launch_relu_forward(L.Z.data, L.A.data, m);
                    break;
                case Activation::LeakyReLU:
                    launch_leaky_relu_forward(L.Z.data, L.A.data, m, kLeakyAlpha);
                    break;
                case Activation::Tanh:
                    launch_tanh_forward(L.Z.data, L.A.data, m);
                    break;
            }
        }

        // The output of this layer becomes the input to the next.
        prev = &L.A;
    }
}

// ----------------------------------------------------------------------------
//  mlp_backward
// ----------------------------------------------------------------------------
//  Backpropagation: fill every layer's dW and db (and the intermediate dZ/dA)
//  given that mlp_forward has ALREADY been run on the SAME batch_input.
//
//  Parameters:
//    net          : network with valid forward caches (Z, A) from mlp_forward.
//    batch_input  : the SAME device batch [batch, input_features] used in forward.
//                   Needed because dW of layer 0 uses the raw input as A_prev.
//    d_labels     : device int array length `batch`; true class index per sample.
//
//  Math being implemented (see docs/math_derivation.md):
//    Output layer:   dZ_out = (softmax(Z) - onehot(y)) / batch
//                    (the /batch folds in the batch-mean of the loss).
//    For each layer (top to bottom), letting A_prev be its input activation:
//      dW       = A_prev^T · dZ            shapes [in,out]=[in,batch]·[batch,out]
//      db       = column-sum of dZ over batch rows     -> [1,out]
//    And to propagate to the layer below (if any):
//      dA_prev  = dZ · W^T                 shapes [batch,in]=[batch,out]·[out,in]
//      dZ_prev  = dA_prev ⊙ relu'(Z_prev)  (elementwise; relu' is 1 where Z>0)
//
//  TRANSPOSE FLAGS — these must match the spec exactly because launch_gemm only
//  reads physical arrays and reinterprets indices based on transA/transB:
//    * dW = A_prev^T · dZ : A_prev is stored [batch,in] but we need it as [in,batch]
//      logically, so transA=TRUE. dZ is stored [batch,out] used as-is, transB=FALSE.
//      (M=in, N=out, K=batch.)
//    * dA_prev = dZ · W^T : dZ stored [batch,out] used as-is, transA=FALSE. W is
//      stored [in,out] but we need [out,in] logically, so transB=TRUE.
//      (M=batch, N=in, K=out.)
void mlp_backward(MLP& net, const Matrix& batch_input, const int* d_labels) {
    const int batch       = net.batch_size;
    const int num_classes = net.num_classes;
    const int last        = net.num_layers - 1;

    // --- Seed the chain rule at the output layer ----------------------------
    // dZ_out = (probs - onehot(label)) / batch. cross_entropy_grad reads the
    // cached output probabilities (out.A) and the labels, and writes the scaled
    // gradient directly into out.dZ. The division by `batch` here is the ONLY
    // place the batch-mean enters; every gradient below inherits that scale.
    Layer& out = net.layers[last];
    launch_cross_entropy_grad(out.A.data, d_labels, out.dZ.data,
                              batch, num_classes);

    // --- Walk layers from the output down to the input ----------------------
    for (int l = last; l >= 0; --l) {
        Layer& L = net.layers[l];
        const int in  = L.in_features;
        const int out_f = L.out_features;

        // A_prev is what fed THIS layer in the forward pass: the raw input for
        // layer 0, otherwise the previous layer's post-activation A.
        const Matrix& prev_act = (l == 0) ? batch_input
                                          : net.layers[l - 1].A;

        // dW = A_prev^T · dZ.   M=in, N=out, K=batch.  transA=true, transB=false.
        // prev_act is physically [batch,in]; transA=true makes launch_gemm read
        // it as the logical [in,batch] operand.
        launch_gemm(prev_act.data, L.dZ.data, L.dW.data,
                    /*M=*/in, /*N=*/out_f, /*K=*/batch,
                    /*transA=*/true, /*transB=*/false);

        // db = colsum(dZ): db[c] = sum over batch rows of dZ[r,c]. One value per
        // output neuron; result shape [1,out].
        launch_bias_grad(L.dZ.data, L.db.data, batch, out_f);

        // Propagate the gradient to the layer below, unless we are at layer 0
        // (the input has no learnable parameters / no gradient to compute).
        if (l > 0) {
            Layer& prevL = net.layers[l - 1];
            // Sanity in comments: L.in == prevL.out_features (the layers chain).

            // dA_prev = dZ · W^T.  M=batch, N=in, K=out.  transA=false, transB=true.
            // W is physically [in,out]; transB=true makes it the logical [out,in].
            // The result lands in the previous layer's dA buffer [batch,in].
            launch_gemm(L.dZ.data, L.W.data, prevL.dA.data,
                        /*M=*/batch, /*N=*/in, /*K=*/out_f,
                        /*transA=*/false, /*transB=*/true);

            // dZ_prev = dA_prev ⊙ act'(Z_prev): turn the gradient wrt the previous
            // layer's OUTPUT (dA) into the gradient wrt its PRE-activation (dZ) by
            // multiplying through the activation derivative. Dispatch on the
            // previous layer's activation (push 0002). KEY SUBTLETY:
            //   - ReLU / LeakyReLU gate on the PRE-activation Z_prev (sign of z).
            //   - Tanh uses the POST-activation A_prev, because tanh'(z)=1-a^2 is
            //     written in terms of a=tanh(z). Passing the wrong tensor here is
            //     a classic, silent backprop bug — the grad-check would catch it.
            // Treated as a flat array of batch*prevL.out_features elements.
            const int m_prev = batch * prevL.out_features;
            switch (prevL.activation) {
                case Activation::ReLU:
                    launch_relu_backward(prevL.dA.data, prevL.Z.data,
                                         prevL.dZ.data, m_prev);
                    break;
                case Activation::LeakyReLU:
                    launch_leaky_relu_backward(prevL.dA.data, prevL.Z.data,
                                               prevL.dZ.data, m_prev, kLeakyAlpha);
                    break;
                case Activation::Tanh:
                    // NOTE: passes prevL.A (the tanh OUTPUT), not prevL.Z.
                    launch_tanh_backward(prevL.dA.data, prevL.A.data,
                                         prevL.dZ.data, m_prev);
                    break;
            }
        }
    }
}

// ----------------------------------------------------------------------------
//  mlp_sgd_step
// ----------------------------------------------------------------------------
//  In-place vanilla SGD update for every weight and bias:  param -= lr * grad.
//
//  Parameters:
//    net : network whose dW/db were just filled by mlp_backward.
//    lr  : learning rate (step size). Units: dimensionless multiplier on grad.
//
//  Because the 1/batch batch-mean was already folded into the gradients (in
//  cross_entropy_grad), there is NO extra scaling here — the gradient is already
//  the gradient of the *mean* loss, so a plain step is correct.
void mlp_sgd_step(MLP& net, float lr) {
    for (int l = 0; l < net.num_layers; ++l) {
        Layer& L = net.layers[l];
        const int in  = L.in_features;
        const int out = L.out_features;

        // Update the weight matrix: in*out elements.
        launch_sgd_update(L.W.data, L.dW.data, lr, in * out);
        // Update the bias vector: out elements.
        launch_sgd_update(L.b.data, L.db.data, lr, out);
    }
}

// ----------------------------------------------------------------------------
//  mlp_compute_loss
// ----------------------------------------------------------------------------
//  Mean cross-entropy loss over the current batch, using the output
//  probabilities cached by the most recent mlp_forward.
//
//  Parameters:
//    net      : network whose output layer A holds current probabilities.
//    d_labels : device int array length `batch` of true class indices.
//
//  Returns: scalar mean loss = (1/batch) * sum_r -log(probs[r, label[r]]).
//
//  Strategy: cross_entropy_loss computes the per-row loss on the device (one
//  thread per row), then (push 0002) launch_reduce_sum sums those per-row losses
//  ON THE GPU with a parallel tree reduction, and we divide by batch on the host.
//  Only a SINGLE float crosses the PCIe bus (the total), versus copying the whole
//  length-`batch` vector back as the original host reduction did. This is the
//  payoff of the reduction lesson: keep the data on the device and read back one
//  scalar. (The earlier host-sum version is preserved in git history / push 0001.)
float mlp_compute_loss(MLP& net, const int* d_labels) {
    const int batch       = net.batch_size;
    const int num_classes = net.num_classes;
    Layer& out = net.layers[net.num_layers - 1];

    // Per-row loss buffer on the device, length `batch`, shape [batch,1].
    Matrix loss_per_row = matrix_alloc(batch, 1);

    launch_cross_entropy_loss(out.A.data, d_labels, loss_per_row.data,
                              batch, num_classes);

    // Sum the per-row losses on the GPU; only the scalar total comes back.
    float total = launch_reduce_sum(loss_per_row.data, batch);

    matrix_free(loss_per_row);   // release the temporary device buffer
    return total / static_cast<float>(batch);
}

// ----------------------------------------------------------------------------
//  mlp_accuracy
// ----------------------------------------------------------------------------
//  Fraction of the current batch classified correctly.
//
//  Parameters:
//    net      : network whose output layer A holds current probabilities.
//    h_labels : HOST int array length `batch` of true class indices.
//
//  Returns: (# correct) / batch in [0,1].
//
//  We copy the output probabilities [batch, num_classes] to the host and do the
//  argmax there. Argmax is invariant to the softmax normalization, so comparing
//  predicted argmax to the label is a valid correctness test. (A device-side
//  argmax-and-compare reduction is left as an exercise.)
float mlp_accuracy(MLP& net, const int* h_labels) {
    const int batch       = net.batch_size;
    const int num_classes = net.num_classes;
    Layer& out = net.layers[net.num_layers - 1];

    // Pull the whole probability matrix D2H into a flat row-major host buffer.
    std::vector<float> h_probs(static_cast<size_t>(batch) * num_classes);
    matrix_copy_to_host(out.A, h_probs.data());

    int correct = 0;
    for (int r = 0; r < batch; ++r) {
        // argmax over row r: scan the num_classes probabilities for this sample.
        const float* row = &h_probs[static_cast<size_t>(r) * num_classes];
        int   best_c = 0;
        float best_p = row[0];
        for (int c = 1; c < num_classes; ++c) {
            if (row[c] > best_p) { best_p = row[c]; best_c = c; }
        }
        if (best_c == h_labels[r]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(batch);
}

// ----------------------------------------------------------------------------
//  mlp_accuracy_device                                       (added in push 0002)
// ----------------------------------------------------------------------------
//  Accuracy computed entirely on the GPU, as a counterpoint to the host-side
//  argmax loop in mlp_accuracy above. Two kernels do the work:
//    1) predictions_correct : per row, argmax(probs row) == label ? 1.0 : 0.0
//    2) launch_reduce_sum   : parallel tree reduction summing that 0/1 vector
//  Dividing the sum by batch yields the fraction correct. Takes DEVICE labels
//  because the comparison happens inside the kernel. Nothing but the final scalar
//  count is copied back to the host (inside launch_reduce_sum).
float mlp_accuracy_device(MLP& net, const int* d_labels) {
    const int batch       = net.batch_size;
    const int num_classes = net.num_classes;
    Layer& out = net.layers[net.num_layers - 1];

    // 0/1 correctness flag per sample, on the device.
    Matrix correct = matrix_alloc(batch, 1);
    launch_predictions_correct(out.A.data, d_labels, correct.data,
                               batch, num_classes);

    // Sum the flags on the GPU -> number correct -> fraction.
    float n_correct = launch_reduce_sum(correct.data, batch);

    matrix_free(correct);
    return n_correct / static_cast<float>(batch);
}

// ----------------------------------------------------------------------------
//  mlp_evaluate                                              (added in push 0002)
// ----------------------------------------------------------------------------
//  Forward-only "inference mode" pass over a held-out split, used to measure how
//  well the network generalizes to data it was NOT trained on. There is no
//  backward pass and no parameter update here — we only push batches through
//  mlp_forward and read off loss/accuracy. (Conceptually this is where you would
//  also disable train-only behaviors like dropout; this network has none yet.)
//
//  It mirrors the training loop's batching: it processes floor(n_samples/batch)
//  full batches and drops any remainder, so every batch is exactly `batch` rows
//  (which keeps the fixed-size device buffers and the 1/batch loss scaling valid).
//  Device scratch (one [batch, in] matrix + a device label buffer) is allocated
//  and freed inside, so the function is self-contained.
void mlp_evaluate(MLP& net, const float* X, const int* y, int n_samples,
                  float& out_loss, float& out_acc) {
    const int batch     = net.batch_size;
    const int in_f      = net.input_features;
    const int n_batches = n_samples / batch;   // full batches only (drop remainder)

    if (n_batches == 0) {                      // not enough data for one batch
        out_loss = 0.0f;
        out_acc  = 0.0f;
        return;
    }

    // Reusable device scratch for one batch of features + its labels.
    Matrix d_batch  = matrix_alloc(batch, in_f);
    int*   d_labels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_labels, sizeof(int) * batch));

    double loss_sum = 0.0;   // accumulate per-batch means, averaged at the end
    double acc_sum  = 0.0;
    for (int b = 0; b < n_batches; ++b) {
        const int row0 = b * batch;

        // Upload this batch's features and labels (H2D). The features for rows
        // [row0, row0+batch) are contiguous in X (row-major, in_f per row).
        matrix_copy_to_device(d_batch, X + static_cast<size_t>(row0) * in_f);
        CUDA_CHECK(cudaMemcpy(d_labels, y + row0, sizeof(int) * batch,
                              cudaMemcpyHostToDevice));

        // Inference: forward pass only, then read the metrics off the cached
        // output probabilities (both reductions run on the GPU).
        mlp_forward(net, d_batch);
        loss_sum += mlp_compute_loss(net, d_labels);
        acc_sum  += mlp_accuracy_device(net, d_labels);
    }

    out_loss = static_cast<float>(loss_sum / n_batches);
    out_acc  = static_cast<float>(acc_sum  / n_batches);

    matrix_free(d_batch);
    CUDA_CHECK(cudaFree(d_labels));
}

// ----------------------------------------------------------------------------
//  mlp_grad_check  (and its static helper)
// ----------------------------------------------------------------------------
//  Finite-difference gradient check on layer 0's weight matrix. This is the
//  centerpiece "proof" that backprop is implemented correctly: it compares the
//  analytic gradient (from mlp_backward) against a numerical estimate that uses
//  ONLY the forward pass and the definition of a derivative.
//
//  Central difference for a single weight w:
//      dL/dw  ≈  ( L(w + eps) - L(w - eps) ) / (2*eps)
//  Central (two-sided) difference has O(eps^2) truncation error, far better than
//  the one-sided O(eps) form, so it gives a tighter match to the analytic value.

// Static helper: run a forward pass on `batch_input` and return the mean loss
// for the given device labels. Used repeatedly while perturbing weights. Kept
// `static` (file-local) because it is an internal detail, not part of the API.
static float grad_check_loss(MLP& net, const Matrix& batch_input,
                             const int* d_labels) {
    mlp_forward(net, batch_input);          // recompute output probabilities
    return mlp_compute_loss(net, d_labels); // mean CE loss for those probs
}

void mlp_grad_check(MLP& net, const Matrix& batch_input,
                    const int* d_labels, const int* h_labels) {
    (void)h_labels; // accuracy labels are not needed here; kept for API symmetry.

    const float eps = 1e-3f;     // perturbation size; see trade-off note below.
    Layer& L0 = net.layers[0];   // we check the FIRST layer's weights.
    const int in  = L0.in_features;
    const int out = L0.out_features;
    const int n_w = in * out;    // total weights in W of layer 0

    std::printf("[grad_check] layer 0 W is [%d,%d] = %d weights; "
                "checking a few with central differences (eps=%g)\n",
                in, out, n_w, eps);

    // 1) Establish the ANALYTIC gradient at the current weights.
    //    A forward+backward leaves the true dL/dW in L0.dW (already 1/batch
    //    scaled, matching the mean-loss our finite difference measures).
    mlp_forward(net, batch_input);
    mlp_backward(net, batch_input, d_labels);

    // Snapshot the current weights and the analytic gradient onto the host so we
    // can perturb individual entries and read back the corresponding analytic dW.
    std::vector<float> h_W(n_w);
    std::vector<float> h_dW(n_w);
    matrix_copy_to_host(L0.W,  h_W.data());
    matrix_copy_to_host(L0.dW, h_dW.data());

    // 2) Probe a handful of weights spread across the matrix (checking all of
    //    them would be slow — each probe costs two full forward passes).
    const int num_probes = (n_w < 5) ? n_w : 5;
    const int stride     = (n_w + num_probes - 1) / num_probes; // spread indices

    double max_rel_err = 0.0;
    for (int p = 0; p < num_probes; ++p) {
        const int idx = (p * stride) % n_w;   // weight index into the flat W
        const float original = h_W[idx];

        // --- L(w + eps) ------------------------------------------------------
        std::vector<float> probe = h_W;       // copy of all weights
        probe[idx] = original + eps;
        matrix_copy_to_device(L0.W, probe.data());
        const float loss_plus = grad_check_loss(net, batch_input, d_labels);

        // --- L(w - eps) ------------------------------------------------------
        probe[idx] = original - eps;
        matrix_copy_to_device(L0.W, probe.data());
        const float loss_minus = grad_check_loss(net, batch_input, d_labels);

        // Restore the original weight so the check is non-destructive and later
        // probes start from the true parameters.
        probe[idx] = original;
        matrix_copy_to_device(L0.W, probe.data());

        // Numerical (central-difference) gradient for this single weight.
        const float numeric  = (loss_plus - loss_minus) / (2.0f * eps);
        const float analytic = h_dW[idx];

        // Relative error: |a-n| / max(1, |a|, |n|). The max(...,1) denominator
        // avoids a blow-up when both gradients are ~0 (then absolute error is
        // already tiny and "relative" error is not meaningful).
        const float denom = std::fmax(1.0f,
                              std::fmax(std::fabs(analytic), std::fabs(numeric)));
        const float rel_err = std::fabs(analytic - numeric) / denom;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        std::printf("  W[%4d]  analytic=% .6e  numeric=% .6e  rel_err=% .3e\n",
                    idx, analytic, numeric, rel_err);
    }

    // A correct backward pass typically yields rel_err well under 1e-2 with this
    // eps (single precision + eps=1e-3 limits us; tightening eps too far makes
    // round-off dominate — that is the classic finite-difference trade-off).
    std::printf("[grad_check] max relative error = %.3e  -> %s\n",
                max_rel_err, (max_rel_err < 1e-2) ? "PASS" : "SUSPECT");

    // Leave the network in a consistent state: recompute the forward caches at
    // the (restored) original weights so callers can use net.layers[*].A safely.
    mlp_forward(net, batch_input);
}
