// ============================================================================
//  include/mlp.cuh
//  ----------------------------------------------------------------------------
//  Role: declares the network-level data structures (`Layer`, `MLP`) and the
//  full training API (create / forward / backward / SGD step / loss / accuracy /
//  gradient check). This is the "model" layer of the repo: it owns the
//  parameters and per-layer caches as `Matrix` objects (device memory) and
//  orchestrates the kernels declared in kernels.cuh. mlp.cu implements every
//  function here; main.cu drives the whole pipeline through this API. No CUDA
//  kernels are *defined* in this file -- it is pure declarations + comments.
// ============================================================================
#pragma once

// We need the Matrix struct (a {float* device data, int rows, int cols} bundle,
// row-major) because every parameter, cache, and gradient below is a Matrix.
// We do NOT include kernels.cuh here: nothing in this header calls a kernel; the
// kernel orchestration happens inside mlp.cu, which includes kernels.cuh itself.
#include "matrix.cuh"

// ----------------------------------------------------------------------------
//  enum class Activation                                     (added in push 0002)
//  ----------------------------------------------------------------------------
//  Selects which nonlinearity a HIDDEN layer applies after its affine step. The
//  OUTPUT layer ignores this and always uses softmax (gated by Layer::is_output).
//  Each value maps to a forward kernel and a backward kernel in kernels.cu:
//    ReLU      -> relu_forward       / relu_backward        (gate on pre-act Z)
//    LeakyReLU -> leaky_relu_forward / leaky_relu_backward  (gate on pre-act Z)
//    Tanh      -> tanh_forward       / tanh_backward         (uses post-act A!)
//  The Tanh case is the didactic one: its derivative 1 - a^2 is written in terms
//  of the OUTPUT a = tanh(z), so its backward reads A, whereas the (Leaky)ReLU
//  backward reads Z. mlp_backward dispatches to the right tensor per activation.
//  `enum class` (scoped enum) keeps these names from leaking into the global
//  namespace — you write Activation::ReLU, and they don't implicitly convert to
//  int, which prevents a whole category of mix-up bugs.
// ----------------------------------------------------------------------------
enum class Activation { ReLU, LeakyReLU, Tanh };

// ----------------------------------------------------------------------------
//  struct Layer
//  ----------------------------------------------------------------------------
//  One fully-connected (a.k.a. "linear" / "dense") layer plus its activation.
//
//  Forward for a single layer (see §2 of the build spec, row-major throughout):
//      Z = A_prev * W + b           // matmul then bias broadcast over rows
//      A = is_output ? softmax(Z)   // output layer: A holds class probabilities
//                    : relu(Z)      // hidden layer:  A = max(0, Z)
//
//  Shape conventions (row-major flat arrays in GPU global memory):
//    - in_features  = number of inputs  to this layer (call it `in`)
//    - out_features = number of outputs of this layer (call it `out`)
//    - `batch`      = rows in the current minibatch (= MLP::batch_size)
//
//  Every Matrix below stores its own rows/cols, but the shape comments state the
//  *intended* logical shape so a reader can reason about the index math without
//  cross-referencing the allocator. "[r, c]" means r rows, c cols, element at
//  flat index `r_idx * c + c_idx`.
struct Layer {
    int in_features;    // `in`  : length of each input  row  (units: features)
    int out_features;   // `out` : length of each output row  (units: neurons)
    bool is_output;     // true  => softmax activation (this is the final layer);
                        // false => use `activation` below (a hidden layer).
    Activation activation; // (push 0002) which nonlinearity a HIDDEN layer uses.
                        // Ignored when is_output is true (output is softmax).

    // ---- parameters (the learnable weights; persist across batches) ---------
    Matrix W;           // [in, out]  weights. Row-major: W[i*out + o] is the
                        //            weight from input neuron i to output o.
                        //            Forward uses it untransposed: Z = A_prev*W.
    Matrix b;           // [1, out]   bias vector, one scalar per output neuron.
                        //            Broadcast (added) to every row of Z.

    // ---- forward caches (recomputed every forward pass; sized [batch, out]) -
    Matrix Z;           // [batch, out]  pre-activation  = A_prev*W + b.
                        //               Cached because ReLU's backward needs the
                        //               sign of Z (relu'(Z)).
    Matrix A;           // [batch, out]  post-activation = relu(Z) or softmax(Z).
                        //               For the output layer these are the
                        //               per-row class probabilities. A is also
                        //               the `A_prev` feeding the NEXT layer.

    // ---- gradients (filled by mlp_backward; same shapes as their primals) ---
    Matrix dW;          // [in, out]    dL/dW. Same shape as W so sgd_update can
                        //              walk them element-for-element.
    Matrix db;          // [1, out]     dL/db = column-sum of dZ over the batch.
    Matrix dZ;          // [batch, out] dL/dZ : gradient wrt this layer's
                        //              PRE-activation. For the output layer this
                        //              is (probs - onehot)/batch; for hidden
                        //              layers it is dA elementwise-times relu'(Z).
    Matrix dA;          // [batch, out] dL/dA : gradient wrt this layer's OUTPUT
                        //              activation. Produced by the next layer's
                        //              backprop (dA_prev = dZ_next * W_next^T),
                        //              then turned into dZ via the activation
                        //              derivative. (Output layer skips dA: its dZ
                        //              comes straight from cross_entropy_grad.)

    // ---- dropout (push 0003; hidden layers only) ----------------------------
    float  dropout_p;   // drop probability in [0,1). 0 disables dropout for this
                        //   layer. The output layer always has 0.
    Matrix dropout_mask;// [batch, out] cached SCALED mask from dropout_forward
                        //   (1/(1-p) for kept units, 0 for dropped). Reused by
                        //   dropout_backward so gradients flow only through kept
                        //   units. Allocated like Z/A even when p==0 (then unused).
    Matrix A_out;       // [batch, out] the layer's OUTPUT fed to the next layer.
                        //   We keep `A` as the PURE activation act(Z) and write the
                        //   post-dropout result here, so Tanh's backward (which
                        //   needs a = tanh(Z)) can still read the un-dropped `A`.
                        //   When no dropout is applied this pass, the output is just
                        //   `A` and A_out is unused (see `dropped`).
    bool   dropped;     // transient: did THIS forward pass apply dropout to this
                        //   layer? Set in mlp_forward; read in mlp_backward to pick
                        //   the right output buffer (A_out vs A) and to decide
                        //   whether to run dropout_backward. (Not persistent state —
                        //   just bookkeeping between a paired forward/backward.)
};

// ----------------------------------------------------------------------------
//  struct MLP
//  ----------------------------------------------------------------------------
//  The whole network: a flat array of Layers plus the global dimensions shared
//  by all of them. A network built from layer_sizes = {2, 64, 32, 3} has
//  num_layers = 3 layers (2->64 ReLU, 64->32 ReLU, 32->3 softmax). All caches
//  are pre-sized for `batch_size` rows so no per-batch (re)allocation happens in
//  the training loop -- we reuse the same device buffers every iteration.
struct MLP {
    Layer* layers;       // host-side array of Layer, length num_layers. (The
                         // Layer structs live on the host; the Matrix `data`
                         // pointers inside them point into GPU global memory.)
    int num_layers;      // number of layers = (num_sizes - 1).
    int batch_size;      // rows per minibatch; fixes the cache shapes [batch,*].
    int input_features;  // = layer_sizes[0]      : width of the input batch.
    int num_classes;     // = layer_sizes[last]   : width of the softmax output.

    // ---- training vs inference mode + RNG (push 0003) -----------------------
    bool training;       // true => dropout is active in mlp_forward; false =>
                         //   inference (dropout becomes the identity). Toggle via
                         //   mlp_set_training. Inverted dropout means inference
                         //   needs no rescaling, so flipping this flag is enough.
    unsigned long long rng_state; // counter feeding the dropout RNG. The training
                         //   loop advances it once per step so each step draws a
                         //   fresh mask; mlp_grad_check holds it FIXED so the mask
                         //   is identical across its perturbed forward passes
                         //   (that determinism is what makes grad-checking a
                         //   stochastic dropout layer possible).
};

// ----------------------------------------------------------------------------
//  mlp_create
//  ----------------------------------------------------------------------------
//  Builds an MLP from `layer_sizes` (length `num_sizes`, must be >= 2). Creates
//  num_sizes-1 layers, marking ONLY the last one as the softmax output layer;
//  all earlier layers use ReLU. For each layer it cudaMalloc's W, b, and every
//  cache/gradient Matrix at the right shape (caches sized for `batch_size`).
//
//  Weight init: He / Kaiming normal -- each W entry ~ N(0, sqrt(2/in)). This
//  variance keeps the forward activations from shrinking or exploding through
//  ReLU layers (ReLU zeros half its inputs, so we scale up by 2). Biases start
//  at 0. All randomness flows from a single std::mt19937_64(seed) so two runs
//  with the same seed produce byte-identical weights (determinism, §0 rule 7).
//
//  Params:
//    layer_sizes : host int array, e.g. {2,64,32,3} (units: neuron counts).
//    num_sizes   : its length (>= 2).
//    batch_size  : rows per minibatch (fixes every cache's row count).
//    seed        : RNG seed for reproducible weight init.
//    hidden_act  : (push 0002) nonlinearity for ALL hidden layers; defaults to
//                  ReLU so older callers compile unchanged. The output layer is
//                  always softmax regardless. NOTE: He init (sqrt(2/in)) is tuned
//                  for ReLU-family activations; for Tanh, Xavier/Glorot init
//                  would be more principled, but He still trains fine here and we
//                  keep one init path for simplicity (noted as an exercise).
//    dropout_p   : (push 0003) drop probability applied to every HIDDEN layer's
//                  activations during training (0 disables it; default keeps old
//                  behavior). The output layer never uses dropout. The new MLP
//                  starts in TRAINING mode (net.training = true) with rng_state
//                  seeded from `seed`.
//  Returns: a fully-allocated MLP (caller must later call mlp_free).
MLP  mlp_create(const int* layer_sizes, int num_sizes, int batch_size,
                unsigned long long seed,
                Activation hidden_act = Activation::ReLU,
                float dropout_p = 0.0f);

// Frees every device Matrix in every layer, frees the host `layers` array, and
// zeroes the MLP's fields so the struct can't be accidentally reused. Mirrors
// matrix_free's "free + null out" discipline at the network level.
void mlp_free(MLP& net);

// ----------------------------------------------------------------------------
//  mlp_set_training                                          (added in push 0003)
//  ----------------------------------------------------------------------------
//  Switch the network between TRAINING mode (dropout active in mlp_forward) and
//  INFERENCE mode (dropout becomes the identity). Because we use *inverted*
//  dropout (survivors are pre-scaled by 1/(1-p) during training), inference needs
//  no rescaling — flipping this one flag is the entire train/eval switch. The
//  training loop sets true; mlp_evaluate sets false for the duration of the eval.
void mlp_set_training(MLP& net, bool training);

// ----------------------------------------------------------------------------
//  mlp_forward
//  ----------------------------------------------------------------------------
//  Runs the forward pass over a device batch `batch_input` of shape
//  [batch, input_features]. After it returns, layers[num_layers-1].A holds the
//  class probabilities, shape [batch, num_classes]. Each layer's Z and A caches
//  are filled along the way (needed by backward).
//
//  Algorithm (each launch_* is a kernel wrapper from kernels.cuh):
//    prev = batch_input
//    for l in 0 .. num_layers-1:
//        L = layers[l]
//        // Z = prev * W   -> C[batch,out] = A[batch,in] * B[in,out], no transpose
//        launch_gemm(prev.data, L.W.data, L.Z.data, batch, L.out, L.in, false, false)
//        launch_add_bias(L.Z.data, L.b.data, batch, L.out)          // Z[r,c] += b[c]
//        if L.is_output: launch_softmax_rows(L.Z.data, L.A.data, batch, L.out)
//        else:           launch_relu_forward(L.Z.data, L.A.data, batch*L.out)
//        prev = L.A        // this layer's output becomes the next layer's input
void mlp_forward(MLP& net, const Matrix& batch_input);

// ----------------------------------------------------------------------------
//  mlp_backward
//  ----------------------------------------------------------------------------
//  Backpropagation. Precondition: mlp_forward was already run on this SAME
//  `batch_input` (the Z/A caches must match). `d_labels` is a DEVICE int array
//  of length `batch` holding each row's true class index in [0, num_classes).
//  Fills dW and db for every layer (and the intermediate dZ/dA caches).
//
//  Algorithm (derives the dZ = (probs - onehot)/batch identity; the /batch folds
//  in the batch-mean of the loss so SGD is just `p -= lr*grad`):
//    out = layers[num_layers-1]
//    launch_cross_entropy_grad(out.A.data, d_labels, out.dZ.data, batch, num_classes)
//    for l from num_layers-1 down to 0:
//        L = layers[l]
//        prev_act = (l==0) ? batch_input : layers[l-1].A
//        // dW = prev_act^T * dZ   -> [in,out] = [in,batch]*[batch,out] (transA)
//        launch_gemm(prev_act.data, L.dZ.data, L.dW.data, L.in, L.out, batch, true, false)
//        launch_bias_grad(L.dZ.data, L.db.data, batch, L.out)     // db = colsum(dZ)
//        if l > 0:
//            prevL = layers[l-1]
//            // dA_prev = dZ * W^T -> [batch,in] = [batch,out]*[out,in] (transB)
//            launch_gemm(L.dZ.data, L.W.data, prevL.dA.data, batch, L.in, L.out, false, true)
//            // dZ_prev = dA_prev (elementwise *) relu'(Z_prev)
//            launch_relu_backward(prevL.dA.data, prevL.Z.data, prevL.dZ.data,
//                                 batch*prevL.out_features)
//  (Note L.in == prevL.out_features, which is why the shapes line up.)
void mlp_backward(MLP& net, const Matrix& batch_input, const int* d_labels);

// ----------------------------------------------------------------------------
//  mlp_sgd_step
//  ----------------------------------------------------------------------------
//  Vanilla SGD: for every parameter buffer, param -= lr * grad. Applied to each
//  layer's W (using dW) and b (using db) via launch_sgd_update. Because the
//  batch-mean 1/batch was already folded into the gradients in mlp_backward,
//  there is no extra averaging here -- `lr` is the only scale.
//    lr : learning rate (units: step size; e.g. 0.1).
void mlp_sgd_step(MLP& net, float lr);

// ----------------------------------------------------------------------------
//  mlp_compute_loss
//  ----------------------------------------------------------------------------
//  Mean softmax cross-entropy over the current batch, computed from the cached
//  output probabilities (layers[last].A) and the given DEVICE labels `d_labels`
//  (length batch). Runs cross_entropy_loss (per-row -log p[label]), then (push
//  0002) sums the per-row losses on the GPU via launch_reduce_sum and divides by
//  batch — only the scalar total is copied to the host. Returns the mean loss.
//  (Reporting only -- not used to drive gradients.)
float mlp_compute_loss(MLP& net, const int* d_labels);

// ----------------------------------------------------------------------------
//  mlp_accuracy
//  ----------------------------------------------------------------------------
//  Fraction of the current batch classified correctly. `h_labels` is a HOST int
//  array of length batch (true classes). Copies the cached output probs
//  [batch, num_classes] to host, takes the argmax of each row, compares to the
//  label, and returns correct/batch in [0,1]. (Argmax on host is fine: the
//  output is tiny and this is reporting, not the training hot path.)
float mlp_accuracy(MLP& net, const int* h_labels);

// ----------------------------------------------------------------------------
//  mlp_grad_check
//  ----------------------------------------------------------------------------
//  Finite-difference sanity check on layer 0's weights, proving the analytic
//  backward pass is correct without any reference framework. For a handful of
//  weights w it compares:
//      analytic  : dW from mlp_backward
//      numerical : (loss(w+eps) - loss(w-eps)) / (2*eps)   // central difference
//  and prints the relative error |analytic-numerical| / max(|.|, tiny). Tiny
//  relative errors (~1e-4 or smaller) mean the gradients agree. Uses eps ~ 1e-3.
//
//  Needs BOTH label forms: `d_labels` (device, for forward/backward/loss kernels)
//  and `h_labels` (host, only to match mlp_accuracy's signature style / future
//  use). `batch_input` is the device batch to evaluate on.
void mlp_grad_check(MLP& net, const Matrix& batch_input,
                    const int* d_labels, const int* h_labels);

// ----------------------------------------------------------------------------
//  mlp_accuracy_device                                       (added in push 0002)
//  ----------------------------------------------------------------------------
//  Same result as mlp_accuracy, but computed ENTIRELY on the GPU instead of via
//  a host argmax loop. It runs `predictions_correct` (per-row argmax == label ->
//  1/0) then `launch_reduce_sum` (parallel tree reduction) over the batch, and
//  divides by batch. Takes DEVICE labels (`d_labels`, length batch) since the
//  comparison happens in the kernel. Kept alongside the host mlp_accuracy on
//  purpose, to contrast a host-side reduction with a device-side one.
float mlp_accuracy_device(MLP& net, const int* d_labels);

// ----------------------------------------------------------------------------
//  mlp_evaluate                                              (added in push 0002)
//  ----------------------------------------------------------------------------
//  Forward-ONLY evaluation over a held-out split (no backprop, no weight update)
//  — i.e. "inference mode". Used to measure generalization on validation data.
//  It batches the host arrays X/y into full batch_size chunks (dropping any
//  remainder, like training), runs mlp_forward on each, and accumulates the mean
//  cross-entropy loss and accuracy across batches; results are returned via the
//  out-params. It allocates its own small device scratch (a [batch, in] matrix
//  and a device label buffer) and frees it before returning, so it is fully
//  self-contained and safe to call any time after mlp_create.
//
//  Params:
//    net       : trained (or training) network; its caches are overwritten.
//    X         : HOST features, [n_samples, input_features] row-major.
//    y         : HOST labels,   [n_samples], class indices in [0, num_classes).
//    n_samples : number of rows in X/y.
//    out_loss  : (out) mean CE loss over the evaluated full batches.
//    out_acc   : (out) mean accuracy over the evaluated full batches, in [0,1].
void mlp_evaluate(MLP& net, const float* X, const int* y, int n_samples,
                  float& out_loss, float& out_acc);
