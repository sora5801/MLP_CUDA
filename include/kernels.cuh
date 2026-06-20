// ============================================================================
// include/kernels.cuh
// ----------------------------------------------------------------------------
// ROLE IN THE PROJECT
// This header is the single place where EVERY CUDA computation in the MLP is
// declared. It exposes two layers of API:
//
//   1) The raw `__global__` kernels (the code that actually runs on the GPU,
//      one instance per thread). These are declared here so they can be
//      defined in src/kernels.cu and, in principle, inspected/tested directly.
//
//   2) The `__host__` `launch_*` wrappers. These are the ONLY entry points the
//      rest of the program (mlp.cu, main.cu) is allowed to call. Each wrapper:
//        - chooses a sensible block size and computes the grid size with
//          ceil_div(...) so that every output element is covered,
//        - launches the matching kernel with the `<<<grid, block>>>` syntax,
//        - calls CUDA_CHECK_LAST() to surface any launch/runtime error.
//      Hiding the launch configuration here keeps the math code in mlp.cu
//      readable: it reads like ordinary linear algebra, not CUDA plumbing.
//
// Everything operates on `float*` pointers into GPU *global memory*, stored
// ROW-MAJOR (see the math conventions in the build spec). No host pointers are
// dereferenced inside kernels.
// ============================================================================

#pragma once

// common.cuh brings in <cuda_runtime.h> (for __global__, dim3, the CUDA runtime
// API) plus our error-checking macros (CUDA_CHECK / CUDA_CHECK_LAST), the
// kBlockSize / kTileDim constants, and the ceil_div() helper. We include it so
// this header is self-contained: a translation unit can include just kernels.cuh
// and have all the types and helpers the declarations below rely on.
#include "common.cuh"

// ----------------------------------------------------------------------------
// GEMM TRANSPOSE SEMANTICS  (READ THIS BEFORE THE DECLARATIONS)
// ----------------------------------------------------------------------------
// GEMM = GEneral Matrix Multiply. A single GEMM kernel powers BOTH the forward
// pass and the backward pass of the network, because every linear-algebra step
// in backprop is "multiply two matrices, maybe transposing one of them first".
// Rather than write four separate kernels (no-T/no-T, T/no-T, ...), we pass two
// boolean flags `transA` and `transB` that select, per operand, whether we read
// it as stored or as its logical transpose.
//
// CONTRACT for launch_gemm(A, B, C, M, N, K, transA, transB):
//
//   C[M,N] = opA(A) * opB(B)
//
//   where opA(A) is the [M,K] matrix and opB(B) is the [K,N] matrix, so the
//   product is well-defined and C has shape [M,N]. The element formula is the
//   ordinary dot product over the shared dimension K:
//
//       C[m,n] = sum_{k=0..K-1}  A_log[m,k] * B_log[k,n]
//
//   "_log" means the LOGICAL element (the value at logical row/col). How that
//   logical element maps onto the PHYSICAL row-major storage depends on the
//   transpose flag for that operand:
//
//       A_log[m,k] = (!transA) ? A[m*K + k]   // A stored row-major as [M,K]
//                              :  A[k*M + m]   // A stored row-major as [K,M]
//
//       B_log[k,n] = (!transB) ? B[k*N + n]   // B stored row-major as [K,N]
//                              :  B[n*K + k]   // B stored row-major as [N,K]
//
//   IMPORTANT: M, N, K always describe the LOGICAL (post-op) shapes. The flags
//   only change how we index into the *given* arrays; they do NOT change M/N/K.
//   So when transA is true the array `A` is physically [K,M] in memory, but we
//   still treat its logical shape as [M,K].
//
// Output mapping: one thread computes exactly one element of C.
//   m (logical row of C) = blockIdx.y*blockDim.y + threadIdx.y
//   n (logical col of C) = blockIdx.x*blockDim.x + threadIdx.x
//   Guard with (m < M && n < N) because the grid is rounded up and the last
//   blocks contain threads that fall outside the matrix.
//   Each thread accumulates into a local `float acc` and writes C[m*N + n] = acc
//   (C is always stored plainly as [M,N], never transposed).
//
// WHERE THE TRANSPOSES COME FROM IN THIS MLP (so the flags aren't abstract):
//   - Forward linear:  Z[batch,out] = A_prev[batch,in] * W[in,out]
//                      launch_gemm(A_prev, W, Z, batch, out, in, false, false)
//   - Weight grad:     dW[in,out]   = A_prev^T[in,batch] * dZ[batch,out]
//                      launch_gemm(A_prev, dZ, dW, in, out, batch, true,  false)
//                      (A_prev is stored [batch,in]; transA reads it as [in,batch])
//   - Input grad:      dA_prev[batch,in] = dZ[batch,out] * W^T[out,in]
//                      launch_gemm(dZ, W, dA_prev, batch, in, out, false, true)
//                      (W is stored [in,out]; transB reads it as [out,in])
// ----------------------------------------------------------------------------


// ============================================================================
// 1) gemm_naive — general matrix multiply with optional transposes.
// ----------------------------------------------------------------------------
// The correctness workhorse: one thread computes one output element with a
// plain loop over K. No shared memory, no tiling — slow but obviously correct,
// and it is the version actually used throughout the MLP forward/backward.
//
// Parameters (all device pointers are in GPU global memory, row-major):
//   A : input operand A. Physically [M,K] if !transA, else [K,M].
//   B : input operand B. Physically [K,N] if !transB, else [N,K].
//   C : output, physically [M,N]; length M*N floats; overwritten (not added to).
//   M : logical rows of opA(A) and of C.
//   N : logical cols of opB(B) and of C.
//   K : shared/contracted dimension (cols of opA(A), rows of opB(B)).
//   transA / transB : interpret the corresponding operand as its transpose
//                     when indexing (see the semantics block above).
//
// Launch config (set by launch_gemm): 2-D block of kTileDim x kTileDim threads,
// 2-D grid covering the [M,N] output (x = columns = N, y = rows = M).
// ============================================================================
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K, bool transA, bool transB);

// Host wrapper: picks the 2-D launch configuration, launches gemm_naive, and
// checks for errors. The ONLY GEMM entry point the MLP code calls.
void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, bool transA, bool transB);


// ============================================================================
// 2) gemm_tiled — shared-memory tiled GEMM, NO transpose.
// ----------------------------------------------------------------------------
// Optimization lesson and the subject of main.cu's microbenchmark. Computes the
// straightforward product C[M,N] = A[M,K] * B[K,N] (no transpose flags). Each
// kTileDim x kTileDim block cooperatively loads tiles of A and B into shared
// memory, then reuses them across the block — cutting global-memory traffic by
// roughly a factor of kTileDim and exploiting coalesced loads. This is purely
// for teaching; the MLP itself uses gemm_naive for its transpose flexibility.
//
// Parameters:
//   A : [M,K] row-major, GPU global memory.
//   B : [K,N] row-major, GPU global memory.
//   C : [M,N] row-major output (overwritten), length M*N floats.
//   M, N, K : standard GEMM dimensions (no transpose, so storage == logical).
//
// Launch config: block = kTileDim x kTileDim (= 256 threads), grid covers [M,N].
// The kernel uses __shared__ tiles of size kTileDim x kTileDim and __syncthreads
// between the load phase and the compute phase of every tile step.
// ============================================================================
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K);

// Host wrapper for the tiled GEMM (same grid/block scheme as launch_gemm).
void launch_gemm_tiled(const float* A, const float* B, float* C,
                       int M, int N, int K);


// ============================================================================
// 3) add_bias — add a per-column bias, broadcast across all rows.
// ----------------------------------------------------------------------------
// In-place update: Z[r,c] += bias[c]. After the forward GEMM produces the raw
// product A_prev*W, this adds the layer bias. The bias is shared by every row
// (every sample in the batch), hence the "broadcast over rows" — bias has only
// `cols` entries but is applied to all `rows*cols` of Z.
//
// Parameters:
//   Z    : [rows, cols] row-major, in/out (modified in place). GPU global mem.
//   bias : length `cols`, the per-output-feature bias b[c]. GPU global mem.
//   rows : number of rows of Z (the batch size).
//   cols : number of columns of Z (the layer's out_features).
//
// Launch config: a 1-D grid over all rows*cols elements (kBlockSize threads per
// block). Thread i handles element i; its column is (i % cols) -> bias index.
// ============================================================================
__global__ void add_bias(float* Z, const float* bias, int rows, int cols);

// Host wrapper: 1-D launch over rows*cols elements, then CUDA_CHECK_LAST().
void launch_add_bias(float* Z, const float* bias, int rows, int cols);


// ============================================================================
// 4) relu_forward — elementwise rectified linear unit, out[i] = max(0, in[i]).
// ----------------------------------------------------------------------------
// The hidden-layer activation. Embarrassingly parallel: every element is
// independent, so it is the canonical 1-D element-wise kernel.
//
// Parameters:
//   in  : input array, length n (a flattened [batch, out] activation tensor).
//   out : output array, length n. May alias `in` for a true in-place ReLU,
//         but the MLP keeps Z (pre-activation) and A (post-activation) separate
//         so that relu_backward can read the pre-activation later.
//   n   : total number of elements (batch * out_features).
//
// Launch config: 1-D grid, kBlockSize threads/block, ceil_div(n, kBlockSize)
// blocks. Thread global index i = blockIdx.x*blockDim.x + threadIdx.x; guard i<n.
// ============================================================================
__global__ void relu_forward(const float* in, float* out, int n);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_relu_forward(const float* in, float* out, int n);


// ============================================================================
// 5) relu_backward — gradient through a ReLU.
// ----------------------------------------------------------------------------
// ReLU's derivative is 1 where its input was positive and 0 otherwise, so the
// gradient is simply gated by the sign of the original pre-activation:
//   grad_in[i] = (pre_act[i] > 0) ? grad_out[i] : 0
// We use the PRE-activation Z (not the post-activation A) as the gate; for ReLU
// the two give the same mask, but Z is what the math derivation references.
//
// Parameters:
//   grad_out : upstream gradient dA flowing into this activation, length n.
//   pre_act  : the layer's pre-activation Z captured during forward, length n.
//   grad_in  : output gradient dZ (overwritten), length n. May alias grad_out.
//   n        : total elements (batch * out_features).
//
// Launch config: identical 1-D element-wise scheme as relu_forward.
// ============================================================================
__global__ void relu_backward(const float* grad_out, const float* pre_act,
                              float* grad_in, int n);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_relu_backward(const float* grad_out, const float* pre_act,
                          float* grad_in, int n);


// ============================================================================
// 6) softmax_rows — numerically-stable row-wise softmax.
// ----------------------------------------------------------------------------
// Output-layer activation: turns each row of logits into a probability
// distribution over classes. To avoid exp() overflow on large logits we use the
// standard max-subtraction trick (softmax is invariant to adding a constant to
// every logit in a row):
//   m_r        = max_c logits[r,c]
//   probs[r,c] = exp(logits[r,c] - m_r) / sum_c' exp(logits[r,c'] - m_r)
//
// One thread handles an ENTIRE row (the number of classes `cols` is small, so a
// per-row serial loop is fine and avoids cross-thread reductions). This means
// the parallelism is over rows, not elements.
//
// Parameters:
//   logits : [rows, cols] row-major pre-activations Z. GPU global memory.
//   probs  : [rows, cols] row-major output probabilities (overwritten).
//   rows   : number of rows (batch size).
//   cols   : number of columns (num_classes).
//
// Launch config: 1-D grid over rows, kBlockSize threads/block,
// ceil_div(rows, kBlockSize) blocks. Thread r handles row r; guard r<rows.
// ============================================================================
__global__ void softmax_rows(const float* logits, float* probs, int rows, int cols);

// Host wrapper: 1-D launch over `rows`, then CUDA_CHECK_LAST().
void launch_softmax_rows(const float* logits, float* probs, int rows, int cols);


// ============================================================================
// 7) cross_entropy_grad — gradient of the mean softmax cross-entropy wrt logits.
// ----------------------------------------------------------------------------
// For softmax + cross-entropy the gradient wrt the logits collapses to the
// famously simple form (derived in docs/math_derivation.md):
//   grad[r,c] = (probs[r,c] - onehot[r,c]) / rows
// where onehot[r,c] = 1 if labels[r]==c else 0. Dividing by `rows` folds in the
// 1/batch factor of the MEAN loss right here, so every downstream gradient
// (dW, db, dA) already carries the 1/batch scale and the SGD update is just
// param -= lr*grad (no extra averaging anywhere else).
//
// Parameters:
//   probs  : [rows, cols] softmax probabilities from softmax_rows. GPU mem.
//   labels : length `rows`, true class index in [0,cols) per sample. DEVICE
//            int array.
//   grad   : [rows, cols] output gradient dZ for the output layer (overwritten).
//   rows   : batch size.
//   cols   : num_classes.
//
// Launch config: 1-D grid over all rows*cols elements (one thread per element).
// Thread i -> row r=i/cols, col c=i%cols; subtract 1 only when c==labels[r].
// ============================================================================
__global__ void cross_entropy_grad(const float* probs, const int* labels,
                                   float* grad, int rows, int cols);

// Host wrapper: 1-D launch over rows*cols elements, then CUDA_CHECK_LAST().
void launch_cross_entropy_grad(const float* probs, const int* labels,
                               float* grad, int rows, int cols);


// ============================================================================
// 8) cross_entropy_loss — per-row cross-entropy for reporting / grad-check.
// ----------------------------------------------------------------------------
// Computes the (negative log-likelihood) loss of each sample separately:
//   loss[r] = -log( max( probs[r, labels[r]], 1e-12 ) )
// The clamp to 1e-12 guards against log(0) when a probability underflows. The
// host sums these per-row losses and divides by `rows` to get the scalar mean
// CE loss (see mlp_compute_loss). Keeping it per-row keeps the kernel trivially
// parallel and lets the finite-difference grad-check reuse the same path.
//
// Parameters:
//   probs        : [rows, cols] softmax probabilities. GPU global memory.
//   labels       : length `rows`, true class indices. DEVICE int array.
//   loss_per_row : length `rows` output, one loss value per sample (overwritten).
//   rows         : batch size.
//   cols         : num_classes (used to index the correct probability).
//
// Launch config: 1-D grid over `rows`, one thread per row; guard r<rows.
// ============================================================================
__global__ void cross_entropy_loss(const float* probs, const int* labels,
                                   float* loss_per_row, int rows, int cols);

// Host wrapper: 1-D launch over `rows`, then CUDA_CHECK_LAST().
void launch_cross_entropy_loss(const float* probs, const int* labels,
                               float* loss_per_row, int rows, int cols);


// ============================================================================
// 9) bias_grad — bias gradient as a column-sum over the batch.
// ----------------------------------------------------------------------------
// Because the bias is added identically to every row in the forward pass, its
// gradient is the sum of the pre-activation gradient down each column:
//   db[c] = sum_{r=0..rows-1} dZ[r,c]
// (The 1/batch scale is already baked into dZ via cross_entropy_grad, so we do
// a plain sum here.)
//
// One thread per output column c (cols is small = out_features); each thread
// walks all `rows` of its column and accumulates. This avoids any atomics or
// cross-thread reduction at the cost of a serial loop over the batch.
//
// Parameters:
//   dZ   : [rows, cols] gradient wrt pre-activation. GPU global memory.
//   db   : length `cols` output bias gradient (overwritten). GPU global memory.
//   rows : batch size (the dimension summed over).
//   cols : out_features (the number of bias entries / threads needed).
//
// Launch config: 1-D grid over `cols`, kBlockSize threads/block. Thread c sums
// dZ[r*cols + c] for r in [0,rows); guard c<cols.
// ============================================================================
__global__ void bias_grad(const float* dZ, float* db, int rows, int cols);

// Host wrapper: 1-D launch over `cols`, then CUDA_CHECK_LAST().
void launch_bias_grad(const float* dZ, float* db, int rows, int cols);


// ============================================================================
// 10) sgd_update — in-place stochastic gradient descent parameter step.
// ----------------------------------------------------------------------------
// The simplest possible optimizer:
//   param[i] -= lr * grad[i]
// Applied to every weight and bias element. Because the batch-mean (1/batch)
// factor was already folded into the gradients upstream, `lr` here is the raw
// learning rate with no further scaling.
//
// Parameters:
//   param : parameter array, length n, modified in place. GPU global memory.
//   grad  : gradient array, length n (matching `param`). GPU global memory.
//   lr    : learning-rate scalar (passed by value into the kernel).
//   n     : number of elements (e.g. in*out for a weight matrix, out for a bias).
//
// Launch config: 1-D element-wise grid over n, kBlockSize threads/block.
// ============================================================================
__global__ void sgd_update(float* param, const float* grad, float lr, int n);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_sgd_update(float* param, const float* grad, float lr, int n);


// ============================================================================
// ============================================================================
//  ADDED IN PUSH 0002 — extra activations, a parallel reduction, and a
//  device-side "predictions correct" helper. See docs/changelog/0002-*.md.
// ============================================================================
// ============================================================================


// ============================================================================
// 11) leaky_relu_forward — out[i] = (in[i] > 0) ? in[i] : alpha*in[i].
// ----------------------------------------------------------------------------
// A variant of ReLU that, instead of hard-zeroing negative inputs, lets them
// through with a small slope `alpha` (e.g. 0.01). WHY: a plain ReLU unit whose
// input is always negative outputs 0 forever and its weights get a 0 gradient
// (relu'(z)=0) — a "dead" neuron that can never recover. LeakyReLU keeps a tiny
// nonzero gradient `alpha` on the negative side so such a unit can still learn.
//
// Parameters:
//   in    : input array, length n (flattened [batch, out] pre-activation Z).
//   out   : output array, length n (post-activation A). May alias `in`.
//   n     : total elements (batch * out_features).
//   alpha : negative-side slope (0 < alpha < 1; this repo uses 0.01).
//
// Launch config: the standard 1-D element-wise grid (kBlockSize threads/block).
// ============================================================================
__global__ void leaky_relu_forward(const float* in, float* out, int n, float alpha);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_leaky_relu_forward(const float* in, float* out, int n, float alpha);


// ============================================================================
// 12) leaky_relu_backward — gradient through LeakyReLU.
// ----------------------------------------------------------------------------
// d/dz LeakyReLU(z) = 1 if z > 0 else alpha. So the upstream gradient passes
// through unchanged on the positive side and is scaled by `alpha` (not zeroed)
// on the negative side. Like relu_backward we gate on the cached PRE-activation.
//   grad_in[i] = grad_out[i] * (pre_act[i] > 0 ? 1 : alpha)
//
// Parameters:
//   grad_out : upstream gradient dA, length n.
//   pre_act  : the layer's pre-activation Z (used only for its sign), length n.
//   grad_in  : output gradient dZ (overwritten), length n. May alias grad_out.
//   n        : total elements (batch * out_features).
//   alpha    : same negative-side slope used in the forward pass.
// ============================================================================
__global__ void leaky_relu_backward(const float* grad_out, const float* pre_act,
                                    float* grad_in, int n, float alpha);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_leaky_relu_backward(const float* grad_out, const float* pre_act,
                                float* grad_in, int n, float alpha);


// ============================================================================
// 13) tanh_forward — out[i] = tanh(in[i]).
// ----------------------------------------------------------------------------
// The hyperbolic-tangent activation, squashing each input into (-1, 1). Smooth
// and zero-centered (unlike ReLU). Purely element-wise, computed with the CUDA
// math-library intrinsic tanhf().
//
// Parameters:
//   in  : input array, length n (pre-activation Z).
//   out : output array, length n (post-activation A = tanh(Z)). May alias `in`.
//   n   : total elements (batch * out_features).
// ============================================================================
__global__ void tanh_forward(const float* in, float* out, int n);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_tanh_forward(const float* in, float* out, int n);


// ============================================================================
// 14) tanh_backward — gradient through tanh.
// ----------------------------------------------------------------------------
// d/dz tanh(z) = 1 - tanh(z)^2. The crucial, easy-to-miss detail: this is
// expressed in terms of the OUTPUT a = tanh(z), so tanh_backward consumes the
// cached POST-activation `act` (the layer's A), NOT the pre-activation Z that
// relu_backward / leaky_relu_backward use. Mixing these up is a classic bug; we
// keep both Z and A cached so each activation can read whichever it needs.
//   grad_in[i] = grad_out[i] * (1 - act[i]*act[i])
//
// Parameters:
//   grad_out : upstream gradient dA, length n.
//   act      : the layer's POST-activation A = tanh(Z), length n.
//   grad_in  : output gradient dZ (overwritten), length n.
//   n        : total elements (batch * out_features).
// ============================================================================
__global__ void tanh_backward(const float* grad_out, const float* act,
                              float* grad_in, int n);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_tanh_backward(const float* grad_out, const float* act,
                          float* grad_in, int n);


// ============================================================================
// 15) reduce_sum_kernel — one tree-reduction pass summing an array.
// ----------------------------------------------------------------------------
// THE canonical CUDA parallel-reduction lesson. Summing n numbers is inherently
// sequential (each add depends on the last), but a *tree* turns it into log2(n)
// parallel steps: pair up neighbors, sum each pair, repeat on the halved array.
//
// This kernel reduces WITHIN each block using shared memory and emits ONE
// partial sum per block into `out[blockIdx.x]`. Because a block can only sum the
// elements it owns, fully reducing an array of n elements takes multiple passes
// (n -> #blocks -> ... -> 1); the launch_reduce_sum wrapper drives that loop.
//
// Design choices (all explained in src/kernels.cu and docs/cuda_concepts.md):
//   * Each thread first adds TWO global elements at load time ("first add during
//     load"), halving the number of idle threads and global reads.
//   * The in-block reduction walks `s = blockDim.x/2, /4, ... 1`, with a
//     __syncthreads() between steps so every partial is visible before it is
//     consumed. This requires blockDim.x to be a power of two (kBlockSize=256 ✓).
//   * Shared-memory size is passed dynamically as the 3rd launch argument
//     (blockDim.x * sizeof(float)); inside, it is `extern __shared__ float[]`.
//
// Parameters:
//   in  : input array in device global memory, length n.
//   out : output array, length >= gridDim.x; out[b] = sum of block b's slice.
//   n   : number of valid input elements (the grid may cover more; we guard).
// ============================================================================
__global__ void reduce_sum_kernel(const float* in, float* out, int n);

// Host wrapper: fully reduces d_in[0..n) to a single scalar by repeatedly
// launching reduce_sum_kernel (ping-ponging two scratch buffers) until one value
// remains, then copies it to the host and returns it. Returns 0 for n <= 0.
// This is "reduce on the GPU, read back one float" — vastly less host<->device
// traffic than copying the whole array back and summing on the CPU.
float launch_reduce_sum(const float* d_in, int n);


// ============================================================================
// 16) predictions_correct — per-row argmax-equals-label indicator (1.0 / 0.0).
// ----------------------------------------------------------------------------
// For each row (sample) r, find the predicted class = argmax_c probs[r,c] and
// write 1.0 if it equals labels[r], else 0.0. Summing this vector (via
// launch_reduce_sum) and dividing by `rows` gives accuracy — entirely on the
// GPU. This pairs with the reduction above to show a full device-side metric,
// contrasting with the host-side argmax loop in mlp_accuracy().
//
// Parameters:
//   probs   : [rows, cols] softmax probabilities. GPU global memory.
//   labels  : length `rows`, true class indices. DEVICE int array.
//   correct : length `rows` output; correct[r] in {0.0f, 1.0f} (overwritten).
//   rows    : batch size.   cols : num_classes.
//
// Launch config: 1-D grid over rows, one thread per row; guard r < rows.
// ============================================================================
__global__ void predictions_correct(const float* probs, const int* labels,
                                    float* correct, int rows, int cols);

// Host wrapper: 1-D launch over `rows`, then CUDA_CHECK_LAST().
void launch_predictions_correct(const float* probs, const int* labels,
                                float* correct, int rows, int cols);


// ============================================================================
// ============================================================================
//  ADDED IN PUSH 0003 — on-device random numbers + dropout regularization.
//  See docs/changelog/0003-*.md and the "On-device RNG" section in
//  docs/cuda_concepts.md.
// ============================================================================
// ============================================================================


// ============================================================================
// 17) fill_uniform — out[i] = a uniform random float in [0, 1).
// ----------------------------------------------------------------------------
// Demonstrates a COUNTER-BASED (stateless) GPU RNG: instead of each thread
// advancing a shared/per-thread RNG state (which needs storage and careful
// seeding), every element's value is a hash of (seed, index):
//     out[i] = uniform( hash(seed, i) )
// There is NO state to store and NO synchronization — thread i independently
// computes its own number — yet results are fully reproducible given (seed, i).
// This is exactly the idea behind cuRAND's Philox generator; here we hand-roll a
// small splitmix64-style integer hash (see src/kernels.cu) to stay dependency-
// free. Used by the RNG self-test in main.cu, and the same hash drives dropout.
//
// Parameters:
//   out  : output array, length n, filled with uniforms in [0,1). GPU global mem.
//   n    : number of elements.
//   seed : 64-bit seed; change it to get an independent stream of numbers.
//
// Launch config: the standard 1-D element-wise grid (kBlockSize threads/block).
// ============================================================================
__global__ void fill_uniform(float* out, int n, unsigned long long seed);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_fill_uniform(float* out, int n, unsigned long long seed);


// ============================================================================
// 18) dropout_forward — "inverted dropout": randomly zero, scale the survivors.
// ----------------------------------------------------------------------------
// Dropout is a regularizer: during TRAINING it randomly sets each activation to
// 0 with probability p (so the network can't rely on any single unit), which
// reduces overfitting. The surviving activations are scaled UP by 1/(1-p) so the
// expected value of each activation is unchanged — that is "inverted dropout",
// and its payoff is that INFERENCE needs no dropout and no rescaling at all.
//
// Per element i, using the same counter-based RNG as fill_uniform:
//     u        = uniform(seed, i)
//     keep     = (u >= p)                      // dropped with probability p
//     mask[i]  = keep ? (1/(1-p)) : 0          // the SCALED multiplier
//     out[i]   = in[i] * mask[i]
// The scaled mask is written out and CACHED so dropout_backward can reuse the
// exact same pattern (gradients must flow only through the kept units).
//
// REPRODUCIBILITY MATTERS: because the mask depends only on (seed, i) — NOT on
// the activations or weights — holding `seed` fixed makes dropout a deterministic
// function of the inputs. That is what lets the finite-difference gradient check
// validate the dropout backward pass (mlp_grad_check freezes the RNG seed).
//
// Parameters:
//   in         : input activations, length n. May alias out (in-place safe).
//   out        : output activations, length n (in[i]*mask[i]).
//   mask       : output cached scaled mask, length n (reused by the backward).
//   n          : total elements (batch * out_features).
//   keep_scale : the survivor scale 1/(1-p) (the launcher computes it from p).
//   p          : drop probability in [0,1).
//   seed       : per-call RNG seed (varies the mask across training steps).
// ============================================================================
__global__ void dropout_forward(const float* in, float* out, float* mask,
                                int n, float keep_scale, float p,
                                unsigned long long seed);

// Host wrapper: computes keep_scale = 1/(1-p), then 1-D launch over n elements.
void launch_dropout_forward(const float* in, float* out, float* mask,
                            int n, float p, unsigned long long seed);


// ============================================================================
// 19) dropout_backward — grad_in[i] = grad_out[i] * mask[i].
// ----------------------------------------------------------------------------
// Backprop through dropout simply reuses the SCALED mask cached during the
// forward pass: dropped units (mask 0) receive zero gradient, survivors receive
// the upstream gradient scaled by the same 1/(1-p). No RNG is needed here — the
// pattern is already fixed in `mask`. This is why the forward caches the mask.
//
// Parameters:
//   grad_out : upstream gradient, length n. May alias grad_in (in-place safe).
//   mask     : the scaled mask cached by dropout_forward, length n.
//   grad_in  : output gradient, length n (grad_out[i] * mask[i]).
//   n        : total elements (batch * out_features).
// ============================================================================
__global__ void dropout_backward(const float* grad_out, const float* mask,
                                 float* grad_in, int n);

// Host wrapper: 1-D launch over n elements, then CUDA_CHECK_LAST().
void launch_dropout_backward(const float* grad_out, const float* mask,
                             float* grad_in, int n);


// ============================================================================
// ============================================================================
//  ADDED IN PUSH 0006 — KERNEL FUSION. One kernel does the whole forward linear
//  layer (matmul + bias + activation) that used to take three.
//  See docs/changelog/0006-*.md and docs/cuda_concepts.md ("Kernel fusion").
// ============================================================================
// ============================================================================


// ============================================================================
// 20) gemm_bias_act — FUSED linear layer:  Z = Ain·W + bias ;  Aout = act(Z).
// ----------------------------------------------------------------------------
// Fuses the THREE kernels the forward pass used to run per hidden layer — the
// matmul (gemm_tiled), the bias add (add_bias), and the activation
// (relu/leaky/tanh_forward) — into ONE. Each thread computes one output element
//   C[m,n] = Σ_k Ain[m,k]·W[k,n]
// via the same shared-memory tiling as gemm_tiled, then — while that value is
// still in a REGISTER — adds bias[n] and applies the activation, writing the
// pre-activation Z (kept for backprop) and the post-activation Aout. This avoids
// writing Z to global memory and reading it back twice (once to add bias, once to
// activate) and turns 3 kernel launches into 1.
//
// WHY FUSION HELPS: these element-wise epilogue steps are memory-bound — they do
// almost no arithmetic per element but each separate kernel must stream the whole
// [M,N] tensor through global memory. Doing them in-register right after the
// matmul removes those extra round-trips (and two launch + sync overheads).
//
// act_type selects the activation (the 0..2 values match mlp.cuh's Activation):
//     0 = ReLU      1 = LeakyReLU (negative slope `alpha`)
//     2 = Tanh      3 = Identity  (used by the OUTPUT layer; softmax_rows is then
//                                  applied to Z separately, as softmax is row-wise
//                                  and cannot be fused element-by-element)
//
// Shapes (row-major, NO transpose — the forward linear step never needs one):
//   Ain  : [M,K]  previous activations (M = batch, K = in_features)
//   W    : [K,N]  weights              (K = in_features, N = out_features)
//   bias : [N]    per-output bias
//   Z    : [M,N]  output pre-activation  (written for the backward pass)
//   Aout : [M,N]  output post-activation = act(Z)
//
// Launch config (set by launch_gemm_bias_act): block = kTileDim×kTileDim, grid
// covers [M,N] — identical to gemm_tiled; only the per-thread epilogue differs.
// ============================================================================
__global__ void gemm_bias_act(const float* Ain, const float* W, const float* bias,
                              float* Z, float* Aout,
                              int M, int N, int K, int act_type, float alpha);

// Host wrapper: 2-D tiled launch covering [M,N], then CUDA_CHECK_LAST().
void launch_gemm_bias_act(const float* Ain, const float* W, const float* bias,
                          float* Z, float* Aout,
                          int M, int N, int K, int act_type, float alpha);
