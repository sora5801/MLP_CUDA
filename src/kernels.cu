// =============================================================================
// src/kernels.cu  —  THE HEART of the MLP: every CUDA kernel + its launcher.
// -----------------------------------------------------------------------------
// This file contains all the `__global__` device kernels (the code that actually
// runs on the GPU, one instance per thread) together with `__host__` launcher
// wrappers (`launch_*`) that compute the grid/block configuration, fire the
// kernel, and check for errors. The rest of the project (mlp.cu, main.cu) only
// ever calls the `launch_*` wrappers — they never touch `<<<>>>` directly. That
// separation keeps the "how many threads / how big a grid" decision in exactly
// one place per operation, which is the single most error-prone part of CUDA.
//
// Conventions used throughout (see MLP_CUDA_BUILD_SPEC.md §0.2):
//   * All matrices are ROW-MAJOR flat float arrays in GPU global memory.
//   * A weight matrix W for a layer with `in` inputs / `out` outputs is [in,out],
//     so W[i*out + o].  A batch of activations A is [batch,features], A[r*F + c].
//   * Forward linear layer (NO transpose): Z = A_prev · W + b.
//
// A reader who knows C++ but is new to CUDA should be able to learn the model
// from the comments alone. We comment thread-index arithmetic line by line and
// favor clarity over micro-optimization; skipped optimizations are flagged with
// "// (optimization: ... left as an exercise)".
// =============================================================================

#include "kernels.cuh"   // our own kernel + launcher declarations (must agree)
#include "common.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST, kBlockSize, kTileDim, ceil_div

// =============================================================================
// 1) gemm_naive  —  general matrix multiply with optional transposes.
// =============================================================================
//
// WHAT IT COMPUTES
//   C[M,N] = opA(A) · opB(B), where opA(A) is logically [M,K] and opB(B) is
//   logically [K,N]. Element-wise:  C[m,n] = sum_{k=0..K-1} A_log[m,k]*B_log[k,n].
//
// WHY IT EXISTS
//   This single kernel is the correctness workhorse for the WHOLE network. The
//   transpose flags let us reuse it for all three matrix products in training
//   without ever physically transposing a buffer (which would cost a copy):
//     forward  : Z      = A_prev · W           (transA=false, transB=false)
//     dW       = A_prev^T · dZ                 (transA=true,  transB=false)
//     dA_prev  = dZ · W^T                      (transA=false, transB=true)
//
// PARAMETER SHAPES / UNITS (all device pointers, row-major float arrays)
//   A : if !transA it is stored [M,K]; if transA it is stored [K,M].
//   B : if !transB it is stored [K,N]; if transB it is stored [N,K].
//   C : output, stored [M,N]; this kernel writes every element exactly once.
//   M : number of rows of the (logical) result  (= rows of opA(A)).
//   N : number of cols of the (logical) result  (= cols of opB(B)).
//   K : the shared/contraction dimension (cols of opA(A) = rows of opB(B)).
//   transA/transB : whether A/B are stored transposed relative to their logical
//                   shape. We index AROUND the storage, never moving data.
//
// EXECUTION CONFIGURATION (set by launch_gemm)
//   2-D grid of 2-D blocks. Each THREAD computes exactly ONE element C[m,n].
//   blockDim = (kTileDim, kTileDim) = 16x16 = 256 threads; grid covers M x N.
//
// MEMORY LAYOUT / PHYSICAL INDEXING (the crux — get this exactly right)
//   "Logical" index A_log[m,k] is what the math wants; the physical offset into
//   the stored flat array depends on whether that operand is transposed:
//     A_log[m,k] = (!transA) ? A[m*K + k]   // A stored [M,K], row stride K
//                            :  A[k*M + m]   // A stored [K,M], row stride M
//     B_log[k,n] = (!transB) ? B[k*N + n]   // B stored [K,N], row stride N
//                            :  B[n*K + k]   // B stored [N,K], row stride K
//   The reason: in a row-major [R,Cs] array, element (r,c) lives at r*Cs + c.
//   When transposed, the *logical* (m,k) is the *stored* (k,m), so we swap the
//   roles of the indices AND use the stored array's actual column count as the
//   stride. Getting the stride wrong (using K vs M) is the classic GEMM bug, so
//   each branch below names the stride explicitly.
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K, bool transA, bool transB) {
    // ---- Map this thread to one output element C[row, col]. ----
    // blockIdx.{x,y} index the block within the grid; threadIdx.{x,y} index the
    // thread within its block; blockDim.{x,y} is the block's edge length.
    // We deliberately map the FAST-moving x dimension to the COLUMN n, because
    // adjacent threads (consecutive threadIdx.x) then write adjacent addresses
    // C[m*N + n], C[m*N + n+1], ... which is a coalesced (efficient) store.
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // n in [0, N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // m in [0, M)

    // The grid is rounded UP to whole blocks (ceil_div), so the last blocks have
    // threads that fall outside the matrix. They must do nothing — otherwise we
    // read/write out of bounds. This guard is mandatory for non-multiple sizes.
    if (row >= M || col >= N) return;

    // Accumulate the dot product of (logical) row `row` of opA(A) with (logical)
    // column `col` of opB(B). A local float lives in a register: fast, private
    // to this thread, and the partial sums never touch global memory.
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // Fetch A_log[row, k]: pick storage offset by transpose flag (see banner).
        //   !transA: A is [M,K] -> offset row*K + k   (stride K = stored cols)
        //    transA: A is [K,M] -> offset k*M + row   (stride M = stored cols)
        float a = (!transA) ? A[row * K + k]
                            :  A[k * M + row];

        // Fetch B_log[k, col]:
        //   !transB: B is [K,N] -> offset k*N + col    (stride N = stored cols)
        //    transB: B is [N,K] -> offset col*K + k    (stride K = stored cols)
        float b = (!transB) ? B[k * N + col]
                            :  B[col * K + k];

        // Multiply-accumulate. (optimization: shared-memory tiling to reuse
        // operands is shown in gemm_tiled; here we keep it dead simple/correct.)
        acc += a * b;
    }

    // C is always stored [M,N] (never transposed), row stride N. One write/elem.
    C[row * N + col] = acc;
}

// ---- launch_gemm: host wrapper that configures and fires gemm_naive. --------
// WHY a wrapper: callers should not have to recompute the 2-D grid math (and get
// it wrong). We build a 16x16 block and a grid large enough to cover N columns
// (x) and M rows (y), then surface any launch/exec error immediately.
void launch_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, bool transA, bool transB) {
    // 2-D block: x spans columns (N), y spans rows (M). 16*16 = 256 threads,
    // a good general-purpose occupancy sweet spot and == kBlockSize.
    dim3 block(kTileDim, kTileDim);
    // Grid covers the output: ceil_div ensures we have enough blocks even when
    // N/M are not multiples of the block edge (the in-kernel guard handles the
    // leftover threads). x <-> N (cols), y <-> M (rows), matching the kernel map.
    dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
    gemm_naive<<<grid, block>>>(A, B, C, M, N, K, transA, transB);
    CUDA_CHECK_LAST();  // checks cudaGetLastError() then synchronizes (didactic)
}

// =============================================================================
// 2) gemm_tiled  —  shared-memory tiled GEMM, NO transpose: C[M,N]=A[M,K]·B[K,N].
// =============================================================================
//
// WHAT IT COMPUTES
//   Same math as gemm_naive with transA=transB=false, but it is an OPTIMIZATION
//   lesson: it stages square tiles of A and B into fast on-chip __shared__ memory
//   so each loaded value is reused by an entire row/column of the tile instead of
//   being re-fetched from slow global memory by every thread.
//
// WHY TILING IS FASTER (the core CUDA performance idea)
//   Global memory has high latency and limited bandwidth; shared memory is ~100x
//   faster and is shared by all threads in a block. In the naive kernel each
//   element of A and B is read K-ish times straight from global memory. Here a
//   block of TILE*TILE threads cooperatively loads one TILE*TILE chunk of A and
//   one of B (each thread loads ONE element of each), then every thread reads its
//   needed operands from shared memory TILE times. That cuts global-memory
//   traffic by ~TILE and lets the loads coalesce nicely.
//
// PARAMETER SHAPES / UNITS
//   A : [M,K] row-major.   B : [K,N] row-major.   C : [M,N] row-major (output).
//   M,N,K : as in gemm_naive. No transpose flags — this variant is square-tiled
//           and used only by the microbenchmark and as a teaching example.
//
// EXECUTION CONFIGURATION (set by launch_gemm_tiled)
//   block = (kTileDim, kTileDim); grid covers M x N exactly as launch_gemm does.
//   TILE == kTileDim (16). Each block computes one TILE*TILE patch of C.
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    // Compile-time tile edge so the __shared__ arrays have a known size.
    const int TILE = kTileDim;

    // Two tiles in shared memory, one per operand. These are PER-BLOCK: all 256
    // threads of this block see the same sA/sB. Lifetime = the block's lifetime.
    __shared__ float sA[TILE][TILE];   // a TILE*TILE chunk of A
    __shared__ float sB[TILE][TILE];   // a TILE*TILE chunk of B

    // Thread's position WITHIN its block tile (0..TILE-1 each).
    int ty = threadIdx.y;   // local row    (down the tile)
    int tx = threadIdx.x;   // local column (across the tile)

    // This thread's GLOBAL output coordinate C[row, col].
    // row uses the y dimension (down M), col uses the x dimension (across N) —
    // again x maps to the contiguous column so global stores stay coalesced.
    int row = blockIdx.y * TILE + ty;   // m in [0, M)
    int col = blockIdx.x * TILE + tx;   // n in [0, N)

    // Per-thread register accumulator for C[row,col].
    float acc = 0.0f;

    // March along the K dimension one TILE-wide strip at a time. We need
    // ceil(K/TILE) strips; the last strip may be partially out of range when K is
    // not a multiple of TILE, which we mask with zeros below.
    int numTiles = (K + TILE - 1) / TILE;   // == ceil_div(K, TILE)
    for (int t = 0; t < numTiles; ++t) {
        // --- Cooperative load: each thread brings in ONE element of each tile. ---
        // The k-column of A this thread loads, and the k-row of B it loads:
        int aCol = t * TILE + tx;   // column index into A for sA[ty][tx]
        int bRow = t * TILE + ty;   // row    index into B for sB[ty][tx]

        // Load A[row, aCol] into sA[ty][tx], guarding both dims. Threads whose
        // (row, aCol) is outside A store 0 so they contribute nothing to the dot
        // product (zero-padding the edge tiles keeps the inner loop branch-free).
        sA[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // Load B[bRow, col] into sB[ty][tx], guarding both dims likewise.
        sB[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // Barrier: ensure the ENTIRE tile is loaded by all threads before anyone
        // reads it. Without this, a fast thread could read sA/sB slots that a
        // slow thread has not written yet — a classic shared-memory race.
        __syncthreads();

        // --- Compute: this thread's partial dot product over the loaded tile. ---
        // sA[ty][.] is this thread's row of A within the tile; sB[.][tx] is its
        // column of B. Each sA/sB element loaded once is reused TILE times here.
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        // Barrier again: do NOT let any thread overwrite sA/sB with the NEXT
        // tile's data until every thread has finished consuming THIS tile.
        __syncthreads();
    }

    // Write the result, guarding the edge threads (grid was rounded up).
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ---- launch_gemm_tiled: host wrapper for the tiled GEMM. --------------------
// Identical grid/block math to launch_gemm; only the kernel differs. Used by the
// microbenchmark in main.cu to demonstrate the shared-memory speedup.
void launch_gemm_tiled(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    dim3 block(kTileDim, kTileDim);                          // 16x16 = 256 threads
    dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));  // cover N (x) and M (y)
    gemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 3) add_bias  —  broadcast-add a per-column bias across every row.
// =============================================================================
//
// WHAT/WHY: completes the linear layer Z = A_prev·W + b. The GEMM produced the
// A_prev·W part; here we add bias[c] to EVERY row r of column c. The bias is
// shape [1,out] and is reused (broadcast) down all `rows` of Z.
//
// SHAPES: Z is [rows, cols] (row-major), modified IN PLACE. bias has length cols.
//
// EXECUTION: 1-D grid of kBlockSize threads, one thread per element of Z
//   (rows*cols total). We recover (r,c) from the flat index to pick bias[c].
__global__ void add_bias(float* Z, const float* bias, int rows, int cols) {
    // Flat global thread index over all rows*cols elements.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;          // guard the padding threads (ceil_div)

    // Recover the column from the flat row-major index: element idx is at
    // (r,c) with r = idx/cols, c = idx%cols. We only need c to index the bias.
    int c = idx % cols;

    // Broadcast add: every row shares the same bias[c]. In-place update of Z.
    Z[idx] += bias[c];
}

// ---- launch_add_bias: 1-D launch over rows*cols elements. -------------------
void launch_add_bias(float* Z, const float* bias, int rows, int cols) {
    int total = rows * cols;                       // number of elements to touch
    int block = kBlockSize;                        // 256 threads / block (1-D)
    int grid  = ceil_div(total, block);            // enough blocks to cover them
    add_bias<<<grid, block>>>(Z, bias, rows, cols);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 4) relu_forward  —  out[i] = max(0, in[i]).
// =============================================================================
//
// WHAT/WHY: the hidden-layer nonlinearity. ReLU is cheap and avoids vanishing
// gradients for positive inputs. It is purely element-wise, so it is the textbook
// 1-D element-wise kernel.
//
// SHAPES: `in` and `out` are flat arrays of length n (here n = batch*out_features
// for a layer). They may alias the same buffer safely (each element independent).
//
// NOTE ON DIVERGENCE: the ternary causes threads in a warp to take different
// "branches", but since both arms are a single value (x or 0) the compiler emits
// a predicated select, not a real branch — so there is effectively no divergence.
__global__ void relu_forward(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // flat element index
    if (i >= n) return;                             // guard padding threads
    float x = in[i];
    out[i] = (x > 0.0f) ? x : 0.0f;                 // max(0, x)
}

// ---- launch_relu_forward: 1-D launch over n elements. -----------------------
void launch_relu_forward(const float* in, float* out, int n) {
    int block = kBlockSize;
    int grid  = ceil_div(n, block);
    relu_forward<<<grid, block>>>(in, out, n);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 5) relu_backward  —  grad_in[i] = (pre_act[i] > 0) ? grad_out[i] : 0.
// =============================================================================
//
// WHAT/WHY: backprop through ReLU. The derivative of max(0,x) is 1 where the
// PRE-activation x>0 and 0 where x<=0, so the incoming gradient passes through
// unchanged on the active side and is zeroed on the dead side. We use the cached
// pre-activation `pre_act` (the layer's Z), NOT the post-activation, to decide.
//
// SHAPES: grad_out, pre_act, grad_in are all flat length-n arrays (n=batch*out).
//   grad_out : dL/d(post-activation)   (incoming gradient)
//   pre_act  : the layer's Z (pre-activation) used only for its sign mask
//   grad_in  : dL/d(pre-activation)    (output of this kernel)
__global__ void relu_backward(const float* grad_out, const float* pre_act,
                              float* grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // flat element index
    if (i >= n) return;                             // guard padding threads
    // Gate the upstream gradient by the sign of the pre-activation.
    grad_in[i] = (pre_act[i] > 0.0f) ? grad_out[i] : 0.0f;
}

// ---- launch_relu_backward: 1-D launch over n elements. ----------------------
void launch_relu_backward(const float* grad_out, const float* pre_act,
                          float* grad_in, int n) {
    int block = kBlockSize;
    int grid  = ceil_div(n, block);
    relu_backward<<<grid, block>>>(grad_out, pre_act, grad_in, n);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 6) softmax_rows  —  row-wise, numerically-stable softmax.
// =============================================================================
//
// WHAT/WHY: the output-layer activation. For each row r (one sample) it turns the
// `cols` logits into a probability distribution that sums to 1:
//     probs[r,c] = exp(logits[r,c]) / sum_j exp(logits[r,j]).
//
// NUMERICAL STABILITY: exp() of a large logit overflows to +inf. Subtracting the
// row maximum first (softmax(x) == softmax(x - max)) keeps the largest exponent
// at exp(0)=1 and all others in (0,1], so no overflow — the standard trick.
//
// SHAPES: logits and probs are [rows, cols] row-major. probs is the output and
//   may alias logits (we read each row fully into registers before writing? no —
//   we read directly; aliasing is fine because we only overwrite after reads of
//   that same row, and a single thread owns the whole row).
//
// EXECUTION: ONE THREAD PER ROW. `cols` (= num_classes) is tiny (e.g. 3), so a
//   per-row serial loop is simpler and plenty fast; we avoid an intra-row
//   parallel reduction. (optimization: one warp per row with a shuffle reduction
//   is the fast version, left as an exercise.)
__global__ void softmax_rows(const float* logits, float* probs,
                             int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's row
    if (r >= rows) return;                            // guard padding threads

    // Base offset of row r in the flat row-major array (row stride = cols).
    const float* row_in  = logits + (size_t)r * cols;
    float*       row_out = probs  + (size_t)r * cols;

    // Pass 1: find the row max for numerical stability.
    float maxv = row_in[0];
    for (int c = 1; c < cols; ++c) {
        if (row_in[c] > maxv) maxv = row_in[c];
    }

    // Pass 2: exponentiate the shifted logits and accumulate their sum. We stash
    // each exp into row_out temporarily so we don't have to recompute exp later.
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float e = expf(row_in[c] - maxv);   // in (0,1], no overflow
        row_out[c] = e;
        sum += e;
    }

    // Pass 3: normalize so the row sums to 1. (sum >= 1 since the max term is 1,
    // so the division is safe.) Multiply by reciprocal: one divide, cols mults.
    float inv = 1.0f / sum;
    for (int c = 0; c < cols; ++c) {
        row_out[c] *= inv;
    }
}

// ---- launch_softmax_rows: 1-D launch over rows (one thread per row). --------
void launch_softmax_rows(const float* logits, float* probs, int rows, int cols) {
    int block = kBlockSize;
    int grid  = ceil_div(rows, block);     // one thread per row, so cover `rows`
    softmax_rows<<<grid, block>>>(logits, probs, rows, cols);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 7) cross_entropy_grad  —  output-layer gradient of MEAN softmax-CE wrt logits.
// =============================================================================
//
// WHAT/WHY: a beautiful identity (derived in docs/math_derivation.md §3) collapses
// the softmax + cross-entropy backprop into:
//     dL/dZ_out[r,c] = (probs[r,c] - onehot[r,c]) / rows,
// where onehot[r,c] = 1 if c == labels[r] else 0. The "/rows" folds in the
// batch-mean (1/batch) so that EVERY downstream gradient (dW, db, dA) already
// carries the 1/batch scale and the SGD step is simply param -= lr*grad.
//
// SHAPES: probs [rows,cols] (softmax output). labels: device int[rows], each in
//   [0,cols). grad [rows,cols] output = dL/dZ for the output layer.
//
// EXECUTION: one thread per element (rows*cols). Recover (r,c) from the flat idx.
__global__ void cross_entropy_grad(const float* probs, const int* labels,
                                   float* grad, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // flat element index
    int total = rows * cols;
    if (idx >= total) return;                         // guard padding threads

    // Recover (r,c) from the row-major flat index.
    int r = idx / cols;
    int c = idx % cols;

    // onehot term: subtract 1 from the probability of the TRUE class only.
    float onehot = (labels[r] == c) ? 1.0f : 0.0f;

    // (probs - onehot) / rows  — the /rows is the batch-mean factor.
    grad[idx] = (probs[idx] - onehot) / (float)rows;
}

// ---- launch_cross_entropy_grad: 1-D launch over rows*cols elements. ---------
void launch_cross_entropy_grad(const float* probs, const int* labels,
                               float* grad, int rows, int cols) {
    int total = rows * cols;
    int block = kBlockSize;
    int grid  = ceil_div(total, block);
    cross_entropy_grad<<<grid, block>>>(probs, labels, grad, rows, cols);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 8) cross_entropy_loss  —  per-row CE loss, for reporting & the grad-check.
// =============================================================================
//
// WHAT/WHY: the scalar loss for sample r is the negative log-probability the model
// assigned to the TRUE class:  loss[r] = -log(probs[r, labels[r]]). The host sums
// these and divides by rows to get the mean CE used for printing and for the
// finite-difference gradient check. We clamp the probability to >= 1e-12 so that
// log() never sees 0 (which would be -inf) due to round-off.
//
// SHAPES: probs [rows,cols]; labels device int[rows]; loss_per_row device
//   float[rows] (output, one scalar per sample).
//
// EXECUTION: one thread per row.
__global__ void cross_entropy_loss(const float* probs, const int* labels,
                                   float* loss_per_row, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's row
    if (r >= rows) return;                            // guard padding threads

    int label = labels[r];                            // true class for this row
    // Probability assigned to the true class lives at (r, label) = r*cols + label.
    float p = probs[(size_t)r * cols + label];

    // Clamp away from 0 before log to avoid -inf from numerical underflow.
    float clamped = (p > 1e-12f) ? p : 1e-12f;
    loss_per_row[r] = -logf(clamped);
}

// ---- launch_cross_entropy_loss: 1-D launch over rows (one thread per row). --
void launch_cross_entropy_loss(const float* probs, const int* labels,
                               float* loss_per_row, int rows, int cols) {
    int block = kBlockSize;
    int grid  = ceil_div(rows, block);
    cross_entropy_loss<<<grid, block>>>(probs, labels, loss_per_row, rows, cols);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 9) bias_grad  —  bias gradient = column-sum of dZ over the batch rows.
// =============================================================================
//
// WHAT/WHY: because bias is broadcast across all rows in the forward pass
// (add_bias), its gradient is the SUM of the upstream gradient down each column:
//     db[c] = sum_{r=0..rows-1} dZ[r,c].
// The 1/batch factor is already inside dZ (it came from cross_entropy_grad and
// flowed back through the chain), so we do NOT divide again here.
//
// SHAPES: dZ [rows,cols] (= the layer's dZ). db length cols (output).
//
// EXECUTION: ONE THREAD PER COLUMN c (cols is small). Each thread walks all rows
//   of its column and accumulates. (optimization: a parallel reduction across
//   rows would scale better for huge batches — left as an exercise.)
__global__ void bias_grad(const float* dZ, float* db, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's column
    if (c >= cols) return;                            // guard padding threads

    // Serial column-sum: stride by `cols` to hop from row r to row r+1 within
    // the same column c (row-major: dZ[r,c] = dZ[r*cols + c]).
    float sum = 0.0f;
    for (int r = 0; r < rows; ++r) {
        sum += dZ[(size_t)r * cols + c];
    }
    db[c] = sum;
}

// ---- launch_bias_grad: 1-D launch over columns (one thread per column). -----
void launch_bias_grad(const float* dZ, float* db, int rows, int cols) {
    int block = kBlockSize;
    int grid  = ceil_div(cols, block);     // one thread per column
    bias_grad<<<grid, block>>>(dZ, db, rows, cols);
    CUDA_CHECK_LAST();
}

// =============================================================================
// 10) sgd_update  —  in-place stochastic gradient descent step.
// =============================================================================
//
// WHAT/WHY: the parameter update. Because the 1/batch mean was folded into the
// gradients upstream (see cross_entropy_grad), the rule is simply:
//     param[i] -= lr * grad[i].
// Applied to every weight W and bias b of every layer.
//
// SHAPES: param and grad are flat arrays of length n (param updated in place).
//   lr : scalar learning rate (units: step size in parameter space).
//
// EXECUTION: standard 1-D element-wise kernel, one thread per parameter.
__global__ void sgd_update(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // flat parameter index
    if (i >= n) return;                               // guard padding threads
    param[i] -= lr * grad[i];                         // descend the gradient
}

// ---- launch_sgd_update: 1-D launch over n parameters. -----------------------
void launch_sgd_update(float* param, const float* grad, float lr, int n) {
    int block = kBlockSize;
    int grid  = ceil_div(n, block);
    sgd_update<<<grid, block>>>(param, grad, lr, n);
    CUDA_CHECK_LAST();
}
