# CUDA Concepts Used in This Repo

This document is the "CUDA tutorial" companion to the code. Every concept below
is tied to a concrete kernel or launcher in `src/kernels.cu` (declared in
`include/kernels.cuh`) or to a helper in `include/common.cuh` /
`include/matrix.cuh`. The goal is that after reading this, you can open any
kernel in the repo and understand *why* it is written the way it is.

If you know C++ but are new to CUDA, read this top-to-bottom once, then keep it
open beside `src/kernels.cu`.

---

## Table of contents

1. [The mental model: host vs device](#1-the-mental-model-host-vs-device)
2. [Host vs device memory & `cudaMemcpy`](#2-host-vs-device-memory--cudamemcpy)
3. [Threads, blocks, grids](#3-threads-blocks-grids)
4. [The 1-D element-wise launch idiom (`ceil_div`)](#4-the-1-d-element-wise-launch-idiom-ceil_div)
5. [2-D grids for GEMM](#5-2-d-grids-for-gemm)
6. [Memory coalescing](#6-memory-coalescing)
7. [Shared memory & tiling (`gemm_tiled`)](#7-shared-memory--tiling-gemm_tiled)
7b. [Parallel reduction (`reduce_sum`) — push 0002](#7b-parallel-reduction-reduce_sum--push-0002)
8. [Thread divergence (ReLU & guards)](#8-thread-divergence-relu--guards)
9. [Numerical stability (softmax max-subtraction)](#9-numerical-stability-softmax-max-subtraction)
10. [Synchronization (`__syncthreads`, `cudaDeviceSynchronize`)](#10-synchronization-__syncthreads-cudadevicesynchronize)
11. [Asynchronous launches & error checking](#11-asynchronous-launches--error-checking)
12. [Occupancy basics](#12-occupancy-basics)
13. [Putting it together: how the MLP uses all of this](#13-putting-it-together-how-the-mlp-uses-all-of-this)

---

## 1. The mental model: host vs device

A CUDA program runs on two processors at once:

- The **host** = the CPU and its RAM. Your `main()` runs here. Host code is
  ordinary C++.
- The **device** = the GPU and its own separate DRAM ("global memory"). Kernels
  (`__global__` functions) run here, executed by thousands of lightweight
  threads in parallel.

They do **not** share an address space (on the hardware/path this repo targets).
A `float*` returned by `cudaMalloc` is a *device* pointer: it is a valid address
in GPU memory and dereferencing it on the CPU is undefined behavior. Conversely a
`new float[n]` host pointer is meaningless on the GPU. This is the single most
common beginner bug, so this repo encodes the convention in a type:

```c++
// include/matrix.cuh
struct Matrix {
    float* data;   // ALWAYS a device pointer (GPU global memory), row-major
    int rows;
    int cols;
};
```

When you see a `Matrix`, its `.data` lives on the GPU. When you see a bare
`float*` parameter named `host` (e.g. in `matrix_copy_to_device`), it lives on
the CPU. The `Dataset` struct in `include/dataset.cuh` is the opposite: its `X`
and `y` arrays are explicitly documented as **HOST** memory, because dataset
generation, standardization, and shuffling are all done on the CPU.

---

## 2. Host vs device memory & `cudaMemcpy`

All the primitives live in `src/matrix.cu`.

### Allocation

```c++
// matrix_alloc(rows, cols): grab rows*cols floats of GPU global memory.
cudaMalloc(&m.data, rows * cols * sizeof(float));
```

`cudaMalloc` is the GPU analogue of `malloc`: it reserves a contiguous span of
**global memory** (the GPU's large, high-latency DRAM) and writes the device
address into `m.data`. The memory is **uninitialized** — that is why
`matrix_zero` exists (it calls `cudaMemset(m.data, 0, bytes)` to clear it, e.g.
to zero gradient buffers).

`matrix_bytes` centralizes the size computation `rows * cols * sizeof(float)` so
every copy/memset agrees on the byte count. Getting this wrong by a factor of
`sizeof(float)` is a classic crash; one helper removes the chance.

### Copying across the host/device boundary

The CPU cannot write into GPU DRAM directly; data must be DMA'd across the PCIe
bus with `cudaMemcpy`, whose last argument names the **direction**:

```c++
// matrix_copy_to_device: CPU -> GPU.  "H2D"
cudaMemcpy(m.data, host, bytes, cudaMemcpyHostToDevice);

// matrix_copy_to_host:   GPU -> CPU.  "D2H"
cudaMemcpy(host, m.data, bytes, cudaMemcpyDeviceToHost);
```

The direction enum (`cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`) tells
the runtime which pointer is which; passing the wrong one is undefined behavior,
not a compile error. Mnemonic: the **destination is always the first argument**
(same order as `memcpy`), and the direction names *source → destination*.

These transfers are **slow** relative to compute (bus bandwidth ≪ on-GPU
bandwidth) and `cudaMemcpyDeviceToHost` is **synchronous** by default: it blocks
the CPU until the copy finishes, which also means any prior kernels have
completed. That is why `mlp_compute_loss` and `mlp_accuracy` copy only a small
result (per-row losses, or the output probabilities) back to the host rather than
shuttling whole activation tensors around every step. In the training loop in
`src/main.cu`, the per-batch `Matrix` and the device label buffer are **allocated
once and reused** — re-allocating GPU memory every iteration would dominate the
runtime.

---

## 3. Threads, blocks, grids

A kernel launch creates a **grid** of **blocks**, and each block contains
**threads**. The hierarchy exists because it maps onto the hardware:

- A **thread** is the smallest unit; it runs the kernel body once and has its own
  registers and indices.
- A **block** is a group of threads that run on one SM (Streaming
  Multiprocessor), can use fast **shared memory** together, and can synchronize
  with `__syncthreads()`. Threads in *different* blocks cannot synchronize within
  a kernel.
- The **grid** is all the blocks for one launch. Blocks are scheduled
  independently onto whatever SMs are free, in no guaranteed order.

Inside a kernel you locate "which element am I" from built-in variables:

| Variable     | Meaning                                  |
|--------------|------------------------------------------|
| `threadIdx`  | this thread's index within its block     |
| `blockIdx`   | this block's index within the grid       |
| `blockDim`   | number of threads per block (block size) |
| `gridDim`    | number of blocks in the grid             |

Each of these is a `dim3` with `.x`, `.y`, `.z` fields. This repo uses 1-D
indexing (`.x` only) for element-wise kernels and 2-D indexing (`.x`, `.y`) for
GEMM. The block size for 1-D kernels is fixed at `kBlockSize = 256`, and the GEMM
tile is `kTileDim = 16` (so a tiled block is `16*16 = 256` threads); both live in
`include/common.cuh`.

> **Warps.** Within a block, threads execute in lock-step groups of 32 called
> *warps*. You never declare warps, but they explain two things below: divergence
> (§8) and coalescing (§6). Picking block sizes that are multiples of 32 (256 is
> `8*32`) avoids wasting lanes.

---

## 4. The 1-D element-wise launch idiom (`ceil_div`)

Many kernels here just touch every element of a flat array independently:
`relu_forward`, `relu_backward`, and `sgd_update`. Each maps **one thread to one
array element**. The canonical body:

```c++
// relu_forward: out[i] = max(0, in[i])
__global__ void relu_forward(const float* in, float* out, int n) {
    // Global element index of THIS thread:
    //   blockIdx.x  = which block we are in        (0 .. gridDim.x-1)
    //   blockDim.x  = threads per block            (= kBlockSize = 256)
    //   threadIdx.x = our lane inside the block    (0 .. blockDim.x-1)
    // So consecutive threads get consecutive i, and consecutive blocks
    // get consecutive 256-element chunks.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // We almost always launch MORE threads than elements (see launcher below),
    // because the grid is rounded up to whole blocks. Threads with i >= n must
    // do nothing, or they would read/write out of bounds. This `if` guard is
    // mandatory in essentially every CUDA kernel.
    if (i < n) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}
```

The matching launcher computes how many blocks cover `n` elements. We cannot
launch a fractional block, so we round **up** — that is what `ceil_div` is for
(declared `__host__ inline` in `include/common.cuh`):

```c++
// common.cuh: standard "number of blocks to cover n elements" idiom.
__host__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
```

```c++
// launch_relu_forward:
int blocks = ceil_div(n, kBlockSize);     // e.g. n=1000, kBlockSize=256 -> 4 blocks
relu_forward<<<blocks, kBlockSize>>>(in, out, n);  // launches 4*256 = 1024 threads
CUDA_CHECK_LAST();                         // see §11
```

`ceil_div(1000, 256)` = `(1000 + 255) / 256` = `1255 / 256` = `4` (integer
division truncates). `4 * 256 = 1024 >= 1000`, so every element is covered and
the 24 extra threads are killed by the `if (i < n)` guard. **This
`ceil_div` + bounds-guard pattern is the most reused idiom in the whole repo.**

`softmax_rows`, `cross_entropy_grad`, `cross_entropy_loss`, and `bias_grad` use
the same 1-D launch but map a thread to a **row** (or a **column**, for
`bias_grad`) instead of a single element, because each output depends on a whole
row/column. See §9 and §13.

---

## 5. 2-D grids for GEMM

Matrix multiply produces a 2-D result `C[M,N]`, so it is natural to use a **2-D**
block and grid and map **one thread to one output element** `C[m,n]`. This is
`gemm_naive` in `src/kernels.cu`, the correctness workhorse the whole MLP relies
on (forward, `dW`, and `dA` are all `launch_gemm` calls — see the algorithms in
`include/mlp.cuh`).

```c++
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K, bool transA, bool transB) {
    // 2-D thread indexing. By convention here:
    //   y dimension -> rows of C (the M axis)
    //   x dimension -> cols of C (the N axis)
    // We use .y for rows and .x for cols ON PURPOSE so that threads with
    // consecutive threadIdx.x map to consecutive columns n -> consecutive
    // addresses C[m*N + n]. That keeps writes coalesced (see §6).
    int m = blockIdx.y * blockDim.y + threadIdx.y;   // output row
    int n = blockIdx.x * blockDim.x + threadIdx.x;   // output col

    // Guard: the grid is rounded up in both dimensions, so kill threads that
    // fall outside the MxN output.
    if (m < M && n < N) {
        float acc = 0.0f;                  // private accumulator in a register
        for (int k = 0; k < K; ++k) {
            // Logical element A_log[m,k], honoring the optional transpose.
            // A is physically stored [M,K] (no trans) or [K,M] (trans):
            float a = transA ? A[k * M + m] : A[m * K + k];
            // B_log[k,n]; B stored [K,N] (no trans) or [N,K] (trans):
            float b = transB ? B[n * K + k] : B[k * N + n];
            acc += a * b;                  // multiply-accumulate
        }
        C[m * N + n] = acc;                // row-major write of one C element
    }
}
```

The launcher builds a 2-D block and a 2-D grid that tiles the output:

```c++
// launch_gemm:
dim3 block(kTileDim, kTileDim);                       // 16 x 16 = 256 threads
dim3 grid(ceil_div(N, kTileDim), ceil_div(M, kTileDim)); // x covers N, y covers M
gemm_naive<<<grid, block>>>(A, B, C, M, N, K, transA, transB);
CUDA_CHECK_LAST();
```

Note `grid.x = ceil_div(N, ...)` and `grid.y = ceil_div(M, ...)`: the x axis
spans columns (N) and the y axis spans rows (M), matching the index math above.
The transpose flags let one kernel serve all three matmuls without ever
physically transposing memory:

- Forward: `C = A·W`, no transposes.
- Weight grad: `dW = A_prev^T · dZ`, so `transA = true`.
- Input grad: `dA_prev = dZ · W^T`, so `transB = true`.

Reading `A[k*M + m]` instead of `A[m*K + k]` is "transpose for free": same data,
different index arithmetic.

---

## 6. Memory coalescing

Global memory is read/written by the hardware in wide **transactions** (e.g.
128-byte segments). When the 32 threads of a warp access **consecutive** 4-byte
addresses, the hardware fuses them into a few large transactions — this is
*coalescing*, and it is the difference between using the GPU's full bandwidth and
a small fraction of it. When threads of a warp touch scattered addresses, the
access is *serialized* into many transactions and most of each fetched segment is
wasted.

Concretely, in `gemm_naive` a warp is 32 threads with consecutive
`threadIdx.x`, i.e. consecutive output columns `n`. Their final write
`C[m*N + n]` therefore hits consecutive addresses — coalesced. That is exactly
**why** we mapped `.x` to columns and `.y` to rows in §5; the reverse mapping
would still be correct but would scatter the writes and run slower.

The naive kernel is still not great about *reads*: every thread re-reads entire
rows of `A` and columns of `B` from slow global memory, and the same data is
fetched by many threads. Tiling fixes that.

---

## 7. Shared memory & tiling (`gemm_tiled`)

**Shared memory** is a small (tens of KB per block), on-chip scratchpad that is
roughly as fast as registers and is **shared by all threads in a block**. The
idea of tiling: cooperatively load a small `kTileDim × kTileDim` block of `A` and
of `B` from slow global memory into shared memory **once**, then let all 256
threads in the block reuse those values from fast shared memory many times. This
turns O(K) global loads per output element into O(K / kTileDim).

`gemm_tiled` (no-transpose form, `C[M,N] = A[M,K]·B[K,N]`) is the optimization
lesson and the kernel benchmarked against `gemm_naive` in `src/main.cu`.

```c++
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    // Per-block shared scratchpads for one tile of A and one tile of B.
    // __shared__ memory is allocated once per block and visible to all its
    // threads; it lives only for the block's lifetime.
    __shared__ float As[kTileDim][kTileDim];
    __shared__ float Bs[kTileDim][kTileDim];

    int ty = threadIdx.y, tx = threadIdx.x;            // 0..kTileDim-1
    int m  = blockIdx.y * kTileDim + ty;               // output row
    int n  = blockIdx.x * kTileDim + tx;               // output col
    float acc = 0.0f;

    // Walk the K dimension one tile at a time. ceil_div handles K not being a
    // multiple of the tile edge; out-of-range loads are zero-filled below so
    // they contribute nothing to the dot product.
    for (int t = 0; t < ceil_div(K, kTileDim); ++t) {
        int aCol = t * kTileDim + tx;   // column of A this thread loads
        int bRow = t * kTileDim + ty;   // row of B this thread loads

        // Each thread loads ONE element of each tile (bounds-checked).
        As[ty][tx] = (m < M && aCol < K) ? A[m * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && n < N) ? B[bRow * N + n] : 0.0f;

        // Barrier: make sure the WHOLE tile is loaded before anyone reads it.
        __syncthreads();

        // Multiply the two tiles from fast shared memory.
        for (int k = 0; k < kTileDim; ++k)
            acc += As[ty][k] * Bs[k][tx];

        // Barrier again: don't overwrite the tiles (next iteration) until every
        // thread has finished using THIS tile.
        __syncthreads();
    }
    if (m < M && n < N) C[m * N + n] = acc;
}
```

Why it is faster:

- **Bandwidth tier.** Global memory is large but high-latency and
  bandwidth-limited; shared memory is on-chip and ~an order of magnitude faster.
  Tiling moves the repeated reads from the slow tier to the fast tier.
- **Reuse.** Each element loaded into a tile is used `kTileDim` times by
  different threads, so global traffic drops by roughly a factor of `kTileDim`.
- **Coalescing on load.** The `As`/`Bs` loads above use consecutive `tx` for
  consecutive global columns, so the loads themselves are coalesced.

The microbenchmark in `src/main.cu` times both kernels on a 512×512×512 multiply
with CUDA events and prints the speedup, so you can *see* the win. (Further
optimizations — register blocking, vectorized loads, padding `As`/`Bs` to avoid
shared-memory bank conflicts — are left as exercises and noted as such in the
code.)

---

## 7b. Parallel reduction (`reduce_sum`) — push 0002

After the element-wise map and the GEMM, **reduction** is the third foundational
GPU pattern: collapse an array of `n` values into one (a sum, max, etc.). It looks
inherently serial — each add seems to depend on the previous running total — but a
**tree** turns it into `log2(n)` parallel steps. Push 0001 sidestepped this by
copying the per-row loss vector to the host and summing in a C++ loop; push 0002
does it on the GPU and reads back a single float, which is what `reduce_sum_kernel`
+ `launch_reduce_sum` (in `src/kernels.cu`) implement.

### The in-block tree

A block of `B` threads reduces the slice of the array it owns down to one partial
sum, held in `__shared__` memory:

```c++
__global__ void reduce_sum_kernel(const float* in, float* out, int n) {
    extern __shared__ float sdata[];           // B floats (size = 3rd launch arg)
    int tid = threadIdx.x;
    // Each thread adds TWO global elements at load time ("first add during load"):
    // this collapses the first tree level for free and halves global reads.
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float v = 0.0f;
    if (i < n)              v  = in[i];
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();                            // whole tile loaded before reducing

    // Fold the upper half into the lower half, halving the width each step.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();                        // level complete before the next
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];   // block's total
}
```

Things worth pausing on:

- **Why shared memory.** The running partials are read and written `log2(B)` times.
  Keeping them on-chip (shared memory ≈ 100× faster than global) instead of in
  global memory is what makes the reduction fast — the same bandwidth-tier argument
  as tiling (§7).
- **Why `__syncthreads()` every step.** Level `k+1` reads slots that level `k`
  wrote. Without a barrier between them, a fast thread could read a neighbor's slot
  before it was updated — a read-before-write race. And the barrier must be hit by
  *all* threads, so the loop body is uniform (the `if (tid < s)` just makes some
  threads write a no-op-free value; every thread still reaches the `__syncthreads`).
- **Power-of-two block size.** Halving `s = B/2, B/4, … 1` only tiles the block
  exactly when `B` is a power of two — `kBlockSize = 256` (= 2⁸) qualifies.

### Why multiple passes (`launch_reduce_sum`)

A single kernel **cannot** sum across blocks: different blocks share neither memory
nor a barrier, and their scheduling order is undefined. So full reduction is
iterative — each pass turns `n` values into `#blocks` partials:

```
pass 1:  n        elements  ->  ceil(n / 2B)        partials  (in buf_a)
pass 2:  that many          ->  fewer                          (in buf_b)
...      ...                ->  1                               (final scalar)
```

`launch_reduce_sum` "ping-pongs" two scratch buffers (read one, write the other)
until a single value remains, then copies that one float to the host. The MLP uses
it twice: `mlp_compute_loss` sums the per-row losses, and `mlp_accuracy_device`
sums the per-row `predictions_correct` flags — both metrics computed entirely on
the GPU. (Optimization left as an exercise: the *last warp* of each block executes
in lock-step, so its final `log2(32)=5` steps can drop the `__syncthreads` and use
`__shfl_down_sync` warp shuffles; and the two `cudaMalloc`s per call could be a
single preallocated scratch reused across calls.)

---

## 8. Thread divergence (ReLU & guards)

All 32 threads in a warp share one instruction pointer (SIMT execution). When a
data-dependent branch sends some lanes one way and others the other way, the
hardware must execute **both** paths with the inactive lanes masked off — the
paths are *serialized*. That is **warp divergence**, and it can halve throughput
on a 50/50 branch.

ReLU is the textbook example, and it shows up in `relu_forward` and
`relu_backward`:

```c++
out[i] = in[i] > 0.0f ? in[i] : 0.0f;          // relu_forward
grad_in[i] = pre_act[i] > 0.0f ? grad_out[i] : 0.0f;  // relu_backward
```

Whether `in[i] > 0` differs from thread to thread, so within a warp some lanes
take the "keep" branch and others the "zero" branch — divergence. Here it is
cheap because each side is a single assignment (the compiler emits a *predicated*
select rather than a real branch, so there is no serialization at all). The
lesson generalizes: *small* ternaries are fine; *large* divergent branches
(loops with data-dependent trip counts, big `if/else` bodies) are what to avoid.

The `if (i < n)` / `if (m < M && n < N)` bounds guards (§4, §5) are also
branches, but they only diverge in the **last** block along each axis (where some
lanes are in-range and some are not). Every interior block has all lanes in
range, so the guard is uniform and free there. This is why we tolerate the guard
everywhere: it costs essentially nothing except in the tail.

---

## 9. Numerical stability (softmax max-subtraction)

`softmax(z)_j = exp(z_j) / sum_k exp(z_k)`. Computed naively, `exp(z_j)`
**overflows** to `+inf` for moderately large logits (`exp(89.0f)` already exceeds
`float` range), and the resulting `inf/inf` is `NaN` that then poisons the loss
and every gradient. The fix uses the identity that softmax is invariant to adding
a constant to all logits — so subtract the row maximum `M = max_k z_k` first:

```
softmax(z)_j = exp(z_j - M) / sum_k exp(z_k - M)
```

Now the largest exponent argument is `0` (so `exp` ≤ 1, no overflow) and at least
one term in the denominator is `exp(0) = 1` (so no divide-by-zero). This is
implemented in `softmax_rows`, which uses **one thread per row** (the class count
`cols` is small), letting each thread do the full per-row reduction in registers:

```c++
__global__ void softmax_rows(const float* logits, float* probs,
                             int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;   // one thread = one row
    if (r >= rows) return;
    const float* z = logits + r * cols;              // start of this row
    float* p       = probs  + r * cols;

    float maxv = z[0];                               // 1) row max ...
    for (int c = 1; c < cols; ++c) maxv = fmaxf(maxv, z[c]);

    float sum = 0.0f;                                // 2) shifted exp + sum
    for (int c = 0; c < cols; ++c) { p[c] = expf(z[c] - maxv); sum += p[c]; }

    for (int c = 0; c < cols; ++c) p[c] /= sum;      // 3) normalize
}
```

The cross-entropy loss kernel (`cross_entropy_loss`) clamps the probability with
`max(probs[r,labels[r]], 1e-12)` before `-log(...)`, guarding against
`log(0) = -inf` for the same numerical-safety reason. And because the analytic
output-layer gradient is the clean `(probs - onehot)/batch` (derived in
`docs/math_derivation.md`), `cross_entropy_grad` never has to differentiate
through the `exp`/`log` at all — another reason the stable softmax + CE pairing
is used everywhere.

---

## 10. Synchronization (`__syncthreads`, `cudaDeviceSynchronize`)

Two completely different barriers, often confused:

### `__syncthreads()` — *device-side, within one block*

A barrier across the threads **of a single block**. No thread passes it until all
threads in the block have reached it. Used in `gemm_tiled` (§7) for two reasons:

1. After cooperatively loading a tile into `__shared__` memory, every thread must
   wait until the *whole* tile is loaded before it starts reading neighbors'
   contributions — otherwise it reads stale/garbage values (a read-before-write
   race).
2. Before the next loop iteration overwrites the tile, every thread must be done
   *using* the current tile — otherwise it's a write-before-read race.

Rules of thumb: `__syncthreads()` only synchronizes *within* a block (there is no
intra-kernel barrier across blocks), and it must be reached by **all** threads in
the block — putting it inside a divergent `if` that some lanes skip is undefined
behavior and can hang the kernel. That is why the tiled kernel does its bounds
handling by **zero-filling** loads rather than `return`ing early.

### `cudaDeviceSynchronize()` — *host-side, across the whole device*

A host call that blocks the **CPU** until all previously launched GPU work has
finished. Kernel launches are asynchronous (§11), so without this the CPU races
ahead. This repo uses it for two purposes:

- **Didactic error surfacing** inside `CUDA_CHECK_LAST()` (next section).
- **Implicitly**, via the blocking `cudaMemcpyDeviceToHost` in
  `mlp_compute_loss` / `mlp_accuracy`, which can't return correct numbers until
  the kernels that produced them have run.

In production you would *not* sprinkle `cudaDeviceSynchronize()` after every
launch — it serializes the CPU and GPU and throws away overlap. We do it here on
purpose so a mistake shows up at the exact line that caused it.

---

## 11. Asynchronous launches & error checking

A kernel launch `kernel<<<grid, block>>>(...)` is **asynchronous**: it returns to
the CPU almost immediately, having only *queued* the work on a stream. Two
consequences:

1. **Launch-configuration errors** (bad grid/block dims, too much shared memory)
   are reported by the *next* CUDA call, retrievable via `cudaGetLastError()`.
2. **Runtime errors inside the kernel** (e.g. an illegal address) only surface
   once you **synchronize** and the work actually executes.

So a robust check needs *both* a `cudaGetLastError()` and a synchronize. That is
exactly the `CUDA_CHECK_LAST()` macro in `include/common.cuh`, called by every
launcher right after the launch:

```c++
// common.cuh
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(err__));      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Check the launch result, THEN block until the kernel finishes and check that.
#define CUDA_CHECK_LAST()                  \
    do {                                   \
        CUDA_CHECK(cudaGetLastError());    \
        CUDA_CHECK(cudaDeviceSynchronize());\
    } while (0)
```

Notes for learners:

- **Why the `do { ... } while(0)`?** It makes the multi-statement macro behave
  like a single statement, so `if (x) CUDA_CHECK(...); else ...;` still parses
  correctly (no dangling-else / stray-semicolon traps).
- **Why check after *every* runtime call?** `cudaMalloc`, `cudaMemcpy`, etc. all
  return a `cudaError_t` that is silently ignored unless you inspect it. Wrapping
  in `CUDA_CHECK` turns a silent failure into a precise file:line abort.
- **The synchronize in `CUDA_CHECK_LAST()` is a deliberate performance cost.** It
  is there so this study code fails loudly and locally. Real code checks errors
  but synchronizes far less often.

---

## 12. Occupancy basics

**Occupancy** = the ratio of active warps on an SM to the hardware maximum. The
GPU hides global-memory latency by having *other* warps ready to run while one
warp waits on memory, so higher occupancy generally means better latency hiding
(up to a point — it is a means, not the goal).

What limits how many blocks fit on an SM at once:

- **Threads per block.** This repo uses 256 (`kBlockSize`, and `kTileDim*kTileDim
  = 16*16 = 256`). 256 is a sweet spot: a multiple of the 32-wide warp (no wasted
  lanes) and small enough that several blocks can co-reside on an SM.
- **Registers per thread.** Each SM has a fixed register file; the more registers
  a kernel uses, the fewer threads can be resident.
- **Shared memory per block.** `gemm_tiled` uses `2 * kTileDim * kTileDim *
  sizeof(float) = 2*16*16*4 = 2048` bytes per block. Asking for more shared
  memory means fewer blocks per SM — a real tension between the tiling speedup
  and occupancy. Larger tiles reduce global traffic but raise shared-memory and
  register pressure; `kTileDim = 16` is a reasonable, didactic middle ground.

`src/main.cu` prints `cudaGetDeviceProperties` (device name, SM count, etc.) so
you can relate these limits to your actual GPU. Tuning block/tile sizes for peak
occupancy on a specific card is left as an exercise.

---

## 13. Putting it together: how the MLP uses all of this

Mapping each kernel to the concept it demonstrates (all in `src/kernels.cu`,
orchestrated by `src/mlp.cu` per the algorithms in `include/mlp.cuh`):

| Kernel               | Launch shape            | Concepts on display                          |
|----------------------|-------------------------|----------------------------------------------|
| `gemm_naive`         | 2-D grid, 1 thr/elem    | 2-D indexing (§5), coalesced writes (§6)     |
| `gemm_tiled`         | 2-D grid, 16×16 tiles   | shared memory + tiling (§7), `__syncthreads` (§10) |
| `add_bias`           | 1-D grid over elements  | element-wise idiom (§4), broadcast read       |
| `relu_forward`       | 1-D grid over elements  | element-wise idiom (§4), divergence (§8)      |
| `relu_backward`      | 1-D grid over elements  | element-wise idiom (§4), divergence (§8)      |
| `softmax_rows`       | 1-D grid, 1 thr/row     | numerical stability (§9), per-row reduction   |
| `cross_entropy_grad` | 1-D grid over elements  | the `(probs-onehot)/batch` identity (§9)      |
| `cross_entropy_loss` | 1-D grid, 1 thr/row     | `log(0)` clamp (§9), D2H reduce on host (§2)  |
| `bias_grad`          | 1-D grid, 1 thr/column  | column reduction, small-axis-per-thread       |
| `sgd_update`         | 1-D grid over elements  | element-wise idiom (§4)                       |
| `leaky_relu_*` (0002)| 1-D grid over elements  | element-wise idiom (§4), divergence (§8)      |
| `tanh_*` (0002)      | 1-D grid over elements  | element-wise idiom (§4); backward uses post-act|
| `reduce_sum_kernel` (0002) | multi-pass tree  | parallel reduction (§7b), shared mem + `__syncthreads` |
| `predictions_correct` (0002) | 1-D grid, 1 thr/row | per-row argmax for device-side accuracy   |
| `momentum_update` / `adam_update` (0002) | 1-D grid over params | optimizer state in device buffers |

A single training step (see `src/main.cu`) exercises nearly all of it:

1. `cudaMemcpy` the batch `X` and labels **H2D** (§2) into the reused buffers.
2. **Forward** (`mlp_forward`): per layer, `launch_gemm` (§5) → `launch_add_bias`
   → `launch_relu_forward` (hidden, §8) or `launch_softmax_rows` (output, §9).
3. **Loss/accuracy** for reporting: `launch_cross_entropy_loss` then a blocking
   **D2H** copy + host reduction (§2, §10).
4. **Backward** (`mlp_backward`): `launch_cross_entropy_grad` (§9), then per layer
   `launch_gemm` for `dW` (transA) and `dA` (transB), `launch_bias_grad`, and
   `launch_relu_backward` (§8) — all transposes done index-only (§5).
5. **Update** (`mlp_sgd_step`): `launch_sgd_update` (§4) on every `W` and `b`.

Every one of those `launch_*` wrappers ends in `CUDA_CHECK_LAST()` (§11), so if
any thread-index math, bounds guard, or memory direction is wrong, the program
aborts at the exact launcher that caused it. Correctness of the backward pass is
independently confirmed by `mlp_grad_check` (finite differences vs. analytic
`dW`), which is run once before training in `src/main.cu`.

---

*Repo owner: `sora5801`. See `docs/math_derivation.md` for the math this code
implements, and `src/kernels.cu` for the kernels referenced throughout.*
