// =============================================================================
// include/common.cuh
// -----------------------------------------------------------------------------
// ROLE IN THE PROJECT
//   This is the tiny "foundation" header that every other translation unit in
//   the MLP study repo includes (directly or transitively). It centralizes two
//   things that we never want to re-type or get subtly wrong:
//     1. CUDA error checking, via the CUDA_CHECK / CUDA_CHECK_LAST macros.
//     2. A handful of compile-time constants and one host helper that describe
//        how we launch kernels (kBlockSize, kTileDim, ceil_div).
//   Because robust error handling and a consistent launch idiom are needed
//   everywhere, putting them here keeps the rest of the code (matrix.cu,
//   kernels.cu, mlp.cu, dataset.cu, main.cu) short and consistent.
//
//   Nothing in this file allocates memory or launches a kernel. It is pure
//   declarations + macros + constexpr values, so it is safe to include from
//   any number of .cu files (guarded by #pragma once below).
// =============================================================================

#pragma once  // Include exactly once per translation unit; cheaper and less
              // error-prone than the classic #ifndef/#define/#endif guard, and
              // supported by every compiler we target (nvcc + MSVC + gcc/clang).

// ---- Standard / CUDA headers (kept minimal on purpose) ----------------------
#include <cstdio>        // std::fprintf, stderr  -> used by the error macros.
#include <cstdlib>       // std::exit, EXIT_FAILURE -> abort on a CUDA error.
#include <cuda_runtime.h> // cudaError_t, cudaSuccess, cudaGetErrorString,
                          // cudaGetLastError, cudaDeviceSynchronize. This is the
                          // CUDA *runtime* API (the high-level one); we never use
                          // the lower-level driver API in this repo.

// =============================================================================
// CUDA_CHECK(call)
// -----------------------------------------------------------------------------
// WHAT IT DOES
//   Wraps a single CUDA runtime call that returns a cudaError_t (e.g.
//   cudaMalloc, cudaMemcpy, cudaDeviceSynchronize). It evaluates the call once,
//   captures the returned status, and if that status is anything other than
//   cudaSuccess it prints a diagnostic (source file, line number, and the
//   human-readable error string) to stderr and terminates the program.
//
// WHY A MACRO (and not a function)
//   We want __FILE__ and __LINE__ to expand at the *call site*, so the printed
//   location points at the offending CUDA call rather than at some helper. Only
//   a macro can capture the caller's file/line this way.
//
// WHY THE do { ... } while(0) WRAPPER
//   This is the standard idiom for a multi-statement macro. It makes the macro
//   behave like a single statement so it composes correctly with surrounding
//   C/C++ syntax, e.g.:
//       if (cond) CUDA_CHECK(cudaMalloc(...));
//       else      ...;
//   Without the do/while, the trailing ';' and the multiple statements would
//   break the if/else pairing. The `while(0)` is never taken, so there is no
//   runtime cost and no loop.
//
// WHY WE 'exit' ON ERROR (didactic choice)
//   In a teaching codebase, failing loudly and immediately is the clearest
//   behavior: a CUDA error almost always means the GPU state is now suspect, so
//   continuing would only produce confusing downstream garbage. Real production
//   code would more likely propagate the error to a caller; here we optimize for
//   a beginner being able to see exactly which call failed and why.
//
// PARAMETERS
//   call : any expression that evaluates to a cudaError_t. It is evaluated
//          EXACTLY ONCE (we stash it in `err_`), so passing a call with side
//          effects is safe.
// =============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        /* Evaluate the CUDA call once and remember its status code. The      \
         * trailing underscore avoids shadowing any user variable named err.   \
         */                                                                    \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            /* cudaGetErrorString turns the enum (e.g. cudaErrorMemoryAllocation)\
             * into a readable message like "out of memory". __FILE__/__LINE__ \
             * expand here, at the macro call site, pointing at the real bug.  \
             */                                                                \
            std::fprintf(stderr, "CUDA error %s:%d: '%s'\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err_));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// =============================================================================
// CUDA_CHECK_LAST()
// -----------------------------------------------------------------------------
// WHAT IT DOES
//   The error-checking companion you call IMMEDIATELY AFTER every kernel
//   launch. It performs two distinct checks:
//     1. CUDA_CHECK(cudaGetLastError())     -> catches *launch* errors.
//     2. CUDA_CHECK(cudaDeviceSynchronize())-> catches *execution* errors.
//
// WHY TWO SEPARATE CHECKS (the crucial CUDA lesson)
//   A kernel launch ( my_kernel<<<grid, block>>>(...) ) does NOT itself return a
//   cudaError_t you can test directly. Errors come back through two different
//   channels because kernels run ASYNCHRONOUSLY with respect to the host:
//
//     * Launch-time errors (bad launch configuration: too many threads per
//       block, an invalid grid, etc.) are reported synchronously and parked in
//       a per-thread "sticky last error" slot. cudaGetLastError() reads AND
//       clears that slot. Reading it here means the slot is clean before the
//       next launch, so a future error can't be misattributed to this one.
//
//     * Execution-time errors (an illegal memory access inside the kernel, an
//       out-of-bounds index, a misaligned access, etc.) only surface once the
//       GPU actually *runs* the kernel. But the host call returns immediately
//       after queuing the launch -- the kernel may not have started yet. The
//       error has, in effect, not "happened" from the host's point of view.
//       cudaDeviceSynchronize() blocks the host until the GPU has finished all
//       previously queued work; only then can such an asynchronous fault be
//       reported back. THIS is why synchronizing is what actually *surfaces*
//       async kernel errors: it forces the host to wait for (and observe) the
//       result of the kernel it just launched.
//
// THE PERFORMANCE TRADE-OFF (be honest about it)
//   cudaDeviceSynchronize() defeats the whole point of GPU asynchrony: it makes
//   the CPU stall on every kernel instead of overlapping CPU work / queuing
//   more kernels while the GPU is busy. We accept that cost here ON PURPOSE so
//   that, while learning, an error is reported right next to the line that
//   caused it. In real high-performance code you would NOT sync after every
//   launch -- you would check errors far more sparingly (e.g. once per
//   iteration, or only in debug builds) and let the kernels pipeline.
//
//   It is a macro (not a function) for the same reason as CUDA_CHECK: so the
//   __FILE__/__LINE__ inside the expanded CUDA_CHECK calls point at the
//   CUDA_CHECK_LAST() call site (right after your kernel launch).
// =============================================================================
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        /* (1) Did the launch itself fail (bad config)? Reads+clears the       \
         *     sticky last-error slot. */                                      \
        CUDA_CHECK(cudaGetLastError());                                        \
        /* (2) Wait for the kernel to actually run, surfacing any              \
         *     execution-time fault (e.g. out-of-bounds access). This blocking \
         *     sync is the didactic cost described above. */                   \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// =============================================================================
// LAUNCH-CONFIGURATION CONSTANTS
// -----------------------------------------------------------------------------
// These are compile-time (constexpr) so the compiler can fold them into the
// launch math and the kernels with zero runtime cost.
// =============================================================================

// Threads per block for our simple 1-D, element-wise kernels (ReLU, sgd_update,
// add_bias, etc.). 256 is a common, safe default: it is a multiple of the warp
// size (32), so no warp is partially populated; it is small enough to allow
// several blocks to be resident per Streaming Multiprocessor (good occupancy)
// yet large enough to amortize launch/scheduling overhead. A 1-D launch then
// uses grid = ceil_div(n, kBlockSize) blocks to cover n elements.
constexpr int kBlockSize = 256;

// Tile edge length for the tiled (shared-memory) GEMM kernel. Each thread block
// cooperatively loads a kTileDim x kTileDim tile of each input into shared
// memory. With kTileDim = 16, a block has 16*16 = 256 threads -- again a
// multiple of the warp size, and matching kBlockSize's thread count. The tile
// being square keeps the index math symmetric and the shared-memory footprint
// modest (two 16x16 float tiles = 2*256*4 = 2 KiB per block).
constexpr int kTileDim = 16;

// =============================================================================
// ceil_div(a, b)  ->  ceil(a / b) for non-negative integers
// -----------------------------------------------------------------------------
// WHAT / WHY
//   The canonical "how many blocks do I need to cover n elements?" idiom in
//   CUDA. If you have `a` elements and put `b` threads in each block, you need
//   ceil(a/b) blocks so that every element is owned by some thread (the final
//   block may be only partially used -- which is exactly why every kernel must
//   guard with an `if (idx < n)` bounds check).
//
//   Plain integer division `a / b` truncates toward zero and would leave the
//   leftover < b elements uncovered. Adding (b - 1) before dividing rounds up:
//       (a + b - 1) / b
//   e.g. a=1000, b=256 -> (1000+255)/256 = 1255/256 = 4 blocks (covers 1024).
//
// QUALIFIERS / SHAPE / UNITS
//   __host__ : this helper runs on the CPU only (we compute launch dimensions
//              on the host before launching). It is NOT callable from device
//              code, which is fine -- launch math is always done host-side.
//   inline   : defined in a header included by many .cu files; `inline` lets
//              each translation unit have its own copy without violating the
//              One Definition Rule at link time.
//   a, b     : counts (unitless integers). `a` is the total number of elements
//              to cover; `b` is the chosen block/tile size. b must be > 0.
//   returns  : the number of blocks/tiles (an integer count) needed to cover a.
// =============================================================================
__host__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
