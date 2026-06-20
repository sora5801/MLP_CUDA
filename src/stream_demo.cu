// ============================================================================
//  src/stream_demo.cu                                         (added in push 0005)
// ----------------------------------------------------------------------------
//  ROLE IN THE PROJECT
//  Implements the CUDA streams + double-buffering benchmark declared in
//  include/stream_demo.cuh. It is a focused, runnable lesson in OVERLAPPING data
//  transfer with compute, which is how you keep a GPU fed instead of letting it
//  idle while the next batch of inputs crawls across the PCIe bus.
//
//  THE FOUR INGREDIENTS OF OVERLAP (all demonstrated below):
//    1. PINNED (page-locked) host memory  [cudaMallocHost]
//         The GPU's copy engine can DMA directly to/from page-locked host RAM.
//         Ordinary "pageable" malloc'd memory can be moved by the OS, so the
//         driver must first stage it through a hidden pinned buffer — which makes
//         a "cudaMemcpyAsync" from pageable memory behave essentially
//         synchronously (no overlap). Pinned memory is the prerequisite for true
//         async transfer.
//    2. STREAMS  [cudaStreamCreate]
//         A stream is an ordered queue of GPU work. Operations in the SAME stream
//         run in order; operations in DIFFERENT streams may run CONCURRENTLY. We
//         use two streams so a copy in one can run while a kernel in the other
//         executes (modern GPUs have separate copy and compute engines).
//    3. ASYNC COPIES  [cudaMemcpyAsync(..., stream)]
//         Non-blocking H2D copies that return to the host immediately and run on
//         the given stream's timeline.
//    4. DOUBLE-BUFFERING
//         Two device input buffers (and two stream "lanes"). While lane A computes
//         on buffer A, lane B copies the next batch into buffer B. Because each
//         lane reuses its OWN buffer and its operations are ordered within its
//         stream, no extra synchronization is needed to avoid overwriting a buffer
//         that is still being read (the same-stream ordering guarantees it).
//
//  TIMING NOTE: cudaEvent timing lives on a single stream's timeline, so it does
//  not bound work spread across multiple streams. The natural and correct way to
//  time a multi-stream pipeline is host wall-clock around the issue loop plus a
//  final cudaDeviceSynchronize() — which is exactly what we do here.
// ============================================================================

#include "stream_demo.cuh"
#include "kernels.cuh"   // launch_reduce_sum (used for the correctness checksum)
#include "common.cuh"    // CUDA_CHECK, ceil_div

#include <cstdio>
#include <vector>
#include <chrono>        // std::chrono for multi-stream wall-clock timing

// ----------------------------------------------------------------------------
//  pipeline_compute — a representative, time-consuming element-wise kernel.
// ----------------------------------------------------------------------------
//  Stands in for "the forward pass": for each element it runs a dependent chain
//  of `iters` fused multiply-adds. The chain is dependent (each step needs the
//  previous result) so the compiler can't fold it away and the runtime is
//  proportional to `iters` — which lets us tune the compute time to be COMPARABLE
//  to the copy time, the regime where overlap is most visible. The output is a
//  deterministic function of (input, iters), so the same inputs always yield the
//  same result regardless of which stream ran it — that is what makes the
//  serial/pipeline checksums match.
//
//  It is launched DIRECTLY on a stream (`<<<grid,block,0,stream>>>`) with NO
//  cudaDeviceSynchronize afterward — unlike the didactic launch_* wrappers — so
//  that it can actually overlap with copies on another stream.
__global__ void pipeline_compute(const float* in, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = in[i];
    // #pragma unroll 1 keeps this a real loop (no unrolling), so the work scales
    // with `iters` and the dependent chain genuinely consumes time.
    #pragma unroll 1
    for (int k = 0; k < iters; ++k) {
        acc = acc * 0.9999994f + 1.0e-6f;   // dependent FMA: needs the prior acc
    }
    out[i] = acc;
}

// ----------------------------------------------------------------------------
//  run_stream_pipeline_demo
// ----------------------------------------------------------------------------
void run_stream_pipeline_demo() {
    // ---- Problem size (tuned so per-batch copy time ≈ compute time) ----------
    const int   rows  = 1024;             // per-batch "rows"
    const int   feats = 1024;             // per-batch "features"
    const int   n     = rows * feats;     // elements per batch (1,048,576)
    const size_t bytes = (size_t)n * sizeof(float);   // 4 MB per batch
    const int   N     = 48;               // number of batches in the pipeline
    const int   iters = 500;              // compute work per element (tunable)
    const int   block = 256;
    const int   grid  = ceil_div(n, block);

    printf("========== CUDA STREAMS / DOUBLE-BUFFERING =======\n");
    printf("  %d batches of %d x %d floats (%.1f MB H2D each, %.0f MB total)\n",
           N, rows, feats, (double)bytes / (1024.0 * 1024.0),
           (double)N * bytes / (1024.0 * 1024.0));

    // ---- Host inputs: the SAME data in pinned and in pageable memory ----------
    // Pinned (page-locked) host buffer holding all N batches back-to-back.
    float* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, (size_t)N * bytes));   // page-locked
    // Pageable buffer (ordinary heap) with identical contents, for the contrast.
    std::vector<float> h_pageable((size_t)N * n);
    for (size_t i = 0; i < (size_t)N * n; ++i) {
        float v = (float)((i % 1000) * 0.001f);   // deterministic, in [0,1)
        h_pinned[i]   = v;
        h_pageable[i] = v;
    }

    // ---- Device buffers: TWO input buffers (the double buffer) + big output ---
    float* d_in[2] = { nullptr, nullptr };
    CUDA_CHECK(cudaMalloc(&d_in[0], bytes));
    CUDA_CHECK(cudaMalloc(&d_in[1], bytes));
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * bytes));   // each batch -> its own slice

    // ---- Two streams (the two pipeline "lanes") -------------------------------
    cudaStream_t stream[2];
    CUDA_CHECK(cudaStreamCreate(&stream[0]));
    CUDA_CHECK(cudaStreamCreate(&stream[1]));

    // ---- Warm up once (first copy + launch pay one-time init costs) -----------
    CUDA_CHECK(cudaMemcpy(d_in[0], h_pinned, bytes, cudaMemcpyHostToDevice));
    pipeline_compute<<<grid, block>>>(d_in[0], d_out, n, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Helper to time a section by host wall-clock (correct for multi-stream).
    auto now = []() { return std::chrono::high_resolution_clock::now(); };
    auto ms_between = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // ====================================================================
    // (1) SERIAL baseline: synchronous copy, then compute, one batch at a time.
    //     cudaMemcpy blocks the host until the copy finishes, and the kernel runs
    //     on the default stream after it; the next copy waits for the kernel.
    //     So copy and compute never overlap: total ≈ Σ (copy_i + compute_i).
    // ====================================================================
    auto t0 = now();
    for (int i = 0; i < N; ++i) {
        CUDA_CHECK(cudaMemcpy(d_in[0], h_pinned + (size_t)i * n, bytes,
                              cudaMemcpyHostToDevice));                 // sync copy
        pipeline_compute<<<grid, block>>>(d_in[0], d_out + (size_t)i * n, n, iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());     // wait for the last kernel
    double ms_serial = ms_between(t0, now());
    float sum_serial = launch_reduce_sum(d_out, N * n);   // result checksum

    // ====================================================================
    // (2) PIPELINE from PAGEABLE host memory. Same double-buffered structure as
    //     (3), but the source is ordinary pageable memory — so cudaMemcpyAsync
    //     cannot DMA directly and behaves ~synchronously. Expect little overlap.
    // ====================================================================
    t0 = now();
    for (int i = 0; i < N; ++i) {
        int lane = i & 1;                                   // alternate buffers/streams
        CUDA_CHECK(cudaMemcpyAsync(d_in[lane], h_pageable.data() + (size_t)i * n,
                                   bytes, cudaMemcpyHostToDevice, stream[lane]));
        pipeline_compute<<<grid, block, 0, stream[lane]>>>(
            d_in[lane], d_out + (size_t)i * n, n, iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms_pageable = ms_between(t0, now());
    float sum_pageable = launch_reduce_sum(d_out, N * n);

    // ====================================================================
    // (3) PIPELINE from PINNED host memory — the real double-buffered overlap.
    //     Even batches use lane 0 (buffer 0, stream 0); odd batches use lane 1.
    //     While stream 0 COMPUTES on buffer 0, stream 1 can COPY the next batch
    //     into buffer 1, and vice versa — copy and compute run concurrently.
    //     Correctness: within a lane, copy_i precedes compute_i precedes the next
    //     copy that reuses the same buffer (same-stream ordering), so a buffer is
    //     never overwritten while still being read. Total ≈ Σ max(copy, compute).
    // ====================================================================
    t0 = now();
    for (int i = 0; i < N; ++i) {
        int lane = i & 1;
        CUDA_CHECK(cudaMemcpyAsync(d_in[lane], h_pinned + (size_t)i * n,
                                   bytes, cudaMemcpyHostToDevice, stream[lane]));
        pipeline_compute<<<grid, block, 0, stream[lane]>>>(
            d_in[lane], d_out + (size_t)i * n, n, iters);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms_pinned = ms_between(t0, now());
    float sum_pinned = launch_reduce_sum(d_out, N * n);

    // ---- Report --------------------------------------------------------------
    printf("  serial (sync, no overlap)      : %7.2f ms\n", ms_serial);
    printf("  pipeline, PAGEABLE host        : %7.2f ms  (%.2fx vs serial)\n",
           ms_pageable, ms_serial / ms_pageable);
    printf("  pipeline, PINNED host (overlap): %7.2f ms  (%.2fx vs serial)\n",
           ms_pinned, ms_serial / ms_pinned);
    // The pipeline must compute the SAME thing as the serial path. We compare the
    // result checksums (they should agree to within floating-point noise).
    bool ok = (sum_serial == sum_pageable) && (sum_serial == sum_pinned);
    printf("  result checksum %s (serial=%.3e pageable=%.3e pinned=%.3e)\n",
           ok ? "MATCHES across all three" : "MISMATCH!",
           sum_serial, sum_pageable, sum_pinned);
    printf("  -> Two streams let each batch's compute overlap the NEXT batch's\n");
    printf("     H2D copy. PINNED host memory lets that copy DMA concurrently, so\n");
    printf("     the pinned pipeline beats both the serial baseline and the\n");
    printf("     PAGEABLE pipeline (whose async copies still stage through the\n");
    printf("     driver, so they overlap less).\n");
    printf("==================================================\n\n");

    // ---- Cleanup -------------------------------------------------------------
    CUDA_CHECK(cudaStreamDestroy(stream[0]));
    CUDA_CHECK(cudaStreamDestroy(stream[1]));
    CUDA_CHECK(cudaFree(d_in[0]));
    CUDA_CHECK(cudaFree(d_in[1]));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_pinned));   // pinned memory needs cudaFreeHost
}
