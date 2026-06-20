// ============================================================================
//  src/fusion_demo.cu                                         (added in push 0006)
// ----------------------------------------------------------------------------
//  Implements the kernel-fusion benchmark declared in include/fusion_demo.cuh.
//  It compares the UNFUSED forward linear layer (three kernels: gemm_tiled ->
//  add_bias -> relu_forward) against the FUSED one (gemm_bias_act) on a large,
//  representative layer, and checks that they compute the same thing.
//
//  WHY FUSION IS FASTER (the lesson): the bias add and the activation are
//  memory-bound — trivial arithmetic, but each separate kernel has to stream the
//  whole [M,N] output through global memory (read it, tweak it, write it back).
//  Folding them into the matmul's epilogue, while the value is still in a
//  register, removes those two extra round-trips and two kernel launches. The
//  matmul itself is identical (both paths use the same tiled GEMM), so the gap we
//  measure is exactly the cost fusion eliminates.
//
//  TIMING METHOD: we launch the kernels DIRECTLY on the default stream (no
//  per-launch cudaDeviceSynchronize, unlike the launch_* wrappers) and wrap many
//  iterations in CUDA events. All work is on one stream, so cudaEvent timing is
//  valid here (contrast with the multi-stream demo in stream_demo.cu, which must
//  use host wall-clock).
// ============================================================================

#include "fusion_demo.cuh"
#include "kernels.cuh"   // gemm_tiled / add_bias / relu_forward / gemm_bias_act + reduce_sum
#include "common.cuh"    // CUDA_CHECK / CUDA_CHECK_LAST, kBlockSize, kTileDim, ceil_div

#include <cstdio>
#include <vector>

void run_fusion_benchmark() {
    // A wide, shallow layer: large output [M,N] but a modest contraction K. The
    // fused-away work (bias + activation) is element-wise over [M,N], while the
    // matmul cost scales with M*N*K. Keeping K modest makes the epilogue a
    // meaningful fraction of the total, so the fusion win is visible. (For a very
    // deep/compute-bound matmul the epilogue is a tiny slice and fusion saves
    // proportionally less — an honest caveat noted in the changelog.)
    const int M = 4096;   // batch rows
    const int K = 64;     // input features (contraction dim)
    const int N = 4096;   // output features
    const int R = 200;    // timed iterations (averaged)
    const int act_relu = 0;          // gemm_bias_act act_type for ReLU
    const float alpha  = 0.01f;      // unused by ReLU, passed for signature parity

    printf("============== KERNEL FUSION =====================\n");
    printf("  forward layer  Z = A[%d,%d]*W[%d,%d] + b ; A = relu(Z)\n",
           M, K, K, N);

    // ---- Allocate device buffers ----
    const size_t elemsIn = (size_t)M * K, elemsW = (size_t)K * N, elemsO = (size_t)M * N;
    float *Ain = nullptr, *W = nullptr, *bias = nullptr, *Z = nullptr;
    float *A_unfused = nullptr, *A_fused = nullptr;
    CUDA_CHECK(cudaMalloc(&Ain,  elemsIn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W,    elemsW  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Z,    elemsO  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_unfused, elemsO * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&A_fused,   elemsO * sizeof(float)));

    // ---- Fill inputs with cheap deterministic host data, copy H2D ----
    // NOTE the (int) casts: i is size_t (unsigned), so "(i % 17) - 8" would
    // underflow to a huge value when i%17 < 8. Cast to signed before subtracting.
    std::vector<float> hA(elemsIn), hW(elemsW), hB((size_t)N);
    for (size_t i = 0; i < elemsIn; ++i) hA[i] = (float)((int)(i % 17) - 8) * 0.01f;
    for (size_t i = 0; i < elemsW;  ++i) hW[i] = (float)((int)(i % 11) - 5) * 0.01f;
    for (size_t i = 0; i < (size_t)N; ++i) hB[i] = (float)((int)(i % 7) - 3) * 0.1f;
    CUDA_CHECK(cudaMemcpy(Ain,  hA.data(), elemsIn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W,    hW.data(), elemsW  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bias, hB.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Launch configurations (computed once) ----
    dim3 block2d(kTileDim, kTileDim);                            // tiled GEMM block
    dim3 grid2d(ceil_div(N, kTileDim), ceil_div(M, kTileDim));  // covers [M,N]
    int  block1d = kBlockSize;                                   // element-wise block
    int  grid1d  = ceil_div((int)elemsO, kBlockSize);           // covers M*N elements

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    // ---- Warm up both paths (first launches pay one-time costs) ----
    gemm_tiled<<<grid2d, block2d>>>(Ain, W, Z, M, N, K);
    add_bias<<<grid1d, block1d>>>(Z, bias, M, N);
    relu_forward<<<grid1d, block1d>>>(Z, A_unfused, (int)elemsO);
    gemm_bias_act<<<grid2d, block2d>>>(Ain, W, bias, Z, A_fused, M, N, K, act_relu, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Time the UNFUSED path: three kernels per iteration ----
    CUDA_CHECK(cudaEventRecord(s));
    for (int r = 0; r < R; ++r) {
        gemm_tiled<<<grid2d, block2d>>>(Ain, W, Z, M, N, K);           // matmul
        add_bias<<<grid1d, block1d>>>(Z, bias, M, N);                  // + bias (extra pass)
        relu_forward<<<grid1d, block1d>>>(Z, A_unfused, (int)elemsO);  // + relu (extra pass)
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms_unfused = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_unfused, s, e));

    // ---- Time the FUSED path: one kernel per iteration ----
    CUDA_CHECK(cudaEventRecord(s));
    for (int r = 0; r < R; ++r) {
        gemm_bias_act<<<grid2d, block2d>>>(Ain, W, bias, Z, A_fused, M, N, K,
                                           act_relu, alpha);
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms_fused = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_fused, s, e));

    // ---- Correctness: both paths must compute the same A ----
    float sum_unfused = launch_reduce_sum(A_unfused, (int)elemsO);
    float sum_fused   = launch_reduce_sum(A_fused,   (int)elemsO);

    printf("  unfused (gemm + add_bias + relu): %7.3f ms/iter\n", ms_unfused / R);
    printf("  fused   (gemm_bias_act)         : %7.3f ms/iter  (%.2fx faster)\n",
           ms_fused / R, ms_unfused / ms_fused);
    printf("  result %s (unfused=%.4e fused=%.4e)\n",
           (sum_unfused == sum_fused) ? "MATCHES" : "MISMATCH!",
           sum_unfused, sum_fused);
    printf("  -> fusing the bias-add and activation into the matmul epilogue\n");
    printf("     removes two extra global-memory passes over Z and two launches.\n");
    printf("==================================================\n\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaFree(Ain));
    CUDA_CHECK(cudaFree(W));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(Z));
    CUDA_CHECK(cudaFree(A_unfused));
    CUDA_CHECK(cudaFree(A_fused));
}
