// ============================================================================
//  include/fusion_demo.cuh                                    (added in push 0006)
// ----------------------------------------------------------------------------
//  ROLE IN THE PROJECT
//  Declares a small benchmark that demonstrates KERNEL FUSION: it runs a
//  representative forward linear layer TWO ways and times each —
//    * UNFUSED: gemm_tiled  ->  add_bias  ->  relu_forward   (three kernels)
//    * FUSED:   gemm_bias_act                                (one kernel)
//  — and verifies the two produce an identical result. Both use the SAME tiled
//  matmul, so the only difference being measured is the fusion of the bias-add
//  and the activation into the matmul's epilogue (vs. two extra full passes over
//  the [M,N] output through global memory).
//
//  This isolates the fusion benefit: the fused kernel avoids writing the
//  pre-activation Z to global memory and reading it back twice, and saves two
//  kernel launches. The benchmark launches the kernels DIRECTLY on the default
//  stream (not through the didactic launch_* wrappers, which synchronize after
//  every launch) and times many iterations with CUDA events, so the numbers
//  reflect real kernel time rather than per-launch sync overhead.
//
//  mlp_forward itself already uses the fused kernel (push 0006); this benchmark
//  exists to make the "why fuse?" measurable, and is called from src/main.cu and
//  the Visual Studio showcase demo/demo.cu.
// ============================================================================
#pragma once

// Run the fused-vs-unfused forward-layer benchmark and print the result. Allocates
// and frees its own device/host memory. Touches no global state.
void run_fusion_benchmark();
