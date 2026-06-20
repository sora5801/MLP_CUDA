// ============================================================================
//  include/stream_demo.cuh                                    (added in push 0005)
// ----------------------------------------------------------------------------
//  ROLE IN THE PROJECT
//  Declares a self-contained teaching benchmark for CUDA STREAMS and
//  DOUBLE-BUFFERING — the technique of overlapping host->device (H2D) data
//  transfers with GPU compute so the two happen at the same time instead of one
//  after the other. This is the canonical "feed the GPU while it works" pattern
//  used by real training input pipelines.
//
//  The benchmark (implemented in src/stream_demo.cu) processes a sequence of
//  "batches" through a representative compute kernel three ways and times each:
//    1. SERIAL              — synchronous copy, then compute, repeat. No overlap.
//    2. PIPELINE (pageable) — double-buffered async copies from PAGEABLE host
//                             memory. Shows that pageable async does NOT overlap.
//    3. PIPELINE (pinned)   — double-buffered async copies from PINNED (page-
//                             locked) host memory across two streams. This is the
//                             one that actually overlaps copy with compute.
//  It prints the three wall-clock times + speedups and verifies all three produce
//  an identical result checksum (so the pipelining is proven correct, not just
//  fast). The three concepts it teaches — pinned memory, streams, and async
//  copies coordinated by buffer/stream rotation — are explained inline and in
//  docs/cuda_concepts.md ("CUDA streams & double-buffering").
//
//  WHY IT IS A SEPARATE DEMO (and not wired into the training loop): the blobs
//  mini-batch is tiny (a few hundred bytes), so its H2D copy time is negligible
//  and overlapping it would not help; and the didactic per-kernel
//  cudaDeviceSynchronize in our launch_* wrappers would serialize streams anyway.
//  This module uses deliberately larger, balanced copy/compute sizes so the
//  overlap is measurable, and launches its compute kernel directly on a stream
//  (no per-launch sync) — exactly what real overlap requires.
// ============================================================================
#pragma once

// Run the streams + double-buffering benchmark and print the results. Allocates
// and frees all of its own host/device memory (pinned + pageable + device
// buffers + streams + the compute kernel). Safe to call any time; touches no
// global state.
void run_stream_pipeline_demo();
