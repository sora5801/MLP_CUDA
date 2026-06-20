# 0005 — CUDA streams + double-buffering

**Date:** 2026-06-20
**Pushed by:** sora5801
**Type:** Feature / lesson (a self-contained performance demo)

This push adds a focused, runnable lesson in **overlapping host→device (H2D)
transfers with GPU compute** using **CUDA streams** and **double-buffering** — the
"keep the GPU fed while it works" pattern at the heart of real training input
pipelines. It is verified on the RTX 2080 SUPER.

---

## Summary

A new module, `stream_demo.cu`/`.cuh`, runs the same sequence of "batches" through a
representative compute kernel **three ways** and times each:

1. **Serial** — synchronous `cudaMemcpy` then compute, one batch at a time. No
   overlap (total ≈ Σ copy + Σ compute).
2. **Pipeline, pageable host** — double-buffered async copies from ordinary
   (pageable) memory. Shows that async-from-pageable overlaps only partially.
3. **Pipeline, pinned host** — double-buffered async copies from **pinned
   (page-locked)** memory across **two streams**. This is the real overlap.

It prints the three wall-clock times + speedups and verifies all three produce an
**identical result checksum** (so the pipelining is proven correct, not just fast).
The demo is called from both `src/main.cu` and the Visual Studio showcase
`demo/demo.cu`.

---

## The four ingredients of overlap (all in the code, all commented)

| Ingredient | API | Why it matters |
|-----------|-----|----------------|
| **Pinned host memory** | `cudaMallocHost` / `cudaFreeHost` | The copy engine can DMA directly to/from page-locked RAM. Pageable memory must be staged through a hidden pinned buffer, so a `cudaMemcpyAsync` from pageable behaves ~synchronously. Pinned is the prerequisite for true async transfer. |
| **Streams** | `cudaStreamCreate` | A stream is an ordered work queue. Same stream → ordered; different streams → may run **concurrently**. Two streams let a copy in one overlap a kernel in the other. |
| **Async copies** | `cudaMemcpyAsync(..., stream)` | Non-blocking H2D issued on a stream's timeline. |
| **Double-buffering** | two device buffers + two stream "lanes" | While lane A computes on buffer A, lane B copies the next batch into buffer B. Same-stream ordering guarantees a buffer is never overwritten while still being read, so **no extra synchronization is needed**. |

---

## What changed

| File | Change |
|------|--------|
| `include/stream_demo.cuh`, `src/stream_demo.cu` | **New.** `pipeline_compute` (a tunable, dependent-FMA compute kernel launched directly on a stream, no per-launch sync) + `run_stream_pipeline_demo()` (the serial / pageable / pinned benchmark with checksum verification). |
| `src/main.cu` | Calls `run_stream_pipeline_demo()` after the RNG self-test. |
| `demo/demo.cu` | Adds the streams section to the guided showcase. |
| `MLP_CUDA.vcxproj`, `.filters` | Compile `src/stream_demo.cu` in the VS project. |

---

## Two implementation details worth studying

- **Timing a multi-stream pipeline with host wall-clock.** `cudaEvent` timing lives
  on a *single* stream's timeline, so a `stop` event recorded on the default stream
  fires as soon as it is reached there — it does **not** wait for work on other
  streams. The correct, simple way to time a pipeline spread across streams is host
  wall-clock (`std::chrono`) around the issue loop plus one final
  `cudaDeviceSynchronize()`. The demo does exactly that and explains why.
- **Why no events are needed for buffer safety.** Even batches use lane 0 (buffer 0,
  stream 0); odd batches use lane 1. Within a lane, `copy_i → compute_i → next copy
  that reuses the same buffer` are all in the *same* stream, so they are ordered
  automatically — the next copy can't clobber a buffer the previous compute is still
  reading. Across lanes the buffers are different, so there is no conflict. The
  rotation *is* the synchronization.

---

## Verification (RTX 2080 SUPER, CUDA 13.3, sm_75)

48 batches of 1024×1024 floats (4 MB H2D each, 192 MB total), representative compute:

| Variant | Time | Speedup vs serial |
|---------|------|-------------------|
| serial (sync, no overlap) | ~51 ms | 1.00× |
| pipeline, pageable host | ~38 ms | ~1.3× |
| pipeline, **pinned** host | ~33 ms | **~1.5×** |

Result checksum **matches across all three** (`2.516e+07`), proving the
double-buffered pipeline computes exactly the same thing as the serial path. Exact
times vary a little run-to-run with system load, but the ordering
**pinned < pageable < serial** is consistent: pinned memory lets the copy DMA
concurrently with compute, so it beats both the serial baseline and the pageable
pipeline (whose async copies still stage through the driver and overlap less).

---

## Notes / gotchas

- **It is a separate demo, not wired into the training loop — on purpose.** The
  blobs mini-batch is a few hundred bytes, so its H2D copy time is negligible and
  overlapping it would not help; and the didactic per-kernel
  `cudaDeviceSynchronize()` in our `launch_*` wrappers would serialize streams
  anyway. This module uses deliberately larger, balanced copy/compute sizes and
  launches its kernel directly on a stream (no per-launch sync) so the overlap is
  real and measurable. Applying streams to the real training pipeline would mean
  giving the launchers an optional stream parameter and dropping the per-launch
  sync — a good follow-up exercise.
- **`cudaMallocHost` is a limited resource.** Pinned memory can't be paged out, so
  over-allocating it starves the OS. The demo frees it with `cudaFreeHost`.
- **Tuning.** `rows`, `feats`, `N`, and `iters` at the top of
  `run_stream_pipeline_demo` set the copy/compute balance. Overlap is most visible
  when per-batch copy time ≈ compute time; if compute dominates, the copy hides
  "for free" and the speedup shrinks toward 1×.

---

## Build / run (unchanged)

```sh
# Linux / WSL:        make ARCH=sm_75 run
# Windows CMake:      cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release
# Visual Studio 2026: open MLP_CUDA.sln, Release/x64, Ctrl+F5
```
