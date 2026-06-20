# 0006 — Fused GEMM + bias + activation kernel

**Date:** 2026-06-20
**Pushed by:** sora5801
**Type:** Feature / optimization (kernel fusion) — used by the real forward pass

This push adds a **fused** forward-layer kernel, `gemm_bias_act`, that computes the
matmul, the bias add, and the activation in **one** kernel instead of three, and
switches `mlp_forward` to use it. A benchmark demonstrates the speedup and verifies
the fused result is identical to the unfused one. Verified on the RTX 2080 SUPER.

---

## The idea: do the epilogue in a register

The unfused forward ran three kernels per hidden layer:

```
gemm_tiled    : Z = prev · W           (writes Z to global memory)
add_bias      : Z = Z + b              (reads Z, writes Z)
relu_forward  : A = relu(Z)            (reads Z, writes A)
```

Z is bounced through global memory twice just to add a bias and apply an
activation — both of which are essentially free arithmetic. The **fused** kernel
keeps the matmul's output in a register and finishes the job right there:

```
gemm_bias_act : acc = Σ prev·W  (tiled, in register)
                z   = acc + bias[n]    -> write Z   (for backprop)
                a   = act(z)           -> write A
```

One launch, and Z is written once (not written-then-reread-twice). The matmul half
is byte-for-byte the same shared-memory tiling as `gemm_tiled`; only the per-thread
**epilogue** differs — so the benchmark isolates exactly what fusion saves.

---

## What changed

| File | Change |
|------|--------|
| `include/kernels.cuh`, `src/kernels.cu` | **New kernel #20** `gemm_bias_act` + `launch_gemm_bias_act`: tiled matmul with a fused bias + activation epilogue, writing both Z and A. `act_type`: 0=ReLU, 1=LeakyReLU, 2=Tanh, 3=Identity. |
| `src/mlp.cu` | `mlp_forward` now calls `launch_gemm_bias_act` (replacing the `gemm` + `add_bias` + activation trio). The output layer uses the identity epilogue, then `softmax_rows` over Z. A `static_assert` guards that `Activation`'s values still match the kernel's `act_type` codes. |
| `include/fusion_demo.cuh`, `src/fusion_demo.cu` | **New.** `run_fusion_benchmark()` times unfused vs fused on a representative layer and checks they agree. |
| `src/main.cu`, `demo/demo.cu` | Call `run_fusion_benchmark()`. |
| `MLP_CUDA.vcxproj`, `.filters` | Compile `src/fusion_demo.cu` in the VS project. |

The kernel roster is now **20 in `kernels.cu`** (+ 2 optimizer kernels, + the
`pipeline_compute` / `fusion` demo kernels). The unfused kernels (`add_bias`,
`relu_forward`, …) remain in the library — they're the reference, and the benchmark
exercises them as the "before" path.

---

## Correctness

The fused kernel writes the same `Z` (pre-activation, for backprop) and `A`
(post-activation) the unfused path did, so **nothing downstream changes** — the
backward pass and dropout are untouched. The finite-difference **gradient check
still passes** with the fused forward (max rel. err **1.258e-4 → PASS**, identical
to before), and the benchmark confirms the fused and unfused outputs match to the
last bit (same tiled accumulation): checksum `1.4433e+06` for both.

---

## Verification (RTX 2080 SUPER, CUDA 13.3, sm_75)

Benchmarked a wide, shallow layer `Z = A[4096,64]·W[64,4096] + b ; A = relu(Z)`,
200 iterations, kernels launched directly (no per-launch sync), CUDA-event timed:

| Path | Time / iter | |
|------|------------|---|
| unfused (gemm + add_bias + relu) | ~2.4 ms | |
| **fused** (gemm_bias_act) | ~1.75 ms | **~1.4× faster** |

Result checksum identical. The win is the two eliminated global-memory passes over
the [M,N] output and the two saved kernel launches.

---

## Notes / gotchas

- **The win depends on the matmul/epilogue ratio.** For a *compute-bound* deep
  matmul (large K), the bias+activation epilogue is a tiny slice of the runtime and
  fusion saves proportionally little (≈1.05× for 1024³). The benefit is largest when
  the output [M,N] is large relative to the contraction K (the epilogue is
  memory-bound over [M,N]) or for **small** layers where the saved *launch overhead*
  dominates — which is exactly the regime the toy MLP's tiny layers live in. The
  benchmark deliberately uses a wide, shallow layer to make the effect visible.
- **`act_type` encoding is asserted.** `mlp.cu` passes `static_cast<int>(L.activation)`
  to the kernel, so a `static_assert` pins `ReLU/LeakyReLU/Tanh = 0/1/2`. Add new
  activations to both the enum and the kernel's `switch`.
- **Softmax can't be fused element-by-element.** It needs the whole row (max + sum),
  so the output layer fuses only GEMM+bias (identity epilogue) and runs
  `softmax_rows` over Z afterward.
- **Unsigned-underflow bug fixed in the benchmark's data fill.** `(i % 17) - 8` with
  `i` a `size_t` underflows to a huge value when `i%17 < 8` (and overflowed the
  matmul to `inf`). The fix — `(int)(i % 17) - 8` — is a good reminder to cast to
  signed *before* subtracting from an unsigned index. (The pre-existing GEMM-timing
  benchmarks have the same pattern but never read their results, so it was harmless
  there; this one checksums its output, which surfaced it.)

---

## Build / run (unchanged)

```sh
# Linux / WSL:        make ARCH=sm_75 run
# Windows CMake:      cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release
# Visual Studio 2026: open MLP_CUDA.sln, Release/x64, Ctrl+F5
```
