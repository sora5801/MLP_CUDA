# 0002 — Optimizers, activations, a parallel reduction, and train/val split

**Date:** 2026-06-19
**Pushed by:** sora5801
**Type:** Feature expansion (builds on push 0001)

This push grows the push-0001 MLP from "plain SGD, ReLU only, host-side metrics,
train-on-everything" into a small but real training toolkit, and — importantly —
**it is the first push compiled and run on actual hardware** (an RTX 2080 SUPER,
compute capability 7.5, CUDA 13.3). The gradient check now passes *on the GPU*,
not just on paper.

---

## Summary

Four additions, each chosen to teach a distinct concept:

1. **A parallel reduction** (`reduce_sum`) — the canonical CUDA tree-reduction.
   Loss and (a new) accuracy are now reduced **on the GPU** instead of by copying
   the whole per-row vector back to the host.
2. **Stateful optimizers** — `Momentum` and `Adam` join plain `SGD` behind a
   small `Optimizer` abstraction with **per-parameter device state buffers**.
3. **Configurable activations** — `LeakyReLU` and `Tanh` join `ReLU`, selectable
   per network. Includes the subtlety that **Tanh's backward uses the *post*-
   activation** `A`, while (Leaky)ReLU's uses the *pre*-activation `Z`.
4. **Train / validation split + inference-only evaluation** (`mlp_evaluate`) to
   measure generalization on held-out data.

Plus a real-hardware fix: **CUDA 13 removed `cudaDeviceProp::clockRate`**, so the
device-info print now uses `cudaDeviceGetAttribute(cudaDevAttrClockRate)`.

---

## What changed

### New files

| File | Role | Concept taught |
|------|------|----------------|
| `include/optim.cuh` | `OptType` / `OptConfig` / `Optimizer` API + the Momentum & Adam update-kernel declarations. | Optimizer abstraction; per-parameter **state** that persists across steps. |
| `src/optim.cu` | Implements `momentum_update`, `adam_update`, and `optim_create/step/free`. | Momentum (heavy-ball) and Adam (bias-corrected adaptive steps); allocating state that mirrors the parameters. |

### Modified files

| File | Change |
|------|--------|
| `include/kernels.cuh`, `src/kernels.cu` | Added kernels **11–16**: `leaky_relu_forward/backward`, `tanh_forward/backward`, `reduce_sum_kernel` + `launch_reduce_sum` (multi-pass driver), and `predictions_correct` (per-row argmax==label → 1/0, for device-side accuracy). |
| `include/mlp.cuh`, `src/mlp.cu` | New `enum class Activation`; `Layer` gains an `activation` field; `mlp_create` takes a defaulted `hidden_act`. Forward/backward now **dispatch** on the activation. `mlp_compute_loss` reduces on the GPU. Added `mlp_accuracy_device` (predictions_correct + reduce_sum) and `mlp_evaluate` (forward-only over a split). |
| `include/dataset.cuh`, `src/dataset.cu` | Added `dataset_split` (copy the first *n_train* samples into a train set, the rest into a val set; each an owning `Dataset`). |
| `src/main.cu` | Shuffle-once → `dataset_split` (576 train / 192 val); create an `Optimizer` (Adam by default) and call `optim_step` instead of `mlp_sgd_step`; print the optimizer name; **VALIDATION** block after training; free the new resources. Config gains `kHiddenAct`, `kValSamples`, `kTrainSamples`; `kLearningRate` is now the optimizer step size (0.01 for Adam). Fixed the CUDA-13 `clockRate` removal. |

### The kernel roster is now 16

`gemm_naive`, `gemm_tiled`, `add_bias`, `relu_forward`, `relu_backward`,
`softmax_rows`, `cross_entropy_grad`, `cross_entropy_loss`, `bias_grad`,
`sgd_update` (1–10, push 0001) **+** `leaky_relu_forward`, `leaky_relu_backward`,
`tanh_forward`, `tanh_backward`, `reduce_sum_kernel`, `predictions_correct`
(11–16, this push), with `momentum_update` and `adam_update` living in `optim.cu`.

---

## Why

- **Parallel reduction** is *the* foundational CUDA pattern beyond element-wise
  maps. Summing n numbers looks inherently serial, but a tree does it in
  `log2(n)` parallel steps using shared memory and `__syncthreads()`. Push 0001
  dodged it by copying the per-row loss vector to the host and summing in a C++
  loop; now we keep the data on the GPU and read back **one float**. See the new
  "Parallel reduction" section in `docs/cuda_concepts.md`.
- **Optimizers** introduce the idea of **optimizer state**: SGD is memoryless,
  but Momentum keeps a velocity and Adam keeps first/second moment estimates —
  buffers shaped exactly like the parameters, allocated once and updated in place
  every step. This is a clean lesson in managing persistent device memory and in
  the math of modern optimizers (the derivation is in `docs/math_derivation.md`).
- **Activations** show that backprop needs the *right cached tensor*: ReLU and
  LeakyReLU gate on the sign of the pre-activation `Z`, but Tanh's derivative
  `1 - a²` is naturally written with the post-activation `a = tanh(z)`. Caching
  both `Z` and `A` per layer (already done in push 0001) is what makes this clean.
- **Train/val split** turns "did it memorize?" into a measurable number. The demo
  now reports held-out validation loss/accuracy, and `mlp_evaluate` models the
  inference path (forward only, no backward, no update).

---

## Notes / gotchas

- **Tanh backward reads `A`, not `Z`.** This is the one place a copy-paste of the
  ReLU backward would silently produce wrong gradients. The dispatch in
  `mlp_backward` passes `prevL.A.data` for Tanh and `prevL.Z.data` for the ReLU
  family. The gradient check is exactly the tool that would catch a mistake here.
- **`reduce_sum_kernel` requires a power-of-two block size.** The tree halves the
  active width each step (`s = blockDim/2, /4, … 1`), which only covers the block
  cleanly when `blockDim` is a power of two. `kBlockSize = 256` satisfies this.
  Each thread also adds two elements at load time ("first add during load").
- **`launch_reduce_sum` is multi-pass.** A single kernel cannot sum across blocks
  (blocks share neither memory nor a barrier), so the wrapper ping-pongs two
  scratch buffers, reducing n → #blocks → … → 1, then copies the final scalar to
  the host. Two `cudaMalloc`s per call (fine for our infrequent metric use; a real
  trainer would reuse a preallocated scratch — noted as an exercise).
- **Adam bias correction is computed on the host.** The denominators `1 - β₁ᵗ`
  and `1 - β₂ᵗ` are the same for every element on a given step, so `optim_step`
  computes them once (in `double`) and passes them into the kernel — keeping the
  kernel free of `pow`/branching.
- **He init is ReLU-flavored.** `mlp_create` still uses He initialization
  `N(0, √(2/in))` for all activations. That is principled for the ReLU family; for
  Tanh, Xavier/Glorot init would be more standard. Tanh still trains fine here, so
  we kept a single init path (switching to per-activation init is an exercise).
- **Learning-rate scales differ by optimizer.** Adam's good default lr (~1e-2)
  is much smaller than SGD's (~1e-1) because Adam normalizes each step by the
  gradient's running RMS. `main.cu` documents this where you pick the optimizer.

---

## Verification (run on the RTX 2080 SUPER, CUDA 13.3, sm_75)

The gradient check (central finite differences vs. analytic `dW`) passed for
**all three activations**, confirming every new backward kernel — including the
Tanh-uses-`A` path — is correct:

| Hidden activation | Optimizer | Grad-check max rel. err | Final train acc | Held-out val acc |
|-------------------|-----------|-------------------------|-----------------|------------------|
| ReLU (default)    | Adam      | 3.5e-5  → PASS          | 1.000           | 1.000            |
| Tanh              | Momentum  | 1.2e-4  → PASS          | 1.000           | 1.000            |
| LeakyReLU         | SGD       | 7.9e-5  → PASS          | 1.000           | 1.000            |

Adam reached loss ≈1e-4 by epoch 10 (vs. plain SGD's ≈2e-3 in push 0001 at the
same point), illustrating faster convergence. The tiled-vs-naive GEMM
microbenchmark measured a **~2.5–3.6× speedup** for the shared-memory tiled
kernel on a 512³ multiply.

---

## Build / run

Unchanged commands (the Makefile and CMakeLists glob `src/*.cu`, so the new
`optim.cu` is picked up automatically). A CUDA toolkit + NVIDIA GPU are required.

```sh
# Linux / WSL
make ARCH=sm_75 run          # sm_75 = Turing (RTX 20-series); set yours

# Windows (MSVC + CUDA) via CMake
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release
build\Release\mlp.exe
```

To study a different configuration, edit the constants at the top of
`src/main.cu`: `kHiddenAct` (`Activation::ReLU` / `LeakyReLU` / `Tanh`) and the
`optim_create(net, …)` call (`opt_sgd` / `opt_momentum` / `opt_adam`). The
gradient check validates whichever activation you choose.

> Note for CUDA ≥ 13 users: this push replaced the removed
> `cudaDeviceProp::clockRate` with `cudaDeviceGetAttribute(cudaDevAttrClockRate)`.
> If you are on an older toolkit, both work; the attribute query is portable.
