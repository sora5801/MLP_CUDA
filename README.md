# MLP_CUDA

> A heavily-commented, didactic **Multi-Layer Perceptron (MLP)** implemented from
> scratch in **CUDA C++**, with both the **forward** and **backward** passes
> written as hand-rolled kernels (no cuBLAS, cuDNN, Thrust, or any external deps).
> The point is not speed — it is to *learn CUDA* by building a real, trainable
> neural net and seeing every byte move between host and device.

This repo trains a small classifier on a synthetic Gaussian-"blobs" dataset,
verifies its own gradients with a finite-difference check, and includes a
naive-vs-tiled matrix-multiply microbenchmark so you can *measure* why shared
memory matters. Every `.cu`/`.cuh` file is commented like a tutorial: shapes,
units, memory layout, and the thread-index math inside each kernel.

---

## What is an MLP?

A Multi-Layer Perceptron is the simplest "deep" neural network: a stack of
**fully-connected (dense) layers**, each computing an affine transform followed
by a non-linearity.

For one layer with `in` input features and `out` output features, given a batch
of activations `A_prev` (shape `[batch, in]`):

```
Z = A_prev · W + b      // affine:  Z is [batch, out]
A = act(Z)              // non-linearity (ReLU for hidden, softmax for output)
```

Stacking several such layers lets the network approximate non-linear decision
boundaries. We **train** it by:

1. **Forward pass** — push a batch through the layers to get class probabilities.
2. **Loss** — measure how wrong the probabilities are vs. the true labels
   (mean softmax cross-entropy).
3. **Backward pass (backpropagation)** — apply the chain rule to get the
   gradient of the loss w.r.t. every weight and bias.
4. **SGD update** — nudge each parameter downhill: `param -= lr * grad`.

Repeat over many batches/epochs and the loss falls / accuracy rises.

---

## What this repo teaches

- How a dense layer's forward/backward math maps onto concrete CUDA kernels.
- The **host vs. device** memory split and how `cudaMemcpy` (H2D / D2H) moves data.
- The **1-D element-wise launch idiom** (`ceil_div`, `kBlockSize`) used by ReLU,
  bias-add, softmax, the CE gradient, and the SGD update.
- **2-D grids** for matrix multiply, mapping one thread to one output element.
- **Shared-memory tiling** (`gemm_tiled`) and *why* it is faster than the naive
  kernel — measured live by the microbenchmark.
- **Numerical stability** (row-max subtraction in softmax; clamped log in CE).
- **Synchronization & error checking** (`__syncthreads`, `cudaDeviceSynchronize`,
  `cudaGetLastError`) via the `CUDA_CHECK` / `CUDA_CHECK_LAST` macros.
- **Proving correctness without a reference framework** using a central
  finite-difference gradient check.
- **(push 0002) Parallel reduction** — the tree-reduction pattern (`reduce_sum`),
  used to compute loss & accuracy on the GPU instead of copying arrays back.
- **(push 0002) Stateful optimizers** — SGD, Momentum, and Adam behind one
  `Optimizer` API, showing per-parameter device **state** and bias correction.
- **(push 0002) Pluggable activations** — ReLU / LeakyReLU / Tanh, including why
  Tanh's backward needs the *post*-activation while ReLU's needs the *pre*-one.
- **(push 0002) Train/validation split + inference-only evaluation** to measure
  generalization on held-out data.

---

## Repository layout

```
MLP_CUDA/
├── README.md                   This file: the top-level didactic guide.
├── LICENSE                     MIT license (author: sora5801).
├── .gitignore                  Ignores build artifacts and editor/OS cruft.
├── Makefile                    nvcc build for Linux/WSL (all / run / clean).
├── CMakeLists.txt              Cross-platform build (Windows MSVC+CUDA, Linux).
├── include/
│   ├── common.cuh              Error macros (CUDA_CHECK), constants, ceil_div.
│   ├── matrix.cuh              Matrix struct + device-memory helper decls.
│   ├── kernels.cuh             All __global__ kernels + launch_* wrapper decls.
│   ├── mlp.cuh                 Layer / MLP structs + the training-API decls.
│   ├── dataset.cuh             Synthetic "blobs" dataset decls.
│   └── optim.cuh               (0002) SGD / Momentum / Adam optimizer API.
├── src/
│   ├── matrix.cu               Matrix helpers: alloc/free/zero/copy/bytes.
│   ├── kernels.cu              ALL CUDA kernels + launchers — the heart of it.
│   ├── mlp.cu                  forward / backward / loss / grad-check / evaluate.
│   ├── dataset.cu              make_blobs, standardize, shuffle, split (host-side).
│   ├── optim.cu                (0002) Momentum & Adam kernels + optim_create/step.
│   └── main.cu                 Entry point: data → build → train → validate.
└── docs/
    ├── math_derivation.md      Full forward/backward derivation, eq → kernel.
    ├── cuda_concepts.md        Every CUDA concept used here, tied to kernels.
    └── changelog/
        ├── README.md           The per-push changelog convention.
        ├── 0001-initial-implementation.md            The first push.
        └── 0002-optimizers-activations-reduction.md  This push.
```

---

## The math (compact)

Conventions used **everywhere** in this repo (see `docs/math_derivation.md` for
the full derivation):

- Matrices are **row-major** flat `float*` device arrays.
- A layer's weight `W` has shape **`[in, out]`** → `W[i*out + o]`.
- A batch of activations `A` has shape **`[batch, features]`** → `A[r*features + c]`.
- A bias `b` has shape **`[1, out]`** (length `out`) → `b[o]`.

### Forward (per layer `l`)

| Step          | Equation                                   | Kernel                  |
|---------------|--------------------------------------------|-------------------------|
| Affine        | `Z = A_prev · W + b`                        | `gemm_naive` + `add_bias` |
| Hidden act.   | `A = max(0, Z)`                            | `relu_forward`          |
| Output act.   | `A[r,:] = softmax(Z[r,:])`                 | `softmax_rows`          |

Element form of the affine step (note: **no transpose** in forward):

```
Z[r,o] = sum_i A_prev[r,i] * W[i,o] + b[o]
```

### Loss

Mean (over the batch) softmax cross-entropy:

```
L = (1/batch) * sum_r  -log( probs[r, label[r]] )
```

### Backward (backpropagation)

The output-layer pre-activation gradient collapses to a famously clean form
(derivation in the docs). The `1/batch` (batch-mean) factor is folded in here, so
every downstream gradient already carries it — the SGD step is then simply
`param -= lr * grad`:

```
dZ_out[r,c] = ( probs[r,c] - onehot[r,c] ) / batch
```

Then, walking layers from output back to input:

| Quantity   | Equation                          | Shapes                          | Kernel                          |
|------------|-----------------------------------|---------------------------------|---------------------------------|
| `dW`       | `A_prev^T · dZ`                   | `[in,out] = [in,batch]·[batch,out]` | `gemm_naive(transA=true)`  |
| `db`       | `colsum_r(dZ)`                    | `[1,out]`                       | `bias_grad`                     |
| `dA_prev`  | `dZ · W^T`                        | `[batch,in] = [batch,out]·[out,in]` | `gemm_naive(transB=true)`  |
| `dZ_prev`  | `dA_prev ⊙ relu'(Z_prev)`        | `[batch, in]`                   | `relu_backward`                 |

See `docs/math_derivation.md` for the chain-rule derivation of each line.

---

## CUDA concepts used

A quick index; each is explained concretely (with the kernel that uses it) in
`docs/cuda_concepts.md`:

- **Host vs. device memory** and `cudaMemcpy` H2D/D2H directions.
- **Threads / blocks / grids** and the **1-D element-wise launch idiom**
  (`ceil_div(n, kBlockSize)` blocks of `kBlockSize` threads).
- **2-D grids** for GEMM (one thread ↔ one output element `C[m,n]`).
- **Shared memory & tiling** in `gemm_tiled` (tile edge `kTileDim = 16`), and the
  global-vs-shared bandwidth / memory-coalescing reason it wins.
- **Thread divergence** (the `if` in ReLU).
- **Numerical stability** (subtracting the row max before `exp` in softmax).
- **Synchronization**: `__syncthreads()` within a block; `cudaDeviceSynchronize()`
  and `cudaGetLastError()` for async error surfacing.
- **Occupancy** basics (why `kBlockSize = 256` and `16x16 = 256`-thread tiles).

---

## Build & run

> **Prerequisite:** the **NVIDIA CUDA Toolkit** must be installed (it provides
> `nvcc`, the CUDA runtime headers, and `cuda_runtime.h`), plus an NVIDIA GPU
> with a working driver. This repo uses **only** the CUDA runtime API and the C++
> standard library — no cuBLAS / cuRAND / Thrust.
>
> **Verified build:** RTX 2080 SUPER (compute capability **sm_75**), **CUDA 13.3**,
> MSVC (Visual Studio) on Windows, C++14. The gradient check passes (~1e-5) and
> training reaches 100% accuracy. On **CUDA ≥ 13** note that `cudaDeviceProp::clockRate`
> was removed; this repo reads the core clock via `cudaDeviceGetAttribute(cudaDevAttrClockRate)`
> (see `print_device_props` in `src/main.cu`).

### Option A — Makefile (Linux / WSL)

```bash
# Build build/mlp from all src/*.cu, linking into one executable.
make

# Optionally target a specific GPU architecture (recommended for best codegen),
# e.g. sm_75 (Turing), sm_86 (Ampere), sm_89 (Ada):
make ARCH=sm_86

# Build (if needed) and run.
make run

# Remove the build/ directory.
make clean
```

### Option B — CMake (Windows MSVC+CUDA, or Linux)

```bash
# Configure into a build/ directory. CMAKE_CUDA_ARCHITECTURES=native asks CMake
# to detect and target the GPU in this machine.
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native

# Compile (use --config Release on Windows multi-config generators).
cmake --build build --config Release

# Run the resulting executable.
#   Linux:   ./build/mlp
#   Windows: build\Release\mlp.exe
```

### Configuring the demo (push 0002)

All experiment knobs are compile-time constants at the top of `src/main.cu`.
Edit and rebuild to study different setups — the **gradient check validates
whichever you choose**:

| Constant / call | Options |
|-----------------|---------|
| `kHiddenAct` | `Activation::ReLU` (default) · `Activation::LeakyReLU` · `Activation::Tanh` |
| `optim_create(net, …)` | `opt_adam(0.01f)` (default) · `opt_momentum(0.1f, 0.9f)` · `opt_sgd(0.1f)` |
| `kValSamples` | size of the held-out validation split (default 192) |

Note Adam's learning rate (~`0.01`) is intentionally smaller than SGD's (~`0.1`)
because Adam normalizes each step by the gradient's running RMS.

---

## Expected output

When it runs you should see (numbers will vary slightly by GPU and seed, but the
*trends* are the contract):

1. **GPU info** — the device name and a few properties (printed via
   `cudaGetDeviceProperties`).
2. **GEMM microbenchmark** — naive vs. tiled time on a `512×512×512` multiply,
   with the **tiled kernel faster** (a speedup `> 1×`), e.g.:

   ```
   GEMM 512x512x512:  naive = 1.83 ms   tiled = 0.71 ms   speedup = 2.58x
   ```

3. **Gradient check** — small **relative errors** (roughly `1e-3` or smaller for
   a central difference with `eps = 1e-3`), confirming the analytic backward
   pass matches the numerical gradient:

   ```
   grad-check W[0]:  analytic=-0.014213  numeric=-0.014219  rel_err=4.2e-04  PASS
   ...
   ```

4. **Training** — the chosen optimizer's name, then the **loss decreasing** and
   **train accuracy rising** over the epochs. With the default Adam optimizer on
   this easy dataset it converges within a few epochs, e.g.:

   ```
   ==================== TRAINING (Adam, lr=0.01) ==========
   epoch  1/60  loss 0.3890  train_acc 0.873
   epoch 10/60  loss 0.0001  train_acc 1.000
   ...
   epoch 60/60  loss 0.0000  train_acc 1.000
   ```

5. **Throughput** — total training time and samples/second (CUDA-event timed).
6. **(push 0002) Validation** — held-out loss/accuracy on the data the optimizer
   never trained on, e.g. `held-out val: loss 0.0000  acc 1.000`. Close to the
   train accuracy ⇒ the model generalized (this dataset is easily separable).

---

## Study guide / suggested exercises

Work through these in roughly increasing difficulty — each one forces you to
touch a different CUDA concept:

1. **Read the data flow.** Trace one batch from `main.cu` through `mlp_forward`:
   which kernel runs, with what grid/block, on what shapes? Write it down.
2. **Break a transpose.** In `mlp_backward`, flip a `transA`/`transB` flag and
   predict (then observe) how the gradient check fails. This teaches why the
   `[in,out]` vs `[out,in]` distinction matters.
3. **Tile the rest.** The MLP uses `gemm_naive` for correctness. Swap in
   `gemm_tiled` for the no-transpose forward GEMM and confirm results are
   identical but faster. (Then think about why backward still needs transposes.)
4. **Tune the launch.** Change `kBlockSize` (e.g. 128, 512) and `kTileDim`
   (e.g. 8, 32) and re-run the microbenchmark. Explain the occupancy trade-offs.
5. **Add an activation.** Implement `tanh`/`leaky_relu` forward+backward kernels
   and route a hidden layer through them. Re-run the grad-check.
6. **Reduce properly.** `softmax_rows`, `bias_grad`, and `cross_entropy_loss` use
   one thread per row/column. Rewrite one as a block-level parallel reduction
   with shared memory and `__syncthreads()`.
7. **Stream the data.** Overlap H2D copies of the next batch with compute on the
   current batch using a second `cudaStream_t`.
8. **Make it deeper/wider.** Change `layer_sizes` and `cluster_std`; observe how
   capacity and class overlap affect the final accuracy.

---

## Changelog convention

Every time something new is pushed to GitHub, a new numbered markdown file is
added under `docs/changelog/` as study notes describing exactly what changed and
why. The naming format is:

```
docs/changelog/NNNN-short-title.md      e.g. 0001-initial-implementation.md
```

See `docs/changelog/README.md` for the full convention and
`docs/changelog/0001-initial-implementation.md` for the notes on this first push.

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE). Author / owner
handle: **sora5801**.
