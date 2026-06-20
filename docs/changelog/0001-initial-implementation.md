# 0001 — Initial Implementation

**Date:** 2026-06-19
**Author:** sora5801
**Type:** Initial push (greenfield)

This is the first entry in the per-push changelog (see
[`README.md`](./README.md) for the convention). It documents the entire
initial implementation of **MLP_CUDA**: a heavily-commented, didactic CUDA
C++ implementation of a Multi-Layer Perceptron with **forward and backward
passes**, trained with SGD on a synthetic Gaussian-blobs classification
problem. The repository is study material — correctness and comment quality
are valued over cleverness.

---

## 1. What this push adds (one paragraph)

A complete, runnable end-to-end pipeline: synthetic data generation →
standardization → He-initialized MLP construction → forward pass → mean
softmax cross-entropy loss → analytic backward pass → SGD update → training
loop with per-epoch loss/accuracy reporting. Every numerical operation runs
on the GPU through hand-written CUDA kernels (no cuBLAS / Thrust / cuRAND).
The push also includes a **finite-difference gradient check** that proves the
backward pass is correct without any reference framework, and a
**tiled-vs-naive GEMM microbenchmark** that demonstrates the payoff of
shared-memory tiling. All source is documented as a tutorial for a reader who
knows C++ but is learning CUDA.

---

## 2. Every file added (role + concept taught)

### Build & meta files

| File | Role | Concept taught |
|------|------|----------------|
| `README.md` | Project overview: what an MLP is, repo layout, the math, build/run, expected output, study exercises. | How the pieces fit; entry point for a new reader. |
| `LICENSE` | MIT license, author `sora5801`. | — |
| `.gitignore` | Ignores `build/`, `*.o`, `*.exe`, `mlp`, and editor/OS cruft. | Keeping build artifacts out of version control. |
| `Makefile` | `nvcc` build for Linux/WSL. Targets `all`, `run`, `clean`; compiles each `src/*.cu` to `build/*.o` and links `build/mlp`. | The `nvcc` compile/link flow; `-Iinclude`, `-std=c++14`, optional `-arch=sm_XX`. |
| `CMakeLists.txt` | Cross-platform build (Windows MSVC+CUDA and Linux). Enables the `CUDA` language, sets `CMAKE_CUDA_STANDARD 14`. | CMake's first-class CUDA support and `CMAKE_CUDA_ARCHITECTURES=native`. |

### Headers (`include/`)

| File | Role | Concept taught |
|------|------|----------------|
| `common.cuh` | Error-checking macros `CUDA_CHECK` / `CUDA_CHECK_LAST`, the constants `kBlockSize=256` and `kTileDim=16`, and the `ceil_div` host helper. | Robust CUDA runtime error handling; the "how many blocks cover *n* elements" idiom; why we synchronize after launches for didactic error surfacing. |
| `matrix.cuh` | The `Matrix` struct (device `float* data`, `rows`, `cols`) and device-memory helper declarations. | Row-major flat storage in GPU global memory; H2D/D2H transfer direction. |
| `kernels.cuh` | Declarations of all 10 `__global__` kernels **and** their `__host__` `launch_*` wrappers. | Separating kernel code from the grid/block configuration that drives it. |
| `mlp.cuh` | The `Layer` and `MLP` structs and the network API (`mlp_create`, `mlp_forward`, `mlp_backward`, `mlp_sgd_step`, `mlp_compute_loss`, `mlp_accuracy`, `mlp_grad_check`). | How layer parameters, caches, and gradients are laid out; the public training API. |
| `dataset.cuh` | The `Dataset` struct (host `X`, `y`) and `make_blobs` / `dataset_free` / `dataset_standardize` / `dataset_shuffle`. | Deterministic synthetic data; feature standardization; epoch shuffling. |

### Sources (`src/`)

| File | Role | Concept taught |
|------|------|----------------|
| `matrix.cu` | Implements the `Matrix` helpers: `matrix_alloc` (`cudaMalloc`), `matrix_free` (`cudaFree`), `matrix_zero` (`cudaMemset`), `matrix_copy_to_device`/`matrix_copy_to_host` (`cudaMemcpy` H2D/D2H), `matrix_bytes`. | What `cudaMalloc`/`cudaMemcpy`/`cudaMemset` actually do and the direction enums. |
| `kernels.cu` | **The heart of the repo.** All 10 kernels and launchers: `gemm_naive`, `gemm_tiled`, `add_bias`, `relu_forward`, `relu_backward`, `softmax_rows`, `cross_entropy_grad`, `cross_entropy_loss`, `bias_grad`, `sgd_update`. | Thread-index math, 1-D vs 2-D launch grids, shared-memory tiling, numerically stable softmax, thread divergence in ReLU. |
| `mlp.cu` | Wires kernels into `forward`/`backward`/`update`/`loss`/`accuracy`/`grad_check`. Allocates layer caches; He-initializes weights via `std::mt19937_64`. | Backprop as a sequence of GEMMs with transposes; central-difference gradient verification. |
| `dataset.cu` | Generates Gaussian "blobs", standardizes columns, Fisher-Yates shuffle. | Reproducible RNG; why standardization keeps activations well-scaled. |
| `main.cu` | Entry point: config → data → build → device-property print → GEMM benchmark → grad-check → training loop → cleanup → `cudaDeviceReset()`. CUDA-event timing of training. | How every kernel/layer ties together into a real program. |

### Docs (`docs/`)

| File | Role | Concept taught |
|------|------|----------------|
| `math_derivation.md` | Full forward/backward derivation, including the `dL/dZ_out = (softmax − onehot)/batch` identity and the hidden-ReLU backprop chain. | The math each kernel implements. |
| `cuda_concepts.md` | CUDA concepts used here, tied to actual kernels: host/device memory, threads/blocks/grids, 2-D GEMM grids, shared-memory tiling, divergence, numerical stability, synchronization, occupancy. | The CUDA vocabulary needed to read the source. |
| `changelog/README.md` | Explains the per-push changelog convention and file-name format. | Keeping legible study notes per push. |
| `changelog/0001-initial-implementation.md` | **This file.** | What the first push contains. |

---

## 3. Architecture summary

### Data and math conventions (must never deviate)

- All matrices are **row-major** flat `float*` device arrays.
- A weight matrix `W` for a layer with `in` inputs / `out` outputs has shape
  **`[in, out]`**, indexed `W[i*out + o]`.
- A batch of activations `A` has shape **`[batch, features]`**, indexed
  `A[r*features + c]`.
- Bias `b` has length `out`, indexed `b[o]`.
- Forward of a linear layer: `Z = A_prev · W + b`, i.e.
  `Z[r,o] = sum_i A_prev[r,i]·W[i,o] + b[o]` (**no transpose** in forward).
- **Loss = mean (over the batch) softmax cross-entropy.** The `1/batch`
  factor is folded into the output-layer gradient (`cross_entropy_grad`), so
  every downstream gradient (`dW`, `db`, `dA`) already carries the `1/batch`
  scale. The SGD update is therefore plain `param -= lr * grad`.
- Hidden layers use **ReLU**; the output layer uses **softmax** (its
  post-activation `A` holds class probabilities).

### Network structure

`mlp_create({2, 64, 32, 3}, ...)` builds **3 layers** (`num_sizes − 1`):

```
input[batch,2]
   └─ Layer 0: W[2,64],  b[64]  → ReLU
   └─ Layer 1: W[64,32], b[32]  → ReLU
   └─ Layer 2: W[32,3],  b[3]   → softmax  (is_output = true)
```

Each `Layer` owns its parameters (`W`, `b`), its forward caches (`Z`
pre-activation, `A` post-activation), and its gradients (`dW`, `db`, `dZ`,
`dA`). Weights are **He-initialized** `N(0, sqrt(2/in))` via a seeded
`std::mt19937_64`; biases start at zero.

### Forward pass (`mlp_forward`)

```
prev = batch_input
for l in 0..num_layers-1:
    launch_gemm(prev.data, L.W.data, L.Z.data, batch, L.out, L.in, false, false)
    launch_add_bias(L.Z.data, L.b.data, batch, L.out)
    if L.is_output: launch_softmax_rows(L.Z.data, L.A.data, batch, L.out)
    else:           launch_relu_forward(L.Z.data, L.A.data, batch*L.out)
    prev = L.A
```

### Backward pass (`mlp_backward`)

```
// output layer => dZ = (probs - onehot)/batch
launch_cross_entropy_grad(out.A.data, d_labels, out.dZ.data, batch, num_classes)
for l from num_layers-1 down to 0:
    prev_act = (l==0) ? batch_input : layers[l-1].A
    // dW = prev_act^T · dZ        [in,out] = [in,batch]·[batch,out]
    launch_gemm(prev_act.data, L.dZ.data, L.dW.data, L.in, L.out, batch, true,  false)
    // db = colsum(dZ)
    launch_bias_grad(L.dZ.data, L.db.data, batch, L.out)
    if l > 0:
        prevL = layers[l-1]
        // dA_prev = dZ · W^T       [batch,in] = [batch,out]·[out,in]
        launch_gemm(L.dZ.data, L.W.data, prevL.dA.data, batch, L.in, L.out, false, true)
        // dZ_prev = dA_prev ⊙ relu'(Z_prev)
        launch_relu_backward(prevL.dA.data, prevL.Z.data, prevL.dZ.data,
                             batch*prevL.out_features)
```

Note `L.in == prevL.out_features`. The single GEMM kernel (`gemm_naive`)
serves all three matmul shapes — forward, `dW = A_prevᵀ·dZ`, and
`dA_prev = dZ·Wᵀ` — purely by toggling the `transA`/`transB` flags, which is
the central economy of this design.

### Update (`mlp_sgd_step`)

For every layer, `launch_sgd_update(W, dW, lr, ...)` and
`launch_sgd_update(b, db, lr, ...)`. Because `1/batch` is already baked into
the gradients, no extra scaling is needed.

---

## 4. Gradient check (correctness proof)

`mlp_grad_check` validates the analytic backward pass against a numerical
reference using **central finite differences** on a few of layer 0's weights:

```
numeric  dL/dW[i] ≈ ( loss(W[i]+eps) - loss(W[i]-eps) ) / (2*eps)   // eps ~ 1e-3
analytic dL/dW[i] = layers[0].dW   // from mlp_backward
relative error = |analytic - numeric| / (|analytic| + |numeric| + tiny)
```

It is run on the first batch **before** training and prints per-weight
relative errors. Passing values are on the order of `1e-4` or smaller, which
confirms the chain of GEMM-with-transpose, ReLU-backward, and
`cross_entropy_grad` kernels is implemented correctly — no PyTorch/TensorFlow
reference required. This is the single most important didactic artifact in the
push: it turns "I think backprop is right" into a measured fact.

---

## 5. GEMM microbenchmark (optimization lesson)

`main.cu` runs a one-shot comparison of `gemm_naive` vs `gemm_tiled` on a
moderate problem (e.g. `512×512×512`), timed with **CUDA events**
(`cudaEventRecord` / `cudaEventElapsedTime`). It prints both elapsed times and
the speedup. The tiled kernel stages `kTileDim×kTileDim` (16×16 = 256-thread)
sub-blocks of `A` and `B` into shared memory so each loaded element is reused
across a whole tile row/column, cutting global-memory traffic and improving
coalescing. This is the concrete payoff of the shared-memory concept covered
in `docs/cuda_concepts.md`. (The MLP itself uses the naive GEMM for
correctness clarity and because it must support transposes; the tiled kernel
is the optimization exhibit.)

---

## 6. Build & run (brief)

**Linux / WSL (Makefile):**

```sh
make            # compiles src/*.cu -> build/*.o, links build/mlp
make run        # builds then runs ./build/mlp
make clean      # removes build/
# optional GPU arch: make ARCH=sm_75
```

**Windows / Linux (CMake):**

```sh
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
./build/mlp          # (Windows: build\Debug\mlp.exe or build\mlp.exe)
```

A CUDA toolkit (and an NVIDIA GPU) is required.

**Expected output:** the GPU name/properties, the GEMM benchmark times +
speedup, gradient-check relative errors near zero, then per-epoch loss
decreasing and training accuracy rising toward ~1.0 on the separable blobs,
plus total training time / throughput.

---

## 7. Notes & deliberate non-goals

- No external math libraries (cuBLAS/Thrust/cuRAND) — every kernel is
  hand-written for teaching value.
- `nvcc` may be unavailable on the author's machine; the code is written to be
  **correct by construction** and is not assumed to have been compiled here.
- The last partial batch of an epoch is **dropped** so every batch is exactly
  `batch_size` rows (simplifies fixed-size device buffers); `n_per_class` is
  chosen so the total sample count is divisible by `batch_size`.
- All randomness is seeded `std::mt19937_64` for reproducible runs.
- Optimizations skipped for clarity (e.g. fusing bias-add into GEMM, a tiled
  transposed GEMM for backward) are flagged inline as exercises.
