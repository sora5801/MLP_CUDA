# 0004 — Visual Studio solution + a guided feature-showcase demo

**Date:** 2026-06-19
**Pushed by:** sora5801
**Type:** Tooling + demo (no library/algorithm changes)

This push makes the repo open-and-run in **Visual Studio 2026** and adds a single
**demo program that showcases everything** built so far (pushes 0001–0003), so you
can step through the whole feature set in one run. No kernels or math changed.

---

## Summary

- A native **Visual Studio solution** (`MLP_CUDA.sln` + `MLP_CUDA.vcxproj` +
  `.filters`) that compiles the `.cu` files with nvcc via the CUDA build
  customization. Open the `.sln`, press F5, done.
- A new **`demo/demo.cu`** — an alternate `main()` that runs a guided tour:
  GPU info → GEMM tiling benchmark → on-device RNG self-test → gradient check
  (with dropout) → **optimizer / activation / dropout comparison tables**.
- `.gitignore` updated for Visual Studio / MSBuild artifacts.

Verified by building with **MSBuild** (the same engine VS uses) — Release|x64 — and
running the resulting `x64\Release\MLP_CUDA.exe`.

---

## What changed

| File | Role |
|------|------|
| `MLP_CUDA.sln` | Visual Studio solution: one project, `Debug\|x64` + `Release\|x64`. |
| `MLP_CUDA.vcxproj` | Native CUDA project. Imports `CUDA 13.3.props/.targets`; compiles the library `.cu` files **plus** `demo/demo.cu` (but **not** `src/main.cu` — that has its own `main()`). Targets `sm_75`; `PlatformToolset = $(DefaultPlatformToolset)`. |
| `MLP_CUDA.vcxproj.filters` | Solution Explorer grouping (Header Files / Library Sources / Demo). |
| `demo/demo.cu` | The guided showcase entry point (details below). |
| `.gitignore` | Ignore `.vs/`, `x64/`, `*.user`, `*.pdb`, etc. |

### Two entry points, one library

`src/main.cu` and `demo/demo.cu` both define `main()`, so they can never be in the
same build. The split is clean:

- **Visual Studio** builds `demo/demo.cu` + the library (it lists files explicitly,
  excluding `src/main.cu`).
- **Makefile / CMake** build `src/*.cu` (which includes `main.cu`, and does *not*
  reach into `demo/`).

So each build system has exactly one `main()`. They share all of `include/` and the
library `.cu` files.

---

## What `demo/demo.cu` shows

Run top-to-bottom, it prints a labeled section per feature:

1. **GPU device** — name, compute capability, SM count, clock, memory.
2. **GEMM 512³** — naive vs shared-memory tiled, with the speedup (push 0001).
3. **RNG self-test** — ~1M on-device uniforms reduced to a mean ≈ 0.5, composing
   the counter-based RNG (0003) with the parallel reduction (0002).
4. **Gradient check** — finite differences vs analytic `dW`, run *with dropout
   active*, proving the whole forward/backward chain (push 0001 + 0003).
5. **Optimizer comparison** — SGD vs Momentum vs Adam on identical inits (0002).
6. **Activation comparison** — ReLU vs LeakyReLU vs Tanh (0002).
7. **Dropout comparison** — `p = 0.0` vs `p = 0.5`, training vs held-out
   validation, exercising train/inference mode (0003).

Sections 5–7 share one helper, `train_run(...)`, which builds a network, trains it,
and reports clean (inference-mode) train + validation metrics. All configs use the
same init seed, so each table isolates the one knob under study.

Example (RTX 2080 SUPER, sm_75): tiled GEMM ≈ 2.6× faster; RNG mean = 0.50000;
grad-check max rel.err ≈ 1.3e-4 → PASS; every optimizer/activation/dropout setting
reaches 1.000 train & val accuracy on the (easy) blobs.

---

## How to open & run in Visual Studio 2026

1. Double-click **`MLP_CUDA.sln`** (or File ▸ Open ▸ Project/Solution).
2. Pick the **Release** (or Debug) configuration, **x64** platform.
3. **Ctrl+F5** (Run without debugging) or **F5** (debug).

Requirements: the **NVIDIA CUDA Toolkit 13.x** (it installs the VS build
integration and sets the `CUDA_PATH` environment variable VS reads) and an NVIDIA
GPU. If your GPU is not Turing, change **CodeGeneration** in the `.vcxproj` from
`compute_75,sm_75` to match (e.g. `compute_86,sm_86` for Ampere) — or set it in
Project ▸ Properties ▸ CUDA C/C++ ▸ Device ▸ Code Generation.

### Notes / gotchas

- **`CudaToolkitDir`.** The project resolves the toolkit from `$(CUDA_PATH)` with an
  explicit fallback path, so it builds even if `CUDA_PATH` is unset.
- **CUDA version in the import.** The project imports `CUDA 13.3.props/.targets`. If
  you install a *different* CUDA version, update those two import paths (and the
  `CudaToolkitDir` fallback) to match, or re-add the build customization via
  Project ▸ Build Dependencies ▸ Build Customizations.
- **Windows SDK.** Pinned to `10.0.26100.0` (installed here). Visual Studio will
  offer to retarget if yours differs — accept, or edit `WindowsTargetPlatformVersion`.
- **Command-line builds are unchanged.** `make` / CMake still build `src/main.cu`
  into the `mlp` executable exactly as before.

---

## Build / run (command line, unchanged)

```sh
# Linux / WSL:           make ARCH=sm_75 run        # builds src/main.cu -> mlp
# Windows (CMake):       cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release
# Just the showcase via nvcc (excludes src/main.cu):
#   nvcc -O2 -std=c++14 -arch=sm_75 -Iinclude src\matrix.cu src\kernels.cu src\mlp.cu src\dataset.cu src\optim.cu demo\demo.cu -o build\demo.exe
```
