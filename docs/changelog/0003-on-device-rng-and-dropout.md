# 0003 — On-device RNG + dropout

**Date:** 2026-06-19
**Pushed by:** sora5801
**Type:** Feature expansion (builds on pushes 0001–0002)

This push adds **random numbers generated on the GPU** and the regularizer that
needs them, **dropout** — plus the train-vs-inference mode switch that dropout
forces you to make explicit. As with everything in this repo, no `cuRAND` is used:
the RNG is hand-written so you can see exactly how a parallel generator works.

---

## Summary

- A **stateless, counter-based RNG**: each random value is a hash of
  `(seed, index)` — no per-thread state, no synchronization, perfectly parallel,
  and fully reproducible. (This is the same idea as cuRAND's Philox generator.)
- **Inverted dropout** (`dropout_forward` / `dropout_backward`): randomly zero
  hidden activations with probability `p` during training and scale the survivors
  by `1/(1-p)`, so **inference needs no rescaling** — a single mode flag is the
  whole train/eval switch.
- **Train/inference mode** on the network (`MLP::training`, `mlp_set_training`);
  `mlp_evaluate` now runs in inference mode automatically.
- An **RNG self-test** in `main.cu` that fills ~1M uniforms and reduces them to a
  mean (≈0.5), composing the new RNG (0003) with the reduction (0002).

The headline correctness result: the finite-difference **gradient check passes
with dropout switched on**, for both ReLU and Tanh hidden layers. That is only
possible because the counter-based RNG makes the dropout mask *reproducible* for a
fixed seed (see "Why this works", below).

---

## What changed

### New kernels (in `src/kernels.cu` / `include/kernels.cuh`)

| Kernel | What it does |
|--------|--------------|
| `rng_uniform` (device helper) | Hashes `(seed, idx)` → uniform float in `[0,1)` via a splitmix64-style bit-mixer. The heart of the RNG; inlined into the kernels below. |
| `fill_uniform` (17) | Fills an array with `rng_uniform(seed, i)`. Used by the self-test. |
| `dropout_forward` (18) | `u=rng_uniform(seed,i); mask[i]= u>=p ? 1/(1-p):0; out[i]=in[i]*mask[i]`. Caches the scaled mask. |
| `dropout_backward` (19) | `grad_in[i]=grad_out[i]*mask[i]` — reuses the cached mask (no RNG). |

### Modified files

| File | Change |
|------|--------|
| `include/mlp.cuh`, `src/mlp.cu` | `Layer` gains `dropout_p`, `dropout_mask`, **`A_out`**, and a transient `dropped` flag. `MLP` gains `training` and `rng_state`. `mlp_create` takes a defaulted `dropout_p`. Forward applies dropout (training only) into `A_out` while keeping `A` pure; backward runs `dropout_backward` before the activation derivative. New `mlp_set_training`; `mlp_evaluate` forces inference mode. |
| `src/main.cu` | New `kDropoutP` constant; `run_rng_selftest()`; pass dropout to `mlp_create`; advance `net.rng_state` once per training step; print dropout in the training header. |

The kernel roster is now **19 in `kernels.cu`** (+ 2 optimizer kernels in `optim.cu`).

---

## Why

- **On-device RNG is its own CUDA lesson.** A classic CPU RNG advances a *state*
  one draw at a time — inherently serial. On a GPU you instead want thread *i* to
  compute its own number with no shared state. A **counter-based** RNG does this:
  treat `(seed, index)` as a counter and run it through a strong integer hash. The
  output bits are well-distributed and independent across `index`, with zero
  coordination between threads. See the new "On-device RNG" section in
  `docs/cuda_concepts.md`.
- **Dropout introduces the train/inference distinction.** Many layers behave
  differently at training vs test time; dropout is the canonical example. Inverted
  dropout pushes the `1/(1-p)` scaling into training so inference is a plain
  forward pass — `mlp_set_training(net, false)` (which `mlp_evaluate` does for you)
  is the entire switch. Math in `docs/math_derivation.md` §6.3.

---

## Why this works: gradient-checking a *random* layer

A finite-difference check perturbs a weight by ±ε and compares the change in loss
to the analytic gradient. If the dropout mask were re-randomized on each forward
pass, the loss would be noisy and the check meaningless. The fix falls out of the
RNG design for free: because each mask entry is `hash(seed, index)` — a function of
the **seed and position only, never the data** — holding the seed fixed makes
dropout a deterministic function of the inputs. `mlp_grad_check` does not advance
`rng_state`, so every one of its ±ε forward passes regenerates the *identical*
mask, and the analytic backward (which reads that same cached mask) matches. The
training loop, by contrast, advances `rng_state` once per step to get a fresh mask
each time. This is a clean illustration of why stateless, reproducible RNG is so
convenient on the GPU.

## Why `A` stays pure (a subtle design point)

Push 0002 established that **Tanh's backward reads the post-activation `A`**
(`tanh'(z)=1-a²`), while ReLU/LeakyReLU read the pre-activation `Z`. If dropout
overwrote `A` in place, Tanh's backward would see the *dropped, rescaled* values
and compute the wrong derivative for surviving units. So this push keeps `A` as the
**pure** activation and writes the dropped output into a separate **`A_out`**
buffer (which feeds the next layer and serves as `A_prev` in the weight-gradient).
The backward order mirrors the forward: forward is *activation → dropout*, so
backward is *dropout → activation derivative*. The `dropped` flag records, per
forward pass, whether dropout fired, so backward picks the matching buffer.

---

## Verification (RTX 2080 SUPER, CUDA 13.3, sm_75)

- **RNG self-test:** 1,048,576 uniforms, **mean = 0.50000** (expected 0.5).
- **Gradient check with dropout `p=0.2` active:**

  | Hidden activation | Grad-check max rel. err | Final train acc | Held-out val acc |
  |-------------------|-------------------------|-----------------|------------------|
  | ReLU (default)    | 1.26e-4  → PASS         | 1.000           | 1.000            |
  | Tanh              | 1.09e-4  → PASS         | 1.000           | 1.000            |

  Both pass, confirming the dropout forward/backward and the pure-`A` design are
  correct. Training loss is now slightly noisier (≈1e-4 jitter rather than exactly
  0) — the visible signature of dropout — while the clean, dropout-free validation
  accuracy stays at 1.000.

---

## Notes / gotchas

- **The RNG is not cryptographic.** splitmix64 finalizing a `(seed,index)` counter
  is fast, reproducible, and good enough for dropout/initialization, but it is not
  meant to pass every statistical test suite or resist prediction.
- **Per-element dropout.** The mask is drawn per `(row, unit)` element (classic
  Hinton dropout), not shared across the batch. Each activation is dropped
  independently.
- **Output layer never drops.** Dropout applies to hidden layers only; the softmax
  output is untouched.
- **Mode is restored.** `mlp_evaluate` saves and restores `net.training`, so
  calling it never leaves the network stuck in inference mode.
- **On this dataset dropout isn't needed** for accuracy (the blobs are easily
  separable). It is included to demonstrate the mechanism; set `kDropoutP = 0.0f`
  in `main.cu` to disable it.

---

## Build / run

Commands unchanged (the build globs `src/*.cu`). A CUDA toolkit + NVIDIA GPU are
required.

```sh
# Windows (MSVC + CUDA) — what this push was verified with:
#   nvcc -O2 -std=c++14 -arch=sm_75 -Iinclude src\*.cu -o build\mlp.exe
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build build --config Release

# Linux / WSL:
make ARCH=sm_75 run
```

Study knobs at the top of `src/main.cu`: `kDropoutP` (try `0.0f`, `0.5f`),
`kHiddenAct`, and the `optim_create(...)` optimizer. The gradient check validates
whatever combination you pick.
