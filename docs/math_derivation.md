# Math Derivation — Forward and Backward Passes of the MLP

This document is the mathematical reference for the whole repo. Every equation
here is implemented by exactly one CUDA kernel (or launcher) in `src/kernels.cu`,
and the forward/backward *driver loops* live in `src/mlp.cu`. Wherever an
equation appears, the kernel that computes it is named in **brackets** so you can
jump straight from "the math" to "the code that runs on the GPU".

The single most important result derived below is the output-layer gradient

```
dL/dZ_out = (softmax(Z_out) - onehot(y)) / batch        [cross_entropy_grad]
```

which is what makes softmax + cross-entropy so clean to backprop. We derive it
from scratch in §3.

Conventions match the master build spec exactly. Do not change them here without
changing them everywhere — the kernels assume these shapes and layouts byte for
byte.

--------------------------------------------------------------------------------
## 0. Notation, shapes, and memory layout

All matrices are stored **row-major** in flat `float*` arrays that live in GPU
*global memory* (one contiguous allocation per `Matrix`, see `matrix.cuh`). For a
matrix with `R` rows and `C` columns, the element at logical position `(r, c)`
lives at flat index `r*C + c`. This single rule is what lets a CUDA thread map a
1-D global index back to a 2-D `(row, col)` coordinate.

We use these symbols throughout:

| Symbol        | Meaning                                  | Shape (row-major)        | Stored as / units            |
| ------------- | ---------------------------------------- | ------------------------ | ---------------------------- |
| `B`           | batch size (rows processed together)     | scalar                   | `MLP.batch_size`             |
| `n_in`        | inputs to a layer (`in_features`)        | scalar                   | `Layer.in_features`          |
| `n_out`       | outputs of a layer (`out_features`)      | scalar                   | `Layer.out_features`         |
| `K`           | number of classes (output layer `n_out`) | scalar                   | `MLP.num_classes`            |
| `A_prev`      | input activations to a layer             | `[B, n_in]`              | `A_prev[r*n_in + i]`         |
| `W`           | layer weights                            | `[n_in, n_out]`          | `W[i*n_out + o]` (`Layer.W`) |
| `b`           | layer bias (broadcast over rows)         | `[1, n_out]` (len n_out) | `b[o]` (`Layer.b`)           |
| `Z`           | pre-activation (linear output)           | `[B, n_out]`             | `Z[r*n_out + o]` (`Layer.Z`) |
| `A`           | post-activation (ReLU or softmax output) | `[B, n_out]`             | `A[r*n_out + o]` (`Layer.A`) |
| `y`           | true class label per row                 | `[B]` ints in `[0,K)`    | device `int*` `d_labels`     |
| `L`           | scalar loss (mean over batch)            | scalar                   | dimensionless                |

Gradients use the convention `dX := dL/dX` and have **the same shape as `X`**:
`dW` is `[n_in, n_out]`, `db` is `[1, n_out]`, `dZ` is `[B, n_out]`, and `dA` is
`[B, n_out]`. Keeping gradient shape identical to the parameter shape is what
lets the SGD update [`sgd_update`] be a trivial element-wise `param -= lr*grad`.

A crucial accounting point (master spec §0.3): **the `1/B` batch-mean factor is
folded into the output-layer gradient** at the very first backward step. Every
gradient computed *after* that already carries the `1/B` scale, so no other
kernel ever divides by `B` again, and SGD never multiplies it back in.

--------------------------------------------------------------------------------
## 1. Forward pass

### 1.1 One linear layer

For each layer, the pre-activation is an affine map of the previous activations:

```
Z = A_prev · W + b                                       [launch_gemm, add_bias]

elementwise:   Z[r, o] = ( sum over i of  A_prev[r, i] * W[i, o] )  +  b[o]
                          \________________________________/    \____/
                                matrix product (GEMM)            bias add
```

Shapes line up as `[B, n_out] = [B, n_in] · [n_in, n_out] + [1, n_out]`. Note
there is **no transpose in the forward direction** — `A_prev` is `[B, n_in]` and
`W` is `[n_in, n_out]`, so the contraction is over the shared `n_in` axis.

- The matrix product `A_prev · W` is computed by **[`gemm_naive`]** via the
  launcher **[`launch_gemm`]** with `M=B`, `N=n_out`, `K=n_in`,
  `transA=false`, `transB=false`. (The tiled variant **[`gemm_tiled`]** computes
  the same product faster and is exercised by the microbenchmark in `main.cu`;
  the MLP path uses the naive kernel for clarity.)
- The bias add `+ b[o]` broadcasts the length-`n_out` vector down every one of
  the `B` rows. That broadcast is **[`add_bias`]**: thread `(r, c)` does
  `Z[r*cols + c] += bias[c]`, i.e. the *same* `bias[c]` is read by all `B`
  threads in column `c`.

Why split into two kernels instead of one fused GEMM+bias? Clarity. Each kernel
teaches one idea (a contraction; a broadcast). Fusing them is left as an
exercise (see the "// optimization: ... left as an exercise" comments in the
code).

### 1.2 Activation

Hidden layers apply elementwise **ReLU**; the output layer applies **softmax**
over each row (each row is one sample's class scores).

**ReLU (hidden layers)** — **[`relu_forward`]**, launched over all `B*n_out`
elements as a flat 1-D grid:

```
A[i] = relu(Z[i]) = max(0, Z[i])
```

It is purely elementwise, so the kernel ignores 2-D structure and treats `Z` as a
flat array of `n = B*n_out` floats (hence the launcher takes `batch*out` as `n`).

**Softmax (output layer)** — **[`softmax_rows`]**, one thread per row because the
normalization couples all `K` entries of a row:

```
for each row r:
    m       = max_c  Z[r, c]                 # row max, for numerical stability
    s       = sum_c  exp( Z[r, c] - m )      # normalizer
    A[r, c] = exp( Z[r, c] - m ) / s         # class probability, sums to 1 over c
```

Subtracting the row max `m` before `exp` is the **numerically stable softmax**
trick. Mathematically `softmax(z) == softmax(z - m)` for any constant `m`,
because the constant cancels between numerator and denominator:

```
exp(z_c - m) / sum_k exp(z_k - m)
   = ( exp(z_c)/exp(m) ) / ( (1/exp(m)) sum_k exp(z_k) )
   = exp(z_c) / sum_k exp(z_k).
```

But computationally it matters: `Z` can hold large logits, and `exp(large)`
overflows to `+inf` in `float`. After subtracting the max, the largest exponent
argument is `0`, so `exp` tops out at `1.0` and never overflows. This is the same
reason the loss kernel clamps probabilities away from zero (§2).

### 1.3 Forward driver (chaining layers)

`mlp_forward` (in `src/mlp.cu`) stitches the per-layer math into the full network
by feeding each layer's output activation `A` as the next layer's `A_prev`:

```
prev = batch_input                                   # [B, input_features]
for l in 0 .. num_layers-1:
    launch_gemm(prev.data, L.W.data, L.Z.data,  B, L.out, L.in, false, false)  # Z = prev·W
    launch_add_bias(L.Z.data, L.b.data, B, L.out)                              # Z += b
    if L.is_output: launch_softmax_rows(L.Z.data, L.A.data, B, L.out)          # A = softmax(Z)
    else:           launch_relu_forward(L.Z.data, L.A.data, B*L.out)           # A = relu(Z)
    prev = L.A                                        # output feeds next layer
```

After the loop, `net.layers[num_layers-1].A` holds the class probabilities
`[B, K]`. Both `Z` and `A` are *cached* in each `Layer` because backprop needs
them: `Z` for `relu'` and `A` for the softmax-CE gradient.

--------------------------------------------------------------------------------
## 2. Loss: mean softmax cross-entropy

Let `p_r = A_out[r, :]` be the predicted probability vector for sample `r`, and
let `y_r` be its true class index. Cross-entropy for one sample is the negative
log-probability assigned to the correct class:

```
loss_r = - log( p_r[y_r] )
```

and the batch loss is the **mean** over the `B` samples:

```
L = (1/B) * sum over r of  loss_r
  = -(1/B) * sum over r of  log( A_out[r, y_r] )
```

The per-row term is computed by **[`cross_entropy_loss`]** (one thread per row):

```
loss_per_row[r] = -log( max( A_out[r, y_r], 1e-12 ) )
```

The `max(·, 1e-12)` clamp avoids `log(0) = -inf` if a probability underflows to
exactly `0` in `float`. The host (`mlp_compute_loss`) copies `loss_per_row` to
the CPU, sums it, and divides by `B` to get the scalar `L`. We deliberately do
the final sum/divide on the host: it is `B` cheap adds, and keeping it off the
GPU keeps the kernel a clean one-thread-per-row map (a parallel reduction is left
as an exercise).

Note: `cross_entropy_loss` is for **reporting and the grad-check** only. The
gradient does *not* flow through this kernel — backprop uses the closed-form
result of §3 instead, which is both exact and far cheaper.

--------------------------------------------------------------------------------
## 3. The key identity: dL/dZ_out = (softmax − onehot) / batch

This is the linchpin of the whole backward pass, implemented by
**[`cross_entropy_grad`]**. We derive it for a single sample (drop the row index
`r`), then restore the batch factor at the end.

### 3.1 Setup for one sample

Let `z = (z_1, ..., z_K)` be the output-layer logits (`Z_out` row), let
`p = softmax(z)` be the probabilities, and let `t` be the true class. The
single-sample loss is

```
loss = -log( p_t ),   where   p_c = exp(z_c) / S,   S = sum over k of exp(z_k).
```

We want `d(loss)/d z_j` for every logit `j`. Apply the chain rule through `p`:

```
d(loss)/d z_j = sum over c of  ( d(loss)/d p_c ) * ( d p_c / d z_j ).
```

### 3.2 Derivative of the loss wrt the probabilities

`loss = -log(p_t)` depends only on the single entry `p_t`, so

```
d(loss)/d p_c = -1/p_t   if c == t,    else 0.
```

### 3.3 Jacobian of the softmax

We need `d p_c / d z_j`. Start from `p_c = exp(z_c) / S` and use the quotient
rule, noting `dS/dz_j = exp(z_j)`:

**Case c == j:**

```
d p_c/d z_c = [ exp(z_c)*S - exp(z_c)*exp(z_c) ] / S^2
            = exp(z_c)/S  -  (exp(z_c)/S)^2
            = p_c - p_c^2
            = p_c (1 - p_c).
```

**Case c != j:**

```
d p_c/d z_j = [ 0*S - exp(z_c)*exp(z_j) ] / S^2
            = - (exp(z_c)/S)(exp(z_j)/S)
            = - p_c p_j.
```

Both cases combine into the standard softmax Jacobian using the Kronecker delta
`δ_{cj}` (which is `1` when `c==j`, else `0`):

```
d p_c/d z_j = p_c ( δ_{cj} - p_j ).
```

### 3.4 Combine via the chain rule

Plug §3.2 and §3.3 into the chain-rule sum. Only the `c == t` term of §3.2 is
nonzero, so the sum over `c` collapses to that single term:

```
d(loss)/d z_j = sum over c of ( d loss/d p_c )( d p_c/d z_j )
              = ( -1/p_t ) * ( d p_t/d z_j )                 # only c=t survives
              = ( -1/p_t ) * p_t ( δ_{tj} - p_j )            # softmax Jacobian at c=t
              = -( δ_{tj} - p_j )
              = p_j - δ_{tj}.
```

So for a single sample, the gradient wrt logit `j` is just the predicted
probability minus 1 if `j` is the true class:

```
d(loss)/d z_j = p_j - δ_{tj}  =  p_j - onehot(t)_j.
```

This is the beautiful cancellation that motivates pairing softmax with
cross-entropy: the `1/p_t` from the log-loss and the `p_t` from the softmax
Jacobian cancel exactly, leaving a subtraction with no division.

### 3.5 Restore the batch mean

The full batch loss is `L = (1/B) sum_r loss_r`, and sample `r`'s logits only
affect `loss_r`. Therefore the gradient of `L` wrt row `r`'s logits is the
single-sample result scaled by `1/B`:

```
dL/dZ_out[r, c] = ( p[r, c] - onehot(y_r)_c ) / B.
```

In matrix form, with `P = softmax(Z_out)` and `Y` the one-hot label matrix:

```
dL/dZ_out = ( P - Y ) / B.
```

This is **exactly** what **[`cross_entropy_grad`]** computes, one thread per
element `(r, c)`:

```
grad[r, c] = ( probs[r, c] - (labels[r] == c ? 1.0f : 0.0f) ) / rows;   // rows == B
```

The `/ rows` here *is* the `/ B` batch-mean fold-in promised in §0. Because this
is the first thing the backward pass computes, every downstream gradient inherits
the `1/B` scale automatically — which is why SGD is plain `param -= lr*grad`.

--------------------------------------------------------------------------------
## 4. Backprop through a layer

Given `dZ = dL/dZ` for a layer (shape `[B, n_out]`), we need three things:
the weight gradient `dW`, the bias gradient `db`, and the upstream activation
gradient `dA_prev` to hand to the previous layer. Recall the forward relation
`Z = A_prev · W + b`, i.e. `Z[r,o] = sum_i A_prev[r,i] W[i,o] + b[o]`.

### 4.1 Weight gradient — dW = A_prevᵀ · dZ

`W[i,o]` enters `Z[r,o]` (for every row `r`) multiplied by `A_prev[r,i]`. By the
chain rule, summing the contributions over all rows:

```
dL/dW[i,o] = sum over r of  dZ[r,o] * (∂Z[r,o]/∂W[i,o])
           = sum over r of  dZ[r,o] * A_prev[r,i]
           = sum over r of  A_prev[r,i] * dZ[r,o].
```

That sum over the batch axis `r` is precisely the matrix product of
`A_prevᵀ` (`[n_in, B]`) with `dZ` (`[B, n_out]`):

```
dW = A_prevᵀ · dZ          shape  [n_in, n_out] = [n_in, B] · [B, n_out]
```

Implemented by **[`launch_gemm`]** with `transA=true`: the GEMM contract
(`kernels.cuh`) reads `A_log[m,k] = A[k*M + m]` when `transA` is set, which
*logically transposes* `A_prev` without ever materializing the transpose in
memory. Call shape: `launch_gemm(prev_act, L.dZ, L.dW, M=n_in, N=n_out, K=B,
transA=true, transB=false)`. Because `dZ` already carries `1/B`, so does `dW`.

### 4.2 Bias gradient — db = column-sum of dZ over the batch

`b[o]` enters `Z[r,o]` for every row `r` with coefficient `1`
(`∂Z[r,o]/∂b[o] = 1`), so:

```
dL/db[o] = sum over r of  dZ[r,o] * 1  =  sum over r of  dZ[r,o].
```

That is the column sum of `dZ` down the batch axis:

```
db[o] = sum over r of  dZ[r, o]         shape [1, n_out]
```

Implemented by **[`bias_grad`]**, one thread per column `o` (there are only
`n_out` columns, which is small), each looping over the `B` rows to accumulate.
Again `dZ` already carries `1/B`, so `db` does too.

### 4.3 Upstream activation gradient — dA_prev = dZ · Wᵀ

`A_prev[r,i]` enters `Z[r,o]` (for every output `o`) with coefficient `W[i,o]`:

```
dL/dA_prev[r,i] = sum over o of  dZ[r,o] * (∂Z[r,o]/∂A_prev[r,i])
                = sum over o of  dZ[r,o] * W[i,o].
```

That contraction over the output axis `o` is the product of `dZ` (`[B, n_out]`)
with `Wᵀ` (`[n_out, n_in]`):

```
dA_prev = dZ · Wᵀ          shape  [B, n_in] = [B, n_out] · [n_out, n_in]
```

Implemented by **[`launch_gemm`]** with `transB=true`: the contract reads
`B_log[k,n] = B[n*K + k]` when `transB` is set, logically transposing the stored
`W` (`[n_in, n_out]`) into `[n_out, n_in]` on the fly. Call shape:
`launch_gemm(L.dZ, L.W, prevL.dA, M=B, N=n_in, K=n_out, transA=false,
transB=true)`. Note `n_in` of layer `l` equals `n_out` of layer `l-1`, so the
shape `[B, n_in]` exactly matches the previous layer's activation shape.

### 4.4 Through the ReLU — dZ_prev = dA_prev ⊙ relu'(Z_prev)

The previous (hidden) layer applied `A_prev = relu(Z_prev)` elementwise. The
derivative of ReLU is the step function:

```
relu'(z) = 1 if z > 0,   else 0.
```

So the upstream gradient passes straight through where the pre-activation was
positive and is zeroed where it was not (an elementwise / Hadamard product `⊙`):

```
dZ_prev[i] = dA_prev[i] * relu'(Z_prev[i])
           = (Z_prev[i] > 0) ? dA_prev[i] : 0.
```

Implemented by **[`relu_backward`]** over all `B*n_out_prev` elements as a flat
1-D grid. It reads the cached **pre-activation** `Z_prev` (not `A_prev`) to decide
the gate — which is exactly why the forward pass caches `Z`. (At `z == 0` ReLU is
non-differentiable; we use the common subgradient choice `relu'(0) = 0`, matching
the strict `> 0` test in the kernel.)

The output layer has no ReLU gate: its `dZ` comes directly from §3
(`cross_entropy_grad`), so the softmax/CE pair never needs a separate softmax
Jacobian kernel.

### 4.5 Other activation derivatives — LeakyReLU, Tanh   [added in push 0002]

The backward chain `dZ_prev = dA_prev ⊙ act'(Z_prev)` is identical for *any*
elementwise activation; only `act'` changes. The repo implements three, selected
per network by `Activation` (see `include/mlp.cuh`):

| Activation | forward  `a = act(z)`     | derivative `act'(z)`        | backward reads | kernel                |
| ---------- | ------------------------ | --------------------------- | -------------- | --------------------- |
| ReLU       | `max(0, z)`              | `1 if z>0 else 0`           | **Z** (pre)    | `relu_backward`       |
| LeakyReLU  | `z if z>0 else α·z`      | `1 if z>0 else α`           | **Z** (pre)    | `leaky_relu_backward` |
| Tanh       | `tanh(z)`                | `1 − tanh(z)² = 1 − a²`     | **A** (post)   | `tanh_backward`       |

Two things to internalize:

- **ReLU vs LeakyReLU.** Plain ReLU has a *zero* gradient on the negative side, so
  a unit stuck at `z ≤ 0` for every input receives `act'(z)=0` forever and can
  never update — a "dead" unit. LeakyReLU keeps a small slope `α` (this repo uses
  `α = 0.01`), so `act'(z)=α ≠ 0` on the negative side and the unit can recover.

- **Tanh uses the OUTPUT, not the input.** Because `tanh'(z) = 1 − tanh(z)²`, the
  cheapest correct form reuses the already-computed `a = tanh(z)`. So
  `tanh_backward` consumes the cached *post*-activation `A`, while the ReLU family
  consumes the *pre*-activation `Z`. `mlp_backward` dispatches to the right tensor
  for each layer; passing the wrong one is a classic silent bug — and exactly the
  kind of mistake the gradient check (§7) catches. This is *why* every `Layer`
  caches both `Z` and `A`.

--------------------------------------------------------------------------------
## 5. Backward driver (chaining layers)

`mlp_backward` (in `src/mlp.cu`) walks the layers from the output back to the
input, applying §3 once and then §4 per layer:

```
out = layers[num_layers-1]
// §3: output-layer pre-activation gradient, already scaled by 1/B
launch_cross_entropy_grad(out.A.data, d_labels, out.dZ.data, B, K)

for l from num_layers-1 down to 0:
    L        = layers[l]
    prev_act = (l == 0) ? batch_input : layers[l-1].A           # A_prev for this layer

    // §4.1  dW = A_prevᵀ · dZ        [n_in,n_out] = [n_in,B]·[B,n_out]
    launch_gemm(prev_act.data, L.dZ.data, L.dW.data, L.in, L.out, B, true, false)

    // §4.2  db = colsum(dZ)
    launch_bias_grad(L.dZ.data, L.db.data, B, L.out)

    if l > 0:
        prevL = layers[l-1]
        // §4.3  dA_prev = dZ · Wᵀ     [B,n_in] = [B,n_out]·[n_out,n_in]
        launch_gemm(L.dZ.data, L.W.data, prevL.dA.data, B, L.in, L.out, false, true)
        // §4.4  dZ_prev = dA_prev ⊙ relu'(Z_prev)
        launch_relu_backward(prevL.dA.data, prevL.Z.data, prevL.dZ.data,
                             B * prevL.out_features)
```

Each iteration produces `dW`/`db` for layer `l` (ready for SGD) and, unless `l`
is the input-adjacent layer, the `dZ` for layer `l-1` so the loop can continue.
The recursion `dA_prev → dZ_prev` is the discrete chain rule made literal: §4.3
moves the gradient across the linear map, §4.4 moves it across the nonlinearity.

--------------------------------------------------------------------------------
## 6. Parameter update (SGD)

Because the `1/B` mean was folded in at §3 and carried through every gradient,
the stochastic-gradient-descent step is just a scaled element-wise subtraction,
applied to every `W` and `b` by **[`sgd_update`]**:

```
param[i] -= lr * grad[i]
```

with `lr` the learning rate. No extra `1/B`, no momentum, no weight decay — the
minimal update, deliberately, so the gradient math above is the *only* thing that
determines learning behavior. `mlp_sgd_step` simply calls `launch_sgd_update`
once per weight matrix and once per bias vector.

### 6.1 Momentum (the "heavy ball")   [added in push 0002]

Plain SGD reacts only to the *current* gradient, so it zig-zags across steep,
narrow valleys and crawls along shallow ones. Momentum accumulates a **velocity**
`v` — an exponentially-decayed running sum of gradients — and steps along that:

```
v_t   = μ · v_{t-1} + g_t                 # μ ∈ [0,1), e.g. 0.9; g_t = current grad
θ_t   = θ_{t-1} − lr · v_t                # step along the accumulated velocity
```

Consistent gradient directions reinforce across steps (the ball "picks up speed");
oscillating components partially cancel. With `v_0 = 0`, the first step equals
`−lr·g_1` (plain SGD) and momentum builds from there. Implemented per element by
**[`momentum_update`]** (`optim.cu`), which keeps `v` in a persistent device buffer
the same shape as the parameter.

### 6.2 Adam (adaptive moments)   [added in push 0002]

Adam gives every parameter its **own** step size by tracking two running averages
— the mean of the gradient (1st moment `m`) and the mean of its square (2nd moment
`v`) — then dividing the step by the RMS of recent gradients:

```
m_t = β1·m_{t-1} + (1−β1)·g_t             # 1st moment (β1≈0.9)
v_t = β2·v_{t-1} + (1−β2)·g_t²            # 2nd moment (β2≈0.999)

m̂_t = m_t / (1 − β1ᵗ)                     # bias-corrected 1st moment
v̂_t = v_t / (1 − β2ᵗ)                     # bias-corrected 2nd moment

θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)     # ε≈1e-8 guards the divide
```

Two ideas to take away:

- **Bias correction.** `m` and `v` start at `0`, so for small `t` they are biased
  toward `0` (too small). Dividing by `(1 − βᵗ)` — tiny early, → 1 as `t` grows —
  rescales them into unbiased estimates, so the first few steps aren't artificially
  shrunk. Our `optim_step` computes the two denominators once per step on the host
  (the same scalars for every element) and passes them into **[`adam_update`]**, so
  the kernel needs no `pow`/`t`.
- **Per-parameter adaptivity.** Dividing by `√v̂` normalizes each parameter's step
  by the typical size of its own recent gradients: large-gradient weights are
  damped, small-gradient weights amplified. That is why one global `lr` works
  across very differently-scaled parameters — and why Adam's good `lr` (~`1e-2`
  here) is smaller than SGD's (~`1e-1`). State `m` and `v` live in per-parameter
  device buffers allocated by `optim_create`.

All three optimizers share the SGD-derived fact that the gradients already carry
the `1/B` batch-mean (§3), so none of them re-introduces a `1/B` factor.

--------------------------------------------------------------------------------
## 7. Why the gradient check works (finite differences)

`mlp_grad_check` (in `src/mlp.cu`) validates the analytic backward pass without
any reference framework. For a chosen weight `W[i,o]` it perturbs that single
scalar by `±eps` (central difference, `eps ~ 1e-3`), recomputes the **mean** loss
each time, and forms the numerical derivative:

```
dL/dW[i,o] ≈ ( L(W[i,o] + eps) - L(W[i,o] - eps) ) / (2*eps)
```

The central difference has error `O(eps^2)`, far smaller than the one-sided
`O(eps)` form, so it agrees with the analytic `dW` (from §4.1) to several digits.
A small relative error between the two confirms that the entire chain — softmax,
the §3 identity, the GEMM transposes, the ReLU gate — is implemented correctly.
This is the practical pay-off of deriving everything by hand: the numbers
produced by the kernels must match calculus, and the grad-check proves they do.

--------------------------------------------------------------------------------
## 8. Equation → kernel quick reference

| Step | Equation                                              | Kernel / launcher                    | Source |
| ---- | ----------------------------------------------------- | ------------------------------------ | ------ |
| Fwd  | `Z = A_prev · W`                                       | `gemm_naive` / `launch_gemm`         | §1.1   |
| Fwd  | `Z[r,o] += b[o]`                                       | `add_bias` / `launch_add_bias`       | §1.1   |
| Fwd  | `A = max(0, Z)` (hidden)                               | `relu_forward`                       | §1.2   |
| Fwd  | `A[r,:] = softmax(Z[r,:])` (output)                   | `softmax_rows`                       | §1.2   |
| Loss | `loss_r = -log(max(p[r,y_r],1e-12))`                  | `cross_entropy_loss`                 | §2     |
| Bwd  | `dZ_out = (P - Y)/B`                                   | `cross_entropy_grad`                 | §3     |
| Bwd  | `dW = A_prevᵀ · dZ`                                    | `gemm_naive` (`transA=true`)         | §4.1   |
| Bwd  | `db = colsum_r(dZ)`                                    | `bias_grad`                          | §4.2   |
| Bwd  | `dA_prev = dZ · Wᵀ`                                    | `gemm_naive` (`transB=true`)         | §4.3   |
| Bwd  | `dZ_prev = dA_prev ⊙ relu'(Z_prev)`                  | `relu_backward`                      | §4.4   |
| Upd  | `param -= lr * grad`                                   | `sgd_update`                         | §6     |
| Fwd  | `a = z if z>0 else αz` (LeakyReLU)                     | `leaky_relu_forward`                 | §4.5   |
| Bwd  | `dZ = dA ⊙ (z>0 ? 1 : α)`                              | `leaky_relu_backward`                | §4.5   |
| Fwd  | `a = tanh(z)`                                          | `tanh_forward`                       | §4.5   |
| Bwd  | `dZ = dA ⊙ (1 − a²)`  (uses post-act A)               | `tanh_backward`                      | §4.5   |
| Upd  | `v = μv + g ; θ -= lr·v`  (Momentum)                  | `momentum_update`                    | §6.1   |
| Upd  | Adam moment/bias-corrected step                       | `adam_update`                        | §6.2   |
| Misc | `Σ` over an array (loss/accuracy on GPU)             | `reduce_sum_kernel`                  | —      |
| Misc | per-row `argmax == label`                             | `predictions_correct`                | —      |

Read this table top-to-bottom for a forward pass, then bottom region (Bwd) from
`cross_entropy_grad` upward for a backward pass — that ordering mirrors the
driver loops in `src/mlp.cu`. (The §4.5/§6.1/§6.2 rows and the two Misc kernels
were added in push 0002.)
