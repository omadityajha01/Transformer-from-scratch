# The Mathematics of the Model

Notation: `B` batch size, `T` sequence length, `C = d_model`, `h = n_head`, `d_k = C / h`, `V` vocabulary size.
Configuration used: `V = 512, T = 128, C = 128, h = 4, d_k = 32, n_layer = 4`.

## 1. Forward pass with shapes

| Step | Operation | Shape |
|---|---|---|
| Input | token ids | `(B, T)` |
| Token embedding | `E_tok[idx]` | `(B, T, C)` |
| Positional embedding | `E_pos[0..T-1]`, added | `(B, T, C)` |
| Q, K, V | `x W_q, x W_k, x W_v` | `(B, T, C)` each |
| Split heads | reshape and transpose | `(B, h, T, d_k)` |
| Scores | `Q Kᵀ / √d_k` | `(B, h, T, T)` |
| Causal mask + softmax | over the last dimension | `(B, h, T, T)` |
| Weighted values | `A V` | `(B, h, T, d_k)` |
| Concatenate heads + projection | `concat(...) W_o` | `(B, T, C)` |
| Feed-forward | `W_2 GELU(W_1 x)` | `(B, T, 4C) → (B, T, C)` |
| Final LayerNorm + LM head | `x W_head` | `(B, T, V)` |
| Loss | cross-entropy against the next token | scalar |

## 2. Scaled dot-product attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

`M` is the causal mask: `M[i, j] = 0` if `j ≤ i` and `−∞` otherwise. Since `exp(−∞) = 0`, position `i` assigns zero weight to every later position. Each row of the attention matrix is a probability distribution over the positions the query may attend to, and the output is the corresponding weighted average of value vectors.

## 3. Why divide by √d_k

Let the components of `q` and `k` be independent with mean 0 and variance 1. Then

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \qquad \mathbb{E}[q_i k_i] = 0, \qquad \text{Var}(q_i k_i) = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1,$$

so `Var(q · k) = d_k` and its standard deviation is `√d_k`. As `d_k` grows the scores spread out, and softmax assigns almost all weight to the largest score. In that regime the softmax Jacobian, `∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ − pⱼ)`, is close to zero everywhere because each `pᵢ` is close to 0 or 1, so gradients through attention vanish. Dividing by `√d_k` gives the scores variance 1 for every `d_k`.

Experiment (`python notes/scaling_experiment.py`, 1000 random queries over 16 keys; maximum entropy is ln 16 = 2.773):

| d_k | std(q·k) | entropy (unscaled) | entropy (scaled) | max weight (unscaled) | max weight (scaled) |
|---|---|---|---|---|---|
| 4 | 2.01 | 1.764 | 2.379 | 0.424 | 0.235 |
| 16 | 3.98 | 0.899 | 2.363 | 0.685 | 0.245 |
| 64 | 8.06 | 0.400 | 2.351 | 0.850 | 0.249 |
| 256 | 16.04 | 0.180 | 2.359 | 0.929 | 0.247 |
| 1024 | 31.79 | 0.094 | 2.361 | 0.964 | 0.245 |

Without scaling, the distribution collapses towards one-hot as `d_k` grows. With scaling it does not depend on `d_k`.

## 4. Causal mask

The model is trained to predict token `t+1` from tokens `0..t`. If position `t` could attend to `t+1`, it could copy the answer, and the model would fail at generation time when future tokens do not exist. The mask makes training on all `T` positions in parallel equivalent to `T` separate next-token predictions. `tests/test_shapes.py` checks that changing token `t+1` leaves the outputs at positions `≤ t` unchanged.

## 5. Multi-head attention

A single softmax produces one weighted average per position. Splitting `C` into `h` heads of size `d_k = C/h` lets each head use its own projections, so different heads can attend to different positions for different reasons. The total cost is the same as one head of size `C`.

## 6. Positional embeddings

Attention without positions is permutation-equivariant: permuting the input tokens permutes the outputs in the same way, because `QKᵀ` depends only on token contents. The model then cannot distinguish word order. Adding a learned vector `E_pos[t]` to each token embedding makes position part of the input.

## 7. Residual connections

A block computes `y = x + F(x)`, so

$$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}.$$

The identity term gives the gradient a direct path from the loss to every earlier layer. Without it, the gradient is a product of `n_layer` Jacobians and can shrink or grow geometrically with depth.

## 8. Layer normalisation

For each vector `x ∈ ℝ^C`, independently for every token:

$$\mu = \frac{1}{C}\sum_i x_i, \qquad \sigma^2 = \frac{1}{C}\sum_i (x_i - \mu)^2, \qquad \text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta.$$

It normalises over the feature dimension, not the batch, so it behaves identically during training and generation. It keeps the inputs to each sub-layer on a fixed scale and so stabilises training. This model uses **pre-norm** (`x + F(LN(x))`), which keeps the residual path free of normalisation and trains more stably than post-norm (`LN(x + F(x))`).

## 9. Learning-rate schedule

- **Warmup** (linear, 200 steps): at initialisation the Adam second-moment estimates are based on few samples and the gradients are large and noisy, so large early updates can push the model into a poor region. Ramping the learning rate up avoids this.
- **Cosine decay** to 10% of the maximum: smaller steps late in training let the parameters settle into a minimum.

## 10. Parameter count

Per transformer block:

| Component | Parameters |
|---|---|
| 2 × LayerNorm | `4C` |
| QKV projection | `3C² + 3C` |
| Output projection | `C² + C` |
| FFN layer 1 | `4C² + 4C` |
| FFN layer 2 | `4C² + C` |
| **Total** | `12C² + 13C` |

Whole model: `VC` (token embedding) + `TC` (positional embedding) + `n_layer(12C² + 13C)` + `2C` (final LayerNorm) + `VC` (LM head, no bias).

With `V = 512, T = 128, C = 128, n_layer = 4`:

`65,536 + 16,384 + 4 × 198,272 + 256 + 65,536 = 940,800`

This matches `sum(p.numel() for p in model.parameters())`.
