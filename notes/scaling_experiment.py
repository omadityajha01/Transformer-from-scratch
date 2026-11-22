"""Show why attention scores are divided by sqrt(d_k).

For random unit-variance q and k, the dot product q.k has standard deviation sqrt(d_k).
Without scaling, softmax over these scores becomes nearly one-hot as d_k grows
(low entropy, max weight near 1). With scaling, the distribution stays the same for every d_k.
"""

import numpy as np

T = 16  # number of keys each query attends over
rng = np.random.default_rng(0)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(axis=-1).mean()


print(f"Maximum possible entropy over {T} keys: ln({T}) = {np.log(T):.3f}\n")
print("| d_k | std(q.k) | entropy (unscaled) | entropy (scaled) | max weight (unscaled) | max weight (scaled) |")
print("|---|---|---|---|---|---|")
for d_k in [4, 16, 64, 256, 1024]:
    q = rng.standard_normal((1000, d_k))
    k = rng.standard_normal((1000, T, d_k))
    scores = np.einsum("nd,ntd->nt", q, k)
    p_raw = softmax(scores)
    p_scaled = softmax(scores / np.sqrt(d_k))
    print(f"| {d_k} | {scores.std():.2f} | {entropy(p_raw):.3f} | {entropy(p_scaled):.3f} "
          f"| {p_raw.max(-1).mean():.3f} | {p_scaled.max(-1).mean():.3f} |")
