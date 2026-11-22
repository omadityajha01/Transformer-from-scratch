# Transformer From Scratch: 4-Week Student Plan

## Context
You want to build the project described on your resume: a decoder-only transformer in raw PyTorch with a BPE tokeniser, ablations and a write-up on the maths. The plan has to be something one college student can finish in **under 4 weeks** alongside classes, at roughly **2–3 hrs on weekdays and 5–6 hrs on weekends (about 70–80 hrs total)**. It also has to be simple enough to explain line by line in a placement interview. The folder `C:\Projects\Transformer From Scratch` is empty right now, so everything starts from zero.

**Keep the scope realistic.** Use a small model (about 1–5M params) trained on **TinyShakespeare (~1MB)**. It trains in minutes on a laptop CPU or a free Google Colab T4 GPU. Nobody expects GPT quality from this. Interviewers care that you understand every line.

"Raw PyTorch" means you may use `torch.nn.Linear`, `nn.Embedding`, `nn.Parameter`, autograd and `torch.optim.AdamW`. You don't use `nn.Transformer`, `nn.MultiheadAttention`, `nn.LayerNorm` or HuggingFace. You write attention, LayerNorm and the tokeniser yourself.

---

## Folder structure (final)
```
Transformer From Scratch/
├── data/input.txt              # TinyShakespeare
├── tokenizer.py                # BPE from scratch
├── model.py                    # embeddings, attention, FFN, LayerNorm, blocks, GPT
├── train.py                    # training loop + LR schedule + checkpoints
├── generate.py                 # sample text from a checkpoint
├── ablations.py                # runs all ablation configs, saves losses
├── plot.py                     # matplotlib loss curves
├── tests/test_shapes.py        # small sanity checks (causal mask, shapes, BPE round-trip)
├── results/                    # loss .json files + .png plots
├── notes/MATH.md               # the maths write-up (sqrt(d_k), etc.)
├── requirements.txt            # torch, numpy, matplotlib, tqdm
└── README.md
```

---

## Week 1: Foundations + BPE tokeniser (≈15 hrs)
**Days 1–2: Learn before coding (don't skip this)**
- Watch Karpathy's *"Let's build GPT"* and *"Let's build the GPT Tokenizer"*. Take your own notes; don't copy his code.
- Skim the "Attention Is All You Need" paper (sections 3.1–3.5 only).
- Set up: Python venv, `pip install torch numpy matplotlib tqdm`, `git init`, first commit.

**Days 3–6: `tokenizer.py` (BPE from scratch)**
- Start at the byte level (256 base tokens), count adjacent pair frequencies, and merge the most frequent pair repeatedly.
- `train(text, vocab_size=512)`, `encode(str) -> list[int]`, `decode(list[int]) -> str`, `save/load` merges to JSON.
- Test: `decode(encode(s)) == s` for random strings, including emojis and newlines.
- Encoding can be slow at first. That's fine, just cache the encoded dataset as a `.npy` file with NumPy.

**Day 7: Buffer / catch up.** Write a short README section on "how BPE works".

✅ Milestone: tokeniser round-trips correctly, and the dataset is encoded to `data/train.npy`, `data/val.npy` (90/10 split).

---

## Week 2: The model (≈20 hrs)
Build bottom-up and test each piece's output shape before moving on.

1. **Data loader**: random `block_size` windows → `(x, y)` where y is x shifted by 1.
2. **Bigram baseline** (1 evening): just an embedding table. It gives you a working train loop early and a baseline loss to beat.
3. **Token + positional embeddings**: learned `nn.Embedding(block_size, d_model)`.
4. **Single-head causal self-attention**: Q, K, V projections, `scores = QKᵀ / sqrt(d_k)`, a lower-triangular mask filled with `-inf`, softmax, dropout, then `@ V`.
5. **Multi-head attention**: split into `n_head` heads with reshape/transpose (the efficient way), concatenate, then the output projection.
6. **LayerNorm by hand**: mean/var over the last dim, learnable `gamma`/`beta`, and `eps`. Compare against `torch.nn.functional.layer_norm` in a test to prove it matches.
7. **Feed-forward**: `Linear(d, 4d) → GELU → Linear(4d, d) → dropout`.
8. **Block (pre-norm)**: `x = x + attn(ln1(x)); x = x + ffn(ln2(x))`.
9. **GPT class**: embeddings → N blocks → final LN → LM head. Add a `generate()` with temperature and top-k.
10. Add **config flags** now to make Week 3 easy: `use_pos_emb`, `use_layernorm`, `use_residual` (all default True).

Starter config: `d_model=128, n_head=4, n_layer=4, block_size=128, dropout=0.1`, about 1M params.

✅ Milestone: `tests/test_shapes.py` passes. Changing token t+1 must not change the output at position t, which proves the causal mask works.

---

## Week 3: Training + ablations (≈20 hrs)
**Days 1–3: `train.py`**
- AdamW, `lr=3e-4`, weight decay 0.1, and gradient clipping at 1.0.
- **LR schedule written by hand**: linear warmup (~200 steps) then cosine decay to 10% of max LR.
- Evaluate train/val loss every N steps (averaged over a few batches), save a checkpoint with the best val loss, and log losses to JSON.
- Seed everything so runs are reproducible.
- Train the baseline for ~5,000 steps (CPU: ~30–60 min; Colab T4: ~5–10 min).
- `generate.py`: produce Shakespeare-ish text. Screenshot it for the README.

**Days 4–6: `ablations.py` + `plot.py`**
Run 4 configs with the **same seed, steps and hyperparameters**:
| Run | Change | What you'll likely see |
|---|---|---|
| Baseline | all on | lowest val loss |
| No positional encoding | `use_pos_emb=False` | worse loss, model can't tell word order (attention becomes permutation-invariant) |
| No LayerNorm | `use_layernorm=False` | unstable/slower training, may need lower LR |
| No residuals | `use_residual=False` | much slower or stuck training, since gradients vanish through depth |
- Optional bonus (if time): **no √d_k scaling**. It shows softmax saturation, which ties straight into the maths notes.
- `plot.py`: one figure with all val-loss curves, plus a bar chart of final loss. Save the PNGs to `results/`.

**Day 7: Buffer.** Something will break or take longer, so this day is reserved for that.

✅ Milestone: `results/ablation_val_loss.png` exists and the table of final losses is filled in.

---

## Week 4: Documentation + interview prep (≈15 hrs)
**`notes/MATH.md`** (write it in your own words, it's your interview cheat sheet):
- Scaled dot-product attention formula and shapes at every step.
- **Why divide by √d_k**: if q and k have unit-variance entries, then q·k has variance d_k. Large logits push softmax into near one-hot regions where gradients are ≈ 0. Dividing by √d_k brings the variance back to 1. Back this up with a small NumPy experiment showing softmax entropy with and without scaling.
- Why a causal mask, why multi-head, why residuals help gradient flow, and what LayerNorm normalises.
- Parameter count formula for your model, checked against `sum(p.numel())`.

**README.md**: what it is, how to run it, sample output, ablation plot + table, key learnings, and a "what I'd do next" section (RoPE, KV-cache, bigger data).

**Clean-up**: docstrings, remove dead code, push to GitHub with a clean commit history (commit throughout the month, not all on the last day).

**Interview prep (last 2–3 days)**: practise explaining these out loud:
- Walk through a forward pass with tensor shapes.
- Why √d_k? What happens without positional encodings?
- Pre-norm vs post-norm; why warmup helps.
- What went wrong while building it and how you fixed it (keep a short "bugs log" during the month for this).

---

## If you fall behind (priority order to cut)
1. Drop the √d_k bonus ablation.
2. Use a smaller vocab (256–512) and train BPE on a subset of the text.
3. Use Colab instead of CPU to save training time.
4. Never cut: the working model, the 3 core ablations, the plot and MATH.md. Those are the resume claims.

---

## Verification (how you know it's done)
- `python -m pytest tests/`: BPE round-trip, causal-mask leak test, hand-written LayerNorm matches PyTorch's, output shapes are correct.
- `python train.py`: val loss drops clearly below the bigram baseline (~2.5 → ~1.5–1.8 range with BPE-512; exact numbers vary).
- `python generate.py`: readable Shakespeare-style text with character names and line breaks.
- `python ablations.py && python plot.py`: the plot shows the baseline best and the no-residual/no-LN runs clearly worse.
- Every resume bullet maps to a file you can open and explain in an interview.
