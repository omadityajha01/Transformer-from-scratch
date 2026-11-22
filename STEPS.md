# Implementation Steps

This document records every step performed to implement `plan.md`, in the order performed. Each step lists its sub-steps, followed by an explanation.

---

## Step 1. Environment setup

**Sub-steps**
1. Checked the installed software: Python 3.12.10, NumPy 1.26.0 and Matplotlib 3.9.2 were present; PyTorch was not.
2. Created a virtual environment, `venv/`, inside the project folder.
3. Installed the CPU build of PyTorch, then NumPy, Matplotlib and pytest, into the virtual environment.
4. Wrote `requirements.txt` containing `torch`, `numpy`, `matplotlib` and `pytest`.
5. Created the folders `data/`, `tests/`, `results/` and `notes/`.

**Explanation**
A virtual environment keeps the project's libraries separate from the system Python, so the project can be reproduced on another machine with `pip install -r requirements.txt`. The CPU build of PyTorch was chosen because the machine has no CUDA GPU. `tqdm`, listed in the plan, was left out because no script uses it. `pytest` was added because the tests require it.

---

## Step 2. Dataset

**Sub-steps**
1. Downloaded TinyShakespeare from Karpathy's `char-rnn` repository to `data/input.txt`.
2. Confirmed the size: 1,115,394 characters.

**Explanation**
TinyShakespeare is a single plain-text file of Shakespeare's plays. It is small enough to train on a laptop in under an hour, and its structure (character names, line breaks, verse) makes it easy to judge the generated text by eye.

---

## Step 3. BPE tokeniser (`tokenizer.py`)

**Sub-steps**
1. Wrote `get_pair_counts(ids)`, which counts every adjacent pair of token ids.
2. Wrote `merge(ids, pair, new_id)`, which replaces every occurrence of a pair with a new id.
3. Wrote the `BPETokenizer` class:
   - `train(text, vocab_size)`: starts from the 256 byte values and repeatedly merges the most frequent pair, until the vocabulary reaches `vocab_size`.
   - `encode(text)`: converts text to bytes, then applies the learned merges in the order they were learned.
   - `decode(ids)`: maps each id back to its bytes and decodes the result as UTF-8.
   - `save(path)` and `load(path)`: store and restore the merges as JSON.
4. Ran `python tokenizer.py`, which:
   - learned 256 merges (vocabulary size 512) from the full text;
   - saved the merges to `data/tokenizer.json`;
   - encoded the text into 568,210 tokens (1.96 characters per token);
   - split the tokens 90/10 into `data/train.npy` (511,389 tokens) and `data/val.npy` (56,821 tokens).
5. The run took 2 minutes 28 seconds.

**Explanation**
A model cannot read text directly; it reads integer ids. BPE builds a vocabulary by merging the most common adjacent pairs, so frequent sequences such as `the` or `king` become single tokens. Because it starts from raw bytes, any string can be encoded, including emojis and accented letters, and decoding always returns the original text. Encoding the dataset once and caching it as `.npy` files avoids repeating this slow, pure-Python step for every training run.

---

## Step 4. Model (`model.py`)

**Sub-steps**
1. Defined `GPTConfig`: `vocab_size=512, block_size=128, d_model=128, n_head=4, n_layer=4, dropout=0.1`, plus three ablation flags, `use_pos_emb`, `use_layernorm` and `use_residual`, all `True` by default.
2. Wrote `BigramModel`: a single lookup table that predicts the next token from the current token only.
3. Wrote `LayerNorm` by hand: mean and variance over the last dimension, a learnable scale `gamma`, a learnable shift `beta`, and `eps`.
4. Wrote `CausalSelfAttention`:
   - one linear layer producing Q, K and V together;
   - a reshape and transpose that split them into 4 heads of size 32;
   - scores `QKᵀ / √d_k`;
   - a lower-triangular mask that sets future positions to `−∞`;
   - softmax, dropout and multiplication by V;
   - concatenation of the heads and an output projection.
5. Wrote `FeedForward`: `Linear(128, 512) → GELU → Linear(512, 128) → Dropout`.
6. Wrote `Block` (pre-norm): `x = x + attn(ln1(x))`, then `x = x + ffn(ln2(x))`. The flags replace LayerNorm with an identity function, or remove the `x +` terms.
7. Wrote `GPT`: token embedding + positional embedding → 4 blocks → final LayerNorm → output layer. Weights are initialised from a normal distribution with standard deviation 0.02.
8. Wrote `GPT.generate()`: repeatedly predicts the next token, with temperature and top-k sampling.

**Explanation**
The model is a decoder-only transformer, the same design as GPT. Attention lets each position gather information from earlier positions; the mask stops it from seeing later positions, which it could otherwise copy during training. LayerNorm and residual connections keep training stable. The bigram model has no context at all and gives a minimum standard the transformer must beat. The flags allow each ablation to be run with the same code, changing exactly one component.

---

## Step 5. Tests (`tests/test_shapes.py`)

**Sub-steps**
1. Wrote seven tests:
   - BPE round-trip: `decode(encode(s)) == s` for strings including newlines, tabs, emojis and accented letters; also checks that saving and loading preserve the merges;
   - hand-written LayerNorm equals `torch.nn.functional.layer_norm`;
   - attention output shape;
   - GPT logits shape and scalar loss;
   - causal mask: changing token `t+1` leaves the outputs at positions `0..t` unchanged;
   - each ablation config runs;
   - `generate()` returns the requested length.
2. Ran `python -m pytest tests/`: 7 passed.
3. Counted the parameters with `sum(p.numel())`: 940,800, which equals the formula derived in `notes/MATH.md`.

**Explanation**
The tests confirm the components are correct before any training time is spent. The causal-mask test is the most important: a leaking mask produces a deceptively low training loss and a model that fails at generation.

---

## Step 6. Training script (`train.py`)

**Sub-steps**
1. Set the hyperparameters: batch size 32, AdamW with maximum learning rate `3e-4` and weight decay 0.1, gradient clipping at 1.0, evaluation every 250 steps over 20 batches, seed 1337.
2. Wrote `get_batch()`: samples random 128-token windows; the target `y` is the input `x` shifted by one token.
3. Wrote `get_lr()`: linear warmup over 200 steps, then cosine decay to 10% of the maximum.
4. Wrote `estimate_loss()`: average train and validation loss with dropout disabled.
5. Wrote `train(cfg, steps, model_type, checkpoint_path)`: the loop; it saves the checkpoint whenever validation loss improves and returns the loss history.
6. Set the bigram model's maximum learning rate to `1e-2`.
7. Added command-line options `--model {gpt, bigram}` and `--steps`.

**Explanation**
Warmup avoids large, unstable updates at the start, when the optimiser's statistics are unreliable. Cosine decay reduces the step size later so the model settles. Clipping caps the size of any single update. A fixed seed makes every run reproducible. The bigram model is a single table with no deep layers, so it can use a much higher learning rate; at `3e-4` it would learn too slowly in 5,000 steps, and the baseline would be unfairly weak. `train()` is a function so that `ablations.py` can reuse it.

---

## Step 7. Training runs

**Sub-steps**
1. Trained the bigram model for 2,000 steps as a trial: validation loss 3.943 in 19 seconds.
2. Trained the bigram model for 5,000 steps: validation loss **3.889**. Saved `results/bigram_losses.json`.
3. Trained the GPT model for 5,000 steps on CPU, in 36 minutes 39 seconds:

   | Step | Train loss | Val loss |
   |---|---|---|
   | 0 | 6.255 | 6.252 |
   | 1000 | 3.476 | 3.739 |
   | 2000 | 3.069 | 3.417 |
   | 3000 | 2.875 | 3.234 |
   | 4000 | 2.753 | 3.183 |
   | 5000 | 2.718 | **3.169** |

   Saved `results/model.pt` and `results/gpt_losses.json`.
4. Wrote `generate.py` and ran it. The output has Shakespeare's format (character-style lines, line breaks, verse rhythm) with some misspelled words.

**Explanation**
The starting loss of 6.25 equals `ln 512`, the loss of random guessing over 512 tokens, which confirms the initialisation is neutral. The GPT model ends well below the bigram baseline (3.169 against 3.889), so it is using context. These losses are per token. Since one token is 1.96 characters, 3.169 per token is about 1.62 per character, inside the range the plan expected.

---

## Step 8. Mathematics notes (`notes/MATH.md`)

**Sub-steps**
1. Wrote `notes/scaling_experiment.py`, a NumPy experiment that measures softmax entropy with and without √d_k scaling, for `d_k` from 4 to 1024.
2. Ran it and recorded the results.
3. Wrote `notes/MATH.md`, covering:
   - the forward pass with tensor shapes;
   - the attention formula;
   - the derivation of `Var(q·k) = d_k` and why it requires √d_k scaling;
   - the causal mask, multi-head attention, positional embeddings, residual connections, LayerNorm and the learning-rate schedule;
   - the parameter-count formula.

**Explanation**
Without scaling, the scores grow with `d_k` and softmax becomes nearly one-hot. At `d_k = 1024` the largest weight averages 0.964 and the entropy falls to 0.094. With scaling, the entropy stays at about 2.36 for every `d_k`. The experiment demonstrates the derivation numerically.

---

## Step 9. Ablations (`ablations.py`, `plot.py`)

**Sub-steps**
1. Wrote `ablations.py`: trains four configs with the same seed, steps and hyperparameters, and saves all loss histories to `results/ablations.json`.
2. Wrote `plot.py`: saves the validation-loss curves to `results/ablation_val_loss.png` and a bar chart of the final losses to `results/ablation_final_loss.png`.
3. First attempt, on CPU:
   - Training ran much slower than expected; the laptop likely slept or throttled.
   - Claude Code stopped the process because the system ran low on memory, at step 2,250 of the first run. No results were saved.
   - Up to that point, the losses matched the earlier GPT run exactly, which confirmed the runs are reproducible.
4. Second attempt, on Google Colab (T4 GPU):
   - Packed the four scripts and the encoded data into `colab_upload.zip`. The first version, made with PowerShell, stored paths with backslashes, which do not extract correctly on Linux, so it was rebuilt with Python's `zipfile`.
   - The user ran `ablations.py` and `plot.py` on Colab and copied the three result files into `results/`.
5. Results:

   | Run | Final train loss | Final val loss |
   |---|---|---|
   | Baseline | 2.725 | **3.160** |
   | No positional encoding | 3.050 | 3.448 |
   | No LayerNorm | 3.687 | 3.928 |
   | No residuals | 5.362 | 5.395 |

6. Computed the loss of a unigram model, which predicts each token's overall frequency: 5.389.
7. Deleted `results/ablations.log`, the incomplete log from the stopped CPU attempt.

**Explanation**
Each ablation removes one component and measures the damage:
- **No positional encoding:** trains, but plateaus higher, because attention cannot tell word order.
- **No LayerNorm:** learns more slowly and ends worse than the bigram baseline.
- **No residuals:** stops improving at the unigram level (5.395 against 5.389). The network learns only token frequencies, because gradients cannot pass through all the layers without the identity path.

The Colab baseline (3.160) differs slightly from the CPU run (3.169) because GPU arithmetic differs slightly from CPU arithmetic. All four ablations ran on the same GPU, so they are directly comparable.

---

## Step 10. README and repository files

**Sub-steps**
1. Updated `README.md`:
   - removed the "Work in progress" banner;
   - filled in the parameter count (940,800);
   - added the training-results table and the sample output;
   - added the ablation table, an explanation of each result, and both plots;
   - added `notes/scaling_experiment.py` to the project structure.
2. Removed the resume bullet points that had been pasted at the end of `README.md`.
3. Left "What I learned" for the author to write.
4. Wrote `.gitignore`, excluding `venv/`, `__pycache__/`, `.pytest_cache/`, `data/*.npy`, `results/*.pt` and `colab_upload.zip`.
5. Re-ran the tests after all changes: 7 passed.

**Explanation**
The README now reports measured results rather than placeholders. The `.gitignore` keeps the repository small, since the virtual environment, the checkpoint and the encoded data can all be regenerated from the code.

---

## Final project structure

```
Transformer From Scratch/
├── data/
│   ├── input.txt                 # TinyShakespeare
│   ├── tokenizer.json            # learned BPE merges
│   ├── train.npy, val.npy        # encoded dataset (90/10 split)
├── tokenizer.py
├── model.py
├── train.py
├── generate.py
├── ablations.py
├── plot.py
├── tests/test_shapes.py
├── results/
│   ├── model.pt                  # best GPT checkpoint
│   ├── bigram_losses.json, gpt_losses.json, ablations.json
│   ├── ablation_val_loss.png, ablation_final_loss.png
├── notes/
│   ├── MATH.md
│   ├── scaling_experiment.py
├── requirements.txt
├── .gitignore
├── plan.md
├── README.md
└── STEPS.md
```

## Remaining items (for the author)

1. Write the "What I learned" section of `README.md`.
2. Initialise a Git repository and push the project to GitHub.
3. Delete `colab_upload.zip`, which is no longer needed.
4. Prepare for interviews using `notes/MATH.md`.
