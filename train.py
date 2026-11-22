"""Training loop: AdamW, gradient clipping, linear warmup + cosine LR decay,
periodic train/val evaluation, best-checkpoint saving and JSON loss logging.

Usage:
    python train.py                    # train the GPT baseline
    python train.py --model bigram     # train the bigram baseline
    python train.py --steps 1000       # shorter run
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch

from model import GPT, BigramModel, GPTConfig

# Training hyperparameters
BATCH_SIZE = 32
MAX_LR = 3e-4           # GPT; the bigram table uses 1e-2 because it has no deep layers to destabilise
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 250
EVAL_BATCHES = 20
SEED = 1337

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    train = torch.from_numpy(np.load("data/train.npy").astype(np.int64))
    val = torch.from_numpy(np.load("data/val.npy").astype(np.int64))
    return train, val


def get_batch(data, block_size, batch_size=BATCH_SIZE):
    """Sample random windows. y is x shifted one position to the right."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)


def get_lr(step, total_steps, max_lr=MAX_LR):
    """Linear warmup to max_lr, then cosine decay to 10% of max_lr."""
    min_lr = 0.1 * max_lr
    if step < WARMUP_STEPS:
        return max_lr * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train, val, block_size):
    """Average loss over EVAL_BATCHES random batches for each split."""
    model.eval()
    out = {}
    for name, data in (("train", train), ("val", val)):
        losses = [model(*get_batch(data, block_size))[1].item() for _ in range(EVAL_BATCHES)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train(cfg, steps, model_type="gpt", checkpoint_path=None):
    """Train a model and return its loss history: {"step": [...], "train": [...], "val": [...]}."""
    set_seed(SEED)
    train_data, val_data = load_data()
    model = (GPT(cfg) if model_type == "gpt" else BigramModel(cfg)).to(DEVICE)
    max_lr = MAX_LR if model_type == "gpt" else 1e-2
    print(f"{model_type}: {sum(p.numel() for p in model.parameters()):,} parameters on {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=WEIGHT_DECAY)
    history = {"step": [], "train": [], "val": []}
    best_val = float("inf")

    for step in range(steps + 1):
        if step % EVAL_INTERVAL == 0 or step == steps:
            losses = estimate_loss(model, train_data, val_data, cfg.block_size)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {get_lr(step, steps, max_lr):.2e}")
            if checkpoint_path and losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg.__dict__}, checkpoint_path)
        if step == steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, steps, max_lr)
        x, y = get_batch(train_data, cfg.block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt", "bigram"], default="gpt")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    checkpoint = "results/model.pt" if args.model == "gpt" else None
    history = train(GPTConfig(), args.steps, args.model, checkpoint)
    with open(f"results/{args.model}_losses.json", "w") as f:
        json.dump(history, f, indent=2)
