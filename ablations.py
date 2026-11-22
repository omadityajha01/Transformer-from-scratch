"""Train each ablation config with the same seed, steps and hyperparameters,
and save the loss history of each run to results/ablations.json.

Usage:
    python ablations.py                # 5000 steps per run
    python ablations.py --steps 2000
"""

import argparse
import json
import os

from model import GPTConfig
from train import train

RUNS = {
    "Baseline": {},
    "No positional encoding": {"use_pos_emb": False},
    "No LayerNorm": {"use_layernorm": False},
    "No residuals": {"use_residual": False},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    results = {}
    for name, overrides in RUNS.items():
        print(f"\n=== {name} ===")
        results[name] = train(GPTConfig(**overrides), args.steps)

    with open("results/ablations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n| Run | Final train loss | Final val loss |\n|---|---|---|")
    for name, h in results.items():
        print(f"| {name} | {h['train'][-1]:.4f} | {h['val'][-1]:.4f} |")
