"""Plot the ablation results from results/ablations.json.

Saves:
    results/ablation_val_loss.png    validation loss curves for every run
    results/ablation_final_loss.png  bar chart of the final validation loss
"""

import json

import matplotlib.pyplot as plt

with open("results/ablations.json") as f:
    results = json.load(f)

plt.figure(figsize=(8, 5))
for name, h in results.items():
    plt.plot(h["step"], h["val"], label=name)
plt.xlabel("Step")
plt.ylabel("Validation loss")
plt.title("Ablations: validation loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/ablation_val_loss.png", dpi=150)

names = list(results)
finals = [results[n]["val"][-1] for n in names]
plt.figure(figsize=(8, 5))
bars = plt.bar(names, finals)
plt.bar_label(bars, fmt="%.3f")
plt.ylabel("Final validation loss")
plt.title("Ablations: final validation loss")
plt.tight_layout()
plt.savefig("results/ablation_final_loss.png", dpi=150)

print("Saved results/ablation_val_loss.png and results/ablation_final_loss.png")
