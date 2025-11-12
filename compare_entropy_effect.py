#!/usr/bin/env python3
# =========================================================
# ReMetaBayes — Comparison Plot: Before vs After Entropy-Weighted Exploration
# Author: Aulia Octaviani
# =========================================================

import json
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1️⃣ Load JSON metrics (Before & After Entropy-weighted exploration)
# ---------------------------------------------------------

BASE_BEFORE = "results/gridworld_baseline/metrics.json"   # ganti jika path berbeda
BASE_AFTER = "results/gridworld/metrics.json"              # hasil training terbaru (entropy version)

def load_metrics(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data

before = load_metrics(BASE_BEFORE)
after = load_metrics(BASE_AFTER)

# ---------------------------------------------------------
# 2️⃣ Plot Reward Comparison
# ---------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(before["episode"], before["reward"], label="Before (No Entropy-Weight)", alpha=0.6, color="tab:gray")
plt.plot(after["episode"], after["reward"], label="After (Entropy-Weighted)", alpha=0.8, color="tab:blue")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Comparison — Before vs After Entropy-Weighted Exploration")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/compare_reward_entropy.png", dpi=200)
plt.show()

# ---------------------------------------------------------
# 3️⃣ Plot Uncertainty Comparison
# ---------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(before["episode"], before["uncertainty"], label="Before", alpha=0.6, color="tab:orange")
plt.plot(after["episode"], after["uncertainty"], label="After", alpha=0.8, color="tab:red")
plt.xlabel("Episode")
plt.ylabel("Posterior Variance (Entropy)")
plt.title("Uncertainty Evolution — Before vs After Entropy-Weighted Exploration")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/compare_uncertainty_entropy.png", dpi=200)
plt.show()

# ---------------------------------------------------------
# 4️⃣ Print simple comparison summary
# ---------------------------------------------------------
print("✅ Comparison plots saved in results/plots/")
print(f"Before run: {len(before['reward'])} episodes | mean reward: {sum(before['reward'])/len(before['reward']):.3f}")
print(f"After  run: {len(after['reward'])} episodes | mean reward: {sum(after['reward'])/len(after['reward']):.3f}")
