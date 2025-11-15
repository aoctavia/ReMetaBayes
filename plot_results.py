#!/usr/bin/env python3
# =========================================================
# ReMetaBayes — Plot Training Metrics
# =========================================================

import json
import os
import matplotlib.pyplot as plt

RESULTS_PATH = "results_meta/remetabayes_meta_metrics.json"


def load_metrics(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Metrics file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    episodes = []
    rewards = []
    meta_loss = []
    kl = []

    for entry in data:
        episodes.append(entry.get("episode", 0))
        rewards.append(entry.get("reward", 0.0))
        meta_loss.append(entry.get("meta_loss", None))
        kl.append(entry.get("kl", None))

    return episodes, rewards, meta_loss, kl


def plot_curve(x, y, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=ylabel, color="blue")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📈 Saved: {filename}")
    plt.close()


def main():
    print("📥 Loading metrics...")
    episodes, rewards, meta_loss, kl = load_metrics(RESULTS_PATH)

    # Plot Reward Curve
    plot_curve(
        episodes,
        rewards,
        "Mean Query Reward over Meta-Iterations",
        "Mean Reward",
        "reward_curve.png",
    )

    # Plot Meta-Loss (if logged)
    if any(m is not None for m in meta_loss):
        plot_curve(
            episodes,
            [m if m is not None else 0 for m in meta_loss],
            "Meta-Loss over Meta-Iterations",
            "Meta-Loss",
            "meta_loss_curve.png",
        )
    else:
        print("⚠️ No meta_loss field in metrics — skipping meta-loss plot.")

    # Plot KL (if logged)
    if any(k is not None for k in kl):
        plot_curve(
            episodes,
            [k if k is not None else 0 for k in kl],
            "KL(z) over Meta-Iterations",
            "KL Divergence",
            "kl_curve.png",
        )
    else:
        print("⚠️ No KL field in metrics — skipping KL plot.")

    print("🎉 All plots completed.")


if __name__ == "__main__":
    main()
