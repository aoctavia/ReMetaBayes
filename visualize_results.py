# visualize_results.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Path hasil training
METRICS_PATH = "results/gridworld/metrics.json"

if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

# Baca data
with open(METRICS_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print("Loaded", len(df), "episodes.")
print(df.head())

# Reward curve
plt.figure(figsize=(8,4))
sns.lineplot(x="episode", y="reward", data=df, color="tab:blue")
plt.title("Average Reward over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/reward_curve.png", dpi=200)
plt.show()

# Uncertainty curve
if "uncertainty" in df:
    plt.figure(figsize=(8,4))
    sns.lineplot(x="episode", y="uncertainty", data=df, color="tab:orange")
    plt.title("Uncertainty Evolution")
    plt.xlabel("Episode")
    plt.ylabel("Posterior Variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/uncertainty_evolution.png", dpi=200)
    plt.show()

print("✅ Plots saved in results/plots/")



data = json.load(open("results/gridworld/metrics.json"))
rewards = data["reward"]
print("Average reward:", sum(rewards)/len(rewards))
print("Best reward:", max(rewards))
print("Worst reward:", min(rewards))
