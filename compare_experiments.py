import json
import matplotlib.pyplot as plt

# Baca dua hasil eksperimen
grid = json.load(open("results/gridworld/metrics.json"))
bandit = json.load(open("results/bandit/metrics.json"))

plt.figure(figsize=(8,4))
plt.plot(grid["episode"], grid["reward"], label="GridWorld", alpha=0.7)
plt.plot(bandit["episode"], bandit["reward"], label="ContextualBandit", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Comparison — GridWorld vs Bandit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# simpan hasil
import os
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/comparison.png", dpi=200)
plt.show()
