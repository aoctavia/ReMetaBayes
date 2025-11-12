import json, numpy as np

path = "results/gridworld/metrics.json"  # bisa ubah ke bandit jika ingin
data = json.load(open(path))
rewards = np.array(data["reward"])

window = 100
avg_first = np.mean(rewards[:window])
avg_last = np.mean(rewards[-window:])
adaptation_gain = avg_last - avg_first

print(f"Initial mean reward: {avg_first:.3f}")
print(f"Final mean reward: {avg_last:.3f}")
print(f"Adaptation gain: {adaptation_gain:.3f}")
