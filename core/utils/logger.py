# core/utils/logger.py
import os, json

class Logger:
    def __init__(self, log_dir="results/", filename="metrics.json"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.data = {"episode": [], "reward": [], "uncertainty": [], "kl_divergence": []}

    def log(self, episode, reward, uncertainty=None, kl=None):
        self.data["episode"].append(episode)
        self.data["reward"].append(reward)
        self.data["uncertainty"].append(uncertainty if uncertainty else 0)
        self.data["kl_divergence"].append(kl if kl else 0)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
