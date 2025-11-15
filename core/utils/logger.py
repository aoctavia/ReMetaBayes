# # core/utils/logger.py
# import os, json

# class Logger:
#     def __init__(self, log_dir="results/", filename="metrics.json"):
#         os.makedirs(log_dir, exist_ok=True)
#         self.path = os.path.join(log_dir, filename)
#         self.data = {"episode": [], "reward": [], "uncertainty": [], "kl_divergence": []}

#     def log(self, episode, reward, uncertainty=None, kl=None):
#         self.data["episode"].append(episode)
#         self.data["reward"].append(reward)
#         self.data["uncertainty"].append(uncertainty if uncertainty else 0)
#         self.data["kl_divergence"].append(kl if kl else 0)

#     def save(self):
#         with open(self.path, "w") as f:
#             json.dump(self.data, f, indent=2)
import os
import json
from typing import Any, Dict


class Logger:
    """
    Flexible logger that stores each episode as a dictionary entry.
    Compatible with plot_results.py (expects list-of-dicts).
    """

    def __init__(self, log_dir="results/", filename="metrics.json"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)

        # list of dict entries
        self.data = []

    def log(self, episode: int, reward: float,
            meta_loss: float = None,
            kl: float = None,
            uncertainty: float = None,
            **extra: Dict[str, Any]):
        """
        Log metrics for one episode.
        You can add arbitrary extra fields (e.g. entropy, lr).
        """

        entry = {
            "episode": episode,
            "reward": reward,
            "meta_loss": meta_loss,
            "kl": kl,
            "uncertainty": uncertainty,
        }

        # include extra kwargs (e.g. entropy=..., lr=...)
        entry.update(extra)

        self.data.append(entry)

    def save(self):
        """Save all logged data as a list-of-dicts JSON file."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

        print(f"💾 Metrics saved to {self.path}")
