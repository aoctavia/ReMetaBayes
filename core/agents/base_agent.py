# core/agents/base_agent.py
# Tujuan: menyediakan antarmuka dasar yang dapat digunakan ulang oleh agen lain (Bayesian & Meta).
import abc
import torch
import numpy as np

class BaseAgent(abc.ABC):
    """
    Abstract base class for reinforcement learning agents.
    Defines the essential structure: act(), update(), and reset().
    """

    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.total_steps = 0
        self.episode_rewards = []

    @abc.abstractmethod
    def act(self, state, evaluate=False):
        """Select an action given a state."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Perform a training step."""
        raise NotImplementedError

    def reset(self):
        """Optional reset per episode."""
        pass

    def log_episode(self, reward):
        """Record episode reward for evaluation."""
        self.episode_rewards.append(reward)
        self.total_steps += 1

    def get_average_reward(self, last_n=10):
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards[-last_n:])
