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
        """
        Select an action based on current policy logits.
        Exploration is automatically modulated by entropy:
        high uncertainty -> softer sampling (more exploration),
        low uncertainty -> greedy behavior (more exploitation).
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits = self.policy(state)

        # Hitung distribusi awal dan entropinya
        base_probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(base_probs * torch.log(base_probs + 1e-8)).item()

        # 🔥 Entropy-weighted exploration (adaptive temperature)
        temperature = 1.0 + entropy          # semakin besar entropy → lebih eksploratif
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)

        # Sampling dari distribusi dengan temperature
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Jika mode evaluasi, ambil aksi paling mungkin (greedy)
        if evaluate:
            action = torch.argmax(probs)

        return action.item(), log_prob, entropy

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
