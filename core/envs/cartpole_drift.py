# core/envs/cartpole_drift.py
# Versi OpenAI Gym CartPole dengan parameter fisika yang berubah dinamis.
# Cocok untuk eksperimen “policy transfer” — agent harus beradaptasi pada perubahan gravitasi, massa, dan friction.
import gym
import numpy as np

class CartPoleDrift:
    """
    Modified CartPole environment with drifting dynamics.
    The pole length and gravity vary periodically to simulate non-stationarity.
    """

    def __init__(self, shift_interval=50, seed=None):
        self.env = gym.make("CartPole-v1")
        self.base_gravity = self.env.gravity
        self.base_length = self.env.length
        self.episode_count = 0
        self.shift_interval = shift_interval
        self.rng = np.random.default_rng(seed)

    def _apply_drift(self):
        """Randomly change physical parameters."""
        if self.episode_count % self.shift_interval == 0 and self.episode_count > 0:
            new_gravity = self.base_gravity * self.rng.uniform(0.8, 1.2)
            new_length = self.base_length * self.rng.uniform(0.8, 1.3)
            self.env.gravity = new_gravity
            self.env.length = new_length

    def reset(self):
        """Reset environment and maybe drift parameters."""
        self.episode_count += 1
        self._apply_drift()
        return self.env.reset()[0]

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
