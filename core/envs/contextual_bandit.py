# core/envs/contextual_bandit.py
import numpy as np

class ContextualBandit:
    """
    Non-stationary Contextual Bandit environment.
    At each time step, the agent observes a context vector and selects one of k actions.
    The reward distribution drifts over time to simulate non-stationarity.
    """

    def __init__(self, n_actions=5, n_contexts=10, reward_shift_interval=50, reward_noise_std=0.05, seed=None):
        self.n_actions = n_actions
        self.n_contexts = n_contexts
        self.reward_shift_interval = reward_shift_interval
        self.reward_noise_std = reward_noise_std
        self.rng = np.random.default_rng(seed)

        # reward matrix: context x action
        self.reward_matrix = self.rng.uniform(-1.0, 1.0, (n_contexts, n_actions))
        self.timestep = 0

    def _maybe_shift_rewards(self):
        """Drift the reward matrix slightly to simulate non-stationarity."""
        if self.timestep > 0 and self.timestep % self.reward_shift_interval == 0:
            drift = self.rng.normal(0, 0.1, size=self.reward_matrix.shape)
            self.reward_matrix += drift
            # clip to avoid explosion
            self.reward_matrix = np.clip(self.reward_matrix, -2.0, 2.0)

    def reset(self):
        """Return a random context vector."""
        self.timestep = 0
        context_idx = self.rng.integers(0, self.n_contexts)
        return self._context_vector(context_idx)

    def _context_vector(self, idx):
        """One-hot encode context index."""
        ctx = np.zeros(self.n_contexts)
        ctx[idx] = 1.0
        return ctx

    def step(self, action):
        """Given an action, return reward and next context."""
        self._maybe_shift_rewards()
        context_idx = self.rng.integers(0, self.n_contexts)

        # reward based on context and action
        reward = self.reward_matrix[context_idx, action]
        reward += self.rng.normal(0, self.reward_noise_std)

        self.timestep += 1
        next_context = self._context_vector(context_idx)
        done = False  # bandit never "ends" early
        return next_context, reward, done, {}
