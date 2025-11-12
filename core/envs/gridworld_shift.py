# core/envs/gridworld_shift.py
# Sebuah simplified 2D GridWorld environment dengan reward map yang berubah secara berkala untuk mensimulasikan distribution shift.s
import numpy as np

class GridWorldShift:
    """
    Non-stationary GridWorld environment.
    The reward map shifts after a fixed number of episodes.
    """

    def __init__(self, size=5, shift_interval=100, seed=None):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.shift_interval = shift_interval
        self.episode_count = 0
        self.rng = np.random.default_rng(seed)
        self._generate_reward_map()

    def _generate_reward_map(self):
        """Generate reward landscape (changes after interval)."""
        self.reward_map = np.zeros((self.size, self.size))
        goal_x, goal_y = self.goal
        self.reward_map[goal_x, goal_y] = 1.0
        # Add random perturbations
        for _ in range(self.size // 2):
            x, y = self.rng.integers(0, self.size, 2)
            self.reward_map[x, y] = self.rng.uniform(-1.0, 0.5)

    def _maybe_shift_reward(self):
        if self.episode_count % self.shift_interval == 0 and self.episode_count > 0:
            self._generate_reward_map()

    def reset(self):
        """Reset agent position and possibly shift reward map."""
        self.state = (0, 0)
        self.episode_count += 1
        self._maybe_shift_reward()
        return self._encode_state()

    def _encode_state(self):
        """Convert (x, y) to one-hot encoding."""
        grid = np.zeros((self.size, self.size))
        grid[self.state] = 1
        return grid.flatten()

    def step(self, action):
        """0: up, 1: right, 2: down, 3: left"""
        x, y = self.state
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.size - 1:
            y += 1
        elif action == 2 and x < self.size - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1
        self.state = (x, y)

        reward = self.reward_map[x, y]
        done = self.state == self.goal
        return self._encode_state(), reward, done, {}
