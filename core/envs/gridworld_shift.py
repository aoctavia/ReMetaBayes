# core/envs/gridworld_shift.py
# Clean + fixed + extended version for Probabilistic Meta-RL

import numpy as np


class GridWorldShift:
    """
    Non-stationary 2D GridWorld.
    Reward map shifts periodically to simulate distribution shift.
    """

    def __init__(
        self,
        size=5,
        shift_interval=100,
        reward_noise=0.0,
        reward_scale=1.0,
        max_steps=50,
        seed=None,
    ):
        self.size = size
        self.shift_interval = shift_interval
        self.reward_noise = reward_noise
        self.reward_scale = reward_scale
        self.max_steps = max_steps

        self.rng = np.random.default_rng(seed)

        # agent state
        self.agent_pos = [0, 0]
        self.steps = 0
        self.episode_count = 0

        # initialize map
        self._generate_reward_map()

    # -----------------------------------------------------
    # Reward Map Generation
    # -----------------------------------------------------
    def _generate_reward_map(self):
        self.reward_map = np.zeros((self.size, self.size))
        # assign goal reward
        self.reward_map[self.size - 1, self.size - 1] = 1.0

        # random negative zones
        for _ in range(self.size // 2):
            y = self.rng.integers(0, self.size)
            x = self.rng.integers(0, self.size)
            self.reward_map[y, x] = self.rng.uniform(-1.0, 0.3)

    def _maybe_shift_reward(self):
        if self.episode_count > 0 and (self.episode_count % self.shift_interval == 0):
            self._generate_reward_map()

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------
    def reset(self):
        self.agent_pos = [0, 0]
        self.steps = 0
        self.episode_count += 1
        self._maybe_shift_reward()
        return self._encode_state()

    # one-hot encode
    def _encode_state(self):
        """
        Encode agent position into a FIXED 9x9 grid, regardless of env size.
        """
        MAX = 9  # maximum size used in your task families

        grid = np.zeros((MAX, MAX))

        # agent position inside its own size grid
        y, x = self.agent_pos
        grid[y, x] = 1

        return grid.flatten()


    # -----------------------------------------------------
    # Step Function
    # -----------------------------------------------------
    def step(self, action):
        # move agent
        if action == 0:      # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:    # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:    # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 3:    # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        # base reward
        y, x = self.agent_pos
        reward = self.reward_map[y, x]

        # apply scale
        reward = reward * self.reward_scale

        # apply reward noise
        if self.reward_noise > 0:
            reward += float(np.random.normal(0, self.reward_noise))

        self.steps += 1
        done = False

        # end episode if limit reached
        if self.steps >= self.max_steps:
            done = True

        return self._encode_state(), float(reward), done, {}
