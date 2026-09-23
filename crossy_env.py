import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GridEnv(gym.Env):
    """
    Crossy Road environment.
    Grid: 8 rows x 7 cols. Agent starts at bottom centre, goal is top row.
    Cars occupy rows 1-5, fixed start positions, move right 1 cell per step.
    """

    def __init__(self, width=7, height=8, max_steps=100):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # 0=UP  1=DOWN  2=LEFT  3=RIGHT  4=STAY
        self.action_space = spaces.Discrete(5)

        # [agent_x, agent_y, car0_x, car1_x, car2_x, car3_x, car4_x]
        self.observation_space = spaces.Box(
            low=0, high=max(width, height), shape=(7,), dtype=np.int32
        )

        self.car_rows = [1, 2, 3, 4, 5]
        self.car_start = [1, 3, 0, 4, 2]   # fixed every episode

        self.agent_pos = None
        self.car_positions = None
        self.current_step = 0

    def _get_obs(self):
        return np.array(
            [self.agent_pos[0], self.agent_pos[1]] + list(self.car_positions),
            dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent_pos = [3, self.height - 1]          # centre bottom
        self.car_positions = list(self.car_start)       # fixed each time
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        prev_y = self.agent_pos[1]

        # Move agent
        if   action == 0 and self.agent_pos[1] > 0:                # UP
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.height - 1:  # DOWN
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:                # LEFT
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.width - 1:   # RIGHT
            self.agent_pos[0] += 1

        # Check 1: agent walked into a car
        if self.agent_pos[1] in self.car_rows:
            i = self.car_rows.index(self.agent_pos[1])
            if self.agent_pos[0] == self.car_positions[i]:
                return self._get_obs(), -10, True, False, {"reason": "collision"}

        # Move cars right
        self.car_positions = [(c + 1) % self.width for c in self.car_positions]

        # Check 2: a car moved into agent
        if self.agent_pos[1] in self.car_rows:
            i = self.car_rows.index(self.agent_pos[1])
            if self.agent_pos[0] == self.car_positions[i]:
                return self._get_obs(), -10, True, False, {"reason": "collision"}

        truncated = self.current_step >= self.max_steps

        if self.agent_pos[1] == 0:
            return self._get_obs(), 100, True, False, {"reason": "goal_reached"}

        # Progress reward
        reward = (prev_y - self.agent_pos[1]) * 5.0 - 0.1
        return self._get_obs(), reward, False, truncated, {}

    def render(self):
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for row, col in zip(self.car_rows, self.car_positions):
            grid[row][col] = "C"
        grid[self.agent_pos[1]][self.agent_pos[0]] = "A"
        print("\n" + "=" * (self.width * 2 + 1))
        for row in grid:
            print(" ".join(row))
        print("=" * (self.width * 2 + 1))