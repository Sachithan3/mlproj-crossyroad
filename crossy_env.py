import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEnv(gym.Env):
    """
    Custom Environment for a 2D Frogger-style Grid World
    Grid Size: 8 rows x 7 cols
    """
    def __init__(self, width=7, height=8):
        super(GridEnv, self).__init__()
        
        self.width = width
        self.height = height
        
        # 1. Define Action Space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY
        self.action_space = spaces.Discrete(5)
        
        # 2. Define Observation Space:
        # Agent Position (x, y) + Car Positions for 5 obstacle rows
        # Represented as a 1D NumPy array of integers
        num_obs_features = 2 + 5  # 2 for agent (x,y), 5 for x-coordinates of cars
        self.observation_space = spaces.Box(
            low=0, 
            high=max(self.width, self.height), 
            shape=(num_obs_features,), 
            dtype=np.int32
        )
        
        # Internal Game State variables
        self.agent_pos = None
        self.car_rows = [1, 2, 3, 4, 5]  # Rows containing cars
        self.car_positions = None

    def _get_obs(self):
        """Constructs the state array passed to the RL agent."""
        return np.array([self.agent_pos[0], self.agent_pos[1]] + self.car_positions, dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start agent at the bottom row center: (x=3, y=7)
        self.agent_pos = [3, self.height - 1]
        
        # Initial positions for cars on rows 1 through 5
        self.car_positions = [1, 3, 0, 4, 2] 
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
            # 1. Update Agent Position
            # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY
            if action == 0 and self.agent_pos[1] > 0:                 # UP (decrements Y)
                self.agent_pos[1] -= 1
            elif action == 1 and self.agent_pos[1] < self.height - 1: # DOWN (increments Y)
                self.agent_pos[1] += 1
            elif action == 2 and self.agent_pos[0] > 0:                 # LEFT (decrements X)
                self.agent_pos[0] -= 1
            elif action == 3 and self.agent_pos[0] < self.width - 1:   # RIGHT (increments X)
                self.agent_pos[0] += 1
            # action == 4 is STAY (no position change)

            # 2. Update Cars (Deterministic Horizontal Movement)
            # Each car shifts 1 cell to the right and wraps around using modulo (%)
            for i in range(len(self.car_positions)):
                self.car_positions[i] = (self.car_positions[i] + 1) % self.width

            # 3. Default Step Penalty & Episode Flags
            reward = -1      # Encourages agent to reach goal quickly
            terminated = False
            truncated = False
            info = {}

            # 4. Check Collision Terminal Condition
            # If agent is on an obstacle row AND shares the exact X coordinate with that row's car
            if self.agent_pos[1] in self.car_rows:
                car_idx = self.car_rows.index(self.agent_pos[1])
                if self.agent_pos[0] == self.car_positions[car_idx]:
                    reward = -100
                    terminated = True
                    info["reason"] = "collision"

            # 5. Check Goal Terminal Condition
            # If agent reaches Row 0 (Top)
            elif self.agent_pos[1] == 0:
                reward = 100
                terminated = True
                info["reason"] = "goal_reached"


            # 6. Return standard 5 Gymnasium values
            return self._get_obs(), reward, terminated, truncated, info

    def render(self):
            """Prints an ASCII visual representation of the grid board in terminal."""
            grid = [["." for _ in range(self.width)] for _ in range(self.height)]

            # Place cars
            for row, col in zip(self.car_rows, self.car_positions):
                grid[row][col] = "#"

            # Place agent
            grid[self.agent_pos[1]][self.agent_pos[0]] = "A"

            print("\n" + "=" * (self.width * 2 + 1))
            print(" GOAL ROW ")
            for row in grid:
                print(" ".join(row))
            print(" START ROW ")
            print("=" * (self.width * 2 + 1))


if __name__ == "__main__":
     env = GridEnv()
     pbservation,info = env.reset()
     env.render()
     