Project Overview
Title: A Comparative Study of Value-Based, Policy-Based, and Hybrid RL Algorithms for Control Task Optimization
What it is: You have 3 people, each implementing 2 algorithms (6 total) that all train on the same custom environment. The goal is to compare how different families of RL algorithms perform on the same task.

Person	Category
You (sachi)	Policy-Based
Person 2	Value-Based
Person 3	Hybrid (Actor-Critic)

The comparison will likely be on metrics like: sample efficiency, stability, final reward, convergence speed.
The Environment: GridEnv (Crossy-Road style)


This is a grid world the agent must navigate from bottom to top while avoiding cars.
 Grid: 7 columns × 8 rows (everything is 0 indexed)

===================  
. . . . . . .  (Row 0 - Goal) THIS IS THE GOAL ROW
. # . . . . .  (Row 1 - Car at x=1)
. . . # . . .  (Row 2 - Car at x=3)
# . . . . . .  (Row 3 - Car at x=0)
. . . . # . .  (Row 4 - Car at x=4)
. . # . . . .  (Row 5 - Car at x=2)
. . . . . . .  (Row 6 - Safe) 
. . . A . . .  (Row 7 - Agent Start at x=3) THIS IS THE START ROW
===================

 
Agent: A — starts at position (3, 7) (column 3, row 7).
Cars: # — one per row on rows 1–5. Each step, every car moves 1 cell to the right and wraps around (mod 7).
Actions (5 discrete)
Action	Meaning
0	UP
1	DOWN
2	LEFT
3	RIGHT
4	STAY

Observation: (what the agent "sees")
A 1D array of 7 integers:
[agent_x, agent_y, car_row1_x, car_row2_x, car_row3_x, car_row4_x, car_row5_x]

For example, if car_row3_x = 0, then there is a car present in row 3 and column 0.
So the agent knows its own position and where each car is. That's it — no images, no grids, just 7 numbers.

Rewards:

Every step     :	-1 (penalizes being slow)
Collision with car	:    -100 (episode ends)
Reach row 0 (goal)	:    +100 (episode ends)

Key insight for the agent: It must learn to time its movements upward to dodge cars, taking the shortest path possible because every step costs -1.


How Gymnasium Works (the framework)
Gymnasium (formerly OpenAI Gym) is a standard interface for RL.Think of it as a blueprint for making a RL environment. It uses inheritance and few standard methods (like step(),reset(),etc). Every environment follows the same contract:

1. Create the environment
from crossy_env import GridEnv
env = GridEnv()

2. Reset (start a new episode)
observation, info = env.reset()
# observation = [3, 7, 1, 3, 0, 4, 2]  ← agent at (3,7), car positions

3. Step (take an action)
observation, reward, terminated, truncated, info = env.step(action)
Returns 5 things:
* observation — new state (7-int array)
* reward — scalar reward (-1, -100, or +100)
* terminated — True if episode is over (collision or goal)
* truncated — True if episode was cut short (not used here, always False)
* info — debug dict (e.g. {"reason": "collision"})

4. The episode loop
obs, info = env.reset()
done = False

while not done:
    action = agent.choose_action(obs)   # YOUR ALGORITHM DECIDES HERE
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

5. Render (see what's happening)
env.render()   # prints ASCII grid to terminal
How train.py Works (existing PPO script)
The file train.py already has a working pipeline using Stable-Baselines3 (a library with pre-built RL algorithms):
from stable_baselines3 import PPO
from crossy_env import GridEnv

env = GridEnv()
model = PPO("MlpPolicy", env=env, learning_rate=0.0003, ...)
model.learn(total_timesteps=100000)
model.save("ppo_frogger_model")
* MlpPolicy = a simple neural network (multi-layer perceptron) that takes the 7-int observation and outputs an action
* model.learn() = trains for 100K timesteps (each timestep = one env.step())
* model.save() = saves trained weights to a .zip file
You can swap PPO for any SB3 algorithm (DQN, A2C, SAC, etc.) and it works the same way. That's the beauty of the Gymnasium + SB3 combo.
What Each Person Needs to Do
1. Activate the virtual environment: source myenv/bin/activate
2. Import the env: from crossy_env import GridEnv
3. Implement/train your 2 algorithms using either:
* SB3 (easy): e.g. from stable_baselines3 import DQN — just swap the import in train.py
* From scratch (for the report, shows deeper understanding): implement the algo yourself using PyTorch, interacting with env.step() and env.reset()
4. Evaluate: run the trained agent, log rewards per episode, plot learning curves
5. Compare: all 6 algorithms' performance on the same env
The environment is shared — nobody modifies crossy_env.py. Each person just writes their own training script that imports and uses it.


Next step for the project (after first review is to add visualization using pygame or something to see what the model actually does finally)