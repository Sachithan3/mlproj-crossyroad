For proper viewing on github, switch this file to code and do not view the preview

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
model.save("ppo_model")
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


Standardized Training, Evaluation and 6-Panel Visualization

Pipeline (three separate operations):

    # 1. train one algorithm (repeat for every algorithm)
    for a in ppo reinforce dqn ddqn a2c a3c; do python train.py --algo $a; done

    # 2. evaluate every saved checkpoint with the shared protocol
    python evaluate.py

    # 3. replay all six checkpoints side by side (2 columns x 3 rows)
    python visualize.py

Shared conditions (defined once in rl_common.py). Every algorithm uses the
same environment instance settings, starting state, seed, budget, evaluation
protocol and logging cadence:

    env            GridEnv(width=7, height=8, max_steps=100)  (unchanged)
    observation    [agent_x, agent_y, car_row1_x .. car_row5_x]
    actions        5 discrete (UP, DOWN, LEFT, RIGHT, STAY)
    reward         progress * 5.0 - 0.1 per step, -10 collision, +100 goal
    episode end    goal_reached / collision / truncated at 100 steps
    seed           0                              (--seed to override)
    budget         20,000 environment steps       (--timesteps for the
                   five from-scratch algorithms; PPO's preserved block
                   fixes its own 20,000)
    eval cadence   every 5,000 env steps, 20 greedy episodes
    logging        every 10 episodes (identical log fields)
    checkpoints    ppo_crossy_agent.zip, reinforce/dqn/ddqn/a2c/a3c_crossy_agent.pt
    results        results/<algo>.json (one machine-readable file per algorithm)

Algorithm-specific hyperparameters are intentionally NOT forced to match,
because the same number means different things in different methods:

    REINFORCE   lr 3e-4, grad clip 0.5
    DQN / DDQN  lr 1e-3, buffer 50k, batch 64, learning_starts 1k,
                target sync 500 steps, epsilon 1.0 -> 0.05 over 50% of training
                (DQN and Double DQN differ only in the bootstrap target)
    A2C         5-step rollouts, lr 7e-4, value coef 0.5, entropy coef 0.01,
                grad clip 0.5
    A3C         same shape as A2C, grad clip 40.0 (asynchronous shared update),
                4 workers, seeds seed+worker_id
    PPO         SB3 defaults with ent_coef 0.05 and lr 3e-4 (unchanged)

Shared by all six because the meaning is equivalent: MLP hidden width 64,
gamma 0.99, budget in environment steps, greedy evaluation.

Metrics (training and evaluation are kept separate):

    training    episode rewards/lengths/successes, mean episode reward,
                duration, total environment steps, periodic eval history
    evaluation  mean eval reward, std, min, max, success rate, mean episode
                length, episode count, seed

results/<algo>.json layout:

    algorithm, label, category, seed, total_timesteps, checkpoint,
    shared_config, hyperparameters, notes,
    training   { timesteps, episodes, duration_sec, episode_rewards,
                 episode_lengths, episode_successes, summary, eval_history }
    evaluation { episodes, mean_reward, std_reward, min_reward, max_reward,
                 success_rate, mean_length, seed, source, convergence }

Convergence criterion (explicit, applied to every "converged" label):

    success rate >= 90%  AND  mean evaluation reward >= +100
    (20 greedy evaluation episodes)

The verdict and the numbers it was checked against are stored as
evaluation.convergence in results/<algo>.json. A model without evaluation data
is reported as NOT EVALUATED, never as converged or as failed.

Visualization: visualize.py is fullscreen by default (--windowed and --size for
a window, F toggles at runtime) and shows all six agents in a 2x3 grid (rows
group policy-based / value-based / hybrid). Every panel owns its own GridEnv,
episode counters and restart state, so a goal/crash never pauses or resets the
others. Stepping is strictly uniform: each frame advances every loaded panel by
exactly one environment step; a finished episode restarts on the next frame and
the outcome is reported by a status chip plus a short border flash.

Each panel keeps its board on the left and, under the model name, a labelled
evaluation report: mean reward, std reward, success rate, mean episode length
and episode count, then an explicit verdict badge (CONVERGED / NOT CONVERGED /
NOT EVALUATED / NO CHECKPOINT) with the criterion it was checked against. Live
counters (episode, step, current reward, action, goals so far) sit just above
the report.

Missing or unloadable checkpoints render a placeholder with the command to
train them instead of aborting the window. Keys: Space pause, Up/Down speed,
R restart, F fullscreen, Esc quit. Headless rendering (CI/screenshots):

    python visualize.py --headless --frames 40 --save-frames viz_frames

Layout is computed from the actual display size, so the same command fills a
1080p screen or a smaller window without clipping.

Known limitations:
- GridEnv is fully deterministic and reset() ignores the seed, so every
evaluation episode is identical. Evaluation std is therefore ~0 for greedy
policies; the metric is still computed and reported.
- PPO's implementation is preserved verbatim, so it has no in-training
evaluation hook; its evaluation metrics come from evaluate.py. Its training
rewards are SB3's rolling last-100 episode buffer, not a full episode log.
- Not every algorithm converges within 20k steps (e.g. DQN/DDQN/A3C are still
near-random in the current run). That is expected: the comparison is about
comparable conditions and metrics, not about forcing convergence.