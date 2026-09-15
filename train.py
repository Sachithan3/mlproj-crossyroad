import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from crossy_env import GridEnv  # Import your custom environment class

# 1. Instantiate and Validate Environment
env = GridEnv()

# Optional: Run Gym's internal checker to confirm space compliance
check_env(env)

# 2. Initialize PPO Model
# 'MlpPolicy' handles 1D vector observations using a standard Multi-Layer Perceptron
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,  # Discount factor for future rewards
    verbose=1,
)

# 3. Train the Agent
print("Starting PPO Training...")
model.learn(total_timesteps=100000)
print("Training Complete!")

# 4. Save Trained Model Weights
model.save("ppo_model")