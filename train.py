import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from crossy_env import GridEnv


def main():
    env = GridEnv()
    check_env(env)
    print("Env OK\n")

    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05, learning_rate=3e-4)
    model.learn(total_timesteps=200_000)
    model.save("ppo_crossy_agent")
    print("Model saved\n")

    # Watch one episode
    obs, _ = env.reset()
    env.render()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.3)
        done = terminated or truncated

    print(f"\nResult: {info.get('reason', 'timeout')}  |  Total reward: {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()