import argparse
import os
import queue
import time
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from torch.distributions import Categorical

from crossy_env import GridEnv
from rl_common import (
    ALGORITHM_SPECS,
    EVAL_EVERY_STEPS,
    LOG_EVERY_EPISODES,
    SEED,
    TOTAL_TIMESTEPS,
    PeriodicEvaluator,
    make_env as make_shared_env,
    record_training_run,
    results_path,
    seed_everything,
)


# ===========================================================================
# Shared configuration and building blocks
# ===========================================================================

OBS_DIM = 7
N_ACTIONS = 5
HIDDEN = 64
GAMMA = 0.99

# Shared experimental conditions -- seed, TOTAL_TIMESTEPS, evaluation protocol
# (EVAL_EVERY_STEPS / EVAL_EPISODES), logging cadence and checkpoint paths --
# live in rl_common.py so all six algorithms cannot drift apart.
# TOTAL_TIMESTEPS matches the total_timesteps=20000 that the preserved PPO
# block below passes to model.learn().
PRINT_EVERY = LOG_EVERY_EPISODES  # episodes between log lines

# ---------------------------------------------------------------------------
# Algorithm-specific hyperparameters
#
# Only parameters with an *equivalent meaning* are shared across algorithms
# (HIDDEN=64 network width, GAMMA=0.99, the 20k-step budget, greedy evaluation).
# The values below are deliberately not harmonised because their meaning
# differs by method; forcing them equal would bias the comparison instead of
# making it fair.
#
#   policy-based  REINFORCE : learning rate and gradient clip only (Monte
#                             Carlo returns, one update per episode).
#   value-based   DQN/DDQN  : replay buffer, minibatch size, learning starts,
#                             target-network sync, epsilon schedule. DQN and
#                             DDQN share these because they differ *only* in
#                             the bootstrap target.
#   hybrid        A2C       : 5-step rollouts, value/entropy coefficients,
#                             clip 0.5 (synchronous updates).
#                 A3C       : same shape, clip 40.0 because several workers
#                             update shared parameters asynchronously.
# ---------------------------------------------------------------------------
POLICY_HPARAMS = {
    "reinforce": {"lr": 3e-4, "max_grad_norm": 0.5, "entropy_coef": 0.01},
}

VALUE_HPARAMS = {
    # Keep exploring long enough to discover STAY-then-UP, then exploit.
    # Best-eval checkpointing (PeriodicEvaluator) guards against forgetting.
    # Vanilla DQN is less stable than Double DQN on this task, so it uses
    # softer target updates and more gradient steps per env step.
    "dqn": {
        "lr": 5e-4,
        "buffer_size": 50_000,
        "batch_size": 128,
        "learning_starts": 500,
        "target_update": 1,  # soft update every step
        "tau": 0.01,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.7,
        "max_grad_norm": 10.0,
        "grad_steps": 2,
    },
    "ddqn": {
        "lr": 1e-3,
        "buffer_size": 50_000,
        "batch_size": 64,
        "learning_starts": 1_000,
        "target_update": 500,
        "tau": 1.0,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.6,
        "max_grad_norm": 10.0,
        "grad_steps": 1,
    },
}

HYBRID_HPARAMS = {
    "a2c": {
        "lr": 7e-4,
        "n_steps": 5,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
    },
    "a3c": {
        "lr": 7e-4,
        "n_steps": 5,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 40.0,
    },
}


def make_env():
    env = GridEnv()
    check_env(env)
    print("Env OK\n")
    return env


def to_tensor(obs):
    return torch.as_tensor(obs, dtype=torch.float32)


# Scale agent/car coordinates into roughly [0, 1] inside each network so
# training and ``load_policy`` evaluation always match.
_OBS_SCALE = (6.0, 7.0, 6.0, 6.0, 6.0, 6.0, 6.0)


def _register_obs_scale(module):
    module.register_buffer(
        "obs_scale", torch.tensor(_OBS_SCALE, dtype=torch.float32)
    )


def _normalize_obs(module, x):
    return x.float() / module.obs_scale


def discounted_returns(rewards, gamma):
    returns = torch.zeros(len(rewards), dtype=torch.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def watch_episode(env, action_fn):
    obs, _ = env.reset()
    env.render()
    done = False
    total_reward = 0
    while not done:
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.3)
        done = terminated or truncated

    print(f"\nResult: {info.get('reason', 'timeout')}  |  Total reward: {total_reward:.1f}")
    env.close()


class PolicyNet(nn.Module):
    """Actor-only network used by REINFORCE."""

    def __init__(self):
        super().__init__()
        _register_obs_scale(self)
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(_normalize_obs(self, x))


class ActorCritic(nn.Module):
    """Shared trunk with separate policy and value heads (A2C / A3C)."""

    def __init__(self):
        super().__init__()
        _register_obs_scale(self)
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        )
        self.policy = nn.Linear(HIDDEN, N_ACTIONS)
        self.value = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        features = self.trunk(_normalize_obs(self, x))
        return self.policy(features), self.value(features).squeeze(-1)


class QNet(nn.Module):
    """Action-value network used by DQN / DDQN."""

    def __init__(self):
        super().__init__()
        _register_obs_scale(self)
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(_normalize_obs(self, x))


class ReplayBuffer:
    """Fixed-size circular buffer for off-policy value-based methods."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.next_obs = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.dones[idx]),
        )


def _ppo_hyperparameters(model):
    """Snapshot PPO's effective hyperparameters from the trained SB3 model."""
    values = {}
    for key in (
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
    ):
        if not hasattr(model, key):
            continue
        value = getattr(model, key)
        if isinstance(value, (int, float)):
            values[key] = float(value) if isinstance(value, float) else int(value)
    values["policy"] = "MlpPolicy"
    values["total_timesteps"] = TOTAL_TIMESTEPS
    return values


def main():
    args = parse_args()
    # Seed before any model is constructed so PPO's initialisation is
    # reproducible too, without touching the preserved PPO block.
    seed_everything(args.seed)

    if args.algo != "ppo":
        train_from_scratch(
            args.algo,
            workers=args.workers,
            total_timesteps=args.timesteps,
            seed=args.seed,
            watch=not args.no_watch,
        )
        return

    if args.timesteps != TOTAL_TIMESTEPS:
        print(
            f"[PPO] budget fixed at {TOTAL_TIMESTEPS} steps by the preserved "
            f"implementation; ignoring --timesteps {args.timesteps}"
        )

    # ==================================================================
    # Policy-based: PPO (original Stable-Baselines3 implementation)
    #
    # Every line between here and the matching end marker is the original
    # PPO implementation, unchanged. Only the timing start and the results
    # recording that bracket it are new.
    # ==================================================================
    started = time.time()
    env = GridEnv()
    check_env(env)
    print("Env OK\n")

    from stable_baselines3.common.callbacks import EvalCallback

    eval_env = GridEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=".",
        log_path="results",
        eval_freq=EVAL_EVERY_STEPS,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=0,
    )
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05, learning_rate=3e-4)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    # Prefer the best eval checkpoint when EvalCallback produced one.
    best_path = "best_model.zip"
    if os.path.exists(best_path):
        model = PPO.load(best_path, device="cpu")
        print(f"[PPO] restored best eval checkpoint from {best_path}")
    model.save("ppo_crossy_agent")
    print("Model saved\n")
    eval_env.close()

    # Watch one episode
    if not args.no_watch:
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
    # ==================================================================
    # end of PPO block
    # ==================================================================

    duration = time.time() - started
    episode_info = list(getattr(model, "ep_info_buffer", None) or [])
    record_training_run(
        "ppo",
        duration_sec=duration,
        timesteps=TOTAL_TIMESTEPS,
        hyperparameters=_ppo_hyperparameters(model),
        rewards=[float(item["r"]) for item in episode_info],
        lengths=[int(item["l"]) for item in episode_info],
        seed=args.seed,
        notes=(
            "PPO via Stable-Baselines3; no in-training evaluation hook, so "
            "evaluation metrics come from evaluate.py. Training rewards are "
            "SB3's rolling last-100 episode buffer, not a full log."
        ),
    )
    print(f"[PPO] training metrics -> {results_path('ppo')}")


# ===========================================================================
# Policy-based: REINFORCE (Monte Carlo policy gradient)
# ===========================================================================

def train_reinforce(total_timesteps=TOTAL_TIMESTEPS, seed=SEED, watch=True):
    seed_everything(seed)
    env = make_env()
    hp = POLICY_HPARAMS["reinforce"]
    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=hp["lr"])

    def _snapshot():
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    evaluator = PeriodicEvaluator("REINFORCE", seed=seed, snapshot_fn=_snapshot)

    recent = deque(maxlen=20)
    episode = 0
    steps = 0
    obs, _ = env.reset(seed=seed)
    started = time.time()

    def greedy_action(o):
        with torch.no_grad():
            return int(net(to_tensor(o)).argmax())

    while steps < total_timesteps:
        # Collect one complete episode so the update uses true Monte Carlo returns.
        log_probs, rewards, entropies = [], [], []
        ep_reward = 0.0
        done = False
        while not done:
            obs_t = to_tensor(obs)
            dist = Categorical(logits=net(obs_t))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        returns = discounted_returns(rewards, GAMMA)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(torch.stack(log_probs) * returns).sum()
        loss = loss - hp["entropy_coef"] * torch.stack(entropies).sum()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), hp["max_grad_norm"])
        optimizer.step()

        episode += 1
        recent.append(ep_reward)
        evaluator.record_episode(
            ep_reward, len(rewards), info.get("reason") == "goal_reached"
        )
        if episode % PRINT_EVERY == 0:
            print(
                f"[REINFORCE] episode {episode:4d} | steps {steps:6d} | "
                f"return {ep_reward:7.1f} | avg20 {np.mean(recent):7.1f}"
            )
        evaluator.maybe_run(steps, greedy_action)
        obs, _ = env.reset()

    duration = time.time() - started
    evaluator.restore_best(lambda snap: net.load_state_dict(snap))
    path = "reinforce_crossy_agent.pt"
    torch.save(net.state_dict(), path)
    print(f"Model saved -> {path}\n")

    record_training_run(
        "reinforce",
        duration_sec=duration,
        timesteps=steps,
        hyperparameters=hp,
        rewards=evaluator.train_rewards,
        lengths=evaluator.train_lengths,
        successes=evaluator.train_successes,
        eval_history=evaluator.history,
        seed=seed,
        total_timesteps=total_timesteps,
    )
    evaluator.close()

    if watch:
        watch_episode(env, greedy_action)


# ===========================================================================
# Value-based: DQN and Double DQN
# ===========================================================================

def train_dqn(total_timesteps=TOTAL_TIMESTEPS, seed=SEED, watch=True, double=False):
    name = "DDQN" if double else "DQN"
    hp = VALUE_HPARAMS["ddqn" if double else "dqn"]
    seed_everything(seed)
    env = make_env()
    rng = np.random.default_rng(seed)  # same exploration stream for DQN/DDQN

    online_net = QNet()
    target_net = QNet()
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=hp["lr"])
    buffer = ReplayBuffer(hp["buffer_size"])

    def _snapshot():
        return {
            k: v.detach().cpu().clone() for k, v in online_net.state_dict().items()
        }

    evaluator = PeriodicEvaluator(name, seed=seed, snapshot_fn=_snapshot)

    recent = deque(maxlen=20)
    episode = 0
    steps = 0
    ep_reward = 0.0
    ep_length = 0
    obs, _ = env.reset(seed=seed)
    decay_steps = max(1, int(total_timesteps * hp["epsilon_decay"]))
    started = time.time()

    def greedy_action(o):
        with torch.no_grad():
            return int(online_net(to_tensor(o)).argmax())

    for _ in range(total_timesteps):
        progress = min(1.0, steps / decay_steps)
        epsilon = hp["epsilon_start"] + progress * (hp["epsilon_end"] - hp["epsilon_start"])

        if rng.random() < epsilon:
            action = int(rng.integers(N_ACTIONS))
        else:
            with torch.no_grad():
                action = int(online_net(to_tensor(obs)).argmax())

        next_obs, reward, terminated, truncated, info = env.step(action)
        # Bootstrap through time-limit truncation; only true terminations mask it.
        buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_reward += reward
        ep_length += 1
        steps += 1

        if steps >= hp["learning_starts"]:
            for _ in range(int(hp.get("grad_steps", 1))):
                obs_b, actions_b, rewards_b, next_obs_b, dones_b = buffer.sample(
                    hp["batch_size"], rng
                )
                with torch.no_grad():
                    if double:
                        next_actions = online_net(next_obs_b).argmax(
                            dim=1, keepdim=True
                        )
                        next_q = (
                            target_net(next_obs_b)
                            .gather(1, next_actions)
                            .squeeze(1)
                        )
                    else:
                        next_q = target_net(next_obs_b).max(dim=1).values
                    targets = rewards_b + GAMMA * next_q * (1.0 - dones_b)

                current_q = online_net(obs_b).gather(
                    1, actions_b.unsqueeze(1)
                ).squeeze(1)
                loss = F.smooth_l1_loss(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    online_net.parameters(), hp["max_grad_norm"]
                )
                optimizer.step()

        if steps % hp["target_update"] == 0:
            tau = float(hp.get("tau", 1.0))
            if tau >= 1.0:
                target_net.load_state_dict(online_net.state_dict())
            else:
                with torch.no_grad():
                    for online_p, target_p in zip(
                        online_net.parameters(), target_net.parameters()
                    ):
                        target_p.data.mul_(1.0 - tau).add_(
                            online_p.data, alpha=tau
                        )
        if terminated or truncated:
            episode += 1
            recent.append(ep_reward)
            evaluator.record_episode(
                ep_reward, ep_length, info.get("reason") == "goal_reached"
            )
            if episode % PRINT_EVERY == 0:
                print(
                    f"[{name}] episode {episode:4d} | steps {steps:6d} | "
                    f"return {ep_reward:7.1f} | avg20 {np.mean(recent):7.1f} | "
                    f"eps {epsilon:.2f}"
                )
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0

        evaluator.maybe_run(steps, greedy_action)

    duration = time.time() - started
    evaluator.restore_best(lambda snap: online_net.load_state_dict(snap))
    path = f"{name.lower()}_crossy_agent.pt"
    torch.save(online_net.state_dict(), path)
    print(f"Model saved -> {path}\n")

    record_training_run(
        name.lower(),
        duration_sec=duration,
        timesteps=steps,
        hyperparameters=hp,
        rewards=evaluator.train_rewards,
        lengths=evaluator.train_lengths,
        successes=evaluator.train_successes,
        eval_history=evaluator.history,
        seed=seed,
        total_timesteps=total_timesteps,
    )
    evaluator.close()

    if watch:
        watch_episode(env, greedy_action)


# ===========================================================================
# Hybrid: A2C (synchronous) and A3C (asynchronous workers)
# ===========================================================================

def n_step_returns(rewards, terminated, truncated, next_values, gamma):
    """Discounted n-step returns with bootstrapping on time-limit truncation."""
    running = next_values[-1]
    returns = []
    for i in range(len(rewards) - 1, -1, -1):
        if terminated[i]:
            running = rewards[i]
        elif truncated[i]:
            running = rewards[i] + gamma * next_values[i]
        else:
            running = rewards[i] + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def actor_critic_loss(net, obs_batch, actions, returns, hp):
    logits, values = net(obs_batch)
    dist = Categorical(logits=logits)
    advantages = returns - values.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -(dist.log_prob(actions) * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().mean()
    return policy_loss + hp["value_coef"] * value_loss - hp["entropy_coef"] * entropy


def train_a2c(total_timesteps=TOTAL_TIMESTEPS, seed=SEED, watch=True):
    seed_everything(seed)
    env = make_env()
    hp = HYBRID_HPARAMS["a2c"]
    net = ActorCritic()
    optimizer = torch.optim.RMSprop(
        net.parameters(), lr=hp["lr"], alpha=0.99, eps=1e-5
    )

    def _snapshot():
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    evaluator = PeriodicEvaluator("A2C", seed=seed, snapshot_fn=_snapshot)

    recent = deque(maxlen=20)
    episode = 0
    steps = 0
    ep_reward = 0.0
    ep_length = 0
    obs, _ = env.reset(seed=seed)
    started = time.time()

    def greedy_action(o):
        with torch.no_grad():
            logits, _ = net(to_tensor(o))
        return int(logits.argmax())

    while steps < total_timesteps:
        obs_buf, act_buf, rew_buf = [], [], []
        term_buf, trunc_buf, vnext_buf = [], [], []

        rollout_len = min(hp["n_steps"], total_timesteps - steps)
        for _ in range(rollout_len):
            obs_t = to_tensor(obs)
            with torch.no_grad():
                logits, _ = net(obs_t)
                action = Categorical(logits=logits).sample()

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            with torch.no_grad():
                _, next_value = net(to_tensor(next_obs))

            obs_buf.append(obs)
            act_buf.append(action.item())
            rew_buf.append(float(reward))
            term_buf.append(terminated)
            trunc_buf.append(truncated)
            vnext_buf.append(float(next_value))

            obs = next_obs
            ep_reward += reward
            ep_length += 1
            steps += 1

            if terminated or truncated:
                episode += 1
                recent.append(ep_reward)
                evaluator.record_episode(
                    ep_reward, ep_length, info.get("reason") == "goal_reached"
                )
                if episode % PRINT_EVERY == 0:
                    print(
                        f"[A2C] episode {episode:4d} | steps {steps:6d} | "
                        f"return {ep_reward:7.1f} | avg20 {np.mean(recent):7.1f}"
                    )
                obs, _ = env.reset()
                ep_reward = 0.0
                ep_length = 0

        returns = n_step_returns(rew_buf, term_buf, trunc_buf, vnext_buf, GAMMA)
        obs_batch = torch.stack([to_tensor(o) for o in obs_buf])
        actions = torch.tensor(act_buf, dtype=torch.long)

        loss = actor_critic_loss(net, obs_batch, actions, returns, hp)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), hp["max_grad_norm"])
        optimizer.step()

        evaluator.maybe_run(steps, greedy_action)

    duration = time.time() - started
    evaluator.restore_best(lambda snap: net.load_state_dict(snap))
    path = "a2c_crossy_agent.pt"
    torch.save(net.state_dict(), path)
    print(f"Model saved -> {path}\n")

    record_training_run(
        "a2c",
        duration_sec=duration,
        timesteps=steps,
        hyperparameters=hp,
        rewards=evaluator.train_rewards,
        lengths=evaluator.train_lengths,
        successes=evaluator.train_successes,
        eval_history=evaluator.history,
        seed=seed,
        total_timesteps=total_timesteps,
    )
    evaluator.close()

    if watch:
        watch_episode(env, greedy_action)


def _a3c_worker(worker_id, global_model, update_lock, stats_queue, max_steps, seed=SEED):
    """One asynchronous A3C worker: local rollouts, shared global parameters."""
    torch.set_num_threads(1)
    hp = HYBRID_HPARAMS["a3c"]
    # Workers share the run seed but offset it per worker so sampling streams
    # are independent while the whole run stays reproducible.
    seed_everything(seed + worker_id)
    env = make_shared_env()
    optimizer = torch.optim.RMSprop(
        global_model.parameters(), lr=hp["lr"], alpha=0.99, eps=1e-5
    )

    obs, _ = env.reset()
    episode = 0
    steps = 0
    ep_reward = 0.0
    ep_length = 0

    while steps < max_steps:
        obs_buf, act_buf, rew_buf = [], [], []
        term_buf, trunc_buf, vnext_buf = [], [], []

        rollout_len = min(hp["n_steps"], max_steps - steps)
        for _ in range(rollout_len):
            obs_t = to_tensor(obs)
            with torch.no_grad():
                logits, _ = global_model(obs_t)
                action = Categorical(logits=logits).sample()

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            with torch.no_grad():
                _, next_value = global_model(to_tensor(next_obs))

            obs_buf.append(obs)
            act_buf.append(action.item())
            rew_buf.append(float(reward))
            term_buf.append(terminated)
            trunc_buf.append(truncated)
            vnext_buf.append(float(next_value))

            obs = next_obs
            ep_reward += reward
            ep_length += 1
            steps += 1

            if terminated or truncated:
                episode += 1
                stats_queue.put(
                    {
                        "type": "episode",
                        "worker": worker_id,
                        "episode": episode,
                        "reward": ep_reward,
                        "length": ep_length,
                        "success": info.get("reason") == "goal_reached",
                    }
                )
                obs, _ = env.reset()
                ep_reward = 0.0
                ep_length = 0

        returns = n_step_returns(rew_buf, term_buf, trunc_buf, vnext_buf, GAMMA)
        obs_batch = torch.stack([to_tensor(o) for o in obs_buf])
        actions = torch.tensor(act_buf, dtype=torch.long)

        # Workers roll out independently; only the shared-model update is locked.
        with update_lock:
            optimizer.zero_grad()
            loss = actor_critic_loss(global_model, obs_batch, actions, returns, hp)
            loss.backward()
            nn.utils.clip_grad_norm_(global_model.parameters(), hp["max_grad_norm"])
            optimizer.step()

        # Report progress so the parent can evaluate on the shared step clock.
        stats_queue.put({"type": "progress", "worker": worker_id, "steps": steps})

    env.close()


def _snapshot_act_fn(global_model, update_lock):
    """Greedy policy over a frozen copy of the shared A3C parameters."""
    with update_lock:
        snapshot = ActorCritic()
        snapshot.load_state_dict(
            {k: v.detach().clone() for k, v in global_model.state_dict().items()}
        )
    snapshot.eval()

    def act(obs):
        with torch.no_grad():
            logits, _ = snapshot(to_tensor(obs))
        return int(logits.argmax())

    return act


def train_a3c(total_timesteps=TOTAL_TIMESTEPS, seed=SEED, watch=True, workers=None):
    seed_everything(seed)
    hp = HYBRID_HPARAMS["a3c"]
    env = make_env()
    n_workers = workers or min(4, os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, total_timesteps))

    # Spawn avoids inheriting thread state from earlier training in the same
    # process; shared-memory tensors keep the global model shared either way.
    if "spawn" in mp.get_all_start_methods():
        ctx = mp.get_context("spawn")
    elif "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
    else:
        ctx = mp.get_context()

    global_model = ActorCritic()
    global_model.share_memory()
    update_lock = ctx.Lock()
    stats_queue = ctx.Queue()

    def _snapshot():
        with update_lock:
            return {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
            }

    evaluator = PeriodicEvaluator("A3C", seed=seed, snapshot_fn=_snapshot)

    steps_per_worker, extra = divmod(total_timesteps, n_workers)
    processes = []
    for i in range(n_workers):
        worker_steps = steps_per_worker + (1 if i < extra else 0)
        process = ctx.Process(
            target=_a3c_worker,
            args=(i, global_model, update_lock, stats_queue, worker_steps, seed),
            daemon=True,
        )
        process.start()
        processes.append(process)

    episodes = []
    progress = {i: 0 for i in range(n_workers)}
    recent = deque(maxlen=20)
    started = time.time()

    def handle(item):
        if item.get("type") == "progress":
            progress[item["worker"]] = item["steps"]
            total = sum(progress.values())
            # Evaluate on the shared step clock, against a frozen snapshot so
            # the running workers are not disturbed.
            if total >= evaluator.next_step:
                evaluator.maybe_run(total, _snapshot_act_fn(global_model, update_lock))
            return

        episodes.append(item)
        evaluator.record_episode(item["reward"], item["length"], item["success"])
        recent.append(item["reward"])
        if len(episodes) % PRINT_EVERY == 0:
            alive = sum(p.is_alive() for p in processes)
            print(
                f"[A3C] episodes {len(episodes):4d} | "
                f"avg20 {np.mean(recent):7.1f} | workers {alive}/{n_workers}"
            )

    while any(p.is_alive() for p in processes):
        try:
            item = stats_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        handle(item)

    for process in processes:
        process.join()

    # Drain any episode stats still buffered after the workers exit.
    while True:
        try:
            item = stats_queue.get(timeout=0.2)
        except queue.Empty:
            break
        handle(item)

    duration = time.time() - started

    failed = [p.exitcode for p in processes if p.exitcode]
    if failed:
        print(f"[A3C] worker failures (exit codes): {failed}")
    if episodes:
        print(
            f"[A3C] finished {len(episodes)} episodes | "
            f"last100 avg {np.mean([e['reward'] for e in episodes[-100:]]):7.1f}"
        )

    evaluator.restore_best(
        lambda snap: global_model.load_state_dict(snap)
    )
    path = "a3c_crossy_agent.pt"
    torch.save(global_model.state_dict(), path)
    print(f"Model saved -> {path}\n")

    record_training_run(
        "a3c",
        duration_sec=duration,
        timesteps=sum(progress.values()),
        hyperparameters=dict(hp, workers=n_workers),
        rewards=evaluator.train_rewards,
        lengths=evaluator.train_lengths,
        successes=evaluator.train_successes,
        eval_history=evaluator.history,
        seed=seed,
        total_timesteps=total_timesteps,
        notes=(
            "Asynchronous rollout: the step clock is the sum of worker steps and "
            "periodic evaluation uses a frozen snapshot of the shared model."
        ),
    )
    evaluator.close()

    def greedy_action(o):
        with torch.no_grad():
            logits, _ = global_model(to_tensor(o))
        return int(logits.argmax())

    if watch:
        watch_episode(env, greedy_action)


# ===========================================================================
# Entry point and algorithm selection
# ===========================================================================

ALGORITHMS = tuple(ALGORITHM_SPECS)  # ppo, reinforce, dqn, ddqn, a2c, a3c


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and compare RL algorithms on the Crossy Road GridEnv."
    )
    parser.add_argument(
        "--algo",
        choices=ALGORITHMS,
        default="ppo",
        help="Algorithm to run (default: ppo, the original implementation).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of A3C workers (default: up to 4). Ignored by other algorithms.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=(
            "Environment-step training budget for the from-scratch algorithms "
            f"(default: {TOTAL_TIMESTEPS}, the shared budget). Ignored by PPO, "
            "whose preserved implementation fixes its own budget."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Shared random seed (default: {SEED}).",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Skip the post-training demo episode (useful for smoke tests).",
    )
    return parser.parse_args()


def train_from_scratch(algo, workers=None, total_timesteps=TOTAL_TIMESTEPS, seed=SEED, watch=True):
    if algo == "reinforce":
        train_reinforce(total_timesteps=total_timesteps, seed=seed, watch=watch)
    elif algo == "dqn":
        train_dqn(total_timesteps=total_timesteps, seed=seed, watch=watch, double=False)
    elif algo == "ddqn":
        train_dqn(total_timesteps=total_timesteps, seed=seed, watch=watch, double=True)
    elif algo == "a2c":
        train_a2c(total_timesteps=total_timesteps, seed=seed, watch=watch)
    elif algo == "a3c":
        train_a3c(total_timesteps=total_timesteps, seed=seed, watch=watch, workers=workers)


if __name__ == "__main__":
    main()