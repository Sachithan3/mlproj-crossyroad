"""Shared experiment configuration, evaluation protocol and result format.

All six algorithms (PPO, REINFORCE, DQN, DDQN, A2C, A3C) use the same:

* environment -- ``crossy_env.GridEnv`` (7x8 grid, 5 actions, identical reward
  function and identical termination rules). ``ENV_KWARGS`` mirrors the
  ``GridEnv`` defaults; nothing here modifies the environment.
* episode initialisation -- ``GridEnv.reset`` always starts the agent at
  (3, 7) with fixed car positions, so every algorithm (and every evaluation
  episode) starts from the same state.
* random seed (``SEED``),
* environment-step training budget (``TOTAL_TIMESTEPS``),
* evaluation protocol -- greedy policy, ``EVAL_EPISODES`` episodes every
  ``EVAL_EVERY_STEPS`` steps during training, plus a post-training pass run by
  ``evaluate.py`` under the exact same protocol,
* logging cadence (``LOG_EVERY_EPISODES`` episodes between log lines),
* checkpoint path and machine-readable results file (``results/<algo>.json``).

Algorithm-specific hyperparameters (learning rates, replay buffers, rollout
lengths, ...) intentionally stay in ``train.py``: their meanings differ
between methods, so forcing them to match would make the comparison *less*
fair, not more. See the comment block above them in ``train.py``.
"""

import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch

from crossy_env import GridEnv

# ===========================================================================
# Shared experimental conditions -- single source of truth
# ===========================================================================

SEED = 0
TOTAL_TIMESTEPS = 100_000  # every algorithm, measured in environment steps
EVAL_EVERY_STEPS = 5_000  # evaluation frequency during training
EVAL_EPISODES = 20  # episodes per evaluation point
LOG_EVERY_EPISODES = 10  # episodes between training log lines
RESULTS_DIR = "results"

# Explicit convergence criterion. A model is only labelled "converged" when
# both conditions hold on the shared evaluation protocol; anything else is
# reported as not converged (or not evaluated), never assumed.
CONVERGENCE_MIN_SUCCESS_RATE = 0.9
CONVERGENCE_MIN_MEAN_REWARD = 100.0

# Must match the GridEnv constructor defaults (the environment itself is shared
# and is never modified). Kept here so evaluation/visualisation use exactly the
# same instance settings as training.
ENV_KWARGS = {"width": 7, "height": 8, "max_steps": 100}

# Panel order for the 2x3 visualisation: rows group algorithm families,
# columns hold the two algorithms of each family.
ALGORITHM_SPECS = {
    "ppo": {
        "label": "PPO",
        "category": "policy-based",
        "checkpoint": "ppo_crossy_agent.zip",
        "loader": "sb3",
    },
    "reinforce": {
        "label": "REINFORCE",
        "category": "policy-based",
        "checkpoint": "reinforce_crossy_agent.pt",
        "loader": "policy",
    },
    "dqn": {
        "label": "DQN",
        "category": "value-based",
        "checkpoint": "dqn_crossy_agent.pt",
        "loader": "q",
    },
    "ddqn": {
        "label": "Double DQN",
        "category": "value-based",
        "checkpoint": "ddqn_crossy_agent.pt",
        "loader": "q",
    },
    "a2c": {
        "label": "A2C",
        "category": "hybrid",
        "checkpoint": "a2c_crossy_agent.pt",
        "loader": "actor_critic",
    },
    "a3c": {
        "label": "A3C",
        "category": "hybrid",
        "checkpoint": "a3c_crossy_agent.pt",
        "loader": "actor_critic",
    },
}

ALGORITHMS = tuple(ALGORITHM_SPECS)
PANEL_ORDER = ALGORITHMS  # 2 columns x 3 rows, one panel per algorithm


# ===========================================================================
# Environment / seeding
# ===========================================================================


def make_env(**overrides):
    """New environment instance with the shared settings (never modified)."""
    kwargs = dict(ENV_KWARGS)
    kwargs.update(overrides)
    return GridEnv(**kwargs)


def seed_everything(seed=SEED):
    """Seed python/numpy/torch and return an independent numpy generator.

    ``GridEnv`` itself is deterministic, but seeding keeps network
    initialisation, epsilon-greedy exploration and A3C worker sampling
    reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def shared_config(seed=SEED, total_timesteps=TOTAL_TIMESTEPS):
    """Snapshot of the shared experimental conditions (embedded in results)."""
    return {
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "eval_every_steps": EVAL_EVERY_STEPS,
        "eval_episodes": EVAL_EPISODES,
        "log_every_episodes": LOG_EVERY_EPISODES,
        "convergence_criterion": {
            "min_success_rate": CONVERGENCE_MIN_SUCCESS_RATE,
            "min_mean_reward": CONVERGENCE_MIN_MEAN_REWARD,
        },
        "env": dict(ENV_KWARGS),
        "observation": "[agent_x, agent_y, car_row1_x, ..., car_row5_x]",
        "actions": ["UP", "DOWN", "LEFT", "RIGHT", "STAY"],
        "reward": {
            "step": "progress * 5.0 - 0.1 (progress = prev_y - new_y)",
            "collision": -10.0,
            "goal": "100 + progress * 5.0 - 0.1",
        },
        "episode_end": ["goal_reached", "collision", "truncated at max_steps"],
        "policy_at_eval": "greedy / deterministic",
    }


# ===========================================================================
# Evaluation protocol -- identical for all six algorithms
# ===========================================================================


def run_evaluation(env, act_fn, episodes=EVAL_EPISODES, seed=SEED):
    """Run a greedy policy for ``episodes`` episodes and summarise them.

    ``act_fn(observation) -> int``. Uses a caller-supplied environment so the
    training loop is never disturbed, and resets with the same seed every
    algorithm uses so all agents face identical episodes.
    """
    rewards, lengths, successes = [], [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        steps = 0
        done = False
        reason = "truncated"
        while not done:
            obs, reward, terminated, truncated, info = env.step(int(act_fn(obs)))
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
            reason = info.get("reason") or ("truncated" if truncated else "done")
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(reason == "goal_reached")

    summary = summarize_episodes(rewards, lengths, successes)
    summary["seed"] = int(seed)
    return summary


def summarize_episodes(rewards, lengths=None, successes=None):
    """Standard metric block with only the comparison metrics."""
    rewards = [float(r) for r in rewards]
    lengths = [int(l) for l in (lengths or [])]
    successes = [bool(s) for s in (successes or [])]

    summary = {"episodes": len(rewards)}
    if rewards:
        summary.update(
            {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
            }
        )
    else:
        summary.update(
            {
                "mean_reward": None,
                "std_reward": None,
            }
        )
    summary["success_rate"] = float(np.mean(successes)) if successes else None
    summary["mean_length"] = float(np.mean(lengths)) if lengths else None
    return summary


# ===========================================================================
# Results files -- one JSON document per algorithm, shared by train/evaluate
# ===========================================================================


def results_path(algo):
    return os.path.join(RESULTS_DIR, f"{algo}.json")


def base_payload(algo, seed=SEED, total_timesteps=TOTAL_TIMESTEPS):
    spec = ALGORITHM_SPECS[algo]
    return {
        "algorithm": algo,
        "label": spec["label"],
        "category": spec["category"],
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "checkpoint": spec["checkpoint"],
        "shared_config": shared_config(seed=seed, total_timesteps=total_timesteps),
    }


def load_results(algo):
    """Best-effort read; returns None for missing or unreadable files."""
    try:
        with open(results_path(algo), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_results(algo, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload["algorithm"] = algo
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(results_path(algo), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def record_training_run(
    algo,
    *,
    duration_sec,
    timesteps,
    hyperparameters,
    rewards=(),
    lengths=(),
    successes=(),
    eval_history=(),
    seed=SEED,
    total_timesteps=TOTAL_TIMESTEPS,
    notes="",
):
    """Write the training half of ``results/<algo>.json``.

    Training metrics live under ``training``; evaluation metrics live under
    ``evaluation`` (written by ``evaluate.py``) so the two are never mixed.
    """
    payload = base_payload(algo, seed=seed, total_timesteps=total_timesteps)
    payload["hyperparameters"] = hyperparameters
    payload["notes"] = notes
    payload["training"] = {
        "timesteps": int(timesteps),
        "episodes": len(rewards),
        "duration_sec": round(float(duration_sec), 3),
        "episode_rewards": [round(float(r), 3) for r in rewards],
        "episode_lengths": [int(l) for l in lengths],
        "episode_successes": [bool(s) for s in successes],
        "summary": summarize_episodes(rewards, lengths, successes),
        "eval_history": list(eval_history),
    }

    # Keep a post-training evaluation that was recorded for an earlier run.
    existing = load_results(algo) or {}
    if "evaluation" in existing:
        payload["evaluation"] = existing["evaluation"]
    return save_results(algo, payload)


def convergence(metrics):
    """Decide whether a model meets the explicit convergence criterion.

    Converged means: on the shared evaluation protocol, at least
    ``CONVERGENCE_MIN_SUCCESS_RATE`` of episodes end on the goal row *and* the
    mean episode reward is at least ``CONVERGENCE_MIN_MEAN_REWARD``. Without
    evaluation metrics the answer is "not evaluated" rather than "not
    converged" -- absence of data is never treated as a result.
    """
    criterion = {
        "min_success_rate": CONVERGENCE_MIN_SUCCESS_RATE,
        "min_mean_reward": CONVERGENCE_MIN_MEAN_REWARD,
    }
    if (
        not metrics
        or metrics.get("mean_reward") is None
        or metrics.get("success_rate") is None
    ):
        return {
            "evaluated": False,
            "converged": False,
            "criterion": criterion,
            "reason": "no evaluation metrics",
        }

    success = float(metrics["success_rate"])
    mean_reward = float(metrics["mean_reward"])
    meets_success = success >= CONVERGENCE_MIN_SUCCESS_RATE
    meets_reward = mean_reward >= CONVERGENCE_MIN_MEAN_REWARD
    return {
        "evaluated": True,
        "converged": bool(meets_success and meets_reward),
        "criterion": criterion,
        "checks": {"success_rate": meets_success, "mean_reward": meets_reward},
        "episodes": metrics.get("episodes"),
        "source": metrics.get("source"),
    }


def record_evaluation(algo, summary, source="evaluate.py", seed=SEED):
    """Write the evaluation half of ``results/<algo>.json``."""
    payload = load_results(algo) or base_payload(algo, seed=seed)
    payload["evaluation"] = dict(
        summary, source=source, convergence=convergence(summary)
    )
    # Keep the criterion visible even for files written by an older run.
    payload.setdefault("shared_config", {}).setdefault(
        "convergence_criterion", shared_config()["convergence_criterion"]
    )
    return save_results(algo, payload)


def best_metrics(results):
    """Pick the most comparable evaluation block available for a run."""
    if not results:
        return None
    evaluation = results.get("evaluation")
    if evaluation and evaluation.get("mean_reward") is not None:
        return evaluation
    history = (results.get("training") or {}).get("eval_history") or []
    if history:
        last = dict(history[-1])
        last.setdefault("source", "training eval")
        return last
    return None


def _meets_convergence(mean_reward, success_rate):
    return (
        success_rate is not None
        and mean_reward is not None
        and float(success_rate) >= CONVERGENCE_MIN_SUCCESS_RATE
        and float(mean_reward) >= CONVERGENCE_MIN_MEAN_REWARD
    )


def steps_to_converge_from_history(history):
    """Earliest logged timestep that already meets the convergence criterion."""
    for entry in history or []:
        if _meets_convergence(entry.get("mean_reward"), entry.get("success_rate")):
            step = entry.get("timestep")
            return int(step) if step is not None else None
    return None


def _load_sb3_evals(path=os.path.join(RESULTS_DIR, "evaluations.npz")):
    """PPO's SB3 EvalCallback dump as ``(timesteps, mean_rewards)`` or None."""
    if not os.path.exists(path):
        return None
    try:
        import numpy as np

        data = np.load(path)
        timesteps = data["timesteps"]
        rewards = data["results"]  # shape (n_evals, n_episodes)
    except Exception:
        return None
    return timesteps, rewards.mean(axis=1)


def steps_to_converge_from_sb3_npz(path=os.path.join(RESULTS_DIR, "evaluations.npz")):
    """Read PPO's SB3 EvalCallback dump for the first converging eval point."""
    loaded = _load_sb3_evals(path)
    if loaded is None:
        return None
    timesteps, mean_rewards = loaded
    for step, mean_reward in zip(timesteps, mean_rewards):
        # Goal episodes dominate; treat mean_reward as the success proxy for
        # SB3 logs that do not store success flags.
        success_rate = float((float(mean_reward) >= CONVERGENCE_MIN_MEAN_REWARD))
        if _meets_convergence(float(mean_reward), success_rate):
            return int(step)
    return None


def best_checkpoint_step_from_history(history):
    """Timestep of the eval point that training would restore as "best".

    ``PeriodicEvaluator`` keeps the first entry with the highest
    ``(success_rate, mean_reward)`` score (strict improvement only). That is the
    policy actually saved in the checkpoint, which can be later than the first
    transient convergence (A3C is the clearest example here).
    """
    best_score = None
    best_step = None
    for entry in history or []:
        success = entry.get("success_rate")
        reward = entry.get("mean_reward")
        step = entry.get("timestep")
        if success is None or reward is None or step is None:
            continue
        score = (float(success), float(reward))
        if best_score is None or score > best_score:
            best_score = score
            best_step = int(step)
    return best_step


def best_checkpoint_step_from_sb3_npz(
    path=os.path.join(RESULTS_DIR, "evaluations.npz"),
):
    """Timestep SB3's ``EvalCallback`` saved as ``best_model.zip``.

    The callback keeps the first eval with the highest mean reward, so the first
    ``argmax`` matches the checkpoint that was restored after training.
    """
    loaded = _load_sb3_evals(path)
    if loaded is None:
        return None
    timesteps, mean_rewards = loaded
    if len(timesteps) == 0:
        return None
    index = int(mean_rewards.argmax())
    return int(timesteps[index])


def comparison_metrics(algo):
    """Metrics used to compare and rank algorithms side by side.

    Final-policy quality (eval success / reward / path length) is primary;
    sample efficiency (first convergence, then the restored-checkpoint step)
    breaks remaining ties so the six models are not all identical on the HUD.
    Wall-clock train time is reported but not used for ranking.
    """
    results = load_results(algo)
    evaluation = best_metrics(results) or {}
    training = (results or {}).get("training") or {}
    history = training.get("eval_history") or []

    mean_reward = evaluation.get("mean_reward")
    success_rate = evaluation.get("success_rate")
    mean_length = evaluation.get("mean_length")
    std_reward = evaluation.get("std_reward")

    steps_to_conv = steps_to_converge_from_history(history)
    best_step = best_checkpoint_step_from_history(history)
    if algo == "ppo":
        if steps_to_conv is None:
            steps_to_conv = steps_to_converge_from_sb3_npz()
        if best_step is None:
            best_step = best_checkpoint_step_from_sb3_npz()

    duration = training.get("duration_sec")
    train_success = (training.get("summary") or {}).get("success_rate")

    reward_per_step = None
    if mean_reward is not None and mean_length not in (None, 0):
        reward_per_step = float(mean_reward) / float(mean_length)

    verdict = convergence(evaluation if evaluation else None)
    return {
        "algorithm": algo,
        "label": ALGORITHM_SPECS[algo]["label"],
        "category": ALGORITHM_SPECS[algo]["category"],
        "mean_reward": None if mean_reward is None else float(mean_reward),
        "std_reward": None if std_reward is None else float(std_reward),
        "success_rate": None if success_rate is None else float(success_rate),
        "mean_length": None if mean_length is None else float(mean_length),
        "reward_per_step": reward_per_step,
        "steps_to_converge": steps_to_conv,
        "best_checkpoint_step": best_step,
        "train_duration_sec": None if duration is None else float(duration),
        "train_success_rate": None if train_success is None else float(train_success),
        "eval_episodes": evaluation.get("episodes"),
        "converged": bool(verdict.get("converged")),
        "evaluated": bool(verdict.get("evaluated")),
    }


def _rank_sort_key(profile):
    """Better policies sort first: success, reward, short path, fast converge.

    Wall-clock train time is shown in the HUD but not used for ranking -- it
    mixes hardware noise with algorithmic cost. The timestep of the checkpoint
    that was actually restored breaks a first-convergence tie, because a
    transient early convergence that was later abandoned should not beat a
    checkpoint that is still good. Exact ties fall back to the fixed panel
    order in ``algorithm_rankings``.
    """
    success = profile.get("success_rate")
    reward = profile.get("mean_reward")
    length = profile.get("mean_length")
    converge_steps = profile.get("steps_to_converge")
    best_step = profile.get("best_checkpoint_step")
    # Quantise reward so tiny float noise (1e-14) does not break ties.
    reward_key = None if reward is None else round(float(reward), 2)
    length_key = None if length is None else round(float(length), 1)
    return (
        1 if profile.get("converged") else 0,
        -1.0 if success is None else -float(success),
        -1e18 if reward_key is None else -reward_key,
        1e18 if length_key is None else length_key,
        1e18 if converge_steps is None else float(converge_steps),
        1e18 if best_step is None else float(best_step),
    )


def algorithm_rankings(order=None):
    """Unique 1-based ranks (#1, #2, #3, ...) from ``comparison_metrics``.

    Ranks are strictly ordered, never duplicated: after the quality and
    sample-efficiency keys, the restored-checkpoint timestep and finally the
    fixed panel order break any remaining tie. Algorithms without evaluation
    metrics are omitted (the visualizer shows them unranked).
    """
    order = tuple(order or ALGORITHMS)
    order_index = {algo: index for index, algo in enumerate(order)}
    scored = []
    for algo in order:
        profile = comparison_metrics(algo)
        if not profile.get("evaluated") or profile.get("mean_reward") is None:
            continue
        scored.append((_rank_sort_key(profile), order_index[algo], algo))

    scored.sort(key=lambda item: (item[0], item[1]))
    ranks = {}
    for rank, (_, _, algo) in enumerate(scored, start=1):
        ranks[algo] = rank
    return ranks


def comparison_table(order=None):
    """All comparison profiles plus ranks, sorted best → worst."""
    order = tuple(order or ALGORITHMS)
    ranks = algorithm_rankings(order)
    rows = []
    for algo in order:
        profile = comparison_metrics(algo)
        profile["rank"] = ranks.get(algo)
        rows.append(profile)
    rows.sort(
        key=lambda row: (
            999 if row["rank"] is None else row["rank"],
            row["algorithm"],
        )
    )
    return rows


# ===========================================================================
# Checkpoint loading (shared by evaluate.py and visualize.py)
# ===========================================================================


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing or cannot be restored."""


def load_policy(algo):
    """Restore a trained checkpoint and return ``act_fn(obs) -> int``.

    No training happens here: PPO is loaded from its SB3 ``.zip`` and the five
    from-scratch agents from their ``.pt`` state dicts.
    """
    spec = ALGORITHM_SPECS[algo]
    path = spec["checkpoint"]
    if not os.path.exists(path):
        raise CheckpointError(
            f"missing checkpoint '{path}' for {spec['label']} "
            f"(train it with: python train.py --algo {algo})"
        )

    if spec["loader"] == "sb3":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise CheckpointError(f"stable-baselines3 is required for {path}: {exc}")
        try:
            model = PPO.load(path, device="cpu")
        except Exception as exc:
            raise CheckpointError(f"could not load '{path}': {exc}")
        return lambda obs: int(model.predict(obs, deterministic=True)[0])

    # Imported lazily: train.py imports this module, so a top-level import
    # here would be circular. Importing only defines networks; main() is
    # guarded, so no training runs.
    from train import ActorCritic, PolicyNet, QNet

    networks = {"policy": PolicyNet, "q": QNet, "actor_critic": ActorCritic}
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"could not load '{path}': {exc}")
    if isinstance(state, dict) and "state_dict" in state:  # tolerate wrappers
        state = state["state_dict"]

    try:
        net = networks[spec["loader"]]()
        net.load_state_dict(state)
    except Exception as exc:
        raise CheckpointError(f"checkpoint '{path}' does not match the model: {exc}")
    net.eval()

    if spec["loader"] == "actor_critic":

        def act(obs):
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(obs, dtype=torch.float32))
            return int(logits.argmax())

    else:

        def act(obs):
            with torch.no_grad():
                logits = net(torch.as_tensor(obs, dtype=torch.float32))
            return int(logits.argmax())

    return act


# ===========================================================================
# Periodic evaluation used by the training loop
# ===========================================================================


class PeriodicEvaluator:
    """Runs the shared evaluation protocol every ``every_steps`` env steps.

    Keeps evaluation metrics out of the training metric lists and prints them
    with a distinct ``eval`` prefix so the two never get confused.

    When ``snapshot_fn`` is provided it is called after each eval that improves
    the best (success rate, then mean reward) score so training can restore
    the best weights at the end -- important for methods that find then forget
    the goal policy (notably DQN).
    """

    def __init__(
        self,
        label,
        every_steps=EVAL_EVERY_STEPS,
        episodes=EVAL_EPISODES,
        seed=SEED,
        snapshot_fn=None,
    ):
        self.label = label
        self.every_steps = every_steps
        self.episodes = episodes
        self.seed = seed
        self.snapshot_fn = snapshot_fn
        self.env = make_env()  # separate instance: training env is untouched
        # Evaluation history (one entry per evaluation point).
        self.history = []
        self.next_step = every_steps
        # Training-episode metrics, deliberately stored separately.
        self.train_rewards = []
        self.train_lengths = []
        self.train_successes = []
        self.best_score = None  # (success_rate, mean_reward)
        self.best_summary = None
        self.best_snapshot = None

    def record_episode(self, reward, length, success):
        """Record one *training* episode (never mixed with eval history)."""
        self.train_rewards.append(float(reward))
        self.train_lengths.append(int(length))
        self.train_successes.append(bool(success))

    def maybe_run(self, steps, act_fn):
        while steps >= self.next_step:
            summary = run_evaluation(
                self.env, act_fn, episodes=self.episodes, seed=self.seed
            )
            self.history.append(dict(summary, timestep=int(steps)))
            score = (
                float(summary["success_rate"]),
                float(summary["mean_reward"]),
            )
            improved = self.best_score is None or score > self.best_score
            marker = ""
            if improved:
                self.best_score = score
                self.best_summary = dict(summary, timestep=int(steps))
                if self.snapshot_fn is not None:
                    self.best_snapshot = self.snapshot_fn()
                marker = " *best*"
            print(
                f"[{self.label}] eval @ {steps:6d} steps | "
                f"mean {summary['mean_reward']:7.2f} | "
                f"success {100.0 * summary['success_rate']:5.1f}% | "
                f"len {summary['mean_length']:5.1f}{marker}"
            )
            self.next_step += self.every_steps
        return self.history

    def restore_best(self, load_fn):
        """Load the best snapshot via ``load_fn(snapshot)`` if one exists."""
        if self.best_snapshot is None:
            return False
        load_fn(self.best_snapshot)
        best = self.best_summary or {}
        print(
            f"[{self.label}] restored best checkpoint @ "
            f"{best.get('timestep', '?')} steps | "
            f"mean {best.get('mean_reward', float('nan')):7.2f} | "
            f"success {100.0 * best.get('success_rate', 0.0):5.1f}%"
        )
        return True

    def close(self):
        self.env.close()
