"""Evaluate trained checkpoints under identical conditions (no training).

This is the second, separate stage of the pipeline:

    python train.py --algo <algo>   # writes checkpoint + results/<algo>.json
    python evaluate.py              # writes the evaluation half of each result
    python visualize.py             # replays all six checkpoints side by side

Every algorithm is scored with the same protocol from ``rl_common``: greedy
actions, ``--episodes`` episodes, the same seed, the same environment settings
and the same starting state. Evaluation metrics are stored under ``evaluation``
in ``results/<algo>.json`` and never mixed with the training metrics.
"""

import argparse
import sys

from rl_common import (
    ALGORITHMS,
    ALGORITHM_SPECS,
    CONVERGENCE_MIN_MEAN_REWARD,
    CONVERGENCE_MIN_SUCCESS_RATE,
    EVAL_EPISODES,
    SEED,
    CheckpointError,
    convergence,
    load_policy,
    make_env,
    record_evaluation,
    results_path,
    run_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained Crossy Road agents with the shared protocol and "
            "write results/<algo>.json."
        )
    )
    parser.add_argument(
        "--algo",
        choices=(*ALGORITHMS, "all"),
        default="all",
        help="Which checkpoint to evaluate (default: all six).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help=f"Evaluation episodes per algorithm (default: {EVAL_EPISODES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Evaluation seed, identical for every algorithm (default: {SEED}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected = ALGORITHMS if args.algo == "all" else (args.algo,)

    rows = []
    missing = []
    for algo in selected:
        spec = ALGORITHM_SPECS[algo]
        label = spec["label"]
        try:
            act = load_policy(algo)
        except CheckpointError as exc:
            print(f"[skip] {label}: {exc}")
            missing.append(algo)
            continue

        env = make_env()
        try:
            summary = run_evaluation(
                env, act, episodes=args.episodes, seed=args.seed
            )
        finally:
            env.close()

        record_evaluation(algo, summary, source="evaluate.py", seed=args.seed)
        verdict = convergence(summary)
        rows.append((algo, spec, summary, verdict))
        print(
            f"[eval] {label:11s} ({spec['category']:12s}) | "
            f"mean {summary['mean_reward']:8.2f} +/- {summary['std_reward']:6.2f} | "
            f"success {100.0 * summary['success_rate']:5.1f}% | "
            f"len {summary['mean_length']:6.1f} | {summary['episodes']} episodes | "
            f"{'CONVERGED' if verdict['converged'] else 'not converged'}"
        )

    if rows:
        print("\nSummary (sorted by mean evaluation reward)")
        print(f"{'algorithm':11s} {'category':13s} {'mean':>9s} {'std':>8s} "
              f"{'success':>8s} {'length':>7s}  converged")
        for algo, spec, summary, verdict in sorted(
            rows, key=lambda row: row[2]["mean_reward"], reverse=True
        ):
            print(
                f"{algo:11s} {spec['category']:13s} "
                f"{summary['mean_reward']:9.2f} {summary['std_reward']:8.2f} "
                f"{100.0 * summary['success_rate']:7.1f}% "
                f"{summary['mean_length']:7.1f}  "
                f"{'yes' if verdict['converged'] else 'no'}"
            )
        print(
            "\nConvergence criterion: success rate >= "
            f"{CONVERGENCE_MIN_SUCCESS_RATE:.0%} and mean reward >= "
            f"{CONVERGENCE_MIN_MEAN_REWARD:+.1f} over the evaluation episodes."
        )
        print(f"Results written to {results_path('<algo>')}")

    if missing:
        print(
            f"\n{len(missing)} of {len(selected)} checkpoint(s) missing: "
            f"{', '.join(missing)}"
        )
    if not rows:
        print("Nothing was evaluated; train a model first (python train.py --algo ppo).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
