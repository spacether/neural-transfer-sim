"""Default receiver learning experiments for adult Drosophila."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.drosophila_adult.receiver_learning.experiment import (
    run_learning_vs_trials,
    run_learning_vs_precision,
    run_learning_vs_jitter,
)
from analysis.results_writer import write_results
from analysis.plotting import (
    plot_learning_vs_trials,
    plot_learning_vs_precision,
    plot_learning_vs_jitter,
)

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult fly receiver learning experiments")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    print("=== Receiver Learning: Adaptation vs Number of Trials ===")
    trials_results = run_learning_vs_trials(seeds=args.seeds)
    print(f"{'n_learn':>7}  {'rate':>6}  {'kc_corr':>7}  {'decode':>6}")
    print("-" * 35)
    for r in trials_results:
        print(f"{r['n_learning_odors']:7d}  {r['transfer_rate']:6.3f}  "
              f"{r['kc_correlation']:7.3f}  {r['decoding']:6.3f}")

    print("\n=== Receiver Learning: Learning vs Hardware Precision ===")
    precision_results = run_learning_vs_precision(seeds=args.seeds)
    print(f"{'sigma':>6}  {'learn':>5}  {'rate':>6}  {'kc_corr':>7}  {'decode':>6}")
    print("-" * 40)
    for r in precision_results:
        print(f"{r['noise_sigma']:6.2f}  {'yes' if r['learning'] else 'no':>5}  "
              f"{r['transfer_rate']:6.3f}  {r['kc_correlation']:7.3f}  "
              f"{r['decoding']:6.3f}")

    print("\n=== Receiver Learning: Learning vs Brain Difference ===")
    jitter_results = run_learning_vs_jitter(seeds=args.seeds)
    print(f"{'jitter':>6}  {'learn':>5}  {'rate':>6}  {'kc_corr':>7}  {'decode':>6}")
    print("-" * 40)
    for r in jitter_results:
        print(f"{r['jitter_scale']:6.2f}  {'yes' if r['learning'] else 'no':>5}  "
              f"{r['transfer_rate']:6.3f}  {r['kc_correlation']:7.3f}  "
              f"{r['decoding']:6.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        json_path = write_results(
            experiment="adult_learning_vs_trials",
            description="Adult Drosophila: receiver MBON adaptation vs number of learning trials",
            parameters={"seeds": args.seeds},
            results=trials_results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_learning_vs_trials(trials_results, json_path.with_suffix(".png"))
        print(f"\nTrials results: {json_path}")
        print(f"Trials plot: {plot_path}")

        json_path = write_results(
            experiment="adult_learning_vs_precision",
            description="Adult Drosophila: receiver learning vs hardware precision",
            parameters={"seeds": args.seeds},
            results=precision_results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_learning_vs_precision(precision_results, json_path.with_suffix(".png"))
        print(f"Precision results: {json_path}")
        print(f"Precision plot: {plot_path}")

        json_path = write_results(
            experiment="adult_learning_vs_jitter",
            description="Adult Drosophila: receiver learning vs brain difference",
            parameters={"seeds": args.seeds},
            results=jitter_results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_learning_vs_jitter(jitter_results, json_path.with_suffix(".png"))
        print(f"Jitter results: {json_path}")
        print(f"Jitter plot: {plot_path}")


if __name__ == "__main__":
    main()
