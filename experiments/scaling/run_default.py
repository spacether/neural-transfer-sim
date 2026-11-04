"""Run cross-organism scaling comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.scaling.experiment import (
    run_scaling_comparison,
    estimate_channel_requirements,
)
from analysis.results_writer import write_results
from analysis.plotting import plot_scaling_comparison

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-organism scaling comparison")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=0.75)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    print("Running scaling comparison across all three organisms...")
    raw_results = run_scaling_comparison(seeds=args.seeds, jitter=args.jitter)
    results = estimate_channel_requirements(raw_results)

    print("\n=== Cross-Organism Scaling Comparison ===")
    print(f"{'organism':>12}  {'types':>5}  {'total':>6}  {'glom':>5}  {'KCs':>5}  "
          f"{'beh':>5}  {'ch_90':>5}  {'cal':>5}  {'mode':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['organism']:>12}  {r['receptor_types']:5d}  "
            f"{r['total_receptors']:6d}  {r['glomeruli']:5d}  "
            f"{r['kc_equivalent']:5d}  "
            f"{r['broadcast_behavioral_match']:5.2f}  "
            f"{r['min_channels_90']:5d}  "
            f"{r['calibration_samples']:5d}  "
            f"{r['signal_mode']:>8}"
        )

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="scaling_comparison",
            description="Cross-organism scaling: transfer requirements vs complexity",
            parameters={"seeds": args.seeds, "jitter": args.jitter},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_scaling_comparison(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
