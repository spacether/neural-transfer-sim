"""Default calibration sweep for adult Drosophila."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.drosophila_adult.calibration.experiment import run_calibration_sweep
from analysis.results_writer import write_results
from analysis.plotting import plot_calibration_comparison

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult fly calibration sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_calibration_sweep(seeds_per_level=args.seeds)

    print("=== Adult Drosophila: Calibration vs Direct Broadcast ===")
    print(
        f"{'jitter':>6}  {'d_rate':>6}  {'c_rate':>6}  "
        f"{'d_corr':>6}  {'c_corr':>6}  {'d_mse':>8}  {'c_mse':>8}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['jitter_scale']:6.2f}  "
            f"{r['direct_transfer_rate']:6.3f}  {r['calibrated_transfer_rate']:6.3f}  "
            f"{r['direct_correlation']:6.3f}  {r['calibrated_correlation']:6.3f}  "
            f"{r['direct_mse']:8.2f}  {r['calibrated_mse']:8.2f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="adult_calibration_sweep",
            description="Adult Drosophila calibration vs direct broadcast",
            parameters={"seeds_per_level": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_calibration_comparison(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
