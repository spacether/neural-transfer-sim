"""Default calibration experiment for zebrafish."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.olfactory.zebrafish.calibration.experiment import run_calibration_sweep
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_calibration_comparison

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zebrafish calibration experiment")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_calibration_sweep(seeds_per_level=args.seeds)

    print("=== Zebrafish: Calibration vs Direct Broadcast ===")
    print(f"{'jitter':>6}  {'d_rate':>6}  {'c_rate':>6}  "
          f"{'d_corr':>6}  {'c_corr':>6}  {'d_mse':>9}  {'c_mse':>9}")
    print("-" * 60)
    for r in results:
        print(f"{r['jitter_scale']:6.2f}  {r['direct_transfer_rate']:6.3f}  "
              f"{r['calibrated_transfer_rate']:6.3f}  "
              f"{r['direct_correlation']:6.3f}  {r['calibrated_correlation']:6.3f}  "
              f"{r['direct_mse']:9.2f}  {r['calibrated_mse']:9.2f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="zebrafish_calibration_sweep",
            description="Zebrafish: calibration vs direct broadcast on held-out odors",
            parameters={"seeds": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_calibration_comparison(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
