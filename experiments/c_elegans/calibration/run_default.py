"""Default calibration sweep: 7 jitter levels, 5 seeds each, both models."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.c_elegans.calibration.experiment import (
    MODEL_CLASSES,
    run_calibration_sweep,
)
from analysis.results_writer import write_results
from analysis.plotting import plot_calibration_comparison

RESULTS_DIR = Path(__file__).parent / "results"


def _print_results(results, model_name: str) -> None:
    print(f"\n=== {model_name} ===")
    header = (
        f"{'jitter':>6}  "
        f"{'d_rate':>6} {'c_rate':>6}  "
        f"{'d_corr':>6} {'c_corr':>6}  "
        f"{'d_mse':>8} {'c_mse':>8}  "
        f"{'d_rsa':>6} {'c_rsa':>6}  "
        f"{'d_dec':>6} {'c_dec':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['jitter_scale']:6.2f}  "
            f"{row['direct_transfer_rate']:6.3f} {row['calibrated_transfer_rate']:6.3f}  "
            f"{row['direct_correlation']:6.3f} {row['calibrated_correlation']:6.3f}  "
            f"{row['direct_mse']:8.4f} {row['calibrated_mse']:8.4f}  "
            f"{row['direct_rsa']:6.3f} {row['calibrated_rsa']:6.3f}  "
            f"{row['direct_decoding']:6.3f} {row['calibrated_decoding']:6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration vs receptor-to-receptor sweep")
    parser.add_argument("--jitter", type=float, nargs="+", default=None,
                        help="Jitter levels as multiples of biological variability (default: 0.0 0.25 0.50 0.75 1.0 1.25 1.5)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Receiver variants per jitter level (default: 5)")
    parser.add_argument("--model", choices=list(MODEL_CLASSES), nargs="+",
                        default=list(MODEL_CLASSES),
                        help="Which model(s) to run (default: all)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print only, don't write results or plot")
    args = parser.parse_args()

    jitter_levels = args.jitter or [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    for model_name in args.model:
        results = run_calibration_sweep(
            jitter_levels=jitter_levels,
            seeds_per_level=args.seeds,
            model_name=model_name,
        )
        _print_results(results, model_name)

        if not args.no_save:
            json_path = write_results(
                experiment=f"calibration_sweep_{model_name}",
                description=f"Calibrated vs receptor-to-receptor transfer ({model_name})",
                parameters={
                    "model": model_name,
                    "jitter_levels": jitter_levels,
                    "seeds_per_level": args.seeds,
                },
                results=results,
                output_dir=RESULTS_DIR,
            )
            plot_path = plot_calibration_comparison(
                results, json_path.with_suffix(".png"),
            )
            print(f"\nResults saved to {json_path}")
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
