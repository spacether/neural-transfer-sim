"""Default receptor-to-receptor sweep: 7 jitter levels, 5 seeds each, both models."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.c_elegans.receptor_to_receptor_broadcast.experiment import (
    MODEL_CLASSES,
    run_receptor_to_receptor_sweep,
)
from analysis.results_writer import write_results
from analysis.plotting import plot_receptor_to_receptor_sweep

RESULTS_DIR = Path(__file__).parent / "results"


def _print_results(results, model_name: str) -> None:
    print(f"\n=== {model_name} ===")
    print(f"{'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'rsa':>6}  {'decode':>6}")
    print("-" * 50)
    for row in results:
        print(
            f"{row['jitter_scale']:6.2f}  "
            f"{row['transfer_rate']:6.3f}  "
            f"{row['correlation']:6.3f}  "
            f"{row['mse']:8.4f}  "
            f"{row['rsa']:6.3f}  "
            f"{row['decoding']:6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Receptor-to-receptor baseline sweep")
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
        results = run_receptor_to_receptor_sweep(
            jitter_levels=jitter_levels,
            seeds_per_level=args.seeds,
            model_name=model_name,
        )
        _print_results(results, model_name)

        if not args.no_save:
            json_path = write_results(
                experiment=f"receptor_to_receptor_sweep_{model_name}",
                description=f"Transfer quality vs brain difference using receptor-to-receptor baseline ({model_name})",
                parameters={
                    "model": model_name,
                    "jitter_levels": jitter_levels,
                    "seeds_per_level": args.seeds,
                },
                results=results,
                output_dir=RESULTS_DIR,
            )
            plot_path = plot_receptor_to_receptor_sweep(
                results, json_path.with_suffix(".png"),
            )
            print(f"\nResults saved to {json_path}")
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
