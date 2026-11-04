"""Default hardware boundary sweep: both models, all axes."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.hardware_boundary.experiment import MODEL_CLASSES, run_hardware_sweep
from analysis.results_writer import write_results
from analysis.plotting import plot_hardware_boundary

RESULTS_DIR = Path(__file__).parent / "results"

AXIS_LABELS = {
    "read_noise": ("noise_sigma", "sigma"),
    "write_precision": ("precision_bits", "bits"),
    "channel_dropout": ("dropout_fraction", "drop"),
    "transmit_delay": ("delay_ms", "ms"),
    "combined": ("step", "step"),
}


def _print_results(all_results: dict, model_name: str) -> None:
    print(f"\n=== {model_name} ===")
    for axis, rows in all_results.items():
        param_key, short = AXIS_LABELS[axis]
        print(f"\n  --- {axis} ---")
        print(f"  {short:>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'rsa':>6}  {'decode':>6}")
        print(f"  {'-' * 48}")
        for row in rows:
            val = row[param_key]
            val_str = f"{val:6.2f}" if isinstance(val, float) and val != float("inf") else f"{val!s:>6}"
            print(
                f"  {val_str}  "
                f"{row['transfer_rate']:6.3f}  "
                f"{row['correlation']:6.3f}  "
                f"{row['mse']:8.4f}  "
                f"{row['rsa']:6.3f}  "
                f"{row['decoding']:6.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware boundary sweep")
    parser.add_argument("--model", choices=list(MODEL_CLASSES), nargs="+",
                        default=list(MODEL_CLASSES),
                        help="Which model(s) to run (default: all)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=0.75,
                        help="Brain difference level (default: 0.75)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    for model_name in args.model:
        all_results = run_hardware_sweep(
            model_name=model_name,
            jitter=args.jitter,
            seeds=args.seeds,
        )
        _print_results(all_results, model_name)

        if not args.no_save:
            for axis, rows in all_results.items():
                json_path = write_results(
                    experiment=f"hardware_{axis}_{model_name}",
                    description=f"Hardware boundary: {axis} sweep ({model_name})",
                    parameters={
                        "model": model_name,
                        "axis": axis,
                        "jitter": args.jitter,
                        "seeds": args.seeds,
                    },
                    results=rows,
                    output_dir=RESULTS_DIR,
                )
                print(f"  {axis} results: {json_path}")

            plot_path = RESULTS_DIR / f"hardware_boundary_{model_name}.png"
            plot_hardware_boundary(all_results, plot_path)
            print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
