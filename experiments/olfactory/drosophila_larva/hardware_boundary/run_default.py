"""Default hardware boundary sweep for Drosophila larva."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.olfactory.drosophila_larva.hardware_boundary.experiment import run_hardware_sweep
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Larva hardware boundary sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=0.75)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    all_results = run_hardware_sweep(jitter=args.jitter, seeds=args.seeds)

    print("=== Drosophila Larva: Hardware Boundary ===")
    for axis, rows in all_results.items():
        param_key, short = AXIS_LABELS[axis]
        print(f"\n  --- {axis} ---")
        print(f"  {short:>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'odor_id':>6}")
        print(f"  {'-' * 48}")
        for row in rows:
            val = row[param_key]
            val_str = f"{val:6.2f}" if isinstance(val, float) and val != float("inf") else f"{val!s:>6}"
            print(
                f"  {val_str}  {row['transfer_rate']:6.3f}  "
                f"{row['correlation']:6.3f}  {row['mse']:8.2f}  "
                f"{row['odor_id']:6.3f}"
            )

    if not args.no_save:
        for axis, rows in all_results.items():
            json_path = write_results(
                experiment=f"larva_hardware_{axis}",
                description=f"Drosophila larva hardware boundary: {axis} sweep",
                parameters={"axis": axis, "jitter": args.jitter, "seeds": args.seeds},
                results=rows,
                output_dir=RESULTS_DIR,
            )
            print(f"  {axis} results: {json_path}")

        plot_path = RESULTS_DIR / "hardware_boundary_larva.png"
        plot_hardware_boundary(all_results, plot_path)
        print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
