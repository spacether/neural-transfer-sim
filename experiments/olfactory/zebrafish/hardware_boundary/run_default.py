"""Default hardware boundary experiment for zebrafish."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.olfactory.zebrafish.hardware_boundary.experiment import run_hardware_sweep
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_hardware_boundary

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zebrafish hardware boundary experiment")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_hardware_sweep(seeds=args.seeds)

    for axis, rows in results.items():
        print(f"\n  --- {axis} ---")
        if not rows:
            continue
        param_key = [k for k in rows[0] if k not in ("transfer_rate", "correlation", "mse", "rsa", "odor_id", "step")][0]
        print(f"  {param_key:>8}  {'rate':>6}  {'corr':>6}  {'mse':>9}  {'odor_id':>7}")
        print("  " + "-" * 50)
        for r in rows:
            print(f"  {r[param_key]:8.2f}  {r['transfer_rate']:6.3f}  "
                  f"{r['correlation']:6.3f}  {r['mse']:9.2f}  {r['odor_id']:7.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for axis, rows in results.items():
            json_path = write_results(
                experiment=f"zebrafish_hardware_{axis}",
                description=f"Zebrafish: hardware boundary — {axis}",
                parameters={"seeds": args.seeds},
                results=rows,
                output_dir=RESULTS_DIR,
            )
        plot_path = plot_hardware_boundary(results, RESULTS_DIR / "hardware_boundary_zebrafish.png")
        print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
