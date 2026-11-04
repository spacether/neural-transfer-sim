"""Default mitral cell channel sweep experiment for zebrafish."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.zebrafish.pn_channel_sweep.experiment import run_mitral_channel_sweep
from analysis.results_writer import write_results
from analysis.plotting import plot_pn_channel_sweep

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zebrafish mitral cell channel sweep")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_mitral_channel_sweep(seeds=args.seeds)

    print("=== Zebrafish: Mitral Cell Channel Sweep ===")
    print(f"{'read':>5}  {'write':>5}  {'rate':>6}  {'corr':>6}  {'decode':>6}")
    print("-" * 35)
    for r in results:
        print(f"{r['n_read']:5d}  {r['n_write']:5d}  {r['transfer_rate']:6.3f}  "
              f"{r['correlation']:6.3f}  {r['decoding']:6.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="zebrafish_mitral_channel_sweep",
            description="Zebrafish: 2D sweep of read/write mitral cell counts",
            parameters={"seeds": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_pn_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
