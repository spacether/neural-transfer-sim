"""Default channel sweep experiment for zebrafish."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.zebrafish.channel_sweep.experiment import run_channel_sweep
from analysis.results_writer import write_results
from analysis.plotting import plot_channel_sweep

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zebrafish channel sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_channel_sweep(seeds_per_level=args.seeds)

    print("=== Zebrafish: ORN Channel Sweep ===")
    print(f"{'ch':>3}  {'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>9}  {'rsa':>6}  {'decode':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['n_channels']:3d}  {r['jitter_scale']:6.2f}  {r['transfer_rate']:6.3f}  "
              f"{r['correlation']:6.3f}  {r['mse']:9.2f}  {r['rsa']:6.3f}  {r['decoding']:6.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="zebrafish_channel_sweep",
            description="Zebrafish: transfer quality vs number of connected ORN types",
            parameters={"seeds": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
