"""Default channel sweep for adult Drosophila."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.drosophila_adult.channel_sweep.experiment import run_channel_sweep
from analysis.results_writer import write_results
from analysis.plotting import plot_channel_sweep

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult fly channel sweep")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--channel-seeds", type=int, default=3)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_channel_sweep(
        seeds_per_level=args.seeds,
        channel_seeds=args.channel_seeds,
    )

    print("=== Adult Drosophila: ORN Channel Sweep ===")
    print(f"{'ch':>3}  {'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'rsa':>6}  {'odor_id':>6}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['n_channels']:3d}  {r['jitter_scale']:6.2f}  "
            f"{r['transfer_rate']:6.3f}  {r['correlation']:6.3f}  "
            f"{r['mse']:8.2f}  {r['rsa']:6.3f}  {r['odor_id']:6.3f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="adult_channel_sweep",
            description="Adult Drosophila transfer quality vs number of connected ORN types",
            parameters={
                "seeds_per_level": args.seeds,
                "channel_seeds": args.channel_seeds,
            },
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
