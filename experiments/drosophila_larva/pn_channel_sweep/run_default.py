"""Default PN channel sweep for Drosophila larva."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.drosophila_larva.pn_channel_sweep.experiment import run_pn_channel_sweep
from analysis.results_writer import write_results
from analysis.plotting import plot_pn_channel_sweep

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Larva PN channel sweep (2D)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--channel-seeds", type=int, default=3)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_pn_channel_sweep(
        jitter=args.jitter,
        seeds=args.seeds,
        channel_seeds=args.channel_seeds,
    )

    print("=== Drosophila Larva: PN Channel Sweep (2D) ===")
    print(f"{'read':>5}  {'write':>5}  {'corr':>6}  {'behav':>6}")
    print("-" * 30)
    for r in results:
        print(f"{r['n_read']:5d}  {r['n_write']:5d}  "
              f"{r['correlation']:6.3f}  {r['transfer_rate']:6.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="larva_pn_channel_sweep",
            description="Drosophila larva KC correlation vs read/write PN count",
            parameters={
                "jitter": args.jitter,
                "seeds": args.seeds,
                "channel_seeds": args.channel_seeds,
            },
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_pn_channel_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
