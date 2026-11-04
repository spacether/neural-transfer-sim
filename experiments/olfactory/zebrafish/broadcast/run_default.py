"""Default broadcast experiment for zebrafish."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.olfactory.zebrafish.broadcast.experiment import run_broadcast_sweep
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_receptor_to_receptor_sweep

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Zebrafish broadcast baseline")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_broadcast_sweep(seeds_per_level=args.seeds)

    print("=== Zebrafish: Direct Broadcast Baseline ===")
    print(f"{'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>9}  {'rsa':>6}  {'odor_id':>6}")
    print("-" * 55)
    for r in results:
        print(f"{r['jitter_scale']:6.2f}  {r['transfer_rate']:6.3f}  "
              f"{r['correlation']:6.3f}  {r['mse']:9.2f}  "
              f"{r['rsa']:6.3f}  {r['odor_id']:6.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="zebrafish_broadcast_sweep",
            description="Zebrafish: direct broadcast transfer vs brain difference",
            parameters={"seeds": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_receptor_to_receptor_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
