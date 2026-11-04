"""Default runner for ridge vs diagonal calibration comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.olfactory.calibration_comparison.experiment import run_calibration_comparison
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_calibration_comparison

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ridge vs diagonal calibration comparison")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    all_results = []
    for org in ["larva", "adult"]:
        print(f"\n{'='*70}")
        print(f"  {org.upper()} (connectome): Ridge vs Diagonal Calibration")
        print(f"{'='*70}")
        results = run_calibration_comparison(org, "connectome", seeds_per_level=args.seeds)
        all_results.extend(results)
        for r in results:
            print(
                f"  jitter={r['jitter_scale']:.2f}  {r['method']:8s} ({r['n_params']:>3s})  "
                f"transfer={r['transfer_rate']:.2f}  corr={r['correlation']:.3f}  "
                f"rsa={r['rsa']:.3f}  odor_id={r['odor_id']:.3f}"
            )

    if not args.no_save:
        json_path = write_results(
            experiment="calibration_comparison",
            description="Ridge (N² params) vs diagonal (2N params) calibration across organisms",
            parameters={"seeds_per_level": args.seeds, "organisms": ["larva", "adult"]},
            results=all_results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_calibration_comparison(all_results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
