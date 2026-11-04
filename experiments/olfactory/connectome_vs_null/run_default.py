"""Default runner for connectome vs shuffled-null comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.olfactory.connectome_vs_null.experiment import run_comparison
from neural_transfer.analysis.results_writer import write_results

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Connectome vs shuffled-null broadcast comparison")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    all_results = []
    for org in ["larva", "adult"]:
        print(f"\n{'='*60}")
        print(f"  {org.upper()}: Connectome vs Shuffled Null")
        print(f"{'='*60}")
        results = run_comparison(org, seeds_per_level=args.seeds)
        all_results.extend(results)
        for r in results:
            print(
                f"  jitter={r['jitter_scale']:.2f}  {r['model']:15s}  "
                f"transfer={r['transfer_rate']:.2f}  corr={r['correlation']:.3f}  "
                f"rsa={r['rsa']:.3f}  odor_id={r['odor_id']:.3f}"
            )

    if not args.no_save:
        json_path = write_results(
            experiment="connectome_vs_null",
            description="Broadcast transfer: real connectome vs shuffled-null wiring topology",
            parameters={"seeds_per_level": args.seeds, "organisms": ["larva", "adult"]},
            results=all_results,
            output_dir=RESULTS_DIR,
        )
        print(f"\nResults: {json_path}")


if __name__ == "__main__":
    main()
