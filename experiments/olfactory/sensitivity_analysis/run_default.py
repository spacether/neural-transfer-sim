"""Default runner for parameter sensitivity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.olfactory.sensitivity_analysis.experiment import run_sensitivity
from neural_transfer.analysis.results_writer import write_results

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sensitivity analysis")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    all_results = []
    for org in ["larva", "adult"]:
        print(f"\n{'='*70}")
        print(f"  {org.upper()}: Parameter Sensitivity (broadcast, jitter={args.jitter})")
        print(f"{'='*70}")

        results = run_sensitivity(org, jitter_scale=args.jitter, n_seeds=args.seeds)
        all_results.extend(results)

        current_param = None
        for r in results:
            if r["parameter"] != current_param:
                current_param = r["parameter"]
                print(f"\n  --- {current_param} ---")
            marker = " *" if r["is_default"] else ""
            print(
                f"    {r['value']:8.3f}  "
                f"transfer={r['transfer_rate']:.2f}  "
                f"corr={r['correlation']:.3f}  "
                f"rsa={r['rsa']:.3f}{marker}"
            )

    if not args.no_save:
        json_path = write_results(
            experiment="sensitivity_analysis",
            description="Parameter sensitivity: broadcast transfer vs free-parameter perturbation",
            parameters={
                "n_seeds": args.seeds,
                "jitter_scale": args.jitter,
                "organisms": ["larva", "adult"],
            },
            results=all_results,
            output_dir=RESULTS_DIR,
        )
        print(f"\nResults: {json_path}")


if __name__ == "__main__":
    main()
