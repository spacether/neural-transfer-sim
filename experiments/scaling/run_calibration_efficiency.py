"""Run calibration sample efficiency sweep across all organisms."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.scaling.calibration_efficiency import run_calibration_efficiency
from analysis.results_writer import write_results
from analysis.plotting import plot_calibration_efficiency

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration sample efficiency sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=0.75)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_calibration_efficiency(seeds=args.seeds, jitter=args.jitter)

    print("\n=== Calibration Sample Efficiency ===")
    for organism, rows in results.items():
        print(f"\n  {organism}:")
        print(f"  {'samples':>8}  {'odor_id':>8}  {'std':>6}")
        print(f"  {'-' * 28}")
        for r in rows:
            print(f"  {r['n_samples']:8d}  {r['odor_id']:8.3f}  {r.get('odor_id_std', 0):6.3f}")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        flat_results = []
        for organism, rows in results.items():
            for r in rows:
                flat_results.append({"organism": organism, **r})
        json_path = write_results(
            experiment="calibration_efficiency",
            description="Calibration sample efficiency: odor identification vs training set size",
            parameters={"seeds": args.seeds, "jitter": args.jitter},
            results=flat_results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_calibration_efficiency(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
