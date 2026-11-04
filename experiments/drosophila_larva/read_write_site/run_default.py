"""Default read/write site sweep for Drosophila larva."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.drosophila_larva.read_write_site.experiment import run_site_sweep
from analysis.results_writer import write_results
from analysis.plotting import plot_read_write_site

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Larva read/write site sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_site_sweep(seeds_per_level=args.seeds)

    print("=== Drosophila Larva: Read Site vs Write Site ===")
    print(f"{'read':>5}  {'write':>5}  {'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'decode':>6}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['read_site']:>5}  {r['write_site']:>5}  "
            f"{r['jitter_scale']:6.2f}  {r['transfer_rate']:6.3f}  "
            f"{r['correlation']:6.3f}  {r['mse']:8.2f}  {r['decoding']:6.3f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="larva_read_write_site",
            description="Drosophila larva transfer quality by read and write site pairing",
            parameters={"seeds_per_level": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_read_write_site(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
