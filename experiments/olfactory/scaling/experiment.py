"""Cross-organism scaling comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import argparse
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_scaling_comparison


def _worm_metrics(seeds: int = 5, jitter: float = 1.0) -> Dict:
    """Run worm transfer metrics using the Cook2019 model."""
    from neural_transfer.models.c_elegans.cook2019 import OLFACTORY_RECEPTORS
    from experiments.olfactory.c_elegans.calibration.experiment import (
        run_calibration_sweep,
    )

    n_receptor_classes = len(set(
        cls for neurons in OLFACTORY_RECEPTORS.values() for cls in neurons
    ))

    results = run_calibration_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
        model_name="cook2019",
    )
    beh_match = results[0]["direct_transfer_rate"] if results else 0.0

    return {
        "organism": "C. elegans",
        "receptor_types": n_receptor_classes,
        "total_receptors": n_receptor_classes,
        "glomeruli": 0,
        "kc_equivalent": 0,
        "bottleneck": "none",
        "signal_mode": "graded",
        "broadcast_behavioral_match": beh_match,
    }


def _larva_metrics(seeds: int = 5, jitter: float = 1.0) -> Dict:
    """Run larva transfer metrics using the connectome model."""
    from neural_transfer.models.drosophila_larva import ConnectomeLarvaModel
    from experiments.olfactory.drosophila_larva.calibration.experiment import run_calibration_sweep

    model = ConnectomeLarvaModel()
    results = run_calibration_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
    )
    beh_match = results[0]["direct_transfer_rate"] if results else 0.0

    return {
        "organism": "D. larva",
        "receptor_types": len(model._orn_order),
        "total_receptors": len(model._orn_order),
        "glomeruli": len(model._orn_order),
        "kc_equivalent": model._n_kc,
        "bottleneck": "glomerular (1:1)",
        "signal_mode": "spiking",
        "broadcast_behavioral_match": beh_match,
    }


def _adult_metrics(seeds: int = 5, jitter: float = 1.0) -> Dict:
    """Run adult fly transfer metrics using the connectome model."""
    from neural_transfer.models.drosophila_adult import ConnectomeAdultModel
    from experiments.olfactory.drosophila_adult.calibration.experiment import run_calibration_sweep

    model = ConnectomeAdultModel()
    results = run_calibration_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
    )
    beh_match = results[0]["direct_transfer_rate"] if results else 0.0

    return {
        "organism": "D. adult",
        "receptor_types": model._n_glom,
        "total_receptors": 1117,
        "glomeruli": model._n_glom,
        "kc_equivalent": model._n_kc,
        "bottleneck": "glomerular (23:1)",
        "signal_mode": "spiking",
        "broadcast_behavioral_match": beh_match,
    }


def _zebrafish_metrics(seeds: int = 5, jitter: float = 1.0) -> Dict:
    """Run zebrafish transfer metrics."""
    from neural_transfer.models.zebrafish import (
        ZebrafishModel, N_OR_TYPES, N_TOTAL_ORNS, N_GLOMERULI, N_DP,
    )
    from experiments.olfactory.zebrafish.calibration.experiment import run_calibration_sweep

    results = run_calibration_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
    )
    beh_match = results[0]["direct_transfer_rate"] if results else 0.0

    return {
        "organism": "Zebrafish",
        "receptor_types": N_OR_TYPES,
        "total_receptors": N_TOTAL_ORNS,
        "glomeruli": N_GLOMERULI,
        "kc_equivalent": N_DP,
        "bottleneck": "glomerular (3:1)",
        "signal_mode": "spiking",
        "broadcast_behavioral_match": beh_match,
    }


def run_scaling_comparison(seeds: int = 5, jitter: float = 1.0) -> List[Dict]:
    """Compute transfer metrics for all four organisms."""
    return [
        _worm_metrics(seeds=seeds, jitter=jitter),
        _larva_metrics(seeds=seeds, jitter=jitter),
        _adult_metrics(seeds=seeds, jitter=jitter),
        _zebrafish_metrics(seeds=seeds, jitter=jitter),
    ]


def _load_calibration_thresholds() -> Dict[str, int]:
    """Load measured calibration sample thresholds from calibration_efficiency results."""
    import json
    from pathlib import Path

    path = Path(__file__).parent / "results" / "calibration_efficiency.json"
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    thresholds: Dict[str, int] = {}
    from collections import defaultdict
    by_org = defaultdict(list)
    for r in data["results"]:
        by_org[r["organism"]].append(r)

    for org, rows in by_org.items():
        for r in sorted(rows, key=lambda x: x["n_samples"]):
            if r["odor_id"] >= 0.90:
                thresholds[org] = r["n_samples"]
                break
    return thresholds


def estimate_channel_requirements(results: List[Dict]) -> List[Dict]:
    """Estimate minimum channel counts and calibration samples.

    Uses measured calibration efficiency thresholds when available,
    otherwise falls back to a heuristic estimate.
    """
    cal_thresholds = _load_calibration_thresholds()
    enriched = []
    for r in results:
        row = dict(r)
        n_types = r["receptor_types"]
        row["min_channels_90"] = max(1, int(np.ceil(n_types * 0.7)))
        row["calibration_samples"] = cal_thresholds.get(
            r["organism"], max(10, n_types * 3)
        )
        enriched.append(row)
    return enriched


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-organism scaling comparison")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=1.0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    print("Running scaling comparison across all organisms...")
    raw_results = run_scaling_comparison(seeds=args.seeds, jitter=args.jitter)
    results = estimate_channel_requirements(raw_results)

    print("\n=== Cross-Organism Scaling Comparison ===")
    print(f"{'organism':>12}  {'types':>5}  {'total':>6}  {'glom':>5}  {'KCs':>5}  "
          f"{'beh':>5}  {'ch_90':>5}  {'cal':>5}  {'mode':>8}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['organism']:>12}  {r['receptor_types']:5d}  "
            f"{r['total_receptors']:6d}  {r['glomeruli']:5d}  "
            f"{r['kc_equivalent']:5d}  "
            f"{r['broadcast_behavioral_match']:5.2f}  "
            f"{r['min_channels_90']:5d}  "
            f"{r['calibration_samples']:5d}  "
            f"{r['signal_mode']:>8}"
        )

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = write_results(
            experiment="scaling_comparison",
            description="Cross-organism scaling: transfer requirements vs complexity",
            parameters={"seeds": args.seeds, "jitter": args.jitter},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_scaling_comparison(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
