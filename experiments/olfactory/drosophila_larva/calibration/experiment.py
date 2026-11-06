"""Calibration experiment for the Drosophila larva stage.

Fits a linear calibration transform on shared odors, then compares
calibrated transfer against direct broadcast on held-out odors.
Uses connectome-derived model (Winding et al. 2023).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.drosophila_larva import ConnectomeLarvaModel
from neural_transfer.models.drosophila_larva import ConnectomeLarvaVariantGenerator
from neural_transfer.models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE
from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.analysis.metrics import (
    bootstrap_ci,
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
)
import argparse
from pathlib import Path
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_calibration_comparison

ALL_ODOR_NAMES = list(ODOR_PROFILES.keys())

CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_ODOR_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_ODOR_NAMES for i in HOLDOUT_INTENSITIES]

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "odor_id")


def _get_orn_signal(
    model: ConnectomeLarvaModel, odor_name: str, intensity: float
) -> Dict[str, float]:
    rates = model._orn_rates(odor_name, intensity)
    return {orn: float(rates[i]) for i, orn in enumerate(model._orn_order)}


def _run_one_pair(
    sender: ConnectomeLarvaModel,
    receiver: ConnectomeLarvaModel,
) -> Dict:
    orn_order = sender._orn_order

    sender_cal = [_get_orn_signal(sender, name, intensity) for name, intensity in CAL_SET]
    receiver_cal = [_get_orn_signal(receiver, name, intensity) for name, intensity in CAL_SET]

    transform = CalibrationTransform()
    transform.fit(sender_cal, receiver_cal)

    odor_names: List[str] = []
    natural_states: List[Dict[str, float]] = []
    direct_states: List[Dict[str, float]] = []
    cal_states: List[Dict[str, float]] = []
    direct_ok, cal_ok = 0, 0

    for name, intensity in HOLDOUT_SET:
        odor_names.append(name)

        sender_orn = sender._orn_rates(name, intensity)
        sender_sig = {orn: float(sender_orn[i]) for i, orn in enumerate(orn_order)}
        sender_layers = sender._forward(name, intensity)
        sender_beh = sender._behavior_from_output(sender_layers["output"])

        natural_layers = receiver._forward(name, intensity)
        natural_states.append(receiver._layers_to_dict(natural_layers))

        d_result = receiver.simulate_from_receptors(dict(sender_sig))
        direct_states.append(d_result["neurons"])
        direct_ok += d_result["behavior"] == sender_beh

        cal_sig = transform.transform(sender_sig)
        c_result = receiver.simulate_from_receptors(cal_sig)
        cal_states.append(c_result["neurons"])
        cal_ok += c_result["behavior"] == sender_beh

    n = len(HOLDOUT_SET)
    out: Dict = {"n_holdout": n, "n_calibration": len(CAL_SET)}
    for prefix, states, ok in [("direct", direct_states, direct_ok), ("calibrated", cal_states, cal_ok)]:
        out[f"{prefix}_transfer_rate"] = ok / max(1, n)
        out[f"{prefix}_correlation"] = correlation(states, natural_states)
        out[f"{prefix}_mse"] = mean_mse(states, natural_states)
        out[f"{prefix}_rsa"] = representational_similarity(natural_states, states)
        out[f"{prefix}_odor_id"] = odor_identification(odor_names, natural_states, states)
    return out


def _agg_with_ci(pairs: List[Dict]) -> Dict:
    row: Dict = {}
    for prefix in ("direct", "calibrated"):
        for k in METRIC_KEYS:
            key = f"{prefix}_{k}"
            vals = [p[key] for p in pairs]
            mean, lo, hi = bootstrap_ci(vals)
            row[key] = mean
            row[f"{key}_ci_lo"] = lo
            row[f"{key}_ci_hi"] = hi
    return row


def run_calibration_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    sender = ConnectomeLarvaModel()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = ConnectomeLarvaVariantGenerator(jitter_scale=jitter)
        pairs: List[Dict] = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            pairs.append(_run_one_pair(sender, receiver))

        if not pairs:
            continue

        row = _agg_with_ci(pairs)
        row["jitter_scale"] = jitter
        row["valid_seeds"] = len(pairs)
        results.append(row)

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Larva calibration sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_calibration_sweep(seeds_per_level=args.seeds)

    print("=== Drosophila Larva: Calibration vs Direct Broadcast ===")
    print(
        f"{'jitter':>6}  {'d_rate':>6}  {'c_rate':>6}  "
        f"{'d_corr':>6}  {'c_corr':>6}  {'d_mse':>8}  {'c_mse':>8}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['jitter_scale']:6.2f}  "
            f"{r['direct_transfer_rate']:6.3f}  {r['calibrated_transfer_rate']:6.3f}  "
            f"{r['direct_correlation']:6.3f}  {r['calibrated_correlation']:6.3f}  "
            f"{r['direct_mse']:8.2f}  {r['calibrated_mse']:8.2f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="larva_calibration_sweep",
            description="Drosophila larva calibration vs direct broadcast",
            parameters={"seeds_per_level": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_calibration_comparison(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
