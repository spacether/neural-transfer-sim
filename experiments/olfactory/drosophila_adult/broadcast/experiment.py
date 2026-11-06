"""Direct broadcast baseline for the adult Drosophila stage.

Copies sender ORN-type activity to receiver ORN layer with no learned
translation, across a range of brain-difference levels.

Uses the connectome-based adult model (FlyWire v783, Dorkenwald et al. 2024).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.drosophila_adult import ConnectomeAdultModel
from neural_transfer.models.drosophila_adult import ConnectomeAdultVariantGenerator
from neural_transfer.models.drosophila_adult import ODOR_PROFILES, ODOR_VALENCE
from neural_transfer.analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
    bootstrap_ci,
)
import argparse
from pathlib import Path
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_receptor_to_receptor_sweep


ODOR_SET: List[Dict[str, object]] = [
    {"odor_name": name, "odor_intensity": intensity}
    for name in ODOR_PROFILES
    for intensity in (0.5, 1.0, 1.5)
]

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "odor_id")


def _agg_with_ci(seed_metrics: List[Dict[str, float]]) -> Dict:
    row: Dict = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean, lo, hi = bootstrap_ci(vals)
        row[k] = mean
        row[f"{k}_ci_lo"] = lo
        row[f"{k}_ci_hi"] = hi
    return row


def run_broadcast_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    sender = ConnectomeAdultModel()
    receptor_names = list(sender._receptor_names)
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = ConnectomeAdultVariantGenerator(jitter_scale=jitter)
        seed_metrics: List[Dict[str, float]] = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue

            odor_names: List[str] = []
            natural_states: List[Dict[str, float]] = []
            transferred_states: List[Dict[str, float]] = []
            successes = 0

            for odor in ODOR_SET:
                name = odor["odor_name"]
                intensity = odor["odor_intensity"]
                odor_names.append(name)

                sender_orn = sender._orn_rates(name, intensity)
                sender_signal = {orn: float(sender_orn[i]) for i, orn in enumerate(receptor_names)}
                sender_layers = sender._forward(name, intensity)
                sender_beh = sender._behavior_from_output(sender_layers["output"])

                result = receiver.simulate_from_receptors(sender_signal)
                if result["behavior"] == sender_beh:
                    successes += 1

                natural_layers = receiver._forward(name, intensity)
                natural_state = receiver._layers_to_dict(natural_layers)
                natural_states.append(natural_state)
                transferred_states.append(result["neurons"])

            if len(ODOR_SET) > 0:
                natural_kc = [{k: v for k, v in s.items() if k.startswith("KC_")} for s in natural_states]
                transferred_kc = [{k: v for k, v in s.items() if k.startswith("KC_")} for s in transferred_states]
                seed_metrics.append({
                    "transfer_rate": successes / len(ODOR_SET),
                    "correlation": correlation(transferred_states, natural_states),
                    "mse": mean_mse(transferred_states, natural_states),
                    "rsa": representational_similarity(natural_states, transferred_states),
                    "odor_id": odor_identification(odor_names, natural_kc, transferred_kc),
                })

        if not seed_metrics:
            continue

        row: Dict = {
            "jitter_scale": jitter,
            "valid_seeds": len(seed_metrics),
        }
        row.update(_agg_with_ci(seed_metrics))
        results.append(row)

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult fly broadcast baseline sweep")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_broadcast_sweep(seeds_per_level=args.seeds)

    print("=== Adult Drosophila: Direct Broadcast Baseline ===")
    print(f"{'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'rsa':>6}  {'odor_id':>6}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['jitter_scale']:6.2f}  {r['transfer_rate']:6.3f}  "
            f"{r['correlation']:6.3f}  {r['mse']:8.2f}  "
            f"{r['rsa']:6.3f}  {r['odor_id']:6.3f}"
        )

    if not args.no_save:
        json_path = write_results(
            experiment="adult_broadcast_sweep",
            description="Adult Drosophila direct broadcast baseline: transfer quality vs brain difference",
            parameters={"seeds_per_level": args.seeds},
            results=results,
            output_dir=RESULTS_DIR,
        )
        plot_path = plot_receptor_to_receptor_sweep(results, json_path.with_suffix(".png"))
        print(f"\nResults: {json_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
