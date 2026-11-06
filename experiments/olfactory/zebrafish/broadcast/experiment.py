"""Direct broadcast baseline for the zebrafish stage."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.zebrafish import (
    ZebrafishModel, ODOR_PROFILES, ODOR_VALENCE, OR_TYPES,
)
from neural_transfer.models.zebrafish.variants import ZebrafishVariantGenerator
from neural_transfer.link.receptor_to_receptor import ReceptorToReceptorBaseline
from neural_transfer.analysis.metrics import (
    correlation, mean_mse, representational_similarity,
    odor_identification,
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


def run_broadcast_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    sender = ZebrafishModel()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = ZebrafishVariantGenerator(jitter_scale=jitter)
        seed_metrics: List[Dict[str, float]] = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue

            baseline = ReceptorToReceptorBaseline(
                sender=sender, receiver=receiver,
                source_neurons=tuple(OR_TYPES),
            )

            odor_types: List[str] = []
            odor_names: List[str] = []
            natural_states: List[Dict[str, float]] = []
            transferred_states: List[Dict[str, float]] = []
            successes = 0

            for odor in ODOR_SET:
                name = odor["odor_name"]
                intensity = odor["odor_intensity"]
                otype = ODOR_VALENCE[name]
                odor_types.append(otype)
                odor_names.append(name)

                sender_orn = sender._orn_type_rates(name, intensity)
                sender_signal = {orn: float(sender_orn[i]) for i, orn in enumerate(OR_TYPES)}
                sender_layers = sender._forward(name, intensity)
                sender_beh = sender._behavior_from_output(sender_layers["output"])

                receiver_signal = baseline.copy_signal(sender_signal)
                result = receiver.simulate_from_receptors(receiver_signal)
                if result["behavior"] == sender_beh:
                    successes += 1

                natural_layers = receiver._forward(name, intensity)
                natural_state = receiver._layers_to_dict(natural_layers)
                natural_states.append(natural_state)
                transferred_states.append(result["neurons"])

            if len(ODOR_SET) > 0:
                natural_dp = [{k: v for k, v in s.items() if k.startswith("Dp_")} for s in natural_states]
                transferred_dp = [{k: v for k, v in s.items() if k.startswith("Dp_")} for s in transferred_states]
                seed_metrics.append({
                    "transfer_rate": successes / len(ODOR_SET),
                    "correlation": correlation(transferred_states, natural_states),
                    "mse": mean_mse(transferred_states, natural_states),
                    "rsa": representational_similarity(natural_states, transferred_states),
                    "odor_id": odor_identification(odor_names, natural_dp, transferred_dp),
                })

        if not seed_metrics:
            continue

        results.append({
            "jitter_scale": jitter,
            "valid_seeds": len(seed_metrics),
            **{
                key: float(np.mean([m[key] for m in seed_metrics]))
                for key in ("transfer_rate", "correlation", "mse", "rsa", "odor_id")
            },
        })

    return results


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
