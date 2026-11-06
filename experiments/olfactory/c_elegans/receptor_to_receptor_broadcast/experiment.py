"""Receptor-to-receptor baseline experiment for the worm stage.

Measures transfer quality when the sender's receptor activity is copied
directly to the receiver's receptor layer with no learned translation,
across a range of brain-difference levels.

Reports: correlation (Pearson r), MSE, RSA, odor identification accuracy,
and categorical behavioral match — each comparing the receiver's post-injection
state against its natural response to the same odor.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
from neural_transfer.models.c_elegans.cook2019 import Cook2019WormModel
from neural_transfer.models.c_elegans.variants import WormVariantGenerator
from neural_transfer.link.receptor_to_receptor import ReceptorToReceptorBaseline
from neural_transfer.analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
)
import argparse
from pathlib import Path
from neural_transfer.analysis.results_writer import write_results
from neural_transfer.analysis.plotting import plot_receptor_to_receptor_sweep

MODEL_CLASSES = {
    "hand_built": HandBuiltWormModel,
    "cook2019": Cook2019WormModel,
}

ODOR_SET: List[Dict[str, object]] = [
    {"odor_intensity": intensity, "odor_type": odor_type}
    for odor_type in ("attractive", "repellent")
    for intensity in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
]


def run_receptor_to_receptor_sweep(
    jitter_levels: List[float] | None = None,
    odors: List[Dict[str, object]] | None = None,
    seeds_per_level: int = 5,
    model_name: str = "hand_built",
) -> List[Dict]:
    """Sweep brain difference and measure receptor-to-receptor transfer quality."""
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]
    if odors is None:
        odors = ODOR_SET

    ModelClass = MODEL_CLASSES[model_name]
    sender = ModelClass()
    generator = WormVariantGenerator()
    results: List[Dict] = []

    for jitter in jitter_levels:
        generator.jitter_scale = jitter
        seed_metrics: List[Dict[str, float]] = []

        for seed in range(seeds_per_level):
            receiver = generator.generate_model_variant(sender, seed=seed)
            baseline = ReceptorToReceptorBaseline(sender=sender, receiver=receiver)

            odor_types: List[str] = []
            natural_states: List[Dict[str, float]] = []
            transferred_states: List[Dict[str, float]] = []
            successes = 0

            for odor in odors:
                intensity, otype = odor["odor_intensity"], odor["odor_type"]
                odor_types.append(otype)

                trial = baseline.evaluate_transfer(
                    odor_intensity=intensity, odor_type=otype,
                )
                if trial["success"]:
                    successes += 1

                natural_states.append(receiver.network_state(intensity, otype))
                transferred_states.append(
                    receiver.simulate_from_receptors(
                        trial["receiver_activity"], otype
                    )["neurons"]
                )

            seed_metrics.append({
                "transfer_rate": successes / len(odors),
                "correlation": correlation(transferred_states, natural_states),
                "mse": mean_mse(transferred_states, natural_states),
                "rsa": representational_similarity(natural_states, transferred_states),
                "odor_id": odor_identification(odor_types, natural_states, transferred_states),
            })

        results.append({
            "model": model_name,
            "jitter_scale": jitter,
            **{
                key: float(np.mean([m[key] for m in seed_metrics]))
                for key in ("transfer_rate", "correlation", "mse", "rsa", "odor_id")
            },
            "seeds": seeds_per_level,
        })

    return results


RESULTS_DIR = Path(__file__).parent / "results"


def _print_results(results, model_name: str) -> None:
    print(f"\n=== {model_name} ===")
    print(f"{'jitter':>6}  {'rate':>6}  {'corr':>6}  {'mse':>8}  {'rsa':>6}  {'odor_id':>6}")
    print("-" * 50)
    for row in results:
        print(
            f"{row['jitter_scale']:6.2f}  "
            f"{row['transfer_rate']:6.3f}  "
            f"{row['correlation']:6.3f}  "
            f"{row['mse']:8.4f}  "
            f"{row['rsa']:6.3f}  "
            f"{row['odor_id']:6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Receptor-to-receptor baseline sweep")
    parser.add_argument("--jitter", type=float, nargs="+", default=None,
                        help="Jitter levels as multiples of biological variability (default: 0.0 0.25 0.50 0.75 1.0 1.25 1.5)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Receiver variants per jitter level (default: 5)")
    parser.add_argument("--model", choices=list(MODEL_CLASSES), nargs="+",
                        default=list(MODEL_CLASSES),
                        help="Which model(s) to run (default: all)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print only, don't write results or plot")
    args = parser.parse_args()

    jitter_levels = args.jitter or [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    for model_name in args.model:
        results = run_receptor_to_receptor_sweep(
            jitter_levels=jitter_levels,
            seeds_per_level=args.seeds,
            model_name=model_name,
        )
        _print_results(results, model_name)

        if not args.no_save:
            json_path = write_results(
                experiment=f"receptor_to_receptor_sweep_{model_name}",
                description=f"Transfer quality vs brain difference using receptor-to-receptor baseline ({model_name})",
                parameters={
                    "model": model_name,
                    "jitter_levels": jitter_levels,
                    "seeds_per_level": args.seeds,
                },
                results=results,
                output_dir=RESULTS_DIR,
            )
            plot_path = plot_receptor_to_receptor_sweep(
                results, json_path.with_suffix(".png"),
            )
            print(f"\nResults saved to {json_path}")
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
