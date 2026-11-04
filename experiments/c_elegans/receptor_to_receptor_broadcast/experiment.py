"""Receptor-to-receptor baseline experiment for the worm stage.

Measures transfer quality when the sender's receptor activity is copied
directly to the receiver's receptor layer with no learned translation,
across a range of brain-difference levels.

Reports: correlation (Pearson r), MSE, RSA, odor-type decoding accuracy,
and categorical behavioral match — each comparing the receiver's post-injection
state against its natural response to the same odor.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.worm_connectome import HandBuiltWormModel
from models.cook2019_connectome import Cook2019WormModel
from models.worm_variants import WormVariantGenerator
from link.receptor_to_receptor import ReceptorToReceptorBaseline
from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_type_decoding,
)

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
                "decoding": odor_type_decoding(odor_types, natural_states, transferred_states),
            })

        results.append({
            "model": model_name,
            "jitter_scale": jitter,
            **{
                key: float(np.mean([m[key] for m in seed_metrics]))
                for key in ("transfer_rate", "correlation", "mse", "rsa", "decoding")
            },
            "seeds": seeds_per_level,
        })

    return results
