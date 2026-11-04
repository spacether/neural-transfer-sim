"""Direct broadcast baseline for the Drosophila larva stage.

Copies sender ORN activity to receiver ORN layer with no learned
translation, across a range of brain-difference levels.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.drosophila_larva import (
    DrosophilaLarvaModel,
    ODOR_PROFILES,
    ODOR_VALENCE,
    ORN_TYPES,
)
from models.larva_variants import LarvaVariantGenerator
from link.receptor_to_receptor import ReceptorToReceptorBaseline
from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_type_decoding,
    odor_identification,
)


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

    sender = DrosophilaLarvaModel()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = LarvaVariantGenerator(jitter_scale=jitter)
        seed_metrics: List[Dict[str, float]] = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue

            baseline = ReceptorToReceptorBaseline(
                sender=sender,
                receiver=receiver,
                source_neurons=tuple(ORN_TYPES),
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

                sender_orn = sender._orn_rates(name, intensity)
                sender_signal = {orn: float(sender_orn[i]) for i, orn in enumerate(ORN_TYPES)}
                sender_layers = sender._forward(name, intensity)
                sender_beh = sender._behavior_from_mbon(sender_layers["mbon"])

                receiver_signal = baseline.copy_signal(sender_signal)
                result = receiver.simulate_from_receptors(receiver_signal)
                if result["behavior"] == sender_beh:
                    successes += 1

                natural_layers = receiver._forward(name, intensity)
                natural_state = receiver._layers_to_dict(natural_layers)
                natural_states.append(natural_state)
                transferred_states.append(result["neurons"])

            if len(ODOR_SET) > 0:
                seed_metrics.append({
                    "transfer_rate": successes / len(ODOR_SET),
                    "correlation": correlation(transferred_states, natural_states),
                    "mse": mean_mse(transferred_states, natural_states),
                    "rsa": representational_similarity(natural_states, transferred_states),
                    "decoding": odor_type_decoding(odor_types, natural_states, transferred_states),
                    "odor_id": odor_identification(odor_names, natural_states, transferred_states),
                })

        if not seed_metrics:
            continue

        results.append({
            "jitter_scale": jitter,
            "valid_seeds": len(seed_metrics),
            **{
                key: float(np.mean([m[key] for m in seed_metrics]))
                for key in ("transfer_rate", "correlation", "mse", "rsa", "decoding", "odor_id")
            },
        })

    return results
