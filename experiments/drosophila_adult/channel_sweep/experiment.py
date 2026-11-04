"""Channel sweep experiment for the adult Drosophila stage.

Measures transfer quality versus number of connected ORN types at each
level of brain difference.  At each channel count, the experiment
randomly selects which ORN types to include and zeros out the rest.

With 50 ORN types (vs 21 in larva), this tests how the channel
requirement scales with a larger receptor repertoire.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.drosophila_adult import (
    DrosophilaAdultModel,
    ODOR_PROFILES,
    ODOR_VALENCE,
    ORN_TYPES,
)
from models.adult_variants import AdultVariantGenerator
from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
)

ODOR_SET: List[Dict[str, object]] = [
    {"odor_name": name, "odor_intensity": 1.0}
    for name in ODOR_PROFILES
]


def _mask_signal(signal: Dict[str, float], active_orns: set) -> Dict[str, float]:
    return {k: (v if k in active_orns else 0.0) for k, v in signal.items()}


def run_channel_sweep(
    channel_counts: List[int] | None = None,
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
    channel_seeds: int = 3,
) -> List[Dict]:
    if channel_counts is None:
        channel_counts = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]
    if jitter_levels is None:
        jitter_levels = [0.0, 0.5, 1.0]

    sender = DrosophilaAdultModel()
    results: List[Dict] = []

    for n_channels in channel_counts:
        for jitter in jitter_levels:
            gen = AdultVariantGenerator(jitter_scale=jitter)
            all_metrics: List[Dict[str, float]] = []

            for ch_seed in range(channel_seeds):
                ch_rng = np.random.default_rng(ch_seed)
                active = set(ch_rng.choice(ORN_TYPES, size=min(n_channels, len(ORN_TYPES)), replace=False))

                for seed in range(seeds_per_level):
                    receiver = gen.generate_variant(sender, seed=seed)
                    if not gen.sanity_check(receiver):
                        continue

                    odor_types: List[str] = []
                    odor_names: List[str] = []
                    natural_states: List[Dict[str, float]] = []
                    transferred_states: List[Dict[str, float]] = []
                    successes = 0

                    for odor in ODOR_SET:
                        name = odor["odor_name"]
                        otype = ODOR_VALENCE[name]
                        odor_types.append(otype)
                        odor_names.append(name)

                        sender_orn = sender._orn_type_rates(name, 1.0)
                        full_signal = {orn: float(sender_orn[i]) for i, orn in enumerate(ORN_TYPES)}
                        masked = _mask_signal(full_signal, active)

                        sender_layers = sender._forward(name, 1.0)
                        sender_beh = sender._behavior_from_mbon(sender_layers["mbon"])

                        result = receiver.simulate_from_receptors(masked)
                        if result["behavior"] == sender_beh:
                            successes += 1

                        natural_layers = receiver._forward(name, 1.0)
                        natural_states.append(receiver._layers_to_dict(natural_layers))
                        transferred_states.append(result["neurons"])

                    natural_kc = [{k: v for k, v in s.items() if k.startswith("KC_")} for s in natural_states]
                    transferred_kc = [{k: v for k, v in s.items() if k.startswith("KC_")} for s in transferred_states]
                    all_metrics.append({
                        "transfer_rate": successes / len(ODOR_SET),
                        "correlation": correlation(transferred_states, natural_states),
                        "mse": mean_mse(transferred_states, natural_states),
                        "rsa": representational_similarity(natural_states, transferred_states),
                        "odor_id": odor_identification(odor_names, natural_kc, transferred_kc),
                    })

            if not all_metrics:
                continue

            results.append({
                "n_channels": n_channels,
                "jitter_scale": jitter,
                "valid_trials": len(all_metrics),
                **{
                    key: float(np.mean([m[key] for m in all_metrics]))
                    for key in ("transfer_rate", "correlation", "mse", "rsa", "odor_id")
                },
            })

    return results
