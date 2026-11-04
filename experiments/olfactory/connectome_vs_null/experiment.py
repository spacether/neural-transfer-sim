"""Compare transfer quality: real connectome models vs shuffled-connectome nulls.

For each organism (larva, adult), runs the broadcast baseline on both the
connectome model and a shuffled-connectome null matched in dimensions and
weight statistics.  The difference measures how much real wiring topology
contributes to transfer quality, beyond what dimensionality alone provides.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
)


def _run_broadcast(
    sender,
    receiver,
    odor_profiles: Dict,
    odor_valence: Dict,
    receptor_names: List[str],
    *,
    simulate_fn_name: str = "simulate_from_receptors",
    receptor_fn_name: str = "receptor_activity",
) -> Dict[str, float]:
    """Run a single broadcast (no-translation) transfer between sender and receiver."""
    odor_names = []
    natural_states = []
    transferred_states = []
    successes = 0
    total = 0

    for odor_name in odor_profiles:
        for intensity in (0.5, 1.0, 1.5):
            sender_activity = getattr(sender, receptor_fn_name)(intensity, odor_valence.get(odor_name, "attractive"))
            sender_result = sender.simulate(intensity, odor_valence.get(odor_name, "attractive"))
            sender_beh = sender_result["behavior"]

            receiver_signal = {k: sender_activity.get(k, 0.0) for k in receptor_names}
            receiver_result = getattr(receiver, simulate_fn_name)(receiver_signal)
            receiver_beh = receiver_result["behavior"]

            natural_result = receiver.simulate(intensity, odor_valence.get(odor_name, "attractive"))

            odor_names.append(odor_name)
            natural_states.append(natural_result["neurons"])
            transferred_states.append(receiver_result["neurons"])

            if receiver_beh == sender_beh:
                successes += 1
            total += 1

    if total == 0:
        return {"transfer_rate": 0.0, "correlation": 0.0, "mse": 0.0, "rsa": 0.0, "odor_id": 0.0}

    return {
        "transfer_rate": successes / total,
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


def _run_null_broadcast(
    sender_null,
    receiver_null,
    odor_profiles: Dict,
    odor_valence: Dict,
) -> Dict[str, float]:
    """Run broadcast on null models (different API: simulate takes odor_name, intensity)."""
    odor_names = []
    natural_states = []
    transferred_states = []
    successes = 0
    total = 0

    for odor_name in odor_profiles:
        for intensity in (0.5, 1.0, 1.5):
            sender_result = sender_null.simulate(odor_name, intensity)
            sender_beh = sender_result["behavior"]
            sender_inp = sender_null.input_response(odor_name, intensity)

            receiver_result = receiver_null.simulate_from_input(sender_inp)
            receiver_beh = receiver_result["behavior"]

            natural_result = receiver_null.simulate(odor_name, intensity)

            odor_names.append(odor_name)
            natural_states.append(natural_result["neurons"])
            transferred_states.append(receiver_result["neurons"])

            if receiver_beh == sender_beh:
                successes += 1
            total += 1

    if total == 0:
        return {"transfer_rate": 0.0, "correlation": 0.0, "mse": 0.0, "rsa": 0.0, "odor_id": 0.0}

    return {
        "transfer_rate": successes / total,
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


def run_comparison(
    organism: str = "larva",
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    """Run broadcast sweep on both connectome and shuffled-null models."""
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    from models.null_model import build_shuffled_connectome, build_null_variant

    results = []

    if organism == "larva":
        from models.larva_connectome import ConnectomeLarvaModel
        from models.larva_connectome_variants import ConnectomeLarvaVariantGenerator
        from models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE, ORN_TYPES

        sender_conn = ConnectomeLarvaModel()
        sender_null = build_shuffled_connectome("larva")
        receptor_names = list(ORN_TYPES)

        for jitter in jitter_levels:
            conn_gen = ConnectomeLarvaVariantGenerator(jitter_scale=jitter)
            conn_metrics = []
            null_metrics = []

            for seed in range(seeds_per_level):
                # Connectome receiver
                receiver_conn = conn_gen.generate_variant(sender_conn, seed=seed)
                if not conn_gen.sanity_check(receiver_conn):
                    continue
                m = _run_broadcast(
                    sender_conn, receiver_conn,
                    ODOR_PROFILES, ODOR_VALENCE, receptor_names,
                )
                conn_metrics.append(m)

                # Null receiver
                receiver_null = build_null_variant(sender_null, jitter_scale=jitter, seed=seed)
                nm = _run_null_broadcast(
                    sender_null, receiver_null, ODOR_PROFILES, ODOR_VALENCE,
                )
                null_metrics.append(nm)

            if conn_metrics:
                results.append({
                    "organism": organism,
                    "jitter_scale": jitter,
                    "model": "connectome",
                    "valid_seeds": len(conn_metrics),
                    **{k: float(np.mean([m[k] for m in conn_metrics]))
                       for k in conn_metrics[0]},
                })
            if null_metrics:
                results.append({
                    "organism": organism,
                    "jitter_scale": jitter,
                    "model": "shuffled_null",
                    "valid_seeds": len(null_metrics),
                    **{k: float(np.mean([m[k] for m in null_metrics]))
                       for k in null_metrics[0]},
                })

    elif organism == "adult":
        from models.adult_connectome import ConnectomeAdultModel
        from models.adult_connectome_variants import ConnectomeAdultVariantGenerator
        from models.drosophila_adult import ODOR_PROFILES, ODOR_VALENCE

        sender_conn = ConnectomeAdultModel()
        sender_null = build_shuffled_connectome("adult")
        receptor_names = list(sender_conn._receptor_names)

        for jitter in jitter_levels:
            conn_gen = ConnectomeAdultVariantGenerator(jitter_scale=jitter)
            conn_metrics = []
            null_metrics = []

            for seed in range(seeds_per_level):
                receiver_conn = conn_gen.generate_variant(sender_conn, seed=seed)
                if not conn_gen.sanity_check(receiver_conn):
                    continue
                m = _run_broadcast(
                    sender_conn, receiver_conn,
                    ODOR_PROFILES, ODOR_VALENCE, receptor_names,
                )
                conn_metrics.append(m)

                receiver_null = build_null_variant(sender_null, jitter_scale=jitter, seed=seed)
                nm = _run_null_broadcast(
                    sender_null, receiver_null, ODOR_PROFILES, ODOR_VALENCE,
                )
                null_metrics.append(nm)

            if conn_metrics:
                results.append({
                    "organism": organism,
                    "jitter_scale": jitter,
                    "model": "connectome",
                    "valid_seeds": len(conn_metrics),
                    **{k: float(np.mean([m[k] for m in conn_metrics]))
                       for k in conn_metrics[0]},
                })
            if null_metrics:
                results.append({
                    "organism": organism,
                    "jitter_scale": jitter,
                    "model": "shuffled_null",
                    "valid_seeds": len(null_metrics),
                    **{k: float(np.mean([m[k] for m in null_metrics]))
                       for k in null_metrics[0]},
                })

    return results
