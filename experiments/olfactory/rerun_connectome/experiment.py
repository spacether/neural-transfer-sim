"""Rerun all main experiments on connectome models with bootstrap CIs.

Uses the public API shared by both hand-built and connectome models:
  - receptor_activity(intensity, odor_type)
  - simulate(intensity, odor_type)
  - simulate_from_receptors(signal)

Outputs updated numbers for all paper tables.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from neural_transfer.analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
    bootstrap_ci,
)


# ------------------------------------------------------------------ #
# Generic experiment runners using public API
# ------------------------------------------------------------------ #

def _broadcast_one_pair(sender, receiver, odor_profiles, odor_valence, receptor_names, intensities=(0.5, 1.0, 1.5)):
    odor_names = []
    natural_states = []
    transferred_states = []
    successes = 0
    total = 0

    for odor_name in odor_profiles:
        otype = odor_valence.get(odor_name, "attractive")
        for intensity in intensities:
            sender_act = sender.receptor_activity(intensity, otype)
            sender_beh = sender.simulate(intensity, otype)["behavior"]

            signal = {k: sender_act.get(k, 0.0) for k in receptor_names}
            result = receiver.simulate_from_receptors(signal)
            natural = receiver.simulate(intensity, otype)

            odor_names.append(odor_name)
            natural_states.append(natural["neurons"])
            transferred_states.append(result["neurons"])
            if result["behavior"] == sender_beh:
                successes += 1
            total += 1

    if total == 0:
        return {k: 0.0 for k in ("transfer_rate", "correlation", "mse", "rsa", "odor_id")}
    return {
        "transfer_rate": successes / total,
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


def _mask_signal(signal, active_keys):
    return {k: (v if k in active_keys else 0.0) for k, v in signal.items()}


def _channel_sweep_one(sender, receiver, odor_profiles, odor_valence, receptor_names, active_set):
    odor_names = []
    natural_states = []
    transferred_states = []
    successes = 0
    total = 0

    for odor_name in odor_profiles:
        otype = odor_valence.get(odor_name, "attractive")
        sender_act = sender.receptor_activity(1.0, otype)
        sender_beh = sender.simulate(1.0, otype)["behavior"]

        signal = {k: sender_act.get(k, 0.0) for k in receptor_names}
        masked = _mask_signal(signal, active_set)
        result = receiver.simulate_from_receptors(masked)
        natural = receiver.simulate(1.0, otype)

        odor_names.append(odor_name)
        natural_states.append(natural["neurons"])
        transferred_states.append(result["neurons"])
        if result["behavior"] == sender_beh:
            successes += 1
        total += 1

    if total == 0:
        return {k: 0.0 for k in ("transfer_rate", "correlation", "mse", "rsa", "odor_id")}
    return {
        "transfer_rate": successes / total,
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


def _hw_degrade_one(sender, receiver, odor_profiles, odor_valence, receptor_names, hw_chain, rng):
    odor_names = []
    natural_states = []
    transferred_states = []
    successes = 0
    total = 0

    for odor_name in odor_profiles:
        otype = odor_valence.get(odor_name, "attractive")
        for intensity in (0.5, 1.0, 1.5):
            sender_act = sender.receptor_activity(intensity, otype)
            sender_beh = sender.simulate(intensity, otype)["behavior"]

            signal = {k: sender_act.get(k, 0.0) for k in receptor_names}
            degraded = hw_chain.apply(signal, rng)
            result = receiver.simulate_from_receptors(degraded)
            natural = receiver.simulate(intensity, otype)

            odor_names.append(odor_name)
            natural_states.append(natural["neurons"])
            transferred_states.append(result["neurons"])
            if result["behavior"] == sender_beh:
                successes += 1
            total += 1

    if total == 0:
        return {k: 0.0 for k in ("transfer_rate", "correlation", "mse", "rsa", "odor_id")}
    return {
        "transfer_rate": successes / total,
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "odor_id")


def _agg_with_ci(seed_metrics):
    row = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean, lo, hi = bootstrap_ci(vals)
        row[k] = mean
        row[f"{k}_ci_lo"] = lo
        row[f"{k}_ci_hi"] = hi
    return row


# ------------------------------------------------------------------ #
# Per-organism config
# ------------------------------------------------------------------ #

def _get_config(organism):
    if organism == "larva":
        from neural_transfer.models.drosophila_larva import ConnectomeLarvaModel
        from neural_transfer.models.drosophila_larva import ConnectomeLarvaVariantGenerator
        from neural_transfer.models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE, ORN_TYPES
        sender = ConnectomeLarvaModel()
        return {
            "sender": sender,
            "gen_cls": ConnectomeLarvaVariantGenerator,
            "odor_profiles": ODOR_PROFILES,
            "odor_valence": ODOR_VALENCE,
            "receptor_names": list(ORN_TYPES),
            "channel_counts": [1, 2, 3, 5, 7, 10, 14, 18, 21],
        }
    elif organism == "adult":
        from neural_transfer.models.drosophila_adult import ConnectomeAdultModel
        from neural_transfer.models.drosophila_adult import ConnectomeAdultVariantGenerator
        from neural_transfer.models.drosophila_adult import ODOR_PROFILES, ODOR_VALENCE
        sender = ConnectomeAdultModel()
        return {
            "sender": sender,
            "gen_cls": ConnectomeAdultVariantGenerator,
            "odor_profiles": ODOR_PROFILES,
            "odor_valence": ODOR_VALENCE,
            "receptor_names": list(sender._receptor_names),
            "channel_counts": [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49],
        }
    raise ValueError(organism)


# ------------------------------------------------------------------ #
# Experiment 1: Broadcast sweep
# ------------------------------------------------------------------ #

def run_broadcast(organism, jitter_levels=None, n_seeds=10):
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]
    cfg = _get_config(organism)
    results = []

    for jitter in jitter_levels:
        gen = cfg["gen_cls"](jitter_scale=jitter)
        seed_metrics = []
        for seed in range(n_seeds):
            receiver = gen.generate_variant(cfg["sender"], seed=seed)
            if not gen.sanity_check(receiver):
                continue
            m = _broadcast_one_pair(
                cfg["sender"], receiver,
                cfg["odor_profiles"], cfg["odor_valence"], cfg["receptor_names"],
            )
            seed_metrics.append(m)

        if seed_metrics:
            row = {"organism": organism, "jitter_scale": jitter, "n_seeds": len(seed_metrics)}
            row.update(_agg_with_ci(seed_metrics))
            results.append(row)
    return results


# ------------------------------------------------------------------ #
# Experiment 2: Channel sweep
# ------------------------------------------------------------------ #

def run_channel_sweep(organism, jitter_levels=None, n_seeds=5, channel_seeds=3):
    if jitter_levels is None:
        jitter_levels = [0.0, 0.5, 1.0]
    cfg = _get_config(organism)
    rnames = cfg["receptor_names"]
    results = []

    for n_ch in cfg["channel_counts"]:
        for jitter in jitter_levels:
            gen = cfg["gen_cls"](jitter_scale=jitter)
            all_metrics = []

            for ch_seed in range(channel_seeds):
                ch_rng = np.random.default_rng(ch_seed)
                active = set(ch_rng.choice(rnames, size=min(n_ch, len(rnames)), replace=False))

                for seed in range(n_seeds):
                    receiver = gen.generate_variant(cfg["sender"], seed=seed)
                    if not gen.sanity_check(receiver):
                        continue
                    m = _channel_sweep_one(
                        cfg["sender"], receiver,
                        cfg["odor_profiles"], cfg["odor_valence"], rnames, active,
                    )
                    all_metrics.append(m)

            if all_metrics:
                row = {"organism": organism, "n_channels": n_ch, "jitter_scale": jitter,
                       "n_trials": len(all_metrics)}
                row.update(_agg_with_ci(all_metrics))
                results.append(row)
    return results


# ------------------------------------------------------------------ #
# Experiment 3: Hardware boundary
# ------------------------------------------------------------------ #

def run_hardware_boundary(organism, jitter=0.75, n_seeds=10):
    from neural_transfer.hardware.noise import ReadNoise, WritePrecision, ChannelDropout, TransmitDelay, HardwareChain
    cfg = _get_config(organism)
    gen = cfg["gen_cls"](jitter_scale=jitter)
    results = {}

    def _sweep(make_chain, values, param_name):
        rows = []
        for val in values:
            chain = make_chain(val)
            seed_metrics = []
            for seed in range(n_seeds):
                receiver = gen.generate_variant(cfg["sender"], seed=seed)
                if not gen.sanity_check(receiver):
                    continue
                rng = np.random.default_rng(seed * 1000 + hash(str(val)) % 10000)
                m = _hw_degrade_one(
                    cfg["sender"], receiver,
                    cfg["odor_profiles"], cfg["odor_valence"], cfg["receptor_names"],
                    chain, rng,
                )
                seed_metrics.append(m)
            if seed_metrics:
                row = {param_name: val, "n_seeds": len(seed_metrics)}
                row.update(_agg_with_ci(seed_metrics))
                rows.append(row)
        return rows

    results["read_noise"] = _sweep(
        lambda s: HardwareChain(ReadNoise(sigma=s), WritePrecision(), ChannelDropout()),
        [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0], "noise_sigma")
    results["write_precision"] = _sweep(
        lambda b: HardwareChain(ReadNoise(), WritePrecision(bits=b, clip_max=100.0), ChannelDropout()),
        [float("inf"), 8, 6, 5, 4, 3, 2, 1], "precision_bits")
    results["channel_dropout"] = _sweep(
        lambda f: HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout(drop_fraction=f)),
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], "dropout_fraction")
    results["transmit_delay"] = _sweep(
        lambda ms: HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout(), TransmitDelay(delay_ms=ms, tau_ms=500.0)),
        [0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0], "delay_ms")

    return results
