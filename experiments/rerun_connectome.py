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
import json
import sys

from analysis.metrics import (
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
        from models.larva_connectome import ConnectomeLarvaModel
        from models.larva_connectome_variants import ConnectomeLarvaVariantGenerator
        from models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE, ORN_TYPES
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
        from models.adult_connectome import ConnectomeAdultModel
        from models.adult_connectome_variants import ConnectomeAdultVariantGenerator
        from models.drosophila_adult import ODOR_PROFILES, ODOR_VALENCE
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
    from hardware.noise import ReadNoise, WritePrecision, ChannelDropout, TransmitDelay, HardwareChain
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


# ------------------------------------------------------------------ #
# Main: run everything, print tables
# ------------------------------------------------------------------ #

def fmt_ci(row, key):
    v = row[key]
    lo = row.get(f"{key}_ci_lo", v)
    hi = row.get(f"{key}_ci_hi", v)
    if abs(hi - lo) < 0.005:
        return f"{v:.2f}"
    return f"{v:.2f} [{lo:.2f},{hi:.2f}]"


if __name__ == "__main__":
    organisms = sys.argv[1:] if len(sys.argv) > 1 else ["larva", "adult"]

    all_results = {}

    for org in organisms:
        print(f"\n{'='*70}")
        print(f"  {org.upper()} CONNECTOME — BROADCAST SWEEP")
        print(f"{'='*70}")
        bcast = run_broadcast(org, n_seeds=10)
        all_results[f"{org}_broadcast"] = bcast
        for r in bcast:
            print(f"  jitter={r['jitter_scale']:.2f}  transfer={fmt_ci(r,'transfer_rate')}  "
                  f"corr={fmt_ci(r,'correlation')}  rsa={fmt_ci(r,'rsa')}  "
                  f"odor_id={fmt_ci(r,'odor_id')}")

        print(f"\n{'='*70}")
        print(f"  {org.upper()} CONNECTOME — CHANNEL SWEEP")
        print(f"{'='*70}")
        chsw = run_channel_sweep(org, n_seeds=5, channel_seeds=3)
        all_results[f"{org}_channel_sweep"] = chsw
        cur_jitter = None
        for r in chsw:
            if r["jitter_scale"] != cur_jitter:
                cur_jitter = r["jitter_scale"]
                print(f"\n  --- jitter = {cur_jitter} ---")
            print(f"    {r['n_channels']:3d} ch  transfer={fmt_ci(r,'transfer_rate')}  "
                  f"odor_id={fmt_ci(r,'odor_id')}")

        print(f"\n{'='*70}")
        print(f"  {org.upper()} CONNECTOME — HARDWARE BOUNDARY")
        print(f"{'='*70}")
        hw = run_hardware_boundary(org, n_seeds=10)
        all_results[f"{org}_hardware"] = hw
        for axis_name, rows in hw.items():
            print(f"\n  --- {axis_name} ---")
            param_key = [k for k in rows[0] if k not in METRIC_KEYS and not k.endswith("_ci_lo") and not k.endswith("_ci_hi") and k != "n_seeds"][0]
            for r in rows:
                print(f"    {r[param_key]:>8}  transfer={fmt_ci(r,'transfer_rate')}  "
                      f"corr={fmt_ci(r,'correlation')}")

    # Save raw results
    out_path = "experiments/connectome_results.json"
    # Convert for JSON serialization
    def _clean(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, list):
            serializable[k] = [{kk: _clean(vv) for kk, vv in row.items()} for row in v]
        elif isinstance(v, dict):
            serializable[k] = {kk: [{kkk: _clean(vvv) for kkk, vvv in row.items()} for row in vv]
                                for kk, vv in v.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
