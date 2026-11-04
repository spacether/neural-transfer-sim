"""Calibration experiment for the adult Drosophila stage.

Fits a linear calibration transform on shared odors, then compares
calibrated transfer against direct broadcast on held-out odors.
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
from link.calibration import CalibrationTransform
from link.receptor_to_receptor import ReceptorToReceptorBaseline
from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_type_decoding,
    odor_identification,
)

ALL_ODOR_NAMES = list(ODOR_PROFILES.keys())

CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_ODOR_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_ODOR_NAMES for i in HOLDOUT_INTENSITIES]

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "decoding", "odor_id")


def _get_orn_signal(model: DrosophilaAdultModel, odor_name: str, intensity: float) -> Dict[str, float]:
    rates = model._orn_type_rates(odor_name, intensity)
    return {orn: float(rates[i]) for i, orn in enumerate(ORN_TYPES)}


def _run_one_pair(
    sender: DrosophilaAdultModel,
    receiver: DrosophilaAdultModel,
) -> Dict:
    sender_cal = [_get_orn_signal(sender, name, intensity) for name, intensity in CAL_SET]
    receiver_cal = [_get_orn_signal(receiver, name, intensity) for name, intensity in CAL_SET]

    transform = CalibrationTransform()
    transform.fit(sender_cal, receiver_cal)

    baseline = ReceptorToReceptorBaseline(
        sender=sender, receiver=receiver,
        source_neurons=tuple(ORN_TYPES),
    )

    odor_types: List[str] = []
    odor_names: List[str] = []
    natural_states: List[Dict[str, float]] = []
    direct_states: List[Dict[str, float]] = []
    cal_states: List[Dict[str, float]] = []
    direct_ok, cal_ok = 0, 0

    for name, intensity in HOLDOUT_SET:
        otype = ODOR_VALENCE[name]
        odor_types.append(otype)
        odor_names.append(name)

        sender_sig = _get_orn_signal(sender, name, intensity)
        sender_layers = sender._forward(name, intensity)
        sender_beh = sender._behavior_from_mbon(sender_layers["mbon"])

        natural_layers = receiver._forward(name, intensity)
        natural_states.append(receiver._layers_to_dict(natural_layers))

        direct_sig = baseline.copy_signal(sender_sig)
        d_result = receiver.simulate_from_receptors(direct_sig)
        direct_states.append(d_result["neurons"])
        direct_ok += d_result["behavior"] == sender_beh

        cal_sig = transform.transform(sender_sig)
        c_result = receiver.simulate_from_receptors(cal_sig)
        cal_states.append(c_result["neurons"])
        cal_ok += c_result["behavior"] == sender_beh

    n = len(HOLDOUT_SET)
    natural_kc = [{k: v for k, v in s.items() if k.startswith("KC_")} for s in natural_states]
    out: Dict = {"n_holdout": n, "n_calibration": len(CAL_SET)}
    for prefix, states, ok in [("direct", direct_states, direct_ok), ("calibrated", cal_states, cal_ok)]:
        states_kc = [{k: v for k, v in s.items() if k.startswith("KC_")} for s in states]
        out[f"{prefix}_transfer_rate"] = ok / max(1, n)
        out[f"{prefix}_correlation"] = correlation(states, natural_states)
        out[f"{prefix}_mse"] = mean_mse(states, natural_states)
        out[f"{prefix}_rsa"] = representational_similarity(natural_states, states)
        out[f"{prefix}_decoding"] = odor_type_decoding(odor_types, natural_kc, states_kc)
        out[f"{prefix}_odor_id"] = odor_identification(odor_names, natural_kc, states_kc)
    return out


def run_calibration_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    sender = DrosophilaAdultModel()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = AdultVariantGenerator(jitter_scale=jitter)
        pairs: List[Dict] = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            pairs.append(_run_one_pair(sender, receiver))

        if not pairs:
            continue

        row: Dict = {"jitter_scale": jitter, "valid_seeds": len(pairs)}
        for prefix in ("direct", "calibrated"):
            for metric in METRIC_KEYS:
                key = f"{prefix}_{metric}"
                row[key] = float(np.mean([p[key] for p in pairs]))
        results.append(row)

    return results
