"""Calibration experiment for the worm stage.

For each brain-difference level, fits a linear calibration transform on shared
odors, then compares calibrated transfer against the receptor-to-receptor
baseline on held-out odors using correlation, MSE, RSA, and odor identification
plus categorical behavioral match.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
from neural_transfer.models.c_elegans.cook2019 import Cook2019WormModel
from neural_transfer.models.c_elegans.variants import WormVariantGenerator
from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.link.receptor_to_receptor import ReceptorToReceptorBaseline
from neural_transfer.analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
)

MODEL_CLASSES = {
    "hand_built": HandBuiltWormModel,
    "cook2019": Cook2019WormModel,
}


CALIBRATION_ODORS: List[Dict[str, object]] = [
    {"odor_intensity": i, "odor_type": t}
    for t in ("attractive", "repellent")
    for i in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
]

HOLDOUT_ODORS: List[Dict[str, object]] = [
    {"odor_intensity": i, "odor_type": t}
    for t in ("attractive", "repellent")
    for i in (0.4, 0.9, 1.25, 1.75)
]

METRIC_KEYS = ("transfer_rate", "correlation", "mse", "rsa", "odor_id")


def _evaluate_method(
    receiver: object,
    sender_behavior: str,
    injected_signal: Dict[str, float],
    odor_type: str,
) -> Dict:
    """Run one trial for a single transfer method and return state + success."""
    result = receiver.simulate_from_receptors(injected_signal, odor_type)
    return {
        "state": result["neurons"],
        "success": result["behavior"] == sender_behavior,
    }


def _run_one_pair(
    sender: object,
    receiver: object,
    cal_odors: List[Dict[str, object]],
    test_odors: List[Dict[str, object]],
) -> Dict:
    """Fit calibration on cal_odors, evaluate both methods on test_odors."""
    sender_cal = [sender.odor_response(o["odor_intensity"], o["odor_type"]) for o in cal_odors]
    receiver_cal = [receiver.odor_response(o["odor_intensity"], o["odor_type"]) for o in cal_odors]

    transform = CalibrationTransform()
    transform.fit(sender_cal, receiver_cal)

    baseline = ReceptorToReceptorBaseline(sender=sender, receiver=receiver)

    odor_types: List[str] = []
    natural_states: List[Dict[str, float]] = []
    direct_states: List[Dict[str, float]] = []
    cal_states: List[Dict[str, float]] = []
    direct_ok, cal_ok = 0, 0

    for odor in test_odors:
        intensity, otype = odor["odor_intensity"], odor["odor_type"]
        odor_types.append(otype)
        sender_sig = sender.odor_response(intensity, otype)
        sender_behavior = sender.behavior_from_signal(sender_sig, otype)
        natural_states.append(receiver.network_state(intensity, otype))

        direct_sig = baseline.copy_signal(sender_sig)
        d = _evaluate_method(receiver, sender_behavior, direct_sig, otype)
        direct_states.append(d["state"])
        direct_ok += d["success"]

        cal_sig = transform.transform(sender_sig)
        c = _evaluate_method(receiver, sender_behavior, cal_sig, otype)
        cal_states.append(c["state"])
        cal_ok += c["success"]

    n = len(test_odors)
    out: Dict = {"n_holdout": n, "n_calibration": len(cal_odors)}
    for prefix, states, ok in [("direct", direct_states, direct_ok), ("calibrated", cal_states, cal_ok)]:
        out[f"{prefix}_transfer_rate"] = ok / max(1, n)
        out[f"{prefix}_correlation"] = correlation(states, natural_states)
        out[f"{prefix}_mse"] = mean_mse(states, natural_states)
        out[f"{prefix}_rsa"] = representational_similarity(natural_states, states)
        out[f"{prefix}_odor_id"] = odor_identification(odor_types, natural_states, states)
    return out


def run_calibration_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
    cal_odors: List[Dict[str, object]] | None = None,
    holdout_odors: List[Dict[str, object]] | None = None,
    model_name: str = "hand_built",
) -> List[Dict]:
    """Compare calibrated vs. receptor-to-receptor transfer across brain-difference levels."""
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]
    if cal_odors is None:
        cal_odors = CALIBRATION_ODORS
    if holdout_odors is None:
        holdout_odors = HOLDOUT_ODORS

    ModelClass = MODEL_CLASSES[model_name]
    sender = ModelClass()
    generator = WormVariantGenerator()
    results: List[Dict] = []

    for jitter in jitter_levels:
        generator.jitter_scale = jitter
        pairs: List[Dict] = []

        for seed in range(seeds_per_level):
            receiver = generator.generate_model_variant(sender, seed=seed)
            pairs.append(_run_one_pair(sender, receiver, cal_odors, holdout_odors))

        row: Dict = {"model": model_name, "jitter_scale": jitter, "seeds": seeds_per_level}
        for prefix in ("direct", "calibrated"):
            for metric in METRIC_KEYS:
                key = f"{prefix}_{metric}"
                row[key] = float(np.mean([p[key] for p in pairs]))
        results.append(row)

    return results
