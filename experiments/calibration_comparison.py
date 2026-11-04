"""Compare full ridge calibration vs diagonal-only calibration.

Full ridge has N² parameters (cross-channel mixing).  Diagonal has 2N
parameters (per-channel scale + offset).  If diagonal performs nearly as
well, the cross-channel structure in the calibration is mostly cosmetic.
If it performs much worse, cross-channel remapping is genuinely needed.

Runs on both connectome and hand-built models to test whether the
difference depends on wiring realism.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from link.calibration import CalibrationTransform, DiagonalCalibration
from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_identification,
)

CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]


def _gather_signals(model, odor_profiles, receptor_names, intensities, *, receptor_fn="receptor_activity", odor_valence=None):
    signals = []
    odor_meta = []
    for odor_name in odor_profiles:
        otype = (odor_valence or {}).get(odor_name, "attractive")
        for intensity in intensities:
            act = getattr(model, receptor_fn)(intensity, otype)
            sig = {k: act.get(k, 0.0) for k in receptor_names}
            signals.append(sig)
            odor_meta.append((odor_name, intensity, otype))
    return signals, odor_meta


def _evaluate_transfer(
    transform,
    sender_signals,
    receiver,
    receiver_natural_signals,
    sender_behaviors,
    odor_names,
):
    transferred_states = []
    natural_states = []
    successes = 0

    for i, sender_sig in enumerate(sender_signals):
        cal_sig = transform.transform(sender_sig)
        result = receiver.simulate_from_receptors(cal_sig)
        transferred_states.append(result["neurons"])

        natural_result = receiver.simulate_from_receptors(receiver_natural_signals[i])
        natural_states.append(natural_result["neurons"])

        if result["behavior"] == sender_behaviors[i]:
            successes += 1

    n = len(sender_signals)
    return {
        "transfer_rate": successes / max(1, n),
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


def run_calibration_comparison(
    organism: str = "larva",
    model_type: str = "connectome",
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5]

    results = []

    if organism == "larva":
        from models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE, ORN_TYPES
        receptor_names = list(ORN_TYPES)

        if model_type == "connectome":
            from models.larva_connectome import ConnectomeLarvaModel
            from models.larva_connectome_variants import ConnectomeLarvaVariantGenerator
            sender = ConnectomeLarvaModel()
            gen_cls = ConnectomeLarvaVariantGenerator
        else:
            from models.drosophila_larva import DrosophilaLarvaModel
            from models.larva_variants import LarvaVariantGenerator
            sender = DrosophilaLarvaModel()
            gen_cls = LarvaVariantGenerator

    elif organism == "adult":
        from models.drosophila_adult import ODOR_PROFILES, ODOR_VALENCE
        if model_type == "connectome":
            from models.adult_connectome import ConnectomeAdultModel
            from models.adult_connectome_variants import ConnectomeAdultVariantGenerator
            sender = ConnectomeAdultModel()
            receptor_names = list(sender._receptor_names)
            gen_cls = ConnectomeAdultVariantGenerator
        else:
            from models.drosophila_adult import DrosophilaAdultModel
            from models.adult_variants import AdultVariantGenerator
            sender = DrosophilaAdultModel()
            receptor_names = list(sender._receptor_names) if hasattr(sender, '_receptor_names') else list(ODOR_PROFILES.keys())[:49]
            gen_cls = AdultVariantGenerator
    else:
        raise ValueError(f"Unknown organism: {organism}")

    for jitter in jitter_levels:
        gen = gen_cls(jitter_scale=jitter)
        ridge_metrics = []
        diag_metrics = []

        for seed in range(seeds_per_level):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue

            # Gather calibration data
            sender_cal, cal_meta = _gather_signals(
                sender, ODOR_PROFILES, receptor_names, CAL_INTENSITIES,
                odor_valence=ODOR_VALENCE,
            )
            receiver_cal, _ = _gather_signals(
                receiver, ODOR_PROFILES, receptor_names, CAL_INTENSITIES,
                odor_valence=ODOR_VALENCE,
            )

            # Gather holdout data
            sender_ho, ho_meta = _gather_signals(
                sender, ODOR_PROFILES, receptor_names, HOLDOUT_INTENSITIES,
                odor_valence=ODOR_VALENCE,
            )
            receiver_ho, _ = _gather_signals(
                receiver, ODOR_PROFILES, receptor_names, HOLDOUT_INTENSITIES,
                odor_valence=ODOR_VALENCE,
            )
            ho_odor_names = [m[0] for m in ho_meta]
            sender_behaviors = []
            for name, intensity, otype in ho_meta:
                r = sender.simulate(intensity, otype)
                sender_behaviors.append(r["behavior"])

            # Fit ridge
            ridge = CalibrationTransform()
            ridge.fit(sender_cal, receiver_cal)
            rm = _evaluate_transfer(ridge, sender_ho, receiver, receiver_ho, sender_behaviors, ho_odor_names)
            ridge_metrics.append(rm)

            # Fit diagonal
            diag = DiagonalCalibration()
            diag.fit(sender_cal, receiver_cal)
            dm = _evaluate_transfer(diag, sender_ho, receiver, receiver_ho, sender_behaviors, ho_odor_names)
            diag_metrics.append(dm)

        for method, metrics in [("ridge", ridge_metrics), ("diagonal", diag_metrics)]:
            if metrics:
                results.append({
                    "organism": organism,
                    "model": model_type,
                    "method": method,
                    "jitter_scale": jitter,
                    "n_params": f"{'N²' if method == 'ridge' else '2N'}",
                    "valid_seeds": len(metrics),
                    **{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
                })

    return results


if __name__ == "__main__":
    for org in ["larva", "adult"]:
        print(f"\n{'='*70}")
        print(f"  {org.upper()} (connectome): Ridge vs Diagonal Calibration")
        print(f"{'='*70}")
        results = run_calibration_comparison(org, "connectome", seeds_per_level=5)
        for r in results:
            print(
                f"  jitter={r['jitter_scale']:.2f}  {r['method']:8s} ({r['n_params']:>3s})  "
                f"transfer={r['transfer_rate']:.2f}  corr={r['correlation']:.3f}  "
                f"rsa={r['rsa']:.3f}  odor_id={r['odor_id']:.3f}"
            )
