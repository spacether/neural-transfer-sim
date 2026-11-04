"""Sensitivity analysis for free parameters in connectome models.

Tests how transfer quality changes when each free parameter is perturbed
from its default.  Reports which parameters matter most and which are
safely ignorable, to establish that headline results are robust.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from analysis.metrics import (
    correlation,
    representational_similarity,
    odor_identification,
)


def _broadcast_metrics(sender, receiver, odor_profiles, odor_valence, receptor_names):
    """Compute broadcast transfer metrics at a single parameter setting."""
    odor_names = []
    natural_states = []
    transferred_states = []
    successes = 0
    total = 0

    for odor_name in odor_profiles:
        for intensity in (0.5, 1.0, 1.5):
            otype = odor_valence.get(odor_name, "attractive")
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
        return {"transfer_rate": 0.0, "correlation": 0.0, "rsa": 0.0, "odor_id": 0.0}

    return {
        "transfer_rate": successes / total,
        "correlation": correlation(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_names, natural_states, transferred_states),
    }


def run_sensitivity(
    organism: str = "larva",
    jitter_scale: float = 0.5,
    n_seeds: int = 3,
) -> List[Dict]:
    """Sweep each free parameter while holding others at default.

    Returns one row per parameter-value combination, with mean metrics
    over n_seeds receiver variants.
    """
    param_ranges = {
        "pn_baseline": [1.0, 3.0, 5.0, 7.0, 10.0],
        "lateral_inhibition": [0.0, 0.05, 0.10, 0.15, 0.25, 0.40],
        "kc_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
        "approach_threshold": [0.01, 0.03, 0.05, 0.10, 0.20],
    }

    defaults = {
        "pn_baseline": 5.0,
        "lateral_inhibition": 0.15,
        "kc_threshold": 0.3,
        "approach_threshold": 0.05,
    }

    results = []

    if organism == "larva":
        from models.larva_connectome import ConnectomeLarvaModel
        from models.larva_connectome_variants import ConnectomeLarvaVariantGenerator
        from models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE, ORN_TYPES
        receptor_names = list(ORN_TYPES)

        for param_name, values in param_ranges.items():
            for val in values:
                kwargs = dict(defaults)
                kwargs[param_name] = val
                sender = ConnectomeLarvaModel(**kwargs)
                gen = ConnectomeLarvaVariantGenerator(jitter_scale=jitter_scale)

                seed_metrics = []
                for seed in range(n_seeds):
                    receiver = gen.generate_variant(sender, seed=seed)
                    try:
                        if not gen.sanity_check(receiver):
                            continue
                    except Exception:
                        continue
                    m = _broadcast_metrics(sender, receiver, ODOR_PROFILES, ODOR_VALENCE, receptor_names)
                    seed_metrics.append(m)

                if seed_metrics:
                    results.append({
                        "organism": organism,
                        "parameter": param_name,
                        "value": val,
                        "is_default": val == defaults[param_name],
                        "valid_seeds": len(seed_metrics),
                        **{k: float(np.mean([m[k] for m in seed_metrics]))
                           for k in seed_metrics[0]},
                    })

    elif organism == "adult":
        from models.adult_connectome import ConnectomeAdultModel
        from models.adult_connectome_variants import ConnectomeAdultVariantGenerator
        from models.drosophila_adult import ODOR_PROFILES, ODOR_VALENCE

        for param_name, values in param_ranges.items():
            for val in values:
                kwargs = dict(defaults)
                kwargs[param_name] = val
                sender = ConnectomeAdultModel(**kwargs)
                receptor_names = list(sender._receptor_names)
                gen = ConnectomeAdultVariantGenerator(jitter_scale=jitter_scale)

                seed_metrics = []
                for seed in range(n_seeds):
                    receiver = gen.generate_variant(sender, seed=seed)
                    try:
                        if not gen.sanity_check(receiver):
                            continue
                    except Exception:
                        continue
                    m = _broadcast_metrics(sender, receiver, ODOR_PROFILES, ODOR_VALENCE, receptor_names)
                    seed_metrics.append(m)

                if seed_metrics:
                    results.append({
                        "organism": organism,
                        "parameter": param_name,
                        "value": val,
                        "is_default": val == defaults[param_name],
                        "valid_seeds": len(seed_metrics),
                        **{k: float(np.mean([m[k] for m in seed_metrics]))
                           for k in seed_metrics[0]},
                    })

    return results


if __name__ == "__main__":
    for org in ["larva", "adult"]:
        print(f"\n{'='*70}")
        print(f"  {org.upper()}: Parameter Sensitivity (broadcast, jitter=0.5)")
        print(f"{'='*70}")

        results = run_sensitivity(org, jitter_scale=0.5, n_seeds=3)

        current_param = None
        for r in results:
            if r["parameter"] != current_param:
                current_param = r["parameter"]
                print(f"\n  --- {current_param} ---")
            marker = " *" if r["is_default"] else ""
            print(
                f"    {r['value']:8.3f}  "
                f"transfer={r['transfer_rate']:.2f}  "
                f"corr={r['correlation']:.3f}  "
                f"rsa={r['rsa']:.3f}{marker}"
            )
