"""Read site vs write site experiment for the Drosophila larva stage.

Tests every pairing of read site (ORN, PN, KC) and write site (ORN, PN)
to determine which combination produces the best transfer quality.

The key prediction: reading late (where the signal reflects what the sender
detected) and writing at the bottleneck (PN/glomerular level, where few
channels carry everything) should give the best results.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.drosophila_larva import (
    DrosophilaLarvaModel,
    ODOR_PROFILES,
    ODOR_VALENCE,
    ORN_TYPES,
    PN_NAMES,
    KC_NAMES,
    SITE_NEURONS,
)
from models.larva_variants import LarvaVariantGenerator
from link.calibration import CalibrationTransform
from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_type_decoding,
)


READ_SITES = ["orn", "pn", "kc"]
WRITE_SITES = ["orn", "pn"]

ALL_ODOR_NAMES = list(ODOR_PROFILES.keys())
CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_ODOR_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_ODOR_NAMES for i in HOLDOUT_INTENSITIES]


def _read_signal(model: DrosophilaLarvaModel, odor_name: str, intensity: float, site: str) -> Dict[str, float]:
    """Read the sender's activity at a given pathway site."""
    layers = model._forward(odor_name, intensity)
    neuron_names = SITE_NEURONS[site]
    arr = layers[site]
    return {n: float(arr[i]) for i, n in enumerate(neuron_names)}


def _write_signal(model: DrosophilaLarvaModel, signal: Dict[str, float], site: str) -> Dict:
    """Inject a signal at a given write site and propagate forward."""
    if site == "orn":
        return model.simulate_from_receptors(signal)
    elif site == "pn":
        return model.simulate_from_pn(signal)
    else:
        raise ValueError(f"Cannot write at site: {site}")


def _run_one_pair(
    sender: DrosophilaLarvaModel,
    receiver: DrosophilaLarvaModel,
    read_site: str,
    write_site: str,
) -> Dict[str, float]:
    """Evaluate one read-write site pair with calibration."""
    sender_cal = [_read_signal(sender, name, inten, read_site)
                  for name, inten in CAL_SET]
    receiver_cal = [_read_signal(receiver, name, inten, write_site)
                    for name, inten in CAL_SET]

    transform = CalibrationTransform()
    transform.fit(sender_cal, receiver_cal)

    odor_types: List[str] = []
    natural_kc: List[Dict[str, float]] = []
    transferred_kc: List[Dict[str, float]] = []
    successes = 0

    for name, inten in HOLDOUT_SET:
        otype = ODOR_VALENCE[name]
        odor_types.append(otype)

        sender_sig = _read_signal(sender, name, inten, read_site)
        sender_layers = sender._forward(name, inten)
        sender_beh = sender._behavior_from_mbon(sender_layers["mbon"])

        cal_sig = transform.transform(sender_sig)
        result = _write_signal(receiver, cal_sig, write_site)
        if result["behavior"] == sender_beh:
            successes += 1

        natural_layers = receiver._forward(name, inten)
        nat_full = receiver._layers_to_dict(natural_layers)
        xfer_full = result["neurons"]
        natural_kc.append({k: v for k, v in nat_full.items() if k.startswith("KC_")})
        transferred_kc.append({k: v for k, v in xfer_full.items() if k.startswith("KC_")})

    n = len(HOLDOUT_SET)
    return {
        "transfer_rate": successes / max(1, n),
        "correlation": correlation(transferred_kc, natural_kc),
        "mse": mean_mse(transferred_kc, natural_kc),
        "rsa": representational_similarity(natural_kc, transferred_kc),
        "decoding": odor_type_decoding(odor_types, natural_kc, transferred_kc),
    }


def run_site_sweep(
    jitter_levels: List[float] | None = None,
    seeds_per_level: int = 5,
) -> List[Dict]:
    """Sweep all read-write site pairings at each brain-difference level."""
    if jitter_levels is None:
        jitter_levels = [0.0, 0.5, 1.0]

    sender = DrosophilaLarvaModel()
    results: List[Dict] = []

    for read_site in READ_SITES:
        for write_site in WRITE_SITES:
            for jitter in jitter_levels:
                gen = LarvaVariantGenerator(jitter_scale=jitter)
                seed_metrics: List[Dict[str, float]] = []

                for seed in range(seeds_per_level):
                    receiver = gen.generate_variant(sender, seed=seed)
                    if not gen.sanity_check(receiver):
                        continue
                    m = _run_one_pair(sender, receiver, read_site, write_site)
                    seed_metrics.append(m)

                if not seed_metrics:
                    continue

                results.append({
                    "read_site": read_site,
                    "write_site": write_site,
                    "jitter_scale": jitter,
                    "valid_seeds": len(seed_metrics),
                    **{
                        key: float(np.mean([m[key] for m in seed_metrics]))
                        for key in ("transfer_rate", "correlation", "mse", "rsa", "decoding")
                    },
                })

    return results
