"""PN channel sweep: 2D sweep of read and write PN counts.

Reads from N_read PNs of the sender, writes to N_write PNs of the receiver,
with calibration mapping the read subset to the write subset.  Measures KC
correlation — does the receiver's odor representation match what it would
produce naturally?

Unread PNs are unknown to the calibration; unwritten PNs receive baseline
activity (the receiver's spontaneous PN firing rate).

Uses the connectome-based larva model (Winding et al. 2023).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.larva_connectome import ConnectomeLarvaModel
from models.larva_connectome_variants import ConnectomeLarvaVariantGenerator
from models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE
from link.calibration import CalibrationTransform
from analysis.metrics import (
    correlation,
    odor_identification,
    bootstrap_ci,
)


ALL_ODOR_NAMES = list(ODOR_PROFILES.keys())
CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_ODOR_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_ODOR_NAMES for i in HOLDOUT_INTENSITIES]

METRIC_KEYS = ("transfer_rate", "correlation", "odor_id")


def _agg_with_ci(seed_metrics: List[Dict[str, float]]) -> Dict:
    row: Dict = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in seed_metrics]
        mean, lo, hi = bootstrap_ci(vals)
        row[k] = mean
        row[f"{k}_ci_lo"] = lo
        row[f"{k}_ci_hi"] = hi
    return row


def _read_pn(model: ConnectomeLarvaModel, odor_name: str, intensity: float,
             pn_indices: List[int]) -> Dict[str, float]:
    """Read a subset of the sender's PNs."""
    layers = model._forward(odor_name, intensity)
    pn_names = model.pn_names
    return {pn_names[i]: float(layers["pn"][i]) for i in pn_indices}


def _write_pn(model: ConnectomeLarvaModel, signal: Dict[str, float]) -> Dict:
    """Inject at PN level, filling unspecified PNs with baseline."""
    full_pn = {pn: model.pn_baseline for pn in model.pn_names}
    full_pn.update(signal)
    return model.simulate_from_pn(full_pn)


def _extract_kc(state: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in state.items() if k.startswith("KC_")}


def _run_one(
    sender: ConnectomeLarvaModel,
    receiver: ConnectomeLarvaModel,
    read_indices: List[int],
    write_indices: List[int],
) -> Dict[str, float]:
    """Evaluate one (n_read, n_write) combination with calibration."""
    pn_names = sender.pn_names
    read_pns = [pn_names[i] for i in read_indices]
    write_pns = [pn_names[i] for i in write_indices]

    sender_cal = [_read_pn(sender, n, i, read_indices) for n, i in CAL_SET]
    receiver_cal = []
    for name, inten in CAL_SET:
        layers = receiver._forward(name, inten)
        r_pn_names = receiver.pn_names
        receiver_cal.append({pn: float(layers["pn"][j])
                             for j, pn in enumerate(r_pn_names) if pn in write_pns})

    transform = CalibrationTransform()
    transform.neurons = tuple(sorted(set(read_pns) | set(write_pns)))
    transform.fit(sender_cal, receiver_cal)

    odor_names: List[str] = []
    natural_kc: List[Dict[str, float]] = []
    transferred_kc: List[Dict[str, float]] = []
    successes = 0

    write_set = set(write_pns)
    for name, inten in HOLDOUT_SET:
        odor_names.append(name)

        sender_sig = _read_pn(sender, name, inten, read_indices)
        sender_layers = sender._forward(name, inten)
        sender_beh = sender._behavior_from_output(sender_layers["output"])

        cal_sig = transform.transform(sender_sig)
        write_sig = {k: v for k, v in cal_sig.items() if k in write_set}
        result = _write_pn(receiver, write_sig)
        if result["behavior"] == sender_beh:
            successes += 1

        natural_layers = receiver._forward(name, inten)
        natural_kc.append(_extract_kc(receiver._layers_to_dict(natural_layers)))
        transferred_kc.append(_extract_kc(result["neurons"]))

    n = len(HOLDOUT_SET)
    return {
        "transfer_rate": successes / max(1, n),
        "correlation": correlation(transferred_kc, natural_kc),
        "odor_id": odor_identification(odor_names, natural_kc, transferred_kc),
    }


def run_pn_channel_sweep(
    channel_counts: List[int] | None = None,
    jitter: float = 0.5,
    seeds: int = 5,
    channel_seeds: int = 3,
) -> List[Dict]:
    """Sweep all (n_read, n_write) PN combinations."""
    if channel_counts is None:
        channel_counts = [1, 3, 5, 7, 10, 14, 17, 21]

    sender = ConnectomeLarvaModel()
    gen = ConnectomeLarvaVariantGenerator(jitter_scale=jitter)
    n_pn = len(sender.pn_names)
    results: List[Dict] = []

    for n_read in channel_counts:
        for n_write in channel_counts:
            seed_metrics: List[Dict[str, float]] = []

            for cs in range(channel_seeds):
                rng = np.random.default_rng(cs + 1000)
                read_idx = sorted(rng.choice(n_pn, size=min(n_read, n_pn), replace=False).tolist())
                write_idx = sorted(rng.choice(n_pn, size=min(n_write, n_pn), replace=False).tolist())

                for s in range(seeds):
                    receiver = gen.generate_variant(sender, seed=s)
                    if not gen.sanity_check(receiver):
                        continue
                    m = _run_one(sender, receiver, read_idx, write_idx)
                    seed_metrics.append(m)

            if not seed_metrics:
                continue

            row: Dict = {
                "n_read": n_read,
                "n_write": n_write,
                "jitter_scale": jitter,
                "valid_trials": len(seed_metrics),
            }
            row.update(_agg_with_ci(seed_metrics))
            results.append(row)

    return results
