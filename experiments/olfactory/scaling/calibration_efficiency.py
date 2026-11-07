"""Calibration sample efficiency sweep across organisms.

Measures how transfer quality (odor identification accuracy) changes as
the number of calibration training samples varies. Higher-dimensional
bottlenecks need more samples to fit the linear mapping, so the curves
should diverge across organisms — unlike the channel sweep, which
collapsed when normalized by bottleneck fraction.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.link.calibration import CalibrationTransform
from neural_transfer.analysis.metrics import odor_identification


def _worm_efficiency(sample_counts: List[int], seeds: int, jitter: float) -> List[Dict]:
    from neural_transfer.models.c_elegans.cook2019 import Cook2019WormModel
    from neural_transfer.models.c_elegans.variants import WormVariantGenerator

    sender = Cook2019WormModel()
    gen = WormVariantGenerator(jitter_scale=jitter)

    all_odors = [
        {"odor_intensity": i, "odor_type": t}
        for t in ("attractive", "repellent")
        for i in (0.3, 0.4, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0)
    ]
    holdout = [o for o in all_odors if o["odor_intensity"] in (0.4, 0.9, 1.75)]
    cal_pool = [o for o in all_odors if o not in holdout]

    results = []
    for n_samples in sample_counts:
        n_use = min(n_samples, len(cal_pool))
        seed_scores = []
        for seed in range(seeds):
            receiver = gen.generate_model_variant(sender, seed=seed)
            rng = np.random.default_rng(seed)
            cal_subset = [cal_pool[i] for i in rng.choice(len(cal_pool), size=n_use, replace=n_use > len(cal_pool))]

            sender_cal = [sender.odor_response(o["odor_intensity"], o["odor_type"]) for o in cal_subset]
            receiver_cal = [receiver.odor_response(o["odor_intensity"], o["odor_type"]) for o in cal_subset]

            transform = CalibrationTransform()
            transform.fit(sender_cal, receiver_cal)

            labels, nat_states, cal_states = [], [], []
            for o in holdout:
                labels.append(o["odor_type"])
                nat_states.append(receiver.network_state(o["odor_intensity"], o["odor_type"]))
                sig = sender.odor_response(o["odor_intensity"], o["odor_type"])
                cal_sig = transform.transform(sig)
                result = receiver.simulate_from_receptors(cal_sig, o["odor_type"])
                cal_states.append(result["neurons"])

            seed_scores.append(odor_identification(labels, nat_states, cal_states))

        results.append({
            "n_samples": n_samples,
            "odor_id": float(np.mean(seed_scores)),
            "odor_id_std": float(np.std(seed_scores)),
        })
    return results


def _larva_efficiency(sample_counts: List[int], seeds: int, jitter: float) -> List[Dict]:
    from neural_transfer.models.drosophila_larva import ConnectomeLarvaModel
    from neural_transfer.models.drosophila_larva import ConnectomeLarvaVariantGenerator
    from neural_transfer.models.drosophila_larva import ODOR_PROFILES

    sender = ConnectomeLarvaModel()
    gen = ConnectomeLarvaVariantGenerator(jitter_scale=jitter)
    orn_order = sender._orn_order
    odor_names = list(ODOR_PROFILES.keys())

    cal_intensities = [0.3, 0.5, 0.75, 1.0, 1.5]
    holdout_intensities = [0.4, 0.9, 1.25]
    cal_pool = [(name, i) for name in odor_names for i in cal_intensities]
    holdout = [(name, i) for name in odor_names for i in holdout_intensities]

    def _sig(model, name, intensity):
        rates = model._orn_rates(name, intensity)
        return {orn: float(rates[i]) for i, orn in enumerate(orn_order)}

    results = []
    for n_samples in sample_counts:
        n_use = min(n_samples, len(cal_pool))
        seed_scores = []
        for seed in range(seeds):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(cal_pool), size=n_use, replace=n_use > len(cal_pool))
            cal_subset = [cal_pool[i] for i in indices]

            sender_cal = [_sig(sender, n, i) for n, i in cal_subset]
            receiver_cal = [_sig(receiver, n, i) for n, i in cal_subset]
            transform = CalibrationTransform()
            transform.fit(sender_cal, receiver_cal)

            labels, nat_states, cal_states = [], [], []
            for name, intensity in holdout:
                labels.append(name)
                nat_layers = receiver._forward(name, intensity)
                nat_states.append(receiver._layers_to_dict(nat_layers))
                cal_sig = transform.transform(_sig(sender, name, intensity))
                result = receiver.simulate_from_receptors(cal_sig)
                cal_states.append(result["neurons"])

            seed_scores.append(odor_identification(labels, nat_states, cal_states))

        if seed_scores:
            results.append({
                "n_samples": n_samples,
                "odor_id": float(np.mean(seed_scores)),
                "odor_id_std": float(np.std(seed_scores)),
            })
    return results


def _adult_efficiency(sample_counts: List[int], seeds: int, jitter: float) -> List[Dict]:
    from neural_transfer.models.drosophila_adult import ConnectomeAdultModel
    from neural_transfer.models.drosophila_adult import ConnectomeAdultVariantGenerator
    from neural_transfer.models.drosophila_adult import ODOR_PROFILES

    sender = ConnectomeAdultModel()
    gen = ConnectomeAdultVariantGenerator(jitter_scale=jitter)
    receptor_names = list(sender._receptor_names)
    odor_names = list(ODOR_PROFILES.keys())

    cal_intensities = [0.3, 0.5, 0.75, 1.0, 1.5]
    holdout_intensities = [0.4, 0.9, 1.25]
    cal_pool = [(name, i) for name in odor_names for i in cal_intensities]
    holdout = [(name, i) for name in odor_names for i in holdout_intensities]

    def _sig(model, name, intensity):
        rates = model._orn_rates(name, intensity)
        return {orn: float(rates[i]) for i, orn in enumerate(receptor_names)}

    results = []
    for n_samples in sample_counts:
        n_use = min(n_samples, len(cal_pool))
        seed_scores = []
        for seed in range(seeds):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(cal_pool), size=n_use, replace=n_use > len(cal_pool))
            cal_subset = [cal_pool[i] for i in indices]

            sender_cal = [_sig(sender, n, i) for n, i in cal_subset]
            receiver_cal = [_sig(receiver, n, i) for n, i in cal_subset]
            transform = CalibrationTransform()
            transform.fit(sender_cal, receiver_cal)

            labels, nat_states, cal_states = [], [], []
            for name, intensity in holdout:
                labels.append(name)
                nat_layers = receiver._forward(name, intensity)
                nat_state = receiver._layers_to_dict(nat_layers)
                nat_kc = {k: v for k, v in nat_state.items() if k.startswith("KC_")}
                nat_states.append(nat_kc)
                cal_sig = transform.transform(_sig(sender, name, intensity))
                result = receiver.simulate_from_receptors(cal_sig)
                cal_kc = {k: v for k, v in result["neurons"].items() if k.startswith("KC_")}
                cal_states.append(cal_kc)

            seed_scores.append(odor_identification(labels, nat_states, cal_states))

        if seed_scores:
            results.append({
                "n_samples": n_samples,
                "odor_id": float(np.mean(seed_scores)),
                "odor_id_std": float(np.std(seed_scores)),
            })
    return results


def _zebrafish_efficiency(sample_counts: List[int], seeds: int, jitter: float) -> List[Dict]:
    from neural_transfer.models.zebrafish import ZebrafishModel, ODOR_PROFILES, OR_TYPES
    from neural_transfer.models.zebrafish.variants import ZebrafishVariantGenerator

    sender = ZebrafishModel()
    gen = ZebrafishVariantGenerator(jitter_scale=jitter)
    odor_names = list(ODOR_PROFILES.keys())

    cal_intensities = [0.3, 0.5, 0.75, 1.0, 1.5]
    holdout_intensities = [0.4, 0.9, 1.25]
    cal_pool = [(name, i) for name in odor_names for i in cal_intensities]
    holdout = [(name, i) for name in odor_names for i in holdout_intensities]

    def _sig(model, name, intensity):
        rates = model._orn_type_rates(name, intensity)
        return {orn: float(rates[i]) for i, orn in enumerate(OR_TYPES)}

    results = []
    for n_samples in sample_counts:
        n_use = min(n_samples, len(cal_pool))
        seed_scores = []
        for seed in range(seeds):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(cal_pool), size=n_use, replace=n_use > len(cal_pool))
            cal_subset = [cal_pool[i] for i in indices]

            sender_cal = [_sig(sender, n, i) for n, i in cal_subset]
            receiver_cal = [_sig(receiver, n, i) for n, i in cal_subset]
            transform = CalibrationTransform()
            transform.fit(sender_cal, receiver_cal)

            labels, nat_states, cal_states = [], [], []
            for name, intensity in holdout:
                labels.append(name)
                nat_layers = receiver._forward(name, intensity)
                nat_state = receiver._layers_to_dict(nat_layers)
                nat_dp = {k: v for k, v in nat_state.items() if k.startswith("Dp_")}
                nat_states.append(nat_dp)
                cal_sig = transform.transform(_sig(sender, name, intensity))
                result = receiver.simulate_from_receptors(cal_sig)
                cal_dp = {k: v for k, v in result["neurons"].items() if k.startswith("Dp_")}
                cal_states.append(cal_dp)

            seed_scores.append(odor_identification(labels, nat_states, cal_states))

        if seed_scores:
            results.append({
                "n_samples": n_samples,
                "odor_id": float(np.mean(seed_scores)),
                "odor_id_std": float(np.std(seed_scores)),
            })
    return results


def run_calibration_efficiency(
    seeds: int = 5,
    jitter: float = 1.0,
    sample_counts: List[int] | None = None,
) -> Dict[str, List[Dict]]:
    if sample_counts is None:
        sample_counts = [3, 5, 10, 15, 20, 30, 50, 75, 100, 150]

    return {
        "C. elegans": _worm_efficiency(sample_counts, seeds, jitter),
        "D. larva": _larva_efficiency(sample_counts, seeds, jitter),
        "D. adult": _adult_efficiency(sample_counts, seeds, jitter),
        "Zebrafish": _zebrafish_efficiency(sample_counts, seeds, jitter),
    }
