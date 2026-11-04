"""Receiver learning experiment for the adult Drosophila stage.

Tests how much mushroom body adaptation (KC→MBON plasticity) can
compensate for imperfect transfer.  The receiver's MBON weights are
retrained on transferred KC activity patterns with known labels.

This models the biological observation that Drosophila can learn new
odor-reward associations rapidly through dopaminergic modulation of
KC→MBON synapses.

The experiment sweeps:
  - Number of learning trials (how many odor presentations the receiver
    needs to adapt)
  - Hardware precision (how much learning substitutes for better hardware)
  - Brain difference (whether learning helps more when brains are more
    different)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.drosophila_adult import (
    DrosophilaAdultModel,
    ODOR_PROFILES,
    ODOR_VALENCE,
    ORN_TYPES,
    KC_NAMES,
)
from models.adult_variants import AdultVariantGenerator
from link.calibration import CalibrationTransform
from hardware.noise import ReadNoise, WritePrecision, ChannelDropout, HardwareChain
from analysis.metrics import correlation, odor_type_decoding


ALL_ODOR_NAMES = list(ODOR_PROFILES.keys())
CAL_INTENSITIES = [0.3, 0.5, 0.75, 1.0, 1.5]
HOLDOUT_INTENSITIES = [0.4, 0.9, 1.25]
CAL_SET = [(name, i) for name in ALL_ODOR_NAMES for i in CAL_INTENSITIES]
HOLDOUT_SET = [(name, i) for name in ALL_ODOR_NAMES for i in HOLDOUT_INTENSITIES]


def _get_orn_signal(model: DrosophilaAdultModel, odor_name: str, intensity: float) -> Dict[str, float]:
    rates = model._orn_type_rates(odor_name, intensity)
    return {orn: float(rates[i]) for i, orn in enumerate(ORN_TYPES)}


def _extract_kc(state: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in state.items() if k.startswith("KC_")}


def _run_learning_trial(
    sender: DrosophilaAdultModel,
    receiver: DrosophilaAdultModel,
    hardware: HardwareChain | None = None,
    n_learning_odors: int = 10,
    rng_seed: int = 0,
) -> Dict[str, float]:
    """Run one transfer trial with optional receiver learning.

    n_learning_odors=0 means no receiver adaptation (baseline).
    """
    rng = np.random.default_rng(rng_seed)

    sender_cal = [_get_orn_signal(sender, name, intensity) for name, intensity in CAL_SET]
    receiver_cal = [_get_orn_signal(receiver, name, intensity) for name, intensity in CAL_SET]
    transform = CalibrationTransform()
    transform.fit(sender_cal, receiver_cal)

    if n_learning_odors > 0:
        learn_odors = list(rng.choice(ALL_ODOR_NAMES, size=min(n_learning_odors, len(ALL_ODOR_NAMES)), replace=False))
        learn_kc: List[np.ndarray] = []
        learn_labels: List[str] = []

        for odor_name in learn_odors:
            for intensity in [0.5, 1.0, 1.5]:
                sender_sig = _get_orn_signal(sender, odor_name, intensity)
                if hardware is not None:
                    sender_sig = hardware.apply(sender_sig, rng)
                cal_sig = transform.transform(sender_sig)
                result = receiver.simulate_from_receptors(cal_sig)
                kc_arr = np.array([result["neurons"].get(kc, 0.0) for kc in KC_NAMES])
                learn_kc.append(kc_arr)
                learn_labels.append(ODOR_VALENCE[odor_name])

        receiver.adapt_mbon_weights(learn_kc, learn_labels)

    odor_types: List[str] = []
    natural_kc: List[Dict[str, float]] = []
    transferred_kc: List[Dict[str, float]] = []
    successes = 0

    for name, inten in HOLDOUT_SET:
        otype = ODOR_VALENCE[name]
        odor_types.append(otype)

        sender_sig = _get_orn_signal(sender, name, inten)
        sender_layers = sender._forward(name, inten)
        sender_beh = sender._behavior_from_mbon(sender_layers["mbon"])

        if hardware is not None:
            sender_sig = hardware.apply(sender_sig, rng)
        cal_sig = transform.transform(sender_sig)
        result = receiver.simulate_from_receptors(cal_sig)
        if result["behavior"] == sender_beh:
            successes += 1

        natural_layers = receiver._forward(name, inten)
        natural_kc.append(_extract_kc(receiver._layers_to_dict(natural_layers)))
        transferred_kc.append(_extract_kc(result["neurons"]))

    n = len(HOLDOUT_SET)
    return {
        "transfer_rate": successes / max(1, n),
        "kc_correlation": correlation(transferred_kc, natural_kc),
        "decoding": odor_type_decoding(odor_types, natural_kc, transferred_kc),
    }


def run_learning_vs_trials(
    jitter: float = 0.75,
    seeds: int = 5,
    trial_counts: List[int] | None = None,
) -> List[Dict]:
    """Sweep number of learning trials at fixed brain difference."""
    if trial_counts is None:
        trial_counts = [0, 1, 3, 5, 10, 15, 20, 30]

    sender = DrosophilaAdultModel()
    gen = AdultVariantGenerator(jitter_scale=jitter)
    results: List[Dict] = []

    for n_learn in trial_counts:
        seed_metrics: List[Dict[str, float]] = []
        for seed in range(seeds):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            m = _run_learning_trial(sender, receiver, n_learning_odors=n_learn, rng_seed=seed)
            seed_metrics.append(m)

        if not seed_metrics:
            continue

        results.append({
            "n_learning_odors": n_learn,
            "jitter_scale": jitter,
            "valid_seeds": len(seed_metrics),
            **{key: float(np.mean([m[key] for m in seed_metrics]))
               for key in ("transfer_rate", "kc_correlation", "decoding")},
        })

    return results


def run_learning_vs_precision(
    jitter: float = 0.75,
    seeds: int = 5,
    noise_levels: List[float] | None = None,
    n_learning_odors: int = 10,
) -> List[Dict]:
    """Sweep hardware noise with and without receiver learning."""
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

    sender = DrosophilaAdultModel()
    gen = AdultVariantGenerator(jitter_scale=jitter)
    results: List[Dict] = []

    for sigma in noise_levels:
        hardware = HardwareChain(
            ReadNoise(sigma=sigma),
            WritePrecision(bits=6, clip_max=200.0),
            ChannelDropout(drop_fraction=0.1),
        )

        for learning in [False, True]:
            n_learn = n_learning_odors if learning else 0
            seed_metrics: List[Dict[str, float]] = []

            for seed in range(seeds):
                receiver = gen.generate_variant(sender, seed=seed)
                if not gen.sanity_check(receiver):
                    continue
                m = _run_learning_trial(
                    sender, receiver,
                    hardware=hardware,
                    n_learning_odors=n_learn,
                    rng_seed=seed,
                )
                seed_metrics.append(m)

            if not seed_metrics:
                continue

            results.append({
                "noise_sigma": sigma,
                "learning": learning,
                "n_learning_odors": n_learn,
                "jitter_scale": jitter,
                "valid_seeds": len(seed_metrics),
                **{key: float(np.mean([m[key] for m in seed_metrics]))
                   for key in ("transfer_rate", "kc_correlation", "decoding")},
            })

    return results


def run_learning_vs_jitter(
    jitter_levels: List[float] | None = None,
    seeds: int = 5,
    n_learning_odors: int = 10,
) -> List[Dict]:
    """Sweep brain difference with and without receiver learning."""
    if jitter_levels is None:
        jitter_levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    sender = DrosophilaAdultModel()
    results: List[Dict] = []

    for jitter in jitter_levels:
        gen = AdultVariantGenerator(jitter_scale=jitter)

        for learning in [False, True]:
            n_learn = n_learning_odors if learning else 0
            seed_metrics: List[Dict[str, float]] = []

            for seed in range(seeds):
                receiver = gen.generate_variant(sender, seed=seed)
                if not gen.sanity_check(receiver):
                    continue
                m = _run_learning_trial(
                    sender, receiver,
                    n_learning_odors=n_learn,
                    rng_seed=seed,
                )
                seed_metrics.append(m)

            if not seed_metrics:
                continue

            results.append({
                "jitter_scale": jitter,
                "learning": learning,
                "n_learning_odors": n_learn,
                "valid_seeds": len(seed_metrics),
                **{key: float(np.mean([m[key] for m in seed_metrics]))
                   for key in ("transfer_rate", "kc_correlation", "decoding")},
            })

    return results
