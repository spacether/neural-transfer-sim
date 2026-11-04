"""Hardware boundary experiment for the Drosophila larva stage.

Sweeps hardware impairment axes — read noise, write precision, channel
dropout, and transmit delay — to find where ORN-to-ORN transfer breaks
down in the larval olfactory circuit.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.drosophila_larva import (
    DrosophilaLarvaModel,
    ODOR_PROFILES,
    ODOR_VALENCE,
    ORN_TYPES,
)
from models.larva_variants import LarvaVariantGenerator
from hardware.noise import ReadNoise, WritePrecision, ChannelDropout, TransmitDelay, HardwareChain
from analysis.metrics import (
    correlation,
    mean_mse,
    representational_similarity,
    odor_type_decoding,
)

ODOR_SET: List[Dict[str, object]] = [
    {"odor_name": name, "odor_intensity": intensity}
    for name in ODOR_PROFILES
    for intensity in (0.5, 1.0, 1.5)
]

DEFAULT_JITTER = 0.75
DEFAULT_SEEDS = 5


def _run_one_setting(
    sender: DrosophilaLarvaModel,
    receiver: DrosophilaLarvaModel,
    hardware: HardwareChain,
    rng: np.random.Generator,
) -> Dict[str, float]:
    odor_types: List[str] = []
    natural_states: List[Dict[str, float]] = []
    transferred_states: List[Dict[str, float]] = []
    successes = 0

    for odor in ODOR_SET:
        name = odor["odor_name"]
        intensity = odor["odor_intensity"]
        otype = ODOR_VALENCE[name]
        odor_types.append(otype)

        sender_orn = sender._orn_rates(name, intensity)
        sender_signal = {orn: float(sender_orn[i]) for i, orn in enumerate(ORN_TYPES)}
        sender_layers = sender._forward(name, intensity)
        sender_beh = sender._behavior_from_mbon(sender_layers["mbon"])

        degraded = hardware.apply(sender_signal, rng)
        result = receiver.simulate_from_receptors(degraded)
        if result["behavior"] == sender_beh:
            successes += 1

        natural_layers = receiver._forward(name, intensity)
        natural_states.append(receiver._layers_to_dict(natural_layers))
        transferred_states.append(result["neurons"])

    return {
        "transfer_rate": successes / len(ODOR_SET),
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "decoding": odor_type_decoding(odor_types, natural_states, transferred_states),
    }


def run_hardware_sweep(
    jitter: float = DEFAULT_JITTER,
    seeds: int = DEFAULT_SEEDS,
    noise_levels: List[float] | None = None,
    precision_levels: List[float] | None = None,
    dropout_levels: List[float] | None = None,
    delay_levels: List[float] | None = None,
) -> Dict[str, List[Dict]]:
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    if precision_levels is None:
        precision_levels = [float("inf"), 8, 6, 5, 4, 3, 2, 1]
    if dropout_levels is None:
        dropout_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    if delay_levels is None:
        delay_levels = [0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]

    sender = DrosophilaLarvaModel()
    gen = LarvaVariantGenerator(jitter_scale=jitter)

    def _sweep_axis(make_chain, param_values, param_name):
        rows = []
        for val in param_values:
            chain = make_chain(val)
            seed_metrics = []
            for seed in range(seeds):
                receiver = gen.generate_variant(sender, seed=seed)
                if not gen.sanity_check(receiver):
                    continue
                rng = np.random.default_rng(seed * 1000 + hash(str(val)) % 10000)
                m = _run_one_setting(sender, receiver, chain, rng)
                seed_metrics.append(m)
            if not seed_metrics:
                continue
            row = {param_name: val}
            for key in ("transfer_rate", "correlation", "mse", "rsa", "decoding"):
                row[key] = float(np.mean([m[key] for m in seed_metrics]))
            rows.append(row)
        return rows

    results = {}

    results["read_noise"] = _sweep_axis(
        lambda sigma: HardwareChain(ReadNoise(sigma=sigma), WritePrecision(), ChannelDropout()),
        noise_levels, "noise_sigma",
    )
    results["write_precision"] = _sweep_axis(
        lambda bits: HardwareChain(ReadNoise(), WritePrecision(bits=bits, clip_max=100.0), ChannelDropout()),
        precision_levels, "precision_bits",
    )
    results["channel_dropout"] = _sweep_axis(
        lambda frac: HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout(drop_fraction=frac)),
        dropout_levels, "dropout_fraction",
    )
    results["transmit_delay"] = _sweep_axis(
        lambda ms: HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout(), TransmitDelay(delay_ms=ms, tau_ms=500.0)),
        delay_levels, "delay_ms",
    )

    combined_steps = [
        {"noise_sigma": 0.0, "precision_bits": float("inf"), "dropout_fraction": 0.0, "delay_ms": 0.0},
        {"noise_sigma": 0.05, "precision_bits": 8, "dropout_fraction": 0.05, "delay_ms": 0.1},
        {"noise_sigma": 0.1, "precision_bits": 6, "dropout_fraction": 0.1, "delay_ms": 1.0},
        {"noise_sigma": 0.2, "precision_bits": 5, "dropout_fraction": 0.15, "delay_ms": 5.0},
        {"noise_sigma": 0.3, "precision_bits": 4, "dropout_fraction": 0.2, "delay_ms": 20.0},
        {"noise_sigma": 0.5, "precision_bits": 3, "dropout_fraction": 0.3, "delay_ms": 100.0},
        {"noise_sigma": 0.75, "precision_bits": 2, "dropout_fraction": 0.4, "delay_ms": 500.0},
        {"noise_sigma": 1.0, "precision_bits": 1, "dropout_fraction": 0.5, "delay_ms": 2000.0},
    ]
    combined_rows = []
    for step_idx, params in enumerate(combined_steps):
        chain = HardwareChain(
            ReadNoise(sigma=params["noise_sigma"]),
            WritePrecision(bits=params["precision_bits"], clip_max=100.0),
            ChannelDropout(drop_fraction=params["dropout_fraction"]),
            TransmitDelay(delay_ms=params["delay_ms"]),
        )
        seed_metrics = []
        for seed in range(seeds):
            receiver = gen.generate_variant(sender, seed=seed)
            if not gen.sanity_check(receiver):
                continue
            rng = np.random.default_rng(seed * 1000 + step_idx)
            m = _run_one_setting(sender, receiver, chain, rng)
            seed_metrics.append(m)
        if seed_metrics:
            row = {"step": step_idx, **params}
            for key in ("transfer_rate", "correlation", "mse", "rsa", "decoding"):
                row[key] = float(np.mean([m[key] for m in seed_metrics]))
            combined_rows.append(row)
    results["combined"] = combined_rows

    return results
