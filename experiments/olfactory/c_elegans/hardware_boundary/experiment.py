"""Hardware boundary experiment for the worm stage.

Sweeps three hardware impairment axes — read noise, write precision, and
channel dropout — to find where receptor-to-receptor transfer breaks down.
Each axis is swept independently (other two held at ideal) and then a
combined sweep tests all three degrading together.

At each hardware setting the experiment runs across the standard odor set
with a moderately different receiver (jitter_scale=0.75, the median of the
biological range) and reports the same metrics as the other experiments.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
from neural_transfer.models.c_elegans.cook2019 import Cook2019WormModel
from neural_transfer.models.c_elegans.variants import WormVariantGenerator
from neural_transfer.hardware.noise import ReadNoise, WritePrecision, ChannelDropout, TransmitDelay, HardwareChain
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

ODOR_SET: List[Dict[str, object]] = [
    {"odor_intensity": intensity, "odor_type": odor_type}
    for odor_type in ("attractive", "repellent")
    for intensity in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
]

DEFAULT_JITTER = 0.75
DEFAULT_SEEDS = 5


def _run_one_setting(
    sender: object,
    receiver: object,
    hardware: HardwareChain,
    odors: List[Dict[str, object]],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Evaluate transfer quality for one sender-receiver pair at one hardware setting."""
    odor_types: List[str] = []
    natural_states: List[Dict[str, float]] = []
    transferred_states: List[Dict[str, float]] = []
    successes = 0

    for odor in odors:
        intensity, otype = odor["odor_intensity"], odor["odor_type"]
        odor_types.append(otype)

        sender_signal = sender.odor_response(intensity, otype)
        sender_behavior = sender.behavior_from_signal(sender_signal, otype)

        degraded = hardware.apply(sender_signal, rng)

        result = receiver.simulate_from_receptors(degraded, otype)
        if result["behavior"] == sender_behavior:
            successes += 1

        natural_states.append(receiver.network_state(intensity, otype))
        transferred_states.append(result["neurons"])

    return {
        "transfer_rate": successes / len(odors),
        "correlation": correlation(transferred_states, natural_states),
        "mse": mean_mse(transferred_states, natural_states),
        "rsa": representational_similarity(natural_states, transferred_states),
        "odor_id": odor_identification(odor_types, natural_states, transferred_states),
    }


def run_hardware_sweep(
    model_name: str = "hand_built",
    jitter: float = DEFAULT_JITTER,
    seeds: int = DEFAULT_SEEDS,
    noise_levels: List[float] | None = None,
    precision_levels: List[float] | None = None,
    dropout_levels: List[float] | None = None,
    delay_levels: List[float] | None = None,
    odors: List[Dict[str, object]] | None = None,
) -> Dict[str, List[Dict]]:
    """Sweep each hardware axis independently, plus a combined degradation.

    Returns a dict with keys 'read_noise', 'write_precision',
    'channel_dropout', 'transmit_delay', and 'combined', each mapping
    to a list of result rows.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    if precision_levels is None:
        precision_levels = [float("inf"), 8, 6, 5, 4, 3, 2, 1]
    if dropout_levels is None:
        dropout_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    if delay_levels is None:
        delay_levels = [0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0]
    if odors is None:
        odors = ODOR_SET

    ModelClass = MODEL_CLASSES[model_name]
    sender = ModelClass()
    gen = WormVariantGenerator(jitter_scale=jitter)

    def _sweep_axis(make_chain, param_values, param_name):
        rows = []
        for val in param_values:
            chain = make_chain(val)
            seed_metrics = []
            for seed in range(seeds):
                receiver = gen.generate_model_variant(sender, seed=seed)
                rng = np.random.default_rng(seed * 1000 + hash(str(val)) % 10000)
                m = _run_one_setting(sender, receiver, chain, odors, rng)
                seed_metrics.append(m)
            row = {
                "model": model_name,
                param_name: val,
            }
            for key in ("transfer_rate", "correlation", "mse", "rsa", "odor_id"):
                row[key] = float(np.mean([m[key] for m in seed_metrics]))
            rows.append(row)
        return rows

    results = {}

    results["read_noise"] = _sweep_axis(
        lambda sigma: HardwareChain(
            ReadNoise(sigma=sigma),
            WritePrecision(),
            ChannelDropout(),
        ),
        noise_levels,
        "noise_sigma",
    )

    results["write_precision"] = _sweep_axis(
        lambda bits: HardwareChain(
            ReadNoise(),
            WritePrecision(bits=bits),
            ChannelDropout(),
        ),
        precision_levels,
        "precision_bits",
    )

    results["channel_dropout"] = _sweep_axis(
        lambda frac: HardwareChain(
            ReadNoise(),
            WritePrecision(),
            ChannelDropout(drop_fraction=frac),
        ),
        dropout_levels,
        "dropout_fraction",
    )

    results["transmit_delay"] = _sweep_axis(
        lambda ms: HardwareChain(
            ReadNoise(),
            WritePrecision(),
            ChannelDropout(),
            TransmitDelay(delay_ms=ms),
        ),
        delay_levels,
        "delay_ms",
    )

    # Combined: degrade all four together at matched severity
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
            WritePrecision(bits=params["precision_bits"]),
            ChannelDropout(drop_fraction=params["dropout_fraction"]),
            TransmitDelay(delay_ms=params["delay_ms"]),
        )
        seed_metrics = []
        for seed in range(seeds):
            receiver = gen.generate_model_variant(sender, seed=seed)
            rng = np.random.default_rng(seed * 1000 + step_idx)
            m = _run_one_setting(sender, receiver, chain, odors, rng)
            seed_metrics.append(m)
        row = {"model": model_name, "step": step_idx, **params}
        for key in ("transfer_rate", "correlation", "mse", "rsa", "odor_id"):
            row[key] = float(np.mean([m[key] for m in seed_metrics]))
        combined_rows.append(row)
    results["combined"] = combined_rows

    return results
