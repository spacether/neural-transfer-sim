"""Hardware impairment models for the read-transmit-write pipeline.

Each model degrades the signal at one stage, parameterized so sweeps can
find the boundary where transfer breaks down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ReadNoise:
    """Additive Gaussian noise on the recorded signal, simulating electrode
    or calcium-imaging measurement noise.

    sigma is in the same units as the neural activation (typically 0-2).
    """

    sigma: float = 0.0

    def apply(self, signal: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
        if self.sigma <= 0:
            return dict(signal)
        return {
            k: max(0.0, v + float(rng.normal(0, self.sigma)))
            for k, v in signal.items()
        }


@dataclass
class WritePrecision:
    """Quantization of the stimulation signal, simulating limited bit depth
    of the stimulating device.

    bits=inf means no quantization.  bits=3 means 8 levels over the signal
    range, etc.  The signal is clipped to [0, clip_max] before quantizing.
    """

    bits: float = float("inf")
    clip_max: float = 2.0

    def apply(self, signal: Dict[str, float]) -> Dict[str, float]:
        if self.bits == float("inf") or self.bits <= 0:
            return dict(signal)
        n_levels = 2 ** self.bits
        step = self.clip_max / n_levels
        return {
            k: float(round(min(max(v, 0.0), self.clip_max) / step) * step)
            for k, v in signal.items()
        }


@dataclass
class ChannelDropout:
    """Random per-trial channel dropout, simulating unreliable connections.

    drop_fraction is the probability each channel is zeroed on a given trial.
    This is distinct from the structural dropout in the variant generator,
    which permanently removes connections between individuals.
    """

    drop_fraction: float = 0.0

    def apply(self, signal: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
        if self.drop_fraction <= 0:
            return dict(signal)
        return {
            k: 0.0 if rng.random() < self.drop_fraction else v
            for k, v in signal.items()
        }


@dataclass
class TransmitDelay:
    """Signal drift during transmission delay.

    In a real-time link, the sender's neural state evolves between read
    and write.  If the odor environment changes during the delay, the
    receiver gets a stale signal.  This is modelled as additional noise
    whose magnitude grows with sqrt(delay / tau), where tau is the
    organism's behavioral decision timescale.

    For C. elegans chemotaxis, tau ≈ 3000 ms (a turn decision integrates
    over several seconds).  For fly escape, tau ≈ 200 ms.
    """

    delay_ms: float = 0.0
    tau_ms: float = 3000.0
    drift_rate: float = 0.3

    def apply(self, signal: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
        if self.delay_ms <= 0:
            return dict(signal)
        sigma = self.drift_rate * (self.delay_ms / self.tau_ms) ** 0.5
        return {
            k: max(0.0, v + float(rng.normal(0, sigma)))
            for k, v in signal.items()
        }


@dataclass
class HardwareChain:
    """Applies read noise, delay drift, channel dropout, and write precision in sequence."""

    read_noise: ReadNoise
    write_precision: WritePrecision
    channel_dropout: ChannelDropout
    transmit_delay: TransmitDelay | None = None

    def apply(self, signal: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
        out = self.read_noise.apply(signal, rng)
        if self.transmit_delay is not None:
            out = self.transmit_delay.apply(out, rng)
        out = self.channel_dropout.apply(out, rng)
        out = self.write_precision.apply(out)
        return out
