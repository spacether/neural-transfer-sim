"""Approximate operating points for real recording and stimulation technologies.

Each technology maps to a position on one of the hardware sweep axes.
Values are rough estimates from the literature, intended to give intuition
about which technologies fall where on the degradation curves — not to
serve as precise hardware specifications.

Sources and reasoning:

Read noise (σ, in units of neural activation amplitude ~0–2):
  Signal amplitudes in the simulation are normalized activations where a
  strong odor response is ~0.5–1.5.  σ expresses additive measurement
  noise as a fraction of that range.

  Patch clamp: sub-millivolt RMS noise on signals of tens of millivolts
    in C. elegans graded neurons → σ ≈ 0.02.
  Voltage imaging (ASAP3, Voltron): ~1–5% ΔF/F noise at kHz rates on
    signals of 5–20% ΔF/F → σ ≈ 0.1.
  Calcium imaging (GCaMP, high-SNR): single-neuron SNR of 5–15 in
    bright, slow signals → σ ≈ 0.15.  This is the dominant technology
    for C. elegans whole-brain recording.
  Calcium imaging (low-SNR / deep / fast): dimmer or faster indicators,
    crowded neuropil → σ ≈ 0.4.
  Extracellular MEA: relevant for fly spiking neurons, not worm graded
    potentials; spike-sorting noise yields effective σ ≈ 0.25 on rate
    estimates.

Write precision (effective bits of independent intensity control per channel):
  Two-photon holographic optogenetics: spatial light modulator targets
    individual neurons with graded power levels → ~7 effective bits.
  Current injection via electrode: smooth analog waveform, limited by
    electrode resistance and capacitance → ~5 bits.
  Single-photon optogenetics (wide-field ChR2): controls a population,
    limited spatial selectivity → ~2–3 effective bits.
  Thermogenetics (TrpA1): temperature-gated channel, essentially
    on/off per region → ~1 bit.

Channel dropout (fraction of target neurons not reached):
  Light-sheet calcium imaging can image nearly every neuron in C. elegans
    simultaneously → ~0.02 dropout.
  Two-photon targeted expression: most labeled neurons visible → ~0.1.
  Small electrode array (4 channels on 10 neurons): ~0.6 dropout.
  Single patch electrode on one neuron of 10: ~0.9 dropout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TechPoint:
    name: str
    value: float
    short: str


READ_NOISE_TECHS: List[TechPoint] = [
    TechPoint("Patch clamp", 0.02, "Patch"),
    TechPoint("Voltage imaging", 0.10, "VoltImg"),
    TechPoint("Ca²⁺ imaging (high SNR)", 0.15, "GCaMP-hi"),
    TechPoint("Extracellular MEA", 0.25, "MEA"),
    TechPoint("Ca²⁺ imaging (low SNR)", 0.40, "GCaMP-lo"),
]

WRITE_PRECISION_TECHS: List[TechPoint] = [
    TechPoint("2P holographic optogenetics", 7, "2P-holo"),
    TechPoint("Current injection", 5, "Electrode"),
    TechPoint("Wide-field optogenetics", 2.5, "1P-opto"),
    TechPoint("Thermogenetics", 1, "TrpA1"),
]

CHANNEL_DROPOUT_TECHS: List[TechPoint] = [
    TechPoint("Light-sheet (whole brain)", 0.02, "Light-sheet"),
    TechPoint("2P targeted expression", 0.10, "2P-targeted"),
    TechPoint("Small electrode array", 0.60, "4-ch probe"),
    TechPoint("Single electrode", 0.90, "1 electrode"),
]

# End-to-end loop latency: read + process + write for 10-channel worm system.
# The processing itself (10×10 matrix multiply) is sub-microsecond on anything;
# the latency is dominated by the interface hardware (ADC sampling, DAC/stim
# output, and any software overhead).
TRANSMIT_DELAY_TECHS: List[TechPoint] = [
    TechPoint("FPGA closed-loop (Xilinx/Intel)", 0.1, "FPGA"),
    TechPoint("Intan RHD + CereStim", 1.0, "Intan+CereStim"),
    TechPoint("Real-time Linux + DAQ", 5.0, "RT-Linux"),
    TechPoint("Raspberry Pi + USB DAQ", 20.0, "RPi"),
    TechPoint("Software pipeline (Python)", 100.0, "Python loop"),
]
