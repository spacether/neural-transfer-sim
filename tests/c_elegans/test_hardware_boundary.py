"""Tests for the hardware boundary experiment."""

from experiments.c_elegans.hardware_boundary import run_hardware_sweep
from hardware.noise import ReadNoise, WritePrecision, ChannelDropout, HardwareChain

import numpy as np


def test_hardware_chain_ideal_is_identity() -> None:
    chain = HardwareChain(ReadNoise(), WritePrecision(), ChannelDropout())
    signal = {"AWA": 1.0, "AWC_L": 0.8}
    rng = np.random.default_rng(0)
    out = chain.apply(signal, rng)
    assert out == signal


def test_read_noise_adds_noise() -> None:
    rng = np.random.default_rng(42)
    noisy = ReadNoise(sigma=0.5)
    signal = {"AWA": 1.0}
    out = noisy.apply(signal, rng)
    assert out["AWA"] != 1.0


def test_write_precision_quantizes() -> None:
    wp = WritePrecision(bits=1, clip_max=2.0)
    signal = {"AWA": 0.7}
    out = wp.apply(signal)
    assert out["AWA"] in (0.0, 1.0, 2.0)


def test_channel_dropout_zeros_channels() -> None:
    rng = np.random.default_rng(0)
    cd = ChannelDropout(drop_fraction=1.0)
    signal = {"AWA": 1.0, "AWC_L": 0.5}
    out = cd.apply(signal, rng)
    assert all(v == 0.0 for v in out.values())


def test_hardware_sweep_returns_all_axes() -> None:
    results = run_hardware_sweep(
        model_name="hand_built",
        noise_levels=[0.0, 0.5],
        precision_levels=[float("inf"), 2],
        dropout_levels=[0.0, 0.5],
        seeds=1,
    )
    assert set(results.keys()) == {"read_noise", "write_precision", "channel_dropout", "transmit_delay", "combined"}
    for axis_rows in results.values():
        assert len(axis_rows) >= 2
        for row in axis_rows:
            assert 0.0 <= row["correlation"] <= 1.0 or row["correlation"] >= -1.0
            assert 0.0 <= row["odor_id"] <= 1.0


def test_transmit_delay_adds_drift() -> None:
    from hardware.noise import TransmitDelay
    rng = np.random.default_rng(42)
    td = TransmitDelay(delay_ms=1000.0, tau_ms=3000.0)
    signal = {"AWA": 1.0, "AWC_L": 0.8}
    out = td.apply(signal, rng)
    assert out["AWA"] != 1.0 or out["AWC_L"] != 0.8


def test_transmit_delay_zero_is_identity() -> None:
    from hardware.noise import TransmitDelay
    rng = np.random.default_rng(0)
    td = TransmitDelay(delay_ms=0.0)
    signal = {"AWA": 1.0}
    assert td.apply(signal, rng) == signal


def test_hardware_sweep_cook2019() -> None:
    results = run_hardware_sweep(
        model_name="cook2019",
        noise_levels=[0.0, 0.5],
        precision_levels=[float("inf"), 2],
        dropout_levels=[0.0, 0.5],
        seeds=1,
    )
    assert "read_noise" in results
    assert results["read_noise"][0]["model"] == "cook2019"
