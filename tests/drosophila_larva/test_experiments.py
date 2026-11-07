"""Tests for Drosophila larva experiments."""

from experiments.olfactory.drosophila_larva.calibration import run_calibration_sweep
from experiments.olfactory.drosophila_larva.hardware_boundary import run_hardware_sweep
from experiments.olfactory.drosophila_larva.channel_sweep import run_channel_sweep


def test_calibration_sweep_shape() -> None:
    results = run_calibration_sweep(jitter_levels=[0.0, 0.5], seeds_per_level=2)
    assert len(results) >= 1
    for row in results:
        for prefix in ("direct", "calibrated"):
            assert f"{prefix}_transfer_rate" in row
            assert f"{prefix}_correlation" in row


def test_calibration_perfect_at_zero_jitter() -> None:
    results = run_calibration_sweep(jitter_levels=[0.0], seeds_per_level=1)
    assert results[0]["direct_transfer_rate"] == 1.0
    assert results[0]["calibrated_transfer_rate"] == 1.0


def test_hardware_sweep_keys() -> None:
    results = run_hardware_sweep(
        seeds=1,
        noise_levels=[0.0, 0.5],
        precision_levels=[float("inf"), 2],
        dropout_levels=[0.0, 0.5],
        delay_levels=[0.0, 100.0],
    )
    assert set(results.keys()) == {
        "read_noise", "write_precision", "channel_dropout",
        "transmit_delay", "combined",
    }
    for axis_rows in results.values():
        assert len(axis_rows) >= 1


def test_channel_sweep_shape() -> None:
    results = run_channel_sweep(
        channel_counts=[1, 10, 21],
        jitter_levels=[0.0],
        seeds_per_level=1,
        channel_seeds=1,
    )
    assert len(results) >= 2
    for row in results:
        assert "n_channels" in row
        assert 0.0 <= row["transfer_rate"] <= 1.0


def test_channel_sweep_improves_with_channels() -> None:
    results = run_channel_sweep(
        channel_counts=[1, 21],
        jitter_levels=[0.0],
        seeds_per_level=1,
        channel_seeds=1,
    )
    low = next(r for r in results if r["n_channels"] == 1)
    high = next(r for r in results if r["n_channels"] == 21)
    assert high["correlation"] > low["correlation"]


