"""Tests for the receptor-to-receptor and calibration experiments."""

from experiments.receptor_to_receptor_broadcast import run_receptor_to_receptor_sweep
from experiments.calibration import run_calibration_sweep


def test_receptor_to_receptor_sweep_returns_expected_shape() -> None:
    levels = [0.0, 0.15]
    results = run_receptor_to_receptor_sweep(jitter_levels=levels, seeds_per_level=2)

    assert len(results) == 2
    for row in results:
        assert 0.0 <= row["transfer_rate"] <= 1.0
        assert -1.0 <= row["correlation"] <= 1.0
        assert row["mse"] >= 0.0
        assert -1.0 <= row["rsa"] <= 1.0 or row["rsa"] != row["rsa"]  # allow NaN
        assert 0.0 <= row["decoding"] <= 1.0


def test_receptor_to_receptor_perfect_at_zero_jitter() -> None:
    results = run_receptor_to_receptor_sweep(jitter_levels=[0.0], seeds_per_level=1)
    assert results[0]["transfer_rate"] == 1.0
    assert results[0]["correlation"] > 0.99
    assert results[0]["mse"] < 1e-10
    assert results[0]["rsa"] > 0.99
    assert results[0]["decoding"] > 0.9


def test_calibration_sweep_returns_expected_shape() -> None:
    levels = [0.0, 0.15]
    results = run_calibration_sweep(jitter_levels=levels, seeds_per_level=2)

    assert len(results) == 2
    for row in results:
        for prefix in ("direct", "calibrated"):
            assert f"{prefix}_transfer_rate" in row
            assert f"{prefix}_correlation" in row
            assert f"{prefix}_mse" in row
            assert f"{prefix}_rsa" in row
            assert f"{prefix}_decoding" in row


def test_calibration_perfect_at_zero_jitter() -> None:
    results = run_calibration_sweep(jitter_levels=[0.0], seeds_per_level=1)
    assert results[0]["direct_transfer_rate"] == 1.0
    assert results[0]["calibrated_transfer_rate"] == 1.0
    assert results[0]["direct_mse"] < 1e-10
    assert results[0]["calibrated_mse"] < 1e-5


def test_receptor_to_receptor_sweep_cook2019() -> None:
    results = run_receptor_to_receptor_sweep(
        jitter_levels=[0.0], seeds_per_level=1, model_name="cook2019",
    )
    assert results[0]["model"] == "cook2019"
    assert results[0]["transfer_rate"] == 1.0
    assert results[0]["correlation"] > 0.99


def test_calibration_sweep_cook2019() -> None:
    results = run_calibration_sweep(
        jitter_levels=[0.0], seeds_per_level=1, model_name="cook2019",
    )
    assert results[0]["model"] == "cook2019"
    assert results[0]["direct_transfer_rate"] == 1.0
    assert results[0]["calibrated_transfer_rate"] == 1.0


def test_generate_model_variant_sanity() -> None:
    from models.worm_connectome import HandBuiltWormModel
    from models.worm_variants import WormVariantGenerator

    sender = HandBuiltWormModel()
    generator = WormVariantGenerator(jitter_scale=0.3)

    for seed in range(5):
        variant = generator.generate_model_variant(sender, seed=seed)
        assert generator.sanity_check(variant)
