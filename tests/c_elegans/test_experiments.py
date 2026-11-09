"""Tests for the calibration experiments."""

from experiments.olfactory.c_elegans.calibration import run_calibration_sweep


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
            assert f"{prefix}_odor_id" in row


def test_calibration_perfect_at_zero_jitter() -> None:
    results = run_calibration_sweep(jitter_levels=[0.0], seeds_per_level=1)
    assert results[0]["direct_transfer_rate"] == 1.0
    assert results[0]["calibrated_transfer_rate"] == 1.0
    assert results[0]["direct_mse"] < 1e-10
    assert results[0]["calibrated_mse"] < 1e-5


def test_calibration_sweep_cook2019() -> None:
    results = run_calibration_sweep(
        jitter_levels=[0.0], seeds_per_level=1, model_name="cook2019",
    )
    assert results[0]["model"] == "cook2019"
    assert results[0]["direct_transfer_rate"] == 1.0
    assert results[0]["calibrated_transfer_rate"] == 1.0


def test_generate_model_variant_sanity() -> None:
    from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
    from neural_transfer.models.c_elegans.variants import WormVariantGenerator

    sender = HandBuiltWormModel()
    generator = WormVariantGenerator(jitter_scale=0.3)

    # The base model's repellent margin is thin (~19% above threshold),
    # so some variants legitimately fail sanity — matching real biological
    # variability where not every individual responds identically.
    # Experiments handle this by skipping failed variants.
    n_seeds = 20
    passed = sum(
        generator.sanity_check(generator.generate_model_variant(sender, seed=seed))
        for seed in range(n_seeds)
    )
    assert passed >= 3, f"Only {passed}/{n_seeds} variants passed sanity at jitter_scale=0.3"
