"""Tests for the Drosophila larva variant generator."""

from neural_transfer.models.drosophila_larva import DrosophilaLarvaModel
from neural_transfer.models.drosophila_larva.variants import LarvaVariantGenerator


def test_variant_passes_sanity() -> None:
    base = DrosophilaLarvaModel()
    gen = LarvaVariantGenerator(jitter_scale=0.5)
    passed = sum(
        gen.sanity_check(gen.generate_variant(base, seed=seed))
        for seed in range(10)
    )
    assert passed >= 7, f"Only {passed}/10 variants passed sanity"


def test_variant_differs_from_base() -> None:
    base = DrosophilaLarvaModel()
    gen = LarvaVariantGenerator(jitter_scale=1.0)
    variant = gen.generate_variant(base, seed=0)
    assert variant.orn_gain != base.orn_gain


def test_zero_jitter_preserves_behavior() -> None:
    base = DrosophilaLarvaModel()
    gen = LarvaVariantGenerator(jitter_scale=0.0)
    variant = gen.generate_variant(base, seed=0, vary_kc=False, vary_mbon=False)
    assert gen.sanity_check(variant)


def test_variant_with_kc_rewiring() -> None:
    base = DrosophilaLarvaModel()
    gen = LarvaVariantGenerator(jitter_scale=1.0)
    variant = gen.generate_variant(base, seed=42, vary_mbon=False)
    assert gen.sanity_check(variant)
    assert variant._kc_seed != base._kc_seed


def test_broken_variant_fails_sanity() -> None:
    base = DrosophilaLarvaModel()
    broken = base.clone_perturbed(orn_gain={orn: 0.0 for orn in base.orn_gain})
    gen = LarvaVariantGenerator()
    assert not gen.sanity_check(broken)
