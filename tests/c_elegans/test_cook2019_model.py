"""Tests for the Cook 2019 full connectome model."""

import pytest

from models.cook2019_connectome import Cook2019WormModel
from models.worm_variants import WormVariantGenerator


def test_model_loads_full_connectome() -> None:
    model = Cook2019WormModel()
    assert model.neuron_count == 302
    assert model.connection_count > 3000


def test_attractive_odor_drives_forward() -> None:
    model = Cook2019WormModel()
    result = model.simulate(1.0, "attractive")
    assert result["behavior"] == "forward"
    assert result["neurons"]["AIY"] > result["neurons"]["AIB"]


def test_repellent_odor_drives_reverse() -> None:
    model = Cook2019WormModel()
    result = model.simulate(1.0, "repellent")
    assert result["behavior"] == "reverse"
    assert result["neurons"]["AIB"] > 0.1


def test_zero_input_produces_pause() -> None:
    model = Cook2019WormModel()
    result = model.simulate(0.0, "attractive")
    assert result["behavior"] == "pause"
    assert all(v == 0.0 for v in result["neurons"].values())


def test_graded_response_increases_with_intensity() -> None:
    model = Cook2019WormModel()
    low = model.network_state(0.3, "attractive")
    high = model.network_state(1.5, "attractive")
    assert high["AIY"] > low["AIY"]
    assert high["AWA"] > low["AWA"]


def test_validate_odor_responses_passes() -> None:
    model = Cook2019WormModel()
    v = model.validate_odor_responses(1.0)
    assert v["passes"]


def test_receptor_activity_interface() -> None:
    model = Cook2019WormModel()
    ra = model.receptor_activity(1.0, "attractive")
    assert "AWA" in ra
    assert "AWC_L" in ra
    assert "AWC_R" in ra
    assert "AWB" not in ra


def test_simulate_from_receptors_roundtrip() -> None:
    model = Cook2019WormModel()
    receptors = model.receptor_activity(1.0, "attractive")
    result = model.simulate_from_receptors(receptors, "attractive")
    assert result["behavior"] == "forward"
    assert "neurons" in result


def test_variant_generation_sanity() -> None:
    sender = Cook2019WormModel()
    gen = WormVariantGenerator(jitter_scale=0.5)
    for seed in range(3):
        variant = gen.generate_model_variant(sender, seed=seed)
        assert gen.sanity_check(variant)
        assert isinstance(variant, Cook2019WormModel)


def test_variant_weights_differ_from_sender() -> None:
    sender = Cook2019WormModel()
    gen = WormVariantGenerator(jitter_scale=1.0)
    variant = gen.generate_model_variant(sender, seed=42)
    import numpy as np
    assert not np.allclose(sender._weight_matrix, variant._weight_matrix)
