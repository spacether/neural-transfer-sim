"""Coarse worm behavior sanity checks.

These tests validate only the first-order biological directionality that the
Stage 1 scaffold must satisfy before any transfer experiments are attempted.
They are intentionally not final quantitative validation against published
measurements.
"""

from models.worm_connectome import HandBuiltWormModel
from models.cook2019_connectome import Cook2019WormModel


def test_hand_built_model_attractive_odor_drives_forward_behavior() -> None:
    model = HandBuiltWormModel()
    result = model.simulate(odor_intensity=1.0, odor_type="attractive")

    assert result["behavior"] == "forward"
    assert result["neurons"]["AWA"] > 0.0
    assert result["neurons"]["AWC_L"] > 0.0


def test_cook2019_model_attractive_odor_drives_forward_behavior() -> None:
    model = Cook2019WormModel()
    result = model.simulate(odor_intensity=1.0, odor_type="attractive")

    assert result["behavior"] == "forward"
    assert result["neurons"]["AWA"] > 0.0
    assert result["neurons"]["AWC_L"] > 0.0


def test_hand_built_model_derives_behavior_from_explicit_network_state() -> None:
    model = HandBuiltWormModel()
    state = model.network_state(odor_intensity=1.0, odor_type="attractive")

    assert "AIY" in state
    assert state["AIY"] > 0.0
    assert model.behavior_from_network(state) == "forward"


def test_repellent_model_derives_behavior_from_explicit_network_state() -> None:
    model = HandBuiltWormModel()
    state = model.network_state(odor_intensity=1.0, odor_type="repellent")

    assert "AIB" in state
    assert state["AIB"] > 0.0
    assert model.behavior_from_network(state) == "reverse"
