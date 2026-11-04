"""Tests for signal injection and interface compatibility across all worm models."""

import pytest

from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
from neural_transfer.models.c_elegans.cook2019 import Cook2019WormModel


ALL_MODELS = [
    pytest.param(HandBuiltWormModel, id="HandBuiltWormModel"),
    pytest.param(Cook2019WormModel, id="Cook2019WormModel"),
]


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_odor_response_returns_receptor_neurons(ModelClass) -> None:
    model = ModelClass()
    sig = model.odor_response(1.0, "attractive")
    assert "AWA" in sig
    assert all(v >= 0 for v in sig.values())


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_behavior_from_signal_returns_valid_behavior(ModelClass) -> None:
    model = ModelClass()
    sig = model.odor_response(1.0, "attractive")
    behavior = model.behavior_from_signal(sig, "attractive")
    assert behavior in ("forward", "pause", "reverse")


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_simulate_from_receptors_attractive(ModelClass) -> None:
    model = ModelClass()
    sig = model.odor_response(1.0, "attractive")
    result = model.simulate_from_receptors(sig, "attractive")

    assert result["behavior"] == "forward"
    assert result["injection"] == "receptor_layer"
    assert "AVA" in result["neurons"]


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_simulate_from_receptors_matches_simulate(ModelClass) -> None:
    """Injecting the model's own receptor output should match a normal simulate."""
    model = ModelClass()
    normal = model.simulate(1.0, "attractive")
    sig = model.odor_response(1.0, "attractive")
    injected = model.simulate_from_receptors(sig, "attractive")

    assert normal["behavior"] == injected["behavior"]
    for neuron in injected["neurons"]:
        if neuron in normal["neurons"]:
            assert abs(normal["neurons"][neuron] - injected["neurons"][neuron]) < 1e-10


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_repellent_injection(ModelClass) -> None:
    model = ModelClass()
    sig = model.odor_response(1.0, "repellent")
    result = model.simulate_from_receptors(sig, "repellent")

    assert result["behavior"] == "reverse"


def test_behavior_from_signal_uses_receiver_weights() -> None:
    """behavior_from_signal on a different model should use THAT model's weights."""
    sender = HandBuiltWormModel()
    receiver = HandBuiltWormModel(
        weight_matrix={**sender.weight_matrix, "AIY->AVA": 0.2},
        command_threshold=1.5,
    )
    sig = sender.odor_response(0.5, "attractive")

    sender_behavior = sender.behavior_from_signal(sig, "attractive")
    receiver_behavior = receiver.behavior_from_signal(sig, "attractive")

    assert sender_behavior in ("forward", "pause")
    assert receiver_behavior in ("forward", "pause")
