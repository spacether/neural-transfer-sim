"""Tests for ReceptorToReceptorBaseline across both worm model types."""

import pytest

from models.worm_connectome import HandBuiltWormModel
from models.cook2019_connectome import Cook2019WormModel
from link.receptor_to_receptor import ReceptorToReceptorBaseline


ALL_MODELS = [
    pytest.param(HandBuiltWormModel, id="HandBuiltWormModel"),
    pytest.param(Cook2019WormModel, id="Cook2019WormModel"),
]


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_direct_broadcast_copies_sender_activity(ModelClass) -> None:
    sender = ModelClass()
    receiver = ModelClass()
    baseline = ReceptorToReceptorBaseline(sender=sender, receiver=receiver)

    sender_activity = sender.odor_response(1.2, "attractive")
    receiver_activity = baseline.copy_signal(sender_activity)

    assert receiver_activity == sender_activity


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_direct_broadcast_keeps_forward_behavior_on_attractive_odor(ModelClass) -> None:
    sender = ModelClass()
    receiver = ModelClass()
    baseline = ReceptorToReceptorBaseline(sender=sender, receiver=receiver)

    sender_activity = sender.odor_response(1.5, "attractive")
    receiver_activity = baseline.copy_signal(sender_activity)

    assert receiver.behavior_from_signal(receiver_activity) == "forward"


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_evaluate_transfer_attractive(ModelClass) -> None:
    sender = ModelClass()
    receiver = ModelClass()
    baseline = ReceptorToReceptorBaseline(sender=sender, receiver=receiver)

    trial = baseline.evaluate_transfer(1.0, "attractive")
    assert trial["success"] is True
    assert trial["sender_behavior"] == trial["receiver_behavior"]


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_evaluate_transfer_repellent(ModelClass) -> None:
    sender = ModelClass()
    receiver = ModelClass()
    baseline = ReceptorToReceptorBaseline(sender=sender, receiver=receiver)

    trial = baseline.evaluate_transfer(1.0, "repellent")
    assert trial["success"] is True
    assert trial["sender_behavior"] == trial["receiver_behavior"]
