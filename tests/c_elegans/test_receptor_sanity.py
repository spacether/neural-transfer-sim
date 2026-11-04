"""Coarse receptor-level sanity checks.

These tests verify that the worm receptor layer responds to odors in the
correct general direction, without claiming quantitative agreement with the
published calcium-imaging datasets.
"""

from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
from neural_transfer.models.c_elegans.cook2019 import Cook2019WormModel


def test_hand_built_model_repellent_odor_increases_aversive_signal() -> None:
    model = HandBuiltWormModel()
    result = model.simulate(odor_intensity=1.0, odor_type="repellent")

    assert result["behavior"] == "reverse"
    assert result["neurons"]["AWB"] > 0.0
    assert result["neurons"]["ADL"] > 0.0


def test_cook2019_model_repellent_odor_increases_aversive_signal() -> None:
    model = Cook2019WormModel()
    result = model.simulate(odor_intensity=1.0, odor_type="repellent")

    assert result["behavior"] == "reverse"
    assert result["neurons"]["AWB"] > 0.0
    assert result["neurons"]["ADL"] > 0.0
