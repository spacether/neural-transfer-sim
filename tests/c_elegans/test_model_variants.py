from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
from neural_transfer.models.c_elegans.cook2019 import Cook2019WormModel


def test_hand_built_worm_model_responds_to_attractive_odor() -> None:
    model = HandBuiltWormModel()
    result = model.simulate(odor_intensity=1.2, odor_type="attractive")

    assert result["behavior"] == "forward"
    assert result["neurons"]["AWA"] > 0.0
    assert result["neurons"]["AWC_L"] > 0.0


def test_cook2019_model_reports_realistic_interface() -> None:
    model = Cook2019WormModel()
    result = model.simulate(odor_intensity=1.0, odor_type="attractive")

    assert result["model_type"] == "cook2019_connectome"
    assert result["behavior"] == "forward"
    assert result["source"] == "cook_2019_302_neuron"
