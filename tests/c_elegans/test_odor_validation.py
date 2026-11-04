from models.worm_connectome import HandBuiltWormModel
from models.cook2019_connectome import Cook2019WormModel


def test_hand_built_model_validates_attractive_and_repellent_responses() -> None:
    model = HandBuiltWormModel()
    result = model.validate_odor_responses(odor_intensity=1.0)

    assert result["passes"] is True
    assert result["attractive"]["behavior"] == "forward"
    assert result["repellent"]["behavior"] == "reverse"


def test_cook2019_model_validates_attractive_and_repellent_responses() -> None:
    model = Cook2019WormModel()
    result = model.validate_odor_responses(odor_intensity=1.0)

    assert result["passes"] is True
    assert result["attractive"]["behavior"] == "forward"
    assert result["repellent"]["behavior"] == "reverse"
