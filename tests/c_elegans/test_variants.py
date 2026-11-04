from neural_transfer.models.c_elegans.hand_built import HandBuiltWormModel
from neural_transfer.models.c_elegans.variants import WormVariantGenerator


def test_worm_variant_generator_keeps_behavioral_invariants() -> None:
    base_model = HandBuiltWormModel()
    generator = WormVariantGenerator(jitter_scale=0.12)
    mutated_gain = generator.generate_variant(base_model.receptor_gain)

    mutated_model = HandBuiltWormModel(receptor_gain=mutated_gain)
    assert generator.sanity_check(mutated_model) is True


def test_worm_variant_generator_preserves_receptor_shape() -> None:
    base_gain = {"AWA": 1.0, "AWC_L": 0.9, "AWC_R": 0.8, "AWB": 0.7, "ADL": 0.6}
    mutated_gain = WormVariantGenerator(jitter_scale=0.08).generate_variant(base_gain)

    assert set(mutated_gain) == set(base_gain)
    assert all(value > 0.0 for value in mutated_gain.values())
