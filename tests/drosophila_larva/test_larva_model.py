"""Tests for the Drosophila larva olfactory model."""

import numpy as np

from neural_transfer.models.drosophila_larva import (
    DrosophilaLarvaModel,
    ORN_TYPES,
    PN_NAMES,
    KC_NAMES,
    MBON_NAMES,
    ALL_NEURON_NAMES,
    ODOR_PROFILES,
    ODOR_VALENCE,
    ATTRACTIVE_ODORS,
    REPELLENT_ODORS,
)


def test_model_dimensions() -> None:
    assert len(ORN_TYPES) == 21
    assert len(PN_NAMES) == 21
    assert len(KC_NAMES) == 72
    assert len(MBON_NAMES) == 2
    assert len(ALL_NEURON_NAMES) == 116


def test_odor_profiles_cover_known_receptors() -> None:
    all_orns_in_profiles = set()
    for profile in ODOR_PROFILES.values():
        for orn in profile:
            all_orns_in_profiles.add(orn)
            assert orn in ORN_TYPES, f"{orn} not in ORN_TYPES"
    assert len(all_orns_in_profiles) == 21


def test_attractive_odors_produce_approach() -> None:
    m = DrosophilaLarvaModel()
    for odor in ATTRACTIVE_ODORS:
        layers = m._forward(odor, 1.0)
        beh = m._behavior_from_mbon(layers["mbon"])
        assert beh == "approach", f"{odor}: expected approach, got {beh}"


def test_repellent_odors_produce_avoidance() -> None:
    m = DrosophilaLarvaModel()
    for odor in REPELLENT_ODORS:
        layers = m._forward(odor, 1.0)
        beh = m._behavior_from_mbon(layers["mbon"])
        assert beh == "avoid", f"{odor}: expected avoid, got {beh}"


def test_graded_orn_response() -> None:
    m = DrosophilaLarvaModel()
    low = m._orn_rates("ethyl_acetate", 0.3)
    high = m._orn_rates("ethyl_acetate", 1.5)
    assert high.max() > low.max()


def test_network_state_keys() -> None:
    m = DrosophilaLarvaModel()
    state = m.network_state(1.0, "attractive")
    assert set(state.keys()) == set(ALL_NEURON_NAMES)


def test_receptor_activity_keys() -> None:
    m = DrosophilaLarvaModel()
    recep = m.receptor_activity(1.0, "attractive")
    assert set(recep.keys()) == set(ORN_TYPES)


def test_pn_response_keys() -> None:
    m = DrosophilaLarvaModel()
    pn = m.pn_response(1.0, "attractive")
    assert set(pn.keys()) == set(PN_NAMES)


def test_simulate_returns_behavior_and_neurons() -> None:
    m = DrosophilaLarvaModel()
    result = m.simulate(1.0, "attractive")
    assert "behavior" in result
    assert "neurons" in result
    assert result["behavior"] in {"approach", "avoid", "no_response"}
    assert len(result["neurons"]) == 116


def test_simulate_from_receptors() -> None:
    m = DrosophilaLarvaModel()
    recep = m.receptor_activity(1.0, "attractive")
    result = m.simulate_from_receptors(recep)
    assert result["behavior"] == "approach"


def test_simulate_from_pn() -> None:
    m = DrosophilaLarvaModel()
    pn = m.pn_response(1.0, "attractive")
    result = m.simulate_from_pn(pn)
    assert result["behavior"] == "approach"


def test_behavior_from_signal() -> None:
    m = DrosophilaLarvaModel()
    recep = m.receptor_activity(1.0, "attractive")
    beh = m.behavior_from_signal(recep)
    assert beh == "approach"


def test_validate_odor_responses() -> None:
    m = DrosophilaLarvaModel()
    checks = m.validate_odor_responses()
    assert all(checks.values()), f"Failed checks: {checks}"


def test_lateral_inhibition_reduces_baseline() -> None:
    m = DrosophilaLarvaModel()
    orn_rates = m._orn_rates("ethyl_acetate", 1.0)
    pn_rates = m._pn_rates(orn_rates)
    for i, orn in enumerate(ORN_TYPES):
        if orn_rates[i] == 0.0:
            assert pn_rates[i] < m.pn_baseline + 1.0


def test_kc_sparsity() -> None:
    m = DrosophilaLarvaModel()
    layers = m._forward("ethyl_acetate", 1.0)
    kc = layers["kc"]
    n_active = np.count_nonzero(kc)
    assert n_active < len(KC_NAMES), "KC layer should be sparse"
    assert n_active > 0, "At least some KCs should be active"


def test_zero_intensity_no_response() -> None:
    m = DrosophilaLarvaModel()
    result = m.simulate(0.0, "attractive")
    assert result["behavior"] == "no_response"
