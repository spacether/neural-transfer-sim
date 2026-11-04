"""Tests for the CalibrationTransform."""

import pytest

from neural_transfer.link.calibration import CalibrationTransform


def test_fit_and_transform_returns_all_neurons() -> None:
    ct = CalibrationTransform(neurons=("AWA", "AWC_L", "AWC_R"))
    sender = [{"AWA": 1.0, "AWC_L": 0.9}, {"AWA": 0.5, "AWC_L": 0.45}]
    receiver = [{"AWA": 0.95, "AWC_L": 0.87}, {"AWA": 0.48, "AWC_L": 0.44}]
    ct.fit(sender, receiver)

    result = ct.transform({"AWA": 0.75, "AWC_L": 0.67})
    assert set(result.keys()) == {"AWA", "AWC_L", "AWC_R"}
    assert ct.is_fitted


def test_identity_mapping_on_identical_signals() -> None:
    ct = CalibrationTransform(neurons=("AWA", "AWC_L"))
    signals = [{"AWA": float(i), "AWC_L": float(i) * 0.9} for i in range(1, 6)]
    ct.fit(signals, signals)

    test_sig = {"AWA": 3.5, "AWC_L": 3.15}
    result = ct.transform(test_sig)
    assert abs(result["AWA"] - 3.5) < 0.05
    assert abs(result["AWC_L"] - 3.15) < 0.05


def test_transform_before_fit_raises() -> None:
    ct = CalibrationTransform()
    with pytest.raises(RuntimeError, match="fit"):
        ct.transform({"AWA": 1.0})


def test_is_fitted_false_before_fit() -> None:
    assert not CalibrationTransform().is_fitted
