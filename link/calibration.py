"""Calibration transform: learn a linear sender-to-receiver signal mapping.

During calibration both worms are exposed to the same odors.  The sender's
observed receptor signal and the receiver's natural response form paired
training samples.  A ridge regression maps sender space to receiver space so
that injected signals better match what the receiver would have produced on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CalibrationTransform:
    """Ridge-regression linear map from sender signal space to receiver signal space."""

    neurons: Tuple[str, ...] | None = None
    alpha: float = 0.01

    _coef: np.ndarray | None = field(default=None, init=False, repr=False)
    _intercept: np.ndarray | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _to_vec(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(n, 0.0) for n in self.neurons])

    def _from_vec(self, vec: np.ndarray) -> Dict[str, float]:
        return {n: float(vec[i]) for i, n in enumerate(self.neurons)}

    def fit(
        self,
        sender_signals: List[Dict[str, float]],
        receiver_signals: List[Dict[str, float]],
    ) -> "CalibrationTransform":
        """Fit the linear transform from paired (sender, receiver) observations."""
        if self.neurons is None:
            keys: set = set()
            for s in sender_signals:
                keys.update(s.keys())
            for r in receiver_signals:
                keys.update(r.keys())
            self.neurons = tuple(sorted(keys))

        X = np.array([self._to_vec(s) for s in sender_signals])
        Y = np.array([self._to_vec(r) for r in receiver_signals])

        n = X.shape[1]
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        Xc = X - X_mean
        Yc = Y - Y_mean

        self._coef = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(n), Xc.T @ Yc)
        self._intercept = Y_mean - X_mean @ self._coef
        self._fitted = True
        return self

    def transform(self, sender_signal: Dict[str, float]) -> Dict[str, float]:
        """Map a sender signal to the predicted receiver signal."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        x = self._to_vec(sender_signal)
        y = x @ self._coef + self._intercept
        return self._from_vec(y)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
