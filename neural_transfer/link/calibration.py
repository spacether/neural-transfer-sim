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

    def transform_array(
        self, x: np.ndarray, col_names: tuple | list | None = None,
    ) -> np.ndarray:
        """Map a 1-D or 2-D array without dict overhead.

        If *col_names* matches self.neurons order, x is used directly.
        Otherwise columns are permuted to match the fitted neuron order,
        and the output is permuted back to *col_names* order.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform_array().")
        if col_names is None or tuple(col_names) == self.neurons:
            return x @ self._coef + self._intercept
        neuron_list = list(self.neurons)
        in_perm = [col_names.index(n) if n in col_names else -1 for n in neuron_list]
        n_neurons = len(neuron_list)
        if x.ndim == 1:
            xp = np.zeros(n_neurons)
            for i, pi in enumerate(in_perm):
                if pi >= 0:
                    xp[i] = x[pi]
            yp = xp @ self._coef + self._intercept
            out = np.zeros_like(x)
            for i, pi in enumerate(in_perm):
                if pi >= 0:
                    out[pi] = yp[i]
            return out
        # 2-D: build permutation indices once, apply with fancy indexing
        n_steps = x.shape[0]
        xp = np.zeros((n_steps, n_neurons))
        valid = [(i, pi) for i, pi in enumerate(in_perm) if pi >= 0]
        cal_idx = np.array([v[0] for v in valid])
        src_idx = np.array([v[1] for v in valid])
        xp[:, cal_idx] = x[:, src_idx]
        yp = xp @ self._coef + self._intercept
        out = np.zeros_like(x)
        out[:, src_idx] = yp[:, cal_idx]
        return out

    @property
    def is_fitted(self) -> bool:
        return self._fitted


@dataclass
class DiagonalCalibration:
    """Per-channel scaling + offset: N scale params + N offset params.

    Much more constrained than full ridge (2N vs N² parameters).
    Tests whether cross-channel interactions are needed for good transfer,
    or whether per-channel gain adjustment suffices.
    """

    neurons: Tuple[str, ...] | None = None
    alpha: float = 0.01

    _scale: np.ndarray | None = field(default=None, init=False, repr=False)
    _offset: np.ndarray | None = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _to_vec(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(n, 0.0) for n in self.neurons])

    def _from_vec(self, vec: np.ndarray) -> Dict[str, float]:
        return {n: float(vec[i]) for i, n in enumerate(self.neurons)}

    def fit(
        self,
        sender_signals: List[Dict[str, float]],
        receiver_signals: List[Dict[str, float]],
    ) -> "DiagonalCalibration":
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
        self._scale = np.ones(n)
        self._offset = np.zeros(n)

        for i in range(n):
            x_col = X[:, i]
            y_col = Y[:, i]
            var_x = np.var(x_col)
            if var_x > 1e-12:
                self._scale[i] = np.cov(x_col, y_col)[0, 1] / (var_x + self.alpha)
            self._offset[i] = np.mean(y_col) - self._scale[i] * np.mean(x_col)

        self._fitted = True
        return self

    def transform(self, sender_signal: Dict[str, float]) -> Dict[str, float]:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        x = self._to_vec(sender_signal)
        y = self._scale * x + self._offset
        return self._from_vec(y)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
