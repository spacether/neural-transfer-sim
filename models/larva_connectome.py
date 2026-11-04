"""Connectome-based Drosophila larva olfactory model.

Weight matrices are derived from real synapse counts in the Winding et al.
2023 larval brain connectome (Science 379, eadd9330).  The left hemisphere's
olfactory pathway is extracted: 21 ORNs → 21 uniglomerular PNs → 73 KCs →
24 MBONs, with synapse counts as connection weights.

The same Kreher et al. odor response profiles drive the ORN input layer.
The public interface matches DrosophilaLarvaModel so existing experiments
run without change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from models.drosophila_larva import (
    ODOR_PROFILES,
    ODOR_VALENCE,
    ATTRACTIVE_ODORS,
    REPELLENT_ODORS,
    _pick_odor,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "larva_olfactory_connectome.npz"


def _load_connectome() -> dict:
    """Load the extracted connectome matrices."""
    data = np.load(DATA_PATH, allow_pickle=True)
    return {
        "w_orn_pn": data["w_orn_pn"],
        "w_pn_kc": data["w_pn_kc"],
        "w_kc_mbon": data["w_kc_mbon"],
        "orn_labels": list(data["orn_labels"]),
        "pn_labels": list(data["pn_labels"]),
        "mbon_types": list(data["mbon_types"]),
    }


_CONNECTOME_CACHE: Optional[dict] = None


def _get_connectome() -> dict:
    global _CONNECTOME_CACHE
    if _CONNECTOME_CACHE is None:
        _CONNECTOME_CACHE = _load_connectome()
    return _CONNECTOME_CACHE


ORN_TYPES: List[str] = [
    "Or1a", "Or7a", "Or13a", "Or22c", "Or24a", "Or30a", "Or33a",
    "Or35a", "Or42a", "Or42b", "Or45a", "Or45b", "Or47a", "Or49a",
    "Or59a", "Or63a", "Or67b", "Or74a", "Or82a", "Or83a", "Or85c",
    "Or94a",
]


@dataclass
class ConnectomeLarvaModel:
    """Rate-based model with real connectome-derived weights.

    Architecture: 21 ORN → 21 uPN → 73 KC → 24 MBON → 2 output (approach/avoid).
    The ORN→PN, PN→KC, and KC→MBON weight matrices come directly from
    synapse counts.  The final MBON→output mapping is fitted to known
    odor valences, as in the hand-built model.
    """

    orn_gain: Dict[str, float] = field(
        default_factory=lambda: {orn: 1.0 for orn in ORN_TYPES}
    )
    pn_baseline: float = 5.0
    lateral_inhibition: float = 0.15
    kc_threshold: float = 0.3
    approach_threshold: float = 0.05

    _w_orn_pn: Optional[np.ndarray] = field(default=None, repr=False)
    _w_pn_kc: Optional[np.ndarray] = field(default=None, repr=False)
    _w_kc_mbon: Optional[np.ndarray] = field(default=None, repr=False)
    _w_output: Optional[np.ndarray] = field(default=None, repr=False)

    _orn_order: List[str] = field(default_factory=list)
    _pn_order: List[str] = field(default_factory=list)
    _mbon_types: List[str] = field(default_factory=list)
    _n_kc: int = 0
    _n_mbon: int = 0

    def __post_init__(self) -> None:
        conn = _get_connectome()

        if self._w_orn_pn is None:
            raw = conn["w_orn_pn"].copy().astype(float)
            col_sums = raw.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            self._w_orn_pn = raw / col_sums

        if self._w_pn_kc is None:
            raw = conn["w_pn_kc"].copy().astype(float)
            row_sums = raw.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self._w_pn_kc = raw / row_sums

        if self._w_kc_mbon is None:
            self._w_kc_mbon = conn["w_kc_mbon"].copy().astype(float)

        if not self._orn_order:
            self._orn_order = list(conn["orn_labels"])
        if not self._pn_order:
            self._pn_order = list(conn["pn_labels"])
        if not self._mbon_types:
            self._mbon_types = list(conn["mbon_types"])

        self._n_kc = self._w_pn_kc.shape[0]
        self._n_mbon = self._w_kc_mbon.shape[0]

        if self._w_output is None:
            self._build_output()

        n_orn = len(self._orn_order)
        for orn in ORN_TYPES:
            if orn not in self.orn_gain:
                self.orn_gain[orn] = 1.0

    @property
    def pn_names(self) -> List[str]:
        return [f"PN_{name}" for name in self._pn_order]

    @property
    def kc_names(self) -> List[str]:
        return [f"KC_{i:02d}" for i in range(self._n_kc)]

    @property
    def mbon_names(self) -> List[str]:
        return self._mbon_types

    @property
    def output_names(self) -> List[str]:
        return ["MBON_approach", "MBON_avoid"]

    def _orn_index(self, orn_name: str) -> Optional[int]:
        try:
            return self._orn_order.index(orn_name)
        except ValueError:
            return None

    def _build_output(self) -> None:
        """Fit a 2-output (approach/avoid) layer from MBON activity on known odors."""
        rng = np.random.default_rng(42)

        mbon_vecs = []
        targets = []
        for odor in ODOR_PROFILES:
            orn = self._orn_rates(odor, 1.0)
            pn = self._pn_rates(orn)
            kc = self._kc_rates(pn)
            mbon = self._raw_mbon_rates(kc)
            mbon_vecs.append(mbon)
            if ODOR_VALENCE.get(odor, "attractive") == "attractive":
                targets.append([1.0, 0.0])
            else:
                targets.append([0.0, 1.0])

        if not mbon_vecs:
            self._w_output = rng.normal(0, 0.01, (2, self._n_mbon))
            return

        X = np.array(mbon_vecs)
        Y = np.array(targets)
        w, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        w = w.T
        w += rng.normal(0, 0.001, w.shape)
        self._w_output = w

    def _orn_rates(self, odor_name: str, intensity: float) -> np.ndarray:
        profile = ODOR_PROFILES.get(odor_name, {})
        rates = np.zeros(len(self._orn_order))
        for i, orn_name in enumerate(self._orn_order):
            base = profile.get(orn_name, 0.0)
            gain = self.orn_gain.get(orn_name, 1.0)
            rates[i] = max(0.0, gain * base * intensity)
        return rates

    def _pn_rates(self, orn_rates: np.ndarray) -> np.ndarray:
        raw = self._w_orn_pn @ orn_rates
        global_mean = raw.mean()
        pn = raw - self.lateral_inhibition * global_mean + self.pn_baseline
        return np.maximum(pn, 0.0)

    def _kc_rates(self, pn_rates: np.ndarray) -> np.ndarray:
        baseline_pn = np.full(len(self._pn_order), self.pn_baseline)
        raw = self._w_pn_kc @ pn_rates
        baseline = self._w_pn_kc @ baseline_pn
        driven = np.maximum(raw - baseline, 0.0)
        peak = driven.max()
        if peak < 1e-9:
            return np.zeros(self._n_kc)
        normed = driven / peak
        return np.where(normed > self.kc_threshold, normed, 0.0)

    def _raw_mbon_rates(self, kc_rates: np.ndarray) -> np.ndarray:
        return self._w_kc_mbon @ kc_rates

    def _output_rates(self, mbon_rates: np.ndarray) -> np.ndarray:
        return self._w_output @ mbon_rates

    def _forward(self, odor_name: str, intensity: float) -> Dict[str, np.ndarray]:
        orn = self._orn_rates(odor_name, intensity)
        pn = self._pn_rates(orn)
        kc = self._kc_rates(pn)
        mbon = self._raw_mbon_rates(kc)
        out = self._output_rates(mbon)
        return {"orn": orn, "pn": pn, "kc": kc, "mbon": mbon, "output": out}

    def _forward_from_orn(self, orn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        pn = self._pn_rates(orn_rates)
        kc = self._kc_rates(pn)
        mbon = self._raw_mbon_rates(kc)
        out = self._output_rates(mbon)
        return {"orn": orn_rates, "pn": pn, "kc": kc, "mbon": mbon, "output": out}

    def _forward_from_pn(self, pn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        kc = self._kc_rates(pn_rates)
        mbon = self._raw_mbon_rates(kc)
        out = self._output_rates(mbon)
        return {"pn": pn_rates, "kc": kc, "mbon": mbon, "output": out}

    def _behavior_from_output(self, out: np.ndarray) -> str:
        mag = abs(out[0]) + abs(out[1])
        if mag < self.approach_threshold:
            return "no_response"
        diff = out[0] - out[1]
        if diff > self.approach_threshold:
            return "approach"
        elif diff < -self.approach_threshold:
            return "avoid"
        return "no_response"

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        name_maps = [
            ("orn", self._orn_order),
            ("pn", self.pn_names),
            ("kc", self.kc_names),
            ("output", self.output_names),
        ]
        for key, names in name_maps:
            if key in layers:
                for i, n in enumerate(names):
                    d[n] = float(layers[key][i])
        return d

    def _dict_to_orn_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(orn, 0.0) for orn in self._orn_order])

    def _dict_to_pn_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(pn, 0.0) for pn in self.pn_names])

    # ------------------------------------------------------------------ #
    # Public interface (compatible with DrosophilaLarvaModel)
    # ------------------------------------------------------------------ #

    def receptor_activity(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        rates = self._orn_rates(odor_name, odor_intensity)
        return {orn: float(rates[i]) for i, orn in enumerate(self._orn_order)}

    def network_state(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        layers = self._forward(odor_name, odor_intensity)
        return self._layers_to_dict(layers)

    def simulate(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict:
        odor_name = _pick_odor(odor_type)
        layers = self._forward(odor_name, odor_intensity)
        behavior = self._behavior_from_output(layers["output"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_receptors(
        self, receptor_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        orn_arr = self._dict_to_orn_array(receptor_signal)
        layers = self._forward_from_orn(orn_arr)
        behavior = self._behavior_from_output(layers["output"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_pn(
        self, pn_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        pn_arr = self._dict_to_pn_array(pn_signal)
        layers = self._forward_from_pn(pn_arr)
        behavior = self._behavior_from_output(layers["output"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def odor_response(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        return self.receptor_activity(odor_intensity, odor_type)

    def behavior_from_signal(
        self, receptor_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> str:
        orn_arr = self._dict_to_orn_array(receptor_signal)
        layers = self._forward_from_orn(orn_arr)
        return self._behavior_from_output(layers["output"])

    def pn_response(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        layers = self._forward(odor_name, odor_intensity)
        return {pn: float(layers["pn"][i]) for i, pn in enumerate(self.pn_names)}

    def validate_odor_responses(self) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        att_result = self.simulate(1.0, "attractive")
        checks["attractive_approach"] = att_result["behavior"] == "approach"
        rep_result = self.simulate(1.0, "repellent")
        checks["repellent_avoid"] = rep_result["behavior"] == "avoid"
        att_orn = self.receptor_activity(1.0, "attractive")
        checks["orn_responds"] = max(att_orn.values()) > 0
        att_low = self.receptor_activity(0.3, "attractive")
        att_high = self.receptor_activity(1.5, "attractive")
        checks["graded_response"] = max(att_high.values()) > max(att_low.values())
        return checks

    def clone_perturbed(
        self,
        orn_gain: Optional[Dict[str, float]] = None,
        orn_pn_weight_noise: Optional[np.ndarray] = None,
        kc_weight_noise: Optional[np.ndarray] = None,
        mbon_weight_noise: Optional[np.ndarray] = None,
        kc_seed: Optional[int] = None,
    ) -> "ConnectomeLarvaModel":
        new_gain = dict(orn_gain) if orn_gain is not None else dict(self.orn_gain)

        clone = ConnectomeLarvaModel(orn_gain=new_gain)
        clone._w_orn_pn = self._w_orn_pn.copy()
        clone._w_pn_kc = self._w_pn_kc.copy()
        clone._w_kc_mbon = self._w_kc_mbon.copy()

        if orn_pn_weight_noise is not None:
            clone._w_orn_pn = np.maximum(clone._w_orn_pn + orn_pn_weight_noise, 0.0)
        if kc_weight_noise is not None:
            clone._w_pn_kc = np.maximum(clone._w_pn_kc + kc_weight_noise, 0.0)
        if mbon_weight_noise is not None:
            clone._w_kc_mbon = np.maximum(clone._w_kc_mbon + mbon_weight_noise, 0.0)

        clone._w_output = None
        clone._build_output()
        return clone
