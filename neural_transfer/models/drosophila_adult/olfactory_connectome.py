"""Connectome-derived adult Drosophila olfactory model.

Weight matrices are derived from real synapse counts in the FlyWire v783
whole-brain connectome (Dorkenwald et al. 2024, Nature; annotations from
Schlegel et al. 2024).  Glomerulus-to-receptor mapping follows Couto et al.
2005, Hallem & Carlson 2006, and DoOR 2.0 (Münch & Galizia 2016).

Architecture (left hemisphere, from real synapse counts):
  1117 ORNs (49 glomeruli) → 49 PN channels (aggregated by glomerulus)
  → 2580 KCs (real PN→KC synapse weights, ~4 PNs per KC)
  → 48 MBONs (real KC→MBON synapse weights)
  → 2 output (approach/avoid, fitted)

All three connectivity matrices (ORN→PN, PN→KC, KC→MBON) use real synapse
counts from the connectome, not assumptions or statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from neural_transfer.models.drosophila_adult.olfactory_model import (
    ODOR_PROFILES,
    ODOR_VALENCE,
    ATTRACTIVE_ODORS,
    REPELLENT_ODORS,
    _pick_odor,
)

DATA_PATH = Path(__file__).resolve().parent / "olfactory_connectome.npz"

GLOM_TO_RECEPTOR = {
    'D': 'Or69a', 'DA1': 'Or67d', 'DA2': 'Or56a', 'DA3': 'Or23a',
    'DA4l': 'Or43a', 'DA4m': 'Or2a', 'DC1': 'Or19a', 'DC2': 'Or13a',
    'DC3': 'Or83c', 'DC4': 'Ir64a', 'DL1': 'Or10a', 'DL2d': 'Ir75a',
    'DL2v': 'Ir75b', 'DL3': 'Or65a', 'DL4': 'Or49a', 'DL5': 'Or7a',
    'DM1': 'Or42b', 'DM2': 'Or22a', 'DM3': 'Or47a', 'DM4': 'Or59b',
    'DM5': 'Or85a', 'DM6': 'Or67a', 'DP1l': 'Ir75d', 'DP1m': 'Ir64a_DP1m',
    'V': 'Or67b', 'VA1d': 'Or88a', 'VA1v': 'Or47b', 'VA2': 'Or92a',
    'VA3': 'Or67c', 'VA4': 'Or85b', 'VA5': 'Or49b', 'VA6': 'Or82a',
    'VA7l': 'Or46a', 'VA7m': 'Or59a', 'VC1': 'Or33c', 'VC2': 'Or71a',
    'VC4': 'Or67d_VC4', 'VC5': 'Ir41a', 'VL1': 'Ir75d_VL1',
    'VL2a': 'Ir84a', 'VL2p': 'Ir31a', 'VM1': 'Or42a', 'VM2': 'Or43b',
    'VM3': 'Or9a', 'VM4': 'Ir76a', 'VM5d': 'Or85f', 'VM5v': 'Or98a',
    'VM7d': 'Or42a_VM7d', 'VM7v': 'Or59c',
}


def _load_connectome() -> dict:
    data = np.load(DATA_PATH, allow_pickle=True)
    glom_names = list(data["glomerulus_names"])
    receptor_names = [GLOM_TO_RECEPTOR.get(g, g) for g in glom_names]
    return {
        "w_orn_pn": data["w_orn_pn"],
        "w_pn_kc": data["w_pn_kc"],
        "w_kc_mbon": data["w_kc_mbon"],
        "glomerulus_names": glom_names,
        "receptor_names": receptor_names,
        "n_glom": int(data["n_glom"]),
        "n_kc": int(data["n_kc"]),
        "n_mbon": int(data["n_mbon"]),
    }


_CACHE: Optional[dict] = None


def _get_connectome() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_connectome()
    return _CACHE


@dataclass
class ConnectomeAdultModel:
    """Rate-based model with FlyWire connectome-derived synapse weights.

    All weight matrices come from real synapse counts.  The ORN→PN matrix
    is aggregated by glomerulus (49 channels).  PN→KC and KC→MBON use
    individual neuron-to-neuron synapse counts.
    """

    pn_baseline: float = 5.0
    lateral_inhibition: float = 0.15
    kc_threshold: float = 0.3
    approach_threshold: float = 0.05

    orn_gain: Dict[str, float] = field(default_factory=dict)

    _glom_names: List[str] = field(default_factory=list)
    _receptor_names: List[str] = field(default_factory=list)
    _n_glom: int = 0
    _n_kc: int = 0
    _n_mbon: int = 0

    _w_orn_pn: Optional[np.ndarray] = field(default=None, repr=False)
    _w_pn_kc: Optional[np.ndarray] = field(default=None, repr=False)
    _w_kc_mbon: Optional[np.ndarray] = field(default=None, repr=False)
    _w_output: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        conn = _get_connectome()

        if not self._glom_names:
            self._glom_names = conn["glomerulus_names"]
        if not self._receptor_names:
            self._receptor_names = conn["receptor_names"]
        if self._n_glom == 0:
            self._n_glom = conn["n_glom"]
        if self._n_kc == 0:
            self._n_kc = conn["n_kc"]
        if self._n_mbon == 0:
            self._n_mbon = conn["n_mbon"]

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

        if not self.orn_gain:
            self.orn_gain = {r: 1.0 for r in self._receptor_names}

        if self._w_output is None:
            self._build_output()

    @property
    def n_glom(self) -> int:
        return self._n_glom

    @property
    def pn_names(self) -> List[str]:
        return [f"PN_{name}" for name in self._receptor_names]

    @property
    def kc_names(self) -> List[str]:
        return [f"KC_{i:04d}" for i in range(self._n_kc)]

    @property
    def output_names(self) -> List[str]:
        return ["MBON_approach", "MBON_avoid"]

    def _build_output(self) -> None:
        rng = np.random.default_rng(42)
        mbon_vecs, targets = [], []
        for odor in ODOR_PROFILES:
            pn = self._pn_rates_for_odor(odor, 1.0)
            kc = self._kc_rates(pn)
            mbon = self._w_kc_mbon @ kc
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
        self._w_output = w.T + rng.normal(0, 0.001, (2, self._n_mbon))

    def _orn_rates(self, odor_name: str, intensity: float) -> np.ndarray:
        profile = ODOR_PROFILES.get(odor_name, {})
        orn_input = np.zeros(self._n_glom)
        for i, receptor in enumerate(self._receptor_names):
            base_or = receptor.split("_")[0]
            base = profile.get(base_or, 0.0)
            gain = self.orn_gain.get(receptor, 1.0)
            orn_input[i] = max(0.0, gain * base * intensity)
        return orn_input

    def _pn_rates_for_odor(self, odor_name: str, intensity: float) -> np.ndarray:
        orn_input = self._orn_rates(odor_name, intensity)
        pn_raw = self._w_orn_pn @ orn_input
        global_mean = pn_raw.mean()
        pn = pn_raw - self.lateral_inhibition * global_mean + self.pn_baseline
        return np.maximum(pn, 0.0)

    def _kc_rates(self, pn_rates: np.ndarray) -> np.ndarray:
        baseline_pn = np.full(self._n_glom, self.pn_baseline)
        raw = self._w_pn_kc @ pn_rates
        baseline = self._w_pn_kc @ baseline_pn
        driven = np.maximum(raw - baseline, 0.0)
        peak = driven.max()
        if peak < 1e-9:
            return np.zeros(self._n_kc)
        normed = driven / peak
        return np.where(normed > self.kc_threshold, normed, 0.0)

    def _forward(self, odor_name: str, intensity: float) -> Dict[str, np.ndarray]:
        pn = self._pn_rates_for_odor(odor_name, intensity)
        kc = self._kc_rates(pn)
        mbon = self._w_kc_mbon @ kc
        out = self._w_output @ mbon
        return {"pn": pn, "kc": kc, "mbon": mbon, "output": out}

    def _forward_from_pn(self, pn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        kc = self._kc_rates(pn_rates)
        mbon = self._w_kc_mbon @ kc
        out = self._w_output @ mbon
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
        if "pn" in layers:
            for i, n in enumerate(self.pn_names):
                d[n] = float(layers["pn"][i])
        if "kc" in layers:
            for i, n in enumerate(self.kc_names):
                d[n] = float(layers["kc"][i])
        if "output" in layers:
            for i, n in enumerate(self.output_names):
                d[n] = float(layers["output"][i])
        return d

    def _dict_to_pn_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(pn, 0.0) for pn in self.pn_names])

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def receptor_activity(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        pn = self._pn_rates_for_odor(odor_name, odor_intensity)
        return {r: float(pn[i]) for i, r in enumerate(self._receptor_names)}

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
        pn_arr = np.array([receptor_signal.get(r, 0.0) for r in self._receptor_names])
        global_mean = pn_arr.mean()
        pn = pn_arr - self.lateral_inhibition * global_mean + self.pn_baseline
        pn = np.maximum(pn, 0.0)
        layers = self._forward_from_pn(pn)
        layers["pn"] = pn
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
        pn_arr = np.array([receptor_signal.get(r, 0.0) for r in self._receptor_names])
        global_mean = pn_arr.mean()
        pn = pn_arr - self.lateral_inhibition * global_mean + self.pn_baseline
        pn = np.maximum(pn, 0.0)
        layers = self._forward_from_pn(pn)
        return self._behavior_from_output(layers["output"])

    def pn_response(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        pn = self._pn_rates_for_odor(odor_name, odor_intensity)
        return {name: float(pn[i]) for i, name in enumerate(self.pn_names)}

    def validate_odor_responses(self) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        att = self.simulate(1.0, "attractive")
        checks["attractive_approach"] = att["behavior"] == "approach"
        rep = self.simulate(1.0, "repellent")
        checks["repellent_avoid"] = rep["behavior"] == "avoid"
        att_pn = self.receptor_activity(1.0, "attractive")
        checks["pn_responds"] = max(att_pn.values()) > 0
        return checks

    def clone_perturbed(
        self,
        orn_gain: Optional[Dict[str, float]] = None,
        orn_pn_weight_noise: Optional[np.ndarray] = None,
        kc_weight_noise: Optional[np.ndarray] = None,
        mbon_weight_noise: Optional[np.ndarray] = None,
    ) -> "ConnectomeAdultModel":
        new_gain = dict(orn_gain) if orn_gain else dict(self.orn_gain)

        clone = ConnectomeAdultModel(orn_gain=new_gain)
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
