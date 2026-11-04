"""Drosophila larva olfactory subnetwork model for Stage 2.

Implements the larval olfactory pathway: 21 ORN types → 21 PNs (via
glomeruli in the larval antennal lobe) → ~72 KCs (mushroom body) → 2 MBONs.

ORN odor responses are derived from Kreher et al. 2005/2008 empty-neuron
recordings.  Circuit connectivity follows the architecture described in
Berck et al. 2016 and the Winding et al. 2023 larval connectome.

Unlike the worm, these are spiking neurons — signals are firing rates
(spikes/second) rather than graded voltages.  The model uses a rate-based
formulation for computational efficiency while maintaining the spiking
signal mode for the read/write hardware models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Neuron definitions
# --------------------------------------------------------------------------- #

ORN_TYPES: List[str] = [
    "Or1a", "Or7a", "Or13a", "Or22c", "Or24a", "Or30a", "Or33b",
    "Or35a", "Or42a", "Or42b", "Or45a", "Or45b", "Or47a", "Or49a",
    "Or59a", "Or63a", "Or67b", "Or74a", "Or82a", "Or83a", "Or85c",
]

PN_NAMES: List[str] = [f"PN_{orn}" for orn in ORN_TYPES]

N_KC = 72
KC_NAMES: List[str] = [f"KC_{i:02d}" for i in range(N_KC)]

MBON_NAMES: List[str] = ["MBON_approach", "MBON_avoid"]

ALL_NEURON_NAMES: List[str] = ORN_TYPES + PN_NAMES + KC_NAMES + MBON_NAMES

# Read/write site groupings
SITE_NEURONS = {
    "orn": ORN_TYPES,
    "pn": PN_NAMES,
    "kc": KC_NAMES,
    "mbon": MBON_NAMES,
}

# --------------------------------------------------------------------------- #
# Odor response profiles from Kreher et al. 2005/2008
# --------------------------------------------------------------------------- #
# Each entry maps ORN type → response magnitude (spikes/s above baseline).
# Values are representative of the published empty-neuron recordings,
# simplified to the resolution needed for transfer experiments.

ODOR_PROFILES: Dict[str, Dict[str, float]] = {
    "ethyl_acetate": {
        "Or42a": 95.0, "Or42b": 55.0, "Or59a": 30.0, "Or85c": 15.0,
        "Or7a": 10.0,
    },
    "pentyl_acetate": {
        "Or47a": 80.0, "Or22c": 45.0, "Or42a": 35.0, "Or85c": 20.0,
        "Or67b": 10.0,
    },
    "anisole": {
        "Or49a": 70.0, "Or67b": 40.0, "Or7a": 20.0, "Or33b": 15.0,
    },
    "1_octen_3_ol": {
        "Or13a": 75.0, "Or45b": 40.0, "Or35a": 25.0, "Or85c": 10.0,
    },
    "ethyl_butyrate": {
        "Or35a": 60.0, "Or22c": 50.0, "Or42b": 30.0, "Or85c": 25.0,
    },
    "methyl_salicylate": {
        "Or45a": 55.0, "Or49a": 35.0, "Or63a": 25.0, "Or74a": 20.0,
    },
    "3_octanol": {
        "Or22c": 65.0, "Or13a": 30.0, "Or82a": 20.0,
    },
    "E2_hexenal": {
        "Or35a": 50.0, "Or85c": 40.0, "Or45b": 30.0, "Or67b": 15.0,
    },
    "geranyl_acetate": {
        "Or82a": 45.0, "Or42b": 30.0, "Or24a": 25.0, "Or1a": 15.0,
    },
    "citronellal": {
        "Or82a": 70.0, "Or45a": 35.0, "Or85c": 30.0, "Or30a": 15.0,
    },
    "DEET": {
        "Or82a": 60.0, "Or49a": 40.0, "Or59a": 30.0, "Or83a": 25.0,
    },
    "benzaldehyde": {
        "Or7a": 50.0, "Or45a": 25.0, "Or33b": 20.0, "Or82a": 35.0,
    },
}

ODOR_VALENCE: Dict[str, str] = {
    "ethyl_acetate": "attractive",
    "pentyl_acetate": "attractive",
    "anisole": "attractive",
    "1_octen_3_ol": "attractive",
    "ethyl_butyrate": "attractive",
    "methyl_salicylate": "attractive",
    "3_octanol": "attractive",
    "E2_hexenal": "attractive",
    "geranyl_acetate": "attractive",
    "citronellal": "repellent",
    "DEET": "repellent",
    "benzaldehyde": "repellent",
}

ATTRACTIVE_ODORS = [k for k, v in ODOR_VALENCE.items() if v == "attractive"]
REPELLENT_ODORS = [k for k, v in ODOR_VALENCE.items() if v == "repellent"]


def _pick_odor(odor_type: str, rng: np.random.Generator | None = None) -> str:
    """Deterministically pick an odor from the given type."""
    if odor_type == "attractive":
        pool = ATTRACTIVE_ODORS
    elif odor_type == "repellent":
        pool = REPELLENT_ODORS
    else:
        raise ValueError(f"Unknown odor_type: {odor_type}")
    if rng is not None:
        return rng.choice(pool)
    return pool[0]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@dataclass
class DrosophilaLarvaModel:
    """Rate-based model of the larval olfactory pathway.

    Architecture: ORN → PN (with lateral inhibition) → KC (sparse) → MBON.
    Signal mode: firing rates (spikes/s).
    """

    # Tunable gains per ORN (perturbed in variants)
    orn_gain: Dict[str, float] = field(
        default_factory=lambda: {orn: 1.0 for orn in ORN_TYPES}
    )

    # PN layer: one-to-one from ORN, with lateral inhibition
    pn_baseline: float = 5.0
    lateral_inhibition: float = 0.15

    # KC layer: sparse random connectivity from PNs
    kc_threshold: float = 0.3
    kc_sparsity: int = 6  # each KC samples this many PNs
    _kc_weights: Optional[np.ndarray] = field(default=None, repr=False)
    _kc_seed: int = 42

    # MBON layer: approach MBON driven by KCs that receive from attractive-tuned PNs;
    # avoidance MBON by repellent-tuned PNs
    _mbon_weights: Optional[np.ndarray] = field(default=None, repr=False)

    # Behavior threshold
    approach_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self._kc_weights is None:
            self._build_kc_connectivity()
        if self._mbon_weights is None:
            self._build_mbon_connectivity()

    def _build_kc_connectivity(self) -> None:
        rng = np.random.default_rng(self._kc_seed)
        n_pn = len(PN_NAMES)
        w = np.zeros((N_KC, n_pn))
        for i in range(N_KC):
            chosen = rng.choice(n_pn, size=self.kc_sparsity, replace=False)
            for j in chosen:
                w[i, j] = rng.uniform(0.3, 1.0)
        self._kc_weights = w

    def _build_mbon_connectivity(self) -> None:
        """Fit MBON weights so the circuit produces correct approach/avoid.

        In the real larva, MBON weights are shaped by dopaminergic learning.
        We reproduce the outcome: for each known odor, compute KC activity
        and solve for MBON weights that assign higher approach-MBON output
        for attractive odors and higher avoid-MBON for repellent ones.
        """
        rng = np.random.default_rng(self._kc_seed + 1000)

        kc_vecs = []
        targets = []
        for odor, profile in ODOR_PROFILES.items():
            orn = self._orn_rates(odor, 1.0)
            pn = self._pn_rates(orn)
            kc = self._kc_rates(pn)
            kc_vecs.append(kc)
            if ODOR_VALENCE.get(odor, "attractive") == "attractive":
                targets.append([1.0, 0.0])
            else:
                targets.append([0.0, 1.0])

        X = np.array(kc_vecs)
        Y = np.array(targets)

        w, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        w = w.T
        w += rng.normal(0, 0.001, w.shape)
        self._mbon_weights = w

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def _orn_rates(self, odor_name: str, intensity: float) -> np.ndarray:
        """Compute ORN firing rates for a specific named odor."""
        profile = ODOR_PROFILES.get(odor_name, {})
        rates = np.zeros(len(ORN_TYPES))
        for i, orn in enumerate(ORN_TYPES):
            base_resp = profile.get(orn, 0.0)
            rates[i] = max(0.0, self.orn_gain[orn] * base_resp * intensity)
        return rates

    def _pn_rates(self, orn_rates: np.ndarray) -> np.ndarray:
        """PN rates: direct ORN input minus global lateral inhibition."""
        global_mean = orn_rates.mean()
        pn = orn_rates - self.lateral_inhibition * global_mean + self.pn_baseline
        return np.maximum(pn, 0.0)

    def _kc_rates(self, pn_rates: np.ndarray) -> np.ndarray:
        """KC rates: sparse random projection with baseline subtraction and threshold."""
        baseline_pn = np.full(len(PN_NAMES), self.pn_baseline)
        raw = self._kc_weights @ pn_rates
        baseline = self._kc_weights @ baseline_pn
        driven = np.maximum(raw - baseline, 0.0)
        peak = driven.max()
        if peak < 1e-9:
            return np.zeros(N_KC)
        normed = driven / peak
        kc = np.where(normed > self.kc_threshold, normed, 0.0)
        return kc

    def _mbon_rates(self, kc_rates: np.ndarray) -> np.ndarray:
        """MBON rates: weighted sum from KCs."""
        return self._mbon_weights @ kc_rates

    def _forward(self, odor_name: str, intensity: float) -> Dict[str, np.ndarray]:
        """Run the full pathway, returning rates at each layer."""
        orn = self._orn_rates(odor_name, intensity)
        pn = self._pn_rates(orn)
        kc = self._kc_rates(pn)
        mbon = self._mbon_rates(kc)
        return {"orn": orn, "pn": pn, "kc": kc, "mbon": mbon}

    def _forward_from_orn(self, orn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        """Run from ORN rates forward (for signal injection)."""
        pn = self._pn_rates(orn_rates)
        kc = self._kc_rates(pn)
        mbon = self._mbon_rates(kc)
        return {"orn": orn_rates, "pn": pn, "kc": kc, "mbon": mbon}

    def _forward_from_pn(self, pn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        """Run from PN rates forward (writing at the glomerular bottleneck)."""
        kc = self._kc_rates(pn_rates)
        mbon = self._mbon_rates(kc)
        return {"pn": pn_rates, "kc": kc, "mbon": mbon}

    # ------------------------------------------------------------------ #
    # Dict conversions
    # ------------------------------------------------------------------ #

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Flatten layer arrays into a single {neuron_name: rate} dict."""
        d: Dict[str, float] = {}
        if "orn" in layers:
            for i, n in enumerate(ORN_TYPES):
                d[n] = float(layers["orn"][i])
        if "pn" in layers:
            for i, n in enumerate(PN_NAMES):
                d[n] = float(layers["pn"][i])
        if "kc" in layers:
            for i, n in enumerate(KC_NAMES):
                d[n] = float(layers["kc"][i])
        if "mbon" in layers:
            for i, n in enumerate(MBON_NAMES):
                d[n] = float(layers["mbon"][i])
        return d

    def _dict_to_orn_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(orn, 0.0) for orn in ORN_TYPES])

    def _dict_to_pn_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(pn, 0.0) for pn in PN_NAMES])

    # ------------------------------------------------------------------ #
    # Behavior
    # ------------------------------------------------------------------ #

    def _behavior_from_mbon(self, mbon: np.ndarray) -> str:
        mag = abs(mbon[0]) + abs(mbon[1])
        if mag < self.approach_threshold:
            return "no_response"
        diff = mbon[0] - mbon[1]
        if diff > self.approach_threshold:
            return "approach"
        elif diff < -self.approach_threshold:
            return "avoid"
        return "no_response"

    # ------------------------------------------------------------------ #
    # Public interface (matches HandBuiltWormModel / Cook2019WormModel)
    # ------------------------------------------------------------------ #

    def receptor_activity(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        rates = self._orn_rates(odor_name, odor_intensity)
        return {orn: float(rates[i]) for i, orn in enumerate(ORN_TYPES)}

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
        behavior = self._behavior_from_mbon(layers["mbon"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_receptors(
        self, receptor_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        """Inject a signal at the ORN level and propagate forward."""
        orn_arr = self._dict_to_orn_array(receptor_signal)
        layers = self._forward_from_orn(orn_arr)
        behavior = self._behavior_from_mbon(layers["mbon"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_pn(
        self, pn_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        """Inject at the PN (glomerular) level — the bottleneck write target."""
        pn_arr = self._dict_to_pn_array(pn_signal)
        layers = self._forward_from_pn(pn_arr)
        behavior = self._behavior_from_mbon(layers["mbon"])
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
        return self._behavior_from_mbon(layers["mbon"])

    def pn_response(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        """Return PN-level activity — the glomerular bottleneck signal."""
        odor_name = _pick_odor(odor_type)
        layers = self._forward(odor_name, odor_intensity)
        return {pn: float(layers["pn"][i]) for i, pn in enumerate(PN_NAMES)}

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

    # ------------------------------------------------------------------ #
    # Variant support
    # ------------------------------------------------------------------ #

    def clone_perturbed(
        self,
        orn_gain: Optional[Dict[str, float]] = None,
        kc_weight_noise: Optional[np.ndarray] = None,
        mbon_weight_noise: Optional[np.ndarray] = None,
        kc_seed: Optional[int] = None,
    ) -> "DrosophilaLarvaModel":
        new_gain = dict(orn_gain) if orn_gain is not None else dict(self.orn_gain)
        new_kc_seed = kc_seed if kc_seed is not None else self._kc_seed

        clone = DrosophilaLarvaModel(
            orn_gain=new_gain,
            pn_baseline=self.pn_baseline,
            lateral_inhibition=self.lateral_inhibition,
            kc_threshold=self.kc_threshold,
            kc_sparsity=self.kc_sparsity,
            _kc_seed=new_kc_seed,
            approach_threshold=self.approach_threshold,
        )

        if kc_weight_noise is not None:
            clone._kc_weights = np.maximum(clone._kc_weights + kc_weight_noise, 0.0)
        if mbon_weight_noise is not None:
            clone._mbon_weights = np.maximum(clone._mbon_weights + mbon_weight_noise, 0.0)
            row_sums = clone._mbon_weights.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            clone._mbon_weights /= row_sums

        return clone
