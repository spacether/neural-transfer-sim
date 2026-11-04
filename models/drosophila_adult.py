"""Adult Drosophila olfactory model for Stage 3.

Implements the adult fly olfactory pathway:
  ~2,600 ORNs (50 types × ~50 neurons each) → 50 PNs (via glomeruli in
  the antennal lobe) → 2,000 KCs (mushroom body) → MBONs.

ORN odor responses for 24 types are derived from Hallem & Carlson 2006
(firing rate changes measured in the empty-neuron system).  An additional
26 types use response profiles estimated from the DoOR database, giving
~50 receptor types total — matching the adult Drosophila repertoire.

Key differences from the larval model (Stage 2):
  - Convergence: ~50 ORNs of each type feed one glomerulus, so PN input
    is an average over a noisy population rather than one neuron's output.
  - Expansion: 50 PNs project to 2,000 KCs (vs 21 → 72 in larva).
  - Structural variability: PN→KC wiring differs between individuals even
    when PN→glomerulus wiring is conserved.  This is the strongest test
    of the claim that bottlenecks are better write targets.
  - Receiver learning: KC→MBON weights can adapt to an imperfect incoming
    code, testing how much learning substitutes for hardware precision.

Signal mode: firing rates (spikes/s), same as the larval model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Neuron definitions
# --------------------------------------------------------------------------- #

# 24 ORN types from Hallem & Carlson 2006 + 26 from DoOR
HALLEM_CARLSON_TYPES: List[str] = [
    "Or2a", "Or7a", "Or9a", "Or10a", "Or19a", "Or22a", "Or23a",
    "Or33b", "Or35a", "Or43a", "Or43b", "Or47a", "Or47b", "Or49b",
    "Or56a", "Or59b", "Or67a", "Or67c", "Or82a", "Or85a", "Or85b",
    "Or85f", "Or92a", "Or98a",
]

DOOR_EXTENDED_TYPES: List[str] = [
    "Or1a", "Or13a", "Or22c", "Or24a", "Or30a", "Or33a", "Or33c",
    "Or42a", "Or42b", "Or45a", "Or45b", "Or46a", "Or49a", "Or59a",
    "Or63a", "Or65a", "Or67b", "Or67d", "Or69a", "Or71a", "Or74a",
    "Or83a", "Or83b", "Or83c", "Or85c", "Or88a",
]

ORN_TYPES: List[str] = sorted(HALLEM_CARLSON_TYPES + DOOR_EXTENDED_TYPES)
N_ORN_TYPES = len(ORN_TYPES)

ORNS_PER_TYPE = 50
N_TOTAL_ORNS = N_ORN_TYPES * ORNS_PER_TYPE

PN_NAMES: List[str] = [f"PN_{orn}" for orn in ORN_TYPES]
N_PN = len(PN_NAMES)

N_KC = 2000
KC_NAMES: List[str] = [f"KC_{i:04d}" for i in range(N_KC)]

MBON_NAMES: List[str] = ["MBON_approach", "MBON_avoid"]

ALL_NEURON_NAMES: List[str] = ORN_TYPES + PN_NAMES + KC_NAMES + MBON_NAMES

SITE_NEURONS = {
    "orn": ORN_TYPES,
    "pn": PN_NAMES,
    "kc": KC_NAMES,
    "mbon": MBON_NAMES,
}

# --------------------------------------------------------------------------- #
# Odor response profiles
# --------------------------------------------------------------------------- #
# Hallem & Carlson 2006: firing rate changes (Δ spikes/s) for 24 ORN types
# across 110 odors.  We include a representative subset of ~30 odors covering
# a range of chemical classes and valences.
#
# DoOR-extended types: estimated peak responses from the Door database,
# calibrated to the Hallem & Carlson scale.

ODOR_PROFILES: Dict[str, Dict[str, float]] = {
    # --- Esters (generally attractive) ---
    "ethyl_acetate": {
        "Or42b": 165.0, "Or42a": 60.0, "Or59b": 55.0, "Or85a": 30.0,
        "Or9a": 25.0, "Or47a": 20.0, "Or22a": 15.0,
        "Or45a": 18.0, "Or85c": 12.0,
    },
    "pentyl_acetate": {
        "Or47a": 130.0, "Or22a": 85.0, "Or42b": 45.0, "Or85b": 35.0,
        "Or59b": 30.0, "Or67a": 20.0, "Or49b": 15.0,
        "Or22c": 25.0, "Or67b": 12.0,
    },
    "methyl_salicylate": {
        "Or10a": 95.0, "Or49b": 55.0, "Or85a": 40.0, "Or67a": 35.0,
        "Or43b": 20.0, "Or63a": 28.0, "Or74a": 15.0,
    },
    "ethyl_butyrate": {
        "Or85a": 110.0, "Or35a": 70.0, "Or22a": 55.0, "Or59b": 40.0,
        "Or42b": 30.0, "Or9a": 20.0, "Or85c": 22.0,
    },
    "isoamyl_acetate": {
        "Or47a": 145.0, "Or22a": 75.0, "Or42b": 50.0, "Or85b": 40.0,
        "Or67a": 25.0, "Or49b": 18.0, "Or22c": 30.0,
    },
    "geranyl_acetate": {
        "Or82a": 120.0, "Or56a": 45.0, "Or47b": 35.0, "Or19a": 25.0,
        "Or24a": 20.0, "Or42b": 18.0, "Or1a": 15.0,
    },
    "methyl_butyrate": {
        "Or85a": 100.0, "Or22a": 65.0, "Or35a": 50.0, "Or59b": 35.0,
        "Or42b": 25.0, "Or85c": 18.0,
    },
    "ethyl_propionate": {
        "Or42b": 90.0, "Or59b": 60.0, "Or85a": 45.0, "Or35a": 30.0,
        "Or9a": 22.0, "Or22a": 18.0,
    },
    "butyl_acetate": {
        "Or47a": 115.0, "Or22a": 70.0, "Or42b": 40.0, "Or85b": 30.0,
        "Or59b": 25.0, "Or67a": 18.0,
    },
    "propyl_acetate": {
        "Or42b": 100.0, "Or47a": 55.0, "Or59b": 45.0, "Or22a": 35.0,
        "Or85a": 25.0, "Or9a": 15.0,
    },
    # --- Alcohols (mixed valence) ---
    "1_octen_3_ol": {
        "Or13a": 110.0, "Or35a": 50.0, "Or45b": 45.0, "Or85a": 30.0,
        "Or67c": 25.0, "Or71a": 20.0, "Or85c": 15.0,
    },
    "3_octanol": {
        "Or22a": 80.0, "Or85a": 55.0, "Or13a": 40.0, "Or35a": 30.0,
        "Or82a": 20.0, "Or71a": 15.0,
    },
    "1_hexanol": {
        "Or35a": 75.0, "Or85a": 60.0, "Or67c": 35.0, "Or13a": 25.0,
        "Or59b": 20.0, "Or71a": 18.0,
    },
    "2_heptanone": {
        "Or85b": 90.0, "Or43b": 50.0, "Or35a": 40.0, "Or22a": 30.0,
        "Or67c": 22.0, "Or45b": 15.0,
    },
    # --- Aldehydes ---
    "E2_hexenal": {
        "Or35a": 95.0, "Or67c": 60.0, "Or85a": 40.0, "Or45b": 35.0,
        "Or56a": 25.0, "Or85c": 30.0, "Or67b": 18.0,
    },
    "benzaldehyde": {
        "Or7a": 85.0, "Or10a": 50.0, "Or82a": 40.0, "Or2a": 30.0,
        "Or45a": 28.0, "Or33b": 20.0, "Or49a": 15.0,
    },
    "octanal": {
        "Or23a": 75.0, "Or7a": 45.0, "Or35a": 35.0, "Or85a": 25.0,
        "Or67c": 20.0, "Or71a": 15.0,
    },
    # --- Terpenes ---
    "citronellal": {
        "Or82a": 100.0, "Or56a": 55.0, "Or85a": 35.0, "Or45a": 30.0,
        "Or47b": 20.0, "Or30a": 15.0, "Or85c": 25.0,
    },
    "linalool": {
        "Or69a": 90.0, "Or56a": 50.0, "Or19a": 40.0, "Or82a": 30.0,
        "Or47b": 22.0, "Or30a": 18.0,
    },
    "geraniol": {
        "Or56a": 80.0, "Or82a": 60.0, "Or19a": 35.0, "Or69a": 30.0,
        "Or47b": 25.0, "Or1a": 15.0,
    },
    # --- Aromatics ---
    "anisole": {
        "Or49b": 70.0, "Or67a": 45.0, "Or10a": 35.0, "Or49a": 30.0,
        "Or33b": 20.0, "Or7a": 15.0,
    },
    "4_methylphenol": {
        "Or2a": 85.0, "Or49b": 55.0, "Or71a": 40.0, "Or67a": 30.0,
        "Or65a": 25.0, "Or10a": 15.0,
    },
    "acetophenone": {
        "Or10a": 75.0, "Or42a": 45.0, "Or49b": 35.0, "Or67a": 25.0,
        "Or2a": 20.0, "Or65a": 18.0,
    },
    # --- Repellents ---
    "DEET": {
        "Or82a": 75.0, "Or49a": 50.0, "Or59a": 40.0, "Or83a": 35.0,
        "Or33c": 25.0, "Or56a": 20.0,
    },
    "citronella_oil": {
        "Or82a": 90.0, "Or56a": 55.0, "Or7a": 30.0, "Or83a": 25.0,
        "Or69a": 20.0, "Or30a": 15.0,
    },
    "benzoquinone": {
        "Or83b": 80.0, "Or83c": 45.0, "Or46a": 40.0, "Or33a": 30.0,
        "Or2a": 25.0, "Or65a": 20.0,
    },
    "acetic_acid": {
        "Or42a": 70.0, "Or33a": 55.0, "Or46a": 35.0, "Or83b": 30.0,
        "Or59a": 25.0, "Or43a": 20.0,
    },
    "CO2": {
        "Or56a": 60.0, "Or82a": 40.0, "Or33c": 35.0,
    },
    # --- Additional repellents for class balance ---
    "propionic_acid": {
        "Or59a": 65.0, "Or33a": 50.0, "Or42a": 35.0, "Or46a": 30.0,
        "Or83b": 20.0,
    },
    "butyric_acid": {
        "Or59a": 70.0, "Or33a": 45.0, "Or83b": 35.0, "Or46a": 25.0,
        "Or43a": 18.0,
    },
    "geosmin": {
        "Or56a": 95.0, "Or83a": 40.0, "Or33c": 30.0, "Or82a": 20.0,
    },
    "hydrogen_peroxide": {
        "Or42a": 55.0, "Or83b": 45.0, "Or33a": 30.0, "Or59a": 25.0,
        "Or46a": 20.0,
    },
    "methyl_hexanoate_high": {
        "Or85b": 70.0, "Or22a": 50.0, "Or67c": 40.0, "Or43b": 30.0,
        "Or35a": 20.0,
    },
    "phenylacetaldehyde": {
        "Or2a": 80.0, "Or65a": 50.0, "Or49b": 40.0, "Or67d": 30.0,
        "Or71a": 20.0,
    },
    "trimethylamine": {
        "Or46a": 85.0, "Or83a": 55.0, "Or33c": 40.0, "Or59a": 30.0,
    },
    "sulcatone": {
        "Or85b": 60.0, "Or67c": 45.0, "Or43b": 35.0, "Or10a": 25.0,
        "Or56a": 15.0,
    },
    # --- Additional odors for calibration diversity ---
    "2_butanone": {
        "Or92a": 85.0, "Or42b": 40.0, "Or59b": 35.0, "Or9a": 25.0,
        "Or85a": 20.0,
    },
    "gamma_butyrolactone": {
        "Or67d": 90.0, "Or88a": 45.0, "Or49a": 30.0, "Or92a": 25.0,
    },
}

ODOR_VALENCE: Dict[str, str] = {
    "ethyl_acetate": "attractive",
    "pentyl_acetate": "attractive",
    "methyl_salicylate": "attractive",
    "ethyl_butyrate": "attractive",
    "isoamyl_acetate": "attractive",
    "geranyl_acetate": "attractive",
    "methyl_butyrate": "attractive",
    "ethyl_propionate": "attractive",
    "butyl_acetate": "attractive",
    "propyl_acetate": "attractive",
    "1_octen_3_ol": "attractive",
    "3_octanol": "attractive",
    "1_hexanol": "attractive",
    "2_heptanone": "attractive",
    "E2_hexenal": "attractive",
    "benzaldehyde": "attractive",
    "octanal": "attractive",
    "citronellal": "repellent",
    "linalool": "attractive",
    "geraniol": "attractive",
    "anisole": "attractive",
    "4_methylphenol": "repellent",
    "acetophenone": "attractive",
    "DEET": "repellent",
    "citronella_oil": "repellent",
    "benzoquinone": "repellent",
    "acetic_acid": "repellent",
    "CO2": "repellent",
    "propionic_acid": "repellent",
    "butyric_acid": "repellent",
    "geosmin": "repellent",
    "hydrogen_peroxide": "repellent",
    "methyl_hexanoate_high": "repellent",
    "phenylacetaldehyde": "repellent",
    "trimethylamine": "repellent",
    "sulcatone": "repellent",
    "2_butanone": "attractive",
    "gamma_butyrolactone": "attractive",
}

ATTRACTIVE_ODORS = [k for k, v in ODOR_VALENCE.items() if v == "attractive"]
REPELLENT_ODORS = [k for k, v in ODOR_VALENCE.items() if v == "repellent"]


def _pick_odor(odor_type: str, rng: np.random.Generator | None = None) -> str:
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
class DrosophilaAdultModel:
    """Rate-based model of the adult Drosophila olfactory pathway.

    Architecture: ORNs (50 per type) → convergence → PN (with lateral
    inhibition via LNs) → KC (sparse random projection) → MBON.

    The convergence step averages the noisy responses of ~50 ORNs per type
    into a single glomerular/PN input — this is the key bottleneck.
    """

    orn_gain: Dict[str, float] = field(
        default_factory=lambda: {orn: 1.0 for orn in ORN_TYPES}
    )
    orn_noise_cv: float = 0.20

    pn_baseline: float = 8.0
    lateral_inhibition: float = 0.12

    kc_threshold: float = 0.3
    kc_sparsity: int = 7
    _kc_weights: Optional[np.ndarray] = field(default=None, repr=False)
    _kc_seed: int = 42

    _mbon_weights: Optional[np.ndarray] = field(default=None, repr=False)

    approach_threshold: float = 0.05
    _orn_convergence_seed: int = 99

    def __post_init__(self) -> None:
        if self._kc_weights is None:
            self._build_kc_connectivity()
        if self._mbon_weights is None:
            self._build_mbon_connectivity()

    def _build_kc_connectivity(self) -> None:
        """Random sparse PN→KC projection (~7 PNs per KC)."""
        rng = np.random.default_rng(self._kc_seed)
        w = np.zeros((N_KC, N_PN))
        for i in range(N_KC):
            chosen = rng.choice(N_PN, size=self.kc_sparsity, replace=False)
            for j in chosen:
                w[i, j] = rng.uniform(0.3, 1.0)
        self._kc_weights = w

    def _build_mbon_connectivity(self) -> None:
        """Fit MBON weights via least-squares on known odor valences."""
        rng = np.random.default_rng(self._kc_seed + 1000)

        kc_vecs = []
        targets = []
        for odor, profile in ODOR_PROFILES.items():
            orn = self._orn_type_rates(odor, 1.0)
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

    def _orn_type_rates(self, odor_name: str, intensity: float) -> np.ndarray:
        """Mean ORN firing rate per type (after convergence averaging).

        Each ORN type has ~50 individual neurons.  Their mean response is
        the type's profile value scaled by gain and intensity.  The
        convergence averaging reduces noise by sqrt(ORNS_PER_TYPE).
        """
        profile = ODOR_PROFILES.get(odor_name, {})
        rates = np.zeros(N_ORN_TYPES)
        for i, orn in enumerate(ORN_TYPES):
            base = profile.get(orn, 0.0)
            rates[i] = max(0.0, self.orn_gain[orn] * base * intensity)
        return rates

    def _orn_type_rates_noisy(self, odor_name: str, intensity: float,
                               rng: np.random.Generator | None = None) -> np.ndarray:
        """ORN type rates with convergence noise.

        Simulates the effect of averaging ORNS_PER_TYPE noisy neurons:
        the noise on the mean is CV / sqrt(N).
        """
        rates = self._orn_type_rates(odor_name, intensity)
        if rng is None or self.orn_noise_cv <= 0:
            return rates
        convergence_cv = self.orn_noise_cv / np.sqrt(ORNS_PER_TYPE)
        noise = rng.normal(1.0, convergence_cv, size=rates.shape)
        return np.maximum(rates * noise, 0.0)

    def _pn_rates(self, orn_type_rates: np.ndarray) -> np.ndarray:
        """PN rates: ORN input with lateral inhibition from LNs."""
        global_mean = orn_type_rates.mean()
        pn = orn_type_rates - self.lateral_inhibition * global_mean + self.pn_baseline
        return np.maximum(pn, 0.0)

    def _kc_rates(self, pn_rates: np.ndarray) -> np.ndarray:
        """KC rates: sparse random projection with threshold and normalization."""
        baseline_pn = np.full(N_PN, self.pn_baseline)
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
        return self._mbon_weights @ kc_rates

    def _forward(self, odor_name: str, intensity: float) -> Dict[str, np.ndarray]:
        orn = self._orn_type_rates(odor_name, intensity)
        pn = self._pn_rates(orn)
        kc = self._kc_rates(pn)
        mbon = self._mbon_rates(kc)
        return {"orn": orn, "pn": pn, "kc": kc, "mbon": mbon}

    def _forward_from_orn(self, orn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        pn = self._pn_rates(orn_rates)
        kc = self._kc_rates(pn)
        mbon = self._mbon_rates(kc)
        return {"orn": orn_rates, "pn": pn, "kc": kc, "mbon": mbon}

    def _forward_from_pn(self, pn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        kc = self._kc_rates(pn_rates)
        mbon = self._mbon_rates(kc)
        return {"pn": pn_rates, "kc": kc, "mbon": mbon}

    def _forward_from_kc(self, kc_rates: np.ndarray) -> Dict[str, np.ndarray]:
        """Write directly at KC level — testing the expansion layer."""
        mbon = self._mbon_rates(kc_rates)
        return {"kc": kc_rates, "mbon": mbon}

    # ------------------------------------------------------------------ #
    # Dict conversions
    # ------------------------------------------------------------------ #

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
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

    def _dict_to_kc_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(kc, 0.0) for kc in KC_NAMES])

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
    # Public interface (matches DrosophilaLarvaModel)
    # ------------------------------------------------------------------ #

    def receptor_activity(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        rates = self._orn_type_rates(odor_name, odor_intensity)
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
        orn_arr = self._dict_to_orn_array(receptor_signal)
        layers = self._forward_from_orn(orn_arr)
        behavior = self._behavior_from_mbon(layers["mbon"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_pn(
        self, pn_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        pn_arr = self._dict_to_pn_array(pn_signal)
        layers = self._forward_from_pn(pn_arr)
        behavior = self._behavior_from_mbon(layers["mbon"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_kc(
        self, kc_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        """Inject at the KC (mushroom body) level — testing expansion write."""
        kc_arr = self._dict_to_kc_array(kc_signal)
        layers = self._forward_from_kc(kc_arr)
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
    # Receiver learning
    # ------------------------------------------------------------------ #

    def adapt_mbon_weights(
        self,
        training_kc: List[np.ndarray],
        training_labels: List[str],
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Update KC→MBON weights via supervised Hebbian-like learning.

        Models the mushroom body's ability to adapt to an imperfect incoming
        code.  In the real fly, dopaminergic neurons gate KC→MBON plasticity
        based on reward/punishment.

        training_kc: list of KC activity vectors from transferred signals
        training_labels: "attractive" or "repellent" for each
        """
        X = np.array(training_kc)
        Y = np.array([
            [1.0, 0.0] if lab == "attractive" else [0.0, 1.0]
            for lab in training_labels
        ])

        w = self._mbon_weights.copy()
        n_samples = len(X)
        rng = np.random.default_rng(self._kc_seed + 2000)

        for _ in range(epochs):
            order = rng.permutation(n_samples)
            for idx in order:
                pred = w @ X[idx]
                error = Y[idx] - pred
                w += learning_rate * np.outer(error, X[idx]) / n_samples

        self._mbon_weights = w

    # ------------------------------------------------------------------ #
    # Variant support
    # ------------------------------------------------------------------ #

    def clone_perturbed(
        self,
        orn_gain: Optional[Dict[str, float]] = None,
        kc_weight_noise: Optional[np.ndarray] = None,
        mbon_weight_noise: Optional[np.ndarray] = None,
        kc_seed: Optional[int] = None,
    ) -> "DrosophilaAdultModel":
        new_gain = dict(orn_gain) if orn_gain is not None else dict(self.orn_gain)
        new_kc_seed = kc_seed if kc_seed is not None else self._kc_seed

        clone = DrosophilaAdultModel(
            orn_gain=new_gain,
            orn_noise_cv=self.orn_noise_cv,
            pn_baseline=self.pn_baseline,
            lateral_inhibition=self.lateral_inhibition,
            kc_threshold=self.kc_threshold,
            kc_sparsity=self.kc_sparsity,
            _kc_seed=new_kc_seed,
            approach_threshold=self.approach_threshold,
            _orn_convergence_seed=self._orn_convergence_seed,
        )

        if kc_weight_noise is not None:
            clone._kc_weights = np.maximum(clone._kc_weights + kc_weight_noise, 0.0)
        if mbon_weight_noise is not None:
            clone._mbon_weights = np.maximum(clone._mbon_weights + mbon_weight_noise, 0.0)
            row_sums = clone._mbon_weights.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            clone._mbon_weights /= row_sums

        return clone
