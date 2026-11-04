"""Larval zebrafish olfactory model.

Implements the larval zebrafish olfactory pathway:
  ~300 ORNs (100 types × ~3 neurons each) → 100 glomeruli → 80 mitral
  cells (with granule-cell-mediated recurrent lateral inhibition) →
  500 Dp neurons (dorsal telencephalon, sparse code) → output neurons.

This is the first vertebrate model in the project.  Key architectural
differences from the fly:
  - Mitral cells sample from multiple glomeruli (~2-3 each), unlike fly
    PNs which have 1:1 glomerular identity.
  - Granule cells provide recurrent lateral inhibition within the bulb,
    decorrelating mitral cell output — a feedback loop absent in flies.
  - Dp neurons are the sparse-coding expansion layer (analogous to KCs).

Odor stimuli are aquatic: amino acids (food), bile acids (social cues),
nucleotides (food/damage), and alarm substances.  Response profiles are
grounded in Friedrich & Laurent 2001 (mitral cell recordings to amino
acids) and Kermen et al. 2013 (larval glomerular maps).

Signal mode: firing rates (spikes/s), same as the fly models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Neuron definitions
# --------------------------------------------------------------------------- #

N_OR_TYPES = 100
ORNS_PER_TYPE = 3
N_TOTAL_ORNS = N_OR_TYPES * ORNS_PER_TYPE  # 300
N_GLOMERULI = N_OR_TYPES  # 1:1 with OR types
N_MITRAL = 80
N_GRANULE = 200
N_DP = 500  # dorsal telencephalon — sparse expansion layer
N_OUTPUT = 2

OR_TYPES: List[str] = [f"OR_{i:03d}" for i in range(N_OR_TYPES)]

# Amino-acid-tuned ORs: 0-39
# Bile-acid-tuned ORs: 40-54
# Nucleotide-tuned ORs: 55-64
# Alarm/amine-tuned ORs: 65-79
# Broadly tuned / other: 80-99

GLOM_NAMES: List[str] = [f"Glom_{i:03d}" for i in range(N_GLOMERULI)]
MITRAL_NAMES: List[str] = [f"MC_{i:03d}" for i in range(N_MITRAL)]
DP_NAMES: List[str] = [f"Dp_{i:04d}" for i in range(N_DP)]
OUTPUT_NAMES: List[str] = ["OUT_approach", "OUT_avoid"]

ALL_NEURON_NAMES: List[str] = OR_TYPES + MITRAL_NAMES + DP_NAMES + OUTPUT_NAMES

SITE_NEURONS = {
    "orn": OR_TYPES,
    "mitral": MITRAL_NAMES,
    "dp": DP_NAMES,
    "output": OUTPUT_NAMES,
}

# --------------------------------------------------------------------------- #
# Odor response profiles
# --------------------------------------------------------------------------- #
# Firing rate changes (Δ spikes/s) per OR type.  Based on:
# - Friedrich & Laurent 2001: amino acid responses in adult zebrafish
#   mitral cells — used to constrain OR-level profiles
# - Kermen et al. 2013: larval glomerular activation maps
# - DeMaria et al. 2013: TAAR-mediated amine detection
#
# Each amino acid activates a subset of amino-acid-tuned ORs (0-39) plus
# a few broadly tuned ones (80-99).  Chemical similarity → correlated
# profiles (e.g., branched-chain amino acids Leu/Ile/Val share ORs).

def _aa_profile(primary: List[int], secondary: List[int],
                primary_rate: float = 80.0, secondary_rate: float = 30.0,
                broad: List[int] | None = None) -> Dict[str, float]:
    """Helper to build an amino acid odor profile."""
    p: Dict[str, float] = {}
    for i in primary:
        p[f"OR_{i:03d}"] = primary_rate + (i % 7) * 3.0
    for i in secondary:
        p[f"OR_{i:03d}"] = secondary_rate + (i % 5) * 2.0
    if broad:
        for i in broad:
            p[f"OR_{i:03d}"] = 15.0 + (i % 4) * 3.0
    return p


def _bile_profile(primary: List[int], secondary: List[int],
                  primary_rate: float = 70.0) -> Dict[str, float]:
    p: Dict[str, float] = {}
    for i in primary:
        p[f"OR_{i:03d}"] = primary_rate + (i % 5) * 4.0
    for i in secondary:
        p[f"OR_{i:03d}"] = 25.0 + (i % 4) * 3.0
    return p


def _nuc_profile(primary: List[int], secondary: List[int],
                 primary_rate: float = 65.0) -> Dict[str, float]:
    p: Dict[str, float] = {}
    for i in primary:
        p[f"OR_{i:03d}"] = primary_rate + (i % 6) * 3.0
    for i in secondary:
        p[f"OR_{i:03d}"] = 20.0 + (i % 3) * 4.0
    return p


ODOR_PROFILES: Dict[str, Dict[str, float]] = {
    # --- Amino acids (food-related, mostly attractive) ---
    # Branched-chain group: Leu, Ile, Val share ORs 0-5
    "L_leucine": _aa_profile([0, 1, 2, 3], [4, 5, 12, 15], broad=[80, 85, 90]),
    "L_isoleucine": _aa_profile([0, 1, 3, 4], [2, 5, 13, 16], broad=[80, 86]),
    "L_valine": _aa_profile([0, 2, 4, 5], [1, 3, 14, 17], broad=[81, 85]),
    # Aromatic group: Phe, Trp, Tyr share ORs 6-11
    "L_phenylalanine": _aa_profile([6, 7, 8, 9], [10, 11, 18, 20], broad=[82, 87, 91]),
    "L_tryptophan": _aa_profile([6, 8, 10, 11], [7, 9, 19, 21], broad=[82, 88]),
    "L_tyrosine": _aa_profile([7, 9, 10, 11], [6, 8, 20, 22], broad=[83, 87]),
    # Basic group: Arg, Lys, His share ORs 12-17
    "L_arginine": _aa_profile([12, 13, 14, 15], [16, 17, 0, 6], broad=[84, 89, 92]),
    "L_lysine": _aa_profile([12, 14, 16, 17], [13, 15, 1, 7], broad=[84, 90]),
    "L_histidine": _aa_profile([13, 15, 16, 17], [12, 14, 8, 23], broad=[85, 89]),
    # Polar/hydroxyl: Ser, Thr, Cys share ORs 18-23
    "L_serine": _aa_profile([18, 19, 20], [21, 22, 23, 2], broad=[86, 91]),
    "L_threonine": _aa_profile([18, 20, 22], [19, 21, 23, 3], broad=[86, 92]),
    "L_cysteine": _aa_profile([19, 21, 23], [18, 20, 22, 4], broad=[87, 93]),
    # Acidic: Glu, Asp share ORs 24-29
    "L_glutamate": _aa_profile([24, 25, 26, 27], [28, 29, 5, 30], broad=[88, 94]),
    "L_aspartate": _aa_profile([24, 26, 28, 29], [25, 27, 9, 31], broad=[88, 95]),
    # Small/other: Gly, Ala, Pro, Met
    "glycine": _aa_profile([30, 31, 32], [33, 18, 24], broad=[90, 96]),
    "L_alanine": _aa_profile([30, 33, 34], [31, 32, 0, 12], broad=[91, 97]),
    "L_proline": _aa_profile([34, 35, 36], [33, 37, 26, 15], broad=[92, 98]),
    "L_methionine": _aa_profile([37, 38, 39], [36, 35, 11, 23], broad=[93, 99]),

    # --- Bile acids (social/alarm cues, mixed valence) ---
    "taurocholic_acid": _bile_profile([40, 41, 42, 43], [44, 45, 80, 81]),
    "taurodeoxycholic_acid": _bile_profile([40, 42, 44, 45], [41, 43, 46, 82]),
    "glycocholic_acid": _bile_profile([41, 43, 46, 47], [40, 42, 48, 83]),
    "lithocholic_acid": _bile_profile([44, 46, 48, 49], [40, 45, 47, 84]),

    # --- Nucleotides (food/damage signals, attractive) ---
    "ATP": _nuc_profile([55, 56, 57, 58], [59, 60, 85, 86]),
    "ADP": _nuc_profile([55, 57, 59, 60], [56, 58, 61, 87]),
    "AMP": _nuc_profile([56, 58, 60, 61], [55, 57, 62, 88]),
    "IMP": _nuc_profile([57, 59, 61, 62], [55, 56, 63, 89]),

    # --- Alarm/aversive substances ---
    "alarm_extract": {
        "OR_065": 90.0, "OR_066": 85.0, "OR_067": 75.0, "OR_068": 70.0,
        "OR_069": 60.0, "OR_070": 55.0, "OR_071": 45.0,
        "OR_080": 20.0, "OR_090": 15.0,
    },
    "cadaverine": {
        "OR_072": 95.0, "OR_073": 80.0, "OR_074": 65.0, "OR_075": 55.0,
        "OR_065": 30.0, "OR_066": 25.0,
        "OR_094": 18.0,
    },
    "putrescine": {
        "OR_073": 85.0, "OR_074": 75.0, "OR_076": 60.0, "OR_077": 50.0,
        "OR_065": 25.0, "OR_095": 15.0,
    },
    "chondroitin_sulfate": {
        "OR_068": 70.0, "OR_069": 65.0, "OR_070": 55.0, "OR_078": 50.0,
        "OR_079": 40.0, "OR_067": 30.0,
    },
}

ODOR_VALENCE: Dict[str, str] = {
    # Amino acids — attractive (food)
    "L_leucine": "attractive", "L_isoleucine": "attractive",
    "L_valine": "attractive", "L_phenylalanine": "attractive",
    "L_tryptophan": "attractive", "L_tyrosine": "attractive",
    "L_arginine": "attractive", "L_lysine": "attractive",
    "L_histidine": "attractive", "L_serine": "attractive",
    "L_threonine": "attractive", "L_cysteine": "attractive",
    "L_glutamate": "attractive", "L_aspartate": "attractive",
    "glycine": "attractive", "L_alanine": "attractive",
    "L_proline": "attractive", "L_methionine": "attractive",
    # Bile acids — mixed (social)
    "taurocholic_acid": "attractive", "taurodeoxycholic_acid": "attractive",
    "glycocholic_acid": "repellent", "lithocholic_acid": "repellent",
    # Nucleotides — attractive (food/damage)
    "ATP": "attractive", "ADP": "attractive",
    "AMP": "attractive", "IMP": "attractive",
    # Alarm — repellent
    "alarm_extract": "repellent", "cadaverine": "repellent",
    "putrescine": "repellent", "chondroitin_sulfate": "repellent",
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
class ZebrafishModel:
    """Rate-based model of the larval zebrafish olfactory pathway.

    Architecture: ORNs (3 per type) → glomeruli (1:1 with OR types) →
    mitral cells (each samples ~2-3 glomeruli, with granule-cell
    recurrent lateral inhibition) → Dp (sparse random projection) →
    output neurons.

    The mitral cell layer is the key difference from the fly: each mitral
    cell integrates across multiple glomeruli, and granule cells provide
    recurrent inhibition that decorrelates the output.
    """

    orn_gain: Dict[str, float] = field(
        default_factory=lambda: {orn: 1.0 for orn in OR_TYPES}
    )
    orn_noise_cv: float = 0.25

    # Bulb parameters
    mitral_baseline: float = 5.0
    granule_gain: float = 0.15
    recurrent_iterations: int = 3

    # Dp (telencephalic) sparse coding
    dp_threshold: float = 0.3
    dp_sparsity: int = 10  # mitral cell inputs per Dp neuron
    _dp_weights: Optional[np.ndarray] = field(default=None, repr=False)
    _dp_seed: int = 42

    # Mitral cell connectivity (each samples from ~2-3 glomeruli)
    _mitral_weights: Optional[np.ndarray] = field(default=None, repr=False)
    _mitral_seed: int = 7

    # Granule cell connectivity
    _granule_exc_weights: Optional[np.ndarray] = field(default=None, repr=False)
    _granule_inh_weights: Optional[np.ndarray] = field(default=None, repr=False)

    # Output weights
    _output_weights: Optional[np.ndarray] = field(default=None, repr=False)

    approach_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self._mitral_weights is None:
            self._build_mitral_connectivity()
        if self._granule_exc_weights is None:
            self._build_granule_connectivity()
        if self._dp_weights is None:
            self._build_dp_connectivity()
        if self._output_weights is None:
            self._build_output_connectivity()

    def _build_mitral_connectivity(self) -> None:
        """Each mitral cell receives from 2-3 glomeruli."""
        rng = np.random.default_rng(self._mitral_seed)
        w = np.zeros((N_MITRAL, N_GLOMERULI))
        for i in range(N_MITRAL):
            n_inputs = rng.choice([2, 3], p=[0.4, 0.6])
            chosen = rng.choice(N_GLOMERULI, size=n_inputs, replace=False)
            for j in chosen:
                w[i, j] = rng.uniform(0.5, 1.0)
        self._mitral_weights = w

    def _build_granule_connectivity(self) -> None:
        """Granule cells: excited by mitral cells, inhibit them back."""
        rng = np.random.default_rng(self._mitral_seed + 100)
        # Mitral → Granule (excitatory)
        exc = np.zeros((N_GRANULE, N_MITRAL))
        for i in range(N_GRANULE):
            n_inputs = rng.choice([3, 4, 5])
            chosen = rng.choice(N_MITRAL, size=n_inputs, replace=False)
            for j in chosen:
                exc[i, j] = rng.uniform(0.3, 0.8)
        self._granule_exc_weights = exc
        # Granule → Mitral (inhibitory, broader)
        inh = np.zeros((N_MITRAL, N_GRANULE))
        for i in range(N_MITRAL):
            n_inputs = rng.choice([5, 6, 7, 8])
            chosen = rng.choice(N_GRANULE, size=n_inputs, replace=False)
            for j in chosen:
                inh[i, j] = rng.uniform(0.2, 0.6)
        self._granule_inh_weights = inh

    def _build_dp_connectivity(self) -> None:
        """Random sparse mitral→Dp projection (~10 mitral cells per Dp neuron)."""
        rng = np.random.default_rng(self._dp_seed)
        w = np.zeros((N_DP, N_MITRAL))
        for i in range(N_DP):
            chosen = rng.choice(N_MITRAL, size=self.dp_sparsity, replace=False)
            for j in chosen:
                w[i, j] = rng.uniform(0.3, 1.0)
        self._dp_weights = w

    def _build_output_connectivity(self) -> None:
        """Fit output weights via least-squares on known odor valences."""
        rng = np.random.default_rng(self._dp_seed + 1000)

        dp_vecs = []
        targets = []
        for odor in ODOR_PROFILES:
            orn = self._orn_type_rates(odor, 1.0)
            mitral = self._mitral_rates(orn)
            dp = self._dp_rates(mitral)
            dp_vecs.append(dp)
            if ODOR_VALENCE.get(odor, "attractive") == "attractive":
                targets.append([1.0, 0.0])
            else:
                targets.append([0.0, 1.0])

        X = np.array(dp_vecs)
        Y = np.array(targets)
        w, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        w = w.T
        w += rng.normal(0, 0.001, w.shape)
        self._output_weights = w

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def _orn_type_rates(self, odor_name: str, intensity: float) -> np.ndarray:
        """Mean ORN firing rate per type (after convergence over 3 ORNs)."""
        profile = ODOR_PROFILES.get(odor_name, {})
        rates = np.zeros(N_OR_TYPES)
        for i, orn in enumerate(OR_TYPES):
            base = profile.get(orn, 0.0)
            rates[i] = max(0.0, self.orn_gain[orn] * base * intensity)
        return rates

    def _orn_type_rates_noisy(self, odor_name: str, intensity: float,
                               rng: np.random.Generator | None = None) -> np.ndarray:
        rates = self._orn_type_rates(odor_name, intensity)
        if rng is None or self.orn_noise_cv <= 0:
            return rates
        convergence_cv = self.orn_noise_cv / np.sqrt(ORNS_PER_TYPE)
        noise = rng.normal(1.0, convergence_cv, size=rates.shape)
        return np.maximum(rates * noise, 0.0)

    def _glomerular_rates(self, orn_type_rates: np.ndarray) -> np.ndarray:
        """Glomeruli are 1:1 with OR types in zebrafish."""
        return orn_type_rates.copy()

    def _mitral_rates(self, orn_type_rates: np.ndarray) -> np.ndarray:
        """Mitral cell rates with granule-cell recurrent lateral inhibition.

        Each iteration: granule cells read mitral activity, then inhibit
        mitral cells.  This decorrelates the output and sharpens tuning.
        """
        glom = self._glomerular_rates(orn_type_rates)
        mc = self._mitral_weights @ glom + self.mitral_baseline
        mc = np.maximum(mc, 0.0)

        for _ in range(self.recurrent_iterations):
            granule = self._granule_exc_weights @ mc
            granule = np.maximum(granule, 0.0)
            inhibition = self._granule_inh_weights @ granule
            mc = self._mitral_weights @ glom + self.mitral_baseline - self.granule_gain * inhibition
            mc = np.maximum(mc, 0.0)

        return mc

    def _dp_rates(self, mitral_rates: np.ndarray) -> np.ndarray:
        """Dp neuron rates: sparse random projection with threshold."""
        baseline_mc = np.full(N_MITRAL, self.mitral_baseline)
        raw = self._dp_weights @ mitral_rates
        baseline = self._dp_weights @ baseline_mc
        driven = np.maximum(raw - baseline, 0.0)
        peak = driven.max()
        if peak < 1e-9:
            return np.zeros(N_DP)
        normed = driven / peak
        dp = np.where(normed > self.dp_threshold, normed, 0.0)
        return dp

    def _output_rates(self, dp_rates: np.ndarray) -> np.ndarray:
        return self._output_weights @ dp_rates

    def _forward(self, odor_name: str, intensity: float) -> Dict[str, np.ndarray]:
        orn = self._orn_type_rates(odor_name, intensity)
        mitral = self._mitral_rates(orn)
        dp = self._dp_rates(mitral)
        output = self._output_rates(dp)
        return {"orn": orn, "mitral": mitral, "dp": dp, "output": output}

    def _forward_from_orn(self, orn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        mitral = self._mitral_rates(orn_rates)
        dp = self._dp_rates(mitral)
        output = self._output_rates(dp)
        return {"orn": orn_rates, "mitral": mitral, "dp": dp, "output": output}

    def _forward_from_mitral(self, mitral_rates: np.ndarray) -> Dict[str, np.ndarray]:
        dp = self._dp_rates(mitral_rates)
        output = self._output_rates(dp)
        return {"mitral": mitral_rates, "dp": dp, "output": output}

    def _forward_from_dp(self, dp_rates: np.ndarray) -> Dict[str, np.ndarray]:
        output = self._output_rates(dp_rates)
        return {"dp": dp_rates, "output": output}

    # ------------------------------------------------------------------ #
    # Dict conversions
    # ------------------------------------------------------------------ #

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        if "orn" in layers:
            for i, n in enumerate(OR_TYPES):
                d[n] = float(layers["orn"][i])
        if "mitral" in layers:
            for i, n in enumerate(MITRAL_NAMES):
                d[n] = float(layers["mitral"][i])
        if "dp" in layers:
            for i, n in enumerate(DP_NAMES):
                d[n] = float(layers["dp"][i])
        if "output" in layers:
            for i, n in enumerate(OUTPUT_NAMES):
                d[n] = float(layers["output"][i])
        return d

    def _dict_to_orn_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(orn, 0.0) for orn in OR_TYPES])

    def _dict_to_mitral_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(mc, 0.0) for mc in MITRAL_NAMES])

    def _dict_to_dp_array(self, signal: Dict[str, float]) -> np.ndarray:
        return np.array([signal.get(dp, 0.0) for dp in DP_NAMES])

    # ------------------------------------------------------------------ #
    # Behavior
    # ------------------------------------------------------------------ #

    def _behavior_from_output(self, output: np.ndarray) -> str:
        mag = abs(output[0]) + abs(output[1])
        if mag < self.approach_threshold:
            return "no_response"
        diff = output[0] - output[1]
        if diff > self.approach_threshold:
            return "approach"
        elif diff < -self.approach_threshold:
            return "avoid"
        return "no_response"

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def receptor_activity(
        self, odor_intensity: float, odor_type: str = "attractive"
    ) -> Dict[str, float]:
        odor_name = _pick_odor(odor_type)
        rates = self._orn_type_rates(odor_name, odor_intensity)
        return {orn: float(rates[i]) for i, orn in enumerate(OR_TYPES)}

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

    def simulate_from_mitral(
        self, mitral_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        mc_arr = self._dict_to_mitral_array(mitral_signal)
        layers = self._forward_from_mitral(mc_arr)
        behavior = self._behavior_from_output(layers["output"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_dp(
        self, dp_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict:
        dp_arr = self._dict_to_dp_array(dp_signal)
        layers = self._forward_from_dp(dp_arr)
        behavior = self._behavior_from_output(layers["output"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

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

    def adapt_output_weights(
        self,
        training_dp: List[np.ndarray],
        training_labels: List[str],
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Update Dp→output weights via supervised learning."""
        X = np.array(training_dp)
        Y = np.array([
            [1.0, 0.0] if lab == "attractive" else [0.0, 1.0]
            for lab in training_labels
        ])

        w = self._output_weights.copy()
        n_samples = len(X)
        rng = np.random.default_rng(self._dp_seed + 2000)

        for _ in range(epochs):
            order = rng.permutation(n_samples)
            for idx in order:
                pred = w @ X[idx]
                error = Y[idx] - pred
                w += learning_rate * np.outer(error, X[idx]) / n_samples

        self._output_weights = w

    # ------------------------------------------------------------------ #
    # Variant support
    # ------------------------------------------------------------------ #

    def clone_perturbed(
        self,
        orn_gain: Optional[Dict[str, float]] = None,
        dp_seed: Optional[int] = None,
        mitral_seed: Optional[int] = None,
        output_weight_noise: Optional[np.ndarray] = None,
    ) -> "ZebrafishModel":
        new_gain = dict(orn_gain) if orn_gain is not None else dict(self.orn_gain)
        new_dp_seed = dp_seed if dp_seed is not None else self._dp_seed
        new_mitral_seed = mitral_seed if mitral_seed is not None else self._mitral_seed

        clone = ZebrafishModel(
            orn_gain=new_gain,
            orn_noise_cv=self.orn_noise_cv,
            mitral_baseline=self.mitral_baseline,
            granule_gain=self.granule_gain,
            recurrent_iterations=self.recurrent_iterations,
            dp_threshold=self.dp_threshold,
            dp_sparsity=self.dp_sparsity,
            _dp_seed=new_dp_seed,
            _mitral_seed=new_mitral_seed,
            approach_threshold=self.approach_threshold,
        )

        if output_weight_noise is not None:
            clone._output_weights = clone._output_weights + output_weight_noise

        return clone
