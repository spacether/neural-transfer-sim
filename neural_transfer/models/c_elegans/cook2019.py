"""Full 302-neuron graded connectome model from Cook et al. 2019.

Loads the hermaphrodite chemical synapse adjacency matrix via cect and assigns
excitatory/inhibitory signs from CeNGEN neurotransmitter expression data.
Dynamics are graded and non-spiking: each neuron's activation is a sigmoid of
its weighted input, iterated to steady state.

This replaces the 9-connection HandBuiltWormModel with the real connectome
(3700+ connections, 21000 synapses) while keeping the same public interface.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

OLFACTORY_RECEPTORS = {
    "attractive": ["AWAL", "AWAR", "AWCL", "AWCR"],
    "repellent": ["AWBL", "AWBR", "ADLL", "ADLR"],
}
ALL_RECEPTORS = ["AWAL", "AWAR", "AWCL", "AWCR", "AWBL", "AWBR", "ADLL", "ADLR"]
INTERNEURONS = ["AIYL", "AIYR", "AIBL", "AIBR", "AIAL", "AIAR", "AIZL", "AIZR"]
COMMAND_NEURONS = ["AVAL", "AVAR", "AVBL", "AVBR"]
OLFACTORY_PATHWAY = ALL_RECEPTORS + INTERNEURONS + COMMAND_NEURONS

# Mapping from the HandBuiltWormModel's class-level names to L/R neuron pairs
CLASS_TO_LR = {
    "AWA": ("AWAL", "AWAR"),
    "AWC_L": ("AWCL",),
    "AWC_R": ("AWCR",),
    "AWB": ("AWBL", "AWBR"),
    "ADL": ("ADLL", "ADLR"),
    "AIY": ("AIYL", "AIYR"),
    "AIB": ("AIBL", "AIBR"),
    "AIA": ("AIAL", "AIAR"),
    "AIZ": ("AIZL", "AIZR"),
    "AVA": ("AVAL", "AVAR"),
    "AVB": ("AVBL", "AVBR"),
}
LR_TO_CLASS = {}
for cls, neurons in CLASS_TO_LR.items():
    for n in neurons:
        LR_TO_CLASS[n] = cls


def _load_cengen_neurotransmitters() -> Dict[str, str]:
    """Load neurotransmitter identity per neuron class from CeNGEN expression data.

    Uses medium threshold (threshold 2) which captures AWA and AWB cholinergic
    expression missed at conservative threshold.  Falls back to literature
    assignments for classes still below threshold.

    Returns a dict mapping neuron class -> 'excitatory' or 'inhibitory'.
    """
    try:
        wa_dir = os.path.dirname(__import__("wormneuroatlas").__file__)
    except ImportError:
        return _fallback_nt_signs()

    cengen_path = os.path.join(
        wa_dir, "data", "cengen_021821_medium_threshold2.csv"
    )
    if not os.path.exists(cengen_path):
        return _fallback_nt_signs()

    nt_genes = {
        "unc-17": "ACh",
        "cha-1": "ACh",
        "eat-4": "Glu",
        "unc-25": "GABA",
        "unc-47": "GABA",
    }

    class_nts: Dict[str, set] = {}
    with open(cengen_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        neuron_classes = header[3:]

        for row in reader:
            if len(row) < 3:
                continue
            gene = row[1].strip()
            if gene not in nt_genes:
                continue
            nt = nt_genes[gene]
            for i, cls in enumerate(neuron_classes):
                col_idx = i + 3
                try:
                    val = float(row[col_idx]) if col_idx < len(row) and row[col_idx].strip() else 0.0
                except ValueError:
                    val = 0.0
                if val > 0:
                    class_nts.setdefault(cls, set()).add(nt)

    signs: Dict[str, str] = {}
    for cls, nts in class_nts.items():
        if "GABA" in nts:
            signs[cls] = "inhibitory"
        else:
            signs[cls] = "excitatory"

    for cls, sign in _fallback_nt_signs().items():
        if cls not in signs:
            signs[cls] = sign

    return signs


def _fallback_nt_signs() -> Dict[str, str]:
    """Literature-based neurotransmitter sign assignments for the olfactory pathway."""
    return {
        "AWA": "excitatory",
        "AWC_ON": "excitatory",
        "AWC_OFF": "excitatory",
        "AWB": "excitatory",
        "ADL": "excitatory",
        "AIY": "excitatory",
        "AIB": "excitatory",
        "AIA": "excitatory",
        "AIZ": "excitatory",
        "AVA": "excitatory",
        "AVB": "excitatory",
        "RIM": "excitatory",
        "RIA": "excitatory",
        "DD": "inhibitory",
        "VD": "inhibitory",
        "RME": "inhibitory",
    }


@lru_cache(maxsize=1)
def _load_cook2019() -> Tuple[List[str], Dict[Tuple[str, str], float], Dict[str, str]]:
    """Load Cook 2019 hermaphrodite connectome and neurotransmitter signs.

    Returns (neuron_list, chemical_adjacency, neuron_sign_map).
    chemical_adjacency maps (pre, post) -> synapse_count.
    neuron_sign_map maps neuron_name -> 'excitatory' or 'inhibitory'.
    """
    from cect.readers.Cook2019HermReader import get_instance

    reader = get_instance()
    neurons, conns = reader.get_neuron_to_neuron_conns()

    chem: Dict[Tuple[str, str], float] = {}
    for c in conns:
        if "Chemical" in c.syntype:
            key = (c.pre_cell, c.post_cell)
            chem[key] = chem.get(key, 0.0) + c.number

    class_signs = _load_cengen_neurotransmitters()

    neuron_signs: Dict[str, str] = {}
    for n in neurons:
        cls = n.rstrip("LR")
        if cls.endswith("D"):
            cls = cls[:-1]
        if cls in class_signs:
            neuron_signs[n] = class_signs[cls]
        elif n in class_signs:
            neuron_signs[n] = class_signs[n]
        else:
            neuron_signs[n] = "excitatory"

    return neurons, chem, neuron_signs


@dataclass
class Cook2019WormModel:
    """Full 302-neuron graded connectome model from Cook et al. 2019.

    Dynamics: activation_i = sigmoid(sum_j(sign_j * w_ij * a_j) + input_i)
    where w_ij is normalized synapse count and sign_j is +1 (excitatory) or
    -1 (inhibitory) based on CeNGEN neurotransmitter expression.
    """

    receptor_gain: Dict[str, float] = field(
        default_factory=lambda: {
            "AWA": 1.0,
            "AWC_L": 1.0,
            "AWC_R": 1.0,
            "AWB": 1.0,
            "ADL": 1.0,
        }
    )
    weight_scale: float = 1.0
    sigmoid_gain: float = 5.0
    sigmoid_midpoint: float = 0.5
    max_iterations: int = 80
    convergence_threshold: float = 1e-6
    approach_threshold: float = 0.1
    avoidance_threshold: float = 0.1

    _neurons: List[str] = field(default_factory=list, repr=False)
    _neuron_idx: Dict[str, int] = field(default_factory=dict, repr=False)
    _weight_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    _sign_vector: Optional[np.ndarray] = field(default=None, repr=False)
    _baseline: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._build_matrices()

    def _build_matrices(self) -> None:
        neurons, chem, signs = _load_cook2019()
        self._neurons = list(neurons)
        self._neuron_idx = {n: i for i, n in enumerate(neurons)}
        n = len(neurons)

        W = np.zeros((n, n), dtype=np.float64)
        for (pre, post), count in chem.items():
            if pre in self._neuron_idx and post in self._neuron_idx:
                i_pre = self._neuron_idx[pre]
                i_post = self._neuron_idx[post]
                W[i_post, i_pre] = count

        row_sum = W.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        W = W / row_sum * self.weight_scale

        sign_vec = np.ones(n, dtype=np.float64)
        for neuron, sign in signs.items():
            if neuron in self._neuron_idx:
                if sign == "inhibitory":
                    sign_vec[self._neuron_idx[neuron]] = -1.0

        self._weight_matrix = W
        self._sign_vector = sign_vec

        self._baseline = self._propagate_to_steady_state(np.zeros(n, dtype=np.float64))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.sigmoid_gain * (x - self.sigmoid_midpoint)))

    def _propagate_to_steady_state(self, input_vec: np.ndarray) -> np.ndarray:
        n = len(self._neurons)
        activation = np.zeros(n, dtype=np.float64)

        for _ in range(self.max_iterations):
            signed_activation = activation * self._sign_vector
            raw = self._weight_matrix @ signed_activation + input_vec
            new_activation = self._sigmoid(raw)
            new_activation = np.where(input_vec > 0, np.maximum(new_activation, input_vec), new_activation)

            if np.max(np.abs(new_activation - activation)) < self.convergence_threshold:
                activation = new_activation
                break
            activation = new_activation

        return activation

    def _make_input_vector(self, odor_intensity: float, odor_type: str) -> np.ndarray:
        n = len(self._neurons)
        input_vec = np.zeros(n, dtype=np.float64)
        receptors = OLFACTORY_RECEPTORS.get(odor_type, [])
        for r in receptors:
            if r in self._neuron_idx:
                cls = LR_TO_CLASS.get(r, r)
                gain = self.receptor_gain.get(cls, 1.0)
                input_vec[self._neuron_idx[r]] = gain * odor_intensity
        return input_vec

    def _activation_to_class_state(self, activation: np.ndarray) -> Dict[str, float]:
        """Convert full activation vector to class-level state dict matching HandBuiltWormModel.

        Reports odor-evoked response as the delta above spontaneous baseline
        activity, so that olfactory-specific signals are not masked by
        recurrent drive from the rest of the 302-neuron network.
        """
        state: Dict[str, float] = {}
        for cls, lr_neurons in CLASS_TO_LR.items():
            vals = []
            for n in lr_neurons:
                if n in self._neuron_idx:
                    idx = self._neuron_idx[n]
                    delta = activation[idx] - self._baseline[idx]
                    vals.append(max(0.0, delta))
            if vals:
                state[cls] = float(np.mean(vals))
        return state

    def _activation_to_full_state(self, activation: np.ndarray) -> Dict[str, float]:
        return {n: float(activation[i]) for i, n in enumerate(self._neurons) if activation[i] > 1e-8}

    def receptor_activity(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, float]:
        if odor_intensity < 0:
            raise ValueError("odor_intensity must be non-negative")
        state: Dict[str, float] = {}
        receptors = OLFACTORY_RECEPTORS.get(odor_type, [])
        for r in receptors:
            cls = LR_TO_CLASS.get(r, r)
            state[cls] = self.receptor_gain.get(cls, 1.0) * odor_intensity
        return state

    def network_state(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, float]:
        if odor_intensity < 0:
            raise ValueError("odor_intensity must be non-negative")
        input_vec = self._make_input_vector(odor_intensity, odor_type)
        activation = self._propagate_to_steady_state(input_vec)
        return self._activation_to_class_state(activation)

    def behavior_from_network(self, state: Dict[str, float], odor_type: str | None = None) -> str:
        """Derive behavior from the AIY/AIB interneuron balance.

        In the real worm, AIY drives approach (forward movement) and AIB drives
        avoidance (reversal).  In the full 302-neuron network, the command
        neurons AVA/AVB receive input from many non-olfactory pathways, so the
        olfactory-specific behavioral decision is better read at the
        interneuron level.
        """
        aiy = state.get("AIY", 0.0)
        aib = state.get("AIB", 0.0)

        if odor_type is None:
            attractive_active = any(state.get(k, 0) > 0.05 for k in ("AWA", "AWC_L", "AWC_R"))
            repellent_active = any(state.get(k, 0) > 0.05 for k in ("AWB", "ADL"))
            if repellent_active and not attractive_active:
                odor_type = "repellent"
            else:
                odor_type = "attractive"

        if odor_type == "attractive":
            if aiy >= self.approach_threshold and aiy > aib:
                return "forward"
            return "pause"

        if odor_type == "repellent":
            if aib >= self.avoidance_threshold:
                return "reverse"
            return "pause"

        raise ValueError(f"Unsupported odor_type: {odor_type!r}")

    def odor_response(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, float]:
        return self.receptor_activity(odor_intensity, odor_type)

    def simulate_from_receptors(
        self, receptor_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict[str, object]:
        n = len(self._neurons)
        input_vec = np.zeros(n, dtype=np.float64)

        for cls, value in receptor_signal.items():
            lr_neurons = CLASS_TO_LR.get(cls, ())
            for neuron in lr_neurons:
                if neuron in self._neuron_idx:
                    input_vec[self._neuron_idx[neuron]] = value

        activation = self._propagate_to_steady_state(input_vec)
        state = self._activation_to_class_state(activation)
        behavior = self.behavior_from_network(state, odor_type)

        return {
            "model_type": "cook2019_connectome",
            "source": "cook_2019_302_neuron",
            "odor_type": odor_type,
            "behavior": behavior,
            "neurons": state,
            "injection": "receptor_layer",
        }

    def behavior_from_signal(self, signal: Dict[str, float], odor_type: str = "attractive") -> str:
        result = self.simulate_from_receptors(signal, odor_type)
        return result["behavior"]

    def simulate(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, object]:
        if odor_intensity < 0:
            raise ValueError("odor_intensity must be non-negative")
        state = self.network_state(odor_intensity, odor_type)
        behavior = self.behavior_from_network(state, odor_type)
        return {
            "model_type": "cook2019_connectome",
            "source": "cook_2019_302_neuron",
            "odor_type": odor_type,
            "odor_intensity": odor_intensity,
            "behavior": behavior,
            "neurons": state,
        }

    def validate_odor_responses(self, odor_intensity: float = 1.0) -> Dict[str, object]:
        attractive = self.simulate(odor_intensity, "attractive")
        repellent = self.simulate(odor_intensity, "repellent")
        attractive_ok = attractive["behavior"] == "forward"
        repellent_ok = repellent["behavior"] == "reverse"
        return {
            "model_type": "cook2019_connectome",
            "source": "cook_2019_302_neuron",
            "odor_intensity": odor_intensity,
            "attractive": attractive,
            "repellent": repellent,
            "passes": attractive_ok and repellent_ok,
            "checks": {
                "attractive_drives_forward": attractive_ok,
                "repellent_drives_reverse": repellent_ok,
            },
        }

    @property
    def neuron_count(self) -> int:
        return len(self._neurons)

    @property
    def connection_count(self) -> int:
        return int(np.count_nonzero(self._weight_matrix))

    @property
    def weight_matrix(self) -> Dict[str, float]:
        """Return olfactory pathway connections as a sparse dict for variant generator compatibility."""
        weights: Dict[str, float] = {}
        for pre in OLFACTORY_PATHWAY:
            for post in OLFACTORY_PATHWAY:
                if pre in self._neuron_idx and post in self._neuron_idx:
                    val = self._weight_matrix[self._neuron_idx[post], self._neuron_idx[pre]]
                    if val > 1e-10:
                        pre_cls = LR_TO_CLASS.get(pre, pre)
                        post_cls = LR_TO_CLASS.get(post, post)
                        key = f"{pre_cls}->{post_cls}"
                        weights[key] = weights.get(key, 0.0) + float(val)
        return weights

    def clone_perturbed(
        self,
        *,
        receptor_gain: Optional[Dict[str, float]] = None,
        weight_noise: Optional[np.ndarray] = None,
        dropout_mask: Optional[np.ndarray] = None,
        threshold_factors: Optional[Tuple[float, float]] = None,
    ) -> "Cook2019WormModel":
        """Create a variant with perturbed weights and receptor gains.

        receptor_gain: replacement receptor gain dict.
        weight_noise: multiplicative factors for each weight (same shape as weight matrix).
        dropout_mask: boolean mask, True = keep connection, False = drop.
        threshold_factors: (approach_factor, avoidance_factor) multipliers.
        """
        import copy
        variant = copy.copy(self)
        W = self._weight_matrix.copy()

        if weight_noise is not None:
            W = W * weight_noise
        if dropout_mask is not None:
            W = W * dropout_mask

        variant._weight_matrix = W
        variant._sign_vector = self._sign_vector.copy()
        variant._neurons = list(self._neurons)
        variant._neuron_idx = dict(self._neuron_idx)

        if receptor_gain is not None:
            variant.receptor_gain = dict(receptor_gain)
        else:
            variant.receptor_gain = dict(self.receptor_gain)

        if threshold_factors is not None:
            variant.approach_threshold = max(0.02, self.approach_threshold * threshold_factors[0])
            variant.avoidance_threshold = max(0.02, self.avoidance_threshold * threshold_factors[1])

        variant._baseline = variant._propagate_to_steady_state(
            np.zeros(len(self._neurons), dtype=np.float64)
        )
        return variant

    @property
    def command_threshold(self) -> float:
        """Backward compatibility: map to approach_threshold."""
        return self.approach_threshold

    @property
    def reverse_threshold(self) -> float:
        """Backward compatibility: map to avoidance_threshold."""
        return self.avoidance_threshold
