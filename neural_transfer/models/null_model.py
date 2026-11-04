"""Null (control) network models with matched dimensionality.

These models match each organism's layer sizes (receptor count, bottleneck
count, expansion count, output count) but use a configurable architecture.
By comparing transfer results on a null model against the biological model,
we can separate effects of dimensionality from effects of architecture.

Three architectures are provided:
- "dense": every layer is fully connected to the next (no sparsity, no bottleneck)
- "sparse_random": each neuron samples a fixed number of inputs randomly
- "one_to_one": 1:1 mapping at the bottleneck, then sparse expansion (larva-like)

All use the same odor profiles and valences as the matched organism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class NullModel:
    """A generic feedforward rate-code network with configurable architecture.

    Layers: input (receptors) -> bottleneck -> expansion -> output (2 neurons).
    """

    n_input: int = 21
    n_bottleneck: int = 21
    n_expansion: int = 72
    n_output: int = 2

    architecture: str = "dense"
    sparsity_k: int = 6

    input_names: List[str] = field(default_factory=list)
    bottleneck_names: List[str] = field(default_factory=list)
    expansion_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    odor_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    odor_valence: Dict[str, str] = field(default_factory=dict)

    input_gain: Dict[str, float] = field(default_factory=dict)
    baseline: float = 5.0
    lateral_inhibition: float = 0.15
    expansion_threshold: float = 0.3
    approach_threshold: float = 0.05

    _w_bottleneck: Optional[np.ndarray] = field(default=None, repr=False)
    _w_expansion: Optional[np.ndarray] = field(default=None, repr=False)
    _w_output: Optional[np.ndarray] = field(default=None, repr=False)
    _seed: int = 42

    def __post_init__(self) -> None:
        if not self.input_names:
            self.input_names = [f"IN_{i}" for i in range(self.n_input)]
        if not self.bottleneck_names:
            self.bottleneck_names = [f"BN_{i}" for i in range(self.n_bottleneck)]
        if not self.expansion_names:
            self.expansion_names = [f"EX_{i}" for i in range(self.n_expansion)]
        if not self.output_names:
            self.output_names = ["OUT_approach", "OUT_avoid"]
        if not self.input_gain:
            self.input_gain = {n: 1.0 for n in self.input_names}

        if self._w_bottleneck is None and self.architecture != "shuffled_connectome":
            self._build_bottleneck()
        if self._w_expansion is None and self.architecture != "shuffled_connectome":
            self._build_expansion()
        if self._w_output is None and self.architecture != "shuffled_connectome":
            self._build_output()

    def _build_bottleneck(self) -> None:
        rng = np.random.default_rng(self._seed)
        n_in, n_bn = self.n_input, self.n_bottleneck

        if self.architecture == "one_to_one":
            w = np.zeros((n_bn, n_in))
            for i in range(min(n_bn, n_in)):
                w[i, i] = 1.0
            if n_bn > n_in:
                for i in range(n_in, n_bn):
                    w[i, rng.integers(n_in)] = 1.0
        elif self.architecture == "dense":
            w = rng.uniform(0.3, 1.0, (n_bn, n_in))
        elif self.architecture == "sparse_random":
            w = np.zeros((n_bn, n_in))
            k = min(self.sparsity_k, n_in)
            for i in range(n_bn):
                chosen = rng.choice(n_in, size=k, replace=False)
                for j in chosen:
                    w[i, j] = rng.uniform(0.3, 1.0)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        self._w_bottleneck = w

    def _build_expansion(self) -> None:
        rng = np.random.default_rng(self._seed + 100)
        n_bn, n_ex = self.n_bottleneck, self.n_expansion

        if self.architecture == "one_to_one":
            w = np.zeros((n_ex, n_bn))
            k = min(self.sparsity_k, n_bn)
            for i in range(n_ex):
                chosen = rng.choice(n_bn, size=k, replace=False)
                for j in chosen:
                    w[i, j] = rng.uniform(0.3, 1.0)
        elif self.architecture == "dense":
            w = rng.uniform(0.1, 0.5, (n_ex, n_bn))
        elif self.architecture == "sparse_random":
            w = np.zeros((n_ex, n_bn))
            k = min(self.sparsity_k, n_bn)
            for i in range(n_ex):
                chosen = rng.choice(n_bn, size=k, replace=False)
                for j in chosen:
                    w[i, j] = rng.uniform(0.3, 1.0)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        self._w_expansion = w

    def _build_output(self) -> None:
        rng = np.random.default_rng(self._seed + 200)

        ex_vecs = []
        targets = []
        for odor, profile in self.odor_profiles.items():
            inp = self._input_rates(odor, 1.0)
            bn = self._bottleneck_rates(inp)
            ex = self._expansion_rates(bn)
            ex_vecs.append(ex)
            if self.odor_valence.get(odor, "attractive") == "attractive":
                targets.append([1.0, 0.0])
            else:
                targets.append([0.0, 1.0])

        if not ex_vecs:
            self._w_output = rng.normal(0, 0.01, (self.n_output, self.n_expansion))
            return

        X = np.array(ex_vecs)
        Y = np.array(targets)
        w, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        w = w.T
        w += rng.normal(0, 0.001, w.shape)
        self._w_output = w

    def _input_rates(self, odor_name: str, intensity: float) -> np.ndarray:
        profile = self.odor_profiles.get(odor_name, {})
        rates = np.zeros(self.n_input)
        for i, name in enumerate(self.input_names):
            base = profile.get(name, 0.0)
            rates[i] = max(0.0, self.input_gain[name] * base * intensity)
        return rates

    def _bottleneck_rates(self, input_rates: np.ndarray) -> np.ndarray:
        raw = self._w_bottleneck @ input_rates
        global_mean = raw.mean()
        bn = raw - self.lateral_inhibition * global_mean + self.baseline
        return np.maximum(bn, 0.0)

    def _expansion_rates(self, bn_rates: np.ndarray) -> np.ndarray:
        baseline_bn = np.full(self.n_bottleneck, self.baseline)
        raw = self._w_expansion @ bn_rates
        base = self._w_expansion @ baseline_bn
        driven = np.maximum(raw - base, 0.0)
        peak = driven.max()
        if peak < 1e-9:
            return np.zeros(self.n_expansion)
        normed = driven / peak
        return np.where(normed > self.expansion_threshold, normed, 0.0)

    def _output_rates(self, ex_rates: np.ndarray) -> np.ndarray:
        return self._w_output @ ex_rates

    def _forward(self, odor_name: str, intensity: float) -> Dict[str, np.ndarray]:
        inp = self._input_rates(odor_name, intensity)
        bn = self._bottleneck_rates(inp)
        ex = self._expansion_rates(bn)
        out = self._output_rates(ex)
        return {"input": inp, "bottleneck": bn, "expansion": ex, "output": out}

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
        name_map = {
            "input": self.input_names,
            "bottleneck": self.bottleneck_names,
            "expansion": self.expansion_names,
            "output": self.output_names,
        }
        for key, names in name_map.items():
            if key in layers:
                for i, n in enumerate(names):
                    d[n] = float(layers[key][i])
        return d

    def simulate(self, odor_name: str, intensity: float) -> Dict:
        layers = self._forward(odor_name, intensity)
        behavior = self._behavior_from_output(layers["output"])
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def simulate_from_input(self, signal: Dict[str, float]) -> Dict:
        inp = np.array([signal.get(n, 0.0) for n in self.input_names])
        bn = self._bottleneck_rates(inp)
        ex = self._expansion_rates(bn)
        out = self._output_rates(ex)
        layers = {"input": inp, "bottleneck": bn, "expansion": ex, "output": out}
        behavior = self._behavior_from_output(out)
        return {"behavior": behavior, "neurons": self._layers_to_dict(layers)}

    def input_response(self, odor_name: str, intensity: float) -> Dict[str, float]:
        rates = self._input_rates(odor_name, intensity)
        return {n: float(rates[i]) for i, n in enumerate(self.input_names)}


def build_null_for_organism(
    organism: str,
    architecture: str = "dense",
    seed: int = 42,
) -> NullModel:
    """Build a null model matching an organism's dimensions but with a different architecture."""

    if organism == "larva":
        from neural_transfer.models.drosophila_larva import (
            ORN_TYPES, N_KC, ODOR_PROFILES, ODOR_VALENCE,
        )
        return NullModel(
            n_input=len(ORN_TYPES),
            n_bottleneck=len(ORN_TYPES),
            n_expansion=N_KC,
            n_output=2,
            architecture=architecture,
            sparsity_k=6,
            input_names=list(ORN_TYPES),
            odor_profiles=dict(ODOR_PROFILES),
            odor_valence=dict(ODOR_VALENCE),
            _seed=seed,
        )

    if organism == "adult":
        from neural_transfer.models.drosophila_adult import (
            ORN_TYPES, N_KC, ODOR_PROFILES, ODOR_VALENCE,
        )
        return NullModel(
            n_input=len(ORN_TYPES),
            n_bottleneck=len(ORN_TYPES),
            n_expansion=N_KC,
            n_output=2,
            architecture=architecture,
            sparsity_k=5,
            input_names=list(ORN_TYPES),
            odor_profiles=dict(ODOR_PROFILES),
            odor_valence=dict(ODOR_VALENCE),
            _seed=seed,
        )

    if organism == "zebrafish":
        from neural_transfer.models.zebrafish import (
            OR_TYPES, N_MITRAL, N_DP, ODOR_PROFILES, ODOR_VALENCE,
        )
        return NullModel(
            n_input=len(OR_TYPES),
            n_bottleneck=N_MITRAL,
            n_expansion=N_DP,
            n_output=2,
            architecture=architecture,
            sparsity_k=6,
            input_names=list(OR_TYPES),
            odor_profiles=dict(ODOR_PROFILES),
            odor_valence=dict(ODOR_VALENCE),
            _seed=seed,
        )

    raise ValueError(f"Unknown organism: {organism}")


def build_shuffled_connectome(
    organism: str,
    seed: int = 42,
) -> NullModel:
    """Build a null model using real connectome weight matrices with shuffled structure.

    Preserves the exact weight distribution of every layer but destroys the
    wiring pattern.  This is the strongest architecture control: any difference
    in transfer between the real connectome model and this shuffled version is
    attributable to wiring topology, not weight statistics.
    """
    rng = np.random.default_rng(seed)

    if organism == "larva":
        from neural_transfer.models.drosophila_larva import ConnectomeLarvaModel
        from neural_transfer.models.drosophila_larva import ODOR_PROFILES, ODOR_VALENCE, ORN_TYPES

        base = ConnectomeLarvaModel()
        w_bn = base._w_orn_pn.copy()  # keep real ORN→PN
        w_ex = base._w_pn_kc.copy()
        n_in = w_bn.shape[1]
        n_bn = w_bn.shape[0]
        n_ex = w_ex.shape[0]

        # Shuffle only PN→KC: for each KC, shuffle which PNs it receives from
        for i in range(n_ex):
            row = w_ex[i, :].copy()
            rng.shuffle(row)
            w_ex[i, :] = row

        m = NullModel(
            n_input=n_in, n_bottleneck=n_bn, n_expansion=n_ex, n_output=2,
            architecture="shuffled_connectome",
            input_names=list(ORN_TYPES),
            odor_profiles=dict(ODOR_PROFILES),
            odor_valence=dict(ODOR_VALENCE),
            _seed=seed,
        )
        m._w_bottleneck = w_bn
        m._w_expansion = w_ex
        m._build_output()
        return m

    if organism == "adult":
        from neural_transfer.models.drosophila_adult import ConnectomeAdultModel
        from neural_transfer.models.drosophila_adult import ODOR_PROFILES, ODOR_VALENCE

        base = ConnectomeAdultModel()
        w_bn = base._w_orn_pn.copy()  # keep real ORN→PN (glomerular identity)
        w_ex = base._w_pn_kc.copy()
        n_in = w_bn.shape[1]
        n_bn = w_bn.shape[0]
        n_ex = w_ex.shape[0]

        # Shuffle only PN→KC: for each KC, shuffle which PNs it receives from
        for i in range(n_ex):
            row = w_ex[i, :].copy()
            rng.shuffle(row)
            w_ex[i, :] = row

        m = NullModel(
            n_input=n_in, n_bottleneck=n_bn, n_expansion=n_ex, n_output=2,
            architecture="shuffled_connectome",
            input_names=list(base._receptor_names),
            odor_profiles=dict(ODOR_PROFILES),
            odor_valence=dict(ODOR_VALENCE),
            _seed=seed,
        )
        m._w_bottleneck = w_bn
        m._w_expansion = w_ex
        m._build_output()
        return m

    if organism == "zebrafish":
        from neural_transfer.models.zebrafish import OR_TYPES, ODOR_PROFILES, ODOR_VALENCE, N_MITRAL, N_DP

        # Zebrafish doesn't have a connectome model yet, fall back to sparse_random
        return build_null_for_organism("zebrafish", architecture="sparse_random", seed=seed)

    raise ValueError(f"Unknown organism: {organism}")


def build_null_variant(
    model: NullModel,
    jitter_scale: float = 0.5,
    seed: int = 0,
) -> NullModel:
    """Generate a variant of a null model with jittered weights."""
    rng = np.random.default_rng(seed)

    new_gain = {}
    for name, g in model.input_gain.items():
        factor = rng.lognormal(0, 0.2 * jitter_scale)
        new_gain[name] = max(0.5, g * factor)

    # For shuffled_connectome, pass pre-built weights so __post_init__ doesn't skip them
    extra_kw = {}
    if model.architecture == "shuffled_connectome" and model._w_bottleneck is not None:
        extra_kw["_w_bottleneck"] = model._w_bottleneck.copy()
        extra_kw["_w_expansion"] = model._w_expansion.copy()
        extra_kw["_w_output"] = model._w_output.copy()

    clone = NullModel(
        n_input=model.n_input,
        n_bottleneck=model.n_bottleneck,
        n_expansion=model.n_expansion,
        n_output=model.n_output,
        architecture=model.architecture,
        sparsity_k=model.sparsity_k,
        input_names=list(model.input_names),
        bottleneck_names=list(model.bottleneck_names),
        expansion_names=list(model.expansion_names),
        output_names=list(model.output_names),
        odor_profiles=model.odor_profiles,
        odor_valence=model.odor_valence,
        input_gain=new_gain,
        baseline=model.baseline,
        lateral_inhibition=model.lateral_inhibition,
        expansion_threshold=model.expansion_threshold,
        approach_threshold=model.approach_threshold * rng.lognormal(0, 0.1 * jitter_scale),
        _seed=model._seed,
        **extra_kw,
    )

    noise_scale = jitter_scale * 0.1
    bn_noise = rng.normal(0, noise_scale, clone._w_bottleneck.shape)
    clone._w_bottleneck = np.maximum(clone._w_bottleneck + bn_noise, 0.0)

    ex_noise = rng.normal(0, noise_scale, clone._w_expansion.shape)
    clone._w_expansion = np.maximum(clone._w_expansion + ex_noise, 0.0)

    out_noise = rng.normal(0, noise_scale * 0.5, clone._w_output.shape)
    clone._w_output = clone._w_output + out_noise

    return clone
