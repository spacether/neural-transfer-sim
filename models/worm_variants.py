"""Individual-variation and sanity-check utilities for worm models.

The variant generator is calibrated against real worm-to-worm variability
measured across individually mapped connectomes (Witvliet et al. 2021,
Cook et al. 2019).  A jitter_scale of 1.0 reproduces the median biological
coefficient of variation (CV ≈ 0.55) for olfactory-pathway synapse counts,
meaning the sweep parameter directly represents "multiples of typical
biological variability."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, MutableMapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.worm_connectome import HandBuiltWormModel
    from models.cook2019_connectome import Cook2019WormModel

BIOLOGICAL_MEDIAN_CV = 0.55
BIOLOGICAL_DROPOUT_RATE = 0.30


@dataclass
class WormVariantGenerator:
    """Generate biologically grounded worm model variants.

    jitter_scale=1.0 produces variability matching the median synapse-count
    CV observed across real worms.  Values above 1.0 represent more divergent
    individuals; below 1.0, more similar ones.
    """

    jitter_scale: float = 1.0
    sensitivity_scale: float = 0.2
    minimum_receptor_gain: float = 0.5

    def _lognormal_factor(self, rng: np.random.Generator, target_cv: float) -> float:
        """Draw a multiplicative noise factor with the given CV."""
        if target_cv <= 0:
            return 1.0
        sigma = np.sqrt(np.log(1.0 + target_cv ** 2))
        mu = -0.5 * sigma ** 2
        return float(rng.lognormal(mu, sigma))

    def generate_variant(self, base_gain: MutableMapping[str, float], *, rng: Iterable[float] | None = None) -> Dict[str, float]:
        """Return a copy of the receptor gain map with mild biological variation."""
        values = list(base_gain.values()) if rng is None else list(rng)
        if len(values) < len(base_gain):
            values = [0.95 + (i * 0.03) for i in range(len(base_gain))]

        mutated: Dict[str, float] = {}
        for idx, (neuron, value) in enumerate(base_gain.items()):
            scale = values[idx % len(values)] if idx < len(values) else 1.0
            mutated[neuron] = max(
                self.minimum_receptor_gain,
                value * (1.0 + (scale - 1.0) * self.jitter_scale + self.sensitivity_scale * 0.1),
            )
        return mutated

    def sanity_check(self, model: object) -> bool:
        """Check whether a worm model still has the minimum expected behavioral invariants."""
        if not hasattr(model, "simulate"):
            return False

        attractive = model.simulate(1.0, "attractive")
        repellent = model.simulate(1.0, "repellent")
        return attractive["behavior"] == "forward" and repellent["behavior"] == "reverse"

    def generate_model_variant(
        self,
        base_model: "HandBuiltWormModel | Cook2019WormModel",
        *,
        seed: int | None = None,
        vary_weights: bool = True,
    ) -> "HandBuiltWormModel | Cook2019WormModel":
        """Return a model variant with biologically grounded perturbations.

        Supports both HandBuiltWormModel and Cook2019WormModel.  Noise is
        lognormal so weights stay positive and the distribution matches the
        right-skewed synapse-count variability seen in real connectomes.
        Connection dropout reflects the observation that ~60% of olfactory
        connections are present in some animals but not others.
        """
        from models.cook2019_connectome import Cook2019WormModel

        if isinstance(base_model, Cook2019WormModel):
            return self._generate_cook2019_variant(base_model, seed=seed, vary_weights=vary_weights)
        return self._generate_handbuilt_variant(base_model, seed=seed, vary_weights=vary_weights)

    def _generate_handbuilt_variant(
        self,
        base_model: "HandBuiltWormModel",
        *,
        seed: int | None = None,
        vary_weights: bool = True,
    ) -> "HandBuiltWormModel":
        from models.worm_connectome import HandBuiltWormModel

        rng = np.random.default_rng(seed)
        target_cv = BIOLOGICAL_MEDIAN_CV * self.jitter_scale
        dropout_rate = BIOLOGICAL_DROPOUT_RATE * min(self.jitter_scale, 1.5)

        mutated_gain: Dict[str, float] = {}
        for neuron, value in base_model.receptor_gain.items():
            factor = self._lognormal_factor(rng, target_cv)
            mutated_gain[neuron] = max(self.minimum_receptor_gain, value * factor)

        threshold_cv = target_cv * 0.3
        cmd_factor = self._lognormal_factor(rng, threshold_cv)
        rev_factor = self._lognormal_factor(rng, threshold_cv)
        cmd_thresh = max(0.3, base_model.command_threshold * cmd_factor)
        rev_thresh = max(0.2, base_model.reverse_threshold * rev_factor)

        if not vary_weights:
            return HandBuiltWormModel(
                receptor_gain=mutated_gain,
                command_threshold=cmd_thresh,
                reverse_threshold=rev_thresh,
            )

        mutated_weights: Dict[str, float] = {}
        for key, value in base_model.weight_matrix.items():
            if rng.random() < dropout_rate:
                mutated_weights[key] = 0.0
                continue
            factor = self._lognormal_factor(rng, target_cv)
            mutated_weights[key] = max(0.01, value * factor)

        return HandBuiltWormModel(
            receptor_gain=mutated_gain,
            weight_matrix=mutated_weights,
            command_threshold=cmd_thresh,
            reverse_threshold=rev_thresh,
        )

    def _generate_cook2019_variant(
        self,
        base_model: "Cook2019WormModel",
        *,
        seed: int | None = None,
        vary_weights: bool = True,
    ) -> "Cook2019WormModel":
        rng = np.random.default_rng(seed)
        target_cv = BIOLOGICAL_MEDIAN_CV * self.jitter_scale
        dropout_rate = BIOLOGICAL_DROPOUT_RATE * min(self.jitter_scale, 1.5)

        mutated_gain: Dict[str, float] = {}
        for neuron, value in base_model.receptor_gain.items():
            factor = self._lognormal_factor(rng, target_cv)
            mutated_gain[neuron] = max(self.minimum_receptor_gain, value * factor)

        n = len(base_model._neurons)
        weight_noise = None
        dropout_mask = None

        if vary_weights and target_cv > 0:
            sigma = np.sqrt(np.log(1.0 + target_cv ** 2))
            mu = -0.5 * sigma ** 2
            weight_noise = rng.lognormal(mu, sigma, size=(n, n))
            dropout_mask = (rng.random((n, n)) >= dropout_rate).astype(np.float64)

        threshold_cv = target_cv * 0.3
        approach_factor = self._lognormal_factor(rng, threshold_cv)
        avoidance_factor = self._lognormal_factor(rng, threshold_cv)

        return base_model.clone_perturbed(
            receptor_gain=mutated_gain,
            weight_noise=weight_noise,
            dropout_mask=dropout_mask,
            threshold_factors=(approach_factor, avoidance_factor),
        )


def generate_worm_variant(base_gain: MutableMapping[str, float], *, jitter_scale: float = 1.0) -> Dict[str, float]:
    """Convenience helper for building a single perturbed worm receptor profile."""
    generator = WormVariantGenerator(jitter_scale=jitter_scale)
    return generator.generate_variant(base_gain)
