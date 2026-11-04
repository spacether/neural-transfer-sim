"""Individual-variation generator for adult Drosophila models.

Key biological fact: PN wiring (glomerular identity) is conserved across
individuals, but KC wiring (the random projection from PNs to mushroom
body Kenyon cells) is structurally variable.  This means:
  - Reading/writing at the PN level targets a conserved bottleneck.
  - Reading/writing at the KC level hits the most variable part of the
    circuit, so calibration must bridge a larger gap.

Perturbation axes, in order of increasing severity:
1. ORN gain jitter (receptor sensitivity variation across individuals)
2. KC connectivity rewiring (different random PN→KC projection)
3. MBON weight noise (learning-dependent variation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.drosophila_adult import DrosophilaAdultModel


BIOLOGICAL_MEDIAN_CV = 0.35


@dataclass
class AdultVariantGenerator:
    """Generate biologically grounded adult fly model variants.

    jitter_scale=1.0 produces variability matching typical ORN response
    variability across individual adult flies.
    """

    jitter_scale: float = 1.0
    minimum_gain: float = 0.3

    def _lognormal_factor(self, rng: np.random.Generator, target_cv: float) -> float:
        if target_cv <= 0:
            return 1.0
        sigma = np.sqrt(np.log(1.0 + target_cv ** 2))
        mu = -0.5 * sigma ** 2
        return float(rng.lognormal(mu, sigma))

    def generate_variant(
        self,
        base_model: "DrosophilaAdultModel",
        *,
        seed: int = 0,
        vary_kc: bool = True,
        vary_mbon: bool = True,
    ) -> "DrosophilaAdultModel":
        from models.drosophila_adult import DrosophilaAdultModel

        rng = np.random.default_rng(seed)
        target_cv = BIOLOGICAL_MEDIAN_CV * self.jitter_scale

        mutated_gain: Dict[str, float] = {}
        for orn, g in base_model.orn_gain.items():
            factor = self._lognormal_factor(rng, target_cv)
            mutated_gain[orn] = max(self.minimum_gain, g * factor)

        kc_seed = base_model._kc_seed
        if vary_kc and self.jitter_scale > 0:
            kc_seed = int(rng.integers(0, 2**31))

        clone = base_model.clone_perturbed(
            orn_gain=mutated_gain,
            kc_seed=kc_seed,
        )

        if vary_mbon and target_cv > 0 and clone._mbon_weights is not None:
            w_scale = np.abs(clone._mbon_weights).mean() + 1e-9
            sigma = w_scale * target_cv * 0.1
            mbon_noise = rng.normal(0, sigma, size=clone._mbon_weights.shape)
            clone._mbon_weights = clone._mbon_weights + mbon_noise

        return clone

    def sanity_check(self, model: "DrosophilaAdultModel") -> bool:
        checks = model.validate_odor_responses()
        return checks["attractive_approach"] and checks["repellent_avoid"]
