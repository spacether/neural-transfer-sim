"""Individual-variation generator for zebrafish models.

Perturbation axes:
1. ORN gain jitter (receptor sensitivity variation)
2. Dp connectivity rewiring (different random mitral→Dp projection)
3. Output weight noise (learning-dependent variation)

The mitral cell connectivity (glomerular sampling) can optionally be
varied as well — unlike fly PNs which have fixed 1:1 identity, zebrafish
mitral cells sample from 2-3 glomeruli and this wiring may vary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neural_transfer.models.zebrafish import ZebrafishModel


BIOLOGICAL_MEDIAN_CV = 0.35


@dataclass
class ZebrafishVariantGenerator:
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
        base_model: "ZebrafishModel",
        *,
        seed: int = 0,
        vary_dp: bool = True,
        vary_output: bool = True,
    ) -> "ZebrafishModel":
        from neural_transfer.models.zebrafish import ZebrafishModel

        rng = np.random.default_rng(seed)
        target_cv = BIOLOGICAL_MEDIAN_CV * self.jitter_scale

        mutated_gain: Dict[str, float] = {}
        for orn, g in base_model.orn_gain.items():
            factor = self._lognormal_factor(rng, target_cv)
            mutated_gain[orn] = max(self.minimum_gain, g * factor)

        dp_seed = base_model._dp_seed
        if vary_dp and self.jitter_scale > 0:
            dp_seed = int(rng.integers(0, 2**31))

        clone = base_model.clone_perturbed(
            orn_gain=mutated_gain,
            dp_seed=dp_seed,
        )

        if vary_output and target_cv > 0 and clone._output_weights is not None:
            w_scale = np.abs(clone._output_weights).mean() + 1e-9
            sigma = w_scale * target_cv * 0.1
            output_noise = rng.normal(0, sigma, size=clone._output_weights.shape)
            clone._output_weights = clone._output_weights + output_noise

        return clone

    def sanity_check(self, model: "ZebrafishModel") -> bool:
        checks = model.validate_odor_responses()
        return checks["attractive_approach"] and checks["repellent_avoid"]
