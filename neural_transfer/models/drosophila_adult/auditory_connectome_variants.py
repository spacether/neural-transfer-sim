"""Individual-variation generator for the auditory adult fly model.

Perturbation axes match the FlyWire v783 connectome-derived model:
  1. JO gain jitter — mechanosensory sensitivity variation across 1103 JO neurons
  2. JO→AMMC weight noise — synapse count variation (150 AMMC targets)
  3. AMMC→WED/VLP weight noise — third-order projection variation (~324 targets)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neural_transfer.models.drosophila_adult.auditory_connectome import AuditoryAdultModel


BIOLOGICAL_MEDIAN_CV = 0.40


@dataclass
class AuditoryAdultVariantGenerator:
    """Generate variants of the connectome-derived auditory adult fly model.

    Perturbation axes:
    1. JO gain jitter (mechanosensory sensitivity variation)
    2. JO→AMMC weight noise (synapse count variation)
    3. AMMC→WED/VLP weight noise (third-order projection variation)
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
        base_model: "AuditoryAdultModel",
        *,
        seed: int = 0,
    ) -> "AuditoryAdultModel":
        rng = np.random.default_rng(seed)
        target_cv = BIOLOGICAL_MEDIAN_CV * self.jitter_scale

        mutated_gain: Dict[str, float] = {}
        for jo, g in base_model.jo_gain.items():
            factor = self._lognormal_factor(rng, target_cv)
            mutated_gain[jo] = max(self.minimum_gain, g * factor)

        jo_ammc_noise = None
        if target_cv > 0 and base_model._w_jo_ammc is not None:
            mask = base_model._w_jo_ammc > 0
            if mask.any():
                w_scale = base_model._w_jo_ammc[mask].mean()
                sigma = w_scale * target_cv * 0.15
                noise = rng.normal(0, sigma, size=base_model._w_jo_ammc.shape)
                jo_ammc_noise = noise * mask

        ammc_wedvlp_noise = None
        if target_cv > 0 and base_model._w_ammc_wedvlp is not None:
            mask = base_model._w_ammc_wedvlp > 0
            if mask.any():
                w_scale = base_model._w_ammc_wedvlp[mask].mean()
                sigma = w_scale * target_cv * 0.1
                noise = rng.normal(0, sigma, size=base_model._w_ammc_wedvlp.shape)
                ammc_wedvlp_noise = noise * mask

        return base_model.clone_perturbed(
            jo_gain=mutated_gain,
            jo_ammc_weight_noise=jo_ammc_noise,
            ammc_wed_weight_noise=ammc_wedvlp_noise,
        )

    def sanity_check(self, model: "AuditoryAdultModel") -> bool:
        checks = model.validate_sound_responses()
        return checks["courtship_response"] and checks["startle_escape"]
