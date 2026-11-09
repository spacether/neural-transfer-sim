"""Individual-variation generator for the Mus musculus auditory model.

Perturbation axes match the synthetic mouse auditory pathway:
  1. IHC gain jitter — hair cell sensitivity variation across 120 IHCs
  2. IHC→CN weight noise — synapse count variation (60 CN targets)
  3. CN→CN weight noise — lateral inhibition variation within bottleneck
  4. CN→IC weight noise — projection variation (150 IC targets)

BIOLOGICAL_MEDIAN_CV = 0.55 is the only directly measured per-connection
synapse-count CV, from 8 isogenic C. elegans (Witvliet et al. 2021,
Nature 596:257–261).  Drosophila bilateral data (FlyWire, Dorkenwald
et al. 2024, Nature 634:124–138) implies a comparable CV of ~0.45.
No mouse connectome exists at synapse resolution across individuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neural_transfer.models.mus_musculus.auditory_model import MouseAuditoryModel


BIOLOGICAL_MEDIAN_CV = 0.55


@dataclass
class MouseAuditoryVariantGenerator:
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
        base_model: "MouseAuditoryModel",
        *,
        seed: int = 0,
    ) -> "MouseAuditoryModel":
        rng = np.random.default_rng(seed)
        target_cv = BIOLOGICAL_MEDIAN_CV * self.jitter_scale

        mutated_gain: Dict[str, float] = {}
        for name, g in base_model.ihc_gain.items():
            factor = self._lognormal_factor(rng, target_cv)
            mutated_gain[name] = max(self.minimum_gain, g * factor)

        ihc_cn_noise = None
        if target_cv > 0 and base_model._w_ihc_cn is not None:
            mask = base_model._w_ihc_cn > 0
            if mask.any():
                w_scale = base_model._w_ihc_cn[mask].mean()
                sigma = w_scale * target_cv * 0.5
                noise = rng.normal(0, sigma, size=base_model._w_ihc_cn.shape)
                ihc_cn_noise = noise * mask

        cn_cn_noise = None
        if target_cv > 0 and base_model._w_cn_cn is not None:
            mask = base_model._w_cn_cn != 0
            if mask.any():
                w_scale = np.abs(base_model._w_cn_cn[mask]).mean()
                sigma = w_scale * target_cv * 0.3
                noise = rng.normal(0, sigma, size=base_model._w_cn_cn.shape)
                cn_cn_noise = noise * mask

        cn_ic_noise = None
        if target_cv > 0 and base_model._w_cn_ic is not None:
            mask = base_model._w_cn_ic > 0
            if mask.any():
                w_scale = base_model._w_cn_ic[mask].mean()
                sigma = w_scale * target_cv * 0.3
                noise = rng.normal(0, sigma, size=base_model._w_cn_ic.shape)
                cn_ic_noise = noise * mask

        return base_model.clone_perturbed(
            ihc_gain=mutated_gain,
            ihc_cn_weight_noise=ihc_cn_noise,
            cn_cn_weight_noise=cn_cn_noise,
            cn_ic_weight_noise=cn_ic_noise,
        )

    def sanity_check(self, model: "MouseAuditoryModel") -> bool:
        checks = model.validate_sound_responses()
        return checks["approach_response"] and checks["startle_response"]
