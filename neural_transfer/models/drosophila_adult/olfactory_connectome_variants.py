"""Individual-variation generator for the connectome-based adult fly model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neural_transfer.models.drosophila_adult.olfactory_connectome import ConnectomeAdultModel


BIOLOGICAL_MEDIAN_CV = 0.40


@dataclass
class ConnectomeAdultVariantGenerator:
    """Generate variants of the connectome-based adult fly model.

    Perturbation axes:
    1. ORN gain jitter (receptor sensitivity variation)
    2. ORN→PN weight noise (convergence strength variation, preserving sparsity)
    3. PN→KC weight noise (synaptic strength variation)
    4. KC→MBON weight noise (learning-dependent variation)
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
        base_model: "ConnectomeAdultModel",
        *,
        seed: int = 0,
    ) -> "ConnectomeAdultModel":
        rng = np.random.default_rng(seed)
        target_cv = BIOLOGICAL_MEDIAN_CV * self.jitter_scale

        mutated_gain: Dict[str, float] = {}
        for orn, g in base_model.orn_gain.items():
            factor = self._lognormal_factor(rng, target_cv)
            mutated_gain[orn] = max(self.minimum_gain, g * factor)

        # ORN→PN noise: jitter existing connections only (preserves glomerular identity)
        orn_pn_noise = None
        if target_cv > 0 and base_model._w_orn_pn is not None:
            mask = base_model._w_orn_pn > 0
            if mask.any():
                w_scale = base_model._w_orn_pn[mask].mean()
                sigma = w_scale * target_cv * 0.15
                noise = rng.normal(0, sigma, size=base_model._w_orn_pn.shape)
                orn_pn_noise = noise * mask

        kc_noise = None
        if target_cv > 0 and base_model._w_pn_kc is not None:
            w_scale = base_model._w_pn_kc[base_model._w_pn_kc > 0].mean()
            sigma = w_scale * target_cv * 0.1
            kc_noise = rng.normal(0, sigma, size=base_model._w_pn_kc.shape)

        mbon_noise = None
        if target_cv > 0 and base_model._w_kc_mbon is not None:
            w_scale = base_model._w_kc_mbon[base_model._w_kc_mbon > 0].mean()
            sigma = w_scale * target_cv * 0.1
            mbon_noise = rng.normal(0, sigma, size=base_model._w_kc_mbon.shape)

        return base_model.clone_perturbed(
            orn_gain=mutated_gain,
            orn_pn_weight_noise=orn_pn_noise,
            kc_weight_noise=kc_noise,
            mbon_weight_noise=mbon_noise,
        )

    def sanity_check(self, model: "ConnectomeAdultModel") -> bool:
        checks = model.validate_odor_responses()
        return checks["attractive_approach"] and checks["repellent_avoid"]
