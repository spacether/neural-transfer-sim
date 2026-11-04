"""Empirical synapse-count variability from published C. elegans connectomes.

Source data: Witvliet et al. 2021 (Nature 596, 63-71) — 8 individually
mapped worms at developmental stages L1 through adult — plus Cook et al.
2019 (Nature 571, 63-71) hermaphrodite adult.  Synapse counts extracted
via the OpenWorm Connectome Toolbox (cect) for olfactory-pathway neuron
pairs (AWA, AWC, AWB, ADL → AIY, AIB, AIA, AIZ → AVA, AVB).

The coefficient of variation (CV = std/mean) of synapse counts across
individuals at the same developmental stage is the primary measure of
worm-to-worm wiring variability.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
#  Summary statistics (computed from per-pair synapse counts)                   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StageVariability:
    """Observed synapse-count variability for one developmental stage."""
    stage: str
    n_animals: int
    n_connections: int
    median_cv: float
    mean_cv: float
    cv_25th: float
    cv_75th: float
    frac_variable_presence: float


L1_VARIABILITY = StageVariability(
    stage="L1",
    n_animals=4,
    n_connections=49,
    median_cv=0.577,
    mean_cv=0.570,
    cv_25th=0.346,
    cv_75th=0.707,
    frac_variable_presence=0.60,
)

ADULT_VARIABILITY = StageVariability(
    stage="adult",
    n_animals=3,
    n_connections=74,
    median_cv=0.542,
    mean_cv=0.567,
    cv_25th=0.375,
    cv_75th=0.707,
    frac_variable_presence=0.62,
)

BIOLOGICAL_MEDIAN_CV = 0.55
BIOLOGICAL_DROPOUT_RATE = 0.30
