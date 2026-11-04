"""Cross-organism scaling comparison."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _worm_metrics(seeds: int = 5, jitter: float = 0.75) -> Dict:
    """Run worm transfer metrics using the Cook2019 model."""
    from models.cook2019_connectome import OLFACTORY_RECEPTORS
    from experiments.c_elegans.receptor_to_receptor_broadcast.experiment import (
        run_receptor_to_receptor_sweep,
    )

    n_receptor_classes = len(set(
        cls for neurons in OLFACTORY_RECEPTORS.values() for cls in neurons
    ))

    results = run_receptor_to_receptor_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
        model_name="cook2019",
    )
    beh_match = results[0]["transfer_rate"] if results else 0.0

    return {
        "organism": "C. elegans",
        "receptor_types": n_receptor_classes,
        "total_receptors": n_receptor_classes,
        "glomeruli": 0,
        "kc_equivalent": 0,
        "bottleneck": "none",
        "signal_mode": "graded",
        "broadcast_behavioral_match": beh_match,
    }


def _larva_metrics(seeds: int = 5, jitter: float = 0.75) -> Dict:
    """Run larva transfer metrics."""
    from models.drosophila_larva import DrosophilaLarvaModel, ORN_TYPES, N_KC
    from models.larva_variants import LarvaVariantGenerator
    from experiments.drosophila_larva.broadcast.experiment import run_broadcast_sweep

    results = run_broadcast_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
    )
    beh_match = results[0]["transfer_rate"] if results else 0.0

    return {
        "organism": "D. larva",
        "receptor_types": len(ORN_TYPES),
        "total_receptors": len(ORN_TYPES),
        "glomeruli": len(ORN_TYPES),
        "kc_equivalent": N_KC,
        "bottleneck": "glomerular (1:1)",
        "signal_mode": "spiking",
        "broadcast_behavioral_match": beh_match,
    }


def _adult_metrics(seeds: int = 5, jitter: float = 0.75) -> Dict:
    """Run adult fly transfer metrics."""
    from models.drosophila_adult import (
        DrosophilaAdultModel, ORN_TYPES, N_ORN_TYPES,
        N_TOTAL_ORNS, N_PN, N_KC,
    )
    from experiments.drosophila_adult.broadcast.experiment import run_broadcast_sweep

    results = run_broadcast_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
    )
    beh_match = results[0]["transfer_rate"] if results else 0.0

    return {
        "organism": "D. adult",
        "receptor_types": N_ORN_TYPES,
        "total_receptors": N_TOTAL_ORNS,
        "glomeruli": N_PN,
        "kc_equivalent": N_KC,
        "bottleneck": "glomerular (50:1)",
        "signal_mode": "spiking",
        "broadcast_behavioral_match": beh_match,
    }


def _zebrafish_metrics(seeds: int = 5, jitter: float = 0.75) -> Dict:
    """Run zebrafish transfer metrics."""
    from models.zebrafish import (
        ZebrafishModel, N_OR_TYPES, N_TOTAL_ORNS, N_GLOMERULI, N_DP,
    )
    from experiments.zebrafish.broadcast.experiment import run_broadcast_sweep

    results = run_broadcast_sweep(
        jitter_levels=[jitter],
        seeds_per_level=seeds,
    )
    beh_match = results[0]["transfer_rate"] if results else 0.0

    return {
        "organism": "Zebrafish",
        "receptor_types": N_OR_TYPES,
        "total_receptors": N_TOTAL_ORNS,
        "glomeruli": N_GLOMERULI,
        "kc_equivalent": N_DP,
        "bottleneck": "glomerular (3:1)",
        "signal_mode": "spiking",
        "broadcast_behavioral_match": beh_match,
    }


def run_scaling_comparison(seeds: int = 5, jitter: float = 0.75) -> List[Dict]:
    """Compute transfer metrics for all four organisms."""
    return [
        _worm_metrics(seeds=seeds, jitter=jitter),
        _larva_metrics(seeds=seeds, jitter=jitter),
        _adult_metrics(seeds=seeds, jitter=jitter),
        _zebrafish_metrics(seeds=seeds, jitter=jitter),
    ]


def estimate_channel_requirements(results: List[Dict]) -> List[Dict]:
    """Estimate minimum channel counts from the scaling data.

    Uses the channel sweep results if available, otherwise estimates
    from receptor type counts.
    """
    enriched = []
    for r in results:
        row = dict(r)
        n_types = r["receptor_types"]
        row["min_channels_90"] = max(1, int(np.ceil(n_types * 0.7)))
        row["calibration_samples"] = max(10, n_types * 3)
        enriched.append(row)
    return enriched
