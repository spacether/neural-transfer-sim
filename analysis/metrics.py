"""Transfer quality metrics for comparing neural states across brains.

All metrics compare the receiver's state after signal injection against the
receiver's natural state for the same odor (what it would produce if it smelled
the odor directly).  Each metric answers a different question:

- Correlation (r): how well does the transferred activity track the natural response?
- MSE: how close is the receiver's state to its natural response?
- RSA: is the geometry of odor representations preserved?
- Odor-type decoding: can we tell attractive from repellent after transfer?
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


ALL_NEURONS = ("AWA", "AWC_L", "AWC_R", "AWB", "ADL", "AIY", "AIB", "AVA", "AVB")


def state_to_vec(state: Dict[str, float]) -> np.ndarray:
    """Convert a neuron state dict to a fixed-length vector over ALL_NEURONS."""
    return np.array([state.get(n, 0.0) for n in ALL_NEURONS])


def correlation(
    actual_states: List[Dict[str, float]],
    expected_states: List[Dict[str, float]],
) -> float:
    """Pearson correlation between transferred and natural activity vectors.

    Concatenates all neuron activations across all trials into one long vector
    per condition and returns the Pearson r.  1.0 = perfect linear tracking.
    """
    vecs_a = np.array([state_to_vec(s) for s in actual_states]).ravel()
    vecs_e = np.array([state_to_vec(s) for s in expected_states]).ravel()

    if np.std(vecs_a) < 1e-12 or np.std(vecs_e) < 1e-12:
        return 1.0 if np.allclose(vecs_a, vecs_e) else 0.0

    return float(np.corrcoef(vecs_a, vecs_e)[0, 1])


def mean_mse(
    actual_states: List[Dict[str, float]],
    expected_states: List[Dict[str, float]],
) -> float:
    """Mean squared error between paired states, averaged over all neurons and samples."""
    vecs_a = np.array([state_to_vec(s) for s in actual_states])
    vecs_e = np.array([state_to_vec(s) for s in expected_states])
    return float(np.mean((vecs_a - vecs_e) ** 2))


def representational_similarity(
    natural_states: List[Dict[str, float]],
    transferred_states: List[Dict[str, float]],
) -> float:
    """Spearman correlation between pairwise distance matrices (RSA).

    Measures whether the geometry of odor representations is preserved.
    1.0 = perfect preservation, 0.0 = no relationship.
    Requires at least 3 stimuli.
    """
    if len(natural_states) < 3:
        return float("nan")

    nat_dists = pdist(np.array([state_to_vec(s) for s in natural_states]))
    trans_dists = pdist(np.array([state_to_vec(s) for s in transferred_states]))

    if np.std(nat_dists) < 1e-12 or np.std(trans_dists) < 1e-12:
        return 1.0 if np.allclose(nat_dists, trans_dists) else 0.0

    corr, _ = spearmanr(nat_dists, trans_dists)
    return float(corr)


def odor_type_decoding(
    odor_types: List[str],
    natural_states: List[Dict[str, float]],
    transferred_states: List[Dict[str, float]],
) -> float:
    """Classify each transferred state as attractive or repellent.

    Computes the centroid of natural-response vectors for each odor type,
    then assigns each transferred state to the nearest centroid.
    Returns fraction correctly classified.
    """
    nat_vecs = np.array([state_to_vec(s) for s in natural_states])
    trans_vecs = np.array([state_to_vec(s) for s in transferred_states])

    categories = sorted(set(odor_types))
    centroids = {}
    for cat in categories:
        mask = [i for i, t in enumerate(odor_types) if t == cat]
        centroids[cat] = nat_vecs[mask].mean(axis=0)

    correct = 0
    for i, tv in enumerate(trans_vecs):
        dists = {cat: np.linalg.norm(tv - c) for cat, c in centroids.items()}
        predicted = min(dists, key=dists.get)
        if predicted == odor_types[i]:
            correct += 1

    return correct / max(1, len(transferred_states))
