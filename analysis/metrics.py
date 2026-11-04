"""Transfer quality metrics for comparing neural states across brains.

All metrics compare the receiver's state after signal injection against the
receiver's natural state for the same odor (what it would produce if it smelled
the odor directly).  Each metric answers a different question:

- Correlation (r): how well does the transferred activity track the natural response?
- MSE: how close is the receiver's state to its natural response?
- RSA: is the geometry of odor representations preserved?
- Odor identification: can we tell which specific odor was transferred (N-way)?
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def state_to_vec(state: Dict[str, float], keys: tuple | None = None) -> np.ndarray:
    """Convert a neuron state dict to a fixed-length vector.

    If *keys* is given, the vector follows that order.  Otherwise,
    the dict's sorted keys are used — which is deterministic across
    states that share the same neuron set.
    """
    if keys is None:
        keys = tuple(sorted(state.keys()))
    return np.array([state.get(n, 0.0) for n in keys])


def _shared_keys(states_a: List[Dict[str, float]], states_b: List[Dict[str, float]]) -> tuple:
    """Return a sorted tuple of all neuron names present in either list."""
    keys: set = set()
    for s in states_a:
        keys.update(s.keys())
    for s in states_b:
        keys.update(s.keys())
    return tuple(sorted(keys))


def correlation(
    actual_states: List[Dict[str, float]],
    expected_states: List[Dict[str, float]],
) -> float:
    """Pearson correlation between transferred and natural activity vectors.

    Concatenates all neuron activations across all trials into one long vector
    per condition and returns the Pearson r.  1.0 = perfect linear tracking.
    """
    keys = _shared_keys(actual_states, expected_states)
    vecs_a = np.array([state_to_vec(s, keys) for s in actual_states]).ravel()
    vecs_e = np.array([state_to_vec(s, keys) for s in expected_states]).ravel()

    if np.std(vecs_a) < 1e-12 or np.std(vecs_e) < 1e-12:
        return 1.0 if np.allclose(vecs_a, vecs_e) else 0.0

    return float(np.corrcoef(vecs_a, vecs_e)[0, 1])


def mean_mse(
    actual_states: List[Dict[str, float]],
    expected_states: List[Dict[str, float]],
) -> float:
    """Mean squared error between paired states, averaged over all neurons and samples."""
    keys = _shared_keys(actual_states, expected_states)
    vecs_a = np.array([state_to_vec(s, keys) for s in actual_states])
    vecs_e = np.array([state_to_vec(s, keys) for s in expected_states])
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

    keys = _shared_keys(natural_states, transferred_states)
    nat_dists = pdist(np.array([state_to_vec(s, keys) for s in natural_states]))
    trans_dists = pdist(np.array([state_to_vec(s, keys) for s in transferred_states]))

    if np.std(nat_dists) < 1e-12 or np.std(trans_dists) < 1e-12:
        return 1.0 if np.allclose(nat_dists, trans_dists) else 0.0

    corr, _ = spearmanr(nat_dists, trans_dists)
    return float(corr)


def odor_identification(
    odor_labels: List[str],
    natural_states: List[Dict[str, float]],
    transferred_states: List[Dict[str, float]],
) -> float:
    """Identify which specific odor the sender detected (N-way classification).

    Computes the centroid of natural-response vectors for each distinct odor
    label, then assigns each transferred state to the nearest centroid.
    Returns fraction correctly identified — tests whether the transfer
    preserves enough structure to distinguish similar odors within the
    same valence category.
    """
    keys = _shared_keys(natural_states, transferred_states)
    nat_vecs = np.array([state_to_vec(s, keys) for s in natural_states])
    trans_vecs = np.array([state_to_vec(s, keys) for s in transferred_states])

    labels = sorted(set(odor_labels))
    centroids = {}
    for label in labels:
        mask = [i for i, l in enumerate(odor_labels) if l == label]
        centroids[label] = nat_vecs[mask].mean(axis=0)

    correct = 0
    for i, tv in enumerate(trans_vecs):
        dists = {label: np.linalg.norm(tv - c) for label, c in centroids.items()}
        predicted = min(dists, key=dists.get)
        if predicted == odor_labels[i]:
            correct += 1

    return correct / max(1, len(transferred_states))
