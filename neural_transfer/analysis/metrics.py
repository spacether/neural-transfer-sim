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


stimulus_identification = odor_identification


def _autocorr_features(
    timeseries: np.ndarray, dt_ms: float, lags_ms: tuple = (15, 30, 35, 40, 50, 70),
) -> np.ndarray:
    """Per-neuron autocorrelation at IPI-relevant lags, flattened to a feature vector."""
    n_t, n_n = timeseries.shape
    lags = [max(1, int(l / dt_ms)) for l in lags_ms]
    feat = np.zeros(len(lags) * n_n)
    for li, lag in enumerate(lags):
        if lag >= n_t:
            continue
        a = timeseries[:n_t - lag]
        b = timeseries[lag:]
        a_c = a - a.mean(axis=0)
        b_c = b - b.mean(axis=0)
        num = (a_c * b_c).mean(axis=0)
        denom = a_c.std(axis=0) * b_c.std(axis=0) + 1e-12
        feat[li * n_n:(li + 1) * n_n] = num / denom
    return feat


def temporal_stimulus_identification(
    labels: List[str],
    natural_timeseries: List[np.ndarray],
    transferred_timeseries: List[np.ndarray],
    dt_ms: float = 0.5,
) -> float:
    """N-way identification using temporal autocorrelation features.

    Extracts per-neuron autocorrelation at IPI-relevant lags (15–70 ms),
    capturing the temporal pattern structure that time-averaging destroys.
    Uses centroid-based nearest-neighbor classification on these features.
    """
    unique_labels = sorted(set(labels))

    nat_feats = [_autocorr_features(ts, dt_ms) for ts in natural_timeseries]
    trans_feats = [_autocorr_features(ts, dt_ms) for ts in transferred_timeseries]

    centroids: Dict[str, np.ndarray] = {}
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        centroids[label] = np.mean([nat_feats[i] for i in indices], axis=0)

    correct = 0
    for i, feat in enumerate(trans_feats):
        dists = {
            label: np.sum((feat - centroid) ** 2)
            for label, centroid in centroids.items()
        }
        predicted = min(dists, key=dists.get)
        if predicted == labels[i]:
            correct += 1

    return correct / max(1, len(transferred_timeseries))


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a list of per-seed metric values.

    Returns (mean, ci_low, ci_high).
    """
    arr = np.array(values)
    if len(arr) < 2:
        m = float(arr.mean()) if len(arr) == 1 else 0.0
        return m, m, m

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(arr.mean()), lo, hi


def aggregate_with_ci(
    seed_metrics: List[Dict[str, float]],
    metric_keys: tuple[str, ...] = ("transfer_rate", "correlation", "mse", "rsa", "odor_id"),
    confidence: float = 0.95,
) -> Dict[str, tuple[float, float, float]]:
    """Aggregate per-seed metrics into mean + CI for each key.

    Returns {metric_name: (mean, ci_low, ci_high)}.
    """
    result = {}
    for key in metric_keys:
        values = [m[key] for m in seed_metrics if key in m]
        if values:
            result[key] = bootstrap_ci(values, confidence=confidence)
        else:
            result[key] = (0.0, 0.0, 0.0)
    return result
