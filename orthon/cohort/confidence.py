"""
Cohort Confidence Metrics

Multiple metrics for clustering confidence, robust to small sample sizes.

Metrics:
- Silhouette Score: Cluster separation (-1 to 1)
- Calinski-Harabasz: Between/within variance ratio (higher = better)
- Gap Statistic: Comparison to null reference (higher = better)
- Bootstrap Stability: Consistency across resamples (0 to 1)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans


@dataclass
class ConfidenceMetrics:
    """Comprehensive clustering confidence metrics."""
    silhouette: float          # -1 to 1, higher = better separation
    calinski_harabasz: float   # 0 to inf, higher = better
    gap_statistic: float       # 0 to inf, higher = better
    gap_std: float             # Standard error of gap
    bootstrap_stability: float # 0 to 1, higher = more stable
    n_samples: int             # Sample size
    n_clusters: int            # Number of clusters

    @property
    def composite_score(self) -> float:
        """
        Composite confidence score (0 to 1).

        Combines metrics with appropriate weighting:
        - Silhouette: 40% (most interpretable)
        - Bootstrap stability: 30% (most reliable for small n)
        - Normalized CH: 20% (good for large n)
        - Gap significance: 10% (reference comparison)
        """
        # Normalize silhouette to 0-1
        sil_norm = (self.silhouette + 1) / 2

        # Normalize CH (cap at 100 for normalization)
        ch_norm = min(self.calinski_harabasz / 100, 1.0)

        # Gap significance (gap > 1 std is significant)
        gap_sig = min(self.gap_statistic / (self.gap_std + 1e-6) / 3, 1.0) if self.gap_std > 0 else 0.5

        # Weighted combination
        return (
            0.40 * sil_norm +
            0.30 * self.bootstrap_stability +
            0.20 * ch_norm +
            0.10 * gap_sig
        )

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        score = self.composite_score
        if score >= 0.75:
            return "strong"
        elif score >= 0.50:
            return "moderate"
        elif score >= 0.25:
            return "weak"
        else:
            return "poor"

    def __repr__(self) -> str:
        return (
            f"ConfidenceMetrics(composite={self.composite_score:.3f} [{self.interpretation}], "
            f"silhouette={self.silhouette:.3f}, bootstrap={self.bootstrap_stability:.3f}, "
            f"n={self.n_samples}, k={self.n_clusters})"
        )


def compute_clustering_confidence(
    X: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 100,
    n_reference: int = 10,
    random_state: int = 42,
) -> ConfidenceMetrics:
    """
    Compute comprehensive clustering confidence metrics.

    Parameters
    ----------
    X : Feature matrix (n_samples, n_features)
    labels : Cluster labels
    n_bootstrap : Number of bootstrap iterations for stability
    n_reference : Number of reference datasets for gap statistic
    random_state : Random seed

    Returns
    -------
    ConfidenceMetrics with all computed metrics
    """
    np.random.seed(random_state)

    n_samples = len(X)
    n_clusters = len(np.unique(labels))

    # Handle edge cases
    if n_samples < 3 or n_clusters < 2:
        return ConfidenceMetrics(
            silhouette=0.0,
            calinski_harabasz=0.0,
            gap_statistic=0.0,
            gap_std=0.0,
            bootstrap_stability=0.0,
            n_samples=n_samples,
            n_clusters=n_clusters,
        )

    # 1. Silhouette Score
    try:
        silhouette = silhouette_score(X, labels)
    except Exception:
        silhouette = 0.0

    # 2. Calinski-Harabasz Score
    try:
        ch_score = calinski_harabasz_score(X, labels)
    except Exception:
        ch_score = 0.0

    # 3. Gap Statistic
    gap, gap_std = _compute_gap_statistic(X, labels, n_clusters, n_reference, random_state)

    # 4. Bootstrap Stability
    stability = _compute_bootstrap_stability(X, labels, n_clusters, n_bootstrap, random_state)

    return ConfidenceMetrics(
        silhouette=silhouette,
        calinski_harabasz=ch_score,
        gap_statistic=gap,
        gap_std=gap_std,
        bootstrap_stability=stability,
        n_samples=n_samples,
        n_clusters=n_clusters,
    )


def _compute_gap_statistic(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_reference: int,
    random_state: int,
) -> Tuple[float, float]:
    """
    Compute gap statistic comparing observed clustering to uniform reference.

    Gap = E[log(W_ref)] - log(W_obs)

    Higher gap = clustering captures more structure than random.
    """
    # Compute observed within-cluster dispersion
    w_obs = _compute_within_dispersion(X, labels)
    log_w_obs = np.log(w_obs + 1e-10)

    # Generate reference datasets and compute their dispersions
    log_w_refs = []

    for i in range(n_reference):
        # Uniform reference within bounding box
        X_ref = np.random.uniform(
            X.min(axis=0),
            X.max(axis=0),
            size=X.shape
        )

        # Cluster reference data
        km = KMeans(n_clusters=n_clusters, random_state=random_state + i, n_init=3)
        ref_labels = km.fit_predict(X_ref)

        w_ref = _compute_within_dispersion(X_ref, ref_labels)
        log_w_refs.append(np.log(w_ref + 1e-10))

    # Gap = E[log(W_ref)] - log(W_obs)
    gap = np.mean(log_w_refs) - log_w_obs
    gap_std = np.std(log_w_refs) * np.sqrt(1 + 1/n_reference)

    return gap, gap_std


def _compute_within_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute total within-cluster sum of squares."""
    dispersion = 0.0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            dispersion += np.sum((cluster_points - centroid) ** 2)
    return dispersion


def _compute_bootstrap_stability(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    n_bootstrap: int,
    random_state: int,
) -> float:
    """
    Compute bootstrap stability of clustering.

    Measures how consistently points are assigned to the same cluster
    across bootstrap resamples.

    Returns value between 0 (unstable) and 1 (perfectly stable).
    """
    n_samples = len(X)

    # For very small samples, use leave-one-out instead
    if n_samples < 10:
        return _compute_loo_stability(X, labels, n_clusters, random_state)

    # Count how often each pair of points is clustered together
    pair_counts = np.zeros((n_samples, n_samples))
    pair_totals = np.zeros((n_samples, n_samples))

    np.random.seed(random_state)

    for b in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]

        # Re-cluster
        km = KMeans(n_clusters=n_clusters, random_state=random_state + b, n_init=3)
        boot_labels = km.fit_predict(X_boot)

        # Map back to original indices
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                if i < j:
                    pair_totals[idx_i, idx_j] += 1
                    if boot_labels[i] == boot_labels[j]:
                        pair_counts[idx_i, idx_j] += 1

    # Compute stability as agreement with original clustering
    stability_sum = 0.0
    stability_count = 0

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if pair_totals[i, j] > 0:
                boot_same = pair_counts[i, j] / pair_totals[i, j]
                orig_same = 1.0 if labels[i] == labels[j] else 0.0
                # Agreement: both say same cluster, or both say different
                agreement = 1.0 - abs(boot_same - orig_same)
                stability_sum += agreement
                stability_count += 1

    return stability_sum / stability_count if stability_count > 0 else 0.0


def _compute_loo_stability(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> float:
    """
    Leave-one-out stability for very small samples.

    For each point, remove it, re-cluster, and check if assignment is consistent.
    """
    n_samples = len(X)
    agreements = []

    for i in range(n_samples):
        # Leave one out
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        X_loo = X[mask]

        if len(X_loo) < n_clusters:
            continue

        # Re-cluster
        km = KMeans(n_clusters=min(n_clusters, len(X_loo)), random_state=random_state, n_init=3)
        loo_labels = km.fit_predict(X_loo)

        # Predict cluster for left-out point
        distances = np.linalg.norm(km.cluster_centers_ - X[i], axis=1)
        predicted_cluster = np.argmin(distances)

        # Map original labels to LOO labels by majority vote
        orig_labels_loo = labels[mask]
        label_map = {}
        for k in range(n_clusters):
            if np.sum(orig_labels_loo == k) > 0:
                # Most common LOO label for this original cluster
                loo_for_k = loo_labels[orig_labels_loo == k]
                if len(loo_for_k) > 0:
                    label_map[k] = np.bincount(loo_for_k).argmax()

        # Check agreement
        orig_label = labels[i]
        if orig_label in label_map:
            agreement = 1.0 if predicted_cluster == label_map[orig_label] else 0.0
        else:
            agreement = 0.5  # Can't determine

        agreements.append(agreement)

    return np.mean(agreements) if agreements else 0.0


def find_optimal_k(
    X: np.ndarray,
    max_k: int = 10,
    method: str = "gap",
    random_state: int = 42,
) -> Tuple[int, ConfidenceMetrics]:
    """
    Find optimal number of clusters using specified method.

    Parameters
    ----------
    X : Feature matrix
    max_k : Maximum clusters to try
    method : "gap", "silhouette", or "elbow"
    random_state : Random seed

    Returns
    -------
    Tuple of (optimal_k, confidence_metrics)
    """
    n_samples = len(X)
    max_k = min(max_k, n_samples - 1)

    if max_k < 2:
        labels = np.zeros(n_samples, dtype=int)
        return 1, compute_clustering_confidence(X, labels, random_state=random_state)

    best_k = 2
    best_score = -np.inf
    best_metrics = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)

        metrics = compute_clustering_confidence(X, labels, random_state=random_state)

        if method == "gap":
            score = metrics.gap_statistic - metrics.gap_std
        elif method == "silhouette":
            score = metrics.silhouette
        else:  # composite
            score = metrics.composite_score

        if score > best_score:
            best_score = score
            best_k = k
            best_metrics = metrics

    return best_k, best_metrics
