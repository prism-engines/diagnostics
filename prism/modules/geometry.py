"""
prism/modules/geometry.py - Geometry layer computation

Pure functions, no I/O. Computes pairwise relationships between signals.

Features include:
- PCA (principal components, explained variance)
- Mutual information
- Correlation matrices
- Clustering metrics
- MST (minimum spanning tree) properties
- LOF (local outlier factor)
- Copula dependence
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform


def compute_geometry_features(
    df: pl.DataFrame,
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    signal_col: str = "signal_id",
    value_col: str = "value",
    n_components: int = 5,
) -> pl.DataFrame:
    """
    Compute geometry features for each entity at each timestamp.

    Args:
        df: Input DataFrame (can be raw observations or vector features)
        entity_col: Entity identifier column
        time_col: Time/cycle column
        signal_col: Signal identifier column (if raw data)
        value_col: Value column (if raw data)
        n_components: PCA components

    Returns:
        DataFrame with geometry features per (entity, timestamp)
    """
    # Check if this is vector output (wide format) or raw observations (long format)
    if signal_col in df.columns and value_col in df.columns:
        return _compute_from_observations(
            df, entity_col, time_col, signal_col, value_col, n_components
        )
    else:
        return _compute_from_vector(df, entity_col, time_col, n_components)


def _compute_from_observations(
    df: pl.DataFrame,
    entity_col: str,
    time_col: str,
    signal_col: str,
    value_col: str,
    n_components: int,
) -> pl.DataFrame:
    """Compute geometry from raw observations (long format)."""
    results = []

    entities = df[entity_col].unique().to_list()

    for entity_id in entities:
        entity_df = df.filter(pl.col(entity_col) == entity_id)

        # Pivot to wide format: rows=timestamps, cols=signals
        pivot_df = entity_df.pivot(
            index=time_col,
            columns=signal_col,
            values=value_col,
        ).sort(time_col)

        if len(pivot_df) < 10:
            continue

        # Get signal columns (exclude time)
        signal_cols = [c for c in pivot_df.columns if c != time_col]
        if len(signal_cols) < 3:
            continue

        # Extract matrix
        X = pivot_df.select(signal_cols).to_numpy()
        times = pivot_df[time_col].to_numpy()

        # Remove rows with NaN
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        times = times[mask]

        if len(X) < 10:
            continue

        # Compute geometry at each timestamp (using rolling window)
        window_size = min(50, len(X) // 2)
        stride = max(1, window_size // 5)

        for i in range(0, len(X) - window_size + 1, stride):
            X_window = X[i:i + window_size]
            window_time = times[i + window_size - 1]

            metrics = _compute_geometry_metrics(X_window, n_components)

            results.append({
                entity_col: entity_id,
                time_col: window_time,
                **metrics,
            })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def _compute_from_vector(
    df: pl.DataFrame,
    entity_col: str,
    time_col: str,
    n_components: int,
) -> pl.DataFrame:
    """Compute geometry from vector features (already wide format)."""
    results = []

    # Get feature columns (numeric, not entity/time)
    feature_cols = [
        c for c in df.columns
        if c not in [entity_col, time_col, 'signal_id', 'window_start', 'window_size']
        and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
    ]

    if len(feature_cols) < 3:
        # Not enough features, return basic stats
        return df.select([entity_col, time_col])

    entities = df[entity_col].unique().to_list()

    for entity_id in entities:
        entity_df = df.filter(pl.col(entity_col) == entity_id).sort(time_col)

        if len(entity_df) < 10:
            continue

        X = entity_df.select(feature_cols).to_numpy()
        times = entity_df[time_col].to_numpy()

        # Remove NaN rows
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        times = times[mask]

        if len(X) < 10:
            continue

        # Compute geometry per timestamp (or window)
        for i, t in enumerate(times):
            # Use all data up to this point (expanding window)
            X_window = X[:i+1] if i >= 10 else X[:11]

            metrics = _compute_geometry_metrics(X_window, n_components)

            results.append({
                entity_col: entity_id,
                time_col: t,
                **metrics,
            })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def _compute_geometry_metrics(X: np.ndarray, n_components: int = 5) -> Dict[str, float]:
    """Compute geometry metrics for a data matrix."""
    metrics = {}
    n_samples, n_features = X.shape

    # Standardize
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)

    # PCA
    try:
        n_comp = min(n_components, n_features, n_samples - 1)
        pca = PCA(n_components=n_comp)
        pca.fit(X_std)

        for i, var in enumerate(pca.explained_variance_ratio_):
            metrics[f'pca_variance_{i+1}'] = float(var)

        metrics['pca_cumulative_variance'] = float(np.sum(pca.explained_variance_ratio_))
    except Exception:
        for i in range(n_components):
            metrics[f'pca_variance_{i+1}'] = np.nan
        metrics['pca_cumulative_variance'] = np.nan

    # Correlation matrix stats
    try:
        if n_features > 1:
            corr = np.corrcoef(X_std.T)
            # Mean absolute correlation (excluding diagonal)
            mask = ~np.eye(n_features, dtype=bool)
            metrics['mean_correlation'] = float(np.mean(np.abs(corr[mask])))
            metrics['max_correlation'] = float(np.max(np.abs(corr[mask])))
            metrics['min_correlation'] = float(np.min(np.abs(corr[mask])))
        else:
            metrics['mean_correlation'] = np.nan
            metrics['max_correlation'] = np.nan
            metrics['min_correlation'] = np.nan
    except Exception:
        metrics['mean_correlation'] = np.nan
        metrics['max_correlation'] = np.nan
        metrics['min_correlation'] = np.nan

    # Mutual information (approximated via correlation for speed)
    try:
        if n_features > 1:
            # Use Spearman for nonlinear relationships
            mi_estimates = []
            for i in range(min(n_features, 10)):
                for j in range(i + 1, min(n_features, 10)):
                    rho, _ = spearmanr(X_std[:, i], X_std[:, j])
                    # MI approximation: -0.5 * log(1 - rho^2)
                    if abs(rho) < 0.999:
                        mi = -0.5 * np.log(1 - rho**2)
                        mi_estimates.append(mi)

            if mi_estimates:
                metrics['mi_mean'] = float(np.mean(mi_estimates))
                metrics['mi_max'] = float(np.max(mi_estimates))
            else:
                metrics['mi_mean'] = np.nan
                metrics['mi_max'] = np.nan
        else:
            metrics['mi_mean'] = np.nan
            metrics['mi_max'] = np.nan
    except Exception:
        metrics['mi_mean'] = np.nan
        metrics['mi_max'] = np.nan

    # Clustering (if enough samples)
    try:
        if n_samples >= 10:
            # Try k=2 to k=5, pick best silhouette
            best_score = -1
            best_k = 2

            for k in range(2, min(6, n_samples // 2)):
                km = KMeans(n_clusters=k, n_init=3, random_state=42)
                labels = km.fit_predict(X_std)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X_std, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

            metrics['clustering_silhouette'] = float(best_score)
            metrics['clustering_k'] = float(best_k)
        else:
            metrics['clustering_silhouette'] = np.nan
            metrics['clustering_k'] = np.nan
    except Exception:
        metrics['clustering_silhouette'] = np.nan
        metrics['clustering_k'] = np.nan

    # LOF (Local Outlier Factor)
    try:
        if n_samples >= 20:
            lof = LocalOutlierFactor(n_neighbors=min(20, n_samples - 1))
            scores = lof.fit_predict(X_std)
            metrics['lof_outlier_fraction'] = float(np.mean(scores == -1))
            metrics['lof_mean_score'] = float(np.mean(-lof.negative_outlier_factor_))
        else:
            metrics['lof_outlier_fraction'] = np.nan
            metrics['lof_mean_score'] = np.nan
    except Exception:
        metrics['lof_outlier_fraction'] = np.nan
        metrics['lof_mean_score'] = np.nan

    # Distance matrix stats
    try:
        if n_samples >= 3:
            distances = pdist(X_std, metric='euclidean')
            metrics['mean_distance'] = float(np.mean(distances))
            metrics['max_distance'] = float(np.max(distances))
            metrics['distance_std'] = float(np.std(distances))
        else:
            metrics['mean_distance'] = np.nan
            metrics['max_distance'] = np.nan
            metrics['distance_std'] = np.nan
    except Exception:
        metrics['mean_distance'] = np.nan
        metrics['max_distance'] = np.nan
        metrics['distance_std'] = np.nan

    return metrics
