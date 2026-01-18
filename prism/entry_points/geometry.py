"""
PRISM Geometry Runner - Windowed by Design
===========================================

Orchestrates geometry computation by calling canonical engines + inline modules.

This runner is a PURE ORCHESTRATOR:
- Fetches data from Parquet files
- Calls canonical GEOMETRY engines (9 from prism.engines registry)
- Calls inline MODULES (modes, wavelet_microscope from prism.modules)
- Iterates over date ranges at configured strides
- Stores to Parquet

NO COMPUTATION LOGIC - all delegated to canonical engines and modules.

GEOMETRY ENGINES (9 canonical):
    - distance:            Euclidean/Mahalanobis/cosine distance matrices
    - pca:                 Principal components (dimensionality)
    - clustering:          K-means, hierarchical grouping
    - mutual_information:  Nonlinear dependence
    - copula:              Tail dependence
    - mst:                 Minimum spanning tree (network structure)
    - lof:                 Local outlier factor
    - convex_hull:         Phase space volume
    - barycenter:          Conviction-weighted center of mass across timescales

INLINE MODULES (from prism.modules):
    - modes:               Behavioral mode discovery from Laplace signatures
    - wavelet_microscope:  Frequency-band degradation detection

Window/Stride Configuration (from config/stride.yaml default_tiers):
    Uses get_default_tiers() - typically anchor, bridge, scout
    NO MICRO - 21d/1d is too expensive for geometry pairwise

"When five year olds run around - normal noise. When adults start running - regime change."

Output Schema:
    geometry.cohorts    - Cohort-level structural metrics
    geometry.pairs      - Pairwise geometric relationships

Usage:
    # Full production run (all cohorts, all tiers)
    python -m prism.entry_points.indicator_geometry

    # Force clear progress tracker
    python -m prism.entry_points.indicator_geometry --force

    # Testing mode - filter to specific cohort
    python -m prism.entry_points.indicator_geometry --filter-cohort macro --testing

    # Testing mode - specific date range
    python -m prism.entry_points.indicator_geometry --dates 2023-01-01:2023-12-31 --testing
"""

import argparse
import gc
import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
import warnings

from prism.db.parquet_store import ensure_directories, get_parquet_path
from prism.db.polars_io import (
    read_parquet,
    read_parquet_smart,
    get_file_size_mb,
    upsert_parquet,
    write_parquet_atomic,
)
from prism.utils.memory import force_gc, get_memory_usage_mb
from prism.db.scratch import TempParquet, merge_to_table
from prism.engines.utils.parallel import (
    WorkerAssignment,
    divide_by_count,
    generate_temp_path,
    run_workers,
)

# Canonical engines from registry (9 geometry engines)
from prism.engines import (
    DistanceEngine,
    PCAEngine,
    ClusteringEngine,
    MutualInformationEngine,
    CopulaEngine,
    MSTEngine,
    LOFEngine,
    ConvexHullEngine,
    BarycenterEngine,
    compute_barycenter,
)

# Window/stride configuration
from prism.utils.stride import (
    load_stride_config,
    get_window_dates,
    get_barycenter_weights,
    get_default_tiers,
    get_drilldown_tiers,
    WINDOWS,
)

# Fast config access (Python dicts)
from prism.config.windows import get_window_weight

# Bisection analysis
from prism.utils import bisection

# Inline modules for mode discovery and wavelet analysis
from prism.modules.modes import (
    discover_modes,
    compute_affinity_weighted_features,
)
from prism.modules.wavelet_microscope import (
    run_wavelet_microscope,
    extract_wavelet_features,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA FETCHING (orchestration)
# =============================================================================

def get_curated_indicators() -> Optional[set]:
    """
    Get set of curated (non-redundant) indicators from filter_deep output.

    Returns None if no filter output exists (run all indicators).
    Returns set of indicator IDs if filter output exists.
    """
    curated_path = Path('data/filter/deep_curated.parquet')
    if not curated_path.exists():
        return None

    try:
        df = pl.read_parquet(curated_path)
        return set(df['indicator_id'].to_list())
    except Exception as e:
        logger.warning(f"Could not read curated indicators: {e}")
        return None


def get_redundant_indicators() -> set:
    """Get set of redundant indicators to exclude."""
    redundant_path = Path('data/filter/deep_redundant.parquet')
    if not redundant_path.exists():
        return set()

    try:
        df = pl.read_parquet(redundant_path)
        return set(df['indicator_id'].to_list())
    except Exception as e:
        logger.warning(f"Could not read redundant indicators: {e}")
        return set()


def get_date_range() -> Tuple[date, date]:
    """Get available date range from raw observations (lazy query)."""
    # Use lazy scan - only reads metadata, not full file
    lf = pl.scan_parquet(get_parquet_path('raw', 'observations'))
    result = lf.select([
        pl.col('obs_date').min().alias('min_date'),
        pl.col('obs_date').max().alias('max_date'),
    ]).collect()
    min_date = result['min_date'][0]
    max_date = result['max_date'][0]
    # Convert to Python date if needed
    if hasattr(min_date, 'date'):
        min_date = min_date.date()
    if hasattr(max_date, 'date'):
        max_date = max_date.date()
    return min_date, max_date


# Cache redundant indicators at module load
_REDUNDANT_INDICATORS: Optional[set] = None


def get_cohort_indicators(cohort: str) -> List[str]:
    """
    Fetch all indicators in a cohort, excluding redundant ones.

    Uses filter_deep output if available to exclude redundant indicators.
    """
    global _REDUNDANT_INDICATORS

    # Load redundant indicators once
    if _REDUNDANT_INDICATORS is None:
        _REDUNDANT_INDICATORS = get_redundant_indicators()
        if _REDUNDANT_INDICATORS:
            logger.info(f"Excluding {len(_REDUNDANT_INDICATORS)} redundant indicators from filter_deep")

    cohort_members = pl.read_parquet(get_parquet_path('config', 'cohort_members'))
    # Handle both 'cohort_id' and 'cohort' column names
    cohort_col = 'cohort_id' if 'cohort_id' in cohort_members.columns else 'cohort'
    indicators = cohort_members.filter(
        pl.col(cohort_col) == cohort
    ).select('indicator_id').sort('indicator_id').to_series().to_list()

    # Filter out redundant indicators
    if _REDUNDANT_INDICATORS:
        indicators = [ind for ind in indicators if ind not in _REDUNDANT_INDICATORS]

    return indicators


def get_all_cohorts() -> List[str]:
    """Get list of all cohorts."""
    cohort_members = pl.read_parquet(get_parquet_path('config', 'cohort_members'))
    # Handle both 'cohort_id' and 'cohort' column names
    cohort_col = 'cohort_id' if 'cohort_id' in cohort_members.columns else 'cohort'
    return cohort_members.select(cohort_col).unique().sort(cohort_col).to_series().to_list()


def get_cohort_data_matrix(
    cohort: str,
    window_end: date,
    window_days: int
) -> pd.DataFrame:
    """
    Fetch cohort data as a matrix (rows=time, cols=indicators).

    Returns DataFrame with DateTimeIndex and indicator columns.
    """
    window_start = window_end - timedelta(days=window_days)

    # Get cohort indicators
    indicators = get_cohort_indicators(cohort)

    # Lazy scan with filter pushdown (memory efficient)
    filtered = (
        pl.scan_parquet(get_parquet_path('raw', 'observations'))
        .filter(
            (pl.col('indicator_id').is_in(indicators)) &
            (pl.col('obs_date') >= window_start) &
            (pl.col('obs_date') <= window_end)
        )
        .select(['obs_date', 'indicator_id', 'value'])
        .collect()
    )

    if filtered.is_empty():
        return pd.DataFrame()

    # Deduplicate: take last value for each (indicator_id, obs_date) pair
    filtered = filtered.group_by(['indicator_id', 'obs_date']).agg(
        pl.col('value').last()
    )

    # Pivot to matrix format using Polars
    pivoted = filtered.pivot(
        on='indicator_id',
        index='obs_date',
        values='value'
    ).sort('obs_date')

    # Drop rows with any null (engines need complete data)
    pivoted = pivoted.drop_nulls()

    if pivoted.is_empty():
        return pd.DataFrame()

    # Convert to pandas DataFrame with date index
    # Use numpy conversion to avoid pyarrow dependency
    dates = pivoted['obs_date'].to_list()
    cols = [c for c in pivoted.columns if c != 'obs_date']
    data = {col: pivoted[col].to_numpy() for col in cols}

    matrix = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    return matrix


def get_pairwise_data(
    ind_a: str,
    ind_b: str,
    window_end: date,
    window_days: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch aligned signal topology for two indicators.

    Returns (series_a, series_b) as numpy arrays.
    """
    window_start = window_end - timedelta(days=window_days)

    # Lazy scan with filter pushdown (memory efficient)
    obs_path = get_parquet_path('raw', 'observations')

    # Get data for both indicators
    data_a = (
        pl.scan_parquet(obs_path)
        .filter(
            (pl.col('indicator_id') == ind_a) &
            (pl.col('obs_date') >= window_start) &
            (pl.col('obs_date') <= window_end)
        )
        .select(['obs_date', 'value'])
        .collect()
        .rename({'value': 'value_a'})
    )

    data_b = (
        pl.scan_parquet(obs_path)
        .filter(
            (pl.col('indicator_id') == ind_b) &
            (pl.col('obs_date') >= window_start) &
            (pl.col('obs_date') <= window_end)
        )
        .select(['obs_date', 'value'])
        .collect()
        .rename({'value': 'value_b'})
    )

    # Join on date
    aligned = data_a.join(data_b, on='obs_date', how='inner')
    aligned = aligned.drop_nulls().sort('obs_date')

    if aligned.is_empty():
        return np.array([]), np.array([])

    return aligned['value_a'].to_numpy(), aligned['value_b'].to_numpy()


def get_indicator_window_vectors(
    cohort: str,
    window_end: date,
    window_sizes: List[int]
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Build window_vectors dict for barycenter computation.

    For each indicator, fetch signal topology for each window size and create
    feature vectors (simple stats as representation).

    Returns:
        Dict mapping indicator_id -> {window_days: feature_vector}
    """
    indicators = get_cohort_indicators(cohort)
    window_vectors = {}

    # Lazy scan with cohort filter pushdown (memory efficient)
    obs_cohort = (
        pl.scan_parquet(get_parquet_path('raw', 'observations'))
        .filter(pl.col('indicator_id').is_in(indicators))
        .collect()
    )

    for indicator_id in indicators:
        vectors = {}

        for window_days in window_sizes:
            window_start = window_end - timedelta(days=window_days)

            df = obs_cohort.filter(
                (pl.col('indicator_id') == indicator_id) &
                (pl.col('obs_date') >= window_start) &
                (pl.col('obs_date') <= window_end)
            ).sort('obs_date')

            if len(df) >= 15:  # Minimum observations
                values = df['value'].to_numpy()
                # Create feature vector from the signal topology
                vectors[window_days] = np.array([
                    np.mean(values),
                    np.std(values),
                    np.min(values),
                    np.max(values),
                    len(values)
                ])

        if vectors:
            window_vectors[indicator_id] = vectors

    return window_vectors


# =============================================================================
# ENGINE ORCHESTRATION (delegation to 9 canonical engines)
# =============================================================================

def compute_cohort_geometry(matrix: pd.DataFrame, cohort: str, window_end: date) -> Dict[str, Any]:
    """
    Orchestrate geometry computation on cohort matrix.

    Calls all 9 canonical GEOMETRY engines and extracts metrics.

    Args:
        matrix: DataFrame (rows=time, cols=indicators)
        cohort: Cohort identifier
        window_end: Window end date

    Returns:
        Dict of cohort-level metrics
    """
    if matrix.empty or matrix.shape[1] < 2:
        return {}

    results = {}
    run_id = f"{cohort}_{window_end}"

    # 1. DISTANCE ENGINE
    try:
        distance_engine = DistanceEngine()
        distance_result = distance_engine.run(matrix, run_id=run_id)

        # Extract summary metrics
        if 'distance_matrix_euclidean' in distance_result:
            dist_matrix = distance_result['distance_matrix_euclidean']
            results['distance_mean'] = float(np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))
            results['distance_std'] = float(np.std(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))
    except Exception as e:
        logger.warning(f"Distance engine failed: {e}")

    # 2. PCA ENGINE
    try:
        pca_engine = PCAEngine()
        # n_components must be <= min(n_samples, n_features)
        n_comp = min(5, matrix.shape[0], matrix.shape[1] - 1)
        if n_comp < 1:
            raise ValueError(f"Insufficient data for PCA: {matrix.shape}")
        pca_result = pca_engine.run(matrix, run_id=run_id, n_components=n_comp)

        results['pca_variance_pc1'] = pca_result.get('variance_pc1', 0)
        results['pca_variance_pc2'] = pca_result.get('variance_pc2', 0)
        results['pca_variance_pc3'] = pca_result.get('variance_pc3', 0)
        results['pca_cumulative_3'] = pca_result.get('cumulative_variance_3', 0)
        results['pca_effective_dim'] = pca_result.get('effective_dimensionality', 0)
    except Exception as e:
        logger.warning(f"PCA engine failed: {e}")

    # 3. CLUSTERING ENGINE
    try:
        # n_clusters must be < n_samples and reasonable for n_features
        n_clusters = min(5, matrix.shape[0] - 1, matrix.shape[1])
        if n_clusters < 2:
            raise ValueError(f"Insufficient data for clustering: {matrix.shape}")
        clustering_engine = ClusteringEngine()
        clustering_result = clustering_engine.run(matrix, run_id=run_id, n_clusters=n_clusters)

        results['clustering_silhouette'] = clustering_result.get('silhouette_score', 0)
        results['clustering_n_clusters'] = n_clusters
    except Exception as e:
        logger.warning(f"Clustering engine failed: {e}")

    # 4. MST ENGINE
    try:
        mst_engine = MSTEngine()
        mst_result = mst_engine.run(matrix, run_id=run_id)

        results['mst_total_weight'] = mst_result.get('total_weight', 0)
        results['mst_avg_degree'] = mst_result.get('average_degree', 0)
    except Exception as e:
        logger.warning(f"MST engine failed: {e}")

    # 5. LOF ENGINE
    try:
        lof_engine = LOFEngine()
        lof_result = lof_engine.run(matrix, run_id=run_id)

        results['lof_mean'] = lof_result.get('mean_lof', 0)
        results['lof_n_outliers'] = lof_result.get('n_outliers', 0)
    except Exception as e:
        logger.warning(f"LOF engine failed: {e}")

    # 6. CONVEX HULL ENGINE
    try:
        hull_engine = ConvexHullEngine()
        hull_result = hull_engine.run(matrix, run_id=run_id)

        results['hull_volume'] = hull_result.get('volume', 0)
        results['hull_surface_area'] = hull_result.get('surface_area', 0)
    except Exception as e:
        logger.warning(f"Convex hull engine failed: {e}")

    # 7. MUTUAL INFORMATION ENGINE (cohort-level)
    try:
        mi_engine = MutualInformationEngine()
        mi_result = mi_engine.run(matrix, run_id=run_id)
        # Cohort-level MI summary (mean of pairwise)
        if 'mutual_information_matrix' in mi_result:
            mi_matrix = mi_result['mutual_information_matrix']
            results['mi_mean'] = float(np.mean(mi_matrix[np.triu_indices_from(mi_matrix, k=1)]))
    except Exception as e:
        logger.debug(f"MI cohort-level failed: {e}")

    # 8. COPULA ENGINE (cohort-level)
    try:
        copula_engine = CopulaEngine()
        copula_result = copula_engine.run(matrix, run_id=run_id)
        results['copula_upper_mean'] = copula_result.get('upper_tail_dependence', 0)
        results['copula_lower_mean'] = copula_result.get('lower_tail_dependence', 0)
    except Exception as e:
        logger.debug(f"Copula cohort-level failed: {e}")

    # 9. BARYCENTER ENGINE - handled separately with window_vectors

    # Add metadata
    results['n_indicators'] = matrix.shape[1]
    results['n_observations'] = matrix.shape[0]

    return results


def compute_barycenter_metrics(
    cohort: str,
    window_end: date,
    weights: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Compute barycenter metrics for cohort using the canonical BarycenterEngine.

    Fetches multi-window vectors and calls the engine.

    Returns:
        Dict with barycenter_mean_dispersion, barycenter_mean_alignment, barycenter_n_computed
    """
    if weights is None:
        weights = get_barycenter_weights()

    # Get window sizes from weights
    window_sizes = sorted(weights.keys())

    # Build window_vectors for each indicator
    window_vectors = get_indicator_window_vectors(cohort, window_end, window_sizes)

    if not window_vectors:
        return {
            'barycenter_mean_dispersion': None,
            'barycenter_mean_alignment': None,
            'barycenter_n_computed': 0,
        }

    # Call canonical barycenter engine
    barycenter_result = compute_barycenter(window_vectors, weights)

    return {
        'barycenter_mean_dispersion': barycenter_result.get('mean_dispersion'),
        'barycenter_mean_alignment': barycenter_result.get('mean_alignment'),
        'barycenter_n_computed': barycenter_result.get('n_computed', 0),
    }


def compute_pairwise_geometry(
    series_a: np.ndarray,
    series_b: np.ndarray,
    ind_a: str,
    ind_b: str
) -> Dict[str, float]:
    """
    Compute pairwise geometry metrics directly.

    Direct computation without engine class overhead for efficiency.

    Args:
        series_a, series_b: Aligned signal topology
        ind_a, ind_b: Indicator names

    Returns:
        Dict of pairwise metrics
    """
    if len(series_a) < 5 or len(series_b) < 5:
        return {}

    results = {}

    # 1. DISTANCE METRICS (direct computation)
    try:
        # Euclidean distance between normalized series
        a_norm = (series_a - np.mean(series_a)) / (np.std(series_a) + 1e-10)
        b_norm = (series_b - np.mean(series_b)) / (np.std(series_b) + 1e-10)
        results['distance_euclidean'] = float(np.linalg.norm(a_norm - b_norm))

        # Correlation distance: 1 - correlation
        corr = np.corrcoef(series_a, series_b)[0, 1]
        if not np.isnan(corr):
            results['distance_correlation'] = float(1.0 - corr)
    except Exception as e:
        logger.debug(f"Distance pairwise failed: {e}")

    # 2. MUTUAL INFORMATION (binned estimation)
    try:
        from sklearn.metrics import mutual_info_score
        # Discretize into 10 bins for MI estimation
        n_bins = min(10, len(series_a) // 5)
        if n_bins >= 2:
            a_binned = np.digitize(series_a, np.histogram_bin_edges(series_a, bins=n_bins)[:-1])
            b_binned = np.digitize(series_b, np.histogram_bin_edges(series_b, bins=n_bins)[:-1])
            mi = mutual_info_score(a_binned, b_binned)
            results['mutual_information'] = float(mi)
    except Exception as e:
        logger.debug(f"MI pairwise failed: {e}")

    # 3. COPULA METRICS (tail dependence and rank correlation)
    try:
        from scipy import stats

        # Convert to uniform marginals (empirical CDF)
        n = len(series_a)
        u = stats.rankdata(series_a) / (n + 1)
        v = stats.rankdata(series_b) / (n + 1)

        # Tail dependence via threshold approach
        thresholds = [0.05, 0.10, 0.15]
        lower_deps, upper_deps = [], []

        for q in thresholds:
            # Lower tail
            mask_lower = u <= q
            if mask_lower.sum() > 0:
                lower_deps.append((v[mask_lower] <= q).mean())
            # Upper tail
            mask_upper = u >= (1 - q)
            if mask_upper.sum() > 0:
                upper_deps.append((v[mask_upper] >= (1 - q)).mean())

        results['copula_lower_tail'] = float(np.mean(lower_deps)) if lower_deps else 0.0
        results['copula_upper_tail'] = float(np.mean(upper_deps)) if upper_deps else 0.0

        # Kendall's tau
        tau, _ = stats.kendalltau(series_a, series_b)
        results['copula_kendall_tau'] = float(tau) if not np.isnan(tau) else 0.0
    except Exception as e:
        logger.debug(f"Copula pairwise failed: {e}")

    return results


# =============================================================================
# MODE & WAVELET MODULES (inline computation)
# =============================================================================

def compute_mode_metrics(
    cohort: str,
    domain_id: str = 'default',
    field_df: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute mode discovery metrics for a cohort.

    Calls the modes module to discover behavioral modes from Laplace signatures.

    Args:
        cohort: Cohort identifier
        domain_id: Domain identifier
        field_df: Optional pre-loaded Laplace field DataFrame

    Returns:
        Dict with mode metrics (n_modes, mode_entropy_mean, etc.)
    """
    results = {}

    # Try to load Laplace field data if not provided
    if field_df is None:
        try:
            field_path = get_parquet_path('vector', 'indicator_field')
            if Path(field_path).exists():
                field_df = pl.read_parquet(field_path)
            else:
                logger.debug(f"No Laplace field data found for mode discovery")
                return results
        except Exception as e:
            logger.debug(f"Could not load Laplace field: {e}")
            return results

    # Get cohort indicators
    indicators = get_cohort_indicators(cohort)
    if len(indicators) < 3:
        return results

    try:
        # Discover modes using the module
        modes_df = discover_modes(field_df, domain_id, cohort, indicators)

        if modes_df is not None and len(modes_df) > 0:
            results['mode_n_discovered'] = int(modes_df['mode_id'].nunique())
            results['mode_affinity_mean'] = float(modes_df['mode_affinity'].mean())
            results['mode_affinity_std'] = float(modes_df['mode_affinity'].std())
            results['mode_entropy_mean'] = float(modes_df['mode_entropy'].mean())
            results['mode_entropy_std'] = float(modes_df['mode_entropy'].std())

            # Mode distribution
            mode_counts = modes_df['mode_id'].value_counts()
            if len(mode_counts) > 0:
                results['mode_dominant_size'] = int(mode_counts.iloc[0])
                results['mode_balance'] = float(mode_counts.std() / (mode_counts.mean() + 1e-10))

            logger.debug(f"Mode discovery for {cohort}: {results['mode_n_discovered']} modes")
    except Exception as e:
        logger.debug(f"Mode discovery failed for {cohort}: {e}")

    return results


def compute_wavelet_metrics(
    cohort: str,
    observations: Optional[pl.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Compute wavelet microscope metrics for a cohort.

    Identifies which frequency bands show earliest degradation.

    Args:
        cohort: Cohort identifier
        observations: Optional pre-loaded observations DataFrame

    Returns:
        Dict with wavelet degradation metrics
    """
    results = {}

    # Try to load observations if not provided
    if observations is None:
        try:
            observations = pl.read_parquet(get_parquet_path('raw', 'observations'))
        except Exception as e:
            logger.debug(f"Could not load observations for wavelet: {e}")
            return results

    try:
        # Run wavelet microscope
        wavelet_df = run_wavelet_microscope(observations, cohort)

        if wavelet_df is not None and len(wavelet_df) > 0:
            # Extract cohort-level features
            wavelet_features = extract_wavelet_features(wavelet_df, cohort)
            results.update(wavelet_features)
            logger.debug(f"Wavelet analysis for {cohort}: {len(results)} features")
    except Exception as e:
        logger.debug(f"Wavelet analysis failed for {cohort}: {e}")

    return results


# =============================================================================
# DATABASE STORAGE (orchestration)
# =============================================================================

def ensure_schema():
    """Ensure geometry directory exists."""
    ensure_directories()


# Key columns for upsert operations
GEOMETRY_COHORT_KEY_COLS = ['cohort_id', 'window_end', 'window_days']
GEOMETRY_PAIRS_KEY_COLS = ['indicator_a', 'indicator_b', 'window_end', 'window_days']


def store_cohort_geometry_batch(rows: List[Dict[str, Any]], weighted: bool = False):
    """Store batch of cohort geometry metrics to Parquet (both weighted and unweighted)."""
    if not rows:
        return

    df = pl.DataFrame(rows, infer_schema_length=None)

    # Always write both versions
    df_unweighted = df.drop('window_weight') if 'window_weight' in df.columns else df
    upsert_parquet(df_unweighted, get_parquet_path('geometry', 'cohort'), GEOMETRY_COHORT_KEY_COLS)
    upsert_parquet(df, get_parquet_path('geometry', 'cohort_weighted'), GEOMETRY_COHORT_KEY_COLS)

    logger.info(f"Wrote {len(rows)} cohort geometry rows")


def make_cohort_row(
    cohort: str,
    window_end: date,
    window_days: int,
    metrics: Dict[str, Any],
    include_weight: bool = False
) -> Dict[str, Any]:
    """Create a row dict for cohort geometry."""
    row = {
        'cohort_id': cohort,
        'window_end': window_end,
        'window_days': window_days,
    }
    if include_weight:
        row['window_weight'] = get_window_weight(window_days)
    row.update({
        'n_indicators': metrics.get('n_indicators', 0),
        'n_observations': metrics.get('n_observations', 0),
        'distance_mean': metrics.get('distance_mean'),
        'distance_std': metrics.get('distance_std'),
        'pca_variance_pc1': metrics.get('pca_variance_pc1'),
        'pca_variance_pc2': metrics.get('pca_variance_pc2'),
        'pca_variance_pc3': metrics.get('pca_variance_pc3'),
        'pca_cumulative_3': metrics.get('pca_cumulative_3'),
        'pca_effective_dim': metrics.get('pca_effective_dim'),
        'clustering_silhouette': metrics.get('clustering_silhouette'),
        'clustering_n_clusters': metrics.get('clustering_n_clusters'),
        'mst_total_weight': metrics.get('mst_total_weight'),
        'mst_avg_degree': metrics.get('mst_avg_degree'),
        'lof_mean': metrics.get('lof_mean'),
        'lof_n_outliers': metrics.get('lof_n_outliers'),
        'hull_volume': metrics.get('hull_volume'),
        'hull_surface_area': metrics.get('hull_surface_area'),
        'barycenter_mean_dispersion': metrics.get('barycenter_mean_dispersion'),
        'barycenter_mean_alignment': metrics.get('barycenter_mean_alignment'),
        'barycenter_n_computed': metrics.get('barycenter_n_computed'),
        # Mode discovery metrics (from prism.modules.modes)
        'mode_n_discovered': metrics.get('mode_n_discovered'),
        'mode_affinity_mean': metrics.get('mode_affinity_mean'),
        'mode_affinity_std': metrics.get('mode_affinity_std'),
        'mode_entropy_mean': metrics.get('mode_entropy_mean'),
        'mode_entropy_std': metrics.get('mode_entropy_std'),
        'mode_dominant_size': metrics.get('mode_dominant_size'),
        'mode_balance': metrics.get('mode_balance'),
        # Wavelet degradation metrics (from prism.modules.wavelet_microscope)
        'wavelet_max_degradation': metrics.get('wavelet_max_degradation'),
        'wavelet_mean_degradation': metrics.get('wavelet_mean_degradation'),
        'wavelet_n_degrading': metrics.get('wavelet_n_degrading'),
        'wavelet_worst_snr_change': metrics.get('wavelet_worst_snr_change'),
        'wavelet_dominant_band': metrics.get('wavelet_dominant_band'),
        'computed_at': datetime.now(),
    })
    return row


def store_pairwise_geometry_batch(rows: List[Dict[str, Any]], weighted: bool = False):
    """Store batch of pairwise geometry metrics to Parquet (both weighted and unweighted)."""
    if not rows:
        return

    df = pl.DataFrame(rows, infer_schema_length=None)

    # Always write both versions
    df_unweighted = df.drop('window_weight') if 'window_weight' in df.columns else df
    upsert_parquet(df_unweighted, get_parquet_path('geometry', 'indicator_pair'), GEOMETRY_PAIRS_KEY_COLS)
    upsert_parquet(df, get_parquet_path('geometry', 'indicator_pair_weighted'), GEOMETRY_PAIRS_KEY_COLS)

    logger.info(f"Wrote {len(rows)} pairwise geometry rows")


def make_pairwise_row(
    ind_a: str,
    ind_b: str,
    window_end: date,
    window_days: int,
    metrics: Dict[str, float],
    include_weight: bool = False
) -> Dict[str, Any]:
    """Create a row dict for pairwise geometry."""
    row = {
        'indicator_a': ind_a,
        'indicator_b': ind_b,
        'window_end': window_end,
        'window_days': window_days,
    }
    if include_weight:
        row['window_weight'] = get_window_weight(window_days)
    row.update({
        'distance_euclidean': metrics.get('distance_euclidean'),
        'distance_correlation': metrics.get('distance_correlation'),
        'mutual_information': metrics.get('mutual_information'),
        'copula_upper_tail': metrics.get('copula_upper_tail'),
        'copula_lower_tail': metrics.get('copula_lower_tail'),
        'copula_kendall_tau': metrics.get('copula_kendall_tau'),
        'computed_at': datetime.now(),
    })
    return row


# =============================================================================
# MAIN RUNNERS
# =============================================================================

def run_cohort_geometry(
    cohort: str,
    window_end: date,
    window_days: int,
    include_weight: bool = False,
    run_bisection: bool = False  # Disabled for now - needs parquet adaptation
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Run geometry for a cohort at a specific window.

    Calls all 9 canonical engines.

    Returns:
        Tuple of (result_dict, row_dict for batch storage)
    """
    matrix = get_cohort_data_matrix(cohort, window_end, window_days)

    if matrix.empty or matrix.shape[1] < 2:
        logger.warning(f"Insufficient data for {cohort} at {window_end} ({window_days}d)")
        return {'status': 'insufficient_data'}, None

    # Compute cohort-level geometry (8 canonical engines)
    metrics = compute_cohort_geometry(matrix, cohort, window_end)

    # Compute barycenters (9th canonical engine)
    barycenter_metrics = compute_barycenter_metrics(cohort, window_end)
    metrics.update(barycenter_metrics)

    # Compute mode discovery metrics (inline module)
    mode_metrics = compute_mode_metrics(cohort)
    metrics.update(mode_metrics)

    # Compute wavelet degradation metrics (inline module)
    wavelet_metrics = compute_wavelet_metrics(cohort)
    metrics.update(wavelet_metrics)

    # Create row for batch storage
    row = make_cohort_row(cohort, window_end, window_days, metrics, include_weight=include_weight)

    logger.info(f"  {cohort} @ {window_end} ({window_days}d): {matrix.shape[1]} indicators, "
                f"PCA_1={(metrics.get('pca_variance_pc1') or 0):.3f}, "
                f"barycenter_disp={(metrics.get('barycenter_mean_dispersion') or 0):.3f}")

    return {
        'status': 'success',
        'n_indicators': metrics.get('n_indicators', 0),
        'n_observations': metrics.get('n_observations', 0),
    }, row


def run_pairwise_geometry(
    cohort: str,
    window_end: date,
    window_days: int,
    include_weight: bool = False
) -> List[Dict[str, Any]]:
    """Run pairwise geometry for all indicator pairs in a cohort."""
    indicators = get_cohort_indicators(cohort)
    rows = []

    if len(indicators) < 2:
        logger.warning(f"Cohort {cohort} has fewer than 2 indicators")
        return rows

    for i, ind_a in enumerate(indicators):
        for ind_b in indicators[i+1:]:
            series_a, series_b = get_pairwise_data(ind_a, ind_b, window_end, window_days)

            if len(series_a) < 5:
                continue

            metrics = compute_pairwise_geometry(series_a, series_b, ind_a, ind_b)
            row = make_pairwise_row(ind_a, ind_b, window_end, window_days, metrics, include_weight=include_weight)
            rows.append(row)

    logger.info(f"  Pairwise: {len(rows)} pairs processed for {cohort} @ {window_end} ({window_days}d)")
    return rows


def run_window_tier(
    cohorts: List[str],
    window_name: str,
    start_date: date,
    end_date: date,
    include_pairwise: bool = True
) -> Dict[str, int]:
    """
    Run geometry for all cohorts across a window tier's date range.

    Uses stride from config/stride.yaml.

    Args:
        cohorts: List of cohorts to process
        window_name: Window tier name ('anchor', 'bridge', 'scout', 'micro')
        start_date: Start of date range
        end_date: End of date range
        include_pairwise: Whether to compute pairwise geometry

    Returns:
        Dict with processing stats
    """
    config = load_stride_config()
    window = config.get_window(window_name)
    window_days = window.window_days

    # Generate dates at configured stride
    dates = get_window_dates(window_name, start_date, end_date, config)

    logger.info(f"Window tier: {window_name} ({window_days}d, stride {window.stride_days}d)")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Snapshots: {len(dates)}")

    cohort_rows = []
    pairwise_rows = []

    for window_end in dates:
        for cohort in cohorts:
            result, row = run_cohort_geometry(cohort, window_end, window_days)
            if row:
                cohort_rows.append(row)

            if include_pairwise:
                pair_rows = run_pairwise_geometry(cohort, window_end, window_days)
                pairwise_rows.extend(pair_rows)

    # Batch store results
    if cohort_rows:
        store_cohort_geometry_batch(cohort_rows)
    if pairwise_rows:
        store_pairwise_geometry_batch(pairwise_rows)

    return {
        'cohort_rows': len(cohort_rows),
        'pairwise_rows': len(pairwise_rows),
        'snapshots': len(dates),
    }


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

PROGRESS_PATH = Path('data/geometry/.progress_geometry.parquet')


def get_completed_windows() -> set:
    """Get set of completed (cohort_id, window_end, window_days) tuples."""
    if not PROGRESS_PATH.exists():
        return set()
    try:
        df = pl.read_parquet(PROGRESS_PATH)
        return set(
            (row['cohort_id'], row['window_end'], row['window_days'])
            for row in df.iter_rows(named=True)
        )
    except Exception:
        return set()


def mark_window_complete(cohort_id: str, window_end: date, window_days: int):
    """Mark a window as complete."""
    new_row = pl.DataFrame([{
        'cohort_id': cohort_id,
        'window_end': window_end,
        'window_days': window_days,
        'completed_at': datetime.now(),
    }])

    if PROGRESS_PATH.exists():
        existing = pl.read_parquet(PROGRESS_PATH)
        combined = pl.concat([existing, new_row])
    else:
        combined = new_row

    combined.write_parquet(PROGRESS_PATH)


def clear_progress():
    """Clear progress tracker."""
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
        logger.info("Progress cleared (--force)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Geometry Runner - Windowed by Design (9 canonical engines)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection (mutually exclusive, required)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--indicator', action='store_true',
                            help='Process within-cohort indicator geometry (pairwise + cohort-level)')
    mode_group.add_argument('--cohort', action='store_true',
                            help='Process cross-cohort geometry (cohort comparisons)')

    # Production flags
    parser.add_argument('--force', action='store_true',
                        help='Clear progress tracker and recompute all')
    parser.add_argument('--weighted', action='store_true',
                        help='Include window_weight column for weighted aggregation')

    # Testing mode - REQUIRED to use any limiting flags
    parser.add_argument('--testing', action='store_true',
                        help='Enable testing mode. REQUIRED to use limiting flags.')

    # Testing-only flags
    parser.add_argument('--filter-cohort', type=str,
                        help='[TESTING] Filter to specific cohort')
    parser.add_argument('--dates', type=str,
                        help='[TESTING] Date range as START:END (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--no-pairwise', action='store_true',
                        help='[TESTING] Skip pairwise geometry')

    # Domain selection (required - prompts if not specified)
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain to process (e.g., cheme, cmapss). Prompts if not specified.')

    args = parser.parse_args()

    # Domain selection - prompt if not specified
    from prism.utils.domain import require_domain
    import os
    domain = require_domain(args.domain, "Select domain for geometry")
    os.environ["PRISM_DOMAIN"] = domain
    print(f"Domain: {domain}", flush=True)

    # ==========================================================================
    # CRITICAL: --testing guard
    # Without --testing, ALL limiting flags are ignored and full run executes.
    # ==========================================================================
    if not args.testing:
        limiting_flags_used = []
        if args.filter_cohort:
            limiting_flags_used.append('--filter-cohort')
        if args.dates:
            limiting_flags_used.append('--dates')
        if args.no_pairwise:
            limiting_flags_used.append('--no-pairwise')

        if limiting_flags_used:
            logger.warning("=" * 80)
            logger.warning("LIMITING FLAGS IGNORED - --testing not specified")
            logger.warning(f"Ignored flags: {', '.join(limiting_flags_used)}")
            logger.warning("Running FULL computation instead. Use --testing to enable limiting flags.")
            logger.warning("=" * 80)

        # Override to full defaults
        args.filter_cohort = None
        args.dates = None
        args.no_pairwise = False

    # Clear progress if --force
    if args.force:
        clear_progress()

    # Ensure directories exist
    ensure_schema()

    # Get all cohorts, optionally filter
    cohorts = get_all_cohorts()
    if args.filter_cohort:
        if args.filter_cohort in cohorts:
            cohorts = [args.filter_cohort]
        else:
            logger.error(f"Cohort '{args.filter_cohort}' not found. Available: {cohorts}")
            return 1

    # Get date range from data
    min_date, max_date = get_date_range()

    # Parse dates
    if args.dates:
        try:
            start_str, end_str = args.dates.split(':')
            start_date = pd.to_datetime(start_str).date()
            end_date = pd.to_datetime(end_str).date()
        except ValueError:
            logger.error("Invalid --dates format. Use START:END (YYYY-MM-DD:YYYY-MM-DD)")
            return 1
    else:
        start_date = min_date
        end_date = max_date

    stride_config = load_stride_config()

    # GEOMETRY: use default tiers from config (anchor + bridge + scout)
    # NO micro - 21d/1d is too expensive for geometry pairwise
    window_tiers = get_default_tiers()

    # Determine mode
    mode = "indicator" if args.indicator else "cohort"

    # Memory tracking
    start_memory = get_memory_usage_mb()

    logger.info("=" * 80)
    logger.info("PRISM GEOMETRY - WINDOWED BY DESIGN (9 Canonical Engines)")
    logger.info("=" * 80)
    logger.info(f"Mode: --{mode}")
    logger.info(f"Storage: Parquet files")
    logger.info(f"Starting memory: {start_memory:.0f} MB")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Cohorts: {len(cohorts)}")
    logger.info(f"Window tiers: {window_tiers}")
    for tier in window_tiers:
        w = stride_config.get_window(tier)
        logger.info(f"  {tier}: {w.window_days}d / {w.stride_days}d stride")
    logger.info(f"Pairwise: {'NO (testing)' if args.no_pairwise else 'YES'}")
    logger.info(f"Weighted: {'YES (writes to *_weighted.parquet)' if args.weighted else 'NO'}")
    logger.info("")

    # Branch based on mode
    if args.cohort:
        logger.info("Cross-cohort geometry: comparing cohorts to each other")
        logger.info("(Use prism.entry_points.cohort_geometry for now - integration pending)")
        return 0

    # Get completed windows for progress tracking
    completed = get_completed_windows()
    if completed:
        logger.info(f"Resuming: {len(completed)} windows already complete")

    # Incremental write batch size
    BATCH_SIZE = 10  # Write every 10 windows (was 50)

    # Process each window tier
    total_cohort_rows = 0
    total_pairwise_rows = 0

    for window_name in window_tiers:
        window = stride_config.get_window(window_name)
        window_days = window.window_days

        # Generate dates at configured stride
        dates = get_window_dates(window_name, start_date, end_date, stride_config)

        logger.info(f"[{window_name}] {window_days}d / {window.stride_days}d stride, {len(dates)} snapshots")

        cohort_rows = []
        pairwise_rows = []
        tier_cohort_rows = 0
        tier_pairwise_rows = 0
        window_count = 0

        for window_end in dates:
            for cohort in cohorts:
                # Skip if already complete
                if (cohort, window_end, window_days) in completed:
                    continue

                result, row = run_cohort_geometry(cohort, window_end, window_days, include_weight=True)
                if row:
                    cohort_rows.append(row)

                if not args.no_pairwise:
                    pair_rows = run_pairwise_geometry(cohort, window_end, window_days, include_weight=True)
                    pairwise_rows.extend(pair_rows)

                # Mark complete
                mark_window_complete(cohort, window_end, window_days)
                window_count += 1

                # Incremental write every BATCH_SIZE windows - COMPUTE → WRITE → RELEASE
                if window_count % BATCH_SIZE == 0:
                    if cohort_rows:
                        store_cohort_geometry_batch(cohort_rows, weighted=args.weighted)
                        tier_cohort_rows += len(cohort_rows)
                        del cohort_rows
                        cohort_rows = []
                    if pairwise_rows:
                        store_pairwise_geometry_batch(pairwise_rows, weighted=args.weighted)
                        tier_pairwise_rows += len(pairwise_rows)
                        del pairwise_rows
                        pairwise_rows = []

                    # RELEASE - explicit GC after batch write
                    force_gc()

                    current_mem = get_memory_usage_mb()
                    logger.info(f"  [{window_name}] Saved {window_count} windows ({tier_cohort_rows} cohort, {tier_pairwise_rows} pair rows) [mem: {current_mem:.0f} MB]")

        # Final batch for this tier - WRITE → RELEASE
        if cohort_rows:
            store_cohort_geometry_batch(cohort_rows, weighted=args.weighted)
            tier_cohort_rows += len(cohort_rows)
            del cohort_rows
        if pairwise_rows:
            store_pairwise_geometry_batch(pairwise_rows, weighted=args.weighted)
            tier_pairwise_rows += len(pairwise_rows)
            del pairwise_rows

        # RELEASE after tier
        force_gc()

        total_cohort_rows += tier_cohort_rows
        total_pairwise_rows += tier_pairwise_rows

        tier_mem = get_memory_usage_mb()
        logger.info(f"[{window_name}] Complete: {tier_cohort_rows} cohort rows, {tier_pairwise_rows} pair rows [mem: {tier_mem:.0f} MB]")

    # Final memory summary
    end_memory = get_memory_usage_mb()
    delta = end_memory - start_memory

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Cohort geometry rows: {total_cohort_rows}")
    logger.info(f"Pairwise geometry rows: {total_pairwise_rows}")
    logger.info(f"Memory: {start_memory:.0f} → {end_memory:.0f} MB (Δ{delta:+.0f} MB)")

    return 0


if __name__ == '__main__':
    exit(main())
