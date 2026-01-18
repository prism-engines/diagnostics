"""
PRISM Vector Runner - Config is Law
====================================

Computes vector metrics using sliding windows. Config determines behavior.
Uses characterization to determine which engines to run per indicator.

MODES (mutually exclusive, one required):
    --indicators    Process individual indicators
                    Source: raw/observations + raw/characterization
                    Destination: vector/indicators → vector/indicator_field (via laplace)

    --cohorts       Process cohort-level aggregations
                    Source: raw/cohort_observations
                    Destination: vector/cohorts

ENGINE CLASSIFICATION:
    CORE ENGINES (run on ALL indicators):
        - hurst:        Long-range dependence and memory
        - entropy:      Information content and complexity
        - rqa:          Recurrence quantification (phase space)
        - realized_vol: Short-window volatility, drawdown, distribution

    CONDITIONAL ENGINES (run based on characterization):
        - spectral:     ax_periodicity > 0.3
        - wavelet:      ax_periodicity > 0.3
        - garch:        ax_volatility > 0.3
        - lyapunov:     ax_complexity > 0.3

    DISCONTINUITY ENGINES (run if breaks detected):
        - break_detector: Always runs to detect breaks
        - heaviside:      Runs if breaks found (persistent steps)
        - dirac:          Runs if breaks found (transient impulses)

PIPELINE:
    observations → characterize → indicator_vector → laplace → geometry
                       ↓                 ↓
                 (valid_engines)   (uses valid_engines, chains to laplace)

Storage: Parquet files (no database locks)

Usage:
    # Process all indicators (production)
    python -m prism.entry_points.indicator_vector --indicators

    # Process cohorts (production)
    python -m prism.entry_points.indicator_vector --cohorts

    # Force recompute
    python -m prism.entry_points.indicator_vector --indicators --force

    # Testing mode (allows limiting flags)
    python -m prism.entry_points.indicator_vector --indicators --testing --domain cmapss
    python -m prism.entry_points.indicator_vector --indicators --testing --filter sensor_1,sensor_2
"""

import argparse
import gc
import json
import logging
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import ensure_directories, get_parquet_path
from prism.db.polars_io import read_parquet, upsert_parquet, write_parquet_atomic
from prism.utils.memory import (
    force_gc,
    get_memory_usage_mb,
    get_size_mb,
    MemoryTracker,
)
from prism.db.scratch import TempParquet, merge_to_table
from prism.engines.utils.parallel import (
    WorkerAssignment,
    divide_by_count,
    generate_temp_path,
    run_workers,
)

# Window/stride configuration
from prism.utils.stride import load_stride_config, get_default_tiers

# Normalization configuration
import yaml

# Inline modules (characterize + laplace)
from prism.modules.characterize import (
    characterize_indicator,
    get_engines_from_characterization,
    get_characterization_summary,
)
from prism.modules.laplace import (
    compute_laplace_for_series,
    compute_divergence_for_indicator,
    add_divergence_to_field_rows,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_OBSERVATIONS = 15  # Absolute floor for any computation

DEFAULT_ENGINE_MIN_OBS = {
    "hurst": 20,
    "entropy": 30,
    "lyapunov": 30,
    "garch": 50,
    "spectral": 40,
    "wavelet": 40,
    "rqa": 30,
    "realized_vol": 15,  # Short-window: vol, drawdown, distribution (13 metrics)
    "hilbert": 20,  # Hilbert transform: amplitude, phase, inst_freq (7 metrics)
    # Observation-level (discontinuity) engines
    "break_detector": 50,
    "heaviside": 50,
    "dirac": 50,
}

# Key columns for upsert deduplication
VECTOR_KEY_COLS = ["indicator_id", "obs_date", "target_obs", "engine", "metric_name"]


# =============================================================================
# REGIME-AWARE NORMALIZATION
# =============================================================================

def get_normalization_config(domain: str = None) -> dict:
    """
    Load domain-specific normalization parameters.

    Args:
        domain: Domain name (cmapss, climate, cheme, icu, etc.)
                Uses PRISM_DOMAIN env var if not specified.

    Returns:
        Dict with 'window' and 'min_periods' keys
    """
    import os
    if domain is None:
        domain = os.environ.get('PRISM_DOMAIN')
    if domain is None:
        raise RuntimeError("No domain specified - set PRISM_DOMAIN or pass domain parameter")

    defaults = {'window': 252, 'min_periods': 30}

    config_path = Path(__file__).parent.parent.parent / 'config' / 'normalization.yaml'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get('domains', {}).get(domain, config.get('defaults', defaults))
        except Exception:
            pass

    return defaults


def apply_regime_normalization(
    batch_rows: List[Dict],
    window: int = 252,
    min_periods: int = 30,
) -> pl.DataFrame:
    """
    Standardizes signal behavior relative to trailing window.
    Ensures 'Source' detection is calibrated to current volatility regimes.

    Args:
        batch_rows: List of metric dicts from indicator processing
        window: Rolling window size (default 252 ~ 1 year of daily data)
        min_periods: Minimum observations before computing (default 30)

    Returns:
        DataFrame with original metric_value AND metric_value_norm columns.
        metric_value_norm will be NaN for first min_periods rows (burn-in).

    Philosophy:
        - NaN during burn-in is honest accounting, not a problem to solve
        - Rolling window preserves regime awareness
        - Both raw and normalized preserved for downstream choice
    """
    df = pl.DataFrame(batch_rows, infer_schema_length=None)

    if len(df) == 0:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("metric_value_norm"))

    # Sort by date within each metric group (required for rolling)
    df = df.sort(["indicator_id", "engine", "metric_name", "obs_date"])

    # Apply rolling Z-score within each indicator/engine/metric group
    # Z = (x - rolling_mean) / (rolling_std + epsilon)
    return df.with_columns([
        ((pl.col("metric_value") - pl.col("metric_value").rolling_mean(window, min_periods=min_periods)) /
         (pl.col("metric_value").rolling_std(window, min_periods=min_periods) + 1e-10))
        .over(["indicator_id", "engine", "metric_name"])
        .alias("metric_value_norm")
    ])

# =============================================================================
# ENGINE CLASSIFICATION
# =============================================================================
# CORE engines run on ALL indicators regardless of characterization
# CONDITIONAL engines only run if characterization indicates they're valid

CORE_ENGINES = {"hurst", "entropy", "rqa", "realized_vol", "hilbert"}

CONDITIONAL_ENGINES = {"spectral", "wavelet", "garch", "lyapunov"}

# Discontinuity engines: break_detector always runs, heaviside/dirac run if breaks found
DISCONTINUITY_ENGINES = {"break_detector", "heaviside", "dirac"}


# =============================================================================
# VECTOR ENGINES
# =============================================================================

from prism.engines import VECTOR_ENGINES, OBSERVATION_ENGINES, compute_breaks


# =============================================================================
# INLINE CHARACTERIZATION
# =============================================================================

def characterize_indicator_inline(
    indicator_id: str,
    values: np.ndarray,
    dates: Optional[np.ndarray] = None,
) -> Tuple[set, bool, Dict[str, Any]]:
    """
    Characterize an indicator inline and determine which engines to run.

    This replaces the pre-computed characterization parquet approach.
    Characterization happens as each indicator is processed.

    Args:
        indicator_id: The indicator to characterize
        values: Signal values (full history, will be cleaned of NaN)
        dates: Optional observation dates

    Returns:
        Tuple of (engines_to_run: set, has_discontinuities: bool, char_summary: dict)
    """
    # Clean values
    values = np.asarray(values, dtype=float)
    valid_mask = ~np.isnan(values)
    clean_values = values[valid_mask]

    if len(clean_values) < 50:
        # Insufficient data - run core engines only
        logger.debug(f"{indicator_id}: Insufficient data ({len(clean_values)}), using core engines only")
        return CORE_ENGINES.copy(), False, {'indicator_id': indicator_id, 'reason': 'insufficient_data'}

    # Clean dates if provided
    clean_dates = None
    if dates is not None:
        dates = np.asarray(dates)
        if len(dates) == len(values):
            clean_dates = dates[valid_mask]

    # Characterize inline using the module
    char_result = characterize_indicator(
        indicator_id=indicator_id,
        values=clean_values,
        dates=clean_dates,
    )

    # Get engines to run using module helper
    engines_to_run, has_discontinuities = get_engines_from_characterization(
        char_result=char_result,
        core_engines=CORE_ENGINES,
        conditional_engines=CONDITIONAL_ENGINES,
        discontinuity_engines=DISCONTINUITY_ENGINES,
    )

    # Get summary for logging/storage
    char_summary = get_characterization_summary(char_result)

    return engines_to_run, has_discontinuities, char_summary


# Legacy function for backwards compatibility (reads from parquet)
def load_characterization() -> Dict[str, Dict[str, Any]]:
    """
    Load pre-computed characterization data from parquet.

    DEPRECATED: Use characterize_indicator_inline() instead.
    This is kept for backwards compatibility with parallel workers.
    """
    char_path = get_parquet_path('raw', 'characterization')

    if not char_path.exists():
        return {}

    df = pl.read_parquet(char_path)

    char_dict = {}
    for row in df.iter_rows(named=True):
        indicator_id = row['indicator_id']

        # Parse valid_engines from JSON string
        valid_engines_str = row.get('valid_engines', '[]')
        if isinstance(valid_engines_str, str):
            try:
                valid_engines = json.loads(valid_engines_str)
            except json.JSONDecodeError:
                valid_engines = []
        else:
            valid_engines = valid_engines_str or []

        char_dict[indicator_id] = {
            'valid_engines': set(valid_engines),
            'has_discontinuities': row.get('n_breaks', 0) > 0,
        }

    return char_dict


def get_engines_for_indicator(
    indicator_id: str,
    characterization: Dict[str, Dict[str, Any]],
) -> Tuple[set, bool]:
    """
    Determine which engines to run based on pre-computed characterization.

    DEPRECATED: Use characterize_indicator_inline() instead for inline characterization.
    """
    # Start with CORE engines (always run)
    engines_to_run = CORE_ENGINES.copy()

    # Check if we have characterization for this indicator
    if indicator_id not in characterization:
        engines_to_run.update(CONDITIONAL_ENGINES)
        return engines_to_run, False

    char = characterization[indicator_id]
    valid_engines = char.get('valid_engines', set())
    has_discontinuities = char.get('has_discontinuities', False)

    # Add conditional engines if they're in the valid_engines list
    for engine in CONDITIONAL_ENGINES:
        if engine in valid_engines:
            engines_to_run.add(engine)

    return engines_to_run, has_discontinuities


# =============================================================================
# SLIDING WINDOW GENERATION
# =============================================================================


def generate_windows(
    observations: pl.DataFrame,
    target_obs: int,
    min_obs: int,
    stride: int,
) -> List[Tuple[np.ndarray, date, date, int]]:
    """
    Generate sliding windows from observations using Polars.

    Args:
        observations: DataFrame with obs_date and value columns
        target_obs: Target number of observations per window
        min_obs: Minimum observations required
        stride: Days between window endpoints

    Returns:
        List of (values, obs_date, lookback_start, actual_obs) tuples
    """
    if len(observations) < min_obs:
        return []

    # Sort by date
    df = observations.sort("obs_date")

    # Extract arrays
    dates = df["obs_date"].to_numpy()
    values = df["value"].to_numpy()
    n = len(values)

    windows = []
    start_idx = max(min_obs, target_obs) - 1

    for end_idx in range(start_idx, n, stride):
        window_start_idx = max(0, end_idx - target_obs + 1)
        window_len = end_idx - window_start_idx + 1

        if window_len < min_obs:
            continue

        window_values = values[window_start_idx : end_idx + 1]
        obs_date = dates[end_idx]
        lookback_start = dates[window_start_idx]

        # Convert dates to Python datetime (handles numpy.datetime64, pandas Timestamp, etc.)
        if hasattr(obs_date, "date"):
            obs_date = obs_date.date()
        elif hasattr(obs_date, "astype"):  # numpy.datetime64
            obs_date = pd.Timestamp(obs_date).to_pydatetime().date()
        if hasattr(lookback_start, "date"):
            lookback_start = lookback_start.date()
        elif hasattr(lookback_start, "astype"):  # numpy.datetime64
            lookback_start = pd.Timestamp(lookback_start).to_pydatetime().date()

        windows.append((window_values, obs_date, lookback_start, window_len))

    return windows


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================


def process_indicator_parallel(assignment: WorkerAssignment) -> Dict[str, Any]:
    """
    Worker function for parallel execution.

    Writes to isolated temp parquet file.
    Memory-efficient: reads observations per-indicator using lazy scan.
    """
    indicators = assignment.items
    config = assignment.config
    temp_path = assignment.temp_path

    window_days = config.get("window_days")
    stride_days = config.get("stride_days")
    engine_min_obs = config.get("engine_min_obs", DEFAULT_ENGINE_MIN_OBS)
    engines_filter = config.get("engines_filter", None)

    # Get path for lazy scanning (read per-indicator)
    obs_path = get_parquet_path("raw", "observations")

    total_windows = 0
    total_metrics = 0
    failed = 0
    batch_rows = []
    computed_at = datetime.now()

    for indicator_id in indicators:
        try:
            # Read only this indicator's observations - LAZY SCAN with filter pushdown
            obs_df = (
                pl.scan_parquet(obs_path)
                .filter(pl.col("indicator_id") == indicator_id)
                .collect()
            )

            if len(obs_df) == 0:
                del obs_df
                continue

            # Generate windows
            min_obs = min(engine_min_obs.values())
            windows = generate_windows(obs_df, window_days, min_obs, stride_days)
            total_windows += len(windows)

            # Process each window
            for values, obs_date, lookback_start, actual_obs in windows:
                # Run each engine
                for engine_name, engine_func in VECTOR_ENGINES.items():
                    if engines_filter and engine_name not in engines_filter:
                        continue

                    min_obs_engine = engine_min_obs.get(engine_name, MIN_OBSERVATIONS)
                    if actual_obs < min_obs_engine:
                        continue

                    # Compute metrics
                    try:
                        try:
                            metrics = engine_func(values, min_obs=min_obs_engine)
                        except TypeError:
                            metrics = engine_func(values)
                    except Exception as e:
                        logger.debug(f"Engine {engine_name} failed for {indicator_id}: {e}")
                        continue

                    # Collect metrics
                    for metric_name, metric_value in metrics.items():
                        try:
                            if metric_value is not None:
                                numeric_value = float(metric_value)
                                if np.isfinite(numeric_value):
                                    batch_rows.append(
                                        {
                                            "indicator_id": indicator_id,
                                            "obs_date": obs_date,
                                            "target_obs": window_days,
                                            "actual_obs": actual_obs,
                                            "lookback_start": lookback_start,
                                            "engine": engine_name,
                                            "metric_name": metric_name,
                                            "metric_value": numeric_value,
                                            "computed_at": computed_at,
                                        }
                                    )
                        except (TypeError, ValueError):
                            continue

                # Run OBSERVATION_ENGINES (discontinuity detection)
                # These run on the window but have special handling:
                # - break_detector runs first to find break indices
                # - heaviside/dirac only run if breaks found
                min_obs_break = engine_min_obs.get("break_detector", 50)
                if actual_obs >= min_obs_break:
                    try:
                        # Get break metrics first
                        from prism.engines import get_break_metrics, get_heaviside_metrics, get_dirac_metrics

                        break_metrics = get_break_metrics(values)

                        # Add break_detector metrics
                        for metric_name, metric_value in break_metrics.items():
                            if metric_value is not None:
                                try:
                                    numeric_value = float(metric_value)
                                    if np.isfinite(numeric_value):
                                        batch_rows.append({
                                            "indicator_id": indicator_id,
                                            "obs_date": obs_date,
                                            "target_obs": window_days,
                                            "actual_obs": actual_obs,
                                            "lookback_start": lookback_start,
                                            "engine": "break_detector",
                                            "metric_name": metric_name,
                                            "metric_value": numeric_value,
                                            "computed_at": computed_at,
                                        })
                                except (TypeError, ValueError):
                                    continue

                        # If breaks found, run heaviside and dirac
                        if break_metrics.get('break_n', 0) > 0:
                            break_result = compute_breaks(values)
                            break_indices = break_result['break_indices']

                            # Heaviside (step function measurement)
                            heaviside_metrics = get_heaviside_metrics(values, break_indices)
                            for metric_name, metric_value in heaviside_metrics.items():
                                if metric_value is not None:
                                    try:
                                        numeric_value = float(metric_value)
                                        if np.isfinite(numeric_value):
                                            batch_rows.append({
                                                "indicator_id": indicator_id,
                                                "obs_date": obs_date,
                                                "target_obs": window_days,
                                                "actual_obs": actual_obs,
                                                "lookback_start": lookback_start,
                                                "engine": "heaviside",
                                                "metric_name": metric_name,
                                                "metric_value": numeric_value,
                                                "computed_at": computed_at,
                                            })
                                    except (TypeError, ValueError):
                                        continue

                            # Dirac (impulse measurement)
                            dirac_metrics = get_dirac_metrics(values, break_indices)
                            for metric_name, metric_value in dirac_metrics.items():
                                if metric_value is not None:
                                    try:
                                        numeric_value = float(metric_value)
                                        if np.isfinite(numeric_value):
                                            batch_rows.append({
                                                "indicator_id": indicator_id,
                                                "obs_date": obs_date,
                                                "target_obs": window_days,
                                                "actual_obs": actual_obs,
                                                "lookback_start": lookback_start,
                                                "engine": "dirac",
                                                "metric_name": metric_name,
                                                "metric_value": numeric_value,
                                                "computed_at": computed_at,
                                            })
                                    except (TypeError, ValueError):
                                        continue
                    except Exception as e:
                        logger.debug(f"Observation engines failed for {indicator_id}: {e}")

            # RELEASE - clean up indicator data before next iteration
            del obs_df
            del windows
            gc.collect()

        except Exception as e:
            logger.error(f"Failed processing {indicator_id}: {e}")
            failed += 1

    # Write to temp parquet - then RELEASE
    if batch_rows:
        df = pl.DataFrame(batch_rows)
        df.write_parquet(temp_path)
        total_metrics = len(batch_rows)
        # RELEASE
        del df
        del batch_rows
        gc.collect()

    return {
        "processed": len(indicators) - failed,
        "failed": failed,
        "windows": total_windows,
        "metrics": total_metrics,
        "temp_path": temp_path,
    }


def process_indicator_sequential(
    indicator_id: str,
    window_days: int,
    stride_days: int,
    engine_min_obs: Dict[str, int],
    engines_to_run: Optional[set] = None,
    has_discontinuities: bool = False,
    use_inline_characterization: bool = True,
    compute_laplace_inline: bool = True,
) -> Dict[str, Any]:
    """
    Process a single indicator sequentially.
    Computes engine metrics based on characterization + vector score for each window.
    Optionally characterizes inline and computes laplace field inline.

    Args:
        indicator_id: The indicator to process
        window_days: Window size in days
        stride_days: Stride between windows
        engine_min_obs: Minimum observations per engine
        engines_to_run: Set of engine names to run (from characterization)
                       If None, characterizes inline or runs all engines
        has_discontinuities: Whether characterization found discontinuities
        use_inline_characterization: If True and engines_to_run is None,
                                     characterize inline to determine engines
        compute_laplace_inline: If True, compute laplace field inline

    Memory-efficient: Uses lazy scan to read only this indicator's data.
    """
    # Default to all engines if not specified AND inline characterization disabled
    if engines_to_run is None and not use_inline_characterization:
        engines_to_run = CORE_ENGINES | CONDITIONAL_ENGINES
    from prism.engines.vector_score import (
        ENGINE_CONFIGS,
        compute_baselines,
        compute_vector_score,
    )

    try:
        # Read observations - LAZY SCAN with filter pushdown
        # Only reads the specific indicator's data from disk
        obs_path = get_parquet_path("raw", "observations")
        obs_df = (
            pl.scan_parquet(obs_path)
            .filter(pl.col("indicator_id") == indicator_id)
            .collect()
        )

        if len(obs_df) == 0:
            return {"indicator": indicator_id, "windows": 0, "metrics": 0}

        # INLINE CHARACTERIZATION: If engines_to_run not provided, characterize now
        char_summary = None
        if engines_to_run is None and use_inline_characterization:
            # Get full series for characterization
            full_values = obs_df.sort("obs_date")["value"].to_numpy()
            full_dates = obs_df.sort("obs_date")["obs_date"].to_numpy()

            engines_to_run, has_discontinuities, char_summary = characterize_indicator_inline(
                indicator_id=indicator_id,
                values=full_values,
                dates=full_dates,
            )
            logger.debug(f"{indicator_id}: char={char_summary.get('dynamical_class', 'N/A')}, engines={len(engines_to_run)}")

        # Fallback if still None
        if engines_to_run is None:
            engines_to_run = CORE_ENGINES | CONDITIONAL_ENGINES

        total_windows = 0
        batch_rows = []
        field_rows = []  # For laplace computation
        computed_at = datetime.now()

        # Generate windows
        min_obs = min(engine_min_obs.values())
        windows = generate_windows(obs_df, window_days, min_obs, stride_days)
        total_windows = len(windows)

        # Track metric history for baseline computation (running baselines)
        metric_history: Dict[str, List[float]] = {k: [] for k in ENGINE_CONFIGS.keys()}

        # Process each window
        for values, obs_date, lookback_start, actual_obs in windows:
            # Collect all metrics for this window (for vector score)
            window_metrics: Dict[str, float] = {}

            # Run each engine (only those in engines_to_run based on characterization)
            for engine_name, engine_func in VECTOR_ENGINES.items():
                # Skip engines not in the engines_to_run set
                if engine_name not in engines_to_run:
                    continue

                min_obs_engine = engine_min_obs.get(engine_name, MIN_OBSERVATIONS)
                if actual_obs < min_obs_engine:
                    continue

                # Compute metrics
                try:
                    try:
                        metrics = engine_func(values, min_obs=min_obs_engine)
                    except TypeError:
                        metrics = engine_func(values)
                except Exception as e:
                    logger.debug(f"Engine {engine_name} failed for {indicator_id}: {e}")
                    continue

                # Collect metrics
                for metric_name, metric_value in metrics.items():
                    try:
                        if metric_value is not None:
                            numeric_value = float(metric_value)
                            if np.isfinite(numeric_value):
                                batch_rows.append(
                                    {
                                        "indicator_id": indicator_id,
                                        "obs_date": obs_date,
                                        "target_obs": window_days,
                                        "actual_obs": actual_obs,
                                        "lookback_start": lookback_start,
                                        "engine": engine_name,
                                        "metric_name": metric_name,
                                        "metric_value": numeric_value,
                                        "computed_at": computed_at,
                                    }
                                )
                                # Track for vector score
                                window_metrics[metric_name] = numeric_value
                                if metric_name in metric_history:
                                    metric_history[metric_name].append(numeric_value)
                    except (TypeError, ValueError):
                        continue

            # Run OBSERVATION_ENGINES (discontinuity detection)
            # Only run if characterization indicates discontinuities exist
            # (or if no characterization, always run to detect)
            min_obs_break = engine_min_obs.get("break_detector", 50)
            should_check_discontinuities = has_discontinuities or (engines_to_run == (CORE_ENGINES | CONDITIONAL_ENGINES))

            if actual_obs >= min_obs_break and should_check_discontinuities:
                try:
                    from prism.engines import get_break_metrics, get_heaviside_metrics, get_dirac_metrics

                    break_metrics = get_break_metrics(values)

                    # Add break_detector metrics
                    for metric_name, metric_value in break_metrics.items():
                        if metric_value is not None:
                            try:
                                numeric_value = float(metric_value)
                                if np.isfinite(numeric_value):
                                    batch_rows.append({
                                        "indicator_id": indicator_id,
                                        "obs_date": obs_date,
                                        "target_obs": window_days,
                                        "actual_obs": actual_obs,
                                        "lookback_start": lookback_start,
                                        "engine": "break_detector",
                                        "metric_name": metric_name,
                                        "metric_value": numeric_value,
                                        "computed_at": computed_at,
                                    })
                                    window_metrics[metric_name] = numeric_value
                            except (TypeError, ValueError):
                                continue

                    # If breaks found in this window, run heaviside and dirac
                    if break_metrics.get('break_n', 0) > 0:
                        break_result = compute_breaks(values)
                        break_indices = break_result['break_indices']

                        # Heaviside (step function measurement)
                        heaviside_metrics = get_heaviside_metrics(values, break_indices)
                        for metric_name, metric_value in heaviside_metrics.items():
                            if metric_value is not None:
                                try:
                                    numeric_value = float(metric_value)
                                    if np.isfinite(numeric_value):
                                        batch_rows.append({
                                            "indicator_id": indicator_id,
                                            "obs_date": obs_date,
                                            "target_obs": window_days,
                                            "actual_obs": actual_obs,
                                            "lookback_start": lookback_start,
                                            "engine": "heaviside",
                                            "metric_name": metric_name,
                                            "metric_value": numeric_value,
                                            "computed_at": computed_at,
                                        })
                                        window_metrics[metric_name] = numeric_value
                                except (TypeError, ValueError):
                                    continue

                        # Dirac (impulse measurement)
                        dirac_metrics = get_dirac_metrics(values, break_indices)
                        for metric_name, metric_value in dirac_metrics.items():
                            if metric_value is not None:
                                try:
                                    numeric_value = float(metric_value)
                                    if np.isfinite(numeric_value):
                                        batch_rows.append({
                                            "indicator_id": indicator_id,
                                            "obs_date": obs_date,
                                            "target_obs": window_days,
                                            "actual_obs": actual_obs,
                                            "lookback_start": lookback_start,
                                            "engine": "dirac",
                                            "metric_name": metric_name,
                                            "metric_value": numeric_value,
                                            "computed_at": computed_at,
                                        })
                                        window_metrics[metric_name] = numeric_value
                                except (TypeError, ValueError):
                                    continue
                except Exception as e:
                    logger.debug(f"Observation engines failed for {indicator_id}: {e}")

            # Compute vector score using running baselines (need 10+ observations)
            if len(metric_history.get('hurst_exponent', [])) >= 10:
                # Compute baselines from history
                baselines = {}
                for metric_name, history in metric_history.items():
                    if len(history) >= 10:
                        baselines[metric_name] = compute_baselines(np.array(history))

                # Compute score
                scores = compute_vector_score(window_metrics, baselines)

                # Add score metrics
                for score_name, score_value in scores.items():
                    if score_name in ('n_engines', 'total_weight'):
                        continue  # Skip metadata
                    if score_value is not None and not np.isnan(score_value):
                        batch_rows.append(
                            {
                                "indicator_id": indicator_id,
                                "obs_date": obs_date,
                                "target_obs": window_days,
                                "actual_obs": actual_obs,
                                "lookback_start": lookback_start,
                                "engine": "vector_score",
                                "metric_name": score_name,
                                "metric_value": float(score_value),
                                "computed_at": computed_at,
                            }
                        )

        # INLINE LAPLACE: Compute field vectors if requested
        field_rows = []
        if compute_laplace_inline and len(batch_rows) > 0:
            # Group batch_rows by (engine, metric_name) for laplace computation
            from collections import defaultdict
            metric_series = defaultdict(list)

            for row in batch_rows:
                key = (row['engine'], row['metric_name'])
                metric_series[key].append({
                    'obs_date': row['obs_date'],
                    'metric_value': row['metric_value'],
                })

            # Compute laplace for each metric series
            for (engine, metric_name), series_data in metric_series.items():
                # Sort by date
                series_data.sort(key=lambda x: x['obs_date'])
                dates = [s['obs_date'] for s in series_data]
                values = np.array([s['metric_value'] for s in series_data])

                # Compute laplace field
                laplace_rows = compute_laplace_for_series(
                    indicator_id=indicator_id,
                    dates=dates,
                    values=values,
                    engine=engine,
                    metric_name=metric_name,
                )
                field_rows.extend(laplace_rows)

            # Compute divergence per window
            if field_rows:
                div_by_window = compute_divergence_for_indicator(field_rows)
                field_rows = add_divergence_to_field_rows(field_rows, div_by_window)

        return {
            "indicator": indicator_id,
            "windows": total_windows,
            "metrics": len(batch_rows),
            "rows": batch_rows,
            "field_rows": field_rows,
            "char_summary": char_summary,
        }

    except Exception as e:
        return {"indicator": indicator_id, "windows": 0, "metrics": 0, "error": str(e)}


def run_window_tier(
    indicators: List[str],
    window_name: str,
    engine_min_obs: Optional[Dict[str, int]] = None,
    use_inline_characterization: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run vector computation for a specific window tier.

    Uses INLINE CHARACTERIZATION: Each indicator is characterized as it's processed,
    rather than reading from a pre-computed characterization parquet.

    Args:
        indicators: List of indicator IDs to process
        window_name: Window tier name ('anchor', 'bridge', 'scout', 'micro')
        engine_min_obs: Minimum observations per engine
        use_inline_characterization: If True, characterize each indicator inline
        verbose: Print progress

    Returns:
        Dict with processing statistics
    """
    config = load_stride_config()
    window = config.get_window(window_name)
    window_days = window.window_days
    stride_days = window.stride_days

    if engine_min_obs is None:
        engine_min_obs = DEFAULT_ENGINE_MIN_OBS

    if verbose:
        logger.info(f"Window tier: {window_name} ({window_days}d, stride {stride_days}d)")
        logger.info(f"Indicators: {len(indicators)}")
        logger.info(f"Inline characterization: {'enabled' if use_inline_characterization else 'disabled'}")

    # Sequential execution with incremental batch writes and progress tracking
    from prism.db.progress_tracker import ProgressTracker

    tracker = ProgressTracker("vector", "indicator")
    target_path = get_parquet_path("vector", "indicator")

    # Filter to pending indicators (resume capability)
    pending = tracker.get_pending(indicators, window_name)
    skipped = len(indicators) - len(pending)

    # Memory tracking
    start_memory = get_memory_usage_mb()

    # Batch accumulator - reduced to 1 for very large datasets (TEP: 175k obs/indicator)
    BATCH_SIZE = 1
    batch_rows = []
    batch_field_rows = []  # Field vectors (laplace)
    batch_indicators = []
    total_windows = 0
    total_metrics = 0
    total_field_rows = 0
    errors = []

    # Paths
    field_path = get_parquet_path("vector", "indicator_field")
    FIELD_KEY_COLS = ["indicator_id", "window_end", "engine", "metric_name"]

    if verbose:
        if skipped > 0:
            logger.info(f"Resuming: {skipped} already completed, {len(pending)} pending")
        logger.info(f"Sequential mode: batch writes every {BATCH_SIZE} indicators")
        logger.info(f"Starting memory: {start_memory:.1f} MB")

    for i, indicator_id in enumerate(pending):
        tracker.mark_started(indicator_id, window_name)

        # INLINE CHARACTERIZATION: engines determined inside process_indicator_sequential
        # Pass engines_to_run=None to trigger inline characterization
        result = process_indicator_sequential(
            indicator_id, window_days, stride_days, engine_min_obs,
            engines_to_run=None,  # Triggers inline characterization
            has_discontinuities=False,  # Will be determined inline
            use_inline_characterization=use_inline_characterization,
            compute_laplace_inline=True,
        )

        if "error" in result:
            tracker.mark_failed(indicator_id, window_name, result.get("error", ""))
            errors.append(result)
            print(f"  X {indicator_id} (error)", flush=True)
            continue

        if "rows" in result:
            batch_rows.extend(result["rows"])
            batch_indicators.append((indicator_id, len(result["rows"])))
            total_windows += result.get("windows", 0)

            # Collect field rows (laplace)
            if "field_rows" in result and result["field_rows"]:
                batch_field_rows.extend(result["field_rows"])

            print(f"  {indicator_id}", end="", flush=True)

        # Write batch when full - COMPUTE → WRITE → RELEASE pattern
        if len(batch_indicators) >= BATCH_SIZE:
            if batch_rows:
                # WRITE vector rows with regime normalization
                norm_config = get_normalization_config()
                df = apply_regime_normalization(
                    batch_rows,
                    window=norm_config.get('window', 252),
                    min_periods=norm_config.get('min_periods', 30),
                )
                upsert_parquet(df, target_path, VECTOR_KEY_COLS)
                total_metrics += len(batch_rows)

                # WRITE field rows (laplace)
                if batch_field_rows:
                    field_df = pl.DataFrame(batch_field_rows, infer_schema_length=None)
                    upsert_parquet(field_df, field_path, FIELD_KEY_COLS)
                    total_field_rows += len(batch_field_rows)
                    del field_df
                    batch_field_rows = []

                for ind_id, row_count in batch_indicators:
                    tracker.mark_completed(ind_id, window_name, rows=row_count)

                row_count = len(batch_rows)

                # RELEASE - explicit memory cleanup
                del df
                del batch_rows
                batch_rows = []
                batch_indicators = []
                force_gc()

                # Report memory
                current_mem = get_memory_usage_mb()
                print(f" -> SAVED {row_count:,} rows ({i+1}/{len(pending)}) [mem: {current_mem:.0f} MB]", flush=True)
            else:
                batch_rows = []
                batch_indicators = []
                batch_field_rows = []

        else:
            print(",", end="", flush=True)

    # Final batch write - COMPUTE → WRITE → RELEASE
    if batch_rows:
        # WRITE vector rows with regime normalization
        norm_config = get_normalization_config()
        df = apply_regime_normalization(
            batch_rows,
            window=norm_config.get('window', 252),
            min_periods=norm_config.get('min_periods', 30),
        )
        upsert_parquet(df, target_path, VECTOR_KEY_COLS)
        total_metrics += len(batch_rows)

        # WRITE remaining field rows (laplace)
        if batch_field_rows:
            field_df = pl.DataFrame(batch_field_rows, infer_schema_length=None)
            upsert_parquet(field_df, field_path, FIELD_KEY_COLS)
            total_field_rows += len(batch_field_rows)
            del field_df

        for ind_id, row_count in batch_indicators:
            tracker.mark_completed(ind_id, window_name, rows=row_count)

        row_count = len(batch_rows)

        # RELEASE
        del df
        del batch_rows
        force_gc()

        current_mem = get_memory_usage_mb()
        print(f" -> SAVED {row_count:,} rows (FINAL) [mem: {current_mem:.0f} MB]", flush=True)

    # Summary with memory delta
    end_memory = get_memory_usage_mb()
    delta = end_memory - start_memory
    print(f"\n[{window_name}] Complete: {total_metrics:,} vector rows, {total_field_rows:,} field rows", flush=True)
    print(f"[{window_name}] Indicators: {len(pending)}", flush=True)
    print(f"[{window_name}] Memory: {start_memory:.0f} → {end_memory:.0f} MB (Δ{delta:+.0f} MB)", flush=True)
    if skipped > 0:
        print(f"[{window_name}] Skipped {skipped} already-completed indicators", flush=True)

    return {
        "window_name": window_name,
        "window_days": window_days,
        "stride_days": stride_days,
        "indicators": len(indicators),
        "windows": total_windows,
        "metrics": total_metrics,
        "field_rows": total_field_rows,
        "errors": len(errors),
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================


def get_indicators(cohort: Optional[str] = None) -> List[str]:
    """Get indicator IDs from parquet files."""
    obs_path = get_parquet_path("raw", "observations")
    if not obs_path.exists():
        return []

    if cohort:
        # Filter by cohort membership
        members_path = get_parquet_path("config", "cohort_members")
        if members_path.exists():
            members = pl.read_parquet(members_path)
            indicator_ids = (
                members.filter(pl.col("cohort_id") == cohort)["indicator_id"].unique().to_list()
            )
            return indicator_ids

    # All indicators
    df = pl.scan_parquet(obs_path).select("indicator_id").unique().collect()
    return df["indicator_id"].to_list()


def run_cohort_vector_aggregation(
    cohorts: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate indicator vectors into cohort vectors.

    Reads indicator-level vector metrics and aggregates by cohort:
    - mean, std, min, max, count per metric per cohort per window

    This gives cohort-level behavior metrics using the same "language"
    as indicator vectors (Hurst, entropy, GARCH, etc.)

    Args:
        cohorts: List of cohort IDs to process (None = all)
        verbose: Print progress

    Returns:
        Summary stats
    """
    from datetime import datetime

    print()
    print("=" * 80)
    print("COHORT VECTOR AGGREGATION")
    print("=" * 80)

    # Load indicator vectors
    indicator_path = get_parquet_path("vector", "indicator")
    if not indicator_path.exists():
        print(f"ERROR: No indicator vectors found at {indicator_path}")
        print("Run --indicator mode first to compute indicator vectors.")
        return {"status": "error", "reason": "no indicator vectors"}

    print(f"Reading indicator vectors from {indicator_path}...")
    vectors_df = pl.read_parquet(indicator_path)
    print(f"  Loaded {len(vectors_df):,} indicator vector rows")

    # Load cohort membership
    members_path = get_parquet_path("config", "cohort_members")
    if not members_path.exists():
        print(f"ERROR: No cohort membership found at {members_path}")
        return {"status": "error", "reason": "no cohort membership"}

    members_df = pl.read_parquet(members_path)
    print(f"  Loaded {len(members_df):,} cohort membership rows")

    # Join vectors with cohort membership
    # vectors: indicator_id, obs_date, target_obs, engine, metric_name, metric_value
    # members: indicator_id, cohort_id
    joined = vectors_df.join(
        members_df.select(["indicator_id", "cohort_id"]),
        on="indicator_id",
        how="inner"
    )
    print(f"  Joined: {len(joined):,} rows with cohort assignments")

    # Filter to specific cohorts if requested
    if cohorts:
        joined = joined.filter(pl.col("cohort_id").is_in(cohorts))
        print(f"  Filtered to cohorts {cohorts}: {len(joined):,} rows")

    # Aggregate by cohort + window + engine + metric
    print()
    print("Aggregating by cohort/window/engine/metric...")

    cohort_vectors = joined.group_by(
        ["cohort_id", "obs_date", "target_obs", "engine", "metric_name"]
    ).agg([
        pl.col("metric_value").mean().alias("metric_mean"),
        pl.col("metric_value").std().alias("metric_std"),
        pl.col("metric_value").min().alias("metric_min"),
        pl.col("metric_value").max().alias("metric_max"),
        pl.col("metric_value").count().alias("n_indicators"),
    ]).with_columns([
        pl.lit(datetime.now()).alias("computed_at")
    ])

    print(f"  Aggregated to {len(cohort_vectors):,} cohort vector rows")

    # Show sample
    if verbose:
        print()
        print("=== SAMPLE COHORT VECTOR ROW ===")
        sample = cohort_vectors.head(1)
        for col in sample.columns:
            print(f"  {col}: {sample[col][0]}")

    # Write output
    output_path = get_parquet_path("vector", "cohort")
    print()
    print(f"Writing to {output_path}...")

    # Use upsert for incremental updates
    key_cols = ["cohort_id", "obs_date", "target_obs", "engine", "metric_name"]
    upsert_parquet(cohort_vectors, output_path, key_cols)

    print(f"  Wrote {len(cohort_vectors):,} cohort vector rows")

    # Summary stats
    n_cohorts = cohort_vectors["cohort_id"].n_unique()
    n_windows = cohort_vectors.select(["obs_date", "target_obs"]).unique().height
    n_engines = cohort_vectors["engine"].n_unique()

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Cohorts: {n_cohorts}")
    print(f"Windows: {n_windows}")
    print(f"Engines: {n_engines}")
    print(f"Total rows: {len(cohort_vectors):,}")

    return {
        "status": "success",
        "cohorts": n_cohorts,
        "windows": n_windows,
        "engines": n_engines,
        "rows": len(cohort_vectors),
    }


def run_sliding_vectors(
    indicators: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute sliding window vectors for indicators.

    Runs anchor + bridge tiers only (structural coverage).
    Scout/micro are for geometry/state delta drill-down.

    INLINE ARCHITECTURE:
    - Characterization: Each indicator characterized as it's processed
    - Laplace: Field vectors computed inline after engine metrics
    - No separate characterize.py or laplace.py runner needed

    Args:
        indicators: List of indicator IDs to process (None = all)
        verbose: Print progress

    Returns:
        Dict with processing statistics
    """
    # Ensure directories exist
    ensure_directories()

    # Get configuration
    engine_min_obs = DEFAULT_ENGINE_MIN_OBS.copy()
    stride_config = load_stride_config()

    # VECTORS: use default tiers from config (anchor + bridge)
    # Scout + micro are drilldown tiers - run when delta flags displacement
    window_tiers = get_default_tiers()

    # Get indicators if not provided
    if indicators is None:
        indicators = get_indicators()

    if not indicators:
        if verbose:
            print("No indicators found")
        return {"indicators": 0, "windows": 0, "metrics": 0, "errors": 0}

    if verbose:
        print("=" * 80)
        print("PRISM VECTOR - INLINE CHARACTERIZATION + LAPLACE")
        print("=" * 80)
        print(f"Storage: Parquet files")
        print(f"Indicators: {len(indicators)}")
        print(f"Characterization: INLINE (per indicator)")
        print(f"Laplace: INLINE (computed with metrics)")
        print(f"Window tiers: {window_tiers}")
        for tier in window_tiers:
            w = stride_config.get_window(tier)
            print(f"  {tier}: {w.window_days}d / {w.stride_days}d stride")
        print(f"Core engines (all): {sorted(CORE_ENGINES)}")
        print(f"Conditional engines: {sorted(CONDITIONAL_ENGINES)}")
        print()

    # Process each window tier
    all_results = []
    for window_name in window_tiers:
        result = run_window_tier(
            indicators=indicators,
            window_name=window_name,
            engine_min_obs=engine_min_obs,
            use_inline_characterization=True,
            verbose=verbose,
        )
        all_results.append(result)

    # Aggregate results
    total_windows = sum(r["windows"] for r in all_results)
    total_metrics = sum(r["metrics"] for r in all_results)
    total_field_rows = sum(r.get("field_rows", 0) for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)

    if verbose:
        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)
        print(f"Indicators processed: {len(indicators)}")
        print(f"Window tiers processed: {len(window_tiers)}")
        print(f"Total windows: {total_windows:,}")
        print(f"Total vector metrics: {total_metrics:,}")
        print(f"Total field rows (laplace): {total_field_rows:,}")
        print(f"Errors: {total_errors}")
        print()
        print(f"Output: vector/indicator.parquet (metrics)")
        print(f"Output: vector/indicator_field.parquet (laplace field)")

    return {
        "indicators": len(indicators),
        "window_tiers": len(window_tiers),
        "windows": total_windows,
        "metrics": total_metrics,
        "field_rows": total_field_rows,
        "errors": total_errors,
        "results": all_results,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PRISM Vector Runner - Config is law",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # MODE FLAGS (mutually exclusive, one required)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--indicator",
        action="store_true",
        help="Process individual indicators (raw/observations -> vector/indicator)",
    )
    mode_group.add_argument(
        "--cohort",
        action="store_true",
        help="Process cohort aggregations (raw/cohort_observations -> vector/cohort)",
    )

    # Production flag
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear progress tracker and recompute everything",
    )

    # Testing mode - REQUIRED to use limiting flags
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing mode. REQUIRED to use limiting flags.",
    )

    # Limiting flags (testing only)
    parser.add_argument("--filter", type=str, help="[TESTING] Comma-separated IDs to process")
    parser.add_argument("--filter-cohort", type=str, help="[TESTING] Filter to specific cohort")
    parser.add_argument("--dates", type=str, help="[TESTING] Date range: YYYY-MM-DD:YYYY-MM-DD")

    # Domain selection (required - prompts if not specified)
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain to process (e.g., cheme, cmapss, climate). Prompts if not specified.",
    )

    args = parser.parse_args()

    # Domain selection - prompt if not specified
    from prism.utils.domain import require_domain
    import os
    domain = require_domain(args.domain, "Select domain for indicator_vector")
    os.environ["PRISM_DOMAIN"] = domain
    print(f"Domain: {domain}", flush=True)

    # --testing guard: limiting flags ignored without --testing
    filter_cohort = getattr(args, 'filter_cohort', None)
    if not args.testing:
        if filter_cohort or args.filter or args.dates:
            logger.warning("=" * 80)
            logger.warning("LIMITING FLAGS IGNORED - --testing not specified")
            logger.warning("Running FULL computation.")
            logger.warning("=" * 80)
        filter_cohort = None
        args.filter = None
        args.dates = None

    # Config is law - load or fail
    stride_config = load_stride_config()
    if not stride_config:
        raise RuntimeError("Cannot load stride config - config/stride.yaml required")

    # Determine source/destination based on mode
    if args.indicator:
        schema = "vector"
        table = "indicator"
        source_schema = "raw"
        source_table = "observations"
    else:  # args.cohort
        schema = "vector"
        table = "cohort"
        source_schema = "raw"
        source_table = "cohort_observations"

    # Handle --force: clear progress tracker for this mode
    if args.force:
        from prism.db.progress_tracker import ProgressTracker
        tracker = ProgressTracker(schema, table)
        tracker.clear()
        print("Progress cleared (--force)", flush=True)

    # Get items to process
    if args.filter:
        items = [i.strip() for i in args.filter.split(",")]
    elif args.indicator:
        items = get_indicators(filter_cohort)
    else:
        # Cohort mode: get cohort list from cohort_members
        members_path = get_parquet_path("config", "cohort_members")
        if members_path.exists():
            items = pl.read_parquet(members_path)["cohort_id"].unique().to_list()
        else:
            items = []

    # TODO: Parse args.dates for date range filtering

    if not items:
        raise RuntimeError(f"No {table} found to process")

    print(f"Mode: {'--indicator' if args.indicator else '--cohort'}", flush=True)
    print(f"Source: {source_schema}/{source_table}", flush=True)
    print(f"Destination: {schema}/{table}", flush=True)
    print(f"Items to process: {len(items)}", flush=True)

    # Run
    if args.indicator:
        # Laplace is computed INLINE - no separate chain needed
        # Each indicator gets: characterization → engine metrics → laplace field
        # All in one pass for efficiency
        run_sliding_vectors(
            indicators=items,
            verbose=True,
        )

    else:
        # Cohort mode - aggregate indicator vectors into cohort vectors
        run_cohort_vector_aggregation(
            cohorts=items if items else None,
            verbose=True,
        )

    # Auto-run assessment report at end of processing
    print("\n" + "=" * 80, flush=True)
    print("RUNNING ASSESSMENT REPORT", flush=True)
    print("=" * 80, flush=True)
    try:
        from prism.assessments.tep_assessment import run_tep_assessment
        run_tep_assessment(domain)
    except Exception as e:
        print(f"Assessment report failed: {e}", flush=True)
