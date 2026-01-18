"""
PRISM Laplace Field Vector (Universal)
======================================

Computes Laplace Field Vector on ANY level of the hierarchy.
Same hammer, different nails.

THE MATH ORGANIZES WHAT THE DOMAIN IS.

Usage:
------
    # Level 1: Signal vectors → Signal field
    python -m prism.entry_points.laplace --level signal

    # Level 2: Geometry → Geometry field
    python -m prism.entry_points.laplace --level geometry

Mathematical Foundation:
------------------------

For any entity E at any level L, within each window:

1. GRADIENT FIELD ∇E:
   ∇E(t) = (E(t+1) - E(t-1)) / 2
   "How fast is the entity changing?"

2. LAPLACIAN FIELD ∇²E:
   ∇²E(t) = E(t+1) - 2E(t) + E(t-1)
   "Is change accelerating or decelerating?"

3. DIVERGENCE:
   div(E) = Σ ∇²E across all metrics
   SOURCE (>0) vs SINK (<0)

4. FIELD POTENTIAL φ:
   φ = Σ|∇E| (accumulated energy)

Pipeline:
---------
observations → signal_vector → laplace(signal) → geometry → laplace(geometry) → physics
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from prism.db.parquet_store import get_path, get_data_root, ensure_directory, OBSERVATIONS, SIGNALS, GEOMETRY
from prism.db.polars_io import write_parquet_atomic, upsert_parquet
from prism.utils.stride import load_stride_config

# Adaptive domain clock integration
from prism.config.loader import load_delta_thresholds
import json

# V2 Architecture: Running Laplace transform
from prism.modules.laplace_transform import (
    RunningLaplace,
    compute_laplace_field as compute_laplace_field_v2,
    laplace_gradient,
    laplace_divergence,
    laplace_energy,
    decompose_by_scale,
)
from prism.modules.signals.types import LaplaceField, DenseSignal


# =============================================================================
# CONFIGURATION (from config/stride.yaml or domain_info.json)
# =============================================================================

@dataclass
class WindowConfig:
    """Window configuration for Laplace field computation."""
    window_days: int
    stride_days: int
    min_observations: int = 10


def load_domain_info() -> Optional[Dict]:
    """
    Load domain_info from config/domain_info.json if available.

    This is saved by signal_vector when running in --adaptive mode.
    Contains auto-detected window parameters based on domain frequency.
    """
    import os
    domain = os.environ.get('PRISM_DOMAIN')
    if not domain:
        return None

    try:
        domain_info_path = get_data_root(domain) / "domain_info.json"
        if domain_info_path.exists():
            with open(domain_info_path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def load_config_from_stride(tier: str = 'anchor') -> WindowConfig:
    """Load window config from domain_info.json or stride.yaml. Fails if not configured."""
    # First check for adaptive domain_info (from DomainClock)
    domain_info = load_domain_info()
    if domain_info:
        window_samples = domain_info.get('window_samples')
        stride_samples = domain_info.get('stride_samples')
        if window_samples:
            return WindowConfig(
                window_days=window_samples,
                stride_days=stride_samples or max(1, window_samples // 3),
                min_observations=10,
            )

    # Fall back to stride.yaml
    try:
        stride_config = load_stride_config()
        if stride_config and hasattr(stride_config, 'windows') and tier in stride_config.windows:
            tier_window = stride_config.windows[tier]
            return WindowConfig(
                window_days=tier_window.window_days,
                stride_days=tier_window.stride_days,
                min_observations=getattr(tier_window, 'min_observations', 10),
            )
    except Exception:
        pass

    # No fallback - must be configured
    raise RuntimeError(
        f"No window config found for tier '{tier}'. "
        "Run signal_vector with --adaptive flag first, or configure config/stride.yaml"
    )


DEFAULT_CONFIG = load_config_from_stride('anchor')

# Thresholds for classification (from config/domain.yaml)
_delta_thresholds = load_delta_thresholds()
SOURCE_THRESHOLD = _delta_thresholds.get('geometry_divergence', 0.1)
SINK_THRESHOLD = -_delta_thresholds.get('geometry_divergence', 0.1)


# =============================================================================
# CORE LAPLACE COMPUTATIONS (LEVEL-AGNOSTIC)
# =============================================================================

def compute_gradient(values: np.ndarray) -> np.ndarray:
    """
    Compute first derivative (gradient) with consistent accuracy at boundaries.

    Interior: Central difference O(h²)
        ∇f(t) = (f(t+1) - f(t-1)) / 2

    Boundaries: One-sided second-order O(h²)
        ∇f(0) = (-3f(0) + 4f(1) - f(2)) / 2
        ∇f(n) = (3f(n) - 4f(n-1) + f(n-2)) / 2

    This avoids the ~2x noise variance at boundaries that occurs with
    first-order forward/backward differences, which can cause artificial
    sources/sinks in the Laplace field.
    """
    n = len(values)
    gradient = np.full(n, np.nan)

    if n < 2:
        return gradient

    # Interior: central difference O(h²)
    if n >= 3:
        gradient[1:-1] = (values[2:] - values[:-2]) / 2.0

    # Boundaries: one-sided second-order O(h²)
    if n >= 3:
        # Forward second-order at first point
        gradient[0] = (-3*values[0] + 4*values[1] - values[2]) / 2.0
        # Backward second-order at last point
        gradient[-1] = (3*values[-1] - 4*values[-2] + values[-3]) / 2.0
    elif n == 2:
        # Fallback for very short series (first-order, scaled consistently)
        gradient[0] = (values[1] - values[0])
        gradient[-1] = (values[-1] - values[-2])

    return gradient


def compute_laplacian(values: np.ndarray) -> np.ndarray:
    """
    Compute second derivative (Laplacian) with consistent accuracy at boundaries.

    Interior: Central difference O(h²)
        ∇²f(t) = f(t+1) - 2f(t) + f(t-1)

    Boundaries: One-sided second-order O(h²) (requires n >= 4)
        ∇²f(0) = 2f(0) - 5f(1) + 4f(2) - f(3)
        ∇²f(n) = 2f(n) - 5f(n-1) + 4f(n-2) - f(n-3)

    This avoids boundary artifacts that can create false regime signals.
    """
    n = len(values)
    laplacian = np.full(n, np.nan)

    if n < 3:
        return laplacian

    # Interior: central difference O(h²)
    laplacian[1:-1] = values[2:] - 2 * values[1:-1] + values[:-2]

    # Boundaries: one-sided second-order O(h²) (requires 4+ points)
    if n >= 4:
        laplacian[0] = 2*values[0] - 5*values[1] + 4*values[2] - values[3]
        laplacian[-1] = 2*values[-1] - 5*values[-2] + 4*values[-3] - values[-4]

    return laplacian


def detect_inflection_points(laplacian: np.ndarray) -> int:
    """Count inflection points where Laplacian changes sign."""
    n = len(laplacian)
    if n < 2:
        return 0

    count = 0
    for i in range(1, n):
        if not np.isnan(laplacian[i]) and not np.isnan(laplacian[i-1]):
            if laplacian[i] * laplacian[i-1] < 0:
                count += 1

    return count


def _to_date(d):
    """Convert datetime or date to date for comparison."""
    if hasattr(d, 'date'):
        return d.date()
    return d


def compute_window_field(
    dates: List,
    values: np.ndarray,
    window_start: datetime,
    window_end: datetime
) -> Optional[Dict]:
    """
    Compute Laplace field quantities for a single window.
    Works at ANY level - signal, cohort, domain.
    """
    # Filter to window (handle both date and datetime types)
    ws = _to_date(window_start)
    we = _to_date(window_end)
    mask = [(_to_date(d) >= ws and _to_date(d) <= we) for d in dates]
    window_values = values[mask]

    if len(window_values) < 5:
        return None

    # Compute field quantities
    gradient = compute_gradient(window_values)
    laplacian = compute_laplacian(window_values)

    # Remove NaN for statistics
    grad_clean = gradient[~np.isnan(gradient)]
    lap_clean = laplacian[~np.isnan(laplacian)]

    if len(grad_clean) < 3 or len(lap_clean) < 3:
        return None

    return {
        'n_obs': len(window_values),
        'gradient_mean': float(np.mean(grad_clean)),
        'gradient_std': float(np.std(grad_clean)),
        'gradient_magnitude': float(np.mean(np.abs(grad_clean))),
        'laplacian_mean': float(np.mean(lap_clean)),
        'laplacian_std': float(np.std(lap_clean)),
        'n_inflections': detect_inflection_points(laplacian),
        'field_potential': float(np.sum(np.abs(grad_clean))),
    }


def generate_windows(
    min_date: datetime,
    max_date: datetime,
    config: WindowConfig
) -> List[Tuple[datetime, datetime]]:
    """Generate (window_start, window_end) tuples."""
    windows = []
    window_delta = timedelta(days=config.window_days)
    stride_delta = timedelta(days=config.stride_days)

    current_end = min_date + window_delta
    while current_end <= max_date:
        window_start = current_end - window_delta
        windows.append((window_start, current_end))
        current_end += stride_delta

    return windows


# =============================================================================
# UNIVERSAL LAPLACE COMPUTATION
# =============================================================================

def compute_laplace_field(
    df: pl.DataFrame,
    entity_col: str,
    date_col: str = 'obs_date',
    value_col: str = None,
    metric_cols: List[str] = None,
    config: Optional[WindowConfig] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute Laplace Field Vector for ANY entity type.

    DELTA LOGIC: Computes gradient/laplacian as CHANGE between consecutive
    windows (sorted by date). No additional windowing - the input data
    already has windows from signal_vector.

    Parameters:
        df: Input DataFrame (from signal_vector with window structure)
        entity_col: Column identifying entities (signal_id, cohort_id, domain_id)
        date_col: Column with dates (window end dates from signal_vector)
        value_col: Column with values (auto-detected if None)
        metric_cols: Columns that define metric groups (e.g., ['engine', 'metric_name'])
        config: Window configuration (for min_observations threshold only)
        verbose: Print progress

    Returns:
        DataFrame with Laplace field quantities per entity-metric-date
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Auto-detect value column
    if value_col is None:
        for col in ['metric_value_norm', 'metric_value', 'value']:
            if col in df.columns:
                value_col = col
                break
        if value_col is None:
            raise ValueError("Could not auto-detect value column")

    # Auto-detect metric grouping columns
    if metric_cols is None:
        metric_cols = []
        if 'engine' in df.columns:
            metric_cols.append('engine')
        if 'metric_name' in df.columns:
            metric_cols.append('metric_name')

    if verbose:
        print("=" * 70)
        print("LAPLACE FIELD VECTOR (DELTA LOGIC)")
        print("=" * 70)
        print(f"  Entity column: {entity_col}")
        print(f"  Date column: {date_col}")
        print(f"  Value column: {value_col}")
        print(f"  Metric columns: {metric_cols}")
        print(f"  Min observations: {config.min_observations}")
        print("  Mode: Computing deltas between consecutive windows")

    # Get entities
    entities = df[entity_col].unique().sort().to_list()
    n_entities = len(entities)

    # Get date range
    all_dates = df[date_col].unique().sort().to_list()
    min_date = min(all_dates)
    max_date = max(all_dates)

    if verbose:
        print(f"  Entities: {n_entities}")
        print(f"  Date range: {min_date} to {max_date}")
        print(f"  Unique window dates: {len(all_dates)}")

    # Process each entity - DELTA LOGIC: compute changes between consecutive windows
    results = []

    for idx, entity in enumerate(entities):
        if verbose and (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{n_entities} entities processed")

        entity_data = df.filter(pl.col(entity_col) == entity)

        # Get unique metric combinations
        if metric_cols:
            metric_combos = entity_data.select(metric_cols).unique().to_dicts()
        else:
            metric_combos = [{}]

        for mc in metric_combos:
            # Filter to this metric combination
            series = entity_data
            for col, val in mc.items():
                series = series.filter(pl.col(col) == val)
            series = series.sort(date_col)

            n_obs = len(series)
            if n_obs < 3:  # Need at least 3 points for central difference
                continue

            dates = series[date_col].to_list()
            values = series[value_col].to_numpy()

            # Compute gradient and laplacian using central differences
            # These are DELTAS between consecutive windows
            gradient = compute_gradient(values)
            laplacian = compute_laplacian(values)

            # Create result row for each date point
            for i in range(n_obs):
                if np.isnan(gradient[i]) and np.isnan(laplacian[i]):
                    continue

                row = {
                    entity_col: entity,
                    'window_end': dates[i],  # Use window_end for geometry compatibility
                    **mc,
                    'metric_value': float(values[i]),
                    'gradient': float(gradient[i]) if not np.isnan(gradient[i]) else None,
                    'laplacian': float(laplacian[i]) if not np.isnan(laplacian[i]) else None,
                    'gradient_magnitude': abs(float(gradient[i])) if not np.isnan(gradient[i]) else None,
                }
                results.append(row)

    if verbose:
        print(f"    {n_entities}/{n_entities} entities processed (complete)")

    # Create DataFrame
    field_df = pl.DataFrame(results)

    if len(field_df) == 0:
        if verbose:
            print("  WARNING: No field data computed!")
        return field_df

    if verbose:
        print(f"\n  Computing divergence...")

    # Compute divergence (aggregate laplacian across metrics per entity-window)
    # Divergence = sum of laplacians across all metrics for an entity at a given time
    divergence_df = field_df.group_by([entity_col, 'window_end']).agg([
        pl.col('laplacian').sum().alias('divergence'),
        pl.col('gradient_magnitude').sum().alias('total_gradient_mag'),
        pl.col('gradient_magnitude').mean().alias('mean_gradient_mag'),
        pl.col('laplacian').count().alias('n_metrics'),
    ])

    # Join back
    field_df = field_df.join(
        divergence_df,
        on=[entity_col, 'window_end'],
        how='left'
    )

    # Classify sources and sinks based on divergence
    field_df = field_df.with_columns([
        (pl.col('divergence') > SOURCE_THRESHOLD).alias('is_source'),
        (pl.col('divergence') < SINK_THRESHOLD).alias('is_sink'),
    ])

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("LAPLACE FIELD SUMMARY (DELTA LOGIC)")
        print("=" * 70)
        print(f"\n  Total rows: {len(field_df):,}")

        unique_ew = field_df.select([entity_col, 'window_end']).unique()
        print(f"  Unique entity-windows: {len(unique_ew):,}")

        div_unique = field_df.select([entity_col, 'window_end', 'divergence']).unique()
        div_mean = div_unique['divergence'].mean()
        div_std = div_unique['divergence'].std()
        print(f"\n  Divergence (sum of laplacians per entity-window):")
        print(f"    Mean: {div_mean:.6f}" if div_mean is not None else "    Mean: N/A")
        print(f"    Std:  {div_std:.6f}" if div_std is not None else "    Std:  N/A")

        source_count = field_df.filter(pl.col('is_source')).select([entity_col, 'window_end']).unique().height
        sink_count = field_df.filter(pl.col('is_sink')).select([entity_col, 'window_end']).unique().height
        print(f"\n  Field topology:")
        print(f"    Source entity-windows (divergence > {SOURCE_THRESHOLD}): {source_count:,}")
        print(f"    Sink entity-windows (divergence < {SINK_THRESHOLD}): {sink_count:,}")

        # Count sign changes in laplacian (inflection points)
        n_positive = field_df.filter(pl.col('laplacian') > 0).height
        n_negative = field_df.filter(pl.col('laplacian') < 0).height
        print(f"\n  Laplacian distribution:")
        print(f"    Positive (accelerating): {n_positive:,}")
        print(f"    Negative (decelerating): {n_negative:,}")

    return field_df


# =============================================================================
# V2 ARCHITECTURE: RUNNING LAPLACE TRANSFORM
# =============================================================================

def compute_running_laplace_fields(
    verbose: bool = True,
    domain: str = None,
    s_values: np.ndarray = None,
) -> Dict[str, LaplaceField]:
    """
    V2 Architecture: Compute Running Laplace fields for all signals.

    Uses the RunningLaplace class with O(1) update per observation.
    Each signal gets a LaplaceField with shape [n_timestamps × n_s].

    Args:
        verbose: Print progress
        domain: Domain name (for path resolution)
        s_values: Laplace s-values (default: logarithmic range)

    Returns:
        Dict mapping signal_id to LaplaceField
    """
    if s_values is None:
        # Default logarithmic range covering multiple timescales
        s_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    # Load raw observations
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        raise FileNotFoundError(f"No observations at {obs_path}")

    if verbose:
        print("=" * 70)
        print("V2 RUNNING LAPLACE TRANSFORM")
        print("=" * 70)
        print(f"  s_values: {s_values}")

    # Scan for signals
    signals = (
        pl.scan_parquet(obs_path)
        .select('signal_id')
        .unique()
        .collect()['signal_id']
        .sort()
        .to_list()
    )

    if verbose:
        print(f"  Signals: {len(signals)}")

    # Compute LaplaceField for each signal
    fields = {}
    for idx, signal_id in enumerate(signals):
        if verbose and (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(signals)} signals processed")

        # Load signal data
        signal_data = (
            pl.scan_parquet(obs_path)
            .filter(pl.col('signal_id') == signal_id)
            .select(['obs_date', 'value'])
            .sort('obs_date')
            .collect()
        )

        if len(signal_data) < 10:
            continue

        timestamps = signal_data['obs_date'].to_numpy()
        values = signal_data['value'].to_numpy()

        # Compute LaplaceField using batch function
        field = compute_laplace_field_v2(
            signal_id=signal_id,
            timestamps=timestamps,
            values=values,
            s_values=s_values,
            normalize=True,
        )

        fields[signal_id] = field

    if verbose:
        print(f"    {len(signals)}/{len(signals)} signals processed (complete)")
        print(f"\n  LaplaceFields computed: {len(fields)}")

    return fields


def laplace_fields_to_rows(
    fields: Dict[str, LaplaceField],
    computed_at: datetime = None,
) -> List[Dict]:
    """
    Convert LaplaceFields to row format for parquet storage.

    Args:
        fields: Dict mapping signal_id to LaplaceField
        computed_at: Computation timestamp

    Returns:
        List of row dictionaries
    """
    if computed_at is None:
        computed_at = datetime.now()

    rows = []
    for signal_id, field in fields.items():
        # Store magnitude and phase at each (t, s) point
        for t_idx, t in enumerate(field.timestamps):
            for s_idx, s in enumerate(field.s_values):
                F_val = field.field[t_idx, s_idx]
                rows.append({
                    'signal_id': signal_id,
                    'timestamp': t,
                    's_value': float(np.real(s)),
                    's_idx': s_idx,
                    'magnitude': float(np.abs(F_val)),
                    'phase': float(np.angle(F_val)),
                    'real': float(np.real(F_val)),
                    'imag': float(np.imag(F_val)),
                    'computed_at': computed_at,
                })

    return rows


def compute_laplace_derived_signals(
    fields: Dict[str, LaplaceField],
    verbose: bool = True,
) -> List[DenseSignal]:
    """
    Compute derived signals from LaplaceFields.

    Derives:
    - Gradient (velocity in Laplace space)
    - Divergence (source/sink indicator)
    - Energy (total spectral energy)
    - Scale decomposition (low/mid/high frequency bands)

    Args:
        fields: Dict mapping signal_id to LaplaceField
        verbose: Print progress

    Returns:
        List of DenseSignal objects
    """
    derived = []

    for signal_id, field in fields.items():
        # Gradient (velocity in Laplace space)
        try:
            grad_signal = laplace_gradient(field)
            derived.append(grad_signal)
        except Exception:
            pass

        # Divergence (source/sink indicator)
        try:
            div_signal = laplace_divergence(field)
            derived.append(div_signal)
        except Exception:
            pass

        # Total energy
        try:
            energy_signal = laplace_energy(field)
            derived.append(energy_signal)
        except Exception:
            pass

        # Scale decomposition
        try:
            scale_signals = decompose_by_scale(field)
            derived.extend(scale_signals)
        except Exception:
            pass

    if verbose:
        print(f"  Derived signals: {len(derived)}")

    return derived


def run_v2_laplace(
    verbose: bool = True,
    domain: str = None,
) -> Dict:
    """
    Run V2 Running Laplace computation.

    Computes LaplaceFields for all signals and stores to parquet.

    Args:
        verbose: Print progress
        domain: Domain name

    Returns:
        Dict with processing statistics
    """
    computed_at = datetime.now()

    # Compute LaplaceFields
    fields = compute_running_laplace_fields(verbose=verbose, domain=domain)

    if not fields:
        if verbose:
            print("  No fields computed!")
        return {'signals': 0, 'rows': 0}

    # Convert to rows for storage
    rows = laplace_fields_to_rows(fields, computed_at)

    if verbose:
        print(f"\n  Saving {len(rows):,} field rows...")

    # Save to parquet
    df = pl.DataFrame(rows, infer_schema_length=None)
    field_path = get_path(SIGNALS)
    upsert_parquet(df, field_path, ['signal_id', 'timestamp', 's_idx'])

    if verbose:
        print(f"  Saved: {field_path}")

    # Compute and save derived signals
    derived = compute_laplace_derived_signals(fields, verbose=verbose)

    if derived:
        derived_rows = []
        for signal in derived:
            if not signal.is_valid:
                continue
            for t, v in zip(signal.timestamps, signal.values):
                if not np.isfinite(v):
                    continue
                derived_rows.append({
                    'signal_id': signal.source_signal,
                    'timestamp': t,
                    'engine': signal.engine,
                    'metric_name': signal.signal_id.split('_')[-1],
                    'value': float(v),
                    'computed_at': computed_at,
                })

        if derived_rows:
            derived_df = pl.DataFrame(derived_rows, infer_schema_length=None)
            derived_path = get_path(SIGNALS)
            upsert_parquet(derived_df, derived_path, ['signal_id', 'timestamp', 'engine', 'metric_name'])

            if verbose:
                print(f"  Saved derived: {derived_path} ({len(derived_rows):,} rows)")

    return {
        'signals': len(fields),
        'field_rows': len(rows),
        'derived_signals': len(derived),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Laplace Field Vector (Universal)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
THE MATH ORGANIZES WHAT THE DOMAIN IS.

Same hammer, different nails:

  # Signal level (default)
  python -m prism.entry_points.laplace --level signal

  # Geometry level
  python -m prism.entry_points.laplace --level geometry --input geometry/cohort.parquet

  # Custom entity
  python -m prism.entry_points.laplace --input mydata.parquet --entity-col cohort_id

  # V2 Running Laplace (new architecture)
  python -m prism.entry_points.laplace --v2
        """
    )
    parser.add_argument(
        '--v2',
        action='store_true',
        help='V2 Architecture: Use Running Laplace transform F(s,t)'
    )
    parser.add_argument(
        '--level',
        type=str,
        choices=['signal', 'geometry'],
        default='signal',
        help='Pipeline level (default: signal)'
    )
    parser.add_argument(
        '--domain',
        type=str,
        default=None,
        help='Domain name (e.g., cheme, cmapss, climate). Prompts if not specified.'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input parquet path'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output parquet path'
    )
    parser.add_argument(
        '--entity-col',
        type=str,
        help='Entity column name (auto-detected if not specified)'
    )
    parser.add_argument(
        '--tier',
        type=str,
        choices=['anchor', 'bridge', 'scout', 'micro'],
        default='anchor',
        help='Window tier (only used if input lacks window columns)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=None,
        help='Override window size in days (default: from config)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=None,
        help='Override stride in days (default: from config)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10,
        help='Number of entities to process per chunk (default: 10)'
    )
    parser.add_argument(
        '--value-col',
        type=str,
        default=None,
        choices=['metric_value', 'metric_value_norm'],
        help='Column to compute Laplace field on. Default: auto-detect (prefers metric_value_norm if present)'
    )

    args = parser.parse_args()

    # Domain selection - prompt if not specified
    from prism.utils.domain import require_domain
    import os
    domain = require_domain(args.domain, "Select domain for laplace")
    os.environ["PRISM_DOMAIN"] = domain
    print(f"Domain: {domain}", flush=True)

    ensure_directory()

    # V2 Architecture: Running Laplace transform
    if args.v2:
        if not args.quiet:
            print("=" * 70)
            print("V2 ARCHITECTURE: Running Laplace Transform")
            print("=" * 70)
        result = run_v2_laplace(verbose=not args.quiet, domain=domain)
        if not args.quiet:
            print()
            print("=" * 70)
            print("COMPLETE")
            print("=" * 70)
            print(f"  Signals processed: {result['signals']}")
            print(f"  Field rows: {result['field_rows']:,}")
            print(f"  Derived signals: {result.get('derived_signals', 0)}")
        return

    # Load config (only used if input lacks window columns)
    config = load_config_from_stride(args.tier)
    if args.window:
        config.window_days = args.window
    if args.stride:
        config.stride_days = args.stride

    if args.level == 'signal':
        if args.input:
            input_path = Path(args.input)
        else:
            input_path = get_path(SIGNALS)

        # Output path: laplace field goes to signals.parquet
        output_path = Path(args.output) if args.output else get_path(SIGNALS)
        entity_col = args.entity_col or 'signal_id'

    elif args.level == 'geometry':
        input_path = Path(args.input) if args.input else get_path(GEOMETRY)
        output_path = Path(args.output) if args.output else get_path(SIGNALS)
        entity_col = args.entity_col or 'cohort_id'

    if not args.quiet:
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Level: {args.level}")
        print(f"Domain: {domain}")
        print(f"Value column: {args.value_col or 'auto-detect'}")
        print(f"Chunk size: {args.chunk_size} entities per chunk")

    # =========================================================================
    # CHUNKED PROCESSING - Don't load entire file at once
    # =========================================================================

    # First, scan lazily to get unique entities
    if not args.quiet:
        print(f"\nScanning for entities...")

    lazy_df = pl.scan_parquet(input_path)

    # Auto-detect entity column
    schema = lazy_df.collect_schema()
    if args.entity_col:
        entity_col = args.entity_col
    elif entity_col not in schema.names():
        for col in ['signal_id', 'cohort_id', 'domain_id', 'entity_id']:
            if col in schema.names():
                entity_col = col
                break

    # Get unique entities (this is cheap - just reads the column)
    entities = lazy_df.select(entity_col).unique().collect()[entity_col].sort().to_list()
    n_entities = len(entities)

    if not args.quiet:
        print(f"  Found {n_entities} entities")

    # Process in chunks
    chunk_size = args.chunk_size
    n_chunks = (n_entities + chunk_size - 1) // chunk_size

    if not args.quiet:
        print(f"  Processing in {n_chunks} chunks of {chunk_size} entities each")
        print("=" * 70)
        print("LAPLACE FIELD VECTOR (DELTA LOGIC) - CHUNKED")
        print("=" * 70)

    all_results = []
    total_rows = 0

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_entities)
        chunk_entities = entities[start_idx:end_idx]

        if not args.quiet:
            print(f"\n  Chunk {chunk_idx + 1}/{n_chunks}: entities {start_idx + 1}-{end_idx}")

        # Load only this chunk's data
        chunk_df = lazy_df.filter(pl.col(entity_col).is_in(chunk_entities)).collect()

        if not args.quiet:
            print(f"    Loaded {len(chunk_df):,} rows for {len(chunk_entities)} entities")

        # Compute Laplace field for this chunk
        chunk_result = compute_laplace_field(
            chunk_df,
            entity_col=entity_col,
            value_col=args.value_col,  # None = auto-detect (prefers metric_value_norm)
            config=config,
            verbose=False  # Suppress per-chunk verbosity
        )

        if len(chunk_result) > 0:
            all_results.append(chunk_result)
            total_rows += len(chunk_result)
            if not args.quiet:
                print(f"    -> {len(chunk_result):,} field rows computed")

        # Free memory
        del chunk_df

    # Combine all results
    if not args.quiet:
        print(f"\n  Combining {len(all_results)} chunk results...")

    if all_results:
        field_df = pl.concat(all_results)
    else:
        field_df = pl.DataFrame()

    if not args.quiet:
        print(f"\n  Total rows: {len(field_df):,}")

    write_parquet_atomic(field_df, output_path)

    if not args.quiet:
        print(f"\nSaved: {output_path}")
        print(f"Rows: {len(field_df):,}")


if __name__ == '__main__':
    main()
