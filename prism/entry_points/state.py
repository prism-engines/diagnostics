"""
PRISM State Runner - Temporal Dynamics from Geometry
====================================================

Orchestrates temporal state dynamics by analyzing geometry evolution over time.

This runner is a PURE ORCHESTRATOR:
- Reads from geometry.structure, geometry.displacement, geometry.signals
- Computes temporal dynamics (energy, tension, phases)
- Detects regime shifts and cross-cohort transmission
- Stores to state.system, state.signal_dynamics, state.transfers

NO COMPUTATION LOGIC - all delegated to canonical engines.

Pipeline:
    geometry.* -> state.*

STATE DYNAMICS ENGINES (5):
    - energy_dynamics:    Energy trends, acceleration, z-scores
    - tension_dynamics:   Dispersion velocity, alignment evolution
    - phase_detector:     Regime shifts, phase classification
    - cohort_aggregator:  Signal-level to cohort-level metrics
    - transfer_detector:  Cross-cohort transmission patterns

Output Schema:
    state.system             - System-level temporal dynamics
    state.signal_dynamics - Per-signal temporal evolution
    state.transfers          - Cross-cohort transmission

Storage: Parquet files (no database locks)

Usage:
    # Run on all available geometry data
    python -m prism.entry_points.state

    # Run on specific date range (requires --testing)
    python -m prism.entry_points.state --testing --start 2020-01-01 --end 2024-12-31

    # Single snapshot (requires --testing)
    python -m prism.entry_points.state --testing --snapshot 2024-01-15

    # Parallel execution
    python -m prism.entry_points.state --workers 4
"""

import argparse
import logging
import numpy as np
import polars as pl
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

from prism.db.parquet_store import (
    ensure_directory,
    get_path,
    get_data_root,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)
from prism.db.polars_io import read_parquet, upsert_parquet
from prism.db.scratch import TempParquet, merge_temp_results
from prism.engines.utils.parallel import (
    WorkerAssignment,
    divide_by_count,
    generate_temp_path,
    run_workers,
)

# Canonical temporal dynamics engines
from prism.engines.energy_dynamics import EnergyDynamicsEngine
from prism.engines.tension_dynamics import TensionDynamicsEngine
from prism.engines.phase_detector import PhaseDetectorEngine
from prism.engines.cohort_aggregator import CohortAggregatorEngine
from prism.engines.transfer_detector import TransferDetectorEngine

# V2 Architecture: State trajectory from geometry snapshots
from prism.state.trajectory import (
    compute_state_trajectory,
    detect_failure_acceleration,
    compute_state_metrics,
    find_acceleration_events,
    compute_trajectory_curvature,
)
from prism.geometry.snapshot import (
    compute_geometry_trajectory,
    snapshot_to_vector,
    get_unified_timestamps,
)
from prism.modules.signals.types import GeometrySnapshot, StateTrajectory, LaplaceField

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION (with adaptive domain clock integration)
# =============================================================================

from prism.config.loader import load_delta_thresholds
import json


def load_domain_info() -> Optional[Dict[str, Any]]:
    """
    Load domain_info from config/domain_info.json if available.

    This is saved by signal_vector when running in --adaptive mode.
    """
    import os
    domain = os.environ.get('PRISM_DOMAIN')
    if not domain:
        return None
    domain_info_path = get_data_root(domain) / "domain_info.json"
    if domain_info_path.exists():
        try:
            with open(domain_info_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# Load delta thresholds from config/domain.yaml
_delta_thresholds = load_delta_thresholds()

# State layer thresholds (from domain.yaml or defaults)
VELOCITY_THRESHOLD = _delta_thresholds.get('state_velocity', 0.10)
ACCELERATION_THRESHOLD = _delta_thresholds.get('state_acceleration', 0.05)


def get_lookback_window() -> int:
    """
    Get lookback window from domain_info. Fails if not configured.

    Uses window_samples from DomainClock.
    """
    domain_info = load_domain_info()
    if domain_info:
        window = domain_info.get('window_samples')
        if window:
            return max(20, window)  # Ensure minimum for statistics

    # No fallback - must be configured
    raise RuntimeError(
        "No domain_info.json found. "
        "Run signal_vector with --adaptive flag first to auto-detect window parameters."
    )


def get_default_stride() -> int:
    """
    Get default stride from domain_info. Fails if not configured.

    Uses stride_samples from DomainClock.
    """
    domain_info = load_domain_info()
    if domain_info:
        stride = domain_info.get('stride_samples')
        if stride:
            return max(1, stride)

    # No fallback - must be configured
    raise RuntimeError(
        "No domain_info.json found. "
        "Run signal_vector with --adaptive flag first to auto-detect stride parameters."
    )


# MIN_HISTORY is a statistical minimum, not domain-specific
MIN_HISTORY = 20  # Minimum snapshots needed for dynamics

# Key columns for upsert deduplication
SYSTEM_KEY_COLS = ['state_time']
INDICATOR_DYNAMICS_KEY_COLS = ['signal_id', 'state_time']
TRANSFERS_KEY_COLS = ['state_time', 'cohort_from', 'cohort_to']


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_available_dates() -> List[date]:
    """Get all dates with geometry data from parquet files."""
    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return []

    df = pl.read_parquet(geometry_path, columns=['window_end'])
    if len(df) == 0:
        return []

    # Get unique dates sorted
    dates = df.select('window_end').unique().sort('window_end')
    return dates['window_end'].to_list()


def get_geometry_history(
    end_date: date,
    lookback: int = None
) -> pl.DataFrame:
    """
    Fetch geometry.structure history for dynamics computation.

    Returns DataFrame with columns from geometry.structure.
    """
    if lookback is None:
        lookback = get_lookback_window()

    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return pl.DataFrame()

    start_date = end_date - timedelta(days=lookback * 2)  # Buffer for sparse data

    df = pl.read_parquet(geometry_path)
    df = df.filter(
        (pl.col('window_end') >= start_date) &
        (pl.col('window_end') <= end_date)
    ).select([
        'window_end',
        'n_signals',
        'pca_variance_1',
        'pca_variance_2',
        'pca_variance_3',
        'pca_cumulative_3',
        'n_clusters',
        'total_dispersion',
        'mean_alignment',
        'system_coherence',
        'system_energy',
    ]).sort('window_end')

    return df


def get_displacement_history(
    end_date: date,
    lookback: int = None
) -> pl.DataFrame:
    """
    Fetch geometry.displacement history for dynamics computation.
    """
    if lookback is None:
        lookback = get_lookback_window()

    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return pl.DataFrame()

    start_date = end_date - timedelta(days=lookback * 2)

    df = pl.read_parquet(geometry_path)
    df = df.filter(
        (pl.col('window_end_to') >= start_date) &
        (pl.col('window_end_to') <= end_date)
    ).select([
        pl.col('window_end_to').alias('window_end'),
        'days_elapsed',
        'energy_total',
        'energy_63',
        'energy_126',
        'energy_252',
        'anchor_ratio',
        'barycenter_shift_mean',
        'dispersion_delta',
        'dispersion_velocity',
        'regime_conviction',
    ]).sort('window_end')

    return df


def get_signal_geometry(window_end: date) -> pl.DataFrame:
    """
    Fetch geometry.signals for a specific date.
    """
    signals_path = get_path(SIGNALS)
    if not signals_path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(signals_path)
    df = df.filter(pl.col('window_end') == window_end).select([
        'signal_id',
        'barycenter',
        'timescale_dispersion',
        'timescale_alignment',
    ])

    return df


def get_cohort_membership() -> Dict[str, List[str]]:
    """Get cohort membership mapping from parquet."""
    members_path = get_path(COHORTS)
    if not members_path.exists():
        return {}

    df = pl.read_parquet(members_path)
    df = df.filter(pl.col('cohort_id').is_not_null()).sort(['cohort_id', 'signal_id'])

    membership = {}
    for row in df.iter_rows(named=True):
        cohort_id = row['cohort_id']
        if cohort_id not in membership:
            membership[cohort_id] = []
        membership[cohort_id].append(row['signal_id'])

    return membership


# =============================================================================
# STATE COMPUTATION
# =============================================================================

def compute_system_state(
    structure_history: pl.DataFrame,
    displacement_history: pl.DataFrame,
    state_date: date
) -> Dict[str, Any]:
    """
    Compute system-level state for a single date.

    Uses energy_dynamics, tension_dynamics, and phase_detector engines.
    """
    if len(structure_history) == 0 or len(displacement_history) == 0:
        return {}

    # Initialize engines
    energy_engine = EnergyDynamicsEngine()
    tension_engine = TensionDynamicsEngine()
    phase_engine = PhaseDetectorEngine()

    # Convert to pandas for engine compatibility (engines expect pandas Series)
    struct_pd = structure_history.to_pandas().set_index('window_end')
    disp_pd = displacement_history.to_pandas().set_index('window_end')

    # Prepare series
    energy_series = disp_pd['energy_total']
    dispersion_series = struct_pd['total_dispersion']
    alignment_series = struct_pd['mean_alignment']
    coherence_series = struct_pd['system_coherence']

    # Get current values
    current_disp = displacement_history.filter(pl.col('window_end') == state_date)
    current_struct = structure_history.filter(pl.col('window_end') == state_date)

    if len(current_disp) == 0 and len(current_struct) == 0:
        return {}

    # Energy dynamics
    energy_result = energy_engine.run(energy_series)

    # Tension dynamics
    tension_result = tension_engine.run(
        dispersion_series,
        alignment_series,
        coherence_series
    )

    # Get anchor_ratio and regime_conviction from current displacement
    if len(current_disp) > 0:
        anchor_ratio = current_disp['anchor_ratio'][0]
        regime_conviction = current_disp['regime_conviction'][0]
    else:
        anchor_ratio = 0.0
        regime_conviction = 0.0

    # Phase detection
    phase_result = phase_engine.run(
        energy_total=energy_result.energy_total,
        energy_zscore=energy_result.energy_zscore or 0.0,
        energy_trend=energy_result.energy_trend,
        dispersion_total=tension_result.dispersion_total,
        tension_state=tension_result.tension_state,
        alignment=tension_result.alignment_mean,
        anchor_ratio=anchor_ratio,
        regime_conviction=regime_conviction
    )

    # Get PCA concentration
    if len(current_struct) > 0:
        pca_concentration = current_struct['pca_cumulative_3'][0]
    else:
        pca_concentration = 0.0

    return {
        'state_time': state_date,
        'energy_total': energy_result.energy_total,
        'energy_ma5': energy_result.energy_ma5,
        'energy_ma20': energy_result.energy_ma20,
        'energy_acceleration': energy_result.energy_acceleration,
        'energy_zscore': energy_result.energy_zscore,
        'dispersion_total': tension_result.dispersion_total,
        'dispersion_velocity': tension_result.dispersion_velocity,
        'dispersion_acceleration': tension_result.dispersion_acceleration,
        'alignment_mean': tension_result.alignment_mean,
        'coherence': tension_result.coherence,
        'pca_concentration': pca_concentration,
        'regime_conviction': regime_conviction,
        'anchor_ratio': anchor_ratio,
        'phase_score': phase_result.phase_score,
        'phase_label': phase_result.phase_label,
        'is_regime_shift': bool(phase_result.is_regime_shift),
        'shift_confidence': phase_result.shift_confidence,
    }


def compute_signal_dynamics(
    state_date: date,
    prev_date: Optional[date]
) -> List[Dict[str, Any]]:
    """
    Compute signal-level dynamics for a single date.
    """
    current = get_signal_geometry(state_date)

    if len(current) == 0:
        return []

    results = []

    if prev_date:
        previous = get_signal_geometry(prev_date)
        prev_dict = {row['signal_id']: row for row in previous.iter_rows(named=True)}
    else:
        prev_dict = {}

    # Compute system centroid
    barycenters = []
    for row in current.iter_rows(named=True):
        bc = row['barycenter']
        if bc is not None and len(bc) > 0:
            barycenters.append(np.array(bc))

    if barycenters:
        system_centroid = np.mean(barycenters, axis=0)
    else:
        system_centroid = None

    for row in current.iter_rows(named=True):
        ind_id = row['signal_id']
        bc = row['barycenter']
        disp = row['timescale_dispersion']
        align = row['timescale_alignment']

        result = {
            'signal_id': ind_id,
            'state_time': state_date,
            'dispersion': disp,
            'alignment': align,
            'barycenter_shift': 0.0,
            'barycenter_velocity': 0.0,
            'barycenter_acceleration': 0.0,
            'dispersion_delta': 0.0,
            'distance_to_centroid': 0.0,
            'cluster_id': 0,
            'cluster_changed': False,
            'leads_system': False,
            'lags_system': False,
            'lead_lag_days': 0,
        }

        # Compute shift from previous
        if ind_id in prev_dict:
            prev_row = prev_dict[ind_id]
            prev_bc = prev_row['barycenter']

            if bc is not None and prev_bc is not None and len(bc) > 0 and len(prev_bc) > 0:
                from scipy.spatial.distance import euclidean
                result['barycenter_shift'] = euclidean(np.array(bc), np.array(prev_bc))

            if prev_row['timescale_dispersion'] is not None:
                result['dispersion_delta'] = disp - prev_row['timescale_dispersion']

        # Distance to system centroid
        if system_centroid is not None and bc is not None and len(bc) > 0:
            from scipy.spatial.distance import euclidean
            result['distance_to_centroid'] = euclidean(np.array(bc), system_centroid)

        results.append(result)

    return results


def compute_cross_cohort_transfers(
    state_date: date,
    lookback: int = 30
) -> List[Dict[str, Any]]:
    """
    Compute cross-cohort transfer metrics.
    """
    # Get cohort membership
    membership = get_cohort_membership()

    if len(membership) < 2:
        return []

    # Get energy history by cohort
    start_date = state_date - timedelta(days=lookback * 2)

    # Get displacement data
    geometry_path = get_path(GEOMETRY)
    if not geometry_path.exists():
        return []

    disp_df = pl.read_parquet(geometry_path)
    disp_df = disp_df.filter(
        (pl.col('window_end_to') >= start_date) &
        (pl.col('window_end_to') <= state_date)
    ).select([
        pl.col('window_end_to').alias('window_end'),
        'energy_total',
    ]).sort('window_end')

    if len(disp_df) == 0 or len(disp_df) < 10:
        return []

    # For simplified version, detect transfers between cohorts
    # using system-level metrics as proxy
    # Full implementation would use cohort-specific aggregates

    results = []
    cohorts = list(membership.keys())
    transfer_engine = TransferDetectorEngine()

    # Convert to pandas for engine compatibility
    disp_pd = disp_df.to_pandas().set_index('window_end')
    energy_series = disp_pd['energy_total']

    for i, cohort_a in enumerate(cohorts):
        for cohort_b in cohorts[i+1:]:
            # Simplified: use same series for demo
            # Real implementation: aggregate signal-level metrics per cohort
            result = transfer_engine.run(
                energy_series,
                energy_series,  # Would be cohort_b's series
                cohort_a,
                cohort_b
            )

            results.append({
                'state_time': state_date,
                'cohort_from': result.cohort_from,
                'cohort_to': result.cohort_to,
                'transfer_strength': result.transfer_strength,
                'transfer_lag': result.transfer_lag,
                'transfer_direction': result.transfer_direction,
                'granger_fstat': result.granger_fstat,
                'granger_pvalue': result.granger_pvalue,
                'te_net': None,  # Would come from TransferEntropyEngine
                'correlation': result.correlation,
                'correlation_lag': result.correlation_lag,
                'is_significant': result.is_significant,
                'transfer_type': result.transfer_type,
            })

    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_state_snapshot(
    state_date: date,
    prev_date: Optional[date] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute state for a single snapshot date.

    Returns dict with results to store.
    """
    results = {'system': None, 'signals': [], 'transfers': []}

    # Get history
    structure_history = get_geometry_history(state_date)
    displacement_history = get_displacement_history(state_date)

    if len(structure_history) == 0 or len(displacement_history) == 0:
        if verbose:
            logger.warning(f"  {state_date}: No geometry data")
        return results

    # 1. System state
    system_state = compute_system_state(structure_history, displacement_history, state_date)
    if system_state:
        results['system'] = system_state

    # 2. Signal dynamics
    signal_dynamics = compute_signal_dynamics(state_date, prev_date)
    if signal_dynamics:
        results['signals'] = signal_dynamics

    # 3. Cross-cohort transfers
    transfers = compute_cross_cohort_transfers(state_date)
    if transfers:
        results['transfers'] = transfers

    if verbose and system_state:
        phase = system_state.get('phase_label', 'unknown')
        shift = system_state.get('is_regime_shift', False)
        logger.info(f"  {state_date}: phase={phase}, shift={shift}, signals={len(signal_dynamics)}")

    return results


def run_state_range(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    stride: int = None,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Run state computation for a date range.
    """
    if stride is None:
        stride = get_default_stride()

    ensure_directory()

    # Get available dates
    available_dates = get_available_dates()

    if not available_dates:
        logger.error("No geometry data found. Run geometry runner first.")
        return {'system': 0, 'signals': 0, 'transfers': 0}

    # Filter to range
    if start_date:
        available_dates = [d for d in available_dates if d >= start_date]
    if end_date:
        available_dates = [d for d in available_dates if d <= end_date]

    if not available_dates:
        logger.error(f"No data in range {start_date} to {end_date}")
        return {'system': 0, 'signals': 0, 'transfers': 0}

    # Apply stride
    if stride > 1:
        strided_dates = available_dates[::stride]
    else:
        strided_dates = available_dates

    if verbose:
        logger.info("=" * 80)
        logger.info("PRISM STATE RUNNER - TEMPORAL DYNAMICS")
        logger.info("=" * 80)
        logger.info(f"Storage: Parquet files")
        logger.info(f"Date range: {strided_dates[0]} to {strided_dates[-1]}")
        logger.info(f"Snapshots: {len(strided_dates)} (stride={stride})")
        logger.info("")

    totals = {'system': 0, 'signals': 0, 'transfers': 0}
    prev_date = None
    computed_at = datetime.now()

    # Collect all results
    system_rows = []
    signal_rows = []
    transfer_rows = []

    for i, state_date in enumerate(strided_dates):
        results = run_state_snapshot(state_date, prev_date, verbose)

        if results['system']:
            results['system']['computed_at'] = computed_at
            system_rows.append(results['system'])
            totals['system'] += 1

        for ind in results['signals']:
            ind['computed_at'] = computed_at
            signal_rows.append(ind)
        totals['signals'] += len(results['signals'])

        for tr in results['transfers']:
            tr['computed_at'] = computed_at
            transfer_rows.append(tr)
        totals['transfers'] += len(results['transfers'])

        prev_date = state_date

        # Periodic write (every 50 dates)
        if (i + 1) % 50 == 0:
            if system_rows:
                df = pl.DataFrame(system_rows)
                upsert_parquet(df, get_path(STATE), SYSTEM_KEY_COLS)
                system_rows = []
            if signal_rows:
                df = pl.DataFrame(signal_rows)
                upsert_parquet(df, get_path(STATE), INDICATOR_DYNAMICS_KEY_COLS)
                signal_rows = []
            if transfer_rows:
                df = pl.DataFrame(transfer_rows)
                upsert_parquet(df, get_path(STATE), TRANSFERS_KEY_COLS)
                transfer_rows = []

    # Final write
    if system_rows:
        df = pl.DataFrame(system_rows)
        upsert_parquet(df, get_path(STATE), SYSTEM_KEY_COLS)
    if signal_rows:
        df = pl.DataFrame(signal_rows)
        upsert_parquet(df, get_path(STATE), INDICATOR_DYNAMICS_KEY_COLS)
    if transfer_rows:
        df = pl.DataFrame(transfer_rows)
        upsert_parquet(df, get_path(STATE), TRANSFERS_KEY_COLS)

    if verbose:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"COMPLETE: {totals['system']} system states, "
                    f"{totals['signals']} signal dynamics, "
                    f"{totals['transfers']} transfers")
        logger.info("=" * 80)

    return totals


# =============================================================================
# PARALLEL WORKER
# =============================================================================

def process_state_parallel(assignment: WorkerAssignment) -> Dict[str, Any]:
    """Worker function for parallel state computation."""
    dates = assignment.items
    temp_path = assignment.temp_path
    config = assignment.config

    totals = {'system': 0, 'signals': 0, 'transfers': 0}
    prev_date = None
    computed_at = datetime.now()

    # Collect all results
    system_rows = []
    signal_rows = []
    transfer_rows = []

    try:
        for state_date in dates:
            # Get history
            structure_history = get_geometry_history(state_date)
            displacement_history = get_displacement_history(state_date)

            if len(structure_history) == 0 or len(displacement_history) == 0:
                continue

            # Compute system state
            system_state = compute_system_state(structure_history, displacement_history, state_date)
            if system_state:
                system_state['computed_at'] = computed_at
                system_rows.append(system_state)
                totals['system'] += 1

            # Compute signal dynamics
            signal_dynamics = compute_signal_dynamics(state_date, prev_date)
            for ind in signal_dynamics:
                ind['computed_at'] = computed_at
                signal_rows.append(ind)
            totals['signals'] += len(signal_dynamics)

            # Cross-cohort transfers (less frequently)
            if totals['system'] % 5 == 0:
                transfers = compute_cross_cohort_transfers(state_date)
                for tr in transfers:
                    tr['computed_at'] = computed_at
                    transfer_rows.append(tr)
                totals['transfers'] += len(transfers)

            prev_date = state_date

        # Write all results to temp parquet
        # We write a combined dataframe with a 'table_type' column to distinguish
        all_rows = []

        for row in system_rows:
            row['_table'] = 'system'
            all_rows.append(row)

        for row in signal_rows:
            row['_table'] = 'signal_dynamics'
            all_rows.append(row)

        for row in transfer_rows:
            row['_table'] = 'transfers'
            all_rows.append(row)

        if all_rows:
            # Write combined data to temp path
            df = pl.DataFrame(all_rows, infer_schema_length=None)
            df.write_parquet(temp_path)

    except Exception as e:
        logger.error(f"Worker error: {e}")
        raise

    return totals


def merge_state_results(temp_paths: List[Path], verbose: bool = True) -> Dict[str, int]:
    """
    Merge worker temp files into main state parquet files.
    """
    totals = {'system': 0, 'signal_dynamics': 0, 'transfers': 0}

    # Read all temp files
    all_dfs = []
    for path in temp_paths:
        if path.exists():
            try:
                df = pl.read_parquet(path)
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read temp file {path}: {e}")

    if not all_dfs:
        return totals

    combined = pl.concat(all_dfs, how='diagonal_relaxed')

    # Split by table type and write to respective parquet files
    for table_name, key_cols in [
        ('system', SYSTEM_KEY_COLS),
        ('signal_dynamics', INDICATOR_DYNAMICS_KEY_COLS),
        ('transfers', TRANSFERS_KEY_COLS),
    ]:
        table_df = combined.filter(pl.col('_table') == table_name).drop('_table')

        if len(table_df) > 0:
            target_path = get_path(STATE)
            upsert_parquet(table_df, target_path, key_cols)
            totals[table_name] = len(table_df)

            if verbose:
                logger.info(f"  Merged {len(table_df):,} rows to state.{table_name}")

    # Cleanup temp files
    for path in temp_paths:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    return totals


# =============================================================================
# V2 ARCHITECTURE: STATE TRAJECTORY FROM GEOMETRY SNAPSHOTS
# =============================================================================

def load_geometry_snapshots_v2() -> List[GeometrySnapshot]:
    """
    Load V2 GeometrySnapshots from parquet storage.

    Returns:
        List of GeometrySnapshot objects sorted by timestamp
    """
    geom_path = get_path(GEOMETRY)
    if not geom_path.exists():
        logger.warning(f"No V2 geometry snapshots at {geom_path}. Run geometry --v2 first.")
        return []

    df = pl.read_parquet(geom_path).sort('timestamp')

    # Also load coupling matrices if available
    coupling_path = get_path(GEOMETRY)
    coupling_by_timestamp = {}
    if coupling_path.exists():
        coupling_df = pl.read_parquet(coupling_path)
        for ts in coupling_df['timestamp'].unique().to_list():
            ts_data = coupling_df.filter(pl.col('timestamp') == ts)
            coupling_by_timestamp[ts] = ts_data

    snapshots = []
    for row in df.iter_rows(named=True):
        ts = row['timestamp']
        signal_ids = row['signal_ids'].split(',') if row['signal_ids'] else []
        n_signals = row['n_signals']

        # Reconstruct coupling matrix if available
        if ts in coupling_by_timestamp and n_signals > 0:
            ts_coupling = coupling_by_timestamp[ts]
            coupling_matrix = np.eye(n_signals)  # Start with identity

            for c_row in ts_coupling.iter_rows(named=True):
                try:
                    i = signal_ids.index(c_row['signal_a'])
                    j = signal_ids.index(c_row['signal_b'])
                    coupling_matrix[i, j] = c_row['coupling']
                    coupling_matrix[j, i] = c_row['coupling']
                except ValueError:
                    pass
        else:
            coupling_matrix = np.array([[]])

        # Create mode arrays from summary values
        n_modes = row['n_modes']
        mode_labels = np.zeros(n_signals, dtype=int) if n_signals > 0 else np.array([])
        mode_coherence = np.array([row['mean_mode_coherence']]) if n_modes > 0 else np.array([])

        snapshots.append(GeometrySnapshot(
            timestamp=float(ts) if isinstance(ts, (int, float)) else ts.timestamp() if hasattr(ts, 'timestamp') else 0.0,
            coupling_matrix=coupling_matrix,
            divergence=row['divergence'],
            mode_labels=mode_labels,
            mode_coherence=mode_coherence,
            signal_ids=signal_ids,
        ))

    logger.info(f"Loaded {len(snapshots)} GeometrySnapshots from {geom_path}")
    return snapshots


def state_trajectory_to_rows(
    trajectory: StateTrajectory,
    computed_at: datetime = None,
) -> List[Dict]:
    """
    Convert StateTrajectory to row format for parquet storage.

    Args:
        trajectory: StateTrajectory object
        computed_at: Computation timestamp

    Returns:
        List of row dictionaries
    """
    if computed_at is None:
        computed_at = datetime.now()

    rows = []
    n_timestamps = len(trajectory.timestamps)

    for i in range(n_timestamps):
        # Compute scalar metrics at this timestamp
        speed = trajectory.speed[i] if hasattr(trajectory, 'speed') else np.linalg.norm(trajectory.velocity[i])
        accel_mag = trajectory.acceleration_magnitude[i] if hasattr(trajectory, 'acceleration_magnitude') else np.linalg.norm(trajectory.acceleration[i])

        rows.append({
            'timestamp': trajectory.timestamps[i],
            'speed': float(speed),
            'acceleration_magnitude': float(accel_mag),
            'position_dim': int(trajectory.position.shape[1]) if len(trajectory.position.shape) > 1 else 1,
            'computed_at': computed_at,
        })

    return rows


def run_v2_state(
    verbose: bool = True,
) -> Dict:
    """
    Run V2 state trajectory computation.

    Loads GeometrySnapshots, computes state trajectory (position, velocity,
    acceleration), detects failure signatures, saves to parquet.

    Args:
        verbose: Print progress

    Returns:
        Dict with processing statistics
    """
    computed_at = datetime.now()

    # Load geometry snapshots
    snapshots = load_geometry_snapshots_v2()

    if not snapshots:
        logger.warning("No snapshots loaded. Run geometry --v2 first.")
        return {'snapshots': 0}

    if verbose:
        logger.info("=" * 80)
        logger.info("V2 ARCHITECTURE: State Trajectory from Geometry")
        logger.info("=" * 80)
        logger.info(f"  Snapshots: {len(snapshots)}")

    # Compute state trajectory
    trajectory = compute_state_trajectory(snapshots)

    if verbose:
        logger.info(f"  Trajectory computed: {len(trajectory.timestamps)} timestamps")

    # Compute state metrics
    metrics = compute_state_metrics(trajectory)

    if verbose:
        logger.info(f"  Mean velocity: {metrics['mean_velocity']:.6f}")
        logger.info(f"  Mean acceleration: {metrics['mean_acceleration']:.6f}")

    # Detect failure acceleration signatures
    failure_mask = detect_failure_acceleration(trajectory)
    n_failure_timestamps = int(np.sum(failure_mask))

    if verbose:
        logger.info(f"  Failure signature timestamps: {n_failure_timestamps}")

    # Find acceleration events
    events = find_acceleration_events(trajectory)

    if verbose:
        logger.info(f"  Significant acceleration events: {len(events)}")

    # Compute trajectory curvature
    curvature = compute_trajectory_curvature(trajectory)
    mean_curvature = float(np.mean(curvature[~np.isnan(curvature)])) if len(curvature) > 0 else 0.0

    if verbose:
        logger.info(f"  Mean trajectory curvature: {mean_curvature:.6f}")

    # Convert to rows for storage
    rows = state_trajectory_to_rows(trajectory, computed_at)

    # Add failure flag and curvature to rows
    for i, row in enumerate(rows):
        row['is_failure_signature'] = bool(failure_mask[i]) if i < len(failure_mask) else False
        row['curvature'] = float(curvature[i]) if i < len(curvature) and not np.isnan(curvature[i]) else 0.0

    if verbose:
        logger.info(f"\n  Saving {len(rows)} state trajectory rows...")

    # Save trajectory to parquet
    df = pl.DataFrame(rows, infer_schema_length=None)
    state_path = get_path(STATE)
    upsert_parquet(df, state_path, ['timestamp'])

    if verbose:
        logger.info(f"  Saved: {state_path}")

    # Save events to separate file
    if events:
        event_rows = []
        for event in events:
            event['computed_at'] = computed_at
            event_rows.append(event)

        events_df = pl.DataFrame(event_rows, infer_schema_length=None)
        events_path = get_path(STATE)
        upsert_parquet(events_df, events_path, ['peak_idx'])

        if verbose:
            logger.info(f"  Saved events: {events_path} ({len(events)} events)")

    # Save summary metrics
    summary = {
        **metrics,
        'n_failure_timestamps': n_failure_timestamps,
        'n_events': len(events),
        'mean_curvature': mean_curvature,
        'computed_at': computed_at,
    }
    summary_df = pl.DataFrame([summary], infer_schema_length=None)
    summary_path = get_path(STATE)
    summary_df.write_parquet(summary_path)

    if verbose:
        logger.info(f"  Saved summary: {summary_path}")

    return {
        'snapshots': len(snapshots),
        'timestamps': len(trajectory.timestamps),
        'failure_timestamps': n_failure_timestamps,
        'events': len(events),
        'saved_rows': len(rows),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM State Runner - Temporal Dynamics from Geometry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all geometry data
  python -m prism.entry_points.state

  # Run on specific date range (requires --testing)
  python -m prism.entry_points.state --testing --start 2020-01-01 --end 2024-12-31

  # Single snapshot (requires --testing)
  python -m prism.entry_points.state --testing --snapshot 2024-01-15

  # Parallel execution
  python -m prism.entry_points.state --workers 4

Storage: Parquet files (no database locks)
  data/state/system.parquet             - System-level temporal dynamics
  data/state/signal_dynamics.parquet - Per-signal temporal evolution
  data/state/transfers.parquet          - Cross-cohort transmission patterns
"""
    )

    parser.add_argument('--v2', action='store_true',
                        help='V2 Architecture: Compute state trajectory from geometry snapshots')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--snapshot', type=str, help='Single snapshot date')
    parser.add_argument('--stride', type=int, default=None,
                        help='Days between state computations (default: from domain_info.json)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    # Testing mode - REQUIRED to use any limiting flags
    parser.add_argument('--testing', action='store_true',
                        help='Enable testing mode. REQUIRED to use limiting flags (--start, --end, --snapshot). Without --testing, all limiting flags are ignored and full run executes.')

    args = parser.parse_args()

    # V2 Architecture: State trajectory from geometry snapshots
    if args.v2:
        ensure_directory()
        result = run_v2_state(verbose=not args.quiet)
        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Snapshots processed: {result.get('snapshots', 0)}")
        logger.info(f"  Timestamps: {result.get('timestamps', 0)}")
        logger.info(f"  Failure signatures: {result.get('failure_timestamps', 0)}")
        logger.info(f"  Acceleration events: {result.get('events', 0)}")
        logger.info(f"  Saved rows: {result.get('saved_rows', 0)}")
        return 0

    # ==========================================================================
    # CRITICAL: --testing guard
    # Without --testing, ALL limiting flags are ignored and full run executes.
    # ==========================================================================
    if not args.testing:
        limiting_flags_used = []
        if args.start:
            limiting_flags_used.append('--start')
        if args.end:
            limiting_flags_used.append('--end')
        if args.snapshot:
            limiting_flags_used.append('--snapshot')

        if limiting_flags_used:
            logger.warning("=" * 80)
            logger.warning("LIMITING FLAGS IGNORED - --testing not specified")
            logger.warning(f"Ignored flags: {', '.join(limiting_flags_used)}")
            logger.warning("Running FULL computation instead. Use --testing to enable limiting flags.")
            logger.warning("=" * 80)

        # Override to full defaults
        args.start = None
        args.end = None
        args.snapshot = None

    # Parse dates
    start_date = None
    end_date = None

    if args.snapshot:
        from datetime import datetime as dt
        snapshot_date = dt.strptime(args.snapshot, '%Y-%m-%d').date()

        ensure_directory()
        results = run_state_snapshot(snapshot_date, verbose=not args.quiet)

        # Write single snapshot results
        computed_at = datetime.now()
        if results['system']:
            results['system']['computed_at'] = computed_at
            df = pl.DataFrame([results['system']])
            upsert_parquet(df, get_path(STATE), SYSTEM_KEY_COLS)

        if results['signals']:
            for ind in results['signals']:
                ind['computed_at'] = computed_at
            df = pl.DataFrame(results['signals'])
            upsert_parquet(df, get_path(STATE), INDICATOR_DYNAMICS_KEY_COLS)

        if results['transfers']:
            for tr in results['transfers']:
                tr['computed_at'] = computed_at
            df = pl.DataFrame(results['transfers'])
            upsert_parquet(df, get_path(STATE), TRANSFERS_KEY_COLS)

        return 0

    if args.start:
        from datetime import datetime as dt
        start_date = dt.strptime(args.start, '%Y-%m-%d').date()

    if args.end:
        from datetime import datetime as dt
        end_date = dt.strptime(args.end, '%Y-%m-%d').date()

    # Run with or without parallelization
    if args.workers > 1:
        ensure_directory()
        logger.info(f"Parallel mode: {args.workers} workers")

        available_dates = get_available_dates()

        if start_date:
            available_dates = [d for d in available_dates if d >= start_date]
        if end_date:
            available_dates = [d for d in available_dates if d <= end_date]

        if args.stride > 1:
            available_dates = available_dates[::args.stride]

        if not available_dates:
            logger.error("No dates to process")
            return 1

        logger.info(f"Processing {len(available_dates)} dates with {args.workers} workers")

        chunks = divide_by_count(available_dates, args.workers)
        temp_paths = []
        assignments = []

        for i, chunk in enumerate(chunks):
            temp_path = generate_temp_path(i, prefix='state')
            temp_paths.append(temp_path)
            assignments.append(WorkerAssignment(
                worker_id=i,
                temp_path=temp_path,
                items=chunk,
                config={},
            ))

        results = run_workers(process_state_parallel, assignments, args.workers)

        # Report worker results
        successful = sum(1 for r in results if r.status == 'success')
        failed = sum(1 for r in results if r.status == 'error')
        logger.info(f"Workers complete: {successful} success, {failed} failed")

        # Merge results
        logger.info("Merging results...")
        successful_paths = [r.temp_path for r in results if r.status == 'success' and r.temp_path.exists()]
        totals = merge_state_results(successful_paths, verbose=not args.quiet)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"COMPLETE: {totals.get('system', 0)} system states, "
                    f"{totals.get('signal_dynamics', 0)} signal dynamics, "
                    f"{totals.get('transfers', 0)} transfers")
        logger.info("=" * 80)

    else:
        run_state_range(
            start_date,
            end_date,
            args.stride,
            verbose=not args.quiet
        )

    return 0


if __name__ == '__main__':
    exit(main())
