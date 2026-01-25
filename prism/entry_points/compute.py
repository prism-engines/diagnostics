#!/usr/bin/env python3
"""
PRISM Compute - Pure Numerical Calculation Engine
==================================================

Single entry point for all PRISM computations.
No labels, no classification, no orchestration - just numbers.

Usage:
    python -m prism.entry_points.compute                  # Full pipeline
    python -m prism.entry_points.compute --layer vector   # Specific layer
    python -m prism.entry_points.compute --layer geometry
    python -m prism.entry_points.compute --layer dynamics
    python -m prism.entry_points.compute --layer physics
    python -m prism.entry_points.compute --force          # Recompute all

Outputs:
    data.parquet     - observations + numeric characterization
    vector.parquet   - signal-level metrics (memory, information, frequency, etc.)
    geometry.parquet - pairwise relationships (correlation, distance, clustering)
    dynamics.parquet - state/transition metrics (granger, dtw, cointegration)
    physics.parquet  - energy/momentum metrics (hamiltonian, lagrangian, gibbs)

All outputs are pure numerical - no string classifications except identifiers.
"""

import argparse
import logging
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

import numpy as np
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.polars_io import read_parquet, write_parquet_atomic

warnings.filterwarnings('ignore')


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Domain metadata
    'domain': 'unknown',
    'description': '',

    # Sample thresholds
    'min_samples': 50,           # Minimum samples for vector/physics
    'min_samples_geometry': 30,  # Minimum samples for geometry pairs
    'min_samples_dynamics': 100, # Minimum samples for dynamics

    # Identifier columns
    'entity_col': 'entity_id',
    'signal_col': 'signal_id',
    'time_col': 'timestamp',
    'value_col': 'value',

    # Engine settings
    'engines': {
        'hurst': True,
        'entropy': True,
        'spectral': True,
        'garch': True,
        'rqa': True,
        'lyapunov': True,
        'physics': True,
    }
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """
    Load domain configuration from config.yaml in the data directory.

    Falls back to defaults if config doesn't exist.
    """
    config_path = data_path / 'config.yaml'

    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user config into defaults
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value

        logger.info(f"Loaded config from {config_path}")
    else:
        logger.info("No config.yaml found, using defaults")

    return config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# OUTPUT FILE CONSTANTS
# =============================================================================

DATA = "data"
VECTOR = "vector"
GEOMETRY = "geometry"
DYNAMICS = "dynamics"
PHYSICS = "physics"

PRISM_FILES = [DATA, VECTOR, GEOMETRY, DYNAMICS, PHYSICS]


def get_output_path(file: str) -> Path:
    """Get path to a PRISM output file."""
    return get_path(OBSERVATIONS).parent / f"{file}.parquet"


# =============================================================================
# LAYER 1: DATA (observations + characterization)
# =============================================================================

def compute_data_layer(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute data layer: observations + numeric characterization.

    Output columns:
        entity_id, signal_id, timestamp, value
        n_samples, mean, std, min, max, median, skewness, kurtosis
    """
    logger.info("Computing DATA layer...")
    start = time.time()

    # Group by entity/signal and compute statistics
    stats = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').count().alias('n_samples'),
        pl.col('value').mean().alias('mean'),
        pl.col('value').std().alias('std'),
        pl.col('value').min().alias('min'),
        pl.col('value').max().alias('max'),
        pl.col('value').median().alias('median'),
        pl.col('value').skew().alias('skewness'),
        pl.col('value').kurtosis().alias('kurtosis'),
    ])

    # Join stats back to observations
    result = observations.join(stats, on=['entity_id', 'signal_id'], how='left')

    elapsed = time.time() - start
    logger.info(f"DATA layer: {len(result):,} rows in {elapsed:.1f}s")

    return result


# =============================================================================
# LAYER 2: VECTOR (signal-level metrics)
# =============================================================================

def compute_vector_layer(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute vector layer: signal-level metrics.

    Engine categories:
        - Memory: hurst_rs, hurst_dfa, acf_decay, spectral_slope
        - Information: permutation_entropy, sample_entropy, entropy_rate
        - Frequency: spectral_*, wavelet_*
        - Volatility: garch_*, realized_vol, bipower_variation, hilbert_amplitude
        - Recurrence: rqa_*
        - Discontinuity: dirac_*, heaviside_*, structural_*
        - Typology: cusum_*, derivative_*, rolling_volatility_*
        - Momentum: runs_test_*
    """
    logger.info("Computing VECTOR layer...")
    start = time.time()

    # Import vector computation engines
    try:
        from prism.engines import (
            compute_hurst, compute_entropy, compute_wavelets,
            compute_spectral, compute_garch, compute_rqa,
            compute_lyapunov, compute_realized_vol,
            compute_hilbert_amplitude, compute_breaks,
            compute_heaviside, compute_dirac,
        )
    except ImportError as e:
        logger.warning(f"Some vector engines unavailable: {e}")

    results = []
    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
        pl.col('timestamp').alias('timestamps'),
    ])

    n_signals = len(signals)
    logger.info(f"Processing {n_signals} signals...")

    for i, row in enumerate(signals.iter_rows(named=True)):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)

        if len(values) < config.get('min_samples', 50):
            continue

        # Remove NaN
        values = values[~np.isnan(values)]
        if len(values) < config.get('min_samples', 50):
            continue

        metrics = {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_samples': len(values),
        }

        def safe_float(v):
            """Convert value to float, handling non-numeric types."""
            if v is None:
                return None
            if isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
                return val if np.isfinite(val) else None
            return None

        def add_metrics(result_dict, prefix):
            """Add metrics from engine result with type safety."""
            if result_dict is None:
                return
            for k, v in result_dict.items():
                safe_v = safe_float(v)
                if safe_v is not None:
                    metrics[f'{prefix}_{k}'] = safe_v

        # Memory engines
        try:
            hurst_result = compute_hurst(values)
            add_metrics(hurst_result, 'hurst')
        except Exception:
            pass

        # Information engines
        try:
            entropy_result = compute_entropy(values)
            add_metrics(entropy_result, 'entropy')
        except Exception:
            pass

        # Frequency engines
        try:
            wavelet_result = compute_wavelets(values)
            add_metrics(wavelet_result, 'wavelet')
        except Exception:
            pass

        try:
            spectral_result = compute_spectral(values)
            add_metrics(spectral_result, 'spectral')
        except Exception:
            pass

        # Volatility engines
        try:
            garch_result = compute_garch(values)
            add_metrics(garch_result, 'garch')
        except Exception:
            pass

        try:
            realized_result = compute_realized_vol(values)
            add_metrics(realized_result, 'realized')
        except Exception:
            pass

        try:
            hilbert_amp = compute_hilbert_amplitude(values)
            if len(hilbert_amp) > 0:
                metrics['hilbert_amplitude'] = float(np.mean(hilbert_amp))
        except Exception:
            pass

        # Recurrence engines
        try:
            rqa_result = compute_rqa(values)
            add_metrics(rqa_result, 'rqa')
        except Exception:
            pass

        # Lyapunov
        try:
            lyapunov_result = compute_lyapunov(values)
            add_metrics(lyapunov_result, 'lyapunov')
        except Exception:
            pass

        # Discontinuity engines
        try:
            break_result = compute_breaks(values)
            add_metrics(break_result, 'break')
        except Exception:
            pass

        try:
            heaviside_result = compute_heaviside(values)
            add_metrics(heaviside_result, 'heaviside')
        except Exception:
            pass

        try:
            dirac_result = compute_dirac(values)
            add_metrics(dirac_result, 'dirac')
        except Exception:
            pass

        results.append(metrics)

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{n_signals} signals...")

    if not results:
        logger.warning("No signals with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    elapsed = time.time() - start
    logger.info(f"VECTOR layer: {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return df


# =============================================================================
# LAYER 3: GEOMETRY (pairwise relationships)
# =============================================================================

def compute_geometry_layer(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute geometry layer: pairwise relationships.

    Engines:
        - bg_correlation, bg_distance, bg_clustering
        - bg_network, bg_granger, bg_lead_lag, bg_decoupling
        - pca, mst, lof, convex_hull, copula, mutual_information
    """
    logger.info("Computing GEOMETRY layer...")
    start = time.time()

    # Pivot observations to wide format for pairwise computations
    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
    ])

    # Get unique entities
    entities = observations.select('entity_id').unique()['entity_id'].to_list()

    results = []

    for entity_id in entities:
        entity_signals = signals.filter(pl.col('entity_id') == entity_id)
        signal_ids = entity_signals['signal_id'].to_list()

        if len(signal_ids) < 2:
            continue

        # Build data matrix
        signal_data = {}
        min_len = float('inf')

        for row in entity_signals.iter_rows(named=True):
            sid = row['signal_id']
            vals = np.array(row['values'], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                signal_data[sid] = vals
                min_len = min(min_len, len(vals))

        if len(signal_data) < 2 or min_len < config.get('min_samples_geometry', 30):
            continue

        # Truncate to same length
        for sid in signal_data:
            signal_data[sid] = signal_data[sid][:int(min_len)]

        # Compute pairwise metrics
        signal_ids = list(signal_data.keys())

        for i, sig_i in enumerate(signal_ids):
            for j, sig_j in enumerate(signal_ids):
                if i >= j:
                    continue

                x = signal_data[sig_i]
                y = signal_data[sig_j]

                pair_metrics = {
                    'entity_id': entity_id,
                    'signal_i': sig_i,
                    'signal_j': sig_j,
                }

                # Correlation
                try:
                    corr = np.corrcoef(x, y)[0, 1]
                    pair_metrics['correlation'] = float(corr) if not np.isnan(corr) else np.nan
                except Exception:
                    pass

                # Distance (Euclidean)
                try:
                    dist = np.sqrt(np.mean((x - y) ** 2))
                    pair_metrics['distance_euclidean'] = float(dist)
                except Exception:
                    pass

                # DTW distance
                try:
                    from scipy.spatial.distance import euclidean
                    from scipy.signal import correlate
                    cross_corr = correlate(x - np.mean(x), y - np.mean(y), mode='full')
                    max_corr = np.max(np.abs(cross_corr)) / (np.std(x) * np.std(y) * len(x))
                    pair_metrics['cross_correlation_max'] = float(max_corr)
                except Exception:
                    pass

                # Lead-lag (cross-correlation peak lag)
                try:
                    cross_corr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
                    lags = np.arange(-len(x) + 1, len(x))
                    peak_idx = np.argmax(np.abs(cross_corr))
                    pair_metrics['lead_lag'] = int(lags[peak_idx])
                except Exception:
                    pass

                # Mutual information estimate
                try:
                    from scipy.stats import spearmanr
                    rho, _ = spearmanr(x, y)
                    # Approximate MI using Spearman correlation
                    if not np.isnan(rho) and abs(rho) < 1:
                        mi_approx = -0.5 * np.log(1 - rho**2)
                        pair_metrics['mutual_information'] = float(mi_approx)
                except Exception:
                    pass

                # Granger causality (simple VAR-based)
                try:
                    from scipy import stats
                    # Simple Granger: does lagged x help predict y?
                    lag = 1
                    y_curr = y[lag:]
                    y_lag = y[:-lag]
                    x_lag = x[:-lag]

                    # Restricted model: y_t ~ y_{t-1}
                    slope_r, intercept_r, _, _, _ = stats.linregress(y_lag, y_curr)
                    resid_r = y_curr - (intercept_r + slope_r * y_lag)
                    ssr_r = np.sum(resid_r ** 2)

                    # Unrestricted model: y_t ~ y_{t-1} + x_{t-1}
                    X = np.column_stack([y_lag, x_lag])
                    beta, _, _, _ = np.linalg.lstsq(X, y_curr, rcond=None)
                    resid_u = y_curr - X @ beta
                    ssr_u = np.sum(resid_u ** 2)

                    # F-statistic
                    n = len(y_curr)
                    k = 1  # number of restrictions
                    f_stat = ((ssr_r - ssr_u) / k) / (ssr_u / (n - 2 - k))
                    pair_metrics['granger_f_xy'] = float(f_stat)
                except Exception:
                    pass

                results.append(pair_metrics)

    if not results:
        logger.warning("No entity pairs with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    elapsed = time.time() - start
    logger.info(f"GEOMETRY layer: {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return df


# =============================================================================
# LAYER 4: DYNAMICS (state/transition metrics)
# =============================================================================

def compute_dynamics_layer(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    vector_df: Optional[pl.DataFrame] = None,
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute dynamics layer: state/transition metrics.

    Engines:
        - granger, cross_correlation, cointegration, dtw, dmd
        - transfer_entropy, coupled_inertia
        - energy_dynamics, tension_dynamics, phase_detector
        - embedding, phase_space, lyapunov
    """
    logger.info("Computing DYNAMICS layer...")
    start = time.time()

    # Import dynamics engines
    try:
        from prism.engines.dynamics import compute_embedding, compute_phase_space, compute_lyapunov as compute_lyap_dyn
    except ImportError:
        logger.warning("Dynamics engines unavailable")
        compute_embedding = compute_phase_space = compute_lyap_dyn = None

    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
        pl.col('timestamp').alias('timestamps'),
    ])

    results = []

    for row in signals.iter_rows(named=True):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)
        timestamps = np.array(row['timestamps'], dtype=float)

        # Remove NaN
        mask = ~np.isnan(values)
        values = values[mask]
        timestamps = timestamps[mask]

        if len(values) < config.get('min_samples_dynamics', 100):
            continue

        metrics = {
            'entity_id': entity_id,
            'signal_id': signal_id,
        }

        # Embedding dimension estimation
        if compute_embedding is not None:
            try:
                embed_result = compute_embedding(values)
                metrics.update({f'embedding_{k}': v for k, v in embed_result.items()})
            except Exception:
                pass

        # Phase space reconstruction metrics
        if compute_phase_space is not None:
            try:
                phase_result = compute_phase_space(values)
                metrics.update({f'phase_space_{k}': v for k, v in phase_result.items()})
            except Exception:
                pass

        # Dynamics-specific Lyapunov
        if compute_lyap_dyn is not None:
            try:
                lyap_result = compute_lyap_dyn(values)
                metrics.update({f'dynamics_lyapunov_{k}': v for k, v in lyap_result.items()})
            except Exception:
                pass

        # Trend metrics
        try:
            # Linear trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            metrics['trend_slope'] = float(slope)
            metrics['trend_intercept'] = float(intercept)

            # Detrended variance
            detrended = values - (slope * x + intercept)
            metrics['detrended_variance'] = float(np.var(detrended))
        except Exception:
            pass

        # Rolling statistics dynamics
        try:
            window = min(30, len(values) // 4)
            if window >= 5:
                rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')
                rolling_std = np.array([np.std(values[i:i+window]) for i in range(len(values)-window+1)])

                metrics['rolling_mean_trend'] = float(np.polyfit(np.arange(len(rolling_mean)), rolling_mean, 1)[0])
                metrics['rolling_std_trend'] = float(np.polyfit(np.arange(len(rolling_std)), rolling_std, 1)[0])
        except Exception:
            pass

        # Acceleration (second derivative)
        try:
            velocity = np.diff(values)
            acceleration = np.diff(velocity)
            metrics['velocity_mean'] = float(np.mean(velocity))
            metrics['velocity_std'] = float(np.std(velocity))
            metrics['acceleration_mean'] = float(np.mean(acceleration))
            metrics['acceleration_std'] = float(np.std(acceleration))
        except Exception:
            pass

        results.append(metrics)

    if not results:
        logger.warning("No signals with sufficient data for dynamics")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    elapsed = time.time() - start
    logger.info(f"DYNAMICS layer: {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return df


# =============================================================================
# LAYER 5: PHYSICS (energy/momentum metrics)
# =============================================================================

def compute_physics_layer(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute physics layer: energy/momentum metrics.

    Engines:
        - hamiltonian, lagrangian, kinetic_energy, potential_energy
        - gibbs_free_energy, angular_momentum, momentum_flux, derivatives
    """
    logger.info("Computing PHYSICS layer...")
    start = time.time()

    # Import physics engines
    try:
        from prism.engines.physics import (
            compute_hamiltonian, compute_lagrangian,
            compute_kinetic, compute_potential,
            compute_gibbs, compute_angular_momentum,
            compute_momentum_flux, compute_derivatives,
        )
    except ImportError as e:
        logger.warning(f"Physics engines unavailable: {e}")
        return pl.DataFrame()

    signals = observations.group_by(['entity_id', 'signal_id']).agg([
        pl.col('value').alias('values'),
    ])

    results = []

    for row in signals.iter_rows(named=True):
        entity_id = row['entity_id']
        signal_id = row['signal_id']
        values = np.array(row['values'], dtype=float)

        # Remove NaN
        values = values[~np.isnan(values)]

        if len(values) < 20:
            continue

        metrics = {
            'entity_id': entity_id,
            'signal_id': signal_id,
        }

        # Hamiltonian (total energy)
        try:
            ham_result = compute_hamiltonian(values)
            metrics['hamiltonian_mean'] = ham_result.H_mean
            metrics['hamiltonian_std'] = ham_result.H_std
            metrics['hamiltonian_trend'] = ham_result.H_trend
            metrics['hamiltonian_conserved'] = 1.0 if ham_result.conserved else 0.0
            metrics['kinetic_potential_ratio'] = ham_result.T_V_ratio
        except Exception:
            pass

        # Lagrangian (action)
        try:
            lag_result = compute_lagrangian(values)
            metrics['lagrangian_mean'] = lag_result.L_mean
            metrics['lagrangian_std'] = lag_result.L_std
            metrics['action'] = lag_result.action
        except Exception:
            pass

        # Kinetic energy
        try:
            kin_result = compute_kinetic(values)
            metrics['kinetic_mean'] = kin_result.T_mean
            metrics['kinetic_std'] = kin_result.T_std
            metrics['kinetic_max'] = kin_result.T_max
        except Exception:
            pass

        # Potential energy
        try:
            pot_result = compute_potential(values)
            metrics['potential_mean'] = pot_result.V_mean
            metrics['potential_std'] = pot_result.V_std
            metrics['potential_max'] = pot_result.V_max
        except Exception:
            pass

        # Gibbs free energy
        try:
            gibbs_result = compute_gibbs(values)
            metrics['gibbs_mean'] = gibbs_result.G_mean
            metrics['gibbs_std'] = gibbs_result.G_std
            metrics['gibbs_spontaneous'] = 1.0 if gibbs_result.is_spontaneous else 0.0
        except Exception:
            pass

        # Angular momentum
        try:
            ang_result = compute_angular_momentum(values)
            metrics['angular_momentum_mean'] = ang_result.L_mean
            metrics['angular_momentum_std'] = ang_result.L_std
            metrics['angular_momentum_conserved'] = 1.0 if ang_result.conserved else 0.0
        except Exception:
            pass

        # Momentum flux
        try:
            flux_result = compute_momentum_flux(values)
            metrics['momentum_flux_mean'] = flux_result.flux_mean
            metrics['momentum_flux_std'] = flux_result.flux_std
        except Exception:
            pass

        # Derivatives
        try:
            deriv_result = compute_derivatives(values)
            metrics.update({f'deriv_{k}': v for k, v in deriv_result.items()})
        except Exception:
            pass

        results.append(metrics)

    if not results:
        logger.warning("No signals with sufficient data for physics")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    elapsed = time.time() - start
    logger.info(f"PHYSICS layer: {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    layer: Optional[str] = None,
    force: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    Run the PRISM computation pipeline.

    Args:
        layer: Specific layer to compute (None = all)
        force: Recompute even if output exists

    Returns:
        Dict mapping layer names to DataFrames
    """
    logger.info("=" * 60)
    logger.info("PRISM Compute Pipeline")
    logger.info("=" * 60)

    ensure_directory()

    # Load observations
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error(f"observations.parquet not found at {obs_path}")
        logger.error("Run: python -m prism.entry_points.fetch")
        sys.exit(1)

    data_path = obs_path.parent

    # Load domain configuration
    config = load_config(data_path)
    logger.info(f"Domain: {config.get('domain', 'unknown')}")
    logger.info(f"Min samples: {config.get('min_samples', 50)}")

    observations = read_parquet(obs_path)
    logger.info(f"Loaded {len(observations):,} observations")

    results = {}
    layers_to_run = [layer] if layer else [DATA, VECTOR, GEOMETRY, DYNAMICS, PHYSICS]

    for layer_name in layers_to_run:
        output_path = get_output_path(layer_name)

        if output_path.exists() and not force:
            logger.info(f"{layer_name}.parquet exists, skipping (use --force to recompute)")
            results[layer_name] = read_parquet(output_path)
            continue

        if layer_name == DATA:
            df = compute_data_layer(observations, config, force)
        elif layer_name == VECTOR:
            df = compute_vector_layer(observations, config, force)
        elif layer_name == GEOMETRY:
            df = compute_geometry_layer(observations, config, force)
        elif layer_name == DYNAMICS:
            vector_df = results.get(VECTOR)
            df = compute_dynamics_layer(observations, config, vector_df, force)
        elif layer_name == PHYSICS:
            df = compute_physics_layer(observations, config, force)
        else:
            logger.error(f"Unknown layer: {layer_name}")
            continue

        if len(df) > 0:
            write_parquet_atomic(df, output_path)
            logger.info(f"Wrote {output_path}")

        results[layer_name] = df

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    for name, df in results.items():
        if df is not None and len(df) > 0:
            logger.info(f"  {name}.parquet: {len(df):,} rows, {len(df.columns)} columns")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Compute - Pure Numerical Calculation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs:
    data.parquet     - observations + numeric characterization
    vector.parquet   - signal-level metrics (memory, frequency, volatility)
    geometry.parquet - pairwise relationships (correlation, distance)
    dynamics.parquet - state/transition metrics (granger, dtw)
    physics.parquet  - energy/momentum metrics (hamiltonian, lagrangian)

Examples:
    python -m prism.entry_points.compute                  # Full pipeline
    python -m prism.entry_points.compute --layer vector   # Vector only
    python -m prism.entry_points.compute --force          # Recompute all
        """
    )

    parser.add_argument(
        '--layer',
        choices=[DATA, VECTOR, GEOMETRY, DYNAMICS, PHYSICS],
        help='Compute specific layer only'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Recompute even if output exists'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = run_pipeline(layer=args.layer, force=args.force)

    return 0


if __name__ == '__main__':
    sys.exit(main())
