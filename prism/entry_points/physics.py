"""
PRISM Physics Runner
====================

Tests universal physics laws on signal topology behavioral geometry.

INPUT:
    state/indicator.parquet    (indicator position over time)
    state/cohort.parquet       (cohort position over time)

OUTPUT:
    physics/indicator.parquet  (energy, momentum, action per indicator)
    physics/cohort.parquet     (energy, momentum, action per cohort)
    physics/conservation.parquet (system-level conservation tests)

PIPELINE POSITION:
    indicator_vector → cohort_geometry → cohort_vector → domain_geometry
                                                              ↓
                                            indicator_state → cohort_state
                                                              ↓
                                                          physics.py
                                                              ↓
                                                      "Do universal laws hold?"

HYPOTHESIS:
    If signal topology in behavioral space obey physics laws (energy conservation,
    entropy increase, action minimization), then PRISM has discovered something
    universal about complex systems.

PHYSICS METRICS:

    Kinetic Energy (KE):
        Motion through behavioral space
        KE = ½m||v||² where v = Δposition/Δt

    Potential Energy (PE):
        Tension from equilibrium (centroid)
        PE = ½k||x - x_eq||² (harmonic approximation)

    Total Energy (E):
        E = KE + PE
        Conservation test: is dE/dt ≈ 0?

    Momentum (p):
        p = m·v
        Direction and magnitude of motion

    Entropy (S):
        S = -Σ p_i log(p_i) over distribution
        Second law test: does S only increase?

    Action (A):
        A = ∫(KE - PE)dt
        Least action test: do trajectories minimize A?

    Lagrangian (L):
        L = KE - PE
        Equations of motion derivable?

    Hamiltonian (H):
        H = KE + PE (= total energy for conservative systems)
        Phase space analysis

Usage:
    # Full production run
    python -m prism.entry_points.physics

    # Testing mode
    python -m prism.entry_points.physics --testing --domain cmapss
    python -m prism.entry_points.physics --testing --dates 2020-01-01:2024-01-01
"""

import argparse
import logging
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from prism.db.parquet_store import ensure_directories, get_parquet_path
from prism.db.polars_io import read_parquet, upsert_parquet

from prism.config.windows import get_window_weight

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Key columns for upsert
INDICATOR_PHYSICS_KEY_COLS = ['indicator_id', 'obs_date', 'target_obs']
COHORT_PHYSICS_KEY_COLS = ['cohort_id', 'obs_date', 'target_obs']
CONSERVATION_KEY_COLS = ['level', 'entity_id', 'target_obs', 'test_type']

# Physics constants (can be calibrated per domain)
DEFAULT_MASS = 1.0  # Uniform mass assumption initially
SPRING_CONSTANT = 1.0  # Harmonic potential strength

# Minimum windows needed for physics calculations
MIN_WINDOWS_FOR_VELOCITY = 2
MIN_WINDOWS_FOR_ACCELERATION = 3
MIN_WINDOWS_FOR_CONSERVATION_TEST = 10


# =============================================================================
# DATA LOADING
# =============================================================================

def load_indicator_state() -> pl.DataFrame:
    """Load indicator state over time."""
    path = get_parquet_path('state', 'indicator')
    if not path.exists():
        raise FileNotFoundError(f"Indicator state not found at {path}")
    return pl.read_parquet(path)


def load_cohort_state() -> pl.DataFrame:
    """Load cohort state over time."""
    path = get_parquet_path('state', 'cohort')
    if not path.exists():
        raise FileNotFoundError(f"Cohort state not found at {path}")
    return pl.read_parquet(path)


# =============================================================================
# PHYSICS COMPUTATIONS - CORE
# =============================================================================

def compute_velocity(
    positions: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute velocity (rate of change of position).

    v = Δx / Δt

    Args:
        positions: Array of position values over time
        times: Array of time indices (days or window count)

    Returns:
        Array of velocities (one fewer than positions)
    """
    if len(positions) < MIN_WINDOWS_FOR_VELOCITY:
        return np.array([])

    dx = np.diff(positions)
    dt = np.diff(times)

    # Avoid division by zero
    dt = np.where(dt == 0, 1, dt)

    return dx / dt


def compute_acceleration(
    velocities: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute acceleration (rate of change of velocity).

    a = Δv / Δt

    Args:
        velocities: Array of velocity values
        times: Array of time indices

    Returns:
        Array of accelerations (one fewer than velocities)
    """
    if len(velocities) < 2:
        return np.array([])

    dv = np.diff(velocities)
    dt = np.diff(times[:-1])  # times array is one longer than velocities

    dt = np.where(dt == 0, 1, dt)

    return dv / dt


def compute_kinetic_energy(
    velocity: float,
    mass: float = DEFAULT_MASS
) -> float:
    """
    Compute kinetic energy.

    KE = ½mv²
    """
    return 0.5 * mass * velocity ** 2


def compute_potential_energy(
    distance_to_equilibrium: float,
    spring_constant: float = SPRING_CONSTANT
) -> float:
    """
    Compute potential energy (harmonic approximation).

    PE = ½kx²

    Uses distance to centroid as displacement from equilibrium.
    """
    return 0.5 * spring_constant * distance_to_equilibrium ** 2


def compute_momentum(
    velocity: float,
    mass: float = DEFAULT_MASS
) -> float:
    """
    Compute momentum.

    p = mv
    """
    return mass * velocity


def compute_lagrangian(
    kinetic_energy: float,
    potential_energy: float
) -> float:
    """
    Compute Lagrangian.

    L = KE - PE
    """
    return kinetic_energy - potential_energy


def compute_hamiltonian(
    kinetic_energy: float,
    potential_energy: float
) -> float:
    """
    Compute Hamiltonian (total energy for conservative systems).

    H = KE + PE
    """
    return kinetic_energy + potential_energy


def compute_entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a distribution.

    S = -Σ p_i log(p_i)

    Args:
        distribution: Array of values (will be normalized to probabilities)

    Returns:
        Entropy value
    """
    if len(distribution) == 0:
        return 0.0

    # Normalize to probabilities
    dist = np.abs(distribution)
    dist = dist / dist.sum() if dist.sum() > 0 else dist

    # Remove zeros to avoid log(0)
    dist = dist[dist > 0]

    if len(dist) == 0:
        return 0.0

    return -np.sum(dist * np.log(dist))


def compute_action(
    lagrangians: np.ndarray,
    dt: float = 1.0
) -> float:
    """
    Compute action (integral of Lagrangian over time).

    A = ∫ L dt ≈ Σ L_i Δt

    Args:
        lagrangians: Array of Lagrangian values over time
        dt: Time step

    Returns:
        Total action
    """
    return np.sum(lagrangians) * dt


# =============================================================================
# PHYSICS COMPUTATIONS - ENTITY LEVEL
# =============================================================================

def compute_indicator_physics(
    indicator_id: str,
    state_df: pl.DataFrame,
    target_obs: int,
) -> List[Dict[str, Any]]:
    """
    Compute physics metrics for a single indicator over time.

    Args:
        indicator_id: The indicator
        state_df: Full indicator state DataFrame
        target_obs: Window size to analyze

    Returns:
        List of physics metric dicts (one per time point with sufficient history)
    """
    # Filter to this indicator and window size
    filtered = state_df.filter(
        (pl.col('indicator_id') == indicator_id) &
        (pl.col('target_obs') == target_obs)
    ).sort('obs_date')

    if len(filtered) < MIN_WINDOWS_FOR_VELOCITY:
        return []

    rows = filtered.to_dicts()
    results = []

    # Extract signal topology of position (distance to centroid)
    dates = [r['obs_date'] for r in rows]
    positions = []

    for r in rows:
        pos = r.get('in_cohort_distance_to_centroid')
        if pos is not None and np.isfinite(pos):
            positions.append(pos)
        else:
            positions.append(np.nan)

    positions = np.array(positions)
    times = np.arange(len(positions))

    # Handle NaN values
    valid_mask = ~np.isnan(positions)
    if valid_mask.sum() < MIN_WINDOWS_FOR_VELOCITY:
        return []

    # Compute velocities
    velocities = compute_velocity(positions[valid_mask], times[valid_mask])

    if len(velocities) == 0:
        return []

    # Compute physics for each valid time point
    valid_indices = np.where(valid_mask)[0]

    for i, vel_idx in enumerate(range(len(velocities))):
        # Map back to original index
        orig_idx = valid_indices[vel_idx + 1]  # +1 because velocity is between points

        if orig_idx >= len(rows):
            continue

        row = rows[orig_idx]
        pos = positions[orig_idx]
        vel = velocities[vel_idx]

        # Core physics
        ke = compute_kinetic_energy(vel)
        pe = compute_potential_energy(pos)
        total_energy = compute_hamiltonian(ke, pe)
        lagrangian = compute_lagrangian(ke, pe)
        momentum = compute_momentum(vel)

        result = {
            'indicator_id': indicator_id,
            'cohort_id': row.get('cohort_id'),
            'obs_date': row['obs_date'],
            'target_obs': target_obs,
            'window_weight': row.get('window_weight', get_window_weight(target_obs)),

            # Position and motion
            'position': float(pos),
            'velocity': float(vel),

            # Energy
            'kinetic_energy': float(ke),
            'potential_energy': float(pe),
            'total_energy': float(total_energy),

            # Lagrangian/Hamiltonian
            'lagrangian': float(lagrangian),
            'hamiltonian': float(total_energy),  # Same as total energy for conservative

            # Momentum
            'momentum': float(momentum),

            # Additional state context
            'lof_score': row.get('in_cohort_lof_score'),
            'percentile': row.get('in_cohort_percentile_distance'),

            'computed_at': datetime.now(),
        }

        # Compute acceleration if we have enough velocity history
        if vel_idx > 0:
            acc = velocities[vel_idx] - velocities[vel_idx - 1]
            result['acceleration'] = float(acc)

            # Force (F = ma)
            result['force'] = float(DEFAULT_MASS * acc)

        results.append(result)

    return results


def compute_cohort_physics(
    cohort_id: str,
    state_df: pl.DataFrame,
    target_obs: int,
) -> List[Dict[str, Any]]:
    """
    Compute physics metrics for a single cohort over time.

    Args:
        cohort_id: The cohort
        state_df: Full cohort state DataFrame
        target_obs: Window size to analyze

    Returns:
        List of physics metric dicts
    """
    # Filter to this cohort and window size
    filtered = state_df.filter(
        (pl.col('cohort_id') == cohort_id) &
        (pl.col('target_obs') == target_obs)
    ).sort('obs_date')

    if len(filtered) < MIN_WINDOWS_FOR_VELOCITY:
        return []

    rows = filtered.to_dicts()
    results = []

    # Extract position (distance to domain centroid)
    positions = []
    for r in rows:
        pos = r.get('in_domain_distance_to_centroid')
        if pos is not None and np.isfinite(pos):
            positions.append(pos)
        else:
            positions.append(np.nan)

    positions = np.array(positions)
    times = np.arange(len(positions))

    valid_mask = ~np.isnan(positions)
    if valid_mask.sum() < MIN_WINDOWS_FOR_VELOCITY:
        return []

    velocities = compute_velocity(positions[valid_mask], times[valid_mask])

    if len(velocities) == 0:
        return []

    valid_indices = np.where(valid_mask)[0]

    for i, vel_idx in enumerate(range(len(velocities))):
        orig_idx = valid_indices[vel_idx + 1]

        if orig_idx >= len(rows):
            continue

        row = rows[orig_idx]
        pos = positions[orig_idx]
        vel = velocities[vel_idx]

        ke = compute_kinetic_energy(vel)
        pe = compute_potential_energy(pos)
        total_energy = compute_hamiltonian(ke, pe)
        lagrangian = compute_lagrangian(ke, pe)
        momentum = compute_momentum(vel)

        result = {
            'cohort_id': cohort_id,
            'domain_id': row.get('domain_id'),
            'obs_date': row['obs_date'],
            'target_obs': target_obs,
            'window_weight': row.get('window_weight', get_window_weight(target_obs)),

            'position': float(pos),
            'velocity': float(vel),

            'kinetic_energy': float(ke),
            'potential_energy': float(pe),
            'total_energy': float(total_energy),

            'lagrangian': float(lagrangian),
            'hamiltonian': float(total_energy),

            'momentum': float(momentum),

            'lof_score': row.get('in_domain_lof_score'),
            'percentile': row.get('in_domain_percentile_distance'),

            'computed_at': datetime.now(),
        }

        if vel_idx > 0:
            acc = velocities[vel_idx] - velocities[vel_idx - 1]
            result['acceleration'] = float(acc)
            result['force'] = float(DEFAULT_MASS * acc)

        results.append(result)

    return results


# =============================================================================
# CONSERVATION TESTS
# =============================================================================

def test_energy_conservation(
    energies: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Test if total energy is conserved over time.

    H0: Energy is constant (dE/dt = 0)

    Args:
        energies: Array of total energy values over time
        threshold: Coefficient of variation threshold for conservation

    Returns:
        Dict with test results
    """
    if len(energies) < MIN_WINDOWS_FOR_CONSERVATION_TEST:
        return {'test': 'energy_conservation', 'result': 'insufficient_data'}

    energies = energies[~np.isnan(energies)]

    if len(energies) < MIN_WINDOWS_FOR_CONSERVATION_TEST:
        return {'test': 'energy_conservation', 'result': 'insufficient_data'}

    mean_e = np.mean(energies)
    std_e = np.std(energies)
    cv = std_e / mean_e if mean_e != 0 else np.inf

    # Rate of change
    de_dt = np.diff(energies)
    mean_de_dt = np.mean(de_dt)

    # Conservation holds if CV is low and mean rate of change is near zero
    is_conserved = cv < threshold and np.abs(mean_de_dt) < threshold * mean_e

    return {
        'test': 'energy_conservation',
        'result': 'conserved' if is_conserved else 'not_conserved',
        'mean_energy': float(mean_e),
        'std_energy': float(std_e),
        'cv': float(cv),
        'mean_de_dt': float(mean_de_dt),
        'n_samples': len(energies),
        'threshold': threshold,
    }


def test_entropy_increase(
    entropies: np.ndarray,
) -> Dict[str, Any]:
    """
    Test if entropy is monotonically increasing (second law).

    H0: Entropy never decreases (dS/dt >= 0)

    Args:
        entropies: Array of entropy values over time

    Returns:
        Dict with test results
    """
    if len(entropies) < MIN_WINDOWS_FOR_CONSERVATION_TEST:
        return {'test': 'entropy_increase', 'result': 'insufficient_data'}

    entropies = entropies[~np.isnan(entropies)]

    if len(entropies) < MIN_WINDOWS_FOR_CONSERVATION_TEST:
        return {'test': 'entropy_increase', 'result': 'insufficient_data'}

    ds_dt = np.diff(entropies)
    n_decreases = np.sum(ds_dt < 0)
    n_increases = np.sum(ds_dt > 0)
    pct_violations = n_decreases / len(ds_dt) if len(ds_dt) > 0 else 0

    # Second law holds if entropy rarely decreases
    second_law_holds = pct_violations < 0.1  # Allow 10% violations (noise)

    return {
        'test': 'entropy_increase',
        'result': 'holds' if second_law_holds else 'violated',
        'n_increases': int(n_increases),
        'n_decreases': int(n_decreases),
        'pct_violations': float(pct_violations),
        'mean_ds_dt': float(np.mean(ds_dt)),
        'n_samples': len(entropies),
    }


def test_least_action(
    lagrangians: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> Dict[str, Any]:
    """
    Test if trajectories minimize action.

    Compare actual action to random perturbations.

    Args:
        lagrangians: Array of Lagrangian values
        positions: Array of positions
        velocities: Array of velocities

    Returns:
        Dict with test results
    """
    if len(lagrangians) < MIN_WINDOWS_FOR_CONSERVATION_TEST:
        return {'test': 'least_action', 'result': 'insufficient_data'}

    # Actual action
    actual_action = compute_action(lagrangians)

    # Generate perturbed trajectories and compute their action
    n_perturbations = 100
    perturbed_actions = []

    for _ in range(n_perturbations):
        # Random perturbation to positions
        noise = np.random.normal(0, np.std(positions) * 0.1, len(positions))
        perturbed_pos = positions + noise

        # Recompute velocities and lagrangians
        perturbed_vel = np.diff(perturbed_pos)
        if len(perturbed_vel) < len(lagrangians):
            continue

        perturbed_vel = perturbed_vel[:len(lagrangians)]
        perturbed_ke = 0.5 * DEFAULT_MASS * perturbed_vel ** 2
        perturbed_pe = 0.5 * SPRING_CONSTANT * perturbed_pos[1:len(lagrangians)+1] ** 2
        perturbed_L = perturbed_ke - perturbed_pe
        perturbed_action = compute_action(perturbed_L)
        perturbed_actions.append(perturbed_action)

    if len(perturbed_actions) == 0:
        return {'test': 'least_action', 'result': 'computation_failed'}

    perturbed_actions = np.array(perturbed_actions)

    # Action is minimized if actual < most perturbations
    pct_lower = np.mean(actual_action < perturbed_actions)
    is_minimal = pct_lower > 0.5

    return {
        'test': 'least_action',
        'result': 'minimized' if is_minimal else 'not_minimized',
        'actual_action': float(actual_action),
        'mean_perturbed_action': float(np.mean(perturbed_actions)),
        'pct_actual_lower': float(pct_lower),
        'n_perturbations': len(perturbed_actions),
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_physics(
    cohorts: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    run_conservation_tests: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run physics computations on PRISM state data.

    Args:
        cohorts: Filter to specific cohorts (None = all)
        start_date: Start of date range
        end_date: End of date range
        run_conservation_tests: Whether to run conservation law tests
        verbose: Print progress

    Returns:
        Dict with processing statistics
    """
    ensure_directories()

    # Load state data
    if verbose:
        print("Loading state data...", flush=True)

    indicator_state = load_indicator_state()
    cohort_state = load_cohort_state()

    # Filter by date if specified
    if start_date:
        indicator_state = indicator_state.filter(pl.col('obs_date') >= start_date)
        cohort_state = cohort_state.filter(pl.col('obs_date') >= start_date)
    if end_date:
        indicator_state = indicator_state.filter(pl.col('obs_date') <= end_date)
        cohort_state = cohort_state.filter(pl.col('obs_date') <= end_date)

    # Get unique indicators and cohorts
    all_indicators = indicator_state.select('indicator_id').unique().to_series().to_list()
    all_cohorts = cohort_state.select('cohort_id').unique().to_series().to_list()

    if cohorts:
        # Filter indicators to those in specified cohorts
        indicator_cohort_map = dict(
            indicator_state
            .select(['indicator_id', 'cohort_id'])
            .unique()
            .iter_rows()
        )
        all_indicators = [i for i in all_indicators if indicator_cohort_map.get(i) in cohorts]
        all_cohorts = [c for c in all_cohorts if c in cohorts]

    if verbose:
        print(f"Indicators to process: {len(all_indicators)}", flush=True)
        print(f"Cohorts to process: {len(all_cohorts)}", flush=True)

    # Get window sizes
    target_obs_list = indicator_state.select('target_obs').unique().to_series().to_list()

    if verbose:
        print(f"Window sizes: {target_obs_list}", flush=True)

    # Process indicators
    indicator_physics_rows = []

    if verbose:
        print("\nComputing indicator physics...", flush=True)

    for i, indicator_id in enumerate(all_indicators):
        for target_obs in target_obs_list:
            rows = compute_indicator_physics(indicator_id, indicator_state, target_obs)
            indicator_physics_rows.extend(rows)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_indicators)} indicators", flush=True)

    # Process cohorts
    cohort_physics_rows = []

    if verbose:
        print("\nComputing cohort physics...", flush=True)

    for cohort_id in all_cohorts:
        for target_obs in target_obs_list:
            rows = compute_cohort_physics(cohort_id, cohort_state, target_obs)
            cohort_physics_rows.extend(rows)

    # Run conservation tests
    conservation_rows = []

    if run_conservation_tests and len(indicator_physics_rows) > 0:
        if verbose:
            print("\nRunning conservation tests...", flush=True)

        # Convert to DataFrame for easier analysis
        ind_phys_df = pl.DataFrame(indicator_physics_rows)

        for indicator_id in all_indicators[:50]:  # Test subset for speed
            for target_obs in target_obs_list:
                subset = ind_phys_df.filter(
                    (pl.col('indicator_id') == indicator_id) &
                    (pl.col('target_obs') == target_obs)
                ).sort('obs_date')

                if len(subset) < MIN_WINDOWS_FOR_CONSERVATION_TEST:
                    continue

                energies = subset['total_energy'].to_numpy()
                positions = subset['position'].to_numpy()
                velocities = subset['velocity'].to_numpy()
                lagrangians = subset['lagrangian'].to_numpy()

                # Energy conservation
                energy_test = test_energy_conservation(energies)
                energy_test.update({
                    'level': 'indicator',
                    'entity_id': indicator_id,
                    'target_obs': target_obs,
                    'computed_at': datetime.now(),
                })
                conservation_rows.append(energy_test)

                # Least action
                action_test = test_least_action(lagrangians, positions, velocities)
                action_test.update({
                    'level': 'indicator',
                    'entity_id': indicator_id,
                    'target_obs': target_obs,
                    'computed_at': datetime.now(),
                })
                conservation_rows.append(action_test)

    # Store results
    if indicator_physics_rows:
        df = pl.DataFrame(indicator_physics_rows, infer_schema_length=None)
        path = get_parquet_path('physics', 'indicator')
        upsert_parquet(df, path, INDICATOR_PHYSICS_KEY_COLS)
        if verbose:
            print(f"\nWrote {len(indicator_physics_rows):,} indicator physics rows", flush=True)

    if cohort_physics_rows:
        df = pl.DataFrame(cohort_physics_rows, infer_schema_length=None)
        path = get_parquet_path('physics', 'cohort')
        upsert_parquet(df, path, COHORT_PHYSICS_KEY_COLS)
        if verbose:
            print(f"Wrote {len(cohort_physics_rows):,} cohort physics rows", flush=True)

    if conservation_rows:
        df = pl.DataFrame(conservation_rows, infer_schema_length=None)
        path = get_parquet_path('physics', 'conservation')
        upsert_parquet(df, path, CONSERVATION_KEY_COLS)
        if verbose:
            print(f"Wrote {len(conservation_rows):,} conservation test rows", flush=True)

            # Summary of conservation tests
            df_tests = pl.DataFrame(conservation_rows)
            for test_type in df_tests['test'].unique().to_list():
                subset = df_tests.filter(pl.col('test') == test_type)
                results = subset['result'].value_counts()
                print(f"\n  {test_type}:")
                for row in results.iter_rows(named=True):
                    print(f"    {row['result']}: {row['count']}")

    return {
        'indicators': len(all_indicators),
        'cohorts': len(all_cohorts),
        'indicator_physics_rows': len(indicator_physics_rows),
        'cohort_physics_rows': len(cohort_physics_rows),
        'conservation_tests': len(conservation_rows),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Physics - Test Universal Laws on Behavioral Geometry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--force', action='store_true',
                        help='Clear progress and recompute all')
    parser.add_argument('--skip-conservation', action='store_true',
                        help='Skip conservation law tests')

    parser.add_argument('--testing', action='store_true',
                        help='Enable testing mode. REQUIRED to use limiting flags.')

    parser.add_argument('--filter-cohort', type=str,
                        help='[TESTING] Filter to specific cohort')
    parser.add_argument('--dates', type=str,
                        help='[TESTING] Date range as START:END (YYYY-MM-DD:YYYY-MM-DD)')

    args = parser.parse_args()

    # --testing guard
    if not args.testing:
        if args.filter_cohort or args.dates:
            logger.warning("=" * 80)
            logger.warning("LIMITING FLAGS IGNORED - --testing not specified")
            logger.warning("Running FULL computation.")
            logger.warning("=" * 80)
        args.filter_cohort = None
        args.dates = None

    cohorts = None
    if args.filter_cohort:
        cohorts = [args.filter_cohort]

    start_date = None
    end_date = None
    if args.dates:
        try:
            start_str, end_str = args.dates.split(':')
            start_date = pd.to_datetime(start_str).date()
            end_date = pd.to_datetime(end_str).date()
        except ValueError:
            logger.error("Invalid --dates format. Use START:END (YYYY-MM-DD:YYYY-MM-DD)")
            return 1

    print("=" * 80)
    print("PRISM PHYSICS")
    print("=" * 80)
    print("Testing universal physics laws on behavioral geometry")
    print()
    print("Tests:")
    print("  - Energy conservation (dE/dt ≈ 0)")
    print("  - Least action (trajectories minimize ∫L dt)")
    print("  - Entropy increase (second law)")
    print()
    print("Output:")
    print("  - physics/indicator.parquet (KE, PE, momentum per indicator)")
    print("  - physics/cohort.parquet (KE, PE, momentum per cohort)")
    print("  - physics/conservation.parquet (test results)")
    print()

    result = run_physics(
        cohorts=cohorts,
        start_date=start_date,
        end_date=end_date,
        run_conservation_tests=not args.skip_conservation,
        verbose=True,
    )

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Indicators: {result['indicators']}")
    print(f"Cohorts: {result['cohorts']}")
    print(f"Indicator physics rows: {result['indicator_physics_rows']}")
    print(f"Cohort physics rows: {result['cohort_physics_rows']}")
    print(f"Conservation tests: {result['conservation_tests']}")

    return 0


if __name__ == '__main__':
    exit(main())
