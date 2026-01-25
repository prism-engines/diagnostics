#!/usr/bin/env python3
"""
PRISM Physics Entry Point
=========================

Answers: WHY is it moving?

REQUIRES: vector.parquet AND geometry.parquet AND dynamics.parquet

You cannot compute force without motion.
You cannot compute energy without velocity.

Physics explains WHY the system is moving the way it is.
It requires dynamics (motion) to compute forces.

Output:
    data/physics.parquet - ONE ROW PER ENTITY with:
        - hamiltonian_H: Total energy (T + V)
        - hamiltonian_T: Kinetic energy (from dynamics velocity)
        - hamiltonian_V: Potential energy (distance from equilibrium)
        - lagrangian_L: Action (T - V)
        - gibbs_free_energy: G = H - TS (spontaneity)
        - momentum_magnitude: Current momentum
        - force_mean: Average force magnitude
        - energy_conservation: How stable is H over time?

Usage:
    python -m prism.entry_points.physics
    python -m prism.entry_points.physics --force
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import polars as pl

from prism.core.dependencies import check_dependencies
from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY, DYNAMICS, PHYSICS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'min_samples_physics': 10,
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        if 'min_samples_physics' in user_config:
            config['min_samples_physics'] = user_config['min_samples_physics']

    return config


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dynamics_values(dynamics_df: pl.DataFrame, entity_id: str) -> Dict[str, float]:
    """Extract dynamics values for an entity."""
    entity_dyn = dynamics_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_dyn) == 0:
        return {}

    result = {}
    for col in ['hd_slope', 'hd_velocity_mean', 'hd_velocity_std',
                'hd_acceleration_mean', 'hd_acceleration_std',
                'hd_final_distance', 'hd_max_distance']:
        if col in entity_dyn.columns:
            val = entity_dyn[col][0]
            if val is not None and np.isfinite(val):
                result[col] = float(val)

    return result


def get_geometry_values(geometry_df: pl.DataFrame, entity_id: str) -> Dict[str, float]:
    """Extract geometry values for an entity."""
    entity_geom = geometry_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_geom) == 0:
        return {}

    result = {}
    for col in ['effective_dimensionality', 'cov_trace', 'dist_mean', 'centroid_dist_mean']:
        if col in entity_geom.columns:
            val = entity_geom[col][0]
            if val is not None and np.isfinite(val):
                result[col] = float(val)

    return result


def get_vector_entropy(vector_df: pl.DataFrame, entity_id: str) -> float:
    """Get entropy-related metric from vector layer."""
    entity_vec = vector_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_vec) == 0:
        return 0.0

    # Look for entropy columns
    entropy_cols = [c for c in entity_vec.columns if 'entropy' in c.lower()]

    if not entropy_cols:
        return 0.0

    # Average entropy across signals
    entropies = []
    for col in entropy_cols:
        vals = entity_vec[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            entropies.append(np.mean(vals))

    return float(np.mean(entropies)) if entropies else 0.0


def get_vector_volatility(vector_df: pl.DataFrame, entity_id: str) -> float:
    """Get volatility (temperature proxy) from vector layer."""
    entity_vec = vector_df.filter(pl.col('entity_id') == entity_id)

    if len(entity_vec) == 0:
        return 0.0

    # Look for volatility/variance columns
    vol_cols = [c for c in entity_vec.columns if 'vol' in c.lower() or 'std' in c.lower()]

    if not vol_cols:
        return 0.0

    # Average volatility across signals
    vols = []
    for col in vol_cols:
        vals = entity_vec[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            vols.append(np.mean(vals))

    return float(np.mean(vols)) if vols else 0.0


# =============================================================================
# HAMILTONIAN
# =============================================================================

def compute_hamiltonian(
    entity_id: str,
    dynamics_vals: Dict[str, float],
    geometry_vals: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute Hamiltonian (total energy) for entity.

    H = T + V

    Where:
    - T = kinetic energy = ½ v² (velocity from dynamics)
    - V = potential energy = ½ k x² (distance from equilibrium)
    """
    # Kinetic energy: T = ½ v²
    velocity = dynamics_vals.get('hd_velocity_mean', 0.0)
    T = 0.5 * velocity ** 2

    # Potential energy: V = ½ k x²
    # x = distance from equilibrium (approximated by centroid distance or final distance)
    displacement = dynamics_vals.get('hd_final_distance', 0.0)

    # Spring constant k ≈ 1 / variance (stiffer system = lower variance)
    cov_trace = geometry_vals.get('cov_trace', 1.0)
    k = 1.0 / (cov_trace + 1e-10) if cov_trace > 0 else 1.0
    k = min(k, 10.0)  # Cap to avoid numerical issues

    V = 0.5 * k * displacement ** 2

    # Total energy
    H = T + V

    # Energy partition
    T_fraction = T / H if H > 1e-10 else 0.0

    return {
        'hamiltonian_T': float(T),
        'hamiltonian_V': float(V),
        'hamiltonian_H': float(H),
        'hamiltonian_T_fraction': float(T_fraction),
        'hamiltonian_spring_k': float(k),
    }


# =============================================================================
# LAGRANGIAN
# =============================================================================

def compute_lagrangian(
    entity_id: str,
    dynamics_vals: Dict[str, float],
    geometry_vals: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute Lagrangian and action.

    L = T - V

    The Lagrangian describes the "action" of the system.
    Deviation from expected motion indicates external forcing.
    """
    # Get T and V from Hamiltonian calculation
    velocity = dynamics_vals.get('hd_velocity_mean', 0.0)
    T = 0.5 * velocity ** 2

    displacement = dynamics_vals.get('hd_final_distance', 0.0)
    cov_trace = geometry_vals.get('cov_trace', 1.0)
    k = 1.0 / (cov_trace + 1e-10) if cov_trace > 0 else 1.0
    k = min(k, 10.0)
    V = 0.5 * k * displacement ** 2

    L = T - V

    # Residual force (deviation from expected motion)
    # Expected: system should decelerate as it moves away from equilibrium
    # Actual acceleration
    actual_accel = dynamics_vals.get('hd_acceleration_mean', 0.0)

    # Expected acceleration (from potential gradient): F = -kx, a = -kx/m
    expected_accel = -k * displacement

    residual_force = abs(actual_accel - expected_accel)

    return {
        'lagrangian_L': float(L),
        'lagrangian_residual_force': float(residual_force),
    }


# =============================================================================
# GIBBS FREE ENERGY
# =============================================================================

def compute_gibbs_free_energy(
    entity_id: str,
    dynamics_vals: Dict[str, float],
    entropy: float,
    temperature: float,
    hamiltonian_H: float,
) -> Dict[str, Any]:
    """
    Compute Gibbs free energy.

    G = H - TS

    Where:
    - H = enthalpy (≈ Hamiltonian)
    - T = "temperature" (volatility/noise level from vector)
    - S = entropy (from vector layer)

    Negative dG → spontaneous transition (system naturally degrading)
    Positive dG → requires external work
    """
    # Gibbs free energy
    TS = temperature * entropy
    G = hamiltonian_H - TS

    # Rate of change proxy: hd_slope indicates direction
    hd_slope = dynamics_vals.get('hd_slope', 0.0)

    # If hd_slope > 0, system is moving away from baseline → spontaneous degradation
    spontaneous = hd_slope > 0

    return {
        'gibbs_free_energy': float(G),
        'gibbs_H': float(hamiltonian_H),
        'gibbs_TS': float(TS),
        'gibbs_temperature': float(temperature),
        'gibbs_entropy': float(entropy),
        'gibbs_spontaneous': 1.0 if spontaneous else 0.0,
    }


# =============================================================================
# MOMENTUM AND FORCES
# =============================================================================

def compute_momentum_analysis(
    entity_id: str,
    dynamics_vals: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute momentum and forces from dynamics.

    p = mv (momentum, mass = 1 in behavioral space)
    F = ma (force from acceleration)
    """
    # Momentum (mass = 1)
    velocity = dynamics_vals.get('hd_velocity_mean', 0.0)
    momentum = velocity  # m = 1

    # Force (F = ma, m = 1)
    acceleration = dynamics_vals.get('hd_acceleration_mean', 0.0)
    force = acceleration

    # Impulse proxy (change in momentum over trajectory)
    # Approximated by velocity_std * some factor
    velocity_std = dynamics_vals.get('hd_velocity_std', 0.0)
    impulse_proxy = velocity_std

    return {
        'momentum_magnitude': float(abs(momentum)),
        'momentum_direction': 1.0 if momentum > 0 else -1.0 if momentum < 0 else 0.0,
        'force_mean': float(abs(force)),
        'force_direction': 1.0 if force > 0 else -1.0 if force < 0 else 0.0,
        'impulse_proxy': float(impulse_proxy),
    }


# =============================================================================
# EQUILIBRIUM ANALYSIS
# =============================================================================

def analyze_equilibrium(
    entity_id: str,
    dynamics_vals: Dict[str, float],
    geometry_vals: Dict[str, float],
) -> Dict[str, Any]:
    """
    Analyze equilibrium state and stability.
    """
    # Distance from equilibrium (baseline)
    displacement = dynamics_vals.get('hd_final_distance', 0.0)
    max_displacement = dynamics_vals.get('hd_max_distance', 0.0)

    # Velocity at end
    velocity = dynamics_vals.get('hd_velocity_mean', 0.0)
    acceleration = dynamics_vals.get('hd_acceleration_mean', 0.0)

    # Stability indicators
    # Stable: returning to baseline (negative velocity when displaced)
    # Unstable: moving away (positive velocity when displaced)
    if abs(displacement) < 1e-6:
        stability = 'at_equilibrium'
    elif velocity * np.sign(displacement) < 0:
        stability = 'stable'  # Moving back toward baseline
    elif velocity * np.sign(displacement) > 0:
        stability = 'unstable'  # Moving away from baseline
    else:
        stability = 'unknown'

    stability_score = {
        'at_equilibrium': 1.0,
        'stable': 0.7,
        'unstable': 0.0,
        'unknown': 0.5,
    }.get(stability, 0.5)

    return {
        'equilibrium_distance': float(displacement),
        'equilibrium_max_distance': float(max_displacement),
        'equilibrium_stability_score': float(stability_score),
    }


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_entity_physics(
    entity_id: str,
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    dynamics_df: pl.DataFrame,
) -> Dict[str, Any]:
    """
    Compute all physics metrics for one entity.
    """
    # Get values from upstream layers
    dynamics_vals = get_dynamics_values(dynamics_df, entity_id)
    geometry_vals = get_geometry_values(geometry_df, entity_id)

    if not dynamics_vals:
        return {}

    # Get entropy and temperature from vector
    entropy = get_vector_entropy(vector_df, entity_id)
    temperature = get_vector_volatility(vector_df, entity_id)

    # Hamiltonian
    hamiltonian = compute_hamiltonian(entity_id, dynamics_vals, geometry_vals)

    # Lagrangian
    lagrangian = compute_lagrangian(entity_id, dynamics_vals, geometry_vals)

    # Gibbs free energy
    gibbs = compute_gibbs_free_energy(
        entity_id, dynamics_vals, entropy, temperature, hamiltonian['hamiltonian_H']
    )

    # Momentum
    momentum = compute_momentum_analysis(entity_id, dynamics_vals)

    # Equilibrium
    equilibrium = analyze_equilibrium(entity_id, dynamics_vals, geometry_vals)

    # Combine all
    result = {'entity_id': entity_id}
    result.update(hamiltonian)
    result.update(lagrangian)
    result.update(gibbs)
    result.update(momentum)
    result.update(equilibrium)

    # Include key dynamics values for reference
    result['hd_slope'] = dynamics_vals.get('hd_slope', 0.0)
    result['hd_velocity_mean'] = dynamics_vals.get('hd_velocity_mean', 0.0)
    result['hd_acceleration_mean'] = dynamics_vals.get('hd_acceleration_mean', 0.0)

    return result


def compute_physics(
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    dynamics_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute all physics metrics.

    REQUIRES all three upstream layers.

    Output: ONE ROW PER ENTITY
    """
    entities = dynamics_df.select('entity_id').unique()['entity_id'].to_list()
    n_entities = len(entities)

    logger.info(f"Computing physics for {n_entities} entities")

    results = []

    for i, entity_id in enumerate(entities):
        physics = compute_entity_physics(entity_id, vector_df, geometry_df, dynamics_df)

        if physics:
            results.append(physics)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities")

    if not results:
        logger.warning("No entities with sufficient data for physics")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Physics: {len(df)} rows (one per entity), {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Physics - WHY is it moving? (requires dynamics)"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Physics Engine")
    logger.info("WHY is it moving?")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(VECTOR).parent

    # Check dependencies (HARD FAIL if any missing)
    check_dependencies('physics', data_path)

    output_path = get_path(PHYSICS)

    if output_path.exists() and not args.force:
        logger.info("physics.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    # Load ALL required inputs
    vector_path = get_path(VECTOR)
    geometry_path = get_path(GEOMETRY)
    dynamics_path = get_path(DYNAMICS)

    vector_df = read_parquet(vector_path)
    geometry_df = read_parquet(geometry_path)
    dynamics_df = read_parquet(dynamics_path)

    logger.info(f"Loaded vector.parquet: {len(vector_df):,} rows, {len(vector_df.columns)} columns")
    logger.info(f"Loaded geometry.parquet: {len(geometry_df):,} rows, {len(geometry_df.columns)} columns")
    logger.info(f"Loaded dynamics.parquet: {len(dynamics_df):,} rows, {len(dynamics_df.columns)} columns")

    start = time.time()
    df = compute_physics(vector_df, geometry_df, dynamics_df, config)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
