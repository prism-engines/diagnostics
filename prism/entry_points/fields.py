#!/usr/bin/env python3
"""
PRISM Fields Entry Point
========================

Field analysis for both 3D spatial data and time-series.

Orchestrates field engines - NO INLINE COMPUTATION.
All compute lives in prism/engines/fields/ and prism/engines/laplace/.

Two modes:
    1. Navier-Stokes (3D velocity fields): Real fluid dynamics
    2. Laplace Fields (time-series): Gradient, divergence, laplacian of signals

Engines:
    Navier-Stokes (requires 3D velocity data):
        - vorticity, strain_rate, Q_criterion, lambda2
        - turbulent_kinetic_energy, dissipation, enstrophy, helicity
        - reynolds_number, kolmogorov_scales, energy_spectrum

    Laplace Fields (works on observations.parquet):
        - laplace_transform: s-domain representation
        - gradient: Rate of change field
        - laplacian: Second derivative field
        - divergence: Source/sink detection
        - energy: Field energy density
        - decompose_by_scale: Multi-scale decomposition

Output:
    data/fields.parquet

Usage:
    # Laplace fields from observations (default)
    python -m prism.entry_points.fields

    # Navier-Stokes with 3D velocity data
    python -m prism.entry_points.fields --data /path/to/velocity/

    # Synthetic turbulence test
    python -m prism.entry_points.fields --synthetic --nx 64
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, FIELDS, OBSERVATIONS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS - All compute lives in engines
# =============================================================================

# Navier-Stokes (3D velocity fields)
from prism.engines.fields.navier_stokes import (
    VelocityField,
    analyze_velocity_field,
    compute_vorticity,
    compute_turbulent_kinetic_energy,
    compute_dissipation_rate,
    compute_enstrophy,
    compute_helicity,
    compute_Q_criterion,
    compute_reynolds_number,
    compute_kolmogorov_scales,
    compute_energy_spectrum,
    classify_flow_regime,
)

# Laplace Fields (time-series)
from prism.engines.laplace import (
    compute_laplace_for_series,
    compute_gradient,
    compute_laplacian,
    compute_divergence_for_signal,
    laplace_gradient,
    laplace_divergence,
    laplace_energy,
    decompose_by_scale,
)

# Engine registries
NAVIER_STOKES_ENGINES = {
    'vorticity': compute_vorticity,
    'tke': compute_turbulent_kinetic_energy,
    'dissipation': compute_dissipation_rate,
    'enstrophy': compute_enstrophy,
    'helicity': compute_helicity,
    'q_criterion': compute_Q_criterion,
    'reynolds': compute_reynolds_number,
    'kolmogorov': compute_kolmogorov_scales,
    'spectrum': compute_energy_spectrum,
}

LAPLACE_FIELD_ENGINES = {
    'laplace_transform': compute_laplace_for_series,
    'gradient': compute_gradient,
    'laplacian': compute_laplacian,
    'divergence': compute_divergence_for_signal,
    'laplace_gradient': laplace_gradient,
    'laplace_divergence': laplace_divergence,
    'laplace_energy': laplace_energy,
    'decompose_scale': decompose_by_scale,
}


# =============================================================================
# CONFIG
# =============================================================================

from prism.config.validator import ConfigurationError


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    json_path = data_path / 'config.json'
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    return {}


# =============================================================================
# NAVIER-STOKES MODE (3D Velocity Fields)
# =============================================================================

def load_velocity_data(data_dir: Path) -> Dict[str, np.ndarray]:
    """Load velocity field data from directory."""
    velocity = {}

    for component in ['u', 'v', 'w']:
        npy_path = data_dir / f'{component}.npy'
        npz_path = data_dir / f'{component}.npz'

        if npy_path.exists():
            velocity[component] = np.load(npy_path)
            logger.info(f"Loaded {npy_path.name}: shape {velocity[component].shape}")
        elif npz_path.exists():
            with np.load(npz_path) as npz:
                key = list(npz.keys())[0]
                velocity[component] = npz[key]
            logger.info(f"Loaded {npz_path.name}: shape {velocity[component].shape}")
        else:
            raise FileNotFoundError(
                f"Velocity component '{component}' not found.\n"
                f"Expected: {npy_path} or {npz_path}"
            )

    return velocity


def create_synthetic_turbulence(nx: int, ny: int, nz: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """Create synthetic turbulent velocity field for testing."""
    from prism.utils.fields_orchestrator import create_synthetic_turbulence as create_synth
    logger.info(f"Creating synthetic turbulence: {nx}x{ny}x{nz}")
    return create_synth(nx, ny, nz, Re_target=1000.0, seed=seed)


def run_navier_stokes(
    velocity_data: Dict[str, np.ndarray],
    config: Dict[str, Any],
    entity_id: str,
) -> Dict[str, Any]:
    """
    Run Navier-Stokes analysis on 3D velocity field.

    Pure orchestration - calls analyze_velocity_field engine.
    """
    fields_config = config.get('fields', {})

    # Required config
    dx = fields_config.get('dx')
    dy = fields_config.get('dy')
    dz = fields_config.get('dz')
    nu = fields_config.get('nu')

    if None in [dx, dy, dz, nu]:
        raise ConfigurationError(
            "Navier-Stokes requires explicit config:\n"
            "  fields:\n"
            "    dx: 0.001  # Grid spacing [m]\n"
            "    dy: 0.001\n"
            "    dz: 0.001\n"
            "    nu: 1.0e-6  # Kinematic viscosity [m^2/s]"
        )

    # Create velocity field object
    field = VelocityField(
        u=velocity_data['u'],
        v=velocity_data['v'],
        w=velocity_data['w'],
        dx=dx, dy=dy, dz=dz,
        nu=nu,
    )

    # Run full analysis (calls all Navier-Stokes engines)
    result = analyze_velocity_field(field)

    # Add entity_id
    result['entity_id'] = entity_id

    return result


# =============================================================================
# LAPLACE FIELDS MODE (Time-Series)
# =============================================================================

def run_laplace_fields(
    obs_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute Laplace field metrics for all signals.

    Pure orchestration - routes to Laplace field engines.
    """
    results = []

    # Get unique entity/signal combinations
    entities = obs_df['entity_id'].unique().to_list()

    logger.info(f"Computing Laplace fields for {len(entities)} entities")
    logger.info(f"Engines: {list(LAPLACE_FIELD_ENGINES.keys())}")

    # Get field config
    fields_config = config.get('fields', {})
    enabled_engines = fields_config.get('enabled', list(LAPLACE_FIELD_ENGINES.keys()))

    for entity_id in entities:
        entity_obs = obs_df.filter(pl.col('entity_id') == entity_id)
        signals = entity_obs['signal_id'].unique().to_list()

        row = {'entity_id': entity_id}

        for signal_id in signals[:10]:  # Limit signals per entity
            # Extract signal values
            index_col = 'index' if 'index' in entity_obs.columns else 'timestamp'
            sig_data = entity_obs.filter(pl.col('signal_id') == signal_id).sort(index_col)
            values = sig_data['value'].to_numpy()

            if len(values) < 10:
                continue

            # Clean data
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            # Run Laplace field engines
            prefix = f"{signal_id}"

            if 'laplace_transform' in enabled_engines:
                try:
                    result = compute_laplace_for_series(values)
                    _flatten_result(row, f"{prefix}_laplace", result)
                except Exception as e:
                    logger.debug(f"laplace_transform ({signal_id}): {e}")

            if 'gradient' in enabled_engines:
                try:
                    result = compute_gradient(values)
                    if isinstance(result, np.ndarray):
                        row[f"{prefix}_gradient_mean"] = float(np.nanmean(result))
                        row[f"{prefix}_gradient_std"] = float(np.nanstd(result))
                        row[f"{prefix}_gradient_max"] = float(np.nanmax(np.abs(result)))
                except Exception as e:
                    logger.debug(f"gradient ({signal_id}): {e}")

            if 'laplacian' in enabled_engines:
                try:
                    result = compute_laplacian(values)
                    if isinstance(result, np.ndarray):
                        row[f"{prefix}_laplacian_mean"] = float(np.nanmean(result))
                        row[f"{prefix}_laplacian_std"] = float(np.nanstd(result))
                        row[f"{prefix}_laplacian_energy"] = float(np.sum(result**2))
                except Exception as e:
                    logger.debug(f"laplacian ({signal_id}): {e}")

            if 'divergence' in enabled_engines:
                try:
                    result = compute_divergence_for_signal(values)
                    _flatten_result(row, f"{prefix}_divergence", result)
                except Exception as e:
                    logger.debug(f"divergence ({signal_id}): {e}")

            if 'laplace_energy' in enabled_engines:
                try:
                    result = laplace_energy(values)
                    _flatten_result(row, f"{prefix}_field_energy", result)
                except Exception as e:
                    logger.debug(f"laplace_energy ({signal_id}): {e}")

            if 'decompose_scale' in enabled_engines:
                try:
                    result = decompose_by_scale(values)
                    if isinstance(result, dict):
                        for scale, comp in result.items():
                            if isinstance(comp, np.ndarray):
                                row[f"{prefix}_scale_{scale}_energy"] = float(np.sum(comp**2))
                except Exception as e:
                    logger.debug(f"decompose_scale ({signal_id}): {e}")

        if len(row) > 1:  # More than just entity_id
            results.append(row)

    if not results:
        logger.warning("No Laplace field metrics computed")
        return pl.DataFrame({'entity_id': []})

    df = pl.DataFrame(results)
    logger.info(f"Laplace fields: {len(df)} rows, {len(df.columns)} columns")

    return df


def _flatten_result(row: Dict, prefix: str, result: Any):
    """Flatten engine result into row dict."""
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                if v is not None and np.isfinite(v):
                    row[f"{prefix}_{k}"] = float(v)
    elif isinstance(result, (int, float, np.integer, np.floating)):
        if result is not None and np.isfinite(result):
            row[prefix] = float(result)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Fields - Navier-Stokes & Laplace Field Analysis"
    )
    parser.add_argument('--data', '-d', type=Path,
                        help='Directory containing 3D velocity data (u.npy, v.npy, w.npy)')
    parser.add_argument('--synthetic', '-s', action='store_true',
                        help='Use synthetic turbulence data for testing')
    parser.add_argument('--nx', type=int, default=64,
                        help='Grid size for synthetic data (default: 64)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Fields Engine")
    logger.info("Navier-Stokes (3D) + Laplace Fields (time-series)")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(FIELDS).parent
    output_path = get_path(FIELDS)

    if output_path.exists() and not args.force:
        logger.info("fields.parquet exists, use --force to recompute")
        return 0

    config = load_config(data_path)

    # Determine mode
    if args.synthetic:
        # Synthetic Navier-Stokes
        logger.info("Mode: Synthetic Navier-Stokes")
        velocity_data = create_synthetic_turbulence(args.nx, args.nx, args.nx)
        config['fields'] = {
            'dx': 2 * np.pi / args.nx,
            'dy': 2 * np.pi / args.nx,
            'dz': 2 * np.pi / args.nx,
            'nu': 1e-4,
        }

        start = time.time()
        result = run_navier_stokes(velocity_data, config, f"synthetic_{args.nx}")
        df = pl.DataFrame([result])
        elapsed = time.time() - start

    elif args.data:
        # Real Navier-Stokes data
        logger.info(f"Mode: Navier-Stokes from {args.data}")
        if not args.data.exists():
            logger.error(f"Data directory not found: {args.data}")
            return 1

        velocity_data = load_velocity_data(args.data)

        start = time.time()
        result = run_navier_stokes(velocity_data, config, args.data.name)
        df = pl.DataFrame([result])
        elapsed = time.time() - start

    else:
        # Laplace Fields from observations
        logger.info("Mode: Laplace Fields (from observations.parquet)")
        obs_path = get_path(OBSERVATIONS)

        if not obs_path.exists():
            logger.error("observations.parquet not found")
            logger.error("Run: python -m prism.entry_points.fetch")
            logger.info("Or use --synthetic for test data, --data for 3D velocity")
            return 1

        obs_df = read_parquet(obs_path)
        logger.info(f"Loaded observations: {len(obs_df)} rows")

        start = time.time()
        df = run_laplace_fields(obs_df, config)
        elapsed = time.time() - start

    # Save results
    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df)} rows, {len(df.columns)} columns in {elapsed:.1f}s")
    else:
        logger.warning("No fields computed")

    return 0


if __name__ == '__main__':
    sys.exit(main())
