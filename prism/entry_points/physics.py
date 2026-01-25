#!/usr/bin/env python3
"""
PRISM Physics Entry Point
=========================

Orchestrates energy and momentum calculations using physics engines.

REQUIRES: observations.parquet (for raw signal values)

Engines (7 total):
    Energy: kinetic_energy, potential_energy, hamiltonian, lagrangian
    Momentum: linear_momentum, angular_momentum
    Thermodynamics: gibbs_free_energy, work_energy

Usage:
    python -m prism.entry_points.physics
    python -m prism.entry_points.physics --force

Output:
    data/physics.parquet
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS, PHYSICS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS (direct imports for correct signatures)
# =============================================================================

from prism.engines.physics import (
    compute_kinetic,
    compute_potential,
    compute_hamilton,
    compute_lagrange,
    compute_momentum,
    compute_gibbs,
    compute_work_energy,
)


# =============================================================================
# SIGNAL TYPE DETECTION
# =============================================================================

# Keywords to identify signal types
VELOCITY_KEYWORDS = ['velocity', 'speed', 'vel', 'v_', 'flow', 'rate', 'rpm', 'omega']
POSITION_KEYWORDS = ['position', 'pos', 'displacement', 'x_', 'y_', 'z_', 'distance', 'height', 'depth', 'level']
TEMPERATURE_KEYWORDS = ['temp', 'temperature', 't_', 'celsius', 'kelvin', 'fahrenheit']
PRESSURE_KEYWORDS = ['pressure', 'pres', 'p_', 'psia', 'psig', 'bar', 'pascal', 'atm']
FORCE_KEYWORDS = ['force', 'f_', 'load', 'thrust', 'torque', 'newton']


def classify_signal(signal_id: str) -> str:
    """Classify a signal based on its name."""
    signal_lower = signal_id.lower()

    for kw in VELOCITY_KEYWORDS:
        if kw in signal_lower:
            return 'velocity'
    for kw in POSITION_KEYWORDS:
        if kw in signal_lower:
            return 'position'
    for kw in TEMPERATURE_KEYWORDS:
        if kw in signal_lower:
            return 'temperature'
    for kw in PRESSURE_KEYWORDS:
        if kw in signal_lower:
            return 'pressure'
    for kw in FORCE_KEYWORDS:
        if kw in signal_lower:
            return 'force'

    return 'unknown'


# =============================================================================
# CONFIG
# =============================================================================

def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config.json or config.yaml from data directory."""
    import json

    config_path = data_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    yaml_path = data_path / 'config.yaml'
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    return {}


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def extract_signal_values(
    obs_df: pl.DataFrame,
    entity_id: str,
    signal_id: str,
) -> np.ndarray:
    """Extract sorted values for a specific entity/signal."""
    filtered = obs_df.filter(
        (pl.col('entity_id') == entity_id) &
        (pl.col('signal_id') == signal_id)
    ).sort('index')

    return filtered['value'].to_numpy()


def run_physics_engines(
    obs_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Orchestrate physics engine execution.

    Reads raw observations and computes physics metrics per entity.
    """
    # Get constants from config
    constants = config.get('global_constants', {})
    mass = constants.get('mass', constants.get('mass_kg'))
    spring_constant = constants.get('spring_constant', constants.get('k'))

    # Get unique entities and signals
    entities = obs_df['entity_id'].unique().to_list()

    results = []

    for entity_id in entities:
        entity_obs = obs_df.filter(pl.col('entity_id') == entity_id)
        signals = entity_obs['signal_id'].unique().to_list()

        # Classify signals for this entity
        signal_types = {sig: classify_signal(sig) for sig in signals}

        # Find signals by type
        velocity_signals = [s for s, t in signal_types.items() if t == 'velocity']
        position_signals = [s for s, t in signal_types.items() if t == 'position']
        temperature_signals = [s for s, t in signal_types.items() if t == 'temperature']
        pressure_signals = [s for s, t in signal_types.items() if t == 'pressure']

        row = {'entity_id': entity_id}

        # === KINETIC ENERGY ===
        # Use velocity signals, or derive from position
        for sig in velocity_signals[:3]:  # Limit to first 3
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    result = compute_kinetic(values=values, mass=mass, mode='velocity')
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                            row[f"kinetic_{sig}_{k}"] = float(v)
                except Exception as e:
                    logger.warning(f"kinetic ({sig}): {e}")

        # If no velocity signals, try deriving from position
        if not velocity_signals and position_signals:
            for sig in position_signals[:3]:
                values = extract_signal_values(obs_df, entity_id, sig)
                if len(values) >= 2:
                    try:
                        result = compute_kinetic(values=values, mass=mass, mode='position')
                        for k, v in result.items():
                            if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                                row[f"kinetic_{sig}_{k}"] = float(v)
                    except Exception as e:
                        logger.warning(f"kinetic from position ({sig}): {e}")

        # === POTENTIAL ENERGY ===
        for sig in position_signals[:3]:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    result = compute_potential(values=values, spring_constant=spring_constant, mass=mass)
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                            row[f"potential_{sig}_{k}"] = float(v)
                except Exception as e:
                    logger.warning(f"potential ({sig}): {e}")

        # === HAMILTONIAN & LAGRANGIAN ===
        # Need both position and velocity
        if position_signals and velocity_signals:
            pos_sig = position_signals[0]
            vel_sig = velocity_signals[0]
            pos_values = extract_signal_values(obs_df, entity_id, pos_sig)
            vel_values = extract_signal_values(obs_df, entity_id, vel_sig)

            # Align lengths
            min_len = min(len(pos_values), len(vel_values))
            if min_len >= 2:
                pos_values = pos_values[:min_len]
                vel_values = vel_values[:min_len]

                try:
                    result = compute_hamilton(
                        position=pos_values,
                        velocity=vel_values,
                        mass=mass,
                        spring_constant=spring_constant
                    )
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                            row[f"hamiltonian_{k}"] = float(v)
                except Exception as e:
                    logger.warning(f"hamiltonian: {e}")

                try:
                    result = compute_lagrange(
                        position=pos_values,
                        velocity=vel_values,
                        mass=mass,
                        spring_constant=spring_constant
                    )
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                            row[f"lagrangian_{k}"] = float(v)
                except Exception as e:
                    logger.warning(f"lagrangian: {e}")

        # === MOMENTUM ===
        for sig in velocity_signals[:3]:
            values = extract_signal_values(obs_df, entity_id, sig)
            if len(values) >= 2:
                try:
                    result = compute_momentum(velocity=values, mass=mass)
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                            row[f"momentum_{sig}_{k}"] = float(v)
                except Exception as e:
                    logger.warning(f"momentum ({sig}): {e}")

        # === GIBBS FREE ENERGY ===
        if temperature_signals:
            temp_sig = temperature_signals[0]
            temp_values = extract_signal_values(obs_df, entity_id, temp_sig)

            pres_values = None
            if pressure_signals:
                pres_sig = pressure_signals[0]
                pres_values = extract_signal_values(obs_df, entity_id, pres_sig)
                # Align lengths
                min_len = min(len(temp_values), len(pres_values))
                temp_values = temp_values[:min_len]
                pres_values = pres_values[:min_len]

            if len(temp_values) >= 2:
                try:
                    result = compute_gibbs(temperature=temp_values, pressure=pres_values)
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                            row[f"gibbs_{k}"] = float(v)
                except Exception as e:
                    logger.warning(f"gibbs: {e}")

        # === FALLBACK: Use any signal as generic position/velocity ===
        # If no classified signals, treat first signal as position-like
        if len(row) == 1:  # Only entity_id
            logger.info(f"  {entity_id}: No classified signals, using generic physics")
            all_signals = signals[:5]  # First 5 signals

            for sig in all_signals:
                values = extract_signal_values(obs_df, entity_id, sig)
                if len(values) >= 2:
                    # Treat as position, derive velocity
                    try:
                        result = compute_kinetic(values=values, mass=mass, mode='position')
                        for k, v in result.items():
                            if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                                row[f"kinetic_{sig}_{k}"] = float(v)
                    except Exception as e:
                        logger.warning(f"generic kinetic ({sig}): {e}")

                    try:
                        result = compute_potential(values=values, spring_constant=spring_constant, mass=mass)
                        for k, v in result.items():
                            if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                                row[f"potential_{sig}_{k}"] = float(v)
                    except Exception as e:
                        logger.warning(f"generic potential ({sig}): {e}")

                    try:
                        result = compute_momentum(velocity=np.gradient(values), mass=mass)
                        for k, v in result.items():
                            if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                                row[f"momentum_{sig}_{k}"] = float(v)
                    except Exception as e:
                        logger.warning(f"generic momentum ({sig}): {e}")

        results.append(row)
        logger.info(f"  {entity_id}: {len(row) - 1} physics metrics")

    return pl.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Physics")
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Physics")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(PHYSICS).parent

    # Check dependency - need observations for raw signal values
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error("observations.parquet required - run: python -m prism.entry_points.fetch")
        return 1

    output_path = get_path(PHYSICS)
    if output_path.exists() and not args.force:
        logger.info("physics.parquet exists, use --force to recompute")
        return 0

    # Load observations and config
    obs_df = read_parquet(obs_path)
    config = load_config(data_path)

    n_entities = obs_df['entity_id'].n_unique()
    n_signals = obs_df['signal_id'].n_unique()
    logger.info(f"Observations: {len(obs_df)} rows, {n_entities} entities, {n_signals} signals")

    # Show signal classification
    signals = obs_df['signal_id'].unique().to_list()
    classifications = {sig: classify_signal(sig) for sig in signals}
    by_type = {}
    for sig, typ in classifications.items():
        by_type.setdefault(typ, []).append(sig)

    logger.info("Signal classification:")
    for typ, sigs in by_type.items():
        logger.info(f"  {typ}: {sigs[:5]}{'...' if len(sigs) > 5 else ''}")

    # Run physics engines
    start = time.time()
    physics_df = run_physics_engines(obs_df, config)

    logger.info(f"Complete: {time.time() - start:.1f}s")
    logger.info(f"Output: {len(physics_df)} rows, {len(physics_df.columns)} columns")

    if len(physics_df.columns) <= 1:
        logger.warning("No physics metrics computed! Check signal names and config.")
        logger.warning("Physics engines need: velocity/position/temperature/pressure signals")

    # Save
    write_parquet_atomic(physics_df, output_path)
    logger.info(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
