#!/usr/bin/env python3
"""
PRISM Fields Entry Point
========================

Computes field metrics: Laplace transforms, gradients, divergence, energy.

REQUIRES: observations.parquet

Orchestrates field engines - NO INLINE COMPUTATION.
All compute lives in prism/engines/laplace/.

Engines (8 total):
    - laplace_transform: s-domain representation
    - gradient: Rate of change (first derivative field)
    - laplacian: Second derivative field (curvature)
    - divergence: Source/sink detection
    - laplace_gradient: Gradient in Laplace domain
    - laplace_divergence: Divergence in Laplace domain
    - laplace_energy: Field energy density
    - decompose_by_scale: Multi-scale decomposition

Output:
    data/fields.parquet - One row per entity with field metrics

Usage:
    python -m prism.entry_points.fields
    python -m prism.entry_points.fields --force
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, Any

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

# Engine registry
ENGINES = {
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
# ORCHESTRATOR - Routes to engines, no compute
# =============================================================================

def run_field_engines(
    obs_df: pl.DataFrame,
    entity_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all field engines for one entity.

    Pure orchestration - all compute is in engines.
    """
    results = {'entity_id': entity_id}

    # Get entity observations
    entity_obs = obs_df.filter(pl.col('entity_id') == entity_id)
    signals = entity_obs['signal_id'].unique().to_list()

    # Determine index column
    index_col = 'index' if 'index' in entity_obs.columns else 'timestamp'

    for signal_id in signals:
        # Extract signal values
        sig_data = entity_obs.filter(pl.col('signal_id') == signal_id).sort(index_col)
        values = sig_data['value'].to_numpy()

        if len(values) < 10:
            continue

        # Clean data
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        prefix = signal_id

        # === Run ALL field engines ===

        # Laplace transform
        try:
            result = compute_laplace_for_series(values)
            _flatten_result(results, f"{prefix}_laplace", result)
        except Exception as e:
            logger.debug(f"laplace_transform ({signal_id}): {e}")

        # Gradient (first derivative field)
        try:
            grad = compute_gradient(values)
            if isinstance(grad, np.ndarray) and len(grad) > 0:
                results[f"{prefix}_gradient_mean"] = float(np.nanmean(grad))
                results[f"{prefix}_gradient_std"] = float(np.nanstd(grad))
                results[f"{prefix}_gradient_max"] = float(np.nanmax(np.abs(grad)))
                results[f"{prefix}_gradient_skew"] = float(_skewness(grad))
        except Exception as e:
            logger.debug(f"gradient ({signal_id}): {e}")

        # Laplacian (second derivative field)
        try:
            lap = compute_laplacian(values)
            if isinstance(lap, np.ndarray) and len(lap) > 0:
                results[f"{prefix}_laplacian_mean"] = float(np.nanmean(lap))
                results[f"{prefix}_laplacian_std"] = float(np.nanstd(lap))
                results[f"{prefix}_laplacian_energy"] = float(np.sum(lap**2))
        except Exception as e:
            logger.debug(f"laplacian ({signal_id}): {e}")

        # Divergence
        try:
            result = compute_divergence_for_signal(values)
            _flatten_result(results, f"{prefix}_divergence", result)
        except Exception as e:
            logger.debug(f"divergence ({signal_id}): {e}")

        # Laplace gradient
        try:
            result = laplace_gradient(values)
            _flatten_result(results, f"{prefix}_laplace_grad", result)
        except Exception as e:
            logger.debug(f"laplace_gradient ({signal_id}): {e}")

        # Laplace divergence
        try:
            result = laplace_divergence(values)
            _flatten_result(results, f"{prefix}_laplace_div", result)
        except Exception as e:
            logger.debug(f"laplace_divergence ({signal_id}): {e}")

        # Laplace energy
        try:
            result = laplace_energy(values)
            _flatten_result(results, f"{prefix}_field_energy", result)
        except Exception as e:
            logger.debug(f"laplace_energy ({signal_id}): {e}")

        # Multi-scale decomposition
        try:
            result = decompose_by_scale(values)
            if isinstance(result, dict):
                for scale, comp in result.items():
                    if isinstance(comp, np.ndarray) and len(comp) > 0:
                        results[f"{prefix}_scale_{scale}_energy"] = float(np.sum(comp**2))
                        results[f"{prefix}_scale_{scale}_std"] = float(np.nanstd(comp))
        except Exception as e:
            logger.debug(f"decompose_scale ({signal_id}): {e}")

    return results


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


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of array."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_fields(
    obs_df: pl.DataFrame,
    config: Dict[str, Any],
) -> pl.DataFrame:
    """
    Compute field metrics for all entities.

    Output: ONE ROW PER ENTITY
    """
    entities = obs_df['entity_id'].unique().to_list()
    n_entities = len(entities)

    logger.info(f"Computing fields for {n_entities} entities")
    logger.info(f"Engines: {list(ENGINES.keys())}")

    results = []

    for i, entity_id in enumerate(entities):
        row = run_field_engines(obs_df, entity_id, config)

        n_metrics = len(row) - 1  # Exclude entity_id
        results.append(row)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities ({n_metrics} metrics each)")

    if not results:
        logger.warning("No field metrics computed")
        return pl.DataFrame({'entity_id': []})

    df = pl.DataFrame(results)
    logger.info(f"Fields: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Fields - Laplace Field Analysis")
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Fields")
    logger.info("Laplace transforms, gradients, divergence, energy")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(FIELDS).parent
    output_path = get_path(FIELDS)

    if output_path.exists() and not args.force:
        logger.info("fields.parquet exists, use --force to recompute")
        return 0

    # Check dependency
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error("observations.parquet not found")
        logger.error("Run: python -m prism.entry_points.fetch")
        return 1

    # Load data and config
    obs_df = read_parquet(obs_path)
    config = load_config(data_path)

    n_entities = obs_df['entity_id'].n_unique()
    n_signals = obs_df['signal_id'].n_unique()
    logger.info(f"Observations: {len(obs_df)} rows, {n_entities} entities, {n_signals} signals")

    # Run field engines
    start = time.time()
    fields_df = compute_fields(obs_df, config)
    elapsed = time.time() - start

    logger.info(f"Complete: {elapsed:.1f}s")
    logger.info(f"Output: {len(fields_df)} rows, {len(fields_df.columns)} columns")

    # Save
    write_parquet_atomic(fields_df, output_path)
    logger.info(f"Saved: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
