#!/usr/bin/env python3
"""
PRISM Geometry Entry Point
==========================

Computes structural relationships between signals using ALL geometry engines.

Engines (13 total):
    Class-based:
        - PCAEngine: Principal components, loadings, variance explained
        - MSTEngine: Minimum spanning tree, network metrics
        - ClusteringEngine: Hierarchical/K-means clustering
        - LOFEngine: Local outlier factor
        - DistanceEngine: Euclidean/Mahalanobis/Cosine distances
        - ConvexHullEngine: Geometric enclosure
        - CopulaEngine: Dependency structure
        - MutualInformationEngine: Non-linear dependence
        - BarycenterEngine: Centroid metrics

    Function-based:
        - compute_coupling_matrix: Laplace domain coupling
        - compute_divergence: Distribution divergence (KL, JS)
        - discover_modes: Behavioral mode discovery
        - compute_snapshot: Point-in-time structure

Usage:
    python -m prism.entry_points.geometry
    python -m prism.entry_points.geometry --force
    python -m prism.entry_points.geometry --window 100

Output:
    data/geometry.parquet - Structural metrics per (entity, window)
"""

import argparse
import logging
import sys
import time
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.polars_io import read_parquet, write_parquet_atomic

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE IMPORTS
# =============================================================================

def import_engines(config: Dict[str, Any]):
    """
    Import geometry engines based on config selection.

    Config structure:
        engines:
          geometry:
            pca: true
            mst: false
            ...
    """
    engines = {}
    engine_config = config.get('engines', {}).get('geometry', {})

    # If no config, enable all engines by default
    if not engine_config:
        engine_config = {k: True for k in [
            'pca', 'mst', 'clustering', 'lof', 'distance',
            'convex_hull', 'copula', 'mutual_information', 'barycenter',
        ]}

    # Class-based geometry engines
    if engine_config.get('pca', True):
        try:
            from prism.engines.geometry import PCAEngine
            engines['pca'] = PCAEngine()
        except ImportError:
            pass

    if engine_config.get('mst', True):
        try:
            from prism.engines.geometry import MSTEngine
            engines['mst'] = MSTEngine()
        except ImportError:
            pass

    if engine_config.get('clustering', True):
        try:
            from prism.engines.geometry import ClusteringEngine
            engines['clustering'] = ClusteringEngine()
        except ImportError:
            pass

    if engine_config.get('lof', True):
        try:
            from prism.engines.geometry import LOFEngine
            engines['lof'] = LOFEngine()
        except ImportError:
            pass

    if engine_config.get('distance', True):
        try:
            from prism.engines.geometry import DistanceEngine
            engines['distance'] = DistanceEngine()
        except ImportError:
            pass

    if engine_config.get('convex_hull', True):
        try:
            from prism.engines.geometry import ConvexHullEngine
            engines['convex_hull'] = ConvexHullEngine()
        except ImportError:
            pass

    if engine_config.get('copula', True):
        try:
            from prism.engines.geometry import CopulaEngine
            engines['copula'] = CopulaEngine()
        except ImportError:
            pass

    if engine_config.get('mutual_information', True):
        try:
            from prism.engines.geometry import MutualInformationEngine
            engines['mutual_information'] = MutualInformationEngine()
        except ImportError:
            pass

    if engine_config.get('barycenter', True):
        try:
            from prism.engines.geometry import BarycenterEngine
            engines['barycenter'] = BarycenterEngine()
        except ImportError:
            pass

    logger.info(f"  Loaded {len(engines)} geometry engines from config")
    return engines


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    'min_samples_geometry': 30,
    'min_signals': 2,
    'window_size': None,
    'stride': None,
}


def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config from data directory."""
    config_path = data_path / 'config.yaml'
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        # Extract relevant settings
        if 'min_samples_geometry' in user_config:
            config['min_samples'] = user_config['min_samples_geometry']
        elif 'min_samples' in user_config:
            config['min_samples'] = user_config['min_samples']

        # Store engine config
        config['engines'] = user_config.get('engines', {})

    return config


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_entity_matrix(
    observations: pl.DataFrame,
    entity_id: str,
    min_samples: int,
) -> Optional[pd.DataFrame]:
    """
    Prepare signal matrix for an entity.

    Returns:
        DataFrame with columns=signals, rows=timestamps, or None
    """
    entity_obs = observations.filter(pl.col('entity_id') == entity_id)

    if len(entity_obs) == 0:
        return None

    # Aggregate duplicates (same timestamp+signal) by taking mean
    entity_obs = entity_obs.group_by(['timestamp', 'signal_id']).agg(
        pl.col('value').mean()
    )

    # Pivot to wide format
    try:
        wide = entity_obs.pivot(
            values='value',
            index='timestamp',
            on='signal_id',
        ).sort('timestamp')
    except Exception:
        return None

    # Convert to pandas
    df = wide.to_pandas().set_index('timestamp')

    # Drop columns with too few samples
    valid_cols = [c for c in df.columns if df[c].notna().sum() >= min_samples]
    if len(valid_cols) < 2:
        return None

    df = df[valid_cols].dropna()

    if len(df) < min_samples:
        return None

    return df


# =============================================================================
# RESULT FLATTENING
# =============================================================================

def flatten_engine_result(result: Dict[str, Any], prefix: str) -> Dict[str, float]:
    """Flatten engine result dict to flat float dict."""
    flat = {}

    if result is None:
        return flat

    for k, v in result.items():
        if k in ('run_id', 'signals', 'signal_ids'):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            val = float(v)
            if np.isfinite(val):
                flat[f"{prefix}_{k}"] = val
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                val = float(v.item())
                if np.isfinite(val):
                    flat[f"{prefix}_{k}"] = val
            elif v.ndim == 1 and len(v) <= 10:
                # Store small arrays as individual values
                for i, val in enumerate(v):
                    if np.isfinite(val):
                        flat[f"{prefix}_{k}_{i}"] = float(val)
            elif v.ndim == 2:
                # For matrices, store summary stats
                flat[f"{prefix}_{k}_mean"] = float(np.nanmean(v))
                flat[f"{prefix}_{k}_std"] = float(np.nanstd(v))
                flat[f"{prefix}_{k}_min"] = float(np.nanmin(v))
                flat[f"{prefix}_{k}_max"] = float(np.nanmax(v))

    return flat


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_geometry(
    observations: pl.DataFrame,
    config: Dict[str, Any],
    engines: Dict[str, Any],
    force: bool = False,
) -> pl.DataFrame:
    """
    Compute geometry metrics for all entities.

    Args:
        observations: Raw observations
        config: Domain config
        engines: Dict of engine instances
        force: Recompute all

    Returns:
        DataFrame with structural metrics per entity/window
    """
    min_samples = config.get('min_samples_geometry', 30)
    min_signals = config.get('min_signals', 2)

    entities = observations.select('entity_id').unique()['entity_id'].to_list()
    n_entities = len(entities)
    n_engines = len(engines)

    logger.info(f"Processing {n_entities} entities with {n_engines} engines")

    results = []

    for i, entity_id in enumerate(entities):
        # Prepare entity data matrix
        df = prepare_entity_matrix(observations, entity_id, min_samples)

        if df is None or len(df.columns) < min_signals:
            continue

        run_id = str(uuid.uuid4())[:8]

        row_data = {
            'entity_id': entity_id,
            'n_signals': len(df.columns),
            'n_samples': len(df),
            'window_start': float(df.index.min()),
            'window_end': float(df.index.max()),
        }

        # Run all class-based engines
        for engine_name, engine in engines.items():
            try:
                result = engine.run(df, run_id)
                flat = flatten_engine_result(result, engine_name)
                row_data.update(flat)
            except Exception as e:
                logger.debug(f"  {engine_name} failed for {entity_id}: {e}")

        results.append(row_data)

        if (i + 1) % 5 == 0:
            logger.info(f"  Processed {i + 1}/{n_entities} entities")

    if not results:
        logger.warning("No entities with sufficient data")
        return pl.DataFrame()

    df = pl.DataFrame(results)
    logger.info(f"Geometry: {len(df)} rows, {len(df.columns)} columns")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Geometry - Structural relationship computation"
    )
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Geometry Engine")
    logger.info("=" * 60)

    ensure_directory()

    # Load observations
    obs_path = get_path(OBSERVATIONS)
    if not obs_path.exists():
        logger.error(f"observations.parquet not found")
        sys.exit(1)

    data_path = obs_path.parent
    output_path = data_path / 'geometry.parquet'

    if output_path.exists() and not args.force:
        logger.info("geometry.parquet exists, use --force to recompute")
        return 0

    # Load config
    config = load_config(data_path)

    # Load engines from config
    logger.info("Loading engines from config...")
    engines = import_engines(config)
    logger.info(f"Total engines: {len(engines)}")

    # Load data
    observations = read_parquet(obs_path)
    logger.info(f"Loaded {len(observations):,} observations")

    # Compute
    start = time.time()
    df = compute_geometry(observations, config, engines, args.force)
    elapsed = time.time() - start

    if len(df) > 0:
        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
