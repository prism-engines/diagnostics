#!/usr/bin/env python3
"""
PRISM Fetch Module
==================

Fetch raw observations to observations.parquet.

Output Schema:
    entity_id   | String   | Engine, bearing, unit identifier
    signal_id   | String   | Sensor name
    timestamp   | Float64  | Time (cycles, seconds, etc.)
    value       | Float64  | Raw measurement

Usage:
    python -m prism.db.fetch --cmapss
    python -m prism.db.fetch --tep
    python -m prism.db.fetch --femto
    python -m prism.db.fetch fetchers/yaml/custom.yaml

Fetchers are loaded dynamically from repo_root/fetchers/{source}_fetcher.py.
Results are written to data/{domain}/observations.parquet
"""

import argparse
import importlib.util
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl
import yaml

from prism.db.parquet_store import get_path, ensure_directory, OBSERVATIONS
from prism.db.polars_io import upsert_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT YAML PATHS (shortcuts)
# =============================================================================

DEFAULT_YAMLS = {
    "cmapss": "fetchers/yaml/cmapss.yaml",
    "tep": "fetchers/yaml/tep.yaml",
    "femto": "fetchers/yaml/femto.yaml",
    "hydraulic": "fetchers/yaml/hydraulic.yaml",
    "cwru": "fetchers/yaml/cwru.yaml",
}


def find_repo_root() -> Path:
    """Find repository root by looking for fetchers/ directory."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "fetchers").exists():
            return parent
    return current


def resolve_yaml_path(yaml_arg: Optional[str], source_shortcut: Optional[str]) -> Path:
    """Resolve YAML path from argument or shortcut."""
    repo_root = find_repo_root()

    if yaml_arg:
        path = Path(yaml_arg)
        if not path.is_absolute():
            path = repo_root / path
        return path

    if source_shortcut:
        if source_shortcut not in DEFAULT_YAMLS:
            available = ", ".join(sorted(DEFAULT_YAMLS.keys()))
            raise ValueError(f"Unknown source: {source_shortcut}. Available: {available}")
        return repo_root / DEFAULT_YAMLS[source_shortcut]

    raise ValueError("Must specify YAML file or source shortcut (--cmapss, --tep, etc.)")


def load_fetcher(source: str) -> Callable:
    """
    Dynamically load a fetcher module and return its fetch function.

    Fetchers are expected at: repo_root/fetchers/{source}_fetcher.py
    """
    repo_root = find_repo_root()
    fetcher_path = repo_root / "fetchers" / f"{source}_fetcher.py"

    if not fetcher_path.exists():
        raise FileNotFoundError(f"Fetcher not found: {fetcher_path}")

    spec = importlib.util.spec_from_file_location(f"{source}_fetcher", fetcher_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{source}_fetcher"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "fetch"):
        raise AttributeError(f"Fetcher {source} must have a 'fetch(config)' function")

    return module.fetch


def fetch_to_parquet(
    yaml_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    signals: Optional[List[str]] = None,
    entities: Optional[List[str]] = None,
) -> int:
    """
    Fetch data using config and write to observations.parquet.

    Args:
        yaml_path: Path to YAML config file
        start_date: Override start date
        end_date: Override end date
        signals: Override signal list
        entities: Override entity list

    Returns:
        Number of observations written
    """
    # Load config
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    source = config.get("source")
    if not source:
        raise ValueError("Config must specify 'source' field")

    # Apply overrides
    if start_date:
        config["start_date"] = start_date
    if end_date:
        config["end_date"] = end_date
    if signals:
        config["signals"] = signals
    if entities:
        config["entities"] = entities

    logger.info(f"Fetching from {source}...")
    logger.info(f"Config: {yaml_path}")

    # Load fetcher and fetch data
    fetch_func = load_fetcher(source)
    observations = fetch_func(config)

    if not observations:
        logger.warning("No observations returned")
        return 0

    logger.info(f"Fetched {len(observations):,} observations")

    # Convert to Polars DataFrame
    df = pl.DataFrame(observations)

    # Normalize column names to new schema
    column_mappings = {
        # Old name -> New name
        "unit_id": "entity_id",
        "engine_id": "entity_id",
        "bearing_id": "entity_id",
        "run_id": "entity_id",
        "obs_date": "timestamp",
        "time": "timestamp",
        "cycle": "timestamp",
        "t": "timestamp",
        "sensor_id": "signal_id",
        "indicator_id": "signal_id",
    }

    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename({old_name: new_name})

    # If no entity_id, try to infer from signal_id or use default
    if "entity_id" not in df.columns:
        # Check if we can extract entity from config
        default_entity = config.get("entity_id", config.get("domain", "unit_1"))
        df = df.with_columns(pl.lit(default_entity).alias("entity_id"))
        logger.info(f"No entity_id found, using default: {default_entity}")

    # Ensure required columns
    required_cols = ["entity_id", "signal_id", "timestamp", "value"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {df.columns}")

    # Select and cast columns to final schema
    df = df.select([
        pl.col("entity_id").cast(pl.Utf8),
        pl.col("signal_id").cast(pl.Utf8),
        pl.col("timestamp").cast(pl.Float64),
        pl.col("value").cast(pl.Float64),
    ])

    # Get domain from config
    domain = config.get("domain")
    if domain:
        os.environ["PRISM_DOMAIN"] = domain

    # Ensure directory exists
    ensure_directory(domain)

    # Write to observations.parquet (upsert on entity_id + signal_id + timestamp)
    target_path = get_path(OBSERVATIONS, domain=domain)
    total_rows = upsert_parquet(
        df,
        target_path,
        key_cols=["entity_id", "signal_id", "timestamp"]
    )

    logger.info(f"Wrote {total_rows:,} rows to {target_path}")

    return total_rows


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Fetch - Raw observations to observations.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output: data/{domain}/observations.parquet

Schema:
  entity_id   | String   | The thing that fails (engine, bearing, unit)
  signal_id   | String   | The measurement (sensor_1, temp, vibration)
  timestamp   | Float64  | Time (cycles, seconds, etc.)
  value       | Float64  | Raw measurement value

Examples:
  python -m prism.db.fetch --cmapss          # Fetch NASA turbofan data
  python -m prism.db.fetch --tep             # Fetch Tennessee Eastman data
  python -m prism.db.fetch --femto           # Fetch FEMTO bearing data
  python -m prism.db.fetch custom.yaml       # Fetch from custom config
"""
    )

    parser.add_argument("yaml_file", nargs="?", help="Path to YAML config file")

    # Source shortcuts
    parser.add_argument("--cmapss", action="store_true", help="Fetch NASA C-MAPSS turbofan data")
    parser.add_argument("--tep", action="store_true", help="Fetch Tennessee Eastman process data")
    parser.add_argument("--femto", action="store_true", help="Fetch FEMTO bearing data")
    parser.add_argument("--hydraulic", action="store_true", help="Fetch UCI hydraulic data")
    parser.add_argument("--cwru", action="store_true", help="Fetch CWRU bearing data")

    # Options
    parser.add_argument("--start-date", type=str, help="Override start date")
    parser.add_argument("--end-date", type=str, help="Override end date")
    parser.add_argument("--signals", type=str, help="Comma-separated signal list")
    parser.add_argument("--entities", type=str, help="Comma-separated entity list")

    args = parser.parse_args()

    # Determine source shortcut
    source_shortcut = None
    for source in DEFAULT_YAMLS.keys():
        if getattr(args, source.replace("-", "_"), False):
            source_shortcut = source
            break

    try:
        yaml_path = resolve_yaml_path(args.yaml_file, source_shortcut)
    except ValueError as e:
        parser.error(str(e))

    # Parse lists if provided
    signals = [s.strip() for s in args.signals.split(",")] if args.signals else None
    entities = [e.strip() for e in args.entities.split(",")] if args.entities else None

    # Run fetch
    try:
        count = fetch_to_parquet(
            yaml_path=yaml_path,
            start_date=args.start_date,
            end_date=args.end_date,
            signals=signals,
            entities=entities,
        )
        print(f"\n✓ Fetched {count:,} observations to observations.parquet")
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
