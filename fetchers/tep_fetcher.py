#!/usr/bin/env python3
"""
Tennessee Eastman Process (TEP) Fetcher

Downloads and processes the Tennessee Eastman Process dataset, a benchmark
chemical process simulation used for fault detection and process monitoring.

Dataset Info:
- 52 process variables (41 measurements + 11 manipulated variables)
- Multiple fault scenarios
- Continuous signal topology with random faults injected

Data Sources:
- GitHub: anasouzac/new_tep_datasets (CSV format)
- Harvard Dataverse: doi:10.7910/DVN/6C3JR1 (RData format)

Usage:
    from fetchers.tep_fetcher import fetch

    config = {
        "dataset": "1year",  # "1year" or "3years"
    }
    observations = fetch(config)

Returns:
    list[dict] with keys: indicator_id, observed_at, value, source
"""

import os
import tempfile
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# GitHub raw file URLs
TEP_URLS = {
    "1year": "https://raw.githubusercontent.com/anasouzac/new_tep_datasets/main/python_data_1year.csv",
    "3years": "https://raw.githubusercontent.com/anasouzac/new_tep_datasets/main/python_data_3years.csv",
}

SOURCE = "tep"

# Variable names based on TEP documentation
# 41 process measurements (XMEAS) + 11 manipulated variables (XMV)
XMEAS_NAMES = [f"XMEAS{i:02d}" for i in range(1, 42)]  # XMEAS01-XMEAS41
XMV_NAMES = [f"XMV{i:02d}" for i in range(1, 12)]       # XMV01-XMV11 (12th is often constant)

# XMEAS descriptions
XMEAS_DESC = {
    'XMEAS01': 'A feed (stream 1)',
    'XMEAS02': 'D feed (stream 2)',
    'XMEAS03': 'E feed (stream 3)',
    'XMEAS04': 'A and C feed (stream 4)',
    'XMEAS05': 'Recycle flow (stream 8)',
    'XMEAS06': 'Reactor feed rate (stream 6)',
    'XMEAS07': 'Reactor pressure',
    'XMEAS08': 'Reactor level',
    'XMEAS09': 'Reactor temperature',
    'XMEAS10': 'Purge rate (stream 9)',
    'XMEAS11': 'Product separator temperature',
    'XMEAS12': 'Product separator level',
    'XMEAS13': 'Product separator pressure',
    'XMEAS14': 'Product separator underflow',
    'XMEAS15': 'Stripper level',
    'XMEAS16': 'Stripper pressure',
    'XMEAS17': 'Stripper underflow',
    'XMEAS18': 'Stripper temperature',
    'XMEAS19': 'Stripper steam flow',
    'XMEAS20': 'Compressor work',
    'XMEAS21': 'Reactor cooling water outlet temp',
    'XMEAS22': 'Separator cooling water outlet temp',
    'XMEAS23': 'Reactor feed A (mol%)',
    'XMEAS24': 'Reactor feed B (mol%)',
    'XMEAS25': 'Reactor feed C (mol%)',
    'XMEAS26': 'Reactor feed D (mol%)',
    'XMEAS27': 'Reactor feed E (mol%)',
    'XMEAS28': 'Reactor feed F (mol%)',
    'XMEAS29': 'Purge A (mol%)',
    'XMEAS30': 'Purge B (mol%)',
    'XMEAS31': 'Purge C (mol%)',
    'XMEAS32': 'Purge D (mol%)',
    'XMEAS33': 'Purge E (mol%)',
    'XMEAS34': 'Purge F (mol%)',
    'XMEAS35': 'Purge G (mol%)',
    'XMEAS36': 'Purge H (mol%)',
    'XMEAS37': 'Product D (mol%)',
    'XMEAS38': 'Product E (mol%)',
    'XMEAS39': 'Product F (mol%)',
    'XMEAS40': 'Product G (mol%)',
    'XMEAS41': 'Product H (mol%)',
}

XMV_DESC = {
    'XMV01': 'D feed flow valve',
    'XMV02': 'E feed flow valve',
    'XMV03': 'A feed flow valve',
    'XMV04': 'A and C feed flow valve',
    'XMV05': 'Compressor recycle valve',
    'XMV06': 'Purge valve',
    'XMV07': 'Separator liquid flow valve',
    'XMV08': 'Stripper liquid product valve',
    'XMV09': 'Stripper steam valve',
    'XMV10': 'Reactor cooling water valve',
    'XMV11': 'Condenser cooling water valve',
}


def download_tep(
    dataset: str = "1year",
    cache_dir: Optional[Path] = None
) -> Path:
    """
    Download TEP data from GitHub.

    The GitHub repo uses Git LFS, so we need to clone it properly.
    Falls back to checking common locations if already cloned.

    Args:
        dataset: "1year" or "3years"
        cache_dir: Directory to cache downloaded data

    Returns:
        Path to downloaded CSV file
    """
    if dataset not in ["1year", "3years"]:
        raise ValueError(f"Unknown dataset: {dataset}. Use '1year' or '3years'")

    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "tep_data"

    # Check common locations for already-cloned repo
    possible_paths = [
        cache_dir / "new_tep_datasets" / f"python_data_{dataset}.csv",
        cache_dir / f"python_data_{dataset}.csv",
        Path("/tmp/tep_data/new_tep_datasets") / f"python_data_{dataset}.csv",
    ]

    for path in possible_paths:
        if path.exists() and path.stat().st_size > 1000:  # Real file, not LFS pointer
            print(f"  Using cached data at {path}")
            return path

    # Need to clone the repo with Git LFS
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_dir / "new_tep_datasets"

    if not repo_dir.exists():
        import subprocess
        print(f"  Cloning TEP repo with Git LFS...")
        print(f"    This may take a few minutes for large files...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/anasouzac/new_tep_datasets.git",
             str(repo_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")

    csv_path = repo_dir / f"python_data_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found after clone: {csv_path}")

    print(f"  Using data at {csv_path}")
    return csv_path


def load_tep_csv(filepath: Path) -> pd.DataFrame:
    """
    Load TEP CSV file.

    The GitHub CSV format (semicolon-separated):
    - First column: datetime timestamp
    - Columns XMEAS(1)-XMEAS(41): 41 process measurements
    - Columns XMV(1)-XMV(11): 11 manipulated variables
    - Last column STATUS: fault code

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with standardized column names
    """
    # Read semicolon-separated CSV
    df = pd.read_csv(filepath, sep=';')

    n_cols = len(df.columns)
    print(f"    CSV has {n_cols} columns, {len(df)} rows")

    # Rename columns to standardized format
    col_map = {}
    for col in df.columns:
        if col.startswith('XMEAS('):
            # Extract number from XMEAS(N) format
            num = int(col.replace('XMEAS(', '').replace(')', ''))
            col_map[col] = f'XMEAS{num:02d}'
        elif col.startswith('XMV('):
            # Extract number from XMV(N) format
            num = int(col.replace('XMV(', '').replace(')', ''))
            col_map[col] = f'XMV{num:02d}'
        elif col == 'STATUS':
            col_map[col] = 'fault_code'
        elif col == '' or col.startswith('Unnamed'):
            col_map[col] = 'timestamp'
        else:
            # Keep datetime column as timestamp
            col_map[col] = 'timestamp'

    df = df.rename(columns=col_map)
    return df


def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch Tennessee Eastman Process observations.

    Args:
        config: Dict with keys:
            - dataset: "1year" or "3years" (default: "1year")
            - include_xmv: Include manipulated variables (default: True)
            - include_fault: Include fault code as indicator (default: True)
            - sample_rate: Sample every N rows (default: 1, no subsampling)
            - cache_dir: Directory for cached downloads

    Returns:
        List of observation dicts with keys:
            indicator_id, observed_at, value, source
    """
    dataset = config.get("dataset", "1year")
    include_xmv = config.get("include_xmv", True)
    include_fault = config.get("include_fault", True)
    sample_rate = config.get("sample_rate", 1)
    cache_dir = config.get("cache_dir")

    if cache_dir:
        cache_dir = Path(cache_dir)

    # Download data
    csv_path = download_tep(dataset, cache_dir)

    # Load CSV
    print(f"  Loading TEP {dataset} data...")
    df = load_tep_csv(csv_path)

    # Apply sampling if requested
    if sample_rate > 1:
        df = df.iloc[::sample_rate].reset_index(drop=True)
        print(f"    Sampled to {len(df)} rows (1/{sample_rate})")

    all_observations = []

    # Base date for synthetic signal topology
    # TEP sampling is typically 3 minutes, so use that
    base_date = date(2000, 1, 1)
    minutes_per_sample = 3

    # Determine which variables to include
    xmeas_cols = [c for c in df.columns if c.startswith('XMEAS')]
    xmv_cols = [c for c in df.columns if c.startswith('XMV')]

    print(f"    Found {len(xmeas_cols)} XMEAS columns, {len(xmv_cols)} XMV columns")

    # Convert to observations
    for idx, row in df.iterrows():
        # Calculate observation date (3-minute intervals)
        minutes_offset = idx * minutes_per_sample
        days_offset = minutes_offset // (24 * 60)
        obs_date = base_date + timedelta(days=days_offset)

        # XMEAS variables (process measurements)
        for col in xmeas_cols:
            value = row[col]
            if pd.notna(value):
                all_observations.append({
                    "indicator_id": f"TEP_{col}",
                    "observed_at": obs_date,
                    "value": float(value),
                    "source": SOURCE,
                })

        # XMV variables (manipulated variables)
        if include_xmv:
            for col in xmv_cols:
                value = row[col]
                if pd.notna(value):
                    all_observations.append({
                        "indicator_id": f"TEP_{col}",
                        "observed_at": obs_date,
                        "value": float(value),
                        "source": SOURCE,
                    })

        # Fault code (if present)
        if include_fault and 'fault_code' in df.columns:
            fault = row.get('fault_code')
            if pd.notna(fault):
                all_observations.append({
                    "indicator_id": "TEP_FAULT",
                    "observed_at": obs_date,
                    "value": float(fault),
                    "source": SOURCE,
                })

    print(f"  Total: {len(all_observations):,} observations")
    return all_observations


if __name__ == "__main__":
    # Example usage
    config = {
        "dataset": "1year",
        "include_xmv": True,
        "sample_rate": 100,  # Sample every 100 rows for quick test
    }

    results = fetch(config)
    print(f"\nFetched {len(results):,} observations")

    # Show sample
    if results:
        df = pd.DataFrame(results[:20])
        print("\nSample observations:")
        print(df)

    # Show unique indicators
    if results:
        indicators = set(r['indicator_id'] for r in results)
        print(f"\nUnique indicators: {len(indicators)}")
        for ind in sorted(indicators)[:10]:
            print(f"  - {ind}")
