"""
PRISM Parquet Storage Layer
===========================

5 files. No more, no less.

Directory Structure:
    data/
      {domain}/                     # e.g., cmapss/, tep/, femto/
        observations.parquet        # Raw sensor data
        signals.parquet             # All behavioral signals (dense + sparse)
        geometry.parquet            # System structure at each timestamp
        state.parquet               # Dynamics at each timestamp
        cohorts.parquet             # Discovered entity groupings

Entity Hierarchy:
    domain (cmapss)
    └── entity (engine_47)          # Fails, gets RUL, joins cohort
        └── signal (sensor_1)       # Measures entity
            └── derived (inst_freq) # Computed from signal

File Schemas:
    observations: entity_id, signal_id, timestamp, value
    signals:      entity_id, signal_id, source_signal, engine, signal_type, timestamp, value, mode_id
    geometry:     entity_id, timestamp, divergence, mode_count, coupling_mean, transition_flag, regime, ...
    state:        entity_id, timestamp, position_*, velocity_*, acceleration_*, failure_signature, ...
    cohorts:      entity_id, cohort_id, trajectory_similarity, failure_mode, ...

Usage:
    from prism.db.parquet_store import get_path, OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS

    # Get path to a file
    obs_path = get_path(OBSERVATIONS)  # -> data/{domain}/observations.parquet

    # Explicit domain
    obs_path = get_path(OBSERVATIONS, domain='cmapss')
"""

import os
from pathlib import Path
from typing import List, Optional

# =============================================================================
# THE 5 FILES
# =============================================================================

OBSERVATIONS = "observations"   # Raw sensor data
SIGNALS = "signals"             # All behavioral signals
GEOMETRY = "geometry"           # System structure at each t
STATE = "state"                 # Dynamics at each t
COHORTS = "cohorts"             # Discovered entity groupings

# All valid file names
FILES = [OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS]


# =============================================================================
# PATH FUNCTIONS
# =============================================================================

def get_active_domain() -> str:
    """
    Get the active domain from PRISM_DOMAIN environment variable.

    Returns:
        Active domain name (e.g., 'cmapss', 'tep', 'femto')

    Raises:
        RuntimeError: If no domain is set
    """
    env_domain = os.environ.get("PRISM_DOMAIN")
    if env_domain:
        return env_domain

    raise RuntimeError(
        "No domain specified. Set PRISM_DOMAIN environment variable or use --domain flag."
    )


def get_data_root(domain: str = None) -> Path:
    """
    Return the root data directory for a domain.

    Args:
        domain: Domain name. Defaults to active domain.

    Returns:
        Path to domain data directory (e.g., data/cmapss/)
    """
    env_path = os.environ.get("PRISM_DATA_PATH")
    if env_path:
        base = Path(env_path)
    else:
        base = Path(os.path.expanduser("~/prism-mac/data"))

    domain = domain or get_active_domain()
    return base / domain


def get_path(file: str, domain: str = None) -> Path:
    """
    Return the path to a PRISM output file.

    Args:
        file: File name (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS)
        domain: Domain name. Defaults to active domain.

    Returns:
        Path to parquet file

    Examples:
        >>> get_path(OBSERVATIONS)
        PosixPath('.../data/cmapss/observations.parquet')

        >>> get_path(SIGNALS, domain='tep')
        PosixPath('.../data/tep/signals.parquet')
    """
    if file not in FILES:
        raise ValueError(f"Unknown file: {file}. Valid files: {FILES}")

    return get_data_root(domain) / f"{file}.parquet"


def ensure_directory(domain: str = None) -> Path:
    """
    Create domain directory if it doesn't exist.

    Args:
        domain: Domain name. Defaults to active domain.

    Returns:
        Path to domain directory
    """
    root = get_data_root(domain)
    root.mkdir(parents=True, exist_ok=True)
    return root


def file_exists(file: str, domain: str = None) -> bool:
    """Check if a PRISM output file exists."""
    return get_path(file, domain).exists()


def get_file_size(file: str, domain: str = None) -> Optional[int]:
    """Get file size in bytes, or None if doesn't exist."""
    path = get_path(file, domain)
    if path.exists():
        return path.stat().st_size
    return None


def delete_file(file: str, domain: str = None) -> bool:
    """Delete a file. Returns True if deleted, False if didn't exist."""
    path = get_path(file, domain)
    if path.exists():
        path.unlink()
        return True
    return False


def list_files(domain: str = None) -> List[str]:
    """List all existing PRISM output files for a domain."""
    return [f for f in FILES if file_exists(f, domain)]


def list_domains() -> List[str]:
    """List all domain directories."""
    env_path = os.environ.get("PRISM_DATA_PATH")
    if env_path:
        base = Path(env_path)
    else:
        base = Path(os.path.expanduser("~/prism-mac/data"))

    if not base.exists():
        return []

    # A domain has at least one of the 5 files
    domains = []
    for d in base.iterdir():
        if d.is_dir():
            for f in FILES:
                if (d / f"{f}.parquet").exists():
                    domains.append(d.name)
                    break
    return sorted(domains)


def get_status(domain: str = None) -> dict:
    """
    Get status of all PRISM output files.

    Returns:
        Dict with file status and sizes
    """
    status = {}
    for f in FILES:
        path = get_path(f, domain)
        if path.exists():
            size = path.stat().st_size
            status[f] = {"exists": True, "size_bytes": size, "size_mb": size / 1024 / 1024}
        else:
            status[f] = {"exists": False, "size_bytes": 0, "size_mb": 0}
    return status


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PRISM Storage - 5 Files")
    parser.add_argument("--domain", help="Domain name (cmapss, tep, femto, etc.)")
    parser.add_argument("--init", action="store_true", help="Create domain directory")
    parser.add_argument("--list", action="store_true", help="List files for domain")
    parser.add_argument("--list-domains", action="store_true", help="List all domains")
    parser.add_argument("--status", action="store_true", help="Show file status")

    args = parser.parse_args()

    if args.domain:
        os.environ["PRISM_DOMAIN"] = args.domain

    if args.init:
        path = ensure_directory(args.domain)
        print(f"Created: {path}")
        print("\nExpected files:")
        for f in FILES:
            print(f"  {f}.parquet")

    elif args.list_domains:
        domains = list_domains()
        if domains:
            print("Domains:")
            for d in domains:
                print(f"  {d}/")
        else:
            print("No domains found")

    elif args.list:
        files = list_files(args.domain)
        if files:
            print(f"Files in {args.domain or 'active domain'}:")
            for f in files:
                size = get_file_size(f, args.domain)
                print(f"  {f}.parquet ({size:,} bytes)")
        else:
            print("No files found")

    elif args.status:
        status = get_status(args.domain)
        print(f"Status for {args.domain or 'active domain'}:")
        print("-" * 50)
        for f, info in status.items():
            if info["exists"]:
                print(f"  ✓ {f}.parquet ({info['size_mb']:.2f} MB)")
            else:
                print(f"  ✗ {f}.parquet (missing)")

    else:
        parser.print_help()
