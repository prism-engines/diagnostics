#!/usr/bin/env python3
"""
SABIO-RK Enzyme Kinetics Fetcher for PRISM Validation

Fetches Michaelis-Menten enzyme kinetics data from SABIO-RK database.
Tests whether PRISM can detect substrate saturation regimes.

API Documentation: https://sabiork.h-its.org/sabioRestWebServices

Reference:
    Wittig, U., et al. (2012). SABIO-RK—database for biochemical reaction kinetics.
    Nucleic Acids Research, 40(D1), D790-D798.
    DOI: 10.1093/nar/gkr1046
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
import time

import requests
import polars as pl
import numpy as np


# SABIO-RK REST API endpoints
BASE_URL = "https://sabiork.h-its.org/sabioRestWebServices"
ENTRY_URL = "https://sabiork.h-its.org/entry"


def fetch_michaelis_menten_entries(max_entries: int = 500) -> list[int]:
    """Fetch entry IDs for Michaelis-Menten kinetics."""

    # Query for Michaelis-Menten mechanism type
    query_url = f"{BASE_URL}/searchKineticLaws/entryIDs"
    params = {
        "q": 'KineticMechanismType:"Michaelis-Menten"',
        "format": "txt"
    }

    print(f"Querying SABIO-RK for Michaelis-Menten entries...")
    response = requests.get(query_url, params=params, timeout=60)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []

    # Parse entry IDs (one per line)
    entry_ids = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if line and line.isdigit():
            entry_ids.append(int(line))

    print(f"Found {len(entry_ids)} Michaelis-Menten entries")

    # Limit entries
    if len(entry_ids) > max_entries:
        print(f"Limiting to {max_entries} entries")
        entry_ids = entry_ids[:max_entries]

    return entry_ids


def fetch_entry_details(entry_id: int) -> dict | None:
    """Fetch detailed kinetic parameters for a single entry via SBML."""
    import re

    # Get SBML format (more reliable for kinetic parameters)
    url = f"{BASE_URL}/kineticLaws/{entry_id}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None

        sbml = response.text

        # Extract Km and Vmax from SBML using regex
        km_match = re.search(r'id="Km[^"]*"[^>]*value="([^"]+)"', sbml)
        vmax_match = re.search(r'id="Vmax[^"]*"[^>]*value="([^"]+)"', sbml)

        # Also try alternative patterns
        if not km_match:
            km_match = re.search(r'name="Km"[^>]*value="([^"]+)"', sbml)
        if not vmax_match:
            vmax_match = re.search(r'name="Vmax"[^>]*value="([^"]+)"', sbml)

        km = float(km_match.group(1)) if km_match else None
        vmax = float(vmax_match.group(1)) if vmax_match else None

        # Extract enzyme name from model name or notes
        enzyme_match = re.search(r'<species[^>]*name="([^"]+)"', sbml)
        enzyme_name = enzyme_match.group(1) if enzyme_match else f"enzyme_{entry_id}"

        # Extract organism if present
        org_match = re.search(r'Organism[^<]*<([^>]+)>([^<]+)</\1>', sbml)
        organism = org_match.group(2) if org_match else ""

        result = {
            "entry_id": entry_id,
            "enzyme_name": enzyme_name[:50],
            "ec_number": "",
            "organism": organism[:50],
            "km": km,
            "vmax": vmax,
        }

        return result

    except Exception as e:
        print(f"  Error fetching entry {entry_id}: {e}")
        return None


def simulate_michaelis_menten(
    km: float,
    vmax: float,
    s_max: float = 10.0,
    n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Michaelis-Menten kinetics.

    v = Vmax × [S] / (Km + [S])

    Args:
        km: Michaelis constant
        vmax: Maximum velocity
        s_max: Maximum substrate concentration (as multiple of Km)
        n_points: Number of data points

    Returns:
        substrate: Array of substrate concentrations
        velocity: Array of reaction velocities
    """
    # Substrate range from 0.01*Km to s_max*Km
    substrate = np.linspace(0.01 * km, s_max * km, n_points)
    velocity = vmax * substrate / (km + substrate)

    # Add small noise (5% of Vmax)
    noise = np.random.normal(0, 0.02 * vmax, n_points)
    velocity = np.maximum(velocity + noise, 0)

    return substrate, velocity


def classify_kinetic_regime(substrate: np.ndarray, km: float) -> str:
    """
    Classify substrate concentration regime.

    - Low: [S] << Km (linear regime, v ≈ Vmax*[S]/Km)
    - Saturating: [S] >> Km (plateau regime, v ≈ Vmax)
    - Transition: [S] ≈ Km (half-max velocity)
    """
    s_mean = np.mean(substrate)

    if s_mean < 0.2 * km:
        return "linear"
    elif s_mean > 5 * km:
        return "saturating"
    else:
        return "transition"


def create_prism_observations(
    entries: list[dict],
    output_dir: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate simulated Michaelis-Menten curves and create PRISM-format data.
    """
    observations = []
    indicators = []

    for entry in entries:
        km = entry.get("km")
        vmax = entry.get("vmax")

        if km is None or vmax is None or km <= 0 or vmax <= 0:
            continue

        entry_id = entry["entry_id"]
        enzyme = entry.get("enzyme_name", "unknown")[:30]
        ec = entry.get("ec_number", "")

        # Create indicator ID
        indicator_id = f"sabiork_{entry_id}"

        # Simulate three regimes
        for regime, s_mult in [("linear", 0.2), ("transition", 1.0), ("saturating", 10.0)]:
            regime_id = f"{indicator_id}_{regime}"

            # Simulate kinetics
            substrate, velocity = simulate_michaelis_menten(
                km=km,
                vmax=vmax,
                s_max=s_mult * 2,  # Range around the regime center
                n_points=100
            )

            # Create signal topology (treat substrate as "time")
            base_date = datetime(2020, 1, 1)
            for i, (s, v) in enumerate(zip(substrate, velocity)):
                observations.append({
                    "indicator_id": regime_id,
                    "obs_date": base_date + timedelta(seconds=i),
                    "value": v,
                    "substrate": s,
                })

            # Indicator metadata
            indicators.append({
                "indicator_id": regime_id,
                "entry_id": entry_id,
                "enzyme_name": enzyme,
                "ec_number": ec,
                "organism": entry.get("organism", ""),
                "km": km,
                "vmax": vmax,
                "regime": regime,
                "n_points": 100,
            })

    obs_df = pl.DataFrame(observations)
    ind_df = pl.DataFrame(indicators)

    return obs_df, ind_df


def main():
    parser = argparse.ArgumentParser(description="Fetch SABIO-RK enzyme kinetics data")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/sabiork",
        help="Output directory"
    )
    parser.add_argument(
        "--max-entries", "-n",
        type=int,
        default=100,
        help="Maximum number of entries to fetch"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print("SABIO-RK Enzyme Kinetics Fetcher for PRISM")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Max entries: {args.max_entries}")
    print()

    # Fetch entry IDs
    entry_ids = fetch_michaelis_menten_entries(max_entries=args.max_entries)

    if not entry_ids:
        print("No entries found. Exiting.")
        return

    # Fetch details for each entry
    print(f"\nFetching details for {len(entry_ids)} entries...")
    entries = []

    for i, entry_id in enumerate(entry_ids):
        if i % 10 == 0:
            print(f"  Processing entry {i+1}/{len(entry_ids)}...")

        details = fetch_entry_details(entry_id)
        if details and details.get("km") and details.get("vmax"):
            entries.append(details)

        # Rate limiting
        time.sleep(0.1)

    print(f"\nValid entries with Km and Vmax: {len(entries)}")

    if not entries:
        print("No valid entries found. Exiting.")
        return

    # Create output directories
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    (output_dir / "config").mkdir(parents=True, exist_ok=True)
    (output_dir / "vector").mkdir(parents=True, exist_ok=True)

    # Create PRISM-format data
    from datetime import timedelta

    observations = []
    indicators = []

    for entry in entries:
        km = entry["km"]
        vmax = entry["vmax"]
        entry_id = entry["entry_id"]

        # Simulate three regimes
        for regime, s_mult in [("linear", 0.1), ("transition", 1.0), ("saturating", 10.0)]:
            indicator_id = f"sabiork_{entry_id}_{regime}"

            # Simulate kinetics in this regime
            substrate, velocity = simulate_michaelis_menten(
                km=km,
                vmax=vmax,
                s_max=s_mult * 2,
                n_points=100
            )

            # Create signal topology
            base_date = datetime(2020, 1, 1)
            for i, v in enumerate(velocity):
                observations.append({
                    "indicator_id": indicator_id,
                    "obs_date": base_date + timedelta(seconds=i),
                    "value": float(v),
                })

            # Indicator metadata
            indicators.append({
                "indicator_id": indicator_id,
                "entry_id": entry_id,
                "enzyme_name": entry.get("enzyme_name", "")[:50],
                "ec_number": entry.get("ec_number", ""),
                "organism": entry.get("organism", "")[:50],
                "km": km,
                "vmax": vmax,
                "regime": regime,
                "saturation": s_mult,  # [S]/Km ratio
                "n_points": 100,
            })

    # Convert to DataFrames
    obs_df = pl.DataFrame(observations)
    ind_df = pl.DataFrame(indicators)

    # Save to parquet
    obs_df.write_parquet(output_dir / "raw" / "observations.parquet")
    ind_df.write_parquet(output_dir / "raw" / "indicators.parquet")

    # Create cohorts
    cohorts = pl.DataFrame([{
        "cohort_id": "sabiork_enzyme_kinetics",
        "name": "SABIO-RK Enzyme Kinetics",
        "description": "Michaelis-Menten enzyme kinetics from SABIO-RK database"
    }])
    cohorts.write_parquet(output_dir / "config" / "cohorts.parquet")

    cohort_members = pl.DataFrame([
        {"cohort_id": "sabiork_enzyme_kinetics", "indicator_id": ind_id}
        for ind_id in ind_df["indicator_id"].to_list()
    ])
    cohort_members.write_parquet(output_dir / "config" / "cohort_members.parquet")

    print()
    print("=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Entries with valid kinetics: {len(entries)}")
    print(f"Indicators (3 regimes each): {len(ind_df)}")
    print(f"Observations: {len(obs_df)}")
    print()

    # Summary by regime
    print("Regime summary:")
    print(ind_df.group_by("regime").len())
    print()

    # Sample entries
    print("Sample entries:")
    print(ind_df.select(["indicator_id", "enzyme_name", "km", "vmax", "regime"]).head(10))


if __name__ == "__main__":
    main()
