#!/usr/bin/env python3
"""
USGS Earthquake Fetcher

Standalone fetcher for USGS earthquake data.
Does NOT import from prism - this is a workspace script.

Converts earthquake catalogs into daily signal topology for geometric analysis.

INDICATORS GENERATED per region:
    - {REGION}_COUNT     : Daily earthquake count
    - {REGION}_ENERGY    : Daily seismic energy release (log10 Joules)
    - {REGION}_MAXMAG    : Maximum magnitude per day
    - {REGION}_MEANMAG   : Mean magnitude per day
    - {REGION}_DEPTH     : Mean depth per day (km)

Usage:
    from fetchers.usgs_fetcher import fetch

    config = {
        "indicators": ["SAN_ANDREAS_COUNT", "JAPAN_ENERGY"],
        "start_date": "2020-01-01",  # optional
    }
    observations = fetch(config)
"""

import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


SOURCE = "usgs"
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


# =============================================================================
# REGION DEFINITIONS
# =============================================================================

@dataclass
class SeismicRegion:
    """Definition of a seismic monitoring region."""
    name: str
    description: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_magnitude: float = 2.5

    @property
    def bbox_params(self) -> Dict[str, float]:
        return {
            "minlatitude": self.min_lat,
            "maxlatitude": self.max_lat,
            "minlongitude": self.min_lon,
            "maxlongitude": self.max_lon,
        }


REGIONS: Dict[str, SeismicRegion] = {
    "SAN_ANDREAS": SeismicRegion(
        name="SAN_ANDREAS",
        description="California - San Andreas Fault System",
        min_lat=32.0, max_lat=42.0,
        min_lon=-125.0, max_lon=-114.0,
        min_magnitude=2.0,
    ),
    "CASCADIA": SeismicRegion(
        name="CASCADIA",
        description="Pacific Northwest - Cascadia Subduction Zone",
        min_lat=40.0, max_lat=52.0,
        min_lon=-130.0, max_lon=-120.0,
        min_magnitude=2.5,
    ),
    "NEW_MADRID": SeismicRegion(
        name="NEW_MADRID",
        description="Central US - New Madrid Seismic Zone",
        min_lat=35.0, max_lat=40.0,
        min_lon=-92.0, max_lon=-88.0,
        min_magnitude=2.0,
    ),
    "ALASKA": SeismicRegion(
        name="ALASKA",
        description="Alaska-Aleutian Subduction Zone",
        min_lat=50.0, max_lat=72.0,
        min_lon=-180.0, max_lon=-130.0,
        min_magnitude=3.0,
    ),
    "YELLOWSTONE": SeismicRegion(
        name="YELLOWSTONE",
        description="Yellowstone Volcanic Region",
        min_lat=44.0, max_lat=45.5,
        min_lon=-111.5, max_lon=-109.5,
        min_magnitude=1.5,
    ),
    "JAPAN": SeismicRegion(
        name="JAPAN",
        description="Japan Trench System",
        min_lat=30.0, max_lat=46.0,
        min_lon=128.0, max_lon=146.0,
        min_magnitude=4.0,
    ),
    "INDONESIA": SeismicRegion(
        name="INDONESIA",
        description="Sunda Megathrust",
        min_lat=-12.0, max_lat=8.0,
        min_lon=95.0, max_lon=140.0,
        min_magnitude=4.5,
    ),
    "CHILE": SeismicRegion(
        name="CHILE",
        description="Chile-Peru Trench",
        min_lat=-56.0, max_lat=-17.0,
        min_lon=-80.0, max_lon=-66.0,
        min_magnitude=4.0,
    ),
    "GLOBAL": SeismicRegion(
        name="GLOBAL",
        description="Worldwide M5.0+",
        min_lat=-90.0, max_lat=90.0,
        min_lon=-180.0, max_lon=180.0,
        min_magnitude=5.0,
    ),
}

INDICATOR_TYPES = ["COUNT", "ENERGY", "MAXMAG", "MEANMAG", "DEPTH"]


def magnitude_to_energy(magnitude: float) -> float:
    """Convert earthquake magnitude to energy in Joules (Gutenberg-Richter)."""
    return 10 ** (1.5 * magnitude + 4.8)


def parse_indicator_id(indicator_id: str) -> Tuple[str, str]:
    """Parse indicator ID into (region, type)."""
    for type_suffix in INDICATOR_TYPES:
        if indicator_id.endswith(f"_{type_suffix}"):
            region = indicator_id[:-len(type_suffix)-1]
            return region, type_suffix
    raise ValueError(f"Unknown indicator format: {indicator_id}")


# =============================================================================
# API CLIENT
# =============================================================================

def fetch_earthquakes(
    region: SeismicRegion,
    start_date: date,
    end_date: date,
    rate_limit_delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch earthquakes from USGS API."""
    time.sleep(rate_limit_delay)

    params = {
        "format": "csv",
        "starttime": start_date.isoformat(),
        "endtime": (end_date + timedelta(days=1)).isoformat(),
        "minmagnitude": region.min_magnitude,
        "orderby": "time",
        **region.bbox_params,
    }

    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()

    if not response.text.strip():
        return pd.DataFrame()

    df = pd.read_csv(
        pd.io.common.StringIO(response.text),
        parse_dates=["time"],
        usecols=["time", "latitude", "longitude", "depth", "mag", "place"],
    )
    return df


def fetch_region_range(
    region: SeismicRegion,
    start_date: date,
    end_date: date,
    chunk_months: int = 6,
) -> pd.DataFrame:
    """Fetch earthquakes, chunking to avoid API limits."""
    all_data = []
    current_start = start_date

    while current_start < end_date:
        chunk_end = min(
            current_start + timedelta(days=chunk_months * 30),
            end_date
        )

        df = fetch_earthquakes(region, current_start, chunk_end)
        if not df.empty:
            all_data.append(df)

        current_start = chunk_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time", "latitude", "longitude", "mag"])
    return combined.sort_values("time").reset_index(drop=True)


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_to_daily(
    earthquakes: pd.DataFrame,
    indicator_type: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Aggregate earthquake events to daily signal topology."""
    if earthquakes.empty:
        dates = pd.date_range(start_date, end_date, freq="D")
        if indicator_type == "COUNT":
            return pd.DataFrame({"date": dates, "value": 0})
        else:
            return pd.DataFrame({"date": dates, "value": np.nan})

    eq = earthquakes.copy()
    eq["date"] = pd.to_datetime(eq["time"]).dt.date

    all_dates = pd.date_range(start_date, end_date, freq="D")
    date_df = pd.DataFrame({"date": all_dates.date})

    if indicator_type == "COUNT":
        daily = eq.groupby("date").size().reset_index(name="value")
    elif indicator_type == "ENERGY":
        eq["energy"] = eq["mag"].apply(magnitude_to_energy)
        daily = eq.groupby("date")["energy"].sum().reset_index(name="value")
        daily["value"] = np.log10(daily["value"])
    elif indicator_type == "MAXMAG":
        daily = eq.groupby("date")["mag"].max().reset_index(name="value")
    elif indicator_type == "MEANMAG":
        daily = eq.groupby("date")["mag"].mean().reset_index(name="value")
    elif indicator_type == "DEPTH":
        daily = eq.groupby("date")["depth"].mean().reset_index(name="value")
    else:
        raise ValueError(f"Unknown indicator type: {indicator_type}")

    daily["date"] = pd.to_datetime(daily["date"])
    date_df["date"] = pd.to_datetime(date_df["date"])

    result = date_df.merge(daily, on="date", how="left")

    if indicator_type == "COUNT":
        result["value"] = result["value"].fillna(0).astype(int)

    return result[["date", "value"]]


# =============================================================================
# MAIN FETCH FUNCTION
# =============================================================================

def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch observations for indicators specified in config.

    Args:
        config: Dict with keys:
            - indicators: list of indicator IDs like "SAN_ANDREAS_COUNT"
            - start_date: optional start date string (YYYY-MM-DD)
            - end_date: optional end date string (YYYY-MM-DD)

    Returns:
        List of observation dicts with keys:
            indicator_id, observed_at, value, source
    """
    indicators = config.get("indicators", [])
    if not indicators:
        raise ValueError("Config must contain 'indicators' list")

    # Parse dates
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * 10)

    if config.get("start_date"):
        start_date = datetime.strptime(config["start_date"], "%Y-%m-%d").date()
    if config.get("end_date"):
        end_date = datetime.strptime(config["end_date"], "%Y-%m-%d").date()

    # Cache raw data by region
    region_cache: Dict[str, pd.DataFrame] = {}
    all_observations = []

    for indicator_id in indicators:
        try:
            region_name, indicator_type = parse_indicator_id(indicator_id)
        except ValueError as e:
            print(f"  {indicator_id}: FAILED - {e}")
            continue

        if region_name not in REGIONS:
            print(f"  {indicator_id}: FAILED - Unknown region: {region_name}")
            continue

        region = REGIONS[region_name]

        # Fetch or use cached raw data
        cache_key = f"{region_name}_{start_date}_{end_date}"
        if cache_key not in region_cache:
            print(f"  Fetching raw data for {region_name}...")
            region_cache[cache_key] = fetch_region_range(region, start_date, end_date)
            print(f"    Got {len(region_cache[cache_key])} earthquakes")

        earthquakes = region_cache[cache_key]

        # Aggregate to daily
        df = aggregate_to_daily(earthquakes, indicator_type, start_date, end_date)

        # Convert to observation dicts
        for _, row in df.iterrows():
            if pd.notna(row["value"]):
                all_observations.append({
                    "indicator_id": indicator_id,
                    "observed_at": row["date"],
                    "value": row["value"],
                    "source": SOURCE,
                })

        print(f"  {indicator_id}: {len(df)} days")

    return all_observations


if __name__ == "__main__":
    config = {
        "indicators": ["SAN_ANDREAS_COUNT", "SAN_ANDREAS_MAXMAG"],
        "start_date": "2023-01-01",
    }
    results = fetch(config)
    print(f"\nTotal: {len(results)} observations")
