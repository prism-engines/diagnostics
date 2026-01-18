#!/usr/bin/env python3
"""
PRISM Climate Data Fetcher

Fetches climate and environmental data from multiple authoritative sources:
- NOAA Global Monitoring Laboratory (CO2, CH4, N2O)
- NASA GISS (Global temperature anomalies)
- NOAA NCEI (Temperature, precipitation, drought)
- NOAA CPC (Teleconnection indices)
- NSIDC (Sea ice extent)
- NOAA ESRL (Solar, volcanic)

All data is public domain / freely available for research.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests


def fetch(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Main fetch function called by prism.entry_points.fetch.

    Args:
        config: YAML configuration with 'indicators' list

    Returns:
        List of observation dicts with keys:
        - indicator_id: str
        - observed_at: date
        - value: float
        - source: str
        - domain: str
    """
    indicators = config.get("indicators", [])
    start_date = config.get("start_date", "1950-01-01")
    end_date = config.get("end_date", datetime.now().strftime("%Y-%m-%d"))

    all_observations = []

    # Group indicators by data source for efficient fetching
    indicator_groups = _group_indicators(indicators)

    for source_name, source_indicators in indicator_groups.items():
        logger.info(f"Fetching {len(source_indicators)} indicators from {source_name}")

        try:
            if source_name == "noaa_gml":
                obs = _fetch_noaa_gml(source_indicators, start_date, end_date)
            elif source_name == "nasa_giss":
                obs = _fetch_nasa_giss(source_indicators, start_date, end_date)
            elif source_name == "noaa_ncei":
                obs = _fetch_noaa_ncei(source_indicators, start_date, end_date)
            elif source_name == "noaa_cpc":
                obs = _fetch_noaa_cpc(source_indicators, start_date, end_date)
            elif source_name == "nsidc":
                obs = _fetch_nsidc(source_indicators, start_date, end_date)
            elif source_name == "noaa_sea_level":
                obs = _fetch_sea_level(source_indicators, start_date, end_date)
            elif source_name == "esrl":
                obs = _fetch_esrl(source_indicators, start_date, end_date)
            else:
                logger.warning(f"Unknown source: {source_name}")
                obs = []

            all_observations.extend(obs)
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {e}")
            continue

    logger.info(f"Total observations fetched: {len(all_observations)}")
    return all_observations


def _group_indicators(indicators: List[str]) -> Dict[str, List[str]]:
    """Group indicators by their data source."""
    groups = {
        "noaa_gml": [],      # Greenhouse gases
        "nasa_giss": [],     # Temperature anomalies
        "noaa_ncei": [],     # Temperature, precip, drought
        "noaa_cpc": [],      # Teleconnections
        "nsidc": [],         # Sea ice
        "noaa_sea_level": [],# Sea level
        "esrl": [],          # Solar, volcanic
    }

    source_mapping = {
        # Greenhouse gases -> NOAA GML
        "CO2_": "noaa_gml",
        "CH4_": "noaa_gml",
        "N2O_": "noaa_gml",
        "SF6_": "noaa_gml",

        # Temperature -> NASA GISS / NOAA
        "GISS_": "nasa_giss",
        "HADCRUT": "nasa_giss",
        "NOAA_TEMP": "noaa_ncei",

        # Ocean/SST
        "SST_": "noaa_cpc",
        "OCEAN_HEAT": "noaa_ncei",

        # Sea level
        "SEA_LEVEL": "noaa_sea_level",

        # Ice
        "ARCTIC_": "nsidc",
        "ANTARCTIC_": "nsidc",
        "GREENLAND_": "nsidc",
        "SNOW_": "nsidc",
        "GLACIER_": "nsidc",

        # Precipitation/Drought
        "PRECIP_": "noaa_ncei",
        "PDSI_": "noaa_ncei",
        "SPI_": "noaa_ncei",
        "DROUGHT_": "noaa_ncei",

        # Extreme weather
        "CEI_": "noaa_ncei",
        "HURRICANE_": "noaa_ncei",
        "TORNADO_": "noaa_ncei",
        "HEATWAVE_": "noaa_ncei",
        "COLD_WAVE_": "noaa_ncei",

        # Teleconnections
        "NAO_": "noaa_cpc",
        "AO_": "noaa_cpc",
        "PNA_": "noaa_cpc",
        "SOI_": "noaa_cpc",
        "MEI_": "noaa_cpc",
        "QBO_": "noaa_cpc",
        "AAO_": "noaa_cpc",

        # Solar/Volcanic
        "TSI_": "esrl",
        "SUNSPOT_": "esrl",
        "AOD_": "esrl",

        # Carbon/Vegetation
        "CARBON_": "noaa_gml",
        "FOSSIL_": "noaa_gml",
        "LAND_USE_": "noaa_gml",
        "NDVI_": "noaa_ncei",
        "LAI_": "noaa_ncei",
        "GPP_": "noaa_ncei",
        "FIRE_": "noaa_ncei",

        # Air quality
        "PM25_": "noaa_ncei",
        "OZONE_": "noaa_gml",
        "STRATOSPHERIC_": "noaa_gml",
    }

    for ind in indicators:
        assigned = False
        for prefix, source in source_mapping.items():
            if ind.startswith(prefix):
                groups[source].append(ind)
                assigned = True
                break

        if not assigned:
            # Default to NOAA NCEI for unknown
            groups["noaa_ncei"].append(ind)

    return {k: v for k, v in groups.items() if v}


# =============================================================================
# NOAA Global Monitoring Laboratory - CO2, CH4, N2O, SF6
# =============================================================================

def _fetch_noaa_gml(indicators: List[str], start_date: str, end_date: str) -> List[Dict]:
    """Fetch greenhouse gas data from NOAA GML."""
    observations = []

    # Mauna Loa CO2 - monthly
    if any(ind.startswith("CO2_") for ind in indicators):
        try:
            url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
            df = pd.read_csv(url, comment='#', names=[
                'year', 'month', 'decimal_date', 'monthly_avg',
                'deseasonalized', 'days', 'std', 'uncertainty'
            ], skipinitialspace=True)

            # Convert columns to numeric
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['month'] = pd.to_numeric(df['month'], errors='coerce')
            df['monthly_avg'] = pd.to_numeric(df['monthly_avg'], errors='coerce')
            df = df.dropna(subset=['year', 'month', 'monthly_avg'])

            for _, row in df.iterrows():
                if row['monthly_avg'] < 0:  # Missing data flag
                    continue

                obs_date = date(int(row['year']), int(row['month']), 15)
                if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                    continue

                if "CO2_MONTHLY" in indicators:
                    observations.append({
                        "indicator_id": "CO2_MONTHLY",
                        "observed_at": obs_date,
                        "value": float(row['monthly_avg']),
                        "source": "noaa_gml",
                        "domain": "climate",
                    })

            # Calculate annual and growth rate
            if "CO2_ANNUAL" in indicators or "CO2_GROWTH_RATE" in indicators:
                df_valid = df[df['monthly_avg'] > 0].copy()
                annual = df_valid.groupby('year')['monthly_avg'].mean()

                for year, value in annual.items():
                    obs_date = date(int(year), 7, 1)
                    if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                        continue

                    if "CO2_ANNUAL" in indicators:
                        observations.append({
                            "indicator_id": "CO2_ANNUAL",
                            "observed_at": obs_date,
                            "value": float(value),
                            "source": "noaa_gml",
                            "domain": "climate",
                        })

                # Growth rate
                if "CO2_GROWTH_RATE" in indicators:
                    for i in range(1, len(annual)):
                        year = annual.index[i]
                        growth = annual.iloc[i] - annual.iloc[i-1]
                        obs_date = date(int(year), 7, 1)
                        if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                            observations.append({
                                "indicator_id": "CO2_GROWTH_RATE",
                                "observed_at": obs_date,
                                "value": float(growth),
                                "source": "noaa_gml",
                                "domain": "climate",
                            })

            logger.info(f"  Fetched CO2 data: {len([o for o in observations if 'CO2' in o['indicator_id']])} obs")

        except Exception as e:
            logger.error(f"  Error fetching CO2: {e}")

    # Methane CH4
    if "CH4_MONTHLY" in indicators:
        try:
            url = "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv"
            df = pd.read_csv(url, comment='#', names=[
                'year', 'month', 'decimal_date', 'monthly_avg',
                'std', 'trend', 'uncertainty'
            ], skipinitialspace=True)

            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['month'] = pd.to_numeric(df['month'], errors='coerce')
            df['monthly_avg'] = pd.to_numeric(df['monthly_avg'], errors='coerce')
            df = df.dropna(subset=['year', 'month', 'monthly_avg'])
            df = df[(df['month'] >= 1) & (df['month'] <= 12)]

            for _, row in df.iterrows():
                if row['monthly_avg'] < 0:
                    continue
                obs_date = date(int(row['year']), int(row['month']), 15)
                if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                    continue

                observations.append({
                    "indicator_id": "CH4_MONTHLY",
                    "observed_at": obs_date,
                    "value": float(row['monthly_avg']),
                    "source": "noaa_gml",
                    "domain": "climate",
                })

            logger.info(f"  Fetched CH4 data: {len([o for o in observations if 'CH4' in o['indicator_id']])} obs")

        except Exception as e:
            logger.error(f"  Error fetching CH4: {e}")

    # Nitrous Oxide N2O
    if "N2O_MONTHLY" in indicators:
        try:
            url = "https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.csv"
            df = pd.read_csv(url, comment='#', names=[
                'year', 'month', 'decimal_date', 'monthly_avg',
                'std', 'trend', 'uncertainty'
            ], skipinitialspace=True)

            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['month'] = pd.to_numeric(df['month'], errors='coerce')
            df['monthly_avg'] = pd.to_numeric(df['monthly_avg'], errors='coerce')
            df = df.dropna(subset=['year', 'month', 'monthly_avg'])
            df = df[(df['month'] >= 1) & (df['month'] <= 12)]

            for _, row in df.iterrows():
                if row['monthly_avg'] < 0:
                    continue
                obs_date = date(int(row['year']), int(row['month']), 15)
                if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                    continue

                observations.append({
                    "indicator_id": "N2O_MONTHLY",
                    "observed_at": obs_date,
                    "value": float(row['monthly_avg']),
                    "source": "noaa_gml",
                    "domain": "climate",
                })

            logger.info(f"  Fetched N2O data: {len([o for o in observations if 'N2O' in o['indicator_id']])} obs")

        except Exception as e:
            logger.error(f"  Error fetching N2O: {e}")

    # SF6
    if "SF6_MONTHLY" in indicators:
        try:
            url = "https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_mm_gl.csv"
            df = pd.read_csv(url, comment='#', names=[
                'year', 'month', 'decimal_date', 'monthly_avg',
                'std', 'trend', 'uncertainty'
            ], skipinitialspace=True)

            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['month'] = pd.to_numeric(df['month'], errors='coerce')
            df['monthly_avg'] = pd.to_numeric(df['monthly_avg'], errors='coerce')
            df = df.dropna(subset=['year', 'month', 'monthly_avg'])
            df = df[(df['month'] >= 1) & (df['month'] <= 12)]

            for _, row in df.iterrows():
                if row['monthly_avg'] < 0:
                    continue
                obs_date = date(int(row['year']), int(row['month']), 15)
                if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                    continue

                observations.append({
                    "indicator_id": "SF6_MONTHLY",
                    "observed_at": obs_date,
                    "value": float(row['monthly_avg']),
                    "source": "noaa_gml",
                    "domain": "climate",
                })

            logger.info(f"  Fetched SF6 data: {len([o for o in observations if 'SF6' in o['indicator_id']])} obs")

        except Exception as e:
            logger.error(f"  Error fetching SF6: {e}")

    return observations


# =============================================================================
# NASA GISS - Global Temperature Anomalies
# =============================================================================

def _fetch_nasa_giss(indicators: List[str], start_date: str, end_date: str) -> List[Dict]:
    """Fetch temperature anomaly data from NASA GISS."""
    observations = []

    try:
        # GISS Surface Temperature Analysis
        url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
        df = pd.read_csv(url, skiprows=1)

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for _, row in df.iterrows():
            year = int(row['Year'])

            for month_idx, month_name in enumerate(months, 1):
                if month_name not in row or pd.isna(row[month_name]):
                    continue

                try:
                    value = float(row[month_name])
                except (ValueError, TypeError):
                    continue

                if value == 0 and month_idx > datetime.now().month and year == datetime.now().year:
                    continue  # Skip future months

                obs_date = date(year, month_idx, 15)
                if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                    continue

                if "GISS_TEMP_GLOBAL" in indicators:
                    observations.append({
                        "indicator_id": "GISS_TEMP_GLOBAL",
                        "observed_at": obs_date,
                        "value": value,
                        "source": "nasa_giss",
                        "domain": "climate",
                    })

        logger.info(f"  Fetched GISS global temp: {len(observations)} obs")

    except Exception as e:
        logger.error(f"  Error fetching NASA GISS: {e}")

    # Northern Hemisphere
    if "GISS_TEMP_NH" in indicators:
        try:
            url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv"
            df = pd.read_csv(url, skiprows=1)

            for _, row in df.iterrows():
                year = int(row['Year'])
                for month_idx, month_name in enumerate(months, 1):
                    if month_name not in row or pd.isna(row[month_name]):
                        continue
                    try:
                        value = float(row[month_name])
                    except (ValueError, TypeError):
                        continue

                    obs_date = date(year, month_idx, 15)
                    if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                        observations.append({
                            "indicator_id": "GISS_TEMP_NH",
                            "observed_at": obs_date,
                            "value": value,
                            "source": "nasa_giss",
                            "domain": "climate",
                        })

            logger.info(f"  Fetched GISS NH temp")

        except Exception as e:
            logger.error(f"  Error fetching GISS NH: {e}")

    # Southern Hemisphere
    if "GISS_TEMP_SH" in indicators:
        try:
            url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.csv"
            df = pd.read_csv(url, skiprows=1)

            for _, row in df.iterrows():
                year = int(row['Year'])
                for month_idx, month_name in enumerate(months, 1):
                    if month_name not in row or pd.isna(row[month_name]):
                        continue
                    try:
                        value = float(row[month_name])
                    except (ValueError, TypeError):
                        continue

                    obs_date = date(year, month_idx, 15)
                    if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                        observations.append({
                            "indicator_id": "GISS_TEMP_SH",
                            "observed_at": obs_date,
                            "value": value,
                            "source": "nasa_giss",
                            "domain": "climate",
                        })

            logger.info(f"  Fetched GISS SH temp")

        except Exception as e:
            logger.error(f"  Error fetching GISS SH: {e}")

    return observations


# =============================================================================
# NOAA CPC - Teleconnection Indices
# =============================================================================

def _fetch_noaa_cpc(indicators: List[str], start_date: str, end_date: str) -> List[Dict]:
    """Fetch teleconnection indices from NOAA CPC."""
    observations = []

    # Index definitions: (indicator_id, url_path, column_name)
    indices = [
        ("NAO_INDEX", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table", "NAO"),
        ("AO_INDEX", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table", "AO"),
        ("PNA_INDEX", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table", "PNA"),
        ("AAO_INDEX", "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii.table", "AAO"),
    ]

    for ind_id, url, name in indices:
        if ind_id not in indicators:
            continue

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            lines = response.text.strip().split('\n')

            for line in lines:
                parts = line.split()
                if len(parts) < 13:
                    continue

                try:
                    year = int(parts[0])
                except ValueError:
                    continue

                for month_idx in range(1, 13):
                    try:
                        value = float(parts[month_idx])
                    except (ValueError, IndexError):
                        continue

                    if value == -99.99 or value == -999:  # Missing data
                        continue

                    obs_date = date(year, month_idx, 15)
                    if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                        continue

                    observations.append({
                        "indicator_id": ind_id,
                        "observed_at": obs_date,
                        "value": value,
                        "source": "noaa_cpc",
                        "domain": "climate",
                    })

            logger.info(f"  Fetched {ind_id}: {len([o for o in observations if o['indicator_id'] == ind_id])} obs")
            time.sleep(REQUEST_DELAY)

        except Exception as e:
            logger.error(f"  Error fetching {ind_id}: {e}")

    # SOI - Southern Oscillation Index
    if "SOI_INDEX" in indicators:
        try:
            url = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            lines = response.text.strip().split('\n')

            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) < 13:
                    continue

                try:
                    year = int(parts[0])
                except ValueError:
                    continue

                for month_idx in range(1, 13):
                    try:
                        value = float(parts[month_idx])
                    except (ValueError, IndexError):
                        continue

                    if value == -999.9:
                        continue

                    obs_date = date(year, month_idx, 15)
                    if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                        observations.append({
                            "indicator_id": "SOI_INDEX",
                            "observed_at": obs_date,
                            "value": value,
                            "source": "noaa_cpc",
                            "domain": "climate",
                        })

            logger.info(f"  Fetched SOI_INDEX")

        except Exception as e:
            logger.error(f"  Error fetching SOI: {e}")

    # Nino 3.4 SST
    if "SST_NINO34" in indicators:
        try:
            url = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            lines = response.text.strip().split('\n')

            for line in lines[1:]:
                parts = line.split()
                if len(parts) < 10:
                    continue

                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    nino34 = float(parts[4])  # NINO3.4 column
                except (ValueError, IndexError):
                    continue

                obs_date = date(year, month, 15)
                if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                    observations.append({
                        "indicator_id": "SST_NINO34",
                        "observed_at": obs_date,
                        "value": nino34,
                        "source": "noaa_cpc",
                        "domain": "climate",
                    })

            logger.info(f"  Fetched SST_NINO34")

        except Exception as e:
            logger.error(f"  Error fetching Nino34: {e}")

    return observations


# =============================================================================
# NSIDC - Sea Ice Data (Updated for v4.0 format)
# =============================================================================

def _fetch_nsidc(indicators: List[str], start_date: str, end_date: str) -> List[Dict]:
    """Fetch sea ice data from NSIDC Sea Ice Index v4.0."""
    observations = []

    # Arctic Sea Ice Extent (monthly) - now split by month in v4.0
    if any(ind.startswith("ARCTIC_SEA_ICE") for ind in indicators):
        try:
            base_url = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data"
            all_data = []

            for month in range(1, 13):
                url = f"{base_url}/N_{month:02d}_extent_v4.0.csv"
                try:
                    df = pd.read_csv(url, skipinitialspace=True)
                    df.columns = [c.strip() for c in df.columns]
                    all_data.append(df)
                except Exception as e:
                    logger.warning(f"  Could not fetch Arctic month {month}: {e}")
                    continue

            if all_data:
                df = pd.concat(all_data, ignore_index=True)

                for _, row in df.iterrows():
                    try:
                        year = int(row['year'])
                        month = int(row['mo'])
                        extent = float(row['extent'])
                        area = float(row['area']) if 'area' in row and pd.notna(row['area']) else None
                    except (ValueError, TypeError, KeyError):
                        continue

                    if extent < 0:  # Missing data flag
                        continue

                    obs_date = date(year, month, 15)
                    if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                        continue

                    if "ARCTIC_SEA_ICE_EXTENT" in indicators:
                        observations.append({
                            "indicator_id": "ARCTIC_SEA_ICE_EXTENT",
                            "observed_at": obs_date,
                            "value": extent,
                            "source": "nsidc",
                            "domain": "climate",
                        })

                    if "ARCTIC_SEA_ICE_AREA" in indicators and area is not None and area > 0:
                        observations.append({
                            "indicator_id": "ARCTIC_SEA_ICE_AREA",
                            "observed_at": obs_date,
                            "value": area,
                            "source": "nsidc",
                            "domain": "climate",
                        })

                logger.info(f"  Fetched Arctic sea ice: {len([o for o in observations if 'ARCTIC' in o['indicator_id']])} obs")

        except Exception as e:
            logger.error(f"  Error fetching Arctic ice: {e}")

    # Antarctic Sea Ice Extent
    if any(ind.startswith("ANTARCTIC_SEA_ICE") for ind in indicators):
        try:
            base_url = "https://noaadata.apps.nsidc.org/NOAA/G02135/south/monthly/data"
            all_data = []

            for month in range(1, 13):
                url = f"{base_url}/S_{month:02d}_extent_v4.0.csv"
                try:
                    df = pd.read_csv(url, skipinitialspace=True)
                    df.columns = [c.strip() for c in df.columns]
                    all_data.append(df)
                except Exception:
                    continue

            if all_data:
                df = pd.concat(all_data, ignore_index=True)

                for _, row in df.iterrows():
                    try:
                        year = int(row['year'])
                        month = int(row['mo'])
                        extent = float(row['extent'])
                        area = float(row['area']) if 'area' in row and pd.notna(row['area']) else None
                    except (ValueError, TypeError, KeyError):
                        continue

                    if extent < 0:
                        continue

                    obs_date = date(year, month, 15)
                    if obs_date < _parse_date(start_date) or obs_date > _parse_date(end_date):
                        continue

                    if "ANTARCTIC_SEA_ICE_EXTENT" in indicators:
                        observations.append({
                            "indicator_id": "ANTARCTIC_SEA_ICE_EXTENT",
                            "observed_at": obs_date,
                            "value": extent,
                            "source": "nsidc",
                            "domain": "climate",
                        })

                    if "ANTARCTIC_SEA_ICE_AREA" in indicators and area is not None and area > 0:
                        observations.append({
                            "indicator_id": "ANTARCTIC_SEA_ICE_AREA",
                            "observed_at": obs_date,
                            "value": area,
                            "source": "nsidc",
                            "domain": "climate",
                        })

                logger.info(f"  Fetched Antarctic sea ice: {len([o for o in observations if 'ANTARCTIC' in o['indicator_id']])} obs")

        except Exception as e:
            logger.error(f"  Error fetching Antarctic ice: {e}")

    return observations


# =============================================================================
# NOAA - Sea Level (Updated URL)
# =============================================================================

def _fetch_sea_level(indicators: List[str], start_date: str, end_date: str) -> List[Dict]:
    """Fetch global mean sea level data from NOAA STAR."""
    observations = []

    if "SEA_LEVEL_GLOBAL" in indicators:
        try:
            # NOAA Laboratory for Satellite Altimetry - updated URL
            url = "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gbl_free_ref_90.csv"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            lines = response.text.strip().split('\n')

            # Skip comment lines starting with #
            data_lines = [l for l in lines if not l.startswith('#') and l.strip()]

            for line in data_lines:
                parts = line.split(',')
                if len(parts) < 2:
                    continue

                try:
                    # Format: decimal year, sea level anomaly (mm)
                    decimal_year = float(parts[0].strip())
                    value = float(parts[1].strip())
                except (ValueError, IndexError):
                    continue

                year = int(decimal_year)
                month = int((decimal_year - year) * 12) + 1
                month = max(1, min(12, month))

                obs_date = date(year, month, 15)
                if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                    observations.append({
                        "indicator_id": "SEA_LEVEL_GLOBAL",
                        "observed_at": obs_date,
                        "value": value,
                        "source": "noaa_star",
                        "domain": "climate",
                    })

            logger.info(f"  Fetched sea level: {len(observations)} obs")

        except Exception as e:
            logger.error(f"  Error fetching sea level: {e}")

    return observations


# =============================================================================
# NOAA NCEI - Temperature, Precipitation, Drought
# =============================================================================

def _fetch_noaa_ncei(indicators: List[str], start_date: str, end_date: str) -> List[Dict]:
    """Fetch climate data from NOAA NCEI."""
    observations = []

    # Global temperature anomaly
    if "NOAA_TEMP_GLOBAL" in indicators:
        try:
            url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/signal topology/globe/land_ocean/1/0/1880-2024.csv"

            df = pd.read_csv(url, skiprows=4, names=['date', 'value'])

            for _, row in df.iterrows():
                try:
                    date_str = str(int(row['date']))
                    year = int(date_str[:4])
                    month = int(date_str[4:])
                    value = float(row['value'])
                except (ValueError, TypeError):
                    continue

                obs_date = date(year, month, 15)
                if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                    observations.append({
                        "indicator_id": "NOAA_TEMP_GLOBAL",
                        "observed_at": obs_date,
                        "value": value,
                        "source": "noaa_ncei",
                        "domain": "climate",
                    })

            logger.info(f"  Fetched NOAA global temp: {len([o for o in observations if o['indicator_id'] == 'NOAA_TEMP_GLOBAL'])} obs")

        except Exception as e:
            logger.error(f"  Error fetching NOAA temp: {e}")

    # Land temperature
    if "NOAA_TEMP_LAND" in indicators:
        try:
            url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/signal topology/globe/land/1/0/1880-2024.csv"

            df = pd.read_csv(url, skiprows=4, names=['date', 'value'])

            for _, row in df.iterrows():
                try:
                    date_str = str(int(row['date']))
                    year = int(date_str[:4])
                    month = int(date_str[4:])
                    value = float(row['value'])
                except (ValueError, TypeError):
                    continue

                obs_date = date(year, month, 15)
                if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                    observations.append({
                        "indicator_id": "NOAA_TEMP_LAND",
                        "observed_at": obs_date,
                        "value": value,
                        "source": "noaa_ncei",
                        "domain": "climate",
                    })

            logger.info(f"  Fetched NOAA land temp")

        except Exception as e:
            logger.error(f"  Error fetching NOAA land temp: {e}")

    # Ocean temperature
    if "NOAA_TEMP_OCEAN" in indicators:
        try:
            url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/signal topology/globe/ocean/1/0/1880-2024.csv"

            df = pd.read_csv(url, skiprows=4, names=['date', 'value'])

            for _, row in df.iterrows():
                try:
                    date_str = str(int(row['date']))
                    year = int(date_str[:4])
                    month = int(date_str[4:])
                    value = float(row['value'])
                except (ValueError, TypeError):
                    continue

                obs_date = date(year, month, 15)
                if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                    observations.append({
                        "indicator_id": "NOAA_TEMP_OCEAN",
                        "observed_at": obs_date,
                        "value": value,
                        "source": "noaa_ncei",
                        "domain": "climate",
                    })

            logger.info(f"  Fetched NOAA ocean temp")

        except Exception as e:
            logger.error(f"  Error fetching NOAA ocean temp: {e}")

    return observations


# =============================================================================
# ESRL - Solar and Volcanic
# =============================================================================

def _fetch_esrl(indicators: List[str], start_date: str, end_date: str) -> List[Dict]:
    """Fetch solar and volcanic data from NOAA ESRL."""
    observations = []

    # Sunspot numbers (SILSO)
    if "SUNSPOT_NUMBER" in indicators:
        try:
            url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"

            df = pd.read_csv(url, sep=';', header=None, names=[
                'year', 'month', 'decimal', 'sunspot', 'std', 'obs', 'provisional'
            ])

            for _, row in df.iterrows():
                try:
                    year = int(row['year'])
                    month = int(row['month'])
                    value = float(row['sunspot'])
                except (ValueError, TypeError):
                    continue

                if value < 0:
                    continue

                obs_date = date(year, month, 15)
                if obs_date >= _parse_date(start_date) and obs_date <= _parse_date(end_date):
                    observations.append({
                        "indicator_id": "SUNSPOT_NUMBER",
                        "observed_at": obs_date,
                        "value": value,
                        "source": "silso",
                        "domain": "climate",
                    })

            logger.info(f"  Fetched sunspot numbers: {len([o for o in observations if o['indicator_id'] == 'SUNSPOT_NUMBER'])} obs")

        except Exception as e:
            logger.error(f"  Error fetching sunspots: {e}")

    return observations


# =============================================================================
# Utilities
# =============================================================================

def _parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    if isinstance(date_str, date):
        return date_str
    return datetime.strptime(date_str, "%Y-%m-%d").date()


if __name__ == "__main__":
    # Test fetch
    config = {
        "indicators": [
            "CO2_MONTHLY",
            "CO2_ANNUAL",
            "GISS_TEMP_GLOBAL",
            "NAO_INDEX",
            "ARCTIC_SEA_ICE_EXTENT",
            "SUNSPOT_NUMBER",
        ],
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
    }

    observations = fetch(config)

    print(f"\nFetched {len(observations)} observations")

    # Summary by indicator
    from collections import Counter
    counts = Counter(o['indicator_id'] for o in observations)
    for ind, count in sorted(counts.items()):
        print(f"  {ind}: {count}")
