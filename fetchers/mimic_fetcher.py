#!/usr/bin/env python3
"""
MIMIC-IV Fetcher for PRISM Validation

Downloads and processes ICU vitals data from MIMIC-IV for sepsis regime detection.

Available Datasets:
    - MIMIC-IV Demo (100 patients, open access)
    - MIMIC-IV Full (60K+ ICU stays, requires credentialed access)

PRISM Hypothesis:
    - Sepsis is a regime transition detectable by PRISM metrics
    - Vitals "decouple" before sepsis onset (geometry change)
    - Entropy and divergence increase before clinical diagnosis

Ground Truth:
    - Sepsis-3 criteria: SOFA >= 2 with suspected infection
    - Timestamps for regime transitions

References:
    Johnson, A., et al. (2023). MIMIC-IV, a freely accessible electronic
    health record dataset. Scientific Data, 10(1), 1.

    Singer, M., et al. (2016). The Third International Consensus Definitions
    for Sepsis and Septic Shock (Sepsis-3). JAMA, 315(8), 801-810.

Usage:
    python fetchers/mimic_fetcher.py --demo  # Open access demo (100 patients)
    python fetchers/mimic_fetcher.py --full  # Requires credentialed access
"""

import argparse
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# MIMIC-IV Demo URLs and Configuration
# =============================================================================

DEMO_URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip"

# Vital sign itemids from MIMIC-IV chartevents
VITAL_ITEMIDS = {
    # Heart rate
    220045: "heart_rate",
    # Blood pressure
    220050: "arterial_bp_systolic",
    220051: "arterial_bp_diastolic",
    220052: "arterial_bp_mean",
    220179: "non_invasive_bp_systolic",
    220180: "non_invasive_bp_diastolic",
    220181: "non_invasive_bp_mean",
    # Temperature
    223761: "temperature_f",
    223762: "temperature_c",
    # Respiratory
    220210: "respiratory_rate",
    # Oxygen
    220277: "spo2",
    # GCS components
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
}

# Mapping for regime classification
REGIME_LABELS = {
    "stable": 0,
    "deteriorating": 1,
    "septic": 2,
    "recovered": 3,
}


def download_mimic_demo(output_dir: Path) -> bool:
    """
    Download MIMIC-IV Demo dataset.

    Args:
        output_dir: Output directory for downloaded files

    Returns:
        True if successful, False otherwise
    """
    if not HAS_REQUESTS:
        print("Error: requests package not installed. Run: pip install requests")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MIMIC-IV Demo...")
    print(f"URL: {DEMO_URL}")
    print("Size: ~15 MB")
    print()

    try:
        response = requests.get(DEMO_URL, stream=True, timeout=120)
        response.raise_for_status()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as f:
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  Downloading: {pct}%", end="", flush=True)
            print()
            temp_path = f.name

        # Extract
        print("Extracting...")
        with zipfile.ZipFile(temp_path, 'r') as z:
            z.extractall(output_dir)

        # Clean up
        Path(temp_path).unlink()

        print("Download complete!")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def load_chartevents(mimic_dir: Path) -> pl.DataFrame:
    """
    Load chartevents (vitals) from MIMIC-IV Demo.

    Args:
        mimic_dir: Path to extracted MIMIC-IV demo

    Returns:
        DataFrame with vital sign measurements
    """
    # Find the chartevents file (may be gzipped)
    chartevents_path = mimic_dir / "mimic-iv-clinical-database-demo-2.2" / "icu" / "chartevents.csv.gz"

    if not chartevents_path.exists():
        chartevents_path = mimic_dir / "mimic-iv-clinical-database-demo-2.2" / "icu" / "chartevents.csv"

    if not chartevents_path.exists():
        # Try alternative path structure
        for p in mimic_dir.rglob("chartevents.csv*"):
            chartevents_path = p
            break

    if not chartevents_path.exists():
        raise FileNotFoundError(f"chartevents.csv not found in {mimic_dir}")

    print(f"Loading chartevents from {chartevents_path}...")

    # Read with Polars
    df = pl.read_csv(
        chartevents_path,
        columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        dtypes={"valuenum": pl.Float64}
    )

    # Filter to vital signs only
    vital_ids = list(VITAL_ITEMIDS.keys())
    df = df.filter(pl.col("itemid").is_in(vital_ids))

    # Add vital name using map
    vital_name_map = pl.DataFrame({
        "itemid": list(VITAL_ITEMIDS.keys()),
        "vital_name": list(VITAL_ITEMIDS.values())
    })
    df = df.join(vital_name_map, on="itemid", how="left")

    # Parse datetime
    df = df.with_columns(
        pl.col("charttime").str.to_datetime().alias("charttime")
    )

    return df


def load_sepsis_labels(mimic_dir: Path) -> pl.DataFrame:
    """
    Load sepsis labels from MIMIC-IV.

    Uses suspected infection time + SOFA >= 2 to define sepsis onset.
    For demo, we'll use a simplified approach based on available data.

    Args:
        mimic_dir: Path to extracted MIMIC-IV demo

    Returns:
        DataFrame with sepsis labels per stay
    """
    # In demo, we need to derive sepsis from available tables
    # This is a simplified version - full MIMIC-IV has sepsis3 derived table

    # Load diagnoses for sepsis codes (may be gzipped)
    diagnoses_path = mimic_dir / "mimic-iv-clinical-database-demo-2.2" / "hosp" / "diagnoses_icd.csv.gz"

    if not diagnoses_path.exists():
        diagnoses_path = mimic_dir / "mimic-iv-clinical-database-demo-2.2" / "hosp" / "diagnoses_icd.csv"

    if not diagnoses_path.exists():
        for p in mimic_dir.rglob("diagnoses_icd.csv*"):
            diagnoses_path = p
            break

    if diagnoses_path.exists():
        diagnoses = pl.read_csv(diagnoses_path)

        # Sepsis ICD codes (ICD-9 and ICD-10)
        sepsis_codes = [
            "99591", "99592",  # ICD-9 sepsis
            "A4150", "A4151", "A4152", "A4153", "A4154", "A419",  # ICD-10 sepsis
            "R6520", "R6521",  # ICD-10 severe sepsis
        ]

        sepsis_stays = diagnoses.filter(
            pl.col("icd_code").is_in(sepsis_codes)
        ).select(["subject_id", "hadm_id"]).unique()

        sepsis_stays = sepsis_stays.with_columns(
            pl.lit(True).alias("has_sepsis")
        )

        return sepsis_stays

    return pl.DataFrame({"subject_id": [], "hadm_id": [], "has_sepsis": []})


def create_prism_observations(
    vitals: pl.DataFrame,
    sepsis_labels: pl.DataFrame,
    window_hours: int = 6,
    segment_hours: int = 4,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Create PRISM-format observations from MIMIC vitals.

    Each ICU stay becomes multiple indicators (one per vital sign).
    Signal are segmented into windows for regime analysis.

    Args:
        vitals: DataFrame with vital measurements
        sepsis_labels: DataFrame with sepsis labels
        window_hours: Total window to analyze (hours)
        segment_hours: Segment length for PRISM indicators

    Returns:
        Tuple of (observations, indicators) DataFrames
    """
    obs_rows = []
    ind_rows = []

    # Get unique stays
    stays = vitals.select(["subject_id", "stay_id"]).unique()

    # Join with sepsis labels
    stays = stays.join(
        sepsis_labels.select(["subject_id", "has_sepsis"]),
        on="subject_id",
        how="left"
    ).with_columns(
        pl.col("has_sepsis").fill_null(False)
    )

    print(f"Processing {len(stays)} ICU stays...")

    for i, row in enumerate(stays.iter_rows(named=True)):
        if i % 20 == 0:
            print(f"  {i}/{len(stays)}...")

        stay_id = row["stay_id"]
        subject_id = row["subject_id"]
        has_sepsis = row["has_sepsis"]

        # Get vitals for this stay
        stay_vitals = vitals.filter(pl.col("stay_id") == stay_id)

        if len(stay_vitals) < 10:
            continue

        # Get time range
        min_time = stay_vitals["charttime"].min()
        max_time = stay_vitals["charttime"].max()

        # Process each vital sign
        for vital_name in stay_vitals["vital_name"].unique().to_list():
            vital_data = stay_vitals.filter(pl.col("vital_name") == vital_name)

            if len(vital_data) < 10:
                continue

            # Sort by time
            vital_data = vital_data.sort("charttime")

            # Remove outliers (basic cleaning)
            values = vital_data["valuenum"].drop_nulls()
            if len(values) < 10:
                continue

            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3*iqr, q3 + 3*iqr

            vital_data = vital_data.filter(
                (pl.col("valuenum") >= lower) & (pl.col("valuenum") <= upper)
            )

            if len(vital_data) < 10:
                continue

            # Create indicator
            indicator_id = f"mimic_{stay_id}_{vital_name}"

            # Determine regime
            if has_sepsis:
                regime = "septic"
            else:
                regime = "stable"

            # Create observations
            for j, obs_row in enumerate(vital_data.iter_rows(named=True)):
                obs_rows.append({
                    "indicator_id": indicator_id,
                    "obs_date": obs_row["charttime"],
                    "value": float(obs_row["valuenum"]),
                })

            # Create indicator metadata
            ind_rows.append({
                "indicator_id": indicator_id,
                "subject_id": subject_id,
                "stay_id": stay_id,
                "vital_name": vital_name,
                "regime": regime,
                "has_sepsis": has_sepsis,
                "n_points": len(vital_data),
            })

    observations = pl.DataFrame(obs_rows)
    indicators = pl.DataFrame(ind_rows)

    return observations, indicators


def fetch_mimic_demo(output_dir: Path):
    """
    Fetch and process MIMIC-IV Demo for PRISM validation.

    Args:
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    (output_dir / "vector").mkdir(exist_ok=True)

    # Download if not present
    demo_dir = output_dir / "mimic-iv-clinical-database-demo-2.2"
    if not demo_dir.exists():
        if not download_mimic_demo(output_dir):
            return

    # Load data
    print()
    print("Loading vitals...")
    vitals = load_chartevents(output_dir)
    print(f"  Loaded {len(vitals)} vital measurements")

    print()
    print("Loading sepsis labels...")
    sepsis = load_sepsis_labels(output_dir)
    print(f"  Found {len(sepsis)} sepsis cases")

    print()
    print("Creating PRISM observations...")
    observations, indicators = create_prism_observations(vitals, sepsis)

    # Save
    observations.write_parquet(output_dir / "raw" / "observations.parquet")
    indicators.write_parquet(output_dir / "raw" / "indicators.parquet")

    # Create cohorts
    cohorts = pl.DataFrame([{
        "cohort_id": "mimic_demo",
        "name": "MIMIC-IV Demo ICU Vitals",
        "description": "100-patient demo of ICU vital signs for sepsis detection"
    }])
    cohorts.write_parquet(output_dir / "config" / "cohorts.parquet")

    cohort_members = pl.DataFrame([
        {"cohort_id": "mimic_demo", "indicator_id": ind_id}
        for ind_id in indicators["indicator_id"].to_list()
    ])
    cohort_members.write_parquet(output_dir / "config" / "cohort_members.parquet")

    # Summary
    print()
    print("=" * 60)
    print("MIMIC-IV Demo Processing Complete")
    print("=" * 60)
    print(f"Observations: {len(observations)}")
    print(f"Indicators: {len(indicators)}")
    print()

    print("Regime summary:")
    print(indicators.group_by("regime").len())
    print()

    print("Vital signs:")
    print(indicators.group_by("vital_name").len())
    print()

    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fetch MIMIC-IV data for PRISM")
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Fetch MIMIC-IV Demo (100 patients, open access)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    args = parser.parse_args()

    if args.demo or True:  # Default to demo for safety
        output_dir = Path(args.output) if args.output else Path("data/mimic_demo")

        print("=" * 60)
        print("MIMIC-IV Demo Fetcher")
        print("=" * 60)
        print(f"Output: {output_dir}")
        print()
        print("Note: This uses the 100-patient demo dataset (open access).")
        print("For full MIMIC-IV, you need credentialed PhysioNet access.")
        print()

        fetch_mimic_demo(output_dir)
    else:
        print("For full MIMIC-IV access:")
        print("1. Complete CITI training: https://about.citiprogram.org/")
        print("2. Create PhysioNet account: https://physionet.org/")
        print("3. Sign Data Use Agreement")
        print("4. Download from: https://physionet.org/content/mimiciv/")


if __name__ == "__main__":
    main()
