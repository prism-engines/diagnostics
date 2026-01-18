#!/usr/bin/env python3
"""
PhysioNet Fetcher for PRISM Validation

Downloads and processes physiological signal topology data from PhysioNet.

Available Datasets:
    - MIT-BIH Arrhythmia Database (mitdb): 48 ECG recordings with beat annotations
    - MIT-BIH Normal Sinus Rhythm (nsrdb): 18 long-term ECG recordings

Ground Truth:
    - Beat annotations (Normal, PVC, PAC, etc.)
    - Rhythm annotations (Normal, Atrial Fibrillation, etc.)

PRISM Tests:
    - Can PRISM detect arrhythmia regimes from ECG signal topology?
    - Does entropy increase during irregular rhythms?
    - Can Hurst exponent distinguish normal vs abnormal?

References:
    Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
    IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).

    Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet:
    Components of a New Research Resource for Complex Physiologic Signals.
    Circulation 101(23):e215-e220 (2000).

Usage:
    pip install wfdb
    python fetchers/physionet_fetcher.py --dataset mitdb --records 10
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl


# Beat annotation codes (from MIT-BIH)
BEAT_ANNOTATIONS = {
    'N': 'Normal',
    'L': 'Left bundle branch block',
    'R': 'Right bundle branch block',
    'A': 'Atrial premature',
    'a': 'Aberrated atrial premature',
    'J': 'Nodal (junctional) premature',
    'S': 'Supraventricular premature',
    'V': 'Premature ventricular contraction',
    'F': 'Fusion of ventricular and normal',
    'e': 'Atrial escape',
    'j': 'Nodal (junctional) escape',
    'E': 'Ventricular escape',
    '/': 'Paced',
    'f': 'Fusion of paced and normal',
    'Q': 'Unclassifiable',
}

# Regime classification
NORMAL_BEATS = {'N', 'L', 'R'}
ARRHYTHMIA_BEATS = {'A', 'a', 'J', 'S', 'V', 'F', 'E'}


def fetch_mitdb(output_dir: Path, n_records: int = 10, segment_seconds: int = 30):
    """
    Fetch MIT-BIH Arrhythmia Database records.

    Args:
        output_dir: Output directory
        n_records: Number of records to fetch
        segment_seconds: Segment length for analysis
    """
    try:
        import wfdb
    except ImportError:
        print("Error: wfdb package not installed. Run: pip install wfdb")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    (output_dir / "vector").mkdir(exist_ok=True)

    print("Fetching MIT-BIH Arrhythmia Database...")
    print(f"Records to fetch: {n_records}")
    print(f"Segment length: {segment_seconds} seconds")
    print()

    # Get list of records
    record_list = wfdb.get_record_list('mitdb')
    print(f"Available records: {len(record_list)}")

    # Limit records
    record_list = record_list[:n_records]

    obs_rows = []
    ind_rows = []
    base_date = datetime(2020, 1, 1)

    for record_name in record_list:
        print(f"  Processing record {record_name}...")

        try:
            # Download and read record
            record = wfdb.rdrecord(record_name, pn_dir='mitdb')
            annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')

            # Get signal (use first channel - MLII)
            signal = record.p_signal[:, 0]
            fs = record.fs  # Sampling frequency (360 Hz)

            # Get annotations
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol

            # Segment the signal
            segment_samples = int(segment_seconds * fs)
            n_segments = len(signal) // segment_samples

            for seg_idx in range(min(n_segments, 20)):  # Limit segments per record
                start = seg_idx * segment_samples
                end = start + segment_samples

                segment = signal[start:end]

                # Count beats in this segment
                seg_ann_mask = (ann_samples >= start) & (ann_samples < end)
                seg_symbols = [ann_symbols[i] for i in range(len(ann_symbols)) if seg_ann_mask[i]]

                # Classify regime
                n_normal = sum(1 for s in seg_symbols if s in NORMAL_BEATS)
                n_arrhythmia = sum(1 for s in seg_symbols if s in ARRHYTHMIA_BEATS)
                total_beats = n_normal + n_arrhythmia

                if total_beats == 0:
                    regime = "unknown"
                    arrhythmia_ratio = 0.0
                else:
                    arrhythmia_ratio = n_arrhythmia / total_beats
                    if arrhythmia_ratio < 0.1:
                        regime = "normal"
                    elif arrhythmia_ratio < 0.3:
                        regime = "mild_arrhythmia"
                    else:
                        regime = "severe_arrhythmia"

                indicator_id = f"mitdb_{record_name}_seg{seg_idx}"

                # Downsample signal for PRISM (every 10th sample)
                downsampled = segment[::10]

                # Create observations
                for t, value in enumerate(downsampled):
                    obs_rows.append({
                        "indicator_id": indicator_id,
                        "obs_date": base_date + timedelta(seconds=t * 10 / fs),
                        "value": float(value),
                    })

                # Create indicator metadata
                ind_rows.append({
                    "indicator_id": indicator_id,
                    "record": record_name,
                    "segment": seg_idx,
                    "regime": regime,
                    "n_normal_beats": n_normal,
                    "n_arrhythmia_beats": n_arrhythmia,
                    "arrhythmia_ratio": arrhythmia_ratio,
                    "sampling_freq": fs,
                    "n_points": len(downsampled),
                })

        except Exception as e:
            print(f"    Error processing {record_name}: {e}")
            continue

    if not obs_rows:
        print("No data extracted!")
        return

    # Save to parquet
    obs_df = pl.DataFrame(obs_rows)
    ind_df = pl.DataFrame(ind_rows)

    obs_df.write_parquet(output_dir / "raw" / "observations.parquet")
    ind_df.write_parquet(output_dir / "raw" / "indicators.parquet")

    # Create cohorts
    cohorts = pl.DataFrame([{
        "cohort_id": "physionet_mitdb",
        "name": "MIT-BIH Arrhythmia Database",
        "description": "ECG recordings with arrhythmia annotations"
    }])
    cohorts.write_parquet(output_dir / "config" / "cohorts.parquet")

    cohort_members = pl.DataFrame([
        {"cohort_id": "physionet_mitdb", "indicator_id": ind_id}
        for ind_id in ind_df["indicator_id"].to_list()
    ])
    cohort_members.write_parquet(output_dir / "config" / "cohort_members.parquet")

    print()
    print("=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"Observations: {len(obs_df)}")
    print(f"Indicators: {len(ind_df)}")
    print()

    print("Regime summary:")
    print(ind_df.group_by("regime").len())
    print()

    print("Sample indicators:")
    print(ind_df.select(["indicator_id", "regime", "arrhythmia_ratio", "n_points"]).head(10))


def main():
    parser = argparse.ArgumentParser(description="Fetch PhysioNet data")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["mitdb"],
        default="mitdb",
        help="Dataset to fetch"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--records", "-n",
        type=int,
        default=10,
        help="Number of records to fetch"
    )
    parser.add_argument(
        "--segment-seconds", "-s",
        type=int,
        default=30,
        help="Segment length in seconds"
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(f"data/physionet_{args.dataset}")

    print("=" * 60)
    print(f"PhysioNet Fetcher: {args.dataset}")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print()

    if args.dataset == "mitdb":
        fetch_mitdb(output_dir, args.records, args.segment_seconds)


if __name__ == "__main__":
    main()
