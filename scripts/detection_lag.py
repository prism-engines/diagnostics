#!/usr/bin/env python
"""
PRISM Detection Lag Analysis
============================

Measures how quickly PRISM detects the known regime change in pendulum_regime.

Ground Truth:
- Phase 1 (t=0-25): Stable oscillation
- Phase 2 (t=25-50): Chaotic dynamics
- Transition: t=25.0 (row 2500, obs_date ~2026-11-05)

Usage:
    python scripts/detection_lag.py
"""

import polars as pl
import numpy as np
from datetime import date, timedelta

from prism.db.parquet_store import get_parquet_path


def analyze_prism_lag(domain: str = 'pendulum_regime',
                      event_row: int = 2500,
                      threshold_sigma: float = 3.0):
    """
    Analyze detection lag for PRISM signals.

    Args:
        domain: PRISM domain to analyze
        event_row: Row number of ground truth event (2500 for pendulum_regime)
        threshold_sigma: Number of standard deviations for detection threshold
    """
    print("=" * 70)
    print("PRISM DETECTION LAG ANALYSIS")
    print("=" * 70)
    print(f"\nDomain: {domain}")
    print(f"Ground truth event: row {event_row}")
    print(f"Detection threshold: μ + {threshold_sigma}σ")

    # Load vector results
    df = pl.read_parquet(get_parquet_path('vector', 'indicator', domain=domain))

    # Convert obs_date to row number (days since 2020-01-01)
    base_date = date(2020, 1, 1)
    df = df.with_columns([
        ((pl.col('obs_date') - pl.lit(base_date)).dt.total_days()).alias('row_num')
    ])

    # Get unique indicators
    indicators = df['indicator_id'].unique().sort().to_list()
    print(f"\nIndicators: {indicators}")

    # Key signals to test for regime detection
    detection_signals = [
        # Break detector signals
        ('break_detector', 'break_n'),
        ('break_detector', 'break_rate'),

        # Discontinuity signals
        ('dirac', 'dirac_n_impulses'),
        ('dirac', 'dirac_total_energy'),
        ('dirac', 'dirac_mean_magnitude'),
        ('heaviside', 'heaviside_n_steps'),
        ('heaviside', 'heaviside_mean_magnitude'),

        # Entropy (should increase in chaotic phase)
        ('entropy', 'sample_entropy'),
        ('entropy', 'permutation_entropy'),

        # Lyapunov (positive = chaotic)
        ('lyapunov', 'lyapunov_exponent'),
        ('lyapunov', 'is_chaotic'),

        # RQA (may change structure)
        ('rqa', 'determinism'),
        ('rqa', 'laminarity'),
        ('rqa', 'entropy'),

        # Volatility
        ('realized_vol', 'realized_vol'),
        ('garch', 'unconditional_vol'),
    ]

    results = []

    print("\n" + "-" * 70)
    print("DETECTION ANALYSIS")
    print("-" * 70)

    for indicator in indicators:
        print(f"\n[{indicator}]")
        df_ind = df.filter(pl.col('indicator_id') == indicator)

        for engine, metric in detection_signals:
            # Get this metric's signal topology
            metric_df = df_ind.filter(
                (pl.col('engine') == engine) &
                (pl.col('metric_name') == metric)
            ).sort('row_num')

            if len(metric_df) == 0:
                continue

            # Get values and row numbers
            rows = metric_df['row_num'].to_numpy()
            values = metric_df['metric_value'].to_numpy()

            # Baseline: before event
            baseline_mask = rows < event_row
            if not np.any(baseline_mask):
                continue

            baseline_values = values[baseline_mask]
            mu = np.nanmean(baseline_values)
            sigma = np.nanstd(baseline_values)

            if sigma == 0 or np.isnan(sigma):
                continue

            # Threshold for detection
            threshold = mu + (threshold_sigma * sigma)

            # Find first detection after event
            post_event_mask = rows >= event_row
            post_rows = rows[post_event_mask]
            post_values = values[post_event_mask]

            # Check for exceedance (either direction depending on signal)
            # For most signals, we look for increase
            # But for determinism, we might look for decrease
            if metric in ['determinism', 'laminarity']:
                # Look for decrease
                detections = post_rows[post_values < (mu - threshold_sigma * sigma)]
            else:
                # Look for increase
                detections = post_rows[post_values > threshold]

            if len(detections) > 0:
                first_detection = detections[0]
                lag_rows = first_detection - event_row
                lag_time = lag_rows * 0.01  # dt = 0.01 time units

                result = {
                    'indicator': indicator,
                    'engine': engine,
                    'metric': metric,
                    'baseline_mean': mu,
                    'baseline_std': sigma,
                    'threshold': threshold,
                    'first_detection_row': first_detection,
                    'lag_rows': lag_rows,
                    'lag_time': lag_time,
                    'detected': True
                }
                results.append(result)

                if lag_rows <= 100:  # Only print early detections
                    print(f"  {engine}/{metric}: detected at row {first_detection} "
                          f"(lag: {lag_rows} rows = {lag_time:.2f}s)")
            else:
                results.append({
                    'indicator': indicator,
                    'engine': engine,
                    'metric': metric,
                    'baseline_mean': mu,
                    'baseline_std': sigma,
                    'threshold': threshold,
                    'first_detection_row': None,
                    'lag_rows': None,
                    'lag_time': None,
                    'detected': False
                })

    # Convert to DataFrame
    results_df = pl.DataFrame(results)

    # Summary
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)

    detected = results_df.filter(pl.col('detected'))

    if len(detected) > 0:
        # Best detections (lowest lag)
        best = detected.sort('lag_rows').head(10)

        print("\nTop 10 Fastest Detections:")
        print("-" * 70)
        print(f"{'Indicator':<15} {'Engine':<15} {'Metric':<25} {'Lag (rows)':<10} {'Lag (t)':<10}")
        print("-" * 70)

        for row in best.iter_rows(named=True):
            print(f"{row['indicator']:<15} {row['engine']:<15} {row['metric']:<25} "
                  f"{row['lag_rows']:<10} {row['lag_time']:.4f}s")

        # Average lag by engine
        print("\nAverage Detection Lag by Engine:")
        print("-" * 40)
        engine_summary = detected.group_by('engine').agg([
            pl.col('lag_rows').mean().alias('mean_lag_rows'),
            pl.col('lag_time').mean().alias('mean_lag_time'),
            pl.col('detected').sum().alias('n_detections')
        ]).sort('mean_lag_rows')

        for row in engine_summary.iter_rows(named=True):
            print(f"  {row['engine']:<20} {row['mean_lag_rows']:>8.1f} rows  "
                  f"({row['mean_lag_time']:.4f}s)  [n={row['n_detections']}]")
    else:
        print("\nNo detections found with current threshold.")

    # Not detected
    not_detected = results_df.filter(~pl.col('detected'))
    if len(not_detected) > 0:
        print(f"\nNot detected ({threshold_sigma}σ threshold): {len(not_detected)} signals")

    print("\n" + "=" * 70)
    print(f"Total signals tested: {len(results_df)}")
    print(f"Detected: {len(detected)} ({100*len(detected)/len(results_df):.1f}%)")
    print("=" * 70)

    return results_df


if __name__ == '__main__':
    results = analyze_prism_lag()
