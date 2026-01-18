"""
TEP PRISM FAULT ONSET DETECTION
================================
The right test: WHEN does the system change, not WHICH fault.

PRISM detects regime transitions via:
- Divergence spikes (field singularity)
- Gradient acceleration (decoupling)
- Mode affinity drops (behavioral shift)

Usage:
    python -m prism.assessments.tep_onset_detection --domain cheme
"""

import argparse
import polars as pl
import numpy as np
from datetime import timedelta
from pathlib import Path
import os

from prism.db.parquet_store import get_parquet_path
from prism.utils.domain import require_domain


# ============================================================================
# 1. IDENTIFY FAULT ONSET WINDOWS
# ============================================================================

def find_fault_onsets(obs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Find exact timestamps where fault_code transitions from 0 to non-zero.
    These are the regime change moments PRISM should detect.
    """
    fault_df = obs_df.filter(pl.col('indicator_id') == 'TEP_FAULT').select([
        'obs_date',
        pl.col('value').alias('fault_code')
    ]).group_by('obs_date').agg(
        pl.col('fault_code').mode().first()
    ).sort('obs_date')

    return fault_df.with_columns([
        pl.col('fault_code').shift(1).alias('prev_fault'),
    ]).filter(
        (pl.col('prev_fault') == 0) & (pl.col('fault_code') > 0)
    ).select([
        'obs_date',
        'fault_code',
    ]).rename({'obs_date': 'onset_date', 'fault_code': 'fault_id'})


# ============================================================================
# 2. EXTRACT PRISM FEATURES AROUND ONSET
# ============================================================================

def extract_onset_windows(
    field_df: pl.DataFrame,
    onset_date,
    window_before: int = 7,  # days before onset
    window_after: int = 3,   # days after onset
) -> pl.DataFrame:
    """
    Extract Laplace field features in window around fault onset.

    Returns features for:
    - PRE-ONSET: baseline behavior (should be stable)
    - ONSET: transition moment (should show divergence spike)
    - POST-ONSET: fault regime (should show new stable state or chaos)
    """
    onset = onset_date
    if isinstance(onset, str):
        onset = pl.lit(onset).str.to_date().to_physical()

    before = onset - timedelta(days=window_before)
    after = onset + timedelta(days=window_after)

    window_df = field_df.filter(
        (pl.col('window_end') >= before) &
        (pl.col('window_end') <= after)
    )

    return window_df.with_columns([
        pl.when(pl.col('window_end') < onset)
          .then(pl.lit('PRE'))
          .when(pl.col('window_end') == onset)
          .then(pl.lit('ONSET'))
          .otherwise(pl.lit('POST'))
          .alias('phase')
    ])


# ============================================================================
# 3. COMPUTE ONSET DETECTION METRICS
# ============================================================================

def compute_onset_metrics(window_df: pl.DataFrame) -> dict:
    """
    Compute PRISM detection metrics for a single fault onset.

    Key signals:
    - Divergence spike magnitude
    - Gradient acceleration
    - Cross-indicator divergence correlation
    """
    pre = window_df.filter(pl.col('phase') == 'PRE')
    onset = window_df.filter(pl.col('phase') == 'ONSET')
    post = window_df.filter(pl.col('phase') == 'POST')

    metrics = {}

    # Baseline statistics (PRE)
    if len(pre) > 0:
        metrics['pre_divergence_mean'] = pre['divergence'].mean()
        metrics['pre_divergence_std'] = pre['divergence'].std()
        metrics['pre_gradient_mag_mean'] = pre['gradient_magnitude'].mean()

    # Onset statistics
    if len(onset) > 0:
        metrics['onset_divergence'] = onset['divergence'].mean()
        metrics['onset_gradient_mag'] = onset['gradient_magnitude'].mean()

    # Detection signal: Z-score of onset divergence vs baseline
    if 'pre_divergence_mean' in metrics and 'pre_divergence_std' in metrics:
        if metrics.get('pre_divergence_std', 0) > 0 and metrics.get('onset_divergence') is not None:
            metrics['divergence_zscore'] = (
                metrics['onset_divergence'] - metrics['pre_divergence_mean']
            ) / metrics['pre_divergence_std']
        else:
            metrics['divergence_zscore'] = 0

    # Gradient acceleration: did gradient magnitude spike?
    if 'pre_gradient_mag_mean' in metrics and 'onset_gradient_mag' in metrics:
        if metrics.get('onset_gradient_mag') is not None:
            metrics['gradient_spike_ratio'] = (
                metrics['onset_gradient_mag'] / (metrics['pre_gradient_mag_mean'] + 1e-10)
            )

    return metrics


# ============================================================================
# 4. EARLY WARNING DETECTION
# ============================================================================

def detect_early_warning(
    field_df: pl.DataFrame,
    onset_date,
    lookback_days: int = 14,
    threshold_zscore: float = 2.0,
) -> dict:
    """
    Can PRISM detect the fault BEFORE onset?

    Look for divergence/gradient anomalies in the days leading up to onset.
    Returns earliest detection date and lead time.
    """
    onset = onset_date
    if isinstance(onset, str):
        from datetime import datetime
        onset = datetime.strptime(onset, '%Y-%m-%d').date()

    # Get baseline from early window
    baseline_start = onset - timedelta(days=lookback_days)
    baseline_end = onset - timedelta(days=7)

    baseline = field_df.filter(
        (pl.col('window_end') >= baseline_start) &
        (pl.col('window_end') <= baseline_end)
    )

    if len(baseline) == 0:
        return {'early_detection': False, 'lead_time_days': 0}

    baseline_div_mean = baseline['divergence'].mean()
    baseline_div_std = baseline['divergence'].std()

    if baseline_div_std is None or baseline_div_std == 0:
        return {'early_detection': False, 'lead_time_days': 0}

    # Scan forward looking for anomaly
    detection_window = field_df.filter(
        (pl.col('window_end') > baseline_end) &
        (pl.col('window_end') <= onset)
    ).sort('window_end')

    # Aggregate by date
    daily = detection_window.group_by('window_end').agg(
        pl.col('divergence').mean().alias('avg_div')
    ).sort('window_end')

    for row in daily.iter_rows(named=True):
        zscore = (row['avg_div'] - baseline_div_mean) / baseline_div_std
        if abs(zscore) > threshold_zscore:
            detection_date = row['window_end']
            lead_time = (onset - detection_date).days
            return {
                'early_detection': True,
                'detection_date': str(detection_date),
                'lead_time_days': lead_time,
                'detection_zscore': zscore,
            }

    return {'early_detection': False, 'lead_time_days': 0}


# ============================================================================
# 5. PER-INDICATOR ONSET ANALYSIS
# ============================================================================

def analyze_indicator_response(
    field_df: pl.DataFrame,
    onset_date,
    window_days: int = 3,
) -> pl.DataFrame:
    """
    Which indicators respond first/strongest to fault onset?

    Returns ranking of indicators by onset response magnitude.
    """
    onset = onset_date
    if isinstance(onset, str):
        from datetime import datetime
        onset = datetime.strptime(onset, '%Y-%m-%d').date()

    # Pre-onset baseline (3 days before)
    pre = field_df.filter(
        (pl.col('window_end') >= onset - timedelta(days=window_days)) &
        (pl.col('window_end') < onset)
    ).group_by('indicator_id').agg([
        pl.col('divergence').mean().alias('pre_div'),
        pl.col('gradient_magnitude').mean().alias('pre_grad'),
    ])

    # Onset window
    post = field_df.filter(
        (pl.col('window_end') >= onset) &
        (pl.col('window_end') <= onset + timedelta(days=window_days))
    ).group_by('indicator_id').agg([
        pl.col('divergence').mean().alias('onset_div'),
        pl.col('gradient_magnitude').mean().alias('onset_grad'),
    ])

    # Join and compute response
    response = pre.join(post, on='indicator_id', how='inner').with_columns([
        (pl.col('onset_div') - pl.col('pre_div')).abs().alias('div_change'),
        (pl.col('onset_grad') / (pl.col('pre_grad') + 1e-10)).alias('grad_ratio'),
    ]).sort('div_change', descending=True)

    return response


# ============================================================================
# 6. MAIN EVALUATION LOOP
# ============================================================================

def evaluate_fault_onset_detection(domain: str):
    """
    Run full onset detection evaluation across all fault transitions.
    """
    print("=" * 80)
    print("TEP PRISM FAULT ONSET DETECTION EVALUATION")
    print("=" * 80)

    # Load data
    field_path = get_parquet_path('vector', 'indicator_field', domain)
    obs_path = get_parquet_path('raw', 'observations', domain)

    print(f"\nLoading field data from {field_path}...")
    field_df = pl.read_parquet(field_path)

    # Filter to TEP
    field_df = field_df.filter(pl.col('indicator_id').str.starts_with('TEP_'))
    print(f"Field data: {len(field_df):,} rows")

    print(f"Loading observations from {obs_path}...")
    obs_df = pl.read_parquet(obs_path)

    # Find all fault onsets
    onsets = find_fault_onsets(obs_df)
    print(f"\nFound {len(onsets)} fault onset events")

    if len(onsets) == 0:
        print("No fault onsets found!")
        return None

    print("\nFault onsets:")
    for row in onsets.iter_rows(named=True):
        print(f"  IDV{int(row['fault_id']):02d}: {row['onset_date']}")

    results = []

    for row in onsets.iter_rows(named=True):
        onset_date = row['onset_date']
        fault_id = int(row['fault_id'])

        print(f"\n--- IDV{fault_id:02d} onset: {onset_date} ---")

        # Extract window around onset
        window = extract_onset_windows(field_df, onset_date)

        if len(window) == 0:
            print("  No data in window")
            continue

        # Compute detection metrics
        metrics = compute_onset_metrics(window)

        # Check for early warning
        early = detect_early_warning(field_df, onset_date)

        # Analyze per-indicator response
        response = analyze_indicator_response(field_df, onset_date)
        top_responders = response.head(5)['indicator_id'].to_list() if len(response) > 0 else []

        result = {
            'fault_id': fault_id,
            'onset_date': str(onset_date),
            'divergence_zscore': metrics.get('divergence_zscore', 0),
            'gradient_spike_ratio': metrics.get('gradient_spike_ratio', 0),
            'early_detection': early.get('early_detection', False),
            'lead_time_days': early.get('lead_time_days', 0),
            'top_responders': ','.join(top_responders[:3]),
        }
        results.append(result)

        # Print summary
        z = metrics.get('divergence_zscore', 'N/A')
        g = metrics.get('gradient_spike_ratio', 'N/A')
        print(f"  Divergence Z-score: {z:.2f}" if isinstance(z, (int, float)) else f"  Divergence Z-score: {z}")
        print(f"  Gradient spike ratio: {g:.2f}" if isinstance(g, (int, float)) else f"  Gradient spike ratio: {g}")
        print(f"  Early detection: {early.get('early_detection', False)}")
        if early.get('early_detection'):
            print(f"    Lead time: {early.get('lead_time_days', 0)} days")
        print(f"  Top responders: {', '.join(top_responders[:3])}")

    if len(results) == 0:
        print("No results generated")
        return None

    # Summary statistics
    results_df = pl.DataFrame(results)

    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)

    # Detection rate (divergence z-score > 2)
    detected = results_df.filter(pl.col('divergence_zscore').abs() > 2.0)
    detection_rate = len(detected) / len(results_df) * 100
    print(f"\nDetection rate (|z| > 2): {detection_rate:.1f}% ({len(detected)}/{len(results_df)})")

    # Early warning rate
    early_detected = results_df.filter(pl.col('early_detection') == True)
    early_rate = len(early_detected) / len(results_df) * 100
    print(f"Early warning rate: {early_rate:.1f}% ({len(early_detected)}/{len(results_df)})")

    if len(early_detected) > 0:
        avg_lead = early_detected['lead_time_days'].mean()
        print(f"Average lead time: {avg_lead:.1f} days")

    # Per-fault breakdown
    print("\nPer-fault detection:")
    for fault_id in sorted(results_df['fault_id'].unique().to_list()):
        fault_results = results_df.filter(pl.col('fault_id') == fault_id)
        z = fault_results['divergence_zscore'].mean()
        early = fault_results['early_detection'].sum()
        total = len(fault_results)
        status = "✓" if abs(z) > 2 else "△" if abs(z) > 1 else "✗"
        print(f"  {status} IDV{int(fault_id):02d}: z={z:+.2f}, early={early}/{total}")

    print("\n" + "=" * 80)
    print("ONSET DETECTION EVALUATION COMPLETE")
    print("=" * 80)

    return results_df


# ============================================================================
# 7. DEEP DIVE: SINGLE FAULT ANALYSIS
# ============================================================================

def deep_dive_fault(
    field_df: pl.DataFrame,
    fault_id: int,
    onset_date,
):
    """
    Detailed analysis of a single fault onset.

    Generates:
    - Divergence signal topology plot data
    - Per-indicator response ranking
    - Mode transition analysis
    """
    print(f"\n{'='*80}")
    print(f"DEEP DIVE: IDV{fault_id:02d} at {onset_date}")
    print(f"{'='*80}")

    onset = onset_date
    if isinstance(onset, str):
        from datetime import datetime
        onset = datetime.strptime(onset, '%Y-%m-%d').date()

    # 14-day window centered on onset
    window = field_df.filter(
        (pl.col('window_end') >= onset - timedelta(days=7)) &
        (pl.col('window_end') <= onset + timedelta(days=7))
    )

    if len(window) == 0:
        print("No data in window")
        return None

    # Aggregate divergence (all indicators)
    total_div = window.group_by('window_end').agg([
        pl.col('divergence').sum().alias('total_divergence'),
        pl.col('gradient_magnitude').mean().alias('avg_grad_mag'),
        pl.col('indicator_id').n_unique().alias('n_indicators'),
    ]).sort('window_end')

    print("\nTotal Divergence Timeline:")
    print("-" * 60)
    for row in total_div.iter_rows(named=True):
        date = row['window_end']
        div = row['total_divergence']
        marker = " <-- ONSET" if date == onset else ""
        bar_len = min(50, int(abs(div) / 100)) if div else 0
        bar = "█" * bar_len
        print(f"  {date}: {div:+12.2f} {bar}{marker}")

    # Top responding indicators
    response = analyze_indicator_response(field_df, onset_date)

    print("\nTop 10 Responding Indicators:")
    print("-" * 60)
    for i, row in enumerate(response.head(10).iter_rows(named=True)):
        ind = row['indicator_id']
        change = row['div_change']
        ratio = row['grad_ratio']
        print(f"  {i+1:2d}. {ind:20s}: Δdiv={change:+10.2f}, grad_ratio={ratio:.2f}x")

    return {
        'total_divergence': total_div,
        'indicator_response': response,
    }


def main():
    parser = argparse.ArgumentParser(
        description='TEP PRISM Fault Onset Detection Evaluation'
    )
    parser.add_argument('--domain', type=str, default=None,
                        help='Domain (default: prompts)')
    parser.add_argument('--deep-dive', type=int, default=None,
                        help='Deep dive into specific fault ID (e.g., 4)')

    args = parser.parse_args()

    domain = require_domain(args.domain, "Select domain for TEP onset detection")
    os.environ["PRISM_DOMAIN"] = domain
    print(f"Domain: {domain}")

    results = evaluate_fault_onset_detection(domain)

    if args.deep_dive and results is not None:
        field_path = get_parquet_path('vector', 'indicator_field', domain)
        field_df = pl.read_parquet(field_path)
        field_df = field_df.filter(pl.col('indicator_id').str.starts_with('TEP_'))

        fault_onset = results.filter(pl.col('fault_id') == args.deep_dive)
        if len(fault_onset) > 0:
            onset_date = fault_onset['onset_date'][0]
            deep_dive_fault(field_df, args.deep_dive, onset_date)


if __name__ == '__main__':
    main()
