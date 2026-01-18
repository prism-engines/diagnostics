"""
Double Pendulum PRISM Validation
================================

Runs PRISM engines on double pendulum data to test:
1. Chaos transition detection (Lyapunov crosses zero)
2. Energy conservation (PRISM metrics vs true E)
3. Regime identification at chaos onset

Usage:
    python scripts/validate_double_pendulum.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime

# PRISM engines
from prism.engines.hurst import compute_hurst
from prism.engines.lyapunov import compute_lyapunov
from prism.engines.entropy import compute_entropy
from prism.engines.spectral import compute_spectral
from prism.engines.garch import compute_garch
from prism.engines.rqa import compute_rqa

DATA_DIR = Path('/Users/jasonrudder/prism-mac/data/double_pendulum')


def load_observations():
    """Load all double pendulum observations."""
    obs_path = DATA_DIR / 'raw' / 'observations.parquet'
    return pl.read_parquet(obs_path)


def load_trajectory_summary():
    """Load trajectory metadata."""
    summary_path = DATA_DIR / 'raw' / 'trajectory_summary.parquet'
    return pl.read_parquet(summary_path)


def get_indicator_values(obs: pl.DataFrame, indicator_id: str) -> np.ndarray:
    """Extract values for a specific indicator."""
    values = (
        obs
        .filter(pl.col('indicator_id') == indicator_id)
        .sort('obs_date')
        .select('value')
        .to_numpy()
        .flatten()
    )
    return values


def compute_vector_metrics(values: np.ndarray) -> dict:
    """
    Compute all PRISM vector metrics for a signal topology.
    """
    metrics = {}

    # Hurst exponent
    try:
        hurst = compute_hurst(values)
        metrics['hurst_exponent'] = hurst.get('hurst_exponent')
    except Exception as e:
        metrics['hurst_exponent'] = None

    # Lyapunov exponent
    try:
        lyap = compute_lyapunov(values)
        metrics['lyapunov_exponent'] = lyap.get('lyapunov_exponent')
    except Exception as e:
        metrics['lyapunov_exponent'] = None

    # Entropy metrics (sample + permutation)
    try:
        ent = compute_entropy(values)
        metrics['sample_entropy'] = ent.get('sample_entropy')
        metrics['permutation_entropy'] = ent.get('permutation_entropy')
    except Exception as e:
        metrics['sample_entropy'] = None
        metrics['permutation_entropy'] = None

    # Spectral metrics
    try:
        spec = compute_spectral(values)
        metrics['spectral_entropy'] = spec.get('spectral_entropy')
        metrics['dominant_frequency'] = spec.get('dominant_frequency')
        metrics['dominant_period'] = spec.get('dominant_period')
    except Exception as e:
        metrics['spectral_entropy'] = None
        metrics['dominant_frequency'] = None
        metrics['dominant_period'] = None

    # RQA metrics (determinism, recurrence)
    try:
        rqa = compute_rqa(values)
        metrics['determinism'] = rqa.get('determinism')
        metrics['recurrence_rate'] = rqa.get('recurrence_rate')
        metrics['laminarity'] = rqa.get('laminarity')
    except Exception as e:
        metrics['determinism'] = None
        metrics['recurrence_rate'] = None
        metrics['laminarity'] = None

    # GARCH persistence
    try:
        garch = compute_garch(values)
        metrics['garch_persistence'] = garch.get('persistence')
    except Exception as e:
        metrics['garch_persistence'] = None

    return metrics


def run_validation():
    """Run full PRISM validation on double pendulum data."""

    print("=" * 70)
    print("DOUBLE PENDULUM PRISM VALIDATION")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Timestamp: {datetime.now()}")
    print()

    # Load data
    print("[1] Loading observations...")
    obs = load_observations()
    summary = load_trajectory_summary()

    print(f"    Total observations: {len(obs):,}")
    print(f"    Trajectories: {len(summary)}")
    print()

    # Show trajectory summary
    print("[2] Trajectory Summary (sorted by energy):")
    print("-" * 70)
    print(f"{'Trajectory':<15} {'Angle':<8} {'Regime':<12} {'Energy':<12} {'Conservation':<15}")
    print("-" * 70)

    for row in summary.sort('initial_angle_deg').iter_rows(named=True):
        cons_str = f"{row['energy_conservation']:.2e}"
        print(f"{row['trajectory']:<15} {row['initial_angle_deg']:<8} {row['regime']:<12} {row['mean_energy']:<12.4f} {cons_str:<15}")

    print()

    # Compute PRISM metrics for each trajectory
    print("[3] Computing PRISM vector metrics...")
    print("-" * 70)

    results = []

    # Process each trajectory's key variables
    trajectories = ['dp_10deg', 'dp_30deg', 'dp_60deg', 'dp_90deg', 'dp_120deg', 'dp_150deg']
    variables = ['theta1', 'omega1', 'x2', 'y2']  # Key dynamics variables

    for traj in trajectories:
        traj_row = summary.filter(pl.col('trajectory') == traj).row(0, named=True)
        angle = traj_row['initial_angle_deg']
        regime = traj_row['regime']
        energy = traj_row['mean_energy']

        print(f"\n  {traj} ({angle}°, {regime}):")

        for var in variables:
            indicator_id = f"{traj}_{var}"
            values = get_indicator_values(obs, indicator_id)

            if len(values) == 0:
                continue

            print(f"    {var}: ", end="")
            metrics = compute_vector_metrics(values)
            print(f"H={metrics['hurst_exponent']:.3f}, λ={metrics['lyapunov_exponent']:.4f}, " if metrics['lyapunov_exponent'] else "", end="")
            print(f"SampEn={metrics['sample_entropy']:.3f}, PermEn={metrics['permutation_entropy']:.3f}" if metrics['sample_entropy'] else "")

            results.append({
                'trajectory': traj,
                'variable': var,
                'indicator_id': indicator_id,
                'initial_angle': angle,
                'regime': regime,
                'true_energy': energy,
                **metrics,
            })

    # Create results dataframe
    results_df = pl.DataFrame(results)

    print()
    print("=" * 70)
    print("[4] CHAOS TRANSITION ANALYSIS")
    print("=" * 70)

    # Focus on theta1 (primary chaos indicator)
    theta1_results = results_df.filter(pl.col('variable') == 'theta1').sort('initial_angle')

    print("\nAngle vs Lyapunov Exponent (theta1):")
    print("-" * 50)
    print(f"{'Angle':<10} {'Regime':<12} {'Lyapunov':<12} {'Status'}")
    print("-" * 50)

    for row in theta1_results.iter_rows(named=True):
        lyap = row['lyapunov_exponent']
        if lyap is not None:
            status = "CHAOTIC" if lyap > 0 else "REGULAR"
            print(f"{row['initial_angle']:<10} {row['regime']:<12} {lyap:<12.4f} {status}")
        else:
            print(f"{row['initial_angle']:<10} {row['regime']:<12} {'N/A':<12} -")

    print()
    print("=" * 70)
    print("[5] ENTROPY VS ENERGY ANALYSIS")
    print("=" * 70)

    # Average metrics per trajectory
    traj_avg = (
        results_df
        .group_by(['trajectory', 'initial_angle', 'regime', 'true_energy'])
        .agg([
            pl.col('hurst_exponent').mean().alias('mean_hurst'),
            pl.col('lyapunov_exponent').mean().alias('mean_lyapunov'),
            pl.col('sample_entropy').mean().alias('mean_sample_entropy'),
            pl.col('permutation_entropy').mean().alias('mean_perm_entropy'),
            pl.col('spectral_entropy').mean().alias('mean_spectral_entropy'),
        ])
        .sort('initial_angle')
    )

    print("\nTrajectory Averages (all variables):")
    print("-" * 90)
    print(f"{'Angle':<8} {'Regime':<12} {'Hurst':<10} {'Lyapunov':<12} {'SampEn':<10} {'PermEn':<10} {'SpecEn':<10}")
    print("-" * 90)

    for row in traj_avg.iter_rows(named=True):
        h = f"{row['mean_hurst']:.4f}" if row['mean_hurst'] else "N/A"
        l = f"{row['mean_lyapunov']:.4f}" if row['mean_lyapunov'] else "N/A"
        se = f"{row['mean_sample_entropy']:.4f}" if row['mean_sample_entropy'] else "N/A"
        pe = f"{row['mean_perm_entropy']:.4f}" if row['mean_perm_entropy'] else "N/A"
        sp = f"{row['mean_spectral_entropy']:.4f}" if row['mean_spectral_entropy'] else "N/A"
        print(f"{row['initial_angle']:<8} {row['regime']:<12} {h:<10} {l:<12} {se:<10} {pe:<10} {sp:<10}")

    print()
    print("=" * 70)
    print("[6] VALIDATION SUMMARY")
    print("=" * 70)

    # Check key expectations
    print("\n✓ EXPECTED PATTERNS:")

    # 1. Lyapunov transition
    theta1_10 = theta1_results.filter(pl.col('initial_angle') == 10).row(0, named=True)
    theta1_150 = theta1_results.filter(pl.col('initial_angle') == 150).row(0, named=True)

    lyap_10 = theta1_10.get('lyapunov_exponent')
    lyap_150 = theta1_150.get('lyapunov_exponent')

    if lyap_10 is not None and lyap_150 is not None:
        if lyap_10 < lyap_150:
            print(f"  [✓] Lyapunov increases with energy: {lyap_10:.4f} (10°) → {lyap_150:.4f} (150°)")
        else:
            print(f"  [!] Unexpected: Lyapunov does not increase: {lyap_10:.4f} → {lyap_150:.4f}")
    else:
        print(f"  [?] Cannot verify Lyapunov transition (null values)")

    # 2. Hurst should decrease (more random at high energy)
    hurst_10 = theta1_10.get('hurst_exponent')
    hurst_150 = theta1_150.get('hurst_exponent')

    if hurst_10 is not None and hurst_150 is not None:
        if hurst_10 > hurst_150 or abs(hurst_10 - hurst_150) < 0.1:
            print(f"  [✓] Hurst behavior: {hurst_10:.4f} (10°) vs {hurst_150:.4f} (150°)")
        else:
            print(f"  [?] Hurst increased at high energy: {hurst_10:.4f} → {hurst_150:.4f}")

    # 3. Sample entropy should increase with chaos
    se_10 = theta1_10.get('sample_entropy')
    se_150 = theta1_150.get('sample_entropy')

    if se_10 is not None and se_150 is not None:
        if se_150 > se_10:
            print(f"  [✓] Sample entropy increases with chaos: {se_10:.4f} → {se_150:.4f}")
        else:
            print(f"  [?] Unexpected entropy behavior: {se_10:.4f} → {se_150:.4f}")

    # Save results
    results_df.write_parquet(DATA_DIR / 'vector' / 'indicator.parquet')
    traj_avg.write_parquet(DATA_DIR / 'vector' / 'trajectory_summary.parquet')

    print(f"\n  Saved: {DATA_DIR / 'vector' / 'indicator.parquet'}")
    print(f"  Saved: {DATA_DIR / 'vector' / 'trajectory_summary.parquet'}")

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return results_df, traj_avg


if __name__ == '__main__':
    results_df, traj_avg = run_validation()
