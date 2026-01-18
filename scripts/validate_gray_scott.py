"""
Gray-Scott Reaction-Diffusion PRISM Validation
===============================================

Tests PRISM metrics against PDE simulation data from The Well dataset.

Ground Truth:
    Six distinct pattern regimes from Gray-Scott reaction-diffusion:
    - Gliders: Moving localized structures
    - Bubbles: Expanding circular patterns
    - Maze: Labyrinthine structures
    - Worms: Elongated traveling waves
    - Spirals: Rotating spiral waves
    - Spots: Stationary Turing patterns

Physics:
    ∂A/∂t = δ_A ΔA - AB² + f(1-A)
    ∂B/∂t = δ_B ΔB + AB² - (f+k)B

    Where f (feed rate) and k (kill rate) determine the regime.

PRISM Test:
    Can PRISM distinguish the 6 pattern regimes from spatially-averaged signal topology?

Data Source:
    PolymathicAI / The Well (NeurIPS 2024)
    https://github.com/PolymathicAI/the_well

Usage:
    python scripts/validate_gray_scott.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from scipy import stats

# PRISM engines
from prism.engines.hurst import compute_hurst
from prism.engines.lyapunov import compute_lyapunov
from prism.engines.entropy import compute_entropy
from prism.engines.spectral import compute_spectral

DATA_DIR = Path('/Users/jasonrudder/prism-mac/data/the_well')


def load_data():
    """Load Gray-Scott data in PRISM format."""
    obs = pl.read_parquet(DATA_DIR / 'raw' / 'observations.parquet')
    indicators = pl.read_parquet(DATA_DIR / 'raw' / 'indicators.parquet')
    return obs, indicators


def get_values(obs: pl.DataFrame, indicator_id: str) -> np.ndarray:
    """Extract values for indicator."""
    return (
        obs
        .filter(pl.col('indicator_id') == indicator_id)
        .sort('obs_date')
        .select('value')
        .to_numpy()
        .flatten()
    )


def compute_metrics(values: np.ndarray) -> dict:
    """Compute PRISM vector metrics."""
    metrics = {}

    try:
        h = compute_hurst(values)
        metrics['hurst'] = h.get('hurst_exponent')
    except Exception:
        metrics['hurst'] = None

    try:
        l = compute_lyapunov(values)
        metrics['lyapunov'] = l.get('lyapunov_exponent')
    except Exception:
        metrics['lyapunov'] = None

    try:
        e = compute_entropy(values)
        metrics['sample_entropy'] = e.get('sample_entropy')
        metrics['permutation_entropy'] = e.get('permutation_entropy')
    except Exception:
        metrics['sample_entropy'] = None
        metrics['permutation_entropy'] = None

    try:
        s = compute_spectral(values)
        metrics['spectral_entropy'] = s.get('spectral_entropy')
        metrics['dominant_frequency'] = s.get('dominant_frequency')
    except Exception:
        metrics['spectral_entropy'] = None
        metrics['dominant_frequency'] = None

    return metrics


def run_validation():
    """Run full validation."""
    print("=" * 80)
    print("GRAY-SCOTT REACTION-DIFFUSION PRISM VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()

    # Load data
    print("[1] Loading data...")
    obs, indicators = load_data()
    print(f"    Observations: {len(obs):,}")
    print(f"    Indicators: {len(indicators)}")
    print()

    # Compute PRISM metrics
    print("=" * 80)
    print("[2] Computing PRISM metrics...")
    print("=" * 80)

    results = []
    for row in indicators.iter_rows(named=True):
        indicator_id = row['indicator_id']
        values = get_values(obs, indicator_id)

        if len(values) < 20:
            continue

        metrics = compute_metrics(values)
        metrics['indicator_id'] = indicator_id
        metrics['regime'] = row['regime']
        metrics['species'] = row['species']
        metrics['trajectory'] = row['trajectory']
        results.append(metrics)

        print(f"  {indicator_id}: n={len(values)}")

    df = pl.DataFrame(results)
    print(f"\nComputed metrics for {len(df)} indicators")

    # === TEST: Can PRISM distinguish regimes? ===
    print()
    print("=" * 80)
    print("[3] TEST: Regime Discrimination")
    print("=" * 80)
    print("Question: Do PRISM metrics differ by pattern regime?")
    print()

    # Group by regime
    regime_stats = df.group_by('regime').agg([
        pl.col('hurst').mean().alias('mean_hurst'),
        pl.col('sample_entropy').mean().alias('mean_samp_en'),
        pl.col('spectral_entropy').mean().alias('mean_spec_en'),
        pl.col('lyapunov').mean().alias('mean_lyap'),
        pl.len().alias('count'),
    ])

    print("Mean metrics by regime:")
    print("-" * 80)
    print(regime_stats.sort('regime'))

    # ANOVA test: do regimes differ?
    print()
    print("One-way ANOVA (do regimes differ?):")
    print("-" * 60)

    for metric in ['hurst', 'sample_entropy', 'spectral_entropy']:
        # Get values by regime
        regime_values = {}
        for regime in df['regime'].unique().to_list():
            vals = df.filter(pl.col('regime') == regime)[metric].drop_nulls().to_list()
            if vals:
                regime_values[regime] = vals

        if len(regime_values) >= 2:
            groups = list(regime_values.values())
            if all(len(g) >= 2 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                sig = "**" if p_val < 0.05 else ""
                print(f"  {metric:20s}: F = {f_stat:.3f}, p = {p_val:.4f} {sig}")

    # === Species comparison ===
    print()
    print("=" * 80)
    print("[4] Species A vs B Comparison")
    print("=" * 80)

    species_stats = df.group_by('species').agg([
        pl.col('hurst').mean().alias('mean_hurst'),
        pl.col('sample_entropy').mean().alias('mean_samp_en'),
        pl.len().alias('count'),
    ])
    print(species_stats)

    # === SUMMARY ===
    print()
    print("=" * 80)
    print("[5] VALIDATION SUMMARY")
    print("=" * 80)

    print("""
Gray-Scott Reaction-Diffusion Validation:

GROUND TRUTH:
    - 6 distinct pattern regimes from (f, k) parameter combinations
    - Gliders, Bubbles, Maze, Worms, Spirals, Spots
    - Each regime has characteristic spatiotemporal dynamics

PRISM HYPOTHESIS:
    - Different regimes should have different dynamical signatures
    - Stationary patterns (Spots): lower entropy, higher Hurst
    - Traveling patterns (Gliders, Worms): higher entropy, oscillatory

KEY METRICS:
    - Spectral Entropy: Captures frequency content of dynamics
    - Sample Entropy: Captures unpredictability
    - Hurst: Captures persistence/anti-persistence
""")

    # Save results
    df.write_parquet(DATA_DIR / 'vector' / 'indicator.parquet')
    print(f"Saved: {DATA_DIR / 'vector' / 'indicator.parquet'}")


if __name__ == '__main__':
    run_validation()
