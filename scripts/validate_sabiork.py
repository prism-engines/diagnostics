#!/usr/bin/env python3
"""
PRISM Validation: SABIO-RK Enzyme Kinetics

Tests whether PRISM can detect substrate saturation regimes in
Michaelis-Menten enzyme kinetics.

Ground Truth:
    v = Vmax × [S] / (Km + [S])

    - Linear regime ([S] << Km): v ≈ (Vmax/Km) × [S] (first-order)
    - Transition ([S] ≈ Km): v = Vmax/2 (half-max velocity)
    - Saturating ([S] >> Km): v ≈ Vmax (zero-order plateau)

Hypothesis:
    PRISM metrics should differentiate these regimes:
    - Linear: Higher Hurst (trending), higher entropy
    - Transition: Moderate values
    - Saturating: Lower Hurst (plateau), lower entropy

Reference:
    Wittig, U., et al. (2012). SABIO-RK—database for biochemical reaction kinetics.
    Nucleic Acids Research, 40(D1), D790-D798.
"""

import sys
from pathlib import Path

import polars as pl
import numpy as np
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.engines.hurst import compute_hurst
from prism.engines.spectral import compute_spectral

# Entropy functions
try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False


def compute_entropy(values: np.ndarray) -> dict:
    """Compute entropy metrics."""
    result = {}

    if HAS_ANTROPY and len(values) >= 30:
        try:
            result["sample_entropy"] = ant.sample_entropy(values, order=2)
        except:
            result["sample_entropy"] = np.nan

        try:
            result["permutation_entropy"] = ant.perm_entropy(values, order=3, normalize=True)
        except:
            result["permutation_entropy"] = np.nan
    else:
        result["sample_entropy"] = np.nan
        result["permutation_entropy"] = np.nan

    return result


def run_prism_engines(observations: pl.DataFrame) -> pl.DataFrame:
    """Run PRISM vector engines on observations."""

    results = []

    indicators = observations["indicator_id"].unique().to_list()
    print(f"Processing {len(indicators)} indicators...")

    for i, indicator_id in enumerate(indicators):
        if i % 10 == 0:
            print(f"  {i}/{len(indicators)}...")

        # Get signal topology
        ts = observations.filter(pl.col("indicator_id") == indicator_id)
        values = ts["value"].to_numpy()

        if len(values) < 20:
            continue

        row = {"indicator_id": indicator_id}

        # Hurst (using standalone function)
        try:
            hurst_result = compute_hurst(values)
            row["hurst"] = hurst_result.get("hurst_exponent")
        except:
            row["hurst"] = np.nan

        # Spectral (using standalone function)
        try:
            spectral_result = compute_spectral(values)
            row["spectral_entropy"] = spectral_result.get("spectral_entropy")
        except:
            row["spectral_entropy"] = np.nan

        # Entropy
        entropy = compute_entropy(values)
        row["sample_entropy"] = entropy.get("sample_entropy")
        row["permutation_entropy"] = entropy.get("permutation_entropy")

        results.append(row)

    return pl.DataFrame(results)


def main():
    data_dir = Path("data/sabiork")

    print("=" * 70)
    print("PRISM Validation: SABIO-RK Enzyme Kinetics")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    observations = pl.read_parquet(data_dir / "raw" / "observations.parquet")
    indicators = pl.read_parquet(data_dir / "raw" / "indicators.parquet")

    print(f"Observations: {len(observations)}")
    print(f"Indicators: {len(indicators)}")
    print()

    # Run PRISM
    print("Running PRISM engines...")
    metrics = run_prism_engines(observations)

    # Merge with indicator metadata
    results = metrics.join(indicators, on="indicator_id", how="inner")

    print(f"\nResults: {len(results)} indicators with metrics")
    print()

    # Save results
    results.write_parquet(data_dir / "vector" / "indicator.parquet")
    print(f"Saved to {data_dir / 'vector' / 'indicator.parquet'}")
    print()

    # Analysis by regime
    print("=" * 70)
    print("RESULTS: Mean PRISM Metrics by Kinetic Regime")
    print("=" * 70)
    print()

    summary = results.group_by("regime").agg([
        pl.col("hurst").mean().round(3).alias("hurst"),
        pl.col("sample_entropy").mean().round(3).alias("sample_entropy"),
        pl.col("permutation_entropy").mean().round(3).alias("perm_entropy"),
        pl.col("spectral_entropy").mean().round(3).alias("spectral_entropy"),
        pl.len().alias("n"),
    ]).sort("regime")

    print(summary.to_pandas().to_string(index=False))
    print()

    # ANOVA tests
    print("=" * 70)
    print("STATISTICAL TESTS (ANOVA)")
    print("=" * 70)
    print()

    regimes = results["regime"].unique().to_list()

    for metric in ["hurst", "sample_entropy", "spectral_entropy"]:
        groups = [
            results.filter(pl.col("regime") == r)[metric].drop_nulls().to_numpy()
            for r in regimes
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            f_stat, p_val = scipy_stats.f_oneway(*groups)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{metric:20} F={f_stat:8.2f}  p={p_val:.6f} {sig}")

    print()

    # Physical interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("Expected behavior:")
    print("  - Linear regime: Higher Hurst (trending growth)")
    print("  - Transition: Intermediate values")
    print("  - Saturating: Lower Hurst (plateau/mean-reverting)")
    print()

    # Check if results match expectations
    linear_hurst = results.filter(pl.col("regime") == "linear")["hurst"].mean()
    saturating_hurst = results.filter(pl.col("regime") == "saturating")["hurst"].mean()

    if linear_hurst and saturating_hurst:
        if linear_hurst > saturating_hurst:
            print("✓ PASS: Linear regime shows higher Hurst than saturating")
        else:
            print("✗ UNEXPECTED: Saturating regime shows higher Hurst")

    print()
    print("Full results:")
    print(results.select([
        "indicator_id", "enzyme_name", "km", "vmax", "regime",
        "hurst", "sample_entropy", "spectral_entropy"
    ]).to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
