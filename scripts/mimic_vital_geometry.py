#!/usr/bin/env python3
"""
MIMIC Vital-to-Vital Geometry Analysis

Computes pairwise relationships between vital signs WITHIN each patient
to test the "decoupling" hypothesis:

    Do septic patients show weaker vital-to-vital coupling than stable patients?

Metrics computed for each vital pair:
    - Pearson correlation
    - Cross-correlation (max lag)
    - Spearman correlation (rank-based)

Reference:
    Buchman TG (2002). The community of the self. Nature, 420, 246-251.
    "Organ systems become uncoupled in critical illness"
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, pearsonr, spearmanr
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd
    import pyarrow.parquet as pq


def load_data(data_dir: Path):
    """Load observations and indicators."""
    if HAS_POLARS:
        observations = pl.read_parquet(data_dir / "raw" / "observations.parquet")
        indicators = pl.read_parquet(data_dir / "raw" / "indicators.parquet")
        return observations, indicators
    else:
        observations = pq.read_table(data_dir / "raw" / "observations.parquet").to_pandas()
        indicators = pq.read_table(data_dir / "raw" / "indicators.parquet").to_pandas()
        return observations, indicators


def compute_cross_correlation(x, y, max_lag=10):
    """Compute max cross-correlation within lag range."""
    if len(x) < max_lag * 2 or len(y) < max_lag * 2:
        return np.nan, 0

    # Normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)

    best_corr = 0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]

        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_corr, best_lag


def compute_pairwise_geometry(observations, indicators, min_overlap=20):
    """
    Compute vital-to-vital geometry for each patient.

    Returns DataFrame with one row per (stay, vital_pair) combination.
    """
    results = []

    # Get unique stays
    if HAS_POLARS:
        stays = indicators.select(["stay_id", "regime"]).unique()
        stay_list = stays.to_dicts()
    else:
        stays = indicators[["stay_id", "regime"]].drop_duplicates()
        stay_list = stays.to_dict('records')

    print(f"Computing vital-to-vital geometry for {len(stay_list)} ICU stays...")

    for i, stay_row in enumerate(stay_list):
        if i % 20 == 0:
            print(f"  {i}/{len(stay_list)}...")

        stay_id = stay_row["stay_id"]
        regime = stay_row["regime"]

        # Get all indicators for this stay
        if HAS_POLARS:
            stay_indicators = indicators.filter(pl.col("stay_id") == stay_id)
            vital_names = stay_indicators["vital_name"].unique().to_list()
        else:
            stay_indicators = indicators[indicators["stay_id"] == stay_id]
            vital_names = stay_indicators["vital_name"].unique().tolist()

        if len(vital_names) < 2:
            continue

        # Get signal topology for each vital
        vital_series = {}
        for vital in vital_names:
            if HAS_POLARS:
                ind_row = stay_indicators.filter(pl.col("vital_name") == vital)
                if len(ind_row) == 0:
                    continue
                indicator_id = ind_row["indicator_id"][0]
                ts = observations.filter(pl.col("indicator_id") == indicator_id)
                ts = ts.sort("obs_date")
                values = ts["value"].to_numpy()
            else:
                ind_row = stay_indicators[stay_indicators["vital_name"] == vital]
                if len(ind_row) == 0:
                    continue
                indicator_id = ind_row["indicator_id"].iloc[0]
                ts = observations[observations["indicator_id"] == indicator_id]
                ts = ts.sort_values("obs_date")
                values = ts["value"].values

            if len(values) >= min_overlap:
                vital_series[vital] = values

        # Compute pairwise metrics for all vital combinations
        for vital1, vital2 in combinations(sorted(vital_series.keys()), 2):
            v1 = vital_series[vital1]
            v2 = vital_series[vital2]

            # Align to same length (simple truncation)
            min_len = min(len(v1), len(v2))
            if min_len < min_overlap:
                continue

            v1 = v1[:min_len]
            v2 = v2[:min_len]

            # Compute metrics
            try:
                pearson_r, pearson_p = pearsonr(v1, v2)
            except:
                pearson_r, pearson_p = np.nan, np.nan

            try:
                spearman_r, spearman_p = spearmanr(v1, v2)
            except:
                spearman_r, spearman_p = np.nan, np.nan

            try:
                xcorr, xcorr_lag = compute_cross_correlation(v1, v2)
            except:
                xcorr, xcorr_lag = np.nan, 0

            results.append({
                "stay_id": stay_id,
                "regime": regime,
                "vital1": vital1,
                "vital2": vital2,
                "pair": f"{vital1}___{vital2}",
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "xcorr_max": xcorr,
                "xcorr_lag": xcorr_lag,
                "abs_pearson": abs(pearson_r) if not np.isnan(pearson_r) else np.nan,
                "abs_xcorr": abs(xcorr) if not np.isnan(xcorr) else np.nan,
                "n_points": min_len,
            })

    if HAS_POLARS:
        return pl.DataFrame(results)
    else:
        return pd.DataFrame(results)


def main():
    data_dir = Path("data/mimic_demo")

    print("=" * 70)
    print("MIMIC Vital-to-Vital Geometry: Testing Decoupling Hypothesis")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    observations, indicators = load_data(data_dir)

    if HAS_POLARS:
        print(f"Observations: {len(observations)}")
        print(f"Indicators: {len(indicators)}")
    else:
        print(f"Observations: {len(observations)}")
        print(f"Indicators: {len(indicators)}")
    print()

    # Compute pairwise geometry
    geometry = compute_pairwise_geometry(observations, indicators)

    print(f"\nComputed {len(geometry)} vital pairs")
    print()

    # Save results
    geometry_dir = data_dir / "geometry"
    geometry_dir.mkdir(exist_ok=True)

    if HAS_POLARS:
        geometry.write_parquet(geometry_dir / "vital_pairs.parquet")
    else:
        geometry.to_parquet(geometry_dir / "vital_pairs.parquet")

    print(f"Saved to {geometry_dir / 'vital_pairs.parquet'}")
    print()

    # Analysis: Septic vs Stable coupling
    print("=" * 70)
    print("RESULTS: Vital-to-Vital Coupling by Regime")
    print("=" * 70)
    print()

    if HAS_POLARS:
        valid = geometry.filter(
            pl.col("abs_pearson").is_not_null() &
            pl.col("abs_pearson").is_finite()
        )

        # Overall summary by regime
        summary = valid.group_by("regime").agg([
            pl.col("abs_pearson").mean().round(3).alias("mean_abs_pearson"),
            pl.col("abs_xcorr").mean().round(3).alias("mean_abs_xcorr"),
            pl.col("pearson_r").mean().round(3).alias("mean_pearson"),
            pl.len().alias("n_pairs"),
        ])
        print("Overall coupling by regime:")
        print(summary)
        print()

        # ANOVA
        septic = valid.filter(pl.col("regime") == "septic")["abs_pearson"].drop_nulls().to_numpy()
        stable = valid.filter(pl.col("regime") == "stable")["abs_pearson"].drop_nulls().to_numpy()
    else:
        valid = geometry.dropna(subset=["abs_pearson"])
        valid = valid[np.isfinite(valid["abs_pearson"])]

        # Overall summary by regime
        summary = valid.groupby("regime").agg({
            "abs_pearson": "mean",
            "abs_xcorr": "mean",
            "pearson_r": "mean",
            "stay_id": "count"
        }).round(3)
        summary.columns = ["mean_abs_pearson", "mean_abs_xcorr", "mean_pearson", "n_pairs"]
        print("Overall coupling by regime:")
        print(summary)
        print()

        septic = valid[valid["regime"] == "septic"]["abs_pearson"].dropna().values
        stable = valid[valid["regime"] == "stable"]["abs_pearson"].dropna().values

    if len(septic) > 5 and len(stable) > 5:
        f_stat, p_val = f_oneway(septic, stable)
        print(f"ANOVA (|Pearson|): F = {f_stat:.2f}, p = {p_val:.6f}")
        print(f"  Septic mean: {np.mean(septic):.3f} (n={len(septic)})")
        print(f"  Stable mean: {np.mean(stable):.3f} (n={len(stable)})")

        if np.mean(septic) < np.mean(stable):
            print("\n  ** DECOUPLING CONFIRMED: Septic patients show WEAKER vital-to-vital coupling **")
        else:
            print("\n  Septic patients show STRONGER coupling (unexpected)")
    print()

    # By vital pair
    print("=" * 70)
    print("COUPLING BY VITAL PAIR")
    print("=" * 70)
    print()

    if HAS_POLARS:
        pair_summary = valid.group_by(["pair", "regime"]).agg([
            pl.col("abs_pearson").mean().round(3).alias("mean_coupling"),
            pl.len().alias("n"),
        ]).sort(["pair", "regime"])

        # Pivot to wide format for comparison
        pairs = valid["pair"].unique().to_list()
    else:
        pair_summary = valid.groupby(["pair", "regime"]).agg({
            "abs_pearson": "mean",
            "stay_id": "count"
        }).round(3)
        pair_summary.columns = ["mean_coupling", "n"]
        pair_summary = pair_summary.reset_index().sort_values(["pair", "regime"])
        pairs = valid["pair"].unique().tolist()

    print(f"{'Vital Pair':<50} {'Septic':>10} {'Stable':>10} {'Δ':>10}")
    print("-" * 80)

    for pair in sorted(pairs)[:15]:  # Top 15 pairs
        if HAS_POLARS:
            septic_row = valid.filter((pl.col("pair") == pair) & (pl.col("regime") == "septic"))
            stable_row = valid.filter((pl.col("pair") == pair) & (pl.col("regime") == "stable"))
            septic_mean = septic_row["abs_pearson"].mean() if len(septic_row) > 0 else np.nan
            stable_mean = stable_row["abs_pearson"].mean() if len(stable_row) > 0 else np.nan
        else:
            septic_data = valid[(valid["pair"] == pair) & (valid["regime"] == "septic")]
            stable_data = valid[(valid["pair"] == pair) & (valid["regime"] == "stable")]
            septic_mean = septic_data["abs_pearson"].mean() if len(septic_data) > 0 else np.nan
            stable_mean = stable_data["abs_pearson"].mean() if len(stable_data) > 0 else np.nan

        if not np.isnan(septic_mean) and not np.isnan(stable_mean):
            delta = septic_mean - stable_mean
            print(f"{pair:<50} {septic_mean:>10.3f} {stable_mean:>10.3f} {delta:>+10.3f}")

    print()

    # Key pairs analysis
    print("=" * 70)
    print("KEY VITAL PAIRS: ANOVA by Pair")
    print("=" * 70)
    print()

    key_pairs = [
        "heart_rate___respiratory_rate",
        "heart_rate___spo2",
        "respiratory_rate___spo2",
        "heart_rate___non_invasive_bp_systolic",
    ]

    for pair in key_pairs:
        if HAS_POLARS:
            pair_data = valid.filter(pl.col("pair") == pair)
            if len(pair_data) < 10:
                continue
            septic_vals = pair_data.filter(pl.col("regime") == "septic")["abs_pearson"].drop_nulls().to_numpy()
            stable_vals = pair_data.filter(pl.col("regime") == "stable")["abs_pearson"].drop_nulls().to_numpy()
        else:
            pair_data = valid[valid["pair"] == pair]
            if len(pair_data) < 10:
                continue
            septic_vals = pair_data[pair_data["regime"] == "septic"]["abs_pearson"].dropna().values
            stable_vals = pair_data[pair_data["regime"] == "stable"]["abs_pearson"].dropna().values

        if len(septic_vals) > 3 and len(stable_vals) > 3:
            f_stat, p_val = f_oneway(septic_vals, stable_vals)
            sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            direction = "↓" if np.mean(septic_vals) < np.mean(stable_vals) else "↑"
            print(f"{pair}:")
            print(f"  Septic: {np.mean(septic_vals):.3f}, Stable: {np.mean(stable_vals):.3f}")
            print(f"  F = {f_stat:.2f}, p = {p_val:.4f} {sig} {direction}")
            print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("The 'decoupling hypothesis' (Buchman 2002) predicts that:")
    print("  - Healthy patients: Strong coupling between vital signs")
    print("  - Septic patients: Weaker coupling (organ systems decouple)")
    print()
    print("If septic |correlation| < stable |correlation|, decoupling is supported.")
    print()


if __name__ == "__main__":
    main()
