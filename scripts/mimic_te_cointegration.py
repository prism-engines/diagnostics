#!/usr/bin/env python3
"""
MIMIC Transfer Entropy & Cointegration Analysis

Computes advanced pairwise metrics between vital signs:
- Transfer Entropy (TE): Directional information flow
- Cointegration: Long-run equilibrium relationships

Questions:
1. Do septic patients show disrupted information flow between vitals?
2. Do septic patients lose cointegration (equilibrium breakdown)?
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# =============================================================================
# Transfer Entropy Implementation
# =============================================================================

def discretize(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize continuous values into bins."""
    if len(x) == 0:
        return np.array([])
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(x, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return np.digitize(x, bins[1:-1])


def compute_transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1, n_bins: int = 8) -> dict:
    """
    Compute transfer entropy from X to Y.

    TE(X→Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-1)

    Measures how much knowing X's past reduces uncertainty about Y's future.
    """
    if len(x) < 50 or len(y) < 50:
        return {"te_x_to_y": np.nan, "te_y_to_x": np.nan, "te_asymmetry": np.nan}

    # Align lengths
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Discretize
    x_d = discretize(x, n_bins)
    y_d = discretize(y, n_bins)

    # Create lagged sequences
    n = len(x_d) - lag
    if n < 20:
        return {"te_x_to_y": np.nan, "te_y_to_x": np.nan, "te_asymmetry": np.nan}

    y_t = y_d[lag:]      # Y at time t
    y_lag = y_d[:-lag]   # Y at time t-1
    x_lag = x_d[:-lag]   # X at time t-1

    # Count joint occurrences
    def entropy(counts):
        """Shannon entropy from counts."""
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def joint_counts(*arrays):
        """Count joint occurrences."""
        combined = list(zip(*arrays))
        return np.array(list(Counter(combined).values()))

    # H(Y_t | Y_t-1) via H(Y_t, Y_t-1) - H(Y_t-1)
    h_y_ylag = entropy(joint_counts(y_t, y_lag))
    h_ylag = entropy(joint_counts(y_lag))
    h_y_given_ylag = h_y_ylag - h_ylag

    # H(Y_t | Y_t-1, X_t-1) via H(Y_t, Y_t-1, X_t-1) - H(Y_t-1, X_t-1)
    h_y_ylag_xlag = entropy(joint_counts(y_t, y_lag, x_lag))
    h_ylag_xlag = entropy(joint_counts(y_lag, x_lag))
    h_y_given_ylag_xlag = h_y_ylag_xlag - h_ylag_xlag

    # Transfer entropy X → Y
    te_x_to_y = h_y_given_ylag - h_y_given_ylag_xlag

    # Now compute Y → X
    x_t = x_d[lag:]
    x_lag2 = x_d[:-lag]
    y_lag2 = y_d[:-lag]

    h_x_xlag = entropy(joint_counts(x_t, x_lag2))
    h_xlag = entropy(joint_counts(x_lag2))
    h_x_given_xlag = h_x_xlag - h_xlag

    h_x_xlag_ylag = entropy(joint_counts(x_t, x_lag2, y_lag2))
    h_xlag_ylag = entropy(joint_counts(x_lag2, y_lag2))
    h_x_given_xlag_ylag = h_x_xlag_ylag - h_xlag_ylag

    te_y_to_x = h_x_given_xlag - h_x_given_xlag_ylag

    # Asymmetry: positive = X drives Y more than Y drives X
    te_asymmetry = te_x_to_y - te_y_to_x

    return {
        "te_x_to_y": max(0, te_x_to_y),  # Clip negative (sampling noise)
        "te_y_to_x": max(0, te_y_to_x),
        "te_asymmetry": te_asymmetry,
        "te_total": max(0, te_x_to_y) + max(0, te_y_to_x),
    }


# =============================================================================
# Cointegration Implementation
# =============================================================================

def adf_test(x: np.ndarray, max_lag: int = None) -> tuple:
    """
    Augmented Dickey-Fuller test for stationarity.
    Returns (adf_stat, p_value_approx, is_stationary)
    """
    n = len(x)
    if n < 20:
        return np.nan, np.nan, None

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** 0.25))
    max_lag = min(max_lag, n // 4)

    # First difference
    dx = np.diff(x)
    x_lag = x[:-1]

    # Simple regression: dx = alpha + beta * x_lag + error
    # ADF statistic = beta / SE(beta)

    X = np.column_stack([np.ones(len(x_lag)), x_lag])
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, dx, rcond=None)

        if len(residuals) == 0:
            sse = np.sum((dx - X @ beta) ** 2)
        else:
            sse = residuals[0]

        mse = sse / (len(dx) - 2)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(var_beta[1, 1])

        adf_stat = beta[1] / se_beta

        # Approximate p-value (MacKinnon critical values for n=100)
        # -3.51 (1%), -2.89 (5%), -2.58 (10%)
        if adf_stat < -3.51:
            p_approx = 0.01
        elif adf_stat < -2.89:
            p_approx = 0.05
        elif adf_stat < -2.58:
            p_approx = 0.10
        else:
            p_approx = 0.50

        is_stationary = adf_stat < -2.89  # 5% level

        return adf_stat, p_approx, is_stationary
    except:
        return np.nan, np.nan, None


def compute_cointegration(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Engle-Granger cointegration test.

    1. Regress Y on X: Y = alpha + beta*X + residuals
    2. Test if residuals are stationary (ADF test)
    3. If stationary → cointegrated (share long-run equilibrium)
    """
    if len(x) < 50 or len(y) < 50:
        return {"cointegrated": None, "coint_stat": np.nan, "hedge_ratio": np.nan}

    # Align
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    # Check if both series are non-stationary (required for cointegration)
    adf_x, _, stat_x = adf_test(x)
    adf_y, _, stat_y = adf_test(y)

    # Regress Y on X
    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        hedge_ratio = beta[1]

        # Residuals (spread)
        residuals = y - (beta[0] + beta[1] * x)

        # ADF test on residuals
        adf_resid, p_resid, is_coint = adf_test(residuals)

        # Cointegration strength: more negative ADF = stronger cointegration
        coint_strength = -adf_resid if not np.isnan(adf_resid) else np.nan

        return {
            "cointegrated": is_coint,
            "coint_stat": adf_resid,
            "coint_pvalue": p_resid,
            "coint_strength": coint_strength,
            "hedge_ratio": hedge_ratio,
            "spread_std": np.std(residuals),
        }
    except:
        return {"cointegrated": None, "coint_stat": np.nan, "hedge_ratio": np.nan}


# =============================================================================
# Main Analysis
# =============================================================================

def load_mimic_vitals(data_dir: Path):
    """Load MIMIC chartevents with vital signs."""
    base = data_dir / "mimic-iv-clinical-database-demo-2.2"

    VITAL_ITEMIDS = {
        220045: "heart_rate",
        220052: "arterial_bp_mean",
        220181: "nibp_mean",
        220210: "respiratory_rate",
        220277: "spo2",
    }

    # Load chartevents
    chartevents_path = base / "icu" / "chartevents.csv.gz"
    if HAS_POLARS:
        chartevents = pl.read_csv(
            chartevents_path,
            columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
            schema_overrides={"valuenum": pl.Float64}
        )

        # Filter to vitals
        chartevents = chartevents.filter(pl.col("itemid").is_in(list(VITAL_ITEMIDS.keys())))

        # Add vital name
        vital_map = pl.DataFrame({
            "itemid": list(VITAL_ITEMIDS.keys()),
            "vital_name": list(VITAL_ITEMIDS.values())
        })
        chartevents = chartevents.join(vital_map, on="itemid", how="left")
        chartevents = chartevents.with_columns(
            pl.col("charttime").str.to_datetime().alias("charttime")
        )

    # Load ICU stays
    icustays = pl.read_csv(base / "icu" / "icustays.csv.gz")

    # Load diagnoses for sepsis status
    diagnoses = pl.read_csv(base / "hosp" / "diagnoses_icd.csv.gz")

    return chartevents, icustays, diagnoses


def main():
    data_dir = Path("data/mimic_demo")

    print("=" * 70)
    print("MIMIC Transfer Entropy & Cointegration Analysis")
    print("=" * 70)
    print()
    print("Questions:")
    print("  1. Do septic patients show disrupted information flow (TE)?")
    print("  2. Do septic patients lose cointegration (equilibrium breakdown)?")
    print()

    # Load data
    print("Loading data...")
    chartevents, icustays, diagnoses = load_mimic_vitals(data_dir)

    # Get sepsis status
    sepsis_codes = ["99591", "99592", "A4150", "A4151", "A4152", "A4153", "A4154", "A419", "R6520", "R6521"]
    sepsis_hadms = diagnoses.filter(
        pl.col("icd_code").is_in(sepsis_codes)
    )["hadm_id"].unique().to_list()

    # Vital pairs to analyze
    vital_pairs = [
        ("heart_rate", "arterial_bp_mean"),
        ("heart_rate", "respiratory_rate"),
        ("heart_rate", "spo2"),
        ("respiratory_rate", "spo2"),
        ("heart_rate", "nibp_mean"),
    ]

    results = []
    stays = icustays.to_dicts()

    print(f"\nProcessing {len(stays)} ICU stays...")

    for i, stay in enumerate(stays):
        if i % 20 == 0:
            print(f"  {i}/{len(stays)}...")

        stay_id = stay["stay_id"]
        hadm_id = stay["hadm_id"]
        regime = "septic" if hadm_id in sepsis_hadms else "stable"

        # Get vitals for this stay
        stay_vitals = chartevents.filter(pl.col("stay_id") == stay_id)

        if len(stay_vitals) < 100:
            continue

        # Get signal topology for each vital
        vital_series = {}
        for vital in ["heart_rate", "arterial_bp_mean", "nibp_mean", "respiratory_rate", "spo2"]:
            v_data = stay_vitals.filter(pl.col("vital_name") == vital).sort("charttime")
            if len(v_data) >= 30:
                vital_series[vital] = v_data["valuenum"].drop_nulls().to_numpy()

        # Compute TE and cointegration for each pair
        for v1, v2 in vital_pairs:
            if v1 not in vital_series or v2 not in vital_series:
                continue

            x = vital_series[v1]
            y = vital_series[v2]

            # Transfer Entropy
            te_result = compute_transfer_entropy(x, y, lag=1, n_bins=8)

            # Cointegration
            coint_result = compute_cointegration(x, y)

            if not np.isnan(te_result["te_x_to_y"]):
                results.append({
                    "stay_id": stay_id,
                    "regime": regime,
                    "pair": f"{v1}___{v2}",
                    "te_x_to_y": te_result["te_x_to_y"],
                    "te_y_to_x": te_result["te_y_to_x"],
                    "te_total": te_result["te_total"],
                    "te_asymmetry": te_result["te_asymmetry"],
                    "cointegrated": coint_result.get("cointegrated"),
                    "coint_stat": coint_result.get("coint_stat"),
                    "coint_strength": coint_result.get("coint_strength"),
                    "hedge_ratio": coint_result.get("hedge_ratio"),
                })

    if not results:
        print("No results computed.")
        return

    df = pl.DataFrame(results)
    print(f"\nComputed {len(df)} pair measurements")

    # Save
    output_dir = data_dir / "geometry"
    output_dir.mkdir(exist_ok=True)
    df.write_parquet(output_dir / "te_cointegration.parquet")
    print(f"Saved to {output_dir / 'te_cointegration.parquet'}")
    print()

    # ==========================================================================
    # TRANSFER ENTROPY ANALYSIS
    # ==========================================================================
    print("=" * 70)
    print("TRANSFER ENTROPY RESULTS")
    print("=" * 70)
    print()

    # Summary by regime
    te_summary = df.group_by("regime").agg([
        pl.col("te_total").mean().round(4).alias("mean_te_total"),
        pl.col("te_x_to_y").mean().round(4).alias("mean_te_x_to_y"),
        pl.col("te_y_to_x").mean().round(4).alias("mean_te_y_to_x"),
        pl.col("te_asymmetry").mean().round(4).alias("mean_asymmetry"),
        pl.len().alias("n"),
    ])

    print("Transfer Entropy by regime:")
    print(te_summary)
    print()

    # ANOVA
    from scipy.stats import f_oneway
    septic_te = df.filter(pl.col("regime") == "septic")["te_total"].drop_nulls().to_numpy()
    stable_te = df.filter(pl.col("regime") == "stable")["te_total"].drop_nulls().to_numpy()

    if len(septic_te) > 5 and len(stable_te) > 5:
        f_stat, p_val = f_oneway(septic_te, stable_te)
        print(f"ANOVA (Total TE): F = {f_stat:.2f}, p = {p_val:.4f}")
        print(f"  Septic mean TE: {np.mean(septic_te):.4f}")
        print(f"  Stable mean TE: {np.mean(stable_te):.4f}")

        if np.mean(septic_te) < np.mean(stable_te):
            print("\n  ** Septic shows LOWER TE (reduced information flow) **")
        else:
            print("\n  Septic shows higher TE")
    print()

    # By pair
    print("Transfer Entropy by vital pair:")
    print()
    print(f"{'Pair':<40} {'Septic TE':>12} {'Stable TE':>12} {'Δ':>10}")
    print("-" * 75)

    for pair in df["pair"].unique().to_list():
        pair_data = df.filter(pl.col("pair") == pair)
        septic_mean = pair_data.filter(pl.col("regime") == "septic")["te_total"].mean()
        stable_mean = pair_data.filter(pl.col("regime") == "stable")["te_total"].mean()

        if septic_mean is not None and stable_mean is not None:
            delta = septic_mean - stable_mean
            marker = "↓" if delta < -0.01 else ""
            print(f"{pair:<40} {septic_mean:>12.4f} {stable_mean:>12.4f} {delta:>+10.4f} {marker}")

    print()

    # ==========================================================================
    # COINTEGRATION ANALYSIS
    # ==========================================================================
    print("=" * 70)
    print("COINTEGRATION RESULTS")
    print("=" * 70)
    print()

    # Filter to valid cointegration results
    coint_df = df.filter(pl.col("coint_stat").is_not_null() & pl.col("coint_stat").is_finite())

    if len(coint_df) > 0:
        # Cointegration rate by regime
        coint_rate = coint_df.group_by("regime").agg([
            pl.col("cointegrated").mean().round(3).alias("coint_rate"),
            pl.col("coint_strength").mean().round(3).alias("mean_coint_strength"),
            pl.len().alias("n"),
        ])

        print("Cointegration by regime:")
        print(coint_rate)
        print()

        # ANOVA on cointegration strength
        septic_coint = coint_df.filter(pl.col("regime") == "septic")["coint_strength"].drop_nulls().to_numpy()
        stable_coint = coint_df.filter(pl.col("regime") == "stable")["coint_strength"].drop_nulls().to_numpy()

        if len(septic_coint) > 5 and len(stable_coint) > 5:
            f_stat, p_val = f_oneway(septic_coint, stable_coint)
            print(f"ANOVA (Coint Strength): F = {f_stat:.2f}, p = {p_val:.4f}")
            print(f"  Septic mean strength: {np.mean(septic_coint):.3f}")
            print(f"  Stable mean strength: {np.mean(stable_coint):.3f}")

            if np.mean(septic_coint) < np.mean(stable_coint):
                print("\n  ** Septic shows WEAKER cointegration (equilibrium breakdown) **")
        print()

        # By pair
        print("Cointegration rate by vital pair:")
        print()
        print(f"{'Pair':<40} {'Septic %':>12} {'Stable %':>12} {'Δ':>10}")
        print("-" * 75)

        for pair in coint_df["pair"].unique().to_list():
            pair_data = coint_df.filter(pl.col("pair") == pair)
            septic_rate = pair_data.filter(pl.col("regime") == "septic")["cointegrated"].mean()
            stable_rate = pair_data.filter(pl.col("regime") == "stable")["cointegrated"].mean()

            if septic_rate is not None and stable_rate is not None:
                delta = septic_rate - stable_rate
                marker = "↓" if delta < -0.1 else ""
                print(f"{pair:<40} {septic_rate*100:>11.1f}% {stable_rate*100:>11.1f}% {delta*100:>+9.1f}% {marker}")

    print()

    # ==========================================================================
    # INTERPRETATION
    # ==========================================================================
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("Transfer Entropy (TE):")
    print("  - Measures directed information flow between vitals")
    print("  - Lower TE = less predictive coupling")
    print("  - If septic TE < stable TE → information flow disrupted")
    print()
    print("Cointegration:")
    print("  - Tests long-run equilibrium between vitals")
    print("  - Lower rate = equilibrium breakdown")
    print("  - If septic coint < stable coint → regulatory systems decoupled")
    print()


if __name__ == "__main__":
    main()
