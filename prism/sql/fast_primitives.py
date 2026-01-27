"""
SQL Fast Primitives - DuckDB Vectorized Computations

Instant computations that would take minutes in Python loops.
DuckDB does them in milliseconds.

CANONICAL SPLIT:
    SQL (this file):  correlation, covariance, basic stats, derivatives, typology
    Python (engines): hurst, lyapunov, garch, entropy, fft, granger, dtw, rqa
"""

import duckdb
import pandas as pd
from pathlib import Path


def compute_all_fast(observations_path: str) -> dict:
    """
    Compute ALL SQL-able primitives in one shot.

    Returns dict with DataFrames:
        - 'signal_stats': Basic stats per signal
        - 'pairwise': Correlation, covariance per pair
        - 'pointwise': Derivatives per point
        - 'typology': Signal classification
    """
    con = duckdb.connect()

    # Load observations
    con.execute(f"""
        CREATE TABLE obs AS
        SELECT * FROM read_parquet('{observations_path}')
    """)

    results = {}

    # 1. SIGNAL STATS (per entity_id, signal_id)
    results['signal_stats'] = con.execute("""
        SELECT
            entity_id,
            signal_id,
            COUNT(*) AS n_points,
            AVG(y) AS y_mean,
            STDDEV_POP(y) AS y_std,
            MIN(y) AS y_min,
            MAX(y) AS y_max,
            MEDIAN(y) AS y_median,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS y_q25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS y_q75,
            SKEWNESS(y) AS y_skew,
            KURTOSIS(y) AS y_kurtosis,
            COUNT(DISTINCT y)::FLOAT / COUNT(*)::FLOAT AS unique_ratio,
            VAR_POP(y) AS variance
        FROM obs
        GROUP BY entity_id, signal_id
    """).df()

    # 2. TYPOLOGY (signal classification)
    results['typology'] = con.execute("""
        WITH stats AS (
            SELECT
                entity_id,
                signal_id,
                COUNT(*) AS n_points,
                COUNT(DISTINCT y)::FLOAT / COUNT(*)::FLOAT AS unique_ratio,
                VAR_POP(y) AS variance,
                AVG(y) AS mean,
                STDDEV_POP(y) AS std,
                MIN(y) AS min,
                MAX(y) AS max
            FROM obs
            GROUP BY entity_id, signal_id
        )
        SELECT
            entity_id,
            signal_id,
            n_points,
            unique_ratio,
            variance,
            mean,
            std,
            min,
            max,
            CASE
                WHEN variance < 1e-10 THEN 'constant'
                WHEN unique_ratio < 0.01 THEN 'digital'
                WHEN unique_ratio < 0.05 THEN 'discrete'
                ELSE 'analog'
            END AS signal_class
        FROM stats
    """).df()

    # 3. PAIRWISE CORRELATIONS (the big one - instant in SQL)
    results['pairwise'] = con.execute("""
        SELECT
            a.entity_id,
            a.signal_id AS signal_a,
            b.signal_id AS signal_b,
            CORR(a.y, b.y) AS correlation,
            COVAR_POP(a.y, b.y) AS covariance,
            COUNT(*) AS n_overlap
        FROM obs a
        JOIN obs b
            ON a.entity_id = b.entity_id
            AND a.I = b.I
            AND a.signal_id < b.signal_id
        GROUP BY a.entity_id, a.signal_id, b.signal_id
        HAVING COUNT(*) >= 10
    """).df()

    # 4. LAG CORRELATIONS (lag-1 cross-correlation)
    results['lag_correlation'] = con.execute("""
        WITH lagged AS (
            SELECT
                entity_id,
                signal_id,
                I,
                y,
                LAG(y) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS y_lag1
            FROM obs
        )
        SELECT
            a.entity_id,
            a.signal_id AS signal_a,
            b.signal_id AS signal_b,
            CORR(a.y, b.y_lag1) AS lag1_corr_a_leads,
            CORR(a.y_lag1, b.y) AS lag1_corr_b_leads
        FROM lagged a
        JOIN lagged b
            ON a.entity_id = b.entity_id
            AND a.I = b.I
            AND a.signal_id < b.signal_id
        WHERE a.y_lag1 IS NOT NULL AND b.y_lag1 IS NOT NULL
        GROUP BY a.entity_id, a.signal_id, b.signal_id
    """).df()

    # 5. POINTWISE DERIVATIVES (staged CTEs to avoid nested window functions)
    results['pointwise'] = con.execute("""
        WITH dy_calc AS (
            -- Stage 1: First derivative
            SELECT
                entity_id,
                signal_id,
                I,
                y,
                y - LAG(y) OVER w AS dy,
                LEAD(y) OVER w - LAG(y) OVER w AS dy_central
            FROM obs
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        ),
        d2y_calc AS (
            -- Stage 2: Second derivative from first
            SELECT
                entity_id,
                signal_id,
                I,
                y,
                dy,
                dy_central,
                dy - LAG(dy) OVER w AS d2y,
                LAG(dy) OVER w AS dy_prev
            FROM dy_calc
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        )
        SELECT
            entity_id,
            signal_id,
            I,
            y,
            COALESCE(dy, 0) AS dy,
            COALESCE(d2y, 0) AS d2y,
            COALESCE(dy_central, 0) / 2.0 AS dy_central,
            -- Curvature approximation: d2y / (1 + dy^2)^1.5
            CASE
                WHEN dy IS NOT NULL AND d2y IS NOT NULL
                THEN d2y / POWER(1 + dy*dy, 1.5)
                ELSE 0
            END AS curvature,
            -- Sign changes (regime boundaries)
            CASE
                WHEN dy > 0 AND dy_prev <= 0 THEN 1
                WHEN dy < 0 AND dy_prev >= 0 THEN -1
                ELSE 0
            END AS dy_sign_change
        FROM d2y_calc
    """).df()

    # 6. DERIVATIVE STATS (per signal) - staged CTEs
    results['derivative_stats'] = con.execute("""
        WITH dy_calc AS (
            -- Stage 1: First derivative
            SELECT
                entity_id,
                signal_id,
                I,
                y - LAG(y) OVER w AS dy
            FROM obs
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        ),
        d2y_calc AS (
            -- Stage 2: Second derivative + lag for sign changes
            SELECT
                entity_id,
                signal_id,
                dy,
                dy - LAG(dy) OVER w AS d2y,
                LAG(dy) OVER w AS dy_prev
            FROM dy_calc
            WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
        )
        SELECT
            entity_id,
            signal_id,
            AVG(dy) AS dy_mean,
            STDDEV_POP(dy) AS dy_std,
            MAX(ABS(dy)) AS dy_max_abs,
            AVG(d2y) AS d2y_mean,
            STDDEV_POP(d2y) AS d2y_std,
            -- Count sign changes (volatility proxy)
            SUM(CASE WHEN dy > 0 AND dy_prev <= 0 THEN 1
                     WHEN dy < 0 AND dy_prev >= 0 THEN 1
                     ELSE 0 END) AS n_sign_changes
        FROM d2y_calc
        GROUP BY entity_id, signal_id
    """).df()

    con.close()
    return results


def compute_correlation_matrix(observations_path: str) -> pd.DataFrame:
    """Just the correlation matrix - fastest possible."""
    con = duckdb.connect()

    result = con.execute(f"""
        SELECT
            a.entity_id,
            a.signal_id AS signal_a,
            b.signal_id AS signal_b,
            CORR(a.y, b.y) AS correlation
        FROM read_parquet('{observations_path}') a
        JOIN read_parquet('{observations_path}') b
            ON a.entity_id = b.entity_id
            AND a.I = b.I
            AND a.signal_id < b.signal_id
        GROUP BY a.entity_id, a.signal_id, b.signal_id
    """).df()

    con.close()
    return result


def compute_typology(observations_path: str) -> pd.DataFrame:
    """Signal classification - instant."""
    con = duckdb.connect()

    result = con.execute(f"""
        WITH stats AS (
            SELECT
                entity_id,
                signal_id,
                COUNT(*) AS n_points,
                COUNT(DISTINCT y)::FLOAT / COUNT(*)::FLOAT AS unique_ratio,
                VAR_POP(y) AS variance,
                AVG(y) AS mean,
                STDDEV_POP(y) AS std,
                MIN(y) AS min,
                MAX(y) AS max
            FROM read_parquet('{observations_path}')
            GROUP BY entity_id, signal_id
        )
        SELECT
            entity_id,
            signal_id,
            n_points,
            unique_ratio,
            variance,
            mean,
            std,
            min,
            max,
            CASE
                WHEN variance < 1e-10 THEN 'constant'
                WHEN unique_ratio < 0.01 THEN 'digital'
                WHEN unique_ratio < 0.05 THEN 'discrete'
                ELSE 'analog'
            END AS signal_class
        FROM stats
    """).df()

    con.close()
    return result


def compute_derivatives(observations_path: str) -> pd.DataFrame:
    """Pointwise derivatives - instant."""
    con = duckdb.connect()

    result = con.execute(f"""
        SELECT
            entity_id,
            signal_id,
            I,
            y,
            COALESCE(y - LAG(y) OVER w, 0) AS dy,
            COALESCE((y - LAG(y) OVER w) - LAG(y - LAG(y) OVER w) OVER w, 0) AS d2y
        FROM read_parquet('{observations_path}')
        WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I)
    """).df()

    con.close()
    return result


def compute_typology_complete(observations_path: str) -> pd.DataFrame:
    """
    COMPLETE Signal Typology - 15+ dimensions of classification.

    PhD-level signal classification across orthogonal dimensions:
    - Index continuity (discrete_regular, irregular, event_driven)
    - Amplitude continuity (continuous, discrete, binary)
    - Monotonicity (monotonic_increasing, monotonic_decreasing, non_monotonic)
    - Sparsity (dense, sparse, very_sparse, ultra_sparse)
    - Boundedness (bounded, possibly_unbounded)
    - Energy class (zero_power, finite_power)
    - Stationarity hint (likely_stationary, likely_non_stationary)
    - Boolean flags (has_dc_offset, has_trend)
    - Legacy signal_class for compatibility

    Complex classifications (predictability, standard_form, symmetry, frequency_content)
    require FFT/Hurst/Lyapunov and are computed in Python engine.
    """
    con = duckdb.connect()

    result = con.execute(f"""
        WITH
        -- Basic statistics
        basic_stats AS (
            SELECT
                entity_id,
                signal_id,
                COUNT(*) AS n_points,
                AVG(y) AS mean,
                STDDEV_POP(y) AS std,
                MIN(y) AS min_val,
                MAX(y) AS max_val,
                MEDIAN(y) AS median,
                COUNT(DISTINCT y) AS n_unique,
                COUNT(DISTINCT y)::FLOAT / NULLIF(COUNT(*), 0) AS unique_ratio,
                VAR_POP(y) AS variance,
                SUM(y * y) AS sum_squared,
                SKEWNESS(y) AS skewness,
                KURTOSIS(y) AS kurtosis
            FROM read_parquet('{observations_path}')
            GROUP BY entity_id, signal_id
        ),

        -- Index (time/sequence) analysis
        index_analysis AS (
            SELECT
                entity_id,
                signal_id,
                MIN(I) AS index_min,
                MAX(I) AS index_max,
                MAX(I) - MIN(I) AS duration,
                AVG(dt) AS avg_interval,
                STDDEV_POP(dt) AS interval_std,
                STDDEV_POP(dt) / NULLIF(AVG(dt), 0) AS interval_cv
            FROM (
                SELECT
                    entity_id,
                    signal_id,
                    I,
                    I - LAG(I) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS dt
                FROM read_parquet('{observations_path}')
            ) t
            GROUP BY entity_id, signal_id
        ),

        -- Derivative analysis (monotonicity, trend)
        derivative_analysis AS (
            SELECT
                entity_id,
                signal_id,
                AVG(dy) AS mean_dy,
                STDDEV_POP(dy) AS std_dy,
                COUNT(*) FILTER (WHERE dy > 1e-10) AS n_positive_dy,
                COUNT(*) FILTER (WHERE dy < -1e-10) AS n_negative_dy,
                COUNT(*) FILTER (WHERE ABS(dy) <= 1e-10) AS n_zero_dy,
                COUNT(*) AS n_total
            FROM (
                SELECT
                    entity_id,
                    signal_id,
                    y - LAG(y) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS dy
                FROM read_parquet('{observations_path}')
            ) t
            WHERE dy IS NOT NULL
            GROUP BY entity_id, signal_id
        ),

        -- Zero crossing analysis (periodicity hint)
        zero_crossing AS (
            SELECT
                entity_id,
                signal_id,
                COUNT(*) AS n_zero_crossings
            FROM (
                SELECT
                    entity_id,
                    signal_id,
                    y,
                    LAG(y) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS prev_y
                FROM read_parquet('{observations_path}')
            ) t
            WHERE (y >= 0 AND prev_y < 0) OR (y < 0 AND prev_y >= 0)
            GROUP BY entity_id, signal_id
        ),

        -- Sparsity analysis
        sparsity_analysis AS (
            SELECT
                entity_id,
                signal_id,
                COUNT(*) FILTER (WHERE ABS(y - median_y) > 0.01 * range_y)::FLOAT / COUNT(*) AS density_ratio
            FROM (
                SELECT
                    entity_id,
                    signal_id,
                    y,
                    MEDIAN(y) OVER (PARTITION BY entity_id, signal_id) AS median_y,
                    MAX(ABS(y)) OVER (PARTITION BY entity_id, signal_id) -
                    MIN(ABS(y)) OVER (PARTITION BY entity_id, signal_id) AS range_y
                FROM read_parquet('{observations_path}')
            ) t
            GROUP BY entity_id, signal_id
        )

        SELECT
            b.entity_id,
            b.signal_id,
            b.n_points,

            -- Basic stats
            ROUND(b.mean, 6) AS mean,
            ROUND(b.std, 6) AS std,
            ROUND(b.min_val, 6) AS min,
            ROUND(b.max_val, 6) AS max,
            ROUND(b.median, 6) AS median,
            ROUND(b.skewness, 4) AS skewness,
            ROUND(b.kurtosis, 4) AS kurtosis,

            -- Index continuity
            CASE
                WHEN i.interval_cv IS NULL OR i.interval_cv < 0.01 THEN 'discrete_regular'
                WHEN i.interval_cv < 0.1 THEN 'discrete_irregular'
                ELSE 'event_driven'
            END AS index_continuity,

            ROUND(1.0 / NULLIF(i.avg_interval, 0), 4) AS sampling_rate,
            ROUND(i.duration, 4) AS duration,

            -- Amplitude continuity
            CASE
                WHEN b.n_unique = 1 THEN 'constant'
                WHEN b.n_unique = 2 THEN 'binary'
                WHEN b.n_unique <= 10 THEN 'discrete_few'
                WHEN b.unique_ratio < 0.01 THEN 'discrete_quantized'
                WHEN b.unique_ratio < 0.05 THEN 'discrete_stepped'
                ELSE 'continuous'
            END AS amplitude_continuity,

            -- Stationarity hint (simple check)
            CASE
                WHEN b.std < 1e-10 THEN 'constant'
                WHEN ABS(d.mean_dy) < 0.001 * b.std THEN 'likely_stationary'
                ELSE 'likely_non_stationary'
            END AS stationarity_hint,

            -- Monotonicity
            CASE
                WHEN d.n_negative_dy = 0 AND d.n_positive_dy > 0 THEN 'monotonic_increasing'
                WHEN d.n_positive_dy = 0 AND d.n_negative_dy > 0 THEN 'monotonic_decreasing'
                WHEN d.n_positive_dy < 0.05 * d.n_total OR d.n_negative_dy < 0.05 * d.n_total THEN 'nearly_monotonic'
                ELSE 'non_monotonic'
            END AS monotonicity,

            -- Sparsity
            CASE
                WHEN s.density_ratio > 0.5 THEN 'dense'
                WHEN s.density_ratio > 0.1 THEN 'sparse'
                WHEN s.density_ratio > 0.01 THEN 'very_sparse'
                ELSE 'ultra_sparse'
            END AS sparsity,
            ROUND(s.density_ratio, 4) AS density_ratio,

            -- Boundedness
            CASE
                WHEN b.std < 1e-10 THEN 'constant'
                WHEN b.max_val - b.min_val < 100 * b.std THEN 'bounded'
                ELSE 'possibly_unbounded'
            END AS boundedness,

            -- Energy/Power
            ROUND(b.sum_squared, 4) AS total_energy,
            ROUND(b.sum_squared / NULLIF(b.n_points, 0), 6) AS avg_power,
            CASE
                WHEN b.variance < 1e-10 THEN 'zero_power'
                ELSE 'finite_power'
            END AS energy_class,

            -- Boolean flags
            CASE WHEN ABS(b.mean) > 0.1 * NULLIF(b.std, 0) THEN true ELSE false END AS has_dc_offset,
            CASE WHEN ABS(d.mean_dy) > 0.01 * NULLIF(b.std, 0) THEN true ELSE false END AS has_trend,

            -- Zero crossing (periodicity hint)
            COALESCE(z.n_zero_crossings, 0) AS n_zero_crossings,
            ROUND(COALESCE(z.n_zero_crossings, 0)::FLOAT / NULLIF(b.n_points, 0), 4) AS zero_crossing_rate,

            -- Legacy classification (compatibility)
            CASE
                WHEN b.variance < 1e-10 THEN 'constant'
                WHEN b.n_unique = 2 THEN 'binary'
                WHEN b.unique_ratio < 0.01 THEN 'digital'
                WHEN b.unique_ratio < 0.05 THEN 'discrete'
                WHEN COALESCE(s.density_ratio, 1) < 0.1 THEN 'event'
                ELSE 'analog'
            END AS signal_class,

            -- Raw measurements
            ROUND(b.unique_ratio, 6) AS unique_ratio,
            ROUND(b.variance, 6) AS variance,
            b.n_unique

        FROM basic_stats b
        LEFT JOIN index_analysis i ON b.entity_id = i.entity_id AND b.signal_id = i.signal_id
        LEFT JOIN derivative_analysis d ON b.entity_id = d.entity_id AND b.signal_id = d.signal_id
        LEFT JOIN zero_crossing z ON b.entity_id = z.entity_id AND b.signal_id = z.signal_id
        LEFT JOIN sparsity_analysis s ON b.entity_id = s.entity_id AND b.signal_id = s.signal_id
    """).df()

    con.close()
    return result


if __name__ == '__main__':
    import time

    obs_path = 'data/observations.parquet'

    print("Testing SQL fast primitives...")
    print("=" * 60)

    start = time.time()
    results = compute_all_fast(obs_path)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.3f}s")
    print("\nResults:")
    for name, df in results.items():
        print(f"  {name}: {len(df)} rows × {len(df.columns)} cols")

    print("\n" + "=" * 60)
    print("Testing COMPLETE typology...")
    print("=" * 60)

    start = time.time()
    typology = compute_typology_complete(obs_path)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.3f}s")
    print(f"Typology: {len(typology)} rows × {len(typology.columns)} cols")
    print(f"\nColumns: {list(typology.columns)}")
