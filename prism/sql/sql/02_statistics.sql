-- ============================================================================
-- ORTHON SQL ENGINES: STATISTICS (CALCULUS-BASED)
-- ============================================================================
-- Global statistics and calculus-derived metrics.
-- NO ROLLING WINDOWS - use derivatives for local behavior.
-- ============================================================================

-- ============================================================================
-- 001: GLOBAL STATISTICS (per signal) - KEEP: aggregation for summary
-- ============================================================================

CREATE OR REPLACE VIEW v_stats_global AS
SELECT
    signal_id,
    index_dimension,
    signal_class,
    COUNT(*) AS n_points,
    MIN(y) AS y_min,
    MAX(y) AS y_max,
    MAX(y) - MIN(y) AS y_range,
    AVG(y) AS y_mean,
    STDDEV(y) AS y_std,
    VARIANCE(y) AS y_var,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS y_q1,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY y) AS y_median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS y_q3,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) -
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS y_iqr
FROM v_base
GROUP BY signal_id, index_dimension, signal_class;


-- ============================================================================
-- 002: DERIVATIVE STATISTICS (per signal)
-- ============================================================================
-- Global stats on dy and d2y for threshold calibration

CREATE OR REPLACE VIEW v_derivative_stats AS
SELECT
    signal_id,
    AVG(ABS(dy)) AS dy_mean_abs,
    STDDEV(dy) AS dy_std,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ABS(dy)) AS dy_median_abs,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(dy)) AS dy_p95_abs,
    AVG(ABS(d2y)) AS d2y_mean_abs,
    STDDEV(d2y) AS d2y_std,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ABS(d2y)) AS d2y_median_abs,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(d2y)) AS d2y_p95_abs
FROM v_d2y
WHERE dy IS NOT NULL AND d2y IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- 003: Z-SCORE (global, not rolling)
-- ============================================================================
-- Standardized value relative to signal's global distribution

CREATE OR REPLACE VIEW v_zscore AS
SELECT
    b.signal_id,
    b.I,
    b.y,
    (b.y - s.y_mean) / NULLIF(s.y_std, 0) AS z_score,
    CASE
        WHEN ABS((b.y - s.y_mean) / NULLIF(s.y_std, 0)) > 3 THEN 'extreme'
        WHEN ABS((b.y - s.y_mean) / NULLIF(s.y_std, 0)) > 2 THEN 'outlier'
        ELSE 'normal'
    END AS z_category
FROM v_base b
JOIN v_stats_global s USING (signal_id);


-- ============================================================================
-- 004: DERIVATIVE-BASED ANOMALY DETECTION
-- ============================================================================
-- Anomaly = |dy| or |d2y| exceeds threshold relative to signal

CREATE OR REPLACE VIEW v_derivative_anomaly AS
SELECT
    d.signal_id,
    d.I,
    d.dy,
    d.d2y,
    ds.dy_median_abs,
    ds.d2y_median_abs,
    -- Anomaly if derivative exceeds 3x median
    CASE WHEN ABS(d.dy) > 3 * ds.dy_median_abs THEN TRUE ELSE FALSE END AS dy_anomaly,
    CASE WHEN ABS(d.d2y) > 3 * ds.d2y_median_abs THEN TRUE ELSE FALSE END AS d2y_anomaly,
    -- Combined anomaly score
    ABS(d.dy) / NULLIF(ds.dy_median_abs, 0) +
    ABS(d.d2y) / NULLIF(ds.d2y_median_abs, 0) AS anomaly_score
FROM v_d2y d
JOIN v_derivative_stats ds USING (signal_id)
WHERE d.dy IS NOT NULL AND d.d2y IS NOT NULL;


-- ============================================================================
-- 005: LOCAL EXTREMA (peaks and valleys)
-- ============================================================================
-- Detected from sign change in dy

CREATE OR REPLACE VIEW v_local_extrema AS
SELECT
    signal_id,
    I,
    y,
    dy,
    LAG(dy) OVER (PARTITION BY signal_id ORDER BY I) AS dy_prev,
    CASE
        WHEN dy > 0 AND LAG(dy) OVER (PARTITION BY signal_id ORDER BY I) < 0 THEN 'valley'
        WHEN dy < 0 AND LAG(dy) OVER (PARTITION BY signal_id ORDER BY I) > 0 THEN 'peak'
        ELSE NULL
    END AS extrema_type
FROM v_dy
WHERE dy IS NOT NULL;


-- ============================================================================
-- 006: TREND DIRECTION (from calculus)
-- ============================================================================
-- Trend = sign(dy), persistence = consecutive same-sign dy

CREATE OR REPLACE VIEW v_trend AS
SELECT
    signal_id,
    I,
    y,
    dy,
    CASE
        WHEN dy > 0.001 THEN 'up'
        WHEN dy < -0.001 THEN 'down'
        ELSE 'flat'
    END AS trend_direction,
    -- Trend strength from curvature
    CASE
        WHEN ABS(dy) > 0 THEN ABS(d2y) / ABS(dy)
        ELSE NULL
    END AS trend_acceleration
FROM v_d2y;


-- ============================================================================
-- 007: VOLATILITY PROXY (from d2y)
-- ============================================================================
-- Local volatility = |d2y| or |kappa|

CREATE OR REPLACE VIEW v_volatility_proxy AS
SELECT
    c.signal_id,
    c.I,
    c.y,
    c.d2y,
    c.kappa,
    ABS(c.d2y) AS instantaneous_volatility,
    c.kappa AS curvature_volatility,
    -- Compare to signal's baseline
    ABS(c.d2y) / NULLIF(ds.d2y_median_abs, 0) AS relative_volatility
FROM v_curvature c
JOIN v_derivative_stats ds USING (signal_id);


-- ============================================================================
-- 008: AUTOCORRELATION AT SPECIFIC LAGS (KEEP: inherently needs lag)
-- ============================================================================
-- ACF at lags 1, 5, 10 - this requires comparison across samples

CREATE OR REPLACE VIEW v_autocorrelation AS
SELECT
    signal_id,
    CORR(y, y_lag1) AS acf_lag1,
    CORR(y, y_lag5) AS acf_lag5,
    CORR(y, y_lag10) AS acf_lag10
FROM (
    SELECT
        signal_id,
        y,
        LAG(y, 1) OVER (PARTITION BY signal_id ORDER BY I) AS y_lag1,
        LAG(y, 5) OVER (PARTITION BY signal_id ORDER BY I) AS y_lag5,
        LAG(y, 10) OVER (PARTITION BY signal_id ORDER BY I) AS y_lag10
    FROM v_base
)
WHERE y_lag10 IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- STATISTICS SUMMARY VIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_statistics_complete AS
SELECT
    sg.signal_id,
    sg.n_points,
    sg.y_mean,
    sg.y_std,
    sg.y_median,
    sg.y_iqr,
    ds.dy_median_abs,
    ds.d2y_median_abs,
    ac.acf_lag1,
    ac.acf_lag5,
    ac.acf_lag10
FROM v_stats_global sg
LEFT JOIN v_derivative_stats ds USING (signal_id)
LEFT JOIN v_autocorrelation ac USING (signal_id);
