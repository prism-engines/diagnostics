-- ============================================================================
-- ORTHON SQL ENGINES: SIGNAL TYPOLOGY
-- ============================================================================
-- Behavioral classification: trending, mean-reverting, random, chaotic, etc.
-- Uses statistics + calculus + PRISM primitives (if available)
-- ============================================================================

-- ============================================================================
-- 001: TREND DETECTION (from derivative sign)
-- ============================================================================

CREATE OR REPLACE VIEW v_trend_detection AS
SELECT
    signal_id,
    
    -- Fraction of positive derivatives (uptrend indicator)
    SUM(CASE WHEN dy > 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS frac_positive_dy,
    
    -- Fraction of negative derivatives (downtrend indicator)
    SUM(CASE WHEN dy < 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) AS frac_negative_dy,
    
    -- Net trend direction
    AVG(SIGN(dy)) AS trend_direction,
    
    -- Trend strength (mean of dy relative to std)
    AVG(dy) / NULLIF(STDDEV(dy), 0) AS trend_strength,
    
    -- Monotonicity (how consistently one direction)
    ABS(SUM(CASE WHEN dy > 0 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0) - 0.5) * 2 AS monotonicity
    
FROM v_dy
WHERE dy IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- 002: MEAN REVERSION DETECTION
-- ============================================================================
-- Mean reverting signals cross their mean frequently

CREATE OR REPLACE VIEW v_mean_reversion AS
WITH mean_calc AS (
    SELECT signal_id, AVG(y) AS y_mean FROM v_base GROUP BY signal_id
),
deviations AS (
    SELECT 
        b.signal_id,
        b.I,
        b.y - m.y_mean AS deviation,
        SIGN(b.y - m.y_mean) AS side_of_mean
    FROM v_base b
    JOIN mean_calc m USING (signal_id)
),
crossings AS (
    SELECT
        signal_id,
        I,
        CASE 
            WHEN side_of_mean != LAG(side_of_mean) OVER (PARTITION BY signal_id ORDER BY I)
            THEN 1 ELSE 0 
        END AS mean_crossing
    FROM deviations
)
SELECT
    signal_id,
    SUM(mean_crossing) AS n_mean_crossings,
    COUNT(*) AS n_points,
    SUM(mean_crossing)::FLOAT / NULLIF(COUNT(*), 0) AS crossing_rate,
    CASE
        WHEN SUM(mean_crossing)::FLOAT / NULLIF(COUNT(*), 0) > 0.1 THEN TRUE
        ELSE FALSE
    END AS is_mean_reverting
FROM crossings
GROUP BY signal_id;


-- ============================================================================
-- 003: STATIONARITY INDICATORS
-- ============================================================================
-- Compare early vs late statistics

CREATE OR REPLACE VIEW v_stationarity AS
WITH quartiles AS (
    SELECT 
        signal_id,
        y,
        NTILE(4) OVER (PARTITION BY signal_id ORDER BY I) AS quartile
    FROM v_base
),
quartile_stats AS (
    SELECT
        signal_id,
        quartile,
        AVG(y) AS q_mean,
        STDDEV(y) AS q_std
    FROM quartiles
    GROUP BY signal_id, quartile
)
SELECT
    signal_id,
    MAX(CASE WHEN quartile = 1 THEN q_mean END) AS mean_q1,
    MAX(CASE WHEN quartile = 4 THEN q_mean END) AS mean_q4,
    MAX(CASE WHEN quartile = 1 THEN q_std END) AS std_q1,
    MAX(CASE WHEN quartile = 4 THEN q_std END) AS std_q4,
    ABS(MAX(CASE WHEN quartile = 4 THEN q_mean END) - MAX(CASE WHEN quartile = 1 THEN q_mean END)) 
        / NULLIF(STDDEV(q_mean), 0) AS mean_drift,
    ABS(MAX(CASE WHEN quartile = 4 THEN q_std END) - MAX(CASE WHEN quartile = 1 THEN q_std END)) 
        / NULLIF(AVG(q_std), 0) AS volatility_drift,
    CASE
        WHEN ABS(MAX(CASE WHEN quartile = 4 THEN q_mean END) - MAX(CASE WHEN quartile = 1 THEN q_mean END)) 
            / NULLIF(STDDEV(q_mean), 0) < 1.0
        AND ABS(MAX(CASE WHEN quartile = 4 THEN q_std END) - MAX(CASE WHEN quartile = 1 THEN q_std END)) 
            / NULLIF(AVG(q_std), 0) < 0.5
        THEN TRUE
        ELSE FALSE
    END AS is_stationary
FROM quartile_stats
GROUP BY signal_id;


-- ============================================================================
-- 004: VOLATILITY CLUSTERING DETECTION
-- ============================================================================
-- High volatility tends to follow high volatility (GARCH effect)

CREATE OR REPLACE VIEW v_volatility_clustering AS
WITH volatility AS (
    SELECT 
        signal_id,
        I,
        ABS(dy) AS local_volatility
    FROM v_dy
    WHERE dy IS NOT NULL
)
SELECT
    a.signal_id,
    CORR(a.local_volatility, b.local_volatility) AS volatility_autocorr,
    CASE
        WHEN CORR(a.local_volatility, b.local_volatility) > 0.3 THEN TRUE
        ELSE FALSE
    END AS has_volatility_clustering
FROM volatility a
JOIN volatility b ON a.signal_id = b.signal_id AND a.I = b.I + 1
GROUP BY a.signal_id;


-- ============================================================================
-- 005: BURST DETECTION
-- ============================================================================
-- Detect sudden increases in activity/volatility

CREATE OR REPLACE VIEW v_burst_detection AS
WITH rolling_vol AS (
    SELECT
        signal_id,
        I,
        STDDEV(y) OVER w AS rolling_std
    FROM v_base
    WINDOW w AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)
),
vol_stats AS (
    SELECT signal_id, AVG(rolling_std) AS mean_std, STDDEV(rolling_std) AS std_of_std
    FROM rolling_vol GROUP BY signal_id
)
SELECT
    r.signal_id,
    r.I,
    r.rolling_std,
    (r.rolling_std - v.mean_std) / NULLIF(v.std_of_std, 0) AS volatility_zscore,
    CASE 
        WHEN (r.rolling_std - v.mean_std) / NULLIF(v.std_of_std, 0) > 2 THEN 'burst'
        WHEN (r.rolling_std - v.mean_std) / NULLIF(v.std_of_std, 0) < -1 THEN 'calm'
        ELSE 'normal'
    END AS volatility_state
FROM rolling_vol r
JOIN vol_stats v USING (signal_id);


-- ============================================================================
-- 006: PERSISTENCE CLASSIFICATION (uses autocorrelation as proxy)
-- ============================================================================
-- If PRISM primitives table exists and has hurst, it will be joined later.
-- This view uses autocorrelation as the base proxy for persistence.

CREATE OR REPLACE VIEW v_autocorr_lag1 AS
SELECT
    a.signal_id,
    CORR(a.y, b.y) AS autocorr_lag1
FROM v_base a
JOIN v_base b ON a.signal_id = b.signal_id AND a.I = b.I + 1
GROUP BY a.signal_id;

CREATE OR REPLACE VIEW v_persistence AS
SELECT
    signal_id,

    -- Autocorrelation-based proxy for persistence
    autocorr_lag1,

    -- Classification based on autocorrelation
    CASE
        WHEN autocorr_lag1 > 0.5 THEN 'trending'
        WHEN autocorr_lag1 < -0.2 THEN 'mean_reverting'
        ELSE 'random'
    END AS persistence_class,

    'autocorr_proxy' AS persistence_source

FROM v_autocorr_lag1;


-- ============================================================================
-- 007: REGIME STATE CLASSIFICATION
-- ============================================================================
-- Current state within detected regimes

CREATE OR REPLACE VIEW v_regime_state AS
WITH rolling_stats AS (
    SELECT
        signal_id,
        I,
        y,
        AVG(y) OVER w AS rolling_mean,
        STDDEV(y) OVER w AS rolling_std
    FROM v_base
    WINDOW w AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 50 PRECEDING AND CURRENT ROW)
),
global_stats AS (
    SELECT signal_id, AVG(y) AS global_mean, STDDEV(y) AS global_std
    FROM v_base GROUP BY signal_id
)
SELECT
    r.signal_id,
    r.I,
    
    -- Mean-relative state
    CASE
        WHEN r.rolling_mean > g.global_mean + g.global_std THEN 'elevated'
        WHEN r.rolling_mean < g.global_mean - g.global_std THEN 'depressed'
        ELSE 'normal'
    END AS mean_state,
    
    -- Volatility-relative state
    CASE
        WHEN r.rolling_std > g.global_std * 1.5 THEN 'high_volatility'
        WHEN r.rolling_std < g.global_std * 0.5 THEN 'low_volatility'
        ELSE 'normal_volatility'
    END AS volatility_state,
    
    -- Combined regime state
    CASE
        WHEN r.rolling_mean > g.global_mean + g.global_std 
             AND r.rolling_std > g.global_std * 1.5 THEN 'crisis'
        WHEN r.rolling_mean > g.global_mean + g.global_std THEN 'elevated_stable'
        WHEN r.rolling_std > g.global_std * 1.5 THEN 'volatile'
        WHEN r.rolling_mean < g.global_mean - g.global_std THEN 'depressed'
        ELSE 'stable'
    END AS regime_state

FROM rolling_stats r
JOIN global_stats g USING (signal_id);


-- ============================================================================
-- 008: CHAOS INDICATORS (proxy without Lyapunov)
-- ============================================================================
-- High sensitivity approximation from local divergence rates

CREATE OR REPLACE VIEW v_chaos_proxy AS
SELECT
    signal_id,
    AVG(ABS(d2y) / NULLIF(ABS(dy), 0)) AS sensitivity_ratio,
    STDDEV(kappa) / NULLIF(AVG(kappa), 0) AS curvature_variability,
    CASE
        WHEN AVG(ABS(d2y) / NULLIF(ABS(dy), 0)) > 2 
             AND STDDEV(kappa) / NULLIF(AVG(kappa), 0) > 1.5 
        THEN TRUE
        ELSE FALSE
    END AS chaos_suspected
FROM v_curvature
WHERE dy IS NOT NULL AND d2y IS NOT NULL AND kappa IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- 009: FINAL TYPOLOGY CLASSIFICATION
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_typology AS
SELECT
    sc.signal_id,
    sc.signal_class,
    
    -- Behavioral type (primary classification)
    CASE
        -- Periodic overrides
        WHEN sc.is_periodic THEN 'periodic'
        
        -- Digital/Event have their own types
        WHEN sc.signal_class = 'digital' THEN 'discrete'
        WHEN sc.signal_class = 'event' THEN 'event'
        
        -- Chaos suspected
        WHEN cp.chaos_suspected THEN 'chaotic'
        
        -- Persistence-based
        WHEN p.persistence_class = 'trending' AND t.monotonicity > 0.6 THEN 'trending_strong'
        WHEN p.persistence_class = 'trending' THEN 'trending_weak'
        WHEN p.persistence_class = 'mean_reverting' AND mr.is_mean_reverting THEN 'mean_reverting'
        WHEN p.persistence_class = 'random' THEN 'random_walk'
        
        -- Stationarity-based fallback
        WHEN st.is_stationary THEN 'stationary'
        ELSE 'unclassified'
    END AS behavioral_type,
    
    -- Sub-characteristics
    sc.is_periodic,
    sc.estimated_period,
    t.trend_direction,
    t.trend_strength,
    t.monotonicity,
    mr.crossing_rate AS mean_crossing_rate,
    st.is_stationary,
    st.mean_drift,
    st.volatility_drift,
    vc.has_volatility_clustering,
    p.persistence_class,
    p.persistence_source,
    cp.chaos_suspected,
    
    -- Placeholders for PRISM data (joined later if available)
    NULL::FLOAT AS hurst,
    NULL::FLOAT AS lyapunov

FROM v_signal_class sc
LEFT JOIN v_trend_detection t USING (signal_id)
LEFT JOIN v_mean_reversion mr USING (signal_id)
LEFT JOIN v_stationarity st USING (signal_id)
LEFT JOIN v_volatility_clustering vc USING (signal_id)
LEFT JOIN v_persistence p USING (signal_id)
LEFT JOIN v_chaos_proxy cp USING (signal_id);


-- ============================================================================
-- 010: PRISM WORK ORDER GENERATION
-- ============================================================================
-- Decides what PRISM needs to compute based on SQL analysis

CREATE OR REPLACE VIEW v_prism_requests AS
SELECT
    sc.signal_id,
    sc.signal_class,

    -- Request Hurst for analog signals where persistence matters
    CASE
        WHEN sc.signal_class = 'analog' AND NOT sc.is_periodic THEN TRUE
        ELSE FALSE
    END AS needs_hurst,

    -- Request Lyapunov if chaos suspected
    CASE
        WHEN cp.chaos_suspected THEN TRUE
        ELSE FALSE
    END AS needs_lyapunov,

    -- Request FFT for periodic signals or frequency analysis
    CASE
        WHEN sc.signal_class = 'periodic' THEN TRUE
        WHEN sc.is_periodic THEN TRUE
        ELSE FALSE
    END AS needs_fft,

    -- Request GARCH if volatility clustering detected
    CASE
        WHEN vc.has_volatility_clustering THEN TRUE
        ELSE FALSE
    END AS needs_garch,

    -- Request wavelet for multi-scale analysis
    CASE
        WHEN sc.signal_class IN ('analog', 'periodic') AND NOT st.is_stationary THEN TRUE
        ELSE FALSE
    END AS needs_wavelet,

    -- Request RQA if chaos suspected and long enough
    CASE
        WHEN cp.chaos_suspected AND sg.n_points > 500 THEN TRUE
        ELSE FALSE
    END AS needs_rqa,

    -- Request sample entropy for complexity
    CASE
        WHEN sc.signal_class = 'analog' AND NOT sc.is_periodic THEN TRUE
        ELSE FALSE
    END AS needs_sample_entropy

FROM v_signal_class sc
LEFT JOIN v_chaos_proxy cp USING (signal_id)
LEFT JOIN v_volatility_clustering vc USING (signal_id)
LEFT JOIN v_stationarity st USING (signal_id)
LEFT JOIN v_stats_global sg USING (signal_id);


-- ============================================================================
-- TYPOLOGY SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_typology_complete AS
SELECT
    t.signal_id,
    t.signal_class,
    t.behavioral_type,
    t.is_periodic,
    t.estimated_period,
    t.trend_direction,
    t.trend_strength,
    t.monotonicity,
    t.mean_crossing_rate,
    t.is_stationary,
    t.has_volatility_clustering,
    t.persistence_class,
    t.chaos_suspected,
    t.hurst,
    t.lyapunov,
    pr.needs_hurst,
    pr.needs_lyapunov,
    pr.needs_fft,
    pr.needs_garch,
    pr.needs_wavelet,
    pr.needs_rqa,
    pr.needs_sample_entropy
FROM v_signal_typology t
LEFT JOIN v_prism_requests pr USING (signal_id);
