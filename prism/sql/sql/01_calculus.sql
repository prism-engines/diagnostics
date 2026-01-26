-- ============================================================================
-- ORTHON SQL ENGINES: CALCULUS
-- ============================================================================
-- These engines compute derivatives, curvature, and related calculus operations.
-- All use LAG/LEAD window functions - the foundation of SQL-based calculus.
-- ============================================================================

-- ============================================================================
-- 001: FIRST DERIVATIVE (dy/dI)
-- ============================================================================
-- Central difference: (y[i+1] - y[i-1]) / 2
-- More accurate than forward/backward difference

CREATE OR REPLACE VIEW v_dy AS
SELECT
    signal_id,
    I,
    y,
    index_dimension,
    signal_class,
    (LEAD(y) OVER (PARTITION BY signal_id ORDER BY I) -
     LAG(y) OVER (PARTITION BY signal_id ORDER BY I)) / 2.0 AS dy
FROM v_base;


-- ============================================================================
-- 002: SECOND DERIVATIVE (d²y/dI²)
-- ============================================================================
-- Second central difference: y[i+1] - 2*y[i] + y[i-1]

CREATE OR REPLACE VIEW v_d2y AS
SELECT
    signal_id,
    I,
    y,
    index_dimension,
    signal_class,
    dy,
    LEAD(y) OVER (PARTITION BY signal_id ORDER BY I) - 2*y +
    LAG(y) OVER (PARTITION BY signal_id ORDER BY I) AS d2y
FROM v_dy;


-- ============================================================================
-- 003: THIRD DERIVATIVE (d³y/dI³) - JERK
-- ============================================================================
-- For detecting sudden changes in acceleration

CREATE OR REPLACE VIEW v_d3y AS
SELECT
    signal_id,
    I,
    y,
    index_dimension,
    signal_class,
    dy,
    d2y,
    (LEAD(d2y) OVER (PARTITION BY signal_id ORDER BY I) -
     LAG(d2y) OVER (PARTITION BY signal_id ORDER BY I)) / 2.0 AS d3y
FROM v_d2y;


-- ============================================================================
-- 004: CURVATURE (κ)
-- ============================================================================
-- κ = |d²y| / (1 + dy²)^(3/2)
-- Measures how fast direction changes along the curve

CREATE OR REPLACE VIEW v_curvature AS
SELECT
    signal_id,
    I,
    y,
    index_dimension,
    signal_class,
    dy,
    d2y,
    ABS(d2y) / POWER(1 + dy*dy + 1e-10, 1.5) AS kappa
FROM v_d2y
WHERE dy IS NOT NULL AND d2y IS NOT NULL;


-- ============================================================================
-- 005: LAPLACIAN (∇²y) - For Spatial Fields
-- ============================================================================
-- Same as second derivative but semantic difference

CREATE OR REPLACE VIEW v_laplacian AS
SELECT
    signal_id,
    I,
    y,
    index_dimension,
    signal_class,
    LEAD(y) OVER (PARTITION BY signal_id ORDER BY I) - 2*y +
    LAG(y) OVER (PARTITION BY signal_id ORDER BY I) AS laplacian
FROM v_base;


-- ============================================================================
-- 006: GRADIENT MAGNITUDE (|∇y|)
-- ============================================================================
-- For 1D: just |dy|

CREATE OR REPLACE VIEW v_gradient AS
SELECT
    signal_id,
    I,
    y,
    index_dimension,
    signal_class,
    dy,
    ABS(dy) AS gradient_magnitude
FROM v_dy;


-- ============================================================================
-- 007: ARC LENGTH (cumulative)
-- ============================================================================
-- s = ∫√(1 + dy²) dI

CREATE OR REPLACE VIEW v_arc_length AS
SELECT
    signal_id,
    I,
    y,
    dy,
    SQRT(1 + dy*dy) AS segment_length,
    SUM(SQRT(1 + COALESCE(dy*dy, 0))) OVER (
        PARTITION BY signal_id
        ORDER BY I
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS arc_length_cumulative
FROM v_dy;


-- ============================================================================
-- 008: VELOCITY MAGNITUDE (for phase space)
-- ============================================================================

CREATE OR REPLACE VIEW v_velocity AS
SELECT
    signal_id,
    I,
    y,
    dy AS velocity,
    ABS(dy) AS speed,
    d2y AS acceleration,
    ABS(d2y) AS acceleration_magnitude
FROM v_d2y;


-- ============================================================================
-- 009: DIVERGENCE PROXY (for multi-signal systems)
-- ============================================================================

CREATE OR REPLACE VIEW v_divergence AS
SELECT
    signal_id,
    I,
    d2y AS divergence,
    CASE
        WHEN d2y > 0.1 THEN 'expanding'
        WHEN d2y < -0.1 THEN 'contracting'
        ELSE 'stable'
    END AS divergence_state
FROM v_d2y;


-- ============================================================================
-- 010: SMOOTHNESS INDEX
-- ============================================================================

CREATE OR REPLACE VIEW v_smoothness AS
SELECT
    signal_id,
    I,
    dy,
    d2y,
    CASE
        WHEN ABS(dy) > 1e-10
        THEN ABS(d2y) / ABS(dy)
        ELSE NULL
    END AS roughness_index,
    CASE
        WHEN ABS(dy) > 1e-10
        THEN ABS(dy) / (ABS(d2y) + 1e-10)
        ELSE NULL
    END AS smoothness_index
FROM v_d2y;


-- ============================================================================
-- CALCULUS SUMMARY VIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_calculus_complete AS
SELECT
    c.signal_id,
    c.I,
    c.y,
    c.index_dimension,
    c.signal_class,
    c.dy,
    c.d2y,
    d.d3y,
    c.kappa,
    l.laplacian,
    ABS(c.dy) AS gradient_magnitude,
    a.arc_length_cumulative,
    s.roughness_index,
    s.smoothness_index,
    div.divergence,
    div.divergence_state
FROM v_curvature c
LEFT JOIN v_d3y d USING (signal_id, I)
LEFT JOIN v_laplacian l USING (signal_id, I)
LEFT JOIN v_arc_length a USING (signal_id, I)
LEFT JOIN v_smoothness s USING (signal_id, I)
LEFT JOIN v_divergence div USING (signal_id, I);
