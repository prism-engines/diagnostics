-- ============================================================================
-- ORTHON SQL ENGINES: MANIFOLD ASSEMBLY & EXPORT
-- ============================================================================
-- Final assembly of all layers into phase space representation.
-- Export views for visualization and downstream consumption.
-- ============================================================================

-- ============================================================================
-- 001: PHASE SPACE COORDINATES (from PRISM UMAP or local approximation)
-- ============================================================================

CREATE OR REPLACE VIEW v_phase_space AS
SELECT
    b.signal_id,
    b.I,
    b.y,

    -- Fallback: local approximation using y, dy, d2y as coordinates
    -- (PRISM UMAP coordinates can be joined later if available)
    NULL::FLOAT AS x,
    NULL::FLOAT AS y_coord,
    NULL::FLOAT AS z,

    c.y AS phase_x,
    COALESCE(c.dy * 10, 0) AS phase_y,  -- scaled for visibility
    COALESCE(c.kappa * 100, 0) AS phase_z,

    -- Source info
    'local' AS coord_source

FROM v_base b
LEFT JOIN v_curvature c ON b.signal_id = c.signal_id AND b.I = c.I;


-- ============================================================================
-- 002: TRAJECTORY ASSEMBLY
-- ============================================================================
-- Each signal's path through phase space

-- First compute phase velocity, then cumulative arc length
CREATE OR REPLACE VIEW v_trajectory_base AS
SELECT
    ps.signal_id,
    ps.I,
    ps.phase_x,
    ps.phase_y,
    ps.phase_z,
    sc.signal_class,
    st.behavioral_type,
    ra.regime_id,
    rs.regime_mean,
    rs.regime_std,
    -- Velocity in phase space
    SQRT(
        COALESCE(POWER(ps.phase_x - LAG(ps.phase_x) OVER w, 2), 0) +
        COALESCE(POWER(ps.phase_y - LAG(ps.phase_y) OVER w, 2), 0) +
        COALESCE(POWER(ps.phase_z - LAG(ps.phase_z) OVER w, 2), 0)
    ) AS phase_velocity
FROM v_phase_space ps
LEFT JOIN v_signal_class sc USING (signal_id)
LEFT JOIN v_signal_typology st USING (signal_id)
LEFT JOIN v_regime_assignment ra ON ps.signal_id = ra.signal_id AND ps.I = ra.I
LEFT JOIN v_regime_stats rs ON ra.signal_id = rs.signal_id AND ra.regime_id = rs.regime_id
WINDOW w AS (PARTITION BY ps.signal_id ORDER BY ps.I);

CREATE OR REPLACE VIEW v_trajectory AS
SELECT
    signal_id,
    I,
    phase_x,
    phase_y,
    phase_z,
    signal_class,
    behavioral_type,
    regime_id,
    regime_mean,
    regime_std,
    phase_velocity,
    -- Arc length (cumulative distance traveled)
    SUM(COALESCE(phase_velocity, 0)) OVER (PARTITION BY signal_id ORDER BY I) AS arc_length
FROM v_trajectory_base;


-- ============================================================================
-- 003: SIGNAL SUMMARY (one row per signal)
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_summary AS
SELECT
    sc.signal_id,
    sc.signal_class,
    sc.value_unit,
    sc.interpolation_valid,
    sc.is_periodic,
    sc.estimated_period,
    
    st.behavioral_type,
    st.persistence_class,
    st.chaos_suspected,
    
    sg.n_points,
    sg.y_mean,
    sg.y_std,
    sg.y_min,
    sg.y_max,
    
    cr.causal_role,
    cr.out_degree AS n_drives,
    cr.in_degree AS n_driven_by,
    
    ec.shannon_entropy,
    ec.normalized_entropy,
    ec.permutation_entropy,
    
    -- Aggregate regime info
    (SELECT COUNT(DISTINCT regime_id) FROM v_regime_assignment ra WHERE ra.signal_id = sc.signal_id) AS n_regimes,

    -- PRISM results (placeholders - join with primitives table if available)
    NULL::FLOAT AS hurst,
    NULL::FLOAT AS lyapunov,
    NULL::FLOAT AS sample_entropy

FROM v_signal_class sc
LEFT JOIN v_signal_typology st USING (signal_id)
LEFT JOIN v_stats_global sg USING (signal_id)
LEFT JOIN v_causal_roles cr USING (signal_id)
LEFT JOIN v_entropy_complete ec USING (signal_id);


-- ============================================================================
-- 004: REGIME SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_regime_summary AS
SELECT
    rs.signal_id,
    rs.regime_id,
    rs.regime_start,
    rs.regime_end,
    rs.regime_length,
    rs.regime_mean,
    rs.regime_std,
    rs.regime_min,
    rs.regime_max,
    rs.regime_avg_velocity,
    rs.regime_avg_curvature,
    
    -- Transition info
    rt.transition_type,
    rt.mean_jump,
    rt.volatility_ratio
    
FROM v_regime_stats rs
LEFT JOIN v_regime_transitions rt ON rs.signal_id = rt.signal_id AND rs.regime_id = rt.from_regime;


-- ============================================================================
-- 005: COUPLING SUMMARY (for network visualization)
-- ============================================================================

CREATE OR REPLACE VIEW v_coupling_summary AS
SELECT
    gc.signal_a AS source,
    gc.signal_b AS target,
    gc.instant_correlation,
    gc.optimal_lag,
    gc.optimal_correlation,
    gc.coupling_strength,
    gc.lead_lag_direction,
    gc.velocity_correlation,
    gc.partial_correlation_proxy,
    gc.mutual_information_proxy,
    
    cs.combined_causal_strength,
    cs.granger_score,
    
    CASE
        WHEN cs.combined_causal_strength > 0.5 THEN 'strong_causal'
        WHEN gc.optimal_correlation > 0.7 THEN 'strong_coupling'
        WHEN cs.combined_causal_strength > 0.2 THEN 'moderate_causal'
        WHEN gc.optimal_correlation > 0.3 THEN 'moderate_coupling'
        ELSE 'weak'
    END AS relationship_strength

FROM v_geometry_complete gc
LEFT JOIN v_causal_strength cs ON gc.signal_a = cs.source AND gc.signal_b = cs.target;


-- ============================================================================
-- 006: SYSTEM SUMMARY (aggregate view)
-- ============================================================================

CREATE OR REPLACE VIEW v_system_summary AS
SELECT
    -- Signal counts
    (SELECT COUNT(DISTINCT signal_id) FROM v_base) AS n_signals,
    (SELECT COUNT(*) FROM v_base) AS n_total_points,
    (SELECT MIN(I) FROM v_base) AS I_min,
    (SELECT MAX(I) FROM v_base) AS I_max,
    
    -- Classification breakdown
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'analog') AS n_analog,
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'digital') AS n_digital,
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'periodic') AS n_periodic,
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'event') AS n_event,
    
    -- Typology breakdown
    (SELECT COUNT(*) FROM v_signal_typology WHERE behavioral_type LIKE 'trending%') AS n_trending,
    (SELECT COUNT(*) FROM v_signal_typology WHERE behavioral_type = 'mean_reverting') AS n_mean_reverting,
    (SELECT COUNT(*) FROM v_signal_typology WHERE behavioral_type = 'chaotic') AS n_chaotic,
    
    -- Causal structure
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'SOURCE') AS n_sources,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'SINK') AS n_sinks,
    (SELECT COUNT(*) FROM v_causal_roles WHERE causal_role = 'CONDUIT') AS n_conduits,
    
    -- System state
    (SELECT COUNT(DISTINCT regime_id) FROM v_regime_assignment) AS total_regimes,
    (SELECT AVG(shannon_entropy) FROM v_shannon_entropy) AS avg_system_entropy,
    
    -- Physics consistency
    (SELECT SUM(n_violations)::FLOAT / NULLIF(SUM(n_checks), 0) FROM v_physics_complete) AS physics_violation_rate;


-- ============================================================================
-- 007: ALERTS & ANOMALIES
-- ============================================================================

CREATE OR REPLACE VIEW v_alerts AS
-- Regime changes
SELECT
    'regime_change' AS alert_type,
    signal_id,
    I AS alert_at,
    'Regime change detected' AS description,
    change_score AS severity
FROM v_regime_changes
WHERE is_regime_change

UNION ALL

-- Chaos suspected
SELECT
    'chaos_suspected' AS alert_type,
    signal_id,
    NULL::FLOAT AS alert_at,
    'Chaotic behavior suspected' AS description,
    sensitivity_ratio AS severity
FROM v_chaos_proxy
WHERE chaos_suspected

UNION ALL

-- Bifurcation candidates
SELECT
    'bifurcation' AS alert_type,
    signal_id,
    bifurcation_point AS alert_at,
    'Possible bifurcation: ' || COALESCE(bifurcation_type, 'unknown') AS description,
    kappa_jump AS severity
FROM v_bifurcation_candidates
WHERE bifurcation_type IS NOT NULL

UNION ALL

-- Physics violations
SELECT
    'physics_violation' AS alert_type,
    NULL AS signal_id,
    I AS alert_at,
    'Conservation violation detected' AS description,
    ABS(deviation_zscore) AS severity
FROM v_conservation_check
WHERE conservation_status = 'conservation_violation'

UNION ALL

-- Extreme outliers
SELECT
    'extreme_outlier' AS alert_type,
    signal_id,
    I AS alert_at,
    'Extreme value detected (z > 3)' AS description,
    ABS(z_score) AS severity
FROM v_zscore
WHERE z_category = 'extreme';


-- ============================================================================
-- 008: EXPORT: FINAL PARQUET SCHEMAS
-- ============================================================================

-- Signal Class Output
CREATE OR REPLACE VIEW v_export_signal_class AS
SELECT
    signal_id,
    signal_class,
    class_source,
    value_unit,
    index_dimension,
    interpolation_valid,
    is_periodic,
    estimated_period,
    max_derivative_order,
    unique_ratio,
    sparsity,
    kappa_consistency
FROM v_classification_complete;

-- Signal Typology Output
CREATE OR REPLACE VIEW v_export_signal_typology AS
SELECT
    st.signal_id,
    b.I,
    b.y,
    st.signal_class,
    st.behavioral_type,
    st.persistence_class,
    st.is_periodic,
    st.is_stationary,
    st.has_volatility_clustering,
    st.chaos_suspected,
    st.hurst,
    st.lyapunov
FROM v_signal_typology st
JOIN v_base b ON st.signal_id = b.signal_id;

-- Behavioral Geometry Output
CREATE OR REPLACE VIEW v_export_behavioral_geometry AS
SELECT
    source,
    target,
    instant_correlation,
    optimal_lag,
    optimal_correlation,
    coupling_strength,
    lead_lag_direction,
    velocity_correlation,
    mutual_information_proxy AS mutual_information,
    combined_causal_strength
FROM v_coupling_summary;

-- Dynamical Systems Output
CREATE OR REPLACE VIEW v_export_dynamical_systems AS
SELECT
    signal_id,
    I,
    regime_id,
    regime_mean,
    regime_std,
    stability_state,
    is_locally_stable,
    phase_velocity
FROM v_dynamics_complete;

-- Causal Mechanics Output
CREATE OR REPLACE VIEW v_export_causal_mechanics AS
SELECT
    signal_id,
    causal_role,
    out_degree AS n_drives,
    in_degree AS n_driven_by,
    net_influence,
    total_causal_flow,
    drives,
    driven_by
FROM v_causality_complete;


-- ============================================================================
-- 009: MANIFOLD JSON EXPORT (for viewer)
-- ============================================================================
-- NOTE: Use these views separately for JSON export in application code
-- DuckDB json_group_array has different semantics than SQLite

-- Simplified: Export signals as JSON array
CREATE OR REPLACE VIEW v_export_signals_json AS
SELECT
    LIST(
        {'id': signal_id, 'class': signal_class, 'type': behavioral_type, 'role': causal_role, 'n_points': n_points}
    ) AS signals_json
FROM v_signal_summary;

-- Export alerts sorted by severity
CREATE OR REPLACE VIEW v_export_alerts_json AS
SELECT
    alert_type,
    signal_id,
    alert_at,
    description,
    severity
FROM v_alerts
ORDER BY severity DESC NULLS LAST
LIMIT 100;


-- ============================================================================
-- 010: RUN_ALL.SQL ASSEMBLY ORDER
-- ============================================================================
-- This documents the order for run_all.sql

/*
EXECUTION ORDER:
================

1. 00_load.sql         -- Create v_base from observations
2. 01_calculus.sql     -- Derivatives, curvature (foundation)
3. 02_statistics.sql   -- Rolling stats, autocorrelation
4. 03_signal_class.sql -- Analog/digital/periodic/event
5. 04_typology.sql     -- Behavioral classification
6. 05_geometry.sql     -- Coupling, correlation, networks
7. 06_dynamics.sql     -- Regimes, transitions, stability
8. 07_causality.sql    -- Granger, causal roles
9. 08_entropy.sql      -- Information theory metrics
10. 09_physics.sql     -- Conservation law checks
11. 10_manifold.sql    -- Final assembly, exports

Each layer depends on previous layers.
Views chain: v_base → v_dy → v_d2y → v_curvature → ...
*/
