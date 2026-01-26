-- ============================================================================
-- ORTHON SQL ENGINES: BEHAVIORAL GEOMETRY
-- ============================================================================
-- Coupling, correlation, networks, and signal relationships.
-- This is where we understand how signals relate to each other.
-- ============================================================================

-- ============================================================================
-- 001: PAIRWISE CORRELATION MATRIX
-- ============================================================================

CREATE OR REPLACE VIEW v_correlation_matrix AS
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    CORR(a.y, b.y) AS pearson_correlation,
    COUNT(*) AS n_overlap
FROM v_base a
JOIN v_base b ON a.I = b.I AND a.signal_id < b.signal_id
GROUP BY a.signal_id, b.signal_id
HAVING COUNT(*) > 50;


-- ============================================================================
-- 002: LAGGED CORRELATION (all lags 1-20)
-- ============================================================================

CREATE OR REPLACE VIEW v_lagged_correlation AS
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    1 AS lag,
    CORR(a.y, b.y) AS correlation
FROM v_base a
JOIN v_base b ON a.signal_id != b.signal_id AND a.I = b.I + 1
GROUP BY a.signal_id, b.signal_id
UNION ALL
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    5 AS lag,
    CORR(a.y, b.y) AS correlation
FROM v_base a
JOIN v_base b ON a.signal_id != b.signal_id AND a.I = b.I + 5
GROUP BY a.signal_id, b.signal_id
UNION ALL
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    10 AS lag,
    CORR(a.y, b.y) AS correlation
FROM v_base a
JOIN v_base b ON a.signal_id != b.signal_id AND a.I = b.I + 10
GROUP BY a.signal_id, b.signal_id
UNION ALL
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    15 AS lag,
    CORR(a.y, b.y) AS correlation
FROM v_base a
JOIN v_base b ON a.signal_id != b.signal_id AND a.I = b.I + 15
GROUP BY a.signal_id, b.signal_id
UNION ALL
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    20 AS lag,
    CORR(a.y, b.y) AS correlation
FROM v_base a
JOIN v_base b ON a.signal_id != b.signal_id AND a.I = b.I + 20
GROUP BY a.signal_id, b.signal_id;


-- ============================================================================
-- 003: OPTIMAL LAG DETECTION
-- ============================================================================
-- Find the lag with maximum correlation for each pair

CREATE OR REPLACE VIEW v_optimal_lag AS
WITH all_lags AS (
    SELECT * FROM v_lagged_correlation
),
ranked AS (
    SELECT
        signal_a,
        signal_b,
        lag,
        correlation,
        ABS(correlation) AS abs_corr,
        ROW_NUMBER() OVER (PARTITION BY signal_a, signal_b ORDER BY ABS(correlation) DESC) AS rank
    FROM all_lags
)
SELECT
    signal_a,
    signal_b,
    lag AS optimal_lag,
    correlation AS optimal_correlation,
    abs_corr AS optimal_abs_correlation,
    CASE
        WHEN correlation > 0.7 THEN 'strong_positive'
        WHEN correlation > 0.3 THEN 'moderate_positive'
        WHEN correlation < -0.7 THEN 'strong_negative'
        WHEN correlation < -0.3 THEN 'moderate_negative'
        ELSE 'weak'
    END AS coupling_strength
FROM ranked
WHERE rank = 1;


-- ============================================================================
-- 004: LEAD-LAG RELATIONSHIP DETECTION
-- ============================================================================
-- Determine which signal leads and which follows

CREATE OR REPLACE VIEW v_lead_lag AS
WITH forward_corr AS (
    SELECT
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.y, b.y) AS corr_forward
    FROM v_base a
    JOIN v_base b ON a.signal_id != b.signal_id AND a.I = b.I + 10
    GROUP BY a.signal_id, b.signal_id
),
backward_corr AS (
    SELECT
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.y, b.y) AS corr_backward
    FROM v_base a
    JOIN v_base b ON a.signal_id != b.signal_id AND a.I = b.I - 10
    GROUP BY a.signal_id, b.signal_id
)
SELECT
    f.signal_a,
    f.signal_b,
    f.corr_forward,
    b.corr_backward,
    f.corr_forward - b.corr_backward AS lead_lag_asymmetry,
    CASE
        WHEN f.corr_forward > b.corr_backward + 0.1 THEN 'A_leads_B'
        WHEN b.corr_backward > f.corr_forward + 0.1 THEN 'B_leads_A'
        ELSE 'contemporaneous'
    END AS lead_lag_direction
FROM forward_corr f
JOIN backward_corr b USING (signal_a, signal_b);


-- ============================================================================
-- 005: COUPLING NETWORK EDGES
-- ============================================================================
-- Create edges for network visualization

CREATE OR REPLACE VIEW v_coupling_network AS
SELECT
    signal_a AS source,
    signal_b AS target,
    optimal_correlation AS weight,
    optimal_abs_correlation AS abs_weight,
    optimal_lag AS lag,
    coupling_strength,
    CASE
        WHEN optimal_correlation > 0 THEN 'positive'
        ELSE 'negative'
    END AS edge_type
FROM v_optimal_lag
WHERE optimal_abs_correlation > 0.3;  -- Only significant couplings


-- ============================================================================
-- 006: NODE CENTRALITY (degree-based)
-- ============================================================================
-- How connected is each signal?

CREATE OR REPLACE VIEW v_node_degree AS
SELECT
    signal_id,
    COUNT(*) AS degree,
    SUM(abs_weight) AS weighted_degree,
    AVG(abs_weight) AS avg_coupling_strength
FROM (
    SELECT source AS signal_id, abs_weight FROM v_coupling_network
    UNION ALL
    SELECT target AS signal_id, abs_weight FROM v_coupling_network
) combined
GROUP BY signal_id;


-- ============================================================================
-- 007: INCOMING VS OUTGOING (for directed relationships)
-- ============================================================================

CREATE OR REPLACE VIEW v_directional_degree AS
WITH leads AS (
    SELECT signal_a AS signal_id, COUNT(*) AS n_leads
    FROM v_lead_lag
    WHERE lead_lag_direction = 'A_leads_B'
    GROUP BY signal_a
),
follows AS (
    SELECT signal_b AS signal_id, COUNT(*) AS n_follows
    FROM v_lead_lag
    WHERE lead_lag_direction = 'A_leads_B'
    GROUP BY signal_b
)
SELECT
    COALESCE(l.signal_id, f.signal_id) AS signal_id,
    COALESCE(l.n_leads, 0) AS out_degree,
    COALESCE(f.n_follows, 0) AS in_degree,
    COALESCE(l.n_leads, 0) - COALESCE(f.n_follows, 0) AS net_influence,
    CASE
        WHEN COALESCE(l.n_leads, 0) > COALESCE(f.n_follows, 0) * 2 THEN 'driver'
        WHEN COALESCE(f.n_follows, 0) > COALESCE(l.n_leads, 0) * 2 THEN 'follower'
        ELSE 'bidirectional'
    END AS influence_type
FROM leads l
FULL OUTER JOIN follows f USING (signal_id);


-- ============================================================================
-- 008: SIGNAL CLUSTERS (based on correlation)
-- ============================================================================
-- Group highly correlated signals together

CREATE OR REPLACE VIEW v_correlation_clusters AS
WITH strong_edges AS (
    SELECT signal_a, signal_b
    FROM v_optimal_lag
    WHERE optimal_abs_correlation > 0.6
),
-- Simple connected components via self-join (limited iterations)
components AS (
    SELECT signal_a AS signal_id, signal_a AS cluster_root FROM strong_edges
    UNION
    SELECT signal_b AS signal_id, signal_a AS cluster_root FROM strong_edges
)
SELECT
    signal_id,
    MIN(cluster_root) AS cluster_id,
    COUNT(*) OVER (PARTITION BY MIN(cluster_root)) AS cluster_size
FROM components
GROUP BY signal_id;


-- ============================================================================
-- 009: DERIVATIVE CORRELATION (velocity coupling)
-- ============================================================================
-- Do the rates of change correlate?

CREATE OR REPLACE VIEW v_derivative_correlation AS
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    CORR(a.dy, b.dy) AS velocity_correlation,
    CORR(a.d2y, b.d2y) AS acceleration_correlation
FROM v_d2y a
JOIN v_d2y b ON a.I = b.I AND a.signal_id < b.signal_id
WHERE a.dy IS NOT NULL AND b.dy IS NOT NULL
GROUP BY a.signal_id, b.signal_id
HAVING COUNT(*) > 50;


-- ============================================================================
-- 010: COVARIANCE MATRIX
-- ============================================================================

CREATE OR REPLACE VIEW v_covariance_matrix AS
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    AVG((a.y - a_stats.y_mean) * (b.y - b_stats.y_mean)) AS covariance
FROM v_base a
JOIN v_base b ON a.I = b.I AND a.signal_id < b.signal_id
JOIN v_stats_global a_stats ON a.signal_id = a_stats.signal_id
JOIN v_stats_global b_stats ON b.signal_id = b_stats.signal_id
GROUP BY a.signal_id, b.signal_id, a_stats.y_mean, b_stats.y_mean;


-- ============================================================================
-- 011: PARTIAL CORRELATION PROXY
-- ============================================================================
-- Direct coupling after removing common influences
-- (Simplified: correlation of residuals after regressing on system mean)

CREATE OR REPLACE VIEW v_partial_correlation_proxy AS
WITH system_mean AS (
    SELECT I, AVG(y) AS y_system_mean FROM v_base GROUP BY I
),
residuals AS (
    SELECT
        b.signal_id,
        b.I,
        b.y - sm.y_system_mean AS residual
    FROM v_base b
    JOIN system_mean sm USING (I)
)
SELECT
    a.signal_id AS signal_a,
    b.signal_id AS signal_b,
    CORR(a.residual, b.residual) AS partial_correlation_proxy
FROM residuals a
JOIN residuals b ON a.I = b.I AND a.signal_id < b.signal_id
GROUP BY a.signal_id, b.signal_id;


-- ============================================================================
-- 012: MUTUAL INFORMATION PROXY
-- ============================================================================
-- Binned approximation of mutual information

CREATE OR REPLACE VIEW v_mutual_info_proxy AS
WITH binned AS (
    SELECT
        signal_id,
        I,
        NTILE(10) OVER (PARTITION BY signal_id ORDER BY y) AS bin
    FROM v_base
),
joint_bins AS (
    SELECT
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        a.bin AS bin_a,
        b.bin AS bin_b,
        COUNT(*) AS joint_count
    FROM binned a
    JOIN binned b ON a.I = b.I AND a.signal_id < b.signal_id
    GROUP BY a.signal_id, b.signal_id, a.bin, b.bin
),
marginal_a AS (
    SELECT signal_a, bin_a, SUM(joint_count) AS marginal_count
    FROM joint_bins GROUP BY signal_a, bin_a
),
marginal_b AS (
    SELECT signal_b, bin_b, SUM(joint_count) AS marginal_count
    FROM joint_bins GROUP BY signal_b, bin_b
),
total_count AS (
    SELECT signal_a, signal_b, SUM(joint_count) AS n FROM joint_bins GROUP BY signal_a, signal_b
)
SELECT
    j.signal_a,
    j.signal_b,
    SUM(
        CASE WHEN j.joint_count > 0 THEN
            (j.joint_count::FLOAT / t.n) * 
            LN((j.joint_count::FLOAT * t.n) / (ma.marginal_count::FLOAT * mb.marginal_count::FLOAT))
        ELSE 0 END
    ) AS mutual_information_proxy
FROM joint_bins j
JOIN marginal_a ma ON j.signal_a = ma.signal_a AND j.bin_a = ma.bin_a
JOIN marginal_b mb ON j.signal_b = mb.signal_b AND j.bin_b = mb.bin_b
JOIN total_count t ON j.signal_a = t.signal_a AND j.signal_b = t.signal_b
GROUP BY j.signal_a, j.signal_b;


-- ============================================================================
-- GEOMETRY SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_geometry_complete AS
SELECT
    ol.signal_a,
    ol.signal_b,
    cm.pearson_correlation AS instant_correlation,
    ol.optimal_lag,
    ol.optimal_correlation,
    ol.coupling_strength,
    ll.lead_lag_direction,
    ll.lead_lag_asymmetry,
    dc.velocity_correlation,
    dc.acceleration_correlation,
    pc.partial_correlation_proxy,
    mi.mutual_information_proxy
FROM v_optimal_lag ol
LEFT JOIN v_correlation_matrix cm USING (signal_a, signal_b)
LEFT JOIN v_lead_lag ll USING (signal_a, signal_b)
LEFT JOIN v_derivative_correlation dc USING (signal_a, signal_b)
LEFT JOIN v_partial_correlation_proxy pc USING (signal_a, signal_b)
LEFT JOIN v_mutual_info_proxy mi USING (signal_a, signal_b);
