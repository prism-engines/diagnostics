-- ============================================================================
-- ORTHON SQL ENGINES: ENTROPY & INFORMATION THEORY
-- ============================================================================
-- Shannon entropy, permutation entropy, complexity measures.
-- These quantify disorder, predictability, and information content.
-- ============================================================================

-- ============================================================================
-- 001: BINNED DISTRIBUTION
-- ============================================================================
-- Foundation for entropy calculations

CREATE OR REPLACE VIEW v_binned_distribution AS
WITH binned AS (
    SELECT
        signal_id,
        NTILE(20) OVER (PARTITION BY signal_id ORDER BY y) AS bin_id
    FROM v_base
),
bin_counts AS (
    SELECT
        signal_id,
        bin_id,
        COUNT(*) AS bin_count
    FROM binned
    GROUP BY signal_id, bin_id
),
totals AS (
    SELECT
        signal_id,
        SUM(bin_count) AS total_count
    FROM bin_counts
    GROUP BY signal_id
)
SELECT
    bc.signal_id,
    bc.bin_id,
    bc.bin_count,
    t.total_count,
    bc.bin_count::FLOAT / t.total_count AS probability
FROM bin_counts bc
JOIN totals t USING (signal_id);


-- ============================================================================
-- 002: SHANNON ENTROPY
-- ============================================================================
-- H = -Σ p(x) * log(p(x))
-- Higher = more disorder/uncertainty

CREATE OR REPLACE VIEW v_shannon_entropy AS
SELECT
    signal_id,
    -SUM(probability * LN(probability + 1e-10)) AS shannon_entropy,
    -SUM(probability * LN(probability + 1e-10)) / LN(20) AS normalized_entropy,  -- 0-1 scale
    CASE
        WHEN -SUM(probability * LN(probability + 1e-10)) / LN(20) > 0.9 THEN 'high_entropy'
        WHEN -SUM(probability * LN(probability + 1e-10)) / LN(20) > 0.5 THEN 'medium_entropy'
        ELSE 'low_entropy'
    END AS entropy_category
FROM v_binned_distribution
GROUP BY signal_id;


-- ============================================================================
-- 003: DERIVATIVE ENTROPY
-- ============================================================================
-- Entropy of the rate of change

CREATE OR REPLACE VIEW v_derivative_entropy AS
WITH binned AS (
    SELECT
        signal_id,
        NTILE(20) OVER (PARTITION BY signal_id ORDER BY dy) AS bin_id
    FROM v_dy
    WHERE dy IS NOT NULL
),
bin_counts AS (
    SELECT
        signal_id,
        bin_id,
        COUNT(*) AS bin_count
    FROM binned
    GROUP BY signal_id, bin_id
),
totals AS (
    SELECT
        signal_id,
        SUM(bin_count) AS total_count
    FROM bin_counts
    GROUP BY signal_id
),
probs AS (
    SELECT
        bc.signal_id,
        bc.bin_id,
        bc.bin_count::FLOAT / t.total_count AS probability
    FROM bin_counts bc
    JOIN totals t USING (signal_id)
)
SELECT
    signal_id,
    -SUM(probability * LN(probability + 1e-10)) AS derivative_entropy
FROM probs
GROUP BY signal_id;


-- ============================================================================
-- 004: JOINT ENTROPY (between signal pairs)
-- ============================================================================
-- H(X,Y) - entropy of joint distribution

CREATE OR REPLACE VIEW v_joint_entropy AS
WITH joint_bins AS (
    SELECT
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        NTILE(10) OVER (PARTITION BY a.signal_id ORDER BY a.y) AS bin_a,
        NTILE(10) OVER (PARTITION BY b.signal_id ORDER BY b.y) AS bin_b,
        COUNT(*) OVER () AS total_count
    FROM v_base a
    JOIN v_base b ON a.I = b.I AND a.signal_id < b.signal_id
),
joint_probs AS (
    SELECT
        signal_a,
        signal_b,
        bin_a,
        bin_b,
        COUNT(*)::FLOAT / MAX(total_count) AS probability
    FROM joint_bins
    GROUP BY signal_a, signal_b, bin_a, bin_b, total_count
)
SELECT
    signal_a,
    signal_b,
    -SUM(probability * LN(probability + 1e-10)) AS joint_entropy
FROM joint_probs
GROUP BY signal_a, signal_b;


-- ============================================================================
-- 005: CONDITIONAL ENTROPY
-- ============================================================================
-- H(Y|X) = H(X,Y) - H(X)
-- Uncertainty in Y given X

CREATE OR REPLACE VIEW v_conditional_entropy AS
SELECT
    je.signal_a,
    je.signal_b,
    je.joint_entropy,
    se_a.shannon_entropy AS entropy_a,
    se_b.shannon_entropy AS entropy_b,
    je.joint_entropy - se_a.shannon_entropy AS conditional_entropy_b_given_a,
    je.joint_entropy - se_b.shannon_entropy AS conditional_entropy_a_given_b
FROM v_joint_entropy je
JOIN v_shannon_entropy se_a ON je.signal_a = se_a.signal_id
JOIN v_shannon_entropy se_b ON je.signal_b = se_b.signal_id;


-- ============================================================================
-- 006: MUTUAL INFORMATION
-- ============================================================================
-- I(X;Y) = H(X) + H(Y) - H(X,Y)
-- Shared information between signals

CREATE OR REPLACE VIEW v_mutual_information AS
SELECT
    signal_a,
    signal_b,
    entropy_a + entropy_b - joint_entropy AS mutual_information,
    (entropy_a + entropy_b - joint_entropy) / NULLIF(LEAST(entropy_a, entropy_b), 0) AS normalized_mutual_info
FROM v_conditional_entropy;


-- ============================================================================
-- 007: PERMUTATION ENTROPY
-- ============================================================================
-- Entropy based on ordinal patterns
-- Robust to noise, captures temporal structure

CREATE OR REPLACE VIEW v_permutation_entropy AS
WITH patterns AS (
    SELECT
        signal_id,
        I,
        -- Order pattern of 3 consecutive points (6 possible patterns)
        CASE
            WHEN y < LEAD(y, 1) OVER w AND LEAD(y, 1) OVER w < LEAD(y, 2) OVER w THEN '012'
            WHEN y < LEAD(y, 2) OVER w AND LEAD(y, 2) OVER w < LEAD(y, 1) OVER w THEN '021'
            WHEN LEAD(y, 1) OVER w < y AND y < LEAD(y, 2) OVER w THEN '102'
            WHEN LEAD(y, 1) OVER w < LEAD(y, 2) OVER w AND LEAD(y, 2) OVER w < y THEN '120'
            WHEN LEAD(y, 2) OVER w < y AND y < LEAD(y, 1) OVER w THEN '201'
            WHEN LEAD(y, 2) OVER w < LEAD(y, 1) OVER w AND LEAD(y, 1) OVER w < y THEN '210'
            ELSE 'tie'
        END AS pattern
    FROM v_base
    WINDOW w AS (PARTITION BY signal_id ORDER BY I)
),
pattern_counts AS (
    SELECT
        signal_id,
        pattern,
        COUNT(*) AS pattern_count,
        SUM(COUNT(*)) OVER (PARTITION BY signal_id) AS total_patterns
    FROM patterns
    WHERE pattern != 'tie'
    GROUP BY signal_id, pattern
),
probs AS (
    SELECT
        signal_id,
        pattern,
        pattern_count::FLOAT / total_patterns AS probability
    FROM pattern_counts
)
SELECT
    signal_id,
    -SUM(probability * LN(probability + 1e-10)) AS permutation_entropy,
    -SUM(probability * LN(probability + 1e-10)) / LN(6) AS normalized_permutation_entropy,  -- max is ln(6) for order 3
    CASE
        WHEN -SUM(probability * LN(probability + 1e-10)) / LN(6) > 0.9 THEN 'random'
        WHEN -SUM(probability * LN(probability + 1e-10)) / LN(6) < 0.5 THEN 'structured'
        ELSE 'mixed'
    END AS complexity_category
FROM probs
GROUP BY signal_id;


-- ============================================================================
-- 008: APPROXIMATE ENTROPY PROXY
-- ============================================================================
-- Simplified version using pattern matching

CREATE OR REPLACE VIEW v_approx_entropy_proxy AS
WITH windowed_patterns AS (
    SELECT
        signal_id,
        I,
        y,
        AVG(y) OVER w AS local_mean,
        CASE WHEN y > AVG(y) OVER w THEN 1 ELSE 0 END AS binary_pattern
    FROM v_base
    WINDOW w AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING)
),
pattern_strings AS (
    SELECT
        signal_id,
        I,
        binary_pattern::TEXT || 
        LEAD(binary_pattern, 1) OVER (PARTITION BY signal_id ORDER BY I)::TEXT ||
        LEAD(binary_pattern, 2) OVER (PARTITION BY signal_id ORDER BY I)::TEXT AS pattern_3
    FROM windowed_patterns
),
pattern_counts AS (
    SELECT
        signal_id,
        pattern_3,
        COUNT(*) AS cnt,
        SUM(COUNT(*)) OVER (PARTITION BY signal_id) AS total
    FROM pattern_strings
    WHERE pattern_3 IS NOT NULL AND LENGTH(pattern_3) = 3
    GROUP BY signal_id, pattern_3
)
SELECT
    signal_id,
    -SUM((cnt::FLOAT / total) * LN(cnt::FLOAT / total + 1e-10)) AS approx_entropy_proxy
FROM pattern_counts
GROUP BY signal_id;


-- ============================================================================
-- 009: SPECTRAL ENTROPY
-- ============================================================================
-- Entropy of power spectrum (requires FFT from PRISM)
-- Approximation using derivative variance as proxy

CREATE OR REPLACE VIEW v_spectral_entropy_proxy AS
WITH freq_proxy AS (
    SELECT
        signal_id,
        I,
        ABS(d2y) AS high_freq_component,
        ABS(y - AVG(y) OVER (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING)) AS low_freq_deviation
    FROM v_d2y
    WHERE d2y IS NOT NULL
),
binned_freq AS (
    SELECT
        signal_id,
        NTILE(10) OVER (PARTITION BY signal_id ORDER BY high_freq_component / NULLIF(low_freq_deviation + 0.001, 0)) AS freq_bin,
        COUNT(*) OVER (PARTITION BY signal_id) AS total
    FROM freq_proxy
),
probs AS (
    SELECT
        signal_id,
        freq_bin,
        COUNT(*)::FLOAT / MAX(total) AS probability
    FROM binned_freq
    GROUP BY signal_id, freq_bin
)
SELECT
    signal_id,
    -SUM(probability * LN(probability + 1e-10)) AS spectral_entropy_proxy
FROM probs
GROUP BY signal_id;


-- ============================================================================
-- 010: ROLLING ENTROPY (time-varying complexity)
-- ============================================================================

CREATE OR REPLACE VIEW v_rolling_entropy AS
WITH rolling_bins AS (
    SELECT
        signal_id,
        I,
        NTILE(10) OVER w AS local_bin
    FROM v_base
    WINDOW w AS (PARTITION BY signal_id ORDER BY I ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING)
),
local_probs AS (
    SELECT
        signal_id,
        I,
        local_bin,
        COUNT(*) OVER (PARTITION BY signal_id, I, local_bin) AS bin_count,
        COUNT(*) OVER (PARTITION BY signal_id, I) AS total_count
    FROM rolling_bins
)
SELECT
    signal_id,
    I,
    -SUM(DISTINCT (bin_count::FLOAT / total_count) * LN(bin_count::FLOAT / total_count + 1e-10)) AS rolling_entropy
FROM local_probs
GROUP BY signal_id, I;


-- ============================================================================
-- 011: INFORMATION GAIN (entropy reduction over time)
-- ============================================================================

CREATE OR REPLACE VIEW v_information_gain AS
WITH midpoints AS (
    SELECT
        signal_id,
        MAX(I) / 2 AS midpoint
    FROM v_base
    GROUP BY signal_id
),
entropy_by_half AS (
    SELECT
        b.signal_id,
        CASE WHEN b.I < m.midpoint THEN 'first_half' ELSE 'second_half' END AS half,
        b.y
    FROM v_base b
    JOIN midpoints m USING (signal_id)
),
binned AS (
    SELECT
        signal_id,
        half,
        NTILE(20) OVER (PARTITION BY signal_id, half ORDER BY y) AS bin_id
    FROM entropy_by_half
),
bin_counts AS (
    SELECT
        signal_id,
        half,
        bin_id,
        COUNT(*) AS cnt
    FROM binned
    GROUP BY signal_id, half, bin_id
),
totals AS (
    SELECT
        signal_id,
        half,
        SUM(cnt) AS total
    FROM bin_counts
    GROUP BY signal_id, half
),
probs AS (
    SELECT
        bc.signal_id,
        bc.half,
        bc.cnt::FLOAT / t.total AS probability
    FROM bin_counts bc
    JOIN totals t USING (signal_id, half)
),
half_entropy AS (
    SELECT
        signal_id,
        half,
        -SUM(probability * LN(probability + 1e-10)) AS half_entropy
    FROM probs
    GROUP BY signal_id, half
)
SELECT
    f.signal_id,
    f.half_entropy AS entropy_first_half,
    s.half_entropy AS entropy_second_half,
    f.half_entropy - s.half_entropy AS information_gain,
    CASE
        WHEN f.half_entropy > s.half_entropy THEN 'increasing_order'
        WHEN f.half_entropy < s.half_entropy THEN 'increasing_disorder'
        ELSE 'stable_complexity'
    END AS complexity_trend
FROM half_entropy f
JOIN half_entropy s ON f.signal_id = s.signal_id AND f.half = 'first_half' AND s.half = 'second_half';


-- ============================================================================
-- ENTROPY SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_entropy_complete AS
SELECT
    se.signal_id,
    se.shannon_entropy,
    se.normalized_entropy,
    se.entropy_category,
    de.derivative_entropy,
    pe.permutation_entropy,
    pe.normalized_permutation_entropy,
    pe.complexity_category,
    ae.approx_entropy_proxy,
    sp.spectral_entropy_proxy,
    ig.entropy_first_half,
    ig.entropy_second_half,
    ig.information_gain,
    ig.complexity_trend
FROM v_shannon_entropy se
LEFT JOIN v_derivative_entropy de USING (signal_id)
LEFT JOIN v_permutation_entropy pe USING (signal_id)
LEFT JOIN v_approx_entropy_proxy ae USING (signal_id)
LEFT JOIN v_spectral_entropy_proxy sp USING (signal_id)
LEFT JOIN v_information_gain ig USING (signal_id);
