-- ============================================================================
-- ORTHON SQL ENGINES: PHYSICS & CONSERVATION LAWS
-- ============================================================================
-- Check for physical consistency, conservation violations, thermodynamic laws.
-- These engines verify that the data makes physical sense.
-- ============================================================================

-- ============================================================================
-- 001: CONSERVATION CHECK (summed quantity should be constant)
-- ============================================================================
-- For signals that should sum to a conserved quantity (e.g., mass balance)

CREATE OR REPLACE VIEW v_conservation_check AS
WITH system_total AS (
    SELECT
        I,
        SUM(y) AS total_value
    FROM v_base
    WHERE signal_class = 'analog'  -- Only physical quantities
    GROUP BY I
)
SELECT
    I,
    total_value,
    AVG(total_value) OVER () AS expected_total,
    total_value - AVG(total_value) OVER () AS deviation,
    (total_value - AVG(total_value) OVER ()) / NULLIF(STDDEV(total_value) OVER (), 0) AS deviation_zscore,
    CASE
        WHEN ABS((total_value - AVG(total_value) OVER ()) / NULLIF(STDDEV(total_value) OVER (), 0)) > 3 
        THEN 'conservation_violation'
        ELSE 'conserved'
    END AS conservation_status
FROM system_total;


-- ============================================================================
-- 002: MASS BALANCE (flow in = flow out)
-- ============================================================================
-- For flow systems: inlet - outlet should equal accumulation

CREATE OR REPLACE VIEW v_mass_balance AS
WITH flows AS (
    SELECT
        b.I,
        b.signal_id,
        b.y,
        b.value_unit,
        CASE
            WHEN b.signal_id LIKE '%inlet%' OR b.signal_id LIKE '%in%' THEN 'inlet'
            WHEN b.signal_id LIKE '%outlet%' OR b.signal_id LIKE '%out%' THEN 'outlet'
            ELSE 'unknown'
        END AS flow_type
    FROM v_base b
    WHERE b.value_unit IN ('m³/s', 'L/min', 'gpm', 'kg/s', 'lb/hr')
)
SELECT
    I,
    SUM(CASE WHEN flow_type = 'inlet' THEN y ELSE 0 END) AS total_inlet,
    SUM(CASE WHEN flow_type = 'outlet' THEN y ELSE 0 END) AS total_outlet,
    SUM(CASE WHEN flow_type = 'inlet' THEN y ELSE 0 END) - 
        SUM(CASE WHEN flow_type = 'outlet' THEN y ELSE 0 END) AS net_accumulation,
    CASE
        WHEN ABS(SUM(CASE WHEN flow_type = 'inlet' THEN y ELSE 0 END) - 
                 SUM(CASE WHEN flow_type = 'outlet' THEN y ELSE 0 END)) < 0.01 * 
             GREATEST(SUM(CASE WHEN flow_type = 'inlet' THEN y ELSE 0 END), 0.001)
        THEN 'balanced'
        ELSE 'imbalanced'
    END AS balance_status
FROM flows
WHERE flow_type IN ('inlet', 'outlet')
GROUP BY I;


-- ============================================================================
-- 003: ENERGY BALANCE
-- ============================================================================
-- Power = Pressure × Flow (simplified)

CREATE OR REPLACE VIEW v_energy_balance AS
WITH pressure_flow AS (
    SELECT
        p.I,
        p.signal_id AS pressure_signal,
        p.y AS pressure,
        f.signal_id AS flow_signal,
        f.y AS flow,
        p.y * f.y AS power_estimate
    FROM v_base p
    JOIN v_base f ON p.I = f.I
    WHERE p.value_unit IN ('kPa', 'PSI', 'bar', 'Pa')
      AND f.value_unit IN ('m³/s', 'L/min', 'gpm')
)
SELECT
    I,
    SUM(power_estimate) AS total_power,
    STDDEV(power_estimate) AS power_variation,
    CASE
        WHEN STDDEV(power_estimate) / NULLIF(AVG(power_estimate), 0) < 0.1 THEN 'stable_energy'
        ELSE 'varying_energy'
    END AS energy_status
FROM pressure_flow
GROUP BY I;


-- ============================================================================
-- 004: THERMODYNAMIC CONSISTENCY (PVT relationships)
-- ============================================================================
-- For ideal gas: P * V ∝ T

CREATE OR REPLACE VIEW v_thermodynamic_consistency AS
WITH pvt AS (
    SELECT
        p.I,
        p.y AS pressure,
        v.y AS volume,
        t.y AS temperature,
        (p.y * COALESCE(v.y, 1)) / NULLIF(t.y, 0) AS pv_over_t
    FROM v_base p
    JOIN v_base t ON p.I = t.I
    LEFT JOIN v_base v ON p.I = v.I AND v.value_unit IN ('m³', 'L', 'ft³')
    WHERE p.value_unit IN ('kPa', 'PSI', 'bar', 'Pa')
      AND t.value_unit IN ('K', '°C', '°F')
)
SELECT
    I,
    pressure,
    temperature,
    pv_over_t,
    AVG(pv_over_t) OVER () AS expected_constant,
    (pv_over_t - AVG(pv_over_t) OVER ()) / NULLIF(STDDEV(pv_over_t) OVER (), 0) AS deviation_zscore,
    CASE
        WHEN ABS((pv_over_t - AVG(pv_over_t) OVER ()) / NULLIF(STDDEV(pv_over_t) OVER (), 0)) > 2 
        THEN 'thermodynamic_anomaly'
        ELSE 'consistent'
    END AS thermo_status
FROM pvt
WHERE pv_over_t IS NOT NULL;


-- ============================================================================
-- 005: SECOND LAW CHECK (entropy should not decrease in isolated system)
-- ============================================================================

CREATE OR REPLACE VIEW v_second_law_check AS
WITH system_entropy AS (
    SELECT
        I,
        AVG(normalized_entropy) AS system_entropy
    FROM v_base b
    JOIN v_shannon_entropy se USING (signal_id)
    GROUP BY I
),
entropy_change AS (
    SELECT
        I,
        system_entropy,
        system_entropy - LAG(system_entropy) OVER (ORDER BY I) AS entropy_change
    FROM system_entropy
)
SELECT
    I,
    system_entropy,
    entropy_change,
    CASE
        WHEN entropy_change < -0.1 THEN 'possible_second_law_violation'
        ELSE 'consistent'
    END AS second_law_status,
    SUM(CASE WHEN entropy_change < -0.1 THEN 1 ELSE 0 END) OVER () AS n_violations
FROM entropy_change;


-- ============================================================================
-- 006: CONTINUITY EQUATION CHECK (∂ρ/∂t + ∇·(ρv) = 0)
-- ============================================================================
-- For density and velocity fields in space

CREATE OR REPLACE VIEW v_continuity_equation AS
WITH density_velocity AS (
    SELECT
        d.I,
        d.y AS density,
        v.y AS velocity,
        d.y * v.y AS mass_flux,
        -- Temporal derivative of density
        (LEAD(d.y) OVER (PARTITION BY d.signal_id ORDER BY d.I) - 
         LAG(d.y) OVER (PARTITION BY d.signal_id ORDER BY d.I)) / 2.0 AS drho_dt,
        -- Spatial derivative of mass flux (approximation)
        (LEAD(d.y * v.y) OVER (PARTITION BY d.signal_id ORDER BY d.I) - 
         LAG(d.y * v.y) OVER (PARTITION BY d.signal_id ORDER BY d.I)) / 2.0 AS div_mass_flux
    FROM v_base d
    JOIN v_base v ON d.I = v.I
    WHERE d.value_unit IN ('kg/m³', 'g/cm³')
      AND v.value_unit IN ('m/s', 'ft/s')
)
SELECT
    I,
    density,
    velocity,
    drho_dt,
    div_mass_flux,
    drho_dt + div_mass_flux AS continuity_residual,
    CASE
        WHEN ABS(drho_dt + div_mass_flux) < 0.01 * GREATEST(ABS(drho_dt), ABS(div_mass_flux), 0.001)
        THEN 'satisfied'
        ELSE 'violated'
    END AS continuity_status
FROM density_velocity
WHERE drho_dt IS NOT NULL AND div_mass_flux IS NOT NULL;


-- ============================================================================
-- 007: DIFFUSION CHECK (Fick's Law: J = -D∇C)
-- ============================================================================
-- Flux should be proportional to concentration gradient

CREATE OR REPLACE VIEW v_diffusion_check AS
WITH gradients AS (
    SELECT
        signal_id,
        I,
        y AS concentration,
        (LEAD(y) OVER w - LAG(y) OVER w) / 2.0 AS gradient
    FROM v_base
    WHERE index_dimension = 'space'
    WINDOW w AS (PARTITION BY signal_id ORDER BY I)
)
SELECT
    signal_id,
    CORR(ABS(gradient), concentration) AS gradient_concentration_corr,
    CASE
        WHEN CORR(ABS(gradient), concentration) > 0.5 THEN 'diffusion_dominated'
        WHEN CORR(ABS(gradient), concentration) < -0.5 THEN 'anti_diffusion'
        ELSE 'mixed_transport'
    END AS transport_regime
FROM gradients
WHERE gradient IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- 008: BERNOULLI CHECK (P + ½ρv² + ρgh = constant)
-- ============================================================================

CREATE OR REPLACE VIEW v_bernoulli_check AS
WITH bernoulli_terms AS (
    SELECT
        p.I,
        p.y AS pressure,
        v.y AS velocity,
        -- Simplified: P + 0.5 * v^2 (ignoring ρ and height)
        p.y + 0.5 * v.y * v.y AS bernoulli_sum
    FROM v_base p
    JOIN v_base v ON p.I = v.I
    WHERE p.value_unit IN ('kPa', 'Pa', 'PSI')
      AND v.value_unit IN ('m/s', 'ft/s')
)
SELECT
    I,
    pressure,
    velocity,
    bernoulli_sum,
    AVG(bernoulli_sum) OVER () AS expected_constant,
    (bernoulli_sum - AVG(bernoulli_sum) OVER ()) / NULLIF(STDDEV(bernoulli_sum) OVER (), 0) AS deviation_zscore,
    CASE
        WHEN ABS((bernoulli_sum - AVG(bernoulli_sum) OVER ()) / 
                 NULLIF(STDDEV(bernoulli_sum) OVER (), 0)) > 2 
        THEN 'bernoulli_violation'
        ELSE 'consistent'
    END AS bernoulli_status
FROM bernoulli_terms;


-- ============================================================================
-- 009: HEAT TRANSFER CHECK (Q = mcΔT or Q = kAΔT/Δx)
-- ============================================================================

CREATE OR REPLACE VIEW v_heat_transfer_check AS
WITH temp_gradient AS (
    SELECT
        signal_id,
        I,
        y AS temperature,
        (LEAD(y) OVER w - LAG(y) OVER w) / 2.0 AS dT_dx
    FROM v_base
    WHERE value_unit IN ('K', '°C', '°F')
      AND index_dimension = 'space'
    WINDOW w AS (PARTITION BY signal_id ORDER BY I)
)
SELECT
    signal_id,
    AVG(ABS(dT_dx)) AS avg_temp_gradient,
    STDDEV(dT_dx) AS temp_gradient_variation,
    CASE
        WHEN STDDEV(dT_dx) / NULLIF(AVG(ABS(dT_dx)), 0) < 0.2 THEN 'steady_conduction'
        WHEN STDDEV(dT_dx) / NULLIF(AVG(ABS(dT_dx)), 0) < 0.5 THEN 'quasi_steady'
        ELSE 'transient'
    END AS heat_transfer_regime
FROM temp_gradient
WHERE dT_dx IS NOT NULL
GROUP BY signal_id;


-- ============================================================================
-- 010: MOMENTUM BALANCE (F = ma = m dv/dt)
-- ============================================================================

CREATE OR REPLACE VIEW v_momentum_balance AS
WITH forces AS (
    SELECT
        f.I,
        f.y AS force,
        a.y AS acceleration,
        f.y / NULLIF(a.y, 0) AS implied_mass
    FROM v_base f
    JOIN v_d2y a ON f.I = a.I
    WHERE f.value_unit IN ('N', 'kN', 'lbf')
)
SELECT
    I,
    force,
    acceleration,
    implied_mass,
    AVG(implied_mass) OVER () AS avg_mass,
    (implied_mass - AVG(implied_mass) OVER ()) / NULLIF(STDDEV(implied_mass) OVER (), 0) AS mass_consistency,
    CASE
        WHEN ABS((implied_mass - AVG(implied_mass) OVER ()) / 
                 NULLIF(STDDEV(implied_mass) OVER (), 0)) > 2 
        THEN 'momentum_anomaly'
        ELSE 'consistent'
    END AS momentum_status
FROM forces
WHERE implied_mass IS NOT NULL;


-- ============================================================================
-- PHYSICS SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_physics_complete AS
SELECT
    'conservation' AS check_type,
    SUM(CASE WHEN conservation_status = 'conservation_violation' THEN 1 ELSE 0 END) AS n_violations,
    COUNT(*) AS n_checks
FROM v_conservation_check
UNION ALL
SELECT
    'mass_balance' AS check_type,
    SUM(CASE WHEN balance_status = 'imbalanced' THEN 1 ELSE 0 END) AS n_violations,
    COUNT(*) AS n_checks
FROM v_mass_balance
UNION ALL
SELECT
    'thermodynamic' AS check_type,
    SUM(CASE WHEN thermo_status = 'thermodynamic_anomaly' THEN 1 ELSE 0 END) AS n_violations,
    COUNT(*) AS n_checks
FROM v_thermodynamic_consistency
UNION ALL
SELECT
    'second_law' AS check_type,
    MAX(n_violations) AS n_violations,
    COUNT(*) AS n_checks
FROM v_second_law_check
UNION ALL
SELECT
    'bernoulli' AS check_type,
    SUM(CASE WHEN bernoulli_status = 'bernoulli_violation' THEN 1 ELSE 0 END) AS n_violations,
    COUNT(*) AS n_checks
FROM v_bernoulli_check;
