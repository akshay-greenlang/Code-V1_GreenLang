-- =============================================================================
-- V305: PACK-038 Peak Shaving Pack - Views, Indexes, Audit Trail, Seed Data
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Final migration: pack-level audit trail, materialized views for
-- dashboards, composite indexes for common query patterns, GRANT
-- statements, and seed data for battery chemistry specs and common
-- tariff templates.
--
-- Tables (1):
--   1. pack038_peak_shaving.pack038_audit_trail
--
-- Materialized Views (3):
--   1. pack038_peak_shaving.mv_facility_peak_summary
--   2. pack038_peak_shaving.mv_bess_performance_summary
--   3. pack038_peak_shaving.mv_portfolio_peak_management
--
-- Views (1):
--   1. pack038_peak_shaving.v_peak_shaving_dashboard
--
-- Previous: V304__pack038_peak_shaving_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.pack038_audit_trail
-- =============================================================================
CREATE TABLE pack038_peak_shaving.pack038_audit_trail (
    audit_trail_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID,
    tenant_id               UUID,
    action                  VARCHAR(50)     NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID,
    actor                   TEXT            NOT NULL,
    actor_role              VARCHAR(50),
    ip_address              VARCHAR(45),
    old_values              JSONB,
    new_values              JSONB,
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_p038_trail_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'READ', 'APPROVE', 'REJECT',
                   'SUBMIT', 'EXPORT', 'IMPORT', 'CALCULATE', 'VERIFY',
                   'ARCHIVE', 'RESTORE', 'LOCK', 'UNLOCK', 'DISPATCH',
                   'CHARGE', 'DISCHARGE', 'SIMULATE', 'CONFIGURE',
                   'ALERT', 'ACKNOWLEDGE', 'RESOLVE', 'FORECAST',
                   'SHIFT', 'CURTAIL', 'CORRECT', 'SIZE')
    )
);

CREATE INDEX idx_p038_trail_profile      ON pack038_peak_shaving.pack038_audit_trail(profile_id);
CREATE INDEX idx_p038_trail_tenant       ON pack038_peak_shaving.pack038_audit_trail(tenant_id);
CREATE INDEX idx_p038_trail_action       ON pack038_peak_shaving.pack038_audit_trail(action);
CREATE INDEX idx_p038_trail_entity       ON pack038_peak_shaving.pack038_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p038_trail_actor        ON pack038_peak_shaving.pack038_audit_trail(actor);
CREATE INDEX idx_p038_trail_created      ON pack038_peak_shaving.pack038_audit_trail(created_at DESC);
CREATE INDEX idx_p038_trail_details      ON pack038_peak_shaving.pack038_audit_trail USING GIN(details);

ALTER TABLE pack038_peak_shaving.pack038_audit_trail ENABLE ROW LEVEL SECURITY;
CREATE POLICY p038_trail_tenant_isolation ON pack038_peak_shaving.pack038_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_trail_service_bypass ON pack038_peak_shaving.pack038_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Materialized View 1: mv_facility_peak_summary
-- =============================================================================
-- Per-facility peak shaving summary with load profile stats, BESS
-- configuration, demand charge history, and savings performance for
-- dashboard rendering.

CREATE MATERIALIZED VIEW pack038_peak_shaving.mv_facility_peak_summary AS
SELECT
    lp.id AS profile_id,
    lp.tenant_id,
    lp.facility_name,
    lp.iso_rto_region,
    lp.facility_type,
    lp.peak_demand_kw,
    lp.average_demand_kw,
    lp.load_factor,
    lp.profile_status,
    -- BESS summary
    (SELECT COUNT(*) FROM pack038_peak_shaving.ps_bess_configurations bc
     WHERE bc.profile_id = lp.id AND bc.bess_status = 'OPERATIONAL') AS operational_bess_count,
    (SELECT COALESCE(SUM(bc.capacity_kwh), 0) FROM pack038_peak_shaving.ps_bess_configurations bc
     WHERE bc.profile_id = lp.id AND bc.bess_status = 'OPERATIONAL') AS total_bess_capacity_kwh,
    (SELECT COALESCE(SUM(bc.power_kw), 0) FROM pack038_peak_shaving.ps_bess_configurations bc
     WHERE bc.profile_id = lp.id AND bc.bess_status = 'OPERATIONAL') AS total_bess_power_kw,
    (SELECT AVG(bc.current_soh_pct) FROM pack038_peak_shaving.ps_bess_configurations bc
     WHERE bc.profile_id = lp.id AND bc.bess_status = 'OPERATIONAL') AS avg_bess_soh_pct,
    -- Peak events (last 12 months)
    (SELECT COUNT(*) FROM pack038_peak_shaving.ps_peak_events pe
     WHERE pe.profile_id = lp.id AND pe.peak_timestamp >= NOW() - INTERVAL '12 months') AS peak_events_12m,
    (SELECT MAX(pe.peak_kw) FROM pack038_peak_shaving.ps_peak_events pe
     WHERE pe.profile_id = lp.id AND pe.peak_timestamp >= NOW() - INTERVAL '12 months') AS max_peak_kw_12m,
    -- Demand charges (last 12 months)
    (SELECT COALESCE(SUM(dc.total_demand_charge), 0) FROM pack038_peak_shaving.ps_demand_charges dc
     WHERE dc.profile_id = lp.id AND dc.billing_period_start >= NOW() - INTERVAL '12 months') AS total_demand_charges_12m,
    (SELECT AVG(dc.demand_pct_of_bill) FROM pack038_peak_shaving.ps_demand_charges dc
     WHERE dc.profile_id = lp.id AND dc.billing_period_start >= NOW() - INTERVAL '12 months') AS avg_demand_pct_of_bill,
    -- Dispatch simulations best savings
    (SELECT MAX(ds.annualized_savings) FROM pack038_peak_shaving.ps_dispatch_simulations ds
     WHERE ds.profile_id = lp.id AND ds.simulation_status = 'COMPLETED') AS best_simulated_annual_savings,
    -- Ratchet status
    (SELECT COUNT(*) FROM pack038_peak_shaving.ps_ratchet_history rh
     WHERE rh.profile_id = lp.id AND rh.ratchet_was_binding = true
       AND rh.billing_month >= NOW() - INTERVAL '12 months') AS binding_ratchet_months_12m,
    -- Financial model best NPV
    (SELECT MAX(fm.npv) FROM pack038_peak_shaving.ps_financial_models fm
     WHERE fm.profile_id = lp.id AND fm.model_status IN ('FINAL', 'APPROVED')) AS best_npv
FROM pack038_peak_shaving.ps_load_profiles lp
WITH NO DATA;

CREATE UNIQUE INDEX idx_p038_mv_fps_profile ON pack038_peak_shaving.mv_facility_peak_summary(profile_id);
CREATE INDEX idx_p038_mv_fps_tenant ON pack038_peak_shaving.mv_facility_peak_summary(tenant_id);
CREATE INDEX idx_p038_mv_fps_region ON pack038_peak_shaving.mv_facility_peak_summary(iso_rto_region);
CREATE INDEX idx_p038_mv_fps_type ON pack038_peak_shaving.mv_facility_peak_summary(facility_type);
CREATE INDEX idx_p038_mv_fps_peak ON pack038_peak_shaving.mv_facility_peak_summary(peak_demand_kw DESC);

-- =============================================================================
-- Materialized View 2: mv_bess_performance_summary
-- =============================================================================
-- BESS fleet performance summary aggregating dispatch results,
-- degradation status, and financial performance per BESS asset.

CREATE MATERIALIZED VIEW pack038_peak_shaving.mv_bess_performance_summary AS
SELECT
    bc.id AS bess_id,
    bc.profile_id,
    bc.tenant_id,
    bc.bess_name,
    bc.chemistry,
    bc.capacity_kwh,
    bc.power_kw,
    bc.bess_status,
    bc.current_soh_pct,
    bc.current_cycle_count,
    -- Dispatch performance
    (SELECT COUNT(*) FROM pack038_peak_shaving.ps_dispatch_simulations ds
     WHERE ds.bess_id = bc.id AND ds.simulation_status = 'COMPLETED') AS total_simulations,
    (SELECT AVG(ds.peak_reduction_pct) FROM pack038_peak_shaving.ps_dispatch_simulations ds
     WHERE ds.bess_id = bc.id AND ds.simulation_status = 'COMPLETED') AS avg_peak_reduction_pct,
    (SELECT COALESCE(SUM(ds.net_savings), 0) FROM pack038_peak_shaving.ps_dispatch_simulations ds
     WHERE ds.bess_id = bc.id AND ds.simulation_status = 'COMPLETED') AS total_simulated_savings,
    -- Degradation trend
    (SELECT dt.soh_pct FROM pack038_peak_shaving.ps_degradation_tracking dt
     WHERE dt.bess_id = bc.id ORDER BY dt.measurement_date DESC LIMIT 1) AS latest_soh_pct,
    (SELECT dt.predicted_eol_date FROM pack038_peak_shaving.ps_degradation_tracking dt
     WHERE dt.bess_id = bc.id ORDER BY dt.measurement_date DESC LIMIT 1) AS predicted_eol_date,
    (SELECT dt.warranty_consumed_pct FROM pack038_peak_shaving.ps_degradation_tracking dt
     WHERE dt.bess_id = bc.id ORDER BY dt.measurement_date DESC LIMIT 1) AS warranty_consumed_pct,
    -- Revenue stacking total
    (SELECT COALESCE(SUM(rs.annual_value), 0) FROM pack038_peak_shaving.ps_revenue_stacking rs
     WHERE rs.bess_id = bc.id AND rs.analysis_year = EXTRACT(YEAR FROM NOW())) AS current_year_revenue,
    -- Active dispatch schedules
    (SELECT COUNT(*) FROM pack038_peak_shaving.ps_dispatch_schedules dsc
     WHERE dsc.bess_id = bc.id AND dsc.schedule_status = 'ACTIVE') AS active_schedules
FROM pack038_peak_shaving.ps_bess_configurations bc
WITH NO DATA;

CREATE UNIQUE INDEX idx_p038_mv_bps_bess ON pack038_peak_shaving.mv_bess_performance_summary(bess_id);
CREATE INDEX idx_p038_mv_bps_profile ON pack038_peak_shaving.mv_bess_performance_summary(profile_id);
CREATE INDEX idx_p038_mv_bps_tenant ON pack038_peak_shaving.mv_bess_performance_summary(tenant_id);
CREATE INDEX idx_p038_mv_bps_chemistry ON pack038_peak_shaving.mv_bess_performance_summary(chemistry);
CREATE INDEX idx_p038_mv_bps_status ON pack038_peak_shaving.mv_bess_performance_summary(bess_status);
CREATE INDEX idx_p038_mv_bps_soh ON pack038_peak_shaving.mv_bess_performance_summary(current_soh_pct);

-- =============================================================================
-- Materialized View 3: mv_portfolio_peak_management
-- =============================================================================
-- Portfolio-wide peak management metrics by tenant and region for
-- executive dashboards and multi-site management.

CREATE MATERIALIZED VIEW pack038_peak_shaving.mv_portfolio_peak_management AS
SELECT
    lp.tenant_id,
    lp.iso_rto_region,
    COUNT(DISTINCT lp.id) AS total_facilities,
    SUM(lp.peak_demand_kw) AS total_peak_kw,
    AVG(lp.load_factor) AS avg_load_factor,
    -- BESS fleet
    COUNT(DISTINCT bc.id) FILTER (WHERE bc.bess_status = 'OPERATIONAL') AS operational_bess_count,
    COALESCE(SUM(bc.capacity_kwh) FILTER (WHERE bc.bess_status = 'OPERATIONAL'), 0) AS total_bess_kwh,
    COALESCE(SUM(bc.power_kw) FILTER (WHERE bc.bess_status = 'OPERATIONAL'), 0) AS total_bess_kw,
    -- Demand charges
    COALESCE(SUM(dc_agg.total_dc), 0) AS total_demand_charges_12m,
    -- Peak events
    COALESCE(SUM(pe_agg.event_count), 0) AS total_peak_events_12m,
    -- Ratchet months
    COALESCE(SUM(rh_agg.binding_months), 0) AS total_binding_ratchet_months_12m,
    -- Financial
    COALESCE(SUM(fm_agg.best_npv), 0) AS total_portfolio_npv
FROM pack038_peak_shaving.ps_load_profiles lp
LEFT JOIN pack038_peak_shaving.ps_bess_configurations bc ON bc.profile_id = lp.id
LEFT JOIN LATERAL (
    SELECT COALESCE(SUM(dc.total_demand_charge), 0) AS total_dc
    FROM pack038_peak_shaving.ps_demand_charges dc
    WHERE dc.profile_id = lp.id AND dc.billing_period_start >= NOW() - INTERVAL '12 months'
) dc_agg ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS event_count
    FROM pack038_peak_shaving.ps_peak_events pe
    WHERE pe.profile_id = lp.id AND pe.peak_timestamp >= NOW() - INTERVAL '12 months'
) pe_agg ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS binding_months
    FROM pack038_peak_shaving.ps_ratchet_history rh
    WHERE rh.profile_id = lp.id AND rh.ratchet_was_binding = true
      AND rh.billing_month >= NOW() - INTERVAL '12 months'
) rh_agg ON TRUE
LEFT JOIN LATERAL (
    SELECT MAX(fm.npv) AS best_npv
    FROM pack038_peak_shaving.ps_financial_models fm
    WHERE fm.profile_id = lp.id AND fm.model_status IN ('FINAL', 'APPROVED')
) fm_agg ON TRUE
GROUP BY lp.tenant_id, lp.iso_rto_region
WITH NO DATA;

CREATE UNIQUE INDEX idx_p038_mv_ppm_tenant_region ON pack038_peak_shaving.mv_portfolio_peak_management(tenant_id, iso_rto_region);
CREATE INDEX idx_p038_mv_ppm_tenant ON pack038_peak_shaving.mv_portfolio_peak_management(tenant_id);
CREATE INDEX idx_p038_mv_ppm_region ON pack038_peak_shaving.mv_portfolio_peak_management(iso_rto_region);

-- =============================================================================
-- View: v_peak_shaving_dashboard
-- =============================================================================
-- Real-time operations dashboard combining load profile, latest peak
-- event, BESS state, active alerts, and financial summary.

CREATE OR REPLACE VIEW pack038_peak_shaving.v_peak_shaving_dashboard AS
SELECT
    lp.id AS profile_id,
    lp.tenant_id,
    lp.facility_name,
    lp.iso_rto_region,
    lp.peak_demand_kw,
    lp.average_demand_kw,
    lp.load_factor,
    lp.profile_status,
    -- Latest peak event
    latest_pe.peak_kw AS latest_peak_kw,
    latest_pe.peak_timestamp AS latest_peak_time,
    latest_pe.peak_type AS latest_peak_type,
    latest_pe.is_billing_peak AS latest_is_billing_peak,
    -- Latest demand charge
    latest_dc.total_demand_charge AS latest_demand_charge,
    latest_dc.billing_demand_kw AS latest_billing_demand_kw,
    latest_dc.demand_pct_of_bill AS latest_demand_pct_bill,
    -- BESS summary
    bess.operational_count AS bess_operational,
    bess.total_power_kw AS bess_total_power_kw,
    bess.avg_soh AS bess_avg_soh_pct,
    -- Unresolved alerts
    alerts.unresolved_count AS unresolved_alerts,
    alerts.critical_count AS critical_alerts,
    -- Active ratchet
    ratchet.current_ratchet_kw,
    ratchet.months_remaining AS ratchet_months_remaining,
    -- Best financial model
    finance.best_npv,
    finance.best_irr_pct,
    finance.best_payback_years
FROM pack038_peak_shaving.ps_load_profiles lp
LEFT JOIN LATERAL (
    SELECT peak_kw, peak_timestamp, peak_type, is_billing_peak
    FROM pack038_peak_shaving.ps_peak_events
    WHERE profile_id = lp.id
    ORDER BY peak_timestamp DESC
    LIMIT 1
) latest_pe ON TRUE
LEFT JOIN LATERAL (
    SELECT total_demand_charge, billing_demand_kw, demand_pct_of_bill
    FROM pack038_peak_shaving.ps_demand_charges
    WHERE profile_id = lp.id
    ORDER BY billing_period_start DESC
    LIMIT 1
) latest_dc ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS operational_count,
           COALESCE(SUM(power_kw), 0) AS total_power_kw,
           AVG(current_soh_pct) AS avg_soh
    FROM pack038_peak_shaving.ps_bess_configurations
    WHERE profile_id = lp.id AND bess_status = 'OPERATIONAL'
) bess ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS unresolved_count,
           COUNT(*) FILTER (WHERE alert_severity IN ('CRITICAL', 'EMERGENCY')) AS critical_count
    FROM pack038_peak_shaving.ps_ratchet_alerts
    WHERE profile_id = lp.id AND resolved = false
) alerts ON TRUE
LEFT JOIN LATERAL (
    SELECT ratchet_demand_kw AS current_ratchet_kw, months_remaining
    FROM pack038_peak_shaving.ps_ratchet_history
    WHERE profile_id = lp.id AND ratchet_was_binding = true
    ORDER BY billing_month DESC
    LIMIT 1
) ratchet ON TRUE
LEFT JOIN LATERAL (
    SELECT npv AS best_npv, irr_pct AS best_irr_pct,
           simple_payback_years AS best_payback_years
    FROM pack038_peak_shaving.ps_financial_models
    WHERE profile_id = lp.id AND model_status IN ('FINAL', 'APPROVED')
    ORDER BY npv DESC
    LIMIT 1
) finance ON TRUE;

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Interval data by profile + peak period for peak analysis
CREATE INDEX idx_p038_id_profile_peak    ON pack038_peak_shaving.ps_interval_data(profile_id, timestamp DESC)
    WHERE is_peak_period = true;

-- Demand charges by profile + period for billing history dashboards
CREATE INDEX idx_p038_dc_profile_billing ON pack038_peak_shaving.ps_demand_charges(profile_id, billing_period_start DESC);

-- Peak events by profile + billing peak for demand charge attribution
CREATE INDEX idx_p038_pe_profile_bill    ON pack038_peak_shaving.ps_peak_events(profile_id, peak_timestamp DESC)
    WHERE is_billing_peak = true;

-- BESS configurations by profile + operational for dispatch
CREATE INDEX idx_p038_bc_profile_oper    ON pack038_peak_shaving.ps_bess_configurations(profile_id, chemistry)
    WHERE bess_status = 'OPERATIONAL';

-- Dispatch simulations by profile + best savings for comparison
CREATE INDEX idx_p038_ds_profile_save    ON pack038_peak_shaving.ps_dispatch_simulations(profile_id, net_savings DESC)
    WHERE simulation_status = 'COMPLETED';

-- Ratchet history by profile + binding for cost tracking
CREATE INDEX idx_p038_rh_profile_bind    ON pack038_peak_shaving.ps_ratchet_history(profile_id, billing_month DESC)
    WHERE ratchet_was_binding = true;

-- Financial models by profile + approved for dashboard
CREATE INDEX idx_p038_fm_profile_appr    ON pack038_peak_shaving.ps_financial_models(profile_id, npv DESC)
    WHERE model_status IN ('FINAL', 'APPROVED');

-- CP predictions by region + high probability for alerts
CREATE INDEX idx_p038_cpp_region_prob    ON pack038_peak_shaving.ps_cp_predictions(iso_rto_region, prediction_date, cp_probability_pct DESC)
    WHERE cp_probability_pct >= 50;

-- Shiftable loads by profile + active for coordination
CREATE INDEX idx_p038_sl_profile_active  ON pack038_peak_shaving.ps_shiftable_loads(profile_id, load_category)
    WHERE shift_status = 'ACTIVE';

-- PF penalties by profile + period for trending
CREATE INDEX idx_p038_pfp_profile_trend  ON pack038_peak_shaving.ps_pf_penalties(profile_id, billing_period_start DESC);

-- =============================================================================
-- Grants
-- =============================================================================
GRANT SELECT, INSERT ON pack038_peak_shaving.pack038_audit_trail TO PUBLIC;
GRANT SELECT ON pack038_peak_shaving.mv_facility_peak_summary TO PUBLIC;
GRANT SELECT ON pack038_peak_shaving.mv_bess_performance_summary TO PUBLIC;
GRANT SELECT ON pack038_peak_shaving.mv_portfolio_peak_management TO PUBLIC;
GRANT SELECT ON pack038_peak_shaving.v_peak_shaving_dashboard TO PUBLIC;

-- =============================================================================
-- Seed Data: Battery Chemistry Reference Specifications
-- =============================================================================
-- Reference table with typical specifications for common BESS chemistries
-- used in peak shaving applications.

CREATE TABLE pack038_peak_shaving.ps_chemistry_reference (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    chemistry               VARCHAR(30)     NOT NULL,
    chemistry_full_name     VARCHAR(100)    NOT NULL,
    typical_cycle_life      INTEGER         NOT NULL,
    typical_calendar_life_years INTEGER     NOT NULL,
    typical_rte_pct         NUMERIC(5,2)    NOT NULL,
    typical_dod_pct         NUMERIC(5,2)    NOT NULL,
    energy_density_wh_per_kg NUMERIC(8,2),
    energy_density_wh_per_l NUMERIC(8,2),
    cost_per_kwh_usd        NUMERIC(10,2),
    cost_per_kw_usd         NUMERIC(10,2),
    c_rate_max              NUMERIC(5,3),
    operating_temp_min_c    NUMERIC(6,2),
    operating_temp_max_c    NUMERIC(6,2),
    thermal_runaway_risk    VARCHAR(20),
    self_discharge_pct_month NUMERIC(5,2),
    degradation_rate_annual_pct NUMERIC(5,2),
    recyclability_pct       NUMERIC(5,2),
    maturity_level          VARCHAR(20),
    best_application        VARCHAR(100),
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_p038_cr_chemistry CHECK (
        chemistry IN (
            'LFP', 'NMC', 'NCA', 'LTO', 'SODIUM_ION', 'FLOW_VANADIUM',
            'FLOW_ZINC_BROMINE', 'FLOW_IRON', 'LEAD_ACID', 'SOLID_STATE',
            'ZINC_AIR', 'SUPERCAPACITOR'
        )
    ),
    CONSTRAINT chk_p038_cr_thermal CHECK (
        thermal_runaway_risk IN ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH')
    ),
    CONSTRAINT chk_p038_cr_maturity CHECK (
        maturity_level IN ('COMMERCIAL', 'EARLY_COMMERCIAL', 'DEMONSTRATION', 'PILOT', 'R_AND_D')
    ),
    CONSTRAINT uq_p038_cr_chemistry UNIQUE (chemistry)
);

INSERT INTO pack038_peak_shaving.ps_chemistry_reference (chemistry, chemistry_full_name, typical_cycle_life, typical_calendar_life_years, typical_rte_pct, typical_dod_pct, energy_density_wh_per_kg, energy_density_wh_per_l, cost_per_kwh_usd, cost_per_kw_usd, c_rate_max, operating_temp_min_c, operating_temp_max_c, thermal_runaway_risk, self_discharge_pct_month, degradation_rate_annual_pct, recyclability_pct, maturity_level, best_application, notes) VALUES
('LFP', 'Lithium Iron Phosphate', 6000, 15, 92.00, 80.00, 160, 325, 250, 350, 1.000, -20, 60, 'VERY_LOW', 2.00, 2.00, 95, 'COMMERCIAL', 'Peak shaving, demand charge management, C&I behind-the-meter', 'Industry standard for stationary C&I storage. Excellent cycle life, thermal stability, and safety. Lower energy density than NMC but superior longevity.'),
('NMC', 'Lithium Nickel Manganese Cobalt', 4000, 12, 94.00, 80.00, 230, 580, 220, 300, 1.500, -10, 50, 'MEDIUM', 3.00, 2.50, 90, 'COMMERCIAL', 'High energy density applications, solar+storage, multi-use', 'Higher energy density than LFP, but shorter cycle life and higher thermal risk. NMC811 improving on cost, NMC532 on safety.'),
('NCA', 'Lithium Nickel Cobalt Aluminum', 3000, 10, 93.00, 80.00, 250, 620, 210, 280, 2.000, -10, 45, 'MEDIUM', 3.50, 3.00, 85, 'COMMERCIAL', 'High power applications, frequency regulation', 'Highest energy density Li-ion. Used primarily in EVs, increasingly in stationary for high C-rate applications.'),
('LTO', 'Lithium Titanate', 15000, 25, 95.00, 95.00, 80, 200, 500, 600, 5.000, -30, 55, 'VERY_LOW', 1.00, 0.50, 95, 'COMMERCIAL', 'Fast response peak shaving, frequency regulation, extreme temperatures', 'Premium chemistry with exceptional cycle life and cold temperature performance. High cost limits to applications requiring fast response.'),
('SODIUM_ION', 'Sodium-Ion Battery', 3000, 12, 90.00, 80.00, 140, 280, 150, 250, 1.000, -20, 60, 'VERY_LOW', 3.00, 3.00, 90, 'EARLY_COMMERCIAL', 'Cost-sensitive stationary storage, grid-scale', 'Emerging alternative to Li-ion using abundant sodium. Lower cost trajectory, no lithium or cobalt. Slightly lower performance than LFP.'),
('FLOW_VANADIUM', 'Vanadium Redox Flow Battery', 20000, 25, 75.00, 100.00, 20, 35, 400, 800, 0.250, 5, 45, 'VERY_LOW', 0.10, 0.10, 99, 'COMMERCIAL', 'Long-duration storage (4-12h), grid-scale peak shaving', 'Unlimited cycle life (electrolyte does not degrade). Best for 4+ hour duration. High upfront cost but excellent for long-duration peak shaving.'),
('FLOW_ZINC_BROMINE', 'Zinc-Bromine Flow Battery', 10000, 15, 72.00, 100.00, 35, 60, 350, 700, 0.250, 5, 45, 'LOW', 0.50, 0.50, 85, 'EARLY_COMMERCIAL', 'Long-duration C&I storage, solar integration', 'Lower cost than vanadium flow. Zinc plating/stripping mechanism. Good for C&I 4-6 hour applications.'),
('FLOW_IRON', 'Iron-Air/Iron Flow Battery', 10000, 25, 50.00, 100.00, 50, 70, 80, 300, 0.100, 5, 50, 'VERY_LOW', 0.50, 0.50, 95, 'DEMONSTRATION', 'Ultra-long-duration storage (100+ hours), seasonal', 'Extremely low cost using abundant iron. Best suited for very long duration (multi-day). Lowest round-trip efficiency limits peak shaving use.'),
('LEAD_ACID', 'Advanced Lead-Acid (AGM/Gel)', 1500, 8, 82.00, 50.00, 40, 100, 150, 200, 0.500, -20, 50, 'VERY_LOW', 5.00, 5.00, 98, 'COMMERCIAL', 'Budget applications, UPS with peak shaving dual-use', 'Mature, low-cost, fully recyclable. Limited cycle life and DOD make it suitable only for low-cycling peak shaving or UPS dual-use.'),
('SOLID_STATE', 'Solid-State Lithium', 8000, 20, 95.00, 90.00, 400, 900, 600, 700, 2.000, -20, 80, 'VERY_LOW', 1.00, 1.00, 90, 'PILOT', 'Next-generation high-density, high-safety storage', 'Emerging technology with solid electrolyte eliminating fire risk. Highest energy density potential. Commercial availability expected 2027-2029.'),
('ZINC_AIR', 'Zinc-Air Battery', 8000, 20, 65.00, 80.00, 300, 400, 100, 200, 0.200, 0, 50, 'VERY_LOW', 1.00, 1.50, 90, 'EARLY_COMMERCIAL', 'Long-duration stationary storage, grid resilience', 'Very high theoretical energy density using zinc and ambient air. Electrically rechargeable variants emerging for stationary applications.'),
('SUPERCAPACITOR', 'Supercapacitor / Ultracapacitor', 1000000, 20, 97.00, 100.00, 8, 15, 10000, 500, 100.000, -40, 70, 'VERY_LOW', 20.00, 2.00, 95, 'COMMERCIAL', 'Short-duration spike suppression, power quality', 'Excellent for sub-second to sub-minute power bursts. Very high cycle life. Too expensive per kWh for sustained peak shaving but ideal for spike suppression.');

GRANT SELECT ON pack038_peak_shaving.ps_chemistry_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: Common US Demand Charge Tariff Templates
-- =============================================================================
INSERT INTO pack038_peak_shaving.ps_tariff_structures (tenant_id, tariff_code, tariff_name, utility_name, state_province, country_code, iso_rto_region, rate_structure, customer_class, demand_window_minutes, effective_date, tiers, tou_periods, ratchet_clause, currency_code, tariff_status) VALUES
('00000000-0000-0000-0000-000000000000', 'SCE-TOU-GS3-B', 'SCE TOU-GS-3 Option B', 'Southern California Edison', 'CA', 'US', 'CAISO', 'TOU_TIERED', 'LARGE_COMMERCIAL', 15, '2024-01-01',
 '[{"period": "SUMMER_ON_PEAK", "rate_per_kw": 23.86}, {"period": "SUMMER_MID_PEAK", "rate_per_kw": 8.02}, {"period": "WINTER_MID_PEAK", "rate_per_kw": 8.02}, {"period": "FACILITIES", "rate_per_kw": 19.50}]',
 '[{"name": "SUMMER_ON_PEAK", "months": [6,7,8,9], "start_hour": 16, "end_hour": 21, "days": "WEEKDAY"}, {"name": "SUMMER_MID_PEAK", "months": [6,7,8,9], "start_hour": 12, "end_hour": 16, "days": "WEEKDAY"}, {"name": "WINTER_MID_PEAK", "months": [10,11,12,1,2,3,4,5], "start_hour": 16, "end_hour": 21, "days": "WEEKDAY"}]',
 '{}', 'USD', 'ACTIVE'),

('00000000-0000-0000-0000-000000000000', 'PECO-GS-LARGE', 'PECO General Service Large', 'PECO Energy', 'PA', 'US', 'PJM', 'TOU', 'LARGE_COMMERCIAL', 15, '2024-01-01',
 '[{"period": "ON_PEAK", "rate_per_kw": 14.70}, {"period": "OFF_PEAK", "rate_per_kw": 4.20}]',
 '[{"name": "ON_PEAK", "months": [1,2,3,4,5,6,7,8,9,10,11,12], "start_hour": 8, "end_hour": 21, "days": "WEEKDAY"}]',
 '{"type": "PERCENTAGE", "pct": 80, "lookback_months": 11, "billing_months": [6,7,8,9]}',
 'USD', 'ACTIVE'),

('00000000-0000-0000-0000-000000000000', 'COMED-RIDER-HEP', 'ComEd High Energy Price', 'Commonwealth Edison', 'IL', 'US', 'PJM', 'FLAT', 'INDUSTRIAL', 30, '2024-01-01',
 '[{"period": "ALL_HOURS", "rate_per_kw": 9.45}]',
 '[{"name": "ALL_HOURS", "months": [1,2,3,4,5,6,7,8,9,10,11,12], "start_hour": 0, "end_hour": 23, "days": "ALL_DAYS"}]',
 '{"type": "PERCENTAGE", "pct": 75, "lookback_months": 11}',
 'USD', 'ACTIVE'),

('00000000-0000-0000-0000-000000000000', 'ONCOR-LHLF', 'Oncor Large High Load Factor', 'Oncor Electric Delivery', 'TX', 'US', 'ERCOT', 'FLAT', 'LARGE_COMMERCIAL', 15, '2024-01-01',
 '[{"period": "ALL_HOURS", "rate_per_kw": 5.80}, {"period": "4CP_TRANSMISSION", "rate_per_kw": 3.75}]',
 '[{"name": "ALL_HOURS", "months": [1,2,3,4,5,6,7,8,9,10,11,12], "start_hour": 0, "end_hour": 23, "days": "ALL_DAYS"}, {"name": "4CP_WINDOW", "months": [6,7,8,9], "start_hour": 14, "end_hour": 18, "days": "WEEKDAY"}]',
 '{}', 'USD', 'ACTIVE'),

('00000000-0000-0000-0000-000000000000', 'EVERSOURCE-G3', 'Eversource G-3 Large Time of Use', 'Eversource Energy', 'CT', 'US', 'ISO_NE', 'TOU_TIERED', 'LARGE_COMMERCIAL', 15, '2024-01-01',
 '[{"period": "ON_PEAK", "rate_per_kw": 18.32}, {"period": "OFF_PEAK", "rate_per_kw": 5.40}, {"period": "FACILITIES", "rate_per_kw": 10.50}]',
 '[{"name": "ON_PEAK", "months": [1,2,3,4,5,6,7,8,9,10,11,12], "start_hour": 12, "end_hour": 20, "days": "WEEKDAY"}]',
 '{"type": "PERCENTAGE", "pct": 80, "lookback_months": 11, "billing_months": [6,7,8,9]}',
 'USD', 'ACTIVE'),

('00000000-0000-0000-0000-000000000000', 'CONED-SC9-RP2', 'Con Edison SC9 Rate II', 'Consolidated Edison', 'NY', 'US', 'NYISO', 'TOU_TIERED', 'LARGE_COMMERCIAL', 15, '2024-01-01',
 '[{"period": "SUMMER_ON_PEAK", "rate_per_kw": 37.30}, {"period": "SUMMER_OFF_PEAK", "rate_per_kw": 6.75}, {"period": "WINTER_ON_PEAK", "rate_per_kw": 25.10}, {"period": "WINTER_OFF_PEAK", "rate_per_kw": 5.90}]',
 '[{"name": "SUMMER_ON_PEAK", "months": [6,7,8,9], "start_hour": 8, "end_hour": 22, "days": "WEEKDAY"}, {"name": "WINTER_ON_PEAK", "months": [10,11,12,1,2,3,4,5], "start_hour": 8, "end_hour": 22, "days": "WEEKDAY"}]',
 '{"type": "PERCENTAGE", "pct": 100, "lookback_months": 11}',
 'USD', 'ACTIVE');

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.pack038_audit_trail IS
    'Pack-level audit trail logging all significant actions across PACK-038 entities for compliance and provenance.';

COMMENT ON MATERIALIZED VIEW pack038_peak_shaving.mv_facility_peak_summary IS
    'Per-facility peak shaving summary with BESS fleet, demand charges, peak events, and financial metrics. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack038_peak_shaving.mv_facility_peak_summary;';
COMMENT ON MATERIALIZED VIEW pack038_peak_shaving.mv_bess_performance_summary IS
    'BESS fleet performance summary with dispatch results, degradation status, and revenue stacking. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack038_peak_shaving.mv_bess_performance_summary;';
COMMENT ON MATERIALIZED VIEW pack038_peak_shaving.mv_portfolio_peak_management IS
    'Portfolio-wide peak management metrics by tenant and region for executive dashboards. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack038_peak_shaving.mv_portfolio_peak_management;';
COMMENT ON VIEW pack038_peak_shaving.v_peak_shaving_dashboard IS
    'Real-time operations dashboard combining load profile, latest peak, BESS state, alerts, ratchet status, and financials.';

COMMENT ON TABLE pack038_peak_shaving.ps_chemistry_reference IS
    'Reference table with typical specifications for common BESS chemistries used in peak shaving applications.';
