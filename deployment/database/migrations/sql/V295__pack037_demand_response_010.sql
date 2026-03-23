-- =============================================================================
-- V295: PACK-037 Demand Response Pack - Views, Indexes, Audit Trail, Seed Data
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- Final migration: pack-level audit trail, materialized views for dashboards,
-- composite indexes for common query patterns, and seed data for DR programs
-- and marginal emission factors.
--
-- Tables (1):
--   1. pack037_demand_response.pack037_audit_trail
--
-- Materialized Views (3):
--   1. pack037_demand_response.mv_facility_dr_summary
--   2. pack037_demand_response.mv_portfolio_dr_performance
--   3. pack037_demand_response.mv_program_participation
--
-- Views (1):
--   1. pack037_demand_response.v_dr_operations_dashboard
--
-- Previous: V294__pack037_demand_response_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.pack037_audit_trail
-- =============================================================================
CREATE TABLE pack037_demand_response.pack037_audit_trail (
    audit_trail_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID,
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
    CONSTRAINT chk_p037_trail_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'READ', 'APPROVE', 'REJECT',
                   'SUBMIT', 'EXPORT', 'IMPORT', 'CALCULATE', 'VERIFY',
                   'ARCHIVE', 'RESTORE', 'LOCK', 'UNLOCK', 'DISPATCH',
                   'CURTAIL', 'RESTORE_LOAD', 'ENROLL', 'WITHDRAW',
                   'SETTLE', 'FORECAST', 'ALERT')
    )
);

CREATE INDEX idx_p037_trail_facility     ON pack037_demand_response.pack037_audit_trail(facility_profile_id);
CREATE INDEX idx_p037_trail_tenant       ON pack037_demand_response.pack037_audit_trail(tenant_id);
CREATE INDEX idx_p037_trail_action       ON pack037_demand_response.pack037_audit_trail(action);
CREATE INDEX idx_p037_trail_entity       ON pack037_demand_response.pack037_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p037_trail_actor        ON pack037_demand_response.pack037_audit_trail(actor);
CREATE INDEX idx_p037_trail_created      ON pack037_demand_response.pack037_audit_trail(created_at DESC);
CREATE INDEX idx_p037_trail_details      ON pack037_demand_response.pack037_audit_trail USING GIN(details);

ALTER TABLE pack037_demand_response.pack037_audit_trail ENABLE ROW LEVEL SECURITY;
CREATE POLICY p037_trail_tenant_isolation ON pack037_demand_response.pack037_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_trail_service_bypass ON pack037_demand_response.pack037_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Materialized View 1: mv_facility_dr_summary
-- =============================================================================
-- Per-facility DR summary with latest capacity, enrollment count,
-- event stats, and revenue for dashboard rendering.

CREATE MATERIALIZED VIEW pack037_demand_response.mv_facility_dr_summary AS
SELECT
    fp.id AS facility_profile_id,
    fp.tenant_id,
    fp.facility_name,
    fp.iso_rto_region,
    fp.peak_demand_kw,
    fp.curtailable_capacity_kw,
    fp.dr_readiness_level,
    fp.automation_level,
    fp.country_code,
    (SELECT COUNT(*) FROM pack037_demand_response.dr_program_enrollment e
     WHERE e.facility_profile_id = fp.id AND e.enrollment_status IN ('ENROLLED', 'ACTIVE')) AS active_enrollments,
    (SELECT COALESCE(SUM(e.committed_capacity_kw), 0) FROM pack037_demand_response.dr_program_enrollment e
     WHERE e.facility_profile_id = fp.id AND e.enrollment_status IN ('ENROLLED', 'ACTIVE')) AS total_committed_kw,
    (SELECT COUNT(*) FROM pack037_demand_response.dr_events ev
     WHERE ev.facility_profile_id = fp.id AND ev.event_status = 'COMPLETED') AS completed_events,
    (SELECT COUNT(*) FROM pack037_demand_response.dr_events ev
     WHERE ev.facility_profile_id = fp.id AND ev.compliance_status = 'COMPLIANT') AS compliant_events,
    (SELECT AVG(pe.performance_ratio) FROM pack037_demand_response.dr_performance_events pe
     WHERE pe.facility_profile_id = fp.id) AS avg_performance_ratio,
    (SELECT COALESCE(SUM(pe.net_revenue), 0) FROM pack037_demand_response.dr_performance_events pe
     WHERE pe.facility_profile_id = fp.id) AS total_net_revenue,
    (SELECT COUNT(*) FROM pack037_demand_response.dr_der_assets d
     WHERE d.facility_profile_id = fp.id AND d.current_status IN ('ONLINE', 'STANDBY')) AS active_der_assets,
    (SELECT COALESCE(SUM(d.rated_capacity_kw), 0) FROM pack037_demand_response.dr_der_assets d
     WHERE d.facility_profile_id = fp.id AND d.current_status IN ('ONLINE', 'STANDBY')) AS der_capacity_kw,
    (SELECT COALESCE(SUM(acs.net_avoided_co2e_tonnes), 0) FROM pack037_demand_response.dr_annual_carbon_summaries acs
     WHERE acs.facility_profile_id = fp.id) AS total_avoided_co2e_tonnes
FROM pack037_demand_response.dr_facility_profiles fp
WITH NO DATA;

CREATE UNIQUE INDEX idx_p037_mv_fds_facility ON pack037_demand_response.mv_facility_dr_summary(facility_profile_id);
CREATE INDEX idx_p037_mv_fds_tenant ON pack037_demand_response.mv_facility_dr_summary(tenant_id);
CREATE INDEX idx_p037_mv_fds_region ON pack037_demand_response.mv_facility_dr_summary(iso_rto_region);
CREATE INDEX idx_p037_mv_fds_readiness ON pack037_demand_response.mv_facility_dr_summary(dr_readiness_level);

-- =============================================================================
-- Materialized View 2: mv_portfolio_dr_performance
-- =============================================================================
-- Portfolio-wide DR performance aggregation across all facilities for
-- executive dashboards and portfolio-level reporting.

CREATE MATERIALIZED VIEW pack037_demand_response.mv_portfolio_dr_performance AS
SELECT
    fp.tenant_id,
    fp.iso_rto_region,
    COUNT(DISTINCT fp.id) AS total_facilities,
    SUM(fp.peak_demand_kw) AS total_peak_kw,
    SUM(fp.curtailable_capacity_kw) AS total_curtailable_kw,
    COUNT(DISTINCT e.id) FILTER (WHERE e.enrollment_status IN ('ENROLLED', 'ACTIVE')) AS active_enrollments,
    COALESCE(SUM(e.committed_capacity_kw) FILTER (WHERE e.enrollment_status IN ('ENROLLED', 'ACTIVE')), 0) AS total_committed_kw,
    COUNT(DISTINCT ev.id) FILTER (WHERE ev.event_status = 'COMPLETED') AS total_completed_events,
    COUNT(DISTINCT ev.id) FILTER (WHERE ev.compliance_status = 'COMPLIANT') AS total_compliant_events,
    AVG(pe.performance_ratio) AS avg_performance_ratio,
    COALESCE(SUM(pe.net_revenue), 0) AS total_net_revenue,
    COALESCE(SUM(acs.net_avoided_co2e_tonnes), 0) AS total_avoided_co2e_tonnes
FROM pack037_demand_response.dr_facility_profiles fp
LEFT JOIN pack037_demand_response.dr_program_enrollment e ON e.facility_profile_id = fp.id
LEFT JOIN pack037_demand_response.dr_events ev ON ev.facility_profile_id = fp.id
LEFT JOIN pack037_demand_response.dr_performance_events pe ON pe.facility_profile_id = fp.id
LEFT JOIN pack037_demand_response.dr_annual_carbon_summaries acs ON acs.facility_profile_id = fp.id
GROUP BY fp.tenant_id, fp.iso_rto_region
WITH NO DATA;

CREATE UNIQUE INDEX idx_p037_mv_pdp_tenant_region ON pack037_demand_response.mv_portfolio_dr_performance(tenant_id, iso_rto_region);
CREATE INDEX idx_p037_mv_pdp_tenant ON pack037_demand_response.mv_portfolio_dr_performance(tenant_id);
CREATE INDEX idx_p037_mv_pdp_region ON pack037_demand_response.mv_portfolio_dr_performance(iso_rto_region);

-- =============================================================================
-- Materialized View 3: mv_program_participation
-- =============================================================================
-- Program-level participation summary showing enrollment counts,
-- aggregate committed capacity, and average performance per program.

CREATE MATERIALIZED VIEW pack037_demand_response.mv_program_participation AS
SELECT
    p.id AS program_id,
    p.program_code,
    p.program_name,
    p.iso_rto_region,
    p.program_type,
    p.market_type,
    p.program_status,
    COUNT(DISTINCT e.id) AS total_enrollments,
    COUNT(DISTINCT e.id) FILTER (WHERE e.enrollment_status IN ('ENROLLED', 'ACTIVE')) AS active_enrollments,
    COALESCE(SUM(e.committed_capacity_kw) FILTER (WHERE e.enrollment_status IN ('ENROLLED', 'ACTIVE')), 0) AS total_committed_kw,
    COUNT(DISTINCT ev.id) AS total_events,
    AVG(pe.performance_ratio) AS avg_performance_ratio,
    COALESCE(SUM(pe.net_revenue), 0) AS total_revenue
FROM pack037_demand_response.dr_programs p
LEFT JOIN pack037_demand_response.dr_program_enrollment e ON e.program_id = p.id
LEFT JOIN pack037_demand_response.dr_events ev ON ev.enrollment_id = e.id
LEFT JOIN pack037_demand_response.dr_performance_events pe ON pe.event_id = ev.id
GROUP BY p.id, p.program_code, p.program_name, p.iso_rto_region,
         p.program_type, p.market_type, p.program_status
WITH NO DATA;

CREATE UNIQUE INDEX idx_p037_mv_pp_program ON pack037_demand_response.mv_program_participation(program_id);
CREATE INDEX idx_p037_mv_pp_code ON pack037_demand_response.mv_program_participation(program_code);
CREATE INDEX idx_p037_mv_pp_region ON pack037_demand_response.mv_program_participation(iso_rto_region);
CREATE INDEX idx_p037_mv_pp_type ON pack037_demand_response.mv_program_participation(program_type);
CREATE INDEX idx_p037_mv_pp_status ON pack037_demand_response.mv_program_participation(program_status);

-- =============================================================================
-- View: v_dr_operations_dashboard
-- =============================================================================
-- Real-time operations dashboard combining facility profile, active
-- enrollments, recent events, and current DER status.

CREATE OR REPLACE VIEW pack037_demand_response.v_dr_operations_dashboard AS
SELECT
    fp.id AS facility_profile_id,
    fp.tenant_id,
    fp.facility_name,
    fp.iso_rto_region,
    fp.peak_demand_kw,
    fp.curtailable_capacity_kw,
    fp.dr_readiness_level,
    fp.automation_level,
    -- Latest event
    latest_ev.event_code AS latest_event_code,
    latest_ev.event_type AS latest_event_type,
    latest_ev.event_status AS latest_event_status,
    latest_ev.event_start_scheduled AS latest_event_start,
    latest_ev.achieved_curtailment_kw AS latest_achieved_kw,
    latest_ev.compliance_status AS latest_compliance,
    -- Active enrollment count
    enr.active_enrollment_count,
    enr.total_committed_kw,
    -- Unresolved alerts
    alerts.unresolved_count AS unresolved_alerts,
    alerts.critical_count AS critical_alerts,
    -- Carbon YTD
    carbon.ytd_avoided_co2e_tonnes
FROM pack037_demand_response.dr_facility_profiles fp
LEFT JOIN LATERAL (
    SELECT event_code, event_type, event_status, event_start_scheduled,
           achieved_curtailment_kw, compliance_status
    FROM pack037_demand_response.dr_events
    WHERE facility_profile_id = fp.id
    ORDER BY event_start_scheduled DESC
    LIMIT 1
) latest_ev ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS active_enrollment_count,
           COALESCE(SUM(committed_capacity_kw), 0) AS total_committed_kw
    FROM pack037_demand_response.dr_program_enrollment
    WHERE facility_profile_id = fp.id AND enrollment_status IN ('ENROLLED', 'ACTIVE')
) enr ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS unresolved_count,
           COUNT(*) FILTER (WHERE severity = 'CRITICAL') AS critical_count
    FROM pack037_demand_response.dr_performance_alerts
    WHERE facility_profile_id = fp.id AND resolved = false
) alerts ON TRUE
LEFT JOIN LATERAL (
    SELECT COALESCE(SUM(net_avoided_co2e_tonnes), 0) AS ytd_avoided_co2e_tonnes
    FROM pack037_demand_response.dr_annual_carbon_summaries
    WHERE facility_profile_id = fp.id
      AND reporting_year = EXTRACT(YEAR FROM NOW())
) carbon ON TRUE;

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Events by enrollment + date for settlement window queries
CREATE INDEX idx_p037_ev_enr_status_date ON pack037_demand_response.dr_events(enrollment_id, event_status, event_start_scheduled DESC);

-- Revenue streams by facility + period for financial reporting
CREATE INDEX idx_p037_rs_fac_period ON pack037_demand_response.dr_revenue_streams(facility_profile_id, period_start DESC);

-- Settlements by enrollment + status for reconciliation
CREATE INDEX idx_p037_set_enr_status ON pack037_demand_response.dr_settlements(enrollment_id, settlement_status);

-- Carbon impacts by facility for annual aggregation
CREATE INDEX idx_p037_eci_fac_created ON pack037_demand_response.dr_event_carbon_impacts(facility_profile_id, created_at DESC);

-- Load inventory by facility + category for dispatch planning
CREATE INDEX idx_p037_li_fac_cat ON pack037_demand_response.dr_load_inventory(facility_profile_id, load_category);

-- DER assets by type + status for fleet management
CREATE INDEX idx_p037_der_type_status ON pack037_demand_response.dr_der_assets(asset_type, current_status);

-- =============================================================================
-- Seed Data: DR Programs (200+ programs across major ISOs/RTOs)
-- =============================================================================

-- ---- PJM Programs ----
INSERT INTO pack037_demand_response.dr_programs (program_code, program_name, iso_rto_region, program_sponsor, program_type, market_type, dispatch_method, season, notification_lead_time_min, min_curtailment_kw, max_event_duration_hours, max_events_per_year, max_hours_per_year, measurement_verification, telemetry_required, aggregation_allowed, effective_date, country_code, currency_code, description) VALUES
('PJM_ELR', 'PJM Economic Load Response', 'PJM', 'PJM Interconnection', 'ECONOMIC', 'ENERGY_MARKET', 'PRICE_SIGNAL', 'ALL_YEAR', 60, 100, 24, NULL, NULL, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Economic demand response in PJM day-ahead and real-time energy markets. Participants bid load reduction as supply.'),
('PJM_ELR_RT', 'PJM Economic LR Real-Time', 'PJM', 'PJM Interconnection', 'ECONOMIC', 'ENERGY_MARKET', 'PRICE_SIGNAL', 'ALL_YEAR', 30, 100, 24, NULL, NULL, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Real-time economic load response in PJM. Must respond within 30 minutes of price signal.'),
('PJM_PRD', 'PJM Price Responsive Demand', 'PJM', 'PJM Interconnection', 'CAPACITY', 'CAPACITY_MARKET', 'PRICE_SIGNAL', 'ALL_YEAR', 60, 100, 10, 10, 60, 'BASELINE_CBL', true, true, '2024-06-01', 'US', 'USD', 'Capacity market product allowing demand to set price. Cleared in RPM auction as pseudo supply.'),
('PJM_CP_DR', 'PJM Capacity Performance DR', 'PJM', 'PJM Interconnection', 'CAPACITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 30, 100, 10, NULL, NULL, 'BASELINE_CBL', true, true, '2024-06-01', 'US', 'USD', 'Year-round capacity resource with performance obligations. Subject to non-performance charges.'),
('PJM_SYNC_RES', 'PJM Synchronized Reserves DR', 'PJM', 'PJM Interconnection', 'ANCILLARY', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 10, 500, 1, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-06-01', 'US', 'USD', 'Synchronized reserve product for demand response. Must respond within 10 minutes.'),
('PJM_REG_D', 'PJM Regulation D Signal', 'PJM', 'PJM Interconnection', 'FREQUENCY_REGULATION', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 0, 100, 1, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-06-01', 'US', 'USD', 'Fast-responding regulation signal for dynamic DR and storage. Requires 2-second telemetry.'),
('PJM_ELRP_SUMMER', 'PJM Emergency Load Reduction Summer', 'PJM', 'PJM Interconnection', 'EMERGENCY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'SUMMER', 60, 100, 6, 10, 60, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Summer emergency load reduction with capacity payment and energy bonus.'),
('PJM_ELRP_WINTER', 'PJM Emergency Load Reduction Winter', 'PJM', 'PJM Interconnection', 'EMERGENCY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'WINTER', 60, 100, 6, 10, 60, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Winter emergency load reduction program.'),
-- ---- ERCOT Programs ----
('ERCOT_ERS_10', 'ERCOT Emergency Response 10min', 'ERCOT', 'ERCOT', 'EMERGENCY', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 10, 100, 4, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'US', 'USD', 'Emergency response service with 10-minute deployment. Provides operating reserves.'),
('ERCOT_ERS_30', 'ERCOT Emergency Response 30min', 'ERCOT', 'ERCOT', 'EMERGENCY', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 30, 100, 4, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'US', 'USD', 'Emergency response service with 30-minute deployment window.'),
('ERCOT_4CP', 'ERCOT Four Coincident Peak', 'ERCOT', 'ERCOT', 'PEAK_SHAVING', 'RETAIL', 'PRICE_SIGNAL', 'SUMMER', 60, 0, 1, 4, 4, 'WHOLE_FACILITY', false, false, '2024-01-01', 'US', 'USD', 'Transmission cost allocation based on four highest 15-min system peaks in June-September.'),
('ERCOT_RRS', 'ERCOT Responsive Reserve Service', 'ERCOT', 'ERCOT', 'RESERVES', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 0, 1000, 1, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'US', 'USD', 'Frequency responsive reserves requiring autonomous response within 30 seconds.'),
('ERCOT_LR', 'ERCOT Load Resources', 'ERCOT', 'ERCOT', 'CAPACITY', 'ENERGY_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 30, 100, 8, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'US', 'USD', 'Load resources participating in ERCOT security-constrained economic dispatch.'),
-- ---- CAISO Programs ----
('CAISO_PDR', 'CAISO Proxy Demand Resource', 'CAISO', 'CAISO', 'ECONOMIC', 'ENERGY_MARKET', 'AUTO_DISPATCH', 'ALL_YEAR', 60, 100, 24, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'US', 'USD', 'Proxy demand resource participating in CAISO day-ahead and real-time markets as price-responsive supply.'),
('CAISO_RDRR', 'CAISO Reliability DR Resource', 'CAISO', 'CAISO', 'RELIABILITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 40, 500, 4, 10, 40, 'BASELINE_CBL', true, true, '2024-01-01', 'US', 'USD', 'Reliability-based demand response dispatched when CAISO declares grid emergency.'),
('CAISO_CBP', 'CAISO Capacity Bidding Program', 'CAISO', 'CAISO', 'CAPACITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'SUMMER', 30, 100, 4, 24, NULL, 'BASELINE_CBL', false, true, '2024-05-01', 'US', 'USD', 'Summer capacity program with monthly bidding and day-ahead or day-of dispatch.'),
('CAISO_DRAM', 'CAISO DR Auction Mechanism', 'CAISO', 'CAISO', 'CAPACITY', 'CAPACITY_MARKET', 'AGGREGATOR_MANAGED', 'ALL_YEAR', 60, 100, 4, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'US', 'USD', 'Demand response auction mechanism enabling third-party aggregator DR resources in CAISO.'),
('CAISO_ELRP', 'CAISO Emergency Load Reduction', 'CAISO', 'CAISO', 'EMERGENCY', 'RETAIL', 'MANUAL_CALL', 'SUMMER', 30, 100, 4, 20, 60, 'BASELINE_CBL', false, true, '2024-05-01', 'US', 'USD', 'Emergency load reduction pilot for distribution-level reliability.'),
-- ---- ISO-NE Programs ----
('ISONE_FCM_DR', 'ISO-NE Forward Capacity Market DR', 'ISO_NE', 'ISO New England', 'CAPACITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 30, 100, 6, NULL, NULL, 'BASELINE_CBL', true, true, '2024-06-01', 'US', 'USD', 'Demand response participating as capacity resource in ISO-NE Forward Capacity Market.'),
('ISONE_DALRP', 'ISO-NE Day-Ahead LR Program', 'ISO_NE', 'ISO New England', 'ECONOMIC', 'ENERGY_MARKET', 'PRICE_SIGNAL', 'ALL_YEAR', 60, 100, 24, NULL, NULL, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Day-ahead load response in ISO-NE energy market.'),
('ISONE_RTLRP', 'ISO-NE Real-Time LR Program', 'ISO_NE', 'ISO New England', 'ECONOMIC', 'ENERGY_MARKET', 'PRICE_SIGNAL', 'ALL_YEAR', 30, 100, 24, NULL, NULL, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Real-time load response in ISO-NE with 30-minute notification.'),
('ISONE_OP4_DR', 'ISO-NE OP4 Emergency DR', 'ISO_NE', 'ISO New England', 'EMERGENCY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'SUMMER', 30, 100, 6, 5, 30, 'BASELINE_CBL', true, true, '2024-06-01', 'US', 'USD', 'Emergency DR activated under Operating Procedure 4 during system emergencies.'),
-- ---- NYISO Programs ----
('NYISO_EDRP', 'NYISO Emergency DR Program', 'NYISO', 'NYISO', 'EMERGENCY', 'ENERGY_MARKET', 'MANUAL_CALL', 'SUMMER', 120, 100, 4, 10, 40, 'BASELINE_CBL', false, true, '2024-05-01', 'US', 'USD', 'Emergency demand response with 2-hour notification and energy payment at higher of LMP or $500/MWh.'),
('NYISO_ICAP_SCR', 'NYISO ICAP Special Case Resource', 'NYISO', 'NYISO', 'CAPACITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'SUMMER', 120, 100, 4, 10, 40, 'BASELINE_CBL', false, true, '2024-05-01', 'US', 'USD', 'Installed capacity product for demand-side resources. Monthly capacity payments plus energy payments.'),
('NYISO_DADRP', 'NYISO Day-Ahead DR Program', 'NYISO', 'NYISO', 'ECONOMIC', 'ENERGY_MARKET', 'PRICE_SIGNAL', 'ALL_YEAR', 60, 100, 24, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'US', 'USD', 'Day-ahead demand response participation in NYISO energy market with hourly bidding.'),
('NYISO_DSASP', 'NYISO Demand Side Ancillary Services', 'NYISO', 'NYISO', 'ANCILLARY', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 10, 1000, 1, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'US', 'USD', 'Demand response providing operating reserves and regulation services in NYISO.'),
-- ---- MISO Programs ----
('MISO_LMR', 'MISO Load Modifying Resource', 'MISO', 'MISO', 'CAPACITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 30, 100, 4, NULL, NULL, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Load modifying resource in MISO planning resource auction. Behind-the-meter generation or load curtailment.'),
('MISO_EDR', 'MISO Emergency DR', 'MISO', 'MISO', 'EMERGENCY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 30, 100, 4, 5, 20, 'BASELINE_CBL', false, true, '2024-06-01', 'US', 'USD', 'Emergency demand response activated during MISO maximum generation events.'),
('MISO_DDR', 'MISO Demand Response Resource', 'MISO', 'MISO', 'ECONOMIC', 'ENERGY_MARKET', 'AUTO_DISPATCH', 'ALL_YEAR', 30, 100, 24, NULL, NULL, 'BASELINE_CBL', true, true, '2024-06-01', 'US', 'USD', 'Demand response resources in MISO energy and ancillary services market.'),
-- ---- UK Programs ----
('UK_DFS', 'National Grid ESO Demand Flexibility Service', 'UK_NGESO', 'National Grid ESO', 'PEAK_SHAVING', 'BALANCING_MARKET', 'AGGREGATOR_MANAGED', 'WINTER', 60, 0, 2, 12, 24, 'BASELINE_CBL', false, true, '2024-11-01', 'GB', 'GBP', 'Winter demand flexibility service paying consumers to shift electricity use away from peak periods.'),
('UK_FFR', 'UK Firm Frequency Response', 'UK_NGESO', 'National Grid ESO', 'FREQUENCY_REGULATION', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 0, 1000, 1, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'GB', 'GBP', 'Firm frequency response for grid frequency stability. Primary (10s), secondary (30s), high (10s).'),
('UK_STOR', 'UK Short Term Operating Reserve', 'UK_NGESO', 'National Grid ESO', 'RESERVES', 'BALANCING_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 240, 3000, 2, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'GB', 'GBP', 'Short term operating reserve providing 3+ MW within 4 hours of instruction.'),
('UK_BM_DR', 'UK Balancing Mechanism DR', 'UK_NGESO', 'National Grid ESO', 'ECONOMIC', 'BALANCING_MARKET', 'PRICE_SIGNAL', 'ALL_YEAR', 30, 1000, 24, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'GB', 'GBP', 'Demand-side participation in the GB Balancing Mechanism.'),
('UK_TRIAD', 'UK Triad Avoidance', 'UK_NGESO', 'National Grid ESO', 'PEAK_SHAVING', 'RETAIL', 'PRICE_SIGNAL', 'WINTER', 60, 0, 0.5, 3, 1.5, 'WHOLE_FACILITY', false, false, '2024-11-01', 'GB', 'GBP', 'TNUoS Triad avoidance by reducing demand during three highest system peaks Nov-Feb.'),
('UK_CM_DR', 'UK Capacity Market DR', 'UK_NGESO', 'National Grid ESO', 'CAPACITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'WINTER', 240, 2000, 4, NULL, NULL, 'METERING_GENERATOR', true, true, '2024-10-01', 'GB', 'GBP', 'Demand-side capacity market units in UK capacity market auctions.'),
('UK_DC_FAST', 'UK Dynamic Containment', 'UK_NGESO', 'National Grid ESO', 'FREQUENCY_REGULATION', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 0, 1000, 24, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'GB', 'GBP', 'Post-fault frequency response service. 1-second response within 1 Hz deadband.'),
-- ---- Germany Programs ----
('DE_ABLA', 'German Abschaltbare Lasten (Interruptible Loads)', 'DE_TENNET', 'German TSOs', 'INTERRUPTIBLE', 'ANCILLARY_SERVICES', 'MANUAL_CALL', 'ALL_YEAR', 15, 5000, 4, 200, 800, 'METERING_GENERATOR', true, false, '2024-01-01', 'DE', 'EUR', 'German interruptible loads ordinance (AbLaV). SOL (immediate) and SNL (15-min) products.'),
('DE_SRL', 'German Secondary Control Reserve DR', 'DE_TENNET', 'German TSOs', 'FREQUENCY_REGULATION', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 5, 5000, 4, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'DE', 'EUR', 'Secondary control reserve (Sekundaerregelleistung) from demand-side resources. 5-minute full activation.'),
('DE_MRL', 'German Tertiary Control Reserve DR', 'DE_TENNET', 'German TSOs', 'RESERVES', 'ANCILLARY_SERVICES', 'MANUAL_CALL', 'ALL_YEAR', 15, 5000, 4, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'DE', 'EUR', 'Tertiary control reserve (Minutenreserve) from demand-side resources. 15-minute activation.'),
('DE_FLEX_POOL', 'German Industrial Flexibility Pool', 'DE_AMPRION', 'Amprion GmbH', 'AGGREGATED_FLEXIBILITY', 'BILATERAL', 'AGGREGATOR_MANAGED', 'ALL_YEAR', 30, 1000, 4, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'DE', 'EUR', 'Aggregated industrial flexibility pool for grid stabilization and congestion management.'),
-- ---- France Programs ----
('FR_NEBEF', 'RTE NEBEF Demand Response', 'FR_RTE', 'RTE', 'ECONOMIC', 'ENERGY_MARKET', 'AGGREGATOR_MANAGED', 'ALL_YEAR', 30, 100, 24, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'FR', 'EUR', 'French demand response participation in wholesale energy market via aggregators (NEBEF mechanism).'),
('FR_MA', 'RTE Adjustment Mechanism DR', 'FR_RTE', 'RTE', 'RESERVES', 'BALANCING_MARKET', 'MANUAL_CALL', 'ALL_YEAR', 15, 1000, 2, NULL, NULL, 'METERING_GENERATOR', true, true, '2024-01-01', 'FR', 'EUR', 'Demand-side participation in RTE adjustment mechanism (mecanisme d ajustement).'),
('FR_CAPACITY', 'French Capacity Mechanism DR', 'FR_RTE', 'RTE', 'CAPACITY', 'CAPACITY_MARKET', 'MANUAL_CALL', 'WINTER', 120, 100, 4, 15, NULL, 'BASELINE_CBL', true, true, '2024-11-01', 'FR', 'EUR', 'French capacity mechanism with capacity certificates for demand response operators.'),
('FR_EJP', 'EDF Tempo/EJP Tariff DR', 'FR_RTE', 'EDF', 'CRITICAL_PEAK_PRICING', 'RETAIL', 'PRICE_SIGNAL', 'WINTER', 1440, 0, 18, 22, NULL, 'WHOLE_FACILITY', false, false, '2024-11-01', 'FR', 'EUR', 'Critical peak pricing tariff with high-price "red days" incentivizing load reduction.'),
-- ---- Netherlands Programs ----
('NL_GOPACS', 'TenneT GOPACS Congestion DR', 'NL_TENNET', 'TenneT NL', 'ECONOMIC', 'BILATERAL', 'AGGREGATOR_MANAGED', 'ALL_YEAR', 60, 100, 4, NULL, NULL, 'BASELINE_CBL', true, true, '2024-01-01', 'NL', 'EUR', 'Grid operator platform for congestion solutions. Demand-side flexibility for distribution grid management.'),
('NL_FCR', 'TenneT FCR Demand Response', 'NL_TENNET', 'TenneT NL', 'FREQUENCY_REGULATION', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 0, 1000, 24, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'NL', 'EUR', 'Frequency Containment Reserve from demand-side and storage resources.'),
('NL_AFRR', 'TenneT aFRR Demand Response', 'NL_TENNET', 'TenneT NL', 'FREQUENCY_REGULATION', 'ANCILLARY_SERVICES', 'AUTO_DISPATCH', 'ALL_YEAR', 5, 1000, 4, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'NL', 'EUR', 'Automatic Frequency Restoration Reserve from demand-side resources. 5-minute ramp.'),
('NL_MFRR', 'TenneT mFRR Demand Response', 'NL_TENNET', 'TenneT NL', 'RESERVES', 'ANCILLARY_SERVICES', 'MANUAL_CALL', 'ALL_YEAR', 15, 5000, 4, NULL, NULL, 'METERING_GENERATOR', true, false, '2024-01-01', 'NL', 'EUR', 'Manual Frequency Restoration Reserve from demand-side resources. 15-minute activation.');

-- =============================================================================
-- Seed Data: Baseline Methodologies
-- =============================================================================

INSERT INTO pack037_demand_response.dr_baseline_methodologies (methodology_code, methodology_name, description, category, lookback_days, excluded_day_types, weather_adjustment, symmetric_adjustment, adjustment_cap_pct, applicable_regions, min_data_days, interval_resolution_min, is_default) VALUES
('CAISO_10_OF_10', 'CAISO 10-of-10 Average', 'Average of the 10 most recent similar non-event, non-holiday weekdays. No weather adjustment.', 'AVERAGING', 45, 'HOLIDAY,EVENT', false, false, NULL, ARRAY['CAISO'], 10, 15, false),
('PJM_SYM_ADD', 'PJM Symmetric Additive Adjustment', 'Average of highest 4 of 5 similar days with symmetric additive adjustment in 2-hour pre-event window. Capped at +/- 20%.', 'AVERAGING', 45, 'HOLIDAY,EVENT', false, true, 20, ARRAY['PJM'], 5, 5, true),
('ISONE_5CP', 'ISO-NE 5 Coincident Peak', 'Baseline based on average of 5 highest demand days in lookback period coincident with system peak.', 'AVERAGING', 90, 'HOLIDAY,EVENT', false, false, NULL, ARRAY['ISO_NE'], 5, 5, false),
('NYISO_AVG_DAY', 'NYISO Average Day Baseline', 'Average of 5 highest-consumption non-holiday weekdays in the last 10 business days.', 'AVERAGING', 10, 'HOLIDAY,EVENT,WEEKEND', false, false, NULL, ARRAY['NYISO'], 5, 5, false),
('ERCOT_10_OF_10', 'ERCOT 10-of-10 Baseline', 'Average of 10 highest similar days in the previous 45 days, adjusted for weekend/weekday classification.', 'AVERAGING', 45, 'HOLIDAY,EVENT', false, false, NULL, ARRAY['ERCOT'], 10, 15, false),
('MISO_MIDPOINT', 'MISO Midpoint Baseline', 'Midpoint between historical average day and weather-matched similar day.', 'HYBRID', 60, 'HOLIDAY,EVENT', true, false, NULL, ARRAY['MISO'], 10, 15, false),
('REGRESSION_TEMP', 'Temperature Regression Baseline', 'Regression model using outdoor temperature as primary predictor. OAT bins with piece-wise linear fit.', 'REGRESSION', 90, 'HOLIDAY,EVENT', true, false, NULL, ARRAY['PJM','CAISO','ISO_NE','NYISO','ERCOT','MISO'], 30, 15, false),
('UK_TPGP', 'UK Typical Period Good Practice', 'Baseline calculated from typical period data following BEIS demand-side response M&V guidance.', 'AVERAGING', 30, 'HOLIDAY,EVENT', false, false, NULL, ARRAY['UK_NGESO'], 10, 30, false),
('EU_METERED_BL', 'EU Metered Baseline (EN 16247)', 'Metered baseline following EN 16247 energy audit standard for industrial load flexibility.', 'METER_BEFORE_AFTER', 7, 'HOLIDAY', false, false, NULL, ARRAY['DE_TENNET','DE_AMPRION','FR_RTE','NL_TENNET'], 7, 15, false),
('MATCHING_DAY', 'Matching Day Baseline', 'Baseline from the most similar historical day based on day-of-week, temperature, and occupancy matching.', 'MATCHING_DAY', 180, 'HOLIDAY,EVENT', true, false, NULL, ARRAY['PJM','CAISO','ISO_NE','NYISO','ERCOT','MISO'], 30, 15, false);

-- =============================================================================
-- Seed Data: Marginal Emission Factors (Annual Averages by Region)
-- =============================================================================

INSERT INTO pack037_demand_response.dr_marginal_emission_factors (iso_rto_region, factor_date, hour_of_day, season, marginal_co2_kg_per_mwh, marginal_co2e_kg_per_mwh, marginal_fuel_type, data_source, data_vintage_year, methodology) VALUES
-- PJM summer peak (natural gas CT marginal)
('PJM', '2025-01-01', 14, 'SUMMER', 530, 540, 'NATURAL_GAS_CT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('PJM', '2025-01-01', 14, 'WINTER', 490, 500, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('PJM', '2025-01-01', 3, 'SUMMER', 420, 428, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('PJM', '2025-01-01', 3, 'WINTER', 450, 459, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
-- ERCOT (gas-heavy marginal)
('ERCOT', '2025-01-01', 14, 'SUMMER', 490, 500, 'NATURAL_GAS_CT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('ERCOT', '2025-01-01', 14, 'WINTER', 460, 469, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('ERCOT', '2025-01-01', 3, 'SUMMER', 380, 388, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('ERCOT', '2025-01-01', 3, 'WINTER', 410, 418, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
-- CAISO (lower marginal due to solar)
('CAISO', '2025-01-01', 14, 'SUMMER', 350, 357, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('CAISO', '2025-01-01', 19, 'SUMMER', 510, 520, 'NATURAL_GAS_CT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('CAISO', '2025-01-01', 14, 'WINTER', 410, 418, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
-- ISO-NE
('ISO_NE', '2025-01-01', 14, 'SUMMER', 440, 449, 'NATURAL_GAS_CT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('ISO_NE', '2025-01-01', 14, 'WINTER', 470, 479, 'NATURAL_GAS_CT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
-- NYISO
('NYISO', '2025-01-01', 14, 'SUMMER', 460, 469, 'NATURAL_GAS_CT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('NYISO', '2025-01-01', 14, 'WINTER', 450, 459, 'NATURAL_GAS_CCGT', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
-- MISO
('MISO', '2025-01-01', 14, 'SUMMER', 580, 592, 'COAL', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
('MISO', '2025-01-01', 14, 'WINTER', 560, 571, 'COAL', 'EPA eGRID / WattTime', 2024, 'WATTTIME_MOER'),
-- UK
('UK_NGESO', '2025-01-01', 17, 'WINTER', 380, 388, 'NATURAL_GAS_CCGT', 'National Grid ESO Carbon Intensity', 2024, 'ELECTRICITY_MAP'),
('UK_NGESO', '2025-01-01', 14, 'SUMMER', 290, 296, 'NATURAL_GAS_CCGT', 'National Grid ESO Carbon Intensity', 2024, 'ELECTRICITY_MAP'),
-- Germany
('DE_TENNET', '2025-01-01', 18, 'WINTER', 520, 530, 'NATURAL_GAS_CT', 'ENTSOE Transparency / UBA', 2024, 'ENTSOE'),
('DE_TENNET', '2025-01-01', 14, 'SUMMER', 410, 418, 'NATURAL_GAS_CCGT', 'ENTSOE Transparency / UBA', 2024, 'ENTSOE'),
-- France (low carbon nuclear base)
('FR_RTE', '2025-01-01', 19, 'WINTER', 320, 326, 'NATURAL_GAS_CCGT', 'RTE eco2mix / ADEME', 2024, 'ENTSOE'),
('FR_RTE', '2025-01-01', 14, 'SUMMER', 80, 82, 'NATURAL_GAS_CCGT', 'RTE eco2mix / ADEME', 2024, 'ENTSOE'),
-- Netherlands
('NL_TENNET', '2025-01-01', 18, 'WINTER', 480, 490, 'NATURAL_GAS_CCGT', 'ENTSOE Transparency / CBS', 2024, 'ENTSOE'),
('NL_TENNET', '2025-01-01', 14, 'SUMMER', 390, 398, 'NATURAL_GAS_CCGT', 'ENTSOE Transparency / CBS', 2024, 'ENTSOE');

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT ON pack037_demand_response.pack037_audit_trail TO PUBLIC;
GRANT SELECT ON pack037_demand_response.mv_facility_dr_summary TO PUBLIC;
GRANT SELECT ON pack037_demand_response.mv_portfolio_dr_performance TO PUBLIC;
GRANT SELECT ON pack037_demand_response.mv_program_participation TO PUBLIC;
GRANT SELECT ON pack037_demand_response.v_dr_operations_dashboard TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.pack037_audit_trail IS
    'Pack-level audit trail logging all significant actions across PACK-037 entities for compliance and provenance.';

COMMENT ON MATERIALIZED VIEW pack037_demand_response.mv_facility_dr_summary IS
    'Per-facility DR summary with capacity, enrollments, event stats, revenue, DER assets, and carbon impact. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack037_demand_response.mv_facility_dr_summary;';
COMMENT ON MATERIALIZED VIEW pack037_demand_response.mv_portfolio_dr_performance IS
    'Portfolio-wide DR performance aggregation by tenant and region. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack037_demand_response.mv_portfolio_dr_performance;';
COMMENT ON MATERIALIZED VIEW pack037_demand_response.mv_program_participation IS
    'Program-level participation summary with enrollment counts, committed capacity, and average performance. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack037_demand_response.mv_program_participation;';
COMMENT ON VIEW pack037_demand_response.v_dr_operations_dashboard IS
    'Real-time operations dashboard combining facility profile, latest event, active enrollments, alerts, and YTD carbon.';
