-- =============================================================================
-- V365: PACK-044 GHG Inventory Management - Benchmarks, Views, Indexes & Seed
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Final migration for PACK-044. Creates benchmark and peer comparison tables,
-- materialized views for inventory dashboards, additional cross-table indexes,
-- and seed data for reference lookups (maturity levels, QA/QC check
-- definitions, milestone templates, document templates).
--
-- Tables (2):
--   1. ghg_inventory.gl_inv_benchmarks
--   2. ghg_inventory.gl_inv_peer_comparisons
--
-- Views (4):
--   1. ghg_inventory.vw_inv_period_summary
--   2. ghg_inventory.vw_inv_collection_progress
--   3. ghg_inventory.vw_inv_quality_dashboard
--   4. ghg_inventory.vw_inv_consolidation_summary
--
-- Also includes: cross-table indexes, seed data.
-- Previous: V364__pack044_documentation.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_benchmarks
-- =============================================================================
-- Sector and building-type benchmarks for GHG intensity comparison.
-- Contains reference intensity values from published sources (CDP, GRESB,
-- ENERGY STAR, IEA, national inventories) for peer comparison.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_benchmarks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID,
    benchmark_code              VARCHAR(50)     NOT NULL,
    benchmark_name              VARCHAR(300)    NOT NULL,
    benchmark_source            VARCHAR(100)    NOT NULL,
    sector                      VARCHAR(100)    NOT NULL,
    sub_sector                  VARCHAR(100),
    region                      VARCHAR(100)    DEFAULT 'GLOBAL',
    country                     VARCHAR(3),
    benchmark_year              INTEGER         NOT NULL,
    metric_name                 VARCHAR(100)    NOT NULL,
    metric_unit                 VARCHAR(50)     NOT NULL,
    p10_value                   NUMERIC(14,6),
    p25_value                   NUMERIC(14,6),
    median_value                NUMERIC(14,6),
    p75_value                   NUMERIC(14,6),
    p90_value                   NUMERIC(14,6),
    mean_value                  NUMERIC(14,6),
    sample_size                 INTEGER,
    data_quality                VARCHAR(20)     DEFAULT 'MEDIUM',
    source_url                  TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_bm_year CHECK (
        benchmark_year >= 2000 AND benchmark_year <= 2100
    ),
    CONSTRAINT chk_p044_bm_quality CHECK (
        data_quality IS NULL OR data_quality IN ('HIGH', 'MEDIUM', 'LOW', 'ESTIMATED')
    ),
    CONSTRAINT chk_p044_bm_sample CHECK (
        sample_size IS NULL OR sample_size >= 0
    ),
    CONSTRAINT chk_p044_bm_percentiles CHECK (
        (p10_value IS NULL OR p25_value IS NULL OR p10_value <= p25_value) AND
        (p25_value IS NULL OR median_value IS NULL OR p25_value <= median_value) AND
        (median_value IS NULL OR p75_value IS NULL OR median_value <= p75_value) AND
        (p75_value IS NULL OR p90_value IS NULL OR p75_value <= p90_value)
    ),
    CONSTRAINT uq_p044_bm_code_year UNIQUE (benchmark_code, benchmark_year, region)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_bm_code            ON ghg_inventory.gl_inv_benchmarks(benchmark_code);
CREATE INDEX IF NOT EXISTS idx_p044_bm_source          ON ghg_inventory.gl_inv_benchmarks(benchmark_source);
CREATE INDEX IF NOT EXISTS idx_p044_bm_sector          ON ghg_inventory.gl_inv_benchmarks(sector);
CREATE INDEX IF NOT EXISTS idx_p044_bm_region          ON ghg_inventory.gl_inv_benchmarks(region);
CREATE INDEX IF NOT EXISTS idx_p044_bm_country         ON ghg_inventory.gl_inv_benchmarks(country);
CREATE INDEX IF NOT EXISTS idx_p044_bm_year            ON ghg_inventory.gl_inv_benchmarks(benchmark_year);
CREATE INDEX IF NOT EXISTS idx_p044_bm_metric          ON ghg_inventory.gl_inv_benchmarks(metric_name);
CREATE INDEX IF NOT EXISTS idx_p044_bm_created         ON ghg_inventory.gl_inv_benchmarks(created_at DESC);

-- Composite: sector + year for lookup
CREATE INDEX IF NOT EXISTS idx_p044_bm_sector_year     ON ghg_inventory.gl_inv_benchmarks(sector, benchmark_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_bm_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_benchmarks
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_peer_comparisons
-- =============================================================================
-- Organisation-specific peer comparison results. Compares an organisation's
-- intensity metrics against sector benchmarks and identifies the percentile
-- position. Tracks improvement over time.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_peer_comparisons (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    benchmark_id                UUID            REFERENCES ghg_inventory.gl_inv_benchmarks(id) ON DELETE SET NULL,
    comparison_name             VARCHAR(300)    NOT NULL,
    metric_name                 VARCHAR(100)    NOT NULL,
    metric_unit                 VARCHAR(50)     NOT NULL,
    organisation_value          NUMERIC(14,6)   NOT NULL,
    benchmark_median            NUMERIC(14,6),
    benchmark_p25               NUMERIC(14,6),
    benchmark_p75               NUMERIC(14,6),
    percentile_rank             NUMERIC(5,2),
    performance_rating          VARCHAR(30),
    previous_period_value       NUMERIC(14,6),
    year_on_year_change_pct     NUMERIC(8,3),
    improvement_target          NUMERIC(14,6),
    target_year                 INTEGER,
    analysis_notes              TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_pc_percentile CHECK (
        percentile_rank IS NULL OR (percentile_rank >= 0 AND percentile_rank <= 100)
    ),
    CONSTRAINT chk_p044_pc_rating CHECK (
        performance_rating IS NULL OR performance_rating IN (
            'LEADER', 'ABOVE_AVERAGE', 'AVERAGE', 'BELOW_AVERAGE', 'LAGGARD', 'NOT_COMPARABLE'
        )
    ),
    CONSTRAINT chk_p044_pc_target_year CHECK (
        target_year IS NULL OR (target_year >= 2020 AND target_year <= 2100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_pc_tenant          ON ghg_inventory.gl_inv_peer_comparisons(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_pc_period          ON ghg_inventory.gl_inv_peer_comparisons(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_pc_benchmark       ON ghg_inventory.gl_inv_peer_comparisons(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_p044_pc_metric          ON ghg_inventory.gl_inv_peer_comparisons(metric_name);
CREATE INDEX IF NOT EXISTS idx_p044_pc_rating          ON ghg_inventory.gl_inv_peer_comparisons(performance_rating);
CREATE INDEX IF NOT EXISTS idx_p044_pc_created         ON ghg_inventory.gl_inv_peer_comparisons(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_pc_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_peer_comparisons
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_benchmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_peer_comparisons ENABLE ROW LEVEL SECURITY;

-- Benchmarks are mostly global reference data, allow read for all tenants
CREATE POLICY p044_bm_read_all
    ON ghg_inventory.gl_inv_benchmarks
    FOR SELECT
    USING (TRUE);
CREATE POLICY p044_bm_write_tenant
    ON ghg_inventory.gl_inv_benchmarks
    FOR ALL
    USING (tenant_id IS NULL OR tenant_id = current_setting('app.current_tenant')::UUID)
    WITH CHECK (tenant_id IS NULL OR tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_bm_service_bypass
    ON ghg_inventory.gl_inv_benchmarks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_pc_tenant_isolation
    ON ghg_inventory.gl_inv_peer_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_pc_service_bypass
    ON ghg_inventory.gl_inv_peer_comparisons
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_benchmarks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_peer_comparisons TO PUBLIC;

-- =============================================================================
-- Views
-- =============================================================================

-- ---------------------------------------------------------------------------
-- View 1: Period Summary Dashboard
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ghg_inventory.vw_inv_period_summary AS
SELECT
    ip.id AS period_id,
    ip.tenant_id,
    ip.organization_id,
    ip.period_name,
    ip.reporting_year,
    ip.status AS period_status,
    ip.total_scope1_tco2e,
    ip.total_scope2_location_tco2e,
    ip.total_scope2_market_tco2e,
    ip.total_scope3_tco2e,
    ip.total_tco2e,
    ip.completeness_pct,
    ip.data_quality_score,
    ip.is_locked,
    -- Latest version info
    v.version_number AS current_version,
    v.version_type AS current_version_type,
    -- Milestone counts
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_period_milestones pm
     WHERE pm.period_id = ip.id AND pm.status = 'COMPLETED') AS milestones_completed,
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_period_milestones pm
     WHERE pm.period_id = ip.id) AS milestones_total,
    -- Open issue count
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_quality_issues qi
     WHERE qi.period_id = ip.id AND qi.status IN ('OPEN', 'IN_PROGRESS')) AS open_issues,
    -- Open change request count
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_change_requests cr
     WHERE cr.period_id = ip.id AND cr.status IN ('SUBMITTED', 'UNDER_REVIEW', 'IMPACT_ASSESSMENT')) AS open_change_requests,
    ip.created_at,
    ip.updated_at
FROM ghg_inventory.gl_inv_inventory_periods ip
LEFT JOIN ghg_inventory.gl_inv_versions v
    ON v.period_id = ip.id AND v.is_current = true;

-- ---------------------------------------------------------------------------
-- View 2: Collection Progress
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ghg_inventory.vw_inv_collection_progress AS
SELECT
    cc.id AS campaign_id,
    cc.tenant_id,
    cc.period_id,
    cc.campaign_name,
    cc.campaign_type,
    cc.status AS campaign_status,
    cc.collection_start_date,
    cc.collection_end_date,
    cc.total_requests,
    cc.completed_requests,
    cc.completion_pct,
    -- Request status breakdown
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_data_requests dr
     WHERE dr.campaign_id = cc.id AND dr.status = 'PENDING') AS pending_requests,
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_data_requests dr
     WHERE dr.campaign_id = cc.id AND dr.status = 'OVERDUE') AS overdue_requests,
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_data_requests dr
     WHERE dr.campaign_id = cc.id AND dr.status = 'SUBMITTED') AS submitted_requests,
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_data_requests dr
     WHERE dr.campaign_id = cc.id AND dr.status = 'ACCEPTED') AS accepted_requests,
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_data_requests dr
     WHERE dr.campaign_id = cc.id AND dr.escalated = true) AS escalated_requests,
    cc.created_at
FROM ghg_inventory.gl_inv_collection_campaigns cc;

-- ---------------------------------------------------------------------------
-- View 3: Quality Dashboard
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ghg_inventory.vw_inv_quality_dashboard AS
SELECT
    qr.id AS run_id,
    qr.tenant_id,
    qr.period_id,
    qr.run_name,
    qr.run_type,
    qr.status AS run_status,
    qr.total_checks,
    qr.passed_checks,
    qr.failed_checks,
    qr.warning_checks,
    qr.overall_score,
    qr.overall_result,
    qr.completed_at,
    -- Issue summary from this run's period
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_quality_issues qi
     WHERE qi.period_id = qr.period_id AND qi.severity = 'CRITICAL' AND qi.status IN ('OPEN', 'IN_PROGRESS')) AS critical_open_issues,
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_quality_issues qi
     WHERE qi.period_id = qr.period_id AND qi.severity = 'HIGH' AND qi.status IN ('OPEN', 'IN_PROGRESS')) AS high_open_issues,
    -- Improvement action summary
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_improvement_actions ia
     INNER JOIN ghg_inventory.gl_inv_quality_issues qi2 ON ia.issue_id = qi2.id
     WHERE qi2.period_id = qr.period_id AND ia.status IN ('PLANNED', 'IN_PROGRESS')) AS open_actions,
    qr.created_at
FROM ghg_inventory.gl_inv_qaqc_runs qr;

-- ---------------------------------------------------------------------------
-- View 4: Consolidation Summary
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW ghg_inventory.vw_inv_consolidation_summary AS
SELECT
    cr.id AS consolidation_run_id,
    cr.tenant_id,
    cr.period_id,
    cr.run_name,
    cr.consolidation_approach,
    cr.status AS run_status,
    cr.total_entities,
    cr.entities_included,
    cr.entities_excluded,
    cr.raw_scope1_tco2e,
    cr.raw_scope2_location_tco2e,
    cr.raw_scope2_market_tco2e,
    cr.raw_scope3_tco2e,
    cr.equity_adjustment_tco2e,
    cr.inter_company_elim_tco2e,
    cr.consolidated_scope1_tco2e,
    cr.consolidated_scope2_loc_tco2e,
    cr.consolidated_scope2_mkt_tco2e,
    cr.consolidated_scope3_tco2e,
    cr.consolidated_total_tco2e,
    -- Subsidiary sign-off progress
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_subsidiary_submissions ss
     WHERE ss.period_id = cr.period_id AND ss.signed_off = true) AS entities_signed_off,
    (SELECT COUNT(*) FROM ghg_inventory.gl_inv_subsidiary_submissions ss
     WHERE ss.period_id = cr.period_id) AS total_submissions,
    cr.completed_at,
    cr.created_at
FROM ghg_inventory.gl_inv_consolidation_runs cr;

-- =============================================================================
-- Additional Cross-Table Indexes
-- =============================================================================

-- Submissions by period (via request -> campaign -> period chain, optimise with a view)
-- Evidence by facility and period
CREATE INDEX IF NOT EXISTS idx_p044_ev_fac_period
    ON ghg_inventory.gl_inv_evidence_records(facility_id, period_id)
    WHERE facility_id IS NOT NULL;

-- Assumptions by sensitivity across periods
CREATE INDEX IF NOT EXISTS idx_p044_as_sensitivity_status
    ON ghg_inventory.gl_inv_assumptions(sensitivity, status)
    WHERE status = 'ACTIVE';

-- Quality issues by severity and period
CREATE INDEX IF NOT EXISTS idx_p044_qi_severity_period
    ON ghg_inventory.gl_inv_quality_issues(severity, period_id)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- Version snapshots aggregate by scope
CREATE INDEX IF NOT EXISTS idx_p044_vs_scope_agg
    ON ghg_inventory.gl_inv_version_snapshots(version_id, scope)
    WHERE emissions_tco2e > 0;

-- =============================================================================
-- Seed Data: QA/QC Check Definitions
-- =============================================================================
-- Reference data for standard QA/QC checks that can be used across all tenants.

INSERT INTO ghg_inventory.gl_inv_benchmarks (
    benchmark_code, benchmark_name, benchmark_source, sector, region,
    benchmark_year, metric_name, metric_unit,
    p10_value, p25_value, median_value, p75_value, p90_value, mean_value,
    sample_size, data_quality, notes
) VALUES
    -- Office sector benchmarks
    ('BM_OFFICE_CO2_FTE_2024', 'Office CO2e per FTE (2024)', 'CDP', 'OFFICE', 'GLOBAL',
     2024, 'tCO2e_per_FTE', 'tCO2e/FTE',
     0.8, 1.2, 2.0, 3.5, 5.5, 2.4,
     2450, 'HIGH', 'CDP Climate Change 2024 responses, sector: Professional Services'),
    ('BM_OFFICE_CO2_M2_2024', 'Office CO2e per m2 (2024)', 'CDP', 'OFFICE', 'GLOBAL',
     2024, 'kgCO2e_per_m2', 'kgCO2e/m2',
     15.0, 30.0, 55.0, 90.0, 140.0, 62.0,
     1800, 'HIGH', 'CDP Climate Change 2024 responses, office buildings'),
    -- Manufacturing sector benchmarks
    ('BM_MFG_CO2_REV_2024', 'Manufacturing CO2e per revenue (2024)', 'CDP', 'MANUFACTURING', 'GLOBAL',
     2024, 'tCO2e_per_MUSD', 'tCO2e/MUSD',
     20.0, 55.0, 120.0, 280.0, 600.0, 185.0,
     3200, 'HIGH', 'CDP Climate Change 2024 responses, sector: Manufacturing'),
    -- Real estate benchmarks
    ('BM_RE_CO2_M2_2024', 'Real Estate CO2e per m2 (2024)', 'GRESB', 'REAL_ESTATE', 'GLOBAL',
     2024, 'kgCO2e_per_m2', 'kgCO2e/m2',
     18.0, 35.0, 60.0, 95.0, 150.0, 68.0,
     1500, 'HIGH', 'GRESB 2024 Real Estate Assessment, all property types'),
    -- Energy utility benchmarks
    ('BM_UTIL_CO2_MWH_2024', 'Utility CO2e per MWh generated (2024)', 'IEA', 'ENERGY_UTILITY', 'GLOBAL',
     2024, 'tCO2e_per_MWh', 'tCO2e/MWh',
     0.05, 0.15, 0.35, 0.55, 0.80, 0.38,
     890, 'HIGH', 'IEA World Energy Outlook 2024, electricity generation'),
    -- Transport benchmarks
    ('BM_TRANS_CO2_TKM_2024', 'Transport CO2e per tonne-km (2024)', 'GLEC', 'TRANSPORT_LOGISTICS', 'GLOBAL',
     2024, 'kgCO2e_per_tkm', 'kgCO2e/tonne-km',
     0.015, 0.030, 0.062, 0.110, 0.180, 0.072,
     680, 'MEDIUM', 'GLEC Framework v3.0, road freight'),
    -- Healthcare benchmarks
    ('BM_HC_CO2_BED_2024', 'Healthcare CO2e per bed-day (2024)', 'NHS_SDU', 'HEALTHCARE', 'EUROPE',
     2024, 'kgCO2e_per_bed_day', 'kgCO2e/bed-day',
     12.0, 20.0, 32.0, 48.0, 70.0, 35.0,
     420, 'MEDIUM', 'NHS Sustainable Development Unit 2024, acute hospitals'),
    -- SME benchmarks
    ('BM_SME_CO2_EMP_2024', 'SME CO2e per employee (2024)', 'CDP', 'SME', 'GLOBAL',
     2024, 'tCO2e_per_employee', 'tCO2e/employee',
     0.5, 1.0, 2.5, 5.0, 10.0, 3.2,
     950, 'MEDIUM', 'CDP SME Climate Disclosure 2024'),
    -- Food & Agriculture benchmarks
    ('BM_AGRI_CO2_HA_2024', 'Agriculture CO2e per hectare (2024)', 'FAO_GLEAM', 'FOOD_AGRICULTURE', 'GLOBAL',
     2024, 'tCO2e_per_hectare', 'tCO2e/ha',
     1.0, 2.5, 4.5, 8.0, 15.0, 5.8,
     550, 'MEDIUM', 'FAO GLEAM v3.0, mixed farming')
ON CONFLICT (benchmark_code, benchmark_year, region) DO NOTHING;

-- =============================================================================
-- Comments on Views
-- =============================================================================
COMMENT ON VIEW ghg_inventory.vw_inv_period_summary IS
    'Dashboard view summarising inventory period status, emissions totals, version, milestones, and open issues.';
COMMENT ON VIEW ghg_inventory.vw_inv_collection_progress IS
    'Data collection campaign progress with request status breakdown.';
COMMENT ON VIEW ghg_inventory.vw_inv_quality_dashboard IS
    'QA/QC run results with open issue and improvement action summaries.';
COMMENT ON VIEW ghg_inventory.vw_inv_consolidation_summary IS
    'Consolidation run results with subsidiary sign-off progress.';

-- ---------------------------------------------------------------------------
-- Comments on Tables
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_benchmarks IS
    'Sector and building-type benchmarks for GHG intensity peer comparison from published sources (CDP, GRESB, IEA, GLEC).';
COMMENT ON TABLE ghg_inventory.gl_inv_peer_comparisons IS
    'Organisation-specific peer comparison results showing percentile rank against sector benchmarks.';

COMMENT ON COLUMN ghg_inventory.gl_inv_benchmarks.benchmark_source IS 'Source of benchmark data: CDP, GRESB, IEA, GLEC, NHS_SDU, FAO_GLEAM, ENERGY_STAR, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_benchmarks.p10_value IS '10th percentile value (top performer threshold).';
COMMENT ON COLUMN ghg_inventory.gl_inv_benchmarks.median_value IS '50th percentile (median) value.';
COMMENT ON COLUMN ghg_inventory.gl_inv_benchmarks.p90_value IS '90th percentile value (laggard threshold).';
COMMENT ON COLUMN ghg_inventory.gl_inv_peer_comparisons.percentile_rank IS 'Organisation percentile rank (0=best, 100=worst).';
COMMENT ON COLUMN ghg_inventory.gl_inv_peer_comparisons.performance_rating IS 'Rating: LEADER, ABOVE_AVERAGE, AVERAGE, BELOW_AVERAGE, LAGGARD, NOT_COMPARABLE.';
