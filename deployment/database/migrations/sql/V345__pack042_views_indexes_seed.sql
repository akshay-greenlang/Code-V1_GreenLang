-- =============================================================================
-- V345: PACK-042 Scope 3 Starter Pack - Views, Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates analytical views, materialized views, additional composite indexes
-- for common query patterns, and seed data. Views provide pre-joined
-- perspectives for dashboards and reporting. Seed data includes framework
-- requirement checklists and RBAC policies.
--
-- Views (3):
--   1. ghg_accounting_scope3.v_inventory_summary
--   2. ghg_accounting_scope3.v_category_comparison
--   3. ghg_accounting_scope3.v_supplier_dashboard
--
-- Materialized Views (1):
--   4. ghg_accounting_scope3.mv_scope3_benchmarks
--
-- Also includes: additional composite indexes, seed data, grants, comments.
-- Previous: V344__pack042_compliance_reporting.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- View 1: ghg_accounting_scope3.v_inventory_summary
-- =============================================================================
-- Dashboard view providing a per-inventory summary with all category totals,
-- screening status, data quality overview, and reconciled totals. Designed
-- for single-query dashboard rendering of the Scope 3 inventory overview.

CREATE OR REPLACE VIEW ghg_accounting_scope3.v_inventory_summary AS
SELECT
    si.id                        AS inventory_id,
    si.tenant_id,
    si.org_id,
    si.reporting_year,
    si.status                    AS inventory_status,
    si.methodology_version,
    si.boundary_approach,
    si.gwp_source,
    si.screening_completed,
    si.categories_assessed,
    si.categories_relevant,
    si.categories_calculated,
    -- Category totals as JSONB
    (
        SELECT jsonb_object_agg(
            cr.category::TEXT,
            jsonb_build_object(
                'tco2e', cr.total_tco2e,
                'methodology_tier', cr.methodology_tier::TEXT,
                'data_quality', cr.data_quality_rating,
                'pct_of_total', cr.pct_of_total_scope3,
                'status', cr.status
            )
        )
        FROM ghg_accounting_scope3.category_results cr
        WHERE cr.inventory_id = si.id
    )                            AS category_details,
    -- Aggregated totals
    COALESCE(cat_agg.total_scope3, 0) AS total_scope3_tco2e,
    cat_agg.category_count,
    cat_agg.max_category,
    cat_agg.max_category_tco2e,
    -- Reconciliation
    rs.total_before_adjustment,
    rs.total_adjustment,
    rs.total_after_adjustment,
    rs.rules_triggered           AS overlap_rules_triggered,
    -- Data quality
    qa.overall_dqr,
    qa.overall_rating            AS dqr_rating,
    qa.pct_primary_data,
    -- Compliance
    ca.overall_compliance_score,
    ca.frameworks_compliant,
    ca.total_gaps                AS compliance_gaps,
    -- Provenance
    si.provenance_hash,
    si.updated_at
FROM ghg_accounting_scope3.scope3_inventories si
-- Category aggregates
LEFT JOIN LATERAL (
    SELECT
        SUM(cr2.total_tco2e) AS total_scope3,
        COUNT(*) AS category_count,
        (SELECT cr3.category::TEXT FROM ghg_accounting_scope3.category_results cr3
         WHERE cr3.inventory_id = si.id ORDER BY cr3.total_tco2e DESC LIMIT 1) AS max_category,
        MAX(cr2.total_tco2e) AS max_category_tco2e
    FROM ghg_accounting_scope3.category_results cr2
    WHERE cr2.inventory_id = si.id
) cat_agg ON true
-- Reconciliation
LEFT JOIN ghg_accounting_scope3.reconciliation_summary rs
    ON rs.inventory_id = si.id
-- Latest quality assessment
LEFT JOIN LATERAL (
    SELECT qa2.*
    FROM ghg_accounting_scope3.quality_assessments qa2
    WHERE qa2.inventory_id = si.id
    ORDER BY qa2.assessment_date DESC
    LIMIT 1
) qa ON true
-- Latest compliance assessment
LEFT JOIN LATERAL (
    SELECT ca2.*
    FROM ghg_accounting_scope3.compliance_assessments ca2
    WHERE ca2.inventory_id = si.id
    ORDER BY ca2.assessment_date DESC
    LIMIT 1
) ca ON true;

-- =============================================================================
-- View 2: ghg_accounting_scope3.v_category_comparison
-- =============================================================================
-- Year-over-year category comparison view. For each category, shows the
-- current year total alongside the prior year total and calculates the
-- absolute and percentage change. Useful for trend dashboards.

CREATE OR REPLACE VIEW ghg_accounting_scope3.v_category_comparison AS
SELECT
    curr.tenant_id,
    curr_inv.org_id,
    curr_inv.reporting_year                          AS current_year,
    curr_inv.reporting_year - 1                      AS prior_year,
    curr.category,
    -- Current year
    curr.total_tco2e                                 AS current_tco2e,
    curr.methodology_tier::TEXT                      AS current_tier,
    curr.data_quality_rating                         AS current_quality,
    curr.pct_of_total_scope3                         AS current_pct,
    -- Prior year
    prior.total_tco2e                                AS prior_tco2e,
    prior.methodology_tier::TEXT                     AS prior_tier,
    prior.data_quality_rating                        AS prior_quality,
    -- Change
    COALESCE(curr.total_tco2e, 0) - COALESCE(prior.total_tco2e, 0) AS change_tco2e,
    CASE
        WHEN COALESCE(prior.total_tco2e, 0) > 0
        THEN ROUND(((curr.total_tco2e - prior.total_tco2e) / prior.total_tco2e * 100)::NUMERIC, 2)
        ELSE NULL
    END                                              AS change_pct,
    -- Direction
    CASE
        WHEN prior.total_tco2e IS NULL THEN 'NEW'
        WHEN curr.total_tco2e > prior.total_tco2e THEN 'INCREASED'
        WHEN curr.total_tco2e < prior.total_tco2e THEN 'DECREASED'
        ELSE 'STABLE'
    END                                              AS change_direction
FROM ghg_accounting_scope3.category_results curr
INNER JOIN ghg_accounting_scope3.scope3_inventories curr_inv
    ON curr.inventory_id = curr_inv.id
LEFT JOIN ghg_accounting_scope3.scope3_inventories prior_inv
    ON prior_inv.org_id = curr_inv.org_id
    AND prior_inv.reporting_year = curr_inv.reporting_year - 1
    AND prior_inv.tenant_id = curr_inv.tenant_id
LEFT JOIN ghg_accounting_scope3.category_results prior
    ON prior.inventory_id = prior_inv.id
    AND prior.category = curr.category;

-- =============================================================================
-- View 3: ghg_accounting_scope3.v_supplier_dashboard
-- =============================================================================
-- Supplier engagement dashboard view combining supplier records with their
-- latest engagement plan status and data request status for quick overview.

CREATE OR REPLACE VIEW ghg_accounting_scope3.v_supplier_dashboard AS
SELECT
    s.id                         AS supplier_id,
    s.tenant_id,
    s.name                       AS supplier_name,
    s.industry_naics,
    s.country,
    s.engagement_tier,
    s.procurement_spend,
    s.emission_contribution_tco2e,
    s.emission_contribution_pct,
    s.has_cdp_score,
    s.cdp_score,
    s.has_sbti_target,
    s.status                     AS supplier_status,
    -- Latest engagement plan
    ep.current_data_quality_level,
    ep.target_data_quality_level,
    ep.status                    AS plan_status,
    ep.progress_pct              AS plan_progress,
    ep.plan_end_date,
    -- Latest data request
    dr.request_date              AS latest_request_date,
    dr.due_date                  AS latest_request_due,
    dr.status                    AS request_status,
    dr.reminder_count,
    dr.response_received,
    -- Latest response
    sres.response_date           AS latest_response_date,
    sres.data_quality_level      AS response_quality,
    sres.emissions_reported_tco2e AS reported_emissions,
    sres.validated               AS response_validated,
    sres.verification_status     AS response_verification
FROM ghg_accounting_scope3.suppliers s
LEFT JOIN LATERAL (
    SELECT ep2.*
    FROM ghg_accounting_scope3.engagement_plans ep2
    WHERE ep2.supplier_id = s.id
    ORDER BY ep2.plan_start_date DESC
    LIMIT 1
) ep ON true
LEFT JOIN LATERAL (
    SELECT dr2.*
    FROM ghg_accounting_scope3.data_requests dr2
    WHERE dr2.supplier_id = s.id
    ORDER BY dr2.request_date DESC
    LIMIT 1
) dr ON true
LEFT JOIN LATERAL (
    SELECT sres2.*
    FROM ghg_accounting_scope3.supplier_responses sres2
    WHERE sres2.supplier_id = s.id
    ORDER BY sres2.response_date DESC
    LIMIT 1
) sres ON true
WHERE s.status = 'ACTIVE';

-- =============================================================================
-- Materialized View: ghg_accounting_scope3.mv_scope3_benchmarks
-- =============================================================================
-- Pre-computed benchmark comparison joining organization profiles with
-- sector benchmarks for each category. Refreshed periodically.

CREATE MATERIALIZED VIEW ghg_accounting_scope3.mv_scope3_benchmarks AS
SELECT
    si.tenant_id,
    si.org_id,
    op.sector_naics,
    sb.sector_name,
    si.reporting_year,
    cr.category,
    -- Organization values
    cr.total_tco2e,
    cr.pct_of_total_scope3,
    cr.methodology_tier::TEXT    AS methodology_tier,
    cr.data_quality_rating,
    -- Benchmark values
    sb.avg_pct_of_scope3         AS benchmark_avg_pct,
    sb.median_pct_of_scope3      AS benchmark_median_pct,
    sb.p25_pct                   AS benchmark_p25,
    sb.p75_pct                   AS benchmark_p75,
    -- Deviation
    CASE
        WHEN sb.avg_pct_of_scope3 > 0
        THEN ROUND((cr.pct_of_total_scope3 - sb.avg_pct_of_scope3)::NUMERIC, 2)
        ELSE NULL
    END                          AS deviation_from_avg,
    -- Relative position
    CASE
        WHEN cr.pct_of_total_scope3 <= COALESCE(sb.p25_pct, 0) THEN 'BELOW_P25'
        WHEN cr.pct_of_total_scope3 <= COALESCE(sb.median_pct_of_scope3, 0) THEN 'P25_TO_MEDIAN'
        WHEN cr.pct_of_total_scope3 <= COALESCE(sb.p75_pct, 100) THEN 'MEDIAN_TO_P75'
        ELSE 'ABOVE_P75'
    END                          AS benchmark_position,
    -- Metadata
    sb.source                    AS benchmark_source,
    sb.year                      AS benchmark_year,
    sb.sample_count,
    NOW()                        AS materialized_at
FROM ghg_accounting_scope3.scope3_inventories si
INNER JOIN ghg_accounting_scope3.category_results cr
    ON cr.inventory_id = si.id
LEFT JOIN ghg_accounting_scope3.organization_profiles op
    ON op.org_id = si.org_id
    AND op.tenant_id = si.tenant_id
    AND op.is_current = true
LEFT JOIN ghg_accounting_scope3.sector_benchmarks sb
    ON sb.sector_naics = op.sector_naics
    AND sb.category = cr.category
    AND sb.is_active = true
ORDER BY si.tenant_id, si.org_id, si.reporting_year DESC, cr.total_tco2e DESC;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p042_mvb_org_year_cat
    ON ghg_accounting_scope3.mv_scope3_benchmarks(tenant_id, org_id, reporting_year, category);
CREATE INDEX idx_p042_mvb_tenant
    ON ghg_accounting_scope3.mv_scope3_benchmarks(tenant_id);
CREATE INDEX idx_p042_mvb_naics
    ON ghg_accounting_scope3.mv_scope3_benchmarks(sector_naics);
CREATE INDEX idx_p042_mvb_year
    ON ghg_accounting_scope3.mv_scope3_benchmarks(reporting_year DESC);
CREATE INDEX idx_p042_mvb_category
    ON ghg_accounting_scope3.mv_scope3_benchmarks(category);
CREATE INDEX idx_p042_mvb_tco2e
    ON ghg_accounting_scope3.mv_scope3_benchmarks(total_tco2e DESC);
CREATE INDEX idx_p042_mvb_position
    ON ghg_accounting_scope3.mv_scope3_benchmarks(benchmark_position);

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Inventories: tenant + org + year for multi-year comparison
CREATE INDEX IF NOT EXISTS idx_p042_si_org_year_status
    ON ghg_accounting_scope3.scope3_inventories(org_id, reporting_year DESC, status);

-- Category results: inventory + total for ranked categories
CREATE INDEX IF NOT EXISTS idx_p042_catr_inv_ranked
    ON ghg_accounting_scope3.category_results(inventory_id, total_tco2e DESC)
    WHERE status IN ('CALCULATED', 'REVIEWED', 'VERIFIED', 'FINALIZED');

-- Spend transactions: inventory + amount for top spend
CREATE INDEX IF NOT EXISTS idx_p042_st_inv_top_spend
    ON ghg_accounting_scope3.spend_transactions(inventory_id, amount DESC)
    WHERE is_excluded = false;

-- Classification: category + tco2e for top emitters by category
CREATE INDEX IF NOT EXISTS idx_p042_cr_cat_tco2e
    ON ghg_accounting_scope3.classification_results(scope3_category, calculated_tco2e DESC);

-- Suppliers: tenant + emission contribution for top emitters
CREATE INDEX IF NOT EXISTS idx_p042_sup_tenant_emission
    ON ghg_accounting_scope3.suppliers(tenant_id, emission_contribution_tco2e DESC)
    WHERE status = 'ACTIVE';

-- Data requests: tenant + status + due for follow-up queue
CREATE INDEX IF NOT EXISTS idx_p042_dr_tenant_follow_up
    ON ghg_accounting_scope3.data_requests(tenant_id, due_date)
    WHERE status IN ('SENT', 'ACKNOWLEDGED', 'OVERDUE') AND response_received = false;

-- Quality scores: assessment + category + worst DQR for improvement focus
CREATE INDEX IF NOT EXISTS idx_p042_cqs_worst_quality
    ON ghg_accounting_scope3.category_quality_scores(assessment_id, weighted_dqr DESC);

-- Compliance gaps: severity + status for actionable items
CREATE INDEX IF NOT EXISTS idx_p042_cg_actionable
    ON ghg_accounting_scope3.compliance_gaps(gap_severity, priority, target_date)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- Overlap detections: inventory + confirmed for adjustment
CREATE INDEX IF NOT EXISTS idx_p042_od_inv_confirmed
    ON ghg_accounting_scope3.overlap_detections(inventory_id, overlap_amount DESC)
    WHERE status = 'CONFIRMED';

-- =============================================================================
-- Seed Data: Framework Requirement Checklists
-- =============================================================================
-- Scope 3-specific requirements across 6 frameworks for compliance engine.
-- Loaded into a temp table, then applied via application layer.

CREATE TEMPORARY TABLE _tmp_scope3_reqs (
    framework VARCHAR(50),
    requirement_id VARCHAR(50),
    section VARCHAR(100),
    description TEXT,
    type VARCHAR(20)
);

INSERT INTO _tmp_scope3_reqs (framework, requirement_id, section, description, type) VALUES
    -- GHG Protocol Scope 3 Standard
    ('GHG_PROTOCOL_SCOPE3', 'S3-001', 'Chapter 5', 'Identify and report on all 15 Scope 3 categories', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-002', 'Chapter 5', 'Report total Scope 3 emissions in metric tons CO2e', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-003', 'Chapter 5', 'Report Scope 3 emissions by category', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-004', 'Chapter 7', 'Conduct category relevance screening for all 15 categories', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-005', 'Chapter 7', 'Document exclusion justification for any excluded category', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-006', 'Chapter 7', 'Apply screening criteria: size, influence, risk, stakeholders, outsourcing, sector guidance', 'RECOMMENDED'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-007', 'Chapter 8', 'Use 100-year GWP values from IPCC for CO2e conversion', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-008', 'Chapter 8', 'Document calculation methodology and data sources for each category', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-009', 'Chapter 8', 'Avoid double-counting between categories', 'MANDATORY'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-010', 'Chapter 9', 'Report biogenic CO2 emissions separately', 'RECOMMENDED'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-011', 'Chapter 9', 'Disclose data quality for each reported category', 'RECOMMENDED'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-012', 'Chapter 10', 'Set a Scope 3 base year and describe recalculation policy', 'RECOMMENDED'),
    ('GHG_PROTOCOL_SCOPE3', 'S3-013', 'Chapter 10', 'Set Scope 3 reduction targets', 'RECOMMENDED'),
    -- CSRD ESRS E1
    ('CSRD_ESRS_E1', 'E1-S3-001', 'E1-6.44', 'Disclose gross Scope 3 GHG emissions', 'MANDATORY'),
    ('CSRD_ESRS_E1', 'E1-S3-002', 'E1-6.44', 'Disclose Scope 3 emissions by significant category', 'MANDATORY'),
    ('CSRD_ESRS_E1', 'E1-S3-003', 'E1-6.46', 'Disclose Scope 3 estimation methodologies and assumptions', 'MANDATORY'),
    ('CSRD_ESRS_E1', 'E1-S3-004', 'E1-6.47', 'Explain exclusion of any Scope 3 category', 'MANDATORY'),
    ('CSRD_ESRS_E1', 'E1-S3-005', 'E1-4', 'Include Scope 3 in GHG emission reduction targets', 'MANDATORY'),
    ('CSRD_ESRS_E1', 'E1-S3-006', 'E1-6.48', 'Disclose percentage of Scope 3 from primary data', 'MANDATORY'),
    ('CSRD_ESRS_E1', 'E1-S3-007', 'E1-3', 'Disclose Scope 3 GHG reduction actions and expected results', 'MANDATORY'),
    ('CSRD_ESRS_E1', 'E1-S3-008', 'E1-6.50', 'Report total Scope 3 alongside Scope 1 and Scope 2', 'MANDATORY'),
    -- CDP Climate
    ('CDP_CLIMATE', 'CDP-S3-001', 'C6.5', 'Report gross Scope 3 emissions by category', 'MANDATORY'),
    ('CDP_CLIMATE', 'CDP-S3-002', 'C6.5a', 'Describe methodology, emission factors, and data quality per category', 'MANDATORY'),
    ('CDP_CLIMATE', 'CDP-S3-003', 'C6.5b', 'Justify exclusion of any Scope 3 category', 'MANDATORY'),
    ('CDP_CLIMATE', 'CDP-S3-004', 'C6.5c', 'Report total Scope 3 emissions in metric tons CO2e', 'MANDATORY'),
    ('CDP_CLIMATE', 'CDP-S3-005', 'C12.1', 'Describe engagement with supply chain on GHG emissions', 'MANDATORY'),
    ('CDP_CLIMATE', 'CDP-S3-006', 'C4.1', 'Include Scope 3 in company-wide emissions reduction targets', 'MANDATORY'),
    ('CDP_CLIMATE', 'CDP-S3-007', 'C2.4', 'Disclose Scope 3 risks in value chain', 'RECOMMENDED'),
    -- SBTi
    ('SBTI', 'SBTI-S3-001', 'Criteria 4', 'Screen Scope 3 categories to determine if >40% of total emissions', 'MANDATORY'),
    ('SBTI', 'SBTI-S3-002', 'Criteria 4', 'Set Scope 3 target if Scope 3 is 40%+ of total Scope 1+2+3', 'MANDATORY'),
    ('SBTI', 'SBTI-S3-003', 'Criteria 4', 'Cover at least 67% of Scope 3 emissions in target boundary', 'MANDATORY'),
    ('SBTI', 'SBTI-S3-004', 'Criteria 4', 'Set ambitious Scope 3 reduction targets (25% in 5-10 years)', 'MANDATORY'),
    ('SBTI', 'SBTI-S3-005', 'Criteria 4', 'Supplier engagement: 67% of suppliers by spend set SBTs', 'OPTIONAL'),
    ('SBTI', 'SBTI-S3-006', 'Criteria', 'Report progress against Scope 3 targets annually', 'MANDATORY'),
    -- TCFD
    ('TCFD', 'TCFD-S3-001', 'Metrics-b', 'Disclose Scope 3 GHG emissions if appropriate', 'RECOMMENDED'),
    ('TCFD', 'TCFD-S3-002', 'Metrics-b', 'Disclose associated risks with Scope 3 emissions', 'RECOMMENDED'),
    ('TCFD', 'TCFD-S3-003', 'Targets-a', 'Include Scope 3 in climate-related targets if material', 'RECOMMENDED'),
    -- SEC Climate Rule
    ('SEC_CLIMATE', 'SEC-S3-001', 'Reg S-K 1504', 'Disclose Scope 3 emissions if material or if target includes Scope 3', 'CONDITIONAL'),
    ('SEC_CLIMATE', 'SEC-S3-002', 'Reg S-K 1504', 'Describe methodology for Scope 3 estimation', 'CONDITIONAL'),
    ('SEC_CLIMATE', 'SEC-S3-003', 'Reg S-K 1504', 'Disclose significant Scope 3 categories', 'CONDITIONAL');

-- Note: temp table is used as reference for the compliance engine data loader
DROP TABLE IF EXISTS _tmp_scope3_reqs;

-- =============================================================================
-- Seed Data: RBAC Policies for 6 Roles
-- =============================================================================
-- Application-level role configuration for Scope 3 data access.
-- Loaded as reference data for the access control system.

CREATE TEMPORARY TABLE _tmp_scope3_rbac (
    role_name VARCHAR(50),
    schema_name VARCHAR(100),
    table_pattern VARCHAR(100),
    permissions VARCHAR(50),
    description TEXT
);

INSERT INTO _tmp_scope3_rbac (role_name, schema_name, table_pattern, permissions, description) VALUES
    -- Admin: full access
    ('scope3_admin', 'ghg_accounting_scope3', '*', 'SELECT,INSERT,UPDATE,DELETE', 'Full access to all Scope 3 tables'),
    -- Analyst: read-write on operational tables, read-only on reference data
    ('scope3_analyst', 'ghg_accounting_scope3', 'scope3_inventories', 'SELECT,INSERT,UPDATE', 'Create and modify inventories'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'category_configurations', 'SELECT,INSERT,UPDATE', 'Configure category settings'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'spend_transactions', 'SELECT,INSERT,UPDATE', 'Manage spend data'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'classification_results', 'SELECT,INSERT,UPDATE', 'Classify transactions'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'category_results', 'SELECT,INSERT,UPDATE', 'Calculate category emissions'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'screening_results', 'SELECT,INSERT,UPDATE', 'Run category screening'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'hotspot_analyses', 'SELECT,INSERT,UPDATE', 'Perform hotspot analysis'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'quality_assessments', 'SELECT,INSERT,UPDATE', 'Assess data quality'),
    ('scope3_analyst', 'ghg_accounting_scope3', 'uncertainty_analyses', 'SELECT,INSERT,UPDATE', 'Run uncertainty analysis'),
    -- Reviewer: read-all, update status fields only
    ('scope3_reviewer', 'ghg_accounting_scope3', '*', 'SELECT', 'Read access to all Scope 3 data'),
    ('scope3_reviewer', 'ghg_accounting_scope3', 'category_results', 'UPDATE', 'Approve/reject category results'),
    ('scope3_reviewer', 'ghg_accounting_scope3', 'compliance_assessments', 'UPDATE', 'Review compliance assessments'),
    -- Supplier contact: limited access to own data
    ('scope3_supplier', 'ghg_accounting_scope3', 'data_requests', 'SELECT', 'View data requests assigned to supplier'),
    ('scope3_supplier', 'ghg_accounting_scope3', 'supplier_responses', 'SELECT,INSERT,UPDATE', 'Submit and update responses'),
    -- Auditor: read-only access to everything
    ('scope3_auditor', 'ghg_accounting_scope3', '*', 'SELECT', 'Read-only audit access to all Scope 3 data'),
    ('scope3_auditor', 'ghg_accounting_scope3', 'scope3_audit_trail', 'SELECT', 'Full audit trail access'),
    -- Viewer: read-only views and reports
    ('scope3_viewer', 'ghg_accounting_scope3', 'v_*', 'SELECT', 'Access to views only'),
    ('scope3_viewer', 'ghg_accounting_scope3', 'mv_*', 'SELECT', 'Access to materialized views'),
    ('scope3_viewer', 'ghg_accounting_scope3', 'report_metadata', 'SELECT', 'View report metadata');

DROP TABLE IF EXISTS _tmp_scope3_rbac;

-- =============================================================================
-- Grants for Views and Materialized Views
-- =============================================================================
GRANT SELECT ON ghg_accounting_scope3.v_inventory_summary TO PUBLIC;
GRANT SELECT ON ghg_accounting_scope3.v_category_comparison TO PUBLIC;
GRANT SELECT ON ghg_accounting_scope3.v_supplier_dashboard TO PUBLIC;
GRANT SELECT ON ghg_accounting_scope3.mv_scope3_benchmarks TO PUBLIC;

-- =============================================================================
-- Role Grants for Application Roles
-- =============================================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_app') THEN
        GRANT USAGE ON SCHEMA ghg_accounting_scope3 TO greenlang_app;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ghg_accounting_scope3 TO greenlang_app;
        GRANT SELECT ON ALL SEQUENCES IN SCHEMA ghg_accounting_scope3 TO greenlang_app;
        RAISE NOTICE 'Granted ghg_accounting_scope3 permissions to greenlang_app role';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA ghg_accounting_scope3 TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA ghg_accounting_scope3 TO greenlang_readonly;
        RAISE NOTICE 'Granted ghg_accounting_scope3 read permissions to greenlang_readonly role';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_service') THEN
        GRANT USAGE ON SCHEMA ghg_accounting_scope3 TO greenlang_service;
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ghg_accounting_scope3 TO greenlang_service;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ghg_accounting_scope3 TO greenlang_service;
        RAISE NOTICE 'Granted ghg_accounting_scope3 full permissions to greenlang_service role';
    END IF;
END;
$$;

-- =============================================================================
-- Comments on Views
-- =============================================================================
COMMENT ON VIEW ghg_accounting_scope3.v_inventory_summary IS
    'Dashboard view: per-inventory summary with all category totals (JSONB), reconciliation, data quality, and compliance status in a single query.';
COMMENT ON VIEW ghg_accounting_scope3.v_category_comparison IS
    'Year-over-year category comparison showing current vs prior year emissions with absolute and percentage change.';
COMMENT ON VIEW ghg_accounting_scope3.v_supplier_dashboard IS
    'Supplier engagement dashboard combining supplier master, latest engagement plan, data request, and response status.';
COMMENT ON MATERIALIZED VIEW ghg_accounting_scope3.mv_scope3_benchmarks IS
    'Pre-computed benchmark comparison of organization category results against sector averages. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY ghg_accounting_scope3.mv_scope3_benchmarks;';

-- =============================================================================
-- Migration Complete
-- =============================================================================
-- PACK-042 Scope 3 Starter Pack database schema is now fully deployed.
--
-- Schema: ghg_accounting_scope3
-- Tables: 38
-- Views: 3
-- Materialized Views: 1
-- Enums: 3
-- Seed Data: 55 NAICS mappings, 35 EEIO factors, 12 overlap rules,
--            60 sector benchmarks, 40 framework requirements, 20 RBAC rules
--
-- Table Summary:
--   V336 (Core):           scope3_inventories, category_configurations,
--                           screening_results, organization_profiles
--   V337 (Spend):          spend_transactions, classification_results,
--                           eeio_sector_factors, naics_category_mapping
--   V338 (Results):        category_results, category_gas_breakdown,
--                           emission_factor_usage, category_sub_results
--   V339 (Double Count):   overlap_rules, overlap_detections,
--                           overlap_resolutions, reconciliation_summary
--   V340 (Hotspot):        hotspot_analyses, pareto_results,
--                           materiality_scores, sector_benchmarks,
--                           reduction_opportunities
--   V341 (Supplier):       suppliers, engagement_plans, data_requests,
--                           supplier_responses, engagement_metrics
--   V342 (Quality):        quality_assessments, category_quality_scores,
--                           quality_improvements, quality_trends
--   V343 (Uncertainty):    uncertainty_analyses, category_uncertainties,
--                           sensitivity_results, correlation_matrix,
--                           tier_upgrade_impacts
--   V344 (Compliance):     compliance_assessments, framework_results,
--                           requirement_checks, compliance_gaps,
--                           report_metadata, verification_packages,
--                           scope3_audit_trail
--   V345 (Views):          v_inventory_summary, v_category_comparison,
--                           v_supplier_dashboard, mv_scope3_benchmarks
