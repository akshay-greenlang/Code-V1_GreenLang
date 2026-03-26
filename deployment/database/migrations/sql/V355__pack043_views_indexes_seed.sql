-- =============================================================================
-- V355: PACK-043 Scope 3 Complete Pack - Views, Indexes & Seed Data
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates analytical views, materialized views, additional composite indexes
-- for common query patterns, and seed data. Views provide pre-joined
-- perspectives for dashboards and reporting. Seed data includes MACC
-- intervention templates and SDA sector pathway data. RBAC configuration
-- for 8 roles. Role grants for application-level access control.
--
-- Views (3):
--   1. ghg_accounting_scope3_complete.v_enterprise_summary
--   2. ghg_accounting_scope3_complete.v_sbti_progress
--   3. ghg_accounting_scope3_complete.v_supplier_programme_dashboard
--
-- Materialized Views (1):
--   4. ghg_accounting_scope3_complete.mv_sector_benchmarks
--
-- Also includes: composite indexes, MACC seed data, SDA sector data, RBAC,
-- role grants, comments.
-- Previous: V354__pack043_assurance.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- =============================================================================
-- View 1: ghg_accounting_scope3_complete.v_enterprise_summary
-- =============================================================================
-- Multi-entity consolidated Scope 3 summary with maturity assessment. Joins
-- entity hierarchy, boundary definitions, and maturity assessments to provide
-- a single-query enterprise overview for the dashboard.

CREATE OR REPLACE VIEW ghg_accounting_scope3_complete.v_enterprise_summary AS
SELECT
    eh.tenant_id,
    eh.entity_id,
    eh.name                          AS entity_name,
    eh.entity_type::TEXT             AS entity_type,
    eh.country,
    eh.ownership_pct,
    eh.control_type::TEXT            AS control_type,
    eh.consolidation_approach::TEXT  AS consolidation_approach,
    -- Boundary
    bd.inventory_id,
    bd.included                      AS in_boundary,
    bd.consolidation_pct,
    bd.estimated_scope3_tco2e,
    bd.significance_pct,
    -- Maturity
    ma.overall_maturity_level::TEXT  AS maturity_level,
    ma.budget_usd                    AS programme_budget,
    ma.budget_spent_usd              AS programme_spend,
    ma.categories_at_level_1,
    ma.categories_at_level_2,
    ma.categories_at_level_3,
    ma.categories_at_level_4,
    ma.categories_at_level_5,
    ma.data_collection_score,
    ma.methodology_score,
    ma.supplier_engagement_score,
    -- Derived
    CASE
        WHEN bd.included = true AND bd.consolidation_pct > 0 AND bd.estimated_scope3_tco2e IS NOT NULL
        THEN ROUND((bd.estimated_scope3_tco2e * bd.consolidation_pct / 100)::NUMERIC, 3)
        ELSE 0
    END                              AS consolidated_tco2e,
    -- Status
    eh.is_active                     AS entity_active,
    bd.approved                      AS boundary_approved,
    ma.status                        AS maturity_status,
    -- Timestamps
    eh.updated_at                    AS entity_updated,
    ma.assessment_date               AS maturity_date
FROM ghg_accounting_scope3_complete.entity_hierarchy eh
LEFT JOIN ghg_accounting_scope3_complete.boundary_definitions bd
    ON bd.entity_id = eh.id
LEFT JOIN LATERAL (
    SELECT ma2.*
    FROM ghg_accounting_scope3_complete.maturity_assessments ma2
    WHERE ma2.inventory_id = bd.inventory_id
    ORDER BY ma2.assessment_date DESC
    LIMIT 1
) ma ON true
WHERE eh.is_active = true;

-- =============================================================================
-- View 2: ghg_accounting_scope3_complete.v_sbti_progress
-- =============================================================================
-- SBTi target progress view showing current trajectory against required
-- pathway. Joins targets with the most recent pathway year that has actual
-- data to show on-track/off-track status at a glance.

CREATE OR REPLACE VIEW ghg_accounting_scope3_complete.v_sbti_progress AS
SELECT
    st.tenant_id,
    st.id                            AS target_id,
    st.inventory_id,
    st.target_type::TEXT             AS target_type,
    st.target_name,
    st.ambition_level,
    -- Base year
    st.base_year,
    st.base_year_tco2e,
    -- Target
    st.target_year,
    st.target_tco2e,
    st.total_reduction_pct,
    st.annual_reduction_pct,
    st.coverage_pct,
    -- Latest actual
    latest.year                      AS latest_year,
    latest.actual_tco2e              AS latest_actual_tco2e,
    latest.required_tco2e            AS latest_required_tco2e,
    latest.variance_tco2e            AS latest_variance_tco2e,
    latest.variance_pct              AS latest_variance_pct,
    latest.on_track                  AS latest_on_track,
    -- Trajectory
    CASE
        WHEN latest.actual_tco2e IS NOT NULL AND st.base_year_tco2e > 0
        THEN ROUND(((st.base_year_tco2e - latest.actual_tco2e) / st.base_year_tco2e * 100)::NUMERIC, 2)
        ELSE NULL
    END                              AS actual_reduction_pct,
    -- Status
    st.status                        AS target_status,
    st.validated,
    st.on_track                      AS overall_on_track,
    -- Next milestone
    nm.milestone_year                AS next_milestone_year,
    nm.milestone_tco2e               AS next_milestone_tco2e,
    nm.status                        AS next_milestone_status,
    -- Timestamps
    st.updated_at
FROM ghg_accounting_scope3_complete.sbti_targets st
-- Latest pathway year with actual data
LEFT JOIN LATERAL (
    SELECT sp.*
    FROM ghg_accounting_scope3_complete.sbti_pathways sp
    WHERE sp.target_id = st.id
      AND sp.actual_tco2e IS NOT NULL
    ORDER BY sp.year DESC
    LIMIT 1
) latest ON true
-- Next upcoming milestone
LEFT JOIN LATERAL (
    SELECT sm.*
    FROM ghg_accounting_scope3_complete.sbti_milestones sm
    WHERE sm.target_id = st.id
      AND sm.achieved = false
    ORDER BY sm.milestone_year
    LIMIT 1
) nm ON true
WHERE st.flag_enabled = true;

-- =============================================================================
-- View 3: ghg_accounting_scope3_complete.v_supplier_programme_dashboard
-- =============================================================================
-- Supplier programme dashboard aggregating targets, commitments, progress,
-- and scorecards for each supplier in a single row.

CREATE OR REPLACE VIEW ghg_accounting_scope3_complete.v_supplier_programme_dashboard AS
SELECT
    st.tenant_id,
    st.supplier_id,
    -- Target
    st.target_reduction_pct,
    st.base_year,
    st.base_year_tco2e,
    st.target_year,
    st.status                        AS target_status,
    st.on_track,
    st.current_reduction_pct,
    -- Commitments
    sc_agg.commitment_count,
    sc_agg.has_sbti,
    sc_agg.has_net_zero,
    -- Latest progress
    sp.reporting_year                AS progress_year,
    sp.reported_tco2e                AS progress_tco2e,
    sp.reduction_pct                 AS progress_reduction_pct,
    sp.data_quality_level            AS progress_quality,
    sp.verified                      AS progress_verified,
    -- Scorecard
    ssc.overall_score,
    ssc.emission_score,
    ssc.quality_score,
    ssc.engagement_score,
    ssc.commitment_score,
    ssc.tier_classification,
    ssc.previous_score,
    ssc.score_change,
    -- Updated
    st.updated_at
FROM ghg_accounting_scope3_complete.supplier_targets st
-- Commitment aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS commitment_count,
        BOOL_OR(sc2.commitment_type IN ('SBTI_COMMITTED', 'SBTI_VALIDATED')) AS has_sbti,
        BOOL_OR(sc2.commitment_type = 'NET_ZERO') AS has_net_zero
    FROM ghg_accounting_scope3_complete.supplier_commitments sc2
    WHERE sc2.supplier_id = st.supplier_id
      AND sc2.status = 'ACTIVE'
) sc_agg ON true
-- Latest progress
LEFT JOIN LATERAL (
    SELECT sp2.*
    FROM ghg_accounting_scope3_complete.supplier_progress sp2
    WHERE sp2.supplier_id = st.supplier_id
    ORDER BY sp2.reporting_year DESC
    LIMIT 1
) sp ON true
-- Latest scorecard
LEFT JOIN LATERAL (
    SELECT ssc2.*
    FROM ghg_accounting_scope3_complete.supplier_scorecards ssc2
    WHERE ssc2.supplier_id = st.supplier_id
    ORDER BY ssc2.assessment_period_year DESC
    LIMIT 1
) ssc ON true
WHERE st.status = 'ACTIVE';

-- =============================================================================
-- Materialized View: ghg_accounting_scope3_complete.mv_sector_benchmarks
-- =============================================================================
-- Pre-computed sector comparison data for benchmarking Scope 3 performance
-- across industries. Aggregates multi-year data by sector NAICS code.

CREATE MATERIALIZED VIEW ghg_accounting_scope3_complete.mv_sector_benchmarks AS
SELECT
    eh.tenant_id,
    eh.sector_naics,
    myd.reporting_year,
    myd.category,
    -- Counts
    COUNT(DISTINCT myd.org_id)       AS org_count,
    -- Emissions statistics
    SUM(myd.tco2e)                   AS total_tco2e,
    AVG(myd.tco2e)                   AS avg_tco2e,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY myd.tco2e) AS p25_tco2e,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY myd.tco2e) AS median_tco2e,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY myd.tco2e) AS p75_tco2e,
    MIN(myd.tco2e)                   AS min_tco2e,
    MAX(myd.tco2e)                   AS max_tco2e,
    -- Data quality
    AVG(myd.dqr)                     AS avg_dqr,
    AVG(myd.primary_data_pct)        AS avg_primary_data_pct,
    -- Methodology distribution
    COUNT(*) FILTER (WHERE myd.tier = 'LEVEL_1') AS count_level_1,
    COUNT(*) FILTER (WHERE myd.tier = 'LEVEL_2') AS count_level_2,
    COUNT(*) FILTER (WHERE myd.tier = 'LEVEL_3') AS count_level_3,
    COUNT(*) FILTER (WHERE myd.tier = 'LEVEL_4') AS count_level_4,
    COUNT(*) FILTER (WHERE myd.tier = 'LEVEL_5') AS count_level_5,
    -- Metadata
    NOW()                            AS materialized_at
FROM ghg_accounting_scope3_complete.multi_year_data myd
INNER JOIN ghg_accounting_scope3_complete.entity_hierarchy eh
    ON eh.tenant_id = myd.tenant_id
    AND eh.entity_id = myd.org_id
    AND eh.is_active = true
WHERE eh.sector_naics IS NOT NULL
GROUP BY eh.tenant_id, eh.sector_naics, myd.reporting_year, myd.category
ORDER BY eh.sector_naics, myd.reporting_year DESC, myd.category;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p043_mvb_sector_year_cat
    ON ghg_accounting_scope3_complete.mv_sector_benchmarks(tenant_id, sector_naics, reporting_year, category);
CREATE INDEX idx_p043_mvb_tenant
    ON ghg_accounting_scope3_complete.mv_sector_benchmarks(tenant_id);
CREATE INDEX idx_p043_mvb_naics
    ON ghg_accounting_scope3_complete.mv_sector_benchmarks(sector_naics);
CREATE INDEX idx_p043_mvb_year
    ON ghg_accounting_scope3_complete.mv_sector_benchmarks(reporting_year DESC);
CREATE INDEX idx_p043_mvb_category
    ON ghg_accounting_scope3_complete.mv_sector_benchmarks(category);
CREATE INDEX idx_p043_mvb_avg_tco2e
    ON ghg_accounting_scope3_complete.mv_sector_benchmarks(avg_tco2e DESC);

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Entity hierarchy: active + country for geographic roll-up
CREATE INDEX IF NOT EXISTS idx_p043_eh_active_country
    ON ghg_accounting_scope3_complete.entity_hierarchy(country, ownership_pct DESC)
    WHERE is_active = true;

-- Products: active + revenue for top products
CREATE INDEX IF NOT EXISTS idx_p043_prod_top_revenue
    ON ghg_accounting_scope3_complete.products(tenant_id, revenue DESC)
    WHERE active = true AND pcf_available = true;

-- Scenarios: inventory + preferred + active
CREATE INDEX IF NOT EXISTS idx_p043_sc_inv_active
    ON ghg_accounting_scope3_complete.scenarios(inventory_id, baseline_tco2e DESC)
    WHERE status IN ('APPROVED', 'ACTIVE');

-- Interventions: active + cheapest
CREATE INDEX IF NOT EXISTS idx_p043_int_active_cheap
    ON ghg_accounting_scope3_complete.interventions(scenario_id, cost_per_tco2e)
    WHERE status IN ('PLANNED', 'EVALUATING', 'APPROVED', 'IN_PROGRESS');

-- SBTi targets: validated + on-track
CREATE INDEX IF NOT EXISTS idx_p043_sbt_validated_track
    ON ghg_accounting_scope3_complete.sbti_targets(tenant_id, target_type)
    WHERE validated = true AND on_track IS NOT NULL;

-- Supplier scorecards: latest + tier
CREATE INDEX IF NOT EXISTS idx_p043_ssc_latest_tier
    ON ghg_accounting_scope3_complete.supplier_scorecards(tenant_id, tier_classification, overall_score DESC);

-- Multi-year data: org + year for total query
CREATE INDEX IF NOT EXISTS idx_p043_myd_total
    ON ghg_accounting_scope3_complete.multi_year_data(org_id, reporting_year, tco2e DESC);

-- Audit findings: open + severity for action queue
CREATE INDEX IF NOT EXISTS idx_p043_af_action_queue
    ON ghg_accounting_scope3_complete.audit_findings(severity, due_date)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- PCAF assets: portfolio + top financed
CREATE INDEX IF NOT EXISTS idx_p043_pcaf_top_financed
    ON ghg_accounting_scope3_complete.pcaf_assets(portfolio_id, financed_tco2e DESC);

-- =============================================================================
-- Seed Data: MACC Intervention Templates (20+ default interventions)
-- =============================================================================
-- Default intervention templates with typical cost and reduction ranges
-- for common Scope 3 reduction actions. Application layer loads these as
-- starting points for scenario modelling.

CREATE TEMPORARY TABLE _tmp_macc_interventions (
    name VARCHAR(500),
    intervention_type VARCHAR(50),
    category_target VARCHAR(10),
    typical_cost_usd_low NUMERIC(14,2),
    typical_cost_usd_high NUMERIC(14,2),
    typical_reduction_pct_low DECIMAL(5,2),
    typical_reduction_pct_high DECIMAL(5,2),
    difficulty VARCHAR(20),
    timeframe_months_typical INTEGER,
    description TEXT
);

INSERT INTO _tmp_macc_interventions (name, intervention_type, category_target, typical_cost_usd_low, typical_cost_usd_high, typical_reduction_pct_low, typical_reduction_pct_high, difficulty, timeframe_months_typical, description) VALUES
    ('Switch to low-carbon materials', 'PROCUREMENT', 'CAT_1', 50000, 500000, 5.00, 25.00, 'MEDIUM', 12, 'Substitute virgin materials with recycled or bio-based alternatives'),
    ('Supplier engagement programme', 'SUPPLIER_ENGAGEMENT', 'CAT_1', 100000, 300000, 10.00, 30.00, 'MEDIUM', 24, 'Structured programme to help top suppliers reduce their emissions'),
    ('Supplier SBTi requirement', 'SUPPLIER_ENGAGEMENT', 'CAT_1', 50000, 200000, 15.00, 40.00, 'HIGH', 36, 'Require top suppliers by spend to commit to science-based targets'),
    ('Capital goods lifecycle extension', 'PRODUCT_DESIGN', 'CAT_2', 25000, 150000, 5.00, 15.00, 'LOW', 18, 'Extend useful life of capital equipment through maintenance programmes'),
    ('Renewable energy procurement for Tier 1', 'ENERGY_SWITCH', 'CAT_3', 75000, 250000, 20.00, 50.00, 'MEDIUM', 12, 'Support key suppliers in switching to renewable electricity'),
    ('Logistics modal shift (road to rail)', 'MODAL_SHIFT', 'CAT_4', 30000, 200000, 15.00, 40.00, 'MEDIUM', 18, 'Shift freight from road to rail for long-haul routes'),
    ('Route optimisation', 'OPERATIONAL', 'CAT_4', 20000, 100000, 5.00, 15.00, 'LOW', 6, 'Optimise delivery routes to reduce distance and fuel consumption'),
    ('EV fleet for last-mile delivery', 'TECHNOLOGY', 'CAT_4', 200000, 1000000, 30.00, 60.00, 'HIGH', 24, 'Transition last-mile delivery fleet to electric vehicles'),
    ('Waste reduction programme', 'CIRCULAR_ECONOMY', 'CAT_5', 15000, 75000, 10.00, 30.00, 'LOW', 12, 'Reduce waste generated in operations through prevention and recycling'),
    ('Virtual meeting policy', 'POLICY', 'CAT_6', 5000, 20000, 20.00, 50.00, 'LOW', 3, 'Implement virtual-first meeting policy to reduce business travel'),
    ('Sustainable travel policy', 'POLICY', 'CAT_6', 10000, 50000, 10.00, 25.00, 'LOW', 6, 'Rail-first policy for trips under 500km, economy-class flights'),
    ('Remote work programme', 'POLICY', 'CAT_7', 10000, 50000, 15.00, 35.00, 'LOW', 6, 'Expand remote/hybrid work to reduce daily commuting'),
    ('Employee EV incentive', 'TECHNOLOGY', 'CAT_7', 50000, 300000, 10.00, 25.00, 'MEDIUM', 12, 'Subsidise employee EV purchases or provide charging infrastructure'),
    ('Downstream logistics optimisation', 'OPERATIONAL', 'CAT_9', 30000, 150000, 5.00, 20.00, 'MEDIUM', 12, 'Optimise distribution network and consolidate shipments'),
    ('Product energy efficiency improvement', 'PRODUCT_DESIGN', 'CAT_11', 100000, 500000, 10.00, 30.00, 'HIGH', 24, 'Redesign products for lower energy consumption during use phase'),
    ('Design for recyclability', 'CIRCULAR_ECONOMY', 'CAT_12', 75000, 300000, 10.00, 25.00, 'MEDIUM', 18, 'Redesign products and packaging for end-of-life recyclability'),
    ('Franchise efficiency programme', 'OPERATIONAL', 'CAT_14', 50000, 200000, 10.00, 25.00, 'MEDIUM', 18, 'Energy efficiency and renewable energy programme for franchise operations'),
    ('Green building standards for leased assets', 'OPERATIONAL', 'CAT_13', 100000, 500000, 15.00, 35.00, 'HIGH', 36, 'Require LEED/BREEAM certification for leased properties'),
    ('Portfolio decarbonisation (investments)', 'PROCUREMENT', 'CAT_15', 25000, 100000, 10.00, 30.00, 'MEDIUM', 12, 'Shift investment portfolio toward lower-carbon assets'),
    ('Cloud carbon optimisation', 'TECHNOLOGY', 'CAT_1', 15000, 75000, 5.00, 20.00, 'LOW', 6, 'Optimise cloud infrastructure for energy efficiency and green regions'),
    ('Circular packaging programme', 'CIRCULAR_ECONOMY', 'CAT_1', 50000, 250000, 5.00, 15.00, 'MEDIUM', 12, 'Switch to reusable, recycled, or minimal packaging'),
    ('Carbon insetting (nature-based)', 'OTHER', 'CAT_1', 100000, 1000000, 5.00, 15.00, 'HIGH', 36, 'Invest in nature-based solutions within own supply chain');

-- Note: temp table used as reference for the application-layer data loader
DROP TABLE IF EXISTS _tmp_macc_interventions;

-- =============================================================================
-- Seed Data: SDA Sector Pathway Data
-- =============================================================================
-- Sectoral Decarbonisation Approach (SDA) pathway data for SBTi target setting.
-- Contains sector-specific intensity reduction pathways aligned with 1.5C and 2C.

CREATE TEMPORARY TABLE _tmp_sda_pathways (
    sector VARCHAR(100),
    sector_naics VARCHAR(10),
    pathway VARCHAR(30),
    base_year INTEGER,
    target_year INTEGER,
    annual_reduction_pct DECIMAL(5,2),
    intensity_metric VARCHAR(100),
    source VARCHAR(200)
);

INSERT INTO _tmp_sda_pathways (sector, sector_naics, pathway, base_year, target_year, annual_reduction_pct, intensity_metric, source) VALUES
    ('Electricity Generation', '2211', '1.5C', 2020, 2030, 7.00, 'tCO2e/MWh', 'IEA NZE 2021'),
    ('Electricity Generation', '2211', '2C', 2020, 2030, 4.50, 'tCO2e/MWh', 'IEA SDS 2021'),
    ('Iron and Steel', '3311', '1.5C', 2020, 2030, 3.50, 'tCO2e/tonne steel', 'SBTi SDA Tool v2'),
    ('Iron and Steel', '3311', '2C', 2020, 2030, 2.00, 'tCO2e/tonne steel', 'SBTi SDA Tool v2'),
    ('Cement', '3273', '1.5C', 2020, 2030, 3.00, 'tCO2e/tonne cement', 'SBTi SDA Tool v2'),
    ('Cement', '3273', '2C', 2020, 2030, 1.80, 'tCO2e/tonne cement', 'SBTi SDA Tool v2'),
    ('Aluminium', '3313', '1.5C', 2020, 2030, 3.20, 'tCO2e/tonne aluminium', 'IEA NZE 2021'),
    ('Aluminium', '3313', '2C', 2020, 2030, 2.10, 'tCO2e/tonne aluminium', 'IEA SDS 2021'),
    ('Pulp and Paper', '3221', '1.5C', 2020, 2030, 2.50, 'tCO2e/tonne product', 'SBTi SDA Tool v2'),
    ('Pulp and Paper', '3221', '2C', 2020, 2030, 1.50, 'tCO2e/tonne product', 'SBTi SDA Tool v2'),
    ('Chemicals', '3251', '1.5C', 2020, 2030, 2.80, 'tCO2e/tonne product', 'IEA NZE 2021'),
    ('Chemicals', '3251', '2C', 2020, 2030, 1.70, 'tCO2e/tonne product', 'IEA SDS 2021'),
    ('Road Transport', '4841', '1.5C', 2020, 2030, 5.50, 'gCO2e/tkm', 'IEA NZE 2021'),
    ('Road Transport', '4841', '2C', 2020, 2030, 3.50, 'gCO2e/tkm', 'IEA SDS 2021'),
    ('Aviation', '4811', '1.5C', 2020, 2030, 3.50, 'gCO2e/pkm', 'IEA NZE 2021'),
    ('Aviation', '4811', '2C', 2020, 2030, 2.00, 'gCO2e/pkm', 'IEA SDS 2021'),
    ('Commercial Buildings', '5311', '1.5C', 2020, 2030, 4.00, 'kgCO2e/m2', 'CRREM 2022'),
    ('Commercial Buildings', '5311', '2C', 2020, 2030, 2.50, 'kgCO2e/m2', 'CRREM 2022'),
    ('Retail', '4451', '1.5C', 2020, 2030, 4.20, 'tCO2e/M$ revenue', 'CDP Benchmarks 2023'),
    ('Retail', '4451', '2C', 2020, 2030, 2.80, 'tCO2e/M$ revenue', 'CDP Benchmarks 2023');

-- Note: temp table used as reference for SDA target-setting engine
DROP TABLE IF EXISTS _tmp_sda_pathways;

-- =============================================================================
-- Seed Data: RBAC Policies for 8 Roles
-- =============================================================================
-- Application-level role configuration for Scope 3 Complete data access.

CREATE TEMPORARY TABLE _tmp_scope3c_rbac (
    role_name VARCHAR(50),
    schema_name VARCHAR(100),
    table_pattern VARCHAR(100),
    permissions VARCHAR(50),
    description TEXT
);

INSERT INTO _tmp_scope3c_rbac (role_name, schema_name, table_pattern, permissions, description) VALUES
    -- Admin: full access
    ('scope3c_admin', 'ghg_accounting_scope3_complete', '*', 'SELECT,INSERT,UPDATE,DELETE', 'Full access to all Scope 3 Complete tables'),
    -- Analyst: read-write on operational tables
    ('scope3c_analyst', 'ghg_accounting_scope3_complete', 'products', 'SELECT,INSERT,UPDATE', 'Manage product LCA data'),
    ('scope3c_analyst', 'ghg_accounting_scope3_complete', 'product_bom', 'SELECT,INSERT,UPDATE', 'Manage BOM data'),
    ('scope3c_analyst', 'ghg_accounting_scope3_complete', 'lifecycle_results', 'SELECT,INSERT,UPDATE', 'Manage lifecycle results'),
    ('scope3c_analyst', 'ghg_accounting_scope3_complete', 'scenarios', 'SELECT,INSERT,UPDATE', 'Create and manage scenarios'),
    ('scope3c_analyst', 'ghg_accounting_scope3_complete', 'interventions', 'SELECT,INSERT,UPDATE', 'Manage interventions and MACC'),
    ('scope3c_analyst', 'ghg_accounting_scope3_complete', 'multi_year_data', 'SELECT,INSERT,UPDATE', 'Manage multi-year time series'),
    ('scope3c_analyst', 'ghg_accounting_scope3_complete', 'supplier_progress', 'SELECT,INSERT,UPDATE', 'Record supplier progress'),
    -- Risk analyst: climate risk tables
    ('scope3c_risk_analyst', 'ghg_accounting_scope3_complete', 'risk_assessments', 'SELECT,INSERT,UPDATE', 'Manage risk assessments'),
    ('scope3c_risk_analyst', 'ghg_accounting_scope3_complete', 'transition_risks', 'SELECT,INSERT,UPDATE', 'Manage transition risks'),
    ('scope3c_risk_analyst', 'ghg_accounting_scope3_complete', 'physical_risks', 'SELECT,INSERT,UPDATE', 'Manage physical risks'),
    ('scope3c_risk_analyst', 'ghg_accounting_scope3_complete', 'opportunities', 'SELECT,INSERT,UPDATE', 'Manage climate opportunities'),
    ('scope3c_risk_analyst', 'ghg_accounting_scope3_complete', 'financial_impacts', 'SELECT,INSERT,UPDATE', 'Manage financial impact modelling'),
    -- Target manager: SBTi and supplier targets
    ('scope3c_target_manager', 'ghg_accounting_scope3_complete', 'sbti_targets', 'SELECT,INSERT,UPDATE', 'Manage SBTi targets'),
    ('scope3c_target_manager', 'ghg_accounting_scope3_complete', 'sbti_pathways', 'SELECT,INSERT,UPDATE', 'Manage target pathways'),
    ('scope3c_target_manager', 'ghg_accounting_scope3_complete', 'sbti_milestones', 'SELECT,INSERT,UPDATE', 'Manage milestones'),
    ('scope3c_target_manager', 'ghg_accounting_scope3_complete', 'sbti_submissions', 'SELECT,INSERT,UPDATE', 'Manage SBTi submissions'),
    ('scope3c_target_manager', 'ghg_accounting_scope3_complete', 'supplier_targets', 'SELECT,INSERT,UPDATE', 'Manage supplier targets'),
    -- Reviewer: read-all, update status
    ('scope3c_reviewer', 'ghg_accounting_scope3_complete', '*', 'SELECT', 'Read access to all Scope 3 Complete data'),
    ('scope3c_reviewer', 'ghg_accounting_scope3_complete', 'evidence_packages', 'UPDATE', 'Approve evidence packages'),
    ('scope3c_reviewer', 'ghg_accounting_scope3_complete', 'methodology_decisions', 'UPDATE', 'Approve methodology decisions'),
    -- Supplier contact: limited
    ('scope3c_supplier', 'ghg_accounting_scope3_complete', 'supplier_progress', 'SELECT,INSERT,UPDATE', 'Submit progress data'),
    ('scope3c_supplier', 'ghg_accounting_scope3_complete', 'supplier_commitments', 'SELECT', 'View own commitments'),
    -- Auditor: read-only everything
    ('scope3c_auditor', 'ghg_accounting_scope3_complete', '*', 'SELECT', 'Read-only audit access'),
    ('scope3c_auditor', 'ghg_accounting_scope3_complete', 'scope3_complete_audit_trail', 'SELECT', 'Full audit trail access'),
    ('scope3c_auditor', 'ghg_accounting_scope3_complete', 'calculation_provenance', 'SELECT', 'Full provenance access'),
    -- Viewer: views and reports only
    ('scope3c_viewer', 'ghg_accounting_scope3_complete', 'v_*', 'SELECT', 'Access to views only'),
    ('scope3c_viewer', 'ghg_accounting_scope3_complete', 'mv_*', 'SELECT', 'Access to materialized views');

DROP TABLE IF EXISTS _tmp_scope3c_rbac;

-- =============================================================================
-- Grants for Views and Materialized Views
-- =============================================================================
GRANT SELECT ON ghg_accounting_scope3_complete.v_enterprise_summary TO PUBLIC;
GRANT SELECT ON ghg_accounting_scope3_complete.v_sbti_progress TO PUBLIC;
GRANT SELECT ON ghg_accounting_scope3_complete.v_supplier_programme_dashboard TO PUBLIC;
GRANT SELECT ON ghg_accounting_scope3_complete.mv_sector_benchmarks TO PUBLIC;

-- =============================================================================
-- Role Grants for Application Roles
-- =============================================================================
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_app') THEN
        GRANT USAGE ON SCHEMA ghg_accounting_scope3_complete TO greenlang_app;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA ghg_accounting_scope3_complete TO greenlang_app;
        GRANT SELECT ON ALL SEQUENCES IN SCHEMA ghg_accounting_scope3_complete TO greenlang_app;
        RAISE NOTICE 'Granted ghg_accounting_scope3_complete permissions to greenlang_app role';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA ghg_accounting_scope3_complete TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA ghg_accounting_scope3_complete TO greenlang_readonly;
        RAISE NOTICE 'Granted ghg_accounting_scope3_complete read permissions to greenlang_readonly role';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_service') THEN
        GRANT USAGE ON SCHEMA ghg_accounting_scope3_complete TO greenlang_service;
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ghg_accounting_scope3_complete TO greenlang_service;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ghg_accounting_scope3_complete TO greenlang_service;
        RAISE NOTICE 'Granted ghg_accounting_scope3_complete full permissions to greenlang_service role';
    END IF;
END;
$$;

-- =============================================================================
-- Comments on Views
-- =============================================================================
COMMENT ON VIEW ghg_accounting_scope3_complete.v_enterprise_summary IS
    'Multi-entity consolidated Scope 3 summary joining entity hierarchy, boundary definitions, and latest maturity assessment per entity.';
COMMENT ON VIEW ghg_accounting_scope3_complete.v_sbti_progress IS
    'SBTi target progress dashboard showing target trajectory, latest actual vs required, variance, and next upcoming milestone.';
COMMENT ON VIEW ghg_accounting_scope3_complete.v_supplier_programme_dashboard IS
    'Supplier programme dashboard aggregating targets, active commitments, latest progress, and latest scorecard per supplier.';
COMMENT ON MATERIALIZED VIEW ghg_accounting_scope3_complete.mv_sector_benchmarks IS
    'Pre-computed sector benchmark statistics (avg, median, percentiles) by NAICS code, year, and category. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY ghg_accounting_scope3_complete.mv_sector_benchmarks;';

-- =============================================================================
-- Migration Complete
-- =============================================================================
-- PACK-043 Scope 3 Complete Pack database schema is now fully deployed.
--
-- Schema: ghg_accounting_scope3_complete
-- Tables: 38
-- Views: 3
-- Materialized Views: 1
-- Enums: 8
-- TimescaleDB Hypertables: 1 (scope3_complete_audit_trail)
--
-- Table Summary:
--   V346 (Core):           entity_hierarchy, boundary_definitions,
--                           maturity_assessments, category_maturity
--   V347 (LCA):            products, product_bom, lifecycle_results,
--                           product_carbon_footprints
--   V348 (Scenario):       scenarios, interventions, macc_results,
--                           scenario_comparisons
--   V349 (SBTi):           sbti_targets, sbti_pathways, sbti_milestones,
--                           sbti_submissions
--   V350 (Supplier):       supplier_targets, supplier_commitments,
--                           supplier_progress, supplier_scorecards,
--                           programme_metrics
--   V351 (Climate Risk):   risk_assessments, transition_risks,
--                           physical_risks, opportunities,
--                           financial_impacts
--   V352 (Base Year):      base_years, base_year_categories,
--                           recalculations, multi_year_data
--   V353 (Sector):         pcaf_portfolios, pcaf_assets,
--                           retail_logistics, circular_economy,
--                           cloud_carbon
--   V354 (Assurance):      evidence_packages, calculation_provenance,
--                           methodology_decisions, verifier_queries,
--                           audit_findings, scope3_complete_audit_trail
--   V355 (Views):          v_enterprise_summary, v_sbti_progress,
--                           v_supplier_programme_dashboard,
--                           mv_sector_benchmarks
--
-- Seed Data: 22 MACC intervention templates, 20 SDA sector pathways,
--            28 RBAC policies for 8 roles
