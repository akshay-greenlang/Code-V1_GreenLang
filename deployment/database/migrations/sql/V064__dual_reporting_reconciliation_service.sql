-- V064__dual_reporting_reconciliation_service.sql
-- AGENT-MRV-013: Dual Reporting Reconciliation Agent Database Schema
-- Scope 2 dual-reporting reconciliation (location-based vs market-based)
-- Pattern: V063 (cooling_purchase) with gl_drr_ prefix
-- Tables: 14 tables, 3 hypertables, 2 continuous aggregates
-- Author: GL-BackendDeveloper
-- Date: 2026-02-23

-- ============================================================================
-- DIMENSION TABLES (4)
-- ============================================================================

-- 1. Regulatory Frameworks (7 rows)
CREATE TABLE gl_drr_frameworks (
    id SERIAL PRIMARY KEY,
    framework_key VARCHAR(40) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    version VARCHAR(20) NOT NULL,
    requirements_count INT NOT NULL DEFAULT 0,
    dual_reporting_required BOOLEAN NOT NULL DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_gl_drr_frameworks_tenant ON gl_drr_frameworks(tenant_id);
CREATE INDEX idx_gl_drr_frameworks_key ON gl_drr_frameworks(framework_key);

ALTER TABLE gl_drr_frameworks ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_frameworks_tenant_isolation ON gl_drr_frameworks
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 2. Discrepancy Types (8 rows)
CREATE TABLE gl_drr_discrepancy_types (
    id SERIAL PRIMARY KEY,
    type_key VARCHAR(40) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    typical_direction VARCHAR(20) NOT NULL CHECK (typical_direction IN ('market_lower', 'market_higher', 'either')),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_gl_drr_discrepancy_types_tenant ON gl_drr_discrepancy_types(tenant_id);
CREATE INDEX idx_gl_drr_discrepancy_types_key ON gl_drr_discrepancy_types(type_key);

ALTER TABLE gl_drr_discrepancy_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_discrepancy_types_tenant_isolation ON gl_drr_discrepancy_types
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 3. Residual Mix Factors (30+ regions)
CREATE TABLE gl_drr_residual_mix_factors (
    id SERIAL PRIMARY KEY,
    region_key VARCHAR(40) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    grid_ef_kgco2e_per_mwh NUMERIC(12,6) NOT NULL,
    residual_ef_kgco2e_per_mwh NUMERIC(12,6) NOT NULL,
    ratio NUMERIC(8,4) NOT NULL,
    year INT NOT NULL,
    source VARCHAR(200),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_gl_drr_residual_mix_tenant ON gl_drr_residual_mix_factors(tenant_id);
CREATE INDEX idx_gl_drr_residual_mix_region ON gl_drr_residual_mix_factors(region_key);
CREATE INDEX idx_gl_drr_residual_mix_year ON gl_drr_residual_mix_factors(year);

ALTER TABLE gl_drr_residual_mix_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_residual_mix_factors_tenant_isolation ON gl_drr_residual_mix_factors
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 4. Quality Dimension Weights
CREATE TABLE gl_drr_quality_weights (
    id SERIAL PRIMARY KEY,
    dimension VARCHAR(30) UNIQUE NOT NULL CHECK (dimension IN ('completeness', 'consistency', 'accuracy', 'transparency')),
    weight NUMERIC(5,4) NOT NULL CHECK (weight BETWEEN 0 AND 1),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'
);

CREATE INDEX idx_gl_drr_quality_weights_tenant ON gl_drr_quality_weights(tenant_id);

ALTER TABLE gl_drr_quality_weights ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_quality_weights_tenant_isolation ON gl_drr_quality_weights
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- ============================================================================
-- HYPERTABLES (3)
-- ============================================================================

-- 5. Reconciliation Runs (time-series)
CREATE TABLE gl_drr_reconciliations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    total_location_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    total_market_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    discrepancy_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    discrepancy_pct NUMERIC(10,4) NOT NULL DEFAULT 0,
    direction VARCHAR(20) NOT NULL DEFAULT 'equal' CHECK (direction IN ('market_lower', 'market_higher', 'equal')),
    materiality VARCHAR(20) NOT NULL DEFAULT 'immaterial' CHECK (materiality IN ('immaterial', 'minor', 'material', 'significant', 'extreme')),
    pif NUMERIC(10,8) NOT NULL DEFAULT 0,
    quality_score NUMERIC(6,4) NOT NULL DEFAULT 0,
    quality_grade VARCHAR(2) NOT NULL DEFAULT '',
    discrepancy_count INT NOT NULL DEFAULT 0,
    frameworks_checked INT NOT NULL DEFAULT 0,
    upstream_agent_count INT NOT NULL DEFAULT 0,
    energy_types_covered INT NOT NULL DEFAULT 0,
    facilities_count INT NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    processing_time_ms NUMERIC(12,3) NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_drr_reconciliations', 'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_gl_drr_reconciliations_tenant_time ON gl_drr_reconciliations(tenant_id, created_at DESC);
CREATE INDEX idx_gl_drr_reconciliations_status ON gl_drr_reconciliations(status);
CREATE INDEX idx_gl_drr_reconciliations_period ON gl_drr_reconciliations(period_start, period_end);
CREATE INDEX idx_gl_drr_reconciliations_direction ON gl_drr_reconciliations(direction);
CREATE INDEX idx_gl_drr_reconciliations_materiality ON gl_drr_reconciliations(materiality);

ALTER TABLE gl_drr_reconciliations ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_reconciliations_tenant_isolation ON gl_drr_reconciliations
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 6. Discrepancies (time-series linked to reconciliations)
CREATE TABLE gl_drr_discrepancies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    discrepancy_type VARCHAR(40) NOT NULL,
    energy_type VARCHAR(30) NOT NULL,
    facility_id VARCHAR(200),
    location_value_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    market_value_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    absolute_difference_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    percentage_difference NUMERIC(10,4) NOT NULL DEFAULT 0,
    direction VARCHAR(20) NOT NULL CHECK (direction IN ('market_lower', 'market_higher', 'equal')),
    materiality VARCHAR(20) NOT NULL DEFAULT 'immaterial',
    explanation TEXT,
    root_cause TEXT,
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_drr_discrepancies', 'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_gl_drr_discrepancies_recon ON gl_drr_discrepancies(reconciliation_id);
CREATE INDEX idx_gl_drr_discrepancies_tenant_time ON gl_drr_discrepancies(tenant_id, created_at DESC);
CREATE INDEX idx_gl_drr_discrepancies_type ON gl_drr_discrepancies(discrepancy_type);
CREATE INDEX idx_gl_drr_discrepancies_energy ON gl_drr_discrepancies(energy_type);
CREATE INDEX idx_gl_drr_discrepancies_materiality ON gl_drr_discrepancies(materiality);

ALTER TABLE gl_drr_discrepancies ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_discrepancies_tenant_isolation ON gl_drr_discrepancies
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 7. Aggregations (time-series)
CREATE TABLE gl_drr_aggregations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    aggregation_type VARCHAR(40) NOT NULL,
    group_key VARCHAR(200) NOT NULL,
    total_location_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    total_market_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    discrepancy_pct NUMERIC(10,4) NOT NULL DEFAULT 0,
    pif NUMERIC(10,8) NOT NULL DEFAULT 0,
    reconciliation_count INT NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    aggregated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_drr_aggregations', 'aggregated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_gl_drr_aggregations_tenant_time ON gl_drr_aggregations(tenant_id, aggregated_at DESC);
CREATE INDEX idx_gl_drr_aggregations_type ON gl_drr_aggregations(aggregation_type);
CREATE INDEX idx_gl_drr_aggregations_group ON gl_drr_aggregations(group_key);

ALTER TABLE gl_drr_aggregations ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_aggregations_tenant_isolation ON gl_drr_aggregations
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- ============================================================================
-- REGULAR TABLES (7)
-- ============================================================================

-- 8. Waterfall Items
CREATE TABLE gl_drr_waterfall_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id UUID NOT NULL,
    item_order INT NOT NULL,
    label VARCHAR(200) NOT NULL,
    discrepancy_type VARCHAR(40) NOT NULL,
    contribution_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    contribution_pct NUMERIC(10,4) NOT NULL DEFAULT 0,
    cumulative_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    direction VARCHAR(20) NOT NULL CHECK (direction IN ('increase', 'decrease', 'neutral')),
    explanation TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gl_drr_waterfall_recon ON gl_drr_waterfall_items(reconciliation_id);
CREATE INDEX idx_gl_drr_waterfall_order ON gl_drr_waterfall_items(reconciliation_id, item_order);

-- 9. Quality Assessments
CREATE TABLE gl_drr_quality_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    composite_score NUMERIC(6,4) NOT NULL DEFAULT 0,
    grade VARCHAR(2) NOT NULL DEFAULT '',
    completeness_score NUMERIC(6,4) NOT NULL DEFAULT 0,
    consistency_score NUMERIC(6,4) NOT NULL DEFAULT 0,
    accuracy_score NUMERIC(6,4) NOT NULL DEFAULT 0,
    transparency_score NUMERIC(6,4) NOT NULL DEFAULT 0,
    ef_hierarchy_scores JSONB DEFAULT '{}',
    flags JSONB DEFAULT '[]',
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gl_drr_quality_recon ON gl_drr_quality_assessments(reconciliation_id);
CREATE INDEX idx_gl_drr_quality_tenant ON gl_drr_quality_assessments(tenant_id);
CREATE INDEX idx_gl_drr_quality_grade ON gl_drr_quality_assessments(grade);

ALTER TABLE gl_drr_quality_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_quality_assessments_tenant_isolation ON gl_drr_quality_assessments
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 10. Reporting Tables
CREATE TABLE gl_drr_reporting_tables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    framework VARCHAR(40) NOT NULL,
    table_data JSONB NOT NULL DEFAULT '{}',
    rows_generated INT NOT NULL DEFAULT 0,
    columns_generated INT NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gl_drr_reporting_tables_recon ON gl_drr_reporting_tables(reconciliation_id);
CREATE INDEX idx_gl_drr_reporting_tables_tenant ON gl_drr_reporting_tables(tenant_id);
CREATE INDEX idx_gl_drr_reporting_tables_framework ON gl_drr_reporting_tables(framework);

ALTER TABLE gl_drr_reporting_tables ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_reporting_tables_tenant_isolation ON gl_drr_reporting_tables
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 11. Trend Data Points
CREATE TABLE gl_drr_trend_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    period VARCHAR(20) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    location_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    market_tco2e NUMERIC(18,8) NOT NULL DEFAULT 0,
    pif NUMERIC(10,8) NOT NULL DEFAULT 0,
    re100_pct NUMERIC(8,4) NOT NULL DEFAULT 0,
    discrepancy_pct NUMERIC(10,4) NOT NULL DEFAULT 0,
    yoy_location_pct NUMERIC(10,4),
    yoy_market_pct NUMERIC(10,4),
    intensity_revenue NUMERIC(12,8),
    intensity_fte NUMERIC(12,8),
    intensity_floor_area NUMERIC(12,8),
    intensity_production NUMERIC(12,8),
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gl_drr_trend_data_recon ON gl_drr_trend_data(reconciliation_id);
CREATE INDEX idx_gl_drr_trend_data_tenant ON gl_drr_trend_data(tenant_id);
CREATE INDEX idx_gl_drr_trend_data_period ON gl_drr_trend_data(period_start, period_end);

ALTER TABLE gl_drr_trend_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_trend_data_tenant_isolation ON gl_drr_trend_data
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 12. Compliance Checks
CREATE TABLE gl_drr_compliance_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    framework VARCHAR(40) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('compliant', 'non_compliant', 'partial', 'not_applicable')),
    requirements_total INT NOT NULL DEFAULT 0,
    requirements_met INT NOT NULL DEFAULT 0,
    score NUMERIC(6,4) NOT NULL DEFAULT 0,
    findings JSONB DEFAULT '[]',
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gl_drr_compliance_recon ON gl_drr_compliance_checks(reconciliation_id);
CREATE INDEX idx_gl_drr_compliance_tenant ON gl_drr_compliance_checks(tenant_id);
CREATE INDEX idx_gl_drr_compliance_framework ON gl_drr_compliance_checks(framework);
CREATE INDEX idx_gl_drr_compliance_status ON gl_drr_compliance_checks(status);

ALTER TABLE gl_drr_compliance_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_compliance_checks_tenant_isolation ON gl_drr_compliance_checks
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 13. Exports
CREATE TABLE gl_drr_exports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    export_format VARCHAR(10) NOT NULL CHECK (export_format IN ('json', 'csv', 'xlsx', 'pdf')),
    filename VARCHAR(500) NOT NULL,
    content_type VARCHAR(100) NOT NULL DEFAULT 'application/json',
    file_size_bytes BIGINT,
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gl_drr_exports_recon ON gl_drr_exports(reconciliation_id);
CREATE INDEX idx_gl_drr_exports_tenant ON gl_drr_exports(tenant_id);
CREATE INDEX idx_gl_drr_exports_format ON gl_drr_exports(export_format);

ALTER TABLE gl_drr_exports ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_exports_tenant_isolation ON gl_drr_exports
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- 14. Batch Jobs
CREATE TABLE gl_drr_batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    total_periods INT NOT NULL DEFAULT 0,
    completed INT NOT NULL DEFAULT 0,
    failed INT NOT NULL DEFAULT 0,
    total_location_tco2e NUMERIC(18,8),
    total_market_tco2e NUMERIC(18,8),
    processing_time_ms NUMERIC(12,3),
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_gl_drr_batch_jobs_tenant ON gl_drr_batch_jobs(tenant_id);
CREATE INDEX idx_gl_drr_batch_jobs_status ON gl_drr_batch_jobs(status);
CREATE INDEX idx_gl_drr_batch_jobs_created ON gl_drr_batch_jobs(created_at DESC);

ALTER TABLE gl_drr_batch_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_drr_batch_jobs_tenant_isolation ON gl_drr_batch_jobs
    USING (tenant_id = current_setting('app.current_tenant', TRUE)::uuid);

-- ============================================================================
-- CONTINUOUS AGGREGATES (2)
-- ============================================================================

-- 15. Hourly Reconciliation Stats
CREATE MATERIALIZED VIEW gl_drr_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    tenant_id,
    direction,
    COUNT(*) AS reconciliation_count,
    SUM(total_location_tco2e) AS total_location_tco2e,
    SUM(total_market_tco2e) AS total_market_tco2e,
    AVG(discrepancy_pct) AS avg_discrepancy_pct,
    AVG(quality_score) AS avg_quality_score,
    AVG(pif) AS avg_pif,
    SUM(discrepancy_count) AS total_discrepancies,
    MIN(created_at) AS first_reconciliation,
    MAX(created_at) AS last_reconciliation
FROM gl_drr_reconciliations
GROUP BY bucket, tenant_id, direction
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_drr_hourly_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX idx_gl_drr_hourly_stats_tenant_time ON gl_drr_hourly_stats(tenant_id, bucket DESC);
CREATE INDEX idx_gl_drr_hourly_stats_direction ON gl_drr_hourly_stats(direction);

-- 16. Daily Reconciliation Stats
CREATE MATERIALIZED VIEW gl_drr_daily_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at) AS bucket,
    tenant_id,
    direction,
    COUNT(*) AS reconciliation_count,
    SUM(total_location_tco2e) AS total_location_tco2e,
    SUM(total_market_tco2e) AS total_market_tco2e,
    AVG(discrepancy_pct) AS avg_discrepancy_pct,
    AVG(quality_score) AS avg_quality_score,
    AVG(pif) AS avg_pif,
    SUM(discrepancy_count) AS total_discrepancies,
    MIN(created_at) AS first_reconciliation,
    MAX(created_at) AS last_reconciliation
FROM gl_drr_reconciliations
GROUP BY bucket, tenant_id, direction
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_drr_daily_stats',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_gl_drr_daily_stats_tenant_time ON gl_drr_daily_stats(tenant_id, bucket DESC);
CREATE INDEX idx_gl_drr_daily_stats_direction ON gl_drr_daily_stats(direction);

-- ============================================================================
-- SEED DATA: Regulatory Frameworks (7 rows)
-- ============================================================================

INSERT INTO gl_drr_frameworks (framework_key, display_name, version, requirements_count, dual_reporting_required, notes) VALUES
('ghg_protocol', 'GHG Protocol Scope 2 Guidance', '2015', 13, TRUE, 'Requires both location-based and market-based reporting with disclosure of both totals'),
('csrd_esrs', 'CSRD/ESRS E1 Climate Change', '2024', 10, TRUE, 'European Sustainability Reporting Standards E1 requires dual methodology reporting'),
('cdp', 'CDP Climate Change', '2024', 8, TRUE, 'Carbon Disclosure Project requires both Scope 2 methods since 2015'),
('sbti', 'Science Based Targets initiative', '2023', 6, TRUE, 'SBTi requires market-based for target-setting, location-based for tracking'),
('gri', 'GRI 305 Emissions', '2016', 7, TRUE, 'GRI requires disclosure of both Scope 2 methods'),
('iso_14064', 'ISO 14064-1', '2018', 10, TRUE, 'ISO requires reporting of indirect emissions with methodology disclosure'),
('re100', 'RE100', '2024', 5, FALSE, 'RE100 uses market-based method for renewable energy progress tracking');

-- ============================================================================
-- SEED DATA: Discrepancy Types (8 rows)
-- ============================================================================

INSERT INTO gl_drr_discrepancy_types (type_key, display_name, description, typical_direction) VALUES
('rec_go_impact', 'REC/GO Impact', 'Difference caused by Renewable Energy Certificates or Guarantees of Origin lowering market-based emissions', 'market_lower'),
('residual_mix_uplift', 'Residual Mix Uplift', 'Difference from residual mix EFs being higher than grid average due to tracked renewables being stripped out', 'market_higher'),
('supplier_ef_delta', 'Supplier EF Delta', 'Difference from supplier-specific emission factors differing from grid average', 'either'),
('geographic_mismatch', 'Geographic Mismatch', 'Difference from contractual instruments sourced from different grid regions than consumption', 'either'),
('temporal_mismatch', 'Temporal Mismatch', 'Difference from timing discrepancies between instrument vintage and consumption period', 'either'),
('partial_coverage', 'Partial Coverage', 'Difference from instruments covering only a portion of total consumption', 'market_higher'),
('steam_heat_method', 'Steam/Heat Method Divergence', 'Methodological differences in steam and heating between location and market calculations', 'either'),
('grid_update_timing', 'Grid Update Timing', 'Difference from using different vintage grid emission factors across methods', 'either');

-- ============================================================================
-- SEED DATA: Residual Mix Factors (30 regions)
-- ============================================================================

INSERT INTO gl_drr_residual_mix_factors (region_key, display_name, grid_ef_kgco2e_per_mwh, residual_ef_kgco2e_per_mwh, ratio, year, source) VALUES
-- Europe
('eu_average', 'EU Average', 231.0, 356.0, 1.54, 2023, 'AIB European Residual Mix 2023'),
('germany', 'Germany', 366.0, 530.0, 1.45, 2023, 'Umweltbundesamt 2023'),
('france', 'France', 56.0, 410.0, 7.32, 2023, 'AIB European Residual Mix 2023'),
('uk', 'United Kingdom', 207.0, 349.0, 1.69, 2023, 'UK DEFRA/BEIS 2023'),
('spain', 'Spain', 128.0, 310.0, 2.42, 2023, 'AIB European Residual Mix 2023'),
('italy', 'Italy', 258.0, 392.0, 1.52, 2023, 'AIB European Residual Mix 2023'),
('netherlands', 'Netherlands', 328.0, 472.0, 1.44, 2023, 'AIB European Residual Mix 2023'),
('sweden', 'Sweden', 12.0, 350.0, 29.17, 2023, 'AIB European Residual Mix 2023'),
('norway', 'Norway', 8.0, 342.0, 42.75, 2023, 'AIB European Residual Mix 2023'),
('poland', 'Poland', 635.0, 710.0, 1.12, 2023, 'AIB European Residual Mix 2023'),
('austria', 'Austria', 82.0, 345.0, 4.21, 2023, 'AIB European Residual Mix 2023'),
('denmark', 'Denmark', 119.0, 378.0, 3.18, 2023, 'AIB European Residual Mix 2023'),
('finland', 'Finland', 72.0, 312.0, 4.33, 2023, 'AIB European Residual Mix 2023'),
('belgium', 'Belgium', 148.0, 390.0, 2.64, 2023, 'AIB European Residual Mix 2023'),
('ireland', 'Ireland', 296.0, 452.0, 1.53, 2023, 'AIB European Residual Mix 2023'),
-- North America
('us_average', 'US National Average', 371.0, 420.0, 1.13, 2023, 'EPA eGRID 2023'),
('us_camx', 'US CAMX (California)', 206.0, 310.0, 1.50, 2023, 'EPA eGRID 2023'),
('us_erct', 'US ERCT (Texas)', 373.0, 405.0, 1.09, 2023, 'EPA eGRID 2023'),
('us_nyup', 'US NYUP (New York)', 112.0, 280.0, 2.50, 2023, 'EPA eGRID 2023'),
('us_newe', 'US NEWE (New England)', 180.0, 340.0, 1.89, 2023, 'EPA eGRID 2023'),
('canada', 'Canada', 110.0, 250.0, 2.27, 2023, 'Environment Canada 2023'),
-- Asia-Pacific
('japan', 'Japan', 432.0, 500.0, 1.16, 2023, 'MOEJ Japan 2023'),
('south_korea', 'South Korea', 415.0, 468.0, 1.13, 2023, 'KEITI 2023'),
('china', 'China', 555.0, 600.0, 1.08, 2023, 'MEEE China Grid EF 2023'),
('india', 'India', 708.0, 745.0, 1.05, 2023, 'CEA India 2023'),
('australia', 'Australia', 670.0, 720.0, 1.07, 2023, 'Clean Energy Regulator 2023'),
('singapore', 'Singapore', 408.0, 430.0, 1.05, 2023, 'EMA Singapore 2023'),
-- Other
('brazil', 'Brazil', 60.0, 180.0, 3.00, 2023, 'MCTIC Brazil 2023'),
('south_africa', 'South Africa', 928.0, 960.0, 1.03, 2023, 'Eskom 2023'),
('uae', 'United Arab Emirates', 422.0, 440.0, 1.04, 2023, 'UAE MOEI 2023');

-- ============================================================================
-- SEED DATA: Quality Dimension Weights (4 rows)
-- ============================================================================

INSERT INTO gl_drr_quality_weights (dimension, weight, description) VALUES
('completeness', 0.30, 'Coverage of all energy types, facilities, and reporting periods in both methods'),
('consistency', 0.25, 'Methodological consistency across upstream agents and reporting boundaries'),
('accuracy', 0.25, 'Emission factor hierarchy quality and data tier assessment'),
('transparency', 0.20, 'Documentation of assumptions, methodologies, and provenance trail completeness');

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE gl_drr_frameworks IS 'AGENT-MRV-013: Supported regulatory frameworks for dual reporting compliance';
COMMENT ON TABLE gl_drr_discrepancy_types IS 'AGENT-MRV-013: Eight discrepancy type classifications per GHG Protocol Scope 2 Guidance';
COMMENT ON TABLE gl_drr_residual_mix_factors IS 'AGENT-MRV-013: Grid vs residual mix emission factors by region for market-based uplift analysis';
COMMENT ON TABLE gl_drr_quality_weights IS 'AGENT-MRV-013: Dimensional weights for composite quality score calculation';
COMMENT ON TABLE gl_drr_reconciliations IS 'AGENT-MRV-013: Primary reconciliation run results (hypertable)';
COMMENT ON TABLE gl_drr_discrepancies IS 'AGENT-MRV-013: Individual discrepancies identified per reconciliation (hypertable)';
COMMENT ON TABLE gl_drr_aggregations IS 'AGENT-MRV-013: Aggregated reconciliation data by dimension (hypertable)';
COMMENT ON TABLE gl_drr_waterfall_items IS 'AGENT-MRV-013: Waterfall decomposition items (bridge from location to market)';
COMMENT ON TABLE gl_drr_quality_assessments IS 'AGENT-MRV-013: Quality assessment results per reconciliation';
COMMENT ON TABLE gl_drr_reporting_tables IS 'AGENT-MRV-013: Framework-specific reporting table data';
COMMENT ON TABLE gl_drr_trend_data IS 'AGENT-MRV-013: Multi-period trend analysis data points';
COMMENT ON TABLE gl_drr_compliance_checks IS 'AGENT-MRV-013: Per-framework compliance check results';
COMMENT ON TABLE gl_drr_exports IS 'AGENT-MRV-013: Report export audit trail';
COMMENT ON TABLE gl_drr_batch_jobs IS 'AGENT-MRV-013: Batch reconciliation job tracking';
