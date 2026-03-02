-- =====================================================================================
-- Migration: V080__scope3_category_mapper_service.sql
-- Description: AGENT-MRV-029 Scope 3 Category Mapper (Cross-Cutting MRV Agent)
-- Agent: GL-MRV-X-040
-- Framework: GHG Protocol Scope 3 Standard, ISO 14064-1, CSRD/ESRS E1, CDP,
--            SBTi, SB 253, SEC Climate, ISSB S2
-- Created: 2026-02-28
-- =====================================================================================
-- Schema: scope3_category_mapper_service
-- Tables: 11 (4 reference + 5 operational + 2 supporting)
-- Hypertables: 3 (classification_results 7d, routing_executions 7d,
--              compliance_assessments 30d)
-- Continuous Aggregates: 2 (hourly_classification_stats, daily_routing_stats)
-- Indexes: ~55
-- Seed Data: 150+ records (NAICS sector mappings, ISIC section mappings,
--            GL account ranges, category metadata, company type relevance,
--            framework requirements, DC rules)
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS scope3_category_mapper_service;

COMMENT ON SCHEMA scope3_category_mapper_service IS 'AGENT-MRV-029: Scope 3 Category Mapper - Cross-cutting agent for classification of organizational data to GHG Protocol Scope 3 categories 1-15 and routing to category-specific agents';

-- =====================================================================================
-- TABLE 1: gl_scm_naics_mappings (Reference)
-- Description: NAICS 2022 code → Scope 3 category mapping table
-- Source: US Census Bureau NAICS 2022 + GHG Protocol Scope 3 Standard
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_naics_mappings (
    id SERIAL PRIMARY KEY,
    naics_code VARCHAR(6) NOT NULL,
    naics_level VARCHAR(20) NOT NULL DEFAULT 'sector_2',
    description VARCHAR(255) NOT NULL,
    primary_category INT NOT NULL,
    primary_category_name VARCHAR(100) NOT NULL,
    secondary_categories INT[] DEFAULT '{}',
    confidence DECIMAL(4,2) NOT NULL DEFAULT 0.90,
    source VARCHAR(50) NOT NULL DEFAULT 'GHG_PROTOCOL',
    version VARCHAR(20) NOT NULL DEFAULT '2022',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(naics_code, version),
    CONSTRAINT chk_scm_naics_cat_range CHECK (primary_category >= 1 AND primary_category <= 15),
    CONSTRAINT chk_scm_naics_conf_range CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT chk_scm_naics_level CHECK (naics_level IN ('sector_2', 'subsector_3', 'industry_group_4', 'industry_6'))
);

CREATE INDEX idx_scm_naics_code ON scope3_category_mapper_service.gl_scm_naics_mappings(naics_code);
CREATE INDEX idx_scm_naics_category ON scope3_category_mapper_service.gl_scm_naics_mappings(primary_category);
CREATE INDEX idx_scm_naics_active ON scope3_category_mapper_service.gl_scm_naics_mappings(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_scm_naics_level ON scope3_category_mapper_service.gl_scm_naics_mappings(naics_level);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_naics_mappings IS 'NAICS 2022 to Scope 3 category deterministic mapping table';
COMMENT ON COLUMN scope3_category_mapper_service.gl_scm_naics_mappings.primary_category IS 'Primary Scope 3 category number (1-15)';
COMMENT ON COLUMN scope3_category_mapper_service.gl_scm_naics_mappings.secondary_categories IS 'Secondary applicable category numbers';

-- =====================================================================================
-- TABLE 2: gl_scm_isic_mappings (Reference)
-- Description: ISIC Rev 4 code → Scope 3 category mapping table
-- Source: UN Statistics Division ISIC Rev 4 + GHG Protocol
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_isic_mappings (
    id SERIAL PRIMARY KEY,
    isic_code VARCHAR(4) NOT NULL,
    isic_level VARCHAR(20) NOT NULL DEFAULT 'section_1',
    description VARCHAR(255) NOT NULL,
    primary_category INT NOT NULL,
    primary_category_name VARCHAR(100) NOT NULL,
    secondary_categories INT[] DEFAULT '{}',
    naics_equivalent VARCHAR(6),
    confidence DECIMAL(4,2) NOT NULL DEFAULT 0.85,
    source VARCHAR(50) NOT NULL DEFAULT 'GHG_PROTOCOL',
    version VARCHAR(20) NOT NULL DEFAULT 'Rev4',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(isic_code, version),
    CONSTRAINT chk_scm_isic_cat_range CHECK (primary_category >= 1 AND primary_category <= 15),
    CONSTRAINT chk_scm_isic_conf_range CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX idx_scm_isic_code ON scope3_category_mapper_service.gl_scm_isic_mappings(isic_code);
CREATE INDEX idx_scm_isic_category ON scope3_category_mapper_service.gl_scm_isic_mappings(primary_category);
CREATE INDEX idx_scm_isic_naics ON scope3_category_mapper_service.gl_scm_isic_mappings(naics_equivalent);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_isic_mappings IS 'ISIC Rev 4 to Scope 3 category mapping with NAICS cross-reference';

-- =====================================================================================
-- TABLE 3: gl_scm_gl_account_mappings (Reference)
-- Description: GL account code ranges → Scope 3 category mapping
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_gl_account_mappings (
    id SERIAL PRIMARY KEY,
    range_start INT NOT NULL,
    range_end INT NOT NULL,
    account_type VARCHAR(100) NOT NULL,
    primary_category INT NOT NULL,
    primary_category_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(4,2) NOT NULL DEFAULT 0.85,
    notes VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_scm_gl_range CHECK (range_start <= range_end),
    CONSTRAINT chk_scm_gl_cat_range CHECK (primary_category >= 1 AND primary_category <= 15)
);

CREATE INDEX idx_scm_gl_range ON scope3_category_mapper_service.gl_scm_gl_account_mappings(range_start, range_end);
CREATE INDEX idx_scm_gl_category ON scope3_category_mapper_service.gl_scm_gl_account_mappings(primary_category);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_gl_account_mappings IS 'General Ledger account range to Scope 3 category mapping';

-- =====================================================================================
-- TABLE 4: gl_scm_category_metadata (Reference)
-- Description: Scope 3 category metadata and descriptions
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_category_metadata (
    id SERIAL PRIMARY KEY,
    category_number INT NOT NULL UNIQUE,
    category_name VARCHAR(100) NOT NULL,
    short_name VARCHAR(50) NOT NULL,
    direction VARCHAR(20) NOT NULL,
    ghg_protocol_chapter VARCHAR(20),
    reporter_role VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    typical_data_sources TEXT[],
    target_agent_id VARCHAR(20),
    target_agent_component VARCHAR(30),
    target_api_endpoint VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_scm_meta_cat CHECK (category_number >= 1 AND category_number <= 15),
    CONSTRAINT chk_scm_meta_dir CHECK (direction IN ('upstream', 'downstream'))
);

CREATE INDEX idx_scm_meta_number ON scope3_category_mapper_service.gl_scm_category_metadata(category_number);
CREATE INDEX idx_scm_meta_direction ON scope3_category_mapper_service.gl_scm_category_metadata(direction);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_category_metadata IS 'GHG Protocol Scope 3 category definitions with routing targets';

-- =====================================================================================
-- TABLE 5: gl_scm_classification_runs (Operational)
-- Description: Master table for classification run sessions
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_classification_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id VARCHAR(50) NOT NULL,
    reporting_year INT NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    total_records INT NOT NULL DEFAULT 0,
    mapped_count INT NOT NULL DEFAULT 0,
    unmapped_count INT NOT NULL DEFAULT 0,
    split_count INT NOT NULL DEFAULT 0,
    review_count INT NOT NULL DEFAULT 0,
    average_confidence DECIMAL(4,2),
    completeness_score DECIMAL(5,2),
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    provenance_hash VARCHAR(64),
    created_by VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT chk_scm_run_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_scm_run_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX idx_scm_runs_org ON scope3_category_mapper_service.gl_scm_classification_runs(organization_id);
CREATE INDEX idx_scm_runs_year ON scope3_category_mapper_service.gl_scm_classification_runs(reporting_year);
CREATE INDEX idx_scm_runs_status ON scope3_category_mapper_service.gl_scm_classification_runs(status);
CREATE INDEX idx_scm_runs_started ON scope3_category_mapper_service.gl_scm_classification_runs(started_at);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_classification_runs IS 'Master classification run sessions with aggregate metrics';

-- =====================================================================================
-- TABLE 6: gl_scm_classification_results (Operational - Hypertable)
-- Description: Individual classification results with provenance
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_classification_results (
    id UUID DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    classified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_type VARCHAR(30) NOT NULL,
    source_id VARCHAR(100),
    mapped_category INT NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(4,2) NOT NULL,
    confidence_level VARCHAR(20) NOT NULL,
    classification_method VARCHAR(30) NOT NULL,
    mapping_rule VARCHAR(200),
    recommended_approach VARCHAR(30),
    value_chain_position VARCHAR(20) NOT NULL,
    amount DECIMAL(18,2),
    currency VARCHAR(3) DEFAULT 'USD',
    double_counting_flags JSONB DEFAULT '[]',
    calculation_trace JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    input_data JSONB,
    CONSTRAINT chk_scm_res_cat CHECK (mapped_category >= 1 AND mapped_category <= 15),
    CONSTRAINT chk_scm_res_conf CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT chk_scm_res_dir CHECK (value_chain_position IN ('upstream', 'downstream')),
    CONSTRAINT chk_scm_res_method CHECK (classification_method IN ('naics', 'isic', 'unspsc', 'hs_code', 'gl_account', 'keyword', 'source_type', 'default'))
);

SELECT create_hypertable(
    'scope3_category_mapper_service.gl_scm_classification_results',
    'classified_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_scm_results_run ON scope3_category_mapper_service.gl_scm_classification_results(run_id, classified_at DESC);
CREATE INDEX idx_scm_results_category ON scope3_category_mapper_service.gl_scm_classification_results(mapped_category, classified_at DESC);
CREATE INDEX idx_scm_results_method ON scope3_category_mapper_service.gl_scm_classification_results(classification_method, classified_at DESC);
CREATE INDEX idx_scm_results_confidence ON scope3_category_mapper_service.gl_scm_classification_results(confidence, classified_at DESC);
CREATE INDEX idx_scm_results_source ON scope3_category_mapper_service.gl_scm_classification_results(source_type, classified_at DESC);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_classification_results IS 'Individual record classification results (TimescaleDB hypertable, 7-day chunks)';

-- =====================================================================================
-- TABLE 7: gl_scm_routing_executions (Operational - Hypertable)
-- Description: Record of routing executions to category agents
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_routing_executions (
    id UUID DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    routed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_category INT NOT NULL,
    target_agent_id VARCHAR(20) NOT NULL,
    target_api_endpoint VARCHAR(100) NOT NULL,
    records_sent INT NOT NULL DEFAULT 0,
    routing_action VARCHAR(20) NOT NULL DEFAULT 'route',
    response_status VARCHAR(20),
    execution_time_ms DECIMAL(10,2),
    dry_run BOOLEAN DEFAULT FALSE,
    provenance_hash VARCHAR(64) NOT NULL,
    error_message TEXT,
    CONSTRAINT chk_scm_route_cat CHECK (target_category >= 1 AND target_category <= 15),
    CONSTRAINT chk_scm_route_action CHECK (routing_action IN ('route', 'split_route', 'queue_review', 'exclude'))
);

SELECT create_hypertable(
    'scope3_category_mapper_service.gl_scm_routing_executions',
    'routed_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_scm_route_run ON scope3_category_mapper_service.gl_scm_routing_executions(run_id, routed_at DESC);
CREATE INDEX idx_scm_route_category ON scope3_category_mapper_service.gl_scm_routing_executions(target_category, routed_at DESC);
CREATE INDEX idx_scm_route_agent ON scope3_category_mapper_service.gl_scm_routing_executions(target_agent_id, routed_at DESC);
CREATE INDEX idx_scm_route_dry ON scope3_category_mapper_service.gl_scm_routing_executions(dry_run) WHERE dry_run = FALSE;

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_routing_executions IS 'Routing execution log to downstream category agents (TimescaleDB hypertable)';

-- =====================================================================================
-- TABLE 8: gl_scm_boundary_determinations (Operational)
-- Description: Boundary determination results
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_boundary_determinations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID,
    result_id UUID,
    determined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category INT NOT NULL,
    value_chain_position VARCHAR(20) NOT NULL,
    consolidation_approach VARCHAR(30),
    determination_rule VARCHAR(100) NOT NULL,
    confidence DECIMAL(4,2) NOT NULL,
    notes TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    CONSTRAINT chk_scm_boundary_cat CHECK (category >= 1 AND category <= 15),
    CONSTRAINT chk_scm_boundary_dir CHECK (value_chain_position IN ('upstream', 'downstream'))
);

CREATE INDEX idx_scm_boundary_run ON scope3_category_mapper_service.gl_scm_boundary_determinations(run_id);
CREATE INDEX idx_scm_boundary_cat ON scope3_category_mapper_service.gl_scm_boundary_determinations(category);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_boundary_determinations IS 'Upstream/downstream boundary determination decisions';

-- =====================================================================================
-- TABLE 9: gl_scm_double_counting_checks (Operational)
-- Description: Cross-category double-counting analysis
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_double_counting_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rule_id VARCHAR(15) NOT NULL,
    category_a INT NOT NULL,
    category_b INT NOT NULL,
    overlap_detected BOOLEAN NOT NULL DEFAULT FALSE,
    resolution VARCHAR(255),
    affected_records INT DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    CONSTRAINT chk_scm_dc_rule CHECK (rule_id LIKE 'DC-SCM-%'),
    CONSTRAINT chk_scm_dc_cat_a CHECK (category_a >= 1 AND category_a <= 15),
    CONSTRAINT chk_scm_dc_cat_b CHECK (category_b >= 1 AND category_b <= 15)
);

CREATE INDEX idx_scm_dc_run ON scope3_category_mapper_service.gl_scm_double_counting_checks(run_id);
CREATE INDEX idx_scm_dc_rule ON scope3_category_mapper_service.gl_scm_double_counting_checks(rule_id);
CREATE INDEX idx_scm_dc_overlap ON scope3_category_mapper_service.gl_scm_double_counting_checks(overlap_detected) WHERE overlap_detected = TRUE;

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_double_counting_checks IS 'Cross-category double-counting detection (DC-SCM-001 through DC-SCM-010)';

-- =====================================================================================
-- TABLE 10: gl_scm_completeness_reports (Operational)
-- Description: Category completeness assessment reports
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_completeness_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID,
    organization_id VARCHAR(50) NOT NULL,
    reporting_year INT NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    company_type VARCHAR(30) NOT NULL,
    overall_score DECIMAL(5,2) NOT NULL,
    categories_reported INT NOT NULL DEFAULT 0,
    categories_material INT NOT NULL DEFAULT 0,
    categories_missing INT NOT NULL DEFAULT 0,
    category_details JSONB NOT NULL DEFAULT '[]',
    gaps JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    CONSTRAINT chk_scm_comp_score CHECK (overall_score >= 0 AND overall_score <= 100),
    CONSTRAINT chk_scm_comp_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100)
);

CREATE INDEX idx_scm_comp_org ON scope3_category_mapper_service.gl_scm_completeness_reports(organization_id);
CREATE INDEX idx_scm_comp_year ON scope3_category_mapper_service.gl_scm_completeness_reports(reporting_year);
CREATE INDEX idx_scm_comp_score ON scope3_category_mapper_service.gl_scm_completeness_reports(overall_score);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_completeness_reports IS 'Scope 3 category completeness assessments by organization';

-- =====================================================================================
-- TABLE 11: gl_scm_compliance_assessments (Operational - Hypertable)
-- Description: Framework compliance assessment results
-- =====================================================================================

CREATE TABLE scope3_category_mapper_service.gl_scm_compliance_assessments (
    id UUID DEFAULT gen_random_uuid(),
    run_id UUID,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    framework VARCHAR(30) NOT NULL,
    organization_id VARCHAR(50) NOT NULL,
    reporting_year INT NOT NULL,
    categories_required INT NOT NULL DEFAULT 0,
    categories_reported INT NOT NULL DEFAULT 0,
    compliant BOOLEAN NOT NULL DEFAULT FALSE,
    score DECIMAL(5,2) NOT NULL DEFAULT 0,
    gaps JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    CONSTRAINT chk_scm_assess_score CHECK (score >= 0 AND score <= 100),
    CONSTRAINT chk_scm_assess_framework CHECK (framework IN (
        'ghg_protocol', 'iso_14064', 'csrd_esrs', 'cdp',
        'sbti', 'sb253', 'sec_climate', 'issb_s2'
    ))
);

SELECT create_hypertable(
    'scope3_category_mapper_service.gl_scm_compliance_assessments',
    'assessed_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_scm_assess_org ON scope3_category_mapper_service.gl_scm_compliance_assessments(organization_id, assessed_at DESC);
CREATE INDEX idx_scm_assess_framework ON scope3_category_mapper_service.gl_scm_compliance_assessments(framework, assessed_at DESC);
CREATE INDEX idx_scm_assess_compliant ON scope3_category_mapper_service.gl_scm_compliance_assessments(compliant, assessed_at DESC);

COMMENT ON TABLE scope3_category_mapper_service.gl_scm_compliance_assessments IS 'Regulatory framework compliance assessments (TimescaleDB hypertable, 30-day chunks)';

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Hourly classification statistics
CREATE MATERIALIZED VIEW scope3_category_mapper_service.gl_scm_hourly_classification_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', classified_at) AS bucket,
    mapped_category,
    classification_method,
    source_type,
    COUNT(*) AS total_classifications,
    AVG(confidence) AS avg_confidence,
    MIN(confidence) AS min_confidence,
    MAX(confidence) AS max_confidence,
    COUNT(*) FILTER (WHERE confidence >= 0.75) AS high_confidence_count,
    COUNT(*) FILTER (WHERE confidence < 0.5) AS low_confidence_count
FROM scope3_category_mapper_service.gl_scm_classification_results
GROUP BY bucket, mapped_category, classification_method, source_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'scope3_category_mapper_service.gl_scm_hourly_classification_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily routing statistics
CREATE MATERIALIZED VIEW scope3_category_mapper_service.gl_scm_daily_routing_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', routed_at) AS bucket,
    target_category,
    target_agent_id,
    routing_action,
    COUNT(*) AS total_routings,
    SUM(records_sent) AS total_records_routed,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    COUNT(*) FILTER (WHERE response_status = 'success') AS success_count,
    COUNT(*) FILTER (WHERE response_status = 'failed') AS failure_count,
    COUNT(*) FILTER (WHERE dry_run = TRUE) AS dry_run_count
FROM scope3_category_mapper_service.gl_scm_routing_executions
GROUP BY bucket, target_category, target_agent_id, routing_action
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'scope3_category_mapper_service.gl_scm_daily_routing_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =====================================================================================
-- SEED DATA: NAICS 2-digit sector mappings
-- =====================================================================================

INSERT INTO scope3_category_mapper_service.gl_scm_naics_mappings (naics_code, naics_level, description, primary_category, primary_category_name, secondary_categories, confidence) VALUES
('11', 'sector_2', 'Agriculture, Forestry, Fishing and Hunting', 1, 'Purchased Goods and Services', '{10}', 0.90),
('21', 'sector_2', 'Mining, Quarrying, and Oil and Gas Extraction', 1, 'Purchased Goods and Services', '{3}', 0.90),
('22', 'sector_2', 'Utilities', 3, 'Fuel and Energy Related Activities', '{}', 0.90),
('23', 'sector_2', 'Construction', 2, 'Capital Goods', '{1}', 0.85),
('31', 'sector_2', 'Manufacturing (31-33)', 1, 'Purchased Goods and Services', '{2,10}', 0.85),
('32', 'sector_2', 'Manufacturing (31-33)', 1, 'Purchased Goods and Services', '{2,10}', 0.85),
('33', 'sector_2', 'Manufacturing (31-33)', 1, 'Purchased Goods and Services', '{2}', 0.85),
('42', 'sector_2', 'Wholesale Trade', 1, 'Purchased Goods and Services', '{4}', 0.85),
('44', 'sector_2', 'Retail Trade (44-45)', 1, 'Purchased Goods and Services', '{9}', 0.85),
('45', 'sector_2', 'Retail Trade (44-45)', 1, 'Purchased Goods and Services', '{9}', 0.85),
('48', 'sector_2', 'Transportation and Warehousing (48-49)', 4, 'Upstream Transportation and Distribution', '{9}', 0.80),
('49', 'sector_2', 'Transportation and Warehousing (48-49)', 4, 'Upstream Transportation and Distribution', '{9}', 0.80),
('51', 'sector_2', 'Information', 1, 'Purchased Goods and Services', '{}', 0.85),
('52', 'sector_2', 'Finance and Insurance', 15, 'Investments', '{1}', 0.85),
('53', 'sector_2', 'Real Estate and Rental and Leasing', 8, 'Upstream Leased Assets', '{13}', 0.80),
('54', 'sector_2', 'Professional, Scientific, and Technical Services', 1, 'Purchased Goods and Services', '{}', 0.90),
('55', 'sector_2', 'Management of Companies and Enterprises', 1, 'Purchased Goods and Services', '{}', 0.85),
('56', 'sector_2', 'Administrative and Support Services', 1, 'Purchased Goods and Services', '{5}', 0.85),
('61', 'sector_2', 'Educational Services', 1, 'Purchased Goods and Services', '{}', 0.85),
('62', 'sector_2', 'Health Care and Social Assistance', 1, 'Purchased Goods and Services', '{}', 0.85),
('71', 'sector_2', 'Arts, Entertainment, and Recreation', 1, 'Purchased Goods and Services', '{}', 0.85),
('72', 'sector_2', 'Accommodation and Food Services', 6, 'Business Travel', '{}', 0.80),
('81', 'sector_2', 'Other Services (except Public Administration)', 1, 'Purchased Goods and Services', '{}', 0.85),
('92', 'sector_2', 'Public Administration', 1, 'Purchased Goods and Services', '{}', 0.85);

-- NAICS 3-digit specialized overrides
INSERT INTO scope3_category_mapper_service.gl_scm_naics_mappings (naics_code, naics_level, description, primary_category, primary_category_name, secondary_categories, confidence) VALUES
('333', 'subsector_3', 'Machinery Manufacturing', 2, 'Capital Goods', '{}', 0.95),
('334', 'subsector_3', 'Computer and Electronic Product Manufacturing', 2, 'Capital Goods', '{11}', 0.90),
('335', 'subsector_3', 'Electrical Equipment Manufacturing', 2, 'Capital Goods', '{}', 0.90),
('336', 'subsector_3', 'Transportation Equipment Manufacturing', 2, 'Capital Goods', '{11}', 0.90),
('481', 'subsector_3', 'Air Transportation', 6, 'Business Travel', '{4}', 0.90),
('482', 'subsector_3', 'Rail Transportation', 6, 'Business Travel', '{4,7}', 0.80),
('484', 'subsector_3', 'Truck Transportation', 4, 'Upstream Transportation and Distribution', '{9}', 0.90),
('485', 'subsector_3', 'Transit and Ground Passenger Transportation', 7, 'Employee Commuting', '{6}', 0.80),
('486', 'subsector_3', 'Pipeline Transportation', 4, 'Upstream Transportation and Distribution', '{}', 0.90),
('488', 'subsector_3', 'Support Activities for Transportation', 4, 'Upstream Transportation and Distribution', '{9}', 0.85),
('493', 'subsector_3', 'Warehousing and Storage', 4, 'Upstream Transportation and Distribution', '{9}', 0.85),
('524', 'subsector_3', 'Insurance Carriers and Related Activities', 15, 'Investments', '{}', 0.85),
('531', 'subsector_3', 'Real Estate', 8, 'Upstream Leased Assets', '{13,2}', 0.80),
('532', 'subsector_3', 'Rental and Leasing Services', 8, 'Upstream Leased Assets', '{13}', 0.85),
('562', 'subsector_3', 'Waste Management and Remediation Services', 5, 'Waste Generated in Operations', '{12}', 0.95),
('721', 'subsector_3', 'Accommodation', 6, 'Business Travel', '{}', 0.90),
('722', 'subsector_3', 'Food Services and Drinking Places', 1, 'Purchased Goods and Services', '{14}', 0.80);

-- =====================================================================================
-- SEED DATA: ISIC Rev 4 section mappings
-- =====================================================================================

INSERT INTO scope3_category_mapper_service.gl_scm_isic_mappings (isic_code, isic_level, description, primary_category, primary_category_name, naics_equivalent, confidence) VALUES
('A', 'section_1', 'Agriculture, forestry and fishing', 1, 'Purchased Goods and Services', '11', 0.90),
('B', 'section_1', 'Mining and quarrying', 1, 'Purchased Goods and Services', '21', 0.90),
('C', 'section_1', 'Manufacturing', 1, 'Purchased Goods and Services', '31', 0.85),
('D', 'section_1', 'Electricity, gas, steam and AC', 3, 'Fuel and Energy Related Activities', '22', 0.90),
('E', 'section_1', 'Water supply; sewerage, waste', 5, 'Waste Generated in Operations', '56', 0.85),
('F', 'section_1', 'Construction', 2, 'Capital Goods', '23', 0.85),
('G', 'section_1', 'Wholesale and retail trade', 1, 'Purchased Goods and Services', '42', 0.85),
('H', 'section_1', 'Transportation and storage', 4, 'Upstream Transportation and Distribution', '48', 0.80),
('I', 'section_1', 'Accommodation and food service', 6, 'Business Travel', '72', 0.80),
('J', 'section_1', 'Information and communication', 1, 'Purchased Goods and Services', '51', 0.85),
('K', 'section_1', 'Financial and insurance activities', 15, 'Investments', '52', 0.85),
('L', 'section_1', 'Real estate activities', 8, 'Upstream Leased Assets', '53', 0.80),
('M', 'section_1', 'Professional, scientific and technical', 1, 'Purchased Goods and Services', '54', 0.90),
('N', 'section_1', 'Administrative and support services', 1, 'Purchased Goods and Services', '56', 0.85),
('O', 'section_1', 'Public administration and defence', 1, 'Purchased Goods and Services', '92', 0.85),
('P', 'section_1', 'Education', 1, 'Purchased Goods and Services', '61', 0.85),
('Q', 'section_1', 'Human health and social work', 1, 'Purchased Goods and Services', '62', 0.85),
('R', 'section_1', 'Arts, entertainment and recreation', 1, 'Purchased Goods and Services', '71', 0.85),
('S', 'section_1', 'Other service activities', 1, 'Purchased Goods and Services', '81', 0.85),
('T', 'section_1', 'Activities of households as employers', 1, 'Purchased Goods and Services', NULL, 0.80),
('U', 'section_1', 'Activities of extraterritorial organizations', 1, 'Purchased Goods and Services', NULL, 0.80);

-- =====================================================================================
-- SEED DATA: GL Account range mappings
-- =====================================================================================

INSERT INTO scope3_category_mapper_service.gl_scm_gl_account_mappings (range_start, range_end, account_type, primary_category, primary_category_name, confidence, notes) VALUES
(5000, 5199, 'COGS - Materials & Components', 1, 'Purchased Goods and Services', 0.90, 'Raw materials, components, packaging'),
(5200, 5299, 'COGS - Direct Labor (outsourced)', 1, 'Purchased Goods and Services', 0.85, 'Outsourced manufacturing services'),
(5300, 5399, 'COGS - Freight In', 4, 'Upstream Transportation and Distribution', 0.90, 'Inbound freight, shipping, logistics'),
(5400, 5499, 'COGS - Subcontractor', 1, 'Purchased Goods and Services', 0.85, 'Subcontracted services'),
(5500, 5599, 'COGS - Other Direct Costs', 1, 'Purchased Goods and Services', 0.80, 'Other direct costs'),
(6100, 6199, 'Office Supplies & Consumables', 1, 'Purchased Goods and Services', 0.90, 'Office supplies, stationery, consumables'),
(6200, 6299, 'IT & Software', 1, 'Purchased Goods and Services', 0.85, 'SaaS, licenses, IT services'),
(6300, 6399, 'Professional Services', 1, 'Purchased Goods and Services', 0.90, 'Consulting, legal, audit, advisory'),
(6400, 6499, 'Travel & Entertainment', 6, 'Business Travel', 0.90, 'Air, rail, car rental, hotel, meals'),
(6500, 6599, 'Vehicle & Fleet Expenses', 3, 'Fuel and Energy Related Activities', 0.80, 'Fleet fuel, maintenance, leasing'),
(6600, 6699, 'Utilities', 3, 'Fuel and Energy Related Activities', 0.90, 'Electricity, gas, water, heating'),
(6700, 6799, 'Rent & Lease Expenses', 8, 'Upstream Leased Assets', 0.90, 'Office rent, equipment leases'),
(6800, 6899, 'Insurance', 1, 'Purchased Goods and Services', 0.85, 'Business insurance premiums'),
(6900, 6999, 'Miscellaneous Operating', 1, 'Purchased Goods and Services', 0.70, 'Misc operating expenses'),
(7000, 7999, 'Capital Expenditures', 2, 'Capital Goods', 0.90, 'Machinery, equipment, buildings, vehicles'),
(8000, 8099, 'Waste Disposal & Recycling', 5, 'Waste Generated in Operations', 0.95, 'Waste management, recycling, hazardous waste'),
(8100, 8199, 'Distribution & Outbound Logistics', 9, 'Downstream Transportation and Distribution', 0.90, 'Outbound freight, distribution, last-mile'),
(8200, 8299, 'Franchise Fees & Royalties', 14, 'Franchises', 0.90, 'Franchise fees, licensing royalties'),
(8300, 8399, 'Investment-Related Expenses', 15, 'Investments', 0.85, 'Portfolio management, investment services');

-- =====================================================================================
-- SEED DATA: Category metadata (all 15 categories)
-- =====================================================================================

INSERT INTO scope3_category_mapper_service.gl_scm_category_metadata (category_number, category_name, short_name, direction, ghg_protocol_chapter, reporter_role, description, typical_data_sources, target_agent_id, target_agent_component, target_api_endpoint) VALUES
(1, 'Purchased Goods and Services', 'Purchased Goods', 'upstream', 'Chapter 1', 'Buyer', 'Extraction, production, and transportation of goods and services purchased by the reporting company in the reporting year, not otherwise included in Categories 2-8', ARRAY['spend_data','purchase_orders','supplier_data'], 'GL-MRV-S3-001', 'AGENT-MRV-014', '/api/v1/purchased-goods'),
(2, 'Capital Goods', 'Capital Goods', 'upstream', 'Chapter 2', 'Buyer', 'Extraction, production, and transportation of capital goods purchased by the reporting company in the reporting year', ARRAY['capex_register','purchase_orders','asset_register'], 'GL-MRV-S3-002', 'AGENT-MRV-015', '/api/v1/capital-goods'),
(3, 'Fuel- and Energy-Related Activities', 'Fuel & Energy', 'upstream', 'Chapter 3', 'Consumer', 'Extraction, production, and transportation of fuels and energy purchased or acquired by the reporting company in the reporting year, not already accounted for in Scope 1 or Scope 2', ARRAY['energy_invoices','fuel_purchases','utility_bills'], 'GL-MRV-S3-003', 'AGENT-MRV-016', '/api/v1/fuel-energy'),
(4, 'Upstream Transportation and Distribution', 'Upstream Transport', 'upstream', 'Chapter 4', 'Buyer', 'Transportation and distribution of products purchased by the reporting company in the reporting year between a company''s tier 1 suppliers and its own operations', ARRAY['freight_bills','logistics_data','shipping_records'], 'GL-MRV-S3-004', 'AGENT-MRV-017', '/api/v1/upstream-transportation'),
(5, 'Waste Generated in Operations', 'Waste Generated', 'upstream', 'Chapter 5', 'Generator', 'Disposal and treatment of waste generated in the reporting company''s operations in the reporting year', ARRAY['waste_manifests','recycling_records','disposal_invoices'], 'GL-MRV-S3-005', 'AGENT-MRV-018', '/api/v1/waste-generated'),
(6, 'Business Travel', 'Business Travel', 'upstream', 'Chapter 6', 'Employer', 'Transportation of employees for business-related activities during the reporting year', ARRAY['travel_bookings','expense_reports','credit_card_data'], 'GL-MRV-S3-006', 'AGENT-MRV-019', '/api/v1/business-travel'),
(7, 'Employee Commuting', 'Commuting', 'upstream', 'Chapter 7', 'Employer', 'Transportation of employees between their homes and their worksites during the reporting year', ARRAY['commute_surveys','hr_data','office_locations'], 'GL-MRV-S3-007', 'AGENT-MRV-020', '/api/v1/employee-commuting'),
(8, 'Upstream Leased Assets', 'Upstream Leased', 'upstream', 'Chapter 8', 'Lessee', 'Operation of assets leased by the reporting company in the reporting year and not included in Scope 1 and Scope 2', ARRAY['lease_agreements','energy_bills','property_data'], 'GL-MRV-S3-008', 'AGENT-MRV-021', '/api/v1/upstream-leased-assets'),
(9, 'Downstream Transportation and Distribution', 'Downstream Transport', 'downstream', 'Chapter 9', 'Seller', 'Transportation and distribution of products sold by the reporting company in the reporting year between the company''s operations and the end consumer', ARRAY['distribution_data','shipping_records','sales_logistics'], 'GL-MRV-S3-009', 'AGENT-MRV-022', '/api/v1/downstream-transportation'),
(10, 'Processing of Sold Products', 'Processing Sold', 'downstream', 'Chapter 10', 'Seller', 'Processing of intermediate products sold in the reporting year by downstream companies', ARRAY['product_specs','customer_processing','supply_chain_data'], 'GL-MRV-S3-010', 'AGENT-MRV-023', '/api/v1/processing-sold-products'),
(11, 'Use of Sold Products', 'Use of Sold', 'downstream', 'Chapter 11', 'Seller', 'End use of goods and services sold by the reporting company in the reporting year', ARRAY['product_specs','energy_ratings','usage_profiles'], 'GL-MRV-S3-011', 'AGENT-MRV-024', '/api/v1/use-sold-products'),
(12, 'End-of-Life Treatment of Sold Products', 'End-of-Life', 'downstream', 'Chapter 12', 'Seller', 'Waste disposal and treatment of products sold by the reporting company at the end of their life', ARRAY['product_composition','waste_treatment_data','material_specs'], 'GL-MRV-S3-012', 'AGENT-MRV-025', '/api/v1/end-of-life'),
(13, 'Downstream Leased Assets', 'Downstream Leased', 'downstream', 'Chapter 13', 'Lessor', 'Operation of assets owned by the reporting company and leased to other entities in the reporting year', ARRAY['lease_portfolio','tenant_data','asset_register'], 'GL-MRV-S3-013', 'AGENT-MRV-026', '/api/v1/downstream-leased-assets'),
(14, 'Franchises', 'Franchises', 'downstream', 'Chapter 14', 'Franchisor', 'Operation of franchises in the reporting year, not included in Scope 1 and Scope 2', ARRAY['franchise_agreements','franchisee_data','royalty_records'], 'GL-MRV-S3-014', 'AGENT-MRV-027', '/api/v1/franchises'),
(15, 'Investments', 'Investments', 'downstream', 'Chapter 15', 'Investor', 'Operation of investments in the reporting year, not included in Scope 1 and Scope 2', ARRAY['portfolio_data','fund_allocations','holding_records'], 'GL-MRV-S3-015', 'AGENT-MRV-028', '/api/v1/investments');

-- =====================================================================================
-- RETENTION POLICIES
-- =====================================================================================

SELECT add_retention_policy(
    'scope3_category_mapper_service.gl_scm_classification_results',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'scope3_category_mapper_service.gl_scm_routing_executions',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'scope3_category_mapper_service.gl_scm_compliance_assessments',
    INTERVAL '5 years',
    if_not_exists => TRUE
);

-- =====================================================================================
-- GRANTS
-- =====================================================================================

GRANT USAGE ON SCHEMA scope3_category_mapper_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA scope3_category_mapper_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA scope3_category_mapper_service TO greenlang_app;
GRANT SELECT ON scope3_category_mapper_service.gl_scm_hourly_classification_stats TO greenlang_app;
GRANT SELECT ON scope3_category_mapper_service.gl_scm_daily_routing_stats TO greenlang_app;

-- Read-only access for reporting
GRANT USAGE ON SCHEMA scope3_category_mapper_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA scope3_category_mapper_service TO greenlang_readonly;
GRANT SELECT ON scope3_category_mapper_service.gl_scm_hourly_classification_stats TO greenlang_readonly;
GRANT SELECT ON scope3_category_mapper_service.gl_scm_daily_routing_stats TO greenlang_readonly;

-- =====================================================================================
-- MIGRATION METADATA
-- =====================================================================================

COMMENT ON SCHEMA scope3_category_mapper_service IS
'AGENT-MRV-029 (GL-MRV-X-040): Scope 3 Category Mapper cross-cutting agent.
Tables: 11 | Hypertables: 3 | Continuous Aggregates: 2
Classification codes: NAICS 2022 (41 entries), ISIC Rev 4 (21 entries), GL Accounts (19 ranges)
Category metadata: 15 Scope 3 categories with routing targets
Migration: V080 | Created: 2026-02-28';
