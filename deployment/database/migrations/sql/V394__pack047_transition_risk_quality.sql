-- =============================================================================
-- V394: PACK-047 GHG Emissions Benchmark Pack - Transition Risk & Data Quality
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for transition risk scoring and data quality assessment.
-- Transition risk evaluates climate-related financial risk through a
-- composite score aggregating carbon budget, asset stranding, regulatory
-- exposure, competitive positioning, and financial impact sub-scores.
-- Data quality scoring assesses individual benchmark data points across
-- five dimensions (temporal, geographic, technological, completeness,
-- reliability) with PCAF-aligned composite scoring and confidence intervals.
--
-- Tables (2):
--   1. ghg_benchmark.gl_bm_transition_risk
--   2. ghg_benchmark.gl_bm_data_quality_scores
--
-- Also includes: indexes, RLS, comments.
-- Previous: V393__pack047_portfolio.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_transition_risk
-- =============================================================================
-- Transition risk scoring per entity. The composite score aggregates five
-- sub-dimensions: carbon_budget_score (exposure to carbon budget overshoot),
-- stranding_score (risk of stranded assets), regulatory_score (exposure to
-- carbon pricing and regulation), competitive_score (relative competitive
-- position), and financial_score (impact on financial metrics). Each sub-
-- score uses a 0-100 scale. Additional risk indicators include stranding
-- year, carbon price exposure, overshoot probability, and risk trajectory.

CREATE TABLE ghg_benchmark.gl_bm_transition_risk (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    entity_name                 VARCHAR(255)    NOT NULL,
    entity_identifier           VARCHAR(100),
    composite_score             NUMERIC(5,2)    NOT NULL,
    carbon_budget_score         NUMERIC(5,2),
    stranding_score             NUMERIC(5,2),
    regulatory_score            NUMERIC(5,2),
    competitive_score           NUMERIC(5,2),
    financial_score             NUMERIC(5,2),
    stranding_year              INTEGER,
    carbon_price_exposure       NUMERIC(20,2),
    overshoot_probability       NUMERIC(5,4),
    risk_trajectory             TEXT            NOT NULL DEFAULT 'STABLE',
    risk_level                  VARCHAR(20),
    scenario_reference          VARCHAR(100),
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_trs_composite CHECK (
        composite_score >= 0 AND composite_score <= 100
    ),
    CONSTRAINT chk_p047_trs_carbon CHECK (
        carbon_budget_score IS NULL OR (carbon_budget_score >= 0 AND carbon_budget_score <= 100)
    ),
    CONSTRAINT chk_p047_trs_stranding CHECK (
        stranding_score IS NULL OR (stranding_score >= 0 AND stranding_score <= 100)
    ),
    CONSTRAINT chk_p047_trs_regulatory CHECK (
        regulatory_score IS NULL OR (regulatory_score >= 0 AND regulatory_score <= 100)
    ),
    CONSTRAINT chk_p047_trs_competitive CHECK (
        competitive_score IS NULL OR (competitive_score >= 0 AND competitive_score <= 100)
    ),
    CONSTRAINT chk_p047_trs_financial CHECK (
        financial_score IS NULL OR (financial_score >= 0 AND financial_score <= 100)
    ),
    CONSTRAINT chk_p047_trs_stranding_yr CHECK (
        stranding_year IS NULL OR (stranding_year >= 2000 AND stranding_year <= 2200)
    ),
    CONSTRAINT chk_p047_trs_carbon_price CHECK (
        carbon_price_exposure IS NULL OR carbon_price_exposure >= 0
    ),
    CONSTRAINT chk_p047_trs_overshoot CHECK (
        overshoot_probability IS NULL OR (overshoot_probability >= 0 AND overshoot_probability <= 1)
    ),
    CONSTRAINT chk_p047_trs_trajectory CHECK (
        risk_trajectory IN ('INCREASING', 'DECREASING', 'STABLE')
    ),
    CONSTRAINT chk_p047_trs_level CHECK (
        risk_level IS NULL OR risk_level IN (
            'VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH', 'CRITICAL'
        )
    ),
    CONSTRAINT chk_p047_trs_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_trs_tenant           ON ghg_benchmark.gl_bm_transition_risk(tenant_id);
CREATE INDEX idx_p047_trs_config           ON ghg_benchmark.gl_bm_transition_risk(config_id);
CREATE INDEX idx_p047_trs_entity           ON ghg_benchmark.gl_bm_transition_risk(entity_name);
CREATE INDEX idx_p047_trs_entity_id        ON ghg_benchmark.gl_bm_transition_risk(entity_identifier);
CREATE INDEX idx_p047_trs_composite        ON ghg_benchmark.gl_bm_transition_risk(composite_score);
CREATE INDEX idx_p047_trs_trajectory       ON ghg_benchmark.gl_bm_transition_risk(risk_trajectory);
CREATE INDEX idx_p047_trs_level            ON ghg_benchmark.gl_bm_transition_risk(risk_level);
CREATE INDEX idx_p047_trs_calculated       ON ghg_benchmark.gl_bm_transition_risk(calculated_at DESC);
CREATE INDEX idx_p047_trs_created          ON ghg_benchmark.gl_bm_transition_risk(created_at DESC);
CREATE INDEX idx_p047_trs_provenance       ON ghg_benchmark.gl_bm_transition_risk(provenance_hash);

-- Composite: config + entity for entity-level lookup
CREATE INDEX idx_p047_trs_config_entity    ON ghg_benchmark.gl_bm_transition_risk(config_id, entity_name);

-- Composite: tenant + level for risk-filtered queries
CREATE INDEX idx_p047_trs_tenant_level     ON ghg_benchmark.gl_bm_transition_risk(tenant_id, risk_level);

-- Composite: risk trajectory + composite for trend analysis
CREATE INDEX idx_p047_trs_traj_score       ON ghg_benchmark.gl_bm_transition_risk(risk_trajectory, composite_score);

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_data_quality_scores
-- =============================================================================
-- Data quality assessment per benchmark data point. Each assessment scores
-- a data point across five dimensions: temporal (currency of data),
-- geographic (relevance to entity location), technological (relevance to
-- entity operations), completeness (fraction of required data available),
-- and reliability (trustworthiness of source). Scores are combined into a
-- composite using weighted aggregation. PCAF-aligned score (1-5) is derived
-- from the composite. Confidence intervals reflect the uncertainty range
-- introduced by data quality limitations.

CREATE TABLE ghg_benchmark.gl_bm_data_quality_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    entity_name                 VARCHAR(255)    NOT NULL,
    entity_identifier           VARCHAR(100),
    data_point_type             VARCHAR(100)    NOT NULL,
    temporal_score              NUMERIC(3,1)    NOT NULL,
    geographic_score            NUMERIC(3,1)    NOT NULL,
    technological_score         NUMERIC(3,1)    NOT NULL,
    completeness_score          NUMERIC(3,1)    NOT NULL,
    reliability_score           NUMERIC(3,1)    NOT NULL,
    composite_score             NUMERIC(3,1)    NOT NULL,
    pcaf_score                  INTEGER,
    confidence_lower            NUMERIC(20,10),
    confidence_upper            NUMERIC(20,10),
    source_hierarchy            TEXT,
    improvement_recommendation  TEXT,
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_dqs_temporal CHECK (
        temporal_score >= 1 AND temporal_score <= 5
    ),
    CONSTRAINT chk_p047_dqs_geographic CHECK (
        geographic_score >= 1 AND geographic_score <= 5
    ),
    CONSTRAINT chk_p047_dqs_technological CHECK (
        technological_score >= 1 AND technological_score <= 5
    ),
    CONSTRAINT chk_p047_dqs_completeness CHECK (
        completeness_score >= 1 AND completeness_score <= 5
    ),
    CONSTRAINT chk_p047_dqs_reliability CHECK (
        reliability_score >= 1 AND reliability_score <= 5
    ),
    CONSTRAINT chk_p047_dqs_composite CHECK (
        composite_score >= 1 AND composite_score <= 5
    ),
    CONSTRAINT chk_p047_dqs_pcaf CHECK (
        pcaf_score IS NULL OR (pcaf_score >= 1 AND pcaf_score <= 5)
    ),
    CONSTRAINT chk_p047_dqs_bounds CHECK (
        (confidence_lower IS NULL AND confidence_upper IS NULL) OR
        (confidence_lower IS NOT NULL AND confidence_upper IS NOT NULL AND confidence_upper >= confidence_lower)
    ),
    CONSTRAINT chk_p047_dqs_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_dqs_tenant           ON ghg_benchmark.gl_bm_data_quality_scores(tenant_id);
CREATE INDEX idx_p047_dqs_config           ON ghg_benchmark.gl_bm_data_quality_scores(config_id);
CREATE INDEX idx_p047_dqs_entity           ON ghg_benchmark.gl_bm_data_quality_scores(entity_name);
CREATE INDEX idx_p047_dqs_entity_id        ON ghg_benchmark.gl_bm_data_quality_scores(entity_identifier);
CREATE INDEX idx_p047_dqs_data_point       ON ghg_benchmark.gl_bm_data_quality_scores(data_point_type);
CREATE INDEX idx_p047_dqs_composite        ON ghg_benchmark.gl_bm_data_quality_scores(composite_score);
CREATE INDEX idx_p047_dqs_pcaf             ON ghg_benchmark.gl_bm_data_quality_scores(pcaf_score);
CREATE INDEX idx_p047_dqs_assessed         ON ghg_benchmark.gl_bm_data_quality_scores(assessed_at DESC);
CREATE INDEX idx_p047_dqs_created          ON ghg_benchmark.gl_bm_data_quality_scores(created_at DESC);
CREATE INDEX idx_p047_dqs_provenance       ON ghg_benchmark.gl_bm_data_quality_scores(provenance_hash);

-- Composite: config + entity for entity-level lookup
CREATE INDEX idx_p047_dqs_config_entity    ON ghg_benchmark.gl_bm_data_quality_scores(config_id, entity_name);

-- Composite: entity + data point for point-level history
CREATE INDEX idx_p047_dqs_entity_point     ON ghg_benchmark.gl_bm_data_quality_scores(entity_name, data_point_type);

-- Composite: composite + pcaf for quality-filtered queries
CREATE INDEX idx_p047_dqs_comp_pcaf        ON ghg_benchmark.gl_bm_data_quality_scores(composite_score, pcaf_score);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_transition_risk ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_data_quality_scores ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_trs_tenant_isolation
    ON ghg_benchmark.gl_bm_transition_risk
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_trs_service_bypass
    ON ghg_benchmark.gl_bm_transition_risk
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_dqs_tenant_isolation
    ON ghg_benchmark.gl_bm_data_quality_scores
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_dqs_service_bypass
    ON ghg_benchmark.gl_bm_data_quality_scores
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_transition_risk TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_data_quality_scores TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_transition_risk IS
    'Transition risk scoring with composite score from carbon budget, stranding, regulatory, competitive, and financial sub-dimensions.';
COMMENT ON TABLE ghg_benchmark.gl_bm_data_quality_scores IS
    'Data quality assessment across temporal, geographic, technological, completeness, and reliability dimensions with PCAF-aligned scoring.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.composite_score IS 'Overall transition risk score (0-100). Higher = greater risk exposure. Weighted aggregation of five sub-scores.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.carbon_budget_score IS 'Risk from carbon budget overshoot (0-100). Based on cumulative emissions vs allocated budget.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.stranding_score IS 'Asset stranding risk (0-100). Based on asset economic life vs pathway timeline.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.regulatory_score IS 'Regulatory exposure risk (0-100). Based on carbon pricing, ETS exposure, and policy trajectory.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.competitive_score IS 'Competitive positioning risk (0-100). Based on intensity rank, improvement rate, and peer gap.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.financial_score IS 'Financial impact risk (0-100). Based on carbon cost as percentage of revenue/EBITDA.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.stranding_year IS 'Projected year when assets become stranded (uneconomic) under the reference scenario.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.carbon_price_exposure IS 'Annual carbon cost exposure in EUR at the scenario carbon price.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.risk_trajectory IS 'Direction of risk over time: INCREASING (worsening), DECREASING (improving), STABLE.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_transition_risk.risk_level IS 'Qualitative risk level: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH, CRITICAL.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_quality_scores.data_point_type IS 'Type of data point assessed: scope1_emissions, scope2_emissions, scope3_emissions, revenue, production_volume, etc.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_quality_scores.temporal_score IS 'Data currency score (1-5): 1=current year, 2=prior year, 3=2-3 years old, 4=4-5 years, 5=>5 years.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_quality_scores.geographic_score IS 'Geographic relevance (1-5): 1=exact match, 2=sub-national, 3=national, 4=regional, 5=global average.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_quality_scores.technological_score IS 'Technological relevance (1-5): 1=process-specific, 2=technology-specific, 3=industry average, 4=proxy, 5=generic.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_quality_scores.completeness_score IS 'Data completeness (1-5): 1=>95%, 2=80-95%, 3=60-80%, 4=40-60%, 5=<40%.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_quality_scores.reliability_score IS 'Source reliability (1-5): 1=third-party verified, 2=company reported, 3=CDP/disclosure, 4=estimated, 5=proxy/modelled.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_quality_scores.source_hierarchy IS 'Data source hierarchy applied: VERIFIED > REPORTED > DISCLOSED > ESTIMATED > PROXY.';
