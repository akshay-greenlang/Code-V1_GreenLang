-- ============================================================================
-- V107: AGENT-EUDR-019 Corruption Index Monitor Agent
-- ============================================================================
-- Creates tables for CPI score time series, WGI governance indicators,
-- sector-specific bribery risk assessments, institutional quality scoring,
-- trend analyses, deforestation-corruption correlations, alert management,
-- compliance impact assessments, country profiles, sector risk scores,
-- additional governance indicators, and comprehensive audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_cim_cpi_scores (on synthetic timestamptz from year),
--              gl_eudr_cim_wgi_indicators (on indicator_date),
--              gl_eudr_cim_audit_log (chunk 30d)
-- Continuous Aggregates: 2 (quarterly_cpi_summary + annual_wgi_summary)
-- Retention Policies: 3 (10 years for CPI scores, 10 years for WGI,
--                        5 years for audit logs)
-- Indexes: ~140
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V107: Creating AGENT-EUDR-019 Corruption Index Monitor tables...';

-- ============================================================================
-- 1. gl_eudr_cim_cpi_scores — CPI scores time series (hypertable)
-- ============================================================================
RAISE NOTICE 'V107 [1/12]: Creating gl_eudr_cim_cpi_scores (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_cpi_scores (
    id                          UUID            DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    year                        INTEGER         NOT NULL CHECK (year >= 1995 AND year <= 2100),
        -- CPI publication year
    score                       DECIMAL(5,2)    NOT NULL CHECK (score >= 0 AND score <= 100),
        -- Transparency International CPI score (0 = highly corrupt, 100 = very clean)
    rank                        INTEGER         CHECK (rank >= 1),
        -- Global ranking position
    percentile                  DECIMAL(5,2)    CHECK (percentile >= 0 AND percentile <= 100),
        -- Percentile rank among all countries
    region                      VARCHAR(50),
        -- 'Sub-Saharan Africa', 'Asia-Pacific', 'Americas', 'MENA',
        -- 'Eastern Europe & Central Asia', 'Western Europe & EU'
    data_source                 VARCHAR(50)     DEFAULT 'transparency_international',
        -- 'transparency_international', 'world_bank', 'afrobarometer', 'latinobarometro'
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for data integrity verification
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    -- Synthetic timestamp derived from year for hypertable partitioning
    score_date                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, score_date)
);

SELECT create_hypertable(
    'gl_eudr_cim_cpi_scores',
    'score_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_country_code ON gl_eudr_cim_cpi_scores (country_code, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_year ON gl_eudr_cim_cpi_scores (year, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_score ON gl_eudr_cim_cpi_scores (score, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_rank ON gl_eudr_cim_cpi_scores (rank, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_percentile ON gl_eudr_cim_cpi_scores (percentile, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_region ON gl_eudr_cim_cpi_scores (region, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_data_source ON gl_eudr_cim_cpi_scores (data_source, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_provenance ON gl_eudr_cim_cpi_scores (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_tenant ON gl_eudr_cim_cpi_scores (tenant_id, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_created ON gl_eudr_cim_cpi_scores (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_updated ON gl_eudr_cim_cpi_scores (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_country_year ON gl_eudr_cim_cpi_scores (country_code, year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_region_year ON gl_eudr_cim_cpi_scores (region, year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cpi_metadata ON gl_eudr_cim_cpi_scores USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_cpi_scores IS 'Transparency International Corruption Perceptions Index scores time series by country and year';
COMMENT ON COLUMN gl_eudr_cim_cpi_scores.score IS 'CPI score: 0 = highly corrupt, 100 = very clean';
COMMENT ON COLUMN gl_eudr_cim_cpi_scores.score_date IS 'Synthetic timestamp derived from year for TimescaleDB hypertable partitioning';


-- ============================================================================
-- 2. gl_eudr_cim_wgi_indicators — WGI 6 dimensions (hypertable)
-- ============================================================================
RAISE NOTICE 'V107 [2/12]: Creating gl_eudr_cim_wgi_indicators (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_wgi_indicators (
    id                          UUID            DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    year                        INTEGER         NOT NULL CHECK (year >= 1996 AND year <= 2100),
        -- WGI assessment year
    dimension                   VARCHAR(30)     NOT NULL,
        -- 'voice_accountability', 'political_stability', 'government_effectiveness',
        -- 'regulatory_quality', 'rule_of_law', 'control_of_corruption'
    estimate                    DECIMAL(6,3)    NOT NULL CHECK (estimate >= -2.5 AND estimate <= 2.5),
        -- WGI governance estimate (standard normal distribution, typically -2.5 to 2.5)
    std_error                   DECIMAL(6,3)    CHECK (std_error >= 0),
        -- Standard error of the estimate
    percentile_rank             DECIMAL(5,2)    CHECK (percentile_rank >= 0 AND percentile_rank <= 100),
        -- Percentile rank among all countries (0-100)
    governance_score            DECIMAL(5,2)    CHECK (governance_score >= 0 AND governance_score <= 100),
        -- Normalized governance score (0-100) for cross-index comparison
    data_source                 VARCHAR(50)     DEFAULT 'world_bank',
        -- 'world_bank', 'kaufmann_kraay', 'aggregated'
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for data integrity verification
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    indicator_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp for hypertable partitioning
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, indicator_date)
);

SELECT create_hypertable(
    'gl_eudr_cim_wgi_indicators',
    'indicator_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_country_code ON gl_eudr_cim_wgi_indicators (country_code, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_year ON gl_eudr_cim_wgi_indicators (year, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_dimension ON gl_eudr_cim_wgi_indicators (dimension, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_estimate ON gl_eudr_cim_wgi_indicators (estimate, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_percentile ON gl_eudr_cim_wgi_indicators (percentile_rank, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_gov_score ON gl_eudr_cim_wgi_indicators (governance_score, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_data_source ON gl_eudr_cim_wgi_indicators (data_source, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_provenance ON gl_eudr_cim_wgi_indicators (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_tenant ON gl_eudr_cim_wgi_indicators (tenant_id, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_created ON gl_eudr_cim_wgi_indicators (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_updated ON gl_eudr_cim_wgi_indicators (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_country_year ON gl_eudr_cim_wgi_indicators (country_code, year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_country_dim ON gl_eudr_cim_wgi_indicators (country_code, dimension, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_dim_year ON gl_eudr_cim_wgi_indicators (dimension, year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_wgi_metadata ON gl_eudr_cim_wgi_indicators USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_wgi_indicators IS 'World Bank Worldwide Governance Indicators across 6 dimensions with estimate and percentile data';
COMMENT ON COLUMN gl_eudr_cim_wgi_indicators.estimate IS 'WGI governance estimate on standard normal scale, typically -2.5 to 2.5';
COMMENT ON COLUMN gl_eudr_cim_wgi_indicators.dimension IS 'WGI dimension: voice_accountability, political_stability, government_effectiveness, regulatory_quality, rule_of_law, control_of_corruption';


-- ============================================================================
-- 3. gl_eudr_cim_bribery_risk_assessments — Sector-specific bribery risk
-- ============================================================================
RAISE NOTICE 'V107 [3/12]: Creating gl_eudr_cim_bribery_risk_assessments...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_bribery_risk_assessments (
    assessment_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    sector                      VARCHAR(30)     NOT NULL,
        -- 'agriculture', 'forestry', 'livestock', 'mining', 'oil_palm',
        -- 'cocoa', 'coffee', 'rubber', 'soya', 'timber', 'customs',
        -- 'land_registry', 'environmental_permits', 'export_licensing'
    risk_score                  DECIMAL(5,2)    NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
        -- Sector-specific bribery risk (0 = no risk, 100 = extreme risk)
    contributing_factors        JSONB,
        -- { "regulatory_capture": 0.35, "informal_payments": 0.25,
        --   "weak_oversight": 0.20, "monopolistic_access": 0.20 }
    mitigation_measures         JSONB,
        -- [{ "measure": "third_party_audit", "effectiveness": 0.75, "cost": "medium" },
        --  { "measure": "digital_payments", "effectiveness": 0.60, "cost": "low" }]
    data_quality                VARCHAR(20)     DEFAULT 'medium',
        -- 'very_high', 'high', 'medium', 'low', 'very_low'
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_country_code ON gl_eudr_cim_bribery_risk_assessments (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_sector ON gl_eudr_cim_bribery_risk_assessments (sector);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_risk_score ON gl_eudr_cim_bribery_risk_assessments (risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_data_quality ON gl_eudr_cim_bribery_risk_assessments (data_quality);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_provenance ON gl_eudr_cim_bribery_risk_assessments (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_tenant ON gl_eudr_cim_bribery_risk_assessments (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_assessed_at ON gl_eudr_cim_bribery_risk_assessments (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_created ON gl_eudr_cim_bribery_risk_assessments (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_updated ON gl_eudr_cim_bribery_risk_assessments (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_country_sector ON gl_eudr_cim_bribery_risk_assessments (country_code, sector);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_sector_risk ON gl_eudr_cim_bribery_risk_assessments (sector, risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_contrib ON gl_eudr_cim_bribery_risk_assessments USING GIN (contributing_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_mitigation ON gl_eudr_cim_bribery_risk_assessments USING GIN (mitigation_measures);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_bra_metadata ON gl_eudr_cim_bribery_risk_assessments USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_bribery_risk_assessments IS 'Sector-specific bribery risk assessments with contributing factors and mitigation measures';
COMMENT ON COLUMN gl_eudr_cim_bribery_risk_assessments.risk_score IS 'Sector bribery risk: 0 = no risk, 100 = extreme risk';
COMMENT ON COLUMN gl_eudr_cim_bribery_risk_assessments.contributing_factors IS 'Weighted factors contributing to sector bribery risk';


-- ============================================================================
-- 4. gl_eudr_cim_institutional_quality — Institutional quality scores
-- ============================================================================
RAISE NOTICE 'V107 [4/12]: Creating gl_eudr_cim_institutional_quality...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_institutional_quality (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    year                        INTEGER         NOT NULL CHECK (year >= 1990 AND year <= 2100),
    overall_score               DECIMAL(5,2)    NOT NULL CHECK (overall_score >= 0 AND overall_score <= 100),
        -- Composite institutional quality score (0-100)
    judicial_independence       DECIMAL(5,2)    CHECK (judicial_independence >= 0 AND judicial_independence <= 100),
        -- Judicial independence and impartiality score
    regulatory_enforcement      DECIMAL(5,2)    CHECK (regulatory_enforcement >= 0 AND regulatory_enforcement <= 100),
        -- Effectiveness of regulatory enforcement
    forest_governance           DECIMAL(5,2)    CHECK (forest_governance >= 0 AND forest_governance <= 100),
        -- Forest-specific governance quality
    law_enforcement             DECIMAL(5,2)    CHECK (law_enforcement >= 0 AND law_enforcement <= 100),
        -- Environmental law enforcement effectiveness
    property_rights             DECIMAL(5,2)    CHECK (property_rights >= 0 AND property_rights <= 100),
        -- Land and property rights protection
    transparency_score          DECIMAL(5,2)    CHECK (transparency_score >= 0 AND transparency_score <= 100),
        -- Government transparency and open data
    data_sources                TEXT[],
        -- e.g. ARRAY['WB_WGI', 'FLEG', 'TI_CPI', 'BTI', 'WJP_ROL']
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_country_code ON gl_eudr_cim_institutional_quality (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_year ON gl_eudr_cim_institutional_quality (year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_overall ON gl_eudr_cim_institutional_quality (overall_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_judicial ON gl_eudr_cim_institutional_quality (judicial_independence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_regulatory ON gl_eudr_cim_institutional_quality (regulatory_enforcement);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_forest_gov ON gl_eudr_cim_institutional_quality (forest_governance);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_law_enforce ON gl_eudr_cim_institutional_quality (law_enforcement);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_property ON gl_eudr_cim_institutional_quality (property_rights);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_transparency ON gl_eudr_cim_institutional_quality (transparency_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_provenance ON gl_eudr_cim_institutional_quality (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_tenant ON gl_eudr_cim_institutional_quality (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_assessment ON gl_eudr_cim_institutional_quality (assessment_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_created ON gl_eudr_cim_institutional_quality (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_updated ON gl_eudr_cim_institutional_quality (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_country_year ON gl_eudr_cim_institutional_quality (country_code, year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_iq_metadata ON gl_eudr_cim_institutional_quality USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_institutional_quality IS 'Institutional quality scores covering judicial, regulatory, forest governance, and law enforcement dimensions';
COMMENT ON COLUMN gl_eudr_cim_institutional_quality.forest_governance IS 'Forest-specific governance quality covering FLEG framework indicators';
COMMENT ON COLUMN gl_eudr_cim_institutional_quality.overall_score IS 'Composite institutional quality score (0-100, weighted average of sub-dimensions)';


-- ============================================================================
-- 5. gl_eudr_cim_trend_analyses — Trend analysis results
-- ============================================================================
RAISE NOTICE 'V107 [5/12]: Creating gl_eudr_cim_trend_analyses...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_trend_analyses (
    analysis_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    index_type                  VARCHAR(20)     NOT NULL,
        -- 'cpi', 'wgi', 'bribery', 'institutional', 'composite'
    direction                   VARCHAR(20)     NOT NULL,
        -- 'improving', 'stable', 'deteriorating', 'rapidly_deteriorating', 'volatile'
    slope                       DECIMAL(8,4)    NOT NULL,
        -- Linear regression slope (positive = improving, negative = deteriorating)
    r_squared                   DECIMAL(5,4)    CHECK (r_squared >= 0 AND r_squared <= 1),
        -- Coefficient of determination (0-1, higher = better fit)
    period_years                INTEGER         DEFAULT 5 CHECK (period_years >= 1 AND period_years <= 30),
        -- Number of years in the analysis window
    prediction                  JSONB,
        -- { "year_1": 42.5, "year_3": 40.1, "year_5": 37.8,
        --   "confidence_80_lower": 35.2, "confidence_80_upper": 49.8 }
    confidence_interval         JSONB,
        -- { "lower_95": -0.85, "upper_95": 0.12, "lower_80": -0.72, "upper_80": 0.01 }
    breakpoints                 JSONB,
        -- [{ "year": 2020, "type": "structural_break", "significance": 0.98 }]
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    analysis_date               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_country_code ON gl_eudr_cim_trend_analyses (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_index_type ON gl_eudr_cim_trend_analyses (index_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_direction ON gl_eudr_cim_trend_analyses (direction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_slope ON gl_eudr_cim_trend_analyses (slope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_r_squared ON gl_eudr_cim_trend_analyses (r_squared);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_provenance ON gl_eudr_cim_trend_analyses (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_tenant ON gl_eudr_cim_trend_analyses (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_analysis_date ON gl_eudr_cim_trend_analyses (analysis_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_created ON gl_eudr_cim_trend_analyses (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_updated ON gl_eudr_cim_trend_analyses (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_country_index ON gl_eudr_cim_trend_analyses (country_code, index_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_index_dir ON gl_eudr_cim_trend_analyses (index_type, direction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_prediction ON gl_eudr_cim_trend_analyses USING GIN (prediction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_confidence ON gl_eudr_cim_trend_analyses USING GIN (confidence_interval);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ta_metadata ON gl_eudr_cim_trend_analyses USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_trend_analyses IS 'Corruption index trend analysis results with regression statistics and predictions';
COMMENT ON COLUMN gl_eudr_cim_trend_analyses.slope IS 'Linear regression slope: positive = improving trend, negative = deteriorating';
COMMENT ON COLUMN gl_eudr_cim_trend_analyses.r_squared IS 'Coefficient of determination (0-1): higher values indicate stronger trend fit';


-- ============================================================================
-- 6. gl_eudr_cim_deforestation_correlations — Correlation analysis
-- ============================================================================
RAISE NOTICE 'V107 [6/12]: Creating gl_eudr_cim_deforestation_correlations...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_deforestation_correlations (
    correlation_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    corruption_index            VARCHAR(20)     NOT NULL,
        -- 'cpi', 'wgi_corruption', 'wgi_rule_of_law', 'bribery_index', 'composite'
    deforestation_rate          DECIMAL(8,4)    NOT NULL,
        -- Annual deforestation rate (% of forest cover lost per year)
    correlation_coefficient     DECIMAL(5,4)    NOT NULL CHECK (correlation_coefficient >= -1 AND correlation_coefficient <= 1),
        -- Pearson or Spearman correlation coefficient (-1 to 1)
    p_value                     DECIMAL(8,6)    CHECK (p_value >= 0 AND p_value <= 1),
        -- Statistical significance (lower = more significant)
    sample_size                 INTEGER         CHECK (sample_size >= 2),
        -- Number of data points in the correlation analysis
    correlation_method          VARCHAR(30)     DEFAULT 'pearson',
        -- 'pearson', 'spearman', 'kendall'
    regression_model            JSONB,
        -- { "type": "linear", "intercept": 3.45, "slope": -0.032,
        --   "r_squared": 0.68, "adj_r_squared": 0.65,
        --   "std_error": 0.005, "f_statistic": 42.3 }
    time_lag_years              INTEGER         DEFAULT 0 CHECK (time_lag_years >= 0 AND time_lag_years <= 10),
        -- Time lag between corruption change and deforestation response
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    analysis_date               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_country_code ON gl_eudr_cim_deforestation_correlations (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_corruption_index ON gl_eudr_cim_deforestation_correlations (corruption_index);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_defor_rate ON gl_eudr_cim_deforestation_correlations (deforestation_rate);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_corr_coeff ON gl_eudr_cim_deforestation_correlations (correlation_coefficient);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_p_value ON gl_eudr_cim_deforestation_correlations (p_value);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_method ON gl_eudr_cim_deforestation_correlations (correlation_method);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_provenance ON gl_eudr_cim_deforestation_correlations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_tenant ON gl_eudr_cim_deforestation_correlations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_analysis_date ON gl_eudr_cim_deforestation_correlations (analysis_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_created ON gl_eudr_cim_deforestation_correlations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_updated ON gl_eudr_cim_deforestation_correlations (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_country_index ON gl_eudr_cim_deforestation_correlations (country_code, corruption_index);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_regression ON gl_eudr_cim_deforestation_correlations USING GIN (regression_model);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_dc_metadata ON gl_eudr_cim_deforestation_correlations USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_deforestation_correlations IS 'Statistical correlation analysis between corruption indices and deforestation rates';
COMMENT ON COLUMN gl_eudr_cim_deforestation_correlations.correlation_coefficient IS 'Pearson/Spearman correlation: -1 (perfect negative) to 1 (perfect positive)';
COMMENT ON COLUMN gl_eudr_cim_deforestation_correlations.time_lag_years IS 'Years of lag between corruption index change and observable deforestation impact';


-- ============================================================================
-- 7. gl_eudr_cim_alerts — Alert records
-- ============================================================================
RAISE NOTICE 'V107 [7/12]: Creating gl_eudr_cim_alerts...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_alerts (
    alert_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    alert_type                  VARCHAR(50)     NOT NULL,
        -- 'cpi_score_drop', 'cpi_score_spike', 'wgi_deterioration',
        -- 'bribery_risk_increase', 'institutional_decline', 'trend_reversal',
        -- 'classification_change', 'correlation_anomaly', 'data_quality_issue',
        -- 'threshold_breach', 'governance_crisis'
    severity                    VARCHAR(20)     NOT NULL,
        -- 'info', 'warning', 'high', 'critical'
    description                 TEXT            NOT NULL,
        -- Human-readable alert description
    old_value                   DECIMAL(8,3),
        -- Previous value before change
    new_value                   DECIMAL(8,3),
        -- New value after change
    change_magnitude            DECIMAL(8,4),
        -- Absolute or percentage change magnitude
    change_pct                  DECIMAL(8,4),
        -- Percentage change from old to new value
    affected_index              VARCHAR(20),
        -- 'cpi', 'wgi', 'bribery', 'institutional', 'composite'
    acknowledged                BOOLEAN         DEFAULT FALSE,
    acknowledged_by             VARCHAR(100),
    acknowledged_at             TIMESTAMPTZ,
    resolved                    BOOLEAN         DEFAULT FALSE,
    resolved_at                 TIMESTAMPTZ,
    resolution_notes            TEXT,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_country ON gl_eudr_cim_alerts (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_type ON gl_eudr_cim_alerts (alert_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_severity ON gl_eudr_cim_alerts (severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_acknowledged ON gl_eudr_cim_alerts (acknowledged);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_resolved ON gl_eudr_cim_alerts (resolved);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_index ON gl_eudr_cim_alerts (affected_index);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_provenance ON gl_eudr_cim_alerts (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_tenant ON gl_eudr_cim_alerts (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_created ON gl_eudr_cim_alerts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_updated ON gl_eudr_cim_alerts (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_country_sev ON gl_eudr_cim_alerts (country_code, severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_type_sev ON gl_eudr_cim_alerts (alert_type, severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (unacknowledged) alerts
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_active ON gl_eudr_cim_alerts (severity, created_at DESC)
        WHERE acknowledged = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unresolved alerts
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_unresolved ON gl_eudr_cim_alerts (severity, created_at DESC)
        WHERE resolved = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_alert_metadata ON gl_eudr_cim_alerts USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_alerts IS 'Corruption index monitoring alerts for threshold breaches, trend reversals, and governance changes';
COMMENT ON COLUMN gl_eudr_cim_alerts.change_magnitude IS 'Absolute change magnitude triggering the alert';
COMMENT ON COLUMN gl_eudr_cim_alerts.severity IS 'Alert severity: info, warning, high, critical';


-- ============================================================================
-- 8. gl_eudr_cim_compliance_impacts — Compliance impact assessments
-- ============================================================================
RAISE NOTICE 'V107 [8/12]: Creating gl_eudr_cim_compliance_impacts...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_compliance_impacts (
    impact_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    cpi_score                   DECIMAL(5,2)    CHECK (cpi_score >= 0 AND cpi_score <= 100),
        -- Current CPI score at time of assessment
    wgi_score                   DECIMAL(6,3)    CHECK (wgi_score >= -2.5 AND wgi_score <= 2.5),
        -- Current WGI control of corruption score
    article_29_classification   VARCHAR(20)     NOT NULL,
        -- 'low_risk', 'standard_risk', 'high_risk'
        -- Per EUDR Article 29 country benchmarking
    dd_level                    VARCHAR(20)     NOT NULL,
        -- 'simplified', 'standard', 'enhanced'
        -- Due diligence level required based on risk classification
    risk_adjustment             DECIMAL(5,2)    CHECK (risk_adjustment >= -50 AND risk_adjustment <= 50),
        -- Risk score adjustment based on corruption assessment (-50 to +50)
    previous_classification     VARCHAR(20),
        -- Previous Article 29 classification for change tracking
    previous_dd_level           VARCHAR(20),
        -- Previous DD level for change tracking
    classification_rationale    TEXT,
        -- Explanation of classification decision
    effective_date              DATE,
        -- When the classification takes effect
    review_date                 DATE,
        -- Scheduled review date
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_country_code ON gl_eudr_cim_compliance_impacts (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_cpi_score ON gl_eudr_cim_compliance_impacts (cpi_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_wgi_score ON gl_eudr_cim_compliance_impacts (wgi_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_art29_class ON gl_eudr_cim_compliance_impacts (article_29_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_dd_level ON gl_eudr_cim_compliance_impacts (dd_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_risk_adj ON gl_eudr_cim_compliance_impacts (risk_adjustment);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_effective ON gl_eudr_cim_compliance_impacts (effective_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_review ON gl_eudr_cim_compliance_impacts (review_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_provenance ON gl_eudr_cim_compliance_impacts (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_tenant ON gl_eudr_cim_compliance_impacts (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_assessed_at ON gl_eudr_cim_compliance_impacts (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_created ON gl_eudr_cim_compliance_impacts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_updated ON gl_eudr_cim_compliance_impacts (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_country_class ON gl_eudr_cim_compliance_impacts (country_code, article_29_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_class_dd ON gl_eudr_cim_compliance_impacts (article_29_classification, dd_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_ci_metadata ON gl_eudr_cim_compliance_impacts USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_compliance_impacts IS 'EUDR Article 29 compliance impact assessments linking corruption indices to due diligence requirements';
COMMENT ON COLUMN gl_eudr_cim_compliance_impacts.article_29_classification IS 'EUDR Article 29 country benchmarking: low_risk, standard_risk, high_risk';
COMMENT ON COLUMN gl_eudr_cim_compliance_impacts.dd_level IS 'Required due diligence level: simplified (low risk), standard, enhanced (high risk)';


-- ============================================================================
-- 9. gl_eudr_cim_country_profiles — Comprehensive country profiles
-- ============================================================================
RAISE NOTICE 'V107 [9/12]: Creating gl_eudr_cim_country_profiles...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_country_profiles (
    country_code                VARCHAR(2)      PRIMARY KEY,
        -- ISO 3166-1 alpha-2 country code
    country_name                VARCHAR(100)    NOT NULL,
    region                      VARCHAR(50),
        -- 'Sub-Saharan Africa', 'Asia-Pacific', 'Americas', 'MENA',
        -- 'Eastern Europe & Central Asia', 'Western Europe & EU'
    sub_region                  VARCHAR(50),
        -- 'West Africa', 'Southeast Asia', 'South America', 'Central America', etc.
    latest_cpi                  DECIMAL(5,2)    CHECK (latest_cpi >= 0 AND latest_cpi <= 100),
        -- Most recent CPI score
    latest_wgi_corruption       DECIMAL(6,3)    CHECK (latest_wgi_corruption >= -2.5 AND latest_wgi_corruption <= 2.5),
        -- Most recent WGI Control of Corruption estimate
    overall_risk_level          VARCHAR(20),
        -- 'low', 'medium', 'high', 'very_high', 'critical'
    article_29_class            VARCHAR(20),
        -- 'low_risk', 'standard_risk', 'high_risk'
    eudr_commodity_relevance    JSONB,
        -- { "cattle": true, "cocoa": true, "coffee": false, "oil_palm": true,
        --   "rubber": false, "soya": false, "wood": true }
    key_commodities             TEXT[],
        -- e.g. ARRAY['cocoa', 'oil_palm', 'wood']
    cpi_trend                   VARCHAR(20),
        -- 'improving', 'stable', 'deteriorating'
    forest_area_km2             DECIMAL(12,2)   CHECK (forest_area_km2 >= 0),
    annual_deforestation_rate   DECIMAL(8,4),
        -- Percentage of forest cover lost per year
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    last_updated                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_country_name ON gl_eudr_cim_country_profiles (country_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_region ON gl_eudr_cim_country_profiles (region);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_sub_region ON gl_eudr_cim_country_profiles (sub_region);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_latest_cpi ON gl_eudr_cim_country_profiles (latest_cpi);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_latest_wgi ON gl_eudr_cim_country_profiles (latest_wgi_corruption);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_risk_level ON gl_eudr_cim_country_profiles (overall_risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_art29 ON gl_eudr_cim_country_profiles (article_29_class);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_cpi_trend ON gl_eudr_cim_country_profiles (cpi_trend);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_defor_rate ON gl_eudr_cim_country_profiles (annual_deforestation_rate);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_provenance ON gl_eudr_cim_country_profiles (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_tenant ON gl_eudr_cim_country_profiles (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_last_updated ON gl_eudr_cim_country_profiles (last_updated DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_region_risk ON gl_eudr_cim_country_profiles (region, overall_risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_commodity_rel ON gl_eudr_cim_country_profiles USING GIN (eudr_commodity_relevance);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_cp_metadata ON gl_eudr_cim_country_profiles USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_country_profiles IS 'Comprehensive country corruption profiles with latest CPI/WGI scores, EUDR classification, and deforestation data';
COMMENT ON COLUMN gl_eudr_cim_country_profiles.article_29_class IS 'EUDR Article 29 country benchmarking classification';
COMMENT ON COLUMN gl_eudr_cim_country_profiles.eudr_commodity_relevance IS 'Boolean map of EUDR-relevant commodities produced/exported by this country';


-- ============================================================================
-- 10. gl_eudr_cim_sector_risk_scores — Per-sector risk by country
-- ============================================================================
RAISE NOTICE 'V107 [10/12]: Creating gl_eudr_cim_sector_risk_scores...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_sector_risk_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    sector                      VARCHAR(30)     NOT NULL,
        -- 'agriculture', 'forestry', 'livestock', 'mining', 'oil_palm',
        -- 'cocoa', 'coffee', 'rubber', 'soya', 'timber', 'customs',
        -- 'land_registry', 'environmental_permits', 'export_licensing'
    risk_score                  DECIMAL(5,2)    NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
        -- Composite sector corruption risk (0-100)
    risk_factors                JSONB,
        -- { "bribery_prevalence": 0.30, "regulatory_capture": 0.25,
        --   "informal_economy_share": 0.20, "enforcement_gap": 0.15,
        --   "political_interference": 0.10 }
    trend                       VARCHAR(20),
        -- 'improving', 'stable', 'deteriorating'
    data_quality                VARCHAR(20)     DEFAULT 'medium',
        -- 'very_high', 'high', 'medium', 'low', 'very_low'
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    last_assessed               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_country_code ON gl_eudr_cim_sector_risk_scores (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_sector ON gl_eudr_cim_sector_risk_scores (sector);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_risk_score ON gl_eudr_cim_sector_risk_scores (risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_trend ON gl_eudr_cim_sector_risk_scores (trend);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_data_quality ON gl_eudr_cim_sector_risk_scores (data_quality);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_provenance ON gl_eudr_cim_sector_risk_scores (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_tenant ON gl_eudr_cim_sector_risk_scores (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_last_assessed ON gl_eudr_cim_sector_risk_scores (last_assessed DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_created ON gl_eudr_cim_sector_risk_scores (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_updated ON gl_eudr_cim_sector_risk_scores (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_country_sector ON gl_eudr_cim_sector_risk_scores (country_code, sector);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_sector_risk ON gl_eudr_cim_sector_risk_scores (sector, risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_risk_factors ON gl_eudr_cim_sector_risk_scores USING GIN (risk_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_srs_metadata ON gl_eudr_cim_sector_risk_scores USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_sector_risk_scores IS 'Per-sector corruption risk scores by country with weighted risk factors';
COMMENT ON COLUMN gl_eudr_cim_sector_risk_scores.risk_score IS 'Composite sector corruption risk: 0 = no risk, 100 = extreme risk';
COMMENT ON COLUMN gl_eudr_cim_sector_risk_scores.risk_factors IS 'Weighted contributing factors to sector-level corruption risk';


-- ============================================================================
-- 11. gl_eudr_cim_governance_indicators — Additional governance data
-- ============================================================================
RAISE NOTICE 'V107 [11/12]: Creating gl_eudr_cim_governance_indicators...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_governance_indicators (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    indicator_name              VARCHAR(100)    NOT NULL,
        -- 'open_budget_index', 'press_freedom_index', 'ease_of_doing_business',
        -- 'rule_of_law_index', 'fleg_score', 'forest_transparency_initiative',
        -- 'extractive_industries_transparency', 'anti_money_laundering',
        -- 'beneficial_ownership_transparency', 'land_governance_score'
    indicator_value             DECIMAL(8,4)    NOT NULL,
        -- Raw indicator value (scale depends on indicator)
    normalized_value            DECIMAL(5,2)    CHECK (normalized_value >= 0 AND normalized_value <= 100),
        -- Normalized to 0-100 for cross-indicator comparison
    source                      VARCHAR(50)     NOT NULL,
        -- 'IBP', 'RSF', 'World Bank', 'WJP', 'FLEG', 'FTI', 'EITI', 'FATF'
    year                        INTEGER         NOT NULL CHECK (year >= 1990 AND year <= 2100),
    data_quality                VARCHAR(20)     DEFAULT 'medium',
        -- 'very_high', 'high', 'medium', 'low', 'very_low'
    provenance_hash             VARCHAR(64),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_country_code ON gl_eudr_cim_governance_indicators (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_indicator_name ON gl_eudr_cim_governance_indicators (indicator_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_indicator_value ON gl_eudr_cim_governance_indicators (indicator_value);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_normalized ON gl_eudr_cim_governance_indicators (normalized_value);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_source ON gl_eudr_cim_governance_indicators (source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_year ON gl_eudr_cim_governance_indicators (year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_data_quality ON gl_eudr_cim_governance_indicators (data_quality);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_provenance ON gl_eudr_cim_governance_indicators (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_tenant ON gl_eudr_cim_governance_indicators (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_created ON gl_eudr_cim_governance_indicators (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_updated ON gl_eudr_cim_governance_indicators (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_country_ind ON gl_eudr_cim_governance_indicators (country_code, indicator_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_country_year ON gl_eudr_cim_governance_indicators (country_code, year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_ind_year ON gl_eudr_cim_governance_indicators (indicator_name, year);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_gi_metadata ON gl_eudr_cim_governance_indicators USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_governance_indicators IS 'Additional governance indicators from multiple sources for comprehensive corruption context';
COMMENT ON COLUMN gl_eudr_cim_governance_indicators.normalized_value IS 'Indicator value normalized to 0-100 scale for cross-indicator comparison';
COMMENT ON COLUMN gl_eudr_cim_governance_indicators.source IS 'Data source organization: IBP, RSF, World Bank, WJP, FLEG, FTI, EITI, FATF';


-- ============================================================================
-- 12. gl_eudr_cim_audit_log — Comprehensive audit trail (hypertable, 30d)
-- ============================================================================
RAISE NOTICE 'V107 [12/12]: Creating gl_eudr_cim_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cim_audit_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    event_type                  VARCHAR(50)     NOT NULL,
        -- 'cpi_imported', 'wgi_imported', 'bribery_assessed', 'institutional_scored',
        -- 'trend_analyzed', 'correlation_computed', 'alert_triggered', 'alert_acknowledged',
        -- 'alert_resolved', 'compliance_assessed', 'profile_updated', 'sector_scored',
        -- 'governance_imported', 'classification_changed', 'data_refreshed'
    entity_type                 VARCHAR(50)     NOT NULL,
        -- 'cpi_score', 'wgi_indicator', 'bribery_assessment', 'institutional_quality',
        -- 'trend_analysis', 'deforestation_correlation', 'alert', 'compliance_impact',
        -- 'country_profile', 'sector_risk', 'governance_indicator'
    entity_id                   VARCHAR(100)    NOT NULL,
    actor                       VARCHAR(100)    NOT NULL,
        -- User ID or system agent identifier
    details                     JSONB,
        -- { "changed_fields": ["cpi_score", "article_29_classification"],
        --   "old_values": {...}, "new_values": {...}, "reason": "..." }
    ip_address                  INET,
    user_agent                  TEXT,
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for immutability verification
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (event_id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_cim_audit_log',
    'created_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_event_type ON gl_eudr_cim_audit_log (event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_entity_type ON gl_eudr_cim_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_entity_id ON gl_eudr_cim_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_actor ON gl_eudr_cim_audit_log (actor, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_provenance ON gl_eudr_cim_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_tenant ON gl_eudr_cim_audit_log (tenant_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_entity_action ON gl_eudr_cim_audit_log (entity_type, event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_details ON gl_eudr_cim_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cim_audit_metadata ON gl_eudr_cim_audit_log USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cim_audit_log IS 'Comprehensive audit trail for all corruption index monitor operations';
COMMENT ON COLUMN gl_eudr_cim_audit_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Quarterly CPI summaries by region
RAISE NOTICE 'V107: Creating continuous aggregate: quarterly_cpi_summary...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_cim_quarterly_cpi_summary
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('90 days', score_date) AS quarter,
        tenant_id,
        region,
        COUNT(*) AS country_count,
        AVG(score) AS avg_cpi_score,
        MIN(score) AS min_cpi_score,
        MAX(score) AS max_cpi_score,
        AVG(percentile) AS avg_percentile,
        AVG(rank) AS avg_rank
    FROM gl_eudr_cim_cpi_scores
    GROUP BY quarter, tenant_id, region;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_cim_quarterly_cpi_summary',
        start_offset => INTERVAL '180 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_cim_quarterly_cpi_summary IS 'Quarterly rollup of CPI scores by region with count, average, min, and max';


-- Annual WGI summaries by dimension
RAISE NOTICE 'V107: Creating continuous aggregate: annual_wgi_summary...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_cim_annual_wgi_summary
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('365 days', indicator_date) AS year_bucket,
        tenant_id,
        dimension,
        COUNT(*) AS country_count,
        AVG(estimate) AS avg_estimate,
        MIN(estimate) AS min_estimate,
        MAX(estimate) AS max_estimate,
        AVG(percentile_rank) AS avg_percentile_rank,
        AVG(governance_score) AS avg_governance_score
    FROM gl_eudr_cim_wgi_indicators
    GROUP BY year_bucket, tenant_id, dimension;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_cim_annual_wgi_summary',
        start_offset => INTERVAL '730 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_cim_annual_wgi_summary IS 'Annual rollup of WGI indicators by governance dimension with estimate and percentile statistics';


-- ============================================================================
-- RETENTION POLICIES
-- ============================================================================

RAISE NOTICE 'V107: Creating retention policies...';

-- 10 years for CPI scores (long-term corruption trend analysis)
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_cim_cpi_scores', INTERVAL '10 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 10 years for WGI indicators (long-term governance trend analysis)
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_cim_wgi_indicators', INTERVAL '10 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_cim_audit_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V107: AGENT-EUDR-019 Corruption Index Monitor tables created successfully!';
RAISE NOTICE 'V107: Created 12 tables (3 hypertables), 2 continuous aggregates, ~140 indexes';
RAISE NOTICE 'V107: Retention policies: 10y CPI scores, 10y WGI indicators, 5y audit logs';

COMMIT;
