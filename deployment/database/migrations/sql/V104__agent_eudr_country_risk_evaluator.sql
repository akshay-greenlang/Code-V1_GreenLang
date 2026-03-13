-- ============================================================================
-- V104: AGENT-EUDR-016 Country Risk Evaluator Agent
-- ============================================================================
-- Creates tables for composite country risk scoring, commodity-specific risk
-- analysis, deforestation hotspot detection, governance index integration,
-- due diligence level classification, trade flow analysis, risk report
-- generation, regulatory update tracking, and audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_cre_country_risks, gl_eudr_cre_hotspots,
--              gl_eudr_cre_regulatory_updates
-- Continuous Aggregates: 2 (hourly_assessment_stats + hourly_hotspot_stats)
-- Retention Policies: 3 (5 years for assessments, 5 years for hotspots,
--                        3 years for regulatory updates)
-- Indexes: ~120
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V104: Creating AGENT-EUDR-016 Country Risk Evaluator tables...';

-- ============================================================================
-- 1. gl_eudr_cre_country_risks — Country risk assessments (hypertable, monthly)
-- ============================================================================
RAISE NOTICE 'V104 [1/12]: Creating gl_eudr_cre_country_risks (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_country_risks (
    id                      UUID            DEFAULT gen_random_uuid(),
    country_code            VARCHAR(2)      NOT NULL,
    country_name            VARCHAR(255),
    risk_level              VARCHAR(20)     NOT NULL,
        -- 'low', 'standard', 'high'
    risk_score              NUMERIC(5,2)    NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    composite_factors       JSONB,
        -- { "deforestation_rate": 0.30, "governance_index": 0.20,
        --   "enforcement_score": 0.15, "corruption_perception": 0.15,
        --   "forest_law_compliance": 0.10, "historical_trend": 0.10 }
    confidence              VARCHAR(20),
        -- 'very_high', 'high', 'medium', 'low', 'very_low'
    trend                   VARCHAR(30),
        -- 'improving', 'stable', 'deteriorating', 'rapidly_deteriorating'
    data_sources            TEXT[],
        -- e.g. ARRAY['FAO', 'GFW', 'WB_WGI', 'TI_CPI']
    assessed_by             VARCHAR(255),
    notes                   TEXT,
    metadata                JSONB,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, assessed_at)
);

SELECT create_hypertable(
    'gl_eudr_cre_country_risks',
    'assessed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_country_code ON gl_eudr_cre_country_risks (country_code, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_country_name ON gl_eudr_cre_country_risks (country_name, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_risk_level ON gl_eudr_cre_country_risks (risk_level, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_risk_score ON gl_eudr_cre_country_risks (risk_score, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_confidence ON gl_eudr_cre_country_risks (confidence, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_trend ON gl_eudr_cre_country_risks (trend, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_assessed_by ON gl_eudr_cre_country_risks (assessed_by, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_created ON gl_eudr_cre_country_risks (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_updated ON gl_eudr_cre_country_risks (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_code_level ON gl_eudr_cre_country_risks (country_code, risk_level, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_composite ON gl_eudr_cre_country_risks USING GIN (composite_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cr_metadata ON gl_eudr_cre_country_risks USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_cre_commodity_risks — Commodity-specific risk per country
-- ============================================================================
RAISE NOTICE 'V104 [2/12]: Creating gl_eudr_cre_commodity_risks...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_commodity_risks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
    commodity_type              VARCHAR(30)     NOT NULL,
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    risk_score                  NUMERIC(5,2)    CHECK (risk_score >= 0 AND risk_score <= 100),
    production_volume_tonnes    NUMERIC,
    deforestation_correlation   NUMERIC(5,4),
    certification_schemes       TEXT[],
        -- e.g. ARRAY['FSC', 'RSPO', 'Rainforest Alliance']
    seasonal_risk_factors       JSONB,
        -- { "dry_season_multiplier": 1.3, "fire_season_peak": "Aug-Oct" }
    market_share_pct            NUMERIC(5,2),
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_country ON gl_eudr_cre_commodity_risks (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_commodity ON gl_eudr_cre_commodity_risks (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_risk_score ON gl_eudr_cre_commodity_risks (risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_production ON gl_eudr_cre_commodity_risks (production_volume_tonnes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_defor_corr ON gl_eudr_cre_commodity_risks (deforestation_correlation);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_market ON gl_eudr_cre_commodity_risks (market_share_pct);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_assessed ON gl_eudr_cre_commodity_risks (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_country_commodity ON gl_eudr_cre_commodity_risks (country_code, commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_seasonal ON gl_eudr_cre_commodity_risks USING GIN (seasonal_risk_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_comr_metadata ON gl_eudr_cre_commodity_risks USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_cre_hotspots — Deforestation hotspots (hypertable, monthly)
-- ============================================================================
RAISE NOTICE 'V104 [3/12]: Creating gl_eudr_cre_hotspots (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_hotspots (
    id                          UUID            DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
    region                      VARCHAR(255),
    latitude                    NUMERIC(10,7)   NOT NULL,
    longitude                   NUMERIC(10,7)   NOT NULL,
    area_km2                    NUMERIC(12,4),
    severity                    VARCHAR(20)     NOT NULL,
        -- 'critical', 'high', 'medium', 'low'
    drivers                     TEXT[],
        -- e.g. ARRAY['cattle_ranching', 'soy_expansion', 'logging']
    tree_cover_loss_pct         NUMERIC(5,2),
    fire_alert_count            INTEGER,
    protected_area_overlap      BOOLEAN         DEFAULT FALSE,
    indigenous_territory        BOOLEAN         DEFAULT FALSE,
    detected_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    resolved_at                 TIMESTAMPTZ,
    metadata                    JSONB,

    PRIMARY KEY (id, detected_at)
);

SELECT create_hypertable(
    'gl_eudr_cre_hotspots',
    'detected_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_country ON gl_eudr_cre_hotspots (country_code, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_region ON gl_eudr_cre_hotspots (region, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_lat ON gl_eudr_cre_hotspots (latitude, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_lon ON gl_eudr_cre_hotspots (longitude, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_lat_lon ON gl_eudr_cre_hotspots (latitude, longitude, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_area ON gl_eudr_cre_hotspots (area_km2, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_severity ON gl_eudr_cre_hotspots (severity, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_tree_loss ON gl_eudr_cre_hotspots (tree_cover_loss_pct, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_fire ON gl_eudr_cre_hotspots (fire_alert_count, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_protected ON gl_eudr_cre_hotspots (protected_area_overlap, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_indigenous ON gl_eudr_cre_hotspots (indigenous_territory, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_resolved ON gl_eudr_cre_hotspots (resolved_at, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_country_severity ON gl_eudr_cre_hotspots (country_code, severity, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_hs_metadata ON gl_eudr_cre_hotspots USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_cre_governance_indices — Country governance indicators
-- ============================================================================
RAISE NOTICE 'V104 [4/12]: Creating gl_eudr_cre_governance_indices...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_governance_indices (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
    overall_score               NUMERIC(5,2)    CHECK (overall_score >= 0 AND overall_score <= 100),
    rule_of_law                 NUMERIC(5,2)    CHECK (rule_of_law >= 0 AND rule_of_law <= 100),
    regulatory_quality          NUMERIC(5,2)    CHECK (regulatory_quality >= 0 AND regulatory_quality <= 100),
    control_of_corruption       NUMERIC(5,2)    CHECK (control_of_corruption >= 0 AND control_of_corruption <= 100),
    government_effectiveness    NUMERIC(5,2)    CHECK (government_effectiveness >= 0 AND government_effectiveness <= 100),
    voice_accountability        NUMERIC(5,2)    CHECK (voice_accountability >= 0 AND voice_accountability <= 100),
    political_stability         NUMERIC(5,2)    CHECK (political_stability >= 0 AND political_stability <= 100),
    forest_governance           NUMERIC(5,2)    CHECK (forest_governance >= 0 AND forest_governance <= 100),
    enforcement_effectiveness   NUMERIC(5,2)    CHECK (enforcement_effectiveness >= 0 AND enforcement_effectiveness <= 100),
    data_sources                TEXT[],
        -- e.g. ARRAY['WB_WGI', 'TI_CPI', 'FAO', 'ITTO']
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_country ON gl_eudr_cre_governance_indices (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_overall ON gl_eudr_cre_governance_indices (overall_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_rule_of_law ON gl_eudr_cre_governance_indices (rule_of_law);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_reg_quality ON gl_eudr_cre_governance_indices (regulatory_quality);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_corruption ON gl_eudr_cre_governance_indices (control_of_corruption);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_gov_eff ON gl_eudr_cre_governance_indices (government_effectiveness);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_voice ON gl_eudr_cre_governance_indices (voice_accountability);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_political ON gl_eudr_cre_governance_indices (political_stability);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_forest ON gl_eudr_cre_governance_indices (forest_governance);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_enforcement ON gl_eudr_cre_governance_indices (enforcement_effectiveness);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_assessed ON gl_eudr_cre_governance_indices (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_gi_metadata ON gl_eudr_cre_governance_indices USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_cre_due_diligence_levels — DD level classifications
-- ============================================================================
RAISE NOTICE 'V104 [5/12]: Creating gl_eudr_cre_due_diligence_levels...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_due_diligence_levels (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
    commodity_type              VARCHAR(30)     NOT NULL,
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    level                       VARCHAR(20)     NOT NULL,
        -- 'simplified', 'standard', 'enhanced'
    risk_score                  NUMERIC(5,2)    CHECK (risk_score >= 0 AND risk_score <= 100),
    certification_credit        NUMERIC(5,2)    CHECK (certification_credit >= 0 AND certification_credit <= 100),
    audit_frequency             VARCHAR(50),
        -- 'annual', 'semi_annual', 'quarterly', 'monthly'
    cost_estimate_eur           NUMERIC(12,2),
    requirements                JSONB,
        -- { "satellite_verification": true, "on_site_audit": true,
        --   "supplier_questionnaire": true, "third_party_verification": false }
    classified_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_country ON gl_eudr_cre_due_diligence_levels (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_commodity ON gl_eudr_cre_due_diligence_levels (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_level ON gl_eudr_cre_due_diligence_levels (level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_risk_score ON gl_eudr_cre_due_diligence_levels (risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_cert_credit ON gl_eudr_cre_due_diligence_levels (certification_credit);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_audit_freq ON gl_eudr_cre_due_diligence_levels (audit_frequency);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_cost ON gl_eudr_cre_due_diligence_levels (cost_estimate_eur);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_classified ON gl_eudr_cre_due_diligence_levels (classified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_country_commodity ON gl_eudr_cre_due_diligence_levels (country_code, commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_country_level ON gl_eudr_cre_due_diligence_levels (country_code, level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_requirements ON gl_eudr_cre_due_diligence_levels USING GIN (requirements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ddl_metadata ON gl_eudr_cre_due_diligence_levels USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_cre_trade_flows — Bilateral trade flow analysis
-- ============================================================================
RAISE NOTICE 'V104 [6/12]: Creating gl_eudr_cre_trade_flows...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_trade_flows (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    origin_country              VARCHAR(2)      NOT NULL,
    destination_country         VARCHAR(2)      NOT NULL,
    commodity_type              VARCHAR(30)     NOT NULL,
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    volume_tonnes               NUMERIC(15,2),
    value_usd                   NUMERIC(15,2),
    route_risk_score            NUMERIC(5,2)    CHECK (route_risk_score >= 0 AND route_risk_score <= 100),
    transshipment_countries     TEXT[],
        -- e.g. ARRAY['SG', 'NL', 'BE'] for commodity laundering detection
    hs_codes                    TEXT[],
        -- e.g. ARRAY['1511.10', '1511.90'] for palm oil
    direction                   VARCHAR(20)     NOT NULL,
        -- 'import', 'export', 're_export'
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    re_export_risk              BOOLEAN         DEFAULT FALSE,
    metadata                    JSONB,
    recorded_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_origin ON gl_eudr_cre_trade_flows (origin_country);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_destination ON gl_eudr_cre_trade_flows (destination_country);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_commodity ON gl_eudr_cre_trade_flows (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_volume ON gl_eudr_cre_trade_flows (volume_tonnes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_value ON gl_eudr_cre_trade_flows (value_usd);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_route_risk ON gl_eudr_cre_trade_flows (route_risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_direction ON gl_eudr_cre_trade_flows (direction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_period ON gl_eudr_cre_trade_flows (period_start, period_end);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_re_export ON gl_eudr_cre_trade_flows (re_export_risk);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_recorded ON gl_eudr_cre_trade_flows (recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_origin_dest ON gl_eudr_cre_trade_flows (origin_country, destination_country);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_origin_commodity ON gl_eudr_cre_trade_flows (origin_country, commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_tf_metadata ON gl_eudr_cre_trade_flows USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_cre_risk_reports — Generated risk assessment reports
-- ============================================================================
RAISE NOTICE 'V104 [7/12]: Creating gl_eudr_cre_risk_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_risk_reports (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type                 VARCHAR(50)     NOT NULL,
        -- 'country_profile', 'commodity_matrix', 'executive_summary',
        -- 'comparative_analysis', 'regulatory_submission', 'trend_analysis'
    format                      VARCHAR(20)     NOT NULL,
        -- 'pdf', 'json', 'html', 'csv', 'xlsx'
    title                       VARCHAR(500)    NOT NULL,
    countries                   TEXT[],
        -- e.g. ARRAY['BR', 'ID', 'CO', 'MY']
    commodities                 TEXT[],
        -- e.g. ARRAY['cocoa', 'coffee', 'oil_palm']
    content_hash                VARCHAR(64),
        -- SHA-256 hash of report content
    file_size_bytes             BIGINT,
    generated_by                VARCHAR(255),
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_type ON gl_eudr_cre_risk_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_format ON gl_eudr_cre_risk_reports (format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_title ON gl_eudr_cre_risk_reports (title);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_content_hash ON gl_eudr_cre_risk_reports (content_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_size ON gl_eudr_cre_risk_reports (file_size_bytes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_generated_by ON gl_eudr_cre_risk_reports (generated_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_generated_at ON gl_eudr_cre_risk_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_type_format ON gl_eudr_cre_risk_reports (report_type, format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rr_metadata ON gl_eudr_cre_risk_reports USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_cre_regulatory_updates — EC benchmark and regulation changes
--    (hypertable, monthly on published_at)
-- ============================================================================
RAISE NOTICE 'V104 [8/12]: Creating gl_eudr_cre_regulatory_updates (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_regulatory_updates (
    id                          UUID            DEFAULT gen_random_uuid(),
    regulation                  VARCHAR(255)    NOT NULL,
        -- 'EUDR_2023_1115', 'EC_BENCHMARKING', 'NATIONAL_IMPLEMENTATION'
    country_code                VARCHAR(2),
    change_type                 VARCHAR(50)     NOT NULL,
        -- 'reclassification', 'new_benchmark', 'enforcement_change',
        -- 'exemption_granted', 'exemption_revoked', 'penalty_update'
    effective_date              DATE,
    impact_score                NUMERIC(5,2)    CHECK (impact_score >= 0 AND impact_score <= 100),
    description                 TEXT,
    source_url                  TEXT,
    previous_classification     VARCHAR(20),
        -- 'low', 'standard', 'high' (before change)
    new_classification          VARCHAR(20),
        -- 'low', 'standard', 'high' (after change)
    grace_period_days           INTEGER,
    published_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB,

    PRIMARY KEY (id, published_at)
);

SELECT create_hypertable(
    'gl_eudr_cre_regulatory_updates',
    'published_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_regulation ON gl_eudr_cre_regulatory_updates (regulation, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_country ON gl_eudr_cre_regulatory_updates (country_code, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_change_type ON gl_eudr_cre_regulatory_updates (change_type, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_effective ON gl_eudr_cre_regulatory_updates (effective_date, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_impact ON gl_eudr_cre_regulatory_updates (impact_score, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_prev_class ON gl_eudr_cre_regulatory_updates (previous_classification, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_new_class ON gl_eudr_cre_regulatory_updates (new_classification, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_grace ON gl_eudr_cre_regulatory_updates (grace_period_days, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_country_change ON gl_eudr_cre_regulatory_updates (country_code, change_type, published_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_ru_metadata ON gl_eudr_cre_regulatory_updates USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_cre_risk_factors — Individual risk factor scores per country
-- ============================================================================
RAISE NOTICE 'V104 [9/12]: Creating gl_eudr_cre_risk_factors...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_risk_factors (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
    factor_name                 VARCHAR(100)    NOT NULL,
        -- 'deforestation_rate', 'governance_index', 'enforcement_score',
        -- 'corruption_perception', 'forest_law_compliance', 'historical_trend'
    weight                      NUMERIC(5,2)    CHECK (weight >= 0 AND weight <= 100),
    raw_value                   NUMERIC(10,4),
    normalized_value            NUMERIC(5,2)    CHECK (normalized_value >= 0 AND normalized_value <= 100),
    data_source                 VARCHAR(100),
        -- 'FAO', 'GFW', 'WB_WGI', 'TI_CPI', 'ITTO'
    assessment_id               UUID,
        -- FK to gl_eudr_cre_country_risks (logical, not enforced on hypertable)
    last_updated                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_country ON gl_eudr_cre_risk_factors (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_factor ON gl_eudr_cre_risk_factors (factor_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_weight ON gl_eudr_cre_risk_factors (weight);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_raw ON gl_eudr_cre_risk_factors (raw_value);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_normalized ON gl_eudr_cre_risk_factors (normalized_value);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_source ON gl_eudr_cre_risk_factors (data_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_assessment ON gl_eudr_cre_risk_factors (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_updated ON gl_eudr_cre_risk_factors (last_updated DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_country_factor ON gl_eudr_cre_risk_factors (country_code, factor_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rf_metadata ON gl_eudr_cre_risk_factors USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 10. gl_eudr_cre_risk_history — Risk score change history
-- ============================================================================
RAISE NOTICE 'V104 [10/12]: Creating gl_eudr_cre_risk_history...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_risk_history (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                VARCHAR(2)      NOT NULL,
    risk_score                  NUMERIC(5,2)    CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level                  VARCHAR(20)     NOT NULL,
        -- 'low', 'standard', 'high'
    previous_score              NUMERIC(5,2)    CHECK (previous_score >= 0 AND previous_score <= 100),
    previous_level              VARCHAR(20),
        -- 'low', 'standard', 'high'
    change_reason               TEXT,
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_country ON gl_eudr_cre_risk_history (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_risk_score ON gl_eudr_cre_risk_history (risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_risk_level ON gl_eudr_cre_risk_history (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_prev_score ON gl_eudr_cre_risk_history (previous_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_prev_level ON gl_eudr_cre_risk_history (previous_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_assessed ON gl_eudr_cre_risk_history (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_country_assessed ON gl_eudr_cre_risk_history (country_code, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_rh_metadata ON gl_eudr_cre_risk_history USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 11. gl_eudr_cre_certifications — Certification scheme effectiveness
-- ============================================================================
RAISE NOTICE 'V104 [11/12]: Creating gl_eudr_cre_certifications...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_certifications (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scheme                      VARCHAR(50)     NOT NULL,
        -- 'FSC', 'PEFC', 'RSPO', 'Rainforest_Alliance', 'UTZ',
        -- 'Fairtrade', 'ISCC', 'organic'
    country_code                VARCHAR(2)      NOT NULL,
    commodity_type              VARCHAR(30)     NOT NULL,
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    effectiveness_score         NUMERIC(5,2)    CHECK (effectiveness_score >= 0 AND effectiveness_score <= 100),
    coverage_pct                NUMERIC(5,2)    CHECK (coverage_pct >= 0 AND coverage_pct <= 100),
    certified_area_ha           NUMERIC(15,2),
    certified_operators         INTEGER,
    data_source                 VARCHAR(100),
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_scheme ON gl_eudr_cre_certifications (scheme);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_country ON gl_eudr_cre_certifications (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_commodity ON gl_eudr_cre_certifications (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_effectiveness ON gl_eudr_cre_certifications (effectiveness_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_coverage ON gl_eudr_cre_certifications (coverage_pct);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_area ON gl_eudr_cre_certifications (certified_area_ha);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_operators ON gl_eudr_cre_certifications (certified_operators);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_source ON gl_eudr_cre_certifications (data_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_assessed ON gl_eudr_cre_certifications (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_scheme_country ON gl_eudr_cre_certifications (scheme, country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_country_commodity ON gl_eudr_cre_certifications (country_code, commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_cert_metadata ON gl_eudr_cre_certifications USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 12. gl_eudr_cre_audit_log — Immutable audit trail
-- ============================================================================
RAISE NOTICE 'V104 [12/12]: Creating gl_eudr_cre_audit_log...';

CREATE TABLE IF NOT EXISTS gl_eudr_cre_audit_log (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type                 VARCHAR(50)     NOT NULL,
        -- 'country_risk', 'commodity_risk', 'hotspot', 'governance_index',
        -- 'due_diligence_level', 'trade_flow', 'risk_report',
        -- 'regulatory_update', 'risk_factor', 'risk_history', 'certification'
    entity_id                   UUID            NOT NULL,
    action                      VARCHAR(50)     NOT NULL,
        -- 'created', 'updated', 'assessed', 'classified', 'reclassified',
        -- 'detected', 'resolved', 'generated', 'published', 'archived'
    actor                       VARCHAR(255)    NOT NULL,
    details                     JSONB,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_entity_type ON gl_eudr_cre_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_entity_id ON gl_eudr_cre_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_action ON gl_eudr_cre_audit_log (action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_actor ON gl_eudr_cre_audit_log (actor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_provenance ON gl_eudr_cre_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_created ON gl_eudr_cre_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_entity_action ON gl_eudr_cre_audit_log (entity_type, action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_entity_created ON gl_eudr_cre_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cre_al_details ON gl_eudr_cre_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================
RAISE NOTICE 'V104: Creating continuous aggregates...';

-- 1. Hourly assessment statistics by risk_level, confidence, and trend
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_cre_hourly_assessment_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', assessed_at)              AS bucket,
    risk_level,
    confidence,
    trend,
    COUNT(*)                                        AS assessment_count,
    COUNT(*) FILTER (WHERE risk_level = 'high')             AS high_risk_count,
    COUNT(*) FILTER (WHERE risk_level = 'standard')         AS standard_risk_count,
    COUNT(*) FILTER (WHERE risk_level = 'low')              AS low_risk_count,
    AVG(risk_score)                                 AS avg_risk_score,
    MAX(risk_score)                                 AS max_risk_score,
    MIN(risk_score)                                 AS min_risk_score,
    COUNT(DISTINCT country_code)                    AS countries_assessed
FROM gl_eudr_cre_country_risks
GROUP BY bucket, risk_level, confidence, trend
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_cre_hourly_assessment_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- 2. Hourly hotspot statistics by country_code, severity, and overlap flags
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_cre_hourly_hotspot_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', detected_at)              AS bucket,
    country_code,
    severity,
    COUNT(*)                                        AS hotspot_count,
    COUNT(*) FILTER (WHERE severity = 'critical')           AS critical_count,
    COUNT(*) FILTER (WHERE severity = 'high')               AS high_count,
    COUNT(*) FILTER (WHERE severity = 'medium')             AS medium_count,
    COUNT(*) FILTER (WHERE severity = 'low')                AS low_count,
    SUM(area_km2)                                   AS total_area_km2,
    AVG(tree_cover_loss_pct)                        AS avg_tree_cover_loss,
    SUM(fire_alert_count)                           AS total_fire_alerts,
    COUNT(*) FILTER (WHERE protected_area_overlap = TRUE)   AS protected_overlap_count,
    COUNT(*) FILTER (WHERE indigenous_territory = TRUE)     AS indigenous_overlap_count,
    COUNT(*) FILTER (WHERE resolved_at IS NOT NULL)         AS resolved_count
FROM gl_eudr_cre_hotspots
GROUP BY bucket, country_code, severity
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_cre_hourly_hotspot_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies
-- ============================================================================
RAISE NOTICE 'V104: Adding retention policies...';

-- EUDR Article 14: 5-year retention for country risk assessments
SELECT add_retention_policy('gl_eudr_cre_country_risks',
    INTERVAL '5 years', if_not_exists => TRUE);

-- 5-year retention for deforestation hotspots
SELECT add_retention_policy('gl_eudr_cre_hotspots',
    INTERVAL '5 years', if_not_exists => TRUE);

-- 3-year retention for regulatory updates
SELECT add_retention_policy('gl_eudr_cre_regulatory_updates',
    INTERVAL '3 years', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================
RAISE NOTICE 'V104: Adding table comments...';

COMMENT ON TABLE gl_eudr_cre_country_risks IS 'AGENT-EUDR-016: Composite country risk assessments with weighted multi-factor scoring (0-100) for 200+ countries, combining deforestation rate (30%), governance index (20%), enforcement score (15%), corruption perception (15%), forest law compliance (10%), and historical trend (10%). Supports EC Article 29 benchmarking alignment with confidence levels and trend tracking (hypertable, monthly on assessed_at)';
COMMENT ON TABLE gl_eudr_cre_commodity_risks IS 'AGENT-EUDR-016: Commodity-specific risk profiles for all 7 EUDR-regulated commodities (cattle/cocoa/coffee/oil_palm/rubber/soya/wood) per country, incorporating production volume, deforestation correlation, seasonal risk factors, certification scheme coverage, and market share percentages';
COMMENT ON TABLE gl_eudr_cre_hotspots IS 'AGENT-EUDR-016: Sub-national deforestation hotspot detection with GPS coordinates, area measurement, severity classification (critical/high/medium/low), tree cover loss percentage, fire alert correlation, protected area and indigenous territory overlap flags, and resolution tracking (hypertable, monthly on detected_at)';
COMMENT ON TABLE gl_eudr_cre_governance_indices IS 'AGENT-EUDR-016: Country governance indicators integrating World Bank WGI (rule of law, regulatory quality, corruption control, government effectiveness, voice/accountability, political stability), forest governance quality, and environmental law enforcement effectiveness scores';
COMMENT ON TABLE gl_eudr_cre_due_diligence_levels IS 'AGENT-EUDR-016: Automated 3-tier due diligence classification (simplified/standard/enhanced) per EUDR Articles 10-13 for each country-commodity combination, with certification credit, audit frequency recommendation, cost estimation in EUR, and detailed requirement specifications';
COMMENT ON TABLE gl_eudr_cre_trade_flows IS 'AGENT-EUDR-016: Bilateral trade flow analysis for EUDR-regulated commodities with volume/value tracking, route risk scoring, transshipment country detection for commodity laundering, HS code mapping, direction classification (import/export/re_export), and re-export risk flagging';
COMMENT ON TABLE gl_eudr_cre_risk_reports IS 'AGENT-EUDR-016: Generated risk assessment reports including country profiles, commodity-country matrices, executive summaries, comparative analyses, and regulatory submission documentation in multiple formats (PDF/JSON/HTML/CSV/XLSX) with SHA-256 content integrity hashing';
COMMENT ON TABLE gl_eudr_cre_regulatory_updates IS 'AGENT-EUDR-016: EC benchmarking list updates and regulatory change tracking with country reclassification impact assessment (previous/new classification), effective dates, grace periods, impact scoring, and source URL attribution (hypertable, monthly on published_at)';
COMMENT ON TABLE gl_eudr_cre_risk_factors IS 'AGENT-EUDR-016: Individual risk factor scores per country with configurable weights, raw and normalized values (0-100), data source attribution (FAO/GFW/WB_WGI/TI_CPI/ITTO), and linkage to parent country risk assessments for full audit traceability';
COMMENT ON TABLE gl_eudr_cre_risk_history IS 'AGENT-EUDR-016: Risk score change history tracking per country with current and previous score/level pairs, change reason documentation, and timestamp for trend analysis and regulatory audit support';
COMMENT ON TABLE gl_eudr_cre_certifications IS 'AGENT-EUDR-016: Certification scheme effectiveness tracking per country-commodity combination for schemes including FSC, PEFC, RSPO, Rainforest Alliance, UTZ, Fairtrade, ISCC, and organic, with coverage percentage, certified area in hectares, and operator counts';
COMMENT ON TABLE gl_eudr_cre_audit_log IS 'AGENT-EUDR-016: Immutable audit trail for all Country Risk Evaluator operations with entity tracking, action logging (created/updated/assessed/classified/reclassified/detected/resolved/generated/published/archived), actor identification, and SHA-256 provenance hashing';

RAISE NOTICE 'V104: AGENT-EUDR-016 Country Risk Evaluator migration complete.';

COMMIT;
