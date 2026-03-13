-- ============================================================================
-- V106: AGENT-EUDR-018 Commodity Risk Analyzer Agent
-- ============================================================================
-- Creates tables for commodity risk profiling, derived product tracking,
-- price history time series, production forecasts, substitution detection,
-- regulatory requirements, due diligence workflows, portfolio analyses,
-- commodity risk scores, processing chain definitions, market indicators,
-- and comprehensive audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_cra_price_history, gl_eudr_cra_commodity_risk_scores,
--              gl_eudr_cra_audit_log
-- Continuous Aggregates: 2 (daily_price_avg + weekly_risk_summary)
-- Retention Policies: 3 (5 years for price_history, 5 years for risk_scores,
--                        5 years for audit logs)
-- Indexes: ~130
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V106: Creating AGENT-EUDR-018 Commodity Risk Analyzer tables...';

-- ============================================================================
-- 1. gl_eudr_cra_commodity_profiles — Core commodity risk profiles
-- ============================================================================
RAISE NOTICE 'V106 [1/12]: Creating gl_eudr_cra_commodity_profiles...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_commodity_profiles (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity_type              VARCHAR(50)     NOT NULL,
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    risk_score                  NUMERIC(5,2)    NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    supply_chain_depth          INTEGER         NOT NULL CHECK (supply_chain_depth >= 1 AND supply_chain_depth <= 20),
    deforestation_risk_score    NUMERIC(5,2)    NOT NULL CHECK (deforestation_risk_score >= 0 AND deforestation_risk_score <= 100),
    price_volatility_index      NUMERIC(8,4)    CHECK (price_volatility_index >= 0),
    annual_production_volume    NUMERIC(18,2)   CHECK (annual_production_volume >= 0),
    country_distribution        JSONB,
        -- { "BR": 35.5, "ID": 22.3, "CO": 15.8, "GH": 12.1, "CI": 14.3 }
    processing_chains           JSONB,
        -- [{ "chain_id": "uuid", "stages": 5, "final_product": "chocolate" }]
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for integrity verification
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_commodity_type ON gl_eudr_cra_commodity_profiles (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_risk_score ON gl_eudr_cra_commodity_profiles (risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_supply_chain_depth ON gl_eudr_cra_commodity_profiles (supply_chain_depth);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_deforestation_risk ON gl_eudr_cra_commodity_profiles (deforestation_risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_volatility ON gl_eudr_cra_commodity_profiles (price_volatility_index);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_production_volume ON gl_eudr_cra_commodity_profiles (annual_production_volume);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_provenance ON gl_eudr_cra_commodity_profiles (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_tenant ON gl_eudr_cra_commodity_profiles (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_created ON gl_eudr_cra_commodity_profiles (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_updated ON gl_eudr_cra_commodity_profiles (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_commodity_risk ON gl_eudr_cra_commodity_profiles (commodity_type, risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_country_dist ON gl_eudr_cra_commodity_profiles USING GIN (country_distribution);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_proc_chains ON gl_eudr_cra_commodity_profiles USING GIN (processing_chains);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_cp_metadata ON gl_eudr_cra_commodity_profiles USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_commodity_profiles IS 'Core commodity risk profiles with deforestation risk, volatility, and supply chain depth';
COMMENT ON COLUMN gl_eudr_cra_commodity_profiles.country_distribution IS 'Percentage distribution of commodity sourcing by country';
COMMENT ON COLUMN gl_eudr_cra_commodity_profiles.provenance_hash IS 'SHA-256 hash for data integrity verification';


-- ============================================================================
-- 2. gl_eudr_cra_derived_products — Derived/processed product tracking
-- ============================================================================
RAISE NOTICE 'V106 [2/12]: Creating gl_eudr_cra_derived_products...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_derived_products (
    product_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source_commodity_type       VARCHAR(50)     NOT NULL,
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    product_category            VARCHAR(100)    NOT NULL,
        -- 'chocolate', 'leather', 'palm_oil', 'biofuel', 'furniture', 'paper', 'tire'
    processing_stages           JSONB,
        -- [{ "stage": 1, "process": "fermentation", "duration_days": 7 },
        --  { "stage": 2, "process": "roasting", "duration_days": 1 }]
    transformation_ratio        NUMERIC(8,4)    CHECK (transformation_ratio > 0),
        -- kg input per kg output
    risk_multiplier             NUMERIC(5,2)    CHECK (risk_multiplier >= 0 AND risk_multiplier <= 100),
        -- Risk adjustment factor for derived product
    traceability_score          NUMERIC(5,2)    CHECK (traceability_score >= 0 AND traceability_score <= 100),
        -- How traceable the product is back to source (0-100)
    annex_i_reference           VARCHAR(50),
        -- EUDR Annex I product reference code
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_source_commodity ON gl_eudr_cra_derived_products (source_commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_product_category ON gl_eudr_cra_derived_products (product_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_transformation ON gl_eudr_cra_derived_products (transformation_ratio);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_risk_multiplier ON gl_eudr_cra_derived_products (risk_multiplier);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_traceability ON gl_eudr_cra_derived_products (traceability_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_annex_ref ON gl_eudr_cra_derived_products (annex_i_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_tenant ON gl_eudr_cra_derived_products (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_created ON gl_eudr_cra_derived_products (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_updated ON gl_eudr_cra_derived_products (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_source_category ON gl_eudr_cra_derived_products (source_commodity_type, product_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_proc_stages ON gl_eudr_cra_derived_products USING GIN (processing_stages);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dp_metadata ON gl_eudr_cra_derived_products USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_derived_products IS 'Derived and processed product tracking with transformation ratios and traceability scores';
COMMENT ON COLUMN gl_eudr_cra_derived_products.transformation_ratio IS 'Kilograms of raw commodity input per kilogram of product output';
COMMENT ON COLUMN gl_eudr_cra_derived_products.annex_i_reference IS 'EUDR Annex I product classification reference code';


-- ============================================================================
-- 3. gl_eudr_cra_price_history — Commodity price time series (hypertable, 30d)
-- ============================================================================
RAISE NOTICE 'V106 [3/12]: Creating gl_eudr_cra_price_history (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_price_history (
    id                          UUID            DEFAULT gen_random_uuid(),
    commodity_type              VARCHAR(50)     NOT NULL,
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    price_date                  TIMESTAMPTZ     NOT NULL,
    price                       NUMERIC(12,4)   NOT NULL CHECK (price >= 0),
    currency                    VARCHAR(3)      NOT NULL DEFAULT 'USD',
        -- ISO 4217 currency code
    exchange                    VARCHAR(50),
        -- 'ICE', 'CBOT', 'NYMEX', 'LME', 'TOCOM', 'BMF'
    volatility_30d              NUMERIC(8,4)    CHECK (volatility_30d >= 0),
    volatility_90d              NUMERIC(8,4)    CHECK (volatility_90d >= 0),
    volume                      NUMERIC(18,2)   CHECK (volume >= 0),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, price_date)
);

SELECT create_hypertable(
    'gl_eudr_cra_price_history',
    'price_date',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_commodity_type ON gl_eudr_cra_price_history (commodity_type, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_price ON gl_eudr_cra_price_history (price, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_currency ON gl_eudr_cra_price_history (currency, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_exchange ON gl_eudr_cra_price_history (exchange, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_volatility_30d ON gl_eudr_cra_price_history (volatility_30d, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_volatility_90d ON gl_eudr_cra_price_history (volatility_90d, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_volume ON gl_eudr_cra_price_history (volume, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_tenant ON gl_eudr_cra_price_history (tenant_id, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_created ON gl_eudr_cra_price_history (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_updated ON gl_eudr_cra_price_history (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_commodity_currency ON gl_eudr_cra_price_history (commodity_type, currency, price_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_ph_metadata ON gl_eudr_cra_price_history USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_price_history IS 'Commodity price time series with volatility metrics and exchange data';
COMMENT ON COLUMN gl_eudr_cra_price_history.volatility_30d IS '30-day rolling price volatility (standard deviation of log returns)';
COMMENT ON COLUMN gl_eudr_cra_price_history.volatility_90d IS '90-day rolling price volatility (standard deviation of log returns)';


-- ============================================================================
-- 4. gl_eudr_cra_production_forecasts — Yield/production forecasts
-- ============================================================================
RAISE NOTICE 'V106 [4/12]: Creating gl_eudr_cra_production_forecasts...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_production_forecasts (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity_type              VARCHAR(50)     NOT NULL,
    region                      VARCHAR(100)    NOT NULL,
    country_code                VARCHAR(2)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    forecast_date               DATE            NOT NULL,
    yield_estimate              NUMERIC(12,2)   NOT NULL CHECK (yield_estimate >= 0),
        -- Estimated yield in metric tons
    confidence_lower            NUMERIC(12,2)   CHECK (confidence_lower >= 0),
        -- Lower bound of 95% confidence interval
    confidence_upper            NUMERIC(12,2)   CHECK (confidence_upper >= 0),
        -- Upper bound of 95% confidence interval
    climate_adjustment          NUMERIC(5,4)    CHECK (climate_adjustment >= -1 AND climate_adjustment <= 1),
        -- Climate impact adjustment factor (-1 to +1)
    seasonal_factor             NUMERIC(5,4)    CHECK (seasonal_factor >= 0 AND seasonal_factor <= 2),
        -- Seasonal production factor (0-2, where 1.0 = baseline)
    forecast_model              VARCHAR(50),
        -- 'arima', 'prophet', 'ensemble', 'ml_gradient_boost', 'expert_opinion'
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_commodity_type ON gl_eudr_cra_production_forecasts (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_region ON gl_eudr_cra_production_forecasts (region);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_country_code ON gl_eudr_cra_production_forecasts (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_forecast_date ON gl_eudr_cra_production_forecasts (forecast_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_yield ON gl_eudr_cra_production_forecasts (yield_estimate);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_climate_adj ON gl_eudr_cra_production_forecasts (climate_adjustment);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_seasonal ON gl_eudr_cra_production_forecasts (seasonal_factor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_model ON gl_eudr_cra_production_forecasts (forecast_model);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_tenant ON gl_eudr_cra_production_forecasts (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_created ON gl_eudr_cra_production_forecasts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_updated ON gl_eudr_cra_production_forecasts (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_commodity_country ON gl_eudr_cra_production_forecasts (commodity_type, country_code, forecast_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_commodity_date ON gl_eudr_cra_production_forecasts (commodity_type, forecast_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pf_metadata ON gl_eudr_cra_production_forecasts USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_production_forecasts IS 'Commodity yield and production forecasts with climate and seasonal adjustments';
COMMENT ON COLUMN gl_eudr_cra_production_forecasts.climate_adjustment IS 'Climate impact factor: negative = adverse, positive = favorable';
COMMENT ON COLUMN gl_eudr_cra_production_forecasts.seasonal_factor IS 'Seasonal production multiplier where 1.0 is the annual baseline';


-- ============================================================================
-- 5. gl_eudr_cra_substitution_events — Commodity switching detection
-- ============================================================================
RAISE NOTICE 'V106 [5/12]: Creating gl_eudr_cra_substitution_events...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_substitution_events (
    event_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 VARCHAR(100)    NOT NULL,
    from_commodity              VARCHAR(50)     NOT NULL,
    to_commodity                VARCHAR(50)     NOT NULL,
    detection_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    confidence                  NUMERIC(5,4)    CHECK (confidence >= 0 AND confidence <= 1),
        -- Detection confidence (0-1)
    risk_impact                 NUMERIC(5,2)    CHECK (risk_impact >= -100 AND risk_impact <= 100),
        -- Risk impact of substitution (-100 = risk reduction, +100 = risk increase)
    verified                    BOOLEAN         DEFAULT FALSE,
    verification_notes          TEXT,
    detection_method            VARCHAR(50),
        -- 'volume_analysis', 'price_anomaly', 'supplier_declaration', 'satellite', 'audit'
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_supplier_id ON gl_eudr_cra_substitution_events (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_from_commodity ON gl_eudr_cra_substitution_events (from_commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_to_commodity ON gl_eudr_cra_substitution_events (to_commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_detection_date ON gl_eudr_cra_substitution_events (detection_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_confidence ON gl_eudr_cra_substitution_events (confidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_risk_impact ON gl_eudr_cra_substitution_events (risk_impact);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_verified ON gl_eudr_cra_substitution_events (verified);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_method ON gl_eudr_cra_substitution_events (detection_method);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_tenant ON gl_eudr_cra_substitution_events (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_created ON gl_eudr_cra_substitution_events (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_updated ON gl_eudr_cra_substitution_events (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_supplier_date ON gl_eudr_cra_substitution_events (supplier_id, detection_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_from_to ON gl_eudr_cra_substitution_events (from_commodity, to_commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_se_metadata ON gl_eudr_cra_substitution_events USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_substitution_events IS 'Detection and tracking of commodity substitution events across suppliers';
COMMENT ON COLUMN gl_eudr_cra_substitution_events.confidence IS 'Detection confidence level (0-1) based on method and data quality';
COMMENT ON COLUMN gl_eudr_cra_substitution_events.risk_impact IS 'Net risk impact: negative = risk reduction, positive = risk increase';


-- ============================================================================
-- 6. gl_eudr_cra_regulatory_requirements — Per-commodity EUDR requirements
-- ============================================================================
RAISE NOTICE 'V106 [6/12]: Creating gl_eudr_cra_regulatory_requirements...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_regulatory_requirements (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity_type              VARCHAR(50)     NOT NULL,
    eudr_article                VARCHAR(20)     NOT NULL,
        -- 'Art.3', 'Art.4', 'Art.5', 'Art.9', 'Art.10', 'Art.11', 'Art.12'
    requirement_type            VARCHAR(50)     NOT NULL,
        -- 'due_diligence', 'geolocation', 'traceability', 'deforestation_free',
        -- 'legal_compliance', 'risk_assessment', 'risk_mitigation'
    documentation_needed        JSONB,
        -- [{ "doc_type": "geolocation_data", "mandatory": true, "format": "shapefile/kml" },
        --  { "doc_type": "due_diligence_statement", "mandatory": true, "format": "pdf" }]
    evidence_standard           VARCHAR(50),
        -- 'documentary', 'physical_inspection', 'satellite_verification', 'third_party_audit'
    penalty_category            VARCHAR(20),
        -- 'minor', 'major', 'severe', 'criminal'
    effective_date              DATE,
    applicable_operators        JSONB,
        -- { "sme": true, "large": true, "traders": true, "operators": true }
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_commodity_type ON gl_eudr_cra_regulatory_requirements (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_eudr_article ON gl_eudr_cra_regulatory_requirements (eudr_article);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_req_type ON gl_eudr_cra_regulatory_requirements (requirement_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_evidence ON gl_eudr_cra_regulatory_requirements (evidence_standard);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_penalty ON gl_eudr_cra_regulatory_requirements (penalty_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_effective ON gl_eudr_cra_regulatory_requirements (effective_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_tenant ON gl_eudr_cra_regulatory_requirements (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_created ON gl_eudr_cra_regulatory_requirements (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_updated ON gl_eudr_cra_regulatory_requirements (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_commodity_article ON gl_eudr_cra_regulatory_requirements (commodity_type, eudr_article);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_docs ON gl_eudr_cra_regulatory_requirements USING GIN (documentation_needed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_operators ON gl_eudr_cra_regulatory_requirements USING GIN (applicable_operators);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_rr_metadata ON gl_eudr_cra_regulatory_requirements USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_regulatory_requirements IS 'Per-commodity EUDR regulatory requirements with evidence standards and penalty classifications';
COMMENT ON COLUMN gl_eudr_cra_regulatory_requirements.eudr_article IS 'EUDR article reference defining the requirement';
COMMENT ON COLUMN gl_eudr_cra_regulatory_requirements.penalty_category IS 'Classification of penalties for non-compliance';


-- ============================================================================
-- 7. gl_eudr_cra_dd_workflows — Due diligence workflow tracking
-- ============================================================================
RAISE NOTICE 'V106 [7/12]: Creating gl_eudr_cra_dd_workflows...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_dd_workflows (
    workflow_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity_type              VARCHAR(50)     NOT NULL,
    supplier_id                 VARCHAR(100)    NOT NULL,
    status                      VARCHAR(20)     NOT NULL,
        -- 'draft', 'initiated', 'in_progress', 'review', 'completed', 'failed', 'cancelled'
    evidence_items              JSONB,
        -- [{ "item_id": "uuid", "type": "geolocation", "status": "verified", "score": 95.0 },
        --  { "item_id": "uuid", "type": "certificate", "status": "pending", "score": null }]
    verification_steps          JSONB,
        -- [{ "step": 1, "name": "document_collection", "status": "completed", "date": "..." },
        --  { "step": 2, "name": "risk_assessment", "status": "in_progress", "date": "..." }]
    completion_pct              NUMERIC(5,2)    CHECK (completion_pct >= 0 AND completion_pct <= 100),
    initiated_by                VARCHAR(100)    NOT NULL,
    initiated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at                TIMESTAMPTZ,
    due_date                    DATE,
    priority                    VARCHAR(20)     DEFAULT 'standard',
        -- 'low', 'standard', 'high', 'urgent'
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_commodity_type ON gl_eudr_cra_dd_workflows (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_supplier_id ON gl_eudr_cra_dd_workflows (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_status ON gl_eudr_cra_dd_workflows (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_completion ON gl_eudr_cra_dd_workflows (completion_pct);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_initiated_by ON gl_eudr_cra_dd_workflows (initiated_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_initiated_at ON gl_eudr_cra_dd_workflows (initiated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_completed_at ON gl_eudr_cra_dd_workflows (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_due_date ON gl_eudr_cra_dd_workflows (due_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_priority ON gl_eudr_cra_dd_workflows (priority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_tenant ON gl_eudr_cra_dd_workflows (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_created ON gl_eudr_cra_dd_workflows (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_updated ON gl_eudr_cra_dd_workflows (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_commodity_status ON gl_eudr_cra_dd_workflows (commodity_type, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_supplier_status ON gl_eudr_cra_dd_workflows (supplier_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-completed) workflows
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_active ON gl_eudr_cra_dd_workflows (status, due_date)
        WHERE status NOT IN ('completed', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_evidence ON gl_eudr_cra_dd_workflows USING GIN (evidence_items);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_verification ON gl_eudr_cra_dd_workflows USING GIN (verification_steps);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_dw_metadata ON gl_eudr_cra_dd_workflows USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_dd_workflows IS 'Due diligence workflow tracking for commodity-level EUDR compliance verification';
COMMENT ON COLUMN gl_eudr_cra_dd_workflows.evidence_items IS 'Evidence documents and verification items with status and scores';
COMMENT ON COLUMN gl_eudr_cra_dd_workflows.verification_steps IS 'Sequential verification steps with individual status tracking';


-- ============================================================================
-- 8. gl_eudr_cra_portfolio_analyses — Cross-commodity portfolio risk
-- ============================================================================
RAISE NOTICE 'V106 [8/12]: Creating gl_eudr_cra_portfolio_analyses...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_portfolio_analyses (
    analysis_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_name              VARCHAR(200)    NOT NULL,
    commodities                 JSONB           NOT NULL,
        -- [{ "commodity": "cocoa", "weight_pct": 35.0, "risk_score": 72.5 },
        --  { "commodity": "coffee", "weight_pct": 25.0, "risk_score": 58.3 }]
    concentration_index         NUMERIC(8,2)    CHECK (concentration_index >= 0),
        -- Herfindahl-Hirschman Index (HHI): 0 (perfectly diversified) to 10000 (single commodity)
    diversification_score       NUMERIC(5,2)    CHECK (diversification_score >= 0 AND diversification_score <= 100),
        -- Portfolio diversification score (0-100, higher = more diversified)
    total_risk_exposure         NUMERIC(12,2)   CHECK (total_risk_exposure >= 0),
        -- Weighted sum of commodity risk scores
    analysis_date               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    scenario_analysis           JSONB,
        -- { "baseline": 65.2, "stress_high": 82.1, "stress_low": 48.5, "best_case": 35.0 }
    recommendations             JSONB,
        -- [{ "action": "diversify_sourcing", "commodity": "cocoa", "priority": "high" }]
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_portfolio_name ON gl_eudr_cra_portfolio_analyses (portfolio_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_concentration ON gl_eudr_cra_portfolio_analyses (concentration_index);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_diversification ON gl_eudr_cra_portfolio_analyses (diversification_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_risk_exposure ON gl_eudr_cra_portfolio_analyses (total_risk_exposure);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_analysis_date ON gl_eudr_cra_portfolio_analyses (analysis_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_tenant ON gl_eudr_cra_portfolio_analyses (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_created ON gl_eudr_cra_portfolio_analyses (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_updated ON gl_eudr_cra_portfolio_analyses (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_commodities ON gl_eudr_cra_portfolio_analyses USING GIN (commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_scenario ON gl_eudr_cra_portfolio_analyses USING GIN (scenario_analysis);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_recommendations ON gl_eudr_cra_portfolio_analyses USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pa_metadata ON gl_eudr_cra_portfolio_analyses USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_portfolio_analyses IS 'Cross-commodity portfolio risk analyses with concentration and diversification metrics';
COMMENT ON COLUMN gl_eudr_cra_portfolio_analyses.concentration_index IS 'Herfindahl-Hirschman Index: 0 (diversified) to 10000 (single commodity)';
COMMENT ON COLUMN gl_eudr_cra_portfolio_analyses.diversification_score IS 'Portfolio diversification score (0-100, higher is more diversified)';


-- ============================================================================
-- 9. gl_eudr_cra_commodity_risk_scores — Time series risk scores (hypertable, 30d)
-- ============================================================================
RAISE NOTICE 'V106 [9/12]: Creating gl_eudr_cra_commodity_risk_scores (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_commodity_risk_scores (
    id                          UUID            DEFAULT gen_random_uuid(),
    commodity_type              VARCHAR(50)     NOT NULL,
    score_date                  TIMESTAMPTZ     NOT NULL,
    overall_risk                NUMERIC(5,2)    NOT NULL CHECK (overall_risk >= 0 AND overall_risk <= 100),
    deforestation_risk          NUMERIC(5,2)    CHECK (deforestation_risk >= 0 AND deforestation_risk <= 100),
    market_risk                 NUMERIC(5,2)    CHECK (market_risk >= 0 AND market_risk <= 100),
    regulatory_risk             NUMERIC(5,2)    CHECK (regulatory_risk >= 0 AND regulatory_risk <= 100),
    supply_risk                 NUMERIC(5,2)    CHECK (supply_risk >= 0 AND supply_risk <= 100),
    data_quality_score          NUMERIC(5,2)    CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    calculation_method          VARCHAR(50),
        -- 'weighted_average', 'bayesian', 'ensemble', 'expert_adjusted'
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, score_date)
);

SELECT create_hypertable(
    'gl_eudr_cra_commodity_risk_scores',
    'score_date',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_commodity_type ON gl_eudr_cra_commodity_risk_scores (commodity_type, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_overall_risk ON gl_eudr_cra_commodity_risk_scores (overall_risk, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_deforestation ON gl_eudr_cra_commodity_risk_scores (deforestation_risk, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_market ON gl_eudr_cra_commodity_risk_scores (market_risk, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_regulatory ON gl_eudr_cra_commodity_risk_scores (regulatory_risk, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_supply ON gl_eudr_cra_commodity_risk_scores (supply_risk, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_data_quality ON gl_eudr_cra_commodity_risk_scores (data_quality_score, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_method ON gl_eudr_cra_commodity_risk_scores (calculation_method, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_tenant ON gl_eudr_cra_commodity_risk_scores (tenant_id, score_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_created ON gl_eudr_cra_commodity_risk_scores (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_updated ON gl_eudr_cra_commodity_risk_scores (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_crs_metadata ON gl_eudr_cra_commodity_risk_scores USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_commodity_risk_scores IS 'Time series commodity risk scores across deforestation, market, regulatory, and supply dimensions';
COMMENT ON COLUMN gl_eudr_cra_commodity_risk_scores.overall_risk IS 'Composite risk score (0-100) aggregating all risk dimensions';
COMMENT ON COLUMN gl_eudr_cra_commodity_risk_scores.data_quality_score IS 'Quality score of underlying data used for risk calculation';


-- ============================================================================
-- 10. gl_eudr_cra_processing_chains — Commodity processing chain definitions
-- ============================================================================
RAISE NOTICE 'V106 [10/12]: Creating gl_eudr_cra_processing_chains...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_processing_chains (
    chain_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source_commodity            VARCHAR(50)     NOT NULL,
    final_product               VARCHAR(100)    NOT NULL,
    stages                      JSONB           NOT NULL,
        -- [{ "stage_num": 1, "process": "harvesting", "location_type": "farm", "risk_level": "low" },
        --  { "stage_num": 2, "process": "fermentation", "location_type": "processing_facility", "risk_level": "medium" },
        --  { "stage_num": 3, "process": "export", "location_type": "port", "risk_level": "high" }]
    total_stages                INTEGER         NOT NULL CHECK (total_stages >= 1 AND total_stages <= 50),
    risk_cumulative             NUMERIC(5,2)    CHECK (risk_cumulative >= 0 AND risk_cumulative <= 100),
        -- Cumulative risk across all processing stages
    traceability_difficulty     VARCHAR(20)     NOT NULL,
        -- 'simple', 'moderate', 'complex', 'very_complex', 'opaque'
    typical_duration_days       INTEGER         CHECK (typical_duration_days >= 0),
    geographic_spread           JSONB,
        -- { "countries": ["BR", "NL", "DE"], "continents": 2, "cross_border_stages": 2 }
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_source_commodity ON gl_eudr_cra_processing_chains (source_commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_final_product ON gl_eudr_cra_processing_chains (final_product);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_total_stages ON gl_eudr_cra_processing_chains (total_stages);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_risk_cumulative ON gl_eudr_cra_processing_chains (risk_cumulative);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_traceability ON gl_eudr_cra_processing_chains (traceability_difficulty);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_duration ON gl_eudr_cra_processing_chains (typical_duration_days);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_tenant ON gl_eudr_cra_processing_chains (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_created ON gl_eudr_cra_processing_chains (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_updated ON gl_eudr_cra_processing_chains (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_source_product ON gl_eudr_cra_processing_chains (source_commodity, final_product);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_stages ON gl_eudr_cra_processing_chains USING GIN (stages);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_geo_spread ON gl_eudr_cra_processing_chains USING GIN (geographic_spread);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_pc_metadata ON gl_eudr_cra_processing_chains USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_processing_chains IS 'Commodity processing chain definitions from raw material to finished product';
COMMENT ON COLUMN gl_eudr_cra_processing_chains.traceability_difficulty IS 'Difficulty of tracing product back to source through the processing chain';
COMMENT ON COLUMN gl_eudr_cra_processing_chains.risk_cumulative IS 'Cumulative risk score accumulated across all processing stages';


-- ============================================================================
-- 11. gl_eudr_cra_market_indicators — Market condition indicators
-- ============================================================================
RAISE NOTICE 'V106 [11/12]: Creating gl_eudr_cra_market_indicators...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_market_indicators (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    commodity_type              VARCHAR(50)     NOT NULL,
    indicator_date              TIMESTAMPTZ     NOT NULL,
    indicator_name              VARCHAR(100)    NOT NULL,
        -- 'supply_deficit', 'demand_surge', 'trade_restriction', 'weather_disruption',
        -- 'price_spike', 'inventory_low', 'shipping_delay', 'policy_change'
    indicator_value             NUMERIC(12,4)   NOT NULL,
    trend                       VARCHAR(20),
        -- 'rising', 'falling', 'stable', 'volatile', 'reversing'
    significance                VARCHAR(20),
        -- 'low', 'moderate', 'high', 'critical'
    source                      VARCHAR(100),
        -- 'bloomberg', 'fao', 'world_bank', 'icco', 'ico', 'internal_model'
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_commodity_type ON gl_eudr_cra_market_indicators (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_indicator_date ON gl_eudr_cra_market_indicators (indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_indicator_name ON gl_eudr_cra_market_indicators (indicator_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_indicator_value ON gl_eudr_cra_market_indicators (indicator_value);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_trend ON gl_eudr_cra_market_indicators (trend);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_significance ON gl_eudr_cra_market_indicators (significance);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_source ON gl_eudr_cra_market_indicators (source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_tenant ON gl_eudr_cra_market_indicators (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_created ON gl_eudr_cra_market_indicators (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_updated ON gl_eudr_cra_market_indicators (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_commodity_date ON gl_eudr_cra_market_indicators (commodity_type, indicator_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_commodity_name ON gl_eudr_cra_market_indicators (commodity_type, indicator_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_mi_metadata ON gl_eudr_cra_market_indicators USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_market_indicators IS 'Market condition indicators for commodity risk analysis and disruption detection';
COMMENT ON COLUMN gl_eudr_cra_market_indicators.trend IS 'Direction of indicator movement over recent period';
COMMENT ON COLUMN gl_eudr_cra_market_indicators.significance IS 'Impact significance level of the market indicator';


-- ============================================================================
-- 12. gl_eudr_cra_audit_log — Comprehensive audit trail (hypertable, 30d)
-- ============================================================================
RAISE NOTICE 'V106 [12/12]: Creating gl_eudr_cra_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cra_audit_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    event_type                  VARCHAR(50)     NOT NULL,
        -- 'profile_created', 'risk_scored', 'substitution_detected', 'dd_initiated',
        -- 'dd_completed', 'portfolio_analyzed', 'price_imported', 'forecast_generated',
        -- 'alert_triggered', 'requirement_updated', 'chain_mapped'
    entity_type                 VARCHAR(50)     NOT NULL,
        -- 'commodity_profile', 'derived_product', 'price_history', 'production_forecast',
        -- 'substitution_event', 'regulatory_requirement', 'dd_workflow',
        -- 'portfolio_analysis', 'risk_score', 'processing_chain', 'market_indicator'
    entity_id                   VARCHAR(100)    NOT NULL,
    actor                       VARCHAR(100)    NOT NULL,
        -- User ID or system agent identifier
    details                     JSONB,
        -- { "changed_fields": ["risk_score", "deforestation_risk"],
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
    'gl_eudr_cra_audit_log',
    'created_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_event_type ON gl_eudr_cra_audit_log (event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_entity_type ON gl_eudr_cra_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_entity_id ON gl_eudr_cra_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_actor ON gl_eudr_cra_audit_log (actor, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_provenance ON gl_eudr_cra_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_tenant ON gl_eudr_cra_audit_log (tenant_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_entity_action ON gl_eudr_cra_audit_log (entity_type, event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_details ON gl_eudr_cra_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cra_audit_metadata ON gl_eudr_cra_audit_log USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_cra_audit_log IS 'Comprehensive audit trail for all commodity risk analyzer operations';
COMMENT ON COLUMN gl_eudr_cra_audit_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Daily commodity price averages
RAISE NOTICE 'V106: Creating continuous aggregate: daily_price_avg...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_cra_daily_price_avg
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', price_date) AS day,
        tenant_id,
        commodity_type,
        currency,
        COUNT(*) AS price_count,
        AVG(price) AS avg_price,
        MIN(price) AS min_price,
        MAX(price) AS max_price,
        AVG(volatility_30d) AS avg_volatility_30d,
        AVG(volatility_90d) AS avg_volatility_90d,
        SUM(volume) AS total_volume
    FROM gl_eudr_cra_price_history
    GROUP BY day, tenant_id, commodity_type, currency;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_cra_daily_price_avg',
        start_offset => INTERVAL '7 days',
        end_offset => INTERVAL '1 day',
        schedule_interval => INTERVAL '1 day');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_cra_daily_price_avg IS 'Daily rollup of commodity prices with volume and volatility averages';


-- Weekly risk score summaries per commodity
RAISE NOTICE 'V106: Creating continuous aggregate: weekly_risk_summary...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_cra_weekly_risk_summary
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('7 days', score_date) AS week,
        tenant_id,
        commodity_type,
        COUNT(*) AS score_count,
        AVG(overall_risk) AS avg_overall_risk,
        MIN(overall_risk) AS min_overall_risk,
        MAX(overall_risk) AS max_overall_risk,
        AVG(deforestation_risk) AS avg_deforestation_risk,
        AVG(market_risk) AS avg_market_risk,
        AVG(regulatory_risk) AS avg_regulatory_risk,
        AVG(supply_risk) AS avg_supply_risk,
        AVG(data_quality_score) AS avg_data_quality
    FROM gl_eudr_cra_commodity_risk_scores
    GROUP BY week, tenant_id, commodity_type;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_cra_weekly_risk_summary',
        start_offset => INTERVAL '30 days',
        end_offset => INTERVAL '7 days',
        schedule_interval => INTERVAL '1 day');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_cra_weekly_risk_summary IS 'Weekly rollup of commodity risk scores across all risk dimensions';


-- ============================================================================
-- RETENTION POLICIES
-- ============================================================================

RAISE NOTICE 'V106: Creating retention policies...';

-- 5 years for price history
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_cra_price_history', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for commodity risk scores
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_cra_commodity_risk_scores', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_cra_audit_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V106: AGENT-EUDR-018 Commodity Risk Analyzer tables created successfully!';
RAISE NOTICE 'V106: Created 12 tables (3 hypertables), 2 continuous aggregates, ~130 indexes';
RAISE NOTICE 'V106: Retention policies: 5y price_history, 5y risk_scores, 5y audit logs';

COMMIT;
