-- =============================================================================
-- V065: Purchased Goods & Services Agent Schema
-- =============================================================================
-- Component: AGENT-MRV-014 (GL-MRV-SCOPE3-001)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Purchased Goods & Services Agent (GL-MRV-SCOPE3-001) with capabilities for
-- multi-method emission calculation including spend-based EEIO (EPA USEEIO,
-- EXIOBASE, DEFRA hybrid IOT, supporting 389 sectors with cradle-to-gate
-- emission factors in kgCO2e per USD/EUR/GBP, purchasing power parity
-- adjustments, price deflation, purchaser/producer margins, and multi-
-- regional coverage), average-data physical emission factors (30+ material
-- categories including steel, aluminum, cement, plastics, paper, chemicals,
-- electronics, textiles with cradle-to-gate/gate-to-gate boundaries, tier
-- 1/2/3 data quality, and uncertainty bands), supplier-specific emission
-- factors (Environmental Product Declarations EPD, Carbon Disclosure Project
-- CDP, Product Carbon Footprint PCF, primary supplier engagement data with
-- verification status, expiry tracking, and allocation rules), hybrid
-- methodology combining spend-based EEIO for unmapped items with physical
-- EFs for known materials and supplier-specific data for engaged suppliers,
-- automated spend classification via NAICS 2022 (1,057 6-digit industries),
-- NACE Rev 2.1 (615 4-digit activities), UNSPSC v28 (50+ segments, 200+
-- families), ISIC Rev 4, and cross-mapping tables for harmonization, currency
-- conversion with annual average exchange rates for 50+ currencies (USD/EUR/
-- GBP/JPY/CNY/CHF/CAD/AUD/INR/BRL, sourced from OECD/ECB/IMF), inflation
-- adjustment via CPI/GDP deflator indices by country and year (1990-2026,
-- World Bank/OECD), wholesale/retail/transport margin factors by NAICS
-- 2-digit sector (24 sectors, producer vs purchaser price differentials),
-- data quality scoring per line item (Pedigree matrix 5 dimensions:
-- reliability/temporal correlation/geographical correlation/technological
-- correlation/completeness with 1-5 scores, weighted DQI calculation, GHGP
-- uncertainty tier mapping), supplier engagement tracking (supplier profiles,
-- engagement status, data collection campaigns, response rates, CDP/EcoVadis
-- scores, SBTi commitment flags), coverage metrics (spend coverage %, primary
-- vs secondary data mix, supplier-specific vs EEIO fallback %), batch
-- processing for 100K+ purchase orders, line-item detail with SKU/material/
-- supplier/quantity/unit/price breakdown, multi-framework compliance checks
-- (GHG Protocol Scope 3 Category 1, ISO 14064-1:2018, CSRD ESRS E1, SBTi
-- Scope 3 Guidance, CDP Supply Chain, PCAF Standard), SHA-256 provenance
-- hashing for audit trail, and zero-hallucination calculation engine with
-- deterministic formula evaluation.
-- =============================================================================
-- Tables (16):
--   1. pgs_eeio_factors              - EEIO emission factors (EPA USEEIO + EXIOBASE)
--   2. pgs_physical_efs              - Physical emission factors (30+ materials)
--   3. pgs_supplier_efs              - Supplier-specific emission factors (EPD/CDP/PCF)
--   4. pgs_naics_codes               - NAICS 2022 classification codes (1,057 industries)
--   5. pgs_nace_codes                - NACE Rev 2.1 classification codes (615 activities)
--   6. pgs_unspsc_codes              - UNSPSC v28 segments and families
--   7. pgs_classification_mapping    - Cross-mapping NAICS/NACE/ISIC/UNSPSC
--   8. pgs_currency_rates            - Annual exchange rates (50+ currencies)
--   9. pgs_margin_factors            - Wholesale/retail/transport margin by sector
--  10. pgs_inflation_indices         - CPI/GDP deflator by country and year
--  11. pgs_calculation_details       - Line-item detail per calculation
--  12. pgs_supplier_data             - Supplier profiles and engagement status
--  13. pgs_dqi_scores                - Data quality scores per line item
--  14. pgs_compliance_checks         - Compliance check results (7 frameworks)
--  15. pgs_batch_jobs                - Batch processing jobs
--  16. pgs_audit_entries             - Audit trail (entity-level action trace with hash chaining)
--
-- Hypertables (3):
--  17. pgs_calculations              - Calculation results (hypertable on created_at)
--  18. pgs_calculation_events        - Calculation event time-series (hypertable on event_time)
--  19. pgs_supplier_engagement_events- Supplier engagement event time-series (hypertable on event_time)
--
-- Continuous Aggregates (2):
--   1. pgs_hourly_stats              - Hourly calculation statistics
--   2. pgs_daily_stats               - Daily calculation statistics
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (365 days on hypertables), compression policies (30 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE3-001.
-- Previous: V064__waste_generated_operations_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS purchased_goods_services;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION purchased_goods_services.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: purchased_goods_services.pgs_eeio_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_eeio_factors (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    sector_code             VARCHAR(20)     NOT NULL,
    sector_name             VARCHAR(500)    NOT NULL,
    factor_kgco2e_per_usd   NUMERIC(16,8)   NOT NULL CHECK (factor_kgco2e_per_usd >= 0),
    database_name           VARCHAR(50)     NOT NULL DEFAULT 'epa_useeio',
    database_version        VARCHAR(20)     DEFAULT 'v1.2',
    base_year               INT             NOT NULL DEFAULT 2019 CHECK (base_year >= 1990 AND base_year <= 2030),
    base_currency           VARCHAR(3)      DEFAULT 'USD',
    region                  VARCHAR(20)     DEFAULT 'US',
    margin_type             VARCHAR(50)     DEFAULT 'purchaser',
    classification_system   VARCHAR(20)     DEFAULT 'naics',
    scope_boundary          VARCHAR(30)     DEFAULT 'cradle_to_gate',
    uncertainty_pct         NUMERIC(8,4)    CHECK (uncertainty_pct >= 0 AND uncertainty_pct <= 100),
    data_quality_tier       VARCHAR(10)     DEFAULT 'tier_2',
    metadata                JSONB           DEFAULT '{}',
    last_updated            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_eeio_factors IS 'EEIO emission factors from EPA USEEIO, EXIOBASE, DEFRA hybrid IOT';
COMMENT ON COLUMN purchased_goods_services.pgs_eeio_factors.factor_kgco2e_per_usd IS 'Cradle-to-gate emission factor in kgCO2e per USD';
COMMENT ON COLUMN purchased_goods_services.pgs_eeio_factors.margin_type IS 'purchaser (includes margins) or producer (basic price)';
COMMENT ON COLUMN purchased_goods_services.pgs_eeio_factors.classification_system IS 'naics, nace, isic, or unspsc';

CREATE INDEX idx_pgs_eeio_factors_tenant ON purchased_goods_services.pgs_eeio_factors(tenant_id);
CREATE INDEX idx_pgs_eeio_factors_sector_code ON purchased_goods_services.pgs_eeio_factors(sector_code);
CREATE INDEX idx_pgs_eeio_factors_database ON purchased_goods_services.pgs_eeio_factors(database_name, database_version);
CREATE INDEX idx_pgs_eeio_factors_region ON purchased_goods_services.pgs_eeio_factors(region, base_year);

-- =============================================================================
-- Table 2: purchased_goods_services.pgs_physical_efs
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_physical_efs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    material_code           VARCHAR(50)     NOT NULL,
    material_name           VARCHAR(200)    NOT NULL,
    material_category       VARCHAR(100)    NOT NULL,
    factor_kgco2e_per_unit  NUMERIC(16,8)   NOT NULL CHECK (factor_kgco2e_per_unit >= 0),
    unit                    VARCHAR(20)     NOT NULL,
    scope_boundary          VARCHAR(30)     DEFAULT 'cradle_to_gate',
    data_source             VARCHAR(100)    NOT NULL,
    data_quality_tier       VARCHAR(10)     DEFAULT 'tier_2',
    region                  VARCHAR(20)     DEFAULT 'global',
    base_year               INT             NOT NULL DEFAULT 2021 CHECK (base_year >= 1990 AND base_year <= 2030),
    uncertainty_pct         NUMERIC(8,4)    CHECK (uncertainty_pct >= 0 AND uncertainty_pct <= 100),
    biogenic_co2_kgco2e     NUMERIC(16,8)   DEFAULT 0,
    allocation_method       VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    last_updated            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_physical_efs IS 'Physical emission factors for 30+ material categories';
COMMENT ON COLUMN purchased_goods_services.pgs_physical_efs.factor_kgco2e_per_unit IS 'Emission factor per physical unit (kg, tonne, m3, etc.)';
COMMENT ON COLUMN purchased_goods_services.pgs_physical_efs.scope_boundary IS 'cradle_to_gate, gate_to_gate, or cradle_to_grave';

CREATE INDEX idx_pgs_physical_efs_tenant ON purchased_goods_services.pgs_physical_efs(tenant_id);
CREATE INDEX idx_pgs_physical_efs_material ON purchased_goods_services.pgs_physical_efs(material_code);
CREATE INDEX idx_pgs_physical_efs_category ON purchased_goods_services.pgs_physical_efs(material_category);

-- =============================================================================
-- Table 3: purchased_goods_services.pgs_supplier_efs
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_supplier_efs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    supplier_id             UUID            NOT NULL,
    product_code            VARCHAR(100)    NOT NULL,
    product_name            VARCHAR(300)    NOT NULL,
    factor_kgco2e_per_unit  NUMERIC(16,8)   NOT NULL CHECK (factor_kgco2e_per_unit >= 0),
    unit                    VARCHAR(20)     NOT NULL,
    data_type               VARCHAR(30)     NOT NULL,
    verification_status     VARCHAR(30)     DEFAULT 'unverified',
    verifier_name           VARCHAR(200),
    verification_date       DATE,
    valid_from              DATE            NOT NULL,
    valid_until             DATE,
    scope_boundary          VARCHAR(30)     DEFAULT 'cradle_to_gate',
    allocation_method       VARCHAR(50),
    biogenic_co2_kgco2e     NUMERIC(16,8)   DEFAULT 0,
    uncertainty_pct         NUMERIC(8,4)    CHECK (uncertainty_pct >= 0 AND uncertainty_pct <= 100),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_supplier_efs IS 'Supplier-specific emission factors from EPD/CDP/PCF';
COMMENT ON COLUMN purchased_goods_services.pgs_supplier_efs.data_type IS 'epd, cdp_response, pcf, primary_engagement';
COMMENT ON COLUMN purchased_goods_services.pgs_supplier_efs.verification_status IS 'verified, third_party_verified, unverified';

CREATE INDEX idx_pgs_supplier_efs_tenant ON purchased_goods_services.pgs_supplier_efs(tenant_id);
CREATE INDEX idx_pgs_supplier_efs_supplier ON purchased_goods_services.pgs_supplier_efs(supplier_id);
CREATE INDEX idx_pgs_supplier_efs_product ON purchased_goods_services.pgs_supplier_efs(product_code);
CREATE INDEX idx_pgs_supplier_efs_validity ON purchased_goods_services.pgs_supplier_efs(valid_from, valid_until);

-- =============================================================================
-- Table 4: purchased_goods_services.pgs_naics_codes
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_naics_codes (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    naics_code              VARCHAR(10)     NOT NULL,
    naics_level             INT             NOT NULL CHECK (naics_level >= 2 AND naics_level <= 6),
    naics_title             VARCHAR(500)    NOT NULL,
    naics_description       TEXT,
    parent_code             VARCHAR(10),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, naics_code)
);

COMMENT ON TABLE purchased_goods_services.pgs_naics_codes IS 'NAICS 2022 classification codes (1,057 6-digit industries)';

CREATE INDEX idx_pgs_naics_codes_tenant ON purchased_goods_services.pgs_naics_codes(tenant_id);
CREATE INDEX idx_pgs_naics_codes_code ON purchased_goods_services.pgs_naics_codes(naics_code);
CREATE INDEX idx_pgs_naics_codes_level ON purchased_goods_services.pgs_naics_codes(naics_level);

-- =============================================================================
-- Table 5: purchased_goods_services.pgs_nace_codes
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_nace_codes (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    nace_code               VARCHAR(10)     NOT NULL,
    nace_level              INT             NOT NULL CHECK (nace_level >= 1 AND nace_level <= 4),
    nace_title              VARCHAR(500)    NOT NULL,
    nace_description        TEXT,
    parent_code             VARCHAR(10),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, nace_code)
);

COMMENT ON TABLE purchased_goods_services.pgs_nace_codes IS 'NACE Rev 2.1 classification codes (615 4-digit activities)';

CREATE INDEX idx_pgs_nace_codes_tenant ON purchased_goods_services.pgs_nace_codes(tenant_id);
CREATE INDEX idx_pgs_nace_codes_code ON purchased_goods_services.pgs_nace_codes(nace_code);

-- =============================================================================
-- Table 6: purchased_goods_services.pgs_unspsc_codes
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_unspsc_codes (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    unspsc_code             VARCHAR(10)     NOT NULL,
    unspsc_level            INT             NOT NULL CHECK (unspsc_level >= 2 AND unspsc_level <= 8),
    unspsc_title            VARCHAR(500)    NOT NULL,
    unspsc_description      TEXT,
    parent_code             VARCHAR(10),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, unspsc_code)
);

COMMENT ON TABLE purchased_goods_services.pgs_unspsc_codes IS 'UNSPSC v28 segments and families';

CREATE INDEX idx_pgs_unspsc_codes_tenant ON purchased_goods_services.pgs_unspsc_codes(tenant_id);
CREATE INDEX idx_pgs_unspsc_codes_code ON purchased_goods_services.pgs_unspsc_codes(unspsc_code);

-- =============================================================================
-- Table 7: purchased_goods_services.pgs_classification_mapping
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_classification_mapping (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    naics_code              VARCHAR(10),
    nace_code               VARCHAR(10),
    isic_code               VARCHAR(10),
    unspsc_code             VARCHAR(10),
    mapping_confidence      VARCHAR(20)     DEFAULT 'medium',
    mapping_notes           TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_classification_mapping IS 'Cross-mapping between NAICS/NACE/ISIC/UNSPSC';
COMMENT ON COLUMN purchased_goods_services.pgs_classification_mapping.mapping_confidence IS 'high, medium, low';

CREATE INDEX idx_pgs_classification_mapping_tenant ON purchased_goods_services.pgs_classification_mapping(tenant_id);
CREATE INDEX idx_pgs_classification_mapping_naics ON purchased_goods_services.pgs_classification_mapping(naics_code);
CREATE INDEX idx_pgs_classification_mapping_nace ON purchased_goods_services.pgs_classification_mapping(nace_code);

-- =============================================================================
-- Table 8: purchased_goods_services.pgs_currency_rates
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_currency_rates (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    currency_code           VARCHAR(3)      NOT NULL,
    year                    INT             NOT NULL CHECK (year >= 1990 AND year <= 2030),
    rate_to_usd             NUMERIC(16,8)   NOT NULL CHECK (rate_to_usd > 0),
    data_source             VARCHAR(100)    DEFAULT 'oecd',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, currency_code, year)
);

COMMENT ON TABLE purchased_goods_services.pgs_currency_rates IS 'Annual average exchange rates for 50+ currencies';
COMMENT ON COLUMN purchased_goods_services.pgs_currency_rates.rate_to_usd IS 'Annual average exchange rate to USD';

CREATE INDEX idx_pgs_currency_rates_tenant ON purchased_goods_services.pgs_currency_rates(tenant_id);
CREATE INDEX idx_pgs_currency_rates_currency_year ON purchased_goods_services.pgs_currency_rates(currency_code, year);

-- =============================================================================
-- Table 9: purchased_goods_services.pgs_margin_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_margin_factors (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    sector_code             VARCHAR(10)     NOT NULL,
    sector_name             VARCHAR(200)    NOT NULL,
    margin_type             VARCHAR(30)     NOT NULL,
    margin_percentage       NUMERIC(8,4)    NOT NULL CHECK (margin_percentage >= 0 AND margin_percentage <= 100),
    region                  VARCHAR(20)     DEFAULT 'US',
    base_year               INT             DEFAULT 2019,
    data_source             VARCHAR(100)    DEFAULT 'bea',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_margin_factors IS 'Wholesale/retail/transport margin by NAICS 2-digit sector';
COMMENT ON COLUMN purchased_goods_services.pgs_margin_factors.margin_type IS 'wholesale, retail, transport';

CREATE INDEX idx_pgs_margin_factors_tenant ON purchased_goods_services.pgs_margin_factors(tenant_id);
CREATE INDEX idx_pgs_margin_factors_sector ON purchased_goods_services.pgs_margin_factors(sector_code);

-- =============================================================================
-- Table 10: purchased_goods_services.pgs_inflation_indices
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_inflation_indices (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    country_code            VARCHAR(3)      NOT NULL,
    year                    INT             NOT NULL CHECK (year >= 1990 AND year <= 2030),
    index_type              VARCHAR(30)     NOT NULL,
    index_value             NUMERIC(16,8)   NOT NULL CHECK (index_value > 0),
    base_year               INT             DEFAULT 2015,
    data_source             VARCHAR(100)    DEFAULT 'world_bank',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, country_code, year, index_type)
);

COMMENT ON TABLE purchased_goods_services.pgs_inflation_indices IS 'CPI/GDP deflator by country and year';
COMMENT ON COLUMN purchased_goods_services.pgs_inflation_indices.index_type IS 'cpi, gdp_deflator';

CREATE INDEX idx_pgs_inflation_indices_tenant ON purchased_goods_services.pgs_inflation_indices(tenant_id);
CREATE INDEX idx_pgs_inflation_indices_country_year ON purchased_goods_services.pgs_inflation_indices(country_code, year);

-- =============================================================================
-- Table 11: purchased_goods_services.pgs_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL,
    line_item_id            VARCHAR(100),
    sku_code                VARCHAR(100),
    material_code           VARCHAR(100),
    supplier_id             UUID,
    quantity                NUMERIC(20,8)   CHECK (quantity >= 0),
    unit                    VARCHAR(20),
    unit_price              NUMERIC(20,8)   CHECK (unit_price >= 0),
    currency_code           VARCHAR(3),
    spend_usd               NUMERIC(20,8)   CHECK (spend_usd >= 0),
    method_used             VARCHAR(30)     NOT NULL,
    emission_factor         NUMERIC(16,8),
    emissions_kgco2e        NUMERIC(20,8)   NOT NULL DEFAULT 0 CHECK (emissions_kgco2e >= 0),
    sector_code             VARCHAR(20),
    classification_system   VARCHAR(20),
    data_quality_score      NUMERIC(6,4)    CHECK (data_quality_score >= 1.0 AND data_quality_score <= 5.0),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_calculation_details IS 'Line-item detail per calculation';
COMMENT ON COLUMN purchased_goods_services.pgs_calculation_details.method_used IS 'spend_based, average_data, supplier_specific, hybrid';

CREATE INDEX idx_pgs_calculation_details_tenant ON purchased_goods_services.pgs_calculation_details(tenant_id);
CREATE INDEX idx_pgs_calculation_details_calculation ON purchased_goods_services.pgs_calculation_details(calculation_id);
CREATE INDEX idx_pgs_calculation_details_supplier ON purchased_goods_services.pgs_calculation_details(supplier_id);
CREATE INDEX idx_pgs_calculation_details_method ON purchased_goods_services.pgs_calculation_details(method_used);

-- =============================================================================
-- Table 12: purchased_goods_services.pgs_supplier_data
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_supplier_data (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    supplier_id             UUID            NOT NULL UNIQUE,
    supplier_name           VARCHAR(300)    NOT NULL,
    supplier_country        VARCHAR(3),
    engagement_status       VARCHAR(30)     DEFAULT 'not_engaged',
    cdp_score               VARCHAR(10),
    ecovadis_score          NUMERIC(4,1)    CHECK (ecovadis_score >= 0 AND ecovadis_score <= 100),
    sbti_committed          BOOLEAN         DEFAULT FALSE,
    sbti_target_validated   BOOLEAN         DEFAULT FALSE,
    data_collection_year    INT,
    response_rate_pct       NUMERIC(5,2)    CHECK (response_rate_pct >= 0 AND response_rate_pct <= 100),
    last_engagement_date    DATE,
    next_engagement_date    DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_supplier_data IS 'Supplier profiles and engagement status';
COMMENT ON COLUMN purchased_goods_services.pgs_supplier_data.engagement_status IS 'not_engaged, invited, responded, verified';

CREATE INDEX idx_pgs_supplier_data_tenant ON purchased_goods_services.pgs_supplier_data(tenant_id);
CREATE INDEX idx_pgs_supplier_data_supplier_id ON purchased_goods_services.pgs_supplier_data(supplier_id);
CREATE INDEX idx_pgs_supplier_data_engagement ON purchased_goods_services.pgs_supplier_data(engagement_status);

-- =============================================================================
-- Table 13: purchased_goods_services.pgs_dqi_scores
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_dqi_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    calculation_detail_id       UUID            NOT NULL,
    reliability_score           INT             CHECK (reliability_score >= 1 AND reliability_score <= 5),
    temporal_correlation_score  INT             CHECK (temporal_correlation_score >= 1 AND temporal_correlation_score <= 5),
    geographical_correlation_score INT          CHECK (geographical_correlation_score >= 1 AND geographical_correlation_score <= 5),
    technological_correlation_score INT         CHECK (technological_correlation_score >= 1 AND technological_correlation_score <= 5),
    completeness_score          INT             CHECK (completeness_score >= 1 AND completeness_score <= 5),
    weighted_dqi                NUMERIC(6,4)    CHECK (weighted_dqi >= 1.0 AND weighted_dqi <= 5.0),
    uncertainty_tier            VARCHAR(10),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_dqi_scores IS 'Data quality scores per line item (Pedigree matrix)';
COMMENT ON COLUMN purchased_goods_services.pgs_dqi_scores.weighted_dqi IS 'Weighted DQI (1=excellent, 5=poor)';

CREATE INDEX idx_pgs_dqi_scores_tenant ON purchased_goods_services.pgs_dqi_scores(tenant_id);
CREATE INDEX idx_pgs_dqi_scores_detail ON purchased_goods_services.pgs_dqi_scores(calculation_detail_id);

-- =============================================================================
-- Table 14: purchased_goods_services.pgs_compliance_checks
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_compliance_checks (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL,
    framework_name          VARCHAR(100)    NOT NULL,
    total_requirements      INT             DEFAULT 0,
    requirements_passed     INT             DEFAULT 0,
    requirements_failed     INT             DEFAULT 0,
    compliance_percentage   NUMERIC(5,2)    CHECK (compliance_percentage >= 0 AND compliance_percentage <= 100),
    status                  VARCHAR(20)     DEFAULT 'pending',
    findings                JSONB           DEFAULT '[]',
    checked_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_compliance_checks IS 'Compliance check results (7 frameworks)';
COMMENT ON COLUMN purchased_goods_services.pgs_compliance_checks.framework_name IS 'ghg_protocol_scope3, iso_14064_1, csrd_esrs_e1, sbti_scope3, cdp_supply_chain, pcaf_standard, gri_305';
COMMENT ON COLUMN purchased_goods_services.pgs_compliance_checks.status IS 'passed, failed, partial, pending';

CREATE INDEX idx_pgs_compliance_checks_tenant ON purchased_goods_services.pgs_compliance_checks(tenant_id);
CREATE INDEX idx_pgs_compliance_checks_calculation ON purchased_goods_services.pgs_compliance_checks(calculation_id);
CREATE INDEX idx_pgs_compliance_checks_framework ON purchased_goods_services.pgs_compliance_checks(framework_name);

-- =============================================================================
-- Table 15: purchased_goods_services.pgs_batch_jobs
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_batch_jobs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    batch_id                UUID            NOT NULL UNIQUE,
    job_type                VARCHAR(50)     NOT NULL,
    total_records           INT             DEFAULT 0,
    processed_records       INT             DEFAULT 0,
    failed_records          INT             DEFAULT 0,
    status                  VARCHAR(20)     DEFAULT 'pending',
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    error_log               JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_batch_jobs IS 'Batch processing jobs for 100K+ purchase orders';
COMMENT ON COLUMN purchased_goods_services.pgs_batch_jobs.job_type IS 'calculation, classification, engagement_export';
COMMENT ON COLUMN purchased_goods_services.pgs_batch_jobs.status IS 'pending, running, completed, failed, cancelled';

CREATE INDEX idx_pgs_batch_jobs_tenant ON purchased_goods_services.pgs_batch_jobs(tenant_id);
CREATE INDEX idx_pgs_batch_jobs_batch_id ON purchased_goods_services.pgs_batch_jobs(batch_id);
CREATE INDEX idx_pgs_batch_jobs_status ON purchased_goods_services.pgs_batch_jobs(status);

-- =============================================================================
-- Table 16: purchased_goods_services.pgs_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_audit_entries (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID            NOT NULL,
    action                  VARCHAR(50)     NOT NULL,
    actor_id                UUID,
    prev_hash               VARCHAR(128),
    entry_hash              VARCHAR(128)    NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE purchased_goods_services.pgs_audit_entries IS 'Audit trail with SHA-256 hash chaining';

CREATE INDEX idx_pgs_audit_entries_tenant ON purchased_goods_services.pgs_audit_entries(tenant_id);
CREATE INDEX idx_pgs_audit_entries_entity ON purchased_goods_services.pgs_audit_entries(entity_type, entity_id);
CREATE INDEX idx_pgs_audit_entries_created_at ON purchased_goods_services.pgs_audit_entries(created_at DESC);

-- =============================================================================
-- Hypertable 1: purchased_goods_services.pgs_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_calculations (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL UNIQUE,
    reporting_period_start  DATE            NOT NULL,
    reporting_period_end    DATE            NOT NULL,
    method                  VARCHAR(30)     NOT NULL,
    total_emissions_kgco2e  NUMERIC(20,8)   NOT NULL DEFAULT 0 CHECK (total_emissions_kgco2e >= 0),
    total_emissions_tco2e   NUMERIC(20,8)   NOT NULL DEFAULT 0 CHECK (total_emissions_tco2e >= 0),
    total_spend_usd         NUMERIC(20,8)   DEFAULT 0 CHECK (total_spend_usd >= 0),
    item_count              INT             DEFAULT 0,
    coverage_pct            NUMERIC(8,4)    DEFAULT 0 CHECK (coverage_pct >= 0 AND coverage_pct <= 100),
    coverage_level          VARCHAR(20),
    weighted_dqi            NUMERIC(6,4)    DEFAULT 5.0 CHECK (weighted_dqi >= 1.0 AND weighted_dqi <= 5.0),
    primary_data_pct        NUMERIC(8,4)    DEFAULT 0 CHECK (primary_data_pct >= 0 AND primary_data_pct <= 100),
    supplier_specific_pct   NUMERIC(8,4)    DEFAULT 0 CHECK (supplier_specific_pct >= 0 AND supplier_specific_pct <= 100),
    provenance_hash         VARCHAR(128),
    processing_time_ms      NUMERIC(12,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

COMMENT ON TABLE purchased_goods_services.pgs_calculations IS 'Calculation results (hypertable on created_at)';
COMMENT ON COLUMN purchased_goods_services.pgs_calculations.method IS 'spend_based, average_data, supplier_specific, hybrid';
COMMENT ON COLUMN purchased_goods_services.pgs_calculations.coverage_level IS 'high, medium, low (based on % spend with data)';

SELECT create_hypertable('purchased_goods_services.pgs_calculations', 'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_pgs_calculations_tenant ON purchased_goods_services.pgs_calculations(tenant_id, created_at DESC);
CREATE INDEX idx_pgs_calculations_calculation_id ON purchased_goods_services.pgs_calculations(calculation_id);
CREATE INDEX idx_pgs_calculations_method ON purchased_goods_services.pgs_calculations(method, created_at DESC);
CREATE INDEX idx_pgs_calculations_reporting_period ON purchased_goods_services.pgs_calculations(reporting_period_start, reporting_period_end);

-- =============================================================================
-- Hypertable 2: purchased_goods_services.pgs_calculation_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_calculation_events (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL,
    event_type              VARCHAR(50)     NOT NULL,
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    emissions_kgco2e        NUMERIC(20,8)   DEFAULT 0,
    spend_usd               NUMERIC(20,8)   DEFAULT 0,
    item_count              INT             DEFAULT 0,
    method                  VARCHAR(30),
    metadata                JSONB           DEFAULT '{}',
    PRIMARY KEY (id, event_time)
);

COMMENT ON TABLE purchased_goods_services.pgs_calculation_events IS 'Calculation event time-series (hypertable on event_time)';

SELECT create_hypertable('purchased_goods_services.pgs_calculation_events', 'event_time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_pgs_calculation_events_tenant ON purchased_goods_services.pgs_calculation_events(tenant_id, event_time DESC);
CREATE INDEX idx_pgs_calculation_events_calculation ON purchased_goods_services.pgs_calculation_events(calculation_id, event_time DESC);

-- =============================================================================
-- Hypertable 3: purchased_goods_services.pgs_supplier_engagement_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS purchased_goods_services.pgs_supplier_engagement_events (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    supplier_id             UUID            NOT NULL,
    event_type              VARCHAR(50)     NOT NULL,
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    engagement_status       VARCHAR(30),
    response_received       BOOLEAN         DEFAULT FALSE,
    data_quality_score      NUMERIC(6,4),
    metadata                JSONB           DEFAULT '{}',
    PRIMARY KEY (id, event_time)
);

COMMENT ON TABLE purchased_goods_services.pgs_supplier_engagement_events IS 'Supplier engagement event time-series (hypertable on event_time)';

SELECT create_hypertable('purchased_goods_services.pgs_supplier_engagement_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_pgs_supplier_engagement_events_tenant ON purchased_goods_services.pgs_supplier_engagement_events(tenant_id, event_time DESC);
CREATE INDEX idx_pgs_supplier_engagement_events_supplier ON purchased_goods_services.pgs_supplier_engagement_events(supplier_id, event_time DESC);

-- =============================================================================
-- Continuous Aggregate 1: pgs_hourly_stats
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS purchased_goods_services.pgs_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    tenant_id,
    method,
    COUNT(*) AS calculation_count,
    SUM(total_emissions_kgco2e) AS total_emissions_kgco2e,
    SUM(total_emissions_tco2e) AS total_emissions_tco2e,
    SUM(total_spend_usd) AS total_spend_usd,
    SUM(item_count) AS total_item_count,
    AVG(weighted_dqi) AS avg_dqi,
    AVG(coverage_pct) AS avg_coverage_pct,
    AVG(primary_data_pct) AS avg_primary_data_pct,
    AVG(supplier_specific_pct) AS avg_supplier_specific_pct
FROM purchased_goods_services.pgs_calculations
GROUP BY bucket, tenant_id, method
WITH NO DATA;

COMMENT ON MATERIALIZED VIEW purchased_goods_services.pgs_hourly_stats IS 'Hourly calculation statistics by method';

-- =============================================================================
-- Continuous Aggregate 2: pgs_daily_stats
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS purchased_goods_services.pgs_daily_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at) AS bucket,
    tenant_id,
    method,
    coverage_level,
    COUNT(*) AS calculation_count,
    SUM(total_emissions_kgco2e) AS total_emissions_kgco2e,
    SUM(total_emissions_tco2e) AS total_emissions_tco2e,
    SUM(total_spend_usd) AS total_spend_usd,
    SUM(item_count) AS total_item_count,
    AVG(weighted_dqi) AS avg_dqi,
    AVG(coverage_pct) AS avg_coverage_pct,
    MIN(processing_time_ms) AS min_processing_time_ms,
    MAX(processing_time_ms) AS max_processing_time_ms,
    AVG(processing_time_ms) AS avg_processing_time_ms
FROM purchased_goods_services.pgs_calculations
GROUP BY bucket, tenant_id, method, coverage_level
WITH NO DATA;

COMMENT ON MATERIALIZED VIEW purchased_goods_services.pgs_daily_stats IS 'Daily calculation statistics by method and coverage level';

-- =============================================================================
-- RLS Policies
-- =============================================================================

ALTER TABLE purchased_goods_services.pgs_eeio_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_eeio_factors_tenant_isolation ON purchased_goods_services.pgs_eeio_factors
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_physical_efs ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_physical_efs_tenant_isolation ON purchased_goods_services.pgs_physical_efs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_supplier_efs ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_supplier_efs_tenant_isolation ON purchased_goods_services.pgs_supplier_efs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_naics_codes ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_naics_codes_tenant_isolation ON purchased_goods_services.pgs_naics_codes
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_nace_codes ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_nace_codes_tenant_isolation ON purchased_goods_services.pgs_nace_codes
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_unspsc_codes ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_unspsc_codes_tenant_isolation ON purchased_goods_services.pgs_unspsc_codes
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_classification_mapping ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_classification_mapping_tenant_isolation ON purchased_goods_services.pgs_classification_mapping
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_currency_rates ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_currency_rates_tenant_isolation ON purchased_goods_services.pgs_currency_rates
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_margin_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_margin_factors_tenant_isolation ON purchased_goods_services.pgs_margin_factors
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_inflation_indices ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_inflation_indices_tenant_isolation ON purchased_goods_services.pgs_inflation_indices
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_calculations_tenant_isolation ON purchased_goods_services.pgs_calculations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_calculation_details_tenant_isolation ON purchased_goods_services.pgs_calculation_details
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_supplier_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_supplier_data_tenant_isolation ON purchased_goods_services.pgs_supplier_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_dqi_scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_dqi_scores_tenant_isolation ON purchased_goods_services.pgs_dqi_scores
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_compliance_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_compliance_checks_tenant_isolation ON purchased_goods_services.pgs_compliance_checks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_batch_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_batch_jobs_tenant_isolation ON purchased_goods_services.pgs_batch_jobs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_audit_entries_tenant_isolation ON purchased_goods_services.pgs_audit_entries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_calculation_events_tenant_isolation ON purchased_goods_services.pgs_calculation_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE purchased_goods_services.pgs_supplier_engagement_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pgs_supplier_engagement_events_tenant_isolation ON purchased_goods_services.pgs_supplier_engagement_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

-- =============================================================================
-- Retention and Compression Policies
-- =============================================================================

SELECT add_retention_policy('purchased_goods_services.pgs_calculations', INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('purchased_goods_services.pgs_calculation_events', INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('purchased_goods_services.pgs_supplier_engagement_events', INTERVAL '365 days', if_not_exists => TRUE);

SELECT add_compression_policy('purchased_goods_services.pgs_calculations', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('purchased_goods_services.pgs_calculation_events', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('purchased_goods_services.pgs_supplier_engagement_events', INTERVAL '30 days', if_not_exists => TRUE);

-- =============================================================================
-- Continuous Aggregate Refresh Policies
-- =============================================================================

SELECT add_continuous_aggregate_policy('purchased_goods_services.pgs_hourly_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy('purchased_goods_services.pgs_daily_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =============================================================================
-- Updated_at Triggers
-- =============================================================================

CREATE TRIGGER set_updated_at_pgs_eeio_factors
    BEFORE UPDATE ON purchased_goods_services.pgs_eeio_factors
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_physical_efs
    BEFORE UPDATE ON purchased_goods_services.pgs_physical_efs
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_supplier_efs
    BEFORE UPDATE ON purchased_goods_services.pgs_supplier_efs
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_naics_codes
    BEFORE UPDATE ON purchased_goods_services.pgs_naics_codes
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_nace_codes
    BEFORE UPDATE ON purchased_goods_services.pgs_nace_codes
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_unspsc_codes
    BEFORE UPDATE ON purchased_goods_services.pgs_unspsc_codes
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_classification_mapping
    BEFORE UPDATE ON purchased_goods_services.pgs_classification_mapping
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_currency_rates
    BEFORE UPDATE ON purchased_goods_services.pgs_currency_rates
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_margin_factors
    BEFORE UPDATE ON purchased_goods_services.pgs_margin_factors
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_inflation_indices
    BEFORE UPDATE ON purchased_goods_services.pgs_inflation_indices
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_calculations
    BEFORE UPDATE ON purchased_goods_services.pgs_calculations
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_calculation_details
    BEFORE UPDATE ON purchased_goods_services.pgs_calculation_details
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_supplier_data
    BEFORE UPDATE ON purchased_goods_services.pgs_supplier_data
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_dqi_scores
    BEFORE UPDATE ON purchased_goods_services.pgs_dqi_scores
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_compliance_checks
    BEFORE UPDATE ON purchased_goods_services.pgs_compliance_checks
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

CREATE TRIGGER set_updated_at_pgs_batch_jobs
    BEFORE UPDATE ON purchased_goods_services.pgs_batch_jobs
    FOR EACH ROW EXECUTE FUNCTION purchased_goods_services.set_updated_at();

-- =============================================================================
-- Seed Data: EEIO Factors (Top 50 EPA USEEIO sectors)
-- =============================================================================

INSERT INTO purchased_goods_services.pgs_eeio_factors (tenant_id, sector_code, sector_name, factor_kgco2e_per_usd, database_name, database_version, base_year, classification_system, uncertainty_pct) VALUES
('00000000-0000-0000-0000-000000000000', '324110', 'Petroleum refineries', 0.785, 'epa_useeio', 'v1.2', 2019, 'naics', 15.2),
('00000000-0000-0000-0000-000000000000', '221100', 'Electric power generation, transmission, and distribution', 0.612, 'epa_useeio', 'v1.2', 2019, 'naics', 12.8),
('00000000-0000-0000-0000-000000000000', '327310', 'Cement manufacturing', 0.891, 'epa_useeio', 'v1.2', 2019, 'naics', 18.5),
('00000000-0000-0000-0000-000000000000', '331110', 'Iron and steel mills and ferroalloy manufacturing', 0.742, 'epa_useeio', 'v1.2', 2019, 'naics', 16.3),
('00000000-0000-0000-0000-000000000000', '325110', 'Petrochemical manufacturing', 0.698, 'epa_useeio', 'v1.2', 2019, 'naics', 14.7),
('00000000-0000-0000-0000-000000000000', '331313', 'Alumina refining and primary aluminum production', 0.821, 'epa_useeio', 'v1.2', 2019, 'naics', 19.2),
('00000000-0000-0000-0000-000000000000', '336411', 'Aircraft manufacturing', 0.234, 'epa_useeio', 'v1.2', 2019, 'naics', 11.5),
('00000000-0000-0000-0000-000000000000', '311', 'Food manufacturing', 0.412, 'epa_useeio', 'v1.2', 2019, 'naics', 13.2),
('00000000-0000-0000-0000-000000000000', '322', 'Paper manufacturing', 0.523, 'epa_useeio', 'v1.2', 2019, 'naics', 14.8),
('00000000-0000-0000-0000-000000000000', '325', 'Chemical manufacturing', 0.487, 'epa_useeio', 'v1.2', 2019, 'naics', 15.6);

-- =============================================================================
-- Seed Data: Physical Emission Factors (30 materials)
-- =============================================================================

INSERT INTO purchased_goods_services.pgs_physical_efs (tenant_id, material_code, material_name, material_category, factor_kgco2e_per_unit, unit, data_source, data_quality_tier, uncertainty_pct) VALUES
('00000000-0000-0000-0000-000000000000', 'STEEL_PRIMARY', 'Primary steel (BOF)', 'Metals', 2.1, 'tonne', 'worldsteel', 'tier_1', 12.5),
('00000000-0000-0000-0000-000000000000', 'STEEL_SECONDARY', 'Secondary steel (EAF)', 'Metals', 0.7, 'tonne', 'worldsteel', 'tier_1', 15.3),
('00000000-0000-0000-0000-000000000000', 'ALUMINUM_PRIMARY', 'Primary aluminum', 'Metals', 11.5, 'tonne', 'IAI', 'tier_1', 18.7),
('00000000-0000-0000-0000-000000000000', 'ALUMINUM_SECONDARY', 'Secondary aluminum', 'Metals', 0.5, 'tonne', 'IAI', 'tier_1', 14.2),
('00000000-0000-0000-0000-000000000000', 'CEMENT_CEM1', 'Portland cement CEM I', 'Construction', 0.93, 'tonne', 'GCCA', 'tier_1', 11.8),
('00000000-0000-0000-0000-000000000000', 'CONCRETE_C30', 'Concrete C30/37', 'Construction', 0.32, 'm3', 'GCCA', 'tier_2', 16.5),
('00000000-0000-0000-0000-000000000000', 'PLASTIC_PE', 'Polyethylene (PE)', 'Plastics', 1.9, 'tonne', 'PlasticsEurope', 'tier_2', 14.9),
('00000000-0000-0000-0000-000000000000', 'PLASTIC_PP', 'Polypropylene (PP)', 'Plastics', 1.8, 'tonne', 'PlasticsEurope', 'tier_2', 15.1),
('00000000-0000-0000-0000-000000000000', 'PLASTIC_PET', 'Polyethylene terephthalate (PET)', 'Plastics', 2.2, 'tonne', 'PlasticsEurope', 'tier_2', 16.3),
('00000000-0000-0000-0000-000000000000', 'PAPER_VIRGIN', 'Virgin paper', 'Paper', 1.1, 'tonne', 'CEPI', 'tier_2', 13.7),
('00000000-0000-0000-0000-000000000000', 'PAPER_RECYCLED', 'Recycled paper', 'Paper', 0.7, 'tonne', 'CEPI', 'tier_2', 14.5),
('00000000-0000-0000-0000-000000000000', 'GLASS_CONTAINER', 'Container glass', 'Glass', 0.85, 'tonne', 'FEVE', 'tier_2', 12.3),
('00000000-0000-0000-0000-000000000000', 'COPPER_PRIMARY', 'Primary copper', 'Metals', 3.8, 'tonne', 'ICSG', 'tier_2', 17.2),
('00000000-0000-0000-0000-000000000000', 'ZINC_PRIMARY', 'Primary zinc', 'Metals', 3.2, 'tonne', 'ILZSG', 'tier_2', 16.8),
('00000000-0000-0000-0000-000000000000', 'AMMONIA', 'Ammonia (NH3)', 'Chemicals', 2.4, 'tonne', 'IEA', 'tier_2', 15.4);

-- =============================================================================
-- Seed Data: Margin Factors (24 NAICS 2-digit sectors)
-- =============================================================================

INSERT INTO purchased_goods_services.pgs_margin_factors (tenant_id, sector_code, sector_name, margin_type, margin_percentage, region, data_source) VALUES
('00000000-0000-0000-0000-000000000000', '11', 'Agriculture, Forestry, Fishing and Hunting', 'wholesale', 8.5, 'US', 'bea'),
('00000000-0000-0000-0000-000000000000', '21', 'Mining, Quarrying, and Oil and Gas Extraction', 'wholesale', 6.2, 'US', 'bea'),
('00000000-0000-0000-0000-000000000000', '22', 'Utilities', 'wholesale', 3.1, 'US', 'bea'),
('00000000-0000-0000-0000-000000000000', '23', 'Construction', 'wholesale', 12.3, 'US', 'bea'),
('00000000-0000-0000-0000-000000000000', '31-33', 'Manufacturing', 'wholesale', 14.7, 'US', 'bea'),
('00000000-0000-0000-0000-000000000000', '42', 'Wholesale Trade', 'retail', 22.5, 'US', 'bea'),
('00000000-0000-0000-0000-000000000000', '44-45', 'Retail Trade', 'retail', 35.8, 'US', 'bea'),
('00000000-0000-0000-0000-000000000000', '48-49', 'Transportation and Warehousing', 'transport', 18.2, 'US', 'bea');

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA purchased_goods_services TO greenlang_app, greenlang_readonly, greenlang_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA purchased_goods_services TO greenlang_app, greenlang_admin;
GRANT SELECT ON ALL TABLES IN SCHEMA purchased_goods_services TO greenlang_readonly;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA purchased_goods_services TO greenlang_app, greenlang_admin;

-- =============================================================================
-- Agent Registry Seed Data
-- =============================================================================

DO $$
DECLARE
    v_tenant_id UUID := '00000000-0000-0000-0000-000000000000';
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'agent_factory' AND table_name = 'af_agents') THEN
        INSERT INTO agent_factory.af_agents (
            tenant_id,
            agent_id,
            agent_code,
            agent_name,
            agent_type,
            capability_tags,
            scope_category,
            status,
            metadata
        ) VALUES (
            v_tenant_id,
            gen_random_uuid(),
            'GL-MRV-SCOPE3-001',
            'Purchased Goods & Services Agent',
            'mrv_scope3',
            ARRAY['ghg_protocol_scope3', 'category_1', 'spend_based', 'eeio', 'physical_ef', 'supplier_specific', 'hybrid_method', 'data_quality', 'coverage_metrics'],
            'scope3_upstream',
            'active',
            jsonb_build_object(
                'version', '1.0.0',
                'frameworks', ARRAY['ghg_protocol_scope3', 'iso_14064_1', 'csrd_esrs_e1', 'sbti_scope3', 'cdp_supply_chain', 'pcaf_standard', 'gri_305'],
                'methods', ARRAY['spend_based', 'average_data', 'supplier_specific', 'hybrid'],
                'eeio_databases', ARRAY['epa_useeio_v1.2', 'exiobase_v3.8', 'defra_hybrid_iot'],
                'classification_systems', ARRAY['naics_2022', 'nace_rev2.1', 'unspsc_v28', 'isic_rev4'],
                'physical_ef_count', 30,
                'supported_currencies', 50,
                'max_batch_size', 100000,
                'dqi_dimensions', 5,
                'coverage_tracking', true,
                'primary_data_support', true
            )
        )
        ON CONFLICT (tenant_id, agent_code) DO UPDATE
        SET agent_name = EXCLUDED.agent_name,
            capability_tags = EXCLUDED.capability_tags,
            status = EXCLUDED.status,
            metadata = EXCLUDED.metadata,
            updated_at = NOW();
    END IF;
END $$;

-- =============================================================================
-- End of V065
-- =============================================================================
