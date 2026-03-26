-- =============================================================================
-- V353: PACK-043 Scope 3 Complete Pack - Sector-Specific Modules
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates sector-specific Scope 3 calculation tables for specialized
-- industries. Supports PCAF (Partnership for Carbon Accounting Financials)
-- for financial institutions (Category 15 - Investments), retail/logistics
-- for distribution-heavy companies (Category 4/9), circular economy for
-- manufacturers (Category 5/12), and cloud carbon footprinting for
-- technology companies. Each module provides industry-standard fields
-- and methodologies.
--
-- Tables (5):
--   1. ghg_accounting_scope3_complete.pcaf_portfolios
--   2. ghg_accounting_scope3_complete.pcaf_assets
--   3. ghg_accounting_scope3_complete.retail_logistics
--   4. ghg_accounting_scope3_complete.circular_economy
--   5. ghg_accounting_scope3_complete.cloud_carbon
--
-- Enums (1):
--   1. ghg_accounting_scope3_complete.pcaf_asset_class
--
-- Also includes: indexes, RLS, comments.
-- Previous: V352__pack043_base_year_trends.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- ---------------------------------------------------------------------------
-- Enum: pcaf_asset_class
-- ---------------------------------------------------------------------------
-- PCAF asset classes for financed emissions calculation (PCAF Standard v2).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pcaf_asset_class' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.pcaf_asset_class AS ENUM (
            'LISTED_EQUITY',          -- Listed equity and corporate bonds
            'BUSINESS_LOANS',         -- Business loans and unlisted equity
            'PROJECT_FINANCE',        -- Project finance
            'COMMERCIAL_REAL_ESTATE', -- Commercial real estate
            'MORTGAGES',              -- Residential mortgages
            'MOTOR_VEHICLE_LOANS',    -- Motor vehicle loans
            'SOVEREIGN_DEBT'          -- Sovereign and sub-sovereign debt
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.pcaf_portfolios
-- =============================================================================
-- PCAF portfolio-level financed emissions for financial institutions
-- (Category 15). Each row represents an investment portfolio or asset class
-- with total invested amount, financed emissions, WACI, and data quality.

CREATE TABLE ghg_accounting_scope3_complete.pcaf_portfolios (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    -- Portfolio
    portfolio_name              VARCHAR(500)    NOT NULL,
    portfolio_code              VARCHAR(100),
    asset_class                 ghg_accounting_scope3_complete.pcaf_asset_class NOT NULL,
    -- Financial
    total_invested              NUMERIC(18,2)   NOT NULL,
    invested_currency           VARCHAR(3)      DEFAULT 'USD',
    reporting_year              INTEGER         NOT NULL,
    -- Emissions
    total_financed_tco2e        DECIMAL(15,3)   NOT NULL DEFAULT 0,
    scope1_financed_tco2e       DECIMAL(15,3),
    scope2_financed_tco2e       DECIMAL(15,3),
    scope3_financed_tco2e       DECIMAL(15,3),
    -- Intensity
    waci                        DECIMAL(12,6),
    waci_unit                   VARCHAR(50)     DEFAULT 'tCO2e/M$ revenue',
    emission_intensity          DECIMAL(12,6),
    intensity_unit              VARCHAR(50),
    -- Data quality (PCAF 1-5 scale)
    data_quality_score          DECIMAL(3,1)    NOT NULL DEFAULT 5.0,
    pct_score_1                 DECIMAL(5,2)    DEFAULT 0,
    pct_score_2                 DECIMAL(5,2)    DEFAULT 0,
    pct_score_3                 DECIMAL(5,2)    DEFAULT 0,
    pct_score_4                 DECIMAL(5,2)    DEFAULT 0,
    pct_score_5                 DECIMAL(5,2)    DEFAULT 0,
    -- Coverage
    assets_count                INTEGER         NOT NULL DEFAULT 0,
    assets_with_data            INTEGER         DEFAULT 0,
    coverage_pct                DECIMAL(5,2),
    -- Methodology
    methodology                 VARCHAR(100)    DEFAULT 'PCAF_V2',
    attribution_approach        VARCHAR(50)     DEFAULT 'PROPORTIONAL',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_pcaf_p_invested CHECK (total_invested >= 0),
    CONSTRAINT chk_p043_pcaf_p_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_p043_pcaf_p_financed CHECK (total_financed_tco2e >= 0),
    CONSTRAINT chk_p043_pcaf_p_s1 CHECK (scope1_financed_tco2e IS NULL OR scope1_financed_tco2e >= 0),
    CONSTRAINT chk_p043_pcaf_p_s2 CHECK (scope2_financed_tco2e IS NULL OR scope2_financed_tco2e >= 0),
    CONSTRAINT chk_p043_pcaf_p_s3 CHECK (scope3_financed_tco2e IS NULL OR scope3_financed_tco2e >= 0),
    CONSTRAINT chk_p043_pcaf_p_dqs CHECK (data_quality_score >= 1.0 AND data_quality_score <= 5.0),
    CONSTRAINT chk_p043_pcaf_p_assets CHECK (assets_count >= 0),
    CONSTRAINT chk_p043_pcaf_p_data CHECK (assets_with_data IS NULL OR assets_with_data >= 0),
    CONSTRAINT chk_p043_pcaf_p_coverage CHECK (
        coverage_pct IS NULL OR (coverage_pct >= 0 AND coverage_pct <= 100)
    ),
    CONSTRAINT chk_p043_pcaf_p_attribution CHECK (
        attribution_approach IS NULL OR attribution_approach IN (
            'PROPORTIONAL', 'REVENUE_BASED', 'ENTERPRISE_VALUE'
        )
    ),
    CONSTRAINT uq_p043_pcaf_p_org_portfolio_year UNIQUE (org_id, portfolio_name, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_pcaf_p_tenant         ON ghg_accounting_scope3_complete.pcaf_portfolios(tenant_id);
CREATE INDEX idx_p043_pcaf_p_org           ON ghg_accounting_scope3_complete.pcaf_portfolios(org_id);
CREATE INDEX idx_p043_pcaf_p_asset_class   ON ghg_accounting_scope3_complete.pcaf_portfolios(asset_class);
CREATE INDEX idx_p043_pcaf_p_year          ON ghg_accounting_scope3_complete.pcaf_portfolios(reporting_year);
CREATE INDEX idx_p043_pcaf_p_financed      ON ghg_accounting_scope3_complete.pcaf_portfolios(total_financed_tco2e DESC);
CREATE INDEX idx_p043_pcaf_p_invested      ON ghg_accounting_scope3_complete.pcaf_portfolios(total_invested DESC);
CREATE INDEX idx_p043_pcaf_p_dqs           ON ghg_accounting_scope3_complete.pcaf_portfolios(data_quality_score);
CREATE INDEX idx_p043_pcaf_p_created       ON ghg_accounting_scope3_complete.pcaf_portfolios(created_at DESC);

-- Composite: org + year + asset class
CREATE INDEX idx_p043_pcaf_p_org_year      ON ghg_accounting_scope3_complete.pcaf_portfolios(org_id, reporting_year DESC, asset_class);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_pcaf_p_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.pcaf_portfolios
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.pcaf_assets
-- =============================================================================
-- Individual investee-level financed emissions within a PCAF portfolio.
-- Each row is an investee with invested amount, attribution factor,
-- investee emissions, and calculated financed emissions.

CREATE TABLE ghg_accounting_scope3_complete.pcaf_assets (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    portfolio_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.pcaf_portfolios(id) ON DELETE CASCADE,
    -- Investee
    investee_name               VARCHAR(500)    NOT NULL,
    investee_id                 VARCHAR(100),
    asset_class                 ghg_accounting_scope3_complete.pcaf_asset_class NOT NULL,
    sector_naics                VARCHAR(10),
    country                     VARCHAR(3),
    -- Financial
    invested_amount             NUMERIC(18,2)   NOT NULL,
    investee_revenue            NUMERIC(18,2),
    investee_enterprise_value   NUMERIC(18,2),
    -- Attribution
    attribution_factor          DECIMAL(8,6)    NOT NULL,
    attribution_method          VARCHAR(50)     DEFAULT 'PROPORTIONAL',
    -- Investee emissions
    investee_tco2e              DECIMAL(15,3)   NOT NULL DEFAULT 0,
    investee_scope1             DECIMAL(15,3),
    investee_scope2             DECIMAL(15,3),
    investee_scope3             DECIMAL(15,3),
    -- Financed emissions
    financed_tco2e              DECIMAL(15,3)   NOT NULL DEFAULT 0,
    -- Data quality (PCAF 1-5)
    data_quality_score          INTEGER         NOT NULL DEFAULT 5,
    data_source                 VARCHAR(200),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_pcaf_a_invested CHECK (invested_amount >= 0),
    CONSTRAINT chk_p043_pcaf_a_revenue CHECK (investee_revenue IS NULL OR investee_revenue >= 0),
    CONSTRAINT chk_p043_pcaf_a_ev CHECK (investee_enterprise_value IS NULL OR investee_enterprise_value >= 0),
    CONSTRAINT chk_p043_pcaf_a_factor CHECK (attribution_factor >= 0 AND attribution_factor <= 1),
    CONSTRAINT chk_p043_pcaf_a_attribution CHECK (
        attribution_method IS NULL OR attribution_method IN (
            'PROPORTIONAL', 'REVENUE_BASED', 'ENTERPRISE_VALUE', 'BALANCE_SHEET'
        )
    ),
    CONSTRAINT chk_p043_pcaf_a_tco2e CHECK (investee_tco2e >= 0),
    CONSTRAINT chk_p043_pcaf_a_s1 CHECK (investee_scope1 IS NULL OR investee_scope1 >= 0),
    CONSTRAINT chk_p043_pcaf_a_s2 CHECK (investee_scope2 IS NULL OR investee_scope2 >= 0),
    CONSTRAINT chk_p043_pcaf_a_s3 CHECK (investee_scope3 IS NULL OR investee_scope3 >= 0),
    CONSTRAINT chk_p043_pcaf_a_financed CHECK (financed_tco2e >= 0),
    CONSTRAINT chk_p043_pcaf_a_dqs CHECK (data_quality_score >= 1 AND data_quality_score <= 5)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_pcaf_a_tenant         ON ghg_accounting_scope3_complete.pcaf_assets(tenant_id);
CREATE INDEX idx_p043_pcaf_a_portfolio      ON ghg_accounting_scope3_complete.pcaf_assets(portfolio_id);
CREATE INDEX idx_p043_pcaf_a_name           ON ghg_accounting_scope3_complete.pcaf_assets(investee_name);
CREATE INDEX idx_p043_pcaf_a_class          ON ghg_accounting_scope3_complete.pcaf_assets(asset_class);
CREATE INDEX idx_p043_pcaf_a_naics          ON ghg_accounting_scope3_complete.pcaf_assets(sector_naics);
CREATE INDEX idx_p043_pcaf_a_country        ON ghg_accounting_scope3_complete.pcaf_assets(country);
CREATE INDEX idx_p043_pcaf_a_invested       ON ghg_accounting_scope3_complete.pcaf_assets(invested_amount DESC);
CREATE INDEX idx_p043_pcaf_a_financed       ON ghg_accounting_scope3_complete.pcaf_assets(financed_tco2e DESC);
CREATE INDEX idx_p043_pcaf_a_dqs            ON ghg_accounting_scope3_complete.pcaf_assets(data_quality_score);
CREATE INDEX idx_p043_pcaf_a_created        ON ghg_accounting_scope3_complete.pcaf_assets(created_at DESC);

-- Composite: portfolio + top emitters
CREATE INDEX idx_p043_pcaf_a_port_top       ON ghg_accounting_scope3_complete.pcaf_assets(portfolio_id, financed_tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_pcaf_a_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.pcaf_assets
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.retail_logistics
-- =============================================================================
-- Retail and logistics emissions for distribution-heavy companies (Category
-- 4 upstream and Category 9 downstream transportation). Tracks delivery
-- channel, volume, distance, carrier, and emissions.

CREATE TABLE ghg_accounting_scope3_complete.retail_logistics (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Channel
    channel                     VARCHAR(100)    NOT NULL,
    direction                   VARCHAR(20)     NOT NULL DEFAULT 'DOWNSTREAM',
    -- Volume
    deliveries_count            INTEGER         NOT NULL DEFAULT 0,
    total_weight_tonnes         DECIMAL(12,3),
    total_volume_m3             DECIMAL(12,3),
    -- Distance
    avg_distance_km             DECIMAL(10,2)   NOT NULL,
    total_distance_km           DECIMAL(15,2),
    -- Carrier
    carrier                     VARCHAR(500),
    transport_mode              VARCHAR(50)     NOT NULL DEFAULT 'ROAD',
    vehicle_type                VARCHAR(100),
    fuel_type                   VARCHAR(50),
    -- Emissions
    tco2e                       DECIMAL(15,6)   NOT NULL DEFAULT 0,
    ef_source                   VARCHAR(200),
    ef_unit                     VARCHAR(50)     DEFAULT 'kgCO2e/tonne-km',
    -- Efficiency
    intensity_per_delivery      DECIMAL(12,6),
    intensity_per_tonne_km      DECIMAL(12,6),
    -- Metadata
    reporting_period            VARCHAR(20),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_rl_direction CHECK (direction IN ('UPSTREAM', 'DOWNSTREAM', 'BOTH')),
    CONSTRAINT chk_p043_rl_deliveries CHECK (deliveries_count >= 0),
    CONSTRAINT chk_p043_rl_weight CHECK (total_weight_tonnes IS NULL OR total_weight_tonnes >= 0),
    CONSTRAINT chk_p043_rl_distance CHECK (avg_distance_km >= 0),
    CONSTRAINT chk_p043_rl_total_dist CHECK (total_distance_km IS NULL OR total_distance_km >= 0),
    CONSTRAINT chk_p043_rl_mode CHECK (
        transport_mode IN ('ROAD', 'RAIL', 'SEA', 'AIR', 'INLAND_WATERWAY', 'PIPELINE', 'MULTIMODAL', 'LAST_MILE')
    ),
    CONSTRAINT chk_p043_rl_tco2e CHECK (tco2e >= 0)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_rl_tenant             ON ghg_accounting_scope3_complete.retail_logistics(tenant_id);
CREATE INDEX idx_p043_rl_inventory          ON ghg_accounting_scope3_complete.retail_logistics(inventory_id);
CREATE INDEX idx_p043_rl_channel            ON ghg_accounting_scope3_complete.retail_logistics(channel);
CREATE INDEX idx_p043_rl_direction          ON ghg_accounting_scope3_complete.retail_logistics(direction);
CREATE INDEX idx_p043_rl_carrier            ON ghg_accounting_scope3_complete.retail_logistics(carrier);
CREATE INDEX idx_p043_rl_mode              ON ghg_accounting_scope3_complete.retail_logistics(transport_mode);
CREATE INDEX idx_p043_rl_tco2e             ON ghg_accounting_scope3_complete.retail_logistics(tco2e DESC);
CREATE INDEX idx_p043_rl_created           ON ghg_accounting_scope3_complete.retail_logistics(created_at DESC);

-- Composite: inventory + mode for transport mix
CREATE INDEX idx_p043_rl_inv_mode           ON ghg_accounting_scope3_complete.retail_logistics(inventory_id, transport_mode, tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_rl_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.retail_logistics
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.circular_economy
-- =============================================================================
-- Circular economy emissions for Category 5 (Waste) and Category 12
-- (End-of-Life). Compares virgin vs recycled material emissions and
-- calculates the circular benefit (avoided emissions from recycling).

CREATE TABLE ghg_accounting_scope3_complete.circular_economy (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Material
    material                    VARCHAR(200)    NOT NULL,
    material_category           VARCHAR(200),
    quantity_tonnes             DECIMAL(12,3)   NOT NULL,
    -- Virgin emissions
    virgin_tco2e                DECIMAL(15,6)   NOT NULL DEFAULT 0,
    virgin_ef                   DECIMAL(12,6),
    virgin_ef_source            VARCHAR(200),
    -- Recycled emissions
    recycled_tco2e              DECIMAL(15,6)   NOT NULL DEFAULT 0,
    recycled_ef                 DECIMAL(12,6),
    recycled_ef_source          VARCHAR(200),
    -- Circular benefit
    circular_benefit_tco2e      DECIMAL(15,6)   GENERATED ALWAYS AS (
        virgin_tco2e - recycled_tco2e
    ) STORED,
    recycled_content_pct        DECIMAL(5,2)    NOT NULL DEFAULT 0,
    recyclability_pct           DECIMAL(5,2),
    -- End-of-life
    landfill_pct                DECIMAL(5,2)    DEFAULT 0,
    incineration_pct            DECIMAL(5,2)    DEFAULT 0,
    recycling_pct               DECIMAL(5,2)    DEFAULT 0,
    composting_pct              DECIMAL(5,2)    DEFAULT 0,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_ce_quantity CHECK (quantity_tonnes >= 0),
    CONSTRAINT chk_p043_ce_virgin CHECK (virgin_tco2e >= 0),
    CONSTRAINT chk_p043_ce_recycled CHECK (recycled_tco2e >= 0),
    CONSTRAINT chk_p043_ce_content CHECK (recycled_content_pct >= 0 AND recycled_content_pct <= 100),
    CONSTRAINT chk_p043_ce_recyclability CHECK (recyclability_pct IS NULL OR (recyclability_pct >= 0 AND recyclability_pct <= 100)),
    CONSTRAINT chk_p043_ce_landfill CHECK (landfill_pct IS NULL OR (landfill_pct >= 0 AND landfill_pct <= 100)),
    CONSTRAINT chk_p043_ce_incineration CHECK (incineration_pct IS NULL OR (incineration_pct >= 0 AND incineration_pct <= 100)),
    CONSTRAINT chk_p043_ce_recycling CHECK (recycling_pct IS NULL OR (recycling_pct >= 0 AND recycling_pct <= 100)),
    CONSTRAINT chk_p043_ce_composting CHECK (composting_pct IS NULL OR (composting_pct >= 0 AND composting_pct <= 100))
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_ce_tenant             ON ghg_accounting_scope3_complete.circular_economy(tenant_id);
CREATE INDEX idx_p043_ce_inventory          ON ghg_accounting_scope3_complete.circular_economy(inventory_id);
CREATE INDEX idx_p043_ce_material           ON ghg_accounting_scope3_complete.circular_economy(material);
CREATE INDEX idx_p043_ce_material_cat       ON ghg_accounting_scope3_complete.circular_economy(material_category);
CREATE INDEX idx_p043_ce_virgin             ON ghg_accounting_scope3_complete.circular_economy(virgin_tco2e DESC);
CREATE INDEX idx_p043_ce_benefit            ON ghg_accounting_scope3_complete.circular_economy(circular_benefit_tco2e DESC);
CREATE INDEX idx_p043_ce_recycled_pct       ON ghg_accounting_scope3_complete.circular_economy(recycled_content_pct DESC);
CREATE INDEX idx_p043_ce_created            ON ghg_accounting_scope3_complete.circular_economy(created_at DESC);

-- Composite: inventory + material for breakdown
CREATE INDEX idx_p043_ce_inv_material       ON ghg_accounting_scope3_complete.circular_economy(inventory_id, material, virgin_tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_ce_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.circular_economy
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3_complete.cloud_carbon
-- =============================================================================
-- Cloud computing carbon footprint for technology companies. Tracks cloud
-- provider, service type, usage metrics, and attributed emissions per
-- provider (AWS, Azure, GCP) methodology. Supports Category 1 (purchased
-- cloud services) and Category 8 (upstream leased assets / cloud infrastructure).

CREATE TABLE ghg_accounting_scope3_complete.cloud_carbon (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Provider
    provider                    VARCHAR(100)    NOT NULL,
    account_id                  VARCHAR(200),
    region                      VARCHAR(100),
    -- Service
    service_type                VARCHAR(100)    NOT NULL,
    service_name                VARCHAR(200),
    -- Usage
    usage_metric                VARCHAR(100)    NOT NULL,
    usage_value                 DECIMAL(15,3)   NOT NULL,
    usage_unit                  VARCHAR(50),
    -- Emissions
    tco2e                       DECIMAL(15,6)   NOT NULL DEFAULT 0,
    scope1_tco2e                DECIMAL(15,6),
    scope2_tco2e                DECIMAL(15,6),
    scope3_tco2e                DECIMAL(15,6),
    -- Energy
    energy_kwh                  DECIMAL(15,3),
    renewable_energy_pct        DECIMAL(5,2),
    -- Efficiency
    pue                         DECIMAL(4,2),
    carbon_free_energy_pct      DECIMAL(5,2),
    -- Methodology
    emission_source             VARCHAR(200),
    methodology                 VARCHAR(100),
    -- Period
    reporting_month             INTEGER,
    reporting_year              INTEGER        NOT NULL,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_cc_provider CHECK (
        provider IN ('AWS', 'AZURE', 'GCP', 'ALIBABA', 'OCI', 'IBM', 'OTHER')
    ),
    CONSTRAINT chk_p043_cc_service_type CHECK (
        service_type IN (
            'COMPUTE', 'STORAGE', 'NETWORKING', 'DATABASE', 'ANALYTICS',
            'AI_ML', 'SERVERLESS', 'CONTAINERS', 'CDN', 'OTHER'
        )
    ),
    CONSTRAINT chk_p043_cc_usage CHECK (usage_value >= 0),
    CONSTRAINT chk_p043_cc_tco2e CHECK (tco2e >= 0),
    CONSTRAINT chk_p043_cc_s1 CHECK (scope1_tco2e IS NULL OR scope1_tco2e >= 0),
    CONSTRAINT chk_p043_cc_s2 CHECK (scope2_tco2e IS NULL OR scope2_tco2e >= 0),
    CONSTRAINT chk_p043_cc_s3 CHECK (scope3_tco2e IS NULL OR scope3_tco2e >= 0),
    CONSTRAINT chk_p043_cc_energy CHECK (energy_kwh IS NULL OR energy_kwh >= 0),
    CONSTRAINT chk_p043_cc_renewable CHECK (renewable_energy_pct IS NULL OR (renewable_energy_pct >= 0 AND renewable_energy_pct <= 100)),
    CONSTRAINT chk_p043_cc_pue CHECK (pue IS NULL OR (pue >= 1.0 AND pue <= 5.0)),
    CONSTRAINT chk_p043_cc_cfe CHECK (carbon_free_energy_pct IS NULL OR (carbon_free_energy_pct >= 0 AND carbon_free_energy_pct <= 100)),
    CONSTRAINT chk_p043_cc_month CHECK (reporting_month IS NULL OR (reporting_month >= 1 AND reporting_month <= 12)),
    CONSTRAINT chk_p043_cc_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_cc_tenant             ON ghg_accounting_scope3_complete.cloud_carbon(tenant_id);
CREATE INDEX idx_p043_cc_inventory          ON ghg_accounting_scope3_complete.cloud_carbon(inventory_id);
CREATE INDEX idx_p043_cc_provider           ON ghg_accounting_scope3_complete.cloud_carbon(provider);
CREATE INDEX idx_p043_cc_service            ON ghg_accounting_scope3_complete.cloud_carbon(service_type);
CREATE INDEX idx_p043_cc_region             ON ghg_accounting_scope3_complete.cloud_carbon(region);
CREATE INDEX idx_p043_cc_tco2e              ON ghg_accounting_scope3_complete.cloud_carbon(tco2e DESC);
CREATE INDEX idx_p043_cc_energy             ON ghg_accounting_scope3_complete.cloud_carbon(energy_kwh DESC);
CREATE INDEX idx_p043_cc_year               ON ghg_accounting_scope3_complete.cloud_carbon(reporting_year);
CREATE INDEX idx_p043_cc_created            ON ghg_accounting_scope3_complete.cloud_carbon(created_at DESC);

-- Composite: inventory + provider for provider breakdown
CREATE INDEX idx_p043_cc_inv_provider       ON ghg_accounting_scope3_complete.cloud_carbon(inventory_id, provider, tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_cc_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.cloud_carbon
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.pcaf_portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.pcaf_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.retail_logistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.circular_economy ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.cloud_carbon ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_pcaf_p_tenant_isolation ON ghg_accounting_scope3_complete.pcaf_portfolios
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_pcaf_p_service_bypass ON ghg_accounting_scope3_complete.pcaf_portfolios
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_pcaf_a_tenant_isolation ON ghg_accounting_scope3_complete.pcaf_assets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_pcaf_a_service_bypass ON ghg_accounting_scope3_complete.pcaf_assets
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_rl_tenant_isolation ON ghg_accounting_scope3_complete.retail_logistics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_rl_service_bypass ON ghg_accounting_scope3_complete.retail_logistics
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_ce_tenant_isolation ON ghg_accounting_scope3_complete.circular_economy
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_ce_service_bypass ON ghg_accounting_scope3_complete.circular_economy
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_cc_tenant_isolation ON ghg_accounting_scope3_complete.cloud_carbon
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_cc_service_bypass ON ghg_accounting_scope3_complete.cloud_carbon
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.pcaf_portfolios TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.pcaf_assets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.retail_logistics TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.circular_economy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.cloud_carbon TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.pcaf_portfolios IS
    'PCAF portfolio-level financed emissions (Category 15) with total invested, financed tCO2e, WACI, and data quality score per asset class.';
COMMENT ON TABLE ghg_accounting_scope3_complete.pcaf_assets IS
    'Individual investee-level financed emissions within a PCAF portfolio with attribution factor, investee emissions, and data quality.';
COMMENT ON TABLE ghg_accounting_scope3_complete.retail_logistics IS
    'Retail and logistics emissions (Category 4/9) with delivery channel, transport mode, distance, carrier, and per-tonne-km intensity.';
COMMENT ON TABLE ghg_accounting_scope3_complete.circular_economy IS
    'Circular economy module (Category 5/12) comparing virgin vs recycled material emissions with circular benefit calculation.';
COMMENT ON TABLE ghg_accounting_scope3_complete.cloud_carbon IS
    'Cloud computing carbon footprint per provider (AWS/Azure/GCP) with service type, usage metrics, PUE, and renewable energy percentage.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.pcaf_portfolios.waci IS 'Weighted Average Carbon Intensity per PCAF methodology.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.pcaf_portfolios.data_quality_score IS 'PCAF data quality score (1 = best, 5 = worst) averaged across all assets.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.pcaf_assets.attribution_factor IS 'Share of investee emissions attributed to investor: invested / (equity + debt).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.circular_economy.circular_benefit_tco2e IS 'Generated column: virgin_tco2e - recycled_tco2e (positive = avoided emissions).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.cloud_carbon.pue IS 'Power Usage Effectiveness (1.0 = perfectly efficient, typical data center: 1.2-1.6).';
