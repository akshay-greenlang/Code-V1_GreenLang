-- =============================================================================
-- V336: PACK-042 Scope 3 Starter Pack - Core Schema & Organization Configuration
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_accounting_scope3 schema and foundational tables for GHG
-- Protocol Scope 3 (Corporate Value Chain) inventory management. Establishes
-- the inventory header, per-category configuration, screening results, and
-- organization profiles. Implements GHG Protocol Scope 3 Standard requirements
-- for value chain boundary setting, category relevance screening, and
-- methodology tier selection across all 15 Scope 3 categories.
--
-- Tables (4):
--   1. ghg_accounting_scope3.scope3_inventories
--   2. ghg_accounting_scope3.category_configurations
--   3. ghg_accounting_scope3.screening_results
--   4. ghg_accounting_scope3.organization_profiles
--
-- Enums (3):
--   1. ghg_accounting_scope3.scope3_category_type
--   2. ghg_accounting_scope3.methodology_tier_type
--   3. ghg_accounting_scope3.inventory_status_type
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V335__pack041_views_indexes_seed.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_accounting_scope3;

SET search_path TO ghg_accounting_scope3, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_accounting_scope3.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: scope3_category_type
-- ---------------------------------------------------------------------------
-- The 15 categories of Scope 3 emissions per GHG Protocol Corporate Value
-- Chain (Scope 3) Accounting and Reporting Standard.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'scope3_category_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3')) THEN
        CREATE TYPE ghg_accounting_scope3.scope3_category_type AS ENUM (
            'CAT_1',   -- Purchased Goods and Services
            'CAT_2',   -- Capital Goods
            'CAT_3',   -- Fuel- and Energy-Related Activities
            'CAT_4',   -- Upstream Transportation and Distribution
            'CAT_5',   -- Waste Generated in Operations
            'CAT_6',   -- Business Travel
            'CAT_7',   -- Employee Commuting
            'CAT_8',   -- Upstream Leased Assets
            'CAT_9',   -- Downstream Transportation and Distribution
            'CAT_10',  -- Processing of Sold Products
            'CAT_11',  -- Use of Sold Products
            'CAT_12',  -- End-of-Life Treatment of Sold Products
            'CAT_13',  -- Downstream Leased Assets
            'CAT_14',  -- Franchises
            'CAT_15'   -- Investments
        );
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: methodology_tier_type
-- ---------------------------------------------------------------------------
-- Methodology tier classifications for Scope 3 calculations. Ranges from
-- least specific (SPEND_BASED) to most specific (SUPPLIER_SPECIFIC).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'methodology_tier_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3')) THEN
        CREATE TYPE ghg_accounting_scope3.methodology_tier_type AS ENUM (
            'SPEND_BASED',        -- Tier 1: Spend-based EEIO method
            'AVERAGE_DATA',       -- Tier 2: Average-data method (activity data x secondary EFs)
            'SUPPLIER_SPECIFIC',  -- Tier 3: Supplier-specific primary data
            'HYBRID'              -- Mixed approach combining multiple tiers
        );
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: inventory_status_type
-- ---------------------------------------------------------------------------
-- Lifecycle status of a Scope 3 inventory from creation through publication.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'inventory_status_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3')) THEN
        CREATE TYPE ghg_accounting_scope3.inventory_status_type AS ENUM (
            'DRAFT',
            'IN_PROGRESS',
            'REVIEW',
            'VERIFIED',
            'PUBLISHED'
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.scope3_inventories
-- =============================================================================
-- Top-level Scope 3 inventory header. Each record represents a single
-- reporting period inventory for an organization. Tracks methodology version,
-- boundary approach (operational control or equity share), and overall
-- inventory status. An organization typically has one inventory per
-- reporting year, but may maintain draft and finalized versions.

CREATE TABLE ghg_accounting_scope3.scope3_inventories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    -- Reporting period
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    -- Methodology
    methodology_version         VARCHAR(50)     NOT NULL DEFAULT 'GHG_PROTOCOL_SCOPE3_V1',
    boundary_approach           VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    base_year                   INTEGER,
    -- Status
    status                      ghg_accounting_scope3.inventory_status_type NOT NULL DEFAULT 'DRAFT',
    -- Totals (denormalized for quick access)
    total_scope3_tco2e          DECIMAL(15,3),
    categories_assessed         INTEGER         DEFAULT 0,
    categories_relevant         INTEGER         DEFAULT 0,
    categories_calculated       INTEGER         DEFAULT 0,
    -- Screening
    screening_completed         BOOLEAN         NOT NULL DEFAULT false,
    screening_date              TIMESTAMPTZ,
    -- Metadata
    description                 TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p042_si_period CHECK (
        reporting_period_start < reporting_period_end
    ),
    CONSTRAINT chk_p042_si_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p042_si_boundary CHECK (
        boundary_approach IN (
            'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
        )
    ),
    CONSTRAINT chk_p042_si_gwp CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p042_si_methodology CHECK (
        methodology_version IN (
            'GHG_PROTOCOL_SCOPE3_V1', 'ISO_14064_1_2018', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p042_si_base_year CHECK (
        base_year IS NULL OR (base_year >= 1990 AND base_year <= 2100)
    ),
    CONSTRAINT chk_p042_si_total CHECK (
        total_scope3_tco2e IS NULL OR total_scope3_tco2e >= 0
    ),
    CONSTRAINT chk_p042_si_categories_assessed CHECK (
        categories_assessed >= 0 AND categories_assessed <= 15
    ),
    CONSTRAINT chk_p042_si_categories_relevant CHECK (
        categories_relevant >= 0 AND categories_relevant <= 15
    ),
    CONSTRAINT chk_p042_si_categories_calculated CHECK (
        categories_calculated >= 0 AND categories_calculated <= 15
    ),
    CONSTRAINT uq_p042_si_tenant_org_year UNIQUE (tenant_id, org_id, reporting_year, status)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_si_tenant             ON ghg_accounting_scope3.scope3_inventories(tenant_id);
CREATE INDEX idx_p042_si_org               ON ghg_accounting_scope3.scope3_inventories(org_id);
CREATE INDEX idx_p042_si_year              ON ghg_accounting_scope3.scope3_inventories(reporting_year);
CREATE INDEX idx_p042_si_status            ON ghg_accounting_scope3.scope3_inventories(status);
CREATE INDEX idx_p042_si_boundary          ON ghg_accounting_scope3.scope3_inventories(boundary_approach);
CREATE INDEX idx_p042_si_gwp               ON ghg_accounting_scope3.scope3_inventories(gwp_source);
CREATE INDEX idx_p042_si_screening         ON ghg_accounting_scope3.scope3_inventories(screening_completed);
CREATE INDEX idx_p042_si_created           ON ghg_accounting_scope3.scope3_inventories(created_at DESC);
CREATE INDEX idx_p042_si_metadata          ON ghg_accounting_scope3.scope3_inventories USING GIN(metadata);

-- Composite: tenant + org + year for inventory lookup
CREATE INDEX idx_p042_si_tenant_org_year   ON ghg_accounting_scope3.scope3_inventories(tenant_id, org_id, reporting_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_si_updated
    BEFORE UPDATE ON ghg_accounting_scope3.scope3_inventories
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.category_configurations
-- =============================================================================
-- Per-category configuration within a Scope 3 inventory. Determines which
-- of the 15 categories are enabled, the methodology tier to use, materiality
-- thresholds, and data source preferences. This table drives the calculation
-- engine by telling each category agent how to operate.

CREATE TABLE ghg_accounting_scope3.category_configurations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Configuration
    enabled                     BOOLEAN         NOT NULL DEFAULT true,
    methodology_tier            ghg_accounting_scope3.methodology_tier_type NOT NULL DEFAULT 'SPEND_BASED',
    materiality_threshold_pct   DECIMAL(5,2)    DEFAULT 1.00,
    -- Data sources
    data_sources                JSONB           DEFAULT '[]',
    primary_data_pct            DECIMAL(5,2)    DEFAULT 0.00,
    -- Calculation parameters
    emission_factor_source      VARCHAR(100)    DEFAULT 'EEIO_EXIOBASE',
    emission_factor_year        INTEGER,
    custom_parameters           JSONB           DEFAULT '{}',
    -- Notes
    exclusion_reason            TEXT,
    methodology_justification   TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_cc_materiality CHECK (
        materiality_threshold_pct IS NULL OR (materiality_threshold_pct >= 0 AND materiality_threshold_pct <= 100)
    ),
    CONSTRAINT chk_p042_cc_primary_data CHECK (
        primary_data_pct >= 0 AND primary_data_pct <= 100
    ),
    CONSTRAINT chk_p042_cc_ef_year CHECK (
        emission_factor_year IS NULL OR (emission_factor_year >= 1990 AND emission_factor_year <= 2100)
    ),
    CONSTRAINT uq_p042_cc_inventory_category UNIQUE (inventory_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_cc_tenant             ON ghg_accounting_scope3.category_configurations(tenant_id);
CREATE INDEX idx_p042_cc_inventory          ON ghg_accounting_scope3.category_configurations(inventory_id);
CREATE INDEX idx_p042_cc_category           ON ghg_accounting_scope3.category_configurations(category);
CREATE INDEX idx_p042_cc_enabled            ON ghg_accounting_scope3.category_configurations(enabled) WHERE enabled = true;
CREATE INDEX idx_p042_cc_tier               ON ghg_accounting_scope3.category_configurations(methodology_tier);
CREATE INDEX idx_p042_cc_created            ON ghg_accounting_scope3.category_configurations(created_at DESC);
CREATE INDEX idx_p042_cc_metadata           ON ghg_accounting_scope3.category_configurations USING GIN(metadata);
CREATE INDEX idx_p042_cc_data_sources       ON ghg_accounting_scope3.category_configurations USING GIN(data_sources);

-- Composite: inventory + enabled categories
CREATE INDEX idx_p042_cc_inv_enabled        ON ghg_accounting_scope3.category_configurations(inventory_id, category)
    WHERE enabled = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_cc_updated
    BEFORE UPDATE ON ghg_accounting_scope3.category_configurations
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.screening_results
-- =============================================================================
-- Category screening output per GHG Protocol Scope 3 Standard Chapter 7.
-- For each of the 15 categories, estimates the magnitude of emissions,
-- determines relevance, recommends a methodology tier, and calculates the
-- category's significance as a percentage of total estimated Scope 3.
-- Screening drives the prioritization of which categories to calculate first.

CREATE TABLE ghg_accounting_scope3.screening_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Screening output
    estimated_tco2e             DECIMAL(15,3)   NOT NULL DEFAULT 0,
    relevance_flag              BOOLEAN         NOT NULL DEFAULT true,
    recommended_tier            ghg_accounting_scope3.methodology_tier_type NOT NULL DEFAULT 'SPEND_BASED',
    significance_pct            DECIMAL(5,2)    NOT NULL DEFAULT 0,
    -- Screening criteria
    size_relevance              BOOLEAN         DEFAULT true,
    influence_relevance         BOOLEAN         DEFAULT false,
    risk_relevance              BOOLEAN         DEFAULT false,
    stakeholder_relevance       BOOLEAN         DEFAULT false,
    outsourcing_relevance       BOOLEAN         DEFAULT false,
    sector_guidance_relevance   BOOLEAN         DEFAULT false,
    -- Data availability
    data_availability           VARCHAR(30)     DEFAULT 'PARTIAL',
    data_quality_expectation    VARCHAR(20)     DEFAULT 'MODERATE',
    -- Estimation method
    estimation_method           VARCHAR(50)     NOT NULL DEFAULT 'INDUSTRY_AVERAGE',
    estimation_source           VARCHAR(200),
    estimation_confidence       DECIMAL(3,2)    DEFAULT 0.50,
    -- Notes
    screening_notes             TEXT,
    exclusion_justification     TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_sr_estimated CHECK (
        estimated_tco2e >= 0
    ),
    CONSTRAINT chk_p042_sr_significance CHECK (
        significance_pct >= 0 AND significance_pct <= 100
    ),
    CONSTRAINT chk_p042_sr_data_avail CHECK (
        data_availability IS NULL OR data_availability IN (
            'AVAILABLE', 'PARTIAL', 'ESTIMATED', 'NOT_AVAILABLE', 'PLANNED'
        )
    ),
    CONSTRAINT chk_p042_sr_data_quality CHECK (
        data_quality_expectation IS NULL OR data_quality_expectation IN (
            'HIGH', 'MODERATE', 'LOW', 'VERY_LOW'
        )
    ),
    CONSTRAINT chk_p042_sr_estimation_method CHECK (
        estimation_method IN (
            'INDUSTRY_AVERAGE', 'REVENUE_BASED', 'SPEND_BASED',
            'HEADCOUNT_BASED', 'SECTOR_BENCHMARK', 'PRIOR_YEAR',
            'EXPERT_JUDGMENT', 'SUPPLIER_DATA', 'OTHER'
        )
    ),
    CONSTRAINT chk_p042_sr_confidence CHECK (
        estimation_confidence IS NULL OR (estimation_confidence >= 0 AND estimation_confidence <= 1)
    ),
    CONSTRAINT uq_p042_sr_inventory_category UNIQUE (inventory_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_sr_tenant             ON ghg_accounting_scope3.screening_results(tenant_id);
CREATE INDEX idx_p042_sr_inventory          ON ghg_accounting_scope3.screening_results(inventory_id);
CREATE INDEX idx_p042_sr_category           ON ghg_accounting_scope3.screening_results(category);
CREATE INDEX idx_p042_sr_relevance          ON ghg_accounting_scope3.screening_results(relevance_flag) WHERE relevance_flag = true;
CREATE INDEX idx_p042_sr_significance       ON ghg_accounting_scope3.screening_results(significance_pct DESC);
CREATE INDEX idx_p042_sr_tier               ON ghg_accounting_scope3.screening_results(recommended_tier);
CREATE INDEX idx_p042_sr_created            ON ghg_accounting_scope3.screening_results(created_at DESC);

-- Composite: inventory + relevant + significance for prioritization
CREATE INDEX idx_p042_sr_inv_relevant       ON ghg_accounting_scope3.screening_results(inventory_id, significance_pct DESC)
    WHERE relevance_flag = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_sr_updated
    BEFORE UPDATE ON ghg_accounting_scope3.screening_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.organization_profiles
-- =============================================================================
-- Organization metadata relevant to Scope 3 calculations. Stores sector
-- classification (NAICS), financial data, employee count, facility count,
-- and product type information. This profile drives screening heuristics,
-- sector benchmarking, and spend-based emission factor selection.

CREATE TABLE ghg_accounting_scope3.organization_profiles (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    -- Sector classification
    sector_naics                VARCHAR(10)     NOT NULL,
    sector_naics_description    VARCHAR(500),
    sector_sic                  VARCHAR(10),
    sector_gics                 VARCHAR(10),
    sector_nace                 VARCHAR(10),
    industry_description        TEXT,
    -- Financial data
    annual_revenue              NUMERIC(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    annual_procurement_spend    NUMERIC(18,2),
    procurement_currency        VARCHAR(3)      DEFAULT 'USD',
    capital_expenditure         NUMERIC(18,2),
    -- Organizational data
    employee_count              INTEGER,
    facility_count              INTEGER,
    operating_countries         TEXT[],
    -- Product information
    product_types               JSONB           DEFAULT '[]',
    primary_products            TEXT[],
    sells_physical_products     BOOLEAN         DEFAULT true,
    sells_services              BOOLEAN         DEFAULT false,
    -- Value chain characteristics
    supply_chain_complexity     VARCHAR(20)     DEFAULT 'MODERATE',
    tier1_supplier_count        INTEGER,
    has_franchises              BOOLEAN         DEFAULT false,
    has_leased_assets           BOOLEAN         DEFAULT false,
    has_investments             BOOLEAN         DEFAULT false,
    -- Profile metadata
    profile_year                INTEGER         NOT NULL,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p042_op_revenue CHECK (
        annual_revenue IS NULL OR annual_revenue >= 0
    ),
    CONSTRAINT chk_p042_op_procurement CHECK (
        annual_procurement_spend IS NULL OR annual_procurement_spend >= 0
    ),
    CONSTRAINT chk_p042_op_capex CHECK (
        capital_expenditure IS NULL OR capital_expenditure >= 0
    ),
    CONSTRAINT chk_p042_op_employees CHECK (
        employee_count IS NULL OR employee_count >= 0
    ),
    CONSTRAINT chk_p042_op_facilities CHECK (
        facility_count IS NULL OR facility_count >= 0
    ),
    CONSTRAINT chk_p042_op_suppliers CHECK (
        tier1_supplier_count IS NULL OR tier1_supplier_count >= 0
    ),
    CONSTRAINT chk_p042_op_profile_year CHECK (
        profile_year >= 1990 AND profile_year <= 2100
    ),
    CONSTRAINT chk_p042_op_complexity CHECK (
        supply_chain_complexity IS NULL OR supply_chain_complexity IN (
            'SIMPLE', 'MODERATE', 'COMPLEX', 'VERY_COMPLEX'
        )
    ),
    CONSTRAINT uq_p042_op_tenant_org_year UNIQUE (tenant_id, org_id, profile_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_op_tenant             ON ghg_accounting_scope3.organization_profiles(tenant_id);
CREATE INDEX idx_p042_op_org               ON ghg_accounting_scope3.organization_profiles(org_id);
CREATE INDEX idx_p042_op_naics             ON ghg_accounting_scope3.organization_profiles(sector_naics);
CREATE INDEX idx_p042_op_year              ON ghg_accounting_scope3.organization_profiles(profile_year);
CREATE INDEX idx_p042_op_current           ON ghg_accounting_scope3.organization_profiles(is_current) WHERE is_current = true;
CREATE INDEX idx_p042_op_revenue           ON ghg_accounting_scope3.organization_profiles(annual_revenue DESC);
CREATE INDEX idx_p042_op_employees         ON ghg_accounting_scope3.organization_profiles(employee_count);
CREATE INDEX idx_p042_op_created           ON ghg_accounting_scope3.organization_profiles(created_at DESC);
CREATE INDEX idx_p042_op_metadata          ON ghg_accounting_scope3.organization_profiles USING GIN(metadata);
CREATE INDEX idx_p042_op_products          ON ghg_accounting_scope3.organization_profiles USING GIN(product_types);

-- Composite: tenant + org + current profile
CREATE INDEX idx_p042_op_tenant_org_current ON ghg_accounting_scope3.organization_profiles(tenant_id, org_id)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_op_updated
    BEFORE UPDATE ON ghg_accounting_scope3.organization_profiles
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.scope3_inventories ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.category_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.screening_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.organization_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_si_tenant_isolation
    ON ghg_accounting_scope3.scope3_inventories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_si_service_bypass
    ON ghg_accounting_scope3.scope3_inventories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_cc_tenant_isolation
    ON ghg_accounting_scope3.category_configurations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_cc_service_bypass
    ON ghg_accounting_scope3.category_configurations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_sr_tenant_isolation
    ON ghg_accounting_scope3.screening_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_sr_service_bypass
    ON ghg_accounting_scope3.screening_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_op_tenant_isolation
    ON ghg_accounting_scope3.organization_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_op_service_bypass
    ON ghg_accounting_scope3.organization_profiles
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_accounting_scope3 TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.scope3_inventories TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.category_configurations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.screening_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.organization_profiles TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_accounting_scope3.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_accounting_scope3 IS
    'PACK-042 Scope 3 Starter Pack - GHG Protocol Scope 3 (Corporate Value Chain) inventory management including spend classification, category calculations, double-counting prevention, hotspot analysis, supplier engagement, data quality, uncertainty, and compliance reporting.';

COMMENT ON TABLE ghg_accounting_scope3.scope3_inventories IS
    'Top-level Scope 3 inventory header tracking reporting period, methodology, boundary approach, and overall status per GHG Protocol Scope 3 Standard.';
COMMENT ON TABLE ghg_accounting_scope3.category_configurations IS
    'Per-category configuration within a Scope 3 inventory defining enabled status, methodology tier, materiality thresholds, and data source preferences.';
COMMENT ON TABLE ghg_accounting_scope3.screening_results IS
    'Category relevance screening output per GHG Protocol Chapter 7 with estimated magnitude, relevance criteria, and recommended methodology tier.';
COMMENT ON TABLE ghg_accounting_scope3.organization_profiles IS
    'Organization metadata for Scope 3 calculations including sector classification (NAICS), financials, employee count, and value chain characteristics.';

COMMENT ON COLUMN ghg_accounting_scope3.scope3_inventories.id IS 'Unique identifier for the Scope 3 inventory.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_inventories.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_inventories.org_id IS 'Reference to the reporting organization.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_inventories.methodology_version IS 'Standard used: GHG_PROTOCOL_SCOPE3_V1, ISO_14064_1_2018, or CUSTOM.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_inventories.boundary_approach IS 'Organizational boundary: OPERATIONAL_CONTROL, FINANCIAL_CONTROL, or EQUITY_SHARE.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_inventories.gwp_source IS 'IPCC Assessment Report for GWP conversion: AR4, AR5, AR6, SAR, TAR.';
COMMENT ON COLUMN ghg_accounting_scope3.scope3_inventories.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN ghg_accounting_scope3.category_configurations.category IS 'Scope 3 category (CAT_1 through CAT_15) per GHG Protocol.';
COMMENT ON COLUMN ghg_accounting_scope3.category_configurations.methodology_tier IS 'Calculation methodology: SPEND_BASED, AVERAGE_DATA, SUPPLIER_SPECIFIC, or HYBRID.';
COMMENT ON COLUMN ghg_accounting_scope3.category_configurations.data_sources IS 'JSONB array of data source configurations for this category.';

COMMENT ON COLUMN ghg_accounting_scope3.screening_results.relevance_flag IS 'Whether this category is relevant for the organization based on screening criteria.';
COMMENT ON COLUMN ghg_accounting_scope3.screening_results.significance_pct IS 'Category significance as percentage of total estimated Scope 3 emissions.';
COMMENT ON COLUMN ghg_accounting_scope3.screening_results.size_relevance IS 'Size criterion: category contributes significantly to total Scope 3 by magnitude.';
COMMENT ON COLUMN ghg_accounting_scope3.screening_results.influence_relevance IS 'Influence criterion: company can influence emission reductions in this category.';

COMMENT ON COLUMN ghg_accounting_scope3.organization_profiles.sector_naics IS 'North American Industry Classification System (NAICS) code.';
COMMENT ON COLUMN ghg_accounting_scope3.organization_profiles.supply_chain_complexity IS 'Value chain complexity: SIMPLE, MODERATE, COMPLEX, VERY_COMPLEX.';
COMMENT ON COLUMN ghg_accounting_scope3.organization_profiles.product_types IS 'JSONB array of product/service types offered by the organization.';
