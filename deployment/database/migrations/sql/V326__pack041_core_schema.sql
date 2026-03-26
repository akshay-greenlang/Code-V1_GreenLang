-- =============================================================================
-- V326: PACK-041 Scope 1-2 Complete Pack - Core Schema & Organizational Tables
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_scope12 schema and foundational organizational tables for
-- complete GHG Protocol Scope 1 and Scope 2 inventory management. Tracks
-- organizations, legal entities, facilities, organizational boundaries,
-- boundary inclusions, and source category applicability. These tables
-- implement the GHG Protocol Corporate Standard requirements for
-- organizational boundary setting and completeness assessment.
--
-- Tables (6):
--   1. ghg_scope12.organizations
--   2. ghg_scope12.legal_entities
--   3. ghg_scope12.facilities
--   4. ghg_scope12.organizational_boundaries
--   5. ghg_scope12.boundary_inclusions
--   6. ghg_scope12.source_categories
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V325__pack040_mv_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_scope12;

SET search_path TO ghg_scope12, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_scope12.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: ghg_scope12.organizations
-- =============================================================================
-- Top-level reporting organization. Represents the corporate entity that
-- prepares the GHG inventory. Tracks the consolidation approach (equity share,
-- operational control, or financial control) per GHG Protocol Chapter 3.
-- An organization may own multiple legal entities and facilities.

CREATE TABLE ghg_scope12.organizations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    name                        VARCHAR(500)    NOT NULL,
    legal_name                  VARCHAR(500),
    sector                      VARCHAR(100)    NOT NULL,
    sub_sector                  VARCHAR(100),
    industry_code               VARCHAR(20),
    industry_code_system        VARCHAR(20)     DEFAULT 'NACE',
    country                     VARCHAR(3)      NOT NULL,
    headquarters_address        TEXT,
    website                     VARCHAR(500),
    fiscal_year_end_month       INTEGER         NOT NULL DEFAULT 12,
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    employee_count              INTEGER,
    annual_revenue              NUMERIC(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    is_publicly_listed          BOOLEAN         NOT NULL DEFAULT false,
    stock_exchange              VARCHAR(50),
    ticker_symbol               VARCHAR(20),
    lei_code                    VARCHAR(20),
    reporting_start_year        INTEGER,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_org_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p041_org_consolidation CHECK (
        consolidation_approach IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL'
        )
    ),
    CONSTRAINT chk_p041_org_status CHECK (
        status IN ('ACTIVE', 'INACTIVE', 'MERGED', 'ACQUIRED', 'DISSOLVED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p041_org_industry_system CHECK (
        industry_code_system IS NULL OR industry_code_system IN (
            'NACE', 'NAICS', 'SIC', 'ISIC', 'GICS', 'ICB'
        )
    ),
    CONSTRAINT chk_p041_org_fiscal_month CHECK (
        fiscal_year_end_month >= 1 AND fiscal_year_end_month <= 12
    ),
    CONSTRAINT chk_p041_org_employee_count CHECK (
        employee_count IS NULL OR employee_count >= 0
    ),
    CONSTRAINT chk_p041_org_revenue CHECK (
        annual_revenue IS NULL OR annual_revenue >= 0
    ),
    CONSTRAINT chk_p041_org_start_year CHECK (
        reporting_start_year IS NULL OR (reporting_start_year >= 1990 AND reporting_start_year <= 2100)
    ),
    CONSTRAINT uq_p041_org_tenant_name UNIQUE (tenant_id, name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_org_tenant            ON ghg_scope12.organizations(tenant_id);
CREATE INDEX idx_p041_org_name              ON ghg_scope12.organizations(name);
CREATE INDEX idx_p041_org_sector            ON ghg_scope12.organizations(sector);
CREATE INDEX idx_p041_org_country           ON ghg_scope12.organizations(country);
CREATE INDEX idx_p041_org_consolidation     ON ghg_scope12.organizations(consolidation_approach);
CREATE INDEX idx_p041_org_status            ON ghg_scope12.organizations(status);
CREATE INDEX idx_p041_org_industry          ON ghg_scope12.organizations(industry_code);
CREATE INDEX idx_p041_org_created           ON ghg_scope12.organizations(created_at DESC);
CREATE INDEX idx_p041_org_metadata          ON ghg_scope12.organizations USING GIN(metadata);

-- Composite: tenant + active organizations
CREATE INDEX idx_p041_org_tenant_active     ON ghg_scope12.organizations(tenant_id, name)
    WHERE status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_org_updated
    BEFORE UPDATE ON ghg_scope12.organizations
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.legal_entities
-- =============================================================================
-- Legal entities (subsidiaries, joint ventures, associates) that compose the
-- reporting organization. Tracks ownership percentage and control type per
-- GHG Protocol organizational boundary requirements. The ownership_pct and
-- control flags determine how emissions are consolidated into the org total.

CREATE TABLE ghg_scope12.legal_entities (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    name                        VARCHAR(500)    NOT NULL,
    legal_name                  VARCHAR(500),
    entity_type                 VARCHAR(30)     NOT NULL DEFAULT 'SUBSIDIARY',
    ownership_pct               DECIMAL(5,2)    NOT NULL DEFAULT 100.00,
    has_operational_control     BOOLEAN         NOT NULL DEFAULT true,
    has_financial_control       BOOLEAN         NOT NULL DEFAULT true,
    country                     VARCHAR(3)      NOT NULL,
    registration_number         VARCHAR(100),
    sector                      VARCHAR(100),
    employee_count              INTEGER,
    annual_revenue              NUMERIC(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    acquisition_date            DATE,
    divestiture_date            DATE,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_le_entity_type CHECK (
        entity_type IN (
            'SUBSIDIARY', 'JOINT_VENTURE', 'ASSOCIATE', 'PARTNERSHIP',
            'FRANCHISE', 'BRANCH', 'DIVISION', 'HOLDING', 'SPV', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_le_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p041_le_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p041_le_employee_count CHECK (
        employee_count IS NULL OR employee_count >= 0
    ),
    CONSTRAINT chk_p041_le_revenue CHECK (
        annual_revenue IS NULL OR annual_revenue >= 0
    ),
    CONSTRAINT chk_p041_le_dates CHECK (
        acquisition_date IS NULL OR divestiture_date IS NULL OR
        acquisition_date <= divestiture_date
    ),
    CONSTRAINT uq_p041_le_tenant_org_name UNIQUE (tenant_id, organization_id, name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_le_tenant             ON ghg_scope12.legal_entities(tenant_id);
CREATE INDEX idx_p041_le_org               ON ghg_scope12.legal_entities(organization_id);
CREATE INDEX idx_p041_le_name              ON ghg_scope12.legal_entities(name);
CREATE INDEX idx_p041_le_entity_type       ON ghg_scope12.legal_entities(entity_type);
CREATE INDEX idx_p041_le_country           ON ghg_scope12.legal_entities(country);
CREATE INDEX idx_p041_le_ownership         ON ghg_scope12.legal_entities(ownership_pct);
CREATE INDEX idx_p041_le_active            ON ghg_scope12.legal_entities(is_active) WHERE is_active = true;
CREATE INDEX idx_p041_le_created           ON ghg_scope12.legal_entities(created_at DESC);
CREATE INDEX idx_p041_le_metadata          ON ghg_scope12.legal_entities USING GIN(metadata);

-- Composite: org + active entities
CREATE INDEX idx_p041_le_org_active        ON ghg_scope12.legal_entities(organization_id, entity_type)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_le_updated
    BEFORE UPDATE ON ghg_scope12.legal_entities
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.facilities
-- =============================================================================
-- Physical facilities (sites, plants, offices, warehouses) where emissions
-- occur. Each facility belongs to a legal entity and may have multiple emission
-- source categories. Captures physical characteristics needed for intensity
-- metrics (floor area, headcount, operating hours, production output).

CREATE TABLE ghg_scope12.facilities (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    entity_id                   UUID            NOT NULL REFERENCES ghg_scope12.legal_entities(id) ON DELETE CASCADE,
    name                        VARCHAR(500)    NOT NULL,
    facility_code               VARCHAR(50),
    facility_type               VARCHAR(50)     NOT NULL DEFAULT 'OFFICE',
    address                     TEXT,
    city                        VARCHAR(200),
    state_province              VARCHAR(200),
    postal_code                 VARCHAR(20),
    country                     VARCHAR(3)      NOT NULL,
    latitude                    NUMERIC(10,7),
    longitude                   NUMERIC(10,7),
    sector                      VARCHAR(100),
    sub_sector                  VARCHAR(100),
    floor_area_m2               DECIMAL(14,2),
    headcount                   INTEGER,
    operating_hours             INTEGER,
    production_output           NUMERIC(18,3),
    production_unit             VARCHAR(50),
    annual_revenue              NUMERIC(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    grid_region                 VARCHAR(100),
    electricity_provider        VARCHAR(255),
    gas_provider                VARCHAR(255),
    is_leased                   BOOLEAN         NOT NULL DEFAULT false,
    lease_type                  VARCHAR(30),
    commissioning_year          INTEGER,
    decommissioning_year        INTEGER,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_fac_type CHECK (
        facility_type IN (
            'OFFICE', 'MANUFACTURING', 'WAREHOUSE', 'RETAIL', 'DATA_CENTER',
            'LABORATORY', 'HOSPITAL', 'UNIVERSITY', 'HOTEL', 'COLD_STORAGE',
            'REFINERY', 'POWER_PLANT', 'MINE', 'FARM', 'TRANSPORT_HUB',
            'WATER_TREATMENT', 'MIXED_USE', 'CAMPUS', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_fac_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p041_fac_latitude CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    ),
    CONSTRAINT chk_p041_fac_longitude CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    ),
    CONSTRAINT chk_p041_fac_floor_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT chk_p041_fac_headcount CHECK (
        headcount IS NULL OR headcount >= 0
    ),
    CONSTRAINT chk_p041_fac_operating_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8784)
    ),
    CONSTRAINT chk_p041_fac_production CHECK (
        production_output IS NULL OR production_output >= 0
    ),
    CONSTRAINT chk_p041_fac_lease_type CHECK (
        lease_type IS NULL OR lease_type IN (
            'FINANCE_LEASE', 'OPERATING_LEASE', 'OWNED', 'SUBLEASE'
        )
    ),
    CONSTRAINT chk_p041_fac_commission_year CHECK (
        commissioning_year IS NULL OR (commissioning_year >= 1800 AND commissioning_year <= 2100)
    ),
    CONSTRAINT chk_p041_fac_decommission_year CHECK (
        decommissioning_year IS NULL OR (decommissioning_year >= 1800 AND decommissioning_year <= 2100)
    ),
    CONSTRAINT chk_p041_fac_year_order CHECK (
        commissioning_year IS NULL OR decommissioning_year IS NULL OR
        commissioning_year <= decommissioning_year
    ),
    CONSTRAINT uq_p041_fac_tenant_code UNIQUE (tenant_id, facility_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_fac_tenant            ON ghg_scope12.facilities(tenant_id);
CREATE INDEX idx_p041_fac_entity            ON ghg_scope12.facilities(entity_id);
CREATE INDEX idx_p041_fac_name              ON ghg_scope12.facilities(name);
CREATE INDEX idx_p041_fac_code              ON ghg_scope12.facilities(facility_code);
CREATE INDEX idx_p041_fac_type              ON ghg_scope12.facilities(facility_type);
CREATE INDEX idx_p041_fac_country           ON ghg_scope12.facilities(country);
CREATE INDEX idx_p041_fac_sector            ON ghg_scope12.facilities(sector);
CREATE INDEX idx_p041_fac_grid_region       ON ghg_scope12.facilities(grid_region);
CREATE INDEX idx_p041_fac_active            ON ghg_scope12.facilities(is_active) WHERE is_active = true;
CREATE INDEX idx_p041_fac_created           ON ghg_scope12.facilities(created_at DESC);
CREATE INDEX idx_p041_fac_metadata          ON ghg_scope12.facilities USING GIN(metadata);

-- Composite: entity + active facilities
CREATE INDEX idx_p041_fac_entity_active     ON ghg_scope12.facilities(entity_id, facility_type)
    WHERE is_active = true;

-- Composite: tenant + country for geographic queries
CREATE INDEX idx_p041_fac_tenant_country    ON ghg_scope12.facilities(tenant_id, country);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_fac_updated
    BEFORE UPDATE ON ghg_scope12.facilities
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_scope12.organizational_boundaries
-- =============================================================================
-- Defines the organizational boundary for a reporting year per GHG Protocol
-- Chapter 3. Each boundary specifies which entities and facilities are included,
-- the consolidation approach used, and the approval status. Boundaries may be
-- versioned when structural changes (M&A, divestiture) require re-statement.

CREATE TABLE ghg_scope12.organizational_boundaries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    reporting_year              INTEGER         NOT NULL,
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    boundary_version            INTEGER         NOT NULL DEFAULT 1,
    boundary_description        TEXT,
    total_entities_included     INTEGER,
    total_facilities_included   INTEGER,
    structural_changes          TEXT,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approved_at                 TIMESTAMPTZ,
    approved_by                 VARCHAR(255),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_ob_consolidation CHECK (
        consolidation_approach IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL'
        )
    ),
    CONSTRAINT chk_p041_ob_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p041_ob_version CHECK (
        boundary_version >= 1
    ),
    CONSTRAINT chk_p041_ob_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p041_ob_entities CHECK (
        total_entities_included IS NULL OR total_entities_included >= 0
    ),
    CONSTRAINT chk_p041_ob_facilities CHECK (
        total_facilities_included IS NULL OR total_facilities_included >= 0
    ),
    CONSTRAINT uq_p041_ob_org_year_version UNIQUE (organization_id, reporting_year, boundary_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_ob_tenant             ON ghg_scope12.organizational_boundaries(tenant_id);
CREATE INDEX idx_p041_ob_org               ON ghg_scope12.organizational_boundaries(organization_id);
CREATE INDEX idx_p041_ob_year              ON ghg_scope12.organizational_boundaries(reporting_year);
CREATE INDEX idx_p041_ob_consolidation     ON ghg_scope12.organizational_boundaries(consolidation_approach);
CREATE INDEX idx_p041_ob_status            ON ghg_scope12.organizational_boundaries(status);
CREATE INDEX idx_p041_ob_current           ON ghg_scope12.organizational_boundaries(is_current) WHERE is_current = true;
CREATE INDEX idx_p041_ob_created           ON ghg_scope12.organizational_boundaries(created_at DESC);

-- Composite: org + year + current boundary
CREATE INDEX idx_p041_ob_org_year_current  ON ghg_scope12.organizational_boundaries(organization_id, reporting_year)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_ob_updated
    BEFORE UPDATE ON ghg_scope12.organizational_boundaries
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_scope12.boundary_inclusions
-- =============================================================================
-- Maps entities and facilities into an organizational boundary with their
-- respective inclusion percentages. For equity share approach, inclusion_pct
-- reflects ownership share. For control approaches, it is typically 100%
-- (fully included) or 0% (excluded). Provides justification for exclusions.

CREATE TABLE ghg_scope12.boundary_inclusions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_scope12.organizational_boundaries(id) ON DELETE CASCADE,
    entity_id                   UUID            REFERENCES ghg_scope12.legal_entities(id) ON DELETE SET NULL,
    facility_id                 UUID            REFERENCES ghg_scope12.facilities(id) ON DELETE SET NULL,
    inclusion_pct               DECIMAL(5,2)    NOT NULL DEFAULT 100.00,
    inclusion_type              VARCHAR(30)     NOT NULL DEFAULT 'FULL',
    justification               TEXT,
    exclusion_reason            TEXT,
    emissions_estimate_tco2e    NUMERIC(12,3),
    materiality_assessment      VARCHAR(30),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_bi_inclusion_pct CHECK (
        inclusion_pct >= 0 AND inclusion_pct <= 100
    ),
    CONSTRAINT chk_p041_bi_inclusion_type CHECK (
        inclusion_type IN ('FULL', 'PARTIAL_EQUITY', 'PARTIAL_OTHER', 'EXCLUDED')
    ),
    CONSTRAINT chk_p041_bi_materiality CHECK (
        materiality_assessment IS NULL OR materiality_assessment IN (
            'MATERIAL', 'NOT_MATERIAL', 'UNDER_REVIEW', 'NOT_ASSESSED'
        )
    ),
    CONSTRAINT chk_p041_bi_entity_or_facility CHECK (
        entity_id IS NOT NULL OR facility_id IS NOT NULL
    ),
    CONSTRAINT chk_p041_bi_estimate CHECK (
        emissions_estimate_tco2e IS NULL OR emissions_estimate_tco2e >= 0
    ),
    CONSTRAINT uq_p041_bi_boundary_entity_fac UNIQUE (boundary_id, entity_id, facility_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_bi_tenant             ON ghg_scope12.boundary_inclusions(tenant_id);
CREATE INDEX idx_p041_bi_boundary           ON ghg_scope12.boundary_inclusions(boundary_id);
CREATE INDEX idx_p041_bi_entity             ON ghg_scope12.boundary_inclusions(entity_id);
CREATE INDEX idx_p041_bi_facility           ON ghg_scope12.boundary_inclusions(facility_id);
CREATE INDEX idx_p041_bi_inclusion_type     ON ghg_scope12.boundary_inclusions(inclusion_type);
CREATE INDEX idx_p041_bi_materiality        ON ghg_scope12.boundary_inclusions(materiality_assessment);
CREATE INDEX idx_p041_bi_created            ON ghg_scope12.boundary_inclusions(created_at DESC);

-- Composite: boundary + included items
CREATE INDEX idx_p041_bi_boundary_included  ON ghg_scope12.boundary_inclusions(boundary_id, inclusion_pct)
    WHERE inclusion_type != 'EXCLUDED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_bi_updated
    BEFORE UPDATE ON ghg_scope12.boundary_inclusions
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 6: ghg_scope12.source_categories
-- =============================================================================
-- GHG emission source categories for each facility. Maps to GHG Protocol
-- source categories (stationary combustion, mobile combustion, process,
-- fugitive, refrigerant, etc.) and tracks applicability, materiality, data
-- availability, and the MRV agent responsible for calculating each category.

CREATE TABLE ghg_scope12.source_categories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    facility_id                 UUID            NOT NULL REFERENCES ghg_scope12.facilities(id) ON DELETE CASCADE,
    scope                       VARCHAR(10)     NOT NULL DEFAULT 'SCOPE_1',
    category                    VARCHAR(60)     NOT NULL,
    sub_category                VARCHAR(60),
    description                 TEXT,
    is_applicable               BOOLEAN         NOT NULL DEFAULT true,
    is_material                 BOOLEAN         NOT NULL DEFAULT true,
    materiality_pct             DECIMAL(5,2),
    data_availability           VARCHAR(30)     NOT NULL DEFAULT 'AVAILABLE',
    data_quality_score          NUMERIC(5,2),
    methodology_tier            VARCHAR(20)     DEFAULT 'TIER_1',
    mrv_agent_id                VARCHAR(100),
    emission_estimate_tco2e     NUMERIC(12,3),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_sc_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2')
    ),
    CONSTRAINT chk_p041_sc_category CHECK (
        category IN (
            'STATIONARY_COMBUSTION', 'MOBILE_COMBUSTION', 'PROCESS_EMISSIONS',
            'FUGITIVE_EMISSIONS', 'REFRIGERANT_LEAKAGE', 'LAND_USE_CHANGE',
            'WASTE_TREATMENT', 'AGRICULTURAL_EMISSIONS',
            'PURCHASED_ELECTRICITY', 'PURCHASED_STEAM',
            'PURCHASED_COOLING', 'PURCHASED_HEATING',
            'COMBINED_HEAT_POWER', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_sc_materiality_pct CHECK (
        materiality_pct IS NULL OR (materiality_pct >= 0 AND materiality_pct <= 100)
    ),
    CONSTRAINT chk_p041_sc_data_availability CHECK (
        data_availability IN (
            'AVAILABLE', 'PARTIAL', 'ESTIMATED', 'NOT_AVAILABLE', 'PLANNED'
        )
    ),
    CONSTRAINT chk_p041_sc_quality_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p041_sc_tier CHECK (
        methodology_tier IS NULL OR methodology_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p041_sc_estimate CHECK (
        emission_estimate_tco2e IS NULL OR emission_estimate_tco2e >= 0
    ),
    CONSTRAINT uq_p041_sc_facility_scope_cat UNIQUE (facility_id, scope, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_sc_tenant             ON ghg_scope12.source_categories(tenant_id);
CREATE INDEX idx_p041_sc_facility           ON ghg_scope12.source_categories(facility_id);
CREATE INDEX idx_p041_sc_scope              ON ghg_scope12.source_categories(scope);
CREATE INDEX idx_p041_sc_category           ON ghg_scope12.source_categories(category);
CREATE INDEX idx_p041_sc_applicable         ON ghg_scope12.source_categories(is_applicable) WHERE is_applicable = true;
CREATE INDEX idx_p041_sc_material           ON ghg_scope12.source_categories(is_material) WHERE is_material = true;
CREATE INDEX idx_p041_sc_data_avail         ON ghg_scope12.source_categories(data_availability);
CREATE INDEX idx_p041_sc_tier               ON ghg_scope12.source_categories(methodology_tier);
CREATE INDEX idx_p041_sc_mrv_agent          ON ghg_scope12.source_categories(mrv_agent_id);
CREATE INDEX idx_p041_sc_created            ON ghg_scope12.source_categories(created_at DESC);
CREATE INDEX idx_p041_sc_metadata           ON ghg_scope12.source_categories USING GIN(metadata);

-- Composite: facility + applicable + material sources
CREATE INDEX idx_p041_sc_fac_material       ON ghg_scope12.source_categories(facility_id, scope, category)
    WHERE is_applicable = true AND is_material = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_sc_updated
    BEFORE UPDATE ON ghg_scope12.source_categories
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.legal_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.facilities ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.organizational_boundaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.boundary_inclusions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.source_categories ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_org_tenant_isolation
    ON ghg_scope12.organizations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_org_service_bypass
    ON ghg_scope12.organizations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_le_tenant_isolation
    ON ghg_scope12.legal_entities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_le_service_bypass
    ON ghg_scope12.legal_entities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_fac_tenant_isolation
    ON ghg_scope12.facilities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_fac_service_bypass
    ON ghg_scope12.facilities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_ob_tenant_isolation
    ON ghg_scope12.organizational_boundaries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_ob_service_bypass
    ON ghg_scope12.organizational_boundaries
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_bi_tenant_isolation
    ON ghg_scope12.boundary_inclusions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_bi_service_bypass
    ON ghg_scope12.boundary_inclusions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_sc_tenant_isolation
    ON ghg_scope12.source_categories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_sc_service_bypass
    ON ghg_scope12.source_categories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_scope12 TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.organizations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.legal_entities TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.facilities TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.organizational_boundaries TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.boundary_inclusions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.source_categories TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_scope12.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_scope12 IS
    'PACK-041 Scope 1-2 Complete Pack - Full GHG Protocol Scope 1 and Scope 2 inventory management including organizational boundaries, emission factor registry, consolidation, uncertainty analysis, base year management, trend analysis, compliance mapping, and reporting.';

COMMENT ON TABLE ghg_scope12.organizations IS
    'Top-level reporting organizations with consolidation approach, sector classification, and corporate metadata per GHG Protocol Corporate Standard.';
COMMENT ON TABLE ghg_scope12.legal_entities IS
    'Legal entities (subsidiaries, JVs, associates) composing the organization with ownership percentage and operational/financial control flags.';
COMMENT ON TABLE ghg_scope12.facilities IS
    'Physical facilities where emissions occur with location, physical characteristics, and intensity metric denominators (area, headcount, output).';
COMMENT ON TABLE ghg_scope12.organizational_boundaries IS
    'Versioned organizational boundaries per reporting year defining which entities and facilities are included in the GHG inventory.';
COMMENT ON TABLE ghg_scope12.boundary_inclusions IS
    'Entity/facility inclusion mappings within an organizational boundary with equity share percentages and exclusion justifications.';
COMMENT ON TABLE ghg_scope12.source_categories IS
    'GHG emission source categories per facility tracking applicability, materiality, data availability, and assigned MRV agent.';

COMMENT ON COLUMN ghg_scope12.organizations.id IS 'Unique identifier for the reporting organization.';
COMMENT ON COLUMN ghg_scope12.organizations.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN ghg_scope12.organizations.consolidation_approach IS 'GHG Protocol consolidation: EQUITY_SHARE, OPERATIONAL_CONTROL, or FINANCIAL_CONTROL.';
COMMENT ON COLUMN ghg_scope12.organizations.industry_code_system IS 'Classification system for industry_code: NACE, NAICS, SIC, ISIC, GICS, ICB.';
COMMENT ON COLUMN ghg_scope12.organizations.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN ghg_scope12.legal_entities.ownership_pct IS 'Equity ownership percentage (0-100). Used for equity share consolidation.';
COMMENT ON COLUMN ghg_scope12.legal_entities.has_operational_control IS 'Whether the parent has operational control per GHG Protocol definition.';
COMMENT ON COLUMN ghg_scope12.legal_entities.has_financial_control IS 'Whether the parent has financial control per GHG Protocol definition.';

COMMENT ON COLUMN ghg_scope12.facilities.floor_area_m2 IS 'Total conditioned floor area in square meters for intensity metric calculation.';
COMMENT ON COLUMN ghg_scope12.facilities.headcount IS 'Full-time equivalent employee count for intensity metric calculation.';
COMMENT ON COLUMN ghg_scope12.facilities.operating_hours IS 'Annual operating hours (max 8784 for leap year).';
COMMENT ON COLUMN ghg_scope12.facilities.grid_region IS 'Electricity grid region identifier for location-based emission factors.';

COMMENT ON COLUMN ghg_scope12.organizational_boundaries.consolidation_approach IS 'Consolidation approach for this boundary year (may differ from org default).';
COMMENT ON COLUMN ghg_scope12.organizational_boundaries.boundary_version IS 'Version number incremented when boundary is restated (e.g., after M&A).';

COMMENT ON COLUMN ghg_scope12.boundary_inclusions.inclusion_pct IS 'Percentage of entity/facility emissions included (0-100). 100 for operational/financial control, ownership % for equity share.';

COMMENT ON COLUMN ghg_scope12.source_categories.mrv_agent_id IS 'GreenLang MRV agent responsible for calculating this source category (e.g., gl_stationary_combustion).';
COMMENT ON COLUMN ghg_scope12.source_categories.methodology_tier IS 'GHG Protocol methodology tier: TIER_1 (spend/activity), TIER_2 (supplier-specific), TIER_3 (direct measurement).';
