-- =============================================================================
-- V328: PACK-041 Scope 1-2 Complete Pack - Scope 1 Consolidation & Results
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates Scope 1 inventory consolidation tables that aggregate results from
-- individual MRV agents (stationary combustion, mobile combustion, process,
-- fugitive, refrigerant, land use, waste, agricultural) into a unified
-- inventory per organization and reporting year. Tracks per-gas breakdowns,
-- methodology tiers, calculation hashes, and double-counting resolution.
--
-- Tables (3):
--   1. ghg_scope12.scope1_inventories
--   2. ghg_scope12.scope1_category_results
--   3. ghg_scope12.double_counting_flags
--
-- Also includes: indexes, RLS, comments.
-- Previous: V327__pack041_emission_factors.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.scope1_inventories
-- =============================================================================
-- Top-level Scope 1 inventory for an organization and reporting year. Each
-- inventory aggregates all Scope 1 category results across all facilities
-- within the organizational boundary. The inventory lifecycle tracks from
-- draft collection through finalization and verification.

CREATE TABLE ghg_scope12.scope1_inventories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_scope12.organizational_boundaries(id) ON DELETE RESTRICT,
    reporting_year              INTEGER         NOT NULL,
    reporting_period_start      DATE,
    reporting_period_end        DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    -- Per-gas totals (tonnes)
    total_co2                   DECIMAL(15,3)   DEFAULT 0,
    total_ch4                   DECIMAL(12,6)   DEFAULT 0,
    total_n2o                   DECIMAL(12,6)   DEFAULT 0,
    total_hfc                   DECIMAL(12,6)   DEFAULT 0,
    total_pfc                   DECIMAL(12,6)   DEFAULT 0,
    total_sf6                   DECIMAL(12,6)   DEFAULT 0,
    total_nf3                   DECIMAL(12,6)   DEFAULT 0,
    total_co2e                  DECIMAL(15,3)   NOT NULL DEFAULT 0,
    -- Biogenic CO2 (reported separately per GHG Protocol)
    biogenic_co2_tonnes         DECIMAL(12,3)   DEFAULT 0,
    -- Quality metrics
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    data_completeness_pct       DECIMAL(5,2),
    weighted_data_quality       DECIMAL(5,2),
    categories_reported         INTEGER         DEFAULT 0,
    categories_applicable       INTEGER         DEFAULT 0,
    facilities_reported         INTEGER         DEFAULT 0,
    facilities_total            INTEGER         DEFAULT 0,
    -- Workflow
    prepared_by                 VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    approved_by                 VARCHAR(255),
    notes                       TEXT,
    methodology_notes           TEXT,
    exclusions_description      TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    finalized_at                TIMESTAMPTZ,
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_s1inv_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p041_s1inv_status CHECK (
        status IN (
            'DRAFT', 'DATA_COLLECTION', 'CALCULATION', 'REVIEW',
            'APPROVED', 'FINALIZED', 'VERIFIED', 'RESTATED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p041_s1inv_gwp CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p041_s1inv_total CHECK (
        total_co2e >= 0
    ),
    CONSTRAINT chk_p041_s1inv_co2 CHECK (
        total_co2 IS NULL OR total_co2 >= 0
    ),
    CONSTRAINT chk_p041_s1inv_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p041_s1inv_quality CHECK (
        weighted_data_quality IS NULL OR (weighted_data_quality >= 0 AND weighted_data_quality <= 100)
    ),
    CONSTRAINT chk_p041_s1inv_dates CHECK (
        reporting_period_start IS NULL OR reporting_period_end IS NULL OR
        reporting_period_start <= reporting_period_end
    ),
    CONSTRAINT chk_p041_s1inv_categories CHECK (
        categories_reported IS NULL OR categories_applicable IS NULL OR
        categories_reported <= categories_applicable
    ),
    CONSTRAINT uq_p041_s1inv_org_year UNIQUE (organization_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_s1inv_tenant          ON ghg_scope12.scope1_inventories(tenant_id);
CREATE INDEX idx_p041_s1inv_org             ON ghg_scope12.scope1_inventories(organization_id);
CREATE INDEX idx_p041_s1inv_boundary        ON ghg_scope12.scope1_inventories(boundary_id);
CREATE INDEX idx_p041_s1inv_year            ON ghg_scope12.scope1_inventories(reporting_year);
CREATE INDEX idx_p041_s1inv_status          ON ghg_scope12.scope1_inventories(status);
CREATE INDEX idx_p041_s1inv_total           ON ghg_scope12.scope1_inventories(total_co2e DESC);
CREATE INDEX idx_p041_s1inv_created         ON ghg_scope12.scope1_inventories(created_at DESC);
CREATE INDEX idx_p041_s1inv_metadata        ON ghg_scope12.scope1_inventories USING GIN(metadata);

-- Composite: tenant + year for dashboard queries
CREATE INDEX idx_p041_s1inv_tenant_year     ON ghg_scope12.scope1_inventories(tenant_id, reporting_year DESC);

-- Composite: org + finalized for reporting
CREATE INDEX idx_p041_s1inv_org_final       ON ghg_scope12.scope1_inventories(organization_id, reporting_year DESC)
    WHERE status IN ('FINALIZED', 'VERIFIED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_s1inv_updated
    BEFORE UPDATE ON ghg_scope12.scope1_inventories
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.scope1_category_results
-- =============================================================================
-- Per-category, per-facility Scope 1 emission results produced by individual
-- MRV agents. Each row represents one source category at one facility with
-- full per-gas breakdown and methodology tracking. The calculation_hash
-- links to the MRV agent's deterministic calculation provenance.

CREATE TABLE ghg_scope12.scope1_category_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_scope12.scope1_inventories(id) ON DELETE CASCADE,
    facility_id                 UUID            NOT NULL REFERENCES ghg_scope12.facilities(id) ON DELETE RESTRICT,
    category                    VARCHAR(60)     NOT NULL,
    sub_category                VARCHAR(60),
    agent_id                    VARCHAR(100)    NOT NULL,
    agent_version               VARCHAR(20),
    -- Per-gas breakdown (tonnes)
    co2_tonnes                  DECIMAL(12,3)   NOT NULL DEFAULT 0,
    ch4_tonnes                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    n2o_tonnes                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    hfc_tonnes                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    pfc_tonnes                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    sf6_tonnes                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    nf3_tonnes                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    total_co2e                  DECIMAL(12,3)   NOT NULL DEFAULT 0,
    -- Biogenic
    biogenic_co2_tonnes         DECIMAL(12,3)   DEFAULT 0,
    -- Methodology
    methodology_tier            VARCHAR(20)     NOT NULL DEFAULT 'TIER_1',
    emission_factor_id          UUID            REFERENCES ghg_scope12.emission_factor_registry(id) ON DELETE SET NULL,
    emission_factor_source      VARCHAR(100),
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    -- Activity data
    activity_data_value         DECIMAL(18,6),
    activity_data_unit          VARCHAR(50),
    activity_data_source        VARCHAR(100),
    -- Quality
    data_quality_score          NUMERIC(5,2),
    data_quality_indicator      VARCHAR(20),
    uncertainty_pct             DECIMAL(8,4),
    -- Provenance
    calculation_timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    calculation_hash            VARCHAR(64)     NOT NULL,
    calculation_details         JSONB           DEFAULT '{}',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_s1cr_category CHECK (
        category IN (
            'STATIONARY_COMBUSTION', 'MOBILE_COMBUSTION', 'PROCESS_EMISSIONS',
            'FUGITIVE_EMISSIONS', 'REFRIGERANT_LEAKAGE', 'LAND_USE_CHANGE',
            'WASTE_TREATMENT', 'AGRICULTURAL_EMISSIONS', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_s1cr_tier CHECK (
        methodology_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p041_s1cr_gwp CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p041_s1cr_total CHECK (
        total_co2e >= 0
    ),
    CONSTRAINT chk_p041_s1cr_co2 CHECK (
        co2_tonnes >= 0
    ),
    CONSTRAINT chk_p041_s1cr_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p041_s1cr_quality_ind CHECK (
        data_quality_indicator IS NULL OR data_quality_indicator IN (
            'HIGH', 'MEDIUM', 'LOW', 'ESTIMATED', 'DEFAULT'
        )
    ),
    CONSTRAINT chk_p041_s1cr_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 200)
    ),
    CONSTRAINT uq_p041_s1cr_inv_fac_cat UNIQUE (inventory_id, facility_id, category, sub_category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_s1cr_tenant           ON ghg_scope12.scope1_category_results(tenant_id);
CREATE INDEX idx_p041_s1cr_inventory        ON ghg_scope12.scope1_category_results(inventory_id);
CREATE INDEX idx_p041_s1cr_facility         ON ghg_scope12.scope1_category_results(facility_id);
CREATE INDEX idx_p041_s1cr_category         ON ghg_scope12.scope1_category_results(category);
CREATE INDEX idx_p041_s1cr_agent            ON ghg_scope12.scope1_category_results(agent_id);
CREATE INDEX idx_p041_s1cr_tier             ON ghg_scope12.scope1_category_results(methodology_tier);
CREATE INDEX idx_p041_s1cr_ef               ON ghg_scope12.scope1_category_results(emission_factor_id);
CREATE INDEX idx_p041_s1cr_total            ON ghg_scope12.scope1_category_results(total_co2e DESC);
CREATE INDEX idx_p041_s1cr_calc_hash        ON ghg_scope12.scope1_category_results(calculation_hash);
CREATE INDEX idx_p041_s1cr_created          ON ghg_scope12.scope1_category_results(created_at DESC);
CREATE INDEX idx_p041_s1cr_details          ON ghg_scope12.scope1_category_results USING GIN(calculation_details);

-- Composite: inventory + category for aggregation
CREATE INDEX idx_p041_s1cr_inv_cat          ON ghg_scope12.scope1_category_results(inventory_id, category, total_co2e DESC);

-- Composite: facility + category for facility-level reporting
CREATE INDEX idx_p041_s1cr_fac_cat          ON ghg_scope12.scope1_category_results(facility_id, category);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_s1cr_updated
    BEFORE UPDATE ON ghg_scope12.scope1_category_results
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.double_counting_flags
-- =============================================================================
-- Identifies and tracks potential double-counting between emission source
-- categories within an inventory. Common overlaps include: stationary
-- combustion with process emissions (on-site combustion), mobile with
-- stationary (fleet vehicles at facilities), fugitive with process (gas
-- processing). Each flag records the overlap type and resolution method.

CREATE TABLE ghg_scope12.double_counting_flags (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_scope12.scope1_inventories(id) ON DELETE CASCADE,
    facility_id                 UUID            REFERENCES ghg_scope12.facilities(id) ON DELETE SET NULL,
    category_a                  VARCHAR(60)     NOT NULL,
    category_b                  VARCHAR(60)     NOT NULL,
    overlap_type                VARCHAR(50)     NOT NULL,
    overlap_description         TEXT,
    detection_method            VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATED',
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'FLAGGED',
    resolution                  VARCHAR(50),
    resolution_description      TEXT,
    adjusted_amount             DECIMAL(12,3),
    adjusted_category           VARCHAR(60),
    resolved_by                 VARCHAR(255),
    resolved_at                 TIMESTAMPTZ,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_dc_overlap_type CHECK (
        overlap_type IN (
            'COMBUSTION_PROCESS', 'MOBILE_STATIONARY', 'FUGITIVE_PROCESS',
            'REFRIGERANT_PROCESS', 'WASTE_PROCESS', 'LAND_USE_AGRICULTURAL',
            'BOUNDARY_OVERLAP', 'ENTITY_OVERLAP', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_dc_detection CHECK (
        detection_method IN ('AUTOMATED', 'MANUAL_REVIEW', 'AGENT_FLAGGED', 'AUDIT')
    ),
    CONSTRAINT chk_p041_dc_severity CHECK (
        severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p041_dc_status CHECK (
        status IN ('FLAGGED', 'UNDER_REVIEW', 'RESOLVED', 'ACCEPTED', 'DISMISSED')
    ),
    CONSTRAINT chk_p041_dc_resolution CHECK (
        resolution IS NULL OR resolution IN (
            'DEDUCTED_FROM_A', 'DEDUCTED_FROM_B', 'SPLIT_PROPORTIONAL',
            'RECLASSIFIED', 'NO_OVERLAP_CONFIRMED', 'ACCEPTED_IMMATERIAL',
            'CUSTOM_ADJUSTMENT'
        )
    ),
    CONSTRAINT chk_p041_dc_adjusted CHECK (
        adjusted_amount IS NULL OR adjusted_amount >= 0
    ),
    CONSTRAINT chk_p041_dc_categories_differ CHECK (
        category_a != category_b
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_dc_tenant             ON ghg_scope12.double_counting_flags(tenant_id);
CREATE INDEX idx_p041_dc_inventory          ON ghg_scope12.double_counting_flags(inventory_id);
CREATE INDEX idx_p041_dc_facility           ON ghg_scope12.double_counting_flags(facility_id);
CREATE INDEX idx_p041_dc_category_a         ON ghg_scope12.double_counting_flags(category_a);
CREATE INDEX idx_p041_dc_category_b         ON ghg_scope12.double_counting_flags(category_b);
CREATE INDEX idx_p041_dc_overlap_type       ON ghg_scope12.double_counting_flags(overlap_type);
CREATE INDEX idx_p041_dc_status             ON ghg_scope12.double_counting_flags(status);
CREATE INDEX idx_p041_dc_severity           ON ghg_scope12.double_counting_flags(severity);
CREATE INDEX idx_p041_dc_created            ON ghg_scope12.double_counting_flags(created_at DESC);

-- Composite: inventory + unresolved flags
CREATE INDEX idx_p041_dc_inv_unresolved     ON ghg_scope12.double_counting_flags(inventory_id, severity)
    WHERE status IN ('FLAGGED', 'UNDER_REVIEW');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_dc_updated
    BEFORE UPDATE ON ghg_scope12.double_counting_flags
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.scope1_inventories ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.scope1_category_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.double_counting_flags ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_s1inv_tenant_isolation
    ON ghg_scope12.scope1_inventories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_s1inv_service_bypass
    ON ghg_scope12.scope1_inventories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_s1cr_tenant_isolation
    ON ghg_scope12.scope1_category_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_s1cr_service_bypass
    ON ghg_scope12.scope1_category_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_dc_tenant_isolation
    ON ghg_scope12.double_counting_flags
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_dc_service_bypass
    ON ghg_scope12.double_counting_flags
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.scope1_inventories TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.scope1_category_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.double_counting_flags TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.scope1_inventories IS
    'Top-level Scope 1 inventory per organization and reporting year aggregating all category results with per-gas totals and workflow status.';
COMMENT ON TABLE ghg_scope12.scope1_category_results IS
    'Per-category, per-facility Scope 1 results from MRV agents with full per-gas breakdown, methodology tier, and calculation provenance hash.';
COMMENT ON TABLE ghg_scope12.double_counting_flags IS
    'Double-counting detection and resolution tracking between overlapping Scope 1 source categories within an inventory.';

COMMENT ON COLUMN ghg_scope12.scope1_inventories.id IS 'Unique identifier for the Scope 1 inventory.';
COMMENT ON COLUMN ghg_scope12.scope1_inventories.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN ghg_scope12.scope1_inventories.boundary_id IS 'Reference to the organizational boundary defining included entities/facilities.';
COMMENT ON COLUMN ghg_scope12.scope1_inventories.total_co2e IS 'Total Scope 1 emissions in tonnes CO2-equivalent.';
COMMENT ON COLUMN ghg_scope12.scope1_inventories.biogenic_co2_tonnes IS 'Biogenic CO2 reported separately per GHG Protocol (not included in total_co2e).';
COMMENT ON COLUMN ghg_scope12.scope1_inventories.gwp_source IS 'IPCC Assessment Report used for CO2e conversion across all categories.';
COMMENT ON COLUMN ghg_scope12.scope1_inventories.provenance_hash IS 'SHA-256 hash of all category result hashes for inventory-level provenance.';

COMMENT ON COLUMN ghg_scope12.scope1_category_results.agent_id IS 'GreenLang MRV agent that produced this result (e.g., gl_stationary_combustion_agent).';
COMMENT ON COLUMN ghg_scope12.scope1_category_results.calculation_hash IS 'SHA-256 hash from the MRV agent deterministic calculation for reproducibility.';
COMMENT ON COLUMN ghg_scope12.scope1_category_results.calculation_details IS 'JSON object with detailed calculation inputs, intermediate values, and parameters.';

COMMENT ON COLUMN ghg_scope12.double_counting_flags.overlap_type IS 'Type of overlap: COMBUSTION_PROCESS, MOBILE_STATIONARY, FUGITIVE_PROCESS, etc.';
COMMENT ON COLUMN ghg_scope12.double_counting_flags.resolution IS 'How the overlap was resolved: DEDUCTED_FROM_A, SPLIT_PROPORTIONAL, NO_OVERLAP_CONFIRMED, etc.';
COMMENT ON COLUMN ghg_scope12.double_counting_flags.adjusted_amount IS 'Amount in tCO2e deducted or adjusted to resolve the double-counting.';
