-- =============================================================================
-- V181: PACK-028 Sector Pathway Pack - Schema & Sector Classification
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    001 of 015
-- Date:         March 2026
--
-- Creates the pack028_sector_pathway schema and the sector classification
-- table with NACE Rev.2, GICS, ISIC Rev.4 code mappings, SBTi SDA
-- eligibility flags, and IEA NZE sector alignment metadata.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_sector_classifications
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V180__PACK027_views_and_indexes.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack028_sector_pathway;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack028_sector_pathway.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sector_classifications
-- =============================================================================
-- Automatic sector classification with NACE/GICS/ISIC code mapping,
-- SBTi SDA sector eligibility, IEA NZE chapter alignment, and revenue-based
-- multi-sector weighting for conglomerate organizations.

CREATE TABLE pack028_sector_pathway.gl_sector_classifications (
    classification_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    -- Primary sector classification
    primary_sector              VARCHAR(80)     NOT NULL,
    primary_sector_code         VARCHAR(20)     NOT NULL,
    sector_taxonomy             VARCHAR(30)     NOT NULL DEFAULT 'GREENLANG',
    -- NACE Rev.2 mapping
    nace_code                   VARCHAR(10),
    nace_section                VARCHAR(5),
    nace_division               VARCHAR(5),
    nace_description            VARCHAR(500),
    -- GICS mapping
    gics_code                   VARCHAR(10),
    gics_sector                 VARCHAR(100),
    gics_industry_group         VARCHAR(100),
    gics_industry               VARCHAR(200),
    gics_sub_industry           VARCHAR(200),
    -- ISIC Rev.4 mapping
    isic_code                   VARCHAR(10),
    isic_section                VARCHAR(5),
    isic_division               VARCHAR(5),
    isic_description            VARCHAR(500),
    -- SBTi SDA eligibility
    sda_eligible                BOOLEAN         NOT NULL DEFAULT FALSE,
    sda_sector                  VARCHAR(50),
    sda_methodology_version     VARCHAR(20),
    sda_intensity_metric        VARCHAR(80),
    sda_intensity_unit          VARCHAR(50),
    sda_coverage_requirement    DECIMAL(5,2)    DEFAULT 95.00,
    sda_scope_coverage          VARCHAR(30)     DEFAULT 'SCOPE_1_2',
    -- SBTi FLAG eligibility
    flag_eligible               BOOLEAN         NOT NULL DEFAULT FALSE,
    flag_sector                 VARCHAR(50),
    flag_guidance_version       VARCHAR(20),
    -- IEA NZE alignment
    iea_sector                  VARCHAR(100),
    iea_chapter                 VARCHAR(100),
    iea_sub_sector              VARCHAR(100),
    iea_pathway_available       BOOLEAN         DEFAULT FALSE,
    -- Revenue-based multi-sector weighting
    revenue_share_pct           DECIMAL(6,2)    DEFAULT 100.00,
    is_primary                  BOOLEAN         DEFAULT TRUE,
    multi_sector                BOOLEAN         DEFAULT FALSE,
    sub_sectors                 JSONB           DEFAULT '[]',
    -- Sector characteristics
    carbon_intensity_profile    VARCHAR(30),
    primary_emission_sources    TEXT[]          DEFAULT '{}',
    process_emissions_pct       DECIMAL(5,2),
    energy_emissions_pct        DECIMAL(5,2),
    scope3_dominant             BOOLEAN         DEFAULT FALSE,
    typical_scope3_categories   TEXT[]          DEFAULT '{}',
    -- Classification confidence
    classification_method       VARCHAR(30)     NOT NULL DEFAULT 'MANUAL',
    classification_confidence   DECIMAL(5,2)    DEFAULT 100.00,
    classification_source       VARCHAR(100),
    classification_date         DATE,
    -- Validation
    validation_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    validated_by                VARCHAR(255),
    validated_at                TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_sc_sector_taxonomy CHECK (
        sector_taxonomy IN ('GREENLANG', 'NACE_REV2', 'GICS', 'ISIC_REV4', 'SBTi', 'IEA', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sc_sda_sector CHECK (
        sda_sector IS NULL OR sda_sector IN (
            'POWER_GENERATION', 'CEMENT', 'IRON_STEEL', 'ALUMINIUM', 'PULP_PAPER',
            'CHEMICALS', 'AVIATION', 'MARITIME_SHIPPING', 'ROAD_TRANSPORT', 'RAIL',
            'COMMERCIAL_BUILDINGS', 'RESIDENTIAL_BUILDINGS', 'FOOD_BEVERAGE'
        )
    ),
    CONSTRAINT chk_p028_sc_sda_scope_coverage CHECK (
        sda_scope_coverage IS NULL OR sda_scope_coverage IN (
            'SCOPE_1', 'SCOPE_1_2', 'SCOPE_1_2_3', 'SCOPE_3'
        )
    ),
    CONSTRAINT chk_p028_sc_flag_sector CHECK (
        flag_sector IS NULL OR flag_sector IN (
            'AGRICULTURE', 'FORESTRY', 'LAND_USE', 'FOOD_BEVERAGE', 'LIVESTOCK',
            'CROP_PRODUCTION', 'FISHING_AQUACULTURE'
        )
    ),
    CONSTRAINT chk_p028_sc_carbon_intensity CHECK (
        carbon_intensity_profile IS NULL OR carbon_intensity_profile IN (
            'VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW'
        )
    ),
    CONSTRAINT chk_p028_sc_revenue_share CHECK (
        revenue_share_pct >= 0 AND revenue_share_pct <= 100
    ),
    CONSTRAINT chk_p028_sc_process_emissions CHECK (
        process_emissions_pct IS NULL OR (process_emissions_pct >= 0 AND process_emissions_pct <= 100)
    ),
    CONSTRAINT chk_p028_sc_energy_emissions CHECK (
        energy_emissions_pct IS NULL OR (energy_emissions_pct >= 0 AND energy_emissions_pct <= 100)
    ),
    CONSTRAINT chk_p028_sc_classification_method CHECK (
        classification_method IN ('MANUAL', 'AUTOMATIC', 'NACE_LOOKUP', 'GICS_LOOKUP',
                                  'ISIC_LOOKUP', 'REVENUE_BASED', 'HYBRID')
    ),
    CONSTRAINT chk_p028_sc_confidence CHECK (
        classification_confidence >= 0 AND classification_confidence <= 100
    ),
    CONSTRAINT chk_p028_sc_validation_status CHECK (
        validation_status IN ('PENDING', 'VALIDATED', 'REJECTED', 'REVIEW_REQUIRED', 'EXPIRED')
    ),
    CONSTRAINT chk_p028_sc_sda_coverage_req CHECK (
        sda_coverage_requirement IS NULL OR (sda_coverage_requirement >= 0 AND sda_coverage_requirement <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_sc_tenant             ON pack028_sector_pathway.gl_sector_classifications(tenant_id);
CREATE INDEX idx_p028_sc_company            ON pack028_sector_pathway.gl_sector_classifications(company_id);
CREATE INDEX idx_p028_sc_primary_sector     ON pack028_sector_pathway.gl_sector_classifications(primary_sector);
CREATE INDEX idx_p028_sc_sector_code        ON pack028_sector_pathway.gl_sector_classifications(primary_sector_code);
CREATE INDEX idx_p028_sc_taxonomy           ON pack028_sector_pathway.gl_sector_classifications(sector_taxonomy);
CREATE INDEX idx_p028_sc_nace              ON pack028_sector_pathway.gl_sector_classifications(nace_code);
CREATE INDEX idx_p028_sc_nace_section      ON pack028_sector_pathway.gl_sector_classifications(nace_section);
CREATE INDEX idx_p028_sc_gics             ON pack028_sector_pathway.gl_sector_classifications(gics_code);
CREATE INDEX idx_p028_sc_gics_sector      ON pack028_sector_pathway.gl_sector_classifications(gics_sector);
CREATE INDEX idx_p028_sc_isic             ON pack028_sector_pathway.gl_sector_classifications(isic_code);
CREATE INDEX idx_p028_sc_sda_eligible      ON pack028_sector_pathway.gl_sector_classifications(sda_eligible) WHERE sda_eligible = TRUE;
CREATE INDEX idx_p028_sc_sda_sector        ON pack028_sector_pathway.gl_sector_classifications(sda_sector) WHERE sda_sector IS NOT NULL;
CREATE INDEX idx_p028_sc_flag_eligible     ON pack028_sector_pathway.gl_sector_classifications(flag_eligible) WHERE flag_eligible = TRUE;
CREATE INDEX idx_p028_sc_iea_sector        ON pack028_sector_pathway.gl_sector_classifications(iea_sector);
CREATE INDEX idx_p028_sc_iea_pathway       ON pack028_sector_pathway.gl_sector_classifications(iea_pathway_available) WHERE iea_pathway_available = TRUE;
CREATE INDEX idx_p028_sc_is_primary        ON pack028_sector_pathway.gl_sector_classifications(company_id, is_primary) WHERE is_primary = TRUE;
CREATE INDEX idx_p028_sc_multi_sector      ON pack028_sector_pathway.gl_sector_classifications(company_id, multi_sector) WHERE multi_sector = TRUE;
CREATE INDEX idx_p028_sc_intensity_profile ON pack028_sector_pathway.gl_sector_classifications(carbon_intensity_profile);
CREATE INDEX idx_p028_sc_classification    ON pack028_sector_pathway.gl_sector_classifications(classification_method);
CREATE INDEX idx_p028_sc_validation        ON pack028_sector_pathway.gl_sector_classifications(validation_status);
CREATE INDEX idx_p028_sc_created           ON pack028_sector_pathway.gl_sector_classifications(created_at DESC);
CREATE INDEX idx_p028_sc_sub_sectors       ON pack028_sector_pathway.gl_sector_classifications USING GIN(sub_sectors);
CREATE INDEX idx_p028_sc_emission_sources  ON pack028_sector_pathway.gl_sector_classifications USING GIN(primary_emission_sources);
CREATE INDEX idx_p028_sc_metadata          ON pack028_sector_pathway.gl_sector_classifications USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_sector_classifications_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_classifications
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_sector_classifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_sc_tenant_isolation
    ON pack028_sector_pathway.gl_sector_classifications
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_sc_service_bypass
    ON pack028_sector_pathway.gl_sector_classifications
    TO greenlang_service
    USING (TRUE)
    WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack028_sector_pathway TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_classifications TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack028_sector_pathway.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack028_sector_pathway IS
    'PACK-028 Sector Pathway Pack - Sector-specific decarbonization pathway analysis aligned with SBTi SDA methodology (12 sectors) and IEA Net Zero roadmap (15+ sectors) for science-based transition strategies.';

COMMENT ON TABLE pack028_sector_pathway.gl_sector_classifications IS
    'Automatic sector classification with NACE Rev.2, GICS, ISIC Rev.4 code mappings, SBTi SDA eligibility flags, and IEA NZE sector alignment metadata for multi-sector organizations.';

COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.classification_id IS 'Unique sector classification identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.company_id IS 'Reference to the enterprise/company being classified.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.primary_sector IS 'Human-readable primary sector name (e.g., Power Generation, Steel, Cement).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.primary_sector_code IS 'Machine-readable sector code (e.g., POWER, STEEL, CEMENT).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.nace_code IS 'NACE Rev.2 industry classification code (e.g., D35.11 for electricity production).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.gics_code IS 'GICS industry classification code (e.g., 10101010 for Oil & Gas Exploration).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.isic_code IS 'ISIC Rev.4 industry classification code.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.sda_eligible IS 'Whether this sector is eligible for SBTi Sectoral Decarbonization Approach.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.sda_sector IS 'SBTi SDA sector name (12 SDA sectors supported).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.sda_intensity_metric IS 'Sector-specific intensity metric for SDA (e.g., gCO2/kWh, tCO2e/tonne).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.flag_eligible IS 'Whether this sector is eligible for SBTi FLAG (Forest, Land and Agriculture) pathway.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.iea_sector IS 'IEA NZE 2050 sector classification for pathway alignment.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.revenue_share_pct IS 'Revenue share for multi-sector companies (0-100%).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.carbon_intensity_profile IS 'Sector carbon intensity profile: VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.classification_method IS 'How classification was determined: MANUAL, AUTOMATIC, NACE_LOOKUP, etc.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_classifications.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
