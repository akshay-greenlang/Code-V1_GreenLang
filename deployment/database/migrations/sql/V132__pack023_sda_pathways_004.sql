-- =============================================================================
-- V132: PACK-023-sbti-alignment-004: SDA Sector Convergence Pathway Records
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for SDA sector convergence pathway records.
-- Covers 12-sector intensity convergence calculations with IEA NZE benchmarks,
-- company baseline intensity, sectoral 2050 targets, conversion formulas,
-- annual milestone tracking, and validation against sector-specific criteria.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (pathway baseline)
--   PACK-022: Net Zero Acceleration Pack (scenario pathways)
--   V129: PACK-023 Target Definitions
--
-- 12 Sectors: Power, Cement, Steel, Aluminium, Pulp/Paper, Chemicals,
--   Aviation, Maritime, Road Transport, Buildings Commercial,
--   Buildings Residential, Food & Beverage
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the SDA sector intensity pathway layer for the pack.
-- =============================================================================
-- Tables (3):
--   1. pack023_sda_sector_pathways         - SDA pathway definitions per sector
--   2. pack023_sda_sector_benchmarks       - Sector benchmark data and targets
--   3. pack023_sda_annual_milestones       - Annual intensity milestone tracking
--
-- Also includes: 35+ indexes, update triggers, security grants, and comments.
-- Previous: V131__pack023_scope3_screening_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_sda_sector_pathways
-- =============================================================================
-- SDA pathway definitions for each organization by sector, tracking baseline
-- intensity, convergence calculation, annual reduction rates, and validation.

CREATE TABLE pack023_sbti_alignment.pack023_sda_sector_pathways (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_definition_id    UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_target_definitions(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    sector_code             VARCHAR(50)     NOT NULL,
    sector_name             VARCHAR(255)    NOT NULL,
    sector_category         VARCHAR(100),
    is_primary_sector       BOOLEAN         DEFAULT FALSE,
    company_baseline_year   INTEGER         NOT NULL,
    company_target_year     INTEGER         NOT NULL,
    baseline_intensity      DECIMAL(16,8),
    baseline_intensity_unit VARCHAR(100),
    company_emissions_baseline DECIMAL(18,6),
    company_activity_baseline DECIMAL(18,6),
    sector_2050_target      DECIMAL(16,8),
    sector_2050_unit        VARCHAR(100),
    sector_baseline_2020    DECIMAL(16,8),
    sector_baseline_year    INTEGER         DEFAULT 2020,
    convergence_start_year  INTEGER,
    convergence_end_year    INTEGER,
    convergence_years       INTEGER,
    company_target_intensity DECIMAL(16,8),
    implied_reduction_pct   DECIMAL(8,4),
    annual_intensity_reduction DECIMAL(6,4),
    verification_method     VARCHAR(100),
    data_quality_assessment VARCHAR(30),
    aligned_with_iea_nze    BOOLEAN         DEFAULT FALSE,
    reference_scenario      VARCHAR(100),
    transition_risk         VARCHAR(500),
    implementation_risk     VARCHAR(500),
    assumptions             JSONB           DEFAULT '{}',
    calculation_details     JSONB           DEFAULT '{}',
    validation_status       VARCHAR(30)     DEFAULT 'pending',
    validated_at            TIMESTAMPTZ,
    validated_by            VARCHAR(255),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_sda_sector CHECK (
        sector_code IN ('POWER', 'CEMENT', 'STEEL', 'ALUMINIUM', 'PULP_PAPER',
                        'CHEMICALS', 'AVIATION', 'MARITIME', 'ROAD_TRANSPORT',
                        'BUILDINGS_COMMERCIAL', 'BUILDINGS_RESIDENTIAL', 'FOOD_BEVERAGE')
    ),
    CONSTRAINT chk_pk_sda_years CHECK (
        company_target_year > company_baseline_year
    ),
    CONSTRAINT chk_pk_sda_convergence_years CHECK (
        convergence_years > 0
    ),
    CONSTRAINT chk_pk_sda_intensity CHECK (
        baseline_intensity IS NULL OR baseline_intensity >= 0
    )
);

-- Indexes
CREATE INDEX idx_pk_sda_target_id ON pack023_sbti_alignment.pack023_sda_sector_pathways(target_definition_id);
CREATE INDEX idx_pk_sda_tenant ON pack023_sbti_alignment.pack023_sda_sector_pathways(tenant_id);
CREATE INDEX idx_pk_sda_org ON pack023_sbti_alignment.pack023_sda_sector_pathways(org_id);
CREATE INDEX idx_pk_sda_sector_code ON pack023_sbti_alignment.pack023_sda_sector_pathways(sector_code);
CREATE INDEX idx_pk_sda_sector_name ON pack023_sbti_alignment.pack023_sda_sector_pathways(sector_name);
CREATE INDEX idx_pk_sda_primary ON pack023_sbti_alignment.pack023_sda_sector_pathways(is_primary_sector);
CREATE INDEX idx_pk_sda_baseline_year ON pack023_sbti_alignment.pack023_sda_sector_pathways(company_baseline_year);
CREATE INDEX idx_pk_sda_target_year ON pack023_sbti_alignment.pack023_sda_sector_pathways(company_target_year);
CREATE INDEX idx_pk_sda_validation ON pack023_sbti_alignment.pack023_sda_sector_pathways(validation_status);
CREATE INDEX idx_pk_sda_iea_aligned ON pack023_sbti_alignment.pack023_sda_sector_pathways(aligned_with_iea_nze);
CREATE INDEX idx_pk_sda_created_at ON pack023_sbti_alignment.pack023_sda_sector_pathways(created_at DESC);
CREATE INDEX idx_pk_sda_assumptions ON pack023_sbti_alignment.pack023_sda_sector_pathways USING GIN(assumptions);
CREATE INDEX idx_pk_sda_details ON pack023_sbti_alignment.pack023_sda_sector_pathways USING GIN(calculation_details);

-- Updated_at trigger
CREATE TRIGGER trg_pk_sda_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sda_sector_pathways
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_sda_sector_benchmarks
-- =============================================================================
-- Reference sector benchmark data for all 12 SDA sectors, with baseline intensity,
-- 2050 target intensity, and supporting data from IEA, sector associations, etc.

CREATE TABLE pack023_sbti_alignment.pack023_sda_sector_benchmarks (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sector_code             VARCHAR(50)     NOT NULL,
    sector_name             VARCHAR(255)    NOT NULL,
    sector_description      TEXT,
    baseline_year           INTEGER         NOT NULL,
    baseline_intensity      DECIMAL(16,8)   NOT NULL,
    baseline_intensity_unit VARCHAR(100)    NOT NULL,
    target_year             INTEGER         NOT NULL,
    target_intensity_2050   DECIMAL(16,8)   NOT NULL,
    target_intensity_unit   VARCHAR(100),
    reduction_pathway       VARCHAR(100),
    reduction_rate_annual   DECIMAL(6,4),
    ipcc_scenario           VARCHAR(100),
    iea_reference           VARCHAR(500),
    data_source             VARCHAR(500),
    publication_date        DATE,
    last_updated            DATE,
    confidence_level        VARCHAR(30),
    geographic_scope        VARCHAR(200),
    applicability_notes     TEXT,
    sub_sectors_included    TEXT[],
    data_collection_method  VARCHAR(500),
    methodology_reference   VARCHAR(500),
    version                 VARCHAR(50),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_bench_sector CHECK (
        sector_code IN ('POWER', 'CEMENT', 'STEEL', 'ALUMINIUM', 'PULP_PAPER',
                        'CHEMICALS', 'AVIATION', 'MARITIME', 'ROAD_TRANSPORT',
                        'BUILDINGS_COMMERCIAL', 'BUILDINGS_RESIDENTIAL', 'FOOD_BEVERAGE')
    ),
    CONSTRAINT chk_pk_bench_years CHECK (
        target_year > baseline_year
    ),
    CONSTRAINT chk_pk_bench_intensity CHECK (
        baseline_intensity >= 0 AND target_intensity_2050 >= 0
    )
);

-- Indexes
CREATE UNIQUE INDEX idx_pk_bench_sector_unique ON pack023_sbti_alignment.pack023_sda_sector_benchmarks(sector_code);
CREATE INDEX idx_pk_bench_sector_name ON pack023_sbti_alignment.pack023_sda_sector_benchmarks(sector_name);
CREATE INDEX idx_pk_bench_baseline_year ON pack023_sbti_alignment.pack023_sda_sector_benchmarks(baseline_year);
CREATE INDEX idx_pk_bench_target_year ON pack023_sbti_alignment.pack023_sda_sector_benchmarks(target_year);
CREATE INDEX idx_pk_bench_confidence ON pack023_sbti_alignment.pack023_sda_sector_benchmarks(confidence_level);
CREATE INDEX idx_pk_bench_created_at ON pack023_sbti_alignment.pack023_sda_sector_benchmarks(created_at DESC);
CREATE INDEX idx_pk_bench_subsectors ON pack023_sbti_alignment.pack023_sda_sector_benchmarks USING GIN(sub_sectors_included);

-- Updated_at trigger
CREATE TRIGGER trg_pk_bench_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sda_sector_benchmarks
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_sda_annual_milestones
-- =============================================================================
-- Annual intensity milestone tracking for SDA pathways, showing year-by-year
-- convergence between company baseline and sector 2050 target.

CREATE TABLE pack023_sbti_alignment.pack023_sda_annual_milestones (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sda_pathway_id          UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sda_sector_pathways(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    milestone_year          INTEGER         NOT NULL,
    years_into_pathway      INTEGER,
    target_intensity        DECIMAL(16,8),
    target_intensity_unit   VARCHAR(100),
    target_emissions_mt     DECIMAL(18,6),
    actual_intensity        DECIMAL(16,8),
    actual_emissions_mt     DECIMAL(18,6),
    activity_level          DECIMAL(18,6),
    intensity_variance      DECIMAL(8,4),
    variance_percentage     DECIMAL(8,4),
    on_track                BOOLEAN,
    intensity_reduction_target DECIMAL(6,4),
    intensity_reduction_achieved DECIMAL(6,4),
    reduction_gap           DECIMAL(6,4),
    data_quality_tier       VARCHAR(30),
    verification_status     VARCHAR(30),
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_mile_year CHECK (
        milestone_year >= 2020 AND milestone_year <= 2050
    ),
    CONSTRAINT chk_pk_mile_years_into CHECK (
        years_into_pathway IS NULL OR years_into_pathway >= 0
    ),
    CONSTRAINT chk_pk_mile_variance CHECK (
        variance_percentage IS NULL OR (variance_percentage >= -100 AND variance_percentage <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pk_mile_pathway_id ON pack023_sbti_alignment.pack023_sda_annual_milestones(sda_pathway_id);
CREATE INDEX idx_pk_mile_tenant ON pack023_sbti_alignment.pack023_sda_annual_milestones(tenant_id);
CREATE INDEX idx_pk_mile_org ON pack023_sbti_alignment.pack023_sda_annual_milestones(org_id);
CREATE INDEX idx_pk_mile_year ON pack023_sbti_alignment.pack023_sda_annual_milestones(milestone_year);
CREATE INDEX idx_pk_mile_on_track ON pack023_sbti_alignment.pack023_sda_annual_milestones(on_track);
CREATE INDEX idx_pk_mile_verification ON pack023_sbti_alignment.pack023_sda_annual_milestones(verification_status);
CREATE INDEX idx_pk_mile_created_at ON pack023_sbti_alignment.pack023_sda_annual_milestones(created_at DESC);
CREATE INDEX idx_pk_mile_year_progress ON pack023_sbti_alignment.pack023_sda_annual_milestones(milestone_year, on_track);

-- Updated_at trigger
CREATE TRIGGER trg_pk_mile_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sda_annual_milestones
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack023_sbti_alignment TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack023_sbti_alignment TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack023_sbti_alignment TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack023_sbti_alignment.pack023_sda_sector_pathways IS
'SDA pathway definitions for 12 sectors with baseline intensity, 2050 target, convergence calculation, and annual reduction rate tracking.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sda_sector_benchmarks IS
'Reference benchmark data for all 12 SDA sectors from IEA and sector associations, with baseline and 2050 target intensities.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sda_annual_milestones IS
'Year-by-year intensity milestone tracking for SDA pathways showing convergence progress and on-track status against sector benchmarks.';
