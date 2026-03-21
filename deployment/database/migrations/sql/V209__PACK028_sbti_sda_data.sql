-- =============================================================================
-- V189: PACK-028 Sector Pathway Pack - SBTi SDA Data
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    009 of 015
-- Date:         March 2026
--
-- SBTi Sectoral Decarbonization Approach (SDA) reference data including
-- published sector pathways, convergence factors, validation criteria,
-- and submission tracking for SBTi target compliance.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_sbti_sector_pathways
--
-- Previous: V188__PACK028_scenario_comparisons.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sbti_sector_pathways
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_sbti_sector_pathways (
    sbti_pathway_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID,
    -- SBTi SDA sector
    sda_sector                  VARCHAR(50)     NOT NULL,
    sda_sector_display          VARCHAR(100)    NOT NULL,
    sda_methodology_version     VARCHAR(20)     NOT NULL DEFAULT '2.0',
    -- SBTi pathway reference
    sbti_pathway_ref            VARCHAR(100),
    sbti_tool_version           VARCHAR(20),
    sbti_data_year              INTEGER,
    -- Intensity metric
    intensity_metric            VARCHAR(60)     NOT NULL,
    intensity_unit              VARCHAR(80)     NOT NULL,
    -- Pathway data
    base_year                   INTEGER         NOT NULL DEFAULT 2020,
    target_year                 INTEGER         NOT NULL DEFAULT 2050,
    ambition_level              VARCHAR(20)     NOT NULL DEFAULT '1.5C',
    -- Reference intensity values
    global_base_year_intensity  DECIMAL(18,8)   NOT NULL,
    global_2030_intensity       DECIMAL(18,8),
    global_2040_intensity       DECIMAL(18,8),
    global_2050_intensity       DECIMAL(18,8)   NOT NULL,
    -- Company-specific
    company_base_year_intensity DECIMAL(18,8),
    company_target_intensity    DECIMAL(18,8),
    company_convergence_year    INTEGER,
    -- Annual convergence pathway (year -> intensity)
    annual_pathway              JSONB           NOT NULL DEFAULT '{}',
    -- Convergence parameters
    convergence_model           VARCHAR(30)     DEFAULT 'LINEAR',
    convergence_factor          DECIMAL(18,8),
    annual_reduction_rate       DECIMAL(8,6),
    -- Activity growth
    activity_growth_scenario    VARCHAR(50),
    activity_growth_data        JSONB           DEFAULT '{}',
    -- Coverage requirements
    scope_coverage              VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2',
    coverage_requirement_pct    DECIMAL(5,2)    NOT NULL DEFAULT 95.00,
    exclusions_allowed_pct      DECIMAL(5,2)    DEFAULT 5.00,
    -- Validation criteria
    validation_criteria         JSONB           NOT NULL DEFAULT '{}',
    criteria_c1_boundary        BOOLEAN         DEFAULT FALSE,
    criteria_c2_coverage        BOOLEAN         DEFAULT FALSE,
    criteria_c3_timeframe       BOOLEAN         DEFAULT FALSE,
    criteria_c4_ambition        BOOLEAN         DEFAULT FALSE,
    criteria_c5_scope3          BOOLEAN,
    criteria_c6_recalculation   BOOLEAN         DEFAULT FALSE,
    total_criteria_pass         INTEGER         DEFAULT 0,
    total_criteria_count        INTEGER         DEFAULT 28,
    overall_criteria_pct        DECIMAL(5,2),
    -- SBTi submission tracking
    submission_status           VARCHAR(30)     DEFAULT 'NOT_SUBMITTED',
    submission_date             DATE,
    validation_date             DATE,
    validation_outcome          VARCHAR(20),
    revalidation_due            DATE,
    sbti_commitment_letter_ref  VARCHAR(100),
    sbti_target_id              VARCHAR(100),
    -- Near-term compliance
    near_term_target_pct        DECIMAL(6,2),
    near_term_target_year       INTEGER,
    near_term_annual_rate       DECIMAL(6,3),
    near_term_minimum_rate      DECIMAL(6,3)    DEFAULT 4.20,
    near_term_compliant         BOOLEAN         DEFAULT FALSE,
    -- Long-term compliance
    long_term_target_pct        DECIMAL(6,2),
    long_term_target_year       INTEGER         DEFAULT 2050,
    long_term_residual_pct      DECIMAL(5,2),
    long_term_compliant         BOOLEAN         DEFAULT FALSE,
    -- FLAG integration
    flag_applicable             BOOLEAN         DEFAULT FALSE,
    flag_separate_target        BOOLEAN         DEFAULT FALSE,
    flag_pathway_data           JSONB           DEFAULT '{}',
    -- Regional pathway variant
    region                      VARCHAR(30)     DEFAULT 'GLOBAL',
    regional_pathway            JSONB           DEFAULT '{}',
    -- Data quality
    data_quality_assessment     JSONB           DEFAULT '{}',
    data_completeness_pct       DECIMAL(5,2),
    -- Metadata
    is_reference_data           BOOLEAN         DEFAULT FALSE,
    is_active                   BOOLEAN         DEFAULT TRUE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_ssp_sda_sector CHECK (
        sda_sector IN (
            'POWER_GENERATION', 'CEMENT', 'IRON_STEEL', 'ALUMINIUM', 'PULP_PAPER',
            'CHEMICALS', 'AVIATION', 'MARITIME_SHIPPING', 'ROAD_TRANSPORT', 'RAIL',
            'COMMERCIAL_BUILDINGS', 'RESIDENTIAL_BUILDINGS', 'FOOD_BEVERAGE'
        )
    ),
    CONSTRAINT chk_p028_ssp_ambition CHECK (
        ambition_level IN ('1.5C', 'WELL_BELOW_2C', '2C', 'SECTOR_ALIGNED')
    ),
    CONSTRAINT chk_p028_ssp_scope_coverage CHECK (
        scope_coverage IN ('SCOPE_1', 'SCOPE_1_2', 'SCOPE_1_2_3', 'SCOPE_3')
    ),
    CONSTRAINT chk_p028_ssp_convergence CHECK (
        convergence_model IN ('LINEAR', 'EXPONENTIAL', 'S_CURVE', 'STEPPED', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_ssp_coverage_req CHECK (
        coverage_requirement_pct >= 0 AND coverage_requirement_pct <= 100
    ),
    CONSTRAINT chk_p028_ssp_submission CHECK (
        submission_status IN ('NOT_SUBMITTED', 'COMMITTED', 'SUBMITTED', 'UNDER_REVIEW',
                              'VALIDATED', 'REVISION_REQUESTED', 'REJECTED', 'EXPIRED')
    ),
    CONSTRAINT chk_p028_ssp_validation_outcome CHECK (
        validation_outcome IS NULL OR validation_outcome IN (
            'APPROVED', 'APPROVED_WITH_CONDITIONS', 'REVISION_REQUIRED', 'REJECTED'
        )
    ),
    CONSTRAINT chk_p028_ssp_region CHECK (
        region IN ('GLOBAL', 'OECD', 'NON_OECD', 'EU', 'NORTH_AMERICA', 'ASIA_PACIFIC',
                   'LATIN_AMERICA', 'AFRICA', 'MIDDLE_EAST', 'CHINA', 'INDIA', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_ssp_base_year CHECK (
        base_year >= 2000 AND base_year <= 2030
    ),
    CONSTRAINT chk_p028_ssp_target_year CHECK (
        target_year >= 2030 AND target_year <= 2100
    ),
    CONSTRAINT chk_p028_ssp_near_term_year CHECK (
        near_term_target_year IS NULL OR (near_term_target_year >= 2025 AND near_term_target_year <= 2040)
    ),
    CONSTRAINT chk_p028_ssp_near_term_pct CHECK (
        near_term_target_pct IS NULL OR (near_term_target_pct >= 0 AND near_term_target_pct <= 100)
    ),
    CONSTRAINT chk_p028_ssp_long_term_pct CHECK (
        long_term_target_pct IS NULL OR (long_term_target_pct >= 0 AND long_term_target_pct <= 100)
    ),
    CONSTRAINT chk_p028_ssp_residual CHECK (
        long_term_residual_pct IS NULL OR (long_term_residual_pct >= 0 AND long_term_residual_pct <= 10)
    ),
    CONSTRAINT chk_p028_ssp_criteria_pct CHECK (
        overall_criteria_pct IS NULL OR (overall_criteria_pct >= 0 AND overall_criteria_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_ssp_tenant            ON pack028_sector_pathway.gl_sbti_sector_pathways(tenant_id);
CREATE INDEX idx_p028_ssp_company           ON pack028_sector_pathway.gl_sbti_sector_pathways(company_id);
CREATE INDEX idx_p028_ssp_sda_sector        ON pack028_sector_pathway.gl_sbti_sector_pathways(sda_sector);
CREATE INDEX idx_p028_ssp_ambition          ON pack028_sector_pathway.gl_sbti_sector_pathways(ambition_level);
CREATE INDEX idx_p028_ssp_version           ON pack028_sector_pathway.gl_sbti_sector_pathways(sda_methodology_version);
CREATE INDEX idx_p028_ssp_submission        ON pack028_sector_pathway.gl_sbti_sector_pathways(submission_status);
CREATE INDEX idx_p028_ssp_validation        ON pack028_sector_pathway.gl_sbti_sector_pathways(validation_outcome);
CREATE INDEX idx_p028_ssp_revalidation      ON pack028_sector_pathway.gl_sbti_sector_pathways(revalidation_due) WHERE revalidation_due IS NOT NULL;
CREATE INDEX idx_p028_ssp_near_compliant    ON pack028_sector_pathway.gl_sbti_sector_pathways(near_term_compliant);
CREATE INDEX idx_p028_ssp_long_compliant    ON pack028_sector_pathway.gl_sbti_sector_pathways(long_term_compliant);
CREATE INDEX idx_p028_ssp_flag              ON pack028_sector_pathway.gl_sbti_sector_pathways(flag_applicable) WHERE flag_applicable = TRUE;
CREATE INDEX idx_p028_ssp_region            ON pack028_sector_pathway.gl_sbti_sector_pathways(region);
CREATE INDEX idx_p028_ssp_reference         ON pack028_sector_pathway.gl_sbti_sector_pathways(is_reference_data) WHERE is_reference_data = TRUE;
CREATE INDEX idx_p028_ssp_active            ON pack028_sector_pathway.gl_sbti_sector_pathways(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_ssp_company_sector    ON pack028_sector_pathway.gl_sbti_sector_pathways(company_id, sda_sector);
CREATE INDEX idx_p028_ssp_created           ON pack028_sector_pathway.gl_sbti_sector_pathways(created_at DESC);
CREATE INDEX idx_p028_ssp_annual_pathway    ON pack028_sector_pathway.gl_sbti_sector_pathways USING GIN(annual_pathway);
CREATE INDEX idx_p028_ssp_criteria          ON pack028_sector_pathway.gl_sbti_sector_pathways USING GIN(validation_criteria);
CREATE INDEX idx_p028_ssp_metadata          ON pack028_sector_pathway.gl_sbti_sector_pathways USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_sbti_sector_pathways_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sbti_sector_pathways
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_sbti_sector_pathways ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_ssp_tenant_isolation
    ON pack028_sector_pathway.gl_sbti_sector_pathways
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_ssp_service_bypass
    ON pack028_sector_pathway.gl_sbti_sector_pathways
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sbti_sector_pathways TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_sbti_sector_pathways IS
    'SBTi SDA reference pathways with published sector convergence data, validation criteria (C1-C28), submission tracking, and compliance assessment for 12 SDA sectors.';

COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.sbti_pathway_id IS 'Unique SBTi sector pathway record identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.sda_sector IS 'SBTi SDA sector: POWER_GENERATION, CEMENT, IRON_STEEL, ALUMINIUM, etc.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.annual_pathway IS 'JSONB year-by-year intensity convergence pathway from SBTi published data.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.validation_criteria IS 'JSONB matrix of SBTi validation criteria with pass/fail per criterion.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.submission_status IS 'SBTi submission pipeline status: NOT_SUBMITTED through VALIDATED.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.near_term_minimum_rate IS 'SBTi minimum annual reduction rate for 1.5C alignment (default 4.2%).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.is_reference_data IS 'TRUE for SBTi published reference pathways, FALSE for company-specific calculations.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sbti_sector_pathways.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
