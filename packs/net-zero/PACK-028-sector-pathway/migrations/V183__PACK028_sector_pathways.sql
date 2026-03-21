-- =============================================================================
-- V183: PACK-028 Sector Pathway Pack - Sector Pathways
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    003 of 015
-- Date:         March 2026
--
-- Sector-specific decarbonization pathways with annual intensity targets,
-- scenario variants (1.5C, WB2C, 2C, APS, STEPS), source attribution
-- (SBTi SDA / IEA NZE / IPCC AR6), and convergence model parameters
-- for 15+ sectors.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_sector_pathways
--
-- Previous: V182__PACK028_intensity_metrics.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sector_pathways
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_sector_pathways (
    pathway_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    classification_id           UUID            REFERENCES pack028_sector_pathway.gl_sector_classifications(classification_id) ON DELETE SET NULL,
    -- Sector identification
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    sub_sector                  VARCHAR(80),
    -- Pathway definition
    pathway_name                VARCHAR(255)    NOT NULL,
    pathway_type                VARCHAR(30)     NOT NULL,
    pathway_source              VARCHAR(30)     NOT NULL,
    pathway_version             VARCHAR(20),
    -- Scenario
    scenario                    VARCHAR(30)     NOT NULL DEFAULT 'NZE_1_5C',
    temperature_target          VARCHAR(10)     NOT NULL DEFAULT '1.5C',
    probability_pct             DECIMAL(5,2),
    -- Intensity metric
    intensity_metric            VARCHAR(60)     NOT NULL,
    intensity_unit              VARCHAR(80)     NOT NULL,
    -- Base year
    base_year                   INTEGER         NOT NULL,
    base_year_intensity         DECIMAL(18,8)   NOT NULL,
    base_year_emissions_tco2e   DECIMAL(18,4),
    base_year_activity          DECIMAL(18,4),
    -- Target year
    target_year                 INTEGER         NOT NULL DEFAULT 2050,
    target_intensity            DECIMAL(18,8)   NOT NULL,
    target_reduction_pct        DECIMAL(6,2),
    -- Interim targets
    interim_2025_intensity      DECIMAL(18,8),
    interim_2030_intensity      DECIMAL(18,8),
    interim_2035_intensity      DECIMAL(18,8),
    interim_2040_intensity      DECIMAL(18,8),
    interim_2045_intensity      DECIMAL(18,8),
    -- Annual pathway data (year -> intensity)
    annual_pathway              JSONB           NOT NULL DEFAULT '{}',
    annual_emissions_pathway    JSONB           DEFAULT '{}',
    annual_activity_pathway     JSONB           DEFAULT '{}',
    -- Convergence model
    convergence_model           VARCHAR(30)     NOT NULL DEFAULT 'LINEAR',
    convergence_parameters      JSONB           DEFAULT '{}',
    inflection_year             INTEGER,
    convergence_rate            DECIMAL(8,6),
    -- Regional variant
    region                      VARCHAR(30)     DEFAULT 'GLOBAL',
    regional_adjustment_factor  DECIMAL(8,4)    DEFAULT 1.0,
    -- Source attribution
    sbti_pathway_id             VARCHAR(100),
    sbti_sda_version            VARCHAR(20),
    iea_pathway_ref             VARCHAR(100),
    iea_scenario_year           INTEGER,
    ipcc_pathway_ref            VARCHAR(100),
    ipcc_ssp_scenario           VARCHAR(20),
    -- Activity growth assumptions
    activity_growth_rate_pct    DECIMAL(6,3),
    activity_growth_scenario    VARCHAR(30),
    population_growth_rate_pct  DECIMAL(6,3),
    gdp_growth_rate_pct         DECIMAL(6,3),
    -- Technology assumptions
    technology_assumptions      JSONB           DEFAULT '{}',
    renewable_share_2030_pct    DECIMAL(5,2),
    renewable_share_2050_pct    DECIMAL(5,2),
    ccs_capacity_2050_mtco2     DECIMAL(12,2),
    hydrogen_share_2050_pct     DECIMAL(5,2),
    -- Validation
    pathway_status              VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    validation_status           VARCHAR(20)     DEFAULT 'PENDING',
    validated_by                VARCHAR(255),
    validated_at                TIMESTAMPTZ,
    sbti_aligned                BOOLEAN         DEFAULT FALSE,
    iea_aligned                 BOOLEAN         DEFAULT FALSE,
    -- Governance
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_primary                  BOOLEAN         DEFAULT FALSE,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    methodology_notes           TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_sp_pathway_type CHECK (
        pathway_type IN ('SDA', 'ACA', 'FLAG', 'IEA_NZE', 'IEA_APS', 'IEA_STEPS',
                         'IPCC_SSP1_19', 'IPCC_SSP1_26', 'CUSTOM', 'HYBRID')
    ),
    CONSTRAINT chk_p028_sp_pathway_source CHECK (
        pathway_source IN ('SBTI', 'IEA', 'IPCC', 'NATIONAL_NDC', 'INDUSTRY_BODY',
                           'COMPANY_DEFINED', 'GREENLANG', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sp_scenario CHECK (
        scenario IN ('NZE_1_5C', 'WB2C', '2C', 'APS', 'STEPS', 'NDC_ALIGNED',
                     'SSP1_19', 'SSP1_26', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sp_temperature CHECK (
        temperature_target IN ('1.5C', 'WB2C', '2.0C', '2.5C', '3.0C', '4.0C')
    ),
    CONSTRAINT chk_p028_sp_convergence CHECK (
        convergence_model IN ('LINEAR', 'EXPONENTIAL', 'S_CURVE', 'STEPPED',
                              'POLYNOMIAL', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sp_region CHECK (
        region IN ('GLOBAL', 'OECD', 'NON_OECD', 'EU', 'NORTH_AMERICA', 'ASIA_PACIFIC',
                   'LATIN_AMERICA', 'AFRICA', 'MIDDLE_EAST', 'CHINA', 'INDIA', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sp_base_year CHECK (
        base_year >= 2000 AND base_year <= 2030
    ),
    CONSTRAINT chk_p028_sp_target_year CHECK (
        target_year >= 2030 AND target_year <= 2100
    ),
    CONSTRAINT chk_p028_sp_target_after_base CHECK (
        target_year > base_year
    ),
    CONSTRAINT chk_p028_sp_base_intensity CHECK (
        base_year_intensity >= 0
    ),
    CONSTRAINT chk_p028_sp_target_intensity CHECK (
        target_intensity >= 0
    ),
    CONSTRAINT chk_p028_sp_probability CHECK (
        probability_pct IS NULL OR (probability_pct >= 0 AND probability_pct <= 100)
    ),
    CONSTRAINT chk_p028_sp_pathway_status CHECK (
        pathway_status IN ('DRAFT', 'ACTIVE', 'APPROVED', 'ARCHIVED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p028_sp_validation_status CHECK (
        validation_status IN ('PENDING', 'VALIDATED', 'REJECTED', 'REVIEW_REQUIRED', 'EXPIRED')
    ),
    CONSTRAINT chk_p028_sp_reduction_pct CHECK (
        target_reduction_pct IS NULL OR (target_reduction_pct >= 0 AND target_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p028_sp_ipcc_ssp CHECK (
        ipcc_ssp_scenario IS NULL OR ipcc_ssp_scenario IN (
            'SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_sp_tenant             ON pack028_sector_pathway.gl_sector_pathways(tenant_id);
CREATE INDEX idx_p028_sp_company            ON pack028_sector_pathway.gl_sector_pathways(company_id);
CREATE INDEX idx_p028_sp_classification     ON pack028_sector_pathway.gl_sector_pathways(classification_id);
CREATE INDEX idx_p028_sp_sector             ON pack028_sector_pathway.gl_sector_pathways(sector);
CREATE INDEX idx_p028_sp_sector_code        ON pack028_sector_pathway.gl_sector_pathways(sector_code);
CREATE INDEX idx_p028_sp_pathway_type       ON pack028_sector_pathway.gl_sector_pathways(pathway_type);
CREATE INDEX idx_p028_sp_pathway_source     ON pack028_sector_pathway.gl_sector_pathways(pathway_source);
CREATE INDEX idx_p028_sp_scenario           ON pack028_sector_pathway.gl_sector_pathways(scenario);
CREATE INDEX idx_p028_sp_temperature        ON pack028_sector_pathway.gl_sector_pathways(temperature_target);
CREATE INDEX idx_p028_sp_convergence        ON pack028_sector_pathway.gl_sector_pathways(convergence_model);
CREATE INDEX idx_p028_sp_region             ON pack028_sector_pathway.gl_sector_pathways(region);
CREATE INDEX idx_p028_sp_base_year          ON pack028_sector_pathway.gl_sector_pathways(base_year);
CREATE INDEX idx_p028_sp_target_year        ON pack028_sector_pathway.gl_sector_pathways(target_year);
CREATE INDEX idx_p028_sp_status             ON pack028_sector_pathway.gl_sector_pathways(pathway_status);
CREATE INDEX idx_p028_sp_active             ON pack028_sector_pathway.gl_sector_pathways(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_sp_primary            ON pack028_sector_pathway.gl_sector_pathways(company_id, is_primary) WHERE is_primary = TRUE;
CREATE INDEX idx_p028_sp_sbti_aligned       ON pack028_sector_pathway.gl_sector_pathways(sbti_aligned) WHERE sbti_aligned = TRUE;
CREATE INDEX idx_p028_sp_iea_aligned        ON pack028_sector_pathway.gl_sector_pathways(iea_aligned) WHERE iea_aligned = TRUE;
CREATE INDEX idx_p028_sp_company_scenario   ON pack028_sector_pathway.gl_sector_pathways(company_id, scenario, sector_code);
CREATE INDEX idx_p028_sp_created            ON pack028_sector_pathway.gl_sector_pathways(created_at DESC);
CREATE INDEX idx_p028_sp_annual_pathway     ON pack028_sector_pathway.gl_sector_pathways USING GIN(annual_pathway);
CREATE INDEX idx_p028_sp_tech_assumptions   ON pack028_sector_pathway.gl_sector_pathways USING GIN(technology_assumptions);
CREATE INDEX idx_p028_sp_metadata           ON pack028_sector_pathway.gl_sector_pathways USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_sector_pathways_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_pathways
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_sector_pathways ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_sp_tenant_isolation
    ON pack028_sector_pathway.gl_sector_pathways
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_sp_service_bypass
    ON pack028_sector_pathway.gl_sector_pathways
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_pathways TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_sector_pathways IS
    'Sector-specific decarbonization pathways with annual intensity targets, 5 scenario variants (NZE/WB2C/2C/APS/STEPS), convergence models, and SBTi SDA/IEA NZE/IPCC AR6 source attribution.';

COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.pathway_id IS 'Unique pathway identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.pathway_type IS 'Pathway methodology: SDA, ACA, FLAG, IEA_NZE, IEA_APS, IEA_STEPS, IPCC_SSP, CUSTOM, HYBRID.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.pathway_source IS 'Authoritative source: SBTI, IEA, IPCC, NATIONAL_NDC, INDUSTRY_BODY, COMPANY_DEFINED.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.scenario IS 'Climate scenario: NZE_1_5C, WB2C, 2C, APS, STEPS, NDC_ALIGNED.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.annual_pathway IS 'JSONB of year-by-year intensity targets (e.g., {"2025": 450.0, "2026": 430.0, ...}).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.convergence_model IS 'Mathematical convergence model: LINEAR, EXPONENTIAL, S_CURVE, STEPPED, POLYNOMIAL.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.region IS 'Regional pathway variant: GLOBAL, OECD, NON_OECD, EU, NORTH_AMERICA, etc.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_pathways.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
