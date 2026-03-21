-- =============================================================================
-- V191: PACK-028 Sector Pathway Pack - IPCC AR6 Pathways
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    011 of 015
-- Date:         March 2026
--
-- IPCC AR6 sector-specific pathway reference data with SSP scenarios,
-- carbon budgets, GWP values, and sector emission factor data from
-- IPCC 2006 Guidelines with 2019 refinements.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_ipcc_sector_pathways
--
-- Previous: V190__PACK028_iea_nze_milestones.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_ipcc_sector_pathways
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_ipcc_sector_pathways (
    ipcc_pathway_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID,
    -- IPCC reference
    ipcc_report                 VARCHAR(50)     NOT NULL DEFAULT 'AR6',
    ipcc_working_group          VARCHAR(10)     NOT NULL DEFAULT 'WGIII',
    ipcc_chapter                VARCHAR(100),
    ipcc_chapter_number         INTEGER,
    -- SSP scenario
    ssp_scenario                VARCHAR(20)     NOT NULL,
    ssp_description             VARCHAR(200),
    temperature_outcome         VARCHAR(10)     NOT NULL,
    probability_below_target    DECIMAL(5,2),
    -- Sector
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    ipcc_sector_category        VARCHAR(50),
    -- Pathway data
    base_year                   INTEGER         NOT NULL DEFAULT 2020,
    target_year                 INTEGER         NOT NULL DEFAULT 2100,
    -- Emissions pathway (JSONB year -> GtCO2e)
    global_emissions_pathway    JSONB           NOT NULL DEFAULT '{}',
    sector_emissions_pathway    JSONB           DEFAULT '{}',
    -- Carbon budget
    remaining_carbon_budget_gtco2 DECIMAL(12,2),
    carbon_budget_from_year     INTEGER         DEFAULT 2020,
    carbon_budget_probability   DECIMAL(5,2),
    sector_budget_share_pct     DECIMAL(6,2),
    sector_budget_gtco2         DECIMAL(12,2),
    -- Emission reduction requirements
    reduction_2030_pct          DECIMAL(6,2),
    reduction_2040_pct          DECIMAL(6,2),
    reduction_2050_pct          DECIMAL(6,2),
    net_zero_year               INTEGER,
    net_negative_year           INTEGER,
    -- GWP values
    gwp_co2                     DECIMAL(8,2)    DEFAULT 1.0,
    gwp_ch4_100yr               DECIMAL(8,2)    DEFAULT 27.9,
    gwp_ch4_20yr                DECIMAL(8,2)    DEFAULT 81.2,
    gwp_n2o_100yr               DECIMAL(8,2)    DEFAULT 273.0,
    gwp_sf6_100yr               DECIMAL(8,2)    DEFAULT 25200.0,
    gwp_source                  VARCHAR(20)     DEFAULT 'AR6',
    custom_gwp_values           JSONB           DEFAULT '{}',
    -- Sector emission factors (IPCC 2006 + 2019 refinements)
    sector_emission_factors     JSONB           DEFAULT '{}',
    emission_factor_version     VARCHAR(30)     DEFAULT 'IPCC_2006_2019_REF',
    -- Mitigation potential
    mitigation_potential_2030   DECIMAL(12,2),
    mitigation_potential_2050   DECIMAL(12,2),
    mitigation_potential_unit   VARCHAR(30)     DEFAULT 'GtCO2e',
    mitigation_cost_curve       JSONB           DEFAULT '{}',
    below_100_usd_potential     DECIMAL(12,2),
    -- Key assumptions
    population_scenario         JSONB           DEFAULT '{}',
    gdp_scenario                JSONB           DEFAULT '{}',
    energy_demand_scenario      JSONB           DEFAULT '{}',
    technology_assumptions      JSONB           DEFAULT '{}',
    policy_assumptions          JSONB           DEFAULT '{}',
    -- Land use and removals
    lulucf_pathway              JSONB           DEFAULT '{}',
    cdr_requirement_gtco2       DECIMAL(10,2),
    nature_based_potential      DECIMAL(10,2),
    technical_removal_potential DECIMAL(10,2),
    -- Regional disaggregation
    region                      VARCHAR(30)     DEFAULT 'GLOBAL',
    regional_pathways           JSONB           DEFAULT '{}',
    -- Company alignment
    company_alignment_score     DECIMAL(5,2),
    company_budget_alignment    VARCHAR(20),
    company_trajectory_alignment VARCHAR(20),
    -- Metadata
    is_reference_data           BOOLEAN         DEFAULT FALSE,
    is_active                   BOOLEAN         DEFAULT TRUE,
    data_source                 VARCHAR(200),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_ipp_ssp_scenario CHECK (
        ssp_scenario IN ('SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5',
                         'SSP1-1.9_OVERSHOOT', 'SSP2-4.5_OVERSHOOT', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_ipp_temperature CHECK (
        temperature_outcome IN ('1.5C', 'WB2C', '2.0C', '2.5C', '3.0C', '4.0C', '5.0C')
    ),
    CONSTRAINT chk_p028_ipp_working_group CHECK (
        ipcc_working_group IN ('WGI', 'WGII', 'WGIII', 'SYR')
    ),
    CONSTRAINT chk_p028_ipp_gwp_source CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_ipp_ef_version CHECK (
        emission_factor_version IN ('IPCC_1996', 'IPCC_2006', 'IPCC_2006_2019_REF', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_ipp_region CHECK (
        region IN ('GLOBAL', 'OECD', 'NON_OECD', 'EU', 'NORTH_AMERICA', 'ASIA_PACIFIC',
                   'LATIN_AMERICA', 'AFRICA', 'MIDDLE_EAST', 'CHINA', 'INDIA', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_ipp_budget_alignment CHECK (
        company_budget_alignment IS NULL OR company_budget_alignment IN (
            'WELL_WITHIN', 'WITHIN', 'AT_LIMIT', 'EXCEEDING', 'SIGNIFICANTLY_EXCEEDING'
        )
    ),
    CONSTRAINT chk_p028_ipp_trajectory_alignment CHECK (
        company_trajectory_alignment IS NULL OR company_trajectory_alignment IN (
            'ALIGNED', 'NEARLY_ALIGNED', 'MISALIGNED', 'SIGNIFICANTLY_MISALIGNED'
        )
    ),
    CONSTRAINT chk_p028_ipp_probability CHECK (
        probability_below_target IS NULL OR (probability_below_target >= 0 AND probability_below_target <= 100)
    ),
    CONSTRAINT chk_p028_ipp_alignment_score CHECK (
        company_alignment_score IS NULL OR (company_alignment_score >= 0 AND company_alignment_score <= 100)
    ),
    CONSTRAINT chk_p028_ipp_base_year CHECK (
        base_year >= 1990 AND base_year <= 2030
    ),
    CONSTRAINT chk_p028_ipp_target_year CHECK (
        target_year >= 2030 AND target_year <= 2150
    ),
    CONSTRAINT chk_p028_ipp_reduction_2030 CHECK (
        reduction_2030_pct IS NULL OR (reduction_2030_pct >= -50 AND reduction_2030_pct <= 100)
    ),
    CONSTRAINT chk_p028_ipp_reduction_2050 CHECK (
        reduction_2050_pct IS NULL OR (reduction_2050_pct >= -50 AND reduction_2050_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_ipp_tenant            ON pack028_sector_pathway.gl_ipcc_sector_pathways(tenant_id);
CREATE INDEX idx_p028_ipp_company           ON pack028_sector_pathway.gl_ipcc_sector_pathways(company_id);
CREATE INDEX idx_p028_ipp_report            ON pack028_sector_pathway.gl_ipcc_sector_pathways(ipcc_report);
CREATE INDEX idx_p028_ipp_ssp               ON pack028_sector_pathway.gl_ipcc_sector_pathways(ssp_scenario);
CREATE INDEX idx_p028_ipp_temperature       ON pack028_sector_pathway.gl_ipcc_sector_pathways(temperature_outcome);
CREATE INDEX idx_p028_ipp_sector            ON pack028_sector_pathway.gl_ipcc_sector_pathways(sector_code);
CREATE INDEX idx_p028_ipp_sector_ssp        ON pack028_sector_pathway.gl_ipcc_sector_pathways(sector_code, ssp_scenario);
CREATE INDEX idx_p028_ipp_region            ON pack028_sector_pathway.gl_ipcc_sector_pathways(region);
CREATE INDEX idx_p028_ipp_gwp_source        ON pack028_sector_pathway.gl_ipcc_sector_pathways(gwp_source);
CREATE INDEX idx_p028_ipp_reference         ON pack028_sector_pathway.gl_ipcc_sector_pathways(is_reference_data) WHERE is_reference_data = TRUE;
CREATE INDEX idx_p028_ipp_active            ON pack028_sector_pathway.gl_ipcc_sector_pathways(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_ipp_budget_align      ON pack028_sector_pathway.gl_ipcc_sector_pathways(company_budget_alignment);
CREATE INDEX idx_p028_ipp_net_zero_year     ON pack028_sector_pathway.gl_ipcc_sector_pathways(net_zero_year);
CREATE INDEX idx_p028_ipp_created           ON pack028_sector_pathway.gl_ipcc_sector_pathways(created_at DESC);
CREATE INDEX idx_p028_ipp_emissions_path    ON pack028_sector_pathway.gl_ipcc_sector_pathways USING GIN(global_emissions_pathway);
CREATE INDEX idx_p028_ipp_sector_ef         ON pack028_sector_pathway.gl_ipcc_sector_pathways USING GIN(sector_emission_factors);
CREATE INDEX idx_p028_ipp_regional_paths    ON pack028_sector_pathway.gl_ipcc_sector_pathways USING GIN(regional_pathways);
CREATE INDEX idx_p028_ipp_metadata          ON pack028_sector_pathway.gl_ipcc_sector_pathways USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_ipcc_pathways_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_ipcc_sector_pathways
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_ipcc_sector_pathways ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_ipp_tenant_isolation
    ON pack028_sector_pathway.gl_ipcc_sector_pathways
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_ipp_service_bypass
    ON pack028_sector_pathway.gl_ipcc_sector_pathways
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_ipcc_sector_pathways TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_ipcc_sector_pathways IS
    'IPCC AR6 sector-specific pathway reference data with SSP scenarios, carbon budgets, GWP values (AR6), and emission factors from IPCC 2006 Guidelines with 2019 refinements.';

COMMENT ON COLUMN pack028_sector_pathway.gl_ipcc_sector_pathways.ipcc_pathway_id IS 'Unique IPCC pathway record identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_ipcc_sector_pathways.ssp_scenario IS 'IPCC SSP scenario: SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5.';
COMMENT ON COLUMN pack028_sector_pathway.gl_ipcc_sector_pathways.remaining_carbon_budget_gtco2 IS 'Remaining global carbon budget in GtCO2 from budget start year.';
COMMENT ON COLUMN pack028_sector_pathway.gl_ipcc_sector_pathways.gwp_ch4_100yr IS 'IPCC AR6 Global Warming Potential for CH4 over 100 years (default 27.9).';
COMMENT ON COLUMN pack028_sector_pathway.gl_ipcc_sector_pathways.sector_emission_factors IS 'JSONB of IPCC sector-specific emission factors from 2006/2019 guidelines.';
COMMENT ON COLUMN pack028_sector_pathway.gl_ipcc_sector_pathways.company_budget_alignment IS 'Company alignment with sector carbon budget: WELL_WITHIN to SIGNIFICANTLY_EXCEEDING.';
COMMENT ON COLUMN pack028_sector_pathway.gl_ipcc_sector_pathways.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
