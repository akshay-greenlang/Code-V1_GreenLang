-- =============================================================================
-- V193: PACK-028 Sector Pathway Pack - Multi-Scenario Modeling
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    013 of 015
-- Date:         March 2026
--
-- Multi-scenario modeling infrastructure with scenario definitions,
-- parameter sets, and result storage for sector pathway analysis
-- across 5+ climate scenarios with sensitivity analysis.
--
-- Tables (3):
--   1. pack028_sector_pathway.gl_scenario_definitions
--   2. pack028_sector_pathway.gl_scenario_parameters
--   3. pack028_sector_pathway.gl_scenario_results
--
-- Previous: V192__PACK028_sector_reference_data.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_scenario_definitions
-- =============================================================================
-- Scenario definition metadata for multi-scenario pathway modeling.

CREATE TABLE pack028_sector_pathway.gl_scenario_definitions (
    scenario_def_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    -- Scenario identification
    scenario_name               VARCHAR(255)    NOT NULL,
    scenario_code               VARCHAR(40)     NOT NULL,
    scenario_type               VARCHAR(30)     NOT NULL,
    -- Climate target
    temperature_target          VARCHAR(10)     NOT NULL,
    probability_pct             DECIMAL(5,2),
    -- Source reference
    source_framework            VARCHAR(30)     NOT NULL,
    source_version              VARCHAR(20),
    source_year                 INTEGER,
    -- Timeframe
    start_year                  INTEGER         NOT NULL DEFAULT 2020,
    end_year                    INTEGER         NOT NULL DEFAULT 2050,
    time_step                   VARCHAR(10)     DEFAULT 'ANNUAL',
    -- Description
    description                 TEXT,
    key_assumptions_summary     TEXT,
    -- Global context
    global_emissions_2030_gtco2 DECIMAL(10,2),
    global_emissions_2050_gtco2 DECIMAL(10,2),
    global_net_zero_year        INTEGER,
    -- Energy system
    fossil_fuel_phase_out       JSONB           DEFAULT '{}',
    renewable_deployment        JSONB           DEFAULT '{}',
    electrification_rate        JSONB           DEFAULT '{}',
    hydrogen_deployment         JSONB           DEFAULT '{}',
    ccs_deployment              JSONB           DEFAULT '{}',
    -- Carbon pricing
    carbon_price_trajectory     JSONB           DEFAULT '{}',
    -- Policy assumptions
    policy_assumptions          JSONB           DEFAULT '{}',
    -- Technology assumptions
    technology_assumptions      JSONB           DEFAULT '{}',
    -- Demand assumptions
    demand_assumptions          JSONB           DEFAULT '{}',
    -- Status
    scenario_status             VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    is_reference                BOOLEAN         DEFAULT FALSE,
    is_default                  BOOLEAN         DEFAULT FALSE,
    -- Comparison group
    comparison_group_id         UUID,
    comparison_order            INTEGER         DEFAULT 1,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_sd_scenario_type CHECK (
        scenario_type IN ('NZE_1_5C', 'WB2C', '2C', 'APS', 'STEPS', 'NDC_ALIGNED',
                          'BAU', 'AGGRESSIVE', 'MODERATE', 'CONSERVATIVE', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sd_temperature CHECK (
        temperature_target IN ('1.5C', 'WB2C', '2.0C', '2.5C', '3.0C', '4.0C')
    ),
    CONSTRAINT chk_p028_sd_source CHECK (
        source_framework IN ('IEA', 'IPCC', 'SBTI', 'NGFS', 'OECM', 'COMPANY', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sd_time_step CHECK (
        time_step IN ('ANNUAL', 'SEMI_ANNUAL', 'FIVE_YEAR')
    ),
    CONSTRAINT chk_p028_sd_status CHECK (
        scenario_status IN ('DRAFT', 'ACTIVE', 'ARCHIVED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p028_sd_year_range CHECK (
        start_year >= 2000 AND end_year <= 2150 AND end_year > start_year
    ),
    CONSTRAINT chk_p028_sd_probability CHECK (
        probability_pct IS NULL OR (probability_pct >= 0 AND probability_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p028_sd_tenant             ON pack028_sector_pathway.gl_scenario_definitions(tenant_id);
CREATE INDEX idx_p028_sd_company            ON pack028_sector_pathway.gl_scenario_definitions(company_id);
CREATE INDEX idx_p028_sd_code               ON pack028_sector_pathway.gl_scenario_definitions(scenario_code);
CREATE INDEX idx_p028_sd_type               ON pack028_sector_pathway.gl_scenario_definitions(scenario_type);
CREATE INDEX idx_p028_sd_temperature        ON pack028_sector_pathway.gl_scenario_definitions(temperature_target);
CREATE INDEX idx_p028_sd_source             ON pack028_sector_pathway.gl_scenario_definitions(source_framework);
CREATE INDEX idx_p028_sd_status             ON pack028_sector_pathway.gl_scenario_definitions(scenario_status);
CREATE INDEX idx_p028_sd_reference          ON pack028_sector_pathway.gl_scenario_definitions(is_reference) WHERE is_reference = TRUE;
CREATE INDEX idx_p028_sd_default            ON pack028_sector_pathway.gl_scenario_definitions(is_default) WHERE is_default = TRUE;
CREATE INDEX idx_p028_sd_comparison         ON pack028_sector_pathway.gl_scenario_definitions(comparison_group_id);
CREATE INDEX idx_p028_sd_company_type       ON pack028_sector_pathway.gl_scenario_definitions(company_id, scenario_type);
CREATE INDEX idx_p028_sd_created            ON pack028_sector_pathway.gl_scenario_definitions(created_at DESC);
CREATE INDEX idx_p028_sd_carbon_price       ON pack028_sector_pathway.gl_scenario_definitions USING GIN(carbon_price_trajectory);
CREATE INDEX idx_p028_sd_metadata           ON pack028_sector_pathway.gl_scenario_definitions USING GIN(metadata);

CREATE TRIGGER trg_p028_scenario_definitions_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_scenario_definitions
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

ALTER TABLE pack028_sector_pathway.gl_scenario_definitions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_sd_tenant_isolation
    ON pack028_sector_pathway.gl_scenario_definitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_sd_service_bypass
    ON pack028_sector_pathway.gl_scenario_definitions
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_scenario_definitions TO PUBLIC;

-- =============================================================================
-- Table 2: pack028_sector_pathway.gl_scenario_parameters
-- =============================================================================
-- Per-sector parameter values for each scenario definition.

CREATE TABLE pack028_sector_pathway.gl_scenario_parameters (
    parameter_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    scenario_def_id             UUID            NOT NULL REFERENCES pack028_sector_pathway.gl_scenario_definitions(scenario_def_id) ON DELETE CASCADE,
    -- Sector context
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Parameter definition
    parameter_name              VARCHAR(255)    NOT NULL,
    parameter_code              VARCHAR(60)     NOT NULL,
    parameter_category          VARCHAR(50)     NOT NULL,
    parameter_description       TEXT,
    -- Parameter values
    value_type                  VARCHAR(20)     NOT NULL DEFAULT 'NUMERIC',
    numeric_value               DECIMAL(18,8),
    string_value                VARCHAR(500),
    boolean_value               BOOLEAN,
    json_value                  JSONB,
    -- Trajectory (for time-varying parameters)
    is_time_varying             BOOLEAN         DEFAULT FALSE,
    annual_values               JSONB           DEFAULT '{}',
    -- Uncertainty
    distribution_type           VARCHAR(20),
    value_low                   DECIMAL(18,8),
    value_high                  DECIMAL(18,8),
    std_deviation               DECIMAL(18,8),
    -- Units
    unit                        VARCHAR(50),
    -- Sensitivity
    sensitivity_rank            INTEGER,
    sensitivity_impact_pct      DECIMAL(8,4),
    is_key_driver               BOOLEAN         DEFAULT FALSE,
    -- Metadata
    source                      VARCHAR(200),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_spm_category CHECK (
        parameter_category IN (
            'CARBON_PRICE', 'ENERGY_PRICE', 'TECHNOLOGY_COST', 'TECHNOLOGY_ADOPTION',
            'DEMAND_GROWTH', 'POPULATION', 'GDP', 'POLICY', 'GRID_CARBON_INTENSITY',
            'RENEWABLE_CAPACITY', 'HYDROGEN', 'CCS', 'ELECTRIFICATION',
            'FUEL_MIX', 'EFFICIENCY', 'SUPPLY_CHAIN', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p028_spm_value_type CHECK (
        value_type IN ('NUMERIC', 'STRING', 'BOOLEAN', 'JSON', 'TRAJECTORY')
    ),
    CONSTRAINT chk_p028_spm_distribution CHECK (
        distribution_type IS NULL OR distribution_type IN (
            'NORMAL', 'LOGNORMAL', 'UNIFORM', 'TRIANGULAR', 'BETA', 'POINT_ESTIMATE'
        )
    )
);

-- Indexes
CREATE INDEX idx_p028_spm_tenant            ON pack028_sector_pathway.gl_scenario_parameters(tenant_id);
CREATE INDEX idx_p028_spm_scenario          ON pack028_sector_pathway.gl_scenario_parameters(scenario_def_id);
CREATE INDEX idx_p028_spm_sector            ON pack028_sector_pathway.gl_scenario_parameters(sector_code);
CREATE INDEX idx_p028_spm_param_code        ON pack028_sector_pathway.gl_scenario_parameters(parameter_code);
CREATE INDEX idx_p028_spm_category          ON pack028_sector_pathway.gl_scenario_parameters(parameter_category);
CREATE INDEX idx_p028_spm_key_driver        ON pack028_sector_pathway.gl_scenario_parameters(is_key_driver) WHERE is_key_driver = TRUE;
CREATE INDEX idx_p028_spm_scenario_sector   ON pack028_sector_pathway.gl_scenario_parameters(scenario_def_id, sector_code);
CREATE INDEX idx_p028_spm_sensitivity       ON pack028_sector_pathway.gl_scenario_parameters(sensitivity_rank) WHERE sensitivity_rank IS NOT NULL;
CREATE INDEX idx_p028_spm_time_varying      ON pack028_sector_pathway.gl_scenario_parameters(is_time_varying) WHERE is_time_varying = TRUE;
CREATE INDEX idx_p028_spm_annual_vals       ON pack028_sector_pathway.gl_scenario_parameters USING GIN(annual_values);
CREATE INDEX idx_p028_spm_created           ON pack028_sector_pathway.gl_scenario_parameters(created_at DESC);
CREATE INDEX idx_p028_spm_metadata          ON pack028_sector_pathway.gl_scenario_parameters USING GIN(metadata);

CREATE TRIGGER trg_p028_scenario_parameters_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_scenario_parameters
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

ALTER TABLE pack028_sector_pathway.gl_scenario_parameters ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_spm_tenant_isolation
    ON pack028_sector_pathway.gl_scenario_parameters
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_spm_service_bypass
    ON pack028_sector_pathway.gl_scenario_parameters
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_scenario_parameters TO PUBLIC;

-- =============================================================================
-- Table 3: pack028_sector_pathway.gl_scenario_results
-- =============================================================================
-- Computed scenario results per sector with intensity trajectories,
-- emissions projections, investment requirements, and risk metrics.

CREATE TABLE pack028_sector_pathway.gl_scenario_results (
    result_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    scenario_def_id             UUID            NOT NULL REFERENCES pack028_sector_pathway.gl_scenario_definitions(scenario_def_id) ON DELETE CASCADE,
    pathway_id                  UUID            REFERENCES pack028_sector_pathway.gl_sector_pathways(pathway_id) ON DELETE SET NULL,
    -- Sector context
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Computation metadata
    computation_date            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    computation_version         INTEGER         DEFAULT 1,
    computation_status          VARCHAR(20)     NOT NULL DEFAULT 'COMPLETED',
    computation_time_ms         INTEGER,
    -- Intensity trajectory
    intensity_trajectory        JSONB           NOT NULL DEFAULT '{}',
    intensity_2025              DECIMAL(18,8),
    intensity_2030              DECIMAL(18,8),
    intensity_2035              DECIMAL(18,8),
    intensity_2040              DECIMAL(18,8),
    intensity_2050              DECIMAL(18,8),
    intensity_unit              VARCHAR(80),
    -- Emissions trajectory
    emissions_trajectory        JSONB           DEFAULT '{}',
    emissions_2025_tco2e        DECIMAL(18,4),
    emissions_2030_tco2e        DECIMAL(18,4),
    emissions_2040_tco2e        DECIMAL(18,4),
    emissions_2050_tco2e        DECIMAL(18,4),
    cumulative_emissions_tco2e  DECIMAL(18,4),
    -- Reduction metrics
    total_reduction_pct         DECIMAL(6,2),
    annual_reduction_rate_pct   DECIMAL(8,4),
    net_zero_year               INTEGER,
    -- Investment requirements
    total_investment_usd        DECIMAL(18,2),
    annual_investment_schedule  JSONB           DEFAULT '{}',
    investment_by_category      JSONB           DEFAULT '{}',
    incremental_vs_bau_usd      DECIMAL(18,2),
    npv_investment_usd          DECIMAL(18,2),
    -- Technology deployment
    technology_deployment       JSONB           DEFAULT '{}',
    key_technology_milestones   JSONB           DEFAULT '[]',
    -- Abatement breakdown
    abatement_by_lever          JSONB           DEFAULT '{}',
    total_abatement_tco2e       DECIMAL(18,4),
    -- Risk metrics
    transition_risk_score       DECIMAL(5,2),
    physical_risk_score         DECIMAL(5,2),
    stranded_asset_risk_usd     DECIMAL(18,2),
    carbon_price_exposure_usd   DECIMAL(18,2),
    -- Pathway alignment
    sbti_alignment_score        DECIMAL(5,2),
    iea_alignment_score         DECIMAL(5,2),
    ipcc_alignment_score        DECIMAL(5,2),
    -- Sensitivity results
    sensitivity_analysis        JSONB           DEFAULT '{}',
    top_5_drivers               JSONB           DEFAULT '[]',
    -- Uncertainty range
    intensity_2050_p10          DECIMAL(18,8),
    intensity_2050_p50          DECIMAL(18,8),
    intensity_2050_p90          DECIMAL(18,8),
    -- Metadata
    is_latest                   BOOLEAN         DEFAULT TRUE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_sr_status CHECK (
        computation_status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p028_sr_version CHECK (
        computation_version >= 1
    ),
    CONSTRAINT chk_p028_sr_sbti_score CHECK (
        sbti_alignment_score IS NULL OR (sbti_alignment_score >= 0 AND sbti_alignment_score <= 100)
    ),
    CONSTRAINT chk_p028_sr_iea_score CHECK (
        iea_alignment_score IS NULL OR (iea_alignment_score >= 0 AND iea_alignment_score <= 100)
    ),
    CONSTRAINT chk_p028_sr_ipcc_score CHECK (
        ipcc_alignment_score IS NULL OR (ipcc_alignment_score >= 0 AND ipcc_alignment_score <= 100)
    ),
    CONSTRAINT chk_p028_sr_transition_risk CHECK (
        transition_risk_score IS NULL OR (transition_risk_score >= 0 AND transition_risk_score <= 100)
    ),
    CONSTRAINT chk_p028_sr_physical_risk CHECK (
        physical_risk_score IS NULL OR (physical_risk_score >= 0 AND physical_risk_score <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p028_sr_tenant             ON pack028_sector_pathway.gl_scenario_results(tenant_id);
CREATE INDEX idx_p028_sr_company            ON pack028_sector_pathway.gl_scenario_results(company_id);
CREATE INDEX idx_p028_sr_scenario           ON pack028_sector_pathway.gl_scenario_results(scenario_def_id);
CREATE INDEX idx_p028_sr_pathway            ON pack028_sector_pathway.gl_scenario_results(pathway_id);
CREATE INDEX idx_p028_sr_sector             ON pack028_sector_pathway.gl_scenario_results(sector_code);
CREATE INDEX idx_p028_sr_status             ON pack028_sector_pathway.gl_scenario_results(computation_status);
CREATE INDEX idx_p028_sr_latest             ON pack028_sector_pathway.gl_scenario_results(is_latest) WHERE is_latest = TRUE;
CREATE INDEX idx_p028_sr_company_sector     ON pack028_sector_pathway.gl_scenario_results(company_id, sector_code);
CREATE INDEX idx_p028_sr_company_scenario   ON pack028_sector_pathway.gl_scenario_results(company_id, scenario_def_id);
CREATE INDEX idx_p028_sr_nz_year            ON pack028_sector_pathway.gl_scenario_results(net_zero_year) WHERE net_zero_year IS NOT NULL;
CREATE INDEX idx_p028_sr_sbti_score         ON pack028_sector_pathway.gl_scenario_results(sbti_alignment_score DESC NULLS LAST);
CREATE INDEX idx_p028_sr_created            ON pack028_sector_pathway.gl_scenario_results(created_at DESC);
CREATE INDEX idx_p028_sr_intensity_traj     ON pack028_sector_pathway.gl_scenario_results USING GIN(intensity_trajectory);
CREATE INDEX idx_p028_sr_abatement          ON pack028_sector_pathway.gl_scenario_results USING GIN(abatement_by_lever);
CREATE INDEX idx_p028_sr_invest_cat         ON pack028_sector_pathway.gl_scenario_results USING GIN(investment_by_category);
CREATE INDEX idx_p028_sr_sensitivity        ON pack028_sector_pathway.gl_scenario_results USING GIN(sensitivity_analysis);
CREATE INDEX idx_p028_sr_metadata           ON pack028_sector_pathway.gl_scenario_results USING GIN(metadata);

CREATE TRIGGER trg_p028_scenario_results_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_scenario_results
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

ALTER TABLE pack028_sector_pathway.gl_scenario_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_sr_tenant_isolation
    ON pack028_sector_pathway.gl_scenario_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_sr_service_bypass
    ON pack028_sector_pathway.gl_scenario_results
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_scenario_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_scenario_definitions IS
    'Climate scenario definitions for multi-scenario pathway modeling with temperature targets, source frameworks (IEA/IPCC/SBTi/NGFS), and global assumption sets.';

COMMENT ON TABLE pack028_sector_pathway.gl_scenario_parameters IS
    'Per-sector scenario parameter values including carbon prices, technology costs, demand growth, and policy assumptions with uncertainty distributions.';

COMMENT ON TABLE pack028_sector_pathway.gl_scenario_results IS
    'Computed scenario results per sector with intensity/emissions trajectories, investment requirements, abatement breakdown, risk metrics, and pathway alignment scores.';
