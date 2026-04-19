-- =============================================================================
-- V170: PACK-027 Enterprise Net Zero - Scenario Models
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    005 of 015
-- Date:         March 2026
--
-- Monte Carlo scenario modeling for enterprise decarbonization pathway
-- analysis with 1.5C/2C/BAU/Custom scenarios, probability distributions,
-- sensitivity analysis, and investment requirements modeling.
--
-- Tables (1):
--   1. pack027_enterprise_net_zero.gl_scenario_models
--
-- Previous: V169__PACK027_sbti_targets.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_scenario_models
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_scenario_models (
    scenario_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    baseline_id                 UUID            REFERENCES pack027_enterprise_net_zero.gl_enterprise_baselines(baseline_id) ON DELETE SET NULL,
    -- Scenario definition
    scenario_name               VARCHAR(255)    NOT NULL,
    scenario_type               VARCHAR(30)     NOT NULL,
    pathway                     VARCHAR(30)     NOT NULL,
    temperature_alignment       VARCHAR(10),
    description                 TEXT,
    -- Simulation parameters
    simulation_runs             INTEGER         DEFAULT 10000,
    sampling_method             VARCHAR(30)     DEFAULT 'LATIN_HYPERCUBE',
    random_seed                 INTEGER,
    time_horizon_start          INTEGER         NOT NULL,
    time_horizon_end            INTEGER         NOT NULL DEFAULT 2050,
    -- Input assumptions
    assumptions                 JSONB           NOT NULL DEFAULT '{}',
    carbon_price_trajectory     JSONB           DEFAULT '{}',
    technology_adoption_curves  JSONB           DEFAULT '{}',
    policy_assumptions          JSONB           DEFAULT '{}',
    energy_price_trajectories   JSONB           DEFAULT '{}',
    grid_decarbonization_rates  JSONB           DEFAULT '{}',
    supplier_engagement_rates   JSONB           DEFAULT '{}',
    -- Results
    results                     JSONB           DEFAULT '{}',
    annual_emissions_trajectory JSONB           DEFAULT '{}',
    -- Statistical results
    result_mean_tco2e_2030      DECIMAL(18,4),
    result_median_tco2e_2030    DECIMAL(18,4),
    result_p10_tco2e_2030       DECIMAL(18,4),
    result_p25_tco2e_2030       DECIMAL(18,4),
    result_p75_tco2e_2030       DECIMAL(18,4),
    result_p90_tco2e_2030       DECIMAL(18,4),
    result_mean_tco2e_2050      DECIMAL(18,4),
    result_median_tco2e_2050    DECIMAL(18,4),
    result_p10_tco2e_2050       DECIMAL(18,4),
    result_p90_tco2e_2050       DECIMAL(18,4),
    -- Target achievement probability
    prob_near_term_target       DECIMAL(6,2),
    prob_long_term_target       DECIMAL(6,2),
    prob_net_zero               DECIMAL(6,2),
    -- Sensitivity analysis
    sensitivity_analysis        JSONB           DEFAULT '{}',
    top_10_drivers              JSONB           DEFAULT '{}',
    sobol_indices               JSONB           DEFAULT '{}',
    -- Investment requirements
    total_investment_required   DECIMAL(18,2),
    investment_by_lever         JSONB           DEFAULT '{}',
    marginal_abatement_cost     JSONB           DEFAULT '{}',
    stranded_asset_risk_usd     DECIMAL(18,2),
    -- Execution
    execution_status            VARCHAR(30)     DEFAULT 'DRAFT',
    execution_started_at        TIMESTAMPTZ,
    execution_completed_at      TIMESTAMPTZ,
    execution_time_seconds      DECIMAL(10,2),
    -- Metadata
    version                     INTEGER         DEFAULT 1,
    is_active                   BOOLEAN         DEFAULT TRUE,
    comparison_group_id         UUID,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_sm_scenario_type CHECK (
        scenario_type IN ('AGGRESSIVE', 'MODERATE', 'CONSERVATIVE', 'BAU', 'CUSTOM',
                          'REGULATORY_CHANGE', 'TECHNOLOGY_BREAKTHROUGH', 'STRESS_TEST')
    ),
    CONSTRAINT chk_p027_sm_pathway CHECK (
        pathway IN ('1.5C', '2C', 'WELL_BELOW_2C', 'BAU', 'NDC_ALIGNED', 'CUSTOM')
    ),
    CONSTRAINT chk_p027_sm_temperature CHECK (
        temperature_alignment IS NULL OR temperature_alignment IN ('1.5C', '2.0C', '2.5C', '3.0C', '4.0C')
    ),
    CONSTRAINT chk_p027_sm_simulation_runs CHECK (
        simulation_runs >= 100 AND simulation_runs <= 100000
    ),
    CONSTRAINT chk_p027_sm_sampling CHECK (
        sampling_method IN ('LATIN_HYPERCUBE', 'MONTE_CARLO', 'SOBOL_SEQUENCE', 'STRATIFIED')
    ),
    CONSTRAINT chk_p027_sm_time_horizon CHECK (
        time_horizon_start >= 2020 AND time_horizon_end <= 2100
        AND time_horizon_end > time_horizon_start
    ),
    CONSTRAINT chk_p027_sm_probabilities CHECK (
        (prob_near_term_target IS NULL OR (prob_near_term_target >= 0 AND prob_near_term_target <= 100)) AND
        (prob_long_term_target IS NULL OR (prob_long_term_target >= 0 AND prob_long_term_target <= 100)) AND
        (prob_net_zero IS NULL OR (prob_net_zero >= 0 AND prob_net_zero <= 100))
    ),
    CONSTRAINT chk_p027_sm_execution_status CHECK (
        execution_status IN ('DRAFT', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p027_sm_version CHECK (
        version >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_sm_company            ON pack027_enterprise_net_zero.gl_scenario_models(company_id);
CREATE INDEX idx_p027_sm_tenant             ON pack027_enterprise_net_zero.gl_scenario_models(tenant_id);
CREATE INDEX idx_p027_sm_baseline           ON pack027_enterprise_net_zero.gl_scenario_models(baseline_id);
CREATE INDEX idx_p027_sm_type               ON pack027_enterprise_net_zero.gl_scenario_models(scenario_type);
CREATE INDEX idx_p027_sm_pathway            ON pack027_enterprise_net_zero.gl_scenario_models(pathway);
CREATE INDEX idx_p027_sm_temperature        ON pack027_enterprise_net_zero.gl_scenario_models(temperature_alignment);
CREATE INDEX idx_p027_sm_execution          ON pack027_enterprise_net_zero.gl_scenario_models(execution_status);
CREATE INDEX idx_p027_sm_active             ON pack027_enterprise_net_zero.gl_scenario_models(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_sm_comparison         ON pack027_enterprise_net_zero.gl_scenario_models(comparison_group_id);
CREATE INDEX idx_p027_sm_prob_nt            ON pack027_enterprise_net_zero.gl_scenario_models(prob_near_term_target);
CREATE INDEX idx_p027_sm_prob_nz            ON pack027_enterprise_net_zero.gl_scenario_models(prob_net_zero);
CREATE INDEX idx_p027_sm_created            ON pack027_enterprise_net_zero.gl_scenario_models(created_at DESC);
CREATE INDEX idx_p027_sm_assumptions        ON pack027_enterprise_net_zero.gl_scenario_models USING GIN(assumptions);
CREATE INDEX idx_p027_sm_results            ON pack027_enterprise_net_zero.gl_scenario_models USING GIN(results);
CREATE INDEX idx_p027_sm_sensitivity        ON pack027_enterprise_net_zero.gl_scenario_models USING GIN(sensitivity_analysis);
CREATE INDEX idx_p027_sm_metadata           ON pack027_enterprise_net_zero.gl_scenario_models USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_scenario_models_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_scenario_models
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_scenario_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_sm_tenant_isolation
    ON pack027_enterprise_net_zero.gl_scenario_models
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_sm_service_bypass
    ON pack027_enterprise_net_zero.gl_scenario_models
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_scenario_models TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_scenario_models IS
    'Monte Carlo scenario modeling for enterprise decarbonization pathway analysis with probability distributions, sensitivity analysis, and investment requirements.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.scenario_id IS 'Unique scenario identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.scenario_type IS 'Scenario type: AGGRESSIVE (1.5C), MODERATE (2C), CONSERVATIVE, BAU, CUSTOM, etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.pathway IS 'Temperature pathway alignment: 1.5C, 2C, WELL_BELOW_2C, BAU, NDC_ALIGNED, CUSTOM.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.simulation_runs IS 'Number of Monte Carlo simulation runs (default 10,000).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.assumptions IS 'JSONB of input assumptions including all parameter distributions.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.results IS 'JSONB of full simulation results including per-year statistics.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.prob_near_term_target IS 'Probability (0-100%) of achieving near-term SBTi target under this scenario.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.sobol_indices IS 'Sobol sensitivity indices identifying key drivers of outcome uncertainty.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scenario_models.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
