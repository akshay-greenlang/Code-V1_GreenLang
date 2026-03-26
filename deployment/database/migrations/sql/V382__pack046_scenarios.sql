-- =============================================================================
-- V382: PACK-046 Intensity Metrics Pack - Scenario Analysis & Monte Carlo
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for scenario analysis and Monte Carlo simulation of future
-- intensity pathways. Scenarios model different assumptions (efficiency
-- improvements, growth rates, structural changes, methodology updates) to
-- project intensity evolution. Monte Carlo simulation uses probability
-- distributions for key inputs to quantify the likelihood of achieving
-- intensity targets. Results include percentile bands (P10-P95), mean,
-- median, and probability-of-target for each projection year.
--
-- Tables (3):
--   1. ghg_intensity.gl_im_scenarios
--   2. ghg_intensity.gl_im_scenario_results
--   3. ghg_intensity.gl_im_monte_carlo_runs
--
-- Also includes: indexes, RLS, comments.
-- Previous: V381__pack046_targets.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_scenarios
-- =============================================================================
-- Scenario definitions for intensity projection analysis. Each scenario
-- defines a type (efficiency, growth, structural, methodology, combined),
-- parameter assumptions as JSON, a base year anchor, and the projection
-- horizon. Monte Carlo iterations control the simulation resolution.

CREATE TABLE ghg_intensity.gl_im_scenarios (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    scenario_name               VARCHAR(255)    NOT NULL,
    scenario_type               VARCHAR(30)     NOT NULL,
    description                 TEXT,
    denominator_code            VARCHAR(50)     NOT NULL,
    scope_inclusion             VARCHAR(50)     NOT NULL,
    parameters                  JSONB           NOT NULL DEFAULT '{}',
    base_year                   INTEGER         NOT NULL,
    projection_years            INTEGER         NOT NULL DEFAULT 10,
    monte_carlo_iterations      INTEGER         NOT NULL DEFAULT 10000,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p046_scn_type CHECK (
        scenario_type IN (
            'EFFICIENCY', 'GROWTH', 'STRUCTURAL', 'METHODOLOGY',
            'COMBINED', 'BAU', 'OPTIMISTIC', 'PESSIMISTIC', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p046_scn_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_scn_base_year CHECK (
        base_year >= 2000 AND base_year <= 2100
    ),
    CONSTRAINT chk_p046_scn_projection CHECK (
        projection_years >= 1 AND projection_years <= 50
    ),
    CONSTRAINT chk_p046_scn_iterations CHECK (
        monte_carlo_iterations >= 100 AND monte_carlo_iterations <= 1000000
    ),
    CONSTRAINT chk_p046_scn_status CHECK (
        status IN ('DRAFT', 'RUNNING', 'COMPLETED', 'FAILED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p046_scn_org_name UNIQUE (org_id, config_id, scenario_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_scn_tenant           ON ghg_intensity.gl_im_scenarios(tenant_id);
CREATE INDEX idx_p046_scn_org              ON ghg_intensity.gl_im_scenarios(org_id);
CREATE INDEX idx_p046_scn_config           ON ghg_intensity.gl_im_scenarios(config_id);
CREATE INDEX idx_p046_scn_type             ON ghg_intensity.gl_im_scenarios(scenario_type);
CREATE INDEX idx_p046_scn_denom            ON ghg_intensity.gl_im_scenarios(denominator_code);
CREATE INDEX idx_p046_scn_scope            ON ghg_intensity.gl_im_scenarios(scope_inclusion);
CREATE INDEX idx_p046_scn_status           ON ghg_intensity.gl_im_scenarios(status);
CREATE INDEX idx_p046_scn_active           ON ghg_intensity.gl_im_scenarios(is_active) WHERE is_active = true;
CREATE INDEX idx_p046_scn_created          ON ghg_intensity.gl_im_scenarios(created_at DESC);
CREATE INDEX idx_p046_scn_params           ON ghg_intensity.gl_im_scenarios USING GIN(parameters);

-- Composite: org + config for listing
CREATE INDEX idx_p046_scn_org_config       ON ghg_intensity.gl_im_scenarios(org_id, config_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_scn_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_scenarios
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_scenario_results
-- =============================================================================
-- Aggregated scenario simulation results per projection year. Each row
-- contains the percentile distribution (P10, P25, median, P75, P90, P95)
-- of projected intensity, plus mean values for emissions and denominator.
-- The probability_of_target field indicates the Monte Carlo probability
-- of meeting the organisation's intensity target in that year.

CREATE TABLE ghg_intensity.gl_im_scenario_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id                 UUID            NOT NULL REFERENCES ghg_intensity.gl_im_scenarios(id) ON DELETE CASCADE,
    year                        INTEGER         NOT NULL,
    intensity_mean              NUMERIC(20,10),
    intensity_median            NUMERIC(20,10),
    intensity_p5                NUMERIC(20,10),
    intensity_p10               NUMERIC(20,10),
    intensity_p25               NUMERIC(20,10),
    intensity_p75               NUMERIC(20,10),
    intensity_p90               NUMERIC(20,10),
    intensity_p95               NUMERIC(20,10),
    probability_of_target       NUMERIC(10,6),
    emissions_mean              NUMERIC(20,6),
    emissions_median            NUMERIC(20,6),
    denominator_mean            NUMERIC(20,6),
    denominator_median          NUMERIC(20,6),
    result_metadata             JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_sr_year CHECK (
        year >= 2000 AND year <= 2150
    ),
    CONSTRAINT chk_p046_sr_probability CHECK (
        probability_of_target IS NULL OR (probability_of_target >= 0 AND probability_of_target <= 100)
    ),
    CONSTRAINT uq_p046_sr_scenario_year UNIQUE (scenario_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_sr_scenario          ON ghg_intensity.gl_im_scenario_results(scenario_id);
CREATE INDEX idx_p046_sr_year              ON ghg_intensity.gl_im_scenario_results(year);
CREATE INDEX idx_p046_sr_calculated        ON ghg_intensity.gl_im_scenario_results(calculated_at DESC);
CREATE INDEX idx_p046_sr_created           ON ghg_intensity.gl_im_scenario_results(created_at DESC);

-- Composite: scenario + year for ordered retrieval
CREATE INDEX idx_p046_sr_scenario_year_ord ON ghg_intensity.gl_im_scenario_results(scenario_id, year ASC);

-- =============================================================================
-- Table 3: ghg_intensity.gl_im_monte_carlo_runs
-- =============================================================================
-- Monte Carlo simulation run metadata and aggregated statistics. Each run
-- represents a complete simulation execution with its configuration
-- (distribution parameters, random seed, iteration count), summary
-- statistics, and processing performance metrics. Multiple runs may be
-- executed per scenario for convergence validation.

CREATE TABLE ghg_intensity.gl_im_monte_carlo_runs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id                 UUID            NOT NULL REFERENCES ghg_intensity.gl_im_scenarios(id) ON DELETE CASCADE,
    run_number                  INTEGER         NOT NULL DEFAULT 1,
    run_config                  JSONB           NOT NULL,
    run_summary                 JSONB           NOT NULL,
    iterations_completed        INTEGER         NOT NULL,
    iterations_converged        BOOLEAN         NOT NULL DEFAULT true,
    convergence_tolerance       NUMERIC(10,8),
    seed_value                  BIGINT,
    processing_time_ms          INTEGER,
    peak_memory_mb              NUMERIC(10,2),
    provenance_hash             VARCHAR(64)     NOT NULL,
    completed_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_mc_iterations CHECK (
        iterations_completed > 0
    ),
    CONSTRAINT chk_p046_mc_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    ),
    CONSTRAINT chk_p046_mc_memory CHECK (
        peak_memory_mb IS NULL OR peak_memory_mb >= 0
    ),
    CONSTRAINT uq_p046_mc_scenario_run UNIQUE (scenario_id, run_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_mc_scenario          ON ghg_intensity.gl_im_monte_carlo_runs(scenario_id);
CREATE INDEX idx_p046_mc_completed         ON ghg_intensity.gl_im_monte_carlo_runs(completed_at DESC);
CREATE INDEX idx_p046_mc_created           ON ghg_intensity.gl_im_monte_carlo_runs(created_at DESC);
CREATE INDEX idx_p046_mc_converged         ON ghg_intensity.gl_im_monte_carlo_runs(iterations_converged);
CREATE INDEX idx_p046_mc_config            ON ghg_intensity.gl_im_monte_carlo_runs USING GIN(run_config);
CREATE INDEX idx_p046_mc_summary           ON ghg_intensity.gl_im_monte_carlo_runs USING GIN(run_summary);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_scenario_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_monte_carlo_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_scn_tenant_isolation
    ON ghg_intensity.gl_im_scenarios
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_scn_service_bypass
    ON ghg_intensity.gl_im_scenarios
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Results and runs inherit access via scenario_id FK
CREATE POLICY p046_sr_service_bypass
    ON ghg_intensity.gl_im_scenario_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_mc_service_bypass
    ON ghg_intensity.gl_im_monte_carlo_runs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_scenarios TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_scenario_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_monte_carlo_runs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_scenarios IS
    'Scenario definitions for intensity projection analysis with efficiency, growth, structural, and combined scenario types.';
COMMENT ON TABLE ghg_intensity.gl_im_scenario_results IS
    'Aggregated scenario simulation results per projection year with percentile distribution and probability-of-target.';
COMMENT ON TABLE ghg_intensity.gl_im_monte_carlo_runs IS
    'Monte Carlo simulation run metadata with configuration, summary statistics, convergence status, and performance metrics.';

COMMENT ON COLUMN ghg_intensity.gl_im_scenarios.scenario_type IS 'EFFICIENCY (technology), GROWTH (scale), STRUCTURAL (mix change), METHODOLOGY (factor updates), COMBINED, BAU, OPTIMISTIC, PESSIMISTIC, CUSTOM.';
COMMENT ON COLUMN ghg_intensity.gl_im_scenarios.parameters IS 'JSON scenario parameters: {"emission_growth_rate": {"dist": "normal", "mean": 0.02, "std": 0.01}, "efficiency_improvement": {"dist": "triangular", "min": 0.01, "mode": 0.03, "max": 0.05}}.';
COMMENT ON COLUMN ghg_intensity.gl_im_scenarios.monte_carlo_iterations IS 'Number of Monte Carlo iterations (100-1,000,000). Default 10,000 balances accuracy and performance.';
COMMENT ON COLUMN ghg_intensity.gl_im_scenario_results.probability_of_target IS 'Percentage of Monte Carlo iterations where projected intensity meets or beats the target intensity.';
COMMENT ON COLUMN ghg_intensity.gl_im_monte_carlo_runs.run_config IS 'JSON run configuration: distribution parameters, random seed, iteration count, convergence settings.';
COMMENT ON COLUMN ghg_intensity.gl_im_monte_carlo_runs.run_summary IS 'JSON aggregated statistics: mean, median, std, percentiles, convergence metrics across all years.';
COMMENT ON COLUMN ghg_intensity.gl_im_monte_carlo_runs.iterations_converged IS 'Whether the simulation converged within the specified tolerance. False may indicate insufficient iterations.';
