-- =============================================================================
-- V343: PACK-042 Scope 3 Starter Pack - Uncertainty Analysis
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates uncertainty analysis tables for Scope 3 emissions. Implements
-- IPCC Approach 1 (error propagation) and Approach 2 (Monte Carlo) methods
-- adapted for value chain emissions. Tracks per-category uncertainty,
-- sensitivity analysis results, inter-category correlations, and the
-- impact of methodology tier upgrades on uncertainty reduction.
--
-- Tables (5):
--   1. ghg_accounting_scope3.uncertainty_analyses
--   2. ghg_accounting_scope3.category_uncertainties
--   3. ghg_accounting_scope3.sensitivity_results
--   4. ghg_accounting_scope3.correlation_matrix
--   5. ghg_accounting_scope3.tier_upgrade_impacts
--
-- Also includes: indexes, RLS, comments.
-- Previous: V342__pack042_data_quality.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.uncertainty_analyses
-- =============================================================================
-- Top-level uncertainty analysis for a Scope 3 inventory. Supports both
-- analytical (error propagation) and Monte Carlo simulation methods.
-- Stores overall confidence intervals and statistical parameters.

CREATE TABLE ghg_accounting_scope3.uncertainty_analyses (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Analysis parameters
    method                      VARCHAR(30)     NOT NULL DEFAULT 'ANALYTICAL',
    iterations                  INTEGER,
    confidence_level            DECIMAL(3,2)    NOT NULL DEFAULT 0.95,
    seed                        INTEGER,
    -- Overall results
    point_estimate_tco2e        DECIMAL(15,3)   NOT NULL,
    lower_bound_tco2e           DECIMAL(15,3)   NOT NULL,
    upper_bound_tco2e           DECIMAL(15,3)   NOT NULL,
    overall_uncertainty_pct     DECIMAL(8,4)    NOT NULL,
    half_width_tco2e            DECIMAL(15,3),
    -- Monte Carlo statistics
    mc_mean                     DECIMAL(15,3),
    mc_median                   DECIMAL(15,3),
    mc_std_dev                  DECIMAL(15,3),
    mc_skewness                 DECIMAL(8,4),
    mc_kurtosis                 DECIMAL(8,4),
    mc_coefficient_of_var       DECIMAL(8,4),
    mc_convergence_achieved     BOOLEAN,
    mc_convergence_iteration    INTEGER,
    -- Percentiles
    mc_p2_5                     DECIMAL(15,3),
    mc_p5                       DECIMAL(15,3),
    mc_p10                      DECIMAL(15,3),
    mc_p25                      DECIMAL(15,3),
    mc_p75                      DECIMAL(15,3),
    mc_p90                      DECIMAL(15,3),
    mc_p95                      DECIMAL(15,3),
    mc_p97_5                    DECIMAL(15,3),
    -- Quality assessment
    overall_rating              VARCHAR(20),
    recommended_method          VARCHAR(30),
    -- Metadata
    analysis_timestamp          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    analysis_duration_ms        INTEGER,
    software_version            VARCHAR(50),
    analyst                     VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    methodology_reference       VARCHAR(500),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ua_method CHECK (
        method IN ('ANALYTICAL', 'MONTE_CARLO', 'BOTH', 'EXPERT_JUDGMENT')
    ),
    CONSTRAINT chk_p042_ua_confidence CHECK (
        confidence_level >= 0.50 AND confidence_level <= 0.99
    ),
    CONSTRAINT chk_p042_ua_iterations CHECK (
        iterations IS NULL OR (iterations >= 100 AND iterations <= 10000000)
    ),
    CONSTRAINT chk_p042_ua_point_est CHECK (
        point_estimate_tco2e >= 0
    ),
    CONSTRAINT chk_p042_ua_bounds CHECK (
        lower_bound_tco2e <= upper_bound_tco2e
    ),
    CONSTRAINT chk_p042_ua_lower CHECK (
        lower_bound_tco2e >= 0
    ),
    CONSTRAINT chk_p042_ua_uncertainty CHECK (
        overall_uncertainty_pct >= 0 AND overall_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p042_ua_mc_percentiles CHECK (
        mc_p2_5 IS NULL OR mc_p97_5 IS NULL OR mc_p2_5 <= mc_p97_5
    ),
    CONSTRAINT chk_p042_ua_rating CHECK (
        overall_rating IS NULL OR overall_rating IN (
            'VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH'
        )
    ),
    CONSTRAINT chk_p042_ua_duration CHECK (
        analysis_duration_ms IS NULL OR analysis_duration_ms >= 0
    ),
    CONSTRAINT uq_p042_ua_inventory_method UNIQUE (inventory_id, method)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ua_tenant             ON ghg_accounting_scope3.uncertainty_analyses(tenant_id);
CREATE INDEX idx_p042_ua_inventory          ON ghg_accounting_scope3.uncertainty_analyses(inventory_id);
CREATE INDEX idx_p042_ua_method             ON ghg_accounting_scope3.uncertainty_analyses(method);
CREATE INDEX idx_p042_ua_uncertainty        ON ghg_accounting_scope3.uncertainty_analyses(overall_uncertainty_pct);
CREATE INDEX idx_p042_ua_rating             ON ghg_accounting_scope3.uncertainty_analyses(overall_rating);
CREATE INDEX idx_p042_ua_timestamp          ON ghg_accounting_scope3.uncertainty_analyses(analysis_timestamp DESC);
CREATE INDEX idx_p042_ua_created            ON ghg_accounting_scope3.uncertainty_analyses(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ua_updated
    BEFORE UPDATE ON ghg_accounting_scope3.uncertainty_analyses
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.category_uncertainties
-- =============================================================================
-- Per-category uncertainty breakdown within an analysis. Each category has
-- its own point estimate, confidence interval, probability distribution,
-- and contribution to total inventory uncertainty.

CREATE TABLE ghg_accounting_scope3.category_uncertainties (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.uncertainty_analyses(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Point estimate
    point_estimate              DECIMAL(15,3)   NOT NULL DEFAULT 0,
    pct_of_total                DECIMAL(5,2),
    -- Confidence interval
    lower_bound                 DECIMAL(15,3)   NOT NULL DEFAULT 0,
    upper_bound                 DECIMAL(15,3)   NOT NULL DEFAULT 0,
    uncertainty_pct             DECIMAL(8,4)    NOT NULL DEFAULT 0,
    -- Distribution
    distribution_type           VARCHAR(20)     NOT NULL DEFAULT 'LOGNORMAL',
    std_dev                     DECIMAL(15,3),
    -- Uncertainty sources
    activity_data_unc_pct       DECIMAL(8,4),
    emission_factor_unc_pct     DECIMAL(8,4),
    methodology_unc_pct         DECIMAL(8,4),
    -- Contribution to total
    contribution_to_total_pct   DECIMAL(8,4),
    rank_by_contribution        INTEGER,
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_cu_point_est CHECK (point_estimate >= 0),
    CONSTRAINT chk_p042_cu_lower CHECK (lower_bound >= 0),
    CONSTRAINT chk_p042_cu_upper CHECK (upper_bound >= lower_bound),
    CONSTRAINT chk_p042_cu_unc CHECK (uncertainty_pct >= 0 AND uncertainty_pct <= 500),
    CONSTRAINT chk_p042_cu_pct CHECK (
        pct_of_total IS NULL OR (pct_of_total >= 0 AND pct_of_total <= 100)
    ),
    CONSTRAINT chk_p042_cu_distribution CHECK (
        distribution_type IN ('NORMAL', 'LOGNORMAL', 'UNIFORM', 'TRIANGULAR', 'BETA', 'CUSTOM')
    ),
    CONSTRAINT chk_p042_cu_ad_unc CHECK (
        activity_data_unc_pct IS NULL OR (activity_data_unc_pct >= 0 AND activity_data_unc_pct <= 500)
    ),
    CONSTRAINT chk_p042_cu_ef_unc CHECK (
        emission_factor_unc_pct IS NULL OR (emission_factor_unc_pct >= 0 AND emission_factor_unc_pct <= 500)
    ),
    CONSTRAINT chk_p042_cu_method_unc CHECK (
        methodology_unc_pct IS NULL OR (methodology_unc_pct >= 0 AND methodology_unc_pct <= 500)
    ),
    CONSTRAINT chk_p042_cu_contribution CHECK (
        contribution_to_total_pct IS NULL OR (contribution_to_total_pct >= 0 AND contribution_to_total_pct <= 100)
    ),
    CONSTRAINT chk_p042_cu_rank CHECK (
        rank_by_contribution IS NULL OR (rank_by_contribution >= 1 AND rank_by_contribution <= 15)
    ),
    CONSTRAINT uq_p042_cu_analysis_category UNIQUE (analysis_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_cu_tenant             ON ghg_accounting_scope3.category_uncertainties(tenant_id);
CREATE INDEX idx_p042_cu_analysis           ON ghg_accounting_scope3.category_uncertainties(analysis_id);
CREATE INDEX idx_p042_cu_category           ON ghg_accounting_scope3.category_uncertainties(category);
CREATE INDEX idx_p042_cu_uncertainty        ON ghg_accounting_scope3.category_uncertainties(uncertainty_pct DESC);
CREATE INDEX idx_p042_cu_contribution       ON ghg_accounting_scope3.category_uncertainties(contribution_to_total_pct DESC);
CREATE INDEX idx_p042_cu_rank               ON ghg_accounting_scope3.category_uncertainties(rank_by_contribution);
CREATE INDEX idx_p042_cu_created            ON ghg_accounting_scope3.category_uncertainties(created_at DESC);

-- Composite: analysis + rank for top contributors
CREATE INDEX idx_p042_cu_analysis_rank      ON ghg_accounting_scope3.category_uncertainties(analysis_id, rank_by_contribution)
    WHERE rank_by_contribution IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_cu_updated
    BEFORE UPDATE ON ghg_accounting_scope3.category_uncertainties
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.sensitivity_results
-- =============================================================================
-- Sensitivity analysis results identifying which parameters have the
-- largest influence on total Scope 3 uncertainty. Parameters are ranked
-- by sensitivity index (higher = more influence on total uncertainty).

CREATE TABLE ghg_accounting_scope3.sensitivity_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.uncertainty_analyses(id) ON DELETE CASCADE,
    -- Parameter identification
    parameter_name              VARCHAR(200)    NOT NULL,
    parameter_type              VARCHAR(30)     NOT NULL DEFAULT 'EMISSION_FACTOR',
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Sensitivity metrics
    sensitivity_index           DECIMAL(10,6)   NOT NULL,
    rank                        INTEGER         NOT NULL,
    -- Impact quantification
    base_value                  DECIMAL(15,6),
    perturbation_pct            DECIMAL(5,2)    DEFAULT 10.0,
    result_change_tco2e         DECIMAL(15,3),
    result_change_pct           DECIMAL(8,4),
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_sens_index CHECK (
        sensitivity_index >= 0
    ),
    CONSTRAINT chk_p042_sens_rank CHECK (
        rank >= 1
    ),
    CONSTRAINT chk_p042_sens_param_type CHECK (
        parameter_type IN (
            'EMISSION_FACTOR', 'ACTIVITY_DATA', 'ALLOCATION_FACTOR',
            'GWP_VALUE', 'EXCHANGE_RATE', 'BOUNDARY_CHOICE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p042_sens_perturbation CHECK (
        perturbation_pct IS NULL OR (perturbation_pct > 0 AND perturbation_pct <= 100)
    ),
    CONSTRAINT uq_p042_sens_analysis_param UNIQUE (analysis_id, parameter_name, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_sens_tenant           ON ghg_accounting_scope3.sensitivity_results(tenant_id);
CREATE INDEX idx_p042_sens_analysis         ON ghg_accounting_scope3.sensitivity_results(analysis_id);
CREATE INDEX idx_p042_sens_category         ON ghg_accounting_scope3.sensitivity_results(category);
CREATE INDEX idx_p042_sens_param_type       ON ghg_accounting_scope3.sensitivity_results(parameter_type);
CREATE INDEX idx_p042_sens_index            ON ghg_accounting_scope3.sensitivity_results(sensitivity_index DESC);
CREATE INDEX idx_p042_sens_rank             ON ghg_accounting_scope3.sensitivity_results(rank);
CREATE INDEX idx_p042_sens_created          ON ghg_accounting_scope3.sensitivity_results(created_at DESC);

-- Composite: analysis + top parameters by rank
CREATE INDEX idx_p042_sens_analysis_rank    ON ghg_accounting_scope3.sensitivity_results(analysis_id, rank);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_sens_updated
    BEFORE UPDATE ON ghg_accounting_scope3.sensitivity_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.correlation_matrix
-- =============================================================================
-- Inter-category correlation coefficients. Captures statistical correlations
-- between Scope 3 categories that affect total inventory uncertainty when
-- using error propagation methods. Positive correlations increase total
-- uncertainty; negative correlations reduce it.

CREATE TABLE ghg_accounting_scope3.correlation_matrix (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.uncertainty_analyses(id) ON DELETE CASCADE,
    -- Category pair
    category_a                  ghg_accounting_scope3.scope3_category_type NOT NULL,
    category_b                  ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Correlation
    correlation_coefficient     DECIMAL(5,4)    NOT NULL DEFAULT 0,
    correlation_type            VARCHAR(20)     DEFAULT 'ESTIMATED',
    -- Evidence
    sample_size                 INTEGER,
    p_value                     DECIMAL(6,4),
    confidence_level            DECIMAL(3,2),
    -- Metadata
    justification               TEXT,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_cm_coeff CHECK (
        correlation_coefficient >= -1 AND correlation_coefficient <= 1
    ),
    CONSTRAINT chk_p042_cm_diff_categories CHECK (
        category_a != category_b
    ),
    CONSTRAINT chk_p042_cm_type CHECK (
        correlation_type IS NULL OR correlation_type IN (
            'MEASURED', 'ESTIMATED', 'ASSUMED', 'EXPERT_JUDGMENT', 'DEFAULT'
        )
    ),
    CONSTRAINT chk_p042_cm_sample CHECK (
        sample_size IS NULL OR sample_size >= 0
    ),
    CONSTRAINT chk_p042_cm_p_value CHECK (
        p_value IS NULL OR (p_value >= 0 AND p_value <= 1)
    ),
    CONSTRAINT chk_p042_cm_confidence CHECK (
        confidence_level IS NULL OR (confidence_level >= 0 AND confidence_level <= 1)
    ),
    CONSTRAINT uq_p042_cm_analysis_pair UNIQUE (analysis_id, category_a, category_b)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_cm_tenant             ON ghg_accounting_scope3.correlation_matrix(tenant_id);
CREATE INDEX idx_p042_cm_analysis           ON ghg_accounting_scope3.correlation_matrix(analysis_id);
CREATE INDEX idx_p042_cm_cat_a              ON ghg_accounting_scope3.correlation_matrix(category_a);
CREATE INDEX idx_p042_cm_cat_b              ON ghg_accounting_scope3.correlation_matrix(category_b);
CREATE INDEX idx_p042_cm_coeff              ON ghg_accounting_scope3.correlation_matrix(correlation_coefficient);
CREATE INDEX idx_p042_cm_created            ON ghg_accounting_scope3.correlation_matrix(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_cm_updated
    BEFORE UPDATE ON ghg_accounting_scope3.correlation_matrix
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3.tier_upgrade_impacts
-- =============================================================================
-- Projects the uncertainty reduction achievable by upgrading methodology
-- tiers for specific categories. Helps organizations prioritize where to
-- invest in better data collection (e.g., moving from spend-based to
-- supplier-specific) based on the expected uncertainty improvement.

CREATE TABLE ghg_accounting_scope3.tier_upgrade_impacts (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.uncertainty_analyses(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Current state
    current_tier                ghg_accounting_scope3.methodology_tier_type NOT NULL,
    current_uncertainty_pct     DECIMAL(8,4)    NOT NULL,
    -- Target state
    target_tier                 ghg_accounting_scope3.methodology_tier_type NOT NULL,
    projected_uncertainty_pct   DECIMAL(8,4)    NOT NULL,
    -- Impact
    uncertainty_reduction_pct   DECIMAL(8,4)    GENERATED ALWAYS AS (
        current_uncertainty_pct - projected_uncertainty_pct
    ) STORED,
    impact_on_total_unc_pct     DECIMAL(8,4),
    -- Implementation
    estimated_effort            VARCHAR(20),
    estimated_cost              NUMERIC(18,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    estimated_timeline_months   INTEGER,
    -- Prioritization
    priority_rank               INTEGER,
    cost_effectiveness_score    DECIMAL(8,4),
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_tui_current_unc CHECK (
        current_uncertainty_pct >= 0 AND current_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p042_tui_projected_unc CHECK (
        projected_uncertainty_pct >= 0 AND projected_uncertainty_pct <= 500
    ),
    CONSTRAINT chk_p042_tui_projected_le CHECK (
        projected_uncertainty_pct <= current_uncertainty_pct
    ),
    CONSTRAINT chk_p042_tui_impact CHECK (
        impact_on_total_unc_pct IS NULL OR (impact_on_total_unc_pct >= 0 AND impact_on_total_unc_pct <= 100)
    ),
    CONSTRAINT chk_p042_tui_effort CHECK (
        estimated_effort IS NULL OR estimated_effort IN (
            'VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH'
        )
    ),
    CONSTRAINT chk_p042_tui_cost CHECK (
        estimated_cost IS NULL OR estimated_cost >= 0
    ),
    CONSTRAINT chk_p042_tui_timeline CHECK (
        estimated_timeline_months IS NULL OR estimated_timeline_months >= 0
    ),
    CONSTRAINT chk_p042_tui_rank CHECK (
        priority_rank IS NULL OR priority_rank >= 1
    ),
    CONSTRAINT uq_p042_tui_analysis_cat_tier UNIQUE (analysis_id, category, target_tier)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_tui_tenant            ON ghg_accounting_scope3.tier_upgrade_impacts(tenant_id);
CREATE INDEX idx_p042_tui_analysis          ON ghg_accounting_scope3.tier_upgrade_impacts(analysis_id);
CREATE INDEX idx_p042_tui_category          ON ghg_accounting_scope3.tier_upgrade_impacts(category);
CREATE INDEX idx_p042_tui_current           ON ghg_accounting_scope3.tier_upgrade_impacts(current_tier);
CREATE INDEX idx_p042_tui_target            ON ghg_accounting_scope3.tier_upgrade_impacts(target_tier);
CREATE INDEX idx_p042_tui_impact            ON ghg_accounting_scope3.tier_upgrade_impacts(impact_on_total_unc_pct DESC);
CREATE INDEX idx_p042_tui_rank              ON ghg_accounting_scope3.tier_upgrade_impacts(priority_rank);
CREATE INDEX idx_p042_tui_created           ON ghg_accounting_scope3.tier_upgrade_impacts(created_at DESC);

-- Composite: analysis + rank for prioritized upgrades
CREATE INDEX idx_p042_tui_analysis_rank     ON ghg_accounting_scope3.tier_upgrade_impacts(analysis_id, priority_rank);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_tui_updated
    BEFORE UPDATE ON ghg_accounting_scope3.tier_upgrade_impacts
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.uncertainty_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.category_uncertainties ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.sensitivity_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.correlation_matrix ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.tier_upgrade_impacts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_ua_tenant_isolation ON ghg_accounting_scope3.uncertainty_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_ua_service_bypass ON ghg_accounting_scope3.uncertainty_analyses
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_cu_tenant_isolation ON ghg_accounting_scope3.category_uncertainties
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_cu_service_bypass ON ghg_accounting_scope3.category_uncertainties
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_sens_tenant_isolation ON ghg_accounting_scope3.sensitivity_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_sens_service_bypass ON ghg_accounting_scope3.sensitivity_results
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_cm_tenant_isolation ON ghg_accounting_scope3.correlation_matrix
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_cm_service_bypass ON ghg_accounting_scope3.correlation_matrix
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_tui_tenant_isolation ON ghg_accounting_scope3.tier_upgrade_impacts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_tui_service_bypass ON ghg_accounting_scope3.tier_upgrade_impacts
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.uncertainty_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.category_uncertainties TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.sensitivity_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.correlation_matrix TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.tier_upgrade_impacts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.uncertainty_analyses IS
    'Top-level Scope 3 uncertainty analysis supporting IPCC Approach 1 (analytical) and Approach 2 (Monte Carlo) with confidence intervals.';
COMMENT ON TABLE ghg_accounting_scope3.category_uncertainties IS
    'Per-category uncertainty breakdown with point estimate, confidence interval, distribution type, and contribution to total inventory uncertainty.';
COMMENT ON TABLE ghg_accounting_scope3.sensitivity_results IS
    'Sensitivity analysis ranking parameters by influence on total Scope 3 uncertainty (higher sensitivity index = more influence).';
COMMENT ON TABLE ghg_accounting_scope3.correlation_matrix IS
    'Inter-category correlation coefficients affecting total inventory uncertainty in error propagation calculations.';
COMMENT ON TABLE ghg_accounting_scope3.tier_upgrade_impacts IS
    'Projected uncertainty reduction from methodology tier upgrades to prioritize data collection investment.';

COMMENT ON COLUMN ghg_accounting_scope3.uncertainty_analyses.method IS 'Analysis method: ANALYTICAL (error propagation), MONTE_CARLO (simulation), BOTH, or EXPERT_JUDGMENT.';
COMMENT ON COLUMN ghg_accounting_scope3.uncertainty_analyses.overall_uncertainty_pct IS 'Total inventory uncertainty as percentage of point estimate at specified confidence level.';
COMMENT ON COLUMN ghg_accounting_scope3.category_uncertainties.distribution_type IS 'Probability distribution: NORMAL, LOGNORMAL (most common for emissions), UNIFORM, TRIANGULAR, BETA.';
COMMENT ON COLUMN ghg_accounting_scope3.category_uncertainties.contribution_to_total_pct IS 'This category contribution to total inventory uncertainty (sum across all = 100%).';
COMMENT ON COLUMN ghg_accounting_scope3.sensitivity_results.sensitivity_index IS 'Normalized sensitivity index: change in output / change in input parameter.';
COMMENT ON COLUMN ghg_accounting_scope3.correlation_matrix.correlation_coefficient IS 'Pearson correlation coefficient (-1 to +1) between two category emission estimates.';
COMMENT ON COLUMN ghg_accounting_scope3.tier_upgrade_impacts.uncertainty_reduction_pct IS 'Generated column: current_uncertainty_pct - projected_uncertainty_pct.';
