-- =============================================================================
-- V141: PACK-024-carbon-neutral-004: Portfolio Optimization Results
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for carbon credit portfolio optimization with algorithmic
-- recommendations, scenario analysis, cost-benefit analysis, and portfolio
-- rebalancing tracking.
--
-- EXTENDS:
--   V140: Carbon Credit Inventory
--
-- These tables support data-driven portfolio optimization and management
-- recommendations for carbon neutral goal achievement.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_portfolio_optimization       - Optimization runs
--   2. pack024_carbon_neutral.pack024_optimization_scenarios        - Scenario analysis
--   3. pack024_carbon_neutral.pack024_optimization_recommendations  - Algorithmic recommendations
--   4. pack024_carbon_neutral.pack024_rebalancing_actions           - Rebalancing tracking
--
-- Also includes: 50+ indexes, update triggers, security grants, and comments.
-- Previous: V140__pack024_carbon_neutral_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_portfolio_optimization
-- =============================================================================
-- Portfolio optimization runs with configuration, results, and performance tracking.

CREATE TABLE pack024_carbon_neutral.pack024_portfolio_optimization (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    optimization_date       DATE            NOT NULL,
    optimization_scenario   VARCHAR(100),
    optimization_objective  VARCHAR(100)    NOT NULL,
    optimization_constraints TEXT[],
    algorithm_type          VARCHAR(100),
    algorithm_version       VARCHAR(20),
    run_duration_seconds    INTEGER,
    optimization_status     VARCHAR(30)     DEFAULT 'completed',
    convergence_achieved    BOOLEAN         DEFAULT FALSE,
    convergence_iterations  INTEGER,
    convergence_threshold   DECIMAL(6,4),
    current_portfolio_cost  DECIMAL(18,2),
    optimized_portfolio_cost DECIMAL(18,2),
    cost_difference_usd     DECIMAL(18,2),
    cost_savings_percentage DECIMAL(6,2),
    current_coverage_pct    DECIMAL(6,2),
    optimized_coverage_pct  DECIMAL(6,2),
    coverage_improvement    DECIMAL(6,2),
    portfolio_diversification_score DECIMAL(5,2),
    current_diversity_score DECIMAL(5,2),
    optimized_diversity_score DECIMAL(5,2),
    risk_metric_current     DECIMAL(6,4),
    risk_metric_optimized   DECIMAL(6,4),
    risk_reduction_pct      DECIMAL(6,2),
    co_benefits_score       DECIMAL(5,2),
    sustainability_score    DECIMAL(5,2),
    impact_enhancement_pct  DECIMAL(6,2),
    credit_type_balance     JSONB           DEFAULT '{}',
    standard_balance        JSONB           DEFAULT '{}',
    geographic_balance      JSONB           DEFAULT '{}',
    vintage_distribution    JSONB           DEFAULT '{}',
    vintage_risk_assessment VARCHAR(255),
    total_credits_required  DECIMAL(18,2),
    current_credits_held    DECIMAL(18,2),
    shortage_deficit        DECIMAL(18,2),
    surplus_amount          DECIMAL(18,2),
    rebalancing_recommendation BOOLEAN,
    recommendations_count   INTEGER         DEFAULT 0,
    scenario_comparison_performed BOOLEAN    DEFAULT FALSE,
    confidence_score        DECIMAL(5,2),
    assumptions_applied     JSONB           DEFAULT '{}',
    data_quality_notes      TEXT,
    optimization_notes      TEXT,
    performed_by            VARCHAR(255),
    validation_performed    BOOLEAN         DEFAULT FALSE,
    validator_name          VARCHAR(255),
    validation_date         DATE,
    validation_approved     BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_opt_objective CHECK (
        optimization_objective IN ('COST_MINIMIZATION', 'IMPACT_MAXIMIZATION', 'RISK_MINIMIZATION',
                                   'BALANCED_OPTIMIZATION', 'SPEED_OPTIMIZATION', 'DIVERSITY_OPTIMIZATION')
    ),
    CONSTRAINT chk_pack024_opt_cost_savings CHECK (
        cost_savings_percentage IS NULL OR (cost_savings_percentage >= -100 AND cost_savings_percentage <= 100)
    ),
    CONSTRAINT chk_pack024_opt_duration CHECK (
        run_duration_seconds >= 0
    )
);

-- Indexes
CREATE INDEX idx_pack024_opt_org ON pack024_carbon_neutral.pack024_portfolio_optimization(org_id);
CREATE INDEX idx_pack024_opt_tenant ON pack024_carbon_neutral.pack024_portfolio_optimization(tenant_id);
CREATE INDEX idx_pack024_opt_date ON pack024_carbon_neutral.pack024_portfolio_optimization(optimization_date DESC);
CREATE INDEX idx_pack024_opt_objective ON pack024_carbon_neutral.pack024_portfolio_optimization(optimization_objective);
CREATE INDEX idx_pack024_opt_status ON pack024_carbon_neutral.pack024_portfolio_optimization(optimization_status);
CREATE INDEX idx_pack024_opt_cost_savings ON pack024_carbon_neutral.pack024_portfolio_optimization(cost_savings_percentage DESC);
CREATE INDEX idx_pack024_opt_convergence ON pack024_carbon_neutral.pack024_portfolio_optimization(convergence_achieved);
CREATE INDEX idx_pack024_opt_risk_metric ON pack024_carbon_neutral.pack024_portfolio_optimization(risk_metric_optimized);
CREATE INDEX idx_pack024_opt_rebalancing ON pack024_carbon_neutral.pack024_portfolio_optimization(rebalancing_recommendation);
CREATE INDEX idx_pack024_opt_validation ON pack024_carbon_neutral.pack024_portfolio_optimization(validation_approved);
CREATE INDEX idx_pack024_opt_algorithm ON pack024_carbon_neutral.pack024_portfolio_optimization(algorithm_type);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_opt_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_portfolio_optimization
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_optimization_scenarios
-- =============================================================================
-- Scenario analysis for portfolio optimization with variant configurations.

CREATE TABLE pack024_carbon_neutral.pack024_optimization_scenarios (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_run_id     UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_portfolio_optimization(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    scenario_name           VARCHAR(500)    NOT NULL,
    scenario_type           VARCHAR(50),
    scenario_code           VARCHAR(50),
    sequence_number         INTEGER,
    description             TEXT,
    baseline_scenario       BOOLEAN         DEFAULT FALSE,
    assumptions_detail      JSONB           DEFAULT '{}',
    cost_budget_constraint  DECIMAL(18,2),
    coverage_requirement    DECIMAL(6,2),
    credit_type_preferences TEXT[],
    standard_preferences    TEXT[],
    geographic_constraints  TEXT[],
    vintage_preferences     JSONB           DEFAULT '{}',
    co_benefits_requirement DECIMAL(5,2),
    sustainability_threshold DECIMAL(5,2),
    risk_tolerance_level    VARCHAR(30),
    maximum_concentration_pct DECIMAL(6,2),
    minimum_diversification_score DECIMAL(5,2),
    liquidity_requirements  VARCHAR(30),
    timing_constraints      JSONB           DEFAULT '{}',
    scenario_results_cost   DECIMAL(18,2),
    scenario_results_coverage DECIMAL(6,2),
    scenario_results_diversity DECIMAL(5,2),
    scenario_results_risk   DECIMAL(6,4),
    scenario_results_impact DECIMAL(5,2),
    scenario_feasibility    DECIMAL(5,2),
    scenario_viability_rating VARCHAR(30),
    recommended_scenario    BOOLEAN         DEFAULT FALSE,
    recommendation_reason   TEXT,
    portfolio_composition   JSONB           DEFAULT '{}',
    actions_required        TEXT[],
    implementation_timeline VARCHAR(100),
    implementation_cost     DECIMAL(18,2),
    implementation_effort_hours DECIMAL(8,2),
    risk_assessment         TEXT,
    sensitivity_analysis    JSONB           DEFAULT '{}',
    sensitivity_variables   TEXT[],
    break_even_analysis     JSONB           DEFAULT '{}',
    comparison_vs_baseline  JSONB           DEFAULT '{}',
    stakeholder_preferences JSONB           DEFAULT '{}',
    approval_status         VARCHAR(30),
    approved_by             VARCHAR(255),
    approval_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_scen_risk_tolerance CHECK (
        risk_tolerance_level IN ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_pack024_scen_viability CHECK (
        scenario_viability_rating IN ('HIGHLY_VIABLE', 'VIABLE', 'QUESTIONABLE', 'NOT_VIABLE')
    )
);

-- Indexes
CREATE INDEX idx_pack024_scen_opt_run_id ON pack024_carbon_neutral.pack024_optimization_scenarios(optimization_run_id);
CREATE INDEX idx_pack024_scen_org ON pack024_carbon_neutral.pack024_optimization_scenarios(org_id);
CREATE INDEX idx_pack024_scen_tenant ON pack024_carbon_neutral.pack024_optimization_scenarios(tenant_id);
CREATE INDEX idx_pack024_scen_type ON pack024_carbon_neutral.pack024_optimization_scenarios(scenario_type);
CREATE INDEX idx_pack024_scen_baseline ON pack024_carbon_neutral.pack024_optimization_scenarios(baseline_scenario);
CREATE INDEX idx_pack024_scen_recommended ON pack024_carbon_neutral.pack024_optimization_scenarios(recommended_scenario);
CREATE INDEX idx_pack024_scen_cost ON pack024_carbon_neutral.pack024_optimization_scenarios(scenario_results_cost);
CREATE INDEX idx_pack024_scen_coverage ON pack024_carbon_neutral.pack024_optimization_scenarios(scenario_results_coverage);
CREATE INDEX idx_pack024_scen_feasibility ON pack024_carbon_neutral.pack024_optimization_scenarios(scenario_feasibility DESC);
CREATE INDEX idx_pack024_scen_viability ON pack024_carbon_neutral.pack024_optimization_scenarios(scenario_viability_rating);
CREATE INDEX idx_pack024_scen_approval ON pack024_carbon_neutral.pack024_optimization_scenarios(approval_status);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_scen_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_optimization_scenarios
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_optimization_recommendations
-- =============================================================================
-- Algorithmic recommendations from portfolio optimization analysis.

CREATE TABLE pack024_carbon_neutral.pack024_optimization_recommendations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_run_id     UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_portfolio_optimization(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    recommendation_type     VARCHAR(100)    NOT NULL,
    recommendation_priority VARCHAR(30)     NOT NULL,
    recommendation_category VARCHAR(100),
    recommended_action      TEXT            NOT NULL,
    detailed_rationale      TEXT,
    expected_impact         JSONB           DEFAULT '{}',
    impact_quantification   DECIMAL(18,4),
    impact_unit             VARCHAR(30),
    cost_implication        DECIMAL(18,2),
    implementation_effort   VARCHAR(50),
    implementation_timeline VARCHAR(100),
    implementation_steps    TEXT[],
    prerequisites           TEXT[],
    risk_factors            TEXT[],
    mitigation_measures     TEXT[],
    success_criteria        TEXT[],
    performance_metrics     JSONB           DEFAULT '{}',
    affected_portfolio_sections TEXT[],
    required_approvals      TEXT[],
    stakeholder_impact      JSONB           DEFAULT '{}',
    market_conditions_assumption TEXT,
    confidence_level        DECIMAL(5,2),
    recommendation_status   VARCHAR(30)     DEFAULT 'pending_review',
    decision_date           DATE,
    decision_maker          VARCHAR(255),
    decision_rationale      TEXT,
    decision_outcome        VARCHAR(50),
    implementation_started  BOOLEAN         DEFAULT FALSE,
    implementation_date     DATE,
    actual_impact           DECIMAL(18,4),
    actual_vs_expected_pct  DECIMAL(6,2),
    completion_percentage   DECIMAL(6,2),
    completion_date         DATE,
    lessons_learned         TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_rec_type CHECK (
        recommendation_type IN ('PURCHASE', 'SELL', 'HOLD', 'REBALANCE', 'OPTIMIZE', 'RETIRE',
                               'DIVERSIFY', 'CONCENTRATE', 'HEDGE', 'OTHER')
    ),
    CONSTRAINT chk_pack024_rec_priority CHECK (
        recommendation_priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_pack024_rec_status CHECK (
        recommendation_status IN ('PENDING_REVIEW', 'APPROVED', 'REJECTED', 'DEFERRED', 'IMPLEMENTED')
    ),
    CONSTRAINT chk_pack024_rec_confidence CHECK (
        confidence_level >= 0 AND confidence_level <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_rec_opt_run_id ON pack024_carbon_neutral.pack024_optimization_recommendations(optimization_run_id);
CREATE INDEX idx_pack024_rec_org ON pack024_carbon_neutral.pack024_optimization_recommendations(org_id);
CREATE INDEX idx_pack024_rec_tenant ON pack024_carbon_neutral.pack024_optimization_recommendations(tenant_id);
CREATE INDEX idx_pack024_rec_type ON pack024_carbon_neutral.pack024_optimization_recommendations(recommendation_type);
CREATE INDEX idx_pack024_rec_priority ON pack024_carbon_neutral.pack024_optimization_recommendations(recommendation_priority);
CREATE INDEX idx_pack024_rec_category ON pack024_carbon_neutral.pack024_optimization_recommendations(recommendation_category);
CREATE INDEX idx_pack024_rec_status ON pack024_carbon_neutral.pack024_optimization_recommendations(recommendation_status);
CREATE INDEX idx_pack024_rec_confidence ON pack024_carbon_neutral.pack024_optimization_recommendations(confidence_level DESC);
CREATE INDEX idx_pack024_rec_impact ON pack024_carbon_neutral.pack024_optimization_recommendations(impact_quantification DESC);
CREATE INDEX idx_pack024_rec_implementation ON pack024_carbon_neutral.pack024_optimization_recommendations(implementation_started);
CREATE INDEX idx_pack024_rec_decision_date ON pack024_carbon_neutral.pack024_optimization_recommendations(decision_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_rec_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_optimization_recommendations
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_rebalancing_actions
-- =============================================================================
-- Rebalancing action tracking for portfolio optimization implementation.

CREATE TABLE pack024_carbon_neutral.pack024_rebalancing_actions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_run_id     UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_portfolio_optimization(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    rebalancing_date        DATE            NOT NULL,
    action_sequence         INTEGER         NOT NULL,
    action_type             VARCHAR(50)     NOT NULL,
    credit_inventory_id     UUID            REFERENCES pack024_carbon_neutral.pack024_credit_inventory(id),
    action_description      TEXT,
    current_holding_units   DECIMAL(18,2),
    target_holding_units    DECIMAL(18,2),
    action_units            DECIMAL(18,2)   NOT NULL,
    action_percentage_of_portfolio DECIMAL(6,2),
    action_reason           VARCHAR(255),
    priority_ranking        INTEGER,
    scheduled_start_date    DATE,
    scheduled_completion_date DATE,
    actual_start_date       DATE,
    actual_completion_date  DATE,
    action_status           VARCHAR(30)     DEFAULT 'planned',
    status_percentage_complete DECIMAL(6,2),
    responsible_party       VARCHAR(255),
    approval_required       BOOLEAN         DEFAULT TRUE,
    approval_status         VARCHAR(30),
    approved_by             VARCHAR(255),
    approval_date           DATE,
    budget_required_usd     DECIMAL(18,2),
    actual_cost_usd         DECIMAL(18,2),
    cost_variance_usd       DECIMAL(18,2),
    cost_variance_pct       DECIMAL(6,2),
    timeline_on_track       BOOLEAN,
    timeline_delay_days     INTEGER,
    timeline_delay_reason   TEXT,
    mitigation_actions      TEXT[],
    counterparty_name       VARCHAR(255),
    transaction_reference   VARCHAR(100),
    settlement_status       VARCHAR(30),
    settlement_date         DATE,
    registry_update_pending BOOLEAN         DEFAULT FALSE,
    registry_update_date    DATE,
    verification_required   BOOLEAN         DEFAULT FALSE,
    verification_completed  BOOLEAN         DEFAULT FALSE,
    verifier_name           VARCHAR(255),
    verification_date       DATE,
    audit_trail             JSONB           DEFAULT '{}',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_rebal_action_type CHECK (
        action_type IN ('BUY', 'SELL', 'RETIRE', 'HOLD', 'SWAP', 'TRANSFER')
    ),
    CONSTRAINT chk_pack024_rebal_action_status CHECK (
        action_status IN ('PLANNED', 'APPROVED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED')
    ),
    CONSTRAINT chk_pack024_rebal_units_valid CHECK (
        action_units > 0
    )
);

-- Indexes
CREATE INDEX idx_pack024_rebal_opt_run_id ON pack024_carbon_neutral.pack024_rebalancing_actions(optimization_run_id);
CREATE INDEX idx_pack024_rebal_org ON pack024_carbon_neutral.pack024_rebalancing_actions(org_id);
CREATE INDEX idx_pack024_rebal_tenant ON pack024_carbon_neutral.pack024_rebalancing_actions(tenant_id);
CREATE INDEX idx_pack024_rebal_inv_id ON pack024_carbon_neutral.pack024_rebalancing_actions(credit_inventory_id);
CREATE INDEX idx_pack024_rebal_date ON pack024_carbon_neutral.pack024_rebalancing_actions(rebalancing_date DESC);
CREATE INDEX idx_pack024_rebal_type ON pack024_carbon_neutral.pack024_rebalancing_actions(action_type);
CREATE INDEX idx_pack024_rebal_status ON pack024_carbon_neutral.pack024_rebalancing_actions(action_status);
CREATE INDEX idx_pack024_rebal_approval ON pack024_carbon_neutral.pack024_rebalancing_actions(approval_status);
CREATE INDEX idx_pack024_rebal_priority ON pack024_carbon_neutral.pack024_rebalancing_actions(priority_ranking);
CREATE INDEX idx_pack024_rebal_timeline_on_track ON pack024_carbon_neutral.pack024_rebalancing_actions(timeline_on_track);
CREATE INDEX idx_pack024_rebal_completion_date ON pack024_carbon_neutral.pack024_rebalancing_actions(actual_completion_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_rebal_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_rebalancing_actions
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack024_carbon_neutral TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack024_carbon_neutral TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack024_carbon_neutral TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack024_carbon_neutral.pack024_portfolio_optimization IS
'Portfolio optimization runs with configuration, algorithmic analysis results, and performance metrics tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_optimization_scenarios IS
'Scenario analysis for portfolio optimization with variant configurations, assumptions, and comparative results.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_optimization_recommendations IS
'Algorithmic recommendations from portfolio optimization analysis with impact quantification and decision tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_rebalancing_actions IS
'Rebalancing action tracking for portfolio optimization implementation with approval, execution, and verification status.';
