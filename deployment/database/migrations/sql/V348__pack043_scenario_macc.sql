-- =============================================================================
-- V348: PACK-043 Scope 3 Complete Pack - Scenario Modelling & MACC
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates scenario modelling and Marginal Abatement Cost Curve (MACC) tables
-- for Scope 3 reduction planning. Supports "what-if" scenario analysis with
-- multiple intervention options, cost-per-tonne ranking, and scenario
-- comparison to identify the optimal reduction pathway. Scenarios can be
-- Paris-aligned (1.5C or well-below 2C) and linked to SBTi targets.
--
-- Tables (4):
--   1. ghg_accounting_scope3_complete.scenarios
--   2. ghg_accounting_scope3_complete.interventions
--   3. ghg_accounting_scope3_complete.macc_results
--   4. ghg_accounting_scope3_complete.scenario_comparisons
--
-- Enums (1):
--   1. ghg_accounting_scope3_complete.scenario_type
--
-- Also includes: indexes, RLS, comments.
-- Previous: V347__pack043_lca_integration.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- ---------------------------------------------------------------------------
-- Enum: scenario_type
-- ---------------------------------------------------------------------------
-- Classification of reduction scenario types for pathway modelling.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'scenario_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.scenario_type AS ENUM (
            'BASELINE',         -- Business-as-usual (no interventions)
            'CONSERVATIVE',     -- Low-cost, high-certainty reductions only
            'MODERATE',         -- Balanced cost/reduction portfolio
            'AGGRESSIVE',       -- Maximum reduction regardless of cost
            'PARIS_1_5C',       -- Aligned with 1.5C pathway
            'PARIS_2C',         -- Aligned with well-below 2C pathway
            'SBTI_ALIGNED',     -- Aligned with SBTi target requirements
            'CUSTOM'            -- User-defined scenario
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.scenarios
-- =============================================================================
-- Scenario header defining a reduction pathway for Scope 3 emissions.
-- Each scenario has a baseline, target, budget, and contains a set of
-- interventions. Multiple scenarios can be created for comparison.

CREATE TABLE ghg_accounting_scope3_complete.scenarios (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Scenario identification
    name                        VARCHAR(500)    NOT NULL,
    description                 TEXT,
    scenario_type               ghg_accounting_scope3_complete.scenario_type NOT NULL DEFAULT 'MODERATE',
    -- Emissions
    baseline_tco2e              DECIMAL(15,3)   NOT NULL,
    target_tco2e                DECIMAL(15,3)   NOT NULL,
    target_reduction_pct        DECIMAL(5,2)    GENERATED ALWAYS AS (
        CASE WHEN baseline_tco2e > 0
            THEN ROUND(((baseline_tco2e - target_tco2e) / baseline_tco2e * 100)::NUMERIC, 2)
            ELSE 0
        END
    ) STORED,
    achieved_tco2e              DECIMAL(15,3),
    -- Paris alignment
    paris_aligned               BOOLEAN         NOT NULL DEFAULT false,
    alignment_pathway           VARCHAR(50),
    alignment_confidence        DECIMAL(3,2),
    -- Budget
    budget_usd                  NUMERIC(14,2),
    estimated_total_cost_usd    NUMERIC(14,2),
    budget_available_usd        NUMERIC(14,2),
    -- Timeline
    base_year                   INTEGER,
    target_year                 INTEGER,
    implementation_start        DATE,
    implementation_end          DATE,
    -- Scope
    categories_in_scope         ghg_accounting_scope3_complete.scope3_category_type[],
    categories_count            INTEGER,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    is_preferred                BOOLEAN         NOT NULL DEFAULT false,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    -- Metadata
    assumptions                 JSONB           DEFAULT '[]',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p043_sc_baseline CHECK (baseline_tco2e >= 0),
    CONSTRAINT chk_p043_sc_target CHECK (target_tco2e >= 0),
    CONSTRAINT chk_p043_sc_achieved CHECK (achieved_tco2e IS NULL OR achieved_tco2e >= 0),
    CONSTRAINT chk_p043_sc_alignment CHECK (
        alignment_pathway IS NULL OR alignment_pathway IN (
            '1.5C_NO_OVERSHOOT', '1.5C_LOW_OVERSHOOT', 'WELL_BELOW_2C', '2C', 'NDC_ALIGNED'
        )
    ),
    CONSTRAINT chk_p043_sc_confidence CHECK (
        alignment_confidence IS NULL OR (alignment_confidence >= 0 AND alignment_confidence <= 1)
    ),
    CONSTRAINT chk_p043_sc_budget CHECK (budget_usd IS NULL OR budget_usd >= 0),
    CONSTRAINT chk_p043_sc_est_cost CHECK (estimated_total_cost_usd IS NULL OR estimated_total_cost_usd >= 0),
    CONSTRAINT chk_p043_sc_base_year CHECK (base_year IS NULL OR (base_year >= 1990 AND base_year <= 2100)),
    CONSTRAINT chk_p043_sc_target_year CHECK (target_year IS NULL OR (target_year >= 1990 AND target_year <= 2100)),
    CONSTRAINT chk_p043_sc_years CHECK (base_year IS NULL OR target_year IS NULL OR base_year <= target_year),
    CONSTRAINT chk_p043_sc_impl_dates CHECK (
        implementation_start IS NULL OR implementation_end IS NULL OR implementation_start <= implementation_end
    ),
    CONSTRAINT chk_p043_sc_status CHECK (
        status IN ('DRAFT', 'IN_REVIEW', 'APPROVED', 'ACTIVE', 'COMPLETED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p043_sc_inventory_name UNIQUE (inventory_id, name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_sc_tenant             ON ghg_accounting_scope3_complete.scenarios(tenant_id);
CREATE INDEX idx_p043_sc_inventory          ON ghg_accounting_scope3_complete.scenarios(inventory_id);
CREATE INDEX idx_p043_sc_type               ON ghg_accounting_scope3_complete.scenarios(scenario_type);
CREATE INDEX idx_p043_sc_baseline           ON ghg_accounting_scope3_complete.scenarios(baseline_tco2e DESC);
CREATE INDEX idx_p043_sc_target             ON ghg_accounting_scope3_complete.scenarios(target_tco2e);
CREATE INDEX idx_p043_sc_paris              ON ghg_accounting_scope3_complete.scenarios(paris_aligned) WHERE paris_aligned = true;
CREATE INDEX idx_p043_sc_preferred          ON ghg_accounting_scope3_complete.scenarios(is_preferred) WHERE is_preferred = true;
CREATE INDEX idx_p043_sc_status             ON ghg_accounting_scope3_complete.scenarios(status);
CREATE INDEX idx_p043_sc_target_year        ON ghg_accounting_scope3_complete.scenarios(target_year);
CREATE INDEX idx_p043_sc_created            ON ghg_accounting_scope3_complete.scenarios(created_at DESC);
CREATE INDEX idx_p043_sc_categories         ON ghg_accounting_scope3_complete.scenarios USING GIN(categories_in_scope);
CREATE INDEX idx_p043_sc_metadata           ON ghg_accounting_scope3_complete.scenarios USING GIN(metadata);

-- Composite: inventory + preferred scenario
CREATE INDEX idx_p043_sc_inv_preferred      ON ghg_accounting_scope3_complete.scenarios(inventory_id)
    WHERE is_preferred = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_sc_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.scenarios
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.interventions
-- =============================================================================
-- Individual reduction interventions within a scenario. Each intervention
-- targets one or more Scope 3 categories with a specific cost and reduction
-- potential. Interventions are ranked by cost-per-tCO2e to build the MACC.

CREATE TABLE ghg_accounting_scope3_complete.interventions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    scenario_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.scenarios(id) ON DELETE CASCADE,
    -- Intervention identification
    name                        VARCHAR(500)    NOT NULL,
    description                 TEXT,
    intervention_type           VARCHAR(50)     NOT NULL DEFAULT 'OPERATIONAL',
    -- Target
    category_target             ghg_accounting_scope3_complete.scope3_category_type,
    supplier_target_id          UUID,
    product_target_id           UUID,
    -- Cost
    cost_usd                    NUMERIC(14,2)   NOT NULL DEFAULT 0,
    annual_cost_usd             NUMERIC(14,2),
    upfront_cost_usd            NUMERIC(14,2),
    annual_savings_usd          NUMERIC(14,2),
    -- Reduction
    reduction_tco2e             DECIMAL(15,3)   NOT NULL DEFAULT 0,
    reduction_pct               DECIMAL(5,2),
    cost_per_tco2e              DECIMAL(12,2)   GENERATED ALWAYS AS (
        CASE WHEN reduction_tco2e > 0
            THEN ROUND((cost_usd / reduction_tco2e)::NUMERIC, 2)
            ELSE NULL
        END
    ) STORED,
    -- Feasibility
    difficulty                  VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    confidence                  DECIMAL(3,2)    DEFAULT 0.70,
    risk_level                  VARCHAR(20)     DEFAULT 'MEDIUM',
    -- Timeline
    timeframe_months            INTEGER         NOT NULL DEFAULT 12,
    start_date                  DATE,
    end_date                    DATE,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PLANNED',
    -- Metadata
    dependencies                JSONB           DEFAULT '[]',
    co_benefits                 JSONB           DEFAULT '[]',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_int_type CHECK (
        intervention_type IN (
            'OPERATIONAL', 'PROCUREMENT', 'PRODUCT_DESIGN', 'SUPPLIER_ENGAGEMENT',
            'MODAL_SHIFT', 'ENERGY_SWITCH', 'CIRCULAR_ECONOMY', 'OFFSETTING',
            'TECHNOLOGY', 'POLICY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p043_int_cost CHECK (cost_usd >= 0),
    CONSTRAINT chk_p043_int_annual_cost CHECK (annual_cost_usd IS NULL OR annual_cost_usd >= 0),
    CONSTRAINT chk_p043_int_upfront CHECK (upfront_cost_usd IS NULL OR upfront_cost_usd >= 0),
    CONSTRAINT chk_p043_int_savings CHECK (annual_savings_usd IS NULL OR annual_savings_usd >= 0),
    CONSTRAINT chk_p043_int_reduction CHECK (reduction_tco2e >= 0),
    CONSTRAINT chk_p043_int_reduction_pct CHECK (
        reduction_pct IS NULL OR (reduction_pct >= 0 AND reduction_pct <= 100)
    ),
    CONSTRAINT chk_p043_int_difficulty CHECK (
        difficulty IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p043_int_confidence CHECK (
        confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
    ),
    CONSTRAINT chk_p043_int_risk CHECK (
        risk_level IS NULL OR risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p043_int_timeframe CHECK (timeframe_months > 0 AND timeframe_months <= 360),
    CONSTRAINT chk_p043_int_dates CHECK (
        start_date IS NULL OR end_date IS NULL OR start_date <= end_date
    ),
    CONSTRAINT chk_p043_int_status CHECK (
        status IN ('PLANNED', 'EVALUATING', 'APPROVED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED', 'DEFERRED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_int_tenant            ON ghg_accounting_scope3_complete.interventions(tenant_id);
CREATE INDEX idx_p043_int_scenario          ON ghg_accounting_scope3_complete.interventions(scenario_id);
CREATE INDEX idx_p043_int_type              ON ghg_accounting_scope3_complete.interventions(intervention_type);
CREATE INDEX idx_p043_int_category          ON ghg_accounting_scope3_complete.interventions(category_target);
CREATE INDEX idx_p043_int_cost              ON ghg_accounting_scope3_complete.interventions(cost_usd);
CREATE INDEX idx_p043_int_reduction         ON ghg_accounting_scope3_complete.interventions(reduction_tco2e DESC);
CREATE INDEX idx_p043_int_cost_per_t        ON ghg_accounting_scope3_complete.interventions(cost_per_tco2e);
CREATE INDEX idx_p043_int_difficulty        ON ghg_accounting_scope3_complete.interventions(difficulty);
CREATE INDEX idx_p043_int_status            ON ghg_accounting_scope3_complete.interventions(status);
CREATE INDEX idx_p043_int_timeframe         ON ghg_accounting_scope3_complete.interventions(timeframe_months);
CREATE INDEX idx_p043_int_created           ON ghg_accounting_scope3_complete.interventions(created_at DESC);
CREATE INDEX idx_p043_int_metadata          ON ghg_accounting_scope3_complete.interventions USING GIN(metadata);
CREATE INDEX idx_p043_int_co_benefits       ON ghg_accounting_scope3_complete.interventions USING GIN(co_benefits);

-- Composite: scenario + cost-per-tonne for MACC ordering
CREATE INDEX idx_p043_int_sc_macc           ON ghg_accounting_scope3_complete.interventions(scenario_id, cost_per_tco2e)
    WHERE status NOT IN ('CANCELLED', 'DEFERRED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_int_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.interventions
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.macc_results
-- =============================================================================
-- Pre-computed MACC (Marginal Abatement Cost Curve) results. Each row is an
-- intervention ranked by cost-per-tonne with cumulative reduction and cost
-- calculated. These results power the MACC chart visualization.

CREATE TABLE ghg_accounting_scope3_complete.macc_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    scenario_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.scenarios(id) ON DELETE CASCADE,
    intervention_id             UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.interventions(id) ON DELETE CASCADE,
    -- MACC position
    rank                        INTEGER         NOT NULL,
    cost_per_tco2e              DECIMAL(12,2)   NOT NULL,
    -- Intervention values
    reduction_tco2e             DECIMAL(15,3)   NOT NULL,
    cost_usd                    NUMERIC(14,2)   NOT NULL,
    -- Cumulative
    cumulative_reduction        DECIMAL(15,3)   NOT NULL,
    cumulative_cost             NUMERIC(14,2)   NOT NULL,
    cumulative_reduction_pct    DECIMAL(5,2),
    -- Budget check
    within_budget               BOOLEAN         NOT NULL DEFAULT true,
    remaining_budget_usd        NUMERIC(14,2),
    -- Metadata
    calculation_date            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_macc_rank CHECK (rank >= 1),
    CONSTRAINT chk_p043_macc_reduction CHECK (reduction_tco2e >= 0),
    CONSTRAINT chk_p043_macc_cost CHECK (cost_usd >= 0),
    CONSTRAINT chk_p043_macc_cumul_red CHECK (cumulative_reduction >= 0),
    CONSTRAINT chk_p043_macc_cumul_cost CHECK (cumulative_cost >= 0),
    CONSTRAINT chk_p043_macc_cumul_pct CHECK (
        cumulative_reduction_pct IS NULL OR (cumulative_reduction_pct >= 0 AND cumulative_reduction_pct <= 100)
    ),
    CONSTRAINT uq_p043_macc_scenario_rank UNIQUE (scenario_id, rank),
    CONSTRAINT uq_p043_macc_scenario_intervention UNIQUE (scenario_id, intervention_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_macc_tenant           ON ghg_accounting_scope3_complete.macc_results(tenant_id);
CREATE INDEX idx_p043_macc_scenario         ON ghg_accounting_scope3_complete.macc_results(scenario_id);
CREATE INDEX idx_p043_macc_intervention     ON ghg_accounting_scope3_complete.macc_results(intervention_id);
CREATE INDEX idx_p043_macc_rank             ON ghg_accounting_scope3_complete.macc_results(rank);
CREATE INDEX idx_p043_macc_cost_per_t       ON ghg_accounting_scope3_complete.macc_results(cost_per_tco2e);
CREATE INDEX idx_p043_macc_cumul_red        ON ghg_accounting_scope3_complete.macc_results(cumulative_reduction DESC);
CREATE INDEX idx_p043_macc_within_budget    ON ghg_accounting_scope3_complete.macc_results(within_budget) WHERE within_budget = true;
CREATE INDEX idx_p043_macc_created          ON ghg_accounting_scope3_complete.macc_results(created_at DESC);

-- Composite: scenario + ordered MACC curve
CREATE INDEX idx_p043_macc_sc_ordered       ON ghg_accounting_scope3_complete.macc_results(scenario_id, rank);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_macc_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.macc_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.scenario_comparisons
-- =============================================================================
-- Side-by-side comparison of two scenarios. Calculates the delta in reduction
-- and cost between scenarios and provides a recommendation. Used for
-- decision support when selecting the preferred reduction pathway.

CREATE TABLE ghg_accounting_scope3_complete.scenario_comparisons (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Scenarios
    scenario_a_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.scenarios(id) ON DELETE CASCADE,
    scenario_b_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.scenarios(id) ON DELETE CASCADE,
    -- Comparison results
    scenario_a_reduction_tco2e  DECIMAL(15,3)   NOT NULL,
    scenario_b_reduction_tco2e  DECIMAL(15,3)   NOT NULL,
    delta_reduction             DECIMAL(15,3)   NOT NULL,
    -- Cost
    scenario_a_cost_usd         NUMERIC(14,2)   NOT NULL,
    scenario_b_cost_usd         NUMERIC(14,2)   NOT NULL,
    delta_cost                  NUMERIC(14,2)   NOT NULL,
    -- Efficiency
    scenario_a_cost_per_tco2e   DECIMAL(12,2),
    scenario_b_cost_per_tco2e   DECIMAL(12,2),
    -- Paris alignment
    scenario_a_paris_aligned    BOOLEAN         DEFAULT false,
    scenario_b_paris_aligned    BOOLEAN         DEFAULT false,
    -- Recommendation
    recommendation              VARCHAR(30)     NOT NULL DEFAULT 'SCENARIO_A',
    recommendation_rationale    TEXT,
    -- Metadata
    comparison_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    compared_by                 VARCHAR(255),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_comp_different CHECK (scenario_a_id != scenario_b_id),
    CONSTRAINT chk_p043_comp_a_reduction CHECK (scenario_a_reduction_tco2e >= 0),
    CONSTRAINT chk_p043_comp_b_reduction CHECK (scenario_b_reduction_tco2e >= 0),
    CONSTRAINT chk_p043_comp_a_cost CHECK (scenario_a_cost_usd >= 0),
    CONSTRAINT chk_p043_comp_b_cost CHECK (scenario_b_cost_usd >= 0),
    CONSTRAINT chk_p043_comp_recommendation CHECK (
        recommendation IN ('SCENARIO_A', 'SCENARIO_B', 'EITHER', 'NEITHER', 'CUSTOM')
    ),
    CONSTRAINT uq_p043_comp_pair UNIQUE (inventory_id, scenario_a_id, scenario_b_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_comp_tenant           ON ghg_accounting_scope3_complete.scenario_comparisons(tenant_id);
CREATE INDEX idx_p043_comp_inventory        ON ghg_accounting_scope3_complete.scenario_comparisons(inventory_id);
CREATE INDEX idx_p043_comp_scenario_a       ON ghg_accounting_scope3_complete.scenario_comparisons(scenario_a_id);
CREATE INDEX idx_p043_comp_scenario_b       ON ghg_accounting_scope3_complete.scenario_comparisons(scenario_b_id);
CREATE INDEX idx_p043_comp_recommendation   ON ghg_accounting_scope3_complete.scenario_comparisons(recommendation);
CREATE INDEX idx_p043_comp_date             ON ghg_accounting_scope3_complete.scenario_comparisons(comparison_date DESC);
CREATE INDEX idx_p043_comp_created          ON ghg_accounting_scope3_complete.scenario_comparisons(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_comp_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.scenario_comparisons
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.interventions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.macc_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.scenario_comparisons ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_sc_tenant_isolation ON ghg_accounting_scope3_complete.scenarios
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_sc_service_bypass ON ghg_accounting_scope3_complete.scenarios
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_int_tenant_isolation ON ghg_accounting_scope3_complete.interventions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_int_service_bypass ON ghg_accounting_scope3_complete.interventions
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_macc_tenant_isolation ON ghg_accounting_scope3_complete.macc_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_macc_service_bypass ON ghg_accounting_scope3_complete.macc_results
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_comp_tenant_isolation ON ghg_accounting_scope3_complete.scenario_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_comp_service_bypass ON ghg_accounting_scope3_complete.scenario_comparisons
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.scenarios TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.interventions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.macc_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.scenario_comparisons TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.scenarios IS
    'Reduction scenario header with baseline/target emissions, budget, Paris alignment, and timeline for what-if pathway analysis.';
COMMENT ON TABLE ghg_accounting_scope3_complete.interventions IS
    'Individual reduction interventions within a scenario with cost, reduction potential, difficulty, and cost-per-tCO2e for MACC ranking.';
COMMENT ON TABLE ghg_accounting_scope3_complete.macc_results IS
    'Pre-computed MACC curve with ranked interventions, cumulative reduction, and cumulative cost for chart rendering.';
COMMENT ON TABLE ghg_accounting_scope3_complete.scenario_comparisons IS
    'Side-by-side comparison of two scenarios with delta reduction, delta cost, and recommendation for decision support.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.scenarios.target_reduction_pct IS 'Generated column: ((baseline - target) / baseline) * 100.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.scenarios.paris_aligned IS 'Whether this scenario meets a Paris Agreement pathway (1.5C or 2C).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.scenarios.categories_in_scope IS 'Array of Scope 3 categories addressed by this scenario.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.interventions.cost_per_tco2e IS 'Generated column: cost_usd / reduction_tco2e -- the marginal abatement cost.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.interventions.co_benefits IS 'JSONB array of co-benefits (e.g., cost savings, brand value, risk reduction).';

COMMENT ON COLUMN ghg_accounting_scope3_complete.macc_results.rank IS 'MACC rank (1 = cheapest per tonne, ascending).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.macc_results.cumulative_reduction IS 'Running total of reductions through this rank.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.macc_results.within_budget IS 'Whether cumulative cost at this rank is within scenario budget.';
