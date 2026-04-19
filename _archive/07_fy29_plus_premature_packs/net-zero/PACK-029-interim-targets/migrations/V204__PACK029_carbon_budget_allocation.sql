-- =============================================================================
-- V204: PACK-029 Interim Targets Pack - Carbon Budget Allocation
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    009 of 015
-- Date:         March 2026
--
-- Carbon budget allocation and consumption tracking per year and scope with
-- overshoot detection, rebalancing flags, and cumulative budget management
-- for remaining carbon budget monitoring.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_carbon_budget_allocation
--
-- Previous: V203__PACK029_initiative_schedule.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_carbon_budget_allocation
-- =============================================================================
-- Carbon budget allocation records with annual budget amounts, consumption
-- tracking, remaining budget calculation, overshoot detection, and
-- rebalancing triggers for carbon budget management.

CREATE TABLE pack029_interim_targets.gl_carbon_budget_allocation (
    budget_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Time dimension
    year                        INTEGER         NOT NULL,
    -- Scope
    scope                       VARCHAR(20)     NOT NULL,
    -- Budget allocation
    budget_allocated_tco2e      DECIMAL(18,4)   NOT NULL,
    budget_consumed_tco2e       DECIMAL(18,4)   DEFAULT 0,
    budget_remaining_tco2e      DECIMAL(18,4),
    budget_utilization_pct      DECIMAL(8,4),
    -- Cumulative budget
    cumulative_budget_tco2e     DECIMAL(18,4),
    cumulative_consumed_tco2e   DECIMAL(18,4),
    cumulative_remaining_tco2e  DECIMAL(18,4),
    -- Total remaining (from year to net-zero)
    total_remaining_budget_tco2e DECIMAL(18,4),
    years_remaining             INTEGER,
    avg_annual_budget_remaining DECIMAL(18,4),
    -- Overshoot management
    overshoot_flag              BOOLEAN         DEFAULT FALSE,
    overshoot_amount_tco2e      DECIMAL(18,4),
    overshoot_pct               DECIMAL(8,4),
    overshoot_allowed           BOOLEAN         DEFAULT FALSE,
    overshoot_limit_pct         DECIMAL(8,4)    DEFAULT 5.00,
    -- Rebalancing
    rebalancing_required        BOOLEAN         DEFAULT FALSE,
    rebalancing_strategy        VARCHAR(30),
    rebalancing_applied_at      TIMESTAMPTZ,
    rebalancing_notes           TEXT,
    -- Borrowing/banking
    budget_borrowed_tco2e       DECIMAL(18,4)   DEFAULT 0,
    budget_banked_tco2e         DECIMAL(18,4)   DEFAULT 0,
    net_adjustment_tco2e        DECIMAL(18,4)   DEFAULT 0,
    -- Allocation method
    allocation_method           VARCHAR(30)     DEFAULT 'PATHWAY_DERIVED',
    allocation_basis            VARCHAR(50),
    -- Scenario
    scenario                    VARCHAR(30)     DEFAULT 'BASE_CASE',
    sbti_pathway                VARCHAR(20),
    temperature_alignment       VARCHAR(10),
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_locked                   BOOLEAN         DEFAULT FALSE,
    approved                    BOOLEAN         DEFAULT FALSE,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_cba_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_cba_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p029_cba_budget_allocated CHECK (
        budget_allocated_tco2e >= 0
    ),
    CONSTRAINT chk_p029_cba_budget_consumed CHECK (
        budget_consumed_tco2e >= 0
    ),
    CONSTRAINT chk_p029_cba_overshoot_limit CHECK (
        overshoot_limit_pct >= 0 AND overshoot_limit_pct <= 100
    ),
    CONSTRAINT chk_p029_cba_consumed_vs_allocated CHECK (
        overshoot_allowed = TRUE OR budget_consumed_tco2e <= (budget_allocated_tco2e * (1 + overshoot_limit_pct / 100))
    ),
    CONSTRAINT chk_p029_cba_allocation_method CHECK (
        allocation_method IN ('PATHWAY_DERIVED', 'EQUAL_DISTRIBUTION', 'FRONT_LOADED',
                              'BACK_LOADED', 'PROPORTIONAL', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_cba_rebalancing_strategy CHECK (
        rebalancing_strategy IS NULL OR rebalancing_strategy IN (
            'ACCELERATE_REDUCTION', 'REDISTRIBUTE_BUDGET', 'OFFSET_PURCHASE',
            'SCOPE_SHIFT', 'TARGET_REVISION', 'INITIATIVE_ACCELERATION'
        )
    ),
    CONSTRAINT chk_p029_cba_scenario CHECK (
        scenario IN ('BASE_CASE', 'OPTIMISTIC', 'PESSIMISTIC', 'ACCELERATED', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_cba_sbti_pathway CHECK (
        sbti_pathway IS NULL OR sbti_pathway IN ('1_5C', 'WB2C', '2C', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_cba_temperature CHECK (
        temperature_alignment IS NULL OR temperature_alignment IN ('1.5C', '1.8C', '2.0C', '2.5C', '3.0C')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_cba_tenant            ON pack029_interim_targets.gl_carbon_budget_allocation(tenant_id);
CREATE INDEX idx_p029_cba_org               ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id);
CREATE INDEX idx_p029_cba_target            ON pack029_interim_targets.gl_carbon_budget_allocation(target_id);
CREATE INDEX idx_p029_cba_org_year          ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id, year);
CREATE INDEX idx_p029_cba_org_year_scope    ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id, year, scope);
CREATE INDEX idx_p029_cba_overshoot         ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id, year) WHERE overshoot_flag = TRUE;
CREATE INDEX idx_p029_cba_rebalancing       ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id) WHERE rebalancing_required = TRUE;
CREATE INDEX idx_p029_cba_scenario          ON pack029_interim_targets.gl_carbon_budget_allocation(scenario, organization_id);
CREATE INDEX idx_p029_cba_utilization       ON pack029_interim_targets.gl_carbon_budget_allocation(budget_utilization_pct DESC NULLS LAST);
CREATE INDEX idx_p029_cba_active            ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_cba_unapproved        ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id) WHERE approved = FALSE;
CREATE INDEX idx_p029_cba_created           ON pack029_interim_targets.gl_carbon_budget_allocation(created_at DESC);
CREATE INDEX idx_p029_cba_metadata          ON pack029_interim_targets.gl_carbon_budget_allocation USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_carbon_budget_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_carbon_budget_allocation
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_carbon_budget_allocation ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_cba_tenant_isolation
    ON pack029_interim_targets.gl_carbon_budget_allocation
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_cba_service_bypass
    ON pack029_interim_targets.gl_carbon_budget_allocation
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_carbon_budget_allocation TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_carbon_budget_allocation IS
    'Carbon budget allocation and consumption tracking per year and scope with overshoot detection, rebalancing flags, borrowing/banking mechanisms, and cumulative budget management for remaining carbon budget monitoring.';

COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.budget_id IS 'Unique carbon budget allocation identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.organization_id IS 'Reference to the organization this budget applies to.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.year IS 'Budget year.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.budget_allocated_tco2e IS 'Carbon budget allocated for this year in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.budget_consumed_tco2e IS 'Carbon budget consumed (actual emissions) in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.budget_remaining_tco2e IS 'Remaining carbon budget for this year in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.overshoot_flag IS 'Whether the consumed budget exceeds the allocated budget.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.rebalancing_required IS 'Whether budget rebalancing across future years is needed.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.allocation_method IS 'Method used to allocate budget: PATHWAY_DERIVED, EQUAL_DISTRIBUTION, etc.';
COMMENT ON COLUMN pack029_interim_targets.gl_carbon_budget_allocation.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
