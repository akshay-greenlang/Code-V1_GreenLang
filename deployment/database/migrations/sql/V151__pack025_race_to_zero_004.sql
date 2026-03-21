-- =============================================================================
-- V151: PACK-025 Race to Zero - Action Plans
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Climate action plans with section-level detail covering governance, baseline,
-- targets, roadmap, scope 3, offsets, finance, reporting, and transition.
-- Action items with individual reduction targets, costs, and timelines.
--
-- Tables (2):
--   1. pack025_race_to_zero.action_plans
--   2. pack025_race_to_zero.action_items
--
-- Previous: V150__pack025_race_to_zero_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.action_plans
-- =============================================================================
-- Climate action plan documents with section-level completeness tracking
-- for governance, baseline, targets, roadmap, scope 3, offsets, finance,
-- reporting, and just transition.

CREATE TABLE pack025_race_to_zero.action_plans (
    plan_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            NOT NULL REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    plan_date               DATE            NOT NULL,
    plan_version            INTEGER         DEFAULT 1,
    -- Section completeness (each section stored as JSONB for flexibility)
    governance_section      JSONB           DEFAULT '{}',
    baseline_section        JSONB           DEFAULT '{}',
    targets_section         JSONB           DEFAULT '{}',
    roadmap_section         JSONB           DEFAULT '{}',
    scope3_section          JSONB           DEFAULT '{}',
    offset_section          JSONB           DEFAULT '{}',
    finance_section         JSONB           DEFAULT '{}',
    reporting_section       JSONB           DEFAULT '{}',
    transition_section      JSONB           DEFAULT '{}',
    -- Aggregate scores
    total_sections          INTEGER         DEFAULT 9,
    sections_complete       INTEGER         DEFAULT 0,
    plan_completeness_score DECIMAL(6,2),
    plan_quality_score      DECIMAL(6,2),
    total_abatement_tco2e   DECIMAL(18,2),
    total_investment_usd    DECIMAL(18,2),
    planning_horizon_years  INTEGER         DEFAULT 10,
    publication_url         TEXT,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'draft',
    warnings                TEXT[]          DEFAULT '{}',
    errors                  TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_ap_status CHECK (
        status IN ('draft', 'published', 'updated', 'archived', 'under_review')
    ),
    CONSTRAINT chk_p025_ap_completeness CHECK (
        plan_completeness_score IS NULL OR (plan_completeness_score >= 0 AND plan_completeness_score <= 100)
    ),
    CONSTRAINT chk_p025_ap_quality CHECK (
        plan_quality_score IS NULL OR (plan_quality_score >= 0 AND plan_quality_score <= 100)
    ),
    CONSTRAINT chk_p025_ap_horizon CHECK (
        planning_horizon_years >= 1 AND planning_horizon_years <= 50
    ),
    CONSTRAINT chk_p025_ap_version CHECK (
        plan_version >= 1
    ),
    CONSTRAINT chk_p025_ap_abatement CHECK (
        total_abatement_tco2e IS NULL OR total_abatement_tco2e >= 0
    ),
    CONSTRAINT chk_p025_ap_investment CHECK (
        total_investment_usd IS NULL OR total_investment_usd >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for action_plans
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_ap_org             ON pack025_race_to_zero.action_plans(org_id);
CREATE INDEX idx_p025_ap_pledge          ON pack025_race_to_zero.action_plans(pledge_id);
CREATE INDEX idx_p025_ap_tenant          ON pack025_race_to_zero.action_plans(tenant_id);
CREATE INDEX idx_p025_ap_date            ON pack025_race_to_zero.action_plans(plan_date);
CREATE INDEX idx_p025_ap_status          ON pack025_race_to_zero.action_plans(status);
CREATE INDEX idx_p025_ap_completeness    ON pack025_race_to_zero.action_plans(plan_completeness_score);
CREATE INDEX idx_p025_ap_created         ON pack025_race_to_zero.action_plans(created_at DESC);
CREATE INDEX idx_p025_ap_governance      ON pack025_race_to_zero.action_plans USING GIN(governance_section);
CREATE INDEX idx_p025_ap_roadmap         ON pack025_race_to_zero.action_plans USING GIN(roadmap_section);
CREATE INDEX idx_p025_ap_metadata        ON pack025_race_to_zero.action_plans USING GIN(metadata);

-- =============================================================================
-- Table 2: pack025_race_to_zero.action_items
-- =============================================================================
-- Individual abatement action items within a climate action plan with
-- reduction targets, costs, timelines, and implementation status.

CREATE TABLE pack025_race_to_zero.action_items (
    item_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                 UUID            NOT NULL REFERENCES pack025_race_to_zero.action_plans(plan_id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    category                VARCHAR(100)    NOT NULL,
    description             TEXT            NOT NULL,
    target_reduction_tco2e  DECIMAL(18,4),
    actual_reduction_tco2e  DECIMAL(18,4),
    cost_usd                DECIMAL(18,2),
    cost_per_tco2e          DECIMAL(18,4),
    timeline                VARCHAR(50)     NOT NULL DEFAULT 'MEDIUM_TERM',
    start_date              DATE,
    end_date                DATE,
    milestones              JSONB           DEFAULT '[]',
    responsible_party       VARCHAR(255),
    priority_rank           INTEGER,
    feasibility             VARCHAR(20)     DEFAULT 'MEDIUM',
    status                  VARCHAR(30)     NOT NULL DEFAULT 'planned',
    completion_pct          DECIMAL(6,2)    DEFAULT 0,
    dependencies            TEXT[],
    risks                   JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_ai_category CHECK (
        category IN ('ENERGY_EFFICIENCY', 'RENEWABLE_ENERGY', 'ELECTRIFICATION',
                      'FUEL_SWITCHING', 'PROCESS_CHANGE', 'SUPPLY_CHAIN',
                      'CARBON_REMOVAL', 'BEHAVIORAL', 'CIRCULAR_ECONOMY', 'OTHER')
    ),
    CONSTRAINT chk_p025_ai_timeline CHECK (
        timeline IN ('SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM', 'IMMEDIATE')
    ),
    CONSTRAINT chk_p025_ai_feasibility CHECK (
        feasibility IN ('HIGH', 'MEDIUM', 'LOW', 'EXPERIMENTAL')
    ),
    CONSTRAINT chk_p025_ai_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'cancelled', 'deferred')
    ),
    CONSTRAINT chk_p025_ai_completion CHECK (
        completion_pct >= 0 AND completion_pct <= 100
    ),
    CONSTRAINT chk_p025_ai_reduction_non_neg CHECK (
        target_reduction_tco2e IS NULL OR target_reduction_tco2e >= 0
    ),
    CONSTRAINT chk_p025_ai_cost_non_neg CHECK (
        cost_usd IS NULL OR cost_usd >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for action_items
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_ai_plan            ON pack025_race_to_zero.action_items(plan_id);
CREATE INDEX idx_p025_ai_org             ON pack025_race_to_zero.action_items(org_id);
CREATE INDEX idx_p025_ai_tenant          ON pack025_race_to_zero.action_items(tenant_id);
CREATE INDEX idx_p025_ai_category        ON pack025_race_to_zero.action_items(category);
CREATE INDEX idx_p025_ai_status          ON pack025_race_to_zero.action_items(status);
CREATE INDEX idx_p025_ai_timeline        ON pack025_race_to_zero.action_items(timeline);
CREATE INDEX idx_p025_ai_priority        ON pack025_race_to_zero.action_items(priority_rank);
CREATE INDEX idx_p025_ai_feasibility     ON pack025_race_to_zero.action_items(feasibility);
CREATE INDEX idx_p025_ai_created         ON pack025_race_to_zero.action_items(created_at DESC);
CREATE INDEX idx_p025_ai_milestones      ON pack025_race_to_zero.action_items USING GIN(milestones);
CREATE INDEX idx_p025_ai_risks           ON pack025_race_to_zero.action_items USING GIN(risks);
CREATE INDEX idx_p025_ai_metadata        ON pack025_race_to_zero.action_items USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_action_plans_updated
    BEFORE UPDATE ON pack025_race_to_zero.action_plans
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_action_items_updated
    BEFORE UPDATE ON pack025_race_to_zero.action_items
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.action_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.action_items ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_ap_tenant_isolation
    ON pack025_race_to_zero.action_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_ap_service_bypass
    ON pack025_race_to_zero.action_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_ai_tenant_isolation
    ON pack025_race_to_zero.action_items
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_ai_service_bypass
    ON pack025_race_to_zero.action_items
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.action_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.action_items TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.action_plans IS
    'Climate action plans with 9-section structure covering governance, baseline, targets, roadmap, scope 3, offsets, finance, reporting, and just transition.';
COMMENT ON TABLE pack025_race_to_zero.action_items IS
    'Individual abatement action items with reduction targets, costs, timelines, and implementation tracking.';

COMMENT ON COLUMN pack025_race_to_zero.action_plans.governance_section IS 'Governance section JSONB with board oversight, accountability structure.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.baseline_section IS 'Baseline section JSONB with emissions inventory methodology and coverage.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.targets_section IS 'Targets section JSONB with interim and long-term target details.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.roadmap_section IS 'Roadmap section JSONB with phased decarbonization milestones.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.scope3_section IS 'Scope 3 section JSONB with supply chain engagement strategy.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.offset_section IS 'Offset section JSONB with carbon credit use policy and limitations.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.finance_section IS 'Finance section JSONB with investment allocation and climate finance plan.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.reporting_section IS 'Reporting section JSONB with disclosure channels and verification schedule.';
COMMENT ON COLUMN pack025_race_to_zero.action_plans.transition_section IS 'Just Transition section JSONB with social impact and workforce strategy.';
COMMENT ON COLUMN pack025_race_to_zero.action_items.category IS 'Action category: ENERGY_EFFICIENCY, RENEWABLE_ENERGY, ELECTRIFICATION, etc.';
COMMENT ON COLUMN pack025_race_to_zero.action_items.target_reduction_tco2e IS 'Expected emission reduction in tonnes CO2 equivalent.';
