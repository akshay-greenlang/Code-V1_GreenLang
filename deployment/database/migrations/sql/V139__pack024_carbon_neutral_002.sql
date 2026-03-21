-- =============================================================================
-- V139: PACK-024-carbon-neutral-002: Carbon Management Plans
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for carbon management plans including strategy definition,
-- reduction pathways, action tracking, responsibility assignment, progress
-- monitoring, and governance oversight for achieving carbon neutral targets.
--
-- EXTENDS:
--   V138: Carbon Footprint Quantification Records
--
-- These tables provide the strategic planning and execution framework for
-- carbon neutral goals with detailed action tracking and performance monitoring.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_management_plans           - Strategy & targets
--   2. pack024_carbon_neutral.pack024_reduction_pathways         - Reduction strategy
--   3. pack024_carbon_neutral.pack024_management_actions         - Action tracking
--   4. pack024_carbon_neutral.pack024_action_assignments         - Responsibility & tracking
--
-- Also includes: 45+ indexes, update triggers, security grants, and comments.
-- Previous: V138__pack024_carbon_neutral_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_management_plans
-- =============================================================================
-- High-level carbon management plans with targets, timeline, and governance structure.

CREATE TABLE pack024_carbon_neutral.pack024_management_plans (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    plan_name               VARCHAR(500)    NOT NULL,
    plan_code               VARCHAR(50),
    plan_version            VARCHAR(20)     DEFAULT '1.0',
    baseline_footprint_id   UUID            REFERENCES pack024_carbon_neutral.pack024_footprint_records(id),
    baseline_year           INTEGER,
    baseline_emissions      DECIMAL(18,4),
    target_year             INTEGER         NOT NULL,
    target_emissions        DECIMAL(18,4),
    target_reduction_pct    DECIMAL(6,2),
    carbon_neutral_by_year  INTEGER,
    plan_type               VARCHAR(50)     NOT NULL,
    plan_scope              VARCHAR(100),
    ambition_level          VARCHAR(30),
    strategy_summary        TEXT,
    governance_structure    VARCHAR(500),
    governance_roles        JSONB           DEFAULT '{}',
    leadership_commitment   BOOLEAN         DEFAULT FALSE,
    board_approval_date     DATE,
    board_approval_notes    TEXT,
    stakeholder_engagement  BOOLEAN         DEFAULT FALSE,
    stakeholders_involved   TEXT[],
    engagement_date         DATE,
    timeline_start_date     DATE,
    timeline_end_date       DATE,
    planning_horizon_years  INTEGER,
    financial_commitment    DECIMAL(18,2),
    budget_allocated        DECIMAL(18,2),
    financial_source        VARCHAR(255),
    resource_requirements   JSONB           DEFAULT '{}',
    external_support        BOOLEAN         DEFAULT FALSE,
    support_providers       TEXT[],
    integration_with_ops    BOOLEAN         DEFAULT FALSE,
    integration_description TEXT,
    consistency_with_strategy BOOLEAN       DEFAULT FALSE,
    alignment_notes         TEXT,
    policy_integration      BOOLEAN         DEFAULT FALSE,
    policy_references       TEXT[],
    third_party_certified   BOOLEAN         DEFAULT FALSE,
    certification_type      VARCHAR(100),
    certification_body      VARCHAR(255),
    certification_date      DATE,
    scientific_basis        BOOLEAN         DEFAULT FALSE,
    scientific_standard     VARCHAR(100),
    offset_strategy         BOOLEAN         DEFAULT FALSE,
    offset_details          TEXT,
    plan_status             VARCHAR(30)     DEFAULT 'draft',
    approval_status         VARCHAR(30)     DEFAULT 'pending',
    approved_by             VARCHAR(255),
    approval_date           DATE,
    approval_comments       TEXT,
    public_commitment       BOOLEAN         DEFAULT FALSE,
    public_disclosure_date  DATE,
    public_disclosure_url   VARCHAR(500),
    review_frequency        VARCHAR(50),
    last_review_date        DATE,
    next_review_date        DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_mgmt_plan_type CHECK (
        plan_type IN ('CORPORATE', 'FACILITY', 'SUPPLY_CHAIN', 'PRODUCT', 'SERVICE')
    ),
    CONSTRAINT chk_pack024_mgmt_target_year CHECK (
        target_year >= 2000 AND target_year <= 2100
    ),
    CONSTRAINT chk_pack024_mgmt_budget_non_neg CHECK (
        financial_commitment IS NULL OR financial_commitment >= 0
    ),
    CONSTRAINT chk_pack024_mgmt_reduction_valid CHECK (
        target_reduction_pct IS NULL OR (target_reduction_pct >= 0 AND target_reduction_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pack024_mgmt_org ON pack024_carbon_neutral.pack024_management_plans(org_id);
CREATE INDEX idx_pack024_mgmt_tenant ON pack024_carbon_neutral.pack024_management_plans(tenant_id);
CREATE INDEX idx_pack024_mgmt_baseline_id ON pack024_carbon_neutral.pack024_management_plans(baseline_footprint_id);
CREATE INDEX idx_pack024_mgmt_type ON pack024_carbon_neutral.pack024_management_plans(plan_type);
CREATE INDEX idx_pack024_mgmt_status ON pack024_carbon_neutral.pack024_management_plans(plan_status);
CREATE INDEX idx_pack024_mgmt_approval ON pack024_carbon_neutral.pack024_management_plans(approval_status);
CREATE INDEX idx_pack024_mgmt_target_year ON pack024_carbon_neutral.pack024_management_plans(target_year);
CREATE INDEX idx_pack024_mgmt_neutral_year ON pack024_carbon_neutral.pack024_management_plans(carbon_neutral_by_year);
CREATE INDEX idx_pack024_mgmt_board_approval ON pack024_carbon_neutral.pack024_management_plans(board_approval_date);
CREATE INDEX idx_pack024_mgmt_public_commitment ON pack024_carbon_neutral.pack024_management_plans(public_commitment);
CREATE INDEX idx_pack024_mgmt_certified ON pack024_carbon_neutral.pack024_management_plans(third_party_certified);
CREATE INDEX idx_pack024_mgmt_timeline_start ON pack024_carbon_neutral.pack024_management_plans(timeline_start_date);
CREATE INDEX idx_pack024_mgmt_timeline_end ON pack024_carbon_neutral.pack024_management_plans(timeline_end_date);
CREATE INDEX idx_pack024_mgmt_next_review ON pack024_carbon_neutral.pack024_management_plans(next_review_date);
CREATE INDEX idx_pack024_mgmt_governance ON pack024_carbon_neutral.pack024_management_plans USING GIN(governance_roles);
CREATE INDEX idx_pack024_mgmt_resources ON pack024_carbon_neutral.pack024_management_plans USING GIN(resource_requirements);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_mgmt_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_management_plans
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_reduction_pathways
-- =============================================================================
-- Reduction pathways specifying how baseline will be reduced to target
-- with mechanism breakdown and validation against GHG Protocol principles.

CREATE TABLE pack024_carbon_neutral.pack024_reduction_pathways (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    management_plan_id      UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_management_plans(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    pathway_name            VARCHAR(500)    NOT NULL,
    pathway_sequence        INTEGER         NOT NULL,
    baseline_year           INTEGER,
    target_year             INTEGER,
    start_emissions         DECIMAL(18,4),
    end_emissions           DECIMAL(18,4),
    reduction_amount        DECIMAL(18,4),
    reduction_percentage    DECIMAL(6,2),
    reduction_mechanism     VARCHAR(100)    NOT NULL,
    mechanism_category      VARCHAR(50),
    mechanism_description   TEXT,
    scope_impact            VARCHAR(50),
    abatement_type          VARCHAR(100),
    savings_potential       DECIMAL(18,4),
    savings_cost_usd        DECIMAL(18,2),
    cost_effectiveness      DECIMAL(18,6),
    payback_period_years    DECIMAL(6,2),
    capital_required        DECIMAL(18,2),
    operational_impact      TEXT,
    implementation_timeline JSONB           DEFAULT '{}',
    timeline_milestones     TEXT[],
    timeline_days           INTEGER,
    dependencies            TEXT[],
    risk_factors            TEXT[],
    mitigation_actions      TEXT[],
    stakeholders            TEXT[],
    stakeholder_buy_in      BOOLEAN         DEFAULT FALSE,
    technical_feasibility   VARCHAR(30),
    feasibility_evidence    TEXT,
    market_readiness        VARCHAR(30),
    technology_maturity     VARCHAR(30),
    scaling_potential       VARCHAR(30),
    scaling_timeline_years  INTEGER,
    monitoring_approach     VARCHAR(255),
    kpi_metrics             JSONB           DEFAULT '{}',
    verification_method     VARCHAR(100),
    third_party_verified    BOOLEAN         DEFAULT FALSE,
    verification_body       VARCHAR(255),
    verification_date       DATE,
    pathway_status          VARCHAR(30)     DEFAULT 'planned',
    implementation_started  BOOLEAN         DEFAULT FALSE,
    start_date              DATE,
    completion_date         DATE,
    actual_reduction        DECIMAL(18,4),
    actual_vs_planned_pct   DECIMAL(6,2),
    assumptions_detail      JSONB           DEFAULT '{}',
    documentation_links     TEXT[],
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_pathway_reduction_mechanism CHECK (
        reduction_mechanism IN ('ENERGY_EFFICIENCY', 'RENEWABLE_ENERGY', 'OPERATIONAL_CHANGE',
                               'PROCESS_IMPROVEMENT', 'MATERIAL_SUBSTITUTION', 'FLEET_UPGRADE',
                               'BEHAVIORAL_CHANGE', 'TECHNOLOGY_SWITCH', 'ELECTRIFICATION', 'OTHER')
    ),
    CONSTRAINT chk_pack024_pathway_reduction_valid CHECK (
        reduction_percentage IS NULL OR (reduction_percentage >= 0 AND reduction_percentage <= 100)
    ),
    CONSTRAINT chk_pack024_pathway_emissions_order CHECK (
        start_emissions IS NULL OR end_emissions IS NULL OR start_emissions >= end_emissions
    )
);

-- Indexes
CREATE INDEX idx_pack024_pathway_mgmt_id ON pack024_carbon_neutral.pack024_reduction_pathways(management_plan_id);
CREATE INDEX idx_pack024_pathway_org ON pack024_carbon_neutral.pack024_reduction_pathways(org_id);
CREATE INDEX idx_pack024_pathway_tenant ON pack024_carbon_neutral.pack024_reduction_pathways(tenant_id);
CREATE INDEX idx_pack024_pathway_mechanism ON pack024_carbon_neutral.pack024_reduction_pathways(reduction_mechanism);
CREATE INDEX idx_pack024_pathway_category ON pack024_carbon_neutral.pack024_reduction_pathways(mechanism_category);
CREATE INDEX idx_pack024_pathway_reduction ON pack024_carbon_neutral.pack024_reduction_pathways(reduction_percentage DESC);
CREATE INDEX idx_pack024_pathway_status ON pack024_carbon_neutral.pack024_reduction_pathways(pathway_status);
CREATE INDEX idx_pack024_pathway_implementation ON pack024_carbon_neutral.pack024_reduction_pathways(implementation_started);
CREATE INDEX idx_pack024_pathway_timeline_target ON pack024_carbon_neutral.pack024_reduction_pathways(target_year);
CREATE INDEX idx_pack024_pathway_feasibility ON pack024_carbon_neutral.pack024_reduction_pathways(technical_feasibility);
CREATE INDEX idx_pack024_pathway_verified ON pack024_carbon_neutral.pack024_reduction_pathways(third_party_verified);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_pathway_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_reduction_pathways
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_management_actions
-- =============================================================================
-- Specific management actions aligned to reduction pathways with
-- implementation tracking, resource allocation, and progress monitoring.

CREATE TABLE pack024_carbon_neutral.pack024_management_actions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    management_plan_id      UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_management_plans(id) ON DELETE CASCADE,
    reduction_pathway_id    UUID            REFERENCES pack024_carbon_neutral.pack024_reduction_pathways(id) ON DELETE SET NULL,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    action_title            VARCHAR(500)    NOT NULL,
    action_code             VARCHAR(50),
    action_category         VARCHAR(100)    NOT NULL,
    action_description      TEXT,
    scope_affected          VARCHAR(50),
    annual_savings_mtco2e   DECIMAL(18,4),
    one_time_cost_usd       DECIMAL(18,2),
    recurring_cost_annually DECIMAL(18,2),
    payback_period_years    DECIMAL(6,2),
    internal_rate_of_return DECIMAL(6,2),
    roi_percentage          DECIMAL(6,2),
    priority_ranking        INTEGER,
    priority_level          VARCHAR(30),
    feasibility_score       DECIMAL(5,2),
    co_benefits             TEXT[],
    co_benefit_description  TEXT,
    risks                   TEXT[],
    implementation_barriers TEXT[],
    required_approvals      TEXT[],
    prerequisite_actions    UUID[],
    start_date              DATE            NOT NULL,
    target_completion_date  DATE            NOT NULL,
    actual_completion_date  DATE,
    timeline_status         VARCHAR(30),
    days_overdue            INTEGER,
    responsible_department  VARCHAR(255)    NOT NULL,
    responsible_person      VARCHAR(255),
    contact_email           VARCHAR(255),
    stakeholders            TEXT[],
    resource_requirements   JSONB           DEFAULT '{}',
    budget_allocated        DECIMAL(18,2),
    budget_spent            DECIMAL(18,2),
    budget_variance         DECIMAL(6,2),
    action_status           VARCHAR(30)     DEFAULT 'planned',
    progress_percentage     DECIMAL(6,2)    DEFAULT 0,
    latest_update_date      DATE,
    latest_update_notes     TEXT,
    assumptions             JSONB           DEFAULT '{}',
    measurements            JSONB           DEFAULT '{}',
    verification_method     VARCHAR(100),
    verified                BOOLEAN         DEFAULT FALSE,
    verification_date       DATE,
    verifier_name           VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_action_priority CHECK (
        priority_level IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_pack024_action_progress CHECK (
        progress_percentage >= 0 AND progress_percentage <= 100
    ),
    CONSTRAINT chk_pack024_action_cost_valid CHECK (
        annual_savings_mtco2e >= 0
    )
);

-- Indexes
CREATE INDEX idx_pack024_action_mgmt_id ON pack024_carbon_neutral.pack024_management_actions(management_plan_id);
CREATE INDEX idx_pack024_action_pathway_id ON pack024_carbon_neutral.pack024_management_actions(reduction_pathway_id);
CREATE INDEX idx_pack024_action_org ON pack024_carbon_neutral.pack024_management_actions(org_id);
CREATE INDEX idx_pack024_action_tenant ON pack024_carbon_neutral.pack024_management_actions(tenant_id);
CREATE INDEX idx_pack024_action_category ON pack024_carbon_neutral.pack024_management_actions(action_category);
CREATE INDEX idx_pack024_action_status ON pack024_carbon_neutral.pack024_management_actions(action_status);
CREATE INDEX idx_pack024_action_priority ON pack024_carbon_neutral.pack024_management_actions(priority_level);
CREATE INDEX idx_pack024_action_progress ON pack024_carbon_neutral.pack024_management_actions(progress_percentage);
CREATE INDEX idx_pack024_action_start_date ON pack024_carbon_neutral.pack024_management_actions(start_date);
CREATE INDEX idx_pack024_action_target_completion ON pack024_carbon_neutral.pack024_management_actions(target_completion_date);
CREATE INDEX idx_pack024_action_department ON pack024_carbon_neutral.pack024_management_actions(responsible_department);
CREATE INDEX idx_pack024_action_annual_savings ON pack024_carbon_neutral.pack024_management_actions(annual_savings_mtco2e DESC);
CREATE INDEX idx_pack024_action_roi ON pack024_carbon_neutral.pack024_management_actions(roi_percentage DESC);
CREATE INDEX idx_pack024_action_verified ON pack024_carbon_neutral.pack024_management_actions(verified);
CREATE INDEX idx_pack024_action_prerequisites ON pack024_carbon_neutral.pack024_management_actions USING GIN(prerequisite_actions);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_action_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_management_actions
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_action_assignments
-- =============================================================================
-- Assignment tracking for management actions with responsibility matrix,
-- milestone tracking, and accountability monitoring.

CREATE TABLE pack024_carbon_neutral.pack024_action_assignments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    management_action_id    UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_management_actions(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    assigned_to_name        VARCHAR(255)    NOT NULL,
    assigned_to_email       VARCHAR(255),
    assigned_to_department  VARCHAR(255),
    role_title              VARCHAR(100),
    accountability_level    VARCHAR(30),
    assignment_date         DATE            NOT NULL,
    start_date              DATE,
    due_date                DATE            NOT NULL,
    completion_date         DATE,
    status                  VARCHAR(30)     DEFAULT 'assigned',
    completion_percentage   DECIMAL(6,2)    DEFAULT 0,
    milestones_defined      BOOLEAN         DEFAULT FALSE,
    milestone_count         INTEGER         DEFAULT 0,
    milestones_completed    INTEGER         DEFAULT 0,
    tasks_assigned          INTEGER         DEFAULT 0,
    tasks_completed         INTEGER         DEFAULT 0,
    subtasks                JSONB           DEFAULT '{}',
    resource_allocation     JSONB           DEFAULT '{}',
    time_commitment_hours   DECIMAL(8,2),
    time_logged_hours       DECIMAL(8,2),
    time_variance_hours     DECIMAL(8,2),
    deliverables            TEXT[],
    deliverable_status      JSONB           DEFAULT '{}',
    dependencies            TEXT[],
    blocker_issues          TEXT[],
    risks_identified        TEXT[],
    escalation_status       VARCHAR(30),
    escalation_reason       TEXT,
    escalated_to            VARCHAR(255),
    escalation_date         DATE,
    communication_log       JSONB           DEFAULT '{}',
    last_status_update      TIMESTAMPTZ,
    last_update_notes       TEXT,
    performance_rating      DECIMAL(5,2),
    feedback_notes          TEXT,
    completion_sign_off     BOOLEAN         DEFAULT FALSE,
    sign_off_date           DATE,
    sign_off_by             VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_assign_completion CHECK (
        completion_percentage >= 0 AND completion_percentage <= 100
    ),
    CONSTRAINT chk_pack024_assign_accountability CHECK (
        accountability_level IN ('OWNER', 'CONTRIBUTOR', 'REVIEWER', 'STAKEHOLDER')
    )
);

-- Indexes
CREATE INDEX idx_pack024_assign_action_id ON pack024_carbon_neutral.pack024_action_assignments(management_action_id);
CREATE INDEX idx_pack024_assign_org ON pack024_carbon_neutral.pack024_action_assignments(org_id);
CREATE INDEX idx_pack024_assign_tenant ON pack024_carbon_neutral.pack024_action_assignments(tenant_id);
CREATE INDEX idx_pack024_assign_to_name ON pack024_carbon_neutral.pack024_action_assignments(assigned_to_name);
CREATE INDEX idx_pack024_assign_to_dept ON pack024_carbon_neutral.pack024_action_assignments(assigned_to_department);
CREATE INDEX idx_pack024_assign_status ON pack024_carbon_neutral.pack024_action_assignments(status);
CREATE INDEX idx_pack024_assign_completion ON pack024_carbon_neutral.pack024_action_assignments(completion_percentage);
CREATE INDEX idx_pack024_assign_due_date ON pack024_carbon_neutral.pack024_action_assignments(due_date);
CREATE INDEX idx_pack024_assign_escalation ON pack024_carbon_neutral.pack024_action_assignments(escalation_status);
CREATE INDEX idx_pack024_assign_sign_off ON pack024_carbon_neutral.pack024_action_assignments(completion_sign_off);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_assign_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_action_assignments
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

COMMENT ON TABLE pack024_carbon_neutral.pack024_management_plans IS
'High-level carbon management plans with targets, timeline, governance structure, and strategic commitments for achieving carbon neutrality.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_reduction_pathways IS
'Reduction pathways specifying how baseline will be reduced to target with mechanism breakdown and validation against GHG Protocol principles.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_management_actions IS
'Specific management actions aligned to reduction pathways with implementation tracking, resource allocation, and progress monitoring.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_action_assignments IS
'Assignment tracking for management actions with responsibility matrix, milestone tracking, and accountability monitoring.';
