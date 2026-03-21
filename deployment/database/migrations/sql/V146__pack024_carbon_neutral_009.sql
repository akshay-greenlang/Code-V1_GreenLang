-- =============================================================================
-- V146: PACK-024-carbon-neutral-009: Annual Cycles
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for annual carbon neutral lifecycle management including
-- annual inventory cycles, reporting deadlines, review schedules, and
-- governance activities to maintain carbon neutral status year-over-year.
--
-- EXTENDS:
--   V145: Verification Packages
--
-- These tables support the recurring annual activities and governance
-- requirements for maintaining carbon neutral certification and commitments.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_annual_cycles               - Annual period definition
--   2. pack024_carbon_neutral.pack024_annual_inventory_process    - Inventory execution
--   3. pack024_carbon_neutral.pack024_annual_review_schedule      - Review planning
--   4. pack024_carbon_neutral.pack024_annual_governance_calendar  - Governance activities
--
-- Also includes: 45+ indexes, update triggers, security grants, and comments.
-- Previous: V145__pack024_carbon_neutral_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_annual_cycles
-- =============================================================================
-- Annual carbon neutral lifecycle periods and key dates.

CREATE TABLE pack024_carbon_neutral.pack024_annual_cycles (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    cycle_year              INTEGER         NOT NULL,
    cycle_start_date        DATE            NOT NULL,
    cycle_end_date          DATE            NOT NULL,
    cycle_status            VARCHAR(30)     DEFAULT 'planned',
    cycle_type              VARCHAR(50)     NOT NULL,
    reporting_year          INTEGER,
    baseline_year_reference INTEGER,
    target_reference        VARCHAR(255),
    inventory_baseline_used UUID,
    baseline_emissions      DECIMAL(18,4),
    actual_emissions        DECIMAL(18,4),
    emissions_reduction_from_baseline DECIMAL(6,2),
    credits_required        DECIMAL(18,2),
    credits_procured        DECIMAL(18,2),
    credits_retired         DECIMAL(18,2),
    carbon_neutral_status   VARCHAR(30),
    status_verified         BOOLEAN         DEFAULT FALSE,
    verification_date       DATE,
    verifier_name           VARCHAR(255),
    data_collection_start   DATE,
    data_collection_end     DATE,
    data_collection_status  VARCHAR(30),
    data_quality_target_pct DECIMAL(6,2),
    data_quality_achieved_pct DECIMAL(6,2),
    inventory_deadline      DATE,
    inventory_submission_date DATE,
    inventory_compilation_status VARCHAR(30),
    emissions_calculation_complete BOOLEAN  DEFAULT FALSE,
    emissions_calculation_date DATE,
    offset_strategy_finalized BOOLEAN       DEFAULT FALSE,
    offset_strategy_date    DATE,
    credit_procurement_status VARCHAR(30),
    credit_procurement_completion_date DATE,
    retirement_execution_status VARCHAR(30),
    retirement_execution_deadline DATE,
    verification_scheduled  DATE,
    verification_status     VARCHAR(30),
    reporting_deadline      DATE,
    reporting_submission_date DATE,
    reporting_status        VARCHAR(30),
    public_disclosure_planned BOOLEAN       DEFAULT FALSE,
    public_disclosure_date  DATE,
    stakeholder_update_scheduled BOOLEAN    DEFAULT FALSE,
    stakeholder_update_date DATE,
    next_cycle_planning_start DATE,
    continuous_improvement_topics TEXT[],
    performance_against_targets JSONB       DEFAULT '{}',
    lessons_learned         TEXT,
    improvement_actions     TEXT[],
    budget_allocation       DECIMAL(18,2),
    budget_spent            DECIMAL(18,2),
    budget_variance         DECIMAL(6,2),
    assigned_program_manager VARCHAR(255),
    steering_committee_lead VARCHAR(255),
    cycle_approval_status   VARCHAR(30)     DEFAULT 'pending',
    cycle_approved_by       VARCHAR(255),
    cycle_approval_date     DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_cycle_year CHECK (
        cycle_year >= 2000 AND cycle_year <= 2100
    ),
    CONSTRAINT chk_pack024_cycle_type CHECK (
        cycle_type IN ('REPORTING_YEAR', 'FISCAL_YEAR', 'CALENDAR_YEAR', 'CUSTOM')
    ),
    CONSTRAINT chk_pack024_cycle_status CHECK (
        cycle_status IN ('PLANNED', 'IN_PROGRESS', 'COMPLETED', 'DEFERRED', 'CANCELLED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_cycle_org ON pack024_carbon_neutral.pack024_annual_cycles(org_id);
CREATE INDEX idx_pack024_cycle_tenant ON pack024_carbon_neutral.pack024_annual_cycles(tenant_id);
CREATE INDEX idx_pack024_cycle_year ON pack024_carbon_neutral.pack024_annual_cycles(cycle_year DESC);
CREATE INDEX idx_pack024_cycle_start_date ON pack024_carbon_neutral.pack024_annual_cycles(cycle_start_date);
CREATE INDEX idx_pack024_cycle_end_date ON pack024_carbon_neutral.pack024_annual_cycles(cycle_end_date);
CREATE INDEX idx_pack024_cycle_status ON pack024_carbon_neutral.pack024_annual_cycles(cycle_status);
CREATE INDEX idx_pack024_cycle_type ON pack024_carbon_neutral.pack024_annual_cycles(cycle_type);
CREATE INDEX idx_pack024_cycle_neutral_status ON pack024_carbon_neutral.pack024_annual_cycles(carbon_neutral_status);
CREATE INDEX idx_pack024_cycle_verified ON pack024_carbon_neutral.pack024_annual_cycles(status_verified);
CREATE INDEX idx_pack024_cycle_reporting_deadline ON pack024_carbon_neutral.pack024_annual_cycles(reporting_deadline);
CREATE INDEX idx_pack024_cycle_inventory_deadline ON pack024_carbon_neutral.pack024_annual_cycles(inventory_deadline);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_cycle_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_annual_cycles
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_annual_inventory_process
-- =============================================================================
-- Annual inventory compilation and execution process tracking.

CREATE TABLE pack024_carbon_neutral.pack024_annual_inventory_process (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    annual_cycle_id         UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_annual_cycles(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    process_start_date      DATE            NOT NULL,
    process_phase           VARCHAR(100),
    phase_sequence          INTEGER,
    phase_start_date        DATE,
    phase_end_date          DATE,
    planned_completion_date DATE,
    actual_completion_date  DATE,
    phase_status            VARCHAR(30)     DEFAULT 'not_started',
    phase_completion_pct    DECIMAL(6,2)    DEFAULT 0,
    data_gathering_status   VARCHAR(30),
    data_sources_identified BOOLEAN         DEFAULT FALSE,
    data_sources_contacted  BOOLEAN         DEFAULT FALSE,
    data_collection_progress DECIMAL(6,2),
    data_validation_status  VARCHAR(30),
    data_gaps_identified    TEXT[],
    gap_remediation_plan    TEXT,
    gap_remediation_deadline DATE,
    emissions_calculation_status VARCHAR(30),
    calculation_methodology_applied VARCHAR(255),
    calculation_tools_used  VARCHAR(255)[],
    calculation_complete    BOOLEAN         DEFAULT FALSE,
    calculation_verification_done BOOLEAN   DEFAULT FALSE,
    calculation_verifier    VARCHAR(255),
    calculation_verification_date DATE,
    emissions_by_scope      JSONB           DEFAULT '{}',
    emissions_by_category   JSONB           DEFAULT '{}',
    emissions_uncertainty_calculated BOOLEAN DEFAULT FALSE,
    uncertainty_assessment  VARCHAR(30),
    restatement_prepared    BOOLEAN         DEFAULT FALSE,
    restatement_reason      TEXT,
    prior_year_comparison   BOOLEAN         DEFAULT FALSE,
    comparison_results      TEXT,
    materiality_assessment_done BOOLEAN     DEFAULT FALSE,
    material_changes_identified TEXT[],
    offset_requirement_calculated BOOLEAN   DEFAULT FALSE,
    offset_requirement_amount DECIMAL(18,2),
    offset_procurement_status VARCHAR(30),
    offset_timeline         VARCHAR(100),
    credit_supplier_engagement VARCHAR(30),
    credit_selection_criteria JSONB         DEFAULT '{}',
    retirement_readiness    VARCHAR(30),
    retirement_scheduled_date DATE,
    quality_review_complete BOOLEAN         DEFAULT FALSE,
    quality_reviewer        VARCHAR(255),
    quality_review_date     DATE,
    quality_review_findings TEXT[],
    quality_approval        BOOLEAN         DEFAULT FALSE,
    quality_approval_by     VARCHAR(255),
    inventory_compilation_complete BOOLEAN  DEFAULT FALSE,
    compilation_completion_date DATE,
    draft_report_prepared   BOOLEAN         DEFAULT FALSE,
    draft_report_date       DATE,
    peer_review_required    BOOLEAN         DEFAULT FALSE,
    peer_review_complete    BOOLEAN         DEFAULT FALSE,
    peer_reviewer           VARCHAR(255),
    peer_review_date        DATE,
    peer_review_comments    TEXT,
    final_inventory_ready   BOOLEAN         DEFAULT FALSE,
    sign_off_required       BOOLEAN         DEFAULT TRUE,
    sign_off_by             VARCHAR(255),
    sign_off_date           DATE,
    lessons_learned         TEXT,
    process_improvement_recommendations TEXT[],
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_inv_proc_status CHECK (
        phase_status IN ('NOT_STARTED', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED', 'DELAYED')
    ),
    CONSTRAINT chk_pack024_inv_proc_completion CHECK (
        phase_completion_pct >= 0 AND phase_completion_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_inv_proc_cycle_id ON pack024_carbon_neutral.pack024_annual_inventory_process(annual_cycle_id);
CREATE INDEX idx_pack024_inv_proc_org ON pack024_carbon_neutral.pack024_annual_inventory_process(org_id);
CREATE INDEX idx_pack024_inv_proc_tenant ON pack024_carbon_neutral.pack024_annual_inventory_process(tenant_id);
CREATE INDEX idx_pack024_inv_proc_start_date ON pack024_carbon_neutral.pack024_annual_inventory_process(process_start_date);
CREATE INDEX idx_pack024_inv_proc_phase ON pack024_carbon_neutral.pack024_annual_inventory_process(process_phase);
CREATE INDEX idx_pack024_inv_proc_status ON pack024_carbon_neutral.pack024_annual_inventory_process(phase_status);
CREATE INDEX idx_pack024_inv_proc_completion ON pack024_carbon_neutral.pack024_annual_inventory_process(phase_completion_pct);
CREATE INDEX idx_pack024_inv_proc_calc_complete ON pack024_carbon_neutral.pack024_annual_inventory_process(calculation_complete);
CREATE INDEX idx_pack024_inv_proc_final_ready ON pack024_carbon_neutral.pack024_annual_inventory_process(final_inventory_ready);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_inv_proc_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_annual_inventory_process
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_annual_review_schedule
-- =============================================================================
-- Annual review and verification activities scheduling.

CREATE TABLE pack024_carbon_neutral.pack024_annual_review_schedule (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    annual_cycle_id         UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_annual_cycles(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    review_type             VARCHAR(100)    NOT NULL,
    review_sequence         INTEGER,
    review_phase            VARCHAR(100),
    scheduled_start_date    DATE,
    scheduled_end_date      DATE,
    planned_duration_days   INTEGER,
    actual_start_date       DATE,
    actual_end_date         DATE,
    review_status           VARCHAR(30)     DEFAULT 'scheduled',
    reviewer_primary        VARCHAR(255),
    reviewer_secondary      VARCHAR(255),
    review_scope            VARCHAR(255),
    review_focus_areas      TEXT[],
    documentation_reviewed  TEXT[],
    scope_coverage_pct      DECIMAL(6,2),
    review_procedures       TEXT[],
    sampling_approach       VARCHAR(100),
    sample_size             INTEGER,
    key_findings            TEXT[],
    findings_categorization JSONB           DEFAULT '{}',
    exceptions_identified   INTEGER         DEFAULT 0,
    exceptions_severity     VARCHAR(30),
    review_conclusion       TEXT,
    review_opinion          VARCHAR(50),
    qualified_opinion       BOOLEAN         DEFAULT FALSE,
    opinion_basis           TEXT,
    management_response_required BOOLEAN    DEFAULT FALSE,
    management_response_received BOOLEAN    DEFAULT FALSE,
    management_response     TEXT,
    response_adequacy       VARCHAR(30),
    corrective_action_plan_required BOOLEAN DEFAULT FALSE,
    corrective_action_plan  TEXT,
    corrective_action_deadline DATE,
    corrective_action_tracking BOOLEAN      DEFAULT FALSE,
    follow_up_date          DATE,
    review_approval         BOOLEAN         DEFAULT FALSE,
    approved_by             VARCHAR(255),
    approval_date           DATE,
    approval_conditions     TEXT[],
    report_issued           BOOLEAN         DEFAULT FALSE,
    report_date             DATE,
    report_distribution     TEXT[],
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_review_type CHECK (
        review_type IN ('INTERNAL_AUDIT', 'EXTERNAL_AUDIT', 'MANAGEMENT_REVIEW', 'PEER_REVIEW',
                       'STAKEHOLDER_REVIEW', 'REGULATORY_INSPECTION', 'CERTIFICATION_AUDIT')
    ),
    CONSTRAINT chk_pack024_review_status CHECK (
        review_status IN ('SCHEDULED', 'IN_PROGRESS', 'COMPLETED', 'DEFERRED', 'CANCELLED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_review_cycle_id ON pack024_carbon_neutral.pack024_annual_review_schedule(annual_cycle_id);
CREATE INDEX idx_pack024_review_org ON pack024_carbon_neutral.pack024_annual_review_schedule(org_id);
CREATE INDEX idx_pack024_review_tenant ON pack024_carbon_neutral.pack024_annual_review_schedule(tenant_id);
CREATE INDEX idx_pack024_review_type ON pack024_carbon_neutral.pack024_annual_review_schedule(review_type);
CREATE INDEX idx_pack024_review_status ON pack024_carbon_neutral.pack024_annual_review_schedule(review_status);
CREATE INDEX idx_pack024_review_scheduled_start ON pack024_carbon_neutral.pack024_annual_review_schedule(scheduled_start_date);
CREATE INDEX idx_pack024_review_scheduled_end ON pack024_carbon_neutral.pack024_annual_review_schedule(scheduled_end_date);
CREATE INDEX idx_pack024_review_phase ON pack024_carbon_neutral.pack024_annual_review_schedule(review_phase);
CREATE INDEX idx_pack024_review_opinion ON pack024_carbon_neutral.pack024_annual_review_schedule(review_opinion);
CREATE INDEX idx_pack024_review_approval ON pack024_carbon_neutral.pack024_annual_review_schedule(review_approval);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_review_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_annual_review_schedule
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_annual_governance_calendar
-- =============================================================================
-- Annual governance activities and decision points calendar.

CREATE TABLE pack024_carbon_neutral.pack024_annual_governance_calendar (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    annual_cycle_id         UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_annual_cycles(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    event_date              DATE            NOT NULL,
    event_type              VARCHAR(100)    NOT NULL,
    event_description       TEXT,
    event_sequence          INTEGER,
    event_priority          VARCHAR(30),
    governance_body         VARCHAR(100),
    meeting_required        BOOLEAN         DEFAULT FALSE,
    meeting_scheduled_date  DATE,
    meeting_scheduled_time  TIME,
    meeting_location        VARCHAR(255),
    meeting_format          VARCHAR(50),
    meeting_attendees       VARCHAR(255)[],
    required_attendees      VARCHAR(255)[],
    quorum_requirement      INTEGER,
    expected_attendee_count INTEGER,
    meeting_agenda_items    TEXT[],
    supporting_documents    TEXT[],
    decision_required       BOOLEAN         DEFAULT FALSE,
    decision_type           VARCHAR(100),
    decision_deadline       DATE,
    decision_maker          VARCHAR(255),
    decision_authority      VARCHAR(100),
    decision_options        TEXT[],
    recommendation          TEXT,
    decision_made           BOOLEAN         DEFAULT FALSE,
    decision_date           DATE,
    decision_outcome        TEXT,
    decision_rationale      TEXT,
    dissenting_opinions     TEXT,
    minutes_prepared        BOOLEAN         DEFAULT FALSE,
    minutes_distribution    TEXT[],
    action_items_generated  TEXT[],
    action_item_owners      VARCHAR(255)[],
    action_deadlines        DATE[],
    action_tracking_required BOOLEAN        DEFAULT FALSE,
    follow_up_meeting_date  DATE,
    communication_required  BOOLEAN         DEFAULT FALSE,
    communication_recipients TEXT[],
    communication_date      DATE,
    status                  VARCHAR(30)     DEFAULT 'scheduled',
    completion_date         DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_gov_event_type CHECK (
        event_type IN ('BOARD_MEETING', 'STEERING_COMMITTEE', 'MANAGEMENT_REVIEW', 'STAKEHOLDER_FORUM',
                      'INVESTOR_UPDATE', 'EMPLOYEE_TOWN_HALL', 'EXTERNAL_COMMUNICATION', 'APPROVAL_DECISION',
                      'CERTIFICATION_ASSESSMENT', 'REPORTING_PUBLICATION', 'OTHER')
    ),
    CONSTRAINT chk_pack024_gov_priority CHECK (
        event_priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_pack024_gov_status CHECK (
        status IN ('SCHEDULED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED', 'DEFERRED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_gov_cycle_id ON pack024_carbon_neutral.pack024_annual_governance_calendar(annual_cycle_id);
CREATE INDEX idx_pack024_gov_org ON pack024_carbon_neutral.pack024_annual_governance_calendar(org_id);
CREATE INDEX idx_pack024_gov_tenant ON pack024_carbon_neutral.pack024_annual_governance_calendar(tenant_id);
CREATE INDEX idx_pack024_gov_event_date ON pack024_carbon_neutral.pack024_annual_governance_calendar(event_date);
CREATE INDEX idx_pack024_gov_event_type ON pack024_carbon_neutral.pack024_annual_governance_calendar(event_type);
CREATE INDEX idx_pack024_gov_priority ON pack024_carbon_neutral.pack024_annual_governance_calendar(event_priority);
CREATE INDEX idx_pack024_gov_status ON pack024_carbon_neutral.pack024_annual_governance_calendar(status);
CREATE INDEX idx_pack024_gov_governance_body ON pack024_carbon_neutral.pack024_annual_governance_calendar(governance_body);
CREATE INDEX idx_pack024_gov_decision_required ON pack024_carbon_neutral.pack024_annual_governance_calendar(decision_required);
CREATE INDEX idx_pack024_gov_decision_made ON pack024_carbon_neutral.pack024_annual_governance_calendar(decision_made);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_gov_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_annual_governance_calendar
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

COMMENT ON TABLE pack024_carbon_neutral.pack024_annual_cycles IS
'Annual carbon neutral lifecycle periods and key dates for reporting year cycle management and status tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_annual_inventory_process IS
'Annual inventory compilation and execution process tracking with phase progression and quality assurance.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_annual_review_schedule IS
'Annual review and verification activities scheduling with reviewer assignment, findings, and approval tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_annual_governance_calendar IS
'Annual governance activities and decision points calendar for board/committee meetings and stakeholder communications.';
