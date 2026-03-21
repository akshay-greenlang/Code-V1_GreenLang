-- =============================================================================
-- V137: PACK-023-sbti-alignment-009: Submission Readiness Assessment Records
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for SBTi submission readiness assessment and pre-submission
-- checklist tracking. Covers data completeness, criteria compliance, documentation
-- readiness, governance readiness, and timeline estimation to submission-ready
-- status with detailed gap analysis and remediation tracking.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (readiness baseline)
--   V130: 42-Criterion Validation
--   V129: PACK-023 Target Definitions
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the submission readiness assessment layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_submission_readiness_assessments  - Overall readiness scoring
--   2. pack023_submission_checklist_items        - Detailed checklist items
--   3. pack023_submission_documentation          - Required documentation tracking
--   4. pack023_submission_timeline               - Submission timeline and roadmap
--
-- Also includes: 40+ indexes, update triggers, security grants, and comments.
-- Previous: V136__pack023_fi_portfolio_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_submission_readiness_assessments
-- =============================================================================
-- Overall submission readiness assessment with composite scoring across
-- data completeness, criteria compliance, documentation, and governance dimensions.

CREATE TABLE pack023_sbti_alignment.pack023_submission_readiness_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_type         VARCHAR(50),
    assessment_period_months INTEGER,
    overall_readiness_score DECIMAL(5,2),
    overall_readiness_pct   DECIMAL(6,2),
    data_completeness_score DECIMAL(5,2),
    criteria_compliance_score DECIMAL(5,2),
    documentation_score     DECIMAL(5,2),
    governance_score        DECIMAL(5,2),
    total_checklist_items   INTEGER         DEFAULT 0,
    completed_items         INTEGER         DEFAULT 0,
    in_progress_items       INTEGER         DEFAULT 0,
    blocked_items           INTEGER         DEFAULT 0,
    completion_percentage   DECIMAL(6,2),
    critical_gaps_count     INTEGER         DEFAULT 0,
    high_priority_gaps      INTEGER         DEFAULT 0,
    medium_priority_gaps    INTEGER         DEFAULT 0,
    low_priority_gaps       INTEGER         DEFAULT 0,
    can_submit_now          BOOLEAN         DEFAULT FALSE,
    estimated_weeks_to_ready INTEGER,
    critical_path_timeline  VARCHAR(200),
    dependencies            TEXT[],
    resource_requirements   JSONB           DEFAULT '{}',
    bottlenecks             TEXT[],
    blockers                TEXT[],
    mitigation_actions      TEXT[],
    risk_score              DECIMAL(5,2),
    risk_assessment         TEXT,
    confidence_in_timeline  VARCHAR(30),
    target_submission_date  DATE,
    previous_submission_attempt BOOLEAN     DEFAULT FALSE,
    previous_feedback_items TEXT[],
    corrective_actions      TEXT[],
    advisor_recommendations TEXT,
    assessed_by             VARCHAR(255),
    assessment_notes        TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_ready_score CHECK (
        overall_readiness_score >= 0 AND overall_readiness_score <= 100
    ),
    CONSTRAINT chk_pk_ready_pct CHECK (
        completion_percentage >= 0 AND completion_percentage <= 100
    ),
    CONSTRAINT chk_pk_risk_score CHECK (
        risk_score >= 0 AND risk_score <= 100
    )
);

-- Indexes
CREATE INDEX idx_pk_ready_org ON pack023_sbti_alignment.pack023_submission_readiness_assessments(org_id);
CREATE INDEX idx_pk_ready_tenant ON pack023_sbti_alignment.pack023_submission_readiness_assessments(tenant_id);
CREATE INDEX idx_pk_ready_date ON pack023_sbti_alignment.pack023_submission_readiness_assessments(assessment_date DESC);
CREATE INDEX idx_pk_ready_type ON pack023_sbti_alignment.pack023_submission_readiness_assessments(assessment_type);
CREATE INDEX idx_pk_ready_overall ON pack023_sbti_alignment.pack023_submission_readiness_assessments(overall_readiness_score);
CREATE INDEX idx_pk_ready_can_submit ON pack023_sbti_alignment.pack023_submission_readiness_assessments(can_submit_now);
CREATE INDEX idx_pk_ready_weeks ON pack023_sbti_alignment.pack023_submission_readiness_assessments(estimated_weeks_to_ready);
CREATE INDEX idx_pk_ready_risk ON pack023_sbti_alignment.pack023_submission_readiness_assessments(risk_score);
CREATE INDEX idx_pk_ready_critical_gaps ON pack023_sbti_alignment.pack023_submission_readiness_assessments(critical_gaps_count);
CREATE INDEX idx_pk_ready_completion ON pack023_sbti_alignment.pack023_submission_readiness_assessments(completion_percentage);
CREATE INDEX idx_pk_ready_metadata ON pack023_sbti_alignment.pack023_submission_readiness_assessments USING GIN(metadata);
CREATE INDEX idx_pk_ready_dependencies ON pack023_sbti_alignment.pack023_submission_readiness_assessments USING GIN(dependencies);

-- Updated_at trigger
CREATE TRIGGER trg_pk_ready_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_submission_readiness_assessments
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_submission_checklist_items
-- =============================================================================
-- Detailed pre-submission checklist items with status, completion date,
-- evidence requirements, and remediation tracking.

CREATE TABLE pack023_sbti_alignment.pack023_submission_checklist_items (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    readiness_assessment_id UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_submission_readiness_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    checklist_category      VARCHAR(100)    NOT NULL,
    item_code               VARCHAR(50),
    item_title              VARCHAR(500)    NOT NULL,
    item_description        TEXT,
    requirement_description TEXT,
    critical_for_submission BOOLEAN         DEFAULT FALSE,
    priority_level          VARCHAR(30),
    status                  VARCHAR(30)     NOT NULL DEFAULT 'pending',
    completion_date         DATE,
    evidence_required       TEXT[],
    evidence_provided       TEXT[],
    evidence_complete       BOOLEAN         DEFAULT FALSE,
    responsible_party       VARCHAR(255),
    assigned_date           DATE,
    due_date                DATE,
    completion_target_date  DATE,
    estimated_effort_hours  DECIMAL(8,2),
    actual_effort_hours     DECIMAL(8,2),
    dependencies            TEXT[],
    blocker_reason          TEXT,
    remediation_action      TEXT,
    remediation_status      VARCHAR(30),
    remediation_owner       VARCHAR(255),
    remediation_due_date    DATE,
    notes                   TEXT,
    last_reviewed_date      DATE,
    reviewed_by             VARCHAR(255),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_check_category CHECK (
        checklist_category IN ('GOVERNANCE', 'DATA_QUALITY', 'CRITERIA', 'DOCUMENTATION',
                              'VERIFICATION', 'SUBMISSION_PROCESS', 'COMMUNICATION')
    ),
    CONSTRAINT chk_pk_check_status CHECK (
        status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED', 'DEFERRED')
    ),
    CONSTRAINT chk_pk_check_priority CHECK (
        priority_level IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    )
);

-- Indexes
CREATE INDEX idx_pk_check_ready_id ON pack023_sbti_alignment.pack023_submission_checklist_items(readiness_assessment_id);
CREATE INDEX idx_pk_check_tenant ON pack023_sbti_alignment.pack023_submission_checklist_items(tenant_id);
CREATE INDEX idx_pk_check_org ON pack023_sbti_alignment.pack023_submission_checklist_items(org_id);
CREATE INDEX idx_pk_check_category ON pack023_sbti_alignment.pack023_submission_checklist_items(checklist_category);
CREATE INDEX idx_pk_check_status ON pack023_sbti_alignment.pack023_submission_checklist_items(status);
CREATE INDEX idx_pk_check_priority ON pack023_sbti_alignment.pack023_submission_checklist_items(priority_level);
CREATE INDEX idx_pk_check_critical ON pack023_sbti_alignment.pack023_submission_checklist_items(critical_for_submission);
CREATE INDEX idx_pk_check_due_date ON pack023_sbti_alignment.pack023_submission_checklist_items(due_date);
CREATE INDEX idx_pk_check_responsible ON pack023_sbti_alignment.pack023_submission_checklist_items(responsible_party);
CREATE INDEX idx_pk_check_evidence_complete ON pack023_sbti_alignment.pack023_submission_checklist_items(evidence_complete);
CREATE INDEX idx_pk_check_evidence ON pack023_sbti_alignment.pack023_submission_checklist_items USING GIN(evidence_required);
CREATE INDEX idx_pk_check_deps ON pack023_sbti_alignment.pack023_submission_checklist_items USING GIN(dependencies);

-- Updated_at trigger
CREATE TRIGGER trg_pk_check_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_submission_checklist_items
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_submission_documentation
-- =============================================================================
-- Required documentation tracking for SBTi submission with content verification
-- and SBTi approval requirements.

CREATE TABLE pack023_sbti_alignment.pack023_submission_documentation (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    readiness_assessment_id UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_submission_readiness_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    document_type           VARCHAR(100)    NOT NULL,
    document_name           VARCHAR(500),
    document_code           VARCHAR(50),
    required_for_submission BOOLEAN         DEFAULT TRUE,
    submission_deadline     DATE,
    sbti_specific_template  BOOLEAN         DEFAULT FALSE,
    content_requirements    TEXT[],
    mandatory_sections      TEXT[],
    optional_sections       TEXT[],
    document_status         VARCHAR(30),
    draft_completed_date    DATE,
    review_in_progress      BOOLEAN         DEFAULT FALSE,
    reviewed_by             VARCHAR(255),
    review_completed_date   DATE,
    review_feedback         TEXT,
    revisions_needed        BOOLEAN         DEFAULT FALSE,
    final_version_date      DATE,
    external_assurance      BOOLEAN         DEFAULT FALSE,
    assurer_organization    VARCHAR(255),
    assurance_type          VARCHAR(100),
    assurance_completed     BOOLEAN         DEFAULT FALSE,
    assurance_date          DATE,
    sbti_approval_required  BOOLEAN         DEFAULT FALSE,
    sbti_reviewed           BOOLEAN         DEFAULT FALSE,
    sbti_approval_date      DATE,
    sbti_reference_number   VARCHAR(100),
    file_path               VARCHAR(500),
    file_hash               VARCHAR(255),
    version_number          VARCHAR(50),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_doc_type CHECK (
        document_type IN ('EMISSIONS_INVENTORY', 'TARGET_STATEMENT', 'VALIDATION_EVIDENCE',
                         'METHODOLOGY_DOCUMENT', 'GOVERNANCE_APPROVAL', 'COMMITMENT_LETTER',
                         'TRANSITION_PLAN', 'MONITORING_PLAN', 'OTHER')
    ),
    CONSTRAINT chk_pk_doc_status CHECK (
        document_status IN ('NOT_STARTED', 'DRAFT', 'IN_REVIEW', 'REVISION_NEEDED',
                           'FINAL', 'SUBMITTED', 'APPROVED')
    )
);

-- Indexes
CREATE INDEX idx_pk_doc_ready_id ON pack023_sbti_alignment.pack023_submission_documentation(readiness_assessment_id);
CREATE INDEX idx_pk_doc_tenant ON pack023_sbti_alignment.pack023_submission_documentation(tenant_id);
CREATE INDEX idx_pk_doc_org ON pack023_sbti_alignment.pack023_submission_documentation(org_id);
CREATE INDEX idx_pk_doc_type ON pack023_sbti_alignment.pack023_submission_documentation(document_type);
CREATE INDEX idx_pk_doc_status ON pack023_sbti_alignment.pack023_submission_documentation(document_status);
CREATE INDEX idx_pk_doc_required ON pack023_sbti_alignment.pack023_submission_documentation(required_for_submission);
CREATE INDEX idx_pk_doc_deadline ON pack023_sbti_alignment.pack023_submission_documentation(submission_deadline);
CREATE INDEX idx_pk_doc_sbti_approved ON pack023_sbti_alignment.pack023_submission_documentation(sbti_approval_required);
CREATE INDEX idx_pk_doc_assurance ON pack023_sbti_alignment.pack023_submission_documentation(external_assurance);
CREATE INDEX idx_pk_doc_created_at ON pack023_sbti_alignment.pack023_submission_documentation(created_at DESC);
CREATE INDEX idx_pk_doc_requirements ON pack023_sbti_alignment.pack023_submission_documentation USING GIN(content_requirements);
CREATE INDEX idx_pk_doc_sections ON pack023_sbti_alignment.pack023_submission_documentation USING GIN(mandatory_sections);

-- Updated_at trigger
CREATE TRIGGER trg_pk_doc_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_submission_documentation
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_submission_timeline
-- =============================================================================
-- Submission timeline and roadmap with milestones, dependencies, and
-- critical path analysis for reaching submission-ready status.

CREATE TABLE pack023_sbti_alignment.pack023_submission_timeline (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    readiness_assessment_id UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_submission_readiness_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    timeline_created_date   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    timeline_baseline_date  DATE,
    current_phase           VARCHAR(100),
    phase_sequence          INTEGER,
    phase_start_date        DATE,
    phase_end_date          DATE,
    phase_duration_weeks    INTEGER,
    phase_completion_pct    DECIMAL(6,2),
    phase_on_track          BOOLEAN,
    phase_delays            BOOLEAN         DEFAULT FALSE,
    delay_reason            TEXT,
    delay_mitigation        TEXT,
    critical_path           BOOLEAN         DEFAULT FALSE,
    critical_path_position  INTEGER,
    milestone_count         INTEGER,
    milestones_completed    INTEGER,
    critical_dependency_count INTEGER,
    resource_requirements   JSONB           DEFAULT '{}',
    budget_requirement_usd  DECIMAL(18,2),
    fte_requirement         DECIMAL(6,2),
    external_support_needed VARCHAR(200)[],
    success_criteria        TEXT[],
    success_metrics         JSONB           DEFAULT '{}',
    contingency_plan        TEXT,
    contingency_timeline_days INTEGER,
    sbti_portal_readiness   BOOLEAN         DEFAULT FALSE,
    sbti_portal_submission_target DATE,
    validation_body_assigned BOOLEAN        DEFAULT FALSE,
    validation_body_name    VARCHAR(255),
    estimated_validation_weeks INTEGER,
    expected_decision_date  DATE,
    post_submission_actions TEXT[],
    target_completion_date  DATE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_timeline_pct CHECK (
        phase_completion_pct >= 0 AND phase_completion_pct <= 100
    ),
    CONSTRAINT chk_pk_timeline_phase CHECK (
        phase_sequence >= 1
    )
);

-- Indexes
CREATE INDEX idx_pk_time_ready_id ON pack023_sbti_alignment.pack023_submission_timeline(readiness_assessment_id);
CREATE INDEX idx_pk_time_tenant ON pack023_sbti_alignment.pack023_submission_timeline(tenant_id);
CREATE INDEX idx_pk_time_org ON pack023_sbti_alignment.pack023_submission_timeline(org_id);
CREATE INDEX idx_pk_time_phase ON pack023_sbti_alignment.pack023_submission_timeline(current_phase);
CREATE INDEX idx_pk_time_start_date ON pack023_sbti_alignment.pack023_submission_timeline(phase_start_date);
CREATE INDEX idx_pk_time_end_date ON pack023_sbti_alignment.pack023_submission_timeline(phase_end_date);
CREATE INDEX idx_pk_time_on_track ON pack023_sbti_alignment.pack023_submission_timeline(phase_on_track);
CREATE INDEX idx_pk_time_critical ON pack023_sbti_alignment.pack023_submission_timeline(critical_path);
CREATE INDEX idx_pk_time_sbti_portal ON pack023_sbti_alignment.pack023_submission_timeline(sbti_portal_submission_target);
CREATE INDEX idx_pk_time_target_complete ON pack023_sbti_alignment.pack023_submission_timeline(target_completion_date);
CREATE INDEX idx_pk_time_created_at ON pack023_sbti_alignment.pack023_submission_timeline(created_at DESC);
CREATE INDEX idx_pk_time_resources ON pack023_sbti_alignment.pack023_submission_timeline USING GIN(resource_requirements);
CREATE INDEX idx_pk_time_metrics ON pack023_sbti_alignment.pack023_submission_timeline USING GIN(success_metrics);
CREATE INDEX idx_pk_time_external ON pack023_sbti_alignment.pack023_submission_timeline USING GIN(external_support_needed);

-- Updated_at trigger
CREATE TRIGGER trg_pk_time_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_submission_timeline
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack023_sbti_alignment TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack023_sbti_alignment TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack023_sbti_alignment TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack023_sbti_alignment.pack023_submission_readiness_assessments IS
'Overall submission readiness assessment with composite scoring across data completeness, criteria compliance, documentation, and governance.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_submission_checklist_items IS
'Detailed pre-submission checklist items with status tracking, evidence requirements, and remediation action tracking.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_submission_documentation IS
'Required documentation tracking for SBTi submission with content verification, review, assurance, and SBTi approval status.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_submission_timeline IS
'Submission timeline and roadmap with phase milestones, critical path analysis, and resource requirements to submission readiness.';
