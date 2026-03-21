-- =============================================================================
-- V130: PACK-023-sbti-alignment-002: 42-Criterion Validation Results
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for SBTi 42-criterion validation results.
-- Covers full assessment of C1-C28 (near-term criteria) and NZ-C1 to NZ-C14
-- (net-zero criteria) with per-criterion pass/fail/warning status, evidence
-- links, and detailed gap analysis with remediation guidance.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (validation result structures)
--   V129: PACK-023 Target Definitions
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the comprehensive 42-criterion validation layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_sbti_validation_assessments    - Overall validation run records
--   2. pack023_sbti_criterion_results         - Per-criterion check results
--   3. pack023_sbti_validation_gaps           - Identified gaps per criterion
--   4. pack023_sbti_remediation_guidance      - Gap remediation recommendations
--
-- Hypertables (1):
--   pack023_sbti_validation_assessments on assessment_date (chunk: 3 months)
--
-- Also includes: 45+ indexes, update triggers, security grants, and comments.
-- Previous: V129__pack023_sbti_targets_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_sbti_validation_assessments
-- =============================================================================
-- Overall validation assessment run records tracking validation date, input
-- data quality, total pass/warning/fail counts, and overall recommendation.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_validation_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    assessment_name         VARCHAR(500),
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_type         VARCHAR(50),
    data_quality_tier       VARCHAR(30),
    input_completeness_pct  DECIMAL(6,2),
    total_criteria          INTEGER         DEFAULT 42,
    criteria_passed         INTEGER         DEFAULT 0,
    criteria_warning        INTEGER         DEFAULT 0,
    criteria_failed         INTEGER         DEFAULT 0,
    criteria_na             INTEGER         DEFAULT 0,
    pass_rate_percentage    DECIMAL(6,2),
    overall_status          VARCHAR(30),
    recommendation          TEXT,
    estimated_submission_ready_date DATE,
    weeks_to_ready          INTEGER,
    can_submit              BOOLEAN         DEFAULT FALSE,
    assessed_by             VARCHAR(255),
    approval_status         VARCHAR(30)     DEFAULT 'pending_review',
    approved_by             VARCHAR(255),
    approved_at             TIMESTAMPTZ,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_val_type CHECK (
        assessment_type IN ('NEAR_TERM', 'LONG_TERM', 'NET_ZERO', 'COMBINED')
    ),
    CONSTRAINT chk_pk_val_status CHECK (
        overall_status IN ('PASS', 'WARNING', 'FAIL', 'INCOMPLETE')
    ),
    CONSTRAINT chk_pk_criteria_counts CHECK (
        (criteria_passed + criteria_warning + criteria_failed + criteria_na) = total_criteria
    ),
    CONSTRAINT chk_pk_completeness_pct CHECK (
        input_completeness_pct >= 0 AND input_completeness_pct <= 100
    )
);

-- Hypertable
SELECT create_hypertable(
    'pack023_sbti_alignment.pack023_sbti_validation_assessments',
    'assessment_date',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '3 months'
);

-- Indexes
CREATE INDEX idx_pk_val_tenant ON pack023_sbti_alignment.pack023_sbti_validation_assessments(tenant_id);
CREATE INDEX idx_pk_val_org ON pack023_sbti_alignment.pack023_sbti_validation_assessments(org_id);
CREATE INDEX idx_pk_val_type ON pack023_sbti_alignment.pack023_sbti_validation_assessments(assessment_type);
CREATE INDEX idx_pk_val_date ON pack023_sbti_alignment.pack023_sbti_validation_assessments(assessment_date DESC);
CREATE INDEX idx_pk_val_status ON pack023_sbti_alignment.pack023_sbti_validation_assessments(overall_status);
CREATE INDEX idx_pk_val_approval ON pack023_sbti_alignment.pack023_sbti_validation_assessments(approval_status);
CREATE INDEX idx_pk_val_can_submit ON pack023_sbti_alignment.pack023_sbti_validation_assessments(can_submit);
CREATE INDEX idx_pk_val_pass_rate ON pack023_sbti_alignment.pack023_sbti_validation_assessments(pass_rate_percentage);
CREATE INDEX idx_pk_val_org_date ON pack023_sbti_alignment.pack023_sbti_validation_assessments(org_id, assessment_date DESC);
CREATE INDEX idx_pk_val_metadata ON pack023_sbti_alignment.pack023_sbti_validation_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_pk_val_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_validation_assessments
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_sbti_criterion_results
-- =============================================================================
-- Individual criterion assessment results for all 42 criteria (C1-C28 + NZ-C1-NZ-C14)
-- with pass/fail/warning/NA status, evidence references, and scoring.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_criterion_results (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_assessment_id UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_validation_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    criterion_code          VARCHAR(20)     NOT NULL,
    criterion_name          VARCHAR(500),
    criterion_category      VARCHAR(100),
    criterion_type          VARCHAR(50),
    scope_requirement       VARCHAR(50),
    requirement_text        TEXT,
    assessment_result       VARCHAR(30)     NOT NULL,
    assessment_score        DECIMAL(5,2),
    evidence_provided       BOOLEAN         DEFAULT FALSE,
    evidence_links          TEXT[],
    evidence_description    TEXT,
    data_quality_tier       VARCHAR(30),
    specific_gap            TEXT,
    severity                VARCHAR(30),
    remediation_possible    BOOLEAN         DEFAULT TRUE,
    estimated_effort_days   INTEGER,
    assessor_notes          TEXT,
    reviewed_by             VARCHAR(255),
    reviewed_at             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_crit_code CHECK (
        (criterion_type = 'NEAR_TERM' AND criterion_code ~ '^C[0-9]{1,2}$') OR
        (criterion_type = 'NET_ZERO' AND criterion_code ~ '^NZ-C[0-9]{1,2}$')
    ),
    CONSTRAINT chk_pk_crit_result CHECK (
        assessment_result IN ('PASS', 'WARNING', 'FAIL', 'NA')
    ),
    CONSTRAINT chk_pk_crit_score CHECK (
        assessment_score IS NULL OR (assessment_score >= 0 AND assessment_score <= 100)
    ),
    CONSTRAINT chk_pk_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    )
);

-- Indexes
CREATE INDEX idx_pk_crit_val_id ON pack023_sbti_alignment.pack023_sbti_criterion_results(validation_assessment_id);
CREATE INDEX idx_pk_crit_tenant ON pack023_sbti_alignment.pack023_sbti_criterion_results(tenant_id);
CREATE INDEX idx_pk_crit_org ON pack023_sbti_alignment.pack023_sbti_criterion_results(org_id);
CREATE INDEX idx_pk_crit_code ON pack023_sbti_alignment.pack023_sbti_criterion_results(criterion_code);
CREATE INDEX idx_pk_crit_type ON pack023_sbti_alignment.pack023_sbti_criterion_results(criterion_type);
CREATE INDEX idx_pk_crit_result ON pack023_sbti_alignment.pack023_sbti_criterion_results(assessment_result);
CREATE INDEX idx_pk_crit_severity ON pack023_sbti_alignment.pack023_sbti_criterion_results(severity);
CREATE INDEX idx_pk_crit_evidence ON pack023_sbti_alignment.pack023_sbti_criterion_results(evidence_provided);
CREATE INDEX idx_pk_crit_created_at ON pack023_sbti_alignment.pack023_sbti_criterion_results(created_at DESC);
CREATE INDEX idx_pk_crit_evidence_links ON pack023_sbti_alignment.pack023_sbti_criterion_results USING GIN(evidence_links);
CREATE INDEX idx_pk_crit_org_type ON pack023_sbti_alignment.pack023_sbti_criterion_results(org_id, criterion_type);

-- Updated_at trigger
CREATE TRIGGER trg_pk_crit_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_criterion_results
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_sbti_validation_gaps
-- =============================================================================
-- Detailed gap records for each identified shortfall against criterion requirements,
-- with gap description, root cause analysis, and impact assessment.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_validation_gaps (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    criterion_result_id     UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_criterion_results(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    gap_code                VARCHAR(100),
    gap_title               VARCHAR(500),
    gap_description         TEXT,
    root_cause              VARCHAR(500),
    impact_on_target        VARCHAR(50),
    impact_description      TEXT,
    current_state           TEXT,
    required_state          TEXT,
    gap_severity            VARCHAR(30),
    data_gap_type           VARCHAR(100),
    missing_information     TEXT[],
    timeline_constraint     VARCHAR(200),
    effort_estimate_days    INTEGER,
    resource_requirement    VARCHAR(500),
    dependencies            TEXT[],
    status                  VARCHAR(30)     DEFAULT 'open',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_gap_impact CHECK (
        impact_on_target IN ('BLOCKING', 'SIGNIFICANT', 'MODERATE', 'MINOR')
    ),
    CONSTRAINT chk_pk_gap_severity CHECK (
        gap_severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_pk_gap_status CHECK (
        status IN ('open', 'in_progress', 'resolved', 'deferred', 'waived')
    )
);

-- Indexes
CREATE INDEX idx_pk_gap_crit_id ON pack023_sbti_alignment.pack023_sbti_validation_gaps(criterion_result_id);
CREATE INDEX idx_pk_gap_tenant ON pack023_sbti_alignment.pack023_sbti_validation_gaps(tenant_id);
CREATE INDEX idx_pk_gap_org ON pack023_sbti_alignment.pack023_sbti_validation_gaps(org_id);
CREATE INDEX idx_pk_gap_severity ON pack023_sbti_alignment.pack023_sbti_validation_gaps(gap_severity);
CREATE INDEX idx_pk_gap_status ON pack023_sbti_alignment.pack023_sbti_validation_gaps(status);
CREATE INDEX idx_pk_gap_impact ON pack023_sbti_alignment.pack023_sbti_validation_gaps(impact_on_target);
CREATE INDEX idx_pk_gap_created_at ON pack023_sbti_alignment.pack023_sbti_validation_gaps(created_at DESC);
CREATE INDEX idx_pk_gap_missing_info ON pack023_sbti_alignment.pack023_sbti_validation_gaps USING GIN(missing_information);
CREATE INDEX idx_pk_gap_deps ON pack023_sbti_alignment.pack023_sbti_validation_gaps USING GIN(dependencies);

-- Updated_at trigger
CREATE TRIGGER trg_pk_gap_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_validation_gaps
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_sbti_remediation_guidance
-- =============================================================================
-- Detailed remediation guidance for each gap with step-by-step actions,
-- responsible parties, timelines, and success criteria.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_remediation_guidance (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_gap_id       UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_validation_gaps(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    gap_id                  UUID,
    remediation_action_seq  INTEGER         NOT NULL,
    action_title            VARCHAR(500),
    action_description      TEXT,
    action_type             VARCHAR(100),
    detailed_steps          TEXT[],
    responsible_party       VARCHAR(255),
    required_inputs         VARCHAR(500)[],
    resource_requirements   JSONB           DEFAULT '{}',
    estimated_effort_hours  DECIMAL(8,2),
    start_date_target       DATE,
    completion_date_target  DATE,
    dependencies_on_actions TEXT[],
    evidence_to_collect     TEXT[],
    success_criteria        TEXT[],
    risk_if_not_addressed   VARCHAR(500),
    alternative_approaches  TEXT[],
    status                  VARCHAR(30)     DEFAULT 'pending',
    actual_start_date       DATE,
    actual_completion_date  DATE,
    completion_notes        TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_rem_status CHECK (
        status IN ('pending', 'in_progress', 'completed', 'deferred', 'closed')
    ),
    CONSTRAINT chk_pk_rem_seq CHECK (
        remediation_action_seq >= 1
    )
);

-- Indexes
CREATE INDEX idx_pk_rem_gap_id ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(validation_gap_id);
CREATE INDEX idx_pk_rem_tenant ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(tenant_id);
CREATE INDEX idx_pk_rem_org ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(org_id);
CREATE INDEX idx_pk_rem_status ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(status);
CREATE INDEX idx_pk_rem_type ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(action_type);
CREATE INDEX idx_pk_rem_seq ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(remediation_action_seq);
CREATE INDEX idx_pk_rem_dates ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(completion_date_target);
CREATE INDEX idx_pk_rem_created_at ON pack023_sbti_alignment.pack023_sbti_remediation_guidance(created_at DESC);
CREATE INDEX idx_pk_rem_resources ON pack023_sbti_alignment.pack023_sbti_remediation_guidance USING GIN(resource_requirements);
CREATE INDEX idx_pk_rem_deps ON pack023_sbti_alignment.pack023_sbti_remediation_guidance USING GIN(dependencies_on_actions);

-- Updated_at trigger
CREATE TRIGGER trg_pk_rem_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_remediation_guidance
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

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_validation_assessments IS
'Overall validation assessment records tracking results of 42-criterion SBTi validation with pass/fail counts and submission readiness estimation.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_criterion_results IS
'Individual criterion assessment results for all 42 SBTi criteria (C1-C28 near-term, NZ-C1-NZ-C14 net-zero) with pass/fail/warning status and evidence links.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_validation_gaps IS
'Detailed records of identified gaps for each failed or warning criterion with root cause analysis, impact assessment, and severity scoring.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_remediation_guidance IS
'Step-by-step remediation guidance for each gap with action sequences, responsible parties, timelines, and success criteria.';
