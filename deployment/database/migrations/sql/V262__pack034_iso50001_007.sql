-- =============================================================================
-- V262: PACK-034 ISO 50001 Energy Management System - Compliance Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Creates compliance assessment tables for tracking ISO 50001 gap analyses,
-- internal audits, and certification audits. Manages clause-level scoring,
-- nonconformities, and corrective actions per ISO 50001 Clause 9.2/10.2.
--
-- Tables (4):
--   1. pack034_iso50001.compliance_assessments
--   2. pack034_iso50001.clause_scores
--   3. pack034_iso50001.nonconformities
--   4. pack034_iso50001.corrective_actions
--
-- Previous: V261__pack034_iso50001_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.compliance_assessments
-- =============================================================================
-- Assessment records for gap analyses, internal audits, and certification
-- stage 1/2/surveillance audits against ISO 50001 requirements.

CREATE TABLE pack034_iso50001.compliance_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    assessment_date             DATE            NOT NULL,
    assessment_type             VARCHAR(30)     NOT NULL,
    assessor                    VARCHAR(255)    NOT NULL,
    overall_score               DECIMAL(6,2),
    total_clauses               INTEGER         NOT NULL DEFAULT 0,
    conforming_clauses          INTEGER         NOT NULL DEFAULT 0,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'in_progress',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ca_type CHECK (
        assessment_type IN ('gap_analysis', 'internal_audit', 'stage1', 'stage2', 'surveillance')
    ),
    CONSTRAINT chk_p034_ca_score CHECK (
        overall_score IS NULL OR (overall_score >= 0 AND overall_score <= 100)
    ),
    CONSTRAINT chk_p034_ca_clauses CHECK (
        total_clauses >= 0 AND conforming_clauses >= 0 AND conforming_clauses <= total_clauses
    ),
    CONSTRAINT chk_p034_ca_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'closed')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_assess_enms        ON pack034_iso50001.compliance_assessments(enms_id);
CREATE INDEX idx_p034_assess_date        ON pack034_iso50001.compliance_assessments(assessment_date DESC);
CREATE INDEX idx_p034_assess_type        ON pack034_iso50001.compliance_assessments(assessment_type);
CREATE INDEX idx_p034_assess_score       ON pack034_iso50001.compliance_assessments(overall_score DESC);
CREATE INDEX idx_p034_assess_status      ON pack034_iso50001.compliance_assessments(status);
CREATE INDEX idx_p034_assess_created     ON pack034_iso50001.compliance_assessments(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_assess_updated
    BEFORE UPDATE ON pack034_iso50001.compliance_assessments
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.clause_scores
-- =============================================================================
-- Individual clause-level scores within an assessment, tracking conformity
-- status and supporting evidence references for each ISO 50001 clause.

CREATE TABLE pack034_iso50001.clause_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id               UUID            NOT NULL REFERENCES pack034_iso50001.compliance_assessments(id) ON DELETE CASCADE,
    clause_number               VARCHAR(10)     NOT NULL,
    clause_title                VARCHAR(500)    NOT NULL,
    score                       DECIMAL(6,2),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'not_assessed',
    evidence_refs_json          JSONB           DEFAULT '[]',
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_cs_score CHECK (
        score IS NULL OR (score >= 0 AND score <= 100)
    ),
    CONSTRAINT chk_p034_cs_status CHECK (
        status IN ('conforming', 'minor_nc', 'major_nc', 'opportunity', 'not_applicable', 'not_assessed')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_cs_assessment      ON pack034_iso50001.clause_scores(assessment_id);
CREATE INDEX idx_p034_cs_clause          ON pack034_iso50001.clause_scores(clause_number);
CREATE INDEX idx_p034_cs_status          ON pack034_iso50001.clause_scores(status);
CREATE INDEX idx_p034_cs_score           ON pack034_iso50001.clause_scores(score);
CREATE INDEX idx_p034_cs_evidence        ON pack034_iso50001.clause_scores USING GIN(evidence_refs_json);
CREATE INDEX idx_p034_cs_created         ON pack034_iso50001.clause_scores(created_at DESC);
CREATE UNIQUE INDEX idx_p034_cs_assess_clause
    ON pack034_iso50001.clause_scores(assessment_id, clause_number);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_cs_updated
    BEFORE UPDATE ON pack034_iso50001.clause_scores
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.nonconformities
-- =============================================================================
-- Nonconformities identified during assessments with severity classification,
-- root cause documentation, and lifecycle tracking through to closure.

CREATE TABLE pack034_iso50001.nonconformities (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id               UUID            NOT NULL REFERENCES pack034_iso50001.compliance_assessments(id) ON DELETE CASCADE,
    nc_number                   VARCHAR(50)     NOT NULL,
    clause_ref                  VARCHAR(10)     NOT NULL,
    description                 TEXT            NOT NULL,
    severity                    VARCHAR(20)     NOT NULL,
    root_cause                  TEXT,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'open',
    due_date                    DATE,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_nc_severity CHECK (
        severity IN ('minor', 'major', 'critical')
    ),
    CONSTRAINT chk_p034_nc_status CHECK (
        status IN ('open', 'corrective_action', 'verified', 'closed')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_nc_assessment      ON pack034_iso50001.nonconformities(assessment_id);
CREATE INDEX idx_p034_nc_number          ON pack034_iso50001.nonconformities(nc_number);
CREATE INDEX idx_p034_nc_clause          ON pack034_iso50001.nonconformities(clause_ref);
CREATE INDEX idx_p034_nc_severity        ON pack034_iso50001.nonconformities(severity);
CREATE INDEX idx_p034_nc_status          ON pack034_iso50001.nonconformities(status);
CREATE INDEX idx_p034_nc_due             ON pack034_iso50001.nonconformities(due_date);
CREATE INDEX idx_p034_nc_created         ON pack034_iso50001.nonconformities(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_nc_updated
    BEFORE UPDATE ON pack034_iso50001.nonconformities
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack034_iso50001.corrective_actions
-- =============================================================================
-- Corrective actions addressing nonconformities, with planning, execution,
-- verification, and effectiveness checking per ISO 50001 Clause 10.2.

CREATE TABLE pack034_iso50001.corrective_actions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    nc_id                       UUID            NOT NULL REFERENCES pack034_iso50001.nonconformities(id) ON DELETE CASCADE,
    action_description          TEXT            NOT NULL,
    responsible_person          VARCHAR(255)    NOT NULL,
    planned_date                DATE            NOT NULL,
    completed_date              DATE,
    verification_method         TEXT,
    effectiveness_check         BOOLEAN         DEFAULT FALSE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'planned',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_cora_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'verified', 'ineffective')
    ),
    CONSTRAINT chk_p034_cora_dates CHECK (
        completed_date IS NULL OR completed_date >= planned_date - INTERVAL '30 days'
    ),
    CONSTRAINT chk_p034_cora_effectiveness CHECK (
        effectiveness_check = FALSE OR status IN ('completed', 'verified', 'ineffective')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_cora_nc            ON pack034_iso50001.corrective_actions(nc_id);
CREATE INDEX idx_p034_cora_responsible   ON pack034_iso50001.corrective_actions(responsible_person);
CREATE INDEX idx_p034_cora_planned       ON pack034_iso50001.corrective_actions(planned_date);
CREATE INDEX idx_p034_cora_status        ON pack034_iso50001.corrective_actions(status);
CREATE INDEX idx_p034_cora_effectiveness ON pack034_iso50001.corrective_actions(effectiveness_check);
CREATE INDEX idx_p034_cora_created       ON pack034_iso50001.corrective_actions(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_cora_updated
    BEFORE UPDATE ON pack034_iso50001.corrective_actions
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.compliance_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.clause_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.nonconformities ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.corrective_actions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_assess_tenant_isolation
    ON pack034_iso50001.compliance_assessments
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_assess_service_bypass
    ON pack034_iso50001.compliance_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_cs_tenant_isolation
    ON pack034_iso50001.clause_scores
    USING (assessment_id IN (
        SELECT id FROM pack034_iso50001.compliance_assessments
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_cs_service_bypass
    ON pack034_iso50001.clause_scores
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_nc_tenant_isolation
    ON pack034_iso50001.nonconformities
    USING (assessment_id IN (
        SELECT id FROM pack034_iso50001.compliance_assessments
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_nc_service_bypass
    ON pack034_iso50001.nonconformities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_cora_tenant_isolation
    ON pack034_iso50001.corrective_actions
    USING (nc_id IN (
        SELECT id FROM pack034_iso50001.nonconformities
        WHERE assessment_id IN (
            SELECT id FROM pack034_iso50001.compliance_assessments
            WHERE enms_id IN (
                SELECT id FROM pack034_iso50001.energy_management_systems
                WHERE organization_id = current_setting('app.current_tenant')::UUID
            )
        )
    ));
CREATE POLICY p034_cora_service_bypass
    ON pack034_iso50001.corrective_actions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.compliance_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.clause_scores TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.nonconformities TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.corrective_actions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.compliance_assessments IS
    'Assessment records for gap analyses, internal audits, and certification audits against ISO 50001 requirements.';

COMMENT ON TABLE pack034_iso50001.clause_scores IS
    'Clause-level conformity scores within an assessment with evidence references for each ISO 50001 clause.';

COMMENT ON TABLE pack034_iso50001.nonconformities IS
    'Nonconformities identified during assessments with severity classification and lifecycle tracking.';

COMMENT ON TABLE pack034_iso50001.corrective_actions IS
    'Corrective actions addressing nonconformities with planning, execution, and effectiveness verification.';

COMMENT ON COLUMN pack034_iso50001.compliance_assessments.assessment_type IS
    'Type: gap_analysis (initial), internal_audit (ongoing), stage1/stage2 (certification), surveillance (annual).';
COMMENT ON COLUMN pack034_iso50001.compliance_assessments.overall_score IS
    'Overall conformity score (0-100%). Certification typically requires >= 80% with no major NCs.';
COMMENT ON COLUMN pack034_iso50001.clause_scores.clause_number IS
    'ISO 50001:2018 clause number (e.g., 4.1, 6.3, 8.1).';
COMMENT ON COLUMN pack034_iso50001.clause_scores.status IS
    'Conformity status: conforming, minor_nc, major_nc, opportunity (for improvement), not_applicable.';
COMMENT ON COLUMN pack034_iso50001.clause_scores.evidence_refs_json IS
    'JSON array of evidence document references supporting the assessment (e.g., ["DOC-001", "PROC-045"]).';
COMMENT ON COLUMN pack034_iso50001.nonconformities.severity IS
    'Severity: minor (isolated, no systemic failure), major (systemic or repeated), critical (imminent risk).';
COMMENT ON COLUMN pack034_iso50001.nonconformities.root_cause IS
    'Root cause analysis documenting the underlying reason for the nonconformity.';
COMMENT ON COLUMN pack034_iso50001.corrective_actions.effectiveness_check IS
    'Whether an effectiveness check has been performed to verify the corrective action resolved the NC.';
COMMENT ON COLUMN pack034_iso50001.corrective_actions.verification_method IS
    'Method used to verify the corrective action was implemented effectively.';
