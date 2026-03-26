-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V398 - Readiness Assessment
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates readiness assessment tables for evaluating an organisation's
-- preparedness for external GHG assurance. Assessments produce an overall
-- readiness score and level, broken down by category. Checklist items
-- provide granular scoring against standard-specific requirements with
-- mandatory/gate flags. Gaps track remediation actions with severity,
-- priority ranking, and lifecycle status.
--
-- Tables (3):
--   1. ghg_assurance.gl_ap_readiness_assessments
--   2. ghg_assurance.gl_ap_checklist_items
--   3. ghg_assurance.gl_ap_gaps
--
-- Also includes: indexes, RLS, comments.
-- Previous: V397__pack048_evidence.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_readiness_assessments
-- =============================================================================
-- Readiness assessment results per configuration and assurance standard.
-- Produces an overall readiness score (0-100), readiness level classification,
-- category-level breakdowns, gap count, estimated days to readiness, and
-- full provenance for audit trail.

CREATE TABLE ghg_assurance.gl_ap_readiness_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    standard                    VARCHAR(30)     NOT NULL,
    assessment_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    overall_score               NUMERIC(5,2)    NOT NULL,
    readiness_level             VARCHAR(20)     NOT NULL,
    category_scores             JSONB           NOT NULL DEFAULT '{}',
    gap_count                   INTEGER         NOT NULL DEFAULT 0,
    time_to_ready_days          INTEGER,
    assessed_by                 UUID,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_ra_standard CHECK (
        standard IN (
            'ISAE_3410', 'ISO_14064_3', 'AA1000AS_V3',
            'ISAE_3000', 'SSAE_18'
        )
    ),
    CONSTRAINT chk_p048_ra_score CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_p048_ra_level CHECK (
        readiness_level IN (
            'READY', 'MOSTLY_READY', 'PARTIALLY_READY', 'NOT_READY'
        )
    ),
    CONSTRAINT chk_p048_ra_gap_count CHECK (
        gap_count >= 0
    ),
    CONSTRAINT chk_p048_ra_ttr CHECK (
        time_to_ready_days IS NULL OR time_to_ready_days >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ra_tenant            ON ghg_assurance.gl_ap_readiness_assessments(tenant_id);
CREATE INDEX idx_p048_ra_config            ON ghg_assurance.gl_ap_readiness_assessments(config_id);
CREATE INDEX idx_p048_ra_standard          ON ghg_assurance.gl_ap_readiness_assessments(standard);
CREATE INDEX idx_p048_ra_date              ON ghg_assurance.gl_ap_readiness_assessments(assessment_date DESC);
CREATE INDEX idx_p048_ra_score             ON ghg_assurance.gl_ap_readiness_assessments(overall_score);
CREATE INDEX idx_p048_ra_level             ON ghg_assurance.gl_ap_readiness_assessments(readiness_level);
CREATE INDEX idx_p048_ra_created           ON ghg_assurance.gl_ap_readiness_assessments(created_at DESC);
CREATE INDEX idx_p048_ra_provenance        ON ghg_assurance.gl_ap_readiness_assessments(provenance_hash);
CREATE INDEX idx_p048_ra_category          ON ghg_assurance.gl_ap_readiness_assessments USING GIN(category_scores);
CREATE INDEX idx_p048_ra_metadata          ON ghg_assurance.gl_ap_readiness_assessments USING GIN(metadata);

-- Composite: config + standard for standard-specific queries
CREATE INDEX idx_p048_ra_config_std        ON ghg_assurance.gl_ap_readiness_assessments(config_id, standard);

-- Composite: tenant + config for scoped listing
CREATE INDEX idx_p048_ra_tenant_config     ON ghg_assurance.gl_ap_readiness_assessments(tenant_id, config_id);

-- Composite: config + date for time series
CREATE INDEX idx_p048_ra_config_date       ON ghg_assurance.gl_ap_readiness_assessments(config_id, assessment_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_ra_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_readiness_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_checklist_items
-- =============================================================================
-- Granular checklist items within a readiness assessment. Each item maps to
-- a specific standard requirement with a category, code, description,
-- max score, actual score (0-4), evidence reference, and mandatory/gate
-- flags. Gate items must score above threshold for the assessment to pass.

CREATE TABLE ghg_assurance.gl_ap_checklist_items (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_readiness_assessments(id) ON DELETE CASCADE,
    category                    VARCHAR(100)    NOT NULL,
    item_code                   VARCHAR(30)     NOT NULL,
    item_description            TEXT            NOT NULL,
    max_score                   INTEGER         NOT NULL DEFAULT 4,
    actual_score                INTEGER,
    evidence_reference          TEXT,
    notes                       TEXT,
    is_mandatory                BOOLEAN         NOT NULL DEFAULT false,
    is_gate                     BOOLEAN         NOT NULL DEFAULT false,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_cl_max_score CHECK (
        max_score >= 1 AND max_score <= 10
    ),
    CONSTRAINT chk_p048_cl_actual_score CHECK (
        actual_score IS NULL OR (actual_score >= 0 AND actual_score <= 4)
    ),
    CONSTRAINT uq_p048_cl_assessment_code UNIQUE (assessment_id, item_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_cl_tenant            ON ghg_assurance.gl_ap_checklist_items(tenant_id);
CREATE INDEX idx_p048_cl_assessment        ON ghg_assurance.gl_ap_checklist_items(assessment_id);
CREATE INDEX idx_p048_cl_category          ON ghg_assurance.gl_ap_checklist_items(category);
CREATE INDEX idx_p048_cl_code              ON ghg_assurance.gl_ap_checklist_items(item_code);
CREATE INDEX idx_p048_cl_score             ON ghg_assurance.gl_ap_checklist_items(actual_score);
CREATE INDEX idx_p048_cl_mandatory         ON ghg_assurance.gl_ap_checklist_items(is_mandatory) WHERE is_mandatory = true;
CREATE INDEX idx_p048_cl_gate              ON ghg_assurance.gl_ap_checklist_items(is_gate) WHERE is_gate = true;
CREATE INDEX idx_p048_cl_created           ON ghg_assurance.gl_ap_checklist_items(created_at DESC);
CREATE INDEX idx_p048_cl_metadata          ON ghg_assurance.gl_ap_checklist_items USING GIN(metadata);

-- Composite: assessment + category for grouped display
CREATE INDEX idx_p048_cl_assess_category   ON ghg_assurance.gl_ap_checklist_items(assessment_id, category);

-- =============================================================================
-- Table 3: ghg_assurance.gl_ap_gaps
-- =============================================================================
-- Identified gaps from readiness assessments. Each gap is linked to a
-- checklist item, severity-classified (CRITICAL through LOW), assigned
-- a remediation recommendation with effort estimate, priority ranking,
-- and lifecycle tracking (OPEN through ACCEPTED).

CREATE TABLE ghg_assurance.gl_ap_gaps (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_readiness_assessments(id) ON DELETE CASCADE,
    checklist_item_id           UUID            REFERENCES ghg_assurance.gl_ap_checklist_items(id) ON DELETE SET NULL,
    gap_description             TEXT            NOT NULL,
    severity                    VARCHAR(20)     NOT NULL,
    remediation_recommendation  TEXT,
    estimated_effort_days       INTEGER,
    priority_rank               INTEGER,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    assigned_to                 UUID,
    due_date                    DATE,
    remediated_at               TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_gap_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p048_gap_status CHECK (
        status IN ('OPEN', 'IN_PROGRESS', 'REMEDIATED', 'ACCEPTED')
    ),
    CONSTRAINT chk_p048_gap_effort CHECK (
        estimated_effort_days IS NULL OR estimated_effort_days >= 0
    ),
    CONSTRAINT chk_p048_gap_priority CHECK (
        priority_rank IS NULL OR priority_rank >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_gap_tenant           ON ghg_assurance.gl_ap_gaps(tenant_id);
CREATE INDEX idx_p048_gap_assessment       ON ghg_assurance.gl_ap_gaps(assessment_id);
CREATE INDEX idx_p048_gap_checklist        ON ghg_assurance.gl_ap_gaps(checklist_item_id);
CREATE INDEX idx_p048_gap_severity         ON ghg_assurance.gl_ap_gaps(severity);
CREATE INDEX idx_p048_gap_status           ON ghg_assurance.gl_ap_gaps(status);
CREATE INDEX idx_p048_gap_priority         ON ghg_assurance.gl_ap_gaps(priority_rank);
CREATE INDEX idx_p048_gap_assigned         ON ghg_assurance.gl_ap_gaps(assigned_to);
CREATE INDEX idx_p048_gap_due_date         ON ghg_assurance.gl_ap_gaps(due_date);
CREATE INDEX idx_p048_gap_created          ON ghg_assurance.gl_ap_gaps(created_at DESC);
CREATE INDEX idx_p048_gap_metadata         ON ghg_assurance.gl_ap_gaps USING GIN(metadata);

-- Composite: assessment + severity for priority display
CREATE INDEX idx_p048_gap_assess_sev       ON ghg_assurance.gl_ap_gaps(assessment_id, severity);

-- Composite: status + due_date for overdue tracking
CREATE INDEX idx_p048_gap_status_due       ON ghg_assurance.gl_ap_gaps(status, due_date);

-- Composite: assessment + status for progress tracking
CREATE INDEX idx_p048_gap_assess_status    ON ghg_assurance.gl_ap_gaps(assessment_id, status);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_gap_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_gaps
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_readiness_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_checklist_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_gaps ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_ra_tenant_isolation
    ON ghg_assurance.gl_ap_readiness_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ra_service_bypass
    ON ghg_assurance.gl_ap_readiness_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_cl_tenant_isolation
    ON ghg_assurance.gl_ap_checklist_items
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_cl_service_bypass
    ON ghg_assurance.gl_ap_checklist_items
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_gap_tenant_isolation
    ON ghg_assurance.gl_ap_gaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_gap_service_bypass
    ON ghg_assurance.gl_ap_gaps
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_readiness_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_checklist_items TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_gaps TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_readiness_assessments IS
    'Readiness assessment results with overall score, readiness level, category breakdowns, and time-to-ready estimation.';
COMMENT ON TABLE ghg_assurance.gl_ap_checklist_items IS
    'Granular checklist items per assessment with scoring (0-4), mandatory/gate flags, and evidence references.';
COMMENT ON TABLE ghg_assurance.gl_ap_gaps IS
    'Identified readiness gaps with severity classification, remediation planning, priority ranking, and lifecycle tracking.';

COMMENT ON COLUMN ghg_assurance.gl_ap_readiness_assessments.overall_score IS 'Weighted aggregate score (0-100) across all checklist items.';
COMMENT ON COLUMN ghg_assurance.gl_ap_readiness_assessments.readiness_level IS 'READY (>=80), MOSTLY_READY (60-79), PARTIALLY_READY (40-59), NOT_READY (<40).';
COMMENT ON COLUMN ghg_assurance.gl_ap_readiness_assessments.category_scores IS 'JSON object mapping categories to scores, e.g. {"DATA_COLLECTION":85,"METHODOLOGY":72}.';
COMMENT ON COLUMN ghg_assurance.gl_ap_readiness_assessments.time_to_ready_days IS 'Estimated business days to remediate all gaps and reach READY status.';
COMMENT ON COLUMN ghg_assurance.gl_ap_checklist_items.max_score IS 'Maximum achievable score for this item (default 4). Some items may use different scales.';
COMMENT ON COLUMN ghg_assurance.gl_ap_checklist_items.actual_score IS 'Assessed score: 0=not addressed, 1=initial, 2=developing, 3=established, 4=optimised.';
COMMENT ON COLUMN ghg_assurance.gl_ap_checklist_items.is_mandatory IS 'True for items required by the assurance standard regardless of scope.';
COMMENT ON COLUMN ghg_assurance.gl_ap_checklist_items.is_gate IS 'True for gate items that must score >=2 for the assessment to pass.';
COMMENT ON COLUMN ghg_assurance.gl_ap_gaps.severity IS 'Gap severity: CRITICAL (blocks assurance), HIGH (significant risk), MEDIUM (improvement needed), LOW (minor enhancement).';
COMMENT ON COLUMN ghg_assurance.gl_ap_gaps.priority_rank IS 'Priority ranking (1=highest) based on severity, effort, and impact.';
COMMENT ON COLUMN ghg_assurance.gl_ap_gaps.status IS 'Gap lifecycle: OPEN (identified), IN_PROGRESS (remediation underway), REMEDIATED (fixed), ACCEPTED (risk accepted).';
