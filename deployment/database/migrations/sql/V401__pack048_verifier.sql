-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V401 - Verifier Query & Finding Management
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for managing verifier interactions: queries (information
-- requests, clarifications, evidence requests), findings (non-conformities,
-- observations, recommendations), and responses to both. Supports SLA
-- tracking for query response times, recurring finding detection, and
-- structured evidence submission.
--
-- Tables (3):
--   1. ghg_assurance.gl_ap_queries
--   2. ghg_assurance.gl_ap_findings
--   3. ghg_assurance.gl_ap_responses
--
-- Also includes: indexes, RLS, comments.
-- Previous: V400__pack048_controls.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_queries
-- =============================================================================
-- Verifier queries raised during the assurance engagement. Each query has
-- a type (information request, clarification, evidence request, follow-up),
-- category, priority, SLA deadline, and lifecycle status. Overdue tracking
-- is maintained via the is_overdue flag.

CREATE TABLE ghg_assurance.gl_ap_queries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    engagement_id               UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_engagements(id) ON DELETE CASCADE,
    query_type                  VARCHAR(30)     NOT NULL,
    category                    VARCHAR(100),
    priority                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    query_text                  TEXT            NOT NULL,
    raised_by                   UUID,
    raised_at                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    sla_deadline                DATE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    resolved_at                 TIMESTAMPTZ,
    is_overdue                  BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_qry_type CHECK (
        query_type IN (
            'INFORMATION_REQUEST', 'CLARIFICATION',
            'EVIDENCE_REQUEST', 'FOLLOW_UP'
        )
    ),
    CONSTRAINT chk_p048_qry_priority CHECK (
        priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p048_qry_status CHECK (
        status IN (
            'OPEN', 'IN_PROGRESS', 'RESPONDED',
            'FOLLOW_UP', 'RESOLVED', 'CLOSED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_qry_tenant           ON ghg_assurance.gl_ap_queries(tenant_id);
CREATE INDEX idx_p048_qry_engagement       ON ghg_assurance.gl_ap_queries(engagement_id);
CREATE INDEX idx_p048_qry_type             ON ghg_assurance.gl_ap_queries(query_type);
CREATE INDEX idx_p048_qry_priority         ON ghg_assurance.gl_ap_queries(priority);
CREATE INDEX idx_p048_qry_status           ON ghg_assurance.gl_ap_queries(status);
CREATE INDEX idx_p048_qry_sla              ON ghg_assurance.gl_ap_queries(sla_deadline);
CREATE INDEX idx_p048_qry_raised_at        ON ghg_assurance.gl_ap_queries(raised_at DESC);
CREATE INDEX idx_p048_qry_overdue          ON ghg_assurance.gl_ap_queries(is_overdue) WHERE is_overdue = true;
CREATE INDEX idx_p048_qry_raised_by        ON ghg_assurance.gl_ap_queries(raised_by);
CREATE INDEX idx_p048_qry_created          ON ghg_assurance.gl_ap_queries(created_at DESC);
CREATE INDEX idx_p048_qry_metadata         ON ghg_assurance.gl_ap_queries USING GIN(metadata);

-- Composite: engagement + status for engagement-level tracking
CREATE INDEX idx_p048_qry_eng_status       ON ghg_assurance.gl_ap_queries(engagement_id, status);

-- Composite: engagement + priority for priority-sorted listing
CREATE INDEX idx_p048_qry_eng_priority     ON ghg_assurance.gl_ap_queries(engagement_id, priority);

-- Composite: status + sla_deadline for SLA monitoring
CREATE INDEX idx_p048_qry_status_sla       ON ghg_assurance.gl_ap_queries(status, sla_deadline);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_qry_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_queries
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_findings
-- =============================================================================
-- Verifier findings from the assurance engagement. Each finding has a type
-- (non-conformity, observation, opportunity, recommendation, good practice),
-- severity, affected scope and category, root cause analysis, remediation
-- plan with owner and deadline, and recurring flag for trend detection.

CREATE TABLE ghg_assurance.gl_ap_findings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    engagement_id               UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_engagements(id) ON DELETE CASCADE,
    finding_type                VARCHAR(30)     NOT NULL,
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'OBSERVATION',
    finding_description         TEXT            NOT NULL,
    affected_scope              VARCHAR(20),
    affected_category           VARCHAR(100),
    root_cause                  TEXT,
    remediation_plan            TEXT,
    remediation_owner           UUID,
    remediation_deadline        DATE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    is_recurring                BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_fnd_type CHECK (
        finding_type IN (
            'NON_CONFORMITY', 'OBSERVATION', 'OPPORTUNITY',
            'RECOMMENDATION', 'GOOD_PRACTICE'
        )
    ),
    CONSTRAINT chk_p048_fnd_severity CHECK (
        severity IN ('CRITICAL', 'MAJOR', 'MINOR', 'OBSERVATION')
    ),
    CONSTRAINT chk_p048_fnd_scope CHECK (
        affected_scope IS NULL OR affected_scope IN (
            'SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'CROSS_SCOPE'
        )
    ),
    CONSTRAINT chk_p048_fnd_status CHECK (
        status IN (
            'OPEN', 'IN_PROGRESS', 'REMEDIATED', 'VERIFIED_CLOSED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_fnd_tenant           ON ghg_assurance.gl_ap_findings(tenant_id);
CREATE INDEX idx_p048_fnd_engagement       ON ghg_assurance.gl_ap_findings(engagement_id);
CREATE INDEX idx_p048_fnd_type             ON ghg_assurance.gl_ap_findings(finding_type);
CREATE INDEX idx_p048_fnd_severity         ON ghg_assurance.gl_ap_findings(severity);
CREATE INDEX idx_p048_fnd_scope            ON ghg_assurance.gl_ap_findings(affected_scope);
CREATE INDEX idx_p048_fnd_status           ON ghg_assurance.gl_ap_findings(status);
CREATE INDEX idx_p048_fnd_recurring        ON ghg_assurance.gl_ap_findings(is_recurring) WHERE is_recurring = true;
CREATE INDEX idx_p048_fnd_owner            ON ghg_assurance.gl_ap_findings(remediation_owner);
CREATE INDEX idx_p048_fnd_deadline         ON ghg_assurance.gl_ap_findings(remediation_deadline);
CREATE INDEX idx_p048_fnd_created          ON ghg_assurance.gl_ap_findings(created_at DESC);
CREATE INDEX idx_p048_fnd_metadata         ON ghg_assurance.gl_ap_findings USING GIN(metadata);

-- Composite: engagement + severity for priority display
CREATE INDEX idx_p048_fnd_eng_severity     ON ghg_assurance.gl_ap_findings(engagement_id, severity);

-- Composite: engagement + status for progress tracking
CREATE INDEX idx_p048_fnd_eng_status       ON ghg_assurance.gl_ap_findings(engagement_id, status);

-- Composite: type + severity for classification matrix
CREATE INDEX idx_p048_fnd_type_severity    ON ghg_assurance.gl_ap_findings(finding_type, severity);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_fnd_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_findings
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_assurance.gl_ap_responses
-- =============================================================================
-- Responses to verifier queries and findings. Each response can reference
-- either a query or a finding (or both), includes the response text,
-- evidence references (JSONB array of file paths/URLs), and acceptance
-- status from the verifier.

CREATE TABLE ghg_assurance.gl_ap_responses (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    query_id                    UUID            REFERENCES ghg_assurance.gl_ap_queries(id) ON DELETE SET NULL,
    finding_id                  UUID            REFERENCES ghg_assurance.gl_ap_findings(id) ON DELETE SET NULL,
    response_text               TEXT            NOT NULL,
    evidence_references         JSONB           DEFAULT '[]',
    responded_by                UUID,
    responded_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    is_accepted                 BOOLEAN,
    acceptance_notes            TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_rsp_has_parent CHECK (
        query_id IS NOT NULL OR finding_id IS NOT NULL
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_rsp_tenant           ON ghg_assurance.gl_ap_responses(tenant_id);
CREATE INDEX idx_p048_rsp_query            ON ghg_assurance.gl_ap_responses(query_id);
CREATE INDEX idx_p048_rsp_finding          ON ghg_assurance.gl_ap_responses(finding_id);
CREATE INDEX idx_p048_rsp_responded_by     ON ghg_assurance.gl_ap_responses(responded_by);
CREATE INDEX idx_p048_rsp_responded_at     ON ghg_assurance.gl_ap_responses(responded_at DESC);
CREATE INDEX idx_p048_rsp_accepted         ON ghg_assurance.gl_ap_responses(is_accepted);
CREATE INDEX idx_p048_rsp_created          ON ghg_assurance.gl_ap_responses(created_at DESC);
CREATE INDEX idx_p048_rsp_evidence         ON ghg_assurance.gl_ap_responses USING GIN(evidence_references);
CREATE INDEX idx_p048_rsp_metadata         ON ghg_assurance.gl_ap_responses USING GIN(metadata);

-- Composite: query + responded_at for query response timeline
CREATE INDEX idx_p048_rsp_query_time       ON ghg_assurance.gl_ap_responses(query_id, responded_at DESC);

-- Composite: finding + responded_at for finding response timeline
CREATE INDEX idx_p048_rsp_finding_time     ON ghg_assurance.gl_ap_responses(finding_id, responded_at DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_responses ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_qry_tenant_isolation
    ON ghg_assurance.gl_ap_queries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_qry_service_bypass
    ON ghg_assurance.gl_ap_queries
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_fnd_tenant_isolation
    ON ghg_assurance.gl_ap_findings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_fnd_service_bypass
    ON ghg_assurance.gl_ap_findings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_rsp_tenant_isolation
    ON ghg_assurance.gl_ap_responses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_rsp_service_bypass
    ON ghg_assurance.gl_ap_responses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_queries TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_findings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_responses TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_queries IS
    'Verifier queries with type classification, priority, SLA tracking, and lifecycle management.';
COMMENT ON TABLE ghg_assurance.gl_ap_findings IS
    'Verifier findings with severity classification, scope attribution, root cause analysis, and remediation tracking.';
COMMENT ON TABLE ghg_assurance.gl_ap_responses IS
    'Responses to verifier queries and findings with evidence references and acceptance tracking.';

COMMENT ON COLUMN ghg_assurance.gl_ap_queries.query_type IS 'INFORMATION_REQUEST (new data), CLARIFICATION (explanation), EVIDENCE_REQUEST (supporting docs), FOLLOW_UP (additional info on prior query).';
COMMENT ON COLUMN ghg_assurance.gl_ap_queries.priority IS 'Query priority: CRITICAL (blocks engagement), HIGH (urgent), MEDIUM (standard), LOW (informational).';
COMMENT ON COLUMN ghg_assurance.gl_ap_queries.sla_deadline IS 'Response deadline based on engagement SLA (typically 3-5 business days).';
COMMENT ON COLUMN ghg_assurance.gl_ap_queries.is_overdue IS 'Automatically set true when sla_deadline has passed and status is not RESOLVED/CLOSED.';
COMMENT ON COLUMN ghg_assurance.gl_ap_queries.status IS 'OPEN (raised), IN_PROGRESS (being addressed), RESPONDED (answer submitted), FOLLOW_UP (additional info needed), RESOLVED (accepted), CLOSED (finalised).';
COMMENT ON COLUMN ghg_assurance.gl_ap_findings.finding_type IS 'NON_CONFORMITY (standard breach), OBSERVATION (potential issue), OPPORTUNITY (improvement), RECOMMENDATION (best practice suggestion), GOOD_PRACTICE (positive note).';
COMMENT ON COLUMN ghg_assurance.gl_ap_findings.severity IS 'CRITICAL (qualified opinion risk), MAJOR (significant issue), MINOR (low-impact issue), OBSERVATION (informational).';
COMMENT ON COLUMN ghg_assurance.gl_ap_findings.is_recurring IS 'True if the same or similar finding was raised in a prior engagement.';
COMMENT ON COLUMN ghg_assurance.gl_ap_responses.evidence_references IS 'JSON array of evidence references, e.g. [{"type":"file","path":"/evidence/doc.pdf"},{"type":"url","url":"https://..."}].';
COMMENT ON COLUMN ghg_assurance.gl_ap_responses.is_accepted IS 'Whether the verifier accepted the response. NULL = pending review.';
