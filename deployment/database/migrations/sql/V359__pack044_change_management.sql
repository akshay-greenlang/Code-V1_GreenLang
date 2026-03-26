-- =============================================================================
-- V359: PACK-044 GHG Inventory Management - Change Management Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Change management tables for controlling modifications to finalised or
-- in-progress GHG inventory data. Implements a formal change request workflow
-- with impact assessment and multi-level approval. Ensures that any
-- retrospective changes to inventory data are documented, justified, and
-- traceable for audit and verification purposes.
--
-- Tables (3):
--   1. ghg_inventory.gl_inv_change_requests
--   2. ghg_inventory.gl_inv_change_impacts
--   3. ghg_inventory.gl_inv_change_approvals
--
-- Previous: V358__pack044_quality_management.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_change_requests
-- =============================================================================
-- A formal request to modify inventory data after it has been submitted or
-- approved. Change requests document the reason for the change, the scope
-- of impact, and require approval before the change is applied.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_change_requests (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    request_code                VARCHAR(50)     NOT NULL,
    request_title               VARCHAR(300)    NOT NULL,
    request_description         TEXT            NOT NULL,
    change_type                 VARCHAR(30)     NOT NULL DEFAULT 'DATA_CORRECTION',
    change_scope                VARCHAR(30)     NOT NULL DEFAULT 'FACILITY',
    facility_id                 UUID,
    source_category             VARCHAR(60),
    requested_by_user_id        UUID,
    requested_by_name           VARCHAR(255),
    priority                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    justification               TEXT            NOT NULL,
    supporting_evidence         TEXT,
    evidence_file_ids           UUID[],
    original_value              NUMERIC(18,6),
    proposed_value              NUMERIC(18,6),
    original_unit               VARCHAR(50),
    proposed_unit               VARCHAR(50),
    impact_tco2e                NUMERIC(12,3),
    impact_pct                  NUMERIC(8,3),
    requires_restatement        BOOLEAN         NOT NULL DEFAULT false,
    requires_base_year_recalc   BOOLEAN         NOT NULL DEFAULT false,
    submitted_at                TIMESTAMPTZ,
    decided_at                  TIMESTAMPTZ,
    decided_by                  VARCHAR(255),
    decision_notes              TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p044_chg_type CHECK (
        change_type IN (
            'DATA_CORRECTION', 'METHODOLOGY_CHANGE', 'BOUNDARY_CHANGE',
            'EMISSION_FACTOR_UPDATE', 'STRUCTURAL_CHANGE', 'ERROR_CORRECTION',
            'RESTATEMENT', 'OTHER'
        )
    ),
    CONSTRAINT chk_p044_chg_scope CHECK (
        change_scope IN (
            'FACILITY', 'ENTITY', 'ORGANIZATION', 'SOURCE_CATEGORY', 'PERIOD', 'GLOBAL'
        )
    ),
    CONSTRAINT chk_p044_chg_priority CHECK (
        priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p044_chg_status CHECK (
        status IN (
            'DRAFT', 'SUBMITTED', 'UNDER_REVIEW', 'IMPACT_ASSESSMENT',
            'APPROVED', 'REJECTED', 'APPLIED', 'CANCELLED', 'WITHDRAWN'
        )
    ),
    CONSTRAINT uq_p044_chg_tenant_code UNIQUE (tenant_id, request_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_chg_tenant         ON ghg_inventory.gl_inv_change_requests(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_chg_period         ON ghg_inventory.gl_inv_change_requests(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_chg_code           ON ghg_inventory.gl_inv_change_requests(request_code);
CREATE INDEX IF NOT EXISTS idx_p044_chg_type           ON ghg_inventory.gl_inv_change_requests(change_type);
CREATE INDEX IF NOT EXISTS idx_p044_chg_scope          ON ghg_inventory.gl_inv_change_requests(change_scope);
CREATE INDEX IF NOT EXISTS idx_p044_chg_facility       ON ghg_inventory.gl_inv_change_requests(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_chg_priority       ON ghg_inventory.gl_inv_change_requests(priority);
CREATE INDEX IF NOT EXISTS idx_p044_chg_status         ON ghg_inventory.gl_inv_change_requests(status);
CREATE INDEX IF NOT EXISTS idx_p044_chg_requested_by   ON ghg_inventory.gl_inv_change_requests(requested_by_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_chg_created        ON ghg_inventory.gl_inv_change_requests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_chg_metadata       ON ghg_inventory.gl_inv_change_requests USING GIN(metadata);

-- Composite: period + open change requests
CREATE INDEX IF NOT EXISTS idx_p044_chg_period_open    ON ghg_inventory.gl_inv_change_requests(period_id, priority)
    WHERE status IN ('SUBMITTED', 'UNDER_REVIEW', 'IMPACT_ASSESSMENT');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_chg_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_change_requests
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_change_impacts
-- =============================================================================
-- Impact assessment records for change requests. Quantifies the effect of
-- a proposed change on emissions totals, intensity metrics, compliance
-- status, and base year recalculation triggers.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_change_impacts (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    change_request_id           UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_change_requests(id) ON DELETE CASCADE,
    impact_area                 VARCHAR(50)     NOT NULL,
    impact_description          TEXT            NOT NULL,
    original_total_tco2e        NUMERIC(14,3),
    revised_total_tco2e         NUMERIC(14,3),
    absolute_impact_tco2e       NUMERIC(14,3),
    relative_impact_pct         NUMERIC(8,3),
    affects_compliance          BOOLEAN         NOT NULL DEFAULT false,
    affected_frameworks         TEXT[],
    affects_base_year           BOOLEAN         NOT NULL DEFAULT false,
    affects_targets             BOOLEAN         NOT NULL DEFAULT false,
    risk_level                  VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    assessed_by_user_id         UUID,
    assessed_by_name            VARCHAR(255),
    assessed_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ci_area CHECK (
        impact_area IN (
            'SCOPE_1_TOTAL', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3_TOTAL',
            'TOTAL_EMISSIONS', 'INTENSITY_METRIC', 'COMPLIANCE_STATUS',
            'BASE_YEAR', 'TARGET_PROGRESS', 'ETS_POSITION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p044_ci_risk CHECK (
        risk_level IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ci_tenant          ON ghg_inventory.gl_inv_change_impacts(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ci_change_req      ON ghg_inventory.gl_inv_change_impacts(change_request_id);
CREATE INDEX IF NOT EXISTS idx_p044_ci_area            ON ghg_inventory.gl_inv_change_impacts(impact_area);
CREATE INDEX IF NOT EXISTS idx_p044_ci_risk            ON ghg_inventory.gl_inv_change_impacts(risk_level);
CREATE INDEX IF NOT EXISTS idx_p044_ci_compliance      ON ghg_inventory.gl_inv_change_impacts(affects_compliance) WHERE affects_compliance = true;
CREATE INDEX IF NOT EXISTS idx_p044_ci_base_year       ON ghg_inventory.gl_inv_change_impacts(affects_base_year) WHERE affects_base_year = true;
CREATE INDEX IF NOT EXISTS idx_p044_ci_created         ON ghg_inventory.gl_inv_change_impacts(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ci_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_change_impacts
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_change_approvals
-- =============================================================================
-- Approval records for change requests. Supports multi-level approval with
-- sequential or parallel reviewers. Each approval record captures the
-- reviewer's decision, comments, and conditions.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_change_approvals (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    change_request_id           UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_change_requests(id) ON DELETE CASCADE,
    approval_level              INTEGER         NOT NULL DEFAULT 1,
    approver_user_id            UUID,
    approver_name               VARCHAR(255)    NOT NULL,
    approver_role               VARCHAR(100),
    decision                    VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    decision_date               TIMESTAMPTZ,
    comments                    TEXT,
    conditions                  TEXT,
    is_final_approver           BOOLEAN         NOT NULL DEFAULT false,
    delegated_from_user_id      UUID,
    delegated_from_name         VARCHAR(255),
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ca_level CHECK (
        approval_level >= 1 AND approval_level <= 10
    ),
    CONSTRAINT chk_p044_ca_decision CHECK (
        decision IN ('PENDING', 'APPROVED', 'APPROVED_WITH_CONDITIONS', 'REJECTED', 'DEFERRED', 'ABSTAINED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ca_tenant          ON ghg_inventory.gl_inv_change_approvals(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ca_change_req      ON ghg_inventory.gl_inv_change_approvals(change_request_id);
CREATE INDEX IF NOT EXISTS idx_p044_ca_approver        ON ghg_inventory.gl_inv_change_approvals(approver_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_ca_level           ON ghg_inventory.gl_inv_change_approvals(approval_level);
CREATE INDEX IF NOT EXISTS idx_p044_ca_decision        ON ghg_inventory.gl_inv_change_approvals(decision);
CREATE INDEX IF NOT EXISTS idx_p044_ca_created         ON ghg_inventory.gl_inv_change_approvals(created_at DESC);

-- Composite: change request + pending approvals
CREATE INDEX IF NOT EXISTS idx_p044_ca_req_pending     ON ghg_inventory.gl_inv_change_approvals(change_request_id, approval_level)
    WHERE decision = 'PENDING';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ca_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_change_approvals
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_change_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_change_impacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_change_approvals ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_chg_tenant_isolation
    ON ghg_inventory.gl_inv_change_requests
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_chg_service_bypass
    ON ghg_inventory.gl_inv_change_requests
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ci_tenant_isolation
    ON ghg_inventory.gl_inv_change_impacts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ci_service_bypass
    ON ghg_inventory.gl_inv_change_impacts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ca_tenant_isolation
    ON ghg_inventory.gl_inv_change_approvals
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ca_service_bypass
    ON ghg_inventory.gl_inv_change_approvals
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_change_requests TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_change_impacts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_change_approvals TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_change_requests IS
    'Formal change requests for modifying inventory data after submission or approval, with justification and impact tracking.';
COMMENT ON TABLE ghg_inventory.gl_inv_change_impacts IS
    'Impact assessments quantifying the effect of proposed changes on emissions totals, compliance, and base year.';
COMMENT ON TABLE ghg_inventory.gl_inv_change_approvals IS
    'Multi-level approval records for change requests with reviewer decisions and conditions.';

COMMENT ON COLUMN ghg_inventory.gl_inv_change_requests.change_type IS 'Type of change: DATA_CORRECTION, METHODOLOGY_CHANGE, BOUNDARY_CHANGE, EMISSION_FACTOR_UPDATE, STRUCTURAL_CHANGE, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_change_requests.requires_restatement IS 'Whether this change triggers a formal restatement of the inventory period.';
COMMENT ON COLUMN ghg_inventory.gl_inv_change_requests.requires_base_year_recalc IS 'Whether this change triggers a base year recalculation per GHG Protocol guidance.';
COMMENT ON COLUMN ghg_inventory.gl_inv_change_impacts.impact_area IS 'Which area of the inventory is affected: SCOPE_1_TOTAL, COMPLIANCE_STATUS, BASE_YEAR, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_change_approvals.is_final_approver IS 'Whether this is the final approver whose decision determines the outcome.';
