-- V443__factors_rollback_records.sql
-- GAP-5: Per-factor rollback audit trail.
-- Owned by greenlang.factors.quality.rollback.RollbackStore (SQLite dev mirror)
-- and greenlang.factors.quality.rollback.RollbackService (prod Postgres).
--
-- State machine (CHECK constraint below enforces):
--   planned -> approved -> executing -> (completed | failed)
--   planned | approved -> cancelled
--
-- Two-signature approval gate: approved_by_1 and approved_by_2 are
-- denormalised from the approvals_json blob so operators can filter
-- by signer without parsing JSON.

CREATE TABLE IF NOT EXISTS factors_rollback_records (
    rollback_id              UUID            PRIMARY KEY,
    factor_id                TEXT            NOT NULL,
    from_version             TEXT            NOT NULL,
    to_version               TEXT            NOT NULL,
    reason                   TEXT            NOT NULL,
    status                   TEXT            NOT NULL,
    approved_by_1            TEXT,
    approved_by_2            TEXT,
    approved_at              TIMESTAMPTZ,
    executed_at              TIMESTAMPTZ,
    affected_computations    INTEGER         NOT NULL DEFAULT 0,
    affected_tenants         INTEGER         NOT NULL DEFAULT 0,
    impact_report_json       JSONB,
    approvals_json           JSONB           NOT NULL DEFAULT '[]'::jsonb,
    cascade_json             JSONB           NOT NULL DEFAULT '[]'::jsonb,
    failure_reason           TEXT,
    created_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by               TEXT            NOT NULL,
    CONSTRAINT chk_rollback_status CHECK (status IN (
        'planned', 'approved', 'executing', 'completed', 'failed', 'cancelled'
    )),
    CONSTRAINT chk_rollback_versions CHECK (from_version <> to_version),
    CONSTRAINT chk_rollback_counts CHECK (
        affected_computations >= 0 AND affected_tenants >= 0
    )
);

CREATE INDEX IF NOT EXISTS idx_rollback_factor
    ON factors_rollback_records (factor_id);
CREATE INDEX IF NOT EXISTS idx_rollback_status
    ON factors_rollback_records (status);
CREATE INDEX IF NOT EXISTS idx_rollback_created
    ON factors_rollback_records (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rollback_factor_status
    ON factors_rollback_records (factor_id, status);

COMMENT ON TABLE factors_rollback_records IS
    'GAP-5: audit trail for per-factor rollbacks. Append-only by convention '
    '(the RollbackService only UPDATEs status + approvals; never DELETEs).';

COMMENT ON COLUMN factors_rollback_records.approvals_json IS
    'Array of {approver_id, approver_role, signature, approved_at} objects. '
    'Two-signature gate requires methodology_lead + compliance_lead roles.';
