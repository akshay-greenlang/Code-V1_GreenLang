-- V444__factors_batch_jobs.sql
-- GAP-11: Batch job submission API backing store.
-- Owned by greenlang.factors.batch_jobs.PostgresBatchJobQueue.
--
-- Status lifecycle:
--   queued -> running -> (completed | failed)
--   queued | running -> cancelled
--
-- Rate limits applied at submit-time by
-- greenlang.factors.batch_jobs.max_batch_size_for_tier.

CREATE TABLE IF NOT EXISTS factors_batch_jobs (
    job_id                 UUID            PRIMARY KEY,
    tenant_id              TEXT            NOT NULL,
    job_type               TEXT            NOT NULL,
    status                 TEXT            NOT NULL,
    submitted_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    started_at             TIMESTAMPTZ,
    completed_at           TIMESTAMPTZ,
    request_count          INTEGER         NOT NULL,
    completed_count        INTEGER         NOT NULL DEFAULT 0,
    failed_count           INTEGER         NOT NULL DEFAULT 0,
    results_uri            TEXT,
    request_payload_uri    TEXT,
    error_log              JSONB           NOT NULL DEFAULT '[]'::jsonb,
    webhook_url            TEXT,
    webhook_secret_ref     TEXT,
    created_by             TEXT            NOT NULL,
    CONSTRAINT chk_batch_job_type CHECK (job_type IN (
        'resolve', 'search', 'match', 'diff'
    )),
    CONSTRAINT chk_batch_status CHECK (status IN (
        'queued', 'running', 'completed', 'failed', 'cancelled'
    )),
    CONSTRAINT chk_batch_counts CHECK (
        request_count > 0
        AND completed_count >= 0
        AND failed_count >= 0
        AND completed_count + failed_count <= request_count
    )
);

CREATE INDEX IF NOT EXISTS idx_batch_tenant_status
    ON factors_batch_jobs (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_batch_status_submitted
    ON factors_batch_jobs (status, submitted_at);
CREATE INDEX IF NOT EXISTS idx_batch_tenant_submitted
    ON factors_batch_jobs (tenant_id, submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_completed_cleanup
    ON factors_batch_jobs (completed_at)
    WHERE status IN ('completed', 'failed', 'cancelled');

COMMENT ON TABLE factors_batch_jobs IS
    'GAP-11: async batch-resolution job registry. Payload + results live '
    'in object storage via request_payload_uri + results_uri; this table '
    'tracks status + progress only.';

COMMENT ON COLUMN factors_batch_jobs.webhook_secret_ref IS
    'Logical reference (not the raw secret) to a secret stored in Vault/KMS '
    'used to HMAC-sign batch_job.completed webhook payloads.';
