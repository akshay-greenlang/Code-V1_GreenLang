-- V431: Factors notification audit table
-- Tracks all notifications sent by the factors watch & release pipeline.

CREATE TABLE IF NOT EXISTS factors_catalog.notification_audit (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type      VARCHAR(50) NOT NULL,
    channel         VARCHAR(20) NOT NULL,
    subject         VARCHAR(500) NOT NULL,
    recipient       VARCHAR(500),
    success         BOOLEAN NOT NULL DEFAULT FALSE,
    error_message   TEXT,
    source_id       VARCHAR(255),
    edition_id      VARCHAR(100),
    metadata_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
    retry_count     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_notif_audit_event_ts
    ON factors_catalog.notification_audit (event_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_notif_audit_source
    ON factors_catalog.notification_audit (source_id, timestamp DESC)
    WHERE source_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_notif_audit_edition
    ON factors_catalog.notification_audit (edition_id, timestamp DESC)
    WHERE edition_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_notif_audit_failed
    ON factors_catalog.notification_audit (timestamp DESC)
    WHERE success = FALSE;

COMMENT ON TABLE factors_catalog.notification_audit IS
    'Immutable audit trail of all factor pipeline notifications (Slack, email).';
