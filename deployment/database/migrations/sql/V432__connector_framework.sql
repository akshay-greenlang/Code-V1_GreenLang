-- V432: Connector framework tables (F060)
-- Connector audit log, quota tracking, connector health.

CREATE TABLE IF NOT EXISTS factors_catalog.connector_audit_log (
    id              BIGSERIAL PRIMARY KEY,
    connector_id    VARCHAR(100) NOT NULL,
    operation       VARCHAR(50)  NOT NULL,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    tenant_id       VARCHAR(255) NOT NULL DEFAULT 'default',
    license_key_hash VARCHAR(64),
    request_factor_count  INTEGER DEFAULT 0,
    response_factor_count INTEGER DEFAULT 0,
    latency_ms      INTEGER DEFAULT 0,
    success         BOOLEAN NOT NULL DEFAULT TRUE,
    error           TEXT,
    metadata_json   JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_cal_connector_ts
    ON factors_catalog.connector_audit_log (connector_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_cal_tenant_ts
    ON factors_catalog.connector_audit_log (tenant_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_cal_failed
    ON factors_catalog.connector_audit_log (timestamp DESC)
    WHERE success = FALSE;

-- Connector quota tracking
CREATE TABLE IF NOT EXISTS factors_catalog.connector_quota_usage (
    id              BIGSERIAL PRIMARY KEY,
    connector_id    VARCHAR(100) NOT NULL,
    tenant_id       VARCHAR(255) NOT NULL DEFAULT 'default',
    period_start    DATE NOT NULL,
    period_end      DATE NOT NULL,
    requests_used   INTEGER NOT NULL DEFAULT 0,
    requests_limit  INTEGER NOT NULL DEFAULT 1000,
    factors_fetched INTEGER NOT NULL DEFAULT 0,
    last_updated    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (connector_id, tenant_id, period_start)
);

CREATE INDEX IF NOT EXISTS idx_cqu_connector_period
    ON factors_catalog.connector_quota_usage (connector_id, period_start DESC);

-- Connector health snapshots
CREATE TABLE IF NOT EXISTS factors_catalog.connector_health (
    id              BIGSERIAL PRIMARY KEY,
    connector_id    VARCHAR(100) NOT NULL,
    status          VARCHAR(20) NOT NULL,
    latency_ms      INTEGER DEFAULT 0,
    message         TEXT,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details_json    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_ch_connector_ts
    ON factors_catalog.connector_health (connector_id, checked_at DESC);

COMMENT ON TABLE factors_catalog.connector_audit_log IS
    'Immutable audit trail of all connector API calls (F060).';
COMMENT ON TABLE factors_catalog.connector_quota_usage IS
    'Per-connector, per-tenant API quota tracking (F060).';
COMMENT ON TABLE factors_catalog.connector_health IS
    'Connector health check snapshots (F060).';
