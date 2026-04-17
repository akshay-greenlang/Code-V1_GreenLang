-- V435: Factors pilot program tables (Phase 11 - Design Partners)

-- Pilot partner registry
CREATE TABLE IF NOT EXISTS factors_catalog.pilot_partners (
    partner_id      VARCHAR(64) PRIMARY KEY,
    name            VARCHAR(256) NOT NULL,
    contact_email   VARCHAR(256) NOT NULL,
    organization    VARCHAR(256) NOT NULL,
    tier            VARCHAR(32) DEFAULT 'pro',
    status          VARCHAR(32) DEFAULT 'invited',
    tenant_id       VARCHAR(64) UNIQUE NOT NULL,
    api_key_hash    VARCHAR(128),
    enrolled_at     TIMESTAMPTZ DEFAULT NOW(),
    activated_at    TIMESTAMPTZ,
    target_use_cases JSONB DEFAULT '[]'::jsonb,
    max_api_calls_per_day INTEGER DEFAULT 10000,
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_pilot_partners_status
    ON factors_catalog.pilot_partners (status);
CREATE INDEX IF NOT EXISTS idx_pilot_partners_tenant
    ON factors_catalog.pilot_partners (tenant_id);

-- Pilot usage telemetry
CREATE TABLE IF NOT EXISTS factors_catalog.pilot_telemetry (
    id              BIGSERIAL PRIMARY KEY,
    event_id        VARCHAR(64) NOT NULL,
    partner_id      VARCHAR(64) NOT NULL REFERENCES factors_catalog.pilot_partners(partner_id),
    tenant_id       VARCHAR(64) NOT NULL,
    event_type      VARCHAR(32) NOT NULL,
    endpoint        VARCHAR(256) NOT NULL,
    method          VARCHAR(8) DEFAULT 'GET',
    status_code     INTEGER DEFAULT 200,
    latency_ms      REAL DEFAULT 0,
    result_count    INTEGER DEFAULT 0,
    query_params    JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pilot_telemetry_partner
    ON factors_catalog.pilot_telemetry (partner_id, created_at);
CREATE INDEX IF NOT EXISTS idx_pilot_telemetry_event_type
    ON factors_catalog.pilot_telemetry (event_type);

-- Pilot feedback
CREATE TABLE IF NOT EXISTS factors_catalog.pilot_feedback (
    feedback_id     VARCHAR(64) PRIMARY KEY,
    partner_id      VARCHAR(64) NOT NULL REFERENCES factors_catalog.pilot_partners(partner_id),
    category        VARCHAR(32) NOT NULL,
    priority        VARCHAR(16) DEFAULT 'medium',
    status          VARCHAR(16) DEFAULT 'new',
    title           VARCHAR(512) NOT NULL,
    description     TEXT DEFAULT '',
    factor_ids      JSONB DEFAULT '[]'::jsonb,
    tags            JSONB DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,
    resolution_notes TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pilot_feedback_partner
    ON factors_catalog.pilot_feedback (partner_id);
CREATE INDEX IF NOT EXISTS idx_pilot_feedback_status
    ON factors_catalog.pilot_feedback (status);
CREATE INDEX IF NOT EXISTS idx_pilot_feedback_category
    ON factors_catalog.pilot_feedback (category);
