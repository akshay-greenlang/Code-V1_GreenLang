-- V436: GA billing and SLA tables (Phase 12 - GA Launch)

-- Billing: usage metering
CREATE TABLE IF NOT EXISTS factors_catalog.usage_records (
    id              BIGSERIAL PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL,
    month           VARCHAR(7) NOT NULL,  -- "2026-04"
    api_calls       INTEGER DEFAULT 0,
    search_calls    INTEGER DEFAULT 0,
    match_calls     INTEGER DEFAULT 0,
    batch_calls     INTEGER DEFAULT 0,
    connector_calls INTEGER DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, month)
);

CREATE INDEX IF NOT EXISTS idx_usage_records_tenant
    ON factors_catalog.usage_records (tenant_id, month);

-- Billing: invoices
CREATE TABLE IF NOT EXISTS factors_catalog.invoices (
    invoice_id      VARCHAR(64) PRIMARY KEY,
    tenant_id       VARCHAR(64) NOT NULL,
    month           VARCHAR(7) NOT NULL,
    tier            VARCHAR(32) NOT NULL,
    base_price      NUMERIC(10,2) DEFAULT 0,
    overage_amount  NUMERIC(10,2) DEFAULT 0,
    total_amount    NUMERIC(10,2) DEFAULT 0,
    line_items      JSONB DEFAULT '[]'::jsonb,
    status          VARCHAR(16) DEFAULT 'generated',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    paid_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_invoices_tenant
    ON factors_catalog.invoices (tenant_id, month);

-- Billing: tenant plans
CREATE TABLE IF NOT EXISTS factors_catalog.tenant_plans (
    tenant_id       VARCHAR(64) PRIMARY KEY,
    tier            VARCHAR(32) NOT NULL DEFAULT 'community',
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- SLA: measurements
CREATE TABLE IF NOT EXISTS factors_catalog.sla_measurements (
    id              BIGSERIAL PRIMARY KEY,
    metric          VARCHAR(32) NOT NULL,
    value           REAL NOT NULL,
    window_minutes  INTEGER DEFAULT 5,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sla_measurements_metric
    ON factors_catalog.sla_measurements (metric, created_at);

-- SLA: violations
CREATE TABLE IF NOT EXISTS factors_catalog.sla_violations (
    id              BIGSERIAL PRIMARY KEY,
    sla_name        VARCHAR(128) NOT NULL,
    metric          VARCHAR(32) NOT NULL,
    target_value    REAL NOT NULL,
    actual_value    REAL NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sla_violations_created
    ON factors_catalog.sla_violations (created_at);
