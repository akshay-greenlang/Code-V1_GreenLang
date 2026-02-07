-- ============================================================
-- V008: Agent Factory v1.0 Schema
-- ============================================================
-- PRD: INFRA-010
-- Creates 10 tables, 2 TimescaleDB hypertables, 1 continuous
-- aggregate, retention policies, RLS, and triggers for the
-- Agent Factory lifecycle management platform.
-- ============================================================

-- Ensure infrastructure schema exists
CREATE SCHEMA IF NOT EXISTS infrastructure;

-- Agent Registry Table
CREATE TABLE IF NOT EXISTS infrastructure.agent_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_key VARCHAR(128) NOT NULL UNIQUE,
    display_name VARCHAR(256) NOT NULL,
    description TEXT,
    agent_type VARCHAR(32) NOT NULL CHECK (agent_type IN ('deterministic', 'reasoning', 'insight')),
    base_class VARCHAR(128) NOT NULL,
    version VARCHAR(32) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'created' CHECK (status IN (
        'created', 'validating', 'validated', 'deploying', 'warming_up',
        'running', 'degraded', 'draining', 'retired', 'failed', 'force_stopped'
    )),
    entry_point VARCHAR(512) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    resource_limits JSONB NOT NULL DEFAULT '{"cpu": "500m", "memory": "512Mi", "timeout_seconds": 60}',
    metadata JSONB NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',
    tenant_id UUID,
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

CREATE INDEX idx_ar_key_version ON infrastructure.agent_registry(agent_key, version);
CREATE INDEX idx_ar_status ON infrastructure.agent_registry(status);
CREATE INDEX idx_ar_type ON infrastructure.agent_registry(agent_type);
CREATE INDEX idx_ar_tenant ON infrastructure.agent_registry(tenant_id);
CREATE INDEX idx_ar_tags ON infrastructure.agent_registry USING GIN(tags);

-- Agent Versions Table (version history)
CREATE TABLE IF NOT EXISTS infrastructure.agent_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES infrastructure.agent_registry(id),
    version VARCHAR(32) NOT NULL,
    previous_version VARCHAR(32),
    changelog TEXT,
    input_schema JSONB,
    output_schema JSONB,
    pack_checksum VARCHAR(128),
    pack_url VARCHAR(1024),
    is_breaking BOOLEAN NOT NULL DEFAULT false,
    deployment_strategy VARCHAR(32) NOT NULL DEFAULT 'rolling' CHECK (
        deployment_strategy IN ('rolling', 'canary', 'blue_green', 'recreate')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(128) NOT NULL,
    UNIQUE(agent_id, version)
);

CREATE INDEX idx_av_agent ON infrastructure.agent_versions(agent_id);
CREATE INDEX idx_av_version ON infrastructure.agent_versions(version);

-- Agent Dependencies Table (dependency graph)
CREATE TABLE IF NOT EXISTS infrastructure.agent_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES infrastructure.agent_registry(id),
    depends_on_agent_id UUID NOT NULL REFERENCES infrastructure.agent_registry(id),
    version_constraint VARCHAR(64) NOT NULL DEFAULT '*',
    dependency_type VARCHAR(32) NOT NULL DEFAULT 'runtime' CHECK (
        dependency_type IN ('runtime', 'build', 'test', 'optional')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id, depends_on_agent_id, dependency_type),
    CHECK(agent_id != depends_on_agent_id)
);

CREATE INDEX idx_ad_agent ON infrastructure.agent_dependencies(agent_id);
CREATE INDEX idx_ad_depends ON infrastructure.agent_dependencies(depends_on_agent_id);

-- Agent Executions Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS infrastructure.agent_executions (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    agent_key VARCHAR(128) NOT NULL,
    version VARCHAR(32) NOT NULL,
    tenant_id UUID,
    correlation_id UUID,
    status VARCHAR(32) NOT NULL CHECK (status IN (
        'queued', 'running', 'completed', 'failed', 'timeout', 'cancelled'
    )),
    priority INTEGER NOT NULL DEFAULT 2,
    input_hash VARCHAR(128),
    output_hash VARCHAR(128),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    error_message TEXT,
    error_category VARCHAR(64),
    retry_count INTEGER NOT NULL DEFAULT 0,
    cost_compute_usd NUMERIC(10, 6) DEFAULT 0,
    cost_tokens_usd NUMERIC(10, 6) DEFAULT 0,
    cost_storage_usd NUMERIC(10, 6) DEFAULT 0,
    cost_total_usd NUMERIC(10, 6) DEFAULT 0,
    resource_cpu_ms BIGINT DEFAULT 0,
    resource_memory_mb_peak INTEGER DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable for time-series metrics
SELECT create_hypertable('infrastructure.agent_executions', 'created_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_ae_agent_time ON infrastructure.agent_executions(agent_id, created_at DESC);
CREATE INDEX idx_ae_tenant_time ON infrastructure.agent_executions(tenant_id, created_at DESC);
CREATE INDEX idx_ae_correlation ON infrastructure.agent_executions(correlation_id);
CREATE INDEX idx_ae_status ON infrastructure.agent_executions(status, created_at DESC);

-- Agent Circuit Breaker State Table (rate-based)
CREATE TABLE IF NOT EXISTS infrastructure.agent_circuit_breaker (
    agent_key VARCHAR(128) PRIMARY KEY,
    state VARCHAR(16) NOT NULL DEFAULT 'closed' CHECK (state IN ('closed', 'open', 'half_open')),
    failure_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    total_calls_in_window INTEGER NOT NULL DEFAULT 0,
    error_rate_pct NUMERIC(5, 2) NOT NULL DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    half_opened_at TIMESTAMPTZ,
    config JSONB NOT NULL DEFAULT '{
        "window_seconds": 60,
        "error_rate_threshold_pct": 50,
        "slow_call_p99_ms_threshold": 5000,
        "slow_call_rate_threshold_pct": 80,
        "wait_in_open_seconds": 60,
        "half_open_test_requests": 3,
        "minimum_calls_in_window": 5
    }',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent Cost Budgets Table
CREATE TABLE IF NOT EXISTS infrastructure.agent_cost_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES infrastructure.agent_registry(id),
    tenant_id UUID,
    budget_period VARCHAR(16) NOT NULL DEFAULT 'daily' CHECK (budget_period IN ('hourly', 'daily', 'weekly', 'monthly')),
    budget_amount_usd NUMERIC(10, 2) NOT NULL,
    spent_amount_usd NUMERIC(10, 6) NOT NULL DEFAULT 0,
    alert_threshold_pct INTEGER NOT NULL DEFAULT 80,
    is_hard_limit BOOLEAN NOT NULL DEFAULT false,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_acb_agent ON infrastructure.agent_cost_budgets(agent_id);
CREATE INDEX idx_acb_tenant ON infrastructure.agent_cost_budgets(tenant_id);
CREATE INDEX idx_acb_period ON infrastructure.agent_cost_budgets(period_start, period_end);

-- Agent Audit Log Table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS infrastructure.agent_audit_log (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    agent_id UUID,
    agent_key VARCHAR(128),
    action VARCHAR(64) NOT NULL,
    actor VARCHAR(128) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    previous_state JSONB,
    new_state JSONB,
    ip_address INET,
    tenant_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('infrastructure.agent_audit_log', 'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_aal_agent ON infrastructure.agent_audit_log(agent_id, created_at DESC);
CREATE INDEX idx_aal_action ON infrastructure.agent_audit_log(action, created_at DESC);
CREATE INDEX idx_aal_actor ON infrastructure.agent_audit_log(actor, created_at DESC);

-- Agent Config Store Table
CREATE TABLE IF NOT EXISTS infrastructure.agent_config_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_key VARCHAR(128) NOT NULL,
    config_key VARCHAR(256) NOT NULL,
    config_value JSONB NOT NULL,
    config_version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT true,
    schema_hash VARCHAR(128),
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(agent_key, config_key, config_version)
);

CREATE INDEX idx_acs_agent_key ON infrastructure.agent_config_store(agent_key);
CREATE INDEX idx_acs_active ON infrastructure.agent_config_store(agent_key, is_active) WHERE is_active = true;

-- Agent Operations Table (async long-running operations)
CREATE TABLE IF NOT EXISTS infrastructure.agent_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_type VARCHAR(32) NOT NULL CHECK (operation_type IN ('deploy', 'rollback', 'pack', 'publish', 'migrate')),
    agent_key VARCHAR(128),
    idempotency_key VARCHAR(256) NOT NULL UNIQUE,
    status VARCHAR(32) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress_pct INTEGER NOT NULL DEFAULT 0 CHECK (progress_pct >= 0 AND progress_pct <= 100),
    input_params JSONB NOT NULL DEFAULT '{}',
    result JSONB,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    tenant_id UUID,
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_aop_status ON infrastructure.agent_operations(status, created_at DESC);
CREATE INDEX idx_aop_agent ON infrastructure.agent_operations(agent_key);
CREATE INDEX idx_aop_idempotency ON infrastructure.agent_operations(idempotency_key);
CREATE INDEX idx_aop_tenant ON infrastructure.agent_operations(tenant_id);

-- Agent Tenant Configuration Table (per-tenant overrides)
CREATE TABLE IF NOT EXISTS infrastructure.agent_tenant_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    agent_key VARCHAR(128) NOT NULL,
    pinned_version VARCHAR(32),
    execution_mode VARCHAR(16) CHECK (execution_mode IN ('pool', 'dedicated')),
    max_concurrent INTEGER,
    max_daily_executions INTEGER,
    budget_override_usd NUMERIC(10, 2),
    config_overrides JSONB NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, agent_key)
);

CREATE INDEX idx_atc_tenant ON infrastructure.agent_tenant_config(tenant_id);
CREATE INDEX idx_atc_agent ON infrastructure.agent_tenant_config(agent_key);

-- Agent Execution Metrics - Continuous Aggregate (5-minute buckets)
CREATE MATERIALIZED VIEW IF NOT EXISTS infrastructure.agent_metrics_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', created_at) AS bucket,
    agent_key,
    COUNT(*) AS execution_count,
    COUNT(*) FILTER (WHERE status = 'completed') AS success_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failure_count,
    COUNT(*) FILTER (WHERE status = 'timeout') AS timeout_count,
    AVG(duration_ms) AS avg_duration_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) AS p50_duration_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99_duration_ms,
    SUM(cost_total_usd) AS total_cost_usd,
    AVG(resource_cpu_ms) AS avg_cpu_ms,
    MAX(resource_memory_mb_peak) AS max_memory_mb
FROM infrastructure.agent_executions
GROUP BY bucket, agent_key
WITH NO DATA;

SELECT add_continuous_aggregate_policy('infrastructure.agent_metrics_5m',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- Retention policies
SELECT add_retention_policy('infrastructure.agent_executions', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('infrastructure.agent_audit_log', INTERVAL '365 days', if_not_exists => TRUE);

-- Row-Level Security
ALTER TABLE infrastructure.agent_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE infrastructure.agent_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE infrastructure.agent_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE infrastructure.agent_cost_budgets ENABLE ROW LEVEL SECURITY;

-- Tenant isolation policies
CREATE POLICY agent_registry_tenant_isolation ON infrastructure.agent_registry
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY agent_executions_tenant_isolation ON infrastructure.agent_executions
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY agent_audit_tenant_isolation ON infrastructure.agent_audit_log
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

CREATE POLICY agent_budgets_tenant_isolation ON infrastructure.agent_cost_budgets
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

ALTER TABLE infrastructure.agent_operations ENABLE ROW LEVEL SECURITY;
CREATE POLICY agent_operations_tenant_isolation ON infrastructure.agent_operations
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID OR tenant_id IS NULL);

ALTER TABLE infrastructure.agent_tenant_config ENABLE ROW LEVEL SECURITY;
CREATE POLICY agent_tenant_config_isolation ON infrastructure.agent_tenant_config
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Updated-at trigger
CREATE OR REPLACE FUNCTION infrastructure.update_agent_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_agent_registry_updated
    BEFORE UPDATE ON infrastructure.agent_registry
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();

CREATE TRIGGER trg_agent_budgets_updated
    BEFORE UPDATE ON infrastructure.agent_cost_budgets
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();

CREATE TRIGGER trg_agent_config_updated
    BEFORE UPDATE ON infrastructure.agent_config_store
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();

CREATE TRIGGER trg_agent_operations_updated
    BEFORE UPDATE ON infrastructure.agent_operations
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();

CREATE TRIGGER trg_agent_tenant_config_updated
    BEFORE UPDATE ON infrastructure.agent_tenant_config
    FOR EACH ROW EXECUTE FUNCTION infrastructure.update_agent_timestamp();
