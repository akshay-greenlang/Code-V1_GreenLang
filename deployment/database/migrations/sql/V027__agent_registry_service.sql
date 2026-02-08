-- =============================================================================
-- GreenLang Climate OS - Agent Registry & Service Catalog Schema
-- =============================================================================
-- Migration: V027
-- Component: AGENT-FOUND-007 Agent Registry & Service Catalog
-- Description: Creates agent_registry_service schema with agents, agent_versions,
--              agent_capabilities, agent_dependencies, health_checks (hypertable),
--              registry_audit_log (hypertable), agent_variants, service_catalog,
--              continuous aggregates for hourly health summary and daily registry
--              events, 30+ indexes (including GIN indexes on JSONB and arrays),
--              RLS policies per tenant, 14 security permissions, retention policies
--              (30-day health_checks, 365-day audit_log), compression, and seed
--              data registering the 7 foundation agents (GL-FOUND-X-001 through
--              GL-FOUND-X-007) with capabilities.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS agent_registry_service;

-- =============================================================================
-- Table: agent_registry_service.agents
-- =============================================================================
-- Top-level agent metadata registry. Each agent has a unique ID, name,
-- description, execution layer (1 = foundation, 2 = domain, 3 = orchestrator),
-- execution mode (sync, async, streaming), idempotency and determinism flags,
-- max concurrent runs, GLIP version, checkpointing support, authorship,
-- documentation URL, enable/disable toggle, tenant scoping, and timestamps.

CREATE TABLE agent_registry_service.agents (
    agent_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    layer INTEGER NOT NULL DEFAULT 1,
    execution_mode VARCHAR(30) NOT NULL DEFAULT 'async',
    idempotency_support BOOLEAN NOT NULL DEFAULT FALSE,
    deterministic BOOLEAN NOT NULL DEFAULT FALSE,
    max_concurrent_runs INTEGER NOT NULL DEFAULT 5,
    glip_version VARCHAR(30) NOT NULL DEFAULT '1.0.0',
    supports_checkpointing BOOLEAN NOT NULL DEFAULT FALSE,
    author VARCHAR(255) DEFAULT 'GreenLang Platform Team',
    documentation_url TEXT DEFAULT '',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Layer constraint (1 = foundation, 2 = domain, 3 = orchestrator)
ALTER TABLE agent_registry_service.agents
    ADD CONSTRAINT chk_agent_layer
    CHECK (layer IN (1, 2, 3));

-- Execution mode constraint
ALTER TABLE agent_registry_service.agents
    ADD CONSTRAINT chk_agent_execution_mode
    CHECK (execution_mode IN ('sync', 'async', 'streaming', 'batch'));

-- Max concurrent runs must be positive
ALTER TABLE agent_registry_service.agents
    ADD CONSTRAINT chk_agent_max_concurrent_positive
    CHECK (max_concurrent_runs > 0);

-- =============================================================================
-- Table: agent_registry_service.agent_versions
-- =============================================================================
-- Version records for each agent. Each version captures resource profile (JSONB
-- specifying CPU, memory, GPU requirements), container specification (image,
-- tag, command, args), legacy HTTP configuration for non-GLIP agents, tags
-- for categorization, sectors for industry vertical targeting, provenance hash
-- for integrity verification, and creation timestamp.

CREATE TABLE agent_registry_service.agent_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL REFERENCES agent_registry_service.agents(agent_id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL,
    resource_profile JSONB DEFAULT '{}'::jsonb,
    container_spec JSONB DEFAULT '{}'::jsonb,
    legacy_http_config JSONB DEFAULT '{}'::jsonb,
    tags TEXT[] DEFAULT '{}',
    sectors TEXT[] DEFAULT '{}',
    provenance_hash CHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique constraint on agent_id + version
ALTER TABLE agent_registry_service.agent_versions
    ADD CONSTRAINT uq_agent_version UNIQUE (agent_id, version);

-- Provenance hash must be 64-character hex when present
ALTER TABLE agent_registry_service.agent_versions
    ADD CONSTRAINT chk_version_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: agent_registry_service.agent_capabilities
-- =============================================================================
-- Capability definitions per agent per version. Each capability has a name,
-- category (computation, transformation, validation, integration, reporting),
-- description, accepted input types, produced output types, and a parameters
-- JSONB for fine-grained capability configuration and constraints.

CREATE TABLE agent_registry_service.agent_capabilities (
    capability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL REFERENCES agent_registry_service.agents(agent_id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL DEFAULT 'computation',
    description TEXT DEFAULT '',
    input_types TEXT[] DEFAULT '{}',
    output_types TEXT[] DEFAULT '{}',
    parameters JSONB DEFAULT '{}'::jsonb
);

-- Category constraint
ALTER TABLE agent_registry_service.agent_capabilities
    ADD CONSTRAINT chk_capability_category
    CHECK (category IN (
        'computation', 'transformation', 'validation', 'integration',
        'reporting', 'orchestration', 'normalization', 'analysis',
        'security', 'registry'
    ));

-- =============================================================================
-- Table: agent_registry_service.agent_dependencies
-- =============================================================================
-- Dependency graph between agents. Each dependency specifies which agent
-- depends on which other agent, with a version constraint (semver range),
-- whether the dependency is optional, and a human-readable reason. Used
-- by the orchestrator for DAG construction and by the registry for
-- dependency resolution and cycle detection.

CREATE TABLE agent_registry_service.agent_dependencies (
    dependency_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL REFERENCES agent_registry_service.agents(agent_id) ON DELETE CASCADE,
    depends_on_agent_id VARCHAR(100) NOT NULL REFERENCES agent_registry_service.agents(agent_id) ON DELETE CASCADE,
    version_constraint VARCHAR(100) DEFAULT '*',
    optional BOOLEAN NOT NULL DEFAULT FALSE,
    reason TEXT DEFAULT ''
);

-- Unique constraint on agent_id + depends_on_agent_id
ALTER TABLE agent_registry_service.agent_dependencies
    ADD CONSTRAINT uq_agent_dependency UNIQUE (agent_id, depends_on_agent_id);

-- Agent cannot depend on itself
ALTER TABLE agent_registry_service.agent_dependencies
    ADD CONSTRAINT chk_no_self_dependency
    CHECK (agent_id <> depends_on_agent_id);

-- =============================================================================
-- Table: agent_registry_service.health_checks
-- =============================================================================
-- TimescaleDB hypertable recording agent health check results over time.
-- Each row captures the agent, version, health status, response latency,
-- error message (if any), probe type (liveness, readiness, startup), and
-- the check timestamp. Partitioned by checked_at for time-series queries.
-- Retained for 30 days with compression after 3 days.

CREATE TABLE agent_registry_service.health_checks (
    check_id UUID DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL DEFAULT 'latest',
    status VARCHAR(20) NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    error_message TEXT DEFAULT '',
    probe_type VARCHAR(20) NOT NULL DEFAULT 'liveness',
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (check_id, checked_at)
);

-- Create hypertable partitioned by checked_at
SELECT create_hypertable('agent_registry_service.health_checks', 'checked_at', if_not_exists => TRUE);

-- Status constraint
ALTER TABLE agent_registry_service.health_checks
    ADD CONSTRAINT chk_health_status
    CHECK (status IN ('healthy', 'unhealthy', 'degraded', 'unknown', 'starting'));

-- Probe type constraint
ALTER TABLE agent_registry_service.health_checks
    ADD CONSTRAINT chk_health_probe_type
    CHECK (probe_type IN ('liveness', 'readiness', 'startup', 'deep'));

-- Latency must be non-negative
ALTER TABLE agent_registry_service.health_checks
    ADD CONSTRAINT chk_health_latency_positive
    CHECK (latency_ms >= 0);

-- =============================================================================
-- Table: agent_registry_service.registry_audit_log
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all
-- registry operations. Each event captures the agent, version, event type,
-- change type, old/new data (JSONB), user, provenance hash, tenant, and
-- timestamp. Partitioned by occurred_at for time-series queries.
-- Retained for 365 days with compression after 30 days.

CREATE TABLE agent_registry_service.registry_audit_log (
    event_id UUID DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100),
    version VARCHAR(50),
    event_type VARCHAR(50) NOT NULL,
    change_type VARCHAR(30) NOT NULL,
    old_data JSONB DEFAULT '{}'::jsonb,
    new_data JSONB DEFAULT '{}'::jsonb,
    user_id VARCHAR(255),
    provenance_hash CHAR(64),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event_id, occurred_at)
);

-- Create hypertable partitioned by occurred_at
SELECT create_hypertable('agent_registry_service.registry_audit_log', 'occurred_at', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE agent_registry_service.registry_audit_log
    ADD CONSTRAINT chk_audit_event_type
    CHECK (event_type IN (
        'agent_registered', 'agent_updated', 'agent_deregistered',
        'agent_enabled', 'agent_disabled',
        'version_published', 'version_deprecated', 'version_deleted',
        'capability_added', 'capability_updated', 'capability_removed',
        'dependency_added', 'dependency_removed', 'dependency_updated',
        'health_check_failed', 'health_check_recovered',
        'variant_created', 'variant_updated', 'variant_deleted',
        'catalog_published', 'catalog_unpublished', 'catalog_updated',
        'hot_reload', 'cache_invalidated',
        'admin_action', 'bulk_import', 'bulk_export'
    ));

-- Change type constraint
ALTER TABLE agent_registry_service.registry_audit_log
    ADD CONSTRAINT chk_audit_change_type
    CHECK (change_type IN (
        'create', 'update', 'delete', 'enable', 'disable',
        'publish', 'unpublish', 'deprecate', 'system', 'admin'
    ));

-- Provenance hash must be 64-character hex when present
ALTER TABLE agent_registry_service.registry_audit_log
    ADD CONSTRAINT chk_audit_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: agent_registry_service.agent_variants
-- =============================================================================
-- Variant records for agents. Variants represent alternative configurations
-- of an agent for different contexts -- for example, a "sector:energy" variant
-- optimized for energy industry calculations, or a "region:eu" variant
-- configured for EU regulatory requirements.

CREATE TABLE agent_registry_service.agent_variants (
    variant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL REFERENCES agent_registry_service.agents(agent_id) ON DELETE CASCADE,
    variant_type VARCHAR(50) NOT NULL,
    variant_value VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Variant type constraint
ALTER TABLE agent_registry_service.agent_variants
    ADD CONSTRAINT chk_variant_type
    CHECK (variant_type IN (
        'sector', 'region', 'regulation', 'framework',
        'performance', 'accuracy', 'custom'
    ));

-- Unique constraint on agent_id + variant_type + variant_value + tenant_id
ALTER TABLE agent_registry_service.agent_variants
    ADD CONSTRAINT uq_agent_variant UNIQUE (agent_id, variant_type, variant_value, tenant_id);

-- =============================================================================
-- Table: agent_registry_service.service_catalog
-- =============================================================================
-- Published service catalog entries. The catalog is the public-facing
-- directory of available agents. Each entry has a display name, summary,
-- category, publication status (active, deprecated, coming_soon, retired),
-- and publication timestamp. Catalog entries are linked to agent
-- registrations and may have different visibility than the underlying
-- agent registration.

CREATE TABLE agent_registry_service.service_catalog (
    catalog_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL REFERENCES agent_registry_service.agents(agent_id) ON DELETE CASCADE,
    display_name VARCHAR(255) NOT NULL,
    summary TEXT DEFAULT '',
    category VARCHAR(50) NOT NULL DEFAULT 'foundation',
    status VARCHAR(30) NOT NULL DEFAULT 'active',
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique constraint on agent_id + tenant_id
ALTER TABLE agent_registry_service.service_catalog
    ADD CONSTRAINT uq_catalog_agent_tenant UNIQUE (agent_id, tenant_id);

-- Status constraint
ALTER TABLE agent_registry_service.service_catalog
    ADD CONSTRAINT chk_catalog_status
    CHECK (status IN ('active', 'deprecated', 'coming_soon', 'retired', 'beta'));

-- Category constraint
ALTER TABLE agent_registry_service.service_catalog
    ADD CONSTRAINT chk_catalog_category
    CHECK (category IN (
        'foundation', 'domain', 'orchestration', 'security',
        'integration', 'reporting', 'utility'
    ));

-- =============================================================================
-- Continuous Aggregate: agent_registry_service.hourly_health_summary
-- =============================================================================
-- Precomputed hourly health check summary by agent and status for
-- dashboard queries, trend analysis, and SLI tracking. Shows the
-- number of checks per status, average/max latency, and unique
-- agents checked per hour.

CREATE MATERIALIZED VIEW agent_registry_service.hourly_health_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', checked_at) AS bucket,
    agent_id,
    status,
    tenant_id,
    COUNT(*) AS total_checks,
    AVG(latency_ms) AS avg_latency_ms,
    MAX(latency_ms) AS max_latency_ms,
    COUNT(DISTINCT version) AS versions_checked
FROM agent_registry_service.health_checks
WHERE checked_at IS NOT NULL
GROUP BY bucket, agent_id, status, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('agent_registry_service.hourly_health_summary',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: agent_registry_service.daily_registry_events
-- =============================================================================
-- Precomputed daily counts of registry audit events by event type for
-- compliance reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW agent_registry_service.daily_registry_events
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', occurred_at) AS bucket,
    event_type,
    change_type,
    tenant_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT agent_id) AS unique_agents,
    COUNT(DISTINCT user_id) AS unique_users
FROM agent_registry_service.registry_audit_log
WHERE occurred_at IS NOT NULL
GROUP BY bucket, event_type, change_type, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('agent_registry_service.daily_registry_events',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- agents indexes
CREATE INDEX idx_agents_name ON agent_registry_service.agents(name);
CREATE INDEX idx_agents_layer ON agent_registry_service.agents(layer);
CREATE INDEX idx_agents_execution_mode ON agent_registry_service.agents(execution_mode);
CREATE INDEX idx_agents_enabled ON agent_registry_service.agents(enabled, tenant_id);
CREATE INDEX idx_agents_tenant ON agent_registry_service.agents(tenant_id);
CREATE INDEX idx_agents_glip_version ON agent_registry_service.agents(glip_version);
CREATE INDEX idx_agents_author ON agent_registry_service.agents(author);
CREATE INDEX idx_agents_created_at ON agent_registry_service.agents(created_at DESC);
CREATE INDEX idx_agents_updated_at ON agent_registry_service.agents(updated_at DESC);

-- agent_versions indexes
CREATE INDEX idx_versions_agent ON agent_registry_service.agent_versions(agent_id);
CREATE INDEX idx_versions_version ON agent_registry_service.agent_versions(version);
CREATE INDEX idx_versions_created_at ON agent_registry_service.agent_versions(created_at DESC);
CREATE INDEX idx_versions_provenance ON agent_registry_service.agent_versions(provenance_hash);
CREATE INDEX idx_versions_resource_profile ON agent_registry_service.agent_versions USING GIN (resource_profile);
CREATE INDEX idx_versions_container_spec ON agent_registry_service.agent_versions USING GIN (container_spec);
CREATE INDEX idx_versions_legacy_http ON agent_registry_service.agent_versions USING GIN (legacy_http_config);
CREATE INDEX idx_versions_tags ON agent_registry_service.agent_versions USING GIN (tags);
CREATE INDEX idx_versions_sectors ON agent_registry_service.agent_versions USING GIN (sectors);

-- agent_capabilities indexes
CREATE INDEX idx_capabilities_agent ON agent_registry_service.agent_capabilities(agent_id);
CREATE INDEX idx_capabilities_version ON agent_registry_service.agent_capabilities(version);
CREATE INDEX idx_capabilities_name ON agent_registry_service.agent_capabilities(name);
CREATE INDEX idx_capabilities_category ON agent_registry_service.agent_capabilities(category);
CREATE INDEX idx_capabilities_input_types ON agent_registry_service.agent_capabilities USING GIN (input_types);
CREATE INDEX idx_capabilities_output_types ON agent_registry_service.agent_capabilities USING GIN (output_types);
CREATE INDEX idx_capabilities_parameters ON agent_registry_service.agent_capabilities USING GIN (parameters);

-- agent_dependencies indexes
CREATE INDEX idx_dependencies_agent ON agent_registry_service.agent_dependencies(agent_id);
CREATE INDEX idx_dependencies_depends_on ON agent_registry_service.agent_dependencies(depends_on_agent_id);
CREATE INDEX idx_dependencies_optional ON agent_registry_service.agent_dependencies(optional);

-- health_checks indexes (hypertable-aware)
CREATE INDEX idx_health_agent ON agent_registry_service.health_checks(agent_id, checked_at DESC);
CREATE INDEX idx_health_version ON agent_registry_service.health_checks(version, checked_at DESC);
CREATE INDEX idx_health_status ON agent_registry_service.health_checks(status, checked_at DESC);
CREATE INDEX idx_health_probe_type ON agent_registry_service.health_checks(probe_type, checked_at DESC);
CREATE INDEX idx_health_tenant ON agent_registry_service.health_checks(tenant_id, checked_at DESC);
CREATE INDEX idx_health_latency ON agent_registry_service.health_checks(latency_ms, checked_at DESC);

-- registry_audit_log indexes (hypertable-aware)
CREATE INDEX idx_audit_agent ON agent_registry_service.registry_audit_log(agent_id, occurred_at DESC);
CREATE INDEX idx_audit_version ON agent_registry_service.registry_audit_log(version, occurred_at DESC);
CREATE INDEX idx_audit_event_type ON agent_registry_service.registry_audit_log(event_type, occurred_at DESC);
CREATE INDEX idx_audit_change_type ON agent_registry_service.registry_audit_log(change_type, occurred_at DESC);
CREATE INDEX idx_audit_user ON agent_registry_service.registry_audit_log(user_id, occurred_at DESC);
CREATE INDEX idx_audit_tenant ON agent_registry_service.registry_audit_log(tenant_id, occurred_at DESC);
CREATE INDEX idx_audit_provenance ON agent_registry_service.registry_audit_log(provenance_hash);
CREATE INDEX idx_audit_old_data ON agent_registry_service.registry_audit_log USING GIN (old_data);
CREATE INDEX idx_audit_new_data ON agent_registry_service.registry_audit_log USING GIN (new_data);

-- agent_variants indexes
CREATE INDEX idx_variants_agent ON agent_registry_service.agent_variants(agent_id);
CREATE INDEX idx_variants_type ON agent_registry_service.agent_variants(variant_type);
CREATE INDEX idx_variants_value ON agent_registry_service.agent_variants(variant_value);
CREATE INDEX idx_variants_tenant ON agent_registry_service.agent_variants(tenant_id);
CREATE INDEX idx_variants_created_at ON agent_registry_service.agent_variants(created_at DESC);

-- service_catalog indexes
CREATE INDEX idx_catalog_agent ON agent_registry_service.service_catalog(agent_id);
CREATE INDEX idx_catalog_display_name ON agent_registry_service.service_catalog(display_name);
CREATE INDEX idx_catalog_category ON agent_registry_service.service_catalog(category);
CREATE INDEX idx_catalog_status ON agent_registry_service.service_catalog(status);
CREATE INDEX idx_catalog_tenant ON agent_registry_service.service_catalog(tenant_id);
CREATE INDEX idx_catalog_published_at ON agent_registry_service.service_catalog(published_at DESC);
CREATE INDEX idx_catalog_updated_at ON agent_registry_service.service_catalog(updated_at DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE agent_registry_service.agents ENABLE ROW LEVEL SECURITY;
CREATE POLICY agents_tenant_read ON agent_registry_service.agents
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY agents_tenant_write ON agent_registry_service.agents
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE agent_registry_service.agent_versions ENABLE ROW LEVEL SECURITY;
-- Versions inherit tenant context via their parent agent
CREATE POLICY versions_read ON agent_registry_service.agent_versions
    FOR SELECT USING (true);
CREATE POLICY versions_write ON agent_registry_service.agent_versions
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
        OR EXISTS (
            SELECT 1 FROM agent_registry_service.agents a
            WHERE a.agent_id = agent_versions.agent_id
            AND (
                a.tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.current_tenant', true) IS NULL
            )
        )
    );

ALTER TABLE agent_registry_service.agent_capabilities ENABLE ROW LEVEL SECURITY;
-- Capabilities inherit tenant context via their parent agent
CREATE POLICY capabilities_read ON agent_registry_service.agent_capabilities
    FOR SELECT USING (true);
CREATE POLICY capabilities_write ON agent_registry_service.agent_capabilities
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
        OR EXISTS (
            SELECT 1 FROM agent_registry_service.agents a
            WHERE a.agent_id = agent_capabilities.agent_id
            AND (
                a.tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.current_tenant', true) IS NULL
            )
        )
    );

ALTER TABLE agent_registry_service.agent_dependencies ENABLE ROW LEVEL SECURITY;
-- Dependencies inherit tenant context via their parent agent
CREATE POLICY dependencies_read ON agent_registry_service.agent_dependencies
    FOR SELECT USING (true);
CREATE POLICY dependencies_write ON agent_registry_service.agent_dependencies
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
        OR EXISTS (
            SELECT 1 FROM agent_registry_service.agents a
            WHERE a.agent_id = agent_dependencies.agent_id
            AND (
                a.tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.current_tenant', true) IS NULL
            )
        )
    );

ALTER TABLE agent_registry_service.health_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY health_tenant_read ON agent_registry_service.health_checks
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY health_tenant_write ON agent_registry_service.health_checks
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE agent_registry_service.registry_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY audit_tenant_read ON agent_registry_service.registry_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY audit_tenant_write ON agent_registry_service.registry_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE agent_registry_service.agent_variants ENABLE ROW LEVEL SECURITY;
CREATE POLICY variants_tenant_read ON agent_registry_service.agent_variants
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY variants_tenant_write ON agent_registry_service.agent_variants
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE agent_registry_service.service_catalog ENABLE ROW LEVEL SECURITY;
CREATE POLICY catalog_tenant_read ON agent_registry_service.service_catalog
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY catalog_tenant_write ON agent_registry_service.service_catalog
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA agent_registry_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA agent_registry_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA agent_registry_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON agent_registry_service.hourly_health_summary TO greenlang_app;
GRANT SELECT ON agent_registry_service.daily_registry_events TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA agent_registry_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA agent_registry_service TO greenlang_readonly;
GRANT SELECT ON agent_registry_service.hourly_health_summary TO greenlang_readonly;
GRANT SELECT ON agent_registry_service.daily_registry_events TO greenlang_readonly;

-- Add agent registry service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'agent_registry:agents:read', 'agent_registry', 'agents_read', 'View registered agents and their metadata'),
    (gen_random_uuid(), 'agent_registry:agents:write', 'agent_registry', 'agents_write', 'Register, update, and deregister agents'),
    (gen_random_uuid(), 'agent_registry:agents:delete', 'agent_registry', 'agents_delete', 'Delete agent registrations'),
    (gen_random_uuid(), 'agent_registry:versions:read', 'agent_registry', 'versions_read', 'View agent version records'),
    (gen_random_uuid(), 'agent_registry:versions:write', 'agent_registry', 'versions_write', 'Publish and deprecate agent versions'),
    (gen_random_uuid(), 'agent_registry:capabilities:read', 'agent_registry', 'capabilities_read', 'View agent capability definitions'),
    (gen_random_uuid(), 'agent_registry:capabilities:write', 'agent_registry', 'capabilities_write', 'Manage agent capability definitions'),
    (gen_random_uuid(), 'agent_registry:dependencies:read', 'agent_registry', 'dependencies_read', 'View agent dependency graph'),
    (gen_random_uuid(), 'agent_registry:dependencies:write', 'agent_registry', 'dependencies_write', 'Manage agent dependencies'),
    (gen_random_uuid(), 'agent_registry:health:read', 'agent_registry', 'health_read', 'View agent health check results'),
    (gen_random_uuid(), 'agent_registry:audit:read', 'agent_registry', 'audit_read', 'View registry audit event log'),
    (gen_random_uuid(), 'agent_registry:catalog:read', 'agent_registry', 'catalog_read', 'View service catalog entries'),
    (gen_random_uuid(), 'agent_registry:catalog:write', 'agent_registry', 'catalog_write', 'Publish and manage service catalog entries'),
    (gen_random_uuid(), 'agent_registry:admin', 'agent_registry', 'admin', 'Agent Registry service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep health checks for 30 days
SELECT add_retention_policy('agent_registry_service.health_checks', INTERVAL '30 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('agent_registry_service.registry_audit_log', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on health_checks after 3 days
ALTER TABLE agent_registry_service.health_checks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id,agent_id',
    timescaledb.compress_orderby = 'checked_at DESC'
);

SELECT add_compression_policy('agent_registry_service.health_checks', INTERVAL '3 days');

-- Enable compression on registry_audit_log after 30 days
ALTER TABLE agent_registry_service.registry_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'occurred_at DESC'
);

SELECT add_compression_policy('agent_registry_service.registry_audit_log', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the 7 Foundation Agents (GL-FOUND-X-001 through GL-FOUND-X-007)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES

-- GL-FOUND-X-001: GreenLang Orchestrator (DAG)
('GL-FOUND-X-001', 'GreenLang Orchestrator',
 'DAG execution engine providing topological sort, level-based parallel execution, per-node retry/timeout, deterministic scheduling, DAG-aware checkpointing, and provenance chain (SHA-256). Coordinates all agent pipelines across the GreenLang Climate OS.',
 3, 'async', true, true, 10, '1.0.0', true,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/orchestrator', true, 'default'),

-- GL-FOUND-X-002: Schema Compiler & Validator
('GL-FOUND-X-002', 'Schema Compiler & Validator',
 'JSON Schema Draft 2020-12 validation with 7-phase pipeline, AST-to-IR compilation, ReDoS prevention, RFC 6902 fix suggestions, Git-backed registry, IR caching, and Layer 1/Layer 2 delegation. Ensures all data conforms to GreenLang schemas.',
 1, 'sync', true, true, 20, '1.0.0', false,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/schema-compiler', true, 'default'),

-- GL-FOUND-X-003: Unit & Reference Normalizer
('GL-FOUND-X-003', 'Unit & Reference Normalizer',
 'Decimal-precision conversion across 8 dimensions, IPCC AR5/AR6 GWP factors, entity resolution (67 fuels, 55 materials, 28 processes), 3-tier matching (exact/alias/fuzzy), dimensional analysis, and SHA-256 provenance chains.',
 1, 'sync', true, true, 20, '1.0.0', false,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/normalizer', true, 'default'),

-- GL-FOUND-X-004: Assumptions Registry
('GL-FOUND-X-004', 'Assumptions Registry',
 'Version-controlled assumptions (10 data types, 11 categories), scenario management (7 types, override chains, inheritance), validation engine (min/max/allowed/regex/custom), dependency graph with cycle detection, sensitivity analysis, export/import, and SHA-256 provenance.',
 1, 'sync', true, true, 15, '1.0.0', true,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/assumptions', true, 'default'),

-- GL-FOUND-X-005: Citations & Data Provenance Tracker
('GL-FOUND-X-005', 'Citations & Data Provenance Tracker',
 'Tracks data provenance with citation chains, source verification, DOI resolution, reference integrity checking, cross-reference validation, and SHA-256 hash verification. Ensures every calculation can be traced back to authoritative sources.',
 1, 'async', true, true, 15, '1.0.0', true,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/citations', true, 'default'),

-- GL-FOUND-X-006: Access & Policy Guard Agent
('GL-FOUND-X-006', 'Access & Policy Guard Agent',
 'Policy-based access control with deny-wins evaluation, tenant isolation enforcement, data classification (5 levels), per-role rate limiting, OPA Rego evaluation, SHA-256 decision hashes, compliance reporting, and comprehensive audit trail.',
 1, 'sync', true, true, 50, '1.0.0', false,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/access-guard', true, 'default'),

-- GL-FOUND-X-007: Agent Registry & Service Catalog
('GL-FOUND-X-007', 'Agent Registry & Service Catalog',
 'Centralized agent registry providing agent metadata management, version tracking, capability definitions, dependency graph with cycle detection, health monitoring, service catalog publishing, hot reload, and registry audit trail.',
 1, 'sync', true, true, 30, '1.0.0', false,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/registry', true, 'default')

ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Versions for Foundation Agents
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES

('GL-FOUND-X-001', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "1", "memory_request": "512Mi", "memory_limit": "1Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/orchestrator-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "orchestrator", "dag", "pipeline"}',
 '{"cross-sector"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2'),

('GL-FOUND-X-002', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/schema-compiler-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "schema", "validation", "compiler"}',
 '{"cross-sector"}',
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3'),

('GL-FOUND-X-003', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/normalizer-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "normalizer", "units", "conversion"}',
 '{"cross-sector", "energy", "manufacturing", "transport"}',
 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4'),

('GL-FOUND-X-004', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/assumptions-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "assumptions", "scenarios", "sensitivity"}',
 '{"cross-sector"}',
 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5'),

('GL-FOUND-X-005', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/citations-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "citations", "provenance", "references"}',
 '{"cross-sector"}',
 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6'),

('GL-FOUND-X-006', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/access-guard-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "security", "access-control", "policy"}',
 '{"cross-sector"}',
 'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7'),

('GL-FOUND-X-007', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/agent-registry-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "registry", "catalog", "discovery"}',
 '{"cross-sector"}',
 'a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8')

ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Foundation Agents
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

-- GL-FOUND-X-001 Orchestrator capabilities
('GL-FOUND-X-001', '1.0.0', 'dag_execution', 'orchestration',
 'Execute a DAG of agent tasks with topological sorting, level-based parallel execution, and per-node retry/timeout',
 '{"dag_definition", "execution_context"}', '{"execution_result", "provenance_chain"}',
 '{"max_parallel_nodes": 10, "default_timeout_seconds": 300, "retry_policy": {"max_retries": 3, "backoff": "exponential"}}'::jsonb),

('GL-FOUND-X-001', '1.0.0', 'checkpointing', 'orchestration',
 'DAG-aware checkpointing enabling resume from last successful node after failures',
 '{"dag_state", "checkpoint_id"}', '{"checkpoint_result"}',
 '{"storage_backend": "postgresql", "compression": true}'::jsonb),

-- GL-FOUND-X-002 Schema Compiler capabilities
('GL-FOUND-X-002', '1.0.0', 'schema_validation', 'validation',
 'Validate data against JSON Schema Draft 2020-12 with 7-phase pipeline',
 '{"json_data", "schema_reference"}', '{"validation_result", "fix_suggestions"}',
 '{"draft": "2020-12", "max_errors": 100, "redos_prevention": true}'::jsonb),

('GL-FOUND-X-002', '1.0.0', 'schema_compilation', 'computation',
 'Compile JSON Schema to optimized IR with AST transformation and caching',
 '{"json_schema"}', '{"compiled_ir", "ast"}',
 '{"cache_ttl_seconds": 3600, "optimize": true}'::jsonb),

-- GL-FOUND-X-003 Normalizer capabilities
('GL-FOUND-X-003', '1.0.0', 'unit_conversion', 'transformation',
 'Decimal-precision unit conversion across 8 physical dimensions',
 '{"value", "source_unit", "target_unit"}', '{"converted_value", "provenance"}',
 '{"dimensions": ["mass", "length", "time", "temperature", "energy", "volume", "area", "power"], "precision": 15}'::jsonb),

('GL-FOUND-X-003', '1.0.0', 'entity_resolution', 'normalization',
 'Entity resolution for fuels, materials, and processes with 3-tier matching',
 '{"entity_name", "entity_type"}', '{"resolved_entity", "match_confidence", "provenance"}',
 '{"matching_tiers": ["exact", "alias", "fuzzy"], "fuzzy_threshold": 0.85}'::jsonb),

-- GL-FOUND-X-004 Assumptions capabilities
('GL-FOUND-X-004', '1.0.0', 'assumption_management', 'computation',
 'Version-controlled assumption CRUD with validation and provenance',
 '{"assumption_data", "scenario_context"}', '{"assumption_record", "provenance_hash"}',
 '{"data_types": 10, "categories": 11, "validation_rules": ["min", "max", "allowed", "regex", "custom"]}'::jsonb),

('GL-FOUND-X-004', '1.0.0', 'sensitivity_analysis', 'analysis',
 'Sensitivity analysis across assumption variations',
 '{"assumption_id", "variation_range"}', '{"sensitivity_results", "impact_matrix"}',
 '{"methods": ["one_at_a_time", "monte_carlo", "sobol"]}'::jsonb),

-- GL-FOUND-X-005 Citations capabilities
('GL-FOUND-X-005', '1.0.0', 'citation_tracking', 'computation',
 'Track data provenance with citation chains and source verification',
 '{"source_reference", "data_context"}', '{"citation_record", "provenance_hash"}',
 '{"doi_resolution": true, "cross_reference": true}'::jsonb),

('GL-FOUND-X-005', '1.0.0', 'reference_integrity', 'validation',
 'Validate reference integrity and cross-references across citation chains',
 '{"citation_chain"}', '{"integrity_result", "broken_references"}',
 '{"hash_algorithm": "sha256", "deep_validation": true}'::jsonb),

-- GL-FOUND-X-006 Access Guard capabilities
('GL-FOUND-X-006', '1.0.0', 'access_decision', 'security',
 'Evaluate access control decisions against policies with deny-wins model',
 '{"principal", "resource", "action"}', '{"decision", "matching_rules", "decision_hash"}',
 '{"evaluation_model": "deny_wins", "hash_algorithm": "sha256", "opa_enabled": true}'::jsonb),

('GL-FOUND-X-006', '1.0.0', 'tenant_isolation', 'security',
 'Enforce strict tenant isolation for cross-tenant access prevention',
 '{"principal_tenant", "resource_tenant"}', '{"isolation_result", "violation_details"}',
 '{"enforcement_mode": "strict", "admin_override": true}'::jsonb),

-- GL-FOUND-X-007 Registry capabilities
('GL-FOUND-X-007', '1.0.0', 'agent_registration', 'registry',
 'Register, update, and deregister agents with version tracking and metadata',
 '{"agent_metadata", "version_info"}', '{"registration_result", "agent_id"}',
 '{"hot_reload": true, "cache_invalidation": true}'::jsonb),

('GL-FOUND-X-007', '1.0.0', 'dependency_resolution', 'registry',
 'Resolve agent dependencies with cycle detection and version constraint matching',
 '{"agent_id", "version_constraint"}', '{"dependency_graph", "resolution_order"}',
 '{"cycle_detection": true, "version_matching": "semver"}'::jsonb),

('GL-FOUND-X-007', '1.0.0', 'health_monitoring', 'registry',
 'Monitor agent health with liveness, readiness, and deep health probes',
 '{"agent_id", "probe_type"}', '{"health_result", "latency_ms"}',
 '{"probe_types": ["liveness", "readiness", "startup", "deep"], "timeout_ms": 5000}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Foundation Agents
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Orchestrator depends on Schema Compiler for input validation
('GL-FOUND-X-001', 'GL-FOUND-X-002', '>=1.0.0', false,
 'DAG input definitions must be validated against JSON Schema before execution'),

-- Orchestrator depends on Registry for agent discovery
('GL-FOUND-X-001', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Orchestrator discovers available agents and their capabilities from the registry'),

-- Orchestrator optionally uses Assumptions for scenario context
('GL-FOUND-X-001', 'GL-FOUND-X-004', '>=1.0.0', true,
 'Scenario-aware DAG execution loads assumption overrides from the assumptions registry'),

-- Schema Compiler uses Registry for schema resolution
('GL-FOUND-X-002', 'GL-FOUND-X-007', '>=1.0.0', true,
 'Schema references may resolve to other agents registered schemas'),

-- Normalizer depends on Schema Compiler for input validation
('GL-FOUND-X-003', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Unit conversion requests are validated against normalizer input schemas'),

-- Normalizer optionally uses Citations for provenance
('GL-FOUND-X-003', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Conversion factor provenance is tracked via the citation service'),

-- Assumptions depends on Schema Compiler for validation
('GL-FOUND-X-004', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Assumption data types are validated against JSON Schema definitions'),

-- Assumptions optionally uses Citations for source tracking
('GL-FOUND-X-004', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Assumption sources are tracked via the citation service'),

-- Citations depends on Schema Compiler for reference schema validation
('GL-FOUND-X-005', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Citation records and references are validated against citation schemas'),

-- Access Guard uses Registry for agent identity resolution
('GL-FOUND-X-006', 'GL-FOUND-X-007', '>=1.0.0', true,
 'Agent principals are resolved against the registry for identity-based policies'),

-- Registry depends on Access Guard for authorization
('GL-FOUND-X-007', 'GL-FOUND-X-006', '>=1.0.0', true,
 'Registry operations are authorized by the access guard service')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entries for Foundation Agents
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES

('GL-FOUND-X-001', 'GreenLang Orchestrator',
 'DAG execution engine for coordinating multi-agent pipelines with parallel execution, checkpointing, and provenance tracking.',
 'orchestration', 'active', 'default'),

('GL-FOUND-X-002', 'Schema Compiler & Validator',
 'JSON Schema Draft 2020-12 validation and compilation service for all GreenLang data interchange.',
 'foundation', 'active', 'default'),

('GL-FOUND-X-003', 'Unit & Reference Normalizer',
 'Decimal-precision unit conversion and entity resolution for carbon accounting calculations.',
 'foundation', 'active', 'default'),

('GL-FOUND-X-004', 'Assumptions Registry',
 'Version-controlled assumption management with scenario support and sensitivity analysis.',
 'foundation', 'active', 'default'),

('GL-FOUND-X-005', 'Citations & Data Provenance Tracker',
 'Data provenance tracking with citation chains, source verification, and reference integrity.',
 'foundation', 'active', 'default'),

('GL-FOUND-X-006', 'Access & Policy Guard Agent',
 'Policy-based access control with tenant isolation, data classification, rate limiting, and OPA Rego evaluation.',
 'security', 'active', 'default'),

('GL-FOUND-X-007', 'Agent Registry & Service Catalog',
 'Centralized agent registry, health monitoring, dependency resolution, and service catalog for agent discovery.',
 'foundation', 'active', 'default')

ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA agent_registry_service IS 'Agent Registry & Service Catalog for GreenLang Climate OS (AGENT-FOUND-007) - centralized agent metadata management, version tracking, capability definitions, dependency graph, health monitoring, service catalog publishing, hot reload, and registry audit trail';
COMMENT ON TABLE agent_registry_service.agents IS 'Top-level agent metadata registry with execution configuration, layer classification, GLIP version, checkpointing support, and tenant scoping';
COMMENT ON TABLE agent_registry_service.agent_versions IS 'Version records for agents with resource profiles, container specifications, legacy HTTP config, tags, sectors, and provenance hashes';
COMMENT ON TABLE agent_registry_service.agent_capabilities IS 'Capability definitions per agent per version specifying name, category, description, input/output types, and parameter constraints';
COMMENT ON TABLE agent_registry_service.agent_dependencies IS 'Dependency graph between agents with version constraints, optional flags, and dependency reasons for DAG construction and cycle detection';
COMMENT ON TABLE agent_registry_service.health_checks IS 'TimescaleDB hypertable: agent health check results over time with status, latency, error messages, and probe type';
COMMENT ON TABLE agent_registry_service.registry_audit_log IS 'TimescaleDB hypertable: comprehensive registry audit events with event type, change type, old/new data, user, and provenance hash';
COMMENT ON TABLE agent_registry_service.agent_variants IS 'Variant records for agents representing alternative configurations for different sectors, regions, regulations, or frameworks';
COMMENT ON TABLE agent_registry_service.service_catalog IS 'Published service catalog entries providing the public-facing directory of available agents with display names, summaries, categories, and publication status';
COMMENT ON MATERIALIZED VIEW agent_registry_service.hourly_health_summary IS 'Continuous aggregate: hourly health check summary by agent and status for dashboard queries, trend analysis, and SLI tracking';
COMMENT ON MATERIALIZED VIEW agent_registry_service.daily_registry_events IS 'Continuous aggregate: daily registry event counts by event type and change type for compliance reporting and long-term trend analysis';

COMMENT ON COLUMN agent_registry_service.agents.agent_id IS 'Unique identifier for the agent (e.g., GL-FOUND-X-001)';
COMMENT ON COLUMN agent_registry_service.agents.layer IS 'Execution layer: 1 = foundation, 2 = domain, 3 = orchestrator';
COMMENT ON COLUMN agent_registry_service.agents.execution_mode IS 'Agent execution mode: sync, async, streaming, or batch';
COMMENT ON COLUMN agent_registry_service.agents.idempotency_support IS 'Whether the agent guarantees idempotent execution (same input produces same effect)';
COMMENT ON COLUMN agent_registry_service.agents.deterministic IS 'Whether the agent produces deterministic output for the same input';
COMMENT ON COLUMN agent_registry_service.agents.glip_version IS 'GreenLang Integration Protocol version the agent implements';
COMMENT ON COLUMN agent_registry_service.agents.supports_checkpointing IS 'Whether the agent supports DAG-aware checkpointing for resume after failure';

COMMENT ON COLUMN agent_registry_service.agent_versions.resource_profile IS 'JSONB resource requirements (cpu_request, cpu_limit, memory_request, memory_limit, gpu)';
COMMENT ON COLUMN agent_registry_service.agent_versions.container_spec IS 'JSONB container specification (image, tag, port, command, args)';
COMMENT ON COLUMN agent_registry_service.agent_versions.legacy_http_config IS 'JSONB HTTP configuration for non-GLIP agents (base_url, timeout, auth)';
COMMENT ON COLUMN agent_registry_service.agent_versions.provenance_hash IS 'SHA-256 hash of the version content for integrity verification';

COMMENT ON COLUMN agent_registry_service.agent_capabilities.category IS 'Capability category: computation, transformation, validation, integration, reporting, orchestration, normalization, analysis, security, registry';
COMMENT ON COLUMN agent_registry_service.agent_capabilities.parameters IS 'JSONB capability parameters and constraints';

COMMENT ON COLUMN agent_registry_service.agent_dependencies.version_constraint IS 'Semver version constraint for the dependency (e.g., >=1.0.0, ~1.2, *)';
COMMENT ON COLUMN agent_registry_service.agent_dependencies.optional IS 'Whether this dependency is optional (agent can function without it)';

COMMENT ON COLUMN agent_registry_service.health_checks.status IS 'Health check status: healthy, unhealthy, degraded, unknown, starting';
COMMENT ON COLUMN agent_registry_service.health_checks.probe_type IS 'Health probe type: liveness, readiness, startup, deep';
COMMENT ON COLUMN agent_registry_service.health_checks.latency_ms IS 'Health check response latency in milliseconds';

COMMENT ON COLUMN agent_registry_service.registry_audit_log.event_type IS 'Audit event type: agent_registered, version_published, capability_added, dependency_added, health_check_failed, catalog_published, etc.';
COMMENT ON COLUMN agent_registry_service.registry_audit_log.change_type IS 'Change type: create, update, delete, enable, disable, publish, unpublish, deprecate, system, admin';
COMMENT ON COLUMN agent_registry_service.registry_audit_log.provenance_hash IS 'SHA-256 hash of the event content for integrity verification';

COMMENT ON COLUMN agent_registry_service.agent_variants.variant_type IS 'Variant type: sector, region, regulation, framework, performance, accuracy, custom';

COMMENT ON COLUMN agent_registry_service.service_catalog.status IS 'Catalog entry status: active, deprecated, coming_soon, retired, beta';
COMMENT ON COLUMN agent_registry_service.service_catalog.category IS 'Catalog category: foundation, domain, orchestration, security, integration, reporting, utility';
