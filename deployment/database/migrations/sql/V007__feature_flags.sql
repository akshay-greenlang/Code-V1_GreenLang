-- =============================================================================
-- V007: Feature Flags Infrastructure
-- =============================================================================
-- Description: Creates feature flag management tables, evaluation hypertable,
--              continuous aggregates, RLS policies, and audit triggers for
--              the GreenLang Climate OS feature flag system.
-- Author: GreenLang Infrastructure Team
-- PRD: INFRA-008 Feature Flags System
-- Requires: TimescaleDB (V002), uuid-ossp (V001)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema Setup
-- -----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS infrastructure;

SET search_path TO infrastructure, public;

-- -----------------------------------------------------------------------------
-- 1. Core Feature Flags Table
-- -----------------------------------------------------------------------------

CREATE TABLE infrastructure.feature_flags (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    key VARCHAR(128) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Flag Configuration
    flag_type VARCHAR(30) NOT NULL DEFAULT 'boolean',
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    default_value JSONB NOT NULL DEFAULT 'false'::jsonb,
    rollout_percentage DECIMAL(5,2) DEFAULT 0,

    -- Targeting
    environments TEXT[] NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',

    -- Ownership
    owner VARCHAR(255),

    -- Extended Configuration
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Scheduling
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1,

    -- Constraints
    CONSTRAINT uq_feature_flags_key UNIQUE (key),

    CONSTRAINT chk_feature_flags_type CHECK (
        flag_type IN (
            'boolean',
            'percentage',
            'user_list',
            'environment',
            'segment',
            'scheduled',
            'multivariate'
        )
    ),

    CONSTRAINT chk_feature_flags_status CHECK (
        status IN (
            'draft',
            'active',
            'rolled_out',
            'permanent',
            'archived',
            'killed'
        )
    ),

    CONSTRAINT chk_feature_flags_rollout CHECK (
        rollout_percentage >= 0 AND rollout_percentage <= 100
    ),

    CONSTRAINT chk_feature_flags_schedule CHECK (
        start_time IS NULL OR end_time IS NULL OR start_time < end_time
    )
);

-- -----------------------------------------------------------------------------
-- 2. Feature Flag Rules Table
-- -----------------------------------------------------------------------------

CREATE TABLE infrastructure.feature_flag_rules (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Parent Reference
    flag_id UUID NOT NULL
        REFERENCES infrastructure.feature_flags(id) ON DELETE CASCADE,

    -- Rule Configuration
    rule_type VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    conditions JSONB NOT NULL DEFAULT '[]'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 3. Feature Flag Variants Table
-- -----------------------------------------------------------------------------

CREATE TABLE infrastructure.feature_flag_variants (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Parent Reference
    flag_id UUID NOT NULL
        REFERENCES infrastructure.feature_flags(id) ON DELETE CASCADE,

    -- Variant Configuration
    variant_key VARCHAR(128) NOT NULL,
    variant_value JSONB NOT NULL,
    weight DECIMAL(5,2) NOT NULL DEFAULT 0,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT uq_flag_variant UNIQUE (flag_id, variant_key),

    CONSTRAINT chk_variant_weight CHECK (
        weight >= 0 AND weight <= 100
    )
);

-- -----------------------------------------------------------------------------
-- 4. Feature Flag Overrides Table
-- -----------------------------------------------------------------------------

CREATE TABLE infrastructure.feature_flag_overrides (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Parent Reference
    flag_id UUID NOT NULL
        REFERENCES infrastructure.feature_flags(id) ON DELETE CASCADE,

    -- Override Scope
    scope_type VARCHAR(20) NOT NULL,
    scope_value VARCHAR(255) NOT NULL,

    -- Override Configuration
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    variant_key VARCHAR(128),

    -- Expiry
    expires_at TIMESTAMPTZ,

    -- Audit
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT uq_flag_override_scope UNIQUE (flag_id, scope_type, scope_value),

    CONSTRAINT chk_override_scope_type CHECK (
        scope_type IN ('user', 'tenant', 'segment', 'environment')
    )
);

-- -----------------------------------------------------------------------------
-- 5. Feature Flag Audit Log Table
-- -----------------------------------------------------------------------------

CREATE TABLE infrastructure.feature_flag_audit_log (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- What Changed
    flag_key VARCHAR(128) NOT NULL,
    action VARCHAR(50) NOT NULL,

    -- Change Details
    old_value JSONB,
    new_value JSONB,

    -- Who Changed It
    changed_by VARCHAR(255) NOT NULL,
    change_reason TEXT,

    -- Request Context
    ip_address INET,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- 6. Feature Flag Evaluations Hypertable
-- -----------------------------------------------------------------------------

CREATE TABLE infrastructure.feature_flag_evaluations (
    -- Time dimension (required first column for hypertable)
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Evaluation Context
    flag_key VARCHAR(128) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(255),

    -- Result
    result BOOLEAN NOT NULL,
    variant_key VARCHAR(128),

    -- Performance
    cache_layer VARCHAR(20),
    duration_us INTEGER NOT NULL DEFAULT 0,

    -- Deduplication / Grouping
    context_hash VARCHAR(64)
);

-- Convert evaluations to TimescaleDB hypertable with 1-day chunks
SELECT create_hypertable(
    'infrastructure.feature_flag_evaluations',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- 7. Indexes - Feature Flags
-- -----------------------------------------------------------------------------

-- Primary lookup: key + status (most common query pattern)
CREATE INDEX idx_ff_key_status
    ON infrastructure.feature_flags (key, status);

-- Active flags only (partial index for hot path)
CREATE INDEX idx_ff_active
    ON infrastructure.feature_flags (status)
    WHERE status = 'active';

-- Owner lookup for management UI
CREATE INDEX idx_ff_owner
    ON infrastructure.feature_flags (owner)
    WHERE owner IS NOT NULL;

-- Tag filtering (GIN for array containment queries)
CREATE INDEX idx_ff_tags
    ON infrastructure.feature_flags USING GIN (tags);

-- Environment filtering (GIN for array containment queries)
CREATE INDEX idx_ff_environments
    ON infrastructure.feature_flags USING GIN (environments);

-- Metadata filtering (GIN for JSONB queries)
CREATE INDEX idx_ff_metadata
    ON infrastructure.feature_flags USING GIN (metadata jsonb_path_ops);

-- Scheduled flags lookup
CREATE INDEX idx_ff_schedule
    ON infrastructure.feature_flags (start_time, end_time)
    WHERE flag_type = 'scheduled';

-- -----------------------------------------------------------------------------
-- 8. Indexes - Feature Flag Rules
-- -----------------------------------------------------------------------------

-- Rules by flag (foreign key index + priority ordering)
CREATE INDEX idx_ff_rules_flag_priority
    ON infrastructure.feature_flag_rules (flag_id, priority);

-- Active rules only
CREATE INDEX idx_ff_rules_enabled
    ON infrastructure.feature_flag_rules (flag_id)
    WHERE enabled = TRUE;

-- -----------------------------------------------------------------------------
-- 9. Indexes - Feature Flag Variants
-- -----------------------------------------------------------------------------

-- Variants by flag (foreign key index)
CREATE INDEX idx_ff_variants_flag
    ON infrastructure.feature_flag_variants (flag_id);

-- -----------------------------------------------------------------------------
-- 10. Indexes - Feature Flag Overrides
-- -----------------------------------------------------------------------------

-- Overrides by flag (foreign key index)
CREATE INDEX idx_ff_overrides_flag
    ON infrastructure.feature_flag_overrides (flag_id);

-- Scope lookup (common query: find all overrides for a user/tenant)
CREATE INDEX idx_ff_overrides_scope
    ON infrastructure.feature_flag_overrides (scope_type, scope_value);

-- Expired overrides cleanup
CREATE INDEX idx_ff_overrides_expires
    ON infrastructure.feature_flag_overrides (expires_at)
    WHERE expires_at IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 11. Indexes - Feature Flag Audit Log
-- -----------------------------------------------------------------------------

-- Audit log by flag key and time
CREATE INDEX idx_ff_audit_flag_time
    ON infrastructure.feature_flag_audit_log (flag_key, created_at DESC);

-- Audit log by time (for recent events view)
CREATE INDEX idx_ff_audit_time
    ON infrastructure.feature_flag_audit_log (created_at DESC);

-- Audit log by user
CREATE INDEX idx_ff_audit_changed_by
    ON infrastructure.feature_flag_audit_log (changed_by, created_at DESC);

-- -----------------------------------------------------------------------------
-- 12. Indexes - Feature Flag Evaluations (Hypertable)
-- -----------------------------------------------------------------------------

-- Evaluation lookup by flag and time
CREATE INDEX idx_ff_eval_flag_time
    ON infrastructure.feature_flag_evaluations (flag_key, time DESC);

-- Evaluation lookup by environment
CREATE INDEX idx_ff_eval_env
    ON infrastructure.feature_flag_evaluations (environment, time DESC);

-- Evaluation lookup by tenant
CREATE INDEX idx_ff_eval_tenant
    ON infrastructure.feature_flag_evaluations (tenant_id, time DESC)
    WHERE tenant_id IS NOT NULL;

-- Cache layer analysis
CREATE INDEX idx_ff_eval_cache
    ON infrastructure.feature_flag_evaluations (cache_layer, time DESC)
    WHERE cache_layer IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 13. Continuous Aggregate - Hourly Evaluation Stats
-- -----------------------------------------------------------------------------

CREATE MATERIALIZED VIEW IF NOT EXISTS infrastructure.ff_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    flag_key,
    environment,
    COUNT(*) AS evaluation_count,
    COUNT(*) FILTER (WHERE result = TRUE) AS true_count,
    COUNT(*) FILTER (WHERE result = FALSE) AS false_count,
    AVG(duration_us)::INTEGER AS avg_duration_us,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_us)::INTEGER AS p99_duration_us,
    COUNT(DISTINCT tenant_id) AS unique_tenants,
    COUNT(*) FILTER (WHERE cache_layer = 'memory') AS memory_cache_hits,
    COUNT(*) FILTER (WHERE cache_layer = 'redis') AS redis_cache_hits,
    COUNT(*) FILTER (WHERE cache_layer IS NULL OR cache_layer = 'none') AS cache_misses
FROM infrastructure.feature_flag_evaluations
GROUP BY time_bucket('1 hour', time), flag_key, environment
WITH NO DATA;

-- Refresh policy: every 5 minutes, with a 2-hour start offset and 10-minute end offset
SELECT add_continuous_aggregate_policy(
    'infrastructure.ff_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- 14. Compression Policy - Evaluations
-- -----------------------------------------------------------------------------

ALTER TABLE infrastructure.feature_flag_evaluations SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'flag_key, environment',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy(
    'infrastructure.feature_flag_evaluations',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- 15. Retention Policy - Evaluations (90 days raw data)
-- -----------------------------------------------------------------------------

SELECT add_retention_policy(
    'infrastructure.feature_flag_evaluations',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- 16. Row-Level Security - Overrides Table (Tenant Isolation)
-- -----------------------------------------------------------------------------

ALTER TABLE infrastructure.feature_flag_overrides ENABLE ROW LEVEL SECURITY;

-- Policy: users can see overrides scoped to their tenant
CREATE POLICY ff_overrides_tenant_isolation
    ON infrastructure.feature_flag_overrides
    FOR SELECT
    USING (
        -- Allow access if scope_type is not 'tenant' (non-tenant scopes are visible)
        scope_type != 'tenant'
        -- Or if the scope_value matches the current tenant
        OR scope_value = NULLIF(current_setting('app.tenant_id', true), '')
        -- Or if the user is an admin (can see all)
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'owner')
    );

-- Policy: only admins can insert/update/delete overrides
CREATE POLICY ff_overrides_admin_write
    ON infrastructure.feature_flag_overrides
    FOR ALL
    USING (
        NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'owner')
    )
    WITH CHECK (
        NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'owner')
    );

-- -----------------------------------------------------------------------------
-- 17. Trigger - Auto-update updated_at and Increment Version
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION infrastructure.ff_update_timestamp_and_version()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    -- Only increment version on the feature_flags table
    IF TG_TABLE_NAME = 'feature_flags' THEN
        NEW.version = COALESCE(OLD.version, 0) + 1;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to feature_flags
CREATE TRIGGER trg_feature_flags_update
    BEFORE UPDATE ON infrastructure.feature_flags
    FOR EACH ROW EXECUTE FUNCTION infrastructure.ff_update_timestamp_and_version();

-- Apply trigger to rules
CREATE TRIGGER trg_feature_flag_rules_update
    BEFORE UPDATE ON infrastructure.feature_flag_rules
    FOR EACH ROW EXECUTE FUNCTION infrastructure.ff_update_timestamp_and_version();

-- Apply trigger to variants
CREATE TRIGGER trg_feature_flag_variants_update
    BEFORE UPDATE ON infrastructure.feature_flag_variants
    FOR EACH ROW EXECUTE FUNCTION infrastructure.ff_update_timestamp_and_version();

-- Apply trigger to overrides
CREATE TRIGGER trg_feature_flag_overrides_update
    BEFORE UPDATE ON infrastructure.feature_flag_overrides
    FOR EACH ROW EXECUTE FUNCTION infrastructure.ff_update_timestamp_and_version();

-- -----------------------------------------------------------------------------
-- 18. Table Comments
-- -----------------------------------------------------------------------------

COMMENT ON TABLE infrastructure.feature_flags IS
    'Core feature flag definitions with type, status, rollout configuration, and scheduling. '
    'Supports boolean, percentage, user_list, environment, segment, scheduled, and multivariate flag types.';

COMMENT ON TABLE infrastructure.feature_flag_rules IS
    'Targeting rules for feature flags. Each rule has a type, priority, and JSONB conditions. '
    'Rules are evaluated in priority order during flag evaluation.';

COMMENT ON TABLE infrastructure.feature_flag_variants IS
    'Multivariate flag variants with weighted distribution. '
    'Each variant has a key, JSONB value, and weight for traffic allocation.';

COMMENT ON TABLE infrastructure.feature_flag_overrides IS
    'Per-scope overrides for feature flags (user, tenant, segment, environment). '
    'Overrides take precedence over rules and default values. Supports expiration.';

COMMENT ON TABLE infrastructure.feature_flag_audit_log IS
    'Immutable audit trail for all feature flag changes. '
    'Records who changed what, when, why, and the before/after state.';

COMMENT ON TABLE infrastructure.feature_flag_evaluations IS
    'TimescaleDB hypertable for feature flag evaluation telemetry. '
    'Tracks every evaluation with timing, cache layer, and result for analytics. '
    '1-day chunks, 7-day compression, 90-day retention.';

COMMENT ON MATERIALIZED VIEW infrastructure.ff_hourly_stats IS
    'Continuous aggregate: Hourly evaluation statistics per flag and environment. '
    'Includes counts, latency percentiles, cache hit rates, and unique tenant counts. '
    'Refreshed every 5 minutes.';

COMMENT ON FUNCTION infrastructure.ff_update_timestamp_and_version() IS
    'Trigger function that auto-updates updated_at timestamp on all feature flag tables '
    'and increments the version column on the feature_flags table for optimistic locking.';

-- -----------------------------------------------------------------------------
-- 19. Verification
-- -----------------------------------------------------------------------------

DO $$
DECLARE
    tbl_name TEXT;
    required_tables TEXT[] := ARRAY[
        'feature_flags',
        'feature_flag_rules',
        'feature_flag_variants',
        'feature_flag_overrides',
        'feature_flag_audit_log',
        'feature_flag_evaluations'
    ];
BEGIN
    FOREACH tbl_name IN ARRAY required_tables
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'infrastructure' AND table_name = tbl_name
        ) THEN
            RAISE EXCEPTION 'Required table infrastructure.% was not created', tbl_name;
        ELSE
            RAISE NOTICE 'Table infrastructure.% created successfully', tbl_name;
        END IF;
    END LOOP;

    -- Verify hypertable
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'infrastructure'
          AND hypertable_name = 'feature_flag_evaluations'
    ) THEN
        RAISE EXCEPTION 'feature_flag_evaluations is not a hypertable';
    ELSE
        RAISE NOTICE 'feature_flag_evaluations hypertable verified';
    END IF;

    RAISE NOTICE 'V007 Feature Flags migration completed successfully';
END $$;
