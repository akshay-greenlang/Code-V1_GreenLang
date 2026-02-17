-- =============================================================================
-- V047: Schema Migration Agent Service (AGENT-DATA-017 / GL-DATA-X-020)
-- =============================================================================
-- Component: AGENT-DATA-017 (Schema Migration Agent)
-- Agent ID:  GL-DATA-X-020
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Schema Migration Agent (GL-DATA-X-020) with capabilities for
-- schema registry management (namespace/name/type/owner/tags/status),
-- version control (definition snapshots, changelogs, deprecation, sunset
-- dates), breaking-change detection (field-level diff, severity
-- classification, cosmetic/additive/breaking/destructive changes),
-- compatibility checking (forward/backward/full/none compatibility levels,
-- structured issue and recommendation payloads), migration plan lifecycle
-- (source-to-target planning, step-by-step execution blueprints, dry-run
-- results, effort/record estimation), execution tracking (per-step progress,
-- checkpoint/resume, records processed/failed/skipped, error details,
-- structured execution logs), rollback management (full/partial rollback,
-- step rewind, record reversion tracking), field-level mapping (exact/
-- renamed/transformed/split/merged/dropped/added mapping types, confidence
-- scoring, transform rules), schema drift detection (structural/type/
-- constraint/enumeration/format/distribution drift, severity classification,
-- sample evidence), and full provenance chain tracking with SHA-256 hashes
-- for zero-hallucination audit trails.
-- =============================================================================
-- Tables (10):
--   1. schema_registry             - Registered schemas (namespace/name/type/owner)
--   2. schema_versions             - Versioned schema snapshots with changelogs
--   3. schema_changes              - Field-level diff records between versions
--   4. schema_compatibility_checks - Compatibility analysis results
--   5. schema_migration_plans      - Source-to-target migration blueprints
--   6. schema_migration_executions - Execution progress and checkpoint state
--   7. schema_rollbacks            - Rollback operations and reversion records
--   8. schema_field_mappings       - Per-field source-to-target transform rules
--   9. schema_drift_events         - Detected structural/type/constraint drift
--  10. schema_audit_log            - Full audit trail with provenance chains
--
-- Hypertables (3):
--  11. schema_change_events        - Change event time-series (hypertable)
--  12. migration_execution_events  - Execution event time-series (hypertable)
--  13. drift_detection_events      - Drift event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. schema_changes_hourly_stats     - Hourly rollup of schema_change_events
--   2. migration_executions_hourly_stats - Hourly rollup of migration_execution_events
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-020.
-- Previous: V046__data_freshness_monitor_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS schema_migration_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION schema_migration_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: schema_migration_service.schema_registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_registry (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace       VARCHAR(255) NOT NULL,
    name            VARCHAR(255) NOT NULL,
    schema_type     VARCHAR(50)  NOT NULL DEFAULT 'json_schema',
    owner           VARCHAR(255) NOT NULL DEFAULT '',
    tags            JSONB        NOT NULL DEFAULT '[]'::jsonb,
    status          VARCHAR(50)  NOT NULL DEFAULT 'draft',
    description     TEXT         NOT NULL DEFAULT '',
    metadata        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    definition_json JSONB,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_registry
    ADD CONSTRAINT uq_sr_namespace_name UNIQUE (namespace, name);

ALTER TABLE schema_migration_service.schema_registry
    ADD CONSTRAINT chk_sr_namespace_not_empty CHECK (LENGTH(TRIM(namespace)) > 0);

ALTER TABLE schema_migration_service.schema_registry
    ADD CONSTRAINT chk_sr_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE schema_migration_service.schema_registry
    ADD CONSTRAINT chk_sr_schema_type CHECK (schema_type IN (
        'json_schema', 'avro', 'protobuf', 'openapi', 'graphql',
        'sql_ddl', 'thrift', 'parquet', 'xml_schema', 'custom'
    ));

ALTER TABLE schema_migration_service.schema_registry
    ADD CONSTRAINT chk_sr_status CHECK (status IN (
        'draft', 'active', 'deprecated', 'archived', 'review', 'rejected'
    ));

CREATE TRIGGER trg_sr_updated_at
    BEFORE UPDATE ON schema_migration_service.schema_registry
    FOR EACH ROW EXECUTE FUNCTION schema_migration_service.set_updated_at();

-- =============================================================================
-- Table 2: schema_migration_service.schema_versions
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_versions (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_id       UUID         NOT NULL,
    version         VARCHAR(50)  NOT NULL,
    definition_json JSONB        NOT NULL,
    changelog       TEXT         NOT NULL DEFAULT '',
    is_deprecated   BOOLEAN      NOT NULL DEFAULT FALSE,
    deprecated_at   TIMESTAMPTZ,
    sunset_date     DATE,
    created_by      VARCHAR(255) NOT NULL DEFAULT 'system',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_versions
    ADD CONSTRAINT fk_sv_schema_id
        FOREIGN KEY (schema_id)
        REFERENCES schema_migration_service.schema_registry(id)
        ON DELETE CASCADE;

ALTER TABLE schema_migration_service.schema_versions
    ADD CONSTRAINT uq_sv_schema_version UNIQUE (schema_id, version);

ALTER TABLE schema_migration_service.schema_versions
    ADD CONSTRAINT chk_sv_version_not_empty CHECK (LENGTH(TRIM(version)) > 0);

ALTER TABLE schema_migration_service.schema_versions
    ADD CONSTRAINT chk_sv_deprecated_at_after_created
        CHECK (deprecated_at IS NULL OR deprecated_at >= created_at);

ALTER TABLE schema_migration_service.schema_versions
    ADD CONSTRAINT chk_sv_sunset_after_deprecated
        CHECK (sunset_date IS NULL OR deprecated_at IS NULL OR sunset_date >= deprecated_at::date);

-- =============================================================================
-- Table 3: schema_migration_service.schema_changes
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_changes (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    source_version_id UUID,
    target_version_id UUID,
    change_type       VARCHAR(50)  NOT NULL,
    field_path        VARCHAR(500) NOT NULL DEFAULT '',
    old_value         JSONB,
    new_value         JSONB,
    severity          VARCHAR(50)  NOT NULL DEFAULT 'cosmetic',
    description       TEXT         NOT NULL DEFAULT '',
    detected_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_changes
    ADD CONSTRAINT fk_sc_source_version_id
        FOREIGN KEY (source_version_id)
        REFERENCES schema_migration_service.schema_versions(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_changes
    ADD CONSTRAINT fk_sc_target_version_id
        FOREIGN KEY (target_version_id)
        REFERENCES schema_migration_service.schema_versions(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_changes
    ADD CONSTRAINT chk_sc_change_type CHECK (change_type IN (
        'field_added', 'field_removed', 'field_renamed', 'type_changed',
        'constraint_added', 'constraint_removed', 'constraint_modified',
        'format_changed', 'enum_value_added', 'enum_value_removed',
        'default_changed', 'required_added', 'required_removed',
        'schema_restructured', 'metadata_changed'
    ));

ALTER TABLE schema_migration_service.schema_changes
    ADD CONSTRAINT chk_sc_severity CHECK (severity IN (
        'cosmetic', 'additive', 'compatible', 'breaking', 'destructive'
    ));

-- =============================================================================
-- Table 4: schema_migration_service.schema_compatibility_checks
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_compatibility_checks (
    id                    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    source_version_id     UUID,
    target_version_id     UUID,
    compatibility_level   VARCHAR(50)  NOT NULL,
    issues_json           JSONB        NOT NULL DEFAULT '[]'::jsonb,
    recommendations_json  JSONB        NOT NULL DEFAULT '[]'::jsonb,
    checked_by            VARCHAR(255) NOT NULL DEFAULT 'system',
    checked_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_compatibility_checks
    ADD CONSTRAINT fk_scc_source_version_id
        FOREIGN KEY (source_version_id)
        REFERENCES schema_migration_service.schema_versions(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_compatibility_checks
    ADD CONSTRAINT fk_scc_target_version_id
        FOREIGN KEY (target_version_id)
        REFERENCES schema_migration_service.schema_versions(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_compatibility_checks
    ADD CONSTRAINT chk_scc_compatibility_level CHECK (compatibility_level IN (
        'full', 'forward', 'backward', 'none', 'transitive_full',
        'transitive_forward', 'transitive_backward'
    ));

-- =============================================================================
-- Table 5: schema_migration_service.schema_migration_plans
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_migration_plans (
    id                       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    source_schema_id         UUID,
    target_schema_id         UUID,
    source_version           VARCHAR(50)  NOT NULL,
    target_version           VARCHAR(50)  NOT NULL,
    steps_json               JSONB        NOT NULL DEFAULT '[]'::jsonb,
    status                   VARCHAR(50)  NOT NULL DEFAULT 'draft',
    estimated_effort_minutes FLOAT        NOT NULL DEFAULT 0,
    estimated_records        INTEGER      NOT NULL DEFAULT 0,
    dry_run_result           JSONB,
    created_by               VARCHAR(255) NOT NULL DEFAULT 'system',
    created_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_migration_plans
    ADD CONSTRAINT fk_smp_source_schema_id
        FOREIGN KEY (source_schema_id)
        REFERENCES schema_migration_service.schema_registry(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_migration_plans
    ADD CONSTRAINT fk_smp_target_schema_id
        FOREIGN KEY (target_schema_id)
        REFERENCES schema_migration_service.schema_registry(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_migration_plans
    ADD CONSTRAINT chk_smp_source_version_not_empty CHECK (LENGTH(TRIM(source_version)) > 0);

ALTER TABLE schema_migration_service.schema_migration_plans
    ADD CONSTRAINT chk_smp_target_version_not_empty CHECK (LENGTH(TRIM(target_version)) > 0);

ALTER TABLE schema_migration_service.schema_migration_plans
    ADD CONSTRAINT chk_smp_status CHECK (status IN (
        'draft', 'review', 'approved', 'scheduled', 'executing',
        'completed', 'failed', 'cancelled', 'rolled_back'
    ));

ALTER TABLE schema_migration_service.schema_migration_plans
    ADD CONSTRAINT chk_smp_effort_non_negative CHECK (estimated_effort_minutes >= 0);

ALTER TABLE schema_migration_service.schema_migration_plans
    ADD CONSTRAINT chk_smp_records_non_negative CHECK (estimated_records >= 0);

CREATE TRIGGER trg_smp_updated_at
    BEFORE UPDATE ON schema_migration_service.schema_migration_plans
    FOR EACH ROW EXECUTE FUNCTION schema_migration_service.set_updated_at();

-- =============================================================================
-- Table 6: schema_migration_service.schema_migration_executions
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_migration_executions (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id           UUID,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at      TIMESTAMPTZ,
    status            VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_step      INTEGER     NOT NULL DEFAULT 0,
    total_steps       INTEGER     NOT NULL DEFAULT 0,
    records_processed INTEGER     NOT NULL DEFAULT 0,
    records_failed    INTEGER     NOT NULL DEFAULT 0,
    records_skipped   INTEGER     NOT NULL DEFAULT 0,
    checkpoint_data   JSONB,
    error_details     TEXT,
    execution_log     JSONB       NOT NULL DEFAULT '[]'::jsonb
);

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT fk_sme_plan_id
        FOREIGN KEY (plan_id)
        REFERENCES schema_migration_service.schema_migration_plans(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT chk_sme_status CHECK (status IN (
        'pending', 'running', 'paused', 'completed', 'failed',
        'cancelled', 'rolling_back', 'rolled_back'
    ));

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT chk_sme_current_step_non_negative CHECK (current_step >= 0);

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT chk_sme_total_steps_non_negative CHECK (total_steps >= 0);

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT chk_sme_records_processed_non_negative CHECK (records_processed >= 0);

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT chk_sme_records_failed_non_negative CHECK (records_failed >= 0);

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT chk_sme_records_skipped_non_negative CHECK (records_skipped >= 0);

ALTER TABLE schema_migration_service.schema_migration_executions
    ADD CONSTRAINT chk_sme_completed_after_started
        CHECK (completed_at IS NULL OR completed_at >= started_at);

-- =============================================================================
-- Table 7: schema_migration_service.schema_rollbacks
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_rollbacks (
    id                   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id         UUID,
    reason               TEXT         NOT NULL DEFAULT '',
    rollback_type        VARCHAR(50)  NOT NULL DEFAULT 'full',
    rolled_back_to_step  INTEGER,
    records_reverted     INTEGER      NOT NULL DEFAULT 0,
    started_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at         TIMESTAMPTZ,
    status               VARCHAR(50)  NOT NULL DEFAULT 'pending',
    error_details        TEXT
);

ALTER TABLE schema_migration_service.schema_rollbacks
    ADD CONSTRAINT fk_srb_execution_id
        FOREIGN KEY (execution_id)
        REFERENCES schema_migration_service.schema_migration_executions(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_rollbacks
    ADD CONSTRAINT chk_srb_rollback_type CHECK (rollback_type IN (
        'full', 'partial', 'step', 'checkpoint', 'emergency'
    ));

ALTER TABLE schema_migration_service.schema_rollbacks
    ADD CONSTRAINT chk_srb_status CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'partial'
    ));

ALTER TABLE schema_migration_service.schema_rollbacks
    ADD CONSTRAINT chk_srb_records_reverted_non_negative CHECK (records_reverted >= 0);

ALTER TABLE schema_migration_service.schema_rollbacks
    ADD CONSTRAINT chk_srb_rolled_back_to_step_non_negative
        CHECK (rolled_back_to_step IS NULL OR rolled_back_to_step >= 0);

ALTER TABLE schema_migration_service.schema_rollbacks
    ADD CONSTRAINT chk_srb_completed_after_started
        CHECK (completed_at IS NULL OR completed_at >= started_at);

-- =============================================================================
-- Table 8: schema_migration_service.schema_field_mappings
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_field_mappings (
    id               UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    source_schema_id UUID,
    target_schema_id UUID,
    source_field     VARCHAR(500) NOT NULL,
    target_field     VARCHAR(500) NOT NULL,
    transform_rule   JSONB,
    confidence       FLOAT        NOT NULL DEFAULT 1.0,
    mapping_type     VARCHAR(50)  NOT NULL DEFAULT 'exact',
    created_by       VARCHAR(255) NOT NULL DEFAULT 'system',
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_field_mappings
    ADD CONSTRAINT fk_sfm_source_schema_id
        FOREIGN KEY (source_schema_id)
        REFERENCES schema_migration_service.schema_registry(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_field_mappings
    ADD CONSTRAINT fk_sfm_target_schema_id
        FOREIGN KEY (target_schema_id)
        REFERENCES schema_migration_service.schema_registry(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_field_mappings
    ADD CONSTRAINT uq_sfm_source_target_fields
        UNIQUE (source_schema_id, target_schema_id, source_field, target_field);

ALTER TABLE schema_migration_service.schema_field_mappings
    ADD CONSTRAINT chk_sfm_source_field_not_empty CHECK (LENGTH(TRIM(source_field)) > 0);

ALTER TABLE schema_migration_service.schema_field_mappings
    ADD CONSTRAINT chk_sfm_target_field_not_empty CHECK (LENGTH(TRIM(target_field)) > 0);

ALTER TABLE schema_migration_service.schema_field_mappings
    ADD CONSTRAINT chk_sfm_confidence_range
        CHECK (confidence >= 0.0 AND confidence <= 1.0);

ALTER TABLE schema_migration_service.schema_field_mappings
    ADD CONSTRAINT chk_sfm_mapping_type CHECK (mapping_type IN (
        'exact', 'renamed', 'transformed', 'split', 'merged',
        'dropped', 'added', 'computed', 'conditional'
    ));

-- =============================================================================
-- Table 9: schema_migration_service.schema_drift_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_drift_events (
    id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_id      UUID,
    version_id     UUID,
    dataset_id     VARCHAR(255) NOT NULL DEFAULT '',
    drift_type     VARCHAR(50)  NOT NULL,
    field_path     VARCHAR(500) NOT NULL DEFAULT '',
    expected_value JSONB,
    actual_value   JSONB,
    severity       VARCHAR(50)  NOT NULL DEFAULT 'low',
    sample_count   INTEGER      NOT NULL DEFAULT 0,
    detected_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_drift_events
    ADD CONSTRAINT fk_sde_schema_id
        FOREIGN KEY (schema_id)
        REFERENCES schema_migration_service.schema_registry(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_drift_events
    ADD CONSTRAINT fk_sde_version_id
        FOREIGN KEY (version_id)
        REFERENCES schema_migration_service.schema_versions(id)
        ON DELETE SET NULL;

ALTER TABLE schema_migration_service.schema_drift_events
    ADD CONSTRAINT chk_sde_drift_type CHECK (drift_type IN (
        'structural', 'type_mismatch', 'constraint_violation',
        'enumeration_mismatch', 'format_deviation', 'distribution_shift',
        'null_rate_change', 'cardinality_change', 'pattern_deviation',
        'referential_integrity'
    ));

ALTER TABLE schema_migration_service.schema_drift_events
    ADD CONSTRAINT chk_sde_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'info'
    ));

ALTER TABLE schema_migration_service.schema_drift_events
    ADD CONSTRAINT chk_sde_sample_count_non_negative CHECK (sample_count >= 0);

-- =============================================================================
-- Table 10: schema_migration_service.schema_audit_log
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_audit_log (
    id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    action         VARCHAR(100) NOT NULL,
    entity_type    VARCHAR(100) NOT NULL,
    entity_id      UUID,
    actor          VARCHAR(255) NOT NULL DEFAULT 'system',
    details_json   JSONB,
    previous_state JSONB,
    new_state      JSONB,
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    parent_hash    VARCHAR(128) NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE schema_migration_service.schema_audit_log
    ADD CONSTRAINT chk_sal_action_not_empty CHECK (LENGTH(TRIM(action)) > 0);

ALTER TABLE schema_migration_service.schema_audit_log
    ADD CONSTRAINT chk_sal_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE schema_migration_service.schema_audit_log
    ADD CONSTRAINT chk_sal_action CHECK (action IN (
        'schema_registered', 'schema_updated', 'schema_deprecated', 'schema_archived',
        'version_created', 'version_deprecated', 'version_sunset',
        'change_detected', 'change_classified', 'change_reviewed',
        'compatibility_checked', 'compatibility_failed', 'compatibility_passed',
        'plan_created', 'plan_updated', 'plan_approved', 'plan_rejected',
        'plan_scheduled', 'plan_cancelled',
        'execution_started', 'execution_paused', 'execution_resumed',
        'execution_completed', 'execution_failed', 'execution_cancelled',
        'rollback_initiated', 'rollback_completed', 'rollback_failed',
        'mapping_created', 'mapping_updated', 'mapping_deleted',
        'drift_detected', 'drift_resolved', 'drift_acknowledged',
        'export_generated', 'import_completed', 'config_changed'
    ));

ALTER TABLE schema_migration_service.schema_audit_log
    ADD CONSTRAINT chk_sal_entity_type CHECK (entity_type IN (
        'schema', 'version', 'change', 'compatibility_check',
        'migration_plan', 'execution', 'rollback', 'field_mapping',
        'drift_event', 'config'
    ));

-- =============================================================================
-- Table 11: schema_migration_service.schema_change_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.schema_change_events (
    schema_id       UUID,
    version_from    VARCHAR(50),
    version_to      VARCHAR(50),
    change_count    INTEGER      NOT NULL DEFAULT 0,
    breaking_count  INTEGER      NOT NULL DEFAULT 0,
    ts              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'schema_migration_service.schema_change_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE schema_migration_service.schema_change_events
    ADD CONSTRAINT chk_sce_change_count_non_negative CHECK (change_count >= 0);

ALTER TABLE schema_migration_service.schema_change_events
    ADD CONSTRAINT chk_sce_breaking_count_non_negative CHECK (breaking_count >= 0);

-- =============================================================================
-- Table 12: schema_migration_service.migration_execution_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.migration_execution_events (
    execution_id      UUID,
    plan_id           UUID,
    step              INTEGER      NOT NULL DEFAULT 0,
    records_processed INTEGER      NOT NULL DEFAULT 0,
    records_failed    INTEGER      NOT NULL DEFAULT 0,
    duration_ms       FLOAT        NOT NULL DEFAULT 0,
    ts                TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'schema_migration_service.migration_execution_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE schema_migration_service.migration_execution_events
    ADD CONSTRAINT chk_mee_step_non_negative CHECK (step >= 0);

ALTER TABLE schema_migration_service.migration_execution_events
    ADD CONSTRAINT chk_mee_records_processed_non_negative CHECK (records_processed >= 0);

ALTER TABLE schema_migration_service.migration_execution_events
    ADD CONSTRAINT chk_mee_records_failed_non_negative CHECK (records_failed >= 0);

ALTER TABLE schema_migration_service.migration_execution_events
    ADD CONSTRAINT chk_mee_duration_non_negative CHECK (duration_ms >= 0);

-- =============================================================================
-- Table 13: schema_migration_service.drift_detection_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migration_service.drift_detection_events (
    schema_id    UUID,
    dataset_id   VARCHAR(255) NOT NULL DEFAULT '',
    drift_count  INTEGER      NOT NULL DEFAULT 0,
    severity_max VARCHAR(50),
    ts           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'schema_migration_service.drift_detection_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE schema_migration_service.drift_detection_events
    ADD CONSTRAINT chk_dde_drift_count_non_negative CHECK (drift_count >= 0);

ALTER TABLE schema_migration_service.drift_detection_events
    ADD CONSTRAINT chk_dde_severity_max CHECK (
        severity_max IS NULL OR severity_max IN ('critical', 'high', 'medium', 'low', 'info')
    );

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- schema_changes_hourly_stats: hourly rollup of schema_change_events
CREATE MATERIALIZED VIEW schema_migration_service.schema_changes_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts)      AS bucket,
    schema_id,
    COUNT(*)                        AS total_events,
    SUM(change_count)               AS total_changes,
    SUM(breaking_count)             AS total_breaking_changes,
    AVG(change_count)               AS avg_changes_per_event,
    MAX(breaking_count)             AS max_breaking_per_event
FROM schema_migration_service.schema_change_events
WHERE ts IS NOT NULL
GROUP BY bucket, schema_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'schema_migration_service.schema_changes_hourly_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- migration_executions_hourly_stats: hourly rollup of migration_execution_events
CREATE MATERIALIZED VIEW schema_migration_service.migration_executions_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts)      AS bucket,
    plan_id,
    COUNT(*)                        AS total_events,
    SUM(records_processed)          AS total_records_processed,
    SUM(records_failed)             AS total_records_failed,
    AVG(duration_ms)                AS avg_step_duration_ms,
    MAX(duration_ms)                AS max_step_duration_ms,
    MAX(step)                       AS max_step_reached
FROM schema_migration_service.migration_execution_events
WHERE ts IS NOT NULL
GROUP BY bucket, plan_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'schema_migration_service.migration_executions_hourly_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- schema_registry indexes (10)
CREATE INDEX IF NOT EXISTS idx_sr_namespace          ON schema_migration_service.schema_registry(namespace);
CREATE INDEX IF NOT EXISTS idx_sr_name               ON schema_migration_service.schema_registry(name);
CREATE INDEX IF NOT EXISTS idx_sr_schema_type        ON schema_migration_service.schema_registry(schema_type);
CREATE INDEX IF NOT EXISTS idx_sr_owner              ON schema_migration_service.schema_registry(owner);
CREATE INDEX IF NOT EXISTS idx_sr_status             ON schema_migration_service.schema_registry(status);
CREATE INDEX IF NOT EXISTS idx_sr_created_at         ON schema_migration_service.schema_registry(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sr_updated_at         ON schema_migration_service.schema_registry(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sr_namespace_type     ON schema_migration_service.schema_registry(namespace, schema_type);
CREATE INDEX IF NOT EXISTS idx_sr_tags               ON schema_migration_service.schema_registry USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_sr_metadata           ON schema_migration_service.schema_registry USING GIN (metadata);

-- schema_versions indexes (9)
CREATE INDEX IF NOT EXISTS idx_sv_schema_id          ON schema_migration_service.schema_versions(schema_id);
CREATE INDEX IF NOT EXISTS idx_sv_version            ON schema_migration_service.schema_versions(version);
CREATE INDEX IF NOT EXISTS idx_sv_is_deprecated      ON schema_migration_service.schema_versions(is_deprecated);
CREATE INDEX IF NOT EXISTS idx_sv_deprecated_at      ON schema_migration_service.schema_versions(deprecated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sv_sunset_date        ON schema_migration_service.schema_versions(sunset_date);
CREATE INDEX IF NOT EXISTS idx_sv_created_by         ON schema_migration_service.schema_versions(created_by);
CREATE INDEX IF NOT EXISTS idx_sv_created_at         ON schema_migration_service.schema_versions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sv_schema_version     ON schema_migration_service.schema_versions(schema_id, version);
CREATE INDEX IF NOT EXISTS idx_sv_schema_deprecated  ON schema_migration_service.schema_versions(schema_id, is_deprecated);

-- schema_changes indexes (10)
CREATE INDEX IF NOT EXISTS idx_sc_source_version_id  ON schema_migration_service.schema_changes(source_version_id);
CREATE INDEX IF NOT EXISTS idx_sc_target_version_id  ON schema_migration_service.schema_changes(target_version_id);
CREATE INDEX IF NOT EXISTS idx_sc_change_type        ON schema_migration_service.schema_changes(change_type);
CREATE INDEX IF NOT EXISTS idx_sc_severity           ON schema_migration_service.schema_changes(severity);
CREATE INDEX IF NOT EXISTS idx_sc_field_path         ON schema_migration_service.schema_changes(field_path);
CREATE INDEX IF NOT EXISTS idx_sc_detected_at        ON schema_migration_service.schema_changes(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_source_severity    ON schema_migration_service.schema_changes(source_version_id, severity);
CREATE INDEX IF NOT EXISTS idx_sc_target_severity    ON schema_migration_service.schema_changes(target_version_id, severity);
CREATE INDEX IF NOT EXISTS idx_sc_source_type        ON schema_migration_service.schema_changes(source_version_id, change_type);
CREATE INDEX IF NOT EXISTS idx_sc_target_type        ON schema_migration_service.schema_changes(target_version_id, change_type);

-- schema_compatibility_checks indexes (8)
CREATE INDEX IF NOT EXISTS idx_scc_source_version_id        ON schema_migration_service.schema_compatibility_checks(source_version_id);
CREATE INDEX IF NOT EXISTS idx_scc_target_version_id        ON schema_migration_service.schema_compatibility_checks(target_version_id);
CREATE INDEX IF NOT EXISTS idx_scc_compatibility_level      ON schema_migration_service.schema_compatibility_checks(compatibility_level);
CREATE INDEX IF NOT EXISTS idx_scc_checked_by               ON schema_migration_service.schema_compatibility_checks(checked_by);
CREATE INDEX IF NOT EXISTS idx_scc_checked_at               ON schema_migration_service.schema_compatibility_checks(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_scc_source_target            ON schema_migration_service.schema_compatibility_checks(source_version_id, target_version_id);
CREATE INDEX IF NOT EXISTS idx_scc_source_level             ON schema_migration_service.schema_compatibility_checks(source_version_id, compatibility_level);
CREATE INDEX IF NOT EXISTS idx_scc_issues                   ON schema_migration_service.schema_compatibility_checks USING GIN (issues_json);

-- schema_migration_plans indexes (10)
CREATE INDEX IF NOT EXISTS idx_smp_source_schema_id         ON schema_migration_service.schema_migration_plans(source_schema_id);
CREATE INDEX IF NOT EXISTS idx_smp_target_schema_id         ON schema_migration_service.schema_migration_plans(target_schema_id);
CREATE INDEX IF NOT EXISTS idx_smp_status                   ON schema_migration_service.schema_migration_plans(status);
CREATE INDEX IF NOT EXISTS idx_smp_created_by               ON schema_migration_service.schema_migration_plans(created_by);
CREATE INDEX IF NOT EXISTS idx_smp_created_at               ON schema_migration_service.schema_migration_plans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_smp_updated_at               ON schema_migration_service.schema_migration_plans(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_smp_source_target            ON schema_migration_service.schema_migration_plans(source_schema_id, target_schema_id);
CREATE INDEX IF NOT EXISTS idx_smp_source_status            ON schema_migration_service.schema_migration_plans(source_schema_id, status);
CREATE INDEX IF NOT EXISTS idx_smp_target_status            ON schema_migration_service.schema_migration_plans(target_schema_id, status);
CREATE INDEX IF NOT EXISTS idx_smp_steps                    ON schema_migration_service.schema_migration_plans USING GIN (steps_json);

-- schema_migration_executions indexes (10)
CREATE INDEX IF NOT EXISTS idx_sme_plan_id                  ON schema_migration_service.schema_migration_executions(plan_id);
CREATE INDEX IF NOT EXISTS idx_sme_status                   ON schema_migration_service.schema_migration_executions(status);
CREATE INDEX IF NOT EXISTS idx_sme_started_at               ON schema_migration_service.schema_migration_executions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sme_completed_at             ON schema_migration_service.schema_migration_executions(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_sme_current_step             ON schema_migration_service.schema_migration_executions(current_step);
CREATE INDEX IF NOT EXISTS idx_sme_records_processed        ON schema_migration_service.schema_migration_executions(records_processed DESC);
CREATE INDEX IF NOT EXISTS idx_sme_records_failed           ON schema_migration_service.schema_migration_executions(records_failed DESC);
CREATE INDEX IF NOT EXISTS idx_sme_plan_status              ON schema_migration_service.schema_migration_executions(plan_id, status);
CREATE INDEX IF NOT EXISTS idx_sme_plan_started             ON schema_migration_service.schema_migration_executions(plan_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sme_checkpoint               ON schema_migration_service.schema_migration_executions USING GIN (checkpoint_data);

-- schema_rollbacks indexes (8)
CREATE INDEX IF NOT EXISTS idx_srb_execution_id             ON schema_migration_service.schema_rollbacks(execution_id);
CREATE INDEX IF NOT EXISTS idx_srb_status                   ON schema_migration_service.schema_rollbacks(status);
CREATE INDEX IF NOT EXISTS idx_srb_rollback_type            ON schema_migration_service.schema_rollbacks(rollback_type);
CREATE INDEX IF NOT EXISTS idx_srb_started_at               ON schema_migration_service.schema_rollbacks(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_srb_completed_at             ON schema_migration_service.schema_rollbacks(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_srb_records_reverted         ON schema_migration_service.schema_rollbacks(records_reverted DESC);
CREATE INDEX IF NOT EXISTS idx_srb_execution_status         ON schema_migration_service.schema_rollbacks(execution_id, status);
CREATE INDEX IF NOT EXISTS idx_srb_execution_type           ON schema_migration_service.schema_rollbacks(execution_id, rollback_type);

-- schema_field_mappings indexes (9)
CREATE INDEX IF NOT EXISTS idx_sfm_source_schema_id         ON schema_migration_service.schema_field_mappings(source_schema_id);
CREATE INDEX IF NOT EXISTS idx_sfm_target_schema_id         ON schema_migration_service.schema_field_mappings(target_schema_id);
CREATE INDEX IF NOT EXISTS idx_sfm_source_field             ON schema_migration_service.schema_field_mappings(source_field);
CREATE INDEX IF NOT EXISTS idx_sfm_target_field             ON schema_migration_service.schema_field_mappings(target_field);
CREATE INDEX IF NOT EXISTS idx_sfm_mapping_type             ON schema_migration_service.schema_field_mappings(mapping_type);
CREATE INDEX IF NOT EXISTS idx_sfm_confidence               ON schema_migration_service.schema_field_mappings(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_sfm_created_at               ON schema_migration_service.schema_field_mappings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sfm_source_target            ON schema_migration_service.schema_field_mappings(source_schema_id, target_schema_id);
CREATE INDEX IF NOT EXISTS idx_sfm_transform_rule           ON schema_migration_service.schema_field_mappings USING GIN (transform_rule);

-- schema_drift_events indexes (10)
CREATE INDEX IF NOT EXISTS idx_sde_schema_id                ON schema_migration_service.schema_drift_events(schema_id);
CREATE INDEX IF NOT EXISTS idx_sde_version_id               ON schema_migration_service.schema_drift_events(version_id);
CREATE INDEX IF NOT EXISTS idx_sde_dataset_id               ON schema_migration_service.schema_drift_events(dataset_id);
CREATE INDEX IF NOT EXISTS idx_sde_drift_type               ON schema_migration_service.schema_drift_events(drift_type);
CREATE INDEX IF NOT EXISTS idx_sde_severity                 ON schema_migration_service.schema_drift_events(severity);
CREATE INDEX IF NOT EXISTS idx_sde_detected_at              ON schema_migration_service.schema_drift_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_sde_schema_severity          ON schema_migration_service.schema_drift_events(schema_id, severity);
CREATE INDEX IF NOT EXISTS idx_sde_schema_type              ON schema_migration_service.schema_drift_events(schema_id, drift_type);
CREATE INDEX IF NOT EXISTS idx_sde_version_severity         ON schema_migration_service.schema_drift_events(version_id, severity);
CREATE INDEX IF NOT EXISTS idx_sde_schema_dataset           ON schema_migration_service.schema_drift_events(schema_id, dataset_id);

-- schema_audit_log indexes (10)
CREATE INDEX IF NOT EXISTS idx_sal_action                   ON schema_migration_service.schema_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_sal_entity_type              ON schema_migration_service.schema_audit_log(entity_type);
CREATE INDEX IF NOT EXISTS idx_sal_entity_id                ON schema_migration_service.schema_audit_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_sal_actor                    ON schema_migration_service.schema_audit_log(actor);
CREATE INDEX IF NOT EXISTS idx_sal_created_at               ON schema_migration_service.schema_audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sal_provenance_hash          ON schema_migration_service.schema_audit_log(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_sal_parent_hash              ON schema_migration_service.schema_audit_log(parent_hash);
CREATE INDEX IF NOT EXISTS idx_sal_entity_type_id           ON schema_migration_service.schema_audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_sal_action_created           ON schema_migration_service.schema_audit_log(action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sal_details                  ON schema_migration_service.schema_audit_log USING GIN (details_json);

-- schema_change_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_sce_schema_id                ON schema_migration_service.schema_change_events(schema_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_sce_version_from             ON schema_migration_service.schema_change_events(version_from, ts DESC);
CREATE INDEX IF NOT EXISTS idx_sce_version_to               ON schema_migration_service.schema_change_events(version_to, ts DESC);
CREATE INDEX IF NOT EXISTS idx_sce_schema_versions          ON schema_migration_service.schema_change_events(schema_id, version_from, version_to, ts DESC);
CREATE INDEX IF NOT EXISTS idx_sce_change_count             ON schema_migration_service.schema_change_events(change_count DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_sce_breaking_count           ON schema_migration_service.schema_change_events(breaking_count DESC, ts DESC);

-- migration_execution_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_mee_execution_id             ON schema_migration_service.migration_execution_events(execution_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_mee_plan_id                  ON schema_migration_service.migration_execution_events(plan_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_mee_step                     ON schema_migration_service.migration_execution_events(step, ts DESC);
CREATE INDEX IF NOT EXISTS idx_mee_execution_step           ON schema_migration_service.migration_execution_events(execution_id, step, ts DESC);
CREATE INDEX IF NOT EXISTS idx_mee_records_processed        ON schema_migration_service.migration_execution_events(records_processed DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_mee_duration_ms              ON schema_migration_service.migration_execution_events(duration_ms DESC, ts DESC);

-- drift_detection_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_dde_schema_id                ON schema_migration_service.drift_detection_events(schema_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_dde_dataset_id               ON schema_migration_service.drift_detection_events(dataset_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_dde_severity_max             ON schema_migration_service.drift_detection_events(severity_max, ts DESC);
CREATE INDEX IF NOT EXISTS idx_dde_schema_dataset           ON schema_migration_service.drift_detection_events(schema_id, dataset_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_dde_drift_count              ON schema_migration_service.drift_detection_events(drift_count DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_dde_schema_severity          ON schema_migration_service.drift_detection_events(schema_id, severity_max, ts DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- schema_registry: global resource — no tenant_id; admin-gated writes via is_admin
ALTER TABLE schema_migration_service.schema_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY sr_read  ON schema_migration_service.schema_registry FOR SELECT USING (TRUE);
CREATE POLICY sr_write ON schema_migration_service.schema_registry FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_versions ENABLE ROW LEVEL SECURITY;
CREATE POLICY sv_read  ON schema_migration_service.schema_versions FOR SELECT USING (TRUE);
CREATE POLICY sv_write ON schema_migration_service.schema_versions FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_changes ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_read  ON schema_migration_service.schema_changes FOR SELECT USING (TRUE);
CREATE POLICY sc_write ON schema_migration_service.schema_changes FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_compatibility_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY scc_read  ON schema_migration_service.schema_compatibility_checks FOR SELECT USING (TRUE);
CREATE POLICY scc_write ON schema_migration_service.schema_compatibility_checks FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_migration_plans ENABLE ROW LEVEL SECURITY;
CREATE POLICY smp_read  ON schema_migration_service.schema_migration_plans FOR SELECT USING (TRUE);
CREATE POLICY smp_write ON schema_migration_service.schema_migration_plans FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_migration_executions ENABLE ROW LEVEL SECURITY;
CREATE POLICY sme_read  ON schema_migration_service.schema_migration_executions FOR SELECT USING (TRUE);
CREATE POLICY sme_write ON schema_migration_service.schema_migration_executions FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_rollbacks ENABLE ROW LEVEL SECURITY;
CREATE POLICY srb_read  ON schema_migration_service.schema_rollbacks FOR SELECT USING (TRUE);
CREATE POLICY srb_write ON schema_migration_service.schema_rollbacks FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_field_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY sfm_read  ON schema_migration_service.schema_field_mappings FOR SELECT USING (TRUE);
CREATE POLICY sfm_write ON schema_migration_service.schema_field_mappings FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_drift_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY sde_read  ON schema_migration_service.schema_drift_events FOR SELECT USING (TRUE);
CREATE POLICY sde_write ON schema_migration_service.schema_drift_events FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY sal_read  ON schema_migration_service.schema_audit_log FOR SELECT USING (TRUE);
CREATE POLICY sal_write ON schema_migration_service.schema_audit_log FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE schema_migration_service.schema_change_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY sce_read  ON schema_migration_service.schema_change_events FOR SELECT USING (TRUE);
CREATE POLICY sce_write ON schema_migration_service.schema_change_events FOR ALL   USING (TRUE);

ALTER TABLE schema_migration_service.migration_execution_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY mee_read  ON schema_migration_service.migration_execution_events FOR SELECT USING (TRUE);
CREATE POLICY mee_write ON schema_migration_service.migration_execution_events FOR ALL   USING (TRUE);

ALTER TABLE schema_migration_service.drift_detection_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY dde_read  ON schema_migration_service.drift_detection_events FOR SELECT USING (TRUE);
CREATE POLICY dde_write ON schema_migration_service.drift_detection_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA schema_migration_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA schema_migration_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA schema_migration_service TO greenlang_app;
GRANT SELECT ON schema_migration_service.schema_changes_hourly_stats TO greenlang_app;
GRANT SELECT ON schema_migration_service.migration_executions_hourly_stats TO greenlang_app;

GRANT USAGE ON SCHEMA schema_migration_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA schema_migration_service TO greenlang_readonly;
GRANT SELECT ON schema_migration_service.schema_changes_hourly_stats TO greenlang_readonly;
GRANT SELECT ON schema_migration_service.migration_executions_hourly_stats TO greenlang_readonly;

GRANT ALL ON SCHEMA schema_migration_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA schema_migration_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA schema_migration_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'schema-migration:registry:read',      'schema-migration', 'registry_read',      'View registered schemas in the schema registry'),
    (gen_random_uuid(), 'schema-migration:registry:write',     'schema-migration', 'registry_write',     'Register, update, deprecate, and archive schemas'),
    (gen_random_uuid(), 'schema-migration:versions:read',      'schema-migration', 'versions_read',      'View schema versions and their definitions'),
    (gen_random_uuid(), 'schema-migration:versions:write',     'schema-migration', 'versions_write',     'Create, deprecate, and manage schema versions'),
    (gen_random_uuid(), 'schema-migration:changes:read',       'schema-migration', 'changes_read',       'View field-level changes between schema versions'),
    (gen_random_uuid(), 'schema-migration:changes:write',      'schema-migration', 'changes_write',      'Detect and classify schema changes'),
    (gen_random_uuid(), 'schema-migration:compatibility:read', 'schema-migration', 'compatibility_read', 'View schema compatibility check results'),
    (gen_random_uuid(), 'schema-migration:compatibility:write','schema-migration', 'compatibility_write','Run compatibility checks between schema versions'),
    (gen_random_uuid(), 'schema-migration:plans:read',         'schema-migration', 'plans_read',         'View migration plans, steps, and dry-run results'),
    (gen_random_uuid(), 'schema-migration:plans:write',        'schema-migration', 'plans_write',        'Create, update, approve, and cancel migration plans'),
    (gen_random_uuid(), 'schema-migration:executions:read',    'schema-migration', 'executions_read',    'View migration execution progress, logs, and checkpoints'),
    (gen_random_uuid(), 'schema-migration:executions:write',   'schema-migration', 'executions_write',   'Start, pause, resume, and cancel migration executions'),
    (gen_random_uuid(), 'schema-migration:rollbacks:read',     'schema-migration', 'rollbacks_read',     'View rollback operations and reversion records'),
    (gen_random_uuid(), 'schema-migration:rollbacks:write',    'schema-migration', 'rollbacks_write',    'Initiate and manage rollback operations'),
    (gen_random_uuid(), 'schema-migration:mappings:read',      'schema-migration', 'mappings_read',      'View field-level source-to-target mappings and transform rules'),
    (gen_random_uuid(), 'schema-migration:mappings:write',     'schema-migration', 'mappings_write',     'Create, update, and delete field mappings'),
    (gen_random_uuid(), 'schema-migration:drift:read',         'schema-migration', 'drift_read',         'View schema drift events and deviation details'),
    (gen_random_uuid(), 'schema-migration:drift:write',        'schema-migration', 'drift_write',        'Detect, acknowledge, and resolve schema drift events'),
    (gen_random_uuid(), 'schema-migration:audit:read',         'schema-migration', 'audit_read',         'View schema migration audit log entries and provenance chains'),
    (gen_random_uuid(), 'schema-migration:admin',              'schema-migration', 'admin',              'Schema migration service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('schema_migration_service.schema_change_events',       INTERVAL '90 days');
SELECT add_retention_policy('schema_migration_service.migration_execution_events', INTERVAL '90 days');
SELECT add_retention_policy('schema_migration_service.drift_detection_events',     INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE schema_migration_service.schema_change_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'schema_id',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('schema_migration_service.schema_change_events', INTERVAL '7 days');

ALTER TABLE schema_migration_service.migration_execution_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'plan_id',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('schema_migration_service.migration_execution_events', INTERVAL '7 days');

ALTER TABLE schema_migration_service.drift_detection_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'schema_id',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('schema_migration_service.drift_detection_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Schema Migration Agent (GL-DATA-X-020)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-DATA-X-020',
    'Schema Migration Agent',
    'Schema migration engine for GreenLang Climate OS. Maintains a versioned schema registry (namespace/name/type/owner/tags/status) supporting json_schema, avro, protobuf, openapi, graphql, sql_ddl, thrift, parquet, xml_schema, and custom formats. Creates immutable version snapshots with changelogs, deprecation timestamps, and sunset dates. Detects and classifies field-level changes between versions across 15 change types (field_added, field_removed, field_renamed, type_changed, constraint_added/removed/modified, format_changed, enum_value_added/removed, default_changed, required_added/removed, schema_restructured, metadata_changed) with 5 severity levels (cosmetic/additive/compatible/breaking/destructive). Checks compatibility across 7 levels (full/forward/backward/none/transitive_full/transitive_forward/transitive_backward) with structured issue and recommendation payloads. Generates step-by-step migration plans with dry-run results, effort estimation, and record count forecasting. Executes plans with per-step progress tracking, checkpoint/resume support, and structured execution logs. Manages rollback operations (full/partial/step/checkpoint/emergency) with record reversion tracking. Maintains field-level mappings across 9 mapping types (exact/renamed/transformed/split/merged/dropped/added/computed/conditional) with confidence scoring and transform rules. Detects schema drift across 10 drift types (structural/type_mismatch/constraint_violation/enumeration_mismatch/format_deviation/distribution_shift/null_rate_change/cardinality_change/pattern_deviation/referential_integrity). SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 5, '1.0.0', true,
    'GreenLang Data Team',
    'https://docs.greenlang.ai/agents/schema-migration',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-DATA-X-020', '1.0.0',
    '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/schema-migration-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"schema-migration", "versioning", "compatibility", "drift-detection", "rollback", "field-mapping", "data-governance"}',
    '{"cross-sector", "manufacturing", "retail", "energy", "finance", "agriculture", "utilities"}',
    'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-DATA-X-020', '1.0.0',
    'schema_registration',
    'configuration',
    'Register schemas in the schema registry with namespace, name, type, owner, tags, status, and JSON definition.',
    '{"namespace", "name", "schema_type", "owner", "tags", "status", "definition_json"}',
    '{"schema_id", "registration_status", "validation_result"}',
    '{"schema_types": ["json_schema", "avro", "protobuf", "openapi", "graphql", "sql_ddl", "thrift", "parquet", "xml_schema", "custom"], "statuses": ["draft", "active", "deprecated", "archived", "review", "rejected"]}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'version_management',
    'configuration',
    'Create and manage schema versions with definition snapshots, changelogs, deprecation, and sunset date tracking.',
    '{"schema_id", "version", "definition_json", "changelog"}',
    '{"version_id", "version_status", "deprecation_info"}',
    '{"supports_semver": true, "deprecation_workflow": true, "sunset_enforcement": true}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'change_detection',
    'analysis',
    'Detect and classify field-level changes between schema versions across 15 change types with 5 severity levels.',
    '{"source_version_id", "target_version_id"}',
    '{"changes", "change_summary", "breaking_change_count", "severity_distribution"}',
    '{"change_types": ["field_added", "field_removed", "field_renamed", "type_changed", "constraint_added", "constraint_removed", "constraint_modified", "format_changed", "enum_value_added", "enum_value_removed", "default_changed", "required_added", "required_removed", "schema_restructured", "metadata_changed"], "severities": ["cosmetic", "additive", "compatible", "breaking", "destructive"]}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'compatibility_checking',
    'analysis',
    'Check schema compatibility across 7 levels with structured issue and recommendation payloads.',
    '{"source_version_id", "target_version_id", "compatibility_level"}',
    '{"is_compatible", "issues", "recommendations", "compatibility_level"}',
    '{"compatibility_levels": ["full", "forward", "backward", "none", "transitive_full", "transitive_forward", "transitive_backward"]}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'migration_planning',
    'planning',
    'Generate step-by-step migration plans with dry-run results, effort estimation, and record count forecasting.',
    '{"source_schema_id", "target_schema_id", "source_version", "target_version", "plan_config"}',
    '{"plan_id", "steps", "estimated_effort_minutes", "estimated_records", "dry_run_result"}',
    '{"supports_dry_run": true, "statuses": ["draft", "review", "approved", "scheduled", "executing", "completed", "failed", "cancelled", "rolled_back"]}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'migration_execution',
    'processing',
    'Execute migration plans with per-step progress tracking, checkpoint/resume, and structured execution logs.',
    '{"plan_id", "execution_config"}',
    '{"execution_id", "status", "records_processed", "records_failed", "records_skipped", "execution_log"}',
    '{"supports_checkpointing": true, "supports_pause_resume": true, "statuses": ["pending", "running", "paused", "completed", "failed", "cancelled", "rolling_back", "rolled_back"]}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'rollback_management',
    'processing',
    'Manage rollback operations across 5 rollback types with record reversion tracking and step-level precision.',
    '{"execution_id", "rollback_type", "reason", "rolled_back_to_step"}',
    '{"rollback_id", "status", "records_reverted", "rollback_result"}',
    '{"rollback_types": ["full", "partial", "step", "checkpoint", "emergency"]}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'field_mapping',
    'configuration',
    'Create and manage field-level source-to-target mappings across 9 mapping types with confidence scoring and transform rules.',
    '{"source_schema_id", "target_schema_id", "source_field", "target_field", "mapping_type", "transform_rule"}',
    '{"mapping_id", "confidence", "validation_result"}',
    '{"mapping_types": ["exact", "renamed", "transformed", "split", "merged", "dropped", "added", "computed", "conditional"]}'::jsonb
),
(
    'GL-DATA-X-020', '1.0.0',
    'drift_detection',
    'monitoring',
    'Detect schema drift across 10 drift types with field-path resolution, severity classification, and sample evidence.',
    '{"schema_id", "version_id", "dataset_id", "detection_config"}',
    '{"drift_events", "drift_summary", "severity_distribution", "affected_fields"}',
    '{"drift_types": ["structural", "type_mismatch", "constraint_violation", "enumeration_mismatch", "format_deviation", "distribution_shift", "null_rate_change", "cardinality_change", "pattern_deviation", "referential_integrity"], "severities": ["critical", "high", "medium", "low", "info"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-DATA-X-020', 'GL-FOUND-X-002', '>=1.0.0', false, 'Schema validation of registry entries, version definitions, and migration plan steps'),
    ('GL-DATA-X-020', 'GL-FOUND-X-007', '>=1.0.0', false, 'Agent version and capability lookup for pipeline orchestration'),
    ('GL-DATA-X-020', 'GL-FOUND-X-006', '>=1.0.0', false, 'Access control enforcement for schema registry and migration plan approval'),
    ('GL-DATA-X-020', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for execution progress, drift events, and compatibility checks'),
    ('GL-DATA-X-020', 'GL-FOUND-X-005', '>=1.0.0', true,  'Provenance and audit trail registration with citation service'),
    ('GL-DATA-X-020', 'GL-FOUND-X-008', '>=1.0.0', true,  'Reproducibility verification for deterministic change detection results'),
    ('GL-DATA-X-020', 'GL-FOUND-X-009', '>=1.0.0', true,  'QA Test Harness zero-hallucination verification of migration outputs'),
    ('GL-DATA-X-020', 'GL-DATA-X-013', '>=1.0.0',  true,  'Data quality profiling to validate migrated dataset quality post-execution'),
    ('GL-DATA-X-020', 'GL-DATA-X-018', '>=1.0.0',  true,  'Cross-source reconciliation for pre/post migration consistency checks')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-DATA-X-020',
    'Schema Migration Agent',
    'Schema migration engine. Schema registry (namespace/name/type/owner/tags/status, 10 schema types). Version management (snapshots/changelog/deprecation/sunset). Change detection (15 change types, 5 severity levels). Compatibility checking (7 levels with issues/recommendations). Migration planning (step blueprints/dry-run/effort estimation). Execution tracking (per-step progress/checkpoint/resume). Rollback management (5 types/step rewind/record reversion). Field mappings (9 types/confidence scoring/transform rules). Drift detection (10 drift types/severity). SHA-256 provenance chains.',
    'data', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA schema_migration_service IS
    'Schema Migration Agent (AGENT-DATA-017) - schema registry, version control, change detection, compatibility checking, migration planning/execution, rollback, field mapping, drift detection, provenance chains';

COMMENT ON TABLE schema_migration_service.schema_registry IS
    'Registered schemas: namespace, name, schema_type (10 types), owner, tags, status (6 states), description, metadata, JSON definition';

COMMENT ON TABLE schema_migration_service.schema_versions IS
    'Versioned schema snapshots: schema ref, version string, definition JSON, changelog, deprecation flag/timestamp, sunset date, created_by';

COMMENT ON TABLE schema_migration_service.schema_changes IS
    'Field-level change records: source/target version refs, change_type (15 types), field_path, old/new value, severity (5 levels), description';

COMMENT ON TABLE schema_migration_service.schema_compatibility_checks IS
    'Compatibility analysis results: source/target version refs, compatibility_level (7 levels), issues JSON, recommendations JSON, checked_by';

COMMENT ON TABLE schema_migration_service.schema_migration_plans IS
    'Migration blueprints: source/target schema refs, source/target versions, steps JSON, status (9 states), effort/record estimates, dry-run result';

COMMENT ON TABLE schema_migration_service.schema_migration_executions IS
    'Execution state: plan ref, started/completed timestamps, status (8 states), step counters, record processed/failed/skipped counts, checkpoint data, execution log';

COMMENT ON TABLE schema_migration_service.schema_rollbacks IS
    'Rollback operations: execution ref, reason, rollback_type (5 types), step target, records_reverted, started/completed timestamps, status (5 states)';

COMMENT ON TABLE schema_migration_service.schema_field_mappings IS
    'Field-level mappings: source/target schema refs, source/target field paths, transform_rule JSON, confidence (0-1), mapping_type (9 types)';

COMMENT ON TABLE schema_migration_service.schema_drift_events IS
    'Drift detections: schema/version refs, dataset_id, drift_type (10 types), field_path, expected/actual values, severity (5 levels), sample_count';

COMMENT ON TABLE schema_migration_service.schema_audit_log IS
    'Full audit trail: action (35 types), entity_type (10 types), entity_id, actor, details/previous/new state JSON, SHA-256 provenance and parent hashes';

COMMENT ON TABLE schema_migration_service.schema_change_events IS
    'TimescaleDB hypertable: schema change events with schema_id, version_from/to, change_count, breaking_count (7-day chunks, 90-day retention)';

COMMENT ON TABLE schema_migration_service.migration_execution_events IS
    'TimescaleDB hypertable: execution step events with execution_id, plan_id, step, records_processed/failed, duration_ms (7-day chunks, 90-day retention)';

COMMENT ON TABLE schema_migration_service.drift_detection_events IS
    'TimescaleDB hypertable: drift detection events with schema_id, dataset_id, drift_count, severity_max (7-day chunks, 90-day retention)';

COMMENT ON MATERIALIZED VIEW schema_migration_service.schema_changes_hourly_stats IS
    'Continuous aggregate: hourly schema change stats by schema (total events, total/avg/breaking changes per hour)';

COMMENT ON MATERIALIZED VIEW schema_migration_service.migration_executions_hourly_stats IS
    'Continuous aggregate: hourly execution stats by plan (total events, records processed/failed, avg/max step duration, max step reached)';
