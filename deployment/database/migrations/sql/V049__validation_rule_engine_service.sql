-- =============================================================================
-- V049: Validation Rule Engine Service Tables
-- =============================================================================
-- Component: AGENT-DATA-019 (Validation Rule Engine)
-- Agent ID:  GL-DATA-X-022
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Validation Rule Engine (GL-DATA-X-022) with capabilities for
-- validation rule management (rule_type/column_name/operator/threshold/
-- severity/status/version), rule set composition (hierarchical sets with
-- parent_set_id self-referencing, SLA thresholds, member ordering via
-- position), compound rule logic (AND/OR/NOT boolean operators with
-- child rule and compound rule nesting), rule pack distribution
-- (versioned packs with rule_definitions for bulk import/export),
-- evaluation execution (per-rule-set evaluation with pass/fail/warn
-- counts, pass_rate, SLA result, duration tracking), evaluation detail
-- capture (per-rule granularity with pass/fail counts, result status,
-- failure details), validation report generation (multi-format with
-- content hashing and provenance), rule version history (change tracking
-- with full snapshot capture), conflict event tracking (rule-vs-rule
-- conflicts on same column with severity classification), and full
-- provenance chain tracking with SHA-256 hashes for zero-hallucination
-- audit trails.
-- =============================================================================
-- Tables (10):
--   1. validation_rules                - Validation rule definitions (type/operator/threshold/severity)
--   2. validation_rule_sets            - Rule set containers with SLA thresholds and hierarchy
--   3. validation_compound_rules       - Compound boolean rules (AND/OR/NOT composition)
--   4. validation_rule_set_members     - Rule-to-set membership with position ordering
--   5. validation_rule_versions        - Rule version history with change snapshots
--   6. validation_rule_packs           - Packaged rule collections for distribution
--   7. validation_evaluations          - Evaluation execution results per rule set
--   8. validation_evaluation_details   - Per-rule evaluation detail records
--   9. validation_reports              - Generated validation reports with hashing
--  10. validation_audit_log            - Full audit trail with provenance chains
--
-- Hypertables (3):
--  11. validation_evaluation_events    - Evaluation event time-series (hypertable)
--  12. validation_rule_change_events   - Rule change event time-series (hypertable)
--  13. validation_conflict_events      - Rule conflict event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. validation_evaluations_hourly_stats  - Hourly rollup of validation_evaluation_events
--   2. validation_rule_changes_hourly_stats - Hourly rollup of validation_rule_change_events
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-022.
-- Previous: V048__data_lineage_tracker_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS validation_rule_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION validation_rule_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: validation_rule_service.validation_rules
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_rules (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255)  NOT NULL,
    rule_type       VARCHAR(50)   NOT NULL,
    column_name     VARCHAR(255)  NOT NULL DEFAULT '',
    operator        VARCHAR(50)   NOT NULL,
    threshold       NUMERIC,
    parameters      JSONB         NOT NULL DEFAULT '{}'::jsonb,
    severity        VARCHAR(20)   NOT NULL DEFAULT 'error',
    status          VARCHAR(20)   NOT NULL DEFAULT 'active',
    version         INTEGER       NOT NULL DEFAULT 1,
    description     TEXT          NOT NULL DEFAULT '',
    tags            TEXT[]        NOT NULL DEFAULT '{}',
    metadata        JSONB         NOT NULL DEFAULT '{}'::jsonb,
    created_by      VARCHAR(255)  NOT NULL DEFAULT 'system',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_rules
    ADD CONSTRAINT uq_vr_name UNIQUE (name);

ALTER TABLE validation_rule_service.validation_rules
    ADD CONSTRAINT chk_vr_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE validation_rule_service.validation_rules
    ADD CONSTRAINT chk_vr_rule_type CHECK (rule_type IN (
        'not_null', 'unique', 'range', 'regex', 'enum', 'type_check',
        'length', 'custom', 'cross_field', 'referential', 'statistical',
        'temporal', 'format', 'completeness'
    ));

ALTER TABLE validation_rule_service.validation_rules
    ADD CONSTRAINT chk_vr_operator CHECK (operator IN (
        'eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'between', 'in', 'not_in',
        'is_null', 'is_not_null', 'matches', 'not_matches', 'contains',
        'starts_with', 'ends_with', 'is_unique', 'custom'
    ));

ALTER TABLE validation_rule_service.validation_rules
    ADD CONSTRAINT chk_vr_severity CHECK (severity IN (
        'info', 'warning', 'error', 'critical'
    ));

ALTER TABLE validation_rule_service.validation_rules
    ADD CONSTRAINT chk_vr_status CHECK (status IN (
        'active', 'inactive', 'draft', 'deprecated', 'archived'
    ));

ALTER TABLE validation_rule_service.validation_rules
    ADD CONSTRAINT chk_vr_version_positive CHECK (version >= 1);

CREATE TRIGGER trg_vr_updated_at
    BEFORE UPDATE ON validation_rule_service.validation_rules
    FOR EACH ROW EXECUTE FUNCTION validation_rule_service.set_updated_at();

-- =============================================================================
-- Table 2: validation_rule_service.validation_rule_sets
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_rule_sets (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255)  NOT NULL,
    description     TEXT          NOT NULL DEFAULT '',
    version         INTEGER       NOT NULL DEFAULT 1,
    status          VARCHAR(20)   NOT NULL DEFAULT 'active',
    sla_thresholds  JSONB         NOT NULL DEFAULT '{}'::jsonb,
    parent_set_id   UUID,
    tags            TEXT[]        NOT NULL DEFAULT '{}',
    metadata        JSONB         NOT NULL DEFAULT '{}'::jsonb,
    created_by      VARCHAR(255)  NOT NULL DEFAULT 'system',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_rule_sets
    ADD CONSTRAINT uq_vrs_name UNIQUE (name);

ALTER TABLE validation_rule_service.validation_rule_sets
    ADD CONSTRAINT chk_vrs_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE validation_rule_service.validation_rule_sets
    ADD CONSTRAINT chk_vrs_status CHECK (status IN (
        'active', 'inactive', 'draft', 'deprecated', 'archived'
    ));

ALTER TABLE validation_rule_service.validation_rule_sets
    ADD CONSTRAINT chk_vrs_version_positive CHECK (version >= 1);

ALTER TABLE validation_rule_service.validation_rule_sets
    ADD CONSTRAINT fk_vrs_parent_set_id
        FOREIGN KEY (parent_set_id)
        REFERENCES validation_rule_service.validation_rule_sets(id)
        ON DELETE SET NULL;

ALTER TABLE validation_rule_service.validation_rule_sets
    ADD CONSTRAINT chk_vrs_no_self_parent CHECK (parent_set_id != id);

CREATE TRIGGER trg_vrs_updated_at
    BEFORE UPDATE ON validation_rule_service.validation_rule_sets
    FOR EACH ROW EXECUTE FUNCTION validation_rule_service.set_updated_at();

-- =============================================================================
-- Table 3: validation_rule_service.validation_compound_rules
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_compound_rules (
    id                UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    name              VARCHAR(255)  NOT NULL,
    operator          VARCHAR(10)   NOT NULL,
    child_rule_ids    UUID[]        NOT NULL DEFAULT '{}',
    child_compound_ids UUID[]       NOT NULL DEFAULT '{}',
    description       TEXT          NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_compound_rules
    ADD CONSTRAINT chk_vcr_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE validation_rule_service.validation_compound_rules
    ADD CONSTRAINT chk_vcr_operator CHECK (operator IN (
        'AND', 'OR', 'NOT'
    ));

CREATE TRIGGER trg_vcr_updated_at
    BEFORE UPDATE ON validation_rule_service.validation_compound_rules
    FOR EACH ROW EXECUTE FUNCTION validation_rule_service.set_updated_at();

-- =============================================================================
-- Table 4: validation_rule_service.validation_rule_set_members
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_rule_set_members (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    set_id            UUID         NOT NULL,
    rule_id           UUID         NOT NULL,
    compound_rule_id  UUID,
    position          INTEGER      NOT NULL DEFAULT 0,
    override_severity VARCHAR(20),
    added_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_rule_set_members
    ADD CONSTRAINT uq_vrsm_set_rule UNIQUE (set_id, rule_id);

ALTER TABLE validation_rule_service.validation_rule_set_members
    ADD CONSTRAINT fk_vrsm_set_id
        FOREIGN KEY (set_id)
        REFERENCES validation_rule_service.validation_rule_sets(id)
        ON DELETE CASCADE;

ALTER TABLE validation_rule_service.validation_rule_set_members
    ADD CONSTRAINT fk_vrsm_rule_id
        FOREIGN KEY (rule_id)
        REFERENCES validation_rule_service.validation_rules(id)
        ON DELETE CASCADE;

ALTER TABLE validation_rule_service.validation_rule_set_members
    ADD CONSTRAINT fk_vrsm_compound_rule_id
        FOREIGN KEY (compound_rule_id)
        REFERENCES validation_rule_service.validation_compound_rules(id)
        ON DELETE SET NULL;

ALTER TABLE validation_rule_service.validation_rule_set_members
    ADD CONSTRAINT chk_vrsm_position_non_negative CHECK (position >= 0);

ALTER TABLE validation_rule_service.validation_rule_set_members
    ADD CONSTRAINT chk_vrsm_override_severity CHECK (
        override_severity IS NULL OR override_severity IN (
            'info', 'warning', 'error', 'critical'
        )
    );

-- =============================================================================
-- Table 5: validation_rule_service.validation_rule_versions
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_rule_versions (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id         UUID         NOT NULL,
    version         INTEGER      NOT NULL,
    changes         JSONB        NOT NULL DEFAULT '{}'::jsonb,
    snapshot        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_rule_versions
    ADD CONSTRAINT fk_vrv_rule_id
        FOREIGN KEY (rule_id)
        REFERENCES validation_rule_service.validation_rules(id)
        ON DELETE CASCADE;

ALTER TABLE validation_rule_service.validation_rule_versions
    ADD CONSTRAINT chk_vrv_version_positive CHECK (version >= 1);

-- =============================================================================
-- Table 6: validation_rule_service.validation_rule_packs
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_rule_packs (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    pack_name        VARCHAR(255)  NOT NULL,
    pack_type        VARCHAR(50)   NOT NULL,
    version          VARCHAR(50)   NOT NULL DEFAULT '1.0.0',
    description      TEXT          NOT NULL DEFAULT '',
    rule_definitions JSONB         NOT NULL DEFAULT '[]'::jsonb,
    metadata         JSONB         NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_rule_packs
    ADD CONSTRAINT uq_vrp_pack_name UNIQUE (pack_name);

ALTER TABLE validation_rule_service.validation_rule_packs
    ADD CONSTRAINT chk_vrp_pack_name_not_empty CHECK (LENGTH(TRIM(pack_name)) > 0);

ALTER TABLE validation_rule_service.validation_rule_packs
    ADD CONSTRAINT chk_vrp_pack_type CHECK (pack_type IN (
        'regulatory', 'industry', 'custom', 'framework', 'template',
        'ghg_protocol', 'csrd_esrs', 'iso_14064', 'cdp'
    ));

CREATE TRIGGER trg_vrp_updated_at
    BEFORE UPDATE ON validation_rule_service.validation_rule_packs
    FOR EACH ROW EXECUTE FUNCTION validation_rule_service.set_updated_at();

-- =============================================================================
-- Table 7: validation_rule_service.validation_evaluations
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_evaluations (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_set_id     UUID          NOT NULL,
    dataset_name    VARCHAR(500)  NOT NULL,
    total_rules     INTEGER       NOT NULL DEFAULT 0,
    passed          INTEGER       NOT NULL DEFAULT 0,
    failed          INTEGER       NOT NULL DEFAULT 0,
    warned          INTEGER       NOT NULL DEFAULT 0,
    pass_rate       NUMERIC(5,4)  NOT NULL DEFAULT 0.0000,
    sla_result      VARCHAR(20)   NOT NULL DEFAULT 'unknown',
    duration_ms     NUMERIC       NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(128)  NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT fk_ve_rule_set_id
        FOREIGN KEY (rule_set_id)
        REFERENCES validation_rule_service.validation_rule_sets(id)
        ON DELETE CASCADE;

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_dataset_name_not_empty CHECK (LENGTH(TRIM(dataset_name)) > 0);

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_total_rules_non_negative CHECK (total_rules >= 0);

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_passed_non_negative CHECK (passed >= 0);

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_failed_non_negative CHECK (failed >= 0);

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_warned_non_negative CHECK (warned >= 0);

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_pass_rate_range CHECK (pass_rate >= 0.0000 AND pass_rate <= 1.0000);

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_sla_result CHECK (sla_result IN (
        'pass', 'fail', 'warn', 'unknown'
    ));

ALTER TABLE validation_rule_service.validation_evaluations
    ADD CONSTRAINT chk_ve_duration_ms_non_negative CHECK (duration_ms >= 0);

-- =============================================================================
-- Table 8: validation_rule_service.validation_evaluation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_evaluation_details (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    evaluation_id   UUID          NOT NULL,
    rule_id         UUID          NOT NULL,
    pass_count      INTEGER       NOT NULL DEFAULT 0,
    fail_count      INTEGER       NOT NULL DEFAULT 0,
    total           INTEGER       NOT NULL DEFAULT 0,
    pass_rate       NUMERIC(5,4)  NOT NULL DEFAULT 0.0000,
    result          VARCHAR(20)   NOT NULL DEFAULT 'unknown',
    failures        JSONB         NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_evaluation_details
    ADD CONSTRAINT fk_ved_evaluation_id
        FOREIGN KEY (evaluation_id)
        REFERENCES validation_rule_service.validation_evaluations(id)
        ON DELETE CASCADE;

ALTER TABLE validation_rule_service.validation_evaluation_details
    ADD CONSTRAINT fk_ved_rule_id
        FOREIGN KEY (rule_id)
        REFERENCES validation_rule_service.validation_rules(id)
        ON DELETE CASCADE;

ALTER TABLE validation_rule_service.validation_evaluation_details
    ADD CONSTRAINT chk_ved_pass_count_non_negative CHECK (pass_count >= 0);

ALTER TABLE validation_rule_service.validation_evaluation_details
    ADD CONSTRAINT chk_ved_fail_count_non_negative CHECK (fail_count >= 0);

ALTER TABLE validation_rule_service.validation_evaluation_details
    ADD CONSTRAINT chk_ved_total_non_negative CHECK (total >= 0);

ALTER TABLE validation_rule_service.validation_evaluation_details
    ADD CONSTRAINT chk_ved_pass_rate_range CHECK (pass_rate >= 0.0000 AND pass_rate <= 1.0000);

ALTER TABLE validation_rule_service.validation_evaluation_details
    ADD CONSTRAINT chk_ved_result CHECK (result IN (
        'pass', 'fail', 'warn', 'skip', 'error', 'unknown'
    ));

-- =============================================================================
-- Table 9: validation_rule_service.validation_reports
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_reports (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type     VARCHAR(50)   NOT NULL,
    format          VARCHAR(20)   NOT NULL,
    content         TEXT          NOT NULL DEFAULT '',
    report_hash     VARCHAR(128),
    parameters      JSONB         NOT NULL DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128)  NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_reports
    ADD CONSTRAINT chk_vrpt_report_type CHECK (report_type IN (
        'evaluation_summary', 'rule_coverage', 'trend_analysis',
        'sla_compliance', 'failure_analysis', 'custom'
    ));

ALTER TABLE validation_rule_service.validation_reports
    ADD CONSTRAINT chk_vrpt_format CHECK (format IN (
        'json', 'text', 'html', 'pdf', 'csv', 'markdown'
    ));

-- =============================================================================
-- Table 10: validation_rule_service.validation_audit_log
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_audit_log (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type     VARCHAR(100)  NOT NULL,
    entity_id       UUID,
    action          VARCHAR(100)  NOT NULL,
    actor           VARCHAR(255)  NOT NULL DEFAULT 'system',
    changes         JSONB         NOT NULL DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128)  NOT NULL DEFAULT '',
    parent_hash     VARCHAR(128)  NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE validation_rule_service.validation_audit_log
    ADD CONSTRAINT chk_val_action_not_empty CHECK (LENGTH(TRIM(action)) > 0);

ALTER TABLE validation_rule_service.validation_audit_log
    ADD CONSTRAINT chk_val_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE validation_rule_service.validation_audit_log
    ADD CONSTRAINT chk_val_action CHECK (action IN (
        'rule_created', 'rule_updated', 'rule_activated', 'rule_deactivated',
        'rule_deprecated', 'rule_archived', 'rule_deleted',
        'set_created', 'set_updated', 'set_activated', 'set_deactivated',
        'set_deprecated', 'set_archived', 'set_deleted',
        'member_added', 'member_removed', 'member_reordered',
        'compound_created', 'compound_updated', 'compound_deleted',
        'pack_created', 'pack_updated', 'pack_imported', 'pack_exported',
        'evaluation_started', 'evaluation_completed', 'evaluation_failed',
        'report_generated', 'report_exported',
        'config_changed'
    ));

ALTER TABLE validation_rule_service.validation_audit_log
    ADD CONSTRAINT chk_val_entity_type CHECK (entity_type IN (
        'rule', 'rule_set', 'rule_set_member', 'rule_version',
        'compound_rule', 'rule_pack', 'evaluation', 'evaluation_detail',
        'report', 'config'
    ));

-- =============================================================================
-- Table 11: validation_rule_service.validation_evaluation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_evaluation_events (
    ts              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    rule_set_id     UUID,
    dataset_name    VARCHAR(500),
    rule_type       VARCHAR(50),
    severity        VARCHAR(20),
    result          VARCHAR(20),
    pass_rate       NUMERIC(5,4) NOT NULL DEFAULT 0.0000,
    duration_ms     NUMERIC      NOT NULL DEFAULT 0
);

SELECT create_hypertable(
    'validation_rule_service.validation_evaluation_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE validation_rule_service.validation_evaluation_events
    ADD CONSTRAINT chk_vee_pass_rate_range CHECK (pass_rate >= 0.0000 AND pass_rate <= 1.0000);

ALTER TABLE validation_rule_service.validation_evaluation_events
    ADD CONSTRAINT chk_vee_duration_ms_non_negative CHECK (duration_ms >= 0);

ALTER TABLE validation_rule_service.validation_evaluation_events
    ADD CONSTRAINT chk_vee_result CHECK (
        result IS NULL OR result IN ('pass', 'fail', 'warn', 'skip', 'error', 'unknown')
    );

ALTER TABLE validation_rule_service.validation_evaluation_events
    ADD CONSTRAINT chk_vee_severity CHECK (
        severity IS NULL OR severity IN ('info', 'warning', 'error', 'critical')
    );

-- =============================================================================
-- Table 12: validation_rule_service.validation_rule_change_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_rule_change_events (
    ts              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    rule_id         UUID,
    action          VARCHAR(100),
    rule_type       VARCHAR(50),
    severity        VARCHAR(20),
    actor           VARCHAR(255)
);

SELECT create_hypertable(
    'validation_rule_service.validation_rule_change_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE validation_rule_service.validation_rule_change_events
    ADD CONSTRAINT chk_vrce_severity CHECK (
        severity IS NULL OR severity IN ('info', 'warning', 'error', 'critical')
    );

-- =============================================================================
-- Table 13: validation_rule_service.validation_conflict_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_rule_service.validation_conflict_events (
    ts              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    conflict_type   VARCHAR(50)  NOT NULL,
    rule_a_id       UUID,
    rule_b_id       UUID,
    column_name     VARCHAR(255),
    severity        VARCHAR(20)  NOT NULL DEFAULT 'warning'
);

SELECT create_hypertable(
    'validation_rule_service.validation_conflict_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE validation_rule_service.validation_conflict_events
    ADD CONSTRAINT chk_vce_conflict_type CHECK (conflict_type IN (
        'contradictory', 'overlapping', 'redundant', 'subsumption', 'ordering'
    ));

ALTER TABLE validation_rule_service.validation_conflict_events
    ADD CONSTRAINT chk_vce_severity CHECK (severity IN (
        'info', 'warning', 'error', 'critical'
    ));

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- validation_evaluations_hourly_stats: hourly rollup of validation_evaluation_events
CREATE MATERIALIZED VIEW validation_rule_service.validation_evaluations_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts)      AS bucket,
    rule_type,
    result,
    COUNT(*)                        AS total_events,
    AVG(pass_rate)                  AS avg_pass_rate,
    AVG(duration_ms)                AS avg_duration_ms,
    SUM(duration_ms)                AS total_duration_ms
FROM validation_rule_service.validation_evaluation_events
WHERE ts IS NOT NULL
GROUP BY bucket, rule_type, result
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'validation_rule_service.validation_evaluations_hourly_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- validation_rule_changes_hourly_stats: hourly rollup of validation_rule_change_events
CREATE MATERIALIZED VIEW validation_rule_service.validation_rule_changes_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts)      AS bucket,
    action,
    rule_type,
    COUNT(*)                        AS total_events
FROM validation_rule_service.validation_rule_change_events
WHERE ts IS NOT NULL
GROUP BY bucket, action, rule_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'validation_rule_service.validation_rule_changes_hourly_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- validation_rules indexes (10)
CREATE INDEX IF NOT EXISTS idx_vr_name                 ON validation_rule_service.validation_rules(name);
CREATE INDEX IF NOT EXISTS idx_vr_rule_type            ON validation_rule_service.validation_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_vr_column_name          ON validation_rule_service.validation_rules(column_name);
CREATE INDEX IF NOT EXISTS idx_vr_operator             ON validation_rule_service.validation_rules(operator);
CREATE INDEX IF NOT EXISTS idx_vr_severity             ON validation_rule_service.validation_rules(severity);
CREATE INDEX IF NOT EXISTS idx_vr_status               ON validation_rule_service.validation_rules(status);
CREATE INDEX IF NOT EXISTS idx_vr_created_at           ON validation_rule_service.validation_rules(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vr_updated_at           ON validation_rule_service.validation_rules(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_vr_tags                 ON validation_rule_service.validation_rules USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_vr_metadata             ON validation_rule_service.validation_rules USING GIN (metadata);

-- validation_rule_sets indexes (10)
CREATE INDEX IF NOT EXISTS idx_vrs_name                ON validation_rule_service.validation_rule_sets(name);
CREATE INDEX IF NOT EXISTS idx_vrs_status              ON validation_rule_service.validation_rule_sets(status);
CREATE INDEX IF NOT EXISTS idx_vrs_parent_set_id       ON validation_rule_service.validation_rule_sets(parent_set_id);
CREATE INDEX IF NOT EXISTS idx_vrs_created_by          ON validation_rule_service.validation_rule_sets(created_by);
CREATE INDEX IF NOT EXISTS idx_vrs_created_at          ON validation_rule_service.validation_rule_sets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrs_updated_at          ON validation_rule_service.validation_rule_sets(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrs_status_created      ON validation_rule_service.validation_rule_sets(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrs_tags                ON validation_rule_service.validation_rule_sets USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_vrs_metadata            ON validation_rule_service.validation_rule_sets USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_vrs_sla_thresholds      ON validation_rule_service.validation_rule_sets USING GIN (sla_thresholds);

-- validation_rule_set_members indexes (8)
CREATE INDEX IF NOT EXISTS idx_vrsm_set_id             ON validation_rule_service.validation_rule_set_members(set_id);
CREATE INDEX IF NOT EXISTS idx_vrsm_rule_id            ON validation_rule_service.validation_rule_set_members(rule_id);
CREATE INDEX IF NOT EXISTS idx_vrsm_compound_rule_id   ON validation_rule_service.validation_rule_set_members(compound_rule_id);
CREATE INDEX IF NOT EXISTS idx_vrsm_position           ON validation_rule_service.validation_rule_set_members(position);
CREATE INDEX IF NOT EXISTS idx_vrsm_added_at           ON validation_rule_service.validation_rule_set_members(added_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrsm_set_position       ON validation_rule_service.validation_rule_set_members(set_id, position);
CREATE INDEX IF NOT EXISTS idx_vrsm_set_rule           ON validation_rule_service.validation_rule_set_members(set_id, rule_id);
CREATE INDEX IF NOT EXISTS idx_vrsm_override_severity  ON validation_rule_service.validation_rule_set_members(override_severity);

-- validation_rule_versions indexes (8)
CREATE INDEX IF NOT EXISTS idx_vrv_rule_id             ON validation_rule_service.validation_rule_versions(rule_id);
CREATE INDEX IF NOT EXISTS idx_vrv_version             ON validation_rule_service.validation_rule_versions(version);
CREATE INDEX IF NOT EXISTS idx_vrv_created_at          ON validation_rule_service.validation_rule_versions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrv_rule_version        ON validation_rule_service.validation_rule_versions(rule_id, version);
CREATE INDEX IF NOT EXISTS idx_vrv_rule_created        ON validation_rule_service.validation_rule_versions(rule_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrv_changes             ON validation_rule_service.validation_rule_versions USING GIN (changes);
CREATE INDEX IF NOT EXISTS idx_vrv_snapshot            ON validation_rule_service.validation_rule_versions USING GIN (snapshot);
CREATE INDEX IF NOT EXISTS idx_vrv_version_desc        ON validation_rule_service.validation_rule_versions(version DESC);

-- validation_compound_rules indexes (8)
CREATE INDEX IF NOT EXISTS idx_vcr_name                ON validation_rule_service.validation_compound_rules(name);
CREATE INDEX IF NOT EXISTS idx_vcr_operator            ON validation_rule_service.validation_compound_rules(operator);
CREATE INDEX IF NOT EXISTS idx_vcr_created_at          ON validation_rule_service.validation_compound_rules(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vcr_updated_at          ON validation_rule_service.validation_compound_rules(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_vcr_child_rule_ids      ON validation_rule_service.validation_compound_rules USING GIN (child_rule_ids);
CREATE INDEX IF NOT EXISTS idx_vcr_child_compound_ids  ON validation_rule_service.validation_compound_rules USING GIN (child_compound_ids);
CREATE INDEX IF NOT EXISTS idx_vcr_operator_created    ON validation_rule_service.validation_compound_rules(operator, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vcr_name_operator       ON validation_rule_service.validation_compound_rules(name, operator);

-- validation_rule_packs indexes (8)
CREATE INDEX IF NOT EXISTS idx_vrp_pack_name           ON validation_rule_service.validation_rule_packs(pack_name);
CREATE INDEX IF NOT EXISTS idx_vrp_pack_type           ON validation_rule_service.validation_rule_packs(pack_type);
CREATE INDEX IF NOT EXISTS idx_vrp_version             ON validation_rule_service.validation_rule_packs(version);
CREATE INDEX IF NOT EXISTS idx_vrp_created_at          ON validation_rule_service.validation_rule_packs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrp_updated_at          ON validation_rule_service.validation_rule_packs(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrp_type_version        ON validation_rule_service.validation_rule_packs(pack_type, version);
CREATE INDEX IF NOT EXISTS idx_vrp_rule_definitions    ON validation_rule_service.validation_rule_packs USING GIN (rule_definitions);
CREATE INDEX IF NOT EXISTS idx_vrp_metadata            ON validation_rule_service.validation_rule_packs USING GIN (metadata);

-- validation_evaluations indexes (10)
CREATE INDEX IF NOT EXISTS idx_ve_rule_set_id          ON validation_rule_service.validation_evaluations(rule_set_id);
CREATE INDEX IF NOT EXISTS idx_ve_dataset_name         ON validation_rule_service.validation_evaluations(dataset_name);
CREATE INDEX IF NOT EXISTS idx_ve_sla_result           ON validation_rule_service.validation_evaluations(sla_result);
CREATE INDEX IF NOT EXISTS idx_ve_pass_rate            ON validation_rule_service.validation_evaluations(pass_rate DESC);
CREATE INDEX IF NOT EXISTS idx_ve_created_at           ON validation_rule_service.validation_evaluations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ve_duration_ms          ON validation_rule_service.validation_evaluations(duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_ve_provenance_hash      ON validation_rule_service.validation_evaluations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_ve_ruleset_created      ON validation_rule_service.validation_evaluations(rule_set_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ve_dataset_created      ON validation_rule_service.validation_evaluations(dataset_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ve_sla_created          ON validation_rule_service.validation_evaluations(sla_result, created_at DESC);

-- validation_evaluation_details indexes (10)
CREATE INDEX IF NOT EXISTS idx_ved_evaluation_id       ON validation_rule_service.validation_evaluation_details(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_ved_rule_id             ON validation_rule_service.validation_evaluation_details(rule_id);
CREATE INDEX IF NOT EXISTS idx_ved_result              ON validation_rule_service.validation_evaluation_details(result);
CREATE INDEX IF NOT EXISTS idx_ved_pass_rate           ON validation_rule_service.validation_evaluation_details(pass_rate DESC);
CREATE INDEX IF NOT EXISTS idx_ved_created_at          ON validation_rule_service.validation_evaluation_details(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ved_eval_rule           ON validation_rule_service.validation_evaluation_details(evaluation_id, rule_id);
CREATE INDEX IF NOT EXISTS idx_ved_eval_result         ON validation_rule_service.validation_evaluation_details(evaluation_id, result);
CREATE INDEX IF NOT EXISTS idx_ved_rule_result         ON validation_rule_service.validation_evaluation_details(rule_id, result);
CREATE INDEX IF NOT EXISTS idx_ved_fail_count          ON validation_rule_service.validation_evaluation_details(fail_count DESC);
CREATE INDEX IF NOT EXISTS idx_ved_failures            ON validation_rule_service.validation_evaluation_details USING GIN (failures);

-- validation_reports indexes (8)
CREATE INDEX IF NOT EXISTS idx_vrpt_report_type        ON validation_rule_service.validation_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_vrpt_format             ON validation_rule_service.validation_reports(format);
CREATE INDEX IF NOT EXISTS idx_vrpt_created_at         ON validation_rule_service.validation_reports(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrpt_report_hash        ON validation_rule_service.validation_reports(report_hash);
CREATE INDEX IF NOT EXISTS idx_vrpt_provenance_hash    ON validation_rule_service.validation_reports(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_vrpt_type_format        ON validation_rule_service.validation_reports(report_type, format);
CREATE INDEX IF NOT EXISTS idx_vrpt_type_created       ON validation_rule_service.validation_reports(report_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vrpt_parameters         ON validation_rule_service.validation_reports USING GIN (parameters);

-- validation_audit_log indexes (10)
CREATE INDEX IF NOT EXISTS idx_val_action              ON validation_rule_service.validation_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_val_entity_type         ON validation_rule_service.validation_audit_log(entity_type);
CREATE INDEX IF NOT EXISTS idx_val_entity_id           ON validation_rule_service.validation_audit_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_val_actor               ON validation_rule_service.validation_audit_log(actor);
CREATE INDEX IF NOT EXISTS idx_val_created_at          ON validation_rule_service.validation_audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_val_provenance_hash     ON validation_rule_service.validation_audit_log(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_val_parent_hash         ON validation_rule_service.validation_audit_log(parent_hash);
CREATE INDEX IF NOT EXISTS idx_val_entity_type_id      ON validation_rule_service.validation_audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_val_action_created      ON validation_rule_service.validation_audit_log(action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_val_changes             ON validation_rule_service.validation_audit_log USING GIN (changes);

-- validation_evaluation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_vee_rule_set_id         ON validation_rule_service.validation_evaluation_events(rule_set_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vee_dataset_name        ON validation_rule_service.validation_evaluation_events(dataset_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vee_rule_type           ON validation_rule_service.validation_evaluation_events(rule_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vee_severity            ON validation_rule_service.validation_evaluation_events(severity, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vee_result              ON validation_rule_service.validation_evaluation_events(result, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vee_ruleset_result      ON validation_rule_service.validation_evaluation_events(rule_set_id, result, ts DESC);

-- validation_rule_change_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_vrce_rule_id            ON validation_rule_service.validation_rule_change_events(rule_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vrce_action             ON validation_rule_service.validation_rule_change_events(action, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vrce_rule_type          ON validation_rule_service.validation_rule_change_events(rule_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vrce_severity           ON validation_rule_service.validation_rule_change_events(severity, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vrce_actor              ON validation_rule_service.validation_rule_change_events(actor, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vrce_rule_action        ON validation_rule_service.validation_rule_change_events(rule_id, action, ts DESC);

-- validation_conflict_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_vce_conflict_type       ON validation_rule_service.validation_conflict_events(conflict_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vce_rule_a_id           ON validation_rule_service.validation_conflict_events(rule_a_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vce_rule_b_id           ON validation_rule_service.validation_conflict_events(rule_b_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vce_column_name         ON validation_rule_service.validation_conflict_events(column_name, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vce_severity            ON validation_rule_service.validation_conflict_events(severity, ts DESC);
CREATE INDEX IF NOT EXISTS idx_vce_type_severity       ON validation_rule_service.validation_conflict_events(conflict_type, severity, ts DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- validation_rules: global resource -- no tenant_id; admin-gated writes via is_admin
ALTER TABLE validation_rule_service.validation_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY vr_read  ON validation_rule_service.validation_rules FOR SELECT USING (TRUE);
CREATE POLICY vr_write ON validation_rule_service.validation_rules FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_rule_sets ENABLE ROW LEVEL SECURITY;
CREATE POLICY vrs_read  ON validation_rule_service.validation_rule_sets FOR SELECT USING (TRUE);
CREATE POLICY vrs_write ON validation_rule_service.validation_rule_sets FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_rule_set_members ENABLE ROW LEVEL SECURITY;
CREATE POLICY vrsm_read  ON validation_rule_service.validation_rule_set_members FOR SELECT USING (TRUE);
CREATE POLICY vrsm_write ON validation_rule_service.validation_rule_set_members FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_rule_versions ENABLE ROW LEVEL SECURITY;
CREATE POLICY vrv_read  ON validation_rule_service.validation_rule_versions FOR SELECT USING (TRUE);
CREATE POLICY vrv_write ON validation_rule_service.validation_rule_versions FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_compound_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY vcr_read  ON validation_rule_service.validation_compound_rules FOR SELECT USING (TRUE);
CREATE POLICY vcr_write ON validation_rule_service.validation_compound_rules FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_rule_packs ENABLE ROW LEVEL SECURITY;
CREATE POLICY vrp_read  ON validation_rule_service.validation_rule_packs FOR SELECT USING (TRUE);
CREATE POLICY vrp_write ON validation_rule_service.validation_rule_packs FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_evaluations ENABLE ROW LEVEL SECURITY;
CREATE POLICY ve_read  ON validation_rule_service.validation_evaluations FOR SELECT USING (TRUE);
CREATE POLICY ve_write ON validation_rule_service.validation_evaluations FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_evaluation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY ved_read  ON validation_rule_service.validation_evaluation_details FOR SELECT USING (TRUE);
CREATE POLICY ved_write ON validation_rule_service.validation_evaluation_details FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY vrpt_read  ON validation_rule_service.validation_reports FOR SELECT USING (TRUE);
CREATE POLICY vrpt_write ON validation_rule_service.validation_reports FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY val_read  ON validation_rule_service.validation_audit_log FOR SELECT USING (TRUE);
CREATE POLICY val_write ON validation_rule_service.validation_audit_log FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE validation_rule_service.validation_evaluation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY vee_read  ON validation_rule_service.validation_evaluation_events FOR SELECT USING (TRUE);
CREATE POLICY vee_write ON validation_rule_service.validation_evaluation_events FOR ALL   USING (TRUE);

ALTER TABLE validation_rule_service.validation_rule_change_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY vrce_read  ON validation_rule_service.validation_rule_change_events FOR SELECT USING (TRUE);
CREATE POLICY vrce_write ON validation_rule_service.validation_rule_change_events FOR ALL   USING (TRUE);

ALTER TABLE validation_rule_service.validation_conflict_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY vce_read  ON validation_rule_service.validation_conflict_events FOR SELECT USING (TRUE);
CREATE POLICY vce_write ON validation_rule_service.validation_conflict_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA validation_rule_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA validation_rule_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA validation_rule_service TO greenlang_app;
GRANT SELECT ON validation_rule_service.validation_evaluations_hourly_stats TO greenlang_app;
GRANT SELECT ON validation_rule_service.validation_rule_changes_hourly_stats TO greenlang_app;

GRANT USAGE ON SCHEMA validation_rule_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA validation_rule_service TO greenlang_readonly;
GRANT SELECT ON validation_rule_service.validation_evaluations_hourly_stats TO greenlang_readonly;
GRANT SELECT ON validation_rule_service.validation_rule_changes_hourly_stats TO greenlang_readonly;

GRANT ALL ON SCHEMA validation_rule_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA validation_rule_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA validation_rule_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'validation-rules:rules:read',          'validation-rules', 'rules_read',          'View validation rule definitions, types, operators, and thresholds'),
    (gen_random_uuid(), 'validation-rules:rules:write',         'validation-rules', 'rules_write',         'Create, update, activate, deactivate, and archive validation rules'),
    (gen_random_uuid(), 'validation-rules:sets:read',           'validation-rules', 'sets_read',           'View validation rule sets, SLA thresholds, and hierarchies'),
    (gen_random_uuid(), 'validation-rules:sets:write',          'validation-rules', 'sets_write',          'Create, update, and manage rule sets and member ordering'),
    (gen_random_uuid(), 'validation-rules:members:read',        'validation-rules', 'members_read',        'View rule set membership, positions, and severity overrides'),
    (gen_random_uuid(), 'validation-rules:members:write',       'validation-rules', 'members_write',       'Add, remove, and reorder rules within rule sets'),
    (gen_random_uuid(), 'validation-rules:compounds:read',      'validation-rules', 'compounds_read',      'View compound boolean rules and their child compositions'),
    (gen_random_uuid(), 'validation-rules:compounds:write',     'validation-rules', 'compounds_write',     'Create, update, and delete compound boolean rules (AND/OR/NOT)'),
    (gen_random_uuid(), 'validation-rules:packs:read',          'validation-rules', 'packs_read',          'View rule packs, their types, versions, and rule definitions'),
    (gen_random_uuid(), 'validation-rules:packs:write',         'validation-rules', 'packs_write',         'Create, update, import, and export rule packs'),
    (gen_random_uuid(), 'validation-rules:evaluations:read',    'validation-rules', 'evaluations_read',    'View evaluation results, pass rates, SLA outcomes, and durations'),
    (gen_random_uuid(), 'validation-rules:evaluations:write',   'validation-rules', 'evaluations_write',   'Execute evaluations against datasets with rule sets'),
    (gen_random_uuid(), 'validation-rules:details:read',        'validation-rules', 'details_read',        'View per-rule evaluation detail records and failure breakdowns'),
    (gen_random_uuid(), 'validation-rules:details:write',       'validation-rules', 'details_write',       'Record per-rule evaluation detail results'),
    (gen_random_uuid(), 'validation-rules:reports:read',        'validation-rules', 'reports_read',        'View generated validation reports and their content'),
    (gen_random_uuid(), 'validation-rules:reports:write',       'validation-rules', 'reports_write',       'Generate evaluation summary, coverage, trend, and SLA compliance reports'),
    (gen_random_uuid(), 'validation-rules:versions:read',       'validation-rules', 'versions_read',       'View rule version history, change diffs, and snapshots'),
    (gen_random_uuid(), 'validation-rules:versions:write',      'validation-rules', 'versions_write',      'Create rule version snapshots and track changes'),
    (gen_random_uuid(), 'validation-rules:audit:read',          'validation-rules', 'audit_read',          'View validation rule audit log entries and provenance chains'),
    (gen_random_uuid(), 'validation-rules:admin',               'validation-rules', 'admin',               'Validation rule engine service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('validation_rule_service.validation_evaluation_events', INTERVAL '90 days');
SELECT add_retention_policy('validation_rule_service.validation_rule_change_events', INTERVAL '90 days');
SELECT add_retention_policy('validation_rule_service.validation_conflict_events',    INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE validation_rule_service.validation_evaluation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'rule_set_id',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('validation_rule_service.validation_evaluation_events', INTERVAL '7 days');

ALTER TABLE validation_rule_service.validation_rule_change_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'rule_id',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('validation_rule_service.validation_rule_change_events', INTERVAL '7 days');

ALTER TABLE validation_rule_service.validation_conflict_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'conflict_type',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('validation_rule_service.validation_conflict_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Validation Rule Engine (GL-DATA-X-022)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-DATA-X-022',
    'Validation Rule Engine',
    'Validation rule engine for GreenLang Climate OS. Manages validation rule definitions with configurable rule types (not_null/unique/range/regex/enum/type_check/length/custom/cross_field/referential/statistical/temporal/format/completeness), operators (eq/ne/gt/gte/lt/lte/between/in/not_in/is_null/is_not_null/matches/not_matches/contains/starts_with/ends_with/is_unique/custom), thresholds, severity levels (info/warning/error/critical), and status lifecycle (active/inactive/draft/deprecated/archived). Supports hierarchical rule sets with parent-child inheritance, SLA thresholds, and ordered member positioning. Implements compound boolean logic (AND/OR/NOT) for complex multi-rule conditions with nested composition. Provides versioned rule packs (regulatory/industry/custom/framework/template/ghg_protocol/csrd_esrs/iso_14064/cdp) for bulk distribution and import/export. Executes evaluations per rule set against named datasets with pass/fail/warn counting, pass rate calculation (NUMERIC 5,4 precision), SLA result determination, and duration tracking. Captures per-rule evaluation details with individual pass/fail counts, result statuses, and structured failure records. Generates validation reports (evaluation_summary/rule_coverage/trend_analysis/sla_compliance/failure_analysis/custom) in multiple formats (json/text/html/pdf/csv/markdown) with content hashing. Tracks rule version history with change diffs and full state snapshots. Detects rule conflicts (contradictory/overlapping/redundant/subsumption/ordering) between rules on the same column. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 5, '1.0.0', true,
    'GreenLang Data Team',
    'https://docs.greenlang.ai/agents/validation-rule-engine',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-DATA-X-022', '1.0.0',
    '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/validation-rule-engine-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"validation", "rule-engine", "data-quality", "sla-compliance", "compound-rules", "rule-packs"}',
    '{"cross-sector", "manufacturing", "retail", "energy", "finance", "agriculture", "utilities"}',
    'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-DATA-X-022', '1.0.0',
    'rule_management',
    'configuration',
    'Create, update, and manage validation rules with configurable types, operators, thresholds, severity, and lifecycle status.',
    '{"name", "rule_type", "column_name", "operator", "threshold", "parameters", "severity", "tags"}',
    '{"rule_id", "version", "validation_result"}',
    '{"rule_types": ["not_null", "unique", "range", "regex", "enum", "type_check", "length", "custom", "cross_field", "referential", "statistical", "temporal", "format", "completeness"], "operators": ["eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "not_in", "is_null", "is_not_null", "matches", "not_matches", "contains", "starts_with", "ends_with", "is_unique", "custom"], "severities": ["info", "warning", "error", "critical"]}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'rule_set_composition',
    'configuration',
    'Compose rule sets with ordered members, SLA thresholds, hierarchical parent-child sets, and severity overrides.',
    '{"name", "description", "sla_thresholds", "parent_set_id", "rules", "tags"}',
    '{"set_id", "member_count", "validation_result"}',
    '{"supports_hierarchy": true, "sla_fields": ["min_pass_rate", "max_failures", "max_warnings"], "severity_override": true}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'compound_logic',
    'configuration',
    'Build compound boolean rules using AND/OR/NOT operators with nested child rules and compound compositions.',
    '{"name", "operator", "child_rule_ids", "child_compound_ids", "description"}',
    '{"compound_rule_id", "validation_result"}',
    '{"operators": ["AND", "OR", "NOT"], "supports_nesting": true, "max_depth": 10}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'rule_pack_distribution',
    'configuration',
    'Create and distribute versioned rule packs for regulatory, industry, and custom validation frameworks.',
    '{"pack_name", "pack_type", "version", "rule_definitions", "description"}',
    '{"pack_id", "rule_count", "import_result"}',
    '{"pack_types": ["regulatory", "industry", "custom", "framework", "template", "ghg_protocol", "csrd_esrs", "iso_14064", "cdp"], "supports_import_export": true}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'evaluation_execution',
    'processing',
    'Execute rule set evaluations against datasets with pass/fail/warn counting, pass rate calculation, and SLA determination.',
    '{"rule_set_id", "dataset_name", "dataset_records"}',
    '{"evaluation_id", "total_rules", "passed", "failed", "warned", "pass_rate", "sla_result", "duration_ms", "provenance_hash"}',
    '{"sla_results": ["pass", "fail", "warn", "unknown"], "pass_rate_precision": "NUMERIC(5,4)", "tracks_duration": true}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'evaluation_detail',
    'processing',
    'Capture per-rule evaluation details with individual pass/fail counts, result statuses, and structured failure records.',
    '{"evaluation_id", "rule_id", "records"}',
    '{"detail_id", "pass_count", "fail_count", "total", "pass_rate", "result", "failures"}',
    '{"results": ["pass", "fail", "warn", "skip", "error", "unknown"], "failure_capture": true}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'report_generation',
    'reporting',
    'Generate validation reports including evaluation summaries, rule coverage, trend analysis, SLA compliance, and failure analysis.',
    '{"report_type", "format", "parameters"}',
    '{"report_id", "content", "report_hash", "provenance_hash"}',
    '{"report_types": ["evaluation_summary", "rule_coverage", "trend_analysis", "sla_compliance", "failure_analysis", "custom"], "formats": ["json", "text", "html", "pdf", "csv", "markdown"]}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'version_tracking',
    'configuration',
    'Track rule version history with change diffs and full state snapshots for audit and rollback.',
    '{"rule_id", "version", "changes"}',
    '{"version_id", "snapshot", "created_at"}',
    '{"captures_diffs": true, "captures_snapshots": true, "supports_rollback": true}'::jsonb
),
(
    'GL-DATA-X-022', '1.0.0',
    'conflict_detection',
    'analysis',
    'Detect conflicts between validation rules on the same column including contradictory, overlapping, redundant, and subsumption patterns.',
    '{"rule_ids", "column_name"}',
    '{"conflicts", "conflict_count", "severity_distribution"}',
    '{"conflict_types": ["contradictory", "overlapping", "redundant", "subsumption", "ordering"], "severities": ["info", "warning", "error", "critical"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-DATA-X-022', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-step evaluation pipeline execution ordering'),
    ('GL-DATA-X-022', 'GL-FOUND-X-007', '>=1.0.0', false, 'Agent version and capability lookup for rule engine service registration'),
    ('GL-DATA-X-022', 'GL-FOUND-X-006', '>=1.0.0', false, 'Access control enforcement for rule management and evaluation permissions'),
    ('GL-DATA-X-022', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for evaluation performance, rule change tracking, and conflict events'),
    ('GL-DATA-X-022', 'GL-FOUND-X-005', '>=1.0.0', true,  'Provenance and audit trail registration with citation service'),
    ('GL-DATA-X-022', 'GL-FOUND-X-008', '>=1.0.0', true,  'Reproducibility verification for deterministic evaluation result hashing'),
    ('GL-DATA-X-022', 'GL-FOUND-X-009', '>=1.0.0', true,  'QA Test Harness zero-hallucination verification of evaluation outputs'),
    ('GL-DATA-X-022', 'GL-DATA-X-013', '>=1.0.0',  true,  'Data quality profiling integration for rule-based quality gate enforcement'),
    ('GL-DATA-X-022', 'GL-DATA-X-021', '>=1.0.0',  true,  'Data lineage tracker for validation rule provenance and evaluation lineage')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-DATA-X-022',
    'Validation Rule Engine',
    'Validation rule engine. Rule management (14 rule types, 18 operators, 4 severities, 5 lifecycle statuses). Rule set composition (hierarchical sets, SLA thresholds, ordered members, severity overrides). Compound boolean logic (AND/OR/NOT, nested composition). Rule packs (9 pack types, versioned, import/export). Evaluation execution (pass/fail/warn counting, NUMERIC(5,4) pass rate, SLA determination, duration tracking). Per-rule evaluation details (individual results, failure records). Report generation (6 report types, 6 output formats, content hashing). Version tracking (change diffs, full snapshots, rollback). Conflict detection (5 conflict types, same-column analysis). SHA-256 provenance chains.',
    'data', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA validation_rule_service IS
    'Validation Rule Engine (AGENT-DATA-019) - rule management, rule set composition, compound boolean logic, rule packs, evaluation execution, evaluation details, report generation, version tracking, conflict detection, provenance chains';

COMMENT ON TABLE validation_rule_service.validation_rules IS
    'Validation rule definitions: name (unique), rule_type (14 types), column_name, operator (18 types), threshold, parameters JSONB, severity (4 levels), status (5 states), version, description, tags, metadata, created_by';

COMMENT ON TABLE validation_rule_service.validation_rule_sets IS
    'Rule set containers: name (unique), description, version, status (5 states), sla_thresholds JSONB, parent_set_id (self-referencing hierarchy), tags, metadata, created_by';

COMMENT ON TABLE validation_rule_service.validation_rule_set_members IS
    'Rule-to-set membership: set_id FK, rule_id FK, compound_rule_id FK (nullable), position ordering, override_severity (nullable), added_at, UNIQUE(set_id, rule_id)';

COMMENT ON TABLE validation_rule_service.validation_rule_versions IS
    'Rule version history: rule_id FK, version number, changes JSONB (diff), snapshot JSONB (full state), created_at';

COMMENT ON TABLE validation_rule_service.validation_compound_rules IS
    'Compound boolean rules: name, operator (AND/OR/NOT), child_rule_ids UUID[], child_compound_ids UUID[] (nested), description';

COMMENT ON TABLE validation_rule_service.validation_rule_packs IS
    'Packaged rule collections: pack_name (unique), pack_type (9 types), version, description, rule_definitions JSONB, metadata';

COMMENT ON TABLE validation_rule_service.validation_evaluations IS
    'Evaluation execution results: rule_set_id FK, dataset_name, total_rules, passed/failed/warned counts, pass_rate NUMERIC(5,4), sla_result (4 states), duration_ms, provenance_hash';

COMMENT ON TABLE validation_rule_service.validation_evaluation_details IS
    'Per-rule evaluation details: evaluation_id FK, rule_id FK, pass_count, fail_count, total, pass_rate NUMERIC(5,4), result (6 states), failures JSONB';

COMMENT ON TABLE validation_rule_service.validation_reports IS
    'Generated validation reports: report_type (6 types), format (6 formats), content TEXT, report_hash, parameters JSONB, provenance_hash';

COMMENT ON TABLE validation_rule_service.validation_audit_log IS
    'Full audit trail: entity_type (10 types), entity_id, action (29 actions), actor, changes JSONB, SHA-256 provenance and parent hashes';

COMMENT ON TABLE validation_rule_service.validation_evaluation_events IS
    'TimescaleDB hypertable: evaluation events with rule_set_id, dataset_name, rule_type, severity, result, pass_rate, duration_ms (7-day chunks, 90-day retention)';

COMMENT ON TABLE validation_rule_service.validation_rule_change_events IS
    'TimescaleDB hypertable: rule change events with rule_id, action, rule_type, severity, actor (7-day chunks, 90-day retention)';

COMMENT ON TABLE validation_rule_service.validation_conflict_events IS
    'TimescaleDB hypertable: rule conflict events with conflict_type, rule_a_id, rule_b_id, column_name, severity (7-day chunks, 90-day retention)';

COMMENT ON MATERIALIZED VIEW validation_rule_service.validation_evaluations_hourly_stats IS
    'Continuous aggregate: hourly evaluation stats by rule_type/result (total events, avg pass rate, avg/total duration per hour)';

COMMENT ON MATERIALIZED VIEW validation_rule_service.validation_rule_changes_hourly_stats IS
    'Continuous aggregate: hourly rule change stats by action/rule_type (total events per hour)';
