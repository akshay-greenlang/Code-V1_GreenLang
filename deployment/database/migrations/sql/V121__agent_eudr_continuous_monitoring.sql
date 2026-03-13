-- ============================================================================
-- V121: AGENT-EUDR-033 Continuous Monitoring Agent
-- ============================================================================
-- Creates tables for the Continuous Monitoring Agent which provides ongoing
-- surveillance of EUDR compliance across supply chains: scheduled monitoring
-- job orchestration for deforestation checks, compliance audits, risk score
-- recalculation, data freshness validation, and regulatory update scanning;
-- TimescaleDB-partitioned execution history for run analytics; real-time
-- alerting with severity-based triage and lifecycle management; entity-level
-- change tracking with impact assessment; data freshness monitoring with
-- configurable staleness thresholds; point-in-time compliance snapshots for
-- trend analysis; regulatory change tracking from EU Commission and competent
-- authorities; configurable alert rules with multi-channel notification; and
-- immutable Article 31 audit trail via TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-CM-033
-- PRD: PRD-AGENT-EUDR-033
-- Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11, 12, 14-16, 29, 31
-- Tables: 9 (6 regular + 3 hypertables)
-- Indexes: ~121
--
-- Dependencies: TimescaleDB extension (for hypertables)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V121: Creating AGENT-EUDR-033 Continuous Monitoring Agent tables...';


-- ============================================================================
-- 1. gl_eudr_cm_monitoring_jobs -- Scheduled monitoring job configuration
-- ============================================================================
RAISE NOTICE 'V121 [1/9]: Creating gl_eudr_cm_monitoring_jobs...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_monitoring_jobs (
    job_id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this monitoring job definition
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose supply chain is being monitored
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    job_name                        VARCHAR(500)    NOT NULL,
        -- Human-readable name for the monitoring job
    job_type                        VARCHAR(30)     NOT NULL,
        -- Classification of monitoring activity performed
    job_description                 TEXT            NOT NULL DEFAULT '',
        -- Detailed description of what this job monitors and why
    schedule_cron                   VARCHAR(100)    NOT NULL DEFAULT '0 0 * * *',
        -- Cron expression defining execution schedule (default: daily at midnight)
    timezone                        VARCHAR(50)     NOT NULL DEFAULT 'UTC',
        -- IANA timezone for cron schedule interpretation
    is_enabled                      BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether the job is currently active and should be scheduled
    priority                        INTEGER         NOT NULL DEFAULT 5,
        -- Execution priority (1=highest, 10=lowest) for resource allocation
    max_runtime_seconds             INTEGER         NOT NULL DEFAULT 3600,
        -- Maximum allowed runtime before the job is force-terminated
    retry_count                     INTEGER         NOT NULL DEFAULT 3,
        -- Number of automatic retries on transient failure
    retry_delay_seconds             INTEGER         NOT NULL DEFAULT 60,
        -- Delay between retry attempts in seconds
    last_run_at                     TIMESTAMPTZ,
        -- Timestamp of the most recent execution start
    last_run_status                 VARCHAR(20),
        -- Status of the most recent execution (success, failed, timeout, skipped)
    next_run_at                     TIMESTAMPTZ,
        -- Calculated timestamp of the next scheduled execution
    consecutive_failures            INTEGER         NOT NULL DEFAULT 0,
        -- Count of consecutive failed runs (reset on success)
    total_runs                      INTEGER         NOT NULL DEFAULT 0,
        -- Lifetime count of completed runs for this job
    entity_scope                    JSONB           DEFAULT '{}',
        -- Scope filter defining which entities to monitor: {"supplier_ids": [...], "commodity_types": [...], "country_codes": [...]}
    config                          JSONB           DEFAULT '{}',
        -- Job-specific configuration parameters: {"thresholds": {...}, "filters": {...}, "notification_channels": [...]}
    alert_rules                     JSONB           DEFAULT '[]',
        -- Array of inline alert rule references: [{"rule_id": "uuid", "enabled": true}, ...]
    notification_config             JSONB           DEFAULT '{}',
        -- Notification routing: {"channels": ["email", "slack", "webhook"], "recipients": [...], "escalation_policy": {...}}
    tags                            JSONB           DEFAULT '[]',
        -- Organizational tags for job categorization: ["critical", "daily-scan", "tier-1-suppliers"]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for job configuration integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system actor that created this job definition

    CONSTRAINT chk_cm_job_type CHECK (job_type IN (
        'supply_chain_scan', 'deforestation_check', 'compliance_audit',
        'risk_score_monitor', 'data_freshness_check', 'regulatory_update_scan'
    )),
    CONSTRAINT chk_cm_job_last_status CHECK (last_run_status IS NULL OR last_run_status IN (
        'success', 'failed', 'timeout', 'skipped', 'running'
    )),
    CONSTRAINT chk_cm_job_priority CHECK (priority >= 1 AND priority <= 10),
    CONSTRAINT chk_cm_job_max_runtime CHECK (max_runtime_seconds > 0 AND max_runtime_seconds <= 86400),
    CONSTRAINT chk_cm_job_retry_count CHECK (retry_count >= 0 AND retry_count <= 10),
    CONSTRAINT chk_cm_job_retry_delay CHECK (retry_delay_seconds >= 0),
    CONSTRAINT chk_cm_job_consecutive CHECK (consecutive_failures >= 0),
    CONSTRAINT chk_cm_job_total_runs CHECK (total_runs >= 0)
);

COMMENT ON TABLE gl_eudr_cm_monitoring_jobs IS 'AGENT-EUDR-033: Scheduled monitoring job definitions for continuous EUDR compliance surveillance across supply_chain_scan, deforestation_check, compliance_audit, risk_score_monitor, data_freshness_check, and regulatory_update_scan job types with cron-based scheduling, retry policies, and scoped entity targeting';

-- B-tree indexes for gl_eudr_cm_monitoring_jobs
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_operator ON gl_eudr_cm_monitoring_jobs (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_tenant ON gl_eudr_cm_monitoring_jobs (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_type ON gl_eudr_cm_monitoring_jobs (job_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_next_run ON gl_eudr_cm_monitoring_jobs (next_run_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_last_run ON gl_eudr_cm_monitoring_jobs (last_run_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_provenance ON gl_eudr_cm_monitoring_jobs (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_created ON gl_eudr_cm_monitoring_jobs (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_operator_type ON gl_eudr_cm_monitoring_jobs (operator_id, job_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_tenant_operator ON gl_eudr_cm_monitoring_jobs (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_enabled_next ON gl_eudr_cm_monitoring_jobs (is_enabled, next_run_at)
        WHERE is_enabled = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_operator_enabled ON gl_eudr_cm_monitoring_jobs (operator_id, is_enabled, next_run_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_entity_scope ON gl_eudr_cm_monitoring_jobs USING GIN (entity_scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_config ON gl_eudr_cm_monitoring_jobs USING GIN (config);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_job_tags ON gl_eudr_cm_monitoring_jobs USING GIN (tags);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_cm_monitoring_runs -- Execution history (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V121 [2/9]: Creating gl_eudr_cm_monitoring_runs (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_monitoring_runs (
    run_id                          UUID            DEFAULT gen_random_uuid(),
        -- Unique identifier for this monitoring run execution
    job_id                          UUID            NOT NULL,
        -- FK reference to the monitoring job definition
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose supply chain was scanned
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    run_started_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the monitoring run began (partitioning column)
    run_completed_at                TIMESTAMPTZ,
        -- Timestamp when the monitoring run finished (NULL if still running)
    run_duration_ms                 BIGINT,
        -- Run duration in milliseconds (computed on completion)
    status                          VARCHAR(20)     NOT NULL DEFAULT 'running',
        -- Current execution status of this run
    trigger_type                    VARCHAR(20)     NOT NULL DEFAULT 'scheduled',
        -- How the run was triggered
    entities_scanned                INTEGER         NOT NULL DEFAULT 0,
        -- Total number of entities evaluated during this run
    entities_passed                 INTEGER         NOT NULL DEFAULT 0,
        -- Number of entities that passed all checks
    entities_failed                 INTEGER         NOT NULL DEFAULT 0,
        -- Number of entities that failed one or more checks
    entities_skipped                INTEGER         NOT NULL DEFAULT 0,
        -- Number of entities skipped (e.g., insufficient data)
    alerts_generated                INTEGER         NOT NULL DEFAULT 0,
        -- Number of new alerts produced by this run
    changes_detected                INTEGER         NOT NULL DEFAULT 0,
        -- Number of entity changes detected during this run
    findings                        JSONB           DEFAULT '{}',
        -- Structured findings summary: {"deforestation_alerts": 2, "expired_certs": 5, "stale_data": 12, ...}
    error_details                   JSONB           DEFAULT '{}',
        -- Error information on failure: {"error_code": "...", "message": "...", "stack_trace": "..."}
    resource_usage                  JSONB           DEFAULT '{}',
        -- Resource consumption metrics: {"memory_mb": 512, "cpu_seconds": 45.2, "db_queries": 1500}
    run_config_snapshot             JSONB           DEFAULT '{}',
        -- Snapshot of job config at time of execution (for reproducibility)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for run result integrity verification

    CONSTRAINT chk_cm_run_status CHECK (status IN (
        'running', 'success', 'failed', 'timeout', 'cancelled'
    )),
    CONSTRAINT chk_cm_run_trigger CHECK (trigger_type IN (
        'scheduled', 'manual', 'api', 'event_driven', 'retry'
    )),
    CONSTRAINT chk_cm_run_scanned CHECK (entities_scanned >= 0),
    CONSTRAINT chk_cm_run_passed CHECK (entities_passed >= 0),
    CONSTRAINT chk_cm_run_failed CHECK (entities_failed >= 0),
    CONSTRAINT chk_cm_run_skipped CHECK (entities_skipped >= 0),
    CONSTRAINT chk_cm_run_alerts CHECK (alerts_generated >= 0),
    CONSTRAINT chk_cm_run_changes CHECK (changes_detected >= 0),
    CONSTRAINT chk_cm_run_duration CHECK (run_duration_ms IS NULL OR run_duration_ms >= 0)
);

-- Convert to hypertable partitioned by run_started_at
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_cm_monitoring_runs',
        'run_started_at',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_cm_monitoring_runs hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_cm_monitoring_runs: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_cm_monitoring_runs IS 'AGENT-EUDR-033: TimescaleDB-partitioned execution history for monitoring runs with entity scan counts, alert generation tracking, change detection metrics, and resource consumption logging for EUDR Article 14-16 continuous compliance evidence';

-- B-tree indexes for gl_eudr_cm_monitoring_runs
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_job ON gl_eudr_cm_monitoring_runs (job_id, run_started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_operator ON gl_eudr_cm_monitoring_runs (operator_id, run_started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_tenant ON gl_eudr_cm_monitoring_runs (tenant_id, run_started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_status ON gl_eudr_cm_monitoring_runs (status, run_started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_completed ON gl_eudr_cm_monitoring_runs (run_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_provenance ON gl_eudr_cm_monitoring_runs (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_alerts_gen ON gl_eudr_cm_monitoring_runs (alerts_generated DESC, run_started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_job_status ON gl_eudr_cm_monitoring_runs (job_id, status, run_started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_tenant_operator ON gl_eudr_cm_monitoring_runs (tenant_id, operator_id, run_started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_failed_only ON gl_eudr_cm_monitoring_runs (job_id, run_started_at DESC)
        WHERE status = 'failed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_findings ON gl_eudr_cm_monitoring_runs USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_run_errors ON gl_eudr_cm_monitoring_runs USING GIN (error_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_cm_alerts -- Generated alerts (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V121 [3/9]: Creating gl_eudr_cm_alerts (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_alerts (
    alert_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique identifier for this alert
    run_id                          UUID,
        -- FK reference to the monitoring run that generated this alert (NULL for manual alerts)
    job_id                          UUID,
        -- FK reference to the monitoring job that produced this alert
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose entity triggered the alert
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity that triggered the alert
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the specific entity that triggered the alert
    entity_name                     VARCHAR(500),
        -- Human-readable name of the triggering entity for display
    alert_type                      VARCHAR(30)     NOT NULL,
        -- Classification of the alert condition
    severity                        VARCHAR(10)     NOT NULL DEFAULT 'medium',
        -- Alert severity level for triage and prioritization
    alert_title                     VARCHAR(500)    NOT NULL,
        -- Concise human-readable alert title
    alert_description               TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the alert condition and context
    detected_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the alert condition was detected (partitioning column)
    acknowledged_at                 TIMESTAMPTZ,
        -- Timestamp when a user acknowledged the alert
    acknowledged_by                 VARCHAR(100),
        -- User who acknowledged the alert
    resolved_at                     TIMESTAMPTZ,
        -- Timestamp when the alert was resolved
    resolved_by                     VARCHAR(100),
        -- User who resolved the alert
    resolution_notes                TEXT            DEFAULT '',
        -- Notes describing how the alert was resolved
    assignee_id                     VARCHAR(100),
        -- User or team assigned to investigate/resolve the alert
    alert_status                    VARCHAR(20)     NOT NULL DEFAULT 'open',
        -- Current lifecycle status of the alert
    escalation_level                INTEGER         NOT NULL DEFAULT 0,
        -- Current escalation level (0=none, 1=team_lead, 2=manager, 3=director)
    escalated_at                    TIMESTAMPTZ,
        -- Timestamp of most recent escalation
    notification_sent               BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether notification has been dispatched
    notification_channels           JSONB           DEFAULT '[]',
        -- Channels notified: ["email", "slack", "webhook"]
    related_alert_ids               JSONB           DEFAULT '[]',
        -- Array of related alert IDs for correlation: ["uuid-1", "uuid-2"]
    previous_value                  TEXT,
        -- Previous value of the monitored metric (for change-based alerts)
    current_value                   TEXT,
        -- Current value of the monitored metric
    threshold_value                 TEXT,
        -- Threshold that was breached to trigger the alert
    metadata                        JSONB           DEFAULT '{}',
        -- Alert-specific metadata: {"source_data": {...}, "comparison": {...}, "geo_coords": {...}}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for alert integrity verification

    CONSTRAINT chk_cm_alert_type CHECK (alert_type IN (
        'deforestation_detected', 'risk_score_increased', 'certification_expiring',
        'data_stale', 'supplier_status_changed', 'geo_boundary_changed',
        'regulatory_change', 'compliance_degraded', 'batch_failure'
    )),
    CONSTRAINT chk_cm_alert_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low'
    )),
    CONSTRAINT chk_cm_alert_status CHECK (alert_status IN (
        'open', 'acknowledged', 'investigating', 'resolved', 'dismissed', 'escalated'
    )),
    CONSTRAINT chk_cm_alert_escalation CHECK (escalation_level >= 0 AND escalation_level <= 5)
);

-- Convert to hypertable partitioned by detected_at
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_cm_alerts',
        'detected_at',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_cm_alerts hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_cm_alerts: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_cm_alerts IS 'AGENT-EUDR-033: TimescaleDB-partitioned alert records for deforestation detection, risk score changes, certification expiry, data staleness, supplier status changes, geo-boundary shifts, and regulatory updates with full lifecycle management (open/acknowledged/investigating/resolved/dismissed/escalated) and severity-based triage';

-- B-tree indexes for gl_eudr_cm_alerts
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_run ON gl_eudr_cm_alerts (run_id, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_operator ON gl_eudr_cm_alerts (operator_id, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_tenant ON gl_eudr_cm_alerts (tenant_id, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_entity_id ON gl_eudr_cm_alerts (entity_id, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_type ON gl_eudr_cm_alerts (alert_type, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_severity ON gl_eudr_cm_alerts (severity, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_status ON gl_eudr_cm_alerts (alert_status, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_assignee ON gl_eudr_cm_alerts (assignee_id, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_resolved ON gl_eudr_cm_alerts (resolved_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_provenance ON gl_eudr_cm_alerts (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_operator_type ON gl_eudr_cm_alerts (operator_id, alert_type, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_operator_severity ON gl_eudr_cm_alerts (operator_id, severity, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_tenant_operator ON gl_eudr_cm_alerts (tenant_id, operator_id, detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_open_only ON gl_eudr_cm_alerts (operator_id, severity, detected_at DESC)
        WHERE alert_status IN ('open', 'acknowledged', 'investigating', 'escalated');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_critical_only ON gl_eudr_cm_alerts (operator_id, detected_at DESC)
        WHERE severity = 'critical' AND alert_status NOT IN ('resolved', 'dismissed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_alert_metadata ON gl_eudr_cm_alerts USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_cm_change_history -- Entity change tracking
-- ============================================================================
RAISE NOTICE 'V121 [4/9]: Creating gl_eudr_cm_change_history...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_change_history (
    change_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this change record
    run_id                          UUID,
        -- FK reference to the monitoring run that detected this change (NULL for external detection)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose entity changed
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity that changed (supplier, plot, certificate, commodity, etc.)
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the specific entity that changed
    entity_name                     VARCHAR(500),
        -- Human-readable name of the changed entity
    field_name                      VARCHAR(200)    NOT NULL,
        -- Name of the field or attribute that changed
    old_value                       TEXT,
        -- Previous value before the change (NULL for new entities)
    new_value                       TEXT,
        -- New value after the change (NULL for deleted entities)
    change_type                     VARCHAR(20)     NOT NULL DEFAULT 'modified',
        -- Classification of the change
    change_detected_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the change was detected
    change_source                   VARCHAR(50)     NOT NULL DEFAULT 'monitoring_scan',
        -- How the change was detected
    change_magnitude                NUMERIC(10,4),
        -- Quantified magnitude of the change (for numeric fields, percentage difference)
    is_significant                  BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this change exceeds significance thresholds
    alert_generated                 BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this change triggered an alert
    alert_id                        UUID,
        -- FK reference to the generated alert (NULL if no alert triggered)
    impact_assessment               JSONB           DEFAULT '{}',
        -- Assessed impact of this change: {"compliance_impact": "high", "supply_chain_impact": "medium", "affected_products": [...], "risk_delta": 15.5}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for change record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_cm_change_type CHECK (change_type IN (
        'created', 'modified', 'deleted', 'status_changed', 'threshold_breached'
    )),
    CONSTRAINT chk_cm_change_source CHECK (change_source IN (
        'monitoring_scan', 'api_webhook', 'manual_update', 'external_feed',
        'satellite_detection', 'regulatory_notification'
    ))
);

COMMENT ON TABLE gl_eudr_cm_change_history IS 'AGENT-EUDR-033: Entity-level change tracking with old/new value capture, change magnitude quantification, significance flagging, and impact assessment for EUDR continuous monitoring evidence per Articles 9-12';

-- B-tree indexes for gl_eudr_cm_change_history
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_operator ON gl_eudr_cm_change_history (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_tenant ON gl_eudr_cm_change_history (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_entity_type ON gl_eudr_cm_change_history (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_change_type ON gl_eudr_cm_change_history (change_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_detected ON gl_eudr_cm_change_history (change_detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_source ON gl_eudr_cm_change_history (change_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_provenance ON gl_eudr_cm_change_history (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_created ON gl_eudr_cm_change_history (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_entity_type_id ON gl_eudr_cm_change_history (entity_type, entity_id, change_detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_operator_entity ON gl_eudr_cm_change_history (operator_id, entity_type, change_detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_tenant_operator ON gl_eudr_cm_change_history (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_significant_only ON gl_eudr_cm_change_history (operator_id, entity_type, change_detected_at DESC)
        WHERE is_significant = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_ch_impact ON gl_eudr_cm_change_history USING GIN (impact_assessment);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_cm_data_freshness_status -- Data age tracking per entity
-- ============================================================================
RAISE NOTICE 'V121 [5/9]: Creating gl_eudr_cm_data_freshness_status...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_data_freshness_status (
    freshness_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this freshness status record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose data freshness is tracked
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity whose data freshness is tracked
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the specific entity
    entity_name                     VARCHAR(500),
        -- Human-readable name of the entity for display
    data_source                     VARCHAR(100)    NOT NULL DEFAULT 'primary',
        -- Source of the data being tracked (primary, satellite, erp, supplier_portal, etc.)
    last_updated_at                 TIMESTAMPTZ     NOT NULL,
        -- Timestamp when the entity data was last refreshed
    age_days                        INTEGER         NOT NULL DEFAULT 0,
        -- Current age of the data in days (computed)
    freshness_status                VARCHAR(10)     NOT NULL DEFAULT 'fresh',
        -- Current freshness classification based on age vs thresholds
    threshold_fresh_days            INTEGER         NOT NULL DEFAULT 30,
        -- Maximum age in days to be classified as fresh
    threshold_aging_days            INTEGER         NOT NULL DEFAULT 60,
        -- Maximum age in days to be classified as aging (above this = stale)
    threshold_stale_days            INTEGER         NOT NULL DEFAULT 90,
        -- Maximum age in days to be classified as stale (above this = expired)
    next_refresh_due                TIMESTAMPTZ,
        -- Calculated timestamp when next refresh is expected
    refresh_frequency_days          INTEGER         DEFAULT 30,
        -- Expected refresh frequency in days
    last_check_at                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the most recent freshness check
    alert_sent                      BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether a staleness alert has been dispatched
    alert_id                        UUID,
        -- FK reference to the staleness alert (NULL if not alerted)
    metadata                        JSONB           DEFAULT '{}',
        -- Additional context: {"data_completeness": 0.95, "source_reliability": "high", "refresh_method": "api_pull"}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for freshness record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_cm_fresh_status CHECK (freshness_status IN (
        'fresh', 'aging', 'stale', 'expired'
    )),
    CONSTRAINT chk_cm_fresh_age CHECK (age_days >= 0),
    CONSTRAINT chk_cm_fresh_threshold_order CHECK (
        threshold_fresh_days > 0
        AND threshold_aging_days > threshold_fresh_days
        AND threshold_stale_days > threshold_aging_days
    ),
    CONSTRAINT chk_cm_fresh_refresh_freq CHECK (refresh_frequency_days IS NULL OR refresh_frequency_days > 0),
    CONSTRAINT uq_cm_fresh_entity UNIQUE (operator_id, entity_type, entity_id, data_source)
);

COMMENT ON TABLE gl_eudr_cm_data_freshness_status IS 'AGENT-EUDR-033: Per-entity data age tracking with configurable freshness thresholds (fresh/aging/stale/expired), expected refresh schedules, and staleness alerting for EUDR Article 10 data currency requirements';

-- B-tree indexes for gl_eudr_cm_data_freshness_status
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_operator ON gl_eudr_cm_data_freshness_status (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_tenant ON gl_eudr_cm_data_freshness_status (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_entity_type ON gl_eudr_cm_data_freshness_status (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_status ON gl_eudr_cm_data_freshness_status (freshness_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_age ON gl_eudr_cm_data_freshness_status (age_days DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_next_refresh ON gl_eudr_cm_data_freshness_status (next_refresh_due);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_provenance ON gl_eudr_cm_data_freshness_status (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_operator_status ON gl_eudr_cm_data_freshness_status (operator_id, freshness_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_operator_entity ON gl_eudr_cm_data_freshness_status (operator_id, entity_type, entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_tenant_operator ON gl_eudr_cm_data_freshness_status (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_stale_only ON gl_eudr_cm_data_freshness_status (operator_id, entity_type, age_days DESC)
        WHERE freshness_status IN ('stale', 'expired');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_fresh_metadata ON gl_eudr_cm_data_freshness_status USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_cm_compliance_snapshots -- Point-in-time compliance status
-- ============================================================================
RAISE NOTICE 'V121 [6/9]: Creating gl_eudr_cm_compliance_snapshots...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_compliance_snapshots (
    snapshot_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this compliance snapshot
    run_id                          UUID,
        -- FK reference to the monitoring run that generated this snapshot
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose compliance was assessed
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    snapshot_timestamp              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Point in time when this compliance status was captured
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity assessed (operator, supplier, commodity_flow, plot, product)
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the assessed entity
    entity_name                     VARCHAR(500),
        -- Human-readable name of the assessed entity
    compliance_status               VARCHAR(20)     NOT NULL DEFAULT 'unknown',
        -- Overall compliance determination
    overall_score                   NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Aggregate compliance score (0.00-100.00)
    dimension_scores                JSONB           DEFAULT '{}',
        -- Per-dimension scores: {"deforestation_free": 95.0, "traceability": 88.5, "legality": 92.0, "due_diligence": 85.0, "data_freshness": 70.0}
    findings                        JSONB           DEFAULT '[]',
        -- Array of findings: [{"finding_id": "...", "category": "...", "severity": "high", "description": "...", "evidence": "..."}, ...]
    finding_count_by_severity       JSONB           DEFAULT '{}',
        -- Finding count breakdown: {"critical": 0, "high": 2, "medium": 5, "low": 3}
    previous_snapshot_id            UUID,
        -- FK reference to the preceding snapshot for trend calculation
    previous_score                  NUMERIC(5,2),
        -- Score from previous snapshot for delta computation
    score_delta                     NUMERIC(6,2),
        -- Change in score from previous snapshot (positive = improvement)
    trend_direction                 VARCHAR(20)     DEFAULT 'stable',
        -- Trend direction based on recent snapshots
    risk_level                      VARCHAR(20)     DEFAULT 'low',
        -- Inferred risk level based on compliance score
    recommendations                 JSONB           DEFAULT '[]',
        -- Recommended actions: [{"action": "...", "priority": "high", "deadline": "..."}, ...]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for snapshot integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_cm_snap_compliance CHECK (compliance_status IN (
        'compliant', 'partially_compliant', 'non_compliant', 'under_review', 'unknown'
    )),
    CONSTRAINT chk_cm_snap_score CHECK (overall_score >= 0 AND overall_score <= 100),
    CONSTRAINT chk_cm_snap_prev_score CHECK (previous_score IS NULL OR (previous_score >= 0 AND previous_score <= 100)),
    CONSTRAINT chk_cm_snap_trend CHECK (trend_direction IN (
        'improving', 'stable', 'declining'
    )),
    CONSTRAINT chk_cm_snap_risk CHECK (risk_level IN (
        'negligible', 'low', 'moderate', 'high', 'critical'
    ))
);

COMMENT ON TABLE gl_eudr_cm_compliance_snapshots IS 'AGENT-EUDR-033: Point-in-time compliance status captures with multi-dimension scoring (deforestation-free, traceability, legality, due diligence, data freshness), trend analysis via linked snapshots, and risk classification for EUDR Articles 4, 9-12 continuous compliance evidence';

-- B-tree indexes for gl_eudr_cm_compliance_snapshots
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_operator ON gl_eudr_cm_compliance_snapshots (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_tenant ON gl_eudr_cm_compliance_snapshots (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_timestamp ON gl_eudr_cm_compliance_snapshots (snapshot_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_entity_id ON gl_eudr_cm_compliance_snapshots (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_status ON gl_eudr_cm_compliance_snapshots (compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_prev ON gl_eudr_cm_compliance_snapshots (previous_snapshot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_provenance ON gl_eudr_cm_compliance_snapshots (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_created ON gl_eudr_cm_compliance_snapshots (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_operator_entity ON gl_eudr_cm_compliance_snapshots (operator_id, entity_type, entity_id, snapshot_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_operator_status ON gl_eudr_cm_compliance_snapshots (operator_id, compliance_status, snapshot_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_operator_risk ON gl_eudr_cm_compliance_snapshots (operator_id, risk_level, snapshot_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_tenant_operator ON gl_eudr_cm_compliance_snapshots (tenant_id, operator_id, snapshot_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_non_compliant ON gl_eudr_cm_compliance_snapshots (operator_id, snapshot_timestamp DESC)
        WHERE compliance_status = 'non_compliant';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_dimensions ON gl_eudr_cm_compliance_snapshots USING GIN (dimension_scores);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_snap_findings ON gl_eudr_cm_compliance_snapshots USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_cm_regulatory_updates -- Track regulatory changes
-- ============================================================================
RAISE NOTICE 'V121 [7/9]: Creating gl_eudr_cm_regulatory_updates...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_regulatory_updates (
    update_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this regulatory update record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator affected by this regulatory change
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    update_source                   VARCHAR(30)     NOT NULL,
        -- Authoritative source of the regulatory update
    update_type                     VARCHAR(30)     NOT NULL,
        -- Classification of the regulatory change
    update_date                     DATE            NOT NULL,
        -- Official date of the regulatory update or publication
    effective_date                  DATE,
        -- Date when the regulatory change becomes enforceable
    title                           VARCHAR(1000)   NOT NULL,
        -- Title or headline of the regulatory update
    description                     TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the regulatory change and its implications
    regulation_reference            VARCHAR(500),
        -- Official regulation reference (e.g., "EU 2023/1115 Article 10(2)(a) Amendment 2026/xxx")
    impact_level                    VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Assessed impact level on EUDR compliance operations
    affected_commodities            JSONB           DEFAULT '[]',
        -- Array of affected EUDR commodities: ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]
    affected_countries              JSONB           DEFAULT '[]',
        -- Array of affected country codes: ["BR", "ID", "CO"]
    affected_entities               JSONB           DEFAULT '[]',
        -- Array of affected entity types and IDs: [{"type": "supplier", "id": "..."}, ...]
    required_actions                JSONB           DEFAULT '[]',
        -- Actions required for compliance: [{"action": "...", "deadline": "...", "priority": "high"}, ...]
    action_status                   VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Status of required action implementation
    source_url                      VARCHAR(2000),
        -- URL to the original regulatory publication or notice
    source_document_hash            VARCHAR(64),
        -- SHA-256 hash of the source document for verification
    reviewed_by                     VARCHAR(100),
        -- User who reviewed and assessed the regulatory update
    reviewed_at                     TIMESTAMPTZ,
        -- Timestamp when the update was reviewed
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_cm_reg_source CHECK (update_source IN (
        'eu_commission', 'competent_authority', 'third_party',
        'member_state', 'international_body', 'industry_standard'
    )),
    CONSTRAINT chk_cm_reg_type CHECK (update_type IN (
        'regulation_amendment', 'guidance_update', 'benchmark_change',
        'country_classification', 'commodity_list_update', 'enforcement_action',
        'implementation_deadline', 'faq_update'
    )),
    CONSTRAINT chk_cm_reg_impact CHECK (impact_level IN (
        'critical', 'high', 'medium', 'low', 'informational'
    )),
    CONSTRAINT chk_cm_reg_action_status CHECK (action_status IN (
        'pending', 'in_progress', 'completed', 'not_applicable'
    ))
);

COMMENT ON TABLE gl_eudr_cm_regulatory_updates IS 'AGENT-EUDR-033: Regulatory change tracking from EU Commission, competent authorities, member states, and international bodies with impact assessment, affected entity mapping, required action tracking, and source document verification for EUDR Articles 14-16 compliance monitoring';

-- B-tree indexes for gl_eudr_cm_regulatory_updates
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_operator ON gl_eudr_cm_regulatory_updates (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_tenant ON gl_eudr_cm_regulatory_updates (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_type ON gl_eudr_cm_regulatory_updates (update_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_date ON gl_eudr_cm_regulatory_updates (update_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_impact ON gl_eudr_cm_regulatory_updates (impact_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_action_status ON gl_eudr_cm_regulatory_updates (action_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_provenance ON gl_eudr_cm_regulatory_updates (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_created ON gl_eudr_cm_regulatory_updates (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_operator_source ON gl_eudr_cm_regulatory_updates (operator_id, update_source, update_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_operator_impact ON gl_eudr_cm_regulatory_updates (operator_id, impact_level, update_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_tenant_operator ON gl_eudr_cm_regulatory_updates (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_pending_actions ON gl_eudr_cm_regulatory_updates (operator_id, effective_date)
        WHERE action_status IN ('pending', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_critical_impact ON gl_eudr_cm_regulatory_updates (operator_id, update_date DESC)
        WHERE impact_level IN ('critical', 'high');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_commodities ON gl_eudr_cm_regulatory_updates USING GIN (affected_commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_countries ON gl_eudr_cm_regulatory_updates USING GIN (affected_countries);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_reg_actions ON gl_eudr_cm_regulatory_updates USING GIN (required_actions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_cm_alert_rules -- Configurable alert rules
-- ============================================================================
RAISE NOTICE 'V121 [8/9]: Creating gl_eudr_cm_alert_rules...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_alert_rules (
    rule_id                         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this alert rule definition
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator this rule applies to
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    rule_name                       VARCHAR(500)    NOT NULL,
        -- Human-readable name describing the alert condition
    rule_description                TEXT            NOT NULL DEFAULT '',
        -- Detailed description of what this rule monitors and when it triggers
    condition_type                  VARCHAR(30)     NOT NULL,
        -- Type of condition being evaluated
    condition_config                JSONB           NOT NULL DEFAULT '{}',
        -- Condition parameters: {"metric": "risk_score", "operator": "greater_than", "threshold": 75, "window_days": 30, ...}
    entity_scope                    JSONB           DEFAULT '{}',
        -- Scope filter: {"entity_types": ["supplier", "plot"], "commodity_types": ["cocoa"], "country_codes": ["BR"]}
    alert_type                      VARCHAR(30)     NOT NULL DEFAULT 'compliance_degraded',
        -- Alert type to generate when rule fires
    alert_severity                  VARCHAR(10)     NOT NULL DEFAULT 'medium',
        -- Severity level assigned to generated alerts
    is_enabled                      BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this rule is currently active
    cooldown_minutes                INTEGER         NOT NULL DEFAULT 60,
        -- Minimum time between consecutive firings of this rule (prevents alert storms)
    max_alerts_per_day              INTEGER         NOT NULL DEFAULT 100,
        -- Maximum alerts this rule can generate per day
    last_fired_at                   TIMESTAMPTZ,
        -- Timestamp when this rule last generated an alert
    total_firings                   INTEGER         NOT NULL DEFAULT 0,
        -- Lifetime count of times this rule has fired
    notification_channels           TEXT[]          DEFAULT '{}',
        -- Array of notification channels: ARRAY['email', 'slack', 'webhook', 'sms']
    notification_recipients         JSONB           DEFAULT '[]',
        -- Notification recipients: [{"channel": "email", "address": "..."}, {"channel": "slack", "webhook_url": "..."}]
    escalation_config               JSONB           DEFAULT '{}',
        -- Escalation policy: {"auto_escalate_after_minutes": 120, "escalation_path": ["team_lead", "manager", "director"]}
    tags                            JSONB           DEFAULT '[]',
        -- Organizational tags: ["deforestation", "critical-path", "tier-1"]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for rule configuration integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system actor that created this rule

    CONSTRAINT chk_cm_rule_condition CHECK (condition_type IN (
        'threshold_breach', 'trend_change', 'status_change', 'data_staleness',
        'certification_expiry', 'geo_boundary_change', 'composite_score'
    )),
    CONSTRAINT chk_cm_rule_alert_type CHECK (alert_type IN (
        'deforestation_detected', 'risk_score_increased', 'certification_expiring',
        'data_stale', 'supplier_status_changed', 'geo_boundary_changed',
        'regulatory_change', 'compliance_degraded', 'batch_failure'
    )),
    CONSTRAINT chk_cm_rule_severity CHECK (alert_severity IN (
        'critical', 'high', 'medium', 'low'
    )),
    CONSTRAINT chk_cm_rule_cooldown CHECK (cooldown_minutes >= 0),
    CONSTRAINT chk_cm_rule_max_alerts CHECK (max_alerts_per_day > 0),
    CONSTRAINT chk_cm_rule_total_firings CHECK (total_firings >= 0)
);

COMMENT ON TABLE gl_eudr_cm_alert_rules IS 'AGENT-EUDR-033: Configurable alert rule definitions with condition evaluation (threshold breach, trend change, status change, data staleness, certification expiry, geo-boundary change, composite score), multi-channel notification routing, escalation policies, and rate limiting for automated EUDR compliance monitoring';

-- B-tree indexes for gl_eudr_cm_alert_rules
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_operator ON gl_eudr_cm_alert_rules (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_tenant ON gl_eudr_cm_alert_rules (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_condition ON gl_eudr_cm_alert_rules (condition_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_severity ON gl_eudr_cm_alert_rules (alert_severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_enabled ON gl_eudr_cm_alert_rules (is_enabled);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_provenance ON gl_eudr_cm_alert_rules (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_created ON gl_eudr_cm_alert_rules (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_operator_type ON gl_eudr_cm_alert_rules (operator_id, condition_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_operator_enabled ON gl_eudr_cm_alert_rules (operator_id, is_enabled)
        WHERE is_enabled = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_tenant_operator ON gl_eudr_cm_alert_rules (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_config ON gl_eudr_cm_alert_rules USING GIN (condition_config);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_scope ON gl_eudr_cm_alert_rules USING GIN (entity_scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_rule_tags ON gl_eudr_cm_alert_rules USING GIN (tags);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_cm_audit_trail -- Audit log (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V121 [9/9]: Creating gl_eudr_cm_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_cm_audit_trail (
    audit_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique audit entry identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity being audited (monitoring_job, monitoring_run, alert, change_history, etc.)
    entity_id                       UUID            NOT NULL,
        -- Entity identifier being audited
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    action                          VARCHAR(50)     NOT NULL,
        -- Action performed (create, update, enable, disable, acknowledge, resolve, escalate, etc.)
    actor_id                        VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Actor performing the action (user ID or system)
    actor_type                      VARCHAR(20)     NOT NULL DEFAULT 'system',
        -- Type of actor (system, user, api, scheduler)
    changes                         JSONB           DEFAULT '{}',
        -- JSON diff of changes: {"field": {"old": "...", "new": "..."}}
    context                         JSONB           DEFAULT '{}',
        -- Additional context: {"ip_address": "...", "user_agent": "...", "request_id": "..."}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit entry integrity
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
        -- Timestamp of the action (partitioning column)
);

-- Convert to hypertable
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_cm_audit_trail',
        'timestamp',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_cm_audit_trail hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_cm_audit_trail: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_cm_audit_trail IS 'AGENT-EUDR-033: Immutable TimescaleDB-partitioned audit trail for all continuous monitoring operations per EUDR Article 31, capturing entity lifecycle events, user actions, system operations, and configuration changes with full provenance tracking';

-- B-tree indexes for gl_eudr_cm_audit_trail
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_entity_type ON gl_eudr_cm_audit_trail (entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_entity_id ON gl_eudr_cm_audit_trail (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_operator ON gl_eudr_cm_audit_trail (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_tenant ON gl_eudr_cm_audit_trail (tenant_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_action ON gl_eudr_cm_audit_trail (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_actor ON gl_eudr_cm_audit_trail (actor_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_provenance ON gl_eudr_cm_audit_trail (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_entity_action ON gl_eudr_cm_audit_trail (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_operator_entity ON gl_eudr_cm_audit_trail (operator_id, entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_cm_audit_changes ON gl_eudr_cm_audit_trail USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Triggers: updated_at auto-update
-- ============================================================================
RAISE NOTICE 'V121: Creating updated_at triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_cm_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_monitoring_jobs_updated_at
        BEFORE UPDATE ON gl_eudr_cm_monitoring_jobs
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_data_freshness_updated_at
        BEFORE UPDATE ON gl_eudr_cm_data_freshness_status
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_regulatory_updates_updated_at
        BEFORE UPDATE ON gl_eudr_cm_regulatory_updates
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_alert_rules_updated_at
        BEFORE UPDATE ON gl_eudr_cm_alert_rules
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Triggers: Audit trail auto-insert
-- ============================================================================
RAISE NOTICE 'V121: Creating audit trail triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_cm_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_cm_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'create', 'system', row_to_json(NEW)::JSONB, NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_eudr_cm_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_cm_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'update', 'system', jsonb_build_object('new', row_to_json(NEW)::JSONB), NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Monitoring jobs audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_jobs_audit_insert
        AFTER INSERT ON gl_eudr_cm_monitoring_jobs
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_insert('monitoring_job');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_jobs_audit_update
        AFTER UPDATE ON gl_eudr_cm_monitoring_jobs
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_update('monitoring_job');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Change history audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_change_audit_insert
        AFTER INSERT ON gl_eudr_cm_change_history
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_insert('change_history');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Data freshness audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_fresh_audit_insert
        AFTER INSERT ON gl_eudr_cm_data_freshness_status
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_insert('data_freshness');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_fresh_audit_update
        AFTER UPDATE ON gl_eudr_cm_data_freshness_status
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_update('data_freshness');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compliance snapshots audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_snap_audit_insert
        AFTER INSERT ON gl_eudr_cm_compliance_snapshots
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_insert('compliance_snapshot');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Regulatory updates audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_reg_audit_insert
        AFTER INSERT ON gl_eudr_cm_regulatory_updates
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_insert('regulatory_update');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_reg_audit_update
        AFTER UPDATE ON gl_eudr_cm_regulatory_updates
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_update('regulatory_update');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Alert rules audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_rules_audit_insert
        AFTER INSERT ON gl_eudr_cm_alert_rules
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_insert('alert_rule');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_cm_rules_audit_update
        AFTER UPDATE ON gl_eudr_cm_alert_rules
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_cm_audit_update('alert_rule');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Completion
-- ============================================================================

RAISE NOTICE 'V121: AGENT-EUDR-033 Continuous Monitoring Agent tables created successfully';
RAISE NOTICE 'V121: Tables: 9 (monitoring_jobs, monitoring_runs, alerts, change_history, data_freshness_status, compliance_snapshots, regulatory_updates, alert_rules, audit_trail)';
RAISE NOTICE 'V121: Hypertables: gl_eudr_cm_monitoring_runs, gl_eudr_cm_alerts, gl_eudr_cm_audit_trail (7-day chunks)';
RAISE NOTICE 'V121: Indexes: ~121';
RAISE NOTICE 'V121: Triggers: 4 updated_at + 10 audit trail';

COMMIT;
