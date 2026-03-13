-- ============================================================================
-- V099: AGENT-EUDR-011 Mass Balance Calculator Agent
-- ============================================================================
-- Creates tables for mass balance ledger management, double-entry ledger
-- entries, credit period tracking, conversion factor validation, overdraft
-- monitoring, loss/waste recording, carry-forward management, reconciliation,
-- facility grouping, consolidation reporting, batch jobs, and audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_mbc_ledger_entries, gl_eudr_mbc_overdraft_events,
--              gl_eudr_mbc_loss_records
-- Continuous Aggregates: 2 (hourly entries + daily balances)
-- Retention Policies: 3 (hypertables)
-- Indexes: 55
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V099: Creating AGENT-EUDR-011 Mass Balance Calculator tables...';

-- ============================================================================
-- 1. gl_eudr_mbc_ledgers — Ledger master records (one per facility+commodity+period)
-- ============================================================================
RAISE NOTICE 'V099 [1/12]: Creating gl_eudr_mbc_ledgers...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_ledgers (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    ledger_id               VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Facility & commodity
    facility_id             VARCHAR(100)    NOT NULL,
    commodity               VARCHAR(50)     NOT NULL,
    standard                VARCHAR(50)     NOT NULL DEFAULT 'eudr_default',

    -- Period reference
    period_id               UUID,

    -- Balance tracking
    current_balance         NUMERIC(18,6)   NOT NULL DEFAULT 0,
    total_inputs            NUMERIC(18,6)   NOT NULL DEFAULT 0,
    total_outputs           NUMERIC(18,6)   NOT NULL DEFAULT 0,
    total_losses            NUMERIC(18,6)   NOT NULL DEFAULT 0,
    total_waste             NUMERIC(18,6)   NOT NULL DEFAULT 0,
    utilization_rate        NUMERIC(8,4)    NOT NULL DEFAULT 0,

    -- Status
    status                  VARCHAR(20)     NOT NULL DEFAULT 'active',
        -- active, suspended, closed, archived

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_ledger_tenant ON gl_eudr_mbc_ledgers (tenant_id);
CREATE INDEX idx_eudr_mbc_ledger_facility ON gl_eudr_mbc_ledgers (facility_id);
CREATE INDEX idx_eudr_mbc_ledger_commodity ON gl_eudr_mbc_ledgers (commodity);
CREATE INDEX idx_eudr_mbc_ledger_fac_comm ON gl_eudr_mbc_ledgers (facility_id, commodity);
CREATE INDEX idx_eudr_mbc_ledger_status ON gl_eudr_mbc_ledgers (status);
CREATE INDEX idx_eudr_mbc_ledger_period ON gl_eudr_mbc_ledgers (period_id);
CREATE INDEX idx_eudr_mbc_ledger_standard ON gl_eudr_mbc_ledgers (standard);
CREATE INDEX idx_eudr_mbc_ledger_created ON gl_eudr_mbc_ledgers (created_at DESC);
CREATE INDEX idx_eudr_mbc_ledger_metadata ON gl_eudr_mbc_ledgers USING GIN (metadata);


-- ============================================================================
-- 2. gl_eudr_mbc_ledger_entries — Double-entry transaction records (hypertable)
-- ============================================================================
RAISE NOTICE 'V099 [2/12]: Creating gl_eudr_mbc_ledger_entries (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_ledger_entries (
    id                      BIGSERIAL,
    entry_id                VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Ledger reference
    ledger_id               UUID            NOT NULL,

    -- Entry classification
    entry_type              VARCHAR(30)     NOT NULL,
        -- input, output, loss, waste, adjustment, carry_forward
    batch_id                VARCHAR(100),
    quantity_kg             NUMERIC(18,6)   NOT NULL,

    -- Compliance
    compliance_status       VARCHAR(30)     NOT NULL DEFAULT 'compliant',
        -- compliant, non_compliant, pending, exempt

    -- Source/destination
    source_destination      VARCHAR(200),
    conversion_factor_applied NUMERIC(10,6),

    -- Operator & provenance
    operator_id             VARCHAR(100),
    provenance_hash         VARCHAR(64),
    notes                   TEXT,

    -- Timing
    timestamp               TIMESTAMPTZ     NOT NULL,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'gl_eudr_mbc_ledger_entries',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_mbc_entry_tenant ON gl_eudr_mbc_ledger_entries (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_entry_ledger ON gl_eudr_mbc_ledger_entries (ledger_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_entry_type ON gl_eudr_mbc_ledger_entries (entry_type, timestamp DESC);
CREATE INDEX idx_eudr_mbc_entry_batch ON gl_eudr_mbc_ledger_entries (batch_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_entry_compliance ON gl_eudr_mbc_ledger_entries (compliance_status, timestamp DESC);
CREATE INDEX idx_eudr_mbc_entry_operator ON gl_eudr_mbc_ledger_entries (operator_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_entry_metadata ON gl_eudr_mbc_ledger_entries USING GIN (metadata);


-- ============================================================================
-- 3. gl_eudr_mbc_credit_periods — Credit period tracking
-- ============================================================================
RAISE NOTICE 'V099 [3/12]: Creating gl_eudr_mbc_credit_periods...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_credit_periods (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    period_id               VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Facility & commodity
    facility_id             VARCHAR(100)    NOT NULL,
    commodity               VARCHAR(50)     NOT NULL,
    standard                VARCHAR(50)     NOT NULL DEFAULT 'eudr_default',

    -- Period boundaries
    start_date              TIMESTAMPTZ     NOT NULL,
    end_date                TIMESTAMPTZ     NOT NULL,

    -- Status
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- pending, active, closed, expired, extended

    -- Grace period & extensions
    grace_period_end        TIMESTAMPTZ,
    carry_forward_balance   NUMERIC(18,6)   NOT NULL DEFAULT 0,
    extension_reason        TEXT,
    extended_by             VARCHAR(100),
    extended_at             TIMESTAMPTZ,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Unique constraint: one active period per facility+commodity
CREATE UNIQUE INDEX idx_eudr_mbc_period_active_unique
    ON gl_eudr_mbc_credit_periods (facility_id, commodity)
    WHERE status = 'active';

CREATE INDEX idx_eudr_mbc_period_tenant ON gl_eudr_mbc_credit_periods (tenant_id);
CREATE INDEX idx_eudr_mbc_period_facility ON gl_eudr_mbc_credit_periods (facility_id);
CREATE INDEX idx_eudr_mbc_period_commodity ON gl_eudr_mbc_credit_periods (commodity);
CREATE INDEX idx_eudr_mbc_period_fac_comm ON gl_eudr_mbc_credit_periods (facility_id, commodity);
CREATE INDEX idx_eudr_mbc_period_status ON gl_eudr_mbc_credit_periods (status);
CREATE INDEX idx_eudr_mbc_period_dates ON gl_eudr_mbc_credit_periods (start_date, end_date);
CREATE INDEX idx_eudr_mbc_period_created ON gl_eudr_mbc_credit_periods (created_at DESC);
CREATE INDEX idx_eudr_mbc_period_metadata ON gl_eudr_mbc_credit_periods USING GIN (metadata);


-- ============================================================================
-- 4. gl_eudr_mbc_conversion_factors — Conversion factor definitions & validation
-- ============================================================================
RAISE NOTICE 'V099 [4/12]: Creating gl_eudr_mbc_conversion_factors...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_conversion_factors (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    factor_id               VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Commodity & process
    commodity               VARCHAR(50)     NOT NULL,
    process_name            VARCHAR(100)    NOT NULL,
    input_material          VARCHAR(100)    NOT NULL,
    output_material         VARCHAR(100)    NOT NULL,

    -- Yield & range
    yield_ratio             NUMERIC(10,6)   NOT NULL,
    acceptable_range_min    NUMERIC(10,6),
    acceptable_range_max    NUMERIC(10,6),
    deviation_percent       NUMERIC(8,4),

    -- Source & validation
    source                  VARCHAR(100),
    validation_status       VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- pending, approved, rejected, expired

    -- Facility-specific
    facility_id             VARCHAR(100),

    -- Approval
    approved_by             VARCHAR(100),
    approval_justification  TEXT,
    applied_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_cf_tenant ON gl_eudr_mbc_conversion_factors (tenant_id);
CREATE INDEX idx_eudr_mbc_cf_commodity ON gl_eudr_mbc_conversion_factors (commodity);
CREATE INDEX idx_eudr_mbc_cf_process ON gl_eudr_mbc_conversion_factors (process_name);
CREATE INDEX idx_eudr_mbc_cf_comm_proc ON gl_eudr_mbc_conversion_factors (commodity, process_name);
CREATE INDEX idx_eudr_mbc_cf_facility ON gl_eudr_mbc_conversion_factors (facility_id);
CREATE INDEX idx_eudr_mbc_cf_validation ON gl_eudr_mbc_conversion_factors (validation_status);
CREATE INDEX idx_eudr_mbc_cf_metadata ON gl_eudr_mbc_conversion_factors USING GIN (metadata);


-- ============================================================================
-- 5. gl_eudr_mbc_overdraft_events — Overdraft/negative balance events (hypertable)
-- ============================================================================
RAISE NOTICE 'V099 [5/12]: Creating gl_eudr_mbc_overdraft_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_overdraft_events (
    id                      BIGSERIAL,
    event_id                VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Ledger & facility
    ledger_id               UUID            NOT NULL,
    facility_id             VARCHAR(100)    NOT NULL,
    commodity               VARCHAR(50)     NOT NULL,

    -- Overdraft details
    severity                VARCHAR(20)     NOT NULL,
        -- low, medium, high, critical
    current_balance         NUMERIC(18,6)   NOT NULL,
    overdraft_amount        NUMERIC(18,6)   NOT NULL,

    -- Trigger & resolution
    trigger_entry_id        UUID,
    resolution_deadline     TIMESTAMPTZ,
    resolved                BOOLEAN         NOT NULL DEFAULT FALSE,
    resolved_at             TIMESTAMPTZ,
    resolution_entry_id     UUID,
    exemption_id            UUID,

    -- Timing
    timestamp               TIMESTAMPTZ     NOT NULL,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'gl_eudr_mbc_overdraft_events',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_mbc_od_tenant ON gl_eudr_mbc_overdraft_events (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_od_ledger ON gl_eudr_mbc_overdraft_events (ledger_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_od_facility ON gl_eudr_mbc_overdraft_events (facility_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_od_commodity ON gl_eudr_mbc_overdraft_events (commodity, timestamp DESC);
CREATE INDEX idx_eudr_mbc_od_severity ON gl_eudr_mbc_overdraft_events (severity, timestamp DESC);
CREATE INDEX idx_eudr_mbc_od_resolved ON gl_eudr_mbc_overdraft_events (resolved, timestamp DESC);
CREATE INDEX idx_eudr_mbc_od_deadline ON gl_eudr_mbc_overdraft_events (resolution_deadline);
CREATE INDEX idx_eudr_mbc_od_metadata ON gl_eudr_mbc_overdraft_events USING GIN (metadata);


-- ============================================================================
-- 6. gl_eudr_mbc_loss_records — Loss and waste recording (hypertable)
-- ============================================================================
RAISE NOTICE 'V099 [6/12]: Creating gl_eudr_mbc_loss_records (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_loss_records (
    id                      BIGSERIAL,
    record_id               VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Ledger reference
    ledger_id               UUID            NOT NULL,

    -- Loss classification
    loss_type               VARCHAR(30)     NOT NULL,
        -- processing, storage, transport, handling, spoilage, sampling
    waste_type              VARCHAR(30),
        -- byproduct, reject, contaminated, expired, damaged

    -- Quantities
    quantity_kg             NUMERIC(18,6)   NOT NULL,
    percentage              NUMERIC(8,4),

    -- Batch & process
    batch_id                VARCHAR(100),
    process_type            VARCHAR(100),

    -- Tolerance tracking
    within_tolerance        BOOLEAN         NOT NULL DEFAULT TRUE,
    expected_loss_percent   NUMERIC(8,4),
    max_tolerance_percent   NUMERIC(8,4),

    -- Facility & commodity
    facility_id             VARCHAR(100)    NOT NULL,
    commodity               VARCHAR(50)     NOT NULL,

    -- Waste certificate
    waste_certificate_ref   VARCHAR(200),

    -- Timing
    timestamp               TIMESTAMPTZ     NOT NULL,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable(
    'gl_eudr_mbc_loss_records',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_mbc_loss_tenant ON gl_eudr_mbc_loss_records (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_ledger ON gl_eudr_mbc_loss_records (ledger_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_type ON gl_eudr_mbc_loss_records (loss_type, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_facility ON gl_eudr_mbc_loss_records (facility_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_commodity ON gl_eudr_mbc_loss_records (commodity, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_fac_comm ON gl_eudr_mbc_loss_records (facility_id, commodity, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_batch ON gl_eudr_mbc_loss_records (batch_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_tolerance ON gl_eudr_mbc_loss_records (within_tolerance, timestamp DESC);
CREATE INDEX idx_eudr_mbc_loss_metadata ON gl_eudr_mbc_loss_records USING GIN (metadata);


-- ============================================================================
-- 7. gl_eudr_mbc_carry_forwards — Carry-forward balance management
-- ============================================================================
RAISE NOTICE 'V099 [7/12]: Creating gl_eudr_mbc_carry_forwards...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_carry_forwards (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    carry_forward_id        VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Period references
    from_period_id          UUID            NOT NULL,
    to_period_id            UUID            NOT NULL,

    -- Facility & commodity
    facility_id             VARCHAR(100)    NOT NULL,
    commodity               VARCHAR(50)     NOT NULL,

    -- Amounts
    amount_kg               NUMERIC(18,6)   NOT NULL,
    utilized_amount         NUMERIC(18,6)   NOT NULL DEFAULT 0,

    -- Status & expiry
    status                  VARCHAR(20)     NOT NULL DEFAULT 'active',
        -- active, utilized, expired, cancelled
    expiry_date             TIMESTAMPTZ,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_cf_fwd_tenant ON gl_eudr_mbc_carry_forwards (tenant_id);
CREATE INDEX idx_eudr_mbc_cf_fwd_facility ON gl_eudr_mbc_carry_forwards (facility_id);
CREATE INDEX idx_eudr_mbc_cf_fwd_commodity ON gl_eudr_mbc_carry_forwards (commodity);
CREATE INDEX idx_eudr_mbc_cf_fwd_fac_comm ON gl_eudr_mbc_carry_forwards (facility_id, commodity);
CREATE INDEX idx_eudr_mbc_cf_fwd_from ON gl_eudr_mbc_carry_forwards (from_period_id);
CREATE INDEX idx_eudr_mbc_cf_fwd_to ON gl_eudr_mbc_carry_forwards (to_period_id);
CREATE INDEX idx_eudr_mbc_cf_fwd_status ON gl_eudr_mbc_carry_forwards (status);
CREATE INDEX idx_eudr_mbc_cf_fwd_expiry ON gl_eudr_mbc_carry_forwards (expiry_date);
CREATE INDEX idx_eudr_mbc_cf_fwd_metadata ON gl_eudr_mbc_carry_forwards USING GIN (metadata);


-- ============================================================================
-- 8. gl_eudr_mbc_reconciliations — Period reconciliation results
-- ============================================================================
RAISE NOTICE 'V099 [8/12]: Creating gl_eudr_mbc_reconciliations...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_reconciliations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    reconciliation_id       VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Period reference
    period_id               UUID            NOT NULL,

    -- Facility & commodity
    facility_id             VARCHAR(100)    NOT NULL,
    commodity               VARCHAR(50)     NOT NULL,

    -- Balance comparison
    expected_balance        NUMERIC(18,6)   NOT NULL,
    recorded_balance        NUMERIC(18,6)   NOT NULL,
    variance_absolute       NUMERIC(18,6)   NOT NULL,
    variance_percent        NUMERIC(8,4)    NOT NULL,

    -- Classification
    classification          VARCHAR(20)     NOT NULL DEFAULT 'acceptable',
        -- acceptable, warning, critical, failed

    -- Anomalies
    anomalies_detected      INTEGER         NOT NULL DEFAULT 0,
    anomaly_details         JSONB           DEFAULT '[]',

    -- Sign-off
    signed_off_by           VARCHAR(100),
    signed_off_at           TIMESTAMPTZ,

    -- Status
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- pending, reviewed, approved, rejected

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_recon_tenant ON gl_eudr_mbc_reconciliations (tenant_id);
CREATE INDEX idx_eudr_mbc_recon_period ON gl_eudr_mbc_reconciliations (period_id);
CREATE INDEX idx_eudr_mbc_recon_facility ON gl_eudr_mbc_reconciliations (facility_id);
CREATE INDEX idx_eudr_mbc_recon_commodity ON gl_eudr_mbc_reconciliations (commodity);
CREATE INDEX idx_eudr_mbc_recon_fac_comm ON gl_eudr_mbc_reconciliations (facility_id, commodity);
CREATE INDEX idx_eudr_mbc_recon_class ON gl_eudr_mbc_reconciliations (classification);
CREATE INDEX idx_eudr_mbc_recon_status ON gl_eudr_mbc_reconciliations (status);
CREATE INDEX idx_eudr_mbc_recon_created ON gl_eudr_mbc_reconciliations (created_at DESC);
CREATE INDEX idx_eudr_mbc_recon_metadata ON gl_eudr_mbc_reconciliations USING GIN (metadata);


-- ============================================================================
-- 9. gl_eudr_mbc_facility_groups — Facility grouping for consolidation
-- ============================================================================
RAISE NOTICE 'V099 [9/12]: Creating gl_eudr_mbc_facility_groups...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_facility_groups (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id                VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Group details
    name                    VARCHAR(200)    NOT NULL,
    group_type              VARCHAR(30)     NOT NULL DEFAULT 'custom',
        -- custom, region, commodity, certification
    facility_ids            TEXT[]          NOT NULL DEFAULT '{}',
    description             TEXT,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_fg_tenant ON gl_eudr_mbc_facility_groups (tenant_id);
CREATE INDEX idx_eudr_mbc_fg_type ON gl_eudr_mbc_facility_groups (group_type);
CREATE INDEX idx_eudr_mbc_fg_name ON gl_eudr_mbc_facility_groups (name);
CREATE INDEX idx_eudr_mbc_fg_facility_ids ON gl_eudr_mbc_facility_groups USING GIN (facility_ids);
CREATE INDEX idx_eudr_mbc_fg_metadata ON gl_eudr_mbc_facility_groups USING GIN (metadata);


-- ============================================================================
-- 10. gl_eudr_mbc_consolidation_reports — Multi-facility consolidation reports
-- ============================================================================
RAISE NOTICE 'V099 [10/12]: Creating gl_eudr_mbc_consolidation_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_consolidation_reports (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id               VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Report classification
    report_type             VARCHAR(30)     NOT NULL,
        -- facility_summary, group_summary, period_report, compliance_report
    report_format           VARCHAR(20)     NOT NULL DEFAULT 'json',
        -- json, csv, pdf

    -- Scope
    facility_ids            TEXT[]          NOT NULL DEFAULT '{}',
    group_id                UUID,

    -- Report data
    data                    JSONB           DEFAULT '{}',

    -- Provenance & timing
    provenance_hash         VARCHAR(64),
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_cr_tenant ON gl_eudr_mbc_consolidation_reports (tenant_id);
CREATE INDEX idx_eudr_mbc_cr_type ON gl_eudr_mbc_consolidation_reports (report_type);
CREATE INDEX idx_eudr_mbc_cr_group ON gl_eudr_mbc_consolidation_reports (group_id);
CREATE INDEX idx_eudr_mbc_cr_generated ON gl_eudr_mbc_consolidation_reports (generated_at DESC);
CREATE INDEX idx_eudr_mbc_cr_facility_ids ON gl_eudr_mbc_consolidation_reports USING GIN (facility_ids);
CREATE INDEX idx_eudr_mbc_cr_data ON gl_eudr_mbc_consolidation_reports USING GIN (data);
CREATE INDEX idx_eudr_mbc_cr_metadata ON gl_eudr_mbc_consolidation_reports USING GIN (metadata);


-- ============================================================================
-- 11. gl_eudr_mbc_batch_jobs — Batch processing jobs
-- ============================================================================
RAISE NOTICE 'V099 [11/12]: Creating gl_eudr_mbc_batch_jobs...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_batch_jobs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id                  VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Job config
    job_type                VARCHAR(50)     NOT NULL,
        -- ledger_creation, entry_recording, reconciliation, carry_forward,
        -- report_generation, overdraft_check, loss_recording
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- pending, running, completed, failed, cancelled

    -- Progress
    total_items             INTEGER         NOT NULL DEFAULT 0,
    processed_items         INTEGER         NOT NULL DEFAULT 0,
    failed_items            INTEGER         NOT NULL DEFAULT 0,

    -- Parameters & results
    parameters              JSONB           DEFAULT '{}',
    results                 JSONB           DEFAULT '{}',
    errors                  JSONB           DEFAULT '[]',
    error_message           TEXT,

    -- Timing
    submitted_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,

    -- Metadata
    provenance_hash         VARCHAR(64),
    submitted_by            UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_jobs_tenant ON gl_eudr_mbc_batch_jobs (tenant_id);
CREATE INDEX idx_eudr_mbc_jobs_type ON gl_eudr_mbc_batch_jobs (job_type);
CREATE INDEX idx_eudr_mbc_jobs_status ON gl_eudr_mbc_batch_jobs (status);
CREATE INDEX idx_eudr_mbc_jobs_submitted ON gl_eudr_mbc_batch_jobs (submitted_at DESC);


-- ============================================================================
-- 12. gl_eudr_mbc_audit_log — Immutable audit trail
-- ============================================================================
RAISE NOTICE 'V099 [12/12]: Creating gl_eudr_mbc_audit_log...';

CREATE TABLE IF NOT EXISTS gl_eudr_mbc_audit_log (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,

    -- Entity reference
    entity_type             VARCHAR(50)     NOT NULL,
        -- ledger, ledger_entry, credit_period, conversion_factor,
        -- overdraft_event, loss_record, carry_forward, reconciliation,
        -- facility_group, consolidation_report, batch_job
    entity_id               VARCHAR(100)    NOT NULL,

    -- Action
    action                  VARCHAR(50)     NOT NULL,
        -- created, updated, deleted, closed, reconciled, approved, rejected,
        -- overdraft_triggered, overdraft_resolved, carry_forward_applied
    actor                   VARCHAR(200),

    -- Hash chain
    previous_hash           VARCHAR(64),
    current_hash            VARCHAR(64)     NOT NULL,

    -- Details
    details                 JSONB           DEFAULT '{}',

    -- Timing
    timestamp               TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_mbc_audit_tenant ON gl_eudr_mbc_audit_log (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_audit_entity_type ON gl_eudr_mbc_audit_log (entity_type, timestamp DESC);
CREATE INDEX idx_eudr_mbc_audit_entity_id ON gl_eudr_mbc_audit_log (entity_id, timestamp DESC);
CREATE INDEX idx_eudr_mbc_audit_action ON gl_eudr_mbc_audit_log (action, timestamp DESC);
CREATE INDEX idx_eudr_mbc_audit_hash ON gl_eudr_mbc_audit_log (current_hash);


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================
RAISE NOTICE 'V099: Creating continuous aggregates...';

-- 1. Hourly ledger entry counts by type and commodity
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_mbc_hourly_entries
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp)    AS bucket,
    tenant_id,
    ledger_id,
    entry_type,
    compliance_status,
    COUNT(*)                            AS entry_count,
    SUM(quantity_kg)                    AS total_quantity_kg,
    COUNT(DISTINCT batch_id)            AS unique_batches,
    AVG(quantity_kg)                    AS avg_quantity_kg
FROM gl_eudr_mbc_ledger_entries
GROUP BY bucket, tenant_id, ledger_id, entry_type, compliance_status
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_mbc_hourly_entries',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- 2. Daily balance snapshots by facility and commodity
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_mbc_daily_balances
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp)     AS bucket,
    tenant_id,
    ledger_id,
    entry_type,
    COUNT(*)                            AS entry_count,
    SUM(CASE WHEN entry_type = 'input' THEN quantity_kg ELSE 0 END) AS total_inputs_kg,
    SUM(CASE WHEN entry_type = 'output' THEN quantity_kg ELSE 0 END) AS total_outputs_kg,
    SUM(CASE WHEN entry_type IN ('loss', 'waste') THEN quantity_kg ELSE 0 END) AS total_losses_kg,
    SUM(quantity_kg)                    AS net_quantity_kg
FROM gl_eudr_mbc_ledger_entries
GROUP BY bucket, tenant_id, ledger_id, entry_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_mbc_daily_balances',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies
-- ============================================================================
RAISE NOTICE 'V099: Adding retention policies (5 years)...';

SELECT add_retention_policy('gl_eudr_mbc_ledger_entries',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_mbc_overdraft_events',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_mbc_loss_records',
    INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================
RAISE NOTICE 'V099: Adding table comments...';

COMMENT ON TABLE gl_eudr_mbc_ledgers IS 'AGENT-EUDR-011: Mass balance ledger master records with running balance tracking per facility, commodity, and period';
COMMENT ON TABLE gl_eudr_mbc_ledger_entries IS 'AGENT-EUDR-011: Double-entry ledger transactions with compliance status, batch tracking, and conversion factors (hypertable)';
COMMENT ON TABLE gl_eudr_mbc_credit_periods IS 'AGENT-EUDR-011: Credit period definitions with grace periods, extensions, and carry-forward balances';
COMMENT ON TABLE gl_eudr_mbc_conversion_factors IS 'AGENT-EUDR-011: Conversion factor definitions with yield ratios, acceptable ranges, and validation status';
COMMENT ON TABLE gl_eudr_mbc_overdraft_events IS 'AGENT-EUDR-011: Overdraft/negative balance events with severity classification and resolution tracking (hypertable)';
COMMENT ON TABLE gl_eudr_mbc_loss_records IS 'AGENT-EUDR-011: Loss and waste records with tolerance tracking and waste certificate references (hypertable)';
COMMENT ON TABLE gl_eudr_mbc_carry_forwards IS 'AGENT-EUDR-011: Carry-forward balance management between credit periods with utilization tracking';
COMMENT ON TABLE gl_eudr_mbc_reconciliations IS 'AGENT-EUDR-011: Period reconciliation results with variance analysis, anomaly detection, and sign-off workflow';
COMMENT ON TABLE gl_eudr_mbc_facility_groups IS 'AGENT-EUDR-011: Facility grouping definitions for multi-facility consolidation and reporting';
COMMENT ON TABLE gl_eudr_mbc_consolidation_reports IS 'AGENT-EUDR-011: Multi-facility consolidation reports with provenance hashing and flexible output formats';
COMMENT ON TABLE gl_eudr_mbc_batch_jobs IS 'AGENT-EUDR-011: Batch processing jobs for ledger operations, reconciliation, and report generation';
COMMENT ON TABLE gl_eudr_mbc_audit_log IS 'AGENT-EUDR-011: Immutable audit trail with hash chain for all mass balance calculator data changes';

RAISE NOTICE 'V099: AGENT-EUDR-011 Mass Balance Calculator migration complete.';

COMMIT;
