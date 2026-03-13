-- ============================================================================
-- V097: AGENT-EUDR-009 Chain of Custody Agent
-- ============================================================================
-- Creates tables for chain-of-custody tracking, batch lifecycle management,
-- mass balance accounting, transformation tracking, and compliance reporting.
--
-- Tables: 10 (6 regular + 4 hypertables)
-- Hypertables: gl_eudr_coc_custody_events, gl_eudr_coc_batch_operations,
--              gl_eudr_coc_mass_balance_ledger, gl_eudr_coc_transformations
-- Continuous Aggregates: 2 (daily event summary + daily mass balance summary)
-- Retention Policies: 4 (hypertables)
-- Indexes: 34
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. gl_eudr_coc_custody_events — Custody event records (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_custody_events (
    id                  BIGSERIAL,
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    recorded_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Event classification
    event_type          VARCHAR(30)     NOT NULL,
        -- transfer, receipt, storage_in, storage_out, processing_in,
        -- processing_out, export, import, inspection, sampling
    event_status        VARCHAR(20)     NOT NULL DEFAULT 'active',
        -- active, amended, superseded

    -- Batch reference
    batch_id            UUID            NOT NULL,
    parent_event_id     UUID,           -- predecessor in chain

    -- Actors
    sender_id           UUID,
    sender_name         TEXT,
    receiver_id         UUID,
    receiver_name       TEXT,

    -- Location
    facility_id         UUID,
    facility_name       TEXT,
    location_country    VARCHAR(3),
    location_latitude   DOUBLE PRECISION,
    location_longitude  DOUBLE PRECISION,

    -- Quantity
    quantity            DOUBLE PRECISION NOT NULL,
    unit                VARCHAR(20)     NOT NULL DEFAULT 'kg',
    commodity           VARCHAR(30)     NOT NULL,

    -- Documents
    document_refs       UUID[]          DEFAULT '{}',

    -- Timestamps
    event_timestamp     TIMESTAMPTZ     NOT NULL,
    custody_start       TIMESTAMPTZ,
    custody_end         TIMESTAMPTZ,

    -- Amendment
    amends_event_id     UUID,           -- NULL unless this is an amendment
    amendment_reason    TEXT,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    PRIMARY KEY (id, recorded_at)
);

SELECT create_hypertable(
    'gl_eudr_coc_custody_events',
    'recorded_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_coc_events_tenant ON gl_eudr_coc_custody_events (tenant_id, recorded_at DESC);
CREATE INDEX idx_eudr_coc_events_batch ON gl_eudr_coc_custody_events (batch_id, recorded_at DESC);
CREATE INDEX idx_eudr_coc_events_type ON gl_eudr_coc_custody_events (event_type, recorded_at DESC);
CREATE INDEX idx_eudr_coc_events_commodity ON gl_eudr_coc_custody_events (commodity, recorded_at DESC);
CREATE INDEX idx_eudr_coc_events_sender ON gl_eudr_coc_custody_events (sender_id, recorded_at DESC);
CREATE INDEX idx_eudr_coc_events_receiver ON gl_eudr_coc_custody_events (receiver_id, recorded_at DESC);
CREATE INDEX idx_eudr_coc_events_facility ON gl_eudr_coc_custody_events (facility_id, recorded_at DESC);


-- ============================================================================
-- 2. gl_eudr_coc_batches — Batch master records
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_batches (
    id                  BIGSERIAL       PRIMARY KEY,
    batch_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,

    -- Batch identification
    batch_code          TEXT,           -- human-readable batch code
    commodity           VARCHAR(30)     NOT NULL,
    derived_product     TEXT,

    -- Quantity
    initial_quantity    DOUBLE PRECISION NOT NULL,
    current_quantity    DOUBLE PRECISION NOT NULL,
    unit                VARCHAR(20)     NOT NULL DEFAULT 'kg',
    quality_grade       VARCHAR(20),

    -- Origin
    origin_country      VARCHAR(3),
    production_date     DATE,
    harvest_season      TEXT,

    -- Status
    status              VARCHAR(20)     NOT NULL DEFAULT 'created',
        -- created, in_transit, at_facility, processing, processed,
        -- dispatched, delivered, consumed
    coc_model           VARCHAR(30),
        -- identity_preserved, segregated, mass_balance, controlled_blending

    -- Genealogy
    parent_batch_ids    UUID[]          DEFAULT '{}',
    child_batch_ids     UUID[]          DEFAULT '{}',
    operation_type      VARCHAR(20),    -- split, merge, blend, transform, NULL for root

    -- Current location
    current_facility_id UUID,
    current_facility_name TEXT,

    -- Compliance
    compliance_status   VARCHAR(20)     DEFAULT 'unverified',
        -- compliant, non_compliant, unverified, pending
    deforestation_free  BOOLEAN         DEFAULT FALSE,

    -- Provenance
    provenance_hash     VARCHAR(64),
    version             INTEGER         NOT NULL DEFAULT 1,
    created_by          UUID,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_eudr_coc_batch UNIQUE (batch_id)
);

CREATE INDEX idx_eudr_coc_batches_tenant ON gl_eudr_coc_batches (tenant_id);
CREATE INDEX idx_eudr_coc_batches_commodity ON gl_eudr_coc_batches (commodity);
CREATE INDEX idx_eudr_coc_batches_status ON gl_eudr_coc_batches (status);
CREATE INDEX idx_eudr_coc_batches_coc ON gl_eudr_coc_batches (coc_model);
CREATE INDEX idx_eudr_coc_batches_compliance ON gl_eudr_coc_batches (compliance_status);
CREATE INDEX idx_eudr_coc_batches_parents ON gl_eudr_coc_batches USING GIN (parent_batch_ids);


-- ============================================================================
-- 3. gl_eudr_coc_batch_operations — Split/merge/blend/transform (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_batch_operations (
    id                  BIGSERIAL,
    operation_id        UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    performed_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Operation type
    operation_type      VARCHAR(20)     NOT NULL,
        -- split, merge, blend, transform
    facility_id         UUID,
    facility_name       TEXT,

    -- Input/Output
    input_batch_ids     UUID[]          NOT NULL,
    output_batch_ids    UUID[]          NOT NULL,
    input_total_qty     DOUBLE PRECISION NOT NULL,
    output_total_qty    DOUBLE PRECISION NOT NULL,
    waste_qty           DOUBLE PRECISION DEFAULT 0.0,

    -- Allocation
    allocation_method   VARCHAR(30),    -- proportional, volume_weighted, equal
    blend_ratios        JSONB           DEFAULT '{}',

    -- Process (for transform operations)
    process_type        VARCHAR(50),
    yield_ratio_actual  DOUBLE PRECISION,
    yield_ratio_expected DOUBLE PRECISION,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    PRIMARY KEY (id, performed_at)
);

SELECT create_hypertable(
    'gl_eudr_coc_batch_operations',
    'performed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_coc_ops_tenant ON gl_eudr_coc_batch_operations (tenant_id, performed_at DESC);
CREATE INDEX idx_eudr_coc_ops_type ON gl_eudr_coc_batch_operations (operation_type, performed_at DESC);
CREATE INDEX idx_eudr_coc_ops_facility ON gl_eudr_coc_batch_operations (facility_id, performed_at DESC);
CREATE INDEX idx_eudr_coc_ops_inputs ON gl_eudr_coc_batch_operations USING GIN (input_batch_ids);
CREATE INDEX idx_eudr_coc_ops_outputs ON gl_eudr_coc_batch_operations USING GIN (output_batch_ids);


-- ============================================================================
-- 4. gl_eudr_coc_batch_origins — Origin plot allocations per batch
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_batch_origins (
    id                  BIGSERIAL       PRIMARY KEY,
    origin_id           UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    batch_id            UUID            NOT NULL,

    -- Origin plot
    plot_id             UUID            NOT NULL,
    plot_geolocation    TEXT,           -- GPS coordinates or polygon WKT
    country_code        VARCHAR(3)      NOT NULL,
    admin_region        TEXT,

    -- Allocation
    allocation_pct      DOUBLE PRECISION NOT NULL,  -- 0.0 to 100.0
    allocated_qty       DOUBLE PRECISION,
    unit                VARCHAR(20)     DEFAULT 'kg',

    -- Verification
    gps_verified        BOOLEAN         DEFAULT FALSE,
    deforestation_free  BOOLEAN         DEFAULT FALSE,
    verification_date   DATE,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_eudr_coc_origin UNIQUE (origin_id)
);

CREATE INDEX idx_eudr_coc_origins_tenant ON gl_eudr_coc_batch_origins (tenant_id);
CREATE INDEX idx_eudr_coc_origins_batch ON gl_eudr_coc_batch_origins (batch_id);
CREATE INDEX idx_eudr_coc_origins_plot ON gl_eudr_coc_batch_origins (plot_id);
CREATE INDEX idx_eudr_coc_origins_country ON gl_eudr_coc_batch_origins (country_code);


-- ============================================================================
-- 5. gl_eudr_coc_mass_balance_ledger — Input/output mass balance (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_mass_balance_ledger (
    id                  BIGSERIAL,
    entry_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    entry_date          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Facility + commodity
    facility_id         UUID            NOT NULL,
    commodity           VARCHAR(30)     NOT NULL,

    -- Ledger entry
    entry_type          VARCHAR(20)     NOT NULL,
        -- input, output, adjustment, carry_forward, loss, waste
    batch_id            UUID,
    quantity            DOUBLE PRECISION NOT NULL,
    unit                VARCHAR(20)     NOT NULL DEFAULT 'kg',
    compliance_status   VARCHAR(20)     DEFAULT 'compliant',
        -- compliant, non_compliant, unverified

    -- Period
    credit_period_start DATE,
    credit_period_end   DATE,

    -- Conversion
    conversion_factor   DOUBLE PRECISION,
    process_type        VARCHAR(50),
    pre_conversion_qty  DOUBLE PRECISION,

    -- Running balance
    running_balance     DOUBLE PRECISION,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    PRIMARY KEY (id, entry_date)
);

SELECT create_hypertable(
    'gl_eudr_coc_mass_balance_ledger',
    'entry_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_coc_mb_tenant ON gl_eudr_coc_mass_balance_ledger (tenant_id, entry_date DESC);
CREATE INDEX idx_eudr_coc_mb_facility ON gl_eudr_coc_mass_balance_ledger (facility_id, entry_date DESC);
CREATE INDEX idx_eudr_coc_mb_commodity ON gl_eudr_coc_mass_balance_ledger (commodity, entry_date DESC);
CREATE INDEX idx_eudr_coc_mb_type ON gl_eudr_coc_mass_balance_ledger (entry_type, entry_date DESC);
CREATE INDEX idx_eudr_coc_mb_batch ON gl_eudr_coc_mass_balance_ledger (batch_id, entry_date DESC);


-- ============================================================================
-- 6. gl_eudr_coc_transformations — Processing step records (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_transformations (
    id                  BIGSERIAL,
    transform_id        UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    performed_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Process details
    process_type        VARCHAR(50)     NOT NULL,
    facility_id         UUID            NOT NULL,
    facility_name       TEXT,

    -- Input/Output
    input_batch_ids     UUID[]          NOT NULL,
    output_batch_ids    UUID[]          NOT NULL,
    input_commodity     VARCHAR(30)     NOT NULL,
    output_commodity    VARCHAR(30)     NOT NULL,
    input_quantity      DOUBLE PRECISION NOT NULL,
    output_quantity     DOUBLE PRECISION NOT NULL,
    unit                VARCHAR(20)     NOT NULL DEFAULT 'kg',

    -- Yield
    expected_yield      DOUBLE PRECISION,
    actual_yield        DOUBLE PRECISION,
    yield_variance_pct  DOUBLE PRECISION,

    -- By-products
    by_products         JSONB           DEFAULT '[]',
        -- [{"commodity": "...", "quantity": ..., "batch_id": "..."}]
    waste_quantity      DOUBLE PRECISION DEFAULT 0.0,
    waste_type          TEXT,

    -- Quality
    input_quality       VARCHAR(20),
    output_quality      VARCHAR(20),

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    PRIMARY KEY (id, performed_at)
);

SELECT create_hypertable(
    'gl_eudr_coc_transformations',
    'performed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_coc_transform_tenant ON gl_eudr_coc_transformations (tenant_id, performed_at DESC);
CREATE INDEX idx_eudr_coc_transform_process ON gl_eudr_coc_transformations (process_type, performed_at DESC);
CREATE INDEX idx_eudr_coc_transform_facility ON gl_eudr_coc_transformations (facility_id, performed_at DESC);
CREATE INDEX idx_eudr_coc_transform_in_commodity ON gl_eudr_coc_transformations (input_commodity, performed_at DESC);
CREATE INDEX idx_eudr_coc_transform_out_commodity ON gl_eudr_coc_transformations (output_commodity, performed_at DESC);


-- ============================================================================
-- 7. gl_eudr_coc_documents — Document metadata and linkage
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_documents (
    id                  BIGSERIAL       PRIMARY KEY,
    document_id         UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,

    -- Document details
    document_type       VARCHAR(50)     NOT NULL,
        -- bill_of_lading, packing_list, commercial_invoice, certificate_of_origin,
        -- phytosanitary_cert, weight_cert, quality_cert, customs_declaration,
        -- transport_waybill, warehouse_receipt, fumigation_cert, insurance_cert,
        -- dds_reference, delivery_note, purchase_order
    reference_number    TEXT,
    issuer              TEXT,
    issue_date          DATE,
    expiry_date         DATE,

    -- Linkage
    event_ids           UUID[]          NOT NULL DEFAULT '{}',
    batch_ids           UUID[]          DEFAULT '{}',

    -- Quantities (for cross-reference)
    document_quantity   DOUBLE PRECISION,
    document_unit       VARCHAR(20),

    -- Integrity
    content_hash        VARCHAR(64),    -- SHA-256 of document content
    tamper_detected     BOOLEAN         DEFAULT FALSE,

    -- Status
    status              VARCHAR(20)     NOT NULL DEFAULT 'active',
        -- active, expired, superseded, revoked
    verified            BOOLEAN         DEFAULT FALSE,
    verified_by         UUID,
    verified_at         TIMESTAMPTZ,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_eudr_coc_document UNIQUE (document_id)
);

CREATE INDEX idx_eudr_coc_docs_tenant ON gl_eudr_coc_documents (tenant_id);
CREATE INDEX idx_eudr_coc_docs_type ON gl_eudr_coc_documents (document_type);
CREATE INDEX idx_eudr_coc_docs_status ON gl_eudr_coc_documents (status);
CREATE INDEX idx_eudr_coc_docs_events ON gl_eudr_coc_documents USING GIN (event_ids);
CREATE INDEX idx_eudr_coc_docs_batches ON gl_eudr_coc_documents USING GIN (batch_ids);
CREATE INDEX idx_eudr_coc_docs_expiry ON gl_eudr_coc_documents (expiry_date);


-- ============================================================================
-- 8. gl_eudr_coc_chain_verifications — Chain integrity verification results
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_chain_verifications (
    id                  BIGSERIAL       PRIMARY KEY,
    verification_id     UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    verified_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Target
    batch_id            UUID            NOT NULL,
    commodity           VARCHAR(30),

    -- Results
    verification_status VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- passed, failed, partial, pending
    completeness_score  DOUBLE PRECISION DEFAULT 0.0,  -- 0-100

    -- Dimension scores
    temporal_continuity_score   DOUBLE PRECISION DEFAULT 0.0,
    actor_continuity_score      DOUBLE PRECISION DEFAULT 0.0,
    location_continuity_score   DOUBLE PRECISION DEFAULT 0.0,
    mass_conservation_score     DOUBLE PRECISION DEFAULT 0.0,
    origin_preservation_score   DOUBLE PRECISION DEFAULT 0.0,
    document_coverage_score     DOUBLE PRECISION DEFAULT 0.0,

    -- Issues found
    gaps_found          INTEGER         DEFAULT 0,
    orphans_found       INTEGER         DEFAULT 0,
    mass_violations     INTEGER         DEFAULT 0,
    circular_deps       INTEGER         DEFAULT 0,
    missing_documents   INTEGER         DEFAULT 0,

    -- Evidence
    findings            JSONB           DEFAULT '[]',
    certificate_data    JSONB           DEFAULT '{}',

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    CONSTRAINT uq_eudr_coc_verification UNIQUE (verification_id)
);

CREATE INDEX idx_eudr_coc_verify_tenant ON gl_eudr_coc_chain_verifications (tenant_id);
CREATE INDEX idx_eudr_coc_verify_batch ON gl_eudr_coc_chain_verifications (batch_id);
CREATE INDEX idx_eudr_coc_verify_status ON gl_eudr_coc_chain_verifications (verification_status);
CREATE INDEX idx_eudr_coc_verify_score ON gl_eudr_coc_chain_verifications (completeness_score DESC);


-- ============================================================================
-- 9. gl_eudr_coc_batch_jobs — Batch processing jobs
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_batch_jobs (
    id                  BIGSERIAL       PRIMARY KEY,
    batch_job_id        UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,

    -- Job config
    job_type            VARCHAR(50)     NOT NULL,
        -- event_import, batch_import, verification, mass_balance_reconcile,
        -- report_generation, document_import
    input_config        JSONB           NOT NULL DEFAULT '{}',
    parameters          JSONB           DEFAULT '{}',

    -- Progress
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- pending, running, completed, failed, cancelled
    total_items         INTEGER         DEFAULT 0,
    processed_items     INTEGER         DEFAULT 0,
    failed_items        INTEGER         DEFAULT 0,
    progress_pct        DOUBLE PRECISION DEFAULT 0.0,

    -- Results
    result_summary      JSONB           DEFAULT '{}',
    error_details       JSONB           DEFAULT '[]',

    -- Timing
    submitted_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_ms         BIGINT,

    -- Provenance
    provenance_hash     VARCHAR(64),
    submitted_by        UUID,

    CONSTRAINT uq_eudr_coc_batch_job UNIQUE (batch_job_id)
);

CREATE INDEX idx_eudr_coc_jobs_tenant ON gl_eudr_coc_batch_jobs (tenant_id);
CREATE INDEX idx_eudr_coc_jobs_status ON gl_eudr_coc_batch_jobs (status);
CREATE INDEX idx_eudr_coc_jobs_type ON gl_eudr_coc_batch_jobs (job_type);


-- ============================================================================
-- 10. gl_eudr_coc_audit_log — Immutable audit trail
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_coc_audit_log (
    id                  BIGSERIAL       PRIMARY KEY,
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    recorded_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Event classification
    event_type          VARCHAR(80)     NOT NULL,
        -- event.recorded, event.amended, batch.created, batch.split, batch.merged,
        -- batch.blended, batch.status_changed, model.assigned, model.validated,
        -- balance.input, balance.output, balance.reconciled, balance.overdraft,
        -- transform.recorded, document.linked, document.verified,
        -- chain.verified, chain.failed, report.generated, report.downloaded
    event_category      VARCHAR(30)     NOT NULL,
        -- custody_event, batch, model, balance, transformation, document,
        -- verification, report

    -- Target
    entity_type         VARCHAR(50)     NOT NULL,
    entity_id           UUID            NOT NULL,
    batch_id            UUID,

    -- Change details
    actor_id            UUID,
    actor_type          VARCHAR(20)     DEFAULT 'user',
    changes             JSONB           DEFAULT '{}',
    previous_state      JSONB           DEFAULT '{}',
    new_state           JSONB           DEFAULT '{}',

    -- Context
    ip_address          INET,
    user_agent          TEXT,
    correlation_id      UUID,

    -- Provenance
    provenance_hash     VARCHAR(64)
);

CREATE INDEX idx_eudr_coc_audit_tenant ON gl_eudr_coc_audit_log (tenant_id, recorded_at DESC);
CREATE INDEX idx_eudr_coc_audit_type ON gl_eudr_coc_audit_log (event_type, recorded_at DESC);
CREATE INDEX idx_eudr_coc_audit_entity ON gl_eudr_coc_audit_log (entity_type, entity_id, recorded_at DESC);
CREATE INDEX idx_eudr_coc_audit_batch ON gl_eudr_coc_audit_log (batch_id, recorded_at DESC);
CREATE INDEX idx_eudr_coc_audit_actor ON gl_eudr_coc_audit_log (actor_id, recorded_at DESC);


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================

-- Daily custody event summary
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_coc_daily_event_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', recorded_at)   AS bucket,
    tenant_id,
    event_type,
    commodity,
    COUNT(*)                            AS event_count,
    SUM(quantity)                       AS total_quantity,
    COUNT(DISTINCT batch_id)            AS unique_batches,
    COUNT(DISTINCT facility_id)         AS unique_facilities
FROM gl_eudr_coc_custody_events
GROUP BY bucket, tenant_id, event_type, commodity
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_coc_daily_event_summary',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- Daily mass balance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_coc_daily_balance_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', entry_date)    AS bucket,
    tenant_id,
    facility_id,
    commodity,
    entry_type,
    COUNT(*)                            AS entry_count,
    SUM(quantity)                       AS total_quantity,
    AVG(running_balance)                AS avg_running_balance
FROM gl_eudr_coc_mass_balance_ledger
GROUP BY bucket, tenant_id, facility_id, commodity, entry_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_coc_daily_balance_summary',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies
-- ============================================================================

SELECT add_retention_policy('gl_eudr_coc_custody_events',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_coc_batch_operations',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_coc_mass_balance_ledger',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_coc_transformations',
    INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE gl_eudr_coc_custody_events IS 'AGENT-EUDR-009: Custody event records tracking every transfer, receipt, storage, and processing event (hypertable)';
COMMENT ON TABLE gl_eudr_coc_batches IS 'AGENT-EUDR-009: Batch master records with commodity, origin, status, and genealogy data';
COMMENT ON TABLE gl_eudr_coc_batch_operations IS 'AGENT-EUDR-009: Batch split/merge/blend/transform operations with input/output tracking (hypertable)';
COMMENT ON TABLE gl_eudr_coc_batch_origins IS 'AGENT-EUDR-009: Origin plot allocations per batch with GPS verification status';
COMMENT ON TABLE gl_eudr_coc_mass_balance_ledger IS 'AGENT-EUDR-009: Mass balance input/output ledger per facility per commodity per credit period (hypertable)';
COMMENT ON TABLE gl_eudr_coc_transformations IS 'AGENT-EUDR-009: Processing step records with yield ratios, by-products, and waste tracking (hypertable)';
COMMENT ON TABLE gl_eudr_coc_documents IS 'AGENT-EUDR-009: Document metadata and linkage to custody events and batches';
COMMENT ON TABLE gl_eudr_coc_chain_verifications IS 'AGENT-EUDR-009: Chain integrity verification results with 6 dimension scores';
COMMENT ON TABLE gl_eudr_coc_batch_jobs IS 'AGENT-EUDR-009: Batch processing jobs for event import, verification, and reporting';
COMMENT ON TABLE gl_eudr_coc_audit_log IS 'AGENT-EUDR-009: Immutable audit trail for all chain-of-custody data changes';

COMMIT;
