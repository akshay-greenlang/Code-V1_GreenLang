-- ============================================================================
-- V098: AGENT-EUDR-010 Segregation Verifier Agent
-- ============================================================================
-- Creates tables for segregation control point management, storage zone
-- tracking, transport verification, processing line monitoring, contamination
-- detection, labeling, facility assessments, and audit trails.
--
-- Tables: 12 (8 regular + 4 hypertables)
-- Hypertables: gl_eudr_sgv_storage_events, gl_eudr_sgv_transport_verifications,
--              gl_eudr_sgv_changeover_records, gl_eudr_sgv_contamination_events
-- Continuous Aggregates: 2 (hourly storage events + daily contamination events)
-- Retention Policies: 4 (hypertables)
-- Indexes: 42
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V098: Creating AGENT-EUDR-010 Segregation Verifier tables...';

-- ============================================================================
-- 1. gl_eudr_sgv_segregation_points — Segregation control point master records
-- ============================================================================
RAISE NOTICE 'V098 [1/12]: Creating gl_eudr_sgv_segregation_points...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_segregation_points (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scp_id                  VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Facility & location
    facility_id             VARCHAR(100)    NOT NULL,
    location_lat            DOUBLE PRECISION,
    location_lon            DOUBLE PRECISION,

    -- Classification
    scp_type                VARCHAR(50)     NOT NULL,
        -- storage, transport, processing, handling, loading_unloading
    commodity               VARCHAR(50)     NOT NULL,
    capacity_kg             DOUBLE PRECISION DEFAULT 0,
    segregation_method      VARCHAR(50)     NOT NULL,

    -- Status & risk
    status                  VARCHAR(30)     NOT NULL DEFAULT 'unverified',
    risk_classification     VARCHAR(20)     DEFAULT 'medium',
    compliance_score        DOUBLE PRECISION DEFAULT 0,

    -- Verification schedule
    verification_date       TIMESTAMPTZ,
    next_verification_date  TIMESTAMPTZ,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_sgv_scp_tenant ON gl_eudr_sgv_segregation_points (tenant_id);
CREATE INDEX idx_eudr_sgv_scp_facility ON gl_eudr_sgv_segregation_points (facility_id);
CREATE INDEX idx_eudr_sgv_scp_type ON gl_eudr_sgv_segregation_points (scp_type);
CREATE INDEX idx_eudr_sgv_scp_commodity ON gl_eudr_sgv_segregation_points (commodity);
CREATE INDEX idx_eudr_sgv_scp_status ON gl_eudr_sgv_segregation_points (status);
CREATE INDEX idx_eudr_sgv_scp_risk ON gl_eudr_sgv_segregation_points (risk_classification);


-- ============================================================================
-- 2. gl_eudr_sgv_storage_zones — Storage zone definitions
-- ============================================================================
RAISE NOTICE 'V098 [2/12]: Creating gl_eudr_sgv_storage_zones...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_storage_zones (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id                 VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Zone details
    facility_id             VARCHAR(100)    NOT NULL,
    zone_name               TEXT,
    storage_type            VARCHAR(50),

    -- Status & capacity
    compliance_status       VARCHAR(30),
    barrier_type            VARCHAR(50),
    capacity_kg             DOUBLE PRECISION DEFAULT 0,
    current_occupancy_kg    DOUBLE PRECISION DEFAULT 0,

    -- Adjacency
    adjacent_zones          JSONB           DEFAULT '[]',

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_sgv_zone_tenant ON gl_eudr_sgv_storage_zones (tenant_id);
CREATE INDEX idx_eudr_sgv_zone_facility ON gl_eudr_sgv_storage_zones (facility_id);
CREATE INDEX idx_eudr_sgv_zone_type ON gl_eudr_sgv_storage_zones (storage_type);
CREATE INDEX idx_eudr_sgv_zone_compliance ON gl_eudr_sgv_storage_zones (compliance_status);


-- ============================================================================
-- 3. gl_eudr_sgv_storage_events — Storage material movement (hypertable)
-- ============================================================================
RAISE NOTICE 'V098 [3/12]: Creating gl_eudr_sgv_storage_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_storage_events (
    id                      BIGSERIAL,
    event_id                VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Event details
    zone_id                 VARCHAR(100)    NOT NULL,
    facility_id             VARCHAR(100)    NOT NULL,
    event_type              VARCHAR(50)     NOT NULL,
        -- inbound, outbound, transfer, adjustment, inspection

    -- Batch & quantity
    batch_id                VARCHAR(100),
    quantity_kg             DOUBLE PRECISION DEFAULT 0,
    operator_id             VARCHAR(200),

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
    'gl_eudr_sgv_storage_events',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_sgv_se_tenant ON gl_eudr_sgv_storage_events (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_se_zone ON gl_eudr_sgv_storage_events (zone_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_se_batch ON gl_eudr_sgv_storage_events (batch_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_se_type ON gl_eudr_sgv_storage_events (event_type, timestamp DESC);
CREATE INDEX idx_eudr_sgv_se_facility ON gl_eudr_sgv_storage_events (facility_id, timestamp DESC);


-- ============================================================================
-- 4. gl_eudr_sgv_transport_vehicles — Transport vehicle registry
-- ============================================================================
RAISE NOTICE 'V098 [4/12]: Creating gl_eudr_sgv_transport_vehicles...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_transport_vehicles (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_id              VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Vehicle details
    vehicle_type            VARCHAR(50),
    owner_operator_id       VARCHAR(100),

    -- Dedicated status & cleaning
    dedicated_status        BOOLEAN         DEFAULT FALSE,
    last_cargo_type         VARCHAR(30),
    last_cleaning_date      TIMESTAMPTZ,
    cleaning_method         VARCHAR(50),

    -- History
    cargo_history           JSONB           DEFAULT '[]',

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_sgv_tv_tenant ON gl_eudr_sgv_transport_vehicles (tenant_id);
CREATE INDEX idx_eudr_sgv_tv_type ON gl_eudr_sgv_transport_vehicles (vehicle_type);
CREATE INDEX idx_eudr_sgv_tv_dedicated ON gl_eudr_sgv_transport_vehicles (dedicated_status);
CREATE INDEX idx_eudr_sgv_tv_owner ON gl_eudr_sgv_transport_vehicles (owner_operator_id);


-- ============================================================================
-- 5. gl_eudr_sgv_transport_verifications — Transport segregation verifications
--    (hypertable)
-- ============================================================================
RAISE NOTICE 'V098 [5/12]: Creating gl_eudr_sgv_transport_verifications (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_transport_verifications (
    id                      BIGSERIAL,
    verification_id         VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Vehicle & batch
    vehicle_id              VARCHAR(100)    NOT NULL,
    batch_id                VARCHAR(100),

    -- Route
    route_origin            TEXT,
    route_destination       TEXT,

    -- Cleaning & seal
    cleaning_verified       BOOLEAN         DEFAULT FALSE,
    seal_number             VARCHAR(100),
    seal_intact             BOOLEAN,
    previous_cargoes        JSONB           DEFAULT '[]',

    -- Score
    score                   DOUBLE PRECISION DEFAULT 0,

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
    'gl_eudr_sgv_transport_verifications',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_sgv_tver_tenant ON gl_eudr_sgv_transport_verifications (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_tver_vehicle ON gl_eudr_sgv_transport_verifications (vehicle_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_tver_batch ON gl_eudr_sgv_transport_verifications (batch_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_tver_cleaning ON gl_eudr_sgv_transport_verifications (cleaning_verified, timestamp DESC);


-- ============================================================================
-- 6. gl_eudr_sgv_processing_lines — Processing line registry
-- ============================================================================
RAISE NOTICE 'V098 [6/12]: Creating gl_eudr_sgv_processing_lines...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_processing_lines (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    line_id                 VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Line details
    facility_id             VARCHAR(100)    NOT NULL,
    line_type               VARCHAR(50),
    commodity               VARCHAR(50),
    capacity_kg_per_hour    DOUBLE PRECISION DEFAULT 0,

    -- Dedicated status
    dedicated_status        BOOLEAN         DEFAULT FALSE,
    last_changeover_date    TIMESTAMPTZ,
    shared_equipment        JSONB           DEFAULT '[]',

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_sgv_pl_tenant ON gl_eudr_sgv_processing_lines (tenant_id);
CREATE INDEX idx_eudr_sgv_pl_facility ON gl_eudr_sgv_processing_lines (facility_id);
CREATE INDEX idx_eudr_sgv_pl_type ON gl_eudr_sgv_processing_lines (line_type);
CREATE INDEX idx_eudr_sgv_pl_commodity ON gl_eudr_sgv_processing_lines (commodity);
CREATE INDEX idx_eudr_sgv_pl_dedicated ON gl_eudr_sgv_processing_lines (dedicated_status);


-- ============================================================================
-- 7. gl_eudr_sgv_changeover_records — Line changeover/cleaning records
--    (hypertable)
-- ============================================================================
RAISE NOTICE 'V098 [7/12]: Creating gl_eudr_sgv_changeover_records (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_changeover_records (
    id                      BIGSERIAL,
    changeover_id           VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Line & facility
    line_id                 VARCHAR(100)    NOT NULL,
    facility_id             VARCHAR(100)    NOT NULL,

    -- Batch types
    previous_batch_type     VARCHAR(30),
    next_batch_type         VARCHAR(30),

    -- Flush & cleaning
    flush_volume_liters     DOUBLE PRECISION,
    flush_duration_minutes  INTEGER,
    cleaning_method         VARCHAR(50),
    purge_method            VARCHAR(50),

    -- Verification
    verified_by             VARCHAR(200),
    verification_notes      TEXT,

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
    'gl_eudr_sgv_changeover_records',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_sgv_co_tenant ON gl_eudr_sgv_changeover_records (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_co_line ON gl_eudr_sgv_changeover_records (line_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_co_facility ON gl_eudr_sgv_changeover_records (facility_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_co_cleaning ON gl_eudr_sgv_changeover_records (cleaning_method, timestamp DESC);


-- ============================================================================
-- 8. gl_eudr_sgv_contamination_events — Cross-contamination incidents
--    (hypertable)
-- ============================================================================
RAISE NOTICE 'V098 [8/12]: Creating gl_eudr_sgv_contamination_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_contamination_events (
    id                      BIGSERIAL,
    event_id                VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Facility & classification
    facility_id             VARCHAR(100)    NOT NULL,
    pathway_type            VARCHAR(50),
        -- storage_cross, transport_residue, processing_carryover,
        -- handling_mix, labeling_error
    severity                VARCHAR(20)     NOT NULL,
        -- low, medium, high, critical

    -- Affected batches
    affected_batch_ids      JSONB           DEFAULT '[]',
    affected_quantity_kg    DOUBLE PRECISION DEFAULT 0,

    -- Root cause & resolution
    root_cause              TEXT,
    corrective_action       TEXT,
    resolved                BOOLEAN         DEFAULT FALSE,
    resolved_date           TIMESTAMPTZ,

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
    'gl_eudr_sgv_contamination_events',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_sgv_ce_tenant ON gl_eudr_sgv_contamination_events (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_ce_facility ON gl_eudr_sgv_contamination_events (facility_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_ce_pathway ON gl_eudr_sgv_contamination_events (pathway_type, timestamp DESC);
CREATE INDEX idx_eudr_sgv_ce_severity ON gl_eudr_sgv_contamination_events (severity, timestamp DESC);
CREATE INDEX idx_eudr_sgv_ce_resolved ON gl_eudr_sgv_contamination_events (resolved, timestamp DESC);


-- ============================================================================
-- 9. gl_eudr_sgv_labels — Labeling and marking records
-- ============================================================================
RAISE NOTICE 'V098 [9/12]: Creating gl_eudr_sgv_labels...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_labels (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    label_id                VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- SCP reference
    scp_id                  VARCHAR(100),

    -- Label details
    label_type              VARCHAR(50),
        -- batch_label, zone_marker, vehicle_placard, container_tag,
        -- compliance_sticker, qr_code
    status                  VARCHAR(30),
        -- active, expired, damaged, replaced
    content_fields          JSONB           DEFAULT '{}',

    -- Dates
    placement_verified      BOOLEAN         DEFAULT FALSE,
    applied_date            TIMESTAMPTZ,
    verified_date           TIMESTAMPTZ,
    expiry_date             TIMESTAMPTZ,

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_sgv_label_tenant ON gl_eudr_sgv_labels (tenant_id);
CREATE INDEX idx_eudr_sgv_label_scp ON gl_eudr_sgv_labels (scp_id);
CREATE INDEX idx_eudr_sgv_label_type ON gl_eudr_sgv_labels (label_type);
CREATE INDEX idx_eudr_sgv_label_status ON gl_eudr_sgv_labels (status);


-- ============================================================================
-- 10. gl_eudr_sgv_facility_assessments — Facility assessment results
-- ============================================================================
RAISE NOTICE 'V098 [10/12]: Creating gl_eudr_sgv_facility_assessments...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_facility_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Facility
    facility_id             VARCHAR(100)    NOT NULL,

    -- Capability
    capability_level        VARCHAR(20),
        -- basic, intermediate, advanced, certified

    -- Dimension scores (0-100)
    layout_score            DOUBLE PRECISION DEFAULT 0,
    protocol_score          DOUBLE PRECISION DEFAULT 0,
    history_score           DOUBLE PRECISION DEFAULT 0,
    labeling_score          DOUBLE PRECISION DEFAULT 0,
    documentation_score     DOUBLE PRECISION DEFAULT 0,
    overall_score           DOUBLE PRECISION DEFAULT 0,

    -- Recommendations & certification
    recommendations         JSONB           DEFAULT '[]',
    assessment_date         TIMESTAMPTZ,
    certification_readiness JSONB           DEFAULT '{}',

    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_by              UUID,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_sgv_fa_tenant ON gl_eudr_sgv_facility_assessments (tenant_id);
CREATE INDEX idx_eudr_sgv_fa_facility ON gl_eudr_sgv_facility_assessments (facility_id);
CREATE INDEX idx_eudr_sgv_fa_capability ON gl_eudr_sgv_facility_assessments (capability_level);
CREATE INDEX idx_eudr_sgv_fa_score ON gl_eudr_sgv_facility_assessments (overall_score DESC);
CREATE INDEX idx_eudr_sgv_fa_date ON gl_eudr_sgv_facility_assessments (assessment_date DESC);


-- ============================================================================
-- 11. gl_eudr_sgv_batch_jobs — Batch processing jobs
-- ============================================================================
RAISE NOTICE 'V098 [11/12]: Creating gl_eudr_sgv_batch_jobs...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_batch_jobs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id                  VARCHAR(100)    NOT NULL UNIQUE,
    tenant_id               UUID            NOT NULL,

    -- Job config
    job_type                VARCHAR(50)     NOT NULL,
        -- scp_validation, storage_audit, transport_check, contamination_scan,
        -- facility_assessment, report_generation
    status                  VARCHAR(30)     DEFAULT 'pending',
        -- pending, running, completed, failed, cancelled

    -- Progress
    total_items             INTEGER         DEFAULT 0,
    processed_items         INTEGER         DEFAULT 0,
    failed_items            INTEGER         DEFAULT 0,

    -- Parameters & results
    parameters              JSONB           DEFAULT '{}',
    results                 JSONB           DEFAULT '{}',
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

CREATE INDEX idx_eudr_sgv_jobs_tenant ON gl_eudr_sgv_batch_jobs (tenant_id);
CREATE INDEX idx_eudr_sgv_jobs_type ON gl_eudr_sgv_batch_jobs (job_type);
CREATE INDEX idx_eudr_sgv_jobs_status ON gl_eudr_sgv_batch_jobs (status);
CREATE INDEX idx_eudr_sgv_jobs_submitted ON gl_eudr_sgv_batch_jobs (submitted_at DESC);


-- ============================================================================
-- 12. gl_eudr_sgv_audit_log — Immutable audit trail
-- ============================================================================
RAISE NOTICE 'V098 [12/12]: Creating gl_eudr_sgv_audit_log...';

CREATE TABLE IF NOT EXISTS gl_eudr_sgv_audit_log (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,

    -- Entity reference
    entity_type             VARCHAR(50)     NOT NULL,
        -- segregation_point, storage_zone, storage_event, transport_vehicle,
        -- transport_verification, processing_line, changeover, contamination,
        -- label, facility_assessment, batch_job
    entity_id               VARCHAR(100)    NOT NULL,

    -- Action
    action                  VARCHAR(50)     NOT NULL,
        -- created, updated, deleted, verified, flagged, resolved, assessed
    actor                   VARCHAR(200),

    -- Hash chain
    previous_hash           VARCHAR(64),
    current_hash            VARCHAR(64)     NOT NULL,

    -- Details
    details                 JSONB           DEFAULT '{}',

    -- Timing
    timestamp               TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eudr_sgv_audit_tenant ON gl_eudr_sgv_audit_log (tenant_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_audit_entity_type ON gl_eudr_sgv_audit_log (entity_type, timestamp DESC);
CREATE INDEX idx_eudr_sgv_audit_entity_id ON gl_eudr_sgv_audit_log (entity_id, timestamp DESC);
CREATE INDEX idx_eudr_sgv_audit_action ON gl_eudr_sgv_audit_log (action, timestamp DESC);
CREATE INDEX idx_eudr_sgv_audit_hash ON gl_eudr_sgv_audit_log (current_hash);


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================
RAISE NOTICE 'V098: Creating continuous aggregates...';

-- Hourly storage events summary
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_sgv_storage_events_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp)    AS bucket,
    tenant_id,
    zone_id,
    facility_id,
    event_type,
    COUNT(*)                            AS event_count,
    SUM(quantity_kg)                    AS total_quantity_kg,
    COUNT(DISTINCT batch_id)            AS unique_batches
FROM gl_eudr_sgv_storage_events
GROUP BY bucket, tenant_id, zone_id, facility_id, event_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_sgv_storage_events_hourly',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- Daily contamination events summary
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_sgv_contamination_events_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp)     AS bucket,
    tenant_id,
    facility_id,
    pathway_type,
    severity,
    COUNT(*)                            AS event_count,
    SUM(affected_quantity_kg)           AS total_affected_kg,
    COUNT(*) FILTER (WHERE resolved)    AS resolved_count,
    COUNT(*) FILTER (WHERE NOT resolved) AS unresolved_count
FROM gl_eudr_sgv_contamination_events
GROUP BY bucket, tenant_id, facility_id, pathway_type, severity
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_sgv_contamination_events_daily',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies
-- ============================================================================
RAISE NOTICE 'V098: Adding retention policies (5 years)...';

SELECT add_retention_policy('gl_eudr_sgv_storage_events',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_sgv_transport_verifications',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_sgv_changeover_records',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_sgv_contamination_events',
    INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================
RAISE NOTICE 'V098: Adding table comments...';

COMMENT ON TABLE gl_eudr_sgv_segregation_points IS 'AGENT-EUDR-010: Segregation control point master records with risk classification and compliance scoring';
COMMENT ON TABLE gl_eudr_sgv_storage_zones IS 'AGENT-EUDR-010: Storage zone definitions with barrier types, capacity tracking, and adjacency mapping';
COMMENT ON TABLE gl_eudr_sgv_storage_events IS 'AGENT-EUDR-010: Storage material movement events per zone with batch tracking (hypertable)';
COMMENT ON TABLE gl_eudr_sgv_transport_vehicles IS 'AGENT-EUDR-010: Transport vehicle registry with dedicated status, cleaning records, and cargo history';
COMMENT ON TABLE gl_eudr_sgv_transport_verifications IS 'AGENT-EUDR-010: Transport segregation verifications with seal checks and cleaning verification (hypertable)';
COMMENT ON TABLE gl_eudr_sgv_processing_lines IS 'AGENT-EUDR-010: Processing line registry with dedicated status and shared equipment tracking';
COMMENT ON TABLE gl_eudr_sgv_changeover_records IS 'AGENT-EUDR-010: Line changeover and cleaning records with flush volumes and purge methods (hypertable)';
COMMENT ON TABLE gl_eudr_sgv_contamination_events IS 'AGENT-EUDR-010: Cross-contamination incident records with root cause analysis and corrective actions (hypertable)';
COMMENT ON TABLE gl_eudr_sgv_labels IS 'AGENT-EUDR-010: Labeling and marking records for segregation points, zones, and vehicles';
COMMENT ON TABLE gl_eudr_sgv_facility_assessments IS 'AGENT-EUDR-010: Facility assessment results with dimension scores and certification readiness';
COMMENT ON TABLE gl_eudr_sgv_batch_jobs IS 'AGENT-EUDR-010: Batch processing jobs for SCP validation, audits, and report generation';
COMMENT ON TABLE gl_eudr_sgv_audit_log IS 'AGENT-EUDR-010: Immutable audit trail with hash chain for all segregation verifier data changes';

RAISE NOTICE 'V098: AGENT-EUDR-010 Segregation Verifier migration complete.';

COMMIT;
