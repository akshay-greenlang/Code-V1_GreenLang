-- ============================================================================
-- V103: AGENT-EUDR-015 Mobile Data Collector Agent
-- ============================================================================
-- Creates tables for offline-first mobile data collection, GPS/polygon capture,
-- photo evidence with SHA-256 integrity, CRDT-based synchronization, dynamic
-- form templates, ECDSA digital signatures, Merkle-rooted data packages,
-- device fleet management, telemetry events, and audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_mdc_forms, gl_eudr_mdc_gps_captures,
--              gl_eudr_mdc_device_events
-- Continuous Aggregates: 2 (hourly_form_stats + hourly_sync_stats)
-- Retention Policies: 3 (5 years for forms/GPS, 2 years for device events)
-- Indexes: ~100
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V103: Creating AGENT-EUDR-015 Mobile Data Collector tables...';

-- ============================================================================
-- 1. gl_eudr_mdc_forms — Form submissions (hypertable, monthly on submitted_at)
-- ============================================================================
RAISE NOTICE 'V103 [1/12]: Creating gl_eudr_mdc_forms (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_forms (
    id                      UUID            DEFAULT gen_random_uuid(),
    form_template_id        UUID,
    device_id               UUID,
    operator_id             VARCHAR(255)    NOT NULL,
    collector_name          VARCHAR(500)    NOT NULL,
    form_data               JSONB           NOT NULL,
    status                  VARCHAR(50)     NOT NULL DEFAULT 'draft',
        -- 'draft', 'pending', 'syncing', 'synced', 'failed'
    commodity_type          VARCHAR(100),
        -- 'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    submitted_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    synced_at               TIMESTAMPTZ,
    validation_status       VARCHAR(50)     NOT NULL DEFAULT 'pending',
        -- 'pending', 'valid', 'invalid', 'warnings'
    validation_errors       JSONB,
    offline_created_at      TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, submitted_at)
);

SELECT create_hypertable(
    'gl_eudr_mdc_forms',
    'submitted_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_template_id ON gl_eudr_mdc_forms (form_template_id, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_device_id ON gl_eudr_mdc_forms (device_id, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_operator_id ON gl_eudr_mdc_forms (operator_id, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_collector ON gl_eudr_mdc_forms (collector_name, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_status ON gl_eudr_mdc_forms (status, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_commodity ON gl_eudr_mdc_forms (commodity_type, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_synced_at ON gl_eudr_mdc_forms (synced_at, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_validation ON gl_eudr_mdc_forms (validation_status, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_offline ON gl_eudr_mdc_forms (offline_created_at, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_provenance ON gl_eudr_mdc_forms (provenance_hash, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_form_data ON gl_eudr_mdc_forms USING GIN (form_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_f_validation_errors ON gl_eudr_mdc_forms USING GIN (validation_errors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_mdc_form_templates — Form template definitions
-- ============================================================================
RAISE NOTICE 'V103 [2/12]: Creating gl_eudr_mdc_form_templates...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_form_templates (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    template_type           VARCHAR(100)    NOT NULL,
        -- 'producer_registration', 'plot_survey', 'harvest_log',
        -- 'custody_transfer', 'quality_inspection', 'smallholder_declaration'
    name                    VARCHAR(500)    NOT NULL,
    version                 VARCHAR(50)     NOT NULL DEFAULT '1.0.0',
    description             TEXT,
    fields                  JSONB           NOT NULL,
    validation_rules        JSONB,
    conditional_logic       JSONB,
    supported_languages     JSONB,
        -- array of ISO 639-1 language codes, e.g. ["en","fr","de","pt","sw"]
    is_active               BOOLEAN         NOT NULL DEFAULT TRUE,
    commodity_types         JSONB,
        -- array of commodity types this template applies to
    created_by              VARCHAR(255)    NOT NULL,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (template_type, version)
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_type ON gl_eudr_mdc_form_templates (template_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_name ON gl_eudr_mdc_form_templates (name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_version ON gl_eudr_mdc_form_templates (version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_active ON gl_eudr_mdc_form_templates (is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_created_by ON gl_eudr_mdc_form_templates (created_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_created ON gl_eudr_mdc_form_templates (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_type_active ON gl_eudr_mdc_form_templates (template_type, is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_fields ON gl_eudr_mdc_form_templates USING GIN (fields);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_validation ON gl_eudr_mdc_form_templates USING GIN (validation_rules);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_conditional ON gl_eudr_mdc_form_templates USING GIN (conditional_logic);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_languages ON gl_eudr_mdc_form_templates USING GIN (supported_languages);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ft_commodities ON gl_eudr_mdc_form_templates USING GIN (commodity_types);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_mdc_gps_captures — GPS captures (hypertable, monthly on captured_at)
-- ============================================================================
RAISE NOTICE 'V103 [3/12]: Creating gl_eudr_mdc_gps_captures (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_gps_captures (
    id                      UUID            DEFAULT gen_random_uuid(),
    form_id                 UUID,
    device_id               UUID,
    latitude                DOUBLE PRECISION NOT NULL,
    longitude               DOUBLE PRECISION NOT NULL,
    altitude                DOUBLE PRECISION,
    accuracy_meters         DOUBLE PRECISION,
    hdop                    DOUBLE PRECISION,
    satellite_count         INTEGER,
    capture_method          VARCHAR(50)     NOT NULL DEFAULT 'point',
        -- 'point', 'polygon_vertex', 'waypoint', 'reference'
    crs                     VARCHAR(50)     NOT NULL DEFAULT 'EPSG:4326',
    captured_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, captured_at)
);

SELECT create_hypertable(
    'gl_eudr_mdc_gps_captures',
    'captured_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_form_id ON gl_eudr_mdc_gps_captures (form_id, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_device_id ON gl_eudr_mdc_gps_captures (device_id, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_lat ON gl_eudr_mdc_gps_captures (latitude, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_lon ON gl_eudr_mdc_gps_captures (longitude, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_lat_lon ON gl_eudr_mdc_gps_captures (latitude, longitude, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_accuracy ON gl_eudr_mdc_gps_captures (accuracy_meters, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_hdop ON gl_eudr_mdc_gps_captures (hdop, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_satellites ON gl_eudr_mdc_gps_captures (satellite_count, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_method ON gl_eudr_mdc_gps_captures (capture_method, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_gc_provenance ON gl_eudr_mdc_gps_captures (provenance_hash, captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_mdc_polygon_traces — Plot boundary polygon traces
-- ============================================================================
RAISE NOTICE 'V103 [4/12]: Creating gl_eudr_mdc_polygon_traces...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_polygon_traces (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    form_id                 UUID,
    gps_capture_id          UUID,
    vertices                JSONB           NOT NULL,
        -- array of {lat, lon, alt, accuracy} objects
    vertex_count            INTEGER         NOT NULL,
    area_hectares           DOUBLE PRECISION,
    perimeter_meters        DOUBLE PRECISION,
    centroid_lat            DOUBLE PRECISION,
    centroid_lon            DOUBLE PRECISION,
    is_closed               BOOLEAN         NOT NULL DEFAULT FALSE,
    trace_duration_seconds  INTEGER,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_form_id ON gl_eudr_mdc_polygon_traces (form_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_gps_capture ON gl_eudr_mdc_polygon_traces (gps_capture_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_vertex_count ON gl_eudr_mdc_polygon_traces (vertex_count);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_area ON gl_eudr_mdc_polygon_traces (area_hectares);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_centroid ON gl_eudr_mdc_polygon_traces (centroid_lat, centroid_lon);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_closed ON gl_eudr_mdc_polygon_traces (is_closed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_created ON gl_eudr_mdc_polygon_traces (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_pt_vertices ON gl_eudr_mdc_polygon_traces USING GIN (vertices);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_mdc_photos — Photo evidence records
-- ============================================================================
RAISE NOTICE 'V103 [5/12]: Creating gl_eudr_mdc_photos...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_photos (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    form_id                 UUID,
    device_id               UUID,
    photo_type              VARCHAR(50)     NOT NULL,
        -- 'plot_photo', 'commodity_photo', 'document_photo',
        -- 'facility_photo', 'transport_photo', 'identity_photo'
    file_path               VARCHAR(1024)   NOT NULL,
    file_size_bytes         BIGINT          NOT NULL,
    mime_type               VARCHAR(100)    NOT NULL DEFAULT 'image/jpeg',
    width                   INTEGER,
    height                  INTEGER,
    sha256_hash             VARCHAR(64)     NOT NULL,
    exif_data               JSONB,
    gps_latitude            DOUBLE PRECISION,
    gps_longitude           DOUBLE PRECISION,
    captured_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    compression_ratio       DOUBLE PRECISION,
    original_filename       VARCHAR(500),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_form_id ON gl_eudr_mdc_photos (form_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_device_id ON gl_eudr_mdc_photos (device_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_type ON gl_eudr_mdc_photos (photo_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_sha256 ON gl_eudr_mdc_photos (sha256_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_gps ON gl_eudr_mdc_photos (gps_latitude, gps_longitude);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_captured ON gl_eudr_mdc_photos (captured_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_size ON gl_eudr_mdc_photos (file_size_bytes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_mime ON gl_eudr_mdc_photos (mime_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_created ON gl_eudr_mdc_photos (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_ph_exif ON gl_eudr_mdc_photos USING GIN (exif_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_mdc_sync_queue — Offline sync queue
-- ============================================================================
RAISE NOTICE 'V103 [6/12]: Creating gl_eudr_mdc_sync_queue...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_sync_queue (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id               UUID            NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
        -- 'form', 'gps_capture', 'photo', 'signature', 'data_package'
    entity_id               UUID            NOT NULL,
    action                  VARCHAR(50)     NOT NULL DEFAULT 'create',
        -- 'create', 'update', 'delete'
    payload                 JSONB           NOT NULL,
    priority                INTEGER         NOT NULL DEFAULT 5,
        -- 1=highest (forms), 2=high (GPS/signatures), 3=medium (photo metadata),
        -- 4=low (photo binary), 5=default
    retry_count             INTEGER         NOT NULL DEFAULT 0,
    max_retries             INTEGER         NOT NULL DEFAULT 20,
    last_error              TEXT,
    queued_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    synced_at               TIMESTAMPTZ,
    status                  VARCHAR(50)     NOT NULL DEFAULT 'pending',
        -- 'pending', 'syncing', 'synced', 'failed', 'permanently_failed'
    provenance_hash         VARCHAR(64)
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_device_id ON gl_eudr_mdc_sync_queue (device_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_entity_type ON gl_eudr_mdc_sync_queue (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_entity_id ON gl_eudr_mdc_sync_queue (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_action ON gl_eudr_mdc_sync_queue (action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_priority ON gl_eudr_mdc_sync_queue (priority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_status ON gl_eudr_mdc_sync_queue (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_queued ON gl_eudr_mdc_sync_queue (queued_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_synced ON gl_eudr_mdc_sync_queue (synced_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_retry ON gl_eudr_mdc_sync_queue (retry_count);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_status_priority ON gl_eudr_mdc_sync_queue (status, priority, queued_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_device_status ON gl_eudr_mdc_sync_queue (device_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sq_payload ON gl_eudr_mdc_sync_queue USING GIN (payload);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_mdc_sync_conflicts — Sync conflict records
-- ============================================================================
RAISE NOTICE 'V103 [7/12]: Creating gl_eudr_mdc_sync_conflicts...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_sync_conflicts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sync_queue_id           UUID,
    device_id               UUID            NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
        -- 'form', 'gps_capture', 'photo', 'signature'
    entity_id               UUID            NOT NULL,
    server_version          JSONB           NOT NULL,
    client_version          JSONB           NOT NULL,
    resolution_strategy     VARCHAR(50),
        -- 'last_writer_wins', 'server_wins', 'client_wins', 'set_union',
        -- 'manual', 'state_machine'
    resolved_version        JSONB,
    resolved_at             TIMESTAMPTZ,
    resolved_by             VARCHAR(255),
    status                  VARCHAR(50)     NOT NULL DEFAULT 'unresolved',
        -- 'unresolved', 'auto_resolved', 'manually_resolved', 'escalated'
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_queue_id ON gl_eudr_mdc_sync_conflicts (sync_queue_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_device_id ON gl_eudr_mdc_sync_conflicts (device_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_entity_type ON gl_eudr_mdc_sync_conflicts (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_entity_id ON gl_eudr_mdc_sync_conflicts (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_strategy ON gl_eudr_mdc_sync_conflicts (resolution_strategy);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_status ON gl_eudr_mdc_sync_conflicts (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_resolved_at ON gl_eudr_mdc_sync_conflicts (resolved_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_created ON gl_eudr_mdc_sync_conflicts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_server_ver ON gl_eudr_mdc_sync_conflicts USING GIN (server_version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sc_client_ver ON gl_eudr_mdc_sync_conflicts USING GIN (client_version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_mdc_signatures — Digital signature records (ECDSA P-256)
-- ============================================================================
RAISE NOTICE 'V103 [8/12]: Creating gl_eudr_mdc_signatures...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_signatures (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    form_id                 UUID            NOT NULL,
    signer_name             VARCHAR(500)    NOT NULL,
    signer_role             VARCHAR(100)    NOT NULL,
        -- 'producer', 'collector', 'inspector', 'transport_operator', 'buyer'
    algorithm               VARCHAR(50)     NOT NULL DEFAULT 'ECDSA-P256',
    public_key              TEXT            NOT NULL,
    signature_data          TEXT            NOT NULL,
        -- DER-encoded ECDSA signature bytes (base64)
    signed_data_hash        VARCHAR(64)     NOT NULL,
        -- SHA-256 hash of the signed payload (form data + timestamp + signer)
    signed_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at              TIMESTAMPTZ,
    is_verified             BOOLEAN         NOT NULL DEFAULT FALSE,
    verification_timestamp  TIMESTAMPTZ,
    device_id               UUID,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_form_id ON gl_eudr_mdc_signatures (form_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_signer_name ON gl_eudr_mdc_signatures (signer_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_signer_role ON gl_eudr_mdc_signatures (signer_role);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_algorithm ON gl_eudr_mdc_signatures (algorithm);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_hash ON gl_eudr_mdc_signatures (signed_data_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_signed_at ON gl_eudr_mdc_signatures (signed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_expires ON gl_eudr_mdc_signatures (expires_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_verified ON gl_eudr_mdc_signatures (is_verified);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_device_id ON gl_eudr_mdc_signatures (device_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_sig_created ON gl_eudr_mdc_signatures (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_mdc_data_packages — Data package assembly with Merkle root
-- ============================================================================
RAISE NOTICE 'V103 [9/12]: Creating gl_eudr_mdc_data_packages...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_data_packages (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id             VARCHAR(255)    NOT NULL,
    device_id               UUID,
    form_ids                JSONB,
        -- array of form UUIDs included in this package
    gps_capture_ids         JSONB,
        -- array of GPS capture UUIDs
    photo_ids               JSONB,
        -- array of photo UUIDs
    signature_ids           JSONB,
        -- array of signature UUIDs
    package_hash            VARCHAR(64),
        -- SHA-256 hash of the complete package
    merkle_root             VARCHAR(64),
        -- SHA-256 Merkle root over all artifact hashes
    package_size_bytes      BIGINT,
    compression_format      VARCHAR(50),
        -- 'gzip', 'zip', 'tar_gz', 'none'
    status                  VARCHAR(50)     NOT NULL DEFAULT 'building',
        -- 'building', 'sealed', 'uploading', 'uploaded', 'verified', 'failed'
    built_at                TIMESTAMPTZ,
    uploaded_at             TIMESTAMPTZ,
    retention_until         TIMESTAMPTZ,
        -- EUDR Art. 14: 5-year retention
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_operator_id ON gl_eudr_mdc_data_packages (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_device_id ON gl_eudr_mdc_data_packages (device_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_package_hash ON gl_eudr_mdc_data_packages (package_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_merkle_root ON gl_eudr_mdc_data_packages (merkle_root);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_status ON gl_eudr_mdc_data_packages (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_built_at ON gl_eudr_mdc_data_packages (built_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_uploaded_at ON gl_eudr_mdc_data_packages (uploaded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_retention ON gl_eudr_mdc_data_packages (retention_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_created ON gl_eudr_mdc_data_packages (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_form_ids ON gl_eudr_mdc_data_packages USING GIN (form_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_gps_ids ON gl_eudr_mdc_data_packages USING GIN (gps_capture_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_photo_ids ON gl_eudr_mdc_data_packages USING GIN (photo_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dp_sig_ids ON gl_eudr_mdc_data_packages USING GIN (signature_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 10. gl_eudr_mdc_devices — Registered mobile devices
-- ============================================================================
RAISE NOTICE 'V103 [10/12]: Creating gl_eudr_mdc_devices...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_devices (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    device_name             VARCHAR(500)    NOT NULL,
    platform                VARCHAR(50)     NOT NULL,
        -- 'android', 'ios', 'windows', 'linux'
    os_version              VARCHAR(100),
    app_version             VARCHAR(50),
    operator_id             VARCHAR(255)    NOT NULL,
    assigned_to             VARCHAR(500),
    storage_total_bytes     BIGINT,
    storage_used_bytes      BIGINT,
    battery_level           INTEGER,
        -- 0-100 percentage
    last_seen_at            TIMESTAMPTZ,
    last_sync_at            TIMESTAMPTZ,
    status                  VARCHAR(50)     NOT NULL DEFAULT 'registered',
        -- 'registered', 'active', 'offline', 'low_battery', 'low_storage',
        -- 'decommissioned'
    registration_token      VARCHAR(500),
    geo_fence               JSONB,
        -- GeoJSON polygon defining the assigned collection area
    provenance_hash         VARCHAR(64),
    registered_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_name ON gl_eudr_mdc_devices (device_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_platform ON gl_eudr_mdc_devices (platform);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_operator ON gl_eudr_mdc_devices (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_assigned ON gl_eudr_mdc_devices (assigned_to);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_status ON gl_eudr_mdc_devices (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_last_seen ON gl_eudr_mdc_devices (last_seen_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_last_sync ON gl_eudr_mdc_devices (last_sync_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_battery ON gl_eudr_mdc_devices (battery_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_app_version ON gl_eudr_mdc_devices (app_version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_registered ON gl_eudr_mdc_devices (registered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_dv_geo_fence ON gl_eudr_mdc_devices USING GIN (geo_fence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 11. gl_eudr_mdc_device_events — Device telemetry (hypertable, monthly)
-- ============================================================================
RAISE NOTICE 'V103 [11/12]: Creating gl_eudr_mdc_device_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_device_events (
    id                      UUID            DEFAULT gen_random_uuid(),
    device_id               UUID            NOT NULL,
    event_type              VARCHAR(100)    NOT NULL,
        -- 'heartbeat', 'sync_start', 'sync_complete', 'sync_error',
        -- 'low_battery', 'low_storage', 'gps_fix_lost', 'gps_fix_acquired',
        -- 'form_submitted', 'photo_captured', 'app_updated'
    event_data              JSONB,
    battery_level           INTEGER,
    storage_used_bytes      BIGINT,
    connectivity_type       VARCHAR(20),
        -- 'none', '2g', '3g', '4g', '5g', 'wifi'
    gps_latitude            DOUBLE PRECISION,
    gps_longitude           DOUBLE PRECISION,
    event_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64),

    PRIMARY KEY (id, event_at)
);

SELECT create_hypertable(
    'gl_eudr_mdc_device_events',
    'event_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_de_device_id ON gl_eudr_mdc_device_events (device_id, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_de_event_type ON gl_eudr_mdc_device_events (event_type, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_de_battery ON gl_eudr_mdc_device_events (battery_level, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_de_storage ON gl_eudr_mdc_device_events (storage_used_bytes, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_de_connectivity ON gl_eudr_mdc_device_events (connectivity_type, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_de_gps ON gl_eudr_mdc_device_events (gps_latitude, gps_longitude, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_de_event_data ON gl_eudr_mdc_device_events USING GIN (event_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 12. gl_eudr_mdc_audit_log — Immutable audit trail
-- ============================================================================
RAISE NOTICE 'V103 [12/12]: Creating gl_eudr_mdc_audit_log...';

CREATE TABLE IF NOT EXISTS gl_eudr_mdc_audit_log (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type             VARCHAR(100)    NOT NULL,
        -- 'form', 'form_template', 'gps_capture', 'polygon_trace', 'photo',
        -- 'sync_queue', 'sync_conflict', 'signature', 'data_package', 'device'
    entity_id               UUID            NOT NULL,
    action                  VARCHAR(100)    NOT NULL,
        -- 'created', 'updated', 'validated', 'synced', 'sync_failed',
        -- 'conflict_detected', 'conflict_resolved', 'signed', 'verified',
        -- 'revoked', 'sealed', 'uploaded', 'registered', 'decommissioned'
    actor_id                VARCHAR(255)    NOT NULL,
    device_id               UUID,
    details                 JSONB,
    ip_address              INET,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_entity_type ON gl_eudr_mdc_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_entity_id ON gl_eudr_mdc_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_action ON gl_eudr_mdc_audit_log (action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_actor ON gl_eudr_mdc_audit_log (actor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_device ON gl_eudr_mdc_audit_log (device_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_created ON gl_eudr_mdc_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_entity_action ON gl_eudr_mdc_audit_log (entity_type, action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_entity_created ON gl_eudr_mdc_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mdc_al_details ON gl_eudr_mdc_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================
RAISE NOTICE 'V103: Creating continuous aggregates...';

-- 1. Hourly form submission statistics by status, commodity_type, and validation
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_mdc_hourly_form_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', submitted_at)         AS bucket,
    status,
    commodity_type,
    validation_status,
    COUNT(*)                                    AS form_count,
    COUNT(*) FILTER (WHERE status = 'synced')           AS synced_count,
    COUNT(*) FILTER (WHERE status = 'pending')          AS pending_count,
    COUNT(*) FILTER (WHERE status = 'failed')           AS failed_count,
    COUNT(*) FILTER (WHERE status = 'draft')            AS draft_count,
    COUNT(*) FILTER (WHERE validation_status = 'valid') AS valid_count,
    COUNT(*) FILTER (WHERE validation_status = 'invalid') AS invalid_count,
    COUNT(*) FILTER (WHERE offline_created_at IS NOT NULL) AS offline_count
FROM gl_eudr_mdc_forms
GROUP BY bucket, status, commodity_type, validation_status
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_mdc_hourly_form_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- 2. Hourly sync statistics by device_id, entity_type, and status
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_mdc_hourly_sync_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_at)             AS bucket,
    device_id,
    event_type,
    connectivity_type,
    COUNT(*)                                    AS event_count,
    COUNT(*) FILTER (WHERE event_type = 'sync_complete')    AS sync_success_count,
    COUNT(*) FILTER (WHERE event_type = 'sync_error')       AS sync_error_count,
    COUNT(*) FILTER (WHERE event_type = 'heartbeat')        AS heartbeat_count,
    COUNT(*) FILTER (WHERE event_type = 'low_battery')      AS low_battery_count,
    COUNT(*) FILTER (WHERE event_type = 'low_storage')      AS low_storage_count,
    AVG(battery_level)                          AS avg_battery_level,
    MIN(battery_level)                          AS min_battery_level,
    AVG(storage_used_bytes)                     AS avg_storage_used
FROM gl_eudr_mdc_device_events
GROUP BY bucket, device_id, event_type, connectivity_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_mdc_hourly_sync_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies
-- ============================================================================
RAISE NOTICE 'V103: Adding retention policies...';

-- EUDR Article 14: 5-year retention for forms and GPS captures
SELECT add_retention_policy('gl_eudr_mdc_forms',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_mdc_gps_captures',
    INTERVAL '5 years', if_not_exists => TRUE);

-- Device events: 2-year retention for telemetry data
SELECT add_retention_policy('gl_eudr_mdc_device_events',
    INTERVAL '2 years', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================
RAISE NOTICE 'V103: Adding table comments...';

COMMENT ON TABLE gl_eudr_mdc_forms IS 'AGENT-EUDR-015: Collected form submissions with offline-first storage, status tracking (draft/pending/syncing/synced/failed), commodity classification, validation results, and provenance hashing (hypertable, monthly on submitted_at)';
COMMENT ON TABLE gl_eudr_mdc_form_templates IS 'AGENT-EUDR-015: Dynamic form template definitions with EUDR-specific types (producer_registration/plot_survey/harvest_log/custody_transfer/quality_inspection/smallholder_declaration), conditional logic, validation rules, multi-language support (44 languages), and semantic versioning';
COMMENT ON TABLE gl_eudr_mdc_gps_captures IS 'AGENT-EUDR-015: GPS coordinate captures with accuracy metadata (HDOP, satellite count, capture method), WGS84 datum, point and polygon vertex support per EUDR Article 9(1)(d) (hypertable, monthly on captured_at)';
COMMENT ON TABLE gl_eudr_mdc_polygon_traces IS 'AGENT-EUDR-015: Plot boundary polygon traces with vertex arrays, Shoelace/geodesic area calculation in hectares, perimeter measurement, centroid computation, closure validation, and trace duration tracking';
COMMENT ON TABLE gl_eudr_mdc_photos IS 'AGENT-EUDR-015: Geotagged photo evidence records with SHA-256 integrity hashing at capture time, EXIF metadata extraction, 6 photo categories (plot/commodity/document/facility/transport/identity), GPS coordinates, and compression ratio tracking';
COMMENT ON TABLE gl_eudr_mdc_sync_queue IS 'AGENT-EUDR-015: Offline sync queue with priority-ordered upload management (forms highest, photos lowest), CRDT-based merge strategies, exponential backoff retry logic (max 20), idempotency keys, and bandwidth-optimized delta compression';
COMMENT ON TABLE gl_eudr_mdc_sync_conflicts IS 'AGENT-EUDR-015: Sync conflict records preserving both server and client versions with resolution strategies (LWW/set_union/state_machine/manual), resolved version tracking, and escalation support';
COMMENT ON TABLE gl_eudr_mdc_signatures IS 'AGENT-EUDR-015: Digital signature records using ECDSA P-256 with deterministic k-value (RFC 6979), timestamp binding, form submission binding, DER-encoded signature bytes, public key storage, and verification status tracking';
COMMENT ON TABLE gl_eudr_mdc_data_packages IS 'AGENT-EUDR-015: Self-contained data packages with SHA-256 Merkle root integrity, manifest, provenance chain, artifact bundling (forms/GPS/photos/signatures), compression, 5-year retention per EUDR Article 14';
COMMENT ON TABLE gl_eudr_mdc_devices IS 'AGENT-EUDR-015: Registered mobile device fleet with operator assignment, platform/version tracking, storage/battery telemetry, sync status, GeoJSON geo-fencing, and lifecycle management (registered/active/offline/decommissioned)';
COMMENT ON TABLE gl_eudr_mdc_device_events IS 'AGENT-EUDR-015: Device telemetry events including heartbeats, sync events, battery/storage alerts, GPS fix status, connectivity type, and location tracking (hypertable, monthly on event_at)';
COMMENT ON TABLE gl_eudr_mdc_audit_log IS 'AGENT-EUDR-015: Immutable audit trail for all Mobile Data Collector operations with entity tracking, action logging, actor identification, device attribution, IP address recording, and provenance hashing';

RAISE NOTICE 'V103: AGENT-EUDR-015 Mobile Data Collector migration complete.';

COMMIT;
