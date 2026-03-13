-- ============================================================================
-- V102: AGENT-EUDR-014 QR Code Generator Agent
-- ============================================================================
-- Creates tables for QR code generation, payload composition, label rendering,
-- batch code management, verification URL generation, HMAC signatures,
-- scan event tracking, bulk job processing, lifecycle management,
-- label templates, code associations, and audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_qrg_codes, gl_eudr_qrg_scan_events,
--              gl_eudr_qrg_lifecycle_events
-- Continuous Aggregates: 2 (hourly_code_stats + hourly_scan_stats)
-- Retention Policies: 3 (hypertables, 90 days operational window)
-- Indexes: 102
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V102: Creating AGENT-EUDR-014 QR Code Generator tables...';

-- ============================================================================
-- 1. gl_eudr_qrg_codes — QR code records (hypertable)
-- ============================================================================
RAISE NOTICE 'V102 [1/12]: Creating gl_eudr_qrg_codes (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_codes (
    id                      UUID            DEFAULT gen_random_uuid(),
    code_id                 TEXT            UNIQUE NOT NULL,
    symbology_type          TEXT            NOT NULL DEFAULT 'qr_code',
        -- 'qr_code', 'data_matrix', 'aztec', 'micro_qr', 'gs1_datamatrix'
    version                 TEXT,
        -- QR version (1-40) or symbology-specific version
    error_correction        TEXT            NOT NULL DEFAULT 'M',
        -- 'L' (7%), 'M' (15%), 'Q' (25%), 'H' (30%)
    content_type            TEXT            NOT NULL,
        -- 'dds_reference', 'product_passport', 'shipment_tracking',
        -- 'compliance_certificate', 'geolocation_proof', 'batch_manifest'
    payload_id              UUID,
    payload_data            TEXT            NOT NULL,
    encoded_data            BYTEA,
    image_format            TEXT            NOT NULL DEFAULT 'png',
        -- 'png', 'svg', 'pdf', 'eps'
    image_data              BYTEA,
    module_size             INTEGER         DEFAULT 10,
    dpi                     INTEGER         DEFAULT 300,
    has_logo                BOOLEAN         DEFAULT FALSE,
    status                  TEXT            NOT NULL DEFAULT 'created',
        -- 'created', 'active', 'suspended', 'revoked', 'expired'
    operator_id             UUID            NOT NULL,
    dds_reference           TEXT,
    batch_code              TEXT,
    blockchain_anchor_hash  TEXT,
    hmac_signature          TEXT,
    verification_url        TEXT,
    expires_at              TIMESTAMPTZ,
    activated_at            TIMESTAMPTZ,
    revoked_at              TIMESTAMPTZ,
    provenance_hash         TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_qrg_codes',
    'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_code_id ON gl_eudr_qrg_codes (code_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_symbology ON gl_eudr_qrg_codes (symbology_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_content_type ON gl_eudr_qrg_codes (content_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_payload_id ON gl_eudr_qrg_codes (payload_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_image_format ON gl_eudr_qrg_codes (image_format, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_status ON gl_eudr_qrg_codes (status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_operator ON gl_eudr_qrg_codes (operator_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_dds_ref ON gl_eudr_qrg_codes (dds_reference, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_batch_code ON gl_eudr_qrg_codes (batch_code, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_blockchain ON gl_eudr_qrg_codes (blockchain_anchor_hash, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_expires ON gl_eudr_qrg_codes (expires_at, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_c_activated ON gl_eudr_qrg_codes (activated_at, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_qrg_payloads — Payload composition records
-- ============================================================================
RAISE NOTICE 'V102 [2/12]: Creating gl_eudr_qrg_payloads...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_payloads (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    payload_id              TEXT            UNIQUE NOT NULL,
    content_type            TEXT            NOT NULL,
        -- 'dds_reference', 'product_passport', 'shipment_tracking',
        -- 'compliance_certificate', 'geolocation_proof', 'batch_manifest'
    schema_version          TEXT            NOT NULL DEFAULT '1.0',
    raw_data                JSONB           NOT NULL,
    composed_payload        TEXT            NOT NULL,
    payload_size_bytes      INTEGER         NOT NULL,
    is_compressed           BOOLEAN         DEFAULT FALSE,
    is_encrypted            BOOLEAN         DEFAULT FALSE,
    dds_reference           TEXT,
    batch_id                TEXT,
    commodity_type          TEXT,
        -- 'palm_oil', 'soy', 'wood', 'cocoa', 'coffee', 'rubber', 'cattle'
    country_of_origin       TEXT,
    compliance_status       TEXT,
        -- 'compliant', 'non_compliant', 'pending', 'under_review'
    gs1_uri                 TEXT,
    validation_errors       JSONB,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_payload_id ON gl_eudr_qrg_payloads (payload_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_content_type ON gl_eudr_qrg_payloads (content_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_dds_ref ON gl_eudr_qrg_payloads (dds_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_batch_id ON gl_eudr_qrg_payloads (batch_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_commodity ON gl_eudr_qrg_payloads (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_country ON gl_eudr_qrg_payloads (country_of_origin);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_compliance ON gl_eudr_qrg_payloads (compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_gs1_uri ON gl_eudr_qrg_payloads (gs1_uri);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_created ON gl_eudr_qrg_payloads (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_p_raw_data ON gl_eudr_qrg_payloads USING GIN (raw_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_qrg_labels — Label rendering records
-- ============================================================================
RAISE NOTICE 'V102 [3/12]: Creating gl_eudr_qrg_labels...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_labels (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    label_id                TEXT            UNIQUE NOT NULL,
    template_name           TEXT            NOT NULL,
    code_id                 UUID,
    width_mm                DOUBLE PRECISION NOT NULL,
    height_mm               DOUBLE PRECISION NOT NULL,
    output_format           TEXT            NOT NULL,
        -- 'png', 'svg', 'pdf', 'eps', 'zpl'
    label_data              BYTEA,
    elements                JSONB,
        -- layout elements: text fields, barcodes, logos, borders
    font_family             TEXT,
    font_size               INTEGER,
    color_scheme            TEXT,
        -- 'default', 'high_contrast', 'monochrome', 'eudr_branded'
    operator_id             UUID            NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_l_label_id ON gl_eudr_qrg_labels (label_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_l_template ON gl_eudr_qrg_labels (template_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_l_code_id ON gl_eudr_qrg_labels (code_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_l_output_format ON gl_eudr_qrg_labels (output_format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_l_operator ON gl_eudr_qrg_labels (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_l_created ON gl_eudr_qrg_labels (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_l_elements ON gl_eudr_qrg_labels USING GIN (elements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_qrg_batch_codes — Batch code management
-- ============================================================================
RAISE NOTICE 'V102 [4/12]: Creating gl_eudr_qrg_batch_codes...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_batch_codes (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    code                    TEXT            UNIQUE NOT NULL,
    code_type               TEXT            NOT NULL,
        -- 'serial', 'batch', 'lot', 'sscc', 'gtin', 'custom'
    parent_code             TEXT,
    batch_level             INTEGER         NOT NULL DEFAULT 0,
        -- 0 = individual, 1 = batch, 2 = pallet, 3 = container
    prefix                  TEXT,
    sequence_number         BIGINT          NOT NULL,
    check_digit             TEXT,
    operator_id             UUID            NOT NULL,
    commodity_type          TEXT,
    year                    INTEGER,
    dds_reference           TEXT,
    blockchain_anchor_id    TEXT,
    is_reserved             BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_code ON gl_eudr_qrg_batch_codes (code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_code_type ON gl_eudr_qrg_batch_codes (code_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_parent ON gl_eudr_qrg_batch_codes (parent_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_level ON gl_eudr_qrg_batch_codes (batch_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_operator ON gl_eudr_qrg_batch_codes (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_commodity ON gl_eudr_qrg_batch_codes (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_dds_ref ON gl_eudr_qrg_batch_codes (dds_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_blockchain ON gl_eudr_qrg_batch_codes (blockchain_anchor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_reserved ON gl_eudr_qrg_batch_codes (is_reserved);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bc_created ON gl_eudr_qrg_batch_codes (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_qrg_verification_urls — Verification URL generation
-- ============================================================================
RAISE NOTICE 'V102 [5/12]: Creating gl_eudr_qrg_verification_urls...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_verification_urls (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    url_id                  TEXT            UNIQUE NOT NULL,
    code_id                 UUID,
    full_url                TEXT            NOT NULL,
    short_url               TEXT,
    hmac_signature          TEXT            NOT NULL,
    verification_token      TEXT,
    token_expires_at        TIMESTAMPTZ,
    operator_code           TEXT            NOT NULL,
    deep_link_url           TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_url_id ON gl_eudr_qrg_verification_urls (url_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_code_id ON gl_eudr_qrg_verification_urls (code_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_full_url ON gl_eudr_qrg_verification_urls (full_url);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_short_url ON gl_eudr_qrg_verification_urls (short_url);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_token ON gl_eudr_qrg_verification_urls (verification_token);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_operator ON gl_eudr_qrg_verification_urls (operator_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_token_expires ON gl_eudr_qrg_verification_urls (token_expires_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_vu_created ON gl_eudr_qrg_verification_urls (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_qrg_signatures — HMAC signature records
-- ============================================================================
RAISE NOTICE 'V102 [6/12]: Creating gl_eudr_qrg_signatures...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_signatures (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    signature_id            TEXT            UNIQUE NOT NULL,
    code_id                 UUID,
    algorithm               TEXT            NOT NULL DEFAULT 'hmac-sha256',
        -- 'hmac-sha256', 'hmac-sha512', 'ed25519', 'ecdsa-p256'
    key_id                  TEXT            NOT NULL,
    signature_value         TEXT            NOT NULL,
    is_valid                BOOLEAN         DEFAULT TRUE,
    verified_at             TIMESTAMPTZ,
    key_rotation_date       TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_sig_signature_id ON gl_eudr_qrg_signatures (signature_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_sig_code_id ON gl_eudr_qrg_signatures (code_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_sig_algorithm ON gl_eudr_qrg_signatures (algorithm);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_sig_key_id ON gl_eudr_qrg_signatures (key_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_sig_is_valid ON gl_eudr_qrg_signatures (is_valid);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_sig_key_rotation ON gl_eudr_qrg_signatures (key_rotation_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_sig_created ON gl_eudr_qrg_signatures (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_qrg_scan_events — Scan event tracking (hypertable)
-- ============================================================================
RAISE NOTICE 'V102 [7/12]: Creating gl_eudr_qrg_scan_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_scan_events (
    id                      UUID            DEFAULT gen_random_uuid(),
    event_id                TEXT            UNIQUE NOT NULL,
    code_id                 UUID,
    scanner_id              TEXT,
    scan_outcome            TEXT            NOT NULL,
        -- 'valid', 'invalid', 'expired', 'revoked', 'counterfeit_suspected',
        -- 'unknown_code', 'signature_mismatch', 'tampered'
    latitude                DOUBLE PRECISION,
    longitude               DOUBLE PRECISION,
    country_code            TEXT,
    city                    TEXT,
    user_agent              TEXT,
    ip_address              INET,
    verification_result     JSONB,
    counterfeit_risk_score  DOUBLE PRECISION,
        -- 0.0 (no risk) to 1.0 (high risk)
    scanned_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, scanned_at)
);

SELECT create_hypertable(
    'gl_eudr_qrg_scan_events',
    'scanned_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_event_id ON gl_eudr_qrg_scan_events (event_id, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_code_id ON gl_eudr_qrg_scan_events (code_id, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_scanner ON gl_eudr_qrg_scan_events (scanner_id, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_outcome ON gl_eudr_qrg_scan_events (scan_outcome, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_country ON gl_eudr_qrg_scan_events (country_code, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_city ON gl_eudr_qrg_scan_events (city, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_ip ON gl_eudr_qrg_scan_events (ip_address, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_risk ON gl_eudr_qrg_scan_events (counterfeit_risk_score, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_geo ON gl_eudr_qrg_scan_events (latitude, longitude, scanned_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_se_verification ON gl_eudr_qrg_scan_events USING GIN (verification_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_qrg_bulk_jobs — Bulk job processing
-- ============================================================================
RAISE NOTICE 'V102 [8/12]: Creating gl_eudr_qrg_bulk_jobs...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_bulk_jobs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id                  TEXT            UNIQUE NOT NULL,
    job_type                TEXT            NOT NULL,
        -- 'generate', 'print', 'export', 'revoke', 'reactivate', 'verify_batch'
    status                  TEXT            NOT NULL DEFAULT 'queued',
        -- 'queued', 'processing', 'completed', 'failed', 'cancelled'
    total_codes             INTEGER         NOT NULL DEFAULT 0,
    completed_codes         INTEGER         DEFAULT 0,
    failed_codes            INTEGER         DEFAULT 0,
    output_format           TEXT,
        -- 'png_archive', 'pdf_sheet', 'csv_manifest', 'zpl_stream'
    output_path             TEXT,
    manifest_path           TEXT,
    template_name           TEXT,
    operator_id             UUID            NOT NULL,
    error_details           JSONB,
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_job_id ON gl_eudr_qrg_bulk_jobs (job_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_job_type ON gl_eudr_qrg_bulk_jobs (job_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_status ON gl_eudr_qrg_bulk_jobs (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_operator ON gl_eudr_qrg_bulk_jobs (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_template ON gl_eudr_qrg_bulk_jobs (template_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_created ON gl_eudr_qrg_bulk_jobs (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_started ON gl_eudr_qrg_bulk_jobs (started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_bj_errors ON gl_eudr_qrg_bulk_jobs USING GIN (error_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_qrg_lifecycle_events — Lifecycle state transitions (hypertable)
-- ============================================================================
RAISE NOTICE 'V102 [9/12]: Creating gl_eudr_qrg_lifecycle_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_lifecycle_events (
    id                      UUID            DEFAULT gen_random_uuid(),
    event_id                TEXT            UNIQUE NOT NULL,
    code_id                 UUID,
    previous_status         TEXT,
        -- NULL for initial creation
    new_status              TEXT            NOT NULL,
        -- 'created', 'active', 'suspended', 'revoked', 'expired'
    action                  TEXT            NOT NULL,
        -- 'create', 'activate', 'suspend', 'revoke', 'expire',
        -- 'reactivate', 'extend', 'replace'
    actor_id                UUID,
    reason                  TEXT,
    metadata                JSONB,
    event_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, event_at)
);

SELECT create_hypertable(
    'gl_eudr_qrg_lifecycle_events',
    'event_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_le_event_id ON gl_eudr_qrg_lifecycle_events (event_id, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_le_code_id ON gl_eudr_qrg_lifecycle_events (code_id, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_le_prev_status ON gl_eudr_qrg_lifecycle_events (previous_status, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_le_new_status ON gl_eudr_qrg_lifecycle_events (new_status, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_le_action ON gl_eudr_qrg_lifecycle_events (action, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_le_actor ON gl_eudr_qrg_lifecycle_events (actor_id, event_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_le_metadata ON gl_eudr_qrg_lifecycle_events USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 10. gl_eudr_qrg_templates — Label template definitions
-- ============================================================================
RAISE NOTICE 'V102 [10/12]: Creating gl_eudr_qrg_templates...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_templates (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id             TEXT            UNIQUE NOT NULL,
    template_name           TEXT            NOT NULL,
    width_mm                DOUBLE PRECISION NOT NULL,
    height_mm               DOUBLE PRECISION NOT NULL,
    qr_size_mm              DOUBLE PRECISION NOT NULL,
    layout_elements         JSONB           NOT NULL,
        -- array of positioned elements: qr_code, text, logo, barcode, line, border
    supported_formats       JSONB,
        -- ['png', 'svg', 'pdf', 'eps', 'zpl']
    is_default              BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_t_template_id ON gl_eudr_qrg_templates (template_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_t_template_name ON gl_eudr_qrg_templates (template_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_t_is_default ON gl_eudr_qrg_templates (is_default);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_t_created ON gl_eudr_qrg_templates (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_t_layout ON gl_eudr_qrg_templates USING GIN (layout_elements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_t_formats ON gl_eudr_qrg_templates USING GIN (supported_formats);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 11. gl_eudr_qrg_code_associations — Code-to-entity associations
-- ============================================================================
RAISE NOTICE 'V102 [11/12]: Creating gl_eudr_qrg_code_associations...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_code_associations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    association_id          TEXT            UNIQUE NOT NULL,
    code_id                 UUID,
    association_type        TEXT            NOT NULL,
        -- 'parent_child', 'replacement', 'linked', 'aggregated',
        -- 'dds_reference', 'shipment', 'product_batch'
    target_id               TEXT            NOT NULL,
    target_type             TEXT            NOT NULL,
        -- 'qr_code', 'dds_statement', 'shipment', 'product',
        -- 'batch', 'operator', 'geolocation_plot'
    metadata                JSONB,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_association_id ON gl_eudr_qrg_code_associations (association_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_code_id ON gl_eudr_qrg_code_associations (code_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_assoc_type ON gl_eudr_qrg_code_associations (association_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_target_id ON gl_eudr_qrg_code_associations (target_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_target_type ON gl_eudr_qrg_code_associations (target_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_created ON gl_eudr_qrg_code_associations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_metadata ON gl_eudr_qrg_code_associations USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_ca_code_target ON gl_eudr_qrg_code_associations (code_id, target_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 12. gl_eudr_qrg_audit_log — Immutable audit trail
-- ============================================================================
RAISE NOTICE 'V102 [12/12]: Creating gl_eudr_qrg_audit_log...';

CREATE TABLE IF NOT EXISTS gl_eudr_qrg_audit_log (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    log_id                  TEXT            UNIQUE NOT NULL,
    action                  TEXT            NOT NULL,
        -- 'code_created', 'code_activated', 'code_revoked', 'code_expired',
        -- 'payload_composed', 'label_rendered', 'batch_code_generated',
        -- 'url_generated', 'signature_created', 'signature_verified',
        -- 'scan_recorded', 'bulk_job_started', 'bulk_job_completed',
        -- 'template_created', 'association_created'
    entity_type             TEXT            NOT NULL,
        -- 'code', 'payload', 'label', 'batch_code', 'verification_url',
        -- 'signature', 'scan_event', 'bulk_job', 'lifecycle_event',
        -- 'template', 'code_association'
    entity_id               TEXT            NOT NULL,
    actor_id                UUID            NOT NULL,
    details                 JSONB,
    ip_address              INET,
    provenance_hash         TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_log_id ON gl_eudr_qrg_audit_log (log_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_action ON gl_eudr_qrg_audit_log (action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_entity_type ON gl_eudr_qrg_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_entity_id ON gl_eudr_qrg_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_actor ON gl_eudr_qrg_audit_log (actor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_created ON gl_eudr_qrg_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_details ON gl_eudr_qrg_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_action_created ON gl_eudr_qrg_audit_log (action, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_qrg_al_entity_type_created ON gl_eudr_qrg_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================
RAISE NOTICE 'V102: Creating continuous aggregates...';

-- 1. Hourly code generation statistics by content_type, symbology, and status
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_qrg_hourly_code_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at)       AS bucket,
    content_type,
    symbology_type,
    status,
    COUNT(*)                                AS code_count,
    COUNT(*) FILTER (WHERE status = 'active')     AS active_count,
    COUNT(*) FILTER (WHERE status = 'revoked')    AS revoked_count,
    COUNT(*) FILTER (WHERE status = 'expired')    AS expired_count,
    COUNT(*) FILTER (WHERE has_logo = TRUE)       AS logo_count,
    COUNT(*) FILTER (WHERE blockchain_anchor_hash IS NOT NULL) AS anchored_count
FROM gl_eudr_qrg_codes
GROUP BY bucket, content_type, symbology_type, status
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_qrg_hourly_code_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- 2. Hourly scan statistics by outcome and country
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_qrg_hourly_scan_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', scanned_at)       AS bucket,
    scan_outcome,
    country_code,
    COUNT(*)                                AS scan_count,
    COUNT(*) FILTER (WHERE scan_outcome = 'valid')              AS valid_count,
    COUNT(*) FILTER (WHERE scan_outcome = 'invalid')            AS invalid_count,
    COUNT(*) FILTER (WHERE scan_outcome = 'counterfeit_suspected') AS counterfeit_count,
    COUNT(*) FILTER (WHERE scan_outcome = 'expired')            AS expired_count,
    COUNT(*) FILTER (WHERE scan_outcome = 'revoked')            AS revoked_count,
    AVG(counterfeit_risk_score)             AS avg_risk_score,
    MAX(counterfeit_risk_score)             AS max_risk_score
FROM gl_eudr_qrg_scan_events
GROUP BY bucket, scan_outcome, country_code
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_qrg_hourly_scan_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies (90 days operational window)
-- ============================================================================
RAISE NOTICE 'V102: Adding retention policies (90 days operational window)...';

SELECT add_retention_policy('gl_eudr_qrg_codes',
    INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_qrg_scan_events',
    INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_qrg_lifecycle_events',
    INTERVAL '90 days', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================
RAISE NOTICE 'V102: Adding table comments...';

COMMENT ON TABLE gl_eudr_qrg_codes IS 'AGENT-EUDR-014: QR code records with symbology settings, payload references, image data, HMAC signatures, blockchain anchoring, and lifecycle status tracking (hypertable)';
COMMENT ON TABLE gl_eudr_qrg_payloads IS 'AGENT-EUDR-014: Payload composition records with structured data, GS1 URI, commodity/origin metadata, compression, encryption, and validation error tracking';
COMMENT ON TABLE gl_eudr_qrg_labels IS 'AGENT-EUDR-014: Label rendering records with template-based layout, dimensions, format output, font/color customization, and embedded QR code references';
COMMENT ON TABLE gl_eudr_qrg_batch_codes IS 'AGENT-EUDR-014: Batch code management with hierarchical parent-child codes, GS1/SSCC/GTIN types, sequence numbering, and blockchain anchor linkage';
COMMENT ON TABLE gl_eudr_qrg_verification_urls IS 'AGENT-EUDR-014: Verification URL generation with HMAC-signed endpoints, short URLs, time-limited tokens, operator codes, and deep link support';
COMMENT ON TABLE gl_eudr_qrg_signatures IS 'AGENT-EUDR-014: HMAC and digital signature records with algorithm tracking, key rotation dates, validation status, and verification timestamps';
COMMENT ON TABLE gl_eudr_qrg_scan_events IS 'AGENT-EUDR-014: Scan event tracking with geolocation, outcome classification, counterfeit risk scoring, device fingerprinting, and verification results (hypertable)';
COMMENT ON TABLE gl_eudr_qrg_bulk_jobs IS 'AGENT-EUDR-014: Bulk job processing for mass code generation, printing, export, and revocation with progress counters and manifest tracking';
COMMENT ON TABLE gl_eudr_qrg_lifecycle_events IS 'AGENT-EUDR-014: Lifecycle state transition log tracking code status changes from creation through activation, suspension, revocation, and expiry (hypertable)';
COMMENT ON TABLE gl_eudr_qrg_templates IS 'AGENT-EUDR-014: Label template definitions with dimension specifications, QR code sizing, positioned layout elements, and multi-format support';
COMMENT ON TABLE gl_eudr_qrg_code_associations IS 'AGENT-EUDR-014: Code-to-entity association records linking QR codes to DDS statements, shipments, products, batches, and geolocation plots';
COMMENT ON TABLE gl_eudr_qrg_audit_log IS 'AGENT-EUDR-014: Immutable audit trail for all QR code generator operations with actor tracking, IP logging, and provenance hashing';

RAISE NOTICE 'V102: AGENT-EUDR-014 QR Code Generator migration complete.';

COMMIT;
