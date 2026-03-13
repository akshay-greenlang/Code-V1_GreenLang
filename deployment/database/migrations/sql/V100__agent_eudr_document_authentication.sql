-- ============================================================================
-- V100: AGENT-EUDR-012 Document Authentication & Verification Agent
-- ============================================================================
-- Creates tables for document master records, classification results, digital
-- signature verification, immutable hash registry, X.509 certificate chain
-- validation, extracted metadata, fraud detection alerts, external registry
-- cross-reference results, trusted CA store, document templates, authentication
-- reports, and audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_dav_classifications, gl_eudr_dav_fraud_alerts,
--              gl_eudr_dav_crossref_results
-- Continuous Aggregates: 2 (hourly_classifications + daily_fraud_alerts)
-- Retention Policies: 3 (hypertables, 5-year per EUDR Article 14)
-- Indexes: 88
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V100: Creating AGENT-EUDR-012 Document Authentication tables...';

-- ============================================================================
-- 1. gl_eudr_dav_documents — Document master records
-- ============================================================================
RAISE NOTICE 'V100 [1/12]: Creating gl_eudr_dav_documents...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_documents (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    document_type           VARCHAR(50)     NOT NULL,
    filename                VARCHAR(500)    NOT NULL,
    file_size_bytes         BIGINT          NOT NULL,
    mime_type               VARCHAR(100),
    sha256_hash             VARCHAR(64)     NOT NULL,
    sha512_hash             VARCHAR(128),
    upload_source           VARCHAR(100),
        -- 'api', 'batch', 'dds_submission'
    operator_id             VARCHAR(100)    NOT NULL,
    dds_id                  VARCHAR(100),
    batch_id                VARCHAR(100),
    shipment_id             VARCHAR(100),
    authentication_status   VARCHAR(30)     DEFAULT 'pending',
        -- pending, authentic, suspicious, fraudulent, inconclusive
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100)    NOT NULL,
    provenance_hash         VARCHAR(64)     NOT NULL
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_type ON gl_eudr_dav_documents (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_operator ON gl_eudr_dav_documents (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_dds ON gl_eudr_dav_documents (dds_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_batch ON gl_eudr_dav_documents (batch_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_shipment ON gl_eudr_dav_documents (shipment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_sha256 ON gl_eudr_dav_documents (sha256_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_status ON gl_eudr_dav_documents (authentication_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_upload ON gl_eudr_dav_documents (upload_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_doc_created ON gl_eudr_dav_documents (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_dav_classifications — Classification results (hypertable)
-- ============================================================================
RAISE NOTICE 'V100 [2/12]: Creating gl_eudr_dav_classifications (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_classifications (
    id                      UUID            DEFAULT gen_random_uuid(),
    document_id             UUID            NOT NULL,
    classified_type         VARCHAR(50)     NOT NULL,
    confidence_level        VARCHAR(20)     NOT NULL,
        -- 'high', 'medium', 'low', 'unknown'
    confidence_score        DECIMAL(5,4)    NOT NULL,
    matched_template_id     VARCHAR(100),
    language_detected       VARCHAR(10),
    classification_method   VARCHAR(50),
        -- 'template_match', 'rule_based', 'field_pattern'
    classification_details  JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100)    NOT NULL,
    provenance_hash         VARCHAR(64)     NOT NULL,

    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_dav_classifications',
    'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cls_document ON gl_eudr_dav_classifications (document_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cls_type ON gl_eudr_dav_classifications (classified_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cls_confidence ON gl_eudr_dav_classifications (confidence_level, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cls_template ON gl_eudr_dav_classifications (matched_template_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cls_method ON gl_eudr_dav_classifications (classification_method, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cls_details ON gl_eudr_dav_classifications USING GIN (classification_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_dav_signatures — Signature verification records
-- ============================================================================
RAISE NOTICE 'V100 [3/12]: Creating gl_eudr_dav_signatures...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_signatures (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id             UUID            NOT NULL REFERENCES gl_eudr_dav_documents(id),
    signature_standard      VARCHAR(20)     NOT NULL,
        -- 'CAdES', 'PAdES', 'XAdES', etc.
    verification_status     VARCHAR(30)     NOT NULL,
        -- 'valid', 'invalid', 'expired', 'revoked', 'unknown', 'malformed'
    signer_cn               VARCHAR(500),
    signer_org              VARCHAR(500),
    signer_country          VARCHAR(10),
    signer_email            VARCHAR(300),
    signing_time            TIMESTAMPTZ,
    timestamp_valid         BOOLEAN         DEFAULT FALSE,
    certificate_serial      VARCHAR(200),
    certificate_issuer      VARCHAR(500),
    verification_details    JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100)    NOT NULL,
    provenance_hash         VARCHAR(64)     NOT NULL
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_document ON gl_eudr_dav_signatures (document_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_standard ON gl_eudr_dav_signatures (signature_standard);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_status ON gl_eudr_dav_signatures (verification_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_signer_cn ON gl_eudr_dav_signatures (signer_cn);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_signing_time ON gl_eudr_dav_signatures (signing_time);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_cert_serial ON gl_eudr_dav_signatures (certificate_serial);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_created ON gl_eudr_dav_signatures (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_sig_details ON gl_eudr_dav_signatures USING GIN (verification_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_dav_hash_registry — Immutable document hash registry
-- ============================================================================
RAISE NOTICE 'V100 [4/12]: Creating gl_eudr_dav_hash_registry...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_hash_registry (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256_hash             VARCHAR(64)     NOT NULL UNIQUE,
    sha512_hash             VARCHAR(128),
    document_id             UUID            NOT NULL REFERENCES gl_eudr_dav_documents(id),
    first_seen_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_verified_at        TIMESTAMPTZ,
    verification_count      INTEGER         DEFAULT 1,
    merkle_parent_hash      VARCHAR(64),
    provenance_hash         VARCHAR(64)     NOT NULL
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_hash_sha256 ON gl_eudr_dav_hash_registry (sha256_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_hash_sha512 ON gl_eudr_dav_hash_registry (sha512_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_hash_document ON gl_eudr_dav_hash_registry (document_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_hash_first_seen ON gl_eudr_dav_hash_registry (first_seen_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_hash_last_verified ON gl_eudr_dav_hash_registry (last_verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_hash_merkle ON gl_eudr_dav_hash_registry (merkle_parent_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_dav_certificate_chains — X.509 chain validation
-- ============================================================================
RAISE NOTICE 'V100 [5/12]: Creating gl_eudr_dav_certificate_chains...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_certificate_chains (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id             UUID            NOT NULL REFERENCES gl_eudr_dav_documents(id),
    leaf_cert_subject       VARCHAR(500),
    leaf_cert_issuer        VARCHAR(500),
    leaf_cert_serial        VARCHAR(200),
    leaf_cert_not_before    TIMESTAMPTZ,
    leaf_cert_not_after     TIMESTAMPTZ,
    chain_length            INTEGER         NOT NULL,
    root_ca_subject         VARCHAR(500),
    chain_valid             BOOLEAN         NOT NULL,
    ocsp_status             VARCHAR(30),
        -- 'good', 'revoked', 'unknown', 'error'
    crl_status              VARCHAR(30),
    key_size_bits           INTEGER,
    key_algorithm           VARCHAR(30),
        -- 'RSA', 'ECDSA', 'Ed25519'
    validation_details      JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64)     NOT NULL
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_document ON gl_eudr_dav_certificate_chains (document_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_subject ON gl_eudr_dav_certificate_chains (leaf_cert_subject);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_issuer ON gl_eudr_dav_certificate_chains (leaf_cert_issuer);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_serial ON gl_eudr_dav_certificate_chains (leaf_cert_serial);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_valid ON gl_eudr_dav_certificate_chains (chain_valid);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_ocsp ON gl_eudr_dav_certificate_chains (ocsp_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_algorithm ON gl_eudr_dav_certificate_chains (key_algorithm);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_expiry ON gl_eudr_dav_certificate_chains (leaf_cert_not_after);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_cert_details ON gl_eudr_dav_certificate_chains USING GIN (validation_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_dav_metadata — Extracted document metadata
-- ============================================================================
RAISE NOTICE 'V100 [6/12]: Creating gl_eudr_dav_metadata...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_metadata (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id             UUID            NOT NULL REFERENCES gl_eudr_dav_documents(id),
    title                   VARCHAR(1000),
    author                  VARCHAR(500),
    creator                 VARCHAR(500),
    producer                VARCHAR(500),
    creation_date           TIMESTAMPTZ,
    modification_date       TIMESTAMPTZ,
    keywords                TEXT,
    gps_latitude            DECIMAL(10,7),
    gps_longitude           DECIMAL(10,7),
    exif_data               JSONB           DEFAULT '{}',
    xmp_data                JSONB           DEFAULT '{}',
    consistency_valid       BOOLEAN,
    consistency_issues      JSONB           DEFAULT '[]',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64)     NOT NULL
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_meta_document ON gl_eudr_dav_metadata (document_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_meta_author ON gl_eudr_dav_metadata (author);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_meta_creation_date ON gl_eudr_dav_metadata (creation_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_meta_consistency ON gl_eudr_dav_metadata (consistency_valid);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_meta_gps ON gl_eudr_dav_metadata (gps_latitude, gps_longitude);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_meta_exif ON gl_eudr_dav_metadata USING GIN (exif_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_meta_issues ON gl_eudr_dav_metadata USING GIN (consistency_issues);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_dav_fraud_alerts — Fraud detection alerts (hypertable)
-- ============================================================================
RAISE NOTICE 'V100 [7/12]: Creating gl_eudr_dav_fraud_alerts (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_fraud_alerts (
    id                      UUID            DEFAULT gen_random_uuid(),
    document_id             UUID            NOT NULL,
    fraud_pattern           VARCHAR(50)     NOT NULL,
    severity                VARCHAR(20)     NOT NULL,
        -- 'low', 'medium', 'high', 'critical'
    description             TEXT            NOT NULL,
    evidence                JSONB           DEFAULT '{}',
    related_document_ids    UUID[]          DEFAULT '{}',
    resolved                BOOLEAN         DEFAULT FALSE,
    resolved_by             VARCHAR(100),
    resolved_at             TIMESTAMPTZ,
    resolution_notes        TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100)    NOT NULL,
    provenance_hash         VARCHAR(64)     NOT NULL,

    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_dav_fraud_alerts',
    'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_fraud_document ON gl_eudr_dav_fraud_alerts (document_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_fraud_pattern ON gl_eudr_dav_fraud_alerts (fraud_pattern, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_fraud_severity ON gl_eudr_dav_fraud_alerts (severity, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_fraud_resolved ON gl_eudr_dav_fraud_alerts (resolved, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_fraud_resolved_by ON gl_eudr_dav_fraud_alerts (resolved_by, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_fraud_evidence ON gl_eudr_dav_fraud_alerts USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_fraud_related ON gl_eudr_dav_fraud_alerts USING GIN (related_document_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_dav_crossref_results — External registry cross-reference (hypertable)
-- ============================================================================
RAISE NOTICE 'V100 [8/12]: Creating gl_eudr_dav_crossref_results (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_crossref_results (
    id                      UUID            DEFAULT gen_random_uuid(),
    document_id             UUID            NOT NULL,
    registry_type           VARCHAR(30)     NOT NULL,
    certificate_number      VARCHAR(200),
    registry_status         VARCHAR(30)     NOT NULL,
        -- 'verified', 'not_found', 'expired', 'revoked', 'error'
    registry_response       JSONB           DEFAULT '{}',
    scope_match             BOOLEAN,
    quantity_match          BOOLEAN,
    party_match             BOOLEAN,
    cached                  BOOLEAN         DEFAULT FALSE,
    query_duration_ms       DECIMAL(10,2),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64)     NOT NULL,

    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_dav_crossref_results',
    'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_document ON gl_eudr_dav_crossref_results (document_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_registry ON gl_eudr_dav_crossref_results (registry_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_cert ON gl_eudr_dav_crossref_results (certificate_number, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_status ON gl_eudr_dav_crossref_results (registry_status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_scope ON gl_eudr_dav_crossref_results (scope_match, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_quantity ON gl_eudr_dav_crossref_results (quantity_match, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_party ON gl_eudr_dav_crossref_results (party_match, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_xref_response ON gl_eudr_dav_crossref_results USING GIN (registry_response);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_dav_trusted_cas — Trusted CA certificate store
-- ============================================================================
RAISE NOTICE 'V100 [9/12]: Creating gl_eudr_dav_trusted_cas...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_trusted_cas (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    ca_name                 VARCHAR(500)    NOT NULL,
    ca_category             VARCHAR(50)     NOT NULL,
        -- 'eidas_tsp', 'document_signing', 'government', 'certification_body'
    subject_dn              VARCHAR(1000)   NOT NULL UNIQUE,
    issuer_dn               VARCHAR(1000),
    serial_number           VARCHAR(200),
    not_before              TIMESTAMPTZ     NOT NULL,
    not_after               TIMESTAMPTZ     NOT NULL,
    key_algorithm           VARCHAR(30),
    key_size_bits           INTEGER,
    fingerprint_sha256      VARCHAR(64)     NOT NULL UNIQUE,
    certificate_pem         TEXT            NOT NULL,
    is_active               BOOLEAN         DEFAULT TRUE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_ca_name ON gl_eudr_dav_trusted_cas (ca_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_ca_category ON gl_eudr_dav_trusted_cas (ca_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_ca_subject ON gl_eudr_dav_trusted_cas (subject_dn);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_ca_fingerprint ON gl_eudr_dav_trusted_cas (fingerprint_sha256);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_ca_active ON gl_eudr_dav_trusted_cas (is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_ca_expiry ON gl_eudr_dav_trusted_cas (not_after);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_ca_algorithm ON gl_eudr_dav_trusted_cas (key_algorithm);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 10. gl_eudr_dav_document_templates — Known document templates
-- ============================================================================
RAISE NOTICE 'V100 [10/12]: Creating gl_eudr_dav_document_templates...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_document_templates (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    document_type           VARCHAR(50)     NOT NULL,
    country_code            VARCHAR(10)     NOT NULL,
    issuing_authority       VARCHAR(500)    NOT NULL,
    template_name           VARCHAR(200)    NOT NULL,
    template_version        VARCHAR(20),
    key_indicators          JSONB           NOT NULL,
        -- field patterns, header patterns, logo patterns
    serial_number_pattern   VARCHAR(200),
        -- regex for serial number format
    required_fields         JSONB           DEFAULT '[]',
    is_active               BOOLEAN         DEFAULT TRUE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE(document_type, country_code, issuing_authority, template_version)
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_tpl_type ON gl_eudr_dav_document_templates (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_tpl_country ON gl_eudr_dav_document_templates (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_tpl_authority ON gl_eudr_dav_document_templates (issuing_authority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_tpl_type_country ON gl_eudr_dav_document_templates (document_type, country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_tpl_active ON gl_eudr_dav_document_templates (is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_tpl_indicators ON gl_eudr_dav_document_templates USING GIN (key_indicators);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_tpl_fields ON gl_eudr_dav_document_templates USING GIN (required_fields);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 11. gl_eudr_dav_authentication_reports — Generated authentication reports
-- ============================================================================
RAISE NOTICE 'V100 [11/12]: Creating gl_eudr_dav_authentication_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_authentication_reports (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type             VARCHAR(50)     NOT NULL,
        -- 'authentication', 'evidence_package', 'fraud_summary', 'completeness'
    operator_id             VARCHAR(100)    NOT NULL,
    dds_id                  VARCHAR(100),
    document_count          INTEGER         NOT NULL,
    authentic_count         INTEGER         DEFAULT 0,
    suspicious_count        INTEGER         DEFAULT 0,
    fraudulent_count        INTEGER         DEFAULT 0,
    overall_result          VARCHAR(30)     NOT NULL,
        -- 'pass', 'fail', 'review_required', 'incomplete'
    fraud_risk_score        DECIMAL(5,2),
    report_format           VARCHAR(20)     NOT NULL,
        -- 'json', 'pdf', 'csv', 'xml'
    report_data             JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100)    NOT NULL,
    provenance_hash         VARCHAR(64)     NOT NULL
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_rpt_type ON gl_eudr_dav_authentication_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_rpt_operator ON gl_eudr_dav_authentication_reports (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_rpt_dds ON gl_eudr_dav_authentication_reports (dds_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_rpt_result ON gl_eudr_dav_authentication_reports (overall_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_rpt_risk ON gl_eudr_dav_authentication_reports (fraud_risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_rpt_created ON gl_eudr_dav_authentication_reports (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_rpt_data ON gl_eudr_dav_authentication_reports USING GIN (report_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 12. gl_eudr_dav_audit_log — Immutable audit trail
-- ============================================================================
RAISE NOTICE 'V100 [12/12]: Creating gl_eudr_dav_audit_log...';

CREATE TABLE IF NOT EXISTS gl_eudr_dav_audit_log (
    id                      BIGSERIAL       PRIMARY KEY,
    event_type              VARCHAR(50)     NOT NULL,
        -- 'document_uploaded', 'classification_completed', 'signature_verified',
        -- 'hash_registered', 'chain_validated', 'metadata_extracted',
        -- 'fraud_detected', 'fraud_resolved', 'crossref_completed',
        -- 'report_generated', 'template_updated', 'ca_updated'
    entity_type             VARCHAR(50)     NOT NULL,
        -- 'document', 'classification', 'signature', 'hash_registry',
        -- 'certificate_chain', 'metadata', 'fraud_alert', 'crossref_result',
        -- 'trusted_ca', 'document_template', 'authentication_report'
    entity_id               VARCHAR(100)    NOT NULL,
    action                  VARCHAR(50)     NOT NULL,
        -- 'created', 'updated', 'verified', 'authenticated', 'flagged',
        -- 'resolved', 'revoked', 'expired', 'cross_referenced'
    actor_id                VARCHAR(100)    NOT NULL,
    actor_ip                VARCHAR(45),
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64)     NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_audit_event ON gl_eudr_dav_audit_log (event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_audit_entity_type ON gl_eudr_dav_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_audit_entity_id ON gl_eudr_dav_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_audit_action ON gl_eudr_dav_audit_log (action, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_audit_actor ON gl_eudr_dav_audit_log (actor_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_audit_created ON gl_eudr_dav_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dav_audit_details ON gl_eudr_dav_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================
RAISE NOTICE 'V100: Creating continuous aggregates...';

-- 1. Hourly classification counts by type and confidence level
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_dav_hourly_classifications
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at)       AS bucket,
    classified_type,
    confidence_level,
    classification_method,
    COUNT(*)                                AS classification_count,
    AVG(confidence_score)                   AS avg_confidence_score,
    MIN(confidence_score)                   AS min_confidence_score,
    MAX(confidence_score)                   AS max_confidence_score,
    COUNT(DISTINCT document_id)             AS unique_documents
FROM gl_eudr_dav_classifications
GROUP BY bucket, classified_type, confidence_level, classification_method
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_dav_hourly_classifications',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- 2. Daily fraud alert counts by pattern and severity
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_dav_daily_fraud_alerts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at)        AS bucket,
    fraud_pattern,
    severity,
    COUNT(*)                                AS alert_count,
    COUNT(*) FILTER (WHERE resolved = TRUE) AS resolved_count,
    COUNT(*) FILTER (WHERE resolved = FALSE) AS unresolved_count,
    COUNT(DISTINCT document_id)             AS affected_documents
FROM gl_eudr_dav_fraud_alerts
GROUP BY bucket, fraud_pattern, severity
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_dav_daily_fraud_alerts',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies (5 years per EUDR Article 14)
-- ============================================================================
RAISE NOTICE 'V100: Adding retention policies (5 years per EUDR Article 14)...';

SELECT add_retention_policy('gl_eudr_dav_classifications',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_dav_fraud_alerts',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_dav_crossref_results',
    INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================
RAISE NOTICE 'V100: Adding table comments...';

COMMENT ON TABLE gl_eudr_dav_documents IS 'AGENT-EUDR-012: Document master records with hash fingerprints, operator linkage, and authentication status tracking';
COMMENT ON TABLE gl_eudr_dav_classifications IS 'AGENT-EUDR-012: Document classification results with confidence scoring, template matching, and method tracking (hypertable)';
COMMENT ON TABLE gl_eudr_dav_signatures IS 'AGENT-EUDR-012: Digital signature verification records for CAdES/PAdES/XAdES standards with signer details and timestamp validation';
COMMENT ON TABLE gl_eudr_dav_hash_registry IS 'AGENT-EUDR-012: Immutable SHA-256/SHA-512 hash registry with Merkle tree support for tamper-evident document integrity';
COMMENT ON TABLE gl_eudr_dav_certificate_chains IS 'AGENT-EUDR-012: X.509 certificate chain validation with OCSP/CRL revocation checking and key algorithm tracking';
COMMENT ON TABLE gl_eudr_dav_metadata IS 'AGENT-EUDR-012: Extracted document metadata including GPS coordinates, EXIF/XMP data, and consistency validation';
COMMENT ON TABLE gl_eudr_dav_fraud_alerts IS 'AGENT-EUDR-012: Fraud detection alerts with pattern classification, severity levels, and resolution tracking (hypertable)';
COMMENT ON TABLE gl_eudr_dav_crossref_results IS 'AGENT-EUDR-012: External registry cross-reference results with scope/quantity/party matching and cache tracking (hypertable)';
COMMENT ON TABLE gl_eudr_dav_trusted_cas IS 'AGENT-EUDR-012: Trusted CA certificate store for eIDAS TSPs, government CAs, and certification body signing certificates';
COMMENT ON TABLE gl_eudr_dav_document_templates IS 'AGENT-EUDR-012: Known document templates with field patterns, serial number regex, and required field definitions per country/authority';
COMMENT ON TABLE gl_eudr_dav_authentication_reports IS 'AGENT-EUDR-012: Generated authentication reports with document counts, fraud risk scores, and evidence packages';
COMMENT ON TABLE gl_eudr_dav_audit_log IS 'AGENT-EUDR-012: Immutable audit trail for all document authentication operations with actor tracking and provenance hashing';

RAISE NOTICE 'V100: AGENT-EUDR-012 Document Authentication migration complete.';

COMMIT;
