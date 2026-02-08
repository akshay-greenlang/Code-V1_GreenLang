-- =============================================================================
-- GreenLang Climate OS - PDF & Invoice Extractor Service Schema
-- =============================================================================
-- Migration: V031
-- Component: AGENT-DATA-001 PDF & Invoice Extractor
-- Description: Creates pdf_extractor_service schema with documents,
--              document_pages, extraction_jobs (hypertable), extracted_fields,
--              invoice_extractions, manifest_extractions, utility_bill_extractions,
--              extraction_templates, validation_results, pdf_audit_log (hypertable),
--              continuous aggregates for hourly extraction stats and hourly audit
--              stats, 50+ indexes (including GIN indexes on JSONB), RLS policies
--              per tenant, 14 security permissions, retention policies (30-day
--              extraction_jobs, 365-day audit_log), compression, and seed data
--              registering the PDF Extractor Agent (GL-DATA-X-001) with
--              capabilities in the agent registry.
-- Previous: V030 (reserved) / V029__qa_test_harness_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS pdf_extractor_service;

-- =============================================================================
-- Table: pdf_extractor_service.documents
-- =============================================================================
-- Document registry. Each document record captures file metadata including name,
-- path, size, SHA-256 hash, document type classification, format, page count,
-- upload timestamp, tenant scope, provenance hash for integrity verification,
-- and the uploading user. Tenant-scoped.

CREATE TABLE pdf_extractor_service.documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name VARCHAR(500) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    file_hash VARCHAR(64) NOT NULL,
    document_type VARCHAR(30) NOT NULL,
    document_format VARCHAR(10) NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Document type constraint
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_type
    CHECK (document_type IN (
        'invoice', 'utility_bill', 'shipping_manifest', 'receipt',
        'purchase_order', 'bill_of_lading', 'certificate',
        'report', 'contract', 'form', 'other'
    ));

-- Document format constraint
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_format
    CHECK (document_format IN ('pdf', 'png', 'jpg', 'tiff', 'bmp'));

-- File name must not be empty
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_file_name_not_empty
    CHECK (LENGTH(TRIM(file_name)) > 0);

-- File path must not be empty
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_file_path_not_empty
    CHECK (LENGTH(TRIM(file_path)) > 0);

-- File size must be non-negative
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_file_size_non_negative
    CHECK (file_size_bytes >= 0);

-- File hash must be 64-character hex (SHA-256)
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_file_hash_length
    CHECK (LENGTH(file_hash) = 64);

-- Provenance hash must be 64-character hex
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Page count must be non-negative
ALTER TABLE pdf_extractor_service.documents
    ADD CONSTRAINT chk_document_page_count_non_negative
    CHECK (page_count >= 0);

-- =============================================================================
-- Table: pdf_extractor_service.document_pages
-- =============================================================================
-- Individual page records for multi-page documents. Each page record captures
-- the raw extracted text, word count, OCR confidence score, engine used,
-- bounding boxes (JSONB with positional data for each text region), and
-- provenance hash for integrity verification. Tenant-scoped.

CREATE TABLE pdf_extractor_service.document_pages (
    page_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES pdf_extractor_service.documents(document_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    raw_text TEXT DEFAULT '',
    word_count INTEGER NOT NULL DEFAULT 0,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    ocr_engine_used VARCHAR(30) NOT NULL DEFAULT 'tesseract',
    bounding_boxes JSONB DEFAULT '[]'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Page number must be positive
ALTER TABLE pdf_extractor_service.document_pages
    ADD CONSTRAINT chk_page_number_positive
    CHECK (page_number > 0);

-- Word count must be non-negative
ALTER TABLE pdf_extractor_service.document_pages
    ADD CONSTRAINT chk_page_word_count_non_negative
    CHECK (word_count >= 0);

-- Confidence must be between 0 and 1
ALTER TABLE pdf_extractor_service.document_pages
    ADD CONSTRAINT chk_page_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- OCR engine constraint
ALTER TABLE pdf_extractor_service.document_pages
    ADD CONSTRAINT chk_page_ocr_engine
    CHECK (ocr_engine_used IN (
        'tesseract', 'textract', 'azure_form_recognizer',
        'google_document_ai', 'easyocr', 'paddleocr', 'native_pdf'
    ));

-- Provenance hash must be 64-character hex
ALTER TABLE pdf_extractor_service.document_pages
    ADD CONSTRAINT chk_page_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Unique page number per document
ALTER TABLE pdf_extractor_service.document_pages
    ADD CONSTRAINT uq_document_page_number
    UNIQUE (document_id, page_number);

-- =============================================================================
-- Table: pdf_extractor_service.extraction_jobs (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording extraction job executions. Each extraction
-- job captures the document being processed, job status, document type, OCR
-- engine, confidence threshold, start/completion times, duration, error
-- message, fields extracted count, pages processed count, and tenant scope.
-- Partitioned by started_at for time-series queries. Retained for 30 days
-- with compression after 3 days.

CREATE TABLE pdf_extractor_service.extraction_jobs (
    job_id UUID NOT NULL DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    job_status VARCHAR(20) NOT NULL,
    document_type VARCHAR(30) NOT NULL,
    ocr_engine VARCHAR(30) NOT NULL DEFAULT 'tesseract',
    confidence_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.7,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    error_message TEXT DEFAULT NULL,
    fields_extracted INTEGER NOT NULL DEFAULT 0,
    pages_processed INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system',
    PRIMARY KEY (job_id, started_at)
);

-- Create hypertable partitioned by started_at
SELECT create_hypertable('pdf_extractor_service.extraction_jobs', 'started_at', if_not_exists => TRUE);

-- Job status constraint
ALTER TABLE pdf_extractor_service.extraction_jobs
    ADD CONSTRAINT chk_job_status
    CHECK (job_status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout'));

-- Document type constraint
ALTER TABLE pdf_extractor_service.extraction_jobs
    ADD CONSTRAINT chk_job_document_type
    CHECK (document_type IN (
        'invoice', 'utility_bill', 'shipping_manifest', 'receipt',
        'purchase_order', 'bill_of_lading', 'certificate',
        'report', 'contract', 'form', 'other'
    ));

-- OCR engine constraint
ALTER TABLE pdf_extractor_service.extraction_jobs
    ADD CONSTRAINT chk_job_ocr_engine
    CHECK (ocr_engine IN (
        'tesseract', 'textract', 'azure_form_recognizer',
        'google_document_ai', 'easyocr', 'paddleocr', 'native_pdf'
    ));

-- Confidence threshold must be between 0 and 1
ALTER TABLE pdf_extractor_service.extraction_jobs
    ADD CONSTRAINT chk_job_confidence_threshold_range
    CHECK (confidence_threshold >= 0 AND confidence_threshold <= 1);

-- Duration must be non-negative
ALTER TABLE pdf_extractor_service.extraction_jobs
    ADD CONSTRAINT chk_job_duration_non_negative
    CHECK (duration_ms >= 0);

-- Fields extracted must be non-negative
ALTER TABLE pdf_extractor_service.extraction_jobs
    ADD CONSTRAINT chk_job_fields_extracted_non_negative
    CHECK (fields_extracted >= 0);

-- Pages processed must be non-negative
ALTER TABLE pdf_extractor_service.extraction_jobs
    ADD CONSTRAINT chk_job_pages_processed_non_negative
    CHECK (pages_processed >= 0);

-- =============================================================================
-- Table: pdf_extractor_service.extracted_fields
-- =============================================================================
-- Individual field extraction records. Each record captures a single
-- extracted data field from a document including the field name, value,
-- raw text, confidence score, field type classification, bounding box
-- coordinates, extraction method, validation status, and provenance hash.

CREATE TABLE pdf_extractor_service.extracted_fields (
    field_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES pdf_extractor_service.documents(document_id) ON DELETE CASCADE,
    job_id UUID NOT NULL,
    field_name VARCHAR(100) NOT NULL,
    value TEXT DEFAULT NULL,
    raw_text TEXT DEFAULT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    field_type VARCHAR(30) NOT NULL DEFAULT 'text',
    bounding_box JSONB DEFAULT NULL,
    extraction_method VARCHAR(50) NOT NULL DEFAULT 'ocr',
    validated BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL
);

-- Confidence must be between 0 and 1
ALTER TABLE pdf_extractor_service.extracted_fields
    ADD CONSTRAINT chk_field_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Field type constraint
ALTER TABLE pdf_extractor_service.extracted_fields
    ADD CONSTRAINT chk_field_type
    CHECK (field_type IN (
        'text', 'number', 'currency', 'date', 'address',
        'email', 'phone', 'percentage', 'quantity', 'unit',
        'identifier', 'boolean', 'table_cell', 'other'
    ));

-- Extraction method constraint
ALTER TABLE pdf_extractor_service.extracted_fields
    ADD CONSTRAINT chk_field_extraction_method
    CHECK (extraction_method IN (
        'ocr', 'native_text', 'regex', 'template_match',
        'ml_model', 'rule_based', 'hybrid', 'manual'
    ));

-- Field name must not be empty
ALTER TABLE pdf_extractor_service.extracted_fields
    ADD CONSTRAINT chk_field_name_not_empty
    CHECK (LENGTH(TRIM(field_name)) > 0);

-- Provenance hash must be 64-character hex
ALTER TABLE pdf_extractor_service.extracted_fields
    ADD CONSTRAINT chk_field_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: pdf_extractor_service.invoice_extractions
-- =============================================================================
-- Structured invoice extraction results. Each record contains the full
-- parsed invoice data (JSONB) including vendor, line items, totals, dates,
-- PO numbers, tax amounts. Includes overall confidence, per-field confidence
-- map, validation status, and validation errors.

CREATE TABLE pdf_extractor_service.invoice_extractions (
    extraction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES pdf_extractor_service.documents(document_id) ON DELETE CASCADE,
    invoice_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    overall_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    field_confidences JSONB DEFAULT '{}'::jsonb,
    validation_passed BOOLEAN NOT NULL DEFAULT false,
    validation_errors JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL
);

-- Confidence must be between 0 and 1
ALTER TABLE pdf_extractor_service.invoice_extractions
    ADD CONSTRAINT chk_invoice_confidence_range
    CHECK (overall_confidence >= 0 AND overall_confidence <= 1);

-- Provenance hash must be 64-character hex
ALTER TABLE pdf_extractor_service.invoice_extractions
    ADD CONSTRAINT chk_invoice_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: pdf_extractor_service.manifest_extractions
-- =============================================================================
-- Structured shipping manifest extraction results. Each record contains the
-- full parsed manifest data (JSONB) including shipper, consignee, cargo
-- details, weight, hazmat classification, route. Includes overall confidence,
-- per-field confidence map, and validation status.

CREATE TABLE pdf_extractor_service.manifest_extractions (
    extraction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES pdf_extractor_service.documents(document_id) ON DELETE CASCADE,
    manifest_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    overall_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    field_confidences JSONB DEFAULT '{}'::jsonb,
    validation_passed BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL
);

-- Confidence must be between 0 and 1
ALTER TABLE pdf_extractor_service.manifest_extractions
    ADD CONSTRAINT chk_manifest_confidence_range
    CHECK (overall_confidence >= 0 AND overall_confidence <= 1);

-- Provenance hash must be 64-character hex
ALTER TABLE pdf_extractor_service.manifest_extractions
    ADD CONSTRAINT chk_manifest_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: pdf_extractor_service.utility_bill_extractions
-- =============================================================================
-- Structured utility bill extraction results. Each record contains the full
-- parsed utility data (JSONB) including provider, account, service period,
-- consumption (kWh, therms, gallons), rates, total charges, meter readings.
-- Includes overall confidence and per-field confidence map.

CREATE TABLE pdf_extractor_service.utility_bill_extractions (
    extraction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES pdf_extractor_service.documents(document_id) ON DELETE CASCADE,
    utility_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    overall_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    field_confidences JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL
);

-- Confidence must be between 0 and 1
ALTER TABLE pdf_extractor_service.utility_bill_extractions
    ADD CONSTRAINT chk_utility_confidence_range
    CHECK (overall_confidence >= 0 AND overall_confidence <= 1);

-- Provenance hash must be 64-character hex
ALTER TABLE pdf_extractor_service.utility_bill_extractions
    ADD CONSTRAINT chk_utility_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: pdf_extractor_service.extraction_templates
-- =============================================================================
-- Extraction template definitions. Each template defines a set of field
-- patterns and validation rules for a specific document type. Templates
-- enable configurable, reusable extraction logic without code changes.
-- Unique template name per tenant.

CREATE TABLE pdf_extractor_service.extraction_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    template_type VARCHAR(30) NOT NULL,
    field_patterns JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation_rules JSONB NOT NULL DEFAULT '{}'::jsonb,
    description TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Template type constraint
ALTER TABLE pdf_extractor_service.extraction_templates
    ADD CONSTRAINT chk_template_type
    CHECK (template_type IN (
        'invoice', 'utility_bill', 'shipping_manifest', 'receipt',
        'purchase_order', 'bill_of_lading', 'certificate',
        'report', 'contract', 'form', 'custom'
    ));

-- Template name must not be empty
ALTER TABLE pdf_extractor_service.extraction_templates
    ADD CONSTRAINT chk_template_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Unique template name per tenant
CREATE UNIQUE INDEX uq_template_name_tenant
    ON pdf_extractor_service.extraction_templates (name, tenant_id);

-- =============================================================================
-- Table: pdf_extractor_service.validation_results
-- =============================================================================
-- Validation result records. Each record captures a validation check for
-- an extracted field including severity (error/warning/info), expected vs
-- actual values, rule name, and descriptive message. Used for data quality
-- assessment and confidence scoring.

CREATE TABLE pdf_extractor_service.validation_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES pdf_extractor_service.documents(document_id) ON DELETE CASCADE,
    field_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    expected_value TEXT DEFAULT NULL,
    actual_value TEXT DEFAULT NULL,
    rule_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Severity constraint
ALTER TABLE pdf_extractor_service.validation_results
    ADD CONSTRAINT chk_validation_severity
    CHECK (severity IN ('error', 'warning', 'info'));

-- Field name must not be empty
ALTER TABLE pdf_extractor_service.validation_results
    ADD CONSTRAINT chk_validation_field_name_not_empty
    CHECK (LENGTH(TRIM(field_name)) > 0);

-- Rule name must not be empty
ALTER TABLE pdf_extractor_service.validation_results
    ADD CONSTRAINT chk_validation_rule_name_not_empty
    CHECK (LENGTH(TRIM(rule_name)) > 0);

-- =============================================================================
-- Table: pdf_extractor_service.pdf_audit_log (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all
-- PDF extractor operations. Each event captures the entity being operated
-- on, action, data hashes (current, previous, chain), details (JSONB),
-- user, tenant, and timestamp. Partitioned by created_at for time-series
-- queries. Retained for 365 days with compression after 30 days.

CREATE TABLE pdf_extractor_service.pdf_audit_log (
    audit_id UUID NOT NULL DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    data_hash VARCHAR(64),
    previous_hash VARCHAR(64),
    chain_hash VARCHAR(64),
    user_id VARCHAR(100) DEFAULT 'system',
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    PRIMARY KEY (audit_id, created_at)
);

-- Create hypertable partitioned by created_at
SELECT create_hypertable('pdf_extractor_service.pdf_audit_log', 'created_at', if_not_exists => TRUE);

-- Entity type constraint
ALTER TABLE pdf_extractor_service.pdf_audit_log
    ADD CONSTRAINT chk_pdf_audit_entity_type
    CHECK (entity_type IN (
        'document', 'document_page', 'extraction_job', 'extracted_field',
        'invoice_extraction', 'manifest_extraction', 'utility_bill_extraction',
        'extraction_template', 'validation_result', 'system'
    ));

-- Action constraint
ALTER TABLE pdf_extractor_service.pdf_audit_log
    ADD CONSTRAINT chk_pdf_audit_action
    CHECK (action IN (
        'create', 'update', 'delete', 'extract', 'validate',
        'upload', 'download', 'reprocess', 'template_apply',
        'ocr_complete', 'ocr_failed', 'confidence_low',
        'activate', 'deactivate', 'system', 'admin'
    ));

-- Hash fields must be 64-character hex when present
ALTER TABLE pdf_extractor_service.pdf_audit_log
    ADD CONSTRAINT chk_pdf_audit_data_hash_length
    CHECK (data_hash IS NULL OR LENGTH(data_hash) = 64);

ALTER TABLE pdf_extractor_service.pdf_audit_log
    ADD CONSTRAINT chk_pdf_audit_previous_hash_length
    CHECK (previous_hash IS NULL OR LENGTH(previous_hash) = 64);

ALTER TABLE pdf_extractor_service.pdf_audit_log
    ADD CONSTRAINT chk_pdf_audit_chain_hash_length
    CHECK (chain_hash IS NULL OR LENGTH(chain_hash) = 64);

-- =============================================================================
-- Continuous Aggregate: pdf_extractor_service.hourly_extraction_stats
-- =============================================================================
-- Precomputed hourly extraction job statistics by status and document type
-- for dashboard queries, trend analysis, and SLI tracking. Shows the number
-- of extraction jobs per status, average duration, success/failure counts.

CREATE MATERIALIZED VIEW pdf_extractor_service.hourly_extraction_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', started_at) AS bucket,
    job_status,
    document_type,
    tenant_id,
    COUNT(*) AS job_count,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    SUM(CASE WHEN job_status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
    SUM(CASE WHEN job_status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
    SUM(CASE WHEN job_status = 'timeout' THEN 1 ELSE 0 END) AS timeout_count,
    AVG(fields_extracted) AS avg_fields_extracted,
    AVG(pages_processed) AS avg_pages_processed
FROM pdf_extractor_service.extraction_jobs
WHERE started_at IS NOT NULL
GROUP BY bucket, job_status, document_type, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('pdf_extractor_service.hourly_extraction_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: pdf_extractor_service.hourly_audit_stats
-- =============================================================================
-- Precomputed hourly counts of audit events by entity type and action
-- for compliance reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW pdf_extractor_service.hourly_audit_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    entity_type,
    action,
    tenant_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT entity_id) AS unique_entities,
    COUNT(DISTINCT user_id) AS unique_users
FROM pdf_extractor_service.pdf_audit_log
WHERE created_at IS NOT NULL
GROUP BY bucket, entity_type, action, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('pdf_extractor_service.hourly_audit_stats',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- documents indexes
CREATE INDEX idx_doc_file_name ON pdf_extractor_service.documents(file_name);
CREATE INDEX idx_doc_file_hash ON pdf_extractor_service.documents(file_hash);
CREATE INDEX idx_doc_document_type ON pdf_extractor_service.documents(document_type);
CREATE INDEX idx_doc_document_format ON pdf_extractor_service.documents(document_format);
CREATE INDEX idx_doc_upload_timestamp ON pdf_extractor_service.documents(upload_timestamp DESC);
CREATE INDEX idx_doc_created_at ON pdf_extractor_service.documents(created_at DESC);
CREATE INDEX idx_doc_provenance_hash ON pdf_extractor_service.documents(provenance_hash);
CREATE INDEX idx_doc_tenant ON pdf_extractor_service.documents(tenant_id);
CREATE INDEX idx_doc_tenant_type ON pdf_extractor_service.documents(tenant_id, document_type);
CREATE INDEX idx_doc_tenant_format ON pdf_extractor_service.documents(tenant_id, document_format);
CREATE INDEX idx_doc_tenant_uploaded ON pdf_extractor_service.documents(tenant_id, upload_timestamp DESC);

-- document_pages indexes
CREATE INDEX idx_dp_document ON pdf_extractor_service.document_pages(document_id);
CREATE INDEX idx_dp_page_number ON pdf_extractor_service.document_pages(page_number);
CREATE INDEX idx_dp_confidence ON pdf_extractor_service.document_pages(confidence);
CREATE INDEX idx_dp_ocr_engine ON pdf_extractor_service.document_pages(ocr_engine_used);
CREATE INDEX idx_dp_provenance_hash ON pdf_extractor_service.document_pages(provenance_hash);
CREATE INDEX idx_dp_created_at ON pdf_extractor_service.document_pages(created_at DESC);
CREATE INDEX idx_dp_tenant ON pdf_extractor_service.document_pages(tenant_id);
CREATE INDEX idx_dp_tenant_document ON pdf_extractor_service.document_pages(tenant_id, document_id);
CREATE INDEX idx_dp_bounding_boxes ON pdf_extractor_service.document_pages USING GIN (bounding_boxes);

-- extraction_jobs indexes (hypertable-aware)
CREATE INDEX idx_ej_document ON pdf_extractor_service.extraction_jobs(document_id, started_at DESC);
CREATE INDEX idx_ej_status ON pdf_extractor_service.extraction_jobs(job_status, started_at DESC);
CREATE INDEX idx_ej_document_type ON pdf_extractor_service.extraction_jobs(document_type, started_at DESC);
CREATE INDEX idx_ej_ocr_engine ON pdf_extractor_service.extraction_jobs(ocr_engine, started_at DESC);
CREATE INDEX idx_ej_tenant ON pdf_extractor_service.extraction_jobs(tenant_id, started_at DESC);
CREATE INDEX idx_ej_tenant_status ON pdf_extractor_service.extraction_jobs(tenant_id, job_status, started_at DESC);
CREATE INDEX idx_ej_tenant_type ON pdf_extractor_service.extraction_jobs(tenant_id, document_type, started_at DESC);
CREATE INDEX idx_ej_metadata ON pdf_extractor_service.extraction_jobs USING GIN (metadata);

-- extracted_fields indexes
CREATE INDEX idx_ef_document ON pdf_extractor_service.extracted_fields(document_id);
CREATE INDEX idx_ef_job ON pdf_extractor_service.extracted_fields(job_id);
CREATE INDEX idx_ef_field_name ON pdf_extractor_service.extracted_fields(field_name);
CREATE INDEX idx_ef_field_type ON pdf_extractor_service.extracted_fields(field_type);
CREATE INDEX idx_ef_confidence ON pdf_extractor_service.extracted_fields(confidence);
CREATE INDEX idx_ef_extraction_method ON pdf_extractor_service.extracted_fields(extraction_method);
CREATE INDEX idx_ef_validated ON pdf_extractor_service.extracted_fields(validated);
CREATE INDEX idx_ef_provenance_hash ON pdf_extractor_service.extracted_fields(provenance_hash);
CREATE INDEX idx_ef_created_at ON pdf_extractor_service.extracted_fields(created_at DESC);
CREATE INDEX idx_ef_tenant ON pdf_extractor_service.extracted_fields(tenant_id);
CREATE INDEX idx_ef_tenant_document ON pdf_extractor_service.extracted_fields(tenant_id, document_id);
CREATE INDEX idx_ef_tenant_field ON pdf_extractor_service.extracted_fields(tenant_id, field_name);
CREATE INDEX idx_ef_bounding_box ON pdf_extractor_service.extracted_fields USING GIN (bounding_box);

-- invoice_extractions indexes
CREATE INDEX idx_ie_document ON pdf_extractor_service.invoice_extractions(document_id);
CREATE INDEX idx_ie_confidence ON pdf_extractor_service.invoice_extractions(overall_confidence);
CREATE INDEX idx_ie_validation ON pdf_extractor_service.invoice_extractions(validation_passed);
CREATE INDEX idx_ie_provenance_hash ON pdf_extractor_service.invoice_extractions(provenance_hash);
CREATE INDEX idx_ie_created_at ON pdf_extractor_service.invoice_extractions(created_at DESC);
CREATE INDEX idx_ie_tenant ON pdf_extractor_service.invoice_extractions(tenant_id);
CREATE INDEX idx_ie_tenant_document ON pdf_extractor_service.invoice_extractions(tenant_id, document_id);
CREATE INDEX idx_ie_invoice_data ON pdf_extractor_service.invoice_extractions USING GIN (invoice_data);
CREATE INDEX idx_ie_field_confidences ON pdf_extractor_service.invoice_extractions USING GIN (field_confidences);
CREATE INDEX idx_ie_validation_errors ON pdf_extractor_service.invoice_extractions USING GIN (validation_errors);

-- manifest_extractions indexes
CREATE INDEX idx_me_document ON pdf_extractor_service.manifest_extractions(document_id);
CREATE INDEX idx_me_confidence ON pdf_extractor_service.manifest_extractions(overall_confidence);
CREATE INDEX idx_me_validation ON pdf_extractor_service.manifest_extractions(validation_passed);
CREATE INDEX idx_me_provenance_hash ON pdf_extractor_service.manifest_extractions(provenance_hash);
CREATE INDEX idx_me_created_at ON pdf_extractor_service.manifest_extractions(created_at DESC);
CREATE INDEX idx_me_tenant ON pdf_extractor_service.manifest_extractions(tenant_id);
CREATE INDEX idx_me_tenant_document ON pdf_extractor_service.manifest_extractions(tenant_id, document_id);
CREATE INDEX idx_me_manifest_data ON pdf_extractor_service.manifest_extractions USING GIN (manifest_data);
CREATE INDEX idx_me_field_confidences ON pdf_extractor_service.manifest_extractions USING GIN (field_confidences);

-- utility_bill_extractions indexes
CREATE INDEX idx_ube_document ON pdf_extractor_service.utility_bill_extractions(document_id);
CREATE INDEX idx_ube_confidence ON pdf_extractor_service.utility_bill_extractions(overall_confidence);
CREATE INDEX idx_ube_provenance_hash ON pdf_extractor_service.utility_bill_extractions(provenance_hash);
CREATE INDEX idx_ube_created_at ON pdf_extractor_service.utility_bill_extractions(created_at DESC);
CREATE INDEX idx_ube_tenant ON pdf_extractor_service.utility_bill_extractions(tenant_id);
CREATE INDEX idx_ube_tenant_document ON pdf_extractor_service.utility_bill_extractions(tenant_id, document_id);
CREATE INDEX idx_ube_utility_data ON pdf_extractor_service.utility_bill_extractions USING GIN (utility_data);
CREATE INDEX idx_ube_field_confidences ON pdf_extractor_service.utility_bill_extractions USING GIN (field_confidences);

-- extraction_templates indexes
CREATE INDEX idx_et_name ON pdf_extractor_service.extraction_templates(name);
CREATE INDEX idx_et_template_type ON pdf_extractor_service.extraction_templates(template_type);
CREATE INDEX idx_et_created_at ON pdf_extractor_service.extraction_templates(created_at DESC);
CREATE INDEX idx_et_updated_at ON pdf_extractor_service.extraction_templates(updated_at DESC);
CREATE INDEX idx_et_tenant ON pdf_extractor_service.extraction_templates(tenant_id);
CREATE INDEX idx_et_tenant_type ON pdf_extractor_service.extraction_templates(tenant_id, template_type);
CREATE INDEX idx_et_field_patterns ON pdf_extractor_service.extraction_templates USING GIN (field_patterns);
CREATE INDEX idx_et_validation_rules ON pdf_extractor_service.extraction_templates USING GIN (validation_rules);

-- validation_results indexes
CREATE INDEX idx_vr_document ON pdf_extractor_service.validation_results(document_id);
CREATE INDEX idx_vr_field_name ON pdf_extractor_service.validation_results(field_name);
CREATE INDEX idx_vr_severity ON pdf_extractor_service.validation_results(severity);
CREATE INDEX idx_vr_rule_name ON pdf_extractor_service.validation_results(rule_name);
CREATE INDEX idx_vr_created_at ON pdf_extractor_service.validation_results(created_at DESC);
CREATE INDEX idx_vr_tenant ON pdf_extractor_service.validation_results(tenant_id);
CREATE INDEX idx_vr_tenant_document ON pdf_extractor_service.validation_results(tenant_id, document_id);
CREATE INDEX idx_vr_tenant_severity ON pdf_extractor_service.validation_results(tenant_id, severity);

-- pdf_audit_log indexes (hypertable-aware)
CREATE INDEX idx_pal_entity_type ON pdf_extractor_service.pdf_audit_log(entity_type, created_at DESC);
CREATE INDEX idx_pal_entity_id ON pdf_extractor_service.pdf_audit_log(entity_id, created_at DESC);
CREATE INDEX idx_pal_action ON pdf_extractor_service.pdf_audit_log(action, created_at DESC);
CREATE INDEX idx_pal_user ON pdf_extractor_service.pdf_audit_log(user_id, created_at DESC);
CREATE INDEX idx_pal_tenant ON pdf_extractor_service.pdf_audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_pal_data_hash ON pdf_extractor_service.pdf_audit_log(data_hash);
CREATE INDEX idx_pal_chain_hash ON pdf_extractor_service.pdf_audit_log(chain_hash);
CREATE INDEX idx_pal_details ON pdf_extractor_service.pdf_audit_log USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE pdf_extractor_service.documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY doc_tenant_read ON pdf_extractor_service.documents
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY doc_tenant_write ON pdf_extractor_service.documents
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.document_pages ENABLE ROW LEVEL SECURITY;
CREATE POLICY dp_tenant_read ON pdf_extractor_service.document_pages
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dp_tenant_write ON pdf_extractor_service.document_pages
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.extraction_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY ej_tenant_read ON pdf_extractor_service.extraction_jobs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ej_tenant_write ON pdf_extractor_service.extraction_jobs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.extracted_fields ENABLE ROW LEVEL SECURITY;
CREATE POLICY ef_tenant_read ON pdf_extractor_service.extracted_fields
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ef_tenant_write ON pdf_extractor_service.extracted_fields
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.invoice_extractions ENABLE ROW LEVEL SECURITY;
CREATE POLICY ie_tenant_read ON pdf_extractor_service.invoice_extractions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ie_tenant_write ON pdf_extractor_service.invoice_extractions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.manifest_extractions ENABLE ROW LEVEL SECURITY;
CREATE POLICY me_tenant_read ON pdf_extractor_service.manifest_extractions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY me_tenant_write ON pdf_extractor_service.manifest_extractions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.utility_bill_extractions ENABLE ROW LEVEL SECURITY;
CREATE POLICY ube_tenant_read ON pdf_extractor_service.utility_bill_extractions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ube_tenant_write ON pdf_extractor_service.utility_bill_extractions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.extraction_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY et_tenant_read ON pdf_extractor_service.extraction_templates
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY et_tenant_write ON pdf_extractor_service.extraction_templates
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.validation_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY vr_tenant_read ON pdf_extractor_service.validation_results
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY vr_tenant_write ON pdf_extractor_service.validation_results
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE pdf_extractor_service.pdf_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY pal_tenant_read ON pdf_extractor_service.pdf_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY pal_tenant_write ON pdf_extractor_service.pdf_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA pdf_extractor_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA pdf_extractor_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA pdf_extractor_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON pdf_extractor_service.hourly_extraction_stats TO greenlang_app;
GRANT SELECT ON pdf_extractor_service.hourly_audit_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA pdf_extractor_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA pdf_extractor_service TO greenlang_readonly;
GRANT SELECT ON pdf_extractor_service.hourly_extraction_stats TO greenlang_readonly;
GRANT SELECT ON pdf_extractor_service.hourly_audit_stats TO greenlang_readonly;

-- Add PDF extractor service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'pdf_extractor:documents:read', 'pdf_extractor', 'documents_read', 'View uploaded documents and metadata'),
    (gen_random_uuid(), 'pdf_extractor:documents:write', 'pdf_extractor', 'documents_write', 'Upload and manage documents'),
    (gen_random_uuid(), 'pdf_extractor:pages:read', 'pdf_extractor', 'pages_read', 'View extracted page content and OCR results'),
    (gen_random_uuid(), 'pdf_extractor:pages:write', 'pdf_extractor', 'pages_write', 'Create and update page extraction records'),
    (gen_random_uuid(), 'pdf_extractor:jobs:read', 'pdf_extractor', 'jobs_read', 'View extraction job status and results'),
    (gen_random_uuid(), 'pdf_extractor:jobs:write', 'pdf_extractor', 'jobs_write', 'Create and manage extraction jobs'),
    (gen_random_uuid(), 'pdf_extractor:fields:read', 'pdf_extractor', 'fields_read', 'View extracted field values and confidence scores'),
    (gen_random_uuid(), 'pdf_extractor:fields:write', 'pdf_extractor', 'fields_write', 'Create and validate extracted fields'),
    (gen_random_uuid(), 'pdf_extractor:extractions:read', 'pdf_extractor', 'extractions_read', 'View structured extraction results (invoices, manifests, utility bills)'),
    (gen_random_uuid(), 'pdf_extractor:extractions:write', 'pdf_extractor', 'extractions_write', 'Create structured extraction results'),
    (gen_random_uuid(), 'pdf_extractor:templates:read', 'pdf_extractor', 'templates_read', 'View extraction template definitions'),
    (gen_random_uuid(), 'pdf_extractor:templates:write', 'pdf_extractor', 'templates_write', 'Create and manage extraction templates'),
    (gen_random_uuid(), 'pdf_extractor:audit:read', 'pdf_extractor', 'audit_read', 'View PDF extractor audit event log'),
    (gen_random_uuid(), 'pdf_extractor:admin', 'pdf_extractor', 'admin', 'PDF extractor service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep extraction jobs for 30 days
SELECT add_retention_policy('pdf_extractor_service.extraction_jobs', INTERVAL '30 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('pdf_extractor_service.pdf_audit_log', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on extraction_jobs after 3 days
ALTER TABLE pdf_extractor_service.extraction_jobs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'started_at DESC'
);

SELECT add_compression_policy('pdf_extractor_service.extraction_jobs', INTERVAL '3 days');

-- Enable compression on pdf_audit_log after 30 days
ALTER TABLE pdf_extractor_service.pdf_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('pdf_extractor_service.pdf_audit_log', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the PDF Extractor Agent (GL-DATA-X-001) in Agent Registry
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-001', 'PDF & Invoice Extractor',
 'Extracts structured data from PDF documents, invoices, utility bills, and shipping manifests using multi-engine OCR (Tesseract, AWS Textract, Azure Form Recognizer, Google Document AI). Provides field-level extraction with confidence scoring, template-based extraction, cross-field validation, and SHA-256 provenance hash chains for every extracted value.',
 2, 'async', true, true, 10, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/pdf-extractor', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for PDF Extractor
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-001', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "1Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/pdf-extractor-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"data", "extraction", "ocr", "pdf", "invoice", "utility-bill", "manifest"}',
 '{"cross-sector"}',
 'c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for PDF Extractor
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-X-001', '1.0.0', 'document_upload', 'ingestion',
 'Upload and register PDF, PNG, JPG, TIFF, or BMP documents for extraction processing',
 '{"file_content", "file_name", "document_type"}', '{"document_id", "file_hash", "page_count"}',
 '{"max_file_size_mb": 100, "supported_formats": ["pdf", "png", "jpg", "tiff", "bmp"]}'::jsonb),

('GL-DATA-X-001', '1.0.0', 'ocr_extraction', 'computation',
 'Extract raw text from document pages using configurable OCR engines with confidence scoring and bounding box detection',
 '{"document_id", "ocr_engine"}', '{"pages", "word_count", "confidence"}',
 '{"engines": ["tesseract", "textract", "azure_form_recognizer", "google_document_ai", "easyocr", "paddleocr"], "confidence_threshold": 0.7}'::jsonb),

('GL-DATA-X-001', '1.0.0', 'field_extraction', 'computation',
 'Extract structured fields from document text using template matching, regex patterns, ML models, and rule-based methods',
 '{"document_id", "template_id"}', '{"extracted_fields", "field_confidences"}',
 '{"extraction_methods": ["ocr", "native_text", "regex", "template_match", "ml_model", "rule_based", "hybrid"], "validate_fields": true}'::jsonb),

('GL-DATA-X-001', '1.0.0', 'invoice_extraction', 'computation',
 'Extract structured invoice data including vendor, line items, totals, dates, PO numbers, and tax amounts with cross-field validation',
 '{"document_id"}', '{"invoice_data", "overall_confidence", "validation_results"}',
 '{"required_fields": ["vendor_name", "invoice_number", "invoice_date", "total_amount"], "currency_detection": true}'::jsonb),

('GL-DATA-X-001', '1.0.0', 'manifest_extraction', 'computation',
 'Extract structured shipping manifest data including shipper, consignee, cargo details, weight, and route information',
 '{"document_id"}', '{"manifest_data", "overall_confidence", "validation_results"}',
 '{"required_fields": ["shipper", "consignee", "cargo_description", "weight"], "hazmat_detection": true}'::jsonb),

('GL-DATA-X-001', '1.0.0', 'utility_bill_extraction', 'computation',
 'Extract structured utility bill data including provider, account, consumption (kWh, therms, gallons), rates, and meter readings for emission calculations',
 '{"document_id"}', '{"utility_data", "overall_confidence", "consumption_data"}',
 '{"required_fields": ["provider", "account_number", "service_period", "total_consumption"], "unit_detection": true}'::jsonb),

('GL-DATA-X-001', '1.0.0', 'template_management', 'configuration',
 'Create, update, and apply extraction templates with field patterns and validation rules for repeatable extraction',
 '{"template_definition"}', '{"template_id", "validation_result"}',
 '{"template_types": ["invoice", "utility_bill", "shipping_manifest", "receipt", "custom"]}'::jsonb),

('GL-DATA-X-001', '1.0.0', 'field_validation', 'validation',
 'Validate extracted fields against configurable rules including type checks, range validation, cross-field consistency, and format verification',
 '{"document_id", "extracted_fields"}', '{"validation_results", "severity_counts"}',
 '{"severity_levels": ["error", "warning", "info"], "cross_field_rules": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for PDF Extractor
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- PDF Extractor depends on Schema Compiler for input/output validation
('GL-DATA-X-001', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Extraction results and document metadata are validated against JSON Schema definitions'),

-- PDF Extractor depends on Unit Normalizer for unit conversion
('GL-DATA-X-001', 'GL-FOUND-X-003', '>=1.0.0', false,
 'Utility bill consumption values (kWh, therms, gallons) are normalized to standard units for emission calculations'),

-- PDF Extractor depends on Registry for agent discovery
('GL-DATA-X-001', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for extraction pipeline orchestration'),

-- PDF Extractor optionally uses Citations for provenance tracking
('GL-DATA-X-001', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Extracted data provenance chains are registered with the citation service for audit trail'),

-- PDF Extractor optionally uses Reproducibility for determinism
('GL-DATA-X-001', 'GL-FOUND-X-008', '>=1.0.0', true,
 'OCR extraction results are verified for reproducibility across re-processing runs')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for PDF Extractor
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-001', 'PDF & Invoice Extractor',
 'Multi-engine OCR extraction for invoices, utility bills, and shipping manifests. Template-based field extraction with confidence scoring, cross-field validation, and SHA-256 provenance hash chains for every extracted value.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA pdf_extractor_service IS 'PDF & Invoice Extractor for GreenLang Climate OS (AGENT-DATA-001) - multi-engine OCR, structured extraction for invoices/manifests/utility bills, template management, field validation, and provenance tracking';
COMMENT ON TABLE pdf_extractor_service.documents IS 'Document registry with file metadata, SHA-256 hash, document type classification, format, page count, and provenance hash';
COMMENT ON TABLE pdf_extractor_service.document_pages IS 'Individual page records with raw OCR text, word count, confidence score, engine used, bounding boxes, and provenance hash';
COMMENT ON TABLE pdf_extractor_service.extraction_jobs IS 'TimescaleDB hypertable: extraction job execution records with status, OCR engine, confidence threshold, duration, fields extracted, and pages processed';
COMMENT ON TABLE pdf_extractor_service.extracted_fields IS 'Individual extracted field records with name, value, confidence, type, bounding box, extraction method, validation status, and provenance hash';
COMMENT ON TABLE pdf_extractor_service.invoice_extractions IS 'Structured invoice extraction results with parsed data (JSONB), overall confidence, per-field confidences, validation status, and errors';
COMMENT ON TABLE pdf_extractor_service.manifest_extractions IS 'Structured shipping manifest extraction results with parsed data (JSONB), overall confidence, per-field confidences, and validation status';
COMMENT ON TABLE pdf_extractor_service.utility_bill_extractions IS 'Structured utility bill extraction results with parsed data (JSONB), overall confidence, and per-field confidences for emission calculations';
COMMENT ON TABLE pdf_extractor_service.extraction_templates IS 'Extraction template definitions with field patterns, validation rules, and document type classification for repeatable extraction';
COMMENT ON TABLE pdf_extractor_service.validation_results IS 'Validation result records with severity (error/warning/info), expected vs actual values, rule name, and descriptive message';
COMMENT ON TABLE pdf_extractor_service.pdf_audit_log IS 'TimescaleDB hypertable: comprehensive audit events for all PDF extractor operations with hash chain integrity';
COMMENT ON MATERIALIZED VIEW pdf_extractor_service.hourly_extraction_stats IS 'Continuous aggregate: hourly extraction job statistics by status and document type for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW pdf_extractor_service.hourly_audit_stats IS 'Continuous aggregate: hourly audit event counts by entity type and action for compliance reporting';

COMMENT ON COLUMN pdf_extractor_service.documents.document_type IS 'Document type: invoice, utility_bill, shipping_manifest, receipt, purchase_order, bill_of_lading, certificate, report, contract, form, other';
COMMENT ON COLUMN pdf_extractor_service.documents.document_format IS 'Document format: pdf, png, jpg, tiff, bmp';
COMMENT ON COLUMN pdf_extractor_service.documents.file_hash IS 'SHA-256 hash of the uploaded file for integrity verification and deduplication';
COMMENT ON COLUMN pdf_extractor_service.documents.provenance_hash IS 'SHA-256 hash of document metadata for provenance chain';
COMMENT ON COLUMN pdf_extractor_service.document_pages.ocr_engine_used IS 'OCR engine: tesseract, textract, azure_form_recognizer, google_document_ai, easyocr, paddleocr, native_pdf';
COMMENT ON COLUMN pdf_extractor_service.document_pages.confidence IS 'OCR confidence score (0.0 to 1.0) for the page extraction';
COMMENT ON COLUMN pdf_extractor_service.extraction_jobs.job_status IS 'Job status: pending, running, completed, failed, cancelled, timeout';
COMMENT ON COLUMN pdf_extractor_service.extracted_fields.field_type IS 'Field type: text, number, currency, date, address, email, phone, percentage, quantity, unit, identifier, boolean, table_cell, other';
COMMENT ON COLUMN pdf_extractor_service.extracted_fields.extraction_method IS 'Extraction method: ocr, native_text, regex, template_match, ml_model, rule_based, hybrid, manual';
COMMENT ON COLUMN pdf_extractor_service.validation_results.severity IS 'Validation severity: error, warning, info';
COMMENT ON COLUMN pdf_extractor_service.pdf_audit_log.entity_type IS 'Entity type: document, document_page, extraction_job, extracted_field, invoice_extraction, manifest_extraction, utility_bill_extraction, extraction_template, validation_result, system';
COMMENT ON COLUMN pdf_extractor_service.pdf_audit_log.chain_hash IS 'SHA-256 hash chain linking this event to the previous event for tamper detection';
