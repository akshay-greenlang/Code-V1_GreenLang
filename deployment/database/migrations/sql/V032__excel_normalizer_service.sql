-- =============================================================================
-- GreenLang Climate OS - Excel & CSV Normalizer Service Schema
-- =============================================================================
-- Migration: V032
-- Component: AGENT-DATA-002 Excel & CSV Normalizer
-- Description: Creates excel_normalizer_service schema with spreadsheet_files,
--              sheet_metadata, column_mappings, normalization_jobs (hypertable),
--              normalized_records, mapping_templates, data_quality_reports,
--              validation_results, transform_operations, excel_audit_log
--              (hypertable), continuous aggregates for hourly normalization stats
--              and hourly audit stats, 50+ indexes (including GIN indexes on
--              JSONB), RLS policies per tenant, 14 security permissions,
--              retention policies (30-day normalization_jobs, 365-day
--              audit_log), compression, and seed data registering the Excel
--              Normalizer Agent (GL-DATA-X-016) with capabilities in the agent
--              registry.
-- Previous: V031__pdf_extractor_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS excel_normalizer_service;

-- =============================================================================
-- Table: excel_normalizer_service.spreadsheet_files
-- =============================================================================
-- File registry. Each spreadsheet record captures file metadata including name,
-- path, format (xlsx, xls, csv, tsv, ods), size, SHA-256 hash, sheet count,
-- total rows, total columns, upload timestamp, tenant scope, provenance hash
-- for integrity verification, and the uploading user. Tenant-scoped.

CREATE TABLE excel_normalizer_service.spreadsheet_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name VARCHAR(500) NOT NULL,
    file_path TEXT NOT NULL,
    file_format VARCHAR(10) NOT NULL,
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    file_hash VARCHAR(64) NOT NULL,
    sheet_count INTEGER NOT NULL DEFAULT 1,
    total_rows INTEGER NOT NULL DEFAULT 0,
    total_columns INTEGER NOT NULL DEFAULT 0,
    upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    uploaded_by VARCHAR(255) DEFAULT 'system',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- File format constraint
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_file_format
    CHECK (file_format IN ('xlsx', 'xls', 'csv', 'tsv', 'ods'));

-- File name must not be empty
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_file_name_not_empty
    CHECK (LENGTH(TRIM(file_name)) > 0);

-- File path must not be empty
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_file_path_not_empty
    CHECK (LENGTH(TRIM(file_path)) > 0);

-- File size must be non-negative
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_file_size_non_negative
    CHECK (file_size_bytes >= 0);

-- File hash must be 64-character hex (SHA-256)
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_file_hash_length
    CHECK (LENGTH(file_hash) = 64);

-- Provenance hash must be 64-character hex
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_file_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Sheet count must be positive
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_sheet_count_positive
    CHECK (sheet_count >= 1);

-- Total rows must be non-negative
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_total_rows_non_negative
    CHECK (total_rows >= 0);

-- Total columns must be non-negative
ALTER TABLE excel_normalizer_service.spreadsheet_files
    ADD CONSTRAINT chk_total_columns_non_negative
    CHECK (total_columns >= 0);

-- =============================================================================
-- Table: excel_normalizer_service.sheet_metadata
-- =============================================================================
-- Per-sheet metadata. Each sheet record captures the sheet name, index within
-- the workbook, row count, column count, header row index, header detection
-- flag, detected encoding (for CSV/TSV), detected delimiter (for CSV/TSV),
-- and provenance link to the parent file. Tenant-scoped.

CREATE TABLE excel_normalizer_service.sheet_metadata (
    sheet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES excel_normalizer_service.spreadsheet_files(file_id) ON DELETE CASCADE,
    sheet_name VARCHAR(255) NOT NULL,
    sheet_index INTEGER NOT NULL DEFAULT 0,
    row_count INTEGER NOT NULL DEFAULT 0,
    column_count INTEGER NOT NULL DEFAULT 0,
    header_row_index INTEGER DEFAULT 0,
    has_headers BOOLEAN NOT NULL DEFAULT true,
    detected_encoding VARCHAR(30) DEFAULT 'utf-8',
    detected_delimiter VARCHAR(5) DEFAULT ',',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Sheet index must be non-negative
ALTER TABLE excel_normalizer_service.sheet_metadata
    ADD CONSTRAINT chk_sheet_index_non_negative
    CHECK (sheet_index >= 0);

-- Row count must be non-negative
ALTER TABLE excel_normalizer_service.sheet_metadata
    ADD CONSTRAINT chk_sheet_row_count_non_negative
    CHECK (row_count >= 0);

-- Column count must be non-negative
ALTER TABLE excel_normalizer_service.sheet_metadata
    ADD CONSTRAINT chk_sheet_column_count_non_negative
    CHECK (column_count >= 0);

-- Sheet name must not be empty
ALTER TABLE excel_normalizer_service.sheet_metadata
    ADD CONSTRAINT chk_sheet_name_not_empty
    CHECK (LENGTH(TRIM(sheet_name)) > 0);

-- Detected encoding constraint
ALTER TABLE excel_normalizer_service.sheet_metadata
    ADD CONSTRAINT chk_sheet_encoding
    CHECK (detected_encoding IN (
        'utf-8', 'utf-16', 'utf-32', 'ascii', 'iso-8859-1',
        'iso-8859-15', 'windows-1252', 'shift_jis', 'euc-jp',
        'gb2312', 'gbk', 'big5', 'euc-kr', 'latin-1'
    ));

-- Unique sheet index per file
ALTER TABLE excel_normalizer_service.sheet_metadata
    ADD CONSTRAINT uq_file_sheet_index
    UNIQUE (file_id, sheet_index);

-- =============================================================================
-- Table: excel_normalizer_service.column_mappings
-- =============================================================================
-- Header-to-canonical field mappings. Each mapping record captures the source
-- column name, column index, canonical (normalized) field name, mapping
-- strategy used (exact, synonym, fuzzy, ml, manual), confidence score,
-- detected data type, detected unit, matched synonyms (JSONB), and tenant
-- scope. Used for repeatable normalization across similar spreadsheets.

CREATE TABLE excel_normalizer_service.column_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sheet_id UUID NOT NULL REFERENCES excel_normalizer_service.sheet_metadata(sheet_id) ON DELETE CASCADE,
    source_column VARCHAR(255) NOT NULL,
    source_index INTEGER NOT NULL DEFAULT 0,
    canonical_field VARCHAR(255) NOT NULL,
    mapping_strategy VARCHAR(30) NOT NULL DEFAULT 'exact',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    detected_data_type VARCHAR(30) NOT NULL DEFAULT 'string',
    detected_unit VARCHAR(50) DEFAULT NULL,
    synonyms_matched JSONB DEFAULT '[]'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Mapping strategy constraint
ALTER TABLE excel_normalizer_service.column_mappings
    ADD CONSTRAINT chk_mapping_strategy
    CHECK (mapping_strategy IN (
        'exact', 'synonym', 'fuzzy', 'ml', 'manual', 'template', 'regex'
    ));

-- Confidence must be between 0 and 1
ALTER TABLE excel_normalizer_service.column_mappings
    ADD CONSTRAINT chk_mapping_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Detected data type constraint
ALTER TABLE excel_normalizer_service.column_mappings
    ADD CONSTRAINT chk_mapping_data_type
    CHECK (detected_data_type IN (
        'string', 'integer', 'float', 'decimal', 'boolean', 'date',
        'datetime', 'time', 'currency', 'percentage', 'email',
        'phone', 'url', 'identifier', 'enumeration', 'other'
    ));

-- Source column must not be empty
ALTER TABLE excel_normalizer_service.column_mappings
    ADD CONSTRAINT chk_mapping_source_column_not_empty
    CHECK (LENGTH(TRIM(source_column)) > 0);

-- Canonical field must not be empty
ALTER TABLE excel_normalizer_service.column_mappings
    ADD CONSTRAINT chk_mapping_canonical_field_not_empty
    CHECK (LENGTH(TRIM(canonical_field)) > 0);

-- Source index must be non-negative
ALTER TABLE excel_normalizer_service.column_mappings
    ADD CONSTRAINT chk_mapping_source_index_non_negative
    CHECK (source_index >= 0);

-- =============================================================================
-- Table: excel_normalizer_service.normalization_jobs (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording normalization job executions. Each job
-- captures the file being processed, status, configuration (JSONB), row
-- processing counts (processed, normalized, skipped), errors (JSONB),
-- start/completion times, duration, provenance hash, and tenant scope.
-- Partitioned by started_at for time-series queries. Retained for 30 days
-- with compression after 3 days.

CREATE TABLE excel_normalizer_service.normalization_jobs (
    job_id UUID NOT NULL DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    rows_processed INTEGER NOT NULL DEFAULT 0,
    rows_normalized INTEGER NOT NULL DEFAULT 0,
    rows_skipped INTEGER NOT NULL DEFAULT 0,
    errors JSONB DEFAULT '[]'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, started_at)
);

-- Create hypertable partitioned by started_at
SELECT create_hypertable('excel_normalizer_service.normalization_jobs', 'started_at', if_not_exists => TRUE);

-- Job status constraint
ALTER TABLE excel_normalizer_service.normalization_jobs
    ADD CONSTRAINT chk_norm_job_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout'));

-- Duration must be non-negative
ALTER TABLE excel_normalizer_service.normalization_jobs
    ADD CONSTRAINT chk_norm_job_duration_non_negative
    CHECK (duration_seconds >= 0);

-- Rows processed must be non-negative
ALTER TABLE excel_normalizer_service.normalization_jobs
    ADD CONSTRAINT chk_norm_job_rows_processed_non_negative
    CHECK (rows_processed >= 0);

-- Rows normalized must be non-negative
ALTER TABLE excel_normalizer_service.normalization_jobs
    ADD CONSTRAINT chk_norm_job_rows_normalized_non_negative
    CHECK (rows_normalized >= 0);

-- Rows skipped must be non-negative
ALTER TABLE excel_normalizer_service.normalization_jobs
    ADD CONSTRAINT chk_norm_job_rows_skipped_non_negative
    CHECK (rows_skipped >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE excel_normalizer_service.normalization_jobs
    ADD CONSTRAINT chk_norm_job_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: excel_normalizer_service.normalized_records
-- =============================================================================
-- Output records from normalization. Each record captures the original row
-- values (JSONB), normalized values (JSONB), row index within the sheet,
-- quality score (0.0 to 1.0), validation errors (JSONB), and provenance hash
-- for integrity verification. Used for downstream emission calculations.

CREATE TABLE excel_normalizer_service.normalized_records (
    record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    row_index INTEGER NOT NULL DEFAULT 0,
    original_values JSONB NOT NULL DEFAULT '{}'::jsonb,
    normalized_values JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    validation_errors JSONB DEFAULT '[]'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Quality score must be between 0 and 1
ALTER TABLE excel_normalizer_service.normalized_records
    ADD CONSTRAINT chk_record_quality_score_range
    CHECK (quality_score >= 0 AND quality_score <= 1);

-- Row index must be non-negative
ALTER TABLE excel_normalizer_service.normalized_records
    ADD CONSTRAINT chk_record_row_index_non_negative
    CHECK (row_index >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE excel_normalizer_service.normalized_records
    ADD CONSTRAINT chk_record_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: excel_normalizer_service.mapping_templates
-- =============================================================================
-- Reusable column mapping templates. Each template defines a set of column
-- mappings for a specific source type (e.g., utility_bill, emissions_report,
-- fuel_log), enabling consistent normalization across similar spreadsheets.
-- Unique template name. Includes usage count for popularity tracking.

CREATE TABLE excel_normalizer_service.mapping_templates (
    template_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    source_type VARCHAR(50) NOT NULL DEFAULT 'generic',
    column_mappings JSONB NOT NULL DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    usage_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source type constraint
ALTER TABLE excel_normalizer_service.mapping_templates
    ADD CONSTRAINT chk_template_source_type
    CHECK (source_type IN (
        'generic', 'utility_bill', 'emissions_report', 'fuel_log',
        'travel_log', 'waste_manifest', 'energy_audit', 'fleet_data',
        'procurement', 'supply_chain', 'refrigerant_log', 'custom'
    ));

-- Template name must not be empty
ALTER TABLE excel_normalizer_service.mapping_templates
    ADD CONSTRAINT chk_template_name_not_empty
    CHECK (LENGTH(TRIM(template_name)) > 0);

-- Usage count must be non-negative
ALTER TABLE excel_normalizer_service.mapping_templates
    ADD CONSTRAINT chk_template_usage_count_non_negative
    CHECK (usage_count >= 0);

-- Unique template name per tenant
CREATE UNIQUE INDEX uq_mapping_template_name_tenant
    ON excel_normalizer_service.mapping_templates (template_name, tenant_id);

-- =============================================================================
-- Table: excel_normalizer_service.data_quality_reports
-- =============================================================================
-- Data quality assessment reports. Each report captures overall quality score,
-- completeness/accuracy/consistency sub-scores, quality level classification,
-- row-level counts (total, valid, invalid), null/type-mismatch/duplicate
-- counts, per-column quality scores (JSONB), and identified issues (JSONB).

CREATE TABLE excel_normalizer_service.data_quality_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES excel_normalizer_service.spreadsheet_files(file_id) ON DELETE CASCADE,
    overall_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    completeness_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    accuracy_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    consistency_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    quality_level VARCHAR(20) NOT NULL DEFAULT 'unknown',
    total_rows INTEGER NOT NULL DEFAULT 0,
    valid_rows INTEGER NOT NULL DEFAULT 0,
    invalid_rows INTEGER NOT NULL DEFAULT 0,
    null_count INTEGER NOT NULL DEFAULT 0,
    type_mismatch_count INTEGER NOT NULL DEFAULT 0,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    column_scores JSONB DEFAULT '{}'::jsonb,
    issues JSONB DEFAULT '[]'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Score must be between 0 and 1
ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_overall_score_range
    CHECK (overall_score >= 0 AND overall_score <= 1);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_completeness_score_range
    CHECK (completeness_score >= 0 AND completeness_score <= 1);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_accuracy_score_range
    CHECK (accuracy_score >= 0 AND accuracy_score <= 1);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_consistency_score_range
    CHECK (consistency_score >= 0 AND consistency_score <= 1);

-- Quality level constraint
ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_quality_level
    CHECK (quality_level IN (
        'excellent', 'good', 'acceptable', 'poor', 'critical', 'unknown'
    ));

-- Row counts must be non-negative
ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_total_rows_non_negative
    CHECK (total_rows >= 0);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_valid_rows_non_negative
    CHECK (valid_rows >= 0);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_invalid_rows_non_negative
    CHECK (invalid_rows >= 0);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_null_count_non_negative
    CHECK (null_count >= 0);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_type_mismatch_count_non_negative
    CHECK (type_mismatch_count >= 0);

ALTER TABLE excel_normalizer_service.data_quality_reports
    ADD CONSTRAINT chk_dqr_duplicate_count_non_negative
    CHECK (duplicate_count >= 0);

-- =============================================================================
-- Table: excel_normalizer_service.validation_results
-- =============================================================================
-- Schema validation result records. Each record captures a validation finding
-- for a specific cell or row including the sheet name, row index, column name,
-- severity (error/warning/info), rule name, message, expected vs actual
-- values, and tenant scope.

CREATE TABLE excel_normalizer_service.validation_results (
    finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES excel_normalizer_service.spreadsheet_files(file_id) ON DELETE CASCADE,
    sheet_name VARCHAR(255) NOT NULL,
    row_index INTEGER DEFAULT NULL,
    column_name VARCHAR(255) DEFAULT NULL,
    severity VARCHAR(20) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    expected_value TEXT DEFAULT NULL,
    actual_value TEXT DEFAULT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Severity constraint
ALTER TABLE excel_normalizer_service.validation_results
    ADD CONSTRAINT chk_vr_severity
    CHECK (severity IN ('error', 'warning', 'info'));

-- Rule name must not be empty
ALTER TABLE excel_normalizer_service.validation_results
    ADD CONSTRAINT chk_vr_rule_name_not_empty
    CHECK (LENGTH(TRIM(rule_name)) > 0);

-- Sheet name must not be empty
ALTER TABLE excel_normalizer_service.validation_results
    ADD CONSTRAINT chk_vr_sheet_name_not_empty
    CHECK (LENGTH(TRIM(sheet_name)) > 0);

-- =============================================================================
-- Table: excel_normalizer_service.transform_operations
-- =============================================================================
-- Transform operation log. Each record captures a data transformation applied
-- to a spreadsheet including the operation type (type_cast, unit_convert,
-- date_parse, string_clean, merge_columns, split_column, fill_nulls,
-- deduplicate, filter, aggregate), configuration (JSONB), input/output row
-- counts, rows affected, provenance hash, and tenant scope.

CREATE TABLE excel_normalizer_service.transform_operations (
    operation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES excel_normalizer_service.spreadsheet_files(file_id) ON DELETE CASCADE,
    operation_type VARCHAR(30) NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    input_rows INTEGER NOT NULL DEFAULT 0,
    output_rows INTEGER NOT NULL DEFAULT 0,
    rows_affected INTEGER NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Operation type constraint
ALTER TABLE excel_normalizer_service.transform_operations
    ADD CONSTRAINT chk_transform_operation_type
    CHECK (operation_type IN (
        'type_cast', 'unit_convert', 'date_parse', 'string_clean',
        'merge_columns', 'split_column', 'fill_nulls', 'deduplicate',
        'filter', 'aggregate', 'rename', 'reorder', 'derive', 'custom'
    ));

-- Input rows must be non-negative
ALTER TABLE excel_normalizer_service.transform_operations
    ADD CONSTRAINT chk_transform_input_rows_non_negative
    CHECK (input_rows >= 0);

-- Output rows must be non-negative
ALTER TABLE excel_normalizer_service.transform_operations
    ADD CONSTRAINT chk_transform_output_rows_non_negative
    CHECK (output_rows >= 0);

-- Rows affected must be non-negative
ALTER TABLE excel_normalizer_service.transform_operations
    ADD CONSTRAINT chk_transform_rows_affected_non_negative
    CHECK (rows_affected >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE excel_normalizer_service.transform_operations
    ADD CONSTRAINT chk_transform_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: excel_normalizer_service.excel_audit_log (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all Excel
-- normalizer operations. Each event captures the operation, entity being
-- operated on, details (JSONB), user, tenant, provenance hash, and timestamp.
-- Partitioned by created_at for time-series queries. Retained for 365 days
-- with compression after 30 days.

CREATE TABLE excel_normalizer_service.excel_audit_log (
    log_id UUID NOT NULL DEFAULT gen_random_uuid(),
    operation VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    user_id VARCHAR(100) DEFAULT 'system',
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (log_id, created_at)
);

-- Create hypertable partitioned by created_at
SELECT create_hypertable('excel_normalizer_service.excel_audit_log', 'created_at', if_not_exists => TRUE);

-- Entity type constraint
ALTER TABLE excel_normalizer_service.excel_audit_log
    ADD CONSTRAINT chk_excel_audit_entity_type
    CHECK (entity_type IN (
        'spreadsheet_file', 'sheet_metadata', 'column_mapping',
        'normalization_job', 'normalized_record', 'mapping_template',
        'data_quality_report', 'validation_result', 'transform_operation',
        'system'
    ));

-- Operation constraint
ALTER TABLE excel_normalizer_service.excel_audit_log
    ADD CONSTRAINT chk_excel_audit_operation
    CHECK (operation IN (
        'create', 'update', 'delete', 'upload', 'download',
        'normalize', 'validate', 'transform', 'map_columns',
        'apply_template', 'quality_check', 'export', 'import',
        'activate', 'deactivate', 'system', 'admin'
    ));

-- Provenance hash must be 64-character hex when present
ALTER TABLE excel_normalizer_service.excel_audit_log
    ADD CONSTRAINT chk_excel_audit_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Continuous Aggregate: excel_normalizer_service.hourly_normalization_stats
-- =============================================================================
-- Precomputed hourly normalization job statistics by status for dashboard
-- queries, trend analysis, and SLI tracking. Shows the number of jobs per
-- status, average duration, row processing counts, success/failure counts.

CREATE MATERIALIZED VIEW excel_normalizer_service.hourly_normalization_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', started_at) AS bucket,
    status,
    tenant_id,
    COUNT(*) AS job_count,
    AVG(duration_seconds) AS avg_duration_seconds,
    MAX(duration_seconds) AS max_duration_seconds,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
    SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) AS timeout_count,
    AVG(rows_processed) AS avg_rows_processed,
    AVG(rows_normalized) AS avg_rows_normalized,
    AVG(rows_skipped) AS avg_rows_skipped
FROM excel_normalizer_service.normalization_jobs
WHERE started_at IS NOT NULL
GROUP BY bucket, status, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('excel_normalizer_service.hourly_normalization_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: excel_normalizer_service.hourly_audit_stats
-- =============================================================================
-- Precomputed hourly counts of audit events by entity type and operation
-- for compliance reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW excel_normalizer_service.hourly_audit_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    entity_type,
    operation,
    tenant_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT entity_id) AS unique_entities,
    COUNT(DISTINCT user_id) AS unique_users
FROM excel_normalizer_service.excel_audit_log
WHERE created_at IS NOT NULL
GROUP BY bucket, entity_type, operation, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('excel_normalizer_service.hourly_audit_stats',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- spreadsheet_files indexes
CREATE INDEX idx_sf_file_name ON excel_normalizer_service.spreadsheet_files(file_name);
CREATE INDEX idx_sf_file_hash ON excel_normalizer_service.spreadsheet_files(file_hash);
CREATE INDEX idx_sf_file_format ON excel_normalizer_service.spreadsheet_files(file_format);
CREATE INDEX idx_sf_upload_timestamp ON excel_normalizer_service.spreadsheet_files(upload_timestamp DESC);
CREATE INDEX idx_sf_created_at ON excel_normalizer_service.spreadsheet_files(created_at DESC);
CREATE INDEX idx_sf_provenance_hash ON excel_normalizer_service.spreadsheet_files(provenance_hash);
CREATE INDEX idx_sf_tenant ON excel_normalizer_service.spreadsheet_files(tenant_id);
CREATE INDEX idx_sf_tenant_format ON excel_normalizer_service.spreadsheet_files(tenant_id, file_format);
CREATE INDEX idx_sf_tenant_uploaded ON excel_normalizer_service.spreadsheet_files(tenant_id, upload_timestamp DESC);
CREATE INDEX idx_sf_file_size ON excel_normalizer_service.spreadsheet_files(file_size_bytes);

-- sheet_metadata indexes
CREATE INDEX idx_sm_file ON excel_normalizer_service.sheet_metadata(file_id);
CREATE INDEX idx_sm_sheet_name ON excel_normalizer_service.sheet_metadata(sheet_name);
CREATE INDEX idx_sm_sheet_index ON excel_normalizer_service.sheet_metadata(sheet_index);
CREATE INDEX idx_sm_created_at ON excel_normalizer_service.sheet_metadata(created_at DESC);
CREATE INDEX idx_sm_tenant ON excel_normalizer_service.sheet_metadata(tenant_id);
CREATE INDEX idx_sm_tenant_file ON excel_normalizer_service.sheet_metadata(tenant_id, file_id);
CREATE INDEX idx_sm_encoding ON excel_normalizer_service.sheet_metadata(detected_encoding);

-- column_mappings indexes
CREATE INDEX idx_cm_sheet ON excel_normalizer_service.column_mappings(sheet_id);
CREATE INDEX idx_cm_source_column ON excel_normalizer_service.column_mappings(source_column);
CREATE INDEX idx_cm_canonical_field ON excel_normalizer_service.column_mappings(canonical_field);
CREATE INDEX idx_cm_mapping_strategy ON excel_normalizer_service.column_mappings(mapping_strategy);
CREATE INDEX idx_cm_confidence ON excel_normalizer_service.column_mappings(confidence);
CREATE INDEX idx_cm_detected_data_type ON excel_normalizer_service.column_mappings(detected_data_type);
CREATE INDEX idx_cm_created_at ON excel_normalizer_service.column_mappings(created_at DESC);
CREATE INDEX idx_cm_tenant ON excel_normalizer_service.column_mappings(tenant_id);
CREATE INDEX idx_cm_tenant_sheet ON excel_normalizer_service.column_mappings(tenant_id, sheet_id);
CREATE INDEX idx_cm_synonyms ON excel_normalizer_service.column_mappings USING GIN (synonyms_matched);

-- normalization_jobs indexes (hypertable-aware)
CREATE INDEX idx_nj_file ON excel_normalizer_service.normalization_jobs(file_id, started_at DESC);
CREATE INDEX idx_nj_status ON excel_normalizer_service.normalization_jobs(status, started_at DESC);
CREATE INDEX idx_nj_tenant ON excel_normalizer_service.normalization_jobs(tenant_id, started_at DESC);
CREATE INDEX idx_nj_tenant_status ON excel_normalizer_service.normalization_jobs(tenant_id, status, started_at DESC);
CREATE INDEX idx_nj_provenance_hash ON excel_normalizer_service.normalization_jobs(provenance_hash);
CREATE INDEX idx_nj_config ON excel_normalizer_service.normalization_jobs USING GIN (config);
CREATE INDEX idx_nj_errors ON excel_normalizer_service.normalization_jobs USING GIN (errors);

-- normalized_records indexes
CREATE INDEX idx_nr_job ON excel_normalizer_service.normalized_records(job_id);
CREATE INDEX idx_nr_row_index ON excel_normalizer_service.normalized_records(row_index);
CREATE INDEX idx_nr_quality_score ON excel_normalizer_service.normalized_records(quality_score);
CREATE INDEX idx_nr_provenance_hash ON excel_normalizer_service.normalized_records(provenance_hash);
CREATE INDEX idx_nr_created_at ON excel_normalizer_service.normalized_records(created_at DESC);
CREATE INDEX idx_nr_tenant ON excel_normalizer_service.normalized_records(tenant_id);
CREATE INDEX idx_nr_tenant_job ON excel_normalizer_service.normalized_records(tenant_id, job_id);
CREATE INDEX idx_nr_original_values ON excel_normalizer_service.normalized_records USING GIN (original_values);
CREATE INDEX idx_nr_normalized_values ON excel_normalizer_service.normalized_records USING GIN (normalized_values);
CREATE INDEX idx_nr_validation_errors ON excel_normalizer_service.normalized_records USING GIN (validation_errors);

-- mapping_templates indexes
CREATE INDEX idx_mt_template_name ON excel_normalizer_service.mapping_templates(template_name);
CREATE INDEX idx_mt_source_type ON excel_normalizer_service.mapping_templates(source_type);
CREATE INDEX idx_mt_usage_count ON excel_normalizer_service.mapping_templates(usage_count DESC);
CREATE INDEX idx_mt_created_at ON excel_normalizer_service.mapping_templates(created_at DESC);
CREATE INDEX idx_mt_updated_at ON excel_normalizer_service.mapping_templates(updated_at DESC);
CREATE INDEX idx_mt_tenant ON excel_normalizer_service.mapping_templates(tenant_id);
CREATE INDEX idx_mt_tenant_type ON excel_normalizer_service.mapping_templates(tenant_id, source_type);
CREATE INDEX idx_mt_column_mappings ON excel_normalizer_service.mapping_templates USING GIN (column_mappings);

-- data_quality_reports indexes
CREATE INDEX idx_dqr_file ON excel_normalizer_service.data_quality_reports(file_id);
CREATE INDEX idx_dqr_overall_score ON excel_normalizer_service.data_quality_reports(overall_score);
CREATE INDEX idx_dqr_quality_level ON excel_normalizer_service.data_quality_reports(quality_level);
CREATE INDEX idx_dqr_created_at ON excel_normalizer_service.data_quality_reports(created_at DESC);
CREATE INDEX idx_dqr_tenant ON excel_normalizer_service.data_quality_reports(tenant_id);
CREATE INDEX idx_dqr_tenant_file ON excel_normalizer_service.data_quality_reports(tenant_id, file_id);
CREATE INDEX idx_dqr_column_scores ON excel_normalizer_service.data_quality_reports USING GIN (column_scores);
CREATE INDEX idx_dqr_issues ON excel_normalizer_service.data_quality_reports USING GIN (issues);

-- validation_results indexes
CREATE INDEX idx_vr_file ON excel_normalizer_service.validation_results(file_id);
CREATE INDEX idx_vr_sheet_name ON excel_normalizer_service.validation_results(sheet_name);
CREATE INDEX idx_vr_severity ON excel_normalizer_service.validation_results(severity);
CREATE INDEX idx_vr_rule_name ON excel_normalizer_service.validation_results(rule_name);
CREATE INDEX idx_vr_column_name ON excel_normalizer_service.validation_results(column_name);
CREATE INDEX idx_vr_created_at ON excel_normalizer_service.validation_results(created_at DESC);
CREATE INDEX idx_vr_tenant ON excel_normalizer_service.validation_results(tenant_id);
CREATE INDEX idx_vr_tenant_file ON excel_normalizer_service.validation_results(tenant_id, file_id);
CREATE INDEX idx_vr_tenant_severity ON excel_normalizer_service.validation_results(tenant_id, severity);

-- transform_operations indexes
CREATE INDEX idx_to_file ON excel_normalizer_service.transform_operations(file_id);
CREATE INDEX idx_to_operation_type ON excel_normalizer_service.transform_operations(operation_type);
CREATE INDEX idx_to_provenance_hash ON excel_normalizer_service.transform_operations(provenance_hash);
CREATE INDEX idx_to_created_at ON excel_normalizer_service.transform_operations(created_at DESC);
CREATE INDEX idx_to_tenant ON excel_normalizer_service.transform_operations(tenant_id);
CREATE INDEX idx_to_tenant_file ON excel_normalizer_service.transform_operations(tenant_id, file_id);
CREATE INDEX idx_to_config ON excel_normalizer_service.transform_operations USING GIN (config);

-- excel_audit_log indexes (hypertable-aware)
CREATE INDEX idx_eal_entity_type ON excel_normalizer_service.excel_audit_log(entity_type, created_at DESC);
CREATE INDEX idx_eal_entity_id ON excel_normalizer_service.excel_audit_log(entity_id, created_at DESC);
CREATE INDEX idx_eal_operation ON excel_normalizer_service.excel_audit_log(operation, created_at DESC);
CREATE INDEX idx_eal_user ON excel_normalizer_service.excel_audit_log(user_id, created_at DESC);
CREATE INDEX idx_eal_tenant ON excel_normalizer_service.excel_audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_eal_provenance_hash ON excel_normalizer_service.excel_audit_log(provenance_hash);
CREATE INDEX idx_eal_details ON excel_normalizer_service.excel_audit_log USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE excel_normalizer_service.spreadsheet_files ENABLE ROW LEVEL SECURITY;
CREATE POLICY sf_tenant_read ON excel_normalizer_service.spreadsheet_files
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sf_tenant_write ON excel_normalizer_service.spreadsheet_files
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.sheet_metadata ENABLE ROW LEVEL SECURITY;
CREATE POLICY sm_tenant_read ON excel_normalizer_service.sheet_metadata
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sm_tenant_write ON excel_normalizer_service.sheet_metadata
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.column_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY cm_tenant_read ON excel_normalizer_service.column_mappings
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cm_tenant_write ON excel_normalizer_service.column_mappings
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.normalization_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY nj_tenant_read ON excel_normalizer_service.normalization_jobs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY nj_tenant_write ON excel_normalizer_service.normalization_jobs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.normalized_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY nr_tenant_read ON excel_normalizer_service.normalized_records
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY nr_tenant_write ON excel_normalizer_service.normalized_records
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.mapping_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY mte_tenant_read ON excel_normalizer_service.mapping_templates
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY mte_tenant_write ON excel_normalizer_service.mapping_templates
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.data_quality_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY dqr_tenant_read ON excel_normalizer_service.data_quality_reports
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dqr_tenant_write ON excel_normalizer_service.data_quality_reports
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.validation_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY vr_tenant_read ON excel_normalizer_service.validation_results
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY vr_tenant_write ON excel_normalizer_service.validation_results
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.transform_operations ENABLE ROW LEVEL SECURITY;
CREATE POLICY tro_tenant_read ON excel_normalizer_service.transform_operations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY tro_tenant_write ON excel_normalizer_service.transform_operations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE excel_normalizer_service.excel_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY eal_tenant_read ON excel_normalizer_service.excel_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY eal_tenant_write ON excel_normalizer_service.excel_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA excel_normalizer_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA excel_normalizer_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA excel_normalizer_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON excel_normalizer_service.hourly_normalization_stats TO greenlang_app;
GRANT SELECT ON excel_normalizer_service.hourly_audit_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA excel_normalizer_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA excel_normalizer_service TO greenlang_readonly;
GRANT SELECT ON excel_normalizer_service.hourly_normalization_stats TO greenlang_readonly;
GRANT SELECT ON excel_normalizer_service.hourly_audit_stats TO greenlang_readonly;

-- Add Excel normalizer service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'excel_normalizer:files:read', 'excel_normalizer', 'files_read', 'View uploaded spreadsheet files and metadata'),
    (gen_random_uuid(), 'excel_normalizer:files:write', 'excel_normalizer', 'files_write', 'Upload and manage spreadsheet files'),
    (gen_random_uuid(), 'excel_normalizer:sheets:read', 'excel_normalizer', 'sheets_read', 'View sheet metadata and column structure'),
    (gen_random_uuid(), 'excel_normalizer:sheets:write', 'excel_normalizer', 'sheets_write', 'Create and update sheet metadata records'),
    (gen_random_uuid(), 'excel_normalizer:mappings:read', 'excel_normalizer', 'mappings_read', 'View column mappings and canonical field assignments'),
    (gen_random_uuid(), 'excel_normalizer:mappings:write', 'excel_normalizer', 'mappings_write', 'Create and manage column mappings'),
    (gen_random_uuid(), 'excel_normalizer:jobs:read', 'excel_normalizer', 'jobs_read', 'View normalization job status and results'),
    (gen_random_uuid(), 'excel_normalizer:jobs:write', 'excel_normalizer', 'jobs_write', 'Create and manage normalization jobs'),
    (gen_random_uuid(), 'excel_normalizer:records:read', 'excel_normalizer', 'records_read', 'View normalized output records and quality scores'),
    (gen_random_uuid(), 'excel_normalizer:records:write', 'excel_normalizer', 'records_write', 'Create normalized output records'),
    (gen_random_uuid(), 'excel_normalizer:templates:read', 'excel_normalizer', 'templates_read', 'View mapping template definitions'),
    (gen_random_uuid(), 'excel_normalizer:templates:write', 'excel_normalizer', 'templates_write', 'Create and manage mapping templates'),
    (gen_random_uuid(), 'excel_normalizer:audit:read', 'excel_normalizer', 'audit_read', 'View Excel normalizer audit event log'),
    (gen_random_uuid(), 'excel_normalizer:admin', 'excel_normalizer', 'admin', 'Excel normalizer service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep normalization jobs for 30 days
SELECT add_retention_policy('excel_normalizer_service.normalization_jobs', INTERVAL '30 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('excel_normalizer_service.excel_audit_log', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on normalization_jobs after 3 days
ALTER TABLE excel_normalizer_service.normalization_jobs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'started_at DESC'
);

SELECT add_compression_policy('excel_normalizer_service.normalization_jobs', INTERVAL '3 days');

-- Enable compression on excel_audit_log after 30 days
ALTER TABLE excel_normalizer_service.excel_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('excel_normalizer_service.excel_audit_log', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Excel Normalizer Agent (GL-DATA-X-016) in Agent Registry
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-016', 'Excel & CSV Normalizer',
 'Normalizes structured tabular data from Excel (XLSX, XLS), CSV, TSV, and ODS spreadsheets into canonical GreenLang Climate OS format. Provides automatic header detection, column-to-canonical-field mapping (exact, synonym, fuzzy, ML), data type inference, unit detection, quality scoring, validation, transform operations, reusable mapping templates, and SHA-256 provenance hash chains for every normalized record.',
 2, 'async', true, true, 10, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/excel-normalizer', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Excel Normalizer
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-016', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/excel-normalizer-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"data", "normalization", "excel", "csv", "spreadsheet", "tabular", "mapping"}',
 '{"cross-sector"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Excel Normalizer
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-X-016', '1.0.0', 'file_upload', 'ingestion',
 'Upload and register Excel, CSV, TSV, or ODS spreadsheet files for normalization processing with automatic format detection and sheet enumeration',
 '{"file_content", "file_name", "file_format"}', '{"file_id", "file_hash", "sheet_count", "total_rows"}',
 '{"max_file_size_mb": 50, "supported_formats": ["xlsx", "xls", "csv", "tsv", "ods"]}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'header_detection', 'computation',
 'Automatically detect header rows, column names, data types, encodings, and delimiters in spreadsheet files',
 '{"file_id", "sheet_id"}', '{"headers", "data_types", "encoding", "delimiter"}',
 '{"max_header_scan_rows": 20, "encoding_detection": true, "delimiter_detection": true}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'column_mapping', 'computation',
 'Map source column headers to canonical GreenLang fields using exact match, synonym matching, fuzzy matching, or ML-based classification',
 '{"sheet_id", "template_id"}', '{"column_mappings", "confidence_scores"}',
 '{"strategies": ["exact", "synonym", "fuzzy", "ml", "manual", "template", "regex"], "min_confidence": 0.5}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'normalization', 'computation',
 'Normalize spreadsheet data into canonical format with type casting, unit conversion, date parsing, string cleaning, and quality scoring',
 '{"file_id", "column_mappings"}', '{"normalized_records", "quality_report", "row_counts"}',
 '{"batch_size": 1000, "skip_invalid_rows": false, "quality_threshold": 0.5}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'data_quality_assessment', 'validation',
 'Assess data quality across completeness, accuracy, and consistency dimensions with per-column scoring and issue identification',
 '{"file_id"}', '{"quality_report", "overall_score", "issues"}',
 '{"check_completeness": true, "check_accuracy": true, "check_consistency": true, "check_duplicates": true}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'schema_validation', 'validation',
 'Validate spreadsheet data against schema rules including type checks, range validation, required fields, format patterns, and cross-column consistency',
 '{"file_id", "validation_rules"}', '{"validation_results", "severity_counts"}',
 '{"severity_levels": ["error", "warning", "info"], "cross_column_rules": true}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'template_management', 'configuration',
 'Create, update, and apply reusable column mapping templates for consistent normalization across similar spreadsheet formats',
 '{"template_definition"}', '{"template_id", "validation_result"}',
 '{"source_types": ["utility_bill", "emissions_report", "fuel_log", "travel_log", "waste_manifest", "energy_audit", "fleet_data", "custom"]}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'transform_operations', 'computation',
 'Apply data transformation operations including type casting, unit conversion, date parsing, string cleaning, column merge/split, null filling, deduplication, filtering, and aggregation',
 '{"file_id", "operations"}', '{"transform_results", "rows_affected"}',
 '{"operation_types": ["type_cast", "unit_convert", "date_parse", "string_clean", "merge_columns", "split_column", "fill_nulls", "deduplicate", "filter", "aggregate"]}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Excel Normalizer
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Excel Normalizer depends on Schema Compiler for input/output validation
('GL-DATA-X-016', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Normalized records and spreadsheet metadata are validated against JSON Schema definitions'),

-- Excel Normalizer depends on Unit Normalizer for unit conversion
('GL-DATA-X-016', 'GL-FOUND-X-003', '>=1.0.0', false,
 'Detected units in spreadsheet columns (kWh, kg, liters, etc.) are converted to standard units for emission calculations'),

-- Excel Normalizer depends on Registry for agent discovery
('GL-DATA-X-016', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for normalization pipeline orchestration'),

-- Excel Normalizer optionally uses Citations for provenance tracking
('GL-DATA-X-016', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Normalized data provenance chains are registered with the citation service for audit trail'),

-- Excel Normalizer optionally uses Reproducibility for determinism
('GL-DATA-X-016', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Normalization results are verified for reproducibility across re-processing runs')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Excel Normalizer
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-016', 'Excel & CSV Normalizer',
 'Tabular data normalization for Excel, CSV, TSV, and ODS spreadsheets. Automatic header detection, column-to-canonical mapping, data type inference, quality scoring, validation, transform operations, reusable templates, and SHA-256 provenance hash chains for every normalized record.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA excel_normalizer_service IS 'Excel & CSV Normalizer for GreenLang Climate OS (AGENT-DATA-002) - tabular data normalization for Excel/CSV/TSV/ODS, column mapping, data type inference, quality scoring, validation, transform operations, and provenance tracking';
COMMENT ON TABLE excel_normalizer_service.spreadsheet_files IS 'Spreadsheet file registry with file metadata, SHA-256 hash, format (xlsx/xls/csv/tsv/ods), sheet count, row/column counts, and provenance hash';
COMMENT ON TABLE excel_normalizer_service.sheet_metadata IS 'Per-sheet metadata with sheet name, index, row/column counts, header detection, encoding, and delimiter';
COMMENT ON TABLE excel_normalizer_service.column_mappings IS 'Column-to-canonical-field mappings with strategy (exact/synonym/fuzzy/ml/manual), confidence score, detected data type, unit, and synonyms';
COMMENT ON TABLE excel_normalizer_service.normalization_jobs IS 'TimescaleDB hypertable: normalization job execution records with status, config, row counts, errors, duration, and provenance hash';
COMMENT ON TABLE excel_normalizer_service.normalized_records IS 'Normalized output records with original values, normalized values, quality score, validation errors, and provenance hash';
COMMENT ON TABLE excel_normalizer_service.mapping_templates IS 'Reusable column mapping templates with source type classification and usage tracking';
COMMENT ON TABLE excel_normalizer_service.data_quality_reports IS 'Data quality assessment reports with overall/completeness/accuracy/consistency scores, row counts, per-column scores, and identified issues';
COMMENT ON TABLE excel_normalizer_service.validation_results IS 'Schema validation finding records with severity (error/warning/info), rule name, expected vs actual values, sheet name, and row/column location';
COMMENT ON TABLE excel_normalizer_service.transform_operations IS 'Transform operation log with operation type, config, input/output row counts, rows affected, and provenance hash';
COMMENT ON TABLE excel_normalizer_service.excel_audit_log IS 'TimescaleDB hypertable: comprehensive audit events for all Excel normalizer operations with provenance hash chain';
COMMENT ON MATERIALIZED VIEW excel_normalizer_service.hourly_normalization_stats IS 'Continuous aggregate: hourly normalization job statistics by status for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW excel_normalizer_service.hourly_audit_stats IS 'Continuous aggregate: hourly audit event counts by entity type and operation for compliance reporting';

COMMENT ON COLUMN excel_normalizer_service.spreadsheet_files.file_format IS 'File format: xlsx, xls, csv, tsv, ods';
COMMENT ON COLUMN excel_normalizer_service.spreadsheet_files.file_hash IS 'SHA-256 hash of the uploaded file for integrity verification and deduplication';
COMMENT ON COLUMN excel_normalizer_service.spreadsheet_files.provenance_hash IS 'SHA-256 hash of file metadata for provenance chain';
COMMENT ON COLUMN excel_normalizer_service.sheet_metadata.detected_encoding IS 'Detected encoding: utf-8, utf-16, ascii, iso-8859-1, windows-1252, shift_jis, etc.';
COMMENT ON COLUMN excel_normalizer_service.column_mappings.mapping_strategy IS 'Mapping strategy: exact, synonym, fuzzy, ml, manual, template, regex';
COMMENT ON COLUMN excel_normalizer_service.column_mappings.detected_data_type IS 'Detected data type: string, integer, float, decimal, boolean, date, datetime, currency, percentage, etc.';
COMMENT ON COLUMN excel_normalizer_service.normalization_jobs.status IS 'Job status: pending, running, completed, failed, cancelled, timeout';
COMMENT ON COLUMN excel_normalizer_service.data_quality_reports.quality_level IS 'Quality level: excellent, good, acceptable, poor, critical, unknown';
COMMENT ON COLUMN excel_normalizer_service.validation_results.severity IS 'Validation severity: error, warning, info';
COMMENT ON COLUMN excel_normalizer_service.transform_operations.operation_type IS 'Transform type: type_cast, unit_convert, date_parse, string_clean, merge_columns, split_column, fill_nulls, deduplicate, filter, aggregate, rename, reorder, derive, custom';
COMMENT ON COLUMN excel_normalizer_service.excel_audit_log.entity_type IS 'Entity type: spreadsheet_file, sheet_metadata, column_mapping, normalization_job, normalized_record, mapping_template, data_quality_report, validation_result, transform_operation, system';
COMMENT ON COLUMN excel_normalizer_service.excel_audit_log.provenance_hash IS 'SHA-256 hash for provenance tracking and tamper detection';
