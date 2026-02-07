-- =============================================================================
-- GreenLang Climate OS - Schema Compiler & Validator Service Schema
-- =============================================================================
-- Migration: V022
-- Component: AGENT-FOUND-002 GreenLang Schema Compiler & Validator
-- Description: Creates schema_service schema with schema registry, validation
--              audit log (hypertable), schema cache metadata, hourly validation
--              continuous aggregate, and seed data for built-in GreenLang schemas.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS schema_service;

-- =============================================================================
-- Table: schema_service.schema_registry
-- =============================================================================
-- Central schema registry storing versioned schema definitions. Each row
-- represents a specific version of a schema with its full JSONB content,
-- SHA-256 hash for integrity, content type, and metadata. Composite primary
-- key on (schema_id, version) supports multiple versions per schema.
-- Multi-tenant via tenant_id with Row-Level Security.

CREATE TABLE schema_service.schema_registry (
    schema_id VARCHAR(256) NOT NULL,
    version VARCHAR(64) NOT NULL,
    schema_hash VARCHAR(64) NOT NULL,
    content_type VARCHAR(64) NOT NULL DEFAULT 'application/json',
    schema_content JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    deprecated BOOLEAN NOT NULL DEFAULT FALSE,
    deprecated_message TEXT,
    created_by VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(128),
    PRIMARY KEY (schema_id, version)
);

-- Content type constraint
ALTER TABLE schema_service.schema_registry
    ADD CONSTRAINT chk_schema_content_type
    CHECK (content_type IN ('application/json', 'application/yaml', 'application/x-greenlang-schema'));

-- Schema hash must be 64-character hex
ALTER TABLE schema_service.schema_registry
    ADD CONSTRAINT chk_schema_hash_length
    CHECK (LENGTH(schema_hash) = 64);

-- =============================================================================
-- Table: schema_service.validation_audit_log
-- =============================================================================
-- TimescaleDB hypertable recording every validation request for audit,
-- analytics, and compliance. Each row captures a single validation attempt
-- with the schema reference, result, finding counts, payload hash for
-- deduplication, duration, validation profile, and tenant. Partitioned by
-- timestamp for efficient time-series queries and retention management.

CREATE TABLE schema_service.validation_audit_log (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    schema_id VARCHAR(256) NOT NULL,
    schema_version VARCHAR(64) NOT NULL,
    schema_hash VARCHAR(64) NOT NULL,
    valid BOOLEAN NOT NULL,
    error_count INTEGER NOT NULL DEFAULT 0,
    warning_count INTEGER NOT NULL DEFAULT 0,
    fix_count INTEGER NOT NULL DEFAULT 0,
    payload_hash VARCHAR(64) NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    profile VARCHAR(64) NOT NULL DEFAULT 'standard',
    tenant_id VARCHAR(128)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('schema_service.validation_audit_log', 'timestamp', if_not_exists => TRUE);

-- Validation profile constraint
ALTER TABLE schema_service.validation_audit_log
    ADD CONSTRAINT chk_audit_profile
    CHECK (profile IN ('standard', 'strict', 'lenient', 'performance'));

-- Error/warning/fix counts must be non-negative
ALTER TABLE schema_service.validation_audit_log
    ADD CONSTRAINT chk_audit_counts_positive
    CHECK (error_count >= 0 AND warning_count >= 0 AND fix_count >= 0);

-- Duration must be positive
ALTER TABLE schema_service.validation_audit_log
    ADD CONSTRAINT chk_audit_duration_positive
    CHECK (duration_ms >= 0);

-- =============================================================================
-- Table: schema_service.schema_cache_metadata
-- =============================================================================
-- Tracks compiled schema IR cache entries with access statistics. Each row
-- corresponds to a single cached IR entry, identified by a unique cache key.
-- Used for cache analytics, warmup prioritization, and capacity planning.

CREATE TABLE schema_service.schema_cache_metadata (
    cache_key VARCHAR(256) PRIMARY KEY,
    schema_id VARCHAR(256) NOT NULL,
    version VARCHAR(64) NOT NULL,
    ir_hash VARCHAR(64) NOT NULL,
    compiled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count BIGINT NOT NULL DEFAULT 0,
    size_bytes BIGINT NOT NULL DEFAULT 0,
    tenant_id VARCHAR(128)
);

-- Access count must be non-negative
ALTER TABLE schema_service.schema_cache_metadata
    ADD CONSTRAINT chk_cache_access_count_positive
    CHECK (access_count >= 0);

-- Size must be non-negative
ALTER TABLE schema_service.schema_cache_metadata
    ADD CONSTRAINT chk_cache_size_positive
    CHECK (size_bytes >= 0);

-- =============================================================================
-- Continuous Aggregate: schema_service.hourly_validation_summaries
-- =============================================================================
-- Precomputed hourly validation summaries for efficient dashboard queries.
-- Aggregates validation audit data into per-schema hourly statistics
-- including total validations, valid/invalid counts, average/max duration,
-- and cumulative error/warning totals.

CREATE MATERIALIZED VIEW schema_service.hourly_validation_summaries
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    schema_id,
    COUNT(*) AS total_validations,
    COUNT(CASE WHEN valid = TRUE THEN 1 END) AS valid_count,
    COUNT(CASE WHEN valid = FALSE THEN 1 END) AS invalid_count,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    SUM(error_count) AS total_errors,
    SUM(warning_count) AS total_warnings
FROM schema_service.validation_audit_log
WHERE timestamp IS NOT NULL
GROUP BY bucket, schema_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('schema_service.hourly_validation_summaries',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- schema_registry indexes
CREATE INDEX idx_schema_registry_schema_id ON schema_service.schema_registry(schema_id);
CREATE INDEX idx_schema_registry_schema_hash ON schema_service.schema_registry(schema_hash);
CREATE INDEX idx_schema_registry_tenant ON schema_service.schema_registry(tenant_id);
CREATE INDEX idx_schema_registry_created_at ON schema_service.schema_registry(created_at DESC);
CREATE INDEX idx_schema_registry_content_type ON schema_service.schema_registry(content_type);
CREATE INDEX idx_schema_registry_deprecated ON schema_service.schema_registry(deprecated) WHERE deprecated = false;
CREATE INDEX idx_schema_registry_metadata ON schema_service.schema_registry USING GIN (metadata);

-- validation_audit_log indexes (hypertable-aware)
CREATE INDEX idx_audit_log_schema_id ON schema_service.validation_audit_log(schema_id, timestamp DESC);
CREATE INDEX idx_audit_log_schema_version ON schema_service.validation_audit_log(schema_id, schema_version, timestamp DESC);
CREATE INDEX idx_audit_log_tenant ON schema_service.validation_audit_log(tenant_id, timestamp DESC);
CREATE INDEX idx_audit_log_valid ON schema_service.validation_audit_log(valid, timestamp DESC);
CREATE INDEX idx_audit_log_payload_hash ON schema_service.validation_audit_log(payload_hash);
CREATE INDEX idx_audit_log_profile ON schema_service.validation_audit_log(profile);

-- schema_cache_metadata indexes
CREATE INDEX idx_cache_metadata_schema_id ON schema_service.schema_cache_metadata(schema_id);
CREATE INDEX idx_cache_metadata_last_accessed ON schema_service.schema_cache_metadata(last_accessed DESC);
CREATE INDEX idx_cache_metadata_access_count ON schema_service.schema_cache_metadata(access_count DESC);
CREATE INDEX idx_cache_metadata_tenant ON schema_service.schema_cache_metadata(tenant_id);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE schema_service.schema_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY schema_registry_tenant_read ON schema_service.schema_registry
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY schema_registry_tenant_write ON schema_service.schema_registry
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE schema_service.validation_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY validation_audit_log_tenant_read ON schema_service.validation_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY validation_audit_log_tenant_write ON schema_service.validation_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE schema_service.schema_cache_metadata ENABLE ROW LEVEL SECURITY;
CREATE POLICY schema_cache_metadata_tenant_read ON schema_service.schema_cache_metadata
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY schema_cache_metadata_tenant_write ON schema_service.schema_cache_metadata
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA schema_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA schema_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA schema_service TO greenlang_app;

-- Grant SELECT on the continuous aggregate
GRANT SELECT ON schema_service.hourly_validation_summaries TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA schema_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA schema_service TO greenlang_readonly;
GRANT SELECT ON schema_service.hourly_validation_summaries TO greenlang_readonly;

-- Add schema service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'schema:registry:read', 'schema', 'read', 'View schema definitions and versions'),
    (gen_random_uuid(), 'schema:registry:write', 'schema', 'write', 'Create/update schema definitions'),
    (gen_random_uuid(), 'schema:registry:delete', 'schema', 'delete', 'Delete schema definitions'),
    (gen_random_uuid(), 'schema:validate:execute', 'schema', 'validate', 'Execute schema validation requests'),
    (gen_random_uuid(), 'schema:validate:batch', 'schema', 'validate_batch', 'Execute batch validation requests'),
    (gen_random_uuid(), 'schema:compile:execute', 'schema', 'compile', 'Compile schemas to intermediate representation'),
    (gen_random_uuid(), 'schema:audit:read', 'schema', 'audit_read', 'View validation audit log'),
    (gen_random_uuid(), 'schema:cache:read', 'schema', 'cache_read', 'View schema cache metadata'),
    (gen_random_uuid(), 'schema:cache:manage', 'schema', 'cache_manage', 'Manage schema cache (warmup, invalidate)'),
    (gen_random_uuid(), 'schema:admin', 'schema', 'admin', 'Schema service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Trigger: auto-update updated_at on schema_registry
-- =============================================================================

CREATE OR REPLACE FUNCTION schema_service.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_schema_registry_updated_at
    BEFORE UPDATE ON schema_service.schema_registry
    FOR EACH ROW
    EXECUTE FUNCTION schema_service.update_updated_at();

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep validation audit log for 90 days
SELECT add_retention_policy('schema_service.validation_audit_log', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on validation audit log after 7 days
ALTER TABLE schema_service.validation_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'schema_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('schema_service.validation_audit_log', INTERVAL '7 days');

-- =============================================================================
-- Seed: Built-in GreenLang Schemas
-- =============================================================================

-- Seed 1: gl-emissions-input - Standard emissions data input schema
INSERT INTO schema_service.schema_registry (schema_id, version, schema_hash, content_type, schema_content, metadata, created_by, tenant_id) VALUES
('gl-emissions-input', '1.0.0',
'e1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2',
'application/x-greenlang-schema',
'{
  "schema_id": "gl-emissions-input",
  "version": "1.0.0",
  "description": "Standard GreenLang emissions data input schema for Scope 1, 2, and 3 activity data",
  "properties": {
    "organization_id": { "type": "string", "required": true, "description": "Unique organization identifier" },
    "reporting_period": {
      "type": "object",
      "required": true,
      "properties": {
        "start_date": { "type": "date", "required": true, "format": "ISO8601" },
        "end_date": { "type": "date", "required": true, "format": "ISO8601" }
      }
    },
    "scope": { "type": "integer", "required": true, "enum": [1, 2, 3], "description": "GHG Protocol scope" },
    "category": { "type": "string", "required": true, "description": "Emissions category" },
    "activity_type": { "type": "string", "required": true, "description": "Type of activity (e.g., stationary_combustion)" },
    "quantity": { "type": "number", "required": true, "constraints": { "min": 0 }, "description": "Activity quantity" },
    "unit": { "type": "string", "required": true, "description": "Unit of measurement (e.g., kWh, liters, kg)" },
    "emission_factor_id": { "type": "string", "required": false, "description": "Reference to emission factor" },
    "metadata": { "type": "object", "required": false, "description": "Additional metadata" }
  },
  "rules": [
    { "rule": "end_date >= start_date", "message": "End date must be on or after start date" },
    { "rule": "quantity > 0", "message": "Quantity must be positive" }
  ]
}'::jsonb,
'{"category": "emissions", "ghg_protocol": true, "supported_scopes": [1, 2, 3]}'::jsonb,
'system', NULL),

-- Seed 2: gl-activity-data - Activity data collection schema
('gl-activity-data', '1.0.0',
'f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3',
'application/x-greenlang-schema',
'{
  "schema_id": "gl-activity-data",
  "version": "1.0.0",
  "description": "GreenLang activity data collection schema for raw operational data ingestion",
  "properties": {
    "source_id": { "type": "string", "required": true, "description": "Data source identifier" },
    "source_type": { "type": "string", "required": true, "enum": ["meter", "invoice", "manual", "api", "iot"], "description": "Type of data source" },
    "timestamp": { "type": "datetime", "required": true, "format": "ISO8601", "description": "Activity timestamp" },
    "facility_id": { "type": "string", "required": true, "description": "Facility identifier" },
    "activity_category": { "type": "string", "required": true, "description": "Activity category (e.g., electricity, natural_gas, fleet)" },
    "readings": {
      "type": "array",
      "required": true,
      "items": {
        "type": "object",
        "properties": {
          "metric": { "type": "string", "required": true },
          "value": { "type": "number", "required": true },
          "unit": { "type": "string", "required": true },
          "quality": { "type": "string", "enum": ["measured", "estimated", "calculated"], "default": "measured" }
        }
      }
    },
    "tags": { "type": "object", "required": false, "description": "Free-form tags for filtering" },
    "provenance": {
      "type": "object",
      "required": false,
      "properties": {
        "collector": { "type": "string" },
        "collection_method": { "type": "string" },
        "confidence": { "type": "number", "constraints": { "min": 0, "max": 1 } }
      }
    }
  },
  "rules": [
    { "rule": "readings.length > 0", "message": "At least one reading is required" }
  ]
}'::jsonb,
'{"category": "activity_data", "supports_streaming": true, "data_quality_tracking": true}'::jsonb,
'system', NULL),

-- Seed 3: gl-calculation-result - Emissions calculation result schema
('gl-calculation-result', '1.0.0',
'a3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4',
'application/x-greenlang-schema',
'{
  "schema_id": "gl-calculation-result",
  "version": "1.0.0",
  "description": "GreenLang emissions calculation result schema for validated computation outputs",
  "properties": {
    "calculation_id": { "type": "string", "required": true, "description": "Unique calculation identifier" },
    "execution_id": { "type": "string", "required": false, "description": "DAG execution ID if orchestrated" },
    "scope": { "type": "integer", "required": true, "enum": [1, 2, 3] },
    "category": { "type": "string", "required": true },
    "emissions": {
      "type": "object",
      "required": true,
      "properties": {
        "co2e_kg": { "type": "number", "required": true, "constraints": { "min": 0 }, "description": "CO2 equivalent in kilograms" },
        "co2_kg": { "type": "number", "required": false, "constraints": { "min": 0 } },
        "ch4_kg": { "type": "number", "required": false, "constraints": { "min": 0 } },
        "n2o_kg": { "type": "number", "required": false, "constraints": { "min": 0 } }
      }
    },
    "emission_factor": {
      "type": "object",
      "required": true,
      "properties": {
        "factor_id": { "type": "string", "required": true },
        "value": { "type": "number", "required": true },
        "unit": { "type": "string", "required": true },
        "source": { "type": "string", "required": true },
        "year": { "type": "integer", "required": true }
      }
    },
    "methodology": { "type": "string", "required": true, "description": "Calculation methodology reference" },
    "uncertainty_percent": { "type": "number", "required": false, "constraints": { "min": 0, "max": 100 } },
    "data_quality_score": { "type": "number", "required": false, "constraints": { "min": 1, "max": 5 } },
    "calculated_at": { "type": "datetime", "required": true, "format": "ISO8601" },
    "provenance_hash": { "type": "string", "required": false, "description": "SHA-256 hash of the calculation provenance chain" }
  },
  "rules": [
    { "rule": "emissions.co2e_kg >= 0", "message": "CO2e emissions must be non-negative" },
    { "rule": "data_quality_score >= 1 AND data_quality_score <= 5", "message": "Data quality score must be between 1 and 5" }
  ]
}'::jsonb,
'{"category": "calculation_result", "ghg_protocol": true, "supports_provenance": true, "audit_trail": true}'::jsonb,
'system', NULL)
ON CONFLICT (schema_id, version) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA schema_service IS 'Schema Compiler & Validator service for GreenLang Climate OS (AGENT-FOUND-002)';
COMMENT ON TABLE schema_service.schema_registry IS 'Versioned schema definitions with JSONB content, hash integrity, and multi-tenant support';
COMMENT ON TABLE schema_service.validation_audit_log IS 'TimescaleDB hypertable: per-validation audit records with timing, findings, and tenant isolation';
COMMENT ON TABLE schema_service.schema_cache_metadata IS 'Compiled schema IR cache analytics with access statistics for warmup and capacity planning';
COMMENT ON MATERIALIZED VIEW schema_service.hourly_validation_summaries IS 'Continuous aggregate: hourly validation summaries for dashboard and trend analysis';

COMMENT ON COLUMN schema_service.schema_registry.schema_id IS 'Unique schema identifier (e.g., gl-emissions-input)';
COMMENT ON COLUMN schema_service.schema_registry.version IS 'Semantic version of the schema (e.g., 1.0.0)';
COMMENT ON COLUMN schema_service.schema_registry.schema_hash IS 'SHA-256 hash of schema_content for integrity verification';
COMMENT ON COLUMN schema_service.schema_registry.content_type IS 'MIME type of the schema (JSON, YAML, or GreenLang native)';
COMMENT ON COLUMN schema_service.schema_registry.schema_content IS 'Full schema definition as JSONB';
COMMENT ON COLUMN schema_service.schema_registry.metadata IS 'Additional schema metadata (category, protocol support, etc.)';

COMMENT ON COLUMN schema_service.validation_audit_log.schema_hash IS 'SHA-256 hash of the schema used for this validation';
COMMENT ON COLUMN schema_service.validation_audit_log.payload_hash IS 'SHA-256 hash of the validated payload for deduplication tracking';
COMMENT ON COLUMN schema_service.validation_audit_log.duration_ms IS 'Wall-clock validation duration in milliseconds';
COMMENT ON COLUMN schema_service.validation_audit_log.profile IS 'Validation profile used (standard, strict, lenient, performance)';

COMMENT ON COLUMN schema_service.schema_cache_metadata.cache_key IS 'Unique cache key derived from schema_id, version, and compilation options';
COMMENT ON COLUMN schema_service.schema_cache_metadata.ir_hash IS 'SHA-256 hash of the compiled intermediate representation';
COMMENT ON COLUMN schema_service.schema_cache_metadata.access_count IS 'Number of cache accesses for warmup prioritization';
COMMENT ON COLUMN schema_service.schema_cache_metadata.size_bytes IS 'Size of the cached IR in bytes for capacity planning';
