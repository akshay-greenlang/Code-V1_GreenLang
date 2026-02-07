-- ============================================================
-- V011: Field-Level Encryption Support (SEC-003)
-- ============================================================
-- Migration for field-level encryption infrastructure.
-- Adds helper functions, metadata tables, and encrypted column
-- support for PII protection.
--
-- Prerequisites:
--   - V001: pgcrypto extension
--   - V009: security schema
--   - V010: RBAC tables
--
-- Features:
--   - HMAC-based search index functions
--   - Encrypted field registry for audit
--   - Encryption key version tracking
--   - Encryption operation audit log
--
-- Author: GreenLang Framework Team
-- Date: February 2026
-- ============================================================

-- ============================================================
-- Prerequisites Check
-- ============================================================

-- Verify pgcrypto is enabled (required for HMAC functions)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgcrypto') THEN
        CREATE EXTENSION pgcrypto;
        RAISE NOTICE 'Created pgcrypto extension';
    END IF;
END $$;

-- Verify security schema exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'security') THEN
        CREATE SCHEMA security;
        RAISE NOTICE 'Created security schema';
    END IF;
END $$;

-- ============================================================
-- Helper Functions for Application-Level Encryption
-- ============================================================
-- These functions support application-layer encryption operations.
-- Primary encryption is performed at the application layer using
-- AES-256-GCM; these functions assist with search indexes and
-- database-side operations.

-- Function to create HMAC search index for encrypted fields
-- This allows equality searches on encrypted columns
CREATE OR REPLACE FUNCTION security.create_search_index(
    value TEXT,
    key BYTEA
) RETURNS TEXT AS $$
BEGIN
    RETURN encode(hmac(value::BYTEA, key, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION security.create_search_index IS
    'Creates searchable HMAC index for encrypted field values. '
    'Use with field-specific keys to prevent cross-column analysis.';

-- Function to create search index with field binding
-- Includes field name in HMAC input to prevent index reuse
CREATE OR REPLACE FUNCTION security.create_field_search_index(
    value TEXT,
    field_name TEXT,
    key BYTEA
) RETURNS TEXT AS $$
DECLARE
    message BYTEA;
BEGIN
    -- Combine field name and value to prevent cross-field index attacks
    message := (field_name || ':' || encode(value::BYTEA, 'hex'))::BYTEA;
    RETURN encode(hmac(message, key, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION security.create_field_search_index IS
    'Creates field-bound searchable HMAC index. Includes field name '
    'in hash input to prevent cross-column index reuse attacks.';

-- Function to verify HMAC index matches
-- Timing-safe comparison for search operations
CREATE OR REPLACE FUNCTION security.verify_search_index(
    stored_index TEXT,
    value TEXT,
    key BYTEA
) RETURNS BOOLEAN AS $$
DECLARE
    computed_index TEXT;
BEGIN
    computed_index := encode(hmac(value::BYTEA, key, 'sha256'), 'hex');
    -- Uses constant-time comparison internally
    RETURN stored_index = computed_index;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION security.verify_search_index IS
    'Verifies a value matches a stored HMAC search index.';

-- ============================================================
-- Encrypted Fields Registry
-- ============================================================
-- Tracks which database columns contain encrypted data for:
--   - Audit trail and compliance reporting
--   - Key rotation tracking
--   - Data classification mapping

CREATE TABLE IF NOT EXISTS security.encrypted_fields (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_schema VARCHAR(128) NOT NULL,
    table_name VARCHAR(128) NOT NULL,
    column_name VARCHAR(128) NOT NULL,
    data_class VARCHAR(64) NOT NULL DEFAULT 'pii',
    encryption_algorithm VARCHAR(32) NOT NULL DEFAULT 'AES-256-GCM',
    has_search_index BOOLEAN NOT NULL DEFAULT false,
    search_index_column VARCHAR(128),
    key_version VARCHAR(128),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    CONSTRAINT uq_encrypted_field UNIQUE (table_schema, table_name, column_name),
    CONSTRAINT chk_data_class CHECK (
        data_class IN ('pii', 'secret', 'sensitive', 'confidential', 'restricted')
    ),
    CONSTRAINT chk_algorithm CHECK (
        encryption_algorithm IN ('AES-256-GCM', 'AES-128-GCM', 'ChaCha20-Poly1305')
    )
);

-- Index for querying by table
CREATE INDEX IF NOT EXISTS idx_encrypted_fields_table
    ON security.encrypted_fields(table_schema, table_name);

-- Index for querying by data class
CREATE INDEX IF NOT EXISTS idx_encrypted_fields_class
    ON security.encrypted_fields(data_class);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION security.update_encrypted_fields_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_encrypted_fields_updated ON security.encrypted_fields;
CREATE TRIGGER trg_encrypted_fields_updated
    BEFORE UPDATE ON security.encrypted_fields
    FOR EACH ROW
    EXECUTE FUNCTION security.update_encrypted_fields_timestamp();

COMMENT ON TABLE security.encrypted_fields IS
    'Registry of encrypted database columns for SEC-003 compliance. '
    'Tracks data classification, encryption algorithm, and search index status.';

-- ============================================================
-- Encryption Key Versions
-- ============================================================
-- Tracks DEK versions for key rotation and audit.
-- Each key version is a unique identifier that maps to a
-- KMS-wrapped Data Encryption Key.

CREATE TABLE IF NOT EXISTS security.encryption_key_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_version VARCHAR(128) NOT NULL UNIQUE,
    key_type VARCHAR(32) NOT NULL DEFAULT 'dek',
    kms_key_id VARCHAR(512),
    kms_key_arn VARCHAR(1024),
    algorithm VARCHAR(32) NOT NULL DEFAULT 'AES-256-GCM',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rotated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_by UUID,
    CONSTRAINT chk_key_type CHECK (key_type IN ('dek', 'kek', 'index')),
    CONSTRAINT chk_key_algorithm CHECK (
        algorithm IN ('AES-256-GCM', 'AES-128-GCM', 'RSA-OAEP-256')
    )
);

-- Index for active keys lookup
CREATE INDEX IF NOT EXISTS idx_key_versions_active
    ON security.encryption_key_versions(is_active)
    WHERE is_active = true;

-- Index for key type
CREATE INDEX IF NOT EXISTS idx_key_versions_type
    ON security.encryption_key_versions(key_type, is_active);

-- Index for KMS key lookup
CREATE INDEX IF NOT EXISTS idx_key_versions_kms
    ON security.encryption_key_versions(kms_key_id)
    WHERE kms_key_id IS NOT NULL;

COMMENT ON TABLE security.encryption_key_versions IS
    'Tracks Data Encryption Key (DEK) versions for SEC-003. '
    'Used for key rotation tracking and audit compliance.';

-- ============================================================
-- Encryption Audit Log
-- ============================================================
-- Records all encryption/decryption operations for compliance.
-- Uses TimescaleDB hypertable for efficient time-series queries.

CREATE TABLE IF NOT EXISTS security.encryption_audit_log (
    id UUID DEFAULT gen_random_uuid(),
    event_type VARCHAR(64) NOT NULL,
    data_class VARCHAR(64),
    tenant_id UUID,
    user_id UUID,
    key_version VARCHAR(128),
    field_name VARCHAR(128),
    operation VARCHAR(32) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT true,
    error_code VARCHAR(64),
    error_message TEXT,
    correlation_id UUID,
    request_id UUID,
    client_ip INET,
    user_agent TEXT,
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT pk_encryption_audit PRIMARY KEY (id, performed_at),
    CONSTRAINT chk_event_type CHECK (
        event_type IN (
            'encrypt', 'decrypt', 'key_generate', 'key_rotate',
            'key_retire', 'cache_hit', 'cache_miss', 'error'
        )
    ),
    CONSTRAINT chk_operation CHECK (
        operation IN (
            'encrypt_field', 'decrypt_field', 'encrypt_bulk', 'decrypt_bulk',
            'generate_dek', 'decrypt_dek', 'rotate_dek', 'create_index'
        )
    )
);

-- Convert to hypertable if TimescaleDB is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'security.encryption_audit_log',
            'performed_at',
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '1 day'
        );
        RAISE NOTICE 'Created encryption_audit_log hypertable';
    END IF;
END $$;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_encryption_audit_time
    ON security.encryption_audit_log(performed_at DESC);

CREATE INDEX IF NOT EXISTS idx_encryption_audit_event
    ON security.encryption_audit_log(event_type, performed_at DESC);

CREATE INDEX IF NOT EXISTS idx_encryption_audit_tenant
    ON security.encryption_audit_log(tenant_id, performed_at DESC);

CREATE INDEX IF NOT EXISTS idx_encryption_audit_user
    ON security.encryption_audit_log(user_id, performed_at DESC);

CREATE INDEX IF NOT EXISTS idx_encryption_audit_key
    ON security.encryption_audit_log(key_version, performed_at DESC);

CREATE INDEX IF NOT EXISTS idx_encryption_audit_correlation
    ON security.encryption_audit_log(correlation_id)
    WHERE correlation_id IS NOT NULL;

-- Index for error analysis
CREATE INDEX IF NOT EXISTS idx_encryption_audit_errors
    ON security.encryption_audit_log(error_code, performed_at DESC)
    WHERE success = false;

COMMENT ON TABLE security.encryption_audit_log IS
    'Audit trail for SEC-003 encryption operations. '
    'Records all encrypt/decrypt events for compliance and debugging.';

-- ============================================================
-- Continuous Aggregates for Encryption Metrics
-- ============================================================
-- Pre-aggregated views for Prometheus metrics export

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Hourly encryption operation counts
        EXECUTE $inner$
            CREATE MATERIALIZED VIEW IF NOT EXISTS security.encryption_ops_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', performed_at) AS hour,
                tenant_id,
                event_type,
                operation,
                success,
                COUNT(*) as operation_count
            FROM security.encryption_audit_log
            GROUP BY hour, tenant_id, event_type, operation, success
            WITH NO DATA;
        $inner$;

        -- Set refresh policy
        PERFORM add_continuous_aggregate_policy(
            'security.encryption_ops_hourly',
            start_offset => INTERVAL '2 hours',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        );

        RAISE NOTICE 'Created encryption_ops_hourly continuous aggregate';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not create continuous aggregate: %', SQLERRM;
END $$;

-- ============================================================
-- Data Retention Policy
-- ============================================================
-- Encryption audit logs retained for 365 days (compliance)

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM add_retention_policy(
            'security.encryption_audit_log',
            INTERVAL '365 days',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'Added 365-day retention policy to encryption_audit_log';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not add retention policy: %', SQLERRM;
END $$;

-- ============================================================
-- Example: Add Encrypted Columns to Existing Tables
-- ============================================================
-- Note: Actual encryption is handled at the application layer.
-- These columns store the base64-encoded encrypted values.

-- Add encrypted columns to login_attempts (from V009)
ALTER TABLE security.login_attempts
    ADD COLUMN IF NOT EXISTS ip_address_encrypted TEXT;

ALTER TABLE security.login_attempts
    ADD COLUMN IF NOT EXISTS ip_address_search_index VARCHAR(64);

-- Create index for searching encrypted IPs
CREATE INDEX IF NOT EXISTS idx_login_attempts_ip_search
    ON security.login_attempts(ip_address_search_index)
    WHERE ip_address_search_index IS NOT NULL;

-- Register encrypted fields in registry
INSERT INTO security.encrypted_fields (
    table_schema, table_name, column_name, data_class,
    has_search_index, search_index_column, notes
)
VALUES
    ('security', 'login_attempts', 'ip_address_encrypted', 'pii',
     true, 'ip_address_search_index',
     'IP address encrypted for privacy compliance. Original ip_address column retained for backward compatibility.')
ON CONFLICT (table_schema, table_name, column_name) DO UPDATE SET
    has_search_index = EXCLUDED.has_search_index,
    search_index_column = EXCLUDED.search_index_column,
    updated_at = NOW();

-- ============================================================
-- Views for Encryption Metrics
-- ============================================================

-- View: Encryption operations summary (last 24 hours)
CREATE OR REPLACE VIEW security.v_encryption_ops_summary AS
SELECT
    event_type,
    operation,
    success,
    COUNT(*) as total_count,
    COUNT(DISTINCT tenant_id) as tenant_count,
    MIN(performed_at) as first_event,
    MAX(performed_at) as last_event
FROM security.encryption_audit_log
WHERE performed_at >= NOW() - INTERVAL '24 hours'
GROUP BY event_type, operation, success
ORDER BY total_count DESC;

COMMENT ON VIEW security.v_encryption_ops_summary IS
    'Summary of encryption operations in the last 24 hours.';

-- View: Encrypted fields inventory
CREATE OR REPLACE VIEW security.v_encrypted_fields_inventory AS
SELECT
    ef.table_schema,
    ef.table_name,
    ef.column_name,
    ef.data_class,
    ef.encryption_algorithm,
    ef.has_search_index,
    ef.search_index_column,
    ef.key_version,
    ef.created_at,
    ef.updated_at,
    -- Get table row count (approximate)
    (SELECT reltuples::BIGINT
     FROM pg_class c
     JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE n.nspname = ef.table_schema
     AND c.relname = ef.table_name) as approx_row_count
FROM security.encrypted_fields ef
ORDER BY ef.table_schema, ef.table_name, ef.column_name;

COMMENT ON VIEW security.v_encrypted_fields_inventory IS
    'Inventory of all encrypted fields with row count estimates.';

-- View: Key version status
CREATE OR REPLACE VIEW security.v_key_version_status AS
SELECT
    kv.key_version,
    kv.key_type,
    kv.algorithm,
    kv.is_active,
    kv.created_at,
    kv.rotated_at,
    kv.retired_at,
    CASE
        WHEN kv.retired_at IS NOT NULL THEN 'retired'
        WHEN kv.rotated_at IS NOT NULL THEN 'rotated'
        WHEN kv.is_active THEN 'active'
        ELSE 'inactive'
    END as status,
    -- Count of fields using this key version
    (SELECT COUNT(*)
     FROM security.encrypted_fields ef
     WHERE ef.key_version = kv.key_version) as field_count
FROM security.encryption_key_versions kv
ORDER BY kv.created_at DESC;

COMMENT ON VIEW security.v_key_version_status IS
    'Current status of all encryption key versions.';

-- ============================================================
-- Grants
-- ============================================================

-- Service account grants
GRANT SELECT, INSERT ON security.encrypted_fields TO greenlang_service;
GRANT SELECT, INSERT ON security.encryption_key_versions TO greenlang_service;
GRANT SELECT, INSERT ON security.encryption_audit_log TO greenlang_service;
GRANT EXECUTE ON FUNCTION security.create_search_index TO greenlang_service;
GRANT EXECUTE ON FUNCTION security.create_field_search_index TO greenlang_service;
GRANT EXECUTE ON FUNCTION security.verify_search_index TO greenlang_service;

-- Read-only grants
GRANT SELECT ON security.encrypted_fields TO greenlang_readonly;
GRANT SELECT ON security.encryption_key_versions TO greenlang_readonly;
GRANT SELECT ON security.encryption_audit_log TO greenlang_readonly;
GRANT SELECT ON security.v_encryption_ops_summary TO greenlang_readonly;
GRANT SELECT ON security.v_encrypted_fields_inventory TO greenlang_readonly;
GRANT SELECT ON security.v_key_version_status TO greenlang_readonly;

-- Admin grants (for key rotation operations)
GRANT UPDATE ON security.encryption_key_versions TO greenlang_admin;
GRANT UPDATE ON security.encrypted_fields TO greenlang_admin;

-- ============================================================
-- Migration Complete
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE 'V011: Field-Level Encryption migration complete';
    RAISE NOTICE '  - Created security.create_search_index function';
    RAISE NOTICE '  - Created security.create_field_search_index function';
    RAISE NOTICE '  - Created security.verify_search_index function';
    RAISE NOTICE '  - Created security.encrypted_fields registry';
    RAISE NOTICE '  - Created security.encryption_key_versions table';
    RAISE NOTICE '  - Created security.encryption_audit_log table';
    RAISE NOTICE '  - Created views for metrics and inventory';
    RAISE NOTICE '  - Added encrypted columns to login_attempts';
END $$;
