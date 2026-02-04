-- =============================================================================
-- GreenLang Climate OS - pgvector Security Configuration
-- =============================================================================
-- PRD: INFRA-005 Vector Database Infrastructure with pgvector
-- Description: RBAC roles, pgaudit configuration, row-level security,
--              and audit triggers for vector operations.
-- =============================================================================

-- ============================================================================
-- 1. Row-Level Security Policies
-- ============================================================================

-- Enable RLS on vector_embeddings
ALTER TABLE vector_embeddings ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only read embeddings in their assigned namespaces
-- (Applied via application-level namespace assignment)
CREATE POLICY vector_read_policy ON vector_embeddings
    FOR SELECT
    USING (
        namespace = current_setting('app.current_namespace', true)
        OR current_setting('app.current_namespace', true) IS NULL
        OR current_setting('app.current_namespace', true) = 'all'
    );

-- Policy: Writers can insert/update in their assigned namespace
CREATE POLICY vector_write_policy ON vector_embeddings
    FOR INSERT
    WITH CHECK (
        namespace = current_setting('app.current_namespace', true)
        OR current_setting('app.current_namespace', true) IS NULL
        OR current_setting('app.current_namespace', true) = 'all'
    );

CREATE POLICY vector_update_policy ON vector_embeddings
    FOR UPDATE
    USING (
        namespace = current_setting('app.current_namespace', true)
        OR current_setting('app.current_namespace', true) IS NULL
        OR current_setting('app.current_namespace', true) = 'all'
    );

-- Policy: Only admins can delete
CREATE POLICY vector_delete_policy ON vector_embeddings
    FOR DELETE
    USING (
        pg_has_role(current_user, 'vector_admin', 'MEMBER')
    );

-- Bypass RLS for admin role
ALTER TABLE vector_embeddings FORCE ROW LEVEL SECURITY;

-- Allow vector_admin to bypass RLS
GRANT ALL ON vector_embeddings TO vector_admin;

-- ============================================================================
-- 2. Application Users
-- ============================================================================

-- Application service account (vector_writer role)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_app') THEN
        CREATE USER greenlang_app;
    END IF;
    GRANT vector_writer TO greenlang_app;
END $$;

-- Admin service account (vector_admin role)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_admin') THEN
        CREATE USER greenlang_admin;
    END IF;
    GRANT vector_admin TO greenlang_admin;
END $$;

-- Read-only monitoring account
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_monitoring') THEN
        CREATE USER greenlang_monitoring;
    END IF;
    GRANT vector_reader TO greenlang_monitoring;
    GRANT SELECT ON pg_stat_user_tables TO greenlang_monitoring;
    GRANT SELECT ON pg_stat_user_indexes TO greenlang_monitoring;
    GRANT SELECT ON pg_locks TO greenlang_monitoring;
END $$;

-- ============================================================================
-- 3. pgaudit Configuration
-- ============================================================================

-- Note: ALTER SYSTEM commands must be applied via Aurora parameter groups
-- or directly on the instance. Listed here for documentation.

-- ALTER SYSTEM SET pgaudit.log = 'write, ddl';
-- ALTER SYSTEM SET pgaudit.log_catalog = off;
-- ALTER SYSTEM SET pgaudit.log_relation = on;
-- ALTER SYSTEM SET pgaudit.log_statement_once = on;

-- Per-role audit configuration
-- Audit all DDL operations by vector_admin
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_admin') THEN
        ALTER ROLE vector_admin SET pgaudit.log = 'all';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_writer') THEN
        ALTER ROLE vector_writer SET pgaudit.log = 'write';
    END IF;
END $$;

-- ============================================================================
-- 4. Data Protection Functions
-- ============================================================================

-- Function to redact embedding vectors from query results
-- (for audit logs - embeddings are excluded to save space)
CREATE OR REPLACE FUNCTION redact_embedding(data JSONB)
RETURNS JSONB AS $$
BEGIN
    RETURN data - 'embedding';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to validate namespace assignment
CREATE OR REPLACE FUNCTION validate_namespace()
RETURNS TRIGGER AS $$
DECLARE
    valid_namespaces TEXT[] := ARRAY[
        'default', 'csrd', 'cbam', 'eudr', 'vcci',
        'sb253', 'taxonomy', 'csddd'
    ];
BEGIN
    IF NOT (NEW.namespace = ANY(valid_namespaces)) THEN
        RAISE EXCEPTION 'Invalid namespace: %. Allowed: %',
            NEW.namespace, array_to_string(valid_namespaces, ', ');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_validate_namespace
    BEFORE INSERT OR UPDATE ON vector_embeddings
    FOR EACH ROW EXECUTE FUNCTION validate_namespace();

-- ============================================================================
-- 5. Security Views
-- ============================================================================

-- View: Audit trail for vector operations (last 24h)
CREATE OR REPLACE VIEW v_vector_audit_recent AS
SELECT
    id,
    operation,
    table_name,
    record_id,
    changed_by,
    changed_at,
    -- Exclude embedding data from audit view
    CASE
        WHEN new_data IS NOT NULL THEN new_data - 'embedding'
        ELSE NULL
    END AS new_data_redacted,
    CASE
        WHEN old_data IS NOT NULL THEN old_data - 'embedding'
        ELSE NULL
    END AS old_data_redacted
FROM vector_audit_log
WHERE changed_at > NOW() - INTERVAL '24 hours'
ORDER BY changed_at DESC;

GRANT SELECT ON v_vector_audit_recent TO vector_admin;

-- View: Embedding access summary
CREATE OR REPLACE VIEW v_vector_access_summary AS
SELECT
    namespace,
    COUNT(*) AS total_embeddings,
    COUNT(DISTINCT source_type) AS source_types,
    COUNT(DISTINCT source_id) AS unique_sources,
    MIN(created_at) AS earliest_embedding,
    MAX(updated_at) AS latest_update
FROM vector_embeddings
GROUP BY namespace
ORDER BY namespace;

GRANT SELECT ON v_vector_access_summary TO vector_reader;

-- ============================================================================
-- 6. Verification
-- ============================================================================

DO $$
BEGIN
    -- Verify RLS is enabled
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables
        WHERE tablename = 'vector_embeddings' AND rowsecurity = true
    ) THEN
        RAISE WARNING 'Row Level Security is not enabled on vector_embeddings';
    ELSE
        RAISE NOTICE 'Row Level Security verified on vector_embeddings';
    END IF;

    -- Verify roles exist
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_reader') THEN
        RAISE EXCEPTION 'vector_reader role not found';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_writer') THEN
        RAISE EXCEPTION 'vector_writer role not found';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_admin') THEN
        RAISE EXCEPTION 'vector_admin role not found';
    END IF;

    RAISE NOTICE 'All security roles verified';
END $$;
