-- =============================================================================
-- GreenLang Climate OS - Database Roles and Permissions
-- =============================================================================
-- File: 10_roles.sql
-- Description: Database roles for application access, read-only analytics,
--              and administrative operations with appropriate permissions.
-- =============================================================================

-- =============================================================================
-- ROLE OVERVIEW
-- =============================================================================
--
-- Role                    Purpose                         Access Level
-- ---------------------   -----------------------------   ----------------
-- greenlang_app           Application service account     Read/Write/Execute
-- greenlang_readonly      Analytics and reporting         Read-only
-- greenlang_admin         Administrative operations       Full access
-- greenlang_migration     Schema migrations               DDL + DML
-- greenlang_backup        Backup operations               Read + Replication
--
-- =============================================================================

-- =============================================================================
-- CREATE ROLES
-- =============================================================================

-- Application role (used by API services)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'greenlang_app') THEN
        CREATE ROLE greenlang_app WITH
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            INHERIT
            NOREPLICATION
            CONNECTION LIMIT 100;
        RAISE NOTICE 'Created role: greenlang_app';
    ELSE
        RAISE NOTICE 'Role greenlang_app already exists';
    END IF;
END $$;

-- Read-only role (for analytics, BI tools)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        CREATE ROLE greenlang_readonly WITH
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            INHERIT
            NOREPLICATION
            CONNECTION LIMIT 50;
        RAISE NOTICE 'Created role: greenlang_readonly';
    ELSE
        RAISE NOTICE 'Role greenlang_readonly already exists';
    END IF;
END $$;

-- Admin role (for DBA operations)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'greenlang_admin') THEN
        CREATE ROLE greenlang_admin WITH
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            CREATEROLE
            INHERIT
            NOREPLICATION
            CONNECTION LIMIT 10;
        RAISE NOTICE 'Created role: greenlang_admin';
    ELSE
        RAISE NOTICE 'Role greenlang_admin already exists';
    END IF;
END $$;

-- Migration role (for Flyway/Liquibase)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'greenlang_migration') THEN
        CREATE ROLE greenlang_migration WITH
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            INHERIT
            NOREPLICATION
            CONNECTION LIMIT 5;
        RAISE NOTICE 'Created role: greenlang_migration';
    ELSE
        RAISE NOTICE 'Role greenlang_migration already exists';
    END IF;
END $$;

-- Backup role (for pg_dump/pg_basebackup)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'greenlang_backup') THEN
        CREATE ROLE greenlang_backup WITH
            LOGIN
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            INHERIT
            REPLICATION
            CONNECTION LIMIT 5;
        RAISE NOTICE 'Created role: greenlang_backup';
    ELSE
        RAISE NOTICE 'Role greenlang_backup already exists';
    END IF;
END $$;

-- =============================================================================
-- SCHEMA PERMISSIONS
-- =============================================================================

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO greenlang_app, greenlang_readonly, greenlang_admin, greenlang_migration;
GRANT USAGE ON SCHEMA metrics TO greenlang_app, greenlang_readonly, greenlang_admin, greenlang_migration;
GRANT USAGE ON SCHEMA audit TO greenlang_app, greenlang_readonly, greenlang_admin, greenlang_migration;
GRANT USAGE ON SCHEMA archive TO greenlang_app, greenlang_readonly, greenlang_admin, greenlang_migration;

-- Grant schema creation for admin and migration roles
GRANT CREATE ON SCHEMA public TO greenlang_admin, greenlang_migration;
GRANT CREATE ON SCHEMA metrics TO greenlang_admin, greenlang_migration;
GRANT CREATE ON SCHEMA audit TO greenlang_admin, greenlang_migration;
GRANT CREATE ON SCHEMA archive TO greenlang_admin, greenlang_migration;

-- Backup role needs usage only
GRANT USAGE ON SCHEMA public, metrics, audit, archive TO greenlang_backup;

-- =============================================================================
-- PUBLIC SCHEMA PERMISSIONS
-- =============================================================================

-- Application role: full CRUD on application tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO greenlang_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO greenlang_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO greenlang_app;

-- Read-only role: SELECT only
GRANT SELECT ON ALL TABLES IN SCHEMA public TO greenlang_readonly;

-- Admin role: full access
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO greenlang_admin;

-- Migration role: DDL + DML
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO greenlang_migration;

-- Backup role: read access
GRANT SELECT ON ALL TABLES IN SCHEMA public TO greenlang_backup;

-- =============================================================================
-- METRICS SCHEMA PERMISSIONS
-- =============================================================================

-- Application role: full CRUD on metrics tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA metrics TO greenlang_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA metrics TO greenlang_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA metrics TO greenlang_app;

-- Read-only role: SELECT only
GRANT SELECT ON ALL TABLES IN SCHEMA metrics TO greenlang_readonly;

-- Admin role: full access
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA metrics TO greenlang_admin;

-- Migration role: DDL + DML
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA metrics TO greenlang_migration;

-- Backup role: read access
GRANT SELECT ON ALL TABLES IN SCHEMA metrics TO greenlang_backup;

-- =============================================================================
-- AUDIT SCHEMA PERMISSIONS
-- =============================================================================

-- Application role: INSERT only (audit logs are immutable)
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO greenlang_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO greenlang_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA audit TO greenlang_app;

-- Read-only role: SELECT only
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO greenlang_readonly;

-- Admin role: SELECT, INSERT (no UPDATE/DELETE for compliance)
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA audit TO greenlang_admin;

-- Migration role: DDL + SELECT/INSERT (structure changes only)
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA audit TO greenlang_migration;

-- Backup role: read access
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO greenlang_backup;

-- =============================================================================
-- ARCHIVE SCHEMA PERMISSIONS
-- =============================================================================

-- Application role: SELECT, INSERT (archive is write-once)
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA archive TO greenlang_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA archive TO greenlang_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA archive TO greenlang_app;

-- Read-only role: SELECT only
GRANT SELECT ON ALL TABLES IN SCHEMA archive TO greenlang_readonly;

-- Admin role: full access (may need to manage archives)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA archive TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA archive TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA archive TO greenlang_admin;

-- Migration role: DDL + SELECT/INSERT
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA archive TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA archive TO greenlang_migration;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA archive TO greenlang_migration;

-- Backup role: read access
GRANT SELECT ON ALL TABLES IN SCHEMA archive TO greenlang_backup;

-- =============================================================================
-- DEFAULT PRIVILEGES FOR FUTURE OBJECTS
-- =============================================================================

-- Set default privileges for objects created by migration role
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO greenlang_app;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA public
    GRANT SELECT ON TABLES TO greenlang_readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO greenlang_app;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA public
    GRANT EXECUTE ON FUNCTIONS TO greenlang_app;

ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA metrics
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO greenlang_app;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA metrics
    GRANT SELECT ON TABLES TO greenlang_readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA metrics
    GRANT USAGE, SELECT ON SEQUENCES TO greenlang_app;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA metrics
    GRANT EXECUTE ON FUNCTIONS TO greenlang_app;

ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA audit
    GRANT SELECT, INSERT ON TABLES TO greenlang_app;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA audit
    GRANT SELECT ON TABLES TO greenlang_readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA audit
    GRANT USAGE, SELECT ON SEQUENCES TO greenlang_app;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA audit
    GRANT EXECUTE ON FUNCTIONS TO greenlang_app;

ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA archive
    GRANT SELECT, INSERT ON TABLES TO greenlang_app;
ALTER DEFAULT PRIVILEGES FOR ROLE greenlang_migration IN SCHEMA archive
    GRANT SELECT ON TABLES TO greenlang_readonly;

-- =============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================================================

-- Enable RLS on multi-tenant tables
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics.emission_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics.devices ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics.emission_factors ENABLE ROW LEVEL SECURITY;

-- Note: Hypertables require special handling for RLS
-- The actual RLS policies should be defined based on the application's
-- session variable mechanism (e.g., SET app.current_org_id = 'uuid')

-- Policy for organizations (users can only see their own org)
CREATE POLICY org_isolation ON public.organizations
    FOR ALL
    TO greenlang_app
    USING (id = current_setting('app.current_org_id', TRUE)::UUID);

-- Policy for users (users can only see users in their org)
CREATE POLICY user_org_isolation ON public.users
    FOR ALL
    TO greenlang_app
    USING (org_id = current_setting('app.current_org_id', TRUE)::UUID);

-- Policy for projects
CREATE POLICY project_org_isolation ON public.projects
    FOR ALL
    TO greenlang_app
    USING (org_id = current_setting('app.current_org_id', TRUE)::UUID);

-- Policy for API keys (users can only see their own keys)
CREATE POLICY api_key_user_isolation ON public.api_keys
    FOR ALL
    TO greenlang_app
    USING (
        user_id IN (
            SELECT id FROM public.users
            WHERE org_id = current_setting('app.current_org_id', TRUE)::UUID
        )
    );

-- Policy for emission sources
CREATE POLICY emission_source_org_isolation ON metrics.emission_sources
    FOR ALL
    TO greenlang_app
    USING (org_id = current_setting('app.current_org_id', TRUE)::UUID);

-- Policy for devices
CREATE POLICY device_org_isolation ON metrics.devices
    FOR ALL
    TO greenlang_app
    USING (org_id = current_setting('app.current_org_id', TRUE)::UUID);

-- Emission factors are shared (public data)
CREATE POLICY emission_factors_public ON metrics.emission_factors
    FOR SELECT
    TO greenlang_app
    USING (TRUE);

-- Admin role bypasses RLS
ALTER TABLE public.organizations FORCE ROW LEVEL SECURITY;
ALTER TABLE public.users FORCE ROW LEVEL SECURITY;
ALTER TABLE public.projects FORCE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys FORCE ROW LEVEL SECURITY;
ALTER TABLE metrics.emission_sources FORCE ROW LEVEL SECURITY;
ALTER TABLE metrics.devices FORCE ROW LEVEL SECURITY;

-- Grant bypass to admin role
GRANT BYPASSRLS ON ROLE greenlang_admin TO greenlang_admin;

-- =============================================================================
-- HELPER FUNCTIONS FOR SESSION MANAGEMENT
-- =============================================================================

-- Function to set current organization context
CREATE OR REPLACE FUNCTION public.set_current_org(p_org_id UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_org_id', p_org_id::TEXT, FALSE);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION public.set_current_org TO greenlang_app;

COMMENT ON FUNCTION public.set_current_org IS 'Set the current organization context for RLS policies';

-- Function to set current user context
CREATE OR REPLACE FUNCTION public.set_current_user_context(p_user_id UUID, p_org_id UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_user_id', p_user_id::TEXT, FALSE);
    PERFORM set_config('app.current_org_id', p_org_id::TEXT, FALSE);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION public.set_current_user_context TO greenlang_app;

COMMENT ON FUNCTION public.set_current_user_context IS 'Set the current user and organization context for RLS policies';

-- Function to clear context (for connection pooling)
CREATE OR REPLACE FUNCTION public.clear_context()
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_user_id', '', FALSE);
    PERFORM set_config('app.current_org_id', '', FALSE);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION public.clear_context TO greenlang_app;

COMMENT ON FUNCTION public.clear_context IS 'Clear the current session context (call before returning connection to pool)';

-- =============================================================================
-- PASSWORD MANAGEMENT
-- =============================================================================

-- Note: Passwords should be set via secure mechanism (vault, secrets manager)
-- These are placeholder commands to be replaced with actual passwords

-- Example (DO NOT USE IN PRODUCTION):
-- ALTER ROLE greenlang_app WITH PASSWORD 'CHANGE_ME_FROM_VAULT';
-- ALTER ROLE greenlang_readonly WITH PASSWORD 'CHANGE_ME_FROM_VAULT';
-- ALTER ROLE greenlang_admin WITH PASSWORD 'CHANGE_ME_FROM_VAULT';
-- ALTER ROLE greenlang_migration WITH PASSWORD 'CHANGE_ME_FROM_VAULT';
-- ALTER ROLE greenlang_backup WITH PASSWORD 'CHANGE_ME_FROM_VAULT';

-- =============================================================================
-- PERMISSION VALIDATION
-- =============================================================================

-- Function to validate role permissions
CREATE OR REPLACE FUNCTION public.validate_role_permissions()
RETURNS TABLE (
    role_name TEXT,
    schema_name TEXT,
    table_name TEXT,
    privilege TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        grantee::TEXT,
        table_schema::TEXT,
        table_name::TEXT,
        privilege_type::TEXT
    FROM information_schema.role_table_grants
    WHERE grantee IN ('greenlang_app', 'greenlang_readonly', 'greenlang_admin', 'greenlang_migration', 'greenlang_backup')
      AND table_schema IN ('public', 'metrics', 'audit', 'archive')
    ORDER BY grantee, table_schema, table_name, privilege_type;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION public.validate_role_permissions IS 'List all table permissions for GreenLang roles';

-- =============================================================================
-- SUMMARY
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'GreenLang Database Roles Summary';
    RAISE NOTICE '=============================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Role                 Connection Limit   Purpose';
    RAISE NOTICE '-------------------- ----------------   ---------------------';
    RAISE NOTICE 'greenlang_app        100                API service account';
    RAISE NOTICE 'greenlang_readonly   50                 Analytics/BI access';
    RAISE NOTICE 'greenlang_admin      10                 DBA operations';
    RAISE NOTICE 'greenlang_migration  5                  Schema migrations';
    RAISE NOTICE 'greenlang_backup     5                  Backup operations';
    RAISE NOTICE '';
    RAISE NOTICE 'Row Level Security (RLS) is enabled for multi-tenant tables.';
    RAISE NOTICE 'Use set_current_org() or set_current_user_context() to set context.';
    RAISE NOTICE '';
    RAISE NOTICE 'IMPORTANT: Set passwords from secure vault before use!';
    RAISE NOTICE '=============================================================';
END $$;
