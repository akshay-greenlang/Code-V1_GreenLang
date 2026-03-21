-- =============================================================================
-- V225 DOWN: Revoke PACK-030 RBAC permissions
-- =============================================================================

-- Revoke default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE ALL PRIVILEGES ON FUNCTIONS FROM greenlang_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE ALL PRIVILEGES ON FUNCTIONS FROM greenlang_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE EXECUTE ON FUNCTIONS FROM greenlang_auditor;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE EXECUTE ON FUNCTIONS FROM greenlang_editor;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE EXECUTE ON FUNCTIONS FROM greenlang_reader;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE ALL PRIVILEGES ON TABLES FROM greenlang_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE ALL PRIVILEGES ON TABLES FROM greenlang_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE SELECT ON TABLES FROM greenlang_auditor;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM greenlang_editor;
ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    REVOKE SELECT ON TABLES FROM greenlang_reader;

-- Revoke all privileges on schema
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting FROM greenlang_service;
REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA pack030_nz_reporting FROM greenlang_service;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pack030_nz_reporting FROM greenlang_service;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting FROM greenlang_admin;
REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA pack030_nz_reporting FROM greenlang_admin;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pack030_nz_reporting FROM greenlang_admin;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting FROM greenlang_auditor;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting FROM greenlang_approver;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting FROM greenlang_editor;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting FROM greenlang_reader;

-- Revoke schema usage
REVOKE USAGE ON SCHEMA pack030_nz_reporting FROM greenlang_reader;
REVOKE USAGE ON SCHEMA pack030_nz_reporting FROM greenlang_editor;
REVOKE USAGE ON SCHEMA pack030_nz_reporting FROM greenlang_approver;
REVOKE USAGE ON SCHEMA pack030_nz_reporting FROM greenlang_auditor;
REVOKE USAGE ON SCHEMA pack030_nz_reporting FROM greenlang_admin;
REVOKE USAGE ON SCHEMA pack030_nz_reporting FROM greenlang_service;
