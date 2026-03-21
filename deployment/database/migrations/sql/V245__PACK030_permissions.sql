-- =============================================================================
-- V225: PACK-030 Net Zero Reporting Pack - RBAC Permissions
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    015 of 015
-- Date:         March 2026
--
-- Role-based access control (RBAC) permission definitions for PACK-030
-- integrated with the GreenLang RBAC system (SEC-002). Defines permissions
-- for report management, framework operations, narrative management,
-- XBRL operations, assurance packaging, and dashboard access.
--
-- Permissions: 60+ granular permissions across 10 resource categories
-- Roles: 6 predefined role-permission mappings
--
-- Previous: V224__PACK030_seed_data.sql
-- =============================================================================

-- =============================================================================
-- Create roles if they don't exist
-- =============================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_reader') THEN
        CREATE ROLE greenlang_reader;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_editor') THEN
        CREATE ROLE greenlang_editor;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_approver') THEN
        CREATE ROLE greenlang_approver;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_auditor') THEN
        CREATE ROLE greenlang_auditor;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_admin') THEN
        CREATE ROLE greenlang_admin;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_service') THEN
        CREATE ROLE greenlang_service;
    END IF;
END;
$$;

-- =============================================================================
-- Schema-Level Grants
-- =============================================================================
GRANT USAGE ON SCHEMA pack030_nz_reporting TO greenlang_reader;
GRANT USAGE ON SCHEMA pack030_nz_reporting TO greenlang_editor;
GRANT USAGE ON SCHEMA pack030_nz_reporting TO greenlang_approver;
GRANT USAGE ON SCHEMA pack030_nz_reporting TO greenlang_auditor;
GRANT USAGE ON SCHEMA pack030_nz_reporting TO greenlang_admin;
GRANT USAGE ON SCHEMA pack030_nz_reporting TO greenlang_service;

-- =============================================================================
-- Reader Role: Read-only access to published data
-- =============================================================================

-- Reports (read published only - enforced by RLS)
GRANT SELECT ON pack030_nz_reporting.gl_nz_reports TO greenlang_reader;
GRANT SELECT ON pack030_nz_reporting.gl_nz_report_sections TO greenlang_reader;
GRANT SELECT ON pack030_nz_reporting.gl_nz_report_metrics TO greenlang_reader;

-- Framework reference data (read-only)
GRANT SELECT ON pack030_nz_reporting.gl_nz_framework_schemas TO greenlang_reader;
GRANT SELECT ON pack030_nz_reporting.gl_nz_framework_mappings TO greenlang_reader;
GRANT SELECT ON pack030_nz_reporting.gl_nz_framework_deadlines TO greenlang_reader;

-- Narratives (read-only)
GRANT SELECT ON pack030_nz_reporting.gl_nz_narratives TO greenlang_reader;

-- Dashboard (read published/own - enforced by RLS)
GRANT SELECT ON pack030_nz_reporting.gl_nz_dashboard_views TO greenlang_reader;

-- Views (read-only)
GRANT SELECT ON pack030_nz_reporting.v_reports_summary TO greenlang_reader;
GRANT SELECT ON pack030_nz_reporting.v_framework_coverage TO greenlang_reader;
GRANT SELECT ON pack030_nz_reporting.v_upcoming_deadlines TO greenlang_reader;

-- Functions (read-only)
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_report_quality_score(UUID) TO greenlang_reader;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_framework_coverage(UUID) TO greenlang_reader;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_upcoming_deadlines(UUID, INT) TO greenlang_reader;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_report_health_status(UUID) TO greenlang_reader;

-- =============================================================================
-- Editor Role: Create and modify reports
-- =============================================================================

-- Reports (full CRUD)
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_reports TO greenlang_editor;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_report_sections TO greenlang_editor;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_report_metrics TO greenlang_editor;

-- Framework data (read + use)
GRANT SELECT ON pack030_nz_reporting.gl_nz_framework_schemas TO greenlang_editor;
GRANT SELECT, INSERT, UPDATE ON pack030_nz_reporting.gl_nz_framework_mappings TO greenlang_editor;
GRANT SELECT, INSERT, UPDATE ON pack030_nz_reporting.gl_nz_framework_deadlines TO greenlang_editor;

-- Narratives (full CRUD)
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_narratives TO greenlang_editor;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_translations TO greenlang_editor;

-- Assurance evidence (create and modify)
GRANT SELECT, INSERT, UPDATE ON pack030_nz_reporting.gl_nz_assurance_evidence TO greenlang_editor;

-- XBRL tags (create and modify)
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_xbrl_tags TO greenlang_editor;

-- Validation (view and resolve)
GRANT SELECT, INSERT, UPDATE ON pack030_nz_reporting.gl_nz_validation_results TO greenlang_editor;

-- Data lineage (create)
GRANT SELECT, INSERT, UPDATE ON pack030_nz_reporting.gl_nz_data_lineage TO greenlang_editor;

-- Configuration (manage own org)
GRANT SELECT, INSERT, UPDATE ON pack030_nz_reporting.gl_nz_report_config TO greenlang_editor;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_dashboard_views TO greenlang_editor;

-- Audit trail (read + auto-insert via triggers)
GRANT SELECT, INSERT ON pack030_nz_reporting.gl_nz_audit_trail TO greenlang_editor;

-- Views (all)
GRANT SELECT ON pack030_nz_reporting.v_reports_summary TO greenlang_editor;
GRANT SELECT ON pack030_nz_reporting.v_framework_coverage TO greenlang_editor;
GRANT SELECT ON pack030_nz_reporting.v_validation_issues TO greenlang_editor;
GRANT SELECT ON pack030_nz_reporting.v_upcoming_deadlines TO greenlang_editor;
GRANT SELECT ON pack030_nz_reporting.v_lineage_summary TO greenlang_editor;

-- Functions (all)
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_report_quality_score(UUID) TO greenlang_editor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_framework_coverage(UUID) TO greenlang_editor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_data_completeness(UUID) TO greenlang_editor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_generate_provenance_hash(TEXT) TO greenlang_editor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_upcoming_deadlines(UUID, INT) TO greenlang_editor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_consistency_score(UUID, INT) TO greenlang_editor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_check_cross_framework_consistency(UUID, INT, VARCHAR) TO greenlang_editor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_report_health_status(UUID) TO greenlang_editor;

-- =============================================================================
-- Approver Role: Approve and publish reports
-- =============================================================================

-- Full access to reports for approval workflow
GRANT SELECT, INSERT, UPDATE ON pack030_nz_reporting.gl_nz_reports TO greenlang_approver;
GRANT SELECT ON pack030_nz_reporting.gl_nz_report_sections TO greenlang_approver;
GRANT SELECT ON pack030_nz_reporting.gl_nz_report_metrics TO greenlang_approver;

-- Evidence review
GRANT SELECT ON pack030_nz_reporting.gl_nz_assurance_evidence TO greenlang_approver;

-- Validation review
GRANT SELECT ON pack030_nz_reporting.gl_nz_validation_results TO greenlang_approver;

-- Audit trail
GRANT SELECT ON pack030_nz_reporting.gl_nz_audit_trail TO greenlang_approver;

-- Views
GRANT SELECT ON pack030_nz_reporting.v_reports_summary TO greenlang_approver;
GRANT SELECT ON pack030_nz_reporting.v_framework_coverage TO greenlang_approver;
GRANT SELECT ON pack030_nz_reporting.v_validation_issues TO greenlang_approver;
GRANT SELECT ON pack030_nz_reporting.v_upcoming_deadlines TO greenlang_approver;
GRANT SELECT ON pack030_nz_reporting.v_lineage_summary TO greenlang_approver;

-- Functions
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_report_quality_score(UUID) TO greenlang_approver;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_report_health_status(UUID) TO greenlang_approver;

-- =============================================================================
-- Auditor Role: Full read access + evidence management
-- =============================================================================

-- Full read access to all tables
GRANT SELECT ON pack030_nz_reporting.gl_nz_reports TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_report_sections TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_report_metrics TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_framework_schemas TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_framework_mappings TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_framework_deadlines TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_narratives TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_translations TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_xbrl_tags TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_validation_results TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_data_lineage TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.gl_nz_report_config TO greenlang_auditor;

-- Evidence: read + review (update review fields)
GRANT SELECT, UPDATE ON pack030_nz_reporting.gl_nz_assurance_evidence TO greenlang_auditor;

-- Audit trail: full read
GRANT SELECT ON pack030_nz_reporting.gl_nz_audit_trail TO greenlang_auditor;

-- All views
GRANT SELECT ON pack030_nz_reporting.v_reports_summary TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.v_framework_coverage TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.v_validation_issues TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.v_upcoming_deadlines TO greenlang_auditor;
GRANT SELECT ON pack030_nz_reporting.v_lineage_summary TO greenlang_auditor;

-- All read-only functions
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_report_quality_score(UUID) TO greenlang_auditor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_framework_coverage(UUID) TO greenlang_auditor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_data_completeness(UUID) TO greenlang_auditor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_upcoming_deadlines(UUID, INT) TO greenlang_auditor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_consistency_score(UUID, INT) TO greenlang_auditor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_check_cross_framework_consistency(UUID, INT, VARCHAR) TO greenlang_auditor;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_report_health_status(UUID) TO greenlang_auditor;

-- =============================================================================
-- Admin Role: Full access to everything
-- =============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA pack030_nz_reporting TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pack030_nz_reporting TO greenlang_admin;

-- =============================================================================
-- Service Role: Full access (for internal system operations)
-- =============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pack030_nz_reporting TO greenlang_service;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA pack030_nz_reporting TO greenlang_service;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pack030_nz_reporting TO greenlang_service;

-- =============================================================================
-- Default privileges for future objects
-- =============================================================================

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT SELECT ON TABLES TO greenlang_reader;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO greenlang_editor;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT SELECT ON TABLES TO greenlang_auditor;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT ALL PRIVILEGES ON TABLES TO greenlang_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT ALL PRIVILEGES ON TABLES TO greenlang_service;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT EXECUTE ON FUNCTIONS TO greenlang_reader;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT EXECUTE ON FUNCTIONS TO greenlang_editor;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT EXECUTE ON FUNCTIONS TO greenlang_auditor;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT ALL PRIVILEGES ON FUNCTIONS TO greenlang_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA pack030_nz_reporting
    GRANT ALL PRIVILEGES ON FUNCTIONS TO greenlang_service;

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON SCHEMA pack030_nz_reporting IS
    'PACK-030 Net Zero Reporting Pack - Multi-framework climate disclosure reporting with RBAC (6 roles: reader, editor, approver, auditor, admin, service), RLS multi-tenant isolation, 15 tables, 5 views, 350+ indexes, 30 RLS policies, 8 helper functions, 6 audit trigger functions.';
