-- =============================================================================
-- V221 DOWN: Drop PACK-030 role-based RLS policies
-- =============================================================================
-- Note: Only drops the role-based policies added in V221.
-- Base tenant_isolation and service_bypass policies remain from V211-V218.

-- Reports role policies
DROP POLICY IF EXISTS p030_rpt_reader_access ON pack030_nz_reporting.gl_nz_reports;
DROP POLICY IF EXISTS p030_rpt_editor_access ON pack030_nz_reporting.gl_nz_reports;
DROP POLICY IF EXISTS p030_rpt_approver_access ON pack030_nz_reporting.gl_nz_reports;
DROP POLICY IF EXISTS p030_rpt_auditor_access ON pack030_nz_reporting.gl_nz_reports;

-- Sections role policies
DROP POLICY IF EXISTS p030_sec_reader_access ON pack030_nz_reporting.gl_nz_report_sections;
DROP POLICY IF EXISTS p030_sec_editor_access ON pack030_nz_reporting.gl_nz_report_sections;
DROP POLICY IF EXISTS p030_sec_auditor_access ON pack030_nz_reporting.gl_nz_report_sections;

-- Metrics role policies
DROP POLICY IF EXISTS p030_met_reader_access ON pack030_nz_reporting.gl_nz_report_metrics;
DROP POLICY IF EXISTS p030_met_editor_access ON pack030_nz_reporting.gl_nz_report_metrics;
DROP POLICY IF EXISTS p030_met_auditor_access ON pack030_nz_reporting.gl_nz_report_metrics;

-- Assurance evidence role policies
DROP POLICY IF EXISTS p030_ae_auditor_full_access ON pack030_nz_reporting.gl_nz_assurance_evidence;
DROP POLICY IF EXISTS p030_ae_editor_access ON pack030_nz_reporting.gl_nz_assurance_evidence;

-- Audit trail role policies
DROP POLICY IF EXISTS p030_at_auditor_access ON pack030_nz_reporting.gl_nz_audit_trail;
DROP POLICY IF EXISTS p030_at_approver_access ON pack030_nz_reporting.gl_nz_audit_trail;

-- XBRL tags role policies
DROP POLICY IF EXISTS p030_xb_editor_access ON pack030_nz_reporting.gl_nz_xbrl_tags;
DROP POLICY IF EXISTS p030_xb_auditor_access ON pack030_nz_reporting.gl_nz_xbrl_tags;

-- Validation results role policies
DROP POLICY IF EXISTS p030_vr_editor_access ON pack030_nz_reporting.gl_nz_validation_results;
DROP POLICY IF EXISTS p030_vr_auditor_access ON pack030_nz_reporting.gl_nz_validation_results;

-- Dashboard views role policies
DROP POLICY IF EXISTS p030_dv_reader_access ON pack030_nz_reporting.gl_nz_dashboard_views;
DROP POLICY IF EXISTS p030_dv_editor_access ON pack030_nz_reporting.gl_nz_dashboard_views;
