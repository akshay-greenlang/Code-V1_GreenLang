-- =============================================================================
-- V211 DOWN: Drop PACK-030 core tables, trigger, function, schema
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p030_met_service_bypass ON pack030_nz_reporting.gl_nz_report_metrics;
DROP POLICY IF EXISTS p030_met_tenant_isolation ON pack030_nz_reporting.gl_nz_report_metrics;
DROP POLICY IF EXISTS p030_sec_service_bypass ON pack030_nz_reporting.gl_nz_report_sections;
DROP POLICY IF EXISTS p030_sec_tenant_isolation ON pack030_nz_reporting.gl_nz_report_sections;
DROP POLICY IF EXISTS p030_rpt_service_bypass ON pack030_nz_reporting.gl_nz_reports;
DROP POLICY IF EXISTS p030_rpt_tenant_isolation ON pack030_nz_reporting.gl_nz_reports;

-- Disable RLS
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_report_metrics DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_report_sections DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_reports DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p030_metrics_updated ON pack030_nz_reporting.gl_nz_report_metrics;
DROP TRIGGER IF EXISTS trg_p030_sections_updated ON pack030_nz_reporting.gl_nz_report_sections;
DROP TRIGGER IF EXISTS trg_p030_reports_updated ON pack030_nz_reporting.gl_nz_reports;

-- Drop tables (reverse order for FK dependencies)
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_report_metrics CASCADE;
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_report_sections CASCADE;
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_reports CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack030_nz_reporting CASCADE;
