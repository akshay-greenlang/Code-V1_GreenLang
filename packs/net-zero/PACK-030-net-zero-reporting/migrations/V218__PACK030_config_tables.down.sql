-- =============================================================================
-- V218 DOWN: Drop PACK-030 configuration and dashboard tables
-- =============================================================================

DROP POLICY IF EXISTS p030_dv_service_bypass ON pack030_nz_reporting.gl_nz_dashboard_views;
DROP POLICY IF EXISTS p030_dv_tenant_isolation ON pack030_nz_reporting.gl_nz_dashboard_views;
DROP POLICY IF EXISTS p030_rc_service_bypass ON pack030_nz_reporting.gl_nz_report_config;
DROP POLICY IF EXISTS p030_rc_tenant_isolation ON pack030_nz_reporting.gl_nz_report_config;

ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_dashboard_views DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_report_config DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p030_dashboard_views_updated ON pack030_nz_reporting.gl_nz_dashboard_views;
DROP TRIGGER IF EXISTS trg_p030_report_config_updated ON pack030_nz_reporting.gl_nz_report_config;

DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_dashboard_views CASCADE;
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_report_config CASCADE;
