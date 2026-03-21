-- =============================================================================
-- V212 DOWN: Drop PACK-030 framework tables
-- =============================================================================

DROP POLICY IF EXISTS p030_fd_service_bypass ON pack030_nz_reporting.gl_nz_framework_deadlines;
DROP POLICY IF EXISTS p030_fd_tenant_isolation ON pack030_nz_reporting.gl_nz_framework_deadlines;
DROP POLICY IF EXISTS p030_fm_service_bypass ON pack030_nz_reporting.gl_nz_framework_mappings;
DROP POLICY IF EXISTS p030_fm_tenant_isolation ON pack030_nz_reporting.gl_nz_framework_mappings;
DROP POLICY IF EXISTS p030_fs_service_bypass ON pack030_nz_reporting.gl_nz_framework_schemas;
DROP POLICY IF EXISTS p030_fs_tenant_isolation ON pack030_nz_reporting.gl_nz_framework_schemas;

ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_framework_deadlines DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_framework_mappings DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_framework_schemas DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p030_framework_deadlines_updated ON pack030_nz_reporting.gl_nz_framework_deadlines;
DROP TRIGGER IF EXISTS trg_p030_framework_mappings_updated ON pack030_nz_reporting.gl_nz_framework_mappings;
DROP TRIGGER IF EXISTS trg_p030_framework_schemas_updated ON pack030_nz_reporting.gl_nz_framework_schemas;

DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_framework_deadlines CASCADE;
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_framework_mappings CASCADE;
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_framework_schemas CASCADE;
