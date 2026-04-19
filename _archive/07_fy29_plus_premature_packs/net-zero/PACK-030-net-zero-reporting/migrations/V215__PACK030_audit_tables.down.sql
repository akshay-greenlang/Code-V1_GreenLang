-- =============================================================================
-- V215 DOWN: Drop PACK-030 audit trail and data lineage tables
-- =============================================================================

DROP POLICY IF EXISTS p030_dl_service_bypass ON pack030_nz_reporting.gl_nz_data_lineage;
DROP POLICY IF EXISTS p030_dl_tenant_isolation ON pack030_nz_reporting.gl_nz_data_lineage;
DROP POLICY IF EXISTS p030_at_service_bypass ON pack030_nz_reporting.gl_nz_audit_trail;
DROP POLICY IF EXISTS p030_at_tenant_isolation ON pack030_nz_reporting.gl_nz_audit_trail;

ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_data_lineage DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_audit_trail DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p030_data_lineage_updated ON pack030_nz_reporting.gl_nz_data_lineage;

DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_data_lineage CASCADE;
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_audit_trail CASCADE;
