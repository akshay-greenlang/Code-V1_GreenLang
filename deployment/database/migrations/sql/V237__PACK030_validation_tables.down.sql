-- =============================================================================
-- V217 DOWN: Drop PACK-030 validation results table
-- =============================================================================

DROP POLICY IF EXISTS p030_vr_service_bypass ON pack030_nz_reporting.gl_nz_validation_results;
DROP POLICY IF EXISTS p030_vr_tenant_isolation ON pack030_nz_reporting.gl_nz_validation_results;

ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_validation_results DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p030_validation_results_updated ON pack030_nz_reporting.gl_nz_validation_results;

DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_validation_results CASCADE;
