-- =============================================================================
-- V214 DOWN: Drop PACK-030 assurance evidence table
-- =============================================================================

DROP POLICY IF EXISTS p030_ae_service_bypass ON pack030_nz_reporting.gl_nz_assurance_evidence;
DROP POLICY IF EXISTS p030_ae_tenant_isolation ON pack030_nz_reporting.gl_nz_assurance_evidence;

ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_assurance_evidence DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p030_assurance_evidence_updated ON pack030_nz_reporting.gl_nz_assurance_evidence;

DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_assurance_evidence CASCADE;
