-- =============================================================================
-- V213 DOWN: Drop PACK-030 narrative and translation tables
-- =============================================================================

DROP POLICY IF EXISTS p030_tr_service_bypass ON pack030_nz_reporting.gl_nz_translations;
DROP POLICY IF EXISTS p030_tr_tenant_isolation ON pack030_nz_reporting.gl_nz_translations;
DROP POLICY IF EXISTS p030_nar_service_bypass ON pack030_nz_reporting.gl_nz_narratives;
DROP POLICY IF EXISTS p030_nar_tenant_isolation ON pack030_nz_reporting.gl_nz_narratives;

ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_translations DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_narratives DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p030_translations_updated ON pack030_nz_reporting.gl_nz_translations;
DROP TRIGGER IF EXISTS trg_p030_narratives_updated ON pack030_nz_reporting.gl_nz_narratives;

DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_translations CASCADE;
DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_narratives CASCADE;
