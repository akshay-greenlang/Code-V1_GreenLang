-- =============================================================================
-- V216 DOWN: Drop PACK-030 XBRL tagging table
-- =============================================================================

DROP POLICY IF EXISTS p030_xb_service_bypass ON pack030_nz_reporting.gl_nz_xbrl_tags;
DROP POLICY IF EXISTS p030_xb_tenant_isolation ON pack030_nz_reporting.gl_nz_xbrl_tags;

ALTER TABLE IF EXISTS pack030_nz_reporting.gl_nz_xbrl_tags DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p030_xbrl_tags_updated ON pack030_nz_reporting.gl_nz_xbrl_tags;

DROP TABLE IF EXISTS pack030_nz_reporting.gl_nz_xbrl_tags CASCADE;
