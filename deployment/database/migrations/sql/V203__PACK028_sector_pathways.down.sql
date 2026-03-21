-- =============================================================================
-- V183 DOWN: Drop gl_sector_pathways
-- =============================================================================

DROP POLICY IF EXISTS p028_sp_service_bypass ON pack028_sector_pathway.gl_sector_pathways;
DROP POLICY IF EXISTS p028_sp_tenant_isolation ON pack028_sector_pathway.gl_sector_pathways;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_pathways DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_sector_pathways_updated ON pack028_sector_pathway.gl_sector_pathways;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_pathways CASCADE;
