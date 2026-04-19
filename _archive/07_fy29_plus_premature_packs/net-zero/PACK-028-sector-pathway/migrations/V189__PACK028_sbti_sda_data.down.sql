-- =============================================================================
-- V189 DOWN: Drop gl_sbti_sector_pathways
-- =============================================================================

DROP POLICY IF EXISTS p028_ssp_service_bypass ON pack028_sector_pathway.gl_sbti_sector_pathways;
DROP POLICY IF EXISTS p028_ssp_tenant_isolation ON pack028_sector_pathway.gl_sbti_sector_pathways;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sbti_sector_pathways DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_sbti_sector_pathways_updated ON pack028_sector_pathway.gl_sbti_sector_pathways;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_sbti_sector_pathways CASCADE;
