-- =============================================================================
-- V191 DOWN: Drop gl_ipcc_sector_pathways
-- =============================================================================

DROP POLICY IF EXISTS p028_ipp_service_bypass ON pack028_sector_pathway.gl_ipcc_sector_pathways;
DROP POLICY IF EXISTS p028_ipp_tenant_isolation ON pack028_sector_pathway.gl_ipcc_sector_pathways;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_ipcc_sector_pathways DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_ipcc_pathways_updated ON pack028_sector_pathway.gl_ipcc_sector_pathways;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_ipcc_sector_pathways CASCADE;
