-- =============================================================================
-- V185 DOWN: Drop gl_technology_roadmaps
-- =============================================================================

DROP POLICY IF EXISTS p028_tr_service_bypass ON pack028_sector_pathway.gl_technology_roadmaps;
DROP POLICY IF EXISTS p028_tr_tenant_isolation ON pack028_sector_pathway.gl_technology_roadmaps;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_technology_roadmaps DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_technology_roadmaps_updated ON pack028_sector_pathway.gl_technology_roadmaps;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_technology_roadmaps CASCADE;
