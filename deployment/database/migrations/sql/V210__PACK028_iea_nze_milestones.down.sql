-- =============================================================================
-- V190 DOWN: Drop gl_iea_technology_milestones
-- =============================================================================

DROP POLICY IF EXISTS p028_iem_service_bypass ON pack028_sector_pathway.gl_iea_technology_milestones;
DROP POLICY IF EXISTS p028_iem_tenant_isolation ON pack028_sector_pathway.gl_iea_technology_milestones;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_iea_technology_milestones DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_iea_milestones_updated ON pack028_sector_pathway.gl_iea_technology_milestones;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_iea_technology_milestones CASCADE;
