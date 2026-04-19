-- =============================================================================
-- V194 DOWN: Drop gl_technology_adoption_tracking
-- =============================================================================

DROP POLICY IF EXISTS p028_tat_service_bypass ON pack028_sector_pathway.gl_technology_adoption_tracking;
DROP POLICY IF EXISTS p028_tat_tenant_isolation ON pack028_sector_pathway.gl_technology_adoption_tracking;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_technology_adoption_tracking DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_tech_adoption_updated ON pack028_sector_pathway.gl_technology_adoption_tracking;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_technology_adoption_tracking CASCADE;
