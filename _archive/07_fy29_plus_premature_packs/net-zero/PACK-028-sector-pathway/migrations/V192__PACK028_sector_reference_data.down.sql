-- =============================================================================
-- V192 DOWN: Drop gl_sector_emission_factors, gl_sector_activity_data,
--            gl_sector_technology_catalog
-- =============================================================================

-- Technology catalog
DROP POLICY IF EXISTS p028_stc_service_bypass ON pack028_sector_pathway.gl_sector_technology_catalog;
DROP POLICY IF EXISTS p028_stc_tenant_isolation ON pack028_sector_pathway.gl_sector_technology_catalog;
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_technology_catalog DISABLE ROW LEVEL SECURITY;
DROP TRIGGER IF EXISTS trg_p028_technology_catalog_updated ON pack028_sector_pathway.gl_sector_technology_catalog;
DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_technology_catalog CASCADE;

-- Activity data
DROP POLICY IF EXISTS p028_sad_service_bypass ON pack028_sector_pathway.gl_sector_activity_data;
DROP POLICY IF EXISTS p028_sad_tenant_isolation ON pack028_sector_pathway.gl_sector_activity_data;
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_activity_data DISABLE ROW LEVEL SECURITY;
DROP TRIGGER IF EXISTS trg_p028_activity_data_updated ON pack028_sector_pathway.gl_sector_activity_data;
DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_activity_data CASCADE;

-- Emission factors
DROP POLICY IF EXISTS p028_sef_service_bypass ON pack028_sector_pathway.gl_sector_emission_factors;
DROP POLICY IF EXISTS p028_sef_tenant_isolation ON pack028_sector_pathway.gl_sector_emission_factors;
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_sector_emission_factors DISABLE ROW LEVEL SECURITY;
DROP TRIGGER IF EXISTS trg_p028_emission_factors_updated ON pack028_sector_pathway.gl_sector_emission_factors;
DROP TABLE IF EXISTS pack028_sector_pathway.gl_sector_emission_factors CASCADE;
