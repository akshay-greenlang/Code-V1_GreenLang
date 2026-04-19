-- =============================================================================
-- V193 DOWN: Drop gl_scenario_definitions, gl_scenario_parameters,
--            gl_scenario_results
-- =============================================================================

-- Scenario results
DROP POLICY IF EXISTS p028_sr_service_bypass ON pack028_sector_pathway.gl_scenario_results;
DROP POLICY IF EXISTS p028_sr_tenant_isolation ON pack028_sector_pathway.gl_scenario_results;
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_scenario_results DISABLE ROW LEVEL SECURITY;
DROP TRIGGER IF EXISTS trg_p028_scenario_results_updated ON pack028_sector_pathway.gl_scenario_results;
DROP TABLE IF EXISTS pack028_sector_pathway.gl_scenario_results CASCADE;

-- Scenario parameters
DROP POLICY IF EXISTS p028_spm_service_bypass ON pack028_sector_pathway.gl_scenario_parameters;
DROP POLICY IF EXISTS p028_spm_tenant_isolation ON pack028_sector_pathway.gl_scenario_parameters;
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_scenario_parameters DISABLE ROW LEVEL SECURITY;
DROP TRIGGER IF EXISTS trg_p028_scenario_parameters_updated ON pack028_sector_pathway.gl_scenario_parameters;
DROP TABLE IF EXISTS pack028_sector_pathway.gl_scenario_parameters CASCADE;

-- Scenario definitions
DROP POLICY IF EXISTS p028_sd_service_bypass ON pack028_sector_pathway.gl_scenario_definitions;
DROP POLICY IF EXISTS p028_sd_tenant_isolation ON pack028_sector_pathway.gl_scenario_definitions;
ALTER TABLE IF EXISTS pack028_sector_pathway.gl_scenario_definitions DISABLE ROW LEVEL SECURITY;
DROP TRIGGER IF EXISTS trg_p028_scenario_definitions_updated ON pack028_sector_pathway.gl_scenario_definitions;
DROP TABLE IF EXISTS pack028_sector_pathway.gl_scenario_definitions CASCADE;
