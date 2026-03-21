-- =============================================================================
-- V188 DOWN: Drop gl_scenario_comparisons
-- =============================================================================

DROP POLICY IF EXISTS p028_scc_service_bypass ON pack028_sector_pathway.gl_scenario_comparisons;
DROP POLICY IF EXISTS p028_scc_tenant_isolation ON pack028_sector_pathway.gl_scenario_comparisons;

ALTER TABLE IF EXISTS pack028_sector_pathway.gl_scenario_comparisons DISABLE ROW LEVEL SECURITY;

DROP TRIGGER IF EXISTS trg_p028_scenario_comparisons_updated ON pack028_sector_pathway.gl_scenario_comparisons;

DROP TABLE IF EXISTS pack028_sector_pathway.gl_scenario_comparisons CASCADE;
