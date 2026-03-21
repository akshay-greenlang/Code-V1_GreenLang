-- =============================================================================
-- V170 DOWN: Drop gl_scenario_models table
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_sm_service_bypass ON pack027_enterprise_net_zero.gl_scenario_models;
DROP POLICY IF EXISTS p027_sm_tenant_isolation ON pack027_enterprise_net_zero.gl_scenario_models;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_scenario_models DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p027_scenario_models_updated ON pack027_enterprise_net_zero.gl_scenario_models;

-- Drop table
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_scenario_models CASCADE;
