-- =============================================================================
-- V174 DOWN: Drop gl_carbon_pl_allocation and gl_carbon_assets tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_ca_service_bypass ON pack027_enterprise_net_zero.gl_carbon_assets;
DROP POLICY IF EXISTS p027_ca_tenant_isolation ON pack027_enterprise_net_zero.gl_carbon_assets;
DROP POLICY IF EXISTS p027_cpl_service_bypass ON pack027_enterprise_net_zero.gl_carbon_pl_allocation;
DROP POLICY IF EXISTS p027_cpl_tenant_isolation ON pack027_enterprise_net_zero.gl_carbon_pl_allocation;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_assets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_pl_allocation DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p027_carbon_assets_updated ON pack027_enterprise_net_zero.gl_carbon_assets;
DROP TRIGGER IF EXISTS trg_p027_carbon_pl_updated ON pack027_enterprise_net_zero.gl_carbon_pl_allocation;

-- Drop tables
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_assets CASCADE;
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_pl_allocation CASCADE;
