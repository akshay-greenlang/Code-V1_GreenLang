-- =============================================================================
-- V171 DOWN: Drop gl_carbon_prices and gl_carbon_liabilities tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_cl_service_bypass ON pack027_enterprise_net_zero.gl_carbon_liabilities;
DROP POLICY IF EXISTS p027_cl_tenant_isolation ON pack027_enterprise_net_zero.gl_carbon_liabilities;
DROP POLICY IF EXISTS p027_cp_service_bypass ON pack027_enterprise_net_zero.gl_carbon_prices;
DROP POLICY IF EXISTS p027_cp_tenant_isolation ON pack027_enterprise_net_zero.gl_carbon_prices;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_liabilities DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_prices DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p027_carbon_liabilities_updated ON pack027_enterprise_net_zero.gl_carbon_liabilities;
DROP TRIGGER IF EXISTS trg_p027_carbon_prices_updated ON pack027_enterprise_net_zero.gl_carbon_prices;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_liabilities CASCADE;
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_carbon_prices CASCADE;
