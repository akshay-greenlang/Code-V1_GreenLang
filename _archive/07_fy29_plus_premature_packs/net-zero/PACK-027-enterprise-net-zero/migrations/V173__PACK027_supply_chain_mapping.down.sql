-- =============================================================================
-- V173 DOWN: Drop gl_supply_chain_tiers and gl_supplier_engagement tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_se_service_bypass ON pack027_enterprise_net_zero.gl_supplier_engagement;
DROP POLICY IF EXISTS p027_se_tenant_isolation ON pack027_enterprise_net_zero.gl_supplier_engagement;
DROP POLICY IF EXISTS p027_sct_service_bypass ON pack027_enterprise_net_zero.gl_supply_chain_tiers;
DROP POLICY IF EXISTS p027_sct_tenant_isolation ON pack027_enterprise_net_zero.gl_supply_chain_tiers;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_supplier_engagement DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_supply_chain_tiers DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p027_supplier_engagement_updated ON pack027_enterprise_net_zero.gl_supplier_engagement;
DROP TRIGGER IF EXISTS trg_p027_supply_chain_tiers_updated ON pack027_enterprise_net_zero.gl_supply_chain_tiers;

-- Drop tables
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_supplier_engagement CASCADE;
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_supply_chain_tiers CASCADE;
