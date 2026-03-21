-- =============================================================================
-- V167 DOWN: Drop gl_entity_hierarchy and gl_intercompany_transactions
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_ict_service_bypass ON pack027_enterprise_net_zero.gl_intercompany_transactions;
DROP POLICY IF EXISTS p027_ict_tenant_isolation ON pack027_enterprise_net_zero.gl_intercompany_transactions;
DROP POLICY IF EXISTS p027_eh_service_bypass ON pack027_enterprise_net_zero.gl_entity_hierarchy;
DROP POLICY IF EXISTS p027_eh_tenant_isolation ON pack027_enterprise_net_zero.gl_entity_hierarchy;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_intercompany_transactions DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_entity_hierarchy DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p027_intercompany_txn_updated ON pack027_enterprise_net_zero.gl_intercompany_transactions;
DROP TRIGGER IF EXISTS trg_p027_entity_hierarchy_updated ON pack027_enterprise_net_zero.gl_entity_hierarchy;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_intercompany_transactions CASCADE;
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_entity_hierarchy CASCADE;
