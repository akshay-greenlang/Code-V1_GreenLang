-- =============================================================================
-- V162 DOWN: Drop accounting_connections, spend_categories, spend_transactions
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p026_st_service_bypass ON pack026_sme_net_zero.spend_transactions;
DROP POLICY IF EXISTS p026_st_tenant_isolation ON pack026_sme_net_zero.spend_transactions;
DROP POLICY IF EXISTS p026_sc_service_bypass ON pack026_sme_net_zero.spend_categories;
DROP POLICY IF EXISTS p026_sc_tenant_isolation ON pack026_sme_net_zero.spend_categories;
DROP POLICY IF EXISTS p026_ac_service_bypass ON pack026_sme_net_zero.accounting_connections;
DROP POLICY IF EXISTS p026_ac_tenant_isolation ON pack026_sme_net_zero.accounting_connections;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.spend_transactions DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack026_sme_net_zero.spend_categories DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack026_sme_net_zero.accounting_connections DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p026_spend_categories_updated ON pack026_sme_net_zero.spend_categories;
DROP TRIGGER IF EXISTS trg_p026_accounting_connections_updated ON pack026_sme_net_zero.accounting_connections;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack026_sme_net_zero.spend_transactions CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.spend_categories CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.accounting_connections CASCADE;
