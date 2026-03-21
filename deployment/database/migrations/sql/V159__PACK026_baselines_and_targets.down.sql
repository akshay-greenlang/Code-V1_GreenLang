-- =============================================================================
-- V159 DOWN: Drop sme_baselines and sme_targets tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p026_tgt_service_bypass ON pack026_sme_net_zero.sme_targets;
DROP POLICY IF EXISTS p026_tgt_tenant_isolation ON pack026_sme_net_zero.sme_targets;
DROP POLICY IF EXISTS p026_bl_service_bypass ON pack026_sme_net_zero.sme_baselines;
DROP POLICY IF EXISTS p026_bl_tenant_isolation ON pack026_sme_net_zero.sme_baselines;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.sme_targets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack026_sme_net_zero.sme_baselines DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p026_sme_targets_updated ON pack026_sme_net_zero.sme_targets;
DROP TRIGGER IF EXISTS trg_p026_sme_baselines_updated ON pack026_sme_net_zero.sme_baselines;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack026_sme_net_zero.sme_targets CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.sme_baselines CASCADE;
