-- =============================================================================
-- V164 DOWN: Drop peer_groups and peer_rankings tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p026_pr_service_bypass ON pack026_sme_net_zero.peer_rankings;
DROP POLICY IF EXISTS p026_pr_tenant_isolation ON pack026_sme_net_zero.peer_rankings;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.peer_rankings DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p026_peer_rankings_updated ON pack026_sme_net_zero.peer_rankings;
DROP TRIGGER IF EXISTS trg_p026_peer_groups_updated ON pack026_sme_net_zero.peer_groups;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack026_sme_net_zero.peer_rankings CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.peer_groups CASCADE;
