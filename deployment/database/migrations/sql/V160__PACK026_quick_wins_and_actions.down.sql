-- =============================================================================
-- V160 DOWN: Drop quick_wins_library and selected_actions tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p026_sa_service_bypass ON pack026_sme_net_zero.selected_actions;
DROP POLICY IF EXISTS p026_sa_tenant_isolation ON pack026_sme_net_zero.selected_actions;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.selected_actions DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p026_selected_actions_updated ON pack026_sme_net_zero.selected_actions;
DROP TRIGGER IF EXISTS trg_p026_quick_wins_updated ON pack026_sme_net_zero.quick_wins_library;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack026_sme_net_zero.selected_actions CASCADE;
DROP TABLE IF EXISTS pack026_sme_net_zero.quick_wins_library CASCADE;
