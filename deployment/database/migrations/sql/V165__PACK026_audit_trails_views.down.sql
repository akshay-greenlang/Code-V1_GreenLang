-- =============================================================================
-- V165 DOWN: Drop audit_trail table and all views
-- =============================================================================

-- Drop views first (they depend on tables from earlier migrations)
DROP VIEW IF EXISTS pack026_sme_net_zero.v_peer_leaderboard;
DROP VIEW IF EXISTS pack026_sme_net_zero.v_grant_calendar;
DROP VIEW IF EXISTS pack026_sme_net_zero.v_sme_dashboard;

-- Drop RLS policies
DROP POLICY IF EXISTS p026_at_service_bypass ON pack026_sme_net_zero.audit_trail;
DROP POLICY IF EXISTS p026_at_tenant_isolation ON pack026_sme_net_zero.audit_trail;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.audit_trail DISABLE ROW LEVEL SECURITY;

-- Drop table
DROP TABLE IF EXISTS pack026_sme_net_zero.audit_trail CASCADE;
