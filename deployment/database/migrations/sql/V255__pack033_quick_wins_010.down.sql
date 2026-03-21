-- =============================================================================
-- V255 DOWN: Drop PACK-033 audit trail, views, and functions
-- =============================================================================

-- Drop functions
DROP FUNCTION IF EXISTS pack033_quick_wins.fn_calculate_portfolio_savings(UUID);
DROP FUNCTION IF EXISTS pack033_quick_wins.fn_aggregate_scan_savings(UUID);

-- Drop views (no FK dependencies)
DROP VIEW IF EXISTS pack033_quick_wins.v_rebate_status;
DROP VIEW IF EXISTS pack033_quick_wins.v_savings_progress;
DROP VIEW IF EXISTS pack033_quick_wins.v_action_rankings;
DROP VIEW IF EXISTS pack033_quick_wins.v_quick_wins_summary;

-- Drop RLS policies
DROP POLICY IF EXISTS p033_audit_service_bypass ON pack033_quick_wins.pack033_audit_trail;
DROP POLICY IF EXISTS p033_audit_tenant_isolation ON pack033_quick_wins.pack033_audit_trail;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.pack033_audit_trail DISABLE ROW LEVEL SECURITY;

-- Drop table
DROP TABLE IF EXISTS pack033_quick_wins.pack033_audit_trail CASCADE;
