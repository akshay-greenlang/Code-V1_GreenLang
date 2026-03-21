-- =============================================================================
-- V178 DOWN: Drop gl_board_reports table
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_br_service_bypass ON pack027_enterprise_net_zero.gl_board_reports;
DROP POLICY IF EXISTS p027_br_tenant_isolation ON pack027_enterprise_net_zero.gl_board_reports;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_board_reports DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p027_board_reports_updated ON pack027_enterprise_net_zero.gl_board_reports;

-- Drop table
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_board_reports CASCADE;
