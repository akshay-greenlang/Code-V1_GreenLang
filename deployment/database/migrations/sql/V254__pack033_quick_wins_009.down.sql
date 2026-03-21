-- =============================================================================
-- V254 DOWN: Drop PACK-033 reporting configuration tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_dw_service_bypass ON pack033_quick_wins.dashboard_widgets;
DROP POLICY IF EXISTS p033_dw_tenant_isolation ON pack033_quick_wins.dashboard_widgets;
DROP POLICY IF EXISTS p033_rc_service_bypass ON pack033_quick_wins.report_configs;
DROP POLICY IF EXISTS p033_rc_tenant_isolation ON pack033_quick_wins.report_configs;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.dashboard_widgets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.report_configs DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_dw_updated ON pack033_quick_wins.dashboard_widgets;
DROP TRIGGER IF EXISTS trg_p033_rc_updated ON pack033_quick_wins.report_configs;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.dashboard_widgets CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.report_configs CASCADE;
