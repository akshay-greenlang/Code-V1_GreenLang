-- =============================================================================
-- V246 DOWN: Drop PACK-033 schema, core tables, triggers, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_al_service_bypass ON pack033_quick_wins.action_library;
DROP POLICY IF EXISTS p033_al_tenant_isolation ON pack033_quick_wins.action_library;
DROP POLICY IF EXISTS p033_sr_service_bypass ON pack033_quick_wins.scan_results;
DROP POLICY IF EXISTS p033_sr_tenant_isolation ON pack033_quick_wins.scan_results;
DROP POLICY IF EXISTS p033_qs_service_bypass ON pack033_quick_wins.quick_wins_scans;
DROP POLICY IF EXISTS p033_qs_tenant_isolation ON pack033_quick_wins.quick_wins_scans;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.action_library DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.scan_results DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.quick_wins_scans DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_al_updated ON pack033_quick_wins.action_library;
DROP TRIGGER IF EXISTS trg_p033_sr_updated ON pack033_quick_wins.scan_results;
DROP TRIGGER IF EXISTS trg_p033_qs_updated ON pack033_quick_wins.quick_wins_scans;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.action_library CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.scan_results CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.quick_wins_scans CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack033_quick_wins.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack033_quick_wins CASCADE;
