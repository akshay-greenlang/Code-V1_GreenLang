-- =============================================================================
-- V247 DOWN: Drop PACK-033 financial analysis tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_fs_service_bypass ON pack033_quick_wins.financial_scenarios;
DROP POLICY IF EXISTS p033_fs_tenant_isolation ON pack033_quick_wins.financial_scenarios;
DROP POLICY IF EXISTS p033_pa_service_bypass ON pack033_quick_wins.payback_analyses;
DROP POLICY IF EXISTS p033_pa_tenant_isolation ON pack033_quick_wins.payback_analyses;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.financial_scenarios DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.payback_analyses DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_fs_updated ON pack033_quick_wins.financial_scenarios;
DROP TRIGGER IF EXISTS trg_p033_pa_updated ON pack033_quick_wins.payback_analyses;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.financial_scenarios CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.payback_analyses CASCADE;
