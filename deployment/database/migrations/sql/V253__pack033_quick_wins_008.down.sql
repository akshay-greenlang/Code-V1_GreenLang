-- =============================================================================
-- V253 DOWN: Drop PACK-033 progress tracking tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_sa_service_bypass ON pack033_quick_wins.savings_actuals;
DROP POLICY IF EXISTS p033_sa_tenant_isolation ON pack033_quick_wins.savings_actuals;
DROP POLICY IF EXISTS p033_ip_service_bypass ON pack033_quick_wins.implementation_progress;
DROP POLICY IF EXISTS p033_ip_tenant_isolation ON pack033_quick_wins.implementation_progress;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.savings_actuals DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.implementation_progress DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_sa_updated ON pack033_quick_wins.savings_actuals;
DROP TRIGGER IF EXISTS trg_p033_ip_updated ON pack033_quick_wins.implementation_progress;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.savings_actuals CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.implementation_progress CASCADE;
