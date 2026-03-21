-- =============================================================================
-- V248 DOWN: Drop PACK-033 energy savings estimation tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_ie_service_bypass ON pack033_quick_wins.interactive_effects;
DROP POLICY IF EXISTS p033_ie_tenant_isolation ON pack033_quick_wins.interactive_effects;
DROP POLICY IF EXISTS p033_se_service_bypass ON pack033_quick_wins.savings_estimates;
DROP POLICY IF EXISTS p033_se_tenant_isolation ON pack033_quick_wins.savings_estimates;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.interactive_effects DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.savings_estimates DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_ie_updated ON pack033_quick_wins.interactive_effects;
DROP TRIGGER IF EXISTS trg_p033_se_updated ON pack033_quick_wins.savings_estimates;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.interactive_effects CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.savings_estimates CASCADE;
