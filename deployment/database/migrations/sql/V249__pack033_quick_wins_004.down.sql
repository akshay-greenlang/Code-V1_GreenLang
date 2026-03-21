-- =============================================================================
-- V249 DOWN: Drop PACK-033 carbon reduction tracking tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_efc_service_bypass ON pack033_quick_wins.emission_factors_cache;
DROP POLICY IF EXISTS p033_efc_read_all ON pack033_quick_wins.emission_factors_cache;
DROP POLICY IF EXISTS p033_ci_service_bypass ON pack033_quick_wins.carbon_impacts;
DROP POLICY IF EXISTS p033_ci_tenant_isolation ON pack033_quick_wins.carbon_impacts;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.emission_factors_cache DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.carbon_impacts DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_ci_updated ON pack033_quick_wins.carbon_impacts;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.emission_factors_cache CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.carbon_impacts CASCADE;
