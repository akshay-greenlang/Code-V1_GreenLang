-- =============================================================================
-- V154 DOWN: Drop partnerships and partnership_performance tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_pp_service_bypass ON pack025_race_to_zero.partnership_performance;
DROP POLICY IF EXISTS p025_pp_tenant_isolation ON pack025_race_to_zero.partnership_performance;
DROP POLICY IF EXISTS p025_part_service_bypass ON pack025_race_to_zero.partnerships;
DROP POLICY IF EXISTS p025_part_tenant_isolation ON pack025_race_to_zero.partnerships;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.partnership_performance DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.partnerships DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_partnership_perf_updated ON pack025_race_to_zero.partnership_performance;
DROP TRIGGER IF EXISTS trg_p025_partnerships_updated ON pack025_race_to_zero.partnerships;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.partnership_performance CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.partnerships CASCADE;
