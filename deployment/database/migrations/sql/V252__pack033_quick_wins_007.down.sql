-- =============================================================================
-- V252 DOWN: Drop PACK-033 utility rebate tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_ra_service_bypass ON pack033_quick_wins.rebate_applications;
DROP POLICY IF EXISTS p033_ra_tenant_isolation ON pack033_quick_wins.rebate_applications;
DROP POLICY IF EXISTS p033_rp_service_bypass ON pack033_quick_wins.rebate_programs;
DROP POLICY IF EXISTS p033_rp_read_all ON pack033_quick_wins.rebate_programs;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.rebate_applications DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.rebate_programs DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_ra_updated ON pack033_quick_wins.rebate_applications;
DROP TRIGGER IF EXISTS trg_p033_rp_updated ON pack033_quick_wins.rebate_programs;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.rebate_applications CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.rebate_programs CASCADE;
