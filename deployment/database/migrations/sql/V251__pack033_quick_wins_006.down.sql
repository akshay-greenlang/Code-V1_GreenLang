-- =============================================================================
-- V251 DOWN: Drop PACK-033 behavioral action tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_at_service_bypass ON pack033_quick_wins.adoption_tracking;
DROP POLICY IF EXISTS p033_at_tenant_isolation ON pack033_quick_wins.adoption_tracking;
DROP POLICY IF EXISTS p033_bp_service_bypass ON pack033_quick_wins.behavioral_programs;
DROP POLICY IF EXISTS p033_bp_tenant_isolation ON pack033_quick_wins.behavioral_programs;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.adoption_tracking DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.behavioral_programs DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_at_updated ON pack033_quick_wins.adoption_tracking;
DROP TRIGGER IF EXISTS trg_p033_bp_updated ON pack033_quick_wins.behavioral_programs;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.adoption_tracking CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.behavioral_programs CASCADE;
