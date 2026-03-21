-- =============================================================================
-- V250 DOWN: Drop PACK-033 prioritization tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p033_is_service_bypass ON pack033_quick_wins.implementation_sequences;
DROP POLICY IF EXISTS p033_is_tenant_isolation ON pack033_quick_wins.implementation_sequences;
DROP POLICY IF EXISTS p033_ps_service_bypass ON pack033_quick_wins.priority_scores;
DROP POLICY IF EXISTS p033_ps_tenant_isolation ON pack033_quick_wins.priority_scores;

-- Disable RLS
ALTER TABLE IF EXISTS pack033_quick_wins.implementation_sequences DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack033_quick_wins.priority_scores DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p033_is_updated ON pack033_quick_wins.implementation_sequences;
DROP TRIGGER IF EXISTS trg_p033_ps_updated ON pack033_quick_wins.priority_scores;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack033_quick_wins.implementation_sequences CASCADE;
DROP TABLE IF EXISTS pack033_quick_wins.priority_scores CASCADE;
