-- =============================================================================
-- V157 DOWN: Drop readiness_scores, audit_trail, and all views
-- =============================================================================

-- Drop views first (they depend on tables)
DROP VIEW IF EXISTS pack025_race_to_zero.v_sector_benchmark;
DROP VIEW IF EXISTS pack025_race_to_zero.v_credibility_leaderboard;
DROP VIEW IF EXISTS pack025_race_to_zero.v_race_to_zero_dashboard;

-- Drop RLS policies
DROP POLICY IF EXISTS p025_at_service_bypass ON pack025_race_to_zero.audit_trail;
DROP POLICY IF EXISTS p025_at_tenant_isolation ON pack025_race_to_zero.audit_trail;
DROP POLICY IF EXISTS p025_rs_service_bypass ON pack025_race_to_zero.readiness_scores;
DROP POLICY IF EXISTS p025_rs_tenant_isolation ON pack025_race_to_zero.readiness_scores;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.audit_trail DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.readiness_scores DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_readiness_scores_updated ON pack025_race_to_zero.readiness_scores;

-- Drop tables
DROP TABLE IF EXISTS pack025_race_to_zero.audit_trail CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.readiness_scores CASCADE;
