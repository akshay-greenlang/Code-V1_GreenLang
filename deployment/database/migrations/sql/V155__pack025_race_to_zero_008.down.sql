-- =============================================================================
-- V155 DOWN: Drop credibility_assessments and lobbying_assessments tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_la_service_bypass ON pack025_race_to_zero.lobbying_assessments;
DROP POLICY IF EXISTS p025_la_tenant_isolation ON pack025_race_to_zero.lobbying_assessments;
DROP POLICY IF EXISTS p025_ca_service_bypass ON pack025_race_to_zero.credibility_assessments;
DROP POLICY IF EXISTS p025_ca_tenant_isolation ON pack025_race_to_zero.credibility_assessments;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.lobbying_assessments DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.credibility_assessments DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_lobbying_updated ON pack025_race_to_zero.lobbying_assessments;
DROP TRIGGER IF EXISTS trg_p025_credibility_updated ON pack025_race_to_zero.credibility_assessments;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.lobbying_assessments CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.credibility_assessments CASCADE;
