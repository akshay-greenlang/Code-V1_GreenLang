-- =============================================================================
-- V149 DOWN: Drop pledges and starting_line_assessments tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_sl_assess_service_bypass ON pack025_race_to_zero.starting_line_assessments;
DROP POLICY IF EXISTS p025_sl_assess_tenant_isolation ON pack025_race_to_zero.starting_line_assessments;
DROP POLICY IF EXISTS p025_pledges_service_bypass ON pack025_race_to_zero.pledges;
DROP POLICY IF EXISTS p025_pledges_tenant_isolation ON pack025_race_to_zero.pledges;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.starting_line_assessments DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.pledges DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_sl_assessments_updated ON pack025_race_to_zero.starting_line_assessments;
DROP TRIGGER IF EXISTS trg_p025_pledges_updated ON pack025_race_to_zero.pledges;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.starting_line_assessments CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.pledges CASCADE;
