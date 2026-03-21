-- =============================================================================
-- V156 DOWN: Drop campaign_submissions and verification_schedules tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_vs_service_bypass ON pack025_race_to_zero.verification_schedules;
DROP POLICY IF EXISTS p025_vs_tenant_isolation ON pack025_race_to_zero.verification_schedules;
DROP POLICY IF EXISTS p025_cs_service_bypass ON pack025_race_to_zero.campaign_submissions;
DROP POLICY IF EXISTS p025_cs_tenant_isolation ON pack025_race_to_zero.campaign_submissions;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.verification_schedules DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.campaign_submissions DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_verification_schedules_updated ON pack025_race_to_zero.verification_schedules;
DROP TRIGGER IF EXISTS trg_p025_campaign_submissions_updated ON pack025_race_to_zero.campaign_submissions;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.verification_schedules CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.campaign_submissions CASCADE;
