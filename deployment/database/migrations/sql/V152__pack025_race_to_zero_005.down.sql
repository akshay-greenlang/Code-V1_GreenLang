-- =============================================================================
-- V152 DOWN: Drop annual_reports and progress_tracking tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_pt_service_bypass ON pack025_race_to_zero.progress_tracking;
DROP POLICY IF EXISTS p025_pt_tenant_isolation ON pack025_race_to_zero.progress_tracking;
DROP POLICY IF EXISTS p025_ar_service_bypass ON pack025_race_to_zero.annual_reports;
DROP POLICY IF EXISTS p025_ar_tenant_isolation ON pack025_race_to_zero.annual_reports;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.progress_tracking DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.annual_reports DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_progress_tracking_updated ON pack025_race_to_zero.progress_tracking;
DROP TRIGGER IF EXISTS trg_p025_annual_reports_updated ON pack025_race_to_zero.annual_reports;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.progress_tracking CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.annual_reports CASCADE;
