-- =============================================================================
-- V205 DOWN: Drop gl_reporting_periods
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_rp_service_bypass ON pack029_interim_targets.gl_reporting_periods;
DROP POLICY IF EXISTS p029_rp_tenant_isolation ON pack029_interim_targets.gl_reporting_periods;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_reporting_periods DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_reporting_periods_updated ON pack029_interim_targets.gl_reporting_periods;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_reporting_periods CASCADE;
