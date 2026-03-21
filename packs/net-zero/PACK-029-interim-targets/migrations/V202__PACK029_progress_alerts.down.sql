-- =============================================================================
-- V202 DOWN: Drop gl_progress_alerts
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_pa_service_bypass ON pack029_interim_targets.gl_progress_alerts;
DROP POLICY IF EXISTS p029_pa_tenant_isolation ON pack029_interim_targets.gl_progress_alerts;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_progress_alerts DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_progress_alerts_updated ON pack029_interim_targets.gl_progress_alerts;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_progress_alerts CASCADE;
