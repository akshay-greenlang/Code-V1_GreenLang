-- =============================================================================
-- V209 DOWN: Drop gl_sbti_submissions
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_ss_service_bypass ON pack029_interim_targets.gl_sbti_submissions;
DROP POLICY IF EXISTS p029_ss_tenant_isolation ON pack029_interim_targets.gl_sbti_submissions;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_sbti_submissions DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_sbti_submissions_updated ON pack029_interim_targets.gl_sbti_submissions;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_sbti_submissions CASCADE;
