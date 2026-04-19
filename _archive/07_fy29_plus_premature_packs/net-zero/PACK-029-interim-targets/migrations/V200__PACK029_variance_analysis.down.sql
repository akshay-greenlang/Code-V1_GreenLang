-- =============================================================================
-- V200 DOWN: Drop gl_variance_analysis
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_va_service_bypass ON pack029_interim_targets.gl_variance_analysis;
DROP POLICY IF EXISTS p029_va_tenant_isolation ON pack029_interim_targets.gl_variance_analysis;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_variance_analysis DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_variance_analysis_updated ON pack029_interim_targets.gl_variance_analysis;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_variance_analysis CASCADE;
