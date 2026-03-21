-- =============================================================================
-- V206 DOWN: Drop gl_validation_results
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_vr_service_bypass ON pack029_interim_targets.gl_validation_results;
DROP POLICY IF EXISTS p029_vr_tenant_isolation ON pack029_interim_targets.gl_validation_results;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_validation_results DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_validation_results_updated ON pack029_interim_targets.gl_validation_results;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_validation_results CASCADE;
