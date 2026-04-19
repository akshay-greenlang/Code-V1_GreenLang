-- =============================================================================
-- V201 DOWN: Drop gl_corrective_actions
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_ca_service_bypass ON pack029_interim_targets.gl_corrective_actions;
DROP POLICY IF EXISTS p029_ca_tenant_isolation ON pack029_interim_targets.gl_corrective_actions;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_corrective_actions DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_corrective_actions_updated ON pack029_interim_targets.gl_corrective_actions;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_corrective_actions CASCADE;
