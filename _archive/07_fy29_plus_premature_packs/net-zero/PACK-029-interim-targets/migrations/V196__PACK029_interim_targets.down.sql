-- =============================================================================
-- V196 DOWN: Drop PACK-029 schema, gl_interim_targets, trigger, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_it_service_bypass ON pack029_interim_targets.gl_interim_targets;
DROP POLICY IF EXISTS p029_it_tenant_isolation ON pack029_interim_targets.gl_interim_targets;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_interim_targets DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_interim_targets_updated ON pack029_interim_targets.gl_interim_targets;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_interim_targets CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack029_interim_targets.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack029_interim_targets CASCADE;
