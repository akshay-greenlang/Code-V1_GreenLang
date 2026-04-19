-- =============================================================================
-- V203 DOWN: Drop gl_initiative_schedule
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_is_service_bypass ON pack029_interim_targets.gl_initiative_schedule;
DROP POLICY IF EXISTS p029_is_tenant_isolation ON pack029_interim_targets.gl_initiative_schedule;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_initiative_schedule DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_initiative_schedule_updated ON pack029_interim_targets.gl_initiative_schedule;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_initiative_schedule CASCADE;
