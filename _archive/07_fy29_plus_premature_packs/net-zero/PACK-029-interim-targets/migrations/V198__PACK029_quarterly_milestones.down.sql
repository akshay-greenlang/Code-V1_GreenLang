-- =============================================================================
-- V198 DOWN: Drop gl_quarterly_milestones
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_qm_service_bypass ON pack029_interim_targets.gl_quarterly_milestones;
DROP POLICY IF EXISTS p029_qm_tenant_isolation ON pack029_interim_targets.gl_quarterly_milestones;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_quarterly_milestones DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_quarterly_milestones_updated ON pack029_interim_targets.gl_quarterly_milestones;

-- Drop table (hypertable)
DROP TABLE IF EXISTS pack029_interim_targets.gl_quarterly_milestones CASCADE;
