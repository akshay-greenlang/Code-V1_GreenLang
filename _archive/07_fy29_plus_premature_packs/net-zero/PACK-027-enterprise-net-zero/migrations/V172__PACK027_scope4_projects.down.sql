-- =============================================================================
-- V172 DOWN: Drop gl_scope4_projects table
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_s4_service_bypass ON pack027_enterprise_net_zero.gl_scope4_projects;
DROP POLICY IF EXISTS p027_s4_tenant_isolation ON pack027_enterprise_net_zero.gl_scope4_projects;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_scope4_projects DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p027_scope4_projects_updated ON pack027_enterprise_net_zero.gl_scope4_projects;

-- Drop table
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_scope4_projects CASCADE;
