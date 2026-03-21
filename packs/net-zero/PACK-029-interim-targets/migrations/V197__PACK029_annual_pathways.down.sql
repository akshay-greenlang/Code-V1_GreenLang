-- =============================================================================
-- V197 DOWN: Drop gl_annual_pathways
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_ap_service_bypass ON pack029_interim_targets.gl_annual_pathways;
DROP POLICY IF EXISTS p029_ap_tenant_isolation ON pack029_interim_targets.gl_annual_pathways;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_annual_pathways DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_annual_pathways_updated ON pack029_interim_targets.gl_annual_pathways;

-- Drop table (hypertable)
DROP TABLE IF EXISTS pack029_interim_targets.gl_annual_pathways CASCADE;
