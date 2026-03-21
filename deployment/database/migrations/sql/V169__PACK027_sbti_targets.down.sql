-- =============================================================================
-- V169 DOWN: Drop gl_sbti_targets table
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_st_service_bypass ON pack027_enterprise_net_zero.gl_sbti_targets;
DROP POLICY IF EXISTS p027_st_tenant_isolation ON pack027_enterprise_net_zero.gl_sbti_targets;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_sbti_targets DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p027_sbti_targets_updated ON pack027_enterprise_net_zero.gl_sbti_targets;

-- Drop table
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_sbti_targets CASCADE;
