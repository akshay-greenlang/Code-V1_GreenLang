-- =============================================================================
-- V168 DOWN: Drop gl_enterprise_baselines table
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_bl_service_bypass ON pack027_enterprise_net_zero.gl_enterprise_baselines;
DROP POLICY IF EXISTS p027_bl_tenant_isolation ON pack027_enterprise_net_zero.gl_enterprise_baselines;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_enterprise_baselines DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p027_baselines_updated ON pack027_enterprise_net_zero.gl_enterprise_baselines;

-- Drop table
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_enterprise_baselines CASCADE;
