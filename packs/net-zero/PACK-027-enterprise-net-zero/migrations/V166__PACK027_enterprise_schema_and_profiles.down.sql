-- =============================================================================
-- V166 DOWN: Drop PACK-027 schema, gl_enterprise_profiles, trigger, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p027_ep_service_bypass ON pack027_enterprise_net_zero.gl_enterprise_profiles;
DROP POLICY IF EXISTS p027_ep_tenant_isolation ON pack027_enterprise_net_zero.gl_enterprise_profiles;

-- Disable RLS
ALTER TABLE IF EXISTS pack027_enterprise_net_zero.gl_enterprise_profiles DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p027_enterprise_profiles_updated ON pack027_enterprise_net_zero.gl_enterprise_profiles;

-- Drop table
DROP TABLE IF EXISTS pack027_enterprise_net_zero.gl_enterprise_profiles CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack027_enterprise_net_zero.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack027_enterprise_net_zero CASCADE;
