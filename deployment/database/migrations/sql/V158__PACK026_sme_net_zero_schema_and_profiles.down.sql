-- =============================================================================
-- V158 DOWN: Drop PACK-026 schema, sme_profiles, trigger, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p026_sme_profiles_service_bypass ON pack026_sme_net_zero.sme_profiles;
DROP POLICY IF EXISTS p026_sme_profiles_tenant_isolation ON pack026_sme_net_zero.sme_profiles;

-- Disable RLS
ALTER TABLE IF EXISTS pack026_sme_net_zero.sme_profiles DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p026_sme_profiles_updated ON pack026_sme_net_zero.sme_profiles;

-- Drop table
DROP TABLE IF EXISTS pack026_sme_net_zero.sme_profiles CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack026_sme_net_zero.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack026_sme_net_zero CASCADE;
