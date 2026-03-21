-- =============================================================================
-- V148 DOWN: Drop PACK-025 schema, organization_profiles, trigger, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_org_profiles_service_bypass ON pack025_race_to_zero.organization_profiles;
DROP POLICY IF EXISTS p025_org_profiles_tenant_isolation ON pack025_race_to_zero.organization_profiles;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.organization_profiles DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p025_org_profiles_updated ON pack025_race_to_zero.organization_profiles;

-- Drop table
DROP TABLE IF EXISTS pack025_race_to_zero.organization_profiles CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack025_race_to_zero.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack025_race_to_zero CASCADE;
