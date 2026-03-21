-- =============================================================================
-- V191 DOWN: Drop PACK-032 schema, building profile tables, triggers, function
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_bc_service_bypass ON pack032_building_assessment.building_contacts;
DROP POLICY IF EXISTS p032_bc_tenant_isolation ON pack032_building_assessment.building_contacts;
DROP POLICY IF EXISTS p032_bz_service_bypass ON pack032_building_assessment.building_zones;
DROP POLICY IF EXISTS p032_bz_tenant_isolation ON pack032_building_assessment.building_zones;
DROP POLICY IF EXISTS p032_bp_service_bypass ON pack032_building_assessment.building_profiles;
DROP POLICY IF EXISTS p032_bp_tenant_isolation ON pack032_building_assessment.building_profiles;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.building_contacts DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.building_zones DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.building_profiles DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_bz_updated ON pack032_building_assessment.building_zones;
DROP TRIGGER IF EXISTS trg_p032_bp_updated ON pack032_building_assessment.building_profiles;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack032_building_assessment.building_contacts CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.building_zones CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.building_profiles CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS pack032_building_assessment.fn_set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack032_building_assessment CASCADE;
