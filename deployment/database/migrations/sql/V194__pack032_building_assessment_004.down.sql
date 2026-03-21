-- =============================================================================
-- V194 DOWN: Drop PACK-032 DHW & lighting tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_la_service_bypass ON pack032_building_assessment.lighting_assessments;
DROP POLICY IF EXISTS p032_la_tenant_isolation ON pack032_building_assessment.lighting_assessments;
DROP POLICY IF EXISTS p032_lz_service_bypass ON pack032_building_assessment.lighting_zones;
DROP POLICY IF EXISTS p032_lz_tenant_isolation ON pack032_building_assessment.lighting_zones;
DROP POLICY IF EXISTS p032_dhw_service_bypass ON pack032_building_assessment.dhw_systems;
DROP POLICY IF EXISTS p032_dhw_tenant_isolation ON pack032_building_assessment.dhw_systems;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.lighting_assessments DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.lighting_zones DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.dhw_systems DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_la_updated ON pack032_building_assessment.lighting_assessments;
DROP TRIGGER IF EXISTS trg_p032_lz_updated ON pack032_building_assessment.lighting_zones;
DROP TRIGGER IF EXISTS trg_p032_dhw_updated ON pack032_building_assessment.dhw_systems;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.lighting_assessments CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.lighting_zones CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.dhw_systems CASCADE;
