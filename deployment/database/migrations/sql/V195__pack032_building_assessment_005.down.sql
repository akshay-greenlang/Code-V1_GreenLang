-- =============================================================================
-- V195 DOWN: Drop PACK-032 renewable system tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_st_service_bypass ON pack032_building_assessment.solar_thermal_systems;
DROP POLICY IF EXISTS p032_st_tenant_isolation ON pack032_building_assessment.solar_thermal_systems;
DROP POLICY IF EXISTS p032_pv_service_bypass ON pack032_building_assessment.pv_panels;
DROP POLICY IF EXISTS p032_pv_tenant_isolation ON pack032_building_assessment.pv_panels;
DROP POLICY IF EXISTS p032_rs_service_bypass ON pack032_building_assessment.renewable_systems;
DROP POLICY IF EXISTS p032_rs_tenant_isolation ON pack032_building_assessment.renewable_systems;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.solar_thermal_systems DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.pv_panels DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.renewable_systems DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_rs_updated ON pack032_building_assessment.renewable_systems;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.solar_thermal_systems CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.pv_panels CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.renewable_systems CASCADE;
