-- =============================================================================
-- V193 DOWN: Drop PACK-032 HVAC system tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_rr_service_bypass ON pack032_building_assessment.refrigerant_records;
DROP POLICY IF EXISTS p032_rr_tenant_isolation ON pack032_building_assessment.refrigerant_records;
DROP POLICY IF EXISTS p032_vs_service_bypass ON pack032_building_assessment.ventilation_systems;
DROP POLICY IF EXISTS p032_vs_tenant_isolation ON pack032_building_assessment.ventilation_systems;
DROP POLICY IF EXISTS p032_cs_service_bypass ON pack032_building_assessment.cooling_systems;
DROP POLICY IF EXISTS p032_cs_tenant_isolation ON pack032_building_assessment.cooling_systems;
DROP POLICY IF EXISTS p032_hs_service_bypass ON pack032_building_assessment.heating_systems;
DROP POLICY IF EXISTS p032_hs_tenant_isolation ON pack032_building_assessment.heating_systems;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.refrigerant_records DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.ventilation_systems DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.cooling_systems DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.heating_systems DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_vs_updated ON pack032_building_assessment.ventilation_systems;
DROP TRIGGER IF EXISTS trg_p032_cs_updated ON pack032_building_assessment.cooling_systems;
DROP TRIGGER IF EXISTS trg_p032_hs_updated ON pack032_building_assessment.heating_systems;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.refrigerant_records CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.ventilation_systems CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.cooling_systems CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.heating_systems CASCADE;
