-- =============================================================================
-- V192 DOWN: Drop PACK-032 building envelope element tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_at_service_bypass ON pack032_building_assessment.airtightness_tests;
DROP POLICY IF EXISTS p032_at_tenant_isolation ON pack032_building_assessment.airtightness_tests;
DROP POLICY IF EXISTS p032_tb_service_bypass ON pack032_building_assessment.thermal_bridges;
DROP POLICY IF EXISTS p032_tb_tenant_isolation ON pack032_building_assessment.thermal_bridges;
DROP POLICY IF EXISTS p032_ewn_service_bypass ON pack032_building_assessment.envelope_windows;
DROP POLICY IF EXISTS p032_ewn_tenant_isolation ON pack032_building_assessment.envelope_windows;
DROP POLICY IF EXISTS p032_ef_service_bypass ON pack032_building_assessment.envelope_floors;
DROP POLICY IF EXISTS p032_ef_tenant_isolation ON pack032_building_assessment.envelope_floors;
DROP POLICY IF EXISTS p032_er_service_bypass ON pack032_building_assessment.envelope_roofs;
DROP POLICY IF EXISTS p032_er_tenant_isolation ON pack032_building_assessment.envelope_roofs;
DROP POLICY IF EXISTS p032_ew_service_bypass ON pack032_building_assessment.envelope_walls;
DROP POLICY IF EXISTS p032_ew_tenant_isolation ON pack032_building_assessment.envelope_walls;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.airtightness_tests DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.thermal_bridges DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.envelope_windows DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.envelope_floors DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.envelope_roofs DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.envelope_walls DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_ewn_updated ON pack032_building_assessment.envelope_windows;
DROP TRIGGER IF EXISTS trg_p032_ef_updated ON pack032_building_assessment.envelope_floors;
DROP TRIGGER IF EXISTS trg_p032_er_updated ON pack032_building_assessment.envelope_roofs;
DROP TRIGGER IF EXISTS trg_p032_ew_updated ON pack032_building_assessment.envelope_walls;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.airtightness_tests CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.thermal_bridges CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.envelope_windows CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.envelope_floors CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.envelope_roofs CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.envelope_walls CASCADE;
