-- =============================================================================
-- V199 DOWN: Drop PACK-032 indoor environment & whole life carbon tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_mq_service_bypass ON pack032_building_assessment.material_quantities;
DROP POLICY IF EXISTS p032_mq_tenant_isolation ON pack032_building_assessment.material_quantities;
DROP POLICY IF EXISTS p032_wlc_service_bypass ON pack032_building_assessment.whole_life_carbon;
DROP POLICY IF EXISTS p032_wlc_tenant_isolation ON pack032_building_assessment.whole_life_carbon;
DROP POLICY IF EXISTS p032_iea_service_bypass ON pack032_building_assessment.indoor_environment_assessments;
DROP POLICY IF EXISTS p032_iea_tenant_isolation ON pack032_building_assessment.indoor_environment_assessments;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.material_quantities DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.whole_life_carbon DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.indoor_environment_assessments DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_wlc_updated ON pack032_building_assessment.whole_life_carbon;
DROP TRIGGER IF EXISTS trg_p032_iea_updated ON pack032_building_assessment.indoor_environment_assessments;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.material_quantities CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.whole_life_carbon CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.indoor_environment_assessments CASCADE;
