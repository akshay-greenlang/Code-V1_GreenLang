-- =============================================================================
-- V198 DOWN: Drop PACK-032 retrofit & certification tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_ca_service_bypass ON pack032_building_assessment.certification_assessments;
DROP POLICY IF EXISTS p032_ca_tenant_isolation ON pack032_building_assessment.certification_assessments;
DROP POLICY IF EXISTS p032_rp_service_bypass ON pack032_building_assessment.retrofit_plans;
DROP POLICY IF EXISTS p032_rp_tenant_isolation ON pack032_building_assessment.retrofit_plans;
DROP POLICY IF EXISTS p032_rm_service_bypass ON pack032_building_assessment.retrofit_measures;
DROP POLICY IF EXISTS p032_rm_tenant_isolation ON pack032_building_assessment.retrofit_measures;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.certification_assessments DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.retrofit_plans DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.retrofit_measures DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_ca_updated ON pack032_building_assessment.certification_assessments;
DROP TRIGGER IF EXISTS trg_p032_rp_updated ON pack032_building_assessment.retrofit_plans;
DROP TRIGGER IF EXISTS trg_p032_rm_updated ON pack032_building_assessment.retrofit_measures;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.certification_assessments CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.retrofit_plans CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.retrofit_measures CASCADE;
