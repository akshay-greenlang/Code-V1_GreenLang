-- =============================================================================
-- V197 DOWN: Drop PACK-032 benchmarking & performance tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_or_service_bypass ON pack032_building_assessment.occupancy_records;
DROP POLICY IF EXISTS p032_or_tenant_isolation ON pack032_building_assessment.occupancy_records;
DROP POLICY IF EXISTS p032_ecr_service_bypass ON pack032_building_assessment.energy_consumption_records;
DROP POLICY IF EXISTS p032_ecr_tenant_isolation ON pack032_building_assessment.energy_consumption_records;
DROP POLICY IF EXISTS p032_bm_service_bypass ON pack032_building_assessment.building_benchmarks;
DROP POLICY IF EXISTS p032_bm_tenant_isolation ON pack032_building_assessment.building_benchmarks;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.occupancy_records DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.energy_consumption_records DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.building_benchmarks DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_bm_updated ON pack032_building_assessment.building_benchmarks;

-- Drop tables (reverse FK order; hypertable dropped with CASCADE)
DROP TABLE IF EXISTS pack032_building_assessment.occupancy_records CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.energy_consumption_records CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.building_benchmarks CASCADE;
