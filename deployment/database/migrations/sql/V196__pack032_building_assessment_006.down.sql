-- =============================================================================
-- V196 DOWN: Drop PACK-032 EPC & DEC certificate tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p032_dec_service_bypass ON pack032_building_assessment.dec_certificates;
DROP POLICY IF EXISTS p032_dec_tenant_isolation ON pack032_building_assessment.dec_certificates;
DROP POLICY IF EXISTS p032_epc_service_bypass ON pack032_building_assessment.epc_certificates;
DROP POLICY IF EXISTS p032_epc_tenant_isolation ON pack032_building_assessment.epc_certificates;

-- Disable RLS
ALTER TABLE IF EXISTS pack032_building_assessment.dec_certificates DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack032_building_assessment.epc_certificates DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p032_dec_updated ON pack032_building_assessment.dec_certificates;
DROP TRIGGER IF EXISTS trg_p032_epc_updated ON pack032_building_assessment.epc_certificates;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack032_building_assessment.dec_certificates CASCADE;
DROP TABLE IF EXISTS pack032_building_assessment.epc_certificates CASCADE;
