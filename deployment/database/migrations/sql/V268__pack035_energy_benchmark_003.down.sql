-- =============================================================================
-- V268 DOWN: Drop EUI calculation tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_ecb_service_bypass ON pack035_energy_benchmark.energy_carrier_breakdown;
DROP POLICY IF EXISTS p035_ecb_tenant_isolation ON pack035_energy_benchmark.energy_carrier_breakdown;
DROP POLICY IF EXISTS p035_eui_service_bypass ON pack035_energy_benchmark.eui_calculations;
DROP POLICY IF EXISTS p035_eui_tenant_isolation ON pack035_energy_benchmark.eui_calculations;
DROP POLICY IF EXISTS p035_ecr_service_bypass ON pack035_energy_benchmark.energy_consumption_records;
DROP POLICY IF EXISTS p035_ecr_tenant_isolation ON pack035_energy_benchmark.energy_consumption_records;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.energy_carrier_breakdown DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.eui_calculations DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.energy_consumption_records DISABLE ROW LEVEL SECURITY;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.energy_carrier_breakdown CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.eui_calculations CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.energy_consumption_records CASCADE;
