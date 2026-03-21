-- =============================================================================
-- V186 DOWN: Drop compressed air system tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_pp_service_bypass ON pack031_energy_audit.pressure_profiles;
DROP POLICY IF EXISTS p031_pp_tenant_isolation ON pack031_energy_audit.pressure_profiles;
DROP POLICY IF EXISTS p031_leak_service_bypass ON pack031_energy_audit.leak_surveys;
DROP POLICY IF EXISTS p031_leak_tenant_isolation ON pack031_energy_audit.leak_surveys;
DROP POLICY IF EXISTS p031_cinv_service_bypass ON pack031_energy_audit.compressor_inventory;
DROP POLICY IF EXISTS p031_cinv_tenant_isolation ON pack031_energy_audit.compressor_inventory;
DROP POLICY IF EXISTS p031_cas_service_bypass ON pack031_energy_audit.compressed_air_systems;
DROP POLICY IF EXISTS p031_cas_tenant_isolation ON pack031_energy_audit.compressed_air_systems;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.pressure_profiles DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.leak_surveys DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.compressor_inventory DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.compressed_air_systems DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_cas_updated ON pack031_energy_audit.compressed_air_systems;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.pressure_profiles CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.leak_surveys CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.compressor_inventory CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.compressed_air_systems CASCADE;
