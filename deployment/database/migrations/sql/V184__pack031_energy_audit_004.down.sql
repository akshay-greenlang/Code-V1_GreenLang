-- =============================================================================
-- V184 DOWN: Drop equipment registry tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_hvac_service_bypass ON pack031_energy_audit.hvac_data;
DROP POLICY IF EXISTS p031_hvac_tenant_isolation ON pack031_energy_audit.hvac_data;
DROP POLICY IF EXISTS p031_boiler_service_bypass ON pack031_energy_audit.boiler_data;
DROP POLICY IF EXISTS p031_boiler_tenant_isolation ON pack031_energy_audit.boiler_data;
DROP POLICY IF EXISTS p031_compressor_service_bypass ON pack031_energy_audit.compressor_data;
DROP POLICY IF EXISTS p031_compressor_tenant_isolation ON pack031_energy_audit.compressor_data;
DROP POLICY IF EXISTS p031_pump_service_bypass ON pack031_energy_audit.pump_data;
DROP POLICY IF EXISTS p031_pump_tenant_isolation ON pack031_energy_audit.pump_data;
DROP POLICY IF EXISTS p031_motor_service_bypass ON pack031_energy_audit.motor_data;
DROP POLICY IF EXISTS p031_motor_tenant_isolation ON pack031_energy_audit.motor_data;
DROP POLICY IF EXISTS p031_equip_service_bypass ON pack031_energy_audit.equipment;
DROP POLICY IF EXISTS p031_equip_tenant_isolation ON pack031_energy_audit.equipment;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.hvac_data DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.boiler_data DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.compressor_data DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.pump_data DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.motor_data DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.equipment DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_equip_updated ON pack031_energy_audit.equipment;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.hvac_data CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.boiler_data CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.compressor_data CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.pump_data CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.motor_data CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.equipment CASCADE;
