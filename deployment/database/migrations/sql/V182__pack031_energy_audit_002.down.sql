-- =============================================================================
-- V182 DOWN: Drop energy metering and baseline tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_enpi_service_bypass ON pack031_energy_audit.enpi_records;
DROP POLICY IF EXISTS p031_enpi_tenant_isolation ON pack031_energy_audit.enpi_records;
DROP POLICY IF EXISTS p031_baseline_service_bypass ON pack031_energy_audit.energy_baselines;
DROP POLICY IF EXISTS p031_baseline_tenant_isolation ON pack031_energy_audit.energy_baselines;
DROP POLICY IF EXISTS p031_reading_service_bypass ON pack031_energy_audit.meter_readings;
DROP POLICY IF EXISTS p031_reading_tenant_isolation ON pack031_energy_audit.meter_readings;
DROP POLICY IF EXISTS p031_meter_service_bypass ON pack031_energy_audit.energy_meters;
DROP POLICY IF EXISTS p031_meter_tenant_isolation ON pack031_energy_audit.energy_meters;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.enpi_records DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_baselines DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.meter_readings DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_meters DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_baseline_updated ON pack031_energy_audit.energy_baselines;
DROP TRIGGER IF EXISTS trg_p031_meter_updated ON pack031_energy_audit.energy_meters;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.enpi_records CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_baselines CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.meter_readings CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_meters CASCADE;
