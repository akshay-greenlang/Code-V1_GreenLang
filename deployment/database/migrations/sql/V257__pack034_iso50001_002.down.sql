-- =============================================================================
-- V257 DOWN: Drop PACK-034 SEU tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_driver_service_bypass ON pack034_iso50001.energy_drivers;
DROP POLICY IF EXISTS p034_driver_tenant_isolation ON pack034_iso50001.energy_drivers;
DROP POLICY IF EXISTS p034_equip_service_bypass ON pack034_iso50001.seu_equipment;
DROP POLICY IF EXISTS p034_equip_tenant_isolation ON pack034_iso50001.seu_equipment;
DROP POLICY IF EXISTS p034_seu_service_bypass ON pack034_iso50001.significant_energy_uses;
DROP POLICY IF EXISTS p034_seu_tenant_isolation ON pack034_iso50001.significant_energy_uses;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.energy_drivers DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.seu_equipment DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.significant_energy_uses DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_driver_updated ON pack034_iso50001.energy_drivers;
DROP TRIGGER IF EXISTS trg_p034_equip_updated ON pack034_iso50001.seu_equipment;
DROP TRIGGER IF EXISTS trg_p034_seu_updated ON pack034_iso50001.significant_energy_uses;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.energy_drivers CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.seu_equipment CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.significant_energy_uses CASCADE;
