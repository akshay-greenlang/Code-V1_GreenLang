-- =============================================================================
-- V187 DOWN: Drop steam system tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_flue_service_bypass ON pack031_energy_audit.flue_gas_analyses;
DROP POLICY IF EXISTS p031_flue_tenant_isolation ON pack031_energy_audit.flue_gas_analyses;
DROP POLICY IF EXISTS p031_insul_service_bypass ON pack031_energy_audit.insulation_assessments;
DROP POLICY IF EXISTS p031_insul_tenant_isolation ON pack031_energy_audit.insulation_assessments;
DROP POLICY IF EXISTS p031_trap_service_bypass ON pack031_energy_audit.steam_trap_surveys;
DROP POLICY IF EXISTS p031_trap_tenant_isolation ON pack031_energy_audit.steam_trap_surveys;
DROP POLICY IF EXISTS p031_sb_service_bypass ON pack031_energy_audit.steam_boilers;
DROP POLICY IF EXISTS p031_sb_tenant_isolation ON pack031_energy_audit.steam_boilers;
DROP POLICY IF EXISTS p031_ss_service_bypass ON pack031_energy_audit.steam_systems;
DROP POLICY IF EXISTS p031_ss_tenant_isolation ON pack031_energy_audit.steam_systems;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.flue_gas_analyses DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.insulation_assessments DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.steam_trap_surveys DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.steam_boilers DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.steam_systems DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_ss_updated ON pack031_energy_audit.steam_systems;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.flue_gas_analyses CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.insulation_assessments CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.steam_trap_surveys CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.steam_boilers CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.steam_systems CASCADE;
