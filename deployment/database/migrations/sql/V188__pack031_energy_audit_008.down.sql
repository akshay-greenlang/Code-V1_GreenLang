-- =============================================================================
-- V188 DOWN: Drop waste heat recovery tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_hrp_service_bypass ON pack031_energy_audit.heat_recovery_projects;
DROP POLICY IF EXISTS p031_hrp_tenant_isolation ON pack031_energy_audit.heat_recovery_projects;
DROP POLICY IF EXISTS p031_pinch_service_bypass ON pack031_energy_audit.pinch_analyses;
DROP POLICY IF EXISTS p031_pinch_tenant_isolation ON pack031_energy_audit.pinch_analyses;
DROP POLICY IF EXISTS p031_hs_service_bypass ON pack031_energy_audit.heat_sinks;
DROP POLICY IF EXISTS p031_hs_tenant_isolation ON pack031_energy_audit.heat_sinks;
DROP POLICY IF EXISTS p031_whs_service_bypass ON pack031_energy_audit.waste_heat_sources;
DROP POLICY IF EXISTS p031_whs_tenant_isolation ON pack031_energy_audit.waste_heat_sources;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.heat_recovery_projects DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.pinch_analyses DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.heat_sinks DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.waste_heat_sources DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_hrp_updated ON pack031_energy_audit.heat_recovery_projects;
DROP TRIGGER IF EXISTS trg_p031_whs_updated ON pack031_energy_audit.waste_heat_sources;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.heat_recovery_projects CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.pinch_analyses CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.heat_sinks CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.waste_heat_sources CASCADE;
