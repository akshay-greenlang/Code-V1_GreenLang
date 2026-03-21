-- =============================================================================
-- V259 DOWN: Drop PACK-034 EnPI tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_et_service_bypass ON pack034_iso50001.enpi_targets;
DROP POLICY IF EXISTS p034_et_tenant_isolation ON pack034_iso50001.enpi_targets;
DROP POLICY IF EXISTS p034_ev_service_bypass ON pack034_iso50001.enpi_values;
DROP POLICY IF EXISTS p034_ev_tenant_isolation ON pack034_iso50001.enpi_values;
DROP POLICY IF EXISTS p034_enpi_service_bypass ON pack034_iso50001.energy_performance_indicators;
DROP POLICY IF EXISTS p034_enpi_tenant_isolation ON pack034_iso50001.energy_performance_indicators;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.enpi_targets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.enpi_values DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.energy_performance_indicators DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_et_updated ON pack034_iso50001.enpi_targets;
DROP TRIGGER IF EXISTS trg_p034_ev_updated ON pack034_iso50001.enpi_values;
DROP TRIGGER IF EXISTS trg_p034_enpi_updated ON pack034_iso50001.energy_performance_indicators;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.enpi_targets CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.enpi_values CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.energy_performance_indicators CASCADE;
