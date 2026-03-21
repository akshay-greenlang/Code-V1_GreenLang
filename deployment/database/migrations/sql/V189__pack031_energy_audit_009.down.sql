-- =============================================================================
-- V189 DOWN: Drop benchmark and compliance tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_iso_service_bypass ON pack031_energy_audit.iso_50001_records;
DROP POLICY IF EXISTS p031_iso_tenant_isolation ON pack031_energy_audit.iso_50001_records;
DROP POLICY IF EXISTS p031_eed_service_bypass ON pack031_energy_audit.eed_compliance;
DROP POLICY IF EXISTS p031_eed_tenant_isolation ON pack031_energy_audit.eed_compliance;
DROP POLICY IF EXISTS p031_bat_service_bypass ON pack031_energy_audit.bat_ael_comparisons;
DROP POLICY IF EXISTS p031_bat_tenant_isolation ON pack031_energy_audit.bat_ael_comparisons;
DROP POLICY IF EXISTS p031_bench_service_bypass ON pack031_energy_audit.energy_benchmarks;
DROP POLICY IF EXISTS p031_bench_tenant_isolation ON pack031_energy_audit.energy_benchmarks;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.iso_50001_records DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.eed_compliance DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.bat_ael_comparisons DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_benchmarks DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_iso_updated ON pack031_energy_audit.iso_50001_records;
DROP TRIGGER IF EXISTS trg_p031_eed_updated ON pack031_energy_audit.eed_compliance;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.iso_50001_records CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.eed_compliance CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.bat_ael_comparisons CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_benchmarks CASCADE;
