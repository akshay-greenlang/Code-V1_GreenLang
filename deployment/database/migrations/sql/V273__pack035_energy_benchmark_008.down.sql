-- =============================================================================
-- V273 DOWN: Drop trend analysis tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_sc_service_bypass ON pack035_energy_benchmark.step_changes;
DROP POLICY IF EXISTS p035_sc_tenant_isolation ON pack035_energy_benchmark.step_changes;
DROP POLICY IF EXISTS p035_pa_service_bypass ON pack035_energy_benchmark.performance_alerts;
DROP POLICY IF EXISTS p035_pa_tenant_isolation ON pack035_energy_benchmark.performance_alerts;
DROP POLICY IF EXISTS p035_spc_service_bypass ON pack035_energy_benchmark.spc_control_charts;
DROP POLICY IF EXISTS p035_spc_tenant_isolation ON pack035_energy_benchmark.spc_control_charts;
DROP POLICY IF EXISTS p035_tdp_service_bypass ON pack035_energy_benchmark.trend_data_points;
DROP POLICY IF EXISTS p035_tdp_tenant_isolation ON pack035_energy_benchmark.trend_data_points;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.step_changes DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.performance_alerts DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.spc_control_charts DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.trend_data_points DISABLE ROW LEVEL SECURITY;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.step_changes CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.performance_alerts CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.spc_control_charts CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.trend_data_points CASCADE;
