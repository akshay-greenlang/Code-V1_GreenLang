-- =============================================================================
-- V269 DOWN: Drop weather data tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_rm_service_bypass ON pack035_energy_benchmark.regression_models;
DROP POLICY IF EXISTS p035_rm_tenant_isolation ON pack035_energy_benchmark.regression_models;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.regression_models DISABLE ROW LEVEL SECURITY;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.regression_models CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.tmy_data CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.degree_day_data CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.weather_stations CASCADE;
