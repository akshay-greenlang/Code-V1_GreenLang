-- =============================================================================
-- V267 DOWN: Drop benchmark database tables
-- =============================================================================

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.sector_benchmarks CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.benchmark_values CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.benchmark_building_types CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.benchmark_sources CASCADE;
