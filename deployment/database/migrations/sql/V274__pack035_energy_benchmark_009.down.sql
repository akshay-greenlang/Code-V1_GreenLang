-- =============================================================================
-- V274 DOWN: Drop gap analysis tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_it_service_bypass ON pack035_energy_benchmark.improvement_targets;
DROP POLICY IF EXISTS p035_it_tenant_isolation ON pack035_energy_benchmark.improvement_targets;
DROP POLICY IF EXISTS p035_eug_service_bypass ON pack035_energy_benchmark.end_use_gaps;
DROP POLICY IF EXISTS p035_eug_tenant_isolation ON pack035_energy_benchmark.end_use_gaps;
DROP POLICY IF EXISTS p035_ga_service_bypass ON pack035_energy_benchmark.gap_analyses;
DROP POLICY IF EXISTS p035_ga_tenant_isolation ON pack035_energy_benchmark.gap_analyses;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.improvement_targets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.end_use_gaps DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.gap_analyses DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p035_it_updated ON pack035_energy_benchmark.improvement_targets;
DROP TRIGGER IF EXISTS trg_p035_ga_updated ON pack035_energy_benchmark.gap_analyses;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.improvement_targets CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.end_use_gaps CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.gap_analyses CASCADE;
