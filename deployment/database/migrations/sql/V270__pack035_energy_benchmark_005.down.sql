-- =============================================================================
-- V270 DOWN: Drop peer comparison tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_ch_service_bypass ON pack035_energy_benchmark.comparison_history;
DROP POLICY IF EXISTS p035_ch_tenant_isolation ON pack035_energy_benchmark.comparison_history;
DROP POLICY IF EXISTS p035_pcr_service_bypass ON pack035_energy_benchmark.peer_comparison_results;
DROP POLICY IF EXISTS p035_pcr_tenant_isolation ON pack035_energy_benchmark.peer_comparison_results;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.comparison_history DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.peer_comparison_results DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p035_pg_updated ON pack035_energy_benchmark.peer_groups;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.comparison_history CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.peer_comparison_results CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.peer_groups CASCADE;
