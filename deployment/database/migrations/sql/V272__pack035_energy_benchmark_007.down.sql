-- =============================================================================
-- V272 DOWN: Drop performance rating tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_crrem_service_bypass ON pack035_energy_benchmark.crrem_assessments;
DROP POLICY IF EXISTS p035_crrem_tenant_isolation ON pack035_energy_benchmark.crrem_assessments;
DROP POLICY IF EXISTS p035_epc_service_bypass ON pack035_energy_benchmark.epc_certificates;
DROP POLICY IF EXISTS p035_epc_tenant_isolation ON pack035_energy_benchmark.epc_certificates;
DROP POLICY IF EXISTS p035_pr_service_bypass ON pack035_energy_benchmark.performance_ratings;
DROP POLICY IF EXISTS p035_pr_tenant_isolation ON pack035_energy_benchmark.performance_ratings;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.crrem_assessments DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.epc_certificates DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.performance_ratings DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p035_pr_updated ON pack035_energy_benchmark.performance_ratings;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.crrem_assessments CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.epc_certificates CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.performance_ratings CASCADE;
