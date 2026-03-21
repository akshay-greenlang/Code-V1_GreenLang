-- =============================================================================
-- V271 DOWN: Drop portfolio tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p035_fr_service_bypass ON pack035_energy_benchmark.facility_rankings;
DROP POLICY IF EXISTS p035_fr_tenant_isolation ON pack035_energy_benchmark.facility_rankings;
DROP POLICY IF EXISTS p035_pmet_service_bypass ON pack035_energy_benchmark.portfolio_metrics;
DROP POLICY IF EXISTS p035_pmet_tenant_isolation ON pack035_energy_benchmark.portfolio_metrics;
DROP POLICY IF EXISTS p035_pm_service_bypass ON pack035_energy_benchmark.portfolio_memberships;
DROP POLICY IF EXISTS p035_pm_tenant_isolation ON pack035_energy_benchmark.portfolio_memberships;
DROP POLICY IF EXISTS p035_pf_service_bypass ON pack035_energy_benchmark.portfolios;
DROP POLICY IF EXISTS p035_pf_tenant_isolation ON pack035_energy_benchmark.portfolios;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.facility_rankings DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.portfolio_metrics DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.portfolio_memberships DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack035_energy_benchmark.portfolios DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p035_pf_updated ON pack035_energy_benchmark.portfolios;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack035_energy_benchmark.facility_rankings CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.portfolio_metrics CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.portfolio_memberships CASCADE;
DROP TABLE IF EXISTS pack035_energy_benchmark.portfolios CASCADE;
