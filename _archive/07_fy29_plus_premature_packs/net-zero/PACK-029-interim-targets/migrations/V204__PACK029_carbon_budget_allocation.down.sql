-- =============================================================================
-- V204 DOWN: Drop gl_carbon_budget_allocation
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p029_cba_service_bypass ON pack029_interim_targets.gl_carbon_budget_allocation;
DROP POLICY IF EXISTS p029_cba_tenant_isolation ON pack029_interim_targets.gl_carbon_budget_allocation;

-- Disable RLS
ALTER TABLE IF EXISTS pack029_interim_targets.gl_carbon_budget_allocation DISABLE ROW LEVEL SECURITY;

-- Drop trigger
DROP TRIGGER IF EXISTS trg_p029_carbon_budget_updated ON pack029_interim_targets.gl_carbon_budget_allocation;

-- Drop table
DROP TABLE IF EXISTS pack029_interim_targets.gl_carbon_budget_allocation CASCADE;
