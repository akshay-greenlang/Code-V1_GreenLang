-- =============================================================================
-- V185 DOWN: Drop energy savings and ECM tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p031_finance_service_bypass ON pack031_energy_audit.financial_analyses;
DROP POLICY IF EXISTS p031_finance_tenant_isolation ON pack031_energy_audit.financial_analyses;
DROP POLICY IF EXISTS p031_verify_service_bypass ON pack031_energy_audit.savings_verifications;
DROP POLICY IF EXISTS p031_verify_tenant_isolation ON pack031_energy_audit.savings_verifications;
DROP POLICY IF EXISTS p031_ipmvp_service_bypass ON pack031_energy_audit.ipmvp_plans;
DROP POLICY IF EXISTS p031_ipmvp_tenant_isolation ON pack031_energy_audit.ipmvp_plans;
DROP POLICY IF EXISTS p031_measure_service_bypass ON pack031_energy_audit.energy_savings_measures;
DROP POLICY IF EXISTS p031_measure_tenant_isolation ON pack031_energy_audit.energy_savings_measures;

-- Disable RLS
ALTER TABLE IF EXISTS pack031_energy_audit.financial_analyses DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.savings_verifications DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.ipmvp_plans DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack031_energy_audit.energy_savings_measures DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p031_ipmvp_updated ON pack031_energy_audit.ipmvp_plans;
DROP TRIGGER IF EXISTS trg_p031_measure_updated ON pack031_energy_audit.energy_savings_measures;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack031_energy_audit.financial_analyses CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.savings_verifications CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.ipmvp_plans CASCADE;
DROP TABLE IF EXISTS pack031_energy_audit.energy_savings_measures CASCADE;
