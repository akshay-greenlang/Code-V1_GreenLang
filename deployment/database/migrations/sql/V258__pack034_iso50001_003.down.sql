-- =============================================================================
-- V258 DOWN: Drop PACK-034 baseline tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_rv_service_bypass ON pack034_iso50001.relevant_variables;
DROP POLICY IF EXISTS p034_rv_tenant_isolation ON pack034_iso50001.relevant_variables;
DROP POLICY IF EXISTS p034_bm_service_bypass ON pack034_iso50001.baseline_models;
DROP POLICY IF EXISTS p034_bm_tenant_isolation ON pack034_iso50001.baseline_models;
DROP POLICY IF EXISTS p034_bl_service_bypass ON pack034_iso50001.energy_baselines;
DROP POLICY IF EXISTS p034_bl_tenant_isolation ON pack034_iso50001.energy_baselines;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.relevant_variables DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.baseline_models DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.energy_baselines DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_rv_updated ON pack034_iso50001.relevant_variables;
DROP TRIGGER IF EXISTS trg_p034_bm_updated ON pack034_iso50001.baseline_models;
DROP TRIGGER IF EXISTS trg_p034_bl_updated ON pack034_iso50001.energy_baselines;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.relevant_variables CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.baseline_models CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.energy_baselines CASCADE;
