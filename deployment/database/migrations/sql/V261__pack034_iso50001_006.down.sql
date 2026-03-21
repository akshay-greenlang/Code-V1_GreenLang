-- =============================================================================
-- V261 DOWN: Drop PACK-034 action plan tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p034_ai_service_bypass ON pack034_iso50001.action_items;
DROP POLICY IF EXISTS p034_ai_tenant_isolation ON pack034_iso50001.action_items;
DROP POLICY IF EXISTS p034_ap_service_bypass ON pack034_iso50001.action_plans;
DROP POLICY IF EXISTS p034_ap_tenant_isolation ON pack034_iso50001.action_plans;
DROP POLICY IF EXISTS p034_tgt_service_bypass ON pack034_iso50001.energy_targets;
DROP POLICY IF EXISTS p034_tgt_tenant_isolation ON pack034_iso50001.energy_targets;
DROP POLICY IF EXISTS p034_obj_service_bypass ON pack034_iso50001.energy_objectives;
DROP POLICY IF EXISTS p034_obj_tenant_isolation ON pack034_iso50001.energy_objectives;

-- Disable RLS
ALTER TABLE IF EXISTS pack034_iso50001.action_items DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.action_plans DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.energy_targets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack034_iso50001.energy_objectives DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p034_ai_updated ON pack034_iso50001.action_items;
DROP TRIGGER IF EXISTS trg_p034_ap_updated ON pack034_iso50001.action_plans;
DROP TRIGGER IF EXISTS trg_p034_tgt_updated ON pack034_iso50001.energy_targets;
DROP TRIGGER IF EXISTS trg_p034_obj_updated ON pack034_iso50001.energy_objectives;

-- Drop tables (reverse FK order)
DROP TABLE IF EXISTS pack034_iso50001.action_items CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.action_plans CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.energy_targets CASCADE;
DROP TABLE IF EXISTS pack034_iso50001.energy_objectives CASCADE;
