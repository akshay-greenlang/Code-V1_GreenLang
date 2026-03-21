-- =============================================================================
-- V151 DOWN: Drop action_plans and action_items tables
-- =============================================================================

-- Drop RLS policies
DROP POLICY IF EXISTS p025_ai_service_bypass ON pack025_race_to_zero.action_items;
DROP POLICY IF EXISTS p025_ai_tenant_isolation ON pack025_race_to_zero.action_items;
DROP POLICY IF EXISTS p025_ap_service_bypass ON pack025_race_to_zero.action_plans;
DROP POLICY IF EXISTS p025_ap_tenant_isolation ON pack025_race_to_zero.action_plans;

-- Disable RLS
ALTER TABLE IF EXISTS pack025_race_to_zero.action_items DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS pack025_race_to_zero.action_plans DISABLE ROW LEVEL SECURITY;

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p025_action_items_updated ON pack025_race_to_zero.action_items;
DROP TRIGGER IF EXISTS trg_p025_action_plans_updated ON pack025_race_to_zero.action_plans;

-- Drop tables (order matters for FK)
DROP TABLE IF EXISTS pack025_race_to_zero.action_items CASCADE;
DROP TABLE IF EXISTS pack025_race_to_zero.action_plans CASCADE;
