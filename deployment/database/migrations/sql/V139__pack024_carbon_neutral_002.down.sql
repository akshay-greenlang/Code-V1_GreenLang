-- =============================================================================
-- V139 Rollback: Drop Carbon Management Plans
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_assign_updated_at ON pack024_carbon_neutral.pack024_action_assignments;
DROP TRIGGER IF EXISTS trg_pack024_action_updated_at ON pack024_carbon_neutral.pack024_management_actions;
DROP TRIGGER IF EXISTS trg_pack024_pathway_updated_at ON pack024_carbon_neutral.pack024_reduction_pathways;
DROP TRIGGER IF EXISTS trg_pack024_mgmt_updated_at ON pack024_carbon_neutral.pack024_management_plans;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_action_assignments CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_management_actions CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_reduction_pathways CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_management_plans CASCADE;
