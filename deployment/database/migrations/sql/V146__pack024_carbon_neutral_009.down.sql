-- =============================================================================
-- V146 Rollback: Drop Annual Cycles
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_gov_updated_at ON pack024_carbon_neutral.pack024_annual_governance_calendar;
DROP TRIGGER IF EXISTS trg_pack024_review_updated_at ON pack024_carbon_neutral.pack024_annual_review_schedule;
DROP TRIGGER IF EXISTS trg_pack024_inv_proc_updated_at ON pack024_carbon_neutral.pack024_annual_inventory_process;
DROP TRIGGER IF EXISTS trg_pack024_cycle_updated_at ON pack024_carbon_neutral.pack024_annual_cycles;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_annual_governance_calendar CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_annual_review_schedule CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_annual_inventory_process CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_annual_cycles CASCADE;
