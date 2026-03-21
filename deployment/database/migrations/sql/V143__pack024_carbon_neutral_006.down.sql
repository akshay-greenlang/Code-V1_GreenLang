-- =============================================================================
-- V143 Rollback: Drop Neutralization Balance Reconciliation
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_trend_updated_at ON pack024_carbon_neutral.pack024_balance_trend_analysis;
DROP TRIGGER IF EXISTS trg_pack024_nz_updated_at ON pack024_carbon_neutral.pack024_net_zero_achievement;
DROP TRIGGER IF EXISTS trg_pack024_recon_updated_at ON pack024_carbon_neutral.pack024_balance_reconciliation;
DROP TRIGGER IF EXISTS trg_pack024_bal_updated_at ON pack024_carbon_neutral.pack024_neutralization_balance;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_balance_trend_analysis CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_net_zero_achievement CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_balance_reconciliation CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_neutralization_balance CASCADE;
