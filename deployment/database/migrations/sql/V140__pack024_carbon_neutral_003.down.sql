-- =============================================================================
-- V140 Rollback: Drop Carbon Credit Inventory
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_add_updated_at ON pack024_carbon_neutral.pack024_additionality_assessment;
DROP TRIGGER IF EXISTS trg_pack024_val_updated_at ON pack024_carbon_neutral.pack024_credit_validation;
DROP TRIGGER IF EXISTS trg_pack024_trans_updated_at ON pack024_carbon_neutral.pack024_credit_transactions;
DROP TRIGGER IF EXISTS trg_pack024_inv_updated_at ON pack024_carbon_neutral.pack024_credit_inventory;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_additionality_assessment CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_credit_validation CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_credit_transactions CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_credit_inventory CASCADE;
