-- =============================================================================
-- V142 Rollback: Drop Registry Retirements
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_cert_updated_at ON pack024_carbon_neutral.pack024_retirement_certificates;
DROP TRIGGER IF EXISTS trg_pack024_sub_updated_at ON pack024_carbon_neutral.pack024_registry_submissions;
DROP TRIGGER IF EXISTS trg_pack024_stmt_updated_at ON pack024_carbon_neutral.pack024_retirement_statements;
DROP TRIGGER IF EXISTS trg_pack024_ret_updated_at ON pack024_carbon_neutral.pack024_retirement_records;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_retirement_certificates CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_registry_submissions CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_retirement_statements CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_retirement_records CASCADE;
