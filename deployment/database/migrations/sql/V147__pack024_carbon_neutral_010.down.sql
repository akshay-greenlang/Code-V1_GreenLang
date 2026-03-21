-- =============================================================================
-- V147 Rollback: Drop Permanence Assessments
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_insure_updated_at ON pack024_carbon_neutral.pack024_permanence_insurance;
DROP TRIGGER IF EXISTS trg_pack024_monitor_updated_at ON pack024_carbon_neutral.pack024_permanence_monitoring;
DROP TRIGGER IF EXISTS trg_pack024_factor_updated_at ON pack024_carbon_neutral.pack024_permanence_risk_factors;
DROP TRIGGER IF EXISTS trg_pack024_perm_updated_at ON pack024_carbon_neutral.pack024_permanence_assessments;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_permanence_insurance CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_permanence_monitoring CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_permanence_risk_factors CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_permanence_assessments CASCADE;
