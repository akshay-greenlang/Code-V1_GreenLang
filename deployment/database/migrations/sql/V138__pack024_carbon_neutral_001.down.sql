-- =============================================================================
-- V138 Rollback: Drop Carbon Footprint Quantification Records
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_unc_updated_at ON pack024_carbon_neutral.pack024_uncertainty_assessment;
DROP TRIGGER IF EXISTS trg_pack024_base_updated_at ON pack024_carbon_neutral.pack024_baseline_establishment;
DROP TRIGGER IF EXISTS trg_pack024_fpc_updated_at ON pack024_carbon_neutral.pack024_footprint_components;
DROP TRIGGER IF EXISTS trg_pack024_fp_updated_at ON pack024_carbon_neutral.pack024_footprint_records;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_uncertainty_assessment CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_baseline_establishment CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_footprint_components CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_footprint_records CASCADE;

DROP FUNCTION IF EXISTS pack024_carbon_neutral.set_updated_at();

DROP SCHEMA IF EXISTS pack024_carbon_neutral CASCADE;
