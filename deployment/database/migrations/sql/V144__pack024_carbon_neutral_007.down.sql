-- =============================================================================
-- V144 Rollback: Drop Claims Substantiation
-- =============================================================================

DROP TRIGGER IF EXISTS trg_pack024_disc_updated_at ON pack024_carbon_neutral.pack024_claim_disclosure;
DROP TRIGGER IF EXISTS trg_pack024_ver_updated_at ON pack024_carbon_neutral.pack024_claim_verification;
DROP TRIGGER IF EXISTS trg_pack024_evid_updated_at ON pack024_carbon_neutral.pack024_claim_evidence;
DROP TRIGGER IF EXISTS trg_pack024_claim_updated_at ON pack024_carbon_neutral.pack024_claim_substantiation;

DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_claim_disclosure CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_claim_verification CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_claim_evidence CASCADE;
DROP TABLE IF EXISTS pack024_carbon_neutral.pack024_claim_substantiation CASCADE;
