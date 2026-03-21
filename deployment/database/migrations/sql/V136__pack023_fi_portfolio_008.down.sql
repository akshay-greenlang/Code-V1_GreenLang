-- =============================================================================
-- V136 Down: Rollback FI Portfolio Targets and PCAF Scores
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_engage_updated_at ON pack023_sbti_alignment.pack023_fi_engagement_targets;
DROP TRIGGER IF EXISTS trg_pk_asset_updated_at ON pack023_sbti_alignment.pack023_fi_asset_class_coverage;
DROP TRIGGER IF EXISTS trg_pk_pcaf_updated_at ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality;
DROP TRIGGER IF EXISTS trg_pk_fi_updated_at ON pack023_sbti_alignment.pack023_fi_portfolio_targets;

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_fi_engagement_targets;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_fi_asset_class_coverage;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_fi_pcaf_data_quality;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_fi_portfolio_targets;
