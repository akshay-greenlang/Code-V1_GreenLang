-- =============================================================================
-- V133 Down: Rollback FLAG Assessment and Commodity Records
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_supp_updated_at ON pack023_sbti_alignment.pack023_flag_supply_chain_assessment;
DROP TRIGGER IF EXISTS trg_pk_defor_updated_at ON pack023_sbti_alignment.pack023_flag_deforestation_commitments;
DROP TRIGGER IF EXISTS trg_pk_comm_updated_at ON pack023_sbti_alignment.pack023_flag_commodity_breakdown;
DROP TRIGGER IF EXISTS trg_pk_flag_updated_at ON pack023_sbti_alignment.pack023_flag_assessments;

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_flag_supply_chain_assessment;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_flag_deforestation_commitments;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_flag_commodity_breakdown;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_flag_assessments;
