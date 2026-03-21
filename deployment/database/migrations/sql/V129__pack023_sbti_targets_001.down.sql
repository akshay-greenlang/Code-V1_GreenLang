-- =============================================================================
-- V129 Down: Rollback PACK-023 Target Definitions and Pathways
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_ambition_updated_at ON pack023_sbti_alignment.pack023_sbti_ambition_assessments;
DROP TRIGGER IF EXISTS trg_pk_pathway_updated_at ON pack023_sbti_alignment.pack023_sbti_pathway_selections;
DROP TRIGGER IF EXISTS trg_pk_boundary_updated_at ON pack023_sbti_alignment.pack023_sbti_target_boundaries;
DROP TRIGGER IF EXISTS trg_pk_targets_updated_at ON pack023_sbti_alignment.pack023_sbti_target_definitions;

-- Drop hypertables
SELECT drop_chunks('pack023_sbti_alignment.pack023_sbti_target_definitions', INTERVAL '1 day');
SELECT detach_hypertable('pack023_sbti_alignment.pack023_sbti_target_definitions', if_exists => TRUE);

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_ambition_assessments;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_pathway_selections;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_target_boundaries;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_target_definitions;

-- Drop function
DROP FUNCTION IF EXISTS pack023_sbti_alignment.set_updated_at();

-- Drop schema
DROP SCHEMA IF EXISTS pack023_sbti_alignment CASCADE;
