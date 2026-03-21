-- =============================================================================
-- V130 Down: Rollback 42-Criterion Validation Results
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_rem_updated_at ON pack023_sbti_alignment.pack023_sbti_remediation_guidance;
DROP TRIGGER IF EXISTS trg_pk_gap_updated_at ON pack023_sbti_alignment.pack023_sbti_validation_gaps;
DROP TRIGGER IF EXISTS trg_pk_crit_updated_at ON pack023_sbti_alignment.pack023_sbti_criterion_results;
DROP TRIGGER IF EXISTS trg_pk_val_updated_at ON pack023_sbti_alignment.pack023_sbti_validation_assessments;

-- Drop hypertables
SELECT drop_chunks('pack023_sbti_alignment.pack023_sbti_validation_assessments', INTERVAL '1 day');
SELECT detach_hypertable('pack023_sbti_alignment.pack023_sbti_validation_assessments', if_exists => TRUE);

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_remediation_guidance;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_validation_gaps;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_criterion_results;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_sbti_validation_assessments;
