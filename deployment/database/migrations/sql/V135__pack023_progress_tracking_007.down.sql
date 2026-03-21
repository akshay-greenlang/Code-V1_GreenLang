-- =============================================================================
-- V135 Down: Rollback Progress Tracking and Recalculation Records
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_adj_updated_at ON pack023_sbti_alignment.pack023_recalculation_adjustments;
DROP TRIGGER IF EXISTS trg_pk_recalc_updated_at ON pack023_sbti_alignment.pack023_recalculation_events;
DROP TRIGGER IF EXISTS trg_pk_var_updated_at ON pack023_sbti_alignment.pack023_progress_variance_analysis;
DROP TRIGGER IF EXISTS trg_pk_prog_updated_at ON pack023_sbti_alignment.pack023_progress_tracking_records;

-- Drop hypertables
SELECT drop_chunks('pack023_sbti_alignment.pack023_progress_tracking_records', INTERVAL '1 day');
SELECT detach_hypertable('pack023_sbti_alignment.pack023_progress_tracking_records', if_exists => TRUE);

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_recalculation_adjustments;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_recalculation_events;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_progress_variance_analysis;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_progress_tracking_records;
