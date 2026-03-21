-- =============================================================================
-- V137 Down: Rollback Submission Readiness Assessment Records
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_pk_time_updated_at ON pack023_sbti_alignment.pack023_submission_timeline;
DROP TRIGGER IF EXISTS trg_pk_doc_updated_at ON pack023_sbti_alignment.pack023_submission_documentation;
DROP TRIGGER IF EXISTS trg_pk_check_updated_at ON pack023_sbti_alignment.pack023_submission_checklist_items;
DROP TRIGGER IF EXISTS trg_pk_ready_updated_at ON pack023_sbti_alignment.pack023_submission_readiness_assessments;

-- Drop tables
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_submission_timeline;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_submission_documentation;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_submission_checklist_items;
DROP TABLE IF EXISTS pack023_sbti_alignment.pack023_submission_readiness_assessments;
