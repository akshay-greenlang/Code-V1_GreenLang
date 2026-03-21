-- =============================================================================
-- V223 DOWN: Drop PACK-030 audit triggers and trigger functions
-- =============================================================================

-- Drop triggers
DROP TRIGGER IF EXISTS trg_p030_audit_validation_update ON pack030_nz_reporting.gl_nz_validation_results;
DROP TRIGGER IF EXISTS trg_p030_audit_evidence_insert_update ON pack030_nz_reporting.gl_nz_assurance_evidence;
DROP TRIGGER IF EXISTS trg_p030_audit_narrative_insert_update ON pack030_nz_reporting.gl_nz_narratives;
DROP TRIGGER IF EXISTS trg_p030_audit_metric_delete ON pack030_nz_reporting.gl_nz_report_metrics;
DROP TRIGGER IF EXISTS trg_p030_audit_metric_insert_update ON pack030_nz_reporting.gl_nz_report_metrics;
DROP TRIGGER IF EXISTS trg_p030_audit_section_delete ON pack030_nz_reporting.gl_nz_report_sections;
DROP TRIGGER IF EXISTS trg_p030_audit_section_insert_update ON pack030_nz_reporting.gl_nz_report_sections;
DROP TRIGGER IF EXISTS trg_p030_audit_report_delete ON pack030_nz_reporting.gl_nz_reports;
DROP TRIGGER IF EXISTS trg_p030_audit_report_insert_update ON pack030_nz_reporting.gl_nz_reports;

-- Drop trigger functions
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_audit_validation_changes();
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_audit_evidence_changes();
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_audit_narrative_changes();
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_audit_metric_changes();
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_audit_section_changes();
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_audit_report_changes();
