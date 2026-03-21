-- =============================================================================
-- V222 DOWN: Drop PACK-030 helper functions
-- =============================================================================

DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_get_report_health_status(UUID);
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_check_cross_framework_consistency(UUID, INT, VARCHAR);
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_calculate_consistency_score(UUID, INT);
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_get_upcoming_deadlines(UUID, INT);
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_generate_provenance_hash(TEXT);
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_calculate_data_completeness(UUID);
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_calculate_framework_coverage(UUID);
DROP FUNCTION IF EXISTS pack030_nz_reporting.fn_calculate_report_quality_score(UUID);
