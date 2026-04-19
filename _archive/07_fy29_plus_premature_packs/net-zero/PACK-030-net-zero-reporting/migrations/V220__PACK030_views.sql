-- =============================================================================
-- V220: PACK-030 Net Zero Reporting Pack - Views
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    010 of 015
-- Date:         March 2026
--
-- Dashboard views for executive monitoring, framework coverage tracking,
-- validation issue overview, upcoming deadline management, and data lineage
-- summary visualization.
--
-- Views (5):
--   1. pack030_nz_reporting.v_reports_summary
--   2. pack030_nz_reporting.v_framework_coverage
--   3. pack030_nz_reporting.v_validation_issues
--   4. pack030_nz_reporting.v_upcoming_deadlines
--   5. pack030_nz_reporting.v_lineage_summary
--
-- Previous: V219__PACK030_indexes.sql
-- =============================================================================

-- =============================================================================
-- View 1: v_reports_summary
-- =============================================================================
-- Executive-level report summary with section/metric counts, consistency
-- scores, validation issue severity breakdown, evidence completeness, and
-- output format status for multi-framework report monitoring.

CREATE OR REPLACE VIEW pack030_nz_reporting.v_reports_summary AS
SELECT
    r.report_id,
    r.tenant_id,
    r.organization_id,
    r.framework,
    r.framework_version,
    r.reporting_year,
    r.reporting_period_start,
    r.reporting_period_end,
    r.report_type,
    r.report_title,
    r.status,
    r.data_completeness_pct,
    r.version_number,
    r.is_latest,
    r.created_at,
    r.approved_at,
    r.published_at,
    -- Section aggregation
    sec_agg.section_count,
    sec_agg.total_word_count,
    sec_agg.avg_consistency_score,
    sec_agg.min_consistency_score,
    sec_agg.sections_pending_review,
    sec_agg.sections_approved,
    sec_agg.language_count,
    -- Metric aggregation
    met_agg.metric_count,
    met_agg.scope1_metric_count,
    met_agg.scope2_metric_count,
    met_agg.scope3_metric_count,
    met_agg.verified_metric_count,
    met_agg.xbrl_tagged_count,
    met_agg.source_system_count,
    -- Validation aggregation
    val_agg.total_issues,
    val_agg.critical_issues,
    val_agg.high_issues,
    val_agg.medium_issues,
    val_agg.low_issues,
    val_agg.unresolved_issues,
    val_agg.blocking_issues,
    -- Evidence aggregation
    ev_agg.evidence_count,
    ev_agg.avg_evidence_completeness,
    ev_agg.unreviewed_evidence,
    -- Overall health
    CASE
        WHEN val_agg.blocking_issues > 0 THEN 'RED'
        WHEN val_agg.critical_issues > 0 THEN 'RED'
        WHEN val_agg.high_issues > 0 THEN 'AMBER'
        WHEN r.data_completeness_pct < 80 THEN 'AMBER'
        WHEN sec_agg.min_consistency_score IS NOT NULL AND sec_agg.min_consistency_score < 70 THEN 'AMBER'
        ELSE 'GREEN'
    END AS health_status,
    -- Readiness score
    CASE
        WHEN r.data_completeness_pct IS NULL THEN 0
        ELSE ROUND((
            r.data_completeness_pct * 0.3 +
            COALESCE(sec_agg.avg_consistency_score, 0) * 0.2 +
            CASE WHEN val_agg.unresolved_issues = 0 THEN 100 ELSE GREATEST(0, 100 - val_agg.unresolved_issues * 5) END * 0.3 +
            COALESCE(ev_agg.avg_evidence_completeness, 0) * 0.2
        )::NUMERIC, 1)
    END AS readiness_score
FROM pack030_nz_reporting.gl_nz_reports r
-- Section aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS section_count,
        COALESCE(SUM(word_count), 0)                                    AS total_word_count,
        ROUND(AVG(consistency_score)::NUMERIC, 2)                       AS avg_consistency_score,
        MIN(consistency_score)                                          AS min_consistency_score,
        COUNT(*) FILTER (WHERE review_status = 'PENDING_REVIEW')        AS sections_pending_review,
        COUNT(*) FILTER (WHERE review_status = 'APPROVED')              AS sections_approved,
        COUNT(DISTINCT language)                                        AS language_count
    FROM pack030_nz_reporting.gl_nz_report_sections
    WHERE report_id = r.report_id AND is_active = TRUE AND is_latest = TRUE
) sec_agg ON TRUE
-- Metric aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS metric_count,
        COUNT(*) FILTER (WHERE scope = 'SCOPE_1')                       AS scope1_metric_count,
        COUNT(*) FILTER (WHERE scope = 'SCOPE_2')                       AS scope2_metric_count,
        COUNT(*) FILTER (WHERE scope = 'SCOPE_3')                       AS scope3_metric_count,
        COUNT(*) FILTER (WHERE verification_status IN ('EXTERNAL_LIMITED', 'EXTERNAL_REASONABLE')) AS verified_metric_count,
        COUNT(*) FILTER (WHERE xbrl_tag IS NOT NULL)                    AS xbrl_tagged_count,
        COUNT(DISTINCT source_system)                                   AS source_system_count
    FROM pack030_nz_reporting.gl_nz_report_metrics
    WHERE report_id = r.report_id AND is_active = TRUE
) met_agg ON TRUE
-- Validation aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS total_issues,
        COUNT(*) FILTER (WHERE severity = 'CRITICAL')                   AS critical_issues,
        COUNT(*) FILTER (WHERE severity = 'HIGH')                       AS high_issues,
        COUNT(*) FILTER (WHERE severity = 'MEDIUM')                     AS medium_issues,
        COUNT(*) FILTER (WHERE severity = 'LOW')                        AS low_issues,
        COUNT(*) FILTER (WHERE resolved = FALSE)                        AS unresolved_issues,
        COUNT(*) FILTER (WHERE blocking = TRUE AND resolved = FALSE)    AS blocking_issues
    FROM pack030_nz_reporting.gl_nz_validation_results
    WHERE report_id = r.report_id AND is_active = TRUE
) val_agg ON TRUE
-- Evidence aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS evidence_count,
        ROUND(AVG(completeness_pct)::NUMERIC, 2)                       AS avg_evidence_completeness,
        COUNT(*) FILTER (WHERE auditor_reviewed = FALSE)                AS unreviewed_evidence
    FROM pack030_nz_reporting.gl_nz_assurance_evidence
    WHERE report_id = r.report_id AND is_active = TRUE
) ev_agg ON TRUE
WHERE r.is_active = TRUE;

-- =============================================================================
-- View 2: v_framework_coverage
-- =============================================================================
-- Framework coverage heatmap showing percentage of required fields populated
-- per framework per organization, with schema version tracking and deadline
-- proximity for executive dashboard display.

CREATE OR REPLACE VIEW pack030_nz_reporting.v_framework_coverage AS
SELECT
    r.organization_id,
    r.tenant_id,
    r.framework,
    r.reporting_year,
    r.reporting_period_start,
    r.reporting_period_end,
    r.status,
    -- Metrics provided
    met_agg.metrics_provided,
    met_agg.distinct_metric_names,
    -- Schema requirements
    fs.total_field_count                AS metrics_required,
    fs.version                          AS schema_version,
    -- Coverage percentage
    CASE
        WHEN fs.total_field_count IS NOT NULL AND fs.total_field_count > 0
        THEN ROUND(100.0 * met_agg.metrics_provided / fs.total_field_count, 2)
        ELSE NULL
    END AS coverage_percentage,
    -- Data completeness
    r.data_completeness_pct,
    -- Deadline
    dl.deadline_date,
    dl.deadline_date - CURRENT_DATE     AS days_to_deadline,
    dl.submission_status,
    -- Coverage classification
    CASE
        WHEN fs.total_field_count IS NULL THEN 'UNKNOWN'
        WHEN fs.total_field_count = 0 THEN 'N/A'
        WHEN (100.0 * met_agg.metrics_provided / fs.total_field_count) >= 95 THEN 'COMPLETE'
        WHEN (100.0 * met_agg.metrics_provided / fs.total_field_count) >= 75 THEN 'NEAR_COMPLETE'
        WHEN (100.0 * met_agg.metrics_provided / fs.total_field_count) >= 50 THEN 'PARTIAL'
        WHEN (100.0 * met_agg.metrics_provided / fs.total_field_count) >= 25 THEN 'LOW'
        ELSE 'MINIMAL'
    END AS coverage_classification
FROM pack030_nz_reporting.gl_nz_reports r
-- Metric count aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                        AS metrics_provided,
        COUNT(DISTINCT metric_name)     AS distinct_metric_names
    FROM pack030_nz_reporting.gl_nz_report_metrics
    WHERE report_id = r.report_id AND is_active = TRUE
) met_agg ON TRUE
-- Current framework schema
LEFT JOIN LATERAL (
    SELECT * FROM pack030_nz_reporting.gl_nz_framework_schemas
    WHERE framework = r.framework
      AND is_current = TRUE AND is_active = TRUE
    ORDER BY effective_date DESC LIMIT 1
) fs ON TRUE
-- Nearest deadline
LEFT JOIN LATERAL (
    SELECT * FROM pack030_nz_reporting.gl_nz_framework_deadlines
    WHERE framework = r.framework
      AND reporting_year = r.reporting_year
      AND is_active = TRUE
    ORDER BY deadline_date ASC LIMIT 1
) dl ON TRUE
WHERE r.is_active = TRUE AND r.is_latest = TRUE;

-- =============================================================================
-- View 3: v_validation_issues
-- =============================================================================
-- Validation issue overview with severity breakdown, category distribution,
-- resolution progress, and blocking issue identification for report quality
-- management.

CREATE OR REPLACE VIEW pack030_nz_reporting.v_validation_issues AS
SELECT
    v.report_id,
    r.organization_id,
    r.tenant_id,
    r.framework,
    r.reporting_year,
    r.status AS report_status,
    -- Issue counts by severity
    COUNT(*) AS total_issues,
    COUNT(*) FILTER (WHERE v.severity = 'CRITICAL')                     AS critical_issues,
    COUNT(*) FILTER (WHERE v.severity = 'HIGH')                         AS high_issues,
    COUNT(*) FILTER (WHERE v.severity = 'MEDIUM')                       AS medium_issues,
    COUNT(*) FILTER (WHERE v.severity = 'LOW')                          AS low_issues,
    COUNT(*) FILTER (WHERE v.severity = 'INFO')                         AS info_issues,
    -- Resolution status
    COUNT(*) FILTER (WHERE v.resolved = FALSE)                          AS unresolved_issues,
    COUNT(*) FILTER (WHERE v.resolved = TRUE)                           AS resolved_issues,
    COUNT(*) FILTER (WHERE v.blocking = TRUE AND v.resolved = FALSE)    AS blocking_issues,
    COUNT(*) FILTER (WHERE v.auto_fixable = TRUE AND v.auto_fix_applied = FALSE) AS auto_fixable_remaining,
    -- Category distribution
    COUNT(*) FILTER (WHERE v.validation_category = 'SCHEMA')            AS schema_issues,
    COUNT(*) FILTER (WHERE v.validation_category = 'COMPLETENESS')      AS completeness_issues,
    COUNT(*) FILTER (WHERE v.validation_category = 'CONSISTENCY')       AS consistency_issues,
    COUNT(*) FILTER (WHERE v.validation_category = 'XBRL')             AS xbrl_issues,
    COUNT(*) FILTER (WHERE v.validation_category = 'CROSS_FRAMEWORK')   AS cross_framework_issues,
    COUNT(*) FILTER (WHERE v.validation_category = 'DATA_QUALITY')      AS data_quality_issues,
    -- Quality score
    CASE
        WHEN COUNT(*) = 0 THEN 100.0
        ELSE ROUND((100.0 - (
            COUNT(*) FILTER (WHERE v.severity = 'CRITICAL' AND v.resolved = FALSE) * 20 +
            COUNT(*) FILTER (WHERE v.severity = 'HIGH' AND v.resolved = FALSE) * 10 +
            COUNT(*) FILTER (WHERE v.severity = 'MEDIUM' AND v.resolved = FALSE) * 5 +
            COUNT(*) FILTER (WHERE v.severity = 'LOW' AND v.resolved = FALSE) * 2
        ))::NUMERIC, 1)
    END AS quality_score
FROM pack030_nz_reporting.gl_nz_validation_results v
JOIN pack030_nz_reporting.gl_nz_reports r ON v.report_id = r.report_id
WHERE v.is_active = TRUE
GROUP BY v.report_id, r.organization_id, r.tenant_id, r.framework, r.reporting_year, r.status;

-- =============================================================================
-- View 4: v_upcoming_deadlines
-- =============================================================================
-- Upcoming framework deadline dashboard with days remaining, submission
-- status, associated report progress, and urgency classification for
-- deadline management.

CREATE OR REPLACE VIEW pack030_nz_reporting.v_upcoming_deadlines AS
SELECT
    d.deadline_id,
    d.tenant_id,
    d.organization_id,
    d.framework,
    d.reporting_year,
    d.deadline_date,
    d.deadline_type,
    d.description,
    d.deadline_date - CURRENT_DATE      AS days_remaining,
    d.submission_status,
    d.extension_granted,
    d.original_deadline,
    -- Associated report status
    rpt.report_id                       AS latest_report_id,
    rpt.status                          AS report_status,
    rpt.data_completeness_pct           AS report_completeness,
    rpt.version_number                  AS report_version,
    -- Urgency classification
    CASE
        WHEN d.submission_status = 'SUBMITTED' THEN 'COMPLETED'
        WHEN d.deadline_date < CURRENT_DATE THEN 'OVERDUE'
        WHEN (d.deadline_date - CURRENT_DATE) <= 7 THEN 'CRITICAL'
        WHEN (d.deadline_date - CURRENT_DATE) <= 30 THEN 'URGENT'
        WHEN (d.deadline_date - CURRENT_DATE) <= 60 THEN 'APPROACHING'
        WHEN (d.deadline_date - CURRENT_DATE) <= 90 THEN 'PLANNED'
        ELSE 'DISTANT'
    END AS urgency,
    -- Readiness
    CASE
        WHEN rpt.status = 'PUBLISHED' THEN 'READY'
        WHEN rpt.status = 'APPROVED' THEN 'NEARLY_READY'
        WHEN rpt.status = 'REVIEW' THEN 'IN_REVIEW'
        WHEN rpt.status = 'IN_PROGRESS' THEN 'IN_PROGRESS'
        WHEN rpt.status = 'DRAFT' THEN 'STARTED'
        WHEN rpt.report_id IS NULL THEN 'NOT_STARTED'
        ELSE 'UNKNOWN'
    END AS readiness
FROM pack030_nz_reporting.gl_nz_framework_deadlines d
-- Latest report for this framework/year
LEFT JOIN LATERAL (
    SELECT * FROM pack030_nz_reporting.gl_nz_reports
    WHERE framework = d.framework
      AND reporting_year = d.reporting_year
      AND (organization_id = d.organization_id OR d.organization_id IS NULL)
      AND is_latest = TRUE AND is_active = TRUE
    ORDER BY created_at DESC LIMIT 1
) rpt ON TRUE
WHERE d.is_active = TRUE
  AND d.deadline_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY d.deadline_date ASC;

-- =============================================================================
-- View 5: v_lineage_summary
-- =============================================================================
-- Data lineage summary with source system diversity, transformation depth,
-- data quality tier distribution, and confidence level overview for
-- lineage transparency and audit readiness.

CREATE OR REPLACE VIEW pack030_nz_reporting.v_lineage_summary AS
SELECT
    l.report_id,
    r.organization_id,
    r.tenant_id,
    r.framework,
    r.reporting_year,
    l.metric_name,
    -- Source diversity
    COUNT(DISTINCT l.source_system)     AS source_system_count,
    COUNT(DISTINCT l.source_pack)       AS source_pack_count,
    COUNT(DISTINCT l.source_app)        AS source_app_count,
    JSONB_AGG(DISTINCT l.source_system) AS source_systems,
    JSONB_AGG(DISTINCT l.source_pack) FILTER (WHERE l.source_pack IS NOT NULL) AS source_packs,
    JSONB_AGG(DISTINCT l.source_app) FILTER (WHERE l.source_app IS NOT NULL) AS source_apps,
    -- Transformation complexity
    MAX(l.transformation_count)         AS max_transformation_depth,
    AVG(l.transformation_count)         AS avg_transformation_depth,
    -- Aggregation
    MAX(l.records_aggregated)           AS max_records_aggregated,
    -- Data quality
    MODE() WITHIN GROUP (ORDER BY l.data_quality_tier) AS predominant_quality_tier,
    AVG(l.confidence_level)             AS avg_confidence_level,
    MIN(l.confidence_level)             AS min_confidence_level,
    -- Count
    COUNT(*)                            AS lineage_record_count
FROM pack030_nz_reporting.gl_nz_data_lineage l
JOIN pack030_nz_reporting.gl_nz_reports r ON l.report_id = r.report_id
WHERE l.is_active = TRUE
GROUP BY l.report_id, r.organization_id, r.tenant_id, r.framework, r.reporting_year, l.metric_name;

-- =============================================================================
-- Comments on Views
-- =============================================================================
COMMENT ON VIEW pack030_nz_reporting.v_reports_summary IS
    'Executive-level report summary with section/metric counts, consistency scoring, validation issue breakdown (severity), evidence completeness, output format status, health classification (RED/AMBER/GREEN), and composite readiness score for multi-framework report monitoring.';

COMMENT ON VIEW pack030_nz_reporting.v_framework_coverage IS
    'Framework coverage heatmap showing percentage of required fields populated per framework, schema version tracking, deadline proximity, and coverage classification (COMPLETE/NEAR_COMPLETE/PARTIAL/LOW/MINIMAL) for executive dashboard display.';

COMMENT ON VIEW pack030_nz_reporting.v_validation_issues IS
    'Validation issue overview with severity breakdown, category distribution, resolution progress, blocking issue identification, auto-fix remaining count, and composite quality score for report quality management.';

COMMENT ON VIEW pack030_nz_reporting.v_upcoming_deadlines IS
    'Upcoming framework deadline dashboard with days remaining, submission status, associated report progress, urgency classification (OVERDUE/CRITICAL/URGENT/APPROACHING/PLANNED/DISTANT), and readiness assessment for deadline management.';

COMMENT ON VIEW pack030_nz_reporting.v_lineage_summary IS
    'Data lineage summary per metric with source system diversity, transformation depth, data quality tier distribution, confidence level overview, and record counts for lineage transparency and audit readiness.';
