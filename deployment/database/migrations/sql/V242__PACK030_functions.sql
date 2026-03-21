-- =============================================================================
-- V222: PACK-030 Net Zero Reporting Pack - Helper Functions
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    012 of 015
-- Date:         March 2026
--
-- Helper functions for report quality scoring, framework coverage
-- calculation, data completeness assessment, and provenance hash generation.
--
-- Functions (8):
--   1. fn_calculate_report_quality_score
--   2. fn_calculate_framework_coverage
--   3. fn_calculate_data_completeness
--   4. fn_generate_provenance_hash
--   5. fn_get_upcoming_deadlines
--   6. fn_calculate_consistency_score
--   7. fn_check_cross_framework_consistency
--   8. fn_get_report_health_status
--
-- Previous: V221__PACK030_rls_policies.sql
-- =============================================================================

-- =============================================================================
-- Function 1: Calculate report quality score
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_calculate_report_quality_score(
    p_report_id UUID
)
RETURNS NUMERIC
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_completeness NUMERIC;
    v_consistency NUMERIC;
    v_validation NUMERIC;
    v_evidence NUMERIC;
    v_score NUMERIC;
BEGIN
    -- Data completeness (30% weight)
    SELECT COALESCE(data_completeness_pct, 0)
    INTO v_completeness
    FROM pack030_nz_reporting.gl_nz_reports
    WHERE report_id = p_report_id;

    -- Average consistency score (20% weight)
    SELECT COALESCE(AVG(consistency_score), 0)
    INTO v_consistency
    FROM pack030_nz_reporting.gl_nz_report_sections
    WHERE report_id = p_report_id AND is_active = TRUE AND consistency_score IS NOT NULL;

    -- Validation score (30% weight): 100 - penalty for unresolved issues
    SELECT GREATEST(0, 100 - (
        COALESCE(COUNT(*) FILTER (WHERE severity = 'CRITICAL' AND resolved = FALSE), 0) * 20 +
        COALESCE(COUNT(*) FILTER (WHERE severity = 'HIGH' AND resolved = FALSE), 0) * 10 +
        COALESCE(COUNT(*) FILTER (WHERE severity = 'MEDIUM' AND resolved = FALSE), 0) * 5 +
        COALESCE(COUNT(*) FILTER (WHERE severity = 'LOW' AND resolved = FALSE), 0) * 2
    ))
    INTO v_validation
    FROM pack030_nz_reporting.gl_nz_validation_results
    WHERE report_id = p_report_id AND is_active = TRUE;

    -- Evidence completeness (20% weight)
    SELECT COALESCE(AVG(completeness_pct), 0)
    INTO v_evidence
    FROM pack030_nz_reporting.gl_nz_assurance_evidence
    WHERE report_id = p_report_id AND is_active = TRUE;

    -- Weighted score
    v_score := ROUND((
        v_completeness * 0.30 +
        v_consistency * 0.20 +
        v_validation * 0.30 +
        v_evidence * 0.20
    )::NUMERIC, 2);

    RETURN v_score;
END;
$$;

-- =============================================================================
-- Function 2: Calculate framework coverage percentage
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_calculate_framework_coverage(
    p_report_id UUID
)
RETURNS NUMERIC
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_framework VARCHAR(50);
    v_required_count INT;
    v_provided_count INT;
BEGIN
    -- Get framework
    SELECT framework INTO v_framework
    FROM pack030_nz_reporting.gl_nz_reports
    WHERE report_id = p_report_id;

    IF v_framework IS NULL THEN RETURN 0; END IF;

    -- Get required field count from current schema
    SELECT COALESCE(total_field_count, 0) INTO v_required_count
    FROM pack030_nz_reporting.gl_nz_framework_schemas
    WHERE framework = v_framework AND is_current = TRUE AND is_active = TRUE
    ORDER BY effective_date DESC LIMIT 1;

    IF v_required_count = 0 THEN RETURN 100; END IF;

    -- Count provided metrics
    SELECT COUNT(DISTINCT metric_name) INTO v_provided_count
    FROM pack030_nz_reporting.gl_nz_report_metrics
    WHERE report_id = p_report_id AND is_active = TRUE;

    RETURN ROUND((100.0 * v_provided_count / v_required_count)::NUMERIC, 2);
END;
$$;

-- =============================================================================
-- Function 3: Calculate data completeness for a report
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_calculate_data_completeness(
    p_report_id UUID
)
RETURNS NUMERIC
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_section_count INT;
    v_metric_count INT;
    v_narrative_count INT;
    v_has_scope1 BOOLEAN;
    v_has_scope2 BOOLEAN;
    v_has_scope3 BOOLEAN;
    v_score NUMERIC := 0;
BEGIN
    -- Count sections
    SELECT COUNT(*) INTO v_section_count
    FROM pack030_nz_reporting.gl_nz_report_sections
    WHERE report_id = p_report_id AND is_active = TRUE;

    -- Count metrics
    SELECT COUNT(*) INTO v_metric_count
    FROM pack030_nz_reporting.gl_nz_report_metrics
    WHERE report_id = p_report_id AND is_active = TRUE;

    -- Check scope coverage
    SELECT
        EXISTS(SELECT 1 FROM pack030_nz_reporting.gl_nz_report_metrics WHERE report_id = p_report_id AND scope = 'SCOPE_1' AND is_active = TRUE),
        EXISTS(SELECT 1 FROM pack030_nz_reporting.gl_nz_report_metrics WHERE report_id = p_report_id AND scope = 'SCOPE_2' AND is_active = TRUE),
        EXISTS(SELECT 1 FROM pack030_nz_reporting.gl_nz_report_metrics WHERE report_id = p_report_id AND scope = 'SCOPE_3' AND is_active = TRUE)
    INTO v_has_scope1, v_has_scope2, v_has_scope3;

    -- Score: sections (25%), metrics (25%), scope coverage (50%)
    IF v_section_count > 0 THEN v_score := v_score + 25; END IF;
    IF v_metric_count > 0 THEN v_score := v_score + 25; END IF;
    IF v_has_scope1 THEN v_score := v_score + 20; END IF;
    IF v_has_scope2 THEN v_score := v_score + 15; END IF;
    IF v_has_scope3 THEN v_score := v_score + 15; END IF;

    RETURN v_score;
END;
$$;

-- =============================================================================
-- Function 4: Generate provenance hash (SHA-256 simulation)
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_generate_provenance_hash(
    p_data TEXT
)
RETURNS CHAR(64)
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN encode(digest(p_data, 'sha256'), 'hex');
END;
$$;

-- =============================================================================
-- Function 5: Get upcoming deadlines for an organization
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_get_upcoming_deadlines(
    p_organization_id UUID,
    p_days_ahead INT DEFAULT 90
)
RETURNS TABLE(
    framework VARCHAR(50),
    reporting_year INT,
    deadline_date DATE,
    days_remaining INT,
    submission_status VARCHAR(30),
    urgency TEXT
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.framework,
        d.reporting_year,
        d.deadline_date,
        (d.deadline_date - CURRENT_DATE)::INT,
        d.submission_status,
        CASE
            WHEN d.submission_status = 'SUBMITTED' THEN 'COMPLETED'
            WHEN d.deadline_date < CURRENT_DATE THEN 'OVERDUE'
            WHEN (d.deadline_date - CURRENT_DATE) <= 7 THEN 'CRITICAL'
            WHEN (d.deadline_date - CURRENT_DATE) <= 30 THEN 'URGENT'
            WHEN (d.deadline_date - CURRENT_DATE) <= 60 THEN 'APPROACHING'
            ELSE 'PLANNED'
        END
    FROM pack030_nz_reporting.gl_nz_framework_deadlines d
    WHERE (d.organization_id = p_organization_id OR d.organization_id IS NULL)
      AND d.is_active = TRUE
      AND d.deadline_date <= CURRENT_DATE + p_days_ahead
      AND d.deadline_date >= CURRENT_DATE - 30
    ORDER BY d.deadline_date ASC;
END;
$$;

-- =============================================================================
-- Function 6: Calculate narrative consistency score
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_calculate_consistency_score(
    p_organization_id UUID,
    p_reporting_year INT
)
RETURNS NUMERIC
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_avg_score NUMERIC;
BEGIN
    SELECT AVG(s.consistency_score)
    INTO v_avg_score
    FROM pack030_nz_reporting.gl_nz_report_sections s
    JOIN pack030_nz_reporting.gl_nz_reports r ON s.report_id = r.report_id
    WHERE r.organization_id = p_organization_id
      AND r.reporting_year = p_reporting_year
      AND r.is_active = TRUE
      AND s.is_active = TRUE
      AND s.consistency_score IS NOT NULL;

    RETURN COALESCE(ROUND(v_avg_score::NUMERIC, 2), 0);
END;
$$;

-- =============================================================================
-- Function 7: Check cross-framework consistency
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_check_cross_framework_consistency(
    p_organization_id UUID,
    p_reporting_year INT,
    p_metric_name VARCHAR(200)
)
RETURNS TABLE(
    framework VARCHAR(50),
    metric_value NUMERIC,
    unit VARCHAR(50),
    source_system VARCHAR(100),
    is_consistent BOOLEAN
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_min_value NUMERIC;
    v_max_value NUMERIC;
    v_tolerance NUMERIC := 0.01; -- 1% tolerance
BEGIN
    -- Get min/max values across frameworks
    SELECT MIN(m.metric_value), MAX(m.metric_value)
    INTO v_min_value, v_max_value
    FROM pack030_nz_reporting.gl_nz_report_metrics m
    JOIN pack030_nz_reporting.gl_nz_reports r ON m.report_id = r.report_id
    WHERE r.organization_id = p_organization_id
      AND r.reporting_year = p_reporting_year
      AND m.metric_name = p_metric_name
      AND r.is_active = TRUE
      AND m.is_active = TRUE;

    RETURN QUERY
    SELECT
        r.framework,
        m.metric_value,
        m.unit,
        m.source_system,
        CASE
            WHEN v_max_value = 0 THEN TRUE
            WHEN ABS(m.metric_value - v_min_value) / NULLIF(v_max_value, 0) <= v_tolerance THEN TRUE
            ELSE FALSE
        END
    FROM pack030_nz_reporting.gl_nz_report_metrics m
    JOIN pack030_nz_reporting.gl_nz_reports r ON m.report_id = r.report_id
    WHERE r.organization_id = p_organization_id
      AND r.reporting_year = p_reporting_year
      AND m.metric_name = p_metric_name
      AND r.is_active = TRUE
      AND m.is_active = TRUE
    ORDER BY r.framework;
END;
$$;

-- =============================================================================
-- Function 8: Get report health status
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_get_report_health_status(
    p_report_id UUID
)
RETURNS VARCHAR(10)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_blocking_count INT;
    v_critical_count INT;
    v_high_count INT;
    v_completeness NUMERIC;
BEGIN
    -- Count blocking issues
    SELECT COUNT(*) INTO v_blocking_count
    FROM pack030_nz_reporting.gl_nz_validation_results
    WHERE report_id = p_report_id AND blocking = TRUE AND resolved = FALSE AND is_active = TRUE;

    IF v_blocking_count > 0 THEN RETURN 'RED'; END IF;

    -- Count critical issues
    SELECT COUNT(*) INTO v_critical_count
    FROM pack030_nz_reporting.gl_nz_validation_results
    WHERE report_id = p_report_id AND severity = 'CRITICAL' AND resolved = FALSE AND is_active = TRUE;

    IF v_critical_count > 0 THEN RETURN 'RED'; END IF;

    -- Count high issues
    SELECT COUNT(*) INTO v_high_count
    FROM pack030_nz_reporting.gl_nz_validation_results
    WHERE report_id = p_report_id AND severity = 'HIGH' AND resolved = FALSE AND is_active = TRUE;

    IF v_high_count > 0 THEN RETURN 'AMBER'; END IF;

    -- Check completeness
    SELECT data_completeness_pct INTO v_completeness
    FROM pack030_nz_reporting.gl_nz_reports
    WHERE report_id = p_report_id;

    IF v_completeness IS NOT NULL AND v_completeness < 80 THEN RETURN 'AMBER'; END IF;

    RETURN 'GREEN';
END;
$$;

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_report_quality_score(UUID) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_framework_coverage(UUID) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_data_completeness(UUID) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_generate_provenance_hash(TEXT) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_upcoming_deadlines(UUID, INT) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_calculate_consistency_score(UUID, INT) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_check_cross_framework_consistency(UUID, INT, VARCHAR) TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_get_report_health_status(UUID) TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON FUNCTION pack030_nz_reporting.fn_calculate_report_quality_score IS
    'Calculates a weighted quality score (0-100) for a report based on data completeness (30%), narrative consistency (20%), validation results (30%), and evidence completeness (20%).';

COMMENT ON FUNCTION pack030_nz_reporting.fn_calculate_framework_coverage IS
    'Calculates framework coverage percentage by comparing provided metrics against required fields defined in the current framework schema.';

COMMENT ON FUNCTION pack030_nz_reporting.fn_calculate_data_completeness IS
    'Calculates data completeness score based on section presence, metric count, and scope coverage (Scope 1/2/3).';

COMMENT ON FUNCTION pack030_nz_reporting.fn_generate_provenance_hash IS
    'Generates a SHA-256 provenance hash for data integrity verification and audit trail.';

COMMENT ON FUNCTION pack030_nz_reporting.fn_get_upcoming_deadlines IS
    'Returns upcoming framework deadlines for an organization with urgency classification (OVERDUE/CRITICAL/URGENT/APPROACHING/PLANNED).';

COMMENT ON FUNCTION pack030_nz_reporting.fn_calculate_consistency_score IS
    'Calculates the average narrative consistency score across all reports for an organization and reporting year.';

COMMENT ON FUNCTION pack030_nz_reporting.fn_check_cross_framework_consistency IS
    'Checks whether a specific metric has consistent values across all frameworks for an organization, flagging inconsistencies beyond 1% tolerance.';

COMMENT ON FUNCTION pack030_nz_reporting.fn_get_report_health_status IS
    'Returns RED/AMBER/GREEN health status for a report based on blocking issues, critical/high validation issues, and data completeness.';
