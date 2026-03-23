-- =============================================================================
-- V308: PACK-039 Energy Monitoring Pack - Data Validation
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates data validation tables for automated quality assurance of meter
-- readings. Includes configurable validation rules, per-interval validation
-- results, data quality scoring, correction records with audit trail, and
-- completeness monitoring logs. The validation engine applies rules to
-- every incoming interval and produces quality-scored, validated data for
-- downstream consumption by EnPI, cost allocation, and reporting engines.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_validation_rules
--   2. pack039_energy_monitoring.em_validation_results
--   3. pack039_energy_monitoring.em_quality_scores
--   4. pack039_energy_monitoring.em_data_corrections
--   5. pack039_energy_monitoring.em_completeness_logs
--
-- Previous: V307__pack039_energy_monitoring_002.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_validation_rules
-- =============================================================================
-- Configurable validation rules applied to interval data during the VEE
-- (Validation, Estimation, Editing) process. Rules define bounds checking,
-- rate-of-change limits, statistical outlier detection, meter balance
-- verification, and custom formula-based validations. Each rule produces
-- a pass/fail/warning result with configurable severity.

CREATE TABLE pack039_energy_monitoring.em_validation_rules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    rule_name               VARCHAR(255)    NOT NULL,
    rule_code               VARCHAR(50)     NOT NULL,
    rule_category           VARCHAR(50)     NOT NULL DEFAULT 'RANGE_CHECK',
    rule_type               VARCHAR(50)     NOT NULL DEFAULT 'THRESHOLD',
    severity                VARCHAR(20)     NOT NULL DEFAULT 'WARNING',
    description             TEXT,
    applies_to_meter_types  VARCHAR(50)[]   DEFAULT '{}',
    applies_to_energy_types VARCHAR(50)[]   DEFAULT '{}',
    applies_to_meter_ids    UUID[],
    parameter_config        JSONB           NOT NULL DEFAULT '{}',
    threshold_min           NUMERIC(18,6),
    threshold_max           NUMERIC(18,6),
    rate_of_change_max      NUMERIC(12,4),
    rate_of_change_period_minutes INTEGER,
    statistical_method      VARCHAR(30),
    statistical_window_days INTEGER,
    statistical_sigma       NUMERIC(5,2),
    formula_expression      TEXT,
    comparison_meter_id     UUID,
    tolerance_pct           NUMERIC(7,4),
    auto_correct            BOOLEAN         NOT NULL DEFAULT false,
    correction_method       VARCHAR(30),
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    is_mandatory            BOOLEAN         NOT NULL DEFAULT false,
    execution_order         INTEGER         NOT NULL DEFAULT 100,
    effective_from          DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to            DATE,
    last_triggered_at       TIMESTAMPTZ,
    trigger_count           BIGINT          NOT NULL DEFAULT 0,
    false_positive_count    BIGINT          NOT NULL DEFAULT 0,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p039_vr_category CHECK (
        rule_category IN (
            'RANGE_CHECK', 'RATE_OF_CHANGE', 'STATISTICAL',
            'BALANCE_CHECK', 'COMPLETENESS', 'CONSISTENCY',
            'PATTERN', 'FORMULA', 'CROSS_METER', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_vr_type CHECK (
        rule_type IN (
            'THRESHOLD', 'ABSOLUTE_RANGE', 'PERCENTAGE_RANGE',
            'RATE_LIMIT', 'ZSCORE', 'IQR', 'MOVING_AVERAGE',
            'COMPARISON', 'SUM_CHECK', 'GAP_DETECTION',
            'STUCK_VALUE', 'NEGATIVE_CHECK', 'FORMULA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_vr_severity CHECK (
        severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')
    ),
    CONSTRAINT chk_p039_vr_stat_method CHECK (
        statistical_method IS NULL OR statistical_method IN (
            'ZSCORE', 'IQR', 'MOVING_AVERAGE', 'EWMA',
            'GRUBBS', 'DIXON', 'MAD', 'PERCENTILE'
        )
    ),
    CONSTRAINT chk_p039_vr_correction CHECK (
        correction_method IS NULL OR correction_method IN (
            'LINEAR_INTERPOLATION', 'PREVIOUS_VALUE', 'AVERAGE',
            'MEDIAN', 'REGRESSION', 'ZERO', 'MANUAL', 'PROFILE_BASED'
        )
    ),
    CONSTRAINT chk_p039_vr_threshold CHECK (
        threshold_min IS NULL OR threshold_max IS NULL OR
        threshold_min <= threshold_max
    ),
    CONSTRAINT chk_p039_vr_sigma CHECK (
        statistical_sigma IS NULL OR (statistical_sigma > 0 AND statistical_sigma <= 10)
    ),
    CONSTRAINT chk_p039_vr_tolerance CHECK (
        tolerance_pct IS NULL OR (tolerance_pct >= 0 AND tolerance_pct <= 100)
    ),
    CONSTRAINT chk_p039_vr_order CHECK (
        execution_order >= 1 AND execution_order <= 9999
    ),
    CONSTRAINT chk_p039_vr_dates CHECK (
        effective_to IS NULL OR effective_from <= effective_to
    ),
    CONSTRAINT chk_p039_vr_counts CHECK (
        trigger_count >= 0 AND false_positive_count >= 0
    ),
    CONSTRAINT uq_p039_vr_tenant_code UNIQUE (tenant_id, rule_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_vr_tenant          ON pack039_energy_monitoring.em_validation_rules(tenant_id);
CREATE INDEX idx_p039_vr_code            ON pack039_energy_monitoring.em_validation_rules(rule_code);
CREATE INDEX idx_p039_vr_category        ON pack039_energy_monitoring.em_validation_rules(rule_category);
CREATE INDEX idx_p039_vr_type            ON pack039_energy_monitoring.em_validation_rules(rule_type);
CREATE INDEX idx_p039_vr_severity        ON pack039_energy_monitoring.em_validation_rules(severity);
CREATE INDEX idx_p039_vr_enabled         ON pack039_energy_monitoring.em_validation_rules(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_vr_mandatory       ON pack039_energy_monitoring.em_validation_rules(is_mandatory) WHERE is_mandatory = true;
CREATE INDEX idx_p039_vr_order           ON pack039_energy_monitoring.em_validation_rules(execution_order);
CREATE INDEX idx_p039_vr_created         ON pack039_energy_monitoring.em_validation_rules(created_at DESC);
CREATE INDEX idx_p039_vr_meter_types     ON pack039_energy_monitoring.em_validation_rules USING GIN(applies_to_meter_types);
CREATE INDEX idx_p039_vr_energy_types    ON pack039_energy_monitoring.em_validation_rules USING GIN(applies_to_energy_types);
CREATE INDEX idx_p039_vr_params          ON pack039_energy_monitoring.em_validation_rules USING GIN(parameter_config);

-- Composite: active rules in execution order
CREATE INDEX idx_p039_vr_active_order    ON pack039_energy_monitoring.em_validation_rules(execution_order, severity)
    WHERE is_enabled = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_vr_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_validation_rules
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_validation_results
-- =============================================================================
-- Per-interval validation results produced by applying rules to meter data.
-- Each row records the outcome of a single rule applied to a single interval
-- reading. Results drive data quality scoring, correction workflows, and
-- data quality dashboards. High-volume table designed for time-partitioned
-- storage.

CREATE TABLE pack039_energy_monitoring.em_validation_results (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    interval_data_id        UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_interval_data(id) ON DELETE CASCADE,
    rule_id                 UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_validation_rules(id),
    meter_id                UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    validation_timestamp    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    data_timestamp          TIMESTAMPTZ     NOT NULL,
    result_status           VARCHAR(20)     NOT NULL DEFAULT 'PASS',
    original_value          NUMERIC(18,6),
    validated_value         NUMERIC(18,6),
    expected_min            NUMERIC(18,6),
    expected_max            NUMERIC(18,6),
    deviation_value         NUMERIC(18,6),
    deviation_pct           NUMERIC(10,4),
    confidence_score        NUMERIC(5,2),
    auto_corrected          BOOLEAN         NOT NULL DEFAULT false,
    correction_applied      VARCHAR(30),
    correction_value        NUMERIC(18,6),
    is_false_positive       BOOLEAN         NOT NULL DEFAULT false,
    reviewed_by             UUID,
    reviewed_at             TIMESTAMPTZ,
    review_notes            TEXT,
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_vres_status CHECK (
        result_status IN ('PASS', 'FAIL', 'WARNING', 'SKIP', 'ERROR')
    ),
    CONSTRAINT chk_p039_vres_confidence CHECK (
        confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 100)
    ),
    CONSTRAINT chk_p039_vres_correction CHECK (
        correction_applied IS NULL OR correction_applied IN (
            'LINEAR_INTERPOLATION', 'PREVIOUS_VALUE', 'AVERAGE',
            'MEDIAN', 'REGRESSION', 'ZERO', 'MANUAL', 'PROFILE_BASED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_vres_interval      ON pack039_energy_monitoring.em_validation_results(interval_data_id);
CREATE INDEX idx_p039_vres_rule          ON pack039_energy_monitoring.em_validation_results(rule_id);
CREATE INDEX idx_p039_vres_meter         ON pack039_energy_monitoring.em_validation_results(meter_id);
CREATE INDEX idx_p039_vres_tenant        ON pack039_energy_monitoring.em_validation_results(tenant_id);
CREATE INDEX idx_p039_vres_val_ts        ON pack039_energy_monitoring.em_validation_results(validation_timestamp DESC);
CREATE INDEX idx_p039_vres_data_ts       ON pack039_energy_monitoring.em_validation_results(data_timestamp DESC);
CREATE INDEX idx_p039_vres_status        ON pack039_energy_monitoring.em_validation_results(result_status);
CREATE INDEX idx_p039_vres_auto_corr     ON pack039_energy_monitoring.em_validation_results(auto_corrected) WHERE auto_corrected = true;
CREATE INDEX idx_p039_vres_false_pos     ON pack039_energy_monitoring.em_validation_results(is_false_positive) WHERE is_false_positive = true;
CREATE INDEX idx_p039_vres_created       ON pack039_energy_monitoring.em_validation_results(created_at DESC);
CREATE INDEX idx_p039_vres_details       ON pack039_energy_monitoring.em_validation_results USING GIN(details);

-- Composite: failed validations by meter for dashboard
CREATE INDEX idx_p039_vres_meter_fail    ON pack039_energy_monitoring.em_validation_results(meter_id, data_timestamp DESC)
    WHERE result_status IN ('FAIL', 'WARNING');

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_quality_scores
-- =============================================================================
-- Aggregated data quality scores per meter, period, and dimension.
-- Scores are calculated from validation results and provide a summary
-- measure of data reliability. Used for data quality dashboards, SLA
-- compliance monitoring, and trust indicators on reports.

CREATE TABLE pack039_energy_monitoring.em_quality_scores (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    score_period_type       VARCHAR(20)     NOT NULL DEFAULT 'DAILY',
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    overall_score           NUMERIC(5,2)    NOT NULL,
    completeness_score      NUMERIC(5,2)    NOT NULL DEFAULT 100.0,
    accuracy_score          NUMERIC(5,2)    NOT NULL DEFAULT 100.0,
    consistency_score       NUMERIC(5,2)    NOT NULL DEFAULT 100.0,
    timeliness_score        NUMERIC(5,2)    NOT NULL DEFAULT 100.0,
    plausibility_score      NUMERIC(5,2)    NOT NULL DEFAULT 100.0,
    total_intervals         INTEGER         NOT NULL DEFAULT 0,
    valid_intervals         INTEGER         NOT NULL DEFAULT 0,
    estimated_intervals     INTEGER         NOT NULL DEFAULT 0,
    missing_intervals       INTEGER         NOT NULL DEFAULT 0,
    rejected_intervals      INTEGER         NOT NULL DEFAULT 0,
    corrected_intervals     INTEGER         NOT NULL DEFAULT 0,
    rule_failures           INTEGER         NOT NULL DEFAULT 0,
    critical_failures       INTEGER         NOT NULL DEFAULT 0,
    warning_count           INTEGER         NOT NULL DEFAULT 0,
    data_latency_avg_seconds INTEGER,
    data_latency_max_seconds INTEGER,
    quality_grade           VARCHAR(5),
    meets_sla               BOOLEAN,
    sla_target_score        NUMERIC(5,2),
    score_details           JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_qs_period_type CHECK (
        score_period_type IN ('HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUAL')
    ),
    CONSTRAINT chk_p039_qs_overall CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_p039_qs_completeness CHECK (
        completeness_score >= 0 AND completeness_score <= 100
    ),
    CONSTRAINT chk_p039_qs_accuracy CHECK (
        accuracy_score >= 0 AND accuracy_score <= 100
    ),
    CONSTRAINT chk_p039_qs_consistency CHECK (
        consistency_score >= 0 AND consistency_score <= 100
    ),
    CONSTRAINT chk_p039_qs_timeliness CHECK (
        timeliness_score >= 0 AND timeliness_score <= 100
    ),
    CONSTRAINT chk_p039_qs_plausibility CHECK (
        plausibility_score >= 0 AND plausibility_score <= 100
    ),
    CONSTRAINT chk_p039_qs_intervals CHECK (
        total_intervals >= 0 AND valid_intervals >= 0 AND
        estimated_intervals >= 0 AND missing_intervals >= 0 AND
        rejected_intervals >= 0 AND corrected_intervals >= 0
    ),
    CONSTRAINT chk_p039_qs_failures CHECK (
        rule_failures >= 0 AND critical_failures >= 0 AND warning_count >= 0
    ),
    CONSTRAINT chk_p039_qs_grade CHECK (
        quality_grade IS NULL OR quality_grade IN ('A+', 'A', 'B+', 'B', 'C', 'D', 'F')
    ),
    CONSTRAINT chk_p039_qs_sla_target CHECK (
        sla_target_score IS NULL OR (sla_target_score >= 0 AND sla_target_score <= 100)
    ),
    CONSTRAINT chk_p039_qs_dates CHECK (
        period_start <= period_end
    ),
    CONSTRAINT uq_p039_qs_meter_period UNIQUE (meter_id, score_period_type, period_start, period_end)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_qs_meter           ON pack039_energy_monitoring.em_quality_scores(meter_id);
CREATE INDEX idx_p039_qs_tenant          ON pack039_energy_monitoring.em_quality_scores(tenant_id);
CREATE INDEX idx_p039_qs_period_type     ON pack039_energy_monitoring.em_quality_scores(score_period_type);
CREATE INDEX idx_p039_qs_period_start    ON pack039_energy_monitoring.em_quality_scores(period_start DESC);
CREATE INDEX idx_p039_qs_overall         ON pack039_energy_monitoring.em_quality_scores(overall_score);
CREATE INDEX idx_p039_qs_grade           ON pack039_energy_monitoring.em_quality_scores(quality_grade);
CREATE INDEX idx_p039_qs_meets_sla       ON pack039_energy_monitoring.em_quality_scores(meets_sla) WHERE meets_sla = false;
CREATE INDEX idx_p039_qs_created         ON pack039_energy_monitoring.em_quality_scores(created_at DESC);
CREATE INDEX idx_p039_qs_details         ON pack039_energy_monitoring.em_quality_scores USING GIN(score_details);

-- Composite: meter + daily scores for trending
CREATE INDEX idx_p039_qs_meter_daily     ON pack039_energy_monitoring.em_quality_scores(meter_id, period_start DESC)
    WHERE score_period_type = 'DAILY';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_qs_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_quality_scores
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_data_corrections
-- =============================================================================
-- Audit trail of all corrections applied to interval data, whether
-- automatic (by validation rules) or manual (by operators). Each
-- correction records the before/after values, the method used, the
-- reason, and the approver. Provides complete traceability for
-- regulatory compliance and dispute resolution.

CREATE TABLE pack039_energy_monitoring.em_data_corrections (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    interval_data_id        UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_interval_data(id) ON DELETE CASCADE,
    validation_result_id    UUID            REFERENCES pack039_energy_monitoring.em_validation_results(id),
    meter_id                UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    data_timestamp          TIMESTAMPTZ     NOT NULL,
    correction_type         VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATIC',
    correction_method       VARCHAR(30)     NOT NULL,
    correction_reason       VARCHAR(50)     NOT NULL,
    original_value          NUMERIC(18,6)   NOT NULL,
    corrected_value         NUMERIC(18,6)   NOT NULL,
    original_quality        VARCHAR(20),
    corrected_quality       VARCHAR(20)     NOT NULL DEFAULT 'CORRECTED',
    correction_factor       NUMERIC(10,6),
    confidence_pct          NUMERIC(5,2),
    reference_value         NUMERIC(18,6),
    reference_source        VARCHAR(100),
    is_approved             BOOLEAN         NOT NULL DEFAULT false,
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    approval_notes          TEXT,
    is_reverted             BOOLEAN         NOT NULL DEFAULT false,
    reverted_at             TIMESTAMPTZ,
    reverted_by             UUID,
    revert_reason           TEXT,
    applied_by              UUID,
    applied_by_system       VARCHAR(100),
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_dc_type CHECK (
        correction_type IN (
            'AUTOMATIC', 'MANUAL', 'BULK', 'RECALCULATION', 'REVERT'
        )
    ),
    CONSTRAINT chk_p039_dc_method CHECK (
        correction_method IN (
            'LINEAR_INTERPOLATION', 'PREVIOUS_VALUE', 'AVERAGE',
            'MEDIAN', 'REGRESSION', 'ZERO', 'MANUAL_ENTRY',
            'PROFILE_BASED', 'PROPORTIONAL', 'FORMULA', 'VENDOR_FILE'
        )
    ),
    CONSTRAINT chk_p039_dc_reason CHECK (
        correction_reason IN (
            'MISSING_DATA', 'OUTLIER', 'STUCK_VALUE', 'NEGATIVE_VALUE',
            'SPIKE', 'CT_RATIO_ERROR', 'METER_FAILURE', 'COMM_ERROR',
            'ROLLOVER', 'CALIBRATION_DRIFT', 'BILLING_DISPUTE',
            'MANUAL_OVERRIDE', 'BACKFILL', 'RECALCULATION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_dc_confidence CHECK (
        confidence_pct IS NULL OR (confidence_pct >= 0 AND confidence_pct <= 100)
    ),
    CONSTRAINT chk_p039_dc_values CHECK (
        original_value IS DISTINCT FROM corrected_value OR correction_type = 'REVERT'
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_dc_interval        ON pack039_energy_monitoring.em_data_corrections(interval_data_id);
CREATE INDEX idx_p039_dc_val_result      ON pack039_energy_monitoring.em_data_corrections(validation_result_id);
CREATE INDEX idx_p039_dc_meter           ON pack039_energy_monitoring.em_data_corrections(meter_id);
CREATE INDEX idx_p039_dc_tenant          ON pack039_energy_monitoring.em_data_corrections(tenant_id);
CREATE INDEX idx_p039_dc_data_ts         ON pack039_energy_monitoring.em_data_corrections(data_timestamp DESC);
CREATE INDEX idx_p039_dc_type            ON pack039_energy_monitoring.em_data_corrections(correction_type);
CREATE INDEX idx_p039_dc_method          ON pack039_energy_monitoring.em_data_corrections(correction_method);
CREATE INDEX idx_p039_dc_reason          ON pack039_energy_monitoring.em_data_corrections(correction_reason);
CREATE INDEX idx_p039_dc_approved        ON pack039_energy_monitoring.em_data_corrections(is_approved) WHERE is_approved = false;
CREATE INDEX idx_p039_dc_reverted        ON pack039_energy_monitoring.em_data_corrections(is_reverted) WHERE is_reverted = true;
CREATE INDEX idx_p039_dc_created         ON pack039_energy_monitoring.em_data_corrections(created_at DESC);

-- Composite: pending approvals for correction workflow
CREATE INDEX idx_p039_dc_pending_appr    ON pack039_energy_monitoring.em_data_corrections(tenant_id, correction_type, created_at DESC)
    WHERE is_approved = false AND is_reverted = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_dc_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_data_corrections
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_completeness_logs
-- =============================================================================
-- Tracks data completeness metrics for each meter over time periods.
-- Records the expected vs actual interval count, identifies gap periods,
-- and calculates completeness percentages. Used for SLA monitoring, data
-- quality reporting, and triggering gap-fill workflows when completeness
-- drops below configurable thresholds.

CREATE TABLE pack039_energy_monitoring.em_completeness_logs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    check_period_type       VARCHAR(20)     NOT NULL DEFAULT 'DAILY',
    period_start            TIMESTAMPTZ     NOT NULL,
    period_end              TIMESTAMPTZ     NOT NULL,
    expected_intervals      INTEGER         NOT NULL,
    actual_intervals        INTEGER         NOT NULL DEFAULT 0,
    missing_intervals       INTEGER         NOT NULL DEFAULT 0,
    estimated_intervals     INTEGER         NOT NULL DEFAULT 0,
    completeness_pct        NUMERIC(7,4)    NOT NULL DEFAULT 0.0,
    gap_count               INTEGER         NOT NULL DEFAULT 0,
    longest_gap_minutes     INTEGER         DEFAULT 0,
    longest_gap_start       TIMESTAMPTZ,
    longest_gap_end         TIMESTAMPTZ,
    first_reading_at        TIMESTAMPTZ,
    last_reading_at         TIMESTAMPTZ,
    gap_details             JSONB           DEFAULT '[]',
    meets_threshold         BOOLEAN,
    threshold_pct           NUMERIC(5,2)    DEFAULT 95.0,
    gap_fill_triggered      BOOLEAN         NOT NULL DEFAULT false,
    gap_fill_completed      BOOLEAN         NOT NULL DEFAULT false,
    gap_fill_completed_at   TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_cl_period_type CHECK (
        check_period_type IN ('HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY')
    ),
    CONSTRAINT chk_p039_cl_intervals CHECK (
        expected_intervals > 0 AND actual_intervals >= 0 AND
        missing_intervals >= 0 AND estimated_intervals >= 0
    ),
    CONSTRAINT chk_p039_cl_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p039_cl_gap_count CHECK (
        gap_count >= 0
    ),
    CONSTRAINT chk_p039_cl_longest_gap CHECK (
        longest_gap_minutes IS NULL OR longest_gap_minutes >= 0
    ),
    CONSTRAINT chk_p039_cl_threshold CHECK (
        threshold_pct IS NULL OR (threshold_pct >= 0 AND threshold_pct <= 100)
    ),
    CONSTRAINT chk_p039_cl_dates CHECK (
        period_start < period_end
    ),
    CONSTRAINT uq_p039_cl_meter_period UNIQUE (meter_id, check_period_type, period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_cl_meter           ON pack039_energy_monitoring.em_completeness_logs(meter_id);
CREATE INDEX idx_p039_cl_tenant          ON pack039_energy_monitoring.em_completeness_logs(tenant_id);
CREATE INDEX idx_p039_cl_period_type     ON pack039_energy_monitoring.em_completeness_logs(check_period_type);
CREATE INDEX idx_p039_cl_period_start    ON pack039_energy_monitoring.em_completeness_logs(period_start DESC);
CREATE INDEX idx_p039_cl_completeness    ON pack039_energy_monitoring.em_completeness_logs(completeness_pct);
CREATE INDEX idx_p039_cl_meets           ON pack039_energy_monitoring.em_completeness_logs(meets_threshold) WHERE meets_threshold = false;
CREATE INDEX idx_p039_cl_gap_fill        ON pack039_energy_monitoring.em_completeness_logs(gap_fill_triggered) WHERE gap_fill_triggered = true AND gap_fill_completed = false;
CREATE INDEX idx_p039_cl_created         ON pack039_energy_monitoring.em_completeness_logs(created_at DESC);
CREATE INDEX idx_p039_cl_gap_details     ON pack039_energy_monitoring.em_completeness_logs USING GIN(gap_details);

-- Composite: incomplete daily logs by meter for monitoring
CREATE INDEX idx_p039_cl_incomplete      ON pack039_energy_monitoring.em_completeness_logs(meter_id, period_start DESC)
    WHERE completeness_pct < 95 AND check_period_type = 'DAILY';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_cl_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_completeness_logs
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_validation_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_validation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_quality_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_data_corrections ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_completeness_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_vr_tenant_isolation
    ON pack039_energy_monitoring.em_validation_rules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_vr_service_bypass
    ON pack039_energy_monitoring.em_validation_rules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_vres_tenant_isolation
    ON pack039_energy_monitoring.em_validation_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_vres_service_bypass
    ON pack039_energy_monitoring.em_validation_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_qs_tenant_isolation
    ON pack039_energy_monitoring.em_quality_scores
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_qs_service_bypass
    ON pack039_energy_monitoring.em_quality_scores
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_dc_tenant_isolation
    ON pack039_energy_monitoring.em_data_corrections
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_dc_service_bypass
    ON pack039_energy_monitoring.em_data_corrections
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_cl_tenant_isolation
    ON pack039_energy_monitoring.em_completeness_logs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_cl_service_bypass
    ON pack039_energy_monitoring.em_completeness_logs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_validation_rules TO PUBLIC;
GRANT SELECT, INSERT ON pack039_energy_monitoring.em_validation_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_quality_scores TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_data_corrections TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_completeness_logs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_validation_rules IS
    'Configurable VEE validation rules with bounds checking, rate-of-change limits, statistical outlier detection, and custom formula-based validations.';
COMMENT ON TABLE pack039_energy_monitoring.em_validation_results IS
    'Per-interval validation results with pass/fail/warning outcomes, deviation metrics, and auto-correction records.';
COMMENT ON TABLE pack039_energy_monitoring.em_quality_scores IS
    'Aggregated data quality scores per meter and period across five dimensions: completeness, accuracy, consistency, timeliness, plausibility.';
COMMENT ON TABLE pack039_energy_monitoring.em_data_corrections IS
    'Complete audit trail of corrections applied to interval data with before/after values, methods, reasons, and approvals.';
COMMENT ON TABLE pack039_energy_monitoring.em_completeness_logs IS
    'Data completeness tracking per meter with gap detection, gap-fill workflow triggering, and SLA compliance monitoring.';

COMMENT ON COLUMN pack039_energy_monitoring.em_validation_rules.rule_category IS 'Rule family: RANGE_CHECK, RATE_OF_CHANGE, STATISTICAL, BALANCE_CHECK, COMPLETENESS, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_validation_rules.statistical_sigma IS 'Number of standard deviations for statistical outlier detection (e.g., 3.0 for 3-sigma rule).';
COMMENT ON COLUMN pack039_energy_monitoring.em_validation_rules.auto_correct IS 'Whether the rule automatically corrects failing values or only flags them.';
COMMENT ON COLUMN pack039_energy_monitoring.em_validation_rules.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_validation_results.deviation_pct IS 'Percentage deviation of the actual value from the expected range or threshold.';
COMMENT ON COLUMN pack039_energy_monitoring.em_validation_results.confidence_score IS 'Confidence in the validation result (0-100). Lower scores indicate borderline cases.';

COMMENT ON COLUMN pack039_energy_monitoring.em_quality_scores.overall_score IS 'Weighted composite of completeness, accuracy, consistency, timeliness, and plausibility (0-100).';
COMMENT ON COLUMN pack039_energy_monitoring.em_quality_scores.quality_grade IS 'Letter grade derived from overall_score: A+ (>97), A (>93), B+ (>90), B (>85), C (>75), D (>60), F (<60).';

COMMENT ON COLUMN pack039_energy_monitoring.em_data_corrections.correction_reason IS 'Root cause: MISSING_DATA, OUTLIER, STUCK_VALUE, CT_RATIO_ERROR, METER_FAILURE, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_data_corrections.is_reverted IS 'Whether this correction was subsequently reverted (undo).';

COMMENT ON COLUMN pack039_energy_monitoring.em_completeness_logs.completeness_pct IS 'Percentage of expected intervals with actual data (0-100). SLA target typically 95%+.';
COMMENT ON COLUMN pack039_energy_monitoring.em_completeness_logs.gap_details IS 'JSON array of gap periods: [{start, end, duration_minutes, reason}].';
