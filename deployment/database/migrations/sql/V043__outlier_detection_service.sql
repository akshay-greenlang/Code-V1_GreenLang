-- =============================================================================
-- V043: Outlier Detection Service Schema
-- =============================================================================
-- Component: AGENT-DATA-013 (Outlier Detection Agent)
-- Agent ID:  GL-DATA-X-016
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Outlier Detection Agent (GL-DATA-X-016) with capabilities for
-- statistical outlier detection (z-score, modified z-score, IQR,
-- Grubbs' test, Dixon's Q test, Chauvenet's criterion), distribution-
-- based detection (Mahalanobis distance, minimum covariance determinant,
-- chi-squared), density-based detection (LOF, DBSCAN, OPTICS, KDE),
-- model-based detection (Isolation Forest, One-Class SVM, Autoencoder,
-- Elliptic Envelope, HBOS), time-series anomaly detection (STL
-- decomposition, seasonal hybrid ESD, exponential smoothing, CUSUM,
-- Bollinger bands), ensemble methods (voting, stacking, weighted
-- consensus), outlier classification (point/contextual/collective,
-- severity scoring, domain-aware labeling), treatment recommendations
-- (removal, capping/winsorization, transformation, imputation,
-- flagging), impact analysis (effect on mean/variance/correlation/
-- regression, sensitivity testing), and feedback loops (human-in-the-
-- loop confirmation, false positive tracking, threshold tuning).
-- =============================================================================
-- Tables (10):
--   1. outlier_jobs              - Job tracking
--   2. outlier_detections        - Detection results per record/column
--   3. outlier_scores            - Outlier scores from multiple methods
--   4. outlier_classifications   - Outlier type & severity classification
--   5. outlier_treatments        - Treatment actions applied to outliers
--   6. outlier_thresholds        - Per-column/method threshold configs
--   7. outlier_feedback          - Human-in-the-loop feedback entries
--   8. outlier_impact_analyses   - Impact analysis of outlier removal
--   9. outlier_reports           - Outlier summary reports
--  10. outlier_audit_log         - Audit trail
--
-- Hypertables (3):
--  11. outlier_events            - Outlier event time-series (hypertable)
--  12. detection_events          - Detection event time-series (hypertable)
--  13. treatment_events          - Treatment event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. outlier_hourly_stats      - Hourly outlier event stats
--   2. detection_hourly_stats    - Hourly detection event stats
--
-- Also includes: 150+ indexes (B-tree, GIN, partial, composite),
-- 75+ CHECK constraints, 26 RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-016.
-- Previous: V042__missing_value_imputer_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS outlier_detection_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION outlier_detection_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: outlier_detection_service.outlier_jobs
-- =============================================================================
-- Job tracking for outlier detection runs. Each job captures dataset IDs,
-- optional threshold profile reference, processing status and stage,
-- record counts at each pipeline stage (scanned, detected, classified,
-- treated), outlier rate, error messages, configuration, provenance
-- hash, and timing information. Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_ids TEXT[] NOT NULL,
    threshold_profile_id UUID,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    stage VARCHAR(20) DEFAULT 'scan',
    total_records INTEGER NOT NULL DEFAULT 0,
    total_columns INTEGER NOT NULL DEFAULT 0,
    outliers_detected INTEGER NOT NULL DEFAULT 0,
    scanned INTEGER NOT NULL DEFAULT 0,
    detected INTEGER NOT NULL DEFAULT 0,
    classified INTEGER NOT NULL DEFAULT 0,
    treated INTEGER NOT NULL DEFAULT 0,
    outlier_rate NUMERIC(5,4) DEFAULT 0,
    error_message TEXT,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    dry_run BOOLEAN NOT NULL DEFAULT false,
    provenance_hash VARCHAR(64) NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Status constraint
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

-- Stage constraint
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_stage
    CHECK (stage IN ('scan', 'detect', 'classify', 'treat', 'analyze', 'report', 'complete'));

-- Total records must be non-negative
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_total_records_non_negative
    CHECK (total_records >= 0);

-- Total columns must be non-negative
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_total_columns_non_negative
    CHECK (total_columns >= 0);

-- Outliers detected must be non-negative
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_outliers_detected_non_negative
    CHECK (outliers_detected >= 0);

-- Scanned must be non-negative
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_scanned_non_negative
    CHECK (scanned >= 0);

-- Detected must be non-negative
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_detected_non_negative
    CHECK (detected >= 0);

-- Classified must be non-negative
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_classified_non_negative
    CHECK (classified >= 0);

-- Treated must be non-negative
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_treated_non_negative
    CHECK (treated >= 0);

-- Outlier rate must be between 0 and 1
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_outlier_rate_range
    CHECK (outlier_rate IS NULL OR (outlier_rate >= 0 AND outlier_rate <= 1));

-- Provenance hash must not be empty
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Created by must not be empty
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- Dataset IDs array must not be empty
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_dataset_ids_not_empty
    CHECK (array_length(dataset_ids, 1) > 0);

-- Completed_at must be after started_at if both are set
ALTER TABLE outlier_detection_service.outlier_jobs
    ADD CONSTRAINT chk_oj_completed_after_started
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at);

-- updated_at trigger
CREATE TRIGGER trg_oj_updated_at
    BEFORE UPDATE ON outlier_detection_service.outlier_jobs
    FOR EACH ROW
    EXECUTE FUNCTION outlier_detection_service.set_updated_at();

-- =============================================================================
-- Table 2: outlier_detection_service.outlier_detections
-- =============================================================================
-- Detection results per record/column. Each detection captures the
-- column name, record index, original value, detection method used,
-- outlier score, whether it is an outlier, confidence level, and
-- contextual information (e.g., neighboring values). Linked to
-- outlier_jobs. Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    record_index INTEGER NOT NULL,
    original_value TEXT,
    detection_method VARCHAR(30) NOT NULL,
    outlier_score NUMERIC(12,6) NOT NULL,
    is_outlier BOOLEAN NOT NULL DEFAULT false,
    confidence NUMERIC(5,4),
    direction VARCHAR(10),
    context JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_jobs
ALTER TABLE outlier_detection_service.outlier_detections
    ADD CONSTRAINT fk_od_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE CASCADE;

-- Detection method constraint
ALTER TABLE outlier_detection_service.outlier_detections
    ADD CONSTRAINT chk_od_detection_method
    CHECK (detection_method IN (
        'z_score', 'modified_z_score', 'iqr', 'grubbs', 'dixon_q', 'chauvenet',
        'mahalanobis', 'mcd', 'chi_squared',
        'lof', 'dbscan', 'optics', 'kde',
        'isolation_forest', 'one_class_svm', 'autoencoder', 'elliptic_envelope', 'hbos',
        'stl_decomposition', 'seasonal_esd', 'exponential_smoothing', 'cusum', 'bollinger',
        'ensemble_vote', 'ensemble_stack', 'ensemble_weighted'
    ));

-- Direction constraint (upper/lower/both for tail)
ALTER TABLE outlier_detection_service.outlier_detections
    ADD CONSTRAINT chk_od_direction
    CHECK (direction IS NULL OR direction IN ('upper', 'lower', 'both'));

-- Confidence must be between 0 and 1 if specified
ALTER TABLE outlier_detection_service.outlier_detections
    ADD CONSTRAINT chk_od_confidence_range
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));

-- Record index must be non-negative
ALTER TABLE outlier_detection_service.outlier_detections
    ADD CONSTRAINT chk_od_record_index_non_negative
    CHECK (record_index >= 0);

-- Column name must not be empty
ALTER TABLE outlier_detection_service.outlier_detections
    ADD CONSTRAINT chk_od_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_detections
    ADD CONSTRAINT chk_od_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 3: outlier_detection_service.outlier_scores
-- =============================================================================
-- Outlier scores from multiple detection methods per record/column.
-- Each score captures the method used, raw score, normalized score
-- (0-1), threshold applied, and whether it exceeds the threshold.
-- Enables ensemble scoring and method comparison. Linked to
-- outlier_detections. Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID NOT NULL,
    job_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    method VARCHAR(30) NOT NULL,
    raw_score NUMERIC(12,6) NOT NULL,
    normalized_score NUMERIC(5,4) NOT NULL,
    threshold_value NUMERIC(12,6),
    exceeds_threshold BOOLEAN NOT NULL DEFAULT false,
    percentile NUMERIC(5,4),
    parameters JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_detections
ALTER TABLE outlier_detection_service.outlier_scores
    ADD CONSTRAINT fk_os_detection_id
    FOREIGN KEY (detection_id) REFERENCES outlier_detection_service.outlier_detections(id)
    ON DELETE CASCADE;

-- Foreign key to outlier_jobs
ALTER TABLE outlier_detection_service.outlier_scores
    ADD CONSTRAINT fk_os_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE CASCADE;

-- Method constraint
ALTER TABLE outlier_detection_service.outlier_scores
    ADD CONSTRAINT chk_os_method
    CHECK (method IN (
        'z_score', 'modified_z_score', 'iqr', 'grubbs', 'dixon_q', 'chauvenet',
        'mahalanobis', 'mcd', 'chi_squared',
        'lof', 'dbscan', 'optics', 'kde',
        'isolation_forest', 'one_class_svm', 'autoencoder', 'elliptic_envelope', 'hbos',
        'stl_decomposition', 'seasonal_esd', 'exponential_smoothing', 'cusum', 'bollinger',
        'ensemble_vote', 'ensemble_stack', 'ensemble_weighted'
    ));

-- Normalized score must be between 0 and 1
ALTER TABLE outlier_detection_service.outlier_scores
    ADD CONSTRAINT chk_os_normalized_score_range
    CHECK (normalized_score >= 0 AND normalized_score <= 1);

-- Percentile must be between 0 and 1 if specified
ALTER TABLE outlier_detection_service.outlier_scores
    ADD CONSTRAINT chk_os_percentile_range
    CHECK (percentile IS NULL OR (percentile >= 0 AND percentile <= 1));

-- Column name must not be empty
ALTER TABLE outlier_detection_service.outlier_scores
    ADD CONSTRAINT chk_os_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_scores
    ADD CONSTRAINT chk_os_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 4: outlier_detection_service.outlier_classifications
-- =============================================================================
-- Outlier type and severity classification. Each classification
-- captures the outlier type (point, contextual, collective),
-- severity level (low, medium, high, critical), domain category,
-- explanation text, and ensemble consensus information. Linked to
-- outlier_detections. Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_classifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID NOT NULL,
    job_id UUID NOT NULL,
    outlier_type VARCHAR(20) NOT NULL,
    severity VARCHAR(10) NOT NULL DEFAULT 'medium',
    domain_category VARCHAR(50),
    explanation TEXT,
    consensus_score NUMERIC(5,4),
    methods_agreed INTEGER NOT NULL DEFAULT 1,
    methods_total INTEGER NOT NULL DEFAULT 1,
    classification_config JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_detections
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT fk_oc_detection_id
    FOREIGN KEY (detection_id) REFERENCES outlier_detection_service.outlier_detections(id)
    ON DELETE CASCADE;

-- Foreign key to outlier_jobs
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT fk_oc_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE CASCADE;

-- Outlier type constraint
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT chk_oc_outlier_type
    CHECK (outlier_type IN ('point', 'contextual', 'collective', 'seasonal', 'trend', 'unknown'));

-- Severity constraint
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT chk_oc_severity
    CHECK (severity IN ('low', 'medium', 'high', 'critical'));

-- Consensus score must be between 0 and 1 if specified
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT chk_oc_consensus_score_range
    CHECK (consensus_score IS NULL OR (consensus_score >= 0 AND consensus_score <= 1));

-- Methods agreed must be positive
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT chk_oc_methods_agreed_positive
    CHECK (methods_agreed >= 1);

-- Methods total must be positive
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT chk_oc_methods_total_positive
    CHECK (methods_total >= 1);

-- Methods agreed must not exceed methods total
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT chk_oc_methods_agreed_lte_total
    CHECK (methods_agreed <= methods_total);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_classifications
    ADD CONSTRAINT chk_oc_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 5: outlier_detection_service.outlier_treatments
-- =============================================================================
-- Treatment actions applied to detected outliers. Each treatment
-- captures the treatment method (removal, capping, winsorization,
-- transformation, imputation, flagging), original and treated values,
-- rationale, and approval status. Linked to outlier_detections.
-- Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_treatments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID NOT NULL,
    job_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    treatment_method VARCHAR(20) NOT NULL,
    original_value TEXT,
    treated_value TEXT,
    lower_bound NUMERIC(12,6),
    upper_bound NUMERIC(12,6),
    rationale TEXT,
    approval_status VARCHAR(15) NOT NULL DEFAULT 'auto_approved',
    approved_by VARCHAR(100),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_detections
ALTER TABLE outlier_detection_service.outlier_treatments
    ADD CONSTRAINT fk_ot_detection_id
    FOREIGN KEY (detection_id) REFERENCES outlier_detection_service.outlier_detections(id)
    ON DELETE CASCADE;

-- Foreign key to outlier_jobs
ALTER TABLE outlier_detection_service.outlier_treatments
    ADD CONSTRAINT fk_ot_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE CASCADE;

-- Treatment method constraint
ALTER TABLE outlier_detection_service.outlier_treatments
    ADD CONSTRAINT chk_ot_treatment_method
    CHECK (treatment_method IN (
        'removal', 'capping', 'winsorization', 'log_transform',
        'sqrt_transform', 'box_cox', 'imputation', 'flagging', 'no_action'
    ));

-- Approval status constraint
ALTER TABLE outlier_detection_service.outlier_treatments
    ADD CONSTRAINT chk_ot_approval_status
    CHECK (approval_status IN ('auto_approved', 'pending_review', 'approved', 'rejected', 'overridden'));

-- Column name must not be empty
ALTER TABLE outlier_detection_service.outlier_treatments
    ADD CONSTRAINT chk_ot_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Provenance hash must not be empty
ALTER TABLE outlier_detection_service.outlier_treatments
    ADD CONSTRAINT chk_ot_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_treatments
    ADD CONSTRAINT chk_ot_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_ot_updated_at
    BEFORE UPDATE ON outlier_detection_service.outlier_treatments
    FOR EACH ROW
    EXECUTE FUNCTION outlier_detection_service.set_updated_at();

-- =============================================================================
-- Table 6: outlier_detection_service.outlier_thresholds
-- =============================================================================
-- Per-column/method threshold configurations. Each threshold entry
-- captures the column name, detection method, threshold value,
-- sensitivity level, auto-tuning flag, and statistical basis.
-- Enables per-column threshold overrides and adaptive thresholds.
-- Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_thresholds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    column_name VARCHAR(255),
    detection_method VARCHAR(30) NOT NULL,
    threshold_value NUMERIC(12,6) NOT NULL,
    sensitivity VARCHAR(10) NOT NULL DEFAULT 'medium',
    auto_tune BOOLEAN NOT NULL DEFAULT false,
    statistical_basis JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Detection method constraint
ALTER TABLE outlier_detection_service.outlier_thresholds
    ADD CONSTRAINT chk_oth_detection_method
    CHECK (detection_method IN (
        'z_score', 'modified_z_score', 'iqr', 'grubbs', 'dixon_q', 'chauvenet',
        'mahalanobis', 'mcd', 'chi_squared',
        'lof', 'dbscan', 'optics', 'kde',
        'isolation_forest', 'one_class_svm', 'autoencoder', 'elliptic_envelope', 'hbos',
        'stl_decomposition', 'seasonal_esd', 'exponential_smoothing', 'cusum', 'bollinger',
        'ensemble_vote', 'ensemble_stack', 'ensemble_weighted'
    ));

-- Sensitivity constraint
ALTER TABLE outlier_detection_service.outlier_thresholds
    ADD CONSTRAINT chk_oth_sensitivity
    CHECK (sensitivity IN ('low', 'medium', 'high', 'very_high', 'custom'));

-- Version must be positive
ALTER TABLE outlier_detection_service.outlier_thresholds
    ADD CONSTRAINT chk_oth_version_positive
    CHECK (version >= 1);

-- Name must not be empty
ALTER TABLE outlier_detection_service.outlier_thresholds
    ADD CONSTRAINT chk_oth_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Created by must not be empty
ALTER TABLE outlier_detection_service.outlier_thresholds
    ADD CONSTRAINT chk_oth_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_thresholds
    ADD CONSTRAINT chk_oth_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_oth_updated_at
    BEFORE UPDATE ON outlier_detection_service.outlier_thresholds
    FOR EACH ROW
    EXECUTE FUNCTION outlier_detection_service.set_updated_at();

-- =============================================================================
-- Table 7: outlier_detection_service.outlier_feedback
-- =============================================================================
-- Human-in-the-loop feedback entries. Each feedback captures the
-- detection ID, feedback type (confirm, reject, reclassify, adjust),
-- original and corrected labels, confidence adjustment, and notes.
-- Enables threshold tuning and false positive tracking. Linked to
-- outlier_detections. Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID NOT NULL,
    job_id UUID NOT NULL,
    feedback_type VARCHAR(15) NOT NULL,
    original_label BOOLEAN NOT NULL,
    corrected_label BOOLEAN NOT NULL,
    severity_adjustment VARCHAR(10),
    confidence_adjustment NUMERIC(5,4),
    notes TEXT,
    feedback_by VARCHAR(100) NOT NULL,
    feedback_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_detections
ALTER TABLE outlier_detection_service.outlier_feedback
    ADD CONSTRAINT fk_of_detection_id
    FOREIGN KEY (detection_id) REFERENCES outlier_detection_service.outlier_detections(id)
    ON DELETE CASCADE;

-- Foreign key to outlier_jobs
ALTER TABLE outlier_detection_service.outlier_feedback
    ADD CONSTRAINT fk_of_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE CASCADE;

-- Feedback type constraint
ALTER TABLE outlier_detection_service.outlier_feedback
    ADD CONSTRAINT chk_of_feedback_type
    CHECK (feedback_type IN ('confirm', 'reject', 'reclassify', 'adjust_threshold', 'annotate'));

-- Severity adjustment constraint
ALTER TABLE outlier_detection_service.outlier_feedback
    ADD CONSTRAINT chk_of_severity_adjustment
    CHECK (severity_adjustment IS NULL OR severity_adjustment IN ('low', 'medium', 'high', 'critical'));

-- Confidence adjustment must be between -1 and 1 if specified
ALTER TABLE outlier_detection_service.outlier_feedback
    ADD CONSTRAINT chk_of_confidence_adjustment_range
    CHECK (confidence_adjustment IS NULL OR (confidence_adjustment >= -1 AND confidence_adjustment <= 1));

-- Feedback by must not be empty
ALTER TABLE outlier_detection_service.outlier_feedback
    ADD CONSTRAINT chk_of_feedback_by_not_empty
    CHECK (LENGTH(TRIM(feedback_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_feedback
    ADD CONSTRAINT chk_of_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 8: outlier_detection_service.outlier_impact_analyses
-- =============================================================================
-- Impact analysis of outlier removal/treatment on dataset statistics.
-- Each analysis captures the before/after mean, variance, standard
-- deviation, correlation matrix changes, regression coefficient
-- changes, and sensitivity metrics. Linked to outlier_jobs. Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_impact_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    analysis_type VARCHAR(30) NOT NULL,
    outliers_removed INTEGER NOT NULL DEFAULT 0,
    mean_before NUMERIC(12,6),
    mean_after NUMERIC(12,6),
    mean_change_pct NUMERIC(8,4),
    variance_before NUMERIC(12,6),
    variance_after NUMERIC(12,6),
    variance_change_pct NUMERIC(8,4),
    std_dev_before NUMERIC(12,6),
    std_dev_after NUMERIC(12,6),
    correlation_changes JSONB DEFAULT '{}'::jsonb,
    regression_impact JSONB DEFAULT '{}'::jsonb,
    sensitivity_metrics JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_jobs
ALTER TABLE outlier_detection_service.outlier_impact_analyses
    ADD CONSTRAINT fk_oia_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE CASCADE;

-- Analysis type constraint
ALTER TABLE outlier_detection_service.outlier_impact_analyses
    ADD CONSTRAINT chk_oia_analysis_type
    CHECK (analysis_type IN (
        'mean_impact', 'variance_impact', 'correlation_impact',
        'regression_impact', 'distribution_impact', 'sensitivity',
        'comprehensive', 'before_after_comparison'
    ));

-- Outliers removed must be non-negative
ALTER TABLE outlier_detection_service.outlier_impact_analyses
    ADD CONSTRAINT chk_oia_outliers_removed_non_negative
    CHECK (outliers_removed >= 0);

-- Column name must not be empty
ALTER TABLE outlier_detection_service.outlier_impact_analyses
    ADD CONSTRAINT chk_oia_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Provenance hash must not be empty
ALTER TABLE outlier_detection_service.outlier_impact_analyses
    ADD CONSTRAINT chk_oia_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_impact_analyses
    ADD CONSTRAINT chk_oia_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 9: outlier_detection_service.outlier_reports
-- =============================================================================
-- Outlier summary reports. Each report captures the overall outlier
-- rate before/after treatment, per-column summaries, method
-- effectiveness metrics, treatment summary, quality grade, and
-- provenance hash. Linked to outlier_jobs. Tenant-scoped.

CREATE TABLE outlier_detection_service.outlier_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    report_type VARCHAR(20) NOT NULL DEFAULT 'summary',
    outlier_rate_before NUMERIC(5,4) NOT NULL,
    outlier_rate_after NUMERIC(5,4) NOT NULL,
    columns_analyzed INTEGER NOT NULL,
    total_outliers_detected INTEGER NOT NULL,
    total_outliers_treated INTEGER NOT NULL,
    false_positive_rate NUMERIC(5,4),
    overall_quality_score NUMERIC(5,4),
    quality_grade VARCHAR(5),
    column_summaries JSONB NOT NULL DEFAULT '[]'::jsonb,
    method_effectiveness JSONB NOT NULL DEFAULT '{}'::jsonb,
    treatment_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    recommendations JSONB DEFAULT '[]'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_jobs
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT fk_orp_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE CASCADE;

-- Report type constraint
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_report_type
    CHECK (report_type IN ('summary', 'detailed', 'method_comparison', 'treatment', 'audit'));

-- Quality grade constraint
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_quality_grade
    CHECK (quality_grade IS NULL OR quality_grade IN ('A', 'B', 'C', 'D', 'F'));

-- Outlier rate before must be between 0 and 1
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_outlier_rate_before_range
    CHECK (outlier_rate_before >= 0 AND outlier_rate_before <= 1);

-- Outlier rate after must be between 0 and 1
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_outlier_rate_after_range
    CHECK (outlier_rate_after >= 0 AND outlier_rate_after <= 1);

-- Outlier rate after must be <= outlier rate before
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_outlier_rate_reduced
    CHECK (outlier_rate_after <= outlier_rate_before);

-- Columns analyzed must be non-negative
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_columns_analyzed_non_negative
    CHECK (columns_analyzed >= 0);

-- Total outliers detected must be non-negative
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_total_detected_non_negative
    CHECK (total_outliers_detected >= 0);

-- Total outliers treated must be non-negative
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_total_treated_non_negative
    CHECK (total_outliers_treated >= 0);

-- False positive rate must be between 0 and 1 if specified
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_false_positive_rate_range
    CHECK (false_positive_rate IS NULL OR (false_positive_rate >= 0 AND false_positive_rate <= 1));

-- Overall quality score must be between 0 and 1 if specified
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_quality_score_range
    CHECK (overall_quality_score IS NULL OR (overall_quality_score >= 0 AND overall_quality_score <= 1));

-- Provenance hash must not be empty
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_reports
    ADD CONSTRAINT chk_orp_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: outlier_detection_service.outlier_audit_log
-- =============================================================================
-- Comprehensive audit trail for all outlier detection operations. Each
-- entry captures the action performed, entity type and ID, detail
-- payload (JSONB), provenance hash, performer, timestamp, and tenant
-- scope. Linked to outlier_jobs for job-scoped audit queries.

CREATE TABLE outlier_detection_service.outlier_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64),
    performed_by VARCHAR(100) NOT NULL DEFAULT 'system',
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to outlier_jobs (optional, job_id may be null for system-level actions)
ALTER TABLE outlier_detection_service.outlier_audit_log
    ADD CONSTRAINT fk_oal_job_id
    FOREIGN KEY (job_id) REFERENCES outlier_detection_service.outlier_jobs(id)
    ON DELETE SET NULL;

-- Action constraint
ALTER TABLE outlier_detection_service.outlier_audit_log
    ADD CONSTRAINT chk_oal_action
    CHECK (action IN (
        'job_created', 'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'detection_completed', 'scoring_completed', 'classification_completed',
        'treatment_applied', 'treatment_approved', 'treatment_rejected',
        'impact_analysis_completed', 'report_generated',
        'threshold_created', 'threshold_updated', 'threshold_deleted',
        'threshold_activated', 'threshold_deactivated',
        'feedback_submitted', 'feedback_applied',
        'config_changed', 'dry_run_completed', 'rollback_performed',
        'export_generated', 'import_completed'
    ));

-- Entity type constraint
ALTER TABLE outlier_detection_service.outlier_audit_log
    ADD CONSTRAINT chk_oal_entity_type
    CHECK (entity_type IN (
        'job', 'detection', 'score', 'classification', 'treatment',
        'threshold', 'feedback', 'impact_analysis', 'report', 'config'
    ));

-- Action must not be empty
ALTER TABLE outlier_detection_service.outlier_audit_log
    ADD CONSTRAINT chk_oal_action_not_empty
    CHECK (LENGTH(TRIM(action)) > 0);

-- Entity type must not be empty
ALTER TABLE outlier_detection_service.outlier_audit_log
    ADD CONSTRAINT chk_oal_entity_type_not_empty
    CHECK (LENGTH(TRIM(entity_type)) > 0);

-- Performed by must not be empty
ALTER TABLE outlier_detection_service.outlier_audit_log
    ADD CONSTRAINT chk_oal_performed_by_not_empty
    CHECK (LENGTH(TRIM(performed_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_audit_log
    ADD CONSTRAINT chk_oal_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: outlier_detection_service.outlier_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording outlier detection lifecycle events
-- as a time-series. Each event captures the job ID, event type,
-- pipeline stage, record count, duration in milliseconds, details
-- payload, provenance hash, and tenant. Partitioned by event_time
-- for time-series queries. Retained for 90 days with compression
-- after 7 days.

CREATE TABLE outlier_detection_service.outlier_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    event_type VARCHAR(50) NOT NULL,
    stage VARCHAR(20),
    record_count INTEGER,
    duration_ms INTEGER,
    details JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('outlier_detection_service.outlier_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE outlier_detection_service.outlier_events
    ADD CONSTRAINT chk_oe_event_type
    CHECK (event_type IN (
        'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'scan_started', 'scan_completed', 'scan_failed',
        'detection_started', 'detection_completed', 'detection_failed',
        'classification_started', 'classification_completed', 'classification_failed',
        'treatment_started', 'treatment_completed', 'treatment_failed',
        'analysis_started', 'analysis_completed', 'analysis_failed',
        'reporting_started', 'reporting_completed', 'reporting_failed',
        'stage_transition', 'progress_update', 'threshold_breach'
    ));

-- Stage constraint if specified
ALTER TABLE outlier_detection_service.outlier_events
    ADD CONSTRAINT chk_oe_stage
    CHECK (stage IS NULL OR stage IN (
        'scan', 'detect', 'classify', 'treat', 'analyze', 'report', 'complete'
    ));

-- Record count must be non-negative if specified
ALTER TABLE outlier_detection_service.outlier_events
    ADD CONSTRAINT chk_oe_record_count_non_negative
    CHECK (record_count IS NULL OR record_count >= 0);

-- Duration must be non-negative if specified
ALTER TABLE outlier_detection_service.outlier_events
    ADD CONSTRAINT chk_oe_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.outlier_events
    ADD CONSTRAINT chk_oe_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 12: outlier_detection_service.detection_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording detection events as a time-series.
-- Each event captures the job ID, column name, detection method,
-- outlier count, total scanned, duration, and tenant. Partitioned
-- by event_time for time-series queries. Retained for 90 days with
-- compression after 7 days.

CREATE TABLE outlier_detection_service.detection_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    column_name VARCHAR(255),
    detection_method VARCHAR(30),
    outlier_count INTEGER,
    total_scanned INTEGER,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('outlier_detection_service.detection_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Detection method constraint if specified
ALTER TABLE outlier_detection_service.detection_events
    ADD CONSTRAINT chk_de_detection_method
    CHECK (detection_method IS NULL OR detection_method IN (
        'z_score', 'modified_z_score', 'iqr', 'grubbs', 'dixon_q', 'chauvenet',
        'mahalanobis', 'mcd', 'chi_squared',
        'lof', 'dbscan', 'optics', 'kde',
        'isolation_forest', 'one_class_svm', 'autoencoder', 'elliptic_envelope', 'hbos',
        'stl_decomposition', 'seasonal_esd', 'exponential_smoothing', 'cusum', 'bollinger',
        'ensemble_vote', 'ensemble_stack', 'ensemble_weighted'
    ));

-- Outlier count must be non-negative if specified
ALTER TABLE outlier_detection_service.detection_events
    ADD CONSTRAINT chk_de_outlier_count_non_negative
    CHECK (outlier_count IS NULL OR outlier_count >= 0);

-- Total scanned must be non-negative if specified
ALTER TABLE outlier_detection_service.detection_events
    ADD CONSTRAINT chk_de_total_scanned_non_negative
    CHECK (total_scanned IS NULL OR total_scanned >= 0);

-- Duration must be non-negative if specified
ALTER TABLE outlier_detection_service.detection_events
    ADD CONSTRAINT chk_de_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.detection_events
    ADD CONSTRAINT chk_de_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 13: outlier_detection_service.treatment_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording treatment events as a time-series.
-- Each event captures the job ID, treatment method, column name,
-- outliers treated, treatment duration, provenance hash, and tenant.
-- Partitioned by event_time for time-series queries. Retained for
-- 90 days with compression after 7 days.

CREATE TABLE outlier_detection_service.treatment_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    treatment_method VARCHAR(20),
    column_name VARCHAR(255),
    outliers_treated INTEGER,
    duration_ms INTEGER,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('outlier_detection_service.treatment_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Treatment method constraint if specified
ALTER TABLE outlier_detection_service.treatment_events
    ADD CONSTRAINT chk_te_treatment_method
    CHECK (treatment_method IS NULL OR treatment_method IN (
        'removal', 'capping', 'winsorization', 'log_transform',
        'sqrt_transform', 'box_cox', 'imputation', 'flagging', 'no_action'
    ));

-- Outliers treated must be non-negative if specified
ALTER TABLE outlier_detection_service.treatment_events
    ADD CONSTRAINT chk_te_outliers_treated_non_negative
    CHECK (outliers_treated IS NULL OR outliers_treated >= 0);

-- Duration must be non-negative if specified
ALTER TABLE outlier_detection_service.treatment_events
    ADD CONSTRAINT chk_te_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE outlier_detection_service.treatment_events
    ADD CONSTRAINT chk_te_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Continuous Aggregate: outlier_detection_service.outlier_hourly_stats
-- =============================================================================
-- Precomputed hourly outlier detection statistics by event type for
-- dashboard queries, job monitoring, and throughput analysis.

CREATE MATERIALIZED VIEW outlier_detection_service.outlier_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    event_type,
    COUNT(*) AS total_events,
    COUNT(DISTINCT job_id) AS unique_jobs,
    AVG(record_count) AS avg_record_count,
    SUM(record_count) AS total_records_processed,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    MIN(duration_ms) AS min_duration_ms,
    COUNT(*) FILTER (WHERE stage = 'scan') AS scan_events,
    COUNT(*) FILTER (WHERE stage = 'detect') AS detect_events,
    COUNT(*) FILTER (WHERE stage = 'classify') AS classify_events,
    COUNT(*) FILTER (WHERE stage = 'treat') AS treat_events,
    COUNT(*) FILTER (WHERE stage = 'analyze') AS analyze_events,
    COUNT(*) FILTER (WHERE stage = 'report') AS report_events,
    COUNT(*) FILTER (WHERE stage = 'complete') AS complete_events
FROM outlier_detection_service.outlier_events
WHERE event_time IS NOT NULL
GROUP BY bucket, event_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('outlier_detection_service.outlier_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: outlier_detection_service.detection_hourly_stats
-- =============================================================================
-- Precomputed hourly detection statistics by method for dashboard
-- queries, method comparison, and outlier rate monitoring.

CREATE MATERIALIZED VIEW outlier_detection_service.detection_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    detection_method,
    COUNT(*) AS total_detections,
    COUNT(DISTINCT job_id) AS unique_jobs,
    AVG(outlier_count) AS avg_outlier_count,
    SUM(outlier_count) AS total_outliers_found,
    AVG(total_scanned) AS avg_total_scanned,
    SUM(total_scanned) AS total_records_scanned,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    MIN(duration_ms) AS min_duration_ms
FROM outlier_detection_service.detection_events
WHERE event_time IS NOT NULL
GROUP BY bucket, detection_method
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('outlier_detection_service.detection_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- outlier_jobs indexes (19)
CREATE INDEX idx_oj_threshold_profile ON outlier_detection_service.outlier_jobs(threshold_profile_id);
CREATE INDEX idx_oj_status ON outlier_detection_service.outlier_jobs(status);
CREATE INDEX idx_oj_stage ON outlier_detection_service.outlier_jobs(stage);
CREATE INDEX idx_oj_tenant_id ON outlier_detection_service.outlier_jobs(tenant_id);
CREATE INDEX idx_oj_created_by ON outlier_detection_service.outlier_jobs(created_by);
CREATE INDEX idx_oj_provenance ON outlier_detection_service.outlier_jobs(provenance_hash);
CREATE INDEX idx_oj_created_at ON outlier_detection_service.outlier_jobs(created_at DESC);
CREATE INDEX idx_oj_updated_at ON outlier_detection_service.outlier_jobs(updated_at DESC);
CREATE INDEX idx_oj_started_at ON outlier_detection_service.outlier_jobs(started_at DESC);
CREATE INDEX idx_oj_completed_at ON outlier_detection_service.outlier_jobs(completed_at DESC);
CREATE INDEX idx_oj_outlier_rate ON outlier_detection_service.outlier_jobs(outlier_rate DESC);
CREATE INDEX idx_oj_total_records ON outlier_detection_service.outlier_jobs(total_records DESC);
CREATE INDEX idx_oj_dry_run ON outlier_detection_service.outlier_jobs(dry_run);
CREATE INDEX idx_oj_tenant_status ON outlier_detection_service.outlier_jobs(tenant_id, status);
CREATE INDEX idx_oj_tenant_stage ON outlier_detection_service.outlier_jobs(tenant_id, stage);
CREATE INDEX idx_oj_tenant_created ON outlier_detection_service.outlier_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_oj_status_stage ON outlier_detection_service.outlier_jobs(status, stage);
CREATE INDEX idx_oj_dataset_ids ON outlier_detection_service.outlier_jobs USING GIN (dataset_ids);
CREATE INDEX idx_oj_config ON outlier_detection_service.outlier_jobs USING GIN (config);

-- outlier_detections indexes (16)
CREATE INDEX idx_od_job_id ON outlier_detection_service.outlier_detections(job_id);
CREATE INDEX idx_od_column_name ON outlier_detection_service.outlier_detections(column_name);
CREATE INDEX idx_od_record_index ON outlier_detection_service.outlier_detections(record_index);
CREATE INDEX idx_od_detection_method ON outlier_detection_service.outlier_detections(detection_method);
CREATE INDEX idx_od_is_outlier ON outlier_detection_service.outlier_detections(is_outlier);
CREATE INDEX idx_od_outlier_score ON outlier_detection_service.outlier_detections(outlier_score DESC);
CREATE INDEX idx_od_confidence ON outlier_detection_service.outlier_detections(confidence DESC);
CREATE INDEX idx_od_direction ON outlier_detection_service.outlier_detections(direction);
CREATE INDEX idx_od_tenant_id ON outlier_detection_service.outlier_detections(tenant_id);
CREATE INDEX idx_od_created_at ON outlier_detection_service.outlier_detections(created_at DESC);
CREATE INDEX idx_od_job_column ON outlier_detection_service.outlier_detections(job_id, column_name);
CREATE INDEX idx_od_job_method ON outlier_detection_service.outlier_detections(job_id, detection_method);
CREATE INDEX idx_od_job_outlier ON outlier_detection_service.outlier_detections(job_id, is_outlier);
CREATE INDEX idx_od_tenant_job ON outlier_detection_service.outlier_detections(tenant_id, job_id);
CREATE INDEX idx_od_tenant_outlier ON outlier_detection_service.outlier_detections(tenant_id, is_outlier);
CREATE INDEX idx_od_context ON outlier_detection_service.outlier_detections USING GIN (context);

-- outlier_scores indexes (16)
CREATE INDEX idx_os_detection_id ON outlier_detection_service.outlier_scores(detection_id);
CREATE INDEX idx_os_job_id ON outlier_detection_service.outlier_scores(job_id);
CREATE INDEX idx_os_column_name ON outlier_detection_service.outlier_scores(column_name);
CREATE INDEX idx_os_method ON outlier_detection_service.outlier_scores(method);
CREATE INDEX idx_os_raw_score ON outlier_detection_service.outlier_scores(raw_score DESC);
CREATE INDEX idx_os_normalized_score ON outlier_detection_service.outlier_scores(normalized_score DESC);
CREATE INDEX idx_os_exceeds_threshold ON outlier_detection_service.outlier_scores(exceeds_threshold);
CREATE INDEX idx_os_percentile ON outlier_detection_service.outlier_scores(percentile DESC);
CREATE INDEX idx_os_tenant_id ON outlier_detection_service.outlier_scores(tenant_id);
CREATE INDEX idx_os_created_at ON outlier_detection_service.outlier_scores(created_at DESC);
CREATE INDEX idx_os_detection_method ON outlier_detection_service.outlier_scores(detection_id, method);
CREATE INDEX idx_os_job_method ON outlier_detection_service.outlier_scores(job_id, method);
CREATE INDEX idx_os_job_column ON outlier_detection_service.outlier_scores(job_id, column_name);
CREATE INDEX idx_os_tenant_job ON outlier_detection_service.outlier_scores(tenant_id, job_id);
CREATE INDEX idx_os_tenant_method ON outlier_detection_service.outlier_scores(tenant_id, method);
CREATE INDEX idx_os_parameters ON outlier_detection_service.outlier_scores USING GIN (parameters);

-- outlier_classifications indexes (16)
CREATE INDEX idx_oc_detection_id ON outlier_detection_service.outlier_classifications(detection_id);
CREATE INDEX idx_oc_job_id ON outlier_detection_service.outlier_classifications(job_id);
CREATE INDEX idx_oc_outlier_type ON outlier_detection_service.outlier_classifications(outlier_type);
CREATE INDEX idx_oc_severity ON outlier_detection_service.outlier_classifications(severity);
CREATE INDEX idx_oc_domain_category ON outlier_detection_service.outlier_classifications(domain_category);
CREATE INDEX idx_oc_consensus_score ON outlier_detection_service.outlier_classifications(consensus_score DESC);
CREATE INDEX idx_oc_methods_agreed ON outlier_detection_service.outlier_classifications(methods_agreed);
CREATE INDEX idx_oc_tenant_id ON outlier_detection_service.outlier_classifications(tenant_id);
CREATE INDEX idx_oc_created_at ON outlier_detection_service.outlier_classifications(created_at DESC);
CREATE INDEX idx_oc_job_type ON outlier_detection_service.outlier_classifications(job_id, outlier_type);
CREATE INDEX idx_oc_job_severity ON outlier_detection_service.outlier_classifications(job_id, severity);
CREATE INDEX idx_oc_job_domain ON outlier_detection_service.outlier_classifications(job_id, domain_category);
CREATE INDEX idx_oc_tenant_job ON outlier_detection_service.outlier_classifications(tenant_id, job_id);
CREATE INDEX idx_oc_tenant_type ON outlier_detection_service.outlier_classifications(tenant_id, outlier_type);
CREATE INDEX idx_oc_tenant_severity ON outlier_detection_service.outlier_classifications(tenant_id, severity);
CREATE INDEX idx_oc_classification_config ON outlier_detection_service.outlier_classifications USING GIN (classification_config);

-- outlier_treatments indexes (16)
CREATE INDEX idx_ot_detection_id ON outlier_detection_service.outlier_treatments(detection_id);
CREATE INDEX idx_ot_job_id ON outlier_detection_service.outlier_treatments(job_id);
CREATE INDEX idx_ot_column_name ON outlier_detection_service.outlier_treatments(column_name);
CREATE INDEX idx_ot_treatment_method ON outlier_detection_service.outlier_treatments(treatment_method);
CREATE INDEX idx_ot_approval_status ON outlier_detection_service.outlier_treatments(approval_status);
CREATE INDEX idx_ot_approved_by ON outlier_detection_service.outlier_treatments(approved_by);
CREATE INDEX idx_ot_provenance ON outlier_detection_service.outlier_treatments(provenance_hash);
CREATE INDEX idx_ot_tenant_id ON outlier_detection_service.outlier_treatments(tenant_id);
CREATE INDEX idx_ot_created_at ON outlier_detection_service.outlier_treatments(created_at DESC);
CREATE INDEX idx_ot_updated_at ON outlier_detection_service.outlier_treatments(updated_at DESC);
CREATE INDEX idx_ot_job_method ON outlier_detection_service.outlier_treatments(job_id, treatment_method);
CREATE INDEX idx_ot_job_column ON outlier_detection_service.outlier_treatments(job_id, column_name);
CREATE INDEX idx_ot_job_approval ON outlier_detection_service.outlier_treatments(job_id, approval_status);
CREATE INDEX idx_ot_tenant_job ON outlier_detection_service.outlier_treatments(tenant_id, job_id);
CREATE INDEX idx_ot_tenant_method ON outlier_detection_service.outlier_treatments(tenant_id, treatment_method);
CREATE INDEX idx_ot_tenant_approval ON outlier_detection_service.outlier_treatments(tenant_id, approval_status);

-- outlier_thresholds indexes (14)
CREATE INDEX idx_oth_name ON outlier_detection_service.outlier_thresholds(name);
CREATE INDEX idx_oth_column_name ON outlier_detection_service.outlier_thresholds(column_name);
CREATE INDEX idx_oth_detection_method ON outlier_detection_service.outlier_thresholds(detection_method);
CREATE INDEX idx_oth_sensitivity ON outlier_detection_service.outlier_thresholds(sensitivity);
CREATE INDEX idx_oth_is_active ON outlier_detection_service.outlier_thresholds(is_active);
CREATE INDEX idx_oth_version ON outlier_detection_service.outlier_thresholds(version);
CREATE INDEX idx_oth_tenant_id ON outlier_detection_service.outlier_thresholds(tenant_id);
CREATE INDEX idx_oth_created_by ON outlier_detection_service.outlier_thresholds(created_by);
CREATE INDEX idx_oth_created_at ON outlier_detection_service.outlier_thresholds(created_at DESC);
CREATE INDEX idx_oth_updated_at ON outlier_detection_service.outlier_thresholds(updated_at DESC);
CREATE INDEX idx_oth_tenant_active ON outlier_detection_service.outlier_thresholds(tenant_id, is_active);
CREATE INDEX idx_oth_tenant_method ON outlier_detection_service.outlier_thresholds(tenant_id, detection_method);
CREATE INDEX idx_oth_tenant_created ON outlier_detection_service.outlier_thresholds(tenant_id, created_at DESC);
CREATE INDEX idx_oth_statistical_basis ON outlier_detection_service.outlier_thresholds USING GIN (statistical_basis);

-- outlier_feedback indexes (14)
CREATE INDEX idx_of_detection_id ON outlier_detection_service.outlier_feedback(detection_id);
CREATE INDEX idx_of_job_id ON outlier_detection_service.outlier_feedback(job_id);
CREATE INDEX idx_of_feedback_type ON outlier_detection_service.outlier_feedback(feedback_type);
CREATE INDEX idx_of_original_label ON outlier_detection_service.outlier_feedback(original_label);
CREATE INDEX idx_of_corrected_label ON outlier_detection_service.outlier_feedback(corrected_label);
CREATE INDEX idx_of_feedback_by ON outlier_detection_service.outlier_feedback(feedback_by);
CREATE INDEX idx_of_feedback_at ON outlier_detection_service.outlier_feedback(feedback_at DESC);
CREATE INDEX idx_of_tenant_id ON outlier_detection_service.outlier_feedback(tenant_id);
CREATE INDEX idx_of_created_at ON outlier_detection_service.outlier_feedback(created_at DESC);
CREATE INDEX idx_of_job_type ON outlier_detection_service.outlier_feedback(job_id, feedback_type);
CREATE INDEX idx_of_job_detection ON outlier_detection_service.outlier_feedback(job_id, detection_id);
CREATE INDEX idx_of_tenant_job ON outlier_detection_service.outlier_feedback(tenant_id, job_id);
CREATE INDEX idx_of_tenant_type ON outlier_detection_service.outlier_feedback(tenant_id, feedback_type);
CREATE INDEX idx_of_tenant_feedback_at ON outlier_detection_service.outlier_feedback(tenant_id, feedback_at DESC);

-- outlier_impact_analyses indexes (14)
CREATE INDEX idx_oia_job_id ON outlier_detection_service.outlier_impact_analyses(job_id);
CREATE INDEX idx_oia_column_name ON outlier_detection_service.outlier_impact_analyses(column_name);
CREATE INDEX idx_oia_analysis_type ON outlier_detection_service.outlier_impact_analyses(analysis_type);
CREATE INDEX idx_oia_outliers_removed ON outlier_detection_service.outlier_impact_analyses(outliers_removed DESC);
CREATE INDEX idx_oia_provenance ON outlier_detection_service.outlier_impact_analyses(provenance_hash);
CREATE INDEX idx_oia_tenant_id ON outlier_detection_service.outlier_impact_analyses(tenant_id);
CREATE INDEX idx_oia_created_at ON outlier_detection_service.outlier_impact_analyses(created_at DESC);
CREATE INDEX idx_oia_job_column ON outlier_detection_service.outlier_impact_analyses(job_id, column_name);
CREATE INDEX idx_oia_job_type ON outlier_detection_service.outlier_impact_analyses(job_id, analysis_type);
CREATE INDEX idx_oia_tenant_job ON outlier_detection_service.outlier_impact_analyses(tenant_id, job_id);
CREATE INDEX idx_oia_tenant_type ON outlier_detection_service.outlier_impact_analyses(tenant_id, analysis_type);
CREATE INDEX idx_oia_correlation_changes ON outlier_detection_service.outlier_impact_analyses USING GIN (correlation_changes);
CREATE INDEX idx_oia_regression_impact ON outlier_detection_service.outlier_impact_analyses USING GIN (regression_impact);
CREATE INDEX idx_oia_sensitivity_metrics ON outlier_detection_service.outlier_impact_analyses USING GIN (sensitivity_metrics);

-- outlier_reports indexes (16)
CREATE INDEX idx_orp_job_id ON outlier_detection_service.outlier_reports(job_id);
CREATE INDEX idx_orp_report_type ON outlier_detection_service.outlier_reports(report_type);
CREATE INDEX idx_orp_quality_grade ON outlier_detection_service.outlier_reports(quality_grade);
CREATE INDEX idx_orp_overall_quality ON outlier_detection_service.outlier_reports(overall_quality_score DESC);
CREATE INDEX idx_orp_false_positive ON outlier_detection_service.outlier_reports(false_positive_rate);
CREATE INDEX idx_orp_outlier_rate_before ON outlier_detection_service.outlier_reports(outlier_rate_before DESC);
CREATE INDEX idx_orp_outlier_rate_after ON outlier_detection_service.outlier_reports(outlier_rate_after);
CREATE INDEX idx_orp_provenance ON outlier_detection_service.outlier_reports(provenance_hash);
CREATE INDEX idx_orp_tenant_id ON outlier_detection_service.outlier_reports(tenant_id);
CREATE INDEX idx_orp_generated_at ON outlier_detection_service.outlier_reports(generated_at DESC);
CREATE INDEX idx_orp_job_type ON outlier_detection_service.outlier_reports(job_id, report_type);
CREATE INDEX idx_orp_tenant_job ON outlier_detection_service.outlier_reports(tenant_id, job_id);
CREATE INDEX idx_orp_tenant_grade ON outlier_detection_service.outlier_reports(tenant_id, quality_grade);
CREATE INDEX idx_orp_column_summaries ON outlier_detection_service.outlier_reports USING GIN (column_summaries);
CREATE INDEX idx_orp_method_effectiveness ON outlier_detection_service.outlier_reports USING GIN (method_effectiveness);
CREATE INDEX idx_orp_recommendations ON outlier_detection_service.outlier_reports USING GIN (recommendations);

-- outlier_audit_log indexes (16)
CREATE INDEX idx_oal_job_id ON outlier_detection_service.outlier_audit_log(job_id);
CREATE INDEX idx_oal_action ON outlier_detection_service.outlier_audit_log(action);
CREATE INDEX idx_oal_entity_type ON outlier_detection_service.outlier_audit_log(entity_type);
CREATE INDEX idx_oal_entity_id ON outlier_detection_service.outlier_audit_log(entity_id);
CREATE INDEX idx_oal_provenance ON outlier_detection_service.outlier_audit_log(provenance_hash);
CREATE INDEX idx_oal_performed_by ON outlier_detection_service.outlier_audit_log(performed_by);
CREATE INDEX idx_oal_performed_at ON outlier_detection_service.outlier_audit_log(performed_at DESC);
CREATE INDEX idx_oal_tenant_id ON outlier_detection_service.outlier_audit_log(tenant_id);
CREATE INDEX idx_oal_job_action ON outlier_detection_service.outlier_audit_log(job_id, action);
CREATE INDEX idx_oal_job_entity ON outlier_detection_service.outlier_audit_log(job_id, entity_type);
CREATE INDEX idx_oal_action_entity ON outlier_detection_service.outlier_audit_log(action, entity_type);
CREATE INDEX idx_oal_entity_type_id ON outlier_detection_service.outlier_audit_log(entity_type, entity_id);
CREATE INDEX idx_oal_tenant_job ON outlier_detection_service.outlier_audit_log(tenant_id, job_id);
CREATE INDEX idx_oal_tenant_action ON outlier_detection_service.outlier_audit_log(tenant_id, action);
CREATE INDEX idx_oal_tenant_performed ON outlier_detection_service.outlier_audit_log(tenant_id, performed_at DESC);
CREATE INDEX idx_oal_details ON outlier_detection_service.outlier_audit_log USING GIN (details);

-- outlier_events indexes (hypertable-aware) (8)
CREATE INDEX idx_oe_job_id ON outlier_detection_service.outlier_events(job_id, event_time DESC);
CREATE INDEX idx_oe_event_type ON outlier_detection_service.outlier_events(event_type, event_time DESC);
CREATE INDEX idx_oe_stage ON outlier_detection_service.outlier_events(stage, event_time DESC);
CREATE INDEX idx_oe_tenant_id ON outlier_detection_service.outlier_events(tenant_id, event_time DESC);
CREATE INDEX idx_oe_tenant_job ON outlier_detection_service.outlier_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_oe_tenant_type ON outlier_detection_service.outlier_events(tenant_id, event_type, event_time DESC);
CREATE INDEX idx_oe_provenance ON outlier_detection_service.outlier_events(provenance_hash, event_time DESC);
CREATE INDEX idx_oe_details ON outlier_detection_service.outlier_events USING GIN (details);

-- detection_events indexes (hypertable-aware) (8)
CREATE INDEX idx_de_job_id ON outlier_detection_service.detection_events(job_id, event_time DESC);
CREATE INDEX idx_de_column_name ON outlier_detection_service.detection_events(column_name, event_time DESC);
CREATE INDEX idx_de_detection_method ON outlier_detection_service.detection_events(detection_method, event_time DESC);
CREATE INDEX idx_de_outlier_count ON outlier_detection_service.detection_events(outlier_count, event_time DESC);
CREATE INDEX idx_de_tenant_id ON outlier_detection_service.detection_events(tenant_id, event_time DESC);
CREATE INDEX idx_de_tenant_job ON outlier_detection_service.detection_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_de_tenant_method ON outlier_detection_service.detection_events(tenant_id, detection_method, event_time DESC);
CREATE INDEX idx_de_tenant_column ON outlier_detection_service.detection_events(tenant_id, column_name, event_time DESC);

-- treatment_events indexes (hypertable-aware) (8)
CREATE INDEX idx_te_job_id ON outlier_detection_service.treatment_events(job_id, event_time DESC);
CREATE INDEX idx_te_treatment_method ON outlier_detection_service.treatment_events(treatment_method, event_time DESC);
CREATE INDEX idx_te_column_name ON outlier_detection_service.treatment_events(column_name, event_time DESC);
CREATE INDEX idx_te_outliers_treated ON outlier_detection_service.treatment_events(outliers_treated, event_time DESC);
CREATE INDEX idx_te_tenant_id ON outlier_detection_service.treatment_events(tenant_id, event_time DESC);
CREATE INDEX idx_te_tenant_job ON outlier_detection_service.treatment_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_te_tenant_method ON outlier_detection_service.treatment_events(tenant_id, treatment_method, event_time DESC);
CREATE INDEX idx_te_provenance ON outlier_detection_service.treatment_events(provenance_hash, event_time DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- outlier_jobs: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY oj_tenant_read ON outlier_detection_service.outlier_jobs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY oj_tenant_write ON outlier_detection_service.outlier_jobs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_detections: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_detections ENABLE ROW LEVEL SECURITY;
CREATE POLICY od_tenant_read ON outlier_detection_service.outlier_detections
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY od_tenant_write ON outlier_detection_service.outlier_detections
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_scores: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY os_tenant_read ON outlier_detection_service.outlier_scores
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY os_tenant_write ON outlier_detection_service.outlier_scores
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_classifications: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_classifications ENABLE ROW LEVEL SECURITY;
CREATE POLICY oc_tenant_read ON outlier_detection_service.outlier_classifications
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY oc_tenant_write ON outlier_detection_service.outlier_classifications
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_treatments: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_treatments ENABLE ROW LEVEL SECURITY;
CREATE POLICY ot_tenant_read ON outlier_detection_service.outlier_treatments
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ot_tenant_write ON outlier_detection_service.outlier_treatments
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_thresholds: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_thresholds ENABLE ROW LEVEL SECURITY;
CREATE POLICY oth_tenant_read ON outlier_detection_service.outlier_thresholds
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY oth_tenant_write ON outlier_detection_service.outlier_thresholds
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_feedback: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_feedback ENABLE ROW LEVEL SECURITY;
CREATE POLICY of_tenant_read ON outlier_detection_service.outlier_feedback
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY of_tenant_write ON outlier_detection_service.outlier_feedback
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_impact_analyses: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_impact_analyses ENABLE ROW LEVEL SECURITY;
CREATE POLICY oia_tenant_read ON outlier_detection_service.outlier_impact_analyses
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY oia_tenant_write ON outlier_detection_service.outlier_impact_analyses
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_reports: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY orp_tenant_read ON outlier_detection_service.outlier_reports
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY orp_tenant_write ON outlier_detection_service.outlier_reports
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_audit_log: tenant-scoped
ALTER TABLE outlier_detection_service.outlier_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY oal_tenant_read ON outlier_detection_service.outlier_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY oal_tenant_write ON outlier_detection_service.outlier_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- outlier_events: open (hypertable)
ALTER TABLE outlier_detection_service.outlier_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY oe_tenant_read ON outlier_detection_service.outlier_events
    FOR SELECT USING (TRUE);
CREATE POLICY oe_tenant_write ON outlier_detection_service.outlier_events
    FOR ALL USING (TRUE);

-- detection_events: open (hypertable)
ALTER TABLE outlier_detection_service.detection_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY de_tenant_read ON outlier_detection_service.detection_events
    FOR SELECT USING (TRUE);
CREATE POLICY de_tenant_write ON outlier_detection_service.detection_events
    FOR ALL USING (TRUE);

-- treatment_events: open (hypertable)
ALTER TABLE outlier_detection_service.treatment_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY te_tenant_read ON outlier_detection_service.treatment_events
    FOR SELECT USING (TRUE);
CREATE POLICY te_tenant_write ON outlier_detection_service.treatment_events
    FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA outlier_detection_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA outlier_detection_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA outlier_detection_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON outlier_detection_service.outlier_hourly_stats TO greenlang_app;
GRANT SELECT ON outlier_detection_service.detection_hourly_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA outlier_detection_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA outlier_detection_service TO greenlang_readonly;
GRANT SELECT ON outlier_detection_service.outlier_hourly_stats TO greenlang_readonly;
GRANT SELECT ON outlier_detection_service.detection_hourly_stats TO greenlang_readonly;

-- Admin role
GRANT ALL ON SCHEMA outlier_detection_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA outlier_detection_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA outlier_detection_service TO greenlang_admin;

-- Add outlier detection service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'outlier_detection:jobs:read', 'outlier_detection', 'jobs_read', 'View outlier detection jobs and their progress'),
    (gen_random_uuid(), 'outlier_detection:jobs:write', 'outlier_detection', 'jobs_write', 'Create, start, cancel, and manage outlier detection jobs'),
    (gen_random_uuid(), 'outlier_detection:detections:read', 'outlier_detection', 'detections_read', 'View outlier detection results and scores'),
    (gen_random_uuid(), 'outlier_detection:detections:write', 'outlier_detection', 'detections_write', 'Run and manage outlier detections'),
    (gen_random_uuid(), 'outlier_detection:scores:read', 'outlier_detection', 'scores_read', 'View outlier scores from multiple methods'),
    (gen_random_uuid(), 'outlier_detection:scores:write', 'outlier_detection', 'scores_write', 'Create and manage outlier scores'),
    (gen_random_uuid(), 'outlier_detection:classifications:read', 'outlier_detection', 'classifications_read', 'View outlier classifications and severity levels'),
    (gen_random_uuid(), 'outlier_detection:classifications:write', 'outlier_detection', 'classifications_write', 'Create and manage outlier classifications'),
    (gen_random_uuid(), 'outlier_detection:treatments:read', 'outlier_detection', 'treatments_read', 'View outlier treatment actions and approvals'),
    (gen_random_uuid(), 'outlier_detection:treatments:write', 'outlier_detection', 'treatments_write', 'Apply and manage outlier treatments'),
    (gen_random_uuid(), 'outlier_detection:thresholds:read', 'outlier_detection', 'thresholds_read', 'View threshold configurations and profiles'),
    (gen_random_uuid(), 'outlier_detection:thresholds:write', 'outlier_detection', 'thresholds_write', 'Create, update, and manage threshold configurations'),
    (gen_random_uuid(), 'outlier_detection:feedback:read', 'outlier_detection', 'feedback_read', 'View human-in-the-loop feedback entries'),
    (gen_random_uuid(), 'outlier_detection:feedback:write', 'outlier_detection', 'feedback_write', 'Submit and manage feedback on outlier detections'),
    (gen_random_uuid(), 'outlier_detection:audit:read', 'outlier_detection', 'audit_read', 'View outlier detection audit log entries and provenance chains'),
    (gen_random_uuid(), 'outlier_detection:admin', 'outlier_detection', 'admin', 'Outlier detection service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep outlier event records for 90 days
SELECT add_retention_policy('outlier_detection_service.outlier_events', INTERVAL '90 days');

-- Keep detection event records for 90 days
SELECT add_retention_policy('outlier_detection_service.detection_events', INTERVAL '90 days');

-- Keep treatment event records for 90 days
SELECT add_retention_policy('outlier_detection_service.treatment_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on outlier_events after 7 days
ALTER TABLE outlier_detection_service.outlier_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('outlier_detection_service.outlier_events', INTERVAL '7 days');

-- Enable compression on detection_events after 7 days
ALTER TABLE outlier_detection_service.detection_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('outlier_detection_service.detection_events', INTERVAL '7 days');

-- Enable compression on treatment_events after 7 days
ALTER TABLE outlier_detection_service.treatment_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('outlier_detection_service.treatment_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Outlier Detection Agent (GL-DATA-X-016)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-016', 'Outlier Detection Agent',
 'Comprehensive outlier detection and treatment engine for GreenLang Climate OS. Performs statistical outlier detection (z-score, modified z-score, IQR, Grubbs'' test, Dixon''s Q test, Chauvenet''s criterion). Supports distribution-based detection (Mahalanobis distance, minimum covariance determinant, chi-squared). Implements density-based detection (LOF, DBSCAN, OPTICS, KDE). Provides model-based detection (Isolation Forest, One-Class SVM, Autoencoder, Elliptic Envelope, HBOS). Applies time-series anomaly detection (STL decomposition, seasonal hybrid ESD, exponential smoothing, CUSUM, Bollinger bands). Supports ensemble methods (voting, stacking, weighted consensus). Classifies outliers (point/contextual/collective, severity scoring, domain-aware labeling). Recommends treatments (removal, capping/winsorization, transformation, imputation, flagging). Performs impact analysis (effect on mean/variance/correlation/regression, sensitivity testing). Includes human-in-the-loop feedback with false positive tracking and threshold tuning. SHA-256 provenance chains for zero-hallucination audit trail.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/outlier-detection', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Outlier Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-016', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/outlier-detector-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"outlier-detection", "anomaly-detection", "data-quality", "statistical", "machine-learning", "ensemble", "treatment", "impact-analysis"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "healthcare", "agriculture"}',
 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Outlier Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-X-016', '1.0.0', 'statistical_detection', 'detection',
 'Detect outliers using classical statistical methods. Supports z-score and modified z-score for normally distributed data, IQR for robust non-parametric detection, Grubbs'' test for single outlier identification, Dixon''s Q test for small samples, and Chauvenet''s criterion for measurement rejection. Configurable thresholds and tail selection (upper/lower/both)',
 '{"dataset", "columns", "methods", "config"}', '{"detections", "scores", "classifications", "summary"}',
 '{"methods": ["z_score", "modified_z_score", "iqr", "grubbs", "dixon_q", "chauvenet"], "z_threshold": 3.0, "iqr_multiplier": 1.5, "tail": ["upper", "lower", "both"], "min_sample_size": 10}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'distribution_detection', 'detection',
 'Detect multivariate outliers using distribution-based methods. Supports Mahalanobis distance for elliptical distributions, minimum covariance determinant (MCD) for robust estimation, and chi-squared test for multivariate normality assessment. Handles high-dimensional data with dimensionality reduction',
 '{"dataset", "columns", "methods", "config"}', '{"detections", "scores", "distances", "summary"}',
 '{"methods": ["mahalanobis", "mcd", "chi_squared"], "contamination": 0.05, "support_fraction": 0.75, "dimensionality_reduction": true}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'density_detection', 'detection',
 'Detect outliers using density-based methods. Supports Local Outlier Factor (LOF) for local density anomalies, DBSCAN for cluster-based detection, OPTICS for variable-density environments, and Kernel Density Estimation (KDE) for probability-based scoring. Handles non-globular cluster shapes and variable-density data',
 '{"dataset", "columns", "methods", "config"}', '{"detections", "scores", "density_maps", "summary"}',
 '{"methods": ["lof", "dbscan", "optics", "kde"], "n_neighbors": [5, 10, 20], "eps": "auto", "min_samples": 5, "bandwidth": "scott", "contamination": 0.05}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'model_detection', 'detection',
 'Detect outliers using machine learning models. Supports Isolation Forest for efficient anomaly isolation, One-Class SVM for boundary-based detection, Autoencoder for reconstruction error scoring, Elliptic Envelope for Gaussian-assumed data, and Histogram-Based Outlier Score (HBOS) for fast univariate detection. Configurable contamination rates and model parameters',
 '{"dataset", "columns", "methods", "config"}', '{"detections", "scores", "model_metrics", "feature_importance"}',
 '{"methods": ["isolation_forest", "one_class_svm", "autoencoder", "elliptic_envelope", "hbos"], "n_estimators": [100, 200], "contamination": 0.05, "kernel": "rbf", "hidden_layers": [64, 32, 64], "cross_validation": true}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'timeseries_detection', 'detection',
 'Detect anomalies in time-series data. Supports STL decomposition for trend/seasonal separation, seasonal hybrid ESD for multiple anomalies, exponential smoothing for forecast-based detection, CUSUM for change-point detection, and Bollinger bands for volatility-based scoring. Respects temporal ordering and seasonal patterns',
 '{"dataset", "columns", "time_column", "methods", "config"}', '{"detections", "scores", "decomposition", "summary"}',
 '{"methods": ["stl_decomposition", "seasonal_esd", "exponential_smoothing", "cusum", "bollinger"], "seasonal_periods": [7, 12, 24, 52, 365], "alpha": 0.05, "max_anomalies": 0.1, "bollinger_window": 20, "bollinger_std": 2.0}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'ensemble_detection', 'detection',
 'Combine multiple detection methods using ensemble techniques. Supports majority voting for democratic consensus, stacking with meta-learner for optimal combination, and weighted consensus based on method confidence. Reduces false positives and improves robustness across data distributions',
 '{"dataset", "columns", "methods_to_combine", "config"}', '{"detections", "ensemble_scores", "method_weights", "consensus_summary"}',
 '{"ensemble_methods": ["ensemble_vote", "ensemble_stack", "ensemble_weighted"], "min_agreement": 0.5, "weight_strategy": ["equal", "performance", "confidence"], "meta_learner": "logistic_regression", "cross_validation_folds": 5}'::jsonb),

('GL-DATA-X-016', '1.0.0', 'outlier_treatment', 'processing',
 'Apply treatment actions to detected outliers. Supports removal (deletion), capping (floor/ceiling), winsorization (percentile-based), log/sqrt/Box-Cox transformation, imputation (replace with estimated value), and flagging (mark without modification). Treatment requires approval workflow for critical severity outliers',
 '{"detections", "treatment_method", "config"}', '{"treated_data", "treatment_summary", "impact_preview"}',
 '{"methods": ["removal", "capping", "winsorization", "log_transform", "sqrt_transform", "box_cox", "imputation", "flagging", "no_action"], "capping_percentiles": [0.01, 0.05, 0.95, 0.99], "approval_required_for": ["critical", "high"], "dry_run": true, "rollback_enabled": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Outlier Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Outlier Detection depends on Schema Compiler for input/output validation
('GL-DATA-X-016', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Threshold profiles, job configurations, and treatment rules are validated against JSON Schema definitions'),

-- Outlier Detection depends on Registry for agent discovery
('GL-DATA-X-016', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for outlier detection pipeline orchestration'),

-- Outlier Detection depends on Access Guard for policy enforcement
('GL-DATA-X-016', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for outlier detection jobs and results'),

-- Outlier Detection depends on Observability Agent for metrics
('GL-DATA-X-016', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Detection metrics, treatment rates, false positive rates, and pipeline throughput are reported to observability'),

-- Outlier Detection optionally uses Citations for provenance tracking
('GL-DATA-X-016', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Outlier detection provenance and treatment audit trails are registered with the citation service'),

-- Outlier Detection optionally uses Reproducibility for determinism
('GL-DATA-X-016', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Detection results are verified for reproducibility across re-execution with identical inputs and thresholds'),

-- Outlier Detection optionally uses QA Test Harness
('GL-DATA-X-016', 'GL-FOUND-X-009', '>=1.0.0', true,
 'Detection results are validated through the QA Test Harness for zero-hallucination verification'),

-- Outlier Detection optionally integrates with Data Quality Profiler
('GL-DATA-X-016', 'GL-DATA-X-013', '>=1.0.0', true,
 'Data quality profiling identifies distribution characteristics that inform outlier detection threshold selection'),

-- Outlier Detection optionally integrates with Missing Value Imputer
('GL-DATA-X-016', 'GL-DATA-X-015', '>=1.0.0', true,
 'Missing values are imputed before outlier detection to prevent false positives from incomplete data'),

-- Outlier Detection optionally integrates with Duplicate Detector
('GL-DATA-X-016', 'GL-DATA-X-014', '>=1.0.0', true,
 'Deduplicated datasets are passed to the outlier detector to prevent duplicate-induced bias in detection')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Outlier Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-016', 'Outlier Detection Agent',
 'Comprehensive outlier detection and treatment engine. Performs statistical detection (z-score/modified z-score/IQR/Grubbs/Dixon Q/Chauvenet). Supports distribution-based detection (Mahalanobis/MCD/chi-squared). Implements density-based detection (LOF/DBSCAN/OPTICS/KDE). Provides model-based detection (Isolation Forest/One-Class SVM/Autoencoder/Elliptic Envelope/HBOS). Applies time-series anomaly detection (STL/seasonal ESD/exponential smoothing/CUSUM/Bollinger). Ensemble methods (voting/stacking/weighted). Classifies outliers (point/contextual/collective). Recommends treatments (removal/capping/winsorization/transformation/imputation/flagging). Impact analysis. Human-in-the-loop feedback. SHA-256 provenance chains for zero-hallucination audit trail.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA outlier_detection_service IS 'Outlier Detection Agent for GreenLang Climate OS (AGENT-DATA-013) - statistical/distribution/density/model-based detection, time-series anomaly detection, ensemble methods, outlier classification, treatment recommendations, impact analysis, human-in-the-loop feedback, and comprehensive audit logging with provenance chains';
COMMENT ON TABLE outlier_detection_service.outlier_jobs IS 'Job tracking for outlier detection runs with dataset IDs, threshold profile reference, status/stage progression, per-stage record counts, outlier rate, dry-run flag, configuration, and SHA-256 provenance hash';
COMMENT ON TABLE outlier_detection_service.outlier_detections IS 'Per-record/column detection results with detection method, outlier score, is_outlier flag, confidence, direction (upper/lower/both), and contextual information';
COMMENT ON TABLE outlier_detection_service.outlier_scores IS 'Multi-method outlier scores per detection with raw score, normalized score (0-1), threshold value, exceeds_threshold flag, percentile, and method parameters';
COMMENT ON TABLE outlier_detection_service.outlier_classifications IS 'Outlier type (point/contextual/collective/seasonal/trend) and severity (low/medium/high/critical) classification with ensemble consensus scoring';
COMMENT ON TABLE outlier_detection_service.outlier_treatments IS 'Treatment actions applied to detected outliers with method (removal/capping/winsorization/transform/imputation/flagging), original/treated values, approval status, and SHA-256 provenance hash';
COMMENT ON TABLE outlier_detection_service.outlier_thresholds IS 'Per-column/method threshold configurations with sensitivity level, auto-tuning flag, statistical basis, version, and activation state';
COMMENT ON TABLE outlier_detection_service.outlier_feedback IS 'Human-in-the-loop feedback entries with type (confirm/reject/reclassify/adjust), original/corrected labels, severity and confidence adjustments, and notes';
COMMENT ON TABLE outlier_detection_service.outlier_impact_analyses IS 'Impact analysis of outlier removal/treatment on dataset statistics with before/after mean, variance, std dev, correlation changes, regression impact, and sensitivity metrics';
COMMENT ON TABLE outlier_detection_service.outlier_reports IS 'Outlier summary reports with before/after outlier rates, per-column summaries, method effectiveness, treatment summary, false positive rate, quality grade, and recommendations';
COMMENT ON TABLE outlier_detection_service.outlier_audit_log IS 'Comprehensive audit trail for all outlier detection operations with action, entity type/ID, details (JSONB), provenance hash, performer, and timestamp';
COMMENT ON TABLE outlier_detection_service.outlier_events IS 'TimescaleDB hypertable: outlier detection lifecycle event time-series with job ID, event type, pipeline stage, record count, duration, and details';
COMMENT ON TABLE outlier_detection_service.detection_events IS 'TimescaleDB hypertable: detection event time-series with job ID, column name, detection method, outlier count, total scanned, and duration';
COMMENT ON TABLE outlier_detection_service.treatment_events IS 'TimescaleDB hypertable: treatment event time-series with job ID, treatment method, column name, outliers treated, duration, and provenance hash';
COMMENT ON MATERIALIZED VIEW outlier_detection_service.outlier_hourly_stats IS 'Continuous aggregate: hourly outlier event statistics by event type with total events, unique jobs, avg/sum record counts, avg/max/min duration, and per-stage event counts';
COMMENT ON MATERIALIZED VIEW outlier_detection_service.detection_hourly_stats IS 'Continuous aggregate: hourly detection statistics by method with total detections, unique jobs, avg/sum outlier counts, avg/sum total scanned, and avg/max/min duration';

COMMENT ON COLUMN outlier_detection_service.outlier_jobs.status IS 'Job status: pending, running, completed, failed, cancelled';
COMMENT ON COLUMN outlier_detection_service.outlier_jobs.stage IS 'Current pipeline stage: scan, detect, classify, treat, analyze, report, complete';
COMMENT ON COLUMN outlier_detection_service.outlier_jobs.outlier_rate IS 'Overall ratio of outlier values to total values (0-1), computed during detection';
COMMENT ON COLUMN outlier_detection_service.outlier_jobs.dry_run IS 'If true, detection and treatment are simulated without modifying the dataset';
COMMENT ON COLUMN outlier_detection_service.outlier_jobs.provenance_hash IS 'SHA-256 provenance hash for integrity verification and audit trail';
COMMENT ON COLUMN outlier_detection_service.outlier_detections.detection_method IS 'Detection method: z_score, modified_z_score, iqr, grubbs, dixon_q, chauvenet, mahalanobis, mcd, chi_squared, lof, dbscan, optics, kde, isolation_forest, one_class_svm, autoencoder, elliptic_envelope, hbos, stl_decomposition, seasonal_esd, exponential_smoothing, cusum, bollinger, ensemble_vote, ensemble_stack, ensemble_weighted';
COMMENT ON COLUMN outlier_detection_service.outlier_detections.direction IS 'Outlier tail direction: upper (above threshold), lower (below threshold), both';
COMMENT ON COLUMN outlier_detection_service.outlier_detections.confidence IS 'Confidence score (0-1) for the outlier detection';
COMMENT ON COLUMN outlier_detection_service.outlier_scores.normalized_score IS 'Normalized outlier score (0-1) for cross-method comparison';
COMMENT ON COLUMN outlier_detection_service.outlier_scores.percentile IS 'Percentile rank of the value within the column distribution (0-1)';
COMMENT ON COLUMN outlier_detection_service.outlier_classifications.outlier_type IS 'Outlier type: point (single value), contextual (conditional on context), collective (group anomaly), seasonal (temporal pattern), trend (long-term shift), unknown';
COMMENT ON COLUMN outlier_detection_service.outlier_classifications.severity IS 'Outlier severity: low, medium, high, critical';
COMMENT ON COLUMN outlier_detection_service.outlier_classifications.consensus_score IS 'Ensemble consensus score (0-1) indicating agreement among detection methods';
COMMENT ON COLUMN outlier_detection_service.outlier_treatments.treatment_method IS 'Treatment method: removal, capping, winsorization, log_transform, sqrt_transform, box_cox, imputation, flagging, no_action';
COMMENT ON COLUMN outlier_detection_service.outlier_treatments.approval_status IS 'Approval status: auto_approved, pending_review, approved, rejected, overridden';
COMMENT ON COLUMN outlier_detection_service.outlier_treatments.provenance_hash IS 'SHA-256 provenance hash of the treatment result for audit trail verification';
COMMENT ON COLUMN outlier_detection_service.outlier_thresholds.sensitivity IS 'Detection sensitivity: low (fewer false positives), medium (balanced), high (fewer false negatives), very_high (aggressive), custom';
COMMENT ON COLUMN outlier_detection_service.outlier_thresholds.auto_tune IS 'If true, threshold is automatically tuned based on data distribution and feedback';
COMMENT ON COLUMN outlier_detection_service.outlier_feedback.feedback_type IS 'Feedback type: confirm (true positive), reject (false positive), reclassify (wrong type/severity), adjust_threshold (tune sensitivity), annotate (add context)';
COMMENT ON COLUMN outlier_detection_service.outlier_impact_analyses.analysis_type IS 'Impact analysis type: mean_impact, variance_impact, correlation_impact, regression_impact, distribution_impact, sensitivity, comprehensive, before_after_comparison';
COMMENT ON COLUMN outlier_detection_service.outlier_reports.quality_grade IS 'Letter grade (A-F) for overall outlier detection quality based on false positive rate and treatment effectiveness';
COMMENT ON COLUMN outlier_detection_service.outlier_reports.false_positive_rate IS 'Estimated false positive rate (0-1) based on feedback and validation';
COMMENT ON COLUMN outlier_detection_service.outlier_reports.outlier_rate_before IS 'Overall outlier rate before treatment (0-1)';
COMMENT ON COLUMN outlier_detection_service.outlier_reports.outlier_rate_after IS 'Overall outlier rate after treatment (0-1)';
COMMENT ON COLUMN outlier_detection_service.outlier_audit_log.action IS 'Audit action: job_created, job_started, job_completed, job_failed, job_cancelled, detection_completed, scoring_completed, classification_completed, treatment_applied/approved/rejected, impact_analysis_completed, report_generated, threshold_created/updated/deleted/activated/deactivated, feedback_submitted/applied, etc.';
COMMENT ON COLUMN outlier_detection_service.outlier_audit_log.entity_type IS 'Entity type: job, detection, score, classification, treatment, threshold, feedback, impact_analysis, report, config';
COMMENT ON COLUMN outlier_detection_service.outlier_audit_log.provenance_hash IS 'SHA-256 provenance hash linking the audit entry to a specific data state';
COMMENT ON COLUMN outlier_detection_service.outlier_events.event_type IS 'Outlier event type: job_started/completed/failed/cancelled, scan_started/completed/failed, detection_started/completed/failed, classification_started/completed/failed, treatment_started/completed/failed, analysis_started/completed/failed, reporting_started/completed/failed, stage_transition, progress_update, threshold_breach';
COMMENT ON COLUMN outlier_detection_service.outlier_events.stage IS 'Pipeline stage: scan, detect, classify, treat, analyze, report, complete';
COMMENT ON COLUMN outlier_detection_service.detection_events.detection_method IS 'Detection method used for this event';
COMMENT ON COLUMN outlier_detection_service.treatment_events.treatment_method IS 'Treatment method: removal, capping, winsorization, log_transform, sqrt_transform, box_cox, imputation, flagging, no_action';
