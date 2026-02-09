-- =============================================================================
-- V040: Data Quality Profiler Service Schema
-- =============================================================================
-- Component: AGENT-DATA-010 (Data Quality Profiler)
-- Agent ID:  GL-DATA-X-013
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Data Quality Profiler Agent (GL-DATA-X-013) with capabilities
-- for dataset profiling (row/column/type statistics, cardinality,
-- distributions), quality assessment scoring across 6 dimensions
-- (completeness, validity, consistency, timeliness, uniqueness,
-- accuracy), custom quality rule definitions and evaluation engine,
-- quality issue detection and tracking, statistical anomaly detection
-- (z-score, IQR, isolation forest), quality gate enforcement with
-- configurable thresholds, and historical quality trend analysis
-- with provenance chains.
-- =============================================================================
-- Tables (10):
--   1. dataset_profiles          - Core profiling results per dataset
--   2. column_profiles           - Per-column statistics and distributions
--   3. quality_assessments       - Overall quality scores per dataset
--   4. quality_dimensions        - Per-dimension quality scores
--   5. quality_rules             - Custom rule definitions
--   6. rule_evaluations          - Rule evaluation results
--   7. quality_issues            - Detected quality issues
--   8. anomaly_detections        - Anomaly detection results
--   9. quality_gates             - Gate definitions and outcomes
--  10. quality_trends            - Historical quality tracking
--
-- Hypertables (3):
--  11. quality_events            - Quality event time-series (hypertable)
--  12. profile_events            - Profile event time-series (hypertable)
--  13. anomaly_events            - Anomaly event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. quality_hourly_stats      - Hourly avg score, assessment count, issue count
--   2. profile_hourly_stats      - Hourly profile count, avg row count, avg column count
--
-- Also includes: 100+ indexes (B-tree, GIN, partial, composite),
-- 20+ RLS policies per tenant, retention policies (90 days on hypertables),
-- compression policies (7 days), updated_at trigger, security permissions
-- for greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-013.
-- Previous: V039__spend_categorizer_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS data_quality_profiler_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION data_quality_profiler_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: data_quality_profiler_service.dataset_profiles
-- =============================================================================
-- Core profiling results per dataset. Each profile captures row and column
-- counts, memory estimate, schema hash for drift detection, processing
-- status, metadata, and provenance hash. Tenant-scoped.

CREATE TABLE data_quality_profiler_service.dataset_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id VARCHAR(64) UNIQUE NOT NULL,
    dataset_name VARCHAR(512) NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    column_count INTEGER NOT NULL DEFAULT 0,
    memory_estimate_bytes BIGINT DEFAULT 0,
    schema_hash VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'completed',
    metadata JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Status constraint
ALTER TABLE data_quality_profiler_service.dataset_profiles
    ADD CONSTRAINT chk_dp_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

-- Row count must be non-negative
ALTER TABLE data_quality_profiler_service.dataset_profiles
    ADD CONSTRAINT chk_dp_row_count_non_negative
    CHECK (row_count >= 0);

-- Column count must be non-negative
ALTER TABLE data_quality_profiler_service.dataset_profiles
    ADD CONSTRAINT chk_dp_column_count_non_negative
    CHECK (column_count >= 0);

-- Memory estimate must be non-negative
ALTER TABLE data_quality_profiler_service.dataset_profiles
    ADD CONSTRAINT chk_dp_memory_estimate_non_negative
    CHECK (memory_estimate_bytes IS NULL OR memory_estimate_bytes >= 0);

-- Profile ID must not be empty
ALTER TABLE data_quality_profiler_service.dataset_profiles
    ADD CONSTRAINT chk_dp_profile_id_not_empty
    CHECK (LENGTH(TRIM(profile_id)) > 0);

-- Dataset name must not be empty
ALTER TABLE data_quality_profiler_service.dataset_profiles
    ADD CONSTRAINT chk_dp_dataset_name_not_empty
    CHECK (LENGTH(TRIM(dataset_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.dataset_profiles
    ADD CONSTRAINT chk_dp_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_dp_updated_at
    BEFORE UPDATE ON data_quality_profiler_service.dataset_profiles
    FOR EACH ROW
    EXECUTE FUNCTION data_quality_profiler_service.set_updated_at();

-- =============================================================================
-- Table 2: data_quality_profiler_service.column_profiles
-- =============================================================================
-- Per-column statistics and distribution information. Each column profile
-- captures data type, null counts and percentages, unique counts,
-- cardinality, min/max/mean/median/stddev values, percentile
-- distributions, most common values, pattern detection, and
-- provenance hash. Linked to dataset_profiles via profile_id.

CREATE TABLE data_quality_profiler_service.column_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id VARCHAR(64) NOT NULL,
    column_name VARCHAR(256) NOT NULL,
    data_type VARCHAR(32) NOT NULL DEFAULT 'unknown',
    total_values INTEGER DEFAULT 0,
    non_null_count INTEGER DEFAULT 0,
    null_count INTEGER DEFAULT 0,
    null_pct DOUBLE PRECISION DEFAULT 0.0,
    unique_count INTEGER DEFAULT 0,
    cardinality DOUBLE PRECISION DEFAULT 0.0,
    min_value TEXT,
    max_value TEXT,
    mean_value DOUBLE PRECISION,
    median_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION,
    percentiles JSONB DEFAULT '{}'::jsonb,
    most_common JSONB DEFAULT '[]'::jsonb,
    pattern VARCHAR(256),
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to dataset_profiles
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT fk_cp_profile_id
    FOREIGN KEY (profile_id) REFERENCES data_quality_profiler_service.dataset_profiles(profile_id)
    ON DELETE CASCADE;

-- Data type constraint
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_data_type
    CHECK (data_type IN (
        'unknown', 'string', 'integer', 'float', 'boolean', 'date',
        'datetime', 'time', 'decimal', 'text', 'categorical', 'numeric',
        'uuid', 'email', 'url', 'json', 'binary', 'array'
    ));

-- Total values must be non-negative
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_total_values_non_negative
    CHECK (total_values IS NULL OR total_values >= 0);

-- Non-null count must be non-negative
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_non_null_count_non_negative
    CHECK (non_null_count IS NULL OR non_null_count >= 0);

-- Null count must be non-negative
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_null_count_non_negative
    CHECK (null_count IS NULL OR null_count >= 0);

-- Null percentage must be between 0 and 100
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_null_pct_range
    CHECK (null_pct IS NULL OR (null_pct >= 0.0 AND null_pct <= 100.0));

-- Unique count must be non-negative
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_unique_count_non_negative
    CHECK (unique_count IS NULL OR unique_count >= 0);

-- Cardinality must be between 0 and 1
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_cardinality_range
    CHECK (cardinality IS NULL OR (cardinality >= 0.0 AND cardinality <= 1.0));

-- Column name must not be empty
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.column_profiles
    ADD CONSTRAINT chk_cp_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 3: data_quality_profiler_service.quality_assessments
-- =============================================================================
-- Overall quality scores per dataset. Each assessment captures the
-- overall score (0-1), quality level classification, total issues
-- detected, processing status, metadata, and provenance hash.
-- Tenant-scoped.

CREATE TABLE data_quality_profiler_service.quality_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id VARCHAR(64) UNIQUE NOT NULL,
    dataset_name VARCHAR(512) NOT NULL,
    overall_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    quality_level VARCHAR(32) NOT NULL DEFAULT 'critical',
    total_issues INTEGER DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'completed',
    metadata JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Quality level constraint
ALTER TABLE data_quality_profiler_service.quality_assessments
    ADD CONSTRAINT chk_qa_quality_level
    CHECK (quality_level IN ('excellent', 'good', 'acceptable', 'poor', 'critical'));

-- Overall score must be between 0 and 1
ALTER TABLE data_quality_profiler_service.quality_assessments
    ADD CONSTRAINT chk_qa_overall_score_range
    CHECK (overall_score >= 0.0 AND overall_score <= 1.0);

-- Total issues must be non-negative
ALTER TABLE data_quality_profiler_service.quality_assessments
    ADD CONSTRAINT chk_qa_total_issues_non_negative
    CHECK (total_issues IS NULL OR total_issues >= 0);

-- Status constraint
ALTER TABLE data_quality_profiler_service.quality_assessments
    ADD CONSTRAINT chk_qa_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

-- Assessment ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_assessments
    ADD CONSTRAINT chk_qa_assessment_id_not_empty
    CHECK (LENGTH(TRIM(assessment_id)) > 0);

-- Dataset name must not be empty
ALTER TABLE data_quality_profiler_service.quality_assessments
    ADD CONSTRAINT chk_qa_dataset_name_not_empty
    CHECK (LENGTH(TRIM(dataset_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_assessments
    ADD CONSTRAINT chk_qa_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 4: data_quality_profiler_service.quality_dimensions
-- =============================================================================
-- Per-dimension quality scores. Each dimension captures the dimension
-- type (completeness, validity, consistency, timeliness, uniqueness,
-- accuracy), score, weight, weighted score, issues count, and detail
-- breakdown. Linked to quality_assessments via assessment_id.

CREATE TABLE data_quality_profiler_service.quality_dimensions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id VARCHAR(64) NOT NULL,
    dimension VARCHAR(32) NOT NULL,
    score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    weighted_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    issues_count INTEGER DEFAULT 0,
    details JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to quality_assessments
ALTER TABLE data_quality_profiler_service.quality_dimensions
    ADD CONSTRAINT fk_qd_assessment_id
    FOREIGN KEY (assessment_id) REFERENCES data_quality_profiler_service.quality_assessments(assessment_id)
    ON DELETE CASCADE;

-- Dimension constraint
ALTER TABLE data_quality_profiler_service.quality_dimensions
    ADD CONSTRAINT chk_qd_dimension
    CHECK (dimension IN ('completeness', 'validity', 'consistency', 'timeliness', 'uniqueness', 'accuracy'));

-- Score must be between 0 and 1
ALTER TABLE data_quality_profiler_service.quality_dimensions
    ADD CONSTRAINT chk_qd_score_range
    CHECK (score >= 0.0 AND score <= 1.0);

-- Weight must be between 0 and 1
ALTER TABLE data_quality_profiler_service.quality_dimensions
    ADD CONSTRAINT chk_qd_weight_range
    CHECK (weight >= 0.0 AND weight <= 1.0);

-- Weighted score must be between 0 and 1
ALTER TABLE data_quality_profiler_service.quality_dimensions
    ADD CONSTRAINT chk_qd_weighted_score_range
    CHECK (weighted_score >= 0.0 AND weighted_score <= 1.0);

-- Issues count must be non-negative
ALTER TABLE data_quality_profiler_service.quality_dimensions
    ADD CONSTRAINT chk_qd_issues_count_non_negative
    CHECK (issues_count IS NULL OR issues_count >= 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_dimensions
    ADD CONSTRAINT chk_qd_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 5: data_quality_profiler_service.quality_rules
-- =============================================================================
-- Custom quality rule definitions. Each rule captures the rule type
-- (null_check, range_check, regex_check, uniqueness_check,
-- freshness_check, custom), target column, operator, threshold,
-- parameters, priority, activation state, and provenance hash.
-- Tenant-scoped.

CREATE TABLE data_quality_profiler_service.quality_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(256) NOT NULL,
    description TEXT DEFAULT '',
    rule_type VARCHAR(32) NOT NULL,
    column_name VARCHAR(256),
    operator VARCHAR(32),
    threshold DOUBLE PRECISION,
    parameters JSONB DEFAULT '{}'::jsonb,
    active BOOLEAN NOT NULL DEFAULT true,
    priority INTEGER NOT NULL DEFAULT 100,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rule type constraint
ALTER TABLE data_quality_profiler_service.quality_rules
    ADD CONSTRAINT chk_qr_rule_type
    CHECK (rule_type IN (
        'null_check', 'range_check', 'regex_check', 'uniqueness_check',
        'freshness_check', 'custom', 'type_check', 'referential_check',
        'completeness_check', 'consistency_check', 'format_check'
    ));

-- Operator constraint if specified
ALTER TABLE data_quality_profiler_service.quality_rules
    ADD CONSTRAINT chk_qr_operator
    CHECK (operator IS NULL OR operator IN (
        'eq', 'ne', 'gt', 'gte', 'lt', 'lte',
        'between', 'in', 'not_in', 'matches', 'not_matches',
        'is_null', 'is_not_null', 'contains', 'starts_with', 'ends_with'
    ));

-- Rule ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_rules
    ADD CONSTRAINT chk_qr_rule_id_not_empty
    CHECK (LENGTH(TRIM(rule_id)) > 0);

-- Name must not be empty
ALTER TABLE data_quality_profiler_service.quality_rules
    ADD CONSTRAINT chk_qr_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Priority must be non-negative
ALTER TABLE data_quality_profiler_service.quality_rules
    ADD CONSTRAINT chk_qr_priority_non_negative
    CHECK (priority >= 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_rules
    ADD CONSTRAINT chk_qr_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_qr_updated_at
    BEFORE UPDATE ON data_quality_profiler_service.quality_rules
    FOR EACH ROW
    EXECUTE FUNCTION data_quality_profiler_service.set_updated_at();

-- =============================================================================
-- Table 6: data_quality_profiler_service.rule_evaluations
-- =============================================================================
-- Rule evaluation results. Each evaluation captures the rule reference,
-- dataset name, pass/fail status, actual value vs threshold, message,
-- and provenance hash. Tenant-scoped.

CREATE TABLE data_quality_profiler_service.rule_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evaluation_id VARCHAR(64) UNIQUE NOT NULL,
    rule_id VARCHAR(64) NOT NULL,
    dataset_name VARCHAR(512) NOT NULL,
    passed BOOLEAN NOT NULL DEFAULT false,
    actual_value DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    message TEXT DEFAULT '',
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Evaluation ID must not be empty
ALTER TABLE data_quality_profiler_service.rule_evaluations
    ADD CONSTRAINT chk_re_evaluation_id_not_empty
    CHECK (LENGTH(TRIM(evaluation_id)) > 0);

-- Rule ID must not be empty
ALTER TABLE data_quality_profiler_service.rule_evaluations
    ADD CONSTRAINT chk_re_rule_id_not_empty
    CHECK (LENGTH(TRIM(rule_id)) > 0);

-- Dataset name must not be empty
ALTER TABLE data_quality_profiler_service.rule_evaluations
    ADD CONSTRAINT chk_re_dataset_name_not_empty
    CHECK (LENGTH(TRIM(dataset_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.rule_evaluations
    ADD CONSTRAINT chk_re_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 7: data_quality_profiler_service.quality_issues
-- =============================================================================
-- Detected quality issues. Each issue captures the severity, dimension,
-- column, row index, description, suggested fix, and provenance hash.
-- Linked to quality_assessments via assessment_id. Tenant-scoped.

CREATE TABLE data_quality_profiler_service.quality_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id VARCHAR(64) UNIQUE NOT NULL,
    assessment_id VARCHAR(64),
    severity VARCHAR(32) NOT NULL DEFAULT 'warning',
    dimension VARCHAR(32),
    column_name VARCHAR(256),
    row_index INTEGER,
    description TEXT NOT NULL,
    suggested_fix TEXT,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Severity constraint
ALTER TABLE data_quality_profiler_service.quality_issues
    ADD CONSTRAINT chk_qi_severity
    CHECK (severity IN ('critical', 'error', 'warning', 'info'));

-- Dimension constraint if specified
ALTER TABLE data_quality_profiler_service.quality_issues
    ADD CONSTRAINT chk_qi_dimension
    CHECK (dimension IS NULL OR dimension IN (
        'completeness', 'validity', 'consistency', 'timeliness', 'uniqueness', 'accuracy'
    ));

-- Row index must be non-negative if specified
ALTER TABLE data_quality_profiler_service.quality_issues
    ADD CONSTRAINT chk_qi_row_index_non_negative
    CHECK (row_index IS NULL OR row_index >= 0);

-- Issue ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_issues
    ADD CONSTRAINT chk_qi_issue_id_not_empty
    CHECK (LENGTH(TRIM(issue_id)) > 0);

-- Description must not be empty
ALTER TABLE data_quality_profiler_service.quality_issues
    ADD CONSTRAINT chk_qi_description_not_empty
    CHECK (LENGTH(TRIM(description)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_issues
    ADD CONSTRAINT chk_qi_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 8: data_quality_profiler_service.anomaly_detections
-- =============================================================================
-- Anomaly detection results. Each anomaly captures the dataset, column,
-- detection method (z_score, iqr, isolation_forest, mad, percentile),
-- observed value, expected range, z-score, severity, description,
-- and affected row indices. Tenant-scoped.

CREATE TABLE data_quality_profiler_service.anomaly_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    anomaly_id VARCHAR(64) UNIQUE NOT NULL,
    dataset_name VARCHAR(512) NOT NULL,
    column_name VARCHAR(256) NOT NULL,
    method VARCHAR(32) NOT NULL,
    value DOUBLE PRECISION,
    expected_min DOUBLE PRECISION,
    expected_max DOUBLE PRECISION,
    z_score DOUBLE PRECISION,
    severity VARCHAR(32) NOT NULL DEFAULT 'medium',
    description TEXT,
    row_indices JSONB DEFAULT '[]'::jsonb,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Method constraint
ALTER TABLE data_quality_profiler_service.anomaly_detections
    ADD CONSTRAINT chk_ad_method
    CHECK (method IN ('z_score', 'iqr', 'isolation_forest', 'mad', 'percentile', 'grubbs', 'dbscan'));

-- Severity constraint
ALTER TABLE data_quality_profiler_service.anomaly_detections
    ADD CONSTRAINT chk_ad_severity
    CHECK (severity IN ('critical', 'high', 'medium', 'low'));

-- Anomaly ID must not be empty
ALTER TABLE data_quality_profiler_service.anomaly_detections
    ADD CONSTRAINT chk_ad_anomaly_id_not_empty
    CHECK (LENGTH(TRIM(anomaly_id)) > 0);

-- Dataset name must not be empty
ALTER TABLE data_quality_profiler_service.anomaly_detections
    ADD CONSTRAINT chk_ad_dataset_name_not_empty
    CHECK (LENGTH(TRIM(dataset_name)) > 0);

-- Column name must not be empty
ALTER TABLE data_quality_profiler_service.anomaly_detections
    ADD CONSTRAINT chk_ad_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.anomaly_detections
    ADD CONSTRAINT chk_ad_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 9: data_quality_profiler_service.quality_gates
-- =============================================================================
-- Quality gate definitions and outcomes. Each gate captures the name,
-- conditions (JSONB array of required checks), threshold score,
-- outcome (pass/fail/pending), dimension scores snapshot, and
-- provenance hash. Tenant-scoped.

CREATE TABLE data_quality_profiler_service.quality_gates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gate_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(256) NOT NULL,
    conditions JSONB NOT NULL DEFAULT '[]'::jsonb,
    threshold DOUBLE PRECISION DEFAULT 0.70,
    outcome VARCHAR(16) DEFAULT 'pending',
    dimension_scores JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Outcome constraint
ALTER TABLE data_quality_profiler_service.quality_gates
    ADD CONSTRAINT chk_qg_outcome
    CHECK (outcome IN ('pass', 'fail', 'pending', 'warning', 'skipped'));

-- Threshold must be between 0 and 1
ALTER TABLE data_quality_profiler_service.quality_gates
    ADD CONSTRAINT chk_qg_threshold_range
    CHECK (threshold IS NULL OR (threshold >= 0.0 AND threshold <= 1.0));

-- Gate ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_gates
    ADD CONSTRAINT chk_qg_gate_id_not_empty
    CHECK (LENGTH(TRIM(gate_id)) > 0);

-- Name must not be empty
ALTER TABLE data_quality_profiler_service.quality_gates
    ADD CONSTRAINT chk_qg_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_gates
    ADD CONSTRAINT chk_qg_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: data_quality_profiler_service.quality_trends
-- =============================================================================
-- Historical quality tracking per dataset. Each trend record captures
-- the reporting period, overall score, per-dimension score breakdown,
-- trend direction, and percentage change from previous period.
-- Tenant-scoped.

CREATE TABLE data_quality_profiler_service.quality_trends (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name VARCHAR(512) NOT NULL,
    period VARCHAR(32) NOT NULL,
    overall_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    dimension_scores JSONB DEFAULT '{}'::jsonb,
    direction VARCHAR(16) DEFAULT 'unknown',
    change_pct DOUBLE PRECISION DEFAULT 0.0,
    provenance_hash VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Period constraint
ALTER TABLE data_quality_profiler_service.quality_trends
    ADD CONSTRAINT chk_qt_period
    CHECK (period IN ('hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'));

-- Direction constraint
ALTER TABLE data_quality_profiler_service.quality_trends
    ADD CONSTRAINT chk_qt_direction
    CHECK (direction IN ('improving', 'degrading', 'stable', 'unknown'));

-- Overall score must be between 0 and 1
ALTER TABLE data_quality_profiler_service.quality_trends
    ADD CONSTRAINT chk_qt_overall_score_range
    CHECK (overall_score >= 0.0 AND overall_score <= 1.0);

-- Dataset name must not be empty
ALTER TABLE data_quality_profiler_service.quality_trends
    ADD CONSTRAINT chk_qt_dataset_name_not_empty
    CHECK (LENGTH(TRIM(dataset_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_trends
    ADD CONSTRAINT chk_qt_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: data_quality_profiler_service.quality_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording quality assessment events as a
-- time-series. Each event captures the event type, dataset name,
-- dimension, score, details, and tenant. Partitioned by event_time
-- for time-series queries. Retained for 90 days with compression
-- after 7 days.

CREATE TABLE data_quality_profiler_service.quality_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    dataset_name VARCHAR(512) NOT NULL,
    dimension VARCHAR(32),
    score DOUBLE PRECISION,
    details JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('data_quality_profiler_service.quality_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE data_quality_profiler_service.quality_events
    ADD CONSTRAINT chk_qe_event_type
    CHECK (event_type IN (
        'assessment_started', 'assessment_completed', 'assessment_failed',
        'dimension_scored', 'rule_evaluated', 'issue_detected',
        'gate_evaluated', 'trend_recorded', 'threshold_breach'
    ));

-- Dimension constraint if specified
ALTER TABLE data_quality_profiler_service.quality_events
    ADD CONSTRAINT chk_qe_dimension
    CHECK (dimension IS NULL OR dimension IN (
        'completeness', 'validity', 'consistency', 'timeliness', 'uniqueness', 'accuracy'
    ));

-- Score must be between 0 and 1 if specified
ALTER TABLE data_quality_profiler_service.quality_events
    ADD CONSTRAINT chk_qe_score_range
    CHECK (score IS NULL OR (score >= 0.0 AND score <= 1.0));

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.quality_events
    ADD CONSTRAINT chk_qe_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 12: data_quality_profiler_service.profile_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording profile lifecycle events as a
-- time-series. Each event captures the event type, profile ID,
-- dataset name, column count, row count, details, and tenant.
-- Partitioned by event_time for time-series queries. Retained for
-- 90 days with compression after 7 days.

CREATE TABLE data_quality_profiler_service.profile_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    profile_id VARCHAR(64),
    dataset_name VARCHAR(512) NOT NULL,
    column_count INTEGER,
    row_count INTEGER,
    details JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('data_quality_profiler_service.profile_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE data_quality_profiler_service.profile_events
    ADD CONSTRAINT chk_pe_event_type
    CHECK (event_type IN (
        'profile_started', 'profile_completed', 'profile_failed',
        'column_profiled', 'schema_detected', 'statistics_computed',
        'distribution_analyzed', 'pattern_detected', 'drift_detected'
    ));

-- Column count must be non-negative if specified
ALTER TABLE data_quality_profiler_service.profile_events
    ADD CONSTRAINT chk_pe_column_count_non_negative
    CHECK (column_count IS NULL OR column_count >= 0);

-- Row count must be non-negative if specified
ALTER TABLE data_quality_profiler_service.profile_events
    ADD CONSTRAINT chk_pe_row_count_non_negative
    CHECK (row_count IS NULL OR row_count >= 0);

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.profile_events
    ADD CONSTRAINT chk_pe_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 13: data_quality_profiler_service.anomaly_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording anomaly detection events as a
-- time-series. Each event captures the event type, dataset name,
-- column name, detection method, severity, observed value, details,
-- and tenant. Partitioned by event_time for time-series queries.
-- Retained for 90 days with compression after 7 days.

CREATE TABLE data_quality_profiler_service.anomaly_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    dataset_name VARCHAR(512) NOT NULL,
    column_name VARCHAR(256),
    method VARCHAR(32),
    severity VARCHAR(32),
    value DOUBLE PRECISION,
    details JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('data_quality_profiler_service.anomaly_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE data_quality_profiler_service.anomaly_events
    ADD CONSTRAINT chk_ane_event_type
    CHECK (event_type IN (
        'detection_started', 'detection_completed', 'detection_failed',
        'anomaly_found', 'anomaly_confirmed', 'anomaly_dismissed',
        'threshold_exceeded', 'pattern_anomaly', 'distribution_shift'
    ));

-- Method constraint if specified
ALTER TABLE data_quality_profiler_service.anomaly_events
    ADD CONSTRAINT chk_ane_method
    CHECK (method IS NULL OR method IN ('z_score', 'iqr', 'isolation_forest', 'mad', 'percentile', 'grubbs', 'dbscan'));

-- Severity constraint if specified
ALTER TABLE data_quality_profiler_service.anomaly_events
    ADD CONSTRAINT chk_ane_severity
    CHECK (severity IS NULL OR severity IN ('critical', 'high', 'medium', 'low'));

-- Tenant ID must not be empty
ALTER TABLE data_quality_profiler_service.anomaly_events
    ADD CONSTRAINT chk_ane_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Continuous Aggregate: data_quality_profiler_service.quality_hourly_stats
-- =============================================================================
-- Precomputed hourly quality assessment statistics by event type for
-- dashboard queries, quality monitoring, and trend analysis.

CREATE MATERIALIZED VIEW data_quality_profiler_service.quality_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    event_type,
    COUNT(*) AS total_events,
    COUNT(DISTINCT dataset_name) AS unique_datasets,
    AVG(score) AS avg_score,
    MIN(score) AS min_score,
    MAX(score) AS max_score,
    COUNT(*) FILTER (WHERE dimension = 'completeness') AS completeness_count,
    COUNT(*) FILTER (WHERE dimension = 'validity') AS validity_count,
    COUNT(*) FILTER (WHERE dimension = 'consistency') AS consistency_count,
    COUNT(*) FILTER (WHERE dimension = 'timeliness') AS timeliness_count,
    COUNT(*) FILTER (WHERE dimension = 'uniqueness') AS uniqueness_count,
    COUNT(*) FILTER (WHERE dimension = 'accuracy') AS accuracy_count
FROM data_quality_profiler_service.quality_events
WHERE event_time IS NOT NULL
GROUP BY bucket, event_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('data_quality_profiler_service.quality_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: data_quality_profiler_service.profile_hourly_stats
-- =============================================================================
-- Precomputed hourly profile statistics by event type for dashboard
-- queries, profiling throughput monitoring, and dataset size analysis.

CREATE MATERIALIZED VIEW data_quality_profiler_service.profile_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    event_type,
    COUNT(*) AS total_events,
    COUNT(DISTINCT dataset_name) AS unique_datasets,
    COUNT(DISTINCT profile_id) AS unique_profiles,
    AVG(row_count) AS avg_row_count,
    AVG(column_count) AS avg_column_count,
    MAX(row_count) AS max_row_count,
    MAX(column_count) AS max_column_count,
    SUM(row_count) AS total_rows_profiled
FROM data_quality_profiler_service.profile_events
WHERE event_time IS NOT NULL
GROUP BY bucket, event_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('data_quality_profiler_service.profile_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- dataset_profiles indexes (14)
CREATE INDEX idx_dp_profile_id ON data_quality_profiler_service.dataset_profiles(profile_id);
CREATE INDEX idx_dp_dataset_name ON data_quality_profiler_service.dataset_profiles(dataset_name);
CREATE INDEX idx_dp_tenant_id ON data_quality_profiler_service.dataset_profiles(tenant_id);
CREATE INDEX idx_dp_status ON data_quality_profiler_service.dataset_profiles(status);
CREATE INDEX idx_dp_schema_hash ON data_quality_profiler_service.dataset_profiles(schema_hash);
CREATE INDEX idx_dp_provenance ON data_quality_profiler_service.dataset_profiles(provenance_hash);
CREATE INDEX idx_dp_created_at ON data_quality_profiler_service.dataset_profiles(created_at DESC);
CREATE INDEX idx_dp_updated_at ON data_quality_profiler_service.dataset_profiles(updated_at DESC);
CREATE INDEX idx_dp_tenant_dataset ON data_quality_profiler_service.dataset_profiles(tenant_id, dataset_name);
CREATE INDEX idx_dp_tenant_status ON data_quality_profiler_service.dataset_profiles(tenant_id, status);
CREATE INDEX idx_dp_tenant_created ON data_quality_profiler_service.dataset_profiles(tenant_id, created_at DESC);
CREATE INDEX idx_dp_dataset_status ON data_quality_profiler_service.dataset_profiles(dataset_name, status);
CREATE INDEX idx_dp_dataset_created ON data_quality_profiler_service.dataset_profiles(dataset_name, created_at DESC);
CREATE INDEX idx_dp_metadata ON data_quality_profiler_service.dataset_profiles USING GIN (metadata);

-- column_profiles indexes (16)
CREATE INDEX idx_cp_id ON data_quality_profiler_service.column_profiles(id);
CREATE INDEX idx_cp_profile_id ON data_quality_profiler_service.column_profiles(profile_id);
CREATE INDEX idx_cp_column_name ON data_quality_profiler_service.column_profiles(column_name);
CREATE INDEX idx_cp_data_type ON data_quality_profiler_service.column_profiles(data_type);
CREATE INDEX idx_cp_tenant_id ON data_quality_profiler_service.column_profiles(tenant_id);
CREATE INDEX idx_cp_provenance ON data_quality_profiler_service.column_profiles(provenance_hash);
CREATE INDEX idx_cp_created_at ON data_quality_profiler_service.column_profiles(created_at DESC);
CREATE INDEX idx_cp_null_pct ON data_quality_profiler_service.column_profiles(null_pct DESC);
CREATE INDEX idx_cp_cardinality ON data_quality_profiler_service.column_profiles(cardinality DESC);
CREATE INDEX idx_cp_unique_count ON data_quality_profiler_service.column_profiles(unique_count DESC);
CREATE INDEX idx_cp_profile_column ON data_quality_profiler_service.column_profiles(profile_id, column_name);
CREATE INDEX idx_cp_profile_type ON data_quality_profiler_service.column_profiles(profile_id, data_type);
CREATE INDEX idx_cp_tenant_profile ON data_quality_profiler_service.column_profiles(tenant_id, profile_id);
CREATE INDEX idx_cp_tenant_column ON data_quality_profiler_service.column_profiles(tenant_id, column_name);
CREATE INDEX idx_cp_percentiles ON data_quality_profiler_service.column_profiles USING GIN (percentiles);
CREATE INDEX idx_cp_most_common ON data_quality_profiler_service.column_profiles USING GIN (most_common);

-- quality_assessments indexes (13)
CREATE INDEX idx_qa_assessment_id ON data_quality_profiler_service.quality_assessments(assessment_id);
CREATE INDEX idx_qa_dataset_name ON data_quality_profiler_service.quality_assessments(dataset_name);
CREATE INDEX idx_qa_tenant_id ON data_quality_profiler_service.quality_assessments(tenant_id);
CREATE INDEX idx_qa_overall_score ON data_quality_profiler_service.quality_assessments(overall_score DESC);
CREATE INDEX idx_qa_quality_level ON data_quality_profiler_service.quality_assessments(quality_level);
CREATE INDEX idx_qa_status ON data_quality_profiler_service.quality_assessments(status);
CREATE INDEX idx_qa_provenance ON data_quality_profiler_service.quality_assessments(provenance_hash);
CREATE INDEX idx_qa_created_at ON data_quality_profiler_service.quality_assessments(created_at DESC);
CREATE INDEX idx_qa_tenant_dataset ON data_quality_profiler_service.quality_assessments(tenant_id, dataset_name);
CREATE INDEX idx_qa_tenant_level ON data_quality_profiler_service.quality_assessments(tenant_id, quality_level);
CREATE INDEX idx_qa_tenant_status ON data_quality_profiler_service.quality_assessments(tenant_id, status);
CREATE INDEX idx_qa_dataset_level ON data_quality_profiler_service.quality_assessments(dataset_name, quality_level);
CREATE INDEX idx_qa_metadata ON data_quality_profiler_service.quality_assessments USING GIN (metadata);

-- quality_dimensions indexes (12)
CREATE INDEX idx_qd_id ON data_quality_profiler_service.quality_dimensions(id);
CREATE INDEX idx_qd_assessment_id ON data_quality_profiler_service.quality_dimensions(assessment_id);
CREATE INDEX idx_qd_dimension ON data_quality_profiler_service.quality_dimensions(dimension);
CREATE INDEX idx_qd_score ON data_quality_profiler_service.quality_dimensions(score DESC);
CREATE INDEX idx_qd_weighted_score ON data_quality_profiler_service.quality_dimensions(weighted_score DESC);
CREATE INDEX idx_qd_tenant_id ON data_quality_profiler_service.quality_dimensions(tenant_id);
CREATE INDEX idx_qd_provenance ON data_quality_profiler_service.quality_dimensions(provenance_hash);
CREATE INDEX idx_qd_created_at ON data_quality_profiler_service.quality_dimensions(created_at DESC);
CREATE INDEX idx_qd_assessment_dim ON data_quality_profiler_service.quality_dimensions(assessment_id, dimension);
CREATE INDEX idx_qd_tenant_dimension ON data_quality_profiler_service.quality_dimensions(tenant_id, dimension);
CREATE INDEX idx_qd_tenant_assessment ON data_quality_profiler_service.quality_dimensions(tenant_id, assessment_id);
CREATE INDEX idx_qd_details ON data_quality_profiler_service.quality_dimensions USING GIN (details);

-- quality_rules indexes (14)
CREATE INDEX idx_qr_rule_id ON data_quality_profiler_service.quality_rules(rule_id);
CREATE INDEX idx_qr_name ON data_quality_profiler_service.quality_rules(name);
CREATE INDEX idx_qr_rule_type ON data_quality_profiler_service.quality_rules(rule_type);
CREATE INDEX idx_qr_column_name ON data_quality_profiler_service.quality_rules(column_name);
CREATE INDEX idx_qr_operator ON data_quality_profiler_service.quality_rules(operator);
CREATE INDEX idx_qr_active ON data_quality_profiler_service.quality_rules(active);
CREATE INDEX idx_qr_priority ON data_quality_profiler_service.quality_rules(priority);
CREATE INDEX idx_qr_tenant_id ON data_quality_profiler_service.quality_rules(tenant_id);
CREATE INDEX idx_qr_provenance ON data_quality_profiler_service.quality_rules(provenance_hash);
CREATE INDEX idx_qr_created_at ON data_quality_profiler_service.quality_rules(created_at DESC);
CREATE INDEX idx_qr_updated_at ON data_quality_profiler_service.quality_rules(updated_at DESC);
CREATE INDEX idx_qr_tenant_active ON data_quality_profiler_service.quality_rules(tenant_id, active);
CREATE INDEX idx_qr_tenant_type ON data_quality_profiler_service.quality_rules(tenant_id, rule_type);
CREATE INDEX idx_qr_parameters ON data_quality_profiler_service.quality_rules USING GIN (parameters);

-- rule_evaluations indexes (12)
CREATE INDEX idx_re_evaluation_id ON data_quality_profiler_service.rule_evaluations(evaluation_id);
CREATE INDEX idx_re_rule_id ON data_quality_profiler_service.rule_evaluations(rule_id);
CREATE INDEX idx_re_dataset_name ON data_quality_profiler_service.rule_evaluations(dataset_name);
CREATE INDEX idx_re_passed ON data_quality_profiler_service.rule_evaluations(passed);
CREATE INDEX idx_re_tenant_id ON data_quality_profiler_service.rule_evaluations(tenant_id);
CREATE INDEX idx_re_provenance ON data_quality_profiler_service.rule_evaluations(provenance_hash);
CREATE INDEX idx_re_created_at ON data_quality_profiler_service.rule_evaluations(created_at DESC);
CREATE INDEX idx_re_rule_dataset ON data_quality_profiler_service.rule_evaluations(rule_id, dataset_name);
CREATE INDEX idx_re_rule_passed ON data_quality_profiler_service.rule_evaluations(rule_id, passed);
CREATE INDEX idx_re_tenant_rule ON data_quality_profiler_service.rule_evaluations(tenant_id, rule_id);
CREATE INDEX idx_re_tenant_dataset ON data_quality_profiler_service.rule_evaluations(tenant_id, dataset_name);
CREATE INDEX idx_re_tenant_passed ON data_quality_profiler_service.rule_evaluations(tenant_id, passed);

-- quality_issues indexes (14)
CREATE INDEX idx_qi_issue_id ON data_quality_profiler_service.quality_issues(issue_id);
CREATE INDEX idx_qi_assessment_id ON data_quality_profiler_service.quality_issues(assessment_id);
CREATE INDEX idx_qi_severity ON data_quality_profiler_service.quality_issues(severity);
CREATE INDEX idx_qi_dimension ON data_quality_profiler_service.quality_issues(dimension);
CREATE INDEX idx_qi_column_name ON data_quality_profiler_service.quality_issues(column_name);
CREATE INDEX idx_qi_row_index ON data_quality_profiler_service.quality_issues(row_index);
CREATE INDEX idx_qi_tenant_id ON data_quality_profiler_service.quality_issues(tenant_id);
CREATE INDEX idx_qi_provenance ON data_quality_profiler_service.quality_issues(provenance_hash);
CREATE INDEX idx_qi_created_at ON data_quality_profiler_service.quality_issues(created_at DESC);
CREATE INDEX idx_qi_assessment_severity ON data_quality_profiler_service.quality_issues(assessment_id, severity);
CREATE INDEX idx_qi_assessment_dimension ON data_quality_profiler_service.quality_issues(assessment_id, dimension);
CREATE INDEX idx_qi_tenant_severity ON data_quality_profiler_service.quality_issues(tenant_id, severity);
CREATE INDEX idx_qi_tenant_dimension ON data_quality_profiler_service.quality_issues(tenant_id, dimension);
CREATE INDEX idx_qi_tenant_assessment ON data_quality_profiler_service.quality_issues(tenant_id, assessment_id);

-- anomaly_detections indexes (15)
CREATE INDEX idx_ad_anomaly_id ON data_quality_profiler_service.anomaly_detections(anomaly_id);
CREATE INDEX idx_ad_dataset_name ON data_quality_profiler_service.anomaly_detections(dataset_name);
CREATE INDEX idx_ad_column_name ON data_quality_profiler_service.anomaly_detections(column_name);
CREATE INDEX idx_ad_method ON data_quality_profiler_service.anomaly_detections(method);
CREATE INDEX idx_ad_severity ON data_quality_profiler_service.anomaly_detections(severity);
CREATE INDEX idx_ad_z_score ON data_quality_profiler_service.anomaly_detections(z_score DESC);
CREATE INDEX idx_ad_tenant_id ON data_quality_profiler_service.anomaly_detections(tenant_id);
CREATE INDEX idx_ad_provenance ON data_quality_profiler_service.anomaly_detections(provenance_hash);
CREATE INDEX idx_ad_created_at ON data_quality_profiler_service.anomaly_detections(created_at DESC);
CREATE INDEX idx_ad_dataset_column ON data_quality_profiler_service.anomaly_detections(dataset_name, column_name);
CREATE INDEX idx_ad_dataset_method ON data_quality_profiler_service.anomaly_detections(dataset_name, method);
CREATE INDEX idx_ad_dataset_severity ON data_quality_profiler_service.anomaly_detections(dataset_name, severity);
CREATE INDEX idx_ad_tenant_dataset ON data_quality_profiler_service.anomaly_detections(tenant_id, dataset_name);
CREATE INDEX idx_ad_tenant_severity ON data_quality_profiler_service.anomaly_detections(tenant_id, severity);
CREATE INDEX idx_ad_row_indices ON data_quality_profiler_service.anomaly_detections USING GIN (row_indices);

-- quality_gates indexes (11)
CREATE INDEX idx_qg_gate_id ON data_quality_profiler_service.quality_gates(gate_id);
CREATE INDEX idx_qg_name ON data_quality_profiler_service.quality_gates(name);
CREATE INDEX idx_qg_outcome ON data_quality_profiler_service.quality_gates(outcome);
CREATE INDEX idx_qg_threshold ON data_quality_profiler_service.quality_gates(threshold);
CREATE INDEX idx_qg_tenant_id ON data_quality_profiler_service.quality_gates(tenant_id);
CREATE INDEX idx_qg_provenance ON data_quality_profiler_service.quality_gates(provenance_hash);
CREATE INDEX idx_qg_created_at ON data_quality_profiler_service.quality_gates(created_at DESC);
CREATE INDEX idx_qg_tenant_outcome ON data_quality_profiler_service.quality_gates(tenant_id, outcome);
CREATE INDEX idx_qg_tenant_name ON data_quality_profiler_service.quality_gates(tenant_id, name);
CREATE INDEX idx_qg_conditions ON data_quality_profiler_service.quality_gates USING GIN (conditions);
CREATE INDEX idx_qg_dimension_scores ON data_quality_profiler_service.quality_gates USING GIN (dimension_scores);

-- quality_trends indexes (12)
CREATE INDEX idx_qt_id ON data_quality_profiler_service.quality_trends(id);
CREATE INDEX idx_qt_dataset_name ON data_quality_profiler_service.quality_trends(dataset_name);
CREATE INDEX idx_qt_period ON data_quality_profiler_service.quality_trends(period);
CREATE INDEX idx_qt_overall_score ON data_quality_profiler_service.quality_trends(overall_score DESC);
CREATE INDEX idx_qt_direction ON data_quality_profiler_service.quality_trends(direction);
CREATE INDEX idx_qt_tenant_id ON data_quality_profiler_service.quality_trends(tenant_id);
CREATE INDEX idx_qt_provenance ON data_quality_profiler_service.quality_trends(provenance_hash);
CREATE INDEX idx_qt_created_at ON data_quality_profiler_service.quality_trends(created_at DESC);
CREATE INDEX idx_qt_dataset_period ON data_quality_profiler_service.quality_trends(dataset_name, period);
CREATE INDEX idx_qt_tenant_dataset ON data_quality_profiler_service.quality_trends(tenant_id, dataset_name);
CREATE INDEX idx_qt_tenant_period ON data_quality_profiler_service.quality_trends(tenant_id, period);
CREATE INDEX idx_qt_dimension_scores ON data_quality_profiler_service.quality_trends USING GIN (dimension_scores);

-- quality_events indexes (hypertable-aware) (6)
CREATE INDEX idx_qe_dataset_name ON data_quality_profiler_service.quality_events(dataset_name, event_time DESC);
CREATE INDEX idx_qe_tenant_id ON data_quality_profiler_service.quality_events(tenant_id, event_time DESC);
CREATE INDEX idx_qe_event_type ON data_quality_profiler_service.quality_events(event_type, event_time DESC);
CREATE INDEX idx_qe_dimension ON data_quality_profiler_service.quality_events(dimension, event_time DESC);
CREATE INDEX idx_qe_tenant_dataset ON data_quality_profiler_service.quality_events(tenant_id, dataset_name, event_time DESC);
CREATE INDEX idx_qe_details ON data_quality_profiler_service.quality_events USING GIN (details);

-- profile_events indexes (hypertable-aware) (6)
CREATE INDEX idx_pe_profile_id ON data_quality_profiler_service.profile_events(profile_id, event_time DESC);
CREATE INDEX idx_pe_dataset_name ON data_quality_profiler_service.profile_events(dataset_name, event_time DESC);
CREATE INDEX idx_pe_tenant_id ON data_quality_profiler_service.profile_events(tenant_id, event_time DESC);
CREATE INDEX idx_pe_event_type ON data_quality_profiler_service.profile_events(event_type, event_time DESC);
CREATE INDEX idx_pe_tenant_dataset ON data_quality_profiler_service.profile_events(tenant_id, dataset_name, event_time DESC);
CREATE INDEX idx_pe_details ON data_quality_profiler_service.profile_events USING GIN (details);

-- anomaly_events indexes (hypertable-aware) (7)
CREATE INDEX idx_ane_dataset_name ON data_quality_profiler_service.anomaly_events(dataset_name, event_time DESC);
CREATE INDEX idx_ane_column_name ON data_quality_profiler_service.anomaly_events(column_name, event_time DESC);
CREATE INDEX idx_ane_tenant_id ON data_quality_profiler_service.anomaly_events(tenant_id, event_time DESC);
CREATE INDEX idx_ane_event_type ON data_quality_profiler_service.anomaly_events(event_type, event_time DESC);
CREATE INDEX idx_ane_method ON data_quality_profiler_service.anomaly_events(method, event_time DESC);
CREATE INDEX idx_ane_severity ON data_quality_profiler_service.anomaly_events(severity, event_time DESC);
CREATE INDEX idx_ane_details ON data_quality_profiler_service.anomaly_events USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- dataset_profiles: tenant-scoped
ALTER TABLE data_quality_profiler_service.dataset_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY dp_tenant_read ON data_quality_profiler_service.dataset_profiles
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dp_tenant_write ON data_quality_profiler_service.dataset_profiles
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- column_profiles: tenant-scoped
ALTER TABLE data_quality_profiler_service.column_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY cp_tenant_read ON data_quality_profiler_service.column_profiles
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cp_tenant_write ON data_quality_profiler_service.column_profiles
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- quality_assessments: tenant-scoped
ALTER TABLE data_quality_profiler_service.quality_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY qa_tenant_read ON data_quality_profiler_service.quality_assessments
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qa_tenant_write ON data_quality_profiler_service.quality_assessments
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- quality_dimensions: tenant-scoped
ALTER TABLE data_quality_profiler_service.quality_dimensions ENABLE ROW LEVEL SECURITY;
CREATE POLICY qd_tenant_read ON data_quality_profiler_service.quality_dimensions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qd_tenant_write ON data_quality_profiler_service.quality_dimensions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- quality_rules: tenant-scoped
ALTER TABLE data_quality_profiler_service.quality_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY qr_tenant_read ON data_quality_profiler_service.quality_rules
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qr_tenant_write ON data_quality_profiler_service.quality_rules
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- rule_evaluations: tenant-scoped
ALTER TABLE data_quality_profiler_service.rule_evaluations ENABLE ROW LEVEL SECURITY;
CREATE POLICY re_tenant_read ON data_quality_profiler_service.rule_evaluations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY re_tenant_write ON data_quality_profiler_service.rule_evaluations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- quality_issues: tenant-scoped
ALTER TABLE data_quality_profiler_service.quality_issues ENABLE ROW LEVEL SECURITY;
CREATE POLICY qi_tenant_read ON data_quality_profiler_service.quality_issues
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qi_tenant_write ON data_quality_profiler_service.quality_issues
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- anomaly_detections: tenant-scoped
ALTER TABLE data_quality_profiler_service.anomaly_detections ENABLE ROW LEVEL SECURITY;
CREATE POLICY ad_tenant_read ON data_quality_profiler_service.anomaly_detections
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ad_tenant_write ON data_quality_profiler_service.anomaly_detections
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- quality_gates: tenant-scoped
ALTER TABLE data_quality_profiler_service.quality_gates ENABLE ROW LEVEL SECURITY;
CREATE POLICY qg_tenant_read ON data_quality_profiler_service.quality_gates
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qg_tenant_write ON data_quality_profiler_service.quality_gates
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- quality_trends: tenant-scoped
ALTER TABLE data_quality_profiler_service.quality_trends ENABLE ROW LEVEL SECURITY;
CREATE POLICY qt_tenant_read ON data_quality_profiler_service.quality_trends
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qt_tenant_write ON data_quality_profiler_service.quality_trends
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- quality_events: open (hypertable)
ALTER TABLE data_quality_profiler_service.quality_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY qe_tenant_read ON data_quality_profiler_service.quality_events
    FOR SELECT USING (TRUE);
CREATE POLICY qe_tenant_write ON data_quality_profiler_service.quality_events
    FOR ALL USING (TRUE);

-- profile_events: open (hypertable)
ALTER TABLE data_quality_profiler_service.profile_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_tenant_read ON data_quality_profiler_service.profile_events
    FOR SELECT USING (TRUE);
CREATE POLICY pe_tenant_write ON data_quality_profiler_service.profile_events
    FOR ALL USING (TRUE);

-- anomaly_events: open (hypertable)
ALTER TABLE data_quality_profiler_service.anomaly_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ane_tenant_read ON data_quality_profiler_service.anomaly_events
    FOR SELECT USING (TRUE);
CREATE POLICY ane_tenant_write ON data_quality_profiler_service.anomaly_events
    FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA data_quality_profiler_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA data_quality_profiler_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA data_quality_profiler_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON data_quality_profiler_service.quality_hourly_stats TO greenlang_app;
GRANT SELECT ON data_quality_profiler_service.profile_hourly_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA data_quality_profiler_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA data_quality_profiler_service TO greenlang_readonly;
GRANT SELECT ON data_quality_profiler_service.quality_hourly_stats TO greenlang_readonly;
GRANT SELECT ON data_quality_profiler_service.profile_hourly_stats TO greenlang_readonly;

-- Admin role
GRANT ALL ON SCHEMA data_quality_profiler_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA data_quality_profiler_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA data_quality_profiler_service TO greenlang_admin;

-- Add data quality profiler service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'data_quality_profiler:profiles:read', 'data_quality_profiler', 'profiles_read', 'View dataset profiles and column-level statistics'),
    (gen_random_uuid(), 'data_quality_profiler:profiles:write', 'data_quality_profiler', 'profiles_write', 'Create and manage dataset profiles and column profiling'),
    (gen_random_uuid(), 'data_quality_profiler:assessments:read', 'data_quality_profiler', 'assessments_read', 'View quality assessments and dimension scores'),
    (gen_random_uuid(), 'data_quality_profiler:assessments:write', 'data_quality_profiler', 'assessments_write', 'Create and manage quality assessments'),
    (gen_random_uuid(), 'data_quality_profiler:rules:read', 'data_quality_profiler', 'rules_read', 'View quality rules and evaluation results'),
    (gen_random_uuid(), 'data_quality_profiler:rules:write', 'data_quality_profiler', 'rules_write', 'Create and manage quality rules and evaluations'),
    (gen_random_uuid(), 'data_quality_profiler:issues:read', 'data_quality_profiler', 'issues_read', 'View detected quality issues and suggested fixes'),
    (gen_random_uuid(), 'data_quality_profiler:issues:write', 'data_quality_profiler', 'issues_write', 'Create and manage quality issues'),
    (gen_random_uuid(), 'data_quality_profiler:anomalies:read', 'data_quality_profiler', 'anomalies_read', 'View anomaly detection results and statistical analysis'),
    (gen_random_uuid(), 'data_quality_profiler:anomalies:write', 'data_quality_profiler', 'anomalies_write', 'Create and manage anomaly detections'),
    (gen_random_uuid(), 'data_quality_profiler:gates:read', 'data_quality_profiler', 'gates_read', 'View quality gate definitions and outcomes'),
    (gen_random_uuid(), 'data_quality_profiler:gates:write', 'data_quality_profiler', 'gates_write', 'Create and manage quality gates'),
    (gen_random_uuid(), 'data_quality_profiler:trends:read', 'data_quality_profiler', 'trends_read', 'View quality trend analysis and historical tracking'),
    (gen_random_uuid(), 'data_quality_profiler:admin', 'data_quality_profiler', 'admin', 'Data quality profiler service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep quality event records for 90 days
SELECT add_retention_policy('data_quality_profiler_service.quality_events', INTERVAL '90 days');

-- Keep profile event records for 90 days
SELECT add_retention_policy('data_quality_profiler_service.profile_events', INTERVAL '90 days');

-- Keep anomaly event records for 90 days
SELECT add_retention_policy('data_quality_profiler_service.anomaly_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on quality_events after 7 days
ALTER TABLE data_quality_profiler_service.quality_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('data_quality_profiler_service.quality_events', INTERVAL '7 days');

-- Enable compression on profile_events after 7 days
ALTER TABLE data_quality_profiler_service.profile_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('data_quality_profiler_service.profile_events', INTERVAL '7 days');

-- Enable compression on anomaly_events after 7 days
ALTER TABLE data_quality_profiler_service.anomaly_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('data_quality_profiler_service.anomaly_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Data Quality Profiler Agent (GL-DATA-X-013)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-013', 'Data Quality Profiler',
 'Comprehensive data quality profiling and assessment engine for GreenLang Climate OS. Profiles datasets with row/column statistics, data type detection, null analysis, cardinality, value distributions, and percentile calculations. Assesses quality across 6 dimensions (completeness, validity, consistency, timeliness, uniqueness, accuracy) with weighted scoring. Supports custom quality rules with configurable operators and thresholds. Detects statistical anomalies using z-score, IQR, isolation forest, MAD, and percentile methods. Enforces quality gates with pass/fail outcomes. Tracks historical quality trends with direction and change percentage. SHA-256 provenance chains for zero-hallucination audit trail.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/data-quality-profiler', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Data Quality Profiler
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-013', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/data-quality-profiler-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"data-quality", "profiling", "assessment", "anomaly-detection", "quality-gate", "completeness", "validity", "consistency"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "healthcare", "agriculture"}',
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Data Quality Profiler
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-X-013', '1.0.0', 'dataset_profiling', 'analysis',
 'Profile datasets with comprehensive statistics including row/column counts, data type detection, null analysis, cardinality computation, min/max/mean/median/stddev values, percentile distributions, most common values, and pattern detection with schema hash for drift detection',
 '{"dataset", "columns", "config"}', '{"profile_id", "column_profiles", "schema_hash", "memory_estimate"}',
 '{"supported_types": ["string", "integer", "float", "boolean", "date", "datetime", "decimal", "categorical"], "percentiles": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], "max_unique_values": 10000, "pattern_detection": true, "schema_hashing": true}'::jsonb),

('GL-DATA-X-013', '1.0.0', 'quality_assessment', 'assessment',
 'Assess dataset quality across 6 dimensions (completeness, validity, consistency, timeliness, uniqueness, accuracy) with configurable weights, producing an overall quality score (0-1), quality level classification (excellent/good/acceptable/poor/critical), and per-dimension breakdowns',
 '{"dataset", "dimensions", "weights"}', '{"assessment_id", "overall_score", "quality_level", "dimension_scores", "total_issues"}',
 '{"dimensions": ["completeness", "validity", "consistency", "timeliness", "uniqueness", "accuracy"], "default_weights": {"completeness": 0.25, "validity": 0.20, "consistency": 0.20, "timeliness": 0.10, "uniqueness": 0.15, "accuracy": 0.10}, "quality_levels": {"excellent": 0.95, "good": 0.80, "acceptable": 0.60, "poor": 0.40, "critical": 0.0}}'::jsonb),

('GL-DATA-X-013', '1.0.0', 'quality_rules_engine', 'validation',
 'Define and evaluate custom quality rules against datasets with configurable operators (eq, ne, gt, gte, lt, lte, between, in, matches, etc.), thresholds, and priority ordering. Supports null_check, range_check, regex_check, uniqueness_check, freshness_check, type_check, referential_check, and custom rule types',
 '{"dataset", "rules", "config"}', '{"evaluations", "passed_count", "failed_count", "total_rules"}',
 '{"rule_types": ["null_check", "range_check", "regex_check", "uniqueness_check", "freshness_check", "custom", "type_check", "referential_check", "completeness_check", "consistency_check", "format_check"], "operators": ["eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "not_in", "matches", "not_matches", "is_null", "is_not_null", "contains", "starts_with", "ends_with"], "batch_evaluation": true}'::jsonb),

('GL-DATA-X-013', '1.0.0', 'anomaly_detection', 'analysis',
 'Detect statistical anomalies in dataset columns using multiple methods (z-score, IQR, isolation forest, MAD, percentile, Grubbs, DBSCAN) with configurable sensitivity, severity classification, expected range computation, and affected row identification',
 '{"dataset", "columns", "methods", "config"}', '{"anomalies", "total_detected", "severity_breakdown"}',
 '{"methods": ["z_score", "iqr", "isolation_forest", "mad", "percentile", "grubbs", "dbscan"], "default_z_threshold": 3.0, "default_iqr_multiplier": 1.5, "severity_levels": ["critical", "high", "medium", "low"], "max_row_indices": 1000}'::jsonb),

('GL-DATA-X-013', '1.0.0', 'quality_gate_enforcement', 'governance',
 'Define and enforce quality gates with configurable conditions, threshold scores, and pass/fail/warning outcomes. Gates evaluate dimension scores against thresholds and produce deterministic outcomes for pipeline orchestration and data governance',
 '{"dataset", "gate_config", "assessment_id"}', '{"gate_id", "outcome", "dimension_scores", "conditions_met"}',
 '{"outcomes": ["pass", "fail", "pending", "warning", "skipped"], "default_threshold": 0.70, "condition_types": ["min_score", "max_issues", "dimension_threshold", "rule_pass_rate"], "block_on_fail": true}'::jsonb),

('GL-DATA-X-013', '1.0.0', 'issue_tracking', 'governance',
 'Track and manage detected quality issues with severity classification (critical/error/warning/info), dimension attribution, column and row identification, detailed descriptions, and suggested fixes for remediation guidance',
 '{"assessment_id", "issues"}', '{"issue_ids", "total_issues", "severity_breakdown"}',
 '{"severity_levels": ["critical", "error", "warning", "info"], "auto_suggest_fixes": true, "max_issues_per_assessment": 10000, "deduplication": true}'::jsonb),

('GL-DATA-X-013', '1.0.0', 'trend_analysis', 'reporting',
 'Track historical quality trends per dataset across configurable periods (hourly, daily, weekly, monthly, quarterly, yearly) with direction detection (improving, degrading, stable), percentage change calculation, and per-dimension score tracking for quality monitoring dashboards',
 '{"dataset_name", "period", "lookback"}', '{"trends", "direction", "change_pct", "dimension_trends"}',
 '{"periods": ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"], "directions": ["improving", "degrading", "stable", "unknown"], "lookback_default": 30, "export_formats": ["json", "csv"]}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Data Quality Profiler
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Data Quality Profiler depends on Schema Compiler for input/output validation
('GL-DATA-X-013', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Dataset profiles, quality assessments, and rule definitions are validated against JSON Schema definitions'),

-- Data Quality Profiler depends on Registry for agent discovery
('GL-DATA-X-013', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for quality profiling pipeline orchestration'),

-- Data Quality Profiler depends on Access Guard for policy enforcement
('GL-DATA-X-013', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for dataset profiles and quality assessments'),

-- Data Quality Profiler depends on Observability Agent for metrics
('GL-DATA-X-013', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Profiling metrics, quality assessment statistics, and anomaly detection telemetry are reported to observability'),

-- Data Quality Profiler optionally uses Citations for provenance tracking
('GL-DATA-X-013', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Quality assessment provenance and rule evaluation audit trails are registered with the citation service'),

-- Data Quality Profiler optionally uses Reproducibility for determinism
('GL-DATA-X-013', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Quality scores are verified for reproducibility across re-execution with identical inputs'),

-- Data Quality Profiler optionally uses QA Test Harness
('GL-DATA-X-013', 'GL-FOUND-X-009', '>=1.0.0', true,
 'Quality profiling results are validated through the QA Test Harness for zero-hallucination verification'),

-- Data Quality Profiler optionally integrates with Excel Normalizer
('GL-DATA-X-013', 'GL-DATA-X-002', '>=1.0.0', true,
 'Profiled datasets may originate from the Excel/CSV Normalizer for pre-normalized data quality assessment'),

-- Data Quality Profiler optionally integrates with ERP Connector
('GL-DATA-X-013', 'GL-DATA-X-003', '>=1.0.0', true,
 'ERP-sourced datasets are profiled for quality assessment before emission calculations'),

-- Data Quality Profiler optionally integrates with Data Gateway
('GL-DATA-X-013', 'GL-DATA-X-004', '>=1.0.0', true,
 'Data Gateway routes datasets through the quality profiler for pre-ingestion quality checks')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Data Quality Profiler
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-013', 'Data Quality Profiler',
 'Comprehensive data quality profiling, assessment, and governance engine. Profiles datasets with per-column statistics (null analysis, cardinality, distributions, pattern detection). Assesses quality across 6 dimensions (completeness, validity, consistency, timeliness, uniqueness, accuracy) with weighted scoring. Custom quality rules with 15+ operators and 11 rule types. Statistical anomaly detection (z-score, IQR, isolation forest, MAD, percentile, Grubbs, DBSCAN). Quality gate enforcement with pass/fail outcomes. Issue tracking with severity classification and suggested fixes. Historical trend analysis with direction detection. SHA-256 provenance chains for zero-hallucination audit trail.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA data_quality_profiler_service IS 'Data Quality Profiler for GreenLang Climate OS (AGENT-DATA-010) - dataset profiling, quality assessment across 6 dimensions, custom quality rules, anomaly detection, quality gate enforcement, issue tracking, and trend analysis with provenance chains';
COMMENT ON TABLE data_quality_profiler_service.dataset_profiles IS 'Core profiling results per dataset with row/column counts, memory estimate, schema hash for drift detection, processing status, and SHA-256 provenance hash';
COMMENT ON TABLE data_quality_profiler_service.column_profiles IS 'Per-column statistics including data type, null counts/percentages, unique counts, cardinality, min/max/mean/median/stddev, percentile distributions, most common values, and pattern detection';
COMMENT ON TABLE data_quality_profiler_service.quality_assessments IS 'Overall quality scores per dataset with quality level classification (excellent/good/acceptable/poor/critical), total issues, and processing status';
COMMENT ON TABLE data_quality_profiler_service.quality_dimensions IS 'Per-dimension quality scores (completeness/validity/consistency/timeliness/uniqueness/accuracy) with weight, weighted score, issues count, and detail breakdown';
COMMENT ON TABLE data_quality_profiler_service.quality_rules IS 'Custom quality rule definitions with rule type (null_check/range_check/regex_check/etc.), operator, threshold, parameters, priority, and activation state';
COMMENT ON TABLE data_quality_profiler_service.rule_evaluations IS 'Rule evaluation results capturing pass/fail status, actual value vs threshold, and evaluation message';
COMMENT ON TABLE data_quality_profiler_service.quality_issues IS 'Detected quality issues with severity (critical/error/warning/info), dimension attribution, column/row identification, description, and suggested fix';
COMMENT ON TABLE data_quality_profiler_service.anomaly_detections IS 'Anomaly detection results with detection method (z_score/iqr/isolation_forest/mad/percentile), observed value, expected range, z-score, severity, and affected row indices';
COMMENT ON TABLE data_quality_profiler_service.quality_gates IS 'Quality gate definitions and outcomes with conditions, threshold score, pass/fail/pending outcome, and dimension score snapshot';
COMMENT ON TABLE data_quality_profiler_service.quality_trends IS 'Historical quality tracking per dataset with period, overall score, per-dimension breakdown, trend direction (improving/degrading/stable), and change percentage';
COMMENT ON TABLE data_quality_profiler_service.quality_events IS 'TimescaleDB hypertable: quality assessment event time-series with event type, dataset name, dimension, score, and details';
COMMENT ON TABLE data_quality_profiler_service.profile_events IS 'TimescaleDB hypertable: profile lifecycle event time-series with event type, profile ID, dataset name, column/row counts, and details';
COMMENT ON TABLE data_quality_profiler_service.anomaly_events IS 'TimescaleDB hypertable: anomaly detection event time-series with event type, dataset name, column, method, severity, value, and details';
COMMENT ON MATERIALIZED VIEW data_quality_profiler_service.quality_hourly_stats IS 'Continuous aggregate: hourly quality assessment statistics by event type with total events, unique datasets, avg/min/max score, and per-dimension event counts';
COMMENT ON MATERIALIZED VIEW data_quality_profiler_service.profile_hourly_stats IS 'Continuous aggregate: hourly profile statistics by event type with total events, unique datasets/profiles, avg/max row and column counts, and total rows profiled';

COMMENT ON COLUMN data_quality_profiler_service.dataset_profiles.profile_id IS 'Unique profile identifier for cross-referencing with column_profiles';
COMMENT ON COLUMN data_quality_profiler_service.dataset_profiles.schema_hash IS 'SHA-256 hash of the dataset schema for drift detection between profiles';
COMMENT ON COLUMN data_quality_profiler_service.dataset_profiles.memory_estimate_bytes IS 'Estimated memory footprint of the dataset in bytes';
COMMENT ON COLUMN data_quality_profiler_service.dataset_profiles.status IS 'Profiling status: pending, running, completed, failed, cancelled';
COMMENT ON COLUMN data_quality_profiler_service.dataset_profiles.provenance_hash IS 'SHA-256 provenance hash for integrity verification and audit trail';
COMMENT ON COLUMN data_quality_profiler_service.column_profiles.null_pct IS 'Percentage of null values in the column (0-100)';
COMMENT ON COLUMN data_quality_profiler_service.column_profiles.cardinality IS 'Ratio of unique values to total values (0-1), measuring column selectivity';
COMMENT ON COLUMN data_quality_profiler_service.column_profiles.percentiles IS 'JSONB object of percentile values (e.g., p01, p05, p25, p50, p75, p95, p99)';
COMMENT ON COLUMN data_quality_profiler_service.column_profiles.most_common IS 'JSONB array of most frequently occurring values with their counts';
COMMENT ON COLUMN data_quality_profiler_service.column_profiles.pattern IS 'Detected value pattern (e.g., email, phone, date format, UUID)';
COMMENT ON COLUMN data_quality_profiler_service.quality_assessments.overall_score IS 'Weighted average quality score across all dimensions (0-1)';
COMMENT ON COLUMN data_quality_profiler_service.quality_assessments.quality_level IS 'Quality classification: excellent (>=0.95), good (>=0.80), acceptable (>=0.60), poor (>=0.40), critical (<0.40)';
COMMENT ON COLUMN data_quality_profiler_service.quality_dimensions.dimension IS 'Quality dimension: completeness, validity, consistency, timeliness, uniqueness, accuracy';
COMMENT ON COLUMN data_quality_profiler_service.quality_dimensions.weighted_score IS 'Dimension score multiplied by its weight for overall score calculation';
COMMENT ON COLUMN data_quality_profiler_service.quality_rules.rule_type IS 'Rule type: null_check, range_check, regex_check, uniqueness_check, freshness_check, custom, type_check, referential_check, completeness_check, consistency_check, format_check';
COMMENT ON COLUMN data_quality_profiler_service.quality_rules.operator IS 'Comparison operator: eq, ne, gt, gte, lt, lte, between, in, not_in, matches, not_matches, is_null, is_not_null, contains, starts_with, ends_with';
COMMENT ON COLUMN data_quality_profiler_service.quality_rules.priority IS 'Rule evaluation priority (lower numbers evaluated first)';
COMMENT ON COLUMN data_quality_profiler_service.rule_evaluations.passed IS 'Whether the rule evaluation passed (true) or failed (false)';
COMMENT ON COLUMN data_quality_profiler_service.quality_issues.severity IS 'Issue severity: critical, error, warning, info';
COMMENT ON COLUMN data_quality_profiler_service.quality_issues.suggested_fix IS 'Recommended remediation action for the detected issue';
COMMENT ON COLUMN data_quality_profiler_service.anomaly_detections.method IS 'Detection method: z_score, iqr, isolation_forest, mad, percentile, grubbs, dbscan';
COMMENT ON COLUMN data_quality_profiler_service.anomaly_detections.z_score IS 'Z-score of the anomalous value indicating standard deviations from the mean';
COMMENT ON COLUMN data_quality_profiler_service.anomaly_detections.row_indices IS 'JSONB array of row indices where the anomaly was detected';
COMMENT ON COLUMN data_quality_profiler_service.quality_gates.outcome IS 'Gate evaluation outcome: pass, fail, pending, warning, skipped';
COMMENT ON COLUMN data_quality_profiler_service.quality_gates.threshold IS 'Minimum overall score required for the gate to pass (0-1)';
COMMENT ON COLUMN data_quality_profiler_service.quality_gates.conditions IS 'JSONB array of gate conditions (e.g., min_score, max_issues, dimension_threshold)';
COMMENT ON COLUMN data_quality_profiler_service.quality_trends.direction IS 'Quality trend direction: improving, degrading, stable, unknown';
COMMENT ON COLUMN data_quality_profiler_service.quality_trends.change_pct IS 'Percentage change from the previous period overall score';
COMMENT ON COLUMN data_quality_profiler_service.quality_trends.period IS 'Reporting period: hourly, daily, weekly, monthly, quarterly, yearly';
COMMENT ON COLUMN data_quality_profiler_service.quality_events.event_type IS 'Quality event type: assessment_started, assessment_completed, assessment_failed, dimension_scored, rule_evaluated, issue_detected, gate_evaluated, trend_recorded, threshold_breach';
COMMENT ON COLUMN data_quality_profiler_service.profile_events.event_type IS 'Profile event type: profile_started, profile_completed, profile_failed, column_profiled, schema_detected, statistics_computed, distribution_analyzed, pattern_detected, drift_detected';
COMMENT ON COLUMN data_quality_profiler_service.anomaly_events.event_type IS 'Anomaly event type: detection_started, detection_completed, detection_failed, anomaly_found, anomaly_confirmed, anomaly_dismissed, threshold_exceeded, pattern_anomaly, distribution_shift';
