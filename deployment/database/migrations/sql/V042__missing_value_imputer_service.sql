-- =============================================================================
-- V042: Missing Value Imputer Service Schema
-- =============================================================================
-- Component: AGENT-DATA-012 (Missing Value Imputer)
-- Agent ID:  GL-DATA-X-015
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Missing Value Imputer (GL-DATA-X-015) with capabilities for
-- missingness pattern analysis (MCAR/MAR/MNAR via Little's test,
-- pattern matrix, correlation heatmaps), statistical imputation
-- (mean, median, mode, constant, forward/backward fill, interpolation),
-- ML-based imputation (KNN, random forest, iterative/MICE,
-- regression, EM algorithm), rule-based imputation (domain rules,
-- conditional logic, lookup tables, default values, cascading rules),
-- time-series imputation (linear/spline/seasonal interpolation,
-- last observation carried forward, next observation carried backward,
-- moving average, Kalman filter), validation (distribution
-- preservation via KS/chi-squared/Jensen-Shannon, correlation
-- preservation, outlier detection on imputed values, before/after
-- comparison), and pipeline orchestration (multi-strategy chaining,
-- column-level strategy assignment, dry-run mode, rollback support).
-- =============================================================================
-- Tables (10):
--   1. imputation_jobs           - Job tracking
--   2. imputation_analyses       - Missingness analysis results
--   3. imputation_strategies     - Strategy configurations
--   4. imputation_results        - Per-column imputation results
--   5. imputation_validations    - Post-imputation validation results
--   6. imputation_rules          - Rule-based imputation rules
--   7. imputation_rule_sets      - Groups of imputation rules
--   8. imputation_templates      - Reusable imputation templates
--   9. imputation_reports        - Imputation summary reports
--  10. imputation_audit_log      - Audit trail
--
-- Hypertables (3):
--  11. imputation_events         - Imputation event time-series (hypertable)
--  12. validation_events         - Validation event time-series (hypertable)
--  13. pipeline_events           - Pipeline event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. imputation_hourly_stats   - Hourly imputation event stats
--   2. validation_hourly_stats   - Hourly validation event stats
--
-- Also includes: 150+ indexes (B-tree, GIN, partial, composite),
-- 75+ CHECK constraints, 26 RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-015.
-- Previous: V041__duplicate_detector_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS missing_value_imputer_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION missing_value_imputer_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: missing_value_imputer_service.imputation_jobs
-- =============================================================================
-- Job tracking for imputation runs. Each job captures dataset IDs,
-- optional template reference, processing status and stage, record
-- counts at each pipeline stage (analyzed, imputed, validated),
-- missingness rate, error messages, configuration, provenance
-- hash, and timing information. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_ids TEXT[] NOT NULL,
    template_id UUID,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    stage VARCHAR(20) DEFAULT 'analyze',
    total_records INTEGER NOT NULL DEFAULT 0,
    total_columns INTEGER NOT NULL DEFAULT 0,
    missing_cells INTEGER NOT NULL DEFAULT 0,
    analyzed INTEGER NOT NULL DEFAULT 0,
    imputed INTEGER NOT NULL DEFAULT 0,
    validated INTEGER NOT NULL DEFAULT 0,
    missingness_rate NUMERIC(5,4) DEFAULT 0,
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
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

-- Stage constraint
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_stage
    CHECK (stage IN ('analyze', 'strategize', 'impute', 'validate', 'report', 'complete'));

-- Total records must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_total_records_non_negative
    CHECK (total_records >= 0);

-- Total columns must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_total_columns_non_negative
    CHECK (total_columns >= 0);

-- Missing cells must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_missing_cells_non_negative
    CHECK (missing_cells >= 0);

-- Analyzed must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_analyzed_non_negative
    CHECK (analyzed >= 0);

-- Imputed must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_imputed_non_negative
    CHECK (imputed >= 0);

-- Validated must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_validated_non_negative
    CHECK (validated >= 0);

-- Missingness rate must be between 0 and 1
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_missingness_rate_range
    CHECK (missingness_rate IS NULL OR (missingness_rate >= 0 AND missingness_rate <= 1));

-- Provenance hash must not be empty
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Created by must not be empty
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- Dataset IDs array must not be empty
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_dataset_ids_not_empty
    CHECK (array_length(dataset_ids, 1) > 0);

-- Completed_at must be after started_at if both are set
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT chk_ij_completed_after_started
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at);

-- updated_at trigger
CREATE TRIGGER trg_ij_updated_at
    BEFORE UPDATE ON missing_value_imputer_service.imputation_jobs
    FOR EACH ROW
    EXECUTE FUNCTION missing_value_imputer_service.set_updated_at();

-- =============================================================================
-- Table 2: missing_value_imputer_service.imputation_analyses
-- =============================================================================
-- Missingness analysis results. Each analysis captures per-column
-- missingness statistics, missingness mechanism classification
-- (MCAR/MAR/MNAR), pattern matrix, correlation analysis, Little's
-- test results, and recommendations for imputation strategy.
-- Linked to imputation_jobs. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    data_type VARCHAR(30) NOT NULL,
    total_values INTEGER NOT NULL,
    missing_count INTEGER NOT NULL,
    missing_rate NUMERIC(5,4) NOT NULL,
    mechanism VARCHAR(10) NOT NULL DEFAULT 'unknown',
    mechanism_confidence NUMERIC(5,4),
    littles_test_statistic NUMERIC(12,6),
    littles_test_p_value NUMERIC(12,6),
    pattern_correlations JSONB DEFAULT '{}'::jsonb,
    distribution_stats JSONB DEFAULT '{}'::jsonb,
    recommended_strategy VARCHAR(30),
    recommended_params JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to imputation_jobs
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT fk_ia_job_id
    FOREIGN KEY (job_id) REFERENCES missing_value_imputer_service.imputation_jobs(id)
    ON DELETE CASCADE;

-- Mechanism constraint
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_mechanism
    CHECK (mechanism IN ('mcar', 'mar', 'mnar', 'unknown'));

-- Data type constraint
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_data_type
    CHECK (data_type IN (
        'numeric', 'integer', 'float', 'categorical', 'boolean',
        'datetime', 'text', 'ordinal', 'binary', 'currency'
    ));

-- Missing rate must be between 0 and 1
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_missing_rate_range
    CHECK (missing_rate >= 0 AND missing_rate <= 1);

-- Mechanism confidence must be between 0 and 1 if specified
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_mechanism_confidence_range
    CHECK (mechanism_confidence IS NULL OR (mechanism_confidence >= 0 AND mechanism_confidence <= 1));

-- Total values must be positive
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_total_values_positive
    CHECK (total_values > 0);

-- Missing count must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_missing_count_non_negative
    CHECK (missing_count >= 0);

-- Column name must not be empty
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_analyses
    ADD CONSTRAINT chk_ia_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 3: missing_value_imputer_service.imputation_strategies
-- =============================================================================
-- Strategy configurations for imputation. Each strategy captures
-- the method type (statistical, ml, rule_based, time_series),
-- specific algorithm, target columns, parameters, priority order,
-- and activation state. Linked to imputation_jobs. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    strategy_type VARCHAR(20) NOT NULL,
    algorithm VARCHAR(30) NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    priority INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT true,
    fallback_strategy_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to imputation_jobs
ALTER TABLE missing_value_imputer_service.imputation_strategies
    ADD CONSTRAINT fk_is_job_id
    FOREIGN KEY (job_id) REFERENCES missing_value_imputer_service.imputation_jobs(id)
    ON DELETE CASCADE;

-- Self-referencing foreign key for fallback
ALTER TABLE missing_value_imputer_service.imputation_strategies
    ADD CONSTRAINT fk_is_fallback
    FOREIGN KEY (fallback_strategy_id) REFERENCES missing_value_imputer_service.imputation_strategies(id)
    ON DELETE SET NULL;

-- Strategy type constraint
ALTER TABLE missing_value_imputer_service.imputation_strategies
    ADD CONSTRAINT chk_is_strategy_type
    CHECK (strategy_type IN ('statistical', 'ml', 'rule_based', 'time_series'));

-- Algorithm constraint
ALTER TABLE missing_value_imputer_service.imputation_strategies
    ADD CONSTRAINT chk_is_algorithm
    CHECK (algorithm IN (
        'mean', 'median', 'mode', 'constant', 'forward_fill', 'backward_fill',
        'linear_interpolation', 'spline_interpolation', 'seasonal_interpolation',
        'knn', 'random_forest', 'iterative_mice', 'regression', 'em_algorithm',
        'domain_rule', 'conditional', 'lookup_table', 'default_value', 'cascading',
        'locf', 'nocb', 'moving_average', 'kalman_filter', 'hot_deck'
    ));

-- Priority must be positive
ALTER TABLE missing_value_imputer_service.imputation_strategies
    ADD CONSTRAINT chk_is_priority_positive
    CHECK (priority >= 1);

-- Column name must not be empty
ALTER TABLE missing_value_imputer_service.imputation_strategies
    ADD CONSTRAINT chk_is_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_strategies
    ADD CONSTRAINT chk_is_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_is_updated_at
    BEFORE UPDATE ON missing_value_imputer_service.imputation_strategies
    FOR EACH ROW
    EXECUTE FUNCTION missing_value_imputer_service.set_updated_at();

-- =============================================================================
-- Table 4: missing_value_imputer_service.imputation_results
-- =============================================================================
-- Per-column imputation results. Each result captures the column name,
-- strategy used, number of values imputed, original and imputed value
-- distributions, quality score (0-1), imputation duration, and
-- provenance hash. Linked to imputation_jobs. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    strategy_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    values_imputed INTEGER NOT NULL,
    original_distribution JSONB DEFAULT '{}'::jsonb,
    imputed_distribution JSONB DEFAULT '{}'::jsonb,
    quality_score NUMERIC(5,4),
    distribution_shift NUMERIC(5,4),
    correlation_change NUMERIC(5,4),
    imputation_time_ms INTEGER,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to imputation_jobs
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT fk_ir_job_id
    FOREIGN KEY (job_id) REFERENCES missing_value_imputer_service.imputation_jobs(id)
    ON DELETE CASCADE;

-- Foreign key to imputation_strategies
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT fk_ir_strategy_id
    FOREIGN KEY (strategy_id) REFERENCES missing_value_imputer_service.imputation_strategies(id)
    ON DELETE CASCADE;

-- Values imputed must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_values_imputed_non_negative
    CHECK (values_imputed >= 0);

-- Quality score must be between 0 and 1 if specified
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_quality_score_range
    CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1));

-- Distribution shift must be between 0 and 1 if specified
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_distribution_shift_range
    CHECK (distribution_shift IS NULL OR (distribution_shift >= 0 AND distribution_shift <= 1));

-- Correlation change must be between -1 and 1 if specified
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_correlation_change_range
    CHECK (correlation_change IS NULL OR (correlation_change >= -1 AND correlation_change <= 1));

-- Imputation time must be non-negative if specified
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_imputation_time_non_negative
    CHECK (imputation_time_ms IS NULL OR imputation_time_ms >= 0);

-- Provenance hash must not be empty
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Column name must not be empty
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_results
    ADD CONSTRAINT chk_ir_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 5: missing_value_imputer_service.imputation_validations
-- =============================================================================
-- Post-imputation validation results. Each validation captures the
-- test performed (KS test, chi-squared, Jensen-Shannon divergence,
-- correlation preservation, outlier detection), test statistic,
-- p-value, pass/fail status, and details. Linked to imputation_jobs
-- and imputation_results. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_validations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    result_id UUID NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    test_type VARCHAR(30) NOT NULL,
    test_statistic NUMERIC(12,6),
    p_value NUMERIC(12,6),
    threshold NUMERIC(12,6),
    passed BOOLEAN NOT NULL,
    severity VARCHAR(10) NOT NULL DEFAULT 'info',
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to imputation_jobs
ALTER TABLE missing_value_imputer_service.imputation_validations
    ADD CONSTRAINT fk_iv_job_id
    FOREIGN KEY (job_id) REFERENCES missing_value_imputer_service.imputation_jobs(id)
    ON DELETE CASCADE;

-- Foreign key to imputation_results
ALTER TABLE missing_value_imputer_service.imputation_validations
    ADD CONSTRAINT fk_iv_result_id
    FOREIGN KEY (result_id) REFERENCES missing_value_imputer_service.imputation_results(id)
    ON DELETE CASCADE;

-- Test type constraint
ALTER TABLE missing_value_imputer_service.imputation_validations
    ADD CONSTRAINT chk_iv_test_type
    CHECK (test_type IN (
        'ks_test', 'chi_squared', 'jensen_shannon', 'correlation_preservation',
        'outlier_detection', 'range_check', 'distribution_comparison',
        'completeness_check', 'consistency_check', 'statistical_summary'
    ));

-- Severity constraint
ALTER TABLE missing_value_imputer_service.imputation_validations
    ADD CONSTRAINT chk_iv_severity
    CHECK (severity IN ('info', 'warning', 'error', 'critical'));

-- Column name must not be empty
ALTER TABLE missing_value_imputer_service.imputation_validations
    ADD CONSTRAINT chk_iv_column_name_not_empty
    CHECK (LENGTH(TRIM(column_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_validations
    ADD CONSTRAINT chk_iv_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 6: missing_value_imputer_service.imputation_rules
-- =============================================================================
-- Rule-based imputation rules. Each rule captures a condition
-- expression, target column, imputed value or expression, priority,
-- activation state, and version. Rules are evaluated in priority
-- order and support conditional logic, lookup tables, domain rules,
-- default values, and cascading rules. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_set_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_column VARCHAR(255) NOT NULL,
    condition_expression TEXT NOT NULL,
    imputation_expression TEXT NOT NULL,
    rule_type VARCHAR(20) NOT NULL DEFAULT 'domain_rule',
    priority INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Rule type constraint
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_rule_type
    CHECK (rule_type IN ('domain_rule', 'conditional', 'lookup_table', 'default_value', 'cascading'));

-- Priority must be positive
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_priority_positive
    CHECK (priority >= 1);

-- Version must be positive
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_version_positive
    CHECK (version >= 1);

-- Name must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Target column must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_target_column_not_empty
    CHECK (LENGTH(TRIM(target_column)) > 0);

-- Condition expression must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_condition_not_empty
    CHECK (LENGTH(TRIM(condition_expression)) > 0);

-- Imputation expression must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_imputation_not_empty
    CHECK (LENGTH(TRIM(imputation_expression)) > 0);

-- Created by must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT chk_iru_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_iru_updated_at
    BEFORE UPDATE ON missing_value_imputer_service.imputation_rules
    FOR EACH ROW
    EXECUTE FUNCTION missing_value_imputer_service.set_updated_at();

-- =============================================================================
-- Table 7: missing_value_imputer_service.imputation_rule_sets
-- =============================================================================
-- Groups of imputation rules. Each rule set captures a name,
-- description, target data types, activation state, version, and
-- tenant scope. Rule sets are referenced by imputation_rules.

CREATE TABLE missing_value_imputer_service.imputation_rule_sets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_data_types TEXT[],
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key from imputation_rules to imputation_rule_sets
ALTER TABLE missing_value_imputer_service.imputation_rules
    ADD CONSTRAINT fk_iru_rule_set_id
    FOREIGN KEY (rule_set_id) REFERENCES missing_value_imputer_service.imputation_rule_sets(id)
    ON DELETE CASCADE;

-- Version must be positive
ALTER TABLE missing_value_imputer_service.imputation_rule_sets
    ADD CONSTRAINT chk_irs_version_positive
    CHECK (version >= 1);

-- Name must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rule_sets
    ADD CONSTRAINT chk_irs_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Created by must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rule_sets
    ADD CONSTRAINT chk_irs_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_rule_sets
    ADD CONSTRAINT chk_irs_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_irs_updated_at
    BEFORE UPDATE ON missing_value_imputer_service.imputation_rule_sets
    FOR EACH ROW
    EXECUTE FUNCTION missing_value_imputer_service.set_updated_at();

-- =============================================================================
-- Table 8: missing_value_imputer_service.imputation_templates
-- =============================================================================
-- Reusable imputation templates. Each template captures a name,
-- description, column-to-strategy mappings, default parameters,
-- validation configuration, activation state, and version.
-- Templates can be referenced by imputation_jobs. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    column_strategies JSONB NOT NULL,
    default_strategy VARCHAR(30) NOT NULL DEFAULT 'mean',
    default_params JSONB NOT NULL DEFAULT '{}'::jsonb,
    validation_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key from imputation_jobs to imputation_templates
ALTER TABLE missing_value_imputer_service.imputation_jobs
    ADD CONSTRAINT fk_ij_template_id
    FOREIGN KEY (template_id) REFERENCES missing_value_imputer_service.imputation_templates(id)
    ON DELETE SET NULL;

-- Default strategy constraint
ALTER TABLE missing_value_imputer_service.imputation_templates
    ADD CONSTRAINT chk_it_default_strategy
    CHECK (default_strategy IN (
        'mean', 'median', 'mode', 'constant', 'forward_fill', 'backward_fill',
        'linear_interpolation', 'knn', 'random_forest', 'iterative_mice',
        'domain_rule', 'locf', 'nocb', 'moving_average', 'hot_deck'
    ));

-- Version must be positive
ALTER TABLE missing_value_imputer_service.imputation_templates
    ADD CONSTRAINT chk_it_version_positive
    CHECK (version >= 1);

-- Name must not be empty
ALTER TABLE missing_value_imputer_service.imputation_templates
    ADD CONSTRAINT chk_it_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Column strategies must be a non-empty object
ALTER TABLE missing_value_imputer_service.imputation_templates
    ADD CONSTRAINT chk_it_column_strategies_not_empty
    CHECK (column_strategies IS NOT NULL AND column_strategies::text != '{}' AND column_strategies::text != 'null');

-- Created by must not be empty
ALTER TABLE missing_value_imputer_service.imputation_templates
    ADD CONSTRAINT chk_it_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_templates
    ADD CONSTRAINT chk_it_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_it_updated_at
    BEFORE UPDATE ON missing_value_imputer_service.imputation_templates
    FOR EACH ROW
    EXECUTE FUNCTION missing_value_imputer_service.set_updated_at();

-- =============================================================================
-- Table 9: missing_value_imputer_service.imputation_reports
-- =============================================================================
-- Imputation summary reports. Each report captures the overall
-- missingness rate before/after, per-column summaries, validation
-- summary, strategy effectiveness metrics, quality grade, and
-- provenance hash. Linked to imputation_jobs. Tenant-scoped.

CREATE TABLE missing_value_imputer_service.imputation_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    report_type VARCHAR(20) NOT NULL DEFAULT 'summary',
    missingness_before NUMERIC(5,4) NOT NULL,
    missingness_after NUMERIC(5,4) NOT NULL,
    columns_imputed INTEGER NOT NULL,
    total_values_imputed INTEGER NOT NULL,
    validation_pass_rate NUMERIC(5,4),
    overall_quality_score NUMERIC(5,4),
    quality_grade VARCHAR(5),
    column_summaries JSONB NOT NULL DEFAULT '[]'::jsonb,
    strategy_effectiveness JSONB NOT NULL DEFAULT '{}'::jsonb,
    recommendations JSONB DEFAULT '[]'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to imputation_jobs
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT fk_irp_job_id
    FOREIGN KEY (job_id) REFERENCES missing_value_imputer_service.imputation_jobs(id)
    ON DELETE CASCADE;

-- Report type constraint
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_report_type
    CHECK (report_type IN ('summary', 'detailed', 'validation', 'comparison', 'audit'));

-- Quality grade constraint
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_quality_grade
    CHECK (quality_grade IS NULL OR quality_grade IN ('A', 'B', 'C', 'D', 'F'));

-- Missingness before must be between 0 and 1
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_missingness_before_range
    CHECK (missingness_before >= 0 AND missingness_before <= 1);

-- Missingness after must be between 0 and 1
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_missingness_after_range
    CHECK (missingness_after >= 0 AND missingness_after <= 1);

-- Missingness after must be <= missingness before
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_missingness_reduced
    CHECK (missingness_after <= missingness_before);

-- Columns imputed must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_columns_imputed_non_negative
    CHECK (columns_imputed >= 0);

-- Total values imputed must be non-negative
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_total_values_non_negative
    CHECK (total_values_imputed >= 0);

-- Validation pass rate must be between 0 and 1 if specified
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_validation_pass_rate_range
    CHECK (validation_pass_rate IS NULL OR (validation_pass_rate >= 0 AND validation_pass_rate <= 1));

-- Overall quality score must be between 0 and 1 if specified
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_quality_score_range
    CHECK (overall_quality_score IS NULL OR (overall_quality_score >= 0 AND overall_quality_score <= 1));

-- Provenance hash must not be empty
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_reports
    ADD CONSTRAINT chk_irp_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: missing_value_imputer_service.imputation_audit_log
-- =============================================================================
-- Comprehensive audit trail for all imputation operations. Each entry
-- captures the action performed, entity type and ID, detail payload
-- (JSONB), provenance hash, performer, timestamp, and tenant scope.
-- Linked to imputation_jobs for job-scoped audit queries.

CREATE TABLE missing_value_imputer_service.imputation_audit_log (
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

-- Foreign key to imputation_jobs (optional, job_id may be null for system-level actions)
ALTER TABLE missing_value_imputer_service.imputation_audit_log
    ADD CONSTRAINT fk_ial_job_id
    FOREIGN KEY (job_id) REFERENCES missing_value_imputer_service.imputation_jobs(id)
    ON DELETE SET NULL;

-- Action constraint
ALTER TABLE missing_value_imputer_service.imputation_audit_log
    ADD CONSTRAINT chk_ial_action
    CHECK (action IN (
        'job_created', 'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'analysis_completed', 'strategy_selected', 'imputation_performed',
        'validation_completed', 'report_generated', 'template_created',
        'template_updated', 'template_deleted', 'template_activated', 'template_deactivated',
        'rule_created', 'rule_updated', 'rule_deleted', 'rule_activated', 'rule_deactivated',
        'config_changed', 'dry_run_completed', 'rollback_performed',
        'export_generated', 'import_completed'
    ));

-- Entity type constraint
ALTER TABLE missing_value_imputer_service.imputation_audit_log
    ADD CONSTRAINT chk_ial_entity_type
    CHECK (entity_type IN (
        'job', 'analysis', 'strategy', 'result', 'validation',
        'rule', 'rule_set', 'template', 'report', 'config'
    ));

-- Action must not be empty
ALTER TABLE missing_value_imputer_service.imputation_audit_log
    ADD CONSTRAINT chk_ial_action_not_empty
    CHECK (LENGTH(TRIM(action)) > 0);

-- Entity type must not be empty
ALTER TABLE missing_value_imputer_service.imputation_audit_log
    ADD CONSTRAINT chk_ial_entity_type_not_empty
    CHECK (LENGTH(TRIM(entity_type)) > 0);

-- Performed by must not be empty
ALTER TABLE missing_value_imputer_service.imputation_audit_log
    ADD CONSTRAINT chk_ial_performed_by_not_empty
    CHECK (LENGTH(TRIM(performed_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_audit_log
    ADD CONSTRAINT chk_ial_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: missing_value_imputer_service.imputation_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording imputation lifecycle events as a
-- time-series. Each event captures the job ID, event type, pipeline
-- stage, record count, duration in milliseconds, details payload,
-- provenance hash, and tenant. Partitioned by event_time for
-- time-series queries. Retained for 90 days with compression
-- after 7 days.

CREATE TABLE missing_value_imputer_service.imputation_events (
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
SELECT create_hypertable('missing_value_imputer_service.imputation_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE missing_value_imputer_service.imputation_events
    ADD CONSTRAINT chk_ie_event_type
    CHECK (event_type IN (
        'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'analysis_started', 'analysis_completed', 'analysis_failed',
        'strategy_started', 'strategy_completed', 'strategy_failed',
        'imputation_started', 'imputation_completed', 'imputation_failed',
        'validation_started', 'validation_completed', 'validation_failed',
        'reporting_started', 'reporting_completed', 'reporting_failed',
        'stage_transition', 'progress_update', 'quality_breach'
    ));

-- Stage constraint if specified
ALTER TABLE missing_value_imputer_service.imputation_events
    ADD CONSTRAINT chk_ie_stage
    CHECK (stage IS NULL OR stage IN (
        'analyze', 'strategize', 'impute', 'validate', 'report', 'complete'
    ));

-- Record count must be non-negative if specified
ALTER TABLE missing_value_imputer_service.imputation_events
    ADD CONSTRAINT chk_ie_record_count_non_negative
    CHECK (record_count IS NULL OR record_count >= 0);

-- Duration must be non-negative if specified
ALTER TABLE missing_value_imputer_service.imputation_events
    ADD CONSTRAINT chk_ie_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.imputation_events
    ADD CONSTRAINT chk_ie_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 12: missing_value_imputer_service.validation_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording validation events as a
-- time-series. Each event captures the job ID, column name, test
-- type, pass/fail status, test statistic, and tenant.
-- Partitioned by event_time for time-series queries. Retained for
-- 90 days with compression after 7 days.

CREATE TABLE missing_value_imputer_service.validation_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    column_name VARCHAR(255),
    test_type VARCHAR(30),
    passed BOOLEAN,
    test_statistic NUMERIC(12,6),
    p_value NUMERIC(12,6),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('missing_value_imputer_service.validation_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Test type constraint if specified
ALTER TABLE missing_value_imputer_service.validation_events
    ADD CONSTRAINT chk_ve_test_type
    CHECK (test_type IS NULL OR test_type IN (
        'ks_test', 'chi_squared', 'jensen_shannon', 'correlation_preservation',
        'outlier_detection', 'range_check', 'distribution_comparison',
        'completeness_check', 'consistency_check', 'statistical_summary'
    ));

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.validation_events
    ADD CONSTRAINT chk_ve_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 13: missing_value_imputer_service.pipeline_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording pipeline orchestration events as a
-- time-series. Each event captures the job ID, pipeline stage,
-- strategy type, column name, values processed, duration, provenance
-- hash, and tenant. Partitioned by event_time for time-series
-- queries. Retained for 90 days with compression after 7 days.

CREATE TABLE missing_value_imputer_service.pipeline_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    stage VARCHAR(20),
    strategy_type VARCHAR(20),
    column_name VARCHAR(255),
    values_processed INTEGER,
    duration_ms INTEGER,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('missing_value_imputer_service.pipeline_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Stage constraint if specified
ALTER TABLE missing_value_imputer_service.pipeline_events
    ADD CONSTRAINT chk_pe_stage
    CHECK (stage IS NULL OR stage IN (
        'analyze', 'strategize', 'impute', 'validate', 'report', 'complete'
    ));

-- Strategy type constraint if specified
ALTER TABLE missing_value_imputer_service.pipeline_events
    ADD CONSTRAINT chk_pe_strategy_type
    CHECK (strategy_type IS NULL OR strategy_type IN ('statistical', 'ml', 'rule_based', 'time_series'));

-- Values processed must be non-negative if specified
ALTER TABLE missing_value_imputer_service.pipeline_events
    ADD CONSTRAINT chk_pe_values_processed_non_negative
    CHECK (values_processed IS NULL OR values_processed >= 0);

-- Duration must be non-negative if specified
ALTER TABLE missing_value_imputer_service.pipeline_events
    ADD CONSTRAINT chk_pe_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE missing_value_imputer_service.pipeline_events
    ADD CONSTRAINT chk_pe_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Continuous Aggregate: missing_value_imputer_service.imputation_hourly_stats
-- =============================================================================
-- Precomputed hourly imputation statistics by event type for
-- dashboard queries, job monitoring, and throughput analysis.

CREATE MATERIALIZED VIEW missing_value_imputer_service.imputation_hourly_stats
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
    COUNT(*) FILTER (WHERE stage = 'analyze') AS analyze_events,
    COUNT(*) FILTER (WHERE stage = 'strategize') AS strategize_events,
    COUNT(*) FILTER (WHERE stage = 'impute') AS impute_events,
    COUNT(*) FILTER (WHERE stage = 'validate') AS validate_events,
    COUNT(*) FILTER (WHERE stage = 'report') AS report_events,
    COUNT(*) FILTER (WHERE stage = 'complete') AS complete_events
FROM missing_value_imputer_service.imputation_events
WHERE event_time IS NOT NULL
GROUP BY bucket, event_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('missing_value_imputer_service.imputation_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: missing_value_imputer_service.validation_hourly_stats
-- =============================================================================
-- Precomputed hourly validation statistics by test type for
-- dashboard queries, pass rate monitoring, and quality trending.

CREATE MATERIALIZED VIEW missing_value_imputer_service.validation_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    test_type,
    COUNT(*) AS total_validations,
    COUNT(DISTINCT job_id) AS unique_jobs,
    AVG(test_statistic) AS avg_test_statistic,
    MIN(test_statistic) AS min_test_statistic,
    MAX(test_statistic) AS max_test_statistic,
    COUNT(*) FILTER (WHERE passed = true) AS passed_count,
    COUNT(*) FILTER (WHERE passed = false) AS failed_count
FROM missing_value_imputer_service.validation_events
WHERE event_time IS NOT NULL
GROUP BY bucket, test_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('missing_value_imputer_service.validation_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- imputation_jobs indexes (18)
CREATE INDEX idx_ij_template_id ON missing_value_imputer_service.imputation_jobs(template_id);
CREATE INDEX idx_ij_status ON missing_value_imputer_service.imputation_jobs(status);
CREATE INDEX idx_ij_stage ON missing_value_imputer_service.imputation_jobs(stage);
CREATE INDEX idx_ij_tenant_id ON missing_value_imputer_service.imputation_jobs(tenant_id);
CREATE INDEX idx_ij_created_by ON missing_value_imputer_service.imputation_jobs(created_by);
CREATE INDEX idx_ij_provenance ON missing_value_imputer_service.imputation_jobs(provenance_hash);
CREATE INDEX idx_ij_created_at ON missing_value_imputer_service.imputation_jobs(created_at DESC);
CREATE INDEX idx_ij_updated_at ON missing_value_imputer_service.imputation_jobs(updated_at DESC);
CREATE INDEX idx_ij_started_at ON missing_value_imputer_service.imputation_jobs(started_at DESC);
CREATE INDEX idx_ij_completed_at ON missing_value_imputer_service.imputation_jobs(completed_at DESC);
CREATE INDEX idx_ij_missingness_rate ON missing_value_imputer_service.imputation_jobs(missingness_rate DESC);
CREATE INDEX idx_ij_total_records ON missing_value_imputer_service.imputation_jobs(total_records DESC);
CREATE INDEX idx_ij_dry_run ON missing_value_imputer_service.imputation_jobs(dry_run);
CREATE INDEX idx_ij_tenant_status ON missing_value_imputer_service.imputation_jobs(tenant_id, status);
CREATE INDEX idx_ij_tenant_stage ON missing_value_imputer_service.imputation_jobs(tenant_id, stage);
CREATE INDEX idx_ij_tenant_created ON missing_value_imputer_service.imputation_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_ij_status_stage ON missing_value_imputer_service.imputation_jobs(status, stage);
CREATE INDEX idx_ij_dataset_ids ON missing_value_imputer_service.imputation_jobs USING GIN (dataset_ids);
CREATE INDEX idx_ij_config ON missing_value_imputer_service.imputation_jobs USING GIN (config);

-- imputation_analyses indexes (16)
CREATE INDEX idx_ia_job_id ON missing_value_imputer_service.imputation_analyses(job_id);
CREATE INDEX idx_ia_column_name ON missing_value_imputer_service.imputation_analyses(column_name);
CREATE INDEX idx_ia_data_type ON missing_value_imputer_service.imputation_analyses(data_type);
CREATE INDEX idx_ia_mechanism ON missing_value_imputer_service.imputation_analyses(mechanism);
CREATE INDEX idx_ia_missing_rate ON missing_value_imputer_service.imputation_analyses(missing_rate DESC);
CREATE INDEX idx_ia_mechanism_confidence ON missing_value_imputer_service.imputation_analyses(mechanism_confidence DESC);
CREATE INDEX idx_ia_recommended_strategy ON missing_value_imputer_service.imputation_analyses(recommended_strategy);
CREATE INDEX idx_ia_tenant_id ON missing_value_imputer_service.imputation_analyses(tenant_id);
CREATE INDEX idx_ia_created_at ON missing_value_imputer_service.imputation_analyses(created_at DESC);
CREATE INDEX idx_ia_job_column ON missing_value_imputer_service.imputation_analyses(job_id, column_name);
CREATE INDEX idx_ia_job_mechanism ON missing_value_imputer_service.imputation_analyses(job_id, mechanism);
CREATE INDEX idx_ia_job_data_type ON missing_value_imputer_service.imputation_analyses(job_id, data_type);
CREATE INDEX idx_ia_tenant_job ON missing_value_imputer_service.imputation_analyses(tenant_id, job_id);
CREATE INDEX idx_ia_tenant_mechanism ON missing_value_imputer_service.imputation_analyses(tenant_id, mechanism);
CREATE INDEX idx_ia_pattern_correlations ON missing_value_imputer_service.imputation_analyses USING GIN (pattern_correlations);
CREATE INDEX idx_ia_distribution_stats ON missing_value_imputer_service.imputation_analyses USING GIN (distribution_stats);

-- imputation_strategies indexes (16)
CREATE INDEX idx_is_job_id ON missing_value_imputer_service.imputation_strategies(job_id);
CREATE INDEX idx_is_column_name ON missing_value_imputer_service.imputation_strategies(column_name);
CREATE INDEX idx_is_strategy_type ON missing_value_imputer_service.imputation_strategies(strategy_type);
CREATE INDEX idx_is_algorithm ON missing_value_imputer_service.imputation_strategies(algorithm);
CREATE INDEX idx_is_priority ON missing_value_imputer_service.imputation_strategies(priority);
CREATE INDEX idx_is_is_active ON missing_value_imputer_service.imputation_strategies(is_active);
CREATE INDEX idx_is_fallback ON missing_value_imputer_service.imputation_strategies(fallback_strategy_id);
CREATE INDEX idx_is_tenant_id ON missing_value_imputer_service.imputation_strategies(tenant_id);
CREATE INDEX idx_is_created_at ON missing_value_imputer_service.imputation_strategies(created_at DESC);
CREATE INDEX idx_is_job_column ON missing_value_imputer_service.imputation_strategies(job_id, column_name);
CREATE INDEX idx_is_job_type ON missing_value_imputer_service.imputation_strategies(job_id, strategy_type);
CREATE INDEX idx_is_job_algorithm ON missing_value_imputer_service.imputation_strategies(job_id, algorithm);
CREATE INDEX idx_is_job_priority ON missing_value_imputer_service.imputation_strategies(job_id, priority);
CREATE INDEX idx_is_tenant_job ON missing_value_imputer_service.imputation_strategies(tenant_id, job_id);
CREATE INDEX idx_is_tenant_type ON missing_value_imputer_service.imputation_strategies(tenant_id, strategy_type);
CREATE INDEX idx_is_parameters ON missing_value_imputer_service.imputation_strategies USING GIN (parameters);

-- imputation_results indexes (16)
CREATE INDEX idx_ir_job_id ON missing_value_imputer_service.imputation_results(job_id);
CREATE INDEX idx_ir_strategy_id ON missing_value_imputer_service.imputation_results(strategy_id);
CREATE INDEX idx_ir_column_name ON missing_value_imputer_service.imputation_results(column_name);
CREATE INDEX idx_ir_values_imputed ON missing_value_imputer_service.imputation_results(values_imputed DESC);
CREATE INDEX idx_ir_quality_score ON missing_value_imputer_service.imputation_results(quality_score DESC);
CREATE INDEX idx_ir_distribution_shift ON missing_value_imputer_service.imputation_results(distribution_shift);
CREATE INDEX idx_ir_provenance ON missing_value_imputer_service.imputation_results(provenance_hash);
CREATE INDEX idx_ir_tenant_id ON missing_value_imputer_service.imputation_results(tenant_id);
CREATE INDEX idx_ir_created_at ON missing_value_imputer_service.imputation_results(created_at DESC);
CREATE INDEX idx_ir_job_column ON missing_value_imputer_service.imputation_results(job_id, column_name);
CREATE INDEX idx_ir_job_strategy ON missing_value_imputer_service.imputation_results(job_id, strategy_id);
CREATE INDEX idx_ir_job_quality ON missing_value_imputer_service.imputation_results(job_id, quality_score DESC);
CREATE INDEX idx_ir_tenant_job ON missing_value_imputer_service.imputation_results(tenant_id, job_id);
CREATE INDEX idx_ir_tenant_quality ON missing_value_imputer_service.imputation_results(tenant_id, quality_score DESC);
CREATE INDEX idx_ir_original_distribution ON missing_value_imputer_service.imputation_results USING GIN (original_distribution);
CREATE INDEX idx_ir_imputed_distribution ON missing_value_imputer_service.imputation_results USING GIN (imputed_distribution);

-- imputation_validations indexes (16)
CREATE INDEX idx_iv_job_id ON missing_value_imputer_service.imputation_validations(job_id);
CREATE INDEX idx_iv_result_id ON missing_value_imputer_service.imputation_validations(result_id);
CREATE INDEX idx_iv_column_name ON missing_value_imputer_service.imputation_validations(column_name);
CREATE INDEX idx_iv_test_type ON missing_value_imputer_service.imputation_validations(test_type);
CREATE INDEX idx_iv_passed ON missing_value_imputer_service.imputation_validations(passed);
CREATE INDEX idx_iv_severity ON missing_value_imputer_service.imputation_validations(severity);
CREATE INDEX idx_iv_tenant_id ON missing_value_imputer_service.imputation_validations(tenant_id);
CREATE INDEX idx_iv_created_at ON missing_value_imputer_service.imputation_validations(created_at DESC);
CREATE INDEX idx_iv_job_test ON missing_value_imputer_service.imputation_validations(job_id, test_type);
CREATE INDEX idx_iv_job_passed ON missing_value_imputer_service.imputation_validations(job_id, passed);
CREATE INDEX idx_iv_job_column ON missing_value_imputer_service.imputation_validations(job_id, column_name);
CREATE INDEX idx_iv_job_severity ON missing_value_imputer_service.imputation_validations(job_id, severity);
CREATE INDEX idx_iv_result_test ON missing_value_imputer_service.imputation_validations(result_id, test_type);
CREATE INDEX idx_iv_tenant_job ON missing_value_imputer_service.imputation_validations(tenant_id, job_id);
CREATE INDEX idx_iv_tenant_passed ON missing_value_imputer_service.imputation_validations(tenant_id, passed);
CREATE INDEX idx_iv_details ON missing_value_imputer_service.imputation_validations USING GIN (details);

-- imputation_rules indexes (16)
CREATE INDEX idx_iru_rule_set_id ON missing_value_imputer_service.imputation_rules(rule_set_id);
CREATE INDEX idx_iru_name ON missing_value_imputer_service.imputation_rules(name);
CREATE INDEX idx_iru_target_column ON missing_value_imputer_service.imputation_rules(target_column);
CREATE INDEX idx_iru_rule_type ON missing_value_imputer_service.imputation_rules(rule_type);
CREATE INDEX idx_iru_priority ON missing_value_imputer_service.imputation_rules(priority);
CREATE INDEX idx_iru_is_active ON missing_value_imputer_service.imputation_rules(is_active);
CREATE INDEX idx_iru_version ON missing_value_imputer_service.imputation_rules(version);
CREATE INDEX idx_iru_tenant_id ON missing_value_imputer_service.imputation_rules(tenant_id);
CREATE INDEX idx_iru_created_by ON missing_value_imputer_service.imputation_rules(created_by);
CREATE INDEX idx_iru_created_at ON missing_value_imputer_service.imputation_rules(created_at DESC);
CREATE INDEX idx_iru_updated_at ON missing_value_imputer_service.imputation_rules(updated_at DESC);
CREATE INDEX idx_iru_set_column ON missing_value_imputer_service.imputation_rules(rule_set_id, target_column);
CREATE INDEX idx_iru_set_priority ON missing_value_imputer_service.imputation_rules(rule_set_id, priority);
CREATE INDEX idx_iru_set_active ON missing_value_imputer_service.imputation_rules(rule_set_id, is_active);
CREATE INDEX idx_iru_tenant_set ON missing_value_imputer_service.imputation_rules(tenant_id, rule_set_id);
CREATE INDEX idx_iru_tenant_type ON missing_value_imputer_service.imputation_rules(tenant_id, rule_type);

-- imputation_rule_sets indexes (12)
CREATE INDEX idx_irs_name ON missing_value_imputer_service.imputation_rule_sets(name);
CREATE INDEX idx_irs_is_active ON missing_value_imputer_service.imputation_rule_sets(is_active);
CREATE INDEX idx_irs_version ON missing_value_imputer_service.imputation_rule_sets(version);
CREATE INDEX idx_irs_tenant_id ON missing_value_imputer_service.imputation_rule_sets(tenant_id);
CREATE INDEX idx_irs_created_by ON missing_value_imputer_service.imputation_rule_sets(created_by);
CREATE INDEX idx_irs_created_at ON missing_value_imputer_service.imputation_rule_sets(created_at DESC);
CREATE INDEX idx_irs_updated_at ON missing_value_imputer_service.imputation_rule_sets(updated_at DESC);
CREATE INDEX idx_irs_tenant_active ON missing_value_imputer_service.imputation_rule_sets(tenant_id, is_active);
CREATE INDEX idx_irs_tenant_created ON missing_value_imputer_service.imputation_rule_sets(tenant_id, created_at DESC);
CREATE INDEX idx_irs_tenant_name ON missing_value_imputer_service.imputation_rule_sets(tenant_id, name);
CREATE INDEX idx_irs_target_data_types ON missing_value_imputer_service.imputation_rule_sets USING GIN (target_data_types);

-- imputation_templates indexes (14)
CREATE INDEX idx_it_name ON missing_value_imputer_service.imputation_templates(name);
CREATE INDEX idx_it_default_strategy ON missing_value_imputer_service.imputation_templates(default_strategy);
CREATE INDEX idx_it_is_active ON missing_value_imputer_service.imputation_templates(is_active);
CREATE INDEX idx_it_version ON missing_value_imputer_service.imputation_templates(version);
CREATE INDEX idx_it_tenant_id ON missing_value_imputer_service.imputation_templates(tenant_id);
CREATE INDEX idx_it_created_by ON missing_value_imputer_service.imputation_templates(created_by);
CREATE INDEX idx_it_created_at ON missing_value_imputer_service.imputation_templates(created_at DESC);
CREATE INDEX idx_it_updated_at ON missing_value_imputer_service.imputation_templates(updated_at DESC);
CREATE INDEX idx_it_tenant_active ON missing_value_imputer_service.imputation_templates(tenant_id, is_active);
CREATE INDEX idx_it_tenant_strategy ON missing_value_imputer_service.imputation_templates(tenant_id, default_strategy);
CREATE INDEX idx_it_tenant_created ON missing_value_imputer_service.imputation_templates(tenant_id, created_at DESC);
CREATE INDEX idx_it_column_strategies ON missing_value_imputer_service.imputation_templates USING GIN (column_strategies);
CREATE INDEX idx_it_default_params ON missing_value_imputer_service.imputation_templates USING GIN (default_params);
CREATE INDEX idx_it_validation_config ON missing_value_imputer_service.imputation_templates USING GIN (validation_config);

-- imputation_reports indexes (14)
CREATE INDEX idx_irp_job_id ON missing_value_imputer_service.imputation_reports(job_id);
CREATE INDEX idx_irp_report_type ON missing_value_imputer_service.imputation_reports(report_type);
CREATE INDEX idx_irp_quality_grade ON missing_value_imputer_service.imputation_reports(quality_grade);
CREATE INDEX idx_irp_overall_quality ON missing_value_imputer_service.imputation_reports(overall_quality_score DESC);
CREATE INDEX idx_irp_validation_pass ON missing_value_imputer_service.imputation_reports(validation_pass_rate DESC);
CREATE INDEX idx_irp_missingness_before ON missing_value_imputer_service.imputation_reports(missingness_before DESC);
CREATE INDEX idx_irp_missingness_after ON missing_value_imputer_service.imputation_reports(missingness_after);
CREATE INDEX idx_irp_provenance ON missing_value_imputer_service.imputation_reports(provenance_hash);
CREATE INDEX idx_irp_tenant_id ON missing_value_imputer_service.imputation_reports(tenant_id);
CREATE INDEX idx_irp_generated_at ON missing_value_imputer_service.imputation_reports(generated_at DESC);
CREATE INDEX idx_irp_job_type ON missing_value_imputer_service.imputation_reports(job_id, report_type);
CREATE INDEX idx_irp_tenant_job ON missing_value_imputer_service.imputation_reports(tenant_id, job_id);
CREATE INDEX idx_irp_tenant_grade ON missing_value_imputer_service.imputation_reports(tenant_id, quality_grade);
CREATE INDEX idx_irp_column_summaries ON missing_value_imputer_service.imputation_reports USING GIN (column_summaries);
CREATE INDEX idx_irp_strategy_effectiveness ON missing_value_imputer_service.imputation_reports USING GIN (strategy_effectiveness);
CREATE INDEX idx_irp_recommendations ON missing_value_imputer_service.imputation_reports USING GIN (recommendations);

-- imputation_audit_log indexes (16)
CREATE INDEX idx_ial_job_id ON missing_value_imputer_service.imputation_audit_log(job_id);
CREATE INDEX idx_ial_action ON missing_value_imputer_service.imputation_audit_log(action);
CREATE INDEX idx_ial_entity_type ON missing_value_imputer_service.imputation_audit_log(entity_type);
CREATE INDEX idx_ial_entity_id ON missing_value_imputer_service.imputation_audit_log(entity_id);
CREATE INDEX idx_ial_provenance ON missing_value_imputer_service.imputation_audit_log(provenance_hash);
CREATE INDEX idx_ial_performed_by ON missing_value_imputer_service.imputation_audit_log(performed_by);
CREATE INDEX idx_ial_performed_at ON missing_value_imputer_service.imputation_audit_log(performed_at DESC);
CREATE INDEX idx_ial_tenant_id ON missing_value_imputer_service.imputation_audit_log(tenant_id);
CREATE INDEX idx_ial_job_action ON missing_value_imputer_service.imputation_audit_log(job_id, action);
CREATE INDEX idx_ial_job_entity ON missing_value_imputer_service.imputation_audit_log(job_id, entity_type);
CREATE INDEX idx_ial_action_entity ON missing_value_imputer_service.imputation_audit_log(action, entity_type);
CREATE INDEX idx_ial_entity_type_id ON missing_value_imputer_service.imputation_audit_log(entity_type, entity_id);
CREATE INDEX idx_ial_tenant_job ON missing_value_imputer_service.imputation_audit_log(tenant_id, job_id);
CREATE INDEX idx_ial_tenant_action ON missing_value_imputer_service.imputation_audit_log(tenant_id, action);
CREATE INDEX idx_ial_tenant_performed ON missing_value_imputer_service.imputation_audit_log(tenant_id, performed_at DESC);
CREATE INDEX idx_ial_details ON missing_value_imputer_service.imputation_audit_log USING GIN (details);

-- imputation_events indexes (hypertable-aware) (8)
CREATE INDEX idx_ie_job_id ON missing_value_imputer_service.imputation_events(job_id, event_time DESC);
CREATE INDEX idx_ie_event_type ON missing_value_imputer_service.imputation_events(event_type, event_time DESC);
CREATE INDEX idx_ie_stage ON missing_value_imputer_service.imputation_events(stage, event_time DESC);
CREATE INDEX idx_ie_tenant_id ON missing_value_imputer_service.imputation_events(tenant_id, event_time DESC);
CREATE INDEX idx_ie_tenant_job ON missing_value_imputer_service.imputation_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_ie_tenant_type ON missing_value_imputer_service.imputation_events(tenant_id, event_type, event_time DESC);
CREATE INDEX idx_ie_provenance ON missing_value_imputer_service.imputation_events(provenance_hash, event_time DESC);
CREATE INDEX idx_ie_details ON missing_value_imputer_service.imputation_events USING GIN (details);

-- validation_events indexes (hypertable-aware) (8)
CREATE INDEX idx_ve_job_id ON missing_value_imputer_service.validation_events(job_id, event_time DESC);
CREATE INDEX idx_ve_column_name ON missing_value_imputer_service.validation_events(column_name, event_time DESC);
CREATE INDEX idx_ve_test_type ON missing_value_imputer_service.validation_events(test_type, event_time DESC);
CREATE INDEX idx_ve_passed ON missing_value_imputer_service.validation_events(passed, event_time DESC);
CREATE INDEX idx_ve_tenant_id ON missing_value_imputer_service.validation_events(tenant_id, event_time DESC);
CREATE INDEX idx_ve_tenant_job ON missing_value_imputer_service.validation_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_ve_tenant_test ON missing_value_imputer_service.validation_events(tenant_id, test_type, event_time DESC);
CREATE INDEX idx_ve_tenant_passed ON missing_value_imputer_service.validation_events(tenant_id, passed, event_time DESC);

-- pipeline_events indexes (hypertable-aware) (8)
CREATE INDEX idx_pe_job_id ON missing_value_imputer_service.pipeline_events(job_id, event_time DESC);
CREATE INDEX idx_pe_stage ON missing_value_imputer_service.pipeline_events(stage, event_time DESC);
CREATE INDEX idx_pe_strategy_type ON missing_value_imputer_service.pipeline_events(strategy_type, event_time DESC);
CREATE INDEX idx_pe_column_name ON missing_value_imputer_service.pipeline_events(column_name, event_time DESC);
CREATE INDEX idx_pe_tenant_id ON missing_value_imputer_service.pipeline_events(tenant_id, event_time DESC);
CREATE INDEX idx_pe_tenant_job ON missing_value_imputer_service.pipeline_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_pe_tenant_stage ON missing_value_imputer_service.pipeline_events(tenant_id, stage, event_time DESC);
CREATE INDEX idx_pe_provenance ON missing_value_imputer_service.pipeline_events(provenance_hash, event_time DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- imputation_jobs: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY ij_tenant_read ON missing_value_imputer_service.imputation_jobs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ij_tenant_write ON missing_value_imputer_service.imputation_jobs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_analyses: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_analyses ENABLE ROW LEVEL SECURITY;
CREATE POLICY ia_tenant_read ON missing_value_imputer_service.imputation_analyses
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ia_tenant_write ON missing_value_imputer_service.imputation_analyses
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_strategies: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_strategies ENABLE ROW LEVEL SECURITY;
CREATE POLICY is_tenant_read ON missing_value_imputer_service.imputation_strategies
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY is_tenant_write ON missing_value_imputer_service.imputation_strategies
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_results: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY ir_tenant_read ON missing_value_imputer_service.imputation_results
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ir_tenant_write ON missing_value_imputer_service.imputation_results
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_validations: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_validations ENABLE ROW LEVEL SECURITY;
CREATE POLICY iv_tenant_read ON missing_value_imputer_service.imputation_validations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY iv_tenant_write ON missing_value_imputer_service.imputation_validations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_rules: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY iru_tenant_read ON missing_value_imputer_service.imputation_rules
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY iru_tenant_write ON missing_value_imputer_service.imputation_rules
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_rule_sets: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_rule_sets ENABLE ROW LEVEL SECURITY;
CREATE POLICY irs_tenant_read ON missing_value_imputer_service.imputation_rule_sets
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY irs_tenant_write ON missing_value_imputer_service.imputation_rule_sets
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_templates: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY it_tenant_read ON missing_value_imputer_service.imputation_templates
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY it_tenant_write ON missing_value_imputer_service.imputation_templates
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_reports: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY irp_tenant_read ON missing_value_imputer_service.imputation_reports
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY irp_tenant_write ON missing_value_imputer_service.imputation_reports
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_audit_log: tenant-scoped
ALTER TABLE missing_value_imputer_service.imputation_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY ial_tenant_read ON missing_value_imputer_service.imputation_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ial_tenant_write ON missing_value_imputer_service.imputation_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- imputation_events: open (hypertable)
ALTER TABLE missing_value_imputer_service.imputation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ie_tenant_read ON missing_value_imputer_service.imputation_events
    FOR SELECT USING (TRUE);
CREATE POLICY ie_tenant_write ON missing_value_imputer_service.imputation_events
    FOR ALL USING (TRUE);

-- validation_events: open (hypertable)
ALTER TABLE missing_value_imputer_service.validation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ve_tenant_read ON missing_value_imputer_service.validation_events
    FOR SELECT USING (TRUE);
CREATE POLICY ve_tenant_write ON missing_value_imputer_service.validation_events
    FOR ALL USING (TRUE);

-- pipeline_events: open (hypertable)
ALTER TABLE missing_value_imputer_service.pipeline_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_tenant_read ON missing_value_imputer_service.pipeline_events
    FOR SELECT USING (TRUE);
CREATE POLICY pe_tenant_write ON missing_value_imputer_service.pipeline_events
    FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA missing_value_imputer_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA missing_value_imputer_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA missing_value_imputer_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON missing_value_imputer_service.imputation_hourly_stats TO greenlang_app;
GRANT SELECT ON missing_value_imputer_service.validation_hourly_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA missing_value_imputer_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA missing_value_imputer_service TO greenlang_readonly;
GRANT SELECT ON missing_value_imputer_service.imputation_hourly_stats TO greenlang_readonly;
GRANT SELECT ON missing_value_imputer_service.validation_hourly_stats TO greenlang_readonly;

-- Admin role
GRANT ALL ON SCHEMA missing_value_imputer_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA missing_value_imputer_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA missing_value_imputer_service TO greenlang_admin;

-- Add missing value imputer service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'missing_value_imputer:jobs:read', 'missing_value_imputer', 'jobs_read', 'View imputation jobs and their progress'),
    (gen_random_uuid(), 'missing_value_imputer:jobs:write', 'missing_value_imputer', 'jobs_write', 'Create, start, cancel, and manage imputation jobs'),
    (gen_random_uuid(), 'missing_value_imputer:analyses:read', 'missing_value_imputer', 'analyses_read', 'View missingness analyses and mechanism classifications'),
    (gen_random_uuid(), 'missing_value_imputer:analyses:write', 'missing_value_imputer', 'analyses_write', 'Run and manage missingness analyses'),
    (gen_random_uuid(), 'missing_value_imputer:strategies:read', 'missing_value_imputer', 'strategies_read', 'View imputation strategy configurations'),
    (gen_random_uuid(), 'missing_value_imputer:strategies:write', 'missing_value_imputer', 'strategies_write', 'Create and manage imputation strategies'),
    (gen_random_uuid(), 'missing_value_imputer:results:read', 'missing_value_imputer', 'results_read', 'View imputation results and quality scores'),
    (gen_random_uuid(), 'missing_value_imputer:results:write', 'missing_value_imputer', 'results_write', 'Create and manage imputation results'),
    (gen_random_uuid(), 'missing_value_imputer:validations:read', 'missing_value_imputer', 'validations_read', 'View post-imputation validation results'),
    (gen_random_uuid(), 'missing_value_imputer:validations:write', 'missing_value_imputer', 'validations_write', 'Run and manage post-imputation validations'),
    (gen_random_uuid(), 'missing_value_imputer:rules:read', 'missing_value_imputer', 'rules_read', 'View imputation rule definitions and rule sets'),
    (gen_random_uuid(), 'missing_value_imputer:rules:write', 'missing_value_imputer', 'rules_write', 'Create, update, and manage imputation rules'),
    (gen_random_uuid(), 'missing_value_imputer:templates:read', 'missing_value_imputer', 'templates_read', 'View imputation templates and configurations'),
    (gen_random_uuid(), 'missing_value_imputer:templates:write', 'missing_value_imputer', 'templates_write', 'Create, update, and manage imputation templates'),
    (gen_random_uuid(), 'missing_value_imputer:audit:read', 'missing_value_imputer', 'audit_read', 'View imputation audit log entries and provenance chains'),
    (gen_random_uuid(), 'missing_value_imputer:admin', 'missing_value_imputer', 'admin', 'Missing value imputer service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep imputation event records for 90 days
SELECT add_retention_policy('missing_value_imputer_service.imputation_events', INTERVAL '90 days');

-- Keep validation event records for 90 days
SELECT add_retention_policy('missing_value_imputer_service.validation_events', INTERVAL '90 days');

-- Keep pipeline event records for 90 days
SELECT add_retention_policy('missing_value_imputer_service.pipeline_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on imputation_events after 7 days
ALTER TABLE missing_value_imputer_service.imputation_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('missing_value_imputer_service.imputation_events', INTERVAL '7 days');

-- Enable compression on validation_events after 7 days
ALTER TABLE missing_value_imputer_service.validation_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('missing_value_imputer_service.validation_events', INTERVAL '7 days');

-- Enable compression on pipeline_events after 7 days
ALTER TABLE missing_value_imputer_service.pipeline_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('missing_value_imputer_service.pipeline_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Missing Value Imputer (GL-DATA-X-015)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-015', 'Missing Value Imputer',
 'Comprehensive missing value imputation engine for GreenLang Climate OS. Analyzes missingness patterns (MCAR/MAR/MNAR via Little''s test, pattern matrix, correlation heatmaps). Applies statistical imputation (mean, median, mode, constant, forward/backward fill, interpolation). Supports ML-based imputation (KNN, random forest, iterative/MICE, regression, EM algorithm). Provides rule-based imputation (domain rules, conditional logic, lookup tables, default values, cascading rules). Implements time-series imputation (LOCF, NOCB, moving average, Kalman filter, seasonal interpolation). Validates imputed data (KS test, chi-squared, Jensen-Shannon divergence, correlation preservation, outlier detection). Pipeline orchestration with multi-strategy chaining, column-level strategy assignment, dry-run mode, and rollback support. SHA-256 provenance chains for zero-hallucination audit trail.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/missing-value-imputer', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Missing Value Imputer
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-015', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/missing-value-imputer-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"imputation", "missing-values", "data-quality", "statistical", "machine-learning", "time-series", "validation", "pipeline"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "healthcare", "agriculture"}',
 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Missing Value Imputer
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-X-015', '1.0.0', 'missingness_analysis', 'analysis',
 'Analyze missingness patterns across dataset columns. Classifies missingness mechanism (MCAR/MAR/MNAR) using Little''s test and correlation analysis. Computes per-column missing rates, pattern matrices, and distribution statistics. Recommends optimal imputation strategies based on data type and mechanism',
 '{"dataset", "columns", "config"}', '{"analyses", "mechanism_classifications", "recommendations", "pattern_matrix"}',
 '{"mechanisms": ["mcar", "mar", "mnar"], "data_types": ["numeric", "integer", "float", "categorical", "boolean", "datetime", "text", "ordinal", "binary", "currency"], "littles_test": true, "correlation_analysis": true, "pattern_visualization": true}'::jsonb),

('GL-DATA-X-015', '1.0.0', 'statistical_imputation', 'processing',
 'Apply statistical imputation methods to fill missing values. Supports mean, median, mode for central tendency, constant value fill, forward/backward fill for ordered data, and linear/spline interpolation. Preserves original distribution characteristics where possible',
 '{"dataset", "column", "method", "config"}', '{"imputed_values", "quality_score", "distribution_stats"}',
 '{"methods": ["mean", "median", "mode", "constant", "forward_fill", "backward_fill", "linear_interpolation", "spline_interpolation"], "group_by_support": true, "weighted_methods": true, "outlier_handling": ["include", "exclude", "cap"]}'::jsonb),

('GL-DATA-X-015', '1.0.0', 'ml_imputation', 'processing',
 'Apply machine learning-based imputation methods for complex missing data patterns. Supports KNN imputation (distance-weighted), random forest imputation, iterative imputation (MICE), regression imputation, and EM algorithm. Handles multivariate dependencies and non-linear relationships',
 '{"dataset", "columns", "method", "config"}', '{"imputed_values", "model_metrics", "feature_importance", "quality_score"}',
 '{"methods": ["knn", "random_forest", "iterative_mice", "regression", "em_algorithm"], "knn_neighbors": [3, 5, 7, 10], "mice_iterations": [5, 10, 20], "forest_estimators": [50, 100, 200], "cross_validation": true, "feature_selection": true}'::jsonb),

('GL-DATA-X-015', '1.0.0', 'rule_based_imputation', 'processing',
 'Apply rule-based imputation using domain knowledge and business rules. Supports domain-specific rules, conditional logic (if-then-else), lookup table substitution, default value assignment, and cascading rule chains. Rules are evaluated in priority order with fallback support',
 '{"dataset", "column", "rules", "config"}', '{"imputed_values", "rules_applied", "coverage_stats"}',
 '{"rule_types": ["domain_rule", "conditional", "lookup_table", "default_value", "cascading"], "priority_ordering": true, "condition_operators": ["eq", "ne", "gt", "lt", "gte", "lte", "in", "not_in", "between", "regex", "is_null", "is_not_null"], "rule_versioning": true}'::jsonb),

('GL-DATA-X-015', '1.0.0', 'time_series_imputation', 'processing',
 'Apply time-series-specific imputation methods for temporal data with missing observations. Supports LOCF (last observation carried forward), NOCB (next observation carried backward), moving average, Kalman filter, linear/spline/seasonal interpolation. Respects temporal ordering and seasonal patterns',
 '{"dataset", "column", "time_column", "method", "config"}', '{"imputed_values", "temporal_stats", "quality_score"}',
 '{"methods": ["locf", "nocb", "moving_average", "kalman_filter", "linear_interpolation", "spline_interpolation", "seasonal_interpolation"], "window_sizes": [3, 5, 7, 14, 30], "seasonal_periods": [7, 12, 24, 52, 365], "gap_limit": true, "bidirectional": true}'::jsonb),

('GL-DATA-X-015', '1.0.0', 'imputation_validation', 'validation',
 'Validate imputed data quality using statistical tests and distribution comparison. Applies Kolmogorov-Smirnov test, chi-squared test, Jensen-Shannon divergence for distribution preservation. Checks correlation preservation, detects outliers in imputed values, and computes before/after comparison metrics',
 '{"original_data", "imputed_data", "config"}', '{"validation_results", "pass_rate", "quality_score", "recommendations"}',
 '{"tests": ["ks_test", "chi_squared", "jensen_shannon", "correlation_preservation", "outlier_detection", "range_check", "distribution_comparison", "completeness_check", "consistency_check"], "significance_level": 0.05, "outlier_methods": ["iqr", "z_score", "isolation_forest"], "auto_remediation": true}'::jsonb),

('GL-DATA-X-015', '1.0.0', 'pipeline_orchestration', 'orchestration',
 'Orchestrate multi-strategy imputation pipelines with column-level strategy assignment, priority-based execution, fallback strategies, dry-run mode, rollback support, and progress tracking. Chains multiple imputation methods per column with validation gates between stages',
 '{"dataset", "template_or_config", "mode"}', '{"pipeline_results", "report", "provenance_chain"}',
 '{"modes": ["full", "dry_run", "column_only", "validate_only"], "strategy_chaining": true, "fallback_support": true, "rollback_enabled": true, "checkpoint_interval": 1000, "max_pipeline_timeout_seconds": 3600, "parallel_columns": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Missing Value Imputer
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Missing Value Imputer depends on Schema Compiler for input/output validation
('GL-DATA-X-015', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Imputation templates, job configurations, and rule definitions are validated against JSON Schema definitions'),

-- Missing Value Imputer depends on Registry for agent discovery
('GL-DATA-X-015', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for imputation pipeline orchestration'),

-- Missing Value Imputer depends on Access Guard for policy enforcement
('GL-DATA-X-015', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for imputation jobs and results'),

-- Missing Value Imputer depends on Observability Agent for metrics
('GL-DATA-X-015', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Imputation metrics, validation pass rates, quality scores, and pipeline throughput are reported to observability'),

-- Missing Value Imputer optionally uses Citations for provenance tracking
('GL-DATA-X-015', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Imputation result provenance and validation audit trails are registered with the citation service'),

-- Missing Value Imputer optionally uses Reproducibility for determinism
('GL-DATA-X-015', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Imputation results are verified for reproducibility across re-execution with identical inputs and strategies'),

-- Missing Value Imputer optionally uses QA Test Harness
('GL-DATA-X-015', 'GL-FOUND-X-009', '>=1.0.0', true,
 'Imputation results are validated through the QA Test Harness for zero-hallucination verification'),

-- Missing Value Imputer optionally integrates with Excel Normalizer
('GL-DATA-X-015', 'GL-DATA-X-002', '>=1.0.0', true,
 'Normalized datasets from Excel/CSV processing have missing values imputed before downstream emission calculations'),

-- Missing Value Imputer optionally integrates with Data Quality Profiler
('GL-DATA-X-015', 'GL-DATA-X-013', '>=1.0.0', true,
 'Data quality profiling identifies missing value patterns that inform imputation strategy selection'),

-- Missing Value Imputer optionally integrates with Duplicate Detector
('GL-DATA-X-015', 'GL-DATA-X-014', '>=1.0.0', true,
 'Deduplicated datasets are passed to the imputer for missing value treatment after duplicate removal')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Missing Value Imputer
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-015', 'Missing Value Imputer',
 'Comprehensive missing value imputation engine. Analyzes missingness patterns (MCAR/MAR/MNAR via Little''s test). Applies statistical imputation (mean/median/mode/constant/forward-backward fill/interpolation). Supports ML imputation (KNN/random forest/MICE/regression/EM). Provides rule-based imputation (domain rules/conditional/lookup tables/defaults/cascading). Implements time-series imputation (LOCF/NOCB/moving average/Kalman filter/seasonal). Validates with KS test, chi-squared, Jensen-Shannon divergence, correlation preservation, outlier detection. Pipeline orchestration with multi-strategy chaining, column-level assignment, dry-run mode, and rollback. SHA-256 provenance chains for zero-hallucination audit trail.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA missing_value_imputer_service IS 'Missing Value Imputer for GreenLang Climate OS (AGENT-DATA-012) - missingness analysis, statistical/ML/rule-based/time-series imputation, post-imputation validation, pipeline orchestration, reusable templates, and comprehensive audit logging with provenance chains';
COMMENT ON TABLE missing_value_imputer_service.imputation_jobs IS 'Job tracking for imputation runs with dataset IDs, template reference, status/stage progression, per-stage record counts, missingness rate, dry-run flag, configuration, and SHA-256 provenance hash';
COMMENT ON TABLE missing_value_imputer_service.imputation_analyses IS 'Missingness analysis results with per-column statistics, mechanism classification (MCAR/MAR/MNAR), Little''s test results, pattern correlations, distribution stats, and recommended strategies';
COMMENT ON TABLE missing_value_imputer_service.imputation_strategies IS 'Strategy configurations with type (statistical/ml/rule_based/time_series), algorithm, parameters, priority, activation state, and fallback chain';
COMMENT ON TABLE missing_value_imputer_service.imputation_results IS 'Per-column imputation results with values imputed, original/imputed distributions, quality score, distribution shift, correlation change, and SHA-256 provenance hash';
COMMENT ON TABLE missing_value_imputer_service.imputation_validations IS 'Post-imputation validation results with test type (KS/chi-squared/Jensen-Shannon/correlation/outlier), test statistic, p-value, pass/fail status, and severity';
COMMENT ON TABLE missing_value_imputer_service.imputation_rules IS 'Rule-based imputation rules with condition expression, imputation expression, rule type (domain/conditional/lookup/default/cascading), priority, and version';
COMMENT ON TABLE missing_value_imputer_service.imputation_rule_sets IS 'Groups of imputation rules with name, target data types, activation state, version, and tenant scope';
COMMENT ON TABLE missing_value_imputer_service.imputation_templates IS 'Reusable imputation templates with column-to-strategy mappings, default strategy/params, validation config, activation state, and version';
COMMENT ON TABLE missing_value_imputer_service.imputation_reports IS 'Imputation summary reports with before/after missingness rates, per-column summaries, validation pass rate, quality grade, strategy effectiveness, and recommendations';
COMMENT ON TABLE missing_value_imputer_service.imputation_audit_log IS 'Comprehensive audit trail for all imputation operations with action, entity type/ID, details (JSONB), provenance hash, performer, and timestamp';
COMMENT ON TABLE missing_value_imputer_service.imputation_events IS 'TimescaleDB hypertable: imputation lifecycle event time-series with job ID, event type, pipeline stage, record count, duration, and details';
COMMENT ON TABLE missing_value_imputer_service.validation_events IS 'TimescaleDB hypertable: validation event time-series with job ID, column name, test type, pass/fail status, test statistic, and p-value';
COMMENT ON TABLE missing_value_imputer_service.pipeline_events IS 'TimescaleDB hypertable: pipeline orchestration event time-series with job ID, stage, strategy type, column name, values processed, and duration';
COMMENT ON MATERIALIZED VIEW missing_value_imputer_service.imputation_hourly_stats IS 'Continuous aggregate: hourly imputation event statistics by event type with total events, unique jobs, avg/sum record counts, avg/max/min duration, and per-stage event counts';
COMMENT ON MATERIALIZED VIEW missing_value_imputer_service.validation_hourly_stats IS 'Continuous aggregate: hourly validation statistics by test type with total validations, unique jobs, avg/min/max test statistic, and passed/failed counts';

COMMENT ON COLUMN missing_value_imputer_service.imputation_jobs.status IS 'Job status: pending, running, completed, failed, cancelled';
COMMENT ON COLUMN missing_value_imputer_service.imputation_jobs.stage IS 'Current pipeline stage: analyze, strategize, impute, validate, report, complete';
COMMENT ON COLUMN missing_value_imputer_service.imputation_jobs.missingness_rate IS 'Overall ratio of missing values to total cells (0-1), computed during analysis';
COMMENT ON COLUMN missing_value_imputer_service.imputation_jobs.dry_run IS 'If true, imputation is simulated without modifying the dataset';
COMMENT ON COLUMN missing_value_imputer_service.imputation_jobs.provenance_hash IS 'SHA-256 provenance hash for integrity verification and audit trail';
COMMENT ON COLUMN missing_value_imputer_service.imputation_analyses.mechanism IS 'Missingness mechanism: mcar (missing completely at random), mar (missing at random), mnar (missing not at random), unknown';
COMMENT ON COLUMN missing_value_imputer_service.imputation_analyses.mechanism_confidence IS 'Confidence score (0-1) for the mechanism classification';
COMMENT ON COLUMN missing_value_imputer_service.imputation_analyses.littles_test_statistic IS 'Little''s MCAR test chi-squared statistic';
COMMENT ON COLUMN missing_value_imputer_service.imputation_analyses.littles_test_p_value IS 'Little''s MCAR test p-value (p > 0.05 suggests MCAR)';
COMMENT ON COLUMN missing_value_imputer_service.imputation_analyses.recommended_strategy IS 'Strategy recommended based on data type and missingness mechanism';
COMMENT ON COLUMN missing_value_imputer_service.imputation_strategies.strategy_type IS 'Strategy category: statistical, ml, rule_based, time_series';
COMMENT ON COLUMN missing_value_imputer_service.imputation_strategies.algorithm IS 'Specific imputation algorithm within the strategy type';
COMMENT ON COLUMN missing_value_imputer_service.imputation_strategies.fallback_strategy_id IS 'Optional fallback strategy if the primary strategy fails or produces poor quality';
COMMENT ON COLUMN missing_value_imputer_service.imputation_results.quality_score IS 'Overall quality score (0-1) for the imputation result based on validation tests';
COMMENT ON COLUMN missing_value_imputer_service.imputation_results.distribution_shift IS 'Magnitude of distribution shift (0-1) between original and imputed distributions';
COMMENT ON COLUMN missing_value_imputer_service.imputation_results.correlation_change IS 'Change in correlation (-1 to 1) with other columns after imputation';
COMMENT ON COLUMN missing_value_imputer_service.imputation_results.provenance_hash IS 'SHA-256 provenance hash of the imputation result for audit trail verification';
COMMENT ON COLUMN missing_value_imputer_service.imputation_validations.test_type IS 'Validation test: ks_test, chi_squared, jensen_shannon, correlation_preservation, outlier_detection, range_check, distribution_comparison, completeness_check, consistency_check, statistical_summary';
COMMENT ON COLUMN missing_value_imputer_service.imputation_validations.severity IS 'Validation severity: info, warning, error, critical';
COMMENT ON COLUMN missing_value_imputer_service.imputation_rules.rule_type IS 'Rule type: domain_rule, conditional, lookup_table, default_value, cascading';
COMMENT ON COLUMN missing_value_imputer_service.imputation_rules.condition_expression IS 'Boolean expression evaluated to determine if this rule applies to a given record';
COMMENT ON COLUMN missing_value_imputer_service.imputation_rules.imputation_expression IS 'Expression or value used to impute the missing value when the condition is true';
COMMENT ON COLUMN missing_value_imputer_service.imputation_templates.column_strategies IS 'JSONB mapping of column names to strategy configurations for reusable imputation pipelines';
COMMENT ON COLUMN missing_value_imputer_service.imputation_templates.default_strategy IS 'Default imputation strategy for columns not explicitly mapped';
COMMENT ON COLUMN missing_value_imputer_service.imputation_reports.quality_grade IS 'Letter grade (A-F) for overall imputation quality based on validation pass rates and quality scores';
COMMENT ON COLUMN missing_value_imputer_service.imputation_reports.missingness_before IS 'Overall missingness rate before imputation (0-1)';
COMMENT ON COLUMN missing_value_imputer_service.imputation_reports.missingness_after IS 'Overall missingness rate after imputation (0-1)';
COMMENT ON COLUMN missing_value_imputer_service.imputation_audit_log.action IS 'Audit action: job_created, job_started, job_completed, job_failed, job_cancelled, analysis_completed, strategy_selected, imputation_performed, validation_completed, report_generated, template_created/updated/deleted, rule_created/updated/deleted, etc.';
COMMENT ON COLUMN missing_value_imputer_service.imputation_audit_log.entity_type IS 'Entity type: job, analysis, strategy, result, validation, rule, rule_set, template, report, config';
COMMENT ON COLUMN missing_value_imputer_service.imputation_audit_log.provenance_hash IS 'SHA-256 provenance hash linking the audit entry to a specific data state';
COMMENT ON COLUMN missing_value_imputer_service.imputation_events.event_type IS 'Imputation event type: job_started/completed/failed/cancelled, analysis_started/completed/failed, strategy_started/completed/failed, imputation_started/completed/failed, validation_started/completed/failed, reporting_started/completed/failed, stage_transition, progress_update, quality_breach';
COMMENT ON COLUMN missing_value_imputer_service.imputation_events.stage IS 'Pipeline stage: analyze, strategize, impute, validate, report, complete';
COMMENT ON COLUMN missing_value_imputer_service.validation_events.test_type IS 'Validation test type: ks_test, chi_squared, jensen_shannon, correlation_preservation, outlier_detection, etc.';
COMMENT ON COLUMN missing_value_imputer_service.pipeline_events.strategy_type IS 'Strategy type: statistical, ml, rule_based, time_series';
COMMENT ON COLUMN missing_value_imputer_service.pipeline_events.stage IS 'Pipeline stage: analyze, strategize, impute, validate, report, complete';
