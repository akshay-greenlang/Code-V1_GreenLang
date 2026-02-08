-- =============================================================================
-- GreenLang Climate OS - QA Test Harness Service Schema
-- =============================================================================
-- Migration: V029
-- Component: AGENT-FOUND-009 QA Test Harness
-- Description: Creates qa_test_harness_service schema with test_suites,
--              test_cases, test_runs (hypertable), test_assertions,
--              golden_files, performance_baselines, coverage_snapshots,
--              regression_baselines, qa_audit_log (hypertable), continuous
--              aggregates for hourly test stats and hourly audit stats,
--              50+ indexes (including GIN indexes on JSONB and arrays),
--              RLS policies per tenant, 14 security permissions, retention
--              policies (30-day test_runs, 365-day audit_log), compression,
--              and seed data registering the QA Test Harness Agent
--              (GL-FOUND-X-009) with capabilities in the agent registry.
-- Previous: V028__reproducibility_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS qa_test_harness_service;

-- =============================================================================
-- Table: qa_test_harness_service.test_suites
-- =============================================================================
-- Test suite definitions. Each suite groups related test cases with
-- configuration for parallel execution, worker count, fail-fast behavior,
-- and tag-based inclusion/exclusion filters. Tenant-scoped.

CREATE TABLE qa_test_harness_service.test_suites (
    suite_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    parallel BOOLEAN NOT NULL DEFAULT false,
    max_workers INTEGER NOT NULL DEFAULT 4,
    fail_fast BOOLEAN NOT NULL DEFAULT false,
    tags_include TEXT[] DEFAULT '{}',
    tags_exclude TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Max workers must be positive
ALTER TABLE qa_test_harness_service.test_suites
    ADD CONSTRAINT chk_suite_max_workers_positive
    CHECK (max_workers > 0);

-- Suite name must not be empty
ALTER TABLE qa_test_harness_service.test_suites
    ADD CONSTRAINT chk_suite_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- =============================================================================
-- Table: qa_test_harness_service.test_cases
-- =============================================================================
-- Test case definitions. Each test case belongs to a suite and defines
-- the agent under test, input data, expected output, golden file path,
-- timeout, tags, skip configuration, and severity. The category field
-- classifies the type of test (zero_hallucination, determinism, lineage,
-- golden_file, regression, performance, coverage, integration, unit).

CREATE TABLE qa_test_harness_service.test_cases (
    test_case_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    suite_id UUID NOT NULL REFERENCES qa_test_harness_service.test_suites(suite_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    category VARCHAR(30) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    input_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    expected_output JSONB DEFAULT NULL,
    golden_file_path TEXT DEFAULT NULL,
    timeout_seconds INTEGER NOT NULL DEFAULT 60,
    tags TEXT[] DEFAULT '{}',
    skip BOOLEAN NOT NULL DEFAULT false,
    skip_reason TEXT DEFAULT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Category constraint
ALTER TABLE qa_test_harness_service.test_cases
    ADD CONSTRAINT chk_test_case_category
    CHECK (category IN (
        'zero_hallucination', 'determinism', 'lineage',
        'golden_file', 'regression', 'performance',
        'coverage', 'integration', 'unit'
    ));

-- Severity constraint
ALTER TABLE qa_test_harness_service.test_cases
    ADD CONSTRAINT chk_test_case_severity
    CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info'));

-- Timeout must be positive
ALTER TABLE qa_test_harness_service.test_cases
    ADD CONSTRAINT chk_test_case_timeout_positive
    CHECK (timeout_seconds > 0);

-- Test case name must not be empty
ALTER TABLE qa_test_harness_service.test_cases
    ADD CONSTRAINT chk_test_case_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- =============================================================================
-- Table: qa_test_harness_service.test_runs (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording test execution results. Each test run
-- captures the suite, test case, agent type, category, execution status,
-- assertions, input/output hashes, duration, error details, agent result,
-- metadata, and provenance hash. Partitioned by started_at for time-series
-- queries. Retained for 30 days with compression after 3 days.

CREATE TABLE qa_test_harness_service.test_runs (
    run_id UUID NOT NULL DEFAULT gen_random_uuid(),
    suite_id UUID NOT NULL,
    test_case_id UUID NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    category VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL,
    assertions JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64),
    duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    error_message TEXT DEFAULT NULL,
    error_traceback TEXT DEFAULT NULL,
    agent_result JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system',
    PRIMARY KEY (run_id, started_at)
);

-- Create hypertable partitioned by started_at
SELECT create_hypertable('qa_test_harness_service.test_runs', 'started_at', if_not_exists => TRUE);

-- Status constraint
ALTER TABLE qa_test_harness_service.test_runs
    ADD CONSTRAINT chk_test_run_status
    CHECK (status IN ('passed', 'failed', 'error', 'skipped', 'timeout', 'running', 'pending'));

-- Category constraint
ALTER TABLE qa_test_harness_service.test_runs
    ADD CONSTRAINT chk_test_run_category
    CHECK (category IN (
        'zero_hallucination', 'determinism', 'lineage',
        'golden_file', 'regression', 'performance',
        'coverage', 'integration', 'unit'
    ));

-- Input hash must be 64-character hex (SHA-256)
ALTER TABLE qa_test_harness_service.test_runs
    ADD CONSTRAINT chk_test_run_input_hash_length
    CHECK (LENGTH(input_hash) = 64);

-- Output hash must be 64-character hex when present
ALTER TABLE qa_test_harness_service.test_runs
    ADD CONSTRAINT chk_test_run_output_hash_length
    CHECK (output_hash IS NULL OR LENGTH(output_hash) = 64);

-- Provenance hash must be 64-character hex
ALTER TABLE qa_test_harness_service.test_runs
    ADD CONSTRAINT chk_test_run_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Duration must be non-negative
ALTER TABLE qa_test_harness_service.test_runs
    ADD CONSTRAINT chk_test_run_duration_non_negative
    CHECK (duration_ms >= 0);

-- =============================================================================
-- Table: qa_test_harness_service.test_assertions
-- =============================================================================
-- Individual assertion results within a test run. Each assertion records
-- whether it passed, the expected and actual values, a descriptive message,
-- and the severity level. Used for detailed test result analysis and
-- failure diagnosis.

CREATE TABLE qa_test_harness_service.test_assertions (
    assertion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    passed BOOLEAN NOT NULL DEFAULT false,
    expected TEXT DEFAULT NULL,
    actual TEXT DEFAULT NULL,
    message TEXT DEFAULT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Severity constraint
ALTER TABLE qa_test_harness_service.test_assertions
    ADD CONSTRAINT chk_assertion_severity
    CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info'));

-- Assertion name must not be empty
ALTER TABLE qa_test_harness_service.test_assertions
    ADD CONSTRAINT chk_assertion_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- =============================================================================
-- Table: qa_test_harness_service.golden_files
-- =============================================================================
-- Golden file registry for snapshot testing. Each golden file record
-- tracks the agent type, name, version, input hash, content hash, file
-- path, description, and active status. Only one golden file per agent
-- type + name + tenant can be active at a time.

CREATE TABLE qa_test_harness_service.golden_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    input_hash VARCHAR(64) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    file_path TEXT NOT NULL,
    description TEXT DEFAULT '',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT 'system',
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Unique active golden file per agent_type + name + tenant
CREATE UNIQUE INDEX uq_golden_file_active
    ON qa_test_harness_service.golden_files (agent_type, name, tenant_id)
    WHERE is_active = true;

-- Input hash must be 64-character hex
ALTER TABLE qa_test_harness_service.golden_files
    ADD CONSTRAINT chk_golden_input_hash_length
    CHECK (LENGTH(input_hash) = 64);

-- Content hash must be 64-character hex
ALTER TABLE qa_test_harness_service.golden_files
    ADD CONSTRAINT chk_golden_content_hash_length
    CHECK (LENGTH(content_hash) = 64);

-- File path must not be empty
ALTER TABLE qa_test_harness_service.golden_files
    ADD CONSTRAINT chk_golden_file_path_not_empty
    CHECK (LENGTH(TRIM(file_path)) > 0);

-- =============================================================================
-- Table: qa_test_harness_service.performance_baselines
-- =============================================================================
-- Performance benchmark baseline records. Each baseline captures timing
-- statistics (min, max, mean, median, std_dev, p95, p99), memory usage,
-- and a configurable threshold for pass/fail determination. Used to detect
-- performance regressions across agent deployments.

CREATE TABLE qa_test_harness_service.performance_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(100) NOT NULL,
    operation VARCHAR(255) NOT NULL,
    iterations INTEGER NOT NULL DEFAULT 100,
    min_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    mean_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    median_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    std_dev_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    p95_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    p99_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    memory_mb DOUBLE PRECISION NOT NULL DEFAULT 0,
    threshold_ms DOUBLE PRECISION NOT NULL DEFAULT 1000,
    passed_threshold BOOLEAN NOT NULL DEFAULT true,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Iterations must be positive
ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_iterations_positive
    CHECK (iterations > 0);

-- Timing values must be non-negative
ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_min_non_negative
    CHECK (min_ms >= 0);

ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_max_non_negative
    CHECK (max_ms >= 0);

ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_mean_non_negative
    CHECK (mean_ms >= 0);

ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_median_non_negative
    CHECK (median_ms >= 0);

ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_std_dev_non_negative
    CHECK (std_dev_ms >= 0);

ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_p95_non_negative
    CHECK (p95_ms >= 0);

ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_p99_non_negative
    CHECK (p99_ms >= 0);

-- Memory must be non-negative
ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_memory_non_negative
    CHECK (memory_mb >= 0);

-- Threshold must be positive
ALTER TABLE qa_test_harness_service.performance_baselines
    ADD CONSTRAINT chk_perf_threshold_positive
    CHECK (threshold_ms > 0);

-- =============================================================================
-- Table: qa_test_harness_service.coverage_snapshots
-- =============================================================================
-- Coverage tracking snapshots. Each snapshot records the total and covered
-- methods, coverage percentage, uncovered method list, and test count for
-- an agent type. Used for tracking test coverage trends and enforcing
-- coverage thresholds.

CREATE TABLE qa_test_harness_service.coverage_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(100) NOT NULL,
    total_methods INTEGER NOT NULL DEFAULT 0,
    covered_methods INTEGER NOT NULL DEFAULT 0,
    coverage_percent DOUBLE PRECISION NOT NULL DEFAULT 0,
    uncovered_methods TEXT[] DEFAULT '{}',
    test_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Total methods must be non-negative
ALTER TABLE qa_test_harness_service.coverage_snapshots
    ADD CONSTRAINT chk_coverage_total_non_negative
    CHECK (total_methods >= 0);

-- Covered methods must be non-negative and <= total
ALTER TABLE qa_test_harness_service.coverage_snapshots
    ADD CONSTRAINT chk_coverage_covered_non_negative
    CHECK (covered_methods >= 0);

ALTER TABLE qa_test_harness_service.coverage_snapshots
    ADD CONSTRAINT chk_coverage_covered_lte_total
    CHECK (covered_methods <= total_methods);

-- Coverage percent must be between 0 and 100
ALTER TABLE qa_test_harness_service.coverage_snapshots
    ADD CONSTRAINT chk_coverage_percent_range
    CHECK (coverage_percent >= 0 AND coverage_percent <= 100);

-- Test count must be non-negative
ALTER TABLE qa_test_harness_service.coverage_snapshots
    ADD CONSTRAINT chk_coverage_test_count_non_negative
    CHECK (test_count >= 0);

-- =============================================================================
-- Table: qa_test_harness_service.regression_baselines
-- =============================================================================
-- Regression detection baselines. Each baseline records the expected
-- input/output hash pair for an agent type. When a test run produces a
-- different output hash for the same input hash, a regression is detected.
-- Only one active baseline per agent_type + input_hash + tenant.

CREATE TABLE qa_test_harness_service.regression_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type VARCHAR(100) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Unique active baseline per agent_type + input_hash + tenant
CREATE UNIQUE INDEX uq_regression_baseline_active
    ON qa_test_harness_service.regression_baselines (agent_type, input_hash, tenant_id)
    WHERE is_active = true;

-- Input hash must be 64-character hex
ALTER TABLE qa_test_harness_service.regression_baselines
    ADD CONSTRAINT chk_regression_input_hash_length
    CHECK (LENGTH(input_hash) = 64);

-- Output hash must be 64-character hex
ALTER TABLE qa_test_harness_service.regression_baselines
    ADD CONSTRAINT chk_regression_output_hash_length
    CHECK (LENGTH(output_hash) = 64);

-- =============================================================================
-- Table: qa_test_harness_service.qa_audit_log (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all
-- QA test harness operations. Each event captures the entity being operated
-- on, event type, action, data hashes (current, previous, chain), details
-- (JSONB), user, source IP, tenant, and timestamp. Partitioned by timestamp
-- for time-series queries. Retained for 365 days with compression after 30
-- days.

CREATE TABLE qa_test_harness_service.qa_audit_log (
    audit_id UUID NOT NULL DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    data_hash VARCHAR(64),
    previous_hash VARCHAR(64),
    chain_hash VARCHAR(64),
    details JSONB DEFAULT '{}'::jsonb,
    user_id VARCHAR(255) DEFAULT 'system',
    source_ip VARCHAR(45),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    PRIMARY KEY (audit_id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('qa_test_harness_service.qa_audit_log', 'timestamp', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE qa_test_harness_service.qa_audit_log
    ADD CONSTRAINT chk_qa_audit_event_type
    CHECK (event_type IN (
        'suite_created', 'suite_updated', 'suite_deleted',
        'test_case_created', 'test_case_updated', 'test_case_deleted',
        'test_run_started', 'test_run_completed', 'test_run_failed',
        'test_run_timeout', 'test_run_skipped',
        'assertion_passed', 'assertion_failed',
        'golden_file_created', 'golden_file_updated', 'golden_file_deactivated',
        'golden_file_mismatch',
        'performance_baseline_created', 'performance_baseline_updated',
        'performance_threshold_breach',
        'coverage_snapshot_created', 'coverage_threshold_breach',
        'regression_baseline_created', 'regression_baseline_updated',
        'regression_detected',
        'zero_hallucination_failure', 'determinism_failure',
        'admin_action', 'cache_invalidated', 'config_changed'
    ));

-- Entity type constraint
ALTER TABLE qa_test_harness_service.qa_audit_log
    ADD CONSTRAINT chk_qa_audit_entity_type
    CHECK (entity_type IN (
        'test_suite', 'test_case', 'test_run', 'test_assertion',
        'golden_file', 'performance_baseline', 'coverage_snapshot',
        'regression_baseline', 'system'
    ));

-- Action constraint
ALTER TABLE qa_test_harness_service.qa_audit_log
    ADD CONSTRAINT chk_qa_audit_action
    CHECK (action IN (
        'create', 'update', 'delete', 'execute', 'verify',
        'compare', 'detect', 'skip', 'timeout',
        'activate', 'deactivate', 'system', 'admin'
    ));

-- Hash fields must be 64-character hex when present
ALTER TABLE qa_test_harness_service.qa_audit_log
    ADD CONSTRAINT chk_qa_audit_data_hash_length
    CHECK (data_hash IS NULL OR LENGTH(data_hash) = 64);

ALTER TABLE qa_test_harness_service.qa_audit_log
    ADD CONSTRAINT chk_qa_audit_previous_hash_length
    CHECK (previous_hash IS NULL OR LENGTH(previous_hash) = 64);

ALTER TABLE qa_test_harness_service.qa_audit_log
    ADD CONSTRAINT chk_qa_audit_chain_hash_length
    CHECK (chain_hash IS NULL OR LENGTH(chain_hash) = 64);

-- =============================================================================
-- Continuous Aggregate: qa_test_harness_service.hourly_test_stats
-- =============================================================================
-- Precomputed hourly test run statistics by status and category for
-- dashboard queries, trend analysis, and SLI tracking. Shows the number
-- of test runs per status, average duration, pass/fail counts per hour.

CREATE MATERIALIZED VIEW qa_test_harness_service.hourly_test_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', started_at) AS bucket,
    status,
    category,
    tenant_id,
    COUNT(*) AS run_count,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) AS passed_count,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
    SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped_count,
    SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) AS timeout_count
FROM qa_test_harness_service.test_runs
WHERE started_at IS NOT NULL
GROUP BY bucket, status, category, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('qa_test_harness_service.hourly_test_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: qa_test_harness_service.hourly_audit_stats
-- =============================================================================
-- Precomputed hourly counts of audit events by event type and entity type
-- for compliance reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW qa_test_harness_service.hourly_audit_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    event_type,
    entity_type,
    tenant_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT entity_id) AS unique_entities,
    COUNT(DISTINCT user_id) AS unique_users
FROM qa_test_harness_service.qa_audit_log
WHERE timestamp IS NOT NULL
GROUP BY bucket, event_type, entity_type, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('qa_test_harness_service.hourly_audit_stats',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- test_suites indexes
CREATE INDEX idx_ts_name ON qa_test_harness_service.test_suites(name);
CREATE INDEX idx_ts_created_at ON qa_test_harness_service.test_suites(created_at DESC);
CREATE INDEX idx_ts_tenant ON qa_test_harness_service.test_suites(tenant_id);
CREATE INDEX idx_ts_tenant_name ON qa_test_harness_service.test_suites(tenant_id, name);
CREATE INDEX idx_ts_tags_include ON qa_test_harness_service.test_suites USING GIN (tags_include);
CREATE INDEX idx_ts_tags_exclude ON qa_test_harness_service.test_suites USING GIN (tags_exclude);

-- test_cases indexes
CREATE INDEX idx_tc_suite ON qa_test_harness_service.test_cases(suite_id);
CREATE INDEX idx_tc_name ON qa_test_harness_service.test_cases(name);
CREATE INDEX idx_tc_category ON qa_test_harness_service.test_cases(category);
CREATE INDEX idx_tc_agent_type ON qa_test_harness_service.test_cases(agent_type);
CREATE INDEX idx_tc_severity ON qa_test_harness_service.test_cases(severity);
CREATE INDEX idx_tc_skip ON qa_test_harness_service.test_cases(skip);
CREATE INDEX idx_tc_created_at ON qa_test_harness_service.test_cases(created_at DESC);
CREATE INDEX idx_tc_tenant ON qa_test_harness_service.test_cases(tenant_id);
CREATE INDEX idx_tc_tenant_category ON qa_test_harness_service.test_cases(tenant_id, category);
CREATE INDEX idx_tc_tenant_agent ON qa_test_harness_service.test_cases(tenant_id, agent_type);
CREATE INDEX idx_tc_tags ON qa_test_harness_service.test_cases USING GIN (tags);
CREATE INDEX idx_tc_input_data ON qa_test_harness_service.test_cases USING GIN (input_data);
CREATE INDEX idx_tc_expected_output ON qa_test_harness_service.test_cases USING GIN (expected_output);

-- test_runs indexes (hypertable-aware)
CREATE INDEX idx_tr_suite ON qa_test_harness_service.test_runs(suite_id, started_at DESC);
CREATE INDEX idx_tr_test_case ON qa_test_harness_service.test_runs(test_case_id, started_at DESC);
CREATE INDEX idx_tr_agent_type ON qa_test_harness_service.test_runs(agent_type, started_at DESC);
CREATE INDEX idx_tr_category ON qa_test_harness_service.test_runs(category, started_at DESC);
CREATE INDEX idx_tr_status ON qa_test_harness_service.test_runs(status, started_at DESC);
CREATE INDEX idx_tr_input_hash ON qa_test_harness_service.test_runs(input_hash, started_at DESC);
CREATE INDEX idx_tr_output_hash ON qa_test_harness_service.test_runs(output_hash, started_at DESC);
CREATE INDEX idx_tr_provenance_hash ON qa_test_harness_service.test_runs(provenance_hash);
CREATE INDEX idx_tr_tenant ON qa_test_harness_service.test_runs(tenant_id, started_at DESC);
CREATE INDEX idx_tr_tenant_status ON qa_test_harness_service.test_runs(tenant_id, status, started_at DESC);
CREATE INDEX idx_tr_tenant_category ON qa_test_harness_service.test_runs(tenant_id, category, started_at DESC);
CREATE INDEX idx_tr_assertions ON qa_test_harness_service.test_runs USING GIN (assertions);
CREATE INDEX idx_tr_agent_result ON qa_test_harness_service.test_runs USING GIN (agent_result);
CREATE INDEX idx_tr_metadata ON qa_test_harness_service.test_runs USING GIN (metadata);

-- test_assertions indexes
CREATE INDEX idx_ta_run ON qa_test_harness_service.test_assertions(run_id);
CREATE INDEX idx_ta_name ON qa_test_harness_service.test_assertions(name);
CREATE INDEX idx_ta_passed ON qa_test_harness_service.test_assertions(passed);
CREATE INDEX idx_ta_severity ON qa_test_harness_service.test_assertions(severity);
CREATE INDEX idx_ta_created_at ON qa_test_harness_service.test_assertions(created_at DESC);
CREATE INDEX idx_ta_tenant ON qa_test_harness_service.test_assertions(tenant_id);
CREATE INDEX idx_ta_tenant_passed ON qa_test_harness_service.test_assertions(tenant_id, passed);

-- golden_files indexes
CREATE INDEX idx_gf_agent_type ON qa_test_harness_service.golden_files(agent_type);
CREATE INDEX idx_gf_name ON qa_test_harness_service.golden_files(name);
CREATE INDEX idx_gf_version ON qa_test_harness_service.golden_files(version);
CREATE INDEX idx_gf_input_hash ON qa_test_harness_service.golden_files(input_hash);
CREATE INDEX idx_gf_content_hash ON qa_test_harness_service.golden_files(content_hash);
CREATE INDEX idx_gf_is_active ON qa_test_harness_service.golden_files(is_active);
CREATE INDEX idx_gf_created_at ON qa_test_harness_service.golden_files(created_at DESC);
CREATE INDEX idx_gf_tenant ON qa_test_harness_service.golden_files(tenant_id);
CREATE INDEX idx_gf_tenant_agent ON qa_test_harness_service.golden_files(tenant_id, agent_type);
CREATE INDEX idx_gf_active_tenant ON qa_test_harness_service.golden_files(tenant_id, is_active);

-- performance_baselines indexes
CREATE INDEX idx_pb_agent_type ON qa_test_harness_service.performance_baselines(agent_type);
CREATE INDEX idx_pb_operation ON qa_test_harness_service.performance_baselines(operation);
CREATE INDEX idx_pb_is_active ON qa_test_harness_service.performance_baselines(is_active);
CREATE INDEX idx_pb_passed_threshold ON qa_test_harness_service.performance_baselines(passed_threshold);
CREATE INDEX idx_pb_created_at ON qa_test_harness_service.performance_baselines(created_at DESC);
CREATE INDEX idx_pb_tenant ON qa_test_harness_service.performance_baselines(tenant_id);
CREATE INDEX idx_pb_tenant_agent ON qa_test_harness_service.performance_baselines(tenant_id, agent_type);
CREATE INDEX idx_pb_active_tenant ON qa_test_harness_service.performance_baselines(tenant_id, is_active);

-- coverage_snapshots indexes
CREATE INDEX idx_cs_agent_type ON qa_test_harness_service.coverage_snapshots(agent_type);
CREATE INDEX idx_cs_coverage_percent ON qa_test_harness_service.coverage_snapshots(coverage_percent);
CREATE INDEX idx_cs_created_at ON qa_test_harness_service.coverage_snapshots(created_at DESC);
CREATE INDEX idx_cs_tenant ON qa_test_harness_service.coverage_snapshots(tenant_id);
CREATE INDEX idx_cs_tenant_agent ON qa_test_harness_service.coverage_snapshots(tenant_id, agent_type);
CREATE INDEX idx_cs_uncovered ON qa_test_harness_service.coverage_snapshots USING GIN (uncovered_methods);

-- regression_baselines indexes
CREATE INDEX idx_rb_agent_type ON qa_test_harness_service.regression_baselines(agent_type);
CREATE INDEX idx_rb_input_hash ON qa_test_harness_service.regression_baselines(input_hash);
CREATE INDEX idx_rb_output_hash ON qa_test_harness_service.regression_baselines(output_hash);
CREATE INDEX idx_rb_is_active ON qa_test_harness_service.regression_baselines(is_active);
CREATE INDEX idx_rb_created_at ON qa_test_harness_service.regression_baselines(created_at DESC);
CREATE INDEX idx_rb_tenant ON qa_test_harness_service.regression_baselines(tenant_id);
CREATE INDEX idx_rb_tenant_agent ON qa_test_harness_service.regression_baselines(tenant_id, agent_type);
CREATE INDEX idx_rb_active_tenant ON qa_test_harness_service.regression_baselines(tenant_id, is_active);

-- qa_audit_log indexes (hypertable-aware)
CREATE INDEX idx_qal_event_type ON qa_test_harness_service.qa_audit_log(event_type, timestamp DESC);
CREATE INDEX idx_qal_entity_type ON qa_test_harness_service.qa_audit_log(entity_type, timestamp DESC);
CREATE INDEX idx_qal_entity_id ON qa_test_harness_service.qa_audit_log(entity_id, timestamp DESC);
CREATE INDEX idx_qal_action ON qa_test_harness_service.qa_audit_log(action, timestamp DESC);
CREATE INDEX idx_qal_user ON qa_test_harness_service.qa_audit_log(user_id, timestamp DESC);
CREATE INDEX idx_qal_tenant ON qa_test_harness_service.qa_audit_log(tenant_id, timestamp DESC);
CREATE INDEX idx_qal_data_hash ON qa_test_harness_service.qa_audit_log(data_hash);
CREATE INDEX idx_qal_chain_hash ON qa_test_harness_service.qa_audit_log(chain_hash);
CREATE INDEX idx_qal_details ON qa_test_harness_service.qa_audit_log USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE qa_test_harness_service.test_suites ENABLE ROW LEVEL SECURITY;
CREATE POLICY ts_tenant_read ON qa_test_harness_service.test_suites
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ts_tenant_write ON qa_test_harness_service.test_suites
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.test_cases ENABLE ROW LEVEL SECURITY;
CREATE POLICY tc_tenant_read ON qa_test_harness_service.test_cases
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY tc_tenant_write ON qa_test_harness_service.test_cases
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.test_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY tr_tenant_read ON qa_test_harness_service.test_runs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY tr_tenant_write ON qa_test_harness_service.test_runs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.test_assertions ENABLE ROW LEVEL SECURITY;
CREATE POLICY ta_tenant_read ON qa_test_harness_service.test_assertions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ta_tenant_write ON qa_test_harness_service.test_assertions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.golden_files ENABLE ROW LEVEL SECURITY;
CREATE POLICY gf_tenant_read ON qa_test_harness_service.golden_files
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY gf_tenant_write ON qa_test_harness_service.golden_files
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.performance_baselines ENABLE ROW LEVEL SECURITY;
CREATE POLICY pb_tenant_read ON qa_test_harness_service.performance_baselines
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY pb_tenant_write ON qa_test_harness_service.performance_baselines
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.coverage_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY cs_tenant_read ON qa_test_harness_service.coverage_snapshots
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cs_tenant_write ON qa_test_harness_service.coverage_snapshots
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.regression_baselines ENABLE ROW LEVEL SECURITY;
CREATE POLICY rb_tenant_read ON qa_test_harness_service.regression_baselines
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY rb_tenant_write ON qa_test_harness_service.regression_baselines
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE qa_test_harness_service.qa_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY qal_tenant_read ON qa_test_harness_service.qa_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qal_tenant_write ON qa_test_harness_service.qa_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA qa_test_harness_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA qa_test_harness_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA qa_test_harness_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON qa_test_harness_service.hourly_test_stats TO greenlang_app;
GRANT SELECT ON qa_test_harness_service.hourly_audit_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA qa_test_harness_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA qa_test_harness_service TO greenlang_readonly;
GRANT SELECT ON qa_test_harness_service.hourly_test_stats TO greenlang_readonly;
GRANT SELECT ON qa_test_harness_service.hourly_audit_stats TO greenlang_readonly;

-- Add QA test harness service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'qa_test_harness:suites:read', 'qa_test_harness', 'suites_read', 'View test suite definitions'),
    (gen_random_uuid(), 'qa_test_harness:suites:write', 'qa_test_harness', 'suites_write', 'Create and update test suites'),
    (gen_random_uuid(), 'qa_test_harness:cases:read', 'qa_test_harness', 'cases_read', 'View test case definitions'),
    (gen_random_uuid(), 'qa_test_harness:cases:write', 'qa_test_harness', 'cases_write', 'Create and update test cases'),
    (gen_random_uuid(), 'qa_test_harness:runs:read', 'qa_test_harness', 'runs_read', 'View test run execution records'),
    (gen_random_uuid(), 'qa_test_harness:runs:write', 'qa_test_harness', 'runs_write', 'Execute test runs and record results'),
    (gen_random_uuid(), 'qa_test_harness:golden_files:read', 'qa_test_harness', 'golden_files_read', 'View golden file definitions and content'),
    (gen_random_uuid(), 'qa_test_harness:golden_files:write', 'qa_test_harness', 'golden_files_write', 'Create and update golden files'),
    (gen_random_uuid(), 'qa_test_harness:baselines:read', 'qa_test_harness', 'baselines_read', 'View performance and regression baselines'),
    (gen_random_uuid(), 'qa_test_harness:baselines:write', 'qa_test_harness', 'baselines_write', 'Create and update performance and regression baselines'),
    (gen_random_uuid(), 'qa_test_harness:coverage:read', 'qa_test_harness', 'coverage_read', 'View coverage snapshots and trends'),
    (gen_random_uuid(), 'qa_test_harness:coverage:write', 'qa_test_harness', 'coverage_write', 'Create coverage snapshots'),
    (gen_random_uuid(), 'qa_test_harness:audit:read', 'qa_test_harness', 'audit_read', 'View QA test harness audit event log'),
    (gen_random_uuid(), 'qa_test_harness:admin', 'qa_test_harness', 'admin', 'QA test harness service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep test runs for 30 days
SELECT add_retention_policy('qa_test_harness_service.test_runs', INTERVAL '30 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('qa_test_harness_service.qa_audit_log', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on test_runs after 3 days
ALTER TABLE qa_test_harness_service.test_runs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'started_at DESC'
);

SELECT add_compression_policy('qa_test_harness_service.test_runs', INTERVAL '3 days');

-- Enable compression on qa_audit_log after 30 days
ALTER TABLE qa_test_harness_service.qa_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('qa_test_harness_service.qa_audit_log', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the QA Test Harness Agent (GL-FOUND-X-009) in Agent Registry
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-FOUND-X-009', 'QA Test Harness',
 'Comprehensive testing framework for all GreenLang agents. Provides zero-hallucination verification, determinism testing, lineage completeness checks, golden file/snapshot testing, regression detection, performance benchmarking, coverage tracking, and multi-format report generation.',
 1, 'sync', true, true, 20, '1.0.0', false,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/qa-test-harness', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for QA Test Harness
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-FOUND-X-009', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/qa-test-harness-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "qa", "testing", "verification", "golden-files"}',
 '{"cross-sector"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for QA Test Harness
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-FOUND-X-009', '1.0.0', 'zero_hallucination_verification', 'validation',
 'Verify agent outputs contain zero hallucinated data by checking all output values trace to input data or known reference sources',
 '{"agent_output", "input_data", "reference_sources"}', '{"verification_result", "hallucination_report"}',
 '{"strict_mode": true, "trace_depth": 5, "reference_tolerance": 0.0}'::jsonb),

('GL-FOUND-X-009', '1.0.0', 'determinism_testing', 'validation',
 'Execute the same agent with identical inputs N times and verify outputs are bit-for-bit identical',
 '{"agent_type", "input_data", "iterations"}', '{"determinism_result", "hash_comparisons"}',
 '{"default_iterations": 5, "hash_algorithm": "sha256", "tolerance": 0.0}'::jsonb),

('GL-FOUND-X-009', '1.0.0', 'lineage_completeness', 'validation',
 'Verify that agent outputs include complete provenance chains tracing every value to its source',
 '{"agent_output", "provenance_chain"}', '{"lineage_result", "missing_links"}',
 '{"require_sha256": true, "max_chain_depth": 20, "strict_completeness": true}'::jsonb),

('GL-FOUND-X-009', '1.0.0', 'golden_file_testing', 'validation',
 'Compare agent output against golden file snapshots with configurable diff tolerance for regression detection',
 '{"agent_output", "golden_file_id"}', '{"comparison_result", "diff_report"}',
 '{"update_on_pass": false, "tolerance": 0.0, "ignore_fields": []}'::jsonb),

('GL-FOUND-X-009', '1.0.0', 'regression_detection', 'analysis',
 'Detect output regressions by comparing current output hashes against established baselines',
 '{"agent_type", "input_hash", "output_hash"}', '{"regression_result", "baseline_comparison"}',
 '{"auto_update_baseline": false, "alert_on_regression": true}'::jsonb),

('GL-FOUND-X-009', '1.0.0', 'performance_benchmarking', 'computation',
 'Benchmark agent execution time and memory usage with statistical analysis (min, max, mean, median, p95, p99, std_dev)',
 '{"agent_type", "operation", "iterations"}', '{"benchmark_result", "threshold_check"}',
 '{"default_iterations": 100, "warmup_iterations": 10, "threshold_ms": 1000}'::jsonb),

('GL-FOUND-X-009', '1.0.0', 'coverage_tracking', 'analysis',
 'Track test coverage percentage for agent methods and identify uncovered code paths',
 '{"agent_type"}', '{"coverage_snapshot", "uncovered_methods"}',
 '{"minimum_threshold": 80, "include_private": false}'::jsonb),

('GL-FOUND-X-009', '1.0.0', 'report_generation', 'computation',
 'Generate multi-format test reports (JSON, HTML, Markdown, JUnit XML) with detailed results and trend analysis',
 '{"test_results", "format"}', '{"report_content", "summary_stats"}',
 '{"formats": ["json", "html", "markdown", "junit"], "include_trends": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for QA Test Harness
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- QA Test Harness depends on Schema Compiler for input/output validation
('GL-FOUND-X-009', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Test inputs and expected outputs are validated against JSON Schema definitions'),

-- QA Test Harness depends on Registry for agent discovery
('GL-FOUND-X-009', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent discovery and version lookup for comprehensive test coverage'),

-- QA Test Harness depends on Reproducibility for determinism verification
('GL-FOUND-X-009', 'GL-FOUND-X-008', '>=1.0.0', false,
 'Determinism tests leverage the reproducibility agent for hash verification and environment fingerprinting'),

-- QA Test Harness optionally uses Citations for provenance verification
('GL-FOUND-X-009', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Lineage completeness checks verify provenance chains via the citation service'),

-- QA Test Harness optionally uses Orchestrator for integration testing
('GL-FOUND-X-009', 'GL-FOUND-X-001', '>=1.0.0', true,
 'Integration tests may execute agent DAGs via the orchestrator')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for QA Test Harness
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-FOUND-X-009', 'QA Test Harness',
 'Comprehensive testing framework for GreenLang agents: zero-hallucination verification, determinism testing, lineage completeness, golden file testing, regression detection, performance benchmarking, coverage tracking, and multi-format report generation.',
 'foundation', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA qa_test_harness_service IS 'QA Test Harness for GreenLang Climate OS (AGENT-FOUND-009) - zero-hallucination verification, determinism testing, lineage completeness, golden file testing, regression detection, performance benchmarking, coverage tracking, and report generation';
COMMENT ON TABLE qa_test_harness_service.test_suites IS 'Test suite definitions with parallel execution config, worker count, fail-fast, and tag-based filters';
COMMENT ON TABLE qa_test_harness_service.test_cases IS 'Test case definitions with category, agent type, input/expected data, golden file path, timeout, tags, skip, and severity';
COMMENT ON TABLE qa_test_harness_service.test_runs IS 'TimescaleDB hypertable: test execution records with status, assertions, hashes, duration, error details, and provenance';
COMMENT ON TABLE qa_test_harness_service.test_assertions IS 'Individual assertion results within test runs with pass/fail, expected/actual values, and severity';
COMMENT ON TABLE qa_test_harness_service.golden_files IS 'Golden file registry for snapshot testing with agent type, input/content hashes, file path, and active flag';
COMMENT ON TABLE qa_test_harness_service.performance_baselines IS 'Performance benchmark baselines with timing statistics (min/max/mean/median/p95/p99), memory, and threshold';
COMMENT ON TABLE qa_test_harness_service.coverage_snapshots IS 'Coverage tracking snapshots with total/covered methods, coverage percent, uncovered methods list, and test count';
COMMENT ON TABLE qa_test_harness_service.regression_baselines IS 'Regression detection baselines with expected input/output hash pairs per agent type';
COMMENT ON TABLE qa_test_harness_service.qa_audit_log IS 'TimescaleDB hypertable: comprehensive audit events for all QA test harness operations with hash chain integrity';
COMMENT ON MATERIALIZED VIEW qa_test_harness_service.hourly_test_stats IS 'Continuous aggregate: hourly test run statistics by status and category for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW qa_test_harness_service.hourly_audit_stats IS 'Continuous aggregate: hourly audit event counts by event type and entity type for compliance reporting';

COMMENT ON COLUMN qa_test_harness_service.test_cases.category IS 'Test category: zero_hallucination, determinism, lineage, golden_file, regression, performance, coverage, integration, unit';
COMMENT ON COLUMN qa_test_harness_service.test_cases.severity IS 'Test case severity: critical, high, medium, low, info';
COMMENT ON COLUMN qa_test_harness_service.test_runs.status IS 'Test run status: passed, failed, error, skipped, timeout, running, pending';
COMMENT ON COLUMN qa_test_harness_service.test_runs.provenance_hash IS 'SHA-256 hash of the test run content for integrity verification';
COMMENT ON COLUMN qa_test_harness_service.test_runs.input_hash IS 'SHA-256 hash of the test input data';
COMMENT ON COLUMN qa_test_harness_service.test_runs.output_hash IS 'SHA-256 hash of the agent output for regression comparison';
COMMENT ON COLUMN qa_test_harness_service.golden_files.content_hash IS 'SHA-256 hash of the golden file content for integrity verification';
COMMENT ON COLUMN qa_test_harness_service.performance_baselines.threshold_ms IS 'Maximum allowed execution time in milliseconds for pass/fail determination';
COMMENT ON COLUMN qa_test_harness_service.coverage_snapshots.coverage_percent IS 'Coverage percentage (0-100) of tested methods for the agent type';
COMMENT ON COLUMN qa_test_harness_service.regression_baselines.input_hash IS 'SHA-256 hash of the test input for baseline matching';
COMMENT ON COLUMN qa_test_harness_service.regression_baselines.output_hash IS 'SHA-256 hash of the expected output for regression comparison';
COMMENT ON COLUMN qa_test_harness_service.qa_audit_log.event_type IS 'Audit event type: suite_created, test_run_completed, golden_file_mismatch, regression_detected, etc.';
COMMENT ON COLUMN qa_test_harness_service.qa_audit_log.entity_type IS 'Entity type: test_suite, test_case, test_run, test_assertion, golden_file, performance_baseline, coverage_snapshot, regression_baseline, system';
COMMENT ON COLUMN qa_test_harness_service.qa_audit_log.chain_hash IS 'SHA-256 hash chain linking this event to the previous event for tamper detection';
