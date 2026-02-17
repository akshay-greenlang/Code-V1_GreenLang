-- =============================================================================
-- V044: Time Series Gap Filler Service Schema
-- =============================================================================
-- Component: AGENT-DATA-014 (Time Series Gap Filler Agent)
-- Agent ID:  GL-DATA-X-017
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Time Series Gap Filler Agent (GL-DATA-X-017) with capabilities for
-- automatic gap detection in time-series data (missing timestamps,
-- irregular intervals, calendar-aware gaps), frequency analysis
-- (auto-detection, regularity scoring, median interval computation),
-- intelligent fill strategies (linear interpolation, spline, LOCF/NOCB,
-- seasonal decomposition, regression-based, correlation-based,
-- calendar-aware), multi-series correlation analysis (Pearson, Spearman,
-- Kendall, cross-correlation with lag detection), fill validation
-- (plausibility checks, distribution preservation, range enforcement),
-- business calendar support (fiscal calendars, holidays, custom rules),
-- comprehensive reporting (coverage before/after, strategy breakdown,
-- validation summary), and full provenance chain tracking with
-- SHA-256 hashes for zero-hallucination audit trails.
-- =============================================================================
-- Tables (10):
--   1. gap_filler_jobs            - Job tracking
--   2. gap_detections             - Gap detection results per series
--   3. gap_frequencies            - Detected frequency analysis per series
--   4. gap_fills                  - Fill results per gap
--   5. gap_strategies             - Strategy selection per gap type
--   6. gap_validations            - Validation results per fill
--   7. gap_calendars              - Business calendar definitions
--   8. gap_correlations           - Cross-series correlation analysis
--   9. gap_reports                - Gap fill summary reports
--  10. gap_audit_log              - Audit trail
--
-- Hypertables (3):
--  11. gap_events                 - Gap event time-series (hypertable)
--  12. fill_events                - Fill event time-series (hypertable)
--  13. validation_events          - Validation event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. gap_hourly_stats           - Hourly gap event stats
--   2. fill_hourly_stats          - Hourly fill event stats
--
-- Also includes: 150+ indexes (B-tree, GIN, partial, composite),
-- 75+ CHECK constraints, 26 RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-017.
-- Previous: V043__outlier_detection_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS time_series_gap_filler_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION time_series_gap_filler_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: time_series_gap_filler_service.gap_filler_jobs
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_filler_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    dataset_id TEXT NOT NULL,
    series_id TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    strategy VARCHAR(30) NOT NULL DEFAULT 'auto',
    frequency_hint TEXT,
    calendar_id UUID,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

ALTER TABLE time_series_gap_filler_service.gap_filler_jobs
    ADD CONSTRAINT chk_gfj_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));
ALTER TABLE time_series_gap_filler_service.gap_filler_jobs
    ADD CONSTRAINT chk_gfj_strategy CHECK (strategy IN ('auto', 'linear', 'spline', 'locf', 'nocb', 'mean', 'median', 'seasonal', 'regression', 'correlation', 'calendar_aware', 'zero_fill', 'custom'));
ALTER TABLE time_series_gap_filler_service.gap_filler_jobs
    ADD CONSTRAINT chk_gfj_dataset_id_not_empty CHECK (LENGTH(TRIM(dataset_id)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_filler_jobs
    ADD CONSTRAINT chk_gfj_series_id_not_empty CHECK (LENGTH(TRIM(series_id)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_filler_jobs
    ADD CONSTRAINT chk_gfj_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_filler_jobs
    ADD CONSTRAINT chk_gfj_completed_after_created CHECK (completed_at IS NULL OR completed_at >= created_at);

CREATE TRIGGER trg_gfj_updated_at
    BEFORE UPDATE ON time_series_gap_filler_service.gap_filler_jobs
    FOR EACH ROW EXECUTE FUNCTION time_series_gap_filler_service.set_updated_at();

-- =============================================================================
-- Table 2: time_series_gap_filler_service.gap_detections
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    job_id UUID NOT NULL,
    series_id TEXT NOT NULL,
    total_gaps INTEGER NOT NULL DEFAULT 0,
    total_missing INTEGER NOT NULL DEFAULT 0,
    total_expected INTEGER NOT NULL DEFAULT 0,
    coverage_ratio DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    gaps JSONB NOT NULL DEFAULT '[]'::jsonb,
    provenance_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT fk_gd_job_id FOREIGN KEY (job_id) REFERENCES time_series_gap_filler_service.gap_filler_jobs(id) ON DELETE CASCADE;
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_total_gaps_non_negative CHECK (total_gaps >= 0);
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_total_missing_non_negative CHECK (total_missing >= 0);
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_total_expected_non_negative CHECK (total_expected >= 0);
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_coverage_ratio_range CHECK (coverage_ratio >= 0.0 AND coverage_ratio <= 1.0);
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_missing_lte_expected CHECK (total_missing <= total_expected OR total_expected = 0);
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_series_id_not_empty CHECK (LENGTH(TRIM(series_id)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_detections
    ADD CONSTRAINT chk_gd_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 3: time_series_gap_filler_service.gap_frequencies
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_frequencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    series_id TEXT NOT NULL,
    detected_frequency TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    regularity_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    median_interval_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    expected_count INTEGER NOT NULL DEFAULT 0,
    actual_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_confidence_range CHECK (confidence >= 0.0 AND confidence <= 1.0);
ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_regularity_score_range CHECK (regularity_score >= 0.0 AND regularity_score <= 1.0);
ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_median_interval_non_negative CHECK (median_interval_seconds >= 0.0);
ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_expected_count_non_negative CHECK (expected_count >= 0);
ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_actual_count_non_negative CHECK (actual_count >= 0);
ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_detected_frequency_not_empty CHECK (LENGTH(TRIM(detected_frequency)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_series_id_not_empty CHECK (LENGTH(TRIM(series_id)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_frequencies
    ADD CONSTRAINT chk_gf_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 4: time_series_gap_filler_service.gap_fills
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    job_id UUID NOT NULL,
    gap_id UUID NOT NULL,
    strategy TEXT NOT NULL,
    fill_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    filled_values JSONB NOT NULL DEFAULT '[]'::jsonb,
    provenance_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT fk_gfl_job_id FOREIGN KEY (job_id) REFERENCES time_series_gap_filler_service.gap_filler_jobs(id) ON DELETE CASCADE;
ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT fk_gfl_gap_id FOREIGN KEY (gap_id) REFERENCES time_series_gap_filler_service.gap_detections(id) ON DELETE CASCADE;
ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT chk_gfl_strategy CHECK (strategy IN ('linear', 'spline', 'locf', 'nocb', 'mean', 'median', 'seasonal', 'regression', 'correlation', 'calendar_aware', 'zero_fill', 'custom', 'auto'));
ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT chk_gfl_fill_count_non_negative CHECK (fill_count >= 0);
ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT chk_gfl_avg_confidence_range CHECK (avg_confidence >= 0.0 AND avg_confidence <= 1.0);
ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT chk_gfl_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT chk_gfl_strategy_not_empty CHECK (LENGTH(TRIM(strategy)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_fills
    ADD CONSTRAINT chk_gfl_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 5: time_series_gap_filler_service.gap_strategies
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    job_id UUID NOT NULL,
    gap_type TEXT NOT NULL,
    strategy TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    justification TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_strategies
    ADD CONSTRAINT fk_gs_job_id FOREIGN KEY (job_id) REFERENCES time_series_gap_filler_service.gap_filler_jobs(id) ON DELETE CASCADE;
ALTER TABLE time_series_gap_filler_service.gap_strategies
    ADD CONSTRAINT chk_gs_gap_type CHECK (gap_type IN ('single_point', 'short_run', 'long_run', 'periodic', 'trailing', 'leading', 'random', 'systematic', 'business_day', 'holiday', 'weekend', 'custom'));
ALTER TABLE time_series_gap_filler_service.gap_strategies
    ADD CONSTRAINT chk_gs_strategy CHECK (strategy IN ('linear', 'spline', 'locf', 'nocb', 'mean', 'median', 'seasonal', 'regression', 'correlation', 'calendar_aware', 'zero_fill', 'custom', 'auto'));
ALTER TABLE time_series_gap_filler_service.gap_strategies
    ADD CONSTRAINT chk_gs_gap_type_not_empty CHECK (LENGTH(TRIM(gap_type)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_strategies
    ADD CONSTRAINT chk_gs_strategy_not_empty CHECK (LENGTH(TRIM(strategy)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_strategies
    ADD CONSTRAINT chk_gs_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 6: time_series_gap_filler_service.gap_validations
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_validations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    fill_id UUID NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    plausibility_passed BOOLEAN NOT NULL DEFAULT false,
    distribution_preserved BOOLEAN NOT NULL DEFAULT false,
    range_check_passed BOOLEAN NOT NULL DEFAULT false,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_validations
    ADD CONSTRAINT fk_gv_fill_id FOREIGN KEY (fill_id) REFERENCES time_series_gap_filler_service.gap_fills(id) ON DELETE CASCADE;
ALTER TABLE time_series_gap_filler_service.gap_validations
    ADD CONSTRAINT chk_gv_status CHECK (status IN ('pending', 'passed', 'failed', 'warning', 'skipped'));
ALTER TABLE time_series_gap_filler_service.gap_validations
    ADD CONSTRAINT chk_gv_status_not_empty CHECK (LENGTH(TRIM(status)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_validations
    ADD CONSTRAINT chk_gv_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 7: time_series_gap_filler_service.gap_calendars
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_calendars (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    calendar_type TEXT NOT NULL DEFAULT 'business',
    business_days JSONB NOT NULL DEFAULT '["monday","tuesday","wednesday","thursday","friday"]'::jsonb,
    holidays JSONB NOT NULL DEFAULT '[]'::jsonb,
    fiscal_year_start_month INTEGER NOT NULL DEFAULT 1,
    custom_rules JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_calendars
    ADD CONSTRAINT uq_gc_tenant_name UNIQUE (tenant_id, name);
ALTER TABLE time_series_gap_filler_service.gap_calendars
    ADD CONSTRAINT chk_gc_calendar_type CHECK (calendar_type IN ('business', 'fiscal', 'academic', 'retail', 'manufacturing', 'custom', 'iso_8601'));
ALTER TABLE time_series_gap_filler_service.gap_calendars
    ADD CONSTRAINT chk_gc_fiscal_year_start_month_range CHECK (fiscal_year_start_month >= 1 AND fiscal_year_start_month <= 12);
ALTER TABLE time_series_gap_filler_service.gap_calendars
    ADD CONSTRAINT chk_gc_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_calendars
    ADD CONSTRAINT chk_gc_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_gc_updated_at
    BEFORE UPDATE ON time_series_gap_filler_service.gap_calendars
    FOR EACH ROW EXECUTE FUNCTION time_series_gap_filler_service.set_updated_at();

-- =============================================================================
-- Table 8: time_series_gap_filler_service.gap_correlations
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_correlations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    source_series_id TEXT NOT NULL,
    reference_series_id TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT 'pearson',
    coefficient DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    p_value DOUBLE PRECISION,
    lag INTEGER NOT NULL DEFAULT 0,
    sample_size INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_method CHECK (method IN ('pearson', 'spearman', 'kendall', 'cross_correlation', 'mutual_information'));
ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_coefficient_range CHECK (coefficient >= -1.0 AND coefficient <= 1.0);
ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_p_value_range CHECK (p_value IS NULL OR (p_value >= 0.0 AND p_value <= 1.0));
ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_sample_size_non_negative CHECK (sample_size >= 0);
ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_source_series_not_empty CHECK (LENGTH(TRIM(source_series_id)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_reference_series_not_empty CHECK (LENGTH(TRIM(reference_series_id)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_different_series CHECK (source_series_id != reference_series_id);
ALTER TABLE time_series_gap_filler_service.gap_correlations
    ADD CONSTRAINT chk_gcr_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 9: time_series_gap_filler_service.gap_reports
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    job_id UUID NOT NULL,
    format TEXT NOT NULL DEFAULT 'json',
    total_gaps INTEGER NOT NULL DEFAULT 0,
    total_filled INTEGER NOT NULL DEFAULT 0,
    total_validated INTEGER NOT NULL DEFAULT 0,
    coverage_before DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    coverage_after DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    report_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT fk_grp_job_id FOREIGN KEY (job_id) REFERENCES time_series_gap_filler_service.gap_filler_jobs(id) ON DELETE CASCADE;
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_format CHECK (format IN ('json', 'csv', 'markdown', 'html', 'pdf'));
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_total_gaps_non_negative CHECK (total_gaps >= 0);
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_total_filled_non_negative CHECK (total_filled >= 0);
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_total_validated_non_negative CHECK (total_validated >= 0);
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_coverage_before_range CHECK (coverage_before >= 0.0 AND coverage_before <= 1.0);
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_coverage_after_range CHECK (coverage_after >= 0.0 AND coverage_after <= 1.0);
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_coverage_improved CHECK (coverage_after >= coverage_before);
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_filled_lte_gaps CHECK (total_filled <= total_gaps OR total_gaps = 0);
ALTER TABLE time_series_gap_filler_service.gap_reports
    ADD CONSTRAINT chk_grp_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: time_series_gap_filler_service.gap_audit_log
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    entity_type TEXT NOT NULL,
    entity_id UUID,
    action TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance_hash TEXT,
    previous_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE time_series_gap_filler_service.gap_audit_log
    ADD CONSTRAINT chk_gal_action CHECK (action IN ('job_created', 'job_started', 'job_completed', 'job_failed', 'job_cancelled', 'gap_detected', 'frequency_analyzed', 'strategy_selected', 'fill_started', 'fill_completed', 'fill_failed', 'validation_started', 'validation_passed', 'validation_failed', 'calendar_created', 'calendar_updated', 'calendar_deleted', 'correlation_computed', 'report_generated', 'config_changed', 'export_generated', 'import_completed'));
ALTER TABLE time_series_gap_filler_service.gap_audit_log
    ADD CONSTRAINT chk_gal_entity_type CHECK (entity_type IN ('job', 'detection', 'frequency', 'fill', 'strategy', 'validation', 'calendar', 'correlation', 'report', 'config'));
ALTER TABLE time_series_gap_filler_service.gap_audit_log
    ADD CONSTRAINT chk_gal_action_not_empty CHECK (LENGTH(TRIM(action)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_audit_log
    ADD CONSTRAINT chk_gal_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);
ALTER TABLE time_series_gap_filler_service.gap_audit_log
    ADD CONSTRAINT chk_gal_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: time_series_gap_filler_service.gap_events (hypertable)
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.gap_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    event_type VARCHAR(50) NOT NULL,
    series_id TEXT,
    gap_count INTEGER,
    coverage DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('time_series_gap_filler_service.gap_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE time_series_gap_filler_service.gap_events
    ADD CONSTRAINT chk_ge_event_type CHECK (event_type IN ('detection_started', 'detection_completed', 'detection_failed', 'frequency_analyzed', 'strategy_selected', 'gaps_found', 'no_gaps_found', 'coverage_computed', 'progress_update', 'job_started', 'job_completed', 'job_failed'));
ALTER TABLE time_series_gap_filler_service.gap_events
    ADD CONSTRAINT chk_ge_gap_count_non_negative CHECK (gap_count IS NULL OR gap_count >= 0);
ALTER TABLE time_series_gap_filler_service.gap_events
    ADD CONSTRAINT chk_ge_coverage_range CHECK (coverage IS NULL OR (coverage >= 0.0 AND coverage <= 1.0));
ALTER TABLE time_series_gap_filler_service.gap_events
    ADD CONSTRAINT chk_ge_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 12: time_series_gap_filler_service.fill_events (hypertable)
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.fill_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    event_type VARCHAR(50) NOT NULL,
    series_id TEXT,
    method TEXT,
    fill_count INTEGER,
    avg_confidence DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('time_series_gap_filler_service.fill_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE time_series_gap_filler_service.fill_events
    ADD CONSTRAINT chk_fe_event_type CHECK (event_type IN ('fill_started', 'fill_completed', 'fill_failed', 'fill_validated', 'fill_rejected', 'strategy_applied', 'interpolation_completed', 'correlation_fill_completed', 'calendar_fill_completed', 'batch_fill_completed', 'progress_update'));
ALTER TABLE time_series_gap_filler_service.fill_events
    ADD CONSTRAINT chk_fe_fill_count_non_negative CHECK (fill_count IS NULL OR fill_count >= 0);
ALTER TABLE time_series_gap_filler_service.fill_events
    ADD CONSTRAINT chk_fe_avg_confidence_range CHECK (avg_confidence IS NULL OR (avg_confidence >= 0.0 AND avg_confidence <= 1.0));
ALTER TABLE time_series_gap_filler_service.fill_events
    ADD CONSTRAINT chk_fe_method CHECK (method IS NULL OR method IN ('linear', 'spline', 'locf', 'nocb', 'mean', 'median', 'seasonal', 'regression', 'correlation', 'calendar_aware', 'zero_fill', 'custom', 'auto'));
ALTER TABLE time_series_gap_filler_service.fill_events
    ADD CONSTRAINT chk_fe_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 13: time_series_gap_filler_service.validation_events (hypertable)
-- =============================================================================

CREATE TABLE time_series_gap_filler_service.validation_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    event_type VARCHAR(50) NOT NULL,
    fill_id UUID,
    status TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('time_series_gap_filler_service.validation_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE time_series_gap_filler_service.validation_events
    ADD CONSTRAINT chk_ve_event_type CHECK (event_type IN ('validation_started', 'validation_completed', 'validation_failed', 'plausibility_check', 'distribution_check', 'range_check', 'validation_passed', 'validation_warning', 'validation_rejected', 'batch_validation_completed', 'progress_update'));
ALTER TABLE time_series_gap_filler_service.validation_events
    ADD CONSTRAINT chk_ve_status CHECK (status IS NULL OR status IN ('pending', 'passed', 'failed', 'warning', 'skipped'));
ALTER TABLE time_series_gap_filler_service.validation_events
    ADD CONSTRAINT chk_ve_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

CREATE MATERIALIZED VIEW time_series_gap_filler_service.gap_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    COUNT(*) AS total_events,
    SUM(gap_count) AS total_gap_count,
    AVG(coverage) AS avg_coverage
FROM time_series_gap_filler_service.gap_events
WHERE time IS NOT NULL
GROUP BY bucket, tenant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('time_series_gap_filler_service.gap_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

CREATE MATERIALIZED VIEW time_series_gap_filler_service.fill_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    COUNT(*) AS total_events,
    SUM(fill_count) AS total_fill_count,
    AVG(avg_confidence) AS avg_fill_confidence
FROM time_series_gap_filler_service.fill_events
WHERE time IS NOT NULL
GROUP BY bucket, tenant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('time_series_gap_filler_service.fill_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- gap_filler_jobs indexes (16)
CREATE INDEX idx_gfj_tenant_id ON time_series_gap_filler_service.gap_filler_jobs(tenant_id);
CREATE INDEX idx_gfj_dataset_id ON time_series_gap_filler_service.gap_filler_jobs(dataset_id);
CREATE INDEX idx_gfj_series_id ON time_series_gap_filler_service.gap_filler_jobs(series_id);
CREATE INDEX idx_gfj_status ON time_series_gap_filler_service.gap_filler_jobs(status);
CREATE INDEX idx_gfj_strategy ON time_series_gap_filler_service.gap_filler_jobs(strategy);
CREATE INDEX idx_gfj_calendar_id ON time_series_gap_filler_service.gap_filler_jobs(calendar_id);
CREATE INDEX idx_gfj_created_at ON time_series_gap_filler_service.gap_filler_jobs(created_at DESC);
CREATE INDEX idx_gfj_updated_at ON time_series_gap_filler_service.gap_filler_jobs(updated_at DESC);
CREATE INDEX idx_gfj_completed_at ON time_series_gap_filler_service.gap_filler_jobs(completed_at DESC);
CREATE INDEX idx_gfj_tenant_status ON time_series_gap_filler_service.gap_filler_jobs(tenant_id, status);
CREATE INDEX idx_gfj_tenant_series ON time_series_gap_filler_service.gap_filler_jobs(tenant_id, series_id);
CREATE INDEX idx_gfj_tenant_dataset ON time_series_gap_filler_service.gap_filler_jobs(tenant_id, dataset_id);
CREATE INDEX idx_gfj_tenant_created ON time_series_gap_filler_service.gap_filler_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_gfj_status_strategy ON time_series_gap_filler_service.gap_filler_jobs(status, strategy);
CREATE INDEX idx_gfj_config ON time_series_gap_filler_service.gap_filler_jobs USING GIN (config);
CREATE INDEX idx_gfj_result ON time_series_gap_filler_service.gap_filler_jobs USING GIN (result);

-- gap_detections indexes (14)
CREATE INDEX idx_gd_tenant_id ON time_series_gap_filler_service.gap_detections(tenant_id);
CREATE INDEX idx_gd_job_id ON time_series_gap_filler_service.gap_detections(job_id);
CREATE INDEX idx_gd_series_id ON time_series_gap_filler_service.gap_detections(series_id);
CREATE INDEX idx_gd_total_gaps ON time_series_gap_filler_service.gap_detections(total_gaps DESC);
CREATE INDEX idx_gd_total_missing ON time_series_gap_filler_service.gap_detections(total_missing DESC);
CREATE INDEX idx_gd_coverage_ratio ON time_series_gap_filler_service.gap_detections(coverage_ratio);
CREATE INDEX idx_gd_provenance ON time_series_gap_filler_service.gap_detections(provenance_hash);
CREATE INDEX idx_gd_created_at ON time_series_gap_filler_service.gap_detections(created_at DESC);
CREATE INDEX idx_gd_tenant_job ON time_series_gap_filler_service.gap_detections(tenant_id, job_id);
CREATE INDEX idx_gd_tenant_series ON time_series_gap_filler_service.gap_detections(tenant_id, series_id);
CREATE INDEX idx_gd_job_series ON time_series_gap_filler_service.gap_detections(job_id, series_id);
CREATE INDEX idx_gd_tenant_created ON time_series_gap_filler_service.gap_detections(tenant_id, created_at DESC);
CREATE INDEX idx_gd_tenant_coverage ON time_series_gap_filler_service.gap_detections(tenant_id, coverage_ratio);
CREATE INDEX idx_gd_gaps ON time_series_gap_filler_service.gap_detections USING GIN (gaps);

-- gap_frequencies indexes (12)
CREATE INDEX idx_gf_tenant_id ON time_series_gap_filler_service.gap_frequencies(tenant_id);
CREATE INDEX idx_gf_series_id ON time_series_gap_filler_service.gap_frequencies(series_id);
CREATE INDEX idx_gf_detected_frequency ON time_series_gap_filler_service.gap_frequencies(detected_frequency);
CREATE INDEX idx_gf_confidence ON time_series_gap_filler_service.gap_frequencies(confidence DESC);
CREATE INDEX idx_gf_regularity_score ON time_series_gap_filler_service.gap_frequencies(regularity_score DESC);
CREATE INDEX idx_gf_median_interval ON time_series_gap_filler_service.gap_frequencies(median_interval_seconds);
CREATE INDEX idx_gf_created_at ON time_series_gap_filler_service.gap_frequencies(created_at DESC);
CREATE INDEX idx_gf_tenant_series ON time_series_gap_filler_service.gap_frequencies(tenant_id, series_id);
CREATE INDEX idx_gf_tenant_frequency ON time_series_gap_filler_service.gap_frequencies(tenant_id, detected_frequency);
CREATE INDEX idx_gf_tenant_created ON time_series_gap_filler_service.gap_frequencies(tenant_id, created_at DESC);
CREATE INDEX idx_gf_series_frequency ON time_series_gap_filler_service.gap_frequencies(series_id, detected_frequency);
CREATE INDEX idx_gf_series_confidence ON time_series_gap_filler_service.gap_frequencies(series_id, confidence DESC);

-- gap_fills indexes (14)
CREATE INDEX idx_gfl_tenant_id ON time_series_gap_filler_service.gap_fills(tenant_id);
CREATE INDEX idx_gfl_job_id ON time_series_gap_filler_service.gap_fills(job_id);
CREATE INDEX idx_gfl_gap_id ON time_series_gap_filler_service.gap_fills(gap_id);
CREATE INDEX idx_gfl_strategy ON time_series_gap_filler_service.gap_fills(strategy);
CREATE INDEX idx_gfl_fill_count ON time_series_gap_filler_service.gap_fills(fill_count DESC);
CREATE INDEX idx_gfl_avg_confidence ON time_series_gap_filler_service.gap_fills(avg_confidence DESC);
CREATE INDEX idx_gfl_provenance ON time_series_gap_filler_service.gap_fills(provenance_hash);
CREATE INDEX idx_gfl_created_at ON time_series_gap_filler_service.gap_fills(created_at DESC);
CREATE INDEX idx_gfl_tenant_job ON time_series_gap_filler_service.gap_fills(tenant_id, job_id);
CREATE INDEX idx_gfl_tenant_strategy ON time_series_gap_filler_service.gap_fills(tenant_id, strategy);
CREATE INDEX idx_gfl_job_strategy ON time_series_gap_filler_service.gap_fills(job_id, strategy);
CREATE INDEX idx_gfl_job_gap ON time_series_gap_filler_service.gap_fills(job_id, gap_id);
CREATE INDEX idx_gfl_tenant_created ON time_series_gap_filler_service.gap_fills(tenant_id, created_at DESC);
CREATE INDEX idx_gfl_filled_values ON time_series_gap_filler_service.gap_fills USING GIN (filled_values);

-- gap_strategies indexes (12)
CREATE INDEX idx_gs_tenant_id ON time_series_gap_filler_service.gap_strategies(tenant_id);
CREATE INDEX idx_gs_job_id ON time_series_gap_filler_service.gap_strategies(job_id);
CREATE INDEX idx_gs_gap_type ON time_series_gap_filler_service.gap_strategies(gap_type);
CREATE INDEX idx_gs_strategy ON time_series_gap_filler_service.gap_strategies(strategy);
CREATE INDEX idx_gs_created_at ON time_series_gap_filler_service.gap_strategies(created_at DESC);
CREATE INDEX idx_gs_tenant_job ON time_series_gap_filler_service.gap_strategies(tenant_id, job_id);
CREATE INDEX idx_gs_tenant_gap_type ON time_series_gap_filler_service.gap_strategies(tenant_id, gap_type);
CREATE INDEX idx_gs_tenant_strategy ON time_series_gap_filler_service.gap_strategies(tenant_id, strategy);
CREATE INDEX idx_gs_job_gap_type ON time_series_gap_filler_service.gap_strategies(job_id, gap_type);
CREATE INDEX idx_gs_job_strategy ON time_series_gap_filler_service.gap_strategies(job_id, strategy);
CREATE INDEX idx_gs_tenant_created ON time_series_gap_filler_service.gap_strategies(tenant_id, created_at DESC);
CREATE INDEX idx_gs_parameters ON time_series_gap_filler_service.gap_strategies USING GIN (parameters);

-- gap_validations indexes (12)
CREATE INDEX idx_gv_tenant_id ON time_series_gap_filler_service.gap_validations(tenant_id);
CREATE INDEX idx_gv_fill_id ON time_series_gap_filler_service.gap_validations(fill_id);
CREATE INDEX idx_gv_status ON time_series_gap_filler_service.gap_validations(status);
CREATE INDEX idx_gv_plausibility ON time_series_gap_filler_service.gap_validations(plausibility_passed);
CREATE INDEX idx_gv_distribution ON time_series_gap_filler_service.gap_validations(distribution_preserved);
CREATE INDEX idx_gv_range_check ON time_series_gap_filler_service.gap_validations(range_check_passed);
CREATE INDEX idx_gv_created_at ON time_series_gap_filler_service.gap_validations(created_at DESC);
CREATE INDEX idx_gv_tenant_status ON time_series_gap_filler_service.gap_validations(tenant_id, status);
CREATE INDEX idx_gv_tenant_fill ON time_series_gap_filler_service.gap_validations(tenant_id, fill_id);
CREATE INDEX idx_gv_tenant_created ON time_series_gap_filler_service.gap_validations(tenant_id, created_at DESC);
CREATE INDEX idx_gv_fill_status ON time_series_gap_filler_service.gap_validations(fill_id, status);
CREATE INDEX idx_gv_details ON time_series_gap_filler_service.gap_validations USING GIN (details);

-- gap_calendars indexes (12)
CREATE INDEX idx_gc_tenant_id ON time_series_gap_filler_service.gap_calendars(tenant_id);
CREATE INDEX idx_gc_name ON time_series_gap_filler_service.gap_calendars(name);
CREATE INDEX idx_gc_calendar_type ON time_series_gap_filler_service.gap_calendars(calendar_type);
CREATE INDEX idx_gc_fiscal_year ON time_series_gap_filler_service.gap_calendars(fiscal_year_start_month);
CREATE INDEX idx_gc_created_at ON time_series_gap_filler_service.gap_calendars(created_at DESC);
CREATE INDEX idx_gc_updated_at ON time_series_gap_filler_service.gap_calendars(updated_at DESC);
CREATE INDEX idx_gc_tenant_type ON time_series_gap_filler_service.gap_calendars(tenant_id, calendar_type);
CREATE INDEX idx_gc_tenant_created ON time_series_gap_filler_service.gap_calendars(tenant_id, created_at DESC);
CREATE INDEX idx_gc_business_days ON time_series_gap_filler_service.gap_calendars USING GIN (business_days);
CREATE INDEX idx_gc_holidays ON time_series_gap_filler_service.gap_calendars USING GIN (holidays);
CREATE INDEX idx_gc_custom_rules ON time_series_gap_filler_service.gap_calendars USING GIN (custom_rules);
CREATE INDEX idx_gc_tenant_name ON time_series_gap_filler_service.gap_calendars(tenant_id, name);

-- gap_correlations indexes (14)
CREATE INDEX idx_gcr_tenant_id ON time_series_gap_filler_service.gap_correlations(tenant_id);
CREATE INDEX idx_gcr_source_series ON time_series_gap_filler_service.gap_correlations(source_series_id);
CREATE INDEX idx_gcr_reference_series ON time_series_gap_filler_service.gap_correlations(reference_series_id);
CREATE INDEX idx_gcr_method ON time_series_gap_filler_service.gap_correlations(method);
CREATE INDEX idx_gcr_coefficient ON time_series_gap_filler_service.gap_correlations(coefficient DESC);
CREATE INDEX idx_gcr_p_value ON time_series_gap_filler_service.gap_correlations(p_value);
CREATE INDEX idx_gcr_lag ON time_series_gap_filler_service.gap_correlations(lag);
CREATE INDEX idx_gcr_sample_size ON time_series_gap_filler_service.gap_correlations(sample_size DESC);
CREATE INDEX idx_gcr_created_at ON time_series_gap_filler_service.gap_correlations(created_at DESC);
CREATE INDEX idx_gcr_tenant_source ON time_series_gap_filler_service.gap_correlations(tenant_id, source_series_id);
CREATE INDEX idx_gcr_tenant_reference ON time_series_gap_filler_service.gap_correlations(tenant_id, reference_series_id);
CREATE INDEX idx_gcr_tenant_method ON time_series_gap_filler_service.gap_correlations(tenant_id, method);
CREATE INDEX idx_gcr_source_reference ON time_series_gap_filler_service.gap_correlations(source_series_id, reference_series_id);
CREATE INDEX idx_gcr_tenant_created ON time_series_gap_filler_service.gap_correlations(tenant_id, created_at DESC);

-- gap_reports indexes (14)
CREATE INDEX idx_grp_tenant_id ON time_series_gap_filler_service.gap_reports(tenant_id);
CREATE INDEX idx_grp_job_id ON time_series_gap_filler_service.gap_reports(job_id);
CREATE INDEX idx_grp_format ON time_series_gap_filler_service.gap_reports(format);
CREATE INDEX idx_grp_total_gaps ON time_series_gap_filler_service.gap_reports(total_gaps DESC);
CREATE INDEX idx_grp_total_filled ON time_series_gap_filler_service.gap_reports(total_filled DESC);
CREATE INDEX idx_grp_total_validated ON time_series_gap_filler_service.gap_reports(total_validated DESC);
CREATE INDEX idx_grp_coverage_before ON time_series_gap_filler_service.gap_reports(coverage_before);
CREATE INDEX idx_grp_coverage_after ON time_series_gap_filler_service.gap_reports(coverage_after DESC);
CREATE INDEX idx_grp_created_at ON time_series_gap_filler_service.gap_reports(created_at DESC);
CREATE INDEX idx_grp_tenant_job ON time_series_gap_filler_service.gap_reports(tenant_id, job_id);
CREATE INDEX idx_grp_tenant_format ON time_series_gap_filler_service.gap_reports(tenant_id, format);
CREATE INDEX idx_grp_tenant_created ON time_series_gap_filler_service.gap_reports(tenant_id, created_at DESC);
CREATE INDEX idx_grp_job_format ON time_series_gap_filler_service.gap_reports(job_id, format);
CREATE INDEX idx_grp_report_data ON time_series_gap_filler_service.gap_reports USING GIN (report_data);

-- gap_audit_log indexes (16)
CREATE INDEX idx_gal_tenant_id ON time_series_gap_filler_service.gap_audit_log(tenant_id);
CREATE INDEX idx_gal_entity_type ON time_series_gap_filler_service.gap_audit_log(entity_type);
CREATE INDEX idx_gal_entity_id ON time_series_gap_filler_service.gap_audit_log(entity_id);
CREATE INDEX idx_gal_action ON time_series_gap_filler_service.gap_audit_log(action);
CREATE INDEX idx_gal_provenance ON time_series_gap_filler_service.gap_audit_log(provenance_hash);
CREATE INDEX idx_gal_previous_hash ON time_series_gap_filler_service.gap_audit_log(previous_hash);
CREATE INDEX idx_gal_created_at ON time_series_gap_filler_service.gap_audit_log(created_at DESC);
CREATE INDEX idx_gal_tenant_action ON time_series_gap_filler_service.gap_audit_log(tenant_id, action);
CREATE INDEX idx_gal_tenant_entity ON time_series_gap_filler_service.gap_audit_log(tenant_id, entity_type);
CREATE INDEX idx_gal_action_entity ON time_series_gap_filler_service.gap_audit_log(action, entity_type);
CREATE INDEX idx_gal_entity_type_id ON time_series_gap_filler_service.gap_audit_log(entity_type, entity_id);
CREATE INDEX idx_gal_tenant_created ON time_series_gap_filler_service.gap_audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_gal_tenant_provenance ON time_series_gap_filler_service.gap_audit_log(tenant_id, provenance_hash);
CREATE INDEX idx_gal_provenance_chain ON time_series_gap_filler_service.gap_audit_log(provenance_hash, previous_hash);
CREATE INDEX idx_gal_details ON time_series_gap_filler_service.gap_audit_log USING GIN (details);
CREATE INDEX idx_gal_tenant_entity_created ON time_series_gap_filler_service.gap_audit_log(tenant_id, entity_type, created_at DESC);

-- gap_events indexes (hypertable-aware) (8)
CREATE INDEX idx_ge_tenant_id ON time_series_gap_filler_service.gap_events(tenant_id, time DESC);
CREATE INDEX idx_ge_event_type ON time_series_gap_filler_service.gap_events(event_type, time DESC);
CREATE INDEX idx_ge_series_id ON time_series_gap_filler_service.gap_events(series_id, time DESC);
CREATE INDEX idx_ge_gap_count ON time_series_gap_filler_service.gap_events(gap_count, time DESC);
CREATE INDEX idx_ge_tenant_type ON time_series_gap_filler_service.gap_events(tenant_id, event_type, time DESC);
CREATE INDEX idx_ge_tenant_series ON time_series_gap_filler_service.gap_events(tenant_id, series_id, time DESC);
CREATE INDEX idx_ge_coverage ON time_series_gap_filler_service.gap_events(coverage, time DESC);
CREATE INDEX idx_ge_metadata ON time_series_gap_filler_service.gap_events USING GIN (metadata);

-- fill_events indexes (hypertable-aware) (8)
CREATE INDEX idx_fe_tenant_id ON time_series_gap_filler_service.fill_events(tenant_id, time DESC);
CREATE INDEX idx_fe_event_type ON time_series_gap_filler_service.fill_events(event_type, time DESC);
CREATE INDEX idx_fe_series_id ON time_series_gap_filler_service.fill_events(series_id, time DESC);
CREATE INDEX idx_fe_method ON time_series_gap_filler_service.fill_events(method, time DESC);
CREATE INDEX idx_fe_fill_count ON time_series_gap_filler_service.fill_events(fill_count, time DESC);
CREATE INDEX idx_fe_tenant_type ON time_series_gap_filler_service.fill_events(tenant_id, event_type, time DESC);
CREATE INDEX idx_fe_tenant_method ON time_series_gap_filler_service.fill_events(tenant_id, method, time DESC);
CREATE INDEX idx_fe_metadata ON time_series_gap_filler_service.fill_events USING GIN (metadata);

-- validation_events indexes (hypertable-aware) (8)
CREATE INDEX idx_ve_tenant_id ON time_series_gap_filler_service.validation_events(tenant_id, time DESC);
CREATE INDEX idx_ve_event_type ON time_series_gap_filler_service.validation_events(event_type, time DESC);
CREATE INDEX idx_ve_fill_id ON time_series_gap_filler_service.validation_events(fill_id, time DESC);
CREATE INDEX idx_ve_status ON time_series_gap_filler_service.validation_events(status, time DESC);
CREATE INDEX idx_ve_tenant_type ON time_series_gap_filler_service.validation_events(tenant_id, event_type, time DESC);
CREATE INDEX idx_ve_tenant_status ON time_series_gap_filler_service.validation_events(tenant_id, status, time DESC);
CREATE INDEX idx_ve_tenant_fill ON time_series_gap_filler_service.validation_events(tenant_id, fill_id, time DESC);
CREATE INDEX idx_ve_details ON time_series_gap_filler_service.validation_events USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE time_series_gap_filler_service.gap_filler_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY gfj_tenant_read ON time_series_gap_filler_service.gap_filler_jobs FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gfj_tenant_write ON time_series_gap_filler_service.gap_filler_jobs FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_detections ENABLE ROW LEVEL SECURITY;
CREATE POLICY gd_tenant_read ON time_series_gap_filler_service.gap_detections FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gd_tenant_write ON time_series_gap_filler_service.gap_detections FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_frequencies ENABLE ROW LEVEL SECURITY;
CREATE POLICY gf_tenant_read ON time_series_gap_filler_service.gap_frequencies FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gf_tenant_write ON time_series_gap_filler_service.gap_frequencies FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_fills ENABLE ROW LEVEL SECURITY;
CREATE POLICY gfl_tenant_read ON time_series_gap_filler_service.gap_fills FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gfl_tenant_write ON time_series_gap_filler_service.gap_fills FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_strategies ENABLE ROW LEVEL SECURITY;
CREATE POLICY gs_tenant_read ON time_series_gap_filler_service.gap_strategies FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gs_tenant_write ON time_series_gap_filler_service.gap_strategies FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_validations ENABLE ROW LEVEL SECURITY;
CREATE POLICY gv_tenant_read ON time_series_gap_filler_service.gap_validations FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gv_tenant_write ON time_series_gap_filler_service.gap_validations FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_calendars ENABLE ROW LEVEL SECURITY;
CREATE POLICY gc_tenant_read ON time_series_gap_filler_service.gap_calendars FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gc_tenant_write ON time_series_gap_filler_service.gap_calendars FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_correlations ENABLE ROW LEVEL SECURITY;
CREATE POLICY gcr_tenant_read ON time_series_gap_filler_service.gap_correlations FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gcr_tenant_write ON time_series_gap_filler_service.gap_correlations FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY grp_tenant_read ON time_series_gap_filler_service.gap_reports FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY grp_tenant_write ON time_series_gap_filler_service.gap_reports FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY gal_tenant_read ON time_series_gap_filler_service.gap_audit_log FOR SELECT USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY gal_tenant_write ON time_series_gap_filler_service.gap_audit_log FOR ALL USING (tenant_id = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE time_series_gap_filler_service.gap_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ge_tenant_read ON time_series_gap_filler_service.gap_events FOR SELECT USING (TRUE);
CREATE POLICY ge_tenant_write ON time_series_gap_filler_service.gap_events FOR ALL USING (TRUE);

ALTER TABLE time_series_gap_filler_service.fill_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_tenant_read ON time_series_gap_filler_service.fill_events FOR SELECT USING (TRUE);
CREATE POLICY fe_tenant_write ON time_series_gap_filler_service.fill_events FOR ALL USING (TRUE);

ALTER TABLE time_series_gap_filler_service.validation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ve_tenant_read ON time_series_gap_filler_service.validation_events FOR SELECT USING (TRUE);
CREATE POLICY ve_tenant_write ON time_series_gap_filler_service.validation_events FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA time_series_gap_filler_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA time_series_gap_filler_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA time_series_gap_filler_service TO greenlang_app;
GRANT SELECT ON time_series_gap_filler_service.gap_hourly_stats TO greenlang_app;
GRANT SELECT ON time_series_gap_filler_service.fill_hourly_stats TO greenlang_app;

GRANT USAGE ON SCHEMA time_series_gap_filler_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA time_series_gap_filler_service TO greenlang_readonly;
GRANT SELECT ON time_series_gap_filler_service.gap_hourly_stats TO greenlang_readonly;
GRANT SELECT ON time_series_gap_filler_service.fill_hourly_stats TO greenlang_readonly;

GRANT ALL ON SCHEMA time_series_gap_filler_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA time_series_gap_filler_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA time_series_gap_filler_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'gap_filler:jobs:read', 'gap_filler', 'jobs_read', 'View gap filler jobs and their progress'),
    (gen_random_uuid(), 'gap_filler:jobs:write', 'gap_filler', 'jobs_write', 'Create, start, cancel, and manage gap filler jobs'),
    (gen_random_uuid(), 'gap_filler:detections:read', 'gap_filler', 'detections_read', 'View gap detection results and coverage metrics'),
    (gen_random_uuid(), 'gap_filler:detections:write', 'gap_filler', 'detections_write', 'Run and manage gap detections'),
    (gen_random_uuid(), 'gap_filler:frequencies:read', 'gap_filler', 'frequencies_read', 'View frequency analysis results'),
    (gen_random_uuid(), 'gap_filler:frequencies:write', 'gap_filler', 'frequencies_write', 'Run and manage frequency analysis'),
    (gen_random_uuid(), 'gap_filler:fills:read', 'gap_filler', 'fills_read', 'View gap fill results and filled values'),
    (gen_random_uuid(), 'gap_filler:fills:write', 'gap_filler', 'fills_write', 'Create and manage gap fills'),
    (gen_random_uuid(), 'gap_filler:calendars:read', 'gap_filler', 'calendars_read', 'View business calendar definitions'),
    (gen_random_uuid(), 'gap_filler:calendars:write', 'gap_filler', 'calendars_write', 'Create, update, and manage business calendars'),
    (gen_random_uuid(), 'gap_filler:correlations:read', 'gap_filler', 'correlations_read', 'View cross-series correlation analysis results'),
    (gen_random_uuid(), 'gap_filler:correlations:write', 'gap_filler', 'correlations_write', 'Run and manage cross-series correlation analysis'),
    (gen_random_uuid(), 'gap_filler:validations:read', 'gap_filler', 'validations_read', 'View fill validation results'),
    (gen_random_uuid(), 'gap_filler:validations:write', 'gap_filler', 'validations_write', 'Run and manage fill validations'),
    (gen_random_uuid(), 'gap_filler:audit:read', 'gap_filler', 'audit_read', 'View gap filler audit log entries and provenance chains'),
    (gen_random_uuid(), 'gap_filler:admin', 'gap_filler', 'admin', 'Gap filler service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('time_series_gap_filler_service.gap_events', INTERVAL '90 days');
SELECT add_retention_policy('time_series_gap_filler_service.fill_events', INTERVAL '90 days');
SELECT add_retention_policy('time_series_gap_filler_service.validation_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE time_series_gap_filler_service.gap_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('time_series_gap_filler_service.gap_events', INTERVAL '7 days');

ALTER TABLE time_series_gap_filler_service.fill_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('time_series_gap_filler_service.fill_events', INTERVAL '7 days');

ALTER TABLE time_series_gap_filler_service.validation_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('time_series_gap_filler_service.validation_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Time Series Gap Filler Agent (GL-DATA-X-017)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-017', 'Time Series Gap Filler Agent',
 'Intelligent time-series gap detection and filling engine for GreenLang Climate OS. Performs automatic gap detection (missing timestamps, irregular intervals, calendar-aware gaps). Analyzes frequency (auto-detection, regularity scoring, median interval computation). Applies intelligent fill strategies (linear interpolation, spline, LOCF/NOCB, seasonal decomposition, regression-based, correlation-based, calendar-aware). Supports multi-series correlation analysis (Pearson, Spearman, Kendall, cross-correlation with lag detection). Validates fills (plausibility checks, distribution preservation, range enforcement). Provides business calendar support (fiscal calendars, holidays, custom rules). Generates comprehensive reports (coverage before/after, strategy breakdown, validation summary). SHA-256 provenance chains for zero-hallucination audit trail.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/time-series-gap-filler', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-017', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/time-series-gap-filler-service", "tag": "1.0.0", "port": 8000}'::jsonb,
 '{"time-series", "gap-filling", "interpolation", "data-quality", "frequency-analysis", "calendar-aware", "correlation"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "agriculture", "utilities"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2')
ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES
('GL-DATA-X-017', '1.0.0', 'gap_detection', 'detection', 'Detect gaps in time-series data including missing timestamps, irregular intervals, and calendar-aware gaps. Computes coverage ratio, gap intervals, expected vs actual counts.', '{"dataset", "series_id", "time_column", "config"}', '{"detections", "coverage", "gap_intervals", "summary"}', '{"detection_modes": ["missing_timestamp", "irregular_interval", "calendar_aware"], "tolerance_pct": 0.1, "min_gap_size": 1}'::jsonb),
('GL-DATA-X-017', '1.0.0', 'frequency_analysis', 'analysis', 'Auto-detect frequency of time-series data with confidence scoring and regularity analysis.', '{"dataset", "series_id", "time_column"}', '{"frequency", "confidence", "regularity_score", "interval_stats"}', '{"min_samples": 10, "frequency_candidates": ["1min", "5min", "15min", "1h", "1d", "1w", "1M", "1Q", "1Y"], "confidence_threshold": 0.7}'::jsonb),
('GL-DATA-X-017', '1.0.0', 'gap_filling', 'processing', 'Fill detected gaps using intelligent strategies: linear, spline, LOCF, NOCB, mean, median, seasonal, regression, correlation, calendar-aware.', '{"detections", "strategy", "config"}', '{"fills", "filled_values", "confidence_scores", "summary"}', '{"strategies": ["linear", "spline", "locf", "nocb", "mean", "median", "seasonal", "regression", "correlation", "calendar_aware", "zero_fill"], "auto_strategy": true}'::jsonb),
('GL-DATA-X-017', '1.0.0', 'fill_validation', 'validation', 'Validate filled values through plausibility checks, distribution preservation analysis, and range enforcement.', '{"fills", "original_data", "config"}', '{"validations", "pass_rate", "warnings", "summary"}', '{"checks": ["plausibility", "distribution", "range"], "range_factor": 3.0, "distribution_threshold": 0.05}'::jsonb),
('GL-DATA-X-017', '1.0.0', 'correlation_analysis', 'analysis', 'Analyze cross-series correlations for correlation-based gap filling with Pearson, Spearman, Kendall, and cross-correlation.', '{"source_series", "reference_series", "config"}', '{"correlations", "best_references", "lag_analysis"}', '{"methods": ["pearson", "spearman", "kendall", "cross_correlation"], "min_correlation": 0.5, "max_lag": 10}'::jsonb),
('GL-DATA-X-017', '1.0.0', 'calendar_management', 'configuration', 'Create and manage business calendars for calendar-aware gap filling: fiscal, academic, retail, manufacturing.', '{"calendar_definition", "config"}', '{"calendar", "business_days", "holidays", "rules"}', '{"calendar_types": ["business", "fiscal", "academic", "retail", "manufacturing", "custom", "iso_8601"]}'::jsonb),
('GL-DATA-X-017', '1.0.0', 'gap_reporting', 'reporting', 'Generate comprehensive gap fill reports with coverage before/after, strategy breakdown, validation summary.', '{"job_id", "format", "config"}', '{"report", "coverage_delta", "strategy_summary", "recommendations"}', '{"formats": ["json", "csv", "markdown", "html", "pdf"], "include_filled_values": true}'::jsonb)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES
('GL-DATA-X-017', 'GL-FOUND-X-002', '>=1.0.0', false, 'Schema validation for job configs, calendar definitions, and fill parameters'),
('GL-DATA-X-017', 'GL-FOUND-X-007', '>=1.0.0', false, 'Agent version and capability lookup for pipeline orchestration'),
('GL-DATA-X-017', 'GL-FOUND-X-006', '>=1.0.0', false, 'Access control enforcement for gap filler jobs and results'),
('GL-DATA-X-017', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for gap detection, fill rates, and coverage'),
('GL-DATA-X-017', 'GL-FOUND-X-005', '>=1.0.0', true, 'Provenance and audit trail registration with citation service'),
('GL-DATA-X-017', 'GL-FOUND-X-008', '>=1.0.0', true, 'Reproducibility verification for fill results'),
('GL-DATA-X-017', 'GL-FOUND-X-009', '>=1.0.0', true, 'QA Test Harness zero-hallucination verification'),
('GL-DATA-X-017', 'GL-DATA-X-013', '>=1.0.0', true, 'Data quality profiling for temporal pattern identification'),
('GL-DATA-X-017', 'GL-DATA-X-015', '>=1.0.0', true, 'Missing value imputation for non-temporal columns'),
('GL-DATA-X-017', 'GL-DATA-X-016', '>=1.0.0', true, 'Outlier detection on filled values to prevent anomaly introduction')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-017', 'Time Series Gap Filler Agent',
 'Intelligent time-series gap detection and filling engine. Gap detection (missing timestamps/irregular intervals/calendar-aware). Frequency analysis (auto-detection/regularity scoring). Fill strategies (linear/spline/LOCF/NOCB/seasonal/regression/correlation/calendar-aware). Correlation analysis (Pearson/Spearman/Kendall). Fill validation (plausibility/distribution/range). Business calendars. SHA-256 provenance chains.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA time_series_gap_filler_service IS 'Time Series Gap Filler Agent (AGENT-DATA-014) - gap detection, frequency analysis, intelligent fill strategies, correlation analysis, fill validation, business calendar support, provenance chains';
COMMENT ON TABLE time_series_gap_filler_service.gap_filler_jobs IS 'Job tracking for gap filling runs with dataset, series, status, strategy, frequency hint, calendar ref, config, result, timing';
COMMENT ON TABLE time_series_gap_filler_service.gap_detections IS 'Gap detection results per series: total gaps, missing count, expected count, coverage ratio, gap intervals, provenance hash';
COMMENT ON TABLE time_series_gap_filler_service.gap_frequencies IS 'Frequency analysis per series: detected frequency, confidence, regularity score, median interval, expected/actual counts';
COMMENT ON TABLE time_series_gap_filler_service.gap_fills IS 'Fill results per gap: strategy, fill count, average confidence, filled values, provenance hash';
COMMENT ON TABLE time_series_gap_filler_service.gap_strategies IS 'Strategy selection per gap type: gap classification, chosen strategy, parameters, justification';
COMMENT ON TABLE time_series_gap_filler_service.gap_validations IS 'Validation results per fill: status, plausibility, distribution preservation, range check, diagnostics';
COMMENT ON TABLE time_series_gap_filler_service.gap_calendars IS 'Business calendar definitions: name, type, business days, holidays, fiscal year start, custom rules';
COMMENT ON TABLE time_series_gap_filler_service.gap_correlations IS 'Cross-series correlation: source/reference series, method, coefficient, p-value, lag, sample size';
COMMENT ON TABLE time_series_gap_filler_service.gap_reports IS 'Gap fill summary reports: format, gaps/filled/validated totals, coverage before/after, report data';
COMMENT ON TABLE time_series_gap_filler_service.gap_audit_log IS 'Audit trail: entity type/ID, action, details, provenance hash, previous hash chain';
COMMENT ON TABLE time_series_gap_filler_service.gap_events IS 'TimescaleDB hypertable: gap detection lifecycle events (7-day chunks, 90-day retention)';
COMMENT ON TABLE time_series_gap_filler_service.fill_events IS 'TimescaleDB hypertable: fill events with method, count, confidence (7-day chunks, 90-day retention)';
COMMENT ON TABLE time_series_gap_filler_service.validation_events IS 'TimescaleDB hypertable: validation events with fill_id, status (7-day chunks, 90-day retention)';
COMMENT ON MATERIALIZED VIEW time_series_gap_filler_service.gap_hourly_stats IS 'Continuous aggregate: hourly gap stats by tenant (total events, gap count, avg coverage)';
COMMENT ON MATERIALIZED VIEW time_series_gap_filler_service.fill_hourly_stats IS 'Continuous aggregate: hourly fill stats by tenant (total events, fill count, avg confidence)';
