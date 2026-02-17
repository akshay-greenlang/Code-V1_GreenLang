-- =============================================================================
-- V045: Cross-Source Reconciliation Service Schema
-- =============================================================================
-- Component: AGENT-DATA-015 (Cross-Source Reconciliation Agent)
-- Agent ID:  GL-DATA-X-018
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Cross-Source Reconciliation Agent (GL-DATA-X-018) with capabilities for
-- multi-source data reconciliation (source registration, schema mapping,
-- column normalization), record matching (exact, fuzzy, composite,
-- configurable confidence thresholds), field-level comparison (absolute
-- and relative tolerance, numeric/text/date type awareness), discrepancy
-- detection (type classification, severity scoring, deviation analysis),
-- resolution workflows (priority-based, credibility-weighted, manual
-- review, auto-resolve), golden record assembly (best-of-breed field
-- selection, per-field source attribution, composite confidence scoring),
-- reconciliation reporting (match rates, discrepancy breakdown, resolution
-- summary, unresolved tracking), and full provenance chain tracking with
-- SHA-256 hashes for zero-hallucination audit trails.
-- =============================================================================
-- Tables (10):
--   1. reconciliation_jobs           - Job tracking
--   2. reconciliation_sources        - Registered data sources
--   3. reconciliation_schema_maps    - Column mapping rules
--   4. reconciliation_matches        - Matched record pairs
--   5. reconciliation_comparisons    - Field-level comparison results
--   6. reconciliation_discrepancies  - Detected discrepancies
--   7. reconciliation_resolutions    - Resolution decisions
--   8. reconciliation_golden_records - Assembled golden records
--   9. reconciliation_reports        - Generated reports
--  10. reconciliation_audit_log      - Audit trail
--
-- Hypertables (3):
--  11. reconciliation_events         - Reconciliation event time-series (hypertable)
--  12. comparison_events             - Comparison event time-series (hypertable)
--  13. resolution_events             - Resolution event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. reconciliation_hourly_stats   - Hourly reconciliation stats
--   2. discrepancy_hourly_stats      - Hourly discrepancy stats
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-018.
-- Previous: V044__time_series_gap_filler_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS cross_source_reconciliation_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION cross_source_reconciliation_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: cross_source_reconciliation_service.reconciliation_jobs
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    job_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    source_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    match_count INTEGER DEFAULT 0,
    discrepancy_count INTEGER DEFAULT 0,
    golden_record_count INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs
    ADD CONSTRAINT chk_rj_status CHECK (status IN ('pending', 'running', 'matching', 'comparing', 'resolving', 'assembling', 'completed', 'failed', 'cancelled'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs
    ADD CONSTRAINT chk_rj_job_name_not_empty CHECK (LENGTH(TRIM(job_name)) > 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs
    ADD CONSTRAINT chk_rj_match_count_non_negative CHECK (match_count IS NULL OR match_count >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs
    ADD CONSTRAINT chk_rj_discrepancy_count_non_negative CHECK (discrepancy_count IS NULL OR discrepancy_count >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs
    ADD CONSTRAINT chk_rj_golden_record_count_non_negative CHECK (golden_record_count IS NULL OR golden_record_count >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs
    ADD CONSTRAINT chk_rj_completed_after_started CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at);
ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs
    ADD CONSTRAINT chk_rj_started_after_created CHECK (started_at IS NULL OR started_at >= created_at);

CREATE TRIGGER trg_rj_updated_at
    BEFORE UPDATE ON cross_source_reconciliation_service.reconciliation_jobs
    FOR EACH ROW EXECUTE FUNCTION cross_source_reconciliation_service.set_updated_at();

-- =============================================================================
-- Table 2: cross_source_reconciliation_service.reconciliation_sources
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 50,
    credibility_score DOUBLE PRECISION DEFAULT 0.5,
    schema_info JSONB DEFAULT '{}'::jsonb,
    refresh_cadence VARCHAR(50),
    description TEXT,
    tags JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_sources
    ADD CONSTRAINT chk_rs_priority_range CHECK (priority >= 1 AND priority <= 100);
ALTER TABLE cross_source_reconciliation_service.reconciliation_sources
    ADD CONSTRAINT chk_rs_credibility_range CHECK (credibility_score IS NULL OR (credibility_score >= 0.0 AND credibility_score <= 1.0));
ALTER TABLE cross_source_reconciliation_service.reconciliation_sources
    ADD CONSTRAINT chk_rs_source_type CHECK (source_type IN ('erp', 'csv', 'excel', 'api', 'database', 'pdf', 'questionnaire', 'satellite', 'gis', 'manual', 'calculated', 'external'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_sources
    ADD CONSTRAINT chk_rs_status CHECK (status IN ('active', 'inactive', 'deprecated', 'error'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_sources
    ADD CONSTRAINT chk_rs_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_sources
    ADD CONSTRAINT chk_rs_refresh_cadence CHECK (refresh_cadence IS NULL OR refresh_cadence IN ('realtime', 'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'annually', 'on_demand'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_sources
    ADD CONSTRAINT uq_rs_tenant_name UNIQUE (tenant_id, name);

CREATE TRIGGER trg_rs_updated_at
    BEFORE UPDATE ON cross_source_reconciliation_service.reconciliation_sources
    FOR EACH ROW EXECUTE FUNCTION cross_source_reconciliation_service.set_updated_at();

-- =============================================================================
-- Table 3: cross_source_reconciliation_service.reconciliation_schema_maps
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_schema_maps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL,
    source_column VARCHAR(255) NOT NULL,
    canonical_column VARCHAR(255) NOT NULL,
    transform VARCHAR(255),
    unit_from VARCHAR(50),
    unit_to VARCHAR(50),
    date_format VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_schema_maps
    ADD CONSTRAINT fk_rsm_source_id FOREIGN KEY (source_id) REFERENCES cross_source_reconciliation_service.reconciliation_sources(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_schema_maps
    ADD CONSTRAINT chk_rsm_source_column_not_empty CHECK (LENGTH(TRIM(source_column)) > 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_schema_maps
    ADD CONSTRAINT chk_rsm_canonical_column_not_empty CHECK (LENGTH(TRIM(canonical_column)) > 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_schema_maps
    ADD CONSTRAINT uq_rsm_source_column UNIQUE (source_id, source_column);

-- =============================================================================
-- Table 4: cross_source_reconciliation_service.reconciliation_matches
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    job_id UUID,
    source_a_id UUID NOT NULL,
    source_b_id UUID NOT NULL,
    entity_id VARCHAR(255),
    period VARCHAR(50),
    confidence DOUBLE PRECISION NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'matched',
    matched_fields JSONB DEFAULT '[]'::jsonb,
    source_a_record JSONB,
    source_b_record JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_matches
    ADD CONSTRAINT fk_rm_job_id FOREIGN KEY (job_id) REFERENCES cross_source_reconciliation_service.reconciliation_jobs(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_matches
    ADD CONSTRAINT chk_rm_confidence_range CHECK (confidence >= 0.0 AND confidence <= 1.0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_matches
    ADD CONSTRAINT chk_rm_strategy CHECK (strategy IN ('exact', 'fuzzy', 'composite', 'key_based', 'temporal', 'hierarchical', 'manual'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_matches
    ADD CONSTRAINT chk_rm_status CHECK (status IN ('matched', 'partial', 'unmatched', 'conflict', 'review_needed', 'confirmed'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_matches
    ADD CONSTRAINT chk_rm_different_sources CHECK (source_a_id != source_b_id);

-- =============================================================================
-- Table 5: cross_source_reconciliation_service.reconciliation_comparisons
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID,
    field_name VARCHAR(255) NOT NULL,
    field_type VARCHAR(50),
    source_a_value TEXT,
    source_b_value TEXT,
    absolute_diff DOUBLE PRECISION,
    relative_diff_pct DOUBLE PRECISION,
    tolerance_abs DOUBLE PRECISION,
    tolerance_pct DOUBLE PRECISION,
    result VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_comparisons
    ADD CONSTRAINT fk_rc_match_id FOREIGN KEY (match_id) REFERENCES cross_source_reconciliation_service.reconciliation_matches(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_comparisons
    ADD CONSTRAINT chk_rc_field_name_not_empty CHECK (LENGTH(TRIM(field_name)) > 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_comparisons
    ADD CONSTRAINT chk_rc_field_type CHECK (field_type IS NULL OR field_type IN ('numeric', 'text', 'date', 'boolean', 'currency', 'percentage', 'unit_value', 'enum', 'json'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_comparisons
    ADD CONSTRAINT chk_rc_result CHECK (result IN ('match', 'within_tolerance', 'mismatch', 'missing_a', 'missing_b', 'type_mismatch', 'not_comparable'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_comparisons
    ADD CONSTRAINT chk_rc_tolerance_abs_non_negative CHECK (tolerance_abs IS NULL OR tolerance_abs >= 0.0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_comparisons
    ADD CONSTRAINT chk_rc_tolerance_pct_non_negative CHECK (tolerance_pct IS NULL OR tolerance_pct >= 0.0);

-- =============================================================================
-- Table 6: cross_source_reconciliation_service.reconciliation_discrepancies
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_discrepancies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    match_id UUID,
    job_id UUID,
    field_name VARCHAR(255),
    discrepancy_type VARCHAR(50) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    source_a_value TEXT,
    source_b_value TEXT,
    deviation_pct DOUBLE PRECISION,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_discrepancies
    ADD CONSTRAINT fk_rd_match_id FOREIGN KEY (match_id) REFERENCES cross_source_reconciliation_service.reconciliation_matches(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_discrepancies
    ADD CONSTRAINT fk_rd_job_id FOREIGN KEY (job_id) REFERENCES cross_source_reconciliation_service.reconciliation_jobs(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_discrepancies
    ADD CONSTRAINT chk_rd_discrepancy_type CHECK (discrepancy_type IN ('value_mismatch', 'unit_mismatch', 'missing_field', 'type_conflict', 'format_difference', 'temporal_gap', 'aggregation_error', 'rounding_error', 'currency_mismatch', 'scope_mismatch'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_discrepancies
    ADD CONSTRAINT chk_rd_severity CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info'));

-- =============================================================================
-- Table 7: cross_source_reconciliation_service.reconciliation_resolutions
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_resolutions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discrepancy_id UUID,
    job_id UUID,
    strategy VARCHAR(50) NOT NULL,
    winning_source_id UUID,
    resolved_value TEXT,
    confidence DOUBLE PRECISION,
    justification TEXT,
    reviewer VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'resolved',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_resolutions
    ADD CONSTRAINT fk_rr_discrepancy_id FOREIGN KEY (discrepancy_id) REFERENCES cross_source_reconciliation_service.reconciliation_discrepancies(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_resolutions
    ADD CONSTRAINT fk_rr_job_id FOREIGN KEY (job_id) REFERENCES cross_source_reconciliation_service.reconciliation_jobs(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_resolutions
    ADD CONSTRAINT chk_rr_strategy CHECK (strategy IN ('priority_based', 'credibility_weighted', 'most_recent', 'manual_override', 'average', 'median', 'conservative', 'liberal', 'consensus', 'escalated'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_resolutions
    ADD CONSTRAINT chk_rr_status CHECK (status IN ('resolved', 'pending_review', 'escalated', 'overridden', 'rejected', 'deferred'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_resolutions
    ADD CONSTRAINT chk_rr_confidence_range CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0));

-- =============================================================================
-- Table 8: cross_source_reconciliation_service.reconciliation_golden_records
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_golden_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    job_id UUID,
    entity_id VARCHAR(255) NOT NULL,
    period VARCHAR(50),
    fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    field_sources JSONB NOT NULL DEFAULT '{}'::jsonb,
    field_confidences JSONB NOT NULL DEFAULT '{}'::jsonb,
    total_confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_golden_records
    ADD CONSTRAINT fk_rgr_job_id FOREIGN KEY (job_id) REFERENCES cross_source_reconciliation_service.reconciliation_jobs(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_golden_records
    ADD CONSTRAINT chk_rgr_entity_id_not_empty CHECK (LENGTH(TRIM(entity_id)) > 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_golden_records
    ADD CONSTRAINT chk_rgr_total_confidence_range CHECK (total_confidence IS NULL OR (total_confidence >= 0.0 AND total_confidence <= 1.0));

-- =============================================================================
-- Table 9: cross_source_reconciliation_service.reconciliation_reports
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    job_id UUID,
    report_type VARCHAR(50) DEFAULT 'reconciliation',
    total_records INTEGER,
    matched_records INTEGER,
    discrepancies_found INTEGER,
    discrepancies_resolved INTEGER,
    golden_records_created INTEGER,
    unresolved_count INTEGER,
    summary JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT fk_rrp_job_id FOREIGN KEY (job_id) REFERENCES cross_source_reconciliation_service.reconciliation_jobs(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT chk_rrp_report_type CHECK (report_type IS NULL OR report_type IN ('reconciliation', 'discrepancy', 'resolution', 'golden_record', 'audit', 'executive'));
ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT chk_rrp_total_records_non_negative CHECK (total_records IS NULL OR total_records >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT chk_rrp_matched_records_non_negative CHECK (matched_records IS NULL OR matched_records >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT chk_rrp_discrepancies_found_non_negative CHECK (discrepancies_found IS NULL OR discrepancies_found >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT chk_rrp_discrepancies_resolved_non_negative CHECK (discrepancies_resolved IS NULL OR discrepancies_resolved >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT chk_rrp_golden_records_created_non_negative CHECK (golden_records_created IS NULL OR golden_records_created >= 0);
ALTER TABLE cross_source_reconciliation_service.reconciliation_reports
    ADD CONSTRAINT chk_rrp_unresolved_count_non_negative CHECK (unresolved_count IS NULL OR unresolved_count >= 0);

-- =============================================================================
-- Table 10: cross_source_reconciliation_service.reconciliation_audit_log
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    job_id UUID,
    event_type VARCHAR(100) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

ALTER TABLE cross_source_reconciliation_service.reconciliation_audit_log
    ADD CONSTRAINT fk_ral_job_id FOREIGN KEY (job_id) REFERENCES cross_source_reconciliation_service.reconciliation_jobs(id) ON DELETE CASCADE;
ALTER TABLE cross_source_reconciliation_service.reconciliation_audit_log
    ADD CONSTRAINT chk_ral_event_type CHECK (event_type IN (
        'job_created', 'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'source_registered', 'source_updated', 'source_deactivated',
        'schema_map_created', 'schema_map_updated', 'schema_map_deleted',
        'matching_started', 'matching_completed', 'match_found', 'match_confirmed',
        'comparison_started', 'comparison_completed', 'comparison_failed',
        'discrepancy_detected', 'discrepancy_escalated',
        'resolution_started', 'resolution_completed', 'resolution_overridden',
        'golden_record_created', 'golden_record_updated',
        'report_generated', 'config_changed', 'export_generated', 'import_completed'
    ));
ALTER TABLE cross_source_reconciliation_service.reconciliation_audit_log
    ADD CONSTRAINT chk_ral_event_type_not_empty CHECK (LENGTH(TRIM(event_type)) > 0);

-- =============================================================================
-- Table 11: cross_source_reconciliation_service.reconciliation_events (hypertable)
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.reconciliation_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    job_id UUID,
    event_type VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64),
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('cross_source_reconciliation_service.reconciliation_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE cross_source_reconciliation_service.reconciliation_events
    ADD CONSTRAINT chk_re_event_type CHECK (event_type IN (
        'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'matching_started', 'matching_completed', 'matching_failed',
        'comparing_started', 'comparing_completed', 'comparing_failed',
        'resolving_started', 'resolving_completed', 'resolving_failed',
        'assembling_started', 'assembling_completed', 'assembling_failed',
        'progress_update'
    ));

-- =============================================================================
-- Table 12: cross_source_reconciliation_service.comparison_events (hypertable)
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.comparison_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    match_id UUID,
    field_name VARCHAR(255),
    result VARCHAR(50),
    diff_pct DOUBLE PRECISION,
    provenance_hash VARCHAR(64),
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('cross_source_reconciliation_service.comparison_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE cross_source_reconciliation_service.comparison_events
    ADD CONSTRAINT chk_ce_result CHECK (result IS NULL OR result IN ('match', 'within_tolerance', 'mismatch', 'missing_a', 'missing_b', 'type_mismatch', 'not_comparable'));

-- =============================================================================
-- Table 13: cross_source_reconciliation_service.resolution_events (hypertable)
-- =============================================================================

CREATE TABLE cross_source_reconciliation_service.resolution_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    job_id UUID,
    discrepancy_id UUID,
    strategy VARCHAR(50),
    confidence DOUBLE PRECISION,
    provenance_hash VARCHAR(64),
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('cross_source_reconciliation_service.resolution_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE cross_source_reconciliation_service.resolution_events
    ADD CONSTRAINT chk_rese_strategy CHECK (strategy IS NULL OR strategy IN ('priority_based', 'credibility_weighted', 'most_recent', 'manual_override', 'average', 'median', 'conservative', 'liberal', 'consensus', 'escalated'));
ALTER TABLE cross_source_reconciliation_service.resolution_events
    ADD CONSTRAINT chk_rese_confidence_range CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0));

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- reconciliation_hourly_stats: hourly job count, match count, discrepancy count, resolution count, avg confidence
CREATE MATERIALIZED VIEW cross_source_reconciliation_service.reconciliation_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE event_type IN ('matching_completed')) AS match_event_count,
    COUNT(*) FILTER (WHERE event_type IN ('comparing_completed')) AS comparison_event_count,
    COUNT(*) FILTER (WHERE event_type IN ('resolving_completed')) AS resolution_event_count,
    COUNT(*) FILTER (WHERE event_type IN ('job_completed')) AS job_completed_count
FROM cross_source_reconciliation_service.reconciliation_events
WHERE time IS NOT NULL
GROUP BY bucket, tenant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('cross_source_reconciliation_service.reconciliation_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- discrepancy_hourly_stats: hourly discrepancy count by result, avg diff_pct
CREATE MATERIALIZED VIEW cross_source_reconciliation_service.discrepancy_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    COUNT(*) AS total_comparisons,
    COUNT(*) FILTER (WHERE result = 'match') AS match_count,
    COUNT(*) FILTER (WHERE result = 'mismatch') AS mismatch_count,
    COUNT(*) FILTER (WHERE result = 'within_tolerance') AS within_tolerance_count,
    COUNT(*) FILTER (WHERE result IN ('missing_a', 'missing_b')) AS missing_count,
    AVG(diff_pct) AS avg_deviation_pct
FROM cross_source_reconciliation_service.comparison_events
WHERE time IS NOT NULL
GROUP BY bucket, tenant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('cross_source_reconciliation_service.discrepancy_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- reconciliation_jobs indexes (14)
CREATE INDEX idx_rj_tenant_id ON cross_source_reconciliation_service.reconciliation_jobs(tenant_id);
CREATE INDEX idx_rj_status ON cross_source_reconciliation_service.reconciliation_jobs(status);
CREATE INDEX idx_rj_job_name ON cross_source_reconciliation_service.reconciliation_jobs(job_name);
CREATE INDEX idx_rj_created_at ON cross_source_reconciliation_service.reconciliation_jobs(created_at DESC);
CREATE INDEX idx_rj_updated_at ON cross_source_reconciliation_service.reconciliation_jobs(updated_at DESC);
CREATE INDEX idx_rj_started_at ON cross_source_reconciliation_service.reconciliation_jobs(started_at DESC);
CREATE INDEX idx_rj_completed_at ON cross_source_reconciliation_service.reconciliation_jobs(completed_at DESC);
CREATE INDEX idx_rj_provenance ON cross_source_reconciliation_service.reconciliation_jobs(provenance_hash);
CREATE INDEX idx_rj_tenant_status ON cross_source_reconciliation_service.reconciliation_jobs(tenant_id, status);
CREATE INDEX idx_rj_tenant_created ON cross_source_reconciliation_service.reconciliation_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_rj_tenant_name ON cross_source_reconciliation_service.reconciliation_jobs(tenant_id, job_name);
CREATE INDEX idx_rj_status_created ON cross_source_reconciliation_service.reconciliation_jobs(status, created_at DESC);
CREATE INDEX idx_rj_source_ids ON cross_source_reconciliation_service.reconciliation_jobs USING GIN (source_ids);
CREATE INDEX idx_rj_config ON cross_source_reconciliation_service.reconciliation_jobs USING GIN (config);

-- reconciliation_sources indexes (14)
CREATE INDEX idx_rs_tenant_id ON cross_source_reconciliation_service.reconciliation_sources(tenant_id);
CREATE INDEX idx_rs_name ON cross_source_reconciliation_service.reconciliation_sources(name);
CREATE INDEX idx_rs_source_type ON cross_source_reconciliation_service.reconciliation_sources(source_type);
CREATE INDEX idx_rs_priority ON cross_source_reconciliation_service.reconciliation_sources(priority DESC);
CREATE INDEX idx_rs_credibility ON cross_source_reconciliation_service.reconciliation_sources(credibility_score DESC);
CREATE INDEX idx_rs_status ON cross_source_reconciliation_service.reconciliation_sources(status);
CREATE INDEX idx_rs_created_at ON cross_source_reconciliation_service.reconciliation_sources(created_at DESC);
CREATE INDEX idx_rs_updated_at ON cross_source_reconciliation_service.reconciliation_sources(updated_at DESC);
CREATE INDEX idx_rs_tenant_type ON cross_source_reconciliation_service.reconciliation_sources(tenant_id, source_type);
CREATE INDEX idx_rs_tenant_status ON cross_source_reconciliation_service.reconciliation_sources(tenant_id, status);
CREATE INDEX idx_rs_tenant_created ON cross_source_reconciliation_service.reconciliation_sources(tenant_id, created_at DESC);
CREATE INDEX idx_rs_type_status ON cross_source_reconciliation_service.reconciliation_sources(source_type, status);
CREATE INDEX idx_rs_schema_info ON cross_source_reconciliation_service.reconciliation_sources USING GIN (schema_info);
CREATE INDEX idx_rs_tags ON cross_source_reconciliation_service.reconciliation_sources USING GIN (tags);

-- reconciliation_schema_maps indexes (8)
CREATE INDEX idx_rsm_source_id ON cross_source_reconciliation_service.reconciliation_schema_maps(source_id);
CREATE INDEX idx_rsm_source_column ON cross_source_reconciliation_service.reconciliation_schema_maps(source_column);
CREATE INDEX idx_rsm_canonical_column ON cross_source_reconciliation_service.reconciliation_schema_maps(canonical_column);
CREATE INDEX idx_rsm_created_at ON cross_source_reconciliation_service.reconciliation_schema_maps(created_at DESC);
CREATE INDEX idx_rsm_source_canonical ON cross_source_reconciliation_service.reconciliation_schema_maps(source_id, canonical_column);
CREATE INDEX idx_rsm_transform ON cross_source_reconciliation_service.reconciliation_schema_maps(transform);
CREATE INDEX idx_rsm_unit_from ON cross_source_reconciliation_service.reconciliation_schema_maps(unit_from);
CREATE INDEX idx_rsm_unit_to ON cross_source_reconciliation_service.reconciliation_schema_maps(unit_to);

-- reconciliation_matches indexes (16)
CREATE INDEX idx_rm_tenant_id ON cross_source_reconciliation_service.reconciliation_matches(tenant_id);
CREATE INDEX idx_rm_job_id ON cross_source_reconciliation_service.reconciliation_matches(job_id);
CREATE INDEX idx_rm_source_a_id ON cross_source_reconciliation_service.reconciliation_matches(source_a_id);
CREATE INDEX idx_rm_source_b_id ON cross_source_reconciliation_service.reconciliation_matches(source_b_id);
CREATE INDEX idx_rm_entity_id ON cross_source_reconciliation_service.reconciliation_matches(entity_id);
CREATE INDEX idx_rm_period ON cross_source_reconciliation_service.reconciliation_matches(period);
CREATE INDEX idx_rm_confidence ON cross_source_reconciliation_service.reconciliation_matches(confidence DESC);
CREATE INDEX idx_rm_strategy ON cross_source_reconciliation_service.reconciliation_matches(strategy);
CREATE INDEX idx_rm_status ON cross_source_reconciliation_service.reconciliation_matches(status);
CREATE INDEX idx_rm_provenance ON cross_source_reconciliation_service.reconciliation_matches(provenance_hash);
CREATE INDEX idx_rm_created_at ON cross_source_reconciliation_service.reconciliation_matches(created_at DESC);
CREATE INDEX idx_rm_tenant_job ON cross_source_reconciliation_service.reconciliation_matches(tenant_id, job_id);
CREATE INDEX idx_rm_tenant_entity ON cross_source_reconciliation_service.reconciliation_matches(tenant_id, entity_id);
CREATE INDEX idx_rm_tenant_status ON cross_source_reconciliation_service.reconciliation_matches(tenant_id, status);
CREATE INDEX idx_rm_job_status ON cross_source_reconciliation_service.reconciliation_matches(job_id, status);
CREATE INDEX idx_rm_matched_fields ON cross_source_reconciliation_service.reconciliation_matches USING GIN (matched_fields);

-- reconciliation_comparisons indexes (14)
CREATE INDEX idx_rc_match_id ON cross_source_reconciliation_service.reconciliation_comparisons(match_id);
CREATE INDEX idx_rc_field_name ON cross_source_reconciliation_service.reconciliation_comparisons(field_name);
CREATE INDEX idx_rc_field_type ON cross_source_reconciliation_service.reconciliation_comparisons(field_type);
CREATE INDEX idx_rc_result ON cross_source_reconciliation_service.reconciliation_comparisons(result);
CREATE INDEX idx_rc_absolute_diff ON cross_source_reconciliation_service.reconciliation_comparisons(absolute_diff);
CREATE INDEX idx_rc_relative_diff ON cross_source_reconciliation_service.reconciliation_comparisons(relative_diff_pct);
CREATE INDEX idx_rc_provenance ON cross_source_reconciliation_service.reconciliation_comparisons(provenance_hash);
CREATE INDEX idx_rc_created_at ON cross_source_reconciliation_service.reconciliation_comparisons(created_at DESC);
CREATE INDEX idx_rc_match_field ON cross_source_reconciliation_service.reconciliation_comparisons(match_id, field_name);
CREATE INDEX idx_rc_match_result ON cross_source_reconciliation_service.reconciliation_comparisons(match_id, result);
CREATE INDEX idx_rc_field_result ON cross_source_reconciliation_service.reconciliation_comparisons(field_name, result);
CREATE INDEX idx_rc_result_diff ON cross_source_reconciliation_service.reconciliation_comparisons(result, relative_diff_pct);
CREATE INDEX idx_rc_match_type ON cross_source_reconciliation_service.reconciliation_comparisons(match_id, field_type);
CREATE INDEX idx_rc_type_result ON cross_source_reconciliation_service.reconciliation_comparisons(field_type, result);

-- reconciliation_discrepancies indexes (14)
CREATE INDEX idx_rd_tenant_id ON cross_source_reconciliation_service.reconciliation_discrepancies(tenant_id);
CREATE INDEX idx_rd_match_id ON cross_source_reconciliation_service.reconciliation_discrepancies(match_id);
CREATE INDEX idx_rd_job_id ON cross_source_reconciliation_service.reconciliation_discrepancies(job_id);
CREATE INDEX idx_rd_field_name ON cross_source_reconciliation_service.reconciliation_discrepancies(field_name);
CREATE INDEX idx_rd_discrepancy_type ON cross_source_reconciliation_service.reconciliation_discrepancies(discrepancy_type);
CREATE INDEX idx_rd_severity ON cross_source_reconciliation_service.reconciliation_discrepancies(severity);
CREATE INDEX idx_rd_deviation_pct ON cross_source_reconciliation_service.reconciliation_discrepancies(deviation_pct);
CREATE INDEX idx_rd_provenance ON cross_source_reconciliation_service.reconciliation_discrepancies(provenance_hash);
CREATE INDEX idx_rd_created_at ON cross_source_reconciliation_service.reconciliation_discrepancies(created_at DESC);
CREATE INDEX idx_rd_tenant_job ON cross_source_reconciliation_service.reconciliation_discrepancies(tenant_id, job_id);
CREATE INDEX idx_rd_tenant_severity ON cross_source_reconciliation_service.reconciliation_discrepancies(tenant_id, severity);
CREATE INDEX idx_rd_tenant_type ON cross_source_reconciliation_service.reconciliation_discrepancies(tenant_id, discrepancy_type);
CREATE INDEX idx_rd_job_severity ON cross_source_reconciliation_service.reconciliation_discrepancies(job_id, severity);
CREATE INDEX idx_rd_job_type ON cross_source_reconciliation_service.reconciliation_discrepancies(job_id, discrepancy_type);

-- reconciliation_resolutions indexes (14)
CREATE INDEX idx_rr_discrepancy_id ON cross_source_reconciliation_service.reconciliation_resolutions(discrepancy_id);
CREATE INDEX idx_rr_job_id ON cross_source_reconciliation_service.reconciliation_resolutions(job_id);
CREATE INDEX idx_rr_strategy ON cross_source_reconciliation_service.reconciliation_resolutions(strategy);
CREATE INDEX idx_rr_winning_source ON cross_source_reconciliation_service.reconciliation_resolutions(winning_source_id);
CREATE INDEX idx_rr_confidence ON cross_source_reconciliation_service.reconciliation_resolutions(confidence DESC);
CREATE INDEX idx_rr_reviewer ON cross_source_reconciliation_service.reconciliation_resolutions(reviewer);
CREATE INDEX idx_rr_status ON cross_source_reconciliation_service.reconciliation_resolutions(status);
CREATE INDEX idx_rr_provenance ON cross_source_reconciliation_service.reconciliation_resolutions(provenance_hash);
CREATE INDEX idx_rr_created_at ON cross_source_reconciliation_service.reconciliation_resolutions(created_at DESC);
CREATE INDEX idx_rr_job_strategy ON cross_source_reconciliation_service.reconciliation_resolutions(job_id, strategy);
CREATE INDEX idx_rr_job_status ON cross_source_reconciliation_service.reconciliation_resolutions(job_id, status);
CREATE INDEX idx_rr_strategy_status ON cross_source_reconciliation_service.reconciliation_resolutions(strategy, status);
CREATE INDEX idx_rr_discrepancy_status ON cross_source_reconciliation_service.reconciliation_resolutions(discrepancy_id, status);
CREATE INDEX idx_rr_winning_strategy ON cross_source_reconciliation_service.reconciliation_resolutions(winning_source_id, strategy);

-- reconciliation_golden_records indexes (14)
CREATE INDEX idx_rgr_tenant_id ON cross_source_reconciliation_service.reconciliation_golden_records(tenant_id);
CREATE INDEX idx_rgr_job_id ON cross_source_reconciliation_service.reconciliation_golden_records(job_id);
CREATE INDEX idx_rgr_entity_id ON cross_source_reconciliation_service.reconciliation_golden_records(entity_id);
CREATE INDEX idx_rgr_period ON cross_source_reconciliation_service.reconciliation_golden_records(period);
CREATE INDEX idx_rgr_total_confidence ON cross_source_reconciliation_service.reconciliation_golden_records(total_confidence DESC);
CREATE INDEX idx_rgr_provenance ON cross_source_reconciliation_service.reconciliation_golden_records(provenance_hash);
CREATE INDEX idx_rgr_created_at ON cross_source_reconciliation_service.reconciliation_golden_records(created_at DESC);
CREATE INDEX idx_rgr_tenant_entity ON cross_source_reconciliation_service.reconciliation_golden_records(tenant_id, entity_id);
CREATE INDEX idx_rgr_tenant_job ON cross_source_reconciliation_service.reconciliation_golden_records(tenant_id, job_id);
CREATE INDEX idx_rgr_tenant_period ON cross_source_reconciliation_service.reconciliation_golden_records(tenant_id, period);
CREATE INDEX idx_rgr_job_entity ON cross_source_reconciliation_service.reconciliation_golden_records(job_id, entity_id);
CREATE INDEX idx_rgr_fields ON cross_source_reconciliation_service.reconciliation_golden_records USING GIN (fields);
CREATE INDEX idx_rgr_field_sources ON cross_source_reconciliation_service.reconciliation_golden_records USING GIN (field_sources);
CREATE INDEX idx_rgr_field_confidences ON cross_source_reconciliation_service.reconciliation_golden_records USING GIN (field_confidences);

-- reconciliation_reports indexes (12)
CREATE INDEX idx_rrp_tenant_id ON cross_source_reconciliation_service.reconciliation_reports(tenant_id);
CREATE INDEX idx_rrp_job_id ON cross_source_reconciliation_service.reconciliation_reports(job_id);
CREATE INDEX idx_rrp_report_type ON cross_source_reconciliation_service.reconciliation_reports(report_type);
CREATE INDEX idx_rrp_provenance ON cross_source_reconciliation_service.reconciliation_reports(provenance_hash);
CREATE INDEX idx_rrp_created_at ON cross_source_reconciliation_service.reconciliation_reports(created_at DESC);
CREATE INDEX idx_rrp_tenant_job ON cross_source_reconciliation_service.reconciliation_reports(tenant_id, job_id);
CREATE INDEX idx_rrp_tenant_type ON cross_source_reconciliation_service.reconciliation_reports(tenant_id, report_type);
CREATE INDEX idx_rrp_tenant_created ON cross_source_reconciliation_service.reconciliation_reports(tenant_id, created_at DESC);
CREATE INDEX idx_rrp_job_type ON cross_source_reconciliation_service.reconciliation_reports(job_id, report_type);
CREATE INDEX idx_rrp_total_records ON cross_source_reconciliation_service.reconciliation_reports(total_records DESC);
CREATE INDEX idx_rrp_unresolved ON cross_source_reconciliation_service.reconciliation_reports(unresolved_count DESC);
CREATE INDEX idx_rrp_summary ON cross_source_reconciliation_service.reconciliation_reports USING GIN (summary);

-- reconciliation_audit_log indexes (12)
CREATE INDEX idx_ral_tenant_id ON cross_source_reconciliation_service.reconciliation_audit_log(tenant_id);
CREATE INDEX idx_ral_job_id ON cross_source_reconciliation_service.reconciliation_audit_log(job_id);
CREATE INDEX idx_ral_event_type ON cross_source_reconciliation_service.reconciliation_audit_log(event_type);
CREATE INDEX idx_ral_provenance ON cross_source_reconciliation_service.reconciliation_audit_log(provenance_hash);
CREATE INDEX idx_ral_created_at ON cross_source_reconciliation_service.reconciliation_audit_log(created_at DESC);
CREATE INDEX idx_ral_tenant_job ON cross_source_reconciliation_service.reconciliation_audit_log(tenant_id, job_id);
CREATE INDEX idx_ral_tenant_event ON cross_source_reconciliation_service.reconciliation_audit_log(tenant_id, event_type);
CREATE INDEX idx_ral_tenant_created ON cross_source_reconciliation_service.reconciliation_audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_ral_job_event ON cross_source_reconciliation_service.reconciliation_audit_log(job_id, event_type);
CREATE INDEX idx_ral_job_created ON cross_source_reconciliation_service.reconciliation_audit_log(job_id, created_at DESC);
CREATE INDEX idx_ral_tenant_job_event ON cross_source_reconciliation_service.reconciliation_audit_log(tenant_id, job_id, event_type);
CREATE INDEX idx_ral_details ON cross_source_reconciliation_service.reconciliation_audit_log USING GIN (details);

-- reconciliation_events indexes (hypertable-aware) (8)
CREATE INDEX idx_re_tenant_id ON cross_source_reconciliation_service.reconciliation_events(tenant_id, time DESC);
CREATE INDEX idx_re_job_id ON cross_source_reconciliation_service.reconciliation_events(job_id, time DESC);
CREATE INDEX idx_re_event_type ON cross_source_reconciliation_service.reconciliation_events(event_type, time DESC);
CREATE INDEX idx_re_tenant_type ON cross_source_reconciliation_service.reconciliation_events(tenant_id, event_type, time DESC);
CREATE INDEX idx_re_tenant_job ON cross_source_reconciliation_service.reconciliation_events(tenant_id, job_id, time DESC);
CREATE INDEX idx_re_provenance ON cross_source_reconciliation_service.reconciliation_events(provenance_hash, time DESC);
CREATE INDEX idx_re_job_type ON cross_source_reconciliation_service.reconciliation_events(job_id, event_type, time DESC);
CREATE INDEX idx_re_details ON cross_source_reconciliation_service.reconciliation_events USING GIN (details);

-- comparison_events indexes (hypertable-aware) (8)
CREATE INDEX idx_ce_tenant_id ON cross_source_reconciliation_service.comparison_events(tenant_id, time DESC);
CREATE INDEX idx_ce_match_id ON cross_source_reconciliation_service.comparison_events(match_id, time DESC);
CREATE INDEX idx_ce_field_name ON cross_source_reconciliation_service.comparison_events(field_name, time DESC);
CREATE INDEX idx_ce_result ON cross_source_reconciliation_service.comparison_events(result, time DESC);
CREATE INDEX idx_ce_diff_pct ON cross_source_reconciliation_service.comparison_events(diff_pct, time DESC);
CREATE INDEX idx_ce_tenant_result ON cross_source_reconciliation_service.comparison_events(tenant_id, result, time DESC);
CREATE INDEX idx_ce_tenant_match ON cross_source_reconciliation_service.comparison_events(tenant_id, match_id, time DESC);
CREATE INDEX idx_ce_provenance ON cross_source_reconciliation_service.comparison_events(provenance_hash, time DESC);

-- resolution_events indexes (hypertable-aware) (8)
CREATE INDEX idx_rese_tenant_id ON cross_source_reconciliation_service.resolution_events(tenant_id, time DESC);
CREATE INDEX idx_rese_job_id ON cross_source_reconciliation_service.resolution_events(job_id, time DESC);
CREATE INDEX idx_rese_discrepancy_id ON cross_source_reconciliation_service.resolution_events(discrepancy_id, time DESC);
CREATE INDEX idx_rese_strategy ON cross_source_reconciliation_service.resolution_events(strategy, time DESC);
CREATE INDEX idx_rese_confidence ON cross_source_reconciliation_service.resolution_events(confidence, time DESC);
CREATE INDEX idx_rese_tenant_strategy ON cross_source_reconciliation_service.resolution_events(tenant_id, strategy, time DESC);
CREATE INDEX idx_rese_tenant_job ON cross_source_reconciliation_service.resolution_events(tenant_id, job_id, time DESC);
CREATE INDEX idx_rese_provenance ON cross_source_reconciliation_service.resolution_events(provenance_hash, time DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE cross_source_reconciliation_service.reconciliation_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY rj_tenant_read ON cross_source_reconciliation_service.reconciliation_jobs FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY rj_tenant_write ON cross_source_reconciliation_service.reconciliation_jobs FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE cross_source_reconciliation_service.reconciliation_sources ENABLE ROW LEVEL SECURITY;
CREATE POLICY rs_tenant_read ON cross_source_reconciliation_service.reconciliation_sources FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY rs_tenant_write ON cross_source_reconciliation_service.reconciliation_sources FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE cross_source_reconciliation_service.reconciliation_matches ENABLE ROW LEVEL SECURITY;
CREATE POLICY rm_tenant_read ON cross_source_reconciliation_service.reconciliation_matches FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY rm_tenant_write ON cross_source_reconciliation_service.reconciliation_matches FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE cross_source_reconciliation_service.reconciliation_discrepancies ENABLE ROW LEVEL SECURITY;
CREATE POLICY rd_tenant_read ON cross_source_reconciliation_service.reconciliation_discrepancies FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY rd_tenant_write ON cross_source_reconciliation_service.reconciliation_discrepancies FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE cross_source_reconciliation_service.reconciliation_golden_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY rgr_tenant_read ON cross_source_reconciliation_service.reconciliation_golden_records FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY rgr_tenant_write ON cross_source_reconciliation_service.reconciliation_golden_records FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE cross_source_reconciliation_service.reconciliation_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY rrp_tenant_read ON cross_source_reconciliation_service.reconciliation_reports FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY rrp_tenant_write ON cross_source_reconciliation_service.reconciliation_reports FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE cross_source_reconciliation_service.reconciliation_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY ral_tenant_read ON cross_source_reconciliation_service.reconciliation_audit_log FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY ral_tenant_write ON cross_source_reconciliation_service.reconciliation_audit_log FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE cross_source_reconciliation_service.reconciliation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY re_tenant_read ON cross_source_reconciliation_service.reconciliation_events FOR SELECT USING (TRUE);
CREATE POLICY re_tenant_write ON cross_source_reconciliation_service.reconciliation_events FOR ALL USING (TRUE);

ALTER TABLE cross_source_reconciliation_service.comparison_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ce_tenant_read ON cross_source_reconciliation_service.comparison_events FOR SELECT USING (TRUE);
CREATE POLICY ce_tenant_write ON cross_source_reconciliation_service.comparison_events FOR ALL USING (TRUE);

ALTER TABLE cross_source_reconciliation_service.resolution_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY rese_tenant_read ON cross_source_reconciliation_service.resolution_events FOR SELECT USING (TRUE);
CREATE POLICY rese_tenant_write ON cross_source_reconciliation_service.resolution_events FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA cross_source_reconciliation_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA cross_source_reconciliation_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA cross_source_reconciliation_service TO greenlang_app;
GRANT SELECT ON cross_source_reconciliation_service.reconciliation_hourly_stats TO greenlang_app;
GRANT SELECT ON cross_source_reconciliation_service.discrepancy_hourly_stats TO greenlang_app;

GRANT USAGE ON SCHEMA cross_source_reconciliation_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA cross_source_reconciliation_service TO greenlang_readonly;
GRANT SELECT ON cross_source_reconciliation_service.reconciliation_hourly_stats TO greenlang_readonly;
GRANT SELECT ON cross_source_reconciliation_service.discrepancy_hourly_stats TO greenlang_readonly;

GRANT ALL ON SCHEMA cross_source_reconciliation_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA cross_source_reconciliation_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA cross_source_reconciliation_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'reconciliation:jobs:read', 'reconciliation', 'jobs_read', 'View reconciliation jobs and their progress'),
    (gen_random_uuid(), 'reconciliation:jobs:write', 'reconciliation', 'jobs_write', 'Create, start, cancel, and manage reconciliation jobs'),
    (gen_random_uuid(), 'reconciliation:sources:read', 'reconciliation', 'sources_read', 'View registered data sources and schema maps'),
    (gen_random_uuid(), 'reconciliation:sources:write', 'reconciliation', 'sources_write', 'Register, update, and manage data sources and schema maps'),
    (gen_random_uuid(), 'reconciliation:matches:read', 'reconciliation', 'matches_read', 'View matched record pairs and comparisons'),
    (gen_random_uuid(), 'reconciliation:matches:write', 'reconciliation', 'matches_write', 'Create and manage record matches'),
    (gen_random_uuid(), 'reconciliation:discrepancies:read', 'reconciliation', 'discrepancies_read', 'View detected discrepancies and severity levels'),
    (gen_random_uuid(), 'reconciliation:discrepancies:write', 'reconciliation', 'discrepancies_write', 'Manage discrepancies and escalation'),
    (gen_random_uuid(), 'reconciliation:resolutions:read', 'reconciliation', 'resolutions_read', 'View resolution decisions and justifications'),
    (gen_random_uuid(), 'reconciliation:resolutions:write', 'reconciliation', 'resolutions_write', 'Create and manage resolution decisions'),
    (gen_random_uuid(), 'reconciliation:golden_records:read', 'reconciliation', 'golden_records_read', 'View assembled golden records and field sources'),
    (gen_random_uuid(), 'reconciliation:golden_records:write', 'reconciliation', 'golden_records_write', 'Create and manage golden records'),
    (gen_random_uuid(), 'reconciliation:reports:read', 'reconciliation', 'reports_read', 'View reconciliation reports and summaries'),
    (gen_random_uuid(), 'reconciliation:reports:write', 'reconciliation', 'reports_write', 'Generate and manage reconciliation reports'),
    (gen_random_uuid(), 'reconciliation:audit:read', 'reconciliation', 'audit_read', 'View reconciliation audit log entries and provenance chains'),
    (gen_random_uuid(), 'reconciliation:admin', 'reconciliation', 'admin', 'Reconciliation service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('cross_source_reconciliation_service.reconciliation_events', INTERVAL '90 days');
SELECT add_retention_policy('cross_source_reconciliation_service.comparison_events', INTERVAL '90 days');
SELECT add_retention_policy('cross_source_reconciliation_service.resolution_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE cross_source_reconciliation_service.reconciliation_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('cross_source_reconciliation_service.reconciliation_events', INTERVAL '7 days');

ALTER TABLE cross_source_reconciliation_service.comparison_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('cross_source_reconciliation_service.comparison_events', INTERVAL '7 days');

ALTER TABLE cross_source_reconciliation_service.resolution_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('cross_source_reconciliation_service.resolution_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Cross-Source Reconciliation Agent (GL-DATA-X-018)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-018', 'Cross-Source Reconciliation Agent',
 'Multi-source data reconciliation engine for GreenLang Climate OS. Registers and manages data sources with priority and credibility scoring. Maps source schemas to canonical columns with unit and format transforms. Matches records across sources using exact, fuzzy, composite, key-based, temporal, and hierarchical strategies. Compares fields at granular level with absolute/relative tolerance, type-aware comparison (numeric, text, date, currency, percentage). Detects discrepancies with type classification (value/unit/format/scope/aggregation/rounding mismatch) and severity scoring (critical/high/medium/low/info). Resolves discrepancies via priority-based, credibility-weighted, most-recent, manual override, average, median, conservative, liberal, consensus, and escalation strategies. Assembles golden records with best-of-breed field selection, per-field source attribution, and composite confidence scoring. Generates reconciliation reports with match rates, discrepancy breakdown, resolution summary, and unresolved tracking. SHA-256 provenance chains for zero-hallucination audit trail.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/cross-source-reconciliation', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-018', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/cross-source-reconciliation-service", "tag": "1.0.0", "port": 8000}'::jsonb,
 '{"reconciliation", "data-quality", "golden-record", "matching", "discrepancy", "resolution", "multi-source"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "agriculture", "utilities"}',
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3')
ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES
('GL-DATA-X-018', '1.0.0', 'source_registration', 'configuration', 'Register data sources with priority, credibility scoring, schema mapping, and refresh cadence configuration.', '{"source_name", "source_type", "priority", "credibility", "schema_info"}', '{"source_id", "schema_maps", "validation_result"}', '{"source_types": ["erp", "csv", "excel", "api", "database", "pdf", "questionnaire", "satellite", "gis", "manual", "calculated", "external"], "priority_range": [1, 100], "credibility_range": [0.0, 1.0]}'::jsonb),
('GL-DATA-X-018', '1.0.0', 'record_matching', 'processing', 'Match records across data sources using configurable strategies with confidence scoring.', '{"source_a_records", "source_b_records", "match_config"}', '{"matches", "unmatched_a", "unmatched_b", "confidence_distribution"}', '{"strategies": ["exact", "fuzzy", "composite", "key_based", "temporal", "hierarchical"], "confidence_threshold": 0.7, "fuzzy_threshold": 0.8}'::jsonb),
('GL-DATA-X-018', '1.0.0', 'field_comparison', 'analysis', 'Compare matched record fields with type-aware tolerance checking and deviation analysis.', '{"match_id", "tolerance_config"}', '{"comparisons", "mismatches", "within_tolerance", "summary"}', '{"field_types": ["numeric", "text", "date", "boolean", "currency", "percentage", "unit_value"], "default_tolerance_pct": 1.0, "default_tolerance_abs": 0.01}'::jsonb),
('GL-DATA-X-018', '1.0.0', 'discrepancy_detection', 'analysis', 'Detect and classify discrepancies with severity scoring and deviation analysis.', '{"comparisons", "severity_config"}', '{"discrepancies", "severity_distribution", "hotspot_fields", "summary"}', '{"discrepancy_types": ["value_mismatch", "unit_mismatch", "missing_field", "type_conflict", "format_difference", "temporal_gap", "aggregation_error", "rounding_error", "currency_mismatch", "scope_mismatch"], "severity_levels": ["critical", "high", "medium", "low", "info"]}'::jsonb),
('GL-DATA-X-018', '1.0.0', 'discrepancy_resolution', 'processing', 'Resolve discrepancies using configurable strategies with justification and confidence tracking.', '{"discrepancy_ids", "resolution_config"}', '{"resolutions", "resolved_values", "confidence_scores", "summary"}', '{"strategies": ["priority_based", "credibility_weighted", "most_recent", "manual_override", "average", "median", "conservative", "liberal", "consensus", "escalated"], "auto_resolve_threshold": 0.9}'::jsonb),
('GL-DATA-X-018', '1.0.0', 'golden_record_assembly', 'processing', 'Assemble golden records from resolved discrepancies with per-field source attribution and composite confidence.', '{"job_id", "assembly_config"}', '{"golden_records", "field_sources", "confidence_scores", "coverage"}', '{"selection_strategy": "best_of_breed", "min_field_confidence": 0.5, "require_all_fields": false}'::jsonb),
('GL-DATA-X-018', '1.0.0', 'reconciliation_reporting', 'reporting', 'Generate comprehensive reconciliation reports with match rates, discrepancy breakdown, and resolution summary.', '{"job_id", "report_type", "config"}', '{"report", "match_rate", "discrepancy_summary", "resolution_summary", "recommendations"}', '{"report_types": ["reconciliation", "discrepancy", "resolution", "golden_record", "audit", "executive"]}'::jsonb)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES
('GL-DATA-X-018', 'GL-FOUND-X-002', '>=1.0.0', false, 'Schema validation for job configs, source definitions, and tolerance parameters'),
('GL-DATA-X-018', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit normalization for cross-source unit conversion during comparison'),
('GL-DATA-X-018', 'GL-FOUND-X-007', '>=1.0.0', false, 'Agent version and capability lookup for pipeline orchestration'),
('GL-DATA-X-018', 'GL-FOUND-X-006', '>=1.0.0', false, 'Access control enforcement for reconciliation jobs and results'),
('GL-DATA-X-018', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for matching, comparison, and resolution tracking'),
('GL-DATA-X-018', 'GL-FOUND-X-005', '>=1.0.0', true, 'Provenance and audit trail registration with citation service'),
('GL-DATA-X-018', 'GL-FOUND-X-008', '>=1.0.0', true, 'Reproducibility verification for reconciliation results'),
('GL-DATA-X-018', 'GL-FOUND-X-009', '>=1.0.0', true, 'QA Test Harness zero-hallucination verification'),
('GL-DATA-X-018', 'GL-DATA-X-013', '>=1.0.0', true, 'Data quality profiling for source quality assessment'),
('GL-DATA-X-018', 'GL-DATA-X-002', '>=1.0.0', true, 'Excel/CSV normalization for tabular source ingestion')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-018', 'Cross-Source Reconciliation Agent',
 'Multi-source data reconciliation engine. Source registration (priority/credibility/schema mapping). Record matching (exact/fuzzy/composite/key-based/temporal/hierarchical). Field comparison (type-aware/tolerance/deviation). Discrepancy detection (10 types/5 severity levels). Resolution (10 strategies/auto-resolve/manual review). Golden record assembly (best-of-breed/per-field attribution/composite confidence). Reconciliation reporting. SHA-256 provenance chains.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA cross_source_reconciliation_service IS 'Cross-Source Reconciliation Agent (AGENT-DATA-015) - multi-source matching, field comparison, discrepancy detection, resolution workflows, golden record assembly, provenance chains';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_jobs IS 'Job tracking for reconciliation runs: name, status, source IDs, config, match/discrepancy/golden record counts, timing, provenance hash';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_sources IS 'Registered data sources: name, type, priority, credibility score, schema info, refresh cadence, tags, status';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_schema_maps IS 'Column mapping rules: source column to canonical column, transform, unit conversion, date format';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_matches IS 'Matched record pairs: source A/B IDs, entity ID, period, confidence, strategy, status, matched fields, source records';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_comparisons IS 'Field-level comparison results: field name/type, source A/B values, absolute/relative diff, tolerance, result, provenance hash';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_discrepancies IS 'Detected discrepancies: match ref, field name, type, severity, source A/B values, deviation percentage, description';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_resolutions IS 'Resolution decisions: discrepancy ref, strategy, winning source, resolved value, confidence, justification, reviewer, status';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_golden_records IS 'Assembled golden records: entity ID, period, merged fields, per-field source attribution, per-field confidences, total confidence';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_reports IS 'Generated reports: type, total/matched/discrepancy/resolved/golden/unresolved counts, summary JSONB';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_audit_log IS 'Audit trail: job ref, event type, details JSONB, provenance hash';
COMMENT ON TABLE cross_source_reconciliation_service.reconciliation_events IS 'TimescaleDB hypertable: reconciliation lifecycle events (7-day chunks, 90-day retention)';
COMMENT ON TABLE cross_source_reconciliation_service.comparison_events IS 'TimescaleDB hypertable: comparison events with match ref, field, result, diff pct (7-day chunks, 90-day retention)';
COMMENT ON TABLE cross_source_reconciliation_service.resolution_events IS 'TimescaleDB hypertable: resolution events with job/discrepancy refs, strategy, confidence (7-day chunks, 90-day retention)';
COMMENT ON MATERIALIZED VIEW cross_source_reconciliation_service.reconciliation_hourly_stats IS 'Continuous aggregate: hourly reconciliation stats by tenant (total events, match/comparison/resolution/job completed counts)';
COMMENT ON MATERIALIZED VIEW cross_source_reconciliation_service.discrepancy_hourly_stats IS 'Continuous aggregate: hourly discrepancy stats by tenant (total comparisons, match/mismatch/tolerance/missing counts, avg deviation)';
