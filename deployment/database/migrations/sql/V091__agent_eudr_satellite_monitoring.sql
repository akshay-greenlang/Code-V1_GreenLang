-- ============================================================================
-- V091: AGENT-EUDR-003 Satellite Monitoring Agent
-- ============================================================================
-- PRD: PRD-AGENT-EUDR-003
-- Agent ID: GL-EUDR-SAT-003
-- Regulation: EU 2023/1115 (EUDR) Articles 9, 10, 12
-- Description: Schema for satellite monitoring including imagery acquisition,
--              spectral index calculation, baseline establishment, forest
--              change detection, multi-source data fusion, cloud gap filling,
--              continuous monitoring, alert generation, and evidence packaging.
-- ============================================================================

-- Create dedicated schema
CREATE SCHEMA IF NOT EXISTS eudr_satellite_monitoring;

-- ============================================================================
-- 1. Satellite Scenes (cached scene metadata)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.satellite_scenes (
    scene_id VARCHAR(120) PRIMARY KEY,
    source VARCHAR(30) NOT NULL
        CHECK (source IN ('sentinel_2', 'landsat_8', 'landsat_9', 'sentinel_1_sar', 'gfw_alerts')),
    acquisition_date DATE NOT NULL,
    cloud_cover_pct NUMERIC(5,2) DEFAULT 0.0
        CHECK (cloud_cover_pct >= 0 AND cloud_cover_pct <= 100),
    spatial_coverage_pct NUMERIC(5,2) DEFAULT 0.0
        CHECK (spatial_coverage_pct >= 0 AND spatial_coverage_pct <= 100),
    footprint_wkt TEXT,
    bands_available JSONB DEFAULT '[]',
    quality_score NUMERIC(5,2) DEFAULT 0.0
        CHECK (quality_score >= 0 AND quality_score <= 100),
    tile_id VARCHAR(20),
    processing_level VARCHAR(20) DEFAULT 'L2A',
    file_size_bytes BIGINT DEFAULT 0,
    download_url TEXT,
    cached_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.satellite_scenes IS
    'Cached satellite scene metadata from Sentinel-2, Landsat, SAR, and GFW sources';

CREATE INDEX idx_sat_scenes_source ON eudr_satellite_monitoring.satellite_scenes (source);
CREATE INDEX idx_sat_scenes_date ON eudr_satellite_monitoring.satellite_scenes (acquisition_date);
CREATE INDEX idx_sat_scenes_cloud ON eudr_satellite_monitoring.satellite_scenes (cloud_cover_pct);
CREATE INDEX idx_sat_scenes_tile ON eudr_satellite_monitoring.satellite_scenes (tile_id);

-- ============================================================================
-- 2. Plot Baselines (Dec 31, 2020 forest cover baselines)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.plot_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    baseline_date DATE NOT NULL DEFAULT '2020-12-31',
    actual_imagery_date DATE NOT NULL,
    source VARCHAR(30) NOT NULL
        CHECK (source IN ('sentinel_2', 'landsat_8', 'landsat_9', 'sentinel_1_sar', 'gfw_alerts')),
    scene_id VARCHAR(120) REFERENCES eudr_satellite_monitoring.satellite_scenes(scene_id),
    forest_area_ha NUMERIC(12,4) DEFAULT 0.0,
    total_area_ha NUMERIC(12,4) DEFAULT 0.0,
    forest_percentage NUMERIC(5,2) DEFAULT 0.0
        CHECK (forest_percentage >= 0 AND forest_percentage <= 100),
    ndvi_mean NUMERIC(6,4),
    ndvi_min NUMERIC(6,4),
    ndvi_max NUMERIC(6,4),
    ndvi_std_dev NUMERIC(6,4),
    canopy_density_class VARCHAR(30) DEFAULT 'forest_woodland'
        CHECK (canopy_density_class IN ('dense_forest', 'forest_woodland', 'shrubland', 'sparse_vegetation', 'non_vegetation')),
    biome VARCHAR(40) DEFAULT 'tropical_rainforest',
    commodity VARCHAR(50),
    country_code CHAR(2),
    confidence NUMERIC(5,4) DEFAULT 0.0
        CHECK (confidence >= 0 AND confidence <= 1),
    data_quality_score NUMERIC(5,2) DEFAULT 0.0,
    is_active BOOLEAN DEFAULT TRUE,
    superseded_by UUID,
    superseded_reason TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    established_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    established_by VARCHAR(100) DEFAULT 'system',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.plot_baselines IS
    'December 31, 2020 forest cover baselines per production plot for EUDR compliance';

-- Convert to hypertable (monthly partitioning)
SELECT create_hypertable(
    'eudr_satellite_monitoring.plot_baselines',
    'established_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_baselines_plot ON eudr_satellite_monitoring.plot_baselines (plot_id);
CREATE INDEX idx_baselines_commodity ON eudr_satellite_monitoring.plot_baselines (commodity);
CREATE INDEX idx_baselines_country ON eudr_satellite_monitoring.plot_baselines (country_code);
CREATE INDEX idx_baselines_active ON eudr_satellite_monitoring.plot_baselines (is_active) WHERE is_active = TRUE;

-- ============================================================================
-- 3. Forest Change Events (detected change events)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.forest_change_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    baseline_id UUID,
    analysis_date DATE NOT NULL,
    baseline_date DATE NOT NULL DEFAULT '2020-12-31',
    detection_method VARCHAR(30) NOT NULL
        CHECK (detection_method IN ('ndvi_differencing', 'spectral_angle', 'time_series_break', 'multi_source_fusion', 'sar_backscatter')),
    classification VARCHAR(30) NOT NULL
        CHECK (classification IN ('no_change', 'deforestation', 'degradation', 'reforestation', 'regrowth')),
    change_area_ha NUMERIC(12,4) DEFAULT 0.0,
    change_percentage NUMERIC(5,2) DEFAULT 0.0,
    ndvi_baseline NUMERIC(6,4),
    ndvi_current NUMERIC(6,4),
    ndvi_difference NUMERIC(6,4),
    confidence NUMERIC(5,4) DEFAULT 0.0
        CHECK (confidence >= 0 AND confidence <= 1),
    deforestation_detected BOOLEAN DEFAULT FALSE,
    sources_used JSONB DEFAULT '[]',
    agreement_score NUMERIC(5,4) DEFAULT 0.0,
    source VARCHAR(30) NOT NULL DEFAULT 'sentinel_2',
    scene_id VARCHAR(120),
    commodity VARCHAR(50),
    country_code CHAR(2),
    evidence_id UUID,
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.forest_change_events IS
    'Detected forest cover change events with classification and confidence';

-- Convert to hypertable (quarterly partitioning)
SELECT create_hypertable(
    'eudr_satellite_monitoring.forest_change_events',
    'detected_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

CREATE INDEX idx_change_events_plot ON eudr_satellite_monitoring.forest_change_events (plot_id);
CREATE INDEX idx_change_events_class ON eudr_satellite_monitoring.forest_change_events (classification);
CREATE INDEX idx_change_events_deforestation ON eudr_satellite_monitoring.forest_change_events (deforestation_detected) WHERE deforestation_detected = TRUE;
CREATE INDEX idx_change_events_method ON eudr_satellite_monitoring.forest_change_events (detection_method);
CREATE INDEX idx_change_events_commodity ON eudr_satellite_monitoring.forest_change_events (commodity);

-- ============================================================================
-- 4. Monitoring Schedules (continuous monitoring configurations)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.monitoring_schedules (
    schedule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    operator_id UUID,
    polygon_vertices JSONB NOT NULL,
    commodity VARCHAR(50),
    country_code CHAR(2),
    monitoring_interval VARCHAR(20) NOT NULL DEFAULT 'monthly'
        CHECK (monitoring_interval IN ('weekly', 'biweekly', 'monthly', 'quarterly')),
    priority INTEGER DEFAULT 5
        CHECK (priority >= 1 AND priority <= 10),
    is_active BOOLEAN DEFAULT TRUE,
    last_executed TIMESTAMPTZ,
    next_due TIMESTAMPTZ,
    execution_count INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.monitoring_schedules IS
    'Continuous satellite monitoring schedule configurations per plot';

CREATE INDEX idx_schedules_plot ON eudr_satellite_monitoring.monitoring_schedules (plot_id);
CREATE INDEX idx_schedules_active ON eudr_satellite_monitoring.monitoring_schedules (is_active) WHERE is_active = TRUE;
CREATE INDEX idx_schedules_due ON eudr_satellite_monitoring.monitoring_schedules (next_due) WHERE is_active = TRUE;
CREATE INDEX idx_schedules_priority ON eudr_satellite_monitoring.monitoring_schedules (priority DESC);

-- ============================================================================
-- 5. Monitoring Results (per-execution results)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.monitoring_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_id UUID NOT NULL REFERENCES eudr_satellite_monitoring.monitoring_schedules(schedule_id),
    plot_id UUID NOT NULL,
    execution_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    change_detected BOOLEAN DEFAULT FALSE,
    classification VARCHAR(30),
    change_area_ha NUMERIC(12,4) DEFAULT 0.0,
    ndvi_current NUMERIC(6,4),
    ndvi_baseline NUMERIC(6,4),
    confidence NUMERIC(5,4) DEFAULT 0.0,
    data_quality_score NUMERIC(5,2) DEFAULT 0.0,
    cloud_cover_pct NUMERIC(5,2) DEFAULT 0.0,
    source VARCHAR(30),
    scene_id VARCHAR(120),
    alert_generated BOOLEAN DEFAULT FALSE,
    alert_id UUID,
    error_message TEXT,
    duration_seconds NUMERIC(8,2) DEFAULT 0.0,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.monitoring_results IS
    'Per-execution monitoring results for scheduled satellite analyses';

-- Convert to hypertable (monthly partitioning)
SELECT create_hypertable(
    'eudr_satellite_monitoring.monitoring_results',
    'execution_date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_monitoring_results_schedule ON eudr_satellite_monitoring.monitoring_results (schedule_id);
CREATE INDEX idx_monitoring_results_plot ON eudr_satellite_monitoring.monitoring_results (plot_id);
CREATE INDEX idx_monitoring_results_status ON eudr_satellite_monitoring.monitoring_results (status);
CREATE INDEX idx_monitoring_results_change ON eudr_satellite_monitoring.monitoring_results (change_detected) WHERE change_detected = TRUE;

-- ============================================================================
-- 6. Satellite Alerts (generated alerts)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.satellite_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    schedule_id UUID,
    event_id UUID,
    severity VARCHAR(20) NOT NULL DEFAULT 'info'
        CHECK (severity IN ('critical', 'warning', 'info')),
    classification VARCHAR(30) NOT NULL,
    change_area_ha NUMERIC(12,4) DEFAULT 0.0,
    confidence NUMERIC(5,4) DEFAULT 0.0
        CHECK (confidence >= 0 AND confidence <= 1),
    ndvi_drop NUMERIC(6,4),
    commodity VARCHAR(50),
    country_code CHAR(2),
    source VARCHAR(30),
    scene_id VARCHAR(120),
    imagery_date DATE,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    acknowledge_notes TEXT,
    metadata JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.satellite_alerts IS
    'Satellite monitoring alerts for detected forest cover changes';

-- Convert to hypertable (monthly partitioning)
SELECT create_hypertable(
    'eudr_satellite_monitoring.satellite_alerts',
    'detected_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_alerts_plot ON eudr_satellite_monitoring.satellite_alerts (plot_id);
CREATE INDEX idx_alerts_severity ON eudr_satellite_monitoring.satellite_alerts (severity);
CREATE INDEX idx_alerts_unack ON eudr_satellite_monitoring.satellite_alerts (acknowledged) WHERE acknowledged = FALSE;
CREATE INDEX idx_alerts_commodity ON eudr_satellite_monitoring.satellite_alerts (commodity);
CREATE INDEX idx_alerts_country ON eudr_satellite_monitoring.satellite_alerts (country_code);

-- ============================================================================
-- 7. Evidence Packages (DDS evidence for regulatory submission)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.evidence_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    format VARCHAR(20) NOT NULL DEFAULT 'json'
        CHECK (format IN ('json', 'pdf', 'csv', 'eudr_xml')),
    compliance_determination VARCHAR(40) NOT NULL
        CHECK (compliance_determination IN ('COMPLIANT', 'NON_COMPLIANT', 'INSUFFICIENT_DATA', 'MANUAL_REVIEW_REQUIRED')),
    baseline_snapshot JSONB NOT NULL DEFAULT '{}',
    latest_analysis JSONB NOT NULL DEFAULT '{}',
    ndvi_time_series JSONB DEFAULT '[]',
    alerts_summary JSONB DEFAULT '{}',
    data_quality JSONB DEFAULT '{}',
    commodity VARCHAR(50),
    country_code CHAR(2),
    report_content TEXT,
    file_size_bytes BIGINT DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '5 years',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.evidence_packages IS
    'Satellite evidence packages for EUDR Due Diligence Statement submissions';

CREATE INDEX idx_evidence_plot ON eudr_satellite_monitoring.evidence_packages (plot_id);
CREATE INDEX idx_evidence_operator ON eudr_satellite_monitoring.evidence_packages (operator_id);
CREATE INDEX idx_evidence_compliance ON eudr_satellite_monitoring.evidence_packages (compliance_determination);
CREATE INDEX idx_evidence_format ON eudr_satellite_monitoring.evidence_packages (format);

-- ============================================================================
-- 8. Cloud Cover Log (cloud coverage tracking per analysis)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.cloud_cover_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    scene_id VARCHAR(120),
    source VARCHAR(30),
    cloud_cover_pct NUMERIC(5,2) DEFAULT 0.0,
    clear_pixel_count INTEGER DEFAULT 0,
    total_pixel_count INTEGER DEFAULT 0,
    gap_fill_method VARCHAR(30),
    gap_fill_quality NUMERIC(5,2) DEFAULT 0.0,
    filled_pixel_count INTEGER DEFAULT 0,
    original_data_pct NUMERIC(5,2) DEFAULT 100.0,
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.cloud_cover_log IS
    'Cloud coverage and gap filling tracking for satellite analyses';

-- Convert to hypertable (quarterly partitioning)
SELECT create_hypertable(
    'eudr_satellite_monitoring.cloud_cover_log',
    'logged_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

CREATE INDEX idx_cloud_log_plot ON eudr_satellite_monitoring.cloud_cover_log (plot_id);
CREATE INDEX idx_cloud_log_source ON eudr_satellite_monitoring.cloud_cover_log (source);

-- ============================================================================
-- 9. Data Quality Log (per-analysis quality metrics)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.data_quality_log (
    quality_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    analysis_type VARCHAR(40) NOT NULL,
    source VARCHAR(30),
    cloud_cover_pct NUMERIC(5,2),
    temporal_proximity_days INTEGER,
    spatial_coverage_pct NUMERIC(5,2),
    atmospheric_quality VARCHAR(20) DEFAULT 'good'
        CHECK (atmospheric_quality IN ('good', 'moderate', 'poor')),
    sensor_quality VARCHAR(20) DEFAULT 'good'
        CHECK (sensor_quality IN ('good', 'degraded')),
    gap_fill_percentage NUMERIC(5,2) DEFAULT 0.0,
    sources_count INTEGER DEFAULT 1,
    overall_score NUMERIC(5,2) DEFAULT 0.0
        CHECK (overall_score >= 0 AND overall_score <= 100),
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.data_quality_log IS
    'Data quality metrics for each satellite analysis operation';

CREATE INDEX idx_quality_plot ON eudr_satellite_monitoring.data_quality_log (plot_id);
CREATE INDEX idx_quality_score ON eudr_satellite_monitoring.data_quality_log (overall_score);

-- ============================================================================
-- 10. Satellite Audit Log (immutable audit trail)
-- ============================================================================
CREATE TABLE eudr_satellite_monitoring.satellite_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(30) NOT NULL
        CHECK (entity_type IN ('scene', 'baseline', 'change_detection', 'fusion', 'monitoring', 'alert', 'evidence', 'batch')),
    entity_id VARCHAR(120) NOT NULL,
    action VARCHAR(30) NOT NULL
        CHECK (action IN ('create', 'update', 'delete', 'search_scenes', 'download_scene', 'establish_baseline', 'detect_change', 'fuse_sources', 'fill_cloud_gap', 'schedule_monitoring', 'execute_monitoring', 'generate_alert', 'acknowledge_alert', 'generate_evidence', 'export_evidence')),
    actor VARCHAR(100) NOT NULL DEFAULT 'system',
    previous_hash VARCHAR(64),
    current_hash VARCHAR(64) NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_satellite_monitoring.satellite_audit_log IS
    'Immutable audit trail for all satellite monitoring operations with SHA-256 provenance chain';

CREATE INDEX idx_audit_entity ON eudr_satellite_monitoring.satellite_audit_log (entity_type, entity_id);
CREATE INDEX idx_audit_action ON eudr_satellite_monitoring.satellite_audit_log (action);
CREATE INDEX idx_audit_actor ON eudr_satellite_monitoring.satellite_audit_log (actor);
CREATE INDEX idx_audit_created ON eudr_satellite_monitoring.satellite_audit_log (created_at DESC);

-- ============================================================================
-- Continuous Aggregates
-- ============================================================================

-- Daily monitoring statistics
CREATE MATERIALIZED VIEW eudr_satellite_monitoring.daily_monitoring_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', execution_date) AS bucket,
    COUNT(*) AS total_executions,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    COUNT(*) FILTER (WHERE change_detected = TRUE) AS changes_detected,
    AVG(confidence) AS avg_confidence,
    AVG(data_quality_score) AS avg_quality_score,
    AVG(duration_seconds) AS avg_duration_seconds
FROM eudr_satellite_monitoring.monitoring_results
GROUP BY time_bucket('1 day', execution_date)
WITH NO DATA;

-- Weekly alert statistics
CREATE MATERIALIZED VIEW eudr_satellite_monitoring.weekly_alert_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', detected_at) AS bucket,
    severity,
    commodity,
    country_code,
    COUNT(*) AS alert_count,
    AVG(confidence) AS avg_confidence,
    SUM(change_area_ha) AS total_change_area_ha,
    COUNT(*) FILTER (WHERE acknowledged = TRUE) AS acknowledged_count
FROM eudr_satellite_monitoring.satellite_alerts
GROUP BY time_bucket('1 week', detected_at), severity, commodity, country_code
WITH NO DATA;

-- ============================================================================
-- Retention Policies (5-year per EUDR Article 31)
-- ============================================================================
SELECT add_retention_policy(
    'eudr_satellite_monitoring.plot_baselines',
    drop_after => INTERVAL '5 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'eudr_satellite_monitoring.forest_change_events',
    drop_after => INTERVAL '5 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'eudr_satellite_monitoring.monitoring_results',
    drop_after => INTERVAL '5 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'eudr_satellite_monitoring.satellite_alerts',
    drop_after => INTERVAL '5 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'eudr_satellite_monitoring.cloud_cover_log',
    drop_after => INTERVAL '5 years',
    if_not_exists => TRUE
);

-- ============================================================================
-- Continuous Aggregate Refresh Policies
-- ============================================================================
SELECT add_continuous_aggregate_policy(
    'eudr_satellite_monitoring.daily_monitoring_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy(
    'eudr_satellite_monitoring.weekly_alert_stats',
    start_offset => INTERVAL '2 weeks',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '4 hours',
    if_not_exists => TRUE
);

-- ============================================================================
-- Summary
-- ============================================================================
-- Tables: 10
--   1. satellite_scenes           - Scene metadata cache
--   2. plot_baselines             - Dec 2020 baselines (hypertable, monthly)
--   3. forest_change_events       - Change detection events (hypertable, quarterly)
--   4. monitoring_schedules       - Monitoring configurations
--   5. monitoring_results         - Per-execution results (hypertable, monthly)
--   6. satellite_alerts           - Generated alerts (hypertable, monthly)
--   7. evidence_packages          - DDS evidence packages
--   8. cloud_cover_log            - Cloud coverage tracking (hypertable, quarterly)
--   9. data_quality_log           - Quality metrics per analysis
--  10. satellite_audit_log        - Immutable audit trail
--
-- Hypertables: 5 (baselines, change_events, monitoring_results, alerts, cloud_cover_log)
-- Continuous Aggregates: 2 (daily_monitoring_stats, weekly_alert_stats)
-- Indexes: 28
-- Retention Policies: 5 (5-year per EUDR Article 31)
-- ============================================================================
