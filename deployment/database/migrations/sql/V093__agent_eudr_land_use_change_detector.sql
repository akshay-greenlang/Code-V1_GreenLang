-- =============================================================================
-- V093: AGENT-EUDR-005 Land Use Change Detector
-- =============================================================================
-- Agent: Land Use Change Detector (AGENT-EUDR-005)
-- Purpose: Land use classification, transition detection, trajectory analysis,
--          EUDR cutoff date compliance verification, conversion risk assessment
-- Tables: 10 (5 hypertables, 2 continuous aggregates, 28 indexes, 5 retention)
-- DB Prefix: gl_eudr_luc_
-- EUDR: Regulation (EU) 2023/1115, Articles 2(1), 2(3), 2(5), 9, 10, 29, 31
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Land Use Classifications
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_classifications (
    classification_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    classification_date DATE NOT NULL,
    land_use_category VARCHAR(50) NOT NULL,
    sub_category VARCHAR(100),
    classification_method VARCHAR(30) NOT NULL DEFAULT 'ensemble',
    confidence NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    spectral_indices JSONB DEFAULT '{}',
    band_values JSONB DEFAULT '{}',
    texture_features JSONB DEFAULT '{}',
    phenology_data JSONB DEFAULT '{}',
    commodity_context VARCHAR(30),
    is_agricultural BOOLEAN DEFAULT FALSE,
    is_forest BOOLEAN DEFAULT FALSE,
    data_quality VARCHAR(20) DEFAULT 'standard',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    classified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_luc_classifications', 'classified_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_luc_class_plot ON gl_eudr_luc_classifications (plot_id);
CREATE INDEX idx_luc_class_date ON gl_eudr_luc_classifications (classification_date);
CREATE INDEX idx_luc_class_category ON gl_eudr_luc_classifications (land_use_category);
CREATE INDEX idx_luc_class_method ON gl_eudr_luc_classifications (classification_method);
CREATE INDEX idx_luc_class_commodity ON gl_eudr_luc_classifications (commodity_context);

-- ---------------------------------------------------------------------------
-- 2. Land Use Transitions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_transitions (
    transition_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    from_class VARCHAR(50) NOT NULL,
    to_class VARCHAR(50) NOT NULL,
    transition_type VARCHAR(50) NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    estimated_transition_start DATE,
    estimated_transition_end DATE,
    confidence NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    area_affected_ha NUMERIC(12,4),
    is_deforestation BOOLEAN DEFAULT FALSE,
    is_degradation BOOLEAN DEFAULT FALSE,
    eudr_article VARCHAR(20),
    severity VARCHAR(20) DEFAULT 'medium',
    evidence JSONB DEFAULT '{}',
    spectral_changes JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_luc_transitions', 'detected_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_luc_trans_plot ON gl_eudr_luc_transitions (plot_id);
CREATE INDEX idx_luc_trans_type ON gl_eudr_luc_transitions (transition_type);
CREATE INDEX idx_luc_trans_deforestation ON gl_eudr_luc_transitions (is_deforestation) WHERE is_deforestation = TRUE;
CREATE INDEX idx_luc_trans_from_to ON gl_eudr_luc_transitions (from_class, to_class);

-- ---------------------------------------------------------------------------
-- 3. Temporal Trajectories
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_trajectories (
    trajectory_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    trajectory_type VARCHAR(30) NOT NULL,
    change_date DATE,
    change_start_date DATE,
    change_end_date DATE,
    oscillation_period_months INTEGER,
    recovery_completeness NUMERIC(5,4),
    recovery_rate NUMERIC(8,4),
    confidence NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    is_natural_disturbance BOOLEAN DEFAULT FALSE,
    ndvi_time_series JSONB DEFAULT '[]',
    class_time_series JSONB DEFAULT '[]',
    visualization_data JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_luc_trajectories', 'analyzed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_luc_traj_plot ON gl_eudr_luc_trajectories (plot_id);
CREATE INDEX idx_luc_traj_type ON gl_eudr_luc_trajectories (trajectory_type);
CREATE INDEX idx_luc_traj_natural ON gl_eudr_luc_trajectories (is_natural_disturbance) WHERE is_natural_disturbance = TRUE;

-- ---------------------------------------------------------------------------
-- 4. Cutoff Date Verifications
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_cutoff_verifications (
    verification_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    commodity VARCHAR(30),
    verdict VARCHAR(30) NOT NULL,
    cutoff_date DATE NOT NULL DEFAULT '2020-12-31',
    cutoff_land_use VARCHAR(50) NOT NULL,
    cutoff_confidence NUMERIC(5,4) NOT NULL,
    current_land_use VARCHAR(50) NOT NULL,
    current_confidence NUMERIC(5,4) NOT NULL,
    transition_detected BOOLEAN DEFAULT FALSE,
    transition_type VARCHAR(50),
    trajectory_type VARCHAR(30),
    overall_confidence NUMERIC(5,4) NOT NULL,
    evidence JSONB DEFAULT '{}',
    cross_validation JSONB DEFAULT '{}',
    eudr_articles JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_luc_cutoff_verifications', 'verified_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_luc_cutoff_plot ON gl_eudr_luc_cutoff_verifications (plot_id);
CREATE INDEX idx_luc_cutoff_verdict ON gl_eudr_luc_cutoff_verifications (verdict);
CREATE INDEX idx_luc_cutoff_commodity ON gl_eudr_luc_cutoff_verifications (commodity);
CREATE INDEX idx_luc_cutoff_non_compliant ON gl_eudr_luc_cutoff_verifications (verdict) WHERE verdict IN ('non_compliant', 'degraded');

-- ---------------------------------------------------------------------------
-- 5. Cropland Conversions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_cropland_conversions (
    conversion_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    conversion_type VARCHAR(50) NOT NULL,
    commodity VARCHAR(30),
    from_class VARCHAR(50) NOT NULL,
    to_class VARCHAR(50) NOT NULL,
    conversion_date_start DATE,
    conversion_date_end DATE,
    area_ha NUMERIC(12,4),
    scale_category VARCHAR(20) DEFAULT 'medium',
    is_progressive BOOLEAN DEFAULT FALSE,
    expansion_rate_ha_year NUMERIC(10,4),
    confidence NUMERIC(5,4) NOT NULL,
    spatial_pattern JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('gl_eudr_luc_cropland_conversions', 'detected_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

CREATE INDEX idx_luc_conv_plot ON gl_eudr_luc_cropland_conversions (plot_id);
CREATE INDEX idx_luc_conv_type ON gl_eudr_luc_cropland_conversions (conversion_type);
CREATE INDEX idx_luc_conv_commodity ON gl_eudr_luc_cropland_conversions (commodity);
CREATE INDEX idx_luc_conv_scale ON gl_eudr_luc_cropland_conversions (scale_category);

-- ---------------------------------------------------------------------------
-- 6. Conversion Risk Assessments
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_conversion_risks (
    risk_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    commodity VARCHAR(30),
    composite_score NUMERIC(5,2) NOT NULL CHECK (composite_score >= 0 AND composite_score <= 100),
    risk_tier VARCHAR(20) NOT NULL,
    frontier_proximity_score NUMERIC(5,2) DEFAULT 0,
    historical_rate_score NUMERIC(5,2) DEFAULT 0,
    road_proximity_score NUMERIC(5,2) DEFAULT 0,
    population_trend_score NUMERIC(5,2) DEFAULT 0,
    commodity_price_score NUMERIC(5,2) DEFAULT 0,
    protected_area_score NUMERIC(5,2) DEFAULT 0,
    governance_score NUMERIC(5,2) DEFAULT 0,
    slope_accessibility_score NUMERIC(5,2) DEFAULT 0,
    probability_6m NUMERIC(5,4),
    probability_12m NUMERIC(5,4),
    probability_24m NUMERIC(5,4),
    risk_factors JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_luc_risk_plot ON gl_eudr_luc_conversion_risks (plot_id);
CREATE INDEX idx_luc_risk_tier ON gl_eudr_luc_conversion_risks (risk_tier);
CREATE INDEX idx_luc_risk_score ON gl_eudr_luc_conversion_risks (composite_score DESC);
CREATE INDEX idx_luc_risk_high ON gl_eudr_luc_conversion_risks (risk_tier) WHERE risk_tier IN ('high', 'critical');

-- ---------------------------------------------------------------------------
-- 7. Urban Encroachment Records
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_urban_encroachment (
    encroachment_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    buffer_km NUMERIC(6,2) NOT NULL,
    date_from DATE NOT NULL,
    date_to DATE NOT NULL,
    encroachment_detected BOOLEAN DEFAULT FALSE,
    urban_expansion_rate_ha_year NUMERIC(10,4),
    infrastructure_types JSONB DEFAULT '[]',
    pressure_corridors JSONB DEFAULT '[]',
    new_roads_detected INTEGER DEFAULT 0,
    time_to_conversion_months NUMERIC(8,2),
    urban_proximity_risk NUMERIC(5,2) DEFAULT 0,
    encroachment_map JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_luc_urban_plot ON gl_eudr_luc_urban_encroachment (plot_id);
CREATE INDEX idx_luc_urban_detected ON gl_eudr_luc_urban_encroachment (encroachment_detected) WHERE encroachment_detected = TRUE;

-- ---------------------------------------------------------------------------
-- 8. Compliance Reports
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_compliance_reports (
    report_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    report_type VARCHAR(50) NOT NULL,
    format VARCHAR(20) NOT NULL DEFAULT 'json',
    plot_ids JSONB DEFAULT '[]',
    region_bounds JSONB,
    commodity VARCHAR(30),
    status VARCHAR(20) NOT NULL DEFAULT 'generating',
    report_data JSONB DEFAULT '{}',
    summary JSONB DEFAULT '{}',
    compliant_count INTEGER DEFAULT 0,
    non_compliant_count INTEGER DEFAULT 0,
    degraded_count INTEGER DEFAULT 0,
    inconclusive_count INTEGER DEFAULT 0,
    download_url TEXT,
    provenance_hash VARCHAR(64) NOT NULL,
    created_by VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_luc_report_type ON gl_eudr_luc_compliance_reports (report_type);
CREATE INDEX idx_luc_report_status ON gl_eudr_luc_compliance_reports (status);
CREATE INDEX idx_luc_report_created ON gl_eudr_luc_compliance_reports (created_at DESC);

-- ---------------------------------------------------------------------------
-- 9. Batch Jobs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_batch_jobs (
    job_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    parameters JSONB DEFAULT '{}',
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    progress_pct NUMERIC(5,2) DEFAULT 0,
    results JSONB DEFAULT '{}',
    error_message TEXT,
    submitted_by VARCHAR(100),
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_luc_batch_status ON gl_eudr_luc_batch_jobs (status);
CREATE INDEX idx_luc_batch_type ON gl_eudr_luc_batch_jobs (job_type);

-- ---------------------------------------------------------------------------
-- 10. Audit Log (immutable)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gl_eudr_luc_audit_log (
    log_id UUID DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(100),
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    parent_hash VARCHAR(64),
    chain_hash VARCHAR(64) NOT NULL,
    details JSONB DEFAULT '{}',
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_luc_audit_entity ON gl_eudr_luc_audit_log (entity_type, entity_id);
CREATE INDEX idx_luc_audit_action ON gl_eudr_luc_audit_log (action);
CREATE INDEX idx_luc_audit_actor ON gl_eudr_luc_audit_log (actor);
CREATE INDEX idx_luc_audit_chain ON gl_eudr_luc_audit_log (chain_hash);

-- ---------------------------------------------------------------------------
-- Continuous Aggregates
-- ---------------------------------------------------------------------------

-- Daily classification statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_luc_daily_classifications
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', classified_at) AS bucket,
    land_use_category,
    classification_method,
    COUNT(*) AS classification_count,
    AVG(confidence) AS avg_confidence,
    MIN(confidence) AS min_confidence,
    MAX(confidence) AS max_confidence
FROM gl_eudr_luc_classifications
GROUP BY bucket, land_use_category, classification_method
WITH NO DATA;

-- Weekly verification verdict statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_luc_weekly_verdicts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', verified_at) AS bucket,
    verdict,
    commodity,
    COUNT(*) AS verdict_count,
    AVG(overall_confidence) AS avg_confidence
FROM gl_eudr_luc_cutoff_verifications
GROUP BY bucket, verdict, commodity
WITH NO DATA;

-- ---------------------------------------------------------------------------
-- Retention Policies (5 years per EUDR Article 31)
-- ---------------------------------------------------------------------------
SELECT add_retention_policy('gl_eudr_luc_classifications', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_luc_transitions', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_luc_trajectories', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_luc_cutoff_verifications', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_luc_cropland_conversions', INTERVAL '5 years', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Refresh Policies for Continuous Aggregates
-- ---------------------------------------------------------------------------
SELECT add_continuous_aggregate_policy('gl_eudr_luc_daily_classifications',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy('gl_eudr_luc_weekly_verdicts',
    start_offset => INTERVAL '2 weeks',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '4 hours',
    if_not_exists => TRUE
);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE gl_eudr_luc_classifications IS 'AGENT-EUDR-005: Land use classifications (IPCC categories)';
COMMENT ON TABLE gl_eudr_luc_transitions IS 'AGENT-EUDR-005: Land use transition detection results';
COMMENT ON TABLE gl_eudr_luc_trajectories IS 'AGENT-EUDR-005: Temporal trajectory analyses';
COMMENT ON TABLE gl_eudr_luc_cutoff_verifications IS 'AGENT-EUDR-005: EUDR cutoff date compliance verdicts';
COMMENT ON TABLE gl_eudr_luc_cropland_conversions IS 'AGENT-EUDR-005: Agricultural conversion detections';
COMMENT ON TABLE gl_eudr_luc_conversion_risks IS 'AGENT-EUDR-005: Conversion risk assessments';
COMMENT ON TABLE gl_eudr_luc_urban_encroachment IS 'AGENT-EUDR-005: Urban encroachment monitoring';
COMMENT ON TABLE gl_eudr_luc_compliance_reports IS 'AGENT-EUDR-005: Compliance evidence reports';
COMMENT ON TABLE gl_eudr_luc_batch_jobs IS 'AGENT-EUDR-005: Batch processing jobs';
COMMENT ON TABLE gl_eudr_luc_audit_log IS 'AGENT-EUDR-005: Immutable audit trail';
