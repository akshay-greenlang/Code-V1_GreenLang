-- ============================================================================
-- V090: AGENT-EUDR-002 Geolocation Verification Agent
-- ============================================================================
-- PRD: PRD-AGENT-EUDR-002
-- Agent ID: GL-EUDR-GEO-002
-- Regulation: EU 2023/1115 (EUDR) Article 9
-- Description: Schema for geolocation verification including coordinate
--              validation, polygon topology, protected area screening,
--              deforestation cutoff verification, accuracy scoring,
--              temporal boundary tracking, and Article 9 compliance reporting.
-- ============================================================================

-- Create dedicated schema
CREATE SCHEMA IF NOT EXISTS eudr_geolocation_verification;

-- ============================================================================
-- 1. Plot Verifications (core verification results per plot)
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.plot_verifications (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    verification_level VARCHAR(20) NOT NULL DEFAULT 'standard'
        CHECK (verification_level IN ('quick', 'standard', 'deep')),
    overall_status VARCHAR(30) NOT NULL DEFAULT 'pending'
        CHECK (overall_status IN ('pending', 'passed', 'failed', 'warning')),
    accuracy_score NUMERIC(5,2) DEFAULT 0.0
        CHECK (accuracy_score >= 0 AND accuracy_score <= 100),
    quality_tier VARCHAR(20) DEFAULT 'unverified'
        CHECK (quality_tier IN ('gold', 'silver', 'bronze', 'unverified')),
    coordinate_result JSONB DEFAULT '{}',
    polygon_result JSONB DEFAULT '{}',
    protected_area_result JSONB DEFAULT '{}',
    deforestation_result JSONB DEFAULT '{}',
    temporal_result JSONB DEFAULT '{}',
    score_breakdown JSONB DEFAULT '{}',
    issues_count INTEGER DEFAULT 0,
    critical_issues_count INTEGER DEFAULT 0,
    commodity VARCHAR(50),
    declared_country_code CHAR(2),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    declared_area_hectares NUMERIC(12,4),
    provenance_hash VARCHAR(64) NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verified_by VARCHAR(100) DEFAULT 'system',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_geolocation_verification.plot_verifications IS
    'AGENT-EUDR-002: Core verification results for each production plot. '
    'Stores coordinate, polygon, protected area, deforestation, and temporal '
    'verification outcomes with composite accuracy scores.';

-- ============================================================================
-- 2. Batch Verification Jobs
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.batch_jobs (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    verification_level VARCHAR(20) NOT NULL DEFAULT 'standard'
        CHECK (verification_level IN ('quick', 'standard', 'deep')),
    total_plots INTEGER NOT NULL CHECK (total_plots > 0),
    processed INTEGER DEFAULT 0,
    passed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'cancelled', 'failed')),
    average_score NUMERIC(5,2) DEFAULT 0.0,
    priority_sort BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_geolocation_verification.batch_jobs IS
    'AGENT-EUDR-002: Batch verification job tracking. Supports QUICK, '
    'STANDARD, and DEEP verification levels with progress monitoring.';

-- ============================================================================
-- 3. Protected Area Overlaps Detected
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.protected_area_overlaps (
    overlap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    verification_id UUID REFERENCES eudr_geolocation_verification.plot_verifications(verification_id),
    protected_area_id VARCHAR(100) NOT NULL,
    protected_area_name VARCHAR(500),
    protected_area_type VARCHAR(50) NOT NULL
        CHECK (protected_area_type IN (
            'wdpa', 'ramsar', 'unesco_world_heritage',
            'key_biodiversity_area', 'indigenous_territory', 'national_protected'
        )),
    iucn_category VARCHAR(20),
    country_code CHAR(2),
    overlap_percentage NUMERIC(5,2) DEFAULT 0.0,
    overlap_severity VARCHAR(20) NOT NULL DEFAULT 'none'
        CHECK (overlap_severity IN ('none', 'marginal', 'partial', 'full')),
    designation_year INTEGER,
    managing_authority VARCHAR(500),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE eudr_geolocation_verification.protected_area_overlaps IS
    'AGENT-EUDR-002: Records of detected overlaps between production plots '
    'and protected areas (WDPA, Ramsar, UNESCO, KBA, ICCA, national).';

-- ============================================================================
-- 4. Deforestation Events Detected (hypertable)
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.deforestation_events (
    event_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    verification_id UUID,
    event_date DATE,
    tree_cover_loss_hectares NUMERIC(10,4),
    canopy_cover_before NUMERIC(5,2),
    canopy_cover_after NUMERIC(5,2),
    ndvi_before NUMERIC(6,4),
    ndvi_after NUMERIC(6,4),
    data_source VARCHAR(100) NOT NULL,
    confidence_score NUMERIC(5,2) DEFAULT 0.0,
    deforestation_status VARCHAR(30) NOT NULL DEFAULT 'inconclusive'
        CHECK (deforestation_status IN (
            'verified_clear', 'verified_forest',
            'deforestation_detected', 'inconclusive'
        )),
    evidence_package JSONB DEFAULT '{}',
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'eudr_geolocation_verification.deforestation_events',
    'detected_at',
    chunk_time_interval => INTERVAL '1 month'
);

COMMENT ON TABLE eudr_geolocation_verification.deforestation_events IS
    'AGENT-EUDR-002: Time-series tracking of deforestation events detected '
    'per plot. Verifies against EUDR cutoff date (Dec 31, 2020). '
    'Hypertable with monthly chunks.';

-- ============================================================================
-- 5. Accuracy Score History (hypertable)
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.accuracy_score_history (
    score_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    total_score NUMERIC(5,2) NOT NULL CHECK (total_score >= 0 AND total_score <= 100),
    quality_tier VARCHAR(20) NOT NULL,
    coordinate_precision_score NUMERIC(5,2) DEFAULT 0.0,
    polygon_quality_score NUMERIC(5,2) DEFAULT 0.0,
    country_match_score NUMERIC(5,2) DEFAULT 0.0,
    protected_area_score NUMERIC(5,2) DEFAULT 0.0,
    deforestation_score NUMERIC(5,2) DEFAULT 0.0,
    temporal_consistency_score NUMERIC(5,2) DEFAULT 0.0,
    commodity VARCHAR(50),
    country_code CHAR(2),
    provenance_hash VARCHAR(64),
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'eudr_geolocation_verification.accuracy_score_history',
    'scored_at',
    chunk_time_interval => INTERVAL '1 month'
);

COMMENT ON TABLE eudr_geolocation_verification.accuracy_score_history IS
    'AGENT-EUDR-002: Time-series of Geolocation Accuracy Scores (GAS) per '
    'plot. Tracks score improvements over time with component breakdown. '
    'Hypertable with monthly chunks.';

-- ============================================================================
-- 6. Temporal Boundary Changes (hypertable)
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.boundary_changes (
    change_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    change_type VARCHAR(30) NOT NULL
        CHECK (change_type IN ('expansion', 'contraction', 'shift', 'reshape')),
    area_before_hectares NUMERIC(10,4),
    area_after_hectares NUMERIC(10,4),
    area_change_pct NUMERIC(8,4),
    centroid_shift_meters NUMERIC(10,2),
    expansion_direction VARCHAR(20),
    expands_into_forest BOOLEAN DEFAULT FALSE,
    previous_boundary JSONB,
    new_boundary JSONB,
    previous_centroid_lat DOUBLE PRECISION,
    previous_centroid_lon DOUBLE PRECISION,
    new_centroid_lat DOUBLE PRECISION,
    new_centroid_lon DOUBLE PRECISION,
    is_suspicious BOOLEAN DEFAULT FALSE,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'eudr_geolocation_verification.boundary_changes',
    'detected_at',
    chunk_time_interval => INTERVAL '3 months'
);

COMMENT ON TABLE eudr_geolocation_verification.boundary_changes IS
    'AGENT-EUDR-002: Time-series tracking of plot boundary changes. '
    'Detects expansion, contraction, shift, and reshape events with '
    'forest encroachment analysis. Hypertable with quarterly chunks.';

-- ============================================================================
-- 7. Article 9 Compliance Snapshots (hypertable)
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.compliance_snapshots (
    snapshot_id UUID DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50),
    country_code CHAR(2),
    total_plots INTEGER NOT NULL DEFAULT 0,
    compliant_plots INTEGER NOT NULL DEFAULT 0,
    non_compliant_plots INTEGER NOT NULL DEFAULT 0,
    pending_plots INTEGER NOT NULL DEFAULT 0,
    compliance_rate NUMERIC(5,2) DEFAULT 0.0,
    average_accuracy_score NUMERIC(5,2) DEFAULT 0.0,
    top_issues JSONB DEFAULT '[]',
    remediation_count INTEGER DEFAULT 0,
    estimated_effort_hours NUMERIC(8,2) DEFAULT 0.0,
    report_hash VARCHAR(64),
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'eudr_geolocation_verification.compliance_snapshots',
    'snapshot_at',
    chunk_time_interval => INTERVAL '1 month'
);

COMMENT ON TABLE eudr_geolocation_verification.compliance_snapshots IS
    'AGENT-EUDR-002: Article 9 compliance snapshots tracking operator '
    'compliance rates over time by commodity and country. '
    'Hypertable with monthly chunks.';

-- ============================================================================
-- 8. Verification Audit Log (hypertable)
-- ============================================================================
CREATE TABLE eudr_geolocation_verification.verification_audit_log (
    audit_id UUID DEFAULT gen_random_uuid(),
    verification_id UUID,
    plot_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    previous_status VARCHAR(30),
    new_status VARCHAR(30),
    previous_score NUMERIC(5,2),
    new_score NUMERIC(5,2),
    changed_by VARCHAR(100) DEFAULT 'system',
    change_reason TEXT,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'eudr_geolocation_verification.verification_audit_log',
    'created_at',
    chunk_time_interval => INTERVAL '1 month'
);

COMMENT ON TABLE eudr_geolocation_verification.verification_audit_log IS
    'AGENT-EUDR-002: Immutable audit trail for all verification actions. '
    'Tracks status and score changes with provenance hashes per EUDR Art 31.';

-- ============================================================================
-- Indexes
-- ============================================================================

-- Plot verifications indexes
CREATE INDEX idx_geo_verif_plot_id
    ON eudr_geolocation_verification.plot_verifications(plot_id);
CREATE INDEX idx_geo_verif_operator
    ON eudr_geolocation_verification.plot_verifications(operator_id);
CREATE INDEX idx_geo_verif_status
    ON eudr_geolocation_verification.plot_verifications(overall_status);
CREATE INDEX idx_geo_verif_score
    ON eudr_geolocation_verification.plot_verifications(accuracy_score);
CREATE INDEX idx_geo_verif_tier
    ON eudr_geolocation_verification.plot_verifications(quality_tier);
CREATE INDEX idx_geo_verif_commodity
    ON eudr_geolocation_verification.plot_verifications(commodity);
CREATE INDEX idx_geo_verif_country
    ON eudr_geolocation_verification.plot_verifications(declared_country_code);
CREATE INDEX idx_geo_verif_verified_at
    ON eudr_geolocation_verification.plot_verifications(verified_at);

-- Batch jobs indexes
CREATE INDEX idx_geo_batch_operator
    ON eudr_geolocation_verification.batch_jobs(operator_id);
CREATE INDEX idx_geo_batch_status
    ON eudr_geolocation_verification.batch_jobs(status);

-- Protected area overlaps indexes
CREATE INDEX idx_geo_overlaps_plot
    ON eudr_geolocation_verification.protected_area_overlaps(plot_id);
CREATE INDEX idx_geo_overlaps_verif
    ON eudr_geolocation_verification.protected_area_overlaps(verification_id);
CREATE INDEX idx_geo_overlaps_severity
    ON eudr_geolocation_verification.protected_area_overlaps(overlap_severity);
CREATE INDEX idx_geo_overlaps_type
    ON eudr_geolocation_verification.protected_area_overlaps(protected_area_type);

-- Deforestation events indexes (TimescaleDB hypertable)
CREATE INDEX idx_geo_deforest_plot
    ON eudr_geolocation_verification.deforestation_events(plot_id, detected_at DESC);
CREATE INDEX idx_geo_deforest_status
    ON eudr_geolocation_verification.deforestation_events(deforestation_status, detected_at DESC);

-- Accuracy score history indexes (TimescaleDB hypertable)
CREATE INDEX idx_geo_scores_plot
    ON eudr_geolocation_verification.accuracy_score_history(plot_id, scored_at DESC);
CREATE INDEX idx_geo_scores_operator
    ON eudr_geolocation_verification.accuracy_score_history(operator_id, scored_at DESC);
CREATE INDEX idx_geo_scores_tier
    ON eudr_geolocation_verification.accuracy_score_history(quality_tier, scored_at DESC);

-- Boundary changes indexes (TimescaleDB hypertable)
CREATE INDEX idx_geo_boundary_plot
    ON eudr_geolocation_verification.boundary_changes(plot_id, detected_at DESC);
CREATE INDEX idx_geo_boundary_suspicious
    ON eudr_geolocation_verification.boundary_changes(is_suspicious, detected_at DESC)
    WHERE is_suspicious = TRUE;
CREATE INDEX idx_geo_boundary_forest
    ON eudr_geolocation_verification.boundary_changes(expands_into_forest, detected_at DESC)
    WHERE expands_into_forest = TRUE;

-- Compliance snapshots indexes (TimescaleDB hypertable)
CREATE INDEX idx_geo_compliance_operator
    ON eudr_geolocation_verification.compliance_snapshots(operator_id, snapshot_at DESC);
CREATE INDEX idx_geo_compliance_commodity
    ON eudr_geolocation_verification.compliance_snapshots(commodity, snapshot_at DESC);

-- Audit log indexes (TimescaleDB hypertable)
CREATE INDEX idx_geo_audit_plot
    ON eudr_geolocation_verification.verification_audit_log(plot_id, created_at DESC);
CREATE INDEX idx_geo_audit_operator
    ON eudr_geolocation_verification.verification_audit_log(operator_id, created_at DESC);

-- ============================================================================
-- Continuous Aggregates
-- ============================================================================

-- Daily accuracy score aggregates per operator
CREATE MATERIALIZED VIEW eudr_geolocation_verification.daily_accuracy_scores
WITH (timescaledb.continuous) AS
SELECT
    operator_id,
    time_bucket('1 day', scored_at) AS day,
    COUNT(*) AS plots_scored,
    AVG(total_score) AS avg_score,
    MIN(total_score) AS min_score,
    MAX(total_score) AS max_score,
    COUNT(*) FILTER (WHERE quality_tier = 'gold') AS gold_count,
    COUNT(*) FILTER (WHERE quality_tier = 'silver') AS silver_count,
    COUNT(*) FILTER (WHERE quality_tier = 'bronze') AS bronze_count,
    COUNT(*) FILTER (WHERE quality_tier = 'unverified') AS unverified_count
FROM eudr_geolocation_verification.accuracy_score_history
GROUP BY operator_id, time_bucket('1 day', scored_at);

-- Weekly compliance rate aggregates per operator
CREATE MATERIALIZED VIEW eudr_geolocation_verification.weekly_compliance_rates
WITH (timescaledb.continuous) AS
SELECT
    operator_id,
    commodity,
    time_bucket('1 week', snapshot_at) AS week,
    AVG(compliance_rate) AS avg_compliance_rate,
    AVG(average_accuracy_score) AS avg_accuracy_score,
    SUM(total_plots) AS total_plots,
    SUM(compliant_plots) AS total_compliant,
    SUM(non_compliant_plots) AS total_non_compliant
FROM eudr_geolocation_verification.compliance_snapshots
GROUP BY operator_id, commodity, time_bucket('1 week', snapshot_at);

-- ============================================================================
-- Continuous Aggregate Policies
-- ============================================================================

SELECT add_continuous_aggregate_policy('eudr_geolocation_verification.daily_accuracy_scores',
    start_offset => INTERVAL '3 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

SELECT add_continuous_aggregate_policy('eudr_geolocation_verification.weekly_compliance_rates',
    start_offset => INTERVAL '2 weeks',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '4 hours'
);

-- ============================================================================
-- Retention Policies (5 years per EUDR Article 31)
-- ============================================================================

SELECT add_retention_policy('eudr_geolocation_verification.deforestation_events',
    INTERVAL '5 years');
SELECT add_retention_policy('eudr_geolocation_verification.accuracy_score_history',
    INTERVAL '5 years');
SELECT add_retention_policy('eudr_geolocation_verification.boundary_changes',
    INTERVAL '5 years');
SELECT add_retention_policy('eudr_geolocation_verification.compliance_snapshots',
    INTERVAL '5 years');
SELECT add_retention_policy('eudr_geolocation_verification.verification_audit_log',
    INTERVAL '5 years');

-- ============================================================================
-- Summary
-- ============================================================================
-- Tables created: 8
--   1. plot_verifications          (core verification results)
--   2. batch_jobs                   (batch processing tracking)
--   3. protected_area_overlaps      (protected area overlap records)
--   4. deforestation_events         (hypertable - deforestation detection)
--   5. accuracy_score_history       (hypertable - GAS score tracking)
--   6. boundary_changes             (hypertable - temporal boundary tracking)
--   7. compliance_snapshots         (hypertable - Art 9 compliance tracking)
--   8. verification_audit_log       (hypertable - immutable audit trail)
--
-- Hypertables: 5 (with monthly/quarterly chunking)
-- Continuous Aggregates: 2
--   1. daily_accuracy_scores        (daily score aggregation per operator)
--   2. weekly_compliance_rates      (weekly compliance tracking)
-- Indexes: 24
-- Retention policies: 5 (all 5-year per EUDR Article 31)
-- ============================================================================
