-- ============================================================================
-- V108: AGENT-EUDR-020 Deforestation Alert System Agent
-- ============================================================================
-- Creates tables for satellite detection ingestion, alert lifecycle management,
-- severity scoring, spatial buffer zone monitoring, buffer violation tracking,
-- EUDR cutoff date verification, historical forest baselines, baseline
-- comparisons, workflow state machines, compliance impact assessments,
-- notification dispatch, and comprehensive audit trails.
--
-- Tables: 12 (8 regular + 4 hypertables)
-- Hypertables: gl_eudr_das_satellite_detections (30d chunks),
--              gl_eudr_das_alerts (30d chunks),
--              gl_eudr_das_workflow_states (30d chunks),
--              gl_eudr_das_audit_log (30d chunks)
-- Continuous Aggregates: 2 (daily_detection_summary + weekly_alert_summary)
-- Retention Policies: 4 (5y detections, 10y alerts, 5y workflow, 5y audit)
-- Indexes: ~160
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V108: Creating AGENT-EUDR-020 Deforestation Alert System tables...';

-- ============================================================================
-- 1. gl_eudr_das_satellite_detections — Satellite change detections (hypertable)
-- ============================================================================
RAISE NOTICE 'V108 [1/12]: Creating gl_eudr_das_satellite_detections (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_satellite_detections (
    detection_id                UUID            DEFAULT gen_random_uuid(),
    detection_time              TIMESTAMPTZ     NOT NULL,
        -- Timestamp when the satellite acquired the imagery
    source                      VARCHAR(50)     NOT NULL,
        -- Satellite data source identifier
    latitude                    DOUBLE PRECISION NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
        -- WGS-84 latitude of detection centroid
    longitude                   DOUBLE PRECISION NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
        -- WGS-84 longitude of detection centroid
    geometry_wkt                TEXT,
        -- Well-Known Text representation of detection polygon
    area_ha                     DECIMAL(12,4)   CHECK (area_ha >= 0),
        -- Detected change area in hectares
    change_type                 VARCHAR(30),
        -- Type of land cover change detected
    confidence                  DECIMAL(5,4)    CHECK (confidence >= 0 AND confidence <= 1),
        -- Detection confidence score (0.0 = no confidence, 1.0 = certain)
    ndvi_before                 DECIMAL(6,4),
        -- Normalized Difference Vegetation Index before change
    ndvi_after                  DECIMAL(6,4),
        -- NDVI after change
    ndvi_change                 DECIMAL(7,4),
        -- NDVI delta (after - before)
    evi_before                  DECIMAL(6,4),
        -- Enhanced Vegetation Index before change
    evi_after                   DECIMAL(6,4),
        -- EVI after change
    evi_change                  DECIMAL(7,4),
        -- EVI delta (after - before)
    cloud_cover_pct             DECIMAL(5,2)    CHECK (cloud_cover_pct >= 0 AND cloud_cover_pct <= 100),
        -- Cloud cover percentage over tile at acquisition time
    resolution_m                DECIMAL(8,2),
        -- Spatial resolution in meters
    tile_id                     VARCHAR(50),
        -- Satellite tile/scene identifier
    country_code                CHAR(3),
        -- ISO 3166-1 alpha-3 country code
    metadata                    JSONB           DEFAULT '{}',
        -- Additional satellite-specific attributes
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for data integrity verification
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (detection_id, detection_time),

    CONSTRAINT chk_das_det_source CHECK (source IN (
        'sentinel2', 'landsat8', 'landsat9', 'glad',
        'hansen_gfc', 'radd', 'planet', 'custom'
    )),
    CONSTRAINT chk_das_det_change_type CHECK (change_type IN (
        'deforestation', 'degradation', 'fire', 'logging',
        'clearing', 'regrowth', 'no_change'
    ))
);

SELECT create_hypertable(
    'gl_eudr_das_satellite_detections',
    'detection_time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_source ON gl_eudr_das_satellite_detections (source, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_lat ON gl_eudr_das_satellite_detections (latitude, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_lon ON gl_eudr_das_satellite_detections (longitude, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_lat_lon ON gl_eudr_das_satellite_detections (latitude, longitude, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_area ON gl_eudr_das_satellite_detections (area_ha, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_change_type ON gl_eudr_das_satellite_detections (change_type, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_confidence ON gl_eudr_das_satellite_detections (confidence, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_cloud ON gl_eudr_das_satellite_detections (cloud_cover_pct, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_tile ON gl_eudr_das_satellite_detections (tile_id, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_country ON gl_eudr_das_satellite_detections (country_code, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_provenance ON gl_eudr_das_satellite_detections (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_tenant ON gl_eudr_das_satellite_detections (tenant_id, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_created ON gl_eudr_das_satellite_detections (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_source_country ON gl_eudr_das_satellite_detections (source, country_code, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_country_change ON gl_eudr_das_satellite_detections (country_code, change_type, detection_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-confidence deforestation detections
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_deforestation ON gl_eudr_das_satellite_detections (detection_time DESC, area_ha DESC)
        WHERE change_type = 'deforestation' AND confidence >= 0.7;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_det_metadata ON gl_eudr_das_satellite_detections USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_satellite_detections IS 'Satellite-derived land cover change detections from multiple sources (Sentinel-2, Landsat, GLAD, RADD, Hansen GFC, Planet)';
COMMENT ON COLUMN gl_eudr_das_satellite_detections.confidence IS 'Detection confidence: 0.0 = no confidence, 1.0 = certain';
COMMENT ON COLUMN gl_eudr_das_satellite_detections.ndvi_change IS 'NDVI delta (after - before): negative values indicate vegetation loss';


-- ============================================================================
-- 2. gl_eudr_das_alerts — Alert lifecycle records (hypertable)
-- ============================================================================
RAISE NOTICE 'V108 [2/12]: Creating gl_eudr_das_alerts (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_alerts (
    alert_id                    UUID            DEFAULT gen_random_uuid(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Alert creation timestamp (partition key)
    detection_id                UUID,
        -- Reference to the satellite detection that triggered this alert
    severity                    VARCHAR(20)     NOT NULL,
        -- Alert severity level
    status                      VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Current alert lifecycle status
    title                       VARCHAR(500)    NOT NULL,
        -- Human-readable alert title
    description                 TEXT,
        -- Detailed alert description with context
    area_ha                     DECIMAL(12,4),
        -- Affected area in hectares
    latitude                    DOUBLE PRECISION,
        -- WGS-84 latitude of alert location
    longitude                   DOUBLE PRECISION,
        -- WGS-84 longitude of alert location
    country_code                CHAR(3),
        -- ISO 3166-1 alpha-3 country code
    commodity                   VARCHAR(30),
        -- EUDR-relevant commodity (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
    affected_plots              JSONB           DEFAULT '[]',
        -- Array of plot IDs affected by this alert
    proximity_km                DECIMAL(10,3),
        -- Distance in km from nearest monitored plot
    is_post_cutoff              BOOLEAN,
        -- TRUE if detection occurred after EUDR cutoff date (2020-12-31)
    severity_score              DECIMAL(5,2)    CHECK (severity_score >= 0 AND severity_score <= 100),
        -- Numeric severity score for ranking
    assigned_to                 VARCHAR(200),
        -- User or team assigned to investigate
    resolved_at                 TIMESTAMPTZ,
        -- Timestamp when alert was resolved
    resolution_notes            TEXT,
        -- Notes documenting resolution or false-positive determination
    metadata                    JSONB           DEFAULT '{}',
        -- Additional alert-specific attributes
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for data integrity verification
    tenant_id                   UUID            NOT NULL,
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (alert_id, created_at),

    CONSTRAINT chk_das_alert_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'informational'
    )),
    CONSTRAINT chk_das_alert_status CHECK (status IN (
        'pending', 'triaged', 'investigating', 'resolved',
        'escalated', 'false_positive', 'expired'
    ))
);

SELECT create_hypertable(
    'gl_eudr_das_alerts',
    'created_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_detection ON gl_eudr_das_alerts (detection_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_severity ON gl_eudr_das_alerts (severity, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_status ON gl_eudr_das_alerts (status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_country ON gl_eudr_das_alerts (country_code, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_commodity ON gl_eudr_das_alerts (commodity, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_post_cutoff ON gl_eudr_das_alerts (is_post_cutoff, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_score ON gl_eudr_das_alerts (severity_score DESC, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_assigned ON gl_eudr_das_alerts (assigned_to, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_resolved_at ON gl_eudr_das_alerts (resolved_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_provenance ON gl_eudr_das_alerts (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_tenant ON gl_eudr_das_alerts (tenant_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_updated ON gl_eudr_das_alerts (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_sev_status ON gl_eudr_das_alerts (severity, status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_country_sev ON gl_eudr_das_alerts (country_code, severity, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (pending/triaged/investigating/escalated) alerts
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_active ON gl_eudr_das_alerts (severity, created_at DESC)
        WHERE status IN ('pending', 'triaged', 'investigating', 'escalated');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for critical/high unresolved alerts
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_critical ON gl_eudr_das_alerts (created_at DESC, severity_score DESC)
        WHERE severity IN ('critical', 'high') AND status NOT IN ('resolved', 'false_positive', 'expired');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_plots ON gl_eudr_das_alerts USING GIN (affected_plots);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_alert_metadata ON gl_eudr_das_alerts USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_alerts IS 'Deforestation alert lifecycle records with severity scoring, assignment, and resolution tracking';
COMMENT ON COLUMN gl_eudr_das_alerts.is_post_cutoff IS 'TRUE if the detected change occurred after the EUDR cutoff date (2020-12-31)';
COMMENT ON COLUMN gl_eudr_das_alerts.severity_score IS 'Numeric severity score (0-100) computed from area, rate, proximity, protected status, and timing';


-- ============================================================================
-- 3. gl_eudr_das_severity_scores — Multi-factor severity scoring
-- ============================================================================
RAISE NOTICE 'V108 [3/12]: Creating gl_eudr_das_severity_scores...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_severity_scores (
    score_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id                    UUID            NOT NULL,
        -- Reference to the alert being scored
    area_score                  DECIMAL(5,2),
        -- Score contribution from affected area size
    rate_score                  DECIMAL(5,2),
        -- Score contribution from deforestation rate
    proximity_score             DECIMAL(5,2),
        -- Score contribution from proximity to monitored plots
    protected_score             DECIMAL(5,2),
        -- Score contribution from overlap with protected areas
    timing_score                DECIMAL(5,2),
        -- Score contribution from EUDR cutoff date proximity
    area_weight                 DECIMAL(4,3),
        -- Weight applied to area_score in total calculation
    rate_weight                 DECIMAL(4,3),
        -- Weight applied to rate_score
    proximity_weight            DECIMAL(4,3),
        -- Weight applied to proximity_score
    protected_weight            DECIMAL(4,3),
        -- Weight applied to protected_score
    timing_weight               DECIMAL(4,3),
        -- Weight applied to timing_score
    total_score                 DECIMAL(5,2),
        -- Weighted sum of all component scores
    severity_level              VARCHAR(20),
        -- Derived severity: critical, high, medium, low, informational
    contributing_factors        JSONB,
        -- { "large_area": true, "near_protected": true, "post_cutoff": true, ... }
    aggravating_factors         JSONB,
        -- { "repeat_offender": true, "high_biodiversity_zone": true, ... }
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for calculation integrity verification
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_alert ON gl_eudr_das_severity_scores (alert_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_total ON gl_eudr_das_severity_scores (total_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_level ON gl_eudr_das_severity_scores (severity_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_area ON gl_eudr_das_severity_scores (area_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_proximity ON gl_eudr_das_severity_scores (proximity_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_protected ON gl_eudr_das_severity_scores (protected_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_timing ON gl_eudr_das_severity_scores (timing_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_provenance ON gl_eudr_das_severity_scores (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_tenant ON gl_eudr_das_severity_scores (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_created ON gl_eudr_das_severity_scores (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_level_total ON gl_eudr_das_severity_scores (severity_level, total_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_contrib ON gl_eudr_das_severity_scores USING GIN (contributing_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_sev_aggravating ON gl_eudr_das_severity_scores USING GIN (aggravating_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_severity_scores IS 'Multi-factor severity scoring with weighted area, rate, proximity, protected-area, and timing components';
COMMENT ON COLUMN gl_eudr_das_severity_scores.total_score IS 'Weighted sum: SUM(component_score * component_weight) across all five dimensions';
COMMENT ON COLUMN gl_eudr_das_severity_scores.severity_level IS 'Derived from total_score: critical (>=80), high (>=60), medium (>=40), low (>=20), informational (<20)';


-- ============================================================================
-- 4. gl_eudr_das_spatial_buffers — Plot buffer zones for monitoring
-- ============================================================================
RAISE NOTICE 'V108 [4/12]: Creating gl_eudr_das_spatial_buffers...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_spatial_buffers (
    buffer_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     VARCHAR(100)    NOT NULL,
        -- Reference to the monitored production plot
    center_latitude             DOUBLE PRECISION NOT NULL CHECK (center_latitude >= -90 AND center_latitude <= 90),
        -- WGS-84 latitude of buffer center
    center_longitude            DOUBLE PRECISION NOT NULL CHECK (center_longitude >= -180 AND center_longitude <= 180),
        -- WGS-84 longitude of buffer center
    radius_km                   DECIMAL(8,3)    NOT NULL CHECK (radius_km >= 0.1 AND radius_km <= 100),
        -- Buffer radius in kilometers
    buffer_type                 VARCHAR(20)     NOT NULL DEFAULT 'circular',
        -- Buffer geometry type
    geometry_wkt                TEXT,
        -- Well-Known Text representation of buffer polygon
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this buffer zone is actively monitored
    commodities                 JSONB           DEFAULT '[]',
        -- Array of EUDR commodities associated with this plot
    country_code                CHAR(3),
        -- ISO 3166-1 alpha-3 country code
    metadata                    JSONB           DEFAULT '{}',
        -- Additional buffer configuration attributes
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for configuration integrity
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_das_buf_type CHECK (buffer_type IN (
        'circular', 'polygon', 'adaptive'
    )),
    CONSTRAINT uq_das_buf_plot UNIQUE (plot_id, tenant_id)
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_plot ON gl_eudr_das_spatial_buffers (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_lat_lon ON gl_eudr_das_spatial_buffers (center_latitude, center_longitude);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_radius ON gl_eudr_das_spatial_buffers (radius_km);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_type ON gl_eudr_das_spatial_buffers (buffer_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_active ON gl_eudr_das_spatial_buffers (is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_country ON gl_eudr_das_spatial_buffers (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_provenance ON gl_eudr_das_spatial_buffers (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_tenant ON gl_eudr_das_spatial_buffers (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_created ON gl_eudr_das_spatial_buffers (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_updated ON gl_eudr_das_spatial_buffers (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active buffer zones only
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_active_zones ON gl_eudr_das_spatial_buffers (center_latitude, center_longitude, radius_km)
        WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_commodities ON gl_eudr_das_spatial_buffers USING GIN (commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_buf_metadata ON gl_eudr_das_spatial_buffers USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_spatial_buffers IS 'Spatial buffer zones around monitored production plots for proximity-based deforestation alert triggering';
COMMENT ON COLUMN gl_eudr_das_spatial_buffers.radius_km IS 'Buffer radius in km: min 0.1 (100m) to max 100 (100km) depending on commodity and risk';
COMMENT ON COLUMN gl_eudr_das_spatial_buffers.buffer_type IS 'Buffer geometry: circular (radius-based), polygon (custom boundary), adaptive (risk-adjusted)';


-- ============================================================================
-- 5. gl_eudr_das_buffer_violations — Detections within buffer zones
-- ============================================================================
RAISE NOTICE 'V108 [5/12]: Creating gl_eudr_das_buffer_violations...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_buffer_violations (
    violation_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    buffer_id                   UUID            NOT NULL,
        -- Reference to the violated spatial buffer
    detection_id                UUID            NOT NULL,
        -- Reference to the satellite detection causing the violation
    distance_km                 DECIMAL(10,3),
        -- Distance in km between detection centroid and buffer center
    overlap_area_ha             DECIMAL(12,4),
        -- Area of overlap between detection polygon and buffer zone
    violation_time              TIMESTAMPTZ,
        -- Timestamp of the detection that caused the violation
    metadata                    JSONB           DEFAULT '{}',
        -- Additional violation context
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for violation determination integrity
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_buffer ON gl_eudr_das_buffer_violations (buffer_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_detection ON gl_eudr_das_buffer_violations (detection_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_distance ON gl_eudr_das_buffer_violations (distance_km);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_overlap ON gl_eudr_das_buffer_violations (overlap_area_ha DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_time ON gl_eudr_das_buffer_violations (violation_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_provenance ON gl_eudr_das_buffer_violations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_tenant ON gl_eudr_das_buffer_violations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_created ON gl_eudr_das_buffer_violations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_buffer_time ON gl_eudr_das_buffer_violations (buffer_id, violation_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_buffer_dist ON gl_eudr_das_buffer_violations (buffer_id, distance_km);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bv_metadata ON gl_eudr_das_buffer_violations USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_buffer_violations IS 'Records of satellite detections that fall within or overlap monitored spatial buffer zones';
COMMENT ON COLUMN gl_eudr_das_buffer_violations.distance_km IS 'Haversine distance from detection centroid to buffer center in km';
COMMENT ON COLUMN gl_eudr_das_buffer_violations.overlap_area_ha IS 'Intersection area between detection polygon and buffer zone in hectares';


-- ============================================================================
-- 6. gl_eudr_das_cutoff_verifications — EUDR cutoff date verification
-- ============================================================================
RAISE NOTICE 'V108 [6/12]: Creating gl_eudr_das_cutoff_verifications...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_cutoff_verifications (
    verification_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id                UUID            NOT NULL,
        -- Reference to the satellite detection being verified
    cutoff_result               VARCHAR(20)     NOT NULL,
        -- Temporal classification relative to EUDR cutoff date (2020-12-31)
    confidence                  DECIMAL(5,4)    CHECK (confidence >= 0 AND confidence <= 1),
        -- Confidence in the cutoff determination (0.0 to 1.0)
    evidence_sources            JSONB           DEFAULT '[]',
        -- Array of evidence items supporting the determination
        -- [{ "source": "sentinel2", "date": "2020-11-15", "type": "pre_cutoff_clear" }, ...]
    earliest_detection_date     DATE,
        -- Earliest date when change was first detected
    latest_clear_date           DATE,
        -- Latest date when area was confirmed to have forest cover
    temporal_analysis           JSONB,
        -- { "time_series": [...], "change_point": "2021-03-15", "method": "bfast" }
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for verification integrity
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_das_cutoff_result CHECK (cutoff_result IN (
        'pre_cutoff', 'post_cutoff', 'uncertain', 'ongoing'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_detection ON gl_eudr_das_cutoff_verifications (detection_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_result ON gl_eudr_das_cutoff_verifications (cutoff_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_confidence ON gl_eudr_das_cutoff_verifications (confidence DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_earliest ON gl_eudr_das_cutoff_verifications (earliest_detection_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_latest_clear ON gl_eudr_das_cutoff_verifications (latest_clear_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_provenance ON gl_eudr_das_cutoff_verifications (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_tenant ON gl_eudr_das_cutoff_verifications (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_created ON gl_eudr_das_cutoff_verifications (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_result_conf ON gl_eudr_das_cutoff_verifications (cutoff_result, confidence DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for post-cutoff findings requiring action
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_post_cutoff ON gl_eudr_das_cutoff_verifications (confidence DESC, created_at DESC)
        WHERE cutoff_result = 'post_cutoff';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_evidence ON gl_eudr_das_cutoff_verifications USING GIN (evidence_sources);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_cv_temporal ON gl_eudr_das_cutoff_verifications USING GIN (temporal_analysis);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_cutoff_verifications IS 'EUDR cutoff date (2020-12-31) verification determining whether detected changes are pre- or post-cutoff';
COMMENT ON COLUMN gl_eudr_das_cutoff_verifications.cutoff_result IS 'pre_cutoff = before 2020-12-31, post_cutoff = after (non-compliant), uncertain = insufficient data, ongoing = still changing';
COMMENT ON COLUMN gl_eudr_das_cutoff_verifications.latest_clear_date IS 'Last confirmed date when the area had intact forest cover (used for temporal bracketing)';


-- ============================================================================
-- 7. gl_eudr_das_historical_baselines — Forest cover baselines
-- ============================================================================
RAISE NOTICE 'V108 [7/12]: Creating gl_eudr_das_historical_baselines...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_historical_baselines (
    baseline_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     VARCHAR(100)    NOT NULL,
        -- Reference to the monitored production plot
    latitude                    DOUBLE PRECISION NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
        -- WGS-84 latitude of baseline measurement point
    longitude                   DOUBLE PRECISION NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
        -- WGS-84 longitude of baseline measurement point
    baseline_start_date         DATE            NOT NULL,
        -- Start of the baseline reference period
    baseline_end_date           DATE            NOT NULL,
        -- End of the baseline reference period
    canopy_cover_pct            DECIMAL(5,2)    CHECK (canopy_cover_pct >= 0 AND canopy_cover_pct <= 100),
        -- Percentage canopy cover during baseline period
    forest_area_ha              DECIMAL(12,4),
        -- Total forest area within plot boundary during baseline
    reference_sources           JSONB           DEFAULT '[]',
        -- [{ "source": "hansen_gfc", "year": 2020, "resolution_m": 30 }, ...]
    metadata                    JSONB           DEFAULT '{}',
        -- Additional baseline characterization data
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for baseline integrity
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_plot ON gl_eudr_das_historical_baselines (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_lat_lon ON gl_eudr_das_historical_baselines (latitude, longitude);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_start ON gl_eudr_das_historical_baselines (baseline_start_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_end ON gl_eudr_das_historical_baselines (baseline_end_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_canopy ON gl_eudr_das_historical_baselines (canopy_cover_pct);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_forest_area ON gl_eudr_das_historical_baselines (forest_area_ha);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_provenance ON gl_eudr_das_historical_baselines (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_tenant ON gl_eudr_das_historical_baselines (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_created ON gl_eudr_das_historical_baselines (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_updated ON gl_eudr_das_historical_baselines (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_plot_dates ON gl_eudr_das_historical_baselines (plot_id, baseline_start_date, baseline_end_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_ref_sources ON gl_eudr_das_historical_baselines USING GIN (reference_sources);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_hb_metadata ON gl_eudr_das_historical_baselines USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_historical_baselines IS 'Historical forest cover baselines per plot for comparison against current satellite observations';
COMMENT ON COLUMN gl_eudr_das_historical_baselines.canopy_cover_pct IS 'Percentage tree canopy cover during the baseline reference period (0-100)';
COMMENT ON COLUMN gl_eudr_das_historical_baselines.reference_sources IS 'Array of data sources used to establish the baseline (Hansen GFC, Sentinel-2 composites, etc.)';


-- ============================================================================
-- 8. gl_eudr_das_baseline_comparisons — Baseline vs current comparisons
-- ============================================================================
RAISE NOTICE 'V108 [8/12]: Creating gl_eudr_das_baseline_comparisons...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_baseline_comparisons (
    comparison_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_id                 UUID            NOT NULL,
        -- Reference to the historical baseline being compared
    comparison_date             TIMESTAMPTZ     NOT NULL,
        -- Date of the current observation being compared
    current_canopy_pct          DECIMAL(5,2)    CHECK (current_canopy_pct >= 0 AND current_canopy_pct <= 100),
        -- Current canopy cover percentage at comparison date
    canopy_change_pct           DECIMAL(6,2),
        -- Change in canopy cover (current - baseline), negative = loss
    area_change_ha              DECIMAL(12,4),
        -- Change in forest area in hectares (negative = loss)
    change_type                 VARCHAR(30),
        -- Type of change detected relative to baseline
    confidence                  DECIMAL(5,4)    CHECK (confidence >= 0 AND confidence <= 1),
        -- Confidence in the comparison result
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for comparison integrity
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_das_bc_change_type CHECK (change_type IN (
        'deforestation', 'degradation', 'fire', 'logging',
        'clearing', 'regrowth', 'no_change'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_baseline ON gl_eudr_das_baseline_comparisons (baseline_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_date ON gl_eudr_das_baseline_comparisons (comparison_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_canopy_pct ON gl_eudr_das_baseline_comparisons (current_canopy_pct);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_canopy_change ON gl_eudr_das_baseline_comparisons (canopy_change_pct);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_area_change ON gl_eudr_das_baseline_comparisons (area_change_ha);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_change_type ON gl_eudr_das_baseline_comparisons (change_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_confidence ON gl_eudr_das_baseline_comparisons (confidence DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_provenance ON gl_eudr_das_baseline_comparisons (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_tenant ON gl_eudr_das_baseline_comparisons (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_created ON gl_eudr_das_baseline_comparisons (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_baseline_date ON gl_eudr_das_baseline_comparisons (baseline_id, comparison_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_change_conf ON gl_eudr_das_baseline_comparisons (change_type, confidence DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for significant deforestation comparisons
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_bc_deforestation ON gl_eudr_das_baseline_comparisons (canopy_change_pct, area_change_ha)
        WHERE change_type = 'deforestation' AND confidence >= 0.7;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_baseline_comparisons IS 'Comparison of current satellite observations against historical forest baselines to quantify change';
COMMENT ON COLUMN gl_eudr_das_baseline_comparisons.canopy_change_pct IS 'Canopy cover change (current - baseline): negative values indicate forest loss';
COMMENT ON COLUMN gl_eudr_das_baseline_comparisons.area_change_ha IS 'Forest area change in hectares: negative values indicate deforestation';


-- ============================================================================
-- 9. gl_eudr_das_workflow_states — Alert workflow state machine (hypertable)
-- ============================================================================
RAISE NOTICE 'V108 [9/12]: Creating gl_eudr_das_workflow_states (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_workflow_states (
    state_id                    UUID            DEFAULT gen_random_uuid(),
    alert_id                    UUID            NOT NULL,
        -- Reference to the alert whose state is transitioning
    current_status              VARCHAR(20)     NOT NULL,
        -- New status after transition
    previous_status             VARCHAR(20),
        -- Status before transition (NULL for initial state)
    action                      VARCHAR(30)     NOT NULL,
        -- Action that triggered the state transition
    actor                       VARCHAR(200),
        -- User or system agent performing the action
    assigned_to                 VARCHAR(200),
        -- User or team to whom the alert is assigned after this action
    priority                    VARCHAR(20),
        -- Priority classification: urgent, high, normal, low
    sla_deadline                TIMESTAMPTZ,
        -- Service level agreement deadline for this state
    notes                       TEXT,
        -- Notes accompanying the state transition
    transition_time             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the state transition (partition key)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional workflow context
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for workflow integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (state_id, transition_time),

    CONSTRAINT chk_das_wf_action CHECK (action IN (
        'triage', 'assign', 'investigate', 'resolve',
        'escalate', 'close', 'reopen'
    ))
);

SELECT create_hypertable(
    'gl_eudr_das_workflow_states',
    'transition_time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_alert ON gl_eudr_das_workflow_states (alert_id, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_current ON gl_eudr_das_workflow_states (current_status, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_previous ON gl_eudr_das_workflow_states (previous_status, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_action ON gl_eudr_das_workflow_states (action, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_actor ON gl_eudr_das_workflow_states (actor, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_assigned ON gl_eudr_das_workflow_states (assigned_to, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_priority ON gl_eudr_das_workflow_states (priority, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_sla ON gl_eudr_das_workflow_states (sla_deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_provenance ON gl_eudr_das_workflow_states (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_tenant ON gl_eudr_das_workflow_states (tenant_id, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_alert_action ON gl_eudr_das_workflow_states (alert_id, action, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_status_action ON gl_eudr_das_workflow_states (current_status, action, transition_time DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for SLA-breached items
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_sla_breach ON gl_eudr_das_workflow_states (sla_deadline, alert_id)
        WHERE sla_deadline IS NOT NULL AND current_status NOT IN ('resolved', 'false_positive', 'expired');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_wf_metadata ON gl_eudr_das_workflow_states USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_workflow_states IS 'Alert workflow state machine tracking all status transitions with actor, assignment, priority, and SLA data';
COMMENT ON COLUMN gl_eudr_das_workflow_states.action IS 'Workflow transition action: triage, assign, investigate, resolve, escalate, close, reopen';
COMMENT ON COLUMN gl_eudr_das_workflow_states.sla_deadline IS 'SLA deadline for the current state (e.g. critical alerts must be triaged within 4 hours)';


-- ============================================================================
-- 10. gl_eudr_das_compliance_impacts — EUDR compliance impact assessments
-- ============================================================================
RAISE NOTICE 'V108 [10/12]: Creating gl_eudr_das_compliance_impacts...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_compliance_impacts (
    impact_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id                    UUID            NOT NULL,
        -- Reference to the alert triggering the compliance assessment
    compliance_outcome          VARCHAR(30)     NOT NULL,
        -- Compliance determination based on alert investigation
    affected_suppliers          JSONB           DEFAULT '[]',
        -- Array of supplier records affected by this alert
        -- [{ "supplier_id": "...", "name": "...", "tier": 1, "risk_level": "high" }]
    affected_products           JSONB           DEFAULT '[]',
        -- Array of product records affected
        -- [{ "product_id": "...", "hs_code": "...", "name": "...", "volume_tonnes": 150 }]
    market_restriction          BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this triggers an EU market access restriction
    remediation_actions         JSONB           DEFAULT '[]',
        -- [{ "action": "supplier_audit", "deadline": "2026-06-01", "status": "pending" }]
    estimated_financial_impact  DECIMAL(15,2),
        -- Estimated financial impact in EUR
    assessment_notes            TEXT,
        -- Detailed notes on compliance assessment rationale
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for assessment integrity
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_das_ci_outcome CHECK (compliance_outcome IN (
        'compliant', 'non_compliant', 'under_review', 'remediation_required'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_alert ON gl_eudr_das_compliance_impacts (alert_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_outcome ON gl_eudr_das_compliance_impacts (compliance_outcome);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_restriction ON gl_eudr_das_compliance_impacts (market_restriction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_financial ON gl_eudr_das_compliance_impacts (estimated_financial_impact DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_provenance ON gl_eudr_das_compliance_impacts (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_tenant ON gl_eudr_das_compliance_impacts (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_created ON gl_eudr_das_compliance_impacts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_updated ON gl_eudr_das_compliance_impacts (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_outcome_restr ON gl_eudr_das_compliance_impacts (compliance_outcome, market_restriction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for non-compliant outcomes requiring action
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_non_compliant ON gl_eudr_das_compliance_impacts (created_at DESC, estimated_financial_impact DESC)
        WHERE compliance_outcome IN ('non_compliant', 'remediation_required');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_suppliers ON gl_eudr_das_compliance_impacts USING GIN (affected_suppliers);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_products ON gl_eudr_das_compliance_impacts USING GIN (affected_products);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_ci_remediation ON gl_eudr_das_compliance_impacts USING GIN (remediation_actions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_compliance_impacts IS 'EUDR compliance impact assessments linking deforestation alerts to affected suppliers, products, and market access';
COMMENT ON COLUMN gl_eudr_das_compliance_impacts.compliance_outcome IS 'compliant, non_compliant (market restriction), under_review (investigation ongoing), remediation_required (corrective action needed)';
COMMENT ON COLUMN gl_eudr_das_compliance_impacts.market_restriction IS 'TRUE if this alert triggers an EU market access restriction under EUDR Article 4';


-- ============================================================================
-- 11. gl_eudr_das_alert_notifications — Notification dispatch records
-- ============================================================================
RAISE NOTICE 'V108 [11/12]: Creating gl_eudr_das_alert_notifications...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_alert_notifications (
    notification_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id                    UUID            NOT NULL,
        -- Reference to the alert that triggered this notification
    channel                     VARCHAR(30)     NOT NULL,
        -- Delivery channel for the notification
    recipient                   VARCHAR(500)    NOT NULL,
        -- Recipient address or identifier
    sent_at                     TIMESTAMPTZ,
        -- Timestamp when notification was sent
    status                      VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Notification delivery status
    retry_count                 INTEGER         NOT NULL DEFAULT 0,
        -- Number of delivery retry attempts
    metadata                    JSONB           DEFAULT '{}',
        -- Additional notification context (subject, template, payload)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_das_notif_channel CHECK (channel IN (
        'email', 'webhook', 'sms', 'dashboard', 'slack'
    )),
    CONSTRAINT chk_das_notif_status CHECK (status IN (
        'pending', 'sent', 'delivered', 'failed'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_alert ON gl_eudr_das_alert_notifications (alert_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_channel ON gl_eudr_das_alert_notifications (channel);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_recipient ON gl_eudr_das_alert_notifications (recipient);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_sent ON gl_eudr_das_alert_notifications (sent_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_status ON gl_eudr_das_alert_notifications (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_retry ON gl_eudr_das_alert_notifications (retry_count);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_tenant ON gl_eudr_das_alert_notifications (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_created ON gl_eudr_das_alert_notifications (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_alert_ch ON gl_eudr_das_alert_notifications (alert_id, channel);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_ch_status ON gl_eudr_das_alert_notifications (channel, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for pending/failed notifications requiring processing
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_pending ON gl_eudr_das_alert_notifications (created_at DESC, retry_count)
        WHERE status IN ('pending', 'failed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_notif_metadata ON gl_eudr_das_alert_notifications USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_alert_notifications IS 'Notification dispatch records for deforestation alerts across email, webhook, SMS, dashboard, and Slack channels';
COMMENT ON COLUMN gl_eudr_das_alert_notifications.status IS 'Delivery status: pending (queued), sent (dispatched), delivered (confirmed), failed (after retries)';
COMMENT ON COLUMN gl_eudr_das_alert_notifications.retry_count IS 'Number of retry attempts; notifications are retried up to 3 times with exponential backoff';


-- ============================================================================
-- 12. gl_eudr_das_audit_log — Comprehensive audit trail (hypertable, 30d)
-- ============================================================================
RAISE NOTICE 'V108 [12/12]: Creating gl_eudr_das_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_das_audit_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    event_type                  VARCHAR(50)     NOT NULL,
        -- 'detection_ingested', 'alert_created', 'alert_triaged', 'alert_assigned',
        -- 'alert_investigated', 'alert_resolved', 'alert_escalated', 'alert_closed',
        -- 'severity_scored', 'buffer_created', 'buffer_updated', 'buffer_violated',
        -- 'cutoff_verified', 'baseline_created', 'baseline_compared',
        -- 'workflow_transitioned', 'compliance_assessed', 'notification_sent',
        -- 'notification_failed', 'data_refreshed'
    entity_type                 VARCHAR(50)     NOT NULL,
        -- 'satellite_detection', 'alert', 'severity_score', 'spatial_buffer',
        -- 'buffer_violation', 'cutoff_verification', 'historical_baseline',
        -- 'baseline_comparison', 'workflow_state', 'compliance_impact',
        -- 'alert_notification'
    entity_id                   VARCHAR(100)    NOT NULL,
        -- UUID of the entity being audited
    actor                       VARCHAR(100)    NOT NULL,
        -- User ID or system agent identifier
    details                     JSONB,
        -- { "changed_fields": [...], "old_values": {...}, "new_values": {...},
        --   "reason": "...", "detection_source": "sentinel2" }
    ip_address                  INET,
        -- Source IP address of the actor
    user_agent                  TEXT,
        -- HTTP user agent or system agent name
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for immutability verification
    metadata                    JSONB           DEFAULT '{}',
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (event_id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_das_audit_log',
    'created_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_event_type ON gl_eudr_das_audit_log (event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_entity_type ON gl_eudr_das_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_entity_id ON gl_eudr_das_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_actor ON gl_eudr_das_audit_log (actor, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_provenance ON gl_eudr_das_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_tenant ON gl_eudr_das_audit_log (tenant_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_entity_action ON gl_eudr_das_audit_log (entity_type, event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_details ON gl_eudr_das_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_das_audit_metadata ON gl_eudr_das_audit_log USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_das_audit_log IS 'Comprehensive audit trail for all deforestation alert system operations and state changes';
COMMENT ON COLUMN gl_eudr_das_audit_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Daily detection summary by source, country, change_type
RAISE NOTICE 'V108: Creating continuous aggregate: daily_detection_summary...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_das_daily_detection_summary
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', detection_time)    AS day,
        tenant_id,
        source,
        country_code,
        change_type,
        COUNT(*)                                AS detection_count,
        SUM(area_ha)                            AS total_area_ha,
        AVG(confidence)                         AS avg_confidence,
        AVG(cloud_cover_pct)                    AS avg_cloud_cover_pct,
        AVG(ndvi_change)                        AS avg_ndvi_change,
        MIN(confidence)                         AS min_confidence,
        MAX(area_ha)                            AS max_area_ha
    FROM gl_eudr_das_satellite_detections
    GROUP BY day, tenant_id, source, country_code, change_type;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_das_daily_detection_summary',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_das_daily_detection_summary IS 'Daily rollup of satellite detections by source, country, and change type with area, confidence, and vegetation index statistics';


-- Weekly alert summary by severity, status, country
RAISE NOTICE 'V108: Creating continuous aggregate: weekly_alert_summary...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_das_weekly_alert_summary
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('7 days', created_at)       AS week,
        tenant_id,
        severity,
        status,
        country_code,
        COUNT(*)                                AS alert_count,
        SUM(area_ha)                            AS total_area_ha,
        AVG(severity_score)                     AS avg_severity_score,
        AVG(proximity_km)                       AS avg_proximity_km,
        SUM(CASE WHEN is_post_cutoff = TRUE THEN 1 ELSE 0 END) AS post_cutoff_count
    FROM gl_eudr_das_alerts
    GROUP BY week, tenant_id, severity, status, country_code;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_das_weekly_alert_summary',
        start_offset => INTERVAL '14 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_das_weekly_alert_summary IS 'Weekly rollup of alerts by severity, status, and country with area, severity score, proximity, and cutoff statistics';


-- ============================================================================
-- RETENTION POLICIES
-- ============================================================================

RAISE NOTICE 'V108: Creating retention policies...';

-- 5 years for satellite detections (regulatory evidence retention)
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_das_satellite_detections', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 10 years for alerts (long-term compliance audit trail)
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_das_alerts', INTERVAL '10 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for workflow states
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_das_workflow_states', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_das_audit_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V108: AGENT-EUDR-020 Deforestation Alert System tables created successfully!';
RAISE NOTICE 'V108: Created 12 tables (4 hypertables), 2 continuous aggregates, ~160 indexes';
RAISE NOTICE 'V108: Retention policies: 5y detections, 10y alerts, 5y workflow, 5y audit logs';

COMMIT;
