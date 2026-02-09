-- =============================================================================
-- V037: Deforestation Satellite Connector Service Schema
-- =============================================================================
-- Component: AGENT-DATA-007 (Deforestation Satellite Connector)
-- Agent ID:  GL-DATA-GEO-003
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Deforestation Satellite Connector Agent (GL-DATA-GEO-003) with capabilities
-- for satellite scene management, vegetation index computation,
-- forest assessment and EUDR compliance checking, deforestation alert
-- ingestion, change detection analysis, baseline verification,
-- compliance reporting, monitoring job orchestration, land cover
-- classification, and pipeline metrics tracking.
-- =============================================================================
-- Tables (10):
--   1. satellite_scenes          - Satellite imagery scene metadata and bounds
--   2. vegetation_indices        - Computed vegetation indices per scene
--   3. forest_assessments        - Forest status and EUDR compliance assessments
--   4. deforestation_alerts      - Alert ingestion from GLAD/RADD/FIRMS (hypertable)
--   5. change_detections         - Pre/post change detection analysis results
--   6. baseline_checks           - Baseline forest cover verification checks
--   7. compliance_reports        - EUDR compliance report generation
--   8. monitoring_jobs           - Monitoring pipeline job orchestration (hypertable)
--   9. classification_results    - Land cover / forest classification results
--  10. pipeline_metrics          - Per-stage pipeline performance metrics (hypertable)
--
-- Continuous Aggregates (2):
--   1. hourly_alert_stats        - Hourly deforestation alert aggregates
--   2. hourly_pipeline_stats     - Hourly pipeline metrics aggregates
--
-- Also includes: 50+ indexes (B-tree, GIN), RLS policies per tenant,
-- retention policies, compression policies, security permissions, and
-- seed data registering GL-DATA-GEO-003 in the agent registry.
-- Previous: V036__gis_connector_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS deforestation_satellite_service;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION deforestation_satellite_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: deforestation_satellite_service.satellite_scenes
-- =============================================================================
-- Satellite imagery scene metadata. Each record captures a single acquisition
-- from supported satellite platforms (Sentinel-2, Landsat 8/9, MODIS,
-- Harmonized Landsat-Sentinel). Stores bounding box, spectral bands,
-- cloud cover, resolution, CRS, and tile identification. Core reference
-- table for vegetation index computation and change detection. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.satellite_scenes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scene_id VARCHAR(100) UNIQUE NOT NULL,
    satellite VARCHAR(50) NOT NULL,
    acquisition_date TIMESTAMPTZ NOT NULL,
    cloud_cover_percent DECIMAL(5,2),
    bbox_min_lon DECIMAL(12,8),
    bbox_min_lat DECIMAL(12,8),
    bbox_max_lon DECIMAL(12,8),
    bbox_max_lat DECIMAL(12,8),
    bands JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolution_m INTEGER NOT NULL DEFAULT 10,
    crs VARCHAR(50) NOT NULL DEFAULT 'EPSG:4326',
    tile_id VARCHAR(100),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

-- Satellite platform constraint
ALTER TABLE deforestation_satellite_service.satellite_scenes
    ADD CONSTRAINT chk_ss_satellite
    CHECK (satellite IN ('sentinel2', 'landsat8', 'landsat9', 'modis', 'harmonized'));

-- Cloud cover must be between 0 and 100
ALTER TABLE deforestation_satellite_service.satellite_scenes
    ADD CONSTRAINT chk_ss_cloud_cover_range
    CHECK (cloud_cover_percent IS NULL OR (cloud_cover_percent >= 0 AND cloud_cover_percent <= 100));

-- Scene ID must not be empty
ALTER TABLE deforestation_satellite_service.satellite_scenes
    ADD CONSTRAINT chk_ss_scene_id_not_empty
    CHECK (LENGTH(TRIM(scene_id)) > 0);

-- Resolution must be positive
ALTER TABLE deforestation_satellite_service.satellite_scenes
    ADD CONSTRAINT chk_ss_resolution_positive
    CHECK (resolution_m > 0);

-- CRS must not be empty
ALTER TABLE deforestation_satellite_service.satellite_scenes
    ADD CONSTRAINT chk_ss_crs_not_empty
    CHECK (LENGTH(TRIM(crs)) > 0);

-- Bounding box longitude must be between -180 and 180
ALTER TABLE deforestation_satellite_service.satellite_scenes
    ADD CONSTRAINT chk_ss_bbox_lon_range
    CHECK (
        (bbox_min_lon IS NULL OR (bbox_min_lon >= -180 AND bbox_min_lon <= 180))
        AND (bbox_max_lon IS NULL OR (bbox_max_lon >= -180 AND bbox_max_lon <= 180))
    );

-- Bounding box latitude must be between -90 and 90
ALTER TABLE deforestation_satellite_service.satellite_scenes
    ADD CONSTRAINT chk_ss_bbox_lat_range
    CHECK (
        (bbox_min_lat IS NULL OR (bbox_min_lat >= -90 AND bbox_min_lat <= 90))
        AND (bbox_max_lat IS NULL OR (bbox_max_lat >= -90 AND bbox_max_lat <= 90))
    );

-- =============================================================================
-- Table 2: deforestation_satellite_service.vegetation_indices
-- =============================================================================
-- Computed vegetation indices per satellite scene. Each record captures
-- statistical summaries (min, max, mean, std) for a specific index type
-- (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI). Used for forest health
-- assessment, change detection baselines, and deforestation analysis.
-- Linked to satellite_scenes via scene_id. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.vegetation_indices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scene_id VARCHAR(100) NOT NULL,
    index_type VARCHAR(20) NOT NULL,
    min_value DECIMAL(10,6),
    max_value DECIMAL(10,6),
    mean_value DECIMAL(10,6),
    std_value DECIMAL(10,6),
    pixel_count INTEGER,
    computation_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to satellite_scenes
ALTER TABLE deforestation_satellite_service.vegetation_indices
    ADD CONSTRAINT fk_vi_scene_id
    FOREIGN KEY (scene_id) REFERENCES deforestation_satellite_service.satellite_scenes(scene_id)
    ON DELETE CASCADE;

-- Index type constraint
ALTER TABLE deforestation_satellite_service.vegetation_indices
    ADD CONSTRAINT chk_vi_index_type
    CHECK (index_type IN ('ndvi', 'evi', 'ndwi', 'nbr', 'savi', 'msavi', 'ndmi'));

-- Scene ID must not be empty
ALTER TABLE deforestation_satellite_service.vegetation_indices
    ADD CONSTRAINT chk_vi_scene_id_not_empty
    CHECK (LENGTH(TRIM(scene_id)) > 0);

-- Pixel count must be non-negative if specified
ALTER TABLE deforestation_satellite_service.vegetation_indices
    ADD CONSTRAINT chk_vi_pixel_count_non_negative
    CHECK (pixel_count IS NULL OR pixel_count >= 0);

-- Standard deviation must be non-negative if specified
ALTER TABLE deforestation_satellite_service.vegetation_indices
    ADD CONSTRAINT chk_vi_std_non_negative
    CHECK (std_value IS NULL OR std_value >= 0);

-- =============================================================================
-- Table 3: deforestation_satellite_service.forest_assessments
-- =============================================================================
-- Forest status and EUDR compliance assessment records. Each assessment
-- evaluates a geographic point for deforestation risk against the EUDR
-- baseline date (31 Dec 2020). Captures forest cover percentages,
-- change analysis, risk scoring, compliance determination, data sources,
-- and warning conditions. Core output table for EUDR due diligence
-- support. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.forest_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id VARCHAR(100) UNIQUE NOT NULL,
    latitude DECIMAL(12,8) NOT NULL,
    longitude DECIMAL(12,8) NOT NULL,
    country_iso3 VARCHAR(3) NOT NULL,
    forest_status VARCHAR(50) NOT NULL,
    is_eudr_compliant BOOLEAN NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    risk_score DECIMAL(5,2),
    baseline_forest_cover_percent DECIMAL(5,2),
    current_forest_cover_percent DECIMAL(5,2),
    forest_cover_change_percent DECIMAL(8,4),
    baseline_date DATE NOT NULL DEFAULT '2020-12-31',
    assessment_date DATE NOT NULL,
    forest_definition JSONB,
    data_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

-- Risk level constraint
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_risk_level
    CHECK (risk_level IN ('low', 'medium', 'high', 'critical', 'unknown'));

-- Risk score must be between 0 and 100
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_risk_score_range
    CHECK (risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100));

-- Assessment ID must not be empty
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_assessment_id_not_empty
    CHECK (LENGTH(TRIM(assessment_id)) > 0);

-- Country ISO3 must be exactly 3 characters
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_country_iso3_length
    CHECK (LENGTH(TRIM(country_iso3)) = 3);

-- Latitude must be between -90 and 90
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_latitude_range
    CHECK (latitude >= -90 AND latitude <= 90);

-- Longitude must be between -180 and 180
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_longitude_range
    CHECK (longitude >= -180 AND longitude <= 180);

-- Forest cover percentages must be between 0 and 100
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_baseline_cover_range
    CHECK (baseline_forest_cover_percent IS NULL OR (baseline_forest_cover_percent >= 0 AND baseline_forest_cover_percent <= 100));

ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_current_cover_range
    CHECK (current_forest_cover_percent IS NULL OR (current_forest_cover_percent >= 0 AND current_forest_cover_percent <= 100));

-- Forest status constraint
ALTER TABLE deforestation_satellite_service.forest_assessments
    ADD CONSTRAINT chk_fa_forest_status
    CHECK (forest_status IN (
        'intact', 'degraded', 'deforested', 'reforested',
        'non_forest', 'partially_deforested', 'unknown'
    ));

-- =============================================================================
-- Table 4: deforestation_satellite_service.deforestation_alerts (hypertable)
-- =============================================================================
-- TimescaleDB hypertable ingesting deforestation alerts from external
-- sources (GLAD, RADD, FIRMS, GFW) and internal detection. Each alert
-- captures geographic location, affected area, confidence level, severity,
-- and whether the alert is post-EUDR cutoff date. Partitioned by
-- detection_date for time-series queries. Retained for 730 days with
-- compression after 30 days. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.deforestation_alerts (
    id UUID DEFAULT gen_random_uuid(),
    alert_id VARCHAR(100) NOT NULL,
    source VARCHAR(20) NOT NULL,
    detection_date TIMESTAMPTZ NOT NULL,
    latitude DECIMAL(12,8) NOT NULL,
    longitude DECIMAL(12,8) NOT NULL,
    area_ha DECIMAL(12,4),
    confidence VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    alert_type VARCHAR(50) NOT NULL DEFAULT 'deforestation',
    is_post_cutoff BOOLEAN NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, detection_date)
);

-- Create hypertable partitioned by detection_date
SELECT create_hypertable('deforestation_satellite_service.deforestation_alerts', 'detection_date', if_not_exists => TRUE);

-- Source constraint
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_source
    CHECK (source IN ('glad', 'radd', 'firms', 'gfw', 'internal'));

-- Confidence constraint
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_confidence
    CHECK (confidence IN ('low', 'nominal', 'high', 'highest'));

-- Severity constraint
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_severity
    CHECK (severity IN ('low', 'medium', 'high', 'critical'));

-- Alert ID must not be empty
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_alert_id_not_empty
    CHECK (LENGTH(TRIM(alert_id)) > 0);

-- Latitude must be between -90 and 90
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_latitude_range
    CHECK (latitude >= -90 AND latitude <= 90);

-- Longitude must be between -180 and 180
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_longitude_range
    CHECK (longitude >= -180 AND longitude <= 180);

-- Area must be non-negative if specified
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_area_non_negative
    CHECK (area_ha IS NULL OR area_ha >= 0);

-- Alert type constraint
ALTER TABLE deforestation_satellite_service.deforestation_alerts
    ADD CONSTRAINT chk_da_alert_type
    CHECK (alert_type IN ('deforestation', 'degradation', 'fire', 'logging', 'mining', 'agriculture_expansion'));

-- =============================================================================
-- Table 5: deforestation_satellite_service.change_detections
-- =============================================================================
-- Pre/post satellite image change detection analysis results. Each record
-- captures the change type, pre/post dates, NDVI and NBR deltas, affected
-- area, confidence score, pixel count, and optional polygon boundary.
-- Used for quantifying deforestation extent between two acquisition dates.
-- Tenant-scoped.

CREATE TABLE deforestation_satellite_service.change_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    change_id VARCHAR(100) UNIQUE NOT NULL,
    change_type VARCHAR(50) NOT NULL,
    pre_date DATE NOT NULL,
    post_date DATE NOT NULL,
    pre_ndvi DECIMAL(8,6),
    post_ndvi DECIMAL(8,6),
    delta_ndvi DECIMAL(8,6),
    delta_nbr DECIMAL(8,6),
    area_ha DECIMAL(12,4),
    confidence DECIMAL(5,4),
    pixel_count INTEGER,
    polygon_wkt TEXT,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

-- Change type constraint
ALTER TABLE deforestation_satellite_service.change_detections
    ADD CONSTRAINT chk_cd_change_type
    CHECK (change_type IN (
        'deforestation', 'degradation', 'reforestation', 'regrowth',
        'fire_damage', 'selective_logging', 'no_change', 'cloud_interference'
    ));

-- Change ID must not be empty
ALTER TABLE deforestation_satellite_service.change_detections
    ADD CONSTRAINT chk_cd_change_id_not_empty
    CHECK (LENGTH(TRIM(change_id)) > 0);

-- Post date must be after pre date
ALTER TABLE deforestation_satellite_service.change_detections
    ADD CONSTRAINT chk_cd_date_order
    CHECK (post_date >= pre_date);

-- Confidence must be between 0 and 1
ALTER TABLE deforestation_satellite_service.change_detections
    ADD CONSTRAINT chk_cd_confidence_range
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));

-- Pixel count must be non-negative if specified
ALTER TABLE deforestation_satellite_service.change_detections
    ADD CONSTRAINT chk_cd_pixel_count_non_negative
    CHECK (pixel_count IS NULL OR pixel_count >= 0);

-- Area must be non-negative if specified
ALTER TABLE deforestation_satellite_service.change_detections
    ADD CONSTRAINT chk_cd_area_non_negative
    CHECK (area_ha IS NULL OR area_ha >= 0);

-- =============================================================================
-- Table 6: deforestation_satellite_service.baseline_checks
-- =============================================================================
-- Baseline forest cover verification checks. Each record links to a
-- forest assessment and captures polygon boundary, sample point count,
-- aggregation method, overall compliance determination, worst-case
-- forest status, and maximum risk score. Used for multi-point baseline
-- verification across a geographic area. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.baseline_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id VARCHAR(100) NOT NULL,
    polygon_wkt TEXT,
    sample_points INTEGER,
    aggregation_method VARCHAR(50) DEFAULT 'conservative',
    overall_compliance BOOLEAN NOT NULL,
    worst_case_status VARCHAR(50),
    max_risk_score DECIMAL(5,2),
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to forest_assessments
ALTER TABLE deforestation_satellite_service.baseline_checks
    ADD CONSTRAINT fk_bc_assessment_id
    FOREIGN KEY (assessment_id) REFERENCES deforestation_satellite_service.forest_assessments(assessment_id)
    ON DELETE CASCADE;

-- Assessment ID must not be empty
ALTER TABLE deforestation_satellite_service.baseline_checks
    ADD CONSTRAINT chk_bc_assessment_id_not_empty
    CHECK (LENGTH(TRIM(assessment_id)) > 0);

-- Sample points must be positive if specified
ALTER TABLE deforestation_satellite_service.baseline_checks
    ADD CONSTRAINT chk_bc_sample_points_positive
    CHECK (sample_points IS NULL OR sample_points > 0);

-- Max risk score must be between 0 and 100
ALTER TABLE deforestation_satellite_service.baseline_checks
    ADD CONSTRAINT chk_bc_max_risk_score_range
    CHECK (max_risk_score IS NULL OR (max_risk_score >= 0 AND max_risk_score <= 100));

-- Aggregation method constraint
ALTER TABLE deforestation_satellite_service.baseline_checks
    ADD CONSTRAINT chk_bc_aggregation_method
    CHECK (aggregation_method IN ('conservative', 'average', 'majority_vote', 'worst_case', 'best_case'));

-- Worst case status constraint
ALTER TABLE deforestation_satellite_service.baseline_checks
    ADD CONSTRAINT chk_bc_worst_case_status
    CHECK (worst_case_status IS NULL OR worst_case_status IN (
        'intact', 'degraded', 'deforested', 'reforested',
        'non_forest', 'partially_deforested', 'unknown'
    ));

-- =============================================================================
-- Table 7: deforestation_satellite_service.compliance_reports
-- =============================================================================
-- EUDR compliance report generation records. Each report captures a
-- geographic polygon, country, compliance status, risk assessment,
-- area breakdown (total, forest, deforested), alert counts, and
-- recommendations. Used for producing due diligence evidence packages
-- and regulatory submissions. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.compliance_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id VARCHAR(100) UNIQUE NOT NULL,
    polygon_wkt TEXT NOT NULL,
    country_iso3 VARCHAR(3) NOT NULL,
    compliance_status VARCHAR(30) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    risk_score DECIMAL(5,2),
    total_area_ha DECIMAL(12,4),
    forest_area_ha DECIMAL(12,4),
    deforested_area_ha DECIMAL(12,4),
    total_alerts INTEGER DEFAULT 0,
    post_cutoff_alerts INTEGER DEFAULT 0,
    high_confidence_alerts INTEGER DEFAULT 0,
    affected_area_ha DECIMAL(12,4) DEFAULT 0,
    recommendations JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

-- Compliance status constraint
ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_compliance_status
    CHECK (compliance_status IN (
        'compliant', 'non_compliant', 'requires_review',
        'insufficient_data', 'pending', 'expired'
    ));

-- Risk level constraint
ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_risk_level
    CHECK (risk_level IN ('low', 'medium', 'high', 'critical', 'unknown'));

-- Report ID must not be empty
ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_report_id_not_empty
    CHECK (LENGTH(TRIM(report_id)) > 0);

-- Country ISO3 must be exactly 3 characters
ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_country_iso3_length
    CHECK (LENGTH(TRIM(country_iso3)) = 3);

-- Risk score must be between 0 and 100
ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_risk_score_range
    CHECK (risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100));

-- Area fields must be non-negative
ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_total_area_non_negative
    CHECK (total_area_ha IS NULL OR total_area_ha >= 0);

ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_forest_area_non_negative
    CHECK (forest_area_ha IS NULL OR forest_area_ha >= 0);

ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_deforested_area_non_negative
    CHECK (deforested_area_ha IS NULL OR deforested_area_ha >= 0);

ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_affected_area_non_negative
    CHECK (affected_area_ha IS NULL OR affected_area_ha >= 0);

-- Alert counts must be non-negative
ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_total_alerts_non_negative
    CHECK (total_alerts IS NULL OR total_alerts >= 0);

ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_post_cutoff_alerts_non_negative
    CHECK (post_cutoff_alerts IS NULL OR post_cutoff_alerts >= 0);

ALTER TABLE deforestation_satellite_service.compliance_reports
    ADD CONSTRAINT chk_cr_high_confidence_alerts_non_negative
    CHECK (high_confidence_alerts IS NULL OR high_confidence_alerts >= 0);

-- =============================================================================
-- Table 8: deforestation_satellite_service.monitoring_jobs (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording monitoring pipeline job orchestration.
-- Each job tracks a geographic polygon through multiple pipeline stages
-- (scene acquisition, index computation, change detection, alert check,
-- compliance reporting). Partitioned by started_at for time-series queries.
-- Retained for 730 days with compression after 30 days. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.monitoring_jobs (
    id UUID DEFAULT gen_random_uuid(),
    job_id VARCHAR(100) NOT NULL,
    polygon_wkt TEXT NOT NULL,
    country_iso3 VARCHAR(3) NOT NULL,
    frequency VARCHAR(20) NOT NULL DEFAULT 'monthly',
    current_stage VARCHAR(50),
    stages_completed JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_running BOOLEAN NOT NULL DEFAULT false,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    last_result JSONB,
    error_message TEXT,
    tenant_id UUID NOT NULL,
    PRIMARY KEY (id, started_at)
);

-- Create hypertable partitioned by started_at
SELECT create_hypertable('deforestation_satellite_service.monitoring_jobs', 'started_at', if_not_exists => TRUE);

-- Frequency constraint
ALTER TABLE deforestation_satellite_service.monitoring_jobs
    ADD CONSTRAINT chk_mj_frequency
    CHECK (frequency IN ('daily', 'weekly', 'biweekly', 'monthly', 'quarterly', 'on_demand'));

-- Job ID must not be empty
ALTER TABLE deforestation_satellite_service.monitoring_jobs
    ADD CONSTRAINT chk_mj_job_id_not_empty
    CHECK (LENGTH(TRIM(job_id)) > 0);

-- Country ISO3 must be exactly 3 characters
ALTER TABLE deforestation_satellite_service.monitoring_jobs
    ADD CONSTRAINT chk_mj_country_iso3_length
    CHECK (LENGTH(TRIM(country_iso3)) = 3);

-- Current stage constraint
ALTER TABLE deforestation_satellite_service.monitoring_jobs
    ADD CONSTRAINT chk_mj_current_stage
    CHECK (current_stage IS NULL OR current_stage IN (
        'scene_acquisition', 'cloud_filtering', 'index_computation',
        'change_detection', 'alert_check', 'classification',
        'baseline_verification', 'compliance_reporting', 'completed', 'failed'
    ));

-- Completed_at must be after started_at if specified
ALTER TABLE deforestation_satellite_service.monitoring_jobs
    ADD CONSTRAINT chk_mj_completed_after_started
    CHECK (completed_at IS NULL OR completed_at >= started_at);

-- =============================================================================
-- Table 9: deforestation_satellite_service.classification_results
-- =============================================================================
-- Land cover and forest classification results per satellite scene.
-- Each record captures the classification output including land cover
-- class, tree cover percentage, forest/non-forest determination,
-- canopy height, confidence score, and classification method. Used
-- for forest area quantification and change detection baselines.
-- Linked to satellite_scenes via scene_id. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.classification_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    classification_id VARCHAR(100) UNIQUE NOT NULL,
    scene_id VARCHAR(100),
    land_cover_class VARCHAR(50) NOT NULL,
    tree_cover_percent DECIMAL(5,2),
    is_forest BOOLEAN NOT NULL,
    canopy_height_m DECIMAL(6,2),
    confidence DECIMAL(5,4),
    method VARCHAR(50) DEFAULT 'decision_tree',
    pixel_count INTEGER,
    tenant_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64)
);

-- Foreign key to satellite_scenes (nullable for non-scene classifications)
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT fk_clr_scene_id
    FOREIGN KEY (scene_id) REFERENCES deforestation_satellite_service.satellite_scenes(scene_id)
    ON DELETE SET NULL;

-- Land cover class constraint
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT chk_clr_land_cover_class
    CHECK (land_cover_class IN (
        'dense_forest', 'open_forest', 'woodland', 'shrubland',
        'grassland', 'cropland', 'wetland', 'water',
        'urban', 'barren', 'mangrove', 'plantation',
        'agroforestry', 'degraded_forest', 'other'
    ));

-- Classification ID must not be empty
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT chk_clr_classification_id_not_empty
    CHECK (LENGTH(TRIM(classification_id)) > 0);

-- Tree cover percent must be between 0 and 100
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT chk_clr_tree_cover_range
    CHECK (tree_cover_percent IS NULL OR (tree_cover_percent >= 0 AND tree_cover_percent <= 100));

-- Confidence must be between 0 and 1
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT chk_clr_confidence_range
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));

-- Canopy height must be non-negative
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT chk_clr_canopy_height_non_negative
    CHECK (canopy_height_m IS NULL OR canopy_height_m >= 0);

-- Pixel count must be non-negative
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT chk_clr_pixel_count_non_negative
    CHECK (pixel_count IS NULL OR pixel_count >= 0);

-- Method constraint
ALTER TABLE deforestation_satellite_service.classification_results
    ADD CONSTRAINT chk_clr_method
    CHECK (method IS NULL OR method IN (
        'decision_tree', 'random_forest', 'gradient_boosting',
        'neural_network', 'threshold', 'rule_based', 'ensemble'
    ));

-- =============================================================================
-- Table 10: deforestation_satellite_service.pipeline_metrics (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording per-stage pipeline performance metrics.
-- Each metric captures the pipeline and job identifiers, processing stage,
-- execution status, result summary, and duration. Partitioned by created_at
-- for time-series queries. Retained for 730 days with compression after
-- 30 days. Tenant-scoped.

CREATE TABLE deforestation_satellite_service.pipeline_metrics (
    id UUID DEFAULT gen_random_uuid(),
    pipeline_id VARCHAR(100) NOT NULL,
    job_id VARCHAR(100) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    result_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    duration_seconds DECIMAL(10,3),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    PRIMARY KEY (id, created_at)
);

-- Create hypertable partitioned by created_at
SELECT create_hypertable('deforestation_satellite_service.pipeline_metrics', 'created_at', if_not_exists => TRUE);

-- Status constraint
ALTER TABLE deforestation_satellite_service.pipeline_metrics
    ADD CONSTRAINT chk_pm_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'timeout', 'cancelled'));

-- Stage constraint
ALTER TABLE deforestation_satellite_service.pipeline_metrics
    ADD CONSTRAINT chk_pm_stage
    CHECK (stage IN (
        'scene_acquisition', 'cloud_filtering', 'index_computation',
        'change_detection', 'alert_check', 'classification',
        'baseline_verification', 'compliance_reporting', 'full_pipeline'
    ));

-- Pipeline ID must not be empty
ALTER TABLE deforestation_satellite_service.pipeline_metrics
    ADD CONSTRAINT chk_pm_pipeline_id_not_empty
    CHECK (LENGTH(TRIM(pipeline_id)) > 0);

-- Job ID must not be empty
ALTER TABLE deforestation_satellite_service.pipeline_metrics
    ADD CONSTRAINT chk_pm_job_id_not_empty
    CHECK (LENGTH(TRIM(job_id)) > 0);

-- Duration must be non-negative if specified
ALTER TABLE deforestation_satellite_service.pipeline_metrics
    ADD CONSTRAINT chk_pm_duration_non_negative
    CHECK (duration_seconds IS NULL OR duration_seconds >= 0);

-- =============================================================================
-- Continuous Aggregate: deforestation_satellite_service.hourly_alert_stats
-- =============================================================================
-- Precomputed hourly deforestation alert statistics by source and severity
-- for dashboard queries, trend analysis, and SLI tracking.

CREATE MATERIALIZED VIEW deforestation_satellite_service.hourly_alert_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', detection_date) AS bucket,
    source,
    severity,
    tenant_id,
    COUNT(*) AS total_alerts,
    SUM(area_ha) AS total_area_ha,
    COUNT(*) FILTER (WHERE confidence = 'high' OR confidence = 'highest') AS high_confidence_count,
    COUNT(*) FILTER (WHERE confidence = 'low' OR confidence = 'nominal') AS low_confidence_count,
    COUNT(*) FILTER (WHERE is_post_cutoff = TRUE) AS post_cutoff_count,
    COUNT(*) FILTER (WHERE is_post_cutoff = FALSE) AS pre_cutoff_count,
    COUNT(DISTINCT alert_type) AS unique_alert_types
FROM deforestation_satellite_service.deforestation_alerts
WHERE detection_date IS NOT NULL
GROUP BY bucket, source, severity, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('deforestation_satellite_service.hourly_alert_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: deforestation_satellite_service.hourly_pipeline_stats
-- =============================================================================
-- Precomputed hourly pipeline metrics statistics by stage for monitoring
-- pipeline throughput, failure rates, and average duration.

CREATE MATERIALIZED VIEW deforestation_satellite_service.hourly_pipeline_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    stage,
    tenant_id,
    COUNT(*) AS total_executions,
    AVG(duration_seconds) AS avg_duration_seconds,
    MAX(duration_seconds) AS max_duration_seconds,
    MIN(duration_seconds) AS min_duration_seconds,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
    COUNT(*) FILTER (WHERE status = 'timeout') AS timeout_count,
    COUNT(DISTINCT pipeline_id) AS unique_pipelines
FROM deforestation_satellite_service.pipeline_metrics
WHERE created_at IS NOT NULL
GROUP BY bucket, stage, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('deforestation_satellite_service.hourly_pipeline_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- satellite_scenes indexes
CREATE INDEX idx_ss_scene_id ON deforestation_satellite_service.satellite_scenes(scene_id);
CREATE INDEX idx_ss_satellite ON deforestation_satellite_service.satellite_scenes(satellite);
CREATE INDEX idx_ss_acquisition_date ON deforestation_satellite_service.satellite_scenes(acquisition_date DESC);
CREATE INDEX idx_ss_cloud_cover ON deforestation_satellite_service.satellite_scenes(cloud_cover_percent);
CREATE INDEX idx_ss_resolution ON deforestation_satellite_service.satellite_scenes(resolution_m);
CREATE INDEX idx_ss_crs ON deforestation_satellite_service.satellite_scenes(crs);
CREATE INDEX idx_ss_tile_id ON deforestation_satellite_service.satellite_scenes(tile_id);
CREATE INDEX idx_ss_tenant ON deforestation_satellite_service.satellite_scenes(tenant_id);
CREATE INDEX idx_ss_created_at ON deforestation_satellite_service.satellite_scenes(created_at DESC);
CREATE INDEX idx_ss_provenance ON deforestation_satellite_service.satellite_scenes(provenance_hash);
CREATE INDEX idx_ss_tenant_satellite ON deforestation_satellite_service.satellite_scenes(tenant_id, satellite);
CREATE INDEX idx_ss_tenant_tile ON deforestation_satellite_service.satellite_scenes(tenant_id, tile_id);
CREATE INDEX idx_ss_tenant_date ON deforestation_satellite_service.satellite_scenes(tenant_id, acquisition_date DESC);
CREATE INDEX idx_ss_bbox ON deforestation_satellite_service.satellite_scenes(bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat);
CREATE INDEX idx_ss_bands ON deforestation_satellite_service.satellite_scenes USING GIN (bands);
CREATE INDEX idx_ss_metadata ON deforestation_satellite_service.satellite_scenes USING GIN (metadata);

-- vegetation_indices indexes
CREATE INDEX idx_vi_scene_id ON deforestation_satellite_service.vegetation_indices(scene_id);
CREATE INDEX idx_vi_index_type ON deforestation_satellite_service.vegetation_indices(index_type);
CREATE INDEX idx_vi_computation_date ON deforestation_satellite_service.vegetation_indices(computation_date DESC);
CREATE INDEX idx_vi_mean_value ON deforestation_satellite_service.vegetation_indices(mean_value);
CREATE INDEX idx_vi_tenant ON deforestation_satellite_service.vegetation_indices(tenant_id);
CREATE INDEX idx_vi_created_at ON deforestation_satellite_service.vegetation_indices(created_at DESC);
CREATE INDEX idx_vi_tenant_scene ON deforestation_satellite_service.vegetation_indices(tenant_id, scene_id);
CREATE INDEX idx_vi_tenant_type ON deforestation_satellite_service.vegetation_indices(tenant_id, index_type);
CREATE INDEX idx_vi_scene_type ON deforestation_satellite_service.vegetation_indices(scene_id, index_type);

-- forest_assessments indexes
CREATE INDEX idx_fa_assessment_id ON deforestation_satellite_service.forest_assessments(assessment_id);
CREATE INDEX idx_fa_lat_lon ON deforestation_satellite_service.forest_assessments(latitude, longitude);
CREATE INDEX idx_fa_country_iso3 ON deforestation_satellite_service.forest_assessments(country_iso3);
CREATE INDEX idx_fa_forest_status ON deforestation_satellite_service.forest_assessments(forest_status);
CREATE INDEX idx_fa_is_eudr_compliant ON deforestation_satellite_service.forest_assessments(is_eudr_compliant);
CREATE INDEX idx_fa_risk_level ON deforestation_satellite_service.forest_assessments(risk_level);
CREATE INDEX idx_fa_risk_score ON deforestation_satellite_service.forest_assessments(risk_score DESC);
CREATE INDEX idx_fa_assessment_date ON deforestation_satellite_service.forest_assessments(assessment_date DESC);
CREATE INDEX idx_fa_baseline_date ON deforestation_satellite_service.forest_assessments(baseline_date);
CREATE INDEX idx_fa_tenant ON deforestation_satellite_service.forest_assessments(tenant_id);
CREATE INDEX idx_fa_created_at ON deforestation_satellite_service.forest_assessments(created_at DESC);
CREATE INDEX idx_fa_provenance ON deforestation_satellite_service.forest_assessments(provenance_hash);
CREATE INDEX idx_fa_tenant_country ON deforestation_satellite_service.forest_assessments(tenant_id, country_iso3);
CREATE INDEX idx_fa_tenant_risk ON deforestation_satellite_service.forest_assessments(tenant_id, risk_level);
CREATE INDEX idx_fa_tenant_compliance ON deforestation_satellite_service.forest_assessments(tenant_id, is_eudr_compliant);
CREATE INDEX idx_fa_tenant_status ON deforestation_satellite_service.forest_assessments(tenant_id, forest_status);
CREATE INDEX idx_fa_data_sources ON deforestation_satellite_service.forest_assessments USING GIN (data_sources);
CREATE INDEX idx_fa_warnings ON deforestation_satellite_service.forest_assessments USING GIN (warnings);
CREATE INDEX idx_fa_forest_definition ON deforestation_satellite_service.forest_assessments USING GIN (forest_definition);

-- deforestation_alerts indexes (hypertable-aware)
CREATE INDEX idx_da_alert_id ON deforestation_satellite_service.deforestation_alerts(alert_id, detection_date DESC);
CREATE INDEX idx_da_source ON deforestation_satellite_service.deforestation_alerts(source, detection_date DESC);
CREATE INDEX idx_da_lat_lon ON deforestation_satellite_service.deforestation_alerts(latitude, longitude, detection_date DESC);
CREATE INDEX idx_da_confidence ON deforestation_satellite_service.deforestation_alerts(confidence, detection_date DESC);
CREATE INDEX idx_da_severity ON deforestation_satellite_service.deforestation_alerts(severity, detection_date DESC);
CREATE INDEX idx_da_alert_type ON deforestation_satellite_service.deforestation_alerts(alert_type, detection_date DESC);
CREATE INDEX idx_da_is_post_cutoff ON deforestation_satellite_service.deforestation_alerts(is_post_cutoff, detection_date DESC);
CREATE INDEX idx_da_tenant ON deforestation_satellite_service.deforestation_alerts(tenant_id, detection_date DESC);
CREATE INDEX idx_da_tenant_source ON deforestation_satellite_service.deforestation_alerts(tenant_id, source, detection_date DESC);
CREATE INDEX idx_da_tenant_severity ON deforestation_satellite_service.deforestation_alerts(tenant_id, severity, detection_date DESC);
CREATE INDEX idx_da_tenant_cutoff ON deforestation_satellite_service.deforestation_alerts(tenant_id, is_post_cutoff, detection_date DESC);
CREATE INDEX idx_da_metadata ON deforestation_satellite_service.deforestation_alerts USING GIN (metadata);

-- change_detections indexes
CREATE INDEX idx_cd_change_id ON deforestation_satellite_service.change_detections(change_id);
CREATE INDEX idx_cd_change_type ON deforestation_satellite_service.change_detections(change_type);
CREATE INDEX idx_cd_pre_date ON deforestation_satellite_service.change_detections(pre_date DESC);
CREATE INDEX idx_cd_post_date ON deforestation_satellite_service.change_detections(post_date DESC);
CREATE INDEX idx_cd_area_ha ON deforestation_satellite_service.change_detections(area_ha DESC);
CREATE INDEX idx_cd_confidence ON deforestation_satellite_service.change_detections(confidence DESC);
CREATE INDEX idx_cd_delta_ndvi ON deforestation_satellite_service.change_detections(delta_ndvi);
CREATE INDEX idx_cd_tenant ON deforestation_satellite_service.change_detections(tenant_id);
CREATE INDEX idx_cd_created_at ON deforestation_satellite_service.change_detections(created_at DESC);
CREATE INDEX idx_cd_provenance ON deforestation_satellite_service.change_detections(provenance_hash);
CREATE INDEX idx_cd_tenant_type ON deforestation_satellite_service.change_detections(tenant_id, change_type);
CREATE INDEX idx_cd_tenant_dates ON deforestation_satellite_service.change_detections(tenant_id, pre_date, post_date);

-- baseline_checks indexes
CREATE INDEX idx_bc_assessment_id ON deforestation_satellite_service.baseline_checks(assessment_id);
CREATE INDEX idx_bc_overall_compliance ON deforestation_satellite_service.baseline_checks(overall_compliance);
CREATE INDEX idx_bc_worst_case_status ON deforestation_satellite_service.baseline_checks(worst_case_status);
CREATE INDEX idx_bc_max_risk_score ON deforestation_satellite_service.baseline_checks(max_risk_score DESC);
CREATE INDEX idx_bc_aggregation_method ON deforestation_satellite_service.baseline_checks(aggregation_method);
CREATE INDEX idx_bc_tenant ON deforestation_satellite_service.baseline_checks(tenant_id);
CREATE INDEX idx_bc_created_at ON deforestation_satellite_service.baseline_checks(created_at DESC);
CREATE INDEX idx_bc_tenant_compliance ON deforestation_satellite_service.baseline_checks(tenant_id, overall_compliance);
CREATE INDEX idx_bc_tenant_assessment ON deforestation_satellite_service.baseline_checks(tenant_id, assessment_id);

-- compliance_reports indexes
CREATE INDEX idx_cr_report_id ON deforestation_satellite_service.compliance_reports(report_id);
CREATE INDEX idx_cr_country_iso3 ON deforestation_satellite_service.compliance_reports(country_iso3);
CREATE INDEX idx_cr_compliance_status ON deforestation_satellite_service.compliance_reports(compliance_status);
CREATE INDEX idx_cr_risk_level ON deforestation_satellite_service.compliance_reports(risk_level);
CREATE INDEX idx_cr_risk_score ON deforestation_satellite_service.compliance_reports(risk_score DESC);
CREATE INDEX idx_cr_total_area ON deforestation_satellite_service.compliance_reports(total_area_ha DESC);
CREATE INDEX idx_cr_deforested_area ON deforestation_satellite_service.compliance_reports(deforested_area_ha DESC);
CREATE INDEX idx_cr_total_alerts ON deforestation_satellite_service.compliance_reports(total_alerts DESC);
CREATE INDEX idx_cr_post_cutoff_alerts ON deforestation_satellite_service.compliance_reports(post_cutoff_alerts DESC);
CREATE INDEX idx_cr_tenant ON deforestation_satellite_service.compliance_reports(tenant_id);
CREATE INDEX idx_cr_created_at ON deforestation_satellite_service.compliance_reports(created_at DESC);
CREATE INDEX idx_cr_provenance ON deforestation_satellite_service.compliance_reports(provenance_hash);
CREATE INDEX idx_cr_tenant_country ON deforestation_satellite_service.compliance_reports(tenant_id, country_iso3);
CREATE INDEX idx_cr_tenant_status ON deforestation_satellite_service.compliance_reports(tenant_id, compliance_status);
CREATE INDEX idx_cr_tenant_risk ON deforestation_satellite_service.compliance_reports(tenant_id, risk_level);
CREATE INDEX idx_cr_recommendations ON deforestation_satellite_service.compliance_reports USING GIN (recommendations);
CREATE INDEX idx_cr_evidence_summary ON deforestation_satellite_service.compliance_reports USING GIN (evidence_summary);

-- monitoring_jobs indexes (hypertable-aware)
CREATE INDEX idx_mj_job_id ON deforestation_satellite_service.monitoring_jobs(job_id, started_at DESC);
CREATE INDEX idx_mj_country_iso3 ON deforestation_satellite_service.monitoring_jobs(country_iso3, started_at DESC);
CREATE INDEX idx_mj_frequency ON deforestation_satellite_service.monitoring_jobs(frequency, started_at DESC);
CREATE INDEX idx_mj_current_stage ON deforestation_satellite_service.monitoring_jobs(current_stage, started_at DESC);
CREATE INDEX idx_mj_is_running ON deforestation_satellite_service.monitoring_jobs(is_running, started_at DESC);
CREATE INDEX idx_mj_tenant ON deforestation_satellite_service.monitoring_jobs(tenant_id, started_at DESC);
CREATE INDEX idx_mj_tenant_running ON deforestation_satellite_service.monitoring_jobs(tenant_id, is_running, started_at DESC);
CREATE INDEX idx_mj_tenant_country ON deforestation_satellite_service.monitoring_jobs(tenant_id, country_iso3, started_at DESC);
CREATE INDEX idx_mj_tenant_stage ON deforestation_satellite_service.monitoring_jobs(tenant_id, current_stage, started_at DESC);
CREATE INDEX idx_mj_stages_completed ON deforestation_satellite_service.monitoring_jobs USING GIN (stages_completed);
CREATE INDEX idx_mj_last_result ON deforestation_satellite_service.monitoring_jobs USING GIN (last_result);

-- classification_results indexes
CREATE INDEX idx_clr_classification_id ON deforestation_satellite_service.classification_results(classification_id);
CREATE INDEX idx_clr_scene_id ON deforestation_satellite_service.classification_results(scene_id);
CREATE INDEX idx_clr_land_cover_class ON deforestation_satellite_service.classification_results(land_cover_class);
CREATE INDEX idx_clr_is_forest ON deforestation_satellite_service.classification_results(is_forest);
CREATE INDEX idx_clr_tree_cover ON deforestation_satellite_service.classification_results(tree_cover_percent DESC);
CREATE INDEX idx_clr_confidence ON deforestation_satellite_service.classification_results(confidence DESC);
CREATE INDEX idx_clr_method ON deforestation_satellite_service.classification_results(method);
CREATE INDEX idx_clr_tenant ON deforestation_satellite_service.classification_results(tenant_id);
CREATE INDEX idx_clr_created_at ON deforestation_satellite_service.classification_results(created_at DESC);
CREATE INDEX idx_clr_provenance ON deforestation_satellite_service.classification_results(provenance_hash);
CREATE INDEX idx_clr_tenant_class ON deforestation_satellite_service.classification_results(tenant_id, land_cover_class);
CREATE INDEX idx_clr_tenant_forest ON deforestation_satellite_service.classification_results(tenant_id, is_forest);
CREATE INDEX idx_clr_tenant_scene ON deforestation_satellite_service.classification_results(tenant_id, scene_id);

-- pipeline_metrics indexes (hypertable-aware)
CREATE INDEX idx_pm_pipeline_id ON deforestation_satellite_service.pipeline_metrics(pipeline_id, created_at DESC);
CREATE INDEX idx_pm_job_id ON deforestation_satellite_service.pipeline_metrics(job_id, created_at DESC);
CREATE INDEX idx_pm_stage ON deforestation_satellite_service.pipeline_metrics(stage, created_at DESC);
CREATE INDEX idx_pm_status ON deforestation_satellite_service.pipeline_metrics(status, created_at DESC);
CREATE INDEX idx_pm_duration ON deforestation_satellite_service.pipeline_metrics(duration_seconds DESC, created_at DESC);
CREATE INDEX idx_pm_tenant ON deforestation_satellite_service.pipeline_metrics(tenant_id, created_at DESC);
CREATE INDEX idx_pm_tenant_stage ON deforestation_satellite_service.pipeline_metrics(tenant_id, stage, created_at DESC);
CREATE INDEX idx_pm_tenant_status ON deforestation_satellite_service.pipeline_metrics(tenant_id, status, created_at DESC);
CREATE INDEX idx_pm_tenant_pipeline ON deforestation_satellite_service.pipeline_metrics(tenant_id, pipeline_id, created_at DESC);
CREATE INDEX idx_pm_result_summary ON deforestation_satellite_service.pipeline_metrics USING GIN (result_summary);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE deforestation_satellite_service.satellite_scenes ENABLE ROW LEVEL SECURITY;
CREATE POLICY ss_tenant_read ON deforestation_satellite_service.satellite_scenes
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ss_tenant_write ON deforestation_satellite_service.satellite_scenes
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.vegetation_indices ENABLE ROW LEVEL SECURITY;
CREATE POLICY vi_tenant_read ON deforestation_satellite_service.vegetation_indices
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY vi_tenant_write ON deforestation_satellite_service.vegetation_indices
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.forest_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY fa_tenant_read ON deforestation_satellite_service.forest_assessments
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY fa_tenant_write ON deforestation_satellite_service.forest_assessments
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.deforestation_alerts ENABLE ROW LEVEL SECURITY;
CREATE POLICY da_tenant_read ON deforestation_satellite_service.deforestation_alerts
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY da_tenant_write ON deforestation_satellite_service.deforestation_alerts
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.change_detections ENABLE ROW LEVEL SECURITY;
CREATE POLICY cd_tenant_read ON deforestation_satellite_service.change_detections
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cd_tenant_write ON deforestation_satellite_service.change_detections
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.baseline_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY bc_tenant_read ON deforestation_satellite_service.baseline_checks
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY bc_tenant_write ON deforestation_satellite_service.baseline_checks
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.compliance_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY cr_tenant_read ON deforestation_satellite_service.compliance_reports
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cr_tenant_write ON deforestation_satellite_service.compliance_reports
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.monitoring_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY mj_tenant_read ON deforestation_satellite_service.monitoring_jobs
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY mj_tenant_write ON deforestation_satellite_service.monitoring_jobs
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.classification_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY clr_tenant_read ON deforestation_satellite_service.classification_results
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY clr_tenant_write ON deforestation_satellite_service.classification_results
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE deforestation_satellite_service.pipeline_metrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY pm_tenant_read ON deforestation_satellite_service.pipeline_metrics
    FOR SELECT USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY pm_tenant_write ON deforestation_satellite_service.pipeline_metrics
    FOR ALL USING (
        tenant_id::text = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA deforestation_satellite_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA deforestation_satellite_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA deforestation_satellite_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON deforestation_satellite_service.hourly_alert_stats TO greenlang_app;
GRANT SELECT ON deforestation_satellite_service.hourly_pipeline_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA deforestation_satellite_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA deforestation_satellite_service TO greenlang_readonly;
GRANT SELECT ON deforestation_satellite_service.hourly_alert_stats TO greenlang_readonly;
GRANT SELECT ON deforestation_satellite_service.hourly_pipeline_stats TO greenlang_readonly;

-- Add deforestation satellite service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'deforestation_satellite:scenes:read', 'deforestation_satellite', 'scenes_read', 'View satellite scene metadata and acquisition data'),
    (gen_random_uuid(), 'deforestation_satellite:scenes:write', 'deforestation_satellite', 'scenes_write', 'Register and manage satellite scene records'),
    (gen_random_uuid(), 'deforestation_satellite:indices:read', 'deforestation_satellite', 'indices_read', 'View vegetation index computation results'),
    (gen_random_uuid(), 'deforestation_satellite:indices:write', 'deforestation_satellite', 'indices_write', 'Compute and store vegetation indices'),
    (gen_random_uuid(), 'deforestation_satellite:assessments:read', 'deforestation_satellite', 'assessments_read', 'View forest assessments and EUDR compliance status'),
    (gen_random_uuid(), 'deforestation_satellite:assessments:write', 'deforestation_satellite', 'assessments_write', 'Create and manage forest assessments'),
    (gen_random_uuid(), 'deforestation_satellite:alerts:read', 'deforestation_satellite', 'alerts_read', 'View deforestation alerts from GLAD/RADD/FIRMS/GFW'),
    (gen_random_uuid(), 'deforestation_satellite:alerts:write', 'deforestation_satellite', 'alerts_write', 'Ingest and manage deforestation alerts'),
    (gen_random_uuid(), 'deforestation_satellite:changes:read', 'deforestation_satellite', 'changes_read', 'View change detection analysis results'),
    (gen_random_uuid(), 'deforestation_satellite:changes:write', 'deforestation_satellite', 'changes_write', 'Execute and store change detection analyses'),
    (gen_random_uuid(), 'deforestation_satellite:baselines:read', 'deforestation_satellite', 'baselines_read', 'View baseline forest cover verification checks'),
    (gen_random_uuid(), 'deforestation_satellite:baselines:write', 'deforestation_satellite', 'baselines_write', 'Create and manage baseline verification checks'),
    (gen_random_uuid(), 'deforestation_satellite:compliance:read', 'deforestation_satellite', 'compliance_read', 'View compliance reports and evidence summaries'),
    (gen_random_uuid(), 'deforestation_satellite:compliance:write', 'deforestation_satellite', 'compliance_write', 'Generate and manage compliance reports'),
    (gen_random_uuid(), 'deforestation_satellite:jobs:read', 'deforestation_satellite', 'jobs_read', 'View monitoring job status and results'),
    (gen_random_uuid(), 'deforestation_satellite:jobs:write', 'deforestation_satellite', 'jobs_write', 'Create and manage monitoring pipeline jobs'),
    (gen_random_uuid(), 'deforestation_satellite:classification:read', 'deforestation_satellite', 'classification_read', 'View land cover classification results'),
    (gen_random_uuid(), 'deforestation_satellite:classification:write', 'deforestation_satellite', 'classification_write', 'Execute and store land cover classifications'),
    (gen_random_uuid(), 'deforestation_satellite:metrics:read', 'deforestation_satellite', 'metrics_read', 'View pipeline performance metrics and statistics'),
    (gen_random_uuid(), 'deforestation_satellite:admin', 'deforestation_satellite', 'admin', 'Deforestation satellite service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep deforestation alerts records for 730 days (2 years)
SELECT add_retention_policy('deforestation_satellite_service.deforestation_alerts', INTERVAL '730 days');

-- Keep monitoring job records for 730 days (2 years)
SELECT add_retention_policy('deforestation_satellite_service.monitoring_jobs', INTERVAL '730 days');

-- Keep pipeline metrics records for 730 days (2 years)
SELECT add_retention_policy('deforestation_satellite_service.pipeline_metrics', INTERVAL '730 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on deforestation_alerts after 30 days
ALTER TABLE deforestation_satellite_service.deforestation_alerts SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'detection_date DESC'
);

SELECT add_compression_policy('deforestation_satellite_service.deforestation_alerts', INTERVAL '30 days');

-- Enable compression on monitoring_jobs after 30 days
ALTER TABLE deforestation_satellite_service.monitoring_jobs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'started_at DESC'
);

SELECT add_compression_policy('deforestation_satellite_service.monitoring_jobs', INTERVAL '30 days');

-- Enable compression on pipeline_metrics after 30 days
ALTER TABLE deforestation_satellite_service.pipeline_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('deforestation_satellite_service.pipeline_metrics', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-GEO-003', 'Deforestation Satellite Connector',
 'Satellite-based deforestation monitoring and EUDR compliance assessment for GreenLang Climate OS. Manages satellite scene ingestion from Sentinel-2, Landsat 8/9, MODIS, and Harmonized Landsat-Sentinel (HLS). Computes vegetation indices (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI), performs multi-temporal change detection with pre/post analysis, ingests deforestation alerts from GLAD, RADD, FIRMS, and Global Forest Watch, conducts forest status assessments against the EUDR baseline date (31 Dec 2020), verifies baseline forest cover, classifies land cover types, generates EUDR compliance reports with evidence packages, and orchestrates end-to-end monitoring pipelines with per-stage metrics tracking.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/deforestation-satellite', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Deforestation Satellite Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-GEO-003', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/deforestation-satellite-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"satellite", "deforestation", "eudr", "ndvi", "change-detection", "forest-monitoring", "compliance", "sentinel2", "landsat"}',
 '{"agriculture", "forestry", "cross-sector", "environmental"}',
 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Deforestation Satellite Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-GEO-003', '1.0.0', 'satellite_scene_management', 'data_management',
 'Ingest and manage satellite scene metadata from Sentinel-2, Landsat 8/9, MODIS, and Harmonized Landsat-Sentinel with bounding box, cloud cover filtering, spectral band inventory, and tile identification',
 '{"satellite", "bbox", "date_range", "max_cloud_cover"}', '{"scene_id", "acquisition_date", "bands", "cloud_cover_percent"}',
 '{"supported_satellites": ["sentinel2", "landsat8", "landsat9", "modis", "harmonized"], "default_resolution_m": 10, "default_crs": "EPSG:4326", "max_cloud_cover_default": 20}'::jsonb),

('GL-DATA-GEO-003', '1.0.0', 'vegetation_index_computation', 'analysis',
 'Compute vegetation indices (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI) from satellite scenes with min/max/mean/std statistics and pixel-level coverage tracking',
 '{"scene_id", "index_types"}', '{"indices", "statistics", "pixel_count"}',
 '{"supported_indices": ["ndvi", "evi", "ndwi", "nbr", "savi", "msavi", "ndmi"], "band_requirements": {"ndvi": ["red", "nir"], "evi": ["red", "nir", "blue"], "ndwi": ["green", "nir"], "nbr": ["nir", "swir2"]}}'::jsonb),

('GL-DATA-GEO-003', '1.0.0', 'forest_assessment', 'analysis',
 'Assess forest status and EUDR compliance for geographic coordinates against the baseline date (31 Dec 2020) with risk scoring, forest cover change analysis, and multi-source data fusion',
 '{"latitude", "longitude", "country_iso3"}', '{"assessment_id", "is_eudr_compliant", "risk_level", "risk_score", "forest_status"}',
 '{"baseline_date": "2020-12-31", "risk_levels": ["low", "medium", "high", "critical"], "forest_statuses": ["intact", "degraded", "deforested", "reforested", "non_forest"]}'::jsonb),

('GL-DATA-GEO-003', '1.0.0', 'deforestation_alert_ingestion', 'data_management',
 'Ingest deforestation alerts from GLAD, RADD, FIRMS, Global Forest Watch, and internal detection with geographic location, affected area, confidence scoring, and EUDR cutoff date classification',
 '{"source", "bbox", "date_range"}', '{"alerts", "total_count", "post_cutoff_count"}',
 '{"supported_sources": ["glad", "radd", "firms", "gfw", "internal"], "confidence_levels": ["low", "nominal", "high", "highest"], "eudr_cutoff_date": "2020-12-31"}'::jsonb),

('GL-DATA-GEO-003', '1.0.0', 'change_detection', 'analysis',
 'Perform multi-temporal change detection between pre and post satellite scenes using NDVI and NBR differencing with confidence scoring, area estimation, and polygon boundary extraction',
 '{"pre_scene_id", "post_scene_id", "threshold"}', '{"change_id", "change_type", "delta_ndvi", "area_ha", "confidence"}',
 '{"change_types": ["deforestation", "degradation", "reforestation", "regrowth", "fire_damage", "selective_logging"], "default_ndvi_threshold": -0.15, "default_nbr_threshold": -0.20}'::jsonb),

('GL-DATA-GEO-003', '1.0.0', 'baseline_verification', 'verification',
 'Verify baseline forest cover across a geographic polygon using multi-point sampling with conservative, average, majority vote, and worst-case aggregation methods',
 '{"polygon_wkt", "assessment_id", "sample_points"}', '{"overall_compliance", "worst_case_status", "max_risk_score"}',
 '{"aggregation_methods": ["conservative", "average", "majority_vote", "worst_case", "best_case"], "default_sample_points": 100, "default_method": "conservative"}'::jsonb),

('GL-DATA-GEO-003', '1.0.0', 'compliance_reporting', 'reporting',
 'Generate EUDR compliance reports with area breakdown, alert counts, risk assessment, recommendations, and evidence summary for due diligence submissions',
 '{"polygon_wkt", "country_iso3"}', '{"report_id", "compliance_status", "risk_level", "recommendations"}',
 '{"compliance_statuses": ["compliant", "non_compliant", "requires_review", "insufficient_data"], "include_evidence": true, "include_recommendations": true}'::jsonb),

('GL-DATA-GEO-003', '1.0.0', 'land_cover_classification', 'analysis',
 'Classify land cover types from satellite scenes with tree cover percentage, forest/non-forest determination, canopy height estimation, and confidence scoring using decision tree, random forest, or ensemble methods',
 '{"scene_id", "method"}', '{"classification_id", "land_cover_class", "is_forest", "tree_cover_percent", "confidence"}',
 '{"supported_methods": ["decision_tree", "random_forest", "gradient_boosting", "neural_network", "threshold", "rule_based", "ensemble"], "forest_threshold_percent": 10, "canopy_height_enabled": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Deforestation Satellite Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Deforestation Satellite depends on Schema Compiler for input/output validation
('GL-DATA-GEO-003', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Satellite scene metadata, vegetation indices, and assessment results are validated against JSON Schema definitions'),

-- Deforestation Satellite depends on Registry for agent discovery
('GL-DATA-GEO-003', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for deforestation monitoring pipeline orchestration'),

-- Deforestation Satellite depends on Access Guard for policy enforcement
('GL-DATA-GEO-003', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for satellite imagery and compliance assessment data'),

-- Deforestation Satellite depends on Observability Agent for metrics
('GL-DATA-GEO-003', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Pipeline metrics, scene processing statistics, and alert ingestion telemetry are reported to observability'),

-- Deforestation Satellite depends on GIS Connector for geospatial operations
('GL-DATA-GEO-003', 'GL-DATA-GEO-001', '>=1.0.0', false,
 'Geospatial operations (polygon intersection, area calculation, CRS transformation) for change detection and compliance assessment'),

-- Deforestation Satellite optionally integrates with EUDR Traceability
('GL-DATA-GEO-003', 'GL-DATA-EUDR-001', '>=1.0.0', true,
 'Forest assessments and compliance reports feed into EUDR traceability chain for due diligence statement support'),

-- Deforestation Satellite optionally uses Citations for provenance tracking
('GL-DATA-GEO-003', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Satellite scene provenance, assessment results, and compliance report evidence are registered with the citation service'),

-- Deforestation Satellite optionally uses Reproducibility for determinism
('GL-DATA-GEO-003', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Vegetation index computations and change detection results are verified for reproducibility across re-execution')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Deforestation Satellite Connector
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-GEO-003', 'Deforestation Satellite Connector',
 'Satellite-based deforestation monitoring and EUDR compliance assessment. Manages satellite scenes (Sentinel-2, Landsat 8/9, MODIS, HLS), computes vegetation indices (NDVI, EVI, NBR, etc.), ingests deforestation alerts (GLAD, RADD, FIRMS, GFW), performs multi-temporal change detection, assesses forest status against EUDR baseline (31 Dec 2020), verifies baseline forest cover, classifies land cover types, generates compliance reports with evidence packages, and orchestrates end-to-end monitoring pipelines.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA deforestation_satellite_service IS 'Deforestation Satellite Connector for GreenLang Climate OS (AGENT-DATA-007) - Satellite-based deforestation monitoring with scene management, vegetation indices, forest assessment, alert ingestion, change detection, baseline verification, compliance reporting, land cover classification, and pipeline orchestration';
COMMENT ON TABLE deforestation_satellite_service.satellite_scenes IS 'Satellite imagery scene metadata with platform (Sentinel-2/Landsat/MODIS/HLS), acquisition date, cloud cover, bounding box, spectral bands, resolution, CRS, and tile identification';
COMMENT ON TABLE deforestation_satellite_service.vegetation_indices IS 'Computed vegetation indices (NDVI/EVI/NDWI/NBR/SAVI/MSAVI/NDMI) per satellite scene with min/max/mean/std statistics and pixel coverage';
COMMENT ON TABLE deforestation_satellite_service.forest_assessments IS 'Forest status and EUDR compliance assessment records with geographic coordinates, forest cover percentages, change analysis, risk scoring, and compliance determination against baseline date (31 Dec 2020)';
COMMENT ON TABLE deforestation_satellite_service.deforestation_alerts IS 'TimescaleDB hypertable: deforestation alert ingestion from GLAD, RADD, FIRMS, GFW, and internal detection with geographic location, affected area, confidence, severity, and EUDR cutoff classification';
COMMENT ON TABLE deforestation_satellite_service.change_detections IS 'Pre/post satellite image change detection results with change type, NDVI/NBR deltas, affected area, confidence score, pixel count, and polygon boundary';
COMMENT ON TABLE deforestation_satellite_service.baseline_checks IS 'Baseline forest cover verification checks with polygon sampling, aggregation methods (conservative/average/worst-case), compliance determination, and risk scoring';
COMMENT ON TABLE deforestation_satellite_service.compliance_reports IS 'EUDR compliance reports with geographic polygon, area breakdown (total/forest/deforested), alert counts, risk assessment, recommendations, and evidence summaries for due diligence submissions';
COMMENT ON TABLE deforestation_satellite_service.monitoring_jobs IS 'TimescaleDB hypertable: monitoring pipeline job orchestration tracking geographic polygons through acquisition, filtering, index computation, change detection, alert check, classification, and compliance reporting stages';
COMMENT ON TABLE deforestation_satellite_service.classification_results IS 'Land cover and forest classification results per satellite scene with cover class (15 types), tree cover percentage, forest determination, canopy height, confidence, and classification method';
COMMENT ON TABLE deforestation_satellite_service.pipeline_metrics IS 'TimescaleDB hypertable: per-stage pipeline performance metrics with pipeline/job identifiers, stage, status, result summary, and duration for throughput and error rate monitoring';
COMMENT ON MATERIALIZED VIEW deforestation_satellite_service.hourly_alert_stats IS 'Continuous aggregate: hourly deforestation alert statistics by source and severity with count, total area, confidence breakdown, and post-cutoff classification for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW deforestation_satellite_service.hourly_pipeline_stats IS 'Continuous aggregate: hourly pipeline metrics statistics by stage with count, avg/max/min duration, completed/failed/timeout counts, and unique pipeline tracking for throughput monitoring';

COMMENT ON COLUMN deforestation_satellite_service.satellite_scenes.satellite IS 'Satellite platform: sentinel2, landsat8, landsat9, modis, harmonized (HLS)';
COMMENT ON COLUMN deforestation_satellite_service.satellite_scenes.cloud_cover_percent IS 'Scene cloud cover percentage (0-100) for quality filtering';
COMMENT ON COLUMN deforestation_satellite_service.satellite_scenes.bands IS 'JSONB spectral band inventory available in the scene (e.g., red, green, blue, nir, swir1, swir2)';
COMMENT ON COLUMN deforestation_satellite_service.satellite_scenes.resolution_m IS 'Spatial resolution in meters (e.g., 10m for Sentinel-2, 30m for Landsat)';
COMMENT ON COLUMN deforestation_satellite_service.satellite_scenes.provenance_hash IS 'SHA-256 provenance hash of scene metadata for integrity verification';
COMMENT ON COLUMN deforestation_satellite_service.vegetation_indices.index_type IS 'Vegetation index type: ndvi, evi, ndwi, nbr, savi, msavi, ndmi';
COMMENT ON COLUMN deforestation_satellite_service.vegetation_indices.mean_value IS 'Mean vegetation index value across all valid pixels in the scene';
COMMENT ON COLUMN deforestation_satellite_service.forest_assessments.is_eudr_compliant IS 'Whether the assessed location complies with EU Deforestation Regulation (no deforestation after 31 Dec 2020)';
COMMENT ON COLUMN deforestation_satellite_service.forest_assessments.baseline_date IS 'EUDR deforestation cutoff date per Article 2(13): default 31 December 2020';
COMMENT ON COLUMN deforestation_satellite_service.forest_assessments.risk_score IS 'Composite deforestation risk score (0-100) combining satellite evidence, alert history, and change detection';
COMMENT ON COLUMN deforestation_satellite_service.forest_assessments.forest_status IS 'Forest status: intact, degraded, deforested, reforested, non_forest, partially_deforested, unknown';
COMMENT ON COLUMN deforestation_satellite_service.deforestation_alerts.source IS 'Alert source: glad (Global Land Analysis & Discovery), radd (Radar for Detecting Deforestation), firms (Fire Information), gfw (Global Forest Watch), internal';
COMMENT ON COLUMN deforestation_satellite_service.deforestation_alerts.is_post_cutoff IS 'Whether the alert detection date is after the EUDR cutoff date (31 Dec 2020)';
COMMENT ON COLUMN deforestation_satellite_service.deforestation_alerts.confidence IS 'Alert confidence level: low, nominal, high, highest';
COMMENT ON COLUMN deforestation_satellite_service.deforestation_alerts.severity IS 'Alert severity level: low, medium, high, critical';
COMMENT ON COLUMN deforestation_satellite_service.change_detections.change_type IS 'Change type: deforestation, degradation, reforestation, regrowth, fire_damage, selective_logging, no_change, cloud_interference';
COMMENT ON COLUMN deforestation_satellite_service.change_detections.delta_ndvi IS 'NDVI change between pre and post scenes (negative indicates vegetation loss)';
COMMENT ON COLUMN deforestation_satellite_service.change_detections.delta_nbr IS 'NBR change between pre and post scenes (negative indicates burn/degradation)';
COMMENT ON COLUMN deforestation_satellite_service.change_detections.provenance_hash IS 'SHA-256 provenance hash of change detection inputs and results for reproducibility';
COMMENT ON COLUMN deforestation_satellite_service.baseline_checks.aggregation_method IS 'Multi-point aggregation method: conservative, average, majority_vote, worst_case, best_case';
COMMENT ON COLUMN deforestation_satellite_service.baseline_checks.overall_compliance IS 'Whether the polygon passes baseline forest cover verification under the selected aggregation method';
COMMENT ON COLUMN deforestation_satellite_service.compliance_reports.compliance_status IS 'EUDR compliance determination: compliant, non_compliant, requires_review, insufficient_data, pending, expired';
COMMENT ON COLUMN deforestation_satellite_service.compliance_reports.post_cutoff_alerts IS 'Number of deforestation alerts detected after the EUDR cutoff date (31 Dec 2020)';
COMMENT ON COLUMN deforestation_satellite_service.compliance_reports.provenance_hash IS 'SHA-256 provenance hash of compliance report inputs and evidence for audit trail';
COMMENT ON COLUMN deforestation_satellite_service.monitoring_jobs.frequency IS 'Monitoring frequency: daily, weekly, biweekly, monthly, quarterly, on_demand';
COMMENT ON COLUMN deforestation_satellite_service.monitoring_jobs.current_stage IS 'Current pipeline processing stage for active monitoring jobs';
COMMENT ON COLUMN deforestation_satellite_service.classification_results.land_cover_class IS 'Land cover classification: dense_forest, open_forest, woodland, shrubland, grassland, cropland, wetland, water, urban, barren, mangrove, plantation, agroforestry, degraded_forest, other';
COMMENT ON COLUMN deforestation_satellite_service.classification_results.is_forest IS 'Whether the classified area meets forest definition criteria (tree cover >= threshold)';
COMMENT ON COLUMN deforestation_satellite_service.classification_results.method IS 'Classification method: decision_tree, random_forest, gradient_boosting, neural_network, threshold, rule_based, ensemble';
COMMENT ON COLUMN deforestation_satellite_service.pipeline_metrics.stage IS 'Pipeline processing stage: scene_acquisition, cloud_filtering, index_computation, change_detection, alert_check, classification, baseline_verification, compliance_reporting, full_pipeline';
COMMENT ON COLUMN deforestation_satellite_service.pipeline_metrics.duration_seconds IS 'Stage execution duration in seconds for performance monitoring and SLI tracking';
