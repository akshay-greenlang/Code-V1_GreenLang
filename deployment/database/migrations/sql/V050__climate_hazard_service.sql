-- =============================================================================
-- V050: Climate Hazard Connector Service Tables
-- =============================================================================
-- Component: AGENT-DATA-020 (Climate Hazard Connector)
-- Agent ID:  GL-DATA-GEO-002
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Climate Hazard Connector (GL-DATA-GEO-002) with capabilities for
-- hazard source management (multi-provider ingestion with source_type/
-- hazard_types/coverage/config), hazard data record storage (location JSONB
-- with intensity/probability/frequency/duration metrics per observation),
-- historical climate event cataloging (event severity with affected area/
-- deaths/economic loss tracking), risk index computation (multi-dimensional
-- risk scoring with scenario/time_horizon projections and provenance hashing),
-- climate scenario projection modeling (SSP/RCP pathway projections with
-- warming delta/scaling factor calculations), physical asset registration
-- (geospatial asset inventory with sector/value classification), exposure
-- assessment computation (asset-hazard proximity/intensity/frequency scoring
-- with composite exposure metrics), vulnerability scoring (exposure/
-- sensitivity/adaptive capacity tri-factor vulnerability assessment with
-- IPCC AR6 alignment), compliance report generation (TCFD/CDP/CSRD/ESRS
-- framework-aligned reporting with content hashing), and end-to-end pipeline
-- orchestration with provenance chain tracking via SHA-256 hashes for
-- zero-hallucination audit trails.
-- =============================================================================
-- Tables (10):
--   1. hazard_sources            - Climate hazard data source registry (type/coverage/config)
--   2. hazard_data_records       - Observed hazard data with intensity/probability/frequency metrics
--   3. historical_events         - Historical climate event catalog with impact metrics
--   4. risk_indices              - Computed risk scores by hazard/location/scenario
--   5. scenario_projections      - Climate scenario pathway projections with warming deltas
--   6. assets                    - Physical asset registry with geospatial location/value
--   7. exposure_assessments      - Asset-hazard exposure scoring with composite metrics
--   8. vulnerability_scores      - Tri-factor vulnerability assessment scores
--   9. compliance_reports        - Framework-aligned compliance report generation
--  10. pipeline_runs             - End-to-end pipeline orchestration tracking
--
-- Hypertables (3):
--  11. hazard_observation_events - Hazard observation time-series (hypertable on observed_at)
--  12. risk_calculation_events   - Risk calculation time-series (hypertable on calculated_at)
--  13. pipeline_execution_events - Pipeline execution time-series (hypertable on started_at)
--
-- Continuous Aggregates (2):
--   1. hazard_data_hourly        - Hourly count/avg intensity by hazard_type
--   2. risk_index_daily          - Daily avg risk_score by hazard_type/risk_level
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-GEO-002.
-- Previous: V049__validation_rule_engine_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS climate_hazard_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION climate_hazard_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: climate_hazard_service.hazard_sources
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.hazard_sources (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       VARCHAR(255)  UNIQUE NOT NULL,
    name            VARCHAR(500)  NOT NULL DEFAULT '',
    source_type     VARCHAR(50)   NOT NULL DEFAULT 'api',
    hazard_types    JSONB         NOT NULL DEFAULT '[]'::jsonb,
    coverage        VARCHAR(255)  NOT NULL DEFAULT '',
    config          JSONB         NOT NULL DEFAULT '{}'::jsonb,
    registered_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    tenant_id       VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.hazard_sources
    ADD CONSTRAINT chk_hs_source_id_not_empty CHECK (LENGTH(TRIM(source_id)) > 0);

ALTER TABLE climate_hazard_service.hazard_sources
    ADD CONSTRAINT chk_hs_source_type CHECK (source_type IN (
        'api', 'satellite', 'station', 'model', 'reanalysis',
        'database', 'manual', 'aggregator'
    ));

ALTER TABLE climate_hazard_service.hazard_sources
    ADD CONSTRAINT chk_hs_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_hs_updated_at
    BEFORE UPDATE ON climate_hazard_service.hazard_sources
    FOR EACH ROW EXECUTE FUNCTION climate_hazard_service.set_updated_at();

-- =============================================================================
-- Table 2: climate_hazard_service.hazard_data_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.hazard_data_records (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id       VARCHAR(255)  UNIQUE NOT NULL,
    source_id       VARCHAR(255)  NOT NULL,
    hazard_type     VARCHAR(100)  NOT NULL DEFAULT '',
    location        JSONB         NOT NULL,
    intensity       NUMERIC,
    probability     NUMERIC,
    frequency       NUMERIC,
    duration_days   NUMERIC,
    observed_at     TIMESTAMPTZ   NOT NULL,
    metadata        JSONB         NOT NULL DEFAULT '{}'::jsonb,
    tenant_id       VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT chk_hdr_record_id_not_empty CHECK (LENGTH(TRIM(record_id)) > 0);

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT fk_hdr_source_id
        FOREIGN KEY (source_id)
        REFERENCES climate_hazard_service.hazard_sources(source_id)
        ON DELETE CASCADE;

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT chk_hdr_hazard_type CHECK (hazard_type IN (
        'flood', 'drought', 'cyclone', 'wildfire', 'heatwave',
        'cold_wave', 'sea_level_rise', 'storm_surge', 'landslide',
        'precipitation_extreme', 'wind_extreme', 'coastal_erosion',
        'water_stress', 'permafrost_thaw', 'glacial_retreat', ''
    ));

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT chk_hdr_intensity_non_negative CHECK (intensity IS NULL OR intensity >= 0);

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT chk_hdr_probability_range CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1));

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT chk_hdr_frequency_non_negative CHECK (frequency IS NULL OR frequency >= 0);

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT chk_hdr_duration_non_negative CHECK (duration_days IS NULL OR duration_days >= 0);

ALTER TABLE climate_hazard_service.hazard_data_records
    ADD CONSTRAINT chk_hdr_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 3: climate_hazard_service.historical_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.historical_events (
    id                UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id          VARCHAR(255)  UNIQUE NOT NULL,
    hazard_type       VARCHAR(100)  NOT NULL DEFAULT '',
    location          JSONB         NOT NULL DEFAULT '{}'::jsonb,
    start_date        TIMESTAMPTZ,
    end_date          TIMESTAMPTZ,
    intensity         NUMERIC,
    affected_area_km2 NUMERIC,
    deaths            INTEGER,
    economic_loss_usd NUMERIC,
    source            VARCHAR(500)  NOT NULL DEFAULT '',
    tenant_id         VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_event_id_not_empty CHECK (LENGTH(TRIM(event_id)) > 0);

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_hazard_type CHECK (hazard_type IN (
        'flood', 'drought', 'cyclone', 'wildfire', 'heatwave',
        'cold_wave', 'sea_level_rise', 'storm_surge', 'landslide',
        'precipitation_extreme', 'wind_extreme', 'coastal_erosion',
        'water_stress', 'permafrost_thaw', 'glacial_retreat', ''
    ));

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_intensity_non_negative CHECK (intensity IS NULL OR intensity >= 0);

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_affected_area_non_negative CHECK (affected_area_km2 IS NULL OR affected_area_km2 >= 0);

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_deaths_non_negative CHECK (deaths IS NULL OR deaths >= 0);

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_economic_loss_non_negative CHECK (economic_loss_usd IS NULL OR economic_loss_usd >= 0);

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_date_order CHECK (end_date IS NULL OR start_date IS NULL OR end_date >= start_date);

ALTER TABLE climate_hazard_service.historical_events
    ADD CONSTRAINT chk_he_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 4: climate_hazard_service.risk_indices
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.risk_indices (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    index_id         VARCHAR(255)  UNIQUE NOT NULL,
    hazard_type      VARCHAR(100)  NOT NULL DEFAULT '',
    location         JSONB         NOT NULL DEFAULT '{}'::jsonb,
    risk_score       NUMERIC       NOT NULL,
    risk_level       VARCHAR(20)   NOT NULL DEFAULT 'medium',
    probability      NUMERIC,
    intensity        NUMERIC,
    frequency        NUMERIC,
    duration         NUMERIC,
    scenario         VARCHAR(50)   NOT NULL DEFAULT 'baseline',
    time_horizon     VARCHAR(50)   NOT NULL DEFAULT 'current',
    calculated_at    TIMESTAMPTZ   NOT NULL,
    provenance_hash  VARCHAR(128)  NOT NULL DEFAULT '',
    tenant_id        VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_index_id_not_empty CHECK (LENGTH(TRIM(index_id)) > 0);

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_hazard_type CHECK (hazard_type IN (
        'flood', 'drought', 'cyclone', 'wildfire', 'heatwave',
        'cold_wave', 'sea_level_rise', 'storm_surge', 'landslide',
        'precipitation_extreme', 'wind_extreme', 'coastal_erosion',
        'water_stress', 'permafrost_thaw', 'glacial_retreat', ''
    ));

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_risk_score_range CHECK (risk_score >= 0 AND risk_score <= 100);

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_risk_level CHECK (risk_level IN (
        'negligible', 'low', 'medium', 'high', 'critical'
    ));

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_probability_range CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1));

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_intensity_non_negative CHECK (intensity IS NULL OR intensity >= 0);

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_frequency_non_negative CHECK (frequency IS NULL OR frequency >= 0);

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_duration_non_negative CHECK (duration IS NULL OR duration >= 0);

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_scenario CHECK (scenario IN (
        'baseline', 'ssp126', 'ssp245', 'ssp370', 'ssp585',
        'rcp26', 'rcp45', 'rcp60', 'rcp85', 'custom'
    ));

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_time_horizon CHECK (time_horizon IN (
        'current', '2030', '2040', '2050', '2060', '2070', '2080', '2100', 'custom'
    ));

ALTER TABLE climate_hazard_service.risk_indices
    ADD CONSTRAINT chk_ri_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 5: climate_hazard_service.scenario_projections
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.scenario_projections (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    projection_id    VARCHAR(255)  UNIQUE NOT NULL,
    hazard_type      VARCHAR(100)  NOT NULL DEFAULT '',
    location         JSONB         NOT NULL DEFAULT '{}'::jsonb,
    scenario         VARCHAR(50)   NOT NULL,
    time_horizon     VARCHAR(50)   NOT NULL,
    baseline_risk    JSONB         NOT NULL DEFAULT '{}'::jsonb,
    projected_risk   JSONB         NOT NULL DEFAULT '{}'::jsonb,
    warming_delta_c  NUMERIC,
    scaling_factor   NUMERIC,
    projected_at     TIMESTAMPTZ   NOT NULL,
    provenance_hash  VARCHAR(128)  NOT NULL DEFAULT '',
    tenant_id        VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.scenario_projections
    ADD CONSTRAINT chk_sp_projection_id_not_empty CHECK (LENGTH(TRIM(projection_id)) > 0);

ALTER TABLE climate_hazard_service.scenario_projections
    ADD CONSTRAINT chk_sp_hazard_type CHECK (hazard_type IN (
        'flood', 'drought', 'cyclone', 'wildfire', 'heatwave',
        'cold_wave', 'sea_level_rise', 'storm_surge', 'landslide',
        'precipitation_extreme', 'wind_extreme', 'coastal_erosion',
        'water_stress', 'permafrost_thaw', 'glacial_retreat', ''
    ));

ALTER TABLE climate_hazard_service.scenario_projections
    ADD CONSTRAINT chk_sp_scenario CHECK (scenario IN (
        'baseline', 'ssp126', 'ssp245', 'ssp370', 'ssp585',
        'rcp26', 'rcp45', 'rcp60', 'rcp85', 'custom'
    ));

ALTER TABLE climate_hazard_service.scenario_projections
    ADD CONSTRAINT chk_sp_time_horizon CHECK (time_horizon IN (
        'current', '2030', '2040', '2050', '2060', '2070', '2080', '2100', 'custom'
    ));

ALTER TABLE climate_hazard_service.scenario_projections
    ADD CONSTRAINT chk_sp_scaling_factor_non_negative CHECK (scaling_factor IS NULL OR scaling_factor >= 0);

ALTER TABLE climate_hazard_service.scenario_projections
    ADD CONSTRAINT chk_sp_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 6: climate_hazard_service.assets
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.assets (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id        VARCHAR(255)  UNIQUE NOT NULL,
    name            VARCHAR(500)  NOT NULL DEFAULT '',
    asset_type      VARCHAR(100)  NOT NULL DEFAULT '',
    location        JSONB         NOT NULL,
    sector          VARCHAR(100)  NOT NULL DEFAULT '',
    value_usd       NUMERIC,
    metadata        JSONB         NOT NULL DEFAULT '{}'::jsonb,
    registered_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    tenant_id       VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.assets
    ADD CONSTRAINT chk_a_asset_id_not_empty CHECK (LENGTH(TRIM(asset_id)) > 0);

ALTER TABLE climate_hazard_service.assets
    ADD CONSTRAINT chk_a_asset_type CHECK (asset_type IN (
        'facility', 'warehouse', 'office', 'factory', 'port',
        'pipeline', 'power_plant', 'data_center', 'retail_store',
        'agricultural_land', 'mining_site', 'transport_hub',
        'supply_chain_node', 'infrastructure', 'other', ''
    ));

ALTER TABLE climate_hazard_service.assets
    ADD CONSTRAINT chk_a_sector CHECK (sector IN (
        'energy', 'manufacturing', 'agriculture', 'transport',
        'real_estate', 'finance', 'mining', 'utilities',
        'retail', 'technology', 'healthcare', 'other', ''
    ));

ALTER TABLE climate_hazard_service.assets
    ADD CONSTRAINT chk_a_value_usd_non_negative CHECK (value_usd IS NULL OR value_usd >= 0);

ALTER TABLE climate_hazard_service.assets
    ADD CONSTRAINT chk_a_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_a_updated_at
    BEFORE UPDATE ON climate_hazard_service.assets
    FOR EACH ROW EXECUTE FUNCTION climate_hazard_service.set_updated_at();

-- =============================================================================
-- Table 7: climate_hazard_service.exposure_assessments
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.exposure_assessments (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           VARCHAR(255)  UNIQUE NOT NULL,
    asset_id                VARCHAR(255)  NOT NULL,
    hazard_type             VARCHAR(100)  NOT NULL DEFAULT '',
    exposure_level          VARCHAR(20)   NOT NULL DEFAULT 'medium',
    proximity_score         NUMERIC,
    intensity_at_location   NUMERIC,
    frequency_exposure      NUMERIC,
    composite_score         NUMERIC       NOT NULL,
    scenario                VARCHAR(50)   NOT NULL DEFAULT 'baseline',
    time_horizon            VARCHAR(50)   NOT NULL DEFAULT 'current',
    assessed_at             TIMESTAMPTZ   NOT NULL,
    provenance_hash         VARCHAR(128)  NOT NULL DEFAULT '',
    tenant_id               VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_assessment_id_not_empty CHECK (LENGTH(TRIM(assessment_id)) > 0);

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT fk_ea_asset_id
        FOREIGN KEY (asset_id)
        REFERENCES climate_hazard_service.assets(asset_id)
        ON DELETE CASCADE;

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_hazard_type CHECK (hazard_type IN (
        'flood', 'drought', 'cyclone', 'wildfire', 'heatwave',
        'cold_wave', 'sea_level_rise', 'storm_surge', 'landslide',
        'precipitation_extreme', 'wind_extreme', 'coastal_erosion',
        'water_stress', 'permafrost_thaw', 'glacial_retreat', ''
    ));

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_exposure_level CHECK (exposure_level IN (
        'negligible', 'low', 'medium', 'high', 'critical'
    ));

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_proximity_score_range CHECK (proximity_score IS NULL OR (proximity_score >= 0 AND proximity_score <= 100));

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_intensity_non_negative CHECK (intensity_at_location IS NULL OR intensity_at_location >= 0);

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_frequency_non_negative CHECK (frequency_exposure IS NULL OR frequency_exposure >= 0);

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_composite_score_range CHECK (composite_score >= 0 AND composite_score <= 100);

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_scenario CHECK (scenario IN (
        'baseline', 'ssp126', 'ssp245', 'ssp370', 'ssp585',
        'rcp26', 'rcp45', 'rcp60', 'rcp85', 'custom'
    ));

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_time_horizon CHECK (time_horizon IN (
        'current', '2030', '2040', '2050', '2060', '2070', '2080', '2100', 'custom'
    ));

ALTER TABLE climate_hazard_service.exposure_assessments
    ADD CONSTRAINT chk_ea_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 8: climate_hazard_service.vulnerability_scores
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.vulnerability_scores (
    id                       UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    score_id                 VARCHAR(255)  UNIQUE NOT NULL,
    entity_id                VARCHAR(255)  NOT NULL DEFAULT '',
    hazard_type              VARCHAR(100)  NOT NULL DEFAULT '',
    exposure_score           NUMERIC,
    sensitivity_score        NUMERIC,
    adaptive_capacity_score  NUMERIC,
    vulnerability_score      NUMERIC       NOT NULL,
    vulnerability_level      VARCHAR(20)   NOT NULL DEFAULT 'medium',
    scenario                 VARCHAR(50)   NOT NULL DEFAULT 'baseline',
    time_horizon             VARCHAR(50)   NOT NULL DEFAULT 'current',
    scored_at                TIMESTAMPTZ   NOT NULL,
    provenance_hash          VARCHAR(128)  NOT NULL DEFAULT '',
    tenant_id                VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_score_id_not_empty CHECK (LENGTH(TRIM(score_id)) > 0);

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_hazard_type CHECK (hazard_type IN (
        'flood', 'drought', 'cyclone', 'wildfire', 'heatwave',
        'cold_wave', 'sea_level_rise', 'storm_surge', 'landslide',
        'precipitation_extreme', 'wind_extreme', 'coastal_erosion',
        'water_stress', 'permafrost_thaw', 'glacial_retreat', ''
    ));

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_exposure_range CHECK (exposure_score IS NULL OR (exposure_score >= 0 AND exposure_score <= 100));

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_sensitivity_range CHECK (sensitivity_score IS NULL OR (sensitivity_score >= 0 AND sensitivity_score <= 100));

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_adaptive_range CHECK (adaptive_capacity_score IS NULL OR (adaptive_capacity_score >= 0 AND adaptive_capacity_score <= 100));

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_vulnerability_range CHECK (vulnerability_score >= 0 AND vulnerability_score <= 100);

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_vulnerability_level CHECK (vulnerability_level IN (
        'negligible', 'low', 'medium', 'high', 'critical'
    ));

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_scenario CHECK (scenario IN (
        'baseline', 'ssp126', 'ssp245', 'ssp370', 'ssp585',
        'rcp26', 'rcp45', 'rcp60', 'rcp85', 'custom'
    ));

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_time_horizon CHECK (time_horizon IN (
        'current', '2030', '2040', '2050', '2060', '2070', '2080', '2100', 'custom'
    ));

ALTER TABLE climate_hazard_service.vulnerability_scores
    ADD CONSTRAINT chk_vs_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 9: climate_hazard_service.compliance_reports
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.compliance_reports (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id        VARCHAR(255)  UNIQUE NOT NULL,
    report_type      VARCHAR(50)   NOT NULL,
    format           VARCHAR(20)   NOT NULL,
    framework        VARCHAR(50)   NOT NULL DEFAULT '',
    content          TEXT          NOT NULL DEFAULT '',
    report_hash      VARCHAR(128),
    generated_at     TIMESTAMPTZ   NOT NULL,
    provenance_hash  VARCHAR(128)  NOT NULL DEFAULT '',
    tenant_id        VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.compliance_reports
    ADD CONSTRAINT chk_cr_report_id_not_empty CHECK (LENGTH(TRIM(report_id)) > 0);

ALTER TABLE climate_hazard_service.compliance_reports
    ADD CONSTRAINT chk_cr_report_type CHECK (report_type IN (
        'risk_assessment', 'exposure_summary', 'vulnerability_analysis',
        'scenario_analysis', 'compliance_checklist', 'executive_summary',
        'custom'
    ));

ALTER TABLE climate_hazard_service.compliance_reports
    ADD CONSTRAINT chk_cr_format CHECK (format IN (
        'json', 'text', 'html', 'pdf', 'csv', 'markdown'
    ));

ALTER TABLE climate_hazard_service.compliance_reports
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'tcfd', 'cdp', 'csrd_esrs', 'iso_14090', 'iso_14091',
        'ipcc_ar6', 'ngfs', 'tnfd', 'custom', ''
    ));

ALTER TABLE climate_hazard_service.compliance_reports
    ADD CONSTRAINT chk_cr_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: climate_hazard_service.pipeline_runs
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.pipeline_runs (
    id                   UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_id          VARCHAR(255)  UNIQUE NOT NULL,
    status               VARCHAR(20)   NOT NULL DEFAULT 'pending',
    stages_completed     INTEGER       NOT NULL DEFAULT 0,
    results              JSONB         NOT NULL DEFAULT '{}'::jsonb,
    evaluation_summary   JSONB         NOT NULL DEFAULT '{}'::jsonb,
    report_id            VARCHAR(255),
    duration_ms          NUMERIC,
    started_at           TIMESTAMPTZ   NOT NULL,
    completed_at         TIMESTAMPTZ,
    provenance_hash      VARCHAR(128)  NOT NULL DEFAULT '',
    tenant_id            VARCHAR(255)  NOT NULL
);

ALTER TABLE climate_hazard_service.pipeline_runs
    ADD CONSTRAINT chk_pr_pipeline_id_not_empty CHECK (LENGTH(TRIM(pipeline_id)) > 0);

ALTER TABLE climate_hazard_service.pipeline_runs
    ADD CONSTRAINT chk_pr_status CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'cancelled', 'timeout'
    ));

ALTER TABLE climate_hazard_service.pipeline_runs
    ADD CONSTRAINT chk_pr_stages_non_negative CHECK (stages_completed >= 0);

ALTER TABLE climate_hazard_service.pipeline_runs
    ADD CONSTRAINT chk_pr_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE climate_hazard_service.pipeline_runs
    ADD CONSTRAINT chk_pr_date_order CHECK (completed_at IS NULL OR completed_at >= started_at);

ALTER TABLE climate_hazard_service.pipeline_runs
    ADD CONSTRAINT chk_pr_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: climate_hazard_service.hazard_observation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.hazard_observation_events (
    observed_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    source_id       VARCHAR(255),
    hazard_type     VARCHAR(100),
    intensity       NUMERIC,
    probability     NUMERIC,
    frequency       NUMERIC,
    tenant_id       VARCHAR(255)
);

SELECT create_hypertable(
    'climate_hazard_service.hazard_observation_events',
    'observed_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE climate_hazard_service.hazard_observation_events
    ADD CONSTRAINT chk_hoe_intensity_non_negative CHECK (intensity IS NULL OR intensity >= 0);

ALTER TABLE climate_hazard_service.hazard_observation_events
    ADD CONSTRAINT chk_hoe_probability_range CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1));

ALTER TABLE climate_hazard_service.hazard_observation_events
    ADD CONSTRAINT chk_hoe_frequency_non_negative CHECK (frequency IS NULL OR frequency >= 0);

-- =============================================================================
-- Table 12: climate_hazard_service.risk_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.risk_calculation_events (
    calculated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    hazard_type     VARCHAR(100),
    risk_score      NUMERIC,
    risk_level      VARCHAR(20),
    scenario        VARCHAR(50),
    time_horizon    VARCHAR(50),
    tenant_id       VARCHAR(255)
);

SELECT create_hypertable(
    'climate_hazard_service.risk_calculation_events',
    'calculated_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE climate_hazard_service.risk_calculation_events
    ADD CONSTRAINT chk_rce_risk_score_range CHECK (risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100));

ALTER TABLE climate_hazard_service.risk_calculation_events
    ADD CONSTRAINT chk_rce_risk_level CHECK (
        risk_level IS NULL OR risk_level IN ('negligible', 'low', 'medium', 'high', 'critical')
    );

-- =============================================================================
-- Table 13: climate_hazard_service.pipeline_execution_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS climate_hazard_service.pipeline_execution_events (
    started_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    pipeline_id         VARCHAR(255),
    status              VARCHAR(20),
    stages_completed    INTEGER,
    duration_ms         NUMERIC,
    tenant_id           VARCHAR(255)
);

SELECT create_hypertable(
    'climate_hazard_service.pipeline_execution_events',
    'started_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE climate_hazard_service.pipeline_execution_events
    ADD CONSTRAINT chk_pee_status CHECK (
        status IS NULL OR status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')
    );

ALTER TABLE climate_hazard_service.pipeline_execution_events
    ADD CONSTRAINT chk_pee_stages_non_negative CHECK (stages_completed IS NULL OR stages_completed >= 0);

ALTER TABLE climate_hazard_service.pipeline_execution_events
    ADD CONSTRAINT chk_pee_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- hazard_data_hourly: hourly count/avg intensity by hazard_type
CREATE MATERIALIZED VIEW climate_hazard_service.hazard_data_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', observed_at) AS bucket,
    hazard_type,
    COUNT(*)                           AS total_observations,
    AVG(intensity)                     AS avg_intensity,
    MAX(intensity)                     AS max_intensity,
    AVG(probability)                   AS avg_probability,
    AVG(frequency)                     AS avg_frequency
FROM climate_hazard_service.hazard_observation_events
WHERE observed_at IS NOT NULL
GROUP BY bucket, hazard_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'climate_hazard_service.hazard_data_hourly',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- risk_index_daily: daily avg risk_score by hazard_type/risk_level
CREATE MATERIALIZED VIEW climate_hazard_service.risk_index_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    hazard_type,
    risk_level,
    COUNT(*)                            AS total_calculations,
    AVG(risk_score)                     AS avg_risk_score,
    MAX(risk_score)                     AS max_risk_score,
    MIN(risk_score)                     AS min_risk_score
FROM climate_hazard_service.risk_calculation_events
WHERE calculated_at IS NOT NULL
GROUP BY bucket, hazard_type, risk_level
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'climate_hazard_service.risk_index_daily',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- hazard_sources indexes (10)
CREATE INDEX IF NOT EXISTS idx_hs_source_id            ON climate_hazard_service.hazard_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_hs_name                 ON climate_hazard_service.hazard_sources(name);
CREATE INDEX IF NOT EXISTS idx_hs_source_type          ON climate_hazard_service.hazard_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_hs_coverage             ON climate_hazard_service.hazard_sources(coverage);
CREATE INDEX IF NOT EXISTS idx_hs_registered_at        ON climate_hazard_service.hazard_sources(registered_at DESC);
CREATE INDEX IF NOT EXISTS idx_hs_updated_at           ON climate_hazard_service.hazard_sources(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_hs_tenant_id            ON climate_hazard_service.hazard_sources(tenant_id);
CREATE INDEX IF NOT EXISTS idx_hs_type_tenant          ON climate_hazard_service.hazard_sources(source_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_hs_hazard_types         ON climate_hazard_service.hazard_sources USING GIN (hazard_types);
CREATE INDEX IF NOT EXISTS idx_hs_config               ON climate_hazard_service.hazard_sources USING GIN (config);

-- hazard_data_records indexes (12)
CREATE INDEX IF NOT EXISTS idx_hdr_record_id           ON climate_hazard_service.hazard_data_records(record_id);
CREATE INDEX IF NOT EXISTS idx_hdr_source_id           ON climate_hazard_service.hazard_data_records(source_id);
CREATE INDEX IF NOT EXISTS idx_hdr_hazard_type         ON climate_hazard_service.hazard_data_records(hazard_type);
CREATE INDEX IF NOT EXISTS idx_hdr_intensity           ON climate_hazard_service.hazard_data_records(intensity DESC);
CREATE INDEX IF NOT EXISTS idx_hdr_probability         ON climate_hazard_service.hazard_data_records(probability DESC);
CREATE INDEX IF NOT EXISTS idx_hdr_observed_at         ON climate_hazard_service.hazard_data_records(observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hdr_tenant_id           ON climate_hazard_service.hazard_data_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_hdr_source_hazard       ON climate_hazard_service.hazard_data_records(source_id, hazard_type);
CREATE INDEX IF NOT EXISTS idx_hdr_hazard_observed     ON climate_hazard_service.hazard_data_records(hazard_type, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hdr_tenant_hazard       ON climate_hazard_service.hazard_data_records(tenant_id, hazard_type);
CREATE INDEX IF NOT EXISTS idx_hdr_location            ON climate_hazard_service.hazard_data_records USING GIN (location);
CREATE INDEX IF NOT EXISTS idx_hdr_metadata            ON climate_hazard_service.hazard_data_records USING GIN (metadata);

-- historical_events indexes (12)
CREATE INDEX IF NOT EXISTS idx_he_event_id             ON climate_hazard_service.historical_events(event_id);
CREATE INDEX IF NOT EXISTS idx_he_hazard_type          ON climate_hazard_service.historical_events(hazard_type);
CREATE INDEX IF NOT EXISTS idx_he_start_date           ON climate_hazard_service.historical_events(start_date DESC);
CREATE INDEX IF NOT EXISTS idx_he_end_date             ON climate_hazard_service.historical_events(end_date DESC);
CREATE INDEX IF NOT EXISTS idx_he_intensity            ON climate_hazard_service.historical_events(intensity DESC);
CREATE INDEX IF NOT EXISTS idx_he_affected_area        ON climate_hazard_service.historical_events(affected_area_km2 DESC);
CREATE INDEX IF NOT EXISTS idx_he_economic_loss        ON climate_hazard_service.historical_events(economic_loss_usd DESC);
CREATE INDEX IF NOT EXISTS idx_he_source               ON climate_hazard_service.historical_events(source);
CREATE INDEX IF NOT EXISTS idx_he_tenant_id            ON climate_hazard_service.historical_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_he_hazard_start         ON climate_hazard_service.historical_events(hazard_type, start_date DESC);
CREATE INDEX IF NOT EXISTS idx_he_tenant_hazard        ON climate_hazard_service.historical_events(tenant_id, hazard_type);
CREATE INDEX IF NOT EXISTS idx_he_location             ON climate_hazard_service.historical_events USING GIN (location);

-- risk_indices indexes (12)
CREATE INDEX IF NOT EXISTS idx_ri_index_id             ON climate_hazard_service.risk_indices(index_id);
CREATE INDEX IF NOT EXISTS idx_ri_hazard_type          ON climate_hazard_service.risk_indices(hazard_type);
CREATE INDEX IF NOT EXISTS idx_ri_risk_score           ON climate_hazard_service.risk_indices(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_ri_risk_level           ON climate_hazard_service.risk_indices(risk_level);
CREATE INDEX IF NOT EXISTS idx_ri_scenario             ON climate_hazard_service.risk_indices(scenario);
CREATE INDEX IF NOT EXISTS idx_ri_time_horizon         ON climate_hazard_service.risk_indices(time_horizon);
CREATE INDEX IF NOT EXISTS idx_ri_calculated_at        ON climate_hazard_service.risk_indices(calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ri_provenance_hash      ON climate_hazard_service.risk_indices(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_ri_tenant_id            ON climate_hazard_service.risk_indices(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ri_hazard_scenario      ON climate_hazard_service.risk_indices(hazard_type, scenario);
CREATE INDEX IF NOT EXISTS idx_ri_tenant_hazard        ON climate_hazard_service.risk_indices(tenant_id, hazard_type);
CREATE INDEX IF NOT EXISTS idx_ri_location             ON climate_hazard_service.risk_indices USING GIN (location);

-- scenario_projections indexes (12)
CREATE INDEX IF NOT EXISTS idx_sp_projection_id        ON climate_hazard_service.scenario_projections(projection_id);
CREATE INDEX IF NOT EXISTS idx_sp_hazard_type          ON climate_hazard_service.scenario_projections(hazard_type);
CREATE INDEX IF NOT EXISTS idx_sp_scenario             ON climate_hazard_service.scenario_projections(scenario);
CREATE INDEX IF NOT EXISTS idx_sp_time_horizon         ON climate_hazard_service.scenario_projections(time_horizon);
CREATE INDEX IF NOT EXISTS idx_sp_warming_delta        ON climate_hazard_service.scenario_projections(warming_delta_c DESC);
CREATE INDEX IF NOT EXISTS idx_sp_projected_at         ON climate_hazard_service.scenario_projections(projected_at DESC);
CREATE INDEX IF NOT EXISTS idx_sp_provenance_hash      ON climate_hazard_service.scenario_projections(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_sp_tenant_id            ON climate_hazard_service.scenario_projections(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sp_hazard_scenario      ON climate_hazard_service.scenario_projections(hazard_type, scenario);
CREATE INDEX IF NOT EXISTS idx_sp_scenario_horizon     ON climate_hazard_service.scenario_projections(scenario, time_horizon);
CREATE INDEX IF NOT EXISTS idx_sp_baseline_risk        ON climate_hazard_service.scenario_projections USING GIN (baseline_risk);
CREATE INDEX IF NOT EXISTS idx_sp_projected_risk       ON climate_hazard_service.scenario_projections USING GIN (projected_risk);

-- assets indexes (12)
CREATE INDEX IF NOT EXISTS idx_a_asset_id              ON climate_hazard_service.assets(asset_id);
CREATE INDEX IF NOT EXISTS idx_a_name                  ON climate_hazard_service.assets(name);
CREATE INDEX IF NOT EXISTS idx_a_asset_type            ON climate_hazard_service.assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_a_sector                ON climate_hazard_service.assets(sector);
CREATE INDEX IF NOT EXISTS idx_a_value_usd             ON climate_hazard_service.assets(value_usd DESC);
CREATE INDEX IF NOT EXISTS idx_a_registered_at         ON climate_hazard_service.assets(registered_at DESC);
CREATE INDEX IF NOT EXISTS idx_a_updated_at            ON climate_hazard_service.assets(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_a_tenant_id             ON climate_hazard_service.assets(tenant_id);
CREATE INDEX IF NOT EXISTS idx_a_type_sector           ON climate_hazard_service.assets(asset_type, sector);
CREATE INDEX IF NOT EXISTS idx_a_tenant_type           ON climate_hazard_service.assets(tenant_id, asset_type);
CREATE INDEX IF NOT EXISTS idx_a_location              ON climate_hazard_service.assets USING GIN (location);
CREATE INDEX IF NOT EXISTS idx_a_metadata              ON climate_hazard_service.assets USING GIN (metadata);

-- exposure_assessments indexes (12)
CREATE INDEX IF NOT EXISTS idx_ea_assessment_id        ON climate_hazard_service.exposure_assessments(assessment_id);
CREATE INDEX IF NOT EXISTS idx_ea_asset_id             ON climate_hazard_service.exposure_assessments(asset_id);
CREATE INDEX IF NOT EXISTS idx_ea_hazard_type          ON climate_hazard_service.exposure_assessments(hazard_type);
CREATE INDEX IF NOT EXISTS idx_ea_exposure_level       ON climate_hazard_service.exposure_assessments(exposure_level);
CREATE INDEX IF NOT EXISTS idx_ea_composite_score      ON climate_hazard_service.exposure_assessments(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_ea_scenario             ON climate_hazard_service.exposure_assessments(scenario);
CREATE INDEX IF NOT EXISTS idx_ea_time_horizon         ON climate_hazard_service.exposure_assessments(time_horizon);
CREATE INDEX IF NOT EXISTS idx_ea_assessed_at          ON climate_hazard_service.exposure_assessments(assessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_ea_provenance_hash      ON climate_hazard_service.exposure_assessments(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_ea_tenant_id            ON climate_hazard_service.exposure_assessments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ea_asset_hazard         ON climate_hazard_service.exposure_assessments(asset_id, hazard_type);
CREATE INDEX IF NOT EXISTS idx_ea_tenant_hazard        ON climate_hazard_service.exposure_assessments(tenant_id, hazard_type);

-- vulnerability_scores indexes (12)
CREATE INDEX IF NOT EXISTS idx_vs_score_id             ON climate_hazard_service.vulnerability_scores(score_id);
CREATE INDEX IF NOT EXISTS idx_vs_entity_id            ON climate_hazard_service.vulnerability_scores(entity_id);
CREATE INDEX IF NOT EXISTS idx_vs_hazard_type          ON climate_hazard_service.vulnerability_scores(hazard_type);
CREATE INDEX IF NOT EXISTS idx_vs_vulnerability_score  ON climate_hazard_service.vulnerability_scores(vulnerability_score DESC);
CREATE INDEX IF NOT EXISTS idx_vs_vulnerability_level  ON climate_hazard_service.vulnerability_scores(vulnerability_level);
CREATE INDEX IF NOT EXISTS idx_vs_scenario             ON climate_hazard_service.vulnerability_scores(scenario);
CREATE INDEX IF NOT EXISTS idx_vs_time_horizon         ON climate_hazard_service.vulnerability_scores(time_horizon);
CREATE INDEX IF NOT EXISTS idx_vs_scored_at            ON climate_hazard_service.vulnerability_scores(scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_vs_provenance_hash      ON climate_hazard_service.vulnerability_scores(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_vs_tenant_id            ON climate_hazard_service.vulnerability_scores(tenant_id);
CREATE INDEX IF NOT EXISTS idx_vs_entity_hazard        ON climate_hazard_service.vulnerability_scores(entity_id, hazard_type);
CREATE INDEX IF NOT EXISTS idx_vs_tenant_hazard        ON climate_hazard_service.vulnerability_scores(tenant_id, hazard_type);

-- compliance_reports indexes (10)
CREATE INDEX IF NOT EXISTS idx_cr_report_id            ON climate_hazard_service.compliance_reports(report_id);
CREATE INDEX IF NOT EXISTS idx_cr_report_type          ON climate_hazard_service.compliance_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_cr_format               ON climate_hazard_service.compliance_reports(format);
CREATE INDEX IF NOT EXISTS idx_cr_framework            ON climate_hazard_service.compliance_reports(framework);
CREATE INDEX IF NOT EXISTS idx_cr_generated_at         ON climate_hazard_service.compliance_reports(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_cr_report_hash          ON climate_hazard_service.compliance_reports(report_hash);
CREATE INDEX IF NOT EXISTS idx_cr_provenance_hash      ON climate_hazard_service.compliance_reports(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_cr_tenant_id            ON climate_hazard_service.compliance_reports(tenant_id);
CREATE INDEX IF NOT EXISTS idx_cr_type_framework       ON climate_hazard_service.compliance_reports(report_type, framework);
CREATE INDEX IF NOT EXISTS idx_cr_tenant_type          ON climate_hazard_service.compliance_reports(tenant_id, report_type);

-- pipeline_runs indexes (12)
CREATE INDEX IF NOT EXISTS idx_pr_pipeline_id          ON climate_hazard_service.pipeline_runs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_pr_status               ON climate_hazard_service.pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_pr_stages_completed     ON climate_hazard_service.pipeline_runs(stages_completed);
CREATE INDEX IF NOT EXISTS idx_pr_report_id            ON climate_hazard_service.pipeline_runs(report_id);
CREATE INDEX IF NOT EXISTS idx_pr_duration_ms          ON climate_hazard_service.pipeline_runs(duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_pr_started_at           ON climate_hazard_service.pipeline_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pr_completed_at         ON climate_hazard_service.pipeline_runs(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_pr_provenance_hash      ON climate_hazard_service.pipeline_runs(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_pr_tenant_id            ON climate_hazard_service.pipeline_runs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pr_status_started       ON climate_hazard_service.pipeline_runs(status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pr_tenant_status        ON climate_hazard_service.pipeline_runs(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_pr_results              ON climate_hazard_service.pipeline_runs USING GIN (results);

-- hazard_observation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_hoe_source_id           ON climate_hazard_service.hazard_observation_events(source_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hoe_hazard_type         ON climate_hazard_service.hazard_observation_events(hazard_type, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hoe_intensity           ON climate_hazard_service.hazard_observation_events(intensity, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hoe_tenant_id           ON climate_hazard_service.hazard_observation_events(tenant_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hoe_source_hazard       ON climate_hazard_service.hazard_observation_events(source_id, hazard_type, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_hoe_tenant_hazard       ON climate_hazard_service.hazard_observation_events(tenant_id, hazard_type, observed_at DESC);

-- risk_calculation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_rce_hazard_type         ON climate_hazard_service.risk_calculation_events(hazard_type, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rce_risk_level          ON climate_hazard_service.risk_calculation_events(risk_level, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rce_scenario            ON climate_hazard_service.risk_calculation_events(scenario, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rce_tenant_id           ON climate_hazard_service.risk_calculation_events(tenant_id, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rce_hazard_level        ON climate_hazard_service.risk_calculation_events(hazard_type, risk_level, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rce_tenant_hazard       ON climate_hazard_service.risk_calculation_events(tenant_id, hazard_type, calculated_at DESC);

-- pipeline_execution_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_pee_pipeline_id         ON climate_hazard_service.pipeline_execution_events(pipeline_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pee_status              ON climate_hazard_service.pipeline_execution_events(status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pee_tenant_id           ON climate_hazard_service.pipeline_execution_events(tenant_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pee_stages              ON climate_hazard_service.pipeline_execution_events(stages_completed, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pee_tenant_status       ON climate_hazard_service.pipeline_execution_events(tenant_id, status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pee_pipeline_status     ON climate_hazard_service.pipeline_execution_events(pipeline_id, status, started_at DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- hazard_sources: tenant-isolated
ALTER TABLE climate_hazard_service.hazard_sources ENABLE ROW LEVEL SECURITY;
CREATE POLICY hs_read  ON climate_hazard_service.hazard_sources FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY hs_write ON climate_hazard_service.hazard_sources FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- hazard_data_records: tenant-isolated
ALTER TABLE climate_hazard_service.hazard_data_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY hdr_read  ON climate_hazard_service.hazard_data_records FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY hdr_write ON climate_hazard_service.hazard_data_records FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- historical_events: tenant-isolated
ALTER TABLE climate_hazard_service.historical_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY he_read  ON climate_hazard_service.historical_events FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY he_write ON climate_hazard_service.historical_events FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- risk_indices: tenant-isolated
ALTER TABLE climate_hazard_service.risk_indices ENABLE ROW LEVEL SECURITY;
CREATE POLICY ri_read  ON climate_hazard_service.risk_indices FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ri_write ON climate_hazard_service.risk_indices FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- scenario_projections: tenant-isolated
ALTER TABLE climate_hazard_service.scenario_projections ENABLE ROW LEVEL SECURITY;
CREATE POLICY sp_read  ON climate_hazard_service.scenario_projections FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sp_write ON climate_hazard_service.scenario_projections FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- assets: tenant-isolated
ALTER TABLE climate_hazard_service.assets ENABLE ROW LEVEL SECURITY;
CREATE POLICY a_read  ON climate_hazard_service.assets FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY a_write ON climate_hazard_service.assets FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- exposure_assessments: tenant-isolated
ALTER TABLE climate_hazard_service.exposure_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY ea_read  ON climate_hazard_service.exposure_assessments FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ea_write ON climate_hazard_service.exposure_assessments FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- vulnerability_scores: tenant-isolated
ALTER TABLE climate_hazard_service.vulnerability_scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY vs_read  ON climate_hazard_service.vulnerability_scores FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY vs_write ON climate_hazard_service.vulnerability_scores FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- compliance_reports: tenant-isolated
ALTER TABLE climate_hazard_service.compliance_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY cr_read  ON climate_hazard_service.compliance_reports FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY cr_write ON climate_hazard_service.compliance_reports FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pipeline_runs: tenant-isolated
ALTER TABLE climate_hazard_service.pipeline_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY pr_read  ON climate_hazard_service.pipeline_runs FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pr_write ON climate_hazard_service.pipeline_runs FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- hazard_observation_events: open read/write (time-series telemetry)
ALTER TABLE climate_hazard_service.hazard_observation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY hoe_read  ON climate_hazard_service.hazard_observation_events FOR SELECT USING (TRUE);
CREATE POLICY hoe_write ON climate_hazard_service.hazard_observation_events FOR ALL   USING (TRUE);

-- risk_calculation_events: open read/write (time-series telemetry)
ALTER TABLE climate_hazard_service.risk_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY rce_read  ON climate_hazard_service.risk_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY rce_write ON climate_hazard_service.risk_calculation_events FOR ALL   USING (TRUE);

-- pipeline_execution_events: open read/write (time-series telemetry)
ALTER TABLE climate_hazard_service.pipeline_execution_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pee_read  ON climate_hazard_service.pipeline_execution_events FOR SELECT USING (TRUE);
CREATE POLICY pee_write ON climate_hazard_service.pipeline_execution_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA climate_hazard_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA climate_hazard_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA climate_hazard_service TO greenlang_app;
GRANT SELECT ON climate_hazard_service.hazard_data_hourly TO greenlang_app;
GRANT SELECT ON climate_hazard_service.risk_index_daily TO greenlang_app;

GRANT USAGE ON SCHEMA climate_hazard_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA climate_hazard_service TO greenlang_readonly;
GRANT SELECT ON climate_hazard_service.hazard_data_hourly TO greenlang_readonly;
GRANT SELECT ON climate_hazard_service.risk_index_daily TO greenlang_readonly;

GRANT ALL ON SCHEMA climate_hazard_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA climate_hazard_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA climate_hazard_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'climate-hazard:sources:read',           'climate-hazard', 'sources_read',           'View climate hazard data sources, types, coverage, and configuration'),
    (gen_random_uuid(), 'climate-hazard:sources:write',          'climate-hazard', 'sources_write',          'Register, update, and manage climate hazard data sources'),
    (gen_random_uuid(), 'climate-hazard:records:read',           'climate-hazard', 'records_read',           'View hazard data records with intensity, probability, frequency, and duration metrics'),
    (gen_random_uuid(), 'climate-hazard:records:write',          'climate-hazard', 'records_write',          'Ingest and update hazard data records from registered sources'),
    (gen_random_uuid(), 'climate-hazard:events:read',            'climate-hazard', 'events_read',            'View historical climate events with impact metrics (area, deaths, economic loss)'),
    (gen_random_uuid(), 'climate-hazard:events:write',           'climate-hazard', 'events_write',           'Create and update historical climate event records'),
    (gen_random_uuid(), 'climate-hazard:risk:read',              'climate-hazard', 'risk_read',              'View computed risk indices by hazard type, scenario, and time horizon'),
    (gen_random_uuid(), 'climate-hazard:risk:write',             'climate-hazard', 'risk_write',             'Calculate and store risk index scores with provenance tracking'),
    (gen_random_uuid(), 'climate-hazard:projections:read',       'climate-hazard', 'projections_read',       'View climate scenario projections with warming deltas and scaling factors'),
    (gen_random_uuid(), 'climate-hazard:projections:write',      'climate-hazard', 'projections_write',      'Generate and store climate scenario pathway projections'),
    (gen_random_uuid(), 'climate-hazard:assets:read',            'climate-hazard', 'assets_read',            'View physical asset registry with geospatial location and value data'),
    (gen_random_uuid(), 'climate-hazard:assets:write',           'climate-hazard', 'assets_write',           'Register, update, and manage physical assets for exposure assessment'),
    (gen_random_uuid(), 'climate-hazard:exposure:read',          'climate-hazard', 'exposure_read',          'View exposure assessments with proximity, intensity, and composite scores'),
    (gen_random_uuid(), 'climate-hazard:exposure:write',         'climate-hazard', 'exposure_write',         'Compute and store asset-hazard exposure assessment scores'),
    (gen_random_uuid(), 'climate-hazard:vulnerability:read',     'climate-hazard', 'vulnerability_read',     'View vulnerability scores with exposure, sensitivity, and adaptive capacity factors'),
    (gen_random_uuid(), 'climate-hazard:vulnerability:write',    'climate-hazard', 'vulnerability_write',    'Compute and store tri-factor vulnerability assessment scores'),
    (gen_random_uuid(), 'climate-hazard:reports:read',           'climate-hazard', 'reports_read',           'View generated compliance reports and their content'),
    (gen_random_uuid(), 'climate-hazard:reports:write',          'climate-hazard', 'reports_write',          'Generate TCFD/CDP/CSRD/ESRS framework-aligned compliance reports'),
    (gen_random_uuid(), 'climate-hazard:pipelines:read',         'climate-hazard', 'pipelines_read',         'View pipeline execution runs, stages, results, and durations'),
    (gen_random_uuid(), 'climate-hazard:admin',                  'climate-hazard', 'admin',                  'Climate hazard connector service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('climate_hazard_service.hazard_observation_events', INTERVAL '90 days');
SELECT add_retention_policy('climate_hazard_service.risk_calculation_events',   INTERVAL '90 days');
SELECT add_retention_policy('climate_hazard_service.pipeline_execution_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE climate_hazard_service.hazard_observation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'hazard_type',
         timescaledb.compress_orderby   = 'observed_at DESC');
SELECT add_compression_policy('climate_hazard_service.hazard_observation_events', INTERVAL '7 days');

ALTER TABLE climate_hazard_service.risk_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'hazard_type',
         timescaledb.compress_orderby   = 'calculated_at DESC');
SELECT add_compression_policy('climate_hazard_service.risk_calculation_events', INTERVAL '7 days');

ALTER TABLE climate_hazard_service.pipeline_execution_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'status',
         timescaledb.compress_orderby   = 'started_at DESC');
SELECT add_compression_policy('climate_hazard_service.pipeline_execution_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Climate Hazard Connector (GL-DATA-GEO-002)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-DATA-GEO-002',
    'Climate Hazard Connector',
    'Climate hazard connector for GreenLang Climate OS. Manages climate hazard data source registry with multi-provider ingestion (api/satellite/station/model/reanalysis/database/manual/aggregator). Stores hazard data records with intensity/probability/frequency/duration metrics per geospatial observation across 15 hazard types (flood/drought/cyclone/wildfire/heatwave/cold_wave/sea_level_rise/storm_surge/landslide/precipitation_extreme/wind_extreme/coastal_erosion/water_stress/permafrost_thaw/glacial_retreat). Catalogs historical climate events with impact metrics (affected area km2, deaths, economic loss USD). Computes multi-dimensional risk indices (0-100 scale, 5 risk levels) across 10 climate scenarios (SSP1-2.6/SSP2-4.5/SSP3-7.0/SSP5-8.5/RCP2.6/RCP4.5/RCP6.0/RCP8.5/baseline/custom) and 9 time horizons (current/2030-2100/custom). Models scenario projections with warming delta and scaling factor calculations. Registers physical assets with geospatial location and sector/value classification. Computes exposure assessments with proximity/intensity/frequency scoring and composite metrics. Calculates tri-factor vulnerability scores (exposure/sensitivity/adaptive capacity) aligned with IPCC AR6. Generates framework-aligned compliance reports (TCFD/CDP/CSRD-ESRS/ISO 14090/ISO 14091/IPCC AR6/NGFS/TNFD) in 6 formats. Orchestrates end-to-end pipelines with stage tracking and provenance. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 5, '1.0.0', true,
    'GreenLang Data Team',
    'https://docs.greenlang.ai/agents/climate-hazard-connector',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-DATA-GEO-002', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/climate-hazard-connector-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"climate-hazard", "physical-risk", "tcfd", "scenario-analysis", "exposure", "vulnerability"}',
    '{"cross-sector", "energy", "manufacturing", "agriculture", "real-estate", "finance", "mining", "utilities"}',
    'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-DATA-GEO-002', '1.0.0',
    'hazard_source_management',
    'configuration',
    'Register and manage climate hazard data sources with type classification, hazard type mapping, coverage definition, and connection configuration.',
    '{"source_id", "name", "source_type", "hazard_types", "coverage", "config"}',
    '{"source_id", "registration_result"}',
    '{"source_types": ["api", "satellite", "station", "model", "reanalysis", "database", "manual", "aggregator"], "supports_multi_hazard": true}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'hazard_data_ingestion',
    'processing',
    'Ingest hazard observation data from registered sources with intensity, probability, frequency, and duration metrics at geospatial locations.',
    '{"source_id", "hazard_type", "location", "intensity", "probability", "frequency", "duration_days", "observed_at"}',
    '{"record_id", "ingestion_result", "observation_count"}',
    '{"hazard_types": ["flood", "drought", "cyclone", "wildfire", "heatwave", "cold_wave", "sea_level_rise", "storm_surge", "landslide", "precipitation_extreme", "wind_extreme", "coastal_erosion", "water_stress", "permafrost_thaw", "glacial_retreat"], "supports_batch": true}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'historical_event_catalog',
    'processing',
    'Catalog historical climate events with hazard classification, temporal bounds, intensity, affected area, human impact, and economic loss metrics.',
    '{"event_id", "hazard_type", "location", "start_date", "end_date", "intensity", "affected_area_km2", "deaths", "economic_loss_usd", "source"}',
    '{"event_id", "catalog_result"}',
    '{"impact_metrics": ["affected_area_km2", "deaths", "economic_loss_usd"], "supports_temporal_query": true}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'risk_index_computation',
    'processing',
    'Compute multi-dimensional risk indices combining probability, intensity, frequency, and duration factors across climate scenarios and time horizons.',
    '{"hazard_type", "location", "scenario", "time_horizon"}',
    '{"index_id", "risk_score", "risk_level", "provenance_hash"}',
    '{"risk_levels": ["negligible", "low", "medium", "high", "critical"], "score_range": [0, 100], "scenarios": ["baseline", "ssp126", "ssp245", "ssp370", "ssp585", "rcp26", "rcp45", "rcp60", "rcp85", "custom"]}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'scenario_projection',
    'processing',
    'Generate climate scenario pathway projections with warming delta calculations and scaling factor modeling for future risk estimation.',
    '{"hazard_type", "location", "scenario", "time_horizon", "baseline_risk"}',
    '{"projection_id", "projected_risk", "warming_delta_c", "scaling_factor", "provenance_hash"}',
    '{"pathways": ["ssp126", "ssp245", "ssp370", "ssp585", "rcp26", "rcp45", "rcp60", "rcp85"], "time_horizons": ["current", "2030", "2040", "2050", "2060", "2070", "2080", "2100", "custom"]}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'asset_management',
    'configuration',
    'Register and manage physical assets with geospatial coordinates, asset type classification, sector assignment, and valuation for exposure assessment.',
    '{"asset_id", "name", "asset_type", "location", "sector", "value_usd", "metadata"}',
    '{"asset_id", "registration_result"}',
    '{"asset_types": ["facility", "warehouse", "office", "factory", "port", "pipeline", "power_plant", "data_center", "retail_store", "agricultural_land", "mining_site", "transport_hub", "supply_chain_node", "infrastructure", "other"], "sectors": ["energy", "manufacturing", "agriculture", "transport", "real_estate", "finance", "mining", "utilities", "retail", "technology", "healthcare", "other"]}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'exposure_assessment',
    'processing',
    'Compute asset-hazard exposure assessments with proximity scoring, intensity-at-location estimation, frequency exposure, and composite scoring.',
    '{"asset_id", "hazard_type", "scenario", "time_horizon"}',
    '{"assessment_id", "exposure_level", "composite_score", "provenance_hash"}',
    '{"exposure_levels": ["negligible", "low", "medium", "high", "critical"], "score_range": [0, 100], "factors": ["proximity", "intensity", "frequency"]}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'vulnerability_assessment',
    'processing',
    'Calculate tri-factor vulnerability scores combining exposure, sensitivity, and adaptive capacity aligned with IPCC AR6 framework.',
    '{"entity_id", "hazard_type", "exposure_score", "sensitivity_score", "adaptive_capacity_score", "scenario", "time_horizon"}',
    '{"score_id", "vulnerability_score", "vulnerability_level", "provenance_hash"}',
    '{"vulnerability_levels": ["negligible", "low", "medium", "high", "critical"], "score_range": [0, 100], "framework": "IPCC_AR6"}'::jsonb
),
(
    'GL-DATA-GEO-002', '1.0.0',
    'compliance_reporting',
    'reporting',
    'Generate framework-aligned compliance reports for physical climate risk disclosure including TCFD, CDP, CSRD/ESRS, and ISO standards.',
    '{"report_type", "format", "framework", "parameters"}',
    '{"report_id", "content", "report_hash", "provenance_hash"}',
    '{"report_types": ["risk_assessment", "exposure_summary", "vulnerability_analysis", "scenario_analysis", "compliance_checklist", "executive_summary", "custom"], "formats": ["json", "text", "html", "pdf", "csv", "markdown"], "frameworks": ["tcfd", "cdp", "csrd_esrs", "iso_14090", "iso_14091", "ipcc_ar6", "ngfs", "tnfd", "custom"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-DATA-GEO-002', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage climate hazard pipeline execution ordering'),
    ('GL-DATA-GEO-002', 'GL-FOUND-X-007', '>=1.0.0', false, 'Agent version and capability lookup for climate hazard connector service registration'),
    ('GL-DATA-GEO-002', 'GL-FOUND-X-006', '>=1.0.0', false, 'Access control enforcement for hazard data, risk indices, and compliance report permissions'),
    ('GL-DATA-GEO-002', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for hazard ingestion, risk computation, and pipeline execution events'),
    ('GL-DATA-GEO-002', 'GL-FOUND-X-005', '>=1.0.0', true,  'Provenance and audit trail registration with citation service for risk calculation lineage'),
    ('GL-DATA-GEO-002', 'GL-FOUND-X-008', '>=1.0.0', true,  'Reproducibility verification for deterministic risk index and vulnerability score hashing'),
    ('GL-DATA-GEO-002', 'GL-FOUND-X-009', '>=1.0.0', true,  'QA Test Harness zero-hallucination verification of risk calculation outputs'),
    ('GL-DATA-GEO-002', 'GL-DATA-GEO-001', '>=1.0.0', true,  'GIS/Mapping Connector for geospatial coordinate transformation and spatial analysis'),
    ('GL-DATA-GEO-002', 'GL-DATA-SAT-001', '>=1.0.0', true,  'Deforestation Satellite Connector for vegetation index and land cover change data')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-DATA-GEO-002',
    'Climate Hazard Connector',
    'Climate hazard connector. Hazard source management (8 source types, multi-provider ingestion). Hazard data ingestion (15 hazard types, intensity/probability/frequency/duration metrics, geospatial location). Historical event catalog (affected area, deaths, economic loss). Risk index computation (0-100 scale, 5 risk levels, 10 scenarios, 9 time horizons). Scenario projections (SSP/RCP pathways, warming delta, scaling factor). Asset management (15 asset types, 12 sectors, geospatial location, valuation). Exposure assessment (proximity/intensity/frequency scoring, composite metrics). Vulnerability assessment (exposure/sensitivity/adaptive capacity tri-factor, IPCC AR6 aligned). Compliance reporting (TCFD/CDP/CSRD-ESRS/ISO 14090/ISO 14091/IPCC AR6/NGFS/TNFD, 6 formats). Pipeline orchestration (multi-stage, stage tracking). SHA-256 provenance chains.',
    'data', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA climate_hazard_service IS
    'Climate Hazard Connector (AGENT-DATA-020) - hazard source management, hazard data ingestion, historical event catalog, risk index computation, scenario projections, asset management, exposure assessment, vulnerability assessment, compliance reporting, pipeline orchestration, provenance chains';

COMMENT ON TABLE climate_hazard_service.hazard_sources IS
    'Climate hazard data source registry: source_id (unique), name, source_type (8 types), hazard_types JSONB, coverage, config JSONB, registered_at, updated_at, tenant_id';

COMMENT ON TABLE climate_hazard_service.hazard_data_records IS
    'Observed hazard data records: record_id (unique), source_id FK, hazard_type (15 types), location JSONB, intensity, probability (0-1), frequency, duration_days, observed_at, metadata JSONB, tenant_id';

COMMENT ON TABLE climate_hazard_service.historical_events IS
    'Historical climate event catalog: event_id (unique), hazard_type (15 types), location JSONB, start/end dates, intensity, affected_area_km2, deaths, economic_loss_usd, source, tenant_id';

COMMENT ON TABLE climate_hazard_service.risk_indices IS
    'Computed risk index scores: index_id (unique), hazard_type, location JSONB, risk_score (0-100), risk_level (5 levels), probability, intensity, frequency, duration, scenario (10 scenarios), time_horizon (9 horizons), calculated_at, provenance_hash, tenant_id';

COMMENT ON TABLE climate_hazard_service.scenario_projections IS
    'Climate scenario pathway projections: projection_id (unique), hazard_type, location JSONB, scenario (SSP/RCP), time_horizon, baseline_risk JSONB, projected_risk JSONB, warming_delta_c, scaling_factor, projected_at, provenance_hash, tenant_id';

COMMENT ON TABLE climate_hazard_service.assets IS
    'Physical asset registry: asset_id (unique), name, asset_type (15 types), location JSONB, sector (12 sectors), value_usd, metadata JSONB, registered_at, updated_at, tenant_id';

COMMENT ON TABLE climate_hazard_service.exposure_assessments IS
    'Asset-hazard exposure assessment: assessment_id (unique), asset_id FK, hazard_type, exposure_level (5 levels), proximity_score, intensity_at_location, frequency_exposure, composite_score (0-100), scenario, time_horizon, assessed_at, provenance_hash, tenant_id';

COMMENT ON TABLE climate_hazard_service.vulnerability_scores IS
    'Tri-factor vulnerability assessment: score_id (unique), entity_id, hazard_type, exposure_score, sensitivity_score, adaptive_capacity_score (all 0-100), vulnerability_score (0-100), vulnerability_level (5 levels), scenario, time_horizon, scored_at, provenance_hash, tenant_id';

COMMENT ON TABLE climate_hazard_service.compliance_reports IS
    'Framework-aligned compliance reports: report_id (unique), report_type (7 types), format (6 formats), framework (9 frameworks), content TEXT, report_hash, generated_at, provenance_hash, tenant_id';

COMMENT ON TABLE climate_hazard_service.pipeline_runs IS
    'Pipeline execution tracking: pipeline_id (unique), status (6 statuses), stages_completed, results JSONB, evaluation_summary JSONB, report_id, duration_ms, started_at, completed_at, provenance_hash, tenant_id';

COMMENT ON TABLE climate_hazard_service.hazard_observation_events IS
    'TimescaleDB hypertable: hazard observation events with source_id, hazard_type, intensity, probability, frequency, tenant_id (7-day chunks, 90-day retention)';

COMMENT ON TABLE climate_hazard_service.risk_calculation_events IS
    'TimescaleDB hypertable: risk calculation events with hazard_type, risk_score, risk_level, scenario, time_horizon, tenant_id (7-day chunks, 90-day retention)';

COMMENT ON TABLE climate_hazard_service.pipeline_execution_events IS
    'TimescaleDB hypertable: pipeline execution events with pipeline_id, status, stages_completed, duration_ms, tenant_id (7-day chunks, 90-day retention)';

COMMENT ON MATERIALIZED VIEW climate_hazard_service.hazard_data_hourly IS
    'Continuous aggregate: hourly hazard observation stats by hazard_type (total observations, avg/max intensity, avg probability, avg frequency per hour)';

COMMENT ON MATERIALIZED VIEW climate_hazard_service.risk_index_daily IS
    'Continuous aggregate: daily risk calculation stats by hazard_type/risk_level (total calculations, avg/max/min risk score per day)';
