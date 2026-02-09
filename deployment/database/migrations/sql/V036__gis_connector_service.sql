-- =============================================================================
-- V036: GIS/Mapping Connector Service Schema
-- =============================================================================
-- Component: AGENT-DATA-006 (GIS/Mapping Connector)
-- Agent ID:  GL-DATA-GEO-001
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- GIS/Mapping Connector Agent (GL-DATA-GEO-001) with capabilities
-- for geospatial layer management, feature storage, CRS definitions,
-- spatial operations, geocoding caching, boundary datasets,
-- land cover classification, format conversions, spatial indexing,
-- and operation metrics tracking.
-- =============================================================================
-- Tables (10):
--   1. geospatial_layers       - Layer definitions with geometry type and CRS
--   2. layer_features           - Individual features per layer with coordinates
--   3. crs_definitions          - Coordinate Reference System definitions
--   4. spatial_operations       - Spatial operation audit log (hypertable)
--   5. geocoding_cache          - Forward/reverse geocoding result cache
--   6. boundary_datasets        - Administrative and environmental boundaries
--   7. land_cover_data          - Land cover classification records
--   8. format_conversions       - Geospatial format conversion log (hypertable)
--   9. spatial_indexes          - Spatial index metadata per layer
--  10. operation_metrics        - Per-operation performance metrics (hypertable)
--
-- Continuous Aggregates (2):
--   1. gis_operations_hourly    - Hourly spatial operation aggregates
--   2. gis_conversions_hourly   - Hourly format conversion aggregates
--
-- Also includes: 50+ indexes (B-tree, GIN), RLS policies per tenant,
-- retention policies, compression policies, updated_at trigger,
-- security permissions, and seed data registering GL-DATA-GEO-001
-- in the agent registry.
-- Previous: V035__data_gateway_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS gis_connector_service;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION gis_connector_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: gis_connector_service.geospatial_layers
-- =============================================================================
-- Geospatial layer definitions. Each layer captures a named collection of
-- geographic features with a specific geometry type, coordinate reference
-- system, feature count, spatial extent, and status. Core reference table
-- for feature storage and spatial queries. Tenant-scoped.

CREATE TABLE gis_connector_service.geospatial_layers (
    layer_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    geometry_type VARCHAR(32) NOT NULL,
    crs VARCHAR(32) DEFAULT 'EPSG:4326',
    feature_count INTEGER DEFAULT 0,
    extent JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(32) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Geometry type constraint
ALTER TABLE gis_connector_service.geospatial_layers
    ADD CONSTRAINT chk_gl_geometry_type
    CHECK (geometry_type IN (
        'point', 'line_string', 'polygon',
        'multi_point', 'multi_line_string', 'multi_polygon',
        'geometry_collection'
    ));

-- Status constraint
ALTER TABLE gis_connector_service.geospatial_layers
    ADD CONSTRAINT chk_gl_status
    CHECK (status IN ('active', 'archived', 'processing', 'error', 'deleted'));

-- Layer ID must not be empty
ALTER TABLE gis_connector_service.geospatial_layers
    ADD CONSTRAINT chk_gl_layer_id_not_empty
    CHECK (LENGTH(TRIM(layer_id)) > 0);

-- Name must not be empty
ALTER TABLE gis_connector_service.geospatial_layers
    ADD CONSTRAINT chk_gl_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- CRS must not be empty
ALTER TABLE gis_connector_service.geospatial_layers
    ADD CONSTRAINT chk_gl_crs_not_empty
    CHECK (LENGTH(TRIM(crs)) > 0);

-- Feature count must be non-negative
ALTER TABLE gis_connector_service.geospatial_layers
    ADD CONSTRAINT chk_gl_feature_count_non_negative
    CHECK (feature_count >= 0);

-- Updated_at trigger for geospatial_layers
CREATE TRIGGER trg_geospatial_layers_updated_at
    BEFORE UPDATE ON gis_connector_service.geospatial_layers
    FOR EACH ROW
    EXECUTE FUNCTION gis_connector_service.set_updated_at();

-- =============================================================================
-- Table 2: gis_connector_service.layer_features
-- =============================================================================
-- Individual geographic features belonging to a layer. Each feature captures
-- its geometry type, GeoJSON-style coordinates, arbitrary properties,
-- bounding box, and coordinate reference system. Linked to geospatial_layers
-- via layer_id foreign key. Tenant-scoped.

CREATE TABLE gis_connector_service.layer_features (
    feature_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    layer_id VARCHAR(64) NOT NULL,
    geometry_type VARCHAR(32) NOT NULL,
    coordinates JSONB NOT NULL,
    properties JSONB DEFAULT '{}'::jsonb,
    bounding_box JSONB DEFAULT '{}'::jsonb,
    crs VARCHAR(32) DEFAULT 'EPSG:4326',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to geospatial_layers
ALTER TABLE gis_connector_service.layer_features
    ADD CONSTRAINT fk_lf_layer_id
    FOREIGN KEY (layer_id) REFERENCES gis_connector_service.geospatial_layers(layer_id)
    ON DELETE CASCADE;

-- Geometry type constraint
ALTER TABLE gis_connector_service.layer_features
    ADD CONSTRAINT chk_lf_geometry_type
    CHECK (geometry_type IN (
        'point', 'line_string', 'polygon',
        'multi_point', 'multi_line_string', 'multi_polygon',
        'geometry_collection'
    ));

-- Feature ID must not be empty
ALTER TABLE gis_connector_service.layer_features
    ADD CONSTRAINT chk_lf_feature_id_not_empty
    CHECK (LENGTH(TRIM(feature_id)) > 0);

-- Layer ID must not be empty
ALTER TABLE gis_connector_service.layer_features
    ADD CONSTRAINT chk_lf_layer_id_not_empty
    CHECK (LENGTH(TRIM(layer_id)) > 0);

-- CRS must not be empty
ALTER TABLE gis_connector_service.layer_features
    ADD CONSTRAINT chk_lf_crs_not_empty
    CHECK (LENGTH(TRIM(crs)) > 0);

-- =============================================================================
-- Table 3: gis_connector_service.crs_definitions
-- =============================================================================
-- Coordinate Reference System definitions. Each record stores an EPSG code
-- with its name, PROJ.4 definition string, WKT representation, geographic/
-- projected flags, datum, unit, and spatial bounds. Reference table for
-- CRS lookups and coordinate transformations.

CREATE TABLE gis_connector_service.crs_definitions (
    epsg_code INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    proj4 TEXT DEFAULT '',
    wkt TEXT DEFAULT '',
    is_geographic BOOLEAN DEFAULT FALSE,
    is_projected BOOLEAN DEFAULT FALSE,
    datum VARCHAR(128) DEFAULT '',
    unit VARCHAR(64) DEFAULT '',
    bounds JSONB DEFAULT '{}'::jsonb
);

-- EPSG code must be positive
ALTER TABLE gis_connector_service.crs_definitions
    ADD CONSTRAINT chk_cd_epsg_code_positive
    CHECK (epsg_code > 0);

-- Name must not be empty
ALTER TABLE gis_connector_service.crs_definitions
    ADD CONSTRAINT chk_cd_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- =============================================================================
-- Table 4: gis_connector_service.spatial_operations (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording spatial operations. Each record captures
-- the operation type (intersection, union, buffer, etc.), input/output
-- summaries, execution time, status, error messages, and provenance hash.
-- Partitioned by executed_at for time-series queries. Retained for 730 days
-- with compression after 30 days. Tenant-scoped.

CREATE TABLE gis_connector_service.spatial_operations (
    operation_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    operation_type VARCHAR(32) NOT NULL,
    input_summary JSONB DEFAULT '{}'::jsonb,
    output_summary JSONB DEFAULT '{}'::jsonb,
    execution_time_ms FLOAT DEFAULT 0,
    status VARCHAR(32) DEFAULT 'pending',
    error_message TEXT DEFAULT '',
    provenance_hash VARCHAR(128) DEFAULT '',
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (operation_id, executed_at)
);

-- Create hypertable partitioned by executed_at
SELECT create_hypertable('gis_connector_service.spatial_operations', 'executed_at', if_not_exists => TRUE);

-- Operation type constraint
ALTER TABLE gis_connector_service.spatial_operations
    ADD CONSTRAINT chk_so_operation_type
    CHECK (operation_type IN (
        'intersection', 'union', 'difference', 'buffer',
        'contains', 'within', 'distance', 'area',
        'centroid', 'convex_hull', 'simplify'
    ));

-- Status constraint
ALTER TABLE gis_connector_service.spatial_operations
    ADD CONSTRAINT chk_so_status
    CHECK (status IN ('pending', 'executing', 'completed', 'failed', 'timeout', 'cancelled'));

-- Operation ID must not be empty
ALTER TABLE gis_connector_service.spatial_operations
    ADD CONSTRAINT chk_so_operation_id_not_empty
    CHECK (LENGTH(TRIM(operation_id)) > 0);

-- Execution time must be non-negative
ALTER TABLE gis_connector_service.spatial_operations
    ADD CONSTRAINT chk_so_execution_time_non_negative
    CHECK (execution_time_ms >= 0);

-- =============================================================================
-- Table 5: gis_connector_service.geocoding_cache
-- =============================================================================
-- Cache for forward and reverse geocoding results. Each entry stores a
-- cache key, query string, direction (forward/reverse), result JSONB,
-- confidence score, source provider, hit count, and expiration time.
-- Used for reducing geocoding API calls and improving response times.
-- Tenant-scoped.

CREATE TABLE gis_connector_service.geocoding_cache (
    cache_key VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    query TEXT NOT NULL,
    direction VARCHAR(16) NOT NULL,
    result JSONB DEFAULT '{}'::jsonb,
    confidence FLOAT DEFAULT 0,
    source VARCHAR(128) DEFAULT '',
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Direction constraint
ALTER TABLE gis_connector_service.geocoding_cache
    ADD CONSTRAINT chk_gc_direction
    CHECK (direction IN ('forward', 'reverse'));

-- Cache key must not be empty
ALTER TABLE gis_connector_service.geocoding_cache
    ADD CONSTRAINT chk_gc_cache_key_not_empty
    CHECK (LENGTH(TRIM(cache_key)) > 0);

-- Query must not be empty
ALTER TABLE gis_connector_service.geocoding_cache
    ADD CONSTRAINT chk_gc_query_not_empty
    CHECK (LENGTH(TRIM(query)) > 0);

-- Confidence must be between 0 and 1
ALTER TABLE gis_connector_service.geocoding_cache
    ADD CONSTRAINT chk_gc_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Hit count must be non-negative
ALTER TABLE gis_connector_service.geocoding_cache
    ADD CONSTRAINT chk_gc_hit_count_non_negative
    CHECK (hit_count >= 0);

-- Expires_at must be after created_at
ALTER TABLE gis_connector_service.geocoding_cache
    ADD CONSTRAINT chk_gc_expires_after_created
    CHECK (expires_at >= created_at);

-- =============================================================================
-- Table 6: gis_connector_service.boundary_datasets
-- =============================================================================
-- Administrative and environmental boundary datasets. Each record captures
-- a named boundary with its administrative level (country, state, etc.),
-- ISO code, parent reference, spatial extent, feature count, and metadata.
-- Used for spatial containment queries, jurisdiction lookups, and
-- environmental zone identification. Tenant-scoped.

CREATE TABLE gis_connector_service.boundary_datasets (
    boundary_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    level VARCHAR(32) NOT NULL,
    iso_code VARCHAR(16) DEFAULT '',
    parent_boundary_id VARCHAR(64),
    extent JSONB DEFAULT '{}'::jsonb,
    feature_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Level constraint
ALTER TABLE gis_connector_service.boundary_datasets
    ADD CONSTRAINT chk_bd_level
    CHECK (level IN (
        'country', 'state', 'district', 'municipality',
        'protected_area', 'eez', 'watershed',
        'climate_zone', 'biome', 'custom'
    ));

-- Boundary ID must not be empty
ALTER TABLE gis_connector_service.boundary_datasets
    ADD CONSTRAINT chk_bd_boundary_id_not_empty
    CHECK (LENGTH(TRIM(boundary_id)) > 0);

-- Name must not be empty
ALTER TABLE gis_connector_service.boundary_datasets
    ADD CONSTRAINT chk_bd_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Feature count must be non-negative
ALTER TABLE gis_connector_service.boundary_datasets
    ADD CONSTRAINT chk_bd_feature_count_non_negative
    CHECK (feature_count IS NULL OR feature_count >= 0);

-- =============================================================================
-- Table 7: gis_connector_service.land_cover_data
-- =============================================================================
-- Land cover classification records. Each record captures a geographic
-- location, land cover type (forest, cropland, wetland, etc.), confidence
-- score, CORINE code, source provider, carbon stock estimate, and
-- classification date. Used for environmental impact assessment,
-- carbon stock estimation, and land use change analysis. Tenant-scoped.

CREATE TABLE gis_connector_service.land_cover_data (
    classification_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    location JSONB NOT NULL,
    land_cover_type VARCHAR(32) NOT NULL,
    confidence FLOAT DEFAULT 0,
    corine_code VARCHAR(16) DEFAULT '',
    source VARCHAR(128) DEFAULT '',
    carbon_stock_tonnes_per_ha FLOAT DEFAULT 0,
    classification_date TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Land cover type constraint (13 types)
ALTER TABLE gis_connector_service.land_cover_data
    ADD CONSTRAINT chk_lcd_land_cover_type
    CHECK (land_cover_type IN (
        'forest', 'cropland', 'grassland', 'wetland',
        'urban', 'barren', 'water', 'shrubland',
        'snow_ice', 'mangrove', 'peatland',
        'savanna', 'other'
    ));

-- Classification ID must not be empty
ALTER TABLE gis_connector_service.land_cover_data
    ADD CONSTRAINT chk_lcd_classification_id_not_empty
    CHECK (LENGTH(TRIM(classification_id)) > 0);

-- Confidence must be between 0 and 1
ALTER TABLE gis_connector_service.land_cover_data
    ADD CONSTRAINT chk_lcd_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Carbon stock must be non-negative
ALTER TABLE gis_connector_service.land_cover_data
    ADD CONSTRAINT chk_lcd_carbon_stock_non_negative
    CHECK (carbon_stock_tonnes_per_ha >= 0);

-- =============================================================================
-- Table 8: gis_connector_service.format_conversions (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording geospatial format conversion operations.
-- Each record captures the source and target formats, feature count,
-- output size, execution time, status, and provenance hash.
-- Partitioned by converted_at for time-series queries. Retained for
-- 730 days with compression after 30 days. Tenant-scoped.

CREATE TABLE gis_connector_service.format_conversions (
    conversion_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    source_format VARCHAR(64) NOT NULL,
    target_format VARCHAR(64) NOT NULL,
    feature_count INTEGER DEFAULT 0,
    size_bytes BIGINT DEFAULT 0,
    execution_time_ms FLOAT DEFAULT 0,
    status VARCHAR(32) DEFAULT 'pending',
    provenance_hash VARCHAR(128) DEFAULT '',
    converted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversion_id, converted_at)
);

-- Create hypertable partitioned by converted_at
SELECT create_hypertable('gis_connector_service.format_conversions', 'converted_at', if_not_exists => TRUE);

-- Status constraint
ALTER TABLE gis_connector_service.format_conversions
    ADD CONSTRAINT chk_fc_status
    CHECK (status IN ('pending', 'executing', 'completed', 'failed', 'timeout', 'cancelled'));

-- Conversion ID must not be empty
ALTER TABLE gis_connector_service.format_conversions
    ADD CONSTRAINT chk_fc_conversion_id_not_empty
    CHECK (LENGTH(TRIM(conversion_id)) > 0);

-- Source format must not be empty
ALTER TABLE gis_connector_service.format_conversions
    ADD CONSTRAINT chk_fc_source_format_not_empty
    CHECK (LENGTH(TRIM(source_format)) > 0);

-- Target format must not be empty
ALTER TABLE gis_connector_service.format_conversions
    ADD CONSTRAINT chk_fc_target_format_not_empty
    CHECK (LENGTH(TRIM(target_format)) > 0);

-- Feature count must be non-negative
ALTER TABLE gis_connector_service.format_conversions
    ADD CONSTRAINT chk_fc_feature_count_non_negative
    CHECK (feature_count >= 0);

-- Size bytes must be non-negative
ALTER TABLE gis_connector_service.format_conversions
    ADD CONSTRAINT chk_fc_size_bytes_non_negative
    CHECK (size_bytes >= 0);

-- Execution time must be non-negative
ALTER TABLE gis_connector_service.format_conversions
    ADD CONSTRAINT chk_fc_execution_time_non_negative
    CHECK (execution_time_ms >= 0);

-- =============================================================================
-- Table 9: gis_connector_service.spatial_indexes
-- =============================================================================
-- Spatial index metadata per layer. Each record tracks a spatial index
-- created on a layer with its type (r-tree, quadtree, h3, etc.),
-- feature count, and spatial extent. Used for index management,
-- query optimization, and index rebuild scheduling. Tenant-scoped.

CREATE TABLE gis_connector_service.spatial_indexes (
    index_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    layer_id VARCHAR(64) NOT NULL,
    index_type VARCHAR(32) NOT NULL,
    feature_count INTEGER DEFAULT 0,
    extent JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index ID must not be empty
ALTER TABLE gis_connector_service.spatial_indexes
    ADD CONSTRAINT chk_si_index_id_not_empty
    CHECK (LENGTH(TRIM(index_id)) > 0);

-- Layer ID must not be empty
ALTER TABLE gis_connector_service.spatial_indexes
    ADD CONSTRAINT chk_si_layer_id_not_empty
    CHECK (LENGTH(TRIM(layer_id)) > 0);

-- Index type must not be empty
ALTER TABLE gis_connector_service.spatial_indexes
    ADD CONSTRAINT chk_si_index_type_not_empty
    CHECK (LENGTH(TRIM(index_type)) > 0);

-- Feature count must be non-negative
ALTER TABLE gis_connector_service.spatial_indexes
    ADD CONSTRAINT chk_si_feature_count_non_negative
    CHECK (feature_count >= 0);

-- Updated_at trigger for spatial_indexes
CREATE TRIGGER trg_spatial_indexes_updated_at
    BEFORE UPDATE ON gis_connector_service.spatial_indexes
    FOR EACH ROW
    EXECUTE FUNCTION gis_connector_service.set_updated_at();

-- =============================================================================
-- Table 10: gis_connector_service.operation_metrics (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording per-operation performance metrics.
-- Each metric captures operation type, format, duration, feature count,
-- data volume, and cache hit flag. Partitioned by recorded_at for
-- time-series queries. Retained for 365 days with compression after
-- 14 days. Tenant-scoped.

CREATE TABLE gis_connector_service.operation_metrics (
    operation_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    operation_type VARCHAR(64) NOT NULL,
    format VARCHAR(64) DEFAULT '',
    duration_ms FLOAT DEFAULT 0,
    feature_count INTEGER DEFAULT 0,
    data_volume_bytes BIGINT DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (operation_id, recorded_at)
);

-- Create hypertable partitioned by recorded_at
SELECT create_hypertable('gis_connector_service.operation_metrics', 'recorded_at', if_not_exists => TRUE);

-- Operation ID must not be empty
ALTER TABLE gis_connector_service.operation_metrics
    ADD CONSTRAINT chk_om_operation_id_not_empty
    CHECK (LENGTH(TRIM(operation_id)) > 0);

-- Operation type must not be empty
ALTER TABLE gis_connector_service.operation_metrics
    ADD CONSTRAINT chk_om_operation_type_not_empty
    CHECK (LENGTH(TRIM(operation_type)) > 0);

-- Duration must be non-negative
ALTER TABLE gis_connector_service.operation_metrics
    ADD CONSTRAINT chk_om_duration_non_negative
    CHECK (duration_ms >= 0);

-- Feature count must be non-negative
ALTER TABLE gis_connector_service.operation_metrics
    ADD CONSTRAINT chk_om_feature_count_non_negative
    CHECK (feature_count >= 0);

-- Data volume must be non-negative
ALTER TABLE gis_connector_service.operation_metrics
    ADD CONSTRAINT chk_om_data_volume_non_negative
    CHECK (data_volume_bytes >= 0);

-- =============================================================================
-- Continuous Aggregate: gis_connector_service.gis_operations_hourly
-- =============================================================================
-- Precomputed hourly spatial operation statistics by operation type and tenant
-- for dashboard queries, trend analysis, and SLI tracking.

CREATE MATERIALIZED VIEW gis_connector_service.gis_operations_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', executed_at) AS bucket,
    operation_type,
    tenant_id,
    COUNT(*) AS total_operations,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    MAX(execution_time_ms) AS max_execution_time_ms,
    MIN(execution_time_ms) AS min_execution_time_ms,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
    COUNT(*) FILTER (WHERE status = 'timeout') AS timeout_count
FROM gis_connector_service.spatial_operations
WHERE executed_at IS NOT NULL
GROUP BY bucket, operation_type, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('gis_connector_service.gis_operations_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: gis_connector_service.gis_conversions_hourly
-- =============================================================================
-- Precomputed hourly format conversion statistics by source/target format
-- and tenant for monitoring conversion throughput and failure rates.

CREATE MATERIALIZED VIEW gis_connector_service.gis_conversions_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', converted_at) AS bucket,
    source_format,
    target_format,
    tenant_id,
    COUNT(*) AS total_conversions,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    MAX(execution_time_ms) AS max_execution_time_ms,
    SUM(feature_count) AS total_features,
    SUM(size_bytes) AS total_size_bytes,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_count
FROM gis_connector_service.format_conversions
WHERE converted_at IS NOT NULL
GROUP BY bucket, source_format, target_format, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('gis_connector_service.gis_conversions_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- geospatial_layers indexes
CREATE INDEX idx_gl_name ON gis_connector_service.geospatial_layers(name);
CREATE INDEX idx_gl_geometry_type ON gis_connector_service.geospatial_layers(geometry_type);
CREATE INDEX idx_gl_crs ON gis_connector_service.geospatial_layers(crs);
CREATE INDEX idx_gl_status ON gis_connector_service.geospatial_layers(status);
CREATE INDEX idx_gl_tenant ON gis_connector_service.geospatial_layers(tenant_id);
CREATE INDEX idx_gl_feature_count ON gis_connector_service.geospatial_layers(feature_count DESC);
CREATE INDEX idx_gl_created_at ON gis_connector_service.geospatial_layers(created_at DESC);
CREATE INDEX idx_gl_updated_at ON gis_connector_service.geospatial_layers(updated_at DESC);
CREATE INDEX idx_gl_tenant_status ON gis_connector_service.geospatial_layers(tenant_id, status);
CREATE INDEX idx_gl_tenant_type ON gis_connector_service.geospatial_layers(tenant_id, geometry_type);
CREATE INDEX idx_gl_tenant_name ON gis_connector_service.geospatial_layers(tenant_id, name);
CREATE INDEX idx_gl_extent ON gis_connector_service.geospatial_layers USING GIN (extent);
CREATE INDEX idx_gl_metadata ON gis_connector_service.geospatial_layers USING GIN (metadata);

-- layer_features indexes
CREATE INDEX idx_lf_layer_id ON gis_connector_service.layer_features(layer_id);
CREATE INDEX idx_lf_geometry_type ON gis_connector_service.layer_features(geometry_type);
CREATE INDEX idx_lf_crs ON gis_connector_service.layer_features(crs);
CREATE INDEX idx_lf_tenant ON gis_connector_service.layer_features(tenant_id);
CREATE INDEX idx_lf_created_at ON gis_connector_service.layer_features(created_at DESC);
CREATE INDEX idx_lf_tenant_layer ON gis_connector_service.layer_features(tenant_id, layer_id);
CREATE INDEX idx_lf_tenant_type ON gis_connector_service.layer_features(tenant_id, geometry_type);
CREATE INDEX idx_lf_coordinates ON gis_connector_service.layer_features USING GIN (coordinates);
CREATE INDEX idx_lf_properties ON gis_connector_service.layer_features USING GIN (properties);
CREATE INDEX idx_lf_bounding_box ON gis_connector_service.layer_features USING GIN (bounding_box);

-- crs_definitions indexes
CREATE INDEX idx_cd_name ON gis_connector_service.crs_definitions(name);
CREATE INDEX idx_cd_is_geographic ON gis_connector_service.crs_definitions(is_geographic);
CREATE INDEX idx_cd_is_projected ON gis_connector_service.crs_definitions(is_projected);
CREATE INDEX idx_cd_datum ON gis_connector_service.crs_definitions(datum);
CREATE INDEX idx_cd_unit ON gis_connector_service.crs_definitions(unit);
CREATE INDEX idx_cd_bounds ON gis_connector_service.crs_definitions USING GIN (bounds);

-- spatial_operations indexes (hypertable-aware)
CREATE INDEX idx_so_operation_type ON gis_connector_service.spatial_operations(operation_type, executed_at DESC);
CREATE INDEX idx_so_status ON gis_connector_service.spatial_operations(status, executed_at DESC);
CREATE INDEX idx_so_tenant ON gis_connector_service.spatial_operations(tenant_id, executed_at DESC);
CREATE INDEX idx_so_execution_time ON gis_connector_service.spatial_operations(execution_time_ms DESC, executed_at DESC);
CREATE INDEX idx_so_tenant_type ON gis_connector_service.spatial_operations(tenant_id, operation_type, executed_at DESC);
CREATE INDEX idx_so_tenant_status ON gis_connector_service.spatial_operations(tenant_id, status, executed_at DESC);
CREATE INDEX idx_so_provenance ON gis_connector_service.spatial_operations(provenance_hash, executed_at DESC);
CREATE INDEX idx_so_input_summary ON gis_connector_service.spatial_operations USING GIN (input_summary);
CREATE INDEX idx_so_output_summary ON gis_connector_service.spatial_operations USING GIN (output_summary);

-- geocoding_cache indexes
CREATE INDEX idx_gc_direction ON gis_connector_service.geocoding_cache(direction);
CREATE INDEX idx_gc_confidence ON gis_connector_service.geocoding_cache(confidence DESC);
CREATE INDEX idx_gc_source ON gis_connector_service.geocoding_cache(source);
CREATE INDEX idx_gc_hit_count ON gis_connector_service.geocoding_cache(hit_count DESC);
CREATE INDEX idx_gc_tenant ON gis_connector_service.geocoding_cache(tenant_id);
CREATE INDEX idx_gc_created_at ON gis_connector_service.geocoding_cache(created_at DESC);
CREATE INDEX idx_gc_expires_at ON gis_connector_service.geocoding_cache(expires_at);
CREATE INDEX idx_gc_tenant_direction ON gis_connector_service.geocoding_cache(tenant_id, direction);
CREATE INDEX idx_gc_tenant_source ON gis_connector_service.geocoding_cache(tenant_id, source);
CREATE INDEX idx_gc_result ON gis_connector_service.geocoding_cache USING GIN (result);

-- boundary_datasets indexes
CREATE INDEX idx_bd_name ON gis_connector_service.boundary_datasets(name);
CREATE INDEX idx_bd_level ON gis_connector_service.boundary_datasets(level);
CREATE INDEX idx_bd_iso_code ON gis_connector_service.boundary_datasets(iso_code);
CREATE INDEX idx_bd_parent ON gis_connector_service.boundary_datasets(parent_boundary_id);
CREATE INDEX idx_bd_tenant ON gis_connector_service.boundary_datasets(tenant_id);
CREATE INDEX idx_bd_created_at ON gis_connector_service.boundary_datasets(created_at DESC);
CREATE INDEX idx_bd_tenant_level ON gis_connector_service.boundary_datasets(tenant_id, level);
CREATE INDEX idx_bd_tenant_iso ON gis_connector_service.boundary_datasets(tenant_id, iso_code);
CREATE INDEX idx_bd_tenant_name ON gis_connector_service.boundary_datasets(tenant_id, name);
CREATE INDEX idx_bd_extent ON gis_connector_service.boundary_datasets USING GIN (extent);
CREATE INDEX idx_bd_metadata ON gis_connector_service.boundary_datasets USING GIN (metadata);

-- land_cover_data indexes
CREATE INDEX idx_lcd_land_cover_type ON gis_connector_service.land_cover_data(land_cover_type);
CREATE INDEX idx_lcd_confidence ON gis_connector_service.land_cover_data(confidence DESC);
CREATE INDEX idx_lcd_corine_code ON gis_connector_service.land_cover_data(corine_code);
CREATE INDEX idx_lcd_source ON gis_connector_service.land_cover_data(source);
CREATE INDEX idx_lcd_carbon_stock ON gis_connector_service.land_cover_data(carbon_stock_tonnes_per_ha DESC);
CREATE INDEX idx_lcd_classification_date ON gis_connector_service.land_cover_data(classification_date DESC);
CREATE INDEX idx_lcd_tenant ON gis_connector_service.land_cover_data(tenant_id);
CREATE INDEX idx_lcd_created_at ON gis_connector_service.land_cover_data(created_at DESC);
CREATE INDEX idx_lcd_tenant_type ON gis_connector_service.land_cover_data(tenant_id, land_cover_type);
CREATE INDEX idx_lcd_tenant_source ON gis_connector_service.land_cover_data(tenant_id, source);
CREATE INDEX idx_lcd_location ON gis_connector_service.land_cover_data USING GIN (location);

-- format_conversions indexes (hypertable-aware)
CREATE INDEX idx_fc_source_format ON gis_connector_service.format_conversions(source_format, converted_at DESC);
CREATE INDEX idx_fc_target_format ON gis_connector_service.format_conversions(target_format, converted_at DESC);
CREATE INDEX idx_fc_status ON gis_connector_service.format_conversions(status, converted_at DESC);
CREATE INDEX idx_fc_tenant ON gis_connector_service.format_conversions(tenant_id, converted_at DESC);
CREATE INDEX idx_fc_execution_time ON gis_connector_service.format_conversions(execution_time_ms DESC, converted_at DESC);
CREATE INDEX idx_fc_size ON gis_connector_service.format_conversions(size_bytes DESC, converted_at DESC);
CREATE INDEX idx_fc_tenant_source ON gis_connector_service.format_conversions(tenant_id, source_format, converted_at DESC);
CREATE INDEX idx_fc_tenant_target ON gis_connector_service.format_conversions(tenant_id, target_format, converted_at DESC);
CREATE INDEX idx_fc_tenant_status ON gis_connector_service.format_conversions(tenant_id, status, converted_at DESC);
CREATE INDEX idx_fc_provenance ON gis_connector_service.format_conversions(provenance_hash, converted_at DESC);

-- spatial_indexes indexes
CREATE INDEX idx_si_layer_id ON gis_connector_service.spatial_indexes(layer_id);
CREATE INDEX idx_si_index_type ON gis_connector_service.spatial_indexes(index_type);
CREATE INDEX idx_si_tenant ON gis_connector_service.spatial_indexes(tenant_id);
CREATE INDEX idx_si_feature_count ON gis_connector_service.spatial_indexes(feature_count DESC);
CREATE INDEX idx_si_created_at ON gis_connector_service.spatial_indexes(created_at DESC);
CREATE INDEX idx_si_updated_at ON gis_connector_service.spatial_indexes(updated_at DESC);
CREATE INDEX idx_si_tenant_layer ON gis_connector_service.spatial_indexes(tenant_id, layer_id);
CREATE INDEX idx_si_tenant_type ON gis_connector_service.spatial_indexes(tenant_id, index_type);
CREATE INDEX idx_si_extent ON gis_connector_service.spatial_indexes USING GIN (extent);

-- operation_metrics indexes (hypertable-aware)
CREATE INDEX idx_om_operation_type ON gis_connector_service.operation_metrics(operation_type, recorded_at DESC);
CREATE INDEX idx_om_format ON gis_connector_service.operation_metrics(format, recorded_at DESC);
CREATE INDEX idx_om_tenant ON gis_connector_service.operation_metrics(tenant_id, recorded_at DESC);
CREATE INDEX idx_om_duration ON gis_connector_service.operation_metrics(duration_ms DESC, recorded_at DESC);
CREATE INDEX idx_om_feature_count ON gis_connector_service.operation_metrics(feature_count DESC, recorded_at DESC);
CREATE INDEX idx_om_data_volume ON gis_connector_service.operation_metrics(data_volume_bytes DESC, recorded_at DESC);
CREATE INDEX idx_om_cache_hit ON gis_connector_service.operation_metrics(cache_hit, recorded_at DESC);
CREATE INDEX idx_om_tenant_type ON gis_connector_service.operation_metrics(tenant_id, operation_type, recorded_at DESC);
CREATE INDEX idx_om_tenant_format ON gis_connector_service.operation_metrics(tenant_id, format, recorded_at DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE gis_connector_service.geospatial_layers ENABLE ROW LEVEL SECURITY;
CREATE POLICY gl_tenant_read ON gis_connector_service.geospatial_layers
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY gl_tenant_write ON gis_connector_service.geospatial_layers
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.layer_features ENABLE ROW LEVEL SECURITY;
CREATE POLICY lf_tenant_read ON gis_connector_service.layer_features
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY lf_tenant_write ON gis_connector_service.layer_features
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.crs_definitions ENABLE ROW LEVEL SECURITY;
CREATE POLICY cd_tenant_read ON gis_connector_service.crs_definitions
    FOR SELECT USING (TRUE);
CREATE POLICY cd_tenant_write ON gis_connector_service.crs_definitions
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
    );

ALTER TABLE gis_connector_service.spatial_operations ENABLE ROW LEVEL SECURITY;
CREATE POLICY so_tenant_read ON gis_connector_service.spatial_operations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY so_tenant_write ON gis_connector_service.spatial_operations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.geocoding_cache ENABLE ROW LEVEL SECURITY;
CREATE POLICY gc_tenant_read ON gis_connector_service.geocoding_cache
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY gc_tenant_write ON gis_connector_service.geocoding_cache
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.boundary_datasets ENABLE ROW LEVEL SECURITY;
CREATE POLICY bd_tenant_read ON gis_connector_service.boundary_datasets
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY bd_tenant_write ON gis_connector_service.boundary_datasets
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.land_cover_data ENABLE ROW LEVEL SECURITY;
CREATE POLICY lcd_tenant_read ON gis_connector_service.land_cover_data
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY lcd_tenant_write ON gis_connector_service.land_cover_data
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.format_conversions ENABLE ROW LEVEL SECURITY;
CREATE POLICY fc_tenant_read ON gis_connector_service.format_conversions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY fc_tenant_write ON gis_connector_service.format_conversions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.spatial_indexes ENABLE ROW LEVEL SECURITY;
CREATE POLICY si_tenant_read ON gis_connector_service.spatial_indexes
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY si_tenant_write ON gis_connector_service.spatial_indexes
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE gis_connector_service.operation_metrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY om_tenant_read ON gis_connector_service.operation_metrics
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY om_tenant_write ON gis_connector_service.operation_metrics
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA gis_connector_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA gis_connector_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA gis_connector_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON gis_connector_service.gis_operations_hourly TO greenlang_app;
GRANT SELECT ON gis_connector_service.gis_conversions_hourly TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA gis_connector_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA gis_connector_service TO greenlang_readonly;
GRANT SELECT ON gis_connector_service.gis_operations_hourly TO greenlang_readonly;
GRANT SELECT ON gis_connector_service.gis_conversions_hourly TO greenlang_readonly;

-- Add GIS connector service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'gis_connector:layers:read', 'gis_connector', 'layers_read', 'View geospatial layers and feature counts'),
    (gen_random_uuid(), 'gis_connector:layers:write', 'gis_connector', 'layers_write', 'Create, update, and delete geospatial layers'),
    (gen_random_uuid(), 'gis_connector:features:read', 'gis_connector', 'features_read', 'View layer features and coordinates'),
    (gen_random_uuid(), 'gis_connector:features:write', 'gis_connector', 'features_write', 'Create and manage layer features'),
    (gen_random_uuid(), 'gis_connector:crs:read', 'gis_connector', 'crs_read', 'View CRS definitions and projections'),
    (gen_random_uuid(), 'gis_connector:crs:write', 'gis_connector', 'crs_write', 'Create and manage CRS definitions'),
    (gen_random_uuid(), 'gis_connector:operations:read', 'gis_connector', 'operations_read', 'View spatial operation history and results'),
    (gen_random_uuid(), 'gis_connector:operations:write', 'gis_connector', 'operations_write', 'Execute spatial operations (intersection, union, buffer, etc.)'),
    (gen_random_uuid(), 'gis_connector:geocoding:read', 'gis_connector', 'geocoding_read', 'View geocoding cache entries and statistics'),
    (gen_random_uuid(), 'gis_connector:geocoding:write', 'gis_connector', 'geocoding_write', 'Execute geocoding queries and manage cache'),
    (gen_random_uuid(), 'gis_connector:boundaries:read', 'gis_connector', 'boundaries_read', 'View boundary datasets and administrative levels'),
    (gen_random_uuid(), 'gis_connector:boundaries:write', 'gis_connector', 'boundaries_write', 'Create and manage boundary datasets'),
    (gen_random_uuid(), 'gis_connector:landcover:read', 'gis_connector', 'landcover_read', 'View land cover classifications and carbon stock data'),
    (gen_random_uuid(), 'gis_connector:landcover:write', 'gis_connector', 'landcover_write', 'Create and manage land cover classifications'),
    (gen_random_uuid(), 'gis_connector:conversions:read', 'gis_connector', 'conversions_read', 'View format conversion history and metrics'),
    (gen_random_uuid(), 'gis_connector:conversions:write', 'gis_connector', 'conversions_write', 'Execute geospatial format conversions'),
    (gen_random_uuid(), 'gis_connector:admin', 'gis_connector', 'admin', 'GIS connector service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep spatial operations records for 730 days (2 years)
SELECT add_retention_policy('gis_connector_service.spatial_operations', INTERVAL '730 days');

-- Keep format conversions records for 730 days (2 years)
SELECT add_retention_policy('gis_connector_service.format_conversions', INTERVAL '730 days');

-- Keep operation metrics records for 365 days (1 year)
SELECT add_retention_policy('gis_connector_service.operation_metrics', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on spatial_operations after 30 days
ALTER TABLE gis_connector_service.spatial_operations SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'executed_at DESC'
);

SELECT add_compression_policy('gis_connector_service.spatial_operations', INTERVAL '30 days');

-- Enable compression on format_conversions after 30 days
ALTER TABLE gis_connector_service.format_conversions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'converted_at DESC'
);

SELECT add_compression_policy('gis_connector_service.format_conversions', INTERVAL '30 days');

-- Enable compression on operation_metrics after 14 days
ALTER TABLE gis_connector_service.operation_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'recorded_at DESC'
);

SELECT add_compression_policy('gis_connector_service.operation_metrics', INTERVAL '14 days');

-- =============================================================================
-- Seed: Register the GIS/Mapping Connector Agent (GL-DATA-GEO-001)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-GEO-001', 'GIS/Mapping Connector Agent',
 'Geospatial data connector for GreenLang Climate OS. Provides geospatial layer management with multi-geometry support (point, polygon, multi-polygon, geometry collection), CRS definitions and coordinate transformations (EPSG registry), 11 spatial operations (intersection, union, buffer, distance, area, centroid, convex hull, simplify), forward/reverse geocoding with caching, administrative and environmental boundary datasets (country to biome level), land cover classification with carbon stock estimation (13 cover types, CORINE codes), multi-format conversion (GeoJSON, Shapefile, KML, GeoPackage, WKT, WKB, TopoJSON, GML), spatial indexing (R-tree, Quadtree, H3), and comprehensive operation metrics tracking.',
 2, 'async', true, true, 10, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/gis-connector', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for GIS/Mapping Connector Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-GEO-001', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/gis-connector-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"geospatial", "gis", "mapping", "spatial-operations", "geocoding", "boundaries", "land-cover", "format-conversion", "crs"}',
 '{"cross-sector", "agriculture", "forestry", "energy", "logistics", "real-estate", "environmental"}',
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for GIS/Mapping Connector Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-GEO-001', '1.0.0', 'layer_management', 'data_management',
 'Create, manage, and query geospatial layers with multi-geometry type support, CRS configuration, feature count tracking, and spatial extent calculation',
 '{"name", "geometry_type", "crs", "features"}', '{"layer_id", "feature_count", "extent", "status"}',
 '{"supported_geometry_types": ["point", "line_string", "polygon", "multi_point", "multi_line_string", "multi_polygon", "geometry_collection"], "default_crs": "EPSG:4326", "max_features_per_layer": 1000000}'::jsonb),

('GL-DATA-GEO-001', '1.0.0', 'spatial_operations', 'analysis',
 'Execute spatial operations including intersection, union, difference, buffer, containment, distance, area, centroid, convex hull, and simplification with provenance tracking',
 '{"operation_type", "geometries", "parameters"}', '{"operation_id", "result", "execution_time_ms", "provenance_hash"}',
 '{"supported_operations": ["intersection", "union", "difference", "buffer", "contains", "within", "distance", "area", "centroid", "convex_hull", "simplify"], "buffer_units": ["meters", "kilometers", "miles"], "simplify_tolerance_default": 0.001}'::jsonb),

('GL-DATA-GEO-001', '1.0.0', 'geocoding', 'transformation',
 'Forward and reverse geocoding with result caching, confidence scoring, and multi-provider support for address-to-coordinate and coordinate-to-address resolution',
 '{"query", "direction", "options"}', '{"result", "confidence", "source", "cache_hit"}',
 '{"providers": ["nominatim", "google", "mapbox", "here", "opencage"], "cache_ttl_seconds": 86400, "max_results": 10, "language": "en"}'::jsonb),

('GL-DATA-GEO-001', '1.0.0', 'boundary_lookup', 'query',
 'Query administrative and environmental boundary datasets for spatial containment, jurisdiction identification, and zone classification across 10 boundary levels',
 '{"location", "level", "iso_code"}', '{"boundary_id", "name", "level", "extent", "features"}',
 '{"supported_levels": ["country", "state", "district", "municipality", "protected_area", "eez", "watershed", "climate_zone", "biome", "custom"], "point_in_polygon_enabled": true, "hierarchical_lookup": true}'::jsonb),

('GL-DATA-GEO-001', '1.0.0', 'land_cover_classification', 'analysis',
 'Classify land cover types with CORINE code mapping, confidence scoring, carbon stock estimation per hectare, and temporal classification tracking for land use change analysis',
 '{"location", "date_range", "source"}', '{"classification_id", "land_cover_type", "confidence", "carbon_stock_tonnes_per_ha"}',
 '{"supported_types": ["forest", "cropland", "grassland", "wetland", "urban", "barren", "water", "shrubland", "snow_ice", "mangrove", "peatland", "savanna", "other"], "corine_mapping": true, "carbon_stock_estimation": true}'::jsonb),

('GL-DATA-GEO-001', '1.0.0', 'format_conversion', 'transformation',
 'Convert between geospatial data formats with feature count preservation, size tracking, and provenance hash verification for data integrity',
 '{"source_format", "target_format", "data"}', '{"conversion_id", "feature_count", "size_bytes", "provenance_hash"}',
 '{"supported_formats": ["geojson", "shapefile", "kml", "kmz", "geopackage", "wkt", "wkb", "topojson", "gml", "csv_latlon", "geotiff"], "max_file_size_mb": 500, "preserve_properties": true}'::jsonb),

('GL-DATA-GEO-001', '1.0.0', 'crs_transformation', 'transformation',
 'Transform coordinates between coordinate reference systems using EPSG registry with PROJ.4 and WKT definitions, supporting geographic and projected CRS types',
 '{"coordinates", "source_crs", "target_crs"}', '{"transformed_coordinates", "source_epsg", "target_epsg"}',
 '{"epsg_registry_count": 6000, "supported_datum_transforms": true, "batch_transform_enabled": true, "precision_digits": 8}'::jsonb),

('GL-DATA-GEO-001', '1.0.0', 'spatial_indexing', 'performance',
 'Create and manage spatial indexes on geospatial layers for optimized spatial queries with support for R-tree, Quadtree, and H3 hexagonal indexing',
 '{"layer_id", "index_type"}', '{"index_id", "feature_count", "extent", "build_time_ms"}',
 '{"supported_index_types": ["r-tree", "quadtree", "h3", "geohash", "s2"], "auto_rebuild_on_update": true, "h3_resolution_default": 7}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for GIS/Mapping Connector Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- GIS Connector depends on Schema Compiler for GeoJSON/feature validation
('GL-DATA-GEO-001', 'GL-FOUND-X-002', '>=1.0.0', false,
 'GeoJSON feature schemas, layer definitions, and spatial operation inputs are validated against JSON Schema definitions'),

-- GIS Connector depends on Registry for agent discovery
('GL-DATA-GEO-001', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for GIS connector registration and health monitoring'),

-- GIS Connector depends on Access Guard for policy enforcement
('GL-DATA-GEO-001', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for geospatial data and cross-tenant isolation'),

-- GIS Connector depends on Observability Agent for metrics
('GL-DATA-GEO-001', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Spatial operation metrics, geocoding statistics, and format conversion telemetry are reported to observability'),

-- GIS Connector depends on Unit Normalizer for coordinate/distance units
('GL-DATA-GEO-001', 'GL-FOUND-X-003', '>=1.0.0', false,
 'Distance, area, and buffer units are normalized through the unit normalizer for consistent calculations'),

-- GIS Connector optionally uses Citations for provenance tracking
('GL-DATA-GEO-001', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Spatial operation results and format conversion outputs are registered with the citation service for audit trail'),

-- GIS Connector optionally uses Reproducibility for determinism
('GL-DATA-GEO-001', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Spatial operation results and format conversions are verified for reproducibility across re-execution'),

-- GIS Connector optionally integrates with Data Gateway
('GL-DATA-GEO-001', 'GL-DATA-GW-001', '>=1.0.0', true,
 'GIS connector can be accessed through the unified data gateway for cross-source geospatial queries')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for GIS/Mapping Connector Agent
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-GEO-001', 'GIS/Mapping Connector Agent',
 'Geospatial data connector for GreenLang Climate OS. Manages geospatial layers with multi-geometry support, CRS definitions and transformations, 11 spatial operations (intersection, union, buffer, etc.), forward/reverse geocoding with caching, administrative and environmental boundary datasets, land cover classification with carbon stock estimation, multi-format conversion (GeoJSON, Shapefile, KML, GeoPackage, etc.), spatial indexing, and comprehensive operation metrics.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA gis_connector_service IS 'GIS/Mapping Connector Agent for GreenLang Climate OS (AGENT-DATA-006) - Geospatial data connector with layer management, spatial operations, geocoding, boundary datasets, land cover classification, format conversion, CRS transformations, and spatial indexing';
COMMENT ON TABLE gis_connector_service.geospatial_layers IS 'Geospatial layer definitions with geometry type (point/polygon/multi-polygon/etc.), coordinate reference system, feature count, spatial extent, and lifecycle status';
COMMENT ON TABLE gis_connector_service.layer_features IS 'Individual geographic features per layer with GeoJSON-style coordinates, arbitrary properties, bounding box, and CRS reference';
COMMENT ON TABLE gis_connector_service.crs_definitions IS 'Coordinate Reference System definitions with EPSG codes, PROJ.4 strings, WKT representations, geographic/projected flags, datum, unit, and spatial bounds';
COMMENT ON TABLE gis_connector_service.spatial_operations IS 'TimescaleDB hypertable: spatial operation audit log with operation type (intersection/union/buffer/etc.), input/output summaries, execution time, status, and provenance hash';
COMMENT ON TABLE gis_connector_service.geocoding_cache IS 'Forward and reverse geocoding result cache with query, direction, result JSONB, confidence score, source provider, hit count, and expiration';
COMMENT ON TABLE gis_connector_service.boundary_datasets IS 'Administrative and environmental boundary datasets with level hierarchy (country/state/district/municipality/protected_area/eez/watershed/climate_zone/biome/custom), ISO codes, and spatial extent';
COMMENT ON TABLE gis_connector_service.land_cover_data IS 'Land cover classification records with location, cover type (13 types), confidence score, CORINE code, source, carbon stock tonnes per hectare, and classification date';
COMMENT ON TABLE gis_connector_service.format_conversions IS 'TimescaleDB hypertable: geospatial format conversion log with source/target format, feature count, output size, execution time, status, and provenance hash';
COMMENT ON TABLE gis_connector_service.spatial_indexes IS 'Spatial index metadata per layer with index type (R-tree/Quadtree/H3/etc.), feature count, and spatial extent for query optimization';
COMMENT ON TABLE gis_connector_service.operation_metrics IS 'TimescaleDB hypertable: per-operation performance metrics with operation type, format, duration, feature count, data volume, and cache hit flag';
COMMENT ON MATERIALIZED VIEW gis_connector_service.gis_operations_hourly IS 'Continuous aggregate: hourly spatial operation statistics by operation type with count, avg/max/min execution time, completed/failed/timeout counts for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW gis_connector_service.gis_conversions_hourly IS 'Continuous aggregate: hourly format conversion statistics by source/target format with count, avg/max execution time, total features/bytes, completed/failed counts for throughput monitoring';

COMMENT ON COLUMN gis_connector_service.geospatial_layers.geometry_type IS 'Geometry type: point, line_string, polygon, multi_point, multi_line_string, multi_polygon, geometry_collection';
COMMENT ON COLUMN gis_connector_service.geospatial_layers.crs IS 'Coordinate Reference System identifier (default EPSG:4326 / WGS 84)';
COMMENT ON COLUMN gis_connector_service.geospatial_layers.extent IS 'JSONB spatial extent bounding box [minX, minY, maxX, maxY] for the layer';
COMMENT ON COLUMN gis_connector_service.geospatial_layers.status IS 'Layer lifecycle status: active, archived, processing, error, deleted';
COMMENT ON COLUMN gis_connector_service.layer_features.coordinates IS 'GeoJSON-style coordinate array for the feature geometry';
COMMENT ON COLUMN gis_connector_service.layer_features.properties IS 'JSONB arbitrary properties/attributes for the feature';
COMMENT ON COLUMN gis_connector_service.layer_features.bounding_box IS 'JSONB bounding box [minX, minY, maxX, maxY] for the feature';
COMMENT ON COLUMN gis_connector_service.crs_definitions.epsg_code IS 'EPSG numeric code identifying the coordinate reference system';
COMMENT ON COLUMN gis_connector_service.crs_definitions.proj4 IS 'PROJ.4 definition string for coordinate transformation';
COMMENT ON COLUMN gis_connector_service.crs_definitions.wkt IS 'Well-Known Text (WKT) representation of the CRS';
COMMENT ON COLUMN gis_connector_service.crs_definitions.is_geographic IS 'TRUE if CRS uses angular coordinates (latitude/longitude)';
COMMENT ON COLUMN gis_connector_service.crs_definitions.is_projected IS 'TRUE if CRS uses projected (planar) coordinates';
COMMENT ON COLUMN gis_connector_service.spatial_operations.operation_type IS 'Spatial operation type: intersection, union, difference, buffer, contains, within, distance, area, centroid, convex_hull, simplify';
COMMENT ON COLUMN gis_connector_service.spatial_operations.provenance_hash IS 'SHA-256 provenance hash of operation inputs and results for reproducibility verification';
COMMENT ON COLUMN gis_connector_service.geocoding_cache.direction IS 'Geocoding direction: forward (address to coordinates) or reverse (coordinates to address)';
COMMENT ON COLUMN gis_connector_service.geocoding_cache.confidence IS 'Geocoding result confidence score between 0.0 and 1.0';
COMMENT ON COLUMN gis_connector_service.geocoding_cache.expires_at IS 'Cache entry expiration timestamp for TTL-based eviction';
COMMENT ON COLUMN gis_connector_service.boundary_datasets.level IS 'Administrative/environmental boundary level: country, state, district, municipality, protected_area, eez, watershed, climate_zone, biome, custom';
COMMENT ON COLUMN gis_connector_service.boundary_datasets.iso_code IS 'ISO 3166 country/subdivision code for administrative boundaries';
COMMENT ON COLUMN gis_connector_service.boundary_datasets.parent_boundary_id IS 'Reference to parent boundary for hierarchical containment queries';
COMMENT ON COLUMN gis_connector_service.land_cover_data.land_cover_type IS 'Land cover classification: forest, cropland, grassland, wetland, urban, barren, water, shrubland, snow_ice, mangrove, peatland, savanna, other';
COMMENT ON COLUMN gis_connector_service.land_cover_data.corine_code IS 'CORINE Land Cover classification code for EU-standard land use reporting';
COMMENT ON COLUMN gis_connector_service.land_cover_data.carbon_stock_tonnes_per_ha IS 'Estimated carbon stock in tonnes per hectare for carbon accounting';
COMMENT ON COLUMN gis_connector_service.format_conversions.source_format IS 'Input geospatial format (e.g., geojson, shapefile, kml, geopackage)';
COMMENT ON COLUMN gis_connector_service.format_conversions.target_format IS 'Output geospatial format (e.g., geojson, shapefile, kml, geopackage)';
COMMENT ON COLUMN gis_connector_service.format_conversions.provenance_hash IS 'SHA-256 provenance hash of conversion inputs and outputs for data integrity verification';
COMMENT ON COLUMN gis_connector_service.spatial_indexes.index_type IS 'Spatial index type (e.g., r-tree, quadtree, h3, geohash, s2)';
COMMENT ON COLUMN gis_connector_service.operation_metrics.data_volume_bytes IS 'Total data volume processed in bytes for throughput tracking';
COMMENT ON COLUMN gis_connector_service.operation_metrics.cache_hit IS 'Whether the operation result was served from cache';
