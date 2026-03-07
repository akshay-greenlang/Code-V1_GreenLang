-- ============================================================================
-- V094: AGENT-EUDR-006 Plot Boundary Manager
-- ============================================================================
-- Agent: Plot Boundary Manager (AGENT-EUDR-006)
-- Regulation: EU Deforestation Regulation (EU) 2023/1115
-- Purpose: Plot boundary lifecycle, validation, versioning, overlap detection
-- Prefix: gl_eudr_pbm_
-- Tables: 10 (5 hypertables, 2 continuous aggregates)
-- Retention: 5 years per EUDR Article 31
-- ============================================================================

-- Enable required extensions (idempotent)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Table 1: gl_eudr_pbm_boundaries - Plot boundary records
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_boundaries (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    plot_id             VARCHAR(255) NOT NULL,
    geometry_type       VARCHAR(50) NOT NULL DEFAULT 'POLYGON',
    exterior_wkb        BYTEA NOT NULL,
    holes_wkb           BYTEA[],
    crs                 VARCHAR(50) NOT NULL DEFAULT 'EPSG:4326',
    source_crs          VARCHAR(50),
    centroid_lat        DOUBLE PRECISION,
    centroid_lon        DOUBLE PRECISION,
    bbox_min_lat        DOUBLE PRECISION,
    bbox_min_lon        DOUBLE PRECISION,
    bbox_max_lat        DOUBLE PRECISION,
    bbox_max_lon        DOUBLE PRECISION,
    area_m2             DOUBLE PRECISION,
    area_hectares       DOUBLE PRECISION,
    perimeter_m         DOUBLE PRECISION,
    vertex_count        INTEGER,
    commodity           VARCHAR(50),
    country_iso         VARCHAR(10),
    owner_id            VARCHAR(255),
    certification_id    VARCHAR(255),
    version             INTEGER NOT NULL DEFAULT 1,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    metadata            JSONB DEFAULT '{}',
    provenance_hash     VARCHAR(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(255),
    tenant_id           VARCHAR(255)
);

SELECT create_hypertable('gl_eudr_pbm_boundaries', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Table 2: gl_eudr_pbm_versions - Immutable version history
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_versions (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    plot_id             VARCHAR(255) NOT NULL,
    version_number      INTEGER NOT NULL,
    exterior_wkb        BYTEA NOT NULL,
    holes_wkb           BYTEA[],
    area_m2             DOUBLE PRECISION,
    area_hectares       DOUBLE PRECISION,
    area_diff_m2        DOUBLE PRECISION DEFAULT 0,
    change_reason       VARCHAR(50) NOT NULL,
    changed_by          VARCHAR(255),
    previous_version    INTEGER,
    provenance_hash     VARCHAR(128),
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id           VARCHAR(255)
);

SELECT create_hypertable('gl_eudr_pbm_versions', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Table 3: gl_eudr_pbm_validations - Topology validation results
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_validations (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    plot_id             VARCHAR(255),
    is_valid            BOOLEAN NOT NULL,
    ogc_compliant       BOOLEAN NOT NULL DEFAULT FALSE,
    error_count         INTEGER NOT NULL DEFAULT 0,
    warning_count       INTEGER NOT NULL DEFAULT 0,
    errors              JSONB DEFAULT '[]',
    warnings            JSONB DEFAULT '[]',
    repaired            BOOLEAN NOT NULL DEFAULT FALSE,
    repair_actions      JSONB DEFAULT '[]',
    confidence_score    DOUBLE PRECISION,
    vertex_count_before INTEGER,
    vertex_count_after  INTEGER,
    duration_ms         DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(255),
    tenant_id           VARCHAR(255)
);

SELECT create_hypertable('gl_eudr_pbm_validations', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Table 4: gl_eudr_pbm_overlaps - Overlap detection results
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_overlaps (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    plot_id_a           VARCHAR(255) NOT NULL,
    plot_id_b           VARCHAR(255) NOT NULL,
    overlap_area_m2     DOUBLE PRECISION NOT NULL,
    overlap_pct_a       DOUBLE PRECISION NOT NULL,
    overlap_pct_b       DOUBLE PRECISION NOT NULL,
    severity            VARCHAR(20) NOT NULL,
    intersection_wkb    BYTEA,
    resolution_status   VARCHAR(50) DEFAULT 'unresolved',
    resolution_type     VARCHAR(50),
    resolved_at         TIMESTAMPTZ,
    resolved_by         VARCHAR(255),
    metadata            JSONB DEFAULT '{}',
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id           VARCHAR(255)
);

SELECT create_hypertable('gl_eudr_pbm_overlaps', 'detected_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

-- Table 5: gl_eudr_pbm_area_calculations - Area computation records
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_area_calculations (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    plot_id             VARCHAR(255) NOT NULL,
    area_m2             DOUBLE PRECISION NOT NULL,
    area_hectares       DOUBLE PRECISION NOT NULL,
    area_acres          DOUBLE PRECISION,
    area_km2            DOUBLE PRECISION,
    perimeter_m         DOUBLE PRECISION,
    compactness_pp      DOUBLE PRECISION,
    compactness_sw      DOUBLE PRECISION,
    compactness_chr     DOUBLE PRECISION,
    threshold_class     VARCHAR(30) NOT NULL,
    polygon_required    BOOLEAN NOT NULL,
    method              VARCHAR(30) NOT NULL DEFAULT 'karney',
    uncertainty_m2      DOUBLE PRECISION,
    duration_ms         DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id           VARCHAR(255)
);

SELECT create_hypertable('gl_eudr_pbm_area_calculations', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Table 6: gl_eudr_pbm_simplifications - Simplification records
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_simplifications (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    plot_id             VARCHAR(255) NOT NULL,
    method              VARCHAR(50) NOT NULL,
    tolerance           DOUBLE PRECISION,
    original_vertices   INTEGER NOT NULL,
    simplified_vertices INTEGER NOT NULL,
    reduction_ratio     DOUBLE PRECISION NOT NULL,
    area_change_pct     DOUBLE PRECISION NOT NULL,
    hausdorff_distance  DOUBLE PRECISION,
    resolution_level    VARCHAR(30),
    duration_ms         DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(255),
    tenant_id           VARCHAR(255)
);

-- Table 7: gl_eudr_pbm_split_merges - Split/merge genealogy
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_split_merges (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    operation_type      VARCHAR(10) NOT NULL CHECK (operation_type IN ('split', 'merge')),
    parent_plot_ids     VARCHAR(255)[] NOT NULL,
    child_plot_ids      VARCHAR(255)[] NOT NULL,
    parent_area_m2      DOUBLE PRECISION NOT NULL,
    child_areas_m2      DOUBLE PRECISION[] NOT NULL,
    area_conservation   BOOLEAN NOT NULL,
    area_deviation_pct  DOUBLE PRECISION,
    cutting_line_wkb    BYTEA,
    effective_date      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_hash     VARCHAR(128),
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(255),
    tenant_id           VARCHAR(255)
);

-- Table 8: gl_eudr_pbm_exports - Export operation records
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_exports (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    export_format       VARCHAR(30) NOT NULL,
    plot_count          INTEGER NOT NULL,
    plot_ids            VARCHAR(255)[],
    file_size_bytes     BIGINT,
    crs                 VARCHAR(50) NOT NULL DEFAULT 'EPSG:4326',
    precision_digits    INTEGER DEFAULT 8,
    simplified          BOOLEAN NOT NULL DEFAULT FALSE,
    simplification_tol  DOUBLE PRECISION,
    compliance_valid    BOOLEAN,
    duration_ms         DOUBLE PRECISION,
    download_url        TEXT,
    expires_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(255),
    tenant_id           VARCHAR(255)
);

-- Table 9: gl_eudr_pbm_batch_jobs - Batch processing
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_batch_jobs (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    operation           VARCHAR(50) NOT NULL,
    status              VARCHAR(20) NOT NULL DEFAULT 'pending',
    total_items         INTEGER NOT NULL DEFAULT 0,
    processed_items     INTEGER NOT NULL DEFAULT 0,
    failed_items        INTEGER NOT NULL DEFAULT 0,
    progress_pct        DOUBLE PRECISION NOT NULL DEFAULT 0,
    parameters          JSONB DEFAULT '{}',
    results             JSONB DEFAULT '{}',
    error_details       JSONB DEFAULT '[]',
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(255),
    tenant_id           VARCHAR(255)
);

-- Table 10: gl_eudr_pbm_audit_log - Immutable audit trail
CREATE TABLE IF NOT EXISTS gl_eudr_pbm_audit_log (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    entity_type         VARCHAR(50) NOT NULL,
    entity_id           VARCHAR(255) NOT NULL,
    action              VARCHAR(50) NOT NULL,
    actor_id            VARCHAR(255),
    actor_ip            VARCHAR(45),
    details             JSONB DEFAULT '{}',
    previous_state      JSONB,
    new_state           JSONB,
    provenance_hash     VARCHAR(128),
    chain_hash          VARCHAR(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id           VARCHAR(255)
);

-- ============================================================================
-- Indexes
-- ============================================================================

-- Boundaries indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_boundaries_plot_id
    ON gl_eudr_pbm_boundaries (plot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_boundaries_commodity
    ON gl_eudr_pbm_boundaries (commodity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_boundaries_country
    ON gl_eudr_pbm_boundaries (country_iso, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_boundaries_active
    ON gl_eudr_pbm_boundaries (is_active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_boundaries_bbox
    ON gl_eudr_pbm_boundaries (bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_boundaries_tenant
    ON gl_eudr_pbm_boundaries (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_boundaries_owner
    ON gl_eudr_pbm_boundaries (owner_id, created_at DESC);

-- Versions indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_versions_plot
    ON gl_eudr_pbm_versions (plot_id, version_number DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_versions_reason
    ON gl_eudr_pbm_versions (change_reason, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_versions_tenant
    ON gl_eudr_pbm_versions (tenant_id, created_at DESC);

-- Validations indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_validations_plot
    ON gl_eudr_pbm_validations (plot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_validations_valid
    ON gl_eudr_pbm_validations (is_valid, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_validations_tenant
    ON gl_eudr_pbm_validations (tenant_id, created_at DESC);

-- Overlaps indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_overlaps_plot_a
    ON gl_eudr_pbm_overlaps (plot_id_a, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_overlaps_plot_b
    ON gl_eudr_pbm_overlaps (plot_id_b, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_overlaps_severity
    ON gl_eudr_pbm_overlaps (severity, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_overlaps_status
    ON gl_eudr_pbm_overlaps (resolution_status, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_overlaps_tenant
    ON gl_eudr_pbm_overlaps (tenant_id, detected_at DESC);

-- Area calculations indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_area_plot
    ON gl_eudr_pbm_area_calculations (plot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_area_threshold
    ON gl_eudr_pbm_area_calculations (threshold_class, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_area_tenant
    ON gl_eudr_pbm_area_calculations (tenant_id, created_at DESC);

-- Simplifications indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_simplifications_plot
    ON gl_eudr_pbm_simplifications (plot_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_simplifications_method
    ON gl_eudr_pbm_simplifications (method, created_at DESC);

-- Split/merge indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_split_merges_type
    ON gl_eudr_pbm_split_merges (operation_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_split_merges_tenant
    ON gl_eudr_pbm_split_merges (tenant_id, created_at DESC);

-- Exports indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_exports_format
    ON gl_eudr_pbm_exports (export_format, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_exports_tenant
    ON gl_eudr_pbm_exports (tenant_id, created_at DESC);

-- Batch jobs indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_batch_status
    ON gl_eudr_pbm_batch_jobs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_batch_tenant
    ON gl_eudr_pbm_batch_jobs (tenant_id, created_at DESC);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_audit_entity
    ON gl_eudr_pbm_audit_log (entity_type, entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_audit_action
    ON gl_eudr_pbm_audit_log (action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_pbm_audit_tenant
    ON gl_eudr_pbm_audit_log (tenant_id, created_at DESC);

-- ============================================================================
-- Continuous Aggregates
-- ============================================================================

-- Daily boundary statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_pbm_daily_boundary_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at) AS bucket,
    commodity,
    country_iso,
    tenant_id,
    COUNT(*) AS boundaries_created,
    AVG(area_hectares) AS avg_area_ha,
    SUM(area_hectares) AS total_area_ha,
    AVG(vertex_count) AS avg_vertices,
    COUNT(*) FILTER (WHERE area_hectares >= 4.0) AS polygon_required_count,
    COUNT(*) FILTER (WHERE area_hectares < 4.0) AS point_sufficient_count
FROM gl_eudr_pbm_boundaries
GROUP BY bucket, commodity, country_iso, tenant_id
WITH NO DATA;

-- Weekly validation summary
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_pbm_weekly_validation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', created_at) AS bucket,
    tenant_id,
    COUNT(*) AS total_validations,
    COUNT(*) FILTER (WHERE is_valid = TRUE) AS valid_count,
    COUNT(*) FILTER (WHERE is_valid = FALSE) AS invalid_count,
    COUNT(*) FILTER (WHERE repaired = TRUE) AS repaired_count,
    AVG(confidence_score) AS avg_confidence,
    AVG(error_count) AS avg_errors
FROM gl_eudr_pbm_validations
GROUP BY bucket, tenant_id
WITH NO DATA;

-- ============================================================================
-- Retention Policies (5 years per EUDR Article 31)
-- ============================================================================

SELECT add_retention_policy('gl_eudr_pbm_boundaries',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_pbm_versions',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_pbm_validations',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_pbm_overlaps',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_pbm_area_calculations',
    INTERVAL '5 years', if_not_exists => TRUE);

-- ============================================================================
-- Continuous Aggregate Refresh Policies
-- ============================================================================

SELECT add_continuous_aggregate_policy('gl_eudr_pbm_daily_boundary_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('gl_eudr_pbm_weekly_validation_stats',
    start_offset => INTERVAL '14 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '4 hours',
    if_not_exists => TRUE);

-- ============================================================================
-- Comments
-- ============================================================================
COMMENT ON TABLE gl_eudr_pbm_boundaries IS 'AGENT-EUDR-006: Plot boundary records with WKB geometry';
COMMENT ON TABLE gl_eudr_pbm_versions IS 'AGENT-EUDR-006: Immutable boundary version history for EUDR Article 31';
COMMENT ON TABLE gl_eudr_pbm_validations IS 'AGENT-EUDR-006: Topology validation results';
COMMENT ON TABLE gl_eudr_pbm_overlaps IS 'AGENT-EUDR-006: Overlap detection results';
COMMENT ON TABLE gl_eudr_pbm_area_calculations IS 'AGENT-EUDR-006: Geodetic area computation records';
COMMENT ON TABLE gl_eudr_pbm_simplifications IS 'AGENT-EUDR-006: Polygon simplification records';
COMMENT ON TABLE gl_eudr_pbm_split_merges IS 'AGENT-EUDR-006: Split/merge genealogy tracking';
COMMENT ON TABLE gl_eudr_pbm_exports IS 'AGENT-EUDR-006: Multi-format export records';
COMMENT ON TABLE gl_eudr_pbm_batch_jobs IS 'AGENT-EUDR-006: Batch processing job tracker';
COMMENT ON TABLE gl_eudr_pbm_audit_log IS 'AGENT-EUDR-006: Immutable audit trail';
