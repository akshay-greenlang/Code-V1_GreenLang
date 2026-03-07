-- ============================================================================
-- V095: AGENT-EUDR-007 GPS Coordinate Validator
-- ============================================================================
-- Agent: GL-EUDR-GCV-007
-- Description: Database schema for the GPS Coordinate Validator agent
-- Tables: 10 (5 hypertables + 5 regular)
-- Regulation: EU Regulation 2023/1115 (EUDR) Article 9(1)(d)
-- Prefix: gl_eudr_gcv_
-- ============================================================================

-- 1. Individual coordinate validation results (hypertable, monthly partitioning)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_validations (
    id                      BIGSERIAL       NOT NULL,
    validation_id           UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    -- Input coordinate
    input_raw               TEXT            NOT NULL,
    input_format            VARCHAR(30)     NOT NULL,  -- dd, dms, ddm, utm, mgrs, etc.
    input_datum             VARCHAR(30)     DEFAULT 'WGS84',
    -- Parsed/normalized output
    latitude_wgs84          DOUBLE PRECISION,
    longitude_wgs84         DOUBLE PRECISION,
    altitude_m              DOUBLE PRECISION,
    -- Format detection
    format_detected         VARCHAR(30),
    format_confidence       DOUBLE PRECISION,
    -- Precision analysis
    lat_decimal_places      SMALLINT,
    lon_decimal_places      SMALLINT,
    ground_resolution_m     DOUBLE PRECISION,
    precision_class         VARCHAR(20),  -- survey_grade, high, moderate, low, inadequate
    eudr_precision_adequate BOOLEAN,
    -- Validation results
    is_valid                BOOLEAN         NOT NULL DEFAULT false,
    range_check_passed      BOOLEAN,
    swap_detected           BOOLEAN         DEFAULT false,
    sign_error_detected     BOOLEAN         DEFAULT false,
    hemisphere_error        BOOLEAN         DEFAULT false,
    null_island_detected    BOOLEAN         DEFAULT false,
    -- Spatial plausibility
    on_land                 BOOLEAN,
    country_code            VARCHAR(3),
    country_match           BOOLEAN,
    declared_country        VARCHAR(3),
    commodity_plausible     BOOLEAN,
    elevation_plausible     BOOLEAN,
    urban_area_detected     BOOLEAN         DEFAULT false,
    protected_area_nearby   BOOLEAN         DEFAULT false,
    -- Accuracy scoring
    accuracy_score          DOUBLE PRECISION,
    accuracy_tier           VARCHAR(15),  -- gold, silver, bronze, unverified
    precision_subscore      DOUBLE PRECISION,
    plausibility_subscore   DOUBLE PRECISION,
    consistency_subscore    DOUBLE PRECISION,
    source_subscore         DOUBLE PRECISION,
    -- Auto-correction
    auto_corrected          BOOLEAN         DEFAULT false,
    correction_type         VARCHAR(40),
    corrected_latitude      DOUBLE PRECISION,
    corrected_longitude     DOUBLE PRECISION,
    correction_confidence   DOUBLE PRECISION,
    -- Source metadata
    source_type             VARCHAR(30),  -- gnss_survey, mobile_phone, manual_entry, erp, etc.
    plot_id                 UUID,
    supplier_id             UUID,
    commodity               VARCHAR(30),
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    errors                  JSONB           DEFAULT '[]'::jsonb,
    warnings                JSONB           DEFAULT '[]'::jsonb,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_gcv_validations', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- 2. Batch validation job records (hypertable, monthly partitioning)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_batch_validations (
    id                      BIGSERIAL       NOT NULL,
    batch_id                UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    -- Batch statistics
    total_coordinates       INTEGER         NOT NULL DEFAULT 0,
    valid_count             INTEGER         NOT NULL DEFAULT 0,
    invalid_count           INTEGER         NOT NULL DEFAULT 0,
    warning_count           INTEGER         NOT NULL DEFAULT 0,
    auto_corrected_count    INTEGER         NOT NULL DEFAULT 0,
    -- Quality stats
    avg_accuracy_score      DOUBLE PRECISION,
    gold_count              INTEGER         DEFAULT 0,
    silver_count            INTEGER         DEFAULT 0,
    bronze_count            INTEGER         DEFAULT 0,
    unverified_count        INTEGER         DEFAULT 0,
    -- Precision stats
    avg_decimal_places      DOUBLE PRECISION,
    inadequate_precision    INTEGER         DEFAULT 0,
    -- Spatial stats
    ocean_count             INTEGER         DEFAULT 0,
    country_mismatch_count  INTEGER         DEFAULT 0,
    swap_detected_count     INTEGER         DEFAULT 0,
    duplicate_count         INTEGER         DEFAULT 0,
    -- Processing
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    processing_time_ms      INTEGER,
    error_message           TEXT,
    -- Source
    source_file             TEXT,
    source_format           VARCHAR(20),
    declared_country        VARCHAR(3),
    commodity               VARCHAR(30),
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_gcv_batch_validations', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- 3. Datum transformation records (hypertable, monthly partitioning)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_transformations (
    id                      BIGSERIAL       NOT NULL,
    transformation_id       UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    -- Source coordinate
    source_datum            VARCHAR(30)     NOT NULL,
    source_latitude         DOUBLE PRECISION NOT NULL,
    source_longitude        DOUBLE PRECISION NOT NULL,
    source_altitude_m       DOUBLE PRECISION,
    -- Target coordinate (always WGS84)
    target_datum            VARCHAR(10)     NOT NULL DEFAULT 'WGS84',
    target_latitude         DOUBLE PRECISION NOT NULL,
    target_longitude        DOUBLE PRECISION NOT NULL,
    target_altitude_m       DOUBLE PRECISION,
    -- Transformation details
    method                  VARCHAR(30)     NOT NULL,  -- helmert_7p, molodensky, identity
    displacement_m          DOUBLE PRECISION NOT NULL,
    displacement_lat_m      DOUBLE PRECISION,
    displacement_lon_m      DOUBLE PRECISION,
    displacement_alt_m      DOUBLE PRECISION,
    transformation_accuracy VARCHAR(20),  -- sub_meter, meter, decameter, hectometer
    -- Parameters used
    dx                      DOUBLE PRECISION,
    dy                      DOUBLE PRECISION,
    dz                      DOUBLE PRECISION,
    rx                      DOUBLE PRECISION,
    ry                      DOUBLE PRECISION,
    rz                      DOUBLE PRECISION,
    ds                      DOUBLE PRECISION,
    -- Context
    auto_detected_datum     BOOLEAN         DEFAULT false,
    country_hint            VARCHAR(3),
    validation_id           UUID,
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_gcv_transformations', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- 4. Spatial plausibility results (hypertable, quarterly partitioning)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_plausibility_checks (
    id                      BIGSERIAL       NOT NULL,
    check_id                UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    validation_id           UUID,
    -- Coordinate checked
    latitude_wgs84          DOUBLE PRECISION NOT NULL,
    longitude_wgs84         DOUBLE PRECISION NOT NULL,
    -- Land/ocean check
    on_land                 BOOLEAN         NOT NULL,
    distance_to_coast_km    DOUBLE PRECISION,
    -- Country check
    country_code            VARCHAR(3),
    country_name            VARCHAR(100),
    admin_region            VARCHAR(200),
    declared_country        VARCHAR(3),
    country_match           BOOLEAN,
    -- Commodity plausibility
    commodity               VARCHAR(30),
    commodity_plausible     BOOLEAN,
    commodity_reason        TEXT,
    elevation_m             DOUBLE PRECISION,
    elevation_plausible     BOOLEAN,
    elevation_range_min     DOUBLE PRECISION,
    elevation_range_max     DOUBLE PRECISION,
    -- Climate zone
    climate_zone            VARCHAR(30),
    climate_compatible      BOOLEAN,
    -- Land use context
    land_use_class          VARCHAR(30),  -- forest, agricultural, urban, water, etc.
    -- Protected area
    protected_area_nearby   BOOLEAN         DEFAULT false,
    protected_area_name     VARCHAR(200),
    protected_area_dist_km  DOUBLE PRECISION,
    -- Urban detection
    urban_area_detected     BOOLEAN         DEFAULT false,
    nearest_city            VARCHAR(200),
    distance_to_city_km     DOUBLE PRECISION,
    -- Overall plausibility
    plausibility_score      DOUBLE PRECISION,
    plausibility_pass       BOOLEAN         NOT NULL,
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    issues                  JSONB           DEFAULT '[]'::jsonb,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_gcv_plausibility_checks', 'created_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

-- 5. Quality score records (hypertable, monthly partitioning)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_accuracy_scores (
    id                      BIGSERIAL       NOT NULL,
    score_id                UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    validation_id           UUID,
    -- Coordinate assessed
    latitude_wgs84          DOUBLE PRECISION NOT NULL,
    longitude_wgs84         DOUBLE PRECISION NOT NULL,
    -- Composite score
    accuracy_score          DOUBLE PRECISION NOT NULL,  -- 0-100
    accuracy_tier           VARCHAR(15)     NOT NULL,  -- gold, silver, bronze, unverified
    -- Sub-scores (each 0-100)
    precision_subscore      DOUBLE PRECISION NOT NULL,
    plausibility_subscore   DOUBLE PRECISION NOT NULL,
    consistency_subscore    DOUBLE PRECISION NOT NULL,
    source_subscore         DOUBLE PRECISION NOT NULL,
    -- Precision details
    decimal_places          SMALLINT,
    ground_resolution_m     DOUBLE PRECISION,
    precision_class         VARCHAR(20),
    -- Source details
    source_type             VARCHAR(30),
    source_reliability      DOUBLE PRECISION,
    -- Confidence interval
    confidence_radius_m     DOUBLE PRECISION,
    confidence_level        DOUBLE PRECISION DEFAULT 0.95,
    -- Context
    plot_id                 UUID,
    supplier_id             UUID,
    commodity               VARCHAR(30),
    -- Explanations
    score_breakdown         JSONB           DEFAULT '{}'::jsonb,
    recommendations         JSONB           DEFAULT '[]'::jsonb,
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_gcv_accuracy_scores', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- 6. EUDR compliance certificates (regular table)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_compliance_certs (
    id                      BIGSERIAL       PRIMARY KEY,
    cert_id                 UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    -- Certificate details
    cert_type               VARCHAR(30)     NOT NULL,  -- single, batch, dds_ready
    status                  VARCHAR(20)     NOT NULL DEFAULT 'active',  -- active, revoked, expired
    valid_from              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    valid_until             TIMESTAMPTZ     NOT NULL DEFAULT (now() + INTERVAL '5 years'),
    -- Scope
    validation_id           UUID,
    batch_id                UUID,
    coordinate_count        INTEGER         NOT NULL DEFAULT 1,
    -- Quality summary
    all_valid               BOOLEAN         NOT NULL,
    avg_accuracy_score      DOUBLE PRECISION,
    min_accuracy_tier       VARCHAR(15),
    precision_adequate      BOOLEAN         NOT NULL,
    all_on_land             BOOLEAN,
    all_country_match       BOOLEAN,
    all_commodity_plausible BOOLEAN,
    -- EUDR Article 9 compliance
    article_9_compliant     BOOLEAN         NOT NULL,
    compliance_details      JSONB           DEFAULT '{}'::jsonb,
    -- Certificate content
    certificate_data        JSONB           NOT NULL DEFAULT '{}'::jsonb,
    certificate_format      VARCHAR(10)     DEFAULT 'json',  -- json, pdf, xml
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    issued_by               VARCHAR(100)    NOT NULL DEFAULT 'GL-EUDR-GCV-007',
    metadata                JSONB           DEFAULT '{}'::jsonb,
    CONSTRAINT uq_gcv_cert_id UNIQUE (cert_id)
);

-- 7. Reverse geocoding results (regular table)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_reverse_geocodes (
    id                      BIGSERIAL       PRIMARY KEY,
    geocode_id              UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    -- Coordinate
    latitude_wgs84          DOUBLE PRECISION NOT NULL,
    longitude_wgs84         DOUBLE PRECISION NOT NULL,
    -- Results
    country_code            VARCHAR(3),
    country_name            VARCHAR(100),
    admin_level_1           VARCHAR(200),  -- province/state
    admin_level_2           VARCHAR(200),  -- district/county
    nearest_place           VARCHAR(200),
    distance_to_place_km    DOUBLE PRECISION,
    -- Land use context
    land_use_class          VARCHAR(30),
    land_cover_type         VARCHAR(50),
    -- Geographic context
    distance_to_coast_km    DOUBLE PRECISION,
    elevation_m             DOUBLE PRECISION,
    climate_zone            VARCHAR(30),
    -- Commodity context
    commodity_zone          VARCHAR(50),
    in_known_growing_region BOOLEAN         DEFAULT false,
    -- Cache control
    source                  VARCHAR(30)     DEFAULT 'offline',  -- offline, cached, api
    cache_expires           TIMESTAMPTZ,
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    CONSTRAINT uq_gcv_geocode_id UNIQUE (geocode_id)
);

-- 8. Auto-correction records (regular table)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_error_corrections (
    id                      BIGSERIAL       PRIMARY KEY,
    correction_id           UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    validation_id           UUID,
    -- Original values
    original_latitude       DOUBLE PRECISION,
    original_longitude      DOUBLE PRECISION,
    original_format         VARCHAR(30),
    original_datum          VARCHAR(30),
    -- Corrected values
    corrected_latitude      DOUBLE PRECISION NOT NULL,
    corrected_longitude     DOUBLE PRECISION NOT NULL,
    -- Correction details
    correction_type         VARCHAR(40)     NOT NULL,  -- swap_fix, sign_fix, hemisphere_fix, datum_transform, precision_upgrade
    correction_description  TEXT            NOT NULL,
    correction_confidence   DOUBLE PRECISION NOT NULL,
    displacement_m          DOUBLE PRECISION,
    -- Approval
    auto_applied            BOOLEAN         NOT NULL DEFAULT false,
    requires_review         BOOLEAN         NOT NULL DEFAULT true,
    reviewed_by             VARCHAR(100),
    reviewed_at             TIMESTAMPTZ,
    approved                BOOLEAN,
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    CONSTRAINT uq_gcv_correction_id UNIQUE (correction_id)
);

-- 9. Batch processing jobs (regular table)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_batch_jobs (
    id                      BIGSERIAL       PRIMARY KEY,
    job_id                  UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    -- Job details
    job_type                VARCHAR(30)     NOT NULL,  -- validate, transform, assess, geocode, full_pipeline
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
    priority                SMALLINT        NOT NULL DEFAULT 5,
    -- Input
    input_source            TEXT,
    input_format            VARCHAR(20),
    total_items             INTEGER         NOT NULL DEFAULT 0,
    -- Progress
    processed_items         INTEGER         NOT NULL DEFAULT 0,
    successful_items        INTEGER         NOT NULL DEFAULT 0,
    failed_items            INTEGER         NOT NULL DEFAULT 0,
    progress_pct            DOUBLE PRECISION DEFAULT 0.0,
    -- Configuration
    config                  JSONB           DEFAULT '{}'::jsonb,
    declared_country        VARCHAR(3),
    commodity               VARCHAR(30),
    source_datum            VARCHAR(30),
    -- Timing
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    processing_time_ms      INTEGER,
    estimated_completion    TIMESTAMPTZ,
    -- Results
    result_summary          JSONB           DEFAULT '{}'::jsonb,
    batch_validation_id     UUID,
    error_message           TEXT,
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    submitted_by            VARCHAR(100),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    CONSTRAINT uq_gcv_job_id UNIQUE (job_id)
);

-- 10. Immutable audit trail (regular table)
CREATE TABLE IF NOT EXISTS gl_eudr_gcv_audit_log (
    id                      BIGSERIAL       PRIMARY KEY,
    log_id                  UUID            NOT NULL DEFAULT gen_random_uuid(),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    tenant_id               UUID            NOT NULL,
    -- Event details
    event_type              VARCHAR(50)     NOT NULL,
    event_category          VARCHAR(30)     NOT NULL,  -- parse, validate, transform, assess, report, geocode, batch
    severity                VARCHAR(10)     NOT NULL DEFAULT 'info',  -- info, warning, error, critical
    -- Actor
    actor_id                VARCHAR(100),
    actor_type              VARCHAR(20)     DEFAULT 'system',
    -- Entity
    entity_type             VARCHAR(30),  -- coordinate, batch, certificate, correction, etc.
    entity_id               UUID,
    -- Detail
    description             TEXT            NOT NULL,
    old_value               JSONB,
    new_value               JSONB,
    -- Context
    request_id              UUID,
    ip_address              VARCHAR(45),
    user_agent              TEXT,
    -- Provenance
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    CONSTRAINT uq_gcv_log_id UNIQUE (log_id)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Validations indexes
CREATE INDEX IF NOT EXISTS idx_gcv_validations_tenant
    ON gl_eudr_gcv_validations (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_validations_validation_id
    ON gl_eudr_gcv_validations (validation_id);
CREATE INDEX IF NOT EXISTS idx_gcv_validations_plot
    ON gl_eudr_gcv_validations (plot_id, created_at DESC)
    WHERE plot_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_gcv_validations_supplier
    ON gl_eudr_gcv_validations (supplier_id, created_at DESC)
    WHERE supplier_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_gcv_validations_valid
    ON gl_eudr_gcv_validations (is_valid, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_validations_tier
    ON gl_eudr_gcv_validations (accuracy_tier, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_validations_swap
    ON gl_eudr_gcv_validations (swap_detected, created_at DESC)
    WHERE swap_detected = true;

-- Batch validations indexes
CREATE INDEX IF NOT EXISTS idx_gcv_batch_tenant
    ON gl_eudr_gcv_batch_validations (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_batch_id
    ON gl_eudr_gcv_batch_validations (batch_id);
CREATE INDEX IF NOT EXISTS idx_gcv_batch_status
    ON gl_eudr_gcv_batch_validations (status, created_at DESC);

-- Transformations indexes
CREATE INDEX IF NOT EXISTS idx_gcv_transforms_tenant
    ON gl_eudr_gcv_transformations (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_transforms_id
    ON gl_eudr_gcv_transformations (transformation_id);
CREATE INDEX IF NOT EXISTS idx_gcv_transforms_datum
    ON gl_eudr_gcv_transformations (source_datum, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_transforms_validation
    ON gl_eudr_gcv_transformations (validation_id)
    WHERE validation_id IS NOT NULL;

-- Plausibility checks indexes
CREATE INDEX IF NOT EXISTS idx_gcv_plausibility_tenant
    ON gl_eudr_gcv_plausibility_checks (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_plausibility_id
    ON gl_eudr_gcv_plausibility_checks (check_id);
CREATE INDEX IF NOT EXISTS idx_gcv_plausibility_validation
    ON gl_eudr_gcv_plausibility_checks (validation_id)
    WHERE validation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_gcv_plausibility_country
    ON gl_eudr_gcv_plausibility_checks (country_code, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_plausibility_ocean
    ON gl_eudr_gcv_plausibility_checks (on_land, created_at DESC)
    WHERE on_land = false;

-- Accuracy scores indexes
CREATE INDEX IF NOT EXISTS idx_gcv_scores_tenant
    ON gl_eudr_gcv_accuracy_scores (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_scores_id
    ON gl_eudr_gcv_accuracy_scores (score_id);
CREATE INDEX IF NOT EXISTS idx_gcv_scores_tier
    ON gl_eudr_gcv_accuracy_scores (accuracy_tier, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_scores_validation
    ON gl_eudr_gcv_accuracy_scores (validation_id)
    WHERE validation_id IS NOT NULL;

-- Compliance certificates indexes
CREATE INDEX IF NOT EXISTS idx_gcv_certs_tenant
    ON gl_eudr_gcv_compliance_certs (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_certs_status
    ON gl_eudr_gcv_compliance_certs (status, valid_until);
CREATE INDEX IF NOT EXISTS idx_gcv_certs_batch
    ON gl_eudr_gcv_compliance_certs (batch_id)
    WHERE batch_id IS NOT NULL;

-- Reverse geocodes indexes
CREATE INDEX IF NOT EXISTS idx_gcv_geocodes_tenant
    ON gl_eudr_gcv_reverse_geocodes (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_geocodes_country
    ON gl_eudr_gcv_reverse_geocodes (country_code);
CREATE INDEX IF NOT EXISTS idx_gcv_geocodes_latlon
    ON gl_eudr_gcv_reverse_geocodes (latitude_wgs84, longitude_wgs84);

-- Error corrections indexes
CREATE INDEX IF NOT EXISTS idx_gcv_corrections_tenant
    ON gl_eudr_gcv_error_corrections (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_corrections_type
    ON gl_eudr_gcv_error_corrections (correction_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_corrections_review
    ON gl_eudr_gcv_error_corrections (requires_review, approved)
    WHERE requires_review = true AND approved IS NULL;

-- Batch jobs indexes
CREATE INDEX IF NOT EXISTS idx_gcv_jobs_tenant
    ON gl_eudr_gcv_batch_jobs (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_jobs_status
    ON gl_eudr_gcv_batch_jobs (status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_jobs_type
    ON gl_eudr_gcv_batch_jobs (job_type, created_at DESC);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_gcv_audit_tenant
    ON gl_eudr_gcv_audit_log (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_audit_event
    ON gl_eudr_gcv_audit_log (event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_audit_category
    ON gl_eudr_gcv_audit_log (event_category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gcv_audit_entity
    ON gl_eudr_gcv_audit_log (entity_type, entity_id)
    WHERE entity_id IS NOT NULL;

-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Hourly validation statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_gcv_validation_stats_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at)   AS bucket,
    tenant_id,
    COUNT(*)                            AS total_validations,
    COUNT(*) FILTER (WHERE is_valid)    AS valid_count,
    COUNT(*) FILTER (WHERE NOT is_valid) AS invalid_count,
    COUNT(*) FILTER (WHERE swap_detected) AS swap_count,
    COUNT(*) FILTER (WHERE NOT on_land) AS ocean_count,
    COUNT(*) FILTER (WHERE auto_corrected) AS corrected_count,
    AVG(accuracy_score)                 AS avg_accuracy_score,
    COUNT(*) FILTER (WHERE accuracy_tier = 'gold')   AS gold_count,
    COUNT(*) FILTER (WHERE accuracy_tier = 'silver') AS silver_count,
    COUNT(*) FILTER (WHERE accuracy_tier = 'bronze') AS bronze_count,
    COUNT(*) FILTER (WHERE accuracy_tier = 'unverified') AS unverified_count,
    AVG(ground_resolution_m)            AS avg_resolution_m
FROM gl_eudr_gcv_validations
GROUP BY bucket, tenant_id
WITH NO DATA;

-- Daily transformation statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_gcv_transform_stats_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at)    AS bucket,
    tenant_id,
    source_datum,
    COUNT(*)                            AS total_transformations,
    AVG(displacement_m)                 AS avg_displacement_m,
    MAX(displacement_m)                 AS max_displacement_m,
    COUNT(*) FILTER (WHERE auto_detected_datum) AS auto_detected_count
FROM gl_eudr_gcv_transformations
GROUP BY bucket, tenant_id, source_datum
WITH NO DATA;

-- ============================================================================
-- CONTINUOUS AGGREGATE POLICIES
-- ============================================================================

SELECT add_continuous_aggregate_policy('gl_eudr_gcv_validation_stats_hourly',
    start_offset    => INTERVAL '3 hours',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

SELECT add_continuous_aggregate_policy('gl_eudr_gcv_transform_stats_daily',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists   => TRUE
);

-- ============================================================================
-- RETENTION POLICIES (5-year retention per EUDR Article 31)
-- ============================================================================

SELECT add_retention_policy('gl_eudr_gcv_validations',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_gcv_batch_validations',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_gcv_transformations',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_gcv_plausibility_checks',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_gcv_accuracy_scores',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE gl_eudr_gcv_validations IS 'AGENT-EUDR-007: Individual GPS coordinate validation results';
COMMENT ON TABLE gl_eudr_gcv_batch_validations IS 'AGENT-EUDR-007: Batch GPS coordinate validation summaries';
COMMENT ON TABLE gl_eudr_gcv_transformations IS 'AGENT-EUDR-007: Datum transformation records (to WGS84)';
COMMENT ON TABLE gl_eudr_gcv_plausibility_checks IS 'AGENT-EUDR-007: Spatial plausibility verification results';
COMMENT ON TABLE gl_eudr_gcv_accuracy_scores IS 'AGENT-EUDR-007: Coordinate accuracy quality scores (0-100)';
COMMENT ON TABLE gl_eudr_gcv_compliance_certs IS 'AGENT-EUDR-007: EUDR Article 9 compliance certificates';
COMMENT ON TABLE gl_eudr_gcv_reverse_geocodes IS 'AGENT-EUDR-007: Reverse geocoding results (offline)';
COMMENT ON TABLE gl_eudr_gcv_error_corrections IS 'AGENT-EUDR-007: Auto-correction records for coordinate errors';
COMMENT ON TABLE gl_eudr_gcv_batch_jobs IS 'AGENT-EUDR-007: Batch processing job tracking';
COMMENT ON TABLE gl_eudr_gcv_audit_log IS 'AGENT-EUDR-007: Immutable audit trail for all coordinate operations';
