-- =============================================================================
-- V092: AGENT-EUDR-004 Forest Cover Analysis Agent
-- =============================================================================
-- Agent: GL-EUDR-FCA-004
-- Description: Forest cover characterization, classification, historical
--              reconstruction, and deforestation-free verification for EUDR.
-- Regulation: EU 2023/1115 (EUDR) Articles 2, 9, 10, 12
-- Tables: 10 + 5 hypertables + 2 continuous aggregates + 28 indexes
-- Retention: 5 years per EUDR Article 31
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Canopy Density Maps (hypertable, monthly partitioning)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_canopy_density_maps (
    id                  BIGSERIAL       NOT NULL,
    analysis_id         UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    plot_id             TEXT            NOT NULL,
    polygon_wkt         TEXT,
    imagery_date        DATE,
    density_pct         DOUBLE PRECISION NOT NULL CHECK (density_pct >= 0 AND density_pct <= 100),
    density_class       TEXT            NOT NULL CHECK (density_class IN (
                            'very_high', 'high', 'moderate', 'low', 'sparse', 'open')),
    method              TEXT            NOT NULL CHECK (method IN (
                            'spectral_unmixing', 'ndvi_regression', 'dimidiation', 'sub_pixel_detection')),
    confidence          DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    meets_fao_threshold BOOLEAN         NOT NULL DEFAULT FALSE,
    pixel_count         INTEGER,
    spatial_resolution_m DOUBLE PRECISION,
    biome               TEXT,
    data_quality_score  DOUBLE PRECISION CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    provenance_hash     TEXT            NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    analyzed_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (analyzed_at, id)
);

SELECT create_hypertable('eudr_canopy_density_maps', 'analyzed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_eudr_cdm_tenant_plot
    ON eudr_canopy_density_maps (tenant_id, plot_id, analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_cdm_analysis_id
    ON eudr_canopy_density_maps (analysis_id);
CREATE INDEX IF NOT EXISTS idx_eudr_cdm_density_class
    ON eudr_canopy_density_maps (density_class, analyzed_at DESC);

-- ---------------------------------------------------------------------------
-- 2. Forest Classifications (hypertable, monthly partitioning)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_forest_classifications (
    id                      BIGSERIAL       NOT NULL,
    classification_id       UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    plot_id                 TEXT            NOT NULL,
    polygon_wkt             TEXT,
    forest_type             TEXT            NOT NULL CHECK (forest_type IN (
                                'primary_tropical', 'secondary_tropical', 'tropical_dry',
                                'temperate_broadleaf', 'temperate_coniferous', 'boreal',
                                'mangrove', 'plantation', 'agroforestry', 'non_forest')),
    probability             DOUBLE PRECISION NOT NULL CHECK (probability >= 0 AND probability <= 1),
    secondary_type          TEXT,
    secondary_probability   DOUBLE PRECISION CHECK (secondary_probability >= 0 AND secondary_probability <= 1),
    classification_method   TEXT            NOT NULL CHECK (classification_method IN (
                                'spectral_signature', 'phenological', 'structural',
                                'multi_temporal', 'ensemble')),
    is_forest_per_fao       BOOLEAN         NOT NULL DEFAULT FALSE,
    is_forest_per_eudr      BOOLEAN         NOT NULL DEFAULT FALSE,
    spectral_signature      JSONB,
    confidence              DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    provenance_hash         TEXT            NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    classified_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (classified_at, id)
);

SELECT create_hypertable('eudr_forest_classifications', 'classified_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_eudr_fc_tenant_plot
    ON eudr_forest_classifications (tenant_id, plot_id, classified_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_fc_classification_id
    ON eudr_forest_classifications (classification_id);
CREATE INDEX IF NOT EXISTS idx_eudr_fc_forest_type
    ON eudr_forest_classifications (forest_type, classified_at DESC);

-- ---------------------------------------------------------------------------
-- 3. Historical Reconstructions (hypertable, quarterly partitioning)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_historical_reconstructions (
    id                          BIGSERIAL       NOT NULL,
    reconstruction_id           UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    plot_id                     TEXT            NOT NULL,
    polygon_wkt                 TEXT,
    target_date                 DATE            NOT NULL DEFAULT '2020-12-31',
    was_forest                  BOOLEAN         NOT NULL,
    canopy_density_at_cutoff    DOUBLE PRECISION CHECK (canopy_density_at_cutoff >= 0 AND canopy_density_at_cutoff <= 100),
    forest_type_at_cutoff       TEXT,
    data_sources                TEXT[]          NOT NULL DEFAULT '{}',
    composite_quality           DOUBLE PRECISION CHECK (composite_quality >= 0 AND composite_quality <= 1),
    reconstruction_confidence   DOUBLE PRECISION NOT NULL CHECK (reconstruction_confidence >= 0 AND reconstruction_confidence <= 1),
    cross_validation_score      DOUBLE PRECISION CHECK (cross_validation_score >= 0 AND cross_validation_score <= 1),
    source_weights              JSONB           DEFAULT '{}',
    provenance_hash             TEXT            NOT NULL,
    metadata                    JSONB           DEFAULT '{}',
    reconstructed_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (reconstructed_at, id)
);

SELECT create_hypertable('eudr_historical_reconstructions', 'reconstructed_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_eudr_hr_tenant_plot
    ON eudr_historical_reconstructions (tenant_id, plot_id, reconstructed_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_hr_reconstruction_id
    ON eudr_historical_reconstructions (reconstruction_id);
CREATE INDEX IF NOT EXISTS idx_eudr_hr_target_date
    ON eudr_historical_reconstructions (target_date);

-- ---------------------------------------------------------------------------
-- 4. Deforestation-Free Verdicts (hypertable, monthly partitioning)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_deforestation_free_verdicts (
    id                  BIGSERIAL       NOT NULL,
    verdict_id          UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    plot_id             TEXT            NOT NULL,
    polygon_wkt         TEXT,
    commodity           TEXT            NOT NULL CHECK (commodity IN (
                            'cattle', 'cocoa', 'coffee', 'oil_palm',
                            'rubber', 'soya', 'wood')),
    verdict             TEXT            NOT NULL CHECK (verdict IN (
                            'deforestation_free', 'deforested', 'degraded', 'inconclusive')),
    cutoff_canopy_pct   DOUBLE PRECISION,
    current_canopy_pct  DOUBLE PRECISION,
    canopy_change_pct   DOUBLE PRECISION,
    confidence          DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence_summary    JSONB           DEFAULT '{}',
    regulatory_refs     TEXT[]          DEFAULT '{}',
    requires_review     BOOLEAN         NOT NULL DEFAULT FALSE,
    reconstruction_id   UUID,
    density_analysis_id UUID,
    provenance_hash     TEXT            NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    verified_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (verified_at, id)
);

SELECT create_hypertable('eudr_deforestation_free_verdicts', 'verified_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_eudr_dfv_tenant_plot
    ON eudr_deforestation_free_verdicts (tenant_id, plot_id, verified_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_dfv_verdict_id
    ON eudr_deforestation_free_verdicts (verdict_id);
CREATE INDEX IF NOT EXISTS idx_eudr_dfv_verdict
    ON eudr_deforestation_free_verdicts (verdict, verified_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_dfv_commodity
    ON eudr_deforestation_free_verdicts (commodity, verified_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_dfv_requires_review
    ON eudr_deforestation_free_verdicts (requires_review) WHERE requires_review = TRUE;

-- ---------------------------------------------------------------------------
-- 5. Canopy Height Estimates (hypertable, quarterly partitioning)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_canopy_height_estimates (
    id                  BIGSERIAL       NOT NULL,
    estimate_id         UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    plot_id             TEXT            NOT NULL,
    height_m            DOUBLE PRECISION NOT NULL CHECK (height_m >= 0),
    uncertainty_m       DOUBLE PRECISION NOT NULL CHECK (uncertainty_m >= 0),
    source              TEXT            NOT NULL CHECK (source IN (
                            'gedi_l2a', 'icesat2_atl08', 'sentinel2_texture',
                            'global_map_eth', 'global_map_meta', 'fused')),
    meets_fao_threshold BOOLEAN         NOT NULL DEFAULT FALSE,
    measurement_date    DATE,
    sources_used        TEXT[]          DEFAULT '{}',
    source_weights      JSONB           DEFAULT '{}',
    provenance_hash     TEXT            NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    estimated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (estimated_at, id)
);

SELECT create_hypertable('eudr_canopy_height_estimates', 'estimated_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_eudr_che_tenant_plot
    ON eudr_canopy_height_estimates (tenant_id, plot_id, estimated_at DESC);
CREATE INDEX IF NOT EXISTS idx_eudr_che_estimate_id
    ON eudr_canopy_height_estimates (estimate_id);

-- ---------------------------------------------------------------------------
-- 6. Fragmentation Analyses (standard table)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_fragmentation_analyses (
    id                  BIGSERIAL       PRIMARY KEY,
    analysis_id         UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    plot_id             TEXT            NOT NULL,
    patch_count         INTEGER         NOT NULL CHECK (patch_count >= 0),
    mean_patch_size_ha  DOUBLE PRECISION CHECK (mean_patch_size_ha >= 0),
    edge_density_m_ha   DOUBLE PRECISION CHECK (edge_density_m_ha >= 0),
    core_area_pct       DOUBLE PRECISION CHECK (core_area_pct >= 0 AND core_area_pct <= 100),
    connectivity_index  DOUBLE PRECISION CHECK (connectivity_index >= 0),
    shape_complexity    DOUBLE PRECISION CHECK (shape_complexity >= 0),
    effective_mesh_ha   DOUBLE PRECISION CHECK (effective_mesh_ha >= 0),
    fragmentation_level TEXT            NOT NULL CHECK (fragmentation_level IN (
                            'intact', 'slightly_fragmented', 'moderately_fragmented',
                            'highly_fragmented', 'severely_fragmented')),
    risk_score          DOUBLE PRECISION CHECK (risk_score >= 0 AND risk_score <= 1),
    edge_buffer_m       DOUBLE PRECISION DEFAULT 100.0,
    provenance_hash     TEXT            NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    analyzed_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eudr_fa_tenant_plot
    ON eudr_fragmentation_analyses (tenant_id, plot_id);
CREATE INDEX IF NOT EXISTS idx_eudr_fa_analysis_id
    ON eudr_fragmentation_analyses (analysis_id);
CREATE INDEX IF NOT EXISTS idx_eudr_fa_level
    ON eudr_fragmentation_analyses (fragmentation_level);

-- ---------------------------------------------------------------------------
-- 7. Biomass Estimates (standard table)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_biomass_estimates (
    id                  BIGSERIAL       PRIMARY KEY,
    estimate_id         UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    plot_id             TEXT            NOT NULL,
    agb_mg_per_ha       DOUBLE PRECISION NOT NULL CHECK (agb_mg_per_ha >= 0),
    uncertainty_mg_ha   DOUBLE PRECISION CHECK (uncertainty_mg_ha >= 0),
    carbon_stock_tc_ha  DOUBLE PRECISION CHECK (carbon_stock_tc_ha >= 0),
    source              TEXT            NOT NULL CHECK (source IN (
                            'esa_cci', 'gedi_l4a', 'sar_regression',
                            'ndvi_allometric', 'fused')),
    biomass_change_pct  DOUBLE PRECISION,
    sar_saturated       BOOLEAN         DEFAULT FALSE,
    biome               TEXT,
    sources_used        TEXT[]          DEFAULT '{}',
    provenance_hash     TEXT            NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    estimated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eudr_be_tenant_plot
    ON eudr_biomass_estimates (tenant_id, plot_id);
CREATE INDEX IF NOT EXISTS idx_eudr_be_estimate_id
    ON eudr_biomass_estimates (estimate_id);

-- ---------------------------------------------------------------------------
-- 8. Forest Compliance Reports (standard table)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_forest_compliance_reports (
    id                  BIGSERIAL       PRIMARY KEY,
    report_id           UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    plot_id             TEXT            NOT NULL,
    report_type         TEXT            NOT NULL CHECK (report_type IN (
                            'plot_assessment', 'batch_verification',
                            'deforestation_free_evidence', 'historical_report',
                            'dashboard_data')),
    format              TEXT            NOT NULL CHECK (format IN (
                            'json', 'pdf', 'csv', 'eudr_xml')),
    verdict             TEXT            CHECK (verdict IN (
                            'deforestation_free', 'deforested', 'degraded', 'inconclusive')),
    summary             JSONB           DEFAULT '{}',
    report_data         JSONB           DEFAULT '{}',
    requires_review     BOOLEAN         DEFAULT FALSE,
    provenance_hash     TEXT            NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eudr_fcr_tenant_plot
    ON eudr_forest_compliance_reports (tenant_id, plot_id);
CREATE INDEX IF NOT EXISTS idx_eudr_fcr_report_id
    ON eudr_forest_compliance_reports (report_id);
CREATE INDEX IF NOT EXISTS idx_eudr_fcr_report_type
    ON eudr_forest_compliance_reports (report_type, created_at DESC);

-- ---------------------------------------------------------------------------
-- 9. Forest Cover Baselines (standard table)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_forest_cover_baselines (
    id                  BIGSERIAL       PRIMARY KEY,
    baseline_id         UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    plot_id             TEXT            NOT NULL,
    target_date         DATE            NOT NULL DEFAULT '2020-12-31',
    composite_start     DATE            NOT NULL,
    composite_end       DATE            NOT NULL,
    data_sources        TEXT[]          NOT NULL DEFAULT '{}',
    ndvi_composite      DOUBLE PRECISION,
    tree_cover_pct      DOUBLE PRECISION CHECK (tree_cover_pct >= 0 AND tree_cover_pct <= 100),
    quality_score       DOUBLE PRECISION CHECK (quality_score >= 0 AND quality_score <= 1),
    provenance_hash     TEXT            NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, plot_id, target_date)
);

CREATE INDEX IF NOT EXISTS idx_eudr_fcb_tenant_plot
    ON eudr_forest_cover_baselines (tenant_id, plot_id);
CREATE INDEX IF NOT EXISTS idx_eudr_fcb_baseline_id
    ON eudr_forest_cover_baselines (baseline_id);

-- ---------------------------------------------------------------------------
-- 10. Forest Analysis Audit Log (immutable append-only)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eudr_forest_analysis_audit_log (
    id                  BIGSERIAL       PRIMARY KEY,
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    entity_type         TEXT            NOT NULL CHECK (entity_type IN (
                            'density_map', 'classification', 'reconstruction',
                            'verdict', 'height_estimate', 'fragmentation',
                            'biomass', 'report', 'baseline', 'batch')),
    entity_id           UUID            NOT NULL,
    action              TEXT            NOT NULL CHECK (action IN (
                            'create', 'update', 'validate', 'reconstruct',
                            'classify', 'verify', 'estimate', 'analyze',
                            'generate', 'export', 'approve', 'reject',
                            'archive', 'compare', 'merge')),
    actor_id            UUID,
    actor_type          TEXT            DEFAULT 'system',
    input_hash          TEXT,
    output_hash         TEXT,
    chain_hash          TEXT            NOT NULL,
    details             JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eudr_faal_tenant_entity
    ON eudr_forest_analysis_audit_log (tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_eudr_faal_event_id
    ON eudr_forest_analysis_audit_log (event_id);
CREATE INDEX IF NOT EXISTS idx_eudr_faal_chain_hash
    ON eudr_forest_analysis_audit_log (chain_hash);
CREATE INDEX IF NOT EXISTS idx_eudr_faal_created_at
    ON eudr_forest_analysis_audit_log (created_at DESC);

-- ---------------------------------------------------------------------------
-- Continuous Aggregates
-- ---------------------------------------------------------------------------

-- Daily forest verdicts summary
CREATE MATERIALIZED VIEW IF NOT EXISTS eudr_daily_forest_verdicts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', verified_at)    AS bucket,
    tenant_id,
    commodity,
    verdict,
    COUNT(*)                             AS verdict_count,
    AVG(confidence)                      AS avg_confidence,
    AVG(canopy_change_pct)               AS avg_canopy_change,
    COUNT(*) FILTER (WHERE requires_review) AS review_count
FROM eudr_deforestation_free_verdicts
GROUP BY bucket, tenant_id, commodity, verdict
WITH NO DATA;

SELECT add_continuous_aggregate_policy('eudr_daily_forest_verdicts',
    start_offset    => INTERVAL '7 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- Weekly analysis statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS eudr_weekly_analysis_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', analyzed_at)   AS bucket,
    tenant_id,
    method,
    density_class,
    COUNT(*)                             AS analysis_count,
    AVG(density_pct)                     AS avg_density,
    AVG(confidence)                      AS avg_confidence,
    COUNT(*) FILTER (WHERE meets_fao_threshold) AS fao_forest_count
FROM eudr_canopy_density_maps
GROUP BY bucket, tenant_id, method, density_class
WITH NO DATA;

SELECT add_continuous_aggregate_policy('eudr_weekly_analysis_stats',
    start_offset    => INTERVAL '14 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists   => TRUE
);

-- ---------------------------------------------------------------------------
-- Retention policies (5-year per EUDR Article 31)
-- ---------------------------------------------------------------------------
SELECT add_retention_policy('eudr_canopy_density_maps',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('eudr_forest_classifications',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('eudr_historical_reconstructions',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('eudr_deforestation_free_verdicts',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('eudr_canopy_height_estimates',
    drop_after => INTERVAL '5 years', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE eudr_canopy_density_maps IS 'AGENT-EUDR-004: Canopy density mapping results with FAO threshold checks';
COMMENT ON TABLE eudr_forest_classifications IS 'AGENT-EUDR-004: Forest type classification results (10 types)';
COMMENT ON TABLE eudr_historical_reconstructions IS 'AGENT-EUDR-004: Historical forest cover reconstruction at EUDR cutoff date';
COMMENT ON TABLE eudr_deforestation_free_verdicts IS 'AGENT-EUDR-004: Definitive deforestation-free verdicts with evidence';
COMMENT ON TABLE eudr_canopy_height_estimates IS 'AGENT-EUDR-004: Canopy height estimates from multi-source fusion';
COMMENT ON TABLE eudr_fragmentation_analyses IS 'AGENT-EUDR-004: Forest fragmentation metrics (6 landscape indices)';
COMMENT ON TABLE eudr_biomass_estimates IS 'AGENT-EUDR-004: Above-ground biomass estimates with carbon stock';
COMMENT ON TABLE eudr_forest_compliance_reports IS 'AGENT-EUDR-004: Generated EUDR compliance reports';
COMMENT ON TABLE eudr_forest_cover_baselines IS 'AGENT-EUDR-004: Forest cover baseline composites for cutoff date';
COMMENT ON TABLE eudr_forest_analysis_audit_log IS 'AGENT-EUDR-004: Immutable audit trail for all forest cover analyses';
