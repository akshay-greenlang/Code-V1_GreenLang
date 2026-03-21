-- =============================================================================
-- V275: PACK-035 Energy Benchmark Pack - Views, Indexes, RLS, Seed Data
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- Materialized views for dashboards, composite indexes for common query
-- patterns, pack-level audit trail, and seed data for benchmark sources,
-- building type mappings, default weather stations, and sample degree-day
-- structure.
--
-- Tables (1):
--   1. pack035_energy_benchmark.pack035_audit_trail
--
-- Materialized Views (3):
--   1. pack035_energy_benchmark.mv_facility_latest_eui
--   2. pack035_energy_benchmark.mv_portfolio_summary
--   3. pack035_energy_benchmark.mv_peer_group_stats
--
-- Views (1):
--   1. pack035_energy_benchmark.v_benchmark_dashboard
--
-- Previous: V274__pack035_energy_benchmark_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.pack035_audit_trail
-- =============================================================================
-- Pack-level audit trail logging all significant actions across
-- PACK-035 entities for compliance and provenance tracking.

CREATE TABLE pack035_energy_benchmark.pack035_audit_trail (
    audit_trail_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID,
    tenant_id               UUID,
    action                  VARCHAR(50)     NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID,
    actor                   TEXT            NOT NULL,
    actor_role              VARCHAR(50),
    ip_address              VARCHAR(45),
    old_values              JSONB,
    new_values              JSONB,
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_trail_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'READ', 'APPROVE', 'REJECT',
                   'SUBMIT', 'EXPORT', 'IMPORT', 'CALCULATE', 'VERIFY',
                   'ARCHIVE', 'RESTORE', 'LOCK', 'UNLOCK', 'BENCHMARK',
                   'COMPARE', 'NORMALISE', 'ALERT')
    )
);

-- Indexes
CREATE INDEX idx_p035_trail_facility     ON pack035_energy_benchmark.pack035_audit_trail(facility_id);
CREATE INDEX idx_p035_trail_tenant       ON pack035_energy_benchmark.pack035_audit_trail(tenant_id);
CREATE INDEX idx_p035_trail_action       ON pack035_energy_benchmark.pack035_audit_trail(action);
CREATE INDEX idx_p035_trail_entity       ON pack035_energy_benchmark.pack035_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p035_trail_actor        ON pack035_energy_benchmark.pack035_audit_trail(actor);
CREATE INDEX idx_p035_trail_created      ON pack035_energy_benchmark.pack035_audit_trail(created_at DESC);
CREATE INDEX idx_p035_trail_details      ON pack035_energy_benchmark.pack035_audit_trail USING GIN(details);

-- RLS
ALTER TABLE pack035_energy_benchmark.pack035_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_trail_tenant_isolation ON pack035_energy_benchmark.pack035_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_trail_service_bypass ON pack035_energy_benchmark.pack035_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Materialized View 1: mv_facility_latest_eui
-- =============================================================================
-- Latest EUI calculation per facility for dashboard rendering.
-- Refreshed periodically via application scheduler.

CREATE MATERIALIZED VIEW pack035_energy_benchmark.mv_facility_latest_eui AS
SELECT DISTINCT ON (fp.id)
    fp.id AS facility_id,
    fp.tenant_id,
    fp.facility_name,
    fp.building_type,
    fp.country_code,
    fp.climate_zone,
    fp.gross_internal_area_m2,
    eui.id AS eui_calculation_id,
    eui.calculation_period,
    eui.period_start,
    eui.period_end,
    eui.site_eui_kwh_m2,
    eui.source_eui_kwh_m2,
    eui.primary_eui_kwh_m2,
    eui.weather_normalised_eui,
    eui.co2_intensity_kg_m2,
    eui.data_completeness_pct,
    eui.calculated_at
FROM pack035_energy_benchmark.facility_profiles fp
LEFT JOIN pack035_energy_benchmark.eui_calculations eui
    ON eui.facility_id = fp.id
ORDER BY fp.id, eui.calculated_at DESC NULLS LAST
WITH NO DATA;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p035_mv_eui_facility ON pack035_energy_benchmark.mv_facility_latest_eui(facility_id);
CREATE INDEX idx_p035_mv_eui_tenant ON pack035_energy_benchmark.mv_facility_latest_eui(tenant_id);
CREATE INDEX idx_p035_mv_eui_btype ON pack035_energy_benchmark.mv_facility_latest_eui(building_type);
CREATE INDEX idx_p035_mv_eui_site ON pack035_energy_benchmark.mv_facility_latest_eui(site_eui_kwh_m2);

-- =============================================================================
-- Materialized View 2: mv_portfolio_summary
-- =============================================================================
-- Latest portfolio metrics summary for portfolio dashboard.

CREATE MATERIALIZED VIEW pack035_energy_benchmark.mv_portfolio_summary AS
SELECT DISTINCT ON (p.id)
    p.id AS portfolio_id,
    p.tenant_id,
    p.portfolio_name,
    p.portfolio_type,
    p.aggregation_method,
    pm.calculation_date,
    pm.total_facilities,
    pm.facilities_with_data,
    pm.data_coverage_pct,
    pm.total_floor_area_m2,
    pm.total_energy_kwh,
    pm.area_weighted_eui,
    pm.median_eui,
    pm.total_cost_eur,
    pm.total_co2_tonnes,
    pm.co2_intensity_kg_m2,
    pm.yoy_eui_change_pct,
    pm.yoy_co2_change_pct,
    pm.pct_top_quartile,
    pm.pct_bottom_quartile
FROM pack035_energy_benchmark.portfolios p
LEFT JOIN pack035_energy_benchmark.portfolio_metrics pm
    ON pm.portfolio_id = p.id
WHERE p.is_active = true
ORDER BY p.id, pm.calculation_date DESC NULLS LAST
WITH NO DATA;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p035_mv_ps_portfolio ON pack035_energy_benchmark.mv_portfolio_summary(portfolio_id);
CREATE INDEX idx_p035_mv_ps_tenant ON pack035_energy_benchmark.mv_portfolio_summary(tenant_id);
CREATE INDEX idx_p035_mv_ps_type ON pack035_energy_benchmark.mv_portfolio_summary(portfolio_type);

-- =============================================================================
-- Materialized View 3: mv_peer_group_stats
-- =============================================================================
-- Pre-computed peer group statistics for comparison lookups.

CREATE MATERIALIZED VIEW pack035_energy_benchmark.mv_peer_group_stats AS
SELECT
    pg.id AS peer_group_id,
    pg.name AS peer_group_name,
    pg.building_type,
    pg.climate_zone,
    pg.country_code,
    pg.sample_size,
    pg.mean_eui,
    pg.median_eui,
    pg.std_dev,
    pg.p10,
    pg.p25,
    pg.p50,
    pg.p75,
    pg.p90,
    pg.source,
    pg.source_year,
    pg.is_custom,
    COUNT(pcr.id) AS total_comparisons,
    AVG(pcr.percentile_rank) AS avg_percentile,
    MIN(pcr.facility_eui) AS min_facility_eui,
    MAX(pcr.facility_eui) AS max_facility_eui
FROM pack035_energy_benchmark.peer_groups pg
LEFT JOIN pack035_energy_benchmark.peer_comparison_results pcr
    ON pcr.peer_group_id = pg.id
WHERE pg.is_active = true
GROUP BY pg.id, pg.name, pg.building_type, pg.climate_zone, pg.country_code,
         pg.sample_size, pg.mean_eui, pg.median_eui, pg.std_dev,
         pg.p10, pg.p25, pg.p50, pg.p75, pg.p90,
         pg.source, pg.source_year, pg.is_custom
WITH NO DATA;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p035_mv_pgs_group ON pack035_energy_benchmark.mv_peer_group_stats(peer_group_id);
CREATE INDEX idx_p035_mv_pgs_btype ON pack035_energy_benchmark.mv_peer_group_stats(building_type);
CREATE INDEX idx_p035_mv_pgs_climate ON pack035_energy_benchmark.mv_peer_group_stats(climate_zone);
CREATE INDEX idx_p035_mv_pgs_country ON pack035_energy_benchmark.mv_peer_group_stats(country_code);

-- =============================================================================
-- View: v_benchmark_dashboard
-- =============================================================================
-- Consolidated benchmark dashboard combining facility profile, latest EUI,
-- latest peer comparison, latest performance rating, and gap analysis.

CREATE OR REPLACE VIEW pack035_energy_benchmark.v_benchmark_dashboard AS
SELECT
    fp.id AS facility_id,
    fp.tenant_id,
    fp.facility_name,
    fp.building_type,
    fp.country_code,
    fp.city,
    fp.climate_zone,
    fp.gross_internal_area_m2,
    fp.year_built,
    -- Latest EUI
    eui.site_eui_kwh_m2,
    eui.source_eui_kwh_m2,
    eui.weather_normalised_eui,
    eui.co2_intensity_kg_m2 AS eui_co2_intensity,
    eui.calculated_at AS eui_date,
    -- Latest peer comparison
    pcr.percentile_rank,
    pcr.quartile_band,
    pcr.energy_star_score,
    pcr.z_score,
    pcr.distance_to_median,
    pcr.compared_at AS comparison_date,
    -- Latest performance rating
    pr.rating_system,
    pr.rating_value,
    pr.numeric_score AS rating_score,
    pr.valid_until AS rating_expiry,
    -- Latest gap analysis
    ga.overall_gap_pct,
    ga.overall_gap_kwh_m2,
    ga.benchmark_target_eui,
    ga.analysis_date AS gap_analysis_date,
    -- CRREM
    crrem.stranding_year,
    crrem.stranding_risk,
    crrem.annual_reduction_needed_pct AS crrem_reduction_needed
FROM pack035_energy_benchmark.facility_profiles fp
LEFT JOIN LATERAL (
    SELECT site_eui_kwh_m2, source_eui_kwh_m2, weather_normalised_eui,
           co2_intensity_kg_m2, calculated_at
    FROM pack035_energy_benchmark.eui_calculations
    WHERE facility_id = fp.id
    ORDER BY calculated_at DESC
    LIMIT 1
) eui ON TRUE
LEFT JOIN LATERAL (
    SELECT percentile_rank, quartile_band, energy_star_score, z_score,
           distance_to_median, compared_at
    FROM pack035_energy_benchmark.peer_comparison_results
    WHERE facility_id = fp.id
    ORDER BY compared_at DESC
    LIMIT 1
) pcr ON TRUE
LEFT JOIN LATERAL (
    SELECT rating_system, rating_value, numeric_score, valid_until
    FROM pack035_energy_benchmark.performance_ratings
    WHERE facility_id = fp.id AND is_current = true
    ORDER BY rating_date DESC
    LIMIT 1
) pr ON TRUE
LEFT JOIN LATERAL (
    SELECT overall_gap_pct, overall_gap_kwh_m2, benchmark_target_eui, analysis_date
    FROM pack035_energy_benchmark.gap_analyses
    WHERE facility_id = fp.id AND status = 'APPROVED'
    ORDER BY analysis_date DESC
    LIMIT 1
) ga ON TRUE
LEFT JOIN LATERAL (
    SELECT stranding_year, stranding_risk, annual_reduction_needed_pct
    FROM pack035_energy_benchmark.crrem_assessments
    WHERE facility_id = fp.id
    ORDER BY assessment_date DESC
    LIMIT 1
) crrem ON TRUE;

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Facility + building type + country for filtered benchmarking
CREATE INDEX idx_p035_fp_type_country ON pack035_energy_benchmark.facility_profiles(building_type, country_code);

-- EUI by facility + period for time-series queries
CREATE INDEX idx_p035_eui_fac_period ON pack035_energy_benchmark.eui_calculations(facility_id, period_start DESC);

-- Consumption by facility + period for aggregation
CREATE INDEX idx_p035_ecr_fac_period ON pack035_energy_benchmark.energy_consumption_records(facility_id, period_start, period_end);

-- Alerts by facility + unacknowledged for operational views
CREATE INDEX idx_p035_pa_fac_sev_unack ON pack035_energy_benchmark.performance_alerts(facility_id, severity)
    WHERE acknowledged = false AND resolved = false;

-- Gap analyses by facility + approved for dashboard
CREATE INDEX idx_p035_ga_fac_approved ON pack035_energy_benchmark.gap_analyses(facility_id)
    WHERE status = 'APPROVED';

-- Improvement targets by facility + active
CREATE INDEX idx_p035_it_fac_active ON pack035_energy_benchmark.improvement_targets(facility_id, target_year)
    WHERE status IN ('APPROVED', 'IN_PROGRESS');

-- =============================================================================
-- Seed Data: Benchmark Sources
-- =============================================================================

INSERT INTO pack035_energy_benchmark.benchmark_sources (source_name, source_code, source_version, publisher, effective_date, country_scope, description, data_vintage_year, is_active) VALUES
    ('ENERGY STAR Portfolio Manager', 'ENERGY_STAR', '2024.1', 'U.S. Environmental Protection Agency', '2024-01-01', 'US', 'U.S. commercial building energy benchmarking using source EUI and 1-100 score. Covers 80+ building types. National median data from CBECS.', 2024, true),
    ('CIBSE TM46 Energy Benchmarks', 'CIBSE_TM46', '2008', 'Chartered Institution of Building Services Engineers', '2008-01-01', 'GB', 'UK building energy benchmarks by category. Provides typical and good practice EUI values for Display Energy Certificates (DEC). 29 building categories.', 2008, true),
    ('DIN V 18599 Reference Values', 'DIN_V_18599', '2018', 'Deutsches Institut fur Normung', '2018-06-01', 'DE', 'German standard for energy assessment of buildings. Reference building method for EnEV/GEG compliance. Primary energy and CO2 benchmarks.', 2018, true),
    ('BPIE Building Database', 'BPIE', '2023', 'Buildings Performance Institute Europe', '2023-01-01', 'EU', 'Pan-European building stock energy performance data. Cross-country comparisons of residential and non-residential buildings. EPBD alignment.', 2023, true),
    ('TABULA/EPISCOPE Residential', 'TABULA', '2016', 'Institut Wohnen und Umwelt GmbH', '2016-01-01', 'EU', 'European residential building typology with energy benchmarks by country, building age, and type. Covers 20+ EU countries.', 2016, true),
    ('NABERS Energy Rating', 'NABERS', '2024', 'National Australian Built Environment Rating System', '2024-01-01', 'AU', 'Australian building energy rating system (1-6 stars). Office, shopping centre, hotel, and data centre benchmarks.', 2024, true),
    ('EU Energy Efficiency Directive Benchmarks', 'EU_EED', '2023', 'European Commission', '2023-10-01', 'EU', 'Benchmarks from EU EED Article 8 audit data aggregation. Industrial and commercial sector energy intensity references.', 2023, true),
    ('ASHRAE Standard 100', 'ASHRAE_100', '2018', 'ASHRAE', '2018-01-01', 'US', 'Energy efficiency in existing buildings standard. EUI targets by building type and climate zone for ASHRAE Building EQ.', 2018, true);

-- =============================================================================
-- Seed Data: Sample Building Type Mappings (CIBSE TM46)
-- =============================================================================

INSERT INTO pack035_energy_benchmark.benchmark_building_types (source_id, source_building_type, source_type_code, gl_building_type, description) VALUES
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'General Office', 'TM46-01', 'OFFICE', 'Typical naturally ventilated or mechanically ventilated office'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'High Street Agency', 'TM46-02', 'RETAIL', 'Street-level retail premises'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'General Retail', 'TM46-03', 'RETAIL', 'General retail/department store'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Large Non-food Shop', 'TM46-04', 'RETAIL', 'Large format non-food retail'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Small Food Store', 'TM46-05', 'RETAIL', 'Convenience/small grocery store'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Warehouse', 'TM46-06', 'WAREHOUSE', 'Distribution and storage warehouse'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'General Hospital', 'TM46-07', 'HEALTHCARE', 'Acute general hospital'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Schools and Seasonal', 'TM46-08', 'EDUCATION', 'Primary and secondary schools'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'University Campus', 'TM46-09', 'EDUCATION', 'University/college campus buildings'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Hotel', 'TM46-10', 'HOTEL', 'Full service hotel'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Restaurant/Pub/Bar', 'TM46-11', 'RESTAURANT', 'Restaurant, pub, bar, or cafe'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Entertainment Hall', 'TM46-12', 'ENTERTAINMENT', 'Cinema, theatre, concert hall'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Fitness/Sports Centre', 'TM46-13', 'SPORTS', 'Leisure/sports centre, gym'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'CIBSE_TM46'), 'Library/Museum', 'TM46-14', 'LIBRARY', 'Library, museum, gallery');

-- =============================================================================
-- Seed Data: Sample Building Type Mappings (ENERGY STAR)
-- =============================================================================

INSERT INTO pack035_energy_benchmark.benchmark_building_types (source_id, source_building_type, source_type_code, gl_building_type, description) VALUES
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'Office', 'ES-OFFICE', 'OFFICE', 'Office building (all sizes)'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'Retail Store', 'ES-RETAIL', 'RETAIL', 'Enclosed retail store'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'Warehouse (Refrigerated)', 'ES-WHSE-R', 'WAREHOUSE', 'Refrigerated or non-refrigerated warehouse'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'Hospital (Acute Care)', 'ES-HOSP', 'HEALTHCARE', 'Acute care hospital'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'K-12 School', 'ES-K12', 'EDUCATION', 'K-12 school building'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'Data Center', 'ES-DC', 'DATA_CENTER', 'Data center facility'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'Hotel', 'ES-HOTEL', 'HOTEL', 'Hotel (all service levels)'),
    ((SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code = 'ENERGY_STAR'), 'Multifamily Housing', 'ES-MFH', 'RESIDENTIAL_MULTIFAMILY', 'Multifamily residential building');

-- =============================================================================
-- Seed Data: Default Weather Stations (Major EU Cities)
-- =============================================================================

INSERT INTO pack035_energy_benchmark.weather_stations (station_id, station_name, country_code, city, latitude, longitude, altitude_m, data_source, active) VALUES
    ('EGLL', 'London Heathrow', 'GB', 'London', 51.4706, -0.4619, 25.0, 'NOAA_ISD', true),
    ('LFPG', 'Paris Charles de Gaulle', 'FR', 'Paris', 49.0128, 2.5490, 119.0, 'NOAA_ISD', true),
    ('EDDB', 'Berlin Brandenburg', 'DE', 'Berlin', 52.3800, 13.5225, 48.0, 'NOAA_ISD', true),
    ('EHAM', 'Amsterdam Schiphol', 'NL', 'Amsterdam', 52.3086, 4.7639, -3.0, 'NOAA_ISD', true),
    ('LEMD', 'Madrid Barajas', 'ES', 'Madrid', 40.4719, -3.5626, 609.0, 'NOAA_ISD', true),
    ('LIRF', 'Rome Fiumicino', 'IT', 'Rome', 41.8003, 12.2389, 5.0, 'NOAA_ISD', true),
    ('LOWW', 'Vienna Schwechat', 'AT', 'Vienna', 48.1103, 16.5697, 183.0, 'NOAA_ISD', true),
    ('EKCH', 'Copenhagen Kastrup', 'DK', 'Copenhagen', 55.6180, 12.6561, 5.0, 'NOAA_ISD', true),
    ('ESSB', 'Stockholm Bromma', 'SE', 'Stockholm', 59.3544, 17.9414, 14.0, 'NOAA_ISD', true),
    ('EFHK', 'Helsinki Vantaa', 'FI', 'Helsinki', 60.3172, 24.9633, 56.0, 'NOAA_ISD', true),
    ('LPPT', 'Lisbon Portela', 'PT', 'Lisbon', 38.7756, -9.1354, 114.0, 'NOAA_ISD', true),
    ('EPWA', 'Warsaw Chopin', 'PL', 'Warsaw', 52.1657, 20.9671, 110.0, 'NOAA_ISD', true),
    ('EIDW', 'Dublin Airport', 'IE', 'Dublin', 53.4213, -6.2700, 74.0, 'NOAA_ISD', true),
    ('LSZH', 'Zurich Kloten', 'CH', 'Zurich', 47.4647, 8.5492, 432.0, 'NOAA_ISD', true),
    ('EBBR', 'Brussels National', 'BE', 'Brussels', 50.9014, 4.4844, 56.0, 'NOAA_ISD', true);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT ON pack035_energy_benchmark.pack035_audit_trail TO PUBLIC;
GRANT SELECT ON pack035_energy_benchmark.mv_facility_latest_eui TO PUBLIC;
GRANT SELECT ON pack035_energy_benchmark.mv_portfolio_summary TO PUBLIC;
GRANT SELECT ON pack035_energy_benchmark.mv_peer_group_stats TO PUBLIC;
GRANT SELECT ON pack035_energy_benchmark.v_benchmark_dashboard TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.pack035_audit_trail IS
    'Pack-level audit trail logging all significant actions across PACK-035 entities for compliance and provenance.';

COMMENT ON MATERIALIZED VIEW pack035_energy_benchmark.mv_facility_latest_eui IS
    'Latest EUI calculation per facility for dashboard rendering. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY pack035_energy_benchmark.mv_facility_latest_eui;';
COMMENT ON MATERIALIZED VIEW pack035_energy_benchmark.mv_portfolio_summary IS
    'Latest portfolio metrics summary for portfolio dashboard. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY pack035_energy_benchmark.mv_portfolio_summary;';
COMMENT ON MATERIALIZED VIEW pack035_energy_benchmark.mv_peer_group_stats IS
    'Pre-computed peer group statistics for comparison lookups. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY pack035_energy_benchmark.mv_peer_group_stats;';
COMMENT ON VIEW pack035_energy_benchmark.v_benchmark_dashboard IS
    'Consolidated benchmark dashboard: facility profile + latest EUI + peer comparison + performance rating + gap analysis + CRREM.';
