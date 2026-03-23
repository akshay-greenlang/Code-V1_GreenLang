-- =============================================================================
-- V296: PACK-038 Peak Shaving Pack - Load Profiles & Interval Data
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the pack038_peak_shaving schema and foundational tables for
-- facility load profile analysis. Tracks interval-level demand data,
-- load statistics, day-type clustering, and load duration curves used
-- by downstream peak event detection and demand charge optimization.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_load_profiles
--   2. pack038_peak_shaving.ps_interval_data
--   3. pack038_peak_shaving.ps_load_statistics
--   4. pack038_peak_shaving.ps_day_type_clusters
--   5. pack038_peak_shaving.ps_load_duration_curves
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V295__pack037_demand_response_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack038_peak_shaving;

SET search_path TO pack038_peak_shaving, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack038_peak_shaving.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_load_profiles
-- =============================================================================
-- Facility-level load profile configuration capturing meter association,
-- analysis period, baseline demand characteristics, and profile status.
-- Each facility has one or more load profiles that aggregate interval data
-- for peak shaving analysis.

CREATE TABLE pack038_peak_shaving.ps_load_profiles (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    facility_name           VARCHAR(255)    NOT NULL,
    meter_id                VARCHAR(100),
    meter_type              VARCHAR(30)     NOT NULL DEFAULT 'INTERVAL',
    utility_name            VARCHAR(255),
    utility_account_number  VARCHAR(100),
    rate_schedule            VARCHAR(100),
    iso_rto_region          VARCHAR(30),
    country_code            CHAR(2)         NOT NULL DEFAULT 'US',
    timezone                VARCHAR(50)     NOT NULL DEFAULT 'America/New_York',
    latitude                NUMERIC(10,7),
    longitude               NUMERIC(10,7),
    facility_type           VARCHAR(50),
    analysis_start_date     DATE,
    analysis_end_date       DATE,
    peak_demand_kw          NUMERIC(12,3),
    average_demand_kw       NUMERIC(12,3),
    base_load_kw            NUMERIC(12,3),
    load_factor             NUMERIC(5,4),
    annual_energy_kwh       NUMERIC(15,3),
    billing_demand_kw       NUMERIC(12,3),
    contract_demand_kw      NUMERIC(12,3),
    voltage_level           VARCHAR(30),
    service_type            VARCHAR(30)     NOT NULL DEFAULT 'COMMERCIAL',
    profile_status          VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    data_quality_score      NUMERIC(5,2),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p038_lp_meter_type CHECK (
        meter_type IN (
            'INTERVAL', 'SMART_METER', 'SCADA', 'PULSE', 'MANUAL_READ', 'CT_LOGGER'
        )
    ),
    CONSTRAINT chk_p038_lp_region CHECK (
        iso_rto_region IS NULL OR iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'DE_AMPRION', 'DE_50HZ', 'DE_TRANSNET',
            'FR_RTE', 'NL_TENNET', 'ES_REE', 'IT_TERNA', 'AU_AEMO',
            'JP_TEPCO', 'JP_KEPCO', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_lp_facility_type CHECK (
        facility_type IS NULL OR facility_type IN (
            'OFFICE', 'RETAIL', 'WAREHOUSE', 'MANUFACTURING', 'DATA_CENTER',
            'HOSPITAL', 'UNIVERSITY', 'HOTEL', 'COLD_STORAGE', 'WATER_TREATMENT',
            'PUMPING_STATION', 'EV_CHARGING', 'RESIDENTIAL_AGGREGATED',
            'MIXED_USE', 'CAMPUS', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_lp_service_type CHECK (
        service_type IN (
            'RESIDENTIAL', 'COMMERCIAL', 'INDUSTRIAL', 'AGRICULTURAL', 'MUNICIPAL'
        )
    ),
    CONSTRAINT chk_p038_lp_voltage CHECK (
        voltage_level IS NULL OR voltage_level IN (
            'LOW_VOLTAGE', 'MEDIUM_VOLTAGE', 'HIGH_VOLTAGE', 'TRANSMISSION',
            '120V', '208V', '240V', '277V', '480V', '4KV', '13KV', '34KV', '69KV'
        )
    ),
    CONSTRAINT chk_p038_lp_status CHECK (
        profile_status IN (
            'DRAFT', 'COLLECTING', 'COMPLETE', 'VALIDATED', 'ARCHIVED', 'ERROR'
        )
    ),
    CONSTRAINT chk_p038_lp_peak CHECK (
        peak_demand_kw IS NULL OR peak_demand_kw > 0
    ),
    CONSTRAINT chk_p038_lp_average CHECK (
        average_demand_kw IS NULL OR average_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_lp_base CHECK (
        base_load_kw IS NULL OR base_load_kw >= 0
    ),
    CONSTRAINT chk_p038_lp_load_factor CHECK (
        load_factor IS NULL OR (load_factor >= 0 AND load_factor <= 1.0)
    ),
    CONSTRAINT chk_p038_lp_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p038_lp_latitude CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    ),
    CONSTRAINT chk_p038_lp_longitude CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    ),
    CONSTRAINT chk_p038_lp_dates CHECK (
        analysis_start_date IS NULL OR analysis_end_date IS NULL OR
        analysis_start_date <= analysis_end_date
    ),
    CONSTRAINT uq_p038_lp_tenant_facility UNIQUE (tenant_id, facility_id, meter_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_lp_tenant          ON pack038_peak_shaving.ps_load_profiles(tenant_id);
CREATE INDEX idx_p038_lp_facility        ON pack038_peak_shaving.ps_load_profiles(facility_id);
CREATE INDEX idx_p038_lp_meter           ON pack038_peak_shaving.ps_load_profiles(meter_id);
CREATE INDEX idx_p038_lp_region          ON pack038_peak_shaving.ps_load_profiles(iso_rto_region);
CREATE INDEX idx_p038_lp_status          ON pack038_peak_shaving.ps_load_profiles(profile_status);
CREATE INDEX idx_p038_lp_facility_type   ON pack038_peak_shaving.ps_load_profiles(facility_type);
CREATE INDEX idx_p038_lp_peak            ON pack038_peak_shaving.ps_load_profiles(peak_demand_kw DESC);
CREATE INDEX idx_p038_lp_created         ON pack038_peak_shaving.ps_load_profiles(created_at DESC);
CREATE INDEX idx_p038_lp_metadata        ON pack038_peak_shaving.ps_load_profiles USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_lp_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_load_profiles
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_interval_data
-- =============================================================================
-- High-resolution interval demand data (15/30/60-minute intervals) for
-- each load profile. This is the primary time-series table used for peak
-- detection, load analysis, and dispatch simulation. Configured as a
-- TimescaleDB hypertable for efficient time-series queries.

CREATE TABLE pack038_peak_shaving.ps_interval_data (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    kw_demand               NUMERIC(12,3)   NOT NULL,
    kvar_demand              NUMERIC(12,3),
    kva_demand              NUMERIC(12,3),
    power_factor            NUMERIC(5,4),
    voltage_v               NUMERIC(10,3),
    current_a               NUMERIC(10,3),
    energy_kwh              NUMERIC(12,3),
    reactive_energy_kvarh   NUMERIC(12,3),
    interval_length_minutes INTEGER         NOT NULL DEFAULT 15,
    temperature_f           NUMERIC(6,2),
    humidity_pct            NUMERIC(5,2),
    occupancy_pct           NUMERIC(5,2),
    is_peak_period          BOOLEAN         DEFAULT false,
    is_weekend              BOOLEAN         DEFAULT false,
    is_holiday              BOOLEAN         DEFAULT false,
    data_quality            VARCHAR(20)     NOT NULL DEFAULT 'MEASURED',
    source_system           VARCHAR(100),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_id_interval CHECK (
        interval_length_minutes IN (1, 5, 15, 30, 60)
    ),
    CONSTRAINT chk_p038_id_kw CHECK (
        kw_demand >= 0
    ),
    CONSTRAINT chk_p038_id_kvar CHECK (
        kvar_demand IS NULL OR kvar_demand >= -99999.999
    ),
    CONSTRAINT chk_p038_id_kva CHECK (
        kva_demand IS NULL OR kva_demand >= 0
    ),
    CONSTRAINT chk_p038_id_pf CHECK (
        power_factor IS NULL OR (power_factor >= 0 AND power_factor <= 1.0)
    ),
    CONSTRAINT chk_p038_id_voltage CHECK (
        voltage_v IS NULL OR voltage_v >= 0
    ),
    CONSTRAINT chk_p038_id_current CHECK (
        current_a IS NULL OR current_a >= 0
    ),
    CONSTRAINT chk_p038_id_energy CHECK (
        energy_kwh IS NULL OR energy_kwh >= 0
    ),
    CONSTRAINT chk_p038_id_humidity CHECK (
        humidity_pct IS NULL OR (humidity_pct >= 0 AND humidity_pct <= 100)
    ),
    CONSTRAINT chk_p038_id_occupancy CHECK (
        occupancy_pct IS NULL OR (occupancy_pct >= 0 AND occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p038_id_quality CHECK (
        data_quality IN ('MEASURED', 'ESTIMATED', 'INTERPOLATED', 'MISSING', 'VALIDATED')
    ),
    CONSTRAINT uq_p038_id_profile_timestamp UNIQUE (profile_id, timestamp)
);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable (skip if extension not available)
-- ---------------------------------------------------------------------------
-- SELECT create_hypertable('pack038_peak_shaving.ps_interval_data', 'timestamp',
--     chunk_time_interval => INTERVAL '1 month',
--     if_not_exists => TRUE
-- );

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_id_tenant          ON pack038_peak_shaving.ps_interval_data(tenant_id);
CREATE INDEX idx_p038_id_profile         ON pack038_peak_shaving.ps_interval_data(profile_id);
CREATE INDEX idx_p038_id_timestamp       ON pack038_peak_shaving.ps_interval_data(timestamp DESC);
CREATE INDEX idx_p038_id_profile_ts      ON pack038_peak_shaving.ps_interval_data(tenant_id, profile_id, timestamp DESC);
CREATE INDEX idx_p038_id_kw              ON pack038_peak_shaving.ps_interval_data(kw_demand DESC);
CREATE INDEX idx_p038_id_peak_period     ON pack038_peak_shaving.ps_interval_data(is_peak_period) WHERE is_peak_period = true;
CREATE INDEX idx_p038_id_quality         ON pack038_peak_shaving.ps_interval_data(data_quality);
CREATE INDEX idx_p038_id_created         ON pack038_peak_shaving.ps_interval_data(created_at DESC);

-- Composite: profile + peak flag + demand for peak detection
CREATE INDEX idx_p038_id_profile_peak_kw ON pack038_peak_shaving.ps_interval_data(profile_id, kw_demand DESC)
    WHERE is_peak_period = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_id_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_interval_data
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_load_statistics
-- =============================================================================
-- Pre-computed statistical summaries of load data for each profile and
-- period. Includes peak, average, base load, percentiles, and distribution
-- metrics used for rapid dashboard rendering and peak detection tuning.

CREATE TABLE pack038_peak_shaving.ps_load_statistics (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    period_type             VARCHAR(20)     NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    peak_kw                 NUMERIC(12,3)   NOT NULL,
    peak_timestamp          TIMESTAMPTZ,
    average_kw              NUMERIC(12,3)   NOT NULL,
    median_kw               NUMERIC(12,3),
    min_kw                  NUMERIC(12,3),
    base_load_kw            NUMERIC(12,3),
    load_factor             NUMERIC(5,4),
    demand_variability      NUMERIC(8,4),
    std_deviation_kw        NUMERIC(12,3),
    p95_kw                  NUMERIC(12,3),
    p99_kw                  NUMERIC(12,3),
    p75_kw                  NUMERIC(12,3),
    p25_kw                  NUMERIC(12,3),
    on_peak_avg_kw          NUMERIC(12,3),
    off_peak_avg_kw         NUMERIC(12,3),
    shoulder_avg_kw         NUMERIC(12,3),
    total_energy_kwh        NUMERIC(15,3),
    interval_count          INTEGER         NOT NULL DEFAULT 0,
    missing_intervals       INTEGER         DEFAULT 0,
    data_completeness_pct   NUMERIC(5,2),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ls_period_type CHECK (
        period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'BILLING_PERIOD', 'ANNUAL', 'CUSTOM')
    ),
    CONSTRAINT chk_p038_ls_peak CHECK (
        peak_kw >= 0
    ),
    CONSTRAINT chk_p038_ls_average CHECK (
        average_kw >= 0
    ),
    CONSTRAINT chk_p038_ls_load_factor CHECK (
        load_factor IS NULL OR (load_factor >= 0 AND load_factor <= 1.0)
    ),
    CONSTRAINT chk_p038_ls_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p038_ls_dates CHECK (
        period_start <= period_end
    ),
    CONSTRAINT chk_p038_ls_intervals CHECK (
        interval_count >= 0 AND (missing_intervals IS NULL OR missing_intervals >= 0)
    ),
    CONSTRAINT uq_p038_ls_profile_period UNIQUE (profile_id, period_type, period_start, period_end)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ls_profile         ON pack038_peak_shaving.ps_load_statistics(profile_id);
CREATE INDEX idx_p038_ls_tenant          ON pack038_peak_shaving.ps_load_statistics(tenant_id);
CREATE INDEX idx_p038_ls_period_type     ON pack038_peak_shaving.ps_load_statistics(period_type);
CREATE INDEX idx_p038_ls_period_start    ON pack038_peak_shaving.ps_load_statistics(period_start DESC);
CREATE INDEX idx_p038_ls_peak_kw         ON pack038_peak_shaving.ps_load_statistics(peak_kw DESC);
CREATE INDEX idx_p038_ls_load_factor     ON pack038_peak_shaving.ps_load_statistics(load_factor);
CREATE INDEX idx_p038_ls_created         ON pack038_peak_shaving.ps_load_statistics(created_at DESC);

-- Composite: profile + monthly for billing period lookups
CREATE INDEX idx_p038_ls_profile_monthly ON pack038_peak_shaving.ps_load_statistics(profile_id, period_start DESC)
    WHERE period_type = 'MONTHLY';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ls_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_load_statistics
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_day_type_clusters
-- =============================================================================
-- Clustered day-type patterns identified from interval data. Groups similar
-- load shape days (e.g., high cooling weekday, low occupancy weekend) for
-- peak prediction and dispatch schedule optimization.

CREATE TABLE pack038_peak_shaving.ps_day_type_clusters (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    cluster_name            VARCHAR(100)    NOT NULL,
    cluster_index           INTEGER         NOT NULL,
    cluster_method          VARCHAR(30)     NOT NULL DEFAULT 'KMEANS',
    day_count               INTEGER         NOT NULL DEFAULT 0,
    avg_peak_kw             NUMERIC(12,3),
    avg_daily_kwh           NUMERIC(12,3),
    avg_load_factor         NUMERIC(5,4),
    peak_hour               INTEGER,
    valley_hour             INTEGER,
    typical_load_shape      JSONB,
    representative_date     DATE,
    season_label            VARCHAR(20),
    day_type_label          VARCHAR(20),
    temperature_range_low_f NUMERIC(6,2),
    temperature_range_high_f NUMERIC(6,2),
    occupancy_typical_pct   NUMERIC(5,2),
    centroid_vector         JSONB,
    silhouette_score        NUMERIC(5,4),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_dc_method CHECK (
        cluster_method IN ('KMEANS', 'DBSCAN', 'HIERARCHICAL', 'GMM', 'MANUAL')
    ),
    CONSTRAINT chk_p038_dc_index CHECK (
        cluster_index >= 0
    ),
    CONSTRAINT chk_p038_dc_day_count CHECK (
        day_count >= 0
    ),
    CONSTRAINT chk_p038_dc_peak_hour CHECK (
        peak_hour IS NULL OR (peak_hour >= 0 AND peak_hour <= 23)
    ),
    CONSTRAINT chk_p038_dc_valley_hour CHECK (
        valley_hour IS NULL OR (valley_hour >= 0 AND valley_hour <= 23)
    ),
    CONSTRAINT chk_p038_dc_season CHECK (
        season_label IS NULL OR season_label IN ('SUMMER', 'WINTER', 'SHOULDER', 'SPRING', 'FALL')
    ),
    CONSTRAINT chk_p038_dc_day_type CHECK (
        day_type_label IS NULL OR day_type_label IN ('WEEKDAY', 'WEEKEND', 'HOLIDAY', 'MIXED')
    ),
    CONSTRAINT chk_p038_dc_silhouette CHECK (
        silhouette_score IS NULL OR (silhouette_score >= -1 AND silhouette_score <= 1)
    ),
    CONSTRAINT chk_p038_dc_load_factor CHECK (
        avg_load_factor IS NULL OR (avg_load_factor >= 0 AND avg_load_factor <= 1.0)
    ),
    CONSTRAINT uq_p038_dc_profile_cluster UNIQUE (profile_id, cluster_method, cluster_index)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_dc_profile         ON pack038_peak_shaving.ps_day_type_clusters(profile_id);
CREATE INDEX idx_p038_dc_tenant          ON pack038_peak_shaving.ps_day_type_clusters(tenant_id);
CREATE INDEX idx_p038_dc_method          ON pack038_peak_shaving.ps_day_type_clusters(cluster_method);
CREATE INDEX idx_p038_dc_season          ON pack038_peak_shaving.ps_day_type_clusters(season_label);
CREATE INDEX idx_p038_dc_day_type        ON pack038_peak_shaving.ps_day_type_clusters(day_type_label);
CREATE INDEX idx_p038_dc_avg_peak        ON pack038_peak_shaving.ps_day_type_clusters(avg_peak_kw DESC);
CREATE INDEX idx_p038_dc_created         ON pack038_peak_shaving.ps_day_type_clusters(created_at DESC);
CREATE INDEX idx_p038_dc_shape           ON pack038_peak_shaving.ps_day_type_clusters USING GIN(typical_load_shape);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_dc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_day_type_clusters
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_load_duration_curves
-- =============================================================================
-- Load duration curve data points representing the percentage of time that
-- demand exceeds each level. Used for sizing BESS, evaluating peak shaving
-- potential, and calculating demand charge savings at different thresholds.

CREATE TABLE pack038_peak_shaving.ps_load_duration_curves (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    curve_period_start      DATE            NOT NULL,
    curve_period_end        DATE            NOT NULL,
    period_type             VARCHAR(20)     NOT NULL DEFAULT 'ANNUAL',
    demand_threshold_kw     NUMERIC(12,3)   NOT NULL,
    hours_above             NUMERIC(10,3)   NOT NULL,
    pct_time_above          NUMERIC(7,4)    NOT NULL,
    energy_above_kwh        NUMERIC(15,3),
    energy_below_kwh        NUMERIC(15,3),
    peak_shaving_potential_kwh NUMERIC(15,3),
    intervals_above         INTEGER,
    cumulative_pct          NUMERIC(7,4),
    bin_index               INTEGER,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ldc_period_type CHECK (
        period_type IN ('MONTHLY', 'BILLING_PERIOD', 'SEASONAL', 'ANNUAL', 'CUSTOM')
    ),
    CONSTRAINT chk_p038_ldc_threshold CHECK (
        demand_threshold_kw >= 0
    ),
    CONSTRAINT chk_p038_ldc_hours CHECK (
        hours_above >= 0
    ),
    CONSTRAINT chk_p038_ldc_pct CHECK (
        pct_time_above >= 0 AND pct_time_above <= 100
    ),
    CONSTRAINT chk_p038_ldc_dates CHECK (
        curve_period_start <= curve_period_end
    ),
    CONSTRAINT chk_p038_ldc_cumulative CHECK (
        cumulative_pct IS NULL OR (cumulative_pct >= 0 AND cumulative_pct <= 100)
    ),
    CONSTRAINT chk_p038_ldc_intervals CHECK (
        intervals_above IS NULL OR intervals_above >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ldc_profile        ON pack038_peak_shaving.ps_load_duration_curves(profile_id);
CREATE INDEX idx_p038_ldc_tenant         ON pack038_peak_shaving.ps_load_duration_curves(tenant_id);
CREATE INDEX idx_p038_ldc_period_start   ON pack038_peak_shaving.ps_load_duration_curves(curve_period_start DESC);
CREATE INDEX idx_p038_ldc_threshold      ON pack038_peak_shaving.ps_load_duration_curves(demand_threshold_kw);
CREATE INDEX idx_p038_ldc_pct_time       ON pack038_peak_shaving.ps_load_duration_curves(pct_time_above);
CREATE INDEX idx_p038_ldc_created        ON pack038_peak_shaving.ps_load_duration_curves(created_at DESC);

-- Composite: profile + period for curve rendering
CREATE INDEX idx_p038_ldc_profile_period ON pack038_peak_shaving.ps_load_duration_curves(profile_id, curve_period_start, demand_threshold_kw);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ldc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_load_duration_curves
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_load_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_interval_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_load_statistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_day_type_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_load_duration_curves ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_lp_tenant_isolation
    ON pack038_peak_shaving.ps_load_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_lp_service_bypass
    ON pack038_peak_shaving.ps_load_profiles
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_id_tenant_isolation
    ON pack038_peak_shaving.ps_interval_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_id_service_bypass
    ON pack038_peak_shaving.ps_interval_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ls_tenant_isolation
    ON pack038_peak_shaving.ps_load_statistics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ls_service_bypass
    ON pack038_peak_shaving.ps_load_statistics
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_dc_tenant_isolation
    ON pack038_peak_shaving.ps_day_type_clusters
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_dc_service_bypass
    ON pack038_peak_shaving.ps_day_type_clusters
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ldc_tenant_isolation
    ON pack038_peak_shaving.ps_load_duration_curves
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ldc_service_bypass
    ON pack038_peak_shaving.ps_load_duration_curves
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack038_peak_shaving TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_load_profiles TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_interval_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_load_statistics TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_day_type_clusters TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_load_duration_curves TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack038_peak_shaving.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack038_peak_shaving IS
    'PACK-038 Peak Shaving Pack - demand peak detection, BESS dispatch optimization, demand charge management, coincident peak avoidance, ratchet analysis, power factor correction, and financial modeling.';

COMMENT ON TABLE pack038_peak_shaving.ps_load_profiles IS
    'Facility-level load profile configuration with meter association, utility details, baseline demand characteristics, and analysis period.';
COMMENT ON TABLE pack038_peak_shaving.ps_interval_data IS
    'High-resolution interval demand data (1/5/15/30/60-min) for peak detection, load analysis, and BESS dispatch simulation. TimescaleDB hypertable candidate.';
COMMENT ON TABLE pack038_peak_shaving.ps_load_statistics IS
    'Pre-computed statistical summaries (peak, average, percentiles, load factor) per profile and period for dashboard rendering.';
COMMENT ON TABLE pack038_peak_shaving.ps_day_type_clusters IS
    'Clustered day-type load shape patterns for peak prediction and dispatch schedule optimization.';
COMMENT ON TABLE pack038_peak_shaving.ps_load_duration_curves IS
    'Load duration curve data points for BESS sizing, peak shaving potential evaluation, and demand charge savings calculations.';

COMMENT ON COLUMN pack038_peak_shaving.ps_load_profiles.id IS 'Unique identifier for the load profile.';
COMMENT ON COLUMN pack038_peak_shaving.ps_load_profiles.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN pack038_peak_shaving.ps_load_profiles.facility_id IS 'Reference to the facility in the core facility registry.';
COMMENT ON COLUMN pack038_peak_shaving.ps_load_profiles.load_factor IS 'Ratio of average demand to peak demand (0-1). Higher values indicate more consistent load.';
COMMENT ON COLUMN pack038_peak_shaving.ps_load_profiles.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack038_peak_shaving.ps_interval_data.kw_demand IS 'Active power demand in kilowatts at the interval timestamp.';
COMMENT ON COLUMN pack038_peak_shaving.ps_interval_data.kvar_demand IS 'Reactive power demand in kilovars at the interval timestamp.';
COMMENT ON COLUMN pack038_peak_shaving.ps_interval_data.kva_demand IS 'Apparent power demand in kVA at the interval timestamp.';
COMMENT ON COLUMN pack038_peak_shaving.ps_interval_data.power_factor IS 'Power factor at the interval (0-1). Ratio of real to apparent power.';
COMMENT ON COLUMN pack038_peak_shaving.ps_interval_data.interval_length_minutes IS 'Length of the metering interval in minutes: 1, 5, 15, 30, or 60.';
COMMENT ON COLUMN pack038_peak_shaving.ps_interval_data.data_quality IS 'Data quality flag: MEASURED, ESTIMATED, INTERPOLATED, MISSING, VALIDATED.';

COMMENT ON COLUMN pack038_peak_shaving.ps_load_statistics.load_factor IS 'Load factor for the period (average/peak). Indicator of load shape flatness.';
COMMENT ON COLUMN pack038_peak_shaving.ps_load_statistics.demand_variability IS 'Coefficient of variation of demand (std_dev/mean). Higher values indicate spikier load.';
COMMENT ON COLUMN pack038_peak_shaving.ps_load_statistics.p95_kw IS '95th percentile demand in kW. Only 5% of intervals exceed this value.';

COMMENT ON COLUMN pack038_peak_shaving.ps_day_type_clusters.typical_load_shape IS 'JSON array of 24 or 96 values representing the normalized load shape for this cluster.';
COMMENT ON COLUMN pack038_peak_shaving.ps_day_type_clusters.silhouette_score IS 'Clustering quality metric (-1 to 1). Higher values indicate better-defined clusters.';

COMMENT ON COLUMN pack038_peak_shaving.ps_load_duration_curves.pct_time_above IS 'Percentage of time that demand exceeds the threshold (0-100).';
COMMENT ON COLUMN pack038_peak_shaving.ps_load_duration_curves.peak_shaving_potential_kwh IS 'Energy in kWh that could be shaved by limiting demand to this threshold.';
