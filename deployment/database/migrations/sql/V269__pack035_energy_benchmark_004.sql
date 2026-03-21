-- =============================================================================
-- V269: PACK-035 Energy Benchmark Pack - Weather Data Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Weather normalisation infrastructure: weather stations, degree-day data,
-- typical meteorological year (TMY) data, and regression models for
-- weather-normalised EUI calculation (ASHRAE 14 / IPMVP Option C).
--
-- Tables (4):
--   1. pack035_energy_benchmark.weather_stations
--   2. pack035_energy_benchmark.degree_day_data
--   3. pack035_energy_benchmark.tmy_data
--   4. pack035_energy_benchmark.regression_models
--
-- Previous: V268__pack035_energy_benchmark_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.weather_stations
-- =============================================================================
-- Weather station registry for degree-day and TMY data sourcing.
-- Supports NOAA ISD, Eurostat, and commercial degree-day providers.

CREATE TABLE pack035_energy_benchmark.weather_stations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id              VARCHAR(50)     NOT NULL UNIQUE,
    station_name            VARCHAR(255)    NOT NULL,
    country_code            CHAR(2)         NOT NULL,
    region                  VARCHAR(100),
    city                    VARCHAR(100),
    latitude                DECIMAL(10, 7)  NOT NULL,
    longitude               DECIMAL(10, 7)  NOT NULL,
    altitude_m              DECIMAL(8, 2),
    data_source             VARCHAR(100)    NOT NULL,
    provider_url            TEXT,
    active                  BOOLEAN         DEFAULT true,
    first_data_date         DATE,
    last_data_date          DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_ws_latitude CHECK (
        latitude >= -90 AND latitude <= 90
    ),
    CONSTRAINT chk_p035_ws_longitude CHECK (
        longitude >= -180 AND longitude <= 180
    ),
    CONSTRAINT chk_p035_ws_country CHECK (
        LENGTH(country_code) = 2
    ),
    CONSTRAINT chk_p035_ws_data_source CHECK (
        data_source IN ('NOAA_ISD', 'EUROSTAT', 'METEOSTAT', 'DEGREE_DAYS_NET',
                         'CIBSE_TRY', 'DWD', 'MET_OFFICE', 'CUSTOM')
    )
);

-- Indexes
CREATE INDEX idx_p035_ws_station_id      ON pack035_energy_benchmark.weather_stations(station_id);
CREATE INDEX idx_p035_ws_country         ON pack035_energy_benchmark.weather_stations(country_code);
CREATE INDEX idx_p035_ws_city            ON pack035_energy_benchmark.weather_stations(city);
CREATE INDEX idx_p035_ws_active          ON pack035_energy_benchmark.weather_stations(active);
CREATE INDEX idx_p035_ws_location        ON pack035_energy_benchmark.weather_stations(latitude, longitude);
CREATE INDEX idx_p035_ws_source          ON pack035_energy_benchmark.weather_stations(data_source);

-- =============================================================================
-- Table 2: pack035_energy_benchmark.degree_day_data
-- =============================================================================
-- Daily or monthly degree-day records for heating and cooling at multiple
-- base temperatures. Used for weather normalisation regression models.
-- HDD/CDD base temperatures follow regional conventions:
--   EU: HDD base 15.5C / CDD base 18C (Eurostat)
--   UK: HDD base 15.5C / CDD base 15.5C (CIBSE)
--   US: HDD base 65F (18.3C) / CDD base 65F (18.3C)

CREATE TABLE pack035_energy_benchmark.degree_day_data (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id              UUID            NOT NULL REFERENCES pack035_energy_benchmark.weather_stations(id) ON DELETE CASCADE,
    date                    DATE            NOT NULL,
    granularity             VARCHAR(10)     NOT NULL DEFAULT 'DAILY',
    -- Heating degree days at various base temperatures
    hdd_base_155            DECIMAL(8, 4),
    hdd_base_180            DECIMAL(8, 4),
    hdd_base_200            DECIMAL(8, 4),
    -- Cooling degree days at various base temperatures
    cdd_base_180            DECIMAL(8, 4),
    cdd_base_220            DECIMAL(8, 4),
    cdd_base_240            DECIMAL(8, 4),
    -- Temperature statistics
    mean_temp_c             DECIMAL(6, 2),
    min_temp_c              DECIMAL(6, 2),
    max_temp_c              DECIMAL(6, 2),
    -- Source tracking
    source                  VARCHAR(100),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_dd_granularity CHECK (
        granularity IN ('DAILY', 'MONTHLY')
    ),
    CONSTRAINT chk_p035_dd_hdd_155 CHECK (
        hdd_base_155 IS NULL OR hdd_base_155 >= 0
    ),
    CONSTRAINT chk_p035_dd_hdd_180 CHECK (
        hdd_base_180 IS NULL OR hdd_base_180 >= 0
    ),
    CONSTRAINT chk_p035_dd_hdd_200 CHECK (
        hdd_base_200 IS NULL OR hdd_base_200 >= 0
    ),
    CONSTRAINT chk_p035_dd_cdd_180 CHECK (
        cdd_base_180 IS NULL OR cdd_base_180 >= 0
    ),
    CONSTRAINT chk_p035_dd_cdd_220 CHECK (
        cdd_base_220 IS NULL OR cdd_base_220 >= 0
    ),
    CONSTRAINT chk_p035_dd_cdd_240 CHECK (
        cdd_base_240 IS NULL OR cdd_base_240 >= 0
    ),
    CONSTRAINT chk_p035_dd_temp_range CHECK (
        mean_temp_c IS NULL OR (mean_temp_c >= -80 AND mean_temp_c <= 65)
    ),
    CONSTRAINT uq_p035_dd_station_date UNIQUE (station_id, date, granularity)
);

-- Indexes
CREATE INDEX idx_p035_dd_station         ON pack035_energy_benchmark.degree_day_data(station_id);
CREATE INDEX idx_p035_dd_date            ON pack035_energy_benchmark.degree_day_data(date);
CREATE INDEX idx_p035_dd_granularity     ON pack035_energy_benchmark.degree_day_data(granularity);
CREATE INDEX idx_p035_dd_station_date    ON pack035_energy_benchmark.degree_day_data(station_id, date DESC);
CREATE INDEX idx_p035_dd_hdd155          ON pack035_energy_benchmark.degree_day_data(hdd_base_155);
CREATE INDEX idx_p035_dd_cdd180          ON pack035_energy_benchmark.degree_day_data(cdd_base_180);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.tmy_data
-- =============================================================================
-- Typical Meteorological Year (TMY) monthly summaries for weather
-- normalisation baseline. Used to establish long-term normal conditions
-- for converting actual-year EUI to TMY-normalised EUI.

CREATE TABLE pack035_energy_benchmark.tmy_data (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id              UUID            NOT NULL REFERENCES pack035_energy_benchmark.weather_stations(id) ON DELETE CASCADE,
    month                   INTEGER         NOT NULL,
    mean_temp_c             DECIMAL(6, 2)   NOT NULL,
    min_temp_c              DECIMAL(6, 2),
    max_temp_c              DECIMAL(6, 2),
    hdd_monthly             DECIMAL(8, 2),
    cdd_monthly             DECIMAL(8, 2),
    hdd_base_temp_c         DECIMAL(5, 2)   DEFAULT 15.5,
    cdd_base_temp_c         DECIMAL(5, 2)   DEFAULT 18.0,
    global_irradiance_kwh_m2 DECIMAL(8, 2),
    diffuse_irradiance_kwh_m2 DECIMAL(8, 2),
    wind_speed_ms           DECIMAL(6, 2),
    relative_humidity_pct   DECIMAL(5, 2),
    tmy_source              VARCHAR(100),
    tmy_period_start_year   INTEGER,
    tmy_period_end_year     INTEGER,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_tmy_month CHECK (
        month >= 1 AND month <= 12
    ),
    CONSTRAINT chk_p035_tmy_hdd CHECK (
        hdd_monthly IS NULL OR hdd_monthly >= 0
    ),
    CONSTRAINT chk_p035_tmy_cdd CHECK (
        cdd_monthly IS NULL OR cdd_monthly >= 0
    ),
    CONSTRAINT chk_p035_tmy_irradiance CHECK (
        global_irradiance_kwh_m2 IS NULL OR global_irradiance_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_tmy_wind CHECK (
        wind_speed_ms IS NULL OR wind_speed_ms >= 0
    ),
    CONSTRAINT chk_p035_tmy_humidity CHECK (
        relative_humidity_pct IS NULL OR (relative_humidity_pct >= 0 AND relative_humidity_pct <= 100)
    ),
    CONSTRAINT uq_p035_tmy_station_month UNIQUE (station_id, month)
);

-- Indexes
CREATE INDEX idx_p035_tmy_station        ON pack035_energy_benchmark.tmy_data(station_id);
CREATE INDEX idx_p035_tmy_month          ON pack035_energy_benchmark.tmy_data(month);

-- =============================================================================
-- Table 4: pack035_energy_benchmark.regression_models
-- =============================================================================
-- Weather normalisation regression models per facility/meter.
-- Supports single-variable (HDD or CDD), multi-variable (HDD+CDD),
-- and change-point models (3P, 4P, 5P). Stores model coefficients,
-- goodness-of-fit statistics, and the normalised annual consumption.
-- Follows ASHRAE Guideline 14 / IPMVP Option C methodology.

CREATE TABLE pack035_energy_benchmark.regression_models (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    meter_id                UUID            REFERENCES pack035_energy_benchmark.metering_points(id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    station_id              UUID            REFERENCES pack035_energy_benchmark.weather_stations(id) ON DELETE SET NULL,
    model_type              VARCHAR(30)     NOT NULL,
    model_name              VARCHAR(255),
    -- Model parameters
    coefficients            JSONB           NOT NULL DEFAULT '{}',
    intercept               DECIMAL(14, 4),
    base_temperature_c      DECIMAL(5, 2),
    -- Goodness of fit
    r_squared               DECIMAL(6, 5),
    adj_r_squared           DECIMAL(6, 5),
    cv_rmse                 DECIMAL(8, 4),
    nmbe                    DECIMAL(8, 4),
    f_statistic             DECIMAL(12, 4),
    t_ratios                JSONB           DEFAULT '{}',
    durbin_watson           DECIMAL(6, 4),
    aic                     DECIMAL(14, 4),
    bic                     DECIMAL(14, 4),
    -- Change-point model
    change_points           JSONB           DEFAULT '{}',
    -- Results
    normalised_annual_kwh   DECIMAL(14, 2),
    normalised_eui_kwh_m2   DECIMAL(10, 4),
    baseline_period_start   DATE,
    baseline_period_end     DATE,
    n_observations          INTEGER,
    -- Validation
    validation_status       VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    validation_notes        TEXT,
    fitted_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_rm_type CHECK (
        model_type IN ('SIMPLE_HDD', 'SIMPLE_CDD', 'MULTI_HDD_CDD',
                        '3P_HEATING', '3P_COOLING', '4P', '5P',
                        'MEAN', 'DAY_ADJUSTED', 'CUSTOM')
    ),
    CONSTRAINT chk_p035_rm_r2 CHECK (
        r_squared IS NULL OR (r_squared >= 0 AND r_squared <= 1)
    ),
    CONSTRAINT chk_p035_rm_adj_r2 CHECK (
        adj_r_squared IS NULL OR (adj_r_squared >= -1 AND adj_r_squared <= 1)
    ),
    CONSTRAINT chk_p035_rm_cv_rmse CHECK (
        cv_rmse IS NULL OR cv_rmse >= 0
    ),
    CONSTRAINT chk_p035_rm_dw CHECK (
        durbin_watson IS NULL OR (durbin_watson >= 0 AND durbin_watson <= 4)
    ),
    CONSTRAINT chk_p035_rm_norm_kwh CHECK (
        normalised_annual_kwh IS NULL OR normalised_annual_kwh >= 0
    ),
    CONSTRAINT chk_p035_rm_validation CHECK (
        validation_status IN ('PENDING', 'VALID', 'INVALID', 'MARGINAL', 'REVIEW_REQUIRED')
    ),
    CONSTRAINT chk_p035_rm_baseline CHECK (
        baseline_period_start IS NULL OR baseline_period_end IS NULL
        OR baseline_period_start < baseline_period_end
    ),
    CONSTRAINT chk_p035_rm_n_obs CHECK (
        n_observations IS NULL OR n_observations >= 0
    )
);

-- Indexes
CREATE INDEX idx_p035_rm_facility        ON pack035_energy_benchmark.regression_models(facility_id);
CREATE INDEX idx_p035_rm_meter           ON pack035_energy_benchmark.regression_models(meter_id);
CREATE INDEX idx_p035_rm_tenant          ON pack035_energy_benchmark.regression_models(tenant_id);
CREATE INDEX idx_p035_rm_station         ON pack035_energy_benchmark.regression_models(station_id);
CREATE INDEX idx_p035_rm_type            ON pack035_energy_benchmark.regression_models(model_type);
CREATE INDEX idx_p035_rm_validation      ON pack035_energy_benchmark.regression_models(validation_status);
CREATE INDEX idx_p035_rm_r2              ON pack035_energy_benchmark.regression_models(r_squared DESC);
CREATE INDEX idx_p035_rm_fitted          ON pack035_energy_benchmark.regression_models(fitted_at DESC);
CREATE INDEX idx_p035_rm_fac_model       ON pack035_energy_benchmark.regression_models(facility_id, model_type);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.regression_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_rm_tenant_isolation ON pack035_energy_benchmark.regression_models
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_rm_service_bypass ON pack035_energy_benchmark.regression_models
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.weather_stations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.degree_day_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.tmy_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.regression_models TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.weather_stations IS
    'Weather station registry for degree-day and TMY data sourcing (NOAA, Eurostat, Meteostat, DegreeDay.net).';
COMMENT ON TABLE pack035_energy_benchmark.degree_day_data IS
    'Daily/monthly degree-day records for heating and cooling at multiple base temperatures.';
COMMENT ON TABLE pack035_energy_benchmark.tmy_data IS
    'Typical Meteorological Year monthly summaries for weather normalisation baseline.';
COMMENT ON TABLE pack035_energy_benchmark.regression_models IS
    'Weather normalisation regression models per facility following ASHRAE 14 / IPMVP Option C methodology.';

COMMENT ON COLUMN pack035_energy_benchmark.regression_models.cv_rmse IS
    'Coefficient of Variation of Root Mean Square Error. ASHRAE 14 threshold: <=15% for monthly, <=25% for daily.';
COMMENT ON COLUMN pack035_energy_benchmark.regression_models.nmbe IS
    'Normalised Mean Bias Error. ASHRAE 14 threshold: <=5% for monthly, <=10% for daily.';
COMMENT ON COLUMN pack035_energy_benchmark.regression_models.durbin_watson IS
    'Durbin-Watson statistic for autocorrelation detection (ideal ~2.0, range 0-4).';
COMMENT ON COLUMN pack035_energy_benchmark.regression_models.change_points IS
    'Change-point temperatures for 3P/4P/5P models as JSON: {"heating_cp": 15.5, "cooling_cp": 18.0}.';
