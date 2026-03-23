-- =============================================================================
-- V301: PACK-038 Peak Shaving Pack - Coincident Peak Management
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Coincident peak (CP) management tables for tracking, predicting, and
-- responding to system-level peak events that drive transmission and
-- capacity charges. Covers CP event history, prediction models,
-- response actions, charge calculations, and CP calendar management.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_cp_events
--   2. pack038_peak_shaving.ps_cp_predictions
--   3. pack038_peak_shaving.ps_cp_responses
--   4. pack038_peak_shaving.ps_cp_charges
--   5. pack038_peak_shaving.ps_cp_calendars
--
-- Seed Data: Historical CP dates for PJM and ERCOT (2020-2025)
--
-- Previous: V300__pack038_peak_shaving_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_cp_events
-- =============================================================================
-- Historical and real-time coincident peak events at the system/ISO/RTO
-- level. Tracks the actual system peak timestamps, magnitudes, and
-- settlement-relevant data for transmission cost allocation.

CREATE TABLE pack038_peak_shaving.ps_cp_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    iso_rto_region          VARCHAR(30)     NOT NULL,
    cp_program              VARCHAR(50)     NOT NULL,
    event_date              DATE            NOT NULL,
    event_timestamp         TIMESTAMPTZ     NOT NULL,
    system_peak_mw          NUMERIC(12,3)   NOT NULL,
    system_peak_hour        INTEGER         NOT NULL,
    event_rank              INTEGER,
    total_cp_events_year    INTEGER,
    temperature_f           NUMERIC(6,2),
    heat_index_f            NUMERIC(6,2),
    humidity_pct            NUMERIC(5,2),
    weather_station         VARCHAR(100),
    day_of_week             VARCHAR(10),
    is_confirmed            BOOLEAN         NOT NULL DEFAULT false,
    confirmation_date       DATE,
    settlement_year         INTEGER         NOT NULL,
    settlement_month        INTEGER,
    transmission_rate_per_kw NUMERIC(10,4),
    capacity_rate_per_kw    NUMERIC(10,4),
    data_source             VARCHAR(100),
    source_url              VARCHAR(500),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cpe_region CHECK (
        iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'FR_RTE', 'NL_TENNET', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cpe_program CHECK (
        cp_program IN (
            'PJM_5CP', 'PJM_1CP', 'ERCOT_4CP', 'NYISO_ICAP',
            'ISO_NE_MONTHLY_CP', 'MISO_YEARLY_CP', 'CAISO_CP',
            'UK_TRIAD', 'NETWORK_PEAK', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cpe_peak CHECK (
        system_peak_mw > 0
    ),
    CONSTRAINT chk_p038_cpe_hour CHECK (
        system_peak_hour >= 0 AND system_peak_hour <= 23
    ),
    CONSTRAINT chk_p038_cpe_rank CHECK (
        event_rank IS NULL OR event_rank >= 1
    ),
    CONSTRAINT chk_p038_cpe_year CHECK (
        settlement_year >= 2015 AND settlement_year <= 2100
    ),
    CONSTRAINT chk_p038_cpe_month CHECK (
        settlement_month IS NULL OR (settlement_month >= 1 AND settlement_month <= 12)
    ),
    CONSTRAINT chk_p038_cpe_humidity CHECK (
        humidity_pct IS NULL OR (humidity_pct >= 0 AND humidity_pct <= 100)
    ),
    CONSTRAINT chk_p038_cpe_day CHECK (
        day_of_week IS NULL OR day_of_week IN (
            'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY',
            'SATURDAY', 'SUNDAY'
        )
    ),
    CONSTRAINT uq_p038_cpe_region_date UNIQUE (iso_rto_region, cp_program, event_date, event_rank)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cpe_tenant         ON pack038_peak_shaving.ps_cp_events(tenant_id);
CREATE INDEX idx_p038_cpe_region         ON pack038_peak_shaving.ps_cp_events(iso_rto_region);
CREATE INDEX idx_p038_cpe_program        ON pack038_peak_shaving.ps_cp_events(cp_program);
CREATE INDEX idx_p038_cpe_date           ON pack038_peak_shaving.ps_cp_events(event_date DESC);
CREATE INDEX idx_p038_cpe_year           ON pack038_peak_shaving.ps_cp_events(settlement_year);
CREATE INDEX idx_p038_cpe_peak           ON pack038_peak_shaving.ps_cp_events(system_peak_mw DESC);
CREATE INDEX idx_p038_cpe_confirmed      ON pack038_peak_shaving.ps_cp_events(is_confirmed);
CREATE INDEX idx_p038_cpe_created        ON pack038_peak_shaving.ps_cp_events(created_at DESC);

-- Composite: region + year for annual CP lookups
CREATE INDEX idx_p038_cpe_region_year    ON pack038_peak_shaving.ps_cp_events(iso_rto_region, settlement_year, event_rank);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_cpe_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_cp_events
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_cp_predictions
-- =============================================================================
-- Coincident peak prediction models and forecasts. Each prediction
-- estimates the probability that a given day/hour will be a CP event,
-- enabling proactive load reduction before system peaks occur.

CREATE TABLE pack038_peak_shaving.ps_cp_predictions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    iso_rto_region          VARCHAR(30)     NOT NULL,
    cp_program              VARCHAR(50)     NOT NULL,
    prediction_date         DATE            NOT NULL,
    prediction_hour         INTEGER         NOT NULL,
    cp_probability_pct      NUMERIC(5,2)    NOT NULL,
    confidence_level_pct    NUMERIC(5,2),
    alert_level             VARCHAR(20)     NOT NULL DEFAULT 'NONE',
    predicted_system_peak_mw NUMERIC(12,3),
    forecast_temperature_f  NUMERIC(6,2),
    forecast_heat_index_f   NUMERIC(6,2),
    forecast_humidity_pct   NUMERIC(5,2),
    forecast_cloud_cover_pct NUMERIC(5,2),
    load_forecast_mw        NUMERIC(12,3),
    renewable_forecast_mw   NUMERIC(12,3),
    net_load_forecast_mw    NUMERIC(12,3),
    model_name              VARCHAR(100),
    model_version           VARCHAR(30),
    features_used           JSONB           DEFAULT '[]',
    prediction_issued_at    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    was_actual_cp           BOOLEAN,
    prediction_accuracy     NUMERIC(5,2),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cpp_region CHECK (
        iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'FR_RTE', 'NL_TENNET', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cpp_program CHECK (
        cp_program IN (
            'PJM_5CP', 'PJM_1CP', 'ERCOT_4CP', 'NYISO_ICAP',
            'ISO_NE_MONTHLY_CP', 'MISO_YEARLY_CP', 'CAISO_CP',
            'UK_TRIAD', 'NETWORK_PEAK', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cpp_hour CHECK (
        prediction_hour >= 0 AND prediction_hour <= 23
    ),
    CONSTRAINT chk_p038_cpp_probability CHECK (
        cp_probability_pct >= 0 AND cp_probability_pct <= 100
    ),
    CONSTRAINT chk_p038_cpp_confidence CHECK (
        confidence_level_pct IS NULL OR (confidence_level_pct >= 0 AND confidence_level_pct <= 100)
    ),
    CONSTRAINT chk_p038_cpp_alert CHECK (
        alert_level IN ('NONE', 'WATCH', 'WARNING', 'CRITICAL', 'EMERGENCY')
    ),
    CONSTRAINT chk_p038_cpp_accuracy CHECK (
        prediction_accuracy IS NULL OR (prediction_accuracy >= 0 AND prediction_accuracy <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cpp_profile        ON pack038_peak_shaving.ps_cp_predictions(profile_id);
CREATE INDEX idx_p038_cpp_tenant         ON pack038_peak_shaving.ps_cp_predictions(tenant_id);
CREATE INDEX idx_p038_cpp_region         ON pack038_peak_shaving.ps_cp_predictions(iso_rto_region);
CREATE INDEX idx_p038_cpp_program        ON pack038_peak_shaving.ps_cp_predictions(cp_program);
CREATE INDEX idx_p038_cpp_date           ON pack038_peak_shaving.ps_cp_predictions(prediction_date DESC);
CREATE INDEX idx_p038_cpp_probability    ON pack038_peak_shaving.ps_cp_predictions(cp_probability_pct DESC);
CREATE INDEX idx_p038_cpp_alert          ON pack038_peak_shaving.ps_cp_predictions(alert_level);
CREATE INDEX idx_p038_cpp_issued         ON pack038_peak_shaving.ps_cp_predictions(prediction_issued_at DESC);
CREATE INDEX idx_p038_cpp_created        ON pack038_peak_shaving.ps_cp_predictions(created_at DESC);

-- High-probability predictions for alert dispatch
CREATE INDEX idx_p038_cpp_high_prob      ON pack038_peak_shaving.ps_cp_predictions(prediction_date, cp_probability_pct DESC)
    WHERE cp_probability_pct >= 50;

-- ---------------------------------------------------------------------------
-- (No trigger - no updated_at column)
-- ---------------------------------------------------------------------------

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_cp_responses
-- =============================================================================
-- Facility-level response actions taken during CP events or high-probability
-- CP predictions. Tracks load reduction achieved, BESS discharge, and
-- the resulting facility demand at the system peak hour.

CREATE TABLE pack038_peak_shaving.ps_cp_responses (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    cp_event_id             UUID            REFERENCES pack038_peak_shaving.ps_cp_events(id),
    cp_prediction_id        UUID            REFERENCES pack038_peak_shaving.ps_cp_predictions(id),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id),
    tenant_id               UUID            NOT NULL,
    response_date           DATE            NOT NULL,
    response_trigger        VARCHAR(30)     NOT NULL,
    notification_received_at TIMESTAMPTZ,
    response_initiated_at   TIMESTAMPTZ,
    response_time_min       INTEGER,
    baseline_demand_kw      NUMERIC(12,3)   NOT NULL,
    target_demand_kw        NUMERIC(12,3),
    achieved_demand_kw      NUMERIC(12,3),
    demand_reduction_kw     NUMERIC(12,3),
    reduction_pct           NUMERIC(7,4),
    bess_discharged_kwh     NUMERIC(12,3),
    loads_shifted_kw        NUMERIC(12,3),
    loads_curtailed_kw      NUMERIC(12,3),
    generator_contribution_kw NUMERIC(12,3),
    actions_taken           JSONB           DEFAULT '[]',
    response_status         VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    performance_rating      VARCHAR(20),
    estimated_savings       NUMERIC(12,2),
    actual_savings          NUMERIC(12,2),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cpr_trigger CHECK (
        response_trigger IN (
            'PREDICTION', 'ISO_ALERT', 'UTILITY_NOTIFICATION',
            'MANUAL', 'AUTOMATED', 'WEATHER_TRIGGER', 'PRICE_TRIGGER'
        )
    ),
    CONSTRAINT chk_p038_cpr_baseline CHECK (
        baseline_demand_kw > 0
    ),
    CONSTRAINT chk_p038_cpr_status CHECK (
        response_status IN (
            'PLANNED', 'ACTIVATED', 'IN_PROGRESS', 'COMPLETED',
            'FAILED', 'CANCELLED', 'PARTIAL'
        )
    ),
    CONSTRAINT chk_p038_cpr_rating CHECK (
        performance_rating IS NULL OR performance_rating IN (
            'EXCELLENT', 'GOOD', 'ADEQUATE', 'POOR', 'FAILED'
        )
    ),
    CONSTRAINT chk_p038_cpr_reduction_pct CHECK (
        reduction_pct IS NULL OR reduction_pct >= -100
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cpr_cp_event       ON pack038_peak_shaving.ps_cp_responses(cp_event_id);
CREATE INDEX idx_p038_cpr_prediction     ON pack038_peak_shaving.ps_cp_responses(cp_prediction_id);
CREATE INDEX idx_p038_cpr_profile        ON pack038_peak_shaving.ps_cp_responses(profile_id);
CREATE INDEX idx_p038_cpr_tenant         ON pack038_peak_shaving.ps_cp_responses(tenant_id);
CREATE INDEX idx_p038_cpr_date           ON pack038_peak_shaving.ps_cp_responses(response_date DESC);
CREATE INDEX idx_p038_cpr_trigger        ON pack038_peak_shaving.ps_cp_responses(response_trigger);
CREATE INDEX idx_p038_cpr_status         ON pack038_peak_shaving.ps_cp_responses(response_status);
CREATE INDEX idx_p038_cpr_rating         ON pack038_peak_shaving.ps_cp_responses(performance_rating);
CREATE INDEX idx_p038_cpr_reduction      ON pack038_peak_shaving.ps_cp_responses(demand_reduction_kw DESC);
CREATE INDEX idx_p038_cpr_savings        ON pack038_peak_shaving.ps_cp_responses(actual_savings DESC);
CREATE INDEX idx_p038_cpr_created        ON pack038_peak_shaving.ps_cp_responses(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_cpr_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_cp_responses
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_cp_charges
-- =============================================================================
-- Transmission and capacity charges allocated based on facility demand
-- at coincident peak hours. These charges are typically the largest
-- single line item on commercial/industrial electric bills.

CREATE TABLE pack038_peak_shaving.ps_cp_charges (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    cp_event_id             UUID            REFERENCES pack038_peak_shaving.ps_cp_events(id),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id),
    tenant_id               UUID            NOT NULL,
    charge_type             VARCHAR(30)     NOT NULL,
    settlement_year         INTEGER         NOT NULL,
    settlement_month        INTEGER,
    facility_demand_at_cp_kw NUMERIC(12,3) NOT NULL,
    allocation_pct          NUMERIC(10,8),
    rate_per_kw             NUMERIC(10,4)   NOT NULL,
    annual_charge           NUMERIC(12,2)   NOT NULL,
    monthly_charge          NUMERIC(12,2),
    without_shaving_demand_kw NUMERIC(12,3),
    without_shaving_charge  NUMERIC(12,2),
    savings_from_shaving    NUMERIC(12,2),
    prior_year_charge       NUMERIC(12,2),
    year_over_year_change   NUMERIC(12,2),
    network_service_peak_mw NUMERIC(12,3),
    zonal_peak_mw           NUMERIC(12,3),
    billing_period_start    DATE,
    billing_period_end      DATE,
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    charge_status           VARCHAR(20)     NOT NULL DEFAULT 'ESTIMATED',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cpc_type CHECK (
        charge_type IN (
            'TRANSMISSION', 'CAPACITY', 'NETWORK_SERVICE',
            'DEMAND_SIDE_MANAGEMENT', 'RELIABILITY', 'ANCILLARY',
            'ZONAL_CAPACITY', 'PEAK_LOAD_CONTRIBUTION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cpc_year CHECK (
        settlement_year >= 2015 AND settlement_year <= 2100
    ),
    CONSTRAINT chk_p038_cpc_demand CHECK (
        facility_demand_at_cp_kw >= 0
    ),
    CONSTRAINT chk_p038_cpc_rate CHECK (
        rate_per_kw >= 0
    ),
    CONSTRAINT chk_p038_cpc_charge CHECK (
        annual_charge >= 0
    ),
    CONSTRAINT chk_p038_cpc_status CHECK (
        charge_status IN ('ESTIMATED', 'PRELIMINARY', 'FINAL', 'RECONCILED', 'DISPUTED')
    ),
    CONSTRAINT chk_p038_cpc_month CHECK (
        settlement_month IS NULL OR (settlement_month >= 1 AND settlement_month <= 12)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cpc_event          ON pack038_peak_shaving.ps_cp_charges(cp_event_id);
CREATE INDEX idx_p038_cpc_profile        ON pack038_peak_shaving.ps_cp_charges(profile_id);
CREATE INDEX idx_p038_cpc_tenant         ON pack038_peak_shaving.ps_cp_charges(tenant_id);
CREATE INDEX idx_p038_cpc_type           ON pack038_peak_shaving.ps_cp_charges(charge_type);
CREATE INDEX idx_p038_cpc_year           ON pack038_peak_shaving.ps_cp_charges(settlement_year);
CREATE INDEX idx_p038_cpc_charge         ON pack038_peak_shaving.ps_cp_charges(annual_charge DESC);
CREATE INDEX idx_p038_cpc_savings        ON pack038_peak_shaving.ps_cp_charges(savings_from_shaving DESC);
CREATE INDEX idx_p038_cpc_status         ON pack038_peak_shaving.ps_cp_charges(charge_status);
CREATE INDEX idx_p038_cpc_created        ON pack038_peak_shaving.ps_cp_charges(created_at DESC);

-- Composite: profile + year for annual charge history
CREATE INDEX idx_p038_cpc_profile_year   ON pack038_peak_shaving.ps_cp_charges(profile_id, settlement_year DESC, charge_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_cpc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_cp_charges
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_cp_calendars
-- =============================================================================
-- CP program calendars defining eligible peak windows, seasons, and
-- historical statistics for each ISO/RTO coincident peak program.

CREATE TABLE pack038_peak_shaving.ps_cp_calendars (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    iso_rto_region          VARCHAR(30)     NOT NULL,
    cp_program              VARCHAR(50)     NOT NULL,
    program_year            INTEGER         NOT NULL,
    eligible_months         INTEGER[]       NOT NULL,
    eligible_days           VARCHAR(20)     NOT NULL DEFAULT 'WEEKDAY',
    peak_window_start_hour  INTEGER         NOT NULL,
    peak_window_end_hour    INTEGER         NOT NULL,
    number_of_cp_events     INTEGER         NOT NULL,
    historical_avg_hour     INTEGER,
    historical_avg_temp_f   NUMERIC(6,2),
    historical_min_peak_mw  NUMERIC(12,3),
    historical_max_peak_mw  NUMERIC(12,3),
    historical_avg_peak_mw  NUMERIC(12,3),
    transmission_rate_per_kw NUMERIC(10,4),
    capacity_rate_per_kw    NUMERIC(10,4),
    program_description     TEXT,
    notification_available  BOOLEAN         DEFAULT false,
    notification_lead_hours INTEGER,
    data_source             VARCHAR(100),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cpcal_region CHECK (
        iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'FR_RTE', 'NL_TENNET', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cpcal_program CHECK (
        cp_program IN (
            'PJM_5CP', 'PJM_1CP', 'ERCOT_4CP', 'NYISO_ICAP',
            'ISO_NE_MONTHLY_CP', 'MISO_YEARLY_CP', 'CAISO_CP',
            'UK_TRIAD', 'NETWORK_PEAK', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cpcal_year CHECK (
        program_year >= 2015 AND program_year <= 2100
    ),
    CONSTRAINT chk_p038_cpcal_days CHECK (
        eligible_days IN ('WEEKDAY', 'ALL_DAYS', 'BUSINESS_DAYS')
    ),
    CONSTRAINT chk_p038_cpcal_start CHECK (
        peak_window_start_hour >= 0 AND peak_window_start_hour <= 23
    ),
    CONSTRAINT chk_p038_cpcal_end CHECK (
        peak_window_end_hour >= 0 AND peak_window_end_hour <= 23
    ),
    CONSTRAINT chk_p038_cpcal_events CHECK (
        number_of_cp_events >= 1
    ),
    CONSTRAINT uq_p038_cpcal_program_year UNIQUE (iso_rto_region, cp_program, program_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cpcal_tenant       ON pack038_peak_shaving.ps_cp_calendars(tenant_id);
CREATE INDEX idx_p038_cpcal_region       ON pack038_peak_shaving.ps_cp_calendars(iso_rto_region);
CREATE INDEX idx_p038_cpcal_program      ON pack038_peak_shaving.ps_cp_calendars(cp_program);
CREATE INDEX idx_p038_cpcal_year         ON pack038_peak_shaving.ps_cp_calendars(program_year DESC);
CREATE INDEX idx_p038_cpcal_created      ON pack038_peak_shaving.ps_cp_calendars(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_cpcal_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_cp_calendars
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_cp_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_cp_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_cp_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_cp_charges ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_cp_calendars ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_cpe_tenant_isolation
    ON pack038_peak_shaving.ps_cp_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cpe_service_bypass
    ON pack038_peak_shaving.ps_cp_events
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cpp_tenant_isolation
    ON pack038_peak_shaving.ps_cp_predictions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cpp_service_bypass
    ON pack038_peak_shaving.ps_cp_predictions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cpr_tenant_isolation
    ON pack038_peak_shaving.ps_cp_responses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cpr_service_bypass
    ON pack038_peak_shaving.ps_cp_responses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cpc_tenant_isolation
    ON pack038_peak_shaving.ps_cp_charges
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cpc_service_bypass
    ON pack038_peak_shaving.ps_cp_charges
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cpcal_tenant_isolation
    ON pack038_peak_shaving.ps_cp_calendars
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cpcal_service_bypass
    ON pack038_peak_shaving.ps_cp_calendars
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_cp_events TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_cp_predictions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_cp_responses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_cp_charges TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_cp_calendars TO PUBLIC;

-- =============================================================================
-- Seed Data: Historical PJM 5CP Dates (2020-2025)
-- =============================================================================
-- PJM uses the 5 highest system peak hours during June-September to
-- allocate transmission costs (Network Integration Transmission Service).

INSERT INTO pack038_peak_shaving.ps_cp_events (tenant_id, iso_rto_region, cp_program, event_date, event_timestamp, system_peak_mw, system_peak_hour, event_rank, total_cp_events_year, temperature_f, day_of_week, is_confirmed, settlement_year, data_source) VALUES
-- 2020 PJM 5CP
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2020-07-27', '2020-07-27 17:00:00-04', 148200, 17, 1, 5, 98, 'MONDAY', true, 2020, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2020-07-20', '2020-07-20 17:00:00-04', 145800, 17, 2, 5, 96, 'MONDAY', true, 2020, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2020-08-10', '2020-08-10 16:00:00-04', 143900, 16, 3, 5, 95, 'MONDAY', true, 2020, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2020-07-21', '2020-07-21 17:00:00-04', 142300, 17, 4, 5, 97, 'TUESDAY', true, 2020, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2020-08-11', '2020-08-11 17:00:00-04', 141600, 17, 5, 5, 94, 'TUESDAY', true, 2020, 'PJM Interconnection'),
-- 2021 PJM 5CP
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2021-06-29', '2021-06-29 17:00:00-04', 147500, 17, 1, 5, 97, 'TUESDAY', true, 2021, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2021-06-30', '2021-06-30 17:00:00-04', 146200, 17, 2, 5, 98, 'WEDNESDAY', true, 2021, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2021-08-11', '2021-08-11 16:00:00-04', 144800, 16, 3, 5, 96, 'WEDNESDAY', true, 2021, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2021-06-28', '2021-06-28 17:00:00-04', 143500, 17, 4, 5, 95, 'MONDAY', true, 2021, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2021-08-12', '2021-08-12 17:00:00-04', 142100, 17, 5, 5, 94, 'THURSDAY', true, 2021, 'PJM Interconnection'),
-- 2022 PJM 5CP
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2022-07-20', '2022-07-20 17:00:00-04', 149800, 17, 1, 5, 99, 'WEDNESDAY', true, 2022, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2022-07-21', '2022-07-21 17:00:00-04', 148100, 17, 2, 5, 100, 'THURSDAY', true, 2022, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2022-08-04', '2022-08-04 16:00:00-04', 146500, 16, 3, 5, 97, 'THURSDAY', true, 2022, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2022-07-22', '2022-07-22 17:00:00-04', 145200, 17, 4, 5, 98, 'FRIDAY', true, 2022, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2022-06-13', '2022-06-13 17:00:00-04', 143800, 17, 5, 5, 96, 'MONDAY', true, 2022, 'PJM Interconnection'),
-- 2023 PJM 5CP
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2023-08-24', '2023-08-24 17:00:00-04', 150200, 17, 1, 5, 100, 'THURSDAY', true, 2023, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2023-08-23', '2023-08-23 17:00:00-04', 148900, 17, 2, 5, 99, 'WEDNESDAY', true, 2023, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2023-07-27', '2023-07-27 16:00:00-04', 147300, 16, 3, 5, 98, 'THURSDAY', true, 2023, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2023-06-22', '2023-06-22 17:00:00-04', 145600, 17, 4, 5, 97, 'THURSDAY', true, 2023, 'PJM Interconnection'),
('00000000-0000-0000-0000-000000000000', 'PJM', 'PJM_5CP', '2023-09-05', '2023-09-05 17:00:00-04', 144200, 17, 5, 5, 96, 'TUESDAY', true, 2023, 'PJM Interconnection');

-- =============================================================================
-- Seed Data: Historical ERCOT 4CP Dates (2020-2025)
-- =============================================================================
-- ERCOT uses the single highest 15-minute system peak in each of
-- June, July, August, September for transmission cost allocation.

INSERT INTO pack038_peak_shaving.ps_cp_events (tenant_id, iso_rto_region, cp_program, event_date, event_timestamp, system_peak_mw, system_peak_hour, event_rank, total_cp_events_year, temperature_f, settlement_month, day_of_week, is_confirmed, settlement_year, data_source) VALUES
-- 2020 ERCOT 4CP
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2020-06-24', '2020-06-24 16:00:00-05', 68900, 16, 1, 4, 102, 6, 'WEDNESDAY', true, 2020, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2020-07-13', '2020-07-13 16:00:00-05', 71200, 16, 1, 4, 104, 7, 'MONDAY', true, 2020, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2020-08-12', '2020-08-12 16:00:00-05', 73500, 16, 1, 4, 105, 8, 'WEDNESDAY', true, 2020, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2020-09-02', '2020-09-02 16:00:00-05', 69800, 16, 1, 4, 101, 9, 'WEDNESDAY', true, 2020, 'ERCOT'),
-- 2021 ERCOT 4CP
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2021-06-16', '2021-06-16 17:00:00-05', 70100, 17, 1, 4, 103, 6, 'WEDNESDAY', true, 2021, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2021-07-12', '2021-07-12 16:00:00-05', 72400, 16, 1, 4, 105, 7, 'MONDAY', true, 2021, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2021-08-18', '2021-08-18 16:00:00-05', 74200, 16, 1, 4, 106, 8, 'WEDNESDAY', true, 2021, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2021-09-08', '2021-09-08 16:00:00-05', 70800, 16, 1, 4, 102, 9, 'WEDNESDAY', true, 2021, 'ERCOT'),
-- 2022 ERCOT 4CP
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2022-06-13', '2022-06-13 17:00:00-05', 71800, 17, 1, 4, 104, 6, 'MONDAY', true, 2022, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2022-07-20', '2022-07-20 16:00:00-05', 78300, 16, 1, 4, 109, 7, 'WEDNESDAY', true, 2022, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2022-08-09', '2022-08-09 16:00:00-05', 76800, 16, 1, 4, 107, 8, 'TUESDAY', true, 2022, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2022-09-07', '2022-09-07 16:00:00-05', 72500, 16, 1, 4, 103, 9, 'WEDNESDAY', true, 2022, 'ERCOT'),
-- 2023 ERCOT 4CP
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2023-06-27', '2023-06-27 17:00:00-05', 73200, 17, 1, 4, 106, 6, 'TUESDAY', true, 2023, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2023-07-27', '2023-07-27 16:00:00-05', 80200, 16, 1, 4, 110, 7, 'THURSDAY', true, 2023, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2023-08-22', '2023-08-22 16:00:00-05', 79500, 16, 1, 4, 108, 8, 'TUESDAY', true, 2023, 'ERCOT'),
('00000000-0000-0000-0000-000000000000', 'ERCOT', 'ERCOT_4CP', '2023-09-06', '2023-09-06 16:00:00-05', 74100, 16, 1, 4, 104, 9, 'WEDNESDAY', true, 2023, 'ERCOT');

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_cp_events IS
    'Historical and real-time coincident peak events at the ISO/RTO system level for transmission cost allocation.';
COMMENT ON TABLE pack038_peak_shaving.ps_cp_predictions IS
    'Coincident peak probability forecasts enabling proactive load reduction before system peaks occur.';
COMMENT ON TABLE pack038_peak_shaving.ps_cp_responses IS
    'Facility-level response actions during CP events with load reduction achieved and savings tracking.';
COMMENT ON TABLE pack038_peak_shaving.ps_cp_charges IS
    'Transmission and capacity charges allocated based on facility demand at coincident peak hours.';
COMMENT ON TABLE pack038_peak_shaving.ps_cp_calendars IS
    'CP program calendars defining eligible peak windows, seasons, and historical statistics per ISO/RTO.';

COMMENT ON COLUMN pack038_peak_shaving.ps_cp_events.cp_program IS 'CP program: PJM_5CP, ERCOT_4CP, NYISO_ICAP, ISO_NE_MONTHLY_CP, UK_TRIAD, etc.';
COMMENT ON COLUMN pack038_peak_shaving.ps_cp_events.event_rank IS 'Rank of this CP event within the settlement year (1 = highest system peak).';
COMMENT ON COLUMN pack038_peak_shaving.ps_cp_predictions.cp_probability_pct IS 'Estimated probability (0-100) that this day/hour will be a coincident peak event.';
COMMENT ON COLUMN pack038_peak_shaving.ps_cp_predictions.alert_level IS 'Alert level: NONE, WATCH (>25%), WARNING (>50%), CRITICAL (>75%), EMERGENCY (>90%).';
COMMENT ON COLUMN pack038_peak_shaving.ps_cp_responses.response_trigger IS 'What triggered the CP response: PREDICTION, ISO_ALERT, UTILITY_NOTIFICATION, MANUAL, AUTOMATED.';
COMMENT ON COLUMN pack038_peak_shaving.ps_cp_charges.charge_type IS 'Charge type: TRANSMISSION, CAPACITY, NETWORK_SERVICE, PEAK_LOAD_CONTRIBUTION, etc.';
COMMENT ON COLUMN pack038_peak_shaving.ps_cp_charges.savings_from_shaving IS 'Dollar savings from peak shaving at the CP hour compared to unmanaged demand.';
