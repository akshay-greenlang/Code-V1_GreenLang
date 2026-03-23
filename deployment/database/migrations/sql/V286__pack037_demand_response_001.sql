-- =============================================================================
-- V286: PACK-037 Demand Response Pack - Core Schema & Facility Load Profiles
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack037_demand_response schema and foundational tables for
-- demand response operations. Tracks facility DR profiles, load inventories,
-- load flexibility assessments, curtailment capacity, and flexibility
-- registers used by downstream dispatch and event management.
--
-- Tables (5):
--   1. pack037_demand_response.dr_facility_profiles
--   2. pack037_demand_response.dr_load_inventory
--   3. pack037_demand_response.dr_load_flexibility
--   4. pack037_demand_response.dr_curtailment_capacity
--   5. pack037_demand_response.dr_flexibility_registers
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V285__pack036_utility_analysis_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack037_demand_response;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack037_demand_response.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack037_demand_response.dr_facility_profiles
-- =============================================================================
-- Facility-level demand response profiles capturing peak demand, contract
-- capacity, ISO/RTO region, and DR readiness classification. Each facility
-- can participate in multiple DR programs with different load assets.

CREATE TABLE pack037_demand_response.dr_facility_profiles (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    facility_name           VARCHAR(255)    NOT NULL,
    iso_rto_region          VARCHAR(30)     NOT NULL,
    utility_name            VARCHAR(255),
    utility_account_number  VARCHAR(100),
    peak_demand_kw          NUMERIC(12,4)   NOT NULL,
    contract_capacity_kw    NUMERIC(12,4),
    average_demand_kw       NUMERIC(12,4),
    base_load_kw            NUMERIC(12,4),
    curtailable_capacity_kw NUMERIC(12,4),
    dr_readiness_level      VARCHAR(30)     NOT NULL DEFAULT 'ASSESSMENT',
    automation_level        VARCHAR(30)     NOT NULL DEFAULT 'MANUAL',
    metering_type           VARCHAR(30)     NOT NULL DEFAULT 'INTERVAL',
    telemetry_enabled       BOOLEAN         DEFAULT false,
    ems_integration         BOOLEAN         DEFAULT false,
    ems_vendor              VARCHAR(100),
    country_code            CHAR(2)         NOT NULL DEFAULT 'US',
    timezone                VARCHAR(50)     NOT NULL DEFAULT 'America/New_York',
    latitude                NUMERIC(10,7),
    longitude               NUMERIC(10,7),
    facility_type           VARCHAR(50),
    operating_hours_start   TIME,
    operating_hours_end     TIME,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p037_fp_region CHECK (
        iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'DE_AMPRION', 'DE_50HZ', 'DE_TRANSNET',
            'FR_RTE', 'NL_TENNET', 'ES_REE', 'IT_TERNA', 'AU_AEMO',
            'JP_TEPCO', 'JP_KEPCO', 'OTHER'
        )
    ),
    CONSTRAINT chk_p037_fp_readiness CHECK (
        dr_readiness_level IN (
            'ASSESSMENT', 'ENROLLED', 'QUALIFIED', 'ACTIVE', 'SUSPENDED',
            'RETIRED'
        )
    ),
    CONSTRAINT chk_p037_fp_automation CHECK (
        automation_level IN (
            'MANUAL', 'SEMI_AUTO', 'FULLY_AUTO', 'ADR_OPENADR', 'ADR_IEEE2030'
        )
    ),
    CONSTRAINT chk_p037_fp_metering CHECK (
        metering_type IN (
            'INTERVAL', 'SMART_METER', 'SCADA', 'PULSE', 'MANUAL_READ'
        )
    ),
    CONSTRAINT chk_p037_fp_peak CHECK (
        peak_demand_kw > 0
    ),
    CONSTRAINT chk_p037_fp_contract CHECK (
        contract_capacity_kw IS NULL OR contract_capacity_kw > 0
    ),
    CONSTRAINT chk_p037_fp_curtailable CHECK (
        curtailable_capacity_kw IS NULL OR curtailable_capacity_kw >= 0
    ),
    CONSTRAINT chk_p037_fp_base_load CHECK (
        base_load_kw IS NULL OR base_load_kw >= 0
    ),
    CONSTRAINT chk_p037_fp_latitude CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    ),
    CONSTRAINT chk_p037_fp_longitude CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    ),
    CONSTRAINT chk_p037_fp_facility_type CHECK (
        facility_type IS NULL OR facility_type IN (
            'OFFICE', 'RETAIL', 'WAREHOUSE', 'MANUFACTURING', 'DATA_CENTER',
            'HOSPITAL', 'UNIVERSITY', 'HOTEL', 'COLD_STORAGE', 'WATER_TREATMENT',
            'PUMPING_STATION', 'EV_CHARGING', 'RESIDENTIAL_AGGREGATED', 'OTHER'
        )
    ),
    CONSTRAINT uq_p037_fp_tenant_facility UNIQUE (tenant_id, facility_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_fp_tenant          ON pack037_demand_response.dr_facility_profiles(tenant_id);
CREATE INDEX idx_p037_fp_facility        ON pack037_demand_response.dr_facility_profiles(facility_id);
CREATE INDEX idx_p037_fp_region          ON pack037_demand_response.dr_facility_profiles(iso_rto_region);
CREATE INDEX idx_p037_fp_readiness       ON pack037_demand_response.dr_facility_profiles(dr_readiness_level);
CREATE INDEX idx_p037_fp_automation      ON pack037_demand_response.dr_facility_profiles(automation_level);
CREATE INDEX idx_p037_fp_country         ON pack037_demand_response.dr_facility_profiles(country_code);
CREATE INDEX idx_p037_fp_peak            ON pack037_demand_response.dr_facility_profiles(peak_demand_kw DESC);
CREATE INDEX idx_p037_fp_curtailable     ON pack037_demand_response.dr_facility_profiles(curtailable_capacity_kw DESC);
CREATE INDEX idx_p037_fp_created         ON pack037_demand_response.dr_facility_profiles(created_at DESC);
CREATE INDEX idx_p037_fp_metadata        ON pack037_demand_response.dr_facility_profiles USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_fp_updated
    BEFORE UPDATE ON pack037_demand_response.dr_facility_profiles
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack037_demand_response.dr_load_inventory
-- =============================================================================
-- Individual controllable load assets within a facility. Each load is a
-- discrete piece of equipment (HVAC unit, lighting circuit, process line)
-- that can be curtailed, shifted, or shed during DR events.

CREATE TABLE pack037_demand_response.dr_load_inventory (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    load_name               VARCHAR(255)    NOT NULL,
    load_category           VARCHAR(50)     NOT NULL,
    equipment_type          VARCHAR(100),
    rated_capacity_kw       NUMERIC(12,4)   NOT NULL,
    typical_demand_kw       NUMERIC(12,4),
    curtailable_kw          NUMERIC(12,4),
    min_demand_kw           NUMERIC(12,4)   DEFAULT 0,
    ramp_rate_kw_min        NUMERIC(10,4),
    ramp_down_time_min      INTEGER,
    ramp_up_time_min        INTEGER,
    min_runtime_min         INTEGER,
    min_downtime_min        INTEGER,
    max_curtailment_duration_min INTEGER,
    is_interruptible        BOOLEAN         NOT NULL DEFAULT true,
    is_shiftable            BOOLEAN         NOT NULL DEFAULT false,
    comfort_impact          VARCHAR(20)     DEFAULT 'NONE',
    production_impact       VARCHAR(20)     DEFAULT 'NONE',
    safety_critical         BOOLEAN         DEFAULT false,
    control_protocol        VARCHAR(50),
    control_point_id        VARCHAR(100),
    priority_order          INTEGER         DEFAULT 100,
    load_status             VARCHAR(20)     NOT NULL DEFAULT 'AVAILABLE',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_li_category CHECK (
        load_category IN (
            'HVAC', 'LIGHTING', 'PROCESS', 'REFRIGERATION', 'PUMPING',
            'COMPRESSED_AIR', 'EV_CHARGING', 'WATER_HEATING', 'SPACE_HEATING',
            'VENTILATION', 'ELEVATOR', 'IT_LOAD', 'BATTERY_STORAGE',
            'THERMAL_STORAGE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p037_li_rated CHECK (
        rated_capacity_kw > 0
    ),
    CONSTRAINT chk_p037_li_curtailable CHECK (
        curtailable_kw IS NULL OR curtailable_kw >= 0
    ),
    CONSTRAINT chk_p037_li_min_demand CHECK (
        min_demand_kw >= 0
    ),
    CONSTRAINT chk_p037_li_comfort CHECK (
        comfort_impact IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p037_li_production CHECK (
        production_impact IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p037_li_status CHECK (
        load_status IN ('AVAILABLE', 'COMMITTED', 'CURTAILED', 'OFFLINE', 'MAINTENANCE')
    ),
    CONSTRAINT chk_p037_li_control CHECK (
        control_protocol IS NULL OR control_protocol IN (
            'OPENADR_20B', 'OPENADR_20A', 'BACNET', 'MODBUS', 'LONWORKS',
            'IEEE_2030_5', 'ECHONET_LITE', 'MQTT', 'REST_API', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p037_li_priority CHECK (
        priority_order >= 1 AND priority_order <= 999
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_li_facility        ON pack037_demand_response.dr_load_inventory(facility_profile_id);
CREATE INDEX idx_p037_li_tenant          ON pack037_demand_response.dr_load_inventory(tenant_id);
CREATE INDEX idx_p037_li_category        ON pack037_demand_response.dr_load_inventory(load_category);
CREATE INDEX idx_p037_li_status          ON pack037_demand_response.dr_load_inventory(load_status);
CREATE INDEX idx_p037_li_curtailable     ON pack037_demand_response.dr_load_inventory(curtailable_kw DESC);
CREATE INDEX idx_p037_li_priority        ON pack037_demand_response.dr_load_inventory(priority_order);
CREATE INDEX idx_p037_li_interruptible   ON pack037_demand_response.dr_load_inventory(is_interruptible);
CREATE INDEX idx_p037_li_created         ON pack037_demand_response.dr_load_inventory(created_at DESC);

-- Composite: facility + available loads sorted by priority for dispatch
CREATE INDEX idx_p037_li_fac_avail_prio  ON pack037_demand_response.dr_load_inventory(facility_profile_id, priority_order)
    WHERE load_status = 'AVAILABLE' AND is_interruptible = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_li_updated
    BEFORE UPDATE ON pack037_demand_response.dr_load_inventory
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack037_demand_response.dr_load_flexibility
-- =============================================================================
-- Assessed flexibility characteristics of each load asset including
-- temporal availability windows, seasonal patterns, and flexibility
-- scoring for dispatch optimization.

CREATE TABLE pack037_demand_response.dr_load_flexibility (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    load_id                 UUID            NOT NULL REFERENCES pack037_demand_response.dr_load_inventory(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    season                  VARCHAR(20)     NOT NULL,
    day_type                VARCHAR(20)     NOT NULL DEFAULT 'WEEKDAY',
    available_start_hour    INTEGER         NOT NULL DEFAULT 0,
    available_end_hour      INTEGER         NOT NULL DEFAULT 23,
    max_curtailment_kw      NUMERIC(12,4)   NOT NULL,
    expected_curtailment_kw NUMERIC(12,4),
    max_duration_min        INTEGER         NOT NULL,
    min_notice_min          INTEGER         NOT NULL DEFAULT 30,
    recovery_time_min       INTEGER,
    rebound_factor          NUMERIC(5,3)    DEFAULT 1.0,
    flexibility_score       NUMERIC(5,2),
    reliability_score       NUMERIC(5,2),
    cost_per_kw_curtailed   NUMERIC(10,4),
    comfort_degradation_pct NUMERIC(5,2),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_lf_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_lf_day_type CHECK (
        day_type IN ('WEEKDAY', 'WEEKEND', 'HOLIDAY', 'ALL_DAYS')
    ),
    CONSTRAINT chk_p037_lf_hours CHECK (
        available_start_hour >= 0 AND available_start_hour <= 23 AND
        available_end_hour >= 0 AND available_end_hour <= 23
    ),
    CONSTRAINT chk_p037_lf_max_curt CHECK (
        max_curtailment_kw >= 0
    ),
    CONSTRAINT chk_p037_lf_duration CHECK (
        max_duration_min > 0
    ),
    CONSTRAINT chk_p037_lf_notice CHECK (
        min_notice_min >= 0
    ),
    CONSTRAINT chk_p037_lf_rebound CHECK (
        rebound_factor IS NULL OR (rebound_factor >= 0 AND rebound_factor <= 3.0)
    ),
    CONSTRAINT chk_p037_lf_flex_score CHECK (
        flexibility_score IS NULL OR (flexibility_score >= 0 AND flexibility_score <= 100)
    ),
    CONSTRAINT chk_p037_lf_rel_score CHECK (
        reliability_score IS NULL OR (reliability_score >= 0 AND reliability_score <= 100)
    ),
    CONSTRAINT chk_p037_lf_comfort CHECK (
        comfort_degradation_pct IS NULL OR (comfort_degradation_pct >= 0 AND comfort_degradation_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_lf_load            ON pack037_demand_response.dr_load_flexibility(load_id);
CREATE INDEX idx_p037_lf_tenant          ON pack037_demand_response.dr_load_flexibility(tenant_id);
CREATE INDEX idx_p037_lf_season          ON pack037_demand_response.dr_load_flexibility(season);
CREATE INDEX idx_p037_lf_day_type        ON pack037_demand_response.dr_load_flexibility(day_type);
CREATE INDEX idx_p037_lf_assessment      ON pack037_demand_response.dr_load_flexibility(assessment_date DESC);
CREATE INDEX idx_p037_lf_flex_score      ON pack037_demand_response.dr_load_flexibility(flexibility_score DESC);
CREATE INDEX idx_p037_lf_max_curt        ON pack037_demand_response.dr_load_flexibility(max_curtailment_kw DESC);

-- Composite: load + season + day for dispatch lookups
CREATE INDEX idx_p037_lf_load_season     ON pack037_demand_response.dr_load_flexibility(load_id, season, day_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_lf_updated
    BEFORE UPDATE ON pack037_demand_response.dr_load_flexibility
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack037_demand_response.dr_curtailment_capacity
-- =============================================================================
-- Aggregated curtailment capacity per facility over time. Summarises total
-- available, committed, and verified curtailment for each DR season.

CREATE TABLE pack037_demand_response.dr_curtailment_capacity (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    season_year             INTEGER         NOT NULL,
    season                  VARCHAR(20)     NOT NULL,
    total_available_kw      NUMERIC(12,4)   NOT NULL,
    committed_kw            NUMERIC(12,4)   NOT NULL DEFAULT 0,
    verified_kw             NUMERIC(12,4),
    test_result_kw          NUMERIC(12,4),
    test_date               DATE,
    verification_method     VARCHAR(50),
    firm_service_level_kw   NUMERIC(12,4),
    avg_response_time_min   NUMERIC(8,2),
    confidence_level_pct    NUMERIC(5,2),
    status                  VARCHAR(30)     NOT NULL DEFAULT 'ESTIMATED',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_cc_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_cc_year CHECK (
        season_year >= 2020 AND season_year <= 2100
    ),
    CONSTRAINT chk_p037_cc_available CHECK (
        total_available_kw >= 0
    ),
    CONSTRAINT chk_p037_cc_committed CHECK (
        committed_kw >= 0
    ),
    CONSTRAINT chk_p037_cc_verified CHECK (
        verified_kw IS NULL OR verified_kw >= 0
    ),
    CONSTRAINT chk_p037_cc_confidence CHECK (
        confidence_level_pct IS NULL OR (confidence_level_pct >= 0 AND confidence_level_pct <= 100)
    ),
    CONSTRAINT chk_p037_cc_status CHECK (
        status IN (
            'ESTIMATED', 'TESTED', 'VERIFIED', 'COMMITTED', 'EXPIRED'
        )
    ),
    CONSTRAINT chk_p037_cc_verification CHECK (
        verification_method IS NULL OR verification_method IN (
            'BASELINE_COMPARISON', 'METERING_BEFORE_AFTER', 'REGRESSION_MODEL',
            'WHOLE_FACILITY', 'SUB_METERED', 'DEEMED_SAVINGS'
        )
    ),
    CONSTRAINT uq_p037_cc_facility_season UNIQUE (facility_profile_id, season_year, season)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_cc_facility        ON pack037_demand_response.dr_curtailment_capacity(facility_profile_id);
CREATE INDEX idx_p037_cc_tenant          ON pack037_demand_response.dr_curtailment_capacity(tenant_id);
CREATE INDEX idx_p037_cc_season          ON pack037_demand_response.dr_curtailment_capacity(season_year, season);
CREATE INDEX idx_p037_cc_status          ON pack037_demand_response.dr_curtailment_capacity(status);
CREATE INDEX idx_p037_cc_available       ON pack037_demand_response.dr_curtailment_capacity(total_available_kw DESC);
CREATE INDEX idx_p037_cc_committed       ON pack037_demand_response.dr_curtailment_capacity(committed_kw DESC);
CREATE INDEX idx_p037_cc_created         ON pack037_demand_response.dr_curtailment_capacity(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_cc_updated
    BEFORE UPDATE ON pack037_demand_response.dr_curtailment_capacity
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack037_demand_response.dr_flexibility_registers
-- =============================================================================
-- Aggregated flexibility register entries that track the total flexible
-- capacity across load portfolios for a given time period, used for
-- market bidding and program compliance reporting.

CREATE TABLE pack037_demand_response.dr_flexibility_registers (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    register_date           DATE            NOT NULL,
    register_hour           INTEGER         NOT NULL,
    total_flexible_kw       NUMERIC(12,4)   NOT NULL,
    upward_flexible_kw      NUMERIC(12,4),
    downward_flexible_kw    NUMERIC(12,4),
    response_time_category  VARCHAR(30),
    assets_available        INTEGER         NOT NULL DEFAULT 0,
    assets_committed        INTEGER         NOT NULL DEFAULT 0,
    bid_price_per_kw        NUMERIC(10,4),
    market_cleared          BOOLEAN         DEFAULT false,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_fr_hour CHECK (
        register_hour >= 0 AND register_hour <= 23
    ),
    CONSTRAINT chk_p037_fr_flexible CHECK (
        total_flexible_kw >= 0
    ),
    CONSTRAINT chk_p037_fr_upward CHECK (
        upward_flexible_kw IS NULL OR upward_flexible_kw >= 0
    ),
    CONSTRAINT chk_p037_fr_downward CHECK (
        downward_flexible_kw IS NULL OR downward_flexible_kw >= 0
    ),
    CONSTRAINT chk_p037_fr_response CHECK (
        response_time_category IS NULL OR response_time_category IN (
            'INSTANT', 'FAST_5MIN', 'STANDARD_30MIN', 'LONG_60MIN', 'DAY_AHEAD'
        )
    ),
    CONSTRAINT chk_p037_fr_assets_avail CHECK (
        assets_available >= 0
    ),
    CONSTRAINT chk_p037_fr_assets_commit CHECK (
        assets_committed >= 0
    ),
    CONSTRAINT uq_p037_fr_facility_date_hour UNIQUE (facility_profile_id, register_date, register_hour)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_fr_facility        ON pack037_demand_response.dr_flexibility_registers(facility_profile_id);
CREATE INDEX idx_p037_fr_tenant          ON pack037_demand_response.dr_flexibility_registers(tenant_id);
CREATE INDEX idx_p037_fr_date            ON pack037_demand_response.dr_flexibility_registers(register_date DESC);
CREATE INDEX idx_p037_fr_flexible        ON pack037_demand_response.dr_flexibility_registers(total_flexible_kw DESC);
CREATE INDEX idx_p037_fr_response_cat    ON pack037_demand_response.dr_flexibility_registers(response_time_category);
CREATE INDEX idx_p037_fr_cleared         ON pack037_demand_response.dr_flexibility_registers(market_cleared);
CREATE INDEX idx_p037_fr_created         ON pack037_demand_response.dr_flexibility_registers(created_at DESC);

-- Composite: date + hour for time-series flexibility queries
CREATE INDEX idx_p037_fr_date_hour       ON pack037_demand_response.dr_flexibility_registers(register_date, register_hour);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_facility_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_load_inventory ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_load_flexibility ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_curtailment_capacity ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_flexibility_registers ENABLE ROW LEVEL SECURITY;

CREATE POLICY p037_fp_tenant_isolation
    ON pack037_demand_response.dr_facility_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_fp_service_bypass
    ON pack037_demand_response.dr_facility_profiles
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_li_tenant_isolation
    ON pack037_demand_response.dr_load_inventory
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_li_service_bypass
    ON pack037_demand_response.dr_load_inventory
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_lf_tenant_isolation
    ON pack037_demand_response.dr_load_flexibility
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_lf_service_bypass
    ON pack037_demand_response.dr_load_flexibility
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_cc_tenant_isolation
    ON pack037_demand_response.dr_curtailment_capacity
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_cc_service_bypass
    ON pack037_demand_response.dr_curtailment_capacity
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_fr_tenant_isolation
    ON pack037_demand_response.dr_flexibility_registers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_fr_service_bypass
    ON pack037_demand_response.dr_flexibility_registers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack037_demand_response TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_facility_profiles TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_load_inventory TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_load_flexibility TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_curtailment_capacity TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_flexibility_registers TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack037_demand_response.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack037_demand_response IS
    'PACK-037 Demand Response Pack - demand response program enrollment, event management, dispatch optimization, DER coordination, performance tracking, revenue analysis, and carbon impact assessment.';

COMMENT ON TABLE pack037_demand_response.dr_facility_profiles IS
    'Facility-level demand response profiles with peak demand, ISO/RTO region, DR readiness, automation level, and EMS integration status.';
COMMENT ON TABLE pack037_demand_response.dr_load_inventory IS
    'Individual controllable load assets within a facility that can be curtailed, shifted, or shed during DR events.';
COMMENT ON TABLE pack037_demand_response.dr_load_flexibility IS
    'Assessed flexibility characteristics of load assets including temporal availability, seasonal patterns, and flexibility scoring.';
COMMENT ON TABLE pack037_demand_response.dr_curtailment_capacity IS
    'Aggregated curtailment capacity per facility over time with verification status and test results.';
COMMENT ON TABLE pack037_demand_response.dr_flexibility_registers IS
    'Hourly flexibility register entries tracking total flexible capacity for market bidding and compliance reporting.';

COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.id IS 'Unique identifier for the facility DR profile.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.facility_id IS 'Reference to the facility in the core facility registry.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.iso_rto_region IS 'ISO/RTO region: PJM, ERCOT, CAISO, ISO_NE, NYISO, MISO, SPP, or European TSO codes.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.peak_demand_kw IS 'Historical peak demand in kW used for sizing DR commitments.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.curtailable_capacity_kw IS 'Assessed curtailable capacity in kW available for DR events.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.dr_readiness_level IS 'DR readiness classification: ASSESSMENT, ENROLLED, QUALIFIED, ACTIVE, SUSPENDED, RETIRED.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.automation_level IS 'Level of DR automation: MANUAL, SEMI_AUTO, FULLY_AUTO, ADR_OPENADR, ADR_IEEE2030.';
COMMENT ON COLUMN pack037_demand_response.dr_facility_profiles.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_load_inventory.load_category IS 'Load category: HVAC, LIGHTING, PROCESS, REFRIGERATION, PUMPING, EV_CHARGING, BATTERY_STORAGE, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_load_inventory.curtailable_kw IS 'Maximum curtailable capacity of this load in kW.';
COMMENT ON COLUMN pack037_demand_response.dr_load_inventory.comfort_impact IS 'Impact on occupant comfort if this load is curtailed: NONE, LOW, MEDIUM, HIGH, CRITICAL.';
COMMENT ON COLUMN pack037_demand_response.dr_load_inventory.safety_critical IS 'Whether this load is safety-critical and must not be curtailed.';
COMMENT ON COLUMN pack037_demand_response.dr_load_inventory.priority_order IS 'Dispatch priority order (1=first to curtail, 999=last). Lower values are curtailed first.';
COMMENT ON COLUMN pack037_demand_response.dr_load_inventory.control_protocol IS 'Communication protocol used for automated load control.';

COMMENT ON COLUMN pack037_demand_response.dr_load_flexibility.rebound_factor IS 'Post-event energy rebound multiplier (1.0 = no rebound, 1.5 = 50% rebound energy).';
COMMENT ON COLUMN pack037_demand_response.dr_load_flexibility.flexibility_score IS 'Composite flexibility score 0-100 based on capacity, availability, response time, and reliability.';
COMMENT ON COLUMN pack037_demand_response.dr_load_flexibility.min_notice_min IS 'Minimum advance notice required before curtailing this load, in minutes.';

COMMENT ON COLUMN pack037_demand_response.dr_curtailment_capacity.verification_method IS 'Method used to verify curtailment capacity: BASELINE_COMPARISON, METERING_BEFORE_AFTER, REGRESSION_MODEL, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_curtailment_capacity.confidence_level_pct IS 'Statistical confidence level of the verified curtailment capacity (0-100%).';

COMMENT ON COLUMN pack037_demand_response.dr_flexibility_registers.response_time_category IS 'Response time category: INSTANT, FAST_5MIN, STANDARD_30MIN, LONG_60MIN, DAY_AHEAD.';
COMMENT ON COLUMN pack037_demand_response.dr_flexibility_registers.bid_price_per_kw IS 'Bid price per kW for market participation.';
