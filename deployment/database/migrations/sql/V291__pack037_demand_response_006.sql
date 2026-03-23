-- =============================================================================
-- V291: PACK-037 Demand Response Pack - DER Coordination
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Distributed Energy Resource (DER) coordination tables for battery storage,
-- solar PV, EV chargers, thermal storage, and other behind-the-meter
-- assets that participate in demand response. Tracks asset specifications,
-- dispatch plans, state-of-charge, performance, and degradation.
--
-- Tables (5):
--   1. pack037_demand_response.dr_der_assets
--   2. pack037_demand_response.dr_der_dispatch_plans
--   3. pack037_demand_response.dr_der_soc_tracking
--   4. pack037_demand_response.dr_der_performance
--   5. pack037_demand_response.dr_der_degradation
--
-- Previous: V290__pack037_demand_response_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_der_assets
-- =============================================================================
-- DER asset registry tracking battery storage, solar PV, EV chargers,
-- thermal storage, backup generators, and other distributed assets
-- available for demand response coordination.

CREATE TABLE pack037_demand_response.dr_der_assets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    asset_name              VARCHAR(255)    NOT NULL,
    asset_type              VARCHAR(50)     NOT NULL,
    manufacturer            VARCHAR(255),
    model_number            VARCHAR(100),
    serial_number           VARCHAR(100),
    installation_date       DATE,
    rated_capacity_kw       NUMERIC(12,4)   NOT NULL,
    rated_energy_kwh        NUMERIC(14,4),
    max_charge_rate_kw      NUMERIC(12,4),
    max_discharge_rate_kw   NUMERIC(12,4),
    min_soc_pct             NUMERIC(5,2)    DEFAULT 10,
    max_soc_pct             NUMERIC(5,2)    DEFAULT 95,
    round_trip_efficiency   NUMERIC(5,4)    DEFAULT 0.90,
    inverter_type           VARCHAR(50),
    inverter_capacity_kva   NUMERIC(12,4),
    connection_type         VARCHAR(30),
    voltage_level           VARCHAR(20),
    control_protocol        VARCHAR(50),
    control_endpoint        VARCHAR(255),
    grid_export_allowed     BOOLEAN         DEFAULT false,
    grid_export_limit_kw    NUMERIC(12,4),
    islanding_capable       BOOLEAN         DEFAULT false,
    warranty_expiry         DATE,
    current_soc_pct         NUMERIC(5,2),
    current_status          VARCHAR(20)     NOT NULL DEFAULT 'ONLINE',
    last_telemetry          TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_der_type CHECK (
        asset_type IN (
            'BATTERY_LITHIUM_ION', 'BATTERY_FLOW', 'BATTERY_LEAD_ACID',
            'SOLAR_PV', 'SOLAR_PV_BATTERY', 'WIND_TURBINE',
            'EV_CHARGER_L2', 'EV_CHARGER_DCFC', 'EV_FLEET',
            'THERMAL_ICE', 'THERMAL_CHILLED_WATER', 'THERMAL_HOT_WATER',
            'BACKUP_GENERATOR_DIESEL', 'BACKUP_GENERATOR_GAS',
            'FUEL_CELL', 'FLYWHEEL', 'SUPERCAPACITOR', 'CHP', 'OTHER'
        )
    ),
    CONSTRAINT chk_p037_der_capacity CHECK (
        rated_capacity_kw > 0
    ),
    CONSTRAINT chk_p037_der_energy CHECK (
        rated_energy_kwh IS NULL OR rated_energy_kwh > 0
    ),
    CONSTRAINT chk_p037_der_charge CHECK (
        max_charge_rate_kw IS NULL OR max_charge_rate_kw > 0
    ),
    CONSTRAINT chk_p037_der_discharge CHECK (
        max_discharge_rate_kw IS NULL OR max_discharge_rate_kw > 0
    ),
    CONSTRAINT chk_p037_der_min_soc CHECK (
        min_soc_pct >= 0 AND min_soc_pct <= 100
    ),
    CONSTRAINT chk_p037_der_max_soc CHECK (
        max_soc_pct >= 0 AND max_soc_pct <= 100 AND max_soc_pct > min_soc_pct
    ),
    CONSTRAINT chk_p037_der_efficiency CHECK (
        round_trip_efficiency IS NULL OR (round_trip_efficiency > 0 AND round_trip_efficiency <= 1.0)
    ),
    CONSTRAINT chk_p037_der_soc CHECK (
        current_soc_pct IS NULL OR (current_soc_pct >= 0 AND current_soc_pct <= 100)
    ),
    CONSTRAINT chk_p037_der_status CHECK (
        current_status IN (
            'ONLINE', 'OFFLINE', 'CHARGING', 'DISCHARGING', 'STANDBY',
            'MAINTENANCE', 'FAULT', 'ISLANDED', 'DECOMMISSIONED'
        )
    ),
    CONSTRAINT chk_p037_der_connection CHECK (
        connection_type IS NULL OR connection_type IN (
            'BEHIND_METER', 'FRONT_OF_METER', 'MICROGRID', 'VIRTUAL'
        )
    ),
    CONSTRAINT chk_p037_der_voltage CHECK (
        voltage_level IS NULL OR voltage_level IN (
            'LOW_120V', 'LOW_240V', 'LOW_480V', 'MEDIUM_4KV',
            'MEDIUM_13KV', 'MEDIUM_34KV', 'HIGH'
        )
    ),
    CONSTRAINT chk_p037_der_protocol CHECK (
        control_protocol IS NULL OR control_protocol IN (
            'OPENADR_20B', 'IEEE_2030_5', 'SUNSPEC_MODBUS', 'OCPP',
            'MQTT', 'REST_API', 'BACNET', 'DNP3', 'IEC_61850', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p037_der_inverter CHECK (
        inverter_type IS NULL OR inverter_type IN (
            'STRING', 'CENTRAL', 'MICRO', 'HYBRID', 'BIDIRECTIONAL'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_der_facility       ON pack037_demand_response.dr_der_assets(facility_profile_id);
CREATE INDEX idx_p037_der_tenant         ON pack037_demand_response.dr_der_assets(tenant_id);
CREATE INDEX idx_p037_der_type           ON pack037_demand_response.dr_der_assets(asset_type);
CREATE INDEX idx_p037_der_status         ON pack037_demand_response.dr_der_assets(current_status);
CREATE INDEX idx_p037_der_capacity       ON pack037_demand_response.dr_der_assets(rated_capacity_kw DESC);
CREATE INDEX idx_p037_der_soc            ON pack037_demand_response.dr_der_assets(current_soc_pct);
CREATE INDEX idx_p037_der_protocol       ON pack037_demand_response.dr_der_assets(control_protocol);
CREATE INDEX idx_p037_der_created        ON pack037_demand_response.dr_der_assets(created_at DESC);
CREATE INDEX idx_p037_der_metadata       ON pack037_demand_response.dr_der_assets USING GIN(metadata);

-- Composite: facility + online assets for dispatch readiness
CREATE INDEX idx_p037_der_fac_online     ON pack037_demand_response.dr_der_assets(facility_profile_id, asset_type)
    WHERE current_status IN ('ONLINE', 'STANDBY', 'CHARGING');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_der_updated
    BEFORE UPDATE ON pack037_demand_response.dr_der_assets
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack037_demand_response.dr_der_dispatch_plans
-- =============================================================================
-- DER-specific dispatch plans defining how each asset should be operated
-- during DR events, including charge/discharge schedules, export limits,
-- and coordination with building loads.

CREATE TABLE pack037_demand_response.dr_der_dispatch_plans (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    der_asset_id            UUID            NOT NULL REFERENCES pack037_demand_response.dr_der_assets(id) ON DELETE CASCADE,
    dispatch_plan_id        UUID            REFERENCES pack037_demand_response.dr_dispatch_plans(id),
    tenant_id               UUID            NOT NULL,
    plan_name               VARCHAR(255)    NOT NULL,
    strategy                VARCHAR(50)     NOT NULL,
    pre_event_target_soc_pct NUMERIC(5,2),
    event_discharge_kw      NUMERIC(12,4),
    event_discharge_duration_min INTEGER,
    min_soc_during_event_pct NUMERIC(5,2),
    post_event_recharge     BOOLEAN         DEFAULT true,
    recharge_rate_kw        NUMERIC(12,4),
    export_during_event     BOOLEAN         DEFAULT false,
    export_limit_kw         NUMERIC(12,4),
    coordination_mode       VARCHAR(30)     NOT NULL DEFAULT 'LOAD_FOLLOWING',
    priority_order          INTEGER         DEFAULT 50,
    plan_status             VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_ddp_strategy CHECK (
        strategy IN (
            'PEAK_SHAVING', 'LOAD_SHIFTING', 'BACKUP_POWER',
            'FREQUENCY_REGULATION', 'ARBITRAGE', 'SELF_CONSUMPTION',
            'GRID_EXPORT', 'DEMAND_RESPONSE', 'ISLANDING'
        )
    ),
    CONSTRAINT chk_p037_ddp_soc CHECK (
        pre_event_target_soc_pct IS NULL OR (pre_event_target_soc_pct >= 0 AND pre_event_target_soc_pct <= 100)
    ),
    CONSTRAINT chk_p037_ddp_min_soc CHECK (
        min_soc_during_event_pct IS NULL OR (min_soc_during_event_pct >= 0 AND min_soc_during_event_pct <= 100)
    ),
    CONSTRAINT chk_p037_ddp_discharge CHECK (
        event_discharge_kw IS NULL OR event_discharge_kw >= 0
    ),
    CONSTRAINT chk_p037_ddp_coordination CHECK (
        coordination_mode IN (
            'LOAD_FOLLOWING', 'FIXED_OUTPUT', 'PRICE_RESPONSIVE',
            'GRID_FORMING', 'VIRTUAL_POWER_PLANT'
        )
    ),
    CONSTRAINT chk_p037_ddp_status CHECK (
        plan_status IN ('DRAFT', 'APPROVED', 'ACTIVE', 'ARCHIVED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p037_ddp_priority CHECK (
        priority_order >= 1 AND priority_order <= 999
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_ddp_asset          ON pack037_demand_response.dr_der_dispatch_plans(der_asset_id);
CREATE INDEX idx_p037_ddp_plan           ON pack037_demand_response.dr_der_dispatch_plans(dispatch_plan_id);
CREATE INDEX idx_p037_ddp_tenant         ON pack037_demand_response.dr_der_dispatch_plans(tenant_id);
CREATE INDEX idx_p037_ddp_strategy       ON pack037_demand_response.dr_der_dispatch_plans(strategy);
CREATE INDEX idx_p037_ddp_status         ON pack037_demand_response.dr_der_dispatch_plans(plan_status);
CREATE INDEX idx_p037_ddp_coordination   ON pack037_demand_response.dr_der_dispatch_plans(coordination_mode);
CREATE INDEX idx_p037_ddp_created        ON pack037_demand_response.dr_der_dispatch_plans(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_ddp_updated
    BEFORE UPDATE ON pack037_demand_response.dr_der_dispatch_plans
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack037_demand_response.dr_der_soc_tracking
-- =============================================================================
-- Time-series state-of-charge tracking for energy storage assets.
-- Captures SOC, power flow, temperature, and operating mode at
-- regular intervals for monitoring and analytics.

CREATE TABLE pack037_demand_response.dr_der_soc_tracking (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    der_asset_id            UUID            NOT NULL REFERENCES pack037_demand_response.dr_der_assets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    recorded_at             TIMESTAMPTZ     NOT NULL,
    soc_pct                 NUMERIC(5,2)    NOT NULL,
    power_kw                NUMERIC(12,4),
    energy_throughput_kwh   NUMERIC(14,4),
    voltage_v               NUMERIC(8,2),
    current_a               NUMERIC(10,4),
    temperature_c           NUMERIC(6,2),
    operating_mode          VARCHAR(20)     NOT NULL,
    is_dr_event             BOOLEAN         DEFAULT false,
    event_id                UUID            REFERENCES pack037_demand_response.dr_events(id),
    data_quality            VARCHAR(20)     DEFAULT 'ACTUAL',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_soc_pct CHECK (
        soc_pct >= 0 AND soc_pct <= 100
    ),
    CONSTRAINT chk_p037_soc_mode CHECK (
        operating_mode IN (
            'IDLE', 'CHARGING', 'DISCHARGING', 'STANDBY',
            'FAULT', 'MAINTENANCE', 'ISLANDED'
        )
    ),
    CONSTRAINT chk_p037_soc_quality CHECK (
        data_quality IN ('ACTUAL', 'ESTIMATED', 'INTERPOLATED', 'MISSING')
    ),
    CONSTRAINT uq_p037_soc_asset_ts UNIQUE (der_asset_id, recorded_at)
);

-- ---------------------------------------------------------------------------
-- Indexes (BRIN for time-series)
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_soc_asset          ON pack037_demand_response.dr_der_soc_tracking(der_asset_id);
CREATE INDEX idx_p037_soc_tenant         ON pack037_demand_response.dr_der_soc_tracking(tenant_id);
CREATE INDEX idx_p037_soc_recorded       ON pack037_demand_response.dr_der_soc_tracking USING BRIN(recorded_at);
CREATE INDEX idx_p037_soc_mode           ON pack037_demand_response.dr_der_soc_tracking(operating_mode);
CREATE INDEX idx_p037_soc_event          ON pack037_demand_response.dr_der_soc_tracking(event_id);
CREATE INDEX idx_p037_soc_is_dr          ON pack037_demand_response.dr_der_soc_tracking(is_dr_event);

-- Composite: asset + time for time-series queries
CREATE INDEX idx_p037_soc_asset_ts       ON pack037_demand_response.dr_der_soc_tracking(der_asset_id, recorded_at DESC);

-- ---------------------------------------------------------------------------
-- TimescaleDB hypertable conversion (if extension is available)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'pack037_demand_response.dr_der_soc_tracking',
            'recorded_at',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE,
            migrate_data => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertable created for dr_der_soc_tracking';
    ELSE
        RAISE NOTICE 'TimescaleDB not available - dr_der_soc_tracking remains a standard table with BRIN index';
    END IF;
END;
$$;

-- =============================================================================
-- Table 4: pack037_demand_response.dr_der_performance
-- =============================================================================
-- DER performance metrics during and after DR events. Tracks energy
-- delivered, efficiency, response time, and contribution to overall
-- facility curtailment.

CREATE TABLE pack037_demand_response.dr_der_performance (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    der_asset_id            UUID            NOT NULL REFERENCES pack037_demand_response.dr_der_assets(id) ON DELETE CASCADE,
    event_id                UUID            NOT NULL REFERENCES pack037_demand_response.dr_events(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    target_output_kw        NUMERIC(12,4)   NOT NULL,
    actual_output_kw        NUMERIC(12,4)   NOT NULL,
    energy_delivered_kwh    NUMERIC(14,4),
    start_soc_pct           NUMERIC(5,2),
    end_soc_pct             NUMERIC(5,2),
    response_time_seconds   INTEGER,
    ramp_rate_achieved_kw_s NUMERIC(10,4),
    round_trip_efficiency   NUMERIC(5,4),
    performance_ratio       NUMERIC(6,4)    NOT NULL,
    contribution_pct        NUMERIC(5,2),
    recharge_energy_kwh     NUMERIC(14,4),
    recharge_duration_min   INTEGER,
    issues_encountered      TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_dp2_target CHECK (
        target_output_kw >= 0
    ),
    CONSTRAINT chk_p037_dp2_actual CHECK (
        actual_output_kw >= 0
    ),
    CONSTRAINT chk_p037_dp2_perf CHECK (
        performance_ratio >= 0
    ),
    CONSTRAINT chk_p037_dp2_contribution CHECK (
        contribution_pct IS NULL OR (contribution_pct >= 0 AND contribution_pct <= 100)
    ),
    CONSTRAINT chk_p037_dp2_efficiency CHECK (
        round_trip_efficiency IS NULL OR (round_trip_efficiency > 0 AND round_trip_efficiency <= 1.0)
    ),
    CONSTRAINT uq_p037_dp2_asset_event UNIQUE (der_asset_id, event_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_dp2_asset          ON pack037_demand_response.dr_der_performance(der_asset_id);
CREATE INDEX idx_p037_dp2_event          ON pack037_demand_response.dr_der_performance(event_id);
CREATE INDEX idx_p037_dp2_tenant         ON pack037_demand_response.dr_der_performance(tenant_id);
CREATE INDEX idx_p037_dp2_perf           ON pack037_demand_response.dr_der_performance(performance_ratio DESC);
CREATE INDEX idx_p037_dp2_created        ON pack037_demand_response.dr_der_performance(created_at DESC);

-- =============================================================================
-- Table 5: pack037_demand_response.dr_der_degradation
-- =============================================================================
-- Battery and storage asset degradation tracking over time. Monitors
-- state-of-health, cycle counts, and projected end-of-life to inform
-- dispatch decisions and replacement planning.

CREATE TABLE pack037_demand_response.dr_der_degradation (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    der_asset_id            UUID            NOT NULL REFERENCES pack037_demand_response.dr_der_assets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    state_of_health_pct     NUMERIC(5,2)    NOT NULL,
    available_capacity_kwh  NUMERIC(14,4)   NOT NULL,
    rated_capacity_kwh      NUMERIC(14,4)   NOT NULL,
    total_cycles             INTEGER        NOT NULL DEFAULT 0,
    dr_event_cycles         INTEGER         DEFAULT 0,
    total_energy_throughput_mwh NUMERIC(14,4),
    calendar_age_months     INTEGER,
    degradation_rate_pct_yr NUMERIC(6,3),
    projected_eol_date      DATE,
    remaining_useful_life_months INTEGER,
    warranty_remaining_months INTEGER,
    temperature_avg_c       NUMERIC(6,2),
    depth_of_discharge_avg  NUMERIC(5,2),
    assessment_method       VARCHAR(30),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_dd_soh CHECK (
        state_of_health_pct >= 0 AND state_of_health_pct <= 100
    ),
    CONSTRAINT chk_p037_dd_available CHECK (
        available_capacity_kwh >= 0
    ),
    CONSTRAINT chk_p037_dd_rated CHECK (
        rated_capacity_kwh > 0
    ),
    CONSTRAINT chk_p037_dd_cycles CHECK (
        total_cycles >= 0
    ),
    CONSTRAINT chk_p037_dd_dr_cycles CHECK (
        dr_event_cycles >= 0 AND dr_event_cycles <= total_cycles
    ),
    CONSTRAINT chk_p037_dd_deg_rate CHECK (
        degradation_rate_pct_yr IS NULL OR (degradation_rate_pct_yr >= 0 AND degradation_rate_pct_yr <= 50)
    ),
    CONSTRAINT chk_p037_dd_dod CHECK (
        depth_of_discharge_avg IS NULL OR (depth_of_discharge_avg >= 0 AND depth_of_discharge_avg <= 100)
    ),
    CONSTRAINT chk_p037_dd_method CHECK (
        assessment_method IS NULL OR assessment_method IN (
            'COULOMB_COUNTING', 'IMPEDANCE_SPECTROSCOPY', 'CAPACITY_TEST',
            'OCV_CURVE', 'MODEL_BASED', 'MANUFACTURER_REPORT'
        )
    ),
    CONSTRAINT uq_p037_dd_asset_date UNIQUE (der_asset_id, assessment_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_dd_asset           ON pack037_demand_response.dr_der_degradation(der_asset_id);
CREATE INDEX idx_p037_dd_tenant          ON pack037_demand_response.dr_der_degradation(tenant_id);
CREATE INDEX idx_p037_dd_date            ON pack037_demand_response.dr_der_degradation(assessment_date DESC);
CREATE INDEX idx_p037_dd_soh             ON pack037_demand_response.dr_der_degradation(state_of_health_pct);
CREATE INDEX idx_p037_dd_eol             ON pack037_demand_response.dr_der_degradation(projected_eol_date);
CREATE INDEX idx_p037_dd_created         ON pack037_demand_response.dr_der_degradation(created_at DESC);

-- Composite: asset + date for degradation trend analysis
CREATE INDEX idx_p037_dd_asset_date      ON pack037_demand_response.dr_der_degradation(der_asset_id, assessment_date DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_der_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_der_dispatch_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_der_soc_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_der_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_der_degradation ENABLE ROW LEVEL SECURITY;

CREATE POLICY p037_der_tenant_isolation ON pack037_demand_response.dr_der_assets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_der_service_bypass ON pack037_demand_response.dr_der_assets
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_ddp_tenant_isolation ON pack037_demand_response.dr_der_dispatch_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_ddp_service_bypass ON pack037_demand_response.dr_der_dispatch_plans
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_soc_tenant_isolation ON pack037_demand_response.dr_der_soc_tracking
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_soc_service_bypass ON pack037_demand_response.dr_der_soc_tracking
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_dp2_tenant_isolation ON pack037_demand_response.dr_der_performance
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_dp2_service_bypass ON pack037_demand_response.dr_der_performance
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_dd_tenant_isolation ON pack037_demand_response.dr_der_degradation
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_dd_service_bypass ON pack037_demand_response.dr_der_degradation
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_der_assets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_der_dispatch_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_der_soc_tracking TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_der_performance TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_der_degradation TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_der_assets IS
    'DER asset registry for battery storage, solar PV, EV chargers, thermal storage, and other behind-the-meter assets.';
COMMENT ON TABLE pack037_demand_response.dr_der_dispatch_plans IS
    'DER-specific dispatch plans defining charge/discharge schedules, export limits, and coordination with building loads.';
COMMENT ON TABLE pack037_demand_response.dr_der_soc_tracking IS
    'Time-series state-of-charge tracking for energy storage assets with power flow, temperature, and operating mode.';
COMMENT ON TABLE pack037_demand_response.dr_der_performance IS
    'DER performance metrics during DR events including energy delivered, efficiency, and contribution to facility curtailment.';
COMMENT ON TABLE pack037_demand_response.dr_der_degradation IS
    'Battery and storage asset degradation tracking with state-of-health, cycle counts, and projected end-of-life.';

COMMENT ON COLUMN pack037_demand_response.dr_der_assets.asset_type IS 'DER type: BATTERY_LITHIUM_ION, SOLAR_PV, EV_CHARGER_L2, THERMAL_ICE, BACKUP_GENERATOR_DIESEL, FUEL_CELL, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_der_assets.round_trip_efficiency IS 'Round-trip efficiency for storage assets (0-1.0, typically 0.85-0.95 for Li-ion).';
COMMENT ON COLUMN pack037_demand_response.dr_der_assets.min_soc_pct IS 'Minimum allowed state-of-charge to prevent deep discharge damage.';
COMMENT ON COLUMN pack037_demand_response.dr_der_assets.grid_export_allowed IS 'Whether the asset is permitted to export power to the grid under net metering or feed-in tariff.';
COMMENT ON COLUMN pack037_demand_response.dr_der_assets.control_protocol IS 'Communication protocol: OPENADR_20B, IEEE_2030_5, SUNSPEC_MODBUS, OCPP, MQTT, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_der_assets.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_der_dispatch_plans.strategy IS 'Dispatch strategy: PEAK_SHAVING, LOAD_SHIFTING, FREQUENCY_REGULATION, ARBITRAGE, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_der_dispatch_plans.coordination_mode IS 'How the DER coordinates with loads: LOAD_FOLLOWING, FIXED_OUTPUT, PRICE_RESPONSIVE, GRID_FORMING, VPP.';

COMMENT ON COLUMN pack037_demand_response.dr_der_degradation.state_of_health_pct IS 'State of health as percentage of original rated capacity (100% = new, 80% = typical EOL threshold).';
COMMENT ON COLUMN pack037_demand_response.dr_der_degradation.projected_eol_date IS 'Projected end-of-life date when SOH reaches 80% threshold.';
COMMENT ON COLUMN pack037_demand_response.dr_der_degradation.depth_of_discharge_avg IS 'Average depth of discharge per cycle (affects degradation rate).';
