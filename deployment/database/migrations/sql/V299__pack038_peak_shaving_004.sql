-- =============================================================================
-- V299: PACK-038 Peak Shaving Pack - BESS Configuration & Dispatch
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Battery Energy Storage System configuration, dispatch simulation,
-- degradation tracking, technology comparison, and dispatch scheduling
-- tables. Models BESS assets for peak shaving with cycle-level
-- degradation, optimal dispatch windows, and multi-technology evaluation.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_bess_configurations
--   2. pack038_peak_shaving.ps_dispatch_simulations
--   3. pack038_peak_shaving.ps_degradation_tracking
--   4. pack038_peak_shaving.ps_technology_comparisons
--   5. pack038_peak_shaving.ps_dispatch_schedules
--
-- Previous: V298__pack038_peak_shaving_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_bess_configurations
-- =============================================================================
-- Battery Energy Storage System asset definitions with chemistry type,
-- capacity, power rating, state of charge limits, degradation parameters,
-- and operational constraints for peak shaving dispatch optimization.

CREATE TABLE pack038_peak_shaving.ps_bess_configurations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    bess_name               VARCHAR(255)    NOT NULL,
    asset_id                VARCHAR(100),
    chemistry               VARCHAR(30)     NOT NULL,
    manufacturer            VARCHAR(100),
    model_number            VARCHAR(100),
    serial_number           VARCHAR(100),
    capacity_kwh            NUMERIC(12,3)   NOT NULL,
    usable_capacity_kwh     NUMERIC(12,3),
    power_kw                NUMERIC(12,3)   NOT NULL,
    max_charge_kw           NUMERIC(12,3),
    max_discharge_kw        NUMERIC(12,3),
    soc_min                 NUMERIC(5,2)    NOT NULL DEFAULT 10.00,
    soc_max                 NUMERIC(5,2)    NOT NULL DEFAULT 90.00,
    soc_current             NUMERIC(5,2),
    efficiency              NUMERIC(5,4)    NOT NULL DEFAULT 0.9200,
    charge_efficiency       NUMERIC(5,4),
    discharge_efficiency    NUMERIC(5,4),
    inverter_efficiency     NUMERIC(5,4)    DEFAULT 0.9700,
    round_trip_efficiency   NUMERIC(5,4),
    c_rate_max              NUMERIC(5,3)    DEFAULT 1.000,
    cycle_life              INTEGER,
    calendar_life_years     INTEGER,
    warranty_years           INTEGER,
    warranty_throughput_mwh  NUMERIC(12,3),
    current_cycle_count     INTEGER         DEFAULT 0,
    current_soh_pct         NUMERIC(5,2)    DEFAULT 100.00,
    degradation_rate_annual NUMERIC(5,3),
    install_date            DATE,
    commissioning_date      DATE,
    dc_voltage              NUMERIC(10,3),
    ac_voltage              VARCHAR(20),
    interconnection_type    VARCHAR(30),
    grid_tied               BOOLEAN         DEFAULT true,
    behind_meter            BOOLEAN         DEFAULT true,
    auxiliary_load_kw       NUMERIC(8,3),
    thermal_management      VARCHAR(30),
    operating_temp_min_c    NUMERIC(6,2),
    operating_temp_max_c    NUMERIC(6,2),
    footprint_sqft          NUMERIC(10,2),
    weight_lbs              NUMERIC(10,2),
    total_cost              NUMERIC(12,2),
    cost_per_kwh            NUMERIC(10,2),
    cost_per_kw             NUMERIC(10,2),
    annual_om_cost          NUMERIC(10,2),
    augmentation_schedule   JSONB           DEFAULT '{}',
    bess_status             VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_bc_chemistry CHECK (
        chemistry IN (
            'LFP', 'NMC', 'NCA', 'LTO', 'SODIUM_ION', 'FLOW_VANADIUM',
            'FLOW_ZINC_BROMINE', 'FLOW_IRON', 'LEAD_ACID', 'NICKEL_CADMIUM',
            'SOLID_STATE', 'ZINC_AIR', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_bc_capacity CHECK (
        capacity_kwh > 0
    ),
    CONSTRAINT chk_p038_bc_power CHECK (
        power_kw > 0
    ),
    CONSTRAINT chk_p038_bc_soc_min CHECK (
        soc_min >= 0 AND soc_min <= 100
    ),
    CONSTRAINT chk_p038_bc_soc_max CHECK (
        soc_max >= 0 AND soc_max <= 100 AND soc_max > soc_min
    ),
    CONSTRAINT chk_p038_bc_soc_current CHECK (
        soc_current IS NULL OR (soc_current >= 0 AND soc_current <= 100)
    ),
    CONSTRAINT chk_p038_bc_efficiency CHECK (
        efficiency > 0 AND efficiency <= 1.0
    ),
    CONSTRAINT chk_p038_bc_charge_eff CHECK (
        charge_efficiency IS NULL OR (charge_efficiency > 0 AND charge_efficiency <= 1.0)
    ),
    CONSTRAINT chk_p038_bc_discharge_eff CHECK (
        discharge_efficiency IS NULL OR (discharge_efficiency > 0 AND discharge_efficiency <= 1.0)
    ),
    CONSTRAINT chk_p038_bc_inverter_eff CHECK (
        inverter_efficiency IS NULL OR (inverter_efficiency > 0 AND inverter_efficiency <= 1.0)
    ),
    CONSTRAINT chk_p038_bc_rte CHECK (
        round_trip_efficiency IS NULL OR (round_trip_efficiency > 0 AND round_trip_efficiency <= 1.0)
    ),
    CONSTRAINT chk_p038_bc_soh CHECK (
        current_soh_pct IS NULL OR (current_soh_pct >= 0 AND current_soh_pct <= 100)
    ),
    CONSTRAINT chk_p038_bc_c_rate CHECK (
        c_rate_max IS NULL OR c_rate_max > 0
    ),
    CONSTRAINT chk_p038_bc_cycle_count CHECK (
        current_cycle_count IS NULL OR current_cycle_count >= 0
    ),
    CONSTRAINT chk_p038_bc_interconnection CHECK (
        interconnection_type IS NULL OR interconnection_type IN (
            'AC_COUPLED', 'DC_COUPLED', 'HYBRID', 'MICROGRID', 'STANDALONE'
        )
    ),
    CONSTRAINT chk_p038_bc_thermal CHECK (
        thermal_management IS NULL OR thermal_management IN (
            'AIR_COOLED', 'LIQUID_COOLED', 'IMMERSION', 'HVAC_CONTROLLED', 'PASSIVE'
        )
    ),
    CONSTRAINT chk_p038_bc_status CHECK (
        bess_status IN (
            'PLANNED', 'ORDERED', 'INSTALLING', 'COMMISSIONING', 'OPERATIONAL',
            'DEGRADED', 'MAINTENANCE', 'DECOMMISSIONED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_bc_profile         ON pack038_peak_shaving.ps_bess_configurations(profile_id);
CREATE INDEX idx_p038_bc_tenant          ON pack038_peak_shaving.ps_bess_configurations(tenant_id);
CREATE INDEX idx_p038_bc_chemistry       ON pack038_peak_shaving.ps_bess_configurations(chemistry);
CREATE INDEX idx_p038_bc_capacity        ON pack038_peak_shaving.ps_bess_configurations(capacity_kwh DESC);
CREATE INDEX idx_p038_bc_power           ON pack038_peak_shaving.ps_bess_configurations(power_kw DESC);
CREATE INDEX idx_p038_bc_status          ON pack038_peak_shaving.ps_bess_configurations(bess_status);
CREATE INDEX idx_p038_bc_soh             ON pack038_peak_shaving.ps_bess_configurations(current_soh_pct);
CREATE INDEX idx_p038_bc_manufacturer    ON pack038_peak_shaving.ps_bess_configurations(manufacturer);
CREATE INDEX idx_p038_bc_created         ON pack038_peak_shaving.ps_bess_configurations(created_at DESC);
CREATE INDEX idx_p038_bc_metadata        ON pack038_peak_shaving.ps_bess_configurations USING GIN(metadata);

-- Operational BESS assets by profile for dispatch queries
CREATE INDEX idx_p038_bc_operational     ON pack038_peak_shaving.ps_bess_configurations(profile_id, power_kw DESC)
    WHERE bess_status = 'OPERATIONAL';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_bc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_bess_configurations
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_dispatch_simulations
-- =============================================================================
-- BESS dispatch simulation results for peak shaving scenarios. Each
-- simulation models a full billing period with interval-level charge/
-- discharge decisions and computes resulting peak reduction and savings.

CREATE TABLE pack038_peak_shaving.ps_dispatch_simulations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    bess_id                 UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_bess_configurations(id) ON DELETE CASCADE,
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id),
    tariff_id               UUID            REFERENCES pack038_peak_shaving.ps_tariff_structures(id),
    tenant_id               UUID            NOT NULL,
    simulation_name         VARCHAR(255)    NOT NULL,
    dispatch_strategy       VARCHAR(30)     NOT NULL DEFAULT 'THRESHOLD',
    target_peak_kw          NUMERIC(12,3),
    threshold_kw            NUMERIC(12,3),
    simulation_start        DATE            NOT NULL,
    simulation_end          DATE            NOT NULL,
    baseline_peak_kw        NUMERIC(12,3)   NOT NULL,
    achieved_peak_kw        NUMERIC(12,3)   NOT NULL,
    peak_reduction_kw       NUMERIC(12,3),
    peak_reduction_pct      NUMERIC(7,4),
    total_discharged_kwh    NUMERIC(15,3),
    total_charged_kwh       NUMERIC(15,3),
    round_trip_losses_kwh   NUMERIC(15,3),
    charging_cost           NUMERIC(12,2),
    demand_charge_baseline  NUMERIC(12,2),
    demand_charge_with_bess NUMERIC(12,2),
    demand_charge_savings   NUMERIC(12,2),
    energy_arbitrage_value  NUMERIC(12,2),
    net_savings             NUMERIC(12,2),
    annualized_savings      NUMERIC(12,2),
    equivalent_full_cycles  NUMERIC(8,2),
    avg_dod_pct             NUMERIC(5,2),
    max_soc_pct             NUMERIC(5,2),
    min_soc_pct             NUMERIC(5,2),
    discharge_events        INTEGER,
    successful_shaves       INTEGER,
    missed_peaks            INTEGER,
    interval_results        JSONB           DEFAULT '[]',
    run_timestamp           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    execution_time_ms       INTEGER,
    simulation_status       VARCHAR(20)     NOT NULL DEFAULT 'COMPLETED',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ds_strategy CHECK (
        dispatch_strategy IN (
            'THRESHOLD', 'PREDICTIVE', 'OPTIMAL_LP', 'RULE_BASED',
            'DYNAMIC_THRESHOLD', 'MACHINE_LEARNING', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p038_ds_baseline CHECK (
        baseline_peak_kw > 0
    ),
    CONSTRAINT chk_p038_ds_achieved CHECK (
        achieved_peak_kw >= 0
    ),
    CONSTRAINT chk_p038_ds_dates CHECK (
        simulation_start <= simulation_end
    ),
    CONSTRAINT chk_p038_ds_status CHECK (
        simulation_status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p038_ds_dod CHECK (
        avg_dod_pct IS NULL OR (avg_dod_pct >= 0 AND avg_dod_pct <= 100)
    ),
    CONSTRAINT chk_p038_ds_max_soc CHECK (
        max_soc_pct IS NULL OR (max_soc_pct >= 0 AND max_soc_pct <= 100)
    ),
    CONSTRAINT chk_p038_ds_min_soc CHECK (
        min_soc_pct IS NULL OR (min_soc_pct >= 0 AND min_soc_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ds_bess            ON pack038_peak_shaving.ps_dispatch_simulations(bess_id);
CREATE INDEX idx_p038_ds_profile         ON pack038_peak_shaving.ps_dispatch_simulations(profile_id);
CREATE INDEX idx_p038_ds_tariff          ON pack038_peak_shaving.ps_dispatch_simulations(tariff_id);
CREATE INDEX idx_p038_ds_tenant          ON pack038_peak_shaving.ps_dispatch_simulations(tenant_id);
CREATE INDEX idx_p038_ds_strategy        ON pack038_peak_shaving.ps_dispatch_simulations(dispatch_strategy);
CREATE INDEX idx_p038_ds_savings         ON pack038_peak_shaving.ps_dispatch_simulations(net_savings DESC);
CREATE INDEX idx_p038_ds_reduction       ON pack038_peak_shaving.ps_dispatch_simulations(peak_reduction_kw DESC);
CREATE INDEX idx_p038_ds_status          ON pack038_peak_shaving.ps_dispatch_simulations(simulation_status);
CREATE INDEX idx_p038_ds_run             ON pack038_peak_shaving.ps_dispatch_simulations(run_timestamp DESC);
CREATE INDEX idx_p038_ds_created         ON pack038_peak_shaving.ps_dispatch_simulations(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ds_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_dispatch_simulations
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_degradation_tracking
-- =============================================================================
-- BESS degradation tracking over time capturing state of health,
-- capacity fade, cycle counts, calendar aging, and warranty throughput
-- consumption for lifecycle cost analysis.

CREATE TABLE pack038_peak_shaving.ps_degradation_tracking (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    bess_id                 UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_bess_configurations(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    measurement_date        DATE            NOT NULL,
    soh_pct                 NUMERIC(5,2)    NOT NULL,
    capacity_remaining_kwh  NUMERIC(12,3)   NOT NULL,
    capacity_fade_pct       NUMERIC(5,2),
    power_fade_pct          NUMERIC(5,2),
    resistance_increase_pct NUMERIC(5,2),
    cumulative_cycles       INTEGER         NOT NULL DEFAULT 0,
    cumulative_throughput_kwh NUMERIC(15,3),
    cumulative_throughput_mwh NUMERIC(12,3),
    warranty_consumed_pct   NUMERIC(5,2),
    calendar_age_months     INTEGER,
    avg_temperature_c       NUMERIC(6,2),
    max_temperature_c       NUMERIC(6,2),
    avg_dod_pct             NUMERIC(5,2),
    avg_c_rate              NUMERIC(5,3),
    degradation_model       VARCHAR(30),
    predicted_eol_date      DATE,
    predicted_eol_soh_pct   NUMERIC(5,2),
    augmentation_needed     BOOLEAN         DEFAULT false,
    measurement_method      VARCHAR(30),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_dt_soh CHECK (
        soh_pct >= 0 AND soh_pct <= 100
    ),
    CONSTRAINT chk_p038_dt_capacity CHECK (
        capacity_remaining_kwh > 0
    ),
    CONSTRAINT chk_p038_dt_fade CHECK (
        capacity_fade_pct IS NULL OR (capacity_fade_pct >= 0 AND capacity_fade_pct <= 100)
    ),
    CONSTRAINT chk_p038_dt_power_fade CHECK (
        power_fade_pct IS NULL OR (power_fade_pct >= 0 AND power_fade_pct <= 100)
    ),
    CONSTRAINT chk_p038_dt_cycles CHECK (
        cumulative_cycles >= 0
    ),
    CONSTRAINT chk_p038_dt_warranty CHECK (
        warranty_consumed_pct IS NULL OR (warranty_consumed_pct >= 0 AND warranty_consumed_pct <= 200)
    ),
    CONSTRAINT chk_p038_dt_model CHECK (
        degradation_model IS NULL OR degradation_model IN (
            'SEMI_EMPIRICAL', 'RAINFLOW', 'ARRHENIUS', 'PHYSICS_BASED',
            'LINEAR', 'POLYNOMIAL', 'MANUFACTURER', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p038_dt_method CHECK (
        measurement_method IS NULL OR measurement_method IN (
            'BMS_REPORTED', 'CAPACITY_TEST', 'IMPEDANCE_SPECTROSCOPY',
            'COULOMB_COUNTING', 'OCV_METHOD', 'ESTIMATED'
        )
    ),
    CONSTRAINT uq_p038_dt_bess_date UNIQUE (bess_id, measurement_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_dt_bess            ON pack038_peak_shaving.ps_degradation_tracking(bess_id);
CREATE INDEX idx_p038_dt_tenant          ON pack038_peak_shaving.ps_degradation_tracking(tenant_id);
CREATE INDEX idx_p038_dt_date            ON pack038_peak_shaving.ps_degradation_tracking(measurement_date DESC);
CREATE INDEX idx_p038_dt_soh             ON pack038_peak_shaving.ps_degradation_tracking(soh_pct);
CREATE INDEX idx_p038_dt_cycles          ON pack038_peak_shaving.ps_degradation_tracking(cumulative_cycles DESC);
CREATE INDEX idx_p038_dt_warranty        ON pack038_peak_shaving.ps_degradation_tracking(warranty_consumed_pct DESC);
CREATE INDEX idx_p038_dt_created         ON pack038_peak_shaving.ps_degradation_tracking(created_at DESC);

-- Composite: BESS + date for degradation trend analysis
CREATE INDEX idx_p038_dt_bess_trend      ON pack038_peak_shaving.ps_degradation_tracking(bess_id, measurement_date DESC, soh_pct);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_dt_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_degradation_tracking
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_technology_comparisons
-- =============================================================================
-- Comparison of different BESS technologies (chemistries, sizes) for
-- a given peak shaving application. Evaluates LCOE, lifecycle cost,
-- performance metrics, and suitability scoring.

CREATE TABLE pack038_peak_shaving.ps_technology_comparisons (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    comparison_name         VARCHAR(255)    NOT NULL,
    technology_option       VARCHAR(100)    NOT NULL,
    chemistry               VARCHAR(30)     NOT NULL,
    capacity_kwh            NUMERIC(12,3)   NOT NULL,
    power_kw                NUMERIC(12,3)   NOT NULL,
    duration_hours          NUMERIC(6,2),
    round_trip_efficiency   NUMERIC(5,4),
    cycle_life              INTEGER,
    calendar_life_years     INTEGER,
    capital_cost            NUMERIC(12,2)   NOT NULL,
    installation_cost       NUMERIC(12,2),
    annual_om_cost          NUMERIC(10,2),
    augmentation_cost       NUMERIC(12,2),
    decommissioning_cost    NUMERIC(10,2),
    total_lifecycle_cost    NUMERIC(12,2),
    lcoe_per_kwh            NUMERIC(10,4),
    lcos_per_kwh            NUMERIC(10,4),
    annual_savings          NUMERIC(12,2),
    simple_payback_years    NUMERIC(6,2),
    npv                     NUMERIC(12,2),
    irr_pct                 NUMERIC(7,4),
    roi_pct                 NUMERIC(7,4),
    suitability_score       NUMERIC(5,2),
    safety_rating           VARCHAR(20),
    footprint_sqft          NUMERIC(10,2),
    weight_lbs              NUMERIC(10,2),
    lead_time_weeks         INTEGER,
    temperature_sensitivity VARCHAR(20),
    recyclability_pct       NUMERIC(5,2),
    recommendation_rank     INTEGER,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_tcomp_chemistry CHECK (
        chemistry IN (
            'LFP', 'NMC', 'NCA', 'LTO', 'SODIUM_ION', 'FLOW_VANADIUM',
            'FLOW_ZINC_BROMINE', 'FLOW_IRON', 'LEAD_ACID', 'SOLID_STATE',
            'ZINC_AIR', 'SUPERCAPACITOR', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_tcomp_capacity CHECK (
        capacity_kwh > 0
    ),
    CONSTRAINT chk_p038_tcomp_power CHECK (
        power_kw > 0
    ),
    CONSTRAINT chk_p038_tcomp_rte CHECK (
        round_trip_efficiency IS NULL OR (round_trip_efficiency > 0 AND round_trip_efficiency <= 1.0)
    ),
    CONSTRAINT chk_p038_tcomp_capital CHECK (
        capital_cost >= 0
    ),
    CONSTRAINT chk_p038_tcomp_suitability CHECK (
        suitability_score IS NULL OR (suitability_score >= 0 AND suitability_score <= 100)
    ),
    CONSTRAINT chk_p038_tcomp_safety CHECK (
        safety_rating IS NULL OR safety_rating IN ('EXCELLENT', 'GOOD', 'ADEQUATE', 'FAIR', 'POOR')
    ),
    CONSTRAINT chk_p038_tcomp_temp CHECK (
        temperature_sensitivity IS NULL OR temperature_sensitivity IN ('LOW', 'MEDIUM', 'HIGH')
    ),
    CONSTRAINT chk_p038_tcomp_rank CHECK (
        recommendation_rank IS NULL OR recommendation_rank >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_tcomp_profile      ON pack038_peak_shaving.ps_technology_comparisons(profile_id);
CREATE INDEX idx_p038_tcomp_tenant       ON pack038_peak_shaving.ps_technology_comparisons(tenant_id);
CREATE INDEX idx_p038_tcomp_chemistry    ON pack038_peak_shaving.ps_technology_comparisons(chemistry);
CREATE INDEX idx_p038_tcomp_npv          ON pack038_peak_shaving.ps_technology_comparisons(npv DESC);
CREATE INDEX idx_p038_tcomp_payback      ON pack038_peak_shaving.ps_technology_comparisons(simple_payback_years);
CREATE INDEX idx_p038_tcomp_score        ON pack038_peak_shaving.ps_technology_comparisons(suitability_score DESC);
CREATE INDEX idx_p038_tcomp_rank         ON pack038_peak_shaving.ps_technology_comparisons(recommendation_rank);
CREATE INDEX idx_p038_tcomp_created      ON pack038_peak_shaving.ps_technology_comparisons(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_tcomp_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_technology_comparisons
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_dispatch_schedules
-- =============================================================================
-- Operational dispatch schedules defining when and how the BESS should
-- charge and discharge for peak shaving. Includes daily schedules with
-- SOC targets, threshold triggers, and override conditions.

CREATE TABLE pack038_peak_shaving.ps_dispatch_schedules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    bess_id                 UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_bess_configurations(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    schedule_name           VARCHAR(255)    NOT NULL,
    schedule_type           VARCHAR(30)     NOT NULL DEFAULT 'DAILY',
    effective_date          DATE            NOT NULL,
    expiration_date         DATE,
    day_type                VARCHAR(20)     NOT NULL DEFAULT 'WEEKDAY',
    season                  VARCHAR(20)     NOT NULL DEFAULT 'ALL_YEAR',
    charge_start_hour       INTEGER,
    charge_end_hour         INTEGER,
    charge_rate_kw          NUMERIC(12,3),
    target_soc_pct          NUMERIC(5,2),
    discharge_threshold_kw  NUMERIC(12,3),
    discharge_max_kw        NUMERIC(12,3),
    discharge_soc_floor_pct NUMERIC(5,2),
    peak_window_start_hour  INTEGER,
    peak_window_end_hour    INTEGER,
    emergency_reserve_pct   NUMERIC(5,2)    DEFAULT 10.00,
    priority                INTEGER         DEFAULT 100,
    auto_dispatch           BOOLEAN         DEFAULT true,
    override_conditions     JSONB           DEFAULT '{}',
    schedule_status         VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_dsc_type CHECK (
        schedule_type IN ('DAILY', 'WEEKLY', 'SEASONAL', 'EVENT_BASED', 'CUSTOM')
    ),
    CONSTRAINT chk_p038_dsc_day_type CHECK (
        day_type IN ('WEEKDAY', 'WEEKEND', 'HOLIDAY', 'ALL_DAYS', 'MONDAY', 'TUESDAY',
                     'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY')
    ),
    CONSTRAINT chk_p038_dsc_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'SPRING', 'FALL', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p038_dsc_charge_hours CHECK (
        charge_start_hour IS NULL OR (charge_start_hour >= 0 AND charge_start_hour <= 23)
    ),
    CONSTRAINT chk_p038_dsc_charge_end CHECK (
        charge_end_hour IS NULL OR (charge_end_hour >= 0 AND charge_end_hour <= 23)
    ),
    CONSTRAINT chk_p038_dsc_target_soc CHECK (
        target_soc_pct IS NULL OR (target_soc_pct >= 0 AND target_soc_pct <= 100)
    ),
    CONSTRAINT chk_p038_dsc_floor CHECK (
        discharge_soc_floor_pct IS NULL OR (discharge_soc_floor_pct >= 0 AND discharge_soc_floor_pct <= 100)
    ),
    CONSTRAINT chk_p038_dsc_reserve CHECK (
        emergency_reserve_pct IS NULL OR (emergency_reserve_pct >= 0 AND emergency_reserve_pct <= 100)
    ),
    CONSTRAINT chk_p038_dsc_peak_start CHECK (
        peak_window_start_hour IS NULL OR (peak_window_start_hour >= 0 AND peak_window_start_hour <= 23)
    ),
    CONSTRAINT chk_p038_dsc_peak_end CHECK (
        peak_window_end_hour IS NULL OR (peak_window_end_hour >= 0 AND peak_window_end_hour <= 23)
    ),
    CONSTRAINT chk_p038_dsc_status CHECK (
        schedule_status IN ('DRAFT', 'ACTIVE', 'PAUSED', 'EXPIRED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p038_dsc_priority CHECK (
        priority >= 1 AND priority <= 999
    ),
    CONSTRAINT chk_p038_dsc_dates CHECK (
        expiration_date IS NULL OR effective_date <= expiration_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_dsc_bess           ON pack038_peak_shaving.ps_dispatch_schedules(bess_id);
CREATE INDEX idx_p038_dsc_tenant         ON pack038_peak_shaving.ps_dispatch_schedules(tenant_id);
CREATE INDEX idx_p038_dsc_type           ON pack038_peak_shaving.ps_dispatch_schedules(schedule_type);
CREATE INDEX idx_p038_dsc_day_type       ON pack038_peak_shaving.ps_dispatch_schedules(day_type);
CREATE INDEX idx_p038_dsc_season         ON pack038_peak_shaving.ps_dispatch_schedules(season);
CREATE INDEX idx_p038_dsc_status         ON pack038_peak_shaving.ps_dispatch_schedules(schedule_status);
CREATE INDEX idx_p038_dsc_effective      ON pack038_peak_shaving.ps_dispatch_schedules(effective_date DESC);
CREATE INDEX idx_p038_dsc_priority       ON pack038_peak_shaving.ps_dispatch_schedules(priority);
CREATE INDEX idx_p038_dsc_created        ON pack038_peak_shaving.ps_dispatch_schedules(created_at DESC);

-- Active schedules for dispatch engine queries
CREATE INDEX idx_p038_dsc_active         ON pack038_peak_shaving.ps_dispatch_schedules(bess_id, day_type, season, priority)
    WHERE schedule_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_dsc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_dispatch_schedules
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_bess_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_dispatch_simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_degradation_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_technology_comparisons ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_dispatch_schedules ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_bc_tenant_isolation
    ON pack038_peak_shaving.ps_bess_configurations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_bc_service_bypass
    ON pack038_peak_shaving.ps_bess_configurations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ds_tenant_isolation
    ON pack038_peak_shaving.ps_dispatch_simulations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ds_service_bypass
    ON pack038_peak_shaving.ps_dispatch_simulations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_dt_tenant_isolation
    ON pack038_peak_shaving.ps_degradation_tracking
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_dt_service_bypass
    ON pack038_peak_shaving.ps_degradation_tracking
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_tcomp_tenant_isolation
    ON pack038_peak_shaving.ps_technology_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_tcomp_service_bypass
    ON pack038_peak_shaving.ps_technology_comparisons
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_dsc_tenant_isolation
    ON pack038_peak_shaving.ps_dispatch_schedules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_dsc_service_bypass
    ON pack038_peak_shaving.ps_dispatch_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_bess_configurations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_dispatch_simulations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_degradation_tracking TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_technology_comparisons TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_dispatch_schedules TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_bess_configurations IS
    'Battery Energy Storage System asset definitions with chemistry, capacity, power rating, SOC limits, degradation parameters, and operational constraints.';
COMMENT ON TABLE pack038_peak_shaving.ps_dispatch_simulations IS
    'BESS dispatch simulation results for peak shaving scenarios with interval-level charge/discharge decisions and savings calculations.';
COMMENT ON TABLE pack038_peak_shaving.ps_degradation_tracking IS
    'BESS degradation tracking over time with SOH, capacity fade, cycle counts, and warranty throughput for lifecycle cost analysis.';
COMMENT ON TABLE pack038_peak_shaving.ps_technology_comparisons IS
    'Multi-technology BESS comparison evaluating LCOE, lifecycle cost, performance, and suitability scoring for peak shaving applications.';
COMMENT ON TABLE pack038_peak_shaving.ps_dispatch_schedules IS
    'Operational dispatch schedules defining BESS charge/discharge windows, SOC targets, and threshold triggers for peak shaving.';

COMMENT ON COLUMN pack038_peak_shaving.ps_bess_configurations.chemistry IS 'Battery chemistry: LFP, NMC, NCA, LTO, SODIUM_ION, FLOW_VANADIUM, etc.';
COMMENT ON COLUMN pack038_peak_shaving.ps_bess_configurations.soc_min IS 'Minimum state of charge (%) to preserve battery health. Typically 10-20%.';
COMMENT ON COLUMN pack038_peak_shaving.ps_bess_configurations.soc_max IS 'Maximum state of charge (%) to preserve battery health. Typically 80-90%.';
COMMENT ON COLUMN pack038_peak_shaving.ps_bess_configurations.efficiency IS 'One-way DC efficiency of the battery (0-1). Combined round-trip = charge_eff * discharge_eff.';
COMMENT ON COLUMN pack038_peak_shaving.ps_bess_configurations.current_soh_pct IS 'Current state of health as percentage of original capacity (0-100%).';

COMMENT ON COLUMN pack038_peak_shaving.ps_dispatch_simulations.dispatch_strategy IS 'Dispatch strategy: THRESHOLD (fixed limit), PREDICTIVE (forecast-based), OPTIMAL_LP (linear program), etc.';
COMMENT ON COLUMN pack038_peak_shaving.ps_dispatch_simulations.equivalent_full_cycles IS 'Number of equivalent full charge-discharge cycles consumed during the simulation period.';

COMMENT ON COLUMN pack038_peak_shaving.ps_degradation_tracking.soh_pct IS 'State of health as percentage of original nameplate capacity (0-100%).';
COMMENT ON COLUMN pack038_peak_shaving.ps_degradation_tracking.warranty_consumed_pct IS 'Percentage of warranty throughput consumed. Values above 100% indicate warranty exceedance.';

COMMENT ON COLUMN pack038_peak_shaving.ps_technology_comparisons.lcoe_per_kwh IS 'Levelized cost of energy stored ($/kWh lifecycle cost / total kWh throughput).';
COMMENT ON COLUMN pack038_peak_shaving.ps_technology_comparisons.lcos_per_kwh IS 'Levelized cost of storage ($/kWh) accounting for all costs over useful life.';
