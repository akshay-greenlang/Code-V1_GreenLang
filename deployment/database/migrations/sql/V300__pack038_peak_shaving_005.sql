-- =============================================================================
-- V300: PACK-038 Peak Shaving Pack - Load Shifting
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Load shifting tables for identifying, scheduling, and tracking movable
-- loads that can be shifted from peak to off-peak periods. Covers
-- shiftable load inventory, scheduling constraints, optimized shift
-- schedules, multi-load coordination, and energy rebound tracking.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_shiftable_loads
--   2. pack038_peak_shaving.ps_shift_constraints
--   3. pack038_peak_shaving.ps_shift_schedules
--   4. pack038_peak_shaving.ps_coordination_plans
--   5. pack038_peak_shaving.ps_rebound_tracking
--
-- Previous: V299__pack038_peak_shaving_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_shiftable_loads
-- =============================================================================
-- Inventory of loads that can be temporally shifted from peak to off-peak
-- periods. Each entry describes the load characteristics, flexibility
-- window, and estimated peak reduction contribution.

CREATE TABLE pack038_peak_shaving.ps_shiftable_loads (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    load_name               VARCHAR(255)    NOT NULL,
    load_category           VARCHAR(50)     NOT NULL,
    equipment_type          VARCHAR(100),
    equipment_id            VARCHAR(100),
    rated_capacity_kw       NUMERIC(12,3)   NOT NULL,
    typical_demand_kw       NUMERIC(12,3),
    shiftable_kw            NUMERIC(12,3)   NOT NULL,
    typical_duration_hours  NUMERIC(6,2)    NOT NULL,
    energy_per_cycle_kwh    NUMERIC(12,3),
    cycles_per_day          NUMERIC(6,2)    DEFAULT 1.0,
    current_start_hour      INTEGER,
    current_end_hour        INTEGER,
    earliest_start_hour     INTEGER,
    latest_end_hour         INTEGER,
    preferred_off_peak_start INTEGER,
    preferred_off_peak_end  INTEGER,
    min_continuous_runtime_min INTEGER,
    max_interruptions       INTEGER         DEFAULT 0,
    ramp_up_time_min        INTEGER,
    ramp_down_time_min      INTEGER,
    shift_type              VARCHAR(30)     NOT NULL DEFAULT 'TEMPORAL',
    automation_level        VARCHAR(20)     NOT NULL DEFAULT 'MANUAL',
    control_system          VARCHAR(100),
    comfort_impact          VARCHAR(20)     DEFAULT 'NONE',
    production_impact       VARCHAR(20)     DEFAULT 'NONE',
    safety_critical         BOOLEAN         DEFAULT false,
    weather_dependent       BOOLEAN         DEFAULT false,
    occupancy_dependent     BOOLEAN         DEFAULT false,
    estimated_annual_savings NUMERIC(12,2),
    shift_status            VARCHAR(20)     NOT NULL DEFAULT 'IDENTIFIED',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_sl_category CHECK (
        load_category IN (
            'HVAC_PRECOOLING', 'HVAC_PREHEATING', 'WATER_HEATING', 'ICE_STORAGE',
            'THERMAL_STORAGE', 'EV_CHARGING', 'POOL_PUMPING', 'IRRIGATION',
            'LAUNDRY', 'DISHWASHING', 'COMPRESSED_AIR', 'BATCH_PROCESS',
            'DATA_PROCESSING', 'BACKUP_SYSTEMS', 'SNOW_MELTING',
            'REFRIGERATION_PULLDOWN', 'LIGHTING_PRECONDITIONING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_sl_rated CHECK (
        rated_capacity_kw > 0
    ),
    CONSTRAINT chk_p038_sl_shiftable CHECK (
        shiftable_kw > 0
    ),
    CONSTRAINT chk_p038_sl_duration CHECK (
        typical_duration_hours > 0
    ),
    CONSTRAINT chk_p038_sl_shift_type CHECK (
        shift_type IN ('TEMPORAL', 'MODULATION', 'INTERRUPTIBLE', 'PRECONDITIONING')
    ),
    CONSTRAINT chk_p038_sl_automation CHECK (
        automation_level IN ('MANUAL', 'SEMI_AUTO', 'FULLY_AUTO', 'SCHEDULED')
    ),
    CONSTRAINT chk_p038_sl_comfort CHECK (
        comfort_impact IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p038_sl_production CHECK (
        production_impact IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p038_sl_status CHECK (
        shift_status IN ('IDENTIFIED', 'EVALUATED', 'APPROVED', 'ACTIVE', 'PAUSED', 'RETIRED')
    ),
    CONSTRAINT chk_p038_sl_current_start CHECK (
        current_start_hour IS NULL OR (current_start_hour >= 0 AND current_start_hour <= 23)
    ),
    CONSTRAINT chk_p038_sl_current_end CHECK (
        current_end_hour IS NULL OR (current_end_hour >= 0 AND current_end_hour <= 23)
    ),
    CONSTRAINT chk_p038_sl_earliest CHECK (
        earliest_start_hour IS NULL OR (earliest_start_hour >= 0 AND earliest_start_hour <= 23)
    ),
    CONSTRAINT chk_p038_sl_latest CHECK (
        latest_end_hour IS NULL OR (latest_end_hour >= 0 AND latest_end_hour <= 23)
    ),
    CONSTRAINT chk_p038_sl_cycles CHECK (
        cycles_per_day IS NULL OR cycles_per_day > 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_sl_profile         ON pack038_peak_shaving.ps_shiftable_loads(profile_id);
CREATE INDEX idx_p038_sl_tenant          ON pack038_peak_shaving.ps_shiftable_loads(tenant_id);
CREATE INDEX idx_p038_sl_category        ON pack038_peak_shaving.ps_shiftable_loads(load_category);
CREATE INDEX idx_p038_sl_shiftable_kw    ON pack038_peak_shaving.ps_shiftable_loads(shiftable_kw DESC);
CREATE INDEX idx_p038_sl_shift_type      ON pack038_peak_shaving.ps_shiftable_loads(shift_type);
CREATE INDEX idx_p038_sl_status          ON pack038_peak_shaving.ps_shiftable_loads(shift_status);
CREATE INDEX idx_p038_sl_automation      ON pack038_peak_shaving.ps_shiftable_loads(automation_level);
CREATE INDEX idx_p038_sl_savings         ON pack038_peak_shaving.ps_shiftable_loads(estimated_annual_savings DESC);
CREATE INDEX idx_p038_sl_created         ON pack038_peak_shaving.ps_shiftable_loads(created_at DESC);

-- Active shiftable loads sorted by capacity for dispatch
CREATE INDEX idx_p038_sl_active_cap      ON pack038_peak_shaving.ps_shiftable_loads(profile_id, shiftable_kw DESC)
    WHERE shift_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_sl_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_shiftable_loads
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_shift_constraints
-- =============================================================================
-- Operational constraints that govern when and how loads can be shifted.
-- Includes temporal windows, temperature limits, occupancy requirements,
-- production deadlines, and interdependency rules.

CREATE TABLE pack038_peak_shaving.ps_shift_constraints (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    shiftable_load_id       UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_shiftable_loads(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    constraint_name         VARCHAR(255)    NOT NULL,
    constraint_type         VARCHAR(30)     NOT NULL,
    constraint_priority     VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    day_type                VARCHAR(20)     NOT NULL DEFAULT 'ALL_DAYS',
    season                  VARCHAR(20)     NOT NULL DEFAULT 'ALL_YEAR',
    blackout_start_hour     INTEGER,
    blackout_end_hour       INTEGER,
    temperature_min_f       NUMERIC(6,2),
    temperature_max_f       NUMERIC(6,2),
    occupancy_required      BOOLEAN         DEFAULT false,
    min_occupancy_pct       NUMERIC(5,2),
    production_deadline     TIME,
    depends_on_load_id      UUID,
    dependency_type         VARCHAR(20),
    max_shift_hours         NUMERIC(6,2),
    min_rest_between_shifts_hours NUMERIC(6,2),
    max_shifts_per_day      INTEGER,
    penalty_cost_per_violation NUMERIC(10,2),
    constraint_active       BOOLEAN         NOT NULL DEFAULT true,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_sc_type CHECK (
        constraint_type IN (
            'TEMPORAL', 'TEMPERATURE', 'OCCUPANCY', 'PRODUCTION',
            'DEPENDENCY', 'SAFETY', 'COMFORT', 'EQUIPMENT', 'REGULATORY',
            'CONTRACTUAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_sc_priority CHECK (
        constraint_priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'ADVISORY')
    ),
    CONSTRAINT chk_p038_sc_day_type CHECK (
        day_type IN ('WEEKDAY', 'WEEKEND', 'HOLIDAY', 'ALL_DAYS')
    ),
    CONSTRAINT chk_p038_sc_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'SPRING', 'FALL', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p038_sc_blackout_start CHECK (
        blackout_start_hour IS NULL OR (blackout_start_hour >= 0 AND blackout_start_hour <= 23)
    ),
    CONSTRAINT chk_p038_sc_blackout_end CHECK (
        blackout_end_hour IS NULL OR (blackout_end_hour >= 0 AND blackout_end_hour <= 23)
    ),
    CONSTRAINT chk_p038_sc_occupancy CHECK (
        min_occupancy_pct IS NULL OR (min_occupancy_pct >= 0 AND min_occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p038_sc_dependency CHECK (
        dependency_type IS NULL OR dependency_type IN (
            'MUST_PRECEDE', 'MUST_FOLLOW', 'MUST_NOT_OVERLAP', 'MUST_COEXIST'
        )
    ),
    CONSTRAINT chk_p038_sc_max_shifts CHECK (
        max_shifts_per_day IS NULL OR max_shifts_per_day >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_sc_load            ON pack038_peak_shaving.ps_shift_constraints(shiftable_load_id);
CREATE INDEX idx_p038_sc_tenant          ON pack038_peak_shaving.ps_shift_constraints(tenant_id);
CREATE INDEX idx_p038_sc_type            ON pack038_peak_shaving.ps_shift_constraints(constraint_type);
CREATE INDEX idx_p038_sc_priority        ON pack038_peak_shaving.ps_shift_constraints(constraint_priority);
CREATE INDEX idx_p038_sc_day_type        ON pack038_peak_shaving.ps_shift_constraints(day_type);
CREATE INDEX idx_p038_sc_season          ON pack038_peak_shaving.ps_shift_constraints(season);
CREATE INDEX idx_p038_sc_active          ON pack038_peak_shaving.ps_shift_constraints(constraint_active);
CREATE INDEX idx_p038_sc_dependency      ON pack038_peak_shaving.ps_shift_constraints(depends_on_load_id);
CREATE INDEX idx_p038_sc_created         ON pack038_peak_shaving.ps_shift_constraints(created_at DESC);

-- Active constraints by load for scheduling engine
CREATE INDEX idx_p038_sc_load_active     ON pack038_peak_shaving.ps_shift_constraints(shiftable_load_id, constraint_type)
    WHERE constraint_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_sc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_shift_constraints
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_shift_schedules
-- =============================================================================
-- Optimized load shift schedules that define the new operating times for
-- each shiftable load. Includes both planned (recurring) and event-driven
-- (ad hoc) shift schedules with expected peak reduction impact.

CREATE TABLE pack038_peak_shaving.ps_shift_schedules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    shiftable_load_id       UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_shiftable_loads(id) ON DELETE CASCADE,
    coordination_plan_id    UUID,
    tenant_id               UUID            NOT NULL,
    schedule_date           DATE            NOT NULL,
    original_start_hour     INTEGER         NOT NULL,
    original_end_hour       INTEGER         NOT NULL,
    shifted_start_hour      INTEGER         NOT NULL,
    shifted_end_hour        INTEGER         NOT NULL,
    shift_direction         VARCHAR(20)     NOT NULL,
    shift_duration_hours    NUMERIC(6,2)    NOT NULL,
    original_demand_kw      NUMERIC(12,3),
    shifted_demand_kw       NUMERIC(12,3),
    peak_reduction_kw       NUMERIC(12,3),
    energy_shifted_kwh      NUMERIC(12,3),
    demand_charge_savings   NUMERIC(10,2),
    constraint_violations   INTEGER         DEFAULT 0,
    comfort_score           NUMERIC(5,2),
    execution_status        VARCHAR(20)     NOT NULL DEFAULT 'SCHEDULED',
    actual_start             TIMESTAMPTZ,
    actual_end              TIMESTAMPTZ,
    actual_demand_kw        NUMERIC(12,3),
    variance_kw             NUMERIC(12,3),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ss_orig_start CHECK (
        original_start_hour >= 0 AND original_start_hour <= 23
    ),
    CONSTRAINT chk_p038_ss_orig_end CHECK (
        original_end_hour >= 0 AND original_end_hour <= 23
    ),
    CONSTRAINT chk_p038_ss_shift_start CHECK (
        shifted_start_hour >= 0 AND shifted_start_hour <= 23
    ),
    CONSTRAINT chk_p038_ss_shift_end CHECK (
        shifted_end_hour >= 0 AND shifted_end_hour <= 23
    ),
    CONSTRAINT chk_p038_ss_direction CHECK (
        shift_direction IN ('EARLIER', 'LATER', 'SPLIT', 'COMPRESSED', 'ELIMINATED')
    ),
    CONSTRAINT chk_p038_ss_duration CHECK (
        shift_duration_hours > 0
    ),
    CONSTRAINT chk_p038_ss_execution CHECK (
        execution_status IN (
            'SCHEDULED', 'IN_PROGRESS', 'COMPLETED', 'PARTIALLY_COMPLETED',
            'CANCELLED', 'FAILED', 'OVERRIDDEN'
        )
    ),
    CONSTRAINT chk_p038_ss_comfort CHECK (
        comfort_score IS NULL OR (comfort_score >= 0 AND comfort_score <= 100)
    ),
    CONSTRAINT chk_p038_ss_violations CHECK (
        constraint_violations >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ss_load            ON pack038_peak_shaving.ps_shift_schedules(shiftable_load_id);
CREATE INDEX idx_p038_ss_coordination    ON pack038_peak_shaving.ps_shift_schedules(coordination_plan_id);
CREATE INDEX idx_p038_ss_tenant          ON pack038_peak_shaving.ps_shift_schedules(tenant_id);
CREATE INDEX idx_p038_ss_date            ON pack038_peak_shaving.ps_shift_schedules(schedule_date DESC);
CREATE INDEX idx_p038_ss_direction       ON pack038_peak_shaving.ps_shift_schedules(shift_direction);
CREATE INDEX idx_p038_ss_status          ON pack038_peak_shaving.ps_shift_schedules(execution_status);
CREATE INDEX idx_p038_ss_reduction       ON pack038_peak_shaving.ps_shift_schedules(peak_reduction_kw DESC);
CREATE INDEX idx_p038_ss_savings         ON pack038_peak_shaving.ps_shift_schedules(demand_charge_savings DESC);
CREATE INDEX idx_p038_ss_created         ON pack038_peak_shaving.ps_shift_schedules(created_at DESC);

-- Upcoming scheduled shifts for execution
CREATE INDEX idx_p038_ss_upcoming        ON pack038_peak_shaving.ps_shift_schedules(schedule_date, shifted_start_hour)
    WHERE execution_status = 'SCHEDULED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ss_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_shift_schedules
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_coordination_plans
-- =============================================================================
-- Multi-load coordination plans that orchestrate the shifting of
-- multiple loads simultaneously to maximize peak reduction while
-- respecting inter-load dependencies and aggregate constraints.

CREATE TABLE pack038_peak_shaving.ps_coordination_plans (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    plan_name               VARCHAR(255)    NOT NULL,
    plan_type               VARCHAR(30)     NOT NULL DEFAULT 'DAILY',
    effective_date          DATE            NOT NULL,
    expiration_date         DATE,
    target_peak_reduction_kw NUMERIC(12,3)  NOT NULL,
    achieved_reduction_kw   NUMERIC(12,3),
    loads_included          INTEGER         NOT NULL DEFAULT 0,
    total_shiftable_kw      NUMERIC(12,3),
    aggregate_constraint_kw NUMERIC(12,3),
    optimization_method     VARCHAR(30),
    optimization_objective  VARCHAR(30)     NOT NULL DEFAULT 'MAX_REDUCTION',
    estimated_annual_savings NUMERIC(12,2),
    actual_annual_savings   NUMERIC(12,2),
    success_rate_pct        NUMERIC(5,2),
    plan_status             VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by             VARCHAR(255),
    approved_at             TIMESTAMPTZ,
    schedule_template       JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cp_type CHECK (
        plan_type IN ('DAILY', 'WEEKLY', 'SEASONAL', 'EVENT_DRIVEN', 'CONTINUOUS')
    ),
    CONSTRAINT chk_p038_cp_target CHECK (
        target_peak_reduction_kw > 0
    ),
    CONSTRAINT chk_p038_cp_loads CHECK (
        loads_included >= 0
    ),
    CONSTRAINT chk_p038_cp_method CHECK (
        optimization_method IS NULL OR optimization_method IN (
            'LINEAR_PROGRAM', 'MIXED_INTEGER', 'HEURISTIC', 'GENETIC_ALGORITHM',
            'SIMULATED_ANNEALING', 'RULE_BASED', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p038_cp_objective CHECK (
        optimization_objective IN (
            'MAX_REDUCTION', 'MIN_COST', 'MIN_DISRUPTION', 'BALANCED'
        )
    ),
    CONSTRAINT chk_p038_cp_success CHECK (
        success_rate_pct IS NULL OR (success_rate_pct >= 0 AND success_rate_pct <= 100)
    ),
    CONSTRAINT chk_p038_cp_status CHECK (
        plan_status IN ('DRAFT', 'APPROVED', 'ACTIVE', 'PAUSED', 'COMPLETED', 'RETIRED')
    ),
    CONSTRAINT chk_p038_cp_dates CHECK (
        expiration_date IS NULL OR effective_date <= expiration_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cp_profile         ON pack038_peak_shaving.ps_coordination_plans(profile_id);
CREATE INDEX idx_p038_cp_tenant          ON pack038_peak_shaving.ps_coordination_plans(tenant_id);
CREATE INDEX idx_p038_cp_type            ON pack038_peak_shaving.ps_coordination_plans(plan_type);
CREATE INDEX idx_p038_cp_status          ON pack038_peak_shaving.ps_coordination_plans(plan_status);
CREATE INDEX idx_p038_cp_objective       ON pack038_peak_shaving.ps_coordination_plans(optimization_objective);
CREATE INDEX idx_p038_cp_target          ON pack038_peak_shaving.ps_coordination_plans(target_peak_reduction_kw DESC);
CREATE INDEX idx_p038_cp_savings         ON pack038_peak_shaving.ps_coordination_plans(estimated_annual_savings DESC);
CREATE INDEX idx_p038_cp_effective       ON pack038_peak_shaving.ps_coordination_plans(effective_date DESC);
CREATE INDEX idx_p038_cp_created         ON pack038_peak_shaving.ps_coordination_plans(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_cp_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_coordination_plans
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_rebound_tracking
-- =============================================================================
-- Tracks energy rebound effects after load shifting events. Rebound
-- occurs when shifted loads consume additional energy during the
-- recovery period, potentially creating new secondary peaks.

CREATE TABLE pack038_peak_shaving.ps_rebound_tracking (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    shift_schedule_id       UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_shift_schedules(id) ON DELETE CASCADE,
    shiftable_load_id       UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_shiftable_loads(id),
    tenant_id               UUID            NOT NULL,
    shift_date              DATE            NOT NULL,
    rebound_start           TIMESTAMPTZ     NOT NULL,
    rebound_end             TIMESTAMPTZ,
    rebound_duration_min    INTEGER,
    rebound_peak_kw         NUMERIC(12,3),
    normal_demand_kw        NUMERIC(12,3),
    rebound_energy_kwh      NUMERIC(12,3),
    rebound_factor          NUMERIC(5,3),
    net_energy_impact_kwh   NUMERIC(12,3),
    created_secondary_peak  BOOLEAN         DEFAULT false,
    secondary_peak_kw       NUMERIC(12,3),
    secondary_peak_cost     NUMERIC(10,2),
    net_savings_after_rebound NUMERIC(10,2),
    mitigation_applied      VARCHAR(50),
    mitigation_effective    BOOLEAN,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_rt_duration CHECK (
        rebound_duration_min IS NULL OR rebound_duration_min >= 0
    ),
    CONSTRAINT chk_p038_rt_peak CHECK (
        rebound_peak_kw IS NULL OR rebound_peak_kw >= 0
    ),
    CONSTRAINT chk_p038_rt_factor CHECK (
        rebound_factor IS NULL OR (rebound_factor >= 0 AND rebound_factor <= 5.0)
    ),
    CONSTRAINT chk_p038_rt_mitigation CHECK (
        mitigation_applied IS NULL OR mitigation_applied IN (
            'STAGED_RECOVERY', 'DEMAND_LIMITING', 'BESS_SUPPORT',
            'LOAD_ROTATION', 'EXTENDED_SHIFT', 'NONE'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_rt_schedule        ON pack038_peak_shaving.ps_rebound_tracking(shift_schedule_id);
CREATE INDEX idx_p038_rt_load            ON pack038_peak_shaving.ps_rebound_tracking(shiftable_load_id);
CREATE INDEX idx_p038_rt_tenant          ON pack038_peak_shaving.ps_rebound_tracking(tenant_id);
CREATE INDEX idx_p038_rt_date            ON pack038_peak_shaving.ps_rebound_tracking(shift_date DESC);
CREATE INDEX idx_p038_rt_factor          ON pack038_peak_shaving.ps_rebound_tracking(rebound_factor DESC);
CREATE INDEX idx_p038_rt_secondary       ON pack038_peak_shaving.ps_rebound_tracking(created_secondary_peak) WHERE created_secondary_peak = true;
CREATE INDEX idx_p038_rt_created         ON pack038_peak_shaving.ps_rebound_tracking(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_rt_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_rebound_tracking
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_shiftable_loads ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_shift_constraints ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_shift_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_coordination_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_rebound_tracking ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_sl_tenant_isolation
    ON pack038_peak_shaving.ps_shiftable_loads
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_sl_service_bypass
    ON pack038_peak_shaving.ps_shiftable_loads
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_sc_tenant_isolation
    ON pack038_peak_shaving.ps_shift_constraints
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_sc_service_bypass
    ON pack038_peak_shaving.ps_shift_constraints
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ss_tenant_isolation
    ON pack038_peak_shaving.ps_shift_schedules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ss_service_bypass
    ON pack038_peak_shaving.ps_shift_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cp_tenant_isolation
    ON pack038_peak_shaving.ps_coordination_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cp_service_bypass
    ON pack038_peak_shaving.ps_coordination_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_rt_tenant_isolation
    ON pack038_peak_shaving.ps_rebound_tracking
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_rt_service_bypass
    ON pack038_peak_shaving.ps_rebound_tracking
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_shiftable_loads TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_shift_constraints TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_shift_schedules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_coordination_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_rebound_tracking TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_shiftable_loads IS
    'Inventory of loads that can be temporally shifted from peak to off-peak periods with flexibility windows and peak reduction estimates.';
COMMENT ON TABLE pack038_peak_shaving.ps_shift_constraints IS
    'Operational constraints governing load shifting including temporal windows, temperature limits, occupancy requirements, and interdependencies.';
COMMENT ON TABLE pack038_peak_shaving.ps_shift_schedules IS
    'Optimized load shift schedules defining new operating times with expected and actual peak reduction impact.';
COMMENT ON TABLE pack038_peak_shaving.ps_coordination_plans IS
    'Multi-load coordination plans orchestrating simultaneous shifting of multiple loads to maximize aggregate peak reduction.';
COMMENT ON TABLE pack038_peak_shaving.ps_rebound_tracking IS
    'Energy rebound tracking after load shifting events to detect secondary peaks and measure net savings impact.';

COMMENT ON COLUMN pack038_peak_shaving.ps_shiftable_loads.shiftable_kw IS 'Maximum kW that can be shifted from peak to off-peak period.';
COMMENT ON COLUMN pack038_peak_shaving.ps_shiftable_loads.shift_type IS 'Type of shift: TEMPORAL (move in time), MODULATION (reduce during peak), INTERRUPTIBLE (stop during peak), PRECONDITIONING (pre-do work).';
COMMENT ON COLUMN pack038_peak_shaving.ps_shift_constraints.dependency_type IS 'Load dependency: MUST_PRECEDE, MUST_FOLLOW, MUST_NOT_OVERLAP, MUST_COEXIST.';
COMMENT ON COLUMN pack038_peak_shaving.ps_shift_schedules.shift_direction IS 'Direction of temporal shift: EARLIER, LATER, SPLIT, COMPRESSED, ELIMINATED.';
COMMENT ON COLUMN pack038_peak_shaving.ps_rebound_tracking.rebound_factor IS 'Energy rebound multiplier (1.0 = equal energy, 1.3 = 30% more energy in recovery period).';
COMMENT ON COLUMN pack038_peak_shaving.ps_rebound_tracking.created_secondary_peak IS 'Whether the rebound created a new secondary demand peak.';
