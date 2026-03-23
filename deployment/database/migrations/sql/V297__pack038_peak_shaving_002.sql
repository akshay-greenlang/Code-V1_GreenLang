-- =============================================================================
-- V297: PACK-038 Peak Shaving Pack - Peak Events & Attribution
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Peak event detection, attribution analysis, clustering, simulation, and
-- avoidability assessment tables. These tables identify and characterize
-- demand peaks, attribute them to specific loads or conditions, and
-- assess whether each peak was avoidable through operational changes.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_peak_events
--   2. pack038_peak_shaving.ps_peak_attribution
--   3. pack038_peak_shaving.ps_peak_clusters
--   4. pack038_peak_shaving.ps_peak_simulations
--   5. pack038_peak_shaving.ps_avoidability_assessments
--
-- Previous: V296__pack038_peak_shaving_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_peak_events
-- =============================================================================
-- Individual detected peak demand events for a load profile. Each peak
-- event captures the magnitude, timing, weather conditions, billing
-- context, and whether the peak set the billing demand for that period.

CREATE TABLE pack038_peak_shaving.ps_peak_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    peak_timestamp          TIMESTAMPTZ     NOT NULL,
    peak_kw                 NUMERIC(12,3)   NOT NULL,
    peak_kva                NUMERIC(12,3),
    peak_kvar               NUMERIC(12,3),
    power_factor_at_peak    NUMERIC(5,4),
    duration_above_threshold_min INTEGER,
    threshold_kw            NUMERIC(12,3),
    peak_type               VARCHAR(30)     NOT NULL DEFAULT 'FACILITY',
    peak_classification     VARCHAR(30)     NOT NULL DEFAULT 'ON_PEAK',
    billing_period_start    DATE,
    billing_period_end      DATE,
    is_billing_peak         BOOLEAN         DEFAULT false,
    is_annual_peak          BOOLEAN         DEFAULT false,
    is_ratchet_setting      BOOLEAN         DEFAULT false,
    prior_billing_peak_kw   NUMERIC(12,3),
    peak_increase_kw        NUMERIC(12,3),
    peak_increase_pct       NUMERIC(7,4),
    day_of_week             VARCHAR(10),
    hour_of_day             INTEGER,
    temperature_f           NUMERIC(6,2),
    heat_index_f            NUMERIC(6,2),
    wind_chill_f            NUMERIC(6,2),
    humidity_pct            NUMERIC(5,2),
    cloud_cover_pct         NUMERIC(5,2),
    weather_condition       VARCHAR(50),
    weather_data            JSONB           DEFAULT '{}',
    coincident_system_peak  BOOLEAN         DEFAULT false,
    system_load_mw          NUMERIC(12,3),
    operational_context     JSONB           DEFAULT '{}',
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_pe_peak_kw CHECK (
        peak_kw > 0
    ),
    CONSTRAINT chk_p038_pe_peak_kva CHECK (
        peak_kva IS NULL OR peak_kva > 0
    ),
    CONSTRAINT chk_p038_pe_pf CHECK (
        power_factor_at_peak IS NULL OR (power_factor_at_peak >= 0 AND power_factor_at_peak <= 1.0)
    ),
    CONSTRAINT chk_p038_pe_duration CHECK (
        duration_above_threshold_min IS NULL OR duration_above_threshold_min >= 0
    ),
    CONSTRAINT chk_p038_pe_type CHECK (
        peak_type IN (
            'FACILITY', 'COINCIDENT', 'NON_COINCIDENT', 'ZONAL', 'SYSTEM',
            'TRANSMISSION', 'DISTRIBUTION', 'GENERATION'
        )
    ),
    CONSTRAINT chk_p038_pe_classification CHECK (
        peak_classification IN (
            'ON_PEAK', 'MID_PEAK', 'OFF_PEAK', 'SUPER_PEAK', 'CRITICAL_PEAK',
            'SHOULDER', 'ALL_HOURS'
        )
    ),
    CONSTRAINT chk_p038_pe_billing_dates CHECK (
        billing_period_start IS NULL OR billing_period_end IS NULL OR
        billing_period_start <= billing_period_end
    ),
    CONSTRAINT chk_p038_pe_hour CHECK (
        hour_of_day IS NULL OR (hour_of_day >= 0 AND hour_of_day <= 23)
    ),
    CONSTRAINT chk_p038_pe_humidity CHECK (
        humidity_pct IS NULL OR (humidity_pct >= 0 AND humidity_pct <= 100)
    ),
    CONSTRAINT chk_p038_pe_cloud CHECK (
        cloud_cover_pct IS NULL OR (cloud_cover_pct >= 0 AND cloud_cover_pct <= 100)
    ),
    CONSTRAINT chk_p038_pe_increase_pct CHECK (
        peak_increase_pct IS NULL OR peak_increase_pct >= -100
    ),
    CONSTRAINT chk_p038_pe_day CHECK (
        day_of_week IS NULL OR day_of_week IN (
            'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY',
            'SATURDAY', 'SUNDAY'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_pe_profile         ON pack038_peak_shaving.ps_peak_events(profile_id);
CREATE INDEX idx_p038_pe_tenant          ON pack038_peak_shaving.ps_peak_events(tenant_id);
CREATE INDEX idx_p038_pe_timestamp       ON pack038_peak_shaving.ps_peak_events(peak_timestamp DESC);
CREATE INDEX idx_p038_pe_peak_kw         ON pack038_peak_shaving.ps_peak_events(peak_kw DESC);
CREATE INDEX idx_p038_pe_type            ON pack038_peak_shaving.ps_peak_events(peak_type);
CREATE INDEX idx_p038_pe_classification  ON pack038_peak_shaving.ps_peak_events(peak_classification);
CREATE INDEX idx_p038_pe_billing         ON pack038_peak_shaving.ps_peak_events(billing_period_start, billing_period_end);
CREATE INDEX idx_p038_pe_billing_peak    ON pack038_peak_shaving.ps_peak_events(is_billing_peak) WHERE is_billing_peak = true;
CREATE INDEX idx_p038_pe_annual_peak     ON pack038_peak_shaving.ps_peak_events(is_annual_peak) WHERE is_annual_peak = true;
CREATE INDEX idx_p038_pe_ratchet         ON pack038_peak_shaving.ps_peak_events(is_ratchet_setting) WHERE is_ratchet_setting = true;
CREATE INDEX idx_p038_pe_coincident      ON pack038_peak_shaving.ps_peak_events(coincident_system_peak) WHERE coincident_system_peak = true;
CREATE INDEX idx_p038_pe_weather         ON pack038_peak_shaving.ps_peak_events USING GIN(weather_data);
CREATE INDEX idx_p038_pe_created         ON pack038_peak_shaving.ps_peak_events(created_at DESC);

-- Composite: profile + billing period for billing analysis
CREATE INDEX idx_p038_pe_profile_billing ON pack038_peak_shaving.ps_peak_events(profile_id, billing_period_start DESC, peak_kw DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_pe_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_peak_events
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_peak_attribution
-- =============================================================================
-- Attributes each peak event to contributing loads, processes, or
-- conditions. Identifies which loads drove the peak and their relative
-- contribution for targeted peak reduction strategies.

CREATE TABLE pack038_peak_shaving.ps_peak_attribution (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    peak_event_id           UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_peak_events(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    attribution_source      VARCHAR(50)     NOT NULL,
    source_id               UUID,
    source_name             VARCHAR(255)    NOT NULL,
    source_category         VARCHAR(50),
    contribution_kw         NUMERIC(12,3)   NOT NULL,
    contribution_pct        NUMERIC(7,4)    NOT NULL,
    baseline_kw             NUMERIC(12,3),
    incremental_kw          NUMERIC(12,3),
    is_controllable         BOOLEAN         DEFAULT false,
    is_shiftable            BOOLEAN         DEFAULT false,
    curtailment_potential_kw NUMERIC(12,3),
    priority_rank           INTEGER,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_pa_source CHECK (
        attribution_source IN (
            'LOAD_ASSET', 'CIRCUIT', 'PROCESS_LINE', 'BUILDING_ZONE',
            'HVAC_SYSTEM', 'LIGHTING_ZONE', 'PRODUCTION_LINE',
            'EV_CHARGING', 'DATA_HALL', 'WEATHER', 'OCCUPANCY',
            'STARTUP_SEQUENCE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_pa_contribution_kw CHECK (
        contribution_kw >= 0
    ),
    CONSTRAINT chk_p038_pa_contribution_pct CHECK (
        contribution_pct >= 0 AND contribution_pct <= 100
    ),
    CONSTRAINT chk_p038_pa_baseline CHECK (
        baseline_kw IS NULL OR baseline_kw >= 0
    ),
    CONSTRAINT chk_p038_pa_curtailment CHECK (
        curtailment_potential_kw IS NULL OR curtailment_potential_kw >= 0
    ),
    CONSTRAINT chk_p038_pa_category CHECK (
        source_category IS NULL OR source_category IN (
            'HVAC', 'LIGHTING', 'PROCESS', 'REFRIGERATION', 'PUMPING',
            'COMPRESSED_AIR', 'EV_CHARGING', 'WATER_HEATING', 'SPACE_HEATING',
            'VENTILATION', 'IT_LOAD', 'MOTOR_LOAD', 'ELECTRIC_HEAT',
            'COOKING', 'LAUNDRY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_pa_priority CHECK (
        priority_rank IS NULL OR (priority_rank >= 1 AND priority_rank <= 999)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_pa_peak_event      ON pack038_peak_shaving.ps_peak_attribution(peak_event_id);
CREATE INDEX idx_p038_pa_tenant          ON pack038_peak_shaving.ps_peak_attribution(tenant_id);
CREATE INDEX idx_p038_pa_source          ON pack038_peak_shaving.ps_peak_attribution(attribution_source);
CREATE INDEX idx_p038_pa_source_id       ON pack038_peak_shaving.ps_peak_attribution(source_id);
CREATE INDEX idx_p038_pa_category        ON pack038_peak_shaving.ps_peak_attribution(source_category);
CREATE INDEX idx_p038_pa_contribution    ON pack038_peak_shaving.ps_peak_attribution(contribution_kw DESC);
CREATE INDEX idx_p038_pa_controllable    ON pack038_peak_shaving.ps_peak_attribution(is_controllable) WHERE is_controllable = true;
CREATE INDEX idx_p038_pa_created         ON pack038_peak_shaving.ps_peak_attribution(created_at DESC);

-- Composite: peak event + contribution ranking
CREATE INDEX idx_p038_pa_event_rank      ON pack038_peak_shaving.ps_peak_attribution(peak_event_id, contribution_kw DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_pa_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_peak_attribution
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_peak_clusters
-- =============================================================================
-- Groups similar peak events into clusters based on timing, magnitude,
-- weather, and operational patterns. Used to identify recurring peak
-- scenarios and develop targeted mitigation strategies.

CREATE TABLE pack038_peak_shaving.ps_peak_clusters (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    cluster_name            VARCHAR(100)    NOT NULL,
    cluster_type            VARCHAR(30)     NOT NULL,
    event_count             INTEGER         NOT NULL DEFAULT 0,
    avg_peak_kw             NUMERIC(12,3),
    max_peak_kw             NUMERIC(12,3),
    min_peak_kw             NUMERIC(12,3),
    avg_temperature_f       NUMERIC(6,2),
    typical_hour_start      INTEGER,
    typical_hour_end        INTEGER,
    typical_day_of_week     VARCHAR(10),
    typical_season          VARCHAR(20),
    primary_driver          VARCHAR(50),
    secondary_driver        VARCHAR(50),
    avoidability_pct        NUMERIC(5,2),
    mitigation_strategy     TEXT,
    risk_score              NUMERIC(5,2),
    frequency_per_year      NUMERIC(6,2),
    cost_impact_annual      NUMERIC(12,2),
    pattern_description     TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_pc_type CHECK (
        cluster_type IN (
            'WEATHER_DRIVEN', 'OPERATIONAL', 'STARTUP', 'COINCIDENT',
            'EQUIPMENT_FAILURE', 'SCHEDULING', 'SEASONAL', 'RANDOM', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_pc_event_count CHECK (
        event_count >= 0
    ),
    CONSTRAINT chk_p038_pc_hour_start CHECK (
        typical_hour_start IS NULL OR (typical_hour_start >= 0 AND typical_hour_start <= 23)
    ),
    CONSTRAINT chk_p038_pc_hour_end CHECK (
        typical_hour_end IS NULL OR (typical_hour_end >= 0 AND typical_hour_end <= 23)
    ),
    CONSTRAINT chk_p038_pc_season CHECK (
        typical_season IS NULL OR typical_season IN ('SUMMER', 'WINTER', 'SHOULDER', 'SPRING', 'FALL')
    ),
    CONSTRAINT chk_p038_pc_avoidability CHECK (
        avoidability_pct IS NULL OR (avoidability_pct >= 0 AND avoidability_pct <= 100)
    ),
    CONSTRAINT chk_p038_pc_risk CHECK (
        risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)
    ),
    CONSTRAINT chk_p038_pc_driver CHECK (
        primary_driver IS NULL OR primary_driver IN (
            'COOLING_LOAD', 'HEATING_LOAD', 'PRODUCTION_RAMP', 'SHIFT_OVERLAP',
            'EQUIPMENT_STARTUP', 'EV_CHARGING', 'SOLAR_DUCK_CURVE',
            'WEATHER_EVENT', 'OCCUPANCY_SURGE', 'COMPRESSOR_CYCLING',
            'DATA_CENTER_WORKLOAD', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_pc_profile         ON pack038_peak_shaving.ps_peak_clusters(profile_id);
CREATE INDEX idx_p038_pc_tenant          ON pack038_peak_shaving.ps_peak_clusters(tenant_id);
CREATE INDEX idx_p038_pc_type            ON pack038_peak_shaving.ps_peak_clusters(cluster_type);
CREATE INDEX idx_p038_pc_driver          ON pack038_peak_shaving.ps_peak_clusters(primary_driver);
CREATE INDEX idx_p038_pc_risk            ON pack038_peak_shaving.ps_peak_clusters(risk_score DESC);
CREATE INDEX idx_p038_pc_cost            ON pack038_peak_shaving.ps_peak_clusters(cost_impact_annual DESC);
CREATE INDEX idx_p038_pc_created         ON pack038_peak_shaving.ps_peak_clusters(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_pc_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_peak_clusters
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_peak_simulations
-- =============================================================================
-- What-if simulation results for peak reduction scenarios. Models the
-- impact of different peak shaving strategies (BESS, load shifting,
-- demand limiting) on peak demand and demand charges.

CREATE TABLE pack038_peak_shaving.ps_peak_simulations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    simulation_name         VARCHAR(255)    NOT NULL,
    simulation_type         VARCHAR(30)     NOT NULL,
    scenario_params         JSONB           NOT NULL DEFAULT '{}',
    baseline_peak_kw        NUMERIC(12,3)   NOT NULL,
    simulated_peak_kw       NUMERIC(12,3)   NOT NULL,
    peak_reduction_kw       NUMERIC(12,3),
    peak_reduction_pct      NUMERIC(7,4),
    baseline_demand_charge  NUMERIC(12,2),
    simulated_demand_charge NUMERIC(12,2),
    demand_charge_savings   NUMERIC(12,2),
    annual_savings          NUMERIC(12,2),
    energy_shifted_kwh      NUMERIC(15,3),
    energy_stored_kwh       NUMERIC(15,3),
    round_trip_losses_kwh   NUMERIC(15,3),
    peak_events_avoided     INTEGER,
    simulation_period_start DATE,
    simulation_period_end   DATE,
    confidence_level_pct    NUMERIC(5,2),
    run_timestamp           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    execution_time_ms       INTEGER,
    simulation_status       VARCHAR(20)     NOT NULL DEFAULT 'COMPLETED',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ps_type CHECK (
        simulation_type IN (
            'BESS_DISPATCH', 'LOAD_SHIFTING', 'DEMAND_LIMITING',
            'COMBINED_BESS_SHIFT', 'GENERATOR_BACKUP', 'SOLAR_PLUS_STORAGE',
            'HVAC_PRECOOLING', 'PRODUCTION_SCHEDULING', 'WHAT_IF'
        )
    ),
    CONSTRAINT chk_p038_ps_baseline CHECK (
        baseline_peak_kw > 0
    ),
    CONSTRAINT chk_p038_ps_simulated CHECK (
        simulated_peak_kw >= 0
    ),
    CONSTRAINT chk_p038_ps_confidence CHECK (
        confidence_level_pct IS NULL OR (confidence_level_pct >= 0 AND confidence_level_pct <= 100)
    ),
    CONSTRAINT chk_p038_ps_status CHECK (
        simulation_status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p038_ps_dates CHECK (
        simulation_period_start IS NULL OR simulation_period_end IS NULL OR
        simulation_period_start <= simulation_period_end
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ps_profile         ON pack038_peak_shaving.ps_peak_simulations(profile_id);
CREATE INDEX idx_p038_ps_tenant          ON pack038_peak_shaving.ps_peak_simulations(tenant_id);
CREATE INDEX idx_p038_ps_type            ON pack038_peak_shaving.ps_peak_simulations(simulation_type);
CREATE INDEX idx_p038_ps_savings         ON pack038_peak_shaving.ps_peak_simulations(annual_savings DESC);
CREATE INDEX idx_p038_ps_reduction       ON pack038_peak_shaving.ps_peak_simulations(peak_reduction_kw DESC);
CREATE INDEX idx_p038_ps_status          ON pack038_peak_shaving.ps_peak_simulations(simulation_status);
CREATE INDEX idx_p038_ps_run             ON pack038_peak_shaving.ps_peak_simulations(run_timestamp DESC);
CREATE INDEX idx_p038_ps_created         ON pack038_peak_shaving.ps_peak_simulations(created_at DESC);
CREATE INDEX idx_p038_ps_params          ON pack038_peak_shaving.ps_peak_simulations USING GIN(scenario_params);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ps_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_peak_simulations
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_avoidability_assessments
-- =============================================================================
-- Assessment of whether each detected peak event was avoidable and what
-- mitigation actions could have prevented it. Forms the basis for
-- operational recommendations and ROI calculations.

CREATE TABLE pack038_peak_shaving.ps_avoidability_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    peak_event_id           UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_peak_events(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    assessment_type         VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATED',
    avoidability_rating     VARCHAR(20)     NOT NULL,
    avoidability_score      NUMERIC(5,2),
    primary_mitigation      VARCHAR(50),
    secondary_mitigation    VARCHAR(50),
    estimated_reduction_kw  NUMERIC(12,3),
    estimated_savings       NUMERIC(12,2),
    implementation_cost     NUMERIC(12,2),
    payback_months          NUMERIC(8,2),
    advance_notice_needed_min INTEGER,
    operational_impact      VARCHAR(20)     DEFAULT 'NONE',
    comfort_impact          VARCHAR(20)     DEFAULT 'NONE',
    safety_risk             VARCHAR(20)     DEFAULT 'NONE',
    root_cause              TEXT,
    recommendation          TEXT,
    assessed_by             VARCHAR(255),
    assessed_at             TIMESTAMPTZ     DEFAULT NOW(),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_aa_type CHECK (
        assessment_type IN ('AUTOMATED', 'MANUAL', 'AI_ASSISTED', 'HYBRID')
    ),
    CONSTRAINT chk_p038_aa_rating CHECK (
        avoidability_rating IN (
            'FULLY_AVOIDABLE', 'PARTIALLY_AVOIDABLE', 'DIFFICULT_TO_AVOID',
            'UNAVOIDABLE', 'UNKNOWN'
        )
    ),
    CONSTRAINT chk_p038_aa_score CHECK (
        avoidability_score IS NULL OR (avoidability_score >= 0 AND avoidability_score <= 100)
    ),
    CONSTRAINT chk_p038_aa_mitigation CHECK (
        primary_mitigation IS NULL OR primary_mitigation IN (
            'BESS_DISCHARGE', 'LOAD_SHEDDING', 'LOAD_SHIFTING', 'HVAC_SETBACK',
            'PRODUCTION_RESCHEDULE', 'EV_CHARGE_DELAY', 'GENERATOR_START',
            'PRECOOLING', 'THERMAL_STORAGE', 'DEMAND_LIMITING', 'SOLAR_CURTAIL',
            'STAGGER_STARTUP', 'PROCESS_OPTIMIZATION', 'NONE'
        )
    ),
    CONSTRAINT chk_p038_aa_reduction CHECK (
        estimated_reduction_kw IS NULL OR estimated_reduction_kw >= 0
    ),
    CONSTRAINT chk_p038_aa_operational CHECK (
        operational_impact IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p038_aa_comfort CHECK (
        comfort_impact IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p038_aa_safety CHECK (
        safety_risk IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p038_aa_payback CHECK (
        payback_months IS NULL OR payback_months >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_aa_peak_event      ON pack038_peak_shaving.ps_avoidability_assessments(peak_event_id);
CREATE INDEX idx_p038_aa_tenant          ON pack038_peak_shaving.ps_avoidability_assessments(tenant_id);
CREATE INDEX idx_p038_aa_rating          ON pack038_peak_shaving.ps_avoidability_assessments(avoidability_rating);
CREATE INDEX idx_p038_aa_mitigation      ON pack038_peak_shaving.ps_avoidability_assessments(primary_mitigation);
CREATE INDEX idx_p038_aa_score           ON pack038_peak_shaving.ps_avoidability_assessments(avoidability_score DESC);
CREATE INDEX idx_p038_aa_savings         ON pack038_peak_shaving.ps_avoidability_assessments(estimated_savings DESC);
CREATE INDEX idx_p038_aa_payback         ON pack038_peak_shaving.ps_avoidability_assessments(payback_months);
CREATE INDEX idx_p038_aa_created         ON pack038_peak_shaving.ps_avoidability_assessments(created_at DESC);

-- Composite: avoidable peaks with savings for prioritization
CREATE INDEX idx_p038_aa_avoidable_save  ON pack038_peak_shaving.ps_avoidability_assessments(avoidability_rating, estimated_savings DESC)
    WHERE avoidability_rating IN ('FULLY_AVOIDABLE', 'PARTIALLY_AVOIDABLE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_aa_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_avoidability_assessments
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_peak_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_peak_attribution ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_peak_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_peak_simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_avoidability_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_pe_tenant_isolation
    ON pack038_peak_shaving.ps_peak_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_pe_service_bypass
    ON pack038_peak_shaving.ps_peak_events
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_pa_tenant_isolation
    ON pack038_peak_shaving.ps_peak_attribution
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_pa_service_bypass
    ON pack038_peak_shaving.ps_peak_attribution
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_pc_tenant_isolation
    ON pack038_peak_shaving.ps_peak_clusters
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_pc_service_bypass
    ON pack038_peak_shaving.ps_peak_clusters
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ps_tenant_isolation
    ON pack038_peak_shaving.ps_peak_simulations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ps_service_bypass
    ON pack038_peak_shaving.ps_peak_simulations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_aa_tenant_isolation
    ON pack038_peak_shaving.ps_avoidability_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_aa_service_bypass
    ON pack038_peak_shaving.ps_avoidability_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_peak_events TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_peak_attribution TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_peak_clusters TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_peak_simulations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_avoidability_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_peak_events IS
    'Detected peak demand events with magnitude, timing, weather conditions, billing context, and coincident system peak flags.';
COMMENT ON TABLE pack038_peak_shaving.ps_peak_attribution IS
    'Attribution of peak events to contributing loads, processes, or conditions with controllability and curtailment potential.';
COMMENT ON TABLE pack038_peak_shaving.ps_peak_clusters IS
    'Clustered peak event patterns by timing, weather, and operational context for targeted mitigation strategy development.';
COMMENT ON TABLE pack038_peak_shaving.ps_peak_simulations IS
    'What-if simulation results modeling peak reduction impact of BESS, load shifting, and demand limiting strategies.';
COMMENT ON TABLE pack038_peak_shaving.ps_avoidability_assessments IS
    'Assessment of peak avoidability with mitigation recommendations, savings estimates, and implementation cost analysis.';

COMMENT ON COLUMN pack038_peak_shaving.ps_peak_events.peak_kw IS 'Peak demand in kilowatts at the event timestamp.';
COMMENT ON COLUMN pack038_peak_shaving.ps_peak_events.peak_type IS 'Peak type: FACILITY, COINCIDENT, NON_COINCIDENT, ZONAL, SYSTEM, TRANSMISSION, DISTRIBUTION, GENERATION.';
COMMENT ON COLUMN pack038_peak_shaving.ps_peak_events.is_billing_peak IS 'Whether this peak set the billing demand for the billing period.';
COMMENT ON COLUMN pack038_peak_shaving.ps_peak_events.is_ratchet_setting IS 'Whether this peak triggered a demand ratchet clause.';
COMMENT ON COLUMN pack038_peak_shaving.ps_peak_events.coincident_system_peak IS 'Whether this facility peak coincided with the system/grid peak.';
COMMENT ON COLUMN pack038_peak_shaving.ps_peak_events.weather_data IS 'JSONB weather conditions at peak time including temperature, humidity, wind, solar irradiance.';

COMMENT ON COLUMN pack038_peak_shaving.ps_peak_attribution.contribution_pct IS 'Percentage contribution of this source to the total peak demand (0-100).';
COMMENT ON COLUMN pack038_peak_shaving.ps_peak_attribution.curtailment_potential_kw IS 'Estimated kW that could be curtailed from this source during a peak event.';

COMMENT ON COLUMN pack038_peak_shaving.ps_peak_clusters.avoidability_pct IS 'Estimated percentage of peaks in this cluster that were avoidable (0-100).';
COMMENT ON COLUMN pack038_peak_shaving.ps_peak_clusters.cost_impact_annual IS 'Estimated annual demand charge cost impact of this peak cluster pattern.';

COMMENT ON COLUMN pack038_peak_shaving.ps_avoidability_assessments.avoidability_rating IS 'Overall avoidability: FULLY_AVOIDABLE, PARTIALLY_AVOIDABLE, DIFFICULT_TO_AVOID, UNAVOIDABLE, UNKNOWN.';
COMMENT ON COLUMN pack038_peak_shaving.ps_avoidability_assessments.primary_mitigation IS 'Primary recommended mitigation strategy for this peak event.';
COMMENT ON COLUMN pack038_peak_shaving.ps_avoidability_assessments.payback_months IS 'Estimated payback period in months for implementing the mitigation strategy.';
