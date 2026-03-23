-- =============================================================================
-- V302: PACK-038 Peak Shaving Pack - Ratchet Analysis
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Demand ratchet analysis tables for tracking ratchet clause impacts,
-- spike analysis, prevention planning, and alert management. Ratchet
-- clauses set a minimum billing demand based on historical peaks,
-- making even a single demand spike extremely costly over many months.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_ratchet_history
--   2. pack038_peak_shaving.ps_ratchet_impacts
--   3. pack038_peak_shaving.ps_spike_analysis
--   4. pack038_peak_shaving.ps_prevention_plans
--   5. pack038_peak_shaving.ps_ratchet_alerts
--
-- Previous: V301__pack038_peak_shaving_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_ratchet_history
-- =============================================================================
-- Historical record of ratchet demand values per billing period. Tracks
-- the original peak that set the ratchet, the ratchet percentage applied,
-- and the resulting billing demand floor for each month.

CREATE TABLE pack038_peak_shaving.ps_ratchet_history (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tariff_id               UUID            REFERENCES pack038_peak_shaving.ps_tariff_structures(id),
    tenant_id               UUID            NOT NULL,
    billing_month           DATE            NOT NULL,
    ratchet_type            VARCHAR(30)     NOT NULL DEFAULT 'PERCENTAGE',
    ratchet_pct             NUMERIC(5,2)    NOT NULL,
    lookback_months         INTEGER         NOT NULL DEFAULT 11,
    source_peak_kw          NUMERIC(12,3)   NOT NULL,
    source_peak_date        DATE            NOT NULL,
    source_peak_month       DATE,
    ratchet_demand_kw       NUMERIC(12,3)   NOT NULL,
    actual_demand_kw        NUMERIC(12,3)   NOT NULL,
    billing_demand_kw       NUMERIC(12,3)   NOT NULL,
    ratchet_was_binding     BOOLEAN         NOT NULL DEFAULT false,
    excess_billed_kw        NUMERIC(12,3),
    excess_charge           NUMERIC(12,2),
    cumulative_ratchet_cost NUMERIC(12,2),
    months_remaining        INTEGER,
    ratchet_expiration_date DATE,
    rate_per_kw             NUMERIC(10,4),
    season_applied          VARCHAR(20),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_rh_type CHECK (
        ratchet_type IN ('PERCENTAGE', 'FIXED', 'SEASONAL', 'ANNUAL', 'ROLLING')
    ),
    CONSTRAINT chk_p038_rh_pct CHECK (
        ratchet_pct >= 0 AND ratchet_pct <= 100
    ),
    CONSTRAINT chk_p038_rh_lookback CHECK (
        lookback_months >= 1 AND lookback_months <= 36
    ),
    CONSTRAINT chk_p038_rh_source_peak CHECK (
        source_peak_kw > 0
    ),
    CONSTRAINT chk_p038_rh_ratchet_demand CHECK (
        ratchet_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_rh_actual CHECK (
        actual_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_rh_billing CHECK (
        billing_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_rh_months CHECK (
        months_remaining IS NULL OR months_remaining >= 0
    ),
    CONSTRAINT chk_p038_rh_season CHECK (
        season_applied IS NULL OR season_applied IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT uq_p038_rh_profile_month UNIQUE (profile_id, tariff_id, billing_month)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_rh_profile         ON pack038_peak_shaving.ps_ratchet_history(profile_id);
CREATE INDEX idx_p038_rh_tariff          ON pack038_peak_shaving.ps_ratchet_history(tariff_id);
CREATE INDEX idx_p038_rh_tenant          ON pack038_peak_shaving.ps_ratchet_history(tenant_id);
CREATE INDEX idx_p038_rh_month           ON pack038_peak_shaving.ps_ratchet_history(billing_month DESC);
CREATE INDEX idx_p038_rh_binding         ON pack038_peak_shaving.ps_ratchet_history(ratchet_was_binding) WHERE ratchet_was_binding = true;
CREATE INDEX idx_p038_rh_excess          ON pack038_peak_shaving.ps_ratchet_history(excess_charge DESC);
CREATE INDEX idx_p038_rh_source_date     ON pack038_peak_shaving.ps_ratchet_history(source_peak_date DESC);
CREATE INDEX idx_p038_rh_expiration      ON pack038_peak_shaving.ps_ratchet_history(ratchet_expiration_date);
CREATE INDEX idx_p038_rh_created         ON pack038_peak_shaving.ps_ratchet_history(created_at DESC);

-- Binding ratchets by profile for cost analysis
CREATE INDEX idx_p038_rh_binding_cost    ON pack038_peak_shaving.ps_ratchet_history(profile_id, billing_month DESC, excess_charge DESC)
    WHERE ratchet_was_binding = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_rh_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_ratchet_history
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_ratchet_impacts
-- =============================================================================
-- Aggregated ratchet impact analysis per peak event or spike. Calculates
-- the total financial impact of a demand spike over the full ratchet
-- lookback period, including future billing impacts.

CREATE TABLE pack038_peak_shaving.ps_ratchet_impacts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    peak_event_id           UUID            REFERENCES pack038_peak_shaving.ps_peak_events(id),
    tenant_id               UUID            NOT NULL,
    spike_date              DATE            NOT NULL,
    spike_demand_kw         NUMERIC(12,3)   NOT NULL,
    prior_ratchet_kw        NUMERIC(12,3),
    new_ratchet_kw          NUMERIC(12,3)   NOT NULL,
    incremental_kw          NUMERIC(12,3),
    ratchet_pct             NUMERIC(5,2)    NOT NULL,
    months_affected         INTEGER         NOT NULL,
    monthly_excess_charge   NUMERIC(12,2),
    total_ratchet_cost      NUMERIC(12,2)   NOT NULL,
    cost_per_kw_spike       NUMERIC(10,4),
    first_affected_month    DATE,
    last_affected_month     DATE,
    avoidable               BOOLEAN         DEFAULT true,
    avoidance_cost          NUMERIC(12,2),
    net_avoidable_savings   NUMERIC(12,2),
    impact_severity         VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    mitigation_applied      BOOLEAN         DEFAULT false,
    mitigation_type         VARCHAR(50),
    actual_months_binding   INTEGER,
    actual_total_cost       NUMERIC(12,2),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ri_spike CHECK (
        spike_demand_kw > 0
    ),
    CONSTRAINT chk_p038_ri_new_ratchet CHECK (
        new_ratchet_kw >= 0
    ),
    CONSTRAINT chk_p038_ri_pct CHECK (
        ratchet_pct >= 0 AND ratchet_pct <= 100
    ),
    CONSTRAINT chk_p038_ri_months CHECK (
        months_affected >= 1
    ),
    CONSTRAINT chk_p038_ri_total_cost CHECK (
        total_ratchet_cost >= 0
    ),
    CONSTRAINT chk_p038_ri_severity CHECK (
        impact_severity IN ('MINOR', 'MODERATE', 'MAJOR', 'SEVERE', 'CRITICAL')
    ),
    CONSTRAINT chk_p038_ri_dates CHECK (
        first_affected_month IS NULL OR last_affected_month IS NULL OR
        first_affected_month <= last_affected_month
    ),
    CONSTRAINT chk_p038_ri_mitigation CHECK (
        mitigation_type IS NULL OR mitigation_type IN (
            'BESS_INSTALLED', 'DEMAND_LIMITER', 'LOAD_SHIFTING',
            'TARIFF_CHANGE', 'OPERATIONAL_CHANGE', 'GENERATOR_BACKUP',
            'NONE_APPLIED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ri_profile         ON pack038_peak_shaving.ps_ratchet_impacts(profile_id);
CREATE INDEX idx_p038_ri_peak_event      ON pack038_peak_shaving.ps_ratchet_impacts(peak_event_id);
CREATE INDEX idx_p038_ri_tenant          ON pack038_peak_shaving.ps_ratchet_impacts(tenant_id);
CREATE INDEX idx_p038_ri_date            ON pack038_peak_shaving.ps_ratchet_impacts(spike_date DESC);
CREATE INDEX idx_p038_ri_total_cost      ON pack038_peak_shaving.ps_ratchet_impacts(total_ratchet_cost DESC);
CREATE INDEX idx_p038_ri_severity        ON pack038_peak_shaving.ps_ratchet_impacts(impact_severity);
CREATE INDEX idx_p038_ri_avoidable       ON pack038_peak_shaving.ps_ratchet_impacts(avoidable) WHERE avoidable = true;
CREATE INDEX idx_p038_ri_mitigation      ON pack038_peak_shaving.ps_ratchet_impacts(mitigation_applied);
CREATE INDEX idx_p038_ri_created         ON pack038_peak_shaving.ps_ratchet_impacts(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_ri_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_ratchet_impacts
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_spike_analysis
-- =============================================================================
-- Analysis of demand spikes that are short-duration, high-magnitude
-- events capable of setting new billing demand ratchets. Tracks spike
-- characteristics, root causes, and frequency patterns.

CREATE TABLE pack038_peak_shaving.ps_spike_analysis (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    peak_event_id           UUID            REFERENCES pack038_peak_shaving.ps_peak_events(id),
    tenant_id               UUID            NOT NULL,
    spike_timestamp         TIMESTAMPTZ     NOT NULL,
    spike_kw                NUMERIC(12,3)   NOT NULL,
    baseline_kw             NUMERIC(12,3)   NOT NULL,
    spike_magnitude_kw      NUMERIC(12,3)   NOT NULL,
    spike_magnitude_pct     NUMERIC(7,4),
    spike_duration_min      INTEGER         NOT NULL,
    rise_time_min           INTEGER,
    fall_time_min           INTEGER,
    area_under_spike_kwh    NUMERIC(12,3),
    spike_category          VARCHAR(30)     NOT NULL,
    root_cause              VARCHAR(50),
    root_cause_detail       TEXT,
    equipment_involved      JSONB           DEFAULT '[]',
    is_recurring            BOOLEAN         DEFAULT false,
    recurrence_pattern      VARCHAR(50),
    occurrences_last_12m    INTEGER,
    preventable             BOOLEAN         DEFAULT true,
    prevention_method       VARCHAR(50),
    prevention_cost         NUMERIC(12,2),
    ratchet_risk            BOOLEAN         DEFAULT false,
    estimated_ratchet_cost  NUMERIC(12,2),
    severity_score          NUMERIC(5,2),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_sa_spike_kw CHECK (
        spike_kw > 0
    ),
    CONSTRAINT chk_p038_sa_baseline CHECK (
        baseline_kw >= 0
    ),
    CONSTRAINT chk_p038_sa_magnitude CHECK (
        spike_magnitude_kw > 0
    ),
    CONSTRAINT chk_p038_sa_duration CHECK (
        spike_duration_min >= 1
    ),
    CONSTRAINT chk_p038_sa_category CHECK (
        spike_category IN (
            'MOTOR_START', 'COMPRESSOR_CYCLING', 'PRODUCTION_STARTUP',
            'SHIFT_OVERLAP', 'EV_SIMULTANEOUS', 'EQUIPMENT_FAULT',
            'DEMAND_OSCILLATION', 'WEATHER_RESPONSE', 'REBOUND',
            'DATA_CENTER_BURST', 'ELEVATOR_BANK', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_sa_root_cause CHECK (
        root_cause IS NULL OR root_cause IN (
            'INRUSH_CURRENT', 'SEQUENCING_FAILURE', 'CONTROL_MALFUNCTION',
            'SIMULTANEOUS_START', 'WEATHER_EXTREME', 'OCCUPANCY_SURGE',
            'PRODUCTION_CHANGE', 'MAINTENANCE_RESTART', 'POWER_QUALITY',
            'SOLAR_VARIABILITY', 'GRID_EVENT', 'UNKNOWN', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_sa_recurrence CHECK (
        recurrence_pattern IS NULL OR recurrence_pattern IN (
            'DAILY', 'WEEKLY', 'MONTHLY', 'SEASONAL', 'RANDOM',
            'SHIFT_CHANGE', 'STARTUP_DAILY', 'WEATHER_CORRELATED'
        )
    ),
    CONSTRAINT chk_p038_sa_prevention CHECK (
        prevention_method IS NULL OR prevention_method IN (
            'SOFT_STARTER', 'VFD_INSTALL', 'SEQUENCING_CONTROL',
            'DEMAND_LIMITER', 'BESS_BUFFER', 'STAGGER_SCHEDULE',
            'LOAD_INTERLOCK', 'POWER_FACTOR_CORRECTION',
            'PREDICTIVE_CONTROL', 'OPERATIONAL_PROCEDURE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_sa_severity CHECK (
        severity_score IS NULL OR (severity_score >= 0 AND severity_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_sa_profile         ON pack038_peak_shaving.ps_spike_analysis(profile_id);
CREATE INDEX idx_p038_sa_peak_event      ON pack038_peak_shaving.ps_spike_analysis(peak_event_id);
CREATE INDEX idx_p038_sa_tenant          ON pack038_peak_shaving.ps_spike_analysis(tenant_id);
CREATE INDEX idx_p038_sa_timestamp       ON pack038_peak_shaving.ps_spike_analysis(spike_timestamp DESC);
CREATE INDEX idx_p038_sa_magnitude       ON pack038_peak_shaving.ps_spike_analysis(spike_magnitude_kw DESC);
CREATE INDEX idx_p038_sa_category        ON pack038_peak_shaving.ps_spike_analysis(spike_category);
CREATE INDEX idx_p038_sa_root_cause      ON pack038_peak_shaving.ps_spike_analysis(root_cause);
CREATE INDEX idx_p038_sa_ratchet_risk    ON pack038_peak_shaving.ps_spike_analysis(ratchet_risk) WHERE ratchet_risk = true;
CREATE INDEX idx_p038_sa_preventable     ON pack038_peak_shaving.ps_spike_analysis(preventable) WHERE preventable = true;
CREATE INDEX idx_p038_sa_severity        ON pack038_peak_shaving.ps_spike_analysis(severity_score DESC);
CREATE INDEX idx_p038_sa_recurring       ON pack038_peak_shaving.ps_spike_analysis(is_recurring) WHERE is_recurring = true;
CREATE INDEX idx_p038_sa_created         ON pack038_peak_shaving.ps_spike_analysis(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_sa_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_spike_analysis
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_prevention_plans
-- =============================================================================
-- Plans for preventing demand spikes and ratchet-setting peaks. Each
-- plan targets a specific spike category or root cause with estimated
-- cost and savings.

CREATE TABLE pack038_peak_shaving.ps_prevention_plans (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    plan_name               VARCHAR(255)    NOT NULL,
    target_spike_category   VARCHAR(30),
    target_root_cause       VARCHAR(50),
    prevention_method       VARCHAR(50)     NOT NULL,
    description             TEXT,
    implementation_cost     NUMERIC(12,2),
    annual_maintenance_cost NUMERIC(10,2),
    estimated_annual_savings NUMERIC(12,2),
    estimated_peak_reduction_kw NUMERIC(12,3),
    simple_payback_years    NUMERIC(6,2),
    npv                     NUMERIC(12,2),
    implementation_timeline_weeks INTEGER,
    complexity              VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    disruption_level        VARCHAR(20)     NOT NULL DEFAULT 'LOW',
    equipment_required      JSONB           DEFAULT '[]',
    vendor_quotes           JSONB           DEFAULT '[]',
    plan_status             VARCHAR(20)     NOT NULL DEFAULT 'PROPOSED',
    approved_by             VARCHAR(255),
    approved_at             TIMESTAMPTZ,
    implemented_at          DATE,
    effectiveness_verified  BOOLEAN         DEFAULT false,
    actual_reduction_kw     NUMERIC(12,3),
    actual_savings          NUMERIC(12,2),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_pp_method CHECK (
        prevention_method IN (
            'SOFT_STARTER', 'VFD_INSTALL', 'SEQUENCING_CONTROL',
            'DEMAND_LIMITER', 'BESS_BUFFER', 'STAGGER_SCHEDULE',
            'LOAD_INTERLOCK', 'POWER_FACTOR_CORRECTION',
            'PREDICTIVE_CONTROL', 'OPERATIONAL_PROCEDURE',
            'BUILDING_AUTOMATION', 'THERMAL_STORAGE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_pp_complexity CHECK (
        complexity IN ('SIMPLE', 'MEDIUM', 'COMPLEX', 'MAJOR_PROJECT')
    ),
    CONSTRAINT chk_p038_pp_disruption CHECK (
        disruption_level IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'FULL_SHUTDOWN')
    ),
    CONSTRAINT chk_p038_pp_status CHECK (
        plan_status IN (
            'PROPOSED', 'EVALUATING', 'APPROVED', 'IN_PROGRESS',
            'COMPLETED', 'DEFERRED', 'REJECTED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p038_pp_payback CHECK (
        simple_payback_years IS NULL OR simple_payback_years >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_pp_profile         ON pack038_peak_shaving.ps_prevention_plans(profile_id);
CREATE INDEX idx_p038_pp_tenant          ON pack038_peak_shaving.ps_prevention_plans(tenant_id);
CREATE INDEX idx_p038_pp_method          ON pack038_peak_shaving.ps_prevention_plans(prevention_method);
CREATE INDEX idx_p038_pp_category        ON pack038_peak_shaving.ps_prevention_plans(target_spike_category);
CREATE INDEX idx_p038_pp_status          ON pack038_peak_shaving.ps_prevention_plans(plan_status);
CREATE INDEX idx_p038_pp_savings         ON pack038_peak_shaving.ps_prevention_plans(estimated_annual_savings DESC);
CREATE INDEX idx_p038_pp_payback         ON pack038_peak_shaving.ps_prevention_plans(simple_payback_years);
CREATE INDEX idx_p038_pp_complexity      ON pack038_peak_shaving.ps_prevention_plans(complexity);
CREATE INDEX idx_p038_pp_created         ON pack038_peak_shaving.ps_prevention_plans(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_pp_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_prevention_plans
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_ratchet_alerts
-- =============================================================================
-- Real-time and threshold-based alerts for demand approaching or
-- exceeding ratchet-setting levels. Enables proactive intervention
-- before a new ratchet is set.

CREATE TABLE pack038_peak_shaving.ps_ratchet_alerts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    alert_timestamp         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    alert_type              VARCHAR(30)     NOT NULL,
    alert_severity          VARCHAR(20)     NOT NULL,
    current_demand_kw       NUMERIC(12,3)   NOT NULL,
    ratchet_threshold_kw    NUMERIC(12,3)   NOT NULL,
    headroom_kw             NUMERIC(12,3),
    headroom_pct            NUMERIC(7,4),
    projected_peak_kw       NUMERIC(12,3),
    time_to_peak_min        INTEGER,
    estimated_ratchet_cost  NUMERIC(12,2),
    recommended_action      VARCHAR(50),
    action_taken            VARCHAR(50),
    action_timestamp        TIMESTAMPTZ,
    action_effective        BOOLEAN,
    demand_after_action_kw  NUMERIC(12,3),
    ratchet_avoided         BOOLEAN,
    notification_sent       BOOLEAN         DEFAULT false,
    notification_channels   JSONB           DEFAULT '[]',
    acknowledged_by         VARCHAR(255),
    acknowledged_at         TIMESTAMPTZ,
    resolved                BOOLEAN         DEFAULT false,
    resolved_at             TIMESTAMPTZ,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_ra_type CHECK (
        alert_type IN (
            'APPROACHING_THRESHOLD', 'AT_THRESHOLD', 'EXCEEDED_THRESHOLD',
            'NEW_RATCHET_SET', 'RATCHET_EXPIRING', 'SPIKE_DETECTED',
            'PREDICTIVE_WARNING', 'SYSTEM_PEAK_ALERT'
        )
    ),
    CONSTRAINT chk_p038_ra_severity CHECK (
        alert_severity IN ('INFO', 'WARNING', 'CRITICAL', 'EMERGENCY')
    ),
    CONSTRAINT chk_p038_ra_demand CHECK (
        current_demand_kw >= 0
    ),
    CONSTRAINT chk_p038_ra_threshold CHECK (
        ratchet_threshold_kw > 0
    ),
    CONSTRAINT chk_p038_ra_action CHECK (
        recommended_action IS NULL OR recommended_action IN (
            'SHED_LOAD', 'START_BESS', 'REDUCE_HVAC', 'DELAY_EV_CHARGING',
            'START_GENERATOR', 'ACTIVATE_DEMAND_LIMITER', 'MANUAL_OVERRIDE',
            'SHIFT_PRODUCTION', 'NO_ACTION_NEEDED', 'INVESTIGATE'
        )
    ),
    CONSTRAINT chk_p038_ra_action_taken CHECK (
        action_taken IS NULL OR action_taken IN (
            'SHED_LOAD', 'START_BESS', 'REDUCE_HVAC', 'DELAY_EV_CHARGING',
            'START_GENERATOR', 'ACTIVATE_DEMAND_LIMITER', 'MANUAL_OVERRIDE',
            'SHIFT_PRODUCTION', 'NO_ACTION', 'ACKNOWLEDGED_ONLY'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_ra_profile         ON pack038_peak_shaving.ps_ratchet_alerts(profile_id);
CREATE INDEX idx_p038_ra_tenant          ON pack038_peak_shaving.ps_ratchet_alerts(tenant_id);
CREATE INDEX idx_p038_ra_timestamp       ON pack038_peak_shaving.ps_ratchet_alerts(alert_timestamp DESC);
CREATE INDEX idx_p038_ra_type            ON pack038_peak_shaving.ps_ratchet_alerts(alert_type);
CREATE INDEX idx_p038_ra_severity        ON pack038_peak_shaving.ps_ratchet_alerts(alert_severity);
CREATE INDEX idx_p038_ra_resolved        ON pack038_peak_shaving.ps_ratchet_alerts(resolved) WHERE resolved = false;
CREATE INDEX idx_p038_ra_avoided         ON pack038_peak_shaving.ps_ratchet_alerts(ratchet_avoided);
CREATE INDEX idx_p038_ra_created         ON pack038_peak_shaving.ps_ratchet_alerts(created_at DESC);

-- Unresolved critical alerts for dashboard
CREATE INDEX idx_p038_ra_unresolved_crit ON pack038_peak_shaving.ps_ratchet_alerts(profile_id, alert_timestamp DESC)
    WHERE resolved = false AND alert_severity IN ('CRITICAL', 'EMERGENCY');

-- ---------------------------------------------------------------------------
-- (No trigger - no updated_at column)
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_ratchet_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_ratchet_impacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_spike_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_prevention_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_ratchet_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_rh_tenant_isolation
    ON pack038_peak_shaving.ps_ratchet_history
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_rh_service_bypass
    ON pack038_peak_shaving.ps_ratchet_history
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ri_tenant_isolation
    ON pack038_peak_shaving.ps_ratchet_impacts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ri_service_bypass
    ON pack038_peak_shaving.ps_ratchet_impacts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_sa_tenant_isolation
    ON pack038_peak_shaving.ps_spike_analysis
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_sa_service_bypass
    ON pack038_peak_shaving.ps_spike_analysis
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_pp_tenant_isolation
    ON pack038_peak_shaving.ps_prevention_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_pp_service_bypass
    ON pack038_peak_shaving.ps_prevention_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_ra_tenant_isolation
    ON pack038_peak_shaving.ps_ratchet_alerts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_ra_service_bypass
    ON pack038_peak_shaving.ps_ratchet_alerts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_ratchet_history TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_ratchet_impacts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_spike_analysis TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_prevention_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_ratchet_alerts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_ratchet_history IS
    'Historical ratchet demand values per billing period with source peak, ratchet percentage, and binding status.';
COMMENT ON TABLE pack038_peak_shaving.ps_ratchet_impacts IS
    'Aggregated financial impact of demand spikes over the full ratchet lookback period with total cost calculations.';
COMMENT ON TABLE pack038_peak_shaving.ps_spike_analysis IS
    'Demand spike characterization with root cause, duration, recurrence pattern, and ratchet risk assessment.';
COMMENT ON TABLE pack038_peak_shaving.ps_prevention_plans IS
    'Spike prevention plans targeting specific categories and root causes with implementation cost and savings estimates.';
COMMENT ON TABLE pack038_peak_shaving.ps_ratchet_alerts IS
    'Real-time alerts for demand approaching ratchet-setting levels enabling proactive intervention.';

COMMENT ON COLUMN pack038_peak_shaving.ps_ratchet_history.ratchet_was_binding IS 'Whether the ratchet demand exceeded actual demand, meaning the facility paid for more demand than it used.';
COMMENT ON COLUMN pack038_peak_shaving.ps_ratchet_history.excess_billed_kw IS 'The difference between ratchet billing demand and actual demand (kW paid for but not used).';
COMMENT ON COLUMN pack038_peak_shaving.ps_ratchet_impacts.total_ratchet_cost IS 'Total financial cost of this spike over the full ratchet period (months_affected * monthly_excess_charge).';
COMMENT ON COLUMN pack038_peak_shaving.ps_ratchet_impacts.cost_per_kw_spike IS 'Effective cost per kW of the spike over the ratchet period, showing the true cost of each incremental kW.';
COMMENT ON COLUMN pack038_peak_shaving.ps_spike_analysis.area_under_spike_kwh IS 'Energy content of the spike above baseline (kWh), representing the actual energy consumed during the spike.';
COMMENT ON COLUMN pack038_peak_shaving.ps_ratchet_alerts.headroom_kw IS 'Remaining kW headroom before current demand reaches the ratchet threshold.';
