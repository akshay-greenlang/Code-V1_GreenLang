-- =============================================================================
-- V290: PACK-037 Demand Response Pack - Event Management
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Event management tables for tracking DR events from notification through
-- execution to settlement. Covers events, event phases, load control
-- commands issued during events, event performance measurement, and
-- detailed event logs for audit and diagnostics.
--
-- Tables (5):
--   1. pack037_demand_response.dr_events
--   2. pack037_demand_response.dr_event_phases
--   3. pack037_demand_response.dr_load_control_commands
--   4. pack037_demand_response.dr_event_performance
--   5. pack037_demand_response.dr_event_logs
--
-- Previous: V289__pack037_demand_response_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_events
-- =============================================================================
-- Master event records for DR events. Each event represents a single
-- dispatch signal from an ISO/RTO, utility, or aggregator requiring
-- load curtailment during a defined window.

CREATE TABLE pack037_demand_response.dr_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enrollment_id           UUID            NOT NULL REFERENCES pack037_demand_response.dr_program_enrollment(id),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    dispatch_plan_id        UUID            REFERENCES pack037_demand_response.dr_dispatch_plans(id),
    tenant_id               UUID            NOT NULL,
    event_code              VARCHAR(100)    NOT NULL,
    event_type              VARCHAR(30)     NOT NULL,
    event_trigger           VARCHAR(50)     NOT NULL,
    event_status            VARCHAR(30)     NOT NULL DEFAULT 'NOTIFIED',
    notification_received   TIMESTAMPTZ     NOT NULL,
    event_start_scheduled   TIMESTAMPTZ     NOT NULL,
    event_end_scheduled     TIMESTAMPTZ     NOT NULL,
    event_start_actual      TIMESTAMPTZ,
    event_end_actual        TIMESTAMPTZ,
    lead_time_min           INTEGER,
    target_curtailment_kw   NUMERIC(12,4)   NOT NULL,
    achieved_curtailment_kw NUMERIC(12,4),
    performance_factor      NUMERIC(6,4),
    compliance_status       VARCHAR(20),
    opt_out_allowed         BOOLEAN         DEFAULT false,
    opted_out               BOOLEAN         DEFAULT false,
    opt_out_reason          TEXT,
    weather_temperature_f   NUMERIC(6,2),
    weather_condition       VARCHAR(50),
    grid_conditions         VARCHAR(50),
    lmp_price_at_dispatch   NUMERIC(10,4),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_ev_type CHECK (
        event_type IN (
            'CAPACITY', 'ENERGY', 'EMERGENCY', 'TEST', 'ECONOMIC',
            'RELIABILITY', 'ANCILLARY', 'VOLUNTARY'
        )
    ),
    CONSTRAINT chk_p037_ev_trigger CHECK (
        event_trigger IN (
            'ISO_DISPATCH', 'UTILITY_CALL', 'AGGREGATOR_SIGNAL',
            'PRICE_SIGNAL', 'FREQUENCY_DEVIATION', 'OPENADR_EVENT',
            'SCHEDULED_TEST', 'MANUAL_ACTIVATION', 'AUTO_ECONOMIC'
        )
    ),
    CONSTRAINT chk_p037_ev_status CHECK (
        event_status IN (
            'NOTIFIED', 'ACKNOWLEDGED', 'PREPARING', 'ACTIVE',
            'SUSTAINING', 'RAMPING_UP', 'COMPLETED', 'SETTLED',
            'CANCELLED', 'OPTED_OUT', 'FAILED'
        )
    ),
    CONSTRAINT chk_p037_ev_compliance CHECK (
        compliance_status IS NULL OR compliance_status IN (
            'COMPLIANT', 'PARTIAL', 'NON_COMPLIANT', 'EXCUSED', 'PENDING'
        )
    ),
    CONSTRAINT chk_p037_ev_target CHECK (
        target_curtailment_kw > 0
    ),
    CONSTRAINT chk_p037_ev_achieved CHECK (
        achieved_curtailment_kw IS NULL OR achieved_curtailment_kw >= 0
    ),
    CONSTRAINT chk_p037_ev_perf CHECK (
        performance_factor IS NULL OR (performance_factor >= 0 AND performance_factor <= 2.0)
    ),
    CONSTRAINT chk_p037_ev_schedule CHECK (
        event_end_scheduled > event_start_scheduled
    ),
    CONSTRAINT chk_p037_ev_grid CHECK (
        grid_conditions IS NULL OR grid_conditions IN (
            'NORMAL', 'WATCH', 'WARNING', 'EMERGENCY', 'CRITICAL'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_ev_enrollment      ON pack037_demand_response.dr_events(enrollment_id);
CREATE INDEX idx_p037_ev_facility        ON pack037_demand_response.dr_events(facility_profile_id);
CREATE INDEX idx_p037_ev_dispatch_plan   ON pack037_demand_response.dr_events(dispatch_plan_id);
CREATE INDEX idx_p037_ev_tenant          ON pack037_demand_response.dr_events(tenant_id);
CREATE INDEX idx_p037_ev_code            ON pack037_demand_response.dr_events(event_code);
CREATE INDEX idx_p037_ev_type            ON pack037_demand_response.dr_events(event_type);
CREATE INDEX idx_p037_ev_status          ON pack037_demand_response.dr_events(event_status);
CREATE INDEX idx_p037_ev_compliance      ON pack037_demand_response.dr_events(compliance_status);
CREATE INDEX idx_p037_ev_start_sched     ON pack037_demand_response.dr_events(event_start_scheduled DESC);
CREATE INDEX idx_p037_ev_notification    ON pack037_demand_response.dr_events(notification_received DESC);
CREATE INDEX idx_p037_ev_created         ON pack037_demand_response.dr_events(created_at DESC);
CREATE INDEX idx_p037_ev_metadata        ON pack037_demand_response.dr_events USING GIN(metadata);

-- Composite: facility + active events for real-time monitoring
CREATE INDEX idx_p037_ev_fac_active      ON pack037_demand_response.dr_events(facility_profile_id, event_start_scheduled DESC)
    WHERE event_status IN ('NOTIFIED', 'ACKNOWLEDGED', 'PREPARING', 'ACTIVE', 'SUSTAINING');

-- Composite: enrollment + date range for settlement queries
CREATE INDEX idx_p037_ev_enr_dates       ON pack037_demand_response.dr_events(enrollment_id, event_start_scheduled DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_ev_updated
    BEFORE UPDATE ON pack037_demand_response.dr_events
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack037_demand_response.dr_event_phases
-- =============================================================================
-- Tracks the lifecycle phases of each DR event from notification through
-- completion. Each phase records entry/exit times and conditions.

CREATE TABLE pack037_demand_response.dr_event_phases (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            NOT NULL REFERENCES pack037_demand_response.dr_events(id) ON DELETE CASCADE,
    phase_name              VARCHAR(30)     NOT NULL,
    phase_order             INTEGER         NOT NULL,
    phase_start             TIMESTAMPTZ     NOT NULL,
    phase_end               TIMESTAMPTZ,
    duration_seconds        INTEGER,
    entry_demand_kw         NUMERIC(12,4),
    exit_demand_kw          NUMERIC(12,4),
    target_demand_kw        NUMERIC(12,4),
    loads_curtailed         INTEGER         DEFAULT 0,
    actions_executed        INTEGER         DEFAULT 0,
    actions_failed          INTEGER         DEFAULT 0,
    phase_status            VARCHAR(20)     NOT NULL DEFAULT 'IN_PROGRESS',
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_ep_phase CHECK (
        phase_name IN (
            'NOTIFICATION', 'ACKNOWLEDGEMENT', 'PRECONDITIONING',
            'RAMP_DOWN', 'SUSTAINED_CURTAILMENT', 'RAMP_UP',
            'RECOVERY', 'REBOUND', 'SETTLEMENT'
        )
    ),
    CONSTRAINT chk_p037_ep_order CHECK (
        phase_order >= 1 AND phase_order <= 20
    ),
    CONSTRAINT chk_p037_ep_status CHECK (
        phase_status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'SKIPPED', 'FAILED')
    ),
    CONSTRAINT uq_p037_ep_event_phase UNIQUE (event_id, phase_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_ep_event           ON pack037_demand_response.dr_event_phases(event_id);
CREATE INDEX idx_p037_ep_phase           ON pack037_demand_response.dr_event_phases(phase_name);
CREATE INDEX idx_p037_ep_status          ON pack037_demand_response.dr_event_phases(phase_status);
CREATE INDEX idx_p037_ep_start           ON pack037_demand_response.dr_event_phases(phase_start DESC);

-- Composite: event + order for sequential phase lookup
CREATE INDEX idx_p037_ep_event_order     ON pack037_demand_response.dr_event_phases(event_id, phase_order ASC);

-- =============================================================================
-- Table 3: pack037_demand_response.dr_load_control_commands
-- =============================================================================
-- Individual load control commands issued during a DR event. Tracks the
-- command sent to each load, its execution result, and actual impact.

CREATE TABLE pack037_demand_response.dr_load_control_commands (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            NOT NULL REFERENCES pack037_demand_response.dr_events(id) ON DELETE CASCADE,
    curtailment_sequence_id UUID            REFERENCES pack037_demand_response.dr_curtailment_sequences(id),
    load_id                 UUID            NOT NULL REFERENCES pack037_demand_response.dr_load_inventory(id),
    command_type            VARCHAR(30)     NOT NULL,
    command_payload         JSONB           NOT NULL DEFAULT '{}',
    target_reduction_kw     NUMERIC(12,4)   NOT NULL,
    actual_reduction_kw     NUMERIC(12,4),
    pre_command_demand_kw   NUMERIC(12,4),
    post_command_demand_kw  NUMERIC(12,4),
    issued_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    acknowledged_at         TIMESTAMPTZ,
    executed_at             TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    response_time_seconds   INTEGER,
    command_status          VARCHAR(20)     NOT NULL DEFAULT 'ISSUED',
    failure_reason          TEXT,
    retry_count             INTEGER         DEFAULT 0,
    control_protocol        VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_lcc_type CHECK (
        command_type IN (
            'CURTAIL', 'SHED', 'RESTORE', 'SETPOINT_UP', 'SETPOINT_DOWN',
            'CYCLE_OFF', 'CYCLE_ON', 'REDUCE_SPEED', 'FULL_SPEED',
            'DISCONNECT', 'RECONNECT', 'PRECONDITION'
        )
    ),
    CONSTRAINT chk_p037_lcc_status CHECK (
        command_status IN (
            'ISSUED', 'ACKNOWLEDGED', 'EXECUTING', 'COMPLETED',
            'FAILED', 'TIMEOUT', 'CANCELLED', 'RETRYING'
        )
    ),
    CONSTRAINT chk_p037_lcc_target CHECK (
        target_reduction_kw >= 0
    ),
    CONSTRAINT chk_p037_lcc_actual CHECK (
        actual_reduction_kw IS NULL OR actual_reduction_kw >= 0
    ),
    CONSTRAINT chk_p037_lcc_retry CHECK (
        retry_count >= 0 AND retry_count <= 10
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_lcc_event          ON pack037_demand_response.dr_load_control_commands(event_id);
CREATE INDEX idx_p037_lcc_sequence       ON pack037_demand_response.dr_load_control_commands(curtailment_sequence_id);
CREATE INDEX idx_p037_lcc_load           ON pack037_demand_response.dr_load_control_commands(load_id);
CREATE INDEX idx_p037_lcc_type           ON pack037_demand_response.dr_load_control_commands(command_type);
CREATE INDEX idx_p037_lcc_status         ON pack037_demand_response.dr_load_control_commands(command_status);
CREATE INDEX idx_p037_lcc_issued         ON pack037_demand_response.dr_load_control_commands(issued_at DESC);
CREATE INDEX idx_p037_lcc_created        ON pack037_demand_response.dr_load_control_commands(created_at DESC);

-- Composite: event + pending commands for execution monitoring
CREATE INDEX idx_p037_lcc_event_pending  ON pack037_demand_response.dr_load_control_commands(event_id, issued_at)
    WHERE command_status IN ('ISSUED', 'ACKNOWLEDGED', 'EXECUTING', 'RETRYING');

-- =============================================================================
-- Table 4: pack037_demand_response.dr_event_performance
-- =============================================================================
-- Performance measurement results for each DR event comparing actual
-- load reduction against the Customer Baseline Load (CBL).

CREATE TABLE pack037_demand_response.dr_event_performance (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            NOT NULL REFERENCES pack037_demand_response.dr_events(id) ON DELETE CASCADE,
    baseline_calculation_id UUID            REFERENCES pack037_demand_response.dr_baseline_calculations(id),
    tenant_id               UUID            NOT NULL,
    measurement_period      VARCHAR(30)     NOT NULL DEFAULT 'EVENT_WINDOW',
    baseline_avg_kw         NUMERIC(12,4)   NOT NULL,
    actual_avg_kw           NUMERIC(12,4)   NOT NULL,
    curtailment_avg_kw      NUMERIC(12,4)   NOT NULL,
    baseline_peak_kw        NUMERIC(12,4),
    actual_min_kw           NUMERIC(12,4),
    max_curtailment_kw      NUMERIC(12,4),
    curtailment_mwh         NUMERIC(14,6),
    target_curtailment_kw   NUMERIC(12,4)   NOT NULL,
    performance_ratio       NUMERIC(6,4)    NOT NULL,
    compliance_flag         BOOLEAN         NOT NULL DEFAULT true,
    response_time_min       NUMERIC(8,2),
    sustain_duration_min    NUMERIC(8,2),
    rebound_peak_kw         NUMERIC(12,4),
    rebound_duration_min    NUMERIC(8,2),
    data_quality_score      NUMERIC(5,2),
    settlement_kw           NUMERIC(12,4),
    settlement_status       VARCHAR(20)     DEFAULT 'PENDING',
    verified_by             VARCHAR(255),
    verified_at             TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_ep2_period CHECK (
        measurement_period IN (
            'EVENT_WINDOW', 'SETTLEMENT_WINDOW', 'PEAK_HOUR',
            'FULL_DAY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p037_ep2_baseline CHECK (
        baseline_avg_kw >= 0
    ),
    CONSTRAINT chk_p037_ep2_actual CHECK (
        actual_avg_kw >= 0
    ),
    CONSTRAINT chk_p037_ep2_perf_ratio CHECK (
        performance_ratio >= 0
    ),
    CONSTRAINT chk_p037_ep2_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p037_ep2_settlement_status CHECK (
        settlement_status IS NULL OR settlement_status IN (
            'PENDING', 'PRELIMINARY', 'VERIFIED', 'DISPUTED', 'FINAL'
        )
    ),
    CONSTRAINT uq_p037_ep2_event UNIQUE (event_id, measurement_period)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_ep2_event          ON pack037_demand_response.dr_event_performance(event_id);
CREATE INDEX idx_p037_ep2_baseline       ON pack037_demand_response.dr_event_performance(baseline_calculation_id);
CREATE INDEX idx_p037_ep2_tenant         ON pack037_demand_response.dr_event_performance(tenant_id);
CREATE INDEX idx_p037_ep2_compliance     ON pack037_demand_response.dr_event_performance(compliance_flag);
CREATE INDEX idx_p037_ep2_perf_ratio     ON pack037_demand_response.dr_event_performance(performance_ratio DESC);
CREATE INDEX idx_p037_ep2_settlement     ON pack037_demand_response.dr_event_performance(settlement_status);
CREATE INDEX idx_p037_ep2_created        ON pack037_demand_response.dr_event_performance(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_ep2_updated
    BEFORE UPDATE ON pack037_demand_response.dr_event_performance
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack037_demand_response.dr_event_logs
-- =============================================================================
-- Detailed operational log entries for DR events capturing every action,
-- state change, communication, and measurement during event execution.

CREATE TABLE pack037_demand_response.dr_event_logs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            NOT NULL REFERENCES pack037_demand_response.dr_events(id) ON DELETE CASCADE,
    log_timestamp           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    log_level               VARCHAR(10)     NOT NULL DEFAULT 'INFO',
    log_category            VARCHAR(30)     NOT NULL,
    message                 TEXT            NOT NULL,
    actor                   VARCHAR(255),
    load_id                 UUID,
    demand_kw_at_log        NUMERIC(12,4),
    details                 JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_el_level CHECK (
        log_level IN ('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL')
    ),
    CONSTRAINT chk_p037_el_category CHECK (
        log_category IN (
            'NOTIFICATION', 'DISPATCH', 'COMMAND', 'MEASUREMENT',
            'STATUS_CHANGE', 'CONSTRAINT_VIOLATION', 'COMMUNICATION',
            'ERROR', 'OVERRIDE', 'SETTLEMENT', 'AUDIT'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_el_event           ON pack037_demand_response.dr_event_logs(event_id);
CREATE INDEX idx_p037_el_timestamp       ON pack037_demand_response.dr_event_logs(log_timestamp DESC);
CREATE INDEX idx_p037_el_level           ON pack037_demand_response.dr_event_logs(log_level);
CREATE INDEX idx_p037_el_category        ON pack037_demand_response.dr_event_logs(log_category);
CREATE INDEX idx_p037_el_load            ON pack037_demand_response.dr_event_logs(load_id);
CREATE INDEX idx_p037_el_details         ON pack037_demand_response.dr_event_logs USING GIN(details);

-- Composite: event + timestamp for chronological log review
CREATE INDEX idx_p037_el_event_ts        ON pack037_demand_response.dr_event_logs(event_id, log_timestamp ASC);

-- Composite: errors for monitoring dashboards
CREATE INDEX idx_p037_el_errors          ON pack037_demand_response.dr_event_logs(log_timestamp DESC)
    WHERE log_level IN ('ERROR', 'CRITICAL');

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_event_phases ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_load_control_commands ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_event_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_event_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p037_ev_tenant_isolation ON pack037_demand_response.dr_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_ev_service_bypass ON pack037_demand_response.dr_events
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_ep_service_bypass ON pack037_demand_response.dr_event_phases
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_lcc_service_bypass ON pack037_demand_response.dr_load_control_commands
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_ep2_tenant_isolation ON pack037_demand_response.dr_event_performance
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_ep2_service_bypass ON pack037_demand_response.dr_event_performance
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_el_service_bypass ON pack037_demand_response.dr_event_logs
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_events TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_event_phases TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_load_control_commands TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_event_performance TO PUBLIC;
GRANT SELECT, INSERT ON pack037_demand_response.dr_event_logs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_events IS
    'Master event records for DR events from ISO/RTO, utility, or aggregator dispatch signals with scheduled and actual windows.';
COMMENT ON TABLE pack037_demand_response.dr_event_phases IS
    'Lifecycle phases of each DR event from notification through recovery with entry/exit demand readings.';
COMMENT ON TABLE pack037_demand_response.dr_load_control_commands IS
    'Individual load control commands issued during DR events with execution result and actual impact measurement.';
COMMENT ON TABLE pack037_demand_response.dr_event_performance IS
    'Performance measurement comparing actual load reduction against the Customer Baseline Load for settlement.';
COMMENT ON TABLE pack037_demand_response.dr_event_logs IS
    'Detailed operational log entries capturing every action, state change, and measurement during event execution.';

COMMENT ON COLUMN pack037_demand_response.dr_events.event_type IS 'Event classification: CAPACITY, ENERGY, EMERGENCY, TEST, ECONOMIC, RELIABILITY, ANCILLARY, VOLUNTARY.';
COMMENT ON COLUMN pack037_demand_response.dr_events.event_trigger IS 'What triggered the event: ISO_DISPATCH, UTILITY_CALL, PRICE_SIGNAL, OPENADR_EVENT, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_events.event_status IS 'Event lifecycle: NOTIFIED, ACKNOWLEDGED, PREPARING, ACTIVE, SUSTAINING, RAMPING_UP, COMPLETED, SETTLED, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_events.performance_factor IS 'Ratio of achieved to target curtailment (1.0 = full delivery, 0.5 = 50% delivery).';
COMMENT ON COLUMN pack037_demand_response.dr_events.lmp_price_at_dispatch IS 'Locational Marginal Price (LMP) at the time of event dispatch.';
COMMENT ON COLUMN pack037_demand_response.dr_events.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_event_performance.performance_ratio IS 'Ratio of achieved curtailment to target (1.0 = 100% delivery).';
COMMENT ON COLUMN pack037_demand_response.dr_event_performance.rebound_peak_kw IS 'Peak demand during the post-event rebound period in kW.';
COMMENT ON COLUMN pack037_demand_response.dr_event_performance.settlement_kw IS 'Final settled curtailment kW used for payment calculation.';

COMMENT ON COLUMN pack037_demand_response.dr_load_control_commands.response_time_seconds IS 'Seconds between command issuance and confirmed execution.';
COMMENT ON COLUMN pack037_demand_response.dr_load_control_commands.retry_count IS 'Number of retry attempts for failed commands (max 10).';
