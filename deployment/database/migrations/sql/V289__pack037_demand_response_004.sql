-- =============================================================================
-- V289: PACK-037 Demand Response Pack - Dispatch Optimization
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Dispatch optimization tables for planning and executing load curtailment
-- sequences during DR events. Covers dispatch plans, curtailment sequences
-- (priority-ordered load shedding), dispatch constraints, and
-- preconditioning commands issued before events.
--
-- Tables (4):
--   1. pack037_demand_response.dr_dispatch_plans
--   2. pack037_demand_response.dr_curtailment_sequences
--   3. pack037_demand_response.dr_dispatch_constraints
--   4. pack037_demand_response.dr_preconditioning_commands
--
-- Previous: V288__pack037_demand_response_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_dispatch_plans
-- =============================================================================
-- Optimized dispatch plans that define how a facility will achieve its
-- curtailment target during a DR event. Each plan contains an ordered
-- set of curtailment sequences and constraint boundaries.

CREATE TABLE pack037_demand_response.dr_dispatch_plans (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    enrollment_id           UUID            REFERENCES pack037_demand_response.dr_program_enrollment(id),
    tenant_id               UUID            NOT NULL,
    plan_name               VARCHAR(255)    NOT NULL,
    plan_type               VARCHAR(30)     NOT NULL DEFAULT 'EVENT',
    target_curtailment_kw   NUMERIC(12,4)   NOT NULL,
    planned_curtailment_kw  NUMERIC(12,4)   NOT NULL,
    margin_kw               NUMERIC(12,4)   DEFAULT 0,
    optimization_objective  VARCHAR(50)     NOT NULL DEFAULT 'MIN_COST',
    season                  VARCHAR(20)     NOT NULL DEFAULT 'ALL_YEAR',
    day_type                VARCHAR(20)     NOT NULL DEFAULT 'WEEKDAY',
    event_window_start      TIME,
    event_window_end        TIME,
    max_duration_hours      NUMERIC(6,2)    NOT NULL DEFAULT 4,
    preconditioning_min     INTEGER         DEFAULT 0,
    ramp_down_strategy      VARCHAR(30)     NOT NULL DEFAULT 'SEQUENTIAL',
    ramp_up_strategy        VARCHAR(30)     NOT NULL DEFAULT 'REVERSE',
    plan_status             VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by             VARCHAR(255),
    approved_at             TIMESTAMPTZ,
    last_tested             DATE,
    test_result_kw          NUMERIC(12,4),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p037_dp_type CHECK (
        plan_type IN ('EVENT', 'TEST', 'EMERGENCY', 'SCHEDULED', 'ECONOMIC')
    ),
    CONSTRAINT chk_p037_dp_target CHECK (
        target_curtailment_kw > 0
    ),
    CONSTRAINT chk_p037_dp_planned CHECK (
        planned_curtailment_kw > 0
    ),
    CONSTRAINT chk_p037_dp_margin CHECK (
        margin_kw >= 0
    ),
    CONSTRAINT chk_p037_dp_objective CHECK (
        optimization_objective IN (
            'MIN_COST', 'MIN_COMFORT_IMPACT', 'MIN_PRODUCTION_IMPACT',
            'MAX_RELIABILITY', 'BALANCED', 'MIN_REBOUND'
        )
    ),
    CONSTRAINT chk_p037_dp_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_dp_day_type CHECK (
        day_type IN ('WEEKDAY', 'WEEKEND', 'HOLIDAY', 'ALL_DAYS')
    ),
    CONSTRAINT chk_p037_dp_ramp_down CHECK (
        ramp_down_strategy IN (
            'SEQUENTIAL', 'SIMULTANEOUS', 'STAGED', 'PRIORITY_WEIGHTED'
        )
    ),
    CONSTRAINT chk_p037_dp_ramp_up CHECK (
        ramp_up_strategy IN (
            'REVERSE', 'SIMULTANEOUS', 'STAGED', 'PRIORITY_WEIGHTED', 'DELAYED'
        )
    ),
    CONSTRAINT chk_p037_dp_status CHECK (
        plan_status IN ('DRAFT', 'REVIEWED', 'APPROVED', 'ACTIVE', 'ARCHIVED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p037_dp_duration CHECK (
        max_duration_hours > 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_dp_facility        ON pack037_demand_response.dr_dispatch_plans(facility_profile_id);
CREATE INDEX idx_p037_dp_enrollment      ON pack037_demand_response.dr_dispatch_plans(enrollment_id);
CREATE INDEX idx_p037_dp_tenant          ON pack037_demand_response.dr_dispatch_plans(tenant_id);
CREATE INDEX idx_p037_dp_type            ON pack037_demand_response.dr_dispatch_plans(plan_type);
CREATE INDEX idx_p037_dp_status          ON pack037_demand_response.dr_dispatch_plans(plan_status);
CREATE INDEX idx_p037_dp_objective       ON pack037_demand_response.dr_dispatch_plans(optimization_objective);
CREATE INDEX idx_p037_dp_season          ON pack037_demand_response.dr_dispatch_plans(season);
CREATE INDEX idx_p037_dp_target          ON pack037_demand_response.dr_dispatch_plans(target_curtailment_kw DESC);
CREATE INDEX idx_p037_dp_created         ON pack037_demand_response.dr_dispatch_plans(created_at DESC);
CREATE INDEX idx_p037_dp_metadata        ON pack037_demand_response.dr_dispatch_plans USING GIN(metadata);

-- Composite: active plans by facility for event dispatch
CREATE INDEX idx_p037_dp_fac_active      ON pack037_demand_response.dr_dispatch_plans(facility_profile_id, season)
    WHERE plan_status IN ('APPROVED', 'ACTIVE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_dp_updated
    BEFORE UPDATE ON pack037_demand_response.dr_dispatch_plans
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack037_demand_response.dr_curtailment_sequences
-- =============================================================================
-- Ordered sequence of load curtailment steps within a dispatch plan.
-- Each step specifies which load to curtail, the expected reduction,
-- timing, and conditions under which the step is executed.

CREATE TABLE pack037_demand_response.dr_curtailment_sequences (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    dispatch_plan_id        UUID            NOT NULL REFERENCES pack037_demand_response.dr_dispatch_plans(id) ON DELETE CASCADE,
    load_id                 UUID            NOT NULL REFERENCES pack037_demand_response.dr_load_inventory(id),
    sequence_order          INTEGER         NOT NULL,
    action_type             VARCHAR(30)     NOT NULL,
    target_reduction_kw     NUMERIC(12,4)   NOT NULL,
    expected_reduction_kw   NUMERIC(12,4)   NOT NULL,
    confidence_pct          NUMERIC(5,2)    DEFAULT 80,
    delay_from_start_min    INTEGER         NOT NULL DEFAULT 0,
    curtailment_duration_min INTEGER        NOT NULL,
    setpoint_change         NUMERIC(8,2),
    setpoint_unit           VARCHAR(20),
    execution_condition     TEXT,
    rollback_on_failure     BOOLEAN         DEFAULT true,
    fallback_load_id        UUID            REFERENCES pack037_demand_response.dr_load_inventory(id),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_cs_order CHECK (
        sequence_order >= 1
    ),
    CONSTRAINT chk_p037_cs_action CHECK (
        action_type IN (
            'CURTAIL', 'SHED', 'SHIFT', 'SETPOINT_ADJUST', 'CYCLE',
            'REDUCE_SPEED', 'STAGE_DOWN', 'DISCONNECT'
        )
    ),
    CONSTRAINT chk_p037_cs_target CHECK (
        target_reduction_kw >= 0
    ),
    CONSTRAINT chk_p037_cs_expected CHECK (
        expected_reduction_kw >= 0
    ),
    CONSTRAINT chk_p037_cs_confidence CHECK (
        confidence_pct IS NULL OR (confidence_pct >= 0 AND confidence_pct <= 100)
    ),
    CONSTRAINT chk_p037_cs_delay CHECK (
        delay_from_start_min >= 0
    ),
    CONSTRAINT chk_p037_cs_duration CHECK (
        curtailment_duration_min > 0
    ),
    CONSTRAINT uq_p037_cs_plan_order UNIQUE (dispatch_plan_id, sequence_order)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_cs_plan            ON pack037_demand_response.dr_curtailment_sequences(dispatch_plan_id);
CREATE INDEX idx_p037_cs_load            ON pack037_demand_response.dr_curtailment_sequences(load_id);
CREATE INDEX idx_p037_cs_action          ON pack037_demand_response.dr_curtailment_sequences(action_type);
CREATE INDEX idx_p037_cs_order           ON pack037_demand_response.dr_curtailment_sequences(sequence_order);
CREATE INDEX idx_p037_cs_fallback        ON pack037_demand_response.dr_curtailment_sequences(fallback_load_id);

-- Composite: plan + order for sequential dispatch execution
CREATE INDEX idx_p037_cs_plan_seq        ON pack037_demand_response.dr_curtailment_sequences(dispatch_plan_id, sequence_order ASC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_cs_updated
    BEFORE UPDATE ON pack037_demand_response.dr_curtailment_sequences
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack037_demand_response.dr_dispatch_constraints
-- =============================================================================
-- Constraints on dispatch plans that limit when and how loads can be
-- curtailed. Includes temporal, environmental, comfort, production,
-- and regulatory constraints.

CREATE TABLE pack037_demand_response.dr_dispatch_constraints (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    dispatch_plan_id        UUID            NOT NULL REFERENCES pack037_demand_response.dr_dispatch_plans(id) ON DELETE CASCADE,
    load_id                 UUID            REFERENCES pack037_demand_response.dr_load_inventory(id),
    constraint_type         VARCHAR(50)     NOT NULL,
    constraint_name         VARCHAR(255)    NOT NULL,
    constraint_description  TEXT,
    parameter_name          VARCHAR(100)    NOT NULL,
    operator                VARCHAR(10)     NOT NULL,
    threshold_value         NUMERIC(14,4)   NOT NULL,
    threshold_unit          VARCHAR(30),
    violation_action        VARCHAR(30)     NOT NULL DEFAULT 'BLOCK',
    is_hard_constraint      BOOLEAN         NOT NULL DEFAULT true,
    priority                INTEGER         DEFAULT 100,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_dc_type CHECK (
        constraint_type IN (
            'TEMPORAL', 'TEMPERATURE', 'HUMIDITY', 'OCCUPANCY',
            'PRODUCTION', 'SAFETY', 'REGULATORY', 'EQUIPMENT',
            'DEMAND_MINIMUM', 'DURATION', 'FREQUENCY', 'COMFORT'
        )
    ),
    CONSTRAINT chk_p037_dc_operator CHECK (
        operator IN ('GT', 'GTE', 'LT', 'LTE', 'EQ', 'NEQ', 'BETWEEN')
    ),
    CONSTRAINT chk_p037_dc_violation CHECK (
        violation_action IN ('BLOCK', 'WARN', 'REDUCE', 'SKIP_LOAD', 'OVERRIDE_WITH_APPROVAL')
    ),
    CONSTRAINT chk_p037_dc_priority CHECK (
        priority >= 1 AND priority <= 999
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_dc_plan            ON pack037_demand_response.dr_dispatch_constraints(dispatch_plan_id);
CREATE INDEX idx_p037_dc_load            ON pack037_demand_response.dr_dispatch_constraints(load_id);
CREATE INDEX idx_p037_dc_type            ON pack037_demand_response.dr_dispatch_constraints(constraint_type);
CREATE INDEX idx_p037_dc_hard            ON pack037_demand_response.dr_dispatch_constraints(is_hard_constraint);
CREATE INDEX idx_p037_dc_priority        ON pack037_demand_response.dr_dispatch_constraints(priority);

-- =============================================================================
-- Table 4: pack037_demand_response.dr_preconditioning_commands
-- =============================================================================
-- Preconditioning actions executed before a DR event to build thermal
-- mass, pre-cool/pre-heat spaces, charge storage, or prepare loads
-- for curtailment. Reduces rebound and comfort impact.

CREATE TABLE pack037_demand_response.dr_preconditioning_commands (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    dispatch_plan_id        UUID            NOT NULL REFERENCES pack037_demand_response.dr_dispatch_plans(id) ON DELETE CASCADE,
    load_id                 UUID            NOT NULL REFERENCES pack037_demand_response.dr_load_inventory(id),
    command_type            VARCHAR(50)     NOT NULL,
    command_description     TEXT            NOT NULL,
    start_before_event_min  INTEGER         NOT NULL,
    duration_min            INTEGER         NOT NULL,
    setpoint_change         NUMERIC(8,2),
    setpoint_unit           VARCHAR(20),
    energy_cost_kwh         NUMERIC(12,4),
    expected_benefit_kw     NUMERIC(12,4),
    execution_status        VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    executed_at             TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    result_notes            TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_pc_type CHECK (
        command_type IN (
            'PRE_COOL', 'PRE_HEAT', 'CHARGE_BATTERY', 'CHARGE_THERMAL',
            'FILL_TANK', 'RAMP_PRODUCTION', 'ADVANCE_SCHEDULE',
            'BOOST_VENTILATION', 'INCREASE_LIGHTING'
        )
    ),
    CONSTRAINT chk_p037_pc_start CHECK (
        start_before_event_min > 0
    ),
    CONSTRAINT chk_p037_pc_duration CHECK (
        duration_min > 0
    ),
    CONSTRAINT chk_p037_pc_status CHECK (
        execution_status IN ('PENDING', 'EXECUTING', 'COMPLETED', 'FAILED', 'SKIPPED', 'CANCELLED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_pc_plan            ON pack037_demand_response.dr_preconditioning_commands(dispatch_plan_id);
CREATE INDEX idx_p037_pc_load            ON pack037_demand_response.dr_preconditioning_commands(load_id);
CREATE INDEX idx_p037_pc_type            ON pack037_demand_response.dr_preconditioning_commands(command_type);
CREATE INDEX idx_p037_pc_status          ON pack037_demand_response.dr_preconditioning_commands(execution_status);
CREATE INDEX idx_p037_pc_created         ON pack037_demand_response.dr_preconditioning_commands(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_pc_updated
    BEFORE UPDATE ON pack037_demand_response.dr_preconditioning_commands
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_dispatch_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_curtailment_sequences ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_dispatch_constraints ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_preconditioning_commands ENABLE ROW LEVEL SECURITY;

CREATE POLICY p037_dp_tenant_isolation ON pack037_demand_response.dr_dispatch_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_dp_service_bypass ON pack037_demand_response.dr_dispatch_plans
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_cs_service_bypass ON pack037_demand_response.dr_curtailment_sequences
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_dc_service_bypass ON pack037_demand_response.dr_dispatch_constraints
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_pc_service_bypass ON pack037_demand_response.dr_preconditioning_commands
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_dispatch_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_curtailment_sequences TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_dispatch_constraints TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_preconditioning_commands TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_dispatch_plans IS
    'Optimized dispatch plans defining how a facility achieves curtailment targets during DR events with ordered load sequences.';
COMMENT ON TABLE pack037_demand_response.dr_curtailment_sequences IS
    'Ordered sequence of load curtailment steps within a dispatch plan specifying target reduction, timing, and rollback behaviour.';
COMMENT ON TABLE pack037_demand_response.dr_dispatch_constraints IS
    'Constraints limiting when and how loads can be curtailed including temporal, environmental, comfort, and safety boundaries.';
COMMENT ON TABLE pack037_demand_response.dr_preconditioning_commands IS
    'Pre-event commands to build thermal mass, charge storage, or prepare loads for curtailment to reduce rebound and comfort impact.';

COMMENT ON COLUMN pack037_demand_response.dr_dispatch_plans.optimization_objective IS 'Optimization goal: MIN_COST, MIN_COMFORT_IMPACT, MIN_PRODUCTION_IMPACT, MAX_RELIABILITY, BALANCED, MIN_REBOUND.';
COMMENT ON COLUMN pack037_demand_response.dr_dispatch_plans.ramp_down_strategy IS 'Load shedding strategy: SEQUENTIAL (one-by-one), SIMULTANEOUS, STAGED, PRIORITY_WEIGHTED.';
COMMENT ON COLUMN pack037_demand_response.dr_dispatch_plans.margin_kw IS 'Safety margin in kW above the target curtailment to account for curtailment uncertainty.';
COMMENT ON COLUMN pack037_demand_response.dr_dispatch_plans.preconditioning_min IS 'Minutes of preconditioning before the event window starts.';
COMMENT ON COLUMN pack037_demand_response.dr_dispatch_plans.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_curtailment_sequences.sequence_order IS 'Execution order within the dispatch plan (1=first to curtail).';
COMMENT ON COLUMN pack037_demand_response.dr_curtailment_sequences.action_type IS 'Curtailment action: CURTAIL, SHED, SHIFT, SETPOINT_ADJUST, CYCLE, REDUCE_SPEED, STAGE_DOWN, DISCONNECT.';
COMMENT ON COLUMN pack037_demand_response.dr_curtailment_sequences.delay_from_start_min IS 'Minutes after event start before executing this curtailment step.';
COMMENT ON COLUMN pack037_demand_response.dr_curtailment_sequences.rollback_on_failure IS 'Whether to rollback to previous state if this curtailment step fails.';
COMMENT ON COLUMN pack037_demand_response.dr_curtailment_sequences.fallback_load_id IS 'Alternative load to curtail if the primary load is unavailable.';

COMMENT ON COLUMN pack037_demand_response.dr_dispatch_constraints.is_hard_constraint IS 'Hard constraints block curtailment; soft constraints generate warnings.';
COMMENT ON COLUMN pack037_demand_response.dr_dispatch_constraints.violation_action IS 'Action when constraint is violated: BLOCK, WARN, REDUCE, SKIP_LOAD, OVERRIDE_WITH_APPROVAL.';

COMMENT ON COLUMN pack037_demand_response.dr_preconditioning_commands.command_type IS 'Preconditioning type: PRE_COOL, PRE_HEAT, CHARGE_BATTERY, CHARGE_THERMAL, FILL_TANK, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_preconditioning_commands.start_before_event_min IS 'Minutes before the event start when preconditioning should begin.';
COMMENT ON COLUMN pack037_demand_response.dr_preconditioning_commands.expected_benefit_kw IS 'Expected additional curtailment capacity gained from preconditioning in kW.';
