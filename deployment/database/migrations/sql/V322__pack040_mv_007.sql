-- =============================================================================
-- V322: PACK-040 M&V Pack - Metering Plans
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for M&V metering plan management including metering plans,
-- meter assignments, calibration records, sampling protocols, and data
-- quality logs. These tables track the physical measurement infrastructure
-- required for each IPMVP option.
--
-- Tables (5):
--   1. pack040_mv.mv_metering_plans
--   2. pack040_mv.mv_meter_assignments
--   3. pack040_mv.mv_calibration_records
--   4. pack040_mv.mv_sampling_protocols
--   5. pack040_mv.mv_data_quality_logs
--
-- Previous: V321__pack040_mv_006.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_metering_plans
-- =============================================================================
-- M&V metering plans defining the measurement strategy for each project.
-- Specifies which meters are used, measurement frequency, calibration
-- requirements, data collection procedures, and gap handling methods.

CREATE TABLE pack040_mv.mv_metering_plans (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    plan_name                   VARCHAR(255)    NOT NULL,
    plan_version                INTEGER         NOT NULL DEFAULT 1,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    plan_status                 VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    -- Plan scope
    ipmvp_option                VARCHAR(10)     NOT NULL DEFAULT 'OPTION_C',
    measurement_scope           VARCHAR(50)     NOT NULL DEFAULT 'WHOLE_FACILITY',
    energy_types_covered        VARCHAR(50)[]   NOT NULL DEFAULT '{ELECTRICITY}',
    -- Measurement strategy
    data_collection_method      VARCHAR(50)     NOT NULL DEFAULT 'CONTINUOUS',
    collection_frequency        VARCHAR(30)     NOT NULL DEFAULT '15_MINUTE',
    measurement_duration_months INTEGER,
    start_date                  DATE,
    end_date                    DATE,
    -- Meter requirements
    total_meters_required       INTEGER         NOT NULL DEFAULT 1,
    meters_installed            INTEGER         NOT NULL DEFAULT 0,
    meters_commissioned         INTEGER         NOT NULL DEFAULT 0,
    required_accuracy_class     VARCHAR(20)     NOT NULL DEFAULT 'CLASS_1',
    -- Data management
    data_storage_method         VARCHAR(50)     NOT NULL DEFAULT 'AUTOMATED',
    gap_filling_method          VARCHAR(50)     NOT NULL DEFAULT 'LINEAR_INTERPOLATION',
    max_acceptable_gap_hours    INTEGER         NOT NULL DEFAULT 24,
    min_data_completeness_pct   NUMERIC(5,2)    NOT NULL DEFAULT 90.0,
    -- Calibration requirements
    calibration_frequency_months INTEGER        NOT NULL DEFAULT 12,
    calibration_standard        VARCHAR(100)    DEFAULT 'ANSI C12.20',
    -- Quality assurance
    qa_check_frequency          VARCHAR(30)     NOT NULL DEFAULT 'WEEKLY',
    validation_rules            JSONB           DEFAULT '[]',
    -- Budget
    estimated_metering_cost     NUMERIC(18,2),
    estimated_annual_data_cost  NUMERIC(18,2),
    currency_code               VARCHAR(3)      NOT NULL DEFAULT 'USD',
    -- Approval
    prepared_by                 VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_mp_status CHECK (
        plan_status IN (
            'DRAFT', 'REVIEWED', 'APPROVED', 'ACTIVE',
            'COMPLETED', 'SUPERSEDED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p040_mp_option CHECK (
        ipmvp_option IN ('OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D')
    ),
    CONSTRAINT chk_p040_mp_scope CHECK (
        measurement_scope IN (
            'WHOLE_FACILITY', 'SYSTEM_LEVEL', 'COMPONENT_LEVEL',
            'SUB_METERED', 'SPOT_MEASUREMENT', 'SIMULATED'
        )
    ),
    CONSTRAINT chk_p040_mp_collection CHECK (
        data_collection_method IN (
            'CONTINUOUS', 'SHORT_TERM', 'SPOT_MEASUREMENT',
            'UTILITY_BILLS', 'MANUAL_READ', 'MIXED'
        )
    ),
    CONSTRAINT chk_p040_mp_frequency CHECK (
        collection_frequency IN (
            '1_MINUTE', '5_MINUTE', '15_MINUTE', '30_MINUTE',
            'HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'ON_DEMAND'
        )
    ),
    CONSTRAINT chk_p040_mp_accuracy CHECK (
        required_accuracy_class IN (
            'CLASS_0_1', 'CLASS_0_2', 'CLASS_0_5', 'CLASS_1',
            'CLASS_2', 'CLASS_3', 'REVENUE_GRADE', 'MONITORING_GRADE'
        )
    ),
    CONSTRAINT chk_p040_mp_storage CHECK (
        data_storage_method IN (
            'AUTOMATED', 'MANUAL_ENTRY', 'UTILITY_DATA', 'SCADA',
            'BMS', 'IOT_PLATFORM', 'CLOUD', 'LOCAL'
        )
    ),
    CONSTRAINT chk_p040_mp_gap_fill CHECK (
        gap_filling_method IN (
            'LINEAR_INTERPOLATION', 'REGRESSION_PREDICT', 'HISTORICAL_AVERAGE',
            'DEGREE_DAY_SCALING', 'NO_FILL', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p040_mp_qa_freq CHECK (
        qa_check_frequency IN (
            'DAILY', 'WEEKLY', 'BIWEEKLY', 'MONTHLY', 'QUARTERLY'
        )
    ),
    CONSTRAINT chk_p040_mp_completeness CHECK (
        min_data_completeness_pct >= 50 AND min_data_completeness_pct <= 100
    ),
    CONSTRAINT chk_p040_mp_gap_hours CHECK (
        max_acceptable_gap_hours >= 1 AND max_acceptable_gap_hours <= 720
    ),
    CONSTRAINT chk_p040_mp_cal_freq CHECK (
        calibration_frequency_months >= 1 AND calibration_frequency_months <= 120
    ),
    CONSTRAINT chk_p040_mp_meters CHECK (
        total_meters_required >= 0 AND meters_installed >= 0 AND
        meters_commissioned >= 0 AND meters_commissioned <= meters_installed
    ),
    CONSTRAINT chk_p040_mp_version CHECK (
        plan_version >= 1
    ),
    CONSTRAINT chk_p040_mp_dates CHECK (
        start_date IS NULL OR end_date IS NULL OR start_date <= end_date
    ),
    CONSTRAINT uq_p040_mp_project_version UNIQUE (project_id, plan_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_mp_tenant            ON pack040_mv.mv_metering_plans(tenant_id);
CREATE INDEX idx_p040_mp_project           ON pack040_mv.mv_metering_plans(project_id);
CREATE INDEX idx_p040_mp_status            ON pack040_mv.mv_metering_plans(plan_status);
CREATE INDEX idx_p040_mp_option            ON pack040_mv.mv_metering_plans(ipmvp_option);
CREATE INDEX idx_p040_mp_current           ON pack040_mv.mv_metering_plans(is_current) WHERE is_current = true;
CREATE INDEX idx_p040_mp_created           ON pack040_mv.mv_metering_plans(created_at DESC);

-- Composite: project + current plan
CREATE INDEX idx_p040_mp_project_current   ON pack040_mv.mv_metering_plans(project_id, plan_version DESC)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_mp_updated
    BEFORE UPDATE ON pack040_mv.mv_metering_plans
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_meter_assignments
-- =============================================================================
-- Assignment of physical meters to M&V metering plans. Maps each meter to
-- its role within the M&V measurement boundary, specifying what is measured,
-- the measurement point, and any corrections or scaling applied.

CREATE TABLE pack040_mv.mv_meter_assignments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    metering_plan_id            UUID            NOT NULL REFERENCES pack040_mv.mv_metering_plans(id) ON DELETE CASCADE,
    meter_id                    UUID,
    pack039_meter_id            UUID,
    -- Meter identification
    meter_name                  VARCHAR(255)    NOT NULL,
    meter_serial_number         VARCHAR(100),
    meter_type                  VARCHAR(50)     NOT NULL DEFAULT 'INTERVAL',
    -- Assignment details
    assignment_role             VARCHAR(50)     NOT NULL DEFAULT 'PRIMARY',
    measurement_point           VARCHAR(255)    NOT NULL,
    measured_parameter          VARCHAR(50)     NOT NULL DEFAULT 'ENERGY_CONSUMPTION',
    measurement_unit            VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    -- Location
    building_name               VARCHAR(255),
    panel_id                    VARCHAR(100),
    circuit_description         TEXT,
    -- Specifications
    accuracy_class              VARCHAR(20)     NOT NULL DEFAULT 'CLASS_1',
    ct_ratio                    NUMERIC(10,2),
    pt_ratio                    NUMERIC(10,2),
    scaling_factor              NUMERIC(15,6)   NOT NULL DEFAULT 1.0,
    interval_minutes            INTEGER         NOT NULL DEFAULT 15,
    -- Status
    assignment_status           VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    installation_date           DATE,
    commissioning_date          DATE,
    decommission_date           DATE,
    last_reading_date           TIMESTAMPTZ,
    -- Data quality
    data_quality_score          NUMERIC(5,2),
    completeness_pct            NUMERIC(5,2),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_ma_role CHECK (
        assignment_role IN (
            'PRIMARY', 'BACKUP', 'CHECK_METER', 'SUB_METER',
            'GENERATION', 'REFERENCE', 'TEMPORARY'
        )
    ),
    CONSTRAINT chk_p040_ma_measured CHECK (
        measured_parameter IN (
            'ENERGY_CONSUMPTION', 'DEMAND_KW', 'POWER_FACTOR',
            'TEMPERATURE', 'FLOW_RATE', 'PRESSURE', 'HUMIDITY',
            'RUNTIME_HOURS', 'SPEED_RPM', 'AMPERAGE', 'VOLTAGE',
            'PRODUCTION_VOLUME', 'OCCUPANCY'
        )
    ),
    CONSTRAINT chk_p040_ma_accuracy CHECK (
        accuracy_class IN (
            'CLASS_0_1', 'CLASS_0_2', 'CLASS_0_5', 'CLASS_1',
            'CLASS_2', 'CLASS_3', 'REVENUE_GRADE', 'MONITORING_GRADE',
            'INDICATIVE'
        )
    ),
    CONSTRAINT chk_p040_ma_status CHECK (
        assignment_status IN (
            'PLANNED', 'PROCURED', 'INSTALLED', 'COMMISSIONED',
            'ACTIVE', 'MAINTENANCE', 'DECOMMISSIONED', 'FAILED'
        )
    ),
    CONSTRAINT chk_p040_ma_interval CHECK (
        interval_minutes IN (1, 5, 10, 15, 30, 60)
    ),
    CONSTRAINT chk_p040_ma_scaling CHECK (
        scaling_factor > 0
    ),
    CONSTRAINT chk_p040_ma_quality CHECK (
        data_quality_score IS NULL OR
        (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p040_ma_completeness CHECK (
        completeness_pct IS NULL OR
        (completeness_pct >= 0 AND completeness_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_ma_tenant            ON pack040_mv.mv_meter_assignments(tenant_id);
CREATE INDEX idx_p040_ma_plan              ON pack040_mv.mv_meter_assignments(metering_plan_id);
CREATE INDEX idx_p040_ma_meter             ON pack040_mv.mv_meter_assignments(meter_id);
CREATE INDEX idx_p040_ma_p039_meter        ON pack040_mv.mv_meter_assignments(pack039_meter_id);
CREATE INDEX idx_p040_ma_role              ON pack040_mv.mv_meter_assignments(assignment_role);
CREATE INDEX idx_p040_ma_status            ON pack040_mv.mv_meter_assignments(assignment_status);
CREATE INDEX idx_p040_ma_accuracy          ON pack040_mv.mv_meter_assignments(accuracy_class);
CREATE INDEX idx_p040_ma_created           ON pack040_mv.mv_meter_assignments(created_at DESC);

-- Composite: plan + active meters
CREATE INDEX idx_p040_ma_plan_active       ON pack040_mv.mv_meter_assignments(metering_plan_id, assignment_role)
    WHERE assignment_status IN ('COMMISSIONED', 'ACTIVE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_ma_updated
    BEFORE UPDATE ON pack040_mv.mv_meter_assignments
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_calibration_records
-- =============================================================================
-- Calibration records for M&V meters tracking calibration history, accuracy
-- verification, drift analysis, and compliance with calibration standards.
-- Critical for demonstrating measurement uncertainty values used in FSU.

CREATE TABLE pack040_mv.mv_calibration_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    meter_assignment_id         UUID            NOT NULL REFERENCES pack040_mv.mv_meter_assignments(id) ON DELETE CASCADE,
    calibration_date            DATE            NOT NULL,
    calibration_type            VARCHAR(30)     NOT NULL DEFAULT 'ROUTINE',
    calibration_standard        VARCHAR(100)    DEFAULT 'ANSI C12.20',
    calibration_lab             VARCHAR(255),
    technician_name             VARCHAR(255),
    certificate_number          VARCHAR(100),
    -- Results
    pre_cal_accuracy_pct        NUMERIC(8,4),
    post_cal_accuracy_pct       NUMERIC(8,4),
    drift_pct                   NUMERIC(8,4),
    pass_fail                   VARCHAR(10)     NOT NULL DEFAULT 'PASS',
    -- Test points
    test_point_1_pct_load       NUMERIC(5,2),
    test_point_1_error_pct      NUMERIC(8,4),
    test_point_2_pct_load       NUMERIC(5,2),
    test_point_2_error_pct      NUMERIC(8,4),
    test_point_3_pct_load       NUMERIC(5,2),
    test_point_3_error_pct      NUMERIC(8,4),
    additional_test_points      JSONB           DEFAULT '[]',
    -- CT/PT verification
    ct_ratio_verified           BOOLEAN         DEFAULT false,
    pt_ratio_verified           BOOLEAN         DEFAULT false,
    ct_error_pct                NUMERIC(8,4),
    pt_error_pct                NUMERIC(8,4),
    -- Environment
    ambient_temp_c              NUMERIC(6,2),
    ambient_humidity_pct        NUMERIC(5,2),
    -- Next calibration
    next_due_date               DATE,
    calibration_interval_months INTEGER         DEFAULT 12,
    -- Adjustments
    adjustment_made             BOOLEAN         NOT NULL DEFAULT false,
    adjustment_description      TEXT,
    -- Documentation
    certificate_document_id     UUID,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_cal_type CHECK (
        calibration_type IN (
            'ROUTINE', 'INITIAL', 'POST_REPAIR', 'VERIFICATION',
            'RECALIBRATION', 'COMPLIANCE', 'SPOT_CHECK'
        )
    ),
    CONSTRAINT chk_p040_cal_pass_fail CHECK (
        pass_fail IN ('PASS', 'FAIL', 'CONDITIONAL', 'DEFERRED')
    ),
    CONSTRAINT chk_p040_cal_drift CHECK (
        drift_pct IS NULL OR (drift_pct >= -100 AND drift_pct <= 100)
    ),
    CONSTRAINT chk_p040_cal_pre_acc CHECK (
        pre_cal_accuracy_pct IS NULL OR
        (pre_cal_accuracy_pct >= -100 AND pre_cal_accuracy_pct <= 100)
    ),
    CONSTRAINT chk_p040_cal_post_acc CHECK (
        post_cal_accuracy_pct IS NULL OR
        (post_cal_accuracy_pct >= -100 AND post_cal_accuracy_pct <= 100)
    ),
    CONSTRAINT chk_p040_cal_humidity CHECK (
        ambient_humidity_pct IS NULL OR
        (ambient_humidity_pct >= 0 AND ambient_humidity_pct <= 100)
    ),
    CONSTRAINT chk_p040_cal_interval CHECK (
        calibration_interval_months IS NULL OR
        (calibration_interval_months >= 1 AND calibration_interval_months <= 120)
    ),
    CONSTRAINT chk_p040_cal_next_due CHECK (
        next_due_date IS NULL OR calibration_date <= next_due_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_cal_tenant           ON pack040_mv.mv_calibration_records(tenant_id);
CREATE INDEX idx_p040_cal_meter            ON pack040_mv.mv_calibration_records(meter_assignment_id);
CREATE INDEX idx_p040_cal_date             ON pack040_mv.mv_calibration_records(calibration_date DESC);
CREATE INDEX idx_p040_cal_type             ON pack040_mv.mv_calibration_records(calibration_type);
CREATE INDEX idx_p040_cal_pass_fail        ON pack040_mv.mv_calibration_records(pass_fail);
CREATE INDEX idx_p040_cal_next_due         ON pack040_mv.mv_calibration_records(next_due_date);
CREATE INDEX idx_p040_cal_created          ON pack040_mv.mv_calibration_records(created_at DESC);

-- Composite: meter + upcoming calibrations
CREATE INDEX idx_p040_cal_meter_upcoming   ON pack040_mv.mv_calibration_records(meter_assignment_id, next_due_date)
    WHERE pass_fail = 'PASS';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_cal_updated
    BEFORE UPDATE ON pack040_mv.mv_calibration_records
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_sampling_protocols
-- =============================================================================
-- Sampling protocol definitions for IPMVP Option A where a sample of units
-- is measured to represent the full population. Defines sample design,
-- required sample size, stratification, and precision targets.

CREATE TABLE pack040_mv.mv_sampling_protocols (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    metering_plan_id            UUID            REFERENCES pack040_mv.mv_metering_plans(id) ON DELETE SET NULL,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    protocol_name               VARCHAR(255)    NOT NULL,
    -- Population
    population_description      TEXT            NOT NULL,
    population_size             INTEGER         NOT NULL,
    population_homogeneity      VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    -- Sampling design
    sampling_method             VARCHAR(50)     NOT NULL DEFAULT 'SIMPLE_RANDOM',
    confidence_level_pct        NUMERIC(5,2)    NOT NULL DEFAULT 90.0,
    target_precision_pct        NUMERIC(8,4)    NOT NULL DEFAULT 10.0,
    estimated_cv                NUMERIC(8,4),
    -- Required sample size
    calculated_sample_size      INTEGER         NOT NULL,
    actual_sample_size          INTEGER,
    contingency_samples         INTEGER         DEFAULT 0,
    -- Stratification
    is_stratified               BOOLEAN         NOT NULL DEFAULT false,
    num_strata                  INTEGER,
    stratification_variable     VARCHAR(100),
    strata_definitions          JSONB           DEFAULT '[]',
    -- Measurement protocol
    measurement_type            VARCHAR(50)     NOT NULL DEFAULT 'SPOT_MEASUREMENT',
    measurement_duration        VARCHAR(50),
    measurement_frequency       VARCHAR(50),
    -- Equipment required
    equipment_list              JSONB           DEFAULT '[]',
    estimated_field_hours       NUMERIC(8,2),
    estimated_cost              NUMERIC(18,2),
    -- Status
    protocol_status             VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_spr_homogeneity CHECK (
        population_homogeneity IN ('LOW', 'MODERATE', 'HIGH')
    ),
    CONSTRAINT chk_p040_spr_method CHECK (
        sampling_method IN (
            'SIMPLE_RANDOM', 'STRATIFIED', 'SYSTEMATIC', 'CLUSTER',
            'CENSUS', 'PURPOSIVE'
        )
    ),
    CONSTRAINT chk_p040_spr_confidence CHECK (
        confidence_level_pct >= 50 AND confidence_level_pct <= 99.9
    ),
    CONSTRAINT chk_p040_spr_precision CHECK (
        target_precision_pct > 0 AND target_precision_pct <= 100
    ),
    CONSTRAINT chk_p040_spr_population CHECK (
        population_size >= 1
    ),
    CONSTRAINT chk_p040_spr_sample CHECK (
        calculated_sample_size >= 1 AND calculated_sample_size <= population_size
    ),
    CONSTRAINT chk_p040_spr_actual CHECK (
        actual_sample_size IS NULL OR
        (actual_sample_size >= 1 AND actual_sample_size <= population_size)
    ),
    CONSTRAINT chk_p040_spr_meas_type CHECK (
        measurement_type IN (
            'SPOT_MEASUREMENT', 'SHORT_TERM', 'CONTINUOUS',
            'NAMEPLATE_SURVEY', 'VISUAL_INSPECTION'
        )
    ),
    CONSTRAINT chk_p040_spr_status CHECK (
        protocol_status IN (
            'DRAFT', 'REVIEWED', 'APPROVED', 'ACTIVE', 'COMPLETED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p040_spr_cv CHECK (
        estimated_cv IS NULL OR estimated_cv >= 0
    ),
    CONSTRAINT chk_p040_spr_strata CHECK (
        num_strata IS NULL OR num_strata >= 2
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_spr_tenant           ON pack040_mv.mv_sampling_protocols(tenant_id);
CREATE INDEX idx_p040_spr_project          ON pack040_mv.mv_sampling_protocols(project_id);
CREATE INDEX idx_p040_spr_plan             ON pack040_mv.mv_sampling_protocols(metering_plan_id);
CREATE INDEX idx_p040_spr_ecm              ON pack040_mv.mv_sampling_protocols(ecm_id);
CREATE INDEX idx_p040_spr_method           ON pack040_mv.mv_sampling_protocols(sampling_method);
CREATE INDEX idx_p040_spr_status           ON pack040_mv.mv_sampling_protocols(protocol_status);
CREATE INDEX idx_p040_spr_created          ON pack040_mv.mv_sampling_protocols(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_spr_updated
    BEFORE UPDATE ON pack040_mv.mv_sampling_protocols
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_data_quality_logs
-- =============================================================================
-- Data quality assessment logs for M&V meter data. Tracks completeness,
-- accuracy, consistency, and timeliness of data from each metering point
-- on a periodic basis for M&V data quality assurance.

CREATE TABLE pack040_mv.mv_data_quality_logs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    meter_assignment_id         UUID            NOT NULL REFERENCES pack040_mv.mv_meter_assignments(id) ON DELETE CASCADE,
    assessment_date             DATE            NOT NULL,
    assessment_period_start     DATE            NOT NULL,
    assessment_period_end       DATE            NOT NULL,
    -- Completeness
    expected_readings           INTEGER         NOT NULL,
    actual_readings             INTEGER         NOT NULL,
    completeness_pct            NUMERIC(5,2)    NOT NULL,
    gap_count                   INTEGER         NOT NULL DEFAULT 0,
    max_gap_hours               NUMERIC(8,2),
    gaps_filled_count           INTEGER         NOT NULL DEFAULT 0,
    gap_fill_method             VARCHAR(50),
    -- Accuracy
    suspect_readings_count      INTEGER         NOT NULL DEFAULT 0,
    corrected_readings_count    INTEGER         NOT NULL DEFAULT 0,
    out_of_range_count          INTEGER         NOT NULL DEFAULT 0,
    stuck_value_count           INTEGER         NOT NULL DEFAULT 0,
    spike_count                 INTEGER         NOT NULL DEFAULT 0,
    -- Consistency
    checksum_validated          BOOLEAN,
    cross_check_passed          BOOLEAN,
    balance_error_pct           NUMERIC(8,4),
    -- Timeliness
    data_latency_minutes        NUMERIC(10,2),
    meets_latency_sla           BOOLEAN,
    -- Overall
    overall_quality_score       NUMERIC(5,2)    NOT NULL,
    quality_grade               VARCHAR(5)      NOT NULL DEFAULT 'B',
    meets_mv_requirements       BOOLEAN         NOT NULL DEFAULT true,
    -- Issues
    issues_found                JSONB           DEFAULT '[]',
    corrective_actions          JSONB           DEFAULT '[]',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_dql_dates CHECK (
        assessment_period_start < assessment_period_end
    ),
    CONSTRAINT chk_p040_dql_readings CHECK (
        expected_readings >= 0 AND actual_readings >= 0
    ),
    CONSTRAINT chk_p040_dql_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p040_dql_gap_count CHECK (
        gap_count >= 0 AND gaps_filled_count >= 0
    ),
    CONSTRAINT chk_p040_dql_gap_fill CHECK (
        gap_fill_method IS NULL OR gap_fill_method IN (
            'LINEAR_INTERPOLATION', 'REGRESSION_PREDICT', 'HISTORICAL_AVERAGE',
            'DEGREE_DAY_SCALING', 'MANUAL', 'NONE'
        )
    ),
    CONSTRAINT chk_p040_dql_suspect CHECK (
        suspect_readings_count >= 0 AND corrected_readings_count >= 0
    ),
    CONSTRAINT chk_p040_dql_quality CHECK (
        overall_quality_score >= 0 AND overall_quality_score <= 100
    ),
    CONSTRAINT chk_p040_dql_grade CHECK (
        quality_grade IN ('A', 'B', 'C', 'D', 'F')
    ),
    CONSTRAINT chk_p040_dql_balance CHECK (
        balance_error_pct IS NULL OR
        (balance_error_pct >= -100 AND balance_error_pct <= 100)
    ),
    CONSTRAINT uq_p040_dql_meter_date UNIQUE (meter_assignment_id, assessment_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_dql_tenant           ON pack040_mv.mv_data_quality_logs(tenant_id);
CREATE INDEX idx_p040_dql_meter            ON pack040_mv.mv_data_quality_logs(meter_assignment_id);
CREATE INDEX idx_p040_dql_date             ON pack040_mv.mv_data_quality_logs(assessment_date DESC);
CREATE INDEX idx_p040_dql_grade            ON pack040_mv.mv_data_quality_logs(quality_grade);
CREATE INDEX idx_p040_dql_meets            ON pack040_mv.mv_data_quality_logs(meets_mv_requirements);
CREATE INDEX idx_p040_dql_created          ON pack040_mv.mv_data_quality_logs(created_at DESC);

-- Composite: meter + failing quality
CREATE INDEX idx_p040_dql_meter_fail       ON pack040_mv.mv_data_quality_logs(meter_assignment_id, assessment_date DESC)
    WHERE meets_mv_requirements = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_dql_updated
    BEFORE UPDATE ON pack040_mv.mv_data_quality_logs
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_metering_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_meter_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_calibration_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_sampling_protocols ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_data_quality_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_mp_tenant_isolation
    ON pack040_mv.mv_metering_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_mp_service_bypass
    ON pack040_mv.mv_metering_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_ma_tenant_isolation
    ON pack040_mv.mv_meter_assignments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_ma_service_bypass
    ON pack040_mv.mv_meter_assignments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_cal_tenant_isolation
    ON pack040_mv.mv_calibration_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_cal_service_bypass
    ON pack040_mv.mv_calibration_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_spr_tenant_isolation
    ON pack040_mv.mv_sampling_protocols
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_spr_service_bypass
    ON pack040_mv.mv_sampling_protocols
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_dql_tenant_isolation
    ON pack040_mv.mv_data_quality_logs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_dql_service_bypass
    ON pack040_mv.mv_data_quality_logs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_metering_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_meter_assignments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_calibration_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_sampling_protocols TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_data_quality_logs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_metering_plans IS
    'M&V metering plans defining measurement strategy, data collection methods, calibration requirements, and quality assurance procedures.';
COMMENT ON TABLE pack040_mv.mv_meter_assignments IS
    'Meter assignments mapping physical meters to M&V metering plans with role, measurement point, and accuracy specifications.';
COMMENT ON TABLE pack040_mv.mv_calibration_records IS
    'Calibration history for M&V meters with accuracy verification, drift analysis, and compliance certificates.';
COMMENT ON TABLE pack040_mv.mv_sampling_protocols IS
    'Sampling protocol definitions for IPMVP Option A with sample design, required size, and precision targets.';
COMMENT ON TABLE pack040_mv.mv_data_quality_logs IS
    'Periodic data quality assessments tracking completeness, accuracy, consistency, and overall quality score for M&V data.';

COMMENT ON COLUMN pack040_mv.mv_metering_plans.gap_filling_method IS 'Method for filling missing data gaps: LINEAR_INTERPOLATION, REGRESSION_PREDICT, DEGREE_DAY_SCALING.';
COMMENT ON COLUMN pack040_mv.mv_metering_plans.min_data_completeness_pct IS 'Minimum acceptable data completeness percentage for valid M&V calculations.';
COMMENT ON COLUMN pack040_mv.mv_metering_plans.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_meter_assignments.pack039_meter_id IS 'Reference to PACK-039 Energy Monitoring meter for integrated deployments.';
COMMENT ON COLUMN pack040_mv.mv_meter_assignments.assignment_role IS 'Role within M&V plan: PRIMARY, BACKUP, CHECK_METER, SUB_METER, REFERENCE.';

COMMENT ON COLUMN pack040_mv.mv_calibration_records.drift_pct IS 'Measurement drift between calibration events as percentage of reading.';

COMMENT ON COLUMN pack040_mv.mv_sampling_protocols.estimated_cv IS 'Estimated coefficient of variation for sample size calculation (from pilot or historical data).';
COMMENT ON COLUMN pack040_mv.mv_sampling_protocols.calculated_sample_size IS 'Required sample size = (t * CV / precision)^2 per ASHRAE 14.';

COMMENT ON COLUMN pack040_mv.mv_data_quality_logs.quality_grade IS 'Letter grade: A (95-100), B (85-94), C (75-84), D (65-74), F (<65).';
