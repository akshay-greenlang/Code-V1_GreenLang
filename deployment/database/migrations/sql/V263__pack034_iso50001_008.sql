-- =============================================================================
-- V263: PACK-034 ISO 50001 Energy Management System - Monitoring Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- Creates metering and monitoring tables per ISO 50001 Clause 6.6 and 9.1.
-- Supports metering plans, hierarchical meter structures, monitoring schedules,
-- and time-series meter readings with data quality flags.
--
-- Tables (4):
--   1. pack034_iso50001.metering_plans
--   2. pack034_iso50001.meter_hierarchy
--   3. pack034_iso50001.monitoring_schedules
--   4. pack034_iso50001.meter_readings
--
-- Previous: V262__pack034_iso50001_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.metering_plans
-- =============================================================================
-- Metering plans defining the measurement and monitoring approach for each
-- EnMS, including review schedules and plan status.

CREATE TABLE pack034_iso50001.metering_plans (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    plan_name                   VARCHAR(500)    NOT NULL,
    description                 TEXT,
    last_review_date            DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'draft',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_mp_status CHECK (
        status IN ('draft', 'active', 'under_review', 'archived')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_mp_enms            ON pack034_iso50001.metering_plans(enms_id);
CREATE INDEX idx_p034_mp_status          ON pack034_iso50001.metering_plans(status);
CREATE INDEX idx_p034_mp_review          ON pack034_iso50001.metering_plans(last_review_date);
CREATE INDEX idx_p034_mp_created         ON pack034_iso50001.metering_plans(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_mp_updated
    BEFORE UPDATE ON pack034_iso50001.metering_plans
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.meter_hierarchy
-- =============================================================================
-- Hierarchical meter structure supporting main, sub, check, and virtual meters.
-- Self-referencing FK enables unlimited nesting for complex metering topologies.

CREATE TABLE pack034_iso50001.meter_hierarchy (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                     UUID            NOT NULL REFERENCES pack034_iso50001.metering_plans(id) ON DELETE CASCADE,
    meter_name                  VARCHAR(500)    NOT NULL,
    meter_type                  VARCHAR(20)     NOT NULL,
    parent_meter_id             UUID            REFERENCES pack034_iso50001.meter_hierarchy(id) ON DELETE SET NULL,
    location                    VARCHAR(500),
    energy_type                 VARCHAR(100)    NOT NULL,
    accuracy_class              VARCHAR(50),
    calibration_due             DATE,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_mh_type CHECK (
        meter_type IN ('main', 'sub', 'check', 'virtual')
    ),
    CONSTRAINT chk_p034_mh_no_self_ref CHECK (
        parent_meter_id IS NULL OR parent_meter_id != id
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_mh_plan            ON pack034_iso50001.meter_hierarchy(plan_id);
CREATE INDEX idx_p034_mh_type            ON pack034_iso50001.meter_hierarchy(meter_type);
CREATE INDEX idx_p034_mh_parent          ON pack034_iso50001.meter_hierarchy(parent_meter_id);
CREATE INDEX idx_p034_mh_energy_type     ON pack034_iso50001.meter_hierarchy(energy_type);
CREATE INDEX idx_p034_mh_calibration     ON pack034_iso50001.meter_hierarchy(calibration_due);
CREATE INDEX idx_p034_mh_location        ON pack034_iso50001.meter_hierarchy(location);
CREATE INDEX idx_p034_mh_created         ON pack034_iso50001.meter_hierarchy(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_mh_updated
    BEFORE UPDATE ON pack034_iso50001.meter_hierarchy
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.monitoring_schedules
-- =============================================================================
-- Monitoring schedule definitions specifying what parameters are measured,
-- at what frequency, by whom, and using what method.

CREATE TABLE pack034_iso50001.monitoring_schedules (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                     UUID            NOT NULL REFERENCES pack034_iso50001.metering_plans(id) ON DELETE CASCADE,
    parameter_name              VARCHAR(255)    NOT NULL,
    frequency                   VARCHAR(20)     NOT NULL,
    responsible_person          VARCHAR(255),
    method                      TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ms_frequency CHECK (
        frequency IN ('continuous', 'daily', 'weekly', 'monthly', 'quarterly')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_ms_plan            ON pack034_iso50001.monitoring_schedules(plan_id);
CREATE INDEX idx_p034_ms_parameter       ON pack034_iso50001.monitoring_schedules(parameter_name);
CREATE INDEX idx_p034_ms_frequency       ON pack034_iso50001.monitoring_schedules(frequency);
CREATE INDEX idx_p034_ms_responsible     ON pack034_iso50001.monitoring_schedules(responsible_person);
CREATE INDEX idx_p034_ms_created         ON pack034_iso50001.monitoring_schedules(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ms_updated
    BEFORE UPDATE ON pack034_iso50001.monitoring_schedules
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack034_iso50001.meter_readings
-- =============================================================================
-- Time-series meter readings with quality flags indicating whether data
-- was directly measured, estimated, or calculated from sub-meters.

CREATE TABLE pack034_iso50001.meter_readings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                    UUID            NOT NULL REFERENCES pack034_iso50001.meter_hierarchy(id) ON DELETE CASCADE,
    reading_timestamp           TIMESTAMPTZ     NOT NULL,
    value                       DECIMAL(18,4)   NOT NULL,
    unit                        VARCHAR(50)     NOT NULL,
    quality_flag                VARCHAR(20)     NOT NULL DEFAULT 'measured',
    source                      VARCHAR(255),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_mr_quality CHECK (
        quality_flag IN ('measured', 'estimated', 'calculated')
    ),
    CONSTRAINT chk_p034_mr_value CHECK (
        value >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_mr_meter           ON pack034_iso50001.meter_readings(meter_id);
CREATE INDEX idx_p034_mr_timestamp       ON pack034_iso50001.meter_readings(reading_timestamp DESC);
CREATE INDEX idx_p034_mr_quality         ON pack034_iso50001.meter_readings(quality_flag);
CREATE INDEX idx_p034_mr_source          ON pack034_iso50001.meter_readings(source);
CREATE INDEX idx_p034_mr_created         ON pack034_iso50001.meter_readings(created_at DESC);
CREATE INDEX idx_p034_mr_meter_time      ON pack034_iso50001.meter_readings(meter_id, reading_timestamp DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_mr_updated
    BEFORE UPDATE ON pack034_iso50001.meter_readings
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.metering_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.meter_hierarchy ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.monitoring_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.meter_readings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_mp_tenant_isolation
    ON pack034_iso50001.metering_plans
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_mp_service_bypass
    ON pack034_iso50001.metering_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_mh_tenant_isolation
    ON pack034_iso50001.meter_hierarchy
    USING (plan_id IN (
        SELECT id FROM pack034_iso50001.metering_plans
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_mh_service_bypass
    ON pack034_iso50001.meter_hierarchy
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_ms_tenant_isolation
    ON pack034_iso50001.monitoring_schedules
    USING (plan_id IN (
        SELECT id FROM pack034_iso50001.metering_plans
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_ms_service_bypass
    ON pack034_iso50001.monitoring_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_mr_tenant_isolation
    ON pack034_iso50001.meter_readings
    USING (meter_id IN (
        SELECT id FROM pack034_iso50001.meter_hierarchy
        WHERE plan_id IN (
            SELECT id FROM pack034_iso50001.metering_plans
            WHERE enms_id IN (
                SELECT id FROM pack034_iso50001.energy_management_systems
                WHERE organization_id = current_setting('app.current_tenant')::UUID
            )
        )
    ));
CREATE POLICY p034_mr_service_bypass
    ON pack034_iso50001.meter_readings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.metering_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.meter_hierarchy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.monitoring_schedules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.meter_readings TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.metering_plans IS
    'Metering plans defining the measurement and monitoring approach per ISO 50001 Clause 6.6.';

COMMENT ON TABLE pack034_iso50001.meter_hierarchy IS
    'Hierarchical meter structure supporting main, sub, check, and virtual meters with self-referencing parent relationships.';

COMMENT ON TABLE pack034_iso50001.monitoring_schedules IS
    'Monitoring schedule definitions specifying parameters, frequencies, responsibilities, and methods.';

COMMENT ON TABLE pack034_iso50001.meter_readings IS
    'Time-series meter readings with data quality flags for measured, estimated, or calculated values.';

COMMENT ON COLUMN pack034_iso50001.meter_hierarchy.meter_type IS
    'Meter type: main (fiscal/utility), sub (departmental/process), check (verification), virtual (calculated from others).';
COMMENT ON COLUMN pack034_iso50001.meter_hierarchy.parent_meter_id IS
    'Self-referencing FK to parent meter, enabling hierarchical meter topologies.';
COMMENT ON COLUMN pack034_iso50001.meter_hierarchy.accuracy_class IS
    'Meter accuracy class (e.g., Class 0.5, Class 1.0) per IEC 62053 or equivalent.';
COMMENT ON COLUMN pack034_iso50001.meter_hierarchy.calibration_due IS
    'Date when next calibration is due. Overdue calibrations affect data quality scoring.';
COMMENT ON COLUMN pack034_iso50001.monitoring_schedules.frequency IS
    'Measurement frequency: continuous (real-time), daily, weekly, monthly, or quarterly.';
COMMENT ON COLUMN pack034_iso50001.meter_readings.quality_flag IS
    'Data quality: measured (direct reading), estimated (gap-filled), calculated (derived from sub-meters).';
COMMENT ON COLUMN pack034_iso50001.meter_readings.source IS
    'Data source identifier (e.g., BMS, manual reading, AMI, SCADA).';
