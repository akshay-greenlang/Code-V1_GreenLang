-- =============================================================================
-- V306: PACK-039 Energy Monitoring Pack - Meter Registry
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the pack039_energy_monitoring schema and foundational tables for
-- meter registry management. Tracks physical and virtual meters, meter
-- channels, hierarchical meter relationships, calibration records, and
-- virtual meter calculation definitions used by all downstream monitoring
-- engines including data acquisition, validation, and EnPI tracking.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_meters
--   2. pack039_energy_monitoring.em_meter_channels
--   3. pack039_energy_monitoring.em_meter_hierarchy
--   4. pack039_energy_monitoring.em_calibration_records
--   5. pack039_energy_monitoring.em_virtual_meters
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V305__pack038_peak_shaving_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack039_energy_monitoring;

SET search_path TO pack039_energy_monitoring, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack039_energy_monitoring.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_meters
-- =============================================================================
-- Central registry of all energy meters (physical and logical) within a
-- tenant's monitoring infrastructure. Each meter represents a measurement
-- point that captures energy consumption, generation, or sub-metering data.
-- Meters are associated with facilities, buildings, or equipment and serve
-- as the foundational entity for all downstream data collection and analysis.

CREATE TABLE pack039_energy_monitoring.em_meters (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    meter_name              VARCHAR(255)    NOT NULL,
    meter_serial_number     VARCHAR(100),
    meter_type              VARCHAR(50)     NOT NULL DEFAULT 'INTERVAL',
    meter_category          VARCHAR(50)     NOT NULL DEFAULT 'MAIN',
    energy_type             VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    manufacturer            VARCHAR(100),
    model                   VARCHAR(100),
    firmware_version        VARCHAR(50),
    communication_protocol  VARCHAR(50)     NOT NULL DEFAULT 'MODBUS_TCP',
    ip_address              VARCHAR(45),
    port_number             INTEGER,
    slave_id                INTEGER,
    connection_string       TEXT,
    installation_date       DATE,
    commissioning_date      DATE,
    last_calibration_date   DATE,
    next_calibration_date   DATE,
    ct_ratio                NUMERIC(10,2),
    pt_ratio                NUMERIC(10,2),
    rated_voltage_v         NUMERIC(10,2),
    rated_current_a         NUMERIC(10,2),
    accuracy_class          VARCHAR(20),
    measurement_unit        VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    interval_length_minutes INTEGER         NOT NULL DEFAULT 15,
    timezone                VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    building_name           VARCHAR(255),
    floor_level             VARCHAR(50),
    panel_id                VARCHAR(100),
    circuit_id              VARCHAR(100),
    location_description    TEXT,
    latitude                NUMERIC(10,7),
    longitude               NUMERIC(10,7),
    is_revenue_grade        BOOLEAN         NOT NULL DEFAULT false,
    is_bidirectional        BOOLEAN         NOT NULL DEFAULT false,
    is_virtual              BOOLEAN         NOT NULL DEFAULT false,
    meter_status            VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    data_quality_score      NUMERIC(5,2),
    tags                    JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p039_mt_type CHECK (
        meter_type IN (
            'INTERVAL', 'SMART_METER', 'SCADA', 'PULSE', 'CT_LOGGER',
            'POWER_ANALYZER', 'IOT_SENSOR', 'MANUAL_READ', 'VIRTUAL',
            'UTILITY_AMI', 'DATA_LOGGER', 'BMS_POINT'
        )
    ),
    CONSTRAINT chk_p039_mt_category CHECK (
        meter_category IN (
            'MAIN', 'SUB_METER', 'TENANT_METER', 'CHECK_METER',
            'GENERATION', 'STORAGE', 'EV_CHARGER', 'HVAC',
            'LIGHTING', 'PROCESS', 'PLUG_LOAD', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_mt_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'CHILLED_WATER',
            'HOT_WATER', 'COMPRESSED_AIR', 'DIESEL', 'PROPANE',
            'FUEL_OIL', 'DISTRICT_HEATING', 'DISTRICT_COOLING',
            'SOLAR_THERMAL', 'BIOMASS', 'HYDROGEN', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_mt_protocol CHECK (
        communication_protocol IN (
            'MODBUS_TCP', 'MODBUS_RTU', 'BACNET_IP', 'BACNET_MSTP',
            'LONWORKS', 'OPC_UA', 'OPC_DA', 'MQTT', 'HTTP_REST',
            'SNMP', 'DLMS_COSEM', 'IEC_61850', 'DNP3', 'MANUAL',
            'CSV_IMPORT', 'API', 'PULSE_COUNTER', 'ANALOG_4_20MA'
        )
    ),
    CONSTRAINT chk_p039_mt_accuracy CHECK (
        accuracy_class IS NULL OR accuracy_class IN (
            'CLASS_0_1', 'CLASS_0_2', 'CLASS_0_5', 'CLASS_1',
            'CLASS_2', 'CLASS_3', 'REVENUE_GRADE', 'UTILITY_GRADE',
            'MONITORING_GRADE', 'INDICATIVE'
        )
    ),
    CONSTRAINT chk_p039_mt_unit CHECK (
        measurement_unit IN (
            'kWh', 'MWh', 'GJ', 'therms', 'CCF', 'MCF',
            'MMBtu', 'kW', 'MW', 'tonnes_steam', 'ton_hours',
            'gallons', 'liters', 'Nm3', 'kg', 'BTU'
        )
    ),
    CONSTRAINT chk_p039_mt_interval CHECK (
        interval_length_minutes IN (1, 5, 10, 15, 30, 60)
    ),
    CONSTRAINT chk_p039_mt_status CHECK (
        meter_status IN (
            'ACTIVE', 'INACTIVE', 'MAINTENANCE', 'DECOMMISSIONED',
            'PENDING_INSTALL', 'PENDING_COMMISSION', 'FAULT', 'OFFLINE'
        )
    ),
    CONSTRAINT chk_p039_mt_port CHECK (
        port_number IS NULL OR (port_number >= 1 AND port_number <= 65535)
    ),
    CONSTRAINT chk_p039_mt_slave CHECK (
        slave_id IS NULL OR (slave_id >= 0 AND slave_id <= 255)
    ),
    CONSTRAINT chk_p039_mt_ct_ratio CHECK (
        ct_ratio IS NULL OR ct_ratio > 0
    ),
    CONSTRAINT chk_p039_mt_pt_ratio CHECK (
        pt_ratio IS NULL OR pt_ratio > 0
    ),
    CONSTRAINT chk_p039_mt_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p039_mt_latitude CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    ),
    CONSTRAINT chk_p039_mt_longitude CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    ),
    CONSTRAINT chk_p039_mt_cal_dates CHECK (
        last_calibration_date IS NULL OR next_calibration_date IS NULL OR
        last_calibration_date <= next_calibration_date
    ),
    CONSTRAINT uq_p039_mt_tenant_serial UNIQUE (tenant_id, meter_serial_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_mt_tenant          ON pack039_energy_monitoring.em_meters(tenant_id);
CREATE INDEX idx_p039_mt_facility        ON pack039_energy_monitoring.em_meters(facility_id);
CREATE INDEX idx_p039_mt_serial          ON pack039_energy_monitoring.em_meters(meter_serial_number);
CREATE INDEX idx_p039_mt_type            ON pack039_energy_monitoring.em_meters(meter_type);
CREATE INDEX idx_p039_mt_category        ON pack039_energy_monitoring.em_meters(meter_category);
CREATE INDEX idx_p039_mt_energy_type     ON pack039_energy_monitoring.em_meters(energy_type);
CREATE INDEX idx_p039_mt_status          ON pack039_energy_monitoring.em_meters(meter_status);
CREATE INDEX idx_p039_mt_protocol        ON pack039_energy_monitoring.em_meters(communication_protocol);
CREATE INDEX idx_p039_mt_building        ON pack039_energy_monitoring.em_meters(building_name);
CREATE INDEX idx_p039_mt_cal_next        ON pack039_energy_monitoring.em_meters(next_calibration_date);
CREATE INDEX idx_p039_mt_created         ON pack039_energy_monitoring.em_meters(created_at DESC);
CREATE INDEX idx_p039_mt_metadata        ON pack039_energy_monitoring.em_meters USING GIN(metadata);
CREATE INDEX idx_p039_mt_tags            ON pack039_energy_monitoring.em_meters USING GIN(tags);

-- Composite: tenant + facility + active meters for hierarchy queries
CREATE INDEX idx_p039_mt_tenant_fac_act  ON pack039_energy_monitoring.em_meters(tenant_id, facility_id)
    WHERE meter_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_mt_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_meters
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_meter_channels
-- =============================================================================
-- Individual measurement channels within a meter. A single physical meter
-- may have multiple channels (e.g., kW, kWh, kVAR, voltage, current per
-- phase). Each channel has its own scaling, engineering unit, and data
-- quality configuration. Channels are the atomic measurement points that
-- feed the data acquisition engine.

CREATE TABLE pack039_energy_monitoring.em_meter_channels (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    channel_name            VARCHAR(100)    NOT NULL,
    channel_number          INTEGER         NOT NULL DEFAULT 1,
    channel_type            VARCHAR(50)     NOT NULL DEFAULT 'ENERGY_CONSUMPTION',
    register_address        INTEGER,
    register_type           VARCHAR(30),
    data_type               VARCHAR(30)     NOT NULL DEFAULT 'FLOAT32',
    engineering_unit        VARCHAR(30)     NOT NULL DEFAULT 'kWh',
    scaling_factor          NUMERIC(15,6)   NOT NULL DEFAULT 1.0,
    scaling_offset          NUMERIC(15,6)   NOT NULL DEFAULT 0.0,
    byte_order              VARCHAR(20)     NOT NULL DEFAULT 'BIG_ENDIAN',
    word_order              VARCHAR(20)     NOT NULL DEFAULT 'BIG_ENDIAN',
    min_valid_value         NUMERIC(15,3),
    max_valid_value         NUMERIC(15,3),
    deadband_value          NUMERIC(15,6),
    deadband_type           VARCHAR(20)     DEFAULT 'ABSOLUTE',
    rollover_value          NUMERIC(20,3),
    is_cumulative           BOOLEAN         NOT NULL DEFAULT true,
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    phase                   VARCHAR(10),
    description             TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_mc_channel_type CHECK (
        channel_type IN (
            'ENERGY_CONSUMPTION', 'ENERGY_GENERATION', 'DEMAND_KW',
            'DEMAND_KVA', 'REACTIVE_POWER', 'APPARENT_POWER',
            'POWER_FACTOR', 'VOLTAGE', 'CURRENT', 'FREQUENCY',
            'TEMPERATURE', 'PRESSURE', 'FLOW_RATE', 'VOLUME',
            'HUMIDITY', 'CO2_LEVEL', 'STATUS', 'COUNTER',
            'ANALOG_INPUT', 'DIGITAL_INPUT'
        )
    ),
    CONSTRAINT chk_p039_mc_register_type CHECK (
        register_type IS NULL OR register_type IN (
            'HOLDING', 'INPUT', 'COIL', 'DISCRETE', 'ANALOG_VALUE',
            'BINARY_VALUE', 'MULTISTATE_VALUE', 'ACCUMULATOR'
        )
    ),
    CONSTRAINT chk_p039_mc_data_type CHECK (
        data_type IN (
            'INT16', 'UINT16', 'INT32', 'UINT32', 'INT64', 'UINT64',
            'FLOAT32', 'FLOAT64', 'BOOLEAN', 'STRING', 'BCD'
        )
    ),
    CONSTRAINT chk_p039_mc_byte_order CHECK (
        byte_order IN ('BIG_ENDIAN', 'LITTLE_ENDIAN', 'MID_BIG_ENDIAN', 'MID_LITTLE_ENDIAN')
    ),
    CONSTRAINT chk_p039_mc_word_order CHECK (
        word_order IN ('BIG_ENDIAN', 'LITTLE_ENDIAN', 'MID_BIG_ENDIAN', 'MID_LITTLE_ENDIAN')
    ),
    CONSTRAINT chk_p039_mc_deadband_type CHECK (
        deadband_type IS NULL OR deadband_type IN ('ABSOLUTE', 'PERCENTAGE')
    ),
    CONSTRAINT chk_p039_mc_phase CHECK (
        phase IS NULL OR phase IN ('A', 'B', 'C', 'AB', 'BC', 'CA', 'ABC', 'TOTAL', 'N')
    ),
    CONSTRAINT chk_p039_mc_scaling CHECK (
        scaling_factor != 0
    ),
    CONSTRAINT chk_p039_mc_channel_num CHECK (
        channel_number >= 1 AND channel_number <= 256
    ),
    CONSTRAINT chk_p039_mc_valid_range CHECK (
        min_valid_value IS NULL OR max_valid_value IS NULL OR
        min_valid_value < max_valid_value
    ),
    CONSTRAINT uq_p039_mc_meter_channel UNIQUE (meter_id, channel_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_mc_meter            ON pack039_energy_monitoring.em_meter_channels(meter_id);
CREATE INDEX idx_p039_mc_tenant           ON pack039_energy_monitoring.em_meter_channels(tenant_id);
CREATE INDEX idx_p039_mc_type             ON pack039_energy_monitoring.em_meter_channels(channel_type);
CREATE INDEX idx_p039_mc_unit             ON pack039_energy_monitoring.em_meter_channels(engineering_unit);
CREATE INDEX idx_p039_mc_enabled          ON pack039_energy_monitoring.em_meter_channels(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_mc_cumulative       ON pack039_energy_monitoring.em_meter_channels(is_cumulative) WHERE is_cumulative = true;
CREATE INDEX idx_p039_mc_created          ON pack039_energy_monitoring.em_meter_channels(created_at DESC);

-- Composite: meter + enabled channels for acquisition
CREATE INDEX idx_p039_mc_meter_enabled    ON pack039_energy_monitoring.em_meter_channels(meter_id, channel_number)
    WHERE is_enabled = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_mc_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_meter_channels
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_meter_hierarchy
-- =============================================================================
-- Defines parent-child relationships between meters to model metering
-- hierarchies (e.g., main meter -> distribution panel -> sub-meters).
-- Used for automatic aggregation, balance checking, and loss calculation.
-- The hierarchy enables top-down energy allocation and bottom-up
-- reconciliation across the metering infrastructure.

CREATE TABLE pack039_energy_monitoring.em_meter_hierarchy (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    parent_meter_id         UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    child_meter_id          UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    hierarchy_level         INTEGER         NOT NULL DEFAULT 1,
    relationship_type       VARCHAR(30)     NOT NULL DEFAULT 'SUB_METER',
    allocation_pct          NUMERIC(7,4)    DEFAULT 100.0,
    polarity                VARCHAR(10)     NOT NULL DEFAULT 'POSITIVE',
    loss_factor_pct         NUMERIC(5,3)    DEFAULT 0.0,
    effective_from          DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to            DATE,
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_mh_rel_type CHECK (
        relationship_type IN (
            'SUB_METER', 'CHECK_METER', 'BACKUP_METER', 'TENANT_METER',
            'GENERATION', 'STORAGE', 'FEED_IN', 'EXPORT', 'LOSS_METER'
        )
    ),
    CONSTRAINT chk_p039_mh_polarity CHECK (
        polarity IN ('POSITIVE', 'NEGATIVE')
    ),
    CONSTRAINT chk_p039_mh_alloc CHECK (
        allocation_pct >= 0 AND allocation_pct <= 100
    ),
    CONSTRAINT chk_p039_mh_loss CHECK (
        loss_factor_pct IS NULL OR (loss_factor_pct >= 0 AND loss_factor_pct <= 50)
    ),
    CONSTRAINT chk_p039_mh_level CHECK (
        hierarchy_level >= 1 AND hierarchy_level <= 10
    ),
    CONSTRAINT chk_p039_mh_dates CHECK (
        effective_to IS NULL OR effective_from <= effective_to
    ),
    CONSTRAINT chk_p039_mh_no_self CHECK (
        parent_meter_id != child_meter_id
    ),
    CONSTRAINT uq_p039_mh_parent_child UNIQUE (parent_meter_id, child_meter_id, effective_from)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_mh_tenant          ON pack039_energy_monitoring.em_meter_hierarchy(tenant_id);
CREATE INDEX idx_p039_mh_parent          ON pack039_energy_monitoring.em_meter_hierarchy(parent_meter_id);
CREATE INDEX idx_p039_mh_child           ON pack039_energy_monitoring.em_meter_hierarchy(child_meter_id);
CREATE INDEX idx_p039_mh_level           ON pack039_energy_monitoring.em_meter_hierarchy(hierarchy_level);
CREATE INDEX idx_p039_mh_rel_type        ON pack039_energy_monitoring.em_meter_hierarchy(relationship_type);
CREATE INDEX idx_p039_mh_active          ON pack039_energy_monitoring.em_meter_hierarchy(is_active) WHERE is_active = true;
CREATE INDEX idx_p039_mh_effective       ON pack039_energy_monitoring.em_meter_hierarchy(effective_from, effective_to);
CREATE INDEX idx_p039_mh_created         ON pack039_energy_monitoring.em_meter_hierarchy(created_at DESC);

-- Composite: active parent-child relationships
CREATE INDEX idx_p039_mh_parent_active   ON pack039_energy_monitoring.em_meter_hierarchy(parent_meter_id, child_meter_id)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_mh_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_meter_hierarchy
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_calibration_records
-- =============================================================================
-- Tracks meter calibration history including pre/post calibration readings,
-- drift analysis, accuracy verification, and compliance certificates.
-- Calibration records are critical for revenue-grade meters and regulatory
-- reporting where measurement accuracy must be demonstrated and traceable.

CREATE TABLE pack039_energy_monitoring.em_calibration_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    calibration_date        DATE            NOT NULL,
    calibration_type        VARCHAR(30)     NOT NULL DEFAULT 'ROUTINE',
    calibration_standard    VARCHAR(100),
    calibration_lab         VARCHAR(255),
    technician_name         VARCHAR(255),
    certificate_number      VARCHAR(100),
    pre_cal_reading         NUMERIC(15,3),
    post_cal_reading        NUMERIC(15,3),
    pre_cal_error_pct       NUMERIC(8,4),
    post_cal_error_pct      NUMERIC(8,4),
    drift_pct               NUMERIC(8,4),
    accuracy_verified       BOOLEAN         NOT NULL DEFAULT false,
    pass_fail               VARCHAR(10)     NOT NULL DEFAULT 'PASS',
    adjustment_made         BOOLEAN         NOT NULL DEFAULT false,
    adjustment_description  TEXT,
    ct_ratio_verified       BOOLEAN         DEFAULT false,
    pt_ratio_verified       BOOLEAN         DEFAULT false,
    next_due_date           DATE,
    calibration_interval_months INTEGER     DEFAULT 12,
    ambient_temp_c          NUMERIC(6,2),
    ambient_humidity_pct    NUMERIC(5,2),
    test_points             JSONB           DEFAULT '[]',
    certificate_document_id UUID,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_cr_type CHECK (
        calibration_type IN (
            'ROUTINE', 'INITIAL', 'POST_REPAIR', 'VERIFICATION',
            'RECALIBRATION', 'COMPLIANCE', 'SPOT_CHECK', 'FULL_TEST'
        )
    ),
    CONSTRAINT chk_p039_cr_pass_fail CHECK (
        pass_fail IN ('PASS', 'FAIL', 'CONDITIONAL', 'DEFERRED')
    ),
    CONSTRAINT chk_p039_cr_interval CHECK (
        calibration_interval_months IS NULL OR
        (calibration_interval_months >= 1 AND calibration_interval_months <= 120)
    ),
    CONSTRAINT chk_p039_cr_drift CHECK (
        drift_pct IS NULL OR (drift_pct >= -100 AND drift_pct <= 100)
    ),
    CONSTRAINT chk_p039_cr_pre_error CHECK (
        pre_cal_error_pct IS NULL OR (pre_cal_error_pct >= -100 AND pre_cal_error_pct <= 100)
    ),
    CONSTRAINT chk_p039_cr_post_error CHECK (
        post_cal_error_pct IS NULL OR (post_cal_error_pct >= -100 AND post_cal_error_pct <= 100)
    ),
    CONSTRAINT chk_p039_cr_humidity CHECK (
        ambient_humidity_pct IS NULL OR (ambient_humidity_pct >= 0 AND ambient_humidity_pct <= 100)
    ),
    CONSTRAINT chk_p039_cr_dates CHECK (
        next_due_date IS NULL OR calibration_date <= next_due_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_cr_meter           ON pack039_energy_monitoring.em_calibration_records(meter_id);
CREATE INDEX idx_p039_cr_tenant          ON pack039_energy_monitoring.em_calibration_records(tenant_id);
CREATE INDEX idx_p039_cr_date            ON pack039_energy_monitoring.em_calibration_records(calibration_date DESC);
CREATE INDEX idx_p039_cr_type            ON pack039_energy_monitoring.em_calibration_records(calibration_type);
CREATE INDEX idx_p039_cr_pass_fail       ON pack039_energy_monitoring.em_calibration_records(pass_fail);
CREATE INDEX idx_p039_cr_next_due        ON pack039_energy_monitoring.em_calibration_records(next_due_date);
CREATE INDEX idx_p039_cr_certificate     ON pack039_energy_monitoring.em_calibration_records(certificate_number);
CREATE INDEX idx_p039_cr_created         ON pack039_energy_monitoring.em_calibration_records(created_at DESC);

-- Composite: meter + upcoming calibrations for scheduling
CREATE INDEX idx_p039_cr_meter_upcoming  ON pack039_energy_monitoring.em_calibration_records(meter_id, next_due_date)
    WHERE pass_fail = 'PASS';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_cr_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_calibration_records
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_virtual_meters
-- =============================================================================
-- Virtual meter definitions that calculate derived values from combinations
-- of physical meters using mathematical expressions. Virtual meters enable
-- calculated metering points such as net consumption (main minus solar),
-- building-level aggregation, or loss calculation without physical hardware.
-- The formula engine evaluates expressions at each interval using referenced
-- meter data.

CREATE TABLE pack039_energy_monitoring.em_virtual_meters (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    virtual_meter_name      VARCHAR(255)    NOT NULL,
    formula_expression      TEXT            NOT NULL,
    formula_type            VARCHAR(30)     NOT NULL DEFAULT 'ARITHMETIC',
    source_meter_ids        UUID[]          NOT NULL,
    source_channel_ids      UUID[],
    calculation_method      VARCHAR(30)     NOT NULL DEFAULT 'REAL_TIME',
    calculation_interval_minutes INTEGER    NOT NULL DEFAULT 15,
    output_unit             VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    rounding_precision      INTEGER         NOT NULL DEFAULT 3,
    fallback_strategy       VARCHAR(30)     NOT NULL DEFAULT 'SKIP',
    min_sources_required    INTEGER         NOT NULL DEFAULT 1,
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    last_calculated_at      TIMESTAMPTZ,
    last_calculation_status VARCHAR(20),
    error_count             INTEGER         NOT NULL DEFAULT 0,
    description             TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_vm_formula_type CHECK (
        formula_type IN (
            'ARITHMETIC', 'WEIGHTED_SUM', 'DIFFERENCE', 'RATIO',
            'MAX', 'MIN', 'AVERAGE', 'CUSTOM_SCRIPT'
        )
    ),
    CONSTRAINT chk_p039_vm_calc_method CHECK (
        calculation_method IN (
            'REAL_TIME', 'BATCH', 'ON_DEMAND', 'SCHEDULED'
        )
    ),
    CONSTRAINT chk_p039_vm_fallback CHECK (
        fallback_strategy IN (
            'SKIP', 'ZERO', 'LAST_KNOWN', 'INTERPOLATE', 'ESTIMATE', 'ERROR'
        )
    ),
    CONSTRAINT chk_p039_vm_calc_status CHECK (
        last_calculation_status IS NULL OR last_calculation_status IN (
            'SUCCESS', 'PARTIAL', 'FAILED', 'TIMEOUT', 'NO_DATA'
        )
    ),
    CONSTRAINT chk_p039_vm_calc_interval CHECK (
        calculation_interval_minutes IN (1, 5, 10, 15, 30, 60)
    ),
    CONSTRAINT chk_p039_vm_rounding CHECK (
        rounding_precision >= 0 AND rounding_precision <= 10
    ),
    CONSTRAINT chk_p039_vm_min_sources CHECK (
        min_sources_required >= 1 AND min_sources_required <= array_length(source_meter_ids, 1)
    ),
    CONSTRAINT chk_p039_vm_error_count CHECK (
        error_count >= 0
    ),
    CONSTRAINT uq_p039_vm_meter UNIQUE (meter_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_vm_meter           ON pack039_energy_monitoring.em_virtual_meters(meter_id);
CREATE INDEX idx_p039_vm_tenant          ON pack039_energy_monitoring.em_virtual_meters(tenant_id);
CREATE INDEX idx_p039_vm_formula_type    ON pack039_energy_monitoring.em_virtual_meters(formula_type);
CREATE INDEX idx_p039_vm_calc_method     ON pack039_energy_monitoring.em_virtual_meters(calculation_method);
CREATE INDEX idx_p039_vm_enabled         ON pack039_energy_monitoring.em_virtual_meters(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_vm_last_calc       ON pack039_energy_monitoring.em_virtual_meters(last_calculated_at DESC);
CREATE INDEX idx_p039_vm_status          ON pack039_energy_monitoring.em_virtual_meters(last_calculation_status);
CREATE INDEX idx_p039_vm_created         ON pack039_energy_monitoring.em_virtual_meters(created_at DESC);
CREATE INDEX idx_p039_vm_sources         ON pack039_energy_monitoring.em_virtual_meters USING GIN(source_meter_ids);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_vm_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_virtual_meters
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_meters ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_meter_channels ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_meter_hierarchy ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_calibration_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_virtual_meters ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_mt_tenant_isolation
    ON pack039_energy_monitoring.em_meters
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_mt_service_bypass
    ON pack039_energy_monitoring.em_meters
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_mc_tenant_isolation
    ON pack039_energy_monitoring.em_meter_channels
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_mc_service_bypass
    ON pack039_energy_monitoring.em_meter_channels
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_mh_tenant_isolation
    ON pack039_energy_monitoring.em_meter_hierarchy
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_mh_service_bypass
    ON pack039_energy_monitoring.em_meter_hierarchy
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_cr_tenant_isolation
    ON pack039_energy_monitoring.em_calibration_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_cr_service_bypass
    ON pack039_energy_monitoring.em_calibration_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_vm_tenant_isolation
    ON pack039_energy_monitoring.em_virtual_meters
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_vm_service_bypass
    ON pack039_energy_monitoring.em_virtual_meters
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack039_energy_monitoring TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_meters TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_meter_channels TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_meter_hierarchy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_calibration_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_virtual_meters TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack039_energy_monitoring.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack039_energy_monitoring IS
    'PACK-039 Energy Monitoring Pack - real-time energy metering, data acquisition and validation, anomaly detection, EnPI tracking, cost allocation, budgeting, alarm management, dashboards, and reporting.';

COMMENT ON TABLE pack039_energy_monitoring.em_meters IS
    'Central registry of physical and virtual energy meters with connection details, location, accuracy class, and lifecycle status.';
COMMENT ON TABLE pack039_energy_monitoring.em_meter_channels IS
    'Individual measurement channels within a meter, each with register mapping, scaling, engineering unit, and data quality configuration.';
COMMENT ON TABLE pack039_energy_monitoring.em_meter_hierarchy IS
    'Parent-child relationships between meters for aggregation, balance checking, loss calculation, and tenant sub-metering.';
COMMENT ON TABLE pack039_energy_monitoring.em_calibration_records IS
    'Meter calibration history with pre/post readings, drift analysis, accuracy verification, and compliance certificates.';
COMMENT ON TABLE pack039_energy_monitoring.em_virtual_meters IS
    'Virtual meter definitions using mathematical expressions to derive calculated values from combinations of physical meters.';

COMMENT ON COLUMN pack039_energy_monitoring.em_meters.id IS 'Unique identifier for the meter.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.facility_id IS 'Reference to the facility in the core facility registry.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.meter_type IS 'Physical meter technology: INTERVAL, SMART_METER, SCADA, PULSE, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.meter_category IS 'Functional category: MAIN, SUB_METER, TENANT_METER, GENERATION, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.energy_type IS 'Energy commodity measured: ELECTRICITY, NATURAL_GAS, STEAM, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.communication_protocol IS 'Data acquisition protocol: MODBUS_TCP, BACNET_IP, OPC_UA, MQTT, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.is_revenue_grade IS 'Whether the meter meets revenue-grade accuracy standards for billing.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meters.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_meter_channels.channel_type IS 'Measurement type: ENERGY_CONSUMPTION, DEMAND_KW, VOLTAGE, CURRENT, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meter_channels.scaling_factor IS 'Multiplier applied to raw register value for engineering unit conversion.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meter_channels.is_cumulative IS 'Whether the channel value is cumulative (totalizer) or instantaneous.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meter_channels.rollover_value IS 'Maximum counter value before rollover to zero for cumulative channels.';

COMMENT ON COLUMN pack039_energy_monitoring.em_meter_hierarchy.allocation_pct IS 'Percentage of parent meter energy allocated to this child relationship (0-100).';
COMMENT ON COLUMN pack039_energy_monitoring.em_meter_hierarchy.loss_factor_pct IS 'Distribution loss factor between parent and child meter in percent.';
COMMENT ON COLUMN pack039_energy_monitoring.em_meter_hierarchy.polarity IS 'Whether child meter adds (POSITIVE) or subtracts (NEGATIVE) from parent balance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_calibration_records.drift_pct IS 'Measurement drift between calibration events as a percentage of reading.';
COMMENT ON COLUMN pack039_energy_monitoring.em_calibration_records.accuracy_verified IS 'Whether meter accuracy was verified to be within specification.';

COMMENT ON COLUMN pack039_energy_monitoring.em_virtual_meters.formula_expression IS 'Mathematical expression defining the virtual meter calculation (e.g., "M1 - M2 + M3 * 0.95").';
COMMENT ON COLUMN pack039_energy_monitoring.em_virtual_meters.source_meter_ids IS 'Array of meter UUIDs referenced in the formula expression.';
COMMENT ON COLUMN pack039_energy_monitoring.em_virtual_meters.fallback_strategy IS 'Behavior when source meter data is missing: SKIP, ZERO, LAST_KNOWN, INTERPOLATE.';
