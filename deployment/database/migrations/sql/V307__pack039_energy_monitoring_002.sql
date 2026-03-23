-- =============================================================================
-- V307: PACK-039 Energy Monitoring Pack - Data Acquisition
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates data acquisition tables for collecting interval-level energy
-- data from meters. Includes the primary time-series table for interval
-- readings (TimescaleDB hypertable candidate), acquisition scheduling,
-- data buffering for store-and-forward, protocol-specific configurations,
-- and raw reading capture for audit purposes.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_interval_data       (TimescaleDB hypertable)
--   2. pack039_energy_monitoring.em_acquisition_schedules
--   3. pack039_energy_monitoring.em_data_buffers
--   4. pack039_energy_monitoring.em_protocol_configs
--   5. pack039_energy_monitoring.em_raw_readings
--
-- Previous: V306__pack039_energy_monitoring_001.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_interval_data
-- =============================================================================
-- Primary time-series table for normalized interval energy data. Each row
-- represents a single measurement interval from a meter channel. This table
-- is the backbone of the energy monitoring system, feeding validation,
-- anomaly detection, EnPI calculation, cost allocation, and reporting.
-- Configured as a TimescaleDB hypertable for efficient time-series queries
-- with monthly chunk intervals.

CREATE TABLE pack039_energy_monitoring.em_interval_data (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    channel_id              UUID            REFERENCES pack039_energy_monitoring.em_meter_channels(id),
    tenant_id               UUID            NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    value                   NUMERIC(18,6)   NOT NULL,
    engineering_unit        VARCHAR(30)     NOT NULL DEFAULT 'kWh',
    interval_length_minutes INTEGER         NOT NULL DEFAULT 15,
    demand_kw               NUMERIC(12,3),
    reactive_kvar           NUMERIC(12,3),
    apparent_kva            NUMERIC(12,3),
    power_factor            NUMERIC(5,4),
    voltage_avg_v           NUMERIC(10,3),
    current_avg_a           NUMERIC(10,3),
    frequency_hz            NUMERIC(7,4),
    temperature_c           NUMERIC(7,2),
    humidity_pct            NUMERIC(5,2),
    cumulative_reading      NUMERIC(20,3),
    data_quality            VARCHAR(20)     NOT NULL DEFAULT 'RAW',
    quality_score           NUMERIC(5,2),
    is_estimated            BOOLEAN         NOT NULL DEFAULT false,
    is_validated            BOOLEAN         NOT NULL DEFAULT false,
    is_peak_period          BOOLEAN         DEFAULT false,
    is_weekend              BOOLEAN         DEFAULT false,
    is_holiday              BOOLEAN         DEFAULT false,
    tariff_period           VARCHAR(30),
    source_system           VARCHAR(100),
    acquisition_method      VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATIC',
    raw_value               NUMERIC(20,6),
    scaling_applied         BOOLEAN         NOT NULL DEFAULT true,
    batch_id                UUID,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_id_interval CHECK (
        interval_length_minutes IN (1, 5, 10, 15, 30, 60)
    ),
    CONSTRAINT chk_p039_id_quality CHECK (
        data_quality IN (
            'RAW', 'VALIDATED', 'ESTIMATED', 'INTERPOLATED',
            'CORRECTED', 'MISSING', 'SUSPECT', 'REJECTED'
        )
    ),
    CONSTRAINT chk_p039_id_quality_score CHECK (
        quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100)
    ),
    CONSTRAINT chk_p039_id_pf CHECK (
        power_factor IS NULL OR (power_factor >= 0 AND power_factor <= 1.0)
    ),
    CONSTRAINT chk_p039_id_voltage CHECK (
        voltage_avg_v IS NULL OR voltage_avg_v >= 0
    ),
    CONSTRAINT chk_p039_id_current CHECK (
        current_avg_a IS NULL OR current_avg_a >= 0
    ),
    CONSTRAINT chk_p039_id_frequency CHECK (
        frequency_hz IS NULL OR (frequency_hz >= 40 AND frequency_hz <= 70)
    ),
    CONSTRAINT chk_p039_id_humidity CHECK (
        humidity_pct IS NULL OR (humidity_pct >= 0 AND humidity_pct <= 100)
    ),
    CONSTRAINT chk_p039_id_acq_method CHECK (
        acquisition_method IN (
            'AUTOMATIC', 'MANUAL', 'IMPORT', 'API', 'CALCULATED', 'ESTIMATED'
        )
    ),
    CONSTRAINT chk_p039_id_tariff CHECK (
        tariff_period IS NULL OR tariff_period IN (
            'ON_PEAK', 'OFF_PEAK', 'MID_PEAK', 'SHOULDER',
            'SUPER_PEAK', 'CRITICAL_PEAK', 'WEEKEND', 'HOLIDAY'
        )
    ),
    CONSTRAINT uq_p039_id_meter_channel_ts UNIQUE (meter_id, channel_id, timestamp)
);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable (skip if extension not available)
-- ---------------------------------------------------------------------------
-- SELECT create_hypertable('pack039_energy_monitoring.em_interval_data', 'timestamp',
--     chunk_time_interval => INTERVAL '1 month',
--     if_not_exists => TRUE
-- );

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_id_meter           ON pack039_energy_monitoring.em_interval_data(meter_id);
CREATE INDEX idx_p039_id_channel         ON pack039_energy_monitoring.em_interval_data(channel_id);
CREATE INDEX idx_p039_id_tenant          ON pack039_energy_monitoring.em_interval_data(tenant_id);
CREATE INDEX idx_p039_id_timestamp       ON pack039_energy_monitoring.em_interval_data(timestamp DESC);
CREATE INDEX idx_p039_id_meter_ts        ON pack039_energy_monitoring.em_interval_data(tenant_id, meter_id, timestamp DESC);
CREATE INDEX idx_p039_id_quality         ON pack039_energy_monitoring.em_interval_data(data_quality);
CREATE INDEX idx_p039_id_estimated       ON pack039_energy_monitoring.em_interval_data(is_estimated) WHERE is_estimated = true;
CREATE INDEX idx_p039_id_validated       ON pack039_energy_monitoring.em_interval_data(is_validated) WHERE is_validated = false;
CREATE INDEX idx_p039_id_peak            ON pack039_energy_monitoring.em_interval_data(is_peak_period) WHERE is_peak_period = true;
CREATE INDEX idx_p039_id_tariff          ON pack039_energy_monitoring.em_interval_data(tariff_period);
CREATE INDEX idx_p039_id_batch           ON pack039_energy_monitoring.em_interval_data(batch_id);
CREATE INDEX idx_p039_id_created         ON pack039_energy_monitoring.em_interval_data(created_at DESC);

-- Composite: meter + date range + quality for validation queries
CREATE INDEX idx_p039_id_meter_quality   ON pack039_energy_monitoring.em_interval_data(meter_id, timestamp DESC, data_quality);

-- Composite: demand for peak detection
CREATE INDEX idx_p039_id_demand_desc     ON pack039_energy_monitoring.em_interval_data(meter_id, demand_kw DESC NULLS LAST)
    WHERE demand_kw IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_id_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_interval_data
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_acquisition_schedules
-- =============================================================================
-- Defines polling and collection schedules for each meter. Controls when
-- and how frequently the acquisition engine reads data from meters,
-- including retry logic, timeout settings, and maintenance windows.
-- Supports both periodic polling and event-driven acquisition modes.

CREATE TABLE pack039_energy_monitoring.em_acquisition_schedules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    schedule_name           VARCHAR(255)    NOT NULL,
    schedule_type           VARCHAR(30)     NOT NULL DEFAULT 'PERIODIC',
    polling_interval_seconds INTEGER        NOT NULL DEFAULT 900,
    cron_expression         VARCHAR(100),
    start_time              TIME,
    end_time                TIME,
    days_of_week            INTEGER[]       DEFAULT '{1,2,3,4,5,6,7}',
    timezone                VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    priority                INTEGER         NOT NULL DEFAULT 5,
    timeout_seconds         INTEGER         NOT NULL DEFAULT 30,
    retry_count             INTEGER         NOT NULL DEFAULT 3,
    retry_delay_seconds     INTEGER         NOT NULL DEFAULT 10,
    retry_backoff_multiplier NUMERIC(4,2)   NOT NULL DEFAULT 2.0,
    max_retry_delay_seconds INTEGER         NOT NULL DEFAULT 300,
    batch_size              INTEGER         DEFAULT 100,
    concurrent_connections  INTEGER         DEFAULT 1,
    maintenance_window_start TIME,
    maintenance_window_end  TIME,
    last_poll_at            TIMESTAMPTZ,
    last_poll_status        VARCHAR(20),
    last_poll_duration_ms   INTEGER,
    last_error_message      TEXT,
    consecutive_failures    INTEGER         NOT NULL DEFAULT 0,
    total_polls             BIGINT          NOT NULL DEFAULT 0,
    total_successes         BIGINT          NOT NULL DEFAULT 0,
    total_failures          BIGINT          NOT NULL DEFAULT 0,
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    schedule_status         VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_as_type CHECK (
        schedule_type IN (
            'PERIODIC', 'CRON', 'EVENT_DRIVEN', 'ON_DEMAND', 'CONTINUOUS'
        )
    ),
    CONSTRAINT chk_p039_as_polling CHECK (
        polling_interval_seconds >= 1 AND polling_interval_seconds <= 86400
    ),
    CONSTRAINT chk_p039_as_priority CHECK (
        priority >= 1 AND priority <= 10
    ),
    CONSTRAINT chk_p039_as_timeout CHECK (
        timeout_seconds >= 1 AND timeout_seconds <= 600
    ),
    CONSTRAINT chk_p039_as_retry CHECK (
        retry_count >= 0 AND retry_count <= 20
    ),
    CONSTRAINT chk_p039_as_retry_delay CHECK (
        retry_delay_seconds >= 1 AND retry_delay_seconds <= 3600
    ),
    CONSTRAINT chk_p039_as_backoff CHECK (
        retry_backoff_multiplier >= 1.0 AND retry_backoff_multiplier <= 10.0
    ),
    CONSTRAINT chk_p039_as_batch CHECK (
        batch_size IS NULL OR (batch_size >= 1 AND batch_size <= 10000)
    ),
    CONSTRAINT chk_p039_as_concurrent CHECK (
        concurrent_connections IS NULL OR (concurrent_connections >= 1 AND concurrent_connections <= 100)
    ),
    CONSTRAINT chk_p039_as_poll_status CHECK (
        last_poll_status IS NULL OR last_poll_status IN (
            'SUCCESS', 'PARTIAL', 'FAILED', 'TIMEOUT', 'SKIPPED', 'MAINTENANCE'
        )
    ),
    CONSTRAINT chk_p039_as_schedule_status CHECK (
        schedule_status IN (
            'ACTIVE', 'PAUSED', 'DISABLED', 'MAINTENANCE', 'ERROR', 'BACKOFF'
        )
    ),
    CONSTRAINT chk_p039_as_failures CHECK (
        consecutive_failures >= 0
    ),
    CONSTRAINT chk_p039_as_totals CHECK (
        total_polls >= 0 AND total_successes >= 0 AND total_failures >= 0
    ),
    CONSTRAINT uq_p039_as_meter_schedule UNIQUE (meter_id, schedule_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_as_meter           ON pack039_energy_monitoring.em_acquisition_schedules(meter_id);
CREATE INDEX idx_p039_as_tenant          ON pack039_energy_monitoring.em_acquisition_schedules(tenant_id);
CREATE INDEX idx_p039_as_type            ON pack039_energy_monitoring.em_acquisition_schedules(schedule_type);
CREATE INDEX idx_p039_as_status          ON pack039_energy_monitoring.em_acquisition_schedules(schedule_status);
CREATE INDEX idx_p039_as_enabled         ON pack039_energy_monitoring.em_acquisition_schedules(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_as_last_poll       ON pack039_energy_monitoring.em_acquisition_schedules(last_poll_at DESC);
CREATE INDEX idx_p039_as_priority        ON pack039_energy_monitoring.em_acquisition_schedules(priority);
CREATE INDEX idx_p039_as_failures        ON pack039_energy_monitoring.em_acquisition_schedules(consecutive_failures DESC);
CREATE INDEX idx_p039_as_created         ON pack039_energy_monitoring.em_acquisition_schedules(created_at DESC);

-- Composite: enabled + next poll due for scheduler
CREATE INDEX idx_p039_as_next_poll       ON pack039_energy_monitoring.em_acquisition_schedules(last_poll_at, polling_interval_seconds)
    WHERE is_enabled = true AND schedule_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_as_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_acquisition_schedules
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_data_buffers
-- =============================================================================
-- Store-and-forward buffer for meter data during network outages or
-- backfill operations. Data is staged here before insertion into the
-- primary interval_data table after validation. Supports deduplication,
-- ordering guarantees, and retry tracking for reliable data delivery.

CREATE TABLE pack039_energy_monitoring.em_data_buffers (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    channel_id              UUID            REFERENCES pack039_energy_monitoring.em_meter_channels(id),
    tenant_id               UUID            NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    raw_value               NUMERIC(20,6)   NOT NULL,
    scaled_value            NUMERIC(18,6),
    engineering_unit        VARCHAR(30),
    buffer_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    buffer_priority         INTEGER         NOT NULL DEFAULT 5,
    source_system           VARCHAR(100),
    acquisition_timestamp   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    processing_attempts     INTEGER         NOT NULL DEFAULT 0,
    max_attempts            INTEGER         NOT NULL DEFAULT 5,
    last_attempt_at         TIMESTAMPTZ,
    last_error              TEXT,
    batch_id                UUID,
    sequence_number         BIGINT,
    is_backfill             BOOLEAN         NOT NULL DEFAULT false,
    duplicate_check_hash    VARCHAR(64),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_db_status CHECK (
        buffer_status IN (
            'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED',
            'DUPLICATE', 'REJECTED', 'EXPIRED'
        )
    ),
    CONSTRAINT chk_p039_db_priority CHECK (
        buffer_priority >= 1 AND buffer_priority <= 10
    ),
    CONSTRAINT chk_p039_db_attempts CHECK (
        processing_attempts >= 0 AND processing_attempts <= max_attempts
    ),
    CONSTRAINT chk_p039_db_max_attempts CHECK (
        max_attempts >= 1 AND max_attempts <= 20
    ),
    CONSTRAINT chk_p039_db_seq CHECK (
        sequence_number IS NULL OR sequence_number >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_db_meter           ON pack039_energy_monitoring.em_data_buffers(meter_id);
CREATE INDEX idx_p039_db_tenant          ON pack039_energy_monitoring.em_data_buffers(tenant_id);
CREATE INDEX idx_p039_db_status          ON pack039_energy_monitoring.em_data_buffers(buffer_status);
CREATE INDEX idx_p039_db_timestamp       ON pack039_energy_monitoring.em_data_buffers(timestamp DESC);
CREATE INDEX idx_p039_db_acq_ts          ON pack039_energy_monitoring.em_data_buffers(acquisition_timestamp DESC);
CREATE INDEX idx_p039_db_priority        ON pack039_energy_monitoring.em_data_buffers(buffer_priority);
CREATE INDEX idx_p039_db_batch           ON pack039_energy_monitoring.em_data_buffers(batch_id);
CREATE INDEX idx_p039_db_dup_hash        ON pack039_energy_monitoring.em_data_buffers(duplicate_check_hash);
CREATE INDEX idx_p039_db_created         ON pack039_energy_monitoring.em_data_buffers(created_at DESC);

-- Composite: pending buffers ordered by priority for processing
CREATE INDEX idx_p039_db_pending_prio    ON pack039_energy_monitoring.em_data_buffers(buffer_priority, acquisition_timestamp)
    WHERE buffer_status = 'PENDING';

-- Composite: meter + timestamp for deduplication checks
CREATE INDEX idx_p039_db_dedup           ON pack039_energy_monitoring.em_data_buffers(meter_id, channel_id, timestamp)
    WHERE buffer_status != 'DUPLICATE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_db_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_data_buffers
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_protocol_configs
-- =============================================================================
-- Protocol-specific configuration for meter communication. Stores detailed
-- connection parameters, register maps, authentication credentials, and
-- protocol tuning settings for each communication protocol. Separates
-- protocol complexity from the meter registry to support multi-protocol
-- environments and protocol upgrades.

CREATE TABLE pack039_energy_monitoring.em_protocol_configs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    protocol                VARCHAR(50)     NOT NULL,
    config_name             VARCHAR(255)    NOT NULL,
    config_version          INTEGER         NOT NULL DEFAULT 1,
    host_address            VARCHAR(255),
    port                    INTEGER,
    slave_id                INTEGER,
    baud_rate               INTEGER,
    parity                  VARCHAR(10),
    stop_bits               INTEGER,
    data_bits               INTEGER,
    connection_timeout_ms   INTEGER         NOT NULL DEFAULT 5000,
    read_timeout_ms         INTEGER         NOT NULL DEFAULT 3000,
    write_timeout_ms        INTEGER         NOT NULL DEFAULT 3000,
    max_registers_per_read  INTEGER         DEFAULT 125,
    inter_poll_delay_ms     INTEGER         DEFAULT 100,
    authentication_type     VARCHAR(30)     DEFAULT 'NONE',
    username                VARCHAR(100),
    password_encrypted      TEXT,
    certificate_id          UUID,
    api_key_encrypted       TEXT,
    bearer_token_encrypted  TEXT,
    register_map            JSONB           DEFAULT '[]',
    custom_parameters       JSONB           DEFAULT '{}',
    tls_enabled             BOOLEAN         NOT NULL DEFAULT false,
    tls_version             VARCHAR(10),
    verify_certificate      BOOLEAN         NOT NULL DEFAULT true,
    keep_alive              BOOLEAN         NOT NULL DEFAULT true,
    keep_alive_interval_s   INTEGER         DEFAULT 60,
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    last_connected_at       TIMESTAMPTZ,
    connection_status       VARCHAR(20)     DEFAULT 'UNKNOWN',
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_pc_protocol CHECK (
        protocol IN (
            'MODBUS_TCP', 'MODBUS_RTU', 'BACNET_IP', 'BACNET_MSTP',
            'LONWORKS', 'OPC_UA', 'OPC_DA', 'MQTT', 'HTTP_REST',
            'SNMP', 'DLMS_COSEM', 'IEC_61850', 'DNP3', 'MANUAL',
            'CSV_IMPORT', 'API', 'PULSE_COUNTER', 'ANALOG_4_20MA'
        )
    ),
    CONSTRAINT chk_p039_pc_port CHECK (
        port IS NULL OR (port >= 1 AND port <= 65535)
    ),
    CONSTRAINT chk_p039_pc_slave CHECK (
        slave_id IS NULL OR (slave_id >= 0 AND slave_id <= 255)
    ),
    CONSTRAINT chk_p039_pc_baud CHECK (
        baud_rate IS NULL OR baud_rate IN (
            1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200
        )
    ),
    CONSTRAINT chk_p039_pc_parity CHECK (
        parity IS NULL OR parity IN ('NONE', 'EVEN', 'ODD', 'MARK', 'SPACE')
    ),
    CONSTRAINT chk_p039_pc_stop_bits CHECK (
        stop_bits IS NULL OR stop_bits IN (1, 2)
    ),
    CONSTRAINT chk_p039_pc_data_bits CHECK (
        data_bits IS NULL OR data_bits IN (7, 8)
    ),
    CONSTRAINT chk_p039_pc_auth CHECK (
        authentication_type IS NULL OR authentication_type IN (
            'NONE', 'BASIC', 'DIGEST', 'BEARER', 'API_KEY',
            'CERTIFICATE', 'OAUTH2', 'NTLM', 'KERBEROS'
        )
    ),
    CONSTRAINT chk_p039_pc_tls_version CHECK (
        tls_version IS NULL OR tls_version IN ('TLS_1_2', 'TLS_1_3')
    ),
    CONSTRAINT chk_p039_pc_conn_status CHECK (
        connection_status IS NULL OR connection_status IN (
            'UNKNOWN', 'CONNECTED', 'DISCONNECTED', 'ERROR',
            'TIMEOUT', 'AUTH_FAILED', 'MAINTENANCE'
        )
    ),
    CONSTRAINT chk_p039_pc_conn_timeout CHECK (
        connection_timeout_ms >= 100 AND connection_timeout_ms <= 60000
    ),
    CONSTRAINT chk_p039_pc_read_timeout CHECK (
        read_timeout_ms >= 100 AND read_timeout_ms <= 60000
    ),
    CONSTRAINT chk_p039_pc_version CHECK (
        config_version >= 1
    ),
    CONSTRAINT uq_p039_pc_meter_protocol UNIQUE (meter_id, protocol, config_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_pc_meter           ON pack039_energy_monitoring.em_protocol_configs(meter_id);
CREATE INDEX idx_p039_pc_tenant          ON pack039_energy_monitoring.em_protocol_configs(tenant_id);
CREATE INDEX idx_p039_pc_protocol        ON pack039_energy_monitoring.em_protocol_configs(protocol);
CREATE INDEX idx_p039_pc_active          ON pack039_energy_monitoring.em_protocol_configs(is_active) WHERE is_active = true;
CREATE INDEX idx_p039_pc_conn_status     ON pack039_energy_monitoring.em_protocol_configs(connection_status);
CREATE INDEX idx_p039_pc_last_conn       ON pack039_energy_monitoring.em_protocol_configs(last_connected_at DESC);
CREATE INDEX idx_p039_pc_auth_type       ON pack039_energy_monitoring.em_protocol_configs(authentication_type);
CREATE INDEX idx_p039_pc_created         ON pack039_energy_monitoring.em_protocol_configs(created_at DESC);
CREATE INDEX idx_p039_pc_register_map    ON pack039_energy_monitoring.em_protocol_configs USING GIN(register_map);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_pc_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_protocol_configs
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_raw_readings
-- =============================================================================
-- Raw, unprocessed meter readings captured directly from the data source
-- before any scaling, validation, or transformation. Provides a complete
-- audit trail of original data for forensic analysis, dispute resolution,
-- and retroactive recalculation. Stored separately from interval_data to
-- avoid polluting the normalized dataset.

CREATE TABLE pack039_energy_monitoring.em_raw_readings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    channel_id              UUID            REFERENCES pack039_energy_monitoring.em_meter_channels(id),
    tenant_id               UUID            NOT NULL,
    reading_timestamp       TIMESTAMPTZ     NOT NULL,
    raw_value               NUMERIC(20,6)   NOT NULL,
    raw_unit                VARCHAR(30),
    register_address        INTEGER,
    register_value_hex      VARCHAR(20),
    communication_status    VARCHAR(20)     NOT NULL DEFAULT 'SUCCESS',
    response_time_ms        INTEGER,
    packet_data             BYTEA,
    error_code              VARCHAR(50),
    error_message           TEXT,
    source_ip               VARCHAR(45),
    source_port             INTEGER,
    protocol                VARCHAR(50),
    acquisition_schedule_id UUID            REFERENCES pack039_energy_monitoring.em_acquisition_schedules(id),
    batch_id                UUID,
    sequence_in_batch       INTEGER,
    is_processed            BOOLEAN         NOT NULL DEFAULT false,
    processed_at            TIMESTAMPTZ,
    interval_data_id        UUID,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_rr_comm_status CHECK (
        communication_status IN (
            'SUCCESS', 'TIMEOUT', 'ERROR', 'CRC_ERROR', 'PARTIAL',
            'NAK', 'NO_RESPONSE', 'AUTH_FAILED', 'PROTOCOL_ERROR'
        )
    ),
    CONSTRAINT chk_p039_rr_response_time CHECK (
        response_time_ms IS NULL OR response_time_ms >= 0
    ),
    CONSTRAINT chk_p039_rr_source_port CHECK (
        source_port IS NULL OR (source_port >= 1 AND source_port <= 65535)
    ),
    CONSTRAINT chk_p039_rr_seq CHECK (
        sequence_in_batch IS NULL OR sequence_in_batch >= 1
    )
);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable for raw readings (skip if extension not available)
-- ---------------------------------------------------------------------------
-- SELECT create_hypertable('pack039_energy_monitoring.em_raw_readings', 'reading_timestamp',
--     chunk_time_interval => INTERVAL '1 week',
--     if_not_exists => TRUE
-- );

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_rr_meter           ON pack039_energy_monitoring.em_raw_readings(meter_id);
CREATE INDEX idx_p039_rr_channel         ON pack039_energy_monitoring.em_raw_readings(channel_id);
CREATE INDEX idx_p039_rr_tenant          ON pack039_energy_monitoring.em_raw_readings(tenant_id);
CREATE INDEX idx_p039_rr_timestamp       ON pack039_energy_monitoring.em_raw_readings(reading_timestamp DESC);
CREATE INDEX idx_p039_rr_meter_ts        ON pack039_energy_monitoring.em_raw_readings(meter_id, reading_timestamp DESC);
CREATE INDEX idx_p039_rr_comm_status     ON pack039_energy_monitoring.em_raw_readings(communication_status);
CREATE INDEX idx_p039_rr_processed       ON pack039_energy_monitoring.em_raw_readings(is_processed) WHERE is_processed = false;
CREATE INDEX idx_p039_rr_batch           ON pack039_energy_monitoring.em_raw_readings(batch_id);
CREATE INDEX idx_p039_rr_schedule        ON pack039_energy_monitoring.em_raw_readings(acquisition_schedule_id);
CREATE INDEX idx_p039_rr_error           ON pack039_energy_monitoring.em_raw_readings(error_code) WHERE error_code IS NOT NULL;
CREATE INDEX idx_p039_rr_created         ON pack039_energy_monitoring.em_raw_readings(created_at DESC);

-- Composite: unprocessed raw readings for processing queue
CREATE INDEX idx_p039_rr_unprocessed     ON pack039_energy_monitoring.em_raw_readings(meter_id, reading_timestamp)
    WHERE is_processed = false AND communication_status = 'SUCCESS';

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_interval_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_acquisition_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_data_buffers ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_protocol_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_raw_readings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_id_tenant_isolation
    ON pack039_energy_monitoring.em_interval_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_id_service_bypass
    ON pack039_energy_monitoring.em_interval_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_as_tenant_isolation
    ON pack039_energy_monitoring.em_acquisition_schedules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_as_service_bypass
    ON pack039_energy_monitoring.em_acquisition_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_db_tenant_isolation
    ON pack039_energy_monitoring.em_data_buffers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_db_service_bypass
    ON pack039_energy_monitoring.em_data_buffers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_pc_tenant_isolation
    ON pack039_energy_monitoring.em_protocol_configs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_pc_service_bypass
    ON pack039_energy_monitoring.em_protocol_configs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_rr_tenant_isolation
    ON pack039_energy_monitoring.em_raw_readings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_rr_service_bypass
    ON pack039_energy_monitoring.em_raw_readings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_interval_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_acquisition_schedules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_data_buffers TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_protocol_configs TO PUBLIC;
GRANT SELECT, INSERT ON pack039_energy_monitoring.em_raw_readings TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_interval_data IS
    'Primary time-series table for normalized interval energy data from all meters. TimescaleDB hypertable candidate with monthly chunks.';
COMMENT ON TABLE pack039_energy_monitoring.em_acquisition_schedules IS
    'Polling and collection schedules for meters with retry logic, timeout settings, and performance tracking.';
COMMENT ON TABLE pack039_energy_monitoring.em_data_buffers IS
    'Store-and-forward buffer for meter data during outages with deduplication, ordering, and retry tracking.';
COMMENT ON TABLE pack039_energy_monitoring.em_protocol_configs IS
    'Protocol-specific connection parameters, register maps, authentication, and tuning settings per meter.';
COMMENT ON TABLE pack039_energy_monitoring.em_raw_readings IS
    'Raw unprocessed meter readings for audit trail, forensic analysis, and retroactive recalculation.';

COMMENT ON COLUMN pack039_energy_monitoring.em_interval_data.value IS 'Normalized measurement value in the specified engineering unit after scaling.';
COMMENT ON COLUMN pack039_energy_monitoring.em_interval_data.data_quality IS 'Quality flag: RAW, VALIDATED, ESTIMATED, INTERPOLATED, CORRECTED, MISSING, SUSPECT, REJECTED.';
COMMENT ON COLUMN pack039_energy_monitoring.em_interval_data.tariff_period IS 'Time-of-use tariff period for cost allocation: ON_PEAK, OFF_PEAK, MID_PEAK, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_interval_data.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_acquisition_schedules.polling_interval_seconds IS 'Interval between successive meter polls in seconds (default 900 = 15 minutes).';
COMMENT ON COLUMN pack039_energy_monitoring.em_acquisition_schedules.retry_backoff_multiplier IS 'Exponential backoff multiplier for retry delays (e.g., 2.0 doubles the delay each retry).';
COMMENT ON COLUMN pack039_energy_monitoring.em_acquisition_schedules.consecutive_failures IS 'Count of consecutive poll failures; triggers alert escalation at configurable thresholds.';

COMMENT ON COLUMN pack039_energy_monitoring.em_data_buffers.buffer_status IS 'Processing state: PENDING, PROCESSING, COMPLETED, FAILED, DUPLICATE, REJECTED, EXPIRED.';
COMMENT ON COLUMN pack039_energy_monitoring.em_data_buffers.duplicate_check_hash IS 'Hash of meter_id + channel_id + timestamp for deduplication.';

COMMENT ON COLUMN pack039_energy_monitoring.em_protocol_configs.register_map IS 'JSON array of register mappings: [{address, type, channel_id, scaling, description}].';
COMMENT ON COLUMN pack039_energy_monitoring.em_protocol_configs.password_encrypted IS 'AES-256 encrypted password for authenticated protocol connections.';

COMMENT ON COLUMN pack039_energy_monitoring.em_raw_readings.packet_data IS 'Raw binary protocol packet data for forensic analysis.';
COMMENT ON COLUMN pack039_energy_monitoring.em_raw_readings.communication_status IS 'Communication result: SUCCESS, TIMEOUT, ERROR, CRC_ERROR, PARTIAL, etc.';
