-- =============================================================================
-- V004: Sensor Data Tables
-- =============================================================================
-- Description: Creates sensor readings hypertable, device registry,
--              calibration data, and related infrastructure for IoT integration.
-- Author: GreenLang Data Integration Team
-- Requires: TimescaleDB (V002)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Device Types and Status Enums
-- -----------------------------------------------------------------------------

CREATE TYPE public.device_type AS ENUM (
    -- Energy Monitoring
    'electricity_meter',
    'gas_meter',
    'steam_meter',
    'water_meter',

    -- Environmental Sensors
    'co2_sensor',
    'ch4_sensor',
    'temperature_sensor',
    'humidity_sensor',
    'pressure_sensor',
    'air_quality_sensor',

    -- Production Monitoring
    'flow_meter',
    'weight_scale',
    'counter',

    -- Building Automation
    'hvac_sensor',
    'lighting_sensor',
    'occupancy_sensor',

    -- Vehicle/Fleet
    'fuel_sensor',
    'gps_tracker',
    'obd_device',

    -- Industrial
    'plc',
    'scada_endpoint',
    'modbus_device',

    -- Other
    'gateway',
    'aggregator',
    'custom'
);

CREATE TYPE public.device_status AS ENUM (
    'active',
    'inactive',
    'maintenance',
    'offline',
    'decommissioned',
    'pending_installation',
    'error'
);

CREATE TYPE public.calibration_status AS ENUM (
    'valid',
    'due_soon',
    'overdue',
    'in_progress',
    'failed',
    'not_applicable'
);

CREATE TYPE public.reading_quality AS ENUM (
    'good',
    'suspect',
    'bad',
    'missing',
    'interpolated',
    'estimated'
);

-- -----------------------------------------------------------------------------
-- Device Registry Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.device_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Ownership
    organization_id UUID NOT NULL REFERENCES public.organizations(id),
    facility_id UUID,
    asset_id UUID,

    -- Device Identification
    device_name VARCHAR(255) NOT NULL,
    device_code VARCHAR(100) NOT NULL,
    serial_number VARCHAR(100),
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    firmware_version VARCHAR(50),

    -- Classification
    device_type device_type NOT NULL,
    device_category VARCHAR(100),

    -- Communication
    protocol VARCHAR(50),           -- 'modbus', 'mqtt', 'http', 'opc-ua', etc.
    connection_string TEXT,         -- Encrypted connection details
    ip_address INET,
    port INTEGER,
    polling_interval_seconds INTEGER DEFAULT 60,

    -- Location
    location_name VARCHAR(255),
    location_coordinates POINT,     -- lat/long
    floor VARCHAR(50),
    zone VARCHAR(100),

    -- Measurement Configuration
    measurement_unit VARCHAR(50) NOT NULL,
    measurement_type VARCHAR(100) NOT NULL,
    multiplier NUMERIC(20, 10) DEFAULT 1,
    offset_value NUMERIC(20, 10) DEFAULT 0,

    -- Thresholds and Alerts
    min_threshold NUMERIC(20, 6),
    max_threshold NUMERIC(20, 6),
    alert_enabled BOOLEAN DEFAULT TRUE,

    -- Status
    status device_status DEFAULT 'pending_installation',
    last_seen_at TIMESTAMPTZ,
    last_reading_at TIMESTAMPTZ,
    last_error TEXT,
    error_count INTEGER DEFAULT 0,

    -- Commissioning
    installed_at TIMESTAMPTZ,
    commissioned_at TIMESTAMPTZ,
    decommissioned_at TIMESTAMPTZ,
    expected_lifespan_years INTEGER,

    -- Metadata
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,
    configuration JSONB DEFAULT '{}'::jsonb,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by UUID,

    -- Constraints
    CONSTRAINT device_registry_code_unique UNIQUE (organization_id, device_code),
    CONSTRAINT device_registry_thresholds_check
        CHECK (min_threshold IS NULL OR max_threshold IS NULL OR min_threshold <= max_threshold)
);

-- Device registry indexes
CREATE INDEX idx_device_registry_org ON public.device_registry(organization_id);
CREATE INDEX idx_device_registry_facility ON public.device_registry(facility_id) WHERE facility_id IS NOT NULL;
CREATE INDEX idx_device_registry_type ON public.device_registry(device_type);
CREATE INDEX idx_device_registry_status ON public.device_registry(status);
CREATE INDEX idx_device_registry_last_seen ON public.device_registry(last_seen_at DESC);
CREATE INDEX idx_device_registry_tags ON public.device_registry USING gin(tags);
CREATE INDEX idx_device_registry_location ON public.device_registry USING gist(location_coordinates);

-- -----------------------------------------------------------------------------
-- Calibration Data Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.calibration_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Device Reference
    device_id UUID NOT NULL REFERENCES public.device_registry(id) ON DELETE CASCADE,

    -- Calibration Details
    calibration_date TIMESTAMPTZ NOT NULL,
    next_calibration_date TIMESTAMPTZ,
    calibration_interval_days INTEGER DEFAULT 365,

    -- Calibration Results
    status calibration_status NOT NULL DEFAULT 'valid',
    reference_value NUMERIC(20, 10),
    measured_value NUMERIC(20, 10),
    deviation NUMERIC(20, 10),
    deviation_pct NUMERIC(10, 4),
    accuracy_class VARCHAR(20),

    -- Adjustments Made
    adjustment_required BOOLEAN DEFAULT FALSE,
    old_multiplier NUMERIC(20, 10),
    new_multiplier NUMERIC(20, 10),
    old_offset NUMERIC(20, 10),
    new_offset NUMERIC(20, 10),

    -- Calibration Equipment
    reference_equipment VARCHAR(255),
    reference_equipment_cert_id VARCHAR(100),
    reference_equipment_cert_date DATE,

    -- Traceability
    certificate_number VARCHAR(100),
    certificate_url VARCHAR(500),
    calibration_lab VARCHAR(255),
    technician_name VARCHAR(255),
    technician_id VARCHAR(100),

    -- Environmental Conditions
    ambient_temperature NUMERIC(10, 2),
    ambient_humidity NUMERIC(10, 2),
    ambient_pressure NUMERIC(10, 2),

    -- Notes and Documentation
    notes TEXT,
    attachments JSONB DEFAULT '[]'::jsonb,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,

    -- Constraints
    CONSTRAINT calibration_deviation_check
        CHECK (deviation IS NULL OR ABS(deviation_pct) <= 100)
);

-- Calibration indexes
CREATE INDEX idx_calibration_device ON public.calibration_records(device_id);
CREATE INDEX idx_calibration_date ON public.calibration_records(calibration_date DESC);
CREATE INDEX idx_calibration_next_date ON public.calibration_records(next_calibration_date)
    WHERE status = 'valid';
CREATE INDEX idx_calibration_status ON public.calibration_records(status);

-- -----------------------------------------------------------------------------
-- Sensor Readings Hypertable
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS metrics.sensor_readings (
    -- Time dimension (required for hypertable)
    timestamp TIMESTAMPTZ NOT NULL,

    -- Identifiers
    device_id UUID NOT NULL,
    organization_id UUID NOT NULL,

    -- Reading Data
    value NUMERIC(20, 6) NOT NULL,
    unit VARCHAR(50) NOT NULL,
    raw_value NUMERIC(20, 6),           -- Before multiplier/offset

    -- Quality Indicators
    quality reading_quality DEFAULT 'good',
    quality_score INTEGER CHECK (quality_score BETWEEN 0 AND 100),
    quality_flags JSONB DEFAULT '{}'::jsonb,

    -- Derived/Calculated Values
    rate_of_change NUMERIC(20, 6),       -- Change from previous reading
    cumulative_value NUMERIC(20, 6),     -- Running total (for meters)

    -- Environmental Context (optional)
    ambient_temperature NUMERIC(10, 2),
    ambient_humidity NUMERIC(10, 2),

    -- Source
    source_type VARCHAR(50) DEFAULT 'device',  -- 'device', 'manual', 'calculated', 'imported'
    batch_id UUID,                              -- For bulk imports

    -- Processing
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMPTZ,
    emission_measurement_id UUID,        -- Link to emission calculation

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Primary key for hypertable
    PRIMARY KEY (timestamp, device_id, organization_id)
);

-- Convert to hypertable with space partitioning
SELECT create_hypertable(
    'metrics.sensor_readings',
    'timestamp',
    'organization_id',
    4,
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE
);

-- Enable compression
ALTER TABLE metrics.sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id, organization_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy
SELECT add_compression_policy(
    'metrics.sensor_readings',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

-- Add retention policy
SELECT add_retention_policy(
    'metrics.sensor_readings',
    INTERVAL '${retention_days} days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Sensor Readings Indexes
-- -----------------------------------------------------------------------------

CREATE INDEX idx_sensor_readings_device_time
    ON metrics.sensor_readings (device_id, timestamp DESC);

CREATE INDEX idx_sensor_readings_org_time
    ON metrics.sensor_readings (organization_id, timestamp DESC);

CREATE INDEX idx_sensor_readings_quality
    ON metrics.sensor_readings (quality, timestamp DESC)
    WHERE quality != 'good';

CREATE INDEX idx_sensor_readings_unprocessed
    ON metrics.sensor_readings (timestamp DESC)
    WHERE processed = FALSE;

CREATE INDEX idx_sensor_readings_batch
    ON metrics.sensor_readings (batch_id)
    WHERE batch_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Continuous Aggregates for Sensor Data
-- -----------------------------------------------------------------------------

-- Hourly aggregates per device
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.hourly_sensor_readings
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    device_id,
    organization_id,
    COUNT(*) AS reading_count,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    SUM(value) AS sum_value,
    STDDEV(value) AS stddev_value,
    FIRST(value, timestamp) AS first_value,
    LAST(value, timestamp) AS last_value,
    COUNT(*) FILTER (WHERE quality = 'good') AS good_readings,
    COUNT(*) FILTER (WHERE quality != 'good') AS bad_readings
FROM metrics.sensor_readings
GROUP BY time_bucket('1 hour', timestamp), device_id, organization_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'metrics.hourly_sensor_readings',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- Daily aggregates per device
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.daily_sensor_readings
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    device_id,
    organization_id,
    COUNT(*) AS reading_count,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    SUM(value) AS sum_value,
    STDDEV(value) AS stddev_value,
    FIRST(value, timestamp) AS first_value,
    LAST(value, timestamp) AS last_value,
    LAST(value, timestamp) - FIRST(value, timestamp) AS delta_value,  -- For meters
    (COUNT(*) FILTER (WHERE quality = 'good'))::FLOAT / NULLIF(COUNT(*), 0) AS quality_rate
FROM metrics.sensor_readings
GROUP BY time_bucket('1 day', timestamp), device_id, organization_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'metrics.daily_sensor_readings',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Device Alerts Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.device_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Device Reference
    device_id UUID NOT NULL REFERENCES public.device_registry(id) ON DELETE CASCADE,
    organization_id UUID NOT NULL,

    -- Alert Details
    alert_type VARCHAR(50) NOT NULL,     -- 'threshold', 'offline', 'quality', 'calibration'
    severity VARCHAR(20) NOT NULL,        -- 'info', 'warning', 'error', 'critical'
    message TEXT NOT NULL,

    -- Trigger Information
    triggered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    triggered_value NUMERIC(20, 6),
    threshold_value NUMERIC(20, 6),

    -- Resolution
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by UUID,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by UUID,
    resolution_notes TEXT,

    -- Notification
    notifications_sent JSONB DEFAULT '[]'::jsonb,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Device alerts indexes
CREATE INDEX idx_device_alerts_device ON public.device_alerts(device_id);
CREATE INDEX idx_device_alerts_org ON public.device_alerts(organization_id);
CREATE INDEX idx_device_alerts_unresolved ON public.device_alerts(triggered_at DESC)
    WHERE resolved = FALSE;
CREATE INDEX idx_device_alerts_severity ON public.device_alerts(severity)
    WHERE resolved = FALSE;

-- -----------------------------------------------------------------------------
-- Device Health Functions
-- -----------------------------------------------------------------------------

-- Function to check device health
CREATE OR REPLACE FUNCTION public.get_device_health(p_device_id UUID)
RETURNS TABLE (
    device_id UUID,
    device_name VARCHAR(255),
    status device_status,
    last_reading_at TIMESTAMPTZ,
    readings_last_hour BIGINT,
    avg_quality_score NUMERIC,
    calibration_status calibration_status,
    calibration_due_date TIMESTAMPTZ,
    open_alerts_count BIGINT,
    health_score INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id AS device_id,
        d.device_name,
        d.status,
        d.last_reading_at,
        COALESCE(sr.reading_count, 0) AS readings_last_hour,
        COALESCE(sr.avg_quality, 100) AS avg_quality_score,
        COALESCE(cr.status, 'not_applicable'::calibration_status) AS calibration_status,
        cr.next_calibration_date AS calibration_due_date,
        COALESCE(al.open_alerts, 0) AS open_alerts_count,
        -- Calculate health score (0-100)
        GREATEST(0, LEAST(100,
            -- Base score from status
            CASE d.status
                WHEN 'active' THEN 80
                WHEN 'maintenance' THEN 60
                WHEN 'offline' THEN 30
                WHEN 'error' THEN 10
                ELSE 50
            END
            -- Bonus for recent readings
            + CASE WHEN d.last_reading_at > NOW() - INTERVAL '1 hour' THEN 10 ELSE -10 END
            -- Penalty for alerts
            - COALESCE(al.open_alerts, 0) * 5
            -- Penalty for calibration issues
            - CASE WHEN cr.status = 'overdue' THEN 20
                   WHEN cr.status = 'due_soon' THEN 5
                   ELSE 0 END
        ))::INTEGER AS health_score
    FROM public.device_registry d
    LEFT JOIN LATERAL (
        SELECT
            COUNT(*) AS reading_count,
            AVG(quality_score) AS avg_quality
        FROM metrics.sensor_readings r
        WHERE r.device_id = d.id
          AND r.timestamp > NOW() - INTERVAL '1 hour'
    ) sr ON true
    LEFT JOIN LATERAL (
        SELECT c.status, c.next_calibration_date
        FROM public.calibration_records c
        WHERE c.device_id = d.id
        ORDER BY c.calibration_date DESC
        LIMIT 1
    ) cr ON true
    LEFT JOIN LATERAL (
        SELECT COUNT(*) AS open_alerts
        FROM public.device_alerts a
        WHERE a.device_id = d.id
          AND a.resolved = FALSE
    ) al ON true
    WHERE d.id = p_device_id;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get devices requiring attention
CREATE OR REPLACE FUNCTION public.get_devices_requiring_attention(p_organization_id UUID)
RETURNS TABLE (
    device_id UUID,
    device_name VARCHAR(255),
    device_type device_type,
    issue_type VARCHAR(50),
    issue_description TEXT,
    severity VARCHAR(20)
) AS $$
BEGIN
    RETURN QUERY

    -- Devices that are offline
    SELECT
        d.id,
        d.device_name,
        d.device_type,
        'offline'::VARCHAR(50) AS issue_type,
        format('Device offline since %s', d.last_seen_at)::TEXT AS issue_description,
        'warning'::VARCHAR(20) AS severity
    FROM public.device_registry d
    WHERE d.organization_id = p_organization_id
      AND d.status = 'active'
      AND d.last_seen_at < NOW() - INTERVAL '1 hour'

    UNION ALL

    -- Devices with calibration due
    SELECT
        d.id,
        d.device_name,
        d.device_type,
        'calibration'::VARCHAR(50),
        format('Calibration due on %s', cr.next_calibration_date)::TEXT,
        CASE
            WHEN cr.next_calibration_date < NOW() THEN 'error'::VARCHAR(20)
            ELSE 'info'::VARCHAR(20)
        END
    FROM public.device_registry d
    JOIN public.calibration_records cr ON cr.device_id = d.id
    WHERE d.organization_id = p_organization_id
      AND d.status = 'active'
      AND cr.next_calibration_date < NOW() + INTERVAL '30 days'
      AND cr.calibration_date = (
          SELECT MAX(calibration_date)
          FROM public.calibration_records
          WHERE device_id = d.id
      )

    UNION ALL

    -- Devices with errors
    SELECT
        d.id,
        d.device_name,
        d.device_type,
        'error'::VARCHAR(50),
        COALESCE(d.last_error, 'Unknown error')::TEXT,
        'error'::VARCHAR(20)
    FROM public.device_registry d
    WHERE d.organization_id = p_organization_id
      AND d.status = 'error'

    ORDER BY severity DESC, issue_type;
END;
$$ LANGUAGE plpgsql STABLE;

-- -----------------------------------------------------------------------------
-- Data Ingestion Helper Functions
-- -----------------------------------------------------------------------------

-- Function to ingest sensor reading with validation
CREATE OR REPLACE FUNCTION public.ingest_sensor_reading(
    p_device_code VARCHAR(100),
    p_organization_id UUID,
    p_value NUMERIC,
    p_timestamp TIMESTAMPTZ DEFAULT NOW(),
    p_metadata JSONB DEFAULT '{}'::jsonb
) RETURNS UUID AS $$
DECLARE
    v_device RECORD;
    v_quality reading_quality := 'good';
    v_quality_score INTEGER := 100;
    v_adjusted_value NUMERIC;
BEGIN
    -- Get device configuration
    SELECT * INTO v_device
    FROM public.device_registry
    WHERE device_code = p_device_code
      AND organization_id = p_organization_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Device not found: %', p_device_code;
    END IF;

    IF v_device.status NOT IN ('active', 'maintenance') THEN
        RAISE EXCEPTION 'Device is not active: %', v_device.status;
    END IF;

    -- Apply calibration adjustments
    v_adjusted_value := (p_value * v_device.multiplier) + v_device.offset_value;

    -- Check thresholds
    IF v_device.min_threshold IS NOT NULL AND v_adjusted_value < v_device.min_threshold THEN
        v_quality := 'suspect';
        v_quality_score := 50;
    END IF;

    IF v_device.max_threshold IS NOT NULL AND v_adjusted_value > v_device.max_threshold THEN
        v_quality := 'suspect';
        v_quality_score := 50;
    END IF;

    -- Insert reading
    INSERT INTO metrics.sensor_readings (
        timestamp,
        device_id,
        organization_id,
        value,
        unit,
        raw_value,
        quality,
        quality_score,
        metadata
    ) VALUES (
        p_timestamp,
        v_device.id,
        p_organization_id,
        v_adjusted_value,
        v_device.measurement_unit,
        p_value,
        v_quality,
        v_quality_score,
        p_metadata
    );

    -- Update device last seen
    UPDATE public.device_registry
    SET
        last_seen_at = NOW(),
        last_reading_at = p_timestamp,
        error_count = 0
    WHERE id = v_device.id;

    RETURN v_device.id;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Triggers for Device Management
-- -----------------------------------------------------------------------------

-- Trigger to update calibration status
CREATE OR REPLACE FUNCTION public.update_calibration_status()
RETURNS TRIGGER AS $$
BEGIN
    -- Update status based on next calibration date
    IF NEW.next_calibration_date IS NOT NULL THEN
        IF NEW.next_calibration_date < NOW() THEN
            NEW.status := 'overdue';
        ELSIF NEW.next_calibration_date < NOW() + INTERVAL '30 days' THEN
            NEW.status := 'due_soon';
        ELSE
            NEW.status := 'valid';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calibration_status_trigger
    BEFORE INSERT OR UPDATE ON public.calibration_records
    FOR EACH ROW EXECUTE FUNCTION public.update_calibration_status();

-- Trigger to create alert on threshold violation
CREATE OR REPLACE FUNCTION public.check_reading_threshold()
RETURNS TRIGGER AS $$
DECLARE
    v_device RECORD;
BEGIN
    -- Get device thresholds
    SELECT * INTO v_device
    FROM public.device_registry
    WHERE id = NEW.device_id;

    -- Check min threshold
    IF v_device.min_threshold IS NOT NULL AND NEW.value < v_device.min_threshold THEN
        INSERT INTO public.device_alerts (
            device_id,
            organization_id,
            alert_type,
            severity,
            message,
            triggered_value,
            threshold_value
        ) VALUES (
            NEW.device_id,
            NEW.organization_id,
            'threshold',
            'warning',
            format('Value %s below minimum threshold %s', NEW.value, v_device.min_threshold),
            NEW.value,
            v_device.min_threshold
        );
    END IF;

    -- Check max threshold
    IF v_device.max_threshold IS NOT NULL AND NEW.value > v_device.max_threshold THEN
        INSERT INTO public.device_alerts (
            device_id,
            organization_id,
            alert_type,
            severity,
            'warning',
            format('Value %s above maximum threshold %s', NEW.value, v_device.max_threshold),
            NEW.value,
            v_device.max_threshold
        ) VALUES (
            NEW.device_id,
            NEW.organization_id,
            'threshold',
            'warning',
            format('Value %s above maximum threshold %s', NEW.value, v_device.max_threshold),
            NEW.value,
            v_device.max_threshold
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Note: Threshold trigger disabled by default due to performance impact
-- Enable with: CREATE TRIGGER sensor_threshold_trigger AFTER INSERT ON metrics.sensor_readings
--              FOR EACH ROW EXECUTE FUNCTION public.check_reading_threshold();

-- -----------------------------------------------------------------------------
-- Row Level Security
-- -----------------------------------------------------------------------------

ALTER TABLE public.device_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.calibration_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.device_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics.sensor_readings ENABLE ROW LEVEL SECURITY;

-- Device registry policies
CREATE POLICY device_registry_select_policy ON public.device_registry
    FOR SELECT
    USING (
        organization_id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
        )
    );

-- Sensor readings policies
CREATE POLICY sensor_readings_select_policy ON metrics.sensor_readings
    FOR SELECT
    USING (
        organization_id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
        )
    );

-- -----------------------------------------------------------------------------
-- Comments
-- -----------------------------------------------------------------------------

COMMENT ON TABLE public.device_registry IS 'Registry of all IoT devices and sensors connected to the platform';
COMMENT ON TABLE public.calibration_records IS 'Calibration history and certificates for metering devices';
COMMENT ON TABLE metrics.sensor_readings IS 'Time-series hypertable storing raw sensor readings';
COMMENT ON TABLE public.device_alerts IS 'Alerts generated from device monitoring';

COMMENT ON MATERIALIZED VIEW metrics.hourly_sensor_readings IS 'Continuous aggregate: Hourly sensor reading statistics';
COMMENT ON MATERIALIZED VIEW metrics.daily_sensor_readings IS 'Continuous aggregate: Daily sensor reading statistics';

COMMENT ON FUNCTION public.get_device_health(UUID) IS 'Returns comprehensive health status for a device';
COMMENT ON FUNCTION public.get_devices_requiring_attention(UUID) IS 'Returns list of devices that need attention';
COMMENT ON FUNCTION public.ingest_sensor_reading IS 'Ingests a sensor reading with validation and calibration adjustments';
