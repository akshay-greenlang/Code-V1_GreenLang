-- =============================================================================
-- GreenLang Climate OS - Sensor Hypertables
-- =============================================================================
-- File: 04_sensor_hypertables.sql
-- Description: TimescaleDB hypertables for IoT sensor readings and device
--              management with high-frequency data optimizations.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Devices Table (Regular table)
-- -----------------------------------------------------------------------------
-- IoT devices and sensors that collect environmental and operational data.
CREATE TABLE IF NOT EXISTS metrics.devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Organization relationship
    org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,

    -- Device identification
    name VARCHAR(255) NOT NULL,
    device_code VARCHAR(100) NOT NULL,
    serial_number VARCHAR(100),

    -- Device type
    -- Types: energy_meter, gas_analyzer, flow_meter, temperature_sensor, emissions_monitor, weather_station
    device_type VARCHAR(100) NOT NULL,
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    firmware_version VARCHAR(50),

    -- Location information
    location_name VARCHAR(255),
    location_type VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    altitude_meters DOUBLE PRECISION,
    timezone VARCHAR(50) DEFAULT 'UTC',

    -- Location as JSONB for additional details
    location JSONB DEFAULT '{}',

    -- Connection information
    connection_type VARCHAR(50),
    ip_address INET,
    mac_address MACADDR,
    communication_protocol VARCHAR(50),

    -- Calibration data
    calibration_data JSONB DEFAULT '{}',
    last_calibration_date DATE,
    next_calibration_date DATE,

    -- Device configuration
    config JSONB NOT NULL DEFAULT '{}',

    -- Metric types this device reports
    -- Example: ["power_consumption", "gas_flow", "temperature"]
    metric_types VARCHAR(100)[] NOT NULL DEFAULT '{}',

    -- Sampling configuration
    sampling_interval_seconds INTEGER DEFAULT 60,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    last_seen_at TIMESTAMPTZ,
    is_online BOOLEAN DEFAULT false,

    -- Associated emission source (if applicable)
    emission_source_id UUID REFERENCES metrics.emission_sources(id),

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}',
    tags VARCHAR(100)[] DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT devices_org_code_unique UNIQUE (org_id, device_code),
    CONSTRAINT devices_status_valid CHECK (status IN ('active', 'inactive', 'maintenance', 'decommissioned'))
);

CREATE INDEX IF NOT EXISTS idx_devices_org ON metrics.devices(org_id);
CREATE INDEX IF NOT EXISTS idx_devices_type ON metrics.devices(device_type);
CREATE INDEX IF NOT EXISTS idx_devices_status ON metrics.devices(status);
CREATE INDEX IF NOT EXISTS idx_devices_location ON metrics.devices(latitude, longitude)
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_devices_emission_source ON metrics.devices(emission_source_id)
    WHERE emission_source_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_devices_tags ON metrics.devices USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_devices_metric_types ON metrics.devices USING GIN(metric_types);

-- Trigger for updated_at
CREATE TRIGGER devices_updated_at
    BEFORE UPDATE ON metrics.devices
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

COMMENT ON TABLE metrics.devices IS 'IoT devices and sensors collecting environmental and operational data';
COMMENT ON COLUMN metrics.devices.calibration_data IS 'Device calibration parameters and coefficients';
COMMENT ON COLUMN metrics.devices.metric_types IS 'Array of metric types this device reports';

-- -----------------------------------------------------------------------------
-- Sensor Readings Hypertable
-- -----------------------------------------------------------------------------
-- High-frequency time-series data from IoT sensors.
-- Optimized for very frequent writes and time-range queries.
CREATE TABLE IF NOT EXISTS metrics.sensor_readings (
    -- Time is the primary partitioning column
    time TIMESTAMPTZ NOT NULL,

    -- Device reference (denormalized for performance)
    device_id UUID NOT NULL,

    -- Organization (for multi-tenant filtering)
    org_id UUID NOT NULL,

    -- Metric identification
    metric_type VARCHAR(100) NOT NULL,

    -- Measurement value
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50) NOT NULL,

    -- Data quality indicators
    quality VARCHAR(20) NOT NULL DEFAULT 'good',
    quality_code INTEGER,

    -- Raw vs processed flag
    is_raw BOOLEAN NOT NULL DEFAULT true,

    -- For processed readings, reference to raw reading
    raw_reading_time TIMESTAMPTZ,

    -- Processing applied
    processing_applied VARCHAR(100)[],

    -- Device status at time of reading
    device_status VARCHAR(50),

    -- Metadata for additional context
    metadata JSONB DEFAULT '{}',

    -- Data ingestion tracking
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT sensor_readings_quality_valid CHECK (quality IN ('good', 'uncertain', 'bad', 'missing', 'interpolated'))
);

-- Create hypertable with 15 minute chunks
-- Small chunks for high-frequency sensor data (readings every few seconds)
SELECT create_hypertable(
    'metrics.sensor_readings',
    'time',
    chunk_time_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- Primary access pattern: device + time range
CREATE INDEX IF NOT EXISTS idx_sensor_readings_device
    ON metrics.sensor_readings(device_id, time DESC);

-- Organization-wide queries
CREATE INDEX IF NOT EXISTS idx_sensor_readings_org
    ON metrics.sensor_readings(org_id, time DESC);

-- Metric type queries
CREATE INDEX IF NOT EXISTS idx_sensor_readings_metric
    ON metrics.sensor_readings(metric_type, time DESC);

-- Composite index for device + metric type queries
CREATE INDEX IF NOT EXISTS idx_sensor_readings_device_metric
    ON metrics.sensor_readings(device_id, metric_type, time DESC);

-- Quality filtering (find bad readings)
CREATE INDEX IF NOT EXISTS idx_sensor_readings_quality
    ON metrics.sensor_readings(quality, time DESC)
    WHERE quality != 'good';

COMMENT ON TABLE metrics.sensor_readings IS 'High-frequency sensor readings with 15-min chunks, 1-hour compression, 1-year retention';
COMMENT ON COLUMN metrics.sensor_readings.quality IS 'Data quality: good, uncertain, bad, missing, interpolated';

-- -----------------------------------------------------------------------------
-- Device Alerts Table
-- -----------------------------------------------------------------------------
-- Alerts generated from device readings (threshold breaches, anomalies, etc.)
CREATE TABLE IF NOT EXISTS metrics.device_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Time alert was generated
    alert_time TIMESTAMPTZ NOT NULL,

    -- Device reference
    device_id UUID NOT NULL REFERENCES metrics.devices(id) ON DELETE CASCADE,
    org_id UUID NOT NULL,

    -- Alert classification
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'warning',

    -- Alert details
    title VARCHAR(255) NOT NULL,
    description TEXT,

    -- Trigger information
    metric_type VARCHAR(100),
    trigger_value DOUBLE PRECISION,
    threshold_value DOUBLE PRECISION,
    threshold_type VARCHAR(50),

    -- Alert status
    status VARCHAR(50) NOT NULL DEFAULT 'open',
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by UUID,
    resolved_at TIMESTAMPTZ,
    resolved_by UUID,
    resolution_notes TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT device_alerts_severity_valid CHECK (severity IN ('info', 'warning', 'critical', 'emergency')),
    CONSTRAINT device_alerts_status_valid CHECK (status IN ('open', 'acknowledged', 'resolved', 'suppressed'))
);

CREATE INDEX IF NOT EXISTS idx_device_alerts_device ON metrics.device_alerts(device_id, alert_time DESC);
CREATE INDEX IF NOT EXISTS idx_device_alerts_org ON metrics.device_alerts(org_id, alert_time DESC);
CREATE INDEX IF NOT EXISTS idx_device_alerts_status ON metrics.device_alerts(status, alert_time DESC);
CREATE INDEX IF NOT EXISTS idx_device_alerts_severity ON metrics.device_alerts(severity, status, alert_time DESC);

CREATE TRIGGER device_alerts_updated_at
    BEFORE UPDATE ON metrics.device_alerts
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

COMMENT ON TABLE metrics.device_alerts IS 'Alerts generated from device readings (threshold breaches, anomalies)';

-- -----------------------------------------------------------------------------
-- Device Calibration History Table
-- -----------------------------------------------------------------------------
-- Track calibration history for compliance and data quality assurance.
CREATE TABLE IF NOT EXISTS metrics.device_calibration_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    device_id UUID NOT NULL REFERENCES metrics.devices(id) ON DELETE CASCADE,

    -- Calibration details
    calibration_date TIMESTAMPTZ NOT NULL,
    calibration_type VARCHAR(100) NOT NULL,

    -- Calibration data before and after
    previous_calibration JSONB,
    new_calibration JSONB NOT NULL,

    -- Calibration reference standards
    reference_standard VARCHAR(255),
    certificate_number VARCHAR(100),

    -- Calibration results
    passed BOOLEAN NOT NULL,
    deviation_percent DOUBLE PRECISION,

    -- Technician information
    performed_by VARCHAR(255),
    organization VARCHAR(255),

    -- Next calibration
    next_calibration_due DATE,

    -- Notes and documentation
    notes TEXT,
    documentation_url VARCHAR(500),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_calibration_history_device
    ON metrics.device_calibration_history(device_id, calibration_date DESC);

COMMENT ON TABLE metrics.device_calibration_history IS 'Device calibration history for compliance and data quality assurance';

-- -----------------------------------------------------------------------------
-- Helper function to get latest device reading
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION metrics.get_latest_reading(
    p_device_id UUID,
    p_metric_type VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    reading_time TIMESTAMPTZ,
    metric_type VARCHAR,
    value DOUBLE PRECISION,
    unit VARCHAR,
    quality VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        sr.time,
        sr.metric_type,
        sr.value,
        sr.unit,
        sr.quality
    FROM metrics.sensor_readings sr
    WHERE sr.device_id = p_device_id
      AND (p_metric_type IS NULL OR sr.metric_type = p_metric_type)
    ORDER BY sr.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.get_latest_reading IS 'Get the most recent reading for a device, optionally filtered by metric type';

-- -----------------------------------------------------------------------------
-- Helper function to get device statistics
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION metrics.get_device_stats(
    p_device_id UUID,
    p_metric_type VARCHAR,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE (
    reading_count BIGINT,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    avg_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION,
    good_quality_percent DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT,
        MIN(sr.value),
        MAX(sr.value),
        AVG(sr.value),
        STDDEV(sr.value),
        (COUNT(*) FILTER (WHERE sr.quality = 'good')::DOUBLE PRECISION /
         NULLIF(COUNT(*), 0)::DOUBLE PRECISION * 100)
    FROM metrics.sensor_readings sr
    WHERE sr.device_id = p_device_id
      AND sr.metric_type = p_metric_type
      AND sr.time >= p_start_time
      AND sr.time < p_end_time;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.get_device_stats IS 'Get statistical summary of device readings for a time range';
