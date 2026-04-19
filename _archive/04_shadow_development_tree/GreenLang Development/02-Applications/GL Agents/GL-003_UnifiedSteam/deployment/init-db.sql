-- GL-003 UNIFIEDSTEAM - Database Initialization Script
-- Creates initial schema for local development

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS steam;
CREATE SCHEMA IF NOT EXISTS optimization;
CREATE SCHEMA IF NOT EXISTS audit;

-- ==============================================================================
-- Steam System Data Tables
-- ==============================================================================

-- Steam header readings
CREATE TABLE IF NOT EXISTS steam.header_readings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    header_id VARCHAR(50) NOT NULL,
    pressure DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    flow_rate DOUBLE PRECISION,
    quality DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_header_readings_timestamp ON steam.header_readings(timestamp DESC);
CREATE INDEX idx_header_readings_header_id ON steam.header_readings(header_id);

-- Steam trap data
CREATE TABLE IF NOT EXISTS steam.trap_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trap_id VARCHAR(50) NOT NULL,
    inlet_temperature DOUBLE PRECISION,
    outlet_temperature DOUBLE PRECISION,
    status VARCHAR(20),
    leakage_rate DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trap_data_timestamp ON steam.trap_data(timestamp DESC);
CREATE INDEX idx_trap_data_trap_id ON steam.trap_data(trap_id);

-- Boiler data
CREATE TABLE IF NOT EXISTS steam.boiler_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    boiler_id VARCHAR(50) NOT NULL,
    steam_output DOUBLE PRECISION,
    fuel_consumption DOUBLE PRECISION,
    efficiency DOUBLE PRECISION,
    feedwater_temperature DOUBLE PRECISION,
    stack_temperature DOUBLE PRECISION,
    o2_percentage DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_boiler_data_timestamp ON steam.boiler_data(timestamp DESC);
CREATE INDEX idx_boiler_data_boiler_id ON steam.boiler_data(boiler_id);

-- ==============================================================================
-- Optimization Tables
-- ==============================================================================

-- Optimization results
CREATE TABLE IF NOT EXISTS optimization.results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    optimization_type VARCHAR(50) NOT NULL,
    objective_value DOUBLE PRECISION,
    constraints_satisfied BOOLEAN,
    execution_time_ms INTEGER,
    parameters JSONB,
    recommendations JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_optimization_results_timestamp ON optimization.results(timestamp DESC);
CREATE INDEX idx_optimization_results_type ON optimization.results(optimization_type);

-- Setpoint recommendations
CREATE TABLE IF NOT EXISTS optimization.setpoint_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    equipment_id VARCHAR(50) NOT NULL,
    variable_name VARCHAR(100) NOT NULL,
    current_value DOUBLE PRECISION,
    recommended_value DOUBLE PRECISION,
    expected_savings DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_setpoint_recommendations_timestamp ON optimization.setpoint_recommendations(timestamp DESC);
CREATE INDEX idx_setpoint_recommendations_equipment ON optimization.setpoint_recommendations(equipment_id);

-- ==============================================================================
-- Audit Tables
-- ==============================================================================

-- Audit log
CREATE TABLE IF NOT EXISTS audit.log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(100),
    old_value JSONB,
    new_value JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_log_timestamp ON audit.log(timestamp DESC);
CREATE INDEX idx_audit_log_event_type ON audit.log(event_type);
CREATE INDEX idx_audit_log_user_id ON audit.log(user_id);

-- ==============================================================================
-- Views
-- ==============================================================================

-- Latest header readings view
CREATE OR REPLACE VIEW steam.latest_header_readings AS
SELECT DISTINCT ON (header_id)
    id,
    timestamp,
    header_id,
    pressure,
    temperature,
    flow_rate,
    quality
FROM steam.header_readings
ORDER BY header_id, timestamp DESC;

-- System efficiency view
CREATE OR REPLACE VIEW optimization.system_efficiency AS
SELECT
    DATE_TRUNC('hour', timestamp) AS hour,
    AVG(efficiency) AS avg_efficiency,
    MIN(efficiency) AS min_efficiency,
    MAX(efficiency) AS max_efficiency,
    COUNT(*) AS sample_count
FROM steam.boiler_data
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- ==============================================================================
-- Initial Data (for testing)
-- ==============================================================================

-- Insert test header
INSERT INTO steam.header_readings (header_id, pressure, temperature, flow_rate, quality)
VALUES
    ('HEADER-001', 800000, 175.0, 5000, 0.98),
    ('HEADER-002', 600000, 160.0, 3000, 0.97),
    ('HEADER-003', 400000, 145.0, 2000, 0.96)
ON CONFLICT DO NOTHING;

-- Insert test boiler data
INSERT INTO steam.boiler_data (boiler_id, steam_output, fuel_consumption, efficiency, feedwater_temperature, stack_temperature, o2_percentage)
VALUES
    ('BOILER-001', 10000, 850, 0.88, 105, 180, 3.5),
    ('BOILER-002', 8000, 700, 0.86, 102, 185, 4.0),
    ('BOILER-003', 12000, 980, 0.89, 108, 175, 3.2)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA steam TO unifiedsteam;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA optimization TO unifiedsteam;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO unifiedsteam;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA steam TO unifiedsteam;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA optimization TO unifiedsteam;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA audit TO unifiedsteam;
