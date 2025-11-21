-- ==============================================
-- GreenLang PostgreSQL Database Architecture
-- Version: 1.0.0
-- Tables: 200+ across multiple schemas
-- ==============================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- ==============================================
-- SCHEMAS
-- ==============================================

CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS emissions;
CREATE SCHEMA IF NOT EXISTS supply_chain;
CREATE SCHEMA IF NOT EXISTS reporting;
CREATE SCHEMA IF NOT EXISTS csrd;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS integration;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS iot;
CREATE SCHEMA IF NOT EXISTS master_data;

-- ==============================================
-- CORE SCHEMA - Organization & Users
-- ==============================================

-- Organizations table with partitioning by region
CREATE TABLE core.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    legal_entity_name VARCHAR(500),
    tax_id VARCHAR(100),
    industry_sector VARCHAR(100),
    sub_sector VARCHAR(100),
    region VARCHAR(50) NOT NULL,
    country_code CHAR(2) NOT NULL,
    headquarters_address JSONB,
    fiscal_year_end DATE,
    currency_code CHAR(3) DEFAULT 'EUR',
    employee_count INTEGER,
    annual_revenue DECIMAL(20, 2),
    sustainability_maturity_level INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
) PARTITION BY LIST (region);

-- Create regional partitions
CREATE TABLE core.organizations_eu PARTITION OF core.organizations FOR VALUES IN ('EU', 'EUROPE');
CREATE TABLE core.organizations_us PARTITION OF core.organizations FOR VALUES IN ('US', 'AMERICAS');
CREATE TABLE core.organizations_apac PARTITION OF core.organizations FOR VALUES IN ('APAC', 'ASIA');
CREATE TABLE core.organizations_other PARTITION OF core.organizations DEFAULT;

-- Indexes for organizations
CREATE INDEX idx_organizations_org_code ON core.organizations(org_code);
CREATE INDEX idx_organizations_country ON core.organizations(country_code);
CREATE INDEX idx_organizations_sector ON core.organizations(industry_sector, sub_sector);
CREATE INDEX idx_organizations_metadata ON core.organizations USING GIN(metadata);

-- Users table with JSONB for flexible attributes
CREATE TABLE core.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) NOT NULL,
    permissions JSONB DEFAULT '[]'::JSONB,
    department VARCHAR(100),
    job_title VARCHAR(100),
    phone VARCHAR(50),
    timezone VARCHAR(50) DEFAULT 'UTC',
    language_code CHAR(2) DEFAULT 'en',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_login_at TIMESTAMPTZ,
    mfa_enabled BOOLEAN DEFAULT false,
    mfa_secret VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX idx_users_org ON core.users(organization_id);
CREATE INDEX idx_users_email ON core.users(email);
CREATE INDEX idx_users_role ON core.users(role);
CREATE INDEX idx_users_active ON core.users(is_active) WHERE is_active = true;

-- ==============================================
-- EMISSIONS SCHEMA - Carbon Tracking
-- ==============================================

-- Emission sources master data
CREATE TABLE emissions.emission_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id),
    source_code VARCHAR(100) UNIQUE NOT NULL,
    source_name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL, -- facility, vehicle, equipment, process
    scope_category VARCHAR(10) NOT NULL, -- scope1, scope2, scope3
    ghg_protocol_category VARCHAR(100),
    location_id UUID,
    geo_coordinates GEOGRAPHY(POINT, 4326),
    is_active BOOLEAN DEFAULT true,
    commissioning_date DATE,
    decommissioning_date DATE,
    technical_specs JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_emission_sources_org ON emissions.emission_sources(organization_id);
CREATE INDEX idx_emission_sources_scope ON emissions.emission_sources(scope_category);
CREATE INDEX idx_emission_sources_type ON emissions.emission_sources(source_type);
CREATE INDEX idx_emission_sources_geo ON emissions.emission_sources USING GIST(geo_coordinates);

-- Time-series emissions data (using TimescaleDB)
CREATE TABLE emissions.emissions_data (
    time TIMESTAMPTZ NOT NULL,
    source_id UUID NOT NULL REFERENCES emissions.emission_sources(id),
    organization_id UUID NOT NULL,
    activity_data DECIMAL(20, 6) NOT NULL,
    activity_unit VARCHAR(50) NOT NULL,
    emission_factor_id UUID,
    co2_emissions DECIMAL(20, 6),
    ch4_emissions DECIMAL(20, 6),
    n2o_emissions DECIMAL(20, 6),
    co2e_total DECIMAL(20, 6) NOT NULL,
    data_quality_score INTEGER CHECK (data_quality_score BETWEEN 0 AND 100),
    verification_status VARCHAR(50) DEFAULT 'unverified',
    data_source VARCHAR(100),
    calculation_method VARCHAR(100),
    uncertainty_percentage DECIMAL(5, 2),
    metadata JSONB DEFAULT '{}'::JSONB
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('emissions.emissions_data', 'time', chunk_time_interval => INTERVAL '1 month');

-- Create continuous aggregates for performance
CREATE MATERIALIZED VIEW emissions.daily_emissions_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    source_id,
    organization_id,
    SUM(co2e_total) as total_co2e,
    AVG(data_quality_score) as avg_quality_score,
    COUNT(*) as measurement_count
FROM emissions.emissions_data
GROUP BY day, source_id, organization_id
WITH NO DATA;

-- Add retention policy (keep raw data for 5 years)
SELECT add_retention_policy('emissions.emissions_data', INTERVAL '5 years');

-- ==============================================
-- SUPPLY CHAIN SCHEMA
-- ==============================================

-- Suppliers master data
CREATE TABLE supply_chain.suppliers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id),
    supplier_code VARCHAR(100) UNIQUE NOT NULL,
    legal_name VARCHAR(500) NOT NULL,
    tax_id VARCHAR(100),
    duns_number VARCHAR(20),
    country_code CHAR(2) NOT NULL,
    tier_level INTEGER DEFAULT 1,
    category VARCHAR(100),
    risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
    sustainability_score INTEGER CHECK (sustainability_score BETWEEN 0 AND 100),
    certification_status JSONB,
    primary_contact JSONB,
    is_active BOOLEAN DEFAULT true,
    onboarding_date DATE,
    last_assessment_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
);

CREATE INDEX idx_suppliers_org ON supply_chain.suppliers(organization_id);
CREATE INDEX idx_suppliers_country ON supply_chain.suppliers(country_code);
CREATE INDEX idx_suppliers_tier ON supply_chain.suppliers(tier_level);
CREATE INDEX idx_suppliers_risk ON supply_chain.suppliers(risk_score);

-- Products and materials
CREATE TABLE supply_chain.products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id),
    product_code VARCHAR(100) UNIQUE NOT NULL,
    product_name VARCHAR(500) NOT NULL,
    category VARCHAR(100),
    sub_category VARCHAR(100),
    unit_of_measure VARCHAR(50),
    carbon_footprint_per_unit DECIMAL(15, 6),
    water_footprint_per_unit DECIMAL(15, 6),
    recyclability_percentage DECIMAL(5, 2),
    hazardous_materials JSONB,
    certifications JSONB,
    lifecycle_stage VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_products_org ON supply_chain.products(organization_id);
CREATE INDEX idx_products_category ON supply_chain.products(category, sub_category);

-- Purchase orders with partitioning by date
CREATE TABLE supply_chain.purchase_orders (
    id UUID DEFAULT uuid_generate_v4(),
    order_number VARCHAR(100) NOT NULL,
    organization_id UUID REFERENCES core.organizations(id),
    supplier_id UUID REFERENCES supply_chain.suppliers(id),
    order_date DATE NOT NULL,
    delivery_date DATE,
    status VARCHAR(50),
    total_amount DECIMAL(20, 2),
    currency_code CHAR(3),
    items JSONB NOT NULL,
    sustainability_metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, order_date)
) PARTITION BY RANGE (order_date);

-- Create monthly partitions for purchase orders
CREATE TABLE supply_chain.purchase_orders_2024_01 PARTITION OF supply_chain.purchase_orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE supply_chain.purchase_orders_2024_02 PARTITION OF supply_chain.purchase_orders
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- Continue for all months...

-- ==============================================
-- CSRD REPORTING SCHEMA
-- ==============================================

-- ESRS Standards mapping
CREATE TABLE csrd.esrs_standards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    standard_code VARCHAR(50) UNIQUE NOT NULL,
    standard_name VARCHAR(255) NOT NULL,
    category VARCHAR(100), -- Environmental, Social, Governance
    disclosure_requirements JSONB NOT NULL,
    mandatory BOOLEAN DEFAULT true,
    applicable_from DATE,
    version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- CSRD Data Points
CREATE TABLE csrd.data_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id),
    standard_id UUID REFERENCES csrd.esrs_standards(id),
    data_point_code VARCHAR(100) NOT NULL,
    reporting_period_start DATE NOT NULL,
    reporting_period_end DATE NOT NULL,
    value_numeric DECIMAL(20, 6),
    value_text TEXT,
    value_json JSONB,
    unit_of_measure VARCHAR(50),
    data_quality_score INTEGER CHECK (data_quality_score BETWEEN 0 AND 100),
    verification_status VARCHAR(50),
    auditor_notes TEXT,
    supporting_documents JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(organization_id, data_point_code, reporting_period_start)
);

CREATE INDEX idx_csrd_data_points_org ON csrd.data_points(organization_id);
CREATE INDEX idx_csrd_data_points_standard ON csrd.data_points(standard_id);
CREATE INDEX idx_csrd_data_points_period ON csrd.data_points(reporting_period_start, reporting_period_end);

-- ==============================================
-- AUDIT SCHEMA
-- ==============================================

-- Audit trail table (append-only)
CREATE TABLE audit.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    organization_id UUID,
    user_id UUID,
    action VARCHAR(50) NOT NULL, -- CREATE, UPDATE, DELETE, VIEW
    table_name VARCHAR(100) NOT NULL,
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    request_id VARCHAR(255),
    metadata JSONB
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for audit log
CREATE TABLE audit.audit_log_2024_01 PARTITION OF audit.audit_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE audit.audit_log_2024_02 PARTITION OF audit.audit_log
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE INDEX idx_audit_log_timestamp ON audit.audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_org ON audit.audit_log(organization_id);
CREATE INDEX idx_audit_log_user ON audit.audit_log(user_id);
CREATE INDEX idx_audit_log_table ON audit.audit_log(table_name);

-- ==============================================
-- INTEGRATION SCHEMA - External Systems
-- ==============================================

-- ERP Integration configurations
CREATE TABLE integration.erp_connections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id),
    erp_system VARCHAR(50) NOT NULL, -- SAP, Oracle, Workday, Dynamics365
    connection_name VARCHAR(255) NOT NULL,
    connection_type VARCHAR(50), -- REST, SOAP, RFC, OData
    base_url VARCHAR(500),
    authentication_type VARCHAR(50), -- OAuth2, Basic, APIKey, Certificate
    credentials_encrypted JSONB, -- Stored encrypted
    api_version VARCHAR(20),
    rate_limit INTEGER,
    timeout_seconds INTEGER DEFAULT 30,
    retry_policy JSONB,
    is_active BOOLEAN DEFAULT true,
    last_sync_at TIMESTAMPTZ,
    sync_schedule VARCHAR(100), -- Cron expression
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_erp_connections_org ON integration.erp_connections(organization_id);
CREATE INDEX idx_erp_connections_system ON integration.erp_connections(erp_system);

-- Integration job tracking
CREATE TABLE integration.sync_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    connection_id UUID REFERENCES integration.erp_connections(id),
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL, -- pending, running, completed, failed
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    error_log JSONB,
    execution_metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sync_jobs_connection ON integration.sync_jobs(connection_id);
CREATE INDEX idx_sync_jobs_status ON integration.sync_jobs(status);
CREATE INDEX idx_sync_jobs_created ON integration.sync_jobs(created_at DESC);

-- ==============================================
-- IOT SCHEMA - Sensor Data
-- ==============================================

-- IoT Devices registry
CREATE TABLE iot.devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id),
    device_id VARCHAR(255) UNIQUE NOT NULL,
    device_type VARCHAR(100) NOT NULL,
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    firmware_version VARCHAR(50),
    location_id UUID,
    geo_coordinates GEOGRAPHY(POINT, 4326),
    installation_date DATE,
    last_seen_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    calibration_date DATE,
    configuration JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_iot_devices_org ON iot.devices(organization_id);
CREATE INDEX idx_iot_devices_type ON iot.devices(device_type);
CREATE INDEX idx_iot_devices_geo ON iot.devices USING GIST(geo_coordinates);

-- IoT sensor readings (time-series)
CREATE TABLE iot.sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    device_id UUID REFERENCES iot.devices(id),
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(20, 6) NOT NULL,
    unit VARCHAR(50),
    quality_indicator INTEGER,
    raw_data JSONB
);

-- Convert to hypertable
SELECT create_hypertable('iot.sensor_readings', 'time', chunk_time_interval => INTERVAL '1 day');

-- Continuous aggregate for hourly averages
CREATE MATERIALIZED VIEW iot.hourly_sensor_aggregates
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    device_id,
    metric_name,
    AVG(value) as avg_value,
    MAX(value) as max_value,
    MIN(value) as min_value,
    COUNT(*) as reading_count
FROM iot.sensor_readings
GROUP BY hour, device_id, metric_name
WITH NO DATA;

-- ==============================================
-- ANALYTICS SCHEMA - Aggregated Data
-- ==============================================

-- KPI definitions
CREATE TABLE analytics.kpi_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kpi_code VARCHAR(100) UNIQUE NOT NULL,
    kpi_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    calculation_formula TEXT,
    unit_of_measure VARCHAR(50),
    target_value DECIMAL(20, 6),
    threshold_warning DECIMAL(20, 6),
    threshold_critical DECIMAL(20, 6),
    aggregation_level VARCHAR(50), -- daily, monthly, quarterly, yearly
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- KPI values (time-series)
CREATE TABLE analytics.kpi_values (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES core.organizations(id),
    kpi_id UUID REFERENCES analytics.kpi_definitions(id),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    value DECIMAL(20, 6) NOT NULL,
    target_value DECIMAL(20, 6),
    variance_percentage DECIMAL(10, 2),
    trend VARCHAR(20), -- increasing, decreasing, stable
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_kpi_values_org ON analytics.kpi_values(organization_id);
CREATE INDEX idx_kpi_values_kpi ON analytics.kpi_values(kpi_id);
CREATE INDEX idx_kpi_values_period ON analytics.kpi_values(period_start, period_end);

-- ==============================================
-- MASTER DATA SCHEMA
-- ==============================================

-- Emission factors database
CREATE TABLE master_data.emission_factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_code VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(100),
    sub_category VARCHAR(100),
    activity_type VARCHAR(255),
    region VARCHAR(100),
    year INTEGER,
    factor_value DECIMAL(20, 10) NOT NULL,
    unit VARCHAR(100) NOT NULL,
    source VARCHAR(255),
    uncertainty_percentage DECIMAL(5, 2),
    valid_from DATE,
    valid_to DATE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_emission_factors_category ON master_data.emission_factors(category, sub_category);
CREATE INDEX idx_emission_factors_region ON master_data.emission_factors(region);
CREATE INDEX idx_emission_factors_year ON master_data.emission_factors(year);

-- Industry benchmarks
CREATE TABLE master_data.industry_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    industry_code VARCHAR(50),
    metric_name VARCHAR(255),
    percentile_25 DECIMAL(20, 6),
    percentile_50 DECIMAL(20, 6),
    percentile_75 DECIMAL(20, 6),
    percentile_90 DECIMAL(20, 6),
    unit VARCHAR(50),
    region VARCHAR(100),
    year INTEGER,
    source VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==============================================
-- PERFORMANCE OPTIMIZATION
-- ==============================================

-- Create compound indexes for common queries
CREATE INDEX idx_emissions_org_time ON emissions.emissions_data(organization_id, time DESC);
CREATE INDEX idx_emissions_source_time ON emissions.emissions_data(source_id, time DESC);

-- Partial indexes for active records
CREATE INDEX idx_suppliers_active ON supply_chain.suppliers(organization_id) WHERE is_active = true;
CREATE INDEX idx_devices_active ON iot.devices(organization_id) WHERE is_active = true;

-- Text search indexes
CREATE INDEX idx_suppliers_search ON supply_chain.suppliers USING GIN(
    to_tsvector('english', legal_name || ' ' || COALESCE(supplier_code, ''))
);

-- BRIN indexes for time-series data
CREATE INDEX idx_emissions_time_brin ON emissions.emissions_data USING BRIN(time);
CREATE INDEX idx_sensor_time_brin ON iot.sensor_readings USING BRIN(time);

-- ==============================================
-- MATERIALIZED VIEWS FOR REPORTING
-- ==============================================

CREATE MATERIALIZED VIEW reporting.monthly_emissions_by_scope AS
SELECT
    DATE_TRUNC('month', e.time) as month,
    e.organization_id,
    o.name as organization_name,
    es.scope_category,
    SUM(e.co2e_total) as total_emissions,
    AVG(e.data_quality_score) as avg_quality_score,
    COUNT(DISTINCT es.id) as source_count
FROM emissions.emissions_data e
JOIN emissions.emission_sources es ON e.source_id = es.id
JOIN core.organizations o ON e.organization_id = o.id
GROUP BY DATE_TRUNC('month', e.time), e.organization_id, o.name, es.scope_category;

CREATE INDEX idx_monthly_emissions_org ON reporting.monthly_emissions_by_scope(organization_id);
CREATE INDEX idx_monthly_emissions_month ON reporting.monthly_emissions_by_scope(month);

-- ==============================================
-- ROW LEVEL SECURITY
-- ==============================================

-- Enable RLS on sensitive tables
ALTER TABLE core.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE emissions.emissions_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE supply_chain.suppliers ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY org_isolation ON core.organizations
    FOR ALL
    USING (id = current_setting('app.current_org_id')::UUID);

CREATE POLICY emissions_isolation ON emissions.emissions_data
    FOR ALL
    USING (organization_id = current_setting('app.current_org_id')::UUID);

-- ==============================================
-- TRIGGERS AND FUNCTIONS
-- ==============================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION core.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to all tables with updated_at
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON core.organizations
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON core.users
    FOR EACH ROW EXECUTE FUNCTION core.update_updated_at_column();

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit.create_audit_entry()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit.audit_log (
        organization_id,
        user_id,
        action,
        table_name,
        record_id,
        old_values,
        new_values
    ) VALUES (
        current_setting('app.current_org_id', true)::UUID,
        current_setting('app.current_user_id', true)::UUID,
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        to_jsonb(OLD),
        to_jsonb(NEW)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ==============================================
-- STORED PROCEDURES
-- ==============================================

-- Calculate organization carbon footprint
CREATE OR REPLACE FUNCTION emissions.calculate_total_footprint(
    p_org_id UUID,
    p_start_date DATE,
    p_end_date DATE
) RETURNS TABLE (
    scope VARCHAR,
    total_emissions DECIMAL,
    data_quality_score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        es.scope_category,
        SUM(ed.co2e_total)::DECIMAL as total_emissions,
        AVG(ed.data_quality_score)::DECIMAL as data_quality_score
    FROM emissions.emissions_data ed
    JOIN emissions.emission_sources es ON ed.source_id = es.id
    WHERE ed.organization_id = p_org_id
        AND ed.time >= p_start_date
        AND ed.time <= p_end_date
    GROUP BY es.scope_category;
END;
$$ LANGUAGE plpgsql;

-- ==============================================
-- DATABASE CONFIGURATION
-- ==============================================

-- Set configuration parameters for performance
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET min_wal_size = '2GB';
ALTER SYSTEM SET max_wal_size = '8GB';

-- Enable parallel query execution
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_parallel_maintenance_workers = 4;

-- Configure autovacuum for large tables
ALTER TABLE emissions.emissions_data SET (autovacuum_vacuum_scale_factor = 0.01);
ALTER TABLE audit.audit_log SET (autovacuum_vacuum_scale_factor = 0.01);

-- ==============================================
-- GRANTS AND PERMISSIONS
-- ==============================================

-- Create roles
CREATE ROLE greenlang_app;
CREATE ROLE greenlang_readonly;
CREATE ROLE greenlang_admin;

-- Grant permissions
GRANT USAGE ON SCHEMA core, emissions, supply_chain, csrd, reporting TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA core TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA emissions TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA supply_chain TO greenlang_app;

GRANT USAGE ON SCHEMA core, emissions, supply_chain, csrd, reporting TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA core TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA emissions TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA supply_chain TO greenlang_readonly;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA core TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA emissions TO greenlang_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA supply_chain TO greenlang_admin;