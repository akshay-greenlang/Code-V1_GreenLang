-- ============================================================================
-- CSRD/ESRS Platform - PostgreSQL Database Initialization Script
-- ============================================================================
--
-- This script initializes the PostgreSQL database for the CSRD platform.
-- It creates necessary extensions, schemas, and base tables.
--
-- Version: 1.0.0
-- Date: 2025-11-08
-- ============================================================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";       -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";         -- Text similarity search
CREATE EXTENSION IF NOT EXISTS "btree_gin";       -- GIN indexing support
CREATE EXTENSION IF NOT EXISTS "pgcrypto";        -- Cryptographic functions

-- Set timezone to UTC
SET timezone = 'UTC';

-- ============================================================================
-- SCHEMA CREATION
-- ============================================================================

-- Main application schema
CREATE SCHEMA IF NOT EXISTS csrd;

-- Audit and logging schema
CREATE SCHEMA IF NOT EXISTS audit;

-- COMMENT ON SCHEMA
COMMENT ON SCHEMA csrd IS 'CSRD/ESRS reporting platform main schema';
COMMENT ON SCHEMA audit IS 'Audit trail and compliance logging';

-- ============================================================================
-- BASE TABLES
-- ============================================================================

-- Companies table
CREATE TABLE IF NOT EXISTS csrd.companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lei_code VARCHAR(20) UNIQUE,
    name VARCHAR(500) NOT NULL,
    country_code VARCHAR(2),
    sector VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Reporting periods table
CREATE TABLE IF NOT EXISTS csrd.reporting_periods (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID NOT NULL REFERENCES csrd.companies(id) ON DELETE CASCADE,
    reporting_year INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'draft',
    submission_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, reporting_year)
);

-- ESRS data points table
CREATE TABLE IF NOT EXISTS csrd.esrs_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    period_id UUID NOT NULL REFERENCES csrd.reporting_periods(id) ON DELETE CASCADE,
    standard_code VARCHAR(20) NOT NULL,  -- E1, E2, S1, etc.
    metric_id VARCHAR(100) NOT NULL,
    value JSONB,
    unit VARCHAR(50),
    data_quality_score DECIMAL(5,2),
    source VARCHAR(200),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline execution history
CREATE TABLE IF NOT EXISTS csrd.pipeline_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID NOT NULL REFERENCES csrd.companies(id) ON DELETE CASCADE,
    period_id UUID REFERENCES csrd.reporting_periods(id) ON DELETE SET NULL,
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    execution_metadata JSONB
);

-- Audit trail table
CREATE TABLE IF NOT EXISTS audit.activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Companies indexes
CREATE INDEX IF NOT EXISTS idx_companies_lei ON csrd.companies(lei_code);
CREATE INDEX IF NOT EXISTS idx_companies_name ON csrd.companies USING gin(name gin_trgm_ops);

-- Reporting periods indexes
CREATE INDEX IF NOT EXISTS idx_periods_company ON csrd.reporting_periods(company_id);
CREATE INDEX IF NOT EXISTS idx_periods_year ON csrd.reporting_periods(reporting_year);
CREATE INDEX IF NOT EXISTS idx_periods_status ON csrd.reporting_periods(status);

-- ESRS data indexes
CREATE INDEX IF NOT EXISTS idx_esrs_period ON csrd.esrs_data(period_id);
CREATE INDEX IF NOT EXISTS idx_esrs_standard ON csrd.esrs_data(standard_code);
CREATE INDEX IF NOT EXISTS idx_esrs_metric ON csrd.esrs_data(metric_id);
CREATE INDEX IF NOT EXISTS idx_esrs_value ON csrd.esrs_data USING gin(value);

-- Pipeline execution indexes
CREATE INDEX IF NOT EXISTS idx_pipeline_company ON csrd.pipeline_executions(company_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_status ON csrd.pipeline_executions(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_started ON csrd.pipeline_executions(started_at DESC);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit.activity_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit.activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit.activity_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit.activity_log(entity_type, entity_id);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION csrd.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update_updated_at trigger to tables
CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON csrd.companies
    FOR EACH ROW
    EXECUTE FUNCTION csrd.update_updated_at_column();

CREATE TRIGGER update_periods_updated_at
    BEFORE UPDATE ON csrd.reporting_periods
    FOR EACH ROW
    EXECUTE FUNCTION csrd.update_updated_at_column();

CREATE TRIGGER update_esrs_updated_at
    BEFORE UPDATE ON csrd.esrs_data
    FOR EACH ROW
    EXECUTE FUNCTION csrd.update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================================================

-- Insert a sample company
INSERT INTO csrd.companies (lei_code, name, country_code, sector)
VALUES
    ('529900EXAMPLE0000001', 'Example Manufacturing GmbH', 'DE', 'Manufacturing')
ON CONFLICT (lei_code) DO NOTHING;

-- ============================================================================
-- PERMISSIONS AND SECURITY
-- ============================================================================

-- Grant appropriate permissions to csrd_user
GRANT USAGE ON SCHEMA csrd TO csrd_user;
GRANT USAGE ON SCHEMA audit TO csrd_user;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA csrd TO csrd_user;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO csrd_user;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA csrd TO csrd_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO csrd_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA csrd GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO csrd_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT SELECT, INSERT ON TABLES TO csrd_user;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'CSRD Database Initialization Complete';
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'Version: 1.0.0';
    RAISE NOTICE 'Date: %', CURRENT_TIMESTAMP;
    RAISE NOTICE 'Extensions: uuid-ossp, pg_trgm, btree_gin, pgcrypto';
    RAISE NOTICE 'Schemas: csrd, audit';
    RAISE NOTICE 'Tables: 5 main tables, 1 audit table';
    RAISE NOTICE 'Indexes: 14 performance indexes';
    RAISE NOTICE '=============================================================';
END $$;
