-- =============================================================================
-- GreenLang Climate OS - Database Schemas
-- =============================================================================
-- File: 01_schemas.sql
-- Description: Create logical database schemas for organizing application data,
--              time-series metrics, audit logs, and archived historical data.
-- =============================================================================

-- public schema: Core application data
-- Contains organizations, users, projects, and configuration data
-- Note: public schema exists by default, we just add a comment
COMMENT ON SCHEMA public IS 'GreenLang Climate OS - Core application data including organizations, users, and projects';

-- metrics schema: Time-series measurement data
-- Contains hypertables for emissions, sensor readings, and calculations
CREATE SCHEMA IF NOT EXISTS metrics;
COMMENT ON SCHEMA metrics IS 'GreenLang Climate OS - Time-series data including emission measurements, sensor readings, and calculation results';

-- audit schema: Audit and compliance logs
-- Contains immutable audit trails for regulatory compliance
CREATE SCHEMA IF NOT EXISTS audit;
COMMENT ON SCHEMA audit IS 'GreenLang Climate OS - Audit logs and API request tracking for compliance and security';

-- archive schema: Historical data storage
-- Contains archived data moved from active tables
CREATE SCHEMA IF NOT EXISTS archive;
COMMENT ON SCHEMA archive IS 'GreenLang Climate OS - Archived historical data for long-term retention and compliance';

-- Set default search path for application queries
-- This ensures queries can reference tables without schema prefix
ALTER DATABASE CURRENT_DATABASE() SET search_path TO public, metrics, audit;

-- Grant usage on schemas to future roles
-- Actual grants will be defined in 10_roles.sql
DO $$
BEGIN
    RAISE NOTICE 'Schemas created: public, metrics, audit, archive';
    RAISE NOTICE 'Search path set to: public, metrics, audit';
END $$;
