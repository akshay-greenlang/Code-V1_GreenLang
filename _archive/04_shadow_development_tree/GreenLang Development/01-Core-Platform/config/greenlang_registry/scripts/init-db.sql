-- =============================================================================
-- GreenLang Agent Registry - Database Initialization Script
-- =============================================================================
--
-- This script initializes the PostgreSQL database with required extensions
-- and default configuration. It runs automatically when the container starts.
--
-- Usage: Mounted as /docker-entrypoint-initdb.d/init-db.sql
-- =============================================================================

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create additional indexes for full-text search (optional)
-- These will be created by Alembic migrations, but can be pre-created here

-- Grant permissions (if using separate app user)
-- CREATE USER registry_app WITH PASSWORD 'your_password';
-- GRANT ALL PRIVILEGES ON DATABASE greenlang_registry TO registry_app;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'GreenLang Registry database initialized successfully';
END $$;
