-- =============================================================================
-- GreenLang Climate OS - Database Extensions
-- =============================================================================
-- File: 00_extensions.sql
-- Description: Enable required PostgreSQL extensions for TimescaleDB and
--              supporting functionality (crypto, UUID, text search).
-- =============================================================================

-- TimescaleDB: Time-series database extension for hypertables
-- Required for efficient storage and querying of time-series data
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- pg_stat_statements: Query performance monitoring
-- Tracks execution statistics for all SQL statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- pgcrypto: Cryptographic functions
-- Used for password hashing, API key generation, and encryption
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- uuid-ossp: UUID generation functions
-- Provides uuid_generate_v4() for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- pg_trgm: Trigram-based text search
-- Enables similarity searches and fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- pgvector: Vector similarity search (INFRA-005)
-- Enables vector embedding storage, HNSW/IVFFlat indexes, and similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- pgaudit: Audit logging for vector operations (INFRA-005)
-- Tracks write and DDL operations for compliance
CREATE EXTENSION IF NOT EXISTS pgaudit;

-- Verify extensions are installed
DO $$
DECLARE
    ext_name TEXT;
    required_extensions TEXT[] := ARRAY['timescaledb', 'pg_stat_statements', 'pgcrypto', 'uuid-ossp', 'pg_trgm', 'vector'];
BEGIN
    FOREACH ext_name IN ARRAY required_extensions
    LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = ext_name) THEN
            RAISE EXCEPTION 'Required extension % is not installed', ext_name;
        ELSE
            RAISE NOTICE 'Extension % is installed', ext_name;
        END IF;
    END LOOP;
END $$;

-- Log extension versions for audit purposes
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('timescaledb', 'pg_stat_statements', 'pgcrypto', 'uuid-ossp', 'pg_trgm', 'vector')
ORDER BY extname;
