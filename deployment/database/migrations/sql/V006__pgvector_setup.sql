-- =============================================================================
-- GreenLang Climate OS - pgvector Vector Database Infrastructure
-- =============================================================================
-- File: V006__pgvector_setup.sql
-- PRD: INFRA-005 Vector Database Infrastructure with pgvector
-- Description: Enable pgvector extension, create vector embedding storage
--              tables, HNSW/IVFFlat indexes, RBAC roles, audit logging,
--              and partitioned tables for multi-tenant isolation.
-- =============================================================================

-- ============================================================================
-- 1. Enable pgvector Extension
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- Verify pgvector is installed and get version
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'pgvector extension failed to install';
    ELSE
        RAISE NOTICE 'pgvector extension installed successfully';
    END IF;
END $$;

-- Enable pgaudit for vector operation auditing
CREATE EXTENSION IF NOT EXISTS pgaudit;

-- ============================================================================
-- 2. Core Tables
-- ============================================================================

-- Embedding collections (logical groupings)
CREATE TABLE IF NOT EXISTS embedding_collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL UNIQUE,
    description TEXT,
    namespace VARCHAR(100) NOT NULL DEFAULT 'default',
    embedding_model VARCHAR(100) NOT NULL,
    dimensions INTEGER NOT NULL,
    distance_metric VARCHAR(20) NOT NULL DEFAULT 'cosine',
    metadata JSONB NOT NULL DEFAULT '{}',
    vector_count BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vector embeddings table (primary storage)
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Source reference
    source_type VARCHAR(50) NOT NULL,       -- 'document', 'regulation', 'report', 'policy'
    source_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,

    -- Content
    content_hash VARCHAR(64) NOT NULL,      -- SHA-256 of content
    content_preview TEXT,                    -- First 500 chars

    -- Vector embedding (384-dim MiniLM default)
    embedding vector(384) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}',
    namespace VARCHAR(100) NOT NULL DEFAULT 'default',
    collection_id UUID REFERENCES embedding_collections(id) ON DELETE SET NULL,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_source_chunk UNIQUE (source_type, source_id, chunk_index, embedding_model)
);

-- Partitioned vector embeddings table (for multi-tenant isolation)
CREATE TABLE IF NOT EXISTS vector_embeddings_partitioned (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    source_type VARCHAR(50) NOT NULL,
    source_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content_hash VARCHAR(64) NOT NULL,
    content_preview TEXT,
    embedding vector(384) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    metadata JSONB NOT NULL DEFAULT '{}',
    namespace VARCHAR(100) NOT NULL DEFAULT 'default',
    collection_id UUID REFERENCES embedding_collections(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, namespace),
    CONSTRAINT unique_partitioned_source_chunk UNIQUE (source_type, source_id, chunk_index, embedding_model, namespace)
) PARTITION BY LIST (namespace);

-- Create partitions per GreenLang application
CREATE TABLE IF NOT EXISTS vector_embeddings_csrd PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('csrd');
CREATE TABLE IF NOT EXISTS vector_embeddings_cbam PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('cbam');
CREATE TABLE IF NOT EXISTS vector_embeddings_eudr PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('eudr');
CREATE TABLE IF NOT EXISTS vector_embeddings_vcci PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('vcci');
CREATE TABLE IF NOT EXISTS vector_embeddings_sb253 PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('sb253');
CREATE TABLE IF NOT EXISTS vector_embeddings_taxonomy PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('taxonomy');
CREATE TABLE IF NOT EXISTS vector_embeddings_csddd PARTITION OF vector_embeddings_partitioned
    FOR VALUES IN ('csddd');
CREATE TABLE IF NOT EXISTS vector_embeddings_default PARTITION OF vector_embeddings_partitioned
    DEFAULT;

-- Embedding generation jobs
CREATE TABLE IF NOT EXISTS embedding_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID REFERENCES embedding_collections(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    source_type VARCHAR(50) NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 0,
    processed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    error_details JSONB,
    batch_size INTEGER NOT NULL DEFAULT 1000,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_job_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- Search analytics / query logging
CREATE TABLE IF NOT EXISTS vector_search_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_embedding vector(384),
    query_text TEXT,
    search_type VARCHAR(20) NOT NULL DEFAULT 'similarity',
    namespace VARCHAR(100),
    top_k INTEGER,
    threshold FLOAT,
    result_count INTEGER,
    latency_ms INTEGER,
    ef_search_used INTEGER,
    user_id UUID,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_search_type CHECK (search_type IN ('similarity', 'filtered', 'hybrid', 'batch'))
);

-- Vector audit log
CREATE TABLE IF NOT EXISTS vector_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation VARCHAR(10) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id UUID,
    old_data JSONB,
    new_data JSONB,
    changed_by VARCHAR(100) NOT NULL DEFAULT current_user,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- 3. Indexes
-- ============================================================================

-- HNSW index for cosine similarity search (primary)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_hnsw_cosine
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Filtered search indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_namespace
ON vector_embeddings (namespace);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_source
ON vector_embeddings (source_type, source_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_created
ON vector_embeddings (created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_model
ON vector_embeddings (embedding_model);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_collection
ON vector_embeddings (collection_id) WHERE collection_id IS NOT NULL;

-- Metadata GIN index for JSON filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_metadata
ON vector_embeddings USING GIN (metadata jsonb_path_ops);

-- Full-text search index on content_preview for hybrid search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_content_fts
ON vector_embeddings USING GIN (to_tsvector('english', content_preview))
WHERE content_preview IS NOT NULL;

-- Content hash index for deduplication
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_content_hash
ON vector_embeddings (content_hash);

-- Partial HNSW indexes per namespace for faster filtered queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_csrd_hnsw
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200)
WHERE namespace = 'csrd';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_eudr_hnsw
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200)
WHERE namespace = 'eudr';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_cbam_hnsw
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200)
WHERE namespace = 'cbam';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_vcci_hnsw
ON vector_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200)
WHERE namespace = 'vcci';

-- Partitioned table indexes (created on each partition)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_part_csrd_hnsw
ON vector_embeddings_csrd
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_part_cbam_hnsw
ON vector_embeddings_cbam
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_part_eudr_hnsw
ON vector_embeddings_eudr
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_part_vcci_hnsw
ON vector_embeddings_vcci
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Embedding jobs indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_status
ON embedding_jobs (status) WHERE status IN ('pending', 'running');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_collection
ON embedding_jobs (collection_id);

-- Search logs indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_search_logs_created
ON vector_search_logs (created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_search_logs_namespace
ON vector_search_logs (namespace);

-- Audit log indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_table
ON vector_audit_log (table_name, changed_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_record
ON vector_audit_log (record_id);

-- ============================================================================
-- 4. Functions
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_vector_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_vector_embeddings_updated_at
    BEFORE UPDATE ON vector_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_vector_updated_at();

CREATE TRIGGER trg_embedding_collections_updated_at
    BEFORE UPDATE ON embedding_collections
    FOR EACH ROW EXECUTE FUNCTION update_vector_updated_at();

-- Audit trigger for vector changes
CREATE OR REPLACE FUNCTION audit_vector_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO vector_audit_log (
        operation,
        table_name,
        record_id,
        old_data,
        new_data,
        changed_by,
        changed_at
    ) VALUES (
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) - 'embedding' ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN to_jsonb(NEW) - 'embedding' ELSE NULL END,
        current_user,
        NOW()
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER vector_embeddings_audit
    AFTER INSERT OR UPDATE OR DELETE ON vector_embeddings
    FOR EACH ROW EXECUTE FUNCTION audit_vector_changes();

-- Update collection vector count on insert/delete
CREATE OR REPLACE FUNCTION update_collection_vector_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' AND NEW.collection_id IS NOT NULL THEN
        UPDATE embedding_collections
        SET vector_count = vector_count + 1, updated_at = NOW()
        WHERE id = NEW.collection_id;
    ELSIF TG_OP = 'DELETE' AND OLD.collection_id IS NOT NULL THEN
        UPDATE embedding_collections
        SET vector_count = vector_count - 1, updated_at = NOW()
        WHERE id = OLD.collection_id;
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_collection_count
    AFTER INSERT OR DELETE ON vector_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_collection_vector_count();

-- Helper function: cosine similarity search
CREATE OR REPLACE FUNCTION vector_similarity_search(
    query_embedding vector(384),
    search_namespace VARCHAR DEFAULT 'default',
    search_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    source_type VARCHAR,
    source_id UUID,
    chunk_index INTEGER,
    content_preview TEXT,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ve.id,
        ve.source_type,
        ve.source_id,
        ve.chunk_index,
        ve.content_preview,
        ve.metadata,
        (1 - (ve.embedding <=> query_embedding))::FLOAT AS similarity
    FROM vector_embeddings ve
    WHERE
        ve.namespace = search_namespace
        AND (1 - (ve.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY ve.embedding <=> query_embedding
    LIMIT search_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Helper function: hybrid search with RRF
CREATE OR REPLACE FUNCTION vector_hybrid_search(
    query_embedding vector(384),
    query_text TEXT,
    search_namespace VARCHAR DEFAULT 'default',
    search_limit INTEGER DEFAULT 10,
    rrf_k INTEGER DEFAULT 60
)
RETURNS TABLE (
    id UUID,
    source_type VARCHAR,
    source_id UUID,
    content_preview TEXT,
    metadata JSONB,
    rrf_score FLOAT,
    vector_rank BIGINT,
    text_rank BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            ve.id,
            ROW_NUMBER() OVER (ORDER BY ve.embedding <=> query_embedding) AS v_rank
        FROM vector_embeddings ve
        WHERE ve.namespace = search_namespace
        ORDER BY ve.embedding <=> query_embedding
        LIMIT 100
    ),
    text_results AS (
        SELECT
            ve.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank(to_tsvector('english', ve.content_preview), plainto_tsquery('english', query_text)) DESC
            ) AS t_rank
        FROM vector_embeddings ve
        WHERE
            ve.namespace = search_namespace
            AND ve.content_preview IS NOT NULL
            AND to_tsvector('english', ve.content_preview) @@ plainto_tsquery('english', query_text)
        LIMIT 100
    ),
    rrf_scores AS (
        SELECT
            COALESCE(v.id, t.id) AS result_id,
            COALESCE(1.0 / (rrf_k + v.v_rank), 0) +
            COALESCE(1.0 / (rrf_k + t.t_rank), 0) AS score,
            v.v_rank,
            t.t_rank
        FROM vector_results v
        FULL OUTER JOIN text_results t ON v.id = t.id
    )
    SELECT
        ve.id,
        ve.source_type,
        ve.source_id,
        ve.content_preview,
        ve.metadata,
        r.score::FLOAT AS rrf_score,
        r.v_rank AS vector_rank,
        r.t_rank AS text_rank
    FROM rrf_scores r
    JOIN vector_embeddings ve ON ve.id = r.result_id
    ORDER BY r.score DESC
    LIMIT search_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================================
-- 5. RBAC Roles
-- ============================================================================

-- Create roles (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_reader') THEN
        CREATE ROLE vector_reader;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_writer') THEN
        CREATE ROLE vector_writer;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'vector_admin') THEN
        CREATE ROLE vector_admin;
    END IF;
END $$;

-- Reader permissions
GRANT USAGE ON SCHEMA public TO vector_reader;
GRANT SELECT ON vector_embeddings TO vector_reader;
GRANT SELECT ON vector_embeddings_partitioned TO vector_reader;
GRANT SELECT ON embedding_collections TO vector_reader;
GRANT SELECT ON vector_search_logs TO vector_reader;

-- Writer permissions (includes reader)
GRANT vector_reader TO vector_writer;
GRANT INSERT, UPDATE ON vector_embeddings TO vector_writer;
GRANT INSERT, UPDATE ON vector_embeddings_partitioned TO vector_writer;
GRANT INSERT, UPDATE ON embedding_collections TO vector_writer;
GRANT INSERT, UPDATE ON embedding_jobs TO vector_writer;
GRANT INSERT ON vector_search_logs TO vector_writer;
GRANT INSERT ON vector_audit_log TO vector_writer;

-- Admin permissions (includes writer)
GRANT vector_writer TO vector_admin;
GRANT DELETE, TRUNCATE ON vector_embeddings TO vector_admin;
GRANT DELETE, TRUNCATE ON vector_embeddings_partitioned TO vector_admin;
GRANT ALL ON embedding_collections TO vector_admin;
GRANT ALL ON embedding_jobs TO vector_admin;
GRANT ALL ON vector_search_logs TO vector_admin;
GRANT ALL ON vector_audit_log TO vector_admin;
GRANT CREATE ON SCHEMA public TO vector_admin;

-- Grant execute on functions
GRANT EXECUTE ON FUNCTION vector_similarity_search TO vector_reader;
GRANT EXECUTE ON FUNCTION vector_hybrid_search TO vector_reader;

-- ============================================================================
-- 6. Configure pgaudit for vector operations
-- ============================================================================

-- Note: These ALTER SYSTEM commands require superuser and a reload
-- They should be applied via parameter groups in Aurora
-- ALTER SYSTEM SET pgaudit.log = 'write, ddl';
-- ALTER SYSTEM SET pgaudit.log_catalog = off;
-- ALTER SYSTEM SET pgaudit.log_relation = on;
-- ALTER SYSTEM SET pgaudit.log_statement_once = on;

-- ============================================================================
-- 7. Verification
-- ============================================================================

DO $$
DECLARE
    tbl_name TEXT;
    required_tables TEXT[] := ARRAY[
        'vector_embeddings',
        'vector_embeddings_partitioned',
        'embedding_collections',
        'embedding_jobs',
        'vector_search_logs',
        'vector_audit_log'
    ];
BEGIN
    FOREACH tbl_name IN ARRAY required_tables
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = tbl_name
        ) THEN
            RAISE EXCEPTION 'Required table % was not created', tbl_name;
        ELSE
            RAISE NOTICE 'Table % created successfully', tbl_name;
        END IF;
    END LOOP;

    -- Verify pgvector extension version
    RAISE NOTICE 'pgvector version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END $$;

-- Log final state
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size
FROM pg_tables
WHERE tablename LIKE 'vector_%' OR tablename LIKE 'embedding_%'
ORDER BY tablename;
