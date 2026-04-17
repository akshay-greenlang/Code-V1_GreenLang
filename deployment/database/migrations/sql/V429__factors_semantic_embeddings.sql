-- V429: Create factor_embeddings table for semantic search (F040)
-- Requires pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS factors_catalog;

CREATE TABLE IF NOT EXISTS factors_catalog.factor_embeddings (
    edition_id   TEXT        NOT NULL,
    factor_id    TEXT        NOT NULL,
    embedding    vector(384) NOT NULL,
    search_text  TEXT        NOT NULL DEFAULT '',
    content_hash TEXT        NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (edition_id, factor_id)
);

-- HNSW index for cosine similarity search
CREATE INDEX IF NOT EXISTS idx_factor_embeddings_hnsw
    ON factors_catalog.factor_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Edition lookup index
CREATE INDEX IF NOT EXISTS idx_factor_embeddings_edition
    ON factors_catalog.factor_embeddings (edition_id);

COMMENT ON TABLE factors_catalog.factor_embeddings IS
    'Pre-computed vector embeddings for semantic factor matching (F040)';
