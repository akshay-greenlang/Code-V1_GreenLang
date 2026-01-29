-- PostgreSQL Initialization Script for GL Normalizer
-- Component: GL-FOUND-X-003 - Unit & Reference Normalizer
-- Purpose: Initialize database schemas for local development

-- Create databases
CREATE DATABASE IF NOT EXISTS normalizer;
CREATE DATABASE IF NOT EXISTS review_console;

-- Connect to normalizer database
\c normalizer;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS vocabulary;
CREATE SCHEMA IF NOT EXISTS cache;

-- Audit schema tables
CREATE TABLE IF NOT EXISTS audit.normalization_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    normalization_event_id VARCHAR(64) UNIQUE NOT NULL,
    source_record_id VARCHAR(255) NOT NULL,
    pipeline_id VARCHAR(255),
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_id VARCHAR(64) NOT NULL DEFAULT 'GL-FOUND-X-003',
    agent_version VARCHAR(32) NOT NULL,
    policy_mode VARCHAR(16) NOT NULL CHECK (policy_mode IN ('STRICT', 'LENIENT')),
    status VARCHAR(16) NOT NULL CHECK (status IN ('success', 'warning', 'failed')),
    payload JSONB NOT NULL,
    prev_event_hash VARCHAR(64),
    event_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_events_source_record ON audit.normalization_events(source_record_id);
CREATE INDEX idx_audit_events_processed_at ON audit.normalization_events(processed_at);
CREATE INDEX idx_audit_events_status ON audit.normalization_events(status);
CREATE INDEX idx_audit_events_pipeline ON audit.normalization_events(pipeline_id);

-- Vocabulary schema tables
CREATE TABLE IF NOT EXISTS vocabulary.entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reference_id VARCHAR(64) UNIQUE NOT NULL,
    canonical_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(32) NOT NULL CHECK (entity_type IN ('fuel', 'material', 'process')),
    status VARCHAR(16) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deprecated')),
    replaced_by VARCHAR(64) REFERENCES vocabulary.entities(reference_id),
    attributes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX idx_vocab_entities_type ON vocabulary.entities(entity_type);
CREATE INDEX idx_vocab_entities_status ON vocabulary.entities(status);
CREATE INDEX idx_vocab_entities_name ON vocabulary.entities(canonical_name);

CREATE TABLE IF NOT EXISTS vocabulary.aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES vocabulary.entities(id) ON DELETE CASCADE,
    alias VARCHAR(255) NOT NULL,
    alias_type VARCHAR(32) DEFAULT 'synonym',
    locale VARCHAR(10),
    weight DECIMAL(3,2) DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(alias, entity_id)
);

CREATE INDEX idx_vocab_aliases_alias ON vocabulary.aliases(alias);
CREATE INDEX idx_vocab_aliases_entity ON vocabulary.aliases(entity_id);

CREATE TABLE IF NOT EXISTS vocabulary.versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version VARCHAR(32) UNIQUE NOT NULL,
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_by VARCHAR(255),
    changelog TEXT,
    checksum VARCHAR(64) NOT NULL,
    is_current BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE UNIQUE INDEX idx_vocab_versions_current ON vocabulary.versions(is_current) WHERE is_current = TRUE;

-- Cache schema for vocabulary caching
CREATE TABLE IF NOT EXISTS cache.vocabulary_cache (
    key VARCHAR(255) PRIMARY KEY,
    value JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cache_expires ON cache.vocabulary_cache(expires_at);

-- Connect to review_console database
\c review_console;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS reviews;
CREATE SCHEMA IF NOT EXISTS users;

-- Users schema
CREATE TABLE IF NOT EXISTS users.accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(32) NOT NULL DEFAULT 'reviewer' CHECK (role IN ('admin', 'reviewer', 'viewer')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Reviews schema
CREATE TABLE IF NOT EXISTS reviews.pending_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    normalization_event_id VARCHAR(64) NOT NULL,
    entity_type VARCHAR(32) NOT NULL,
    raw_name VARCHAR(1000) NOT NULL,
    suggested_reference_id VARCHAR(64),
    suggested_canonical_name VARCHAR(255),
    match_method VARCHAR(32) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    candidates JSONB,
    status VARCHAR(16) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'merged')),
    assigned_to UUID REFERENCES users.accounts(id),
    resolved_by UUID REFERENCES users.accounts(id),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_reviews_status ON reviews.pending_reviews(status);
CREATE INDEX idx_reviews_assigned ON reviews.pending_reviews(assigned_to);
CREATE INDEX idx_reviews_entity_type ON reviews.pending_reviews(entity_type);
CREATE INDEX idx_reviews_confidence ON reviews.pending_reviews(confidence);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO greenlang;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA vocabulary TO greenlang;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA cache TO greenlang;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO greenlang;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA vocabulary TO greenlang;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA cache TO greenlang;
