-- GreenLang Agent Registry Schema
-- PostgreSQL database schema for agent versioning, certification, and discovery

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Agents table: Core agent metadata
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    namespace VARCHAR(255) NOT NULL DEFAULT 'default',
    description TEXT,
    author VARCHAR(255),
    repository_url VARCHAR(512),
    homepage_url VARCHAR(512),
    spec_hash VARCHAR(64) NOT NULL, -- SHA-256 hash of agent specification
    status VARCHAR(50) NOT NULL DEFAULT 'active', -- active, deprecated, archived
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Ensure unique agent names per namespace
    CONSTRAINT unique_agent_namespace UNIQUE (namespace, name)
);

-- Agent versions table: Version history and artifact storage
CREATE TABLE IF NOT EXISTS agent_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL, -- Semantic versioning: 1.0.0
    pack_path VARCHAR(512) NOT NULL, -- S3/filesystem path to .glpack
    pack_hash VARCHAR(64) NOT NULL, -- SHA-256 hash of pack file
    metadata JSONB DEFAULT '{}', -- Flexible metadata storage
    capabilities JSONB DEFAULT '[]', -- Agent capabilities
    dependencies JSONB DEFAULT '[]', -- Required dependencies
    size_bytes BIGINT, -- Pack file size
    status VARCHAR(50) NOT NULL DEFAULT 'published', -- published, yanked, deprecated
    published_by VARCHAR(255), -- User/system that published
    published_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Ensure unique versions per agent
    CONSTRAINT unique_agent_version UNIQUE (agent_id, version)
);

-- Agent certifications table: GL-CERT certification tracking
CREATE TABLE IF NOT EXISTS agent_certifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL, -- Certified version
    dimension VARCHAR(100) NOT NULL, -- security, performance, reliability, etc.
    status VARCHAR(50) NOT NULL, -- passed, failed, pending, expired
    score DECIMAL(5,2), -- Certification score (0-100)
    evidence JSONB DEFAULT '{}', -- Test results, audit logs
    certified_by VARCHAR(255), -- Certifying authority
    certification_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expiry_date TIMESTAMP WITH TIME ZONE, -- Certifications may expire

    -- Ensure unique certifications per dimension/version
    CONSTRAINT unique_agent_cert UNIQUE (agent_id, version, dimension)
);

-- Agent tags table: Searchable tags and categories
CREATE TABLE IF NOT EXISTS agent_tags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    tag VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_agent_tag UNIQUE (agent_id, tag)
);

-- Agent downloads table: Usage analytics
CREATE TABLE IF NOT EXISTS agent_downloads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL,
    downloaded_by VARCHAR(255),
    downloaded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Indexes for performance

-- Agent lookups
CREATE INDEX idx_agents_namespace ON agents(namespace);
CREATE INDEX idx_agents_name ON agents(name);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_created_at ON agents(created_at DESC);

-- Version lookups
CREATE INDEX idx_agent_versions_agent_id ON agent_versions(agent_id);
CREATE INDEX idx_agent_versions_version ON agent_versions(version);
CREATE INDEX idx_agent_versions_status ON agent_versions(status);
CREATE INDEX idx_agent_versions_published_at ON agent_versions(published_at DESC);

-- Certification lookups
CREATE INDEX idx_agent_certifications_agent_id ON agent_certifications(agent_id);
CREATE INDEX idx_agent_certifications_dimension ON agent_certifications(dimension);
CREATE INDEX idx_agent_certifications_status ON agent_certifications(status);
CREATE INDEX idx_agent_certifications_date ON agent_certifications(certification_date DESC);

-- Tag lookups
CREATE INDEX idx_agent_tags_agent_id ON agent_tags(agent_id);
CREATE INDEX idx_agent_tags_tag ON agent_tags(tag);

-- Download analytics
CREATE INDEX idx_agent_downloads_agent_id ON agent_downloads(agent_id);
CREATE INDEX idx_agent_downloads_downloaded_at ON agent_downloads(downloaded_at DESC);

-- Full-text search on agent descriptions
CREATE INDEX idx_agents_description_fts ON agents USING gin(to_tsvector('english', description));

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Seed data for testing (optional)
-- INSERT INTO agents (name, namespace, description, author, spec_hash) VALUES
-- ('thermosync', 'greenlang', 'Temperature monitoring and HVAC control agent', 'GreenLang Team', 'abc123'),
-- ('dataflow', 'greenlang', 'Data pipeline orchestration agent', 'GreenLang Team', 'def456');

-- Views for common queries

-- Latest version per agent
CREATE OR REPLACE VIEW agent_latest_versions AS
SELECT DISTINCT ON (av.agent_id)
    av.agent_id,
    av.id AS version_id,
    av.version,
    av.pack_path,
    av.status,
    av.published_at
FROM agent_versions av
ORDER BY av.agent_id, av.published_at DESC;

-- Agent summary with latest version
CREATE OR REPLACE VIEW agent_summary AS
SELECT
    a.id,
    a.name,
    a.namespace,
    a.description,
    a.author,
    a.status,
    a.created_at,
    alv.version AS latest_version,
    alv.published_at AS latest_version_date,
    (SELECT COUNT(*) FROM agent_versions WHERE agent_id = a.id) AS version_count,
    (SELECT COUNT(*) FROM agent_downloads WHERE agent_id = a.id) AS download_count
FROM agents a
LEFT JOIN agent_latest_versions alv ON a.id = alv.agent_id;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO greenlang_api;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO greenlang_api;
