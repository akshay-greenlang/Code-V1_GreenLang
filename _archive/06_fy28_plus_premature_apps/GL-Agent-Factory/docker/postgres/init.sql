-- GreenLang PostgreSQL Initialization Script
-- This script runs automatically when the PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS tenants;

-- Grant permissions to greenlang user
GRANT ALL PRIVILEGES ON SCHEMA agents TO greenlang;
GRANT ALL PRIVILEGES ON SCHEMA audit TO greenlang;
GRANT ALL PRIVILEGES ON SCHEMA tenants TO greenlang;

-- Set search path
ALTER DATABASE greenlang SET search_path TO public, agents, audit, tenants;

-- Create agent_status enum
DO $$ BEGIN
    CREATE TYPE agent_status AS ENUM ('draft', 'active', 'deprecated', 'archived');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create execution_status enum
DO $$ BEGIN
    CREATE TYPE execution_status AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create agents table
CREATE TABLE IF NOT EXISTS agents.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    status agent_status NOT NULL DEFAULT 'draft',
    category VARCHAR(100),
    capabilities JSONB DEFAULT '[]'::jsonb,
    config JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- Create agent_versions table
CREATE TABLE IF NOT EXISTS agents.agent_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES agents.agents(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    changelog TEXT,
    config JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,
    UNIQUE(agent_id, version)
);

-- Create executions table
CREATE TABLE IF NOT EXISTS agents.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    agent_id UUID NOT NULL REFERENCES agents.agents(id),
    tenant_id UUID,
    status execution_status NOT NULL DEFAULT 'pending',
    input JSONB NOT NULL,
    output JSONB,
    error TEXT,
    provenance_hash VARCHAR(64),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    cost_tokens INTEGER DEFAULT 0,
    cost_usd DECIMAL(10, 6) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create tenants table
CREATE TABLE IF NOT EXISTS tenants.tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    settings JSONB DEFAULT '{}'::jsonb,
    quotas JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create audit_logs table
CREATE TABLE IF NOT EXISTS audit.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    tenant_id UUID,
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_agents_tenant_id ON agents.agents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents.agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_category ON agents.agents(category);
CREATE INDEX IF NOT EXISTS idx_agents_name_trgm ON agents.agents USING GIN (name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_executions_agent_id ON agents.executions(agent_id);
CREATE INDEX IF NOT EXISTS idx_executions_tenant_id ON agents.executions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON agents.executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_created_at ON agents.executions(created_at);

CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit.audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit.audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit.audit_logs(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to agents table
DROP TRIGGER IF EXISTS update_agents_updated_at ON agents.agents;
CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON agents.agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply updated_at trigger to tenants table
DROP TRIGGER IF EXISTS update_tenants_updated_at ON tenants.tenants;
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants.tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default tenant for development
INSERT INTO tenants.tenants (tenant_id, name, settings, quotas)
VALUES (
    'default',
    'Default Development Tenant',
    '{"environment": "development"}'::jsonb,
    '{"max_agents": 100, "max_executions_per_month": 10000}'::jsonb
)
ON CONFLICT (tenant_id) DO NOTHING;

-- Log initialization
INSERT INTO audit.audit_logs (event_type, action, details)
VALUES (
    'system',
    'database_initialized',
    jsonb_build_object(
        'version', '1.0.0',
        'timestamp', NOW()::text,
        'environment', 'development'
    )
);

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'GreenLang database initialized successfully';
END $$;
