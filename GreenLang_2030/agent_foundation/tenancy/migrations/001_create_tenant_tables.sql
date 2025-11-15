-- Migration: 001_create_tenant_tables.sql
-- Description: Create master tenant tables for multi-tenancy support
-- Author: GL-BackendDeveloper
-- Date: 2025-11-15
-- Resolves: CWE-639 (Data Leakage Between Tenants)

-- ============================================================================
-- MASTER TENANTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS tenants (
    -- Primary identifiers
    tenant_id UUID PRIMARY KEY,
    slug VARCHAR(63) UNIQUE NOT NULL,

    -- Status and tier
    status VARCHAR(50) NOT NULL,
    tier VARCHAR(50) NOT NULL,

    -- Database isolation
    database_name VARCHAR(255) NOT NULL UNIQUE,

    -- Security
    api_key_hash VARCHAR(64) NOT NULL UNIQUE,

    -- Metadata (JSON storage for flexibility)
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Quotas and Usage (JSON storage for dynamic quotas)
    quotas JSONB DEFAULT '{}'::jsonb,
    usage JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    activated_at TIMESTAMP,
    suspended_at TIMESTAMP,
    deleted_at TIMESTAMP,
    trial_ends_at TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('active', 'suspended', 'deleted', 'provisioning', 'trial', 'expired')),
    CONSTRAINT valid_tier CHECK (tier IN ('free', 'starter', 'professional', 'enterprise', 'custom')),
    CONSTRAINT valid_slug CHECK (slug ~ '^[a-z0-9][a-z0-9-]*[a-z0-9]$')
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Primary lookup indexes
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
CREATE INDEX IF NOT EXISTS idx_tenants_tier ON tenants(tier);
CREATE INDEX IF NOT EXISTS idx_tenants_slug ON tenants(slug);
CREATE INDEX IF NOT EXISTS idx_tenants_api_key_hash ON tenants(api_key_hash);

-- Time-based indexes
CREATE INDEX IF NOT EXISTS idx_tenants_created_at ON tenants(created_at);
CREATE INDEX IF NOT EXISTS idx_tenants_activated_at ON tenants(activated_at);
CREATE INDEX IF NOT EXISTS idx_tenants_trial_ends_at ON tenants(trial_ends_at);

-- Composite index for active tenant queries
CREATE INDEX IF NOT EXISTS idx_tenants_active_tier ON tenants(status, tier) WHERE status = 'active';

-- ============================================================================
-- AUDIT LOG TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS tenant_audit_log (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    action VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Optional user tracking
    user_id UUID,
    ip_address INET,
    user_agent TEXT
);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_tenant_id ON tenant_audit_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON tenant_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON tenant_audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_user_id ON tenant_audit_log(user_id);

-- ============================================================================
-- TENANT METRICS TABLE (for analytics)
-- ============================================================================

CREATE TABLE IF NOT EXISTS tenant_metrics (
    id BIGSERIAL PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    metric_type VARCHAR(50) NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT NOW(),

    -- Metadata for context
    tags JSONB DEFAULT '{}'::jsonb
);

-- Metrics indexes
CREATE INDEX IF NOT EXISTS idx_metrics_tenant_id ON tenant_metrics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON tenant_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded_at ON tenant_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_metrics_tenant_type ON tenant_metrics(tenant_id, metric_type);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) - BACKUP PROTECTION
-- ============================================================================

-- Enable RLS on tenants table
ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;

-- Policy: Master admin can see all tenants
CREATE POLICY tenant_admin_policy ON tenants
    FOR ALL
    TO postgres
    USING (true);

-- Policy: Application users can only see their own tenant
-- (This requires setting current_setting('app.current_tenant_id'))
CREATE POLICY tenant_isolation_policy ON tenants
    FOR SELECT
    USING (tenant_id::text = current_setting('app.current_tenant_id', true));

-- ============================================================================
-- FUNCTIONS FOR TENANT MANAGEMENT
-- ============================================================================

-- Function: Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update updated_at on tenant changes
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function: Get tenant quota utilization
CREATE OR REPLACE FUNCTION get_tenant_quota_utilization(p_tenant_id UUID)
RETURNS JSONB AS $$
DECLARE
    v_quotas JSONB;
    v_usage JSONB;
    v_result JSONB;
BEGIN
    SELECT quotas, usage INTO v_quotas, v_usage
    FROM tenants
    WHERE tenant_id = p_tenant_id;

    -- Calculate utilization percentages
    v_result := jsonb_build_object(
        'agents', CASE WHEN (v_quotas->>'max_agents')::int > 0
                  THEN ((v_usage->>'current_agents')::float / (v_quotas->>'max_agents')::float * 100)::int
                  ELSE 0 END,
        'users', CASE WHEN (v_quotas->>'max_users')::int > 0
                 THEN ((v_usage->>'current_users')::float / (v_quotas->>'max_users')::float * 100)::int
                 ELSE 0 END,
        'storage', CASE WHEN (v_quotas->>'max_storage_gb')::float > 0
                   THEN ((v_usage->>'storage_used_gb')::float / (v_quotas->>'max_storage_gb')::float * 100)::int
                   ELSE 0 END
    );

    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Function: Check if tenant is at risk (>90% quota utilization)
CREATE OR REPLACE FUNCTION is_tenant_at_risk(p_tenant_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_utilization JSONB;
BEGIN
    v_utilization := get_tenant_quota_utilization(p_tenant_id);

    RETURN (
        (v_utilization->>'agents')::int > 90 OR
        (v_utilization->>'users')::int > 90 OR
        (v_utilization->>'storage')::int > 90
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR MONITORING
-- ============================================================================

-- View: Active tenants summary
CREATE OR REPLACE VIEW v_active_tenants AS
SELECT
    tenant_id,
    slug,
    tier,
    (metadata->>'company_name') as company_name,
    (usage->>'current_agents')::int as agents_count,
    (usage->>'current_users')::int as users_count,
    get_tenant_quota_utilization(tenant_id) as quota_utilization,
    is_tenant_at_risk(tenant_id) as at_risk,
    created_at,
    activated_at
FROM tenants
WHERE status = 'active'
ORDER BY created_at DESC;

-- View: Tenant audit summary
CREATE OR REPLACE VIEW v_tenant_audit_summary AS
SELECT
    t.tenant_id,
    t.slug,
    COUNT(a.id) as total_actions,
    COUNT(CASE WHEN a.action = 'tenant_created' THEN 1 END) as created_count,
    COUNT(CASE WHEN a.action = 'tenant_updated' THEN 1 END) as updated_count,
    COUNT(CASE WHEN a.action = 'tenant_suspended' THEN 1 END) as suspended_count,
    MAX(a.created_at) as last_action_at
FROM tenants t
LEFT JOIN tenant_audit_log a ON t.tenant_id = a.tenant_id
GROUP BY t.tenant_id, t.slug;

-- View: Tenants by tier
CREATE OR REPLACE VIEW v_tenants_by_tier AS
SELECT
    tier,
    COUNT(*) as tenant_count,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
    COUNT(CASE WHEN status = 'trial' THEN 1 END) as trial_count,
    COUNT(CASE WHEN status = 'suspended' THEN 1 END) as suspended_count
FROM tenants
WHERE status != 'deleted'
GROUP BY tier
ORDER BY
    CASE tier
        WHEN 'free' THEN 1
        WHEN 'starter' THEN 2
        WHEN 'professional' THEN 3
        WHEN 'enterprise' THEN 4
        WHEN 'custom' THEN 5
    END;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE tenants IS 'Master tenant registry with complete isolation metadata';
COMMENT ON TABLE tenant_audit_log IS 'Comprehensive audit trail for all tenant operations';
COMMENT ON TABLE tenant_metrics IS 'Time-series metrics for tenant usage analytics';

COMMENT ON COLUMN tenants.tenant_id IS 'Unique tenant identifier (UUID)';
COMMENT ON COLUMN tenants.slug IS 'URL-friendly tenant identifier (e.g., acme-corp)';
COMMENT ON COLUMN tenants.database_name IS 'Isolated PostgreSQL database name for this tenant';
COMMENT ON COLUMN tenants.api_key_hash IS 'SHA-256 hash of tenant API key for authentication';
COMMENT ON COLUMN tenants.quotas IS 'Resource quotas (JSON): max_agents, max_users, etc.';
COMMENT ON COLUMN tenants.usage IS 'Current resource usage (JSON): current_agents, etc.';

-- ============================================================================
-- GRANTS (adjust based on your user setup)
-- ============================================================================

-- Grant read access to monitoring user (if exists)
-- GRANT SELECT ON tenants, tenant_audit_log, tenant_metrics TO monitoring_user;

-- Grant access to views
-- GRANT SELECT ON v_active_tenants, v_tenant_audit_summary, v_tenants_by_tier TO monitoring_user;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Log migration
INSERT INTO tenant_audit_log (tenant_id, action, details)
VALUES (
    '00000000-0000-0000-0000-000000000000',
    'migration_001_applied',
    '{"migration": "001_create_tenant_tables", "description": "Create master tenant tables", "timestamp": "2025-11-15"}'::jsonb
);

SELECT 'Migration 001 completed successfully' AS status;
