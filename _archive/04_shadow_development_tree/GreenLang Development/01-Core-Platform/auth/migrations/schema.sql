-- ============================================================================
-- GreenLang Authentication System PostgreSQL Schema
-- Version: 1.0.0
-- Date: November 2025
--
-- This file contains the complete database schema for the GreenLang
-- authentication system with permissions, roles, policies, and audit logging.
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- PERMISSIONS TABLE
-- Stores fine-grained permissions for resource access control
-- ============================================================================
CREATE TABLE IF NOT EXISTS permissions (
    permission_id VARCHAR(64) PRIMARY KEY,
    resource VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    effect VARCHAR(10) NOT NULL DEFAULT 'allow' CHECK (effect IN ('allow', 'deny')),
    scope VARCHAR(255),
    conditions JSONB,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    created_by VARCHAR(255),
    provenance_hash VARCHAR(64)
);

-- Indexes for permissions
CREATE INDEX idx_resource_action ON permissions(resource, action);
CREATE INDEX idx_effect_scope ON permissions(effect, scope);
CREATE INDEX idx_created_at ON permissions(created_at);
CREATE INDEX idx_provenance ON permissions(provenance_hash);
CREATE INDEX idx_permission_resource ON permissions(resource);
CREATE INDEX idx_permission_action ON permissions(action);
CREATE INDEX idx_permission_scope ON permissions(scope);

-- Comments
COMMENT ON TABLE permissions IS 'Stores fine-grained permissions for resource access control';
COMMENT ON COLUMN permissions.permission_id IS 'Unique identifier for the permission';
COMMENT ON COLUMN permissions.resource IS 'Resource pattern (e.g., agent:*, workflow:carbon-audit)';
COMMENT ON COLUMN permissions.action IS 'Action pattern (e.g., read, execute, *)';
COMMENT ON COLUMN permissions.effect IS 'Permission effect: allow or deny';
COMMENT ON COLUMN permissions.scope IS 'Optional scope restriction (e.g., tenant:123)';
COMMENT ON COLUMN permissions.conditions IS 'JSON conditions that must be satisfied';
COMMENT ON COLUMN permissions.provenance_hash IS 'SHA-256 hash for audit trail';

-- ============================================================================
-- ROLES TABLE
-- Defines roles that group permissions together
-- ============================================================================
CREATE TABLE IF NOT EXISTS roles (
    role_id VARCHAR(64) PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    created_by VARCHAR(255),
    is_system_role BOOLEAN NOT NULL DEFAULT FALSE,
    priority INTEGER NOT NULL DEFAULT 0
);

-- Indexes for roles
CREATE INDEX idx_role_name ON roles(role_name);
CREATE INDEX idx_role_priority ON roles(priority);
CREATE INDEX idx_system_role ON roles(is_system_role);

-- Comments
COMMENT ON TABLE roles IS 'Defines roles that group permissions together';
COMMENT ON COLUMN roles.role_id IS 'Unique identifier for the role';
COMMENT ON COLUMN roles.role_name IS 'Human-readable role name';
COMMENT ON COLUMN roles.description IS 'Description of the role purpose';
COMMENT ON COLUMN roles.is_system_role IS 'Whether this is a protected system role';
COMMENT ON COLUMN roles.priority IS 'Role priority for conflict resolution';

-- ============================================================================
-- POLICIES TABLE
-- Stores access control policies (RBAC, ABAC, temporal)
-- ============================================================================
CREATE TABLE IF NOT EXISTS policies (
    policy_id VARCHAR(64) PRIMARY KEY,
    policy_name VARCHAR(100) UNIQUE NOT NULL,
    policy_type VARCHAR(50) NOT NULL,
    rules JSONB NOT NULL,
    conditions JSONB,
    priority INTEGER NOT NULL DEFAULT 0,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    created_by VARCHAR(255),
    expires_at TIMESTAMP
);

-- Indexes for policies
CREATE INDEX idx_policy_name ON policies(policy_name);
CREATE INDEX idx_policy_type ON policies(policy_type);
CREATE INDEX idx_policy_enabled ON policies(enabled);
CREATE INDEX idx_policy_priority ON policies(priority);
CREATE INDEX idx_policy_expires ON policies(expires_at);

-- Comments
COMMENT ON TABLE policies IS 'Stores access control policies (RBAC, ABAC, temporal)';
COMMENT ON COLUMN policies.policy_id IS 'Unique identifier for the policy';
COMMENT ON COLUMN policies.policy_type IS 'Type of policy: rbac, abac, or temporal';
COMMENT ON COLUMN policies.rules IS 'JSON rules defining the policy logic';
COMMENT ON COLUMN policies.enabled IS 'Whether the policy is currently active';
COMMENT ON COLUMN policies.expires_at IS 'Optional expiration timestamp for temporal policies';

-- ============================================================================
-- AUDIT_LOGS TABLE
-- Comprehensive audit logging for all authentication events
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id VARCHAR(64) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(255),
    resource VARCHAR(255),
    action VARCHAR(100),
    result VARCHAR(20) NOT NULL CHECK (result IN ('success', 'failure', 'error')),
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    session_id VARCHAR(255),
    correlation_id VARCHAR(64),
    provenance_hash VARCHAR(64)
);

-- Indexes for audit_logs
CREATE INDEX idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_session ON audit_logs(session_id);
CREATE INDEX idx_audit_correlation ON audit_logs(correlation_id);
CREATE INDEX idx_audit_result ON audit_logs(result);

-- Partial index for recent logs (last 7 days)
CREATE INDEX idx_audit_recent ON audit_logs(timestamp)
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '7 days';

-- Comments
COMMENT ON TABLE audit_logs IS 'Comprehensive audit logging for all authentication events';
COMMENT ON COLUMN audit_logs.log_id IS 'Unique identifier for the audit log entry';
COMMENT ON COLUMN audit_logs.event_type IS 'Type of event (e.g., permission.created, role.assigned)';
COMMENT ON COLUMN audit_logs.result IS 'Result of the operation: success, failure, or error';
COMMENT ON COLUMN audit_logs.correlation_id IS 'ID for correlating related events';
COMMENT ON COLUMN audit_logs.provenance_hash IS 'SHA-256 hash for data integrity';

-- ============================================================================
-- ROLE_PERMISSIONS TABLE
-- Many-to-many relationship between roles and permissions
-- ============================================================================
CREATE TABLE IF NOT EXISTS role_permissions (
    role_id VARCHAR(64) NOT NULL REFERENCES roles(role_id) ON DELETE CASCADE,
    permission_id VARCHAR(64) NOT NULL REFERENCES permissions(permission_id) ON DELETE CASCADE,
    granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    granted_by VARCHAR(255),
    CONSTRAINT uq_role_permission UNIQUE (role_id, permission_id)
);

-- Indexes
CREATE INDEX idx_role_permissions_role ON role_permissions(role_id);
CREATE INDEX idx_role_permissions_permission ON role_permissions(permission_id);

-- Comments
COMMENT ON TABLE role_permissions IS 'Many-to-many relationship between roles and permissions';

-- ============================================================================
-- USER_ROLES TABLE
-- Assigns roles to users with optional expiration
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_roles (
    user_id VARCHAR(255) NOT NULL,
    role_id VARCHAR(64) NOT NULL REFERENCES roles(role_id) ON DELETE CASCADE,
    assigned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    assigned_by VARCHAR(255),
    expires_at TIMESTAMP,
    CONSTRAINT uq_user_role UNIQUE (user_id, role_id)
);

-- Indexes
CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_role ON user_roles(role_id);
CREATE INDEX idx_user_roles_expires ON user_roles(expires_at);

-- Comments
COMMENT ON TABLE user_roles IS 'Assigns roles to users with optional expiration';
COMMENT ON COLUMN user_roles.expires_at IS 'Optional expiration for temporary role assignments';

-- ============================================================================
-- DEFAULT SYSTEM DATA
-- Insert default roles and permissions for system operation
-- ============================================================================

-- Default system roles
INSERT INTO roles (role_id, role_name, description, is_system_role, priority, created_by)
VALUES
    ('sys-admin', 'System Administrator', 'Full system access with all permissions', true, 100, 'system'),
    ('sys-operator', 'System Operator', 'Operational access for running agents and workflows', true, 80, 'system'),
    ('sys-viewer', 'System Viewer', 'Read-only access to all resources', true, 50, 'system'),
    ('sys-auditor', 'System Auditor', 'Access to audit logs and compliance reports', true, 60, 'system')
ON CONFLICT (role_id) DO NOTHING;

-- Default system permissions
INSERT INTO permissions (permission_id, resource, action, effect, created_by)
VALUES
    ('perm-admin-all', '*', '*', 'allow', 'system'),
    ('perm-operator-exec', 'agent:*', 'execute', 'allow', 'system'),
    ('perm-operator-run', 'workflow:*', 'run', 'allow', 'system'),
    ('perm-operator-read', '*', 'read', 'allow', 'system'),
    ('perm-viewer-read', '*', 'read', 'allow', 'system'),
    ('perm-viewer-list', '*', 'list', 'allow', 'system'),
    ('perm-auditor-logs', 'audit:*', '*', 'allow', 'system'),
    ('perm-auditor-reports', 'report:compliance', '*', 'allow', 'system')
ON CONFLICT (permission_id) DO NOTHING;

-- Assign permissions to system roles
INSERT INTO role_permissions (role_id, permission_id, granted_by)
VALUES
    ('sys-admin', 'perm-admin-all', 'system'),
    ('sys-operator', 'perm-operator-exec', 'system'),
    ('sys-operator', 'perm-operator-run', 'system'),
    ('sys-operator', 'perm-operator-read', 'system'),
    ('sys-viewer', 'perm-viewer-read', 'system'),
    ('sys-viewer', 'perm-viewer-list', 'system'),
    ('sys-auditor', 'perm-auditor-logs', 'system'),
    ('sys-auditor', 'perm-auditor-reports', 'system')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_permissions_updated_at
    BEFORE UPDATE ON permissions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_roles_updated_at
    BEFORE UPDATE ON roles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_policies_updated_at
    BEFORE UPDATE ON policies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate provenance hash
CREATE OR REPLACE FUNCTION calculate_provenance_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.provenance_hash = encode(
        digest(
            COALESCE(NEW.resource, '') ||
            COALESCE(NEW.action, '') ||
            COALESCE(NEW.effect, '') ||
            COALESCE(NEW.scope, ''),
            'sha256'
        ),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for automatic provenance hash calculation
CREATE TRIGGER calculate_permission_provenance
    BEFORE INSERT OR UPDATE ON permissions
    FOR EACH ROW
    EXECUTE FUNCTION calculate_provenance_hash();

-- Function to clean up expired data
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS void AS $$
BEGIN
    -- Delete expired user-role assignments
    DELETE FROM user_roles
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;

    -- Disable expired policies
    UPDATE policies
    SET enabled = FALSE
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP AND enabled = TRUE;

    -- Delete old audit logs (keep last 90 days by default)
    DELETE FROM audit_logs
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View for active user permissions
CREATE OR REPLACE VIEW v_user_permissions AS
SELECT
    ur.user_id,
    r.role_name,
    p.resource,
    p.action,
    p.effect,
    p.scope,
    p.conditions
FROM user_roles ur
JOIN roles r ON ur.role_id = r.role_id
JOIN role_permissions rp ON r.role_id = rp.role_id
JOIN permissions p ON rp.permission_id = p.permission_id
WHERE (ur.expires_at IS NULL OR ur.expires_at > CURRENT_TIMESTAMP);

-- View for audit summary
CREATE OR REPLACE VIEW v_audit_summary AS
SELECT
    DATE(timestamp) as audit_date,
    event_type,
    result,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users
FROM audit_logs
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE(timestamp), event_type, result
ORDER BY audit_date DESC, event_type;

-- ============================================================================
-- PERFORMANCE STATISTICS TABLE
-- Track backend performance metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    operation VARCHAR(50) NOT NULL,
    duration_ms FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    details JSONB
);

CREATE INDEX idx_perf_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_perf_operation ON performance_metrics(operation);

-- ============================================================================
-- GRANTS (adjust based on your database users)
-- ============================================================================
-- Example grants for application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO greenlang_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO greenlang_app;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO greenlang_app;