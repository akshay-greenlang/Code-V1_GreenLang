-- ============================================================
-- V010: RBAC Authorization Layer (SEC-002)
-- ============================================================
-- PRD: SEC-002 - RBAC Authorization Layer
-- Depends on: V009 (security schema already exists)
--
-- Creates 5 tables in the `security` schema for role-based
-- access control with hierarchical roles, fine-grained
-- permissions, deny-wins evaluation, and full audit trail.
--
-- Tables:
--   security.roles              - Role definitions with hierarchy
--   security.permissions        - Fine-grained permission catalogue
--   security.role_permissions   - Role-to-permission assignments
--   security.user_roles         - User-to-role assignments (tenant-scoped)
--   security.rbac_audit_log     - RBAC change audit trail
--
-- Seeds:
--   10 system roles, 60+ standard permissions, default mappings
-- ============================================================

-- Ensure security schema exists (idempotent)
CREATE SCHEMA IF NOT EXISTS security;

-- ============================================================
-- 1. Roles
-- ============================================================
-- Hierarchical role definitions supporting parent-child
-- inheritance.  System roles (is_system_role=true) cannot be
-- deleted or modified by non-super-admin users.  Tenant-scoped
-- roles have a non-NULL tenant_id; system roles have NULL.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.roles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID,
    name            VARCHAR(128)  NOT NULL,
    display_name    VARCHAR(256),
    description     TEXT,
    parent_role_id  UUID REFERENCES security.roles(id),
    is_system_role  BOOLEAN       NOT NULL DEFAULT false,
    is_enabled      BOOLEAN       NOT NULL DEFAULT true,
    metadata        JSONB         DEFAULT '{}',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    created_by      VARCHAR(128),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_role_name_tenant UNIQUE (tenant_id, name)
);

-- Indexes
CREATE INDEX idx_roles_tenant      ON security.roles (tenant_id);
CREATE INDEX idx_roles_parent      ON security.roles (parent_role_id)
    WHERE parent_role_id IS NOT NULL;
CREATE INDEX idx_roles_system      ON security.roles (is_system_role)
    WHERE is_system_role = true;
CREATE INDEX idx_roles_name        ON security.roles (name);
CREATE INDEX idx_roles_enabled     ON security.roles (is_enabled)
    WHERE is_enabled = true;

-- Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION security.update_roles_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_roles_updated_at
    BEFORE UPDATE ON security.roles
    FOR EACH ROW
    EXECUTE FUNCTION security.update_roles_updated_at();

-- Row-Level Security
ALTER TABLE security.roles ENABLE ROW LEVEL SECURITY;

-- System roles (tenant_id IS NULL) are visible to all tenants.
-- Tenant-scoped roles are visible only to their own tenant.
CREATE POLICY roles_tenant_isolation ON security.roles
    USING (
        tenant_id IS NULL
        OR tenant_id::text = current_setting('app.current_tenant', TRUE)
    );

CREATE POLICY roles_service_access ON security.roles
    FOR ALL
    TO greenlang_service
    USING (TRUE);

-- ============================================================
-- 2. Permissions
-- ============================================================
-- Permission catalogue.  Each entry represents a single
-- resource:action pair (e.g. "agents:execute").  System
-- permissions are seeded by migration and should not be
-- modified by application code.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.permissions (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource             VARCHAR(256) NOT NULL,
    action               VARCHAR(128) NOT NULL,
    description          TEXT,
    is_system_permission BOOLEAN      NOT NULL DEFAULT false,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_permission UNIQUE (resource, action)
);

-- Indexes
CREATE INDEX idx_permissions_resource ON security.permissions (resource);
CREATE INDEX idx_permissions_action   ON security.permissions (action);
CREATE INDEX idx_permissions_system   ON security.permissions (is_system_permission)
    WHERE is_system_permission = true;

-- ============================================================
-- 3. Role Permissions (join table)
-- ============================================================
-- Maps roles to permissions with an allow/deny effect.  Deny
-- entries always win during evaluation (deny-wins strategy).
-- Optional JSONB conditions and scope for attribute-based
-- access control.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.role_permissions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id       UUID         NOT NULL REFERENCES security.roles(id) ON DELETE CASCADE,
    permission_id UUID         NOT NULL REFERENCES security.permissions(id) ON DELETE CASCADE,
    effect        VARCHAR(8)   NOT NULL DEFAULT 'allow'
                  CHECK (effect IN ('allow', 'deny')),
    conditions    JSONB        DEFAULT '{}',
    scope         VARCHAR(256),
    granted_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    granted_by    VARCHAR(128),
    CONSTRAINT uq_role_permission UNIQUE (role_id, permission_id)
);

-- Indexes
CREATE INDEX idx_rp_role       ON security.role_permissions (role_id);
CREATE INDEX idx_rp_permission ON security.role_permissions (permission_id);
CREATE INDEX idx_rp_effect     ON security.role_permissions (effect);

-- ============================================================
-- 4. User Roles (assignment table)
-- ============================================================
-- Maps users to roles within a tenant.  Supports expiration
-- and soft-revocation.  A user may hold multiple roles within
-- a single tenant.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.user_roles (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID         NOT NULL,
    role_id     UUID         NOT NULL REFERENCES security.roles(id) ON DELETE CASCADE,
    tenant_id   UUID         NOT NULL,
    assigned_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    assigned_by VARCHAR(128),
    expires_at  TIMESTAMPTZ,
    revoked_at  TIMESTAMPTZ,
    revoked_by  VARCHAR(128),
    is_active   BOOLEAN      NOT NULL DEFAULT true,
    CONSTRAINT uq_user_role_tenant UNIQUE (user_id, role_id, tenant_id)
);

-- Indexes
CREATE INDEX idx_ur_user       ON security.user_roles (user_id);
CREATE INDEX idx_ur_role       ON security.user_roles (role_id);
CREATE INDEX idx_ur_tenant     ON security.user_roles (tenant_id);
CREATE INDEX idx_ur_active     ON security.user_roles (is_active)
    WHERE is_active = true;
CREATE INDEX idx_ur_user_tenant ON security.user_roles (user_id, tenant_id)
    WHERE is_active = true;
CREATE INDEX idx_ur_expires    ON security.user_roles (expires_at)
    WHERE expires_at IS NOT NULL;

-- Row-Level Security
ALTER TABLE security.user_roles ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_roles_tenant_isolation ON security.user_roles
    USING (tenant_id::text = current_setting('app.current_tenant', TRUE));

CREATE POLICY user_roles_service_access ON security.user_roles
    FOR ALL
    TO greenlang_service
    USING (TRUE);

-- ============================================================
-- 5. RBAC Audit Log
-- ============================================================
-- Records every mutation to roles, permissions, and
-- assignments for compliance audit.  Immutable append-only.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.rbac_audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID,
    actor_id        UUID         NOT NULL,
    event_type      VARCHAR(64)  NOT NULL,
    target_type     VARCHAR(64)  NOT NULL,
    target_id       UUID         NOT NULL,
    action          VARCHAR(64)  NOT NULL,
    old_value       JSONB,
    new_value       JSONB,
    ip_address      INET,
    correlation_id  UUID,
    performed_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_ral_tenant      ON security.rbac_audit_log (tenant_id);
CREATE INDEX idx_ral_actor       ON security.rbac_audit_log (actor_id);
CREATE INDEX idx_ral_target      ON security.rbac_audit_log (target_type, target_id);
CREATE INDEX idx_ral_event       ON security.rbac_audit_log (event_type);
CREATE INDEX idx_ral_performed   ON security.rbac_audit_log (performed_at);

-- Convert to TimescaleDB hypertable if the extension is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'security.rbac_audit_log',
            'performed_at',
            if_not_exists => TRUE,
            migrate_data  => TRUE
        );
    END IF;
END $$;

-- Row-Level Security
ALTER TABLE security.rbac_audit_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY rbac_audit_tenant_isolation ON security.rbac_audit_log
    USING (
        tenant_id IS NULL
        OR tenant_id::text = current_setting('app.current_tenant', TRUE)
    );

CREATE POLICY rbac_audit_service_access ON security.rbac_audit_log
    FOR ALL
    TO greenlang_service
    USING (TRUE);

-- ============================================================
-- Seed: System Roles
-- ============================================================

INSERT INTO security.roles (name, display_name, description, is_system_role, created_by)
VALUES
    ('super_admin',     'Super Administrator', 'Full system access across all tenants. Reserved for platform operators.',                     true, 'system'),
    ('admin',           'Administrator',       'Tenant-level administrator with full access to tenant resources.',                            true, 'system'),
    ('manager',         'Manager',             'Manages agents, emissions, jobs, compliance, and factory resources within a tenant.',         true, 'system'),
    ('developer',       'Developer',           'Develops and configures agents, manages emissions data, and factory resources.',              true, 'system'),
    ('operator',        'Operator',            'Executes agents, calculates emissions, manages jobs, and operates factory pipelines.',        true, 'system'),
    ('analyst',         'Analyst',             'Read-only access to agents and factory, full access to emissions and compliance data.',        true, 'system'),
    ('viewer',          'Viewer',              'Read-only and list access to all resources. Cannot modify or execute anything.',              true, 'system'),
    ('auditor',         'Auditor',             'Compliance auditor with read access to audit logs, sessions, compliance, and RBAC metadata.', true, 'system'),
    ('service_account', 'Service Account',     'Machine-to-machine identity for automated agent execution and emissions calculations.',       true, 'system'),
    ('guest',           'Guest',               'Unauthenticated or minimal-access identity. No default permissions.',                         true, 'system')
ON CONFLICT (tenant_id, name) DO NOTHING;

-- ============================================================
-- Seed: Standard Permissions
-- ============================================================

INSERT INTO security.permissions (resource, action, description, is_system_permission)
VALUES
    -- agents
    ('agents', 'list',      'List agents',                        true),
    ('agents', 'read',      'View agent details',                 true),
    ('agents', 'execute',   'Execute an agent',                   true),
    ('agents', 'configure', 'Configure agent settings',           true),
    ('agents', 'create',    'Create a new agent',                 true),
    ('agents', 'update',    'Update an existing agent',           true),
    ('agents', 'delete',    'Delete an agent',                    true),

    -- emissions
    ('emissions', 'list',      'List emission records',           true),
    ('emissions', 'read',      'View emission record details',    true),
    ('emissions', 'calculate', 'Execute emissions calculation',   true),
    ('emissions', 'create',    'Create emission records',         true),
    ('emissions', 'update',    'Update emission records',         true),
    ('emissions', 'delete',    'Delete emission records',         true),
    ('emissions', 'export',    'Export emission data',            true),

    -- jobs
    ('jobs', 'list',    'List jobs',                              true),
    ('jobs', 'read',    'View job details',                       true),
    ('jobs', 'create',  'Create a new job',                       true),
    ('jobs', 'cancel',  'Cancel a running job',                   true),
    ('jobs', 'delete',  'Delete a job',                           true),

    -- compliance
    ('compliance', 'list',    'List compliance reports',          true),
    ('compliance', 'read',    'View compliance report details',   true),
    ('compliance', 'create',  'Create compliance reports',        true),
    ('compliance', 'update',  'Update compliance reports',        true),
    ('compliance', 'delete',  'Delete compliance reports',        true),
    ('compliance', 'approve', 'Approve compliance reports',       true),

    -- factory
    ('factory', 'list',     'List factory agents',                true),
    ('factory', 'read',     'View factory agent details',         true),
    ('factory', 'create',   'Create factory agent entries',       true),
    ('factory', 'update',   'Update factory agent entries',       true),
    ('factory', 'delete',   'Delete factory agent entries',       true),
    ('factory', 'execute',  'Execute factory agent pipelines',    true),
    ('factory', 'metrics',  'View factory agent metrics',         true),
    ('factory', 'deploy',   'Deploy factory agents',              true),
    ('factory', 'rollback', 'Rollback factory agent deployments', true),

    -- flags
    ('flags', 'list',     'List feature flags',                   true),
    ('flags', 'read',     'View feature flag details',            true),
    ('flags', 'create',   'Create feature flags',                 true),
    ('flags', 'update',   'Update feature flags',                 true),
    ('flags', 'delete',   'Delete feature flags',                 true),
    ('flags', 'evaluate', 'Evaluate feature flags',               true),
    ('flags', 'rollout',  'Manage flag rollout percentages',      true),
    ('flags', 'kill',     'Activate kill switch on a flag',       true),
    ('flags', 'restore',  'Restore a killed flag',                true),

    -- admin
    ('admin:users',    'list',    'List users',                   true),
    ('admin:users',    'read',    'View user details',            true),
    ('admin:users',    'unlock',  'Unlock locked accounts',       true),
    ('admin:users',    'revoke',  'Revoke user tokens',           true),
    ('admin:users',    'reset',   'Force password reset',         true),
    ('admin:users',    'mfa',     'Manage user MFA settings',     true),
    ('admin:sessions', 'list',    'List active sessions',         true),
    ('admin:sessions', 'terminate', 'Terminate sessions',         true),
    ('admin:audit',    'read',    'Read audit logs',              true),
    ('admin:lockouts', 'list',    'List account lockouts',        true),

    -- rbac
    ('rbac:roles',       'list',   'List RBAC roles',             true),
    ('rbac:roles',       'read',   'View RBAC role details',      true),
    ('rbac:roles',       'create', 'Create RBAC roles',           true),
    ('rbac:roles',       'update', 'Update RBAC roles',           true),
    ('rbac:roles',       'delete', 'Delete RBAC roles',           true),
    ('rbac:permissions', 'list',   'List RBAC permissions',       true),
    ('rbac:permissions', 'read',   'View RBAC permission details', true),
    ('rbac:assignments', 'list',   'List role assignments',       true),
    ('rbac:assignments', 'read',   'View role assignment details', true),
    ('rbac:assignments', 'create', 'Assign roles to users',       true),
    ('rbac:assignments', 'revoke', 'Revoke role assignments',     true)
ON CONFLICT (resource, action) DO NOTHING;

-- ============================================================
-- Seed: Default Role-Permission Mappings
-- ============================================================
-- Strategy:
--   super_admin  -> ALL permissions
--   admin        -> ALL except rbac:roles:delete
--   manager      -> agents, emissions, jobs, compliance, factory(list/read/execute)
--   developer    -> agents(list/read/execute/configure), emissions, factory(list/read/create/update/execute)
--   operator     -> agents(list/read/execute), emissions(list/read/calculate), jobs, factory(list/read/execute)
--   analyst      -> agents(list/read), emissions, compliance(list/read), factory(list/read/metrics)
--   viewer       -> all list + read permissions
--   auditor      -> admin:audit:read, admin:sessions:list, compliance, rbac(list)
--   service_account -> agents:execute, emissions:calculate, factory:execute
--   guest        -> (no permissions)
-- ============================================================

-- super_admin: ALL permissions
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'super_admin'
    AND r.is_system_role = true
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- admin: ALL permissions except rbac:roles:delete
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'admin'
    AND r.is_system_role = true
    AND NOT (p.resource = 'rbac:roles' AND p.action = 'delete')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- manager: agents, emissions, jobs, compliance, factory(list/read/execute)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'manager'
    AND r.is_system_role = true
    AND (
        p.resource = 'agents'
        OR p.resource = 'emissions'
        OR p.resource = 'jobs'
        OR p.resource = 'compliance'
        OR (p.resource = 'factory' AND p.action IN ('list', 'read', 'execute'))
    )
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- developer: agents(list/read/execute/configure), emissions, factory(list/read/create/update/execute)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'developer'
    AND r.is_system_role = true
    AND (
        (p.resource = 'agents' AND p.action IN ('list', 'read', 'execute', 'configure'))
        OR p.resource = 'emissions'
        OR (p.resource = 'factory' AND p.action IN ('list', 'read', 'create', 'update', 'execute'))
    )
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- operator: agents(list/read/execute), emissions(list/read/calculate), jobs, factory(list/read/execute)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'operator'
    AND r.is_system_role = true
    AND (
        (p.resource = 'agents' AND p.action IN ('list', 'read', 'execute'))
        OR (p.resource = 'emissions' AND p.action IN ('list', 'read', 'calculate'))
        OR p.resource = 'jobs'
        OR (p.resource = 'factory' AND p.action IN ('list', 'read', 'execute'))
    )
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- analyst: agents(list/read), emissions, compliance(list/read), factory(list/read/metrics)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'analyst'
    AND r.is_system_role = true
    AND (
        (p.resource = 'agents' AND p.action IN ('list', 'read'))
        OR p.resource = 'emissions'
        OR (p.resource = 'compliance' AND p.action IN ('list', 'read'))
        OR (p.resource = 'factory' AND p.action IN ('list', 'read', 'metrics'))
    )
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- viewer: all list + read permissions
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'viewer'
    AND r.is_system_role = true
    AND p.action IN ('list', 'read')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- auditor: admin:audit:read, admin:sessions:list, compliance, rbac(list)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'auditor'
    AND r.is_system_role = true
    AND (
        (p.resource = 'admin:audit' AND p.action = 'read')
        OR (p.resource = 'admin:sessions' AND p.action = 'list')
        OR p.resource = 'compliance'
        OR (p.resource LIKE 'rbac:%' AND p.action = 'list')
    )
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- service_account: agents:execute, emissions:calculate, factory:execute
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'service_account'
    AND r.is_system_role = true
    AND (
        (p.resource = 'agents' AND p.action = 'execute')
        OR (p.resource = 'emissions' AND p.action = 'calculate')
        OR (p.resource = 'factory' AND p.action = 'execute')
    )
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- guest: no permissions (nothing to insert)

-- ============================================================
-- Cleanup function for expired audit log entries
-- ============================================================

CREATE OR REPLACE FUNCTION security.cleanup_old_rbac_audit_log(
    retention_days INTEGER DEFAULT 365
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INTEGER;
    cutoff TIMESTAMPTZ;
BEGIN
    cutoff := NOW() - (retention_days || ' days')::INTERVAL;

    DELETE FROM security.rbac_audit_log
    WHERE performed_at < cutoff;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Cleaned up % old rbac_audit_log entries', deleted_count;
    RETURN deleted_count;
END;
$$;

-- ============================================================
-- Expire stale user-role assignments
-- ============================================================

CREATE OR REPLACE FUNCTION security.expire_stale_user_roles()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE security.user_roles
    SET is_active = false,
        revoked_at = NOW(),
        revoked_by = 'system:auto_expire'
    WHERE is_active = true
      AND expires_at IS NOT NULL
      AND expires_at < NOW();

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    RAISE NOTICE 'Expired % stale user_role assignments', updated_count;
    RETURN updated_count;
END;
$$;

-- ============================================================
-- Grants
-- ============================================================

DO $$
BEGIN
    -- Service role (used by the application)
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_service') THEN
        GRANT USAGE ON SCHEMA security TO greenlang_service;
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON ALL TABLES IN SCHEMA security TO greenlang_service;
        GRANT EXECUTE
            ON ALL FUNCTIONS IN SCHEMA security TO greenlang_service;
    END IF;

    -- Read-only role (for reporting/analytics)
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA security TO greenlang_readonly;
        GRANT SELECT
            ON ALL TABLES IN SCHEMA security TO greenlang_readonly;
    END IF;
END $$;

-- ============================================================
-- Comments
-- ============================================================

COMMENT ON TABLE security.roles IS
    'Hierarchical role definitions supporting parent-child inheritance and tenant scoping. SEC-002.';

COMMENT ON COLUMN security.roles.tenant_id IS
    'NULL for system roles (visible to all tenants), non-NULL for tenant-scoped custom roles.';

COMMENT ON COLUMN security.roles.parent_role_id IS
    'Self-referencing FK for role hierarchy. Max depth enforced by application (5 levels).';

COMMENT ON COLUMN security.roles.is_system_role IS
    'System roles are seeded by migration and cannot be deleted or renamed by application code.';

COMMENT ON TABLE security.permissions IS
    'Permission catalogue of resource:action pairs. System permissions are seeded by migration. SEC-002.';

COMMENT ON TABLE security.role_permissions IS
    'Maps roles to permissions with allow/deny effect and optional JSONB conditions. SEC-002.';

COMMENT ON COLUMN security.role_permissions.effect IS
    'allow or deny. Deny entries always win during permission evaluation (deny-wins strategy).';

COMMENT ON COLUMN security.role_permissions.conditions IS
    'Optional JSONB conditions for attribute-based access control (e.g. {"department": "engineering"}).';

COMMENT ON TABLE security.user_roles IS
    'User-to-role assignments scoped by tenant. Supports expiration and soft-revocation. SEC-002.';

COMMENT ON COLUMN security.user_roles.is_active IS
    'Soft-delete flag. Set to false on revocation or expiration. Active assignments have is_active=true.';

COMMENT ON TABLE security.rbac_audit_log IS
    'Immutable audit trail for all RBAC mutations (role/permission/assignment changes). SEC-002.';

COMMENT ON FUNCTION security.cleanup_old_rbac_audit_log(INTEGER) IS
    'Prunes RBAC audit log entries older than retention_days (default 365). SEC-002.';

COMMENT ON FUNCTION security.expire_stale_user_roles() IS
    'Deactivates user-role assignments past their expires_at timestamp. Called by K8s CronJob. SEC-002.';
