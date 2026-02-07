-- ============================================================
-- V014: Secrets Service Permissions (SEC-006)
-- ============================================================
-- Adds permissions for the secrets management service.
-- These permissions control access to secret CRUD operations,
-- rotation management, and administrative functions.
--
-- Migration: V014__secrets_permissions.sql
-- PRD: PRD-SEC-006-Secrets-Management.md
-- Date: 2026-02
-- ============================================================

-- ============================================================
-- 1. Add secrets permissions
-- ============================================================

INSERT INTO security.permissions (resource, action, description, is_system_permission)
VALUES
    ('secrets', 'list', 'List secret metadata and paths', true),
    ('secrets', 'read', 'Read secret values', true),
    ('secrets', 'write', 'Create and update secrets', true),
    ('secrets', 'admin', 'Delete and restore secrets, manage versions', true),
    ('secrets', 'rotate', 'Trigger manual secret rotation', true)
ON CONFLICT (resource, action) DO NOTHING;

-- ============================================================
-- 2. Grant secrets permissions to appropriate roles
-- ============================================================

-- super_admin gets all secrets permissions
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'super_admin'
  AND r.is_system_role = true
  AND p.resource = 'secrets'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- admin gets all secrets permissions
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'admin'
  AND r.is_system_role = true
  AND p.resource = 'secrets'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- operator gets list, read, rotate (not write/admin)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'operator'
  AND r.is_system_role = true
  AND p.resource = 'secrets'
  AND p.action IN ('list', 'read', 'rotate')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- viewer gets list only (no read of actual values)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'viewer'
  AND r.is_system_role = true
  AND p.resource = 'secrets'
  AND p.action = 'list'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- auditor gets list and read (for compliance verification)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'auditor'
  AND r.is_system_role = true
  AND p.resource = 'secrets'
  AND p.action IN ('list', 'read')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- compliance_officer gets list, read, rotate
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'compliance_officer'
  AND r.is_system_role = true
  AND p.resource = 'secrets'
  AND p.action IN ('list', 'read', 'rotate')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- data_steward gets list (for data lineage tracking)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'data_steward'
  AND r.is_system_role = true
  AND p.resource = 'secrets'
  AND p.action = 'list'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- ============================================================
-- 3. Add audit log entry for this migration
-- ============================================================

INSERT INTO security.audit_log (
    event_type,
    actor_id,
    resource_type,
    resource_id,
    action,
    details,
    occurred_at
)
VALUES (
    'permission_grant',
    'system',
    'permission',
    'secrets:*',
    'migrate',
    '{"migration": "V014__secrets_permissions", "description": "Added secrets service permissions (SEC-006)", "permissions_added": ["secrets:list", "secrets:read", "secrets:write", "secrets:admin", "secrets:rotate"]}',
    NOW()
);
