-- ============================================================
-- V013: Audit Service Permissions (SEC-005)
-- ============================================================
-- Adds audit permissions for the centralized audit logging service.
-- These permissions control access to audit event viewing, searching,
-- exporting, and compliance report generation.
--
-- Migration: V013__audit_permissions.sql
-- PRD: PRD-SEC-005-Audit-Logging.md
-- Date: 2026-02
-- ============================================================

-- ============================================================
-- 1. Add audit permissions
-- ============================================================

INSERT INTO security.permissions (resource, action, description, is_system_permission)
VALUES
    ('audit', 'read', 'View audit events and statistics', true),
    ('audit', 'search', 'Advanced search of audit events', true),
    ('audit', 'export', 'Export audit data to files (CSV, JSON, Parquet)', true),
    ('audit', 'admin', 'Generate compliance reports (SOC2, ISO27001, GDPR)', true)
ON CONFLICT (resource, action) DO NOTHING;

-- ============================================================
-- 2. Grant audit permissions to appropriate roles
-- ============================================================

-- super_admin gets all audit permissions
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'super_admin'
  AND r.is_system_role = true
  AND p.resource = 'audit'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- admin gets all audit permissions
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'admin'
  AND r.is_system_role = true
  AND p.resource = 'audit'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- auditor gets read, search, export, admin (compliance reports)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'auditor'
  AND r.is_system_role = true
  AND p.resource = 'audit'
  AND p.action IN ('read', 'search', 'export', 'admin')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- compliance_officer gets read, search, export, admin (compliance reports)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'compliance_officer'
  AND r.is_system_role = true
  AND p.resource = 'audit'
  AND p.action IN ('read', 'search', 'export', 'admin')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- operator gets read, search, export
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'operator'
  AND r.is_system_role = true
  AND p.resource = 'audit'
  AND p.action IN ('read', 'search', 'export')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- viewer gets read only
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'viewer'
  AND r.is_system_role = true
  AND p.resource = 'audit'
  AND p.action = 'read'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- data_steward gets read, search (for data lineage tracking)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'data_steward'
  AND r.is_system_role = true
  AND p.resource = 'audit'
  AND p.action IN ('read', 'search')
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
    'audit:*',
    'migrate',
    '{"migration": "V013__audit_permissions", "description": "Added audit service permissions (SEC-005)", "permissions_added": ["audit:read", "audit:search", "audit:export", "audit:admin"]}',
    NOW()
);
