-- ============================================================
-- V012: Encryption Service Permissions (SEC-003)
-- ============================================================
-- Adds encryption permissions for the encryption service API.
-- These permissions control access to encryption, decryption,
-- key management, and audit capabilities.
--
-- Migration: V012__encryption_permissions.sql
-- PRD: PRD-SEC-003-Encryption-at-Rest.md
-- Date: 2026-02
-- ============================================================

-- ============================================================
-- 1. Add encryption permissions
-- ============================================================

INSERT INTO security.permissions (resource, action, description, is_system_permission)
VALUES
    ('encryption', 'encrypt', 'Encrypt data using the encryption service', true),
    ('encryption', 'decrypt', 'Decrypt data using the encryption service', true),
    ('encryption', 'admin', 'Administer encryption keys and configuration', true),
    ('encryption', 'audit', 'View encryption audit logs', true),
    ('encryption', 'read', 'View encryption service status', true)
ON CONFLICT (resource, action) DO NOTHING;

-- ============================================================
-- 2. Grant encryption permissions to appropriate roles
-- ============================================================

-- super_admin gets all encryption permissions
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'super_admin'
  AND r.is_system_role = true
  AND p.resource = 'encryption'
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- admin gets encrypt, decrypt, audit, read (not admin)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'admin'
  AND r.is_system_role = true
  AND p.resource = 'encryption'
  AND p.action IN ('encrypt', 'decrypt', 'audit', 'read')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- service_account gets encrypt, decrypt
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'service_account'
  AND r.is_system_role = true
  AND p.resource = 'encryption'
  AND p.action IN ('encrypt', 'decrypt')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- auditor gets audit, read
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'auditor'
  AND r.is_system_role = true
  AND p.resource = 'encryption'
  AND p.action IN ('audit', 'read')
ON CONFLICT (role_id, permission_id) DO NOTHING;

-- data_steward gets encrypt, decrypt, read (handles sensitive data)
INSERT INTO security.role_permissions (role_id, permission_id, effect, granted_by)
SELECT r.id, p.id, 'allow', 'system'
FROM security.roles r
CROSS JOIN security.permissions p
WHERE r.name = 'data_steward'
  AND r.is_system_role = true
  AND p.resource = 'encryption'
  AND p.action IN ('encrypt', 'decrypt', 'read')
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
    'encryption:*',
    'migrate',
    '{"migration": "V012__encryption_permissions", "description": "Added encryption service permissions (SEC-003)", "permissions_added": ["encryption:encrypt", "encryption:decrypt", "encryption:admin", "encryption:audit", "encryption:read"]}',
    NOW()
);
