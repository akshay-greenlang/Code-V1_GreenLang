# Auth Admin API Reference (SEC-001)

## Overview

Protected administrative endpoints for the Authentication Service. All endpoints require the `admin:*` permission or the `admin` / `super_admin` role. Provides user management, lockout handling, session administration, and auth audit log queries.

**Router Prefix:** `/auth/admin`
**Tags:** `Auth Admin`
**Source:** `greenlang/infrastructure/auth_service/api/admin_routes.py`
**Compliance:** SOC 2 CC6.2 (Privileged Access), ISO 27001 A.9.2 (User Access Management)

---

## Endpoint Summary

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/auth/admin/users` | List users (paginated, filterable) | Admin |
| GET | `/auth/admin/users/{user_id}` | Get user details | Admin |
| POST | `/auth/admin/users/{user_id}/unlock` | Unlock locked account | Admin |
| POST | `/auth/admin/users/{user_id}/revoke-tokens` | Revoke all tokens for user | Admin |
| POST | `/auth/admin/users/{user_id}/force-password-reset` | Force password reset | Admin |
| POST | `/auth/admin/users/{user_id}/disable-mfa` | Emergency MFA disable | Admin |
| GET | `/auth/admin/sessions` | List active sessions | Admin |
| DELETE | `/auth/admin/sessions/{session_id}` | Terminate session | Admin |
| GET | `/auth/admin/audit-log` | Query auth audit log | Admin |
| GET | `/auth/admin/lockouts` | List locked accounts | Admin |

---

## Endpoints

### GET /auth/admin/users

Retrieve a paginated list of user accounts. Supports filtering by tenant, status, and role.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenant_id` | string | - | Filter by tenant identifier |
| `status` | string | - | Filter by account status: `active`, `locked`, `disabled`, `pending`, `suspended` |
| `role` | string | - | Filter by role |
| `limit` | integer | 20 | Items per page (1-100) |
| `offset` | integer | 0 | Number of items to skip |

**Response (200 OK):**

```json
{
  "users": [
    {
      "user_id": "usr_001",
      "email": "admin@acme.com",
      "name": "Platform Admin",
      "tenant_id": "t-acme-corp",
      "status": "active",
      "roles": ["admin"],
      "mfa_enabled": true,
      "last_login_at": "2026-04-04T10:30:00Z",
      "created_at": "2025-01-15T08:00:00Z",
      "locked_at": null,
      "lock_reason": null
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "has_next": false
}
```

---

### GET /auth/admin/users/{user_id}

Retrieve detailed information about a specific user, including active session count, failed login attempts, and MFA configuration.

**Response (200 OK):**

```json
{
  "user_id": "usr_001",
  "email": "analyst@acme.com",
  "name": "Jane Analyst",
  "tenant_id": "t-acme-corp",
  "org_id": "org-sustainability",
  "status": "active",
  "roles": ["emissions_analyst"],
  "permissions": ["emissions:list", "emissions:read"],
  "mfa_enabled": true,
  "mfa_methods": ["totp"],
  "active_sessions": 2,
  "failed_login_attempts": 0,
  "last_login_at": "2026-04-04T09:15:00Z",
  "last_login_ip": "10.0.0.42",
  "created_at": "2025-06-01T08:00:00Z",
  "updated_at": "2026-04-01T14:30:00Z",
  "locked_at": null,
  "lock_reason": null
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 404 | User not found |

---

### POST /auth/admin/users/{user_id}/unlock

Unlock a user account locked due to failed login attempts or administrative action. Resets the failed attempt counter and transitions status from `locked` to `active`.

**Response (200 OK):**

```json
{
  "user_id": "usr_001",
  "previous_status": "locked",
  "current_status": "active",
  "unlocked_at": "2026-04-04T10:45:00Z",
  "unlocked_by": "admin_user",
  "message": "Account 'usr_001' has been unlocked successfully"
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 404 | User not found |
| 409 | Account is not locked |

---

### POST /auth/admin/users/{user_id}/revoke-tokens

Immediately revoke all active JWT access tokens and refresh tokens for the specified user. Active sessions are also terminated. This is the emergency action for compromised accounts.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reason` | string | `"admin_revoke"` | Reason for revocation (logged in audit trail) |

**Response (200 OK):**

```json
{
  "user_id": "usr_001",
  "tokens_revoked": 3,
  "sessions_terminated": 2,
  "revoked_at": "2026-04-04T10:50:00Z",
  "revoked_by": "admin_user",
  "reason": "suspected_compromise"
}
```

---

### POST /auth/admin/users/{user_id}/force-password-reset

Force a password reset for the specified user. All existing tokens are revoked and a reset email is triggered.

**Response (200 OK):**

```json
{
  "user_id": "usr_001",
  "reset_token_sent": true,
  "forced_at": "2026-04-04T10:55:00Z",
  "forced_by": "admin_user",
  "message": "Password reset forced for 'usr_001'. All tokens revoked. Reset email sent."
}
```

---

### POST /auth/admin/users/{user_id}/disable-mfa

Emergency MFA disable for a user who has lost access to their second factor. This is a privileged operation logged in the audit trail.

**Response (200 OK):**

```json
{
  "user_id": "usr_001",
  "previous_mfa_methods": ["totp"],
  "disabled_at": "2026-04-04T11:00:00Z",
  "disabled_by": "admin_user",
  "message": "MFA disabled for 'usr_001'. User should re-enroll at next login."
}
```

---

### GET /auth/admin/sessions

List currently active sessions with optional filtering by user or tenant.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | - | Filter by user identifier |
| `tenant_id` | string | - | Filter by tenant identifier |
| `limit` | integer | 20 | Items per page (1-100) |
| `offset` | integer | 0 | Number of items to skip |

**Response (200 OK):**

```json
{
  "sessions": [
    {
      "session_id": "sess_abc123",
      "user_id": "usr_001",
      "tenant_id": "t-acme-corp",
      "client_ip": "10.0.0.42",
      "user_agent": "Mozilla/5.0...",
      "created_at": "2026-04-04T08:00:00Z",
      "last_activity_at": "2026-04-04T10:30:00Z",
      "expires_at": "2026-04-04T20:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "has_next": false
}
```

---

### DELETE /auth/admin/sessions/{session_id}

Forcefully terminate a specific active session.

**Response (200 OK):**

```json
{
  "session_id": "sess_abc123",
  "user_id": "usr_001",
  "terminated_at": "2026-04-04T11:05:00Z",
  "terminated_by": "admin_user"
}
```

---

### GET /auth/admin/audit-log

Query the authentication audit log with filtering by user, event type, and time range. Results are ordered newest-first. Backed by TimescaleDB hypertable for efficient time-range scans.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | string | - | Filter by user identifier |
| `event_type` | string | - | Filter by event type (see below) |
| `tenant_id` | string | - | Filter by tenant identifier |
| `start` | datetime | - | Events after this timestamp (ISO 8601) |
| `end` | datetime | - | Events before this timestamp (ISO 8601) |
| `limit` | integer | 50 | Items per page (1-200) |
| `offset` | integer | 0 | Number of items to skip |

**Supported Event Types:** `login_success`, `login_failure`, `logout`, `token_issued`, `token_revoked`, `password_changed`, `password_reset`, `mfa_enabled`, `mfa_disabled`, `account_locked`, `account_unlocked`, `session_created`, `session_terminated`, `permission_changed`

**Response (200 OK):**

```json
{
  "entries": [
    {
      "event_id": "evt_abc123",
      "event_type": "login_success",
      "user_id": "usr_001",
      "tenant_id": "t-acme-corp",
      "client_ip": "10.0.0.42",
      "user_agent": "Mozilla/5.0...",
      "details": {},
      "timestamp": "2026-04-04T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 50,
  "has_next": false
}
```

---

### GET /auth/admin/lockouts

Retrieve all user accounts that are currently in the `locked` state.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenant_id` | string | - | Filter by tenant identifier |

**Response (200 OK):**

```json
{
  "lockouts": [
    {
      "user_id": "usr_002",
      "email": "user@acme.com",
      "tenant_id": "t-acme-corp",
      "locked_at": "2026-04-04T09:00:00Z",
      "lock_reason": "excessive_failed_attempts",
      "failed_attempts": 11
    }
  ],
  "total": 1
}
```

---

## Authorization

All endpoints in this router require admin-level access. The `_require_admin` dependency checks:

1. `request.state.auth` is present (401 if missing)
2. User has `admin:*` permission OR `admin`/`super_admin` role (403 if missing)
