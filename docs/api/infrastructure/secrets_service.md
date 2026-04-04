# Secrets Management Service API Reference (SEC-006)

## Overview

The Secrets Management Service provides Vault-backed secret storage with tenant isolation, version history, soft delete/undelete, manual and scheduled rotation, and health monitoring. All endpoints enforce tenant isolation and emit audit logs.

**Router Prefix:** `/api/v1/secrets`
**Tags:** `Secrets`
**Source:** `greenlang/infrastructure/secrets_service/api/`

---

## Endpoint Summary

### Secret CRUD

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secrets/` | List secrets (metadata only, paginated) | Yes |
| GET | `/api/v1/secrets/{path}` | Get secret value | Yes |
| POST | `/api/v1/secrets/{path}` | Create secret | Yes |
| PUT | `/api/v1/secrets/{path}` | Update secret | Yes |
| DELETE | `/api/v1/secrets/{path}` | Soft delete secret | Yes |
| GET | `/api/v1/secrets/{path}/versions` | Version history | Yes |
| POST | `/api/v1/secrets/{path}/undelete` | Restore deleted version | Yes |

### Rotation

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/secrets/rotate/{path}` | Trigger manual rotation | Yes |
| GET | `/api/v1/secrets/rotation/status` | Current rotation status | Yes |
| GET | `/api/v1/secrets/rotation/schedule` | Rotation schedule | Yes |
| POST | `/api/v1/secrets/rotation/schedule` | Update rotation schedule | Yes |

### Health

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/secrets/health` | Vault health (sealed, initialized, standby) | No |
| GET | `/api/v1/secrets/status` | Service status (connected, authenticated) | No |
| GET | `/api/v1/secrets/stats` | Operation statistics | Yes |

---

## Secret CRUD Endpoints

### GET /api/v1/secrets/

List secrets with metadata only (values are not returned in list view). Supports pagination and tenant filtering.

**Response (200 OK):**

```json
{
  "secrets": [
    {
      "path": "database/prod/credentials",
      "tenant_id": "t-acme-corp",
      "version": 3,
      "created_at": "2025-06-01T08:00:00Z",
      "updated_at": "2026-03-15T10:00:00Z",
      "rotation_enabled": true,
      "next_rotation_at": "2026-04-15T00:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20
}
```

---

### GET /api/v1/secrets/{path}

Get the current value of a secret. The `path` parameter supports nested paths (e.g., `database/prod/credentials`).

**Response (200 OK):**

```json
{
  "path": "database/prod/credentials",
  "value": {
    "username": "greenlang_prod",
    "password": "s3cure-db-pass",
    "host": "db.greenlang.internal",
    "port": 5432
  },
  "version": 3,
  "tenant_id": "t-acme-corp",
  "created_at": "2026-03-15T10:00:00Z"
}
```

---

### POST /api/v1/secrets/{path}

Create a new secret at the specified path.

**Request Body:**

```json
{
  "value": {
    "username": "greenlang_prod",
    "password": "s3cure-db-pass"
  },
  "metadata": {
    "rotation_interval_days": 30,
    "owner": "platform-team"
  }
}
```

---

### DELETE /api/v1/secrets/{path}

Soft delete a secret. The secret data is retained but marked as deleted. Use the undelete endpoint to restore.

---

## Rotation Endpoints

### POST /api/v1/secrets/rotate/{path}

Trigger manual rotation of a secret. Creates a new version with a generated value.

**Response (200 OK):**

```json
{
  "path": "database/prod/credentials",
  "previous_version": 3,
  "new_version": 4,
  "rotated_at": "2026-04-04T12:00:00Z"
}
```

---

## Health Endpoints

### GET /api/v1/secrets/health

Check Vault backend health status.

**Response (200 OK):**

```json
{
  "sealed": false,
  "initialized": true,
  "standby": false,
  "server_time_utc": "2026-04-04T12:00:00Z",
  "version": "1.15.0"
}
```

### GET /api/v1/secrets/status

Check service-level connectivity and authentication.

**Response (200 OK):**

```json
{
  "connected": true,
  "authenticated": true,
  "token_ttl_seconds": 3200,
  "backend": "vault"
}
```
