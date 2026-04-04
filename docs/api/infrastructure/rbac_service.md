# RBAC Authorization Service API Reference (SEC-002)

## Overview

The RBAC (Role-Based Access Control) Authorization Service provides role management, permission management, role-to-user assignments, and runtime permission checking. It supports hierarchical roles with permission inheritance, ABAC-style conditional permissions, time-limited assignments, and sub-millisecond authorization decisions with caching.

**Router Prefix:** `/api/v1/rbac`
**Tags:** `RBAC Roles`, `RBAC Permissions`, `RBAC Assignments`, `RBAC Authorization Check`
**Source:** `greenlang/infrastructure/rbac_service/api/`

---

## Endpoint Summary

### Roles

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/rbac/roles` | List roles (paginated) | Yes |
| POST | `/api/v1/rbac/roles` | Create role | Yes |
| GET | `/api/v1/rbac/roles/{role_id}` | Get role details | Yes |
| PUT | `/api/v1/rbac/roles/{role_id}` | Update role | Yes |
| DELETE | `/api/v1/rbac/roles/{role_id}` | Delete role | Yes |
| GET | `/api/v1/rbac/roles/{role_id}/permissions` | List role permissions | Yes |

### Permissions

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/rbac/permissions` | List all permissions | Yes |
| POST | `/api/v1/rbac/roles/{role_id}/permissions` | Grant permission to role | Yes |
| DELETE | `/api/v1/rbac/roles/{role_id}/permissions/{permission_id}` | Revoke permission from role | Yes |

### Assignments

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/rbac/assignments` | List assignments (paginated) | Yes |
| POST | `/api/v1/rbac/assignments` | Assign role to user | Yes |
| DELETE | `/api/v1/rbac/assignments/{assignment_id}` | Revoke assignment | Yes |
| GET | `/api/v1/rbac/users/{user_id}/roles` | Get user's roles | Yes |
| GET | `/api/v1/rbac/users/{user_id}/permissions` | Get user's effective permissions | Yes |

### Authorization Check

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/api/v1/rbac/check` | Evaluate authorization decision | Yes |

---

## Role Endpoints

### GET /api/v1/rbac/roles

Retrieve a paginated list of RBAC roles, optionally filtered by tenant.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenant_id` | string | - | Filter by tenant ID |
| `include_system` | boolean | `true` | Include system-defined roles |
| `page` | integer | 1 | Page number (1-indexed) |
| `page_size` | integer | 20 | Items per page (1-100) |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": "r-emissions-analyst",
      "name": "emissions_analyst",
      "display_name": "Emissions Analyst",
      "description": "Can view and analyze emissions data.",
      "parent_role_id": null,
      "is_system": false,
      "is_enabled": true,
      "tenant_id": "t-acme-corp",
      "metadata": {"department": "sustainability"},
      "created_at": "2025-06-01T08:00:00Z",
      "updated_at": "2026-01-15T14:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "total_pages": 1,
  "has_next": false,
  "has_prev": false
}
```

---

### POST /api/v1/rbac/roles

Create a new RBAC role definition.

**Request Body:**

```json
{
  "name": "emissions_analyst",
  "display_name": "Emissions Analyst",
  "description": "Can view and analyze emissions data.",
  "parent_role_id": null,
  "metadata": {"department": "sustainability"}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique role name (lowercase snake_case, 2-128 chars) |
| `display_name` | string | Yes | Human-readable display name (1-256 chars) |
| `description` | string | No | Role description (max 2048 chars) |
| `parent_role_id` | string | No | Parent role UUID for hierarchy inheritance |
| `metadata` | object | No | Arbitrary key-value metadata |

**Response (201 Created):** Same as single role response.

**Error Responses:**

| Status | Description |
|--------|-------------|
| 409 | Role with the same name already exists |
| 422 | Validation error (invalid name format) |

---

### PUT /api/v1/rbac/roles/{role_id}

Update an existing RBAC role. System roles cannot be modified.

**Request Body:**

```json
{
  "display_name": "Senior Emissions Analyst",
  "description": "Updated description.",
  "is_enabled": true
}
```

All fields are optional. Only provided fields are updated.

**Error Responses:**

| Status | Description |
|--------|-------------|
| 403 | Cannot modify a system-defined role |
| 404 | Role not found |
| 422 | No fields provided for update |

---

### DELETE /api/v1/rbac/roles/{role_id}

Delete an RBAC role. System roles cannot be deleted.

**Response:** 204 No Content

**Error Responses:**

| Status | Description |
|--------|-------------|
| 403 | Cannot delete a system-defined role |
| 404 | Role not found |

---

## Permission Endpoints

### GET /api/v1/rbac/permissions

Retrieve a paginated list of all available permissions, optionally filtered by resource type.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resource` | string | - | Filter by resource type (e.g., `"agents"`, `"reports"`) |
| `page` | integer | 1 | Page number |
| `page_size` | integer | 50 | Items per page (1-100) |

**Response (200 OK):**

```json
{
  "items": [
    {
      "id": "perm-reports-read",
      "resource": "reports",
      "action": "read",
      "description": "Read access to reports",
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "total": 61,
  "page": 1,
  "page_size": 50,
  "total_pages": 2,
  "has_next": true,
  "has_prev": false
}
```

---

### POST /api/v1/rbac/roles/{role_id}/permissions

Grant a permission to a role with optional ABAC-style conditions and scope.

**Request Body:**

```json
{
  "permission_id": "perm-reports-read",
  "effect": "allow",
  "conditions": {"environment": "production"},
  "scope": "tenant"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `permission_id` | string | Yes | UUID of the permission to grant |
| `effect` | string | No | `"allow"` (default) or `"deny"` |
| `conditions` | object | No | ABAC-style conditions for conditional evaluation |
| `scope` | string | No | `"global"`, `"tenant"` (default), or `"resource"` |

**Response (201 Created):**

```json
{
  "id": "grant-abc123",
  "role_id": "r-emissions-analyst",
  "permission_id": "perm-reports-read",
  "effect": "allow",
  "conditions": null,
  "scope": "tenant",
  "granted_by": "admin_user",
  "granted_at": "2026-04-04T11:00:00Z"
}
```

---

## Assignment Endpoints

### POST /api/v1/rbac/assignments

Create a new role assignment for a user within a tenant scope. Supports time-limited assignments.

**Request Body:**

```json
{
  "user_id": "u-analyst-01",
  "role_id": "r-emissions-analyst",
  "tenant_id": "t-acme-corp",
  "expires_at": "2026-12-31T23:59:59Z"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | UUID of the user receiving the role |
| `role_id` | string | Yes | UUID of the role to assign |
| `tenant_id` | string | Yes | UUID of the tenant scope |
| `expires_at` | datetime | No | Expiration for time-limited assignments |

**Response (201 Created):**

```json
{
  "id": "asgn-abc123",
  "user_id": "u-analyst-01",
  "role_id": "r-emissions-analyst",
  "tenant_id": "t-acme-corp",
  "assigned_by": "admin_user",
  "assigned_at": "2026-04-04T11:10:00Z",
  "expires_at": "2026-12-31T23:59:59Z",
  "is_active": true,
  "revoked_at": null,
  "revoked_by": null
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 | Role is disabled and cannot be assigned |
| 404 | Role not found |
| 409 | Assignment already exists |

---

### GET /api/v1/rbac/users/{user_id}/permissions

Resolve all effective permissions for a user within a tenant, considering role hierarchy and inheritance. Returns a flat list of `resource:action` permission strings.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenant_id` | string | - | Tenant scope (falls back to `X-Tenant-Id` header) |

**Response (200 OK):**

```json
{
  "user_id": "u-analyst-01",
  "tenant_id": "t-acme-corp",
  "permissions": [
    "emissions:read",
    "emissions:list",
    "reports:read",
    "reports:download",
    "agents:execute"
  ]
}
```

---

## Authorization Check

### POST /api/v1/rbac/check

Evaluate whether a user has the required permission to perform an action on a resource within a tenant scope. This is the primary runtime authorization endpoint used by the API gateway and services. Results may be served from cache for sub-millisecond latency.

**Request Body:**

```json
{
  "user_id": "u-analyst-01",
  "resource": "reports",
  "action": "read",
  "tenant_id": "t-acme-corp",
  "context": {"environment": "production"}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | Yes | UUID of the user to check |
| `resource` | string | Yes | Resource being accessed (e.g., `"reports"`, `"agents"`) |
| `action` | string | Yes | Action being performed (e.g., `"read"`, `"write"`, `"delete"`) |
| `tenant_id` | string | Yes | UUID of the tenant scope |
| `context` | object | No | ABAC-style context for conditional evaluation |

**Response (200 OK):**

```json
{
  "allowed": true,
  "matched_permissions": ["reports:read"],
  "evaluation_time_ms": 0.432,
  "cache_hit": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | boolean | Whether the action is authorized |
| `matched_permissions` | array | Permissions that matched the request |
| `evaluation_time_ms` | float | Evaluation duration in milliseconds |
| `cache_hit` | boolean | Whether the result was served from cache |
