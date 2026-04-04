# Feature Flags API Reference (INFRA-008)

## Overview

The Feature Flags Service provides a complete feature flag management system with support for boolean, percentage-based, and multivariate flags. It includes rollout control, kill switch for emergency disabling, user/tenant overrides, variant support, audit trail, and metrics.

**Router Prefix:** `/api/v1/flags`
**Tags:** `Feature Flags`
**Source:** `greenlang/infrastructure/feature_flags/api/router.py`

---

## Endpoint Summary

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/api/v1/flags` | List feature flags (paginated) | Yes |
| POST | `/api/v1/flags` | Create a feature flag | Yes |
| GET | `/api/v1/flags/stale` | List stale flags | Yes |
| GET | `/api/v1/flags/{key}` | Get flag details | Yes |
| PUT | `/api/v1/flags/{key}` | Update flag | Yes |
| DELETE | `/api/v1/flags/{key}` | Archive flag (soft delete) | Yes |
| POST | `/api/v1/flags/{key}/evaluate` | Evaluate flag for context | Yes |
| POST | `/api/v1/flags/evaluate-batch` | Batch evaluate multiple flags | Yes |
| PUT | `/api/v1/flags/{key}/rollout` | Set rollout percentage | Yes |
| POST | `/api/v1/flags/{key}/kill` | Activate kill switch | Yes |
| POST | `/api/v1/flags/{key}/restore` | Deactivate kill switch | Yes |
| GET | `/api/v1/flags/{key}/audit` | Get audit trail | Yes |
| GET | `/api/v1/flags/{key}/metrics` | Get flag metrics | Yes |
| POST | `/api/v1/flags/{key}/variants` | Add/update variant | Yes |
| POST | `/api/v1/flags/{key}/overrides` | Set scoped override | Yes |
| DELETE | `/api/v1/flags/{key}/overrides` | Clear scoped override | Yes |
| GET | `/api/v1/flags/system/statistics` | Get system statistics | Yes |

---

## Endpoints

### GET /api/v1/flags

Retrieve a paginated, filterable list of feature flags.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | - | Filter by status: `draft`, `active`, `disabled`, `archived`, `permanent` |
| `flag_type` | string | - | Filter by flag type |
| `tag` | string | - | Filter by tag |
| `owner` | string | - | Filter by owner |
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (1-100) |

**Response (200 OK):**

```json
{
  "items": [
    {
      "key": "enable-scope3-v2",
      "name": "Enable Scope 3 V2 Engine",
      "description": "Rolls out the new Scope 3 calculation engine.",
      "flag_type": "boolean",
      "status": "active",
      "default_value": false,
      "rollout_percentage": 25,
      "environments": ["staging", "production"],
      "tags": ["scope3", "engine"],
      "owner": "platform-team",
      "metadata": {},
      "start_time": null,
      "end_time": null,
      "created_at": "2026-01-15T08:00:00Z",
      "updated_at": "2026-04-01T14:30:00Z",
      "version": 5
    }
  ],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "total_pages": 3,
  "has_next": true,
  "has_prev": false
}
```

---

### POST /api/v1/flags

Create a new feature flag definition. The flag is created in `draft` status.

**Request Body:**

```json
{
  "key": "enable-scope3-v2",
  "name": "Enable Scope 3 V2 Engine",
  "description": "Rolls out the new Scope 3 calculation engine.",
  "flag_type": "boolean",
  "default_value": false,
  "rollout_percentage": 0,
  "environments": ["staging"],
  "tags": ["scope3", "engine"],
  "owner": "platform-team",
  "metadata": {},
  "start_time": null,
  "end_time": null
}
```

**Response (201 Created):** Same as single flag response.

**Error Responses:**

| Status | Description |
|--------|-------------|
| 409 | Flag with the same key already exists |
| 422 | Invalid flag_type |

---

### POST /api/v1/flags/{key}/evaluate

Evaluate a flag for a given user/tenant context. This is the runtime endpoint used by application code to check flag state.

**Request Body:**

```json
{
  "user_id": "u-analyst-01",
  "tenant_id": "t-acme-corp",
  "environment": "production",
  "user_segments": ["beta-testers"],
  "user_attributes": {"plan": "enterprise", "region": "eu"}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | No | User ID for percentage-based rollout |
| `tenant_id` | string | No | Tenant ID for tenant-scoped evaluation |
| `environment` | string | No | Environment context |
| `user_segments` | array | No | User segments for targeting |
| `user_attributes` | object | No | User attributes for targeting rules |

**Response (200 OK):**

```json
{
  "flag_key": "enable-scope3-v2",
  "enabled": true,
  "reason": "rollout_percentage",
  "variant_key": null,
  "metadata": {},
  "duration_us": 42
}
```

| Field | Type | Description |
|-------|------|-------------|
| `flag_key` | string | The evaluated flag key |
| `enabled` | boolean | Whether the flag is enabled for this context |
| `reason` | string | Why this decision was made |
| `variant_key` | string | Selected variant key (for multivariate flags) |
| `metadata` | object | Additional evaluation metadata |
| `duration_us` | integer | Evaluation duration in microseconds |

---

### POST /api/v1/flags/evaluate-batch

Evaluate multiple flags in a single request.

**Request Body:**

```json
{
  "flag_keys": ["enable-scope3-v2", "dark-mode", "new-dashboard"],
  "context": {
    "user_id": "u-analyst-01",
    "tenant_id": "t-acme-corp",
    "environment": "production"
  }
}
```

**Response (200 OK):**

```json
{
  "results": {
    "enable-scope3-v2": {
      "flag_key": "enable-scope3-v2",
      "enabled": true,
      "reason": "rollout_percentage",
      "variant_key": null,
      "metadata": {},
      "duration_us": 32
    },
    "dark-mode": {
      "flag_key": "dark-mode",
      "enabled": false,
      "reason": "default",
      "variant_key": null,
      "metadata": {},
      "duration_us": 8
    }
  },
  "total_duration_us": 42
}
```

---

### PUT /api/v1/flags/{key}/rollout

Set the rollout percentage for a flag.

**Request Body:**

```json
{
  "percentage": 50,
  "updated_by": "platform-team"
}
```

---

### POST /api/v1/flags/{key}/kill

Emergency: immediately disable a flag via the kill switch. The flag is disabled across all environments and overrides.

**Request Body:**

```json
{
  "actor": "oncall-engineer",
  "reason": "Performance regression detected in staging"
}
```

**Response (200 OK):**

```json
{
  "flag_key": "enable-scope3-v2",
  "killed": true,
  "actor": "oncall-engineer"
}
```

---

### POST /api/v1/flags/{key}/restore

Deactivate the kill switch and restore a killed flag to active status.

---

### POST /api/v1/flags/{key}/overrides

Set a scoped override for a flag. Overrides take precedence over the default flag value for the specified scope.

**Request Body:**

```json
{
  "scope_type": "user",
  "scope_value": "u-analyst-01",
  "enabled": true,
  "variant_key": null,
  "expires_at": "2026-05-01T00:00:00Z",
  "created_by": "admin"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scope_type` | string | Yes | Override scope: `user`, `tenant`, `environment` |
| `scope_value` | string | Yes | Scope value (e.g., user ID, tenant ID) |
| `enabled` | boolean | Yes | Override enabled state |
| `variant_key` | string | No | Override variant selection |
| `expires_at` | datetime | No | Override expiration |
| `created_by` | string | No | Actor who created the override |

---

### GET /api/v1/flags/{key}/audit

Get the paginated audit trail for a flag, showing all changes.

**Response (200 OK):**

```json
{
  "items": [
    {
      "flag_key": "enable-scope3-v2",
      "action": "rollout_updated",
      "old_value": {"rollout_percentage": 25},
      "new_value": {"rollout_percentage": 50},
      "changed_by": "platform-team",
      "change_reason": "Expanding rollout after successful canary",
      "ip_address": "10.0.0.42",
      "created_at": "2026-04-04T14:00:00Z"
    }
  ],
  "total": 5,
  "page": 1,
  "page_size": 20
}
```
