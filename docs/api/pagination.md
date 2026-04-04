# GreenLang API Pagination Patterns

## Overview

GreenLang APIs use **offset-based pagination** for list endpoints.  The
implementation is consistent across all paginated resources: agent registry,
WAF rules, RBAC roles, runs, and more.

---

## Offset-Based Pagination

### Request Parameters

All list endpoints accept these query parameters:

| Parameter | Type | Default | Min | Max | Description |
|-----------|------|---------|-----|-----|-------------|
| `page` | integer | 1 | 1 | -- | Page number (1-indexed) |
| `page_size` | integer | 20 | 1 | 100 | Number of items per page |
| `sort_by` | string | `created_at` | -- | -- | Field to sort by |
| `sort_order` | string | `desc` | -- | -- | Sort direction: `asc` or `desc` |

### Response Envelope

All paginated responses use the following structure:

```json
{
  "items": [
    { "id": "agent_001", "name": "Stationary Combustion Agent", "..." : "..." },
    { "id": "agent_002", "name": "Mobile Combustion Agent", "..." : "..." }
  ],
  "total": 101,
  "page": 1,
  "page_size": 20,
  "total_pages": 6,
  "has_next": true,
  "has_prev": false
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `items` | array | Array of resource objects for the current page. The key name may vary by endpoint (e.g., `agents`, `rules`, `runs`). |
| `total` | integer | Total number of matching items across all pages |
| `page` | integer | Current page number (1-indexed) |
| `page_size` | integer | Number of items per page (as requested) |
| `total_pages` | integer | Total number of pages (`ceil(total / page_size)`) |
| `has_next` | boolean | `true` if there are more pages after the current page |
| `has_prev` | boolean | `true` if there are pages before the current page |

---

## Examples

### Basic Pagination

Retrieve the first page with default settings:

```http
GET /api/v1/agents?page=1&page_size=20
Authorization: Bearer <token>
```

**Response:**

```json
{
  "agents": [
    {
      "agent_id": "gl_stationary_combustion",
      "name": "Stationary Combustion Agent",
      "domain": "mrv",
      "created_at": "2026-03-01T10:00:00Z"
    },
    {
      "agent_id": "gl_mobile_combustion",
      "name": "Mobile Combustion Agent",
      "domain": "mrv",
      "created_at": "2026-03-01T10:05:00Z"
    }
  ],
  "total": 101,
  "page": 1,
  "page_size": 20,
  "total_pages": 6,
  "has_next": true,
  "has_prev": false
}
```

### Navigating Pages

Retrieve the next page:

```http
GET /api/v1/agents?page=2&page_size=20
Authorization: Bearer <token>
```

Retrieve the last page:

```http
GET /api/v1/agents?page=6&page_size=20
Authorization: Bearer <token>
```

### Custom Page Size

Retrieve 50 items per page:

```http
GET /api/v1/agents?page=1&page_size=50
Authorization: Bearer <token>
```

### Filtering with Pagination

Combine filters with pagination:

```http
GET /api/v1/agents?domain=mrv&lifecycle_state=active&page=1&page_size=10&sort_by=name&sort_order=asc
Authorization: Bearer <token>
```

---

## SQL Implementation Pattern

Internally, pagination is implemented using SQL `LIMIT` and `OFFSET`:

```sql
SELECT *
FROM agents
WHERE domain = $1 AND lifecycle_state = $2
ORDER BY created_at DESC
LIMIT $3 OFFSET $4
```

Where:
- `$3` = `page_size`
- `$4` = `(page - 1) * page_size`

The total count is obtained separately:

```sql
SELECT COUNT(*) FROM agents WHERE domain = $1 AND lifecycle_state = $2
```

---

## Pagination in Python

### Using the GreenLang SDK

```python
from greenlang.sdk import Client

client = Client(api_key="glk_...")

# Fetch first page
result = client.agents.list(page=1, page_size=20, domain="mrv")
print(f"Total agents: {result.total}")
print(f"Page {result.page} of {result.total_pages}")

# Iterate through all pages
all_agents = []
page = 1
while True:
    result = client.agents.list(page=page, page_size=100)
    all_agents.extend(result.agents)
    if not result.has_next:
        break
    page += 1

print(f"Fetched {len(all_agents)} agents")
```

### Using requests

```python
import requests

headers = {"Authorization": f"Bearer {token}"}
base_url = "https://api.greenlang.io/api/v1/agents"

# Iterate all pages
all_agents = []
page = 1

while True:
    response = requests.get(
        base_url,
        headers=headers,
        params={"page": page, "page_size": 100, "sort_by": "name", "sort_order": "asc"}
    )
    data = response.json()
    all_agents.extend(data["agents"])

    if not data["has_next"]:
        break
    page += 1

print(f"Total: {len(all_agents)} agents")
```

### Using cURL

```bash
# Page 1
curl -s "https://api.greenlang.io/api/v1/agents?page=1&page_size=20" \
  -H "Authorization: Bearer $TOKEN" | jq '.total_pages'

# Page 3
curl -s "https://api.greenlang.io/api/v1/agents?page=3&page_size=20" \
  -H "Authorization: Bearer $TOKEN" | jq '.agents[].name'
```

---

## Paginated Endpoints

The following endpoints support pagination:

| Endpoint | Items Key | Sortable Fields |
|----------|-----------|-----------------|
| `GET /api/v1/agents` | `agents` | `created_at`, `name`, `domain`, `type` |
| `GET /api/v1/runs` | `runs` | `created_at`, `status`, `app_id` |
| `GET /api/v1/roles` | `roles` | `name`, `created_at` |
| `GET /api/v1/waf/rules` | `rules` | `created_at`, `severity`, `name` |
| `GET /api/v1/waf/attacks` | `attacks` | `detected_at`, `severity` |
| `GET /api/v1/audit/events` | `events` | `timestamp`, `event_type` |
| `GET /api/v1/api-keys` | `keys` | `created_at`, `name`, `last_used_at` |

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| `page` exceeds `total_pages` | Returns empty `items` array with correct metadata |
| `page_size` exceeds 100 | Clamped to 100 (server enforces `le=100` validation) |
| `page_size` is 0 or negative | Returns HTTP 422 validation error |
| No results match filters | Returns `{"items": [], "total": 0, "page": 1, "page_size": 20, "total_pages": 0, "has_next": false, "has_prev": false}` |
| Concurrent modifications | Offset-based pagination may skip or duplicate items if data changes between page requests. For real-time consistency, use the audit event stream instead. |

---

## Performance Considerations

1. **Prefer smaller page sizes** for initial rendering (20 is a good default).
2. **Use `page_size=100`** (the maximum) when fetching all data in a script.
3. **Add filters** to reduce the total result set before paginating.
4. **Sort by indexed columns** (`created_at`, `name`) for best database performance.
5. **Cache total counts** on the client side when iterating pages -- the count query is executed on every request.

---

## Source Files

| File | Purpose |
|------|---------|
| `greenlang/config/greenlang_registry/models.py` | `ListAgentsQuery` (page, page_size, sort_by, sort_order) and `ListAgentsResponse` |
| `greenlang/config/greenlang_registry/api/routes.py` | Agent registry list endpoint with offset pagination |
| `greenlang/config/registry/api.py` | Registry API with page/page_size query parameters |
| `greenlang/infrastructure/waf_management/api/waf_routes.py` | WAF rules and attacks pagination |
| `greenlang/infrastructure/rbac_service/api/roles_routes.py` | RBAC roles pagination |
