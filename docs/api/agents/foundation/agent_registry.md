# AGENT-FOUND-007: Agent Registry & Service Catalog API Reference

**Agent ID:** AGENT-FOUND-007
**Service:** Agent Registry & Service Catalog
**Status:** Production Ready
**Base Path:** `/api/v1/agent-registry`
**Tag:** `agent-registry`
**Source:** `greenlang/agents/foundation/agent_registry/api/router.py`

The Agent Registry provides endpoints for registering, discovering, and managing
agents within the GreenLang platform. It includes health monitoring, dependency
resolution, capability matching, hot-reload support, and provenance tracking.

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | GET | `/health` | Health check | 200, 503 |
| 2 | POST | `/agents` | Register agent | 201, 400, 503 |
| 3 | GET | `/agents` | List agents | 200, 400, 503 |
| 4 | GET | `/agents/{agent_id}` | Get agent by ID | 200, 404, 503 |
| 5 | PUT | `/agents/{agent_id}` | Update agent | 200, 400, 404, 503 |
| 6 | DELETE | `/agents/{agent_id}` | Unregister agent | 200, 404, 503 |
| 7 | GET | `/agents/{agent_id}/versions` | List agent versions | 200, 404, 503 |
| 8 | POST | `/agents/{agent_id}/reload` | Hot-reload agent | 200, 400, 503 |
| 9 | GET | `/agents/{agent_id}/health` | Check agent health | 200, 503 |
| 10 | PUT | `/agents/{agent_id}/health` | Set agent health | 200, 400, 404, 503 |
| 11 | GET | `/agents/{agent_id}/health/history` | Get health history | 200, 503 |
| 12 | GET | `/health/unhealthy` | Get unhealthy agents | 200, 503 |
| 13 | GET | `/health/summary` | Get health summary | 200, 503 |
| 14 | POST | `/dependencies/resolve` | Resolve dependencies | 200, 503 |
| 15 | GET | `/dependencies/{agent_id}/dependents` | Get dependents | 200, 503 |
| 16 | GET | `/dependencies/{agent_id}/tree` | Get dependency tree | 200, 503 |
| 17 | GET | `/capabilities/matrix` | Get capability matrix | 200, 503 |
| 18 | GET | `/provenance/{entity_id}` | Get provenance chain | 200, 503 |
| 19 | GET | `/statistics` | Registry statistics | 200, 503 |
| 20 | GET | `/metrics` | Metrics summary | 200, 503 |

---

## Detailed Endpoints

### POST /agents -- Register Agent

Register a new agent in the service catalog.

**Request Body:**

```json
{
  "agent_id": "gl-mrv-scope1-stationary",
  "name": "Scope 1 Stationary Combustion Agent",
  "description": "Calculates Scope 1 GHG emissions from stationary combustion sources",
  "version": "1.0.0",
  "layer": "mrv",
  "sectors": ["energy", "manufacturing", "buildings"],
  "capabilities": [
    {
      "name": "ghg_calculation",
      "description": "GHG emission calculation",
      "input_schema": "emissions/activity@1.3.0",
      "output_schema": "emissions/result@1.0.0"
    }
  ],
  "tags": ["scope1", "stationary", "ghg", "combustion"],
  "execution_mode": "legacy_http",
  "legacy_http_config": {
    "base_url": "http://mrv-scope1:8080",
    "timeout_seconds": 120
  },
  "resource_profile": {
    "cpu_request": "250m",
    "memory_request": "256Mi",
    "cpu_limit": "1000m",
    "memory_limit": "1Gi"
  },
  "dependencies": [
    { "agent_id": "gl-found-normalizer", "required": true },
    { "agent_id": "gl-found-citations", "required": false }
  ],
  "author": "GreenLang Platform Team",
  "documentation_url": "https://docs.greenlang.io/agents/mrv/scope1-stationary"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | string | Yes | Unique agent identifier |
| `name` | string | Yes | Human-readable name |
| `description` | string | Yes | Agent description |
| `version` | string | Yes | Semantic version |
| `layer` | string | Yes | Agent layer (`foundation`, `data`, `mrv`, `eudr`) |
| `sectors` | array | No | Applicable sectors |
| `capabilities` | array | No | Agent capabilities |
| `tags` | array | No | Searchable tags |
| `execution_mode` | string | No | `legacy_http` (default), `container` |
| `legacy_http_config` | object | No | HTTP execution config |
| `container_spec` | object | No | Container spec for GLIP v1 |
| `resource_profile` | object | No | K8s resource requirements |
| `dependencies` | array | No | Agent dependencies |
| `author` | string | No | Agent author |
| `documentation_url` | string | No | Documentation URL |

**Response (201):**

```json
{
  "agent_id": "gl-mrv-scope1-stationary",
  "version": "1.0.0",
  "provenance_hash": "sha256:..."
}
```

---

### GET /agents -- List Agents

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer` | string | Filter by layer |
| `sector` | string | Filter by sector |
| `capability` | string | Filter by capability name |
| `tag` | string | Filter by tag |
| `health` | string | Filter by health status |
| `search` | string | Text search in name/description |
| `limit` | integer | Max results (default: 100, max: 1000) |
| `offset` | integer | Offset (default: 0) |

**Response (200):**

```json
{
  "agents": [
    {
      "agent_id": "gl-mrv-scope1-stationary",
      "name": "Scope 1 Stationary Combustion Agent",
      "version": "1.0.0",
      "layer": "mrv",
      "health_status": "healthy"
    }
  ],
  "count": 1
}
```

---

### POST /agents/{agent_id}/reload -- Hot-Reload Agent

Hot-reload an agent with new metadata without downtime. The agent's configuration
is updated in-place.

**Request Body:** Same schema as `POST /agents` (RegisterAgentRequest).

**Response (200):**

```json
{
  "agent_id": "gl-mrv-scope1-stationary",
  "reloaded": true
}
```

---

### GET /agents/{agent_id}/health -- Check Agent Health

Run a health check probe against a registered agent.

**Response (200):**

```json
{
  "agent_id": "gl-mrv-scope1-stationary",
  "status": "healthy",
  "latency_ms": 12.5,
  "last_check_at": "2026-04-04T10:00:00Z",
  "consecutive_failures": 0,
  "details": {
    "endpoint_reachable": true,
    "response_valid": true
  }
}
```

---

### POST /dependencies/resolve -- Resolve Dependencies

Resolve the dependency graph for a set of agents and return topological
execution order.

**Request Body:**

```json
{
  "agent_ids": ["gl-mrv-scope1-stationary", "gl-mrv-scope2-location"],
  "include_optional": true,
  "fail_on_missing": false
}
```

**Response (200):**

```json
{
  "resolved_order": [
    "gl-found-normalizer",
    "gl-found-citations",
    "gl-mrv-scope1-stationary",
    "gl-mrv-scope2-location"
  ],
  "dependency_count": 4,
  "missing_agents": [],
  "circular_dependencies": []
}
```

---

### GET /dependencies/{agent_id}/tree -- Get Dependency Tree

**Response (200):**

```json
{
  "agent_id": "gl-mrv-scope1-stationary",
  "dependencies": [
    {
      "agent_id": "gl-found-normalizer",
      "required": true,
      "dependencies": []
    },
    {
      "agent_id": "gl-found-citations",
      "required": false,
      "dependencies": []
    }
  ]
}
```

---

### GET /capabilities/matrix -- Capability Matrix

Returns a mapping of capabilities to the agents that provide them.

**Response (200):**

```json
{
  "matrix": {
    "ghg_calculation": ["gl-mrv-scope1-stationary", "gl-mrv-scope2-location"],
    "unit_conversion": ["gl-found-normalizer"],
    "schema_validation": ["gl-found-schema-compiler"]
  },
  "capability_count": 3
}
```

---

### GET /health/summary -- Health Summary

**Response (200):**

```json
{
  "total_agents": 101,
  "healthy": 98,
  "degraded": 2,
  "unhealthy": 1,
  "unknown": 0
}
```
