# AGENT-FOUND-004: Assumptions Registry API Reference

**Agent ID:** AGENT-FOUND-004
**Service:** Assumptions Registry
**Status:** Production Ready
**Base Path:** `/api/v1/assumptions`
**Tag:** `assumptions`
**Source:** `greenlang/agents/foundation/assumptions/api/router.py`

The Assumptions Registry provides endpoints for managing calculation assumptions,
scenario modeling, validation, version tracking, dependency analysis, sensitivity
analysis, and export/import.

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | GET | `/health` | Health check | 200 |
| 2 | GET | `/metrics` | Metrics summary | 200, 503 |
| 3 | POST | `/` | Create assumption | 200, 400, 503 |
| 4 | GET | `/` | List assumptions | 200, 503 |
| 5 | GET | `/{assumption_id}` | Get assumption | 200, 404, 503 |
| 6 | PUT | `/{assumption_id}` | Update assumption | 200, 400, 503 |
| 7 | DELETE | `/{assumption_id}` | Delete assumption | 200, 400, 404, 503 |
| 8 | GET | `/{assumption_id}/versions` | Get version history | 200, 404, 503 |
| 9 | GET | `/{assumption_id}/value` | Get resolved value | 200, 404, 503 |
| 10 | PUT | `/{assumption_id}/value` | Set assumption value | 200, 400, 503 |
| 11 | POST | `/validate` | Validate a value | 200, 404, 503 |
| 12 | POST | `/scenarios` | Create scenario | 200, 400, 503 |
| 13 | GET | `/scenarios/list` | List scenarios | 200, 503 |
| 14 | GET | `/scenarios/{scenario_id}` | Get scenario | 200, 404, 503 |
| 15 | PUT | `/scenarios/{scenario_id}` | Update scenario | 200, 400, 503 |
| 16 | DELETE | `/scenarios/{scenario_id}` | Delete scenario | 200, 400, 404, 503 |
| 17 | GET | `/{assumption_id}/dependencies` | Get dependency graph | 200, 503 |
| 18 | GET | `/{assumption_id}/sensitivity` | Get sensitivity analysis | 200, 404, 503 |
| 19 | POST | `/export` | Export all assumptions | 200, 503 |
| 20 | POST | `/import` | Import assumptions | 200, 503 |

---

## Detailed Endpoints

### POST / -- Create Assumption

Create a new assumption with value, type, unit, and validation rules.

**Request Body:**

```json
{
  "assumption_id": "grid_ef_uk_2025",
  "name": "UK Grid Emission Factor 2025",
  "description": "Average grid emission factor for the UK in 2025",
  "category": "emission_factor",
  "data_type": "float",
  "value": 0.233,
  "unit": "kgCO2e/kWh",
  "default_value": 0.25,
  "user_id": "analyst@company.com",
  "change_reason": "Initial creation from DEFRA 2025 data",
  "metadata_source": "defra_2025",
  "metadata_tags": ["uk", "grid", "electricity", "2025"],
  "validation_rules": [
    { "rule_type": "range", "min": 0.0, "max": 1.5 },
    { "rule_type": "type", "expected_type": "float" }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `assumption_id` | string | Yes | Unique identifier |
| `name` | string | Yes | Human-readable name |
| `description` | string | No | Detailed description |
| `category` | string | No | Category (default: `custom`) |
| `data_type` | string | No | Data type (default: `float`) |
| `value` | any | Yes | Initial value |
| `unit` | string | No | Unit of measurement |
| `default_value` | any | No | Default fallback value |
| `user_id` | string | No | Creator (default: `system`) |
| `change_reason` | string | No | Reason for creation |
| `metadata_source` | string | No | Data source identifier |
| `metadata_tags` | array | No | Searchable tags |
| `validation_rules` | array | No | Validation rules |

**Response (200):**

```json
{
  "assumption_id": "grid_ef_uk_2025",
  "name": "UK Grid Emission Factor 2025",
  "current_value": 0.233,
  "unit": "kgCO2e/kWh",
  "category": "emission_factor",
  "version": 1,
  "created_at": "2026-04-04T10:00:00Z",
  "updated_at": "2026-04-04T10:00:00Z",
  "provenance_hash": "sha256:..."
}
```

---

### GET / -- List Assumptions

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | string | Filter by category |
| `search` | string | Search in name/description |

**Response (200):**

```json
{
  "assumptions": [
    {
      "assumption_id": "grid_ef_uk_2025",
      "name": "UK Grid Emission Factor 2025",
      "current_value": 0.233,
      "category": "emission_factor"
    }
  ],
  "count": 1
}
```

---

### GET /{assumption_id}/value -- Get Resolved Value

Get the resolved value for an assumption, optionally applying scenario overrides.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `scenario_id` | string | Scenario ID for override resolution |

**Response (200):**

```json
{
  "assumption_id": "grid_ef_uk_2025",
  "value": 0.20,
  "value_source": "scenario_override",
  "scenario_id": "optimistic_2030"
}
```

---

### POST /validate -- Validate a Value

Validate a candidate value against an assumption's validation rules.

**Request Body:**

```json
{
  "assumption_id": "grid_ef_uk_2025",
  "value": 2.5
}
```

**Response (200):**

```json
{
  "valid": false,
  "errors": [
    {
      "rule_type": "range",
      "message": "Value 2.5 exceeds maximum 1.5"
    }
  ],
  "warnings": []
}
```

---

### POST /scenarios -- Create Scenario

Create a named scenario with value overrides for what-if analysis.

**Request Body:**

```json
{
  "name": "Net Zero 2050",
  "description": "Optimistic scenario with aggressive decarbonization",
  "scenario_type": "custom",
  "overrides": {
    "grid_ef_uk_2025": 0.10,
    "transport_ef_diesel": 0.05
  },
  "user_id": "analyst@company.com",
  "tags": ["net-zero", "2050", "optimistic"]
}
```

**Response (200):**

```json
{
  "scenario_id": "scn_abc123",
  "name": "Net Zero 2050",
  "scenario_type": "custom",
  "overrides": {
    "grid_ef_uk_2025": 0.10,
    "transport_ef_diesel": 0.05
  },
  "is_active": true,
  "created_at": "2026-04-04T10:00:00Z"
}
```

---

### GET /{assumption_id}/dependencies -- Get Dependency Graph

Get the dependency graph showing what calculations depend on this assumption.

**Response (200):**

```json
{
  "assumption_id": "grid_ef_uk_2025",
  "node": {
    "assumption_id": "grid_ef_uk_2025",
    "depends_on": ["defra_conversion_factors"],
    "used_by": ["scope2_location_based", "scope2_market_based"]
  },
  "impact": {
    "direct_dependents": 2,
    "transitive_dependents": 5
  }
}
```

---

### GET /{assumption_id}/sensitivity -- Sensitivity Analysis

Get sensitivity analysis showing how assumption variations affect outcomes.

**Response (200):**

```json
{
  "assumption_id": "grid_ef_uk_2025",
  "baseline_value": 0.233,
  "scenario_values": {
    "Net Zero 2050": 0.10,
    "BAU 2030": 0.20,
    "Conservative": 0.30
  },
  "dependency_count": 2,
  "dependent_calculations": ["scope2_location_based", "scope2_market_based"],
  "min_value": 0.10,
  "max_value": 0.30,
  "range": 0.20,
  "range_pct": 85.84
}
```

---

### POST /export -- Export All Assumptions

**Request Body:**

```json
{
  "user_id": "admin@company.com"
}
```

**Response (200):** Full export object containing all assumptions, versions, and metadata.

---

### POST /import -- Import Assumptions

**Request Body:**

```json
{
  "data": { ... },
  "user_id": "admin@company.com"
}
```

**Response (200):**

```json
{
  "imported_count": 42
}
```

---

## Common Error Responses

**400 Bad Request:**

```json
{
  "detail": "Assumption grid_ef_uk_2025 already exists"
}
```

**404 Not Found:**

```json
{
  "detail": "Assumption grid_ef_unknown not found"
}
```
