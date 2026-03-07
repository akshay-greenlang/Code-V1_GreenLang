# AGENT-EUDR-001: API Reference

**Base URL:** `/api/v1/eudr-scm`
**Authentication:** JWT Bearer token (OAuth2) or X-API-Key header
**Content-Type:** `application/json`
**Agent ID:** GL-EUDR-SCM-001
**Version:** 1.0.0

---

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [Pagination](#pagination)
4. [Error Handling](#error-handling)
5. [Graph CRUD Endpoints](#graph-crud-endpoints)
6. [Multi-Tier Mapping Endpoints](#multi-tier-mapping-endpoints)
7. [Traceability Endpoints](#traceability-endpoints)
8. [Risk Assessment Endpoints](#risk-assessment-endpoints)
9. [Gap Analysis Endpoints](#gap-analysis-endpoints)
10. [Visualization Endpoints](#visualization-endpoints)
11. [Supplier Onboarding Endpoints](#supplier-onboarding-endpoints)
12. [System Endpoints](#system-endpoints)
13. [RBAC Permissions Reference](#rbac-permissions-reference)

---

## Authentication

All endpoints (except health check and public onboarding endpoints) require
authentication via one of two methods:

### Bearer Token (JWT)

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
```

### API Key

```http
X-API-Key: gl_api_key_abc123def456...
```

### Obtaining a Token

```bash
curl -X POST "https://api.greenlang.io/api/v1/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=YOUR_ID&client_secret=YOUR_SECRET"
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

## Rate Limiting

Rate limits are enforced per user per endpoint:

| Tier | Limit | Endpoints |
|------|-------|-----------|
| Standard | 100 requests/minute | GET (read operations) |
| Write | 30 requests/minute | POST, PUT, DELETE (mutations) |
| Heavy | 10 requests/minute | Discovery, risk propagation, gap analysis |
| Export | 5 requests/minute | DDS export, PDF generation |

When rate limit is exceeded, the API returns `429 Too Many Requests`:

```json
{
  "detail": "Rate limit exceeded: 100 requests per 60 seconds"
}
```

**Response Headers:**

| Header | Description |
|--------|-------------|
| `Retry-After` | Seconds to wait before retrying |
| `X-RateLimit-Limit` | Maximum requests allowed |
| `X-RateLimit-Remaining` | Remaining requests in window |

---

## Pagination

List endpoints support pagination via query parameters:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `limit` | integer | 50 | 1-1000 | Maximum results per page |
| `offset` | integer | 0 | >= 0 | Number of results to skip |

**Response Metadata:**

```json
{
  "items": [...],
  "meta": {
    "total": 1250,
    "limit": 50,
    "offset": 0,
    "has_more": true
  }
}
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Human-readable error message"
}
```

**Standard Error Codes:**

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| 400 | Bad Request | Invalid request body, validation failure |
| 401 | Unauthorized | Missing or invalid JWT/API key |
| 403 | Forbidden | Insufficient RBAC permissions |
| 404 | Not Found | Graph, node, edge, or gap not found |
| 409 | Conflict | Duplicate submission (onboarding) |
| 410 | Gone | Expired invitation |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Unexpected server failure |

---

## Graph CRUD Endpoints

### POST /graphs -- Create a New Supply Chain Graph

Create a new supply chain graph for a specific EUDR commodity.

**Permission:** `eudr-supply-chain:graphs:create`
**Rate Limit:** Write (30/min)

```bash
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/graphs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "commodity": "cocoa",
    "graph_name": "Ghana Cocoa Supply Chain Q1 2026"
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `commodity` | string | Yes | EUDR commodity: cattle, cocoa, coffee, oil_palm, rubber, soya, wood (or derived) |
| `graph_name` | string | No | Human-readable name (max 500 chars) |

**Response (201 Created):**

```json
{
  "graph_id": "550e8400-e29b-41d4-a716-446655440000",
  "operator_id": "op-eu-001",
  "commodity": "cocoa",
  "graph_name": "Ghana Cocoa Supply Chain Q1 2026",
  "status": "created",
  "created_at": "2026-03-07T10:30:00Z"
}
```

---

### GET /graphs -- List Supply Chain Graphs

List all supply chain graphs accessible to the authenticated user.

**Permission:** `eudr-supply-chain:graphs:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs?commodity=cocoa&limit=20&offset=0" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `commodity` | string | No | Filter by EUDR commodity |
| `limit` | integer | No | Results per page (default: 50) |
| `offset` | integer | No | Pagination offset (default: 0) |

**Response (200 OK):**

```json
{
  "graphs": [
    {
      "graph_id": "550e8400-...",
      "operator_id": "op-eu-001",
      "commodity": "cocoa",
      "graph_name": "Ghana Cocoa Supply Chain Q1 2026",
      "total_nodes": 127,
      "total_edges": 245,
      "max_tier_depth": 5,
      "traceability_score": 87.5,
      "compliance_readiness": 72.3,
      "version": 14,
      "created_at": "2026-03-07T10:30:00Z",
      "updated_at": "2026-03-07T14:22:00Z"
    }
  ],
  "meta": {
    "total": 3,
    "limit": 20,
    "offset": 0,
    "has_more": false
  }
}
```

---

### GET /graphs/{graph_id} -- Get Graph Details

Retrieve full supply chain graph details including all nodes, edges, and gaps.

**Permission:** `eudr-supply-chain:graphs:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-..." \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "operator_id": "op-eu-001",
  "commodity": "cocoa",
  "graph_name": "Ghana Cocoa Supply Chain Q1 2026",
  "nodes": {
    "node-001": {
      "node_id": "node-001",
      "node_type": "producer",
      "operator_id": "gh-coop-alpha",
      "operator_name": "Cooperative Alpha",
      "country_code": "GH",
      "coordinates": [6.6885, -1.6244],
      "commodities": ["cocoa"],
      "tier_depth": 3,
      "risk_score": 42.5,
      "risk_level": "standard",
      "compliance_status": "compliant",
      "certifications": ["RA-2024-GH-001"],
      "plot_ids": ["plot-gh-001", "plot-gh-002"]
    }
  },
  "edges": {
    "edge-001": {
      "edge_id": "edge-001",
      "source_node_id": "node-001",
      "target_node_id": "node-002",
      "commodity": "cocoa",
      "product_description": "Dried cocoa beans",
      "quantity": "5000",
      "unit": "kg",
      "batch_number": "BATCH-GH-2026-001",
      "custody_model": "segregated",
      "transfer_date": "2026-02-15T00:00:00Z"
    }
  },
  "total_nodes": 127,
  "total_edges": 245,
  "max_tier_depth": 5,
  "traceability_score": 87.5,
  "compliance_readiness": 72.3,
  "risk_summary": {"low": 45, "standard": 62, "high": 20},
  "gaps": [],
  "version": 14,
  "created_at": "2026-03-07T10:30:00Z",
  "updated_at": "2026-03-07T14:22:00Z"
}
```

---

### DELETE /graphs/{graph_id} -- Delete a Graph

Permanently delete a supply chain graph and all associated data.

**Permission:** `eudr-supply-chain:graphs:delete`
**Rate Limit:** Write (30/min)

```bash
curl -X DELETE "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-..." \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "status": "deleted",
  "deleted_at": "2026-03-07T15:00:00Z"
}
```

---

### GET /graphs/{graph_id}/export -- Export Graph as DDS Data

Export supply chain graph data formatted for Due Diligence Statement inclusion.

**Permission:** `eudr-supply-chain:graphs:export`
**Rate Limit:** Export (5/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../export" \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "operator_id": "op-eu-001",
  "commodity": "cocoa",
  "total_supply_chain_actors": 127,
  "tier_depth": 5,
  "traceability_score": 87.5,
  "origin_countries": ["GH", "CI", "CM"],
  "origin_plot_count": 342,
  "custody_transfers_count": 245,
  "risk_level": "standard",
  "compliance_readiness": 72.3,
  "provenance_hash": "sha256:a1b2c3d4...",
  "export_timestamp": "2026-03-07T15:00:00Z",
  "supply_chain_summary": {
    "total_producers": 78,
    "total_processors": 12,
    "total_traders": 8
  }
}
```

---

## Multi-Tier Mapping Endpoints

### POST /graphs/{graph_id}/discover -- Trigger Multi-Tier Discovery

Initiate recursive supply chain discovery from Tier 1 through Tier N.

**Permission:** `eudr-supply-chain:mapping:write`
**Rate Limit:** Heavy (10/min)

```bash
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../discover" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "max_depth": 5,
    "include_certifications": true,
    "commodity_filter": "cocoa"
  }'
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `max_depth` | integer | No | 10 | Maximum tier depth (1-50) |
| `include_certifications` | boolean | No | true | Include certification body nodes |
| `commodity_filter` | string | No | null | Restrict to specific commodity |

**Response (202 Accepted):**

```json
{
  "graph_id": "550e8400-...",
  "tiers_discovered": 4,
  "new_nodes_added": 87,
  "new_edges_added": 134,
  "opaque_segments": 3,
  "processing_time_ms": 4523.7,
  "status": "completed"
}
```

---

### GET /graphs/{graph_id}/tiers -- Get Tier Distribution

Return the distribution of nodes by tier depth with statistics.

**Permission:** `eudr-supply-chain:mapping:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../tiers" \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "tier_counts": {
    "0": 1,
    "1": 5,
    "2": 18,
    "3": 42,
    "4": 61
  },
  "max_depth": 4,
  "average_depth": 3.2,
  "median_depth": 3.0
}
```

---

## Traceability Endpoints

### GET /graphs/{graph_id}/trace/forward/{node_id} -- Forward Trace

Trace forward (downstream) from a node to all downstream actors.

**Permission:** `eudr-supply-chain:trace:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../trace/forward/node-001?max_depth=50" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_depth` | integer | 50 | 1-100 | Maximum trace depth |

**Response (200 OK):**

```json
{
  "trace_id": "trace-001",
  "direction": "forward",
  "start_node_id": "node-001",
  "visited_nodes": ["node-001", "node-002", "node-005", "node-008"],
  "visited_edges": ["edge-001", "edge-004", "edge-007"],
  "origin_plot_ids": [],
  "trace_depth": 3,
  "total_quantity": null,
  "is_complete": true,
  "broken_at": [],
  "processing_time_ms": 12.5
}
```

---

### GET /graphs/{graph_id}/trace/backward/{node_id} -- Backward Trace

Trace backward (upstream) from a node to all origin producers and plots.

**Permission:** `eudr-supply-chain:trace:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../trace/backward/node-008?max_depth=50" \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "trace_id": "trace-002",
  "direction": "backward",
  "start_node_id": "node-008",
  "visited_nodes": ["node-008", "node-005", "node-002", "node-001"],
  "visited_edges": ["edge-007", "edge-004", "edge-001"],
  "origin_plot_ids": ["plot-gh-001", "plot-gh-002", "plot-ci-003"],
  "trace_depth": 3,
  "total_quantity": "14200.00",
  "is_complete": true,
  "broken_at": [],
  "processing_time_ms": 18.3
}
```

---

### GET /graphs/{graph_id}/trace/batch/{batch_id} -- Batch Trace

Find all edges and nodes associated with a specific batch/lot number.

**Permission:** `eudr-supply-chain:trace:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../trace/batch/BATCH-GH-2026-001" \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "batch_id": "BATCH-GH-2026-001",
  "graph_id": "550e8400-...",
  "edges": [
    {
      "edge_id": "edge-001",
      "source_node_id": "node-001",
      "target_node_id": "node-002",
      "commodity": "cocoa",
      "quantity": "5000",
      "unit": "kg",
      "transfer_date": "2026-02-15T00:00:00Z"
    }
  ],
  "origin_nodes": ["node-001"],
  "destination_nodes": ["node-002"],
  "total_quantity": "5000",
  "custody_model": "segregated",
  "is_complete": true
}
```

---

## Risk Assessment Endpoints

### POST /graphs/{graph_id}/risk/propagate -- Run Risk Propagation

Execute risk propagation across all nodes in the supply chain graph.

**Permission:** `eudr-supply-chain:risk:write`
**Rate Limit:** Heavy (10/min)

```bash
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../risk/propagate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "risk_weights": {
      "country": 0.30,
      "commodity": 0.20,
      "supplier": 0.25,
      "deforestation": 0.25
    },
    "propagation_source": "quarterly_review"
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `risk_weights` | object | No | Custom weights (must sum to 1.0). If null, uses config defaults |
| `propagation_source` | string | No | Source identifier for audit trail (default: "api_request") |

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "nodes_updated": 127,
  "propagation_results": [
    {
      "node_id": "node-001",
      "previous_risk_score": 35.0,
      "new_risk_score": 42.5,
      "previous_risk_level": "standard",
      "new_risk_level": "standard",
      "propagation_source": "quarterly_review",
      "risk_factors": {
        "country": 0.30,
        "commodity": 0.20,
        "supplier": 0.25,
        "deforestation": 0.25
      },
      "inherited_risk": 0.0,
      "calculated_at": "2026-03-07T15:00:00Z"
    }
  ],
  "processing_time_ms": 256.8,
  "status": "completed"
}
```

---

### GET /graphs/{graph_id}/risk/summary -- Get Risk Summary

Retrieve aggregated risk statistics for the graph.

**Permission:** `eudr-supply-chain:risk:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../risk/summary" \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "total_nodes": 127,
  "risk_distribution": {"low": 45, "standard": 62, "high": 20},
  "average_risk_score": 48.3,
  "max_risk_score": 89.2,
  "high_risk_nodes": ["node-045", "node-067", "node-089"],
  "risk_concentration": [
    {
      "node_id": "node-089",
      "operator_name": "Plantation Delta",
      "risk_score": 89.2,
      "risk_level": "high",
      "country_code": "ID"
    }
  ],
  "propagation_results": []
}
```

---

### GET /graphs/{graph_id}/risk/heatmap -- Get Risk Heatmap

Retrieve geospatial risk data for map visualization.

**Permission:** `eudr-supply-chain:risk:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../risk/heatmap" \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "heatmap_data": [
    {
      "node_id": "node-001",
      "operator_name": "Cooperative Alpha",
      "risk_score": 42.5,
      "risk_level": "standard",
      "country_code": "GH",
      "node_type": "producer",
      "lat": 6.6885,
      "lon": -1.6244
    }
  ],
  "risk_distribution": {"low": 45, "standard": 62, "high": 20}
}
```

---

## Gap Analysis Endpoints

### POST /graphs/{graph_id}/gaps/analyze -- Run Gap Analysis

Execute comprehensive gap analysis detecting 10 gap types mapped to EUDR articles.

**Permission:** `eudr-supply-chain:gaps:write`
**Rate Limit:** Heavy (10/min)

```bash
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../gaps/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "include_resolved": false,
    "severity_filter": "high"
  }'
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `include_resolved` | boolean | No | false | Include previously resolved gaps |
| `severity_filter` | string | No | null | Only gaps of this severity or higher (critical, high, medium, low) |

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "total_gaps": 15,
  "gaps_by_severity": {
    "critical": 3,
    "high": 5,
    "medium": 4,
    "low": 3
  },
  "gaps_by_type": {
    "missing_geolocation": 3,
    "unverified_actor": 5,
    "orphan_node": 3,
    "missing_certification": 4
  },
  "compliance_readiness": 72.3,
  "gaps": [
    {
      "gap_id": "gap-001",
      "gap_type": "missing_geolocation",
      "severity": "critical",
      "affected_node_id": "node-045",
      "affected_edge_id": null,
      "description": "Producer node 'Farm Epsilon' lacks GPS coordinates (EUDR Article 9)",
      "remediation": "Add GPS coordinates via satellite imagery or field survey",
      "eudr_article": "Article 9",
      "is_resolved": false,
      "resolved_at": null,
      "detected_at": "2026-03-07T15:00:00Z"
    }
  ],
  "remediation_priority": ["gap-001", "gap-002", "gap-003"],
  "analysis_timestamp": "2026-03-07T15:00:00Z"
}
```

---

### GET /graphs/{graph_id}/gaps -- List Gaps

List compliance gaps with filtering and pagination.

**Permission:** `eudr-supply-chain:gaps:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../gaps?severity=critical&is_resolved=false&limit=20" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `severity` | string | No | Filter: critical, high, medium, low |
| `gap_type` | string | No | Filter by gap type |
| `is_resolved` | boolean | No | Filter by resolution status |
| `limit` | integer | No | Results per page (default: 50) |
| `offset` | integer | No | Pagination offset (default: 0) |

**Response (200 OK):**

```json
{
  "gaps": [...],
  "meta": {
    "total": 3,
    "limit": 20,
    "offset": 0,
    "has_more": false
  }
}
```

---

### PUT /graphs/{graph_id}/gaps/{gap_id}/resolve -- Resolve a Gap

Mark a compliance gap as resolved with resolution notes and evidence.

**Permission:** `eudr-supply-chain:gaps:write`
**Rate Limit:** Write (30/min)

```bash
curl -X PUT "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../gaps/gap-001/resolve" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "resolution_notes": "GPS coordinates added for all producer plots via satellite imagery verification",
    "evidence_ids": ["ev-001", "ev-002"]
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resolution_notes` | string | Yes | Explanation of how the gap was resolved (1-2000 chars) |
| `evidence_ids` | array | No | IDs of supporting evidence documents |

**Response (200 OK):**

```json
{
  "gap_id": "gap-001",
  "graph_id": "550e8400-...",
  "status": "resolved",
  "resolved_at": "2026-03-07T16:00:00Z",
  "compliance_readiness": 78.5
}
```

---

## Visualization Endpoints

### GET /graphs/{graph_id}/layout -- Get Graph Layout

Generate node positions and edge paths for frontend rendering.

**Permission:** `eudr-supply-chain:visualization:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../layout?algorithm=hierarchical" \
  -H "Authorization: Bearer $TOKEN"
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | string | hierarchical | Layout: force_directed, hierarchical, radial |

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "layout_algorithm": "hierarchical",
  "node_positions": {
    "node-001": [0.0, 0.0],
    "node-002": [200.0, 0.0],
    "node-003": [200.0, 120.0]
  },
  "edge_paths": {
    "edge-001": [[0.0, 0.0], [200.0, 0.0]]
  },
  "node_styles": {
    "node-001": {
      "color": "#4CAF50",
      "border_color": "#FFC107",
      "size": 20,
      "shape": "diamond",
      "label": "Cooperative Alpha",
      "tooltip": "Cooperative Alpha\nType: producer\nCountry: GH\nRisk: standard (42.5)"
    }
  },
  "edge_styles": {
    "edge-001": {
      "color": "#757575",
      "width": 2.5,
      "label": "5000 kg",
      "dashed": false
    }
  },
  "viewport": {
    "min_x": -50.0,
    "min_y": -50.0,
    "max_x": 850.0,
    "max_y": 650.0
  }
}
```

---

### GET /graphs/{graph_id}/sankey -- Get Sankey Diagram Data

Generate Sankey diagram data showing commodity flow volumes.

**Permission:** `eudr-supply-chain:visualization:read`
**Rate Limit:** Standard (100/min)

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/550e8400-.../sankey" \
  -H "Authorization: Bearer $TOKEN"
```

**Response (200 OK):**

```json
{
  "graph_id": "550e8400-...",
  "nodes": [
    {
      "id": 0,
      "node_id": "node-001",
      "name": "Cooperative Alpha",
      "node_type": "producer",
      "tier_depth": 3,
      "country_code": "GH",
      "color": "#4CAF50"
    }
  ],
  "links": [
    {
      "source": 0,
      "target": 1,
      "value": 5000.0,
      "commodity": "cocoa",
      "unit": "kg",
      "custody_model": "segregated"
    }
  ]
}
```

---

## Supplier Onboarding Endpoints

### POST /onboarding/invite -- Create Onboarding Invitation

Invite a supplier to self-register their EUDR supply chain data.

**Permission:** `eudr-supply-chain:onboarding:write`
**Rate Limit:** Write (30/min)

```bash
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/onboarding/invite" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "supplier_name": "Cooperative Alpha GH",
    "supplier_email": "contact@cooperative-alpha.gh",
    "supplier_country": "GH",
    "commodity": "cocoa",
    "graph_id": "550e8400-...",
    "message": "Please complete your EUDR supply chain profile.",
    "expires_in_days": 30
  }'
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `supplier_name` | string | Yes | -- | Legal name (1-500 chars) |
| `supplier_email` | string | Yes | -- | Contact email (5-320 chars) |
| `supplier_country` | string | Yes | -- | ISO 3166-1 alpha-2 code |
| `commodity` | string | Yes | -- | EUDR commodity |
| `graph_id` | string | No | null | Graph to associate supplier with |
| `message` | string | No | null | Custom invitation message |
| `expires_in_days` | integer | No | 30 | Days until expiry (1-365) |

**Response (201 Created):**

```json
{
  "invitation_id": "inv-001",
  "token": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6",
  "supplier_name": "Cooperative Alpha GH",
  "supplier_email": "contact@cooperative-alpha.gh",
  "status": "pending",
  "expires_at": "2026-04-06T10:30:00Z",
  "onboarding_url": "https://api.greenlang.io/api/v1/eudr-scm/onboarding/a1b2c3d4..."
}
```

---

### GET /onboarding/{token} -- Get Invitation Status (Public)

Check the status of an onboarding invitation. **No authentication required.**

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/onboarding/a1b2c3d4..."
```

**Response (200 OK):**

```json
{
  "invitation_id": "inv-001",
  "supplier_name": "Cooperative Alpha GH",
  "supplier_email": "contact@cooperative-alpha.gh",
  "status": "pending",
  "commodity": "cocoa",
  "supplier_country": "GH",
  "graph_id": "550e8400-...",
  "expires_at": "2026-04-06T10:30:00Z",
  "submitted_at": null
}
```

---

### POST /onboarding/{token}/submit -- Submit Onboarding Data (Public)

Submit supply chain data for a pending invitation. **No authentication required.**

```bash
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/onboarding/a1b2c3d4.../submit" \
  -H "Content-Type: application/json" \
  -d '{
    "operator_name": "Cooperative Alpha GH",
    "country_code": "GH",
    "region": "Ashanti",
    "coordinates": [6.6885, -1.6244],
    "commodities": ["cocoa"],
    "certifications": ["RA-2024-GH-001"],
    "plot_ids": ["plot-gh-001", "plot-gh-002"],
    "node_type": "producer",
    "sub_suppliers": []
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `operator_name` | string | Yes | Legal name (1-500 chars) |
| `country_code` | string | Yes | ISO 3166-1 alpha-2 code |
| `region` | string | No | Sub-national region |
| `coordinates` | [float, float] | No | GPS (lat, lon) in WGS84 |
| `commodities` | array | No | EUDR commodities handled |
| `certifications` | array | No | Certification IDs (FSC, RSPO, etc.) |
| `plot_ids` | array | No | Production plot IDs |
| `node_type` | string | No | Supply chain role (default: producer) |
| `sub_suppliers` | array | No | Sub-tier supplier details |

**Response (200 OK):**

```json
{
  "invitation_id": "inv-001",
  "node_id": "node-new-001",
  "status": "submitted",
  "submitted_at": "2026-03-08T09:15:00Z"
}
```

---

## System Endpoints

### GET /health -- Health Check

Check API health status. **No authentication required.**

```bash
curl "https://api.greenlang.io/api/v1/eudr-scm/health"
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "agent_id": "GL-EUDR-SCM-001",
  "agent_name": "EUDR Supply Chain Mapping Master",
  "version": "1.0.0",
  "timestamp": "2026-03-07T15:00:00Z"
}
```

---

## RBAC Permissions Reference

All permissions use the `eudr-supply-chain:` prefix.

| Permission | Description | Used By |
|------------|-------------|---------|
| `eudr-supply-chain:graphs:create` | Create new graphs | POST /graphs |
| `eudr-supply-chain:graphs:read` | Read graph data | GET /graphs, GET /graphs/{id} |
| `eudr-supply-chain:graphs:delete` | Delete graphs | DELETE /graphs/{id} |
| `eudr-supply-chain:graphs:export` | Export graph for DDS | GET /graphs/{id}/export |
| `eudr-supply-chain:mapping:write` | Trigger tier discovery | POST /discover |
| `eudr-supply-chain:mapping:read` | Read tier data | GET /tiers |
| `eudr-supply-chain:trace:read` | Run trace operations | GET /trace/* |
| `eudr-supply-chain:risk:write` | Run risk propagation | POST /risk/propagate |
| `eudr-supply-chain:risk:read` | Read risk data | GET /risk/summary, /heatmap |
| `eudr-supply-chain:gaps:write` | Run gap analysis, resolve gaps | POST /gaps/analyze, PUT /gaps/resolve |
| `eudr-supply-chain:gaps:read` | Read gap data | GET /gaps |
| `eudr-supply-chain:visualization:read` | Access visualization data | GET /layout, /sankey |
| `eudr-supply-chain:onboarding:write` | Create onboarding invites | POST /onboarding/invite |
| `eudr-supply-chain:*` | Wildcard -- all permissions | Admin access |

**Role Mapping:**

| Role | Permissions |
|------|------------|
| `admin` / `platform_admin` | All permissions (bypass check) |
| `eudr_compliance_officer` | Full read/write access |
| `eudr_analyst` | Read + trace + visualization |
| `eudr_auditor` | Read-only access |
| `api-user` | Depends on API key configuration |

---

## Related Documentation

- [README.md](README.md) -- Agent overview and quick start
- [INTEGRATION.md](INTEGRATION.md) -- Integration guide with data agents
- [DEPLOYMENT.md](DEPLOYMENT.md) -- Deployment and operations guide
