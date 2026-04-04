# AGENT-EUDR-001: Supply Chain Mapper API

**Agent ID:** `GL-EUDR-SCM-001`
**Prefix:** `/v1/eudr-scm`
**Version:** 1.0.0
**PRD:** GL-EUDR-SCM-001
**Regulation:** EU 2023/1115 (EUDR) -- Supply chain traceability per Articles 4 and 9

## Purpose

The Supply Chain Mapper agent builds and maintains a directed acyclic graph
(DAG) representation of commodity supply chains from production plot to EU
border. It discovers multi-tier supplier relationships, traces commodity flow
forward and backward through the graph, propagates risk scores, identifies
traceability gaps, and generates Sankey visualizations for due diligence
reporting.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/graphs` | Create a supply chain graph | JWT |
| GET | `/graphs` | List supply chain graphs | JWT |
| GET | `/graphs/{graph_id}` | Get graph details | JWT |
| DELETE | `/graphs/{graph_id}` | Delete a graph | JWT |
| GET | `/graphs/{graph_id}/export` | Export graph data | JWT |
| POST | `/mapping/discover` | Discover supplier tiers | JWT |
| GET | `/mapping/tiers` | Get tier structure | JWT |
| GET | `/traceability/forward` | Forward trace from origin | JWT |
| GET | `/traceability/backward` | Backward trace from product | JWT |
| GET | `/traceability/batch` | Batch trace query | JWT |
| POST | `/risk/propagate` | Propagate risk through graph | JWT |
| GET | `/risk/summary` | Get risk summary | JWT |
| GET | `/risk/heatmap` | Get risk heatmap | JWT |
| POST | `/gaps/analyze` | Analyze traceability gaps | JWT |
| GET | `/gaps` | List identified gaps | JWT |
| PUT | `/gaps/{gap_id}/resolve` | Mark gap as resolved | JWT |
| GET | `/visualization/layout` | Get graph layout data | JWT |
| GET | `/visualization/sankey` | Get Sankey diagram data | JWT |
| POST | `/onboarding/invite` | Invite supplier to onboard | JWT |
| GET | `/onboarding/status` | Get onboarding status | JWT |
| POST | `/onboarding/submit` | Submit onboarding data | JWT |
| GET | `/health` | Health check | None |

**Total: 22 endpoints + health**

---

## Endpoints

### POST /v1/eudr-scm/graphs

Create a new supply chain graph for an operator and commodity.

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "name": "Ghana Cocoa Supply Chain Q1 2026",
  "description": "Full supply chain mapping for cocoa sourced from Ghana",
  "nodes": [
    {
      "node_id": "farm-001",
      "node_type": "producer",
      "name": "Adansi Farm Cooperative",
      "country_code": "GH",
      "latitude": 6.1256,
      "longitude": -1.5231
    },
    {
      "node_id": "trader-001",
      "node_type": "trader",
      "name": "Accra Trading Co.",
      "country_code": "GH"
    }
  ],
  "edges": [
    {
      "source": "farm-001",
      "target": "trader-001",
      "commodity": "cocoa_beans",
      "volume_tonnes": 150.0
    }
  ]
}
```

**Response (201 Created):**

```json
{
  "graph_id": "grph_abc123",
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "name": "Ghana Cocoa Supply Chain Q1 2026",
  "node_count": 2,
  "edge_count": 1,
  "completeness_score": 0.45,
  "created_at": "2026-04-04T10:00:00Z",
  "provenance_hash": "sha256:a1b2c3d4..."
}
```

---

### POST /v1/eudr-scm/risk/propagate

Propagate risk scores through the supply chain graph edges using configurable
decay factors.

**Request:**

```json
{
  "graph_id": "grph_abc123",
  "risk_source": "country_risk",
  "decay_factor": 0.85,
  "propagation_direction": "upstream"
}
```

**Response (200 OK):**

```json
{
  "graph_id": "grph_abc123",
  "nodes_updated": 12,
  "max_risk_score": 0.78,
  "high_risk_nodes": ["farm-004", "trader-002"],
  "propagation_direction": "upstream",
  "completed_at": "2026-04-04T10:05:00Z"
}
```

---

### POST /v1/eudr-scm/gaps/analyze

Analyze the supply chain graph for traceability gaps where commodity flow
cannot be fully traced back to a production plot.

**Request:**

```json
{
  "graph_id": "grph_abc123",
  "gap_threshold": 0.8,
  "include_recommendations": true
}
```

**Response (200 OK):**

```json
{
  "graph_id": "grph_abc123",
  "total_gaps": 3,
  "critical_gaps": 1,
  "gaps": [
    {
      "gap_id": "gap-001",
      "gap_type": "missing_origin",
      "affected_node": "trader-002",
      "severity": "critical",
      "description": "No traceable production plot for 30% of volume",
      "recommendation": "Request geolocation data from supplier"
    }
  ],
  "completeness_after": 0.72,
  "analyzed_at": "2026-04-04T10:10:00Z"
}
```

---

### GET /v1/eudr-scm/health

Health check endpoint. No authentication required.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "agent_id": "GL-EUDR-SCM-001",
  "agent_name": "EUDR Supply Chain Mapper Agent",
  "version": "1.0.0",
  "timestamp": "2026-04-04T12:00:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_graph` | Graph structure is invalid (cycles, orphan nodes) |
| 404 | `graph_not_found` | Requested graph does not exist |
| 422 | `validation_error` | Request body fails schema validation |
| 429 | `rate_limit_exceeded` | Too many requests |

**Example Error:**

```json
{
  "detail": "Graph grph_xyz not found",
  "error_code": "graph_not_found"
}
```
