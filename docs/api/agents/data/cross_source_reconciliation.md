# Cross-Source Reconciliation API Reference

**Agent:** AGENT-DATA-015 (GL-DATA-X-018)
**Prefix:** `/api/v1/reconciliation`
**Source:** `greenlang/agents/data/cross_source_reconciliation/api/router.py`
**Status:** Production Ready

## Overview

The Cross-Source Reconciliation agent provides 20 REST API endpoints for reconciling data from multiple sources -- a critical capability for GHG inventories, CSRD reporting, and supply chain data where the same metrics often come from ERP systems, utility bills, meters, and questionnaires. Capabilities include data source registration with credibility scoring, record matching across sources (exact, fuzzy, composite, rule-based), field-level comparison with configurable tolerances, discrepancy detection and severity classification, conflict resolution strategies, golden record assembly, and full pipeline orchestration.

Resolution strategies: `auto`, `priority_wins`, `most_recent`, `weighted_average`, `most_complete`, `consensus`, `manual_override`.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/jobs` | Create reconciliation job | Yes |
| 2 | GET | `/jobs` | List reconciliation jobs | Yes |
| 3 | GET | `/jobs/{job_id}` | Get job details | Yes |
| 4 | DELETE | `/jobs/{job_id}` | Cancel/delete job | Yes |
| 5 | POST | `/sources` | Register data source | Yes |
| 6 | GET | `/sources` | List registered sources | Yes |
| 7 | GET | `/sources/{source_id}` | Get source details | Yes |
| 8 | PUT | `/sources/{source_id}` | Update source metadata | Yes |
| 9 | POST | `/match` | Match records across sources | Yes |
| 10 | GET | `/matches` | List match results | Yes |
| 11 | GET | `/matches/{match_id}` | Get match details | Yes |
| 12 | POST | `/compare` | Compare matched records | Yes |
| 13 | GET | `/discrepancies` | List discrepancies | Yes |
| 14 | GET | `/discrepancies/{discrepancy_id}` | Get discrepancy details | Yes |
| 15 | POST | `/resolve` | Resolve discrepancies | Yes |
| 16 | GET | `/golden-records` | List golden records | Yes |
| 17 | GET | `/golden-records/{record_id}` | Get golden record details | Yes |
| 18 | POST | `/pipeline` | Run full reconciliation pipeline | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 5. Register Data Source

Register a data source with metadata including type, priority, credibility, and refresh cadence.

```http
POST /api/v1/reconciliation/sources
```

**Request Body:**

```json
{
  "name": "SAP Energy Module",
  "source_type": "erp",
  "schema_info": {
    "columns": {
      "entity_id": "string",
      "period": "string",
      "energy_kwh": "float",
      "emissions_tco2e": "float"
    }
  },
  "priority": 2,
  "credibility_score": 0.9,
  "refresh_cadence": "monthly",
  "metadata": {"system_version": "S/4HANA 2025"}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Human-readable source name |
| `source_type` | string | No | `erp`, `utility`, `meter`, `questionnaire`, `registry`, `manual`, `api` (default: `manual`) |
| `schema_info` | object | No | Schema definition with column types |
| `priority` | integer | No | Priority 1 (highest) to 10 (lowest), default 5 |
| `credibility_score` | float | No | Credibility 0.0 to 1.0 (default: 0.8) |
| `refresh_cadence` | string | No | `daily`, `weekly`, `monthly`, `quarterly`, `annual`, `real_time` (default: `monthly`) |
| `metadata` | object | No | Additional metadata |

**Response (201):**

```json
{
  "status": "created",
  "data": {
    "source_id": "src_abc123",
    "name": "SAP Energy Module",
    "source_type": "erp",
    "priority": 2,
    "credibility_score": 0.9,
    "refresh_cadence": "monthly",
    "record_count": 0,
    "status": "active",
    "created_at": "2026-04-04T10:30:00Z"
  }
}
```

---

### 9. Match Records Across Sources

Match records from two data sources using configurable matching strategies.

```http
POST /api/v1/reconciliation/match
```

**Request Body:**

```json
{
  "source_ids": ["src_001", "src_002"],
  "records_a": [
    {"entity_id": "SITE-A", "period": "2025-Q4", "energy_kwh": 45000, "emissions_tco2e": 18.5}
  ],
  "records_b": [
    {"entity_id": "SITE-A", "period": "2025-Q4", "energy_kwh": 44800, "emissions_tco2e": 19.1}
  ],
  "match_keys": ["entity_id", "period"],
  "threshold": 0.85,
  "strategy": "composite"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_ids` | array | No | Source IDs for provenance |
| `records_a` | array | No | First record set |
| `records_b` | array | No | Second record set |
| `match_keys` | array | No | Composite match key fields (default: `["entity_id", "period"]`) |
| `threshold` | float | No | Minimum match confidence (0.0-1.0, default: 0.85) |
| `strategy` | string | No | `exact`, `fuzzy`, `composite`, `rule_based` (default: `composite`) |

---

### 12. Compare Matched Records

Compare matched records field by field, identifying discrepancies with configurable tolerances.

```http
POST /api/v1/reconciliation/compare
```

**Request Body:**

```json
{
  "record_a": {"entity_id": "SITE-A", "period": "2025-Q4", "energy_kwh": 45000, "emissions_tco2e": 18.5},
  "record_b": {"entity_id": "SITE-A", "period": "2025-Q4", "energy_kwh": 44800, "emissions_tco2e": 19.1},
  "fields": ["energy_kwh", "emissions_tco2e"],
  "tolerance_pct": 5.0,
  "tolerance_abs": 0.01
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `match_id` | string | No | Stored match ID (alternative to inline records) |
| `record_a` | object | No | First record for inline comparison |
| `record_b` | object | No | Second record for inline comparison |
| `fields` | array | No | Fields to compare (all shared fields if omitted) |
| `tolerance_pct` | float | No | Relative tolerance as percentage (default: 5.0) |
| `tolerance_abs` | float | No | Absolute tolerance for numeric fields (default: 0.01) |

---

### 15. Resolve Discrepancies

Resolve detected discrepancies using a configurable conflict resolution strategy.

```http
POST /api/v1/reconciliation/resolve
```

**Request Body:**

```json
{
  "discrepancy_ids": ["disc_001", "disc_002"],
  "strategy": "priority_wins",
  "source_priorities": {
    "src_001": 1,
    "src_002": 3
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `discrepancy_ids` | array | No | IDs to resolve |
| `strategy` | string | No | `priority_wins`, `most_recent`, `weighted_average`, `most_complete`, `consensus`, `manual_override` |
| `source_priorities` | object | No | Source-to-priority mapping for `priority_wins` |
| `manual_values` | object | No | Override values for `manual_override` strategy |

---

### 18. Run Full Pipeline

Run the complete reconciliation pipeline: match, compare, detect discrepancies, resolve conflicts, and assemble golden records.

```http
POST /api/v1/reconciliation/pipeline
```

**Request Body:**

```json
{
  "source_ids": ["src_001", "src_002"],
  "records_a": [
    {"entity_id": "SITE-A", "period": "2025-Q4", "energy_kwh": 45000, "emissions_tco2e": 18.5}
  ],
  "records_b": [
    {"entity_id": "SITE-A", "period": "2025-Q4", "energy_kwh": 44800, "emissions_tco2e": 19.1}
  ],
  "match_keys": ["entity_id", "period"],
  "match_threshold": 0.85,
  "tolerance_pct": 5.0,
  "tolerance_abs": 0.01,
  "resolution_strategy": "priority_wins",
  "generate_golden_records": true
}
```

**Response (200):** Pipeline result including match counts, discrepancies found, resolutions applied, golden records assembled, and provenance hash.

---

## Error Responses

All responses are wrapped in the envelope:

```json
{
  "status": "ok",
  "data": { ... }
}
```

Error responses:

```json
{
  "detail": "Descriptive error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error or unsupported strategy |
| 404 | Not Found -- job, source, match, discrepancy, or golden record not found |
| 503 | Service Unavailable -- reconciliation service not configured |
