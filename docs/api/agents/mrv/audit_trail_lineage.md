# Audit Trail & Lineage API Reference

**Agent:** AGENT-MRV-030 (GL-MRV-X-042)
**Prefix:** `/api/v1/audit-trail-lineage`
**Source:** `greenlang/agents/mrv/audit_trail_lineage/api/router.py`
**Status:** Production Ready

## Overview

The Audit Trail & Lineage agent provides immutable, tamper-evident audit trails and end-to-end calculation lineage across all MRV emissions calculations (Scope 1, 2, and 3). Every audit event is cryptographically chained via SHA-256. The lineage graph links inputs, intermediate values, and outputs into a traversable DAG (directed acyclic graph) that supports forward-impact analysis and backward-traceability. It also creates evidence packages for third-party assurance engagements (ISAE 3410), performs compliance traceability assessments, and detects changes with impact analysis. Uses GreenLangBase schema models.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/events` | Record audit event | Yes |
| 2 | POST | `/events/batch` | Batch record events (up to 10,000) | Yes |
| 3 | GET | `/events/{event_id}` | Get event details | Yes |
| 4 | GET | `/events` | List audit events | Yes |
| 5 | DELETE | `/events/{event_id}` | Soft-delete event | Yes |
| 6 | POST | `/chain/verify` | Verify hash chain integrity | Yes |
| 7 | GET | `/chain/{org_id}/{year}` | Get hash chain | Yes |
| 8 | POST | `/lineage/nodes` | Create lineage node | Yes |
| 9 | POST | `/lineage/edges` | Create lineage edge | Yes |
| 10 | GET | `/lineage/graph/{org_id}/{year}` | Get lineage graph | Yes |
| 11 | POST | `/lineage/trace` | Trace lineage (forward/backward) | Yes |
| 12 | GET | `/lineage/visualize/{org_id}/{year}` | Visualize lineage graph | Yes |
| 13 | POST | `/evidence/create` | Create evidence package | Yes |
| 14 | POST | `/evidence/sign` | Sign evidence package | Yes |
| 15 | POST | `/evidence/verify` | Verify evidence package | Yes |
| 16 | POST | `/compliance/trace` | Trace compliance lineage | Yes |
| 17 | GET | `/compliance/coverage/{org_id}/{year}` | Coverage assessment | Yes |
| 18 | POST | `/changes/detect` | Detect changes | Yes |
| 19 | POST | `/changes/impact` | Analyze change impact | Yes |
| 20 | POST | `/pipeline` | Run full pipeline | Yes |
| 21 | POST | `/pipeline/batch` | Run batch pipeline | Yes |
| 22 | GET | `/summary/{org_id}/{year}` | Get audit summary | Yes |
| 23 | GET | `/health` | Health check | No |

---

## Key Concepts

### SHA-256 Hash Chain

Every audit event is appended to a cryptographic hash chain for the (organization, reporting_year) pair. Each event's hash is computed as `SHA-256(previous_hash + event_payload)`, creating an immutable, tamper-evident record.

### Lineage DAG

The lineage graph is a directed acyclic graph (DAG) with typed nodes and edges:

**Node Types:**
- `source` -- Raw input data (ERP extract, CSV upload)
- `activity_data` -- Activity data point (kWh, litres, km)
- `emission_factor` -- Emission factor used in calculation
- `intermediate` -- Intermediate calculation result
- `calculation` -- Final emission calculation
- `aggregation` -- Aggregated total
- `disclosure` -- Reported/disclosed value

**Edge Types:**
- `input_to` -- Data flows as input
- `derived_from` -- Result derived from source
- `aggregated_into` -- Values aggregated into total
- `transformed_by` -- Data transformed by process
- `validated_by` -- Data validated by rule
- `overrides` -- Value overrides previous value

### Evidence Packages

Bundle audit events, lineage graphs, and supporting documents into a verifiable package for third-party assurance (ISAE 3410 limited or reasonable assurance).

---

## Key Endpoints

### 1. Record Audit Event

```http
POST /api/v1/audit-trail-lineage/events
```

**Request Body:**

```json
{
  "event_type": "calculation_completed",
  "agent_id": "GL-MRV-S1-001",
  "scope": "scope_1",
  "organization_id": "org_abc",
  "reporting_year": 2025,
  "calculation_id": "calc_stat_001",
  "payload": {
    "fuel_type": "natural_gas",
    "quantity": 10000.0,
    "unit": "therms",
    "total_co2e_kg": 5310.5,
    "method": "emission_factor"
  },
  "data_quality_score": 2,
  "metadata": {"facility_id": "facility_hq"}
}
```

**Audit Event Types:**
`data_ingestion`, `data_validation`, `emission_factor_lookup`, `calculation_started`, `calculation_completed`, `aggregation`, `compliance_check`, `report_generation`, `data_correction`, `recalculation`, `approval`, `signature`, `export`, `deletion`, `configuration_change`

**Response:**

```json
{
  "event_id": "evt_abc123",
  "event_type": "calculation_completed",
  "chain_position": 42,
  "event_hash": "sha256:...",
  "previous_hash": "sha256:...",
  "recorded_at": "2026-04-04T10:30:00Z"
}
```

### 6. Verify Hash Chain Integrity

Verify that the SHA-256 hash chain is intact (no tampering or gaps).

```http
POST /api/v1/audit-trail-lineage/chain/verify
```

**Request Body:**

```json
{
  "organization_id": "org_abc",
  "reporting_year": 2025,
  "start_position": 0,
  "end_position": null
}
```

**Response:**

```json
{
  "organization_id": "org_abc",
  "reporting_year": 2025,
  "chain_length": 42,
  "is_valid": true,
  "verified_from": 0,
  "verified_to": 41,
  "root_hash": "sha256:...",
  "verification_timestamp": "2026-04-04T10:35:00Z"
}
```

### 8. Create Lineage Node

```http
POST /api/v1/audit-trail-lineage/lineage/nodes
```

**Request Body:**

```json
{
  "node_type": "calculation",
  "level": 2,
  "qualified_name": "scope1.stationary.natural_gas.facility_hq",
  "display_name": "Stationary Combustion - Natural Gas - HQ",
  "organization_id": "org_abc",
  "reporting_year": 2025,
  "agent_id": "GL-MRV-S1-001",
  "value": 5310.5,
  "unit": "tCO2e",
  "data_quality_score": 2
}
```

### 9. Create Lineage Edge

```http
POST /api/v1/audit-trail-lineage/lineage/edges
```

**Request Body:**

```json
{
  "source_node_id": "node_ef_001",
  "target_node_id": "node_calc_001",
  "edge_type": "input_to",
  "organization_id": "org_abc",
  "reporting_year": 2025,
  "transformation_description": "EPA emission factor applied to natural gas consumption",
  "confidence": 1.0
}
```

### 11. Trace Lineage

Traverse the lineage DAG forward (impact analysis) or backward (provenance tracing).

```http
POST /api/v1/audit-trail-lineage/lineage/trace
```

**Request Body:**

```json
{
  "start_node_id": "node_calc_001",
  "direction": "backward",
  "max_depth": 10,
  "node_type_filter": null,
  "level_filter": null
}
```

**Response:**

```json
{
  "start_node_id": "node_calc_001",
  "direction": "backward",
  "depth_reached": 3,
  "nodes_visited": 5,
  "trace": [
    {"node_id": "node_calc_001", "type": "calculation", "depth": 0, "value": 5310.5},
    {"node_id": "node_ef_001", "type": "emission_factor", "depth": 1, "value": 53.06},
    {"node_id": "node_ad_001", "type": "activity_data", "depth": 1, "value": 10000.0},
    {"node_id": "node_src_001", "type": "source", "depth": 2, "value": null},
    {"node_id": "node_src_002", "type": "source", "depth": 2, "value": null}
  ]
}
```

### 12. Visualize Lineage Graph

Export the lineage graph in various visualization formats.

```http
GET /api/v1/audit-trail-lineage/lineage/visualize/{org_id}/{year}?format=mermaid
```

**Supported Formats:** `dot` (Graphviz), `json`, `mermaid`, `d3`, `cytoscape`

### 13. Create Evidence Package

```http
POST /api/v1/audit-trail-lineage/evidence/create
```

**Request Body:**

```json
{
  "organization_id": "org_abc",
  "reporting_year": 2025,
  "frameworks": ["ghg_protocol", "iso_14064", "csrd_esrs"],
  "scope_filter": "scope_1",
  "assurance_level": "limited",
  "include_lineage": true,
  "include_chain": true
}
```

**Assurance Levels (per ISAE 3410):**
- `none` -- No assurance engagement
- `limited` -- Limited assurance (negative form conclusion)
- `reasonable` -- Reasonable assurance (positive form conclusion)

### Compliance Frameworks

| Framework | Description |
|-----------|-------------|
| GHG Protocol | Corporate Accounting and Reporting Standard |
| ISO 14064 | Greenhouse gas quantification and reporting |
| CSRD / ESRS | EU Corporate Sustainability Reporting |
| CDP | Carbon Disclosure Project |
| SBTi | Science Based Targets initiative |
| SB 253 | California Climate Corporate Data Accountability Act |
| SEC Climate | US SEC Climate Disclosure Rules |
| EU Taxonomy | EU Taxonomy for Sustainable Activities |
| ISAE 3410 | Assurance Engagements on Greenhouse Gas Statements |

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid event type, missing required fields |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- event, node, chain, or evidence package not found |
| 409 | Conflict -- hash chain integrity violation |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- AuditTrailLineageService not initialized |
