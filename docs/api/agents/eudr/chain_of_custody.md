# AGENT-EUDR-009: Chain of Custody API

**Agent ID:** `GL-EUDR-COC-009`
**Prefix:** `/v1/eudr-coc`
**Version:** 1.0.0
**PRD:** PRD-AGENT-EUDR-009
**Regulation:** EU 2023/1115 (EUDR) -- Chain of custody tracking per Articles 4 and 9

## Purpose

The Chain of Custody agent tracks the physical flow of EUDR-regulated
commodities through every processing, storage, and transport step from
production plot to EU market entry. It supports segregation, mass balance,
and identity preservation custody models. The agent records custody events,
manages batch splitting/merging/blending, reconciles mass balance ledgers,
verifies chain integrity, and generates traceability reports.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/events` | Create custody event | JWT |
| POST | `/events/batch` | Batch create events | JWT |
| GET | `/events/{event_id}` | Get event details | JWT |
| GET | `/events/chain/{batch_id}` | Get event chain for batch | JWT |
| POST | `/events/{event_id}/amend` | Amend an event | JWT |
| POST | `/batches` | Create a commodity batch | JWT |
| GET | `/batches/{batch_id}` | Get batch details | JWT |
| POST | `/batches/{batch_id}/split` | Split batch | JWT |
| POST | `/batches/{batch_id}/merge` | Merge batches | JWT |
| POST | `/batches/{batch_id}/blend` | Blend batches | JWT |
| GET | `/batches/{batch_id}/genealogy` | Get batch genealogy | JWT |
| POST | `/batches/search` | Search batches | JWT |
| POST | `/models/assign` | Assign custody model | JWT |
| GET | `/models/facility/{facility_id}` | Get facility model | JWT |
| POST | `/models/validate` | Validate model compliance | JWT |
| GET | `/models/compliance` | Get model compliance status | JWT |
| POST | `/balance/input` | Record balance input | JWT |
| POST | `/balance/output` | Record balance output | JWT |
| GET | `/balance/{ledger_id}` | Get balance ledger | JWT |
| POST | `/balance/reconcile` | Reconcile balance | JWT |
| GET | `/balance/history` | Get balance history | JWT |
| POST | `/transform/create` | Record transformation | JWT |
| POST | `/transform/batch` | Batch record transformations | JWT |
| GET | `/transform/{transform_id}` | Get transformation details | JWT |
| POST | `/documents/link` | Link document to event | JWT |
| GET | `/documents/list` | List linked documents | JWT |
| POST | `/documents/validate` | Validate document chain | JWT |
| POST | `/verification/chain` | Verify chain integrity | JWT |
| POST | `/verification/batch` | Batch chain verification | JWT |
| GET | `/verification/{result_id}` | Get verification result | JWT |
| POST | `/reports/traceability` | Generate traceability report | JWT |
| POST | `/reports/mass-balance` | Generate mass balance report | JWT |
| GET | `/reports/{report_id}` | Get report details | JWT |
| GET | `/reports/{report_id}/download` | Download report | JWT |
| POST | `/batch` | Submit batch job | JWT |
| DELETE | `/batch/{batch_id}` | Cancel batch job | JWT |
| GET | `/health` | Health check | None |

**Total: 37 endpoints**

---

## Endpoints

### POST /v1/eudr-coc/events

Record a chain of custody event (receipt, dispatch, processing, storage, etc.)
that tracks the physical movement or transformation of a commodity batch.

**Request:**

```json
{
  "batch_id": "batch-001",
  "event_type": "receipt",
  "facility_id": "fac-GH-001",
  "timestamp": "2026-01-15T08:00:00Z",
  "quantity_kg": 5000.0,
  "commodity": "cocoa_beans",
  "source_entity": "farm-coop-001",
  "destination_entity": "fac-GH-001",
  "documents": ["waybill-001", "phyto-cert-001"],
  "gps_coordinates": {"latitude": 6.1256, "longitude": -1.5231}
}
```

**Response (201 Created):**

```json
{
  "event_id": "evt_001",
  "batch_id": "batch-001",
  "event_type": "receipt",
  "status": "recorded",
  "provenance_hash": "sha256:a1b2c3d4...",
  "created_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-coc/verification/chain

Verify the integrity of a chain of custody by checking that all events form
an unbroken sequence from origin to destination with consistent quantities.

**Request:**

```json
{
  "batch_id": "batch-001",
  "verification_depth": "full",
  "include_genealogy": true,
  "tolerance_pct": 2.0
}
```

**Response (200 OK):**

```json
{
  "result_id": "vrf_001",
  "batch_id": "batch-001",
  "chain_intact": true,
  "total_events": 8,
  "verified_events": 8,
  "quantity_variance_pct": 1.2,
  "within_tolerance": true,
  "custody_model": "segregation",
  "origin_verified": true,
  "issues": [],
  "verified_at": "2026-04-04T10:15:00Z"
}
```

---

### POST /v1/eudr-coc/batch

Submit an asynchronous batch processing job for bulk operations such as
event import, batch verification, balance reconciliation, or report generation.

**Request:**

```json
{
  "job_type": "event_import",
  "priority": 5,
  "parameters": {
    "file_url": "https://storage.greenlang.io/imports/events_q1.csv",
    "format": "csv"
  },
  "callback_url": "https://client.example.com/webhooks/coc"
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "job_coc_001",
  "job_type": "event_import",
  "status": "queued",
  "priority": 5,
  "progress_percent": 0.0,
  "submitted_at": "2026-04-04T10:20:00Z",
  "provenance_hash": "sha256:e5f6g7h8..."
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_event` | Event data fails validation |
| 404 | `batch_not_found` | Commodity batch ID not found |
| 404 | `job_not_found` | Batch job ID not found |
| 409 | `job_not_cancellable` | Job already completed/cancelled |
| 422 | `quantity_mismatch` | Input/output quantities do not reconcile |
