# Dual Reporting Reconciliation Agent API Reference (AGENT-MRV-013)

## Overview

The Dual Reporting Reconciliation Agent (GL-MRV-X-024) reconciles location-based and market-based Scope 2 emission calculations to ensure consistency across dual reporting as required by the GHG Protocol Scope 2 Guidance (2015). Identifies discrepancies, generates waterfall analyses, assesses data quality, and checks compliance across 7 regulatory frameworks.

**API Prefix:** `/api/v1/dual-reporting`
**Agent ID:** GL-MRV-X-024
**Status:** Production Ready

**Regulatory Frameworks (7):**
GHG Protocol, CSRD/ESRS, CDP, SBTi, GRI, ISO 14064, RE100

**Key Features:**
- Reconciles location-based vs market-based Scope 2 totals
- Identifies and classifies discrepancies by root cause
- Generates waterfall decomposition of differences
- Assesses data quality across both methods
- Produces dual-column reporting tables
- Tracks trends over time periods
- Checks compliance across 7 frameworks simultaneously
- Uses Python Decimal arithmetic for zero-hallucination deterministic results

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/reconciliations` | Create reconciliation | 201, 400, 500 |
| 2 | POST | `/reconciliations/batch` | Batch reconciliations | 201, 400, 500 |
| 3 | GET | `/reconciliations` | List reconciliations | 200, 500 |
| 4 | GET | `/reconciliations/{id}` | Get reconciliation | 200, 404, 500 |
| 5 | DELETE | `/reconciliations/{id}` | Delete reconciliation | 200, 404, 500 |
| 6 | GET | `/reconciliations/{id}/discrepancies` | Get discrepancies | 200, 404, 500 |
| 7 | GET | `/reconciliations/{id}/waterfall` | Get waterfall decomposition | 200, 404, 500 |
| 8 | GET | `/reconciliations/{id}/quality` | Get data quality assessment | 200, 404, 500 |
| 9 | GET | `/reconciliations/{id}/tables` | Get dual-column tables | 200, 404, 500 |
| 10 | GET | `/reconciliations/{id}/trends` | Get trend analysis | 200, 404, 500 |
| 11 | GET | `/reconciliations/{id}/compliance` | Get compliance results | 200, 404, 500 |
| 12 | GET | `/compliance/{id}` | Get compliance detail | 200, 404, 500 |
| 13 | GET | `/aggregations` | Aggregated reconciliation stats | 200, 500 |
| 14 | POST | `/export` | Export reconciliation report | 200, 400, 500 |
| 15 | GET | `/health` | Service health check | 200 |
| 16 | GET | `/stats` | Service statistics | 200, 500 |

---

## Endpoints

### 1. POST /reconciliations

Create a new dual-reporting reconciliation comparing location-based and market-based Scope 2 results.

**Request Body:**

```json
{
  "location_based_results": {
    "total_co2e_tonnes": 1250.5,
    "by_facility": {
      "facility-001": 800.2,
      "facility-002": 450.3
    },
    "grid_factors_used": {
      "facility-001": 0.45,
      "facility-002": 0.38
    }
  },
  "market_based_results": {
    "total_co2e_tonnes": 950.8,
    "by_facility": {
      "facility-001": 600.5,
      "facility-002": 350.3
    },
    "instruments": {
      "facility-001": "REC",
      "facility-002": "PPA"
    }
  },
  "reporting_period": "2025",
  "organization_id": "org-001",
  "frameworks": ["ghg_protocol", "csrd_esrs", "cdp"],
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "reconciliation_id": "dr_abc123",
  "status": "completed",
  "location_total_co2e_tonnes": 1250.5,
  "market_total_co2e_tonnes": 950.8,
  "difference_co2e_tonnes": 299.7,
  "difference_pct": 23.97,
  "discrepancy_count": 2,
  "data_quality_score": 0.85,
  "compliance_status": "pass",
  "provenance_hash": "sha256:y5z6a7b8...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 6. GET /reconciliations/{id}/discrepancies

Retrieve detailed discrepancy analysis between location-based and market-based results, classified by root cause (grid factor mismatch, instrument coverage, boundary difference, temporal mismatch).

---

### 7. GET /reconciliations/{id}/waterfall

Retrieve waterfall decomposition showing how the location-based total transforms to the market-based total through sequential adjustments (RECs, PPAs, residual mix, boundary adjustments).

---

### 8. GET /reconciliations/{id}/quality

Retrieve data quality assessment across both reporting methods, with per-facility and per-instrument quality scores.

---

### 9. GET /reconciliations/{id}/tables

Retrieve dual-column reporting tables formatted for GHG Protocol, CSRD, and CDP disclosure requirements.

---

### 10. GET /reconciliations/{id}/trends

Retrieve trend analysis across multiple reporting periods showing convergence/divergence of location and market-based methods over time.

---

### 14. POST /export

Export a reconciliation report in PDF, Excel, or JSON format.

**Request Body:**

```json
{
  "reconciliation_id": "dr_abc123",
  "format": "pdf",
  "sections": ["summary", "discrepancies", "waterfall", "compliance"]
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid input data, missing required fields |
| 404 | Not Found -- Reconciliation not found |
| 500 | Internal Server Error |
