# AGENT-EUDR-020: Deforestation Alert System API

**Agent ID:** `GL-EUDR-DAS-020`
**Prefix:** `/v1/eudr-das`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-020
**Regulation:** EU 2023/1115 (EUDR) -- Deforestation detection and alerting

## Purpose

The Deforestation Alert System agent ingests near-real-time satellite
deforestation alerts (GLAD, RADD, Hansen GFC), correlates them against
operator supply chains, classifies alert severity, manages buffer zones
around production plots, verifies compliance with the December 31, 2020
cutoff date, establishes forest cover baselines, orchestrates investigation
workflows, and assesses impact on product compliance.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/satellite/detect` | Trigger deforestation detection | JWT |
| POST | `/satellite/scan` | Scan area for alerts | JWT |
| GET | `/satellite/sources` | List data sources | JWT |
| GET | `/satellite/imagery` | Get imagery metadata | JWT |
| GET | `/alerts` | List alerts | JWT |
| GET | `/alerts/{alert_id}` | Get alert details | JWT |
| POST | `/alerts` | Create manual alert | JWT |
| POST | `/alerts/batch` | Batch create alerts | JWT |
| GET | `/alerts/summary` | Get alert summary | JWT |
| GET | `/alerts/statistics` | Get alert statistics | JWT |
| POST | `/severity/classify` | Classify alert severity | JWT |
| POST | `/severity/reclassify` | Reclassify severity | JWT |
| GET | `/severity/thresholds` | Get severity thresholds | JWT |
| GET | `/severity/distribution` | Get severity distribution | JWT |
| POST | `/buffer/create` | Create buffer zone | JWT |
| PUT | `/buffer/{zone_id}` | Update buffer zone | JWT |
| POST | `/buffer/check` | Check buffer overlap | JWT |
| GET | `/buffer/violations` | List buffer violations | JWT |
| GET | `/buffer/zones` | List buffer zones | JWT |
| POST | `/cutoff/verify` | Verify cutoff compliance | JWT |
| POST | `/cutoff/batch-verify` | Batch verify cutoff | JWT |
| GET | `/cutoff/evidence` | Get cutoff evidence | JWT |
| GET | `/cutoff/timeline` | Get cutoff timeline | JWT |
| POST | `/baseline/establish` | Establish forest baseline | JWT |
| POST | `/baseline/compare` | Compare against baseline | JWT |
| PUT | `/baseline/{baseline_id}` | Update baseline | JWT |
| GET | `/baseline/coverage` | Get baseline coverage | JWT |
| POST | `/workflow/triage` | Triage alert | JWT |
| POST | `/workflow/assign` | Assign investigation | JWT |
| POST | `/workflow/investigate` | Submit investigation | JWT |
| POST | `/workflow/resolve` | Resolve alert | JWT |
| POST | `/workflow/escalate` | Escalate alert | JWT |
| GET | `/workflow/sla` | Get SLA metrics | JWT |
| POST | `/compliance/assess` | Assess compliance impact | JWT |
| GET | `/compliance/affected-products` | Get affected products | JWT |
| GET | `/compliance/recommendations` | Get recommendations | JWT |
| POST | `/compliance/remediation` | Submit remediation plan | JWT |
| GET | `/health` | Health check | None |

**Total: 38 endpoints + health**

---

## Endpoints

### POST /v1/eudr-das/satellite/detect

Trigger deforestation detection analysis for a specified area of interest,
ingesting data from GLAD, RADD, and Hansen GFC alert systems.

**Request:**

```json
{
  "area_of_interest": {
    "type": "Polygon",
    "coordinates": [[[-1.5, 6.0], [-1.5, 6.2], [-1.3, 6.2], [-1.3, 6.0], [-1.5, 6.0]]]
  },
  "date_from": "2025-12-01",
  "date_to": "2026-01-31",
  "sources": ["glad", "radd", "hansen_gfc"],
  "min_confidence": 0.7
}
```

**Response (200 OK):**

```json
{
  "detection_id": "det_001",
  "alerts_found": 3,
  "alerts": [
    {
      "alert_id": "alt_001",
      "source": "glad",
      "centroid": {"latitude": 6.12, "longitude": -1.45},
      "area_hectares": 2.3,
      "confidence": 0.89,
      "detection_date": "2026-01-10",
      "severity": "high"
    }
  ],
  "total_area_affected_hectares": 5.8,
  "detected_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-das/cutoff/verify

Verify whether a deforestation alert falls before or after the EUDR
December 31, 2020 cutoff date, determining compliance impact.

**Request:**

```json
{
  "alert_id": "alt_001",
  "plot_id": "plot-GH-001",
  "evidence_sources": ["sentinel2", "landsat8", "hansen_gfc"],
  "confidence_threshold": 0.8
}
```

**Response (200 OK):**

```json
{
  "verification_id": "cv_001",
  "alert_id": "alt_001",
  "plot_id": "plot-GH-001",
  "cutoff_date": "2020-12-31",
  "deforestation_date_estimate": "2025-08-15",
  "after_cutoff": true,
  "compliance_impact": "non_compliant",
  "confidence": 0.91,
  "evidence": [
    {"source": "sentinel2", "date": "2020-06-15", "forest_cover_pct": 78.2},
    {"source": "sentinel2", "date": "2025-09-01", "forest_cover_pct": 12.1}
  ],
  "verified_at": "2026-04-04T10:10:00Z"
}
```

---

### POST /v1/eudr-das/compliance/assess

Assess the compliance impact of deforestation alerts on products in the
supply chain, identifying affected batches and recommending remediation.

**Request:**

```json
{
  "alert_ids": ["alt_001", "alt_002"],
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "include_supply_chain": true
}
```

**Response (200 OK):**

```json
{
  "assessment_id": "ca_001",
  "operator_id": "OP-2024-001",
  "alerts_assessed": 2,
  "affected_products": 3,
  "affected_batches": ["batch-012", "batch-015"],
  "compliance_status": "at_risk",
  "risk_level": "high",
  "recommendations": [
    "Suspend sourcing from affected plots pending investigation",
    "Initiate enhanced due diligence for supplier sup-003",
    "Update DDS-2026-001 with deforestation findings"
  ],
  "remediation_deadline": "2026-05-04T00:00:00Z",
  "assessed_at": "2026-04-04T10:15:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_geometry` | GeoJSON area of interest is invalid |
| 404 | `alert_not_found` | Alert ID not found |
| 422 | `invalid_date_range` | Date range is invalid or too broad |
| 503 | `alert_source_unavailable` | GLAD/RADD/GFC API is unreachable |
