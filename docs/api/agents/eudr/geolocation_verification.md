# AGENT-EUDR-002: Geolocation Verification API

**Agent ID:** `GL-EUDR-GEO-002`
**Prefix:** `/v1/eudr-geo`
**Version:** 1.0.0
**PRD:** GL-EUDR-GEO-002
**Regulation:** EU 2023/1115 (EUDR) -- Geolocation requirements per Article 9(1)(d)

## Purpose

The Geolocation Verification agent validates GPS coordinates and polygon
boundaries against EU requirements. It checks coordinates for plausibility,
verifies that production plots do not overlap protected areas or deforested
zones, scores geolocation accuracy, and generates compliance reports. This
agent is foundational for all EUDR due diligence -- Article 9 mandates that
operators provide geolocation coordinates for every production plot.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/verify/coordinates` | Validate single coordinate | JWT |
| POST | `/verify/coordinates/batch` | Validate coordinates in batch | JWT |
| POST | `/verify/polygon` | Verify polygon geometry | JWT |
| POST | `/verify/polygon/repair` | Repair invalid polygon | JWT |
| POST | `/verify/protected-areas` | Check protected area overlap | JWT |
| GET | `/verify/protected-areas/{check_id}` | Get protected area result | JWT |
| POST | `/verify/deforestation` | Check deforestation overlap | JWT |
| GET | `/verify/deforestation/{check_id}` | Get deforestation result | JWT |
| POST | `/verify/plot` | Verify production plot | JWT |
| GET | `/verify/plot/{plot_id}` | Get plot verification result | JWT |
| GET | `/verify/history` | Get verification history | JWT |
| POST | `/verify/batch` | Submit batch verification | JWT |
| GET | `/verify/batch/{job_id}` | Get batch job status | JWT |
| GET | `/verify/batch/{job_id}/progress` | Get batch progress | JWT |
| DELETE | `/verify/batch/{job_id}` | Cancel batch job | JWT |
| GET | `/scores/{entity_id}` | Get accuracy score | JWT |
| GET | `/scores/{entity_id}/history` | Get score history | JWT |
| GET | `/scores/summary` | Get score summary | JWT |
| PUT | `/scores/weights` | Update scoring weights | JWT |
| POST | `/compliance/report` | Generate compliance report | JWT |
| GET | `/compliance/report/{report_id}` | Get compliance report | JWT |
| GET | `/compliance/summary` | Get compliance summary | JWT |
| GET | `/health` | Health check | None |

**Total: 23 endpoints + health**

---

## Endpoints

### POST /v1/eudr-geo/verify/coordinates

Validate a single GPS coordinate pair for EUDR plausibility (WGS 84,
within land boundaries, not in ocean).

**Request:**

```json
{
  "latitude": 6.1256,
  "longitude": -1.5231,
  "country_code": "GH",
  "coordinate_system": "WGS84",
  "collection_method": "gps_field_survey",
  "accuracy_meters": 5.0
}
```

**Response (200 OK):**

```json
{
  "valid": true,
  "latitude": 6.1256,
  "longitude": -1.5231,
  "country_code": "GH",
  "country_match": true,
  "on_land": true,
  "plausibility_score": 0.95,
  "warnings": [],
  "verified_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-geo/verify/protected-areas

Check whether a coordinate or polygon overlaps any WDPA-listed protected
area, which would indicate a potential EUDR compliance violation.

**Request:**

```json
{
  "latitude": 6.1256,
  "longitude": -1.5231,
  "radius_km": 5.0,
  "include_buffer_zones": true
}
```

**Response (200 OK):**

```json
{
  "check_id": "chk_pa_001",
  "overlap_detected": false,
  "nearest_protected_area": {
    "name": "Kakum National Park",
    "wdpa_id": "900123",
    "distance_km": 12.3,
    "designation": "National Park",
    "iucn_category": "II"
  },
  "buffer_zone_overlap": false,
  "risk_level": "low",
  "checked_at": "2026-04-04T10:05:00Z"
}
```

---

### POST /v1/eudr-geo/compliance/report

Generate a comprehensive geolocation compliance report for an operator,
aggregating all verification results and scoring.

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "report_format": "pdf",
  "include_maps": true,
  "date_range_start": "2025-01-01",
  "date_range_end": "2025-12-31"
}
```

**Response (202 Accepted):**

```json
{
  "report_id": "rpt_geo_001",
  "status": "generating",
  "estimated_completion": "2026-04-04T10:15:00Z",
  "format": "pdf"
}
```

---

### GET /v1/eudr-geo/health

**Response (200 OK):**

```json
{
  "status": "healthy",
  "agent_id": "GL-EUDR-GEO-002",
  "agent_name": "EUDR Geolocation Verification Agent",
  "version": "1.0.0",
  "timestamp": "2026-04-04T12:00:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_coordinates` | Coordinates outside valid range |
| 404 | `check_not_found` | Verification check ID not found |
| 422 | `invalid_polygon` | Polygon geometry is invalid |
| 429 | `rate_limit_exceeded` | Too many requests |
