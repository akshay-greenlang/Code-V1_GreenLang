# AGENT-EUDR-003: Satellite Monitoring API

**Agent ID:** `GL-EUDR-SAT-003`
**Prefix:** `/v1/eudr-sat`
**Version:** 1.0.0
**PRD:** GL-EUDR-SAT-003
**Regulation:** EU 2023/1115 (EUDR) -- Satellite-based deforestation monitoring

## Purpose

The Satellite Monitoring agent provides continuous Earth observation
capabilities for EUDR compliance. It searches and downloads satellite imagery
(Sentinel-2, Landsat, Planet), computes NDVI baselines, detects land cover
change, schedules continuous monitoring jobs, generates deforestation alerts,
and packages satellite evidence for due diligence statements.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/imagery/search` | Search satellite imagery catalog | JWT |
| POST | `/imagery/download` | Download imagery scene | JWT |
| GET | `/imagery/{scene_id}` | Get scene metadata | JWT |
| GET | `/imagery/availability` | Check imagery availability | JWT |
| POST | `/analysis/ndvi` | Compute NDVI analysis | JWT |
| POST | `/analysis/baseline` | Establish vegetation baseline | JWT |
| GET | `/analysis/baseline/{baseline_id}` | Get baseline details | JWT |
| POST | `/analysis/change-detect` | Run change detection | JWT |
| POST | `/analysis/fusion` | Multi-source data fusion | JWT |
| GET | `/analysis/history` | Get analysis history | JWT |
| POST | `/monitoring/schedule` | Create monitoring schedule | JWT |
| GET | `/monitoring/schedule/{schedule_id}` | Get schedule details | JWT |
| PUT | `/monitoring/schedule/{schedule_id}` | Update schedule | JWT |
| DELETE | `/monitoring/schedule/{schedule_id}` | Delete schedule | JWT |
| GET | `/monitoring/results` | Get monitoring results | JWT |
| POST | `/monitoring/execute` | Execute monitoring run | JWT |
| GET | `/alerts` | List deforestation alerts | JWT |
| GET | `/alerts/{alert_id}` | Get alert details | JWT |
| PUT | `/alerts/{alert_id}/acknowledge` | Acknowledge alert | JWT |
| GET | `/alerts/summary` | Get alert summary | JWT |
| POST | `/evidence/package` | Create evidence package | JWT |
| GET | `/evidence/package/{package_id}` | Get evidence package | JWT |
| GET | `/evidence/download/{package_id}` | Download evidence | JWT |
| POST | `/batch/submit` | Submit batch analysis | JWT |
| GET | `/batch/{job_id}/results` | Get batch results | JWT |
| GET | `/batch/{job_id}/progress` | Get batch progress | JWT |
| DELETE | `/batch/{job_id}` | Cancel batch job | JWT |
| GET | `/health` | Health check | None |

**Total: 28 endpoints + health**

---

## Endpoints

### POST /v1/eudr-sat/imagery/search

Search the satellite imagery catalog for available scenes covering a given
area of interest and time window.

**Request:**

```json
{
  "bbox": [-1.8, 5.9, -1.3, 6.4],
  "date_from": "2025-12-01",
  "date_to": "2026-01-31",
  "max_cloud_cover": 20,
  "platforms": ["sentinel2", "landsat8"],
  "min_resolution_m": 10
}
```

**Response (200 OK):**

```json
{
  "total_scenes": 14,
  "scenes": [
    {
      "scene_id": "S2A_20260115_T30NUN",
      "platform": "sentinel2",
      "acquisition_date": "2026-01-15T10:30:00Z",
      "cloud_cover_pct": 8.2,
      "resolution_m": 10,
      "bbox": [-1.8, 5.9, -1.3, 6.4],
      "thumbnail_url": "https://storage.greenlang.io/sat/thumb/S2A_20260115.png"
    }
  ]
}
```

---

### POST /v1/eudr-sat/analysis/change-detect

Run land cover change detection between two dates for a specified area,
comparing against the EUDR December 31, 2020 cutoff date.

**Request:**

```json
{
  "area_of_interest": {
    "type": "Polygon",
    "coordinates": [[[-1.5, 6.0], [-1.5, 6.2], [-1.3, 6.2], [-1.3, 6.0], [-1.5, 6.0]]]
  },
  "baseline_date": "2020-12-31",
  "comparison_date": "2026-01-15",
  "change_threshold": 0.15,
  "algorithm": "ndvi_difference"
}
```

**Response (200 OK):**

```json
{
  "analysis_id": "cda_001",
  "status": "completed",
  "change_detected": true,
  "change_area_hectares": 12.5,
  "total_area_hectares": 450.0,
  "change_percentage": 2.78,
  "deforestation_risk": "high",
  "ndvi_baseline": 0.72,
  "ndvi_current": 0.31,
  "ndvi_change": -0.41,
  "confidence": 0.92,
  "completed_at": "2026-04-04T10:20:00Z"
}
```

---

### POST /v1/eudr-sat/evidence/package

Create a tamper-evident evidence package containing satellite imagery,
analysis results, and provenance data for inclusion in a due diligence
statement.

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "analysis_ids": ["cda_001", "ndvi_003"],
  "include_imagery": true,
  "include_provenance": true,
  "format": "zip"
}
```

**Response (202 Accepted):**

```json
{
  "package_id": "evpkg_001",
  "status": "assembling",
  "components": ["imagery", "ndvi_analysis", "change_detection", "provenance"],
  "estimated_size_mb": 45.2,
  "provenance_hash": "sha256:e5f6g7h8...",
  "created_at": "2026-04-04T10:25:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_bbox` | Bounding box coordinates are invalid |
| 404 | `scene_not_found` | Satellite scene does not exist |
| 404 | `analysis_not_found` | Analysis ID not found |
| 422 | `invalid_geojson` | GeoJSON geometry is malformed |
| 503 | `imagery_provider_unavailable` | Upstream satellite data provider is down |
