# Deforestation Satellite Connector API Reference

**Agent:** AGENT-DATA-007 (GL-DATA-GEO-003)
**Prefix:** `/v1/deforestation`
**Source:** `greenlang/agents/data/deforestation_satellite/api/router.py`
**Status:** Production Ready

## Overview

The Deforestation Satellite Connector agent provides 20 REST API endpoints for satellite-based deforestation monitoring in support of EUDR compliance. Capabilities include satellite imagery acquisition (Sentinel-2, Landsat 8/9, MODIS), vegetation index computation (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI), bi-temporal forest change detection, deforestation alert integration (GLAD, RADD, FIRMS), EUDR baseline assessment for points and polygons, land cover classification, compliance report generation, and continuous monitoring job management.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/imagery/acquire` | Acquire satellite imagery | Yes |
| 2 | POST | `/imagery/time-series` | Acquire imagery time series | Yes |
| 3 | GET | `/imagery/{scene_id}` | Get satellite scene details | Yes |
| 4 | POST | `/indices/calculate` | Calculate vegetation indices | Yes |
| 5 | POST | `/change/detect` | Detect forest change (bi-temporal) | Yes |
| 6 | POST | `/change/trend` | Analyze NDVI trend | Yes |
| 7 | POST | `/alerts/query` | Query deforestation alerts | Yes |
| 8 | POST | `/alerts/filter-cutoff` | Filter alerts by EUDR cutoff date | Yes |
| 9 | POST | `/baseline/check` | Check EUDR baseline (point) | Yes |
| 10 | POST | `/baseline/check-polygon` | Check EUDR baseline (polygon) | Yes |
| 11 | GET | `/baseline/forest-definition/{country_iso3}` | Get forest definition for country | Yes |
| 12 | POST | `/classify` | Classify land cover | Yes |
| 13 | POST | `/classify/batch` | Batch classify land cover | Yes |
| 14 | POST | `/compliance/report` | Generate EUDR compliance report | Yes |
| 15 | GET | `/compliance/report/{report_id}` | Get compliance report | Yes |
| 16 | POST | `/monitoring/start` | Start monitoring job | Yes |
| 17 | GET | `/monitoring/{job_id}` | Get monitoring job status | Yes |
| 18 | POST | `/monitoring/{job_id}/stop` | Stop monitoring job | Yes |
| 19 | GET | `/statistics` | Get service statistics | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Key Endpoints

### 1. Acquire Satellite Imagery

Acquire satellite imagery for a polygon area of interest.

```http
POST /v1/deforestation/imagery/acquire
```

**Request Body:**

```json
{
  "polygon_coordinates": [[107.5, -6.9], [107.6, -6.9], [107.6, -6.8], [107.5, -6.8], [107.5, -6.9]],
  "satellite": "sentinel2",
  "start_date": "2025-06-01",
  "end_date": "2025-12-31",
  "max_cloud_cover": 20
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `polygon_coordinates` | array | Yes | List of `[lon, lat]` coordinate pairs defining the polygon |
| `satellite` | string | No | Satellite source: `sentinel2`, `landsat8`, `landsat9`, `modis` |
| `start_date` | string | Yes | Start date (ISO YYYY-MM-DD) |
| `end_date` | string | Yes | End date (ISO YYYY-MM-DD) |
| `max_cloud_cover` | integer | No | Maximum cloud cover percentage (0-100) |

**Response (200):** Scene metadata dictionary including scene_id, acquisition date, bands, and cloud cover.

**Status Codes:** `200` Success | `400` Validation error | `500` Server error | `503` Engine not available

---

### 4. Calculate Vegetation Indices

Compute vegetation indices for a previously acquired scene.

```http
POST /v1/deforestation/indices/calculate
```

**Request Body:**

```json
{
  "scene_id": "S2A_20250601_T48MYU",
  "indices": ["ndvi", "evi", "ndwi"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scene_id` | string | Yes | Scene ID from a prior acquisition |
| `indices` | array | Yes | Index names: `ndvi`, `evi`, `ndwi`, `nbr`, `savi`, `msavi`, `ndmi` |

**Response (200):** Dictionary keyed by index name with computed values and metadata.

---

### 5. Detect Forest Change

Perform bi-temporal change detection for a polygon by comparing pre-change and post-change imagery.

```http
POST /v1/deforestation/change/detect
```

**Request Body:**

```json
{
  "polygon_coordinates": [[107.5, -6.9], [107.6, -6.9], [107.6, -6.8], [107.5, -6.8], [107.5, -6.9]],
  "pre_start_date": "2019-01-01",
  "pre_end_date": "2019-12-31",
  "post_start_date": "2025-01-01",
  "post_end_date": "2025-12-31",
  "satellite": "sentinel2"
}
```

**Response (200):** Change detection result with change magnitude, direction, affected area, and confidence.

---

### 7. Query Deforestation Alerts

Query deforestation alerts for a polygon from multiple sources (GLAD, RADD, FIRMS).

```http
POST /v1/deforestation/alerts/query
```

**Request Body:**

```json
{
  "polygon_coordinates": [[107.5, -6.9], [107.6, -6.9], [107.6, -6.8], [107.5, -6.8], [107.5, -6.9]],
  "start_date": "2020-01-01",
  "end_date": "2026-01-01",
  "sources": ["glad", "radd"],
  "min_confidence": "high"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `polygon_coordinates` | array | Yes | Polygon coordinate pairs |
| `start_date` | string | Yes | Query start date |
| `end_date` | string | Yes | Query end date |
| `sources` | array | No | Alert sources: `glad`, `radd`, `firms` |
| `min_confidence` | string | No | Minimum confidence: `low`, `nominal`, `high` |

---

### 8. Filter Alerts by EUDR Cutoff

Filter a list of alerts to only include those occurring after the EUDR cutoff date (default: 2020-12-31).

```http
POST /v1/deforestation/alerts/filter-cutoff
```

**Request Body:**

```json
{
  "alerts": [
    {"alert_id": "A001", "date": "2019-06-15", "source": "glad"},
    {"alert_id": "A002", "date": "2021-03-20", "source": "radd"}
  ],
  "cutoff_date": "2020-12-31"
}
```

**Response (200):** Filtered list containing only post-cutoff alerts.

---

### 9. Check EUDR Baseline (Point)

Check EUDR baseline compliance for a single geographic coordinate.

```http
POST /v1/deforestation/baseline/check
```

**Request Body:**

```json
{
  "latitude": -6.85,
  "longitude": 107.55,
  "country_iso3": "IDN",
  "observation_date": "2025-06-01"
}
```

**Response (200):** Baseline assessment including forest status, compliance determination, and confidence score.

---

### 14. Generate EUDR Compliance Report

Generate a full EUDR compliance report for a polygon, combining baseline assessment and alert data.

```http
POST /v1/deforestation/compliance/report
```

**Request Body:**

```json
{
  "alert_aggregation_polygon": [[107.5, -6.9], [107.6, -6.9], [107.6, -6.8], [107.5, -6.8], [107.5, -6.9]],
  "polygon_wkt": "POLYGON((107.5 -6.9, 107.6 -6.9, 107.6 -6.8, 107.5 -6.8, 107.5 -6.9))",
  "country_iso3": "IDN",
  "alert_start_date": "2020-01-01",
  "alert_end_date": "2025-12-31"
}
```

**Response (200):** Compliance report with overall status, baseline assessment, alert summary, and recommendations.

---

### 16. Start Monitoring Job

Start a continuous deforestation monitoring pipeline job for a polygon.

```http
POST /v1/deforestation/monitoring/start
```

**Request Body:**

```json
{
  "polygon_coordinates": [[107.5, -6.9], [107.6, -6.9], [107.6, -6.8], [107.5, -6.8], [107.5, -6.9]],
  "country_iso3": "IDN",
  "frequency": "monthly",
  "satellite": "sentinel2"
}
```

**Response (200):** Monitoring job record with job_id, status, and schedule.

---

## Error Responses

All error responses follow a standard format:

```json
{
  "detail": "Descriptive error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error or invalid parameters |
| 404 | Not Found -- scene, report, or job does not exist |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- engine or service not configured |
