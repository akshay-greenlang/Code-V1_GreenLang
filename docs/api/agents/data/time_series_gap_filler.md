# Time Series Gap Filler API Reference

**Agent:** AGENT-DATA-014 (GL-DATA-X-017)
**Prefix:** `/api/v1/gap-filler`
**Source:** `greenlang/agents/data/time_series_gap_filler/api/router.py`
**Status:** Production Ready

## Overview

The Time Series Gap Filler agent provides 20 REST API endpoints for detecting and filling gaps in time series data used in environmental monitoring, energy metering, and compliance reporting. Capabilities include gap detection with pattern classification, automatic frequency analysis, configurable fill strategies (linear, spline, seasonal, Kalman, forward/backward fill), statistical validation of filled values, cross-series correlation analysis for reference-based filling, business calendar management for calendar-aware gap handling, and full pipeline orchestration.

Supported fill strategies: `auto`, `linear`, `spline`, `seasonal`, `forward_fill`, `backward_fill`, `kalman`.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/jobs` | Create gap filling job | Yes |
| 2 | GET | `/jobs` | List gap filling jobs | Yes |
| 3 | GET | `/jobs/{job_id}` | Get job details | Yes |
| 4 | DELETE | `/jobs/{job_id}` | Cancel/delete job | Yes |
| 5 | POST | `/detect` | Detect gaps in time series | Yes |
| 6 | POST | `/detect/batch` | Batch gap detection (multiple series) | Yes |
| 7 | GET | `/detections` | List detection results | Yes |
| 8 | GET | `/detections/{detection_id}` | Get detection result | Yes |
| 9 | POST | `/frequency` | Analyze time series frequency | Yes |
| 10 | GET | `/frequency/{analysis_id}` | Get frequency analysis result | Yes |
| 11 | POST | `/fill` | Fill detected gaps | Yes |
| 12 | GET | `/fills/{fill_id}` | Get fill details | Yes |
| 13 | POST | `/validate` | Validate filled values | Yes |
| 14 | GET | `/validations/{validation_id}` | Get validation result | Yes |
| 15 | POST | `/correlations` | Compute cross-series correlations | Yes |
| 16 | GET | `/correlations` | List correlation results | Yes |
| 17 | POST | `/calendars` | Create calendar definition | Yes |
| 18 | GET | `/calendars` | List calendar definitions | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 5. Detect Gaps

Detect gaps in a time series, identifying missing timestamps and classifying gap types.

```http
POST /api/v1/gap-filler/detect
```

**Request Body:**

```json
{
  "series": [100.5, 102.3, null, null, 105.8, 107.2, null, 110.0],
  "timestamps": [
    "2026-01-01T00:00:00Z",
    "2026-02-01T00:00:00Z",
    "2026-03-01T00:00:00Z",
    "2026-04-01T00:00:00Z",
    "2026-05-01T00:00:00Z",
    "2026-06-01T00:00:00Z",
    "2026-07-01T00:00:00Z",
    "2026-08-01T00:00:00Z"
  ],
  "frequency": "monthly",
  "name": "site_a_energy_consumption"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `series` | array | Yes | Time series values (`null` for gaps) |
| `timestamps` | array | Yes | ISO timestamp strings |
| `frequency` | string | No | Expected frequency: `sub_minute`, `minutely`, `hourly`, `daily`, `weekly`, `monthly`, `quarterly`, `yearly` (auto-detected if omitted) |
| `name` | string | No | Series name for labeling |

**Response (200):**

```json
{
  "detection_id": "det_abc123",
  "series_name": "site_a_energy_consumption",
  "total_points": 8,
  "total_gaps": 3,
  "gap_pct": 37.5,
  "gaps": [
    {"start_idx": 2, "end_idx": 3, "length": 2, "type": "consecutive"},
    {"start_idx": 6, "end_idx": 6, "length": 1, "type": "isolated"}
  ],
  "gap_types": {"consecutive": 1, "isolated": 1},
  "avg_gap_length": 1.5,
  "max_gap_length": 2,
  "processing_time_ms": 12.5,
  "provenance_hash": "sha256:..."
}
```

**Status Codes:** `200` Success | `400` Validation error | `503` Service not configured

---

### 9. Analyze Frequency

Analyze the frequency and regularity of a time series.

```http
POST /api/v1/gap-filler/frequency
```

**Request Body:**

```json
{
  "timestamps": [
    "2026-01-01T00:00:00Z",
    "2026-02-01T00:00:00Z",
    "2026-03-01T00:00:00Z",
    "2026-04-01T00:00:00Z"
  ]
}
```

**Response (200):**

```json
{
  "analysis_id": "freq_abc123",
  "detected_frequency": "monthly",
  "frequency_seconds": 2592000.0,
  "regularity_score": 0.98,
  "confidence": 0.95,
  "num_points": 4,
  "median_interval": 2592000.0,
  "std_interval": 86400.0,
  "is_regular": true,
  "processing_time_ms": 5.2,
  "provenance_hash": "sha256:..."
}
```

---

### 11. Fill Gaps

Fill detected gaps in a time series using a specified strategy.

```http
POST /api/v1/gap-filler/fill
```

**Request Body:**

```json
{
  "series": [100.5, 102.3, null, null, 105.8, 107.2, null, 110.0],
  "timestamps": [
    "2026-01-01T00:00:00Z",
    "2026-02-01T00:00:00Z",
    "2026-03-01T00:00:00Z",
    "2026-04-01T00:00:00Z",
    "2026-05-01T00:00:00Z",
    "2026-06-01T00:00:00Z",
    "2026-07-01T00:00:00Z",
    "2026-08-01T00:00:00Z"
  ],
  "strategy": "spline"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `series` | array | Yes | Time series values (`null` for gaps) |
| `timestamps` | array | Yes | ISO timestamp strings |
| `gaps` | array | No | Pre-detected gap descriptors (auto-detected from `null`/NaN if omitted) |
| `strategy` | string | No | Fill strategy: `auto`, `linear`, `spline`, `seasonal`, `forward_fill`, `backward_fill`, `kalman` |

**Response (200):**

```json
{
  "fill_id": "fill_abc123",
  "series_name": "",
  "strategy": "spline",
  "total_filled": 3,
  "total_gaps": 3,
  "fill_rate": 1.0,
  "filled_values": [
    {"index": 2, "timestamp": "2026-03-01T00:00:00Z", "value": 103.4, "confidence": 0.92},
    {"index": 3, "timestamp": "2026-04-01T00:00:00Z", "value": 104.6, "confidence": 0.90},
    {"index": 6, "timestamp": "2026-07-01T00:00:00Z", "value": 108.6, "confidence": 0.94}
  ],
  "avg_confidence": 0.92,
  "min_confidence": 0.90,
  "distribution_preserved": true,
  "processing_time_ms": 18.3,
  "provenance_hash": "sha256:..."
}
```

---

### 15. Compute Cross-Series Correlations

Compute correlations between a target series and reference series to identify suitable proxies for gap filling.

```http
POST /api/v1/gap-filler/correlations
```

**Request Body:**

```json
{
  "target": [100.5, 102.3, 103.8, 105.2],
  "references": [
    [200.1, 204.5, 207.6, 210.4],
    [50.2, 48.9, 47.1, 46.0]
  ],
  "method": "pearson"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target` | array | Yes | Target series values |
| `references` | array | Yes | List of reference series |
| `method` | string | No | Method: `pearson`, `spearman`, `kendall` (default: `pearson`) |

**Response (200):**

```json
{
  "correlations": [
    {"correlation_id": "cor_001", "coefficient": 0.99, "p_value": 0.001, "is_significant": true, "suitable_for_fill": true},
    {"correlation_id": "cor_002", "coefficient": -0.97, "p_value": 0.003, "is_significant": true, "suitable_for_fill": true}
  ],
  "count": 2,
  "suitable_count": 2
}
```

---

### 17. Create Calendar

Create a business calendar definition for calendar-aware gap filling.

```http
POST /api/v1/gap-filler/calendars
```

**Request Body:**

```json
{
  "name": "EU Business Calendar 2026",
  "calendar_type": "business",
  "timezone": "Europe/Berlin",
  "business_days": [0, 1, 2, 3, 4],
  "holidays": ["2026-01-01", "2026-12-25", "2026-12-26"],
  "fiscal_year_start_month": 1,
  "reporting_periods": []
}
```

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
| 400 | Bad Request -- invalid series data or unsupported strategy |
| 404 | Not Found -- job, detection, fill, validation, or calendar not found |
| 503 | Service Unavailable -- gap filler service not configured |
