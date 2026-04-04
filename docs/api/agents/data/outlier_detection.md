# Outlier Detection API Reference

**Agent:** AGENT-DATA-013 (GL-DATA-X-016)
**Prefix:** `/api/v1/outlier`
**Source:** `greenlang/agents/data/outlier_detector/api/router.py`
**Status:** Production Ready

## Overview

The Outlier Detection agent provides 20 REST API endpoints for detecting, classifying, and treating outliers in environmental and compliance datasets. Capabilities include multi-method outlier detection (statistical and ML-based), root cause classification, configurable treatment strategies (cap, winsorize, flag, remove, replace), domain-specific threshold management, human feedback integration, statistical impact analysis, and full pipeline orchestration.

Supported detection methods: `iqr`, `zscore`, `modified_zscore`, `mad`, `grubbs`, `tukey`, `percentile`, `lof`, `isolation_forest`, `mahalanobis`, `dbscan`.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/jobs` | Create detection job | Yes |
| 2 | GET | `/jobs` | List detection jobs | Yes |
| 3 | GET | `/jobs/{job_id}` | Get job details | Yes |
| 4 | DELETE | `/jobs/{job_id}` | Cancel/delete job | Yes |
| 5 | POST | `/detect` | Detect outliers (single column) | Yes |
| 6 | POST | `/detect/batch` | Batch detect (multiple columns) | Yes |
| 7 | GET | `/detections` | List detection results | Yes |
| 8 | GET | `/detections/{detection_id}` | Get detection result | Yes |
| 9 | POST | `/classify` | Classify outliers by root cause | Yes |
| 10 | GET | `/classify/{classification_id}` | Get classification result | Yes |
| 11 | POST | `/treat` | Apply outlier treatment | Yes |
| 12 | GET | `/treat/{treatment_id}` | Get treatment result | Yes |
| 13 | POST | `/treat/{treatment_id}/undo` | Undo treatment | Yes |
| 14 | POST | `/thresholds` | Create domain threshold | Yes |
| 15 | GET | `/thresholds` | List domain thresholds | Yes |
| 16 | POST | `/feedback` | Submit human feedback | Yes |
| 17 | POST | `/impact` | Analyze treatment impact | Yes |
| 18 | POST | `/pipeline` | Run full detection pipeline | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 5. Detect Outliers (Single Column)

Detect outliers in a single column using one or more detection methods.

```http
POST /api/v1/outlier/detect
```

**Request Body:**

```json
{
  "records": [
    {"site_id": "S1", "emissions_co2": 1250.5},
    {"site_id": "S2", "emissions_co2": 1180.0},
    {"site_id": "S3", "emissions_co2": 9999.9},
    {"site_id": "S4", "emissions_co2": 1300.2},
    {"site_id": "S5", "emissions_co2": 1220.8}
  ],
  "column": "emissions_co2",
  "methods": ["iqr", "zscore", "isolation_forest"],
  "options": {"contamination": 0.1}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `records` | array | Yes | List of record dictionaries |
| `column` | string | Yes | Column to analyze |
| `methods` | array | No | Detection methods (auto-selected if omitted) |
| `options` | object | No | Method-specific options |

**Response (200):** Detection result with outlier scores, flags, method agreement, and confidence.

**Status Codes:** `200` Success | `400` Validation error | `503` Service not configured

---

### 6. Batch Detect (Multiple Columns)

Detect outliers across multiple columns simultaneously.

```http
POST /api/v1/outlier/detect/batch
```

**Request Body:**

```json
{
  "records": [
    {"emissions_co2": 1250.5, "energy_kwh": 45000, "water_m3": 500},
    {"emissions_co2": 9999.9, "energy_kwh": 44000, "water_m3": 99999}
  ],
  "columns": ["emissions_co2", "energy_kwh", "water_m3"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `records` | array | Yes | List of record dictionaries |
| `columns` | array | No | Columns to analyze (auto-detect numeric if omitted) |

---

### 9. Classify Outliers

Classify detected outliers by their likely root cause (e.g., data entry error, measurement fault, genuine extreme).

```http
POST /api/v1/outlier/classify
```

**Request Body:**

```json
{
  "detections": [
    {"record_idx": 2, "column": "emissions_co2", "value": 9999.9, "score": 0.98}
  ],
  "records": [
    {"site_id": "S1", "emissions_co2": 1250.5},
    {"site_id": "S2", "emissions_co2": 1180.0},
    {"site_id": "S3", "emissions_co2": 9999.9}
  ]
}
```

---

### 11. Apply Treatment

Apply a treatment strategy to detected outliers.

```http
POST /api/v1/outlier/treat
```

**Request Body:**

```json
{
  "records": [
    {"site_id": "S1", "emissions_co2": 1250.5},
    {"site_id": "S3", "emissions_co2": 9999.9}
  ],
  "detections": [
    {"record_idx": 1, "column": "emissions_co2", "value": 9999.9, "score": 0.98}
  ],
  "strategy": "winsorize",
  "options": {"percentile": 95}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `records` | array | Yes | Original record dictionaries |
| `detections` | array | Yes | Outlier detection score dictionaries |
| `strategy` | string | No | Treatment: `cap`, `winsorize`, `flag`, `remove`, `replace`, `investigate` (default: `flag`) |
| `options` | object | No | Strategy-specific options |

---

### 13. Undo Treatment

Revert a previously applied treatment, restoring original values.

```http
POST /api/v1/outlier/treat/{treatment_id}/undo
```

**Response (200):**

```json
{
  "undone": true,
  "treatment_id": "treat_abc123"
}
```

**Status Codes:** `200` Success | `404` Treatment not found or not reversible

---

### 14. Create Domain Threshold

Define a domain-specific acceptable range for a column.

```http
POST /api/v1/outlier/thresholds
```

**Request Body:**

```json
{
  "column": "emissions_co2",
  "min_val": 0.0,
  "max_val": 5000.0,
  "source": "regulatory",
  "context": "EU ETS maximum plausible emission factor per site"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `column` | string | Yes | Column this threshold applies to |
| `min_val` | float | No | Lower acceptable bound |
| `max_val` | float | No | Upper acceptable bound |
| `source` | string | No | Source: `domain`, `statistical`, `regulatory`, `custom`, `learned` |
| `context` | string | No | Description or justification |

---

### 16. Submit Feedback

Submit human feedback on an outlier detection to improve future accuracy.

```http
POST /api/v1/outlier/feedback
```

**Request Body:**

```json
{
  "detection_id": "det_abc123",
  "feedback_type": "false_positive",
  "notes": "Value is correct -- site underwent capacity expansion in Q1 2026"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `detection_id` | string | Yes | Detection being reviewed |
| `feedback_type` | string | No | `confirmed_outlier`, `false_positive`, `reclassified`, `unknown` |
| `notes` | string | No | Human notes or justification |

---

### 18. Run Full Pipeline

Run the complete outlier detection pipeline: detect, classify, treat, and analyze impact.

```http
POST /api/v1/outlier/pipeline
```

**Request Body:**

```json
{
  "records": [
    {"site": "A", "co2": 1250.5, "energy": 45000},
    {"site": "B", "co2": 9999.9, "energy": 44000}
  ],
  "config": {
    "methods": ["iqr", "isolation_forest"],
    "treatment_strategy": "winsorize",
    "auto_classify": true
  }
}
```

**Response (200):** Pipeline result including detections, classifications, treated records, impact analysis, and provenance hash.

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
| 400 | Bad Request -- validation error or unsupported method |
| 404 | Not Found -- job, detection, classification, treatment, or threshold not found |
| 503 | Service Unavailable -- outlier detector service not configured |
