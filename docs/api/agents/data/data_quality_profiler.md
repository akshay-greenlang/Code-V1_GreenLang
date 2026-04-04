# Data Quality Profiler API Reference

**Agent:** AGENT-DATA-010 (GL-DATA-X-013)
**Prefix:** `/api/v1/data-quality`
**Source:** `greenlang/agents/data/data_quality_profiler/api/router.py`
**Status:** Production Ready

## Overview

The Data Quality Profiler assesses datasets across six quality dimensions (completeness, validity, consistency, timeliness, uniqueness, accuracy), detects anomalies, manages quality rules and gates, tracks quality trends over time, and generates scorecard reports.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/v1/profile` | Profile a dataset | Yes |
| 2 | POST | `/v1/profile/batch` | Batch profile datasets | Yes |
| 3 | GET | `/v1/profiles` | List profiles | Yes |
| 4 | GET | `/v1/profiles/{profile_id}` | Get single profile | Yes |
| 5 | POST | `/v1/assess` | Assess dataset quality | Yes |
| 6 | POST | `/v1/assess/batch` | Batch quality assessment | Yes |
| 7 | GET | `/v1/assessments` | List assessments | Yes |
| 8 | GET | `/v1/assessments/{assessment_id}` | Get single assessment | Yes |
| 9 | POST | `/v1/validate` | Validate dataset with rules | Yes |
| 10 | POST | `/v1/detect-anomalies` | Detect anomalies | Yes |
| 11 | GET | `/v1/anomalies` | List anomaly results | Yes |
| 12 | POST | `/v1/check-freshness` | Check dataset freshness | Yes |
| 13 | POST | `/v1/rules` | Create quality rule | Yes |
| 14 | GET | `/v1/rules` | List quality rules | Yes |
| 15 | PUT | `/v1/rules/{rule_id}` | Update quality rule | Yes |
| 16 | DELETE | `/v1/rules/{rule_id}` | Delete quality rule | Yes |
| 17 | POST | `/v1/gates` | Evaluate quality gate | Yes |
| 18 | GET | `/v1/trends` | Get quality trends | Yes |
| 19 | POST | `/v1/reports` | Generate report | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Key Endpoints

### 5. Assess Dataset Quality

```http
POST /api/v1/data-quality/v1/assess
```

**Request Body:**

```json
{
  "data": [
    {"facility": "HQ", "emissions_kg": 1500.0, "year": 2025, "scope": "1"},
    {"facility": "Plant A", "emissions_kg": null, "year": 2025, "scope": "1"}
  ],
  "dataset_name": "facility_emissions",
  "dimensions": ["completeness", "validity", "consistency"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | object[] | Yes | Row dictionaries representing the dataset |
| `dataset_name` | string | Optional | Dataset name (default: `"unnamed"`) |
| `dimensions` | string[] | Optional | Dimensions to assess (default: all six) |

**Response:**

```json
{
  "assessment_id": "assess_abc123",
  "dataset_name": "facility_emissions",
  "overall_score": 87.5,
  "dimensions": {
    "completeness": 75.0,
    "validity": 95.0,
    "consistency": 92.5
  },
  "record_count": 2,
  "issues": [
    {
      "dimension": "completeness",
      "column": "emissions_kg",
      "issue": "null_value",
      "row_index": 1,
      "severity": "warning"
    }
  ]
}
```

### 10. Detect Anomalies

```http
POST /api/v1/data-quality/v1/detect-anomalies
```

**Request Body:**

```json
{
  "data": [
    {"emissions_kg": 1500},
    {"emissions_kg": 1600},
    {"emissions_kg": 15000},
    {"emissions_kg": 1450}
  ],
  "dataset_name": "emissions",
  "columns": ["emissions_kg"],
  "method": "iqr"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | string | Optional | Detection method: `iqr`, `zscore`, `percentile` |

### 13. Create Quality Rule

```http
POST /api/v1/data-quality/v1/rules
```

**Request Body:**

```json
{
  "name": "emissions_non_negative",
  "rule_type": "range",
  "column": "emissions_kg",
  "operator": "gte",
  "threshold": 0,
  "parameters": {"max_value": 1000000},
  "priority": 10
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rule_type` | string | Yes | Type: `not_null`, `unique`, `range`, `regex`, `custom`, `referential` |
| `priority` | integer | Optional | Priority (lower = higher priority, default: 100) |

### 17. Evaluate Quality Gate

```http
POST /api/v1/data-quality/v1/gates
```

**Request Body:**

```json
{
  "conditions": [
    {"dimension": "completeness", "operator": "gte", "threshold": 95.0},
    {"dimension": "validity", "operator": "gte", "threshold": 90.0}
  ],
  "dimension_scores": {
    "completeness": 97.5,
    "validity": 88.0
  }
}
```

**Response:**

```json
{
  "passed": false,
  "results": [
    {"dimension": "completeness", "score": 97.5, "threshold": 95.0, "passed": true},
    {"dimension": "validity", "score": 88.0, "threshold": 90.0, "passed": false}
  ]
}
```
