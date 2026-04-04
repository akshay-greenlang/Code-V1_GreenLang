# Missing Value Imputer API Reference

**Agent:** AGENT-DATA-012 (GL-DATA-X-015)
**Prefix:** `/api/v1/imputer`
**Source:** `greenlang/agents/data/missing_value_imputer/api/router.py`
**Status:** Production Ready

## Overview

The Missing Value Imputer agent provides 20 REST API endpoints for comprehensive missing value handling in environmental and compliance datasets. Capabilities include missingness pattern analysis, strategy-aware imputation (statistical, ML-based, interpolation, and rule-based), batch imputation across multiple columns, statistical validation of imputed values, rule management for domain-specific defaults, reusable imputation templates, and full pipeline orchestration.

Supported imputation strategies: `mean`, `median`, `mode`, `knn`, `regression`, `mice`, `random_forest`, `gradient_boosting`, `linear_interpolation`, `spline_interpolation`, `seasonal_decomposition`, `rule_based`, `lookup_table`, `regulatory_default`.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/jobs` | Create imputation job | Yes |
| 2 | GET | `/jobs` | List imputation jobs | Yes |
| 3 | GET | `/jobs/{job_id}` | Get job details | Yes |
| 4 | DELETE | `/jobs/{job_id}` | Cancel/delete job | Yes |
| 5 | POST | `/analyze` | Analyze missingness patterns | Yes |
| 6 | GET | `/analyze/{analysis_id}` | Get analysis result | Yes |
| 7 | POST | `/impute` | Impute values (single column) | Yes |
| 8 | POST | `/impute/batch` | Batch impute (multiple columns) | Yes |
| 9 | GET | `/results/{result_id}` | Get imputation result | Yes |
| 10 | POST | `/validate` | Validate imputation quality | Yes |
| 11 | GET | `/validate/{validation_id}` | Get validation result | Yes |
| 12 | POST | `/rules` | Create imputation rule | Yes |
| 13 | GET | `/rules` | List imputation rules | Yes |
| 14 | PUT | `/rules/{rule_id}` | Update imputation rule | Yes |
| 15 | DELETE | `/rules/{rule_id}` | Delete imputation rule | Yes |
| 16 | POST | `/templates` | Create imputation template | Yes |
| 17 | GET | `/templates` | List imputation templates | Yes |
| 18 | POST | `/pipeline` | Run full imputation pipeline | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 5. Analyze Missingness

Analyze missingness patterns in a dataset to understand the type, extent, and distribution of missing values.

```http
POST /api/v1/imputer/analyze
```

**Request Body:**

```json
{
  "records": [
    {"site_id": "S1", "emissions_co2": 1250.5, "energy_kwh": null, "water_m3": 500},
    {"site_id": "S2", "emissions_co2": null, "energy_kwh": 45000, "water_m3": null},
    {"site_id": "S3", "emissions_co2": 980.2, "energy_kwh": 38000, "water_m3": 420}
  ],
  "columns": ["emissions_co2", "energy_kwh", "water_m3"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `records` | array | Yes | List of record dictionaries |
| `columns` | array | No | Columns to analyze (all columns if omitted) |

**Response (200):** Missingness analysis with per-column statistics, pattern types (MCAR, MAR, MNAR), and recommended strategies.

---

### 7. Impute Values (Single Column)

Impute missing values in a single column using a specified strategy.

```http
POST /api/v1/imputer/impute
```

**Request Body:**

```json
{
  "records": [
    {"site_id": "S1", "emissions_co2": 1250.5},
    {"site_id": "S2", "emissions_co2": null},
    {"site_id": "S3", "emissions_co2": 980.2}
  ],
  "column": "emissions_co2",
  "strategy": "knn",
  "options": {"n_neighbors": 5}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `records` | array | Yes | List of record dictionaries |
| `column` | string | Yes | Column to impute |
| `strategy` | string | No | Imputation strategy (auto-selected if omitted) |
| `options` | object | No | Strategy-specific options |

**Response (200):** Imputation result with filled records, strategy used, confidence scores, and audit trail.

---

### 8. Batch Impute

Impute missing values across multiple columns simultaneously.

```http
POST /api/v1/imputer/impute/batch
```

**Request Body:**

```json
{
  "records": [
    {"emissions_co2": 1250.5, "energy_kwh": null, "water_m3": 500},
    {"emissions_co2": null, "energy_kwh": 45000, "water_m3": null}
  ],
  "strategies": {
    "emissions_co2": "regression",
    "energy_kwh": "median",
    "water_m3": "knn"
  }
}
```

---

### 10. Validate Imputation

Validate imputation quality by comparing original and imputed distributions.

```http
POST /api/v1/imputer/validate
```

**Request Body:**

```json
{
  "original_records": [
    {"emissions_co2": 1250.5, "energy_kwh": null},
    {"emissions_co2": null, "energy_kwh": 45000}
  ],
  "imputed_records": [
    {"emissions_co2": 1250.5, "energy_kwh": 41500},
    {"emissions_co2": 1115.3, "energy_kwh": 45000}
  ],
  "method": "ks_test"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `original_records` | array | Yes | Records before imputation |
| `imputed_records` | array | Yes | Records after imputation |
| `method` | string | No | Validation method: `ks_test`, `chi_square`, `plausibility_range`, `distribution_preservation`, `cross_validation` |

---

### 12. Create Imputation Rule

Create a domain-specific rule for deterministic imputation (e.g., regulatory defaults).

```http
POST /api/v1/imputer/rules
```

**Request Body:**

```json
{
  "name": "EU default emission factor for natural gas",
  "target_column": "emission_factor_co2",
  "conditions": [
    {"field": "fuel_type", "operator": "eq", "value": "natural_gas"},
    {"field": "region", "operator": "eq", "value": "EU"}
  ],
  "impute_value": 56.1,
  "priority": "high",
  "justification": "IPCC 2006 default factor for natural gas"
}
```

---

### 18. Run Full Pipeline

Run the complete imputation pipeline: analyze, select strategies, impute, and validate.

```http
POST /api/v1/imputer/pipeline
```

**Request Body:**

```json
{
  "records": [
    {"site": "A", "co2": 1250.5, "energy": null, "water": 500},
    {"site": "B", "co2": null, "energy": 45000, "water": null}
  ],
  "config": {
    "validation_method": "plausibility_range",
    "auto_strategy_selection": true
  }
}
```

**Response (200):** Pipeline result including analysis summary, imputed records, validation results, and provenance hash.

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
| 400 | Bad Request -- validation error or unsupported strategy |
| 404 | Not Found -- job, analysis, result, rule, or template not found |
| 503 | Service Unavailable -- imputer service not configured |
