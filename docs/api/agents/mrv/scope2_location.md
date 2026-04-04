# Scope 2 Location-Based Emissions API Reference

**Agent:** AGENT-MRV-009 (GL-MRV-SCOPE2-001)
**Prefix:** `/api/v1/scope2-location`
**Source:** `greenlang/agents/mrv/scope2_location/api/router.py`
**Status:** Production Ready

## Overview

The Scope 2 Location-Based agent calculates indirect GHG emissions from purchased electricity, steam, heating, and cooling using grid-average emission factors. It supports four major grid factor databases (EPA eGRID with 26 US subregions, IEA with 130+ countries, EU EEA with 27 member states, UK DEFRA), Monte Carlo uncertainty analysis (100-1,000,000 iterations), compliance checking against 7 regulatory frameworks with 80 requirements, and facility-level aggregation. Uses the `create_router()` factory pattern with typed Pydantic request models.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculations` | Execute single calculation | Yes |
| 2 | POST | `/calculations/batch` | Execute batch calculations | Yes |
| 3 | GET | `/calculations` | List calculations with filters | Yes |
| 4 | GET | `/calculations/{id}` | Get calculation by ID | Yes |
| 5 | DELETE | `/calculations/{id}` | Delete calculation | Yes |
| 6 | POST | `/facilities` | Register a facility | Yes |
| 7 | GET | `/facilities` | List facilities | Yes |
| 8 | PUT | `/facilities/{id}` | Update facility | Yes |
| 9 | POST | `/grid-factors` | Register custom grid factor | Yes |
| 10 | GET | `/grid-factors` | List grid factors | Yes |
| 11 | GET | `/grid-factors/{id}` | Get grid factor details | Yes |
| 12 | POST | `/compliance/check` | Run compliance check | Yes |
| 13 | GET | `/compliance/{id}` | Get compliance result | Yes |
| 14 | POST | `/uncertainty` | Run uncertainty analysis | Yes |
| 15 | GET | `/aggregations` | Get facility aggregations | Yes |
| 16 | POST | `/time-series` | Get time-series emissions | Yes |
| 17 | POST | `/export` | Export calculations | Yes |
| 18 | GET | `/health` | Service health check | No |
| 19 | GET | `/stats` | Service statistics | Yes |
| 20 | GET | `/engines` | Engine availability status | Yes |

---

## Key Endpoints

### 1. Execute Single Calculation

```http
POST /api/v1/scope2-location/calculations
```

**Request Body:**

```json
{
  "facility_id": "facility_hq",
  "energy_type": "electricity",
  "consumption_kwh": 500000.0,
  "grid_region": "US-RFCW",
  "grid_factor_source": "eGRID",
  "year": 2025,
  "tenant_id": "tenant_abc"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility_id` | string | Yes | Facility identifier |
| `energy_type` | string | Yes | Energy type: `electricity`, `steam`, `heating`, `cooling` |
| `consumption_kwh` | float | Yes | Energy consumption in kWh |
| `grid_region` | string | Yes | Grid region code (e.g., US-RFCW, DE, GB, FR) |
| `grid_factor_source` | string | Optional | Factor source: `eGRID`, `IEA`, `EU_EEA`, `DEFRA` |
| `year` | integer | Yes | Calculation year (1990-2100) |
| `tenant_id` | string | Yes | Tenant identifier |

**Response:**

```json
{
  "calculation_id": "calc_s2l_001",
  "facility_id": "facility_hq",
  "energy_type": "electricity",
  "consumption_kwh": 500000.0,
  "grid_region": "US-RFCW",
  "grid_factor_source": "eGRID",
  "co2_kg": 220000.0,
  "ch4_kg": 18.5,
  "n2o_kg": 3.2,
  "total_co2e_kg": 221050.0,
  "grid_factor_used": {
    "co2_kg_per_mwh": 440.0,
    "source": "eGRID",
    "year": 2023,
    "region": "US-RFCW"
  },
  "provenance_hash": "sha256:...",
  "calculated_at": "2026-04-04T10:30:00Z"
}
```

### Grid Factor Databases

| Database | Coverage | Regions |
|----------|----------|---------|
| EPA eGRID | United States | 26 subregions (RFCW, SRMW, CAMX, ERCT, etc.) |
| IEA | Global | 130+ countries |
| EU EEA | European Union | 27 member states |
| UK DEFRA | United Kingdom | National averages |

### 12. Run Compliance Check

```http
POST /api/v1/scope2-location/compliance/check
```

**Request Body:**

```json
{
  "calculation_id": "calc_s2l_001",
  "frameworks": ["ghg_protocol", "iso_14064", "csrd_esrs_e1"],
  "tenant_id": "tenant_abc"
}
```

**Response:**

```json
{
  "check_id": "chk_001",
  "calculation_id": "calc_s2l_001",
  "overall_compliant": true,
  "frameworks_checked": 3,
  "requirements_met": 24,
  "requirements_total": 24,
  "findings": [],
  "checked_at": "2026-04-04T10:35:00Z"
}
```

**Supported Frameworks (7):**
- GHG Protocol Scope 2 Guidance
- ISO 14064-1
- CSRD / ESRS E1
- CDP Climate Change
- SBTi
- EPA 40 CFR Part 98
- UK SECR

### 14. Run Uncertainty Analysis

Monte Carlo uncertainty analysis with configurable iterations.

```http
POST /api/v1/scope2-location/uncertainty
```

**Request Body:**

```json
{
  "calculation_id": "calc_s2l_001",
  "method": "monte_carlo",
  "iterations": 10000,
  "confidence_level": 0.95,
  "tenant_id": "tenant_abc"
}
```

**Response:**

```json
{
  "analysis_id": "unc_001",
  "calculation_id": "calc_s2l_001",
  "method": "monte_carlo",
  "iterations": 10000,
  "confidence_level": 0.95,
  "co2e_mean_kg": 221050.0,
  "co2e_std_kg": 11052.5,
  "co2e_lower_bound_kg": 199387.5,
  "co2e_upper_bound_kg": 242712.5,
  "relative_uncertainty_pct": 5.0,
  "provenance_hash": "sha256:..."
}
```

### 17. Export Calculations

```http
POST /api/v1/scope2-location/export
```

**Request Body:**

```json
{
  "tenant_id": "tenant_abc",
  "format": "CSV",
  "from_date": "2025-01-01",
  "to_date": "2025-12-31",
  "include_uncertainty": true
}
```

### 20. Engine Availability

```http
GET /api/v1/scope2-location/engines
```

**Response:**

```json
{
  "engines": {
    "calculation_engine": true,
    "grid_factor_engine": true,
    "facility_engine": true,
    "compliance_engine": true,
    "uncertainty_engine": true,
    "aggregation_engine": true,
    "export_engine": true
  },
  "all_available": true
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid input parameters |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation or facility not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- service not initialized |
