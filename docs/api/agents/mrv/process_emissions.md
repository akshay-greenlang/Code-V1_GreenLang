# Process Emissions API Reference

**Agent:** AGENT-MRV-004 (GL-MRV-SCOPE1-004)
**Prefix:** `/api/v1/process-emissions`
**Source:** `greenlang/agents/mrv/process_emissions/api/router.py`
**Status:** Production Ready

## Overview

The Process Emissions agent calculates Scope 1 GHG emissions from industrial chemical and physical processes (cement, iron/steel, aluminum, glass, ammonia, lime, etc.). It supports four calculation methods (emission factor, mass balance, stoichiometric, direct measurement), three calculation tiers per IPCC guidelines, abatement technology tracking, and typed Pydantic request models with validation. Uses the `create_router()` factory pattern.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculate` | Calculate process emissions | Yes |
| 2 | POST | `/calculate/batch` | Batch calculation | Yes |
| 3 | GET | `/calculations` | List calculations (paginated) | Yes |
| 4 | GET | `/calculations/{calc_id}` | Get calculation details | Yes |
| 5 | POST | `/processes` | Register process type | Yes |
| 6 | GET | `/processes` | List process types | Yes |
| 7 | GET | `/processes/{process_id}` | Get process details | Yes |
| 8 | POST | `/materials` | Register raw material | Yes |
| 9 | GET | `/materials` | List raw materials | Yes |
| 10 | GET | `/materials/{material_id}` | Get material details | Yes |
| 11 | POST | `/units` | Register process unit | Yes |
| 12 | GET | `/units` | List process units | Yes |
| 13 | POST | `/factors` | Register emission factor | Yes |
| 14 | GET | `/factors` | List emission factors | Yes |
| 15 | POST | `/abatement` | Register abatement technology | Yes |
| 16 | GET | `/abatement` | List abatement records | Yes |
| 17 | POST | `/uncertainty` | Run uncertainty analysis | Yes |
| 18 | POST | `/compliance/check` | Run compliance check | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Calculate Process Emissions

```http
POST /api/v1/process-emissions/calculate
```

**Request Body:**

```json
{
  "process_type": "cement_clinker",
  "activity_data": 50000.0,
  "activity_unit": "tonne",
  "calculation_method": "EMISSION_FACTOR",
  "calculation_tier": "TIER_2",
  "gwp_source": "AR6",
  "ef_source": "IPCC",
  "production_route": "dry_process"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `process_type` | string | Yes | Industrial process type (cement_clinker, iron_steel, aluminum, glass, ammonia, lime, etc.) |
| `activity_data` | float | Yes | Production quantity (must be > 0) |
| `activity_unit` | string | Optional | Unit (default: `tonne`) |
| `calculation_method` | string | Optional | `EMISSION_FACTOR`, `MASS_BALANCE`, `STOICHIOMETRIC`, `DIRECT_MEASUREMENT` |
| `calculation_tier` | string | Optional | `TIER_1`, `TIER_2`, `TIER_3` |
| `gwp_source` | string | Optional | `AR4`, `AR5`, `AR6`, `AR6_20YR` |
| `ef_source` | string | Optional | `EPA`, `IPCC`, `DEFRA`, `EU_ETS`, `CUSTOM` |
| `production_route` | string | Optional | Production route (e.g., dry_process, wet_process, BF-BOF, EAF) |
| `abatement_type` | string | Optional | Abatement technology type |
| `abatement_efficiency` | float | Optional | Abatement efficiency 0.0-1.0 |
| `materials` | object[] | Optional | Material inputs for mass balance method |

**Response:**

```json
{
  "calculation_id": "calc_proc_001",
  "process_type": "cement_clinker",
  "activity_data": 50000.0,
  "activity_unit": "tonne",
  "method": "EMISSION_FACTOR",
  "tier": "TIER_2",
  "gas_breakdown": {
    "co2_kg": 26150000.0,
    "ch4_kg": 0.0,
    "n2o_kg": 0.0
  },
  "total_co2e_kg": 26150000.0,
  "emission_factor_value": 0.523,
  "emission_factor_source": "IPCC",
  "provenance_hash": "sha256:..."
}
```

### 5. Register Process Type

```http
POST /api/v1/process-emissions/processes
```

**Request Body:**

```json
{
  "process_type": "glass_container",
  "category": "glass",
  "name": "Container Glass Manufacturing",
  "description": "Melting of glass batch materials for container production",
  "primary_gases": ["CO2"],
  "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
  "default_emission_factor": 0.21,
  "production_routes": ["soda_lime", "borosilicate"]
}
```

### 15. Register Abatement Technology

```http
POST /api/v1/process-emissions/abatement
```

**Request Body:**

```json
{
  "unit_id": "kiln_01",
  "abatement_type": "carbon_capture",
  "efficiency": 0.90,
  "target_gas": "CO2"
}
```

### 18. Run Compliance Check

Evaluates the calculation against GHG Protocol, ISO 14064, CSRD/ESRS E1, EPA 40 CFR Part 98, UK SECR, and EU ETS.

```http
POST /api/v1/process-emissions/compliance/check
```

**Request Body:**

```json
{
  "calculation_id": "calc_proc_001",
  "frameworks": ["ghg_protocol", "eu_ets"]
}
```

---

## Pagination

List endpoints use page-based pagination:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number (1-indexed) |
| `page_size` | integer | 20 | Items per page (max: 100) |

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid input |
| 404 | Not Found -- process type or calculation not found |
| 422 | Unprocessable Entity -- validation error |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- Process Emissions service not initialized |
