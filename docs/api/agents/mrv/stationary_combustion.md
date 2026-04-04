# Stationary Combustion API Reference

**Agent:** AGENT-MRV-001 (GL-MRV-SCOPE1-001)
**Prefix:** `/api/v1/stationary-combustion`
**Source:** `greenlang/agents/mrv/stationary_combustion/api/router.py`
**Status:** Production Ready

## Overview

The Stationary Combustion agent calculates Scope 1 GHG emissions from stationary combustion sources (boilers, furnaces, heaters, generators). It manages fuel types with emission factors, equipment profiles, facility-level aggregation, uncertainty analysis, compliance checking against 7 regulatory frameworks, and immutable audit trails with SHA-256 provenance hashing.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculate` | Calculate emissions (single) | Yes |
| 2 | POST | `/calculate/batch` | Batch calculation | Yes |
| 3 | GET | `/calculations` | List calculations | Yes |
| 4 | GET | `/calculations/{calc_id}` | Get calculation details | Yes |
| 5 | POST | `/fuels` | Register fuel type | Yes |
| 6 | GET | `/fuels` | List fuel types | Yes |
| 7 | GET | `/fuels/{fuel_id}` | Get fuel details | Yes |
| 8 | POST | `/emission-factors` | Register emission factor | Yes |
| 9 | GET | `/emission-factors` | List emission factors | Yes |
| 10 | GET | `/emission-factors/{factor_id}` | Get factor details | Yes |
| 11 | POST | `/equipment` | Register equipment profile | Yes |
| 12 | GET | `/equipment` | List equipment | Yes |
| 13 | GET | `/equipment/{equip_id}` | Get equipment details | Yes |
| 14 | POST | `/facility-aggregate` | Facility-level aggregation | Yes |
| 15 | POST | `/compliance/check` | Check compliance | Yes |
| 16 | GET | `/compliance` | List compliance records | Yes |
| 17 | POST | `/uncertainty` | Run uncertainty analysis | Yes |
| 18 | GET | `/audit/{calc_id}` | Get audit trail | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Calculate Emissions (Single)

Calculate GHG emissions for a single stationary combustion record.

```http
POST /api/v1/stationary-combustion/calculate
```

**Request Body:**

```json
{
  "fuel_type": "natural_gas",
  "quantity": 10000.0,
  "unit": "therms",
  "facility_id": "facility_hq",
  "equipment_id": "boiler_01",
  "method": "emission_factor",
  "gwp_source": "AR6"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fuel_type` | string | Yes | Fuel type identifier |
| `quantity` | float | Yes | Fuel quantity consumed |
| `unit` | string | Yes | Unit of measurement (therms, gallons, kg, tonnes, mmbtu, etc.) |
| `facility_id` | string | Optional | Facility identifier |
| `equipment_id` | string | Optional | Equipment identifier |
| `method` | string | Optional | Calculation method: `emission_factor` (default), `mass_balance`, `cems` |
| `gwp_source` | string | Optional | GWP source: `AR4`, `AR5`, `AR6` (default), `AR6_20YR` |

**Response:**

```json
{
  "calculation_id": "calc_abc123",
  "fuel_type": "natural_gas",
  "quantity": 10000.0,
  "unit": "therms",
  "co2_kg": 5306.0,
  "ch4_kg": 0.10,
  "n2o_kg": 0.01,
  "total_co2e_kg": 5310.5,
  "method": "emission_factor",
  "gwp_source": "AR6",
  "emission_factor_source": "EPA",
  "provenance_hash": "sha256:a1b2c3...",
  "calculated_at": "2026-04-04T10:30:00Z"
}
```

### 2. Batch Calculation

```http
POST /api/v1/stationary-combustion/calculate/batch
```

**Request Body:**

```json
{
  "inputs": [
    {"fuel_type": "natural_gas", "quantity": 10000.0, "unit": "therms"},
    {"fuel_type": "diesel", "quantity": 5000.0, "unit": "gallons"},
    {"fuel_type": "propane", "quantity": 2000.0, "unit": "gallons"}
  ]
}
```

**Response:**

```json
{
  "batch_id": "batch_abc123",
  "total_records": 3,
  "successful": 3,
  "failed": 0,
  "total_co2e_kg": 78234.5,
  "results": [
    {"calculation_id": "calc_001", "fuel_type": "natural_gas", "total_co2e_kg": 5310.5},
    {"calculation_id": "calc_002", "fuel_type": "diesel", "total_co2e_kg": 50924.0},
    {"calculation_id": "calc_003", "fuel_type": "propane", "total_co2e_kg": 22000.0}
  ],
  "provenance_hash": "sha256:..."
}
```

### 5. Register Fuel Type

```http
POST /api/v1/stationary-combustion/fuels
```

**Request Body:**

```json
{
  "fuel_type": "biomass_wood_pellets",
  "display_name": "Wood Pellets (Biomass)",
  "category": "solid",
  "heat_content_mmbtu_per_unit": 16.4,
  "unit": "short_ton",
  "co2_factor": 93.8,
  "ch4_factor": 0.0072,
  "n2o_factor": 0.0036,
  "biogenic": true
}
```

### 14. Facility-Level Aggregation

Aggregate emissions across all equipment and fuel types for a facility.

```http
POST /api/v1/stationary-combustion/facility-aggregate
```

**Request Body:**

```json
{
  "facility_id": "facility_hq",
  "year": 2025,
  "include_equipment_breakdown": true,
  "include_fuel_breakdown": true
}
```

**Response:**

```json
{
  "facility_id": "facility_hq",
  "year": 2025,
  "total_co2e_kg": 125000.5,
  "by_fuel": {
    "natural_gas": 85000.0,
    "diesel": 40000.5
  },
  "by_equipment": {
    "boiler_01": 65000.0,
    "generator_01": 60000.5
  },
  "provenance_hash": "sha256:..."
}
```

### 15. Check Compliance

```http
POST /api/v1/stationary-combustion/compliance/check
```

**Request Body:**

```json
{
  "calculation_id": "calc_abc123",
  "frameworks": ["ghg_protocol", "iso_14064", "csrd_esrs_e1"]
}
```

**Response:**

```json
{
  "calculation_id": "calc_abc123",
  "overall_compliant": true,
  "records": [
    {
      "framework": "ghg_protocol",
      "compliant": true,
      "requirements_met": 12,
      "requirements_total": 12,
      "findings": []
    },
    {
      "framework": "iso_14064",
      "compliant": true,
      "requirements_met": 8,
      "requirements_total": 8,
      "findings": []
    }
  ]
}
```

### 17. Run Uncertainty Analysis

```http
POST /api/v1/stationary-combustion/uncertainty
```

**Request Body:**

```json
{
  "calculation_id": "calc_abc123",
  "iterations": 10000
}
```

**Response:**

```json
{
  "calculation_id": "calc_abc123",
  "method": "monte_carlo",
  "iterations": 10000,
  "mean_co2e_kg": 5310.5,
  "std_dev_kg": 265.5,
  "ci_lower_95": 4790.1,
  "ci_upper_95": 5830.9,
  "relative_uncertainty_pct": 5.0,
  "provenance_hash": "sha256:..."
}
```

### 18. Get Audit Trail

```http
GET /api/v1/stationary-combustion/audit/{calc_id}
```

**Response:**

```json
{
  "calculation_id": "calc_abc123",
  "entries": [
    {
      "event": "input_received",
      "timestamp": "2026-04-04T10:30:00Z",
      "data_hash": "sha256:..."
    },
    {
      "event": "emission_factor_resolved",
      "timestamp": "2026-04-04T10:30:01Z",
      "data_hash": "sha256:..."
    },
    {
      "event": "calculation_completed",
      "timestamp": "2026-04-04T10:30:02Z",
      "data_hash": "sha256:..."
    }
  ],
  "total_entries": 3,
  "chain_hash": "sha256:..."
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- fuel_type and quantity are required |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation or entity not found |
| 503 | Service Unavailable -- Stationary Combustion service not initialized |
