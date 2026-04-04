# Refrigerants & F-Gas API Reference

**Agent:** AGENT-MRV-002 (GL-MRV-SCOPE1-002)
**Prefix:** `/api/v1/refrigerants-fgas`
**Source:** `greenlang/agents/mrv/refrigerants_fgas/api/router.py`
**Status:** Production Ready

## Overview

The Refrigerants & F-Gas agent calculates Scope 1 GHG emissions from refrigerant leakage and fluorinated gas (F-gas) usage. It manages refrigerant databases (HFC, PFC, SF6 with GWP values across AR4/AR5/AR6), equipment profiles, service events, leak rate calculations, compliance checking (EU F-Gas Regulation, EPA Section 608, Kigali Amendment), and uncertainty analysis.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculate` | Single calculation | Yes |
| 2 | POST | `/calculate/batch` | Batch calculation | Yes |
| 3 | GET | `/calculations` | List calculations | Yes |
| 4 | GET | `/calculations/{calc_id}` | Get calculation details | Yes |
| 5 | POST | `/refrigerants` | Register custom refrigerant | Yes |
| 6 | GET | `/refrigerants` | List refrigerants | Yes |
| 7 | GET | `/refrigerants/{ref_id}` | Get refrigerant properties | Yes |
| 8 | POST | `/equipment` | Register equipment | Yes |
| 9 | GET | `/equipment` | List equipment | Yes |
| 10 | GET | `/equipment/{equip_id}` | Get equipment details | Yes |
| 11 | POST | `/service-events` | Log service event | Yes |
| 12 | GET | `/service-events` | List service events | Yes |
| 13 | POST | `/leak-rates` | Register custom leak rate | Yes |
| 14 | GET | `/leak-rates` | List leak rates | Yes |
| 15 | POST | `/compliance/check` | Check compliance | Yes |
| 16 | GET | `/compliance` | List compliance records | Yes |
| 17 | POST | `/uncertainty` | Run uncertainty analysis | Yes |
| 18 | GET | `/audit/{calc_id}` | Get audit trail | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Calculate Emissions (Single)

Calculate GHG emissions for a single refrigerant record using equipment-based, mass balance, or screening method.

```http
POST /api/v1/refrigerants-fgas/calculate
```

**Request Body:**

```json
{
  "refrigerant_type": "R-410A",
  "charge_kg": 12.5,
  "method": "equipment_based",
  "gwp_source": "AR6",
  "equipment_type": "commercial_refrigeration",
  "equipment_id": "chiller_01",
  "facility_id": "facility_hq"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `refrigerant_type` | string | Yes | Refrigerant identifier (R-410A, R-134a, R-404A, SF6, etc.) |
| `charge_kg` | float | Yes | Refrigerant charge in kilograms |
| `method` | string | Optional | Calculation method: `equipment_based` (default), `mass_balance`, `screening` |
| `gwp_source` | string | Optional | GWP source: `AR4`, `AR5`, `AR6` (default), `AR6_20YR` |
| `equipment_type` | string | Optional | Equipment category for leak rate lookup |
| `equipment_id` | string | Optional | Specific equipment identifier |
| `facility_id` | string | Optional | Facility identifier |
| `custom_leak_rate_pct` | float | Optional | Override default leak rate (0-100) |
| `mass_balance_data` | object | Optional | Mass balance inputs (for mass_balance method) |

**Response:**

```json
{
  "calculation_id": "calc_ref_001",
  "refrigerant_type": "R-410A",
  "charge_kg": 12.5,
  "gwp": 2088,
  "gwp_source": "AR6",
  "leak_rate_pct": 8.5,
  "leaked_kg": 1.0625,
  "co2e_kg": 2218.5,
  "method": "equipment_based",
  "status": "completed",
  "provenance_hash": "sha256:..."
}
```

### 5. Register Custom Refrigerant

```http
POST /api/v1/refrigerants-fgas/refrigerants
```

**Request Body:**

```json
{
  "refrigerant_type": "R-454B",
  "category": "HFO_blend",
  "display_name": "R-454B (Opteon XL41)",
  "formula": "R-32/R-1234yf",
  "gwp_ar4": 466,
  "gwp_ar5": 466,
  "gwp_ar6": 466,
  "gwp_ar6_20yr": 1400,
  "is_blend": true,
  "components": [
    {"refrigerant": "R-32", "fraction": 0.689},
    {"refrigerant": "R-1234yf", "fraction": 0.311}
  ],
  "ozone_depletion_potential": 0.0
}
```

### 11. Log Service Event

Record a maintenance or servicing event for an equipment item.

```http
POST /api/v1/refrigerants-fgas/service-events
```

**Request Body:**

```json
{
  "equipment_id": "chiller_01",
  "event_type": "recharge",
  "refrigerant_type": "R-410A",
  "quantity_kg": 3.5,
  "date": "2026-03-15",
  "technician": "tech_john_doe",
  "notes": "Annual top-up after leak detection"
}
```

### 13. Register Custom Leak Rate

```http
POST /api/v1/refrigerants-fgas/leak-rates
```

**Request Body:**

```json
{
  "equipment_type": "commercial_refrigeration",
  "base_rate_pct": 8.0,
  "age_factor": 1.15,
  "climate_factor": 1.05,
  "ldar_adjustment": 0.85,
  "source": "site_measurement_2025"
}
```

**Response:**

```json
{
  "leak_rate_id": "lr_abc123",
  "equipment_type": "commercial_refrigeration",
  "base_rate_pct": 8.0,
  "age_factor": 1.15,
  "climate_factor": 1.05,
  "ldar_adjustment": 0.85,
  "effective_rate_pct": 8.281,
  "source": "site_measurement_2025",
  "provenance_hash": "sha256:..."
}
```

**Leak Rate Calculation:** `effective_rate = base_rate * age_factor * climate_factor * ldar_adjustment` (capped at 100%).

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- refrigerant_type and charge_kg are required |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation, equipment, or refrigerant not found |
| 503 | Service Unavailable -- Refrigerants & F-Gas service not initialized |
