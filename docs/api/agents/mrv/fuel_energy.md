# Fuel & Energy Activities Agent API Reference (AGENT-MRV-016)

## Overview

The Fuel & Energy Activities Agent (GL-MRV-S3-003) calculates GHG Protocol Scope 3 Category 3 emissions from upstream fuel and energy-related activities not included in Scope 1 or 2. Covers three sub-categories: well-to-tank (WTT) emissions for fuel combustion (3a), WTT for purchased electricity (3b), and transmission and distribution (T&D) losses for purchased electricity (3c).

**API Prefix:** `/api/v1/fuel-energy-activities`
**Agent ID:** GL-MRV-S3-003
**Status:** Production Ready

**Activity Types:**
- `UPSTREAM_FUEL_WTT` (Category 3a) -- WTT emissions from extraction, processing, and transport of fuels
- `UPSTREAM_ELECTRICITY_WTT` (Category 3b) -- WTT emissions from fuel used in electricity generation
- `UPSTREAM_ELECTRICITY_TD_LOSS` (Category 3c) -- T&D line losses for purchased electricity
- `ALL` -- All three sub-categories combined

**Calculation Methods:**
- `AVERAGE_DATA` -- Published national/regional WTT and T&D factors
- `SUPPLIER_SPECIFIC` -- Supplier-disclosed upstream emission data
- `HYBRID` -- Combination of average and supplier-specific methods

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculate` | Calculate fuel & energy emissions | 201, 400, 500 |
| 2 | POST | `/calculate/batch` | Batch calculation | 201, 400, 500 |
| 3 | GET | `/calculations` | List calculations | 200, 500 |
| 4 | GET | `/calculations/{calc_id}` | Get calculation | 200, 404, 500 |
| 5 | DELETE | `/calculations/{calc_id}` | Delete calculation | 200, 404, 500 |
| 6 | POST | `/fuel-consumption` | Create fuel consumption record | 201, 400, 500 |
| 7 | GET | `/fuel-consumption` | List fuel consumption records | 200, 500 |
| 8 | PUT | `/fuel-consumption/{record_id}` | Update fuel consumption record | 200, 400, 404, 500 |
| 9 | POST | `/electricity-consumption` | Create electricity consumption record | 201, 400, 500 |
| 10 | GET | `/electricity-consumption` | List electricity consumption records | 200, 500 |
| 11 | GET | `/emission-factors` | List emission factors | 200, 500 |
| 12 | GET | `/emission-factors/{factor_id}` | Get specific factor | 200, 404, 500 |
| 13 | POST | `/emission-factors/custom` | Register custom emission factor | 201, 400, 500 |
| 14 | GET | `/td-loss-factors` | List T&D loss factors | 200, 500 |
| 15 | GET | `/td-loss-factors/{country}` | Get T&D loss factor by country | 200, 404, 500 |
| 16 | POST | `/compliance/check` | Check compliance | 200, 400, 500 |
| 17 | GET | `/compliance/{compliance_id}` | Get compliance result | 200, 404, 500 |
| 18 | POST | `/uncertainty` | Run uncertainty analysis | 200, 400, 500 |
| 19 | GET | `/aggregations` | Get aggregated results | 200, 500 |
| 20 | GET | `/health` | Service health check | 200 |

---

## Endpoints

### 1. POST /calculate

Calculate Scope 3 Category 3 emissions for fuel and energy-related activities.

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "activity_type": "ALL",
  "method": "AVERAGE_DATA",
  "fuel_consumption": [
    {
      "fuel_type": "natural_gas",
      "quantity_gj": 50000.0,
      "scope1_co2e_kg": 2500000.0
    },
    {
      "fuel_type": "diesel",
      "quantity_litres": 100000.0,
      "scope1_co2e_kg": 260000.0
    }
  ],
  "electricity_consumption": {
    "total_kwh": 2000000.0,
    "country": "US",
    "grid_ef_kgco2e_per_kwh": 0.42,
    "scope2_co2e_kg": 840000.0
  },
  "gwp_source": "AR6",
  "reporting_period": "2025",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "calc_id": "fea_abc123",
  "activity_type": "ALL",
  "total_co2e_kg": 485000.0,
  "wtt_fuel_co2e_kg": 312500.0,
  "wtt_electricity_co2e_kg": 105000.0,
  "td_loss_co2e_kg": 67500.0,
  "by_fuel_type": {
    "natural_gas": { "wtt_co2e_kg": 250000.0 },
    "diesel": { "wtt_co2e_kg": 62500.0 }
  },
  "wtt_fuel_ratio": 0.125,
  "wtt_elec_ratio": 0.125,
  "td_loss_pct": 6.5,
  "provenance_hash": "sha256:g3h4i5j6...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 6. POST /fuel-consumption

Create a fuel consumption record for Category 3a WTT calculations.

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "fuel_type": "natural_gas",
  "quantity_gj": 50000.0,
  "period": "2025-Q1",
  "scope1_co2e_kg": 2500000.0,
  "tenant_id": "tenant-001"
}
```

---

### 9. POST /electricity-consumption

Create an electricity consumption record for Category 3b/3c calculations.

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "total_kwh": 500000.0,
  "country": "US",
  "grid_region": "ERCT",
  "scope2_co2e_kg": 210000.0,
  "period": "2025-Q1",
  "tenant_id": "tenant-001"
}
```

---

### 11. GET /emission-factors

List WTT emission factors for fuels and electricity. Supports filtering by fuel type and country.

---

### 14. GET /td-loss-factors

List T&D loss factors by country. Returns grid loss percentages used for Category 3c calculations.

---

### 15. GET /td-loss-factors/{country}

Get the T&D loss factor for a specific country (ISO 3166-1 alpha-2 code).

---

### 16. POST /compliance/check

Check results against GHG Protocol, ISO 14064, CSRD/ESRS E1, CDP, and SBTi frameworks.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid activity type, method, or fuel parameters |
| 404 | Not Found -- Calculation, record, or factor not found |
| 500 | Internal Server Error |
