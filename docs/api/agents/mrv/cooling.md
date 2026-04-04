# Cooling Purchase Agent API Reference (AGENT-MRV-012)

## Overview

The Cooling Purchase Agent (GL-MRV-X-023) calculates GHG Protocol Scope 2 emissions from purchased cooling across 5 cooling technology categories: electric chillers, absorption chillers, district cooling networks, free cooling systems, and thermal energy storage (TES). Implements COP-based energy conversion with technology-specific emission factors.

**API Prefix:** `/api/v1/cooling-purchase`
**Agent ID:** GL-MRV-X-023
**Status:** Production Ready

**Cooling Technologies:**
- Electric chillers (centrifugal, screw, reciprocating)
- Absorption chillers (single, double, triple-effect)
- District cooling networks
- Free cooling (air, water, ground-source)
- Thermal Energy Storage (ice, chilled water, eutectic)

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculate/electric` | Calculate electric chiller emissions | 201, 400, 500 |
| 2 | POST | `/calculate/absorption` | Calculate absorption chiller emissions | 201, 400, 500 |
| 3 | POST | `/calculate/district` | Calculate district cooling emissions | 201, 400, 500 |
| 4 | POST | `/calculate/free-cooling` | Calculate free cooling emissions | 201, 400, 500 |
| 5 | POST | `/calculate/tes` | Calculate TES emissions | 201, 400, 500 |
| 6 | POST | `/calculate/batch` | Batch calculation | 201, 400, 500 |
| 7 | GET | `/technologies` | List cooling technologies | 200 |
| 8 | GET | `/technologies/{tech_id}` | Get technology details | 200, 404 |
| 9 | GET | `/factors/district/{region}` | Get district cooling factors | 200, 404 |
| 10 | GET | `/factors/heat-source/{source}` | Get heat source factors | 200, 404 |
| 11 | GET | `/factors/refrigerants` | List refrigerant GWP data | 200 |
| 12 | POST | `/facilities` | Register facility | 201, 400, 409 |
| 13 | GET | `/facilities/{facility_id}` | Get facility | 200, 404 |
| 14 | POST | `/suppliers` | Register cooling supplier | 201, 400, 409 |
| 15 | GET | `/suppliers/{supplier_id}` | Get supplier | 200, 404 |
| 16 | POST | `/uncertainty` | Run uncertainty analysis | 200, 400, 500 |
| 17 | POST | `/compliance/check` | Check compliance | 200, 400, 500 |
| 18 | GET | `/compliance/frameworks` | List frameworks | 200 |
| 19 | POST | `/aggregate` | Aggregate results | 200, 400, 500 |
| 20 | GET | `/health` | Service health check | 200 |

---

## Endpoints

### 1. POST /calculate/electric

Calculate emissions from electric chiller cooling.

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "cooling_output_kwh": 500000,
  "chiller_type": "centrifugal",
  "cop": 5.5,
  "grid_ef_kgco2e_per_kwh": 0.45,
  "refrigerant_type": "R-134a",
  "annual_leakage_rate": 0.02,
  "charge_kg": 150.0,
  "gwp_source": "AR6",
  "tier_level": "tier_2",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "calc_id": "cp_abc123",
  "cooling_technology": "electric_centrifugal",
  "energy_co2e_kg": 40909.09,
  "refrigerant_co2e_kg": 4290.0,
  "total_co2e_kg": 45199.09,
  "cop_used": 5.5,
  "electricity_input_kwh": 90909.09,
  "provenance_hash": "sha256:u1v2w3x4...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

---

### 2. POST /calculate/absorption

Calculate emissions from absorption chiller cooling driven by heat sources.

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "cooling_output_kwh": 300000,
  "absorption_type": "double_effect",
  "heat_source": "natural_gas",
  "cop": 1.2,
  "heat_source_ef_kgco2e_per_kwh": 0.20,
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

---

### 3. POST /calculate/district

Calculate emissions from district cooling network consumption.

---

### 4. POST /calculate/free-cooling

Calculate emissions from free cooling systems (air-side, water-side, ground-source).

---

### 5. POST /calculate/tes

Calculate emissions from thermal energy storage systems (ice storage, chilled water, eutectic).

---

### 6. POST /calculate/batch

Batch multiple cooling calculations across technology types.

---

### 7. GET /technologies

List all supported cooling technologies with COP ranges, capacity ranges, and applicable emission factor sources.

---

### 11. GET /factors/refrigerants

List refrigerant types with GWP values across AR4, AR5, and AR6 sources.

---

### 16. POST /uncertainty

Run Monte Carlo or analytical uncertainty on cooling calculations.

---

### 17. POST /compliance/check

Check cooling purchase calculations against GHG Protocol Scope 2 Guidance, ISO 14064, CSRD/ESRS E1, CDP, and other frameworks.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid technology, absorption type, or parameters |
| 404 | Not Found -- Technology, region, or resource not found |
| 409 | Conflict -- Duplicate facility or supplier |
| 500 | Internal Server Error |
