# Scope 2 Market-Based Emissions API Reference

**Agent:** AGENT-MRV-010 (GL-MRV-SCOPE2-002)
**Prefix:** `/api/v1/scope2-market`
**Source:** `greenlang/agents/mrv/scope2_market/api/router.py`
**Status:** Production Ready

## Overview

The Scope 2 Market-Based agent calculates indirect GHG emissions from purchased electricity using contractual instruments (RECs, GOs, PPAs, green tariffs, I-RECs). It applies the GHG Protocol Scope 2 quality criteria hierarchy, manages facility registrations, tracks instrument retirement (to prevent double counting), performs coverage analysis (percentage of consumption matched by instruments), dual-method reporting reconciliation, and compliance checking. Uses the `create_router()` factory pattern with typed Pydantic request models.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculations` | Execute single market-based calculation | Yes |
| 2 | POST | `/calculations/batch` | Execute batch calculations | Yes |
| 3 | GET | `/calculations` | List calculations with filters | Yes |
| 4 | GET | `/calculations/{id}` | Get calculation by ID | Yes |
| 5 | DELETE | `/calculations/{id}` | Delete calculation | Yes |
| 6 | POST | `/facilities` | Register a facility | Yes |
| 7 | GET | `/facilities` | List facilities | Yes |
| 8 | PUT | `/facilities/{id}` | Update a facility | Yes |
| 9 | POST | `/instruments` | Register contractual instrument | Yes |
| 10 | GET | `/instruments` | List instruments | Yes |
| 11 | POST | `/instruments/{id}/retire` | Retire instrument | Yes |
| 12 | POST | `/compliance/check` | Run compliance check | Yes |
| 13 | GET | `/compliance/{id}` | Get compliance result | Yes |
| 14 | POST | `/uncertainty` | Run uncertainty analysis | Yes |
| 15 | POST | `/dual-report` | Generate dual-method report | Yes |
| 16 | GET | `/aggregations` | Get aggregated emissions | Yes |
| 17 | GET | `/coverage/{facility_id}` | Get coverage analysis | Yes |
| 18 | GET | `/health` | Service health check | No |
| 19 | GET | `/stats` | Service statistics | Yes |
| 20 | GET | `/engines` | Engine availability status | Yes |

---

## Key Endpoints

### 1. Execute Single Calculation

```http
POST /api/v1/scope2-market/calculations
```

**Request Body:**

```json
{
  "facility_id": "facility_hq",
  "consumption_kwh": 500000.0,
  "year": 2025,
  "instrument_ids": ["inst_rec_001", "inst_ppa_002"],
  "residual_mix_region": "US-RFCW",
  "tenant_id": "tenant_abc"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `facility_id` | string | Yes | Facility identifier |
| `consumption_kwh` | float | Yes | Total electricity consumption in kWh |
| `year` | integer | Yes | Calculation year |
| `instrument_ids` | string[] | Optional | Contractual instrument IDs to apply |
| `residual_mix_region` | string | Optional | Residual mix region for uncovered portion |
| `tenant_id` | string | Yes | Tenant identifier |

**Response:**

```json
{
  "calculation_id": "calc_s2m_001",
  "facility_id": "facility_hq",
  "consumption_kwh": 500000.0,
  "covered_kwh": 400000.0,
  "uncovered_kwh": 100000.0,
  "coverage_pct": 80.0,
  "instrument_emissions_kg": 0.0,
  "residual_mix_emissions_kg": 44000.0,
  "total_co2e_kg": 44000.0,
  "instruments_applied": [
    {"instrument_id": "inst_rec_001", "type": "REC", "kwh_covered": 250000.0},
    {"instrument_id": "inst_ppa_002", "type": "PPA", "kwh_covered": 150000.0}
  ],
  "provenance_hash": "sha256:...",
  "calculated_at": "2026-04-04T10:30:00Z"
}
```

### 9. Register Contractual Instrument

```http
POST /api/v1/scope2-market/instruments
```

**Request Body:**

```json
{
  "instrument_type": "REC",
  "issuing_body": "PJM-EIS",
  "certificate_id": "REC-2025-00001",
  "generation_source": "wind",
  "generation_region": "US-RFCW",
  "vintage_year": 2025,
  "quantity_mwh": 500.0,
  "emission_factor_kg_per_mwh": 0.0,
  "tracking_system": "PJM-GATS",
  "tenant_id": "tenant_abc"
}
```

**Supported Instrument Types:**

| Type | Description |
|------|-------------|
| `REC` | Renewable Energy Certificate (US) |
| `GO` | Guarantee of Origin (EU) |
| `I-REC` | International REC |
| `PPA` | Power Purchase Agreement |
| `GREEN_TARIFF` | Green electricity tariff |
| `DIRECT_CONTRACT` | Direct line / wire contract |

### 11. Retire Instrument

Retire an instrument to allocate its clean energy to a facility (prevents double counting).

```http
POST /api/v1/scope2-market/instruments/{instrument_id}/retire
```

**Request Body:**

```json
{
  "facility_id": "facility_hq",
  "retirement_date": "2026-03-31",
  "quantity_mwh": 250.0,
  "tenant_id": "tenant_abc"
}
```

### 15. Generate Dual-Method Report

Generate a side-by-side comparison of location-based and market-based Scope 2 emissions.

```http
POST /api/v1/scope2-market/dual-report
```

**Request Body:**

```json
{
  "facility_id": "facility_hq",
  "year": 2025,
  "tenant_id": "tenant_abc"
}
```

### 17. Get Coverage Analysis

Analyze what percentage of a facility's electricity consumption is covered by contractual instruments.

```http
GET /api/v1/scope2-market/coverage/{facility_id}?year=2025&tenant_id=tenant_abc
```

**Response:**

```json
{
  "facility_id": "facility_hq",
  "year": 2025,
  "total_consumption_kwh": 500000.0,
  "covered_kwh": 400000.0,
  "uncovered_kwh": 100000.0,
  "coverage_pct": 80.0,
  "by_instrument_type": {
    "REC": 250000.0,
    "PPA": 150000.0
  },
  "by_generation_source": {
    "wind": 300000.0,
    "solar": 100000.0
  }
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid input parameters |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation, facility, or instrument not found |
| 409 | Conflict -- instrument already retired |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- service not initialized |
