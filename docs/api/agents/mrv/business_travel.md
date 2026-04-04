# Business Travel API Reference

**Agent:** AGENT-MRV-019 (GL-MRV-S3-006)
**Prefix:** `/api/v1/business-travel`
**Source:** `greenlang/agents/mrv/business_travel/api/router.py`
**Status:** Production Ready

## Overview

The Business Travel agent calculates Scope 3 Category 6 emissions from employee business travel across all transport modes (air, rail, road, bus, taxi, ferry, motorcycle) and hotel accommodation. It provides mode-specific calculation endpoints (flight with IATA airport pairs, rail with 8 rail types, road with 13 vehicle types, hotel with 16 country-specific factors), spend-based EEIO calculations, compliance checking against 7 frameworks, Monte Carlo/analytical/IPCC Tier 2 uncertainty analysis, provenance chain verification, and Pareto-based hot-spot analysis. Uses GreenLangBase schema models and DEFRA 2024 emission factors.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculate` | Full pipeline calculation | Yes |
| 2 | POST | `/calculate/batch` | Batch calculation (up to 10,000) | Yes |
| 3 | POST | `/calculate/flight` | Flight-specific calculation | Yes |
| 4 | POST | `/calculate/rail` | Rail-specific calculation | Yes |
| 5 | POST | `/calculate/road` | Road-specific calculation | Yes |
| 6 | POST | `/calculate/hotel` | Hotel accommodation calculation | Yes |
| 7 | POST | `/calculate/spend` | Spend-based EEIO calculation | Yes |
| 8 | GET | `/calculations/{calculation_id}` | Get calculation detail | Yes |
| 9 | GET | `/calculations` | List calculations (paginated) | Yes |
| 10 | DELETE | `/calculations/{calculation_id}` | Soft-delete calculation | Yes |
| 11 | GET | `/emission-factors` | List emission factors | Yes |
| 12 | GET | `/emission-factors/{mode}` | Get factors by mode | Yes |
| 13 | GET | `/airports` | Search airports | Yes |
| 14 | GET | `/transport-modes` | List transport modes | Yes |
| 15 | GET | `/cabin-classes` | List cabin class multipliers | Yes |
| 16 | POST | `/compliance/check` | Multi-framework compliance check | Yes |
| 17 | POST | `/uncertainty/analyze` | Uncertainty analysis | Yes |
| 18 | GET | `/aggregations/{period}` | Get aggregated emissions | Yes |
| 19 | POST | `/hot-spots/analyze` | Hot-spot analysis | Yes |
| 20 | GET | `/provenance/{calculation_id}` | Get provenance chain | Yes |
| 21 | GET | `/health` | Health check | No |
| 22 | GET | `/stats` | Agent statistics | Yes |

---

## Key Endpoints

### 3. Calculate Flight Emissions

Calculate air travel emissions using great-circle distance, DEFRA 2024 distance-band factors, cabin class multipliers, and optional radiative forcing uplift.

```http
POST /api/v1/business-travel/calculate/flight
```

**Request Body:**

```json
{
  "origin_iata": "LHR",
  "destination_iata": "JFK",
  "cabin_class": "business",
  "passengers": 1,
  "round_trip": true,
  "rf_option": "with_rf"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `origin_iata` | string | Yes | 3-letter IATA origin airport code |
| `destination_iata` | string | Yes | 3-letter IATA destination airport code |
| `cabin_class` | string | Optional | `economy` (default), `premium_economy`, `business`, `first` |
| `passengers` | integer | Optional | Number of passengers (default: 1, max: 500) |
| `round_trip` | boolean | Optional | Doubles distance if true (default: false) |
| `rf_option` | string | Optional | Radiative forcing: `with_rf` (default), `without_rf`, `both` |

**Cabin Class Multipliers (DEFRA 2024):**

| Class | Multiplier |
|-------|------------|
| Economy | 1.0 |
| Premium Economy | 1.6 |
| Business | 2.9 |
| First | 4.0 |

**Response:**

```json
{
  "calculation_id": "calc_flight_001",
  "origin_iata": "LHR",
  "destination_iata": "JFK",
  "distance_km": 5570.0,
  "distance_band": "long_haul",
  "cabin_class": "business",
  "class_multiplier": 2.9,
  "co2e_without_rf_kg": 1856.0,
  "co2e_with_rf_kg": 3340.8,
  "wtt_co2e_kg": 371.2,
  "total_co2e_kg": 3712.0,
  "rf_option": "with_rf",
  "provenance_hash": "sha256:..."
}
```

### 4. Calculate Rail Travel Emissions

```http
POST /api/v1/business-travel/calculate/rail
```

**Request Body:**

```json
{
  "rail_type": "eurostar",
  "distance_km": 450.0,
  "passengers": 2
}
```

**Supported Rail Types:**
`national`, `international`, `light_rail`, `underground`, `eurostar`, `high_speed`, `us_intercity`, `us_commuter`

### 6. Calculate Hotel Accommodation Emissions

```http
POST /api/v1/business-travel/calculate/hotel
```

**Request Body:**

```json
{
  "country_code": "GB",
  "room_nights": 3,
  "hotel_class": "upscale"
}
```

**Hotel Classes:** `budget`, `standard`, `upscale`, `luxury`

### 7. Calculate Spend-Based Emissions

```http
POST /api/v1/business-travel/calculate/spend
```

**Request Body:**

```json
{
  "naics_code": "481111",
  "amount": 12500.00,
  "currency": "USD",
  "reporting_year": 2025
}
```

### 16. Multi-Framework Compliance Check

```http
POST /api/v1/business-travel/compliance/check
```

**Request Body:**

```json
{
  "frameworks": ["ghg_protocol", "csrd_esrs", "cdp", "sbti"],
  "calculation_results": [
    {"calculation_id": "calc_001", "mode": "air", "total_co2e_kg": 3712.0},
    {"calculation_id": "calc_002", "mode": "rail", "total_co2e_kg": 12.5}
  ],
  "rf_disclosed": true,
  "mode_breakdown_provided": true
}
```

**Response:**

```json
{
  "results": [
    {"framework": "ghg_protocol", "status": "pass", "score": 1.0, "findings": []},
    {"framework": "csrd_esrs", "status": "pass", "score": 0.95, "findings": ["Recommend disclosing RF uplift separately"]},
    {"framework": "cdp", "status": "pass", "score": 1.0, "findings": []},
    {"framework": "sbti", "status": "pass", "score": 0.90, "findings": []}
  ],
  "overall_status": "pass",
  "overall_score": 0.96
}
```

### 20. Get Provenance Chain

Retrieve the complete 10-stage SHA-256 provenance chain for a calculation.

```http
GET /api/v1/business-travel/provenance/{calculation_id}
```

**Response:**

```json
{
  "calculation_id": "calc_flight_001",
  "chain": [
    {"stage": "validate", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:00Z"},
    {"stage": "classify", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:01Z"},
    {"stage": "normalize", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:01Z"},
    {"stage": "resolve_efs", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:02Z"},
    {"stage": "calculate_flights", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:02Z"},
    {"stage": "calculate_ground", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:02Z"},
    {"stage": "allocate", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:03Z"},
    {"stage": "compliance", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:03Z"},
    {"stage": "aggregate", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:03Z"},
    {"stage": "seal", "hash": "sha256:...", "timestamp": "2026-04-04T10:30:04Z"}
  ],
  "is_valid": true,
  "root_hash": "sha256:final_root_hash..."
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid IATA code, missing distance/fuel data, invalid parameters |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- BusinessTravelService not initialized |
