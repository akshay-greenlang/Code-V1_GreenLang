# Mobile Combustion API Reference

**Agent:** AGENT-MRV-003 (GL-MRV-SCOPE1-003)
**Prefix:** `/api/v1/mobile-combustion`
**Source:** `greenlang/agents/mrv/mobile_combustion/api/router.py`
**Status:** Production Ready

## Overview

The Mobile Combustion agent calculates Scope 1 GHG emissions from company-owned or company-controlled vehicles and mobile equipment. It supports three calculation methods (fuel-based, distance-based, spend-based), vehicle fleet management, trip logging, fleet aggregation, custom fuel and emission factor registration, compliance checking, and uncertainty analysis. Follows the GHG Protocol Corporate Standard and EPA emission factor guidance.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculate` | Calculate emissions | Yes |
| 2 | POST | `/calculate/batch` | Batch calculation | Yes |
| 3 | GET | `/calculations` | List calculations | Yes |
| 4 | GET | `/calculations/{calc_id}` | Get calculation detail | Yes |
| 5 | POST | `/vehicles` | Register vehicle | Yes |
| 6 | GET | `/vehicles` | List vehicles | Yes |
| 7 | GET | `/vehicles/{vehicle_id}` | Get vehicle detail | Yes |
| 8 | POST | `/trips` | Log a trip | Yes |
| 9 | GET | `/trips` | List trips | Yes |
| 10 | GET | `/trips/{trip_id}` | Get trip detail | Yes |
| 11 | POST | `/fuels` | Register custom fuel | Yes |
| 12 | GET | `/fuels` | List fuel types | Yes |
| 13 | POST | `/factors` | Register custom emission factor | Yes |
| 14 | GET | `/factors` | List emission factors | Yes |
| 15 | POST | `/aggregate` | Aggregate fleet emissions | Yes |
| 16 | GET | `/aggregations` | List aggregations | Yes |
| 17 | POST | `/uncertainty` | Run uncertainty analysis | Yes |
| 18 | POST | `/compliance/check` | Run compliance check | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Calculate Mobile Combustion Emissions

```http
POST /api/v1/mobile-combustion/calculate
```

**Request Body (Fuel-Based):**

```json
{
  "calculation_method": "FUEL_BASED",
  "fuel_type": "diesel",
  "fuel_quantity": 500.0,
  "fuel_unit": "gallons",
  "vehicle_type": "heavy_duty_truck",
  "vehicle_id": "truck_001",
  "facility_id": "facility_hq",
  "gwp_source": "AR6"
}
```

**Request Body (Distance-Based):**

```json
{
  "calculation_method": "DISTANCE_BASED",
  "vehicle_type": "passenger_car",
  "fuel_type": "gasoline",
  "distance": 15000.0,
  "distance_unit": "miles",
  "vehicle_id": "car_fleet_01",
  "gwp_source": "AR6"
}
```

**Request Body (Spend-Based):**

```json
{
  "calculation_method": "SPEND_BASED",
  "vehicle_type": "light_duty_truck",
  "spend_amount": 25000.0,
  "spend_currency": "USD",
  "gwp_source": "AR6"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `calculation_method` | string | Yes | `FUEL_BASED`, `DISTANCE_BASED`, or `SPEND_BASED` |
| `fuel_type` | string | Cond. | Fuel type (gasoline, diesel, biodiesel, CNG, LPG, propane, etc.) |
| `fuel_quantity` | float | Cond. | Fuel consumed (required for FUEL_BASED) |
| `fuel_unit` | string | Cond. | Fuel unit (gallons, liters, kg, therms, scf) |
| `vehicle_type` | string | Optional | Vehicle category |
| `distance` | float | Cond. | Distance traveled (required for DISTANCE_BASED) |
| `distance_unit` | string | Cond. | Distance unit (miles, km) |
| `spend_amount` | float | Cond. | Spend amount (required for SPEND_BASED) |
| `spend_currency` | string | Optional | Currency code (default: USD) |
| `vehicle_id` | string | Optional | Vehicle identifier |
| `facility_id` | string | Optional | Facility identifier |
| `gwp_source` | string | Optional | GWP source: `AR4`, `AR5`, `AR6` (default), `AR6_20YR` |
| `tier` | string | Optional | Calculation tier: `TIER_1`, `TIER_2`, `TIER_3` |

**Response:**

```json
{
  "calculation_id": "calc_mob_001",
  "calculation_method": "FUEL_BASED",
  "fuel_type": "diesel",
  "fuel_quantity": 500.0,
  "fuel_unit": "gallons",
  "vehicle_type": "heavy_duty_truck",
  "co2_kg": 5096.0,
  "ch4_kg": 0.015,
  "n2o_kg": 0.082,
  "total_co2e_kg": 5120.5,
  "emission_factor_source": "EPA",
  "gwp_source": "AR6",
  "provenance_hash": "sha256:...",
  "calculated_at": "2026-04-04T10:30:00Z"
}
```

### 5. Register Vehicle

```http
POST /api/v1/mobile-combustion/vehicles
```

**Request Body:**

```json
{
  "vehicle_id": "truck_001",
  "vehicle_type": "heavy_duty_truck",
  "make": "Volvo",
  "model": "FH16",
  "year": 2023,
  "fuel_type": "diesel",
  "gvw_class": "class_8",
  "facility_id": "facility_hq",
  "odometer_km": 85000
}
```

**Vehicle Types:**
`passenger_car`, `light_duty_truck`, `medium_duty_truck`, `heavy_duty_truck`, `bus`, `motorcycle`, `off_road_equipment`, `agricultural_equipment`, `construction_equipment`, `marine_vessel`, `rail_locomotive`, `aircraft`

### 8. Log a Trip

```http
POST /api/v1/mobile-combustion/trips
```

**Request Body:**

```json
{
  "vehicle_id": "truck_001",
  "trip_date": "2026-03-15",
  "origin": "Munich",
  "destination": "Stuttgart",
  "distance_km": 230.0,
  "fuel_consumed_l": 65.0,
  "fuel_type": "diesel",
  "purpose": "delivery",
  "driver_id": "driver_42"
}
```

### 15. Aggregate Fleet Emissions

Aggregate emissions across the entire vehicle fleet for a facility or time period.

```http
POST /api/v1/mobile-combustion/aggregate
```

**Request Body:**

```json
{
  "facility_id": "facility_hq",
  "year": 2025,
  "group_by": ["vehicle_type", "fuel_type"],
  "include_trip_details": false
}
```

**Response:**

```json
{
  "aggregation_id": "agg_mob_001",
  "facility_id": "facility_hq",
  "year": 2025,
  "total_co2e_kg": 285000.0,
  "total_vehicles": 45,
  "total_trips": 1250,
  "by_vehicle_type": {
    "heavy_duty_truck": {"co2e_kg": 180000.0, "vehicles": 12},
    "passenger_car": {"co2e_kg": 75000.0, "vehicles": 28},
    "light_duty_truck": {"co2e_kg": 30000.0, "vehicles": 5}
  },
  "by_fuel_type": {
    "diesel": {"co2e_kg": 210000.0},
    "gasoline": {"co2e_kg": 75000.0}
  },
  "provenance_hash": "sha256:..."
}
```

### 18. Compliance Check

```http
POST /api/v1/mobile-combustion/compliance/check
```

**Request Body:**

```json
{
  "calculation_id": "calc_mob_001",
  "frameworks": ["ghg_protocol", "iso_14064", "epa_40cfr98"]
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- fuel_quantity required for FUEL_BASED, distance required for DISTANCE_BASED |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation, vehicle, or trip not found |
| 503 | Service Unavailable -- Mobile Combustion service not initialized |
