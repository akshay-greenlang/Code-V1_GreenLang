# Upstream Transportation & Distribution API Reference

**Agent:** AGENT-MRV-017 (GL-MRV-S3-004)
**Prefix:** `/api/v1/upstream-transportation`
**Source:** `greenlang/agents/mrv/upstream_transportation/api/router.py`
**Status:** Production Ready

## Overview

The Upstream Transportation & Distribution agent calculates Scope 3 Category 4 emissions from transportation and distribution of purchased goods in vehicles and facilities not owned or controlled by the reporting company. It supports six transport modes (road, rail, air, sea, pipeline, multimodal), four calculation methods (distance-based, spend-based, fuel-based, supplier-specific), multi-modal transport chain management, auto-classification of transport activities via ML, compliance checking against GHG Protocol/ISO 14083/GLEC frameworks, and hot-spot analysis. All numeric outputs use Python `Decimal` arithmetic for deterministic, zero-hallucination results.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculate` | Calculate emissions (single) | Yes |
| 2 | POST | `/calculate/batch` | Batch calculation (up to 10,000) | Yes |
| 3 | GET | `/calculations` | List calculations with filters | Yes |
| 4 | GET | `/calculations/{calculation_id}` | Get calculation detail | Yes |
| 5 | DELETE | `/calculations/{calculation_id}` | Delete calculation (soft delete) | Yes |
| 6 | POST | `/transport-chains` | Create multi-modal transport chain | Yes |
| 7 | GET | `/transport-chains` | List transport chains | Yes |
| 8 | GET | `/transport-chains/{chain_id}` | Get transport chain detail | Yes |
| 9 | GET | `/emission-factors` | List emission factors | Yes |
| 10 | GET | `/emission-factors/{factor_id}` | Get emission factor detail | Yes |
| 11 | POST | `/emission-factors/custom` | Create custom emission factor | Yes |
| 12 | POST | `/classify` | Auto-classify transport activity | Yes |
| 13 | POST | `/compliance/check` | Check calculation compliance | Yes |
| 14 | GET | `/compliance/{check_id}` | Get compliance detail | Yes |
| 15 | POST | `/uncertainty` | Analyze calculation uncertainty | Yes |
| 16 | GET | `/aggregations` | Get aggregated emissions | Yes |
| 17 | GET | `/hot-spots` | Identify emission hot-spots | Yes |
| 18 | POST | `/export` | Export calculations | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Calculate Emissions (Single)

```http
POST /api/v1/upstream-transportation/calculate
```

**Request Body (Distance-Based):**

```json
{
  "tenant_id": "tenant_abc",
  "mode": "ROAD",
  "vehicle_type": "HEAVY_TRUCK",
  "distance_km": 850.0,
  "mass_tonnes": 22.5,
  "method": "DISTANCE_BASED",
  "origin_country": "DEU",
  "destination_country": "FRA",
  "year": 2025
}
```

**Request Body (Fuel-Based):**

```json
{
  "tenant_id": "tenant_abc",
  "mode": "SEA",
  "vehicle_type": "CONTAINER_SHIP",
  "method": "FUEL_BASED",
  "fuel_type": "marine_diesel",
  "fuel_amount_l": 45000.0,
  "year": 2025
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tenant_id` | string | Yes | Tenant identifier |
| `mode` | enum | Yes | `ROAD`, `RAIL`, `AIR`, `SEA`, `PIPELINE`, `MULTIMODAL` |
| `vehicle_type` | enum | Cond. | Vehicle type (required for most modes) |
| `distance_km` | decimal | Cond. | Distance in km (required for DISTANCE_BASED) |
| `mass_tonnes` | decimal | Optional | Mass in metric tonnes |
| `method` | enum | Optional | `DISTANCE_BASED` (default), `SPEND_BASED`, `FUEL_BASED`, `SUPPLIER_SPECIFIC` |
| `fuel_type` | string | Cond. | Fuel type (required for FUEL_BASED) |
| `fuel_amount_l` | decimal | Cond. | Fuel amount in liters (required for FUEL_BASED) |
| `spend_usd` | decimal | Cond. | Spend amount in USD (required for SPEND_BASED) |
| `emission_factor_id` | UUID | Optional | Custom emission factor ID |
| `origin_country` | string | Optional | ISO 3166-1 alpha-3 origin country code |
| `destination_country` | string | Optional | ISO 3166-1 alpha-3 destination country code |
| `year` | integer | Yes | Calculation year (1990-2100) |

**Response:**

```json
{
  "calculation_id": "550e8400-e29b-41d4-a716-446655440000",
  "tenant_id": "tenant_abc",
  "mode": "ROAD",
  "vehicle_type": "HEAVY_TRUCK",
  "method": "DISTANCE_BASED",
  "distance_km": 850.0,
  "mass_tonnes": 22.5,
  "co2_kg": 1234.56,
  "ch4_kg": 0.05,
  "n2o_kg": 0.02,
  "co2e_kg": 1241.83,
  "emission_factor_source": "DEFRA 2024",
  "calculation_timestamp": "2026-04-04T10:30:00Z",
  "provenance_hash": "sha256:a1b2c3..."
}
```

### 6. Create Multi-Modal Transport Chain

Model a complete supply chain transportation route with multiple sequential legs across different modes.

```http
POST /api/v1/upstream-transportation/transport-chains
```

**Request Body:**

```json
{
  "tenant_id": "tenant_abc",
  "chain_name": "Shanghai to Munich via Rotterdam",
  "legs": [
    {
      "tenant_id": "tenant_abc",
      "mode": "SEA",
      "vehicle_type": "CONTAINER_SHIP",
      "distance_km": 19500.0,
      "mass_tonnes": 22.5,
      "method": "DISTANCE_BASED",
      "year": 2025
    },
    {
      "tenant_id": "tenant_abc",
      "mode": "ROAD",
      "vehicle_type": "HEAVY_TRUCK",
      "distance_km": 850.0,
      "mass_tonnes": 22.5,
      "method": "DISTANCE_BASED",
      "year": 2025
    }
  ]
}
```

**Response:**

```json
{
  "chain_id": "660e8400-e29b-41d4-a716-446655440001",
  "tenant_id": "tenant_abc",
  "chain_name": "Shanghai to Munich via Rotterdam",
  "total_legs": 2,
  "total_distance_km": 20350.0,
  "total_co2e_kg": 4567.89,
  "legs": [
    {"mode": "SEA", "co2e_kg": 3333.33, "distance_km": 19500.0},
    {"mode": "ROAD", "co2e_kg": 1234.56, "distance_km": 850.0}
  ],
  "created_at": "2026-04-04T10:30:00Z"
}
```

### 12. Auto-Classify Transport Activity

Use ML to classify a text description into transport mode and vehicle type.

```http
POST /api/v1/upstream-transportation/classify
```

**Request Body:**

```json
{
  "tenant_id": "tenant_abc",
  "description": "20ft container shipped from Shenzhen port to Rotterdam via Maersk liner service",
  "additional_context": {"origin": "China", "destination": "Netherlands"}
}
```

**Response:**

```json
{
  "tenant_id": "tenant_abc",
  "description": "20ft container shipped from Shenzhen port to Rotterdam...",
  "predicted_mode": "SEA",
  "predicted_vehicle_type": "CONTAINER_SHIP",
  "confidence": 0.96,
  "alternatives": [
    {"mode": "MULTIMODAL", "vehicle_type": null, "confidence": 0.03}
  ],
  "classification_timestamp": "2026-04-04T10:30:00Z"
}
```

### 13. Check Compliance

```http
POST /api/v1/upstream-transportation/compliance/check
```

**Request Body:**

```json
{
  "tenant_id": "tenant_abc",
  "calculation_id": "550e8400-e29b-41d4-a716-446655440000",
  "frameworks": ["GHG_PROTOCOL", "ISO_14083", "GLEC"]
}
```

### 17. Identify Emission Hot-Spots

```http
GET /api/v1/upstream-transportation/hot-spots?tenant_id=tenant_abc&analysis_type=emissions&top_n=10
```

**Response:**

```json
{
  "tenant_id": "tenant_abc",
  "analysis_type": "emissions",
  "top_n": 10,
  "hot_spots": [
    {
      "category": "SEA - CONTAINER_SHIP",
      "total_co2e_kg": 125000.0,
      "pct_of_total": 45.0,
      "calculation_count": 50
    },
    {
      "category": "AIR - CARGO_AIRCRAFT",
      "total_co2e_kg": 95000.0,
      "pct_of_total": 34.0,
      "calculation_count": 15
    }
  ],
  "total_co2e_kg": 277777.0,
  "analysis_timestamp": "2026-04-04T10:30:00Z"
}
```

---

## Transport Modes and Vehicle Types

| Mode | Vehicle Types |
|------|---------------|
| ROAD | LIGHT_TRUCK, MEDIUM_TRUCK, HEAVY_TRUCK, ARTICULATED_TRUCK, VAN, COURIER |
| RAIL | FREIGHT_TRAIN, INTERMODAL_TRAIN, BULK_TRAIN |
| AIR | CARGO_AIRCRAFT, BELLY_FREIGHT, EXPRESS_AIR |
| SEA | CONTAINER_SHIP, BULK_CARRIER, TANKER, RORO |
| PIPELINE | GAS_PIPELINE, OIL_PIPELINE, PRODUCT_PIPELINE |
| MULTIMODAL | (combines multiple modes in a transport chain) |

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error (e.g., distance required for DISTANCE_BASED) |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation, chain, or factor not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- service not initialized |
