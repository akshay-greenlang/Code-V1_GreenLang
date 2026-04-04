# Fugitive Emissions Agent API Reference (AGENT-MRV-005)

## Overview

The Fugitive Emissions Agent (GL-MRV-X-005) calculates GHG Protocol Scope 1 fugitive emissions from equipment leaks, tank storage, wastewater treatment, pneumatic devices, compressor seals, and dehydrators. Implements EPA Method 21 and LDAR survey workflows with full provenance tracking.

**API Prefix:** `/api/v1/fugitive-emissions`
**Agent ID:** GL-MRV-X-005
**Status:** Production Ready

**Emission Source Types:**
- `EQUIPMENT_LEAK` - Valves, connectors, pump seals, flanges
- `TANK_STORAGE` - Standing and working losses
- `WASTEWATER_TREATMENT` - Dissolved and entrained gases
- `PNEUMATIC_DEVICE` - High/low bleed controllers
- `COMPRESSOR_SEAL` - Centrifugal and reciprocating seals
- `DEHYDRATOR` - Glycol dehydrator vents

**Calculation Methods:**
- `AVERAGE_EMISSION_FACTOR` - EPA default component factors
- `SCREENING_RANGE` - Background + pegged source screening
- `UNIT_CORRELATION` - Correlation equations from screening data
- `DIRECT_MEASUREMENT` - OGI/Method 21 measurements
- `MASS_BALANCE` - Input/output mass balance

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculate` | Calculate fugitive emissions | 201, 400, 500 |
| 2 | POST | `/calculate/batch` | Batch calculation | 201, 400, 500 |
| 3 | GET | `/calculations` | List calculations | 200, 500 |
| 4 | GET | `/calculations/{calc_id}` | Get calculation by ID | 200, 404, 500 |
| 5 | POST | `/sources` | Register emission source | 201, 400, 409, 500 |
| 6 | GET | `/sources` | List emission sources | 200, 500 |
| 7 | GET | `/sources/{source_id}` | Get emission source | 200, 404, 500 |
| 8 | POST | `/components` | Register component | 201, 400, 409, 500 |
| 9 | GET | `/components` | List components | 200, 500 |
| 10 | GET | `/components/{component_id}` | Get component | 200, 404, 500 |
| 11 | POST | `/surveys` | Register LDAR survey | 201, 400, 500 |
| 12 | GET | `/surveys` | List surveys | 200, 500 |
| 13 | POST | `/factors` | Register emission factor | 201, 400, 500 |
| 14 | GET | `/factors` | List emission factors | 200, 500 |
| 15 | POST | `/repairs` | Register repair event | 201, 400, 500 |
| 16 | GET | `/repairs` | List repair events | 200, 500 |
| 17 | POST | `/uncertainty` | Run uncertainty analysis | 200, 400, 404, 500 |
| 18 | POST | `/compliance/check` | Check regulatory compliance | 200, 400, 500 |
| 19 | GET | `/health` | Service health check | 200 |
| 20 | GET | `/stats` | Service statistics | 200, 500 |

---

## Endpoints

### 1. POST /calculate

Calculate fugitive emissions for a single source or component set.

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "source_type": "EQUIPMENT_LEAK",
  "method": "AVERAGE_EMISSION_FACTOR",
  "component_counts": {
    "valves": 120,
    "connectors": 450,
    "pump_seals": 15,
    "flanges": 200
  },
  "gas_composition": {
    "methane_fraction": 0.85,
    "voc_fraction": 0.05
  },
  "operating_hours": 8760,
  "gwp_source": "AR6",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "calc_id": "fc_abc123",
  "facility_id": "facility-001",
  "source_type": "EQUIPMENT_LEAK",
  "method": "AVERAGE_EMISSION_FACTOR",
  "total_co2e_kg": 4521.75,
  "ch4_kg": 185.42,
  "co2_kg": 12.30,
  "n2o_kg": 0.0,
  "voc_kg": 45.21,
  "gwp_source": "AR6",
  "provenance_hash": "sha256:a1b2c3d4...",
  "created_at": "2026-04-01T10:30:00Z"
}
```

| Status | Description |
|--------|-------------|
| 201 | Calculation completed |
| 400 | Invalid input (unsupported source type, method, or missing fields) |
| 500 | Internal processing error |

---

### 2. POST /calculate/batch

Process multiple fugitive emission calculations in one request.

**Request Body:**

```json
{
  "batch_id": "batch-001",
  "requests": [
    {
      "facility_id": "facility-001",
      "source_type": "EQUIPMENT_LEAK",
      "method": "AVERAGE_EMISSION_FACTOR",
      "component_counts": { "valves": 50, "connectors": 200 },
      "operating_hours": 8760
    },
    {
      "facility_id": "facility-002",
      "source_type": "PNEUMATIC_DEVICE",
      "method": "DIRECT_MEASUREMENT",
      "measurement_data": { "device_count": 10, "bleed_rate_scfh": 6.5 },
      "operating_hours": 8760
    }
  ],
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):**

```json
{
  "batch_id": "batch-001",
  "total_requests": 2,
  "success_count": 2,
  "failure_count": 0,
  "total_co2e_kg": 6250.80,
  "results": [ { "...": "..." } ],
  "errors": []
}
```

---

### 3. GET /calculations

List fugitive emission calculations with pagination.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 50 | Results per page (max 500) |
| `facility_id` | string | -- | Filter by facility |
| `source_type` | string | -- | Filter by source type |

**Response (200 OK):**

```json
{
  "calculations": [ { "calc_id": "fc_abc123", "...": "..." } ],
  "count": 42,
  "page": 1,
  "page_size": 50
}
```

---

### 4. GET /calculations/{calc_id}

Retrieve a specific calculation by its unique identifier.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `calc_id` | string | Calculation UUID |

**Response (200 OK):** Full calculation result object.

| Status | Description |
|--------|-------------|
| 200 | Calculation found |
| 404 | Calculation not found |

---

### 5. POST /sources

Register an emission source (equipment, tank, or process unit).

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "source_type": "EQUIPMENT_LEAK",
  "name": "Process Unit A - Valve Bank",
  "location": "Unit A, Row 3",
  "component_counts": { "valves": 120, "connectors": 450 },
  "service_type": "gas_wet",
  "tenant_id": "tenant-001"
}
```

**Response (201 Created):** Source registration record with generated `source_id`.

---

### 6-10. Sources and Components CRUD

Standard CRUD endpoints for emission sources (`/sources`, `/sources/{source_id}`) and individual components (`/components`, `/components/{component_id}`). Supports registration (POST), listing with pagination (GET), and individual retrieval (GET by ID).

---

### 11-12. LDAR Surveys

**POST /surveys** -- Register a Leak Detection and Repair (LDAR) survey result.

**Request Body:**

```json
{
  "facility_id": "facility-001",
  "survey_type": "OGI",
  "survey_date": "2026-03-15",
  "components_surveyed": 785,
  "leaks_detected": 12,
  "total_leak_rate_kg_hr": 0.85,
  "surveyor": "Inspector-A",
  "tenant_id": "tenant-001"
}
```

**GET /surveys** -- List surveys with pagination and optional facility filter.

---

### 13-14. Emission Factors

**POST /factors** -- Register a custom emission factor.
**GET /factors** -- List available emission factors with optional source_type and method filters.

---

### 15-16. Repair Events

**POST /repairs** -- Register a repair event following leak detection.
**GET /repairs** -- List repair events with pagination.

---

### 17. POST /uncertainty

Run Monte Carlo or analytical uncertainty analysis on a completed calculation.

**Request Body:**

```json
{
  "calc_id": "fc_abc123",
  "method": "monte_carlo",
  "iterations": 10000,
  "confidence_level": 95.0,
  "seed": 42
}
```

**Response (200 OK):**

```json
{
  "calc_id": "fc_abc123",
  "method": "monte_carlo",
  "mean_co2e_kg": 4521.75,
  "std_dev_kg": 450.12,
  "ci_lower_kg": 3700.50,
  "ci_upper_kg": 5342.99,
  "uncertainty_pct": 18.2,
  "iterations": 10000
}
```

---

### 18. POST /compliance/check

Check calculation results against regulatory frameworks (GHG Protocol, EPA Subpart W, CSRD, ISO 14064).

**Request Body:**

```json
{
  "calc_id": "fc_abc123",
  "frameworks": ["ghg_protocol", "epa_subpart_w", "iso_14064"]
}
```

---

### 19. GET /health

Health check endpoint. Returns service status, agent version, and engine availability.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "agent_id": "GL-MRV-X-005",
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

---

### 20. GET /stats

Service statistics including calculation count, source count, and throughput metrics.

---

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Human-readable error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid input data or unsupported parameter values |
| 404 | Not Found -- Requested resource does not exist |
| 409 | Conflict -- Duplicate resource (e.g., duplicate source_id) |
| 500 | Internal Server Error -- Unexpected processing failure |
