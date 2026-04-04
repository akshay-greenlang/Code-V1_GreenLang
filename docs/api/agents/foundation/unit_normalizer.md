# AGENT-FOUND-003: Unit & Reference Normalizer API Reference

**Agent ID:** AGENT-FOUND-003
**Service:** Unit & Reference Normalizer
**Status:** Production Ready
**Base Path:** `/api/v1/normalizer`
**Tag:** `normalizer`
**Source:** `greenlang/agents/foundation/normalizer/api/router.py`

The Unit & Reference Normalizer provides endpoints for unit conversion (single
and batch), GHG conversion using GWP factors, entity resolution (fuels,
materials, processes), dimensional analysis, and vocabulary search.

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | GET | `/health` | Health check | 200 |
| 2 | POST | `/convert` | Convert single unit | 200, 400, 503 |
| 3 | POST | `/convert/batch` | Batch convert units | 200, 400, 503 |
| 4 | POST | `/convert/ghg` | GHG conversion (GWP) | 200, 400, 503 |
| 5 | GET | `/factor` | Get conversion factor | 200, 400, 503 |
| 6 | POST | `/resolve/fuel` | Resolve fuel name | 200, 503 |
| 7 | POST | `/resolve/material` | Resolve material name | 200, 503 |
| 8 | POST | `/resolve/process` | Resolve process name | 200, 503 |
| 9 | POST | `/resolve/batch` | Batch resolve entities | 200, 400, 503 |
| 10 | POST | `/vocabulary/search` | Search entity vocabulary | 200, 400, 503 |
| 11 | POST | `/dimensions/check` | Check unit compatibility | 200, 503 |
| 12 | GET | `/dimensions` | List supported dimensions | 200, 503 |
| 13 | GET | `/units` | List supported units | 200, 400, 503 |
| 14 | GET | `/gwp` | Get GWP table | 200, 400 |
| 15 | GET | `/normalize` | Normalize unit name | 200, 503 |

---

## Detailed Endpoints

### POST /convert -- Convert Single Unit

Convert a numeric value from one unit to another.

**Request Body:**

```json
{
  "value": 1500.0,
  "from_unit": "m3",
  "to_unit": "liters",
  "precision": 4
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `value` | float | Yes | Numeric value to convert |
| `from_unit` | string | Yes | Source unit |
| `to_unit` | string | Yes | Target unit |
| `precision` | integer | No | Decimal precision (0-30) |

**Response (200):**

```json
{
  "original_value": 1500.0,
  "from_unit": "m3",
  "converted_value": 1500000.0,
  "to_unit": "liters",
  "conversion_factor": "1000",
  "precision": 4,
  "provenance_hash": "sha256:..."
}
```

---

### POST /convert/batch -- Batch Convert

Convert multiple values in a single request.

**Request Body:**

```json
{
  "items": [
    { "value": 100, "from_unit": "kg", "to_unit": "tonnes" },
    { "value": 50, "from_unit": "miles", "to_unit": "km" },
    { "value": 1000, "from_unit": "kWh", "to_unit": "MJ" }
  ]
}
```

**Response (200):**

```json
{
  "results": [
    { "original_value": 100, "from_unit": "kg", "converted_value": 0.1, "to_unit": "tonnes" },
    { "original_value": 50, "from_unit": "miles", "converted_value": 80.4672, "to_unit": "km" },
    { "original_value": 1000, "from_unit": "kWh", "converted_value": 3600.0, "to_unit": "MJ" }
  ],
  "total": 3,
  "errors": 0
}
```

---

### POST /convert/ghg -- GHG Conversion

Convert GHG emissions using Global Warming Potential (GWP) factors from IPCC
Assessment Reports.

**Request Body:**

```json
{
  "value": 100.0,
  "from_gas": "CH4",
  "to_gas": "CO2e",
  "gwp_version": "AR6",
  "gwp_timeframe": 100
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `value` | float | Yes | Mass value |
| `from_gas` | string | Yes | Source gas (CO2, CH4, N2O, HFC-134a, etc.) |
| `to_gas` | string | No | Target gas (default: `CO2e`) |
| `gwp_version` | string | No | IPCC AR version: `AR5` or `AR6` |
| `gwp_timeframe` | integer | No | GWP timeframe: `20` or `100` years |

**Response (200):**

```json
{
  "original_value": 100.0,
  "from_gas": "CH4",
  "converted_value": 2750.0,
  "to_gas": "CO2e",
  "gwp_factor": "27.5",
  "gwp_version": "AR6",
  "gwp_timeframe": 100,
  "provenance_hash": "sha256:..."
}
```

---

### GET /factor -- Get Conversion Factor

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `from_unit` | string | Yes | Source unit |
| `to_unit` | string | Yes | Target unit |

**Response (200):**

```json
{
  "from_unit": "kg",
  "to_unit": "tonnes",
  "factor": "0.001"
}
```

---

### POST /resolve/fuel -- Resolve Fuel Name

Resolve a fuel name to its canonical form using fuzzy matching.

**Request Body:**

```json
{
  "name": "nat gas"
}
```

**Response (200):**

```json
{
  "original": "nat gas",
  "canonical_name": "natural_gas",
  "confidence": 0.95,
  "category": "gaseous_fuels",
  "aliases": ["natural gas", "nat gas", "methane"],
  "provenance_hash": "sha256:..."
}
```

---

### POST /resolve/batch -- Batch Resolve Entities

**Request Body:**

```json
{
  "items": ["nat gas", "diesel", "anthracite"],
  "entity_type": "fuel"
}
```

**Response (200):**

```json
{
  "results": [
    { "original": "nat gas", "canonical_name": "natural_gas", "confidence": 0.95 },
    { "original": "diesel", "canonical_name": "diesel", "confidence": 1.0 },
    { "original": "anthracite", "canonical_name": "anthracite_coal", "confidence": 0.98 }
  ],
  "total": 3,
  "resolved": 3
}
```

---

### POST /dimensions/check -- Check Unit Compatibility

Check if two units are dimensionally compatible for conversion.

**Request Body:**

```json
{
  "from_unit": "kg",
  "to_unit": "tonnes"
}
```

**Response (200):**

```json
{
  "from_unit": "kg",
  "to_unit": "tonnes",
  "compatible": true,
  "from_dimension": "mass",
  "to_dimension": "mass"
}
```

---

### GET /dimensions -- List Supported Dimensions

**Response (200):**

```json
{
  "dimensions": [
    { "name": "mass", "base_unit": "kg", "unit_count": 12 },
    { "name": "length", "base_unit": "m", "unit_count": 8 },
    { "name": "energy", "base_unit": "J", "unit_count": 15 },
    { "name": "volume", "base_unit": "m3", "unit_count": 10 }
  ]
}
```

---

### GET /units -- List Supported Units

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dimension` | string | No | Filter by dimension (e.g., `mass`, `energy`) |

**Response (200):**

```json
{
  "units": [
    { "name": "kg", "dimension": "mass", "aliases": ["kilogram", "kilograms"] },
    { "name": "tonnes", "dimension": "mass", "aliases": ["metric_ton", "t"] }
  ]
}
```

---

### GET /gwp -- Get GWP Table

Get Global Warming Potential values for a specific IPCC version and timeframe.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | string | `AR6` | IPCC version (`AR5` or `AR6`) |
| `timeframe` | integer | `100` | GWP timeframe (20 or 100 years) |

**Response (200):**

```json
{
  "version": "AR6",
  "timeframe": 100,
  "values": {
    "CO2": "1",
    "CH4": "27.5",
    "N2O": "273",
    "SF6": "25200",
    "HFC-134a": "1530"
  }
}
```

---

### GET /normalize -- Normalize Unit Name

Normalize a unit name to its canonical form and identify its dimension.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `unit` | string | Yes | Unit name to normalize |

**Response (200):**

```json
{
  "original": "kilowatt-hours",
  "normalized": "kWh",
  "dimension": "energy",
  "is_known": true
}
```

---

## Common Error Responses

**400 Bad Request:**

```json
{
  "detail": "Units 'kg' and 'kWh' are not dimensionally compatible"
}
```

**503 Service Unavailable:**

```json
{
  "detail": "Normalizer service not configured"
}
```
