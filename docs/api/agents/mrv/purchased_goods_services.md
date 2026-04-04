# Purchased Goods & Services API Reference

**Agent:** AGENT-MRV-014 (GL-MRV-S3-001)
**Prefix:** `/api/v1/purchased-goods`
**Source:** `greenlang/agents/mrv/purchased_goods_services/api/router.py`
**Status:** Production Ready

## Overview

The Purchased Goods & Services agent calculates Scope 3 Category 1 emissions using four methods: spend-based (EEIO), average-data, supplier-specific, and hybrid. It manages supplier profiles with data quality indicator (DQI) scoring, maintains EEIO emission factor databases (USEEIO v2.0, EXIOBASE v3.8, GTAP v11, EORA v26), performs compliance checking against 7 regulatory frameworks, and supports export in multiple formats. Uses the `create_router()` factory pattern with typed Pydantic request models.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/calculations` | Execute single calculation | Yes |
| 2 | POST | `/calculations/batch` | Execute batch calculations | Yes |
| 3 | GET | `/calculations` | List calculations with filters | Yes |
| 4 | GET | `/calculations/{id}` | Get calculation by ID | Yes |
| 5 | DELETE | `/calculations/{id}` | Delete calculation | Yes |
| 6 | POST | `/suppliers` | Register a supplier | Yes |
| 7 | GET | `/suppliers` | List suppliers | Yes |
| 8 | PUT | `/suppliers/{id}` | Update supplier | Yes |
| 9 | GET | `/suppliers/{id}` | Get supplier details | Yes |
| 10 | GET | `/emission-factors` | List EEIO emission factors | Yes |
| 11 | GET | `/emission-factors/{code}` | Get factor by sector code | Yes |
| 12 | POST | `/compliance/check` | Run compliance check | Yes |
| 13 | GET | `/compliance/{id}` | Get compliance result | Yes |
| 14 | POST | `/uncertainty` | Run uncertainty analysis | Yes |
| 15 | GET | `/aggregations` | Get aggregated emissions | Yes |
| 16 | GET | `/hot-spots` | Identify emission hot-spots | Yes |
| 17 | GET | `/dqi/{supplier_id}` | Get supplier DQI score | Yes |
| 18 | POST | `/export` | Export calculations | Yes |
| 19 | GET | `/health` | Service health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Execute Single Calculation

```http
POST /api/v1/purchased-goods/calculations
```

**Request Body (Spend-Based):**

```json
{
  "method": "SPEND_BASED",
  "supplier_id": "sup_001",
  "sector_code": "325110",
  "spend_amount": 150000.00,
  "spend_currency": "USD",
  "eeio_database": "USEEIO_v2_0",
  "year": 2025,
  "tenant_id": "tenant_abc"
}
```

**Request Body (Supplier-Specific):**

```json
{
  "method": "SUPPLIER_SPECIFIC",
  "supplier_id": "sup_002",
  "supplier_emissions_kg": 12500.0,
  "allocation_factor": 0.15,
  "year": 2025,
  "tenant_id": "tenant_abc"
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | string | Yes | `SPEND_BASED`, `AVERAGE_DATA`, `SUPPLIER_SPECIFIC`, `HYBRID` |
| `supplier_id` | string | Optional | Supplier identifier |
| `sector_code` | string | Cond. | NAICS/NACE sector code (required for spend-based) |
| `spend_amount` | float | Cond. | Spend amount (required for spend-based) |
| `spend_currency` | string | Optional | ISO 4217 currency code (default: USD) |
| `eeio_database` | string | Optional | EEIO database to use |
| `supplier_emissions_kg` | float | Cond. | Supplier-reported emissions (for supplier-specific) |
| `allocation_factor` | float | Cond. | Revenue allocation factor (for supplier-specific) |
| `year` | integer | Yes | Calculation year |
| `tenant_id` | string | Yes | Tenant identifier |

**EEIO Databases:**

| Database | Coverage | Sectors |
|----------|----------|---------|
| USEEIO v2.0 | United States | 400+ NAICS sectors |
| EXIOBASE v3.8 | Global (49 regions) | 200 product groups |
| GTAP v11 | Global (141 regions) | 65 sectors |
| EORA v26 | Global (190 countries) | 26 sectors |

**Response:**

```json
{
  "calculation_id": "calc_pgs_001",
  "method": "SPEND_BASED",
  "supplier_id": "sup_001",
  "sector_code": "325110",
  "spend_amount": 150000.00,
  "eeio_database": "USEEIO_v2_0",
  "emission_factor_kg_per_usd": 0.45,
  "total_co2e_kg": 67500.0,
  "dqi_score": 3,
  "provenance_hash": "sha256:...",
  "calculated_at": "2026-04-04T10:30:00Z"
}
```

### 6. Register Supplier

```http
POST /api/v1/purchased-goods/suppliers
```

**Request Body:**

```json
{
  "supplier_name": "Acme Chemicals Inc.",
  "supplier_id": "sup_001",
  "sector_code": "325110",
  "country": "US",
  "data_quality_level": "supplier_specific",
  "has_verified_emissions": true,
  "contact_email": "sustainability@acme.com",
  "tenant_id": "tenant_abc"
}
```

### 16. Identify Emission Hot-Spots

```http
GET /api/v1/purchased-goods/hot-spots?tenant_id=tenant_abc&top_n=10&year=2025
```

**Response:**

```json
{
  "tenant_id": "tenant_abc",
  "year": 2025,
  "hot_spots": [
    {
      "sector_code": "325110",
      "sector_description": "Petrochemical manufacturing",
      "total_spend": 5000000.0,
      "total_co2e_kg": 2250000.0,
      "supplier_count": 12,
      "pct_of_total_emissions": 35.0
    }
  ],
  "total_co2e_kg": 6428571.0
}
```

### 17. Get Supplier DQI Score

Data Quality Indicator scores follow GHG Protocol Scope 3 guidance (1 = best, 5 = worst).

```http
GET /api/v1/purchased-goods/dqi/{supplier_id}?tenant_id=tenant_abc
```

**Response:**

```json
{
  "supplier_id": "sup_001",
  "dqi_score": 2,
  "dqi_components": {
    "data_source": 2,
    "verification": 1,
    "temporal_correlation": 2,
    "geographical_correlation": 2,
    "technological_correlation": 3
  },
  "recommendation": "Supplier provides verified emissions data. Consider requesting product-level LCA data to improve technological correlation."
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid method or missing required fields |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- calculation, supplier, or emission factor not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- service not initialized |
