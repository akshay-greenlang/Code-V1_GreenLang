# Capital Goods Agent API Reference (AGENT-MRV-015)

## Overview

The Capital Goods Agent (GL-MRV-S3-002) calculates GHG Protocol Scope 3 Category 2 emissions from purchased capital goods (machinery, equipment, buildings, vehicles, IT infrastructure). Supports three calculation methods: spend-based (EEIO), average-data (physical quantity), and supplier-specific (EPD/PCF/CDP), plus a hybrid aggregation mode that combines all three.

**API Prefix:** `/api/v1/capital-goods`
**Agent ID:** GL-MRV-S3-002
**Status:** Production Ready

**Calculation Methods:**
- **Spend-Based (EEIO)** -- Environmentally Extended Input-Output factors by NAICS/ISIC codes
- **Average-Data (Physical)** -- Physical quantity-based emission factors per unit/tonne/m2
- **Supplier-Specific** -- EPD/PCF/CDP actual supplier emission data
- **Hybrid** -- Multi-method aggregation with hot-spot identification

**Engines (7):**
1. CapitalAssetDatabaseEngine -- EEIO/Physical EF lookup, classification
2. SpendBasedCalculatorEngine -- EEIO spend-based calculation
3. AverageDataCalculatorEngine -- Physical quantity-based calculation
4. SupplierSpecificCalculatorEngine -- EPD/PCF/CDP supplier-specific calc
5. HybridAggregatorEngine -- Multi-method aggregation and hot-spot analysis
6. ComplianceCheckerEngine -- Multi-framework regulatory compliance
7. CapitalGoodsPipelineEngine -- Orchestrated 10-stage pipeline

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/calculate` | Execute full pipeline calculation | 200, 400, 500 |
| 2 | POST | `/calculate/batch` | Batch calculation | 200, 400, 500 |
| 3 | GET | `/calculations` | List stored calculations | 200, 500 |
| 4 | GET | `/calculations/{calc_id}` | Get calculation by ID | 200, 404, 500 |
| 5 | DELETE | `/calculations/{calc_id}` | Delete calculation | 200, 404, 500 |
| 6 | POST | `/assets` | Register a capital asset | 200, 400, 500 |
| 7 | GET | `/assets` | List registered assets | 200, 500 |
| 8 | GET | `/emission-factors` | Get emission factors | 200, 500 |
| 9 | POST | `/compliance` | Check regulatory compliance | 200, 400, 500 |
| 10 | GET | `/health` | Service health check | 200 |
| 11 | GET | `/stats` | Service statistics | 200, 500 |

---

## Endpoints

### 1. POST /calculate

Execute the full 10-stage pipeline calculation for capital goods emissions.

**Request Body:**

```json
{
  "organization_id": "org-001",
  "reporting_year": 2025,
  "method": "spend_based",
  "records": [
    {
      "asset_name": "CNC Machine Tool",
      "category": "machinery",
      "sub_category": "industrial_equipment",
      "spend_amount": 250000.0,
      "currency": "USD",
      "naics_code": "333517",
      "supplier_name": "Precision Machining Inc."
    }
  ],
  "eeio_database": "EPA_USEEIO",
  "tenant_id": "tenant-001"
}
```

**Response (200 OK):**

```json
{
  "calculation_id": "cg_abc123",
  "method": "spend_based",
  "total_co2e_kg": 87500.0,
  "records_processed": 1,
  "by_category": {
    "machinery": 87500.0
  },
  "dqi_score": 2.5,
  "provenance_hash": "sha256:c9d0e1f2...",
  "calculated_at": "2026-04-01T10:30:00Z"
}
```

---

### 2. POST /calculate/batch

Execute batch calculation for multiple capital asset records.

---

### 3. GET /calculations

List stored calculations with pagination.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Results per page (max 1000) |
| `offset` | integer | 0 | Offset for pagination |

---

### 6. POST /assets

Register a capital asset for tracking and calculation.

**Request Body:**

```json
{
  "asset_name": "Office Building Renovation",
  "category": "buildings",
  "sub_category": "office",
  "acquisition_date": "2025-06-15",
  "spend_amount": 2000000.0,
  "currency": "USD",
  "useful_life_years": 30,
  "naics_code": "236220",
  "supplier_name": "BuildCorp",
  "supplier_epd_id": "EPD-2025-12345",
  "tenant_id": "tenant-001"
}
```

---

### 8. GET /emission-factors

Get emission factors with optional filtering by source and asset category.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | string | EEIO database: EPA_USEEIO, EXIOBASE, GTAP |
| `category` | string | Asset category filter |

---

### 9. POST /compliance

Check calculation results against regulatory frameworks.

**Request Body:**

```json
{
  "result": { "calculation_id": "cg_abc123", "...": "..." },
  "frameworks": ["ghg_protocol", "csrd_esrs", "cdp", "iso_14064"]
}
```

---

### 10. GET /health

Service health check. Returns status, agent ID, version, and engine availability.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "agent_id": "GL-MRV-S3-002",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "engines": {
    "database": true,
    "spend_based": true,
    "average_data": true,
    "supplier_specific": true,
    "hybrid_aggregator": true,
    "compliance": true,
    "pipeline": true
  }
}
```

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid method, category, or input parameters |
| 404 | Not Found -- Calculation or asset not found |
| 500 | Internal Server Error |

## Notes

The capital goods router is defined in `setup.py` via `get_router()` rather than the standard `api/router.py` pattern. The `api/router.py` file exists but is currently empty/corrupted. All endpoints are functional through the setup module's router factory.
