# Scope 3 Category Mapper Agent API Reference (AGENT-MRV-029)

## Overview

The Scope 3 Category Mapper Agent (GL-MRV-S3-CM) is a cross-cutting MRV agent that classifies business activity records into the appropriate Scope 3 category (Categories 1-15) and routes them to the corresponding category-specific calculation agents. Supports classification by NAICS, ISIC, and UNSPSC codes; spend data, purchase orders, and bill of materials records; boundary determination; completeness screening; double-counting detection; and multi-framework compliance assessment.

**API Prefix:** `/api/v1/scope3-category-mapper`
**Agent ID:** AGENT-MRV-029
**Status:** Production Ready

**Engines (7):**
1. CategoryDatabaseEngine -- Code mapping lookups (NAICS, ISIC, UNSPSC)
2. SpendClassifierEngine -- Deterministic spend classification
3. ActivityRouterEngine -- Category agent routing (routes to MRV-014 through MRV-028)
4. BoundaryDeterminerEngine -- Upstream/downstream boundary determination
5. CompletenessScreenerEngine -- Completeness analysis and gap detection
6. ComplianceCheckerEngine -- Multi-framework compliance assessment
7. CategoryMapperPipelineEngine -- 10-stage orchestrated pipeline

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/classify` | Classify single record | 200, 400, 500 |
| 2 | POST | `/classify/batch` | Batch classify (up to 50K records) | 200, 400, 500 |
| 3 | POST | `/classify/spend` | Classify spend data | 200, 400, 500 |
| 4 | POST | `/classify/purchase-orders` | Classify purchase orders | 200, 400, 500 |
| 5 | POST | `/classify/bom` | Classify bill of materials | 200, 400, 500 |
| 6 | POST | `/route` | Route classified records to agents | 200, 400, 500 |
| 7 | POST | `/route/dry-run` | Preview routing without execution | 200, 400, 500 |
| 8 | POST | `/boundary/determine` | Determine upstream/downstream boundary | 200, 400, 500 |
| 9 | POST | `/completeness/screen` | Screen category completeness | 200, 400, 500 |
| 10 | POST | `/completeness/gap-analysis` | Detailed gap analysis | 200, 400, 500 |
| 11 | POST | `/double-counting/check` | Check for cross-category double counting | 200, 400, 500 |
| 12 | POST | `/compliance/assess` | Assess mapping compliance | 200, 400, 500 |
| 13 | GET | `/categories` | List all 15 Scope 3 categories | 200 |
| 14 | GET | `/categories/{number}` | Get category details | 200, 404 |
| 15 | GET | `/codes/naics/{code}` | Look up NAICS code mapping | 200, 404 |
| 16 | GET | `/codes/isic/{code}` | Look up ISIC code mapping | 200, 404 |
| 17 | GET | `/codes/unspsc/{code}` | Look up UNSPSC code mapping | 200, 404 |
| 18 | GET | `/health` | Service health check | 200 |
| 19 | GET | `/metrics` | Prometheus metrics summary | 200 |

---

## Endpoints

### 1. POST /classify

Classify a single business activity record into a Scope 3 category.

**Request Body:**

```json
{
  "record": {
    "naics_code": "481",
    "description": "Air transportation services",
    "amount": 50000.0,
    "currency": "USD",
    "supplier_name": "AirFreight Corp"
  },
  "source_type": "spend",
  "organization_id": "org-001",
  "reporting_year": 2025
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "result": {
    "category_number": 6,
    "category_name": "Business Travel",
    "confidence": 0.95,
    "classification_method": "naics_lookup",
    "boundary": "upstream",
    "record_id": "rec_abc123",
    "naics_code": "481",
    "double_counting_flags": []
  },
  "processing_time_ms": 12.5,
  "error": null
}
```

---

### 2. POST /classify/batch

Classify up to 50,000 records in a single batch request.

**Request Body:**

```json
{
  "records": [
    { "naics_code": "481", "amount": 50000.0, "description": "Air travel" },
    { "naics_code": "324110", "amount": 200000.0, "description": "Fuel purchases" },
    { "isic_code": "4520", "amount": 30000.0, "description": "Vehicle maintenance" }
  ],
  "source_type": "spend",
  "organization_id": "org-001",
  "reporting_year": 2025,
  "company_type": "manufacturing"
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "results": [
    { "category_number": 6, "category_name": "Business Travel", "confidence": 0.95 },
    { "category_number": 1, "category_name": "Purchased Goods and Services", "confidence": 0.88 },
    { "category_number": 2, "category_name": "Capital Goods", "confidence": 0.72 }
  ],
  "summary": {
    "total_records": 3,
    "classified": 3,
    "unclassified": 0,
    "by_category": { "1": 1, "2": 1, "6": 1 }
  },
  "processing_time_ms": 45.2,
  "error": null
}
```

---

### 3. POST /classify/spend

Classify spend data specifically. Automatically sets `source_type` to `"spend"`.

---

### 4. POST /classify/purchase-orders

Classify purchase order records. Automatically sets `source_type` to `"purchase_order"`.

---

### 5. POST /classify/bom

Classify bill of materials records. Automatically sets `source_type` to `"bom"`.

---

### 6. POST /route

Route classified records to the appropriate Scope 3 category-specific agents (MRV-014 through MRV-028).

**Request Body:**

```json
{
  "results": [
    { "category_number": 6, "record_id": "rec_001", "record": { "...": "..." } },
    { "category_number": 1, "record_id": "rec_002", "record": { "...": "..." } }
  ],
  "dry_run": false
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "routing_plan": [
    {
      "record_id": "rec_001",
      "target_agent": "GL-MRV-S3-006",
      "category_number": 6,
      "status": "routed"
    },
    {
      "record_id": "rec_002",
      "target_agent": "GL-MRV-S3-001",
      "category_number": 1,
      "status": "routed"
    }
  ],
  "error": null
}
```

---

### 7. POST /route/dry-run

Preview routing plan without actually dispatching records to category agents.

---

### 8. POST /boundary/determine

Determine whether a record falls within the upstream or downstream boundary of the reporting organization.

**Request Body:**

```json
{
  "record": {
    "naics_code": "481",
    "description": "Air freight services",
    "supplier_type": "third_party"
  },
  "organization_context": {
    "consolidation_approach": "operational_control",
    "industry": "manufacturing"
  }
}
```

---

### 9. POST /completeness/screen

Screen Scope 3 completeness across all 15 categories. Identifies reported vs unreported categories and estimates materiality.

**Request Body:**

```json
{
  "company_type": "manufacturing",
  "categories_reported": [1, 2, 3, 4, 5, 6, 7],
  "data_by_category": {
    "1": { "method": "spend_based", "coverage_pct": 85 },
    "2": { "method": "average_data", "coverage_pct": 70 }
  }
}
```

---

### 10. POST /completeness/gap-analysis

Detailed gap analysis with recommendations for improving coverage of unreported categories.

---

### 11. POST /double-counting/check

Check for cross-category double counting in a set of classified records.

**Response (200 OK):**

```json
{
  "success": true,
  "total_checked": 100,
  "double_counting_found": 3,
  "details": [
    {
      "record_id": "rec_045",
      "flags": ["DC-001: Fuel in Cat 1 and Cat 3"]
    }
  ]
}
```

---

### 12. POST /compliance/assess

Assess classification and mapping compliance against a specific regulatory framework.

**Request Body:**

```json
{
  "framework": "ghg_protocol",
  "company_type": "manufacturing",
  "categories_reported": [1, 2, 3, 4, 5, 6, 7]
}
```

---

### 13. GET /categories

List all 15 Scope 3 categories with descriptions, examples, and typical calculation methods.

---

### 14. GET /categories/{number}

Get detailed information for a specific Scope 3 category (1-15).

---

### 15-17. GET /codes/{system}/{code}

Look up classification code mappings for NAICS, ISIC, or UNSPSC codes. Returns the mapped Scope 3 category, description, and confidence level.

---

### 18. GET /health

Service health check with per-engine availability status.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agent_id": "AGENT-MRV-029",
  "uptime_seconds": 3600.5,
  "engines_status": {
    "category_database": true,
    "spend_classifier": true,
    "activity_router": true,
    "boundary_determiner": true,
    "completeness_screener": true,
    "compliance_checker": true,
    "pipeline": true
  }
}
```

---

### 19. GET /metrics

Prometheus metrics summary for the category mapper service.

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- Invalid record format, missing required fields |
| 404 | Not Found -- Category number or code not found |
| 500 | Internal Server Error |

## Notes

This agent does not have a standard `api/router.py` file. The REST API is defined in `setup.py` via the `get_router()` function and mounted at `/api/v1/scope3-category-mapper` using `app.include_router(get_router(), prefix="/api/v1/scope3-category-mapper")`.
