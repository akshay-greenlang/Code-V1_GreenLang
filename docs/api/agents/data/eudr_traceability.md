# EUDR Traceability Connector API Reference

**Agent:** AGENT-DATA-005 (GL-DATA-EUDR-001)
**Prefix:** `/v1/eudr`
**Source:** `greenlang/agents/data/eudr_traceability/api/router.py`
**Status:** Production Ready

## Overview

The EUDR Traceability Connector agent provides 20 REST API endpoints for EU Deforestation Regulation compliance operations. Capabilities include production plot management with WGS84 geolocation (Article 9), chain of custody transfers and batch tracing, Due Diligence Statement generation and submission to the EU Information System (Articles 4 and 12), risk assessment per Article 10, commodity classification by CN/HS code, supplier compliance declarations, and compliance summary reporting.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | GET | `/health` | Service health check | No |
| 2 | POST | `/plots` | Register production plot | Yes |
| 3 | GET | `/plots` | List production plots | Yes |
| 4 | GET | `/plots/{plot_id}` | Get plot details | Yes |
| 5 | PUT | `/plots/{plot_id}/compliance` | Update plot compliance status | Yes |
| 6 | DELETE | `/plots/{plot_id}` | Archive plot (soft delete) | Yes |
| 7 | POST | `/custody/transfers` | Record custody transfer | Yes |
| 8 | GET | `/custody/transfers` | List custody transfers | Yes |
| 9 | GET | `/custody/trace/{batch_id}` | Trace batch to origin | Yes |
| 10 | POST | `/custody/batches/split` | Split batch | Yes |
| 11 | POST | `/dds` | Generate Due Diligence Statement | Yes |
| 12 | GET | `/dds` | List Due Diligence Statements | Yes |
| 13 | GET | `/dds/{dds_id}` | Get DDS details | Yes |
| 14 | POST | `/dds/{dds_id}/submit` | Submit DDS to EU Information System | Yes |
| 15 | POST | `/risk/assess` | Perform risk assessment | Yes |
| 16 | GET | `/risk/countries` | Get country risk classifications | Yes |
| 17 | POST | `/commodities/classify` | Classify commodity by CN/HS code | Yes |
| 18 | POST | `/suppliers/declarations` | Register supplier declaration | Yes |
| 19 | GET | `/suppliers/declarations` | List supplier declarations | Yes |
| 20 | GET | `/compliance/summary` | Get compliance summary report | Yes |
| 21 | GET | `/statistics` | Get service statistics | Yes |

---

## Key Endpoints

### 2. Register Production Plot

Register a production plot with WGS84 geolocation coordinates, commodity type, and producer information per EUDR Article 9.

```http
POST /v1/eudr/plots
```

**Request Body:**

```json
{
  "commodity": "palm_oil",
  "country_code": "IDN",
  "producer_name": "Green Harvest Ltd",
  "coordinates": {
    "type": "Polygon",
    "coordinates": [[[107.5, -6.9], [107.6, -6.9], [107.6, -6.8], [107.5, -6.8], [107.5, -6.9]]]
  },
  "area_hectares": 50.0,
  "legal_documents": ["land_title_2024.pdf"]
}
```

**Response (200):**

```json
{
  "plot_id": "PLOT-A1B2C3D4E5F6",
  "commodity": "palm_oil",
  "country_code": "IDN",
  "risk_level": "standard",
  "deforestation_free": false,
  "legal_compliance": false,
  "created_at": "2026-04-04T10:30:00Z"
}
```

**Status Codes:** `200` Success | `400` Validation error | `500` Server error

---

### 3. List Production Plots

```http
GET /v1/eudr/plots?commodity=palm_oil&country=IDN&risk_level=high&limit=100&offset=0
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `commodity` | string | - | Filter by EUDR commodity |
| `country` | string | - | Filter by ISO country code |
| `risk_level` | string | - | Filter by risk level: `low`, `standard`, `high` |
| `limit` | integer | 100 | Max results (1-1000) |
| `offset` | integer | 0 | Pagination offset |

**Response (200):**

```json
[
  {
    "plot_id": "PLOT-A1B2C3D4E5F6",
    "commodity": "palm_oil",
    "country_code": "IDN",
    "risk_level": "standard",
    "deforestation_free": true,
    "legal_compliance": true,
    "created_at": "2026-04-04T10:30:00Z"
  }
]
```

---

### 5. Update Plot Compliance Status

```http
PUT /v1/eudr/plots/{plot_id}/compliance
```

**Request Body:**

```json
{
  "deforestation_free": true,
  "legal_compliance": true,
  "supporting_documents": ["satellite_report_2026.pdf", "legal_cert_2026.pdf"]
}
```

**Response (200):** Updated plot record.

**Status Codes:** `200` Success | `400` Validation error | `404` Plot not found

---

### 7. Record Custody Transfer

Record a chain of custody transfer between operators, linking commodities to their origin plots.

```http
POST /v1/eudr/custody/transfers
```

**Request Body:**

```json
{
  "source_operator_id": "OP-001",
  "target_operator_id": "OP-002",
  "commodity": "palm_oil",
  "quantity": "25000",
  "unit": "kg",
  "batch_id": "BATCH-2026-001",
  "custody_model": "identity_preserved"
}
```

**Response (200):**

```json
{
  "transfer_id": "TRF-ABC123DEF456",
  "transaction_id": "TXN-2026-001",
  "commodity": "palm_oil",
  "quantity": "25000",
  "custody_model": "identity_preserved",
  "verification_status": "pending_verification"
}
```

---

### 9. Trace Batch to Origin

Trace a product batch back through the chain of custody to its origin production plots.

```http
GET /v1/eudr/custody/trace/{batch_id}
```

**Response (200):** Array of origin plot records.

---

### 11. Generate Due Diligence Statement

Generate a Due Diligence Statement per EUDR Article 4.

```http
POST /v1/eudr/dds
```

**Request Body:**

```json
{
  "commodity": "soya",
  "operator_id": "OP-003",
  "plot_ids": ["PLOT-001", "PLOT-002"],
  "product_description": "Soya beans, whole",
  "quantity": "50000",
  "unit": "kg"
}
```

**Response (200):**

```json
{
  "dds_id": "DDS-XYZ789",
  "commodity": "soya",
  "status": "draft",
  "risk_level": "standard",
  "origin_countries": ["BRA"],
  "eu_reference_number": null
}
```

---

### 14. Submit DDS to EU Information System

Submit a completed Due Diligence Statement to the EU Information System per Article 12.

```http
POST /v1/eudr/dds/{dds_id}/submit
```

**Response (200):**

```json
{
  "dds_id": "DDS-XYZ789",
  "commodity": "soya",
  "status": "submitted",
  "risk_level": "standard",
  "origin_countries": ["BRA"],
  "eu_reference_number": "EU-2026-00012345"
}
```

**Status Codes:** `200` Success | `400` DDS not ready for submission | `404` DDS not found | `500` Server error

---

### 15. Perform Risk Assessment

Perform a risk assessment for a product, plot, or operator per Article 10.

```http
POST /v1/eudr/risk/assess
```

**Request Body:**

```json
{
  "entity_type": "plot",
  "entity_id": "PLOT-001",
  "commodity": "palm_oil",
  "country_code": "IDN"
}
```

**Response (200):**

```json
{
  "assessment_id": "RISK-ABC123",
  "overall_risk_score": 0.35,
  "risk_level": "standard",
  "risk_factors": ["country_risk_moderate", "commodity_high_deforestation_risk"]
}
```

---

### 17. Classify Commodity

Classify a product by CN/HS code to determine EUDR coverage.

```http
POST /v1/eudr/commodities/classify
```

**Request Body:**

```json
{
  "product_name": "Chocolate bars",
  "cn_code": "18063100",
  "hs_code": "1806.31"
}
```

**Response (200):**

```json
{
  "classification_id": "CLS-001",
  "product_name": "Chocolate bars",
  "commodity": "cocoa",
  "is_derived_product": true,
  "primary_commodity": "cocoa"
}
```

---

### 20. Get Compliance Summary

```http
GET /v1/eudr/compliance/summary
```

**Response (200):** Compliance statistics across all tracked entities including pass/fail counts and compliance score.

---

## Error Responses

All error responses follow a standard format:

```json
{
  "detail": "Descriptive error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error or malformed input |
| 404 | Not Found -- entity does not exist |
| 500 | Internal Server Error |
