# Spend Data Categorizer API Reference

**Agent:** AGENT-DATA-009 (GL-DATA-SUP-002)
**Prefix:** `/api/v1/spend-categorizer`
**Source:** `greenlang/agents/data/spend_categorizer/api/router.py`
**Status:** Production Ready

## Overview

The Spend Data Categorizer agent classifies spend records into taxonomy categories (UNSPSC, NAICS, NACE), maps them to GHG Protocol Scope 3 categories (1-15), and calculates emissions using EEIO, EXIOBASE, or DEFRA emission factors. Supports rule-based and ML-assisted classification.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/v1/ingest` | Ingest spend records | Yes |
| 2 | POST | `/v1/ingest/file` | Ingest from file | Yes |
| 3 | GET | `/v1/records` | List spend records | Yes |
| 4 | GET | `/v1/records/{record_id}` | Get single record | Yes |
| 5 | POST | `/v1/classify` | Classify spend record | Yes |
| 6 | POST | `/v1/classify/batch` | Batch classification | Yes |
| 7 | POST | `/v1/map-scope3` | Map to Scope 3 | Yes |
| 8 | POST | `/v1/map-scope3/batch` | Batch Scope 3 mapping | Yes |
| 9 | POST | `/v1/calculate-emissions` | Calculate emissions | Yes |
| 10 | POST | `/v1/calculate-emissions/batch` | Batch emission calculation | Yes |
| 11 | GET | `/v1/emission-factors` | List emission factors | Yes |
| 12 | GET | `/v1/emission-factors/{taxonomy_code}` | Get factor by code | Yes |
| 13 | POST | `/v1/rules` | Create classification rule | Yes |
| 14 | GET | `/v1/rules` | List rules | Yes |
| 15 | PUT | `/v1/rules/{rule_id}` | Update rule | Yes |
| 16 | DELETE | `/v1/rules/{rule_id}` | Delete rule | Yes |
| 17 | GET | `/v1/analytics` | Get analytics summary | Yes |
| 18 | GET | `/v1/analytics/hotspots` | Get emission hotspots | Yes |
| 19 | POST | `/v1/reports` | Generate report | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Key Endpoints

### 5. Classify Spend Record

```http
POST /api/v1/spend-categorizer/v1/classify
```

**Request Body:**

```json
{
  "record_id": "rec_abc123",
  "taxonomy_system": "unspsc"
}
```

**Response:**

```json
{
  "record_id": "rec_abc123",
  "taxonomy_system": "unspsc",
  "taxonomy_code": "15101500",
  "taxonomy_description": "Fuel oils",
  "confidence": 0.94,
  "classification_method": "rule_based",
  "provenance_hash": "sha256:..."
}
```

### 9. Calculate Emissions

```http
POST /api/v1/spend-categorizer/v1/calculate-emissions
```

**Request Body:**

```json
{
  "record_id": "rec_abc123",
  "factor_source": "eeio"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `record_id` | string | Yes | Spend record identifier |
| `factor_source` | string | Optional | Emission factor source: `eeio`, `exiobase`, `defra` |

### 18. Get Emission Hotspots

```http
GET /api/v1/spend-categorizer/v1/analytics/hotspots?top_n=10
```

**Response:**

```json
{
  "hotspots": [
    {
      "taxonomy_code": "15101500",
      "description": "Fuel oils",
      "total_spend": 2500000.00,
      "total_emissions_tco2e": 850.5,
      "emission_intensity": 0.00034,
      "record_count": 125
    }
  ],
  "count": 10
}
```
