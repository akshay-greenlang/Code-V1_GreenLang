# Data Lineage Tracker API Reference

**Agent:** AGENT-DATA-018 (GL-DATA-X-021)
**Prefix:** `/api/v1/data-lineage`
**Source:** `greenlang/agents/data/data_lineage_tracker/api/router.py`
**Status:** Production Ready

## Overview

The Data Lineage Tracker registers data assets, records transformation events, builds a directed lineage graph, and supports backward/forward traversal, impact analysis, and lineage validation. It provides full provenance tracking across the GreenLang data pipeline.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/assets` | Register a data asset | Yes |
| 2 | GET | `/assets` | List assets | Yes |
| 3 | GET | `/assets/{asset_id}` | Get asset details | Yes |
| 4 | PUT | `/assets/{asset_id}` | Update asset metadata | Yes |
| 5 | DELETE | `/assets/{asset_id}` | Deregister asset (soft delete) | Yes |
| 6 | POST | `/transformations` | Record transformation event | Yes |
| 7 | GET | `/transformations` | List transformation events | Yes |
| 8 | POST | `/edges` | Create a lineage edge | Yes |
| 9 | GET | `/edges` | List lineage edges | Yes |
| 10 | GET | `/graph` | Get the full lineage graph | Yes |
| 11 | GET | `/graph/subgraph/{asset_id}` | Extract subgraph | Yes |
| 12 | GET | `/backward/{asset_id}` | Backward lineage traversal | Yes |
| 13 | GET | `/forward/{asset_id}` | Forward lineage traversal | Yes |
| 14 | POST | `/impact` | Run impact analysis | Yes |
| 15 | POST | `/validate` | Validate lineage completeness | Yes |
| 16 | GET | `/validate/{validation_id}` | Get validation result | Yes |
| 17 | POST | `/reports` | Generate lineage report | Yes |
| 18 | POST | `/pipeline` | Run full lineage tracking pipeline | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Register a Data Asset

```http
POST /api/v1/data-lineage/assets
```

**Request Body:**

```json
{
  "asset_id": "asset_emissions_csv",
  "name": "Q1 2026 Emissions Data",
  "asset_type": "dataset",
  "owner": "data-engineering",
  "classification": "internal",
  "description": "Quarterly facility emissions data"
}
```

### 12. Backward Lineage Traversal

Trace the full upstream data provenance chain for an asset back to its original sources.

```http
GET /api/v1/data-lineage/backward/{asset_id}?max_depth=5
```

**Response:**

```json
{
  "asset_id": "asset_ghg_report",
  "direction": "backward",
  "depth": 3,
  "lineage": [
    {"asset_id": "asset_ghg_report", "depth": 0},
    {"asset_id": "asset_normalized_data", "depth": 1, "transformation": "normalization"},
    {"asset_id": "asset_raw_csv", "depth": 2, "transformation": "ingestion"},
    {"asset_id": "asset_erp_extract", "depth": 3, "transformation": "erp_sync"}
  ]
}
```

### 14. Run Impact Analysis

Analyze what downstream assets would be affected if a source asset changes.

```http
POST /api/v1/data-lineage/impact
```

**Request Body:**

```json
{
  "asset_id": "asset_emission_factors",
  "direction": "forward",
  "max_depth": 10
}
```

**Response:**

```json
{
  "asset_id": "asset_emission_factors",
  "direction": "forward",
  "impacted_assets": 15,
  "critical_paths": [
    ["asset_emission_factors", "asset_scope1_calc", "asset_ghg_report"],
    ["asset_emission_factors", "asset_scope2_calc", "asset_ghg_report"]
  ]
}
```
