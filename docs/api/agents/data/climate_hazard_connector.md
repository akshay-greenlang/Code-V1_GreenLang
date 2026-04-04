# Climate Hazard Connector API Reference

**Agent:** AGENT-DATA-020 (GL-DATA-GEO-002)
**Prefix:** `/api/v1/climate-hazard`
**Source:** `greenlang/agents/data/climate_hazard/api/router.py`
**Status:** Production Ready

## Overview

The Climate Hazard Connector registers external climate hazard data sources, ingests and queries hazard data, calculates single and multi-hazard risk indices, projects scenarios under SSP/RCP pathways, registers physical and financial assets, assesses exposure and vulnerability, and generates compliance reports (TCFD, CSRD, EU Taxonomy).

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/sources` | Register hazard source | Yes |
| 2 | GET | `/sources` | List sources | Yes |
| 3 | GET | `/sources/{source_id}` | Get source details | Yes |
| 4 | POST | `/hazard-data/ingest` | Ingest hazard data | Yes |
| 5 | GET | `/hazard-data` | Query hazard data | Yes |
| 6 | GET | `/hazard-data/events` | Historical events | Yes |
| 7 | POST | `/risk-index/calculate` | Calculate risk index | Yes |
| 8 | POST | `/risk-index/multi-hazard` | Multi-hazard index | Yes |
| 9 | POST | `/risk-index/compare` | Compare locations | Yes |
| 10 | POST | `/scenarios/project` | Project scenario | Yes |
| 11 | GET | `/scenarios` | List scenarios | Yes |
| 12 | POST | `/assets` | Register asset | Yes |
| 13 | GET | `/assets` | List assets | Yes |
| 14 | POST | `/exposure/assess` | Assess exposure | Yes |
| 15 | POST | `/exposure/portfolio` | Portfolio exposure | Yes |
| 16 | POST | `/vulnerability/score` | Score vulnerability | Yes |
| 17 | POST | `/reports/generate` | Generate report | Yes |
| 18 | GET | `/reports/{report_id}` | Get report | Yes |
| 19 | POST | `/pipeline/run` | Run pipeline | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Key Endpoints

### 1. Register Hazard Source

```http
POST /api/v1/climate-hazard/sources
```

**Request Body:**

```json
{
  "source_id": "src_copernicus_era5",
  "name": "Copernicus ERA5 Reanalysis",
  "provider": "ECMWF",
  "hazard_types": ["flood", "heat_wave", "drought"],
  "data_format": "NetCDF",
  "update_frequency": "daily",
  "spatial_resolution_km": 25,
  "temporal_range": {"start": "1950-01-01", "end": "2025-12-31"}
}
```

**Response:**

```json
{
  "source_id": "src_copernicus_era5",
  "name": "Copernicus ERA5 Reanalysis",
  "status": "registered",
  "provenance_hash": "sha256:..."
}
```

### 7. Calculate Risk Index

Calculate a composite climate risk index for a location considering multiple hazard types.

```http
POST /api/v1/climate-hazard/risk-index/calculate
```

**Request Body:**

```json
{
  "location_id": "loc_factory_munich",
  "latitude": 48.1351,
  "longitude": 11.5820,
  "hazard_types": ["flood", "heat_wave", "wildfire"],
  "time_horizon": "2050",
  "scenario": "SSP2-4.5",
  "weighting_method": "equal"
}
```

**Response:**

```json
{
  "location_id": "loc_factory_munich",
  "composite_risk_index": 0.42,
  "risk_category": "medium",
  "hazard_scores": {
    "flood": 0.55,
    "heat_wave": 0.38,
    "wildfire": 0.33
  },
  "scenario": "SSP2-4.5",
  "time_horizon": "2050",
  "confidence": 0.82,
  "provenance_hash": "sha256:..."
}
```

### 9. Compare Locations

```http
POST /api/v1/climate-hazard/risk-index/compare
```

**Request Body:**

```json
{
  "locations": [
    {"location_id": "loc_factory_munich", "latitude": 48.1351, "longitude": 11.5820},
    {"location_id": "loc_warehouse_rotterdam", "latitude": 51.9225, "longitude": 4.4792}
  ],
  "hazard_types": ["flood", "heat_wave"],
  "scenario": "SSP2-4.5",
  "time_horizon": "2050"
}
```

### 10. Project Scenario

Project climate hazard under a given SSP or RCP scenario.

```http
POST /api/v1/climate-hazard/scenarios/project
```

**Request Body:**

```json
{
  "location_id": "loc_factory_munich",
  "scenario": "SSP5-8.5",
  "time_horizons": ["2030", "2050", "2100"],
  "hazard_types": ["flood", "heat_wave", "drought"]
}
```

### 14. Assess Exposure

Assess climate hazard exposure for a physical or financial asset.

```http
POST /api/v1/climate-hazard/exposure/assess
```

**Request Body:**

```json
{
  "asset_id": "asset_factory_001",
  "hazard_types": ["flood", "heat_wave"],
  "scenario": "SSP2-4.5",
  "time_horizon": "2050",
  "include_financial_impact": true
}
```

**Response:**

```json
{
  "asset_id": "asset_factory_001",
  "exposure_score": 0.58,
  "exposure_category": "moderate",
  "hazard_exposures": {
    "flood": {"score": 0.72, "annual_loss_usd": 125000},
    "heat_wave": {"score": 0.44, "annual_loss_usd": 45000}
  },
  "total_expected_annual_loss_usd": 170000,
  "provenance_hash": "sha256:..."
}
```

### 15. Portfolio Exposure

Assess climate hazard exposure for an entire asset portfolio.

```http
POST /api/v1/climate-hazard/exposure/portfolio
```

**Request Body:**

```json
{
  "portfolio_id": "portfolio_europe",
  "asset_ids": ["asset_factory_001", "asset_warehouse_002", "asset_office_003"],
  "scenario": "SSP2-4.5",
  "time_horizon": "2050"
}
```

### 16. Score Vulnerability

```http
POST /api/v1/climate-hazard/vulnerability/score
```

**Request Body:**

```json
{
  "asset_id": "asset_factory_001",
  "hazard_type": "flood",
  "adaptive_capacity": 0.6,
  "sensitivity": 0.7,
  "coping_capacity": 0.5
}
```

### 17. Generate Report

Generate a climate hazard compliance report aligned with TCFD, CSRD, or EU Taxonomy.

```http
POST /api/v1/climate-hazard/reports/generate
```

**Request Body:**

```json
{
  "portfolio_id": "portfolio_europe",
  "framework": "tcfd",
  "format": "pdf",
  "include_scenarios": true,
  "include_financial_impact": true
}
```

### 19. Run Full Pipeline

Run the complete climate hazard assessment pipeline end to end.

```http
POST /api/v1/climate-hazard/pipeline/run
```

**Request Body:**

```json
{
  "portfolio_id": "portfolio_europe",
  "hazard_types": ["flood", "heat_wave", "drought", "wildfire"],
  "scenarios": ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"],
  "time_horizons": ["2030", "2050", "2100"],
  "include_vulnerability": true,
  "include_financial_impact": true,
  "generate_report": true,
  "report_framework": "tcfd"
}
```

---

## Query Parameters

List endpoints support the following common filters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hazard_type` | string | null | Filter by hazard type (flood, heat_wave, drought, wildfire, etc.) |
| `source_id` | string | null | Filter by data source |
| `location_id` | string | null | Filter by location |
| `scenario` | string | null | Filter by climate scenario (SSP/RCP) |
| `severity` | string | null | Filter by event severity |
| `asset_type` | string | null | Filter by asset type |
| `limit` | integer | 100 | Maximum results (1-1000) |
| `offset` | integer | 0 | Pagination offset |

---

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid input or parameters |
| 401 | Unauthorized -- invalid or missing JWT |
| 404 | Not Found -- source, report, or asset not found |
| 503 | Service Unavailable -- Climate Hazard service not initialized |
