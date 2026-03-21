# PACK-028 Sector Pathway Pack -- API Reference

**Pack ID:** PACK-028-sector-pathway
**Version:** 1.0.0
**API Version:** v1
**Base URL:** `https://api.greenlang.io/v1/packs/028`
**Authentication:** JWT Bearer Token (SEC-001)
**Content Type:** `application/json`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Common Types](#common-types)
3. [Error Handling](#error-handling)
4. [Rate Limits](#rate-limits)
5. [Engines API](#engines-api)
   - [Sector Classification Engine](#sector-classification-engine)
   - [Intensity Calculator Engine](#intensity-calculator-engine)
   - [Pathway Generator Engine](#pathway-generator-engine)
   - [Convergence Analyzer Engine](#convergence-analyzer-engine)
   - [Technology Roadmap Engine](#technology-roadmap-engine)
   - [Abatement Waterfall Engine](#abatement-waterfall-engine)
   - [Sector Benchmark Engine](#sector-benchmark-engine)
   - [Scenario Comparison Engine](#scenario-comparison-engine)
6. [Workflows API](#workflows-api)
   - [Sector Pathway Design Workflow](#sector-pathway-design-workflow)
   - [Pathway Validation Workflow](#pathway-validation-workflow)
   - [Technology Planning Workflow](#technology-planning-workflow)
   - [Progress Monitoring Workflow](#progress-monitoring-workflow)
   - [Multi-Scenario Analysis Workflow](#multi-scenario-analysis-workflow)
   - [Full Sector Assessment Workflow](#full-sector-assessment-workflow)
7. [Templates API](#templates-api)
   - [Sector Pathway Report](#sector-pathway-report)
   - [Intensity Convergence Report](#intensity-convergence-report)
   - [Technology Roadmap Report](#technology-roadmap-report)
   - [Abatement Waterfall Report](#abatement-waterfall-report)
   - [Sector Benchmark Report](#sector-benchmark-report)
   - [Scenario Comparison Report](#scenario-comparison-report)
   - [SBTi Validation Report](#sbti-validation-report)
   - [Sector Strategy Report](#sector-strategy-report)
8. [Integrations API](#integrations-api)
   - [Pack Orchestrator](#pack-orchestrator)
   - [SBTi SDA Bridge](#sbti-sda-bridge)
   - [IEA NZE Bridge](#iea-nze-bridge)
   - [IPCC AR6 Bridge](#ipcc-ar6-bridge)
   - [PACK-021 Bridge](#pack-021-bridge)
   - [MRV Bridge](#mrv-bridge)
   - [Decarbonization Bridge](#decarb-bridge)
   - [Data Bridge](#data-bridge)
   - [Health Check](#health-check)
   - [Setup Wizard](#setup-wizard)
9. [Webhooks](#webhooks)
10. [SDK Reference](#sdk-reference)

---

## Authentication

All API requests require a valid JWT token in the `Authorization` header.

```http
Authorization: Bearer <jwt_token>
```

### Obtaining a Token

```http
POST /v1/auth/token
Content-Type: application/json

{
  "username": "sector_analyst@company.com",
  "password": "********"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2g..."
}
```

### Required Permissions

| Endpoint Category | Required Role | Permission |
|-------------------|--------------|------------|
| Engine execution | `pathway_designer` or higher | `pack028:engine:execute` |
| Workflow execution | `pathway_designer` or higher | `pack028:workflow:execute` |
| Report generation | `sector_analyst` or higher | `pack028:template:render` |
| Configuration | `sector_pathway_admin` | `pack028:config:write` |
| Read-only access | `sector_analyst` or higher | `pack028:read` |
| Health check | Any authenticated user | `pack028:health:read` |

---

## Common Types

### Sector Enum

```typescript
enum Sector {
  POWER_GENERATION = "power_generation"
  STEEL = "steel"
  CEMENT = "cement"
  ALUMINUM = "aluminum"
  PULP_PAPER = "pulp_paper"
  CHEMICALS = "chemicals"
  AVIATION = "aviation"
  SHIPPING = "shipping"
  ROAD_TRANSPORT = "road_transport"
  RAIL = "rail"
  BUILDINGS_RESIDENTIAL = "buildings_residential"
  BUILDINGS_COMMERCIAL = "buildings_commercial"
  AGRICULTURE = "agriculture"
  FOOD_BEVERAGE = "food_beverage"
  OIL_GAS = "oil_gas"
  CROSS_SECTOR = "cross_sector"
}
```

### Scenario Enum

```typescript
enum Scenario {
  NZE_15C = "nze_15c"          // Net Zero Emissions by 2050, 1.5C
  WB2C = "wb2c"                // Well-Below 2C
  TWO_C = "2c"                 // 2 Degrees Celsius
  APS = "aps"                  // Announced Pledges Scenario
  STEPS = "steps"              // Stated Policies Scenario
}
```

### Convergence Model Enum

```typescript
enum ConvergenceModel {
  LINEAR = "linear"
  EXPONENTIAL = "exponential"
  S_CURVE = "s_curve"
  STEPPED = "stepped"
}
```

### Region Enum

```typescript
enum Region {
  GLOBAL = "global"
  OECD = "oecd"
  EMERGING_MARKETS = "emerging_markets"
  EU = "eu"
  NORTH_AMERICA = "north_america"
  ASIA_PACIFIC = "asia_pacific"
}
```

### Risk Level Enum

```typescript
enum RiskLevel {
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"
  CRITICAL = "critical"
}
```

### IntensityMetric Type

```typescript
interface IntensityMetric {
  metric_id: string           // e.g., "STL-01"
  name: string               // e.g., "Crude steel intensity (BF-BOF)"
  unit: string               // e.g., "tCO2e/tonne crude steel"
  value: number              // e.g., 1.85
  year: number               // e.g., 2023
  data_quality_score: number // 1.0-5.0
}
```

### PathwayPoint Type

```typescript
interface PathwayPoint {
  year: number
  intensity: number
  absolute_emissions: number
  production_volume: number
  cumulative_reduction: number
  convergence_pct: number
}
```

### ProvenanceHash Type

```typescript
interface ProvenanceHash {
  algorithm: "sha256"
  hash: string               // hex-encoded SHA-256
  timestamp: string          // ISO 8601
  inputs_hash: string        // SHA-256 of inputs
  calculation_version: string
}
```

### Pagination

All list endpoints support pagination:

```http
GET /v1/packs/028/pathways?page=1&page_size=20
```

**Response includes pagination metadata:**

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 156,
    "total_pages": 8,
    "has_next": true,
    "has_previous": false
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "SECTOR_NOT_FOUND",
    "message": "The specified sector 'plastics' is not supported.",
    "details": {
      "supported_sectors": ["power_generation", "steel", "cement", "..."],
      "suggestion": "Use 'chemicals' for plastics manufacturers."
    },
    "request_id": "req_abc123def456",
    "timestamp": "2026-03-19T10:30:00Z"
  }
}
```

### Error Codes

| HTTP Status | Error Code | Description |
|-------------|-----------|-------------|
| 400 | `INVALID_INPUT` | Request body validation failed |
| 400 | `INVALID_SECTOR` | Unrecognized sector identifier |
| 400 | `INVALID_SCENARIO` | Unrecognized scenario identifier |
| 400 | `INVALID_YEAR_RANGE` | Base year or target year out of range |
| 400 | `INVALID_INTENSITY` | Intensity value out of sector-valid range |
| 400 | `MISSING_REQUIRED_FIELD` | Required field not provided |
| 401 | `UNAUTHORIZED` | Missing or invalid JWT token |
| 403 | `FORBIDDEN` | Insufficient permissions for operation |
| 404 | `PATHWAY_NOT_FOUND` | Referenced pathway does not exist |
| 404 | `SECTOR_NOT_FOUND` | Referenced sector not in supported list |
| 409 | `PATHWAY_ALREADY_EXISTS` | Pathway for sector/scenario already exists |
| 422 | `VALIDATION_FAILED` | Input data fails business rules validation |
| 422 | `SDA_ELIGIBILITY_FAILED` | Company not eligible for SDA in sector |
| 422 | `CONVERGENCE_INFEASIBLE` | Pathway convergence mathematically infeasible |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Unexpected server error |
| 503 | `SERVICE_UNAVAILABLE` | Pack temporarily unavailable |

### Validation Error Details

```json
{
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Input validation failed with 2 errors.",
    "details": {
      "validation_errors": [
        {
          "field": "base_year_intensity",
          "value": -0.5,
          "constraint": "Intensity must be a positive number.",
          "suggestion": "Provide a positive intensity value (e.g., 1.85 for steel)."
        },
        {
          "field": "target_year_near",
          "value": 2024,
          "constraint": "Near-term target must be 5-10 years from base year.",
          "suggestion": "Set target year between 2028 and 2033 for base year 2023."
        }
      ]
    }
  }
}
```

---

## Rate Limits

| Endpoint Category | Rate Limit | Burst | Window |
|-------------------|-----------|-------|--------|
| Engine execution | 60 requests/min | 10 | 1 minute |
| Workflow execution | 10 requests/min | 3 | 1 minute |
| Report generation | 20 requests/min | 5 | 1 minute |
| Read-only queries | 120 requests/min | 20 | 1 minute |
| Health check | 30 requests/min | 5 | 1 minute |

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1711036800
```

---

## Engines API

### Sector Classification Engine

Classifies a company into one of 16 supported sectors based on industry classification codes (NACE, GICS, ISIC) and revenue breakdown.

#### `POST /v1/packs/028/engines/sector-classification/classify`

**Request Body:**

```json
{
  "company_profile": {
    "name": "SteelCorp International",
    "nace_codes": ["C24.10"],
    "gics_code": "15104020",
    "isic_code": "2410",
    "revenue_breakdown": {
      "integrated_steel": 0.75,
      "eaf_steel": 0.20,
      "other": 0.05
    },
    "primary_products": ["hot_rolled_coil", "cold_rolled_coil", "rebar"],
    "country": "DE"
  }
}
```

**Response (200 OK):**

```json
{
  "data": {
    "primary_sector": "steel",
    "sector_display_name": "Steel",
    "sub_sectors": ["integrated_steel", "eaf_steel"],
    "sda_eligible": true,
    "sda_methodology": "SDA-Steel",
    "intensity_metric": {
      "metric_id": "STL-01",
      "name": "Crude steel intensity",
      "unit": "tCO2e/tonne crude steel"
    },
    "iea_chapter": "Chapter 5: Industry (Steel)",
    "classification_codes": {
      "nace": "C24.10",
      "gics": "15104020",
      "isic": "2410"
    },
    "confidence_score": 0.98,
    "classification_trace": [
      "NACE C24.10 -> Steel sector (match)",
      "GICS 15104020 -> Steel (match)",
      "Revenue: 95% steel-related (>50% threshold)",
      "Final classification: Steel (high confidence)"
    ]
  },
  "metadata": {
    "engine": "sector_classification_engine",
    "version": "1.0.0",
    "processing_time_ms": 124,
    "provenance": {
      "algorithm": "sha256",
      "hash": "a1b2c3d4e5f6...",
      "timestamp": "2026-03-19T10:00:00Z"
    }
  }
}
```

#### `GET /v1/packs/028/engines/sector-classification/sectors`

Returns all supported sectors with their metadata.

**Response (200 OK):**

```json
{
  "data": {
    "sectors": [
      {
        "id": "power_generation",
        "name": "Power Generation",
        "sda_eligible": true,
        "sda_id": "SDA-Power",
        "intensity_metric": "gCO2/kWh",
        "iea_chapter": "Chapter 3: Electricity",
        "nace_codes": ["D35.11"],
        "gics_codes": ["55101010"],
        "isic_codes": ["3510"]
      },
      {
        "id": "steel",
        "name": "Steel",
        "sda_eligible": true,
        "sda_id": "SDA-Steel",
        "intensity_metric": "tCO2e/tonne crude steel",
        "iea_chapter": "Chapter 5: Industry (Steel)",
        "nace_codes": ["C24.10"],
        "gics_codes": ["15104020"],
        "isic_codes": ["2410"]
      }
    ],
    "total_sectors": 16,
    "sda_sectors": 12,
    "extended_sectors": 4
  }
}
```

#### `GET /v1/packs/028/engines/sector-classification/nace-map`

Returns the full NACE-to-sector mapping table.

**Response (200 OK):**

```json
{
  "data": {
    "mappings": [
      {
        "nace_code": "D35.11",
        "nace_description": "Production of electricity",
        "sector": "power_generation",
        "sda_eligible": true
      },
      {
        "nace_code": "C24.10",
        "nace_description": "Manufacture of basic iron and steel",
        "sector": "steel",
        "sda_eligible": true
      }
    ],
    "total_mappings": 48
  }
}
```

---

### Intensity Calculator Engine

Calculates sector-specific intensity metrics from activity data and emissions.

#### `POST /v1/packs/028/engines/intensity-calculator/calculate`

**Request Body:**

```json
{
  "sector": "steel",
  "activity_data": {
    "crude_steel_production_tonnes": 5000000,
    "bf_bof_production_tonnes": 3750000,
    "eaf_production_tonnes": 1000000,
    "dri_production_tonnes": 250000
  },
  "emissions_data": {
    "scope1_tco2e": 7500000,
    "scope2_location_tco2e": 1500000,
    "scope2_market_tco2e": 1200000,
    "process_emissions_tco2e": 3200000,
    "combustion_emissions_tco2e": 4300000
  },
  "reporting_year": 2023,
  "region": "eu"
}
```

**Response (200 OK):**

```json
{
  "data": {
    "primary_intensity": {
      "metric_id": "STL-01",
      "name": "Crude steel intensity (overall)",
      "value": 1.80,
      "unit": "tCO2e/tonne crude steel",
      "year": 2023,
      "data_quality_score": 1.5
    },
    "sub_intensities": [
      {
        "metric_id": "STL-01a",
        "name": "BF-BOF intensity",
        "value": 2.13,
        "unit": "tCO2e/tonne crude steel",
        "year": 2023,
        "production_share": 0.75
      },
      {
        "metric_id": "STL-02",
        "name": "EAF intensity",
        "value": 0.40,
        "unit": "tCO2e/tonne crude steel",
        "year": 2023,
        "production_share": 0.20
      },
      {
        "metric_id": "STL-03",
        "name": "DRI intensity",
        "value": 1.20,
        "unit": "tCO2e/tonne DRI",
        "year": 2023,
        "production_share": 0.05
      }
    ],
    "trend": {
      "direction": "decreasing",
      "annual_change_pct": -2.8,
      "five_year_trend": "improving",
      "data_points": [
        {"year": 2019, "intensity": 1.95},
        {"year": 2020, "intensity": 1.92},
        {"year": 2021, "intensity": 1.88},
        {"year": 2022, "intensity": 1.85},
        {"year": 2023, "intensity": 1.80}
      ]
    },
    "sector_context": {
      "global_average": 1.85,
      "sector_leader_p10": 0.95,
      "company_percentile": 42,
      "above_below_average": "below_average"
    }
  },
  "metadata": {
    "engine": "intensity_calculator_engine",
    "version": "1.0.0",
    "processing_time_ms": 342,
    "provenance": {
      "algorithm": "sha256",
      "hash": "b2c3d4e5f6a7...",
      "timestamp": "2026-03-19T10:01:00Z"
    }
  }
}
```

#### `GET /v1/packs/028/engines/intensity-calculator/metrics`

Returns all supported intensity metrics across sectors.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sector` | string | No | Filter metrics by sector |

**Response (200 OK):**

```json
{
  "data": {
    "metrics": [
      {
        "metric_id": "PWR-01",
        "sector": "power_generation",
        "name": "Grid average emission intensity",
        "unit": "gCO2/kWh",
        "source": "SBTi SDA-Power",
        "required_inputs": ["electricity_generated_mwh", "total_emissions_tco2e"]
      },
      {
        "metric_id": "STL-01",
        "sector": "steel",
        "name": "Crude steel intensity",
        "unit": "tCO2e/tonne crude steel",
        "source": "SBTi SDA-Steel",
        "required_inputs": ["crude_steel_production_tonnes", "scope1_scope2_tco2e"]
      }
    ],
    "total_metrics": 24
  }
}
```

---

### Pathway Generator Engine

Generates sector-specific decarbonization pathways aligned with SBTi SDA and IEA NZE scenarios.

#### `POST /v1/packs/028/engines/pathway-generator/generate`

**Request Body:**

```json
{
  "sector": "steel",
  "base_year": 2023,
  "base_year_intensity": 1.85,
  "target_year_near": 2030,
  "target_year_long": 2050,
  "scenario": "nze_15c",
  "convergence_model": "s_curve",
  "production_forecast": {
    "2023": 50000000,
    "2025": 51000000,
    "2030": 55000000,
    "2035": 57000000,
    "2040": 58000000,
    "2045": 59000000,
    "2050": 60000000
  },
  "region": "global",
  "include_absolute_emissions": true,
  "include_milestone_tracking": true
}
```

**Response (200 OK):**

```json
{
  "data": {
    "pathway_id": "pw_steel_nze15c_2023_abc123",
    "pathway_name": "SDA-Steel-NZE-1.5C",
    "sector": "steel",
    "scenario": "nze_15c",
    "convergence_model": "s_curve",
    "base_year": 2023,
    "base_intensity": 1.85,
    "target_2030": 1.25,
    "target_2040": 0.55,
    "target_2050": 0.10,
    "annual_reduction_rate_near": 0.057,
    "annual_reduction_rate_long": 0.064,
    "annual_pathway": [
      {"year": 2023, "intensity": 1.850, "absolute_emissions": 92500000, "production": 50000000, "convergence_pct": 0.0},
      {"year": 2024, "intensity": 1.760, "absolute_emissions": 89760000, "production": 51000000, "convergence_pct": 5.1},
      {"year": 2025, "intensity": 1.670, "absolute_emissions": 85170000, "production": 51000000, "convergence_pct": 10.3},
      {"year": 2026, "intensity": 1.580, "absolute_emissions": 82680000, "production": 52320000, "convergence_pct": 15.4},
      {"year": 2027, "intensity": 1.490, "absolute_emissions": 79218000, "production": 53170000, "convergence_pct": 20.6},
      {"year": 2028, "intensity": 1.400, "absolute_emissions": 75600000, "production": 54000000, "convergence_pct": 25.7},
      {"year": 2029, "intensity": 1.325, "absolute_emissions": 72540000, "production": 54750000, "convergence_pct": 30.0},
      {"year": 2030, "intensity": 1.250, "absolute_emissions": 68750000, "production": 55000000, "convergence_pct": 34.3},
      {"year": 2035, "intensity": 0.850, "absolute_emissions": 48450000, "production": 57000000, "convergence_pct": 57.1},
      {"year": 2040, "intensity": 0.550, "absolute_emissions": 31900000, "production": 58000000, "convergence_pct": 74.3},
      {"year": 2045, "intensity": 0.280, "absolute_emissions": 16520000, "production": 59000000, "convergence_pct": 89.7},
      {"year": 2050, "intensity": 0.100, "absolute_emissions": 6000000, "production": 60000000, "convergence_pct": 100.0}
    ],
    "sbti_alignment": {
      "aligned": true,
      "ambition_level": "1.5C",
      "sda_methodology": "SDA-Steel",
      "coverage_scope1_2": 0.98,
      "near_term_valid": true,
      "long_term_valid": true
    },
    "iea_milestone_tracking": [
      {
        "milestone_id": "IEA-STL-001",
        "year": 2025,
        "description": "First commercial green hydrogen DRI plant",
        "status": "on_track",
        "company_alignment": 0.85
      },
      {
        "milestone_id": "IEA-STL-002",
        "year": 2030,
        "description": "10% of steel production via green hydrogen DRI",
        "status": "on_track",
        "company_alignment": 0.78
      }
    ]
  },
  "metadata": {
    "engine": "pathway_generator_engine",
    "version": "1.0.0",
    "processing_time_ms": 2840,
    "sbti_reference_version": "SDA Tool V3.0 (2025)",
    "iea_reference_version": "NZE 2050 (2023 update)",
    "provenance": {
      "algorithm": "sha256",
      "hash": "c3d4e5f6a7b8...",
      "timestamp": "2026-03-19T10:02:00Z"
    }
  }
}
```

#### `GET /v1/packs/028/engines/pathway-generator/pathways`

Returns all generated pathways.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sector` | Sector | No | Filter by sector |
| `scenario` | Scenario | No | Filter by scenario |
| `page` | integer | No | Page number (default: 1) |
| `page_size` | integer | No | Items per page (default: 20) |

#### `GET /v1/packs/028/engines/pathway-generator/pathways/{pathway_id}`

Returns a specific pathway by ID.

#### `DELETE /v1/packs/028/engines/pathway-generator/pathways/{pathway_id}`

Deletes a specific pathway. Requires `sector_pathway_admin` role.

#### `GET /v1/packs/028/engines/pathway-generator/sda-factors`

Returns SBTi SDA convergence factors for all sectors.

**Response (200 OK):**

```json
{
  "data": {
    "sda_factors": [
      {
        "sector": "steel",
        "sda_id": "SDA-Steel",
        "base_year_global_intensity": 1.85,
        "2030_target_intensity": 1.25,
        "2050_target_intensity": 0.10,
        "convergence_type": "global",
        "source": "SBTi SDA Tool V3.0",
        "effective_date": "2025-01-01"
      }
    ],
    "version": "SDA Tool V3.0 (2025)",
    "total_sectors": 12
  }
}
```

---

### Convergence Analyzer Engine

Analyzes company trajectory convergence toward sector pathway targets.

#### `POST /v1/packs/028/engines/convergence-analyzer/analyze`

**Request Body:**

```json
{
  "current_intensity": 1.65,
  "current_year": 2025,
  "sector_pathway_id": "pw_steel_nze15c_2023_abc123",
  "company_trajectory": [
    {"year": 2021, "intensity": 1.92},
    {"year": 2022, "intensity": 1.85},
    {"year": 2023, "intensity": 1.78},
    {"year": 2024, "intensity": 1.72},
    {"year": 2025, "intensity": 1.65}
  ],
  "include_projections": true,
  "projection_method": "trend_extrapolation"
}
```

**Response (200 OK):**

```json
{
  "data": {
    "convergence_status": "converging",
    "gap_to_pathway": {
      "current_gap_absolute": 0.02,
      "current_gap_pct": 0.012,
      "direction": "above_pathway",
      "narrowing": true
    },
    "gap_to_2030_target": {
      "gap_absolute": 0.40,
      "gap_pct": 0.242,
      "achievable_at_current_rate": true
    },
    "gap_to_2050_target": {
      "gap_absolute": 1.55,
      "gap_pct": 0.939,
      "achievable_at_current_rate": false,
      "acceleration_needed": true
    },
    "required_annual_reduction": {
      "to_2030_target": 0.057,
      "to_2050_target": 0.064,
      "current_rate": 0.048,
      "acceleration_needed_pct": 0.019
    },
    "risk_level": "medium",
    "risk_assessment": {
      "near_term": "low",
      "medium_term": "medium",
      "long_term": "high",
      "rationale": "Current reduction rate sufficient for 2030 but insufficient for 2050 without technology shifts"
    },
    "time_to_convergence_years": 3.2,
    "projected_trajectory": [
      {"year": 2025, "intensity": 1.65, "type": "actual"},
      {"year": 2026, "intensity": 1.57, "type": "projected"},
      {"year": 2027, "intensity": 1.49, "type": "projected"},
      {"year": 2028, "intensity": 1.42, "type": "projected"},
      {"year": 2029, "intensity": 1.35, "type": "projected"},
      {"year": 2030, "intensity": 1.28, "type": "projected"}
    ],
    "recommendations": [
      {
        "priority": 1,
        "action": "Accelerate EAF transition from 20% to 35% by 2030",
        "impact_tco2e_per_tonne": 0.18,
        "confidence": "high"
      },
      {
        "priority": 2,
        "action": "Pilot green hydrogen DRI at 1 facility by 2027",
        "impact_tco2e_per_tonne": 0.08,
        "confidence": "medium"
      }
    ]
  },
  "metadata": {
    "engine": "convergence_analyzer_engine",
    "version": "1.0.0",
    "processing_time_ms": 856,
    "provenance": {
      "algorithm": "sha256",
      "hash": "d4e5f6a7b8c9...",
      "timestamp": "2026-03-19T10:03:00Z"
    }
  }
}
```

---

### Technology Roadmap Engine

Builds sector-specific technology transition roadmaps with IEA milestone mapping.

#### `POST /v1/packs/028/engines/technology-roadmap/build`

**Request Body:**

```json
{
  "sector": "steel",
  "pathway_id": "pw_steel_nze15c_2023_abc123",
  "current_technology_mix": {
    "bf_bof": 0.75,
    "eaf_scrap": 0.20,
    "dri_natural_gas": 0.05
  },
  "installed_capacity": {
    "total_capacity_mtpa": 5.0,
    "bf_bof_capacity_mtpa": 3.75,
    "eaf_capacity_mtpa": 1.0,
    "dri_capacity_mtpa": 0.25
  },
  "capex_budget_annual_usd": 500000000,
  "region": "eu",
  "planning_horizon": 2050,
  "include_iea_milestones": true,
  "include_dependency_analysis": true
}
```

**Response (200 OK):**

```json
{
  "data": {
    "roadmap_id": "rm_steel_abc123",
    "sector": "steel",
    "planning_horizon": "2025-2050",
    "technology_transitions": [
      {
        "id": "tr_001",
        "from_technology": "bf_bof",
        "to_technology": "eaf_scrap",
        "start_year": 2025,
        "completion_year": 2035,
        "capacity_change_mtpa": -1.0,
        "new_capacity_mtpa": 2.0,
        "capex_total_usd": 1200000000,
        "emission_reduction_tco2e_pa": 1500000,
        "trl": 9,
        "confidence": "high"
      },
      {
        "id": "tr_002",
        "from_technology": "bf_bof",
        "to_technology": "dri_green_hydrogen",
        "start_year": 2027,
        "completion_year": 2040,
        "capacity_change_mtpa": -1.75,
        "new_capacity_mtpa": 2.0,
        "capex_total_usd": 3500000000,
        "emission_reduction_tco2e_pa": 3200000,
        "trl": 7,
        "confidence": "medium",
        "dependencies": ["green_hydrogen_supply", "renewable_electricity"]
      },
      {
        "id": "tr_003",
        "from_technology": "bf_bof",
        "to_technology": "bf_bof_with_ccs",
        "start_year": 2030,
        "completion_year": 2040,
        "capacity_change_mtpa": 0,
        "capex_total_usd": 800000000,
        "emission_reduction_tco2e_pa": 2000000,
        "capture_rate": 0.90,
        "trl": 7,
        "confidence": "medium",
        "dependencies": ["co2_transport_storage"]
      }
    ],
    "target_technology_mix": {
      "2030": {"bf_bof": 0.50, "eaf_scrap": 0.35, "dri_green_h2": 0.10, "dri_ng": 0.05},
      "2040": {"bf_bof_ccs": 0.15, "eaf_scrap": 0.40, "dri_green_h2": 0.40, "dri_ng": 0.05},
      "2050": {"eaf_scrap": 0.45, "dri_green_h2": 0.50, "bf_bof_ccs": 0.05}
    },
    "capex_schedule": [
      {"year": 2025, "amount_usd": 200000000, "technology": "eaf_expansion", "description": "EAF capacity expansion Phase 1"},
      {"year": 2026, "amount_usd": 250000000, "technology": "eaf_expansion", "description": "EAF capacity expansion Phase 2"},
      {"year": 2027, "amount_usd": 400000000, "technology": "dri_h2_pilot", "description": "Green hydrogen DRI pilot plant"},
      {"year": 2028, "amount_usd": 350000000, "technology": "dri_h2_pilot", "description": "DRI pilot commissioning + H2 supply"},
      {"year": 2029, "amount_usd": 450000000, "technology": "dri_h2_scale", "description": "DRI scale-up Phase 1"},
      {"year": 2030, "amount_usd": 500000000, "technology": "ccs_retrofit", "description": "CCS retrofit on remaining BF-BOF"}
    ],
    "iea_milestones": [
      {
        "milestone_id": "IEA-STL-001",
        "year": 2025,
        "description": "First commercial green hydrogen DRI plant",
        "company_status": "on_track",
        "company_action": "DRI pilot plant contracted for 2027 commissioning"
      },
      {
        "milestone_id": "IEA-STL-002",
        "year": 2030,
        "description": "10% of steel production via green hydrogen DRI",
        "company_status": "on_track",
        "company_action": "10% target achievable with planned DRI expansion"
      }
    ],
    "dependency_graph": {
      "nodes": [
        {"id": "green_hydrogen_supply", "status": "planned", "earliest_availability": 2027},
        {"id": "renewable_electricity", "status": "contracted", "earliest_availability": 2025},
        {"id": "co2_transport_storage", "status": "under_development", "earliest_availability": 2029},
        {"id": "scrap_supply", "status": "available", "constraint": "regional_availability"}
      ],
      "edges": [
        {"from": "dri_green_h2", "to": "green_hydrogen_supply", "type": "requires"},
        {"from": "dri_green_h2", "to": "renewable_electricity", "type": "requires"},
        {"from": "bf_bof_ccs", "to": "co2_transport_storage", "type": "requires"},
        {"from": "eaf_scrap", "to": "scrap_supply", "type": "requires"}
      ]
    },
    "risk_assessment": {
      "overall_risk": "medium",
      "technology_risks": [
        {"technology": "dri_green_h2", "risk": "medium", "factors": ["hydrogen cost uncertainty", "electrolyzer scale-up"]},
        {"technology": "ccs_retrofit", "risk": "medium", "factors": ["CO2 storage permitting", "capture cost"]},
        {"technology": "eaf_scrap", "risk": "low", "factors": ["scrap availability in EU"]}
      ],
      "cost_uncertainty_range": {"low": 4500000000, "mid": 5500000000, "high": 7200000000}
    }
  },
  "metadata": {
    "engine": "technology_roadmap_engine",
    "version": "1.0.0",
    "processing_time_ms": 4280,
    "iea_milestones_mapped": 18,
    "provenance": {
      "algorithm": "sha256",
      "hash": "e5f6a7b8c9d0...",
      "timestamp": "2026-03-19T10:04:00Z"
    }
  }
}
```

#### `GET /v1/packs/028/engines/technology-roadmap/milestones`

Returns IEA technology milestones.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sector` | Sector | No | Filter by sector |
| `year_from` | integer | No | Milestone year start |
| `year_to` | integer | No | Milestone year end |

---

### Abatement Waterfall Engine

Generates sector-specific abatement waterfall analysis with lever-by-lever contributions.

#### `POST /v1/packs/028/engines/abatement-waterfall/analyze`

**Request Body:**

```json
{
  "sector": "cement",
  "pathway_id": "pw_cement_nze15c_2023_def456",
  "current_emissions_tco2e": 3100000,
  "current_production_tonnes": 5000000,
  "sector_parameters": {
    "clinker_to_cement_ratio": 0.75,
    "alternative_fuel_share": 0.15,
    "thermal_efficiency_gj_per_tonne": 3.5,
    "process_emission_factor": 0.525,
    "electricity_intensity_kwh_per_tonne": 110,
    "grid_emission_factor_tco2e_per_mwh": 0.35
  },
  "target_year": 2030,
  "include_cost_curves": true,
  "include_interactions": true
}
```

**Response (200 OK):**

```json
{
  "data": {
    "waterfall_id": "wf_cement_abc123",
    "sector": "cement",
    "start_emissions_tco2e": 3100000,
    "target_emissions_tco2e": 2325000,
    "total_reduction_tco2e": 775000,
    "total_reduction_pct": 0.25,
    "levers": [
      {
        "id": "lever_001",
        "name": "Clinker Substitution",
        "description": "Reduce clinker-to-cement ratio from 0.75 to 0.65 using fly ash, slag, calcined clay",
        "reduction_tco2e": 262500,
        "reduction_pct": 0.085,
        "cost_per_tco2e_eur": -15,
        "cost_type": "negative_cost",
        "start_year": 2024,
        "end_year": 2028,
        "certainty": "high",
        "trl": 9,
        "dependencies": [],
        "sector_lever_category": "material_efficiency"
      },
      {
        "id": "lever_002",
        "name": "Alternative Fuels (Biomass/Waste)",
        "description": "Increase alternative fuel share from 15% to 40% using biomass and waste-derived fuels",
        "reduction_tco2e": 186000,
        "reduction_pct": 0.060,
        "cost_per_tco2e_eur": 10,
        "cost_type": "low_cost",
        "start_year": 2024,
        "end_year": 2029,
        "certainty": "high",
        "trl": 9,
        "dependencies": ["waste_supply_chain"],
        "sector_lever_category": "fuel_switching"
      },
      {
        "id": "lever_003",
        "name": "Energy Efficiency (Kiln Upgrade)",
        "description": "Upgrade to high-efficiency precalciner kiln, reducing thermal energy from 3.5 to 3.1 GJ/t",
        "reduction_tco2e": 124000,
        "reduction_pct": 0.040,
        "cost_per_tco2e_eur": -5,
        "cost_type": "negative_cost",
        "start_year": 2025,
        "end_year": 2028,
        "certainty": "high",
        "trl": 9,
        "dependencies": [],
        "sector_lever_category": "energy_efficiency"
      },
      {
        "id": "lever_004",
        "name": "Renewable Electricity Procurement",
        "description": "Switch from grid electricity (0.35 tCO2e/MWh) to 100% renewable via PPA",
        "reduction_tco2e": 93000,
        "reduction_pct": 0.030,
        "cost_per_tco2e_eur": 20,
        "cost_type": "moderate_cost",
        "start_year": 2024,
        "end_year": 2026,
        "certainty": "high",
        "trl": 9,
        "dependencies": ["renewable_ppa_availability"],
        "sector_lever_category": "renewable_energy"
      },
      {
        "id": "lever_005",
        "name": "Carbon Capture (Pilot)",
        "description": "Pilot CCS on 1 kiln line capturing 30% of process emissions",
        "reduction_tco2e": 109500,
        "reduction_pct": 0.035,
        "cost_per_tco2e_eur": 80,
        "cost_type": "high_cost",
        "start_year": 2027,
        "end_year": 2030,
        "certainty": "medium",
        "trl": 7,
        "dependencies": ["co2_storage_access", "capex_approval"],
        "sector_lever_category": "carbon_capture"
      }
    ],
    "lever_interactions": [
      {
        "lever_a": "lever_001",
        "lever_b": "lever_003",
        "type": "synergy",
        "effect_tco2e": 8500,
        "description": "Lower clinker ratio reduces kiln throughput, amplifying efficiency gains"
      },
      {
        "lever_a": "lever_002",
        "lever_b": "lever_005",
        "type": "conflict",
        "effect_tco2e": -12000,
        "description": "Higher biomass share reduces fossil CO2 available for CCS capture"
      }
    ],
    "residual_emissions_tco2e": 2325000,
    "residual_pct_of_base": 0.75,
    "cost_summary": {
      "total_capex_eur": 245000000,
      "annual_opex_change_eur": -8500000,
      "weighted_avg_cost_per_tco2e_eur": 18.5,
      "payback_period_years": 6.2
    }
  },
  "metadata": {
    "engine": "abatement_waterfall_engine",
    "version": "1.0.0",
    "processing_time_ms": 2140,
    "provenance": {
      "algorithm": "sha256",
      "hash": "f6a7b8c9d0e1...",
      "timestamp": "2026-03-19T10:05:00Z"
    }
  }
}
```

---

### Sector Benchmark Engine

Provides multi-dimensional benchmarking against sector peers, leaders, and pathway targets.

#### `POST /v1/packs/028/engines/sector-benchmark/benchmark`

**Request Body:**

```json
{
  "sector": "steel",
  "company_intensity": 1.65,
  "company_year": 2025,
  "company_production_tonnes": 5000000,
  "region": "eu",
  "include_peer_comparison": true,
  "include_regulatory_benchmarks": true,
  "benchmark_dimensions": ["sector_average", "sector_leader", "sbti_peers", "iea_pathway", "regulatory"]
}
```

**Response (200 OK):**

```json
{
  "data": {
    "benchmark_id": "bm_steel_abc123",
    "sector": "steel",
    "company_intensity": 1.65,
    "company_year": 2025,
    "benchmarks": {
      "sector_average": {
        "value": 1.82,
        "year": 2024,
        "source": "World Steel Association",
        "company_vs_benchmark": -0.093,
        "company_status": "below_average"
      },
      "sector_leader_p10": {
        "value": 0.95,
        "year": 2024,
        "source": "CDP Climate Change 2024",
        "company_vs_benchmark": 0.737,
        "company_status": "above_leader"
      },
      "sbti_peer_average": {
        "value": 1.55,
        "year": 2024,
        "count": 28,
        "source": "SBTi Validated Targets Database",
        "company_vs_benchmark": 0.065,
        "company_status": "above_sbti_peers"
      },
      "iea_pathway_2025": {
        "value": 1.60,
        "year": 2025,
        "scenario": "nze_15c",
        "source": "IEA NZE 2050 (2023 update)",
        "company_vs_benchmark": 0.031,
        "company_status": "slightly_above_pathway"
      },
      "eu_ets_benchmark": {
        "value": 1.52,
        "year": 2025,
        "source": "EU ETS Phase 4 benchmarks",
        "company_vs_benchmark": 0.086,
        "company_status": "above_regulatory"
      }
    },
    "percentile_rank": 42,
    "improvement_needed": {
      "to_average": 0,
      "to_sbti_peer": 0.10,
      "to_iea_pathway": 0.05,
      "to_leader": 0.70,
      "to_regulatory": 0.13
    },
    "peer_comparison": {
      "total_peers_in_sector": 156,
      "peers_with_sbti_targets": 28,
      "company_rank": 65,
      "peer_intensity_distribution": {
        "p10": 0.95,
        "p25": 1.25,
        "p50": 1.70,
        "p75": 2.05,
        "p90": 2.35
      }
    },
    "recommendations": [
      {
        "priority": 1,
        "action": "Close 5% gap to IEA 2025 pathway through EAF transition acceleration",
        "impact": "Would move to 38th percentile"
      },
      {
        "priority": 2,
        "action": "Target SBTi peer average (1.55) by 2027 through energy efficiency + scrap optimization",
        "impact": "Would align with validated peer group"
      }
    ]
  },
  "metadata": {
    "engine": "sector_benchmark_engine",
    "version": "1.0.0",
    "processing_time_ms": 1420,
    "benchmark_data_freshness": "2024-12-31",
    "provenance": {
      "algorithm": "sha256",
      "hash": "a7b8c9d0e1f2...",
      "timestamp": "2026-03-19T10:06:00Z"
    }
  }
}
```

---

### Scenario Comparison Engine

Compares sector pathways across multiple IEA climate scenarios.

#### `POST /v1/packs/028/engines/scenario-comparison/compare`

**Request Body:**

```json
{
  "sector": "power_generation",
  "base_year": 2023,
  "base_year_intensity": 0.45,
  "scenarios": ["nze_15c", "wb2c", "2c", "aps", "steps"],
  "milestones": [2025, 2030, 2035, 2040, 2045, 2050],
  "include_investment_analysis": true,
  "include_technology_comparison": true,
  "include_risk_analysis": true,
  "region": "global"
}
```

**Response (200 OK):**

```json
{
  "data": {
    "comparison_id": "sc_power_abc123",
    "sector": "power_generation",
    "scenario_pathways": [
      {
        "scenario": "nze_15c",
        "name": "Net Zero Emissions (1.5C)",
        "milestones": [
          {"year": 2025, "intensity": 0.40},
          {"year": 2030, "intensity": 0.22},
          {"year": 2035, "intensity": 0.10},
          {"year": 2040, "intensity": 0.04},
          {"year": 2045, "intensity": 0.01},
          {"year": 2050, "intensity": 0.00}
        ],
        "total_reduction_pct": 1.00,
        "cumulative_investment_usd": 12500000000
      },
      {
        "scenario": "wb2c",
        "name": "Well-Below 2C",
        "milestones": [
          {"year": 2025, "intensity": 0.42},
          {"year": 2030, "intensity": 0.28},
          {"year": 2035, "intensity": 0.16},
          {"year": 2040, "intensity": 0.08},
          {"year": 2045, "intensity": 0.03},
          {"year": 2050, "intensity": 0.01}
        ],
        "total_reduction_pct": 0.98,
        "cumulative_investment_usd": 10200000000
      },
      {
        "scenario": "2c",
        "name": "2 Degrees Celsius",
        "milestones": [
          {"year": 2025, "intensity": 0.43},
          {"year": 2030, "intensity": 0.32},
          {"year": 2035, "intensity": 0.22},
          {"year": 2040, "intensity": 0.14},
          {"year": 2045, "intensity": 0.08},
          {"year": 2050, "intensity": 0.04}
        ],
        "total_reduction_pct": 0.91,
        "cumulative_investment_usd": 8500000000
      },
      {
        "scenario": "aps",
        "name": "Announced Pledges",
        "milestones": [
          {"year": 2025, "intensity": 0.43},
          {"year": 2030, "intensity": 0.35},
          {"year": 2035, "intensity": 0.28},
          {"year": 2040, "intensity": 0.22},
          {"year": 2045, "intensity": 0.16},
          {"year": 2050, "intensity": 0.12}
        ],
        "total_reduction_pct": 0.73,
        "cumulative_investment_usd": 6800000000
      },
      {
        "scenario": "steps",
        "name": "Stated Policies",
        "milestones": [
          {"year": 2025, "intensity": 0.44},
          {"year": 2030, "intensity": 0.38},
          {"year": 2035, "intensity": 0.33},
          {"year": 2040, "intensity": 0.28},
          {"year": 2045, "intensity": 0.24},
          {"year": 2050, "intensity": 0.20}
        ],
        "total_reduction_pct": 0.56,
        "cumulative_investment_usd": 4200000000
      }
    ],
    "risk_return_analysis": {
      "nze_15c": {"transition_risk": "low", "physical_risk": "lowest", "investment": "highest", "positioning": "first_mover"},
      "wb2c": {"transition_risk": "low_medium", "physical_risk": "low", "investment": "high", "positioning": "strong"},
      "2c": {"transition_risk": "medium", "physical_risk": "medium", "investment": "medium", "positioning": "adequate"},
      "aps": {"transition_risk": "medium_high", "physical_risk": "medium_high", "investment": "medium_low", "positioning": "reactive"},
      "steps": {"transition_risk": "highest", "physical_risk": "highest", "investment": "lowest", "positioning": "stranded_risk"}
    },
    "optimal_pathway": {
      "recommended_scenario": "nze_15c",
      "rationale": "NZE 1.5C provides strongest strategic positioning with manageable investment premium. EU policy trajectory (Fit for 55, REPowerEU) already aligned with NZE milestones.",
      "investment_premium_vs_steps": 8300000000,
      "risk_reduction_value": "Avoids stranded asset exposure estimated at $4.5B+"
    },
    "sensitivity_analysis": {
      "key_drivers": [
        {"parameter": "renewable_lcoe", "sensitivity": 0.35, "direction": "Lower LCOE accelerates all scenarios"},
        {"parameter": "carbon_price", "sensitivity": 0.28, "direction": "Higher carbon price favors NZE/WB2C"},
        {"parameter": "gas_price", "sensitivity": 0.18, "direction": "Higher gas prices accelerate coal-to-gas-to-renewable"},
        {"parameter": "storage_cost", "sensitivity": 0.12, "direction": "Lower storage cost enables higher renewable penetration"},
        {"parameter": "demand_growth", "sensitivity": 0.07, "direction": "Higher demand increases absolute investment"}
      ]
    }
  },
  "metadata": {
    "engine": "scenario_comparison_engine",
    "version": "1.0.0",
    "processing_time_ms": 8420,
    "scenarios_compared": 5,
    "provenance": {
      "algorithm": "sha256",
      "hash": "b8c9d0e1f2a3...",
      "timestamp": "2026-03-19T10:07:00Z"
    }
  }
}
```

---

## Workflows API

### Sector Pathway Design Workflow

End-to-end workflow for designing an SBTi SDA-aligned sector pathway.

#### `POST /v1/packs/028/workflows/sector-pathway-design/execute`

**Request Body:**

```json
{
  "company_profile": {
    "name": "CementWorks SA",
    "nace_codes": ["C23.51"],
    "base_year": 2023,
    "base_year_production_tonnes": 5000000,
    "base_year_emissions_tco2e": 3100000,
    "current_clinker_ratio": 0.75,
    "current_alt_fuel_share": 0.15,
    "region": "eu"
  },
  "target_scenario": "nze_15c",
  "target_year_near": 2030,
  "target_year_long": 2050,
  "include_validation_report": true
}
```

**Response (200 OK):**

```json
{
  "data": {
    "workflow_id": "wf_spd_abc123",
    "status": "completed",
    "phases": [
      {
        "phase": 1,
        "name": "SectorClassify",
        "status": "completed",
        "duration_ms": 124,
        "result": {
          "primary_sector": "cement",
          "sda_methodology": "SDA-Cement",
          "intensity_metric": "tCO2e/tonne cement"
        }
      },
      {
        "phase": 2,
        "name": "IntensityCalc",
        "status": "completed",
        "duration_ms": 342,
        "result": {
          "base_year_intensity": 0.62,
          "data_quality_score": 1.8
        }
      },
      {
        "phase": 3,
        "name": "PathwayGen",
        "status": "completed",
        "duration_ms": 2840,
        "result": {
          "pathway_id": "pw_cement_nze15c_2023_def456",
          "target_2030": 0.47,
          "target_2050": 0.04
        }
      },
      {
        "phase": 4,
        "name": "GapAnalysis",
        "status": "completed",
        "duration_ms": 856,
        "result": {
          "gap_to_2030": "24.2%",
          "required_annual_reduction": "3.4%",
          "risk_level": "medium"
        }
      },
      {
        "phase": 5,
        "name": "ValidationReport",
        "status": "completed",
        "duration_ms": 1240,
        "result": {
          "sbti_aligned": true,
          "validation_score": "95/100",
          "report_id": "rpt_sbti_val_abc123"
        }
      }
    ],
    "total_duration_ms": 5402,
    "final_result": {
      "pathway_id": "pw_cement_nze15c_2023_def456",
      "sector": "cement",
      "base_intensity": 0.62,
      "target_2030_intensity": 0.47,
      "target_2050_intensity": 0.04,
      "gap_to_pathway": "24.2%",
      "sbti_aligned": true,
      "validation_report_id": "rpt_sbti_val_abc123"
    }
  }
}
```

### Pathway Validation Workflow

#### `POST /v1/packs/028/workflows/pathway-validation/execute`

Validates an existing pathway against SBTi SDA criteria.

### Technology Planning Workflow

#### `POST /v1/packs/028/workflows/technology-planning/execute`

Generates a complete technology transition roadmap with CapEx mapping and dependency analysis.

### Progress Monitoring Workflow

#### `POST /v1/packs/028/workflows/progress-monitoring/execute`

Updates intensity metrics, checks convergence, and generates a progress report.

### Multi-Scenario Analysis Workflow

#### `POST /v1/packs/028/workflows/multi-scenario-analysis/execute`

Runs full multi-scenario analysis across 5 IEA scenarios with risk assessment.

### Full Sector Assessment Workflow

#### `POST /v1/packs/028/workflows/full-sector-assessment/execute`

Runs the complete 7-phase sector pathway assessment (all engines).

---

## Templates API

### Common Template Parameters

All template endpoints accept the following common parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `format` | string | Yes | Output format: `md`, `html`, `json`, `pdf` |
| `language` | string | No | Report language (default: `en`) |
| `include_appendices` | boolean | No | Include detailed appendices (default: true) |
| `include_charts` | boolean | No | Include chart data/markup (default: true) |

### Sector Pathway Report

#### `POST /v1/packs/028/templates/sector-pathway-report/render`

**Request Body:**

```json
{
  "pathway_id": "pw_steel_nze15c_2023_abc123",
  "convergence_analysis_id": "ca_steel_abc123",
  "format": "html",
  "include_appendices": true,
  "include_charts": true,
  "branding": {
    "company_name": "SteelCorp International",
    "logo_url": "https://steelcorp.com/logo.png",
    "report_date": "2026-03-19"
  }
}
```

**Response (200 OK):**

```json
{
  "data": {
    "report_id": "rpt_sp_abc123",
    "format": "html",
    "content": "<html>...</html>",
    "content_length_bytes": 245000,
    "sections": [
      "Executive Summary",
      "Sector Classification",
      "Base Year Intensity Analysis",
      "SDA Convergence Pathway",
      "Annual Intensity Targets",
      "Absolute Emission Trajectory",
      "SBTi Alignment Assessment",
      "IEA Milestone Tracking",
      "Gap Analysis",
      "Recommendations",
      "Methodology Notes",
      "Appendix: Raw Data Tables"
    ],
    "charts_included": [
      "intensity_convergence_chart",
      "absolute_emissions_trajectory",
      "pathway_vs_actual_comparison",
      "sector_benchmark_position"
    ]
  },
  "metadata": {
    "template": "sector_pathway_report",
    "version": "1.0.0",
    "rendering_time_ms": 2840,
    "provenance": {
      "algorithm": "sha256",
      "hash": "c9d0e1f2a3b4...",
      "timestamp": "2026-03-19T10:10:00Z"
    }
  }
}
```

### Intensity Convergence Report

#### `POST /v1/packs/028/templates/intensity-convergence-report/render`

Renders intensity tracking and convergence analysis report.

### Technology Roadmap Report

#### `POST /v1/packs/028/templates/technology-roadmap-report/render`

Renders technology transition roadmap with IEA milestones.

### Abatement Waterfall Report

#### `POST /v1/packs/028/templates/abatement-waterfall-report/render`

Renders sector abatement waterfall with lever contributions.

### Sector Benchmark Report

#### `POST /v1/packs/028/templates/sector-benchmark-report/render`

Renders multi-dimensional sector benchmarking dashboard.

### Scenario Comparison Report

#### `POST /v1/packs/028/templates/scenario-comparison-report/render`

Renders multi-scenario pathway comparison and risk analysis.

### SBTi Validation Report

#### `POST /v1/packs/028/templates/sbti-validation-report/render`

Renders SBTi SDA pathway validation and compliance report.

### Sector Strategy Report

#### `POST /v1/packs/028/templates/sector-strategy-report/render`

Renders executive sector transition strategy document.

---

## Integrations API

### Pack Orchestrator

#### `POST /v1/packs/028/integrations/orchestrator/execute`

Executes the 10-phase DAG pipeline with sector-specific conditional routing.

**Request Body:**

```json
{
  "pipeline": "full_sector_assessment",
  "company_profile": { "..." : "..." },
  "options": {
    "skip_phases": [],
    "parallel_execution": true,
    "timeout_seconds": 300
  }
}
```

### SBTi SDA Bridge

#### `GET /v1/packs/028/integrations/sbti-sda/sectors`

Returns all SBTi SDA sector definitions and convergence factors.

#### `GET /v1/packs/028/integrations/sbti-sda/pathway/{sector}`

Returns the SBTi SDA pathway data for a specific sector.

#### `POST /v1/packs/028/integrations/sbti-sda/validate`

Validates a pathway against SBTi SDA criteria.

### IEA NZE Bridge

#### `GET /v1/packs/028/integrations/iea-nze/sectors`

Returns all IEA NZE sector pathway data.

#### `GET /v1/packs/028/integrations/iea-nze/milestones`

Returns IEA technology milestones (400+).

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sector` | Sector | No | Filter by sector |
| `year` | integer | No | Filter by milestone year |
| `status` | string | No | Filter by status (`on_track`, `off_track`, `achieved`) |

#### `GET /v1/packs/028/integrations/iea-nze/scenarios/{scenario_id}`

Returns pathway data for a specific IEA scenario.

### IPCC AR6 Bridge

#### `GET /v1/packs/028/integrations/ipcc-ar6/emission-factors`

Returns IPCC AR6 sector-specific emission factors.

#### `GET /v1/packs/028/integrations/ipcc-ar6/gwp-values`

Returns IPCC AR6 GWP-100 values.

### PACK-021 Bridge

#### `GET /v1/packs/028/integrations/pack021/baseline`

Retrieves baseline emissions data from PACK-021.

#### `GET /v1/packs/028/integrations/pack021/targets`

Retrieves target definitions from PACK-021.

### MRV Bridge

#### `POST /v1/packs/028/integrations/mrv/calculate`

Routes emission calculations to appropriate MRV agents based on sector.

### Decarbonization Bridge

#### `POST /v1/packs/028/integrations/decarb/actions`

Retrieves sector-specific decarbonization actions.

### Data Bridge

#### `POST /v1/packs/028/integrations/data/intake`

Routes activity data intake to appropriate DATA agents.

### Health Check

#### `GET /v1/packs/028/integrations/health-check`

Returns pack health status across 20 categories.

**Response (200 OK):**

```json
{
  "data": {
    "overall_score": 100,
    "overall_status": "HEALTHY",
    "categories": [
      {"name": "Database Connectivity", "score": 100, "status": "HEALTHY"},
      {"name": "Redis Cache", "score": 100, "status": "HEALTHY"},
      {"name": "SBTi SDA Data", "score": 100, "status": "HEALTHY"},
      {"name": "IEA NZE Data", "score": 100, "status": "HEALTHY"},
      {"name": "IPCC AR6 Data", "score": 100, "status": "HEALTHY"},
      {"name": "MRV Agent Connectivity", "score": 100, "status": "HEALTHY"},
      {"name": "DATA Agent Connectivity", "score": 100, "status": "HEALTHY"},
      {"name": "FOUND Agent Connectivity", "score": 100, "status": "HEALTHY"},
      {"name": "Engine Availability", "score": 100, "status": "HEALTHY"},
      {"name": "Workflow Availability", "score": 100, "status": "HEALTHY"},
      {"name": "Template Availability", "score": 100, "status": "HEALTHY"},
      {"name": "Integration Availability", "score": 100, "status": "HEALTHY"},
      {"name": "Migration Status", "score": 100, "status": "HEALTHY"},
      {"name": "PACK-021 Bridge", "score": 100, "status": "HEALTHY"},
      {"name": "Sector Data Freshness", "score": 100, "status": "HEALTHY"},
      {"name": "Benchmark Data Freshness", "score": 100, "status": "HEALTHY"},
      {"name": "Emission Factor Data", "score": 100, "status": "HEALTHY"},
      {"name": "Cache Performance", "score": 100, "status": "HEALTHY"},
      {"name": "API Response Time", "score": 100, "status": "HEALTHY"},
      {"name": "Provenance Integrity", "score": 100, "status": "HEALTHY"}
    ],
    "last_check": "2026-03-19T10:15:00Z"
  }
}
```

### Setup Wizard

#### `POST /v1/packs/028/integrations/setup-wizard/step/{step_number}`

Executes a specific step of the 7-step setup wizard.

**Step 1: Sector Selection**
```json
{
  "sector": "steel",
  "sub_sectors": ["integrated_steel", "eaf_steel"]
}
```

**Step 2: Company Profile**
```json
{
  "name": "SteelCorp International",
  "nace_codes": ["C24.10"],
  "country": "DE",
  "production_volume_tonnes": 5000000
}
```

**Step 3: Base Year Configuration**
```json
{
  "base_year": 2023,
  "base_year_intensity": 1.85,
  "base_year_production": 5000000
}
```

**Step 4: Scenario Selection**
```json
{
  "primary_scenario": "nze_15c",
  "comparison_scenarios": ["wb2c", "2c"],
  "convergence_model": "s_curve"
}
```

**Step 5: Technology Assessment**
```json
{
  "current_technology_mix": {
    "bf_bof": 0.75,
    "eaf_scrap": 0.20,
    "dri_natural_gas": 0.05
  }
}
```

**Step 6: Budget and Timeline**
```json
{
  "capex_budget_annual_usd": 500000000,
  "planning_horizon": 2050,
  "near_term_target_year": 2030
}
```

**Step 7: Validation and Go-Live**
```json
{
  "run_health_check": true,
  "generate_initial_pathway": true,
  "enable_monitoring": true
}
```

---

## Webhooks

PACK-028 supports webhooks for asynchronous notifications.

### Supported Events

| Event | Description |
|-------|-------------|
| `pathway.generated` | New pathway successfully generated |
| `pathway.validation.completed` | Pathway validation completed |
| `convergence.alert` | Convergence gap exceeds threshold |
| `milestone.achieved` | IEA milestone achieved |
| `milestone.missed` | IEA milestone deadline passed |
| `benchmark.update` | Sector benchmark data updated |
| `workflow.completed` | Workflow execution completed |
| `workflow.failed` | Workflow execution failed |

### Webhook Registration

```http
POST /v1/packs/028/webhooks
Content-Type: application/json

{
  "url": "https://company.com/webhooks/pack028",
  "events": ["pathway.generated", "convergence.alert", "milestone.missed"],
  "secret": "whsec_abc123..."
}
```

### Webhook Payload

```json
{
  "event": "convergence.alert",
  "timestamp": "2026-03-19T10:20:00Z",
  "data": {
    "sector": "steel",
    "pathway_id": "pw_steel_nze15c_2023_abc123",
    "current_gap_pct": 0.08,
    "threshold_pct": 0.05,
    "risk_level": "medium",
    "message": "Steel sector intensity gap to NZE pathway exceeds 5% threshold"
  },
  "signature": "sha256=abc123..."
}
```

---

## SDK Reference

### Python SDK

```python
from greenlang.packs.pack028 import SectorPathwayClient

client = SectorPathwayClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_abc123...",
)

# Classify sector
classification = client.classify_sector(
    nace_codes=["C24.10"],
    gics_code="15104020",
)

# Generate pathway
pathway = client.generate_pathway(
    sector="steel",
    base_year=2023,
    base_year_intensity=1.85,
    scenario="nze_15c",
)

# Run full assessment
assessment = client.run_full_assessment(
    company_profile={...},
    scenarios=["nze_15c", "wb2c", "2c"],
)

# Generate report
report = client.render_report(
    template="sector_pathway_report",
    pathway_id=pathway.pathway_id,
    format="pdf",
)
```

### TypeScript SDK

```typescript
import { SectorPathwayClient } from '@greenlang/pack-028';

const client = new SectorPathwayClient({
  baseUrl: 'https://api.greenlang.io',
  apiKey: 'gl_pk_abc123...',
});

// Classify sector
const classification = await client.classifySector({
  naceCodes: ['C24.10'],
  gicsCode: '15104020',
});

// Generate pathway
const pathway = await client.generatePathway({
  sector: 'steel',
  baseYear: 2023,
  baseYearIntensity: 1.85,
  scenario: 'nze_15c',
});
```

---

## Appendix: OpenAPI Specification

The full OpenAPI 3.1 specification is available at:

```
https://api.greenlang.io/v1/packs/028/openapi.json
https://api.greenlang.io/v1/packs/028/openapi.yaml
```

Interactive API documentation (Swagger UI) is available at:

```
https://api.greenlang.io/v1/packs/028/docs
```

ReDoc documentation is available at:

```
https://api.greenlang.io/v1/packs/028/redoc
```
