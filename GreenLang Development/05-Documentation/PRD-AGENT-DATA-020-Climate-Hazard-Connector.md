# PRD: AGENT-DATA-020 - Climate Hazard Connector

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-020 |
| **Agent ID** | GL-DATA-GEO-002 |
| **Component** | Climate Hazard Connector Agent (Hazard Database Ingestion, Risk Index Calculation, Scenario Projection, Exposure Assessment, Vulnerability Scoring, Compliance Reporting, Monitoring Pipeline) |
| **Category** | Data Intake Agent (Geospatial / Climate Risk) |
| **Priority** | P0 - Critical (required for TCFD physical risk, CSRD ESRS E1, EU Taxonomy climate adaptation, EUDR) |
| **Status** | Not Built - Full SDK Build Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires climate hazard data processing capabilities for physical climate risk
assessment, TCFD compliance, CSRD ESRS E1 climate metrics, EU Taxonomy adaptation screening, and
EUDR deforestation correlation. Without a production-grade Climate Hazard Connector:

- **No unified climate hazard data ingestion**: Multiple hazard databases (IPCC, Aqueduct, ThinkHazard, EM-DAT, NGFS) not uniformly handled
- **No climate hazard risk indexing**: Cannot compute composite risk indices from multi-source hazard data
- **No scenario projection engine**: Cannot project hazard intensity under SSP/RCP climate scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5)
- **No exposure assessment**: Cannot assess asset/supply chain exposure to physical climate hazards
- **No vulnerability scoring**: Cannot score entity vulnerability based on hazard proximity, intensity, and frequency
- **No time horizon analysis**: Cannot project hazard changes across short (2030), medium (2040), and long-term (2050, 2100) horizons
- **No compliance reporting**: Cannot generate TCFD/CSRD-compliant physical climate risk reports
- **No monitoring pipeline**: No automated pipeline for continuous hazard monitoring per asset/location
- **No multi-hazard correlation**: Cannot analyze compound hazards (e.g., drought + wildfire, flood + landslide)
- **No audit trail**: Climate hazard assessments not tracked for regulatory compliance

## 3. Existing Implementation

### 3.1 Layer 1: Related Modules

#### GIS/Mapping Connector (GL-DATA-GEO-001)
**Directory**: `greenlang/gis_connector/` (15 files, ~20K lines)
- Spatial analysis, CRS transformation, land cover classification
- Can provide geospatial primitives (point-in-polygon, distance calculations)
- **Re-export candidates**: `SpatialAnalyzerEngine`, `CRSTransformerEngine`, `BoundaryResolverEngine`

#### Deforestation Satellite Connector (GL-DATA-GEO-003)
**Directory**: `greenlang/deforestation_satellite/` (15 files, ~20K lines)
- Satellite imagery, vegetation indices, alert aggregation
- **Re-export candidates**: `SatelliteDataEngine`, `AlertAggregationEngine`

#### Climate Hazard Variant Dimension
From `AGENT_CATALOG.csv` and `02-COMPREHENSIVE-PRD.md`:
- 12 hazard types: Heat, Flood, Drought, Wildfire, Cyclone, Storm Surge, Sea Level Rise, Landslide, Extreme Precipitation, Cold Wave, Water Stress, Coastal Erosion
- 300 estimated variants from Geography x Climate_Hazard x Scenario combinations

### 3.2 Layer 1 Tests
None found for climate hazard functionality.

## 4. Identified Gaps

### Gap 1: No Climate Hazard SDK Package
No `greenlang/climate_hazard/` package providing a clean SDK for climate hazard data processing.

### Gap 2: No Hazard Database Connector
No unified interface to major climate hazard databases: WRI Aqueduct, ThinkHazard (GFDRR), EM-DAT, NGFS Climate Scenarios, IPCC AR6 regional hazard data, Copernicus Climate Data Store (CDS), NASA SEDAC.

### Gap 3: No Risk Index Engine
No composite climate risk index calculation engine combining probability, intensity, frequency, and duration for each hazard type.

### Gap 4: No Scenario Projection Engine
Cannot project hazard intensity under SSP/RCP scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5) with time horizon support (2030, 2040, 2050, 2100).

### Gap 5: No Exposure Assessment Engine
Cannot assess geographic exposure of assets, facilities, supply chain nodes to climate hazards with proximity and intensity scoring.

### Gap 6: No Vulnerability Scoring Engine
Cannot compute entity-level vulnerability scores combining hazard exposure, sensitivity, and adaptive capacity.

### Gap 7: No Compliance Report Engine
Cannot generate TCFD Metrics & Targets, CSRD ESRS E1, EU Taxonomy climate adaptation screening reports.

### Gap 8: No Monitoring Pipeline
No automated pipeline for continuous hazard monitoring with alerting on hazard threshold breaches.

### Gap 9: No Prometheus Metrics
No 12-metric pattern for climate hazard connector monitoring.

### Gap 10: No REST API
No FastAPI endpoints for climate hazard operations.

### Gap 11: No Database Migration
No persistent storage for hazard data, risk indices, and assessments.

### Gap 12: No K8s/CI/CD
No deployment manifests or CI/CD pipeline.

## 5. Architecture (Final State)

### 5.1 SDK Package Structure

```
greenlang/climate_hazard/
├── __init__.py              # Public API, agent metadata (GL-DATA-GEO-002)
├── config.py                # ClimateHazardConfig with GL_CLIMATE_HAZARD_ env prefix
├── models.py                # Pydantic v2 models for all climate hazard data structures
├── hazard_database.py       # HazardDatabaseEngine - multi-source hazard data ingestion
├── risk_index.py            # RiskIndexEngine - composite climate risk index calculation
├── scenario_projector.py    # ScenarioProjectorEngine - SSP/RCP scenario projection
├── exposure_assessor.py     # ExposureAssessorEngine - asset/supply chain exposure analysis
├── vulnerability_scorer.py  # VulnerabilityScorerEngine - entity vulnerability scoring
├── compliance_reporter.py   # ComplianceReporterEngine - TCFD/CSRD/EU Taxonomy reports
├── hazard_pipeline.py       # HazardPipelineEngine - end-to-end monitoring orchestration
├── provenance.py            # ProvenanceTracker - SHA-256 chain-hashed audit trails
├── metrics.py               # 12 Prometheus metrics with gl_chc_ prefix
├── setup.py                 # ClimateHazardService facade
├── _smoke_test.py           # Quick validation script
└── api/
    ├── __init__.py
    └── router.py            # 20 FastAPI endpoints at /api/v1/climate-hazard
```

### 5.2 Engine Architecture (7 Engines)

#### Engine 1: HazardDatabaseEngine (`hazard_database.py`)
**Purpose**: Unified ingestion from multiple climate hazard data sources.

**Data Sources** (10 sources):
| Source | Type | Coverage | Hazards |
|--------|------|----------|---------|
| WRI Aqueduct | Water Risk Atlas | Global | Flood, Drought, Water Stress |
| ThinkHazard (GFDRR) | Multi-hazard database | Global | 11 hazard types |
| EM-DAT (CRED) | Disaster database | Global, 1900-present | All natural disasters |
| NGFS Climate Scenarios | Financial risk | Global | Transition + Physical |
| IPCC AR6 WG2 | Regional assessments | Global, 12 regions | All climate hazards |
| Copernicus CDS | Reanalysis/projections | Global | Temperature, Precipitation |
| NASA SEDAC | Population/exposure | Global | Multi-hazard exposure |
| NOAA NCEI | Historical events | Global | Storms, Floods, Fire |
| Swiss Re CatNet | Insurance risk | Global | NatCat perils |
| Munich Re NatCatSERVICE | Loss data | Global | NatCat events |

**Key Methods**:
- `register_source(source_id, name, source_type, config)` - Register a hazard data source
- `ingest_hazard_data(source_id, hazard_type, region, time_range)` - Ingest data from source
- `get_hazard_data(hazard_type, location, time_range)` - Query hazard data by location
- `list_sources()` / `get_source(source_id)` - Source registry CRUD
- `search_hazard_data(hazard_type, region, severity_min)` - Search across all sources
- `get_historical_events(hazard_type, region, start_year, end_year)` - Historical event lookup
- `aggregate_sources(hazard_type, location, strategy)` - Multi-source data fusion
- `get_statistics()` / `clear()` - Engine state management

**12 Hazard Types** (enum `HazardType`):
1. RIVERINE_FLOOD - River overflow flooding
2. COASTAL_FLOOD - Storm surge and tidal flooding
3. DROUGHT - Meteorological/hydrological/agricultural drought
4. EXTREME_HEAT - Heat waves, temperature extremes
5. EXTREME_COLD - Cold waves, frost events
6. WILDFIRE - Forest/brush fires
7. TROPICAL_CYCLONE - Hurricanes, typhoons, cyclones
8. EXTREME_PRECIPITATION - Heavy rainfall, hail
9. WATER_STRESS - Water scarcity, depletion
10. SEA_LEVEL_RISE - Chronic sea level rise
11. LANDSLIDE - Rainfall/seismic-induced landslides
12. COASTAL_EROSION - Shoreline retreat

#### Engine 2: RiskIndexEngine (`risk_index.py`)
**Purpose**: Compute composite climate risk indices from raw hazard data.

**Risk Components** (4 dimensions):
- **Probability** (0-1): Likelihood of hazard occurrence in time window
- **Intensity** (0-10): Magnitude/severity of hazard when it occurs
- **Frequency** (events/year): Expected occurrence rate
- **Duration** (days): Average duration per event

**Composite Risk Score** (0-100):
```
risk_score = (probability * 0.30 + normalized_intensity * 0.30 +
              normalized_frequency * 0.25 + normalized_duration * 0.15) * 100
```

**Risk Levels** (5 tiers):
| Level | Score | Description |
|-------|-------|-------------|
| NEGLIGIBLE | 0-20 | No significant hazard risk |
| LOW | 20-40 | Minor hazard potential |
| MEDIUM | 40-60 | Moderate hazard potential |
| HIGH | 60-80 | Significant hazard risk |
| EXTREME | 80-100 | Critical/imminent hazard |

**Key Methods**:
- `calculate_risk_index(hazard_type, location, time_horizon, scenario)` - Single hazard risk
- `calculate_multi_hazard_index(location, hazard_types, time_horizon, scenario)` - Multi-hazard composite
- `calculate_compound_risk(location, hazard_combinations, time_horizon)` - Compound event risk (e.g., drought + wildfire)
- `rank_hazards(location, time_horizon, scenario)` - Rank all hazards by risk score
- `compare_locations(locations, hazard_type, time_horizon)` - Location risk comparison
- `get_risk_trend(location, hazard_type, time_horizons)` - Risk evolution over time
- `get_statistics()` / `clear()` - Engine state management

#### Engine 3: ScenarioProjectorEngine (`scenario_projector.py`)
**Purpose**: Project climate hazard intensity under IPCC SSP/RCP scenarios.

**Supported Scenarios** (8 scenarios):
| Scenario | Description | Warming by 2100 |
|----------|-------------|-----------------|
| SSP1-1.9 | Sustainability (very low) | ~1.0-1.8°C |
| SSP1-2.6 | Sustainability (low) | ~1.3-2.4°C |
| SSP2-4.5 | Middle of the road | ~2.1-3.5°C |
| SSP3-7.0 | Regional rivalry (high) | ~2.8-4.6°C |
| SSP5-8.5 | Fossil-fueled (very high) | ~3.3-5.7°C |
| RCP2.6 | Low emissions (legacy) | ~0.9-2.3°C |
| RCP4.5 | Medium emissions (legacy) | ~1.7-3.2°C |
| RCP8.5 | High emissions (legacy) | ~3.2-5.4°C |

**Time Horizons** (5 periods):
- BASELINE (1995-2014), NEAR_TERM (2021-2040), MID_TERM (2041-2060), LONG_TERM (2061-2080), END_CENTURY (2081-2100)

**Key Methods**:
- `project_hazard(hazard_type, location, scenario, time_horizon)` - Single projection
- `project_multi_scenario(hazard_type, location, scenarios, time_horizon)` - Multi-scenario comparison
- `project_time_series(hazard_type, location, scenario, time_horizons)` - Time series projection
- `calculate_warming_factor(scenario, time_horizon)` - Temperature delta calculation
- `apply_scaling_factors(baseline_risk, hazard_type, warming_delta)` - Scale hazard by warming
- `get_scenario_info(scenario)` / `list_scenarios()` - Scenario metadata
- `get_statistics()` / `clear()` - Engine state management

**Hazard Scaling Factors** (per °C warming):
| Hazard Type | Scaling per °C | Source |
|-------------|---------------|--------|
| EXTREME_HEAT | +2.0x intensity, +1.8x frequency | IPCC AR6 WG1 Ch11 |
| RIVERINE_FLOOD | +1.3x intensity, +1.2x frequency | IPCC AR6 WG1 Ch11 |
| DROUGHT | +1.5x intensity, +1.4x frequency | IPCC AR6 WG1 Ch11 |
| WILDFIRE | +1.6x intensity, +1.5x frequency | IPCC AR6 WG1 Ch12 |
| TROPICAL_CYCLONE | +1.1x intensity, +0.9x frequency | IPCC AR6 WG1 Ch11 |
| SEA_LEVEL_RISE | +0.3m per °C (cumulative) | IPCC AR6 WG1 Ch9 |
| EXTREME_PRECIPITATION | +1.4x intensity per °C | Clausius-Clapeyron |
| WATER_STRESS | +1.3x intensity, +1.2x frequency | IPCC AR6 WG2 Ch4 |

#### Engine 4: ExposureAssessorEngine (`exposure_assessor.py`)
**Purpose**: Assess asset and supply chain exposure to climate hazards.

**Asset Types** (8 types):
1. FACILITY - Manufacturing plants, offices, warehouses
2. SUPPLY_CHAIN_NODE - Supplier locations, logistics hubs
3. AGRICULTURAL_PLOT - Crop fields, forestry plots (EUDR)
4. INFRASTRUCTURE - Roads, bridges, ports, power plants
5. REAL_ESTATE - Buildings, housing, commercial property
6. NATURAL_ASSET - Forests, wetlands, biodiversity areas
7. WATER_SOURCE - Rivers, aquifers, reservoirs
8. COASTAL_ASSET - Ports, coastal infrastructure

**Exposure Metrics**:
- **Proximity Score** (0-1): Distance-based hazard exposure decay
- **Intensity at Location** (0-10): Projected hazard intensity at asset coordinates
- **Frequency Exposure** (events/year): Expected hazard frequency at location
- **Population Density Factor**: Local population density weighting
- **Elevation Factor**: Elevation-based flood/sea level exposure
- **Composite Exposure Score** (0-100): Weighted combination

**Key Methods**:
- `register_asset(asset_id, name, asset_type, location, metadata)` - Register an asset
- `assess_exposure(asset_id, hazard_type, scenario, time_horizon)` - Single asset exposure
- `assess_portfolio_exposure(asset_ids, hazard_types, scenario, time_horizon)` - Portfolio assessment
- `assess_supply_chain_exposure(supply_chain_nodes, hazard_types, scenario)` - Supply chain mapping
- `identify_hotspots(asset_ids, hazard_types, threshold)` - Exposure hotspot detection
- `get_exposure_map(hazard_type, bounding_box, resolution)` - Spatial exposure grid
- `list_assets()` / `get_asset(asset_id)` - Asset registry CRUD
- `get_statistics()` / `clear()` - Engine state management

#### Engine 5: VulnerabilityScorerEngine (`vulnerability_scorer.py`)
**Purpose**: Score entity vulnerability combining exposure, sensitivity, and adaptive capacity.

**Vulnerability Framework** (IPCC AR5/AR6):
```
Vulnerability = f(Exposure, Sensitivity, Adaptive Capacity)
vulnerability_score = (exposure_weight * exposure + sensitivity_weight * sensitivity -
                       adaptive_weight * adaptive_capacity) * 100
```

Default weights: Exposure=0.40, Sensitivity=0.35, Adaptive Capacity=0.25

**Sensitivity Factors** (sector-specific):
| Sector | Factors |
|--------|---------|
| Agriculture | Crop type, irrigation, soil quality, growing season |
| Real Estate | Building age, construction material, flood proofing |
| Infrastructure | Design standard, maintenance level, redundancy |
| Supply Chain | Supplier diversity, inventory buffer, lead time |
| Natural Assets | Ecosystem health, biodiversity, connectivity |

**Adaptive Capacity Indicators**:
- Financial reserves, insurance coverage
- Early warning systems, emergency preparedness
- Infrastructure redundancy, backup systems
- Workforce flexibility, remote work capability
- Governance quality, regulatory environment

**Key Methods**:
- `score_vulnerability(entity_id, hazard_type, exposure_data, sensitivity_profile, adaptive_capacity)` - Full scoring
- `score_sector_vulnerability(sector, location, hazard_type, scenario, time_horizon)` - Sector-level
- `create_sensitivity_profile(entity_id, sector, factors)` - Define sensitivity
- `create_adaptive_capacity_profile(entity_id, indicators)` - Define adaptive capacity
- `calculate_residual_risk(vulnerability_score, adaptation_measures)` - Post-adaptation risk
- `rank_entities(entity_ids, hazard_type)` - Entity vulnerability ranking
- `get_statistics()` / `clear()` - Engine state management

#### Engine 6: ComplianceReporterEngine (`compliance_reporter.py`)
**Purpose**: Generate regulatory compliance reports for climate hazard assessments.

**Supported Frameworks** (6 frameworks):
| Framework | Sections | Requirements |
|-----------|----------|--------------|
| TCFD | Strategy, Risk Management, Metrics & Targets | Physical risk scenario analysis, resilience assessment |
| CSRD/ESRS E1 | E1-1 through E1-9 | Climate adaptation plan, physical risk metrics |
| EU Taxonomy | Climate Adaptation (Art 11) | Substantial contribution + DNSH screening |
| CDP Climate | C2.3, C2.4 | Physical risk identification, financial impact |
| ISSB/IFRS S2 | Physical risks & opportunities | Scenario analysis, financial quantification |
| NGFS | Physical risk assessment | Financial stability physical risk metrics |

**Report Types** (5 types):
1. PHYSICAL_RISK_ASSESSMENT - Comprehensive multi-hazard physical risk report
2. SCENARIO_ANALYSIS - Climate scenario comparison report
3. ADAPTATION_SCREENING - EU Taxonomy climate adaptation screening
4. EXPOSURE_SUMMARY - Portfolio/supply chain exposure report
5. EXECUTIVE_DASHBOARD - C-suite executive summary

**Output Formats**: JSON, HTML, Markdown, Text, CSV

**Key Methods**:
- `generate_report(report_type, format, assessment_data, framework, parameters)` - Main report generation
- `generate_tcfd_report(assessment_data, format)` - TCFD physical risk report
- `generate_csrd_report(assessment_data, format)` - CSRD ESRS E1 report
- `generate_taxonomy_screening(assessment_data, format)` - EU Taxonomy adaptation screening
- `get_report(report_id)` / `list_reports(report_type, format, limit)` - Report retrieval
- `get_statistics()` / `clear()` - Engine state management

#### Engine 7: HazardPipelineEngine (`hazard_pipeline.py`)
**Purpose**: End-to-end orchestration of climate hazard assessment workflows.

**Pipeline Stages** (7 stages):
1. **ingest** - Gather hazard data from registered sources
2. **index** - Calculate risk indices for target hazards
3. **project** - Project under selected SSP/RCP scenarios
4. **assess** - Assess exposure of registered assets
5. **score** - Calculate vulnerability scores
6. **report** - Generate compliance reports
7. **audit** - Record provenance and notify

**Key Methods**:
- `run_pipeline(assets, hazard_types, scenarios, time_horizons, report_frameworks, stages, parameters)` - Full pipeline
- `run_batch_pipeline(asset_portfolios, hazard_types, scenarios, parameters)` - Batch processing
- `get_pipeline_run(run_id)` / `list_pipeline_runs(limit)` - Run tracking
- `get_health()` - Engine availability check
- `get_statistics()` / `clear()` - Engine state management

### 5.3 Models (`models.py`)

**Enums** (12+):
- `HazardType` (12 types) - Climate hazard classification
- `RiskLevel` (5 levels) - NEGLIGIBLE, LOW, MEDIUM, HIGH, EXTREME
- `Scenario` (8 scenarios) - SSP1-1.9 through RCP8.5
- `TimeHorizon` (5 periods) - BASELINE through END_CENTURY
- `AssetType` (8 types) - Facility through Coastal Asset
- `ReportType` (5 types) - Assessment report types
- `ReportFormat` (5 formats) - JSON, HTML, Markdown, Text, CSV
- `DataSourceType` (6 types) - GLOBAL_DATABASE, REGIONAL_INDEX, EVENT_CATALOG, SCENARIO_MODEL, SATELLITE, REANALYSIS
- `ExposureLevel` (5 levels) - NONE, LOW, MODERATE, HIGH, CRITICAL
- `SensitivityLevel` (5 levels) - VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH
- `AdaptiveCapacity` (5 levels) - VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH
- `VulnerabilityLevel` (5 levels) - NEGLIGIBLE, LOW, MODERATE, HIGH, CRITICAL

**Pydantic Models** (14+):
- `HazardDataRecord` - Individual hazard data point
- `HazardSource` - Data source registration
- `RiskIndex` - Composite risk index result
- `ScenarioProjection` - Projected hazard under scenario
- `Asset` - Registered asset with location
- `ExposureResult` - Asset exposure assessment
- `VulnerabilityScore` - Entity vulnerability result
- `SensitivityProfile` - Sector-specific sensitivity factors
- `AdaptiveCapacityProfile` - Adaptive capacity indicators
- `ComplianceReport` - Generated compliance report
- `PipelineRun` - Pipeline execution record
- `HazardEvent` - Historical hazard event
- `CompoundHazard` - Multi-hazard combination
- `Location` - WGS84 coordinate with elevation

**Layer 1 Re-exports**:
- `SpatialAnalyzerEngine` from `greenlang.gis_connector.spatial_analyzer`
- `BoundaryResolverEngine` from `greenlang.gis_connector.boundary_resolver`
- `CRSTransformerEngine` from `greenlang.gis_connector.crs_transformer`

### 5.4 Configuration (`config.py`)

**Environment Variables** (22+ settings with `GL_CLIMATE_HAZARD_` prefix):
- `GL_CLIMATE_HAZARD_DEFAULT_SCENARIO` - Default SSP scenario (default: SSP2-4.5)
- `GL_CLIMATE_HAZARD_DEFAULT_TIME_HORIZON` - Default time horizon (default: MID_TERM)
- `GL_CLIMATE_HAZARD_DEFAULT_REPORT_FORMAT` - Default report format (default: json)
- `GL_CLIMATE_HAZARD_MAX_HAZARD_SOURCES` - Max registered sources (default: 50)
- `GL_CLIMATE_HAZARD_MAX_ASSETS` - Max registered assets (default: 10000)
- `GL_CLIMATE_HAZARD_RISK_INDEX_PROBABILITY_WEIGHT` - Probability weight (default: 0.30)
- `GL_CLIMATE_HAZARD_RISK_INDEX_INTENSITY_WEIGHT` - Intensity weight (default: 0.30)
- `GL_CLIMATE_HAZARD_RISK_INDEX_FREQUENCY_WEIGHT` - Frequency weight (default: 0.25)
- `GL_CLIMATE_HAZARD_RISK_INDEX_DURATION_WEIGHT` - Duration weight (default: 0.15)
- `GL_CLIMATE_HAZARD_VULNERABILITY_EXPOSURE_WEIGHT` - Exposure weight (default: 0.40)
- `GL_CLIMATE_HAZARD_VULNERABILITY_SENSITIVITY_WEIGHT` - Sensitivity weight (default: 0.35)
- `GL_CLIMATE_HAZARD_VULNERABILITY_ADAPTIVE_WEIGHT` - Adaptive capacity weight (default: 0.25)
- `GL_CLIMATE_HAZARD_EXTREME_RISK_THRESHOLD` - Extreme risk threshold (default: 80)
- `GL_CLIMATE_HAZARD_HIGH_RISK_THRESHOLD` - High risk threshold (default: 60)
- `GL_CLIMATE_HAZARD_MEDIUM_RISK_THRESHOLD` - Medium risk threshold (default: 40)
- `GL_CLIMATE_HAZARD_LOW_RISK_THRESHOLD` - Low risk threshold (default: 20)
- `GL_CLIMATE_HAZARD_MAX_PIPELINE_RUNS` - Max stored pipeline runs (default: 500)
- `GL_CLIMATE_HAZARD_MAX_REPORTS` - Max stored reports (default: 1000)
- `GL_CLIMATE_HAZARD_MAX_RISK_INDICES` - Max stored risk indices (default: 5000)
- `GL_CLIMATE_HAZARD_PROVENANCE_ENABLED` - Enable provenance tracking (default: true)
- `GL_CLIMATE_HAZARD_METRICS_ENABLED` - Enable Prometheus metrics (default: true)
- `GL_CLIMATE_HAZARD_LOG_LEVEL` - Logging level (default: INFO)

### 5.5 Provenance (`provenance.py`)

SHA-256 chain-hashed audit trail with:
- **8 entity types**: hazard_source, hazard_data, risk_index, scenario_projection, asset, exposure, vulnerability, compliance_report
- **36+ actions**: register_source, ingest_data, calculate_risk, project_scenario, register_asset, assess_exposure, score_vulnerability, generate_report, run_pipeline, etc.

### 5.6 Metrics (`metrics.py`)

12 Prometheus metrics with `gl_chc_` prefix:
| Metric | Type | Description |
|--------|------|-------------|
| `gl_chc_hazard_data_ingested_total` | Counter | Total hazard data records ingested |
| `gl_chc_risk_indices_calculated_total` | Counter | Total risk indices calculated |
| `gl_chc_scenario_projections_total` | Counter | Total scenario projections computed |
| `gl_chc_exposure_assessments_total` | Counter | Total exposure assessments performed |
| `gl_chc_vulnerability_scores_total` | Counter | Total vulnerability scores computed |
| `gl_chc_reports_generated_total` | Counter | Total compliance reports generated |
| `gl_chc_pipeline_runs_total` | Counter | Total pipeline runs executed |
| `gl_chc_active_sources` | Gauge | Currently registered hazard sources |
| `gl_chc_active_assets` | Gauge | Currently registered assets |
| `gl_chc_high_risk_locations` | Gauge | Locations with HIGH/EXTREME risk |
| `gl_chc_ingestion_duration_seconds` | Histogram | Data ingestion latency |
| `gl_chc_pipeline_duration_seconds` | Histogram | Pipeline execution latency |

## 6. API Design

### 6.1 REST Endpoints (20 endpoints at `/api/v1/climate-hazard`)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/sources` | Register a hazard data source |
| 2 | GET | `/sources` | List registered sources |
| 3 | GET | `/sources/{source_id}` | Get source details |
| 4 | POST | `/hazard-data/ingest` | Ingest hazard data from source |
| 5 | GET | `/hazard-data` | Query hazard data by location/type |
| 6 | GET | `/hazard-data/events` | Get historical hazard events |
| 7 | POST | `/risk-index/calculate` | Calculate risk index for location |
| 8 | POST | `/risk-index/multi-hazard` | Calculate multi-hazard composite index |
| 9 | POST | `/risk-index/compare` | Compare risk across locations |
| 10 | POST | `/scenarios/project` | Project hazard under scenario |
| 11 | GET | `/scenarios` | List available scenarios |
| 12 | POST | `/assets` | Register an asset for monitoring |
| 13 | GET | `/assets` | List registered assets |
| 14 | POST | `/exposure/assess` | Assess asset exposure |
| 15 | POST | `/exposure/portfolio` | Assess portfolio exposure |
| 16 | POST | `/vulnerability/score` | Score entity vulnerability |
| 17 | POST | `/reports/generate` | Generate compliance report |
| 18 | GET | `/reports/{report_id}` | Get generated report |
| 19 | POST | `/pipeline/run` | Run full hazard assessment pipeline |
| 20 | GET | `/health` | Service health check |

## 7. Database Migration (V050)

### Schema: `climate_hazard_service`

**10 Tables**:
1. `hazard_sources` - Registered climate hazard data sources
2. `hazard_data_records` - Ingested hazard data points
3. `historical_events` - Historical climate hazard events (EM-DAT, NOAA)
4. `risk_indices` - Computed risk index results
5. `scenario_projections` - Climate scenario projections
6. `assets` - Registered assets for exposure monitoring
7. `exposure_assessments` - Asset exposure results
8. `vulnerability_scores` - Entity vulnerability scores
9. `compliance_reports` - Generated compliance reports
10. `pipeline_runs` - Pipeline execution records

**3 Hypertables** (TimescaleDB):
- `hazard_data_records` - Time-series hazard observations
- `risk_indices` - Time-series risk calculations
- `pipeline_runs` - Time-series pipeline executions

**2 Continuous Aggregates**:
- `hazard_data_hourly` - Hourly hazard data summaries
- `risk_index_daily` - Daily risk index averages

**Standard additions**: 108+ indexes, 13 RLS policies, 20 security permissions

## 8. Deployment

### 8.1 Dockerfile
`deployment/docker/Dockerfile.climate-hazard` - Python 3.11-slim, health check, non-root user

### 8.2 Kubernetes Manifests (10 files)
```
deployment/kubernetes/climate-hazard-service/
├── configmap.yaml          # Environment configuration
├── deployment.yaml         # 2 replicas, resource limits
├── hpa.yaml               # Auto-scaling (2-8 replicas)
├── networkpolicy.yaml     # Ingress/egress rules
├── pdb.yaml               # Pod disruption budget
├── secret.yaml            # Sensitive configuration
├── service.yaml           # ClusterIP service
├── servicemonitor.yaml    # Prometheus scraping
├── alerts/
│   └── climate-hazard-alerts.yaml  # PrometheusRule alerts
└── grafana/
    └── climate-hazard-dashboard.json  # Grafana dashboard
```

### 8.3 CI/CD
`.github/workflows/climate-hazard-ci.yml` - 7 CI jobs (lint, type-check, unit-tests, integration-tests, security-scan, build, deploy)

### 8.4 Auth Integration
- Import `chc_router` in `auth_setup.py`
- Add 20 PERMISSION_MAP entries in `route_protector.py`

## 9. Testing Requirements

### 9.1 Unit Tests (15+ files, 1200+ test functions)
| Test File | Engine | Target Tests |
|-----------|--------|-------------|
| `test_config.py` | Config | 100+ |
| `test_models.py` | Models | 200+ |
| `test_provenance.py` | Provenance | 60+ |
| `test_metrics.py` | Metrics | 60+ |
| `test_hazard_database.py` | Engine 1 | 150+ |
| `test_risk_index.py` | Engine 2 | 120+ |
| `test_scenario_projector.py` | Engine 3 | 100+ |
| `test_exposure_assessor.py` | Engine 4 | 100+ |
| `test_vulnerability_scorer.py` | Engine 5 | 80+ |
| `test_compliance_reporter.py` | Engine 6 | 80+ |
| `test_hazard_pipeline.py` | Engine 7 | 60+ |
| `test_setup.py` | Service facade | 80+ |
| `test_router.py` | API endpoints | 60+ |

### 9.2 Integration Tests (2+ files, 80+ test functions)
- `test_full_pipeline_integration.py` - End-to-end pipeline tests
- `test_regulatory_compliance_integration.py` - Framework-specific compliance tests

## 10. Downstream Consumers

1. **GL-RISK-X-002** Flood Risk Calculator
2. **GL-RISK-X-003** Heat Stress Analyzer
3. **GL-RISK-X-004** Wildfire Risk Assessor
4. **GL-RISK-X-005** Drought Risk Evaluator
5. **GL-RISK-X-006** Cyclone/Hurricane Modeler
6. **GL-RISK-X-007** Sea Level Rise Projector
7. **GL-EUDR-APP** EU Deforestation Regulation Platform
8. **GL-CSRD-APP** CSRD Reporting Platform (ESRS E1)
9. **GL-Taxonomy-APP** EU Taxonomy Platform (Climate Adaptation)
10. **GL-DATA-X-008** API Gateway Agent

## 11. Success Metrics

| Metric | Target |
|--------|--------|
| Unit test pass rate | 100% |
| Integration test pass rate | 100% |
| Test function count | 1,300+ |
| Code coverage | 85%+ |
| All 7 engines functional | Yes |
| All 20 API endpoints working | Yes |
| 12 Prometheus metrics emitting | Yes |
| Provenance chain valid | Yes |
| TCFD report generation | Functional |
| CSRD ESRS E1 report generation | Functional |
| EU Taxonomy screening | Functional |
