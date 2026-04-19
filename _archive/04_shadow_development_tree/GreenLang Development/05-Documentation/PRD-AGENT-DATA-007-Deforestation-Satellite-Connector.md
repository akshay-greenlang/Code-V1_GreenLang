# PRD: AGENT-DATA-007 - Deforestation Satellite Connector

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-007 |
| **Agent ID** | GL-DATA-GEO-003 |
| **Component** | Deforestation Satellite Connector Agent (Satellite Imagery Acquisition, Forest Cover Change Detection, Near-Real-Time Alert Aggregation, EUDR Baseline Assessment, Deforestation Classification, Compliance Reporting, Monitoring Pipeline) |
| **Category** | Data Intake Agent (Geospatial / Remote Sensing) |
| **Priority** | P0 - Critical (required for EUDR deforestation compliance, Dec 30, 2025 deadline) |
| **Status** | Layer 1 Extensive (~14 files in extensions/satellite/ + governance/validation/geolocation/), SDK Build Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires satellite-based deforestation monitoring for EU Deforestation
Regulation (EUDR, EU 2023/1115) compliance. The EUDR mandates that products placed on the EU
market must not originate from land deforested after December 31, 2020. Without a production-grade
Deforestation Satellite Connector:

- **No unified satellite data acquisition**: Multiple satellite sources (Sentinel-2, Landsat, MODIS) not uniformly handled
- **No forest cover change detection**: Cannot systematically detect deforestation between baseline and current dates
- **No near-real-time alert aggregation**: GLAD, RADD, and FIRMS alerts not aggregated into unified monitoring
- **No EUDR baseline verification**: Cannot determine forest status as of December 31, 2020 cutoff date
- **No deforestation classification**: Cannot classify land cover change types (clear-cut, degradation, regrowth)
- **No compliance reporting**: Cannot generate EUDR-compliant deforestation assessment reports
- **No monitoring pipeline**: No automated pipeline for continuous deforestation monitoring per production plot
- **No vegetation index calculation**: NDVI, EVI, NBR, NDWI not computed from raw satellite bands
- **No multi-temporal analysis**: Cannot track forest cover trends over time series
- **No audit trail**: Satellite-derived assessments not tracked for regulatory compliance

## 3. Existing Implementation

### 3.1 Layer 1: Satellite Extensions Module
**Directory**: `greenlang/extensions/satellite/` (13 files, ~5,600 lines)

#### Satellite Clients
- `clients/sentinel2_client.py` (621 lines): Copernicus Data Space API, 6 bands (B2-B12), SCL cloud masking, tile caching, mock data generation
- `clients/landsat_client.py` (665 lines): USGS Earth Explorer API, band harmonization to Sentinel-2 equivalents, QA_PIXEL cloud masking, WRS-2 path/row calculation, HarmonizedSatelliteClient

#### Vegetation Index Analysis
- `analysis/vegetation_indices.py` (692 lines): 7 indices (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI), unified VegetationIndexCalculator, band resampling

#### Change Detection
- `analysis/change_detection.py` (723 lines): BiTemporalChangeDetector (NDVI/NBR differencing, 5 change types), MultiTemporalAnalyzer (trend analysis, breakpoint detection), EUDR compliance report generator

#### Forest Classification
- `models/forest_classifier.py` (612 lines): ForestClassifier (binary + 10-class land cover), tree cover % estimation calibrated against Hansen GFC, GEDI canopy height integration, AdaptiveThresholdClassifier

#### Alert System
- `alerts/deforestation_alert.py` (745 lines): GlobalForestWatchClient (GLAD 30m weekly + RADD 10m daily), DeforestationAlertSystem (multi-source aggregation, EUDR compliance assessment, Dec 31 2020 cutoff)

#### Pipeline Orchestrator
- `pipeline/analysis_pipeline.py` (704 lines): 7-stage DeforestationAnalysisPipeline (init -> acquisition -> indices -> classification -> change detection -> alert integration -> reporting), ThreadPoolExecutor parallel polygon analysis

### 3.2 Layer 1: Deforestation Baseline Checker
**File**: `greenlang/governance/validation/geolocation/deforestation_baseline.py` (835 lines)

- EUDR_CUTOFF_DATE constant (Dec 31, 2020)
- 8 country-specific forest definitions (BRA, IDN, COD, MYS, CIV, GHA, COL, PER)
- FAO default definition (10% tree cover, 5m height, 0.5 ha area)
- Risk scoring (0-100) with 6 weighted factors
- Polygon-based grid sampling with point-in-polygon
- Conservative aggregation (worst-case compliance)
- SHA-256 provenance hashing

### 3.3 Layer 1 Tests
None found in production test suite.

## 4. Identified Gaps

### Gap 1: No Deforestation Satellite SDK Package
No `greenlang/deforestation_satellite/` package providing a clean SDK wrapping Layer 1 capabilities.

### Gap 2: No Unified Satellite Data Acquisition Engine
Layer 1 has separate clients with mock-only implementations. Need production-ready unified acquisition with sensor harmonization.

### Gap 3: No Production Forest Change Engine
Layer 1 has basic bi-temporal and multi-temporal detectors. Need production-grade change detection with configurable thresholds and multi-sensor fusion.

### Gap 4: No Unified Alert Aggregation SDK
Layer 1 alert system uses dataclasses. Need Pydantic v2 models, de-duplication, priority scoring, and FIRMS fire alert integration.

### Gap 5: No Baseline Assessment SDK
Layer 1 baseline checker uses Decimal-based stubs. Need production wrapper with Hansen GFC integration and country-specific forest definitions.

### Gap 6: No Classification SDK Engine
Layer 1 classifier uses numpy arrays. Need SDK wrapper with confidence scoring, multi-class output, and provenance tracking.

### Gap 7: No Compliance Report Engine
Layer 1 generates basic dicts. Need structured Pydantic v2 compliance reports with evidence packaging for DDS integration.

### Gap 8: No Monitoring Pipeline SDK
Layer 1 pipeline uses ThreadPoolExecutor. Need production SDK with scheduling, checkpointing, and incremental monitoring.

### Gap 9: No Prometheus Metrics
No 12-metric pattern for satellite connector monitoring.

### Gap 10: No REST API
No FastAPI endpoints for deforestation monitoring operations.

### Gap 11: No Database Migration
No persistent storage for satellite assessments, alerts, and monitoring results.

### Gap 12: No K8s/CI/CD
No deployment manifests or CI/CD pipeline.

## 5. Architecture (Final State)

### 5.1 SDK Package Structure

```
greenlang/deforestation_satellite/
+-- __init__.py              # Public API, agent metadata (GL-DATA-GEO-003)
+-- config.py                # DeforestationSatelliteConfig with GL_DEFORESTATION_SAT_ env prefix
+-- models.py                # Pydantic v2 models for all data structures
+-- satellite_data.py        # SatelliteDataEngine - multi-source imagery acquisition
+-- forest_change.py         # ForestChangeEngine - bi-temporal & multi-temporal change detection
+-- alert_aggregation.py     # AlertAggregationEngine - GLAD/RADD/FIRMS alert aggregation
+-- baseline_assessment.py   # BaselineAssessmentEngine - EUDR Dec 31 2020 baseline verification
+-- deforestation_classifier.py  # DeforestationClassifierEngine - land cover classification
+-- compliance_report.py     # ComplianceReportEngine - EUDR compliance report generation
+-- monitoring_pipeline.py   # MonitoringPipelineEngine - continuous monitoring orchestration
+-- provenance.py            # ProvenanceTracker - SHA-256 chain-hashed audit trails
+-- metrics.py               # 12 Prometheus metrics
+-- setup.py                 # DeforestationSatelliteService facade
+-- api/
    +-- __init__.py
    +-- router.py            # FastAPI HTTP service with 20 endpoints
```

### 5.2 Seven Core Engines

#### Engine 1: SatelliteDataEngine
- Acquire Sentinel-2 L2A imagery (10m optical, 5-day revisit, 6 bands: B2/B3/B4/B8/B11/B12)
- Acquire Landsat 8/9 OLI imagery (30m optical, 8-day revisit, band harmonization)
- Harmonized multi-sensor acquisition with automatic fallback (Sentinel-2 primary, Landsat backup)
- Cloud masking using SCL (Sentinel-2) and QA_PIXEL (Landsat) layers
- 7 vegetation index calculation: NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI
- Band resampling for cross-resolution analysis (20m -> 10m)
- Time series acquisition for multi-temporal analysis
- Scene metadata extraction (date, cloud cover, satellite, tile ID)
- Bounding box and polygon-based spatial filtering
- Configurable date range and cloud cover thresholds

#### Engine 2: ForestChangeEngine
- Bi-temporal NDVI differencing (dNDVI) with 5 change types: NO_CHANGE, CLEAR_CUT, DEGRADATION, PARTIAL_LOSS, REGROWTH
- Bi-temporal NBR differencing (dNBR) for burn severity and fire-related deforestation
- Multi-index fusion (NDVI + NBR) for enhanced confidence scoring
- Multi-temporal trend analysis (linear regression per pixel: slope, intercept, R-squared)
- Breakpoint detection for abrupt changes (threshold-based, configurable)
- Annual tree cover loss/gain computation
- Area quantification per change type (hectares)
- Configurable thresholds: clear_cut (-0.3), degradation (-0.15), partial_loss (-0.05), regrowth (+0.1)
- Pre/post comparison with EUDR cutoff date alignment

#### Engine 3: AlertAggregationEngine
- GLAD alert retrieval (30m resolution, weekly, Landsat-based, humid tropics)
- RADD alert retrieval (10m resolution, near-daily, Sentinel-1 SAR, tropics only, cloud-penetrating)
- FIRMS fire alert integration (MODIS/VIIRS, daily, global coverage)
- Multi-source alert de-duplication (spatial + temporal proximity matching)
- Polygon-based spatial filtering (ray casting point-in-polygon)
- Alert confidence scoring: LOW, NOMINAL, HIGH
- Alert severity classification by area: LOW (<0.5ha), MEDIUM (0.5-5ha), HIGH (5-50ha), CRITICAL (>50ha)
- Post-EUDR-cutoff alert filtering (after Dec 31, 2020)
- Alert aggregation with summary statistics (by source, severity, confidence)
- Alert priority scoring for triage

#### Engine 4: BaselineAssessmentEngine
- EUDR cutoff date verification (December 31, 2020 - hardcoded constant)
- Historical forest cover determination at baseline date using Hansen GFC (tree cover 2000 + annual loss)
- Country-specific forest definitions for 8 priority countries (BRA, IDN, COD, MYS, CIV, GHA, COL, PER)
- FAO default forest definition (10% tree cover, 5m height, 0.5 ha minimum)
- Current vs. baseline forest cover comparison
- Forest cover change quantification (absolute and percentage)
- Polygon-based grid sampling with deterministic point generation
- Conservative aggregation (ALL sample points must be compliant)
- Risk scoring (0-100) with 6 weighted factors: deforestation proximity (30%), historical rate (25%), road proximity (15%), fire alerts (15%), protected area pressure (10%), commodity suitability (5%)
- Country risk adjustments for 10 high-risk countries

#### Engine 5: DeforestationClassifierEngine
- Binary forest/non-forest classification from NDVI thresholds
- 10-class land cover classification: DENSE_FOREST, OPEN_FOREST, SHRUBLAND, GRASSLAND, CROPLAND, BARE_SOIL, WATER, URBAN, WETLAND, UNKNOWN
- Hierarchical decision tree: Water (NDWI>0.3) -> Dense Forest (NDVI>=0.6, EVI>=0.35) -> Open Forest (NDVI>=0.4, EVI>=0.2) -> Shrubland -> Grassland -> Bare Soil
- Tree cover percentage estimation calibrated against Hansen GFC (NDVI 0.2->0%, 0.8->100%)
- EVI-blended estimation (60% NDVI + 40% EVI) when available
- GEDI canopy height integration (25m lidar footprints)
- Adaptive threshold classification for heterogeneous landscapes
- Per-pixel confidence scoring
- Change type discrimination: clear-cut vs. degradation vs. partial loss

#### Engine 6: ComplianceReportEngine
- Per-plot EUDR deforestation assessment with compliance status: COMPLIANT, REVIEW_REQUIRED, NON_COMPLIANT
- Risk level assignment: LOW (0-25), MEDIUM (25-50), HIGH (50-75), CRITICAL (75-100), VIOLATION
- Evidence packaging for Due Diligence Statement (DDS) integration with AGENT-DATA-005
- Satellite imagery provenance (scene ID, sensor, date, cloud cover, resolution)
- Forest cover change summary (baseline vs. current, change percentage, change type)
- Alert summary (total alerts, post-cutoff alerts, high-confidence alerts, affected area)
- Recommendations generation based on compliance status
- Multi-polygon consolidated compliance report
- Temporal analysis summary (trend direction, rate of change)
- Export formats: JSON, structured dict for DDS integration

#### Engine 7: MonitoringPipelineEngine
- 7-stage pipeline orchestration: INITIALIZATION -> IMAGE_ACQUISITION -> INDEX_CALCULATION -> CLASSIFICATION -> CHANGE_DETECTION -> ALERT_INTEGRATION -> REPORT_GENERATION
- Per-polygon progress tracking with stage-level status
- Configurable monitoring schedules (on-demand, weekly, monthly, quarterly)
- Incremental monitoring (only re-analyze changed areas)
- Batch polygon processing with parallel execution
- Pipeline checkpointing for long-running analyses
- Intermediate result caching with configurable TTL
- Confidence scoring across pipeline (data quality, classification quality, sensor consistency)
- Pipeline versioning for reproducibility
- SHA-256 provenance chains on all pipeline outputs

### 5.3 Database Schema

**Schema**: `deforestation_satellite_service`

| Table | Purpose | Type |
|-------|---------|------|
| `satellite_scenes` | Acquired satellite imagery metadata | Regular |
| `vegetation_indices` | Computed vegetation index results | Regular |
| `forest_assessments` | Per-plot forest cover assessments | Regular |
| `deforestation_alerts` | Aggregated deforestation alerts | Hypertable |
| `change_detections` | Forest cover change detection results | Regular |
| `baseline_checks` | EUDR baseline verification results | Regular |
| `compliance_reports` | Generated compliance reports | Regular |
| `monitoring_jobs` | Pipeline monitoring job tracking | Hypertable |
| `classification_results` | Land cover classification outputs | Regular |
| `pipeline_metrics` | Per-pipeline performance metrics | Hypertable |

### 5.4 Prometheus Metrics (12)

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | `gl_deforestation_sat_scenes_acquired_total` | Counter | `satellite`, `status` |
| 2 | `gl_deforestation_sat_acquisition_duration_seconds` | Histogram | `satellite` |
| 3 | `gl_deforestation_sat_change_detections_total` | Counter | `change_type`, `status` |
| 4 | `gl_deforestation_sat_alerts_processed_total` | Counter | `source`, `severity` |
| 5 | `gl_deforestation_sat_baseline_checks_total` | Counter | `country`, `compliance_status` |
| 6 | `gl_deforestation_sat_classifications_total` | Counter | `land_cover_type` |
| 7 | `gl_deforestation_sat_compliance_reports_total` | Counter | `status` |
| 8 | `gl_deforestation_sat_pipeline_runs_total` | Counter | `stage`, `status` |
| 9 | `gl_deforestation_sat_active_monitoring_jobs` | Gauge | - |
| 10 | `gl_deforestation_sat_processing_errors_total` | Counter | `engine`, `error_type` |
| 11 | `gl_deforestation_sat_forest_area_monitored_ha` | Gauge | - |
| 12 | `gl_deforestation_sat_pipeline_duration_seconds` | Histogram | `stage` |

### 5.5 REST API Endpoints (20)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/v1/deforestation/acquire` | Acquire satellite imagery for polygon |
| 2 | POST | `/v1/deforestation/indices` | Calculate vegetation indices |
| 3 | POST | `/v1/deforestation/classify` | Classify land cover from imagery |
| 4 | POST | `/v1/deforestation/detect-change` | Detect forest cover changes |
| 5 | POST | `/v1/deforestation/check-baseline` | Check EUDR baseline compliance |
| 6 | POST | `/v1/deforestation/check-baseline/polygon` | Check baseline for polygon (grid sampling) |
| 7 | POST | `/v1/deforestation/alerts/query` | Query deforestation alerts for polygon |
| 8 | GET | `/v1/deforestation/alerts/{alert_id}` | Get alert details |
| 9 | POST | `/v1/deforestation/alerts/aggregate` | Aggregate alerts from multiple sources |
| 10 | POST | `/v1/deforestation/compliance/assess` | Assess EUDR compliance |
| 11 | POST | `/v1/deforestation/compliance/report` | Generate compliance report |
| 12 | GET | `/v1/deforestation/compliance/{report_id}` | Get compliance report |
| 13 | POST | `/v1/deforestation/monitor/start` | Start monitoring pipeline for polygon |
| 14 | GET | `/v1/deforestation/monitor/{job_id}` | Get monitoring job status |
| 15 | POST | `/v1/deforestation/monitor/{job_id}/stop` | Stop monitoring job |
| 16 | GET | `/v1/deforestation/monitor/jobs` | List monitoring jobs |
| 17 | GET | `/v1/deforestation/scenes` | List acquired satellite scenes |
| 18 | GET | `/v1/deforestation/forest-definitions` | List forest definitions by country |
| 19 | GET | `/v1/deforestation/health` | Health check |
| 20 | GET | `/v1/deforestation/statistics` | Service statistics |

### 5.6 Configuration

**Environment Variable Prefix**: `GL_DEFORESTATION_SAT_`

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_DEFORESTATION_SAT_DATABASE_URL` | `""` | PostgreSQL connection string |
| `GL_DEFORESTATION_SAT_REDIS_URL` | `""` | Redis connection string |
| `GL_DEFORESTATION_SAT_LOG_LEVEL` | `"INFO"` | Logging level |
| `GL_DEFORESTATION_SAT_EUDR_CUTOFF_DATE` | `"2020-12-31"` | EUDR cutoff date (should never change) |
| `GL_DEFORESTATION_SAT_DEFAULT_SATELLITE` | `"sentinel2"` | Primary satellite source |
| `GL_DEFORESTATION_SAT_MAX_CLOUD_COVER` | `30` | Maximum cloud cover % for scene selection |
| `GL_DEFORESTATION_SAT_NDVI_CLEARCUT_THRESHOLD` | `-0.3` | NDVI threshold for clear-cut detection |
| `GL_DEFORESTATION_SAT_NDVI_DEGRADATION_THRESHOLD` | `-0.15` | NDVI threshold for degradation detection |
| `GL_DEFORESTATION_SAT_NDVI_PARTIAL_LOSS_THRESHOLD` | `-0.05` | NDVI threshold for partial loss detection |
| `GL_DEFORESTATION_SAT_NDVI_REGROWTH_THRESHOLD` | `0.1` | NDVI threshold for regrowth detection |
| `GL_DEFORESTATION_SAT_MIN_ALERT_CONFIDENCE` | `"nominal"` | Minimum alert confidence level |
| `GL_DEFORESTATION_SAT_ALERT_DEDUP_RADIUS_M` | `100` | De-duplication radius in meters |
| `GL_DEFORESTATION_SAT_ALERT_DEDUP_DAYS` | `7` | De-duplication temporal window in days |
| `GL_DEFORESTATION_SAT_BASELINE_SAMPLE_POINTS` | `9` | Grid sample points for polygon baseline |
| `GL_DEFORESTATION_SAT_BATCH_SIZE` | `50` | Batch processing size for polygons |
| `GL_DEFORESTATION_SAT_WORKER_COUNT` | `4` | Parallel workers for pipeline |
| `GL_DEFORESTATION_SAT_CACHE_TTL_SECONDS` | `3600` | Scene cache TTL |
| `GL_DEFORESTATION_SAT_POOL_MIN_SIZE` | `2` | DB pool minimum |
| `GL_DEFORESTATION_SAT_POOL_MAX_SIZE` | `10` | DB pool maximum |
| `GL_DEFORESTATION_SAT_RETENTION_DAYS` | `730` | Assessment retention (2 years) |
| `GL_DEFORESTATION_SAT_USE_MOCK` | `true` | Use mock satellite data (dev/test) |
| `GL_DEFORESTATION_SAT_GFW_API_KEY` | `""` | Global Forest Watch API key |
| `GL_DEFORESTATION_SAT_COPERNICUS_API_KEY` | `""` | Copernicus Data Space API key |

## 6. Completion Plan

### Phase 1: SDK Core
1. Build config.py, models.py, __init__.py
2. Build 7 core engines
3. Build provenance.py, metrics.py, setup.py
4. Build api/router.py

### Phase 2: Infrastructure
5. Build V037 database migration
6. Build K8s manifests (8 files)
7. Build CI/CD pipeline
8. Build Grafana dashboard + alerts

### Phase 3: Testing
9. Build 600+ unit tests across 13 test files

## 7. Success Criteria

- [ ] 7 engines with deterministic deforestation assessment
- [ ] 4 satellite sources supported (Sentinel-2, Landsat 8/9, GLAD, RADD)
- [ ] 7 vegetation indices computed (NDVI, EVI, NDWI, NBR, SAVI, MSAVI, NDMI)
- [ ] 5 change detection types (no_change, clear_cut, degradation, partial_loss, regrowth)
- [ ] 10-class land cover classification
- [ ] 8 country-specific forest definitions + FAO default
- [ ] EUDR Dec 31, 2020 cutoff date enforcement
- [ ] 3 compliance statuses (COMPLIANT, REVIEW_REQUIRED, NON_COMPLIANT)
- [ ] 7-stage monitoring pipeline with checkpointing
- [ ] 20 REST API endpoints operational
- [ ] 12 Prometheus metrics instrumented
- [ ] SHA-256 provenance on all operations
- [ ] V037 database migration with 10 tables
- [ ] 600+ tests passing
- [ ] K8s manifests with full security hardening

## 8. Integration Points

### Upstream Dependencies
- AGENT-DATA-005 EUDR Traceability (plot geolocation, DDS evidence)
- AGENT-DATA-006 GIS/Mapping Connector (coordinate transformation, spatial analysis)
- AGENT-FOUND-002 Schema Compiler (schema validation)
- AGENT-FOUND-006 Access Guard (authorization)
- AGENT-FOUND-010 Observability Agent (metrics/tracing)

### Downstream Consumers
- GL-EUDR-APP DeforestationRiskAgent (satellite deforestation data)
- EUDR Due Diligence Statements (compliance evidence)
- Carbon credit verification (forest cover assessment)
- Deforestation monitoring dashboards (alert visualization)
- Supply chain risk assessment (sourcing region deforestation risk)

### Layer 1 Foundation Integration
- `greenlang/extensions/satellite/clients/` - Sentinel-2 and Landsat clients
- `greenlang/extensions/satellite/analysis/` - Vegetation indices, change detection
- `greenlang/extensions/satellite/models/` - Forest classifier
- `greenlang/extensions/satellite/alerts/` - Deforestation alert system
- `greenlang/extensions/satellite/pipeline/` - Analysis pipeline
- `greenlang/governance/validation/geolocation/deforestation_baseline.py` - Baseline checker
