# PRD-AGENT-EUDR-003: Satellite Monitoring Agent

**Document ID:** PRD-AGENT-EUDR-003
**Version:** 1.0.0
**Date:** March 7, 2026
**Author:** GreenLang Platform Team
**Status:** APPROVED
**Agent ID:** GL-EUDR-SAT-003
**Category:** EUDR Agents (Category 5)
**Regulation:** EU 2023/1115 (EUDR) - Articles 9, 10, 12
**Priority:** P0 (Critical Path for December 30, 2025 enforcement)

---

## 1. Executive Summary

### 1.1 Purpose

The Satellite Monitoring Agent provides continuous, multi-source satellite imagery analysis for verifying deforestation-free compliance under the EU Deforestation Regulation (EUDR). It establishes baselines at the December 31, 2020 cutoff date, monitors forest cover changes using Sentinel-2, Landsat 8/9, Sentinel-1 SAR, and Global Forest Watch alert data, and generates regulatory-grade evidence packages for Due Diligence Statements (DDS).

### 1.2 Relationship to Other EUDR Agents

| Agent | Function | Integration |
|-------|----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping | Provides plot geometries and commodity types for monitoring |
| AGENT-EUDR-002 | Geolocation Verification | Provides coordinate/polygon validation; consumes satellite deforestation results |
| **AGENT-EUDR-003** | **Satellite Monitoring** | **Core satellite imagery analysis, change detection, evidence generation** |
| AGENT-DATA-007 | Deforestation Satellite Connector | Low-level data connector; EUDR-003 orchestrates higher-level analysis |

### 1.3 Key Metrics

| Metric | Target |
|--------|--------|
| Deforestation detection accuracy | >90% F1 Score |
| False positive rate | <5% |
| False negative rate | <3% |
| Single plot processing time | <30s (STANDARD), <120s (DEEP) |
| Batch throughput | 10,000+ plots/hour |
| Temporal coverage | 2018-present |
| Spatial resolution | 10m (Sentinel-2) / 30m (Landsat) |
| Cloud gap recovery | >85% tropical coverage |
| Baseline establishment success | >95% of plots |

---

## 2. Regulatory Requirements

### 2.1 EUDR Article 9 - Geolocation Requirements

- Article 9(1)(d): Operators must provide "adequate and verifiable" information that products are deforestation-free
- Article 9(1)(a-c): Geolocation coordinates and polygon boundaries for production plots
- The satellite monitoring agent provides the **verification evidence** that plots were forest-free or have not experienced deforestation since December 31, 2020

### 2.2 EUDR Article 10 - Due Diligence Statement

- Article 10(2)(f): DDS must include risk assessment procedures
- Article 10(3): Risk mitigation must include "independent auditing and verification through satellite monitoring"
- The agent generates structured evidence packages for DDS attachment

### 2.3 EUDR Article 3 - Prohibition

- Article 3(a): Products must be "deforestation-free" - produced on land not subject to deforestation after December 31, 2020
- Article 3(b): Products must be "produced in accordance with the relevant legislation of the country of production"

### 2.4 Deforestation Definition

Per EUDR Article 2(3): "Deforestation" means the conversion of forest to agricultural use, whether human-induced or not. "Forest" means land spanning more than 0.5 hectares with trees higher than 5 meters and a canopy cover of more than 10%, or trees able to reach those thresholds in situ.

### 2.5 Cutoff Date

**December 31, 2020** - All EUDR-regulated commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood) must be produced on land that was not deforested after this date.

---

## 3. Functional Requirements

### 3.1 P0 Features (Must Have)

#### F-001: Satellite Imagery Acquisition Engine
- Query and retrieve imagery from Sentinel-2 (10m), Landsat 8/9 (30m), Sentinel-1 SAR
- Filter by cloud cover (<20% preferred, <50% acceptable), date range, spatial coverage
- Support OAuth2 authentication for Copernicus Data Space
- Manage API rate limiting and retry with exponential backoff
- Cache downloaded scenes to avoid re-downloads

#### F-002: Spectral Index Calculator
- Calculate NDVI (Normalized Difference Vegetation Index) from Red + NIR bands
- Calculate EVI (Enhanced Vegetation Index) for dense canopy correction
- Calculate NBR (Normalized Burn Ratio) for fire/burn scar detection
- Calculate NDMI (Normalized Difference Moisture Index) for water stress
- Calculate SAVI (Soil-Adjusted Vegetation Index) for sparse canopy
- All calculations deterministic with Decimal arithmetic where regulatory-grade precision required

#### F-003: Baseline Establishment Manager
- Establish forest cover baseline at December 31, 2020 for each production plot
- Select best available imagery within +/-90 days of cutoff date
- Generate forest/non-forest classification mask using NDVI thresholds
- Calculate baseline forest area (hectares), forest percentage, canopy density
- Store immutable baseline snapshots with SHA-256 provenance hashing
- Support re-baseline with full audit trail

#### F-004: Forest Change Detection Engine
- Detect deforestation between baseline and current imagery
- Support multiple detection methods: NDVI differencing, spectral angle mapping, time-series break detection (BFAST)
- Classify changes: No Change, Deforestation, Degradation, Reforestation, Regrowth
- Calculate change area in hectares with pixel-level accuracy
- Generate change maps and change probability surfaces
- Minimum detectable change: 0.1 hectares

#### F-005: Multi-Source Data Fusion Engine
- Fuse Sentinel-2 (weight: 0.50), Landsat (0.30), and GFW alerts (0.20)
- Weighted consensus for deforestation detection
- Agreement scoring across sources
- Confidence calibration using validation datasets
- Fallback strategy when primary sources unavailable

#### F-006: Cloud Gap Filling Pipeline
- Detect and mask cloud cover in optical imagery
- Fill gaps using: temporal compositing (median pixel), Sentinel-1 SAR backscatter, multi-temporal interpolation
- Support cloud-persistent tropical regions (Amazon, Congo Basin, Southeast Asia)
- Track data quality degradation from gap filling

#### F-007: Continuous Monitoring Pipeline
- Schedule recurring satellite analysis for all active plots
- Configurable monitoring intervals: weekly, biweekly, monthly, quarterly
- Automated alert generation when changes detected
- Track monitoring history with complete temporal coverage
- Support monitoring of 100,000+ plots at scale

#### F-008: Alert Generation & Evidence Packaging
- Generate structured alerts when deforestation detected
- Alert severity: CRITICAL (confirmed deforestation), WARNING (potential/degradation), INFO (minor change)
- Package satellite evidence for DDS: before/after imagery metadata, NDVI time series, change maps, confidence scores
- Export evidence in JSON, PDF-ready, and EUDR XML formats
- Include provenance chain for audit trail

### 3.2 P1 Features (Should Have)

#### F-009: SAR-Based Cloud-Free Monitoring
- Process Sentinel-1 SAR data (VV/VH polarization)
- SAR change detection using backscatter coefficient thresholds
- SAR-optical data fusion for enhanced accuracy in cloudy regions

#### F-010: Seasonal Adjustment
- Account for phenological cycles (wet/dry season vegetation changes)
- Region-specific seasonal NDVI baselines for 25+ EUDR-relevant countries
- Avoid false positives from natural seasonal variation

### 3.3 P2 Features (Nice to Have)

#### F-011: ML-Enhanced Change Detection
- U-Net semantic segmentation model for forest/non-forest classification
- Transfer learning from pre-trained models (DeepForest, TensorFlow)
- Active learning pipeline for model improvement

---

## 4. Technical Architecture

### 4.1 Module Structure

```
greenlang/agents/eudr/satellite_monitoring/
    __init__.py                          # Package exports
    config.py                            # SatelliteMonitoringConfig (GL_EUDR_SAT_ prefix)
    models.py                            # Pydantic v2 models (40+ types)
    provenance.py                        # SHA-256 chain-hashed audit trail
    metrics.py                           # 18 Prometheus metrics (gl_eudr_sat_ prefix)

    # Core Engines
    imagery_acquisition.py               # Satellite scene query, download, caching
    spectral_index_calculator.py         # NDVI, EVI, NBR, NDMI, SAVI calculations
    baseline_manager.py                  # Dec 31, 2020 baseline establishment
    forest_change_detector.py            # Multi-method change detection
    data_fusion_engine.py                # Multi-source weighted fusion
    cloud_gap_filler.py                  # Cloud masking and gap filling
    continuous_monitor.py                # Scheduled monitoring pipeline
    alert_generator.py                   # Alert creation and evidence packaging
    setup.py                             # SatelliteMonitoringService facade

    # Reference Data
    reference_data/
        __init__.py
        forest_thresholds.py             # NDVI/EVI thresholds per biome
        seasonal_baselines.py            # Phenological profiles per region
        satellite_specs.py               # Sensor band specifications

    # API Layer
    api/
        __init__.py
        router.py                        # Main FastAPI router
        schemas.py                       # API request/response models
        dependencies.py                  # DI for engines, auth, rate limiting
        imagery_routes.py                # Scene search and download endpoints
        analysis_routes.py               # NDVI/change detection endpoints
        monitoring_routes.py             # Continuous monitoring CRUD
        alert_routes.py                  # Alert management endpoints
        evidence_routes.py               # Evidence package endpoints
        batch_routes.py                  # Batch analysis endpoints
```

### 4.2 Database Tables (V091)

```sql
-- Core satellite monitoring tables
eudr_satellite_scenes             -- Downloaded/cached scene metadata
eudr_plot_baselines               -- Dec 31, 2020 baselines per plot (hypertable)
eudr_forest_change_events         -- Detected change events (hypertable)
eudr_monitoring_schedules         -- Active monitoring configurations
eudr_monitoring_results           -- Per-execution monitoring results (hypertable)
eudr_satellite_alerts             -- Generated alerts (hypertable)
eudr_evidence_packages            -- Packaged evidence for DDS
eudr_cloud_cover_log              -- Cloud coverage tracking (hypertable)
eudr_data_quality_log             -- Data quality metrics per analysis
eudr_satellite_audit_log          -- Immutable audit trail
```

### 4.3 Prometheus Metrics (18 metrics, `gl_eudr_sat_` prefix)

| # | Metric Name | Type | Labels |
|---|-------------|------|--------|
| 1 | `scenes_queried_total` | Counter | source, status |
| 2 | `scenes_downloaded_total` | Counter | source |
| 3 | `imagery_download_bytes_total` | Counter | source |
| 4 | `baselines_established_total` | Counter | commodity, country |
| 5 | `ndvi_calculations_total` | Counter | index_type |
| 6 | `change_detections_total` | Counter | method, result |
| 7 | `deforestation_detected_total` | Counter | commodity, country, severity |
| 8 | `alerts_generated_total` | Counter | severity |
| 9 | `evidence_packages_total` | Counter | format |
| 10 | `monitoring_executions_total` | Counter | status |
| 11 | `cloud_gap_fills_total` | Counter | method |
| 12 | `fusion_analyses_total` | Counter | sources_count |
| 13 | `analysis_duration_seconds` | Histogram | operation |
| 14 | `batch_duration_seconds` | Histogram | - |
| 15 | `active_monitoring_plots` | Gauge | - |
| 16 | `avg_detection_confidence` | Gauge | - |
| 17 | `api_errors_total` | Counter | operation |
| 18 | `data_quality_score` | Gauge | source |

### 4.4 API Endpoints (24 endpoints)

```
# Imagery
POST   /api/v1/eudr-sat/imagery/search          # Search available scenes
POST   /api/v1/eudr-sat/imagery/download         # Download scene bands
GET    /api/v1/eudr-sat/imagery/{scene_id}       # Get scene metadata
GET    /api/v1/eudr-sat/imagery/availability      # Check data availability for plot

# Analysis
POST   /api/v1/eudr-sat/analysis/ndvi            # Calculate spectral indices
POST   /api/v1/eudr-sat/analysis/baseline        # Establish Dec 2020 baseline
GET    /api/v1/eudr-sat/analysis/baseline/{plot_id}  # Get stored baseline
POST   /api/v1/eudr-sat/analysis/change-detect   # Run change detection
POST   /api/v1/eudr-sat/analysis/fusion          # Multi-source data fusion
GET    /api/v1/eudr-sat/analysis/history/{plot_id}   # Analysis history

# Monitoring
POST   /api/v1/eudr-sat/monitoring/schedule      # Create monitoring schedule
GET    /api/v1/eudr-sat/monitoring/schedule/{schedule_id}  # Get schedule
PUT    /api/v1/eudr-sat/monitoring/schedule/{schedule_id}  # Update schedule
DELETE /api/v1/eudr-sat/monitoring/schedule/{schedule_id}  # Delete schedule
GET    /api/v1/eudr-sat/monitoring/results/{plot_id}  # Get monitoring results
POST   /api/v1/eudr-sat/monitoring/execute       # Trigger manual execution

# Alerts
GET    /api/v1/eudr-sat/alerts                   # List alerts (paginated)
GET    /api/v1/eudr-sat/alerts/{alert_id}        # Get alert details
PUT    /api/v1/eudr-sat/alerts/{alert_id}/acknowledge  # Acknowledge alert
GET    /api/v1/eudr-sat/alerts/summary           # Alert summary statistics

# Evidence
POST   /api/v1/eudr-sat/evidence/package         # Generate evidence package
GET    /api/v1/eudr-sat/evidence/{package_id}    # Retrieve evidence package
GET    /api/v1/eudr-sat/evidence/{package_id}/download  # Download evidence

# System
GET    /api/v1/eudr-sat/health                   # Health check
```

---

## 5. Data Models

### 5.1 Enumerations

```python
class SatelliteSource(str, Enum):
    SENTINEL_2 = "sentinel_2"
    LANDSAT_8 = "landsat_8"
    LANDSAT_9 = "landsat_9"
    SENTINEL_1_SAR = "sentinel_1_sar"
    GFW_ALERTS = "gfw_alerts"

class SpectralIndex(str, Enum):
    NDVI = "ndvi"          # (NIR - Red) / (NIR + Red)
    EVI = "evi"            # 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    NBR = "nbr"            # (NIR - SWIR2) / (NIR + SWIR2)
    NDMI = "ndmi"          # (NIR - SWIR1) / (NIR + SWIR1)
    SAVI = "savi"          # 1.5 * (NIR - Red) / (NIR + Red + 0.5)

class ForestClassification(str, Enum):
    DENSE_FOREST = "dense_forest"       # NDVI > 0.6
    FOREST_WOODLAND = "forest_woodland" # NDVI 0.4-0.6
    SHRUBLAND = "shrubland"             # NDVI 0.2-0.4
    SPARSE_VEGETATION = "sparse"        # NDVI 0.0-0.2
    NON_VEGETATION = "non_vegetation"   # NDVI < 0.0

class ChangeClassification(str, Enum):
    NO_CHANGE = "no_change"
    DEFORESTATION = "deforestation"
    DEGRADATION = "degradation"
    REFORESTATION = "reforestation"
    REGROWTH = "regrowth"

class DetectionMethod(str, Enum):
    NDVI_DIFFERENCING = "ndvi_differencing"
    SPECTRAL_ANGLE = "spectral_angle"
    TIME_SERIES_BREAK = "time_series_break"
    MULTI_SOURCE_FUSION = "multi_source_fusion"
    SAR_BACKSCATTER = "sar_backscatter"

class AlertSeverity(str, Enum):
    CRITICAL = "critical"    # Confirmed deforestation
    WARNING = "warning"      # Potential/degradation
    INFO = "info"            # Minor change, monitoring

class MonitoringInterval(str, Enum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class EvidenceFormat(str, Enum):
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    EUDR_XML = "eudr_xml"

class CloudFillMethod(str, Enum):
    TEMPORAL_COMPOSITE = "temporal_composite"   # Median pixel from multi-date
    SAR_FUSION = "sar_fusion"                   # Sentinel-1 SAR backscatter
    INTERPOLATION = "interpolation"             # Temporal interpolation
    NEAREST_CLEAR = "nearest_clear"             # Nearest cloud-free date
```

### 5.2 Core Models

```python
class SceneMetadata(BaseModel):
    scene_id: str
    source: SatelliteSource
    acquisition_date: date
    cloud_cover_pct: float
    spatial_coverage_pct: float
    footprint_wkt: str
    bands_available: List[str]
    quality_score: float  # 0-100

class BaselineSnapshot(BaseModel):
    plot_id: str
    baseline_date: date  # Target: 2020-12-31
    actual_imagery_date: date
    forest_area_ha: float
    forest_percentage: float
    ndvi_mean: float
    ndvi_min: float
    ndvi_max: float
    canopy_density_class: ForestClassification
    confidence: float
    source: SatelliteSource
    scene_id: str
    provenance_hash: str

class ChangeDetectionResult(BaseModel):
    plot_id: str
    baseline_date: date
    analysis_date: date
    method: DetectionMethod
    classification: ChangeClassification
    change_area_ha: float
    change_percentage: float
    ndvi_baseline: float
    ndvi_current: float
    ndvi_difference: float
    confidence: float
    deforestation_detected: bool
    sources_used: List[SatelliteSource]
    agreement_score: float
    evidence_id: Optional[str]
    provenance_hash: str

class SatelliteAlert(BaseModel):
    alert_id: str
    plot_id: str
    severity: AlertSeverity
    classification: ChangeClassification
    change_area_ha: float
    confidence: float
    detected_date: datetime
    imagery_date: date
    source: SatelliteSource
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]

class EvidencePackage(BaseModel):
    package_id: str
    plot_id: str
    operator_id: str
    created_at: datetime
    baseline_snapshot: BaselineSnapshot
    latest_analysis: ChangeDetectionResult
    ndvi_time_series: List[Dict[str, Any]]
    alerts: List[SatelliteAlert]
    data_quality: DataQualityAssessment
    compliance_determination: str  # COMPLIANT / NON_COMPLIANT / INSUFFICIENT_DATA / MANUAL_REVIEW
    format: EvidenceFormat
    provenance_hash: str

class DataQualityAssessment(BaseModel):
    cloud_cover_pct: float
    temporal_proximity_days: int
    spatial_coverage_pct: float
    atmospheric_quality: str  # Good / Moderate / Poor
    sensor_quality: str  # Good / Degraded
    gap_fill_percentage: float
    sources_count: int
    overall_score: float  # 0-100
```

---

## 6. Core Engine Specifications

### 6.1 Imagery Acquisition Engine

**Purpose:** Query, retrieve, and cache satellite imagery from multiple providers.

**Key Methods:**
- `search_scenes(polygon, date_range, source, cloud_cover_max)` -> `List[SceneMetadata]`
- `download_bands(scene_id, bands)` -> `Dict[str, ndarray]` (simulated with reference data)
- `assess_scene_quality(scene, target_date, polygon)` -> `DataQualityAssessment`
- `get_best_scene(scenes, target_date)` -> `Optional[SceneMetadata]`
- `check_availability(polygon, date_range)` -> `AvailabilityReport`

**Data Sources:**
- Copernicus Data Space Ecosystem (Sentinel-2 L2A)
- USGS EarthExplorer M2M API (Landsat 8/9 Collection 2)
- Global Forest Watch API (GLAD/RADD alerts, Hansen GFC)
- Copernicus Sentinel-1 SAR

### 6.2 Spectral Index Calculator

**Purpose:** Calculate vegetation and change indices from multispectral bands.

**Indices:**
| Index | Formula | Use Case |
|-------|---------|----------|
| NDVI | (NIR - Red) / (NIR + Red) | Forest cover |
| EVI | 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1) | Dense canopy correction |
| NBR | (NIR - SWIR2) / (NIR + SWIR2) | Burn/fire detection |
| NDMI | (NIR - SWIR1) / (NIR + SWIR1) | Moisture/drought stress |
| SAVI | 1.5 * (NIR - Red) / (NIR + Red + 0.5) | Sparse canopy |

**Forest Classification Thresholds (biome-adjusted):**
| Biome | Dense Forest | Forest | Shrubland | Sparse |
|-------|-------------|--------|-----------|--------|
| Tropical rainforest | >0.65 | 0.45-0.65 | 0.25-0.45 | 0.05-0.25 |
| Tropical dry forest | >0.55 | 0.35-0.55 | 0.20-0.35 | 0.05-0.20 |
| Temperate forest | >0.60 | 0.40-0.60 | 0.20-0.40 | 0.05-0.20 |
| Boreal forest | >0.50 | 0.30-0.50 | 0.15-0.30 | 0.05-0.15 |
| Mangrove | >0.50 | 0.30-0.50 | 0.15-0.30 | 0.00-0.15 |
| Cerrado/Savanna | >0.45 | 0.25-0.45 | 0.15-0.25 | 0.05-0.15 |

### 6.3 Baseline Manager

**Purpose:** Establish and manage the December 31, 2020 forest cover baseline.

**Algorithm:**
1. Query imagery within +/-90 days of cutoff date (Oct 1, 2020 - Mar 31, 2021)
2. Select best scene (closest to cutoff, lowest cloud cover, highest coverage)
3. Calculate NDVI for the plot polygon
4. Generate forest mask using biome-specific thresholds
5. Calculate forest area (ha), percentage, canopy density
6. Store immutable baseline with provenance hash
7. Flag plots where no suitable imagery available (data quality < 50)

### 6.4 Forest Change Detector

**Purpose:** Detect and classify forest cover changes between baseline and current dates.

**Detection Methods:**

1. **NDVI Differencing:**
   - delta_NDVI = NDVI_current - NDVI_baseline
   - Deforestation threshold: delta_NDVI < -0.15
   - Degradation threshold: -0.15 < delta_NDVI < -0.05
   - Regrowth threshold: delta_NDVI > 0.10

2. **Spectral Angle Mapping:**
   - Calculate spectral angle between baseline and current reflectance vectors
   - Larger angles indicate greater change
   - Threshold: >15 degrees = significant change

3. **Time-Series Break Detection (BFAST-lite):**
   - Analyze NDVI time series for structural breaks
   - Detect abrupt changes (deforestation) vs. gradual (degradation)
   - Use harmonic model to separate seasonal from trend

### 6.5 Data Fusion Engine

**Purpose:** Combine results from multiple satellite sources for robust detection.

**Source Weights:**
| Source | Weight | Rationale |
|--------|--------|-----------|
| Sentinel-2 | 0.50 | Highest resolution (10m), best spectral coverage |
| Landsat 8/9 | 0.30 | Longer archive, independent verification |
| GFW Alerts | 0.20 | Pre-processed, peer-reviewed, near-real-time |

**Consensus Logic:**
- Weighted deforestation score > 0.5 -> `DEFORESTATION_DETECTED`
- Weighted score 0.2-0.5 -> `POTENTIAL_DEFORESTATION` (manual review)
- Weighted score < 0.2 -> `NO_DEFORESTATION`
- Agreement score = 1 - std_dev(per-source binary detection)

### 6.6 Cloud Gap Filler

**Purpose:** Handle cloud-obscured imagery common in tropical EUDR regions.

**Methods:**
1. **Temporal Composite:** Median pixel from 30-day window of multi-date imagery
2. **SAR Fusion:** Use Sentinel-1 VV/VH backscatter to classify forest/non-forest through clouds
3. **Interpolation:** Linear/cubic interpolation of NDVI time series across cloudy dates
4. **Nearest Clear:** Use nearest cloud-free date observation

**Cloud Persistence Regions (requiring SAR fallback):**
- Amazon Basin (Jun-Nov wet season, >70% cloud cover)
- Congo Basin (Sep-Dec, >65% cloud cover)
- Borneo/Sumatra (Nov-Mar, >60% cloud cover)

### 6.7 Continuous Monitor

**Purpose:** Schedule and execute recurring satellite analysis for registered plots.

**Features:**
- CRUD for monitoring schedules (create, read, update, delete)
- Configurable intervals: weekly, biweekly, monthly, quarterly
- Priority queue based on risk level and last analysis date
- Batch execution with configurable concurrency
- Status tracking: ACTIVE, PAUSED, COMPLETED, ERROR

### 6.8 Alert Generator

**Purpose:** Create and manage alerts and evidence packages.

**Alert Rules:**
- NDVI drop > 0.15 + confidence > 0.7 -> CRITICAL alert
- NDVI drop 0.05-0.15 + confidence > 0.5 -> WARNING alert
- Any detectable change -> INFO alert (monitoring)

**Evidence Package Contents:**
- Baseline snapshot (forest area, NDVI, date, source)
- Latest analysis result (change area, classification, confidence)
- NDVI time series (all available observations)
- Change map metadata
- Alert history for the plot
- Data quality assessment
- Compliance determination (COMPLIANT / NON_COMPLIANT / INSUFFICIENT_DATA / MANUAL_REVIEW)
- SHA-256 provenance hash chain

---

## 7. Configuration

### 7.1 Environment Variables (GL_EUDR_SAT_ prefix)

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_EUDR_SAT_SENTINEL2_CLIENT_ID` | "" | Copernicus Data Space client ID |
| `GL_EUDR_SAT_SENTINEL2_CLIENT_SECRET` | "" | Copernicus Data Space client secret |
| `GL_EUDR_SAT_LANDSAT_API_KEY` | "" | USGS M2M API key |
| `GL_EUDR_SAT_GFW_API_KEY` | "" | Global Forest Watch API key |
| `GL_EUDR_SAT_CUTOFF_DATE` | "2020-12-31" | EUDR deforestation cutoff |
| `GL_EUDR_SAT_BASELINE_WINDOW_DAYS` | 90 | Days before/after cutoff for baseline imagery |
| `GL_EUDR_SAT_CLOUD_COVER_MAX` | 20.0 | Maximum acceptable cloud cover (%) |
| `GL_EUDR_SAT_CLOUD_COVER_ABSOLUTE_MAX` | 50.0 | Absolute max cloud cover before rejection |
| `GL_EUDR_SAT_NDVI_DEFORESTATION_THRESHOLD` | -0.15 | NDVI drop threshold for deforestation |
| `GL_EUDR_SAT_NDVI_DEGRADATION_THRESHOLD` | -0.05 | NDVI drop threshold for degradation |
| `GL_EUDR_SAT_REGROWTH_THRESHOLD` | 0.10 | NDVI increase for regrowth |
| `GL_EUDR_SAT_MIN_CHANGE_AREA_HA` | 0.1 | Minimum detectable change area |
| `GL_EUDR_SAT_SENTINEL2_WEIGHT` | 0.50 | Sentinel-2 fusion weight |
| `GL_EUDR_SAT_LANDSAT_WEIGHT` | 0.30 | Landsat fusion weight |
| `GL_EUDR_SAT_GFW_WEIGHT` | 0.20 | GFW fusion weight |
| `GL_EUDR_SAT_MONITORING_MAX_CONCURRENCY` | 50 | Max concurrent monitoring jobs |
| `GL_EUDR_SAT_CACHE_TTL_SECONDS` | 86400 | Scene cache TTL (1 day) |
| `GL_EUDR_SAT_BASELINE_CACHE_TTL_SECONDS` | 7776000 | Baseline cache TTL (90 days) |
| `GL_EUDR_SAT_QUICK_TIMEOUT_SECONDS` | 10.0 | Quick analysis timeout |
| `GL_EUDR_SAT_STANDARD_TIMEOUT_SECONDS` | 30.0 | Standard analysis timeout |
| `GL_EUDR_SAT_DEEP_TIMEOUT_SECONDS` | 120.0 | Deep analysis timeout |
| `GL_EUDR_SAT_MAX_BATCH_SIZE` | 10000 | Maximum plots per batch |
| `GL_EUDR_SAT_ALERT_CONFIDENCE_THRESHOLD` | 0.7 | Min confidence for CRITICAL alert |
| `GL_EUDR_SAT_SEASONAL_ADJUSTMENT_ENABLED` | true | Enable seasonal NDVI adjustment |
| `GL_EUDR_SAT_SAR_ENABLED` | true | Enable Sentinel-1 SAR processing |

---

## 8. Testing Strategy

### 8.1 Test Coverage Targets

| Category | Tests | Coverage |
|----------|-------|----------|
| Unit Tests | 500+ | 85%+ |
| Parametrized Data Points | 800+ | All edge cases |
| Integration Tests | 50+ | Cross-engine |
| Determinism Tests | 30+ | Provenance hash reproducibility |

### 8.2 Test Files

```
tests/agents/eudr/satellite_monitoring/
    __init__.py
    conftest.py                           # Shared fixtures
    test_imagery_acquisition.py           # Scene search, download, quality
    test_spectral_index_calculator.py     # NDVI/EVI/NBR/NDMI/SAVI
    test_baseline_manager.py              # Baseline establishment
    test_forest_change_detector.py        # Change detection methods
    test_data_fusion_engine.py            # Multi-source fusion
    test_cloud_gap_filler.py             # Cloud masking, gap filling
    test_alert_generator.py              # Alert creation, evidence
    test_models.py                        # Model validation
    test_setup.py                         # Service facade, config
```

---

## 9. Integration Points

### 9.1 AGENT-EUDR-001 (Supply Chain Mapper)
- **Consumes:** Plot geometries, commodity types, country codes from supply chain graph
- **Provides:** Satellite monitoring status, deforestation risk scores per plot

### 9.2 AGENT-EUDR-002 (Geolocation Verification)
- **Consumes:** Validated coordinates and polygons
- **Provides:** Satellite-based deforestation verification evidence, NDVI time series

### 9.3 AGENT-DATA-007 (Deforestation Satellite Connector)
- **Uses as dependency:** Low-level satellite API client wrappers
- **Adds value:** Higher-level orchestration, multi-source fusion, regulatory evidence packaging

### 9.4 GL-EUDR-APP
- **Provides:** Satellite monitoring API endpoints for the EUDR compliance platform
- **Consumes:** DDS data for evidence package generation

---

## 10. RBAC Permissions

### 10.1 Permission Definitions (16 permissions)

| Permission | Description |
|------------|-------------|
| `eudr-sat:imagery:search` | Search satellite scenes |
| `eudr-sat:imagery:download` | Download scene bands |
| `eudr-sat:imagery:read` | View scene metadata |
| `eudr-sat:analysis:create` | Run NDVI/change analysis |
| `eudr-sat:analysis:read` | View analysis results |
| `eudr-sat:baseline:create` | Establish new baselines |
| `eudr-sat:baseline:read` | View baselines |
| `eudr-sat:monitoring:create` | Create monitoring schedules |
| `eudr-sat:monitoring:read` | View monitoring results |
| `eudr-sat:monitoring:update` | Modify schedules |
| `eudr-sat:monitoring:delete` | Delete schedules |
| `eudr-sat:alerts:read` | View alerts |
| `eudr-sat:alerts:acknowledge` | Acknowledge alerts |
| `eudr-sat:evidence:create` | Generate evidence packages |
| `eudr-sat:evidence:read` | View evidence packages |
| `eudr-sat:evidence:download` | Download evidence |

### 10.2 Role Mappings

| Role | Permissions |
|------|-------------|
| `auditor` | `read` permissions only (imagery, analysis, baseline, monitoring, alerts, evidence) |
| `compliance_officer` | All 16 permissions |
| `supply_chain_analyst` | Search/read + analysis + monitoring read + alerts read |
| `procurement_manager` | Read + alerts read + evidence read/download |
| `admin` | All permissions (inherited) |

---

## 11. Deployment

### 11.1 Database Migration: V091

- 10 tables with TimescaleDB hypertables for time-series data
- Continuous aggregates for daily/weekly monitoring statistics
- 5-year retention per EUDR Article 31
- Indexes for plot_id, date ranges, source, severity

### 11.2 Grafana Dashboard

- Sections: Overview, Imagery Acquisition, Analysis Pipeline, Monitoring, Alerts, Data Quality
- Key panels: Active plots, detection rate, confidence distribution, cloud cover trends, alert timeline

---

## 12. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-07 | GreenLang Platform Team | Initial specification |

---

**END OF PRD**
