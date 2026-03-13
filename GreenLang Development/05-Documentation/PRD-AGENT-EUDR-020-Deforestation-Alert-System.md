# PRD: AGENT-EUDR-020 -- Deforestation Alert System

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-020 |
| **Agent ID** | GL-EUDR-DAS-020 |
| **Component** | Deforestation Alert System Agent |
| **Category** | EUDR Regulatory Agent -- Real-Time Deforestation Intelligence |
| **Priority** | P0 -- Critical (EUDR Satellite Monitoring Dependency) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-10 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) establishes a December 31, 2020 cutoff date (Article 2(1)) after which no commodity products linked to deforestation may be placed on or exported from the EU market. Operators and traders must demonstrate through due diligence statements that their supply chain plots have not been subject to deforestation or forest degradation after this cutoff date. Articles 9, 10, and 11 require continuous monitoring and risk assessment using satellite-derived geolocation data.

The fundamental challenge is that deforestation is a spatially and temporally dynamic phenomenon. Forest loss can occur rapidly (slash-and-burn clearing can eliminate hectares within days), detection requires multi-source satellite monitoring at varying resolutions and revisit frequencies, and the compliance determination depends on whether the event occurred before or after the EUDR cutoff date. Today, operators face critical monitoring gaps:

- **No automated satellite change detection**: Operators rely on manual review of satellite imagery or third-party reports with weeks-long delays, missing rapid deforestation events that affect their supply chain plots.
- **No multi-source data fusion**: Sentinel-2 provides 10m resolution at 5-day revisit, Landsat 8/9 provides 30m at 8/16-day revisit, GLAD provides weekly alerts, Hansen GFC provides annual data, and RADD provides SAR-based alerts penetrating cloud cover. No system combines these complementary sources for comprehensive detection.
- **No spectral index analysis pipeline**: NDVI, EVI, NBR, NDMI, and SAVI indices each capture different aspects of vegetation health and change. Single-index monitoring misses degradation patterns visible only through multi-index analysis.
- **No spatial proximity monitoring**: Deforestation occurring near (but not directly on) a supply chain plot can indicate expanding clearing pressure, illegal encroachment, or fire risk that threatens plot integrity. Operators have no buffer zone monitoring capability.
- **No EUDR cutoff date verification**: Determining whether a detected forest change occurred before or after December 31, 2020 requires multi-temporal evidence from historical satellite archives. Operators cannot systematically perform this temporal classification.
- **No historical baseline comparison**: Understanding whether current forest cover on a plot represents degradation from the 2018-2020 reference period requires baseline establishment and ongoing comparison that operators lack tooling for.
- **No alert workflow management**: When deforestation is detected, there is no structured process for triage, investigation, resolution, or escalation with SLA tracking to ensure timely compliance response.
- **No compliance impact mapping**: Even when deforestation events are detected, operators cannot systematically map these events to affected supply chain commodities, products, suppliers, market restrictions, and required remediation actions.

Without solving these problems, operators risk failing EUDR due diligence obligations, facing penalties of up to 4% of annual EU turnover, goods confiscation at EU borders, mandatory market withdrawal, and reputational damage from association with deforestation-linked supply chains.

### 1.2 Solution Overview

Agent-EUDR-020: Deforestation Alert System is a production-grade real-time satellite monitoring and alerting system for deforestation events affecting EUDR-regulated supply chain plots. The agent integrates five satellite data sources, performs multi-spectral change detection, maintains spatial buffer zones around production plots, verifies EUDR cutoff date compliance through multi-temporal evidence analysis, establishes historical forest baselines for ongoing comparison, manages alert workflows with SLA enforcement, and maps deforestation events to concrete supply chain compliance impacts.

Core capabilities:

1. **Multi-Source Satellite Change Detection** -- Integrate Sentinel-2 (10m, 5-day), Landsat 8/9 (30m, 8-day), GLAD weekly alerts, Hansen Global Forest Change annual data, and RADD SAR alerts. Compute NDVI, EVI, NBR, NDMI, and SAVI spectral indices. Apply configurable confidence thresholds (0.75 default), cloud cover filtering (20% max), and minimum clearing area detection (0.5 ha).

2. **Alert Generation with Deduplication** -- Generate deforestation alerts from detected changes with batch processing (1000/batch), real-time streaming support, 72-hour deduplication window to prevent duplicate alerts for the same event, daily cap (10,000), and 5-year retention per EUDR Article 31.

3. **Five-Tier Severity Classification** -- Classify alerts using weighted scoring across five dimensions: affected area (25% weight, critical >= 50 ha), deforestation rate (20%), proximity to supply chain plots (25%, critical <= 1 km), protected area overlay (15%, 1.5x multiplier), and post-cutoff timing (15%, 2.0x multiplier). Severity levels: CRITICAL (>= 80), HIGH (>= 60), MEDIUM (>= 40), LOW (>= 20), INFORMATIONAL (< 20).

4. **Spatial Buffer Zone Monitoring** -- Monitor circular, polygon, and adaptive buffer zones around supply chain plots with configurable radii (1-50 km, default 10 km) at 64-point resolution. Detect proximity violations using Haversine distance calculations and ray-casting point-in-polygon tests.

5. **EUDR Cutoff Date Verification** -- Verify whether detected deforestation occurred before or after December 31, 2020 using multi-source temporal evidence with minimum 2 evidence sources, 90-day grace period handling, and 0.85 confidence threshold. Classify as PRE_CUTOFF (compliant), POST_CUTOFF (non-compliant), WITHIN_GRACE_PERIOD, or INDETERMINATE.

6. **Historical Baseline Comparison** -- Establish forest baselines from 2018-2020 reference period canopy cover and forest area measurements with minimum 3 samples. Compare current conditions to baselines with configurable 10% canopy cover threshold for degradation detection.

7. **Alert Workflow Management** -- Manage alert lifecycle through PENDING -> TRIAGED -> INVESTIGATING -> RESOLVED/ESCALATED/FALSE_POSITIVE states. Auto-triage support with configurable SLAs (triage 4h, investigation 48h, resolution 168h/7 days). Up to 3 escalation levels.

8. **Compliance Impact Assessment** -- Map deforestation alerts to affected suppliers, products, commodities, market restrictions, and financial impact. Auto-assessment triggers at HIGH severity. Generate remediation action plans, supplier notifications, and market restriction recommendations.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Satellite source coverage | 5 sources (Sentinel-2, Landsat, GLAD, Hansen GFC, RADD) | Count of integrated sources |
| Detection spatial resolution | 10m (Sentinel-2) to 30m (Landsat) | Resolution per source |
| Detection temporal frequency | 5-day (Sentinel-2), weekly (GLAD), daily (RADD SAR) | Revisit period per source |
| Spectral indices supported | 5 (NDVI, EVI, NBR, NDMI, SAVI) | Count of implemented indices |
| Alert generation latency | < 5 seconds for single detection | p99 latency under load |
| Alert deduplication effectiveness | > 95% duplicate elimination | Duplicate ratio in 72h window |
| Severity classification accuracy | 90%+ agreement with expert assessment | Backtested against labeled events |
| Cutoff date verification confidence | >= 0.85 threshold | Mean confidence score |
| Buffer zone monitoring coverage | 1-50 km configurable radius | Spatial coverage validation |
| Workflow SLA compliance | 95%+ SLA adherence (triage 4h, investigation 48h, resolution 7d) | SLA breach rate |
| Processing performance | < 2 seconds single-plot analysis | p99 latency |
| Determinism | 100% reproducible (zero-hallucination) | Bit-perfect reproducibility tests |
| EUDR Article compliance | Full coverage of Articles 2, 9, 10, 11, 31 | Regulatory mapping completeness |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR, plus the broader forest monitoring technology market estimated at 3-5 billion EUR for satellite-based environmental compliance.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of EUDR-regulated commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood) requiring continuous satellite monitoring, estimated at 800M-1.2B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers using GreenLang's EUDR platform, with deforestation alerting as the core differentiating capability, representing 30M-50M EUR in monitoring module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers sourcing from tropical forest regions (Amazon, Congo Basin, Southeast Asia)
- Multinational food and beverage companies with palm oil, cocoa, coffee, and soya supply chains
- Timber and paper industry operators with tropical wood sourcing
- Compliance teams responsible for EUDR due diligence statements and Article 4 obligations

**Secondary:**
- Certification bodies (FSC, RSPO, Rainforest Alliance) requiring satellite-based verification
- Financial institutions with exposure to EUDR-regulated commodity supply chains
- Government agencies responsible for EUDR enforcement and border controls
- NGOs and researchers monitoring deforestation trends in commodity-producing regions
- Commodity traders and intermediaries requiring plot-level deforestation evidence

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Global Forest Watch (WRI) | Free; GLAD/Hansen data; global coverage | No EUDR-specific workflow; no compliance mapping; no buffer monitoring; no SLA tracking | Full EUDR integration; 8-engine pipeline; compliance impact; SLA workflow |
| Planet Labs | High-resolution daily imagery (3-5m); global coverage | Expensive; no EUDR analysis pipeline; imagery-only, no compliance logic | Multi-source fusion; built-in EUDR analysis; complete compliance workflow |
| Starling (Airbus/Earthworm) | Deforestation monitoring for commodity supply chains | Closed platform; limited customization; no EUDR cutoff date logic | Open architecture; cutoff verification; full regulatory alignment |
| Satelligence | AI-powered deforestation monitoring | Limited to specific commodities; no EUDR compliance workflow | All 7 EUDR commodities; complete workflow; severity classification |
| Manual satellite review | Low cost; human expert judgment | Slow (weeks); non-scalable; no systematic SLA tracking | Real-time automated; scalable; SLA-enforced workflow |

---

## 3. Regulatory Requirements

### 3.1 EUDR Article Mapping

| EUDR Article | Requirement | Agent Feature |
|-------------|-------------|---------------|
| Article 2(1) | December 31, 2020 cutoff date | Feature 5: Cutoff Date Verifier with multi-source temporal evidence |
| Article 9(1) | Due diligence system with information collection | Feature 1: Satellite Change Detection with 5 sources |
| Article 9(1)(d) | Geolocation of production plots | Feature 4: Spatial Buffer Monitor with plot coordinates |
| Article 10(2)(b) | Risk assessment considering deforestation | Feature 3: Five-tier Severity Classification |
| Article 10(2)(c) | Risk assessment of country risk | Feature 8: Compliance Impact Assessment with country data |
| Article 10(2)(e) | Satellite monitoring imagery | Features 1+6: Multi-source detection + Historical Baseline |
| Article 11 | Risk mitigation measures | Feature 7: Alert Workflow Management; Feature 8: Remediation plans |
| Article 31 | Record retention (5 years) | Built-in 5-year retention on all tables; configurable via retention policy |

### 3.2 Data Sources and Regulatory Basis

| Data Source | Provider | Resolution | Frequency | Regulatory Basis |
|------------|----------|-----------|-----------|------------------|
| Sentinel-2 | ESA/Copernicus | 10m multispectral | 5-day revisit | EC delegated acts reference Copernicus for EUDR monitoring |
| Landsat 8/9 | USGS/NASA | 30m multispectral | 8/16-day revisit | Hansen GFC (foundational for EUDR) uses Landsat time series |
| GLAD Alerts | University of Maryland | 30m (Landsat-based) | Weekly | Referenced by EC technical guidance for near-real-time monitoring |
| Hansen GFC | University of Maryland | 30m annual | Annual | EU JRC uses Hansen data for country benchmarking |
| RADD Alerts | Wageningen/WRI | 10m (Sentinel-1 SAR) | 6-12 days | SAR penetrates cloud cover; critical for tropical regions |

---

## 4. Technical Architecture

### 4.1 System Components

```
AGENT-EUDR-020: Deforestation Alert System
  |
  +-- Foundation Modules
  |     +-- config.py .............. DeforestationAlertSystemConfig (60+ GL_EUDR_DAS_ env vars)
  |     +-- models.py .............. 12 enums, 12 core models, 8 request, 8 response models
  |     +-- provenance.py .......... SHA-256 chain-hashed audit trail (12 entity types, 12 actions)
  |     +-- metrics.py ............. 20 Prometheus metrics (gl_eudr_das_ prefix)
  |
  +-- Root Engines (4)
  |     +-- satellite_change_detector.py .. Feature 1: Multi-source satellite change detection
  |     +-- alert_generator.py ........... Feature 2: Alert generation with deduplication
  |     +-- severity_classifier.py ....... Feature 3: Five-tier severity classification
  |     +-- spatial_buffer_monitor.py .... Feature 4: Spatial buffer zone monitoring
  |
  +-- Sub-Engines (4)
  |     +-- engines/cutoff_date_verifier.py ......... Feature 5: EUDR cutoff date verification
  |     +-- engines/historical_baseline_engine.py ... Feature 6: Historical baseline comparison
  |     +-- engines/alert_workflow_engine.py ......... Feature 7: Alert workflow management
  |     +-- engines/compliance_impact_assessor.py ... Feature 8: Compliance impact assessment
  |
  +-- Setup Facade
  |     +-- setup.py ............... DeforestationAlertSystemSetup (singleton, 8 engines)
  |
  +-- Reference Data
  |     +-- satellite_sources.py ... 5 satellite source specifications with spectral bands
  |     +-- deforestation_hotspots.py ... 30+ global hotspot regions with FAO data
  |     +-- protected_areas.py .... 100+ WDPA protected areas with IUCN categories
  |     +-- country_forest_data.py . 180+ country forest cover statistics
  |
  +-- API Layer (12 files)
        +-- router.py .............. Central FastAPI router aggregator
        +-- schemas.py ............. Request/response validation schemas
        +-- dependencies.py ........ FastAPI dependency injection
        +-- satellite_routes.py .... /satellite/* endpoints
        +-- alert_routes.py ........ /alerts/* endpoints
        +-- severity_routes.py ..... /severity/* endpoints
        +-- buffer_routes.py ....... /buffer/* endpoints
        +-- cutoff_routes.py ....... /cutoff/* endpoints
        +-- baseline_routes.py ..... /baseline/* endpoints
        +-- workflow_routes.py ..... /workflow/* endpoints
        +-- compliance_routes.py ... /compliance/* endpoints
```

### 4.2 Database Schema (V108)

12 tables with TimescaleDB hypertables for time-series data:

| Table | Type | Purpose |
|-------|------|---------|
| `gl_eudr_das_satellite_detections` | Hypertable (30d) | Raw satellite change detections with spectral indices |
| `gl_eudr_das_alerts` | Hypertable (30d) | Generated deforestation alerts with severity and status |
| `gl_eudr_das_severity_scores` | Regular | Detailed severity score breakdowns with 5 dimension scores |
| `gl_eudr_das_spatial_buffers` | Regular | Buffer zone definitions with geometry and configuration |
| `gl_eudr_das_buffer_violations` | Regular | Buffer zone proximity violations with distance metrics |
| `gl_eudr_das_cutoff_verifications` | Regular | Cutoff date verification results with temporal evidence |
| `gl_eudr_das_historical_baselines` | Regular | Reference period forest baselines (2018-2020) |
| `gl_eudr_das_baseline_comparisons` | Regular | Current vs. baseline comparison results |
| `gl_eudr_das_workflow_states` | Hypertable (30d) | Alert workflow state machine transitions |
| `gl_eudr_das_compliance_impacts` | Regular | Compliance impact assessment results |
| `gl_eudr_das_notifications` | Regular | Notification dispatch records |
| `gl_eudr_das_audit_log` | Hypertable (30d) | Comprehensive audit trail |

**Continuous Aggregates:**
- `gl_eudr_das_daily_detection_summary` -- Daily detection counts, areas, and confidence by source
- `gl_eudr_das_weekly_alert_summary` -- Weekly alert counts by severity and commodity

**Retention Policies:**
- Satellite detections: 5 years
- Alerts: 10 years (regulatory evidence)
- Workflow states: 5 years
- Audit log: 5 years (per EUDR Article 31)

**Indexes:** ~160 covering all query patterns (composite, GIN on JSONB, partial on status/severity)

### 4.3 Configuration Architecture

All settings managed through `DeforestationAlertSystemConfig` singleton with `GL_EUDR_DAS_` environment variable prefix (60+ configurable parameters):

| Category | Key Settings |
|----------|-------------|
| **Database** | PostgreSQL URL, pool size (10), pool timeout (30s), pool recycle (3600s) |
| **Cache** | Redis URL, TTL (3600s), key prefix `gl_eudr_das` |
| **Satellite Sources** | Per-source enable/disable, resolution, revisit period, cloud cover max |
| **Change Detection** | NDVI threshold (-0.15), EVI threshold (-0.12), min area (0.5 ha), confidence (0.75), temporal window (30d) |
| **Alert Generation** | Batch size (1000), real-time enabled, dedup window (72h), daily cap (10000), retention (5y) |
| **Severity Weights** | Area (0.25), rate (0.20), proximity (0.25), protected (0.15), timing (0.15) |
| **Severity Thresholds** | Critical area >= 50 ha, high >= 10 ha, medium >= 1 ha; proximity critical <= 1 km, high <= 5 km |
| **Buffer Zones** | Default radius (10 km), min (1 km), max (50 km), resolution (64 points) |
| **Cutoff Date** | Date (2020-12-31), grace period (90d), min evidence sources (2), confidence (0.85) |
| **Baselines** | Reference period 2018-2020, min samples (3), canopy threshold (10%) |
| **Workflow** | Auto-triage enabled, SLA triage (4h), investigation (48h), resolution (168h), escalation (3 levels) |
| **Compliance** | Auto impact assessment, market restriction at HIGH severity, remediation required |
| **Rate Limiting** | Anonymous (10/min), basic (60/min), standard (300/min), premium (1000/min), admin (10000/min) |

### 4.4 Provenance and Audit Trail

SHA-256 chain-hashed provenance tracking covers 12 entity types:

| Entity Type | Tracked Actions |
|------------|----------------|
| satellite_detection | created, updated, verified |
| alert | created, classified, triaged, investigated, resolved, escalated, closed |
| severity_score | computed, reclassified |
| buffer_zone | created, updated, violation_detected |
| cutoff_verification | verified, evidence_added |
| baseline | established, compared, updated |
| workflow_state | transitioned, sla_breach_detected |
| compliance_impact | assessed, market_restricted, remediation_planned |
| notification | dispatched, delivered, failed |
| report | generated, exported |
| data_import | initiated, completed, failed |
| configuration | changed, validated |

### 4.5 Metrics and Observability

20 Prometheus metrics with `gl_eudr_das_` prefix:

| Metric | Type | Purpose |
|--------|------|---------|
| `satellite_detections_total` | Counter | Total detections by source and change type |
| `satellite_detection_latency_seconds` | Histogram | Detection processing time |
| `satellite_detection_confidence` | Histogram | Detection confidence distribution |
| `alerts_generated_total` | Counter | Alerts generated by severity |
| `alerts_deduplicated_total` | Counter | Deduplicated alert count |
| `alert_generation_latency_seconds` | Histogram | Alert generation time |
| `severity_classifications_total` | Counter | Classifications by severity level |
| `severity_score_distribution` | Histogram | Severity score distribution (0-100) |
| `buffer_checks_total` | Counter | Buffer zone checks performed |
| `buffer_violations_total` | Counter | Buffer violations detected by type |
| `buffer_check_latency_seconds` | Histogram | Buffer check processing time |
| `cutoff_verifications_total` | Counter | Cutoff verifications by result |
| `cutoff_verification_confidence` | Histogram | Verification confidence distribution |
| `baseline_comparisons_total` | Counter | Baseline comparisons performed |
| `baseline_degradation_detected_total` | Counter | Degradation events detected |
| `workflow_transitions_total` | Counter | Workflow state transitions |
| `workflow_sla_breaches_total` | Counter | SLA breaches by deadline type |
| `compliance_assessments_total` | Counter | Compliance assessments by outcome |
| `compliance_market_restrictions_total` | Counter | Market restrictions triggered |
| `api_requests_total` | Counter | API requests by endpoint and status |

---

## 5. Feature Specifications

### 5.1 Feature 1: Multi-Source Satellite Change Detection

**Engine**: `SatelliteChangeDetector`

**Purpose**: Integrate five satellite data sources to detect land cover changes indicating deforestation or forest degradation near EUDR-regulated supply chain plots.

**Satellite Sources**:

| Source | Resolution | Revisit | Spectral Bands | Key Advantage |
|--------|-----------|---------|----------------|---------------|
| Sentinel-2 | 10m | 5 days | 13 bands (VIS/NIR/SWIR) | High spatial resolution for small clearings |
| Landsat 8/9 | 30m | 8/16 days | 11 bands (VIS/NIR/SWIR/TIR) | Long-term historical archive (1984+) |
| GLAD | 30m | Weekly | Landsat-derived | Near-real-time alerting, University of Maryland |
| Hansen GFC | 30m | Annual | Landsat time series | Authoritative annual tree cover loss data |
| RADD | 10m | 6-12 days | Sentinel-1 SAR (C-band) | Cloud penetration for tropical regions |

**Spectral Indices**:

| Index | Formula | Detection Purpose |
|-------|---------|------------------|
| NDVI | (NIR - RED) / (NIR + RED) | General vegetation health; deforestation threshold: drop > 0.15 |
| EVI | 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1) | Enhanced vegetation; less atmospheric sensitivity |
| NBR | (NIR - SWIR) / (NIR + SWIR) | Burn severity detection for fire-driven deforestation |
| NDMI | (NIR - SWIR1) / (NIR + SWIR1) | Moisture stress indicating degradation |
| SAVI | ((NIR - RED) / (NIR + RED + L)) * (1 + L) | Soil-adjusted; useful for sparse canopy areas |

**Processing Pipeline**:
1. Ingest satellite imagery tiles for target area
2. Apply cloud mask and quality filtering (max 20% cloud cover)
3. Compute spectral indices (NDVI, EVI, NBR, NDMI, SAVI)
4. Compare to previous acquisition (temporal change detection)
5. Apply minimum clearing area threshold (0.5 ha)
6. Fuse multi-source detections for confidence scoring
7. Generate detection records with provenance hash

**Acceptance Criteria**:
- Detects clearing events >= 0.5 ha with >= 0.75 confidence
- Processes single tile in < 2 seconds
- Cloud cover filtering eliminates false positives from cloud shadows
- Multi-source fusion increases confidence for confirmed changes
- All detections include provenance hash for audit trail

### 5.2 Feature 2: Alert Generation with Deduplication

**Engine**: `AlertGenerator`

**Purpose**: Transform satellite detections into actionable deforestation alerts with intelligent deduplication, batch processing, and real-time streaming support.

**Alert Generation Pipeline**:
1. Receive detection events from SatelliteChangeDetector
2. Apply deduplication within 72-hour sliding window (spatial + temporal matching)
3. Enrich with geolocation metadata (country, region, nearby plots)
4. Assign initial priority based on detection confidence and source
5. Generate alert with unique ID and provenance hash
6. Dispatch to batch queue or real-time stream

**Deduplication Logic**:
- Spatial: detections within 100m of existing alert centroid
- Temporal: within 72-hour deduplication window
- Source: cross-source deduplication (same event from different satellites)

**Batch Processing**:
- Batch size: 1000 alerts per batch
- Concurrent workers: configurable (default 4)
- Timeout: 300 seconds per batch
- Daily cap: 10,000 alerts to prevent system overload

**Acceptance Criteria**:
- > 95% duplicate elimination within deduplication window
- Alert generation latency < 5 seconds per detection
- 5-year retention per EUDR Article 31
- Provenance hash chain maintained for all generated alerts

### 5.3 Feature 3: Five-Tier Severity Classification

**Engine**: `SeverityClassifier`

**Purpose**: Classify deforestation alert severity using a weighted multi-factor scoring system aligned with EUDR risk assessment requirements.

**Scoring Dimensions (5)**:

| Dimension | Weight | Description | Thresholds |
|-----------|--------|-------------|-----------|
| Area | 0.25 | Size of affected deforested area | Critical >= 50 ha, High >= 10 ha, Medium >= 1 ha |
| Deforestation Rate | 0.20 | Speed of forest loss (ha/week) | Critical >= 10 ha/wk, High >= 2 ha/wk, Medium >= 0.5 ha/wk |
| Proximity | 0.25 | Distance to nearest supply chain plot | Critical <= 1 km, High <= 5 km, Medium <= 25 km |
| Protected Area | 0.15 | Overlap with WDPA protected areas | 1.5x multiplier when in/near protected area |
| Post-Cutoff Timing | 0.15 | Whether event occurred after 2020-12-31 | 2.0x multiplier for confirmed post-cutoff events |

**Severity Levels**:

| Level | Score Range | Response Required |
|-------|-------------|-------------------|
| CRITICAL | >= 80 | Immediate triage; senior management escalation; market restriction consideration |
| HIGH | >= 60 | 24-hour triage SLA; compliance officer notification; enhanced due diligence |
| MEDIUM | >= 40 | 48-hour triage SLA; monitoring team review; standard due diligence |
| LOW | >= 20 | Periodic review; information logging; no immediate action required |
| INFORMATIONAL | < 20 | Record-keeping only; trend analysis input |

**Acceptance Criteria**:
- Deterministic scoring: same inputs always produce same severity level
- Protected area multiplier applied correctly for WDPA overlaps
- Post-cutoff multiplier doubles timing component for events after 2020-12-31
- Reclassification supported when new evidence emerges

### 5.4 Feature 4: Spatial Buffer Zone Monitoring

**Engine**: `SpatialBufferMonitor`

**Purpose**: Monitor deforestation activity within configurable buffer zones around EUDR-regulated supply chain production plots using spatial proximity analysis.

**Buffer Types**:
- **Circular**: Standard radius-based buffer (default 10 km, range 1-50 km) using 64-point polygon approximation
- **Polygon**: Custom boundary-following buffers for irregular plot shapes
- **Adaptive**: Dynamic radius adjustment based on local deforestation pressure

**Spatial Algorithms**:
- **Haversine Distance**: Great-circle distance calculation between detection and plot centroid for circular buffer checks
- **Ray Casting**: Point-in-polygon test for polygon and adaptive buffers
- **Nearest Point**: Minimum distance from detection to buffer boundary

**Buffer Violation Detection**:
1. Receive detection event with lat/lon coordinates
2. Query all active buffers intersecting detection bounding box
3. Calculate exact distance to buffer centroid or boundary
4. Classify violation: INSIDE_BUFFER, ON_BOUNDARY, APPROACHING (within 2x radius)
5. Record violation with distance, bearing, and affected plot IDs

**Acceptance Criteria**:
- Buffer checks complete in < 500ms per detection
- Haversine accuracy within 0.1% of geodetic reference
- 64-point polygon resolution adequate for 1-50 km radii
- Violations tracked with distance, direction, and temporal trend

### 5.5 Feature 5: EUDR Cutoff Date Verification

**Engine**: `CutoffDateVerifier`

**Purpose**: Determine whether detected deforestation events occurred before or after the EUDR cutoff date of December 31, 2020, using multi-source temporal evidence.

**Verification Method**:
1. Collect temporal evidence from all available satellite sources
2. Establish earliest and latest possible dates for the change event
3. Compare event timeframe against EUDR cutoff (2020-12-31)
4. Apply 90-day pre-cutoff grace period for events near the boundary
5. Calculate confidence score based on evidence quality and consistency
6. Classify result: PRE_CUTOFF, POST_CUTOFF, WITHIN_GRACE_PERIOD, INDETERMINATE

**Evidence Sources**:
- Multi-temporal Sentinel-2 imagery (post-2015)
- Landsat archive (1984-present) for historical reference
- GLAD weekly alerts (2016-present)
- Hansen GFC annual tree cover loss (2000-present)
- RADD alerts (2019-present)

**Confidence Scoring**:
- Single-source evidence: 0.5-0.7 confidence
- Dual-source corroboration: 0.7-0.85 confidence
- Triple-source agreement: 0.85-0.95 confidence
- Full multi-source consensus: 0.95+ confidence
- Minimum threshold for determination: 0.85

**Acceptance Criteria**:
- Minimum 2 independent temporal evidence sources required
- Confidence threshold >= 0.85 for definitive classification
- Grace period handling for events within 90 days of cutoff
- INDETERMINATE classification when evidence is insufficient
- All verification results include provenance hash

### 5.6 Feature 6: Historical Baseline Comparison

**Engine**: `HistoricalBaselineEngine`

**Purpose**: Establish 2018-2020 reference period forest baselines for EUDR-regulated plots and perform ongoing comparison to detect degradation.

**Baseline Establishment**:
1. Collect satellite-derived canopy cover data for reference period (2018-2020)
2. Compute mean canopy cover percentage from minimum 3 cloud-free observations
3. Calculate forest area in hectares using resolution-appropriate area estimation
4. Record baseline with spatial extent, sample count, and confidence interval
5. Store provenance hash for reproducibility

**Comparison Analysis**:
- Current canopy cover vs. baseline canopy cover
- Change magnitude (absolute and percentage)
- Degradation threshold: > 10% canopy cover loss from baseline
- Statistical significance test for change detection
- Temporal trend analysis for gradual degradation

**Acceptance Criteria**:
- Minimum 3 baseline samples from reference period
- 10% canopy cover threshold for degradation classification
- Comparison results include statistical confidence interval
- Support for re-baselining when improved data becomes available

### 5.7 Feature 7: Alert Workflow Management

**Engine**: `AlertWorkflowEngine`

**Purpose**: Manage the complete alert lifecycle from detection through resolution with structured workflow states, SLA enforcement, and escalation.

**Workflow States**:
```
PENDING -> TRIAGED -> INVESTIGATING -> RESOLVED
                                    -> ESCALATED -> INVESTIGATING
                                    -> FALSE_POSITIVE
```

**SLA Deadlines**:

| Transition | SLA | Escalation |
|-----------|-----|------------|
| PENDING -> TRIAGED | 4 hours | Auto-escalate to Level 1 after breach |
| TRIAGED -> INVESTIGATING | 48 hours | Auto-escalate to Level 2 after breach |
| INVESTIGATING -> RESOLVED | 168 hours (7 days) | Auto-escalate to Level 3 after breach |

**Auto-Triage Rules**:
- CRITICAL severity: auto-triage with HIGH priority, immediate notification
- HIGH severity: auto-triage with MEDIUM priority, same-day notification
- MEDIUM severity: auto-triage with LOW priority, next-day review queue
- LOW/INFORMATIONAL: auto-triage to monitoring queue

**Acceptance Criteria**:
- SLA tracking with minute-level precision
- Escalation triggers at 3 configurable levels
- Audit trail for all workflow transitions
- State machine prevents invalid transitions

### 5.8 Feature 8: Compliance Impact Assessment

**Engine**: `ComplianceImpactAssessor`

**Purpose**: Map deforestation alerts to concrete EUDR compliance impacts including affected suppliers, products, commodities, market restrictions, remediation actions, and estimated financial impact.

**Assessment Pipeline**:
1. Receive alert with severity classification and cutoff verification
2. Identify affected supply chain plots within buffer zone
3. Map plots to suppliers, commodities, and product lines
4. Classify compliance outcome: COMPLIANT, NON_COMPLIANT, UNDER_REVIEW, REMEDIATION_REQUIRED
5. Generate market restriction recommendation for POST_CUTOFF + HIGH/CRITICAL
6. Calculate estimated financial impact (shipment value, penalty risk)
7. Create remediation action plan with timeline

**Compliance Outcomes**:

| Outcome | Condition | Required Actions |
|---------|-----------|-----------------|
| COMPLIANT | Pre-cutoff deforestation only; current forest intact | Maintain monitoring; retain evidence |
| NON_COMPLIANT | Post-cutoff deforestation confirmed at HIGH/CRITICAL | Market restriction; enhanced due diligence; supplier remediation |
| UNDER_REVIEW | Cutoff date indeterminate or medium severity | Investigation required; temporary enhanced monitoring |
| REMEDIATION_REQUIRED | Confirmed impact but remediation path available | Supplier action plan; timeline; re-verification |

**Acceptance Criteria**:
- Auto-assessment triggers at HIGH severity and above
- Market restriction threshold configurable (default: HIGH)
- Remediation plans include specific actions and timelines
- Financial impact estimation based on commodity value data

---

## 6. API Specification

### 6.1 Endpoint Overview

| Category | Method | Path | Description |
|----------|--------|------|-------------|
| Satellite | POST | /satellite/detect | Trigger change detection for area |
| Satellite | POST | /satellite/scan | Batch scan multiple areas |
| Satellite | GET | /satellite/sources | List satellite source configurations |
| Satellite | GET | /satellite/{detection_id}/imagery | Get detection imagery details |
| Alerts | GET | /alerts | List alerts with filtering and pagination |
| Alerts | GET | /alerts/{alert_id} | Get alert detail |
| Alerts | POST | /alerts | Create manual alert |
| Alerts | POST | /alerts/batch | Batch generate alerts from detections |
| Alerts | GET | /alerts/summary | Alert summary statistics |
| Alerts | GET | /alerts/statistics | Detailed alert analytics |
| Severity | POST | /severity/classify | Classify alert severity |
| Severity | POST | /severity/reclassify | Reclassify with updated evidence |
| Severity | GET | /severity/thresholds | Get severity threshold configuration |
| Severity | GET | /severity/distribution | Severity distribution statistics |
| Buffer | POST | /buffer/create | Create spatial buffer zone |
| Buffer | GET | /buffer/{buffer_id} | Get buffer details |
| Buffer | POST | /buffer/check | Check detection against buffers |
| Buffer | GET | /buffer/violations | List buffer violations |
| Buffer | GET | /buffer/zones | List all active buffer zones |
| Cutoff | POST | /cutoff/verify | Verify cutoff date for detection |
| Cutoff | POST | /cutoff/batch-verify | Batch cutoff verification |
| Cutoff | GET | /cutoff/{detection_id}/evidence | Get temporal evidence |
| Cutoff | GET | /cutoff/timeline | Cutoff verification timeline |
| Baseline | POST | /baseline/establish | Establish historical baseline |
| Baseline | POST | /baseline/compare | Compare current to baseline |
| Baseline | GET | /baseline/{baseline_id} | Get baseline details |
| Baseline | GET | /baseline/coverage | Baseline coverage statistics |
| Workflow | POST | /workflow/triage | Triage an alert |
| Workflow | POST | /workflow/assign | Assign investigation |
| Workflow | POST | /workflow/investigate | Begin investigation |
| Workflow | POST | /workflow/resolve | Resolve alert |
| Workflow | POST | /workflow/escalate | Escalate alert |
| Workflow | GET | /workflow/sla | SLA compliance dashboard |
| Compliance | POST | /compliance/assess | Assess compliance impact |
| Compliance | GET | /compliance/affected-products | List affected products |
| Compliance | GET | /compliance/recommendations | Get compliance recommendations |
| Compliance | POST | /compliance/remediation | Create remediation plan |
| Health | GET | /health | System health check |

### 6.2 Authentication and Authorization

**RBAC Integration**: 26 permission definitions across `eudr-das` resource:

| Permission | Description |
|-----------|-------------|
| `eudr-das:read` | View deforestation alert data |
| `eudr-das:write` | Create/update deforestation alerts |
| `eudr-das:satellite:read` | View satellite detection data |
| `eudr-das:satellite:detect` | Trigger satellite change detection |
| `eudr-das:alerts:read` | View deforestation alerts |
| `eudr-das:alerts:write` | Create/manage deforestation alerts |
| `eudr-das:severity:read` | View severity classifications |
| `eudr-das:severity:analyze` | Classify/reclassify alert severity |
| `eudr-das:buffer:read` | View spatial buffer zones |
| `eudr-das:buffer:write` | Create/update spatial buffer zones |
| `eudr-das:cutoff:read` | View cutoff date verifications |
| `eudr-das:cutoff:analyze` | Perform cutoff date verification |
| `eudr-das:baseline:read` | View historical baselines |
| `eudr-das:baseline:write` | Establish/update baselines |
| `eudr-das:baseline:analyze` | Compare baselines |
| `eudr-das:workflow:read` | View workflow states and SLAs |
| `eudr-das:workflow:write` | Manage alert workflow transitions |
| `eudr-das:compliance:read` | View compliance impact data |
| `eudr-das:compliance:analyze` | Assess compliance impact |
| `eudr-das:compliance:write` | Create remediation plans |
| `eudr-das:reports:generate` | Generate deforestation reports |
| `eudr-das:reports:read` | View deforestation reports |
| `eudr-das:reports:export` | Export deforestation data |
| `eudr-das:data:import` | Import satellite data sources |
| `eudr-das:data:refresh` | Refresh satellite data feeds |
| `eudr-das:admin` | Full deforestation alert system access |

**Role Mappings**:

| Role | Permissions |
|------|-----------|
| auditor | 11 read-only permissions (read, satellite:read, alerts:read, severity:read, buffer:read, cutoff:read, baseline:read, workflow:read, compliance:read, reports:read, reports:export) |
| compliance_officer | 27 full permissions (all read + write + analyze + generate + data:import + data:refresh) |
| supply_chain_analyst | 25 operational permissions (all except data:import, data:refresh) |
| data_analyst | 11 read-only permissions (same as auditor) |

### 6.3 Rate Limiting

| Tier | Rate | Use Case |
|------|------|----------|
| Anonymous | 10/min | Public health check only |
| Basic | 60/min | Read-only integration |
| Standard | 300/min | Normal operational use |
| Premium | 1000/min | High-volume monitoring |
| Admin | 10000/min | System administration |

---

## 7. Reference Data

### 7.1 Satellite Sources Database

5 satellite source specifications with full spectral band configurations:

| Source | Bands | Key Bands for Vegetation | API/Access |
|--------|-------|--------------------------|-----------|
| Sentinel-2 MSI | 13 bands | B4 (Red), B8 (NIR), B11 (SWIR) | Copernicus Open Access Hub |
| Landsat 8 OLI | 11 bands | B4 (Red), B5 (NIR), B6 (SWIR1) | USGS EarthExplorer |
| Landsat 9 OLI-2 | 11 bands | B4 (Red), B5 (NIR), B6 (SWIR1) | USGS EarthExplorer |
| GLAD Alerts | Derived | Landsat-derived alert tiles | UMD GLAD Portal |
| RADD (Sentinel-1) | 2 bands (VV, VH) | SAR backscatter change | WRI/Wageningen |

### 7.2 Deforestation Hotspots Database

30+ global deforestation hotspot regions with FAO-sourced data:

**Key Regions**:
- Amazon Basin (Brazil, Peru, Colombia, Bolivia, Ecuador)
- Congo Basin (DRC, Republic of Congo, Cameroon, Gabon)
- Southeast Asia (Indonesia, Malaysia, Myanmar, Thailand)
- West Africa (Ghana, Ivory Coast, Guinea, Sierra Leone)
- Central America (Guatemala, Honduras, Nicaragua)
- East Africa (Madagascar, Tanzania, Mozambique)

### 7.3 Protected Areas Database

100+ WDPA (World Database on Protected Areas) entries with IUCN categories:

| IUCN Category | Description | Severity Multiplier |
|---------------|-------------|-------------------|
| Ia | Strict Nature Reserve | 1.5x |
| Ib | Wilderness Area | 1.5x |
| II | National Park | 1.5x |
| III | Natural Monument | 1.3x |
| IV | Habitat Management Area | 1.3x |
| V | Protected Landscape | 1.2x |
| VI | Sustainable Use | 1.2x |

### 7.4 Country Forest Data

180+ countries with forest statistics:
- Total forest area (ha) from FAO Forest Resources Assessment
- Forest cover percentage
- Annual deforestation rate (%)
- Primary forest percentage
- EUDR commodity production indicators
- EC country risk classification (HIGH/STANDARD/LOW)

---

## 8. Testing and Quality Assurance

### 8.1 Test Suite

| Test File | Engine Under Test | Test Count |
|-----------|-------------------|-----------|
| `test_satellite_change_detector.py` | SatelliteChangeDetector | ~60 tests |
| `test_alert_generator.py` | AlertGenerator | ~55 tests |
| `test_severity_classifier.py` | SeverityClassifier | ~55 tests |
| `test_spatial_buffer_monitor.py` | SpatialBufferMonitor | ~55 tests |
| `test_cutoff_date_verifier.py` | CutoffDateVerifier | ~55 tests |
| `test_historical_baseline_engine.py` | HistoricalBaselineEngine | ~55 tests |
| `test_alert_workflow_engine.py` | AlertWorkflowEngine | ~55 tests |
| `test_compliance_impact_assessor.py` | ComplianceImpactAssessor | ~55 tests |
| `conftest.py` | Shared fixtures | -- |
| `__init__.py` | Package init | -- |
| **Total** | **8 engines** | **456 tests, 18 parametrized** |

### 8.2 Test Categories

- **Unit Tests**: Individual engine method testing with mocked dependencies
- **Parametrized Tests**: Data-driven testing across satellite sources, severity levels, buffer types, cutoff scenarios
- **Edge Cases**: Empty inputs, boundary conditions, maximum values, invalid coordinates
- **Determinism Tests**: Bit-perfect reproducibility verification
- **Provenance Tests**: SHA-256 hash chain integrity verification
- **Integration Tests**: Multi-engine pipeline testing through setup facade

---

## 9. Deployment and Operations

### 9.1 Infrastructure

- **Database Migration**: V108 (12 tables, 4 hypertables, 2 continuous aggregates, ~160 indexes)
- **Grafana Dashboard**: 20 panels across 5 sections (Satellite Detection, Alert Management, Severity Distribution, Buffer Monitoring, Compliance Impact)
- **Prometheus Metrics**: 20 metrics with `gl_eudr_das_` prefix

### 9.2 Dependencies

| Dependency | Purpose |
|-----------|---------|
| PostgreSQL + TimescaleDB | Primary data store with time-series optimization |
| Redis | Caching for reference data and deduplication windows |
| FastAPI | API framework with async support |
| Pydantic v2 | Data validation and serialization |
| psycopg + psycopg_pool | Async PostgreSQL client |
| prometheus_client | Metrics export |

### 9.3 Operational Runbook

| Scenario | Action |
|----------|--------|
| High alert volume spike | Check satellite source for mass event; adjust daily cap; review dedup window |
| SLA breach rate > 5% | Scale investigation team; adjust auto-triage rules; review severity thresholds |
| False positive rate > 10% | Increase confidence threshold; review cloud masking; adjust NDVI threshold |
| Database growth > projected | Verify retention policies; check continuous aggregate refresh; optimize indexes |
| Satellite source outage | Fall back to remaining sources; adjust confidence scoring; alert operations team |

---

## 10. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cloud cover blocking optical satellites | High | Medium | RADD SAR source penetrates clouds; multi-source fusion reduces gaps |
| False positive detections (cloud shadows, burn scars) | Medium | High | Multi-spectral index analysis; confidence thresholds; human review workflow |
| EUDR cutoff date ambiguity near boundary | Medium | High | 90-day grace period; multi-source temporal evidence; INDETERMINATE classification |
| Satellite data source downtime | Low | Medium | 5 independent sources; graceful degradation; source health monitoring |
| Alert volume overwhelming investigation capacity | Medium | Medium | Auto-triage; severity-based prioritization; SLA escalation |
| Regulatory interpretation changes | Low | High | Configurable thresholds; version-controlled reference data; modular architecture |

---

## 11. Appendices

### 11.1 Glossary

| Term | Definition |
|------|-----------|
| **NDVI** | Normalized Difference Vegetation Index: (NIR - RED) / (NIR + RED). Range -1 to +1; healthy vegetation > 0.6 |
| **EVI** | Enhanced Vegetation Index: improved NDVI with atmospheric correction. Less saturation in dense canopy |
| **NBR** | Normalized Burn Ratio: (NIR - SWIR) / (NIR + SWIR). Detects fire-driven vegetation loss |
| **NDMI** | Normalized Difference Moisture Index: (NIR - SWIR1) / (NIR + SWIR1). Detects water stress |
| **SAVI** | Soil Adjusted Vegetation Index: NDVI with soil brightness correction factor L |
| **EUDR Cutoff Date** | December 31, 2020 (Article 2(1)). Products from areas deforested after this date cannot enter EU market |
| **Haversine** | Great-circle distance formula for calculating distance between lat/lon coordinates on Earth's surface |
| **GLAD** | Global Land Analysis & Discovery, University of Maryland. Provides weekly Landsat-based deforestation alerts |
| **Hansen GFC** | Hansen Global Forest Change dataset. Annual global tree cover loss from Landsat time series |
| **RADD** | Radar Alerts for Detecting Deforestation. Sentinel-1 SAR-based alerts that penetrate cloud cover |
| **WDPA** | World Database on Protected Areas. Comprehensive global database of protected areas |
| **SLA** | Service Level Agreement. Time-bound commitments for alert processing stages |

### 11.2 EUDR Article References

- **Article 2(1)**: Defines deforestation-free as no forest conversion after 31 December 2020
- **Article 9(1)**: Due diligence system requirements including information collection
- **Article 9(1)(d)**: Geolocation coordinates requirement for production plots
- **Article 10(2)(b)**: Risk assessment must consider deforestation and forest degradation
- **Article 10(2)(c)**: Country risk level consideration in risk assessment
- **Article 10(2)(e)**: Satellite monitoring imagery and other relevant evidence
- **Article 11**: Risk mitigation requirements when risks are identified
- **Article 31**: Five-year record retention obligation

---

*Document generated: March 2026*
*Agent: GL-EUDR-DAS-020*
*Platform: GreenLang v1.0*
