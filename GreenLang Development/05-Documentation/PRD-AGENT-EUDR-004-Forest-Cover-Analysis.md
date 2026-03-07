# PRD: AGENT-EUDR-004 — Forest Cover Analysis Agent

**Document ID:** PRD-AGENT-EUDR-004
**Version:** 1.0
**Date:** 2026-03-07
**Author:** GreenLang Platform Team
**Status:** Approved for Development
**Agent ID:** GL-EUDR-FCA-004
**Category:** Category 5 — EUDR Agents
**Regulation:** EU 2023/1115 (EUDR) Articles 2, 9, 10, 12
**Enforcement:** December 30, 2025 (large operators); June 30, 2026 (SMEs)

---

## 1. Executive Summary

AGENT-EUDR-004: Forest Cover Analysis Agent provides comprehensive forest cover
characterization, classification, and deforestation-free verification for EUDR
compliance. While AGENT-EUDR-003 (Satellite Monitoring) detects changes via
spectral indices and temporal analysis, this agent focuses on **static and
historical forest cover assessment** — mapping canopy density, classifying forest
types (primary/secondary/plantation/agroforestry), reconstructing historical
forest cover back to the EUDR cutoff date (December 31, 2020), performing
definitive deforestation-free status determination, modeling canopy height,
analyzing forest fragmentation, estimating above-ground biomass, and generating
regulatory-compliant verification reports.

The agent serves as the **authoritative forest cover determination engine** that
answers the fundamental EUDR question: "Was this production plot covered by
forest on December 31, 2020, and has that forest been degraded or removed since?"

### 1.1 Relationship to Other EUDR Agents

| Agent | Role | Relationship |
|-------|------|-------------|
| EUDR-001 | Supply Chain Mapping | Provides plot geometries for analysis |
| EUDR-002 | Geolocation Verification | Validates plot coordinates before analysis |
| EUDR-003 | Satellite Monitoring | Provides change detection alerts (input) |
| **EUDR-004** | **Forest Cover Analysis** | **Definitive forest cover determination** |

### 1.2 Key Differentiator from EUDR-003

| Aspect | EUDR-003 (Satellite Monitoring) | EUDR-004 (Forest Cover Analysis) |
|--------|--------------------------------|----------------------------------|
| Focus | Change detection over time | Forest characterization & status |
| Method | NDVI differencing, BFAST | Canopy density, tree height, biomass |
| Output | Alerts and change events | Forest/non-forest determination |
| Temporal | Continuous monitoring | Point-in-time assessment |
| Use | Ongoing surveillance | Due diligence verification |

---

## 2. Regulatory Requirements

### 2.1 EUDR Article Mapping

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 2(1) | Definition of "deforestation" and "forest" | Forest definition engine (FAO criteria) |
| Art. 2(5) | "Forest degradation" definition | Degradation assessment engine |
| Art. 2(6) | "Deforestation-free" definition | Deforestation-free verification engine |
| Art. 9(1)(d) | Geolocation of production plots | Integration with EUDR-002 |
| Art. 9(1)(e) | Adequate & verifiable info: no deforestation | Forest cover determination |
| Art. 10(1) | Due diligence system requirements | Historical reconstruction + reports |
| Art. 10(2) | Risk assessment and mitigation | Forest risk scoring engine |
| Art. 12 | Record keeping (5 years) | Provenance tracking + DB retention |

### 2.2 Forest Definitions (FAO/EUDR)

Per EUDR Article 2(4), "forest" means:
- Land spanning > 0.5 hectares
- Tree canopy cover > 10%
- Trees reaching height > 5 meters at maturity in situ
- Excludes: agricultural tree plantations (palm oil, rubber when in monoculture)
- Includes: temporarily unstocked areas expected to regenerate

### 2.3 Cutoff Date Compliance

- **Cutoff date:** December 31, 2020
- Products must be "deforestation-free" — produced on land not subject to
  deforestation after this date
- Agent must reconstruct forest cover state as of December 31, 2020

---

## 3. Features (P0 — Must Have)

### F-001: Canopy Density Mapping Engine
Quantifies tree canopy cover percentage at sub-pixel resolution using spectral
unmixing and vegetation index regression.

- **Inputs:** Multi-spectral imagery (Sentinel-2, Landsat), plot polygon
- **Outputs:** Canopy density percentage (0-100%), confidence interval, spatial map
- **Methods:**
  - Linear spectral unmixing (endmember: forest, soil, water, impervious)
  - NDVI-canopy cover regression (biome-calibrated models)
  - Fractional cover estimation (Dimidiation model)
  - Sub-pixel tree detection for sparse canopies
- **Accuracy target:** +/-5% canopy cover at 10m resolution
- **FAO threshold check:** Flags plots with canopy >10% as potential forest

### F-002: Forest Type Classification Engine
Classifies forest into regulatory-relevant categories using spectral signatures,
phenological patterns, and structural metrics.

- **Categories (10 types):**
  1. Primary/old-growth tropical forest
  2. Secondary tropical forest
  3. Tropical dry forest
  4. Temperate broadleaf forest
  5. Temperate coniferous forest
  6. Boreal/taiga forest
  7. Mangrove forest
  8. Plantation forest (eucalyptus, pine, teak)
  9. Agroforestry systems (shade-grown coffee/cocoa)
  10. Non-forest vegetation (shrubland, grassland, cropland)
- **Method:** Multi-temporal spectral analysis + phenological profiling
- **EUDR relevance:** Distinguishes legally-relevant forest from agricultural
  tree crops (Art. 2(4) exclusions)

### F-003: Historical Forest Cover Reconstruction Engine
Reconstructs forest cover state at the EUDR cutoff date (December 31, 2020)
using archived satellite imagery composites.

- **Temporal range:** 2018-2020 (3-year composite for robust baseline)
- **Data sources:** Landsat archive, Sentinel-2 archive, Hansen Global Forest
  Change (UMD), JAXA Forest/Non-Forest map
- **Methods:**
  - Multi-temporal median composite (cloud-free)
  - Random Forest / decision tree classification
  - Temporal interpolation for gap years
  - Cross-validation with independent datasets (GFW, JRC TMF)
- **Output:** Binary forest/non-forest map + canopy density at cutoff date
- **Confidence scoring:** 0-100% based on data availability and agreement

### F-004: Deforestation-Free Verification Engine
Performs the definitive EUDR deforestation-free determination by comparing
cutoff-date forest cover with current state.

- **Logic:**
  1. Establish forest cover at cutoff date (F-003)
  2. Assess current forest cover (F-001)
  3. If cutoff had forest AND current has no forest → DEFORESTED
  4. If cutoff had forest AND current has reduced canopy → DEGRADED
  5. If cutoff had no forest → DEFORESTATION-FREE (land was already non-forest)
  6. If cutoff had forest AND current has forest → DEFORESTATION-FREE
- **Verdicts:** DEFORESTATION_FREE, DEFORESTED, DEGRADED, INCONCLUSIVE
- **Degradation thresholds:** Configurable per biome (default >30% canopy loss)
- **Evidence package:** Before/after imagery, spectral indices, canopy density diff

### F-005: Canopy Height Modeling Engine
Estimates tree canopy height using available data sources for FAO 5-meter
threshold verification.

- **Data sources:**
  - GEDI L2A/L2B canopy height (25m footprint, ±3m accuracy)
  - ICESat-2 ATL08 land/canopy height
  - Sentinel-2 texture metrics (GLCM) as height proxy
  - Global canopy height maps (ETH Zurich, Meta)
- **Method:** Multi-source fusion with weighted confidence
- **Output:** Estimated canopy height (meters), uncertainty bounds
- **FAO check:** Flags plots with trees ≥5m potential height at maturity

### F-006: Forest Fragmentation Analysis Engine
Analyzes landscape-level forest fragmentation patterns to assess forest
integrity and edge effects.

- **Metrics (6 indices):**
  1. Patch count and size distribution
  2. Edge density (m/ha)
  3. Core area percentage (excluding 100m edge buffer)
  4. Connectivity index (nearest-neighbor distance)
  5. Shape complexity (perimeter-area ratio)
  6. Effective mesh size (landscape division)
- **EUDR relevance:** Heavily fragmented plots suggest historical encroachment
- **Risk scoring:** Higher fragmentation = higher deforestation risk

### F-007: Above-Ground Biomass Estimation Engine
Estimates above-ground biomass (AGB) from remote sensing data for forest
characterization and carbon stock assessment.

- **Data sources:**
  - ESA CCI Biomass maps (100m, Mg/ha)
  - GEDI L4A biomass predictions
  - Sentinel-1 SAR backscatter regression
  - NDVI-biomass allometric relationships (biome-specific)
- **Output:** AGB estimate (Mg/ha), uncertainty, carbon stock (tC/ha)
- **EUDR relevance:** Biomass loss indicates forest degradation even without
  complete canopy removal
- **Reference values:** Biome-specific AGB ranges for forest/non-forest

### F-008: Regulatory Compliance Report Generator
Generates comprehensive EUDR compliance reports with forest cover evidence
for Due Diligence Statements.

- **Report types:**
  1. Plot-level forest cover assessment
  2. Batch commodity verification report
  3. Deforestation-free certification evidence
  4. Historical forest cover reconstruction report
  5. Summary dashboard data for DDS
- **Formats:** JSON, PDF, CSV, EUDR XML
- **Contents:**
  - Forest/non-forest determination with confidence
  - Before/after comparison (cutoff vs current)
  - All supporting metrics (canopy density, height, biomass, fragmentation)
  - Provenance chain with SHA-256 hashes
  - Regulatory references (EUDR articles, FAO definitions)
- **Compliance flags:** Auto-flag high-risk determinations requiring review

---

## 4. Technical Architecture

### 4.1 Module Structure

```
greenlang/agents/eudr/forest_cover_analysis/
├── __init__.py                          # Package exports (~270 lines)
├── config.py                            # ForestCoverConfig singleton (~800 lines)
├── models.py                            # Pydantic v2 models (~2500 lines)
├── provenance.py                        # SHA-256 chain hashing (~700 lines)
├── metrics.py                           # Prometheus metrics (~550 lines)
├── setup.py                             # Service facade (~2500 lines)
│
├── canopy_density_mapper.py             # F-001 (~1200 lines)
├── forest_type_classifier.py            # F-002 (~1300 lines)
├── historical_reconstructor.py          # F-003 (~1400 lines)
├── deforestation_free_verifier.py       # F-004 (~1300 lines)
├── canopy_height_modeler.py             # F-005 (~1100 lines)
├── fragmentation_analyzer.py            # F-006 (~1200 lines)
├── biomass_estimator.py                 # F-007 (~1100 lines)
├── compliance_reporter.py               # F-008 (~1300 lines)
│
├── reference_data/
│   ├── __init__.py                      # Package exports
│   ├── biome_parameters.py              # Biome-specific thresholds (~450 lines)
│   ├── allometric_equations.py          # Biomass allometric models (~400 lines)
│   └── forest_definitions.py            # FAO/EUDR forest definitions (~350 lines)
│
└── api/
    ├── __init__.py                      # API package exports
    ├── router.py                        # Main router (~170 lines)
    ├── schemas.py                       # Pydantic schemas (~1800 lines)
    ├── dependencies.py                  # FastAPI DI (~650 lines)
    ├── density_routes.py                # Canopy density endpoints (~500 lines)
    ├── classification_routes.py         # Forest type endpoints (~500 lines)
    ├── historical_routes.py             # Historical reconstruction endpoints (~500 lines)
    ├── verification_routes.py           # Deforestation-free endpoints (~600 lines)
    ├── analysis_routes.py               # Height/fragmentation/biomass endpoints (~600 lines)
    └── report_routes.py                 # Compliance report endpoints (~500 lines)
```

### 4.2 Database Schema (V092)

| Table | Purpose | Type |
|-------|---------|------|
| `eudr_canopy_density_maps` | Canopy density assessments | Hypertable (monthly) |
| `eudr_forest_classifications` | Forest type classification results | Hypertable (monthly) |
| `eudr_historical_reconstructions` | Cutoff-date forest cover reconstructions | Hypertable (quarterly) |
| `eudr_deforestation_free_verdicts` | Definitive deforestation-free verdicts | Hypertable (monthly) |
| `eudr_canopy_height_estimates` | Canopy height model outputs | Hypertable (quarterly) |
| `eudr_fragmentation_analyses` | Fragmentation metric results | Standard |
| `eudr_biomass_estimates` | Above-ground biomass estimates | Standard |
| `eudr_forest_compliance_reports` | Generated compliance reports | Standard |
| `eudr_forest_cover_baselines` | Historical baseline composites | Standard |
| `eudr_forest_analysis_audit_log` | Immutable audit trail | Standard |
| + 2 continuous aggregates | daily_forest_verdicts, weekly_analysis_stats |

### 4.3 Prometheus Metrics (18 metrics, `gl_eudr_fca_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_fca_density_analyses_total` | Counter | Canopy density analyses performed |
| `gl_eudr_fca_classifications_total` | Counter | Forest type classifications |
| `gl_eudr_fca_reconstructions_total` | Counter | Historical reconstructions |
| `gl_eudr_fca_verdicts_total` | Counter | Deforestation-free verdicts issued |
| `gl_eudr_fca_height_estimates_total` | Counter | Canopy height estimates |
| `gl_eudr_fca_fragmentation_analyses_total` | Counter | Fragmentation analyses |
| `gl_eudr_fca_biomass_estimates_total` | Counter | Biomass estimates |
| `gl_eudr_fca_reports_generated_total` | Counter | Compliance reports generated |
| `gl_eudr_fca_deforested_plots_total` | Counter | Plots determined as deforested |
| `gl_eudr_fca_degraded_plots_total` | Counter | Plots determined as degraded |
| `gl_eudr_fca_analysis_duration_seconds` | Histogram | Analysis duration by operation |
| `gl_eudr_fca_batch_duration_seconds` | Histogram | Batch analysis duration |
| `gl_eudr_fca_api_errors_total` | Counter | API errors by operation |
| `gl_eudr_fca_active_analyses` | Gauge | Currently running analyses |
| `gl_eudr_fca_avg_canopy_density` | Gauge | Running average canopy density |
| `gl_eudr_fca_avg_confidence_score` | Gauge | Running average confidence score |
| `gl_eudr_fca_data_quality_score` | Gauge | Data quality by source |
| `gl_eudr_fca_forest_area_ha` | Gauge | Total forest area assessed |

### 4.4 API Endpoints (30 endpoints)

| Route Group | Endpoints | Description |
|-------------|-----------|-------------|
| `/api/v1/eudr-fca/density/` | 5 | Canopy density mapping |
| `/api/v1/eudr-fca/classify/` | 4 | Forest type classification |
| `/api/v1/eudr-fca/historical/` | 5 | Historical reconstruction |
| `/api/v1/eudr-fca/verify/` | 5 | Deforestation-free verification |
| `/api/v1/eudr-fca/analysis/` | 5 | Height/fragmentation/biomass |
| `/api/v1/eudr-fca/reports/` | 4 | Compliance reporting |
| `/api/v1/eudr-fca/batch/` | 2 | Batch operations |

### 4.5 Environment Variables (`GL_EUDR_FCA_` prefix)

| Variable | Description | Default |
|----------|-------------|---------|
| `GL_EUDR_FCA_DATABASE_URL` | PostgreSQL connection URL | required |
| `GL_EUDR_FCA_REDIS_URL` | Redis connection URL | `redis://localhost:6379/11` |
| `GL_EUDR_FCA_LOG_LEVEL` | Logging level | `INFO` |
| `GL_EUDR_FCA_CANOPY_COVER_THRESHOLD` | FAO forest canopy threshold (%) | `10.0` |
| `GL_EUDR_FCA_TREE_HEIGHT_THRESHOLD` | FAO tree height threshold (m) | `5.0` |
| `GL_EUDR_FCA_MIN_FOREST_AREA_HA` | Min forest area (ha) per FAO | `0.5` |
| `GL_EUDR_FCA_DEGRADATION_THRESHOLD` | % canopy loss for degradation | `30.0` |
| `GL_EUDR_FCA_CUTOFF_DATE` | EUDR cutoff date | `2020-12-31` |
| `GL_EUDR_FCA_BASELINE_WINDOW_YEARS` | Years for historical composite | `3` |
| `GL_EUDR_FCA_GEDI_API_KEY` | NASA GEDI data API key | optional |
| `GL_EUDR_FCA_ESA_CCI_API_KEY` | ESA CCI Biomass data key | optional |
| `GL_EUDR_FCA_HANSEN_GFC_VERSION` | Hansen GFC dataset version | `v1.11` |
| `GL_EUDR_FCA_MAX_BATCH_SIZE` | Max plots per batch | `5000` |
| `GL_EUDR_FCA_ANALYSIS_CONCURRENCY` | Max concurrent analyses | `8` |
| `GL_EUDR_FCA_CACHE_TTL_SECONDS` | Cache TTL for results | `3600` |
| `GL_EUDR_FCA_BIOMASS_CACHE_TTL` | Biomass cache TTL | `86400` |
| `GL_EUDR_FCA_CONFIDENCE_MIN` | Min confidence for determination | `0.6` |
| `GL_EUDR_FCA_GENESIS_HASH` | Provenance genesis hash | auto-generated |
| `GL_EUDR_FCA_ENABLE_METRICS` | Enable Prometheus metrics | `true` |

---

## 5. Data Quality & Zero-Hallucination

### 5.1 Deterministic Calculations
- All calculations use fixed formulas with documented references
- No stochastic or ML-based outputs without confidence intervals
- Identical inputs always produce identical outputs
- SHA-256 provenance chain ensures tamper detection

### 5.2 Data Quality Scoring
- Each analysis includes a data quality score (0-100)
- Quality factors: temporal proximity, cloud cover, spatial resolution,
  cross-source agreement, measurement uncertainty
- Low-quality results flagged as INCONCLUSIVE (never hallucinated)

### 5.3 Conservative Determination
- When data is ambiguous, default to INCONCLUSIVE verdict
- Never issue DEFORESTATION_FREE verdict with confidence <60%
- Always report uncertainty bounds alongside point estimates

---

## 6. Testing Strategy

### 6.1 Target: 500+ test functions

| Test Category | Target Count |
|---------------|-------------|
| Canopy density mapping | ~65 |
| Forest type classification | ~65 |
| Historical reconstruction | ~60 |
| Deforestation-free verification | ~80 |
| Canopy height modeling | ~50 |
| Fragmentation analysis | ~50 |
| Biomass estimation | ~50 |
| Compliance reporting | ~40 |
| Models & config validation | ~60 |
| Determinism & provenance | ~30 |

### 6.2 Test Patterns
- Parametrized tests for all biomes, commodities, forest types
- Determinism tests (same input = same output + same hash)
- Edge cases: boundary values, zero area, missing data
- Config validation: invalid thresholds, credential redaction
- Cross-engine integration scenarios

---

## 7. Integration Points

### 7.1 Upstream Dependencies
- AGENT-EUDR-001: Plot geometries and supply chain data
- AGENT-EUDR-002: Validated coordinates and polygon topology
- AGENT-EUDR-003: Satellite imagery, spectral indices, change alerts

### 7.2 Downstream Consumers
- GL-EUDR-APP: Forest cover results for DDS generation
- AGENT-EUDR-005+: Risk assessment agents consume forest verdicts
- Compliance reporting: Regulatory report generation

### 7.3 RBAC Permissions (18 permissions)
Resource prefix: `eudr-fca`
Permissions: `read`, `write`, `density:analyze`, `density:batch`,
`classify:run`, `classify:batch`, `historical:reconstruct`,
`historical:compare`, `verify:single`, `verify:batch`,
`height:estimate`, `fragmentation:analyze`, `biomass:estimate`,
`reports:generate`, `reports:download`, `batch:submit`,
`batch:cancel`, `admin:configure`

---

## 8. Acceptance Criteria

- [ ] 8 core engines implemented with zero-hallucination formulas
- [ ] All FAO forest definitions codified (0.5ha, 10% canopy, 5m height)
- [ ] Historical reconstruction back to Dec 31, 2020 cutoff
- [ ] Deforestation-free verdict with 4 possible outcomes
- [ ] 500+ test functions with determinism verification
- [ ] V092 database migration with 10+ tables, hypertables, retention
- [ ] 30 API endpoints with auth protection
- [ ] 18 Prometheus metrics with Grafana dashboard
- [ ] RBAC integration with 18 permissions across 4 roles
- [ ] Provenance chain with SHA-256 hashing
- [ ] Evidence packaging in JSON/PDF/CSV/EUDR XML formats
