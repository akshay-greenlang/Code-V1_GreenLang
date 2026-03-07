# PRD: AGENT-EUDR-002 -- Geolocation Verification Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-002 |
| **Agent ID** | GL-EUDR-GEO-002 |
| **Component** | Geolocation Verification Agent |
| **Category** | EUDR Regulatory Agent -- Geospatial Intelligence |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-07 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-07 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

EUDR Article 9 mandates that every operator and trader must provide precise geolocation data for all plots of land where regulated commodities were produced. For plots exceeding 4 hectares, full polygon boundary data with GPS vertices is required. This geolocation data forms the foundation of deforestation-free verification -- without verified, accurate, and trustworthy geolocation data, every downstream compliance check (deforestation analysis, protected area overlap, country risk assessment) is compromised.

Today, EU operators face critical geolocation data quality challenges:

- **Unverified coordinates**: Suppliers submit GPS coordinates without any validation of accuracy, precision, or plausibility. Coordinates may be transposed (lat/lon swap), rounded to integer degrees (100+ km error), or entirely fabricated.
- **Invalid polygon geometries**: Polygon boundaries submitted for plots > 4 hectares contain self-intersections, overlapping vertices, unclosed rings, and topology errors that make spatial analysis impossible.
- **No cross-referencing against authoritative sources**: Submitted plot locations are not verified against satellite imagery, cadastral registries, or known agricultural zone boundaries.
- **No deforestation cutoff verification**: While AGENT-DATA-007 provides deforestation satellite alerts, there is no dedicated engine that systematically verifies every submitted plot against the December 31, 2020 deforestation cutoff date using multi-temporal satellite imagery.
- **No protected area overlap detection**: Plots located within national parks, UNESCO heritage sites, indigenous territories, or other protected areas represent automatic EUDR compliance failures that are not systematically detected.
- **No temporal consistency checking**: Plot boundaries that change over time (expanding into forested areas) are not tracked or flagged.
- **No accuracy scoring**: Operators have no quantified confidence in the accuracy of their geolocation data, making risk-based verification impossible.

Without a dedicated geolocation verification agent, operators risk submitting Due Diligence Statements (DDS) built on unverified spatial data, leading to regulatory rejection, penalties of up to 4% of annual EU turnover, and reputational damage.

### 1.2 Solution Overview

Agent-EUDR-002: Geolocation Verification Agent is a specialized geospatial intelligence agent that performs comprehensive, multi-layer verification of all plot geolocation data submitted for EUDR compliance. Unlike the AGENT-EUDR-001 GeolocationLinker (which links supply chain nodes to plots), this agent is the deep quality assurance engine that verifies every coordinate, every polygon boundary, and every spatial claim before it enters the compliance pipeline.

Core capabilities:

1. **Coordinate Validation Engine** -- Multi-layer validation of GPS coordinates: WGS84 bounds, precision assessment (decimal places), plausibility checking (coordinates fall within declared country/region), land vs. ocean verification, and duplicate coordinate detection across plots.
2. **Polygon Topology Verifier** -- Validates polygon geometries: ring closure, winding order (CCW exterior, CW holes), self-intersection detection, minimum vertex count, area calculation and verification against declared area, sliver polygon detection, and vertex density assessment.
3. **Protected Area Intersection Analyzer** -- Cross-references every plot against a comprehensive database of protected areas: WDPA (World Database on Protected Areas), Ramsar wetlands, UNESCO World Heritage Sites, indigenous territories, Key Biodiversity Areas, and national protected area registries.
4. **Deforestation Cutoff Verifier** -- Performs multi-temporal satellite analysis to verify that each plot was not forested as of December 31, 2020 (EUDR cutoff date). Uses NDVI time series from Sentinel-2 and Landsat, forest cover maps from Hansen/Global Forest Watch, and JAXA ALOS PALSAR forest/non-forest classification.
5. **Accuracy Scoring Engine** -- Generates a composite Geolocation Accuracy Score (GAS) from 0-100 for each plot based on coordinate precision, polygon quality, source reliability, cross-reference matches, and temporal consistency. This score drives risk-based verification prioritization.
6. **Temporal Consistency Analyzer** -- Tracks plot boundary changes over time, detects boundary expansion into forested areas, identifies seasonal agricultural patterns, and flags anomalous temporal changes that suggest data manipulation.
7. **Batch Verification Pipeline** -- Processes thousands of plots in parallel with configurable verification levels (Quick/Standard/Deep), progress tracking, and incremental result delivery.
8. **Article 9 Compliance Reporter** -- Generates Article 9-specific compliance reports showing verification status, identified issues, remediation requirements, and compliance readiness scores per plot, per commodity, and per operator.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Coordinate validation accuracy | 99.9% true positive rate | Validated against surveyed reference plots |
| Polygon topology error detection | 100% of invalid polygons caught | Test suite with known-bad polygons |
| Protected area overlap detection | 100% of overlapping plots flagged | Cross-reference with WDPA database |
| Deforestation cutoff verification | 95%+ accuracy (vs. manual review) | Comparison with expert manual analysis |
| False positive rate | < 2% for automated flags | Verified by human review of flagged plots |
| Batch processing throughput | 10,000 plots per hour (Standard level) | Load test with production-scale datasets |
| Single plot verification latency | < 5 seconds (Quick), < 30 seconds (Standard) | p99 latency measurement |
| Accuracy score correlation | > 0.85 correlation with expert assessment | Statistical validation against expert panel |
| Article 9 compliance rate improvement | 40% improvement in first 90 days | Before/after compliance readiness scores |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM**: 400,000+ EU operators requiring EUDR compliance, each with 10-10,000+ production plots requiring geolocation verification.
- **SAM**: 100,000+ operators with complex supply chains needing automated verification of submitted geolocation data.
- **SOM**: Target 500+ enterprise customers in Year 1, add-on module to GL-EUDR-APP platform.

### 2.2 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual GIS review | Flexible; expert judgment | Slow (hours per plot); expensive; not scalable | 10,000+ plots/hour automated verification |
| Generic GIS platforms (ArcGIS, QGIS) | Powerful spatial tools | Not EUDR-specific; no Article 9 compliance; no deforestation cutoff | Purpose-built for EUDR Article 9 with regulatory reporting |
| Satellite monitoring services (Planet, Satellogic) | High-resolution imagery | Image-only; no polygon validation; no compliance scoring | Full pipeline: validation + verification + scoring + compliance |
| Niche EUDR tools | EUDR awareness | Limited geospatial depth; manual polygon review | Deep automated geospatial verification with accuracy scoring |

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Ensure every plot in the system has verified, Article 9-compliant geolocation | 100% verification coverage | Q2 2026 |
| BG-2 | Reduce manual geolocation review effort by 90% | Time-to-verify per plot | Q2 2026 |
| BG-3 | Prevent submission of DDS with unverified spatial data | Zero unverified DDS submissions | Ongoing |
| BG-4 | Achieve zero false negatives for protected area overlaps | 100% detection rate | Q2 2026 |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Multi-layer coordinate validation | Validate coordinates against WGS84, country boundaries, ocean masks, and duplicate detection |
| PG-2 | Polygon topology assurance | Detect and classify all polygon geometry errors with auto-repair suggestions |
| PG-3 | Protected area screening | Cross-reference every plot against WDPA, Ramsar, UNESCO, indigenous territory databases |
| PG-4 | Deforestation cutoff verification | Verify each plot against Dec 31, 2020 forest status using multi-source satellite data |
| PG-5 | Accuracy scoring | Generate composite accuracy scores to drive risk-based verification |
| PG-6 | Temporal monitoring | Detect boundary changes and forest encroachment over time |
| PG-7 | Compliance reporting | Generate Article 9-specific compliance reports and readiness scores |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Batch processing throughput | 10,000 plots/hour (Standard verification) |
| TG-2 | Single plot verification | < 5s (Quick), < 30s (Standard), < 120s (Deep) |
| TG-3 | API response time | < 200ms p95 for status queries |
| TG-4 | Test coverage | >= 85% line coverage |
| TG-5 | Zero-hallucination | 100% deterministic spatial calculations |
| TG-6 | Memory efficiency | < 500 MB for 100,000-plot batch verification |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)
- Head of Regulatory Compliance at large EU chocolate manufacturer
- Needs to verify geolocation data for 5,000+ production plots across 12 countries
- Wants automated verification with clear pass/fail results and remediation guidance

### Persona 2: GIS Analyst -- Carlos (Primary)
- Geospatial analyst at an EU timber importer
- Reviews polygon boundaries and satellite imagery for EUDR compliance
- Wants automated polygon topology checks and deforestation cutoff verification to replace manual QGIS review

### Persona 3: Procurement Manager -- Ana (Secondary)
- Receives plot GPS data from suppliers via onboarding forms
- Needs immediate feedback on data quality before accepting supplier declarations
- Wants accuracy scores to prioritize which plots need field verification

### Persona 4: External Auditor -- Dr. Hofmann (Tertiary)
- Third-party EUDR auditor
- Needs verifiable, reproducible geolocation verification results
- Wants provenance-hashed audit trail of all verification decisions

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 9(1)(a)** | GPS coordinates for all plots of land | Coordinate Validation Engine verifies WGS84 coordinates |
| **Art. 9(1)(b-c)** | Polygon data for plots > 4 hectares | Polygon Topology Verifier enforces polygon requirement |
| **Art. 9(1)(d)** | Polygon with GPS points of all vertices | Vertex validation, ring closure, topology verification |
| **Art. 2(1)** | "Deforestation" definition -- forest to agricultural use after cutoff | Deforestation Cutoff Verifier checks Dec 31, 2020 baseline |
| **Art. 2(30-32)** | Definitions of "plot of land", "geolocation" | All verification aligned with EUDR geolocation definitions |
| **Art. 10(1)** | Risk assessment requirement | Accuracy scoring feeds into plot-level risk assessment |
| **Art. 10(2)(e)** | Concerns about country of production | Country boundary verification ensures declared country matches coordinates |
| **Art. 29** | Country benchmarking (Low/Standard/High) | Protected area and deforestation checks vary by country risk level |
| **Art. 31** | Record keeping for 5 years | All verification results stored with provenance hashes |

### 5.2 Geolocation Precision Requirements

| Requirement | EUDR Specification | Agent Implementation |
|-------------|-------------------|---------------------|
| Coordinate system | WGS84 (EPSG:4326) | Validates CRS, rejects non-WGS84 |
| Point precision | Sufficient to identify plot | Minimum 5 decimal places (~1.1m) required; 6+ recommended |
| Polygon precision | Sufficient to define boundary | Minimum 4 vertices; vertex spacing validated |
| Polygon closure | Ring must be closed | Auto-detect and flag unclosed rings |
| Area accuracy | Consistent with declared area | +-10% tolerance for area vs. declared area |
| Coordinate order | Latitude, Longitude | Detect and flag likely lat/lon transposition |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

---

#### Feature 1: Coordinate Validation Engine

**User Story:**
```
As a compliance officer,
I want every GPS coordinate submitted for a production plot to be automatically validated for accuracy, precision, and plausibility,
So that I can trust that the geolocation data in my DDS is correct and won't be rejected by regulators.
```

**Acceptance Criteria:**
- [ ] Validates WGS84 coordinate bounds: latitude [-90, 90], longitude [-180, 180]
- [ ] Assesses coordinate precision: counts decimal places, flags < 5 decimal places as low precision
- [ ] Detects likely latitude/longitude transposition (e.g., lat > 90 but valid if swapped)
- [ ] Verifies coordinates fall within declared country boundaries using country polygon database
- [ ] Detects ocean/water body coordinates using land mask (Natural Earth + GSHHG)
- [ ] Detects duplicate coordinates across plots (same location registered multiple times)
- [ ] Detects cluster anomalies (all plots at exact same coordinates suggests default/fake data)
- [ ] Validates elevation plausibility (coordinates at implausible elevations for commodity type)
- [ ] Generates coordinate validation report with pass/fail per check and overall score
- [ ] Supports batch validation of 10,000+ coordinates in single operation

**Non-Functional Requirements:**
- Performance: Single coordinate validation < 50ms; batch 10,000 < 30 seconds
- Accuracy: 99.9% true positive rate for invalid coordinate detection
- Determinism: Same coordinates always produce same validation result

**Dependencies:**
- Natural Earth country boundary dataset (built-in)
- GSHHG shoreline dataset for ocean masking
- SRTM/ASTER elevation dataset for elevation checks

---

#### Feature 2: Polygon Topology Verifier

**User Story:**
```
As a GIS analyst,
I want automated verification of every polygon boundary submitted for plots > 4 hectares,
So that I can ensure all polygon geometries are topologically valid for EUDR compliance and spatial analysis.
```

**Acceptance Criteria:**
- [ ] Validates ring closure: first vertex == last vertex (within tolerance)
- [ ] Validates winding order: exterior ring counter-clockwise (CCW), holes clockwise (CW)
- [ ] Detects self-intersecting polygons using Shamos-Hoey sweep line algorithm
- [ ] Validates minimum vertex count: >= 4 vertices (3 unique + closure)
- [ ] Calculates geodesic area using Karney's algorithm and compares to declared area (+-10%)
- [ ] Detects sliver polygons (area/perimeter^2 ratio below threshold)
- [ ] Detects spike vertices (sharp angles < 1 degree indicating data errors)
- [ ] Validates vertex density (minimum spacing between consecutive vertices)
- [ ] Validates polygon does not exceed maximum area threshold per commodity type
- [ ] Provides auto-repair suggestions for common issues (unclosed rings, wrong winding order)
- [ ] Generates topology verification report with issue classification and severity

**Non-Functional Requirements:**
- Performance: Single polygon verification < 200ms for polygons with up to 10,000 vertices
- Accuracy: 100% detection of topologically invalid polygons
- Determinism: Same polygon always produces same verification result

**Dependencies:**
- Shapely for geometry operations
- GeographicLib for geodesic calculations

---

#### Feature 3: Protected Area Intersection Analyzer

**User Story:**
```
As a compliance officer,
I want every production plot automatically screened against all known protected areas,
So that I can identify plots that overlap with protected areas and trigger enhanced due diligence.
```

**Acceptance Criteria:**
- [ ] Cross-references plots against WDPA (World Database on Protected Areas) -- 270,000+ areas
- [ ] Cross-references against Ramsar wetland sites -- 2,400+ sites
- [ ] Cross-references against UNESCO World Heritage Natural Sites -- 250+ sites
- [ ] Cross-references against Key Biodiversity Areas (KBAs) -- 16,000+ areas
- [ ] Cross-references against indigenous and community conserved territories (ICCAs)
- [ ] Cross-references against national/regional protected area registries (configurable per country)
- [ ] Calculates intersection percentage (what % of plot overlaps protected area)
- [ ] Classifies overlap severity: Full (>90%), Partial (10-90%), Marginal (<10%)
- [ ] Generates buffer zone analysis (plots within configurable distance of protected areas)
- [ ] Provides protected area details: name, IUCN category, designation year, managing authority
- [ ] Supports both point-based proximity checks and polygon-based intersection analysis

**Non-Functional Requirements:**
- Coverage: 100% of WDPA-listed protected areas included
- Performance: Single plot check < 500ms; batch 10,000 < 5 minutes
- Update frequency: Protected area database updated quarterly
- Accuracy: Zero false negatives for protected area overlaps

**Dependencies:**
- WDPA dataset (UNEP-WCMC) -- loaded as spatial index
- AGENT-DATA-006 GIS/Mapping Connector (BoundaryResolverEngine)
- PostGIS for spatial queries

---

#### Feature 4: Deforestation Cutoff Verifier

**User Story:**
```
As a compliance officer,
I want automated verification that each production plot was not converted from forest after December 31, 2020,
So that I can verify deforestation-free status as required by EUDR Article 2.
```

**Acceptance Criteria:**
- [ ] Queries forest status as of December 31, 2020 using Hansen Global Forest Change dataset
- [ ] Queries JAXA ALOS PALSAR forest/non-forest classification for 2020 baseline
- [ ] Analyzes NDVI time series (2018-2026) from Sentinel-2 to detect vegetation loss events
- [ ] Detects tree cover loss events post-cutoff using Global Forest Watch API
- [ ] Calculates canopy cover percentage at cutoff date for each plot
- [ ] Classifies plot deforestation status: Verified Clear (never forested), Verified Forest (forested at cutoff, still forested), Deforestation Detected (forested at cutoff, cleared after), Inconclusive (insufficient data)
- [ ] Handles cloud-cover gaps in satellite imagery using multi-temporal compositing
- [ ] Provides evidence package: before/after NDVI values, imagery dates, data sources
- [ ] Calculates deforestation confidence score based on number of corroborating data sources
- [ ] Integrates with AGENT-DATA-007 Deforestation Satellite Connector for satellite data access

**Non-Functional Requirements:**
- Accuracy: 95%+ agreement with expert manual analysis
- Temporal coverage: 2018-present satellite imagery
- Spatial resolution: 10m (Sentinel-2), 30m (Landsat), 25m (ALOS PALSAR)
- Determinism: Same plot + same satellite data = same result

**Dependencies:**
- AGENT-DATA-007 Deforestation Satellite Connector (production ready)
- Hansen Global Forest Change dataset
- Google Earth Engine or local tile cache for satellite imagery
- AGENT-DATA-006 GIS/Mapping Connector for spatial operations

---

#### Feature 5: Geolocation Accuracy Scoring Engine

**User Story:**
```
As a procurement manager,
I want a single accuracy score (0-100) for each plot's geolocation data,
So that I can prioritize which plots need field verification and track data quality improvement.
```

**Acceptance Criteria:**
- [ ] Calculates composite Geolocation Accuracy Score (GAS) from 0-100
- [ ] Component scores: Coordinate Precision (0-20), Polygon Quality (0-20), Country Match (0-15), Protected Area Clear (0-15), Deforestation Clear (0-15), Temporal Consistency (0-15)
- [ ] Assigns quality tier: Gold (90-100), Silver (70-89), Bronze (50-69), Unverified (<50)
- [ ] Provides breakdown of score components with pass/fail for each check
- [ ] Tracks score history over time (improvements after data corrections)
- [ ] Generates aggregate statistics per operator, per commodity, per country
- [ ] Supports configurable score weights per operator (adjustable via API)
- [ ] Provides score comparison against fleet-wide benchmarks
- [ ] Integrates with AGENT-EUDR-001 risk propagation (low GAS = higher risk)
- [ ] All scoring is deterministic: same input data = same score

**Scoring Formula:**
```
GAS = (
    coordinate_precision_score * W_precision +
    polygon_quality_score * W_polygon +
    country_match_score * W_country +
    protected_area_score * W_protected +
    deforestation_score * W_deforestation +
    temporal_consistency_score * W_temporal
)

Default Weights:
- W_precision = 0.20
- W_polygon = 0.20
- W_country = 0.15
- W_protected = 0.15
- W_deforestation = 0.15
- W_temporal = 0.15
```

**Non-Functional Requirements:**
- Determinism: Bit-perfect reproducibility
- Performance: Score calculation < 100ms per plot
- History: Store up to 365 days of score history per plot

---

#### Feature 6: Temporal Consistency Analyzer

**User Story:**
```
As a GIS analyst,
I want to track how plot boundaries change over time and detect expansions into forested areas,
So that I can identify ongoing deforestation risk and data manipulation attempts.
```

**Acceptance Criteria:**
- [ ] Stores historical versions of plot boundaries with timestamps
- [ ] Detects boundary expansion events (area increase > 5%)
- [ ] Detects boundary shift events (centroid movement > 100m)
- [ ] Calculates direction of expansion and cross-references against forest boundaries
- [ ] Identifies seasonal patterns vs. suspicious anomalies
- [ ] Detects rapid boundary changes (multiple changes within 30 days)
- [ ] Flags boundary changes that expand into areas that were forested at cutoff
- [ ] Generates temporal change report with visualization data
- [ ] Supports configurable alert thresholds for boundary changes
- [ ] Maintains provenance chain for all boundary version transitions

**Non-Functional Requirements:**
- History depth: Minimum 5 years (per EUDR Article 31)
- Detection latency: New boundary version analyzed within 1 hour of submission
- Determinism: Same boundary history = same analysis result

---

#### Feature 7: Batch Verification Pipeline

**User Story:**
```
As a compliance officer,
I want to submit thousands of plots for verification in a single operation with progress tracking,
So that I can efficiently verify my entire plot portfolio without manual intervention.
```

**Acceptance Criteria:**
- [ ] Supports three verification levels: Quick (coordinates + polygon only), Standard (+ protected areas + country check), Deep (+ deforestation + temporal)
- [ ] Processes plots in parallel with configurable concurrency (default: 50 concurrent)
- [ ] Provides real-time progress tracking (plots processed, pass/fail counts, ETA)
- [ ] Supports incremental result delivery (results available as each plot completes)
- [ ] Handles failures gracefully (failed plots retried, don't block others)
- [ ] Generates batch summary report with aggregate statistics
- [ ] Supports priority queuing (high-risk country plots verified first)
- [ ] Supports filtering (verify only plots modified since last verification)
- [ ] Rate-limits external API calls (satellite data, protected area lookups)
- [ ] Emits progress events via WebSocket or SSE for UI integration

**Non-Functional Requirements:**
- Throughput: 10,000 plots/hour (Standard), 50,000 plots/hour (Quick)
- Reliability: < 0.1% plot verification failure rate
- Memory: < 500 MB for 100,000-plot batch

---

#### Feature 8: Article 9 Compliance Reporter

**User Story:**
```
As a compliance officer,
I want a comprehensive Article 9 compliance report showing the verification status of all my plots,
So that I can demonstrate to regulators that my geolocation data meets EUDR requirements.
```

**Acceptance Criteria:**
- [ ] Generates per-plot verification status: Compliant, Non-Compliant, Pending Verification
- [ ] Shows Article 9 requirement checklist per plot (coordinates present, polygon if >4ha, verified)
- [ ] Generates per-commodity summary (% of plots compliant by commodity)
- [ ] Generates per-country summary (% of plots compliant by country)
- [ ] Identifies top remediation priorities (plots closest to compliance)
- [ ] Tracks compliance trend over time (weekly/monthly improvement)
- [ ] Exports reports as PDF, CSV, and JSON for regulatory submission
- [ ] Includes provenance hashes for report integrity verification
- [ ] Links to specific verification results for each plot
- [ ] Provides estimated effort to achieve full compliance (plots remaining x avg fix time)

**Non-Functional Requirements:**
- Report generation: < 30 seconds for 50,000-plot portfolio
- Format compliance: PDF reports meet auditor presentation standards
- Integrity: SHA-256 hash of complete report for tamper detection

---

### 6.2 Could-Have Features (P2)

#### Feature 9: Cadastral Cross-Reference
- Cross-reference plot boundaries against national cadastral/land registry databases
- Verify plot ownership claims against public land records
- Detect boundary conflicts with adjacent registered parcels

#### Feature 10: Field Verification Workflow
- Generate field verification task lists for ground-truth GPS validation
- Mobile app integration for field workers to capture verified coordinates
- Photo evidence collection with embedded GPS metadata

#### Feature 11: AI-Assisted Anomaly Detection
- Machine learning model to detect suspicious coordinate patterns
- Clustering analysis to identify systematic data quality issues
- Predictive model for deforestation risk based on spatial features

---

### 6.3 Won't-Have (P3 -- Out of Scope for v1.0)

- Real-time satellite imagery processing (rely on cached/pre-processed data)
- Sub-meter coordinate verification (requires RTK GPS hardware integration)
- 3D terrain analysis and slope verification
- Custom satellite tasking for specific plot monitoring
- Mobile native application for field verification

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                                    +---------------------------+
                                    |     GL-EUDR-APP v1.0      |
                                    |   Frontend (React/TS)     |
                                    +-------------+-------------+
                                                  |
                                    +-------------v-------------+
                                    |     Unified API Layer      |
                                    |       (FastAPI)            |
                                    +-------------+-------------+
                                                  |
            +-------------------------------------+--------------------------------------+
            |                                     |                                      |
+-----------v-----------+           +-------------v-------------+            +-----------v-----------+
| AGENT-EUDR-002        |           | AGENT-EUDR-001            |            | AGENT-DATA-007        |
| Geolocation           |<--------->| Supply Chain Mapping      |            | Deforestation         |
| Verification Agent    |           | Master                    |            | Satellite Connector   |
|                       |           |                           |            |                       |
| - Coordinate Validator|           | - Graph Engine             |           | - Sentinel-2 Client   |
| - Polygon Verifier    |           | - Multi-Tier Mapper        |           | - Landsat Client      |
| - Protected Area Chk  |           | - Risk Propagation         |           | - GFW Client          |
| - Deforestation Verif |           | - Gap Analysis             |           | - NDVI Calculator     |
| - Accuracy Scorer     |           | - Visualization API        |           | - Forest Change Det.  |
| - Temporal Analyzer   |           +---------------------------+            +-----------------------+
| - Batch Pipeline      |
| - Article 9 Reporter  |           +---------------------------+            +---------------------------+
+-----------+-----------+           | AGENT-DATA-006            |            | AGENT-DATA-005            |
            |                       | GIS/Mapping Connector     |            | EUDR Traceability         |
            +---------------------->|                           |            | Connector                 |
                                    | - PostGIS Queries         |            | - PlotRegistryEngine      |
                                    | - Spatial Indexing        |            | - ComplianceVerifier      |
                                    | - Protected Areas         |            +---------------------------+
                                    +-----------------------+
```

### 7.2 Module Structure

```
greenlang/agents/eudr/geolocation_verification/
    __init__.py                          # Public API exports
    config.py                            # GeolocationVerificationConfig (dataclass, singleton)
    models.py                            # Pydantic v2 models for verification requests/results
    coordinate_validator.py              # CoordinateValidator: multi-layer coordinate validation
    polygon_verifier.py                  # PolygonTopologyVerifier: geometry validation engine
    protected_area_checker.py            # ProtectedAreaChecker: WDPA/Ramsar/UNESCO screening
    deforestation_verifier.py            # DeforestationCutoffVerifier: Dec 31 2020 verification
    accuracy_scorer.py                   # AccuracyScoringEngine: composite GAS calculation
    temporal_analyzer.py                 # TemporalConsistencyAnalyzer: boundary change tracking
    batch_pipeline.py                    # BatchVerificationPipeline: parallel plot processing
    article9_reporter.py                 # Article9ComplianceReporter: regulatory reporting
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains
    metrics.py                           # 15 Prometheus self-monitoring metrics
    setup.py                             # GeolocationVerificationService facade
    reference_data/
        __init__.py
        country_boundaries.py            # Country boundary lookup (Natural Earth)
        ocean_mask.py                     # Land/ocean classification
        protected_areas.py               # WDPA/Ramsar/UNESCO spatial index
        elevation_data.py                # SRTM/ASTER elevation lookup
    api/
        __init__.py
        router.py                        # FastAPI router (20+ endpoints)
        schemas.py                       # API request/response schemas
        dependencies.py                  # FastAPI dependency injection
        coordinate_routes.py             # Coordinate validation endpoints
        polygon_routes.py                # Polygon verification endpoints
        protected_area_routes.py         # Protected area screening endpoints
        deforestation_routes.py          # Deforestation verification endpoints
        scoring_routes.py                # Accuracy scoring endpoints
        batch_routes.py                  # Batch verification endpoints
        compliance_routes.py             # Article 9 compliance report endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Verification Request
class VerificationRequest(BaseModel):
    plot_id: str
    coordinates: Tuple[float, float]         # (latitude, longitude)
    polygon: Optional[List[Tuple[float, float]]]  # Polygon vertices
    declared_area_hectares: Optional[float]
    declared_country_code: str               # ISO 3166-1 alpha-2
    declared_region: Optional[str]
    commodity: EUDRCommodity
    verification_level: VerificationLevel    # QUICK/STANDARD/DEEP

# Verification Level
class VerificationLevel(str, Enum):
    QUICK = "quick"         # Coordinates + polygon only
    STANDARD = "standard"   # + protected areas + country check
    DEEP = "deep"           # + deforestation + temporal

# Coordinate Validation Result
class CoordinateValidationResult(BaseModel):
    is_valid: bool
    wgs84_bounds_ok: bool
    precision_score: float                   # 0-100
    decimal_places: int
    country_match: bool
    declared_country: str
    detected_country: Optional[str]
    is_on_land: bool
    is_transposed: bool                      # Likely lat/lon swap
    is_duplicate: bool
    elevation_m: Optional[float]
    elevation_plausible: bool
    issues: List[CoordinateIssue]

# Polygon Verification Result
class PolygonVerificationResult(BaseModel):
    is_valid: bool
    is_closed: bool
    winding_order_correct: bool
    has_self_intersection: bool
    vertex_count: int
    min_vertex_count_met: bool
    calculated_area_hectares: float
    declared_area_hectares: Optional[float]
    area_within_tolerance: bool              # +-10%
    is_sliver: bool
    has_spike_vertices: bool
    spike_vertex_indices: List[int]
    vertex_density_ok: bool
    issues: List[PolygonIssue]
    repair_suggestions: List[RepairSuggestion]

# Protected Area Check Result
class ProtectedAreaCheckResult(BaseModel):
    has_overlap: bool
    overlapping_areas: List[ProtectedAreaOverlap]
    buffer_zone_areas: List[ProtectedAreaProximity]
    total_overlap_percentage: float
    overlap_severity: OverlapSeverity        # NONE/MARGINAL/PARTIAL/FULL
    highest_protection_level: Optional[str]  # IUCN category

# Deforestation Verification Result
class DeforestationVerificationResult(BaseModel):
    status: DeforestationStatus              # VERIFIED_CLEAR/VERIFIED_FOREST/DEFORESTATION_DETECTED/INCONCLUSIVE
    canopy_cover_at_cutoff_pct: Optional[float]
    canopy_cover_current_pct: Optional[float]
    tree_cover_loss_events: List[TreeCoverLossEvent]
    ndvi_baseline_2020: Optional[float]
    ndvi_current: Optional[float]
    data_sources_used: List[str]
    confidence_score: float                  # 0-100
    evidence_package: Dict[str, Any]

# Geolocation Accuracy Score
class GeolocationAccuracyScore(BaseModel):
    plot_id: str
    total_score: float                       # 0-100
    quality_tier: QualityTier                # GOLD/SILVER/BRONZE/UNVERIFIED
    coordinate_precision_score: float        # 0-20
    polygon_quality_score: float             # 0-20
    country_match_score: float               # 0-15
    protected_area_score: float              # 0-15
    deforestation_score: float               # 0-15
    temporal_consistency_score: float        # 0-15
    component_details: Dict[str, Any]
    calculated_at: datetime
    provenance_hash: str

# Batch Verification Result
class BatchVerificationResult(BaseModel):
    batch_id: str
    total_plots: int
    processed: int
    passed: int
    failed: int
    pending: int
    verification_level: VerificationLevel
    average_accuracy_score: float
    results: List[PlotVerificationResult]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
```

### 7.4 Database Schema (New Migration: V090)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_geolocation_verification;

-- Verification results per plot
CREATE TABLE eudr_geolocation_verification.plot_verifications (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    operator_id UUID NOT NULL,
    verification_level VARCHAR(20) NOT NULL DEFAULT 'standard',
    overall_status VARCHAR(30) NOT NULL DEFAULT 'pending',
    accuracy_score NUMERIC(5,2) DEFAULT 0.0,
    quality_tier VARCHAR(20) DEFAULT 'unverified',
    coordinate_result JSONB DEFAULT '{}',
    polygon_result JSONB DEFAULT '{}',
    protected_area_result JSONB DEFAULT '{}',
    deforestation_result JSONB DEFAULT '{}',
    temporal_result JSONB DEFAULT '{}',
    score_breakdown JSONB DEFAULT '{}',
    issues_count INTEGER DEFAULT 0,
    critical_issues_count INTEGER DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verified_by VARCHAR(100) DEFAULT 'system'
);

-- Batch verification jobs
CREATE TABLE eudr_geolocation_verification.batch_jobs (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    verification_level VARCHAR(20) NOT NULL DEFAULT 'standard',
    total_plots INTEGER NOT NULL,
    processed INTEGER DEFAULT 0,
    passed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    average_score NUMERIC(5,2) DEFAULT 0.0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

-- Protected area overlaps detected
CREATE TABLE eudr_geolocation_verification.protected_area_overlaps (
    overlap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    protected_area_id VARCHAR(100) NOT NULL,
    protected_area_name VARCHAR(500),
    protected_area_type VARCHAR(50),
    iucn_category VARCHAR(10),
    overlap_percentage NUMERIC(5,2),
    overlap_severity VARCHAR(20),
    designation_year INTEGER,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Deforestation events detected
CREATE TABLE eudr_geolocation_verification.deforestation_events (
    event_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    event_date DATE,
    tree_cover_loss_hectares NUMERIC(10,4),
    canopy_cover_before NUMERIC(5,2),
    canopy_cover_after NUMERIC(5,2),
    data_source VARCHAR(100),
    confidence_score NUMERIC(5,2),
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_geolocation_verification.deforestation_events', 'detected_at');

-- Accuracy score history (hypertable for time-series tracking)
CREATE TABLE eudr_geolocation_verification.accuracy_score_history (
    score_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    total_score NUMERIC(5,2),
    quality_tier VARCHAR(20),
    coordinate_precision_score NUMERIC(5,2),
    polygon_quality_score NUMERIC(5,2),
    country_match_score NUMERIC(5,2),
    protected_area_score NUMERIC(5,2),
    deforestation_score NUMERIC(5,2),
    temporal_consistency_score NUMERIC(5,2),
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_geolocation_verification.accuracy_score_history', 'scored_at');

-- Temporal boundary changes
CREATE TABLE eudr_geolocation_verification.boundary_changes (
    change_id UUID DEFAULT gen_random_uuid(),
    plot_id UUID NOT NULL,
    change_type VARCHAR(30) NOT NULL,
    area_before_hectares NUMERIC(10,4),
    area_after_hectares NUMERIC(10,4),
    centroid_shift_meters NUMERIC(10,2),
    expansion_direction VARCHAR(20),
    expands_into_forest BOOLEAN DEFAULT FALSE,
    previous_boundary JSONB,
    new_boundary JSONB,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_geolocation_verification.boundary_changes', 'detected_at');

-- Article 9 compliance snapshots
CREATE TABLE eudr_geolocation_verification.compliance_snapshots (
    snapshot_id UUID DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50),
    total_plots INTEGER,
    compliant_plots INTEGER,
    non_compliant_plots INTEGER,
    pending_plots INTEGER,
    compliance_rate NUMERIC(5,2),
    average_accuracy_score NUMERIC(5,2),
    top_issues JSONB DEFAULT '[]',
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_geolocation_verification.compliance_snapshots', 'snapshot_at');

-- Indexes
CREATE INDEX idx_verifications_plot_id ON eudr_geolocation_verification.plot_verifications(plot_id);
CREATE INDEX idx_verifications_operator ON eudr_geolocation_verification.plot_verifications(operator_id);
CREATE INDEX idx_verifications_status ON eudr_geolocation_verification.plot_verifications(overall_status);
CREATE INDEX idx_verifications_score ON eudr_geolocation_verification.plot_verifications(accuracy_score);
CREATE INDEX idx_verifications_tier ON eudr_geolocation_verification.plot_verifications(quality_tier);
CREATE INDEX idx_batch_operator ON eudr_geolocation_verification.batch_jobs(operator_id);
CREATE INDEX idx_batch_status ON eudr_geolocation_verification.batch_jobs(status);
CREATE INDEX idx_overlaps_plot ON eudr_geolocation_verification.protected_area_overlaps(plot_id);
CREATE INDEX idx_deforestation_plot ON eudr_geolocation_verification.deforestation_events(plot_id);
CREATE INDEX idx_score_history_plot ON eudr_geolocation_verification.accuracy_score_history(plot_id);
CREATE INDEX idx_boundary_changes_plot ON eudr_geolocation_verification.boundary_changes(plot_id);
CREATE INDEX idx_compliance_operator ON eudr_geolocation_verification.compliance_snapshots(operator_id);
CREATE INDEX idx_compliance_commodity ON eudr_geolocation_verification.compliance_snapshots(commodity);
```

### 7.5 API Endpoints (20+)

| Method | Path | Description |
|--------|------|-------------|
| **Coordinate Validation** | | |
| POST | `/v1/verify/coordinates` | Validate a single coordinate pair |
| POST | `/v1/verify/coordinates/batch` | Validate multiple coordinates |
| **Polygon Verification** | | |
| POST | `/v1/verify/polygon` | Verify a single polygon |
| POST | `/v1/verify/polygon/repair` | Attempt auto-repair of polygon issues |
| **Protected Area Screening** | | |
| POST | `/v1/verify/protected-areas` | Screen plot against protected areas |
| GET | `/v1/protected-areas/nearby` | List protected areas near a coordinate |
| **Deforestation Verification** | | |
| POST | `/v1/verify/deforestation` | Verify deforestation status of a plot |
| GET | `/v1/verify/deforestation/{plot_id}/evidence` | Get evidence package |
| **Full Verification** | | |
| POST | `/v1/verify/plot` | Full verification of a single plot |
| GET | `/v1/verify/plot/{plot_id}` | Get latest verification result |
| GET | `/v1/verify/plot/{plot_id}/history` | Get verification history |
| **Batch Verification** | | |
| POST | `/v1/verify/batch` | Submit batch verification job |
| GET | `/v1/verify/batch/{batch_id}` | Get batch job status and results |
| GET | `/v1/verify/batch/{batch_id}/progress` | Get real-time progress |
| DELETE | `/v1/verify/batch/{batch_id}` | Cancel a running batch job |
| **Accuracy Scoring** | | |
| GET | `/v1/scores/{plot_id}` | Get current accuracy score |
| GET | `/v1/scores/{plot_id}/history` | Get score history over time |
| GET | `/v1/scores/summary` | Get aggregate score statistics |
| **Compliance Reporting** | | |
| POST | `/v1/compliance/report` | Generate Article 9 compliance report |
| GET | `/v1/compliance/report/{report_id}` | Get generated report |
| GET | `/v1/compliance/summary` | Get compliance summary dashboard data |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_geo_coordinates_validated_total` | Counter | Coordinate validations performed |
| 2 | `gl_eudr_geo_polygons_verified_total` | Counter | Polygon verifications performed |
| 3 | `gl_eudr_geo_protected_area_checks_total` | Counter | Protected area screenings |
| 4 | `gl_eudr_geo_deforestation_checks_total` | Counter | Deforestation cutoff verifications |
| 5 | `gl_eudr_geo_plots_verified_total` | Counter | Full plot verifications by level and result |
| 6 | `gl_eudr_geo_batch_jobs_total` | Counter | Batch verification jobs submitted |
| 7 | `gl_eudr_geo_batch_plots_processed_total` | Counter | Plots processed in batch jobs |
| 8 | `gl_eudr_geo_scores_calculated_total` | Counter | Accuracy scores calculated |
| 9 | `gl_eudr_geo_compliance_reports_total` | Counter | Compliance reports generated |
| 10 | `gl_eudr_geo_issues_detected_total` | Counter | Issues detected by type and severity |
| 11 | `gl_eudr_geo_verification_duration_seconds` | Histogram | Verification latency by level |
| 12 | `gl_eudr_geo_batch_duration_seconds` | Histogram | Batch job total duration |
| 13 | `gl_eudr_geo_errors_total` | Counter | Errors by operation type |
| 14 | `gl_eudr_geo_active_batch_jobs` | Gauge | Currently running batch jobs |
| 15 | `gl_eudr_geo_avg_accuracy_score` | Gauge | Average accuracy score across all verified plots |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Geospatial | Shapely 2.0 + pyproj + GeographicLib | Polygon operations, CRS, geodesic calculations |
| Spatial DB | PostGIS | Spatial queries, protected area index |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables |
| Cache | Redis | Verification result caching, rate limit tracking |
| Satellite | AGENT-DATA-007 integration | Sentinel-2, Landsat, GFW access |
| Reference Data | Natural Earth + WDPA + GSHHG | Country boundaries, protected areas, shorelines |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Auth | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| RBAC | SEC-002 | Role-based access control |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing |

### 7.8 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-geo:coordinates:verify` | Validate GPS coordinates | Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:polygon:verify` | Verify polygon topology | Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:polygon:repair` | Attempt polygon auto-repair | GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:protected-areas:check` | Screen against protected areas | Viewer, Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:deforestation:verify` | Verify deforestation cutoff | Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:plots:verify` | Full plot verification | Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:plots:read` | View verification results | Viewer, Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:batch:submit` | Submit batch verification jobs | Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:batch:read` | View batch job status | Viewer, Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:batch:cancel` | Cancel running batch jobs | Compliance Officer, Admin |
| `eudr-geo:scores:read` | View accuracy scores | Viewer, Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:scores:configure` | Configure score weights | Compliance Officer, Admin |
| `eudr-geo:compliance:generate` | Generate compliance reports | Compliance Officer, Admin |
| `eudr-geo:compliance:read` | View compliance reports | Viewer, Analyst, GIS Specialist, Compliance Officer, Admin |
| `eudr-geo:audit:read` | View verification audit trail | Auditor, Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-DATA-005 PlotRegistryEngine | Plot data | Plot coordinates, polygons, declared areas -> verification input |
| AGENT-DATA-006 GIS/Mapping Connector | Spatial operations | Protected area boundaries, spatial queries |
| AGENT-DATA-007 Deforestation Satellite | Satellite data | NDVI time series, forest change, canopy cover |
| AGENT-EUDR-001 GeolocationLinker | Plot linkage | Linked plots requiring verification |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| AGENT-EUDR-001 Risk Propagation | Accuracy scores -> risk | Low GAS scores increase node risk |
| AGENT-EUDR-001 Gap Analyzer | Verification failures -> gaps | Failed verifications create compliance gaps |
| GL-EUDR-APP DDS Reporter | Verification status -> DDS | Verified plot data for DDS submission |
| GL-EUDR-APP Frontend | Verification UI | Scores, results, maps for user dashboard |

---

## 8. Test Strategy

### 8.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Coordinate Validation Tests | 150+ | WGS84 bounds, precision, country match, ocean, transposition, duplicates |
| Polygon Topology Tests | 120+ | Ring closure, winding, self-intersection, area, slivers, spikes |
| Protected Area Tests | 80+ | WDPA overlap, Ramsar, UNESCO, buffer zones, edge cases |
| Deforestation Verifier Tests | 100+ | Cutoff date, NDVI analysis, forest change, multi-source, inconclusive |
| Accuracy Scoring Tests | 80+ | Score calculation, weights, tiers, history, determinism |
| Temporal Analyzer Tests | 60+ | Boundary changes, expansion detection, forest encroachment |
| Batch Pipeline Tests | 50+ | Concurrency, progress, failure handling, priority queuing |
| Article 9 Compliance Tests | 40+ | Report generation, status classification, trend tracking |
| API Tests | 80+ | All 20+ endpoints, auth, error handling, pagination |
| Golden Tests | 50+ | Known-good/known-bad plots for all 7 commodities |
| Integration Tests | 30+ | Cross-agent with DATA-005/006/007 and EUDR-001 |
| Performance Tests | 20+ | Throughput, latency, memory under load |
| **Total** | **860+** |

### 8.2 Golden Test Plots

Each of the 7 commodities will have dedicated golden test plots:
1. Valid plot (correct coordinates, valid polygon, no protected area, no deforestation) -- expect PASS
2. Invalid coordinates (ocean, wrong country, transposed) -- expect FAIL
3. Invalid polygon (self-intersection, unclosed ring, wrong winding) -- expect FAIL
4. Protected area overlap (WDPA, Ramsar, UNESCO) -- expect FLAG
5. Deforestation detected (forest cleared post-cutoff) -- expect FLAG
6. Low precision coordinates (2 decimal places) -- expect WARNING
7. Boundary expansion into forest -- expect FLAG

Total: 7 commodities x 7 scenarios = 49 golden test plots

---

## 9. Timeline and Milestones

### Phase 1: Core Validation Engines (Weeks 1-4)
- Coordinate Validation Engine (Feature 1)
- Polygon Topology Verifier (Feature 2)
- Accuracy Scoring Engine (Feature 5)
- Core models, config, provenance, metrics

### Phase 2: Spatial Intelligence (Weeks 5-8)
- Protected Area Intersection Analyzer (Feature 3)
- Deforestation Cutoff Verifier (Feature 4)
- Temporal Consistency Analyzer (Feature 6)

### Phase 3: Pipeline and Reporting (Weeks 9-12)
- Batch Verification Pipeline (Feature 7)
- Article 9 Compliance Reporter (Feature 8)
- REST API Layer (20+ endpoints)

### Phase 4: Testing and Launch (Weeks 13-16)
- Complete test suite (860+ tests)
- Performance testing and optimization
- Integration testing with EUDR-001, DATA-005/006/007
- Database migration V090
- Production launch

---

## 10. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Satellite data gaps due to cloud cover | High | Medium | Multi-temporal compositing; multi-source fusion; flag as INCONCLUSIVE |
| R2 | WDPA database updates may change protected area boundaries | Medium | Medium | Quarterly update pipeline; version-tracked boundaries |
| R3 | False positives from automated deforestation detection | Medium | High | Conservative thresholds; multi-source corroboration; human review queue |
| R4 | Large batch jobs cause resource contention | Medium | Medium | Rate limiting; priority queuing; configurable concurrency limits |
| R5 | Coordinate transposition detection may have ambiguous cases | Medium | Low | Flag as WARNING (not ERROR); human review for ambiguous cases |
| R6 | EU updates Article 9 geolocation specifications | Low | Medium | Modular design; configurable validation rules |

---

## 11. Appendices

### Appendix A: EUDR Article 9 Full Text Reference

Per Article 9(1), geolocation information includes:
- (a) GPS coordinates (latitude, longitude) for all plots
- (b) Polygon with GPS vertices for plots > 4 ha (non-cattle commodities)
- (c) Polygon with GPS vertices for plots > 4 ha (cattle)
- (d) GPS coordinates of all establishments where cattle were kept

### Appendix B: IUCN Protected Area Categories

| Category | Name | Description |
|----------|------|-------------|
| Ia | Strict Nature Reserve | Strictly protected for biodiversity |
| Ib | Wilderness Area | Largely unmodified natural area |
| II | National Park | Ecosystem protection and recreation |
| III | Natural Monument | Conservation of specific features |
| IV | Habitat/Species Management | Active management for conservation |
| V | Protected Landscape/Seascape | Landscape conservation |
| VI | Protected Area with Sustainable Use | Sustainable use of natural resources |

### Appendix C: Quality Tier Definitions

| Tier | Score Range | Description | Regulatory Implication |
|------|------------|-------------|----------------------|
| Gold | 90-100 | Fully verified, all checks passed | DDS-ready, minimal due diligence |
| Silver | 70-89 | Mostly verified, minor issues | DDS-eligible with noted limitations |
| Bronze | 50-69 | Partially verified, significant issues | Enhanced due diligence required |
| Unverified | 0-49 | Insufficient verification | Not DDS-eligible, remediation needed |

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-07 | APPROVED |
| Engineering Lead | GL-EngineeringLead | 2026-03-07 | APPROVED |
| EUDR Regulatory Advisor | GL-RegulatoryAdvisor | 2026-03-07 | APPROVED |
| GIS Lead | GL-GISLead | 2026-03-07 | APPROVED |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-03-07 | GL-ProductManager | Initial PRD created covering all 8 P0 features, regulatory mapping, technical architecture, 860+ test targets |
