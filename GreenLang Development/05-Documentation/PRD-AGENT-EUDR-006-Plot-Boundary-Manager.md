# PRD: AGENT-EUDR-006 -- Plot Boundary Manager

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-006 |
| **Agent ID** | GL-EUDR-PBM-006 |
| **Component** | Plot Boundary Manager Agent |
| **Category** | EUDR Regulatory Agent -- Geospatial Boundary Intelligence |
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

The EU Deforestation Regulation (Regulation (EU) 2023/1115) mandates in Article 9 that every Due Diligence Statement (DDS) must include the geolocation of all plots of land where the relevant commodity was produced. For plots of land larger than four hectares, the geolocation must be provided as polygons using "adequate positional accuracy and format" (Article 9(1)(d)). For plots smaller than four hectares, a single GPS point with latitude and longitude suffices.

This creates an unprecedented spatial data management challenge:

- **No standardized boundary management**: Production plots across 7 regulated commodities span millions of hectares in tropical and temperate regions. Boundaries arrive from diverse sources -- hand-drawn GPS tracks from smallholders, high-resolution satellite-derived polygons from corporate plantations, cadastral records from national land registries, and certification body shapefiles. Each source has different accuracy, coordinate reference systems, and topological quality.
- **No polygon validation for EUDR compliance**: Raw boundary data frequently contains topological errors -- self-intersections, unclosed rings, duplicate vertices, sliver polygons, and invalid geometries that fail EUDR submission validation. Without automated repair, operators must manually fix thousands of polygons.
- **No geodetic area calculations**: EUDR's 4-hectare threshold for polygon vs. point geolocation requires precise geodetic area calculations on the WGS84 ellipsoid, not planar approximations. Incorrect area calculations cause operators to submit point geolocations for plots that legally require polygons, or vice versa.
- **No overlap detection at scale**: When multiple operators source from the same production region, overlapping plot boundaries create double-counting risk. EUDR competent authorities will flag overlapping claims, yet there is no automated system to detect and resolve boundary conflicts before DDS submission.
- **No boundary versioning**: Plot boundaries change over time as land is cleared, restored, split, or merged. EUDR Article 31 requires 5-year record retention. Without versioning, operators cannot demonstrate which boundary was valid at the time a commodity was produced.
- **No multi-format interoperability**: Boundaries arrive in GeoJSON, KML, Shapefile, WKT, WKB, GPX, and proprietary formats. Converting between formats while preserving topology and precision is error-prone without a canonical internal representation.
- **No split/merge tracking**: When cooperatives divide collective plots or adjacent plots are consolidated, the genealogy of boundary changes must be preserved for audit trails.
- **No simplification for bandwidth**: Transmitting full-resolution polygons (10,000+ vertices) to the EU Information System is impractical. Controlled simplification that stays within accuracy tolerances is needed.

Without solving these problems, operators risk DDS rejection, regulatory penalties of up to 4% of annual EU turnover, and inability to prove deforestation-free status.

### 1.2 Solution Overview

Agent-EUDR-006: Plot Boundary Manager is a specialized geospatial agent that provides end-to-end lifecycle management of production plot boundaries for EUDR compliance. It ingests, validates, stores, versions, analyzes, and exports plot polygons at the precision required by the regulation. The agent operates as the canonical spatial registry for all EUDR production plots, ensuring every boundary is topologically valid, geodetically accurate, properly versioned, and export-ready for DDS submission.

Core capabilities:

1. **Polygon lifecycle management** -- Create, read, update, delete plot boundaries with full CRUD support. Canonical storage in WGS84 (EPSG:4326) with automatic CRS transformation from 50+ input projections.
2. **Topological validation and repair** -- Automated detection and repair of 12+ topological errors (self-intersection, unclosed rings, duplicate vertices, spike removal, narrow gaps, sliver polygons). OGC Simple Features compliance.
3. **Geodetic area calculation** -- Precise area computation on the WGS84 ellipsoid using Karney's algorithm (GeographicLib). Automatic 4-hectare threshold determination for polygon vs. point geolocation requirement.
4. **Overlap detection and resolution** -- Spatial indexing (R-tree) for O(n log n) pairwise overlap detection across millions of plots. Overlap quantification, conflict classification, and resolution suggestions.
5. **Boundary versioning** -- Immutable version history with temporal queries ("What was this plot's boundary on 2020-12-31?"). Change tracking with diff visualization. EUDR Article 31 5-year retention.
6. **Simplification and generalization** -- Douglas-Peucker, Visvalingam-Whyatt, and topology-preserving simplification with configurable tolerance. Area deviation guarantees (< 1%).
7. **Split/merge operations** -- Genealogical tracking when plots divide or consolidate. Parent-child relationship chains with full provenance.
8. **Multi-format export** -- Export to GeoJSON, KML, WKT, WKB, Shapefile, GPX, EUDR XML, and GML. Format-specific validation on export.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Polygon validation accuracy | 100% OGC Simple Features compliance after repair | Automated topology test suite |
| Area calculation precision | < 0.01% deviation from GeographicLib reference | Cross-validation against reference implementations |
| 4-hectare threshold accuracy | 100% correct classification | Automated test with known-area polygons |
| Overlap detection recall | >= 99.5% of true overlaps detected | Benchmark against manually verified overlaps |
| Overlap detection speed | < 5 seconds for 100,000 plots | p99 latency under load |
| Boundary version query | < 100ms for temporal point queries | p99 latency for version retrieval |
| Simplification area preservation | < 1% area deviation | Automated area comparison before/after |
| CRS transformation accuracy | < 1 meter positional error | Comparison against proj.4 reference |
| Format conversion fidelity | Zero topology loss across all formats | Round-trip conversion test suite |
| EUDR DDS acceptance rate | 100% of exported boundaries accepted | EU Information System submission validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: An estimated 400,000+ EUDR-affected operators managing 50-100 million production plots globally, representing a boundary management technology market of 1-2 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers requiring boundary management for their supply chain plots across 7 regulated commodities, estimated at 300-500M EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 25M-40M EUR in boundary management module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers managing 10,000+ production plots across their supply chains
- Commodity traders aggregating plots from thousands of smallholders
- Certification bodies (FSC, RSPO, Rainforest Alliance) digitizing producer maps
- Cooperative management organizations consolidating member plot boundaries

**Secondary:**
- National land registry agencies in producer countries
- Satellite monitoring providers needing validated boundary inputs
- Agricultural development agencies mapping smallholder plots
- Customs authorities validating DDS geolocation submissions

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Desktop GIS (QGIS, ArcGIS) | Full-featured spatial analysis | Not EUDR-specific; no versioning; manual workflow | Automated EUDR compliance; API-first; versioned |
| GPS tracking apps (Mapbox, Google Maps) | Easy data collection | No polygon validation; no overlap detection; no EUDR export | Full validation pipeline; EUDR XML export |
| Certification platforms (Starling, Regrow) | Satellite integration | Limited boundary management; single commodity | All 7 commodities; full lifecycle management |
| Custom GIS solutions | Tailored workflows | Expensive to build/maintain; no EUDR compliance | Purpose-built for EUDR; zero-hallucination |

---

## 3. Goals and Non-Goals

### 3.1 Goals

1. Provide a canonical, versioned spatial registry for all EUDR production plot boundaries
2. Automatically validate and repair topological errors in polygon boundaries
3. Calculate precise geodetic areas and determine EUDR Article 9 requirements (polygon vs. point)
4. Detect and classify boundary overlaps across the entire plot registry
5. Support boundary versioning with temporal queries for EUDR Article 31 compliance
6. Enable controlled simplification while preserving area and topology
7. Track split/merge genealogy with full provenance chains
8. Export boundaries in all formats required by EUDR and partner systems

### 3.2 Non-Goals

1. Satellite imagery processing (handled by AGENT-EUDR-003)
2. Forest cover analysis (handled by AGENT-EUDR-004)
3. Land use classification (handled by AGENT-EUDR-005)
4. Supply chain graph modeling (handled by AGENT-EUDR-001)
5. GPS hardware integration or mobile data collection (AGENT-EUDR-015)
6. 3D terrain modeling or elevation analysis
7. Real-time GNSS correction or RTK processing

---

## 4. User Personas

### 4.1 Compliance Officer -- Maria (Primary)
- **Role**: EUDR compliance lead at a multinational food company
- **Goal**: Ensure all production plot boundaries in DDS submissions are topologically valid and meet EUDR requirements
- **Pain point**: Receives thousands of boundary files in mixed formats with frequent topological errors; manually fixing takes weeks
- **Key features needed**: Batch validation, auto-repair, EUDR export, overlap detection

### 4.2 Supply Chain Analyst -- David (Primary)
- **Role**: Supply chain mapping specialist at a commodity trader
- **Goal**: Maintain accurate, up-to-date boundaries for all supplier production plots
- **Pain point**: Boundaries change seasonally; no version history; cannot prove what boundary was valid at commodity production date
- **Key features needed**: Versioning, temporal queries, change tracking, split/merge support

### 4.3 GIS Specialist -- Amara (Secondary)
- **Role**: Geospatial data engineer at a certification body
- **Goal**: Standardize and validate boundary data from 50,000+ smallholders for RSPO/FSC certification and EUDR compliance
- **Pain point**: Data arrives in 10+ formats with inconsistent CRS; overlap detection across regions is manual
- **Key features needed**: Multi-format import, CRS transformation, batch overlap detection, simplification

### 4.4 Auditor -- Klaus (Secondary)
- **Role**: Third-party EUDR auditor for a Big Four firm
- **Goal**: Verify that historical boundary records match DDS submissions
- **Pain point**: No audit trail for boundary changes; cannot reconstruct past states
- **Key features needed**: Version history, provenance chain, temporal queries, diff visualization

---

## 5. Regulatory Requirements

### 5.1 EUDR Article 9 -- Geolocation Requirements

| Requirement | EUDR Reference | Implementation |
|-------------|---------------|----------------|
| Plot geolocation in DDS | Article 9(1)(d) | Plot boundary storage and export |
| Polygons for plots >= 4 hectares | Article 9(1)(d) | Geodetic area calculation + threshold check |
| GPS points for plots < 4 hectares | Article 9(1)(d) | Centroid calculation as fallback |
| Adequate positional accuracy | Article 9(1)(d) | CRS validation and transformation |
| Plot identification | Article 9(1)(c) | Unique plot ID with versioned boundary |

### 5.2 EUDR Article 31 -- Record Retention

| Requirement | Implementation |
|-------------|----------------|
| 5-year retention of geolocation data | Immutable version history with 5-year retention policy |
| Temporal reconstruction | Point-in-time boundary queries |
| Audit trail | SHA-256 provenance chain for all boundary operations |

### 5.3 Related Standards

| Standard | Relevance |
|----------|-----------|
| OGC Simple Features | Polygon topology compliance |
| ISO 19107 | Spatial schema for geometric primitives |
| EPSG:4326 (WGS84) | Canonical coordinate reference system |
| GeoJSON RFC 7946 | Primary exchange format |
| KML (OGC 07-147r2) | Visualization and Google Earth compatibility |
| Well-Known Text (ISO 13249) | Database spatial representation |

---

## 6. Features and Requirements

### 6.1 Feature 1: Polygon Lifecycle Management (P0)

**Description**: Full CRUD operations for plot boundaries with canonical WGS84 storage.

**Requirements**:
- F1.1: Create plot boundary from GeoJSON, KML, WKT, WKB, Shapefile, or GPX input
- F1.2: Automatic CRS detection and transformation to WGS84 (EPSG:4326)
- F1.3: Support for Polygon, MultiPolygon, and Point geometries
- F1.4: Plot metadata storage (plot_id, commodity, country, owner, certification)
- F1.5: Centroid calculation for point-geolocation fallback
- F1.6: Bounding box computation for spatial indexing
- F1.7: Batch import of 10,000+ boundaries in a single operation
- F1.8: Canonical internal format with full precision preservation

### 6.2 Feature 2: Topological Validation and Repair (P0)

**Description**: Automated detection and repair of 12+ polygon topology errors.

**Requirements**:
- F2.1: Self-intersection detection and repair (node insertion at crossings)
- F2.2: Unclosed ring detection and closure
- F2.3: Duplicate vertex removal with configurable tolerance
- F2.4: Spike and narrow appendage removal
- F2.5: Sliver polygon detection (aspect ratio check)
- F2.6: Ring orientation enforcement (CCW exterior, CW holes)
- F2.7: Invalid coordinate detection (NaN, Inf, out-of-range)
- F2.8: Minimum vertex count enforcement (>= 4 for closed polygon)
- F2.9: Hole-in-polygon containment validation
- F2.10: OGC Simple Features compliance verification
- F2.11: Repair report with before/after statistics
- F2.12: Confidence scoring for repaired boundaries

### 6.3 Feature 3: Geodetic Area Calculation (P0)

**Description**: Precise area computation on the WGS84 ellipsoid.

**Requirements**:
- F3.1: Karney's algorithm for ellipsoidal polygon area (GeographicLib)
- F3.2: Vincenty's formula as fallback for simpler cases
- F3.3: Planar area in UTM projection for comparison
- F3.4: EUDR 4-hectare threshold classification (polygon_required vs. point_sufficient)
- F3.5: Perimeter calculation (geodetic and planar)
- F3.6: Compactness index (Polsby-Popper, Schwartzberg, Convex Hull Ratio)
- F3.7: Area uncertainty estimation based on coordinate precision
- F3.8: Unit conversion (m2, hectares, acres, km2)

### 6.4 Feature 4: Overlap Detection and Resolution (P0)

**Description**: Spatial indexing and O(n log n) overlap detection across plot registries.

**Requirements**:
- F4.1: R-tree spatial index for efficient candidate pair identification
- F4.2: Precise intersection geometry computation for candidate pairs
- F4.3: Overlap area and percentage calculation
- F4.4: Overlap classification (minor < 1%, moderate 1-10%, major 10-50%, critical > 50%)
- F4.5: Conflict owner identification (which operators claim overlapping areas)
- F4.6: Resolution suggestions (boundary adjustment, priority assignment, arbitration)
- F4.7: Batch overlap analysis for entire registries (100,000+ plots)
- F4.8: Temporal overlap detection (did boundaries overlap at a specific date?)

### 6.5 Feature 5: Boundary Versioning (P0)

**Description**: Immutable version history with temporal queries for EUDR Article 31.

**Requirements**:
- F5.1: Automatic version increment on boundary update
- F5.2: Immutable version records (never modified or deleted)
- F5.3: Point-in-time queries ("boundary at date X")
- F5.4: Version diff computation (added/removed area between versions)
- F5.5: Change reason tracking (survey_update, split, merge, correction, seasonal)
- F5.6: Version metadata (author, timestamp, source, accuracy)
- F5.7: Version lineage traversal (first version to current)
- F5.8: 5-year retention policy with configurable extension

### 6.6 Feature 6: Simplification and Generalization (P0)

**Description**: Controlled polygon simplification with topology and area preservation.

**Requirements**:
- F6.1: Douglas-Peucker simplification with configurable tolerance
- F6.2: Visvalingam-Whyatt area-based simplification
- F6.3: Topology-preserving simplification (no self-intersection introduced)
- F6.4: Area deviation guarantee (< 1% change after simplification)
- F6.5: Vertex count target mode (simplify to N vertices)
- F6.6: Multi-resolution output (original, standard, simplified, minimal)
- F6.7: Simplification quality metrics (area change, Hausdorff distance, vertex reduction ratio)
- F6.8: Batch simplification for export optimization

### 6.7 Feature 7: Split/Merge Operations (P0)

**Description**: Genealogical boundary operations with full provenance tracking.

**Requirements**:
- F7.1: Split a plot into 2+ child plots along a cutting line
- F7.2: Merge 2+ adjacent plots into a single consolidated boundary
- F7.3: Parent-child relationship tracking with genealogy graph
- F7.4: Area conservation verification (sum of children = parent area)
- F7.5: Attribute inheritance rules (commodity, owner, certification propagation)
- F7.6: Temporal effective dates for split/merge operations
- F7.7: Provenance chain linking split/merge to source operation
- F7.8: Undo/rollback support for recent operations

### 6.8 Feature 8: Multi-Format Export and Compliance Reporting (P0)

**Description**: Export boundaries in all EUDR-required formats with compliance validation.

**Requirements**:
- F8.1: GeoJSON export (RFC 7946 compliant)
- F8.2: KML export (OGC 07-147r2 compliant)
- F8.3: WKT export (ISO 13249)
- F8.4: WKB export (binary format)
- F8.5: Shapefile export (.shp/.shx/.dbf/.prj bundle)
- F8.6: EUDR XML namespace export for DDS submission
- F8.7: GPX export for field verification
- F8.8: GML export for OGC web services
- F8.9: Format validation on export (schema compliance)
- F8.10: Batch export with ZIP packaging
- F8.11: Compliance summary report (valid/invalid/needs-repair counts)
- F8.12: Export metadata (CRS, precision, simplification level, timestamp)

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/plot_boundary/
    __init__.py                 # Package exports (80+ symbols)
    config.py                   # PlotBoundaryConfig singleton (GL_EUDR_PBM_ prefix)
    models.py                   # Pydantic v2 models, enums, data classes
    provenance.py               # SHA-256 chain hashing for audit trails
    metrics.py                  # Prometheus metrics (gl_eudr_pbm_ prefix)
    polygon_manager.py          # Engine 1: CRUD + CRS transformation
    boundary_validator.py       # Engine 2: 12+ topology checks + repair
    area_calculator.py          # Engine 3: Geodetic area + 4ha threshold
    overlap_detector.py         # Engine 4: R-tree + intersection analysis
    boundary_versioner.py       # Engine 5: Immutable version history
    simplification_engine.py    # Engine 6: Douglas-Peucker + Visvalingam
    split_merge_engine.py       # Engine 7: Split/merge + genealogy
    compliance_reporter.py      # Engine 8: Multi-format export + reporting
    setup.py                    # PlotBoundaryService facade
    reference_data/
        __init__.py
        projection_parameters.py  # 50+ CRS definitions + transformation params
        boundary_standards.py     # OGC/ISO/EUDR boundary standards
        simplification_rules.py   # Simplification tolerance presets
    api/
        __init__.py
        router.py               # Main router aggregating sub-routers
        schemas.py              # Pydantic request/response schemas
        dependencies.py         # Auth, rate limiting, dependency injection
        boundary_routes.py      # CRUD endpoints
        validation_routes.py    # Validation + repair endpoints
        area_routes.py          # Area calculation endpoints
        overlap_routes.py       # Overlap detection endpoints
        version_routes.py       # Version management endpoints
        export_routes.py        # Multi-format export endpoints
```

### 7.2 Database Schema (V094)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_pbm_boundaries` | hypertable (monthly) | Plot boundary records with geometry |
| `gl_eudr_pbm_versions` | hypertable (monthly) | Immutable boundary version history |
| `gl_eudr_pbm_validations` | hypertable (monthly) | Topology validation results |
| `gl_eudr_pbm_overlaps` | hypertable (quarterly) | Overlap detection results |
| `gl_eudr_pbm_area_calculations` | hypertable (monthly) | Geodetic area computation records |
| `gl_eudr_pbm_simplifications` | regular | Simplification operation records |
| `gl_eudr_pbm_split_merges` | regular | Split/merge genealogy records |
| `gl_eudr_pbm_exports` | regular | Export operation records |
| `gl_eudr_pbm_batch_jobs` | regular | Batch processing jobs |
| `gl_eudr_pbm_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_pbm_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_pbm_boundaries_created_total` | Counter | Total boundaries created |
| `gl_eudr_pbm_boundaries_updated_total` | Counter | Total boundaries updated |
| `gl_eudr_pbm_validations_total` | Counter | Total validations performed |
| `gl_eudr_pbm_validation_errors_total` | Counter | Total validation errors found |
| `gl_eudr_pbm_repairs_total` | Counter | Total auto-repairs performed |
| `gl_eudr_pbm_area_calculations_total` | Counter | Total area calculations |
| `gl_eudr_pbm_overlaps_detected_total` | Counter | Total overlaps detected |
| `gl_eudr_pbm_overlap_scans_total` | Counter | Total overlap scan operations |
| `gl_eudr_pbm_versions_created_total` | Counter | Total versions created |
| `gl_eudr_pbm_simplifications_total` | Counter | Total simplifications performed |
| `gl_eudr_pbm_splits_total` | Counter | Total split operations |
| `gl_eudr_pbm_merges_total` | Counter | Total merge operations |
| `gl_eudr_pbm_exports_total` | Counter | Total exports by format |
| `gl_eudr_pbm_batch_jobs_total` | Counter | Total batch jobs |
| `gl_eudr_pbm_operation_duration_seconds` | Histogram | Operation latency |
| `gl_eudr_pbm_polygon_vertex_count` | Histogram | Vertex count distribution |
| `gl_eudr_pbm_area_hectares` | Histogram | Area distribution |
| `gl_eudr_pbm_api_errors_total` | Counter | Total API errors |

### 7.4 API Endpoints (~32 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Boundary CRUD | POST | `/api/v1/eudr-pbm/boundaries` | Create plot boundary |
| | GET | `/api/v1/eudr-pbm/boundaries/{plot_id}` | Get plot boundary |
| | PUT | `/api/v1/eudr-pbm/boundaries/{plot_id}` | Update plot boundary |
| | DELETE | `/api/v1/eudr-pbm/boundaries/{plot_id}` | Delete plot boundary |
| | POST | `/api/v1/eudr-pbm/boundaries/batch` | Batch create boundaries |
| | POST | `/api/v1/eudr-pbm/boundaries/search` | Search boundaries by area/commodity/country |
| Validation | POST | `/api/v1/eudr-pbm/validate` | Validate polygon topology |
| | POST | `/api/v1/eudr-pbm/validate/batch` | Batch validation |
| | POST | `/api/v1/eudr-pbm/repair` | Validate and auto-repair |
| | POST | `/api/v1/eudr-pbm/repair/batch` | Batch repair |
| Area | POST | `/api/v1/eudr-pbm/area/calculate` | Calculate geodetic area |
| | POST | `/api/v1/eudr-pbm/area/batch` | Batch area calculation |
| | POST | `/api/v1/eudr-pbm/area/threshold` | Check 4-hectare threshold |
| Overlap | POST | `/api/v1/eudr-pbm/overlaps/detect` | Detect overlaps for a plot |
| | POST | `/api/v1/eudr-pbm/overlaps/scan` | Full registry overlap scan |
| | GET | `/api/v1/eudr-pbm/overlaps/{plot_id}` | Get overlap records |
| | POST | `/api/v1/eudr-pbm/overlaps/resolve` | Suggest overlap resolution |
| Version | GET | `/api/v1/eudr-pbm/versions/{plot_id}` | Get version history |
| | GET | `/api/v1/eudr-pbm/versions/{plot_id}/at` | Get boundary at date |
| | GET | `/api/v1/eudr-pbm/versions/{plot_id}/diff` | Get version diff |
| | GET | `/api/v1/eudr-pbm/versions/{plot_id}/lineage` | Get version lineage |
| Export | POST | `/api/v1/eudr-pbm/export/geojson` | Export to GeoJSON |
| | POST | `/api/v1/eudr-pbm/export/kml` | Export to KML |
| | POST | `/api/v1/eudr-pbm/export/shapefile` | Export to Shapefile |
| | POST | `/api/v1/eudr-pbm/export/eudr-xml` | Export to EUDR XML |
| | POST | `/api/v1/eudr-pbm/export/batch` | Batch multi-format export |
| | GET | `/api/v1/eudr-pbm/export/{export_id}` | Get export result |
| Split/Merge | POST | `/api/v1/eudr-pbm/split` | Split plot boundary |
| | POST | `/api/v1/eudr-pbm/merge` | Merge plot boundaries |
| | GET | `/api/v1/eudr-pbm/genealogy/{plot_id}` | Get split/merge genealogy |
| Batch | POST | `/api/v1/eudr-pbm/batch` | Submit batch job |
| | DELETE | `/api/v1/eudr-pbm/batch/{batch_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-pbm/health` | Health check |

### 7.5 Integration Points

| System | Integration |
|--------|------------|
| AGENT-EUDR-001 | Supply chain nodes reference plot boundaries |
| AGENT-EUDR-002 | Geolocation verification uses validated boundaries |
| AGENT-EUDR-003 | Satellite monitoring clips imagery to plot boundaries |
| AGENT-EUDR-004 | Forest cover analysis bounded by plot polygons |
| AGENT-EUDR-005 | Land use change detection within plot boundaries |
| GL-EUDR-APP | DDS export uses validated, simplified boundaries |
| AGENT-DATA-006 | GIS/Mapping connector provides raw boundary data |

---

## 8. UX and Workflow

### 8.1 Primary Workflow: Boundary Ingestion and Validation

1. User uploads boundary data (GeoJSON, KML, Shapefile, etc.)
2. System detects CRS and transforms to WGS84
3. Topological validation runs automatically
4. Invalid boundaries are auto-repaired where possible
5. Geodetic area calculated; 4-hectare threshold applied
6. Boundary stored with version 1, provenance recorded
7. Overlap scan triggered against existing boundaries
8. Validation report returned with statistics

### 8.2 DDS Export Workflow

1. User selects plots for DDS submission
2. System validates all selected boundaries (topology + area)
3. Boundaries simplified to EUDR submission tolerance
4. Export generated in EUDR XML format
5. Compliance summary report included
6. ZIP package created with boundaries + metadata

---

## 9. Success Criteria

### 9.1 Launch Criteria (v1.0)

- [ ] 8 core engines fully implemented and tested
- [ ] 500+ unit tests with 90%+ coverage
- [ ] All 12+ topological error types detected and repaired
- [ ] Geodetic area calculation < 0.01% error vs. GeographicLib
- [ ] Overlap detection handles 100,000+ plots in < 30 seconds
- [ ] 8 export formats supported with round-trip fidelity
- [ ] V094 database migration deployed
- [ ] Grafana monitoring dashboard operational
- [ ] Auth/RBAC integration complete
- [ ] API documentation complete

### 9.2 Quality Criteria

- Zero-hallucination: All calculations deterministic, no LLM in critical path
- Provenance: SHA-256 chain hash on every boundary operation
- Reproducibility: Identical inputs produce identical outputs
- Performance: p99 latency < 500ms for single-plot operations

---

## 10. Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Phase 1 | Sprint 1 | Foundation + core engines (polygon manager, validator, area calculator) |
| Phase 2 | Sprint 1 | Remaining engines (overlap, versioning, simplification, split/merge, export) |
| Phase 3 | Sprint 2 | API layer, integration, testing, deployment |

---

## 11. Dependencies

| Dependency | Status | Impact |
|-----------|--------|--------|
| AGENT-EUDR-001 (Supply Chain) | BUILT | Plot IDs referenced in supply chain nodes |
| AGENT-EUDR-002 (Geolocation) | BUILT | Validated coordinates feed into boundaries |
| INFRA-002 (PostgreSQL+TimescaleDB) | PRODUCTION READY | Spatial data storage |
| SEC-001/002 (Auth/RBAC) | PRODUCTION READY | API security |
| OBS-001/002 (Prometheus/Grafana) | PRODUCTION READY | Monitoring |

---

## 12. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Large polygon vertex counts degrade performance | Medium | High | Spatial indexing + simplification engine |
| CRS transformation introduces positional errors | Low | High | Validated transformation chains + error bounds |
| Overlap detection quadratic scaling | Medium | Medium | R-tree spatial index for O(n log n) |
| Complex self-intersection repair fails | Low | Medium | Fallback to convex hull with warning |
| TimescaleDB geometry support limitations | Low | Low | Store WKB in BYTEA with separate spatial index |

---

## 13. Test Strategy

### 13.1 Unit Tests (500+)
- Polygon creation and CRS transformation
- All 12 topological error types (detection + repair)
- Geodetic area calculation edge cases (poles, anti-meridian, large polygons)
- Overlap detection accuracy (true positives, false negatives)
- Version history CRUD and temporal queries
- Simplification tolerance and area preservation
- Split/merge operations and genealogy
- All 8 export formats with round-trip validation

### 13.2 Integration Tests
- End-to-end ingestion pipeline
- Batch operations at scale (10,000+ boundaries)
- API endpoint response validation
- Auth/RBAC permission enforcement

### 13.3 Performance Tests
- Overlap detection at 100,000 plots
- Batch validation at 50,000 polygons
- Area calculation for complex polygons (10,000+ vertices)

---

## Appendices

### A. EUDR Article 9 -- Geolocation (Full Text Extract)

> The due diligence statement shall contain the following information: ... (d) geolocation of all plots of land where the relevant commodities were produced, as well as the date or time range of production; geolocation shall, for plots of land larger than four hectares used for production of the relevant commodities other than cattle, be provided using polygons with sufficient number of decimal latitude and longitude points to describe the perimeter of each plot of land.

### B. Supported Coordinate Reference Systems (Top 10)

| EPSG Code | Name | Region |
|-----------|------|--------|
| 4326 | WGS84 | Global (canonical) |
| 32601-32660 | UTM Zones 1-60N | Northern hemisphere |
| 32701-32760 | UTM Zones 1-60S | Southern hemisphere |
| 3857 | Web Mercator | Web mapping |
| 4674 | SIRGAS 2000 | South America |
| 4674 | GDA2020 | Australia |
| 4258 | ETRS89 | Europe |
| 32737 | UTM Zone 37S | East Africa |
| 32748 | UTM Zone 48S | Indonesia/SE Asia |
| 4269 | NAD83 | North America |

### C. Topological Error Catalog

| Error Type | Description | Auto-Repair Strategy |
|-----------|-------------|---------------------|
| Self-intersection | Ring crosses itself | Insert node at crossing, split into valid rings |
| Unclosed ring | First/last vertices differ | Append copy of first vertex |
| Duplicate vertices | Consecutive identical points | Remove duplicates (configurable tolerance) |
| Spike/appendage | Narrow protruding feature | Remove spike vertices |
| Sliver polygon | Very thin polygon (high aspect ratio) | Flag for manual review |
| Wrong orientation | Exterior CW, hole CCW | Reverse ring direction |
| Invalid coordinates | NaN, Inf, out of range | Remove or interpolate |
| Too few vertices | < 4 points for polygon | Flag as invalid (cannot repair) |
| Hole outside shell | Hole ring outside exterior | Remove hole or flag |
| Overlapping holes | Interior rings intersect | Union overlapping holes |
| Nested shells | Multiple exterior rings when single expected | Convert to MultiPolygon |
| Zero-area polygon | Degenerate polygon | Flag and remove |
