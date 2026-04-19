# PRD: AGENT-EUDR-007 -- GPS Coordinate Validator

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-007 |
| **Agent ID** | GL-EUDR-GCV-007 |
| **Component** | GPS Coordinate Validator Agent |
| **Category** | EUDR Regulatory Agent -- Geospatial Coordinate Intelligence |
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

EUDR Article 9(1)(d) mandates that every Due Diligence Statement (DDS) includes the geolocation of all production plots as GPS coordinates with adequate positional accuracy. In practice, GPS coordinates arrive from hundreds of diverse sources -- handheld GPS devices carried by smallholder farmers, mobile phone apps, survey-grade GNSS receivers, digitized paper maps, ERP system exports, certification body databases, and government land registries. Each source introduces unique error patterns:

- **Format inconsistency**: Coordinates arrive in decimal degrees, degrees-minutes-seconds (DMS), degrees-decimal-minutes (DDM), UTM easting/northing, MGRS grid references, and even local grid systems. Without automated parsing, operators mis-interpret formats causing plots to appear on wrong continents.
- **Coordinate swapping**: Latitude and longitude values are frequently transposed (lat/lon vs. lon/lat ordering varies by system -- GeoJSON uses lon/lat, most databases use lat/lon). A single swap can place a cocoa farm in Ghana (5.5N, -1.5W) in the Gulf of Guinea instead.
- **Datum confusion**: Coordinates collected on local datums (e.g., Pulkovo 1942, Indian 1975, Arc 1960) are submitted without datum transformation to WGS84, introducing errors of 100-1000+ meters.
- **Precision inadequacy**: EUDR requires "adequate positional accuracy" but coordinates are often truncated to 2-3 decimal places (1.1-11km resolution), making them useless for plot-level identification. Some arrive as integers (degree-level only).
- **Implausible locations**: Coordinates point to oceans, deserts, urban centers, or countries where the specified commodity cannot possibly be grown. Without plausibility checks, these pass silently into DDS submissions.
- **Duplicate/near-duplicate coordinates**: Multiple plots share identical coordinates (copy-paste errors) or cluster suspiciously close together, suggesting data fabrication.
- **Altitude anomalies**: Coordinates include elevation values that are impossible for the claimed commodity (e.g., coffee at sea level or palm oil above 2000m).
- **No standardized validation pipeline**: Operators lack a systematic way to validate, normalize, and certify coordinates before DDS submission.

Without solving these problems, operators risk DDS rejection, regulatory penalties of up to 4% of annual EU turnover, and inability to prove deforestation-free status.

### 1.2 Solution Overview

Agent-EUDR-007: GPS Coordinate Validator is a specialized agent that provides comprehensive validation, normalization, and quality assessment of GPS coordinates for EUDR compliance. It accepts coordinates in any common format, transforms them to canonical WGS84 decimal degrees, runs 15+ validation checks, assesses positional accuracy, and certifies coordinates as EUDR-ready or flags them for correction.

Core capabilities:

1. **Multi-format coordinate parsing** -- Parse coordinates from 10+ formats: decimal degrees, DMS, DDM, UTM, MGRS, plus common text variations. Automatic format detection with confidence scoring.
2. **Datum transformation** -- Transform coordinates from 30+ geodetic datums to WGS84 using Helmert 7-parameter transformations. Detect and warn about unspecified datums.
3. **Precision analysis** -- Assess coordinate precision from decimal places, compute ground resolution in meters, and verify adequacy for EUDR Article 9 requirements.
4. **Format validation** -- Validate coordinate ranges (lat -90 to 90, lon -180 to 180), detect common errors (swapped lat/lon, sign errors, hemisphere errors), and auto-correct where confidence is high.
5. **Spatial plausibility checking** -- Verify coordinates fall on land (not ocean), in the correct country, in a commodity-plausible region, and at a plausible elevation for the claimed commodity.
6. **Reverse geocoding** -- Look up country, administrative region, and land use context from coordinates to verify consistency with declared metadata.
7. **Accuracy assessment** -- Score coordinate quality on a 0-100 scale based on precision, plausibility, consistency, and source reliability. Classify as Gold/Silver/Bronze/Unverified.
8. **Compliance reporting** -- Generate validation reports for DDS submission, including EUDR Article 9 compliance certificates, batch validation summaries, and remediation guidance.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Format detection accuracy | >= 99% correct format identification | Test against 10,000+ coordinate samples |
| Datum transformation precision | < 1 meter positional error vs. reference | Cross-validation against EPSG Guidance Note 7-2 |
| Lat/lon swap detection | >= 98% recall | Test against known swapped datasets |
| Ocean/land classification | >= 99.5% accuracy | Comparison against coastline reference |
| Country identification | >= 99% accuracy | Test against admin boundary reference |
| Precision classification | 100% correct decimal-place analysis | Automated unit tests |
| Validation throughput | > 10,000 coordinates/second | Performance benchmark |
| EUDR compliance rate | 100% of certified coordinates accepted in DDS | EU Information System submission |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM**: 400,000+ EUDR-affected operators processing billions of GPS coordinate data points annually, representing a coordinate validation market of 500M-1B EUR.
- **SAM**: 100,000+ EU importers needing systematic coordinate validation for EUDR compliance, estimated at 150-300M EUR.
- **SOM**: Target 500+ enterprise customers in Year 1, representing 15M-25M EUR in coordinate validation module ARR.

### 2.2 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual GPS validation | Low cost | Error-prone; slow; no datum awareness | Automated; deterministic; 10,000+ coords/sec |
| Generic geocoding APIs (Google, Mapbox) | High accuracy reverse geocoding | No EUDR compliance; no datum transformation; per-call pricing | Purpose-built for EUDR; offline; deterministic |
| GIS desktop tools (QGIS, ArcGIS) | Full-featured | Manual workflow; no batch EUDR validation | Automated pipeline; API-first; batch processing |
| Survey-grade GNSS software | High precision | Expensive; not EUDR-specific; complex UI | EUDR-focused; simple API; commodity-aware |

---

## 3. Goals and Non-Goals

### 3.1 Goals

1. Parse and normalize GPS coordinates from 10+ input formats to canonical WGS84 decimal degrees
2. Transform coordinates from 30+ datums to WGS84 with sub-meter accuracy
3. Detect and flag precision issues, swapped coordinates, and sign errors
4. Verify spatial plausibility (land/ocean, country, commodity, elevation)
5. Score coordinate quality and classify into accuracy tiers
6. Generate EUDR Article 9 compliance reports
7. Process 10,000+ coordinates per second in batch mode

### 3.2 Non-Goals

1. Real-time GNSS signal processing or RTK correction
2. Satellite imagery analysis (AGENT-EUDR-003)
3. Polygon boundary management (AGENT-EUDR-006)
4. Supply chain mapping (AGENT-EUDR-001)
5. Mobile device GPS hardware integration
6. Indoor positioning or WiFi-based location

---

## 4. User Personas

### 4.1 Data Entry Operator -- Priya (Primary)
- **Role**: Supply chain data specialist at a commodity trader
- **Goal**: Upload and validate thousands of GPS coordinates from diverse suppliers
- **Pain point**: Coordinates arrive in mixed formats with no standardization; manual checking takes days
- **Key features**: Auto-format detection, batch validation, error highlighting with remediation guidance

### 4.2 Compliance Officer -- Maria (Primary)
- **Role**: EUDR compliance lead at a multinational food company
- **Goal**: Ensure all coordinates in DDS submissions meet EUDR Article 9 requirements
- **Pain point**: Cannot certify coordinate quality; no audit trail of validation
- **Key features**: Quality scoring, compliance certification, provenance tracking

### 4.3 Field Data Collector -- Kwame (Secondary)
- **Role**: GPS data collector visiting smallholder farms in West Africa
- **Goal**: Record accurate coordinates in the field using mobile devices
- **Pain point**: Low-precision GPS on mobile phones; uncertain datum; no immediate validation
- **Key features**: Precision assessment, datum detection, instant feedback

### 4.4 Auditor -- Klaus (Secondary)
- **Role**: Third-party EUDR auditor
- **Goal**: Verify that coordinate validation was performed correctly
- **Pain point**: No standardized validation methodology; no audit trail
- **Key features**: Validation reports, provenance chain, compliance certificates

---

## 5. Regulatory Requirements

### 5.1 EUDR Article 9 -- Geolocation Requirements

| Requirement | EUDR Reference | Implementation |
|-------------|---------------|----------------|
| GPS coordinates for all production plots | Article 9(1)(d) | Coordinate parsing and storage |
| Adequate positional accuracy | Article 9(1)(d) | Precision analysis and quality scoring |
| WGS84 coordinate reference system | Article 9(1)(d), implied | Datum transformation to WGS84 |
| Decimal latitude and longitude | Article 9(1)(d) | Format normalization to decimal degrees |
| Point geolocation for plots < 4 hectares | Article 9(1)(d) | Single coordinate validation |
| Polygon geolocation for plots >= 4 hectares | Article 9(1)(d) | Polygon vertex coordinate validation |

### 5.2 EUDR Article 31 -- Record Retention

| Requirement | Implementation |
|-------------|----------------|
| 5-year retention of geolocation data | Immutable validation records with 5-year retention |
| Audit trail | SHA-256 provenance chain for all coordinate operations |

---

## 6. Features and Requirements

### 6.1 Feature 1: Multi-Format Coordinate Parsing (P0)

**Requirements**:
- F1.1: Parse decimal degrees (DD): 5.603716, -0.186964
- F1.2: Parse degrees-minutes-seconds (DMS): 5d36'13.4"N, 0d11'13.1"W
- F1.3: Parse degrees-decimal-minutes (DDM): 5d36.2233'N, 0d11.2183'W
- F1.4: Parse UTM coordinates: 30N 808820 620350
- F1.5: Parse MGRS grid references: 30NUN0882020350
- F1.6: Parse signed decimal degrees with N/S/E/W suffixes
- F1.7: Parse comma-separated, space-separated, and tab-separated pairs
- F1.8: Auto-detect format from input string with confidence score
- F1.9: Handle common text variations (degree symbols, prime marks, Unicode)
- F1.10: Batch parsing of mixed-format inputs
- F1.11: Output canonical form: (latitude: float, longitude: float) in WGS84

### 6.2 Feature 2: Datum Transformation (P0)

**Requirements**:
- F2.1: Transform from 30+ geodetic datums to WGS84 using 7-parameter Helmert
- F2.2: Common datums: NAD27, NAD83, ED50, ETRS89, SIRGAS 2000, Indian 1975, Arc 1960, Pulkovo 1942, Tokyo, GDA94, GDA2020, NZGD2000
- F2.3: Auto-detect datum from country of origin when unspecified
- F2.4: Report transformation displacement in meters
- F2.5: Flag coordinates with unspecified or unknown datums
- F2.6: Molodensky transformation as fast approximation
- F2.7: Support geographic (lat/lon) and projected (easting/northing) inputs
- F2.8: Transformation accuracy reporting per datum pair

### 6.3 Feature 3: Precision Analysis (P0)

**Requirements**:
- F3.1: Count significant decimal places in coordinate values
- F3.2: Compute ground resolution from decimal places (5dp = ~1.1m at equator)
- F3.3: Latitude-dependent resolution (higher precision needed near poles)
- F3.4: Classify precision: survey_grade (< 1m), high (1-10m), moderate (10-100m), low (100-1000m), inadequate (> 1km)
- F3.5: EUDR adequacy check (minimum 5 decimal places recommended)
- F3.6: Detect truncated coordinates (e.g., integer-only)
- F3.7: Detect artificially rounded coordinates (e.g., xxx.000000)
- F3.8: Source-specific precision expectations (GNSS survey vs. mobile phone vs. manual entry)

### 6.4 Feature 4: Format Validation (P0)

**Requirements**:
- F4.1: Range check: latitude [-90, 90], longitude [-180, 180]
- F4.2: Lat/lon swap detection (statistical heuristic + country context)
- F4.3: Sign error detection (e.g., positive longitude in Western Hemisphere)
- F4.4: Hemisphere error detection (N/S or E/W mixup)
- F4.5: Zero-coordinate detection (0.0, 0.0 = Null Island)
- F4.6: NaN/Inf/null detection
- F4.7: Duplicate coordinate detection across datasets
- F4.8: Near-duplicate clustering (coordinates < 1m apart)
- F4.9: Auto-correction suggestions with confidence scores
- F4.10: Batch validation with error aggregation

### 6.5 Feature 5: Spatial Plausibility Checking (P0)

**Requirements**:
- F5.1: Land/ocean classification using coastline reference data
- F5.2: Country identification from coordinates (point-in-polygon against admin boundaries)
- F5.3: Country match validation (declared country vs. coordinate country)
- F5.4: Commodity plausibility check (commodity can be grown at this location)
- F5.5: Elevation plausibility for commodity (e.g., palm oil < 1500m, coffee 600-2200m)
- F5.6: Climate zone compatibility check
- F5.7: Protected area proximity check (within or near protected areas)
- F5.8: Urban area detection (coordinates in city centers unlikely to be farms)

### 6.6 Feature 6: Reverse Geocoding (P0)

**Requirements**:
- F6.1: Country lookup from coordinates (offline, no external API dependency)
- F6.2: Administrative region lookup (province/state level)
- F6.3: Nearest named place identification
- F6.4: Land use context classification (forest, agricultural, urban, water)
- F6.5: Distance to coast calculation
- F6.6: Distance to nearest road/infrastructure
- F6.7: Commodity production zone identification (known growing regions)
- F6.8: Batch reverse geocoding

### 6.7 Feature 7: Accuracy Assessment (P0)

**Requirements**:
- F7.1: Composite quality score 0-100 based on weighted factors
- F7.2: Scoring dimensions: precision (25%), plausibility (25%), consistency (25%), source_reliability (25%)
- F7.3: Tier classification: Gold (>= 90), Silver (70-89), Bronze (50-69), Unverified (< 50)
- F7.4: Per-dimension subscores with explanations
- F7.5: Confidence interval estimation for coordinate position
- F7.6: Comparison with nearby validated coordinates
- F7.7: Historical consistency check (same plot, different submissions)
- F7.8: Data quality trend analysis across submissions

### 6.8 Feature 8: Compliance Reporting (P0)

**Requirements**:
- F8.1: EUDR Article 9 compliance certificate generation
- F8.2: Batch validation summary report (valid/invalid/warning counts)
- F8.3: Error remediation guidance with specific fix instructions
- F8.4: JSON, PDF, CSV, and EUDR XML report formats
- F8.5: DDS-ready coordinate export (normalized, validated, certified)
- F8.6: Audit trail report with provenance chain
- F8.7: Quality trend report across time periods
- F8.8: Submission readiness assessment

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/gps_coordinate_validator/
    __init__.py                     # Package exports (80+ symbols)
    config.py                       # GPSCoordinateValidatorConfig singleton
    models.py                       # Pydantic v2 models, enums
    provenance.py                   # SHA-256 chain hashing
    metrics.py                      # Prometheus metrics (gl_eudr_gcv_ prefix)
    coordinate_parser.py            # Engine 1: Multi-format parsing
    datum_transformer.py            # Engine 2: Datum transformations
    precision_analyzer.py           # Engine 3: Precision assessment
    format_validator.py             # Engine 4: Range/swap/error checks
    spatial_plausibility_checker.py # Engine 5: Land/ocean/country/commodity
    reverse_geocoder.py             # Engine 6: Offline reverse geocoding
    accuracy_assessor.py            # Engine 7: Quality scoring
    compliance_reporter.py          # Engine 8: Report generation
    setup.py                        # GPSCoordinateValidatorService facade
    reference_data/
        __init__.py
        datum_parameters.py         # 30+ datum transformation parameters
        country_boundaries.py       # Country bounding boxes + centroids
        commodity_zones.py          # Commodity growing regions + elevation ranges
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        parsing_routes.py
        validation_routes.py
        plausibility_routes.py
        assessment_routes.py
        report_routes.py
        batch_routes.py
```

### 7.2 Database Schema (V095)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_gcv_validations` | hypertable (monthly) | Individual coordinate validation results |
| `gl_eudr_gcv_batch_validations` | hypertable (monthly) | Batch validation job records |
| `gl_eudr_gcv_transformations` | hypertable (monthly) | Datum transformation records |
| `gl_eudr_gcv_plausibility_checks` | hypertable (quarterly) | Spatial plausibility results |
| `gl_eudr_gcv_accuracy_scores` | hypertable (monthly) | Quality score records |
| `gl_eudr_gcv_compliance_certs` | regular | EUDR compliance certificates |
| `gl_eudr_gcv_reverse_geocodes` | regular | Reverse geocoding results |
| `gl_eudr_gcv_error_corrections` | regular | Auto-correction records |
| `gl_eudr_gcv_batch_jobs` | regular | Batch processing jobs |
| `gl_eudr_gcv_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_gcv_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_gcv_coordinates_parsed_total` | Counter | Total coordinates parsed |
| `gl_eudr_gcv_parse_errors_total` | Counter | Total parse failures |
| `gl_eudr_gcv_validations_total` | Counter | Total validations performed |
| `gl_eudr_gcv_validation_errors_total` | Counter | Total validation errors found |
| `gl_eudr_gcv_transformations_total` | Counter | Total datum transformations |
| `gl_eudr_gcv_precision_checks_total` | Counter | Total precision assessments |
| `gl_eudr_gcv_plausibility_checks_total` | Counter | Total plausibility checks |
| `gl_eudr_gcv_swap_detections_total` | Counter | Lat/lon swaps detected |
| `gl_eudr_gcv_ocean_detections_total` | Counter | Coordinates in ocean detected |
| `gl_eudr_gcv_country_mismatches_total` | Counter | Country mismatches detected |
| `gl_eudr_gcv_reverse_geocodes_total` | Counter | Total reverse geocoding operations |
| `gl_eudr_gcv_accuracy_scores_total` | Counter | Total accuracy assessments |
| `gl_eudr_gcv_compliance_certs_total` | Counter | Total compliance certificates |
| `gl_eudr_gcv_batch_jobs_total` | Counter | Total batch jobs |
| `gl_eudr_gcv_operation_duration_seconds` | Histogram | Operation latency |
| `gl_eudr_gcv_precision_decimal_places` | Histogram | Decimal places distribution |
| `gl_eudr_gcv_accuracy_score` | Histogram | Quality score distribution |
| `gl_eudr_gcv_api_errors_total` | Counter | Total API errors |

### 7.4 API Endpoints (~32 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Parsing | POST | `/api/v1/eudr-gcv/parse` | Parse single coordinate |
| | POST | `/api/v1/eudr-gcv/parse/batch` | Batch parse |
| | POST | `/api/v1/eudr-gcv/parse/detect-format` | Detect coordinate format |
| | POST | `/api/v1/eudr-gcv/parse/normalize` | Normalize to WGS84 DD |
| Validation | POST | `/api/v1/eudr-gcv/validate` | Validate single coordinate |
| | POST | `/api/v1/eudr-gcv/validate/batch` | Batch validation |
| | POST | `/api/v1/eudr-gcv/validate/range` | Range check only |
| | POST | `/api/v1/eudr-gcv/validate/swap-detect` | Detect lat/lon swaps |
| | POST | `/api/v1/eudr-gcv/validate/duplicates` | Detect duplicates |
| Plausibility | POST | `/api/v1/eudr-gcv/plausibility/check` | Full plausibility check |
| | POST | `/api/v1/eudr-gcv/plausibility/land-ocean` | Land/ocean check |
| | POST | `/api/v1/eudr-gcv/plausibility/country` | Country verification |
| | POST | `/api/v1/eudr-gcv/plausibility/commodity` | Commodity plausibility |
| | POST | `/api/v1/eudr-gcv/plausibility/elevation` | Elevation check |
| Assessment | POST | `/api/v1/eudr-gcv/assess` | Full accuracy assessment |
| | POST | `/api/v1/eudr-gcv/assess/batch` | Batch assessment |
| | GET | `/api/v1/eudr-gcv/assess/{coord_id}` | Get assessment result |
| | POST | `/api/v1/eudr-gcv/assess/precision` | Precision-only analysis |
| Reporting | POST | `/api/v1/eudr-gcv/reports/compliance` | Generate compliance cert |
| | POST | `/api/v1/eudr-gcv/reports/batch-summary` | Batch validation summary |
| | POST | `/api/v1/eudr-gcv/reports/remediation` | Remediation guidance |
| | GET | `/api/v1/eudr-gcv/reports/{report_id}` | Get report |
| | GET | `/api/v1/eudr-gcv/reports/{report_id}/download` | Download report |
| Reverse Geocoding | POST | `/api/v1/eudr-gcv/geocode/reverse` | Reverse geocode |
| | POST | `/api/v1/eudr-gcv/geocode/batch` | Batch reverse geocode |
| | POST | `/api/v1/eudr-gcv/geocode/country` | Country lookup |
| Datum | POST | `/api/v1/eudr-gcv/datum/transform` | Transform datum |
| | POST | `/api/v1/eudr-gcv/datum/batch` | Batch transform |
| | GET | `/api/v1/eudr-gcv/datum/list` | List supported datums |
| Batch | POST | `/api/v1/eudr-gcv/batch` | Submit batch job |
| | DELETE | `/api/v1/eudr-gcv/batch/{batch_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-gcv/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)
- All 10+ coordinate format parsing with edge cases
- All 30+ datum transformations with known reference points
- Precision analysis for 1-10 decimal places at multiple latitudes
- Lat/lon swap detection heuristics
- Land/ocean classification for coastal boundary points
- Country identification for border regions
- Commodity plausibility for all 7 EUDR commodities
- Quality scoring edge cases
- All report format generation

### 8.2 Performance Tests
- Batch parsing of 100,000 coordinates
- Batch validation throughput measurement
- R-tree spatial query performance

---

## Appendices

### A. Coordinate Format Examples

| Format | Example | Decimal Degrees |
|--------|---------|----------------|
| DD | 5.603716, -0.186964 | 5.603716, -0.186964 |
| DMS | 5d36'13.4"N, 0d11'13.1"W | 5.603722, -0.186972 |
| DDM | 5d36.2233'N, 0d11.2183'W | 5.603722, -0.186972 |
| UTM | 30N 808820 620350 | ~5.60, ~-0.19 |
| MGRS | 30NUN0882020350 | ~5.60, ~-0.19 |
| Signed DD | -5.603716, 0.186964 | -5.603716, 0.186964 |
| DD+suffix | 5.603716N, 0.186964W | 5.603716, -0.186964 |

### B. Precision Resolution Table

| Decimal Places | Resolution (equator) | Resolution (45 deg) | EUDR Adequacy |
|----------------|---------------------|---------------------|---------------|
| 0 | 111 km | 78.7 km | INADEQUATE |
| 1 | 11.1 km | 7.87 km | INADEQUATE |
| 2 | 1.11 km | 787 m | INADEQUATE |
| 3 | 111 m | 78.7 m | LOW |
| 4 | 11.1 m | 7.87 m | MODERATE |
| 5 | 1.11 m | 787 mm | HIGH |
| 6 | 111 mm | 78.7 mm | SURVEY_GRADE |
| 7 | 11.1 mm | 7.87 mm | SURVEY_GRADE |
| 8 | 1.11 mm | 0.787 mm | SURVEY_GRADE |

### C. Commodity Elevation Ranges

| Commodity | Min Elevation (m) | Max Elevation (m) | Typical Range |
|-----------|-------------------|-------------------|---------------|
| Palm oil | 0 | 1500 | 0-800 |
| Cocoa | 0 | 1200 | 100-800 |
| Coffee (Arabica) | 600 | 2200 | 900-1800 |
| Coffee (Robusta) | 0 | 1200 | 0-800 |
| Soya | 0 | 2000 | 0-1200 |
| Rubber | 0 | 1000 | 0-600 |
| Cattle | 0 | 4500 | 0-2500 |
| Wood/Timber | 0 | 4000 | 0-3000 |
