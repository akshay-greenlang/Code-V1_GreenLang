# PRD: Geolocation Collector Agent (GL-EUDR-002)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Geolocation data collection, coordinate validation, plot management
**Priority:** P0 (highest)
**Doc version:** 1.1
**Last updated:** 2026-01-30 (Asia/Kolkata)
**Interview Status:** COMPLETED

---

## 1. Executive Summary

**Geolocation Collector Agent (GL-EUDR-002)** is responsible for collecting, validating, and managing the geolocation data of all production plots in the supply chain. This agent ensures that all origin locations meet EUDR's strict geolocation requirements (6 decimal precision, WGS-84 coordinate system).

This is the most critical data quality agent because:
- EUDR mandates precise geolocation for every production plot
- Without valid coordinates, deforestation risk assessment is impossible
- Geolocation data enables satellite imagery analysis
- Invalid coordinates will cause DDS rejection by EU portal

---

## 2. Problem Statement

EUDR Article 9 requires operators to collect geolocation data for all plots of land where commodities were produced. Challenges include:
1. Many suppliers lack GPS technology or training
2. Coordinates are often provided in wrong formats
3. Precision requirements (6 decimal places) are frequently not met
4. Plot boundaries for areas >4 hectares require polygon data
5. Data quality varies dramatically across suppliers

---

## 3. Goals and Non-Goals

### 3.1 Goals (must deliver)

1. **Multi-channel data collection**
   - Mobile app for field collection (GPS)
   - Web form for manual entry
   - Bulk upload (CSV, GeoJSON, KML, Shapefile)
   - API for programmatic submission

2. **Deterministic validation** (zero hallucination)
   - WGS-84 coordinate system validation
   - Precision verification (≥6 decimal places)
   - Range validation (-90≤lat≤90, -180≤lon≤180)
   - Country boundary verification
   - Water body exclusion

3. **Plot geometry management**
   - Point geometry for plots <4 hectares
   - Polygon geometry for plots ≥4 hectares
   - Area calculation (hectares)
   - Centroid computation
   - Bounding box generation

4. **Data enrichment**
   - Reverse geocoding (coordinates → address)
   - Administrative region identification
   - Biome/ecosystem classification
   - Protected area detection

### 3.2 Non-Goals

- Satellite imagery analysis (GL-EUDR-020+)
- Deforestation detection (GL-EUDR-020+)
- Supply chain mapping (GL-EUDR-001)
- Risk scoring (GL-EUDR-020+)

---

## 4. EUDR Geolocation Requirements

### 4.1 Precision Requirements (Article 9)

| Criteria | Requirement |
|---|---|
| Coordinate System | WGS-84 (EPSG:4326) |
| Decimal Precision | Minimum 6 decimal places |
| Plot <4 hectares | Point (latitude, longitude) |
| Plot ≥4 hectares | Polygon (perimeter coordinates) |
| Format | Decimal degrees (not DMS) |

### 4.2 Precision to Distance Mapping

| Decimal Places | Precision | Acceptable? |
|---|---|---|
| 6 | ~0.11 meters | ✅ Yes |
| 5 | ~1.1 meters | ❌ No |
| 4 | ~11 meters | ❌ No |
| 3 | ~111 meters | ❌ No |

---

## 5. High-Level Requirements

### 5.1 Inputs
- **GPS coordinates** (from mobile devices, GPS units)
- **Uploaded files** (CSV, GeoJSON, KML, Shapefile)
- **Manual entry** (web forms)
- **Supplier declarations** (questionnaire responses)
- **Existing GIS data** (from certifications, surveys)

### 5.2 Outputs
- **Validated plot geometries** (Point or Polygon)
- **Validation reports** (errors, warnings)
- **Enriched plot data** (address, region, biome)
- **Plot registry entries** (ready for risk assessment)

### 5.3 Dependencies
- GL-EUDR-001 (Supply Chain Mapper) - plot-to-supplier linking
- PostGIS database for spatial operations
- Reverse geocoding service (Nominatim/Mapbox)
- Country boundary datasets

---

## 6. Data Model

### 6.1 Plot Geometry Schema

```sql
-- Production Plots
CREATE TABLE production_plots (
    plot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255),  -- Supplier's reference
    supplier_id UUID NOT NULL,

    -- Geometry (EUDR compliant)
    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,
    geometry_type VARCHAR(20) NOT NULL,  -- POINT or POLYGON
    centroid GEOMETRY(POINT, 4326),
    bounding_box GEOMETRY(POLYGON, 4326),
    area_hectares DECIMAL(10,4),
    perimeter_km DECIMAL(10,4),

    -- Location details
    country_code CHAR(2) NOT NULL,
    admin_level_1 VARCHAR(255),  -- State/Province
    admin_level_2 VARCHAR(255),  -- District/County
    admin_level_3 VARCHAR(255),  -- Municipality
    nearest_place VARCHAR(255),

    -- Commodity
    commodity VARCHAR(50) NOT NULL,
    crop_type VARCHAR(100),  -- Specific variety

    -- Validation
    validation_status VARCHAR(50) DEFAULT 'PENDING',
    validation_errors JSONB DEFAULT '[]',
    validation_warnings JSONB DEFAULT '[]',
    precision_lat INTEGER,  -- Decimal places
    precision_lon INTEGER,

    -- Data quality
    collection_method VARCHAR(50),  -- GPS, MANUAL, UPLOAD
    collection_device VARCHAR(100),
    collection_accuracy_m DECIMAL(10,2),  -- GPS accuracy
    collection_date TIMESTAMP,
    collected_by VARCHAR(255),

    -- Enrichment
    biome VARCHAR(100),
    ecosystem VARCHAR(100),
    elevation_m INTEGER,
    slope_degrees DECIMAL(5,2),
    in_protected_area BOOLEAN DEFAULT FALSE,
    protected_area_name VARCHAR(255),

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),

    CONSTRAINT valid_geometry CHECK (ST_IsValid(geometry)),
    CONSTRAINT valid_commodity CHECK (
        commodity IN ('CATTLE', 'COCOA', 'COFFEE', 'PALM_OIL', 'RUBBER', 'SOY', 'WOOD')
    ),
    CONSTRAINT valid_precision CHECK (
        precision_lat >= 6 AND precision_lon >= 6
    )
);

-- Spatial indexes
CREATE INDEX idx_plots_geometry ON production_plots USING GIST (geometry);
CREATE INDEX idx_plots_country ON production_plots (country_code);
CREATE INDEX idx_plots_supplier ON production_plots (supplier_id);
CREATE INDEX idx_plots_commodity ON production_plots (commodity);
CREATE INDEX idx_plots_status ON production_plots (validation_status);

-- Validation history
CREATE TABLE plot_validation_history (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID REFERENCES production_plots(plot_id),
    validation_date TIMESTAMP DEFAULT NOW(),
    status VARCHAR(50) NOT NULL,
    errors JSONB,
    warnings JSONB,
    validated_by VARCHAR(100),
    validation_method VARCHAR(50)  -- AUTO, MANUAL, REVIEW
);
```

---

## 7. Functional Requirements

### 7.1 Data Collection
- **FR-001 (P0):** Accept coordinates via REST API
- **FR-002 (P0):** Accept bulk upload (CSV, GeoJSON, KML, Shapefile)
- **FR-003 (P0):** Provide mobile-optimized web form
- **FR-004 (P1):** Support offline mobile collection with sync
- **FR-005 (P1):** Accept coordinates from integration partners

### 7.2 Coordinate Validation (Deterministic)
- **FR-010 (P0):** Validate coordinate format (decimal degrees)
- **FR-011 (P0):** Verify precision (≥6 decimal places)
- **FR-012 (P0):** Check valid ranges (-90≤lat≤90, -180≤lon≤180)
- **FR-013 (P0):** Verify point within claimed country
- **FR-014 (P0):** Detect coordinates in water bodies
- **FR-015 (P1):** Detect coordinates in urban areas (suspicious)
- **FR-016 (P1):** Validate against known agricultural zones

### 7.3 Polygon Validation
- **FR-020 (P0):** Validate polygon closure (first=last point)
- **FR-021 (P0):** Check polygon validity (no self-intersection)
- **FR-022 (P0):** Calculate area and verify >0.01 hectares
- **FR-023 (P0):** Enforce polygon for plots ≥4 hectares
- **FR-024 (P1):** Detect overlapping plots (same supplier)
- **FR-025 (P1):** Detect overlapping plots (different suppliers)

### 7.4 Data Enrichment
- **FR-030 (P0):** Compute centroid for polygons
- **FR-031 (P0):** Generate bounding box
- **FR-032 (P1):** Reverse geocode to address
- **FR-033 (P1):** Identify administrative regions
- **FR-034 (P1):** Classify biome/ecosystem
- **FR-035 (P1):** Check protected area status

### 7.5 Output and Reporting
- **FR-040 (P0):** Return validation status with error codes
- **FR-041 (P0):** Generate validation report per upload
- **FR-042 (P0):** Export validated plots in multiple formats
- **FR-043 (P1):** Provide data quality dashboard
- **FR-044 (P1):** Alert on systematic validation failures

---

## 8. Validation Engine

### 8.1 Validation Pipeline

```python
class GeolocationValidator:
    """
    Deterministic coordinate validation engine.
    Zero hallucination - all rules are explicit.
    """

    def __init__(self):
        self.country_boundaries = self._load_country_boundaries()
        self.water_bodies = self._load_water_bodies()
        self.protected_areas = self._load_protected_areas()
        self.urban_areas = self._load_urban_areas()

    def validate(
        self,
        coordinates: Union[Point, Polygon],
        country_code: str,
        commodity: str,
        expected_area: Optional[float] = None
    ) -> ValidationResult:
        """
        Main validation entry point.
        Returns structured validation result.
        """
        errors = []
        warnings = []
        metadata = {}

        # Stage 1: Format validation
        format_result = self._validate_format(coordinates)
        errors.extend(format_result.errors)
        metadata.update(format_result.metadata)

        if format_result.is_fatal:
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )

        # Stage 2: Precision validation
        precision_result = self._validate_precision(coordinates)
        errors.extend(precision_result.errors)
        metadata.update(precision_result.metadata)

        # Stage 3: Geographic validation
        geo_result = self._validate_geography(coordinates, country_code)
        errors.extend(geo_result.errors)
        warnings.extend(geo_result.warnings)

        # Stage 4: Geometry-specific validation
        if isinstance(coordinates, Polygon):
            poly_result = self._validate_polygon(
                coordinates, expected_area
            )
            errors.extend(poly_result.errors)
            warnings.extend(poly_result.warnings)
            metadata.update(poly_result.metadata)
        else:
            # Point validation
            point_result = self._validate_point(coordinates)
            errors.extend(point_result.errors)

        # Stage 5: Commodity-specific validation
        commodity_result = self._validate_for_commodity(
            coordinates, commodity, country_code
        )
        warnings.extend(commodity_result.warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

    def _validate_precision(self, coordinates: Geometry) -> StageResult:
        """
        Validate coordinate precision (≥6 decimal places).
        """
        errors = []
        metadata = {}

        if isinstance(coordinates, Point):
            points = [(coordinates.y, coordinates.x)]
        else:
            points = list(coordinates.exterior.coords)

        for lat, lon in points:
            lat_precision = self._count_decimal_places(lat)
            lon_precision = self._count_decimal_places(lon)

            metadata['min_lat_precision'] = min(
                metadata.get('min_lat_precision', 99),
                lat_precision
            )
            metadata['min_lon_precision'] = min(
                metadata.get('min_lon_precision', 99),
                lon_precision
            )

            if lat_precision < 6:
                errors.append(ValidationError(
                    code="INSUFFICIENT_LAT_PRECISION",
                    message=f"Latitude has {lat_precision} decimal places, requires 6",
                    severity="ERROR",
                    coordinate=(lat, lon)
                ))

            if lon_precision < 6:
                errors.append(ValidationError(
                    code="INSUFFICIENT_LON_PRECISION",
                    message=f"Longitude has {lon_precision} decimal places, requires 6",
                    severity="ERROR",
                    coordinate=(lat, lon)
                ))

        return StageResult(errors=errors, metadata=metadata)

    def _validate_geography(
        self,
        coordinates: Geometry,
        country_code: str
    ) -> StageResult:
        """
        Validate geographic placement.
        """
        errors = []
        warnings = []

        centroid = coordinates.centroid if isinstance(coordinates, Polygon) else coordinates

        # Check country boundary
        if not self._point_in_country(centroid, country_code):
            errors.append(ValidationError(
                code="NOT_IN_COUNTRY",
                message=f"Coordinates not within {country_code} boundaries",
                severity="ERROR"
            ))

        # Check water bodies
        if self._point_in_water(centroid):
            errors.append(ValidationError(
                code="IN_WATER_BODY",
                message="Coordinates located in water body",
                severity="ERROR"
            ))

        # Check protected areas
        protected_area = self._check_protected_area(coordinates)
        if protected_area:
            warnings.append(ValidationWarning(
                code="IN_PROTECTED_AREA",
                message=f"Plot overlaps with protected area: {protected_area.name}",
                severity="WARNING"
            ))

        # Check urban areas (suspicious for agriculture)
        if self._point_in_urban(centroid):
            warnings.append(ValidationWarning(
                code="IN_URBAN_AREA",
                message="Coordinates located in urban area",
                severity="WARNING"
            ))

        return StageResult(errors=errors, warnings=warnings)

    def _validate_polygon(
        self,
        polygon: Polygon,
        expected_area: Optional[float]
    ) -> StageResult:
        """
        Validate polygon geometry.
        """
        errors = []
        warnings = []
        metadata = {}

        # Check validity
        if not polygon.is_valid:
            errors.append(ValidationError(
                code="INVALID_POLYGON",
                message=explain_validity(polygon),
                severity="ERROR"
            ))
            return StageResult(errors=errors, is_fatal=True)

        # Calculate area
        area_hectares = self._calculate_area_hectares(polygon)
        metadata['area_hectares'] = area_hectares

        if area_hectares < 0.01:
            errors.append(ValidationError(
                code="AREA_TOO_SMALL",
                message=f"Plot area {area_hectares:.4f} ha is below minimum 0.01 ha",
                severity="ERROR"
            ))

        # Check expected area if provided
        if expected_area and abs(area_hectares - expected_area) / expected_area > 0.2:
            warnings.append(ValidationWarning(
                code="AREA_MISMATCH",
                message=f"Calculated area {area_hectares:.2f} differs from declared {expected_area:.2f} by >20%",
                severity="WARNING"
            ))

        # Check for self-intersection
        if not polygon.is_simple:
            errors.append(ValidationError(
                code="SELF_INTERSECTING",
                message="Polygon has self-intersecting edges",
                severity="ERROR"
            ))

        # Check minimum points
        if len(polygon.exterior.coords) < 4:  # Including closing point
            errors.append(ValidationError(
                code="INSUFFICIENT_POINTS",
                message="Polygon requires at least 3 distinct points",
                severity="ERROR"
            ))

        metadata['perimeter_km'] = polygon.length * 111  # Approximate
        metadata['centroid'] = (polygon.centroid.y, polygon.centroid.x)
        metadata['bounds'] = polygon.bounds

        return StageResult(errors=errors, warnings=warnings, metadata=metadata)

    @staticmethod
    def _count_decimal_places(value: float) -> int:
        """
        Count significant decimal places in coordinate.
        """
        str_value = f"{value:.15f}"  # High precision string
        integer_part, decimal_part = str_value.split('.')

        # Find last non-zero digit
        last_nonzero = len(decimal_part.rstrip('0'))

        return last_nonzero
```

---

## 9. API Specification

```yaml
openapi: 3.0.0
info:
  title: GL-EUDR-002 Geolocation Collector API
  version: 1.0.0

paths:
  /api/v1/plots:
    post:
      summary: Submit plot geolocation
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PlotSubmission'
      responses:
        201:
          description: Plot validated and stored
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PlotValidationResult'

  /api/v1/plots/bulk:
    post:
      summary: Bulk upload plots
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                format:
                  type: string
                  enum: [csv, geojson, kml, shapefile]
                supplier_id:
                  type: string
                  format: uuid
      responses:
        202:
          description: Upload accepted for processing
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BulkUploadResponse'

  /api/v1/plots/{plot_id}/validate:
    post:
      summary: Re-validate existing plot
      parameters:
        - name: plot_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        200:
          description: Validation result

  /api/v1/plots/validate:
    post:
      summary: Validate coordinates without storing
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ValidationRequest'
      responses:
        200:
          description: Validation result

components:
  schemas:
    PlotSubmission:
      type: object
      required:
        - supplier_id
        - coordinates
        - country_code
        - commodity
      properties:
        supplier_id:
          type: string
          format: uuid
        external_id:
          type: string
        coordinates:
          oneOf:
            - $ref: '#/components/schemas/PointCoordinates'
            - $ref: '#/components/schemas/PolygonCoordinates'
        country_code:
          type: string
          pattern: '^[A-Z]{2}$'
        commodity:
          type: string
          enum: [CATTLE, COCOA, COFFEE, PALM_OIL, RUBBER, SOY, WOOD]
        declared_area_hectares:
          type: number
        collection_method:
          type: string
          enum: [GPS, MANUAL, UPLOAD]
        collection_date:
          type: string
          format: date

    PointCoordinates:
      type: object
      required:
        - type
        - latitude
        - longitude
      properties:
        type:
          type: string
          const: point
        latitude:
          type: number
          minimum: -90
          maximum: 90
        longitude:
          type: number
          minimum: -180
          maximum: 180

    PolygonCoordinates:
      type: object
      required:
        - type
        - coordinates
      properties:
        type:
          type: string
          const: polygon
        coordinates:
          type: array
          items:
            type: array
            items:
              type: number
            minItems: 2
            maxItems: 2
          minItems: 4

    PlotValidationResult:
      type: object
      properties:
        plot_id:
          type: string
          format: uuid
        valid:
          type: boolean
        validation_status:
          type: string
          enum: [VALID, INVALID, NEEDS_REVIEW]
        errors:
          type: array
          items:
            $ref: '#/components/schemas/ValidationError'
        warnings:
          type: array
          items:
            $ref: '#/components/schemas/ValidationWarning'
        computed:
          type: object
          properties:
            area_hectares:
              type: number
            centroid:
              type: object
            bounding_box:
              type: object
            admin_region:
              type: string
            biome:
              type: string

    ValidationError:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        severity:
          type: string
        coordinate:
          type: array
          items:
            type: number
```

---

## 10. Error Codes

| Code | Description | Severity |
|---|---|---|
| INSUFFICIENT_LAT_PRECISION | Latitude has <6 decimal places | ERROR |
| INSUFFICIENT_LON_PRECISION | Longitude has <6 decimal places | ERROR |
| INVALID_LAT_RANGE | Latitude outside -90 to 90 | ERROR |
| INVALID_LON_RANGE | Longitude outside -180 to 180 | ERROR |
| NOT_IN_COUNTRY | Coordinates outside claimed country | ERROR |
| IN_WATER_BODY | Coordinates in ocean/lake/river | ERROR |
| INVALID_POLYGON | Polygon geometry invalid | ERROR |
| SELF_INTERSECTING | Polygon has self-intersection | ERROR |
| AREA_TOO_SMALL | Plot area < 0.01 hectares | ERROR |
| INSUFFICIENT_POINTS | Polygon has < 3 points | ERROR |
| IN_PROTECTED_AREA | Plot in protected area | WARNING |
| IN_URBAN_AREA | Plot in urban area | WARNING |
| AREA_MISMATCH | Calculated vs declared area differs | WARNING |
| NEEDS_POLYGON | Plot ≥4 ha requires polygon | WARNING |

---

## 11. Success Metrics

- **Validation Accuracy:** 100% detection of invalid coordinates
- **Processing Speed:** <1 second per point, <5 seconds per polygon
- **Bulk Upload:** 10,000 plots in <5 minutes
- **Zero False Negatives:** Never accept invalid coordinates
- **Data Enrichment:** 95% of plots with admin region identified

---

## 12. Testing Strategy

### 12.1 Unit Tests
- Precision counting for various coordinate formats
- Range validation for edge cases
- Polygon validity checks
- Area calculations

### 12.2 Integration Tests
- End-to-end plot submission flow
- Bulk upload processing
- PostGIS spatial queries

### 12.3 Golden Tests
- Known valid coordinates
- Known invalid coordinates
- Edge cases (dateline, poles, small areas)

### 12.4 Validation Test Cases

```python
# Test precision validation
assert not validate({"lat": -4.12345, "lon": 102.1234}).valid  # Only 4/5 decimals
assert validate({"lat": -4.123456, "lon": 102.123456}).valid  # 6 decimals ✓

# Test range validation
assert not validate({"lat": 91.0, "lon": 0.0}).valid  # Lat > 90
assert not validate({"lat": 0.0, "lon": 181.0}).valid  # Lon > 180

# Test country boundary
assert validate({"lat": -1.234567, "lon": 36.817223}, country="KE").valid  # Kenya
assert not validate({"lat": -1.234567, "lon": 36.817223}, country="BR").valid  # Not Brazil

# Test water body
assert not validate({"lat": 0.0, "lon": -30.0}).valid  # Atlantic Ocean

# Test polygon
polygon = [(0,0), (0,1), (1,1), (1,0), (0,0)]  # Valid square
assert validate(polygon).valid

self_intersecting = [(0,0), (1,1), (0,1), (1,0), (0,0)]  # Bowtie
assert not validate(self_intersecting).valid
```

---

---

## 13. Implementation Decisions (Interview Results)

Based on stakeholder interviews conducted 2026-01-30, the following implementation decisions have been finalized:

### 13.1 Architecture & Deployment

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary Stack** | PostGIS + Python (Shapely) | PRD-aligned; PostGIS for spatial ops, Python for validation pipeline |
| **Mobile Collection** | Progressive Web App (PWA) | Offline-first via service workers, no app store deployment |
| **Integration Pattern** | API-first (REST/gRPC) | Clean boundaries with GL-EUDR-001, synchronous, easier debugging |
| **Multi-tenancy** | Organization-scoped (row-level security) | Simpler ops, shared infrastructure, org_id filtering |
| **Scale Target** | Medium (10K plots/day, 100 concurrent users) | Async job queue for bulk, connection pooling required |

### 13.2 Data Sources & Loading Strategy

| Source | Purpose | Loading Strategy |
|--------|---------|------------------|
| **GADM** | Country + admin boundaries (Admin 0-3) | **Pre-load** at deployment |
| **OSM (OpenStreetMap)** | Water bodies, urban areas | **Pre-load** at deployment |
| **Protected Planet (WDPA)** | Protected area detection | **Lazy load** on demand |
| **Nominatim/Mapbox** | Reverse geocoding | **API call** on demand |
| **Biome/Ecosystem datasets** | Enrichment classification | **Lazy load** on demand |

### 13.3 Validation & Error Handling

**Tiered Validation Strategy:**

```
┌─────────────────────────────────────────────────────────────────┐
│ CRITICAL ERRORS (Auto-Reject → INVALID status)                  │
├─────────────────────────────────────────────────────────────────┤
│ • NOT_IN_COUNTRY - Coordinates outside claimed country          │
│ • IN_WATER_BODY - Coordinates in ocean/lake/river               │
│ • INVALID_LAT_RANGE / INVALID_LON_RANGE - Outside valid range   │
│ • INVALID_POLYGON - Geometry invalid / self-intersecting        │
│ • INSUFFICIENT_PRECISION - <6 decimal places                    │
│ • AREA_TOO_SMALL - Plot area < 0.01 hectares                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ WARNINGS (Accept → NEEDS_REVIEW queue)                          │
├─────────────────────────────────────────────────────────────────┤
│ • IN_PROTECTED_AREA - Plot overlaps protected area              │
│ • IN_URBAN_AREA - Coordinates in urban area (suspicious)        │
│ • AREA_MISMATCH - Calculated vs declared area differs >20%      │
│ • NEEDS_POLYGON - Plot ≥4 ha should have polygon                │
│ • POOR_GPS_ACCURACY - Device accuracy >10m                      │
└─────────────────────────────────────────────────────────────────┘
```

**GPS Accuracy Handling:**
- Accept all submissions but record device-reported accuracy as metadata
- Add WARNING if accuracy > 10m threshold
- Downstream risk assessment can factor in accuracy scores

**Format Policy:**
- Accept **decimal degrees only** (strict)
- Reject DMS, UTM, MGRS formats with clear error message
- No auto-conversion to avoid precision loss and audit complexity

### 13.4 LLM Integration

LLM will be used for **non-validation tasks only** (zero hallucination for core validation):

| LLM Task | Purpose | When Invoked |
|----------|---------|--------------|
| **Address Parsing** | Convert messy supplier address text → structured fields | On manual entry/upload |
| **Error Explanations** | Generate human-readable explanations for warnings | For supplier feedback |
| **NOT for validation** | Core validation remains 100% deterministic | Never |

### 13.5 User Interface Components

**Compliance Officer Review Queue:**
- Map-centric interface with interactive plot visualization
- Batch review with smart grouping (by error type, region, supplier)
- AI-suggested priority scoring based on volume/commodity/region risk
- Inline approve/reject with audit trail

**Data Quality Dashboard (Full Analytics):**
- KPI cards: valid/invalid/pending counts, validation rate
- Geographic heatmap showing validation issues by region
- Trend analysis: validation rates over time
- Supplier-level drill-down
- Error type breakdown
- Export capabilities (CSV, PDF reports)

**Supplier Self-Service Portal:**
- Supplier registration and authentication
- Plot submission (single + bulk upload)
- Real-time validation feedback
- Historical submission tracking
- Error remediation guidance
- Status notifications

### 13.6 Edge Cases to Handle

| Edge Case | Handling |
|-----------|----------|
| **Dateline crossing (180° longitude)** | Special polygon handling for plots spanning International Date Line |
| **Micro-plots (<0.01 ha)** | Accept with WARNING, flag for compliance review |
| **Shared/cooperative plots** | Support multiple producers per plot with share percentages (PlotProducer model) |
| **Complex polygons (>1000 vertices)** | Simplify with Douglas-Peucker, preserve required precision |

### 13.7 Audit Trail & Versioning

**Full audit trail required for EUDR compliance:**
- Log all plot CRUD operations
- Track who/when/what for every change
- Checksum integrity verification
- 90-day retention minimum (configurable)

**Plot version history:**
- Track historical versions of plot geometry
- Support boundary corrections over time
- Enable "as-of" queries for compliance audits
- Maintain complete change history

### 13.8 Bulk Upload Processing

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ASYNC BULK UPLOAD FLOW                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. INGESTION (immediate)                                            │
│     POST /api/v1/plots/bulk                                          │
│     ├── Accept: multipart/form-data (file, format, supplier_id)      │
│     ├── Create BulkUpload job record (job_id)                        │
│     └── Return 202 Accepted with BulkUploadResponse                  │
│                                                                      │
│  2. BACKGROUND PROCESSING                                            │
│     Worker picks up job from queue                                   │
│     ├── Parse file based on format (CSV/GeoJSON/KML/Shapefile)       │
│     ├── For each row/feature:                                        │
│     │   ├── Normalize coordinates                                    │
│     │   ├── Run deterministic validation                             │
│     │   ├── Store with validation_status + errors/warnings           │
│     │   └── Update job progress                                      │
│     └── Generate validation report                                   │
│                                                                      │
│  3. POLLING                                                          │
│     GET /api/v1/bulk/{job_id}                                        │
│     ├── Status: queued | running | complete | failed                 │
│     ├── Progress: processed_count / total_count                      │
│     ├── Summary: valid_count, invalid_count, warning_count           │
│     └── Report URL when complete                                     │
│                                                                      │
│  Performance Target: 10,000 plots in <5 minutes                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.9 Testing Strategy

**Golden Test Datasets (Mix of Three Sources):**

| Source | Purpose | Examples |
|--------|---------|----------|
| **Synthetic** | Edge cases, boundary conditions | Dateline crossing, precision limits, invalid polygons |
| **Real (anonymized)** | Real-world supplier data issues | GPS drift, format variations, typical errors |
| **Public (GFW, agricultural)** | Known valid plots for benchmarks | Global Forest Watch, agricultural registries |

### 13.10 Technical Risk Mitigations

| Risk | Mitigation |
|------|------------|
| **GPS Drift (±5m typical)** | Record accuracy metadata, don't auto-reject, let risk assessment weight it |
| **PostGIS Query Performance** | Spatial indexes (GiST), query optimization, consider caching hot data |
| **External Service Reliability** | Fallback to cached data, graceful degradation, retry with backoff |
| **Polygon Complexity** | Limit vertex count (max 10,000), auto-simplify with Douglas-Peucker |

### 13.11 Implementation Priority

**Phase 1: Core Validation Engine**
1. Coordinate precision validation
2. Range validation
3. Country boundary checking (pre-loaded GADM)
4. Water body exclusion (pre-loaded OSM)
5. Polygon geometry validation
6. Basic API endpoints

**Phase 2: Enhanced Validation + Bulk**
1. Protected area detection (WDPA integration)
2. Urban area detection
3. Bulk upload async processing
4. Validation report generation

**Phase 3: Enrichment + UX**
1. Reverse geocoding integration
2. Admin region identification
3. Biome/ecosystem classification
4. LLM address parsing
5. Compliance review queue (map-based)

**Phase 4: Analytics + Self-Service**
1. Data quality dashboard
2. Supplier self-service portal
3. Trend analysis
4. Priority scoring

---

*Document Version: 1.1*
*Created: 2026-01-30*
*Interview Completed: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
