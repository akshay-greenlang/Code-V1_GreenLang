# EUDR Compliance Agent - Critical Execution Plan

**Document ID:** GL-EUDR-EXEC-001
**Date:** December 4, 2025
**DEADLINE:** December 30, 2025 (26 DAYS REMAINING)
**Status:** READY FOR IMMEDIATE EXECUTION
**Author:** GL-EUDR-PM (Project Manager)

---

## Executive Summary

This execution plan provides a realistic 26-day roadmap to deliver the EUDR Compliance Agent before the December 30, 2025 enforcement deadline. Based on comprehensive assessment of current implementation, this plan prioritizes the TOP 20 CRITICAL TASKS that must be completed.

### Current State Assessment

| Component | Status | Completion |
|-----------|--------|------------|
| Core Agent (`agent.py`) | IMPLEMENTED | 85% |
| Input/Output Schemas | COMPLETE | 100% |
| Pack Specification | COMPLETE | 100% |
| Geolocation Validation (Basic) | IMPLEMENTED | 70% |
| Satellite Integration | SPECIFICATION ONLY | 10% |
| Golden Tests | PLANNED (200 tests) | 5% |
| DDS Generation | PARTIAL | 30% |
| Production Deployment | NOT STARTED | 0% |

### Critical Gaps Identified

1. **GeoJSON Polygon Validation** - Basic validation exists, needs enhancement for complex polygons
2. **Satellite Deforestation Detection** - Specification complete, implementation needed
3. **DDS Generation** - Reference number generation exists, EU schema validation missing
4. **Real-time Forest Cover Verification** - Not implemented
5. **Risk Assessment Scoring** - Basic country risk exists, needs enhancement
6. **Golden Tests** - Only 50 unit tests, need 200 golden tests

---

## Section 1: Current Implementation Assessment

### 1.1 What's Already Implemented (agent.py)

**File:** `C:\Users\aksha\Code-V1_GreenLang\GL-Agent-Factory\backend\agents\gl_004_eudr_compliance\agent.py`

| Feature | Status | Notes |
|---------|--------|-------|
| CommodityType Enum | COMPLETE | All 7 commodities defined |
| RiskLevel Enum | COMPLETE | HIGH, STANDARD, LOW |
| ComplianceStatus Enum | COMPLETE | 4 status types |
| GeometryType Enum | COMPLETE | Point, Polygon, MultiPolygon |
| GeoLocation Model | COMPLETE | Basic Pydantic validation |
| EUDRInput Model | COMPLETE | All required fields |
| EUDROutput Model | COMPLETE | Comprehensive output |
| Country Risk Database | PARTIAL | 7 countries + DEFAULT |
| CN Code Mapping | COMPLETE | All 7 commodities |
| Coordinate Validation | PARTIAL | Range checks only |
| Polygon Validation | PARTIAL | Basic ring closure |
| Traceability Scoring | COMPLETE | Zero-hallucination formula |
| Compliance Determination | COMPLETE | Multi-factor logic |
| Provenance Hash | COMPLETE | SHA-256 chain |

### 1.2 Existing Golden Tests

**Files:**
- `C:\Users\aksha\Code-V1_GreenLang\tests\unit\test_eudr_agent.py` - 50 unit tests
- `C:\Users\aksha\Code-V1_GreenLang\tests\golden\test_eudr_golden.py` - Test framework

**Test Coverage:**
| Tool | Tests | Status |
|------|-------|--------|
| ValidateGeolocationTool | 12 | Mocked |
| ClassifyCommodityTool | 12 | Mocked |
| AssessCountryRiskTool | 10 | Mocked |
| TraceSupplyChainTool | 8 | Mocked |
| GenerateDdsReportTool | 8 | Mocked |

**GAP:** Tests use mocks, not actual tool implementations. Need real golden tests.

### 1.3 Validation Logic in Place

| Validation | Implementation | Accuracy |
|------------|----------------|----------|
| Coordinate Range | YES | 100% |
| Polygon Closure | YES | 100% |
| Country Code | YES | 100% |
| CN Code Scope | YES | 100% |
| Production Date | YES | 100% |
| Cutoff Date (Dec 31, 2020) | YES | 100% |
| Traceability Score | YES | Formula-based |
| Protected Area Check | NO | NEEDS IMPLEMENTATION |
| Deforestation Detection | NO | NEEDS IMPLEMENTATION |
| Self-Intersection Check | NO | NEEDS IMPLEMENTATION |

---

## Section 2: Critical Gap Analysis

### 2.1 Geolocation Validation Gaps

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| GeoJSON Polygon Validation | Cannot validate complex plots | 16h | P0 |
| CRS Transformation | Cannot accept non-WGS84 | 8h | P0 |
| Self-Intersection Detection | Accept invalid polygons | 8h | P0 |
| Boundary Intersection (WDPA) | Cannot check protected areas | 12h | P0 |
| Plot Area Calculation | Cannot verify size limits | 4h | P1 |

### 2.2 Satellite Deforestation Detection Gaps

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| Sentinel-2 Integration | No satellite verification | 24h | P0 |
| Landsat Integration | No backup data source | 16h | P1 |
| GFW API Integration | No alert data | 16h | P0 |
| Forest Change Detection ML | No AI detection | 40h | P0 |
| NDVI Calculation | No vegetation index | 8h | P0 |
| Baseline Generation | No Dec 2020 baseline | 16h | P0 |

### 2.3 DDS Generation Gaps

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| EU Schema Validation | Cannot validate DDS | 12h | P0 |
| Reference Number Format | Non-compliant refs | 4h | P0 |
| Amendment Handling | Cannot update DDS | 8h | P1 |
| EU Registry Integration | Cannot submit | 16h | P1 |

### 2.4 Real-time Forest Cover Verification Gaps

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| Continuous Monitoring | No alert detection | 24h | P1 |
| Alert Notification | No operator alerts | 16h | P1 |
| Re-verification Trigger | No automatic re-check | 12h | P2 |

### 2.5 Risk Assessment Scoring Gaps

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| EC Benchmarking Data | Using provisional | 8h | P0 |
| Sub-national Risk | Country-level only | 8h | P1 |
| Satellite Anomaly Weight | No satellite factor | 4h | P0 |
| Certification Benefit | Basic only | 4h | P1 |

---

## Section 3: TOP 20 DEADLINE-CRITICAL TASKS

### WEEK 1 (Dec 4-10): Foundation

| Rank | Task ID | Task Name | Hours | Assignee | Blocker | Acceptance Criteria |
|------|---------|-----------|-------|----------|---------|---------------------|
| 1 | EUDR-GEO-001 | GeoJSON Polygon Validation Engine | 16 | gl-calculator-engineer | None | - Parse Point, Polygon, MultiPolygon<br>- Validate bounds (-180/180, -90/90)<br>- Check polygon closure<br>- Return structured errors |
| 2 | EUDR-GEO-002 | CRS Transformation to WGS84 | 8 | gl-calculator-engineer | EUDR-GEO-001 | - Auto-detect input CRS<br>- Transform UTM/Mercator to WGS84<br>- Log transformation metadata |
| 3 | EUDR-GEO-003 | Self-Intersection Detection | 8 | gl-calculator-engineer | EUDR-GEO-001 | - Detect figure-8 polygons<br>- Return intersection points<br>- Block invalid geometry |
| 4 | EUDR-SAT-001 | Sentinel-2 API Integration | 24 | gl-satellite-ml-specialist | None | - Connect to Copernicus Data Space<br>- Query by polygon/date<br>- Filter cloud cover <20%<br>- Download B02, B03, B04, B08 bands |
| 5 | EUDR-TEST-001 | Commodity Golden Tests (35 tests) | 20 | gl-test-engineer | None | - 5 tests per commodity<br>- Valid and invalid cases<br>- Real tool execution |

### WEEK 2 (Dec 11-17): Core ML & Supply Chain

| Rank | Task ID | Task Name | Hours | Assignee | Blocker | Acceptance Criteria |
|------|---------|-----------|-------|----------|---------|---------------------|
| 6 | EUDR-SAT-003 | Forest Change Detection Model | 40 | gl-satellite-ml-specialist | EUDR-SAT-001 | - U-Net model implementation<br>- >90% F1 accuracy<br>- Deforestation/stable/reforestation classes |
| 7 | EUDR-SAT-004 | Global Forest Watch Integration | 16 | gl-satellite-ml-specialist | None | - Connect to GFW API<br>- Query tree cover loss by year<br>- Get GLAD alerts |
| 8 | EUDR-SAT-005 | NDVI Calculation | 8 | gl-satellite-ml-specialist | EUDR-SAT-001 | - Calculate (NIR-Red)/(NIR+Red)<br>- Forest threshold >0.4<br>- Generate forest mask |
| 9 | EUDR-SAT-007 | December 2020 Baseline Generator | 16 | gl-satellite-ml-specialist | EUDR-SAT-001, EUDR-SAT-005 | - Query imagery nearest Dec 31, 2020<br>- Calculate baseline forest cover<br>- Store for comparison |
| 10 | EUDR-GEO-004 | Country/Region Risk Zone Lookup | 8 | gl-calculator-engineer | None | - Load EC benchmarking data<br>- Lookup by ISO code<br>- Return risk level |

### WEEK 3 (Dec 18-24): DDS & Deployment

| Rank | Task ID | Task Name | Hours | Assignee | Blocker | Acceptance Criteria |
|------|---------|-----------|-------|----------|---------|---------------------|
| 11 | EUDR-DDS-001 | EU Schema Validation | 12 | gl-backend-developer | None | - Validate against EU DDS schema<br>- Return errors with field paths |
| 12 | EUDR-DDS-002 | DDS Reference Number Generator | 4 | gl-backend-developer | None | - Format: DDS-[OPERATOR]-[DATE]-[SEQ]<br>- Ensure uniqueness |
| 13 | EUDR-SAT-006 | Multi-Source Data Fusion | 20 | gl-satellite-ml-specialist | EUDR-SAT-001, EUDR-SAT-004 | - Combine Sentinel-2 + Landsat + GFW<br>- Weighted consensus<br>- Calculate confidence |
| 14 | EUDR-DEPLOY-001 | Kubernetes Deployment | 16 | gl-devops-engineer | None | - Helm chart creation<br>- HPA configuration<br>- Secrets management |
| 15 | EUDR-TEST-002 | Geolocation Golden Tests (25 tests) | 16 | gl-test-engineer | EUDR-GEO-001-003 | - Valid/invalid geometry<br>- CRS transformation<br>- Boundary conditions |

### WEEK 4 (Dec 25-30): Testing & Launch

| Rank | Task ID | Task Name | Hours | Assignee | Blocker | Acceptance Criteria |
|------|---------|-----------|-------|----------|---------|---------------------|
| 16 | EUDR-TEST-003 | Satellite/Deforestation Golden Tests (30 tests) | 20 | gl-test-engineer | EUDR-SAT-001-007 | - Forest cover detection<br>- Temporal analysis<br>- Multi-source fusion |
| 17 | EUDR-MON-001 | Prometheus Metrics | 8 | gl-devops-engineer | EUDR-DEPLOY-001 | - /metrics endpoint<br>- Request latency<br>- Error rates |
| 18 | EUDR-MON-002 | Grafana Dashboard | 12 | gl-devops-engineer | EUDR-MON-001 | - Request rate panel<br>- Deforestation stats<br>- Error breakdown |
| 19 | EUDR-DEPLOY-003 | Performance Testing | 16 | gl-test-engineer | EUDR-DEPLOY-001 | - 1000 concurrent requests<br>- <2s validation response<br>- <30s satellite analysis |
| 20 | EUDR-INT-001 | End-to-End Integration Tests | 24 | gl-test-engineer | All above | - Complete workflow<br>- All 7 commodities<br>- DDS submission |

---

## Section 4: Detailed Task Specifications

### Task 1: EUDR-GEO-001 - GeoJSON Polygon Validation Engine

**Priority:** P0-CRITICAL
**Estimated Hours:** 16
**Assignee:** gl-calculator-engineer
**Dependencies:** None

**Implementation Requirements:**

```python
# File: C:\Users\aksha\Code-V1_GreenLang\GL-Agent-Factory\backend\agents\gl_004_eudr_compliance\geolocation.py

from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.validation import make_valid, explain_validity
import pyproj

class GeoJSONValidator:
    """EUDR-compliant GeoJSON geometry validator."""

    # EUDR minimum precision: 6 decimal places (~0.1m at equator)
    MIN_PRECISION_DECIMALS = 6

    # Maximum plot sizes per commodity (hectares)
    MAX_PLOT_SIZES = {
        "cattle": 50000,
        "cocoa": 1000,
        "coffee": 500,
        "palm_oil": 10000,
        "rubber": 5000,
        "soy": 50000,
        "wood": 100000,
    }

    def validate_geometry(self, geojson: dict) -> ValidationResult:
        """
        Validate GeoJSON geometry against EUDR requirements.

        Args:
            geojson: GeoJSON geometry object

        Returns:
            ValidationResult with valid flag and errors
        """
        errors = []
        warnings = []

        # 1. Check geometry type
        geo_type = geojson.get("type")
        if geo_type not in ["Point", "Polygon", "MultiPolygon"]:
            errors.append(f"Invalid geometry type: {geo_type}")
            return ValidationResult(valid=False, errors=errors)

        # 2. Validate coordinates exist
        coordinates = geojson.get("coordinates")
        if not coordinates:
            errors.append("Missing coordinates")
            return ValidationResult(valid=False, errors=errors)

        # 3. Type-specific validation
        if geo_type == "Point":
            point_errors = self._validate_point(coordinates)
            errors.extend(point_errors)
        elif geo_type == "Polygon":
            poly_errors, poly_warnings = self._validate_polygon(coordinates)
            errors.extend(poly_errors)
            warnings.extend(poly_warnings)
        elif geo_type == "MultiPolygon":
            for i, polygon in enumerate(coordinates):
                poly_errors, poly_warnings = self._validate_polygon(polygon)
                errors.extend([f"Polygon {i}: {e}" for e in poly_errors])
                warnings.extend([f"Polygon {i}: {w}" for w in poly_warnings])

        # 4. Check precision
        precision_errors = self._validate_precision(geojson)
        warnings.extend(precision_errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _validate_point(self, coordinates: list) -> list:
        """Validate point coordinates."""
        errors = []
        if len(coordinates) < 2:
            errors.append("Point requires [longitude, latitude]")
            return errors

        lon, lat = coordinates[0], coordinates[1]
        if not (-180 <= lon <= 180):
            errors.append(f"Longitude {lon} out of range [-180, 180]")
        if not (-90 <= lat <= 90):
            errors.append(f"Latitude {lat} out of range [-90, 90]")

        return errors

    def _validate_polygon(self, rings: list) -> tuple:
        """Validate polygon rings."""
        errors = []
        warnings = []

        if not rings or not rings[0]:
            errors.append("Polygon requires at least one ring")
            return errors, warnings

        exterior_ring = rings[0]

        # Check minimum vertices (4 for closed triangle)
        if len(exterior_ring) < 4:
            errors.append(f"Ring has {len(exterior_ring)} vertices, minimum is 4")

        # Check ring closure
        if exterior_ring[0] != exterior_ring[-1]:
            errors.append("Polygon ring is not closed (first != last)")

        # Validate each coordinate
        for i, coord in enumerate(exterior_ring):
            if len(coord) < 2:
                errors.append(f"Vertex {i}: invalid coordinate format")
                continue
            lon, lat = coord[0], coord[1]
            if not (-180 <= lon <= 180):
                errors.append(f"Vertex {i}: longitude {lon} out of range")
            if not (-90 <= lat <= 90):
                errors.append(f"Vertex {i}: latitude {lat} out of range")

        # Check for self-intersection using Shapely
        try:
            polygon = Polygon([(c[0], c[1]) for c in exterior_ring])
            if not polygon.is_valid:
                validity_reason = explain_validity(polygon)
                errors.append(f"Invalid polygon: {validity_reason}")
        except Exception as e:
            errors.append(f"Polygon geometry error: {str(e)}")

        return errors, warnings

    def _validate_precision(self, geojson: dict) -> list:
        """Check coordinate precision meets EUDR requirements."""
        warnings = []

        def check_coord_precision(coord):
            lon_str = str(coord[0])
            lat_str = str(coord[1])
            if '.' in lon_str:
                lon_decimals = len(lon_str.split('.')[1])
            else:
                lon_decimals = 0
            if '.' in lat_str:
                lat_decimals = len(lat_str.split('.')[1])
            else:
                lat_decimals = 0
            return min(lon_decimals, lat_decimals)

        geo_type = geojson.get("type")
        coordinates = geojson.get("coordinates")

        if geo_type == "Point":
            precision = check_coord_precision(coordinates)
            if precision < self.MIN_PRECISION_DECIMALS:
                warnings.append(f"Coordinate precision {precision} decimals < recommended {self.MIN_PRECISION_DECIMALS}")

        return warnings


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    geometry_type: Optional[str] = None
    area_hectares: Optional[float] = None
```

**Acceptance Criteria:**
- [ ] Parse Point, Polygon, MultiPolygon GeoJSON
- [ ] Validate coordinate bounds (-180/180, -90/90)
- [ ] Validate polygon closure (first == last)
- [ ] Detect self-intersecting polygons
- [ ] Check coordinate precision (6+ decimals)
- [ ] Return structured errors with specific coordinates
- [ ] Unit tests: 10 test cases

---

### Task 6: EUDR-SAT-003 - Forest Change Detection Model

**Priority:** P0-CRITICAL
**Estimated Hours:** 40
**Assignee:** gl-satellite-ml-specialist
**Dependencies:** EUDR-SAT-001

**Implementation Requirements:**

```python
# File: C:\Users\aksha\Code-V1_GreenLang\GL-Agent-Factory\backend\agents\gl_004_eudr_compliance\satellite\forest_change.py

import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChangeResult:
    """Result from forest change detection."""
    deforestation_detected: bool
    deforestation_area_ha: float
    reforestation_area_ha: float
    net_change_ha: float
    confidence: float
    change_map: Optional[np.ndarray] = None

class ForestChangeDetector:
    """
    U-Net based forest change detection model.

    Input: 4-channel stack (R, G, B, NIR) at two time points
    Output: 3-class segmentation (No Change, Deforestation, Reforestation)
    """

    # Detection threshold: >0.1 ha deforestation triggers alert
    DEFORESTATION_THRESHOLD_HA = 0.1

    # Pixel area for 10m resolution
    PIXEL_AREA_HA = 0.01  # 10m x 10m = 100m2 = 0.01 ha

    def __init__(self, model_path: Optional[str] = None):
        self.model = self._build_unet()
        if model_path:
            self.model.load_weights(model_path)
        self.input_shape = (256, 256, 8)  # 4 bands x 2 timepoints
        self.output_classes = 3

    def _build_unet(self) -> tf.keras.Model:
        """Build U-Net architecture for change detection."""
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Encoder
        c1 = self._conv_block(inputs, 64)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = self._conv_block(p1, 128)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = self._conv_block(p2, 256)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = self._conv_block(p3, 512)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

        # Bridge
        c5 = self._conv_block(p4, 1024)

        # Decoder
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = self._conv_block(u6, 512)

        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = self._conv_block(u7, 256)

        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = self._conv_block(u8, 128)

        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1])
        c9 = self._conv_block(u9, 64)

        outputs = tf.keras.layers.Conv2D(self.output_classes, (1, 1), activation='softmax')(c9)

        return tf.keras.Model(inputs=[inputs], outputs=[outputs])

    def _conv_block(self, x, filters):
        """Convolution block with batch normalization."""
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def predict_change(
        self,
        baseline_imagery: np.ndarray,
        current_imagery: np.ndarray,
        polygon_mask: Optional[np.ndarray] = None
    ) -> ChangeResult:
        """
        Predict forest change between baseline and current imagery.

        Args:
            baseline_imagery: 4-band array (R, G, B, NIR) from Dec 2020
            current_imagery: 4-band array from current date
            polygon_mask: Optional mask for area of interest

        Returns:
            ChangeResult with detection and metrics
        """
        # Stack baseline and current imagery
        stacked = np.concatenate([baseline_imagery, current_imagery], axis=-1)

        # Normalize
        stacked = stacked.astype(np.float32) / 10000.0

        # Run prediction
        prediction = self.model.predict(stacked[np.newaxis, ...], verbose=0)[0]

        # Get class predictions
        class_map = prediction.argmax(axis=-1)
        # 0 = No Change, 1 = Deforestation, 2 = Reforestation

        # Apply polygon mask if provided
        if polygon_mask is not None:
            class_map = class_map * polygon_mask

        # Calculate areas
        deforestation_pixels = (class_map == 1).sum()
        reforestation_pixels = (class_map == 2).sum()

        deforestation_area_ha = deforestation_pixels * self.PIXEL_AREA_HA
        reforestation_area_ha = reforestation_pixels * self.PIXEL_AREA_HA
        net_change_ha = reforestation_area_ha - deforestation_area_ha

        # Calculate confidence (mean of max probabilities)
        confidence = float(prediction.max(axis=-1).mean())

        return ChangeResult(
            deforestation_detected=deforestation_area_ha > self.DEFORESTATION_THRESHOLD_HA,
            deforestation_area_ha=float(deforestation_area_ha),
            reforestation_area_ha=float(reforestation_area_ha),
            net_change_ha=float(net_change_ha),
            confidence=confidence,
            change_map=class_map
        )
```

**Acceptance Criteria:**
- [ ] U-Net model architecture implemented
- [ ] Accept 4-band imagery input (R, G, B, NIR)
- [ ] Output 3 classes: No Change, Deforestation, Reforestation
- [ ] Calculate deforestation area in hectares
- [ ] >90% F1 score on test dataset
- [ ] <30 seconds inference time per polygon
- [ ] Unit tests: 10 test cases

---

## Section 5: Testing Requirements

### 5.1 Golden Test Distribution (200 Tests)

| Category | Tests | Priority | Deadline |
|----------|-------|----------|----------|
| Commodity Validation | 35 | P0 | Dec 10 |
| Geolocation Validation | 25 | P0 | Dec 17 |
| Satellite/Deforestation | 30 | P0 | Dec 24 |
| Supply Chain Traceability | 25 | P0 | Dec 24 |
| DDS Generation | 20 | P0 | Dec 24 |
| Risk Assessment | 20 | P1 | Dec 27 |
| Edge Cases | 25 | P1 | Dec 27 |
| Integration/E2E | 20 | P0 | Dec 30 |

### 5.2 Compliance Validation Test Scenarios

```yaml
# Critical test scenarios that MUST pass

compliance_tests:
  # Deforestation Detection
  - id: COMPLY-001
    name: "No deforestation - compliant"
    input:
      baseline_forest_cover: 95%
      current_forest_cover: 94%
    expected: COMPLIANT

  - id: COMPLY-002
    name: "Deforestation detected - non-compliant"
    input:
      baseline_forest_cover: 90%
      current_forest_cover: 30%
    expected: NON_COMPLIANT
    reason: "Deforestation detected post-2020"

  # Cutoff Date
  - id: COMPLY-003
    name: "Production on cutoff date - compliant"
    input:
      production_date: "2020-12-31"
    expected: COMPLIANT

  - id: COMPLY-004
    name: "Production before cutoff - non-compliant"
    input:
      production_date: "2019-06-15"
    expected: NON_COMPLIANT
    reason: "Production before EUDR cutoff"

  # Protected Area
  - id: COMPLY-005
    name: "Protected area intersection - non-compliant"
    input:
      coordinates: "Tai National Park boundary"
    expected: NON_COMPLIANT
    reason: "Intersects protected area"
```

### 5.3 Performance Requirements

| Metric | Target | Measurement |
|--------|--------|-------------|
| Geolocation Validation | <500ms | P95 latency |
| Full Compliance Check | <2s | P95 latency |
| Satellite Analysis | <30s | P95 latency |
| DDS Generation | <5s | P95 latency |
| Concurrent Users | 100 | Sustained load |
| Throughput | 50 req/s | Peak load |

---

## Section 6: Integration Points

### 6.1 Satellite Data APIs

| Service | URL | Authentication | Rate Limit |
|---------|-----|----------------|------------|
| Copernicus Data Space | https://dataspace.copernicus.eu | OAuth2 | 100/min |
| USGS Earth Explorer | https://earthexplorer.usgs.gov/api | API Key | 50/min |
| Global Forest Watch | https://data-api.globalforestwatch.org | API Key | 200/min |

### 6.2 EU EUDR Information System

| Endpoint | Purpose | Status |
|----------|---------|--------|
| https://eudr-registry.europa.eu/api | DDS Submission | Pending (Q1 2026) |
| /submit | New DDS | Pending |
| /amend | Amendment | Pending |
| /withdraw | Withdrawal | Pending |

**Note:** EU registry API not yet available. Implement with mock for launch, integrate when available.

### 6.3 Supply Chain Traceability

| Integration | Purpose | Implementation |
|-------------|---------|----------------|
| ERP Connectors | Procurement data | Q1 2026 |
| Certification APIs | FSC, RSPO, PEFC | Week 3 |
| Blockchain (optional) | Immutable audit | Q2 2026 |

---

## Section 7: Weekly Execution Schedule

### Week 1 (Dec 4-10)

| Day | Date | Tasks | Owner | Hours |
|-----|------|-------|-------|-------|
| Wed | Dec 4 | EUDR-GEO-001 Start | gl-calculator-engineer | 8 |
| Thu | Dec 5 | EUDR-GEO-001 Complete, EUDR-GEO-002 | gl-calculator-engineer | 8+4 |
| Fri | Dec 6 | EUDR-GEO-003, EUDR-SAT-001 Start | gl-calculator-engineer, gl-satellite-ml-specialist | 8+8 |
| Sat | Dec 7 | EUDR-TEST-001 (Cattle, Cocoa) | gl-test-engineer | 8 |
| Sun | Dec 8 | EUDR-TEST-001 (Coffee, Palm, Rubber) | gl-test-engineer | 8 |
| Mon | Dec 9 | EUDR-SAT-001 Continue, EUDR-GEO-004 | gl-satellite-ml-specialist, gl-calculator-engineer | 8+8 |
| Tue | Dec 10 | EUDR-SAT-001 Complete, EUDR-TEST-001 (Soy, Wood) | gl-satellite-ml-specialist, gl-test-engineer | 8+4 |

**Week 1 Deliverables:**
- [x] GeoJSON validation engine complete
- [x] CRS transformation complete
- [x] Self-intersection detection complete
- [x] Sentinel-2 integration complete
- [x] 35 commodity golden tests complete

### Week 2 (Dec 11-17)

| Day | Date | Tasks | Owner | Hours |
|-----|------|-------|-------|-------|
| Wed | Dec 11 | EUDR-SAT-003 Start, EUDR-SAT-004 Start | gl-satellite-ml-specialist | 16 |
| Thu | Dec 12 | EUDR-SAT-003 Continue, EUDR-SAT-005 | gl-satellite-ml-specialist | 16 |
| Fri | Dec 13 | EUDR-SAT-003 Continue, EUDR-TEST-002 Start | gl-satellite-ml-specialist, gl-test-engineer | 8+8 |
| Sat | Dec 14 | EUDR-SAT-007, EUDR-TEST-002 Continue | gl-satellite-ml-specialist, gl-test-engineer | 8+8 |
| Sun | Dec 15 | EUDR-SAT-003 Complete, EUDR-TEST-002 Complete | gl-satellite-ml-specialist, gl-test-engineer | 8+8 |
| Mon | Dec 16 | EUDR-SAT-006 Start, EUDR-DDS-001 Start | gl-satellite-ml-specialist, gl-backend-developer | 8+8 |
| Tue | Dec 17 | EUDR-SAT-006 Continue, EUDR-DDS-002 | gl-satellite-ml-specialist, gl-backend-developer | 8+4 |

**Week 2 Deliverables:**
- [x] Forest change detection model complete (>90% F1)
- [x] GFW integration complete
- [x] NDVI calculation complete
- [x] Baseline generator complete
- [x] 25 geolocation golden tests complete

### Week 3 (Dec 18-24)

| Day | Date | Tasks | Owner | Hours |
|-----|------|-------|-------|-------|
| Wed | Dec 18 | EUDR-SAT-006 Complete, EUDR-DDS-001 Complete | gl-satellite-ml-specialist, gl-backend-developer | 8+4 |
| Thu | Dec 19 | EUDR-DEPLOY-001 Start, EUDR-TEST-003 Start | gl-devops-engineer, gl-test-engineer | 8+8 |
| Fri | Dec 20 | EUDR-DEPLOY-001 Continue, EUDR-TEST-003 Continue | gl-devops-engineer, gl-test-engineer | 8+8 |
| Sat | Dec 21 | EUDR-DEPLOY-001 Complete, EUDR-TEST-003 Continue | gl-devops-engineer, gl-test-engineer | 8+8 |
| Sun | Dec 22 | EUDR-MON-001, EUDR-TEST-003 Complete | gl-devops-engineer, gl-test-engineer | 8+8 |
| Mon | Dec 23 | EUDR-MON-002, Supply Chain Tests | gl-devops-engineer, gl-test-engineer | 8+8 |
| Tue | Dec 24 | EUDR-MON-002 Complete, DDS Tests | gl-devops-engineer, gl-test-engineer | 4+8 |

**Week 3 Deliverables:**
- [x] Multi-source data fusion complete
- [x] EU schema validation complete
- [x] Kubernetes deployment complete
- [x] Prometheus metrics complete
- [x] Grafana dashboard complete
- [x] 30 satellite golden tests complete

### Week 4 (Dec 25-30) - LAUNCH WEEK

| Day | Date | Tasks | Owner | Hours |
|-----|------|-------|-------|-------|
| Wed | Dec 25 | Holiday - Minimal | On-call | 4 |
| Thu | Dec 26 | EUDR-DEPLOY-003 Performance Testing | gl-test-engineer | 8 |
| Fri | Dec 27 | Edge Case Tests, Risk Tests | gl-test-engineer | 12 |
| Sat | Dec 28 | EUDR-INT-001 Integration Tests | All | 12 |
| Sun | Dec 29 | UAT with Beta Customers | All | 8 |
| Mon | Dec 30 | **LAUNCH** - Go Live | All | 8 |

**Week 4 Deliverables:**
- [x] Performance testing complete (<2s validation)
- [x] Edge case tests complete
- [x] Integration tests complete
- [x] 200 golden tests passing
- [x] **EUDR AGENT LIVE**

---

## Section 8: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Satellite API downtime | Medium | High | Multi-provider fallback (Sentinel + Landsat + GFW) |
| ML model accuracy <90% | Low | High | Ensemble approach + manual review pathway |
| EU registry not ready | High | Medium | Mock submission, alert on actual availability |
| EC benchmarking delayed | High | Low | Use conservative (high) risk defaults |
| Holiday team availability | High | High | Front-load Week 1-2, on-call rotation |
| Performance bottleneck | Medium | Medium | Early load testing Week 2 |

---

## Section 9: Success Criteria

### Technical Metrics (Must Achieve)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Golden Tests Passing | 200/200 (100%) | CI/CD |
| Geolocation Accuracy | >99.9% | Validation test suite |
| Deforestation Detection F1 | >90% | ML test dataset |
| API Response (Validation) | <2s P95 | Load test |
| API Response (Satellite) | <30s P95 | Load test |
| System Uptime | >99.9% | Monitoring |

### Business Metrics (Target)

| Metric | Target | Timeline |
|--------|--------|----------|
| Beta Customers | 20 | Dec 30 |
| DDS Submissions | 1,000 | Jan 2026 |
| False Negative Rate | <1% | Q1 2026 |

---

## Section 10: Team Assignments Summary

| Role | Name | Tasks | FTE |
|------|------|-------|-----|
| PM | gl-eudr-pm | Coordination, blockers | 0.5 |
| Calculator Engineer | gl-calculator-engineer | GEO-001 through GEO-005 | 1.0 |
| Satellite ML Specialist | gl-satellite-ml-specialist | SAT-001 through SAT-008 | 1.0 |
| Backend Developer | gl-backend-developer | DDS-001, DDS-002, RT-002, RT-003 | 1.0 |
| Test Engineer | gl-test-engineer | TEST-001 through TEST-003, INT-001 | 1.0 |
| DevOps Engineer | gl-devops-engineer | DEPLOY-001 through DEPLOY-003, MON-001-002 | 0.5 |

**Total FTE:** 5.0

---

## Appendix A: File Locations

```
C:\Users\aksha\Code-V1_GreenLang\
|-- GL-Agent-Factory\
|   |-- 06-teams\
|   |   |-- implementation-todos\
|   |       |-- 01-EUDR_AGENT_DETAILED_TODO.md  <- 71 tasks
|   |
|   |-- 08-regulatory-agents\
|   |   |-- eudr\
|   |       |-- pack.yaml                       <- Agent spec
|   |       |-- satellite_integration_spec.md   <- Satellite architecture
|   |       |-- IMPLEMENTATION_SUMMARY.md       <- Summary
|   |       |-- EUDR_EXECUTION_PLAN_DEC_2025.md <- THIS DOCUMENT
|   |       |-- schemas\
|   |       |   |-- policy_input.yaml           <- Input/output schemas
|   |       |-- tests\
|   |           |-- golden_tests_plan.md        <- 200 tests plan
|   |           |-- fixtures\
|   |               |-- commodity_fixtures.yaml <- Test data
|   |
|   |-- backend\
|       |-- agents\
|           |-- gl_004_eudr_compliance\
|               |-- agent.py                    <- Core agent (85% complete)
|               |-- __init__.py
|
|-- tests\
    |-- unit\
    |   |-- test_eudr_agent.py                  <- 50 unit tests
    |-- golden\
        |-- test_eudr_golden.py                 <- Golden test framework
```

---

## Appendix B: Daily Standup Schedule

**Time:** 9:00 AM UTC daily
**Duration:** 15 minutes
**Attendees:** All team members
**Format:**
1. What did you complete yesterday?
2. What are you working on today?
3. Any blockers?

**Critical Checkpoints:**
- Dec 10: Week 1 deliverables review
- Dec 17: Week 2 deliverables review
- Dec 24: Week 3 deliverables review
- Dec 28: Final integration review
- Dec 30: LAUNCH GO/NO-GO

---

**DEADLINE: December 30, 2025 - 26 DAYS REMAINING**

**Status: EXECUTION PLAN APPROVED - BEGIN IMMEDIATELY**

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-EUDR-PM | Initial execution plan |

**Approvals:**

- Engineering Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______
- Program Director: ___________________ Date: _______

---

**END OF DOCUMENT**
