# PRD: AGENT-DATA-006 - GIS/Mapping Connector

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-006 |
| **Agent ID** | GL-DATA-GEO-001 |
| **Component** | GIS/Mapping Connector Agent (Geospatial Data Ingestion, Coordinate Transformation, Spatial Analysis, Format Conversion, Land Cover Classification, Boundary Detection, Geocoding) |
| **Category** | Data Intake Agent (Geospatial) |
| **Priority** | P0 - Critical (required for EUDR geolocation, deforestation monitoring, asset mapping) |
| **Status** | Layer 1 Partial (~7 files in governance/validation/geolocation), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires geospatial data processing capabilities for EUDR compliance
(plot geolocation), deforestation monitoring, carbon credit verification, asset mapping, and
climate hazard assessment. Without a production-grade GIS/Mapping connector:

- **No unified geospatial data ingestion**: Multiple formats (GeoJSON, Shapefile, KML, WKT, GeoPackage) not uniformly handled
- **No coordinate reference system transformation**: Cannot convert between WGS84, UTM, national CRS systems
- **No spatial analysis engine**: No intersection, buffer, distance, area, containment operations
- **No land cover classification**: Cannot classify land use from geospatial data
- **No boundary detection**: Country, region, protected area boundaries not resolvable
- **No geocoding/reverse geocoding**: Cannot convert between addresses and coordinates
- **No format conversion**: Cannot convert between geospatial formats bidirectionally
- **No raster data processing**: Satellite imagery and elevation data not processable
- **No spatial indexing**: No efficient spatial queries for large datasets
- **No audit trail**: Geospatial operations not tracked for compliance

## 3. Existing Implementation

### 3.1 Layer 1: Geolocation Validation Module
**Directory**: `greenlang/governance/validation/geolocation/` (7 files)

- `geojson_parser.py`: GeoJSON parsing (Point, Polygon, MultiPolygon), bounding box calculation, coordinate extraction, ring validation
- `coordinate_validator.py`: WGS84 validation, distance calculation (Haversine/Vincenty), buffer generation, coordinate system support (WGS84, UTM, Web Mercator), area calculation, precision levels
- `protected_area_checker.py`: WDPA (World Database on Protected Areas) intersection checking, IUCN category classification, protection level assessment
- `deforestation_baseline.py`: Dec 31 2020 EUDR cutoff checking, forest status assessment, deforestation risk scoring, Hansen Global Forest Change integration
- `country_risk_db.py`: 249 country risk profiles, governance scores, commodity risk by country, EU Article 29 risk benchmarking
- `validation_engine.py`: Orchestrates all checks, generates compliance reports, SHA-256 provenance, compliance scoring

### 3.2 Layer 1 Tests
None found in production test suite.

## 4. Identified Gaps

### Gap 1: No GIS Connector SDK Package
No `greenlang/gis_connector/` package providing a clean SDK for geospatial operations.

### Gap 2: No Multi-Format Ingestion
Layer 1 only handles GeoJSON. Missing: Shapefile (.shp/.dbf/.shx/.prj), KML/KMZ, GML, WKT/WKB, GeoPackage (.gpkg), CSV with coordinates, GeoTIFF (raster).

### Gap 3: No CRS Transformation Engine
Layer 1 has basic WGS84 validation. Missing: full CRS transformation (EPSG database), UTM zone detection, datum transformations, projection-aware calculations.

### Gap 4: No Spatial Query Engine
Missing: R-tree spatial indexing, spatial joins, intersection/union/difference operations, nearest neighbor, point-in-polygon, polygon clipping.

### Gap 5: No Land Cover Classification
Missing: Corine Land Cover integration, GlobeLand30, land use/land cover (LULC) classification, change detection.

### Gap 6: No Geocoding Engine
Missing: Forward geocoding (address -> coordinates), reverse geocoding (coordinates -> address/region), batch geocoding.

### Gap 7: No Prometheus Metrics
No 12-metric pattern for GIS connector monitoring.

### Gap 8: No REST API
No FastAPI endpoints for geospatial operations.

### Gap 9: No Database Migration
No persistent storage for geospatial data, layers, and operations.

### Gap 10: No K8s/CI/CD
No deployment manifests or CI/CD pipeline.

## 5. Architecture (Final State)

### 5.1 SDK Package Structure

```
greenlang/gis_connector/
├── __init__.py              # Public API, agent metadata (GL-DATA-GEO-001)
├── config.py                # GISConnectorConfig with GL_GIS_CONNECTOR_ env prefix
├── models.py                # Pydantic v2 models for all geospatial data structures
├── format_parser.py         # FormatParserEngine - multi-format geospatial ingestion
├── crs_transformer.py       # CRSTransformerEngine - coordinate reference system transformations
├── spatial_analyzer.py      # SpatialAnalyzerEngine - spatial operations and queries
├── land_cover.py            # LandCoverEngine - land cover/land use classification
├── boundary_resolver.py     # BoundaryResolverEngine - country/region/protected area boundaries
├── geocoder.py              # GeocoderEngine - forward/reverse geocoding
├── layer_manager.py         # LayerManagerEngine - geospatial layer management and indexing
├── provenance.py            # ProvenanceTracker - SHA-256 chain-hashed audit trails
├── metrics.py               # 12 Prometheus metrics
├── setup.py                 # GISConnectorService facade
└── api/
    ├── __init__.py
    └── router.py            # FastAPI HTTP service with 20 endpoints
```

### 5.2 Seven Core Engines

#### Engine 1: FormatParserEngine
- Parse GeoJSON (Point, LineString, Polygon, MultiPolygon, GeometryCollection, Feature, FeatureCollection)
- Parse Shapefile bundles (.shp/.dbf/.shx/.prj) from zip archives
- Parse KML/KMZ files with style extraction
- Parse GML documents
- Parse WKT/WKB strings
- Parse CSV with coordinate columns (auto-detect lat/lon columns)
- Parse GeoPackage (.gpkg) SQLite containers
- Validate geometry (ring closure, winding order, self-intersection detection)
- Extract attributes/properties from all formats
- Detect coordinate reference system from format metadata

#### Engine 2: CRSTransformerEngine
- Transform coordinates between any two EPSG-coded CRS
- Built-in support for common CRS: WGS84 (4326), Web Mercator (3857), UTM zones (326xx/327xx), NAD83 (4269), ETRS89 (4258)
- Auto-detect UTM zone from WGS84 coordinates
- Datum transformation (NAD27->NAD83, ED50->ETRS89, etc.)
- Batch coordinate transformation
- Projection-aware area and distance calculations
- CRS metadata lookup from EPSG database (5000+ entries)
- Validate CRS compatibility

#### Engine 3: SpatialAnalyzerEngine
- Point-in-polygon testing (ray casting algorithm)
- Polygon intersection, union, difference, symmetric difference
- Buffer generation (point, line, polygon buffers with configurable distance)
- Distance calculation (Haversine for spherical, Vincenty for ellipsoidal)
- Area calculation (geodesic area using Shoelace formula with WGS84 correction)
- Centroid calculation
- Bounding box computation
- Nearest neighbor search
- Spatial containment (within, contains, overlaps, touches, crosses)
- Convex hull generation
- Polygon simplification (Douglas-Peucker algorithm)

#### Engine 4: LandCoverEngine
- Land cover type classification (13 types: forest, cropland, grassland, wetland, settlement, bare_land, water, shrubland, mangrove, plantation, snow_ice, desert, other)
- CORINE Land Cover code mapping (CLC 2018, 44 classes -> 13 simplified types)
- GlobeLand30 classification integration
- Forest canopy cover percentage estimation
- Land cover change detection between two dates
- Carbon stock estimation by land cover type
- Protected area overlap assessment
- Deforestation risk scoring based on land cover transitions

#### Engine 5: BoundaryResolverEngine
- Country boundary detection from coordinates (ISO 3166 alpha-2/alpha-3)
- Administrative region resolution (level 0-3: country/state/district/municipality)
- Protected area boundary detection (WDPA database, IUCN categories)
- Exclusive Economic Zone (EEZ) detection for maritime coordinates
- River basin / watershed boundary resolution
- Climate zone classification (Koppen-Geiger)
- Biome classification
- Custom boundary layer registration and query

#### Engine 6: GeocoderEngine
- Forward geocoding (address/place name -> coordinates)
- Reverse geocoding (coordinates -> nearest address/place name)
- Batch geocoding (multiple addresses in parallel)
- Geocoding confidence scoring
- Address normalization and parsing
- Country-specific address format handling
- Coordinate format parsing (DMS, DD, DDM, MGRS, UTM)
- Place name search with fuzzy matching
- Elevation lookup for coordinates

#### Engine 7: LayerManagerEngine
- Register and manage geospatial layers
- Layer metadata (name, CRS, extent, feature count, geometry type)
- Layer indexing for fast spatial queries (R-tree index simulation)
- Layer merging and splitting
- Feature extraction from layers
- Layer statistics (extent, feature count, geometry type distribution)
- Layer export to multiple formats (GeoJSON, WKT, KML)
- Layer versioning and change tracking
- Spatial join between layers

### 5.3 Database Schema

**Schema**: `gis_connector_service`

| Table | Purpose | Type |
|-------|---------|------|
| `geospatial_layers` | Registered geospatial data layers | Regular |
| `layer_features` | Individual features within layers | Regular |
| `crs_definitions` | Coordinate reference system registry | Regular |
| `spatial_operations` | Audit log of spatial operations | Hypertable |
| `geocoding_cache` | Cached geocoding results | Regular |
| `boundary_datasets` | Registered boundary datasets | Regular |
| `land_cover_data` | Land cover classification data | Regular |
| `format_conversions` | Format conversion audit log | Hypertable |
| `spatial_indexes` | Spatial index metadata | Regular |
| `operation_metrics` | Per-operation performance metrics | Hypertable |

### 5.4 Prometheus Metrics (12)

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | `gl_gis_connector_operations_total` | Counter | `operation`, `format`, `status` |
| 2 | `gl_gis_connector_operation_duration_seconds` | Histogram | `operation`, `format` |
| 3 | `gl_gis_connector_format_conversions_total` | Counter | `source_format`, `target_format` |
| 4 | `gl_gis_connector_crs_transformations_total` | Counter | `source_crs`, `target_crs` |
| 5 | `gl_gis_connector_spatial_queries_total` | Counter | `query_type`, `status` |
| 6 | `gl_gis_connector_geocoding_requests_total` | Counter | `direction`, `status` |
| 7 | `gl_gis_connector_features_processed_total` | Counter | `layer`, `operation` |
| 8 | `gl_gis_connector_active_layers` | Gauge | - |
| 9 | `gl_gis_connector_layer_features_count` | Gauge | `layer` |
| 10 | `gl_gis_connector_processing_errors_total` | Counter | `operation`, `error_type` |
| 11 | `gl_gis_connector_cache_hit_rate` | Gauge | `cache_type` |
| 12 | `gl_gis_connector_data_volume_bytes` | Histogram | `format` |

### 5.5 REST API Endpoints (20)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/v1/gis/parse` | Parse geospatial data (auto-detect format) |
| 2 | POST | `/v1/gis/parse/{format}` | Parse specific format (geojson/shapefile/kml/wkt/csv) |
| 3 | POST | `/v1/gis/convert` | Convert between geospatial formats |
| 4 | POST | `/v1/gis/transform` | Transform coordinates between CRS |
| 5 | POST | `/v1/gis/analyze/intersection` | Compute geometry intersection |
| 6 | POST | `/v1/gis/analyze/buffer` | Generate geometry buffer |
| 7 | POST | `/v1/gis/analyze/distance` | Calculate distance between geometries |
| 8 | POST | `/v1/gis/analyze/area` | Calculate geometry area |
| 9 | POST | `/v1/gis/analyze/contains` | Test spatial containment |
| 10 | POST | `/v1/gis/classify/landcover` | Classify land cover type |
| 11 | POST | `/v1/gis/resolve/boundary` | Resolve boundary from coordinates |
| 12 | POST | `/v1/gis/resolve/country` | Resolve country from coordinates |
| 13 | POST | `/v1/gis/geocode/forward` | Forward geocoding |
| 14 | POST | `/v1/gis/geocode/reverse` | Reverse geocoding |
| 15 | GET | `/v1/gis/layers` | List registered layers |
| 16 | POST | `/v1/gis/layers` | Register new layer |
| 17 | GET | `/v1/gis/layers/{layer_id}` | Get layer details |
| 18 | GET | `/v1/gis/layers/{layer_id}/features` | Get layer features |
| 19 | GET | `/v1/gis/health` | Health check |
| 20 | GET | `/v1/gis/statistics` | Service statistics |

### 5.6 Configuration

**Environment Variable Prefix**: `GL_GIS_CONNECTOR_`

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_GIS_CONNECTOR_DATABASE_URL` | `""` | PostgreSQL connection string |
| `GL_GIS_CONNECTOR_REDIS_URL` | `""` | Redis connection string |
| `GL_GIS_CONNECTOR_LOG_LEVEL` | `"INFO"` | Logging level |
| `GL_GIS_CONNECTOR_DEFAULT_CRS` | `"EPSG:4326"` | Default coordinate reference system |
| `GL_GIS_CONNECTOR_MAX_FEATURES` | `100000` | Maximum features per layer |
| `GL_GIS_CONNECTOR_MAX_FILE_SIZE_MB` | `500` | Maximum upload file size |
| `GL_GIS_CONNECTOR_COORDINATE_PRECISION` | `6` | Decimal precision for coordinates |
| `GL_GIS_CONNECTOR_BUFFER_DISTANCE_DEFAULT` | `1000.0` | Default buffer distance in meters |
| `GL_GIS_CONNECTOR_GEOCODING_CACHE_TTL` | `86400` | Geocoding cache TTL in seconds (24h) |
| `GL_GIS_CONNECTOR_SIMPLIFICATION_TOLERANCE` | `0.001` | Douglas-Peucker simplification tolerance |
| `GL_GIS_CONNECTOR_BATCH_SIZE` | `1000` | Batch processing size |
| `GL_GIS_CONNECTOR_WORKER_COUNT` | `4` | Parallel workers |
| `GL_GIS_CONNECTOR_POOL_MIN_SIZE` | `2` | DB pool minimum |
| `GL_GIS_CONNECTOR_POOL_MAX_SIZE` | `10` | DB pool maximum |
| `GL_GIS_CONNECTOR_RETENTION_DAYS` | `365` | Operation log retention |
| `GL_GIS_CONNECTOR_ENABLE_RASTER` | `false` | Enable raster data support |
| `GL_GIS_CONNECTOR_ENABLE_3D` | `false` | Enable 3D coordinate support |

## 6. Completion Plan

### Phase 1: SDK Core
1. Build config.py, models.py, __init__.py
2. Build 7 core engines
3. Build provenance.py, metrics.py, setup.py
4. Build api/router.py

### Phase 2: Infrastructure
5. Build V036 database migration
6. Build K8s manifests (8 files)
7. Build CI/CD pipeline
8. Build Grafana dashboard + alerts

### Phase 3: Testing
9. Build 600+ unit tests across 13 test files

## 7. Success Criteria

- [ ] 7 engines with deterministic geospatial operations
- [ ] 8 geospatial format support (GeoJSON, Shapefile, KML, GML, WKT, WKB, CSV, GeoPackage)
- [ ] 5000+ EPSG CRS definitions for transformation
- [ ] 13 land cover types with CORINE mapping
- [ ] 249 country boundaries with ISO 3166 codes
- [ ] Haversine + Vincenty distance calculations
- [ ] R-tree spatial indexing simulation
- [ ] 20 REST API endpoints operational
- [ ] 12 Prometheus metrics instrumented
- [ ] SHA-256 provenance on all operations
- [ ] V036 database migration with 10 tables
- [ ] 600+ tests passing
- [ ] K8s manifests with full security hardening

## 8. Integration Points

### Upstream Dependencies
- AGENT-DATA-005 EUDR Traceability (geolocation plot validation)
- AGENT-FOUND-002 Schema Compiler (schema validation)
- AGENT-FOUND-003 Unit Normalizer (unit conversion)
- AGENT-FOUND-006 Access Guard (authorization)

### Downstream Consumers
- EUDR plot registration (geolocation validation)
- Carbon credit verification (land cover assessment)
- Climate hazard assessment (spatial analysis)
- Deforestation monitoring (boundary + land cover)
- Asset mapping (geocoding + boundary resolution)
