# GIS/Mapping Connector API Reference

**Agent:** AGENT-DATA-006 (GL-DATA-GEO-001)
**Prefix:** `/v1/gis`
**Source:** `greenlang/agents/data/gis_connector/api/router.py`
**Status:** Production Ready

## Overview

The GIS/Mapping Connector agent provides 20 REST API endpoints for geospatial data operations. Capabilities include multi-format parsing (GeoJSON, WKT, CSV, KML), coordinate reference system (CRS) transformations, spatial analysis (distance, area, containment), land cover classification with carbon stock estimation, boundary resolution (country and climate zone), forward and reverse geocoding, and geospatial layer management.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/parse` | Parse geospatial data | Yes |
| 2 | POST | `/parse/batch` | Batch parse multiple datasets | Yes |
| 3 | GET | `/parse/{parse_id}` | Get parse result by ID | Yes |
| 4 | POST | `/transform` | Transform CRS coordinates | Yes |
| 5 | POST | `/transform/geometry` | Transform full geometry CRS | Yes |
| 6 | GET | `/crs` | List available CRS definitions | Yes |
| 7 | GET | `/crs/{epsg_code}` | Get CRS info | Yes |
| 8 | POST | `/spatial/distance` | Calculate distance between points | Yes |
| 9 | POST | `/spatial/area` | Calculate polygon area | Yes |
| 10 | POST | `/spatial/contains` | Test geometry containment | Yes |
| 11 | POST | `/land-cover/classify` | Classify land cover | Yes |
| 12 | POST | `/land-cover/carbon` | Estimate carbon stock | Yes |
| 13 | POST | `/boundary/country` | Resolve country from coordinates | Yes |
| 14 | POST | `/boundary/climate` | Resolve climate zone | Yes |
| 15 | POST | `/geocode/forward` | Forward geocode (address to coordinates) | Yes |
| 16 | POST | `/geocode/reverse` | Reverse geocode (coordinates to address) | Yes |
| 17 | POST | `/layers` | Create geospatial layer | Yes |
| 18 | GET | `/layers` | List geospatial layers | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/statistics` | Service statistics | Yes |

---

## Key Endpoints

### 1. Parse Geospatial Data

Parse geospatial data from GeoJSON, WKT, CSV, or KML format. Auto-detects format if not specified.

```http
POST /v1/gis/parse
```

**Request Body:**

```json
{
  "data": "{\"type\": \"Point\", \"coordinates\": [13.4050, 52.5200]}",
  "format": "geojson",
  "source_crs": "EPSG:4326"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | string | Yes | Raw geospatial data string |
| `format` | string | No | Data format: `geojson`, `wkt`, `csv`, `kml` (auto-detect if omitted) |
| `source_crs` | string | No | Source CRS (e.g., `EPSG:4326`) |

**Response (200):**

```json
{
  "parse_id": "parse_abc123",
  "source_format": "geojson",
  "geometry_type": "Point",
  "feature_count": 1,
  "is_valid": true,
  "errors": []
}
```

**Status Codes:** `200` Success | `400` Invalid data | `500` Server error

---

### 2. Batch Parse

Parse multiple geospatial datasets in a single request.

```http
POST /v1/gis/parse/batch
```

**Request Body:**

```json
{
  "items": [
    {"data": "POINT(13.4050 52.5200)", "format": "wkt"},
    {"data": "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", "format": "wkt"}
  ]
}
```

**Response (200):**

```json
{
  "results": [
    {"parse_id": "parse_001", "source_format": "wkt", "geometry_type": "Point", "feature_count": 1, "is_valid": true, "errors": []},
    {"parse_id": "parse_002", "source_format": "wkt", "geometry_type": "Polygon", "feature_count": 1, "is_valid": true, "errors": []}
  ],
  "total": 2
}
```

---

### 4. Transform CRS Coordinates

Transform coordinates between coordinate reference systems.

```http
POST /v1/gis/transform
```

**Request Body:**

```json
{
  "coordinates": [13.4050, 52.5200],
  "source_crs": "EPSG:4326",
  "target_crs": "EPSG:3857"
}
```

**Response (200):**

```json
{
  "transform_id": "tfm_abc123",
  "source_crs": "EPSG:4326",
  "target_crs": "EPSG:3857",
  "output_coordinates": [1492166.24, 6894032.88],
  "method": "proj_transform"
}
```

---

### 8. Calculate Distance

Calculate the distance between two geographic points.

```http
POST /v1/gis/spatial/distance
```

**Request Body:**

```json
{
  "point_a": [13.4050, 52.5200],
  "point_b": [2.3522, 48.8566],
  "method": "haversine"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `point_a` | array | Yes | First point as `[longitude, latitude]` |
| `point_b` | array | Yes | Second point as `[longitude, latitude]` |
| `method` | string | No | Calculation method: `haversine` (default), `vincenty` |

**Response (200):**

```json
{
  "result_id": "sr_abc123",
  "operation": "distance",
  "output_data": {"distance_meters": 877463.12},
  "unit": "meters"
}
```

---

### 9. Calculate Polygon Area

```http
POST /v1/gis/spatial/area
```

**Request Body:**

```json
{
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[107.5, -6.9], [107.6, -6.9], [107.6, -6.8], [107.5, -6.8], [107.5, -6.9]]]
  }
}
```

**Response (200):**

```json
{
  "result_id": "sr_def456",
  "operation": "area",
  "output_data": {"area_sq_meters": 12345678.90},
  "unit": "square_meters"
}
```

---

### 11. Classify Land Cover

Classify land cover type at a geographic coordinate, using CORINE land cover codes.

```http
POST /v1/gis/land-cover/classify
```

**Request Body:**

```json
{
  "coordinate": [107.55, -6.85],
  "corine_code": "311"
}
```

---

### 13. Resolve Country

Resolve the country from geographic coordinates.

```http
POST /v1/gis/boundary/country
```

**Request Body:**

```json
{
  "coordinate": [107.55, -6.85]
}
```

---

### 15. Forward Geocode

Convert an address string to geographic coordinates.

```http
POST /v1/gis/geocode/forward
```

**Request Body:**

```json
{
  "address": "Alexanderplatz 1, Berlin, Germany",
  "country_hint": "DE",
  "limit": 5
}
```

---

### 17. Create Layer

Create a new geospatial layer with features.

```http
POST /v1/gis/layers
```

**Request Body:**

```json
{
  "name": "Production Sites 2026",
  "geometry_type": "Point",
  "crs": "EPSG:4326",
  "description": "All production sites for 2026 reporting",
  "tags": ["production", "2026"],
  "features": [
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [13.4, 52.5]}, "properties": {"name": "Berlin Site"}}
  ]
}
```

**Response (200):**

```json
{
  "layer_id": "layer_abc123",
  "name": "Production Sites 2026",
  "geometry_type": "Point",
  "crs": "EPSG:4326",
  "feature_count": 1,
  "status": "active"
}
```

---

## Error Responses

All error responses follow a standard format:

```json
{
  "detail": "Descriptive error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- invalid input data or unsupported format |
| 404 | Not Found -- resource does not exist |
| 500 | Internal Server Error |
