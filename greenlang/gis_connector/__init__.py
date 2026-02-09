# -*- coding: utf-8 -*-
"""
GL-DATA-GEO-001: GreenLang GIS/Mapping Connector Agent Service SDK
====================================================================

This package provides geospatial data processing capabilities for the
GreenLang platform. It supports:

- 8 geospatial data formats (GeoJSON, WKT, WKB, KML, GML, Shapefile,
  CSV, TopoJSON)
- 7 geometry types (Point, LineString, Polygon, MultiPoint,
  MultiLineString, MultiPolygon, GeometryCollection)
- CRS transformation between 50+ EPSG codes (WGS84, UTM, Web Mercator)
- Spatial analysis (distance, area, point-in-polygon, convex hull,
  buffer, simplification)
- Land cover classification using CORINE codes and IPCC carbon stocks
- Administrative boundary resolution (country, state, county, city,
  district, postal code) using ISO 3166 and bounding box databases
- Forward and reverse geocoding with multi-format coordinate parsing
  (DMS, DD, DDM, UTM, MGRS)
- Layer management with CRUD, spatial indexing, and GeoJSON/WKT export
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_GIS_CONNECTOR_ env prefix

Key Components:
    - config: GISConnectorConfig with GL_GIS_CONNECTOR_ env prefix
    - models: Pydantic v2 models for all data structures
    - format_parser: Multi-format geospatial data parsing engine
    - crs_transformer: Coordinate Reference System transformation engine
    - spatial_analyzer: Spatial analysis and computation engine
    - land_cover: Land cover classification and carbon stock engine
    - boundary_resolver: Administrative boundary resolution engine
    - geocoder: Forward and reverse geocoding engine
    - layer_manager: Geospatial layer CRUD and export engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - setup: GISConnectorService facade

Example:
    >>> from greenlang.gis_connector import GISConnectorService
    >>> service = GISConnectorService()
    >>> result = service.parse_data('{"type":"Point","coordinates":[0,0]}')
    >>> print(result["parse_id"])

Agent ID: GL-DATA-GEO-001
Agent Name: GIS/Mapping Connector Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-GEO-001"
__agent_name__ = "GIS/Mapping Connector Agent"

# SDK availability flag
GIS_CONNECTOR_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.gis_connector.config import (
    GISConnectorConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, data models, request models)
# ---------------------------------------------------------------------------
from greenlang.gis_connector.models import (
    # Enumerations
    GeometryType,
    CRSType,
    GeoFormat,
    SpatialOperation,
    LandCoverType,
    BoundaryType,
    DataSourceStatus,
    LayerVisibility,
    TransformStatus,
    ValidationSeverity,
    # Core data models
    Coordinate,
    BoundingBox,
    Geometry,
    Feature,
    GeoLayer,
    CRSDefinition,
    SpatialResult,
    LandCoverClassification,
    BoundaryResult,
    GeocodingResult,
    FormatConversionResult,
    TransformResult,
    GISStatistics,
    OperationLog,
    # Request models
    ParseFormatRequest,
    TransformCRSRequest,
    SpatialAnalysisRequest,
    GeocodingRequest,
    LayerCreateRequest,
    BoundaryQueryRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.gis_connector.format_parser import FormatParserEngine
from greenlang.gis_connector.crs_transformer import CRSTransformerEngine
from greenlang.gis_connector.spatial_analyzer import SpatialAnalyzerEngine
from greenlang.gis_connector.land_cover import LandCoverEngine
from greenlang.gis_connector.boundary_resolver import BoundaryResolverEngine
from greenlang.gis_connector.geocoder import GeocoderEngine
from greenlang.gis_connector.layer_manager import LayerManagerEngine
from greenlang.gis_connector.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.gis_connector.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    gis_connector_operations_total,
    gis_connector_operation_duration_seconds,
    gis_connector_format_conversions_total,
    gis_connector_crs_transformations_total,
    gis_connector_spatial_queries_total,
    gis_connector_geocoding_requests_total,
    gis_connector_features_processed_total,
    gis_connector_active_layers,
    gis_connector_layer_features_count,
    gis_connector_processing_errors_total,
    gis_connector_cache_hit_rate,
    gis_connector_data_volume_bytes,
    # Helper functions
    record_operation,
    record_format_conversion,
    record_crs_transformation,
    record_spatial_query,
    record_geocoding_request,
    record_features_processed,
    record_processing_error,
    update_active_layers,
    update_layer_features,
    update_cache_hit_rate,
    record_data_volume,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.gis_connector.setup import (
    GISConnectorService,
    configure_gis_connector,
    get_gis_connector,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "GIS_CONNECTOR_SDK_AVAILABLE",
    # Configuration
    "GISConnectorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations
    "GeometryType",
    "CRSType",
    "GeoFormat",
    "SpatialOperation",
    "LandCoverType",
    "BoundaryType",
    "DataSourceStatus",
    "LayerVisibility",
    "TransformStatus",
    "ValidationSeverity",
    # Core data models
    "Coordinate",
    "BoundingBox",
    "Geometry",
    "Feature",
    "GeoLayer",
    "CRSDefinition",
    "SpatialResult",
    "LandCoverClassification",
    "BoundaryResult",
    "GeocodingResult",
    "FormatConversionResult",
    "TransformResult",
    "GISStatistics",
    "OperationLog",
    # Request models
    "ParseFormatRequest",
    "TransformCRSRequest",
    "SpatialAnalysisRequest",
    "GeocodingRequest",
    "LayerCreateRequest",
    "BoundaryQueryRequest",
    # Core engines
    "FormatParserEngine",
    "CRSTransformerEngine",
    "SpatialAnalyzerEngine",
    "LandCoverEngine",
    "BoundaryResolverEngine",
    "GeocoderEngine",
    "LayerManagerEngine",
    "ProvenanceTracker",
    # Metric flag
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "gis_connector_operations_total",
    "gis_connector_operation_duration_seconds",
    "gis_connector_format_conversions_total",
    "gis_connector_crs_transformations_total",
    "gis_connector_spatial_queries_total",
    "gis_connector_geocoding_requests_total",
    "gis_connector_features_processed_total",
    "gis_connector_active_layers",
    "gis_connector_layer_features_count",
    "gis_connector_processing_errors_total",
    "gis_connector_cache_hit_rate",
    "gis_connector_data_volume_bytes",
    # Metric helper functions
    "record_operation",
    "record_format_conversion",
    "record_crs_transformation",
    "record_spatial_query",
    "record_geocoding_request",
    "record_features_processed",
    "record_processing_error",
    "update_active_layers",
    "update_layer_features",
    "update_cache_hit_rate",
    "record_data_volume",
    # Service setup facade
    "GISConnectorService",
    "configure_gis_connector",
    "get_gis_connector",
    "get_router",
]
