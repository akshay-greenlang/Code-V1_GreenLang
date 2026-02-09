# -*- coding: utf-8 -*-
"""
GIS Connector Service Facade - AGENT-DATA-006: GIS/Mapping Connector (GL-DATA-GEO-001)

Provides the main service class and FastAPI integration functions:
- GISConnectorService: Composes all 7 engines into a single facade
- configure_gis_connector(app): Register service on FastAPI app
- get_gis_connector(app): Retrieve service from app state
- get_router(): Return FastAPI router for mounting

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GISConnectorService:
    """Facade composing all GIS Connector engines.

    Provides a single entry point for all GIS/mapping operations,
    delegating to the appropriate engine for each operation type.

    Attributes:
        config: Configuration dictionary or object.
        provenance: ProvenanceTracker instance.
        format_parser: FormatParserEngine instance.
        crs_transformer: CRSTransformerEngine instance.
        spatial_analyzer: SpatialAnalyzerEngine instance.
        land_cover: LandCoverEngine instance.
        boundary_resolver: BoundaryResolverEngine instance.
        geocoder: GeocoderEngine instance.
        layer_manager: LayerManagerEngine instance.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the GIS Connector Service with all engines.

        Args:
            config: Configuration dictionary or object. If None, uses defaults.
        """
        self.config = config or {}

        # Initialize provenance
        from greenlang.gis_connector.provenance import ProvenanceTracker
        self.provenance = ProvenanceTracker()

        # Initialize engines
        from greenlang.gis_connector.format_parser import FormatParserEngine
        from greenlang.gis_connector.crs_transformer import CRSTransformerEngine
        from greenlang.gis_connector.spatial_analyzer import SpatialAnalyzerEngine
        from greenlang.gis_connector.land_cover import LandCoverEngine
        from greenlang.gis_connector.boundary_resolver import BoundaryResolverEngine
        from greenlang.gis_connector.geocoder import GeocoderEngine
        from greenlang.gis_connector.layer_manager import LayerManagerEngine

        self.format_parser = FormatParserEngine(
            config=config,
            provenance=self.provenance,
        )
        self.crs_transformer = CRSTransformerEngine(
            config=config,
            provenance=self.provenance,
        )
        self.spatial_analyzer = SpatialAnalyzerEngine(
            config=config,
            provenance=self.provenance,
        )
        self.land_cover = LandCoverEngine(
            config=config,
            provenance=self.provenance,
        )
        self.boundary_resolver = BoundaryResolverEngine(
            config=config,
            provenance=self.provenance,
        )
        self.geocoder = GeocoderEngine(
            config=config,
            provenance=self.provenance,
        )
        self.layer_manager = LayerManagerEngine(
            config=config,
            provenance=self.provenance,
        )

        logger.info(
            "GISConnectorService initialized with all 7 engines + provenance"
        )

    # =========================================================================
    # Format Parsing Delegation
    # =========================================================================

    def parse_data(
        self,
        data: Any,
        format: Optional[str] = None,
        source_crs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse geospatial data. Delegates to FormatParserEngine.

        Args:
            data: Raw geospatial data (string, dict, bytes).
            format: Optional format hint.
            source_crs: Optional source CRS.

        Returns:
            ParseResult dictionary.
        """
        return self.format_parser.parse(data, format=format, source_crs=source_crs)

    def detect_format(self, data: Any) -> Optional[str]:
        """Detect geospatial format. Delegates to FormatParserEngine.

        Args:
            data: Raw data to detect.

        Returns:
            Format string or None.
        """
        return self.format_parser.detect_format(data)

    # =========================================================================
    # CRS Transformation Delegation
    # =========================================================================

    def transform_coordinates(
        self,
        coordinates: List[float],
        source_crs: str,
        target_crs: str,
    ) -> Dict[str, Any]:
        """Transform coordinates. Delegates to CRSTransformerEngine.

        Args:
            coordinates: [x, y] coordinate pair.
            source_crs: Source CRS identifier.
            target_crs: Target CRS identifier.

        Returns:
            TransformResult dictionary.
        """
        return self.crs_transformer.transform(coordinates, source_crs, target_crs)

    def transform_geometry(
        self,
        geometry: Dict[str, Any],
        source_crs: str,
        target_crs: str,
    ) -> Dict[str, Any]:
        """Transform geometry. Delegates to CRSTransformerEngine.

        Args:
            geometry: Geometry dictionary.
            source_crs: Source CRS identifier.
            target_crs: Target CRS identifier.

        Returns:
            Transformed geometry dictionary.
        """
        return self.crs_transformer.transform_geometry(geometry, source_crs, target_crs)

    # =========================================================================
    # Spatial Analysis Delegation
    # =========================================================================

    def calculate_distance(
        self,
        point_a: List[float],
        point_b: List[float],
        method: str = "haversine",
    ) -> Dict[str, Any]:
        """Calculate distance. Delegates to SpatialAnalyzerEngine.

        Args:
            point_a: [lon, lat] first point.
            point_b: [lon, lat] second point.
            method: Distance method.

        Returns:
            SpatialResult dictionary.
        """
        return self.spatial_analyzer.distance(point_a, point_b, method)

    def calculate_area(self, polygon: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate polygon area. Delegates to SpatialAnalyzerEngine.

        Args:
            polygon: Polygon geometry dictionary.

        Returns:
            SpatialResult dictionary.
        """
        return self.spatial_analyzer.area(polygon)

    def point_in_polygon(
        self,
        point: List[float],
        polygon: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Test point in polygon. Delegates to SpatialAnalyzerEngine.

        Args:
            point: [lon, lat] coordinate.
            polygon: Polygon geometry dictionary.

        Returns:
            SpatialResult dictionary.
        """
        return self.spatial_analyzer.point_in_polygon(point, polygon)

    # =========================================================================
    # Land Cover Delegation
    # =========================================================================

    def classify_land_cover(
        self,
        coordinate: List[float],
        corine_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify land cover. Delegates to LandCoverEngine.

        Args:
            coordinate: [lon, lat] coordinate.
            corine_code: Optional CORINE code.

        Returns:
            LandCoverClassification dictionary.
        """
        return self.land_cover.classify(coordinate, corine_code)

    def estimate_carbon_stock(self, land_cover_type: str) -> Dict[str, Any]:
        """Estimate carbon stock. Delegates to LandCoverEngine.

        Args:
            land_cover_type: Land cover type.

        Returns:
            Carbon stock dictionary.
        """
        return self.land_cover.estimate_carbon_stock(land_cover_type)

    # =========================================================================
    # Boundary Resolution Delegation
    # =========================================================================

    def resolve_country(self, coordinate: List[float]) -> Dict[str, Any]:
        """Resolve country. Delegates to BoundaryResolverEngine.

        Args:
            coordinate: [lon, lat] coordinate.

        Returns:
            BoundaryResult dictionary.
        """
        return self.boundary_resolver.resolve_country(coordinate)

    def resolve_climate_zone(self, coordinate: List[float]) -> Dict[str, Any]:
        """Resolve climate zone. Delegates to BoundaryResolverEngine.

        Args:
            coordinate: [lon, lat] coordinate.

        Returns:
            BoundaryResult dictionary.
        """
        return self.boundary_resolver.resolve_climate_zone(coordinate)

    # =========================================================================
    # Geocoding Delegation
    # =========================================================================

    def forward_geocode(
        self,
        address: str,
        country_hint: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Forward geocode. Delegates to GeocoderEngine.

        Args:
            address: Address string.
            country_hint: Optional country hint.
            limit: Max results.

        Returns:
            GeocodingResult dictionary.
        """
        return self.geocoder.forward(address, country_hint, limit)

    def reverse_geocode(
        self,
        coordinate: List[float],
        limit: int = 1,
    ) -> Dict[str, Any]:
        """Reverse geocode. Delegates to GeocoderEngine.

        Args:
            coordinate: [lon, lat] coordinate.
            limit: Max results.

        Returns:
            GeocodingResult dictionary.
        """
        return self.geocoder.reverse(coordinate, limit)

    # =========================================================================
    # Layer Management Delegation
    # =========================================================================

    def create_layer(
        self,
        name: str,
        geometry_type: str,
        crs: str = "EPSG:4326",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a layer. Delegates to LayerManagerEngine.

        Args:
            name: Layer name.
            geometry_type: Geometry type.
            crs: CRS identifier.
            **kwargs: Additional layer parameters.

        Returns:
            GeoLayer dictionary.
        """
        return self.layer_manager.create_layer(name, geometry_type, crs, **kwargs)

    def get_layer(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Get layer. Delegates to LayerManagerEngine.

        Args:
            layer_id: Layer identifier.

        Returns:
            GeoLayer dictionary or None.
        """
        return self.layer_manager.get_layer(layer_id)

    def export_layer(
        self,
        layer_id: str,
        format: str = "geojson",
    ) -> Dict[str, Any]:
        """Export layer. Delegates to LayerManagerEngine.

        Args:
            layer_id: Layer identifier.
            format: Export format.

        Returns:
            Export result dictionary.
        """
        return self.layer_manager.export_layer(layer_id, format)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics.

        Returns:
            Dictionary with statistics from all engines.
        """
        return {
            "agent_id": "GL-DATA-GEO-001",
            "agent_name": "GIS/Mapping Connector",
            "version": "1.0.0",
            "format_parser": self.format_parser.get_statistics(),
            "crs_transformer": self.crs_transformer.get_statistics(),
            "spatial_analyzer": self.spatial_analyzer.get_statistics(),
            "land_cover": self.land_cover.get_statistics(),
            "boundary_resolver": self.boundary_resolver.get_statistics(),
            "geocoder": self.geocoder.get_statistics(),
            "layer_manager": self.layer_manager.get_statistics(),
            "provenance": {
                "total_entries": self.provenance.entry_count,
                "total_entities": self.provenance.entity_count,
            },
        }


# =============================================================================
# FastAPI Integration
# =============================================================================

_SERVICE_KEY = "gis_connector_service"


def configure_gis_connector(app: Any) -> GISConnectorService:
    """Register the GIS Connector Service on a FastAPI application.

    Creates the service, attaches it to app.state, and includes the
    API router.

    Args:
        app: FastAPI application instance.

    Returns:
        Configured GISConnectorService instance.
    """
    service = GISConnectorService()
    app.state.gis_connector_service = service

    # Include router
    from greenlang.gis_connector.api.router import router
    app.include_router(router)

    logger.info("GIS Connector Service configured on FastAPI app")
    return service


def get_gis_connector(app: Any) -> GISConnectorService:
    """Retrieve the GIS Connector Service from a FastAPI application.

    Args:
        app: FastAPI application instance.

    Returns:
        GISConnectorService instance.

    Raises:
        RuntimeError: If service not configured.
    """
    service = getattr(app.state, _SERVICE_KEY, None)
    if service is None:
        raise RuntimeError(
            "GIS Connector Service not configured. "
            "Call configure_gis_connector(app) first."
        )
    return service


def get_router():
    """Return the FastAPI router for the GIS Connector Service.

    Returns:
        FastAPI APIRouter instance.
    """
    from greenlang.gis_connector.api.router import router
    return router
