"""
GIS Analytics Bridge - PACK-007 Professional

This module provides enhanced GIS analytics for PACK-007 with advanced spatial analysis.
It integrates with protected area databases (WDPA, KBA) and indigenous land registries.

GIS capabilities:
- Coordinate transformation (WGS84, UTM, local CRS)
- Boundary resolution and validation
- Spatial analysis (intersection, buffer, overlay)
- Land cover classification
- Protected area screening (WDPA, KBA, UNESCO)
- Indigenous lands overlay
- Buffer analysis
- Spatial statistics

Example:
    >>> config = GISAnalyticsConfig(enable_protected_areas=True)
    >>> bridge = GISAnalyticsBridge(config)
    >>> result = await bridge.protected_area_overlay([(lat, lon)])
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class GISAnalyticsConfig(BaseModel):
    """Configuration for GIS analytics bridge."""

    target_crs: str = Field(
        default="EPSG:4326",
        description="Target coordinate reference system (WGS84)"
    )
    enable_protected_areas: bool = Field(
        default=True,
        description="Enable WDPA/KBA protected area screening"
    )
    enable_indigenous_lands: bool = Field(
        default=True,
        description="Enable indigenous lands overlay"
    )
    buffer_distance_meters: float = Field(
        default=1000.0,
        ge=0.0,
        description="Default buffer distance for spatial analysis"
    )
    land_cover_source: Literal["esa_worldcover", "copernicus", "modis"] = Field(
        default="esa_worldcover",
        description="Land cover data source"
    )


class GISAnalyticsBridge:
    """
    Enhanced GIS analytics bridge for PACK-007.

    Provides advanced spatial analysis with protected area screening,
    indigenous lands overlay, and land cover classification.

    Example:
        >>> config = GISAnalyticsConfig()
        >>> bridge = GISAnalyticsBridge(config)
        >>> # Transform coordinates
        >>> result = await bridge.transform_coordinates([(6.5, 3.5)], "EPSG:32631")
    """

    def __init__(self, config: GISAnalyticsConfig):
        """Initialize bridge."""
        self.config = config
        self._service: Any = None
        logger.info("GISAnalyticsBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real GIS service."""
        self._service = service
        logger.info("Injected GIS analytics service")

    async def transform_coordinates(
        self,
        coordinates: List[Tuple[float, float]],
        target_crs: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transform coordinates between coordinate reference systems.

        Args:
            coordinates: List of (latitude, longitude) tuples
            target_crs: Target CRS (default from config)

        Returns:
            Transformed coordinates
        """
        try:
            target = target_crs or self.config.target_crs

            if self._service and hasattr(self._service, "transform_coordinates"):
                return await self._service.transform_coordinates(
                    coordinates=coordinates,
                    source_crs="EPSG:4326",
                    target_crs=target
                )

            # Fallback - identity transformation (assume already in WGS84)
            return {
                "status": "fallback",
                "source_crs": "EPSG:4326",
                "target_crs": target,
                "transformed_coordinates": coordinates,
                "total_points": len(coordinates),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Coordinate transformation failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def resolve_boundaries(
        self,
        plot_id: str,
        coordinates: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Resolve and validate plot boundaries.

        Args:
            plot_id: Plot identifier
            coordinates: Plot boundary coordinates

        Returns:
            Boundary validation results
        """
        try:
            if self._service and hasattr(self._service, "resolve_boundaries"):
                return await self._service.resolve_boundaries(
                    plot_id=plot_id,
                    coordinates=coordinates
                )

            # Fallback
            return {
                "status": "fallback",
                "plot_id": plot_id,
                "coordinates": coordinates,
                "area_hectares": 0.0,
                "perimeter_meters": 0.0,
                "is_valid": True,
                "self_intersecting": False,
                "provenance_hash": self._calculate_hash({
                    "plot": plot_id,
                    "coords": coordinates
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Boundary resolution failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def spatial_analysis(
        self,
        plot_coordinates: List[Tuple[float, float]],
        reference_layer: str,
        analysis_type: Literal["intersection", "buffer", "distance"]
    ) -> Dict[str, Any]:
        """
        Perform spatial analysis operations.

        Args:
            plot_coordinates: Plot boundary coordinates
            reference_layer: Reference spatial layer
            analysis_type: Type of spatial analysis

        Returns:
            Spatial analysis results
        """
        try:
            if self._service and hasattr(self._service, "spatial_analysis"):
                return await self._service.spatial_analysis(
                    plot_coordinates=plot_coordinates,
                    reference_layer=reference_layer,
                    analysis_type=analysis_type
                )

            # Fallback
            return {
                "status": "fallback",
                "analysis_type": analysis_type,
                "reference_layer": reference_layer,
                "plot_coordinates": plot_coordinates,
                "result": {},
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Spatial analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def land_cover_classification(
        self,
        plot_coordinates: List[Tuple[float, float]],
        reference_date: datetime
    ) -> Dict[str, Any]:
        """
        Classify land cover for plot.

        Args:
            plot_coordinates: Plot boundary coordinates
            reference_date: Reference date for land cover

        Returns:
            Land cover classification results
        """
        try:
            if self._service and hasattr(self._service, "land_cover_classification"):
                return await self._service.land_cover_classification(
                    plot_coordinates=plot_coordinates,
                    reference_date=reference_date,
                    source=self.config.land_cover_source
                )

            # Fallback
            return {
                "status": "fallback",
                "plot_coordinates": plot_coordinates,
                "reference_date": reference_date.isoformat(),
                "source": self.config.land_cover_source,
                "classification": {
                    "tree_cover": 0.0,
                    "cropland": 0.0,
                    "grassland": 0.0,
                    "built_area": 0.0,
                    "bare_soil": 0.0,
                    "water": 0.0
                },
                "dominant_class": "unknown",
                "confidence": 0.0,
                "provenance_hash": self._calculate_hash({
                    "coords": plot_coordinates,
                    "date": reference_date.isoformat()
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Land cover classification failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def protected_area_overlay(
        self,
        plot_coordinates: List[Tuple[float, float]],
        include_kba: bool = True,
        include_unesco: bool = True
    ) -> Dict[str, Any]:
        """
        Screen plot against protected area databases.

        Args:
            plot_coordinates: Plot boundary coordinates
            include_kba: Include Key Biodiversity Areas
            include_unesco: Include UNESCO World Heritage Sites

        Returns:
            Protected area screening results
        """
        try:
            if not self.config.enable_protected_areas:
                return {
                    "status": "disabled",
                    "message": "Protected area screening not enabled",
                    "timestamp": datetime.utcnow().isoformat()
                }

            if self._service and hasattr(self._service, "protected_area_overlay"):
                return await self._service.protected_area_overlay(
                    plot_coordinates=plot_coordinates,
                    include_kba=include_kba,
                    include_unesco=include_unesco
                )

            # Fallback - no overlaps
            return {
                "status": "fallback",
                "plot_coordinates": plot_coordinates,
                "wdpa_overlaps": [],
                "kba_overlaps": [] if include_kba else None,
                "unesco_overlaps": [] if include_unesco else None,
                "total_overlaps": 0,
                "max_iucn_category": None,
                "provenance_hash": self._calculate_hash(plot_coordinates),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Protected area overlay failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def indigenous_lands_overlay(
        self,
        plot_coordinates: List[Tuple[float, float]],
        country_code: str
    ) -> Dict[str, Any]:
        """
        Screen plot against indigenous lands registries.

        Args:
            plot_coordinates: Plot boundary coordinates
            country_code: Country code for indigenous lands database

        Returns:
            Indigenous lands screening results
        """
        try:
            if not self.config.enable_indigenous_lands:
                return {
                    "status": "disabled",
                    "message": "Indigenous lands screening not enabled",
                    "timestamp": datetime.utcnow().isoformat()
                }

            if self._service and hasattr(self._service, "indigenous_lands_overlay"):
                return await self._service.indigenous_lands_overlay(
                    plot_coordinates=plot_coordinates,
                    country_code=country_code
                )

            # Fallback - no overlaps
            return {
                "status": "fallback",
                "plot_coordinates": plot_coordinates,
                "country_code": country_code,
                "indigenous_lands_overlaps": [],
                "total_overlaps": 0,
                "requires_fpic": False,
                "provenance_hash": self._calculate_hash({
                    "coords": plot_coordinates,
                    "country": country_code
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Indigenous lands overlay failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def buffer_analysis(
        self,
        plot_coordinates: List[Tuple[float, float]],
        buffer_distance_meters: Optional[float] = None,
        analysis_layers: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform buffer analysis around plot.

        Args:
            plot_coordinates: Plot boundary coordinates
            buffer_distance_meters: Buffer distance (default from config)
            analysis_layers: Layers to analyze within buffer

        Returns:
            Buffer analysis results
        """
        try:
            buffer_dist = buffer_distance_meters or self.config.buffer_distance_meters
            layers = analysis_layers or ["protected_areas", "water_bodies", "roads"]

            if self._service and hasattr(self._service, "buffer_analysis"):
                return await self._service.buffer_analysis(
                    plot_coordinates=plot_coordinates,
                    buffer_distance_meters=buffer_dist,
                    analysis_layers=layers
                )

            # Fallback
            return {
                "status": "fallback",
                "plot_coordinates": plot_coordinates,
                "buffer_distance_meters": buffer_dist,
                "analysis_layers": layers,
                "buffer_area_hectares": 0.0,
                "features_within_buffer": {},
                "provenance_hash": self._calculate_hash({
                    "coords": plot_coordinates,
                    "buffer": buffer_dist
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Buffer analysis failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def comprehensive_gis_analysis(
        self,
        plot_id: str,
        plot_coordinates: List[Tuple[float, float]],
        country_code: str,
        reference_date: datetime
    ) -> Dict[str, Any]:
        """
        Comprehensive GIS analysis for a plot.

        Performs all available GIS analyses:
        - Boundary validation
        - Land cover classification
        - Protected area screening
        - Indigenous lands overlay
        - Buffer analysis

        Args:
            plot_id: Plot identifier
            plot_coordinates: Plot boundary coordinates
            country_code: Country code
            reference_date: Reference date for analysis

        Returns:
            Complete GIS analysis results
        """
        try:
            analysis_result = {
                "plot_id": plot_id,
                "country_code": country_code,
                "reference_date": reference_date.isoformat()
            }

            # Boundary resolution
            boundary_result = await self.resolve_boundaries(plot_id, plot_coordinates)
            analysis_result["boundaries"] = boundary_result

            # Land cover classification
            land_cover_result = await self.land_cover_classification(
                plot_coordinates, reference_date
            )
            analysis_result["land_cover"] = land_cover_result

            # Protected area screening
            if self.config.enable_protected_areas:
                protected_areas_result = await self.protected_area_overlay(
                    plot_coordinates, include_kba=True, include_unesco=True
                )
                analysis_result["protected_areas"] = protected_areas_result

            # Indigenous lands screening
            if self.config.enable_indigenous_lands:
                indigenous_lands_result = await self.indigenous_lands_overlay(
                    plot_coordinates, country_code
                )
                analysis_result["indigenous_lands"] = indigenous_lands_result

            # Buffer analysis
            buffer_result = await self.buffer_analysis(plot_coordinates)
            analysis_result["buffer_analysis"] = buffer_result

            # Summary
            analysis_result["summary"] = {
                "plot_area_hectares": boundary_result.get("area_hectares", 0.0),
                "protected_area_overlaps": len(
                    analysis_result.get("protected_areas", {}).get("wdpa_overlaps", [])
                ),
                "indigenous_land_overlaps": len(
                    analysis_result.get("indigenous_lands", {}).get("indigenous_lands_overlaps", [])
                ),
                "dominant_land_cover": land_cover_result.get("dominant_class", "unknown")
            }

            analysis_result["provenance_hash"] = self._calculate_hash(analysis_result)
            analysis_result["timestamp"] = datetime.utcnow().isoformat()

            logger.info(f"Comprehensive GIS analysis complete for plot {plot_id}")
            return analysis_result

        except Exception as e:
            logger.error(f"Comprehensive GIS analysis failed: {str(e)}")
            return {
                "plot_id": plot_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
