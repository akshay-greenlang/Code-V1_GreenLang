# -*- coding: utf-8 -*-
"""
Geolocation Formatter Engine - AGENT-EUDR-037

Engine 2 of 7: Formats production plot coordinates per EUDR Article 9
requirements. Handles single-point coordinates for plots under 4 hectares
and polygon boundaries for plots 4 hectares or larger. Ensures WGS84
datum, validates coordinate precision, and generates GeoJSON output.

Algorithm:
    1. Validate raw coordinate input (latitude/longitude bounds)
    2. Determine plot size to select point vs polygon format
    3. Apply precision rounding per configuration
    4. Validate polygon closure (first point == last point)
    5. Generate Article 9 compliant GeoJSON representation
    6. Compute provenance hash for each formatted plot

Zero-Hallucination Guarantees:
    - All coordinate transformations via deterministic arithmetic
    - No LLM involvement in geolocation formatting
    - Precision rounding uses Python Decimal for bit-perfect results
    - Complete provenance trail for every formatted plot

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 (GL-EUDR-DDSC-037)
Regulation: EU 2023/1115 (EUDR) Article 9
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import DDSCreatorConfig, get_config
from .models import GeolocationData, GeolocationMethod
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# Article 9 threshold: plots >= 4 hectares require polygon boundaries
_POLYGON_THRESHOLD_HECTARES = Decimal("4.0")


class GeolocationFormatter:
    """Geolocation formatting engine per EUDR Article 9.

    Formats raw coordinate data into Article 9 compliant geolocation
    records with appropriate precision and polygon/point selection.

    Attributes:
        config: Agent configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> formatter = GeolocationFormatter()
        >>> geo = await formatter.format_geolocation(
        ...     plot_id="PLT-001", latitude=5.123456,
        ...     longitude=-3.456789, area_hectares=2.5,
        ...     country_code="CI",
        ... )
        >>> assert geo.plot_id == "PLT-001"
    """

    def __init__(
        self,
        config: Optional[DDSCreatorConfig] = None,
    ) -> None:
        """Initialize the geolocation formatter engine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._formatted_count = 0
        logger.info("GeolocationFormatter engine initialized")

    async def format_geolocation(
        self,
        plot_id: str,
        latitude: float,
        longitude: float,
        area_hectares: float = 0.0,
        country_code: str = "",
        method: str = "gps_field_survey",
        polygon_coordinates: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> GeolocationData:
        """Format a single geolocation record per Article 9.

        Applies precision rounding, validates bounds, and determines
        whether point or polygon format is required based on area.

        Args:
            plot_id: Unique plot identifier.
            latitude: Plot centroid latitude (WGS84).
            longitude: Plot centroid longitude (WGS84).
            area_hectares: Plot area in hectares.
            country_code: ISO 3166-1 country code.
            method: Geolocation collection method.
            polygon_coordinates: Polygon vertices for plots >= 4ha.
            **kwargs: Additional fields (region, accuracy, etc.).

        Returns:
            Formatted GeolocationData record.

        Raises:
            ValueError: If coordinates are out of bounds.
        """
        # Validate coordinate bounds
        if latitude < -90.0 or latitude > 90.0:
            raise ValueError(f"Latitude {latitude} out of bounds [-90, 90]")
        if longitude < -180.0 or longitude > 180.0:
            raise ValueError(f"Longitude {longitude} out of bounds [-180, 180]")

        precision = self.config.geolocation_precision_digits
        quantize_exp = Decimal(10) ** -precision

        lat_dec = Decimal(str(latitude)).quantize(
            quantize_exp, rounding=ROUND_HALF_UP
        )
        lon_dec = Decimal(str(longitude)).quantize(
            quantize_exp, rounding=ROUND_HALF_UP
        )
        area_dec = Decimal(str(area_hectares)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Format polygon coordinates
        poly: List[List[Decimal]] = []
        if polygon_coordinates:
            max_vertices = self.config.geolocation_max_polygon_vertices
            for coord in polygon_coordinates[:max_vertices]:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    poly.append([
                        Decimal(str(coord[0])).quantize(
                            quantize_exp, rounding=ROUND_HALF_UP
                        ),
                        Decimal(str(coord[1])).quantize(
                            quantize_exp, rounding=ROUND_HALF_UP
                        ),
                    ])

            # Ensure polygon closure
            if len(poly) >= 3 and poly[0] != poly[-1]:
                poly.append(list(poly[0]))

        # Parse collection method
        try:
            geo_method = GeolocationMethod(method)
        except ValueError:
            geo_method = GeolocationMethod.GPS_FIELD_SURVEY

        geo = GeolocationData(
            plot_id=plot_id,
            latitude=lat_dec,
            longitude=lon_dec,
            area_hectares=area_dec,
            polygon_coordinates=poly,
            country_code=country_code,
            region=kwargs.get("region", ""),
            collection_method=geo_method,
            collection_date=kwargs.get("collection_date"),
            accuracy_meters=Decimal(str(kwargs.get("accuracy_meters", 0))),
            verified=kwargs.get("verified", False),
            verification_source=kwargs.get("verification_source", ""),
            provenance_hash=self._provenance.compute_hash({
                "plot_id": plot_id,
                "latitude": str(lat_dec),
                "longitude": str(lon_dec),
                "area_hectares": str(area_dec),
                "country_code": country_code,
            }),
        )

        self._formatted_count += 1
        return geo

    async def format_batch(
        self,
        plots: List[Dict[str, Any]],
    ) -> List[GeolocationData]:
        """Format a batch of geolocation records.

        Args:
            plots: List of plot dictionaries.

        Returns:
            List of formatted GeolocationData records.
        """
        start = time.monotonic()
        results: List[GeolocationData] = []
        max_plots = self.config.dds_max_plots_per_commodity

        for plot in plots[:max_plots]:
            geo = await self.format_geolocation(**plot)
            results.append(geo)

        elapsed = time.monotonic() - start
        logger.info(
            "Formatted %d geolocation plots in %.1fms",
            len(results), elapsed * 1000,
        )

        return results

    async def validate_geolocation(
        self,
        geo: GeolocationData,
    ) -> Dict[str, Any]:
        """Validate a formatted geolocation record.

        Args:
            geo: Formatted geolocation data.

        Returns:
            Validation result dictionary.
        """
        issues: List[str] = []

        if geo.latitude < Decimal("-90") or geo.latitude > Decimal("90"):
            issues.append(f"Latitude {geo.latitude} out of bounds")
        if geo.longitude < Decimal("-180") or geo.longitude > Decimal("180"):
            issues.append(f"Longitude {geo.longitude} out of bounds")

        if geo.area_hectares >= _POLYGON_THRESHOLD_HECTARES:
            if not geo.polygon_coordinates:
                issues.append(
                    "Plots >= 4ha require polygon boundaries (Article 9)"
                )
            elif len(geo.polygon_coordinates) < 4:
                issues.append(
                    "Polygon must have at least 4 vertices (3 + closure)"
                )

        if not geo.country_code:
            issues.append("Country code is required")

        # Check precision
        lat_str = str(geo.latitude)
        if "." in lat_str:
            decimals = len(lat_str.split(".")[1])
        else:
            decimals = 0
        if decimals < self.config.geolocation_precision_digits:
            issues.append(
                f"Latitude precision {decimals} digits below required "
                f"{self.config.geolocation_precision_digits}"
            )

        return {
            "plot_id": geo.plot_id,
            "valid": len(issues) == 0,
            "issues": issues,
        }

    async def generate_geojson(
        self,
        geolocations: List[GeolocationData],
    ) -> Dict[str, Any]:
        """Generate GeoJSON FeatureCollection from geolocation data.

        Args:
            geolocations: List of formatted geolocation records.

        Returns:
            GeoJSON FeatureCollection dictionary.
        """
        features: List[Dict[str, Any]] = []

        for geo in geolocations:
            if geo.polygon_coordinates and len(geo.polygon_coordinates) >= 3:
                geometry: Dict[str, Any] = {
                    "type": "Polygon",
                    "coordinates": [
                        [[float(c[1]), float(c[0])]
                         for c in geo.polygon_coordinates]
                    ],
                }
            else:
                geometry = {
                    "type": "Point",
                    "coordinates": [
                        float(geo.longitude), float(geo.latitude)
                    ],
                }

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "plot_id": geo.plot_id,
                    "area_hectares": float(geo.area_hectares),
                    "country_code": geo.country_code,
                    "collection_method": geo.collection_method.value,
                    "verified": geo.verified,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Health check dictionary.
        """
        return {
            "engine": "GeolocationFormatter",
            "status": "healthy",
            "plots_formatted": self._formatted_count,
            "precision_digits": self.config.geolocation_precision_digits,
        }
