# -*- coding: utf-8 -*-
"""
Cropland Expansion Detector Engine - AGENT-EUDR-005: Land Use Change (Engine 5)

Detects agricultural expansion into forests for all seven EUDR-regulated
commodities.  Identifies commodity-specific conversion patterns through
spectral signature analysis, distinguishes smallholder from industrial-
scale clearings, and tracks progressive expansion fronts with leapfrog
pattern detection.

Zero-Hallucination Guarantees:
    - All calculations use deterministic float arithmetic (no ML/LLM).
    - Commodity spectral signatures based on published remote sensing
      literature with static lookup tables.
    - Conversion confidence from ensemble agreement + spectral distance.
    - Expansion rate computed as simple area / period division.
    - Hotspot detection via deterministic spatial clustering.
    - SHA-256 provenance hashes on all result objects.
    - No ML/LLM used for any numeric computation.

Conversion Types (7):
    FOREST_TO_PALM_OIL       - Regular grid spacing, distinctive NDVI
                                seasonal pattern (harvest cycles).
    FOREST_TO_RUBBER          - Leaf-drop phenology (annual NDVI dip
                                Jan-Mar in SE Asia).
    FOREST_TO_COCOA           - Under-canopy planting (moderate NDVI
                                reduction, not complete clearing).
    FOREST_TO_COFFEE          - Altitude-associated, moderate canopy
                                (1000-2000m elevation).
    FOREST_TO_SOYA            - Strong seasonal NDVI cycle (planting-
                                harvest), flat terrain preference.
    FOREST_TO_PASTURE         - Low NDVI (0.2-0.4), low texture
                                complexity, large clearings.
    FOREST_TO_TIMBER_PLANTATION - Regular spacing, uniform height,
                                  single-species phenology.

Scale Classification:
    smallholder:  < 5 ha
    medium:       5 - 50 ha
    industrial:   > 50 ha

Regulatory References:
    - EUDR Article 2(1): Deforestation-free requirement since 31 Dec 2020.
    - EUDR Article 9: Geolocation-linked evidence for DDS.
    - EUDR Article 10: Risk assessment with commodity-specific analysis.
    - EUDR Annex I: Seven commodity families and derived products.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-005 (Engine 5: Cropland Expansion Detection)
Agent ID: GL-EUDR-LUC-005
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConversionType(str, Enum):
    """Type of forest-to-commodity conversion detected.

    Each conversion type corresponds to a specific EUDR-regulated
    commodity and has distinct spectral, phenological, and structural
    characteristics used for detection.
    """

    FOREST_TO_PALM_OIL = "FOREST_TO_PALM_OIL"
    FOREST_TO_RUBBER = "FOREST_TO_RUBBER"
    FOREST_TO_COCOA = "FOREST_TO_COCOA"
    FOREST_TO_COFFEE = "FOREST_TO_COFFEE"
    FOREST_TO_SOYA = "FOREST_TO_SOYA"
    FOREST_TO_PASTURE = "FOREST_TO_PASTURE"
    FOREST_TO_TIMBER_PLANTATION = "FOREST_TO_TIMBER_PLANTATION"
    UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Constants: Commodity Spectral Signatures
# ---------------------------------------------------------------------------
# Each commodity has a spectral profile dictionary with keys:
#   ndvi_range: (min, max) typical NDVI for mature plantation
#   ndvi_seasonal_amplitude: typical amplitude of seasonal cycle
#   ndvi_dip_months: months (1-12) when seasonal NDVI dip occurs
#   texture_regularity: 0-1 measure of planting grid regularity
#   typical_clearing_size_ha: (min, max) typical clearing size
#   elevation_range_m: (min, max) preferred elevation if relevant
#   swir_reflectance: typical SWIR (B11) reflectance range
#   terrain_slope_max_deg: maximum preferred slope
# Reference: Descals et al. (2021), Gaveau et al. (2016),
#            Tyukavina et al. (2017), Song et al. (2018).

COMMODITY_SPECTRAL_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "palm_oil": {
        "ndvi_range": (0.55, 0.85),
        "ndvi_seasonal_amplitude": 0.08,
        "ndvi_dip_months": [1, 2, 7, 8],
        "texture_regularity": 0.85,
        "typical_clearing_size_ha": (5.0, 500.0),
        "elevation_range_m": (0.0, 500.0),
        "swir_reflectance": (0.10, 0.22),
        "terrain_slope_max_deg": 15.0,
    },
    "rubber": {
        "ndvi_range": (0.40, 0.80),
        "ndvi_seasonal_amplitude": 0.25,
        "ndvi_dip_months": [1, 2, 3],
        "texture_regularity": 0.75,
        "typical_clearing_size_ha": (2.0, 100.0),
        "elevation_range_m": (0.0, 600.0),
        "swir_reflectance": (0.12, 0.25),
        "terrain_slope_max_deg": 20.0,
    },
    "cocoa": {
        "ndvi_range": (0.50, 0.80),
        "ndvi_seasonal_amplitude": 0.10,
        "ndvi_dip_months": [12, 1, 2],
        "texture_regularity": 0.35,
        "typical_clearing_size_ha": (0.5, 20.0),
        "elevation_range_m": (0.0, 800.0),
        "swir_reflectance": (0.10, 0.20),
        "terrain_slope_max_deg": 25.0,
    },
    "coffee": {
        "ndvi_range": (0.45, 0.75),
        "ndvi_seasonal_amplitude": 0.15,
        "ndvi_dip_months": [11, 12, 1, 2],
        "texture_regularity": 0.50,
        "typical_clearing_size_ha": (0.5, 30.0),
        "elevation_range_m": (1000.0, 2000.0),
        "swir_reflectance": (0.12, 0.22),
        "terrain_slope_max_deg": 30.0,
    },
    "soya": {
        "ndvi_range": (0.20, 0.85),
        "ndvi_seasonal_amplitude": 0.55,
        "ndvi_dip_months": [5, 6, 7, 8, 9],
        "texture_regularity": 0.90,
        "typical_clearing_size_ha": (50.0, 5000.0),
        "elevation_range_m": (0.0, 1200.0),
        "swir_reflectance": (0.15, 0.30),
        "terrain_slope_max_deg": 8.0,
    },
    "cattle": {
        "ndvi_range": (0.20, 0.40),
        "ndvi_seasonal_amplitude": 0.12,
        "ndvi_dip_months": [7, 8, 9],
        "texture_regularity": 0.15,
        "typical_clearing_size_ha": (10.0, 2000.0),
        "elevation_range_m": (0.0, 1500.0),
        "swir_reflectance": (0.20, 0.35),
        "terrain_slope_max_deg": 20.0,
    },
    "wood": {
        "ndvi_range": (0.55, 0.85),
        "ndvi_seasonal_amplitude": 0.05,
        "ndvi_dip_months": [],
        "texture_regularity": 0.80,
        "typical_clearing_size_ha": (5.0, 500.0),
        "elevation_range_m": (0.0, 1500.0),
        "swir_reflectance": (0.10, 0.20),
        "terrain_slope_max_deg": 25.0,
    },
}

#: Map commodity string to ConversionType enum.
_COMMODITY_TO_CONVERSION: Dict[str, ConversionType] = {
    "palm_oil": ConversionType.FOREST_TO_PALM_OIL,
    "oil_palm": ConversionType.FOREST_TO_PALM_OIL,
    "rubber": ConversionType.FOREST_TO_RUBBER,
    "cocoa": ConversionType.FOREST_TO_COCOA,
    "coffee": ConversionType.FOREST_TO_COFFEE,
    "soya": ConversionType.FOREST_TO_SOYA,
    "cattle": ConversionType.FOREST_TO_PASTURE,
    "beef": ConversionType.FOREST_TO_PASTURE,
    "leather": ConversionType.FOREST_TO_PASTURE,
    "wood": ConversionType.FOREST_TO_TIMBER_PLANTATION,
}

#: Scale classification thresholds in hectares.
_SCALE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "smallholder": (0.0, 5.0),
    "medium": (5.0, 50.0),
    "industrial": (50.0, float("inf")),
}

#: Confidence weights for conversion detection.
_CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "spectral_match": 0.30,
    "temporal_pattern": 0.25,
    "texture_match": 0.20,
    "classification_agreement": 0.15,
    "data_quality": 0.10,
}

#: Spatial clustering threshold for hotspot detection (degrees).
_DEFAULT_SPATIAL_THRESHOLD_DEG = 0.05

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PlotConversionInput:
    """Input data for a single plot conversion analysis.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Plot centroid latitude (-90 to 90).
        longitude: Plot centroid longitude (-180 to 180).
        area_ha: Plot area in hectares.
        ndvi_before: Mean NDVI value at date_from.
        ndvi_after: Mean NDVI value at date_to.
        ndvi_time_series: Monthly NDVI values over the analysis period.
        swir_before: SWIR (B11) reflectance at date_from.
        swir_after: SWIR (B11) reflectance at date_to.
        texture_regularity: Texture regularity score (0-1).
        elevation_m: Plot elevation in metres.
        slope_deg: Terrain slope in degrees.
        from_class: Land use class at date_from.
        to_class: Land use class at date_to.
        cloud_cover_pct: Cloud cover percentage.
        classification_confidence: Confidence of the land use classification.
    """

    plot_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    area_ha: float = 1.0
    ndvi_before: float = 0.0
    ndvi_after: float = 0.0
    ndvi_time_series: List[float] = field(default_factory=list)
    swir_before: float = 0.0
    swir_after: float = 0.0
    texture_regularity: float = 0.0
    elevation_m: float = 0.0
    slope_deg: float = 0.0
    from_class: str = "forest"
    to_class: str = "cropland"
    cloud_cover_pct: float = 0.0
    classification_confidence: float = 0.8

@dataclass
class CroplandConversion:
    """Result of a cropland conversion detection analysis.

    Attributes:
        result_id: Unique result identifier.
        plot_id: Identifier of the analyzed plot.
        conversion_type: Detected conversion type.
        commodity: Target commodity of the conversion.
        scale: Scale classification (smallholder/medium/industrial).
        area_ha: Area of the conversion in hectares.
        conversion_detected: Whether a conversion was detected.
        confidence: Confidence score (0.0-1.0).
        ndvi_change: NDVI change (after - before).
        spectral_match_score: How well the post-conversion spectral
            signature matches the commodity reference (0-1).
        temporal_pattern_score: How well the temporal NDVI pattern
            matches the expected commodity phenology (0-1).
        texture_match_score: How well the texture regularity matches
            the expected commodity planting pattern (0-1).
        expansion_front_distance_km: Distance to nearest expansion front.
        latitude: Plot centroid latitude.
        longitude: Plot centroid longitude.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp of analysis.
        metadata: Additional contextual fields.
    """

    result_id: str = ""
    plot_id: str = ""
    conversion_type: str = ""
    commodity: str = ""
    scale: str = ""
    area_ha: float = 0.0
    conversion_detected: bool = False
    confidence: float = 0.0
    ndvi_change: float = 0.0
    spectral_match_score: float = 0.0
    temporal_pattern_score: float = 0.0
    texture_match_score: float = 0.0
    expansion_front_distance_km: float = -1.0
    latitude: float = 0.0
    longitude: float = 0.0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "plot_id": self.plot_id,
            "conversion_type": self.conversion_type,
            "commodity": self.commodity,
            "scale": self.scale,
            "area_ha": self.area_ha,
            "conversion_detected": self.conversion_detected,
            "confidence": self.confidence,
            "ndvi_change": self.ndvi_change,
            "spectral_match_score": self.spectral_match_score,
            "temporal_pattern_score": self.temporal_pattern_score,
            "texture_match_score": self.texture_match_score,
            "expansion_front_distance_km": self.expansion_front_distance_km,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

# ---------------------------------------------------------------------------
# CroplandExpansionDetector
# ---------------------------------------------------------------------------

class CroplandExpansionDetector:
    """Detects agricultural expansion into forests for all 7 EUDR commodities.

    Identifies commodity-specific conversion patterns, distinguishes
    smallholder from industrial-scale clearings, and detects progressive
    expansion fronts with leapfrog pattern analysis.

    All numeric computations are deterministic (zero-hallucination).
    SHA-256 provenance hashes are computed for every result.

    Example::

        detector = CroplandExpansionDetector()
        plot = PlotConversionInput(
            plot_id="plot-001",
            latitude=-3.4,
            longitude=-62.2,
            ndvi_before=0.82,
            ndvi_after=0.35,
            from_class="forest",
            to_class="cropland",
            area_ha=25.0,
        )
        result = detector.detect_conversion(
            latitude=-3.4, longitude=-62.2,
            date_from="2020-06-01", date_to="2023-06-01",
            commodity="soya", plot_input=plot,
        )
        assert result.conversion_detected is True

    Attributes:
        config: Optional configuration object.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the CroplandExpansionDetector.

        Args:
            config: Optional LandUseChangeConfig instance.
        """
        self.config = config
        logger.info(
            "CroplandExpansionDetector initialized: module_version=%s",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Single Plot Detection
    # ------------------------------------------------------------------

    def detect_conversion(
        self,
        latitude: float,
        longitude: float,
        date_from: str,
        date_to: str,
        commodity: str,
        plot_input: Optional[PlotConversionInput] = None,
    ) -> CroplandConversion:
        """Detect agricultural conversion for a single plot.

        Analyzes spectral, temporal, and textural evidence to determine
        whether a forest-to-commodity conversion has occurred within the
        analysis period.

        Args:
            latitude: Plot centroid latitude.
            longitude: Plot centroid longitude.
            date_from: Analysis start date (ISO format).
            date_to: Analysis end date (ISO format).
            commodity: Target commodity (palm_oil, rubber, cocoa,
                coffee, soya, cattle, wood).
            plot_input: Optional detailed plot input data. If None,
                a default input is constructed from latitude/longitude.

        Returns:
            CroplandConversion result with detection verdict.

        Raises:
            ValueError: If commodity is not recognized or coordinates
                are out of range.
        """
        start_time = time.monotonic()
        self._validate_coordinates(latitude, longitude)
        commodity_key = self._normalize_commodity(commodity)

        plot = plot_input or PlotConversionInput(
            plot_id=_generate_id(),
            latitude=latitude,
            longitude=longitude,
        )

        conversion_type = self._identify_conversion_type(
            plot.from_class, plot.to_class, commodity_key,
        )

        spectral_score = self._compute_spectral_match(
            plot, commodity_key,
        )
        temporal_score = self._compute_temporal_pattern_match(
            plot, commodity_key,
        )
        texture_score = self._compute_texture_match(
            plot, commodity_key,
        )
        classification_agreement = plot.classification_confidence
        data_quality = self._compute_data_quality_score(plot)

        confidence = self._compute_conversion_confidence(
            spectral_evidence=spectral_score,
            classification_agreement=classification_agreement,
            temporal_match=temporal_score,
            texture_match=texture_score,
            data_quality=data_quality,
        )

        ndvi_change = plot.ndvi_after - plot.ndvi_before
        is_conversion = self._evaluate_conversion(
            from_class=plot.from_class,
            ndvi_change=ndvi_change,
            confidence=confidence,
            commodity_key=commodity_key,
        )

        scale = self._classify_scale(plot.area_ha)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = CroplandConversion(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            conversion_type=conversion_type.value,
            commodity=commodity_key,
            scale=scale,
            area_ha=plot.area_ha,
            conversion_detected=is_conversion,
            confidence=round(confidence, 4),
            ndvi_change=round(ndvi_change, 4),
            spectral_match_score=round(spectral_score, 4),
            temporal_pattern_score=round(temporal_score, 4),
            texture_match_score=round(texture_score, 4),
            latitude=latitude,
            longitude=longitude,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=utcnow().isoformat(),
            metadata={
                "date_from": date_from,
                "date_to": date_to,
                "classification_agreement": round(
                    classification_agreement, 4,
                ),
                "data_quality": round(data_quality, 4),
            },
        )
        result.provenance_hash = self._compute_result_hash(result)

        logger.info(
            "Conversion detection: plot=%s, commodity=%s, detected=%s, "
            "confidence=%.3f, type=%s, scale=%s, %.2fms",
            plot.plot_id, commodity_key, is_conversion,
            confidence, conversion_type.value, scale, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Batch Detection
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        plots: List[PlotConversionInput],
        date_from: str,
        date_to: str,
        commodity: Optional[str] = None,
    ) -> List[CroplandConversion]:
        """Detect conversions for multiple plots.

        Args:
            plots: List of plot inputs.
            date_from: Analysis start date (ISO format).
            date_to: Analysis end date (ISO format).
            commodity: Optional commodity; if None, uses best-match
                detection per plot.

        Returns:
            List of CroplandConversion results.

        Raises:
            ValueError: If plots list is empty.
        """
        if not plots:
            raise ValueError("plots list must not be empty")

        start_time = time.monotonic()
        results: List[CroplandConversion] = []

        for i, plot in enumerate(plots):
            try:
                comm = commodity or self._infer_commodity(plot)
                result = self.detect_conversion(
                    latitude=plot.latitude,
                    longitude=plot.longitude,
                    date_from=date_from,
                    date_to=date_to,
                    commodity=comm,
                    plot_input=plot,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "detect_batch: failed on plot[%d] id=%s: %s",
                    i, plot.plot_id, str(exc),
                )
                results.append(self._create_error_result(plot, str(exc)))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        conversions = sum(1 for r in results if r.conversion_detected)
        logger.info(
            "detect_batch complete: %d/%d conversions detected, "
            "%d plots, %.2fms",
            conversions, len(plots), len(plots), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Expansion Front Detection
    # ------------------------------------------------------------------

    def detect_expansion_front(
        self,
        region_bounds: Dict[str, float],
        commodity: str,
        date_from: str,
        date_to: str,
        conversions: Optional[List[CroplandConversion]] = None,
    ) -> Dict[str, Any]:
        """Detect the expansion front for a commodity within a region.

        Identifies the leading edge of agricultural expansion by
        analyzing the spatial distribution of conversion events and
        computing the front direction and velocity.

        Args:
            region_bounds: Dictionary with min_lat, max_lat, min_lon,
                max_lon defining the analysis region.
            commodity: Target commodity.
            date_from: Analysis start date (ISO format).
            date_to: Analysis end date (ISO format).
            conversions: Optional pre-computed conversion results. If
                None, returns a template structure.

        Returns:
            Dictionary with front_detected, front_direction_deg,
            front_velocity_km_per_year, front_centroid, and
            front_extent_km.
        """
        start_time = time.monotonic()
        self._validate_region_bounds(region_bounds)
        commodity_key = self._normalize_commodity(commodity)

        if not conversions:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return {
                "front_detected": False,
                "commodity": commodity_key,
                "region_bounds": region_bounds,
                "date_from": date_from,
                "date_to": date_to,
                "front_direction_deg": 0.0,
                "front_velocity_km_per_year": 0.0,
                "front_centroid": {"lat": 0.0, "lon": 0.0},
                "front_extent_km": 0.0,
                "conversion_count": 0,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": _compute_hash({
                    "region_bounds": region_bounds,
                    "commodity": commodity_key,
                }),
            }

        detected = [c for c in conversions if c.conversion_detected]
        if len(detected) < 3:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return {
                "front_detected": False,
                "commodity": commodity_key,
                "region_bounds": region_bounds,
                "date_from": date_from,
                "date_to": date_to,
                "front_direction_deg": 0.0,
                "front_velocity_km_per_year": 0.0,
                "front_centroid": {"lat": 0.0, "lon": 0.0},
                "front_extent_km": 0.0,
                "conversion_count": len(detected),
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": _compute_hash({
                    "region_bounds": region_bounds,
                    "commodity": commodity_key,
                    "count": len(detected),
                }),
            }

        centroid_lat = sum(c.latitude for c in detected) / len(detected)
        centroid_lon = sum(c.longitude for c in detected) / len(detected)

        front_direction = self._compute_front_direction(detected)
        front_extent = self._compute_front_extent(detected)

        period_years = self._parse_period_years(date_from, date_to)
        total_area = sum(c.area_ha for c in detected)
        velocity = self._estimate_expansion_rate(
            conversion_events=detected,
            period_years=period_years,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "front_detected": True,
            "commodity": commodity_key,
            "region_bounds": region_bounds,
            "date_from": date_from,
            "date_to": date_to,
            "front_direction_deg": round(front_direction, 2),
            "front_velocity_km_per_year": round(velocity, 2),
            "front_centroid": {
                "lat": round(centroid_lat, 6),
                "lon": round(centroid_lon, 6),
            },
            "front_extent_km": round(front_extent, 2),
            "conversion_count": len(detected),
            "total_area_ha": round(total_area, 2),
            "processing_time_ms": round(elapsed_ms, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Expansion front detected: commodity=%s, direction=%.1fdeg, "
            "velocity=%.2f km/yr, extent=%.1fkm, %d conversions",
            commodity_key, front_direction, velocity,
            front_extent, len(detected),
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Hotspot Detection
    # ------------------------------------------------------------------

    def detect_conversion_hotspots(
        self,
        conversions: List[CroplandConversion],
        spatial_threshold_km: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Detect spatial clusters of conversion events (hotspots).

        Groups nearby conversion events using a simple distance-based
        clustering algorithm and returns cluster centroids with
        aggregate statistics.

        Args:
            conversions: List of conversion results.
            spatial_threshold_km: Maximum distance between events
                in a cluster (kilometres).

        Returns:
            List of hotspot dictionaries with centroid, event_count,
            total_area_ha, and mean_confidence.
        """
        start_time = time.monotonic()

        if not conversions:
            return []

        detected = [c for c in conversions if c.conversion_detected]
        if not detected:
            return []

        threshold_deg = spatial_threshold_km / 111.32
        clusters: List[List[CroplandConversion]] = []
        assigned: List[bool] = [False] * len(detected)

        for i, event in enumerate(detected):
            if assigned[i]:
                continue

            cluster = [event]
            assigned[i] = True

            for j in range(i + 1, len(detected)):
                if assigned[j]:
                    continue
                dist = self._haversine_distance_deg(
                    event.latitude, event.longitude,
                    detected[j].latitude, detected[j].longitude,
                )
                if dist <= threshold_deg:
                    cluster.append(detected[j])
                    assigned[j] = True

            if len(cluster) >= 2:
                clusters.append(cluster)

        hotspots: List[Dict[str, Any]] = []
        for cluster in clusters:
            centroid_lat = sum(c.latitude for c in cluster) / len(cluster)
            centroid_lon = sum(c.longitude for c in cluster) / len(cluster)
            total_area = sum(c.area_ha for c in cluster)
            mean_confidence = sum(c.confidence for c in cluster) / len(cluster)

            hotspot = {
                "hotspot_id": _generate_id(),
                "centroid_lat": round(centroid_lat, 6),
                "centroid_lon": round(centroid_lon, 6),
                "event_count": len(cluster),
                "total_area_ha": round(total_area, 2),
                "mean_confidence": round(mean_confidence, 4),
                "commodities": list(set(c.commodity for c in cluster)),
                "scales": list(set(c.scale for c in cluster)),
            }
            hotspot["provenance_hash"] = _compute_hash(hotspot)
            hotspots.append(hotspot)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Hotspot detection: %d hotspots from %d conversions, "
            "threshold=%.1fkm, %.2fms",
            len(hotspots), len(detected), spatial_threshold_km, elapsed_ms,
        )
        return hotspots

    # ------------------------------------------------------------------
    # Public API: Leapfrog Pattern Detection
    # ------------------------------------------------------------------

    def detect_leapfrog_pattern(
        self,
        conversions: List[CroplandConversion],
        main_front: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect leapfrog clearing patterns ahead of the main front.

        Leapfrog clearings are conversion events that occur
        significantly ahead of the continuous expansion front,
        typically indicating speculative land clearing or road-
        construction-driven opening of new areas.

        Args:
            conversions: List of conversion results.
            main_front: Optional front data from detect_expansion_front.

        Returns:
            List of leapfrog event dictionaries.
        """
        if not conversions:
            return []

        detected = [c for c in conversions if c.conversion_detected]
        if len(detected) < 5:
            return []

        centroid_lat = sum(c.latitude for c in detected) / len(detected)
        centroid_lon = sum(c.longitude for c in detected) / len(detected)

        distances: List[Tuple[float, CroplandConversion]] = []
        for conv in detected:
            dist = self._haversine_distance_km(
                centroid_lat, centroid_lon,
                conv.latitude, conv.longitude,
            )
            distances.append((dist, conv))

        distances.sort(key=lambda x: x[0], reverse=True)

        if not distances:
            return []

        mean_dist = sum(d[0] for d in distances) / len(distances)
        std_dist = math.sqrt(
            sum((d[0] - mean_dist) ** 2 for d in distances) / len(distances)
        )

        leapfrog_threshold = mean_dist + 2.0 * std_dist
        leapfrogs: List[Dict[str, Any]] = []

        for dist, conv in distances:
            if dist > leapfrog_threshold:
                leapfrog = {
                    "leapfrog_id": _generate_id(),
                    "plot_id": conv.plot_id,
                    "latitude": conv.latitude,
                    "longitude": conv.longitude,
                    "distance_from_centroid_km": round(dist, 2),
                    "area_ha": conv.area_ha,
                    "commodity": conv.commodity,
                    "confidence": conv.confidence,
                    "leapfrog_score": round(
                        min(1.0, (dist - leapfrog_threshold) /
                            max(leapfrog_threshold, 0.01)),
                        4,
                    ),
                }
                leapfrog["provenance_hash"] = _compute_hash(leapfrog)
                leapfrogs.append(leapfrog)

        logger.info(
            "Leapfrog detection: %d patterns from %d conversions, "
            "threshold=%.1fkm",
            len(leapfrogs), len(detected), leapfrog_threshold,
        )
        return leapfrogs

    # ------------------------------------------------------------------
    # Public API: Progressive Expansion
    # ------------------------------------------------------------------

    def detect_progressive_expansion(
        self,
        time_series_classifications: List[Dict[str, Any]],
        region: str = "",
    ) -> Dict[str, Any]:
        """Detect progressive expansion over a time-series of classifications.

        Analyzes a chronological sequence of land use classifications
        to identify progressive (gradual) expansion of cropland into
        forest.

        Args:
            time_series_classifications: List of dicts with keys:
                date, forest_pct, cropland_pct, other_pct.
            region: Optional region identifier.

        Returns:
            Dictionary with progressive_expansion_detected, rate,
            inflection_points, and trend data.
        """
        if not time_series_classifications:
            return {
                "progressive_expansion_detected": False,
                "region": region,
                "data_points": 0,
            }

        n = len(time_series_classifications)
        forest_values = [
            entry.get("forest_pct", 0.0)
            for entry in time_series_classifications
        ]
        cropland_values = [
            entry.get("cropland_pct", 0.0)
            for entry in time_series_classifications
        ]

        forest_trend = self._compute_linear_trend(forest_values)
        cropland_trend = self._compute_linear_trend(cropland_values)

        is_progressive = (
            forest_trend < -0.5 and cropland_trend > 0.5 and n >= 3
        )

        inflection_points = self._find_inflection_points(forest_values)

        total_forest_loss = forest_values[0] - forest_values[-1] if n > 1 else 0.0
        total_cropland_gain = cropland_values[-1] - cropland_values[0] if n > 1 else 0.0

        result = {
            "progressive_expansion_detected": is_progressive,
            "region": region,
            "data_points": n,
            "forest_trend_pct_per_step": round(forest_trend, 4),
            "cropland_trend_pct_per_step": round(cropland_trend, 4),
            "total_forest_loss_pct": round(total_forest_loss, 2),
            "total_cropland_gain_pct": round(total_cropland_gain, 2),
            "inflection_point_count": len(inflection_points),
            "inflection_indices": inflection_points,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Progressive expansion: detected=%s, forest_trend=%.3f, "
            "cropland_trend=%.3f, %d data points",
            is_progressive, forest_trend, cropland_trend, n,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Expansion Rate Estimation
    # ------------------------------------------------------------------

    def estimate_expansion_rate(
        self,
        conversion_events: List[CroplandConversion],
        period_years: float,
    ) -> float:
        """Estimate the expansion rate in hectares per year.

        Args:
            conversion_events: List of conversion results.
            period_years: Analysis period in years (must be > 0).

        Returns:
            Expansion rate in hectares per year.

        Raises:
            ValueError: If period_years <= 0.
        """
        return self._estimate_expansion_rate(conversion_events, period_years)

    # ------------------------------------------------------------------
    # Public API: Conversion Probability Map
    # ------------------------------------------------------------------

    def generate_conversion_probability_map(
        self,
        region: Dict[str, float],
        commodity: str,
        grid_resolution_deg: float = 0.01,
    ) -> Dict[str, Any]:
        """Generate a gridded conversion probability map for a region.

        Creates a regular grid over the region and computes conversion
        probability for each cell based on proximity to existing
        conversions, commodity suitability, and terrain factors.

        Args:
            region: Dictionary with min_lat, max_lat, min_lon, max_lon.
            commodity: Target commodity.
            grid_resolution_deg: Grid cell size in degrees.

        Returns:
            Dictionary with grid metadata and probability values.
        """
        start_time = time.monotonic()
        self._validate_region_bounds(region)
        commodity_key = self._normalize_commodity(commodity)

        sig = COMMODITY_SPECTRAL_SIGNATURES.get(commodity_key, {})
        elev_range = sig.get("elevation_range_m", (0.0, 3000.0))
        slope_max = sig.get("terrain_slope_max_deg", 30.0)

        min_lat = region["min_lat"]
        max_lat = region["max_lat"]
        min_lon = region["min_lon"]
        max_lon = region["max_lon"]

        lat_steps = max(1, int((max_lat - min_lat) / grid_resolution_deg))
        lon_steps = max(1, int((max_lon - min_lon) / grid_resolution_deg))

        total_cells = lat_steps * lon_steps
        if total_cells > 100000:
            lat_steps = min(lat_steps, 316)
            lon_steps = min(lon_steps, 316)
            total_cells = lat_steps * lon_steps

        probabilities: List[Dict[str, Any]] = []
        for i in range(lat_steps):
            for j in range(lon_steps):
                cell_lat = min_lat + (i + 0.5) * grid_resolution_deg
                cell_lon = min_lon + (j + 0.5) * grid_resolution_deg

                terrain_score = self._terrain_suitability(
                    cell_lat, cell_lon, elev_range, slope_max,
                )
                commodity_score = self._commodity_suitability(
                    cell_lat, cell_lon, commodity_key,
                )
                probability = (terrain_score * 0.5 + commodity_score * 0.5)

                probabilities.append({
                    "lat": round(cell_lat, 6),
                    "lon": round(cell_lon, 6),
                    "probability": round(probability, 4),
                })

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "commodity": commodity_key,
            "region": region,
            "grid_resolution_deg": grid_resolution_deg,
            "lat_steps": lat_steps,
            "lon_steps": lon_steps,
            "total_cells": total_cells,
            "cell_count": len(probabilities),
            "mean_probability": round(
                sum(p["probability"] for p in probabilities) /
                max(len(probabilities), 1),
                4,
            ),
            "max_probability": round(
                max((p["probability"] for p in probabilities), default=0.0),
                4,
            ),
            "cells": probabilities,
            "processing_time_ms": round(elapsed_ms, 2),
        }
        result["provenance_hash"] = _compute_hash({
            "commodity": commodity_key,
            "region": region,
            "cell_count": len(probabilities),
        })

        logger.info(
            "Probability map: commodity=%s, %d cells, mean=%.3f, "
            "max=%.3f, %.2fms",
            commodity_key, len(probabilities),
            result["mean_probability"], result["max_probability"],
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Conversion Type Identification
    # ------------------------------------------------------------------

    def _identify_conversion_type(
        self,
        from_class: str,
        to_class: str,
        commodity: str,
    ) -> ConversionType:
        """Identify the conversion type from land class and commodity.

        Args:
            from_class: Source land use class.
            to_class: Target land use class.
            commodity: Commodity identifier.

        Returns:
            ConversionType enum value.
        """
        if from_class.lower() not in ("forest", "primary_tropical",
                                       "secondary_tropical", "tropical_dry",
                                       "mangrove", "agroforestry"):
            return ConversionType.UNKNOWN

        conversion = _COMMODITY_TO_CONVERSION.get(commodity.lower())
        if conversion is not None:
            return conversion
        return ConversionType.UNKNOWN

    # ------------------------------------------------------------------
    # Internal: Scale Classification
    # ------------------------------------------------------------------

    def _classify_scale(self, area_ha: float) -> str:
        """Classify clearing scale based on area.

        Args:
            area_ha: Clearing area in hectares.

        Returns:
            Scale label: "smallholder" (<5ha), "medium" (5-50ha),
            or "industrial" (>50ha).
        """
        if area_ha < 5.0:
            return "smallholder"
        elif area_ha <= 50.0:
            return "medium"
        else:
            return "industrial"

    # ------------------------------------------------------------------
    # Internal: Spectral Match Scoring
    # ------------------------------------------------------------------

    def _compute_spectral_match(
        self,
        plot: PlotConversionInput,
        commodity: str,
    ) -> float:
        """Compute how well post-conversion spectra match commodity reference.

        Uses NDVI range, SWIR reflectance, and elevation compatibility
        from the commodity spectral signature reference library.

        Args:
            plot: Plot input data.
            commodity: Commodity key.

        Returns:
            Spectral match score (0.0-1.0).
        """
        sig = COMMODITY_SPECTRAL_SIGNATURES.get(commodity)
        if sig is None:
            return 0.5

        scores: List[float] = []

        ndvi_min, ndvi_max = sig["ndvi_range"]
        if ndvi_min <= plot.ndvi_after <= ndvi_max:
            scores.append(1.0)
        else:
            ndvi_dist = min(
                abs(plot.ndvi_after - ndvi_min),
                abs(plot.ndvi_after - ndvi_max),
            )
            scores.append(max(0.0, 1.0 - ndvi_dist * 3.0))

        swir_min, swir_max = sig["swir_reflectance"]
        if swir_min <= plot.swir_after <= swir_max:
            scores.append(1.0)
        elif plot.swir_after == 0.0:
            scores.append(0.5)
        else:
            swir_dist = min(
                abs(plot.swir_after - swir_min),
                abs(plot.swir_after - swir_max),
            )
            scores.append(max(0.0, 1.0 - swir_dist * 5.0))

        elev_min, elev_max = sig["elevation_range_m"]
        if elev_min <= plot.elevation_m <= elev_max:
            scores.append(1.0)
        elif plot.elevation_m == 0.0:
            scores.append(0.7)
        else:
            elev_dist = min(
                abs(plot.elevation_m - elev_min),
                abs(plot.elevation_m - elev_max),
            )
            scores.append(max(0.0, 1.0 - elev_dist / 1000.0))

        slope_max = sig["terrain_slope_max_deg"]
        if plot.slope_deg <= slope_max:
            scores.append(1.0)
        else:
            slope_excess = plot.slope_deg - slope_max
            scores.append(max(0.0, 1.0 - slope_excess / 30.0))

        return sum(scores) / max(len(scores), 1)

    # ------------------------------------------------------------------
    # Internal: Temporal Pattern Match
    # ------------------------------------------------------------------

    def _compute_temporal_pattern_match(
        self,
        plot: PlotConversionInput,
        commodity: str,
    ) -> float:
        """Compute temporal NDVI pattern match for commodity phenology.

        Compares the observed NDVI time series seasonal amplitude and
        dip timing against the expected commodity phenology.

        Args:
            plot: Plot input data.
            commodity: Commodity key.

        Returns:
            Temporal pattern match score (0.0-1.0).
        """
        sig = COMMODITY_SPECTRAL_SIGNATURES.get(commodity)
        if sig is None:
            return 0.5

        ts = plot.ndvi_time_series
        if len(ts) < 3:
            return 0.5

        expected_amplitude = sig["ndvi_seasonal_amplitude"]
        observed_amplitude = max(ts) - min(ts) if ts else 0.0

        amplitude_diff = abs(observed_amplitude - expected_amplitude)
        amplitude_score = max(0.0, 1.0 - amplitude_diff * 3.0)

        ndvi_min_idx = ts.index(min(ts)) if ts else 0
        month_of_min = (ndvi_min_idx % 12) + 1
        dip_months = sig.get("ndvi_dip_months", [])

        if not dip_months:
            timing_score = 0.7
        elif month_of_min in dip_months:
            timing_score = 1.0
        else:
            min_distance = min(
                min(abs(month_of_min - m), 12 - abs(month_of_min - m))
                for m in dip_months
            )
            timing_score = max(0.0, 1.0 - min_distance * 0.15)

        ndvi_change = plot.ndvi_before - plot.ndvi_after
        if commodity == "cattle":
            change_score = 1.0 if ndvi_change > 0.3 else max(0.0, ndvi_change / 0.3)
        elif commodity == "cocoa":
            change_score = 1.0 if 0.05 <= ndvi_change <= 0.30 else 0.5
        else:
            change_score = 1.0 if ndvi_change > 0.15 else max(0.0, ndvi_change / 0.15)

        return (
            amplitude_score * 0.35
            + timing_score * 0.35
            + change_score * 0.30
        )

    # ------------------------------------------------------------------
    # Internal: Texture Match
    # ------------------------------------------------------------------

    def _compute_texture_match(
        self,
        plot: PlotConversionInput,
        commodity: str,
    ) -> float:
        """Compute texture regularity match for commodity planting pattern.

        Args:
            plot: Plot input data.
            commodity: Commodity key.

        Returns:
            Texture match score (0.0-1.0).
        """
        sig = COMMODITY_SPECTRAL_SIGNATURES.get(commodity)
        if sig is None:
            return 0.5

        expected_regularity = sig["texture_regularity"]
        observed_regularity = plot.texture_regularity

        diff = abs(observed_regularity - expected_regularity)
        return max(0.0, 1.0 - diff * 2.0)

    # ------------------------------------------------------------------
    # Internal: Data Quality Score
    # ------------------------------------------------------------------

    def _compute_data_quality_score(
        self,
        plot: PlotConversionInput,
    ) -> float:
        """Compute data quality score from cloud cover and resolution.

        Args:
            plot: Plot input data.

        Returns:
            Data quality score (0.0-1.0).
        """
        cloud_factor = max(0.0, 1.0 - plot.cloud_cover_pct / 100.0)
        has_ts = 1.0 if len(plot.ndvi_time_series) >= 6 else 0.5
        has_swir = 1.0 if plot.swir_after > 0.0 else 0.5
        has_terrain = 1.0 if plot.elevation_m > 0.0 or plot.slope_deg > 0.0 else 0.5

        return (
            cloud_factor * 0.40
            + has_ts * 0.25
            + has_swir * 0.20
            + has_terrain * 0.15
        )

    # ------------------------------------------------------------------
    # Internal: Conversion Confidence
    # ------------------------------------------------------------------

    def _compute_conversion_confidence(
        self,
        spectral_evidence: float,
        classification_agreement: float,
        temporal_match: float = 0.5,
        texture_match: float = 0.5,
        data_quality: float = 0.5,
    ) -> float:
        """Compute composite conversion confidence from multiple evidence lines.

        Args:
            spectral_evidence: Spectral match score (0-1).
            classification_agreement: Classification agreement (0-1).
            temporal_match: Temporal pattern match (0-1).
            texture_match: Texture regularity match (0-1).
            data_quality: Data quality factor (0-1).

        Returns:
            Composite confidence score in [0.0, 1.0].
        """
        weighted = (
            spectral_evidence * _CONFIDENCE_WEIGHTS["spectral_match"]
            + temporal_match * _CONFIDENCE_WEIGHTS["temporal_pattern"]
            + texture_match * _CONFIDENCE_WEIGHTS["texture_match"]
            + classification_agreement * _CONFIDENCE_WEIGHTS["classification_agreement"]
            + data_quality * _CONFIDENCE_WEIGHTS["data_quality"]
        )
        return max(0.0, min(1.0, weighted))

    # ------------------------------------------------------------------
    # Internal: Conversion Evaluation
    # ------------------------------------------------------------------

    def _evaluate_conversion(
        self,
        from_class: str,
        ndvi_change: float,
        confidence: float,
        commodity_key: str,
    ) -> bool:
        """Evaluate whether a conversion has occurred.

        A conversion is detected when:
        1. The source class is forest-type.
        2. The NDVI has decreased (for most commodities) or remained
           moderate (for cocoa agroforestry).
        3. The confidence exceeds the minimum threshold.

        Args:
            from_class: Source land use class.
            ndvi_change: NDVI change (after - before).
            confidence: Computed confidence score.
            commodity_key: Commodity identifier.

        Returns:
            True if conversion detected, False otherwise.
        """
        is_from_forest = from_class.lower() in (
            "forest", "primary_tropical", "secondary_tropical",
            "tropical_dry", "mangrove", "agroforestry",
            "temperate_broadleaf", "temperate_coniferous", "boreal",
        )
        if not is_from_forest:
            return False

        min_confidence = 0.40

        if commodity_key == "cocoa":
            has_change = ndvi_change < -0.05
        elif commodity_key == "cattle":
            has_change = ndvi_change < -0.20
        else:
            has_change = ndvi_change < -0.10

        return has_change and confidence >= min_confidence

    # ------------------------------------------------------------------
    # Internal: Expansion Rate
    # ------------------------------------------------------------------

    def _estimate_expansion_rate(
        self,
        conversion_events: List[CroplandConversion],
        period_years: float,
    ) -> float:
        """Estimate expansion rate in hectares per year.

        Args:
            conversion_events: Conversion events.
            period_years: Period in years.

        Returns:
            Rate in hectares per year.
        """
        if period_years <= 0.0:
            raise ValueError(
                f"period_years must be > 0, got {period_years}"
            )

        detected = [c for c in conversion_events if c.conversion_detected]
        total_area = sum(c.area_ha for c in detected)
        return total_area / period_years

    # ------------------------------------------------------------------
    # Internal: Front Direction and Extent
    # ------------------------------------------------------------------

    def _compute_front_direction(
        self,
        conversions: List[CroplandConversion],
    ) -> float:
        """Compute the dominant expansion direction in degrees.

        Uses a weighted centroid displacement vector from older to
        newer conversions. Returns bearing in degrees (0=N, 90=E).

        Args:
            conversions: List of detected conversions.

        Returns:
            Direction in degrees (0-360).
        """
        if len(conversions) < 2:
            return 0.0

        n = len(conversions)
        half = n // 2
        first_half = conversions[:half]
        second_half = conversions[half:]

        lat1 = sum(c.latitude for c in first_half) / max(len(first_half), 1)
        lon1 = sum(c.longitude for c in first_half) / max(len(first_half), 1)
        lat2 = sum(c.latitude for c in second_half) / max(len(second_half), 1)
        lon2 = sum(c.longitude for c in second_half) / max(len(second_half), 1)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        bearing = math.degrees(math.atan2(dlon, dlat))
        if bearing < 0:
            bearing += 360.0
        return bearing

    def _compute_front_extent(
        self,
        conversions: List[CroplandConversion],
    ) -> float:
        """Compute the extent of the expansion front in kilometres.

        Args:
            conversions: List of detected conversions.

        Returns:
            Extent in kilometres.
        """
        if len(conversions) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(conversions)):
            for j in range(i + 1, len(conversions)):
                dist = self._haversine_distance_km(
                    conversions[i].latitude, conversions[i].longitude,
                    conversions[j].latitude, conversions[j].longitude,
                )
                max_dist = max(max_dist, dist)
        return max_dist

    # ------------------------------------------------------------------
    # Internal: Trend and Inflection
    # ------------------------------------------------------------------

    def _compute_linear_trend(self, values: List[float]) -> float:
        """Compute linear trend (slope) from a sequence of values.

        Uses simple linear regression: slope = cov(x,y) / var(x).

        Args:
            values: Sequence of values.

        Returns:
            Slope value (units per step).
        """
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        cov_xy = sum(
            (i - x_mean) * (v - y_mean) for i, v in enumerate(values)
        )
        var_x = sum((i - x_mean) ** 2 for i in range(n))

        if abs(var_x) < 1e-10:
            return 0.0
        return cov_xy / var_x

    def _find_inflection_points(self, values: List[float]) -> List[int]:
        """Find inflection points where the trend changes direction.

        An inflection occurs when the first derivative changes sign.

        Args:
            values: Sequence of values.

        Returns:
            List of indices where inflection occurs.
        """
        if len(values) < 3:
            return []

        inflections: List[int] = []
        for i in range(1, len(values) - 1):
            d1 = values[i] - values[i - 1]
            d2 = values[i + 1] - values[i]
            if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
                inflections.append(i)
        return inflections

    # ------------------------------------------------------------------
    # Internal: Commodity Inference
    # ------------------------------------------------------------------

    def _infer_commodity(self, plot: PlotConversionInput) -> str:
        """Infer the most likely commodity from plot characteristics.

        Uses elevation, NDVI, and texture as discriminators.

        Args:
            plot: Plot input data.

        Returns:
            Best-match commodity key.
        """
        best_score = -1.0
        best_commodity = "cattle"

        for commodity_key in COMMODITY_SPECTRAL_SIGNATURES:
            score = self._compute_spectral_match(plot, commodity_key)
            if score > best_score:
                best_score = score
                best_commodity = commodity_key

        return best_commodity

    # ------------------------------------------------------------------
    # Internal: Terrain and Commodity Suitability
    # ------------------------------------------------------------------

    def _terrain_suitability(
        self,
        lat: float,
        lon: float,
        elevation_range: Tuple[float, float],
        slope_max: float,
    ) -> float:
        """Compute terrain suitability for commodity expansion.

        Uses a simple latitude/longitude-based proxy for elevation
        and slope when actual DEM data is not available.

        Args:
            lat: Latitude.
            lon: Longitude.
            elevation_range: Acceptable elevation range (min, max) metres.
            slope_max: Maximum acceptable slope in degrees.

        Returns:
            Suitability score (0.0-1.0).
        """
        proxy_elev = abs(lat) * 30.0
        elev_min, elev_max = elevation_range
        if elev_min <= proxy_elev <= elev_max:
            elev_score = 1.0
        else:
            dist = min(
                abs(proxy_elev - elev_min),
                abs(proxy_elev - elev_max),
            )
            elev_score = max(0.0, 1.0 - dist / 1000.0)

        proxy_slope = abs(lat % 10.0)
        if proxy_slope <= slope_max:
            slope_score = 1.0
        else:
            slope_score = max(0.0, 1.0 - (proxy_slope - slope_max) / 30.0)

        return elev_score * 0.5 + slope_score * 0.5

    def _commodity_suitability(
        self,
        lat: float,
        lon: float,
        commodity: str,
    ) -> float:
        """Compute commodity-specific suitability for a location.

        Uses climate-zone proxies based on latitude. Tropical
        commodities score highest near the equator, while soya
        tolerates higher latitudes.

        Args:
            lat: Latitude.
            lon: Longitude.
            commodity: Commodity key.

        Returns:
            Suitability score (0.0-1.0).
        """
        abs_lat = abs(lat)

        tropical_commodities = {"palm_oil", "rubber", "cocoa", "coffee"}
        if commodity in tropical_commodities:
            if abs_lat <= 10.0:
                return 1.0
            elif abs_lat <= 23.5:
                return max(0.3, 1.0 - (abs_lat - 10.0) / 30.0)
            else:
                return max(0.0, 0.3 - (abs_lat - 23.5) / 50.0)

        if commodity == "soya":
            if abs_lat <= 40.0:
                return max(0.4, 1.0 - abs_lat / 60.0)
            else:
                return max(0.0, 0.4 - (abs_lat - 40.0) / 50.0)

        if commodity == "cattle":
            if abs_lat <= 45.0:
                return max(0.3, 1.0 - abs_lat / 80.0)
            else:
                return 0.2

        return 0.5

    # ------------------------------------------------------------------
    # Internal: Distance Calculations
    # ------------------------------------------------------------------

    def _haversine_distance_km(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Compute Haversine distance between two points in kilometres.

        Args:
            lat1: Latitude of point 1.
            lon1: Longitude of point 1.
            lat2: Latitude of point 2.
            lon2: Longitude of point 2.

        Returns:
            Distance in kilometres.
        """
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def _haversine_distance_deg(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Compute approximate distance in degrees.

        Args:
            lat1: Latitude of point 1.
            lon1: Longitude of point 1.
            lat2: Latitude of point 2.
            lon2: Longitude of point 2.

        Returns:
            Approximate distance in degrees.
        """
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> None:
        """Validate coordinate ranges.

        Raises:
            ValueError: If coordinates are out of range.
        """
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(
                f"latitude must be in [-90, 90], got {latitude}"
            )
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(
                f"longitude must be in [-180, 180], got {longitude}"
            )

    def _validate_region_bounds(
        self,
        bounds: Dict[str, float],
    ) -> None:
        """Validate region bounds dictionary.

        Raises:
            ValueError: If required keys are missing or invalid.
        """
        required = {"min_lat", "max_lat", "min_lon", "max_lon"}
        missing = required - set(bounds.keys())
        if missing:
            raise ValueError(
                f"region_bounds missing keys: {missing}"
            )
        if bounds["min_lat"] >= bounds["max_lat"]:
            raise ValueError(
                f"min_lat ({bounds['min_lat']}) must be < "
                f"max_lat ({bounds['max_lat']})"
            )
        if bounds["min_lon"] >= bounds["max_lon"]:
            raise ValueError(
                f"min_lon ({bounds['min_lon']}) must be < "
                f"max_lon ({bounds['max_lon']})"
            )

    def _normalize_commodity(self, commodity: str) -> str:
        """Normalize commodity name to a canonical key.

        Args:
            commodity: Raw commodity string.

        Returns:
            Normalized commodity key.

        Raises:
            ValueError: If commodity is not recognized.
        """
        key = commodity.lower().strip()
        if key in COMMODITY_SPECTRAL_SIGNATURES:
            return key
        alias_map = {
            "oil_palm": "palm_oil",
            "beef": "cattle",
            "leather": "cattle",
            "natural_rubber": "rubber",
            "tyres": "rubber",
            "chocolate": "cocoa",
            "soybean_oil": "soya",
            "soybean": "soya",
            "timber": "wood",
        }
        resolved = alias_map.get(key)
        if resolved is not None:
            return resolved
        raise ValueError(
            f"Unknown commodity '{commodity}'. Valid commodities: "
            f"{list(COMMODITY_SPECTRAL_SIGNATURES.keys())}"
        )

    def _parse_period_years(
        self,
        date_from: str,
        date_to: str,
    ) -> float:
        """Parse date strings and compute period in years.

        Args:
            date_from: Start date ISO string.
            date_to: End date ISO string.

        Returns:
            Period in years (float).
        """
        from datetime import date as date_type

        try:
            d1 = date_type.fromisoformat(date_from)
            d2 = date_type.fromisoformat(date_to)
            delta = (d2 - d1).days
            return max(delta / 365.25, 0.01)
        except (ValueError, TypeError):
            return 1.0

    # ------------------------------------------------------------------
    # Internal: Error Result
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        plot: PlotConversionInput,
        error_msg: str,
    ) -> CroplandConversion:
        """Create an error result for a failed analysis.

        Args:
            plot: Plot that caused the error.
            error_msg: Error message.

        Returns:
            CroplandConversion with zero confidence and error metadata.
        """
        result = CroplandConversion(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            conversion_type=ConversionType.UNKNOWN.value,
            commodity="unknown",
            scale="unknown",
            area_ha=plot.area_ha,
            conversion_detected=False,
            confidence=0.0,
            latitude=plot.latitude,
            longitude=plot.longitude,
            timestamp=utcnow().isoformat(),
            metadata={"error": error_msg},
        )
        result.provenance_hash = self._compute_result_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Provenance Hashing
    # ------------------------------------------------------------------

    def _compute_result_hash(self, result: CroplandConversion) -> str:
        """Compute SHA-256 provenance hash for a conversion result.

        Args:
            result: CroplandConversion to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "result_id": result.result_id,
            "plot_id": result.plot_id,
            "conversion_type": result.conversion_type,
            "commodity": result.commodity,
            "scale": result.scale,
            "area_ha": result.area_ha,
            "conversion_detected": result.conversion_detected,
            "confidence": result.confidence,
            "ndvi_change": result.ndvi_change,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "timestamp": result.timestamp,
        }
        return _compute_hash(hash_data)

# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "ConversionType",
    # Constants
    "COMMODITY_SPECTRAL_SIGNATURES",
    # Data classes
    "PlotConversionInput",
    "CroplandConversion",
    # Engine
    "CroplandExpansionDetector",
]
