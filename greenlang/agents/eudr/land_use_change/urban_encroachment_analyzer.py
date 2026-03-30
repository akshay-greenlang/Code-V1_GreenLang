# -*- coding: utf-8 -*-
"""
Urban Encroachment Analyzer Engine - AGENT-EUDR-005: Land Use Change (Engine 7)

Monitors urban and infrastructure expansion near forested production
areas.  Detects settlement growth, road construction, mining activity,
and other infrastructure development that signals conversion pressure
on adjacent forests.

Zero-Hallucination Guarantees:
    - All calculations use deterministic float arithmetic (no ML/LLM).
    - Infrastructure classification via static spectral thresholds
      from published remote sensing literature.
    - Expansion rates computed as simple area-over-time division.
    - Proximity risk follows a deterministic exponential decay model.
    - Time-to-conversion estimated from distance / expansion rate.
    - SHA-256 provenance hashes on all result objects.
    - No ML/LLM used for any numeric computation.

Infrastructure Types (5):
    ROADS:       Linear features, high NIR contrast with surroundings.
    BUILDINGS:   High SWIR reflectance, low NDVI, regular geometry.
    MINING:      Bare soil spectral signature, high SWIR, very low NDVI.
    INDUSTRIAL:  High albedo, regular geometry, large footprint.
    RESIDENTIAL: Mixed spectral signature with vegetation, medium NDVI.

Urban Detection via Spectral Features:
    Settlement:  SWIR (B11) > 0.25, NDVI < 0.15, high texture contrast.
    Roads:       Linear features in NIR, high contrast.
    Mining:      Bare soil (SWIR > 0.25, NDVI < 0.10).
    Industrial:  High albedo, regular geometry.

Regulatory References:
    - EUDR Article 10: Risk assessment considering infrastructure.
    - EUDR Article 12: Country risk assessment includes governance.
    - EUDR Recital 20: Recognizes infrastructure as deforestation driver.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-005 (Engine 7: Urban Encroachment Analysis)
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

class InfrastructureType(str, Enum):
    """Type of infrastructure detected near forested areas.

    Each type has distinct spectral and spatial characteristics.
    """

    ROADS = "ROADS"
    BUILDINGS = "BUILDINGS"
    MINING = "MINING"
    INDUSTRIAL = "INDUSTRIAL"
    RESIDENTIAL = "RESIDENTIAL"
    UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Constants: Spectral Thresholds for Infrastructure Detection
# ---------------------------------------------------------------------------
# Reference: Zha et al. (2003), Xu (2008), As-syakur et al. (2012).

SPECTRAL_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "settlement": {
        "swir_min": 0.25,
        "ndvi_max": 0.15,
        "texture_contrast_min": 0.3,
        "description": "Urban settlement with high SWIR, low NDVI",
    },
    "road": {
        "nir_contrast_min": 0.2,
        "linearity_min": 0.7,
        "ndvi_max": 0.15,
        "width_m_max": 30.0,
        "description": "Linear feature with high NIR contrast",
    },
    "mining": {
        "swir_min": 0.25,
        "ndvi_max": 0.10,
        "bare_soil_index_min": 0.2,
        "description": "Bare soil with high SWIR and very low NDVI",
    },
    "industrial": {
        "albedo_min": 0.3,
        "regularity_min": 0.6,
        "ndvi_max": 0.12,
        "min_area_ha": 1.0,
        "description": "High albedo, regular geometry",
    },
}

# ---------------------------------------------------------------------------
# Constants: Proximity Risk Decay Parameters
# ---------------------------------------------------------------------------

_PROXIMITY_DECAY_RATE: float = 0.05
_PROXIMITY_MAX_DISTANCE_KM: float = 50.0

# ---------------------------------------------------------------------------
# Constants: Infrastructure Risk Weights
# ---------------------------------------------------------------------------
# How much each infrastructure type contributes to conversion pressure.

INFRASTRUCTURE_RISK_WEIGHTS: Dict[str, float] = {
    InfrastructureType.ROADS.value: 0.30,
    InfrastructureType.BUILDINGS.value: 0.20,
    InfrastructureType.MINING.value: 0.25,
    InfrastructureType.INDUSTRIAL.value: 0.15,
    InfrastructureType.RESIDENTIAL.value: 0.10,
}

# ---------------------------------------------------------------------------
# Constants: Expansion Rate References (ha/year)
# ---------------------------------------------------------------------------
# Typical urban expansion rates by region for time-to-conversion estimation.

REGIONAL_EXPANSION_RATES: Dict[str, float] = {
    "tropical_lowland": 250.0,
    "tropical_highland": 150.0,
    "subtropical": 200.0,
    "temperate": 100.0,
    "arid_semi_arid": 80.0,
    "DEFAULT": 150.0,
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class EncroachmentInput:
    """Input data for a single plot encroachment analysis.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Plot centroid latitude (-90 to 90).
        longitude: Plot centroid longitude (-180 to 180).
        buffer_km: Analysis buffer radius in kilometres.
        swir_values: SWIR (B11) reflectance values in the buffer.
        ndvi_values: NDVI values in the buffer.
        nir_values: NIR values in the buffer.
        texture_contrast: Texture contrast values in the buffer.
        albedo_values: Albedo values in the buffer.
        urban_fraction_before: Urban area fraction at date_from.
        urban_fraction_after: Urban area fraction at date_to.
        road_density_before_km_per_km2: Road density at date_from.
        road_density_after_km_per_km2: Road density at date_to.
        distance_to_urban_km: Distance to nearest urban area (km).
        forest_fraction: Current forest fraction in the buffer.
        area_ha: Total buffer area in hectares.
        date_from: Analysis start date.
        date_to: Analysis end date.
    """

    plot_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    buffer_km: float = 10.0
    swir_values: List[float] = field(default_factory=list)
    ndvi_values: List[float] = field(default_factory=list)
    nir_values: List[float] = field(default_factory=list)
    texture_contrast: List[float] = field(default_factory=list)
    albedo_values: List[float] = field(default_factory=list)
    urban_fraction_before: float = 0.0
    urban_fraction_after: float = 0.0
    road_density_before_km_per_km2: float = 0.0
    road_density_after_km_per_km2: float = 0.0
    distance_to_urban_km: float = -1.0
    forest_fraction: float = 0.0
    area_ha: float = 0.0
    date_from: str = ""
    date_to: str = ""

@dataclass
class UrbanEncroachment:
    """Result of urban encroachment analysis for a single plot.

    Attributes:
        result_id: Unique result identifier.
        plot_id: Identifier of the analyzed plot.
        encroachment_detected: Whether encroachment was detected.
        infrastructure_types_detected: List of infrastructure types found.
        urban_expansion_rate_ha_per_year: Urban expansion rate (ha/yr).
        road_density_change: Change in road density (km/km2).
        urban_fraction_change: Change in urban area fraction.
        distance_to_urban_km: Distance to nearest urban area (km).
        time_to_conversion_months: Estimated months until forest
            conversion at current expansion rate.
        pressure_score: Encroachment pressure score (0-100).
        pressure_corridors: Detected pressure corridors.
        new_roads_detected: Number of new road segments detected.
        latitude: Plot centroid latitude.
        longitude: Plot centroid longitude.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp.
        metadata: Additional contextual fields.
    """

    result_id: str = ""
    plot_id: str = ""
    encroachment_detected: bool = False
    infrastructure_types_detected: List[str] = field(default_factory=list)
    urban_expansion_rate_ha_per_year: float = 0.0
    road_density_change: float = 0.0
    urban_fraction_change: float = 0.0
    distance_to_urban_km: float = -1.0
    time_to_conversion_months: float = -1.0
    pressure_score: float = 0.0
    pressure_corridors: List[Dict[str, Any]] = field(default_factory=list)
    new_roads_detected: int = 0
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
            "encroachment_detected": self.encroachment_detected,
            "infrastructure_types_detected": self.infrastructure_types_detected,
            "urban_expansion_rate_ha_per_year": self.urban_expansion_rate_ha_per_year,
            "road_density_change": self.road_density_change,
            "urban_fraction_change": self.urban_fraction_change,
            "distance_to_urban_km": self.distance_to_urban_km,
            "time_to_conversion_months": self.time_to_conversion_months,
            "pressure_score": self.pressure_score,
            "pressure_corridors": self.pressure_corridors,
            "new_roads_detected": self.new_roads_detected,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

# ---------------------------------------------------------------------------
# UrbanEncroachmentAnalyzer
# ---------------------------------------------------------------------------

class UrbanEncroachmentAnalyzer:
    """Monitors urban and infrastructure expansion near forested production areas.

    Detects settlement growth, road construction, mining activity, and
    other infrastructure development that signals conversion pressure.
    All numeric computations are deterministic (zero-hallucination).

    Example::

        analyzer = UrbanEncroachmentAnalyzer()
        inp = EncroachmentInput(
            plot_id="plot-001",
            latitude=-3.4,
            longitude=-62.2,
            buffer_km=10.0,
            urban_fraction_before=0.05,
            urban_fraction_after=0.12,
            distance_to_urban_km=8.0,
        )
        result = analyzer.analyze_encroachment(
            latitude=-3.4, longitude=-62.2,
            buffer_km=10.0,
            date_from="2020-01-01", date_to="2023-01-01",
            plot_input=inp,
        )
        assert result.encroachment_detected in (True, False)

    Attributes:
        config: Optional configuration object.
        max_buffer_km: Maximum allowed buffer distance.
    """

    def __init__(
        self,
        config: Any = None,
        max_buffer_km: float = 50.0,
    ) -> None:
        """Initialize the UrbanEncroachmentAnalyzer.

        Args:
            config: Optional LandUseChangeConfig instance.
            max_buffer_km: Maximum buffer distance in kilometres.
        """
        self.config = config
        self.max_buffer_km = max_buffer_km
        logger.info(
            "UrbanEncroachmentAnalyzer initialized: max_buffer=%.1fkm, "
            "module_version=%s",
            self.max_buffer_km, _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Single Plot Analysis
    # ------------------------------------------------------------------

    def analyze_encroachment(
        self,
        latitude: float,
        longitude: float,
        buffer_km: float,
        date_from: str,
        date_to: str,
        plot_input: Optional[EncroachmentInput] = None,
    ) -> UrbanEncroachment:
        """Analyze urban encroachment for a single plot.

        Detects infrastructure types, computes expansion rates,
        estimates time-to-conversion, and scores encroachment
        pressure.

        Args:
            latitude: Plot centroid latitude.
            longitude: Plot centroid longitude.
            buffer_km: Analysis buffer radius in kilometres.
            date_from: Analysis start date (ISO format).
            date_to: Analysis end date (ISO format).
            plot_input: Optional detailed input data.

        Returns:
            UrbanEncroachment result with detection verdict.

        Raises:
            ValueError: If coordinates or buffer are invalid.
        """
        start_time = time.monotonic()
        self._validate_coordinates(latitude, longitude)
        self._validate_buffer(buffer_km)

        plot = plot_input or EncroachmentInput(
            plot_id=_generate_id(),
            latitude=latitude,
            longitude=longitude,
            buffer_km=buffer_km,
            date_from=date_from,
            date_to=date_to,
        )

        infra_types = self._detect_infrastructure_types(plot)
        expansion_rate = self._calculate_urban_expansion_rate(
            plot, date_from, date_to,
        )
        road_density_change = (
            plot.road_density_after_km_per_km2
            - plot.road_density_before_km_per_km2
        )
        urban_fraction_change = (
            plot.urban_fraction_after - plot.urban_fraction_before
        )
        new_roads = self._detect_new_roads(plot, date_from, date_to)
        corridors = self._detect_pressure_corridors(plot)
        time_to_conversion = self._estimate_time_to_conversion(
            distance_to_frontier=plot.distance_to_urban_km,
            expansion_rate=expansion_rate,
        )
        pressure_score = self._compute_urban_proximity_risk(
            distance_km=plot.distance_to_urban_km,
            expansion_rate=expansion_rate,
            infrastructure_types=infra_types,
        )

        is_encroachment = (
            urban_fraction_change > 0.01
            or road_density_change > 0.1
            or len(infra_types) > 0
        ) and pressure_score > 20.0

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = UrbanEncroachment(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            encroachment_detected=is_encroachment,
            infrastructure_types_detected=[t.value for t in infra_types],
            urban_expansion_rate_ha_per_year=round(expansion_rate, 2),
            road_density_change=round(road_density_change, 4),
            urban_fraction_change=round(urban_fraction_change, 4),
            distance_to_urban_km=round(plot.distance_to_urban_km, 2),
            time_to_conversion_months=round(time_to_conversion, 1),
            pressure_score=round(pressure_score, 2),
            pressure_corridors=corridors,
            new_roads_detected=new_roads,
            latitude=latitude,
            longitude=longitude,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=utcnow().isoformat(),
            metadata={
                "buffer_km": buffer_km,
                "date_from": date_from,
                "date_to": date_to,
                "forest_fraction": plot.forest_fraction,
            },
        )
        result.provenance_hash = self._compute_result_hash(result)

        logger.info(
            "Encroachment analysis: plot=%s, detected=%s, "
            "pressure=%.1f, expansion=%.1f ha/yr, "
            "time_to_conv=%.0f months, %.2fms",
            plot.plot_id, is_encroachment, pressure_score,
            expansion_rate, time_to_conversion, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Batch Analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        plots: List[EncroachmentInput],
        buffer_km: float = 10.0,
    ) -> List[UrbanEncroachment]:
        """Analyze encroachment for multiple plots.

        Args:
            plots: List of encroachment inputs.
            buffer_km: Default buffer radius (overridden by per-plot).

        Returns:
            List of UrbanEncroachment results.

        Raises:
            ValueError: If plots list is empty.
        """
        if not plots:
            raise ValueError("plots list must not be empty")

        start_time = time.monotonic()
        results: List[UrbanEncroachment] = []

        for i, plot in enumerate(plots):
            try:
                eff_buffer = plot.buffer_km if plot.buffer_km > 0 else buffer_km
                result = self.analyze_encroachment(
                    latitude=plot.latitude,
                    longitude=plot.longitude,
                    buffer_km=eff_buffer,
                    date_from=plot.date_from,
                    date_to=plot.date_to,
                    plot_input=plot,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "analyze_batch: failed on plot[%d] id=%s: %s",
                    i, plot.plot_id, str(exc),
                )
                results.append(self._create_error_result(plot, str(exc)))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        detected_count = sum(
            1 for r in results if r.encroachment_detected
        )
        logger.info(
            "analyze_batch complete: %d/%d encroachment detected, "
            "%d plots, %.2fms",
            detected_count, len(plots), len(plots), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Encroachment Map Generation
    # ------------------------------------------------------------------

    def generate_encroachment_map(
        self,
        region: Dict[str, float],
        urban_changes: List[UrbanEncroachment],
    ) -> Dict[str, Any]:
        """Generate an encroachment map from analysis results.

        Args:
            region: Dictionary with min_lat, max_lat, min_lon, max_lon.
            urban_changes: List of encroachment results.

        Returns:
            Dictionary with map data, statistics, and metadata.
        """
        start_time = time.monotonic()

        if not urban_changes:
            return {
                "region": region,
                "plot_count": 0,
                "encroachment_count": 0,
                "map_points": [],
                "provenance_hash": _compute_hash(region),
            }

        map_points: List[Dict[str, Any]] = []
        for result in urban_changes:
            map_points.append({
                "lat": result.latitude,
                "lon": result.longitude,
                "pressure_score": result.pressure_score,
                "encroachment_detected": result.encroachment_detected,
                "infrastructure_types": result.infrastructure_types_detected,
                "expansion_rate": result.urban_expansion_rate_ha_per_year,
            })

        detected = [r for r in urban_changes if r.encroachment_detected]
        scores = [r.pressure_score for r in urban_changes]

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "region": region,
            "plot_count": len(urban_changes),
            "encroachment_count": len(detected),
            "mean_pressure_score": round(
                sum(scores) / max(len(scores), 1), 2,
            ),
            "max_pressure_score": round(max(scores, default=0.0), 2),
            "mean_expansion_rate": round(
                sum(r.urban_expansion_rate_ha_per_year for r in urban_changes) /
                max(len(urban_changes), 1),
                2,
            ),
            "map_points": map_points,
            "processing_time_ms": round(elapsed_ms, 2),
        }
        result["provenance_hash"] = _compute_hash({
            "region": region,
            "plot_count": len(urban_changes),
        })

        logger.info(
            "Encroachment map: %d plots, %d detected, "
            "mean_pressure=%.1f, %.2fms",
            len(urban_changes), len(detected),
            result["mean_pressure_score"], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Accessibility Change Analysis
    # ------------------------------------------------------------------

    def analyze_accessibility_change(
        self,
        region: Dict[str, float],
        road_network_before: List[Dict[str, Any]],
        road_network_after: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze change in road-network accessibility between two dates.

        Compares road networks to identify new segments, increased
        density, and changed accessibility for forest areas.

        Args:
            region: Dictionary with min_lat, max_lat, min_lon, max_lon.
            road_network_before: Road segments at date_from (list of
                dicts with start_lat, start_lon, end_lat, end_lon, type).
            road_network_after: Road segments at date_to.

        Returns:
            Dictionary with accessibility change metrics.
        """
        start_time = time.monotonic()

        before_count = len(road_network_before)
        after_count = len(road_network_after)

        before_length_km = sum(
            self._segment_length_km(seg) for seg in road_network_before
        )
        after_length_km = sum(
            self._segment_length_km(seg) for seg in road_network_after
        )

        new_segments = max(0, after_count - before_count)
        length_change_km = after_length_km - before_length_km
        length_change_pct = (
            (length_change_km / max(before_length_km, 0.01)) * 100.0
        )

        region_area_km2 = self._region_area_km2(region)
        density_before = before_length_km / max(region_area_km2, 0.01)
        density_after = after_length_km / max(region_area_km2, 0.01)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "region": region,
            "segments_before": before_count,
            "segments_after": after_count,
            "new_segments": new_segments,
            "total_length_before_km": round(before_length_km, 2),
            "total_length_after_km": round(after_length_km, 2),
            "length_change_km": round(length_change_km, 2),
            "length_change_pct": round(length_change_pct, 2),
            "density_before_km_per_km2": round(density_before, 4),
            "density_after_km_per_km2": round(density_after, 4),
            "density_change_km_per_km2": round(
                density_after - density_before, 4,
            ),
            "accessibility_increased": length_change_km > 0.0,
            "processing_time_ms": round(elapsed_ms, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Accessibility change: +%d segments, +%.1fkm, "
            "density %.4f->%.4f km/km2, %.2fms",
            new_segments, length_change_km,
            density_before, density_after, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Infrastructure Type Classification
    # ------------------------------------------------------------------

    def _classify_infrastructure_type(
        self,
        swir: float,
        ndvi: float,
        texture_contrast: float,
        albedo: float,
        nir_contrast: float,
    ) -> InfrastructureType:
        """Classify infrastructure type from spectral features.

        Args:
            swir: SWIR (B11) reflectance.
            ndvi: NDVI value.
            texture_contrast: Texture contrast metric.
            albedo: Surface albedo.
            nir_contrast: NIR contrast with surroundings.

        Returns:
            InfrastructureType enum value.
        """
        mining_thresh = SPECTRAL_THRESHOLDS["mining"]
        if (swir >= mining_thresh["swir_min"]
                and ndvi < mining_thresh["ndvi_max"]):
            return InfrastructureType.MINING

        industrial_thresh = SPECTRAL_THRESHOLDS["industrial"]
        if (albedo >= industrial_thresh["albedo_min"]
                and ndvi < industrial_thresh["ndvi_max"]):
            return InfrastructureType.INDUSTRIAL

        road_thresh = SPECTRAL_THRESHOLDS["road"]
        if (nir_contrast >= road_thresh["nir_contrast_min"]
                and ndvi < road_thresh["ndvi_max"]):
            return InfrastructureType.ROADS

        settlement_thresh = SPECTRAL_THRESHOLDS["settlement"]
        if (swir >= settlement_thresh["swir_min"]
                and ndvi < settlement_thresh["ndvi_max"]
                and texture_contrast >= settlement_thresh["texture_contrast_min"]):
            return InfrastructureType.BUILDINGS

        if ndvi < 0.20 and swir > 0.15:
            return InfrastructureType.RESIDENTIAL

        return InfrastructureType.UNKNOWN

    def _detect_infrastructure_types(
        self,
        plot: EncroachmentInput,
    ) -> List[InfrastructureType]:
        """Detect infrastructure types present in the plot buffer.

        Args:
            plot: Encroachment input data.

        Returns:
            List of distinct InfrastructureType values detected.
        """
        detected: set = set()

        n = max(
            len(plot.swir_values),
            len(plot.ndvi_values),
            1,
        )

        for i in range(n):
            swir = plot.swir_values[i] if i < len(plot.swir_values) else 0.0
            ndvi = plot.ndvi_values[i] if i < len(plot.ndvi_values) else 0.5
            nir = plot.nir_values[i] if i < len(plot.nir_values) else 0.0
            tex = (
                plot.texture_contrast[i]
                if i < len(plot.texture_contrast) else 0.0
            )
            alb = plot.albedo_values[i] if i < len(plot.albedo_values) else 0.0

            nir_contrast = abs(nir - 0.3) if nir > 0 else 0.0

            infra = self._classify_infrastructure_type(
                swir=swir,
                ndvi=ndvi,
                texture_contrast=tex,
                albedo=alb,
                nir_contrast=nir_contrast,
            )
            if infra != InfrastructureType.UNKNOWN:
                detected.add(infra)

        if plot.urban_fraction_after > 0.05 and not detected:
            detected.add(InfrastructureType.RESIDENTIAL)

        if plot.road_density_after_km_per_km2 > 0.5 and InfrastructureType.ROADS not in detected:
            detected.add(InfrastructureType.ROADS)

        return sorted(detected, key=lambda x: x.value)

    # ------------------------------------------------------------------
    # Internal: Urban Expansion Rate
    # ------------------------------------------------------------------

    def _calculate_urban_expansion_rate(
        self,
        plot: EncroachmentInput,
        date_from: str,
        date_to: str,
    ) -> float:
        """Calculate urban expansion rate in ha/year.

        Args:
            plot: Encroachment input data.
            date_from: Start date ISO string.
            date_to: End date ISO string.

        Returns:
            Expansion rate in hectares per year.
        """
        period_years = self._parse_period_years(date_from, date_to)
        urban_change_fraction = (
            plot.urban_fraction_after - plot.urban_fraction_before
        )
        urban_change_ha = urban_change_fraction * plot.area_ha
        if period_years <= 0.0:
            return 0.0
        return max(0.0, urban_change_ha / period_years)

    # ------------------------------------------------------------------
    # Internal: Pressure Corridors
    # ------------------------------------------------------------------

    def _detect_pressure_corridors(
        self,
        plot: EncroachmentInput,
    ) -> List[Dict[str, Any]]:
        """Detect pressure corridors between urban growth and forest.

        A pressure corridor is a zone where urban expansion is
        directed towards forest land, creating a linear front of
        conversion pressure.

        Args:
            plot: Encroachment input data.

        Returns:
            List of corridor dictionaries.
        """
        corridors: List[Dict[str, Any]] = []

        if plot.forest_fraction <= 0.0 or plot.urban_fraction_after <= 0.0:
            return corridors

        urban_change = plot.urban_fraction_after - plot.urban_fraction_before
        if urban_change <= 0.01:
            return corridors

        pressure_intensity = min(1.0, urban_change * 10.0)

        corridor = {
            "corridor_id": _generate_id(),
            "pressure_intensity": round(pressure_intensity, 4),
            "forest_fraction": round(plot.forest_fraction, 4),
            "urban_fraction": round(plot.urban_fraction_after, 4),
            "urban_change": round(urban_change, 4),
            "direction": "towards_forest",
            "width_estimate_km": round(
                plot.buffer_km * 0.3 * pressure_intensity, 2,
            ),
        }
        corridor["provenance_hash"] = _compute_hash(corridor)
        corridors.append(corridor)

        return corridors

    # ------------------------------------------------------------------
    # Internal: New Road Detection
    # ------------------------------------------------------------------

    def _detect_new_roads(
        self,
        plot: EncroachmentInput,
        date_from: str,
        date_to: str,
    ) -> int:
        """Detect number of new road segments in the analysis period.

        Uses road density change as a proxy for new road segments.

        Args:
            plot: Encroachment input data.
            date_from: Start date.
            date_to: End date.

        Returns:
            Estimated number of new road segments.
        """
        density_change = (
            plot.road_density_after_km_per_km2
            - plot.road_density_before_km_per_km2
        )

        if density_change <= 0.0:
            return 0

        segment_length_km = 2.0
        area_km2 = plot.area_ha / 100.0 if plot.area_ha > 0 else 1.0
        new_length_km = density_change * area_km2
        return max(0, int(new_length_km / segment_length_km))

    # ------------------------------------------------------------------
    # Internal: Time-to-Conversion Estimation
    # ------------------------------------------------------------------

    def _estimate_time_to_conversion(
        self,
        distance_to_frontier: float,
        expansion_rate: float,
    ) -> float:
        """Estimate time until forest conversion in months.

        Divides the distance to the urban frontier by the expansion
        rate, converting from years to months.

        Args:
            distance_to_frontier: Distance to urban edge in km.
            expansion_rate: Urban expansion rate in ha/year.

        Returns:
            Estimated months until conversion. -1 if uncertain.
        """
        if distance_to_frontier < 0.0 or expansion_rate <= 0.0:
            return -1.0

        expansion_km_per_year = math.sqrt(expansion_rate / 100.0)
        if expansion_km_per_year <= 0.0:
            return -1.0

        years = distance_to_frontier / expansion_km_per_year
        months = years * 12.0
        return max(0.0, months)

    # ------------------------------------------------------------------
    # Internal: Urban Proximity Risk
    # ------------------------------------------------------------------

    def _compute_urban_proximity_risk(
        self,
        distance_km: float,
        expansion_rate: float,
        infrastructure_types: List[InfrastructureType],
    ) -> float:
        """Compute encroachment pressure score from proximity and expansion.

        Combines:
        - Distance-based exponential decay (0-50 points).
        - Expansion rate contribution (0-30 points).
        - Infrastructure type risk (0-20 points).

        Args:
            distance_km: Distance to nearest urban area (km).
            expansion_rate: Urban expansion rate (ha/year).
            infrastructure_types: Detected infrastructure types.

        Returns:
            Pressure score (0-100).
        """
        if distance_km < 0.0:
            distance_score = 25.0
        else:
            distance_score = 50.0 * math.exp(
                -_PROXIMITY_DECAY_RATE * distance_km
            )

        if expansion_rate <= 0.0:
            expansion_score = 0.0
        elif expansion_rate < 10.0:
            expansion_score = 5.0
        elif expansion_rate < 50.0:
            expansion_score = 10.0
        elif expansion_rate < 200.0:
            expansion_score = 20.0
        else:
            expansion_score = 30.0

        infra_score = 0.0
        for infra_type in infrastructure_types:
            weight = INFRASTRUCTURE_RISK_WEIGHTS.get(
                infra_type.value, 0.05,
            )
            infra_score += weight * 20.0

        infra_score = min(20.0, infra_score)

        return max(0.0, min(100.0,
                            distance_score + expansion_score + infra_score))

    # ------------------------------------------------------------------
    # Internal: Utility Methods
    # ------------------------------------------------------------------

    def _segment_length_km(self, segment: Dict[str, Any]) -> float:
        """Compute length of a road segment in kilometres.

        Args:
            segment: Dictionary with start_lat, start_lon,
                end_lat, end_lon.

        Returns:
            Length in kilometres.
        """
        lat1 = segment.get("start_lat", 0.0)
        lon1 = segment.get("start_lon", 0.0)
        lat2 = segment.get("end_lat", 0.0)
        lon2 = segment.get("end_lon", 0.0)
        return self._haversine_distance_km(lat1, lon1, lat2, lon2)

    def _region_area_km2(self, region: Dict[str, float]) -> float:
        """Compute approximate area of a region in km2.

        Args:
            region: Dictionary with min_lat, max_lat, min_lon, max_lon.

        Returns:
            Area in km2.
        """
        dlat = abs(region.get("max_lat", 0.0) - region.get("min_lat", 0.0))
        dlon = abs(region.get("max_lon", 0.0) - region.get("min_lon", 0.0))
        mid_lat = (region.get("min_lat", 0.0) + region.get("max_lat", 0.0)) / 2.0
        lat_km = dlat * 111.32
        lon_km = dlon * 111.32 * math.cos(math.radians(mid_lat))
        return lat_km * lon_km

    def _haversine_distance_km(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Compute Haversine distance in kilometres.

        Args:
            lat1, lon1: First point coordinates.
            lat2, lon2: Second point coordinates.

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
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> None:
        """Validate coordinate ranges."""
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(
                f"latitude must be in [-90, 90], got {latitude}"
            )
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(
                f"longitude must be in [-180, 180], got {longitude}"
            )

    def _validate_buffer(self, buffer_km: float) -> None:
        """Validate buffer distance."""
        if buffer_km <= 0.0:
            raise ValueError(
                f"buffer_km must be > 0, got {buffer_km}"
            )
        if buffer_km > self.max_buffer_km:
            raise ValueError(
                f"buffer_km ({buffer_km}) exceeds max ({self.max_buffer_km})"
            )

    # ------------------------------------------------------------------
    # Internal: Error Result
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        plot: EncroachmentInput,
        error_msg: str,
    ) -> UrbanEncroachment:
        """Create an error result for a failed analysis.

        Args:
            plot: Input that caused the error.
            error_msg: Error message.

        Returns:
            UrbanEncroachment with defaults and error metadata.
        """
        result = UrbanEncroachment(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            encroachment_detected=False,
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

    def _compute_result_hash(self, result: UrbanEncroachment) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            result: UrbanEncroachment to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "result_id": result.result_id,
            "plot_id": result.plot_id,
            "encroachment_detected": result.encroachment_detected,
            "pressure_score": result.pressure_score,
            "urban_expansion_rate": result.urban_expansion_rate_ha_per_year,
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
    "InfrastructureType",
    # Constants
    "SPECTRAL_THRESHOLDS",
    "INFRASTRUCTURE_RISK_WEIGHTS",
    "REGIONAL_EXPANSION_RATES",
    # Data classes
    "EncroachmentInput",
    "UrbanEncroachment",
    # Engine
    "UrbanEncroachmentAnalyzer",
]
