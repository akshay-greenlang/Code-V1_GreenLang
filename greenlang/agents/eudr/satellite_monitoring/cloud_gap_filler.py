# -*- coding: utf-8 -*-
"""
CloudGapFiller - AGENT-EUDR-003 Feature 6: Cloud-Obscured Imagery Gap Filling

Handles cloud-contaminated satellite imagery by detecting cloud cover,
classifying cloud types, and filling observational gaps using multiple
deterministic methods: temporal compositing, SAR-optical fusion, temporal
interpolation, and nearest-clear-observation selection.

Cloud Detection:
    Uses brightness threshold on visible bands with optional thermal band
    refinement. Returns a per-pixel cloud mask, cloud percentage, and
    clear pixel count for quality assessment.

Gap-Filling Methods:
    temporal_composite: Median-pixel composite from multi-date clear observations.
    sar_fusion:         Sentinel-1 VV/VH backscatter to classify forest through clouds.
    temporal_interpolation: Linear interpolation of NDVI across cloudy dates.
    nearest_clear:      Select nearest-in-time cloud-free observation.

Cloud Persistence Regions:
    Includes reference data for tropical regions with persistent cloud cover
    (Amazon Basin, Congo Basin, Borneo/Sumatra, Central America) to inform
    seasonal compositing windows and expected gap-fill quality.

Zero-Hallucination Guarantees:
    - All gap-filling uses deterministic numerical methods (median, linear interp).
    - Forest classification via SAR uses fixed dB thresholds (no ML/LLM).
    - SHA-256 provenance hashes on all filled results.
    - Quality degradation scores are arithmetic (no probabilistic models).

Performance Targets:
    - Cloud detection (single scene): <10ms
    - Temporal composite (10 scenes): <50ms
    - SAR fusion (single scene pair): <15ms

Regulatory References:
    - EUDR Article 2(1): Deforestation-free verification requires clear imagery.
    - EUDR Article 10: Risk assessment must account for data gaps.
    - Copernicus Sentinel-1/2: Primary European satellite data sources.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003, Feature 6
Agent ID: GL-EUDR-SAT-003
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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Module version for provenance tracking.
_MODULE_VERSION: str = "1.0.0"

#: Default brightness threshold for cloud detection (reflectance 0-1 scale).
DEFAULT_BRIGHTNESS_THRESHOLD: float = 0.35

#: Default thermal threshold for cloud confirmation (Kelvin).
DEFAULT_THERMAL_THRESHOLD_K: float = 270.0

#: Default temporal compositing window in days.
DEFAULT_COMPOSITE_WINDOW_DAYS: int = 30

#: SAR forest classification thresholds (Sentinel-1, C-band).
SAR_VH_FOREST_THRESHOLD_DB: float = -12.0
SAR_VH_VV_RATIO_THRESHOLD_DB: float = -7.0

#: Minimum clear pixel percentage for a scene to be considered usable.
MIN_CLEAR_PIXEL_PCT: float = 20.0

#: Maximum cloud cover for a scene to be considered "clear".
MAX_CLEAR_CLOUD_PCT: float = 15.0


# ---------------------------------------------------------------------------
# Cloud Persistence Regions Reference Data
# ---------------------------------------------------------------------------

CLOUD_PERSISTENCE_REGIONS: List[Dict[str, Any]] = [
    {
        "region_name": "Amazon Basin",
        "lat_range": (-15.0, 5.0),
        "lon_range": (-75.0, -50.0),
        "wet_months": [6, 7, 8, 9, 10, 11],
        "avg_cloud_pct": 72.0,
        "description": (
            "Persistent cloud cover Jun-Nov over the Amazon rainforest. "
            "SAR fusion critical for wet-season monitoring of soya and cattle."
        ),
    },
    {
        "region_name": "Congo Basin",
        "lat_range": (-5.0, 5.0),
        "lon_range": (15.0, 30.0),
        "wet_months": [9, 10, 11, 12],
        "avg_cloud_pct": 68.0,
        "description": (
            "Dense tropical cloud cover Sep-Dec over the Congo Basin. "
            "Affects monitoring of cocoa and wood commodity sourcing areas."
        ),
    },
    {
        "region_name": "Borneo/Sumatra",
        "lat_range": (-5.0, 5.0),
        "lon_range": (100.0, 120.0),
        "wet_months": [11, 12, 1, 2, 3],
        "avg_cloud_pct": 63.0,
        "description": (
            "Monsoon-driven cloud cover Nov-Mar over Southeast Asian "
            "palm oil and rubber production areas."
        ),
    },
    {
        "region_name": "Central America",
        "lat_range": (8.0, 20.0),
        "lon_range": (-92.0, -78.0),
        "wet_months": [5, 6, 7, 8, 9, 10],
        "avg_cloud_pct": 58.0,
        "description": (
            "Rainy season cloud cover May-Oct affecting coffee and cocoa "
            "monitoring in Guatemala, Honduras, and Costa Rica."
        ),
    },
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CloudCoverAnalysis:
    """Result of cloud cover detection for a single scene.

    Attributes:
        analysis_id: Unique analysis identifier.
        analyzed_at: UTC timestamp of analysis.
        cloud_percentage: Percentage of scene obscured by cloud (0-100).
        clear_pixel_count: Number of clear (non-cloud) pixels.
        total_pixel_count: Total number of pixels in the scene.
        cloud_mask: Per-pixel cloud mask (1 = cloud, 0 = clear).
        has_thermal: Whether thermal band was used for confirmation.
        brightness_threshold: Brightness threshold used for detection.
        thermal_threshold_k: Thermal threshold used (if applicable).
        is_usable: Whether enough clear pixels exist for analysis.
        cloud_persistence_region: Name of cloud persistence region (if applicable).
        provenance_hash: SHA-256 hash for tamper detection.
    """

    analysis_id: str = field(default_factory=lambda: _generate_id("CCA"))
    analyzed_at: datetime = field(default_factory=_utcnow)
    cloud_percentage: float = 0.0
    clear_pixel_count: int = 0
    total_pixel_count: int = 0
    cloud_mask: List[List[int]] = field(default_factory=list)
    has_thermal: bool = False
    brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD
    thermal_threshold_k: float = DEFAULT_THERMAL_THRESHOLD_K
    is_usable: bool = True
    cloud_persistence_region: Optional[str] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "analysis_id": self.analysis_id,
            "analyzed_at": self.analyzed_at.isoformat(),
            "cloud_percentage": round(self.cloud_percentage, 2),
            "clear_pixel_count": self.clear_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "cloud_mask_shape": (
                f"{len(self.cloud_mask)}x{len(self.cloud_mask[0])}"
                if self.cloud_mask and self.cloud_mask[0] else "0x0"
            ),
            "has_thermal": self.has_thermal,
            "brightness_threshold": self.brightness_threshold,
            "thermal_threshold_k": self.thermal_threshold_k,
            "is_usable": self.is_usable,
            "cloud_persistence_region": self.cloud_persistence_region,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class GapFillResult:
    """Result of a gap-filling operation.

    Attributes:
        result_id: Unique result identifier.
        filled_at: UTC timestamp of gap filling.
        method: Gap-filling method used.
        input_cloud_pct: Original cloud coverage percentage.
        output_cloud_pct: Residual cloud coverage after filling.
        pixels_filled: Number of pixels that were filled.
        quality_score: Quality degradation score (0-100, 100 = perfect).
        filled_data: The gap-filled data (band values, NDVI, etc.).
        metadata: Additional method-specific metadata.
        provenance_hash: SHA-256 hash for tamper detection.
    """

    result_id: str = field(default_factory=lambda: _generate_id("GFR"))
    filled_at: datetime = field(default_factory=_utcnow)
    method: str = ""
    input_cloud_pct: float = 0.0
    output_cloud_pct: float = 0.0
    pixels_filled: int = 0
    quality_score: float = 0.0
    filled_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "result_id": self.result_id,
            "filled_at": self.filled_at.isoformat(),
            "method": self.method,
            "input_cloud_pct": round(self.input_cloud_pct, 2),
            "output_cloud_pct": round(self.output_cloud_pct, 2),
            "pixels_filled": self.pixels_filled,
            "quality_score": round(self.quality_score, 2),
            "filled_data_keys": list(self.filled_data.keys()),
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# CloudGapFiller
# ---------------------------------------------------------------------------


class CloudGapFiller:
    """Cloud-obscured imagery gap-filling engine for EUDR satellite monitoring.

    Detects cloud cover in satellite imagery and fills observational gaps
    using multiple deterministic methods. Supports temporal compositing,
    SAR-optical fusion, temporal interpolation, and nearest-clear selection.

    All gap-filling methods are deterministic (median, linear interpolation,
    threshold classification) with zero ML/LLM involvement.

    Attributes:
        _brightness_threshold: Brightness threshold for cloud detection.
        _thermal_threshold_k: Thermal threshold for cloud confirmation.
        _composite_window_days: Default temporal compositing window.
        _gap_fill_store: In-memory store of gap-fill results.

    Example::

        filler = CloudGapFiller()

        # Detect cloud cover in a scene
        band_data = [[0.1, 0.2, 0.5], [0.4, 0.15, 0.6], [0.3, 0.1, 0.45]]
        analysis = filler.detect_cloud_cover(band_data, {"sensor": "S2"})
        assert analysis.cloud_percentage >= 0.0

        # Fill gaps using temporal composite
        scenes = [
            {"date": "2024-06-01", "band_data": [[0.1, 0.2], [0.3, 0.4]],
             "cloud_pct": 10.0},
            {"date": "2024-06-15", "band_data": [[0.12, 0.22], [0.32, 0.38]],
             "cloud_pct": 5.0},
        ]
        filled = filler.temporal_composite(scenes, "2024-06-10", window_days=30)
        assert filled["quality_score"] > 0
    """

    def __init__(
        self,
        brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
        thermal_threshold_k: float = DEFAULT_THERMAL_THRESHOLD_K,
        composite_window_days: int = DEFAULT_COMPOSITE_WINDOW_DAYS,
        config: Any = None,
    ) -> None:
        """Initialize the CloudGapFiller.

        Args:
            brightness_threshold: Reflectance threshold above which pixels
                are classified as cloud (0.0-1.0). Default 0.35.
            thermal_threshold_k: Temperature threshold below which bright
                pixels are confirmed as cloud (Kelvin). Default 270.0.
            composite_window_days: Default temporal compositing window
                in days. Default 30.
            config: Optional configuration object. Reserved for future use.
        """
        self._brightness_threshold = brightness_threshold
        self._thermal_threshold_k = thermal_threshold_k
        self._composite_window_days = composite_window_days
        self._gap_fill_store: Dict[str, GapFillResult] = {}

        logger.info(
            "CloudGapFiller initialized: brightness_thresh=%.2f, "
            "thermal_thresh=%.1fK, composite_window=%d days",
            self._brightness_threshold,
            self._thermal_threshold_k,
            self._composite_window_days,
        )

    # ------------------------------------------------------------------
    # Public API: Cloud Detection
    # ------------------------------------------------------------------

    def detect_cloud_cover(
        self,
        band_data: List[List[float]],
        scene_metadata: Optional[Dict[str, Any]] = None,
    ) -> CloudCoverAnalysis:
        """Estimate cloud coverage from reflectance band data.

        Uses a brightness threshold on the visible band reflectance values.
        If a thermal band is available in scene_metadata, applies thermal
        confirmation to distinguish bright surfaces from actual clouds.

        Args:
            band_data: 2D array of reflectance values (0.0-1.0), where each
                inner list is a row of pixel values.
            scene_metadata: Optional metadata dict. Recognized keys:
                - ``thermal_band``: 2D array of temperature values (Kelvin).
                - ``sensor``: Sensor name string.
                - ``date``: Observation date string.

        Returns:
            CloudCoverAnalysis with cloud mask and statistics.

        Raises:
            ValueError: If band_data is empty.
        """
        start_time = time.monotonic()
        metadata = scene_metadata or {}

        if not band_data or not band_data[0]:
            raise ValueError("band_data must be a non-empty 2D array")

        rows = len(band_data)
        cols = len(band_data[0])
        total_pixels = rows * cols

        # Extract thermal band if available
        thermal_band = metadata.get("thermal_band")
        has_thermal = thermal_band is not None and len(thermal_band) == rows

        # Build cloud mask
        cloud_mask: List[List[int]] = []
        cloud_pixel_count = 0

        for r in range(rows):
            row_mask: List[int] = []
            for c in range(cols):
                reflectance = band_data[r][c] if c < len(band_data[r]) else 0.0
                is_bright = reflectance > self._brightness_threshold

                if is_bright and has_thermal:
                    # Thermal confirmation: clouds are cold
                    temp = (
                        thermal_band[r][c]
                        if c < len(thermal_band[r])
                        else 300.0
                    )
                    is_cloud = temp < self._thermal_threshold_k
                else:
                    is_cloud = is_bright

                pixel_val = 1 if is_cloud else 0
                row_mask.append(pixel_val)
                cloud_pixel_count += pixel_val

            cloud_mask.append(row_mask)

        cloud_pct = (
            (cloud_pixel_count / total_pixels) * 100.0
            if total_pixels > 0
            else 0.0
        )
        clear_pixels = total_pixels - cloud_pixel_count
        is_usable = (clear_pixels / total_pixels * 100.0) >= MIN_CLEAR_PIXEL_PCT

        # Check cloud persistence region
        lat = metadata.get("center_lat")
        lon = metadata.get("center_lon")
        persistence_region = self._check_cloud_persistence_region(lat, lon)

        analysis = CloudCoverAnalysis(
            cloud_percentage=round(cloud_pct, 2),
            clear_pixel_count=clear_pixels,
            total_pixel_count=total_pixels,
            cloud_mask=cloud_mask,
            has_thermal=has_thermal,
            brightness_threshold=self._brightness_threshold,
            thermal_threshold_k=self._thermal_threshold_k,
            is_usable=is_usable,
            cloud_persistence_region=persistence_region,
        )
        analysis.provenance_hash = _compute_hash(analysis)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Cloud detection %s: cloud=%.1f%%, clear=%d/%d pixels, "
            "usable=%s, thermal=%s, region=%s, elapsed=%.2fms",
            analysis.analysis_id,
            cloud_pct,
            clear_pixels,
            total_pixels,
            is_usable,
            has_thermal,
            persistence_region or "none",
            elapsed_ms,
        )

        return analysis

    # ------------------------------------------------------------------
    # Public API: Gap Filling Router
    # ------------------------------------------------------------------

    def fill_gaps(
        self,
        cloudy_scene: Dict[str, Any],
        available_scenes: List[Dict[str, Any]],
        method: str = "temporal_composite",
    ) -> GapFillResult:
        """Fill cloud gaps using the specified method.

        Routes to the appropriate gap-filling implementation based on the
        method parameter.

        Args:
            cloudy_scene: Dict with keys: band_data, cloud_mask, date,
                cloud_pct, and optional metadata.
            available_scenes: List of alternative scene dicts for filling.
            method: Gap-filling method. One of: temporal_composite,
                sar_fusion, temporal_interpolation, nearest_clear.

        Returns:
            GapFillResult with filled data and quality metrics.

        Raises:
            ValueError: If method is not recognized.
        """
        method_lower = method.lower().strip()

        if method_lower == "temporal_composite":
            return self.temporal_composite(
                scenes=available_scenes,
                target_date=cloudy_scene.get("date", ""),
                window_days=self._composite_window_days,
            )
        elif method_lower == "sar_fusion":
            sar_data = cloudy_scene.get("sar_data", {})
            return self.sar_fusion(
                optical_cloudy=cloudy_scene,
                sar_data=sar_data,
            )
        elif method_lower == "temporal_interpolation":
            ndvi_series = []
            for scene in available_scenes:
                if "date" in scene and "ndvi" in scene:
                    ndvi_series.append({
                        "date": scene["date"],
                        "ndvi": scene["ndvi"],
                    })
            cloudy_dates = [cloudy_scene.get("date", "")]
            return self.temporal_interpolation(
                ndvi_series=ndvi_series,
                cloudy_dates=cloudy_dates,
            )
        elif method_lower == "nearest_clear":
            return self.nearest_clear(
                scenes=available_scenes,
                target_date=cloudy_scene.get("date", ""),
            )
        else:
            raise ValueError(
                f"Unrecognized gap-filling method: '{method}'. "
                f"Valid methods: temporal_composite, sar_fusion, "
                f"temporal_interpolation, nearest_clear"
            )

    # ------------------------------------------------------------------
    # Public API: Temporal Composite
    # ------------------------------------------------------------------

    def temporal_composite(
        self,
        scenes: List[Dict[str, Any]],
        target_date: str,
        window_days: Optional[int] = None,
    ) -> GapFillResult:
        """Build a median-pixel composite from multi-date cloud-free observations.

        Selects scenes within the temporal window around the target date,
        filters to scenes below the cloud threshold, and computes a
        per-pixel median across all qualifying scenes.

        Args:
            scenes: List of scene dicts with keys: date, band_data, cloud_pct.
            target_date: Target observation date (ISO 8601 YYYY-MM-DD).
            window_days: Temporal window in days (default: instance default).

        Returns:
            GapFillResult with composite band data and quality score.
        """
        start_time = time.monotonic()
        effective_window = (
            window_days if window_days is not None
            else self._composite_window_days
        )

        # Parse target date
        try:
            target_dt = datetime.fromisoformat(target_date)
        except (ValueError, TypeError):
            target_dt = _utcnow()

        # Filter scenes within window and below cloud threshold
        qualifying_scenes: List[Dict[str, Any]] = []
        for scene in scenes:
            scene_date_str = scene.get("date", "")
            scene_cloud = float(scene.get("cloud_pct", 100.0))
            try:
                scene_dt = datetime.fromisoformat(scene_date_str)
                days_diff = abs((scene_dt - target_dt).days)
                if (
                    days_diff <= effective_window
                    and scene_cloud <= MAX_CLEAR_CLOUD_PCT
                ):
                    qualifying_scenes.append(scene)
            except (ValueError, TypeError):
                continue

        if not qualifying_scenes:
            result = GapFillResult(
                method="temporal_composite",
                input_cloud_pct=100.0,
                output_cloud_pct=100.0,
                pixels_filled=0,
                quality_score=0.0,
                filled_data={},
                metadata={
                    "target_date": target_date,
                    "window_days": effective_window,
                    "qualifying_scenes": 0,
                    "error": "No qualifying clear scenes found within window",
                },
            )
            result.provenance_hash = _compute_hash(result)
            self._gap_fill_store[result.result_id] = result
            return result

        # Compute median composite
        composite_data, pixels_filled = self._compute_median_composite(
            qualifying_scenes
        )

        # Estimate quality: more scenes and closer in time = higher quality
        quality_score = self._calculate_composite_quality(
            qualifying_scenes, target_dt, effective_window
        )

        avg_input_cloud = sum(
            float(s.get("cloud_pct", 0.0)) for s in qualifying_scenes
        ) / len(qualifying_scenes)

        result = GapFillResult(
            method="temporal_composite",
            input_cloud_pct=round(avg_input_cloud, 2),
            output_cloud_pct=0.0,
            pixels_filled=pixels_filled,
            quality_score=round(quality_score, 2),
            filled_data=composite_data,
            metadata={
                "target_date": target_date,
                "window_days": effective_window,
                "qualifying_scenes": len(qualifying_scenes),
                "scene_dates": [
                    s.get("date", "") for s in qualifying_scenes
                ],
            },
        )
        result.provenance_hash = _compute_hash(result)
        self._gap_fill_store[result.result_id] = result

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Temporal composite %s: %d qualifying scenes, %d pixels filled, "
            "quality=%.1f, elapsed=%.2fms",
            result.result_id,
            len(qualifying_scenes),
            pixels_filled,
            quality_score,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: SAR Fusion
    # ------------------------------------------------------------------

    def sar_fusion(
        self,
        optical_cloudy: Dict[str, Any],
        sar_data: Dict[str, Any],
    ) -> GapFillResult:
        """Use Sentinel-1 VV/VH backscatter to fill optical cloud gaps.

        SAR penetrates clouds, enabling forest/non-forest classification
        even under complete cloud cover. Uses fixed dB thresholds:
            - Forest: VH < -12 dB AND VH/VV ratio < -7 dB
            - Non-forest: Otherwise

        Args:
            optical_cloudy: Cloudy optical scene dict with keys: band_data,
                cloud_mask, cloud_pct, date.
            sar_data: SAR observation dict with keys: vv_band (2D array of
                dB values), vh_band (2D array of dB values), date.

        Returns:
            GapFillResult with SAR-derived forest classification.
        """
        start_time = time.monotonic()

        vv_band = sar_data.get("vv_band", [])
        vh_band = sar_data.get("vh_band", [])
        cloud_pct = float(optical_cloudy.get("cloud_pct", 0.0))

        if not vv_band or not vh_band:
            result = GapFillResult(
                method="sar_fusion",
                input_cloud_pct=cloud_pct,
                output_cloud_pct=cloud_pct,
                pixels_filled=0,
                quality_score=0.0,
                filled_data={},
                metadata={"error": "Missing SAR VV/VH band data"},
            )
            result.provenance_hash = _compute_hash(result)
            self._gap_fill_store[result.result_id] = result
            return result

        rows = len(vh_band)
        cols = len(vh_band[0]) if rows > 0 else 0

        # Classify each pixel as forest (1) or non-forest (0)
        forest_mask: List[List[int]] = []
        forest_pixel_count = 0
        total_pixels = 0

        for r in range(rows):
            row_mask: List[int] = []
            for c in range(cols):
                vh_val = vh_band[r][c] if c < len(vh_band[r]) else 0.0
                vv_val = vv_band[r][c] if c < len(vv_band[r]) else 0.0

                # Calculate VH/VV ratio in dB
                vh_vv_ratio = vh_val - vv_val

                # Forest classification: VH < -12 dB AND VH/VV < -7 dB
                is_forest = (
                    vh_val < SAR_VH_FOREST_THRESHOLD_DB
                    and vh_vv_ratio < SAR_VH_VV_RATIO_THRESHOLD_DB
                )

                pixel_val = 1 if is_forest else 0
                row_mask.append(pixel_val)
                forest_pixel_count += pixel_val
                total_pixels += 1

            forest_mask.append(row_mask)

        # All cloudy pixels are effectively "filled" by SAR
        pixels_filled = total_pixels

        # Quality depends on SAR data characteristics
        quality_score = self._calculate_sar_quality(
            total_pixels, forest_pixel_count, sar_data
        )

        forest_pct = (
            (forest_pixel_count / total_pixels * 100.0)
            if total_pixels > 0
            else 0.0
        )

        result = GapFillResult(
            method="sar_fusion",
            input_cloud_pct=round(cloud_pct, 2),
            output_cloud_pct=0.0,
            pixels_filled=pixels_filled,
            quality_score=round(quality_score, 2),
            filled_data={
                "forest_mask": forest_mask,
                "forest_percentage": round(forest_pct, 2),
                "forest_pixels": forest_pixel_count,
                "non_forest_pixels": total_pixels - forest_pixel_count,
            },
            metadata={
                "sar_date": sar_data.get("date", ""),
                "optical_date": optical_cloudy.get("date", ""),
                "vh_threshold_db": SAR_VH_FOREST_THRESHOLD_DB,
                "vh_vv_ratio_threshold_db": SAR_VH_VV_RATIO_THRESHOLD_DB,
                "rows": rows,
                "cols": cols,
            },
        )
        result.provenance_hash = _compute_hash(result)
        self._gap_fill_store[result.result_id] = result

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "SAR fusion %s: %d pixels classified, forest=%.1f%%, "
            "quality=%.1f, elapsed=%.2fms",
            result.result_id,
            total_pixels,
            forest_pct,
            quality_score,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Temporal Interpolation
    # ------------------------------------------------------------------

    def temporal_interpolation(
        self,
        ndvi_series: List[Dict[str, Any]],
        cloudy_dates: List[str],
    ) -> GapFillResult:
        """Linearly interpolate NDVI values across cloudy dates.

        Finds the nearest clear observations before and after each cloudy
        date, then performs linear interpolation to estimate the NDVI
        at the cloudy date.

        Args:
            ndvi_series: List of clear-sky NDVI observations, each a dict
                with keys: date (ISO 8601), ndvi (float).
            cloudy_dates: List of dates (ISO 8601) where NDVI is missing
                due to cloud cover.

        Returns:
            GapFillResult with interpolated NDVI values.
        """
        start_time = time.monotonic()

        if not ndvi_series or not cloudy_dates:
            result = GapFillResult(
                method="temporal_interpolation",
                input_cloud_pct=100.0,
                output_cloud_pct=0.0,
                pixels_filled=0,
                quality_score=0.0,
                filled_data={"interpolated_ndvi": []},
                metadata={"error": "Insufficient data for interpolation"},
            )
            result.provenance_hash = _compute_hash(result)
            self._gap_fill_store[result.result_id] = result
            return result

        # Parse and sort clear observations by date
        parsed_obs: List[Tuple[datetime, float]] = []
        for obs in ndvi_series:
            try:
                obs_dt = datetime.fromisoformat(str(obs.get("date", "")))
                ndvi_val = float(obs.get("ndvi", 0.0))
                parsed_obs.append((obs_dt, ndvi_val))
            except (ValueError, TypeError):
                continue

        parsed_obs.sort(key=lambda x: x[0])

        # Interpolate for each cloudy date
        interpolated: List[Dict[str, Any]] = []
        filled_count = 0
        total_gap_days = 0

        for cloudy_date_str in cloudy_dates:
            try:
                cloudy_dt = datetime.fromisoformat(cloudy_date_str)
            except (ValueError, TypeError):
                continue

            # Find bracketing clear observations
            before: Optional[Tuple[datetime, float]] = None
            after: Optional[Tuple[datetime, float]] = None

            for obs_dt, ndvi_val in parsed_obs:
                if obs_dt <= cloudy_dt:
                    before = (obs_dt, ndvi_val)
                elif obs_dt > cloudy_dt and after is None:
                    after = (obs_dt, ndvi_val)
                    break

            if before is not None and after is not None:
                # Linear interpolation
                total_days = (after[0] - before[0]).total_seconds() / 86400.0
                target_days = (
                    (cloudy_dt - before[0]).total_seconds() / 86400.0
                )

                if total_days > 0:
                    fraction = target_days / total_days
                    interp_ndvi = (
                        before[1] + fraction * (after[1] - before[1])
                    )
                else:
                    interp_ndvi = before[1]

                total_gap_days += int(total_days)
                interpolated.append({
                    "date": cloudy_date_str,
                    "ndvi": round(interp_ndvi, 4),
                    "method": "linear_interpolation",
                    "before_date": before[0].isoformat(),
                    "after_date": after[0].isoformat(),
                    "gap_days": int(total_days),
                })
                filled_count += 1

            elif before is not None:
                # Extrapolate forward (use last known value)
                interpolated.append({
                    "date": cloudy_date_str,
                    "ndvi": round(before[1], 4),
                    "method": "forward_extrapolation",
                    "before_date": before[0].isoformat(),
                })
                filled_count += 1

            elif after is not None:
                # Extrapolate backward (use next known value)
                interpolated.append({
                    "date": cloudy_date_str,
                    "ndvi": round(after[1], 4),
                    "method": "backward_extrapolation",
                    "after_date": after[0].isoformat(),
                })
                filled_count += 1

        # Quality depends on gap size and interpolation method
        quality_score = self._calculate_interpolation_quality(
            interpolated, total_gap_days, len(cloudy_dates)
        )

        result = GapFillResult(
            method="temporal_interpolation",
            input_cloud_pct=100.0,
            output_cloud_pct=0.0 if filled_count == len(cloudy_dates) else 50.0,
            pixels_filled=filled_count,
            quality_score=round(quality_score, 2),
            filled_data={"interpolated_ndvi": interpolated},
            metadata={
                "total_cloudy_dates": len(cloudy_dates),
                "dates_filled": filled_count,
                "clear_observations": len(parsed_obs),
                "avg_gap_days": (
                    round(total_gap_days / max(filled_count, 1), 1)
                ),
            },
        )
        result.provenance_hash = _compute_hash(result)
        self._gap_fill_store[result.result_id] = result

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Temporal interpolation %s: %d/%d dates filled, quality=%.1f, "
            "elapsed=%.2fms",
            result.result_id,
            filled_count,
            len(cloudy_dates),
            quality_score,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Nearest Clear Observation
    # ------------------------------------------------------------------

    def nearest_clear(
        self,
        scenes: List[Dict[str, Any]],
        target_date: str,
    ) -> GapFillResult:
        """Select the nearest cloud-free observation to the target date.

        Searches available scenes for the closest observation in time
        that has cloud cover below the clear threshold.

        Args:
            scenes: List of scene dicts with keys: date, band_data, cloud_pct.
            target_date: Target observation date (ISO 8601 YYYY-MM-DD).

        Returns:
            GapFillResult with the selected clear scene data.
        """
        start_time = time.monotonic()

        try:
            target_dt = datetime.fromisoformat(target_date)
        except (ValueError, TypeError):
            target_dt = _utcnow()

        # Find nearest clear scene
        best_scene: Optional[Dict[str, Any]] = None
        best_days_diff: float = float("inf")

        for scene in scenes:
            scene_cloud = float(scene.get("cloud_pct", 100.0))
            if scene_cloud > MAX_CLEAR_CLOUD_PCT:
                continue

            try:
                scene_dt = datetime.fromisoformat(str(scene.get("date", "")))
                days_diff = abs((scene_dt - target_dt).days)
                if days_diff < best_days_diff:
                    best_days_diff = days_diff
                    best_scene = scene
            except (ValueError, TypeError):
                continue

        if best_scene is None:
            result = GapFillResult(
                method="nearest_clear",
                input_cloud_pct=100.0,
                output_cloud_pct=100.0,
                pixels_filled=0,
                quality_score=0.0,
                filled_data={},
                metadata={
                    "target_date": target_date,
                    "error": "No clear scenes available",
                },
            )
            result.provenance_hash = _compute_hash(result)
            self._gap_fill_store[result.result_id] = result
            return result

        # Quality degrades with temporal distance
        quality_score = max(0.0, 100.0 - (best_days_diff * 2.0))

        band_data = best_scene.get("band_data", [])
        pixels_filled = sum(len(row) for row in band_data) if band_data else 0

        result = GapFillResult(
            method="nearest_clear",
            input_cloud_pct=100.0,
            output_cloud_pct=round(
                float(best_scene.get("cloud_pct", 0.0)), 2
            ),
            pixels_filled=pixels_filled,
            quality_score=round(quality_score, 2),
            filled_data={
                "band_data": band_data,
                "source_date": best_scene.get("date", ""),
                "source_cloud_pct": best_scene.get("cloud_pct", 0.0),
            },
            metadata={
                "target_date": target_date,
                "selected_date": best_scene.get("date", ""),
                "days_difference": int(best_days_diff),
            },
        )
        result.provenance_hash = _compute_hash(result)
        self._gap_fill_store[result.result_id] = result

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Nearest clear %s: selected scene from %s (%d days away), "
            "quality=%.1f, elapsed=%.2fms",
            result.result_id,
            best_scene.get("date", "unknown"),
            int(best_days_diff),
            quality_score,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Gap-Fill Quality Assessment
    # ------------------------------------------------------------------

    def assess_gap_fill_quality(
        self,
        original: Dict[str, Any],
        filled: GapFillResult,
        method: str,
    ) -> float:
        """Assess quality degradation from the gap-filling process.

        Computes a quality degradation score (0-100, 100 = perfect) based
        on the method used, temporal gap, number of pixels filled, and
        cloud cover reduction.

        Args:
            original: Original cloudy scene dict.
            filled: GapFillResult from the filling process.
            method: Method name used for filling.

        Returns:
            Quality degradation score (0-100, 100 = no degradation).
        """
        # Base quality from the fill result
        base_quality = filled.quality_score

        # Method-specific adjustment
        method_factors: Dict[str, float] = {
            "temporal_composite": 1.0,
            "sar_fusion": 0.85,
            "temporal_interpolation": 0.75,
            "nearest_clear": 0.70,
        }
        method_factor = method_factors.get(method.lower(), 0.5)

        # Cloud reduction bonus
        original_cloud = float(original.get("cloud_pct", 100.0))
        output_cloud = filled.output_cloud_pct
        cloud_reduction = max(0.0, original_cloud - output_cloud) / 100.0

        # Final quality: base * method_factor + cloud_reduction_bonus
        quality = base_quality * method_factor + cloud_reduction * 10.0
        quality = max(0.0, min(100.0, quality))

        logger.debug(
            "Gap-fill quality assessment: method=%s, base=%.1f, "
            "factor=%.2f, cloud_reduction=%.2f, final=%.1f",
            method, base_quality, method_factor, cloud_reduction, quality,
        )

        return round(quality, 2)

    # ------------------------------------------------------------------
    # Public API: Retrieval
    # ------------------------------------------------------------------

    def get_gap_fill_result(
        self, result_id: str
    ) -> Optional[GapFillResult]:
        """Retrieve a stored gap-fill result by ID.

        Args:
            result_id: Gap-fill result identifier.

        Returns:
            GapFillResult if found, else None.
        """
        return self._gap_fill_store.get(result_id)

    # ------------------------------------------------------------------
    # Internal: Cloud Persistence Region Check
    # ------------------------------------------------------------------

    def _check_cloud_persistence_region(
        self,
        lat: Optional[float],
        lon: Optional[float],
    ) -> Optional[str]:
        """Check if coordinates fall within a known cloud persistence region.

        Args:
            lat: Latitude (or None).
            lon: Longitude (or None).

        Returns:
            Region name if within a persistence region, else None.
        """
        if lat is None or lon is None:
            return None

        for region in CLOUD_PERSISTENCE_REGIONS:
            lat_min, lat_max = region["lat_range"]
            lon_min, lon_max = region["lon_range"]
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return region["region_name"]

        return None

    # ------------------------------------------------------------------
    # Internal: Median Composite Computation
    # ------------------------------------------------------------------

    def _compute_median_composite(
        self,
        scenes: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        """Compute per-pixel median composite from multiple scenes.

        Args:
            scenes: List of scene dicts with band_data (2D arrays).

        Returns:
            Tuple of (composite data dict, number of pixels composited).
        """
        if not scenes:
            return {}, 0

        # Collect all band data arrays
        all_band_data: List[List[List[float]]] = []
        for scene in scenes:
            bd = scene.get("band_data", [])
            if bd:
                all_band_data.append(bd)

        if not all_band_data:
            return {}, 0

        # Determine grid dimensions from the first scene
        rows = len(all_band_data[0])
        cols = len(all_band_data[0][0]) if rows > 0 else 0

        composite: List[List[float]] = []
        pixels_filled = 0

        for r in range(rows):
            row_values: List[float] = []
            for c in range(cols):
                pixel_stack: List[float] = []
                for band_data in all_band_data:
                    if r < len(band_data) and c < len(band_data[r]):
                        pixel_stack.append(band_data[r][c])

                if pixel_stack:
                    # Median
                    sorted_stack = sorted(pixel_stack)
                    n = len(sorted_stack)
                    if n % 2 == 0:
                        median_val = (
                            sorted_stack[n // 2 - 1] + sorted_stack[n // 2]
                        ) / 2.0
                    else:
                        median_val = sorted_stack[n // 2]
                    row_values.append(round(median_val, 4))
                    pixels_filled += 1
                else:
                    row_values.append(0.0)

            composite.append(row_values)

        return {"band_data": composite, "rows": rows, "cols": cols}, pixels_filled

    # ------------------------------------------------------------------
    # Internal: Quality Calculations
    # ------------------------------------------------------------------

    def _calculate_composite_quality(
        self,
        scenes: List[Dict[str, Any]],
        target_dt: datetime,
        window_days: int,
    ) -> float:
        """Calculate quality score for a temporal composite.

        Quality factors:
            - Number of qualifying scenes (more = better).
            - Average temporal distance from target date.
            - Average cloud cover of input scenes.

        Args:
            scenes: Qualifying scenes used for compositing.
            target_dt: Target date.
            window_days: Compositing window in days.

        Returns:
            Quality score (0-100).
        """
        if not scenes:
            return 0.0

        # Scene count factor (4+ scenes = max score)
        scene_factor = min(1.0, len(scenes) / 4.0)

        # Temporal proximity factor
        total_days_diff = 0.0
        for scene in scenes:
            try:
                scene_dt = datetime.fromisoformat(str(scene.get("date", "")))
                total_days_diff += abs((scene_dt - target_dt).days)
            except (ValueError, TypeError):
                total_days_diff += window_days

        avg_days_diff = total_days_diff / len(scenes)
        temporal_factor = max(0.0, 1.0 - (avg_days_diff / window_days))

        # Cloud factor (lower average cloud = higher quality)
        avg_cloud = sum(
            float(s.get("cloud_pct", 0.0)) for s in scenes
        ) / len(scenes)
        cloud_factor = max(0.0, 1.0 - (avg_cloud / 100.0))

        quality = (
            scene_factor * 40.0
            + temporal_factor * 35.0
            + cloud_factor * 25.0
        )

        return max(0.0, min(100.0, quality))

    def _calculate_sar_quality(
        self,
        total_pixels: int,
        forest_pixels: int,
        sar_data: Dict[str, Any],
    ) -> float:
        """Calculate quality score for SAR fusion result.

        Args:
            total_pixels: Total pixel count.
            forest_pixels: Forest-classified pixel count.
            sar_data: SAR data dict.

        Returns:
            Quality score (0-100).
        """
        if total_pixels == 0:
            return 0.0

        # SAR inherently provides all-weather capability -> base quality 70
        base_quality = 70.0

        # Pixel coverage bonus
        coverage_bonus = min(15.0, (total_pixels / 10000.0) * 15.0)

        # Data completeness
        has_vv = bool(sar_data.get("vv_band"))
        has_vh = bool(sar_data.get("vh_band"))
        completeness_bonus = 0.0
        if has_vv and has_vh:
            completeness_bonus = 15.0
        elif has_vv or has_vh:
            completeness_bonus = 7.5

        return min(100.0, base_quality + coverage_bonus + completeness_bonus)

    def _calculate_interpolation_quality(
        self,
        interpolated: List[Dict[str, Any]],
        total_gap_days: int,
        total_cloudy_dates: int,
    ) -> float:
        """Calculate quality score for temporal interpolation.

        Quality degrades with larger temporal gaps and extrapolation usage.

        Args:
            interpolated: List of interpolated observations.
            total_gap_days: Total gap days across all interpolations.
            total_cloudy_dates: Number of cloudy dates requested.

        Returns:
            Quality score (0-100).
        """
        if not interpolated or total_cloudy_dates == 0:
            return 0.0

        # Fill completion factor
        fill_rate = len(interpolated) / total_cloudy_dates
        fill_factor = fill_rate * 40.0

        # Gap size factor (smaller gaps = higher quality)
        avg_gap = total_gap_days / max(len(interpolated), 1)
        gap_factor = max(0.0, 35.0 - (avg_gap * 0.5))

        # Method quality factor (interpolation > extrapolation)
        interp_count = sum(
            1 for i in interpolated
            if i.get("method") == "linear_interpolation"
        )
        interp_rate = interp_count / max(len(interpolated), 1)
        method_factor = interp_rate * 25.0

        return max(0.0, min(100.0, fill_factor + gap_factor + method_factor))


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "CloudGapFiller",
    # Data classes
    "CloudCoverAnalysis",
    "GapFillResult",
    # Constants
    "CLOUD_PERSISTENCE_REGIONS",
    "DEFAULT_BRIGHTNESS_THRESHOLD",
    "DEFAULT_THERMAL_THRESHOLD_K",
    "DEFAULT_COMPOSITE_WINDOW_DAYS",
    "SAR_VH_FOREST_THRESHOLD_DB",
    "SAR_VH_VV_RATIO_THRESHOLD_DB",
    "MIN_CLEAR_PIXEL_PCT",
    "MAX_CLEAR_CLOUD_PCT",
]
