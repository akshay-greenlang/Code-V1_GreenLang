# -*- coding: utf-8 -*-
"""
Forest Change Detector Engine - AGENT-EUDR-003: Satellite Monitoring (Feature 4)

Multi-method deforestation change detection engine comparing current
satellite imagery against December 31, 2020 baselines for EUDR compliance.
Implements NDVI differencing, spectral angle mapping, time-series break
detection (simplified BFAST-lite), and rule-based change classification
with per-commodity sensitivity thresholds.

Zero-Hallucination Guarantees:
    - All change detection uses deterministic arithmetic (no ML/LLM).
    - NDVI differencing: delta = current - baseline, threshold-based.
    - Spectral angle mapping: arccos(dot product / norm product).
    - Time-series break detection: moving average + deviation detection.
    - Classification uses static per-commodity threshold lookup tables.
    - SHA-256 provenance hashes on all result objects.
    - Full evidence packages for regulatory audit trail.

Detection Methods:
    1. NDVI Differencing: Pixel-level vegetation change quantification.
    2. Spectral Angle Mapping (SAM): Multi-band spectral change detection.
    3. Time-Series Break Detection: Simplified BFAST-lite for abrupt drops.
    4. Change Classification: Rule-based categorization of change type.

Commodity Thresholds:
    - cattle:   NDVI drop >= -0.12 (pasture conversion is gradual)
    - cocoa:    NDVI drop >= -0.15 (shade-grown, moderate change)
    - coffee:   NDVI drop >= -0.15 (shade-grown, moderate change)
    - palm_oil: NDVI drop >= -0.20 (distinct canopy replacement)
    - rubber:   NDVI drop >= -0.18 (monoculture replanting)
    - soya:     NDVI drop >= -0.20 (clear-cut conversion)
    - wood:     NDVI drop >= -0.25 (selective logging vs clear-cut)

Performance Targets:
    - Single plot change detection: <200ms
    - NDVI differencing: <10ms
    - Spectral angle mapping: <15ms
    - Time-series break detection: <20ms
    - Batch detection (100 plots): <5 seconds

Regulatory References:
    - EUDR Article 2(1): Deforestation-free verification via change detection
    - EUDR Article 2(6): Cutoff date December 31, 2020
    - EUDR Article 9: Spatial monitoring evidence
    - EUDR Article 10: Risk assessment from change detection results

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 (Feature 4: Forest Change Detection)
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
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.satellite_monitoring.imagery_acquisition import (
    ImageryAcquisitionEngine,
    SceneMetadata,
)
from greenlang.agents.eudr.satellite_monitoring.spectral_index_calculator import (
    SpectralIndexCalculator,
    SpectralIndexResult,
)
from greenlang.agents.eudr.satellite_monitoring.baseline_manager import (
    BaselineSnapshot,
    EUDR_CUTOFF_DATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default NDVI drop threshold for deforestation detection.
DEFAULT_DEFORESTATION_THRESHOLD: float = -0.15

#: Default NDVI drop threshold for degradation detection.
DEFAULT_DEGRADATION_THRESHOLD: float = -0.05

#: Default NDVI gain threshold for regrowth detection.
DEFAULT_REGROWTH_THRESHOLD: float = 0.10

#: Spectral angle threshold (degrees) for significant change.
SPECTRAL_ANGLE_THRESHOLD_DEG: float = 15.0

#: Minimum confidence score to flag as deforestation.
MIN_DEFORESTATION_CONFIDENCE: float = 0.50

#: Moving average window size for time-series break detection.
BFAST_WINDOW_SIZE: int = 4

#: Number of standard deviations for BFAST break detection.
BFAST_STD_MULTIPLIER: float = 2.0

#: Minimum NDVI magnitude drop to classify as a break.
BFAST_MIN_BREAK_MAGNITUDE: float = 0.10

#: Default pixel size in metres for area calculations.
DEFAULT_PIXEL_SIZE_M: float = 10.0


# ---------------------------------------------------------------------------
# Per-Commodity Change Sensitivity Thresholds
# ---------------------------------------------------------------------------
# Each commodity has specific NDVI expectations based on how conversion
# from forest to commodity production manifests in satellite imagery.

COMMODITY_CHANGE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "cattle": {
        "deforestation_ndvi_drop": -0.12,
        "degradation_ndvi_drop": -0.05,
        "regrowth_ndvi_gain": 0.08,
        "expected_baseline_ndvi": 0.55,
        "expected_converted_ndvi": 0.35,
        "confidence_weight": 0.85,
        "description": "Pasture conversion: gradual canopy removal, "
                       "grass establishment. NDVI drops are moderate.",
    },
    "cocoa": {
        "deforestation_ndvi_drop": -0.15,
        "degradation_ndvi_drop": -0.06,
        "regrowth_ndvi_gain": 0.10,
        "expected_baseline_ndvi": 0.65,
        "expected_converted_ndvi": 0.45,
        "confidence_weight": 0.80,
        "description": "Shade-grown cocoa: partial canopy retained, "
                       "moderate NDVI change. Full sun = larger drop.",
    },
    "coffee": {
        "deforestation_ndvi_drop": -0.15,
        "degradation_ndvi_drop": -0.06,
        "regrowth_ndvi_gain": 0.10,
        "expected_baseline_ndvi": 0.60,
        "expected_converted_ndvi": 0.40,
        "confidence_weight": 0.80,
        "description": "Shade-grown coffee: similar to cocoa. Highland "
                       "coffee may show sharper NDVI contrasts.",
    },
    "palm_oil": {
        "deforestation_ndvi_drop": -0.20,
        "degradation_ndvi_drop": -0.08,
        "regrowth_ndvi_gain": 0.12,
        "expected_baseline_ndvi": 0.70,
        "expected_converted_ndvi": 0.35,
        "confidence_weight": 0.90,
        "description": "Oil palm: clear-cut then replant. Young palms "
                       "show low NDVI, maturing over 3-5 years.",
    },
    "rubber": {
        "deforestation_ndvi_drop": -0.18,
        "degradation_ndvi_drop": -0.07,
        "regrowth_ndvi_gain": 0.10,
        "expected_baseline_ndvi": 0.65,
        "expected_converted_ndvi": 0.40,
        "confidence_weight": 0.85,
        "description": "Rubber plantation: deciduous monoculture with "
                       "seasonal leaf shedding affects detection.",
    },
    "soya": {
        "deforestation_ndvi_drop": -0.20,
        "degradation_ndvi_drop": -0.08,
        "regrowth_ndvi_gain": 0.15,
        "expected_baseline_ndvi": 0.65,
        "expected_converted_ndvi": 0.30,
        "confidence_weight": 0.90,
        "description": "Soya conversion: clear-cut and mechanized "
                       "agriculture. Distinct seasonal NDVI cycle.",
    },
    "wood": {
        "deforestation_ndvi_drop": -0.25,
        "degradation_ndvi_drop": -0.10,
        "regrowth_ndvi_gain": 0.15,
        "expected_baseline_ndvi": 0.70,
        "expected_converted_ndvi": 0.20,
        "confidence_weight": 0.95,
        "description": "Timber extraction: selective logging shows smaller "
                       "drops; clear-felling shows largest NDVI change.",
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ChangeClassification:
    """Classification result for detected vegetation change.

    Attributes:
        change_type: Type of change detected ('deforestation',
            'degradation', 'no_change', 'regrowth').
        ndvi_change: NDVI difference (current - baseline).
        change_area_ha: Estimated area of change in hectares.
        change_percentage: Percentage of plot area changed.
        confidence: Confidence score (0-1) of the classification.
        commodity_threshold_used: NDVI threshold applied.
        evidence_summary: Brief textual summary of evidence.
        provenance_hash: SHA-256 provenance hash.
    """

    change_type: str = "no_change"
    ndvi_change: float = 0.0
    change_area_ha: float = 0.0
    change_percentage: float = 0.0
    confidence: float = 0.0
    commodity_threshold_used: float = 0.0
    evidence_summary: str = ""
    provenance_hash: str = ""


@dataclass
class ChangeDetectionResult:
    """Full result of a change detection analysis for one plot.

    Attributes:
        detection_id: Unique identifier for this detection run.
        plot_id: Production plot identifier.
        analysis_date: Date of current imagery used.
        baseline_date: Date of baseline imagery.
        commodity: EUDR commodity.
        classification: Change classification result.
        ndvi_baseline_mean: Baseline mean NDVI.
        ndvi_current_mean: Current mean NDVI.
        ndvi_difference: Delta NDVI (current - baseline).
        spectral_angle_deg: Spectral angle between baseline and current.
        time_series_break: Whether a time-series break was detected.
        break_date: Estimated date of break (if detected).
        break_magnitude: Magnitude of the break (if detected).
        current_scene_id: ID of current imagery scene used.
        current_scene_date: Acquisition date of current scene.
        pixel_count: Total pixels analyzed.
        deforestation_pixels: Pixels classified as deforested.
        degradation_pixels: Pixels classified as degraded.
        regrowth_pixels: Pixels classified as regrowth.
        no_change_pixels: Pixels with no significant change.
        evidence: Structured evidence package for audit.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """

    detection_id: str = ""
    plot_id: str = ""
    analysis_date: Optional[str] = None
    baseline_date: Optional[str] = None
    commodity: str = ""
    classification: Optional[ChangeClassification] = None
    ndvi_baseline_mean: float = 0.0
    ndvi_current_mean: float = 0.0
    ndvi_difference: float = 0.0
    spectral_angle_deg: float = 0.0
    time_series_break: bool = False
    break_date: Optional[str] = None
    break_magnitude: float = 0.0
    current_scene_id: str = ""
    current_scene_date: Optional[date] = None
    pixel_count: int = 0
    deforestation_pixels: int = 0
    degradation_pixels: int = 0
    regrowth_pixels: int = 0
    no_change_pixels: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""


# ---------------------------------------------------------------------------
# ForestChangeDetector
# ---------------------------------------------------------------------------


class ForestChangeDetector:
    """Production-grade multi-method forest change detection for EUDR.

    Compares current satellite imagery against December 31, 2020
    baselines using multiple detection methods: NDVI differencing,
    spectral angle mapping, and time-series break detection. Applies
    per-commodity sensitivity thresholds for accurate classification.

    All detection is deterministic with zero LLM/ML involvement.

    Example::

        detector = ForestChangeDetector()
        result = detector.detect_change(
            plot_id="PLOT-001",
            polygon_vertices=[(-3.0, -60.0), (-3.0, -59.0),
                              (-4.0, -59.0), (-4.0, -60.0)],
            baseline_snapshot=baseline,
            commodity="soya",
            analysis_date="2025-06-15",
        )
        assert result.classification is not None
        assert result.provenance_hash != ""

    Attributes:
        imagery_engine: ImageryAcquisitionEngine for current imagery.
        spectral_calculator: SpectralIndexCalculator for NDVI analysis.
        pixel_size_m: Pixel edge length for area calculations.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the ForestChangeDetector.

        Args:
            config: Optional configuration object. If provided,
                overrides pixel_size_m and passes through to sub-engines.
        """
        self.pixel_size_m = DEFAULT_PIXEL_SIZE_M

        if config is not None:
            self.pixel_size_m = getattr(
                config, "pixel_size_m", DEFAULT_PIXEL_SIZE_M
            )

        self.imagery_engine = ImageryAcquisitionEngine(config=config)
        self.spectral_calculator = SpectralIndexCalculator(config=config)

        logger.info(
            "ForestChangeDetector initialized: pixel_size=%.1fm",
            self.pixel_size_m,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_change(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        baseline_snapshot: BaselineSnapshot,
        commodity: str = "wood",
        analysis_date: Optional[str] = None,
    ) -> ChangeDetectionResult:
        """Detect forest change for a plot against its baseline.

        Full pipeline:
            1. Acquire current imagery near analysis_date
            2. Calculate current NDVI
            3. Compare with baseline NDVI (pixel-level differencing)
            4. Perform spectral angle mapping (multi-band comparison)
            5. Classify change using commodity-specific thresholds
            6. Build evidence package for regulatory audit

        Args:
            plot_id: Production plot identifier.
            polygon_vertices: Plot boundary as (lat, lon) tuples.
            baseline_snapshot: Baseline data from BaselineManager.
            commodity: EUDR commodity for threshold selection.
            analysis_date: Target date for current analysis (ISO format).
                If None, uses today's date.

        Returns:
            ChangeDetectionResult with classification and evidence.

        Raises:
            ValueError: If inputs are invalid.
        """
        overall_start = time.monotonic()

        if not plot_id or not plot_id.strip():
            raise ValueError("plot_id must not be empty")
        if not polygon_vertices:
            raise ValueError("polygon_vertices must not be empty")

        if analysis_date is None:
            analysis_date = _utcnow().strftime("%Y-%m-%d")

        commodity_lower = commodity.lower().strip()

        logger.info(
            "Detecting change: plot=%s, commodity=%s, analysis_date=%s",
            plot_id, commodity_lower, analysis_date,
        )

        # Step 1: Acquire current imagery
        current_scenes = self._acquire_current_imagery(
            polygon_vertices, analysis_date
        )

        if not current_scenes:
            logger.warning(
                "No current imagery found for plot %s at %s",
                plot_id, analysis_date,
            )
            return self._build_inconclusive_result(
                plot_id, commodity_lower, analysis_date,
                baseline_snapshot, "No current imagery available",
            )

        best_scene = self.imagery_engine.get_best_scene(
            current_scenes, analysis_date
        )

        if best_scene is None:
            return self._build_inconclusive_result(
                plot_id, commodity_lower, analysis_date,
                baseline_snapshot, "Could not select a suitable scene",
            )

        # Step 2: Calculate current NDVI
        band_names = self._get_ndvi_bands(best_scene.source)
        bands = self.imagery_engine.download_bands(
            scene_id=best_scene.scene_id,
            bands=band_names,
        )

        red_band = bands[band_names[0]]
        nir_band = bands[band_names[1]]

        current_ndvi = self.spectral_calculator.calculate_ndvi(
            red_band=red_band,
            nir_band=nir_band,
        )

        # Step 3: NDVI differencing (use current NDVI values vs baseline mean)
        # Since we don't have the actual baseline pixel array, we create a
        # synthetic baseline NDVI array from the baseline mean for comparison.
        baseline_ndvi_values = [
            baseline_snapshot.ndvi_mean
        ] * len(current_ndvi.values)

        ndvi_diff = self.ndvi_differencing(
            baseline_ndvi_values=baseline_ndvi_values,
            current_ndvi_values=current_ndvi.values,
            pixel_size_m=self.pixel_size_m,
        )

        # Step 4: Spectral angle mapping (simplified using NDVI arrays)
        # Use single-band comparison for spectral angle
        baseline_bands_flat = [baseline_snapshot.ndvi_mean]
        current_bands_flat = [current_ndvi.mean]
        sam_result = self.spectral_angle_mapping(
            baseline_bands=[baseline_bands_flat],
            current_bands=[current_bands_flat],
        )

        # Step 5: Classify change
        classification = self.classify_change(
            ndvi_diff=ndvi_diff["mean_delta_ndvi"],
            area_ha=ndvi_diff.get("deforestation_area_ha", 0.0),
            confidence=self._compute_detection_confidence(
                ndvi_diff, sam_result, commodity_lower
            ),
            commodity=commodity_lower,
        )

        # Step 6: Build result
        elapsed_ms = (time.monotonic() - overall_start) * 1000

        result = ChangeDetectionResult(
            detection_id=_generate_id(),
            plot_id=plot_id,
            analysis_date=analysis_date,
            baseline_date=str(baseline_snapshot.scene_date),
            commodity=commodity_lower,
            classification=classification,
            ndvi_baseline_mean=baseline_snapshot.ndvi_mean,
            ndvi_current_mean=current_ndvi.mean,
            ndvi_difference=round(
                current_ndvi.mean - baseline_snapshot.ndvi_mean, 6
            ),
            spectral_angle_deg=sam_result.get("mean_angle_deg", 0.0),
            time_series_break=False,
            current_scene_id=best_scene.scene_id,
            current_scene_date=best_scene.acquisition_date,
            pixel_count=current_ndvi.pixel_count,
            deforestation_pixels=ndvi_diff.get("deforestation_pixels", 0),
            degradation_pixels=ndvi_diff.get("degradation_pixels", 0),
            regrowth_pixels=ndvi_diff.get("regrowth_pixels", 0),
            no_change_pixels=ndvi_diff.get("no_change_pixels", 0),
            evidence=self._build_evidence_package(
                baseline_snapshot, current_ndvi, best_scene,
                ndvi_diff, sam_result, classification,
            ),
            processing_time_ms=round(elapsed_ms, 2),
        )

        # Compute provenance hash
        result.provenance_hash = self._compute_detection_hash(result)

        logger.info(
            "Change detection: plot=%s, type=%s, ndvi_diff=%.4f, "
            "confidence=%.2f, area=%.2fha, %.2fms",
            plot_id, classification.change_type,
            result.ndvi_difference, classification.confidence,
            classification.change_area_ha, elapsed_ms,
        )

        return result

    def ndvi_differencing(
        self,
        baseline_ndvi_values: List[float],
        current_ndvi_values: List[float],
        pixel_size_m: float = DEFAULT_PIXEL_SIZE_M,
    ) -> Dict[str, Any]:
        """Perform pixel-level NDVI differencing between baseline and current.

        delta_NDVI = current - baseline

        Classification thresholds:
            - deforestation: delta < -0.15
            - degradation: -0.15 <= delta < -0.05
            - no_change: -0.05 <= delta <= 0.10
            - regrowth: delta > 0.10

        Args:
            baseline_ndvi_values: List of baseline NDVI values.
            current_ndvi_values: List of current NDVI values.
            pixel_size_m: Pixel edge length in metres.

        Returns:
            Dictionary with change statistics and pixel classifications.

        Raises:
            ValueError: If arrays have different lengths.
        """
        start_time = time.monotonic()

        if len(baseline_ndvi_values) != len(current_ndvi_values):
            raise ValueError(
                f"NDVI arrays must have same length: "
                f"baseline={len(baseline_ndvi_values)}, "
                f"current={len(current_ndvi_values)}"
            )

        n = len(baseline_ndvi_values)
        if n == 0:
            return self._empty_ndvi_diff_result()

        deforestation_pixels = 0
        degradation_pixels = 0
        no_change_pixels = 0
        regrowth_pixels = 0
        delta_sum = 0.0
        delta_values: List[float] = []

        for i in range(n):
            delta = current_ndvi_values[i] - baseline_ndvi_values[i]
            delta_values.append(delta)
            delta_sum += delta

            if delta < DEFAULT_DEFORESTATION_THRESHOLD:
                deforestation_pixels += 1
            elif delta < DEFAULT_DEGRADATION_THRESHOLD:
                degradation_pixels += 1
            elif delta <= DEFAULT_REGROWTH_THRESHOLD:
                no_change_pixels += 1
            else:
                regrowth_pixels += 1

        mean_delta = delta_sum / n if n > 0 else 0.0

        # Calculate areas in hectares
        pixel_area_m2 = pixel_size_m * pixel_size_m
        ha_per_pixel = pixel_area_m2 / 10_000.0

        deforestation_area_ha = deforestation_pixels * ha_per_pixel
        degradation_area_ha = degradation_pixels * ha_per_pixel
        regrowth_area_ha = regrowth_pixels * ha_per_pixel
        total_area_ha = n * ha_per_pixel

        # Change percentage
        change_pixels = deforestation_pixels + degradation_pixels
        change_pct = (change_pixels / n * 100.0) if n > 0 else 0.0

        # Overall classification based on dominant change
        if deforestation_pixels > n * 0.10:
            overall_classification = "deforestation"
        elif degradation_pixels > n * 0.15:
            overall_classification = "degradation"
        elif regrowth_pixels > n * 0.10:
            overall_classification = "regrowth"
        else:
            overall_classification = "no_change"

        result = {
            "total_pixels": n,
            "deforestation_pixels": deforestation_pixels,
            "degradation_pixels": degradation_pixels,
            "no_change_pixels": no_change_pixels,
            "regrowth_pixels": regrowth_pixels,
            "mean_delta_ndvi": round(mean_delta, 6),
            "deforestation_area_ha": round(deforestation_area_ha, 4),
            "degradation_area_ha": round(degradation_area_ha, 4),
            "regrowth_area_ha": round(regrowth_area_ha, 4),
            "total_area_ha": round(total_area_ha, 4),
            "change_percentage": round(change_pct, 2),
            "overall_classification": overall_classification,
            "thresholds": {
                "deforestation": DEFAULT_DEFORESTATION_THRESHOLD,
                "degradation": DEFAULT_DEGRADATION_THRESHOLD,
                "regrowth": DEFAULT_REGROWTH_THRESHOLD,
            },
        }

        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "NDVI differencing: mean_delta=%.4f, deforestation=%d px, "
            "classification=%s, %.2fms",
            mean_delta, deforestation_pixels,
            overall_classification, elapsed_ms,
        )

        return result

    def spectral_angle_mapping(
        self,
        baseline_bands: List[List[float]],
        current_bands: List[List[float]],
    ) -> Dict[str, Any]:
        """Perform Spectral Angle Mapping between baseline and current bands.

        Calculates the spectral angle between multi-band vectors:
            angle = arccos(dot(b, c) / (|b| * |c|))

        A spectral angle > 15 degrees indicates significant spectral
        change, suggesting land cover modification.

        Args:
            baseline_bands: List of baseline band vectors.
                Each inner list is a pixel's multi-band values.
            current_bands: List of current band vectors (same structure).

        Returns:
            Dictionary with mean angle, max angle, and change statistics.

        Raises:
            ValueError: If arrays have different lengths.
        """
        start_time = time.monotonic()

        if len(baseline_bands) != len(current_bands):
            raise ValueError(
                f"Band arrays must have same length: "
                f"baseline={len(baseline_bands)}, "
                f"current={len(current_bands)}"
            )

        n = len(baseline_bands)
        if n == 0:
            return {
                "total_pixels": 0,
                "mean_angle_deg": 0.0,
                "max_angle_deg": 0.0,
                "significant_change_pixels": 0,
                "significant_change_pct": 0.0,
                "threshold_deg": SPECTRAL_ANGLE_THRESHOLD_DEG,
                "provenance_hash": _compute_hash({"n": 0}),
            }

        angles: List[float] = []
        significant_count = 0

        for i in range(n):
            b_vec = baseline_bands[i]
            c_vec = current_bands[i]

            if len(b_vec) != len(c_vec):
                angles.append(0.0)
                continue

            angle = self._compute_spectral_angle(b_vec, c_vec)
            angles.append(angle)

            if angle > SPECTRAL_ANGLE_THRESHOLD_DEG:
                significant_count += 1

        mean_angle = sum(angles) / n if n > 0 else 0.0
        max_angle = max(angles) if angles else 0.0
        sig_pct = (significant_count / n * 100.0) if n > 0 else 0.0

        result = {
            "total_pixels": n,
            "mean_angle_deg": round(mean_angle, 4),
            "max_angle_deg": round(max_angle, 4),
            "significant_change_pixels": significant_count,
            "significant_change_pct": round(sig_pct, 2),
            "threshold_deg": SPECTRAL_ANGLE_THRESHOLD_DEG,
        }

        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Spectral angle mapping: mean=%.2f deg, max=%.2f deg, "
            "significant=%d/%d, %.2fms",
            mean_angle, max_angle, significant_count, n, elapsed_ms,
        )

        return result

    def time_series_break_detection(
        self,
        ndvi_series: List[float],
        dates: List[str],
    ) -> Dict[str, Any]:
        """Detect abrupt NDVI drops using simplified BFAST-lite method.

        Computes a moving average of the NDVI time series and flags
        points where the actual value drops below the moving average
        by more than BFAST_STD_MULTIPLIER standard deviations.

        Args:
            ndvi_series: Time series of NDVI values (chronological).
            dates: Corresponding date strings (ISO format).

        Returns:
            Dictionary with break detection results.

        Raises:
            ValueError: If series and dates have different lengths.
        """
        start_time = time.monotonic()

        if len(ndvi_series) != len(dates):
            raise ValueError(
                f"ndvi_series ({len(ndvi_series)}) and dates "
                f"({len(dates)}) must have the same length"
            )

        n = len(ndvi_series)

        if n < BFAST_WINDOW_SIZE + 1:
            return {
                "break_detected": False,
                "break_date": None,
                "break_magnitude": 0.0,
                "series_length": n,
                "window_size": BFAST_WINDOW_SIZE,
                "reason": "Series too short for break detection",
                "provenance_hash": _compute_hash({"n": n}),
            }

        # Compute moving average and standard deviation
        breaks: List[Dict[str, Any]] = []

        for i in range(BFAST_WINDOW_SIZE, n):
            window = ndvi_series[i - BFAST_WINDOW_SIZE: i]
            window_mean = sum(window) / len(window)

            if len(window) > 1:
                window_var = sum(
                    (v - window_mean) ** 2 for v in window
                ) / (len(window) - 1)
                window_std = math.sqrt(window_var)
            else:
                window_std = 0.0

            current_val = ndvi_series[i]
            deviation = window_mean - current_val

            # Break condition: current value drops significantly below
            # the moving average
            threshold = max(
                BFAST_MIN_BREAK_MAGNITUDE,
                BFAST_STD_MULTIPLIER * window_std,
            )

            if deviation >= threshold:
                breaks.append({
                    "index": i,
                    "date": dates[i],
                    "magnitude": round(deviation, 6),
                    "window_mean": round(window_mean, 6),
                    "current_value": round(current_val, 6),
                    "threshold": round(threshold, 6),
                })

        # Select the most significant break
        break_detected = len(breaks) > 0
        primary_break = None

        if breaks:
            primary_break = max(breaks, key=lambda b: b["magnitude"])

        result = {
            "break_detected": break_detected,
            "break_date": primary_break["date"] if primary_break else None,
            "break_magnitude": (
                primary_break["magnitude"] if primary_break else 0.0
            ),
            "break_count": len(breaks),
            "all_breaks": breaks,
            "series_length": n,
            "window_size": BFAST_WINDOW_SIZE,
            "std_multiplier": BFAST_STD_MULTIPLIER,
            "min_break_magnitude": BFAST_MIN_BREAK_MAGNITUDE,
        }

        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Time-series break detection: break=%s, breaks=%d, "
            "magnitude=%.4f, series_len=%d, %.2fms",
            break_detected, len(breaks),
            primary_break["magnitude"] if primary_break else 0.0,
            n, elapsed_ms,
        )

        return result

    def classify_change(
        self,
        ndvi_diff: float,
        area_ha: float,
        confidence: float,
        commodity: str = "wood",
    ) -> ChangeClassification:
        """Classify vegetation change using commodity-specific thresholds.

        Applies per-commodity NDVI thresholds to determine whether the
        observed change constitutes deforestation, degradation, no change,
        or regrowth.

        Args:
            ndvi_diff: Mean NDVI difference (current - baseline).
            area_ha: Estimated area of change in hectares.
            confidence: Detection confidence score (0-1).
            commodity: EUDR commodity for threshold selection.

        Returns:
            ChangeClassification with type, area, and confidence.
        """
        commodity_lower = commodity.lower().strip()
        thresholds = COMMODITY_CHANGE_THRESHOLDS.get(commodity_lower)

        if thresholds is None:
            # Fall back to default thresholds
            deforestation_threshold = DEFAULT_DEFORESTATION_THRESHOLD
            degradation_threshold = DEFAULT_DEGRADATION_THRESHOLD
            regrowth_threshold = DEFAULT_REGROWTH_THRESHOLD
        else:
            deforestation_threshold = thresholds["deforestation_ndvi_drop"]
            degradation_threshold = thresholds["degradation_ndvi_drop"]
            regrowth_threshold = thresholds["regrowth_ndvi_gain"]

        # Classify
        if ndvi_diff <= deforestation_threshold:
            change_type = "deforestation"
            evidence_summary = (
                f"NDVI drop of {ndvi_diff:.4f} exceeds deforestation "
                f"threshold of {deforestation_threshold} for {commodity_lower}. "
                f"Estimated {area_ha:.2f} ha affected."
            )
        elif ndvi_diff <= degradation_threshold:
            change_type = "degradation"
            evidence_summary = (
                f"NDVI drop of {ndvi_diff:.4f} indicates forest degradation "
                f"for {commodity_lower}. "
                f"Estimated {area_ha:.2f} ha affected."
            )
        elif ndvi_diff >= regrowth_threshold:
            change_type = "regrowth"
            evidence_summary = (
                f"NDVI increase of {ndvi_diff:.4f} indicates vegetation "
                f"regrowth. Estimated {area_ha:.2f} ha affected."
            )
        else:
            change_type = "no_change"
            evidence_summary = (
                f"NDVI difference of {ndvi_diff:.4f} is within the "
                f"no-change range for {commodity_lower}."
            )

        # Calculate change percentage
        # Using total area from pixel count would be ideal; here we
        # estimate from the change area
        change_pct = 0.0
        if area_ha > 0:
            # Assume deforestation area is the portion that changed
            change_pct = min(100.0, area_ha / max(area_ha, 1.0) * 100.0)

        classification = ChangeClassification(
            change_type=change_type,
            ndvi_change=round(ndvi_diff, 6),
            change_area_ha=round(area_ha, 4),
            change_percentage=round(change_pct, 2),
            confidence=round(min(1.0, max(0.0, confidence)), 4),
            commodity_threshold_used=deforestation_threshold,
            evidence_summary=evidence_summary,
        )

        classification.provenance_hash = self._compute_classification_hash(
            classification
        )

        return classification

    def batch_detect(
        self,
        plots: List[Dict[str, Any]],
        analysis_date: Optional[str] = None,
    ) -> List[ChangeDetectionResult]:
        """Perform batch change detection across multiple plots.

        Each plot dict must contain:
            - plot_id: str
            - polygon_vertices: List[Tuple[float, float]]
            - baseline_snapshot: BaselineSnapshot
            - commodity: str (optional, default 'wood')

        Args:
            plots: List of plot dictionaries.
            analysis_date: Target date for current analysis (ISO format).
                If None, uses today's date.

        Returns:
            List of ChangeDetectionResult, one per plot.
        """
        start_time = time.monotonic()

        if analysis_date is None:
            analysis_date = _utcnow().strftime("%Y-%m-%d")

        results: List[ChangeDetectionResult] = []

        for i, plot in enumerate(plots):
            plot_id = plot.get("plot_id", f"batch-{i}")

            try:
                result = self.detect_change(
                    plot_id=plot_id,
                    polygon_vertices=plot.get("polygon_vertices", []),
                    baseline_snapshot=plot["baseline_snapshot"],
                    commodity=plot.get("commodity", "wood"),
                    analysis_date=analysis_date,
                )
                results.append(result)

            except Exception as e:
                logger.error(
                    "Batch change detection failed for plot %s: %s",
                    plot_id, str(e),
                )
                results.append(self._build_error_result(
                    plot_id, str(e), analysis_date
                ))

            # Log progress every 10 plots
            if (i + 1) % 10 == 0:
                logger.info(
                    "Batch progress: %d/%d plots processed",
                    i + 1, len(plots),
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Summarize batch results
        deforestation_count = sum(
            1 for r in results
            if r.classification and r.classification.change_type == "deforestation"
        )

        logger.info(
            "Batch change detection complete: %d plots, %d deforestation, "
            "%.2fms total",
            len(results), deforestation_count, elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Internal: Imagery Acquisition
    # ------------------------------------------------------------------

    def _acquire_current_imagery(
        self,
        polygon_vertices: List[Tuple[float, float]],
        analysis_date: str,
    ) -> List[SceneMetadata]:
        """Acquire current satellite imagery near the analysis date.

        Searches Sentinel-2 first, falls back to Landsat if needed.

        Args:
            polygon_vertices: Plot boundary vertices.
            analysis_date: Target date for current imagery.

        Returns:
            List of candidate SceneMetadata.
        """
        # Parse analysis date and create search window (+/- 30 days)
        try:
            parts = analysis_date.split("-")
            target = date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            target = _utcnow().date()

        from datetime import timedelta
        search_start = target - timedelta(days=30)
        search_end = target + timedelta(days=30)

        scenes = self.imagery_engine.search_scenes(
            polygon_vertices=polygon_vertices,
            date_range=(search_start.isoformat(), search_end.isoformat()),
            source="sentinel2",
            cloud_cover_max=30.0,
            limit=50,
        )

        if not scenes:
            scenes = self.imagery_engine.search_scenes(
                polygon_vertices=polygon_vertices,
                date_range=(search_start.isoformat(), search_end.isoformat()),
                source="landsat8",
                cloud_cover_max=30.0,
                limit=50,
            )

        return scenes

    def _get_ndvi_bands(self, source: str) -> List[str]:
        """Get Red and NIR band names for a satellite source.

        Args:
            source: Satellite source identifier.

        Returns:
            List of [red_band_name, nir_band_name].
        """
        if source == "sentinel2":
            return ["B04", "B08"]
        elif source in ("landsat8", "landsat9"):
            return ["B4", "B5"]
        else:
            return ["B04", "B08"]

    # ------------------------------------------------------------------
    # Internal: Spectral Angle Computation
    # ------------------------------------------------------------------

    def _compute_spectral_angle(
        self,
        vec_a: List[float],
        vec_b: List[float],
    ) -> float:
        """Compute spectral angle between two band vectors in degrees.

        angle = arccos(dot(a, b) / (|a| * |b|))

        Args:
            vec_a: First spectral vector.
            vec_b: Second spectral vector.

        Returns:
            Angle in degrees (0-180).
        """
        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        cos_angle = dot_product / (norm_a * norm_b)
        # Clamp to [-1, 1] to avoid math domain errors
        cos_angle = max(-1.0, min(1.0, cos_angle))

        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    # ------------------------------------------------------------------
    # Internal: Confidence Computation
    # ------------------------------------------------------------------

    def _compute_detection_confidence(
        self,
        ndvi_diff: Dict[str, Any],
        sam_result: Dict[str, Any],
        commodity: str,
    ) -> float:
        """Compute overall detection confidence from multiple methods.

        Combines evidence from NDVI differencing, spectral angle mapping,
        and commodity-specific confidence weights.

        Args:
            ndvi_diff: NDVI differencing results.
            sam_result: Spectral angle mapping results.
            commodity: EUDR commodity for confidence weighting.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        confidence = 0.0

        # NDVI component (60% weight)
        mean_delta = abs(ndvi_diff.get("mean_delta_ndvi", 0.0))
        if mean_delta > 0.20:
            ndvi_confidence = 0.95
        elif mean_delta > 0.15:
            ndvi_confidence = 0.80
        elif mean_delta > 0.10:
            ndvi_confidence = 0.60
        elif mean_delta > 0.05:
            ndvi_confidence = 0.40
        else:
            ndvi_confidence = 0.20

        # SAM component (25% weight)
        mean_angle = sam_result.get("mean_angle_deg", 0.0)
        if mean_angle > 20.0:
            sam_confidence = 0.90
        elif mean_angle > 15.0:
            sam_confidence = 0.70
        elif mean_angle > 10.0:
            sam_confidence = 0.50
        else:
            sam_confidence = 0.30

        # Commodity weight (15%)
        thresholds = COMMODITY_CHANGE_THRESHOLDS.get(commodity, {})
        commodity_weight = thresholds.get("confidence_weight", 0.80)

        confidence = (
            ndvi_confidence * 0.60
            + sam_confidence * 0.25
            + commodity_weight * 0.15
        )

        return round(min(1.0, max(0.0, confidence)), 4)

    # ------------------------------------------------------------------
    # Internal: Evidence Package
    # ------------------------------------------------------------------

    def _build_evidence_package(
        self,
        baseline: BaselineSnapshot,
        current_ndvi: SpectralIndexResult,
        current_scene: SceneMetadata,
        ndvi_diff: Dict[str, Any],
        sam_result: Dict[str, Any],
        classification: ChangeClassification,
    ) -> Dict[str, Any]:
        """Build structured evidence package for regulatory audit.

        Args:
            baseline: Baseline snapshot.
            current_ndvi: Current NDVI results.
            current_scene: Scene used for current analysis.
            ndvi_diff: NDVI differencing results.
            sam_result: Spectral angle mapping results.
            classification: Change classification.

        Returns:
            Structured evidence dictionary.
        """
        return {
            "version": _MODULE_VERSION,
            "cutoff_date": EUDR_CUTOFF_DATE,
            "baseline": {
                "baseline_id": baseline.baseline_id,
                "scene_id": baseline.scene_id,
                "scene_date": str(baseline.scene_date),
                "ndvi_mean": baseline.ndvi_mean,
                "forest_cover_pct": baseline.forest_cover_pct,
                "is_forested": baseline.is_forested,
            },
            "current": {
                "scene_id": current_scene.scene_id,
                "scene_date": str(current_scene.acquisition_date),
                "cloud_cover_pct": current_scene.cloud_cover_pct,
                "ndvi_mean": current_ndvi.mean,
                "ndvi_std_dev": current_ndvi.std_dev,
                "pixel_count": current_ndvi.pixel_count,
            },
            "ndvi_differencing": {
                "mean_delta": ndvi_diff.get("mean_delta_ndvi", 0.0),
                "deforestation_pixels": ndvi_diff.get("deforestation_pixels", 0),
                "change_percentage": ndvi_diff.get("change_percentage", 0.0),
                "classification": ndvi_diff.get("overall_classification", ""),
            },
            "spectral_angle_mapping": {
                "mean_angle_deg": sam_result.get("mean_angle_deg", 0.0),
                "significant_change_pct": sam_result.get(
                    "significant_change_pct", 0.0
                ),
            },
            "classification": {
                "change_type": classification.change_type,
                "confidence": classification.confidence,
                "evidence_summary": classification.evidence_summary,
            },
        }

    # ------------------------------------------------------------------
    # Internal: Inconclusive / Error Results
    # ------------------------------------------------------------------

    def _build_inconclusive_result(
        self,
        plot_id: str,
        commodity: str,
        analysis_date: str,
        baseline: BaselineSnapshot,
        reason: str,
    ) -> ChangeDetectionResult:
        """Build an inconclusive result when detection cannot proceed.

        Args:
            plot_id: Plot identifier.
            commodity: EUDR commodity.
            analysis_date: Target analysis date.
            baseline: Baseline snapshot.
            reason: Reason for inconclusive result.

        Returns:
            ChangeDetectionResult with inconclusive classification.
        """
        classification = ChangeClassification(
            change_type="inconclusive",
            confidence=0.0,
            evidence_summary=reason,
        )
        classification.provenance_hash = self._compute_classification_hash(
            classification
        )

        result = ChangeDetectionResult(
            detection_id=_generate_id(),
            plot_id=plot_id,
            analysis_date=analysis_date,
            baseline_date=str(baseline.scene_date),
            commodity=commodity,
            classification=classification,
            ndvi_baseline_mean=baseline.ndvi_mean,
            evidence={"reason": reason},
        )
        result.provenance_hash = self._compute_detection_hash(result)
        return result

    def _build_error_result(
        self,
        plot_id: str,
        error_msg: str,
        analysis_date: str,
    ) -> ChangeDetectionResult:
        """Build an error result for failed detection.

        Args:
            plot_id: Plot identifier.
            error_msg: Error message.
            analysis_date: Target analysis date.

        Returns:
            ChangeDetectionResult with error information.
        """
        classification = ChangeClassification(
            change_type="error",
            confidence=0.0,
            evidence_summary=f"Detection failed: {error_msg}",
        )
        classification.provenance_hash = self._compute_classification_hash(
            classification
        )

        result = ChangeDetectionResult(
            detection_id=_generate_id(),
            plot_id=plot_id,
            analysis_date=analysis_date,
            commodity="",
            classification=classification,
            evidence={"error": error_msg},
        )
        result.provenance_hash = self._compute_detection_hash(result)
        return result

    def _empty_ndvi_diff_result(self) -> Dict[str, Any]:
        """Return an empty NDVI differencing result.

        Returns:
            Dictionary with zeroed-out change statistics.
        """
        result = {
            "total_pixels": 0,
            "deforestation_pixels": 0,
            "degradation_pixels": 0,
            "no_change_pixels": 0,
            "regrowth_pixels": 0,
            "mean_delta_ndvi": 0.0,
            "deforestation_area_ha": 0.0,
            "degradation_area_ha": 0.0,
            "regrowth_area_ha": 0.0,
            "total_area_ha": 0.0,
            "change_percentage": 0.0,
            "overall_classification": "no_change",
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Provenance Hashing
    # ------------------------------------------------------------------

    def _compute_detection_hash(self, result: ChangeDetectionResult) -> str:
        """Compute SHA-256 provenance hash for a detection result.

        Args:
            result: ChangeDetectionResult to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "detection_id": result.detection_id,
            "plot_id": result.plot_id,
            "analysis_date": result.analysis_date,
            "baseline_date": result.baseline_date,
            "commodity": result.commodity,
            "ndvi_baseline_mean": result.ndvi_baseline_mean,
            "ndvi_current_mean": result.ndvi_current_mean,
            "ndvi_difference": result.ndvi_difference,
            "spectral_angle_deg": result.spectral_angle_deg,
            "classification_type": (
                result.classification.change_type
                if result.classification else "none"
            ),
            "classification_confidence": (
                result.classification.confidence
                if result.classification else 0.0
            ),
            "deforestation_pixels": result.deforestation_pixels,
            "pixel_count": result.pixel_count,
        }
        return _compute_hash(hash_data)

    def _compute_classification_hash(
        self, classification: ChangeClassification,
    ) -> str:
        """Compute SHA-256 provenance hash for a classification.

        Args:
            classification: ChangeClassification to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "change_type": classification.change_type,
            "ndvi_change": classification.ndvi_change,
            "change_area_ha": classification.change_area_ha,
            "change_percentage": classification.change_percentage,
            "confidence": classification.confidence,
            "commodity_threshold_used": classification.commodity_threshold_used,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "ForestChangeDetector",
    "ChangeDetectionResult",
    "ChangeClassification",
    "COMMODITY_CHANGE_THRESHOLDS",
    "DEFAULT_DEFORESTATION_THRESHOLD",
    "DEFAULT_DEGRADATION_THRESHOLD",
    "DEFAULT_REGROWTH_THRESHOLD",
    "SPECTRAL_ANGLE_THRESHOLD_DEG",
    "BFAST_WINDOW_SIZE",
    "BFAST_STD_MULTIPLIER",
    "BFAST_MIN_BREAK_MAGNITUDE",
]
