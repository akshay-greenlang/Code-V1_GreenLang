# -*- coding: utf-8 -*-
"""
Canopy Density Mapper Engine - AGENT-EUDR-004: Forest Cover Analysis (Feature 1)

Quantifies tree canopy cover percentage across geospatial plots using four
deterministic density mapping methods: linear spectral unmixing, biome-
calibrated NDVI-to-canopy regression, the dimidiation fractional vegetation
cover model, and sub-pixel texture-based detection for sparse canopies.

Zero-Hallucination Guarantees:
    - All calculations use deterministic float arithmetic (no ML/LLM).
    - Linear spectral unmixing: bounded least-squares with sum-to-one
      constraint (no negative fractions).
    - NDVI regression: static biome-calibrated coefficients, clipped [0, 100].
    - Dimidiation model: FVC from static NDVI_soil and NDVI_veg references.
    - Sub-pixel detection: GLCM contrast/homogeneity from pixel windows.
    - SHA-256 provenance hashes on all result objects.
    - No ML/LLM used for any numeric computation.

Density Mapping Methods:
    1. Linear Spectral Unmixing (LSU): Endmembers (forest, soil, water,
       impervious) with bounded least-squares to find sub-pixel fractions.
    2. NDVI Canopy Regression: Biome-specific linear regression
       canopy_pct = a * NDVI + b, clipped to [0, 100].
    3. Dimidiation Model: FVC = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil).
    4. Sub-Pixel Detection: GLCM texture analysis for sparse canopies (<30%).

Canopy Density Classes:
    VERY_HIGH: > 80%     (closed canopy, primary tropical forest)
    HIGH:      60-80%    (mature secondary forest, dense plantation)
    MODERATE:  40-60%    (open forest, mixed land use)
    LOW:       20-40%    (degraded forest, agroforestry)
    SPARSE:    10-20%    (woodland, sparse tree cover)
    OPEN:      < 10%     (grassland, cropland, bare soil)

FAO Forest Threshold:
    A plot qualifies as "forest" under FAO/EUDR definitions if canopy
    density >= 10% AND contiguous area >= 0.5 hectares.

Performance Targets:
    - Single plot analysis (64x64 grid): <50ms
    - Linear spectral unmixing: <20ms
    - NDVI regression: <5ms
    - Dimidiation model: <5ms
    - Sub-pixel detection: <15ms
    - Batch analysis (100 plots): <3 seconds

Regulatory References:
    - EUDR Article 2(1): Deforestation-free verification requires canopy
      density quantification to assess forest status.
    - EUDR Article 2(4): Forest definition aligns with FAO (>10% canopy,
      >5m height, >0.5ha area).
    - EUDR Article 9: Geolocation-based spatial analysis.
    - EUDR Article 10: Risk assessment evidence from density mapping.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Feature 1: Canopy Density Mapping)
Agent ID: GL-EUDR-FCA-004
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
# Enumerations
# ---------------------------------------------------------------------------


class CanopyDensityClass(str, Enum):
    """Canopy density classification following EUDR/FAO tiers.

    Thresholds are based on FAO Global Forest Resources Assessment
    (FRA 2020) canopy cover classes, extended with EUDR-specific
    OPEN class for non-forest detection.
    """

    VERY_HIGH = "VERY_HIGH"    # > 80%
    HIGH = "HIGH"              # 60-80%
    MODERATE = "MODERATE"      # 40-60%
    LOW = "LOW"                # 20-40%
    SPARSE = "SPARSE"          # 10-20%
    OPEN = "OPEN"              # < 10%


class DensityMappingMethod(str, Enum):
    """Available canopy density mapping methods."""

    LINEAR_SPECTRAL_UNMIXING = "LINEAR_SPECTRAL_UNMIXING"
    NDVI_CANOPY_REGRESSION = "NDVI_CANOPY_REGRESSION"
    DIMIDIATION_MODEL = "DIMIDIATION_MODEL"
    SUB_PIXEL_DETECTION = "SUB_PIXEL_DETECTION"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: FAO minimum canopy cover percentage for forest classification.
FAO_CANOPY_THRESHOLD_PCT: float = 10.0

#: FAO minimum contiguous forest area in hectares.
FAO_AREA_THRESHOLD_HA: float = 0.5

#: Default pixel size in metres (Sentinel-2 10m band).
DEFAULT_PIXEL_SIZE_M: float = 10.0

#: Default degradation threshold for canopy density change.
DEFAULT_DEGRADATION_THRESHOLD_PCT: float = 30.0

#: GLCM window size for texture analysis.
GLCM_WINDOW_SIZE: int = 5

#: Sparse canopy threshold for sub-pixel detection trigger.
SPARSE_CANOPY_THRESHOLD_PCT: float = 30.0

# ---------------------------------------------------------------------------
# Reference spectral endmembers for Linear Spectral Unmixing
# ---------------------------------------------------------------------------
# Each endmember is a list of reflectance values for 6 spectral bands:
#   [Blue, Green, Red, NIR, SWIR1, SWIR2]
# Derived from USGS Spectral Library Version 7 (Kokaly et al., 2017).

ENDMEMBER_LIBRARY: Dict[str, List[float]] = {
    "forest": [0.02, 0.04, 0.03, 0.40, 0.18, 0.08],
    "soil": [0.10, 0.12, 0.15, 0.22, 0.28, 0.25],
    "water": [0.05, 0.04, 0.03, 0.02, 0.01, 0.01],
    "impervious": [0.12, 0.12, 0.13, 0.16, 0.20, 0.18],
}

#: Endmember names in fixed order (determines fraction index mapping).
ENDMEMBER_NAMES: List[str] = ["forest", "soil", "water", "impervious"]

#: Number of spectral bands used for unmixing.
NUM_BANDS: int = 6

#: Number of endmembers.
NUM_ENDMEMBERS: int = 4

# ---------------------------------------------------------------------------
# Biome-calibrated NDVI-to-canopy regression coefficients
# ---------------------------------------------------------------------------
# Each biome has (a, b) where: canopy_pct = a * NDVI + b
# Coefficients calibrated against MODIS VCF (Vegetation Continuous Fields)
# product (MOD44B) and field validation data.
# Reference: Sexton et al. (2013), DiMiceli et al. (2011).

BIOME_REGRESSION_COEFFICIENTS: Dict[str, Tuple[float, float]] = {
    "tropical_rainforest":     (125.0, -5.0),
    "tropical_moist_forest":   (120.0, -3.0),
    "tropical_dry_forest":     (130.0, -10.0),
    "temperate_broadleaf":     (115.0, -2.0),
    "temperate_coniferous":    (110.0, -1.0),
    "temperate_deciduous":     (118.0, -4.0),
    "boreal_forest":           (120.0, -5.0),
    "mangrove":                (115.0, -3.0),
    "cerrado_savanna":         (135.0, -12.0),
    "tropical_savanna":        (130.0, -10.0),
    "woodland_savanna":        (128.0, -8.0),
    "montane_cloud_forest":    (118.0, -2.0),
    "montane_dry_forest":      (125.0, -6.0),
    "peat_swamp_forest":       (110.0, -1.0),
    "dry_woodland":            (140.0, -15.0),
    "thorn_forest":            (145.0, -18.0),
}

# ---------------------------------------------------------------------------
# Biome-specific NDVI reference values for dimidiation model
# ---------------------------------------------------------------------------
# Each biome has (NDVI_soil, NDVI_veg) where:
#   FVC = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)
# NDVI_soil: bare soil NDVI typical for the biome region.
# NDVI_veg: dense vegetation NDVI typical for full canopy.
# Reference: Gutman & Ignatov (1998), Carlson & Ripley (1997).

BIOME_NDVI_REFERENCES: Dict[str, Tuple[float, float]] = {
    "tropical_rainforest":     (0.05, 0.85),
    "tropical_moist_forest":   (0.06, 0.82),
    "tropical_dry_forest":     (0.08, 0.75),
    "temperate_broadleaf":     (0.07, 0.80),
    "temperate_coniferous":    (0.06, 0.78),
    "temperate_deciduous":     (0.07, 0.78),
    "boreal_forest":           (0.05, 0.72),
    "mangrove":                (0.04, 0.75),
    "cerrado_savanna":         (0.10, 0.70),
    "tropical_savanna":        (0.09, 0.68),
    "woodland_savanna":        (0.08, 0.72),
    "montane_cloud_forest":    (0.06, 0.80),
    "montane_dry_forest":      (0.08, 0.74),
    "peat_swamp_forest":       (0.04, 0.78),
    "dry_woodland":            (0.12, 0.65),
    "thorn_forest":            (0.14, 0.60),
}

# ---------------------------------------------------------------------------
# Method reliability weights (used for confidence scoring)
# ---------------------------------------------------------------------------

METHOD_RELIABILITY: Dict[str, float] = {
    DensityMappingMethod.LINEAR_SPECTRAL_UNMIXING.value: 0.85,
    DensityMappingMethod.NDVI_CANOPY_REGRESSION.value: 0.75,
    DensityMappingMethod.DIMIDIATION_MODEL.value: 0.78,
    DensityMappingMethod.SUB_PIXEL_DETECTION.value: 0.70,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CanopyDensityResult:
    """Result of canopy density analysis for a single plot.

    Attributes:
        result_id: Unique identifier for this result.
        plot_id: Identifier of the analyzed plot.
        density_pct: Estimated canopy density as a percentage [0, 100].
        density_class: Classified density tier (VERY_HIGH through OPEN).
        method_used: Density mapping method employed.
        meets_fao_forest: Whether the plot meets FAO forest criteria
            (>= 10% canopy AND >= 0.5ha area).
        pixel_densities: Per-pixel canopy density percentages.
        mean_ndvi: Mean NDVI value across the plot (if applicable).
        forest_fraction: Sub-pixel forest fraction from unmixing
            (if LSU method was used).
        confidence: Confidence score [0.0, 1.0] based on data quality,
            method reliability, and cloud cover.
        cloud_cover_pct: Cloud cover percentage in the imagery.
        resolution_m: Spatial resolution of input imagery in metres.
        biome: Biome type used for calibration.
        area_ha: Plot area in hectares.
        processing_time_ms: Time taken for analysis in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp of analysis.
        metadata: Additional contextual fields.
    """

    result_id: str = ""
    plot_id: str = ""
    density_pct: float = 0.0
    density_class: str = ""
    method_used: str = ""
    meets_fao_forest: bool = False
    pixel_densities: List[float] = field(default_factory=list)
    mean_ndvi: float = 0.0
    forest_fraction: float = 0.0
    confidence: float = 0.0
    cloud_cover_pct: float = 0.0
    resolution_m: float = 10.0
    biome: str = ""
    area_ha: float = 0.0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "result_id": self.result_id,
            "plot_id": self.plot_id,
            "density_pct": self.density_pct,
            "density_class": self.density_class,
            "method_used": self.method_used,
            "meets_fao_forest": self.meets_fao_forest,
            "mean_ndvi": self.mean_ndvi,
            "forest_fraction": self.forest_fraction,
            "confidence": self.confidence,
            "cloud_cover_pct": self.cloud_cover_pct,
            "resolution_m": self.resolution_m,
            "biome": self.biome,
            "area_ha": self.area_ha,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class PlotInput:
    """Input data for a single plot to be analyzed.

    Attributes:
        plot_id: Unique identifier for the plot.
        ndvi_values: 2D grid of NDVI values (rows x cols).
        multi_band_reflectance: Optional 3D data [band][row][col] for
            unmixing. Expected band order: Blue, Green, Red, NIR, SWIR1, SWIR2.
        biome: Biome type for calibration.
        cloud_cover_pct: Cloud cover percentage in the imagery.
        resolution_m: Spatial resolution of input imagery in metres.
        area_ha: Plot area in hectares.
        canopy_height_m: Optional mean canopy height in metres (for FAO).
    """

    plot_id: str = ""
    ndvi_values: List[List[float]] = field(default_factory=list)
    multi_band_reflectance: List[List[List[float]]] = field(
        default_factory=list
    )
    biome: str = "tropical_rainforest"
    cloud_cover_pct: float = 0.0
    resolution_m: float = 10.0
    area_ha: float = 1.0
    canopy_height_m: Optional[float] = None


# ---------------------------------------------------------------------------
# CanopyDensityMapper
# ---------------------------------------------------------------------------


class CanopyDensityMapper:
    """Production-grade canopy density mapping engine for EUDR compliance.

    Quantifies tree canopy cover percentage using four deterministic
    methods: linear spectral unmixing, NDVI-to-canopy regression,
    fractional vegetation cover (dimidiation model), and sub-pixel
    texture-based detection. All computations are zero-hallucination
    with full SHA-256 provenance tracking.

    Example::

        mapper = CanopyDensityMapper()
        plot = PlotInput(
            plot_id="plot-001",
            ndvi_values=[[0.6, 0.7], [0.5, 0.65]],
            biome="tropical_rainforest",
            area_ha=2.0,
        )
        result = mapper.analyze_plot(plot)
        assert 0.0 <= result.density_pct <= 100.0
        assert result.provenance_hash != ""

    Attributes:
        config: Optional configuration object.
        default_method: Default density mapping method.
    """

    def __init__(
        self,
        config: Any = None,
        default_method: DensityMappingMethod = (
            DensityMappingMethod.NDVI_CANOPY_REGRESSION
        ),
    ) -> None:
        """Initialize the CanopyDensityMapper.

        Args:
            config: Optional configuration object with overrides.
            default_method: Default mapping method to use when not
                specified in analyze_plot.
        """
        self.config = config
        self.default_method = default_method

        logger.info(
            "CanopyDensityMapper initialized: default_method=%s, "
            "module_version=%s",
            self.default_method.value,
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API: Main Entry Points
    # ------------------------------------------------------------------

    def analyze_plot(
        self,
        plot: PlotInput,
        method: Optional[DensityMappingMethod] = None,
    ) -> CanopyDensityResult:
        """Analyze a single plot to determine canopy density.

        Runs the configured mapping method and returns a comprehensive
        result with density percentage, classification, FAO forest
        check, and provenance hash.

        Args:
            plot: Input data for the plot including NDVI values and/or
                multi-band reflectance.
            method: Mapping method to use. Defaults to self.default_method.

        Returns:
            CanopyDensityResult with density percentage, classification,
            confidence, and provenance hash.

        Raises:
            ValueError: If plot_id is empty or NDVI values are missing.
        """
        start_time = time.monotonic()
        selected_method = method or self.default_method

        self._validate_plot_input(plot, selected_method)

        result_id = _generate_id()
        timestamp = _utcnow().isoformat()

        # Dispatch to the appropriate mapping method
        density_pct, pixel_densities, method_meta = self._dispatch_method(
            plot, selected_method,
        )

        # Classify density
        density_class = self.classify_density(density_pct)

        # Check FAO forest threshold
        meets_fao = self.check_fao_forest_threshold(
            density_pct=density_pct,
            area_ha=plot.area_ha,
            canopy_height_m=plot.canopy_height_m,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            method=selected_method,
            cloud_cover_pct=plot.cloud_cover_pct,
            resolution_m=plot.resolution_m,
            pixel_count=len(pixel_densities),
        )

        # Compute mean NDVI from the plot
        mean_ndvi = self._compute_mean_ndvi(plot.ndvi_values)

        # Extract forest fraction from metadata if LSU was used
        forest_fraction = method_meta.get("forest_fraction", 0.0)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = CanopyDensityResult(
            result_id=result_id,
            plot_id=plot.plot_id,
            density_pct=round(density_pct, 2),
            density_class=density_class.value,
            method_used=selected_method.value,
            meets_fao_forest=meets_fao,
            pixel_densities=pixel_densities,
            mean_ndvi=round(mean_ndvi, 6),
            forest_fraction=round(forest_fraction, 4),
            confidence=round(confidence, 4),
            cloud_cover_pct=plot.cloud_cover_pct,
            resolution_m=plot.resolution_m,
            biome=plot.biome,
            area_ha=plot.area_ha,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=timestamp,
            metadata=method_meta,
        )

        # Compute provenance hash
        result.provenance_hash = self._compute_result_hash(result)

        logger.info(
            "Canopy density analyzed: plot=%s, density=%.1f%%, "
            "class=%s, fao_forest=%s, confidence=%.2f, method=%s, "
            "%.2fms",
            plot.plot_id,
            density_pct,
            density_class.value,
            meets_fao,
            confidence,
            selected_method.value,
            elapsed_ms,
        )

        return result

    def batch_analyze(
        self,
        plots: List[PlotInput],
        method: Optional[DensityMappingMethod] = None,
    ) -> List[CanopyDensityResult]:
        """Analyze multiple plots for canopy density.

        Processes each plot sequentially. For I/O-bound workloads,
        consider using async batch processing at the orchestration layer.

        Args:
            plots: List of plot inputs to analyze.
            method: Mapping method to use for all plots. If None, uses
                self.default_method.

        Returns:
            List of CanopyDensityResult objects, one per input plot.

        Raises:
            ValueError: If plots list is empty.
        """
        if not plots:
            raise ValueError("plots list must not be empty")

        start_time = time.monotonic()
        results: List[CanopyDensityResult] = []

        for i, plot in enumerate(plots):
            try:
                result = self.analyze_plot(plot, method=method)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "batch_analyze: failed on plot[%d] id=%s: %s",
                    i, plot.plot_id, str(exc),
                )
                # Create an error result instead of failing the batch
                error_result = self._create_error_result(
                    plot=plot,
                    method=method or self.default_method,
                    error_msg=str(exc),
                )
                results.append(error_result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        successful = sum(1 for r in results if r.confidence > 0.0)

        logger.info(
            "batch_analyze complete: %d/%d successful, %.2fms total",
            successful, len(plots), elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Public API: Density Classification
    # ------------------------------------------------------------------

    def classify_density(self, density_pct: float) -> CanopyDensityClass:
        """Classify a canopy density percentage into a density class.

        Classification thresholds:
            VERY_HIGH: > 80%
            HIGH:      60-80%
            MODERATE:  40-60%
            LOW:       20-40%
            SPARSE:    10-20%
            OPEN:      < 10%

        Args:
            density_pct: Canopy density as a percentage [0, 100].

        Returns:
            CanopyDensityClass enum value.
        """
        if density_pct > 80.0:
            return CanopyDensityClass.VERY_HIGH
        if density_pct > 60.0:
            return CanopyDensityClass.HIGH
        if density_pct > 40.0:
            return CanopyDensityClass.MODERATE
        if density_pct > 20.0:
            return CanopyDensityClass.LOW
        if density_pct >= 10.0:
            return CanopyDensityClass.SPARSE
        return CanopyDensityClass.OPEN

    # ------------------------------------------------------------------
    # Public API: FAO Forest Threshold Check
    # ------------------------------------------------------------------

    def check_fao_forest_threshold(
        self,
        density_pct: float,
        area_ha: float,
        canopy_height_m: Optional[float] = None,
    ) -> bool:
        """Check if a plot meets the FAO/EUDR forest definition.

        FAO criteria (also adopted by EUDR Article 2(4)):
            1. Canopy cover >= 10%
            2. Contiguous area >= 0.5 hectares
            3. (Optional) Tree height potential >= 5 metres at maturity

        Args:
            density_pct: Canopy density percentage.
            area_ha: Plot area in hectares.
            canopy_height_m: Optional measured canopy height. If provided
                and < 5m, the plot does not qualify even if density and
                area thresholds are met.

        Returns:
            True if the plot meets FAO forest criteria, False otherwise.
        """
        if density_pct < FAO_CANOPY_THRESHOLD_PCT:
            return False

        if area_ha < FAO_AREA_THRESHOLD_HA:
            return False

        # If canopy height data is available, apply the 5m threshold
        if canopy_height_m is not None and canopy_height_m < 5.0:
            return False

        return True

    # ------------------------------------------------------------------
    # Public API: Mapping Methods
    # ------------------------------------------------------------------

    def linear_spectral_unmixing(
        self,
        multi_band: List[List[List[float]]],
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Estimate sub-pixel canopy fraction via linear spectral unmixing.

        Decomposes each pixel's multi-band reflectance into fractions of
        reference endmembers (forest, soil, water, impervious) using
        bounded least-squares. The forest fraction directly yields
        canopy density percentage.

        Linear Mixing Model:
            R_observed = f_forest * R_forest + f_soil * R_soil
                       + f_water * R_water + f_impervious * R_impervious
            Subject to: sum(fractions) = 1.0, fractions >= 0.0

        Args:
            multi_band: 3D array [band][row][col] with 6 spectral bands
                (Blue, Green, Red, NIR, SWIR1, SWIR2).

        Returns:
            Tuple of (pixel_densities, metadata) where pixel_densities
            is a list of canopy density percentages and metadata contains
            per-endmember mean fractions.

        Raises:
            ValueError: If multi_band does not have exactly 6 bands or
                bands have inconsistent dimensions.
        """
        start_time = time.monotonic()

        if len(multi_band) != NUM_BANDS:
            raise ValueError(
                f"Expected {NUM_BANDS} bands, got {len(multi_band)}"
            )

        # Validate band dimensions
        rows = len(multi_band[0])
        cols = len(multi_band[0][0]) if rows > 0 else 0
        for b in range(NUM_BANDS):
            if len(multi_band[b]) != rows:
                raise ValueError(
                    f"Band {b} has {len(multi_band[b])} rows, "
                    f"expected {rows}"
                )
            for r in range(rows):
                if len(multi_band[b][r]) != cols:
                    raise ValueError(
                        f"Band {b} row {r} has {len(multi_band[b][r])} cols, "
                        f"expected {cols}"
                    )

        # Build endmember matrix as list of band-vectors
        endmember_matrix = [
            ENDMEMBER_LIBRARY[name] for name in ENDMEMBER_NAMES
        ]

        pixel_densities: List[float] = []
        fraction_sums: Dict[str, float] = {
            name: 0.0 for name in ENDMEMBER_NAMES
        }
        total_pixels = 0

        for r in range(rows):
            for c in range(cols):
                # Extract pixel reflectance vector
                pixel_reflectance = [
                    multi_band[b][r][c] for b in range(NUM_BANDS)
                ]

                # Solve for fractions using bounded least-squares
                fractions = self._solve_unmixing(
                    pixel_reflectance, endmember_matrix,
                )

                # Forest fraction is the first endmember
                forest_frac = fractions[0]
                density = max(0.0, min(100.0, forest_frac * 100.0))
                pixel_densities.append(round(density, 2))

                for i, name in enumerate(ENDMEMBER_NAMES):
                    fraction_sums[name] += fractions[i]

                total_pixels += 1

        # Compute mean fractions
        mean_fractions: Dict[str, float] = {}
        if total_pixels > 0:
            for name in ENDMEMBER_NAMES:
                mean_fractions[name] = round(
                    fraction_sums[name] / total_pixels, 4,
                )
        else:
            for name in ENDMEMBER_NAMES:
                mean_fractions[name] = 0.0

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": DensityMappingMethod.LINEAR_SPECTRAL_UNMIXING.value,
            "endmembers": ENDMEMBER_NAMES,
            "mean_fractions": mean_fractions,
            "forest_fraction": mean_fractions.get("forest", 0.0),
            "total_pixels": total_pixels,
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.debug(
            "LSU complete: %d pixels, mean_forest_frac=%.4f, %.2fms",
            total_pixels,
            mean_fractions.get("forest", 0.0),
            elapsed_ms,
        )

        return pixel_densities, metadata

    def ndvi_canopy_regression(
        self,
        ndvi_values: List[List[float]],
        biome: str = "tropical_rainforest",
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Estimate canopy cover using biome-calibrated NDVI regression.

        Applies a linear regression model: canopy_pct = a * NDVI + b
        where (a, b) are biome-specific coefficients calibrated against
        MODIS VCF (MOD44B) tree cover products.

        Output is clipped to [0, 100] to prevent physically impossible
        values.

        Args:
            ndvi_values: 2D grid of NDVI values (rows x cols).
            biome: Biome type for coefficient selection. Must match a key
                in BIOME_REGRESSION_COEFFICIENTS.

        Returns:
            Tuple of (pixel_densities, metadata) where pixel_densities
            is a list of canopy density percentages.

        Raises:
            ValueError: If biome is not recognized or NDVI grid is empty.
        """
        start_time = time.monotonic()

        biome_key = biome.lower().strip()
        coefficients = BIOME_REGRESSION_COEFFICIENTS.get(biome_key)
        if coefficients is None:
            raise ValueError(
                f"Unknown biome '{biome}'. Valid biomes: "
                f"{list(BIOME_REGRESSION_COEFFICIENTS.keys())}"
            )

        if not ndvi_values or not ndvi_values[0]:
            raise ValueError("NDVI grid is empty")

        a_coeff, b_coeff = coefficients
        pixel_densities: List[float] = []

        for row in ndvi_values:
            for ndvi in row:
                if math.isnan(ndvi) or math.isinf(ndvi):
                    pixel_densities.append(0.0)
                    continue

                canopy = a_coeff * ndvi + b_coeff
                canopy = max(0.0, min(100.0, canopy))
                pixel_densities.append(round(canopy, 2))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": DensityMappingMethod.NDVI_CANOPY_REGRESSION.value,
            "biome": biome_key,
            "a_coefficient": a_coeff,
            "b_coefficient": b_coeff,
            "formula": f"canopy_pct = {a_coeff} * NDVI + {b_coeff}",
            "total_pixels": len(pixel_densities),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.debug(
            "NDVI regression complete: biome=%s, %d pixels, "
            "a=%.1f, b=%.1f, %.2fms",
            biome_key, len(pixel_densities), a_coeff, b_coeff, elapsed_ms,
        )

        return pixel_densities, metadata

    def dimidiation_model(
        self,
        ndvi_values: List[List[float]],
        biome: str = "tropical_rainforest",
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Estimate fractional vegetation cover using the dimidiation model.

        FVC = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)

        Where NDVI_soil and NDVI_veg are biome-specific reference values
        representing bare soil and fully vegetated surfaces respectively.
        FVC is clipped to [0, 1] and converted to percentage.

        Reference: Gutman & Ignatov (1998), Carlson & Ripley (1997).

        Args:
            ndvi_values: 2D grid of NDVI values (rows x cols).
            biome: Biome type for reference value selection.

        Returns:
            Tuple of (pixel_densities, metadata) where pixel_densities
            is a list of canopy density percentages.

        Raises:
            ValueError: If biome is not recognized or NDVI grid is empty.
        """
        start_time = time.monotonic()

        biome_key = biome.lower().strip()
        references = BIOME_NDVI_REFERENCES.get(biome_key)
        if references is None:
            raise ValueError(
                f"Unknown biome '{biome}'. Valid biomes: "
                f"{list(BIOME_NDVI_REFERENCES.keys())}"
            )

        if not ndvi_values or not ndvi_values[0]:
            raise ValueError("NDVI grid is empty")

        ndvi_soil, ndvi_veg = references
        denominator = ndvi_veg - ndvi_soil

        if abs(denominator) < 1e-10:
            raise ValueError(
                f"NDVI_veg ({ndvi_veg}) and NDVI_soil ({ndvi_soil}) "
                f"are too close for biome '{biome_key}'"
            )

        pixel_densities: List[float] = []

        for row in ndvi_values:
            for ndvi in row:
                if math.isnan(ndvi) or math.isinf(ndvi):
                    pixel_densities.append(0.0)
                    continue

                fvc = (ndvi - ndvi_soil) / denominator
                fvc = max(0.0, min(1.0, fvc))
                density = fvc * 100.0
                pixel_densities.append(round(density, 2))

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": DensityMappingMethod.DIMIDIATION_MODEL.value,
            "biome": biome_key,
            "ndvi_soil": ndvi_soil,
            "ndvi_veg": ndvi_veg,
            "formula": (
                f"FVC = (NDVI - {ndvi_soil}) / ({ndvi_veg} - {ndvi_soil})"
            ),
            "total_pixels": len(pixel_densities),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.debug(
            "Dimidiation model complete: biome=%s, ndvi_soil=%.2f, "
            "ndvi_veg=%.2f, %d pixels, %.2fms",
            biome_key, ndvi_soil, ndvi_veg,
            len(pixel_densities), elapsed_ms,
        )

        return pixel_densities, metadata

    def sub_pixel_detection(
        self,
        ndvi_values: List[List[float]],
        window_size: int = GLCM_WINDOW_SIZE,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Detect sparse canopy cover using GLCM texture metrics.

        For sparse canopies (<30% cover) where spectral methods
        underperform, this method uses Grey-Level Co-occurrence Matrix
        (GLCM) contrast and homogeneity computed over local windows
        to identify tree presence in mixed pixels.

        GLCM Contrast measures local intensity variation:
            contrast = sum(|i - j|^2 * P(i, j))
        GLCM Homogeneity measures local similarity:
            homogeneity = sum(P(i, j) / (1 + |i - j|))

        Higher contrast + lower homogeneity indicates tree-grass mosaic
        (sparse canopy). The texture signature is combined with NDVI to
        yield a refined density estimate.

        Args:
            ndvi_values: 2D grid of NDVI values (rows x cols).
            window_size: Size of the sliding window for GLCM computation.
                Must be an odd integer >= 3.

        Returns:
            Tuple of (pixel_densities, metadata).

        Raises:
            ValueError: If NDVI grid is empty or window_size is invalid.
        """
        start_time = time.monotonic()

        if not ndvi_values or not ndvi_values[0]:
            raise ValueError("NDVI grid is empty")

        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be an odd integer >= 3, got {window_size}"
            )

        rows = len(ndvi_values)
        cols = len(ndvi_values[0])
        half_w = window_size // 2

        pixel_densities: List[float] = []
        total_contrast = 0.0
        total_homogeneity = 0.0
        window_count = 0

        for r in range(rows):
            for c in range(cols):
                ndvi = ndvi_values[r][c]

                if math.isnan(ndvi) or math.isinf(ndvi):
                    pixel_densities.append(0.0)
                    continue

                # Extract local window
                window_vals = self._extract_window(
                    ndvi_values, r, c, half_w, rows, cols,
                )

                # Compute GLCM-like texture metrics on the window
                contrast, homogeneity = self._compute_texture_metrics(
                    window_vals,
                )

                total_contrast += contrast
                total_homogeneity += homogeneity
                window_count += 1

                # Combine NDVI and texture for density estimate
                # Higher contrast in sparse canopy = more tree presence
                # Scale: texture_boost adds 0-15% based on contrast
                texture_boost = min(15.0, contrast * 50.0)
                base_density = max(0.0, ndvi * 80.0)
                density = base_density + texture_boost
                density = max(0.0, min(100.0, density))
                pixel_densities.append(round(density, 2))

        mean_contrast = (
            total_contrast / window_count if window_count > 0 else 0.0
        )
        mean_homogeneity = (
            total_homogeneity / window_count if window_count > 0 else 0.0
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        metadata = {
            "method": DensityMappingMethod.SUB_PIXEL_DETECTION.value,
            "window_size": window_size,
            "mean_contrast": round(mean_contrast, 6),
            "mean_homogeneity": round(mean_homogeneity, 6),
            "total_pixels": len(pixel_densities),
            "processing_time_ms": round(elapsed_ms, 2),
        }

        logger.debug(
            "Sub-pixel detection complete: window=%d, mean_contrast=%.4f, "
            "mean_homogeneity=%.4f, %d pixels, %.2fms",
            window_size, mean_contrast, mean_homogeneity,
            len(pixel_densities), elapsed_ms,
        )

        return pixel_densities, metadata

    # ------------------------------------------------------------------
    # Internal: Method Dispatch
    # ------------------------------------------------------------------

    def _dispatch_method(
        self,
        plot: PlotInput,
        method: DensityMappingMethod,
    ) -> Tuple[float, List[float], Dict[str, Any]]:
        """Dispatch to the appropriate mapping method and compute mean density.

        Args:
            plot: Plot input data.
            method: Selected mapping method.

        Returns:
            Tuple of (mean_density_pct, pixel_densities, metadata).
        """
        if method == DensityMappingMethod.LINEAR_SPECTRAL_UNMIXING:
            pixel_densities, metadata = self.linear_spectral_unmixing(
                plot.multi_band_reflectance,
            )
        elif method == DensityMappingMethod.NDVI_CANOPY_REGRESSION:
            pixel_densities, metadata = self.ndvi_canopy_regression(
                plot.ndvi_values, plot.biome,
            )
        elif method == DensityMappingMethod.DIMIDIATION_MODEL:
            pixel_densities, metadata = self.dimidiation_model(
                plot.ndvi_values, plot.biome,
            )
        elif method == DensityMappingMethod.SUB_PIXEL_DETECTION:
            pixel_densities, metadata = self.sub_pixel_detection(
                plot.ndvi_values,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Compute mean density
        mean_density = self._compute_mean_density(pixel_densities)

        return mean_density, pixel_densities, metadata

    # ------------------------------------------------------------------
    # Internal: Linear Spectral Unmixing Solver
    # ------------------------------------------------------------------

    def _solve_unmixing(
        self,
        pixel_reflectance: List[float],
        endmember_matrix: List[List[float]],
    ) -> List[float]:
        """Solve bounded least-squares unmixing for a single pixel.

        Finds fractions f_1..f_n that minimize:
            ||R_pixel - sum(f_i * R_endmember_i)||^2
        Subject to:
            f_i >= 0 for all i
            sum(f_i) = 1.0

        Uses iterative projection onto the simplex constraint set.
        This is a simplified but deterministic approach suitable for
        4-endmember unmixing.

        Args:
            pixel_reflectance: Observed reflectance vector (6 bands).
            endmember_matrix: List of endmember reflectance vectors.

        Returns:
            List of fractions for each endmember, summing to 1.0.
        """
        n_endmembers = len(endmember_matrix)
        n_bands = len(pixel_reflectance)

        # Initialize with equal fractions
        fractions = [1.0 / n_endmembers] * n_endmembers

        # Iterative NNLS-like solver (10 iterations sufficient for 4 endmembers)
        for _iteration in range(10):
            # Compute current modelled reflectance
            modelled = [0.0] * n_bands
            for j in range(n_endmembers):
                for b in range(n_bands):
                    modelled[b] += fractions[j] * endmember_matrix[j][b]

            # Compute residuals
            residuals = [
                pixel_reflectance[b] - modelled[b]
                for b in range(n_bands)
            ]

            # Update each fraction using gradient step
            for j in range(n_endmembers):
                gradient = 0.0
                for b in range(n_bands):
                    gradient += residuals[b] * endmember_matrix[j][b]

                # Step size scaled by number of bands
                step = gradient / max(n_bands, 1)
                fractions[j] += step

            # Project onto non-negative simplex
            fractions = self._project_to_simplex(fractions)

        return [round(f, 6) for f in fractions]

    def _project_to_simplex(
        self,
        fractions: List[float],
    ) -> List[float]:
        """Project fractions onto the probability simplex.

        Ensures all fractions are >= 0 and sum to 1.0 using the
        Duchi et al. (2008) algorithm.

        Args:
            fractions: Unconstrained fraction values.

        Returns:
            Projected fractions on the simplex (non-negative, sum to 1).
        """
        n = len(fractions)
        if n == 0:
            return []

        # Sort in descending order
        sorted_vals = sorted(fractions, reverse=True)

        # Find the threshold
        cumsum = 0.0
        threshold = 0.0
        for i in range(n):
            cumsum += sorted_vals[i]
            candidate = (cumsum - 1.0) / (i + 1)
            if sorted_vals[i] - candidate > 0:
                threshold = candidate

        # Apply threshold and clip
        result = [max(0.0, f - threshold) for f in fractions]

        # Normalize to exactly 1.0
        total = sum(result)
        if total > 1e-10:
            result = [f / total for f in result]
        else:
            # Fallback: equal distribution
            result = [1.0 / n] * n

        return result

    # ------------------------------------------------------------------
    # Internal: Texture Metrics
    # ------------------------------------------------------------------

    def _extract_window(
        self,
        grid: List[List[float]],
        center_r: int,
        center_c: int,
        half_w: int,
        rows: int,
        cols: int,
    ) -> List[float]:
        """Extract a local window of values around a center pixel.

        Handles boundary conditions by clamping indices to the grid
        extent (replication padding).

        Args:
            grid: 2D grid of values.
            center_r: Center row index.
            center_c: Center column index.
            half_w: Half-width of the window.
            rows: Total number of rows.
            cols: Total number of columns.

        Returns:
            Flattened list of values in the window.
        """
        values: List[float] = []
        for dr in range(-half_w, half_w + 1):
            for dc in range(-half_w, half_w + 1):
                r = max(0, min(rows - 1, center_r + dr))
                c = max(0, min(cols - 1, center_c + dc))
                val = grid[r][c]
                if math.isnan(val) or math.isinf(val):
                    values.append(0.0)
                else:
                    values.append(val)
        return values

    def _compute_texture_metrics(
        self,
        window_values: List[float],
    ) -> Tuple[float, float]:
        """Compute GLCM-like contrast and homogeneity from a window.

        Simplified GLCM computation using pairwise differences between
        adjacent values in the window.

        Contrast = mean(|v_i - v_j|^2) for adjacent pairs
        Homogeneity = mean(1 / (1 + |v_i - v_j|)) for adjacent pairs

        Args:
            window_values: Flattened window pixel values.

        Returns:
            Tuple of (contrast, homogeneity).
        """
        n = len(window_values)
        if n < 2:
            return 0.0, 1.0

        contrast_sum = 0.0
        homogeneity_sum = 0.0
        pair_count = 0

        for i in range(n - 1):
            diff = abs(window_values[i] - window_values[i + 1])
            contrast_sum += diff * diff
            homogeneity_sum += 1.0 / (1.0 + diff)
            pair_count += 1

        if pair_count == 0:
            return 0.0, 1.0

        contrast = contrast_sum / pair_count
        homogeneity = homogeneity_sum / pair_count

        return contrast, homogeneity

    # ------------------------------------------------------------------
    # Internal: Confidence Calculation
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        method: DensityMappingMethod,
        cloud_cover_pct: float,
        resolution_m: float,
        pixel_count: int,
    ) -> float:
        """Calculate confidence score for a density analysis result.

        Combines method reliability, cloud cover penalty, resolution
        bonus, and sample size factor.

        Confidence = method_reliability
                   * cloud_factor
                   * resolution_factor
                   * sample_size_factor

        Args:
            method: Mapping method used.
            cloud_cover_pct: Cloud cover percentage in imagery.
            resolution_m: Spatial resolution in metres.
            pixel_count: Number of pixels analyzed.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        # Base reliability from method
        base = METHOD_RELIABILITY.get(method.value, 0.70)

        # Cloud cover penalty: linearly decreases confidence
        # 0% cloud = 1.0, 50% cloud = 0.5, 100% cloud = 0.0
        cloud_factor = max(0.0, 1.0 - (cloud_cover_pct / 100.0))

        # Resolution factor: higher resolution = higher confidence
        # 10m = 1.0, 20m = 0.9, 30m = 0.85, 250m = 0.5
        if resolution_m <= 10.0:
            resolution_factor = 1.0
        elif resolution_m <= 30.0:
            resolution_factor = 0.95 - (resolution_m - 10.0) * 0.005
        else:
            resolution_factor = max(0.5, 0.85 - (resolution_m - 30.0) * 0.001)

        # Sample size factor: more pixels = more reliable
        # Saturates at 1000 pixels
        if pixel_count >= 1000:
            sample_factor = 1.0
        elif pixel_count >= 100:
            sample_factor = 0.9 + (pixel_count - 100) / 9000.0
        elif pixel_count > 0:
            sample_factor = 0.5 + (pixel_count / 200.0)
        else:
            sample_factor = 0.0

        confidence = base * cloud_factor * resolution_factor * sample_factor
        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # Internal: Statistics Helpers
    # ------------------------------------------------------------------

    def _compute_mean_density(
        self,
        pixel_densities: List[float],
    ) -> float:
        """Compute mean canopy density from pixel-level values.

        Filters out NaN and infinity values before averaging.

        Args:
            pixel_densities: List of per-pixel density percentages.

        Returns:
            Mean density percentage.
        """
        valid = [
            d for d in pixel_densities
            if not (math.isnan(d) or math.isinf(d))
        ]
        if not valid:
            return 0.0
        return round(sum(valid) / len(valid), 2)

    def _compute_mean_ndvi(
        self,
        ndvi_grid: List[List[float]],
    ) -> float:
        """Compute mean NDVI from a 2D grid.

        Args:
            ndvi_grid: 2D grid of NDVI values.

        Returns:
            Mean NDVI value, or 0.0 if grid is empty.
        """
        if not ndvi_grid or not ndvi_grid[0]:
            return 0.0

        total = 0.0
        count = 0
        for row in ndvi_grid:
            for val in row:
                if not (math.isnan(val) or math.isinf(val)):
                    total += val
                    count += 1

        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_plot_input(
        self,
        plot: PlotInput,
        method: DensityMappingMethod,
    ) -> None:
        """Validate plot input data for the selected method.

        Args:
            plot: Plot input data.
            method: Selected mapping method.

        Raises:
            ValueError: If required data is missing or invalid.
        """
        if not plot.plot_id:
            raise ValueError("plot_id must not be empty")

        if method == DensityMappingMethod.LINEAR_SPECTRAL_UNMIXING:
            if not plot.multi_band_reflectance:
                raise ValueError(
                    "multi_band_reflectance is required for "
                    "LINEAR_SPECTRAL_UNMIXING method"
                )
            if len(plot.multi_band_reflectance) != NUM_BANDS:
                raise ValueError(
                    f"multi_band_reflectance must have {NUM_BANDS} bands, "
                    f"got {len(plot.multi_band_reflectance)}"
                )
        else:
            if not plot.ndvi_values or not plot.ndvi_values[0]:
                raise ValueError(
                    "ndvi_values are required for "
                    f"{method.value} method"
                )

        if plot.cloud_cover_pct < 0.0 or plot.cloud_cover_pct > 100.0:
            raise ValueError(
                f"cloud_cover_pct must be in [0, 100], "
                f"got {plot.cloud_cover_pct}"
            )

        if plot.resolution_m <= 0.0:
            raise ValueError(
                f"resolution_m must be > 0, got {plot.resolution_m}"
            )

        if plot.area_ha < 0.0:
            raise ValueError(
                f"area_ha must be >= 0, got {plot.area_ha}"
            )

    # ------------------------------------------------------------------
    # Internal: Error Result
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        plot: PlotInput,
        method: DensityMappingMethod,
        error_msg: str,
    ) -> CanopyDensityResult:
        """Create an error result for a failed analysis.

        Args:
            plot: Plot input that caused the error.
            method: Method that was attempted.
            error_msg: Error message.

        Returns:
            CanopyDensityResult with zero confidence and error metadata.
        """
        result = CanopyDensityResult(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            density_pct=0.0,
            density_class=CanopyDensityClass.OPEN.value,
            method_used=method.value,
            meets_fao_forest=False,
            confidence=0.0,
            biome=plot.biome,
            area_ha=plot.area_ha,
            timestamp=_utcnow().isoformat(),
            metadata={"error": error_msg},
        )
        result.provenance_hash = self._compute_result_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Provenance Hashing
    # ------------------------------------------------------------------

    def _compute_result_hash(self, result: CanopyDensityResult) -> str:
        """Compute SHA-256 provenance hash for a density result.

        Hashes key result fields (not the full pixel array) for
        performance while maintaining audit integrity.

        Args:
            result: CanopyDensityResult to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "result_id": result.result_id,
            "plot_id": result.plot_id,
            "density_pct": result.density_pct,
            "density_class": result.density_class,
            "method_used": result.method_used,
            "meets_fao_forest": result.meets_fao_forest,
            "mean_ndvi": result.mean_ndvi,
            "forest_fraction": result.forest_fraction,
            "confidence": result.confidence,
            "cloud_cover_pct": result.cloud_cover_pct,
            "biome": result.biome,
            "area_ha": result.area_ha,
            "timestamp": result.timestamp,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "CanopyDensityClass",
    "DensityMappingMethod",
    # Constants
    "FAO_CANOPY_THRESHOLD_PCT",
    "FAO_AREA_THRESHOLD_HA",
    "DEFAULT_PIXEL_SIZE_M",
    "SPARSE_CANOPY_THRESHOLD_PCT",
    "ENDMEMBER_LIBRARY",
    "ENDMEMBER_NAMES",
    "BIOME_REGRESSION_COEFFICIENTS",
    "BIOME_NDVI_REFERENCES",
    "METHOD_RELIABILITY",
    # Data classes
    "CanopyDensityResult",
    "PlotInput",
    # Engine
    "CanopyDensityMapper",
]
