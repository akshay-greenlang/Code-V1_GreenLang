# -*- coding: utf-8 -*-
"""
Spectral Index Calculator Engine - AGENT-EUDR-003: Satellite Monitoring (Feature 2)

Computes vegetation and land cover spectral indices from satellite imagery
band data for EUDR deforestation monitoring. Implements NDVI, EVI, NBR,
NDMI, and SAVI calculations with biome-specific forest classification
thresholds and forest area estimation from pixel counts.

Zero-Hallucination Guarantees:
    - All index calculations use deterministic float arithmetic.
    - Division-by-zero is explicitly handled (returns 0.0).
    - All indices are clipped to valid physical ranges [-1, 1].
    - NaN/inf values are filtered before statistical aggregation.
    - Forest classification uses static biome threshold lookup tables.
    - SHA-256 provenance hashes on all result objects.
    - No ML/LLM used for any spectral computation.

Spectral Indices:
    - NDVI: Normalized Difference Vegetation Index (NIR - Red) / (NIR + Red)
    - EVI:  Enhanced Vegetation Index with atmospheric correction
    - NBR:  Normalized Burn Ratio for fire/burn scar detection
    - NDMI: Normalized Difference Moisture Index for vegetation water content
    - SAVI: Soil-Adjusted Vegetation Index for sparse canopy conditions

Performance Targets:
    - Single index calculation (64x64 grid): <10ms
    - Forest classification: <5ms
    - Forest area estimation: <2ms

Regulatory References:
    - EUDR Article 2(1): Deforestation-free verification via vegetation monitoring
    - EUDR Article 9: Spatial analysis using spectral indices
    - EUDR Article 10: Risk assessment evidence from vegetation change

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 (Feature 2: Spectral Index Calculation)
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
from datetime import datetime, timezone
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
# Constants
# ---------------------------------------------------------------------------

#: Valid range for normalized difference indices.
INDEX_MIN: float = -1.0
INDEX_MAX: float = 1.0

#: EVI gain factor (standard value from Huete et al., 2002).
EVI_GAIN: float = 2.5

#: EVI atmospheric correction coefficient C1 (red band).
EVI_C1: float = 6.0

#: EVI atmospheric correction coefficient C2 (blue band).
EVI_C2: float = 7.5

#: EVI soil adjustment factor L.
EVI_L: float = 1.0

#: Default NDVI threshold for forest classification.
DEFAULT_FOREST_NDVI_THRESHOLD: float = 0.4

#: Default SAVI soil adjustment factor.
DEFAULT_SAVI_SOIL_FACTOR: float = 0.5

# ---------------------------------------------------------------------------
# Biome-specific NDVI Thresholds for Forest Classification
# ---------------------------------------------------------------------------
# Each biome has: (dense_forest_min, forest_min, sparse_forest_min,
#                  degraded_min, non_forest_max)

BIOME_NDVI_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "tropical_rainforest": {
        "dense_forest_min": 0.70,
        "forest_min": 0.50,
        "sparse_forest_min": 0.35,
        "degraded_min": 0.25,
        "non_forest_max": 0.25,
    },
    "tropical_dry": {
        "dense_forest_min": 0.55,
        "forest_min": 0.40,
        "sparse_forest_min": 0.30,
        "degraded_min": 0.20,
        "non_forest_max": 0.20,
    },
    "temperate": {
        "dense_forest_min": 0.60,
        "forest_min": 0.45,
        "sparse_forest_min": 0.30,
        "degraded_min": 0.20,
        "non_forest_max": 0.20,
    },
    "boreal": {
        "dense_forest_min": 0.50,
        "forest_min": 0.35,
        "sparse_forest_min": 0.25,
        "degraded_min": 0.15,
        "non_forest_max": 0.15,
    },
    "mangrove": {
        "dense_forest_min": 0.60,
        "forest_min": 0.45,
        "sparse_forest_min": 0.30,
        "degraded_min": 0.20,
        "non_forest_max": 0.20,
    },
    "cerrado_savanna": {
        "dense_forest_min": 0.50,
        "forest_min": 0.35,
        "sparse_forest_min": 0.25,
        "degraded_min": 0.15,
        "non_forest_max": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SpectralIndexResult:
    """Result of a spectral index calculation.

    Attributes:
        index_name: Name of the spectral index (e.g., 'NDVI').
        values: Flattened list of index values for all pixels.
        mean: Mean index value across all valid pixels.
        min_val: Minimum index value.
        max_val: Maximum index value.
        std_dev: Standard deviation of index values.
        pixel_count: Total number of pixels processed.
        valid_pixel_count: Number of valid (non-NaN) pixels.
        forest_mask: List of booleans indicating forest pixels.
        forest_pixel_count: Number of pixels classified as forest.
        formula: Formula string used for calculation.
        provenance_hash: SHA-256 provenance hash.
    """

    index_name: str = ""
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    std_dev: float = 0.0
    pixel_count: int = 0
    valid_pixel_count: int = 0
    forest_mask: List[bool] = field(default_factory=list)
    forest_pixel_count: int = 0
    formula: str = ""
    provenance_hash: str = ""

@dataclass
class ForestClassification:
    """Result of biome-specific forest classification.

    Attributes:
        biome: Biome type used for classification.
        total_pixels: Total number of pixels classified.
        dense_forest_pixels: Pixels classified as dense forest.
        forest_pixels: Pixels classified as forest.
        sparse_forest_pixels: Pixels classified as sparse forest.
        degraded_pixels: Pixels classified as degraded.
        non_forest_pixels: Pixels classified as non-forest.
        dense_forest_pct: Percentage of dense forest.
        forest_pct: Total forest percentage (dense + forest + sparse).
        thresholds_used: NDVI thresholds used for this biome.
        provenance_hash: SHA-256 provenance hash.
    """

    biome: str = ""
    total_pixels: int = 0
    dense_forest_pixels: int = 0
    forest_pixels: int = 0
    sparse_forest_pixels: int = 0
    degraded_pixels: int = 0
    non_forest_pixels: int = 0
    dense_forest_pct: float = 0.0
    forest_pct: float = 0.0
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    provenance_hash: str = ""

# ---------------------------------------------------------------------------
# SpectralIndexCalculator
# ---------------------------------------------------------------------------

class SpectralIndexCalculator:
    """Production-grade spectral index calculator for EUDR satellite monitoring.

    Computes vegetation and land cover indices from satellite imagery
    band data with biome-aware forest classification. All calculations
    are deterministic with explicit handling of edge cases (division by
    zero, NaN, infinity).

    Example::

        calculator = SpectralIndexCalculator()
        ndvi = calculator.calculate_ndvi(red_band, nir_band)
        assert -1.0 <= ndvi.mean <= 1.0
        assert ndvi.provenance_hash != ""

    Attributes:
        forest_ndvi_threshold: Default NDVI threshold for forest detection.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the SpectralIndexCalculator.

        Args:
            config: Optional configuration object. If provided,
                overrides default forest_ndvi_threshold from config.
        """
        self.forest_ndvi_threshold = DEFAULT_FOREST_NDVI_THRESHOLD

        if config is not None:
            self.forest_ndvi_threshold = getattr(
                config, "forest_ndvi_threshold", DEFAULT_FOREST_NDVI_THRESHOLD
            )

        logger.info(
            "SpectralIndexCalculator initialized: "
            "forest_ndvi_threshold=%.2f",
            self.forest_ndvi_threshold,
        )

    # ------------------------------------------------------------------
    # Public API: Index Calculations
    # ------------------------------------------------------------------

    def calculate_ndvi(
        self,
        red_band: List[List[float]],
        nir_band: List[List[float]],
    ) -> SpectralIndexResult:
        """Calculate Normalized Difference Vegetation Index (NDVI).

        NDVI = (NIR - Red) / (NIR + Red)

        Healthy vegetation reflects strongly in NIR and absorbs Red,
        yielding NDVI values near +1.0. Bare soil/water yields values
        near 0.0 or negative.

        Args:
            red_band: 2D array of Red band reflectance values [0, 1].
            nir_band: 2D array of NIR band reflectance values [0, 1].

        Returns:
            SpectralIndexResult with NDVI values and statistics.

        Raises:
            ValueError: If bands have different dimensions.
        """
        start_time = time.monotonic()

        self._validate_band_dimensions(red_band, nir_band, "Red", "NIR")

        values: List[float] = []
        forest_mask: List[bool] = []

        for r in range(len(red_band)):
            for c in range(len(red_band[0])):
                red_val = red_band[r][c]
                nir_val = nir_band[r][c]
                denominator = nir_val + red_val

                if abs(denominator) < 1e-10:
                    ndvi = 0.0
                else:
                    ndvi = (nir_val - red_val) / denominator

                ndvi = self._clip_index(ndvi)
                values.append(ndvi)
                forest_mask.append(ndvi >= self.forest_ndvi_threshold)

        result = self._build_index_result(
            index_name="NDVI",
            values=values,
            forest_mask=forest_mask,
            formula="(NIR - Red) / (NIR + Red)",
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "NDVI calculated: mean=%.4f, pixels=%d, forest=%d, %.2fms",
            result.mean, result.pixel_count,
            result.forest_pixel_count, elapsed_ms,
        )

        return result

    def calculate_evi(
        self,
        blue_band: List[List[float]],
        red_band: List[List[float]],
        nir_band: List[List[float]],
    ) -> SpectralIndexResult:
        """Calculate Enhanced Vegetation Index (EVI).

        EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

        Where G=2.5, C1=6.0, C2=7.5, L=1.0 (Huete et al., 2002).
        EVI corrects for atmospheric and soil background influences,
        providing improved sensitivity in high biomass regions compared
        to NDVI.

        Args:
            blue_band: 2D array of Blue band reflectance [0, 1].
            red_band: 2D array of Red band reflectance [0, 1].
            nir_band: 2D array of NIR band reflectance [0, 1].

        Returns:
            SpectralIndexResult with EVI values and statistics.

        Raises:
            ValueError: If bands have different dimensions.
        """
        start_time = time.monotonic()

        self._validate_band_dimensions(red_band, nir_band, "Red", "NIR")
        self._validate_band_dimensions(red_band, blue_band, "Red", "Blue")

        values: List[float] = []
        forest_mask: List[bool] = []

        for r in range(len(red_band)):
            for c in range(len(red_band[0])):
                blue_val = blue_band[r][c]
                red_val = red_band[r][c]
                nir_val = nir_band[r][c]

                denominator = (
                    nir_val
                    + EVI_C1 * red_val
                    - EVI_C2 * blue_val
                    + EVI_L
                )

                if abs(denominator) < 1e-10:
                    evi = 0.0
                else:
                    evi = EVI_GAIN * (nir_val - red_val) / denominator

                evi = self._clip_index(evi)
                values.append(evi)
                forest_mask.append(evi >= 0.3)

        result = self._build_index_result(
            index_name="EVI",
            values=values,
            forest_mask=forest_mask,
            formula=f"G*(NIR-Red)/(NIR+C1*Red-C2*Blue+L) "
                    f"[G={EVI_GAIN}, C1={EVI_C1}, C2={EVI_C2}, L={EVI_L}]",
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "EVI calculated: mean=%.4f, pixels=%d, %.2fms",
            result.mean, result.pixel_count, elapsed_ms,
        )

        return result

    def calculate_nbr(
        self,
        nir_band: List[List[float]],
        swir2_band: List[List[float]],
    ) -> SpectralIndexResult:
        """Calculate Normalized Burn Ratio (NBR).

        NBR = (NIR - SWIR2) / (NIR + SWIR2)

        Used for fire and burn scar detection. Healthy vegetation has
        high NIR and low SWIR2 reflectance (NBR near +1). Burned areas
        have low NIR and high SWIR2 (NBR near -1).

        Args:
            nir_band: 2D array of NIR band reflectance [0, 1].
            swir2_band: 2D array of SWIR2 band reflectance [0, 1].

        Returns:
            SpectralIndexResult with NBR values and statistics.

        Raises:
            ValueError: If bands have different dimensions.
        """
        start_time = time.monotonic()

        self._validate_band_dimensions(nir_band, swir2_band, "NIR", "SWIR2")

        values: List[float] = []
        forest_mask: List[bool] = []

        for r in range(len(nir_band)):
            for c in range(len(nir_band[0])):
                nir_val = nir_band[r][c]
                swir2_val = swir2_band[r][c]
                denominator = nir_val + swir2_val

                if abs(denominator) < 1e-10:
                    nbr = 0.0
                else:
                    nbr = (nir_val - swir2_val) / denominator

                nbr = self._clip_index(nbr)
                values.append(nbr)
                # Healthy vegetation has NBR > 0.1
                forest_mask.append(nbr >= 0.1)

        result = self._build_index_result(
            index_name="NBR",
            values=values,
            forest_mask=forest_mask,
            formula="(NIR - SWIR2) / (NIR + SWIR2)",
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "NBR calculated: mean=%.4f, pixels=%d, %.2fms",
            result.mean, result.pixel_count, elapsed_ms,
        )

        return result

    def calculate_ndmi(
        self,
        nir_band: List[List[float]],
        swir1_band: List[List[float]],
    ) -> SpectralIndexResult:
        """Calculate Normalized Difference Moisture Index (NDMI).

        NDMI = (NIR - SWIR1) / (NIR + SWIR1)

        Measures vegetation water content. Higher values indicate
        more moisture in the canopy. Used for drought stress and
        forest health monitoring.

        Args:
            nir_band: 2D array of NIR band reflectance [0, 1].
            swir1_band: 2D array of SWIR1 band reflectance [0, 1].

        Returns:
            SpectralIndexResult with NDMI values and statistics.

        Raises:
            ValueError: If bands have different dimensions.
        """
        start_time = time.monotonic()

        self._validate_band_dimensions(nir_band, swir1_band, "NIR", "SWIR1")

        values: List[float] = []
        forest_mask: List[bool] = []

        for r in range(len(nir_band)):
            for c in range(len(nir_band[0])):
                nir_val = nir_band[r][c]
                swir1_val = swir1_band[r][c]
                denominator = nir_val + swir1_val

                if abs(denominator) < 1e-10:
                    ndmi = 0.0
                else:
                    ndmi = (nir_val - swir1_val) / denominator

                ndmi = self._clip_index(ndmi)
                values.append(ndmi)
                # Healthy vegetation has NDMI > 0.0
                forest_mask.append(ndmi >= 0.0)

        result = self._build_index_result(
            index_name="NDMI",
            values=values,
            forest_mask=forest_mask,
            formula="(NIR - SWIR1) / (NIR + SWIR1)",
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "NDMI calculated: mean=%.4f, pixels=%d, %.2fms",
            result.mean, result.pixel_count, elapsed_ms,
        )

        return result

    def calculate_savi(
        self,
        red_band: List[List[float]],
        nir_band: List[List[float]],
        soil_factor: float = DEFAULT_SAVI_SOIL_FACTOR,
    ) -> SpectralIndexResult:
        """Calculate Soil-Adjusted Vegetation Index (SAVI).

        SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L)

        Where L is the soil brightness correction factor (0 = high
        vegetation cover, 1 = low cover). Default L=0.5 for intermediate
        cover conditions common in EUDR commodity production areas.

        Args:
            red_band: 2D array of Red band reflectance [0, 1].
            nir_band: 2D array of NIR band reflectance [0, 1].
            soil_factor: Soil brightness correction factor L (0-1).

        Returns:
            SpectralIndexResult with SAVI values and statistics.

        Raises:
            ValueError: If bands have different dimensions or
                soil_factor is outside [0, 1].
        """
        start_time = time.monotonic()

        if not (0.0 <= soil_factor <= 1.0):
            raise ValueError(
                f"soil_factor must be between 0 and 1, got {soil_factor}"
            )

        self._validate_band_dimensions(red_band, nir_band, "Red", "NIR")

        values: List[float] = []
        forest_mask: List[bool] = []

        for r in range(len(red_band)):
            for c in range(len(red_band[0])):
                red_val = red_band[r][c]
                nir_val = nir_band[r][c]
                denominator = nir_val + red_val + soil_factor

                if abs(denominator) < 1e-10:
                    savi = 0.0
                else:
                    savi = (1.0 + soil_factor) * (nir_val - red_val) / denominator

                savi = self._clip_index(savi)
                values.append(savi)
                forest_mask.append(savi >= 0.35)

        result = self._build_index_result(
            index_name="SAVI",
            values=values,
            forest_mask=forest_mask,
            formula=f"(1+L)*(NIR-Red)/(NIR+Red+L) [L={soil_factor}]",
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "SAVI calculated: mean=%.4f, pixels=%d, L=%.2f, %.2fms",
            result.mean, result.pixel_count, soil_factor, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Forest Classification
    # ------------------------------------------------------------------

    def classify_forest(
        self,
        ndvi_values: List[float],
        biome: str = "tropical_rainforest",
    ) -> ForestClassification:
        """Classify pixels as forest/non-forest using biome-specific thresholds.

        Applies the NDVI thresholds for the specified biome to classify
        each pixel into one of five categories: dense forest, forest,
        sparse forest, degraded, or non-forest.

        Args:
            ndvi_values: List of NDVI values for each pixel.
            biome: Biome type for threshold lookup. Must be one of:
                'tropical_rainforest', 'tropical_dry', 'temperate',
                'boreal', 'mangrove', 'cerrado_savanna'.

        Returns:
            ForestClassification with pixel counts and percentages.

        Raises:
            ValueError: If biome is not recognized.
        """
        start_time = time.monotonic()

        biome_lower = biome.lower().strip()
        thresholds = BIOME_NDVI_THRESHOLDS.get(biome_lower)
        if thresholds is None:
            raise ValueError(
                f"Unknown biome '{biome}'. Valid biomes: "
                f"{list(BIOME_NDVI_THRESHOLDS.keys())}"
            )

        dense_min = thresholds["dense_forest_min"]
        forest_min = thresholds["forest_min"]
        sparse_min = thresholds["sparse_forest_min"]
        degraded_min = thresholds["degraded_min"]

        dense_count = 0
        forest_count = 0
        sparse_count = 0
        degraded_count = 0
        non_forest_count = 0

        for ndvi in ndvi_values:
            if math.isnan(ndvi) or math.isinf(ndvi):
                non_forest_count += 1
            elif ndvi >= dense_min:
                dense_count += 1
            elif ndvi >= forest_min:
                forest_count += 1
            elif ndvi >= sparse_min:
                sparse_count += 1
            elif ndvi >= degraded_min:
                degraded_count += 1
            else:
                non_forest_count += 1

        total = len(ndvi_values)
        total_forest = dense_count + forest_count + sparse_count

        dense_pct = (dense_count / total * 100.0) if total > 0 else 0.0
        forest_pct = (total_forest / total * 100.0) if total > 0 else 0.0

        classification = ForestClassification(
            biome=biome_lower,
            total_pixels=total,
            dense_forest_pixels=dense_count,
            forest_pixels=forest_count,
            sparse_forest_pixels=sparse_count,
            degraded_pixels=degraded_count,
            non_forest_pixels=non_forest_count,
            dense_forest_pct=round(dense_pct, 2),
            forest_pct=round(forest_pct, 2),
            thresholds_used=thresholds,
        )

        # Compute provenance hash
        classification.provenance_hash = self._compute_classification_hash(
            classification
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Forest classification: biome=%s, total=%d, forest=%.1f%%, "
            "dense=%.1f%%, %.2fms",
            biome_lower, total, forest_pct, dense_pct, elapsed_ms,
        )

        return classification

    def calculate_forest_area(
        self,
        ndvi_values: List[float],
        pixel_size_m: float = 10.0,
        forest_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculate forest area in hectares from NDVI pixel classification.

        Counts pixels exceeding the forest threshold, multiplies by
        pixel area, and converts to hectares.

        Args:
            ndvi_values: List of NDVI values for each pixel.
            pixel_size_m: Pixel edge length in metres (default: 10m).
            forest_threshold: NDVI threshold for forest classification.
                If None, uses engine default.

        Returns:
            Dictionary with forest area statistics.
        """
        start_time = time.monotonic()

        threshold = (
            forest_threshold
            if forest_threshold is not None
            else self.forest_ndvi_threshold
        )

        total_pixels = len(ndvi_values)
        forest_pixels = 0
        non_forest_pixels = 0

        for ndvi in ndvi_values:
            if math.isnan(ndvi) or math.isinf(ndvi):
                non_forest_pixels += 1
            elif ndvi >= threshold:
                forest_pixels += 1
            else:
                non_forest_pixels += 1

        # Pixel area in square metres
        pixel_area_m2 = pixel_size_m * pixel_size_m

        # Convert to hectares (1 ha = 10,000 m2)
        forest_area_ha = (forest_pixels * pixel_area_m2) / 10_000.0
        total_area_ha = (total_pixels * pixel_area_m2) / 10_000.0
        non_forest_area_ha = (non_forest_pixels * pixel_area_m2) / 10_000.0

        forest_pct = (
            (forest_pixels / total_pixels * 100.0)
            if total_pixels > 0 else 0.0
        )

        result = {
            "total_pixels": total_pixels,
            "forest_pixels": forest_pixels,
            "non_forest_pixels": non_forest_pixels,
            "pixel_size_m": pixel_size_m,
            "pixel_area_m2": pixel_area_m2,
            "forest_area_ha": round(forest_area_ha, 4),
            "non_forest_area_ha": round(non_forest_area_ha, 4),
            "total_area_ha": round(total_area_ha, 4),
            "forest_percentage": round(forest_pct, 2),
            "forest_threshold": threshold,
        }

        result["provenance_hash"] = _compute_hash(result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Forest area: %.4f ha (%.1f%% of %.4f ha), threshold=%.2f, %.2fms",
            forest_area_ha, forest_pct, total_area_ha, threshold, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: Band Validation
    # ------------------------------------------------------------------

    def _validate_band_dimensions(
        self,
        band_a: List[List[float]],
        band_b: List[List[float]],
        name_a: str,
        name_b: str,
    ) -> None:
        """Validate that two bands have matching dimensions.

        Args:
            band_a: First band 2D array.
            band_b: Second band 2D array.
            name_a: Name of first band for error messages.
            name_b: Name of second band for error messages.

        Raises:
            ValueError: If bands have different dimensions or are empty.
        """
        if not band_a or not band_a[0]:
            raise ValueError(f"{name_a} band is empty")
        if not band_b or not band_b[0]:
            raise ValueError(f"{name_b} band is empty")

        rows_a, cols_a = len(band_a), len(band_a[0])
        rows_b, cols_b = len(band_b), len(band_b[0])

        if rows_a != rows_b or cols_a != cols_b:
            raise ValueError(
                f"{name_a} band ({rows_a}x{cols_a}) and "
                f"{name_b} band ({rows_b}x{cols_b}) have "
                f"different dimensions"
            )

    # ------------------------------------------------------------------
    # Internal: Index Clipping
    # ------------------------------------------------------------------

    def _clip_index(self, value: float) -> float:
        """Clip an index value to the valid range [-1, 1].

        Handles NaN and infinity by returning 0.0.

        Args:
            value: Raw index value.

        Returns:
            Clipped value in [-1, 1].
        """
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return max(INDEX_MIN, min(INDEX_MAX, value))

    # ------------------------------------------------------------------
    # Internal: Statistics
    # ------------------------------------------------------------------

    def _compute_statistics(
        self, values: List[float],
    ) -> Tuple[float, float, float, float, int]:
        """Compute descriptive statistics for a list of index values.

        Filters out NaN and infinity values before computation.

        Args:
            values: List of index values.

        Returns:
            Tuple of (mean, min_val, max_val, std_dev, valid_count).
        """
        valid = [v for v in values if not (math.isnan(v) or math.isinf(v))]
        n = len(valid)

        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        mean_val = sum(valid) / n
        min_val = min(valid)
        max_val = max(valid)

        if n > 1:
            variance = sum((v - mean_val) ** 2 for v in valid) / (n - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0

        return (
            round(mean_val, 6),
            round(min_val, 6),
            round(max_val, 6),
            round(std_dev, 6),
            n,
        )

    # ------------------------------------------------------------------
    # Internal: Result Building
    # ------------------------------------------------------------------

    def _build_index_result(
        self,
        index_name: str,
        values: List[float],
        forest_mask: List[bool],
        formula: str,
    ) -> SpectralIndexResult:
        """Build a SpectralIndexResult from computed values.

        Args:
            index_name: Name of the spectral index.
            values: List of computed index values.
            forest_mask: List of boolean forest classifications.
            formula: Formula string used.

        Returns:
            Populated SpectralIndexResult.
        """
        mean_val, min_val, max_val, std_dev, valid_count = (
            self._compute_statistics(values)
        )

        forest_count = sum(1 for m in forest_mask if m)

        result = SpectralIndexResult(
            index_name=index_name,
            values=values,
            mean=mean_val,
            min_val=min_val,
            max_val=max_val,
            std_dev=std_dev,
            pixel_count=len(values),
            valid_pixel_count=valid_count,
            forest_mask=forest_mask,
            forest_pixel_count=forest_count,
            formula=formula,
        )

        result.provenance_hash = self._compute_index_result_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Provenance Hashing
    # ------------------------------------------------------------------

    def _compute_index_result_hash(self, result: SpectralIndexResult) -> str:
        """Compute SHA-256 provenance hash for an index result.

        Hashes statistics and metadata but not the full pixel array
        for performance.

        Args:
            result: SpectralIndexResult to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "index_name": result.index_name,
            "mean": result.mean,
            "min_val": result.min_val,
            "max_val": result.max_val,
            "std_dev": result.std_dev,
            "pixel_count": result.pixel_count,
            "valid_pixel_count": result.valid_pixel_count,
            "forest_pixel_count": result.forest_pixel_count,
            "formula": result.formula,
        }
        return _compute_hash(hash_data)

    def _compute_classification_hash(
        self, classification: ForestClassification,
    ) -> str:
        """Compute SHA-256 provenance hash for a forest classification.

        Args:
            classification: ForestClassification to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "biome": classification.biome,
            "total_pixels": classification.total_pixels,
            "dense_forest_pixels": classification.dense_forest_pixels,
            "forest_pixels": classification.forest_pixels,
            "sparse_forest_pixels": classification.sparse_forest_pixels,
            "degraded_pixels": classification.degraded_pixels,
            "non_forest_pixels": classification.non_forest_pixels,
            "dense_forest_pct": classification.dense_forest_pct,
            "forest_pct": classification.forest_pct,
        }
        return _compute_hash(hash_data)

# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SpectralIndexCalculator",
    "SpectralIndexResult",
    "ForestClassification",
    "BIOME_NDVI_THRESHOLDS",
    "DEFAULT_FOREST_NDVI_THRESHOLD",
    "DEFAULT_SAVI_SOIL_FACTOR",
    "EVI_GAIN",
    "EVI_C1",
    "EVI_C2",
    "EVI_L",
]
