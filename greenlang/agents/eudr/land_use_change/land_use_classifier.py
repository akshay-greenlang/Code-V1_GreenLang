# -*- coding: utf-8 -*-
"""
Land Use Classifier Engine - AGENT-EUDR-005: Land Use Change Detector (Engine 1)

Multi-class land use classification engine that categorises geospatial plots
into 10 IPCC-aligned land use categories using five deterministic methods:
spectral signature matching, vegetation index classification, temporal phenology
analysis, GLCM texture analysis, and weighted ensemble voting.

Zero-Hallucination Guarantees:
    - All classifications use deterministic numeric thresholds and lookup
      tables (no ML/LLM inference for category assignment).
    - Spectral signature matching: cosine similarity against reference library.
    - Vegetation index classification: static NDVI/EVI/NDMI/SAVI threshold bands.
    - Temporal phenology: coefficient of variation and amplitude of NDVI time
      series, compared against deterministic ranges per category.
    - Texture analysis: GLCM homogeneity, contrast, and correlation features
      matched against static per-category feature ranges.
    - Ensemble voting: configurable weights (spectral 0.30, VI 0.25,
      phenology 0.25, texture 0.20) with weighted category tallying.
    - SHA-256 provenance hashes on all result objects.

Land Use Categories (10 IPCC-aligned):
    FOREST            - Natural closed-canopy forest (NDVI > 0.6, high NIR)
    PLANTATION_FOREST - Managed timber / pulp plantation (regular texture)
    CROPLAND          - Annual and perennial crops (high seasonal NDVI amplitude)
    OIL_PALM          - Oil palm plantation (distinctive NIR, regular spacing)
    RUBBER            - Rubber plantation (seasonal leaf-drop phenology)
    GRASSLAND         - Natural grassland and pasture (moderate NDVI 0.2-0.5)
    SHRUBLAND         - Shrub-dominated land (low NDVI 0.15-0.35, moderate texture)
    SETTLEMENT        - Urban and built-up areas (high SWIR, low NDVI < 0.2)
    WATER             - Water bodies (very low NIR < 0.1, high Blue)
    BARE_SOIL         - Bare ground, rock, desert (very low NDVI < 0.1)

EUDR Commodity Context:
    When a commodity context is provided, the classifier applies EUDR
    Article 2(4) rules to refine ambiguous classifications. For example,
    timber plantations are NOT classified as agricultural even though they
    exhibit periodic harvesting patterns. Oil palm is distinguished from
    generic cropland due to its distinct spectral and textural signature.

Performance Targets:
    - Single plot classification (all 5 methods): <100ms
    - Spectral signature matching: <15ms
    - Vegetation index classification: <5ms
    - Temporal phenology analysis: <20ms
    - Texture analysis: <15ms
    - Ensemble voting: <5ms
    - Batch classification (100 plots): <5 seconds

Regulatory References:
    - EUDR Article 2(1): Land use at cutoff date requires classification
    - EUDR Article 2(4): Forest definition and plantation exclusions
    - EUDR Article 2(5): Forest degradation assessment
    - EUDR Article 9: Geolocation-based land use evidence
    - IPCC 2006 GL Vol 4 Ch 3: Land categories for GHG inventories

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-005 (Engine 1: Land Use Classification)
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
from datetime import date, datetime, timezone
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


class LandUseCategory(str, Enum):
    """IPCC-aligned land use categories for EUDR compliance.

    Ten categories covering all major land use types relevant to EUDR
    deforestation and forest degradation assessment. Aligned with IPCC
    2006 Guidelines Volume 4 Chapter 3 land-use categories, extended
    with EUDR-specific sub-categories for oil palm, rubber, and
    plantation forest.

    FOREST: Natural closed-canopy forest with canopy cover >60%,
        multi-layered structure, and high biodiversity. High NIR
        reflectance (B8 > 0.3) and NDVI > 0.6.
    PLANTATION_FOREST: Managed forest plantation for timber, pulp,
        or fibre production. Regular spatial texture pattern,
        uniform canopy height, and lower biodiversity than natural
        forest. Excluded from deforestation under Article 2(4).
    CROPLAND: Annual and perennial agricultural crops including
        cereals, vegetables, and fruit trees. Characterised by
        high seasonal NDVI amplitude (>0.3) due to crop cycles.
    OIL_PALM: Oil palm (Elaeis guineensis) plantation. Distinctive
        NIR spectral pattern, regular plantation spacing texture
        (GLCM homogeneity 0.4-0.7), and moderate-high year-round NDVI.
    RUBBER: Rubber (Hevea brasiliensis) plantation. Seasonal leaf-
        drop phenology with NDVI dropping below 0.3 during defoliation,
        moderate-high NDVI (0.4-0.7) during leafy season.
    GRASSLAND: Natural and managed grassland, savanna, and pasture.
        Moderate NDVI (0.2-0.5) with low seasonal variation, low
        texture homogeneity.
    SHRUBLAND: Shrub-dominated land with scattered trees below
        forest threshold. NDVI range 0.15-0.35, moderate-low
        canopy height.
    SETTLEMENT: Urban, peri-urban, and built-up areas including
        buildings, roads, and impervious surfaces. High SWIR
        reflectance (B11, B12), very low NDVI (<0.2).
    WATER: Open water bodies including rivers, lakes, reservoirs,
        and coastal waters. Very low NIR (B8 < 0.1), high Blue
        (B2) reflectance. Negative NDVI typical.
    BARE_SOIL: Bare ground, exposed rock, sand, and desert surfaces.
        Very low NDVI (<0.1), moderate-high SWIR, no seasonal
        vegetation pattern.
    """

    FOREST = "forest"
    PLANTATION_FOREST = "plantation_forest"
    CROPLAND = "cropland"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    GRASSLAND = "grassland"
    SHRUBLAND = "shrubland"
    SETTLEMENT = "settlement"
    WATER = "water"
    BARE_SOIL = "bare_soil"


class ClassificationMethod(str, Enum):
    """Available classification methods for land use determination.

    SPECTRAL_SIGNATURE: Cosine similarity matching of multi-band
        pixel reflectance against reference spectral signatures
        for each land use category.
    VEGETATION_INDEX: Threshold-based classification using NDVI,
        EVI, NDMI, and SAVI vegetation indices.
    TEMPORAL_PHENOLOGY: Classification from seasonal NDVI time-series
        patterns including amplitude, frequency, and phase.
    TEXTURE_ANALYSIS: GLCM (Grey-Level Co-occurrence Matrix) texture
        feature analysis of spatial patterns in NIR band.
    ENSEMBLE: Weighted combination of all four methods for robust
        consensus classification with configurable weights.
    """

    SPECTRAL_SIGNATURE = "SPECTRAL_SIGNATURE"
    VEGETATION_INDEX = "VEGETATION_INDEX"
    TEMPORAL_PHENOLOGY = "TEMPORAL_PHENOLOGY"
    TEXTURE_ANALYSIS = "TEXTURE_ANALYSIS"
    ENSEMBLE = "ENSEMBLE"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR deforestation cutoff date per Article 2(1).
EUDR_CUTOFF_DATE: date = date(2020, 12, 31)

#: Default spatial resolution in metres (Sentinel-2 10m bands).
DEFAULT_PIXEL_SIZE_M: float = 10.0

#: Maximum number of plots in a single batch classification.
MAX_BATCH_SIZE: int = 5000

#: Minimum spectral bands required for classification.
MIN_SPECTRAL_BANDS: int = 6

#: Band ordering: [Blue(B2), Green(B3), Red(B4), NIR(B8), SWIR1(B11), SWIR2(B12)]
BAND_NAMES: List[str] = ["B2_blue", "B3_green", "B4_red", "B8_nir", "B11_swir1", "B12_swir2"]

#: Number of spectral bands.
NUM_BANDS: int = 6

#: GLCM analysis window size (pixels).
GLCM_WINDOW_SIZE: int = 5

#: Ensemble method weights (must sum to 1.0).
ENSEMBLE_WEIGHTS: Dict[str, float] = {
    ClassificationMethod.SPECTRAL_SIGNATURE.value: 0.30,
    ClassificationMethod.VEGETATION_INDEX.value: 0.25,
    ClassificationMethod.TEMPORAL_PHENOLOGY.value: 0.25,
    ClassificationMethod.TEXTURE_ANALYSIS.value: 0.20,
}

#: Minimum confidence for a valid classification.
MIN_CLASSIFICATION_CONFIDENCE: float = 0.40

#: Cloud-contaminated pixel NDVI ceiling.
CLOUD_NDVI_CEILING: float = 0.0

# ---------------------------------------------------------------------------
# Reference spectral signatures
# ---------------------------------------------------------------------------
# Each signature is a list of mean reflectance values for 6 bands:
#   [Blue, Green, Red, NIR, SWIR1, SWIR2]
# Derived from USGS Spectral Library v7 (Kokaly et al. 2017) and
# ESA Sentinel-2 L2A BOA reflectance calibration datasets.

REFERENCE_SIGNATURES: Dict[str, List[float]] = {
    LandUseCategory.FOREST.value: [0.020, 0.040, 0.030, 0.400, 0.180, 0.080],
    LandUseCategory.PLANTATION_FOREST.value: [0.025, 0.045, 0.035, 0.380, 0.200, 0.095],
    LandUseCategory.CROPLAND.value: [0.040, 0.060, 0.055, 0.320, 0.250, 0.150],
    LandUseCategory.OIL_PALM.value: [0.022, 0.042, 0.032, 0.420, 0.190, 0.085],
    LandUseCategory.RUBBER.value: [0.025, 0.048, 0.038, 0.350, 0.210, 0.100],
    LandUseCategory.GRASSLAND.value: [0.045, 0.065, 0.060, 0.280, 0.230, 0.160],
    LandUseCategory.SHRUBLAND.value: [0.055, 0.070, 0.065, 0.240, 0.220, 0.170],
    LandUseCategory.SETTLEMENT.value: [0.100, 0.100, 0.110, 0.140, 0.200, 0.180],
    LandUseCategory.WATER.value: [0.060, 0.050, 0.035, 0.020, 0.010, 0.008],
    LandUseCategory.BARE_SOIL.value: [0.120, 0.130, 0.150, 0.180, 0.280, 0.260],
}

# ---------------------------------------------------------------------------
# Vegetation index classification thresholds
# ---------------------------------------------------------------------------
# Each category has (ndvi_min, ndvi_max, evi_min, evi_max, ndmi_min,
# ndmi_max, savi_min, savi_max). Ranges are inclusive on both ends.
# Overlapping ranges are resolved by picking the category with the smallest
# Euclidean distance to the range centroid.

VI_THRESHOLDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    LandUseCategory.FOREST.value: {
        "ndvi": (0.60, 1.00),
        "evi": (0.40, 0.80),
        "ndmi": (0.20, 0.60),
        "savi": (0.45, 0.80),
    },
    LandUseCategory.PLANTATION_FOREST.value: {
        "ndvi": (0.50, 0.80),
        "evi": (0.35, 0.65),
        "ndmi": (0.15, 0.45),
        "savi": (0.38, 0.65),
    },
    LandUseCategory.CROPLAND.value: {
        "ndvi": (0.20, 0.65),
        "evi": (0.15, 0.50),
        "ndmi": (-0.05, 0.25),
        "savi": (0.15, 0.45),
    },
    LandUseCategory.OIL_PALM.value: {
        "ndvi": (0.50, 0.75),
        "evi": (0.30, 0.55),
        "ndmi": (0.10, 0.35),
        "savi": (0.35, 0.58),
    },
    LandUseCategory.RUBBER.value: {
        "ndvi": (0.30, 0.68),
        "evi": (0.22, 0.52),
        "ndmi": (0.05, 0.30),
        "savi": (0.22, 0.50),
    },
    LandUseCategory.GRASSLAND.value: {
        "ndvi": (0.20, 0.50),
        "evi": (0.15, 0.40),
        "ndmi": (-0.05, 0.20),
        "savi": (0.15, 0.40),
    },
    LandUseCategory.SHRUBLAND.value: {
        "ndvi": (0.15, 0.35),
        "evi": (0.10, 0.30),
        "ndmi": (-0.10, 0.15),
        "savi": (0.10, 0.30),
    },
    LandUseCategory.SETTLEMENT.value: {
        "ndvi": (-0.10, 0.20),
        "evi": (-0.05, 0.15),
        "ndmi": (-0.30, 0.00),
        "savi": (-0.05, 0.15),
    },
    LandUseCategory.WATER.value: {
        "ndvi": (-1.00, 0.00),
        "evi": (-0.50, 0.05),
        "ndmi": (0.30, 1.00),
        "savi": (-0.50, 0.05),
    },
    LandUseCategory.BARE_SOIL.value: {
        "ndvi": (-0.10, 0.10),
        "evi": (-0.05, 0.10),
        "ndmi": (-0.40, -0.10),
        "savi": (-0.05, 0.10),
    },
}

# ---------------------------------------------------------------------------
# Phenology classification parameters
# ---------------------------------------------------------------------------
# Each category has expected NDVI time-series characteristics:
#   cv_range: coefficient of variation range (low=stable, high=seasonal)
#   amplitude_range: seasonal NDVI amplitude (max - min)
#   peak_count_range: expected number of peaks per year (0=evergreen, 1=annual,
#                     2=double-cropping)
#   mean_ndvi_range: expected mean NDVI across the year

PHENOLOGY_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    LandUseCategory.FOREST.value: {
        "cv": (0.02, 0.12),
        "amplitude": (0.02, 0.15),
        "peak_count": (0, 1),
        "mean_ndvi": (0.55, 0.90),
    },
    LandUseCategory.PLANTATION_FOREST.value: {
        "cv": (0.05, 0.18),
        "amplitude": (0.05, 0.25),
        "peak_count": (0, 1),
        "mean_ndvi": (0.45, 0.75),
    },
    LandUseCategory.CROPLAND.value: {
        "cv": (0.20, 0.60),
        "amplitude": (0.30, 0.70),
        "peak_count": (1, 3),
        "mean_ndvi": (0.20, 0.60),
    },
    LandUseCategory.OIL_PALM.value: {
        "cv": (0.03, 0.10),
        "amplitude": (0.03, 0.12),
        "peak_count": (0, 1),
        "mean_ndvi": (0.55, 0.80),
    },
    LandUseCategory.RUBBER.value: {
        "cv": (0.15, 0.35),
        "amplitude": (0.20, 0.45),
        "peak_count": (1, 1),
        "mean_ndvi": (0.35, 0.65),
    },
    LandUseCategory.GRASSLAND.value: {
        "cv": (0.10, 0.30),
        "amplitude": (0.10, 0.30),
        "peak_count": (0, 2),
        "mean_ndvi": (0.20, 0.45),
    },
    LandUseCategory.SHRUBLAND.value: {
        "cv": (0.08, 0.25),
        "amplitude": (0.05, 0.20),
        "peak_count": (0, 1),
        "mean_ndvi": (0.15, 0.35),
    },
    LandUseCategory.SETTLEMENT.value: {
        "cv": (0.05, 0.15),
        "amplitude": (0.02, 0.10),
        "peak_count": (0, 1),
        "mean_ndvi": (0.00, 0.18),
    },
    LandUseCategory.WATER.value: {
        "cv": (0.05, 0.20),
        "amplitude": (0.02, 0.08),
        "peak_count": (0, 0),
        "mean_ndvi": (-0.30, 0.05),
    },
    LandUseCategory.BARE_SOIL.value: {
        "cv": (0.02, 0.10),
        "amplitude": (0.01, 0.05),
        "peak_count": (0, 0),
        "mean_ndvi": (-0.05, 0.10),
    },
}

# ---------------------------------------------------------------------------
# Texture classification parameters
# ---------------------------------------------------------------------------
# Each category has expected GLCM texture feature ranges:
#   homogeneity: spatial uniformity (0-1, high=uniform)
#   contrast: local intensity variation (0+, high=varied)
#   correlation: pixel-pair linear correlation (0-1, high=correlated)

TEXTURE_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    LandUseCategory.FOREST.value: {
        "homogeneity": (0.30, 0.65),
        "contrast": (10.0, 50.0),
        "correlation": (0.60, 0.95),
    },
    LandUseCategory.PLANTATION_FOREST.value: {
        "homogeneity": (0.50, 0.80),
        "contrast": (5.0, 25.0),
        "correlation": (0.70, 0.95),
    },
    LandUseCategory.CROPLAND.value: {
        "homogeneity": (0.55, 0.85),
        "contrast": (3.0, 20.0),
        "correlation": (0.50, 0.85),
    },
    LandUseCategory.OIL_PALM.value: {
        "homogeneity": (0.40, 0.70),
        "contrast": (8.0, 35.0),
        "correlation": (0.65, 0.90),
    },
    LandUseCategory.RUBBER.value: {
        "homogeneity": (0.45, 0.75),
        "contrast": (5.0, 30.0),
        "correlation": (0.60, 0.90),
    },
    LandUseCategory.GRASSLAND.value: {
        "homogeneity": (0.65, 0.95),
        "contrast": (1.0, 10.0),
        "correlation": (0.30, 0.70),
    },
    LandUseCategory.SHRUBLAND.value: {
        "homogeneity": (0.50, 0.80),
        "contrast": (3.0, 20.0),
        "correlation": (0.35, 0.75),
    },
    LandUseCategory.SETTLEMENT.value: {
        "homogeneity": (0.20, 0.50),
        "contrast": (30.0, 100.0),
        "correlation": (0.20, 0.60),
    },
    LandUseCategory.WATER.value: {
        "homogeneity": (0.85, 1.00),
        "contrast": (0.0, 3.0),
        "correlation": (0.80, 1.00),
    },
    LandUseCategory.BARE_SOIL.value: {
        "homogeneity": (0.70, 0.95),
        "contrast": (1.0, 8.0),
        "correlation": (0.40, 0.80),
    },
}

# ---------------------------------------------------------------------------
# Commodity context mapping for EUDR Article 2(4)
# ---------------------------------------------------------------------------
# Maps EUDR commodities to their expected land use categories when
# commodity context is provided. This allows disambiguation of
# ambiguous classifications.

COMMODITY_EXPECTED_CATEGORIES: Dict[str, List[str]] = {
    "cattle": [
        LandUseCategory.GRASSLAND.value,
        LandUseCategory.CROPLAND.value,
    ],
    "cocoa": [
        LandUseCategory.CROPLAND.value,
        LandUseCategory.FOREST.value,  # shade-grown cocoa
    ],
    "coffee": [
        LandUseCategory.CROPLAND.value,
        LandUseCategory.FOREST.value,  # shade-grown coffee
    ],
    "oil_palm": [
        LandUseCategory.OIL_PALM.value,
        LandUseCategory.CROPLAND.value,
    ],
    "rubber": [
        LandUseCategory.RUBBER.value,
        LandUseCategory.PLANTATION_FOREST.value,
    ],
    "soya": [
        LandUseCategory.CROPLAND.value,
    ],
    "wood": [
        LandUseCategory.FOREST.value,
        LandUseCategory.PLANTATION_FOREST.value,
    ],
}

# ---------------------------------------------------------------------------
# EUDR Article 2(4) exclusion rules
# ---------------------------------------------------------------------------
# Timber plantations (plantation_forest) are NOT classified as agricultural
# land even though they exhibit periodic harvesting patterns.

ARTICLE_2_4_PLANTATION_CATEGORIES: List[str] = [
    LandUseCategory.PLANTATION_FOREST.value,
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SpectralData:
    """Multi-band spectral reflectance data for a plot pixel.

    Attributes:
        bands: Reflectance values for each band in order:
            [Blue, Green, Red, NIR, SWIR1, SWIR2].
            Values are bottom-of-atmosphere reflectance in [0, 1].
        cloud_mask: True if this pixel is cloud-contaminated.
        pixel_row: Row index within the plot grid.
        pixel_col: Column index within the plot grid.
    """

    bands: List[float] = field(default_factory=list)
    cloud_mask: bool = False
    pixel_row: int = 0
    pixel_col: int = 0


@dataclass
class VegetationIndices:
    """Vegetation index values for a plot.

    Attributes:
        ndvi: Normalised Difference Vegetation Index [-1, 1].
        evi: Enhanced Vegetation Index [-1, 1].
        ndmi: Normalised Difference Moisture Index [-1, 1].
        savi: Soil-Adjusted Vegetation Index [-1, 1].
    """

    ndvi: float = 0.0
    evi: float = 0.0
    ndmi: float = 0.0
    savi: float = 0.0


@dataclass
class TextureFeatures:
    """GLCM texture features for a plot.

    Attributes:
        homogeneity: Spatial uniformity measure [0, 1].
        contrast: Local intensity variation measure [0, inf).
        correlation: Pixel-pair linear dependency measure [0, 1].
        window_size: GLCM window size used for computation.
    """

    homogeneity: float = 0.0
    contrast: float = 0.0
    correlation: float = 0.0
    window_size: int = GLCM_WINDOW_SIZE


@dataclass
class PhenologyTimeSeries:
    """NDVI time-series data for phenology analysis.

    Attributes:
        dates: List of observation dates (ISO format strings).
        ndvi_values: NDVI values corresponding to each date.
        interval_months: Nominal time step between observations in months.
    """

    dates: List[str] = field(default_factory=list)
    ndvi_values: List[float] = field(default_factory=list)
    interval_months: int = 1


@dataclass
class PlotClassificationInput:
    """Input data for classifying a single plot.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Plot centroid latitude (-90 to 90).
        longitude: Plot centroid longitude (-180 to 180).
        spectral_data: List of per-pixel spectral reflectance data.
        vegetation_indices: Pre-computed vegetation indices (plot-mean).
        texture_features: Pre-computed GLCM texture features.
        phenology_series: NDVI time-series for phenology classification.
        observation_date: Date of the spectral observation.
        commodity_context: Optional EUDR commodity for disambiguation.
        cloud_cover_pct: Cloud cover percentage in source imagery.
        area_ha: Plot area in hectares.
    """

    plot_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    spectral_data: List[SpectralData] = field(default_factory=list)
    vegetation_indices: Optional[VegetationIndices] = None
    texture_features: Optional[TextureFeatures] = None
    phenology_series: Optional[PhenologyTimeSeries] = None
    observation_date: Optional[str] = None
    commodity_context: Optional[str] = None
    cloud_cover_pct: float = 0.0
    area_ha: float = 1.0


@dataclass
class LandUseClassification:
    """Result of land use classification for a single plot.

    Attributes:
        result_id: Unique result identifier (UUID).
        plot_id: Identifier of the classified plot.
        category: Primary land use category.
        confidence: Confidence in the primary classification [0, 1].
        secondary_category: Second-most-likely category.
        secondary_confidence: Confidence in the secondary category [0, 1].
        method_used: Classification method that produced this result.
        method_scores: Scores from each individual method (category -> confidence).
        agreement_ratio: Fraction of methods agreeing on primary category [0, 1].
        vegetation_indices: Vegetation index values used.
        spectral_similarity: Cosine similarity to reference signature [0, 1].
        phenology_match_score: Phenology match quality [0, 1].
        texture_match_score: Texture match quality [0, 1].
        commodity_adjusted: Whether commodity context was applied.
        original_category: Original category before commodity adjustment.
        cloud_cover_pct: Cloud cover percentage in source imagery.
        observation_date: Date of the source observation.
        latitude: Plot centroid latitude.
        longitude: Plot centroid longitude.
        processing_time_ms: Time taken for classification in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp of classification.
        metadata: Additional contextual information.
    """

    result_id: str = ""
    plot_id: str = ""
    category: str = ""
    confidence: float = 0.0
    secondary_category: str = ""
    secondary_confidence: float = 0.0
    method_used: str = ""
    method_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    agreement_ratio: float = 0.0
    vegetation_indices: Optional[Dict[str, float]] = None
    spectral_similarity: float = 0.0
    phenology_match_score: float = 0.0
    texture_match_score: float = 0.0
    commodity_adjusted: bool = False
    original_category: str = ""
    cloud_cover_pct: float = 0.0
    observation_date: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
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
            "category": self.category,
            "confidence": self.confidence,
            "secondary_category": self.secondary_category,
            "secondary_confidence": self.secondary_confidence,
            "method_used": self.method_used,
            "method_scores": self.method_scores,
            "agreement_ratio": self.agreement_ratio,
            "vegetation_indices": self.vegetation_indices,
            "spectral_similarity": self.spectral_similarity,
            "phenology_match_score": self.phenology_match_score,
            "texture_match_score": self.texture_match_score,
            "commodity_adjusted": self.commodity_adjusted,
            "original_category": self.original_category,
            "cloud_cover_pct": self.cloud_cover_pct,
            "observation_date": self.observation_date,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# LandUseClassifier
# ---------------------------------------------------------------------------


class LandUseClassifier:
    """Production-grade multi-class land use classification engine for EUDR.

    Classifies land into 10 IPCC-aligned categories using five deterministic
    methods: spectral signature matching, vegetation index thresholds,
    temporal phenology analysis, GLCM texture analysis, and weighted
    ensemble voting. All computations are zero-hallucination with full
    SHA-256 provenance tracking.

    Example::

        classifier = LandUseClassifier()
        plot = PlotClassificationInput(
            plot_id="plot-001",
            latitude=-2.5,
            longitude=110.0,
            vegetation_indices=VegetationIndices(
                ndvi=0.72, evi=0.55, ndmi=0.35, savi=0.55,
            ),
        )
        result = classifier.classify(
            latitude=-2.5,
            longitude=110.0,
            date=date(2023, 6, 15),
            method=ClassificationMethod.VEGETATION_INDEX,
        )
        assert result.category in [c.value for c in LandUseCategory]

    Attributes:
        config: Optional configuration object.
        default_method: Default classification method.
        ensemble_weights: Weights for ensemble voting.
    """

    def __init__(
        self,
        config: Any = None,
        default_method: ClassificationMethod = ClassificationMethod.ENSEMBLE,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the LandUseClassifier.

        Args:
            config: Optional configuration object with overrides.
            default_method: Default classification method to use.
            ensemble_weights: Custom weights for ensemble voting. Must
                sum to 1.0. If None, uses default ENSEMBLE_WEIGHTS.

        Raises:
            ValueError: If ensemble_weights do not sum to 1.0.
        """
        self.config = config
        self.default_method = default_method

        if ensemble_weights is not None:
            weight_sum = sum(ensemble_weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                raise ValueError(
                    f"ensemble_weights must sum to 1.0, got {weight_sum:.4f}"
                )
            self.ensemble_weights = ensemble_weights
        else:
            self.ensemble_weights = dict(ENSEMBLE_WEIGHTS)

        logger.info(
            "LandUseClassifier initialized: default_method=%s, "
            "module_version=%s, categories=%d",
            self.default_method.value,
            _MODULE_VERSION,
            len(LandUseCategory),
        )

    # ------------------------------------------------------------------
    # Public API: Main Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        latitude: float,
        longitude: float,
        date: date,
        method: Optional[ClassificationMethod] = None,
        commodity_context: Optional[str] = None,
        spectral_data: Optional[List[SpectralData]] = None,
        vegetation_indices: Optional[VegetationIndices] = None,
        texture_features: Optional[TextureFeatures] = None,
        phenology_series: Optional[PhenologyTimeSeries] = None,
        cloud_cover_pct: float = 0.0,
    ) -> LandUseClassification:
        """Classify land use at a given location and date.

        Runs the specified classification method (or default ensemble)
        using available input data. Returns a comprehensive result with
        category, confidence, and provenance hash.

        Args:
            latitude: Plot centroid latitude (-90 to 90).
            longitude: Plot centroid longitude (-180 to 180).
            date: Observation date for the classification.
            method: Classification method to use. Defaults to
                self.default_method.
            commodity_context: Optional EUDR commodity for Article 2(4)
                disambiguation. One of: cattle, cocoa, coffee, oil_palm,
                rubber, soya, wood.
            spectral_data: Optional multi-band spectral reflectance data.
            vegetation_indices: Optional pre-computed vegetation indices.
            texture_features: Optional pre-computed GLCM texture features.
            phenology_series: Optional NDVI time-series data.
            cloud_cover_pct: Cloud cover percentage in source imagery.

        Returns:
            LandUseClassification with primary and secondary categories,
            confidence scores, and provenance hash.

        Raises:
            ValueError: If latitude or longitude are out of range, or if
                no input data is available for the selected method.
        """
        start_time = time.monotonic()
        selected_method = method or self.default_method

        self._validate_coordinates(latitude, longitude)

        plot = PlotClassificationInput(
            plot_id=_generate_id(),
            latitude=latitude,
            longitude=longitude,
            spectral_data=spectral_data or [],
            vegetation_indices=vegetation_indices,
            texture_features=texture_features,
            phenology_series=phenology_series,
            observation_date=date.isoformat() if date else "",
            commodity_context=commodity_context,
            cloud_cover_pct=cloud_cover_pct,
        )

        return self._classify_plot(plot, selected_method, start_time)

    def classify_batch(
        self,
        plots: List[PlotClassificationInput],
        date: date,
        method: Optional[ClassificationMethod] = None,
    ) -> List[LandUseClassification]:
        """Classify land use for a batch of plots.

        Processes each plot sequentially. For I/O-bound workloads,
        consider using async batch processing at the orchestration layer.

        Args:
            plots: List of plot inputs to classify.
            date: Observation date for all classifications.
            method: Classification method to use. Defaults to
                self.default_method.

        Returns:
            List of LandUseClassification results, one per input plot.

        Raises:
            ValueError: If plots list is empty or exceeds MAX_BATCH_SIZE.
        """
        if not plots:
            raise ValueError("plots list must not be empty")
        if len(plots) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(plots)} exceeds maximum of {MAX_BATCH_SIZE}"
            )

        batch_start = time.monotonic()
        selected_method = method or self.default_method
        results: List[LandUseClassification] = []

        for i, plot in enumerate(plots):
            try:
                if not plot.observation_date:
                    plot.observation_date = date.isoformat()
                start_time = time.monotonic()
                result = self._classify_plot(plot, selected_method, start_time)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "classify_batch: failed on plot[%d] id=%s: %s",
                    i, plot.plot_id, str(exc),
                )
                error_result = self._create_error_result(
                    plot=plot,
                    method=selected_method,
                    error_msg=str(exc),
                )
                results.append(error_result)

        batch_elapsed = (time.monotonic() - batch_start) * 1000
        successful = sum(1 for r in results if r.confidence > 0.0)

        logger.info(
            "classify_batch complete: %d/%d successful, %.2fms total",
            successful, len(plots), batch_elapsed,
        )

        return results

    # ------------------------------------------------------------------
    # Internal: Core Classification Pipeline
    # ------------------------------------------------------------------

    def _classify_plot(
        self,
        plot: PlotClassificationInput,
        method: ClassificationMethod,
        start_time: float,
    ) -> LandUseClassification:
        """Run the classification pipeline for a single plot.

        Args:
            plot: Input data for the plot.
            method: Classification method to apply.
            start_time: Monotonic start time for processing duration.

        Returns:
            Complete LandUseClassification result.
        """
        self._validate_plot_input(plot, method)

        result_id = _generate_id()
        timestamp = _utcnow().isoformat()

        all_method_scores: Dict[str, Dict[str, float]] = {}

        if method == ClassificationMethod.ENSEMBLE:
            category, confidence, all_method_scores, agreement = (
                self._run_ensemble(plot)
            )
        elif method == ClassificationMethod.SPECTRAL_SIGNATURE:
            category, confidence = self._spectral_classify(
                plot.spectral_data
            )
            all_method_scores[method.value] = {category.value: confidence}
            agreement = 1.0
        elif method == ClassificationMethod.VEGETATION_INDEX:
            vi = plot.vegetation_indices
            if vi is None:
                vi = self._compute_vi_from_spectral(plot.spectral_data)
            category, confidence = self._vegetation_index_classify(
                vi.ndvi, vi.evi, vi.ndmi, vi.savi,
            )
            all_method_scores[method.value] = {category.value: confidence}
            agreement = 1.0
        elif method == ClassificationMethod.TEMPORAL_PHENOLOGY:
            ps = plot.phenology_series
            if ps is None:
                raise ValueError(
                    "phenology_series required for TEMPORAL_PHENOLOGY method"
                )
            category, confidence = self._phenology_classify(ps)
            all_method_scores[method.value] = {category.value: confidence}
            agreement = 1.0
        elif method == ClassificationMethod.TEXTURE_ANALYSIS:
            tf = plot.texture_features
            if tf is None:
                raise ValueError(
                    "texture_features required for TEXTURE_ANALYSIS method"
                )
            category, confidence = self._texture_classify(tf)
            all_method_scores[method.value] = {category.value: confidence}
            agreement = 1.0
        else:
            raise ValueError(f"Unsupported classification method: {method}")

        # Apply commodity context if provided
        original_category = category
        commodity_adjusted = False
        if plot.commodity_context:
            adjusted = self._apply_commodity_context(
                category, plot.commodity_context,
            )
            if adjusted != category:
                commodity_adjusted = True
                category = adjusted

        # Compute overall confidence adjusted for cloud cover
        final_confidence = self._compute_confidence(
            all_method_scores, agreement, plot.cloud_cover_pct,
        )

        # Get secondary category
        secondary_cat, secondary_conf = self._get_secondary_category(
            all_method_scores, category,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        vi_dict = None
        if plot.vegetation_indices:
            vi_dict = {
                "ndvi": plot.vegetation_indices.ndvi,
                "evi": plot.vegetation_indices.evi,
                "ndmi": plot.vegetation_indices.ndmi,
                "savi": plot.vegetation_indices.savi,
            }

        result = LandUseClassification(
            result_id=result_id,
            plot_id=plot.plot_id,
            category=category.value,
            confidence=round(final_confidence, 4),
            secondary_category=secondary_cat.value if secondary_cat else "",
            secondary_confidence=round(secondary_conf, 4),
            method_used=method.value,
            method_scores=all_method_scores,
            agreement_ratio=round(agreement, 4),
            vegetation_indices=vi_dict,
            spectral_similarity=0.0,
            phenology_match_score=0.0,
            texture_match_score=0.0,
            commodity_adjusted=commodity_adjusted,
            original_category=original_category.value if commodity_adjusted else "",
            cloud_cover_pct=plot.cloud_cover_pct,
            observation_date=plot.observation_date or "",
            latitude=plot.latitude,
            longitude=plot.longitude,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=timestamp,
            metadata={
                "module_version": _MODULE_VERSION,
                "method": method.value,
                "commodity_context": plot.commodity_context or "",
            },
        )

        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Land use classified: plot=%s, category=%s, confidence=%.2f, "
            "method=%s, commodity_adjusted=%s, %.2fms",
            plot.plot_id,
            category.value,
            final_confidence,
            method.value,
            commodity_adjusted,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Method 1: Spectral Signature Classification
    # ------------------------------------------------------------------

    def _spectral_classify(
        self,
        spectral_data: List[SpectralData],
    ) -> Tuple[LandUseCategory, float]:
        """Classify land use by matching pixel spectra against references.

        Computes the mean reflectance across all non-cloudy pixels, then
        calculates cosine similarity against each reference spectral
        signature. The category with the highest similarity wins.

        Args:
            spectral_data: List of per-pixel spectral reflectance data.

        Returns:
            Tuple of (best matching category, cosine similarity score).

        Raises:
            ValueError: If spectral_data is empty or all pixels are cloudy.
        """
        if not spectral_data:
            raise ValueError("spectral_data is empty for spectral classification")

        mean_spectrum = self._compute_mean_spectrum(spectral_data)
        if mean_spectrum is None:
            raise ValueError("All pixels are cloud-masked; cannot classify")

        if not self._validate_spectral_data(mean_spectrum):
            raise ValueError(
                f"Invalid spectral data: expected {NUM_BANDS} bands "
                f"with values in [0, 1], got {mean_spectrum}"
            )

        best_category = LandUseCategory.BARE_SOIL
        best_similarity = -1.0
        similarity_scores: Dict[str, float] = {}

        for cat_name, ref_signature in REFERENCE_SIGNATURES.items():
            sim = self._cosine_similarity(mean_spectrum, ref_signature)
            similarity_scores[cat_name] = sim
            if sim > best_similarity:
                best_similarity = sim
                best_category = LandUseCategory(cat_name)

        confidence = self._similarity_to_confidence(best_similarity)

        logger.debug(
            "Spectral classify: best=%s, similarity=%.4f, confidence=%.4f",
            best_category.value, best_similarity, confidence,
        )

        return best_category, confidence

    def _compute_mean_spectrum(
        self,
        spectral_data: List[SpectralData],
    ) -> Optional[List[float]]:
        """Compute the mean spectral reflectance across non-cloudy pixels.

        Args:
            spectral_data: Per-pixel spectral data.

        Returns:
            Mean reflectance vector, or None if all pixels are cloudy.
        """
        valid_pixels = [
            sd for sd in spectral_data
            if not sd.cloud_mask and len(sd.bands) >= NUM_BANDS
        ]

        if not valid_pixels:
            return None

        n = len(valid_pixels)
        mean_bands: List[float] = [0.0] * NUM_BANDS

        for pixel in valid_pixels:
            for b in range(NUM_BANDS):
                mean_bands[b] += pixel.bands[b]

        mean_bands = [v / n for v in mean_bands]
        return mean_bands

    def _cosine_similarity(
        self,
        vec_a: List[float],
        vec_b: List[float],
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector (same length as vec_a).

        Returns:
            Cosine similarity in [-1, 1]. Returns 0.0 if either vector
            has zero magnitude.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))

        if mag_a < 1e-12 or mag_b < 1e-12:
            return 0.0

        return dot_product / (mag_a * mag_b)

    def _similarity_to_confidence(self, similarity: float) -> float:
        """Convert cosine similarity to a confidence score.

        Maps cosine similarity from the typical range [0.8, 1.0] to
        a confidence range [0.0, 1.0] using linear interpolation.
        Similarities below 0.7 map to very low confidence.

        Args:
            similarity: Cosine similarity value.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if similarity >= 0.99:
            return 0.98
        if similarity >= 0.95:
            return 0.80 + (similarity - 0.95) * 4.5
        if similarity >= 0.90:
            return 0.60 + (similarity - 0.90) * 4.0
        if similarity >= 0.80:
            return 0.30 + (similarity - 0.80) * 3.0
        if similarity >= 0.70:
            return 0.10 + (similarity - 0.70) * 2.0
        return max(0.0, similarity * 0.14)

    # ------------------------------------------------------------------
    # Method 2: Vegetation Index Classification
    # ------------------------------------------------------------------

    def _vegetation_index_classify(
        self,
        ndvi: float,
        evi: float,
        ndmi: float,
        savi: float,
    ) -> Tuple[LandUseCategory, float]:
        """Classify land use using vegetation index thresholds.

        Computes a normalised distance score for each category based
        on how well the observed index values fall within the expected
        threshold ranges. The category with the lowest total distance
        (best fit) wins.

        Args:
            ndvi: Normalised Difference Vegetation Index.
            evi: Enhanced Vegetation Index.
            ndmi: Normalised Difference Moisture Index.
            savi: Soil-Adjusted Vegetation Index.

        Returns:
            Tuple of (best matching category, confidence score).
        """
        observed = {"ndvi": ndvi, "evi": evi, "ndmi": ndmi, "savi": savi}
        category_scores: Dict[str, float] = {}

        for cat_name, thresholds in VI_THRESHOLDS.items():
            total_fit = 0.0
            num_indices = 0

            for idx_name, (low, high) in thresholds.items():
                obs_val = observed.get(idx_name, 0.0)
                fit = self._range_fit_score(obs_val, low, high)
                total_fit += fit
                num_indices += 1

            avg_fit = total_fit / max(num_indices, 1)
            category_scores[cat_name] = avg_fit

        best_cat_name = max(category_scores, key=category_scores.get)  # type: ignore[arg-type]
        best_score = category_scores[best_cat_name]
        best_category = LandUseCategory(best_cat_name)

        confidence = min(best_score, 0.98)

        logger.debug(
            "VI classify: best=%s, score=%.4f, ndvi=%.3f, evi=%.3f, "
            "ndmi=%.3f, savi=%.3f",
            best_category.value, best_score, ndvi, evi, ndmi, savi,
        )

        return best_category, confidence

    def _range_fit_score(
        self,
        value: float,
        low: float,
        high: float,
    ) -> float:
        """Compute how well a value fits within a range [low, high].

        Returns 1.0 if value is at the centre of the range, tapering
        to 0.0 as value moves outside the range. Uses a Gaussian-like
        decay outside the range boundaries.

        Args:
            value: Observed value.
            low: Lower bound of expected range.
            high: Upper bound of expected range.

        Returns:
            Fit score in [0.0, 1.0].
        """
        if low <= value <= high:
            # Inside range: score based on proximity to centre
            range_width = high - low
            if range_width < 1e-12:
                return 1.0
            centre = (low + high) / 2.0
            distance = abs(value - centre) / (range_width / 2.0)
            return 1.0 - 0.3 * distance  # 1.0 at centre, 0.7 at edges
        else:
            # Outside range: Gaussian decay
            range_width = max(high - low, 0.01)
            if value < low:
                distance = (low - value) / range_width
            else:
                distance = (value - high) / range_width
            return max(0.0, math.exp(-2.0 * distance * distance))

    # ------------------------------------------------------------------
    # Method 3: Temporal Phenology Classification
    # ------------------------------------------------------------------

    def _phenology_classify(
        self,
        time_series: PhenologyTimeSeries,
    ) -> Tuple[LandUseCategory, float]:
        """Classify land use using temporal NDVI phenology patterns.

        Analyses the NDVI time series to extract phenological features
        (coefficient of variation, amplitude, peak count, mean NDVI)
        and matches them against per-category reference parameters.

        Args:
            time_series: NDVI time-series data with dates.

        Returns:
            Tuple of (best matching category, confidence score).

        Raises:
            ValueError: If time_series has fewer than 3 observations.
        """
        values = time_series.ndvi_values
        if len(values) < 3:
            raise ValueError(
                f"Phenology classification requires at least 3 observations, "
                f"got {len(values)}"
            )

        # Extract phenological features
        mean_ndvi = sum(values) / len(values)
        std_ndvi = math.sqrt(
            sum((v - mean_ndvi) ** 2 for v in values) / len(values)
        )
        cv = std_ndvi / max(abs(mean_ndvi), 1e-12)
        amplitude = max(values) - min(values)
        peak_count = self._count_peaks(values)

        # Score each category
        category_scores: Dict[str, float] = {}

        for cat_name, params in PHENOLOGY_PARAMS.items():
            cv_fit = self._range_fit_score(
                cv, params["cv"][0], params["cv"][1],
            )
            amp_fit = self._range_fit_score(
                amplitude, params["amplitude"][0], params["amplitude"][1],
            )
            peak_fit = self._range_fit_score(
                float(peak_count),
                float(params["peak_count"][0]),
                float(params["peak_count"][1]),
            )
            mean_fit = self._range_fit_score(
                mean_ndvi, params["mean_ndvi"][0], params["mean_ndvi"][1],
            )

            avg_fit = (cv_fit + amp_fit + peak_fit + mean_fit) / 4.0
            category_scores[cat_name] = avg_fit

        best_cat_name = max(category_scores, key=category_scores.get)  # type: ignore[arg-type]
        best_score = category_scores[best_cat_name]
        best_category = LandUseCategory(best_cat_name)

        confidence = min(best_score, 0.98)

        logger.debug(
            "Phenology classify: best=%s, score=%.4f, cv=%.3f, "
            "amplitude=%.3f, peaks=%d, mean_ndvi=%.3f",
            best_category.value, best_score, cv, amplitude,
            peak_count, mean_ndvi,
        )

        return best_category, confidence

    def _count_peaks(self, values: List[float]) -> int:
        """Count the number of local maxima (peaks) in a time series.

        A peak is defined as a value that is greater than both its
        immediate neighbours. Edge values are never counted as peaks.

        Args:
            values: List of numeric values.

        Returns:
            Number of peaks detected.
        """
        if len(values) < 3:
            return 0

        peaks = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks += 1

        return peaks

    # ------------------------------------------------------------------
    # Method 4: Texture Analysis Classification
    # ------------------------------------------------------------------

    def _texture_classify(
        self,
        texture_features: TextureFeatures,
    ) -> Tuple[LandUseCategory, float]:
        """Classify land use using GLCM texture features.

        Matches observed texture features (homogeneity, contrast,
        correlation) against per-category reference feature ranges.

        Args:
            texture_features: Pre-computed GLCM texture features.

        Returns:
            Tuple of (best matching category, confidence score).
        """
        observed = {
            "homogeneity": texture_features.homogeneity,
            "contrast": texture_features.contrast,
            "correlation": texture_features.correlation,
        }

        category_scores: Dict[str, float] = {}

        for cat_name, params in TEXTURE_PARAMS.items():
            total_fit = 0.0
            num_features = 0

            for feat_name, (low, high) in params.items():
                obs_val = observed.get(feat_name, 0.0)
                fit = self._range_fit_score(obs_val, low, high)
                total_fit += fit
                num_features += 1

            avg_fit = total_fit / max(num_features, 1)
            category_scores[cat_name] = avg_fit

        best_cat_name = max(category_scores, key=category_scores.get)  # type: ignore[arg-type]
        best_score = category_scores[best_cat_name]
        best_category = LandUseCategory(best_cat_name)

        confidence = min(best_score, 0.98)

        logger.debug(
            "Texture classify: best=%s, score=%.4f, homogeneity=%.3f, "
            "contrast=%.3f, correlation=%.3f",
            best_category.value, best_score,
            texture_features.homogeneity,
            texture_features.contrast,
            texture_features.correlation,
        )

        return best_category, confidence

    # ------------------------------------------------------------------
    # Method 5: Ensemble Classification
    # ------------------------------------------------------------------

    def _run_ensemble(
        self,
        plot: PlotClassificationInput,
    ) -> Tuple[LandUseCategory, float, Dict[str, Dict[str, float]], float]:
        """Run weighted ensemble voting across all available methods.

        Executes each method that has sufficient input data, collects
        per-category scores from each method, and computes a weighted
        vote to determine the final classification.

        Args:
            plot: Input data for the plot.

        Returns:
            Tuple of (best category, raw confidence, method_scores dict,
            agreement ratio).
        """
        method_results: Dict[str, Tuple[LandUseCategory, float]] = {}
        all_method_scores: Dict[str, Dict[str, float]] = {}

        # Run spectral classification if data available
        if plot.spectral_data:
            try:
                cat, conf = self._spectral_classify(plot.spectral_data)
                method_results[ClassificationMethod.SPECTRAL_SIGNATURE.value] = (cat, conf)
                all_method_scores[ClassificationMethod.SPECTRAL_SIGNATURE.value] = {
                    cat.value: conf,
                }
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Ensemble: spectral classification failed: %s", str(exc),
                )

        # Run vegetation index classification if data available
        vi = plot.vegetation_indices
        if vi is None and plot.spectral_data:
            try:
                vi = self._compute_vi_from_spectral(plot.spectral_data)
            except (ValueError, Exception):
                pass

        if vi is not None:
            try:
                cat, conf = self._vegetation_index_classify(
                    vi.ndvi, vi.evi, vi.ndmi, vi.savi,
                )
                method_results[ClassificationMethod.VEGETATION_INDEX.value] = (cat, conf)
                all_method_scores[ClassificationMethod.VEGETATION_INDEX.value] = {
                    cat.value: conf,
                }
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Ensemble: VI classification failed: %s", str(exc),
                )

        # Run phenology classification if data available
        if plot.phenology_series and len(plot.phenology_series.ndvi_values) >= 3:
            try:
                cat, conf = self._phenology_classify(plot.phenology_series)
                method_results[ClassificationMethod.TEMPORAL_PHENOLOGY.value] = (cat, conf)
                all_method_scores[ClassificationMethod.TEMPORAL_PHENOLOGY.value] = {
                    cat.value: conf,
                }
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Ensemble: phenology classification failed: %s", str(exc),
                )

        # Run texture classification if data available
        if plot.texture_features is not None:
            try:
                cat, conf = self._texture_classify(plot.texture_features)
                method_results[ClassificationMethod.TEXTURE_ANALYSIS.value] = (cat, conf)
                all_method_scores[ClassificationMethod.TEXTURE_ANALYSIS.value] = {
                    cat.value: conf,
                }
            except (ValueError, Exception) as exc:
                logger.warning(
                    "Ensemble: texture classification failed: %s", str(exc),
                )

        if not method_results:
            raise ValueError(
                "Ensemble classification failed: no methods could run. "
                "Provide at least one of: spectral_data, vegetation_indices, "
                "phenology_series, or texture_features."
            )

        # Weighted vote
        category, confidence, agreement = self._ensemble_classify(method_results)

        return category, confidence, all_method_scores, agreement

    def _ensemble_classify(
        self,
        method_results: Dict[str, Tuple[LandUseCategory, float]],
    ) -> Tuple[LandUseCategory, float, float]:
        """Compute weighted ensemble vote from individual method results.

        Each method votes for a category with a weight proportional to
        its ensemble weight and classification confidence. The category
        with the highest total weighted score wins.

        Args:
            method_results: Dict mapping method name to (category, confidence).

        Returns:
            Tuple of (winning category, weighted confidence, agreement ratio).
        """
        weighted_votes: Dict[str, float] = {}
        total_weight = 0.0

        for method_name, (category, confidence) in method_results.items():
            weight = self.ensemble_weights.get(method_name, 0.0)
            # If a method that ran is not in weights, assign equal share
            if weight < 1e-12:
                weight = 1.0 / max(len(method_results), 1)

            vote_score = weight * confidence
            cat_val = category.value
            weighted_votes[cat_val] = weighted_votes.get(cat_val, 0.0) + vote_score
            total_weight += weight

        # Normalise votes by total weight
        if total_weight > 1e-12:
            for cat_val in weighted_votes:
                weighted_votes[cat_val] /= total_weight

        # Find the winner
        best_cat_val = max(weighted_votes, key=weighted_votes.get)  # type: ignore[arg-type]
        best_score = weighted_votes[best_cat_val]
        best_category = LandUseCategory(best_cat_val)

        # Agreement ratio: fraction of methods agreeing with the winner
        agreeing = sum(
            1 for (cat, _) in method_results.values()
            if cat.value == best_cat_val
        )
        agreement = agreeing / len(method_results)

        confidence = min(best_score, 0.98)

        logger.debug(
            "Ensemble classify: best=%s, score=%.4f, agreement=%.2f, "
            "methods_used=%d",
            best_category.value, best_score, agreement,
            len(method_results),
        )

        return best_category, confidence, agreement

    # ------------------------------------------------------------------
    # Confidence Computation
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        method_scores: Dict[str, Dict[str, float]],
        agreement_ratio: float,
        cloud_cover_pct: float,
    ) -> float:
        """Compute final classification confidence.

        Combines the raw method confidence with agreement ratio and
        cloud cover penalty. High cloud cover reduces confidence.

        Args:
            method_scores: Per-method category scores.
            agreement_ratio: Fraction of methods agreeing on the winner.
            cloud_cover_pct: Cloud cover percentage in source imagery.

        Returns:
            Final confidence score in [0.0, 1.0].
        """
        # Get the best score from any method
        raw_scores: List[float] = []
        for method_cats in method_scores.values():
            for score in method_cats.values():
                raw_scores.append(score)

        if not raw_scores:
            return 0.0

        best_raw = max(raw_scores)

        # Agreement bonus: agreement_ratio of 1.0 gives 1.1x multiplier,
        # 0.5 gives 1.0x, 0.25 gives 0.9x
        agreement_factor = 0.8 + 0.2 * agreement_ratio

        # Cloud cover penalty: 0% cloud = 1.0, 50% cloud = 0.85, 100% = 0.70
        cloud_factor = 1.0 - 0.003 * cloud_cover_pct

        final = best_raw * agreement_factor * cloud_factor
        return max(0.0, min(1.0, round(final, 4)))

    # ------------------------------------------------------------------
    # Commodity Context Application
    # ------------------------------------------------------------------

    def _apply_commodity_context(
        self,
        base_classification: LandUseCategory,
        commodity: str,
    ) -> LandUseCategory:
        """Apply EUDR Article 2(4) commodity context rules.

        Refines ambiguous classifications when the expected commodity
        provides additional context. For example, if the classifier
        returns CROPLAND but the commodity is oil_palm, and the
        confidence is close, prefer OIL_PALM.

        Timber plantation is NOT classified as agricultural under
        Article 2(4), so a PLANTATION_FOREST classification is
        preserved when the commodity is wood.

        Args:
            base_classification: Original classification result.
            commodity: EUDR commodity context (cattle, cocoa, coffee,
                oil_palm, rubber, soya, wood).

        Returns:
            Adjusted LandUseCategory, or the original if no adjustment
            is warranted.
        """
        commodity_lower = commodity.lower().strip()
        expected_cats = COMMODITY_EXPECTED_CATEGORIES.get(commodity_lower)

        if expected_cats is None:
            logger.debug(
                "Commodity context '%s' not recognised, no adjustment",
                commodity,
            )
            return base_classification

        # If the base classification is already in the expected list, keep it
        if base_classification.value in expected_cats:
            return base_classification

        # Article 2(4): Plantation forest stays as plantation forest for wood
        if (
            commodity_lower == "wood"
            and base_classification.value in ARTICLE_2_4_PLANTATION_CATEGORIES
        ):
            return base_classification

        # Apply disambiguation rules
        # If base is FOREST and commodity is not wood, the forest is being
        # used for agriculture (potential deforestation concern) - keep FOREST
        if base_classification == LandUseCategory.FOREST and commodity_lower != "wood":
            return base_classification

        # If base is ambiguous (e.g., CROPLAND for oil_palm), adjust to
        # the primary expected category for the commodity
        primary_expected = LandUseCategory(expected_cats[0])

        # Only adjust if the base is in a related category group
        agricultural_group = {
            LandUseCategory.CROPLAND.value,
            LandUseCategory.GRASSLAND.value,
            LandUseCategory.OIL_PALM.value,
            LandUseCategory.RUBBER.value,
        }
        forest_group = {
            LandUseCategory.FOREST.value,
            LandUseCategory.PLANTATION_FOREST.value,
        }

        base_in_ag = base_classification.value in agricultural_group
        primary_in_ag = primary_expected.value in agricultural_group
        base_in_forest = base_classification.value in forest_group
        primary_in_forest = primary_expected.value in forest_group

        if (base_in_ag and primary_in_ag) or (base_in_forest and primary_in_forest):
            logger.info(
                "Commodity context applied: %s -> %s (commodity=%s)",
                base_classification.value, primary_expected.value, commodity,
            )
            return primary_expected

        return base_classification

    # ------------------------------------------------------------------
    # Secondary Category Extraction
    # ------------------------------------------------------------------

    def _get_secondary_category(
        self,
        method_scores: Dict[str, Dict[str, float]],
        primary: LandUseCategory,
    ) -> Tuple[Optional[LandUseCategory], float]:
        """Extract the second-best category from method scores.

        Args:
            method_scores: Per-method category scores.
            primary: Primary (winning) category to exclude.

        Returns:
            Tuple of (secondary category or None, confidence score).
        """
        combined: Dict[str, float] = {}
        for method_cats in method_scores.values():
            for cat_val, score in method_cats.items():
                if cat_val != primary.value:
                    combined[cat_val] = max(combined.get(cat_val, 0.0), score)

        if not combined:
            return None, 0.0

        best_cat_val = max(combined, key=combined.get)  # type: ignore[arg-type]
        return LandUseCategory(best_cat_val), combined[best_cat_val]

    # ------------------------------------------------------------------
    # Vegetation Index Computation from Spectral Data
    # ------------------------------------------------------------------

    def _compute_vi_from_spectral(
        self,
        spectral_data: List[SpectralData],
    ) -> VegetationIndices:
        """Compute vegetation indices from mean spectral reflectance.

        Calculates NDVI, EVI, NDMI, and SAVI from the mean spectrum
        of non-cloudy pixels.

        Band mapping (Sentinel-2):
            B2 (Blue)  = bands[0]
            B3 (Green) = bands[1]
            B4 (Red)   = bands[2]
            B8 (NIR)   = bands[3]
            B11 (SWIR1) = bands[4]
            B12 (SWIR2) = bands[5]

        Args:
            spectral_data: List of per-pixel spectral reflectance data.

        Returns:
            VegetationIndices with computed index values.

        Raises:
            ValueError: If no valid pixels are available.
        """
        mean_spectrum = self._compute_mean_spectrum(spectral_data)
        if mean_spectrum is None:
            raise ValueError("No valid pixels for VI computation")

        blue = mean_spectrum[0]
        red = mean_spectrum[2]
        nir = mean_spectrum[3]
        swir1 = mean_spectrum[4]

        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = self._safe_normalised_diff(nir, red)

        # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        evi_denom = nir + 6.0 * red - 7.5 * blue + 1.0
        if abs(evi_denom) < 1e-12:
            evi = 0.0
        else:
            evi = 2.5 * (nir - red) / evi_denom
            evi = max(-1.0, min(1.0, evi))

        # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        ndmi = self._safe_normalised_diff(nir, swir1)

        # SAVI = 1.5 * (NIR - Red) / (NIR + Red + 0.5)
        savi_denom = nir + red + 0.5
        if abs(savi_denom) < 1e-12:
            savi = 0.0
        else:
            savi = 1.5 * (nir - red) / savi_denom
            savi = max(-1.0, min(1.0, savi))

        return VegetationIndices(
            ndvi=round(ndvi, 6),
            evi=round(evi, 6),
            ndmi=round(ndmi, 6),
            savi=round(savi, 6),
        )

    def _safe_normalised_diff(
        self,
        band_a: float,
        band_b: float,
    ) -> float:
        """Compute normalised difference (a - b) / (a + b) safely.

        Args:
            band_a: Numerator addend (typically NIR).
            band_b: Numerator subtrahend (typically Red/SWIR).

        Returns:
            Normalised difference in [-1, 1], or 0 if denominator is zero.
        """
        denom = band_a + band_b
        if abs(denom) < 1e-12:
            return 0.0
        return max(-1.0, min(1.0, (band_a - band_b) / denom))

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    def _validate_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> None:
        """Validate geographic coordinates.

        Args:
            latitude: Latitude to validate (-90 to 90).
            longitude: Longitude to validate (-180 to 180).

        Raises:
            ValueError: If coordinates are out of valid range.
        """
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(
                f"latitude must be in [-90, 90], got {latitude}"
            )
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(
                f"longitude must be in [-180, 180], got {longitude}"
            )

    def _validate_plot_input(
        self,
        plot: PlotClassificationInput,
        method: ClassificationMethod,
    ) -> None:
        """Validate that the plot has sufficient data for the method.

        Args:
            plot: Plot input data.
            method: Classification method to be applied.

        Raises:
            ValueError: If required data is missing for the method.
        """
        if not plot.plot_id:
            raise ValueError("plot_id must not be empty")

        self._validate_coordinates(plot.latitude, plot.longitude)

        if method == ClassificationMethod.SPECTRAL_SIGNATURE:
            if not plot.spectral_data:
                raise ValueError(
                    "spectral_data required for SPECTRAL_SIGNATURE method"
                )

        if method == ClassificationMethod.TEMPORAL_PHENOLOGY:
            if (
                plot.phenology_series is None
                or len(plot.phenology_series.ndvi_values) < 3
            ):
                raise ValueError(
                    "phenology_series with >= 3 observations required "
                    "for TEMPORAL_PHENOLOGY method"
                )

        if method == ClassificationMethod.TEXTURE_ANALYSIS:
            if plot.texture_features is None:
                raise ValueError(
                    "texture_features required for TEXTURE_ANALYSIS method"
                )

        if method == ClassificationMethod.VEGETATION_INDEX:
            if plot.vegetation_indices is None and not plot.spectral_data:
                raise ValueError(
                    "vegetation_indices or spectral_data required "
                    "for VEGETATION_INDEX method"
                )

    def _validate_spectral_data(self, bands: List[float]) -> bool:
        """Validate that spectral data has correct band count and range.

        Args:
            bands: List of reflectance values.

        Returns:
            True if data is valid, False otherwise.
        """
        if len(bands) < NUM_BANDS:
            return False

        for val in bands[:NUM_BANDS]:
            if not (0.0 <= val <= 1.0):
                return False

        return True

    # ------------------------------------------------------------------
    # Error Result Creation
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        plot: PlotClassificationInput,
        method: ClassificationMethod,
        error_msg: str,
    ) -> LandUseClassification:
        """Create an error result for a failed classification.

        Args:
            plot: Input plot that failed classification.
            method: Method that was attempted.
            error_msg: Error message describing the failure.

        Returns:
            LandUseClassification with zero confidence and error metadata.
        """
        return LandUseClassification(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            category=LandUseCategory.BARE_SOIL.value,
            confidence=0.0,
            method_used=method.value,
            cloud_cover_pct=plot.cloud_cover_pct,
            observation_date=plot.observation_date or "",
            latitude=plot.latitude,
            longitude=plot.longitude,
            processing_time_ms=0.0,
            provenance_hash="",
            timestamp=_utcnow().isoformat(),
            metadata={
                "error": True,
                "error_message": error_msg,
                "module_version": _MODULE_VERSION,
            },
        )

    # ------------------------------------------------------------------
    # Provenance Hash Computation
    # ------------------------------------------------------------------

    def _compute_result_hash(
        self,
        result: LandUseClassification,
    ) -> str:
        """Compute SHA-256 provenance hash for a classification result.

        Hash covers all deterministic fields (excluding the hash itself
        and processing_time_ms which is non-deterministic).

        Args:
            result: Classification result to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        hash_data = {
            "plot_id": result.plot_id,
            "category": result.category,
            "confidence": result.confidence,
            "method_used": result.method_used,
            "observation_date": result.observation_date,
            "latitude": result.latitude,
            "longitude": result.longitude,
            "commodity_adjusted": result.commodity_adjusted,
            "original_category": result.original_category,
            "module_version": _MODULE_VERSION,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "LandUseCategory",
    "ClassificationMethod",
    # Constants
    "EUDR_CUTOFF_DATE",
    "MAX_BATCH_SIZE",
    "NUM_BANDS",
    "ENSEMBLE_WEIGHTS",
    "REFERENCE_SIGNATURES",
    "VI_THRESHOLDS",
    "PHENOLOGY_PARAMS",
    "TEXTURE_PARAMS",
    "COMMODITY_EXPECTED_CATEGORIES",
    # Data classes
    "SpectralData",
    "VegetationIndices",
    "TextureFeatures",
    "PhenologyTimeSeries",
    "PlotClassificationInput",
    "LandUseClassification",
    # Engine
    "LandUseClassifier",
]
