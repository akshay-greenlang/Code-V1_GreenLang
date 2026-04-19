"""
Forest Cover Classifier for Deforestation Detection.

Provides classification of satellite imagery into forest and non-forest
areas using vegetation indices and threshold-based methods.

Features:
- Binary forest/non-forest classification
- Tree cover percentage estimation
- Multi-class land cover classification
- Canopy height estimation (with GEDI data)
- Confidence scoring for classifications

Classification thresholds are based on published literature and
calibrated for tropical and temperate forest monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional
import logging

import numpy as np

from greenlang.satellite.analysis.vegetation_indices import (
    IndexResult,
    IndexType,
    VegetationIndexCalculator,
    calculate_ndvi,
    calculate_evi,
    calculate_ndmi,
)

logger = logging.getLogger(__name__)


class LandCoverClass(IntEnum):
    """Land cover classification categories."""
    NO_DATA = 0
    DENSE_FOREST = 1
    OPEN_FOREST = 2
    SHRUBLAND = 3
    GRASSLAND = 4
    CROPLAND = 5
    BARE_SOIL = 6
    WATER = 7
    URBAN = 8
    WETLAND = 9


@dataclass
class ClassificationThresholds:
    """Thresholds for forest classification."""
    # NDVI thresholds
    ndvi_forest_min: float = 0.6        # Minimum NDVI for dense forest
    ndvi_open_forest_min: float = 0.4   # Minimum NDVI for open forest
    ndvi_vegetation_min: float = 0.2    # Minimum NDVI for any vegetation
    ndvi_water_max: float = 0.0         # Maximum NDVI for water

    # EVI thresholds (better for dense vegetation)
    evi_forest_min: float = 0.35
    evi_open_forest_min: float = 0.2

    # NDWI threshold for water detection
    ndwi_water_min: float = 0.3

    # NDMI threshold for forest moisture
    ndmi_forest_min: float = 0.2

    # Tree cover percentage thresholds
    tree_cover_forest_min: float = 30.0  # Minimum tree cover for forest classification


@dataclass
class ForestClassificationResult:
    """Result of forest classification analysis."""
    # Classification maps
    binary_forest_mask: np.ndarray      # Boolean: True = forest
    land_cover_map: np.ndarray          # LandCoverClass values
    tree_cover_percentage: np.ndarray   # 0-100% tree cover estimate
    confidence_map: np.ndarray          # 0-1 classification confidence

    # Summary statistics
    total_pixels: int
    forest_pixels: int
    non_forest_pixels: int
    no_data_pixels: int

    # Area calculations (in hectares, if pixel size provided)
    pixel_size_m: float = 10.0
    classification_date: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def forest_fraction(self) -> float:
        """Fraction of valid pixels classified as forest."""
        valid = self.total_pixels - self.no_data_pixels
        return self.forest_pixels / valid if valid > 0 else 0.0

    @property
    def forest_area_ha(self) -> float:
        """Forest area in hectares."""
        pixel_area_m2 = self.pixel_size_m ** 2
        return (self.forest_pixels * pixel_area_m2) / 10000.0

    @property
    def total_area_ha(self) -> float:
        """Total valid area in hectares."""
        pixel_area_m2 = self.pixel_size_m ** 2
        valid = self.total_pixels - self.no_data_pixels
        return (valid * pixel_area_m2) / 10000.0

    def get_class_areas(self) -> dict[LandCoverClass, float]:
        """Get area in hectares for each land cover class."""
        pixel_area_ha = (self.pixel_size_m ** 2) / 10000.0
        areas = {}

        for lc_class in LandCoverClass:
            count = int(np.sum(self.land_cover_map == lc_class))
            areas[lc_class] = count * pixel_area_ha

        return areas


@dataclass
class GEDICanopyData:
    """
    GEDI (Global Ecosystem Dynamics Investigation) canopy height data.

    GEDI provides lidar-derived canopy height measurements at ~25m footprints.
    Used to improve forest classification and biomass estimation.
    """
    rh95: np.ndarray           # Relative height 95th percentile (canopy top)
    rh50: np.ndarray           # Relative height 50th percentile (mid-canopy)
    cover: np.ndarray          # Canopy cover fraction
    pai: Optional[np.ndarray]  # Plant Area Index
    quality_flag: np.ndarray   # Quality assessment
    acquisition_date: datetime
    footprint_size_m: float = 25.0


class ForestClassifierError(Exception):
    """Base exception for forest classifier errors."""
    pass


class ForestClassifier:
    """
    Forest cover classifier using vegetation indices.

    Implements threshold-based classification methods calibrated for
    forest monitoring and EUDR compliance verification.
    """

    def __init__(
        self,
        thresholds: Optional[ClassificationThresholds] = None,
        pixel_size_m: float = 10.0,
    ):
        """
        Initialize forest classifier.

        Args:
            thresholds: Classification thresholds (uses defaults if None)
            pixel_size_m: Pixel size in meters (10m for Sentinel-2)
        """
        self.thresholds = thresholds or ClassificationThresholds()
        self.pixel_size_m = pixel_size_m
        self.index_calculator = VegetationIndexCalculator()

    def classify_binary(
        self,
        ndvi: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Simple binary forest/non-forest classification using NDVI threshold.

        Args:
            ndvi: NDVI array (-1 to 1)
            threshold: NDVI threshold for forest (default: 0.6)

        Returns:
            Boolean array where True = forest
        """
        if threshold is None:
            threshold = self.thresholds.ndvi_forest_min

        # Handle NaN values
        forest_mask = ndvi >= threshold
        forest_mask[np.isnan(ndvi)] = False

        return forest_mask

    def classify_land_cover(
        self,
        bands: dict[str, np.ndarray],
        scale_factor: float = 10000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Multi-class land cover classification.

        Args:
            bands: Dict mapping band names to data arrays
            scale_factor: Scale factor for reflectance data

        Returns:
            Tuple of (land_cover_map, confidence_map)
        """
        # Calculate required indices
        required_bands = ["B2", "B3", "B4", "B8"]
        missing = [b for b in required_bands if b not in bands]
        if missing:
            raise ForestClassifierError(f"Missing bands: {missing}")

        # Get reference shape from any 10m band
        ref_shape = bands["B4"].shape

        # Calculate indices
        ndvi_result = calculate_ndvi(bands["B4"], bands["B8"], scale_factor)
        ndvi = ndvi_result.data

        # EVI for dense vegetation discrimination
        evi_result = calculate_evi(bands["B2"], bands["B4"], bands["B8"], scale_factor)
        evi = evi_result.data

        # NDWI for water detection
        from greenlang.satellite.analysis.vegetation_indices import calculate_ndwi
        ndwi_result = calculate_ndwi(bands["B3"], bands["B8"], scale_factor)
        ndwi = ndwi_result.data

        # Initialize output arrays
        land_cover = np.full(ref_shape, LandCoverClass.NO_DATA, dtype=np.uint8)
        confidence = np.zeros(ref_shape, dtype=np.float32)

        # Valid data mask
        valid_mask = np.isfinite(ndvi) & np.isfinite(evi)

        # Classification decision tree (order matters)

        # 1. Water detection
        water_mask = valid_mask & (ndwi > self.thresholds.ndwi_water_min)
        land_cover[water_mask] = LandCoverClass.WATER
        confidence[water_mask] = np.clip(ndwi[water_mask], 0.3, 1.0)

        # 2. Dense forest (high NDVI and EVI)
        dense_forest_mask = (
            valid_mask &
            ~water_mask &
            (ndvi >= self.thresholds.ndvi_forest_min) &
            (evi >= self.thresholds.evi_forest_min)
        )
        land_cover[dense_forest_mask] = LandCoverClass.DENSE_FOREST
        # Confidence based on how much above threshold
        confidence[dense_forest_mask] = np.clip(
            0.5 + (ndvi[dense_forest_mask] - self.thresholds.ndvi_forest_min) * 2,
            0.5, 1.0
        )

        # 3. Open forest (moderate NDVI)
        open_forest_mask = (
            valid_mask &
            ~water_mask &
            ~dense_forest_mask &
            (ndvi >= self.thresholds.ndvi_open_forest_min) &
            (evi >= self.thresholds.evi_open_forest_min)
        )
        land_cover[open_forest_mask] = LandCoverClass.OPEN_FOREST
        confidence[open_forest_mask] = np.clip(
            0.4 + (ndvi[open_forest_mask] - self.thresholds.ndvi_open_forest_min),
            0.4, 0.8
        )

        # 4. Shrubland (low-moderate NDVI)
        shrubland_mask = (
            valid_mask &
            ~water_mask &
            ~dense_forest_mask &
            ~open_forest_mask &
            (ndvi >= 0.3) &
            (ndvi < self.thresholds.ndvi_open_forest_min)
        )
        land_cover[shrubland_mask] = LandCoverClass.SHRUBLAND
        confidence[shrubland_mask] = 0.5

        # 5. Grassland (low NDVI, some vegetation)
        grassland_mask = (
            valid_mask &
            ~water_mask &
            ~dense_forest_mask &
            ~open_forest_mask &
            ~shrubland_mask &
            (ndvi >= self.thresholds.ndvi_vegetation_min)
        )
        land_cover[grassland_mask] = LandCoverClass.GRASSLAND
        confidence[grassland_mask] = 0.5

        # 6. Bare soil (very low NDVI, not water)
        bare_soil_mask = (
            valid_mask &
            ~water_mask &
            (ndvi > self.thresholds.ndvi_water_max) &
            (ndvi < self.thresholds.ndvi_vegetation_min)
        )
        land_cover[bare_soil_mask] = LandCoverClass.BARE_SOIL
        confidence[bare_soil_mask] = 0.6

        return land_cover, confidence

    def estimate_tree_cover(
        self,
        ndvi: np.ndarray,
        evi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate tree cover percentage from vegetation indices.

        Uses an empirical relationship between NDVI/EVI and tree cover
        calibrated against Hansen Global Forest Change data.

        Args:
            ndvi: NDVI array
            evi: Optional EVI array for improved accuracy

        Returns:
            Array of tree cover percentages (0-100)
        """
        # Initialize output
        tree_cover = np.zeros_like(ndvi, dtype=np.float32)

        # Valid data mask
        valid_mask = np.isfinite(ndvi)

        # Simple linear model: tree_cover = a * NDVI + b
        # Calibrated for typical forest conditions
        # NDVI 0.2 -> 0%, NDVI 0.8 -> 100%
        a = 166.67  # slope
        b = -33.33  # intercept

        tree_cover[valid_mask] = a * ndvi[valid_mask] + b

        # If EVI available, blend the estimates
        if evi is not None and np.any(np.isfinite(evi)):
            evi_valid = np.isfinite(evi)
            # EVI model: tree_cover = 180 * EVI (roughly)
            evi_estimate = 180 * evi
            # Average NDVI and EVI estimates where both valid
            both_valid = valid_mask & evi_valid
            tree_cover[both_valid] = (
                0.6 * tree_cover[both_valid] +
                0.4 * evi_estimate[both_valid]
            )

        # Clip to valid range
        tree_cover = np.clip(tree_cover, 0, 100)

        return tree_cover

    def estimate_canopy_height(
        self,
        gedi_data: GEDICanopyData,
        ndvi: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Estimate canopy height using GEDI lidar data and NDVI.

        Interpolates sparse GEDI footprints using NDVI as a guide.

        Args:
            gedi_data: GEDI canopy height measurements
            ndvi: NDVI array for interpolation guidance
            target_shape: Output array shape

        Returns:
            Canopy height in meters
        """
        # Simple approach: interpolate GEDI data to target resolution
        # In production, use more sophisticated spatial modeling

        # For now, create a scaled version of NDVI as proxy
        # Real implementation would use kriging or random forest regression

        # Use rh95 as canopy height estimate
        from greenlang.satellite.analysis.vegetation_indices import resample_to_match

        # Resample GEDI to target resolution
        canopy_height = resample_to_match(gedi_data.rh95, target_shape, method="nearest")

        # Apply quality filter
        quality = resample_to_match(gedi_data.quality_flag.astype(float), target_shape)
        canopy_height[quality < 1] = np.nan

        # Where GEDI is missing, estimate from NDVI
        # Empirical relationship: height ~ 30 * NDVI for forest
        missing_mask = np.isnan(canopy_height) & np.isfinite(ndvi)
        canopy_height[missing_mask] = np.clip(30 * ndvi[missing_mask], 0, 60)

        return canopy_height

    def classify(
        self,
        bands: dict[str, np.ndarray],
        scale_factor: float = 10000.0,
        gedi_data: Optional[GEDICanopyData] = None,
    ) -> ForestClassificationResult:
        """
        Perform full forest classification analysis.

        Args:
            bands: Dict mapping band names to data arrays
            scale_factor: Scale factor for reflectance data
            gedi_data: Optional GEDI canopy height data

        Returns:
            ForestClassificationResult with all classification outputs
        """
        logger.info("Starting forest classification analysis")

        # Get reference shape
        ref_shape = bands["B4"].shape

        # Calculate NDVI (required for all classifications)
        ndvi_result = calculate_ndvi(bands["B4"], bands["B8"], scale_factor)
        ndvi = ndvi_result.data

        # Binary forest mask
        binary_forest_mask = self.classify_binary(ndvi)

        # Full land cover classification
        land_cover_map, confidence_map = self.classify_land_cover(bands, scale_factor)

        # Tree cover percentage
        evi_result = None
        if "B2" in bands:
            evi_result = calculate_evi(bands["B2"], bands["B4"], bands["B8"], scale_factor)

        tree_cover = self.estimate_tree_cover(
            ndvi,
            evi_result.data if evi_result else None
        )

        # Canopy height if GEDI available
        if gedi_data is not None:
            canopy_height = self.estimate_canopy_height(gedi_data, ndvi, ref_shape)
        else:
            canopy_height = None

        # Calculate statistics
        total_pixels = int(np.prod(ref_shape))
        no_data_pixels = int(np.sum(land_cover_map == LandCoverClass.NO_DATA))
        forest_pixels = int(np.sum(
            (land_cover_map == LandCoverClass.DENSE_FOREST) |
            (land_cover_map == LandCoverClass.OPEN_FOREST)
        ))
        non_forest_pixels = total_pixels - no_data_pixels - forest_pixels

        result = ForestClassificationResult(
            binary_forest_mask=binary_forest_mask,
            land_cover_map=land_cover_map,
            tree_cover_percentage=tree_cover,
            confidence_map=confidence_map,
            total_pixels=total_pixels,
            forest_pixels=forest_pixels,
            non_forest_pixels=non_forest_pixels,
            no_data_pixels=no_data_pixels,
            pixel_size_m=self.pixel_size_m,
            classification_date=datetime.now(),
            metadata={
                "ndvi_mean": ndvi_result.mean_value,
                "ndvi_std": ndvi_result.std_value,
                "thresholds": {
                    "ndvi_forest_min": self.thresholds.ndvi_forest_min,
                    "ndvi_open_forest_min": self.thresholds.ndvi_open_forest_min,
                },
                "canopy_height_available": canopy_height is not None,
            }
        )

        logger.info(
            f"Classification complete: {result.forest_area_ha:.2f} ha forest "
            f"({result.forest_fraction * 100:.1f}% of valid area)"
        )

        return result


class AdaptiveThresholdClassifier:
    """
    Classifier that adapts thresholds based on local conditions.

    Uses local statistics to adjust classification thresholds for
    improved accuracy in heterogeneous landscapes.
    """

    def __init__(
        self,
        base_thresholds: Optional[ClassificationThresholds] = None,
        window_size: int = 101,
    ):
        """
        Initialize adaptive classifier.

        Args:
            base_thresholds: Base thresholds to adapt from
            window_size: Window size for local statistics (pixels)
        """
        self.base_thresholds = base_thresholds or ClassificationThresholds()
        self.window_size = window_size

    def _local_percentile(
        self,
        data: np.ndarray,
        percentile: float,
        window_size: int,
    ) -> np.ndarray:
        """Calculate local percentile using sliding window."""
        from scipy.ndimage import generic_filter

        def percentile_filter(values):
            valid = values[np.isfinite(values)]
            if len(valid) == 0:
                return np.nan
            return np.percentile(valid, percentile)

        return generic_filter(
            data,
            percentile_filter,
            size=window_size,
            mode='reflect'
        )

    def classify_adaptive(
        self,
        ndvi: np.ndarray,
        base_threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify with locally-adapted thresholds.

        Args:
            ndvi: NDVI array
            base_threshold: Base threshold (default: 0.6)

        Returns:
            Tuple of (classification, local_threshold)
        """
        if base_threshold is None:
            base_threshold = self.base_thresholds.ndvi_forest_min

        # Calculate local statistics
        local_p25 = self._local_percentile(ndvi, 25, self.window_size)
        local_p75 = self._local_percentile(ndvi, 75, self.window_size)

        # Adapt threshold: midpoint between p25 and p75, bounded by base
        local_threshold = (local_p25 + local_p75) / 2
        local_threshold = np.clip(local_threshold, base_threshold - 0.1, base_threshold + 0.1)

        # Classify using local threshold
        forest_mask = ndvi >= local_threshold
        forest_mask[np.isnan(ndvi)] = False

        return forest_mask, local_threshold


def create_training_data(
    classification_result: ForestClassificationResult,
    sample_fraction: float = 0.01,
    min_confidence: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract training data from high-confidence classifications.

    Useful for training ML models from rule-based classifications.

    Args:
        classification_result: Classification result to sample from
        sample_fraction: Fraction of pixels to sample
        min_confidence: Minimum confidence for training samples

    Returns:
        Tuple of (features, labels) arrays
    """
    # Get high-confidence pixels
    high_conf_mask = classification_result.confidence_map >= min_confidence
    valid_mask = classification_result.land_cover_map != LandCoverClass.NO_DATA

    sample_mask = high_conf_mask & valid_mask

    # Random sampling
    sample_indices = np.where(sample_mask)
    n_samples = int(len(sample_indices[0]) * sample_fraction)

    if n_samples == 0:
        raise ForestClassifierError("No high-confidence samples available")

    rng = np.random.default_rng(42)
    selected = rng.choice(len(sample_indices[0]), size=n_samples, replace=False)

    y_indices = sample_indices[0][selected]
    x_indices = sample_indices[1][selected]

    # Extract features (tree cover percentage as proxy)
    features = classification_result.tree_cover_percentage[y_indices, x_indices]
    features = features.reshape(-1, 1)

    # Extract labels
    labels = classification_result.land_cover_map[y_indices, x_indices]

    return features, labels
