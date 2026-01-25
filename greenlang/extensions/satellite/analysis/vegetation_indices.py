"""
Vegetation Indices Calculator for Satellite Imagery Analysis.

Provides calculations for common vegetation indices used in
forest monitoring and deforestation detection:

- NDVI: Normalized Difference Vegetation Index
- EVI: Enhanced Vegetation Index
- NDWI: Normalized Difference Water Index
- NBR: Normalized Burn Ratio
- SAVI: Soil Adjusted Vegetation Index
- MSAVI: Modified Soil Adjusted Vegetation Index

All indices are calculated using standard formulas with proper
handling of edge cases and no-data values.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported vegetation index types."""
    NDVI = "ndvi"
    EVI = "evi"
    NDWI = "ndwi"
    NBR = "nbr"
    SAVI = "savi"
    MSAVI = "msavi"
    NDMI = "ndmi"  # Normalized Difference Moisture Index


@dataclass
class IndexResult:
    """Container for vegetation index calculation result."""
    index_type: IndexType
    data: np.ndarray
    valid_pixel_count: int
    min_value: float
    max_value: float
    mean_value: float
    std_value: float

    @property
    def valid_fraction(self) -> float:
        """Fraction of valid (non-NaN) pixels."""
        total = self.data.size
        return self.valid_pixel_count / total if total > 0 else 0.0


class VegetationIndicesError(Exception):
    """Base exception for vegetation index calculation errors."""
    pass


class MissingBandError(VegetationIndicesError):
    """Required band is missing."""
    pass


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Perform division with proper handling of division by zero.

    Returns NaN where denominator is zero or very small.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        # Set invalid values to NaN
        result[~np.isfinite(result)] = np.nan
        # Also set to NaN where denominator is effectively zero
        result[np.abs(denominator) < 1e-10] = np.nan
    return result


def _normalize_reflectance(band_data: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    """
    Normalize band data to 0-1 reflectance range.

    Assumes input is integer values (e.g., 0-10000 for Sentinel-2 L2A).
    """
    return band_data.astype(np.float64) / scale_factor


def _calculate_stats(data: np.ndarray) -> tuple[int, float, float, float, float]:
    """Calculate statistics for index data, ignoring NaN values."""
    valid_mask = np.isfinite(data)
    valid_count = int(np.sum(valid_mask))

    if valid_count == 0:
        return 0, np.nan, np.nan, np.nan, np.nan

    valid_data = data[valid_mask]
    return (
        valid_count,
        float(np.min(valid_data)),
        float(np.max(valid_data)),
        float(np.mean(valid_data)),
        float(np.std(valid_data)),
    )


def calculate_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    scale_factor: float = 10000.0,
) -> IndexResult:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Values range from -1 to 1:
    - Dense vegetation: 0.6 to 0.9
    - Sparse vegetation: 0.2 to 0.5
    - Bare soil: 0.1 to 0.2
    - Water: < 0
    - Clouds/snow: near 0

    Args:
        red: Red band data (Sentinel-2 B4 or Landsat B4)
        nir: NIR band data (Sentinel-2 B8 or Landsat B5)
        scale_factor: Scale factor for input data (10000 for L2A)

    Returns:
        IndexResult with NDVI values
    """
    if red.shape != nir.shape:
        raise VegetationIndicesError(
            f"Band shape mismatch: red={red.shape}, nir={nir.shape}"
        )

    # Convert to float and normalize
    red_norm = _normalize_reflectance(red, scale_factor)
    nir_norm = _normalize_reflectance(nir, scale_factor)

    # Calculate NDVI
    numerator = nir_norm - red_norm
    denominator = nir_norm + red_norm

    ndvi = _safe_divide(numerator, denominator)

    # Clip to valid range
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Calculate statistics
    valid_count, min_val, max_val, mean_val, std_val = _calculate_stats(ndvi)

    logger.debug(f"NDVI calculated: mean={mean_val:.3f}, valid pixels={valid_count}")

    return IndexResult(
        index_type=IndexType.NDVI,
        data=ndvi,
        valid_pixel_count=valid_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
    )


def calculate_evi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    scale_factor: float = 10000.0,
    gain: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
) -> IndexResult:
    """
    Calculate Enhanced Vegetation Index (EVI).

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

    EVI reduces atmospheric and soil background influences, making it
    more sensitive in high biomass regions where NDVI saturates.

    Values typically range from -1 to 1:
    - Dense vegetation: 0.3 to 0.8
    - Sparse vegetation: 0.1 to 0.3
    - Bare soil/water: < 0.1

    Args:
        blue: Blue band data (Sentinel-2 B2 or Landsat B2)
        red: Red band data (Sentinel-2 B4 or Landsat B4)
        nir: NIR band data (Sentinel-2 B8 or Landsat B5)
        scale_factor: Scale factor for input data
        gain: Gain factor (default 2.5)
        c1: Coefficient for red band (default 6.0)
        c2: Coefficient for blue band (default 7.5)
        l: Canopy background adjustment (default 1.0)

    Returns:
        IndexResult with EVI values
    """
    if not (blue.shape == red.shape == nir.shape):
        raise VegetationIndicesError(
            f"Band shape mismatch: blue={blue.shape}, red={red.shape}, nir={nir.shape}"
        )

    # Normalize
    blue_norm = _normalize_reflectance(blue, scale_factor)
    red_norm = _normalize_reflectance(red, scale_factor)
    nir_norm = _normalize_reflectance(nir, scale_factor)

    # Calculate EVI
    numerator = nir_norm - red_norm
    denominator = nir_norm + c1 * red_norm - c2 * blue_norm + l

    evi = gain * _safe_divide(numerator, denominator)

    # Clip to reasonable range
    evi = np.clip(evi, -1.0, 1.0)

    valid_count, min_val, max_val, mean_val, std_val = _calculate_stats(evi)

    logger.debug(f"EVI calculated: mean={mean_val:.3f}, valid pixels={valid_count}")

    return IndexResult(
        index_type=IndexType.EVI,
        data=evi,
        valid_pixel_count=valid_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
    )


def calculate_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    scale_factor: float = 10000.0,
) -> IndexResult:
    """
    Calculate Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)

    Also known as Gao NDWI. Used for water body detection and
    vegetation water content assessment.

    Values range from -1 to 1:
    - Water: > 0.3
    - Vegetation: < 0
    - Bare soil: near 0

    Args:
        green: Green band data (Sentinel-2 B3 or Landsat B3)
        nir: NIR band data (Sentinel-2 B8 or Landsat B5)
        scale_factor: Scale factor for input data

    Returns:
        IndexResult with NDWI values
    """
    if green.shape != nir.shape:
        raise VegetationIndicesError(
            f"Band shape mismatch: green={green.shape}, nir={nir.shape}"
        )

    green_norm = _normalize_reflectance(green, scale_factor)
    nir_norm = _normalize_reflectance(nir, scale_factor)

    numerator = green_norm - nir_norm
    denominator = green_norm + nir_norm

    ndwi = _safe_divide(numerator, denominator)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    valid_count, min_val, max_val, mean_val, std_val = _calculate_stats(ndwi)

    logger.debug(f"NDWI calculated: mean={mean_val:.3f}, valid pixels={valid_count}")

    return IndexResult(
        index_type=IndexType.NDWI,
        data=ndwi,
        valid_pixel_count=valid_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
    )


def calculate_nbr(
    nir: np.ndarray,
    swir2: np.ndarray,
    scale_factor: float = 10000.0,
) -> IndexResult:
    """
    Calculate Normalized Burn Ratio (NBR).

    NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Used for burn severity assessment and forest disturbance detection.
    Also useful for detecting deforestation as it responds to
    changes in vegetation structure.

    Values range from -1 to 1:
    - Healthy vegetation: 0.2 to 0.9
    - Bare/burned areas: < 0
    - Recently burned: < -0.2

    Args:
        nir: NIR band data (Sentinel-2 B8 or Landsat B5)
        swir2: SWIR2 band data (Sentinel-2 B12 or Landsat B7)
        scale_factor: Scale factor for input data

    Returns:
        IndexResult with NBR values
    """
    if nir.shape != swir2.shape:
        raise VegetationIndicesError(
            f"Band shape mismatch: nir={nir.shape}, swir2={swir2.shape}"
        )

    nir_norm = _normalize_reflectance(nir, scale_factor)
    swir2_norm = _normalize_reflectance(swir2, scale_factor)

    numerator = nir_norm - swir2_norm
    denominator = nir_norm + swir2_norm

    nbr = _safe_divide(numerator, denominator)
    nbr = np.clip(nbr, -1.0, 1.0)

    valid_count, min_val, max_val, mean_val, std_val = _calculate_stats(nbr)

    logger.debug(f"NBR calculated: mean={mean_val:.3f}, valid pixels={valid_count}")

    return IndexResult(
        index_type=IndexType.NBR,
        data=nbr,
        valid_pixel_count=valid_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
    )


def calculate_savi(
    red: np.ndarray,
    nir: np.ndarray,
    scale_factor: float = 10000.0,
    l: float = 0.5,
) -> IndexResult:
    """
    Calculate Soil Adjusted Vegetation Index (SAVI).

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    Minimizes soil brightness influences on vegetation indices.
    Use when vegetation cover is sparse (20-50%).

    Values range from -1 to 1 (similar interpretation to NDVI).

    Args:
        red: Red band data
        nir: NIR band data
        scale_factor: Scale factor for input data
        l: Soil brightness correction factor (0.5 for intermediate vegetation)

    Returns:
        IndexResult with SAVI values
    """
    if red.shape != nir.shape:
        raise VegetationIndicesError(
            f"Band shape mismatch: red={red.shape}, nir={nir.shape}"
        )

    red_norm = _normalize_reflectance(red, scale_factor)
    nir_norm = _normalize_reflectance(nir, scale_factor)

    numerator = nir_norm - red_norm
    denominator = nir_norm + red_norm + l

    savi = _safe_divide(numerator, denominator) * (1 + l)
    savi = np.clip(savi, -1.0, 1.0)

    valid_count, min_val, max_val, mean_val, std_val = _calculate_stats(savi)

    logger.debug(f"SAVI calculated: mean={mean_val:.3f}, valid pixels={valid_count}")

    return IndexResult(
        index_type=IndexType.SAVI,
        data=savi,
        valid_pixel_count=valid_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
    )


def calculate_msavi(
    red: np.ndarray,
    nir: np.ndarray,
    scale_factor: float = 10000.0,
) -> IndexResult:
    """
    Calculate Modified Soil Adjusted Vegetation Index (MSAVI2).

    MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2

    Self-adjusting soil correction factor that doesn't require
    prior knowledge of vegetation amount.

    Values range from -1 to 1.

    Args:
        red: Red band data
        nir: NIR band data
        scale_factor: Scale factor for input data

    Returns:
        IndexResult with MSAVI values
    """
    if red.shape != nir.shape:
        raise VegetationIndicesError(
            f"Band shape mismatch: red={red.shape}, nir={nir.shape}"
        )

    red_norm = _normalize_reflectance(red, scale_factor)
    nir_norm = _normalize_reflectance(nir, scale_factor)

    # MSAVI2 formula
    term1 = 2 * nir_norm + 1
    term2 = np.sqrt(np.maximum(0, term1 ** 2 - 8 * (nir_norm - red_norm)))

    msavi = (term1 - term2) / 2
    msavi = np.clip(msavi, -1.0, 1.0)

    valid_count, min_val, max_val, mean_val, std_val = _calculate_stats(msavi)

    logger.debug(f"MSAVI calculated: mean={mean_val:.3f}, valid pixels={valid_count}")

    return IndexResult(
        index_type=IndexType.MSAVI,
        data=msavi,
        valid_pixel_count=valid_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
    )


def calculate_ndmi(
    nir: np.ndarray,
    swir1: np.ndarray,
    scale_factor: float = 10000.0,
) -> IndexResult:
    """
    Calculate Normalized Difference Moisture Index (NDMI).

    NDMI = (NIR - SWIR1) / (NIR + SWIR1)

    Also known as Normalized Difference Water Index (NDWI-Gao).
    Sensitive to vegetation water content and canopy moisture stress.

    Values range from -1 to 1:
    - High moisture: > 0.3
    - Moderate moisture: 0.0 to 0.3
    - Low moisture/stressed: < 0

    Args:
        nir: NIR band data (Sentinel-2 B8 or Landsat B5)
        swir1: SWIR1 band data (Sentinel-2 B11 or Landsat B6)
        scale_factor: Scale factor for input data

    Returns:
        IndexResult with NDMI values
    """
    if nir.shape != swir1.shape:
        raise VegetationIndicesError(
            f"Band shape mismatch: nir={nir.shape}, swir1={swir1.shape}"
        )

    nir_norm = _normalize_reflectance(nir, scale_factor)
    swir1_norm = _normalize_reflectance(swir1, scale_factor)

    numerator = nir_norm - swir1_norm
    denominator = nir_norm + swir1_norm

    ndmi = _safe_divide(numerator, denominator)
    ndmi = np.clip(ndmi, -1.0, 1.0)

    valid_count, min_val, max_val, mean_val, std_val = _calculate_stats(ndmi)

    logger.debug(f"NDMI calculated: mean={mean_val:.3f}, valid pixels={valid_count}")

    return IndexResult(
        index_type=IndexType.NDMI,
        data=ndmi,
        valid_pixel_count=valid_count,
        min_value=min_val,
        max_value=max_val,
        mean_value=mean_val,
        std_value=std_val,
    )


class VegetationIndexCalculator:
    """
    Calculator for multiple vegetation indices from satellite imagery.

    Provides a unified interface for calculating all supported indices
    from Sentinel-2 or Landsat imagery.
    """

    # Band mapping for supported indices
    REQUIRED_BANDS = {
        IndexType.NDVI: ["B4", "B8"],       # Red, NIR
        IndexType.EVI: ["B2", "B4", "B8"],  # Blue, Red, NIR
        IndexType.NDWI: ["B3", "B8"],       # Green, NIR
        IndexType.NBR: ["B8", "B12"],       # NIR, SWIR2
        IndexType.SAVI: ["B4", "B8"],       # Red, NIR
        IndexType.MSAVI: ["B4", "B8"],      # Red, NIR
        IndexType.NDMI: ["B8", "B11"],      # NIR, SWIR1
    }

    def __init__(self, scale_factor: float = 10000.0):
        """
        Initialize calculator.

        Args:
            scale_factor: Scale factor for input reflectance data
        """
        self.scale_factor = scale_factor

    def get_required_bands(self, index_type: IndexType) -> list[str]:
        """Get list of required band names for an index type."""
        return self.REQUIRED_BANDS.get(index_type, [])

    def check_available_indices(
        self,
        available_bands: list[str],
    ) -> list[IndexType]:
        """
        Check which indices can be calculated with available bands.

        Args:
            available_bands: List of available band names

        Returns:
            List of IndexType that can be calculated
        """
        available_set = set(available_bands)
        computable = []

        for index_type, required in self.REQUIRED_BANDS.items():
            if all(band in available_set for band in required):
                computable.append(index_type)

        return computable

    def calculate(
        self,
        index_type: IndexType,
        bands: dict[str, np.ndarray],
    ) -> IndexResult:
        """
        Calculate a vegetation index.

        Args:
            index_type: Type of index to calculate
            bands: Dict mapping band names to data arrays

        Returns:
            IndexResult with calculated index

        Raises:
            MissingBandError: If required band is not available
        """
        required = self.REQUIRED_BANDS.get(index_type)
        if required is None:
            raise VegetationIndicesError(f"Unknown index type: {index_type}")

        # Check for missing bands
        missing = [b for b in required if b not in bands]
        if missing:
            raise MissingBandError(f"Missing bands for {index_type.value}: {missing}")

        # Route to appropriate calculation function
        if index_type == IndexType.NDVI:
            return calculate_ndvi(
                bands["B4"], bands["B8"], self.scale_factor
            )
        elif index_type == IndexType.EVI:
            return calculate_evi(
                bands["B2"], bands["B4"], bands["B8"], self.scale_factor
            )
        elif index_type == IndexType.NDWI:
            return calculate_ndwi(
                bands["B3"], bands["B8"], self.scale_factor
            )
        elif index_type == IndexType.NBR:
            return calculate_nbr(
                bands["B8"], bands["B12"], self.scale_factor
            )
        elif index_type == IndexType.SAVI:
            return calculate_savi(
                bands["B4"], bands["B8"], self.scale_factor
            )
        elif index_type == IndexType.MSAVI:
            return calculate_msavi(
                bands["B4"], bands["B8"], self.scale_factor
            )
        elif index_type == IndexType.NDMI:
            return calculate_ndmi(
                bands["B8"], bands["B11"], self.scale_factor
            )
        else:
            raise VegetationIndicesError(f"Calculation not implemented for {index_type}")

    def calculate_all(
        self,
        bands: dict[str, np.ndarray],
    ) -> dict[IndexType, IndexResult]:
        """
        Calculate all possible indices from available bands.

        Args:
            bands: Dict mapping band names to data arrays

        Returns:
            Dict mapping IndexType to IndexResult
        """
        available = self.check_available_indices(list(bands.keys()))
        results = {}

        for index_type in available:
            try:
                results[index_type] = self.calculate(index_type, bands)
            except VegetationIndicesError as e:
                logger.warning(f"Failed to calculate {index_type.value}: {e}")

        return results


def resample_to_match(
    source: np.ndarray,
    target_shape: tuple[int, int],
    method: str = "nearest",
) -> np.ndarray:
    """
    Resample array to match target shape.

    Used to align bands with different resolutions (e.g., 20m to 10m).

    Args:
        source: Source array to resample
        target_shape: Target (height, width)
        method: Resampling method ("nearest" or "bilinear")

    Returns:
        Resampled array
    """
    if source.shape == target_shape:
        return source

    if method == "nearest":
        # Nearest neighbor resampling
        y_ratio = source.shape[0] / target_shape[0]
        x_ratio = source.shape[1] / target_shape[1]

        y_indices = (np.arange(target_shape[0]) * y_ratio).astype(int)
        x_indices = (np.arange(target_shape[1]) * x_ratio).astype(int)

        y_indices = np.clip(y_indices, 0, source.shape[0] - 1)
        x_indices = np.clip(x_indices, 0, source.shape[1] - 1)

        return source[np.ix_(y_indices, x_indices)]

    elif method == "bilinear":
        # Simple bilinear interpolation
        from scipy import ndimage
        zoom_y = target_shape[0] / source.shape[0]
        zoom_x = target_shape[1] / source.shape[1]
        return ndimage.zoom(source.astype(float), (zoom_y, zoom_x), order=1)

    else:
        raise ValueError(f"Unknown resampling method: {method}")
