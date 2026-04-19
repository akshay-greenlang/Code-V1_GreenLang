"""
Change Detection Module for Deforestation Analysis.

Provides bi-temporal and multi-temporal change detection algorithms
for identifying forest loss, degradation, and land use change.

Methods:
- Image differencing for NDVI change
- Bi-temporal change detection
- Multi-temporal trend analysis
- Forest loss area calculation
- Degradation vs. clear-cut discrimination

All area calculations are in hectares for EUDR compliance reporting.
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
    calculate_ndvi,
    calculate_nbr,
)
from greenlang.satellite.models.forest_classifier import (
    ForestClassificationResult,
    ForestClassifier,
    LandCoverClass,
)

logger = logging.getLogger(__name__)


class ChangeType(IntEnum):
    """Forest change classification types."""
    NO_CHANGE = 0
    CLEAR_CUT = 1           # Complete forest removal (> 90% canopy loss)
    DEGRADATION = 2         # Partial canopy loss (30-90%)
    PARTIAL_LOSS = 3        # Minor loss (< 30%)
    REGROWTH = 4            # Forest recovery
    NO_DATA = 255


@dataclass
class ChangeThresholds:
    """Thresholds for change detection."""
    # NDVI change thresholds
    ndvi_clear_cut: float = -0.3       # NDVI change for clear-cut detection
    ndvi_degradation: float = -0.15    # NDVI change for degradation
    ndvi_partial_loss: float = -0.05   # NDVI change for partial loss
    ndvi_regrowth: float = 0.1         # NDVI change for regrowth

    # NBR thresholds (better for burn severity)
    nbr_severe_change: float = -0.27   # dNBR for severe disturbance
    nbr_moderate_change: float = -0.1  # dNBR for moderate disturbance

    # Minimum forest NDVI for pre-change baseline
    min_forest_ndvi: float = 0.4

    # Confidence threshold
    min_confidence: float = 0.6


@dataclass
class ChangeDetectionResult:
    """Result of change detection analysis."""
    # Change maps
    change_magnitude: np.ndarray     # Continuous change magnitude (-1 to 1)
    change_type: np.ndarray          # ChangeType classification
    confidence: np.ndarray           # Detection confidence (0 to 1)

    # Area statistics (hectares)
    total_area_ha: float
    forest_loss_ha: float
    clear_cut_ha: float
    degradation_ha: float
    regrowth_ha: float
    no_change_ha: float

    # Temporal information
    pre_date: datetime
    post_date: datetime
    days_between: int

    # Pixel information
    pixel_size_m: float
    total_pixels: int
    valid_pixels: int

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def forest_loss_fraction(self) -> float:
        """Fraction of area experiencing forest loss."""
        if self.total_area_ha <= 0:
            return 0.0
        return self.forest_loss_ha / self.total_area_ha

    @property
    def annual_loss_rate(self) -> float:
        """Annualized forest loss rate (% per year)."""
        if self.days_between <= 0 or self.total_area_ha <= 0:
            return 0.0
        daily_rate = self.forest_loss_fraction / self.days_between
        return daily_rate * 365 * 100


class ChangeDetectionError(Exception):
    """Base exception for change detection errors."""
    pass


class TemporalMismatchError(ChangeDetectionError):
    """Pre and post images have mismatched dimensions."""
    pass


def calculate_ndvi_difference(
    pre_ndvi: np.ndarray,
    post_ndvi: np.ndarray,
) -> np.ndarray:
    """
    Calculate NDVI difference (dNDVI = post - pre).

    Negative values indicate vegetation loss.

    Args:
        pre_ndvi: Pre-change NDVI array
        post_ndvi: Post-change NDVI array

    Returns:
        NDVI difference array
    """
    if pre_ndvi.shape != post_ndvi.shape:
        raise TemporalMismatchError(
            f"Shape mismatch: pre={pre_ndvi.shape}, post={post_ndvi.shape}"
        )

    dndvi = post_ndvi - pre_ndvi

    # Propagate NaN values
    dndvi[np.isnan(pre_ndvi) | np.isnan(post_ndvi)] = np.nan

    return dndvi


def calculate_nbr_difference(
    pre_nbr: np.ndarray,
    post_nbr: np.ndarray,
) -> np.ndarray:
    """
    Calculate NBR difference (dNBR = pre - post).

    Note: dNBR convention is pre - post (opposite of dNDVI).
    Positive values indicate disturbance/loss.

    Args:
        pre_nbr: Pre-change NBR array
        post_nbr: Post-change NBR array

    Returns:
        NBR difference array
    """
    if pre_nbr.shape != post_nbr.shape:
        raise TemporalMismatchError(
            f"Shape mismatch: pre={pre_nbr.shape}, post={post_nbr.shape}"
        )

    # Note: dNBR = pre - post (positive = disturbance)
    dnbr = pre_nbr - post_nbr

    # Propagate NaN values
    dnbr[np.isnan(pre_nbr) | np.isnan(post_nbr)] = np.nan

    return dnbr


def classify_change(
    dndvi: np.ndarray,
    pre_ndvi: np.ndarray,
    thresholds: Optional[ChangeThresholds] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify change type based on dNDVI.

    Args:
        dndvi: NDVI difference array
        pre_ndvi: Pre-change NDVI for baseline verification
        thresholds: Change detection thresholds

    Returns:
        Tuple of (change_type, confidence) arrays
    """
    if thresholds is None:
        thresholds = ChangeThresholds()

    shape = dndvi.shape
    change_type = np.full(shape, ChangeType.NO_DATA, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)

    # Valid data mask (both pre and post valid)
    valid_mask = np.isfinite(dndvi) & np.isfinite(pre_ndvi)

    # Only consider areas that were forest before (pre_ndvi > threshold)
    was_forest = pre_ndvi >= thresholds.min_forest_ndvi

    # Classification logic
    # Clear-cut: severe NDVI decrease in forested areas
    clear_cut_mask = (
        valid_mask &
        was_forest &
        (dndvi <= thresholds.ndvi_clear_cut)
    )
    change_type[clear_cut_mask] = ChangeType.CLEAR_CUT
    # Confidence increases with magnitude of change
    confidence[clear_cut_mask] = np.clip(
        -dndvi[clear_cut_mask] / 0.5, 0.6, 1.0
    )

    # Degradation: moderate NDVI decrease in forested areas
    degradation_mask = (
        valid_mask &
        was_forest &
        ~clear_cut_mask &
        (dndvi <= thresholds.ndvi_degradation)
    )
    change_type[degradation_mask] = ChangeType.DEGRADATION
    confidence[degradation_mask] = np.clip(
        -dndvi[degradation_mask] / 0.3, 0.5, 0.9
    )

    # Partial loss: minor NDVI decrease
    partial_loss_mask = (
        valid_mask &
        was_forest &
        ~clear_cut_mask &
        ~degradation_mask &
        (dndvi <= thresholds.ndvi_partial_loss)
    )
    change_type[partial_loss_mask] = ChangeType.PARTIAL_LOSS
    confidence[partial_loss_mask] = 0.5

    # Regrowth: NDVI increase
    regrowth_mask = (
        valid_mask &
        (dndvi >= thresholds.ndvi_regrowth)
    )
    change_type[regrowth_mask] = ChangeType.REGROWTH
    confidence[regrowth_mask] = np.clip(
        dndvi[regrowth_mask] / 0.2, 0.4, 0.9
    )

    # No change: valid data but no significant change
    no_change_mask = (
        valid_mask &
        ~clear_cut_mask &
        ~degradation_mask &
        ~partial_loss_mask &
        ~regrowth_mask
    )
    change_type[no_change_mask] = ChangeType.NO_CHANGE
    confidence[no_change_mask] = 1.0 - np.abs(dndvi[no_change_mask])

    return change_type, confidence


class BiTemporalChangeDetector:
    """
    Bi-temporal change detection for forest loss analysis.

    Compares two images from different dates to detect changes.
    """

    def __init__(
        self,
        thresholds: Optional[ChangeThresholds] = None,
        pixel_size_m: float = 10.0,
    ):
        """
        Initialize change detector.

        Args:
            thresholds: Change detection thresholds
            pixel_size_m: Pixel size in meters
        """
        self.thresholds = thresholds or ChangeThresholds()
        self.pixel_size_m = pixel_size_m

    def detect_change(
        self,
        pre_bands: dict[str, np.ndarray],
        post_bands: dict[str, np.ndarray],
        pre_date: datetime,
        post_date: datetime,
        scale_factor: float = 10000.0,
    ) -> ChangeDetectionResult:
        """
        Detect changes between two image dates.

        Args:
            pre_bands: Pre-change band data
            post_bands: Post-change band data
            pre_date: Pre-change image date
            post_date: Post-change image date
            scale_factor: Scale factor for reflectance data

        Returns:
            ChangeDetectionResult with full analysis
        """
        logger.info(f"Detecting changes between {pre_date.date()} and {post_date.date()}")

        # Validate required bands
        required = ["B4", "B8"]
        for bands, name in [(pre_bands, "pre"), (post_bands, "post")]:
            missing = [b for b in required if b not in bands]
            if missing:
                raise ChangeDetectionError(f"Missing bands in {name} image: {missing}")

        # Calculate NDVI for both dates
        pre_ndvi = calculate_ndvi(pre_bands["B4"], pre_bands["B8"], scale_factor)
        post_ndvi = calculate_ndvi(post_bands["B4"], post_bands["B8"], scale_factor)

        # Ensure shapes match
        if pre_ndvi.data.shape != post_ndvi.data.shape:
            raise TemporalMismatchError(
                f"Image shapes don't match: {pre_ndvi.data.shape} vs {post_ndvi.data.shape}"
            )

        # Calculate NDVI difference
        dndvi = calculate_ndvi_difference(pre_ndvi.data, post_ndvi.data)

        # Calculate NBR difference if SWIR bands available
        dnbr = None
        if "B12" in pre_bands and "B12" in post_bands:
            # Resample B12 to match B4 resolution if needed
            pre_b12 = pre_bands["B12"]
            post_b12 = post_bands["B12"]

            if pre_b12.shape != pre_bands["B4"].shape:
                from greenlang.satellite.analysis.vegetation_indices import resample_to_match
                pre_b12 = resample_to_match(pre_b12, pre_bands["B4"].shape)
                post_b12 = resample_to_match(post_b12, post_bands["B4"].shape)

            pre_nbr = calculate_nbr(pre_bands["B8"], pre_b12, scale_factor)
            post_nbr = calculate_nbr(post_bands["B8"], post_b12, scale_factor)
            dnbr = calculate_nbr_difference(pre_nbr.data, post_nbr.data)

        # Classify change types
        change_type, confidence = classify_change(dndvi, pre_ndvi.data, self.thresholds)

        # Enhance confidence with NBR if available
        if dnbr is not None:
            # High dNBR increases confidence for forest loss detection
            high_dnbr = dnbr > self.thresholds.nbr_moderate_change
            confidence[high_dnbr & (change_type != ChangeType.NO_CHANGE)] *= 1.1
            confidence = np.clip(confidence, 0, 1)

        # Calculate area statistics
        pixel_area_ha = (self.pixel_size_m ** 2) / 10000.0
        total_pixels = int(np.prod(change_type.shape))
        valid_pixels = int(np.sum(change_type != ChangeType.NO_DATA))

        total_area_ha = valid_pixels * pixel_area_ha

        clear_cut_pixels = int(np.sum(change_type == ChangeType.CLEAR_CUT))
        degradation_pixels = int(np.sum(change_type == ChangeType.DEGRADATION))
        partial_loss_pixels = int(np.sum(change_type == ChangeType.PARTIAL_LOSS))
        regrowth_pixels = int(np.sum(change_type == ChangeType.REGROWTH))
        no_change_pixels = int(np.sum(change_type == ChangeType.NO_CHANGE))

        forest_loss_pixels = clear_cut_pixels + degradation_pixels + partial_loss_pixels

        days_between = (post_date - pre_date).days

        result = ChangeDetectionResult(
            change_magnitude=dndvi,
            change_type=change_type,
            confidence=confidence,
            total_area_ha=total_area_ha,
            forest_loss_ha=forest_loss_pixels * pixel_area_ha,
            clear_cut_ha=clear_cut_pixels * pixel_area_ha,
            degradation_ha=degradation_pixels * pixel_area_ha,
            regrowth_ha=regrowth_pixels * pixel_area_ha,
            no_change_ha=no_change_pixels * pixel_area_ha,
            pre_date=pre_date,
            post_date=post_date,
            days_between=days_between,
            pixel_size_m=self.pixel_size_m,
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
            metadata={
                "pre_ndvi_mean": pre_ndvi.mean_value,
                "post_ndvi_mean": post_ndvi.mean_value,
                "dndvi_mean": float(np.nanmean(dndvi)),
                "dndvi_std": float(np.nanstd(dndvi)),
                "nbr_available": dnbr is not None,
                "thresholds": {
                    "clear_cut": self.thresholds.ndvi_clear_cut,
                    "degradation": self.thresholds.ndvi_degradation,
                },
            }
        )

        logger.info(
            f"Change detection complete: {result.forest_loss_ha:.2f} ha loss "
            f"({result.forest_loss_fraction * 100:.2f}% of area)"
        )

        return result


class MultiTemporalAnalyzer:
    """
    Multi-temporal change analysis for trend detection.

    Analyzes time series of images to detect long-term trends,
    seasonal patterns, and abrupt changes.
    """

    def __init__(
        self,
        pixel_size_m: float = 10.0,
        min_observations: int = 3,
    ):
        """
        Initialize analyzer.

        Args:
            pixel_size_m: Pixel size in meters
            min_observations: Minimum observations for trend analysis
        """
        self.pixel_size_m = pixel_size_m
        self.min_observations = min_observations

    def calculate_trend(
        self,
        ndvi_series: list[np.ndarray],
        dates: list[datetime],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate linear NDVI trend over time.

        Args:
            ndvi_series: List of NDVI arrays
            dates: List of corresponding dates

        Returns:
            Tuple of (slope, intercept, r_squared) arrays
        """
        if len(ndvi_series) < self.min_observations:
            raise ChangeDetectionError(
                f"Need at least {self.min_observations} observations, got {len(ndvi_series)}"
            )

        # Ensure all arrays have same shape
        ref_shape = ndvi_series[0].shape
        for i, arr in enumerate(ndvi_series):
            if arr.shape != ref_shape:
                raise TemporalMismatchError(
                    f"Array {i} shape {arr.shape} doesn't match reference {ref_shape}"
                )

        # Convert dates to numeric (days since first date)
        t0 = dates[0]
        t_days = np.array([(d - t0).days for d in dates], dtype=np.float64)

        # Stack NDVI arrays
        ndvi_stack = np.stack(ndvi_series, axis=0)  # Shape: (n_times, height, width)
        n_times = len(ndvi_series)

        # Calculate linear regression per pixel
        # y = mx + b
        # Using least squares: m = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)

        t_mean = np.mean(t_days)
        t_centered = t_days - t_mean

        # Mean NDVI per pixel across time
        ndvi_mean = np.nanmean(ndvi_stack, axis=0)

        # Calculate slope
        numerator = np.zeros(ref_shape)
        denominator = np.sum(t_centered ** 2)

        for i in range(n_times):
            ndvi_centered = ndvi_stack[i] - ndvi_mean
            numerator += t_centered[i] * ndvi_centered

        slope = numerator / denominator

        # Calculate intercept
        intercept = ndvi_mean - slope * t_mean

        # Calculate R-squared
        ss_tot = np.zeros(ref_shape)
        ss_res = np.zeros(ref_shape)

        for i in range(n_times):
            predicted = slope * t_days[i] + intercept
            ss_tot += (ndvi_stack[i] - ndvi_mean) ** 2
            ss_res += (ndvi_stack[i] - predicted) ** 2

        # Avoid division by zero
        r_squared = 1 - np.divide(ss_res, ss_tot, where=ss_tot > 0)
        r_squared[ss_tot == 0] = np.nan

        return slope, intercept, r_squared

    def detect_breakpoints(
        self,
        ndvi_series: list[np.ndarray],
        dates: list[datetime],
        threshold: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect abrupt changes (breakpoints) in NDVI time series.

        Uses simple threshold-based detection on consecutive differences.

        Args:
            ndvi_series: List of NDVI arrays
            dates: List of corresponding dates
            threshold: NDVI change threshold for breakpoint

        Returns:
            Tuple of (breakpoint_count, max_change) per pixel
        """
        if len(ndvi_series) < 2:
            raise ChangeDetectionError("Need at least 2 observations")

        ref_shape = ndvi_series[0].shape

        breakpoint_count = np.zeros(ref_shape, dtype=np.int32)
        max_change = np.zeros(ref_shape, dtype=np.float32)

        for i in range(1, len(ndvi_series)):
            change = ndvi_series[i] - ndvi_series[i - 1]

            # Detect negative breakpoints (loss)
            loss_mask = change < -threshold
            breakpoint_count[loss_mask] += 1

            # Track maximum negative change
            neg_change = -change
            max_change = np.maximum(max_change, np.where(loss_mask, neg_change, 0))

        return breakpoint_count, max_change

    def analyze_time_series(
        self,
        images: list[dict[str, np.ndarray]],
        dates: list[datetime],
        scale_factor: float = 10000.0,
    ) -> dict[str, Any]:
        """
        Full multi-temporal analysis.

        Args:
            images: List of band data dicts
            dates: List of acquisition dates
            scale_factor: Scale factor for reflectance

        Returns:
            Dict containing all analysis results
        """
        logger.info(f"Analyzing time series: {len(images)} images from {dates[0]} to {dates[-1]}")

        # Calculate NDVI for all images
        ndvi_series = []
        for bands in images:
            ndvi_result = calculate_ndvi(bands["B4"], bands["B8"], scale_factor)
            ndvi_series.append(ndvi_result.data)

        # Trend analysis
        slope, intercept, r_squared = self.calculate_trend(ndvi_series, dates)

        # Breakpoint detection
        breakpoints, max_change = self.detect_breakpoints(ndvi_series, dates)

        # Overall statistics
        # Declining trend indicates potential deforestation
        declining_mask = (slope < -0.0001) & (r_squared > 0.3)  # Significant negative trend
        pixel_area_ha = (self.pixel_size_m ** 2) / 10000.0
        declining_area_ha = float(np.sum(declining_mask) * pixel_area_ha)

        # Breakpoint area
        breakpoint_area_ha = float(np.sum(breakpoints > 0) * pixel_area_ha)

        results = {
            "trend": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "declining_area_ha": declining_area_ha,
            },
            "breakpoints": {
                "count": breakpoints,
                "max_change": max_change,
                "affected_area_ha": breakpoint_area_ha,
            },
            "summary": {
                "n_images": len(images),
                "date_range_days": (dates[-1] - dates[0]).days,
                "mean_ndvi_first": float(np.nanmean(ndvi_series[0])),
                "mean_ndvi_last": float(np.nanmean(ndvi_series[-1])),
            }
        }

        logger.info(
            f"Multi-temporal analysis complete: {declining_area_ha:.2f} ha showing decline"
        )

        return results


def calculate_forest_loss_area(
    change_result: ChangeDetectionResult,
    min_confidence: float = 0.6,
) -> dict[str, float]:
    """
    Calculate forest loss area with confidence filtering.

    Args:
        change_result: Change detection result
        min_confidence: Minimum confidence threshold

    Returns:
        Dict with area breakdown in hectares
    """
    pixel_area_ha = (change_result.pixel_size_m ** 2) / 10000.0

    high_conf_mask = change_result.confidence >= min_confidence

    areas = {
        "total_loss_ha": 0.0,
        "clear_cut_ha": 0.0,
        "degradation_ha": 0.0,
        "partial_loss_ha": 0.0,
        "high_confidence_loss_ha": 0.0,
    }

    # All loss (any confidence)
    loss_mask = (
        (change_result.change_type == ChangeType.CLEAR_CUT) |
        (change_result.change_type == ChangeType.DEGRADATION) |
        (change_result.change_type == ChangeType.PARTIAL_LOSS)
    )
    areas["total_loss_ha"] = float(np.sum(loss_mask) * pixel_area_ha)

    # By type
    areas["clear_cut_ha"] = float(np.sum(
        change_result.change_type == ChangeType.CLEAR_CUT
    ) * pixel_area_ha)
    areas["degradation_ha"] = float(np.sum(
        change_result.change_type == ChangeType.DEGRADATION
    ) * pixel_area_ha)
    areas["partial_loss_ha"] = float(np.sum(
        change_result.change_type == ChangeType.PARTIAL_LOSS
    ) * pixel_area_ha)

    # High confidence loss
    areas["high_confidence_loss_ha"] = float(np.sum(
        loss_mask & high_conf_mask
    ) * pixel_area_ha)

    return areas


def generate_change_report(
    change_result: ChangeDetectionResult,
) -> dict[str, Any]:
    """
    Generate a summary report for EUDR compliance.

    Args:
        change_result: Change detection result

    Returns:
        Dict containing report data
    """
    report = {
        "analysis_period": {
            "start_date": change_result.pre_date.isoformat(),
            "end_date": change_result.post_date.isoformat(),
            "duration_days": change_result.days_between,
        },
        "area_summary": {
            "total_analyzed_ha": round(change_result.total_area_ha, 2),
            "forest_loss_ha": round(change_result.forest_loss_ha, 2),
            "forest_loss_percent": round(change_result.forest_loss_fraction * 100, 2),
            "annual_loss_rate_percent": round(change_result.annual_loss_rate, 3),
        },
        "change_breakdown": {
            "clear_cut_ha": round(change_result.clear_cut_ha, 2),
            "degradation_ha": round(change_result.degradation_ha, 2),
            "regrowth_ha": round(change_result.regrowth_ha, 2),
            "no_change_ha": round(change_result.no_change_ha, 2),
        },
        "data_quality": {
            "valid_pixels": change_result.valid_pixels,
            "total_pixels": change_result.total_pixels,
            "valid_fraction": round(
                change_result.valid_pixels / change_result.total_pixels * 100, 1
            ),
            "pixel_resolution_m": change_result.pixel_size_m,
        },
        "eudr_compliance": {
            "deforestation_detected": change_result.forest_loss_ha > 0,
            "clear_cut_detected": change_result.clear_cut_ha > 0,
            "requires_review": change_result.forest_loss_ha > 0.5,  # >0.5 ha threshold
        },
        "metadata": change_result.metadata,
    }

    return report
