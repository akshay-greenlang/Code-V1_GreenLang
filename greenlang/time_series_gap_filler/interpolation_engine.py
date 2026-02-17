# -*- coding: utf-8 -*-
"""
Interpolation Engine - AGENT-DATA-014: Time Series Gap Filler

Engine 3 of 7. Pure-Python interpolation methods for filling gaps in
time series data. Supports linear, cubic spline, polynomial, Akima,
nearest-neighbour, and PCHIP (Piecewise Cubic Hermite Interpolating
Polynomial) interpolation strategies.

Zero-Hallucination: All calculations use deterministic Python arithmetic
via the ``math`` module. No external numerical libraries (NumPy, SciPy)
are required. No LLM calls for numeric computations.

Interpolation Methods:
    1. Linear -- straight-line interpolation between known neighbours
    2. Cubic Spline -- natural cubic spline through known data points
    3. Polynomial -- local polynomial fit (configurable degree)
    4. Akima -- piecewise cubic with reduced oscillation
    5. Nearest -- fill with closest non-missing value
    6. PCHIP -- monotonicity-preserving piecewise cubic Hermite

Example:
    >>> from greenlang.time_series_gap_filler.interpolation_engine import (
    ...     InterpolationEngine,
    ... )
    >>> engine = InterpolationEngine()
    >>> result = engine.fill_gaps([1.0, None, 3.0], method="linear")
    >>> assert result.filled_values == [1.0, 2.0, 3.0]

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
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

from greenlang.time_series_gap_filler.config import get_config
from greenlang.time_series_gap_filler.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for optional sibling modules (metrics, models)
# ---------------------------------------------------------------------------

try:
    from greenlang.time_series_gap_filler import metrics as _metrics_mod
    _METRICS_AVAILABLE = True
except ImportError:
    _metrics_mod = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False

try:
    from greenlang.time_series_gap_filler.models import (  # type: ignore[import-untyped]
        FillMethod as _ExtFillMethod,
        FillPoint as _ExtFillPoint,
        FillResult as _ExtFillResult,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _ExtFillMethod = None  # type: ignore[assignment, misc]
    _ExtFillPoint = None  # type: ignore[assignment, misc]
    _ExtFillResult = None  # type: ignore[assignment, misc]
    _MODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Local metric helper stubs (delegate when metrics module is present)
# ---------------------------------------------------------------------------


def _inc_gaps_filled(method: str, count: int = 1) -> None:
    """Increment the gaps-filled counter by method.

    Delegates to ``metrics.inc_gaps_filled(method, count)``.

    Args:
        method: Interpolation method name.
        count: Number of gaps filled.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.inc_gaps_filled(method, count)


def _observe_confidence(method: str, confidence: float) -> None:
    """Observe a fill confidence score.

    The underlying ``metrics.observe_confidence`` accepts only the
    confidence value (no method label), so the method argument is
    logged but not forwarded.

    Args:
        method: Interpolation method name (logged, not forwarded).
        confidence: Confidence value (0.0-1.0).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.observe_confidence(confidence)


def _observe_duration(operation: str, duration: float) -> None:
    """Observe processing duration in seconds.

    Delegates to ``metrics.observe_duration(operation, duration)``.

    Args:
        operation: Operation label.
        duration: Duration in seconds.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        _metrics_mod.observe_duration(operation, duration)


# ---------------------------------------------------------------------------
# Local enumerations and data models (used when models.py is absent)
# ---------------------------------------------------------------------------


class FillStrategy(str, Enum):
    """Strategy for filling time series gaps.

    LINEAR: Standard linear interpolation between neighbours.
    CUBIC_SPLINE: Natural cubic spline through known points.
    POLYNOMIAL: Local polynomial fit of configurable degree.
    AKIMA: Akima piecewise cubic (reduced oscillation).
    NEAREST: Fill with nearest non-missing value.
    PCHIP: Piecewise Cubic Hermite Interpolating Polynomial.
    """

    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    POLYNOMIAL = "polynomial"
    AKIMA = "akima"
    NEAREST = "nearest"
    PCHIP = "pchip"


@dataclass
class FilledPoint:
    """A single filled data point with metadata.

    Attributes:
        index: Position in the original series.
        original_value: Original value (None if missing).
        filled_value: Value after gap filling.
        was_missing: Whether this point was originally missing.
        confidence: Confidence in the filled value (0.0-1.0).
        method: Interpolation method used for this point.
        gap_length: Length of the gap this point belonged to.
        provenance_hash: SHA-256 hash for audit trail.
    """

    index: int
    original_value: Optional[float]
    filled_value: float
    was_missing: bool
    confidence: float
    method: str
    gap_length: int = 0
    provenance_hash: str = ""


@dataclass
class FillResult:
    """Complete result of a gap-filling operation.

    Attributes:
        result_id: Unique identifier for this fill result.
        filled_values: Complete series with gaps filled.
        filled_points: Per-point fill metadata.
        method: Interpolation method used.
        gaps_found: Number of gaps detected.
        gaps_filled: Number of gaps successfully filled.
        total_missing: Total number of missing values filled.
        mean_confidence: Average confidence across filled points.
        min_confidence: Minimum confidence across filled points.
        processing_time_ms: Wall-clock processing time in milliseconds.
        provenance_hash: SHA-256 hash of the entire operation.
        created_at: Timestamp of result creation.
        metadata: Optional additional metadata.
    """

    result_id: str = ""
    filled_values: List[float] = field(default_factory=list)
    filled_points: List[FilledPoint] = field(default_factory=list)
    method: str = "linear"
    gaps_found: int = 0
    gaps_filled: int = 0
    total_missing: int = 0
    mean_confidence: float = 0.0
    min_confidence: float = 1.0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum polynomial degree to prevent overfitting.
MAX_POLYNOMIAL_DEGREE: int = 10

#: Default local window size for polynomial fitting.
DEFAULT_POLY_WINDOW: int = 5

#: Small epsilon to avoid division by zero.
_EPS: float = 1e-15

#: Confidence base values per method (higher = inherently more confident).
_METHOD_CONFIDENCE_BASE: Dict[str, float] = {
    "linear": 0.80,
    "cubic_spline": 0.85,
    "polynomial": 0.75,
    "akima": 0.88,
    "nearest": 0.55,
    "pchip": 0.87,
}


# ============================================================================
# InterpolationEngine
# ============================================================================


class InterpolationEngine:
    """Pure-Python interpolation engine for time series gap filling.

    Implements six interpolation strategies (linear, cubic spline,
    polynomial, Akima, nearest, PCHIP) plus a dispatcher method
    ``fill_gaps`` that selects the appropriate strategy. All methods
    operate on ``List[Optional[float]]`` where ``None`` represents
    missing values.

    Attributes:
        _config: Time series gap filler configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = InterpolationEngine()
        >>> result = engine.interpolate_linear([1.0, None, None, 4.0])
        >>> assert abs(result.filled_values[1] - 2.0) < 1e-9
        >>> assert abs(result.filled_values[2] - 3.0) < 1e-9
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize InterpolationEngine.

        Args:
            config: Optional TimeSeriesGapFillerConfig override.
                Falls back to the singleton from ``get_config()``.
        """
        self._config = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()
        logger.info("InterpolationEngine initialized")

    # ------------------------------------------------------------------
    # Public API: fill_gaps dispatcher
    # ------------------------------------------------------------------

    def fill_gaps(
        self,
        values: List[Optional[float]],
        method: str = "linear",
        timestamps: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> FillResult:
        """Fill gaps using the specified interpolation method.

        Dispatches to the appropriate interpolation method based on the
        ``method`` parameter, which should match a ``FillStrategy`` value.

        Args:
            values: Series with ``None`` for missing values.
            method: One of ``linear``, ``cubic_spline``, ``polynomial``,
                ``akima``, ``nearest``, ``pchip``.
            timestamps: Optional monotonic timestamps for non-uniform
                spacing. When provided, interpolation accounts for
                unequal intervals.
            **kwargs: Additional keyword arguments forwarded to the
                underlying interpolation method (e.g. ``degree`` for
                polynomial).

        Returns:
            FillResult with filled values and metadata.

        Raises:
            ValueError: If the method name is not recognised.
        """
        method_lower = method.lower().strip()

        dispatcher: Dict[str, Any] = {
            "linear": self.interpolate_linear,
            "cubic_spline": self.interpolate_cubic_spline,
            "polynomial": self.interpolate_polynomial,
            "akima": self.interpolate_akima,
            "nearest": self.interpolate_nearest,
            "pchip": self.interpolate_pchip,
        }

        fn = dispatcher.get(method_lower)
        if fn is None:
            raise ValueError(
                f"Unrecognised interpolation method: {method!r}. "
                f"Supported: {sorted(dispatcher.keys())}"
            )

        logger.info(
            "fill_gaps dispatching to method=%s, series_length=%d",
            method_lower, len(values),
        )
        return fn(values, timestamps=timestamps, **kwargs)

    # ------------------------------------------------------------------
    # 1. Linear Interpolation
    # ------------------------------------------------------------------

    def interpolate_linear(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> FillResult:
        """Fill gaps using linear interpolation.

        For each gap, draws a straight line between the nearest non-missing
        neighbours on either side. Edge gaps (at the start or end of the
        series) are extrapolated from the nearest two known points when
        possible.

        Confidence is high for short gaps and decays for longer gaps.

        Args:
            values: Series with ``None`` for missing values.
            timestamps: Optional monotonic timestamps for non-uniform
                spacing.

        Returns:
            FillResult with linearly interpolated values.
        """
        start_t = time.time()
        n = len(values)
        method_name = "linear"
        filled = list(values)
        filled_points: List[FilledPoint] = []
        gaps_found = 0
        total_missing = 0

        # Identify gap segments
        gap_segments = self._find_gap_segments(values)
        gaps_found = len(gap_segments)

        for seg_start, seg_end in gap_segments:
            gap_len = seg_end - seg_start + 1
            total_missing += gap_len

            left_idx, left_val, right_idx, right_val = self._find_neighbors(
                values, seg_start,
            )

            for pos in range(seg_start, seg_end + 1):
                fill_val: float
                if left_val is not None and right_val is not None:
                    # Interior gap: linear interpolation
                    fill_val = self._lerp(
                        left_idx, left_val,
                        right_idx, right_val,
                        pos, timestamps,
                    )
                elif left_val is not None:
                    # Right edge gap: extrapolate from two leftmost knowns
                    fill_val = self._extrapolate_edge(
                        values, pos, direction="right",
                        timestamps=timestamps,
                    )
                elif right_val is not None:
                    # Left edge gap: extrapolate from two rightmost knowns
                    fill_val = self._extrapolate_edge(
                        values, pos, direction="left",
                        timestamps=timestamps,
                    )
                else:
                    fill_val = 0.0

                conf = self._compute_confidence(gap_len, method_name, n)
                prov = self._build_point_provenance(
                    method_name, pos, fill_val, gap_len,
                )

                filled[pos] = fill_val
                filled_points.append(FilledPoint(
                    index=pos,
                    original_value=None,
                    filled_value=fill_val,
                    was_missing=True,
                    confidence=conf,
                    method=method_name,
                    gap_length=gap_len,
                    provenance_hash=prov,
                ))

        result = self._build_result(
            filled, filled_points, method_name,
            gaps_found, total_missing, start_t,
        )
        return result

    # ------------------------------------------------------------------
    # 2. Cubic Spline Interpolation
    # ------------------------------------------------------------------

    def interpolate_cubic_spline(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> FillResult:
        """Fill gaps using natural cubic spline interpolation.

        Constructs a natural cubic spline (second derivative is zero at
        endpoints) through all known data points and evaluates it at gap
        positions. Solves a tridiagonal linear system for the spline
        coefficients.

        Higher confidence than linear for smooth, continuous data.

        Args:
            values: Series with ``None`` for missing values.
            timestamps: Optional monotonic timestamps for non-uniform
                spacing.

        Returns:
            FillResult with cubic-spline interpolated values.
        """
        start_t = time.time()
        n = len(values)
        method_name = "cubic_spline"
        filled = list(values)
        filled_points: List[FilledPoint] = []
        total_missing = 0

        # Collect known points
        known_x, known_y = self._collect_known(values, timestamps)

        if len(known_x) < 2:
            return self._fallback_fill(
                values, method_name, start_t,
                reason="fewer than 2 known points",
            )

        # Compute cubic spline coefficients
        coeffs = self._cubic_spline_coefficients(known_x, known_y)

        gap_segments = self._find_gap_segments(values)
        gaps_found = len(gap_segments)

        for seg_start, seg_end in gap_segments:
            gap_len = seg_end - seg_start + 1
            total_missing += gap_len

            for pos in range(seg_start, seg_end + 1):
                x = float(pos) if timestamps is None else timestamps[pos]
                fill_val = self._evaluate_cubic_spline(
                    known_x, known_y, coeffs, x,
                )
                conf = self._compute_confidence(gap_len, method_name, n)
                prov = self._build_point_provenance(
                    method_name, pos, fill_val, gap_len,
                )

                filled[pos] = fill_val
                filled_points.append(FilledPoint(
                    index=pos,
                    original_value=None,
                    filled_value=fill_val,
                    was_missing=True,
                    confidence=conf,
                    method=method_name,
                    gap_length=gap_len,
                    provenance_hash=prov,
                ))

        result = self._build_result(
            filled, filled_points, method_name,
            gaps_found, total_missing, start_t,
        )
        return result

    # ------------------------------------------------------------------
    # 3. Polynomial Interpolation
    # ------------------------------------------------------------------

    def interpolate_polynomial(
        self,
        values: List[Optional[float]],
        degree: int = 2,
        timestamps: Optional[List[float]] = None,
    ) -> FillResult:
        """Fill gaps using local polynomial interpolation.

        Fits a polynomial of the given degree to a local window of
        known points surrounding each gap, then evaluates the polynomial
        at gap positions.

        Args:
            values: Series with ``None`` for missing values.
            degree: Polynomial degree (capped at MAX_POLYNOMIAL_DEGREE).
            timestamps: Optional monotonic timestamps.

        Returns:
            FillResult with polynomial-interpolated values.
        """
        start_t = time.time()
        n = len(values)
        method_name = "polynomial"
        degree = min(degree, MAX_POLYNOMIAL_DEGREE)
        filled = list(values)
        filled_points: List[FilledPoint] = []
        total_missing = 0

        gap_segments = self._find_gap_segments(values)
        gaps_found = len(gap_segments)

        for seg_start, seg_end in gap_segments:
            gap_len = seg_end - seg_start + 1
            total_missing += gap_len

            # Gather local window of known points
            local_x, local_y = self._gather_local_known(
                values, seg_start, seg_end,
                window=max(DEFAULT_POLY_WINDOW, degree + 1),
                timestamps=timestamps,
            )

            if len(local_x) < degree + 1:
                # Not enough points for this degree; fall back to linear
                for pos in range(seg_start, seg_end + 1):
                    left_idx, left_val, right_idx, right_val = (
                        self._find_neighbors(values, pos)
                    )
                    fill_val = self._lerp_safe(
                        left_idx, left_val, right_idx, right_val,
                        pos, timestamps,
                    )
                    conf = self._compute_confidence(gap_len, "linear", n)
                    prov = self._build_point_provenance(
                        method_name, pos, fill_val, gap_len,
                    )
                    filled[pos] = fill_val
                    filled_points.append(FilledPoint(
                        index=pos,
                        original_value=None,
                        filled_value=fill_val,
                        was_missing=True,
                        confidence=conf,
                        method=method_name,
                        gap_length=gap_len,
                        provenance_hash=prov,
                    ))
                continue

            # Fit polynomial via Vandermonde + Gaussian elimination
            poly_coeffs = self._fit_polynomial(local_x, local_y, degree)

            for pos in range(seg_start, seg_end + 1):
                x = float(pos) if timestamps is None else timestamps[pos]
                fill_val = self._evaluate_polynomial(poly_coeffs, x)
                conf = self._compute_confidence(gap_len, method_name, n)
                prov = self._build_point_provenance(
                    method_name, pos, fill_val, gap_len,
                )
                filled[pos] = fill_val
                filled_points.append(FilledPoint(
                    index=pos,
                    original_value=None,
                    filled_value=fill_val,
                    was_missing=True,
                    confidence=conf,
                    method=method_name,
                    gap_length=gap_len,
                    provenance_hash=prov,
                ))

        result = self._build_result(
            filled, filled_points, method_name,
            gaps_found, total_missing, start_t,
        )
        return result

    # ------------------------------------------------------------------
    # 4. Akima Interpolation
    # ------------------------------------------------------------------

    def interpolate_akima(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> FillResult:
        """Fill gaps using Akima piecewise cubic interpolation.

        The Akima method computes slopes at each knot using a weighted
        average of neighbouring divided differences. This avoids the
        overshooting and oscillation typical of natural cubic splines,
        especially near outliers.

        Requires at least 5 known points for proper slope weighting.
        Falls back to linear interpolation with fewer points.

        Args:
            values: Series with ``None`` for missing values.
            timestamps: Optional monotonic timestamps.

        Returns:
            FillResult with Akima-interpolated values.
        """
        start_t = time.time()
        n = len(values)
        method_name = "akima"
        filled = list(values)
        filled_points: List[FilledPoint] = []
        total_missing = 0

        known_x, known_y = self._collect_known(values, timestamps)

        if len(known_x) < 2:
            return self._fallback_fill(
                values, method_name, start_t,
                reason="fewer than 2 known points",
            )

        # Compute Akima slopes
        slopes = self._akima_slopes(known_x, known_y)

        gap_segments = self._find_gap_segments(values)
        gaps_found = len(gap_segments)

        for seg_start, seg_end in gap_segments:
            gap_len = seg_end - seg_start + 1
            total_missing += gap_len

            for pos in range(seg_start, seg_end + 1):
                x = float(pos) if timestamps is None else timestamps[pos]
                fill_val = self._evaluate_akima(
                    known_x, known_y, slopes, x,
                )
                conf = self._compute_confidence(gap_len, method_name, n)
                prov = self._build_point_provenance(
                    method_name, pos, fill_val, gap_len,
                )
                filled[pos] = fill_val
                filled_points.append(FilledPoint(
                    index=pos,
                    original_value=None,
                    filled_value=fill_val,
                    was_missing=True,
                    confidence=conf,
                    method=method_name,
                    gap_length=gap_len,
                    provenance_hash=prov,
                ))

        result = self._build_result(
            filled, filled_points, method_name,
            gaps_found, total_missing, start_t,
        )
        return result

    # ------------------------------------------------------------------
    # 5. Nearest Interpolation
    # ------------------------------------------------------------------

    def interpolate_nearest(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> FillResult:
        """Fill gaps with the nearest non-missing value.

        For each missing position, finds the closest known value to the
        left and right. If both are equidistant, the left value is used.
        This is the simplest method and carries the lowest confidence.

        Args:
            values: Series with ``None`` for missing values.
            timestamps: Optional monotonic timestamps (used for distance
                calculation when spacing is non-uniform).

        Returns:
            FillResult with nearest-value filled series.
        """
        start_t = time.time()
        n = len(values)
        method_name = "nearest"
        filled = list(values)
        filled_points: List[FilledPoint] = []
        total_missing = 0

        gap_segments = self._find_gap_segments(values)
        gaps_found = len(gap_segments)

        for seg_start, seg_end in gap_segments:
            gap_len = seg_end - seg_start + 1
            total_missing += gap_len

            for pos in range(seg_start, seg_end + 1):
                left_idx, left_val, right_idx, right_val = (
                    self._find_neighbors(values, pos)
                )

                fill_val = self._pick_nearest(
                    pos, left_idx, left_val,
                    right_idx, right_val, timestamps,
                )
                conf = self._compute_confidence(gap_len, method_name, n)
                prov = self._build_point_provenance(
                    method_name, pos, fill_val, gap_len,
                )

                filled[pos] = fill_val
                filled_points.append(FilledPoint(
                    index=pos,
                    original_value=None,
                    filled_value=fill_val,
                    was_missing=True,
                    confidence=conf,
                    method=method_name,
                    gap_length=gap_len,
                    provenance_hash=prov,
                ))

        result = self._build_result(
            filled, filled_points, method_name,
            gaps_found, total_missing, start_t,
        )
        return result

    # ------------------------------------------------------------------
    # 6. PCHIP Interpolation
    # ------------------------------------------------------------------

    def interpolate_pchip(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> FillResult:
        """Fill gaps using PCHIP interpolation.

        Piecewise Cubic Hermite Interpolating Polynomial preserves
        monotonicity of the data. Where data is monotonically increasing
        or decreasing, the interpolant will not overshoot. This makes
        PCHIP ideal for data that should respect natural bounds.

        Requires at least 2 known points. Falls back to nearest with
        fewer.

        Args:
            values: Series with ``None`` for missing values.
            timestamps: Optional monotonic timestamps.

        Returns:
            FillResult with PCHIP-interpolated values.
        """
        start_t = time.time()
        n = len(values)
        method_name = "pchip"
        filled = list(values)
        filled_points: List[FilledPoint] = []
        total_missing = 0

        known_x, known_y = self._collect_known(values, timestamps)

        if len(known_x) < 2:
            return self._fallback_fill(
                values, method_name, start_t,
                reason="fewer than 2 known points",
            )

        # Compute PCHIP slopes (Fritsch-Carlson)
        slopes = self._pchip_slopes(known_x, known_y)

        gap_segments = self._find_gap_segments(values)
        gaps_found = len(gap_segments)

        for seg_start, seg_end in gap_segments:
            gap_len = seg_end - seg_start + 1
            total_missing += gap_len

            for pos in range(seg_start, seg_end + 1):
                x = float(pos) if timestamps is None else timestamps[pos]
                fill_val = self._evaluate_hermite(
                    known_x, known_y, slopes, x,
                )
                conf = self._compute_confidence(gap_len, method_name, n)
                prov = self._build_point_provenance(
                    method_name, pos, fill_val, gap_len,
                )
                filled[pos] = fill_val
                filled_points.append(FilledPoint(
                    index=pos,
                    original_value=None,
                    filled_value=fill_val,
                    was_missing=True,
                    confidence=conf,
                    method=method_name,
                    gap_length=gap_len,
                    provenance_hash=prov,
                ))

        result = self._build_result(
            filled, filled_points, method_name,
            gaps_found, total_missing, start_t,
        )
        return result

    # ==================================================================
    # Private helpers -- gap detection
    # ==================================================================

    @staticmethod
    def _is_missing(value: Optional[float]) -> bool:
        """Check whether a value is missing (None or NaN).

        Args:
            value: Value to check.

        Returns:
            True if the value is None or ``float('nan')``.
        """
        if value is None:
            return True
        try:
            return math.isnan(value)
        except (TypeError, ValueError):
            return False

    def _find_gap_segments(
        self,
        values: List[Optional[float]],
    ) -> List[Tuple[int, int]]:
        """Identify contiguous gap segments in the series.

        Args:
            values: Series with possible None/NaN gaps.

        Returns:
            List of (start_index, end_index) tuples for each gap.
        """
        segments: List[Tuple[int, int]] = []
        n = len(values)
        i = 0
        while i < n:
            if self._is_missing(values[i]):
                seg_start = i
                while i < n and self._is_missing(values[i]):
                    i += 1
                segments.append((seg_start, i - 1))
            else:
                i += 1
        return segments

    def _find_neighbors(
        self,
        values: List[Optional[float]],
        index: int,
    ) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[float]]:
        """Find the nearest non-missing left and right neighbours.

        Args:
            values: Series with possible missing values.
            index: Position to search from.

        Returns:
            Tuple of (left_index, left_value, right_index, right_value).
            Either pair may be ``(None, None)`` if no neighbour exists
            in that direction.
        """
        left_idx: Optional[int] = None
        left_val: Optional[float] = None
        right_idx: Optional[int] = None
        right_val: Optional[float] = None

        # Search left
        for j in range(index - 1, -1, -1):
            if not self._is_missing(values[j]):
                left_idx = j
                left_val = values[j]
                break

        # Search right
        for j in range(index + 1, len(values)):
            if not self._is_missing(values[j]):
                right_idx = j
                right_val = values[j]
                break

        return left_idx, left_val, right_idx, right_val

    # ==================================================================
    # Private helpers -- linear interpolation primitives
    # ==================================================================

    @staticmethod
    def _lerp(
        x0: int, y0: float,
        x1: int, y1: float,
        x: int,
        timestamps: Optional[List[float]] = None,
    ) -> float:
        """Linear interpolation between two points.

        When timestamps are provided, uses timestamp values for the
        x-coordinates instead of integer indices.

        Args:
            x0: Left index.
            y0: Left value.
            x1: Right index.
            y1: Right value.
            x: Query index.
            timestamps: Optional timestamps.

        Returns:
            Interpolated value at position x.
        """
        if timestamps is not None:
            tx0 = timestamps[x0]
            tx1 = timestamps[x1]
            tx = timestamps[x]
            denom = tx1 - tx0
            if abs(denom) < _EPS:
                return (y0 + y1) / 2.0
            t = (tx - tx0) / denom
        else:
            denom = x1 - x0
            if denom == 0:
                return (y0 + y1) / 2.0
            t = (x - x0) / denom

        return y0 + t * (y1 - y0)

    def _lerp_safe(
        self,
        left_idx: Optional[int],
        left_val: Optional[float],
        right_idx: Optional[int],
        right_val: Optional[float],
        pos: int,
        timestamps: Optional[List[float]] = None,
    ) -> float:
        """Linear interpolation with None-safety.

        Falls back to the available neighbour or 0.0 if both are None.

        Args:
            left_idx: Left neighbour index.
            left_val: Left neighbour value.
            right_idx: Right neighbour index.
            right_val: Right neighbour value.
            pos: Query position.
            timestamps: Optional timestamps.

        Returns:
            Interpolated (or extrapolated) value.
        """
        if left_val is not None and right_val is not None:
            return self._lerp(
                left_idx, left_val,  # type: ignore[arg-type]
                right_idx, right_val,  # type: ignore[arg-type]
                pos, timestamps,
            )
        if left_val is not None:
            return left_val
        if right_val is not None:
            return right_val
        return 0.0

    def _extrapolate_edge(
        self,
        values: List[Optional[float]],
        pos: int,
        direction: str,
        timestamps: Optional[List[float]] = None,
    ) -> float:
        """Extrapolate from two nearest known points at a series edge.

        When direction is "right", we have known points only to the left
        of the gap. When direction is "left", known points are only to
        the right.

        Args:
            values: Full series.
            pos: Gap position to fill.
            direction: "left" (gap at start) or "right" (gap at end).
            timestamps: Optional timestamps.

        Returns:
            Extrapolated value. Falls back to the single nearest known
            value if only one known point exists.
        """
        known_x: List[float] = []
        known_y: List[float] = []

        if direction == "right":
            # Gap at end: known points are to the left
            for j in range(pos - 1, -1, -1):
                if not self._is_missing(values[j]):
                    tx = float(j) if timestamps is None else timestamps[j]
                    known_x.insert(0, tx)
                    known_y.insert(0, values[j])  # type: ignore[arg-type]
                    if len(known_x) == 2:
                        break
        else:
            # Gap at start: known points are to the right
            for j in range(pos + 1, len(values)):
                if not self._is_missing(values[j]):
                    tx = float(j) if timestamps is None else timestamps[j]
                    known_x.append(tx)
                    known_y.append(values[j])  # type: ignore[arg-type]
                    if len(known_x) == 2:
                        break

        if len(known_x) == 0:
            return 0.0
        if len(known_x) == 1:
            return known_y[0]

        # Linear extrapolation from two points
        tx = float(pos) if timestamps is None else timestamps[pos]
        dx = known_x[1] - known_x[0]
        if abs(dx) < _EPS:
            return (known_y[0] + known_y[1]) / 2.0
        slope = (known_y[1] - known_y[0]) / dx
        return known_y[0] + slope * (tx - known_x[0])

    # ==================================================================
    # Private helpers -- cubic spline
    # ==================================================================

    def _cubic_spline_coefficients(
        self,
        x: List[float],
        y: List[float],
    ) -> List[Tuple[float, float, float, float]]:
        """Compute natural cubic spline coefficients.

        Solves the tridiagonal system for the second derivatives using
        Thomas's algorithm, then derives the (a, b, c, d) coefficients
        for each spline segment.

        For segment i, the spline is:
            S_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3

        Args:
            x: Known x-coordinates (sorted, strictly increasing).
            y: Known y-values.

        Returns:
            List of (a, b, c, d) coefficient tuples, one per segment.
        """
        n = len(x)
        if n < 2:
            return []
        if n == 2:
            # Single linear segment
            dx = x[1] - x[0]
            slope = (y[1] - y[0]) / dx if abs(dx) > _EPS else 0.0
            return [(y[0], slope, 0.0, 0.0)]

        # Step 1: compute h intervals and alpha
        h = [x[i + 1] - x[i] for i in range(n - 1)]
        alpha = [0.0] * n
        for i in range(1, n - 1):
            if abs(h[i - 1]) < _EPS or abs(h[i]) < _EPS:
                alpha[i] = 0.0
            else:
                alpha[i] = (
                    3.0 / h[i] * (y[i + 1] - y[i])
                    - 3.0 / h[i - 1] * (y[i] - y[i - 1])
                )

        # Step 2: solve tridiagonal system for c (second derivatives / 2)
        # Natural spline: c[0] = c[n-1] = 0
        diag_a = [0.0] * n  # sub-diagonal
        diag_b = [1.0] * n  # main diagonal
        diag_c = [0.0] * n  # super-diagonal
        diag_d = [0.0] * n  # right-hand side

        for i in range(1, n - 1):
            diag_a[i] = h[i - 1]
            diag_b[i] = 2.0 * (h[i - 1] + h[i])
            diag_c[i] = h[i]
            diag_d[i] = alpha[i]

        c = self._solve_tridiagonal(diag_a, diag_b, diag_c, diag_d)

        # Step 3: compute b and d from c
        coefficients: List[Tuple[float, float, float, float]] = []
        for i in range(n - 1):
            a_i = y[i]
            c_i = c[i]
            hi = h[i] if abs(h[i]) > _EPS else _EPS
            d_i = (c[i + 1] - c[i]) / (3.0 * hi)
            b_i = (y[i + 1] - y[i]) / hi - hi * (2.0 * c[i] + c[i + 1]) / 3.0
            coefficients.append((a_i, b_i, c_i, d_i))

        return coefficients

    @staticmethod
    def _solve_tridiagonal(
        a: List[float],
        b: List[float],
        c: List[float],
        d: List[float],
    ) -> List[float]:
        """Solve a tridiagonal linear system using Thomas's algorithm.

        Solves the system::

            b[0]*x[0] + c[0]*x[1]                               = d[0]
            a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]             = d[i]
            a[n-1]*x[n-2] + b[n-1]*x[n-1]                       = d[n-1]

        Args:
            a: Sub-diagonal coefficients (a[0] is unused).
            b: Main diagonal coefficients.
            c: Super-diagonal coefficients (c[n-1] is unused).
            d: Right-hand side values.

        Returns:
            Solution vector x.
        """
        n = len(b)
        if n == 0:
            return []
        if n == 1:
            return [d[0] / b[0] if abs(b[0]) > _EPS else 0.0]

        # Forward sweep (work on copies)
        cp = [0.0] * n
        dp = [0.0] * n

        cp[0] = c[0] / b[0] if abs(b[0]) > _EPS else 0.0
        dp[0] = d[0] / b[0] if abs(b[0]) > _EPS else 0.0

        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < _EPS:
                denom = _EPS
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        # Back substitution
        x = [0.0] * n
        x[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]

        return x

    @staticmethod
    def _evaluate_cubic_spline(
        known_x: List[float],
        known_y: List[float],
        coeffs: List[Tuple[float, float, float, float]],
        x: float,
    ) -> float:
        """Evaluate the cubic spline at a given x position.

        Finds the correct spline segment and computes
        S_i(t) = a + b*(t-x_i) + c*(t-x_i)^2 + d*(t-x_i)^3.

        Clamps x to the known data range for edge queries.

        Args:
            known_x: Sorted knot x-coordinates.
            known_y: Knot y-values.
            coeffs: Spline coefficients from ``_cubic_spline_coefficients``.
            x: Query position.

        Returns:
            Interpolated y value at x.
        """
        n = len(known_x)
        if n == 0 or not coeffs:
            return 0.0

        # Clamp to range
        if x <= known_x[0]:
            seg = 0
        elif x >= known_x[-1]:
            seg = len(coeffs) - 1
        else:
            # Binary search for segment
            seg = 0
            for i in range(n - 1):
                if known_x[i] <= x <= known_x[i + 1]:
                    seg = i
                    break

        a, b, c, d = coeffs[seg]
        dx = x - known_x[seg]
        return a + b * dx + c * dx * dx + d * dx * dx * dx

    # ==================================================================
    # Private helpers -- Akima
    # ==================================================================

    @staticmethod
    def _akima_slopes(
        x: List[float],
        y: List[float],
    ) -> List[float]:
        """Compute Akima slopes at each knot.

        The Akima method assigns slopes using a weighted average of
        neighbouring divided differences. The weights are proportional
        to the absolute difference between consecutive slopes, which
        reduces oscillation near outliers.

        For endpoints and when fewer than 5 points are available, a
        simpler approximation is used.

        Args:
            x: Sorted knot x-coordinates.
            y: Knot y-values.

        Returns:
            List of slopes, one per knot.
        """
        n = len(x)
        if n < 2:
            return [0.0] * n

        # Divided differences
        m: List[float] = []
        for i in range(n - 1):
            dx = x[i + 1] - x[i]
            if abs(dx) < _EPS:
                m.append(0.0)
            else:
                m.append((y[i + 1] - y[i]) / dx)

        if n == 2:
            return [m[0], m[0]]
        if n == 3:
            avg = (m[0] + m[1]) / 2.0
            return [m[0], avg, m[1]]

        # Extend m with two extra values at each end (Akima extension)
        m_ext = [
            2.0 * m[0] - m[1],
            2.0 * m[0] - (2.0 * m[0] - m[1]),
        ]
        # Simplified: use reflection
        m_ext = [2.0 * m[0] - m[1], m[0]] + m + [
            m[-1], 2.0 * m[-1] - m[-2],
        ]

        slopes: List[float] = []
        for i in range(n):
            idx = i + 2  # offset due to padding
            w1 = abs(m_ext[idx + 1] - m_ext[idx])
            w2 = abs(m_ext[idx - 1] - m_ext[idx - 2])
            total_w = w1 + w2

            if total_w < _EPS:
                # When weights are zero, use simple average
                slope = (m_ext[idx - 1] + m_ext[idx]) / 2.0
            else:
                slope = (w1 * m_ext[idx - 1] + w2 * m_ext[idx]) / total_w
            slopes.append(slope)

        return slopes

    def _evaluate_akima(
        self,
        known_x: List[float],
        known_y: List[float],
        slopes: List[float],
        x: float,
    ) -> float:
        """Evaluate Akima interpolant at a given x position.

        Uses Hermite basis functions with Akima slopes for the cubic
        piece on the appropriate segment.

        Args:
            known_x: Sorted knot x-coordinates.
            known_y: Knot y-values.
            slopes: Akima slopes at each knot.
            x: Query position.

        Returns:
            Interpolated y value at x.
        """
        return self._evaluate_hermite(known_x, known_y, slopes, x)

    # ==================================================================
    # Private helpers -- PCHIP (Fritsch-Carlson)
    # ==================================================================

    @staticmethod
    def _pchip_slopes(
        x: List[float],
        y: List[float],
    ) -> List[float]:
        """Compute PCHIP slopes using the Fritsch-Carlson method.

        Ensures monotonicity preservation: if the data is locally
        monotone, the interpolant will be too. Where the data changes
        direction, the slope is set to zero to prevent overshoot.

        Args:
            x: Sorted knot x-coordinates.
            y: Knot y-values.

        Returns:
            List of slopes, one per knot.
        """
        n = len(x)
        if n < 2:
            return [0.0] * n

        # Compute secant slopes
        delta: List[float] = []
        for i in range(n - 1):
            dx = x[i + 1] - x[i]
            if abs(dx) < _EPS:
                delta.append(0.0)
            else:
                delta.append((y[i + 1] - y[i]) / dx)

        if n == 2:
            return [delta[0], delta[0]]

        # Initial slopes from three-point formula
        slopes: List[float] = [0.0] * n

        # Interior points
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] <= 0:
                # Change of sign: set slope to zero (monotonicity)
                slopes[i] = 0.0
            else:
                # Harmonic mean weighted by interval lengths
                w1 = 2.0 * (x[i + 1] - x[i]) + (x[i] - x[i - 1])
                w2 = (x[i + 1] - x[i]) + 2.0 * (x[i] - x[i - 1])
                if abs(w1) < _EPS or abs(w2) < _EPS:
                    slopes[i] = (delta[i - 1] + delta[i]) / 2.0
                else:
                    slopes[i] = (w1 + w2) / (
                        w1 / delta[i - 1] + w2 / delta[i]
                    )

        # Endpoint slopes (one-sided)
        slopes[0] = InterpolationEngine._pchip_endpoint(
            x[0], x[1], x[2] if n > 2 else x[1],
            delta[0], delta[1] if n > 2 else delta[0],
        )
        slopes[-1] = InterpolationEngine._pchip_endpoint(
            x[-1], x[-2], x[-3] if n > 2 else x[-2],
            delta[-1], delta[-2] if n > 2 else delta[-1],
        )

        # Fritsch-Carlson monotonicity fix
        for i in range(n - 1):
            if abs(delta[i]) < _EPS:
                slopes[i] = 0.0
                slopes[i + 1] = 0.0
            else:
                alpha = slopes[i] / delta[i]
                beta = slopes[i + 1] / delta[i]
                # Ensure we stay in the monotone region
                mag = math.sqrt(alpha * alpha + beta * beta)
                if mag > 3.0:
                    tau = 3.0 / mag
                    slopes[i] = tau * alpha * delta[i]
                    slopes[i + 1] = tau * beta * delta[i]

        return slopes

    @staticmethod
    def _pchip_endpoint(
        x0: float, x1: float, x2: float,
        d0: float, d1: float,
    ) -> float:
        """Compute a PCHIP endpoint slope (one-sided).

        Uses a non-centred three-point formula, clamped to preserve
        monotonicity.

        Args:
            x0: Endpoint x.
            x1: Next x.
            x2: Third x.
            d0: First secant slope.
            d1: Second secant slope.

        Returns:
            Endpoint slope value.
        """
        h0 = x1 - x0
        h1 = x2 - x1
        if abs(h0) < _EPS:
            return d0

        # Non-centred formula
        slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1 + _EPS)

        # Clamp to preserve monotonicity
        if slope * d0 <= 0:
            slope = 0.0
        elif abs(slope) > 3.0 * abs(d0):
            slope = 3.0 * d0

        return slope

    # ==================================================================
    # Private helpers -- Hermite evaluation (shared by Akima and PCHIP)
    # ==================================================================

    @staticmethod
    def _evaluate_hermite(
        known_x: List[float],
        known_y: List[float],
        slopes: List[float],
        x: float,
    ) -> float:
        """Evaluate a Hermite cubic interpolant at x.

        Uses the standard Hermite basis functions on the segment
        containing x.

        Args:
            known_x: Sorted knot x-coordinates.
            known_y: Knot y-values.
            slopes: Slopes at each knot (from Akima or PCHIP).
            x: Query position.

        Returns:
            Interpolated y value at x.
        """
        n = len(known_x)
        if n == 0:
            return 0.0
        if n == 1:
            return known_y[0]

        # Clamp to range
        if x <= known_x[0]:
            seg = 0
        elif x >= known_x[-1]:
            seg = n - 2
        else:
            seg = 0
            for i in range(n - 1):
                if known_x[i] <= x <= known_x[i + 1]:
                    seg = i
                    break

        x0 = known_x[seg]
        x1 = known_x[seg + 1]
        y0 = known_y[seg]
        y1 = known_y[seg + 1]
        m0 = slopes[seg]
        m1 = slopes[seg + 1]

        h = x1 - x0
        if abs(h) < _EPS:
            return (y0 + y1) / 2.0

        t = (x - x0) / h
        t2 = t * t
        t3 = t2 * t

        # Hermite basis
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

    # ==================================================================
    # Private helpers -- polynomial fitting
    # ==================================================================

    def _gather_local_known(
        self,
        values: List[Optional[float]],
        seg_start: int,
        seg_end: int,
        window: int = DEFAULT_POLY_WINDOW,
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Gather known points in a local window around a gap segment.

        Expands outward from the gap boundaries to collect up to
        ``window`` known points on each side.

        Args:
            values: Full series.
            seg_start: Gap start index.
            seg_end: Gap end index.
            window: Number of known points to collect on each side.
            timestamps: Optional timestamps.

        Returns:
            Tuple of (x_coords, y_values) for the local known points.
        """
        n = len(values)
        local_x: List[float] = []
        local_y: List[float] = []

        # Left side
        count = 0
        for j in range(seg_start - 1, -1, -1):
            if not self._is_missing(values[j]):
                tx = float(j) if timestamps is None else timestamps[j]
                local_x.insert(0, tx)
                local_y.insert(0, values[j])  # type: ignore[arg-type]
                count += 1
                if count >= window:
                    break

        # Right side
        count = 0
        for j in range(seg_end + 1, n):
            if not self._is_missing(values[j]):
                tx = float(j) if timestamps is None else timestamps[j]
                local_x.append(tx)
                local_y.append(values[j])  # type: ignore[arg-type]
                count += 1
                if count >= window:
                    break

        return local_x, local_y

    @staticmethod
    def _fit_polynomial(
        x: List[float],
        y: List[float],
        degree: int,
    ) -> List[float]:
        """Fit a polynomial of given degree via Vandermonde + Gauss elimination.

        Solves the normal equations V^T * V * c = V^T * y where V is the
        Vandermonde matrix. Uses partial pivoting for numerical stability.

        Args:
            x: Known x-coordinates.
            y: Known y-values.
            degree: Polynomial degree.

        Returns:
            List of polynomial coefficients [c0, c1, ..., c_degree]
            such that p(x) = c0 + c1*x + c2*x^2 + ... .
        """
        n = len(x)
        d = min(degree, n - 1)
        cols = d + 1

        # Build Vandermonde matrix V (n x cols)
        v: List[List[float]] = []
        for i in range(n):
            row = [1.0]
            for j in range(1, cols):
                row.append(row[-1] * x[i])
            v.append(row)

        # Compute V^T * V  (cols x cols)
        vtv: List[List[float]] = [
            [0.0] * cols for _ in range(cols)
        ]
        for i in range(cols):
            for j in range(cols):
                s = 0.0
                for k in range(n):
                    s += v[k][i] * v[k][j]
                vtv[i][j] = s

        # Compute V^T * y  (cols x 1)
        vty: List[float] = [0.0] * cols
        for i in range(cols):
            s = 0.0
            for k in range(n):
                s += v[k][i] * y[k]
            vty[i] = s

        # Gaussian elimination with partial pivoting
        aug: List[List[float]] = [
            vtv[i][:] + [vty[i]] for i in range(cols)
        ]

        for col in range(cols):
            # Partial pivot
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, cols):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if abs(pivot) < _EPS:
                continue

            for row in range(col + 1, cols):
                factor = aug[row][col] / pivot
                for k in range(col, cols + 1):
                    aug[row][k] -= factor * aug[col][k]

        # Back substitution
        coeffs = [0.0] * cols
        for i in range(cols - 1, -1, -1):
            if abs(aug[i][i]) < _EPS:
                coeffs[i] = 0.0
                continue
            s = aug[i][cols]
            for j in range(i + 1, cols):
                s -= aug[i][j] * coeffs[j]
            coeffs[i] = s / aug[i][i]

        return coeffs

    @staticmethod
    def _evaluate_polynomial(
        coeffs: List[float],
        x: float,
    ) -> float:
        """Evaluate polynomial using Horner's method.

        p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...

        Args:
            coeffs: Polynomial coefficients (low-to-high order).
            x: Query position.

        Returns:
            Polynomial value at x.
        """
        if not coeffs:
            return 0.0
        result = coeffs[-1]
        for i in range(len(coeffs) - 2, -1, -1):
            result = result * x + coeffs[i]
        return result

    # ==================================================================
    # Private helpers -- collecting known points
    # ==================================================================

    def _collect_known(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Collect all non-missing (x, y) pairs from the series.

        Args:
            values: Series with possible missing values.
            timestamps: Optional timestamps; if None, integer indices
                are used.

        Returns:
            Tuple of (x_coords, y_values) for known points.
        """
        known_x: List[float] = []
        known_y: List[float] = []
        for i, v in enumerate(values):
            if not self._is_missing(v):
                tx = float(i) if timestamps is None else timestamps[i]
                known_x.append(tx)
                known_y.append(v)  # type: ignore[arg-type]
        return known_x, known_y

    # ==================================================================
    # Private helpers -- nearest-value selection
    # ==================================================================

    @staticmethod
    def _pick_nearest(
        pos: int,
        left_idx: Optional[int],
        left_val: Optional[float],
        right_idx: Optional[int],
        right_val: Optional[float],
        timestamps: Optional[List[float]] = None,
    ) -> float:
        """Pick the nearest non-missing value by distance.

        When both are equidistant, returns the left value.

        Args:
            pos: Gap position.
            left_idx: Left neighbour index.
            left_val: Left neighbour value.
            right_idx: Right neighbour index.
            right_val: Right neighbour value.
            timestamps: Optional timestamps for distance computation.

        Returns:
            Nearest known value, or 0.0 if none exist.
        """
        if left_val is None and right_val is None:
            return 0.0
        if left_val is None:
            return right_val  # type: ignore[return-value]
        if right_val is None:
            return left_val

        if timestamps is not None and left_idx is not None and right_idx is not None:
            dist_left = abs(timestamps[pos] - timestamps[left_idx])
            dist_right = abs(timestamps[right_idx] - timestamps[pos])
        else:
            dist_left = abs(pos - (left_idx or 0))
            dist_right = abs((right_idx or 0) - pos)

        if dist_left <= dist_right:
            return left_val
        return right_val

    # ==================================================================
    # Private helpers -- confidence computation
    # ==================================================================

    @staticmethod
    def _compute_confidence(
        gap_length: int,
        method: str,
        series_length: int,
    ) -> float:
        """Compute fill confidence based on gap size and method.

        Confidence starts at the method's base value and decays as
        the gap length grows relative to the series length. The decay
        follows an exponential curve.

        Args:
            gap_length: Number of consecutive missing values.
            method: Interpolation method name.
            series_length: Total length of the series.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        base = _METHOD_CONFIDENCE_BASE.get(method, 0.70)

        if series_length <= 0 or gap_length <= 0:
            return base

        # Gap ratio: fraction of series that is this gap
        gap_ratio = gap_length / max(series_length, 1)

        # Decay factor: exponential decay as gap grows
        # k controls decay rate; gaps > 20% of series lose confidence fast
        k = 5.0
        decay = math.exp(-k * gap_ratio)

        # Additional penalty for absolute gap length
        length_penalty = max(0.0, 1.0 - gap_length / 50.0)

        confidence = base * decay * (0.5 + 0.5 * length_penalty)

        return max(0.05, min(1.0, confidence))

    # ==================================================================
    # Private helpers -- provenance
    # ==================================================================

    def _build_point_provenance(
        self,
        method: str,
        index: int,
        value: float,
        gap_length: int,
    ) -> str:
        """Build a SHA-256 provenance hash for a single filled point.

        Args:
            method: Interpolation method used.
            index: Position in the series.
            value: Filled value.
            gap_length: Length of the gap this point belongs to.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._provenance.build_hash({
            "engine": "interpolation",
            "method": method,
            "index": index,
            "value": value,
            "gap_length": gap_length,
        })

    def _build_operation_provenance(
        self,
        method: str,
        input_hash: str,
        output_hash: str,
        gaps_found: int,
        total_missing: int,
    ) -> str:
        """Build a SHA-256 provenance hash for the entire fill operation.

        Records the operation in the provenance chain and returns the
        resulting chain hash.

        Args:
            method: Interpolation method used.
            input_hash: Hash of the input series.
            output_hash: Hash of the filled series.
            gaps_found: Number of gap segments.
            total_missing: Total missing values filled.

        Returns:
            Chain hash from the provenance tracker.
        """
        return self._provenance.add_to_chain(
            operation=f"interpolate_{method}",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "method": method,
                "gaps_found": gaps_found,
                "total_missing": total_missing,
            },
        )

    # ==================================================================
    # Private helpers -- result building
    # ==================================================================

    def _build_result(
        self,
        filled: List[Optional[float]],
        filled_points: List[FilledPoint],
        method_name: str,
        gaps_found: int,
        total_missing: int,
        start_time: float,
    ) -> FillResult:
        """Build a FillResult from the completed interpolation.

        Computes statistics, records provenance, and emits metrics.

        Args:
            filled: Completed series with gaps filled.
            filled_points: Per-point fill metadata.
            method_name: Interpolation method used.
            gaps_found: Number of gap segments detected.
            total_missing: Total missing values filled.
            start_time: Wall-clock start time (from ``time.time()``).

        Returns:
            Complete FillResult.
        """
        elapsed_ms = (time.time() - start_time) * 1000.0

        # Cast to float list (replace any remaining None with 0.0)
        filled_floats: List[float] = [
            v if v is not None else 0.0 for v in filled
        ]

        # Confidence statistics
        if filled_points:
            confidences = [fp.confidence for fp in filled_points]
            mean_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
        else:
            mean_conf = 1.0
            min_conf = 1.0

        # Provenance for full operation
        input_hash = self._provenance.build_hash({
            "operation": "interpolation_input",
            "method": method_name,
            "series_length": len(filled),
            "gaps_found": gaps_found,
        })
        output_hash = self._provenance.build_hash({
            "operation": "interpolation_output",
            "method": method_name,
            "filled_count": total_missing,
            "mean_confidence": round(mean_conf, 6),
        })
        op_provenance = self._build_operation_provenance(
            method_name, input_hash, output_hash,
            gaps_found, total_missing,
        )

        # Emit metrics
        _inc_gaps_filled(method_name, total_missing)
        _observe_confidence(method_name, mean_conf)
        _observe_duration(f"interpolate_{method_name}", elapsed_ms / 1000.0)

        result = FillResult(
            result_id=str(uuid.uuid4()),
            filled_values=filled_floats,
            filled_points=filled_points,
            method=method_name,
            gaps_found=gaps_found,
            gaps_filled=gaps_found,
            total_missing=total_missing,
            mean_confidence=round(mean_conf, 6),
            min_confidence=round(min_conf, 6),
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=op_provenance,
            created_at=_utcnow().isoformat(),
            metadata={
                "engine": "InterpolationEngine",
                "version": "1.0.0",
            },
        )

        logger.info(
            "InterpolationEngine.%s complete: gaps=%d, filled=%d, "
            "mean_conf=%.4f, elapsed=%.1fms",
            method_name, gaps_found, total_missing,
            mean_conf, elapsed_ms,
        )
        return result

    def _fallback_fill(
        self,
        values: List[Optional[float]],
        method_name: str,
        start_time: float,
        reason: str = "insufficient_data",
    ) -> FillResult:
        """Build a FillResult when filling is not possible.

        Returns the original values with any None replaced by 0.0
        and zero confidence for those points.

        Args:
            values: Original series.
            method_name: Requested method name.
            start_time: Wall-clock start time.
            reason: Reason the fill could not be performed.

        Returns:
            FillResult with low-confidence fallback values.
        """
        filled: List[float] = []
        filled_points: List[FilledPoint] = []
        total_missing = 0

        for i, v in enumerate(values):
            if self._is_missing(v):
                filled.append(0.0)
                total_missing += 1
                prov = self._build_point_provenance(
                    method_name, i, 0.0, 0,
                )
                filled_points.append(FilledPoint(
                    index=i,
                    original_value=None,
                    filled_value=0.0,
                    was_missing=True,
                    confidence=0.0,
                    method=method_name,
                    gap_length=0,
                    provenance_hash=prov,
                ))
            else:
                filled.append(v)  # type: ignore[arg-type]

        gap_segments = self._find_gap_segments(values)
        gaps_found = len(gap_segments)

        logger.warning(
            "InterpolationEngine.%s fallback: %s (gaps=%d, missing=%d)",
            method_name, reason, gaps_found, total_missing,
        )

        return self._build_result(
            filled, filled_points, method_name,
            gaps_found, total_missing, start_time,
        )


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "InterpolationEngine",
    "FillResult",
    "FillStrategy",
    "FilledPoint",
    "MAX_POLYNOMIAL_DEGREE",
    "DEFAULT_POLY_WINDOW",
]
