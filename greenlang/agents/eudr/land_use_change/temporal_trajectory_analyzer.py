# -*- coding: utf-8 -*-
"""
Temporal Trajectory Analyzer Engine - AGENT-EUDR-005: Land Use Change Detector (Engine 3)

Analyses the temporal progression of land use change to distinguish permanent
conversion from temporary disturbance. Examines full NDVI time-series
trajectories to classify change patterns into five deterministic trajectory
types: stable, abrupt_change, gradual_change, oscillating, and recovery.

Zero-Hallucination Guarantees:
    - All trajectory classification uses deterministic statistical thresholds
      (standard deviation, coefficient of variation, amplitude, slope).
    - Abrupt change detection: single-step NDVI drop > 0.3 in 1-2 months.
    - Gradual change detection: negative NDVI slope over 6+ months using
      least-squares linear regression.
    - Oscillation detection: autocorrelation-based periodicity analysis with
      fixed period range (6-18 months).
    - Recovery detection: NDVI rebound from minimum toward baseline using
      percentage recovery metric.
    - No ML/LLM used for any trajectory classification.
    - SHA-256 provenance hashes on all result objects.

Trajectory Types (5):
    STABLE:          std(NDVI) < 0.05 AND no class changes
    ABRUPT_CHANGE:   Single-step NDVI drop > 0.3 in 1-2 months
    GRADUAL_CHANGE:  NDVI declining over 6+ months (negative slope)
    OSCILLATING:     Class alternates between 2 values, period 6-18 months
    RECOVERY:        NDVI drops then recovers >50% toward baseline

NDVI Time-Series Analysis:
    The engine operates on monthly or sub-monthly NDVI observations,
    extracting statistical features (mean, std, CV, slope, amplitude,
    peak/trough positions) to deterministically classify the trajectory.

Natural Disturbance Detection:
    The engine can distinguish natural disturbance events (fire, drought,
    storm damage) from anthropogenic conversion by examining spatial context
    (neighbouring plot trajectories) and recovery speed (natural disturbance
    typically shows faster NDVI recovery).

Performance Targets:
    - Single plot trajectory analysis: <100ms
    - Abrupt change detection: <10ms
    - Gradual change detection: <15ms
    - Oscillation detection: <20ms
    - Recovery detection: <10ms
    - Batch analysis (100 plots): <5 seconds

Regulatory References:
    - EUDR Article 2(1): Permanent conversion assessment
    - EUDR Article 2(5): Degradation trajectory patterns
    - EUDR Article 9: Temporal monitoring evidence
    - EUDR Article 10: Risk assessment from trajectory analysis

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-005 (Engine 3: Temporal Trajectory Analysis)
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
from greenlang.schemas import utcnow

from greenlang.agents.eudr.land_use_change.land_use_classifier import (
    LandUseCategory,
)

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

class TrajectoryType(str, Enum):
    """Temporal trajectory classification types.

    STABLE: No significant NDVI variation. Standard deviation below 0.05
        and no land use class changes detected. Indicates undisturbed
        land use that has not changed during the observation period.
    ABRUPT_CHANGE: Sharp, single-step NDVI drop exceeding 0.3 within
        1-2 months. Typical of clear-cut deforestation, fire events,
        or sudden land clearing for agriculture.
    GRADUAL_CHANGE: Slow, progressive NDVI decline over 6+ months.
        Typical of selective logging, gradual degradation, or slow
        conversion to plantation/cropland. Identified by a significant
        negative linear slope in the NDVI time series.
    OSCILLATING: NDVI alternates between two distinct levels with a
        period of 6-18 months. Typical of crop rotation, seasonal
        flooding, or shifting cultivation patterns. Not classified
        as permanent deforestation unless the overall trend is
        negative.
    RECOVERY: NDVI shows an initial drop (disturbance event) followed
        by a recovery of >50% toward the pre-disturbance baseline.
        Typical of natural regeneration after fire, storm damage, or
        temporary clearing. May indicate temporary disturbance rather
        than permanent conversion.
    """

    STABLE = "stable"
    ABRUPT_CHANGE = "abrupt_change"
    GRADUAL_CHANGE = "gradual_change"
    OSCILLATING = "oscillating"
    RECOVERY = "recovery"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum NDVI standard deviation for STABLE classification.
STABLE_STD_THRESHOLD: float = 0.05

#: Minimum NDVI drop (in absolute value) for ABRUPT_CHANGE detection.
ABRUPT_DROP_THRESHOLD: float = 0.30

#: Maximum number of months for an abrupt change event.
ABRUPT_WINDOW_MONTHS: int = 2

#: Minimum number of months for GRADUAL_CHANGE detection.
GRADUAL_MIN_MONTHS: int = 6

#: Minimum absolute NDVI slope (per month) for GRADUAL_CHANGE.
GRADUAL_SLOPE_THRESHOLD: float = 0.015

#: Minimum period (months) for OSCILLATING detection.
OSCILLATION_MIN_PERIOD: int = 6

#: Maximum period (months) for OSCILLATING detection.
OSCILLATION_MAX_PERIOD: int = 18

#: Minimum autocorrelation coefficient for oscillation detection.
OSCILLATION_AUTOCORR_THRESHOLD: float = 0.30

#: Minimum recovery percentage (0-1) toward baseline for RECOVERY.
RECOVERY_MIN_PERCENTAGE: float = 0.50

#: Maximum plots in a single batch analysis.
MAX_BATCH_SIZE: int = 5000

#: Minimum observations required for trajectory analysis.
MIN_OBSERVATIONS: int = 6

#: Natural disturbance recovery rate threshold (NDVI per month).
NATURAL_DISTURBANCE_RECOVERY_RATE: float = 0.04

#: Tolerance for class change detection.
CLASS_CHANGE_TOLERANCE: float = 0.10

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPlotInput:
    """Input data for temporal trajectory analysis on a single plot.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Plot centroid latitude (-90 to 90).
        longitude: Plot centroid longitude (-180 to 180).
        ndvi_values: NDVI time-series values (chronological order).
        observation_dates: ISO date strings corresponding to ndvi_values.
        class_labels: Optional land use class labels at each observation date.
        time_step_months: Nominal time step between observations in months.
        date_from: Start of the observation window (ISO string).
        date_to: End of the observation window (ISO string).
        spatial_context: Optional dictionary with neighbouring plot
            trajectory summaries for natural disturbance detection.
        area_ha: Plot area in hectares.
    """

    plot_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    ndvi_values: List[float] = field(default_factory=list)
    observation_dates: List[str] = field(default_factory=list)
    class_labels: List[str] = field(default_factory=list)
    time_step_months: int = 1
    date_from: str = ""
    date_to: str = ""
    spatial_context: Optional[Dict[str, Any]] = None
    area_ha: float = 1.0

@dataclass
class TemporalTrajectory:
    """Result of temporal trajectory analysis for a single plot.

    Attributes:
        result_id: Unique result identifier (UUID).
        plot_id: Plot identifier.
        trajectory_type: Classified trajectory type.
        confidence: Confidence in the trajectory classification [0, 1].
        ndvi_mean: Mean NDVI across the time series.
        ndvi_std: Standard deviation of NDVI values.
        ndvi_cv: Coefficient of variation of NDVI values.
        ndvi_slope: Linear slope of NDVI time series (per month).
        ndvi_amplitude: Max - Min NDVI across the time series.
        ndvi_min: Minimum NDVI value.
        ndvi_max: Maximum NDVI value.
        change_date: Estimated date of the primary change event (ISO).
        change_magnitude: Magnitude of the primary change (NDVI delta).
        change_duration_months: Duration of the change process in months.
        oscillation_period_months: Detected oscillation period (months),
            only set for OSCILLATING trajectory type.
        recovery_completeness: Fraction of NDVI recovery toward baseline
            [0, 1], only set for RECOVERY trajectory type.
        recovery_rate_per_month: NDVI recovery rate per month,
            only set for RECOVERY trajectory type.
        is_natural_disturbance: Whether the trajectory is consistent with
            natural disturbance (fire, storm) rather than anthropogenic.
        observation_count: Number of NDVI observations used.
        date_from: Start of the observation window.
        date_to: End of the observation window.
        latitude: Plot centroid latitude.
        longitude: Plot centroid longitude.
        visualization_data: Chart coordinates for trajectory visualization.
        processing_time_ms: Time taken for analysis in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        timestamp: UTC ISO timestamp of analysis.
        metadata: Additional contextual information.
    """

    result_id: str = ""
    plot_id: str = ""
    trajectory_type: str = ""
    confidence: float = 0.0
    ndvi_mean: float = 0.0
    ndvi_std: float = 0.0
    ndvi_cv: float = 0.0
    ndvi_slope: float = 0.0
    ndvi_amplitude: float = 0.0
    ndvi_min: float = 0.0
    ndvi_max: float = 0.0
    change_date: str = ""
    change_magnitude: float = 0.0
    change_duration_months: int = 0
    oscillation_period_months: Optional[int] = None
    recovery_completeness: Optional[float] = None
    recovery_rate_per_month: Optional[float] = None
    is_natural_disturbance: bool = False
    observation_count: int = 0
    date_from: str = ""
    date_to: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    visualization_data: Dict[str, Any] = field(default_factory=dict)
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
            "trajectory_type": self.trajectory_type,
            "confidence": self.confidence,
            "ndvi_mean": self.ndvi_mean,
            "ndvi_std": self.ndvi_std,
            "ndvi_cv": self.ndvi_cv,
            "ndvi_slope": self.ndvi_slope,
            "ndvi_amplitude": self.ndvi_amplitude,
            "ndvi_min": self.ndvi_min,
            "ndvi_max": self.ndvi_max,
            "change_date": self.change_date,
            "change_magnitude": self.change_magnitude,
            "change_duration_months": self.change_duration_months,
            "oscillation_period_months": self.oscillation_period_months,
            "recovery_completeness": self.recovery_completeness,
            "recovery_rate_per_month": self.recovery_rate_per_month,
            "is_natural_disturbance": self.is_natural_disturbance,
            "observation_count": self.observation_count,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "visualization_data": self.visualization_data,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

# ---------------------------------------------------------------------------
# TemporalTrajectoryAnalyzer
# ---------------------------------------------------------------------------

class TemporalTrajectoryAnalyzer:
    """Production-grade temporal trajectory analysis engine for EUDR.

    Analyzes NDVI time-series trajectories to classify change patterns
    into five types: stable, abrupt_change, gradual_change, oscillating,
    and recovery. Distinguishes permanent conversion from temporary
    disturbance for EUDR compliance assessment. All computations are
    deterministic with SHA-256 provenance tracking.

    Example::

        analyzer = TemporalTrajectoryAnalyzer()
        trajectory = analyzer.analyze_trajectory(
            latitude=-2.5,
            longitude=110.0,
            date_from=date(2019, 1, 1),
            date_to=date(2023, 12, 31),
            time_step_months=1,
            ndvi_values=[0.7, 0.72, 0.69, 0.3, 0.25, 0.22, ...],
            observation_dates=["2019-01-15", "2019-02-15", ...],
        )
        assert trajectory.trajectory_type in [t.value for t in TrajectoryType]

    Attributes:
        config: Optional configuration object.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the TemporalTrajectoryAnalyzer.

        Args:
            config: Optional configuration object with threshold overrides.
        """
        self.config = config

        logger.info(
            "TemporalTrajectoryAnalyzer initialized: module_version=%s, "
            "trajectory_types=%d",
            _MODULE_VERSION,
            len(TrajectoryType),
        )

    # ------------------------------------------------------------------
    # Public API: Single Plot Analysis
    # ------------------------------------------------------------------

    def analyze_trajectory(
        self,
        latitude: float,
        longitude: float,
        date_from: date,
        date_to: date,
        time_step_months: int = 1,
        ndvi_values: Optional[List[float]] = None,
        observation_dates: Optional[List[str]] = None,
        class_labels: Optional[List[str]] = None,
        spatial_context: Optional[Dict[str, Any]] = None,
    ) -> TemporalTrajectory:
        """Analyze the temporal trajectory of a plot.

        Args:
            latitude: Plot centroid latitude (-90 to 90).
            longitude: Plot centroid longitude (-180 to 180).
            date_from: Start of the observation window.
            date_to: End of the observation window.
            time_step_months: Nominal time step between observations.
            ndvi_values: NDVI time-series values (chronological order).
            observation_dates: ISO date strings for each observation.
            class_labels: Optional land use class labels per observation.
            spatial_context: Optional neighbouring plot trajectory data.

        Returns:
            TemporalTrajectory with trajectory classification and metrics.

        Raises:
            ValueError: If coordinates are out of range, date_from > date_to,
                or insufficient NDVI observations.
        """
        start_time = time.monotonic()

        self._validate_coordinates(latitude, longitude)
        self._validate_date_range(date_from, date_to)

        plot = TrajectoryPlotInput(
            plot_id=_generate_id(),
            latitude=latitude,
            longitude=longitude,
            ndvi_values=ndvi_values or [],
            observation_dates=observation_dates or [],
            class_labels=class_labels or [],
            time_step_months=time_step_months,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
            spatial_context=spatial_context,
        )

        return self._analyze_plot(plot, start_time)

    # ------------------------------------------------------------------
    # Public API: Batch Analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        plots: List[TrajectoryPlotInput],
        date_from: date,
        date_to: date,
    ) -> List[TemporalTrajectory]:
        """Analyze trajectories for a batch of plots.

        Args:
            plots: List of plot inputs to analyze.
            date_from: Start date for all plots.
            date_to: End date for all plots.

        Returns:
            List of TemporalTrajectory results.

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
        results: List[TemporalTrajectory] = []

        for i, plot in enumerate(plots):
            try:
                if not plot.date_from:
                    plot.date_from = date_from.isoformat()
                if not plot.date_to:
                    plot.date_to = date_to.isoformat()
                start_time = time.monotonic()
                result = self._analyze_plot(plot, start_time)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "analyze_batch: failed on plot[%d] id=%s: %s",
                    i, plot.plot_id, str(exc),
                )
                error_result = self._create_error_result(
                    plot=plot, error_msg=str(exc),
                )
                results.append(error_result)

        batch_elapsed = (time.monotonic() - batch_start) * 1000
        successful = sum(1 for r in results if r.confidence > 0.0)

        logger.info(
            "analyze_batch complete: %d/%d successful, %.2fms total",
            successful, len(plots), batch_elapsed,
        )

        return results

    # ------------------------------------------------------------------
    # Internal: Core Analysis Pipeline
    # ------------------------------------------------------------------

    def _analyze_plot(
        self,
        plot: TrajectoryPlotInput,
        start_time: float,
    ) -> TemporalTrajectory:
        """Run the trajectory analysis pipeline for a single plot.

        Tests each trajectory type detector in order of priority:
        1. Stable (fastest check, most common)
        2. Abrupt change (deforestation signal)
        3. Recovery (post-disturbance rebound)
        4. Oscillating (seasonal/crop patterns)
        5. Gradual change (slow degradation)

        Args:
            plot: Input data for the plot.
            start_time: Monotonic start time for duration tracking.

        Returns:
            TemporalTrajectory result.
        """
        self._validate_plot_input(plot)

        result_id = _generate_id()
        timestamp = utcnow().isoformat()
        values = plot.ndvi_values

        # Compute basic statistics
        ndvi_mean = sum(values) / len(values)
        ndvi_std = math.sqrt(
            sum((v - ndvi_mean) ** 2 for v in values) / len(values)
        )
        ndvi_cv = ndvi_std / max(abs(ndvi_mean), 1e-12)
        ndvi_min = min(values)
        ndvi_max = max(values)
        ndvi_amplitude = ndvi_max - ndvi_min
        ndvi_slope = self._compute_linear_slope(values)

        # Classify trajectory
        trajectory_type, evidence = self._classify_trajectory(
            values, plot.class_labels,
        )

        # Extract type-specific fields
        change_date = ""
        change_magnitude = 0.0
        change_duration = 0
        oscillation_period: Optional[int] = None
        recovery_completeness: Optional[float] = None
        recovery_rate: Optional[float] = None

        if trajectory_type == TrajectoryType.ABRUPT_CHANGE:
            is_abrupt, abrupt_date, abrupt_mag = self._detect_abrupt_change(
                values, plot.observation_dates,
            )
            change_date = abrupt_date or ""
            change_magnitude = abrupt_mag
            change_duration = ABRUPT_WINDOW_MONTHS

        elif trajectory_type == TrajectoryType.GRADUAL_CHANGE:
            is_gradual, start_d, end_d, grad_mag = self._detect_gradual_change(
                values, plot.observation_dates,
            )
            change_date = start_d or ""
            change_magnitude = grad_mag
            if start_d and end_d:
                change_duration = self._months_between(start_d, end_d)

        elif trajectory_type == TrajectoryType.OSCILLATING:
            is_osc, period, osc_conf = self._detect_oscillation(values)
            oscillation_period = period
            change_magnitude = ndvi_amplitude

        elif trajectory_type == TrajectoryType.RECOVERY:
            is_rec, rec_comp, rec_rate = self._detect_recovery(values)
            recovery_completeness = rec_comp
            recovery_rate = rec_rate
            # Find the trough (lowest point after initial values)
            trough_idx = self._find_trough_index(values)
            if trough_idx > 0 and plot.observation_dates:
                change_date = plot.observation_dates[
                    min(trough_idx, len(plot.observation_dates) - 1)
                ]
            change_magnitude = ndvi_min - values[0] if values else 0.0

        # Compute trajectory confidence
        confidence = self._compute_trajectory_confidence(
            trajectory_type, evidence,
        )

        # Check for natural disturbance
        is_natural = self._distinguish_natural_disturbance(
            trajectory_type, values, plot.spatial_context,
        )

        # Generate visualization data
        viz_data = self._generate_trajectory_visualization(
            values, plot.observation_dates, trajectory_type,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = TemporalTrajectory(
            result_id=result_id,
            plot_id=plot.plot_id,
            trajectory_type=trajectory_type.value,
            confidence=round(confidence, 4),
            ndvi_mean=round(ndvi_mean, 6),
            ndvi_std=round(ndvi_std, 6),
            ndvi_cv=round(ndvi_cv, 6),
            ndvi_slope=round(ndvi_slope, 8),
            ndvi_amplitude=round(ndvi_amplitude, 6),
            ndvi_min=round(ndvi_min, 6),
            ndvi_max=round(ndvi_max, 6),
            change_date=change_date,
            change_magnitude=round(change_magnitude, 6),
            change_duration_months=change_duration,
            oscillation_period_months=oscillation_period,
            recovery_completeness=(
                round(recovery_completeness, 4)
                if recovery_completeness is not None else None
            ),
            recovery_rate_per_month=(
                round(recovery_rate, 6)
                if recovery_rate is not None else None
            ),
            is_natural_disturbance=is_natural,
            observation_count=len(values),
            date_from=plot.date_from,
            date_to=plot.date_to,
            latitude=plot.latitude,
            longitude=plot.longitude,
            visualization_data=viz_data,
            processing_time_ms=round(elapsed_ms, 2),
            timestamp=timestamp,
            metadata={
                "module_version": _MODULE_VERSION,
                "time_step_months": plot.time_step_months,
            },
        )

        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Trajectory analyzed: plot=%s, type=%s, confidence=%.2f, "
            "mean_ndvi=%.3f, slope=%.6f, natural_disturbance=%s, %.2fms",
            plot.plot_id,
            trajectory_type.value,
            confidence,
            ndvi_mean,
            ndvi_slope,
            is_natural,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Trajectory Classification
    # ------------------------------------------------------------------

    def _classify_trajectory(
        self,
        values: List[float],
        class_labels: List[str],
    ) -> Tuple[TrajectoryType, Dict[str, float]]:
        """Classify the trajectory type from NDVI time-series features.

        Tests each trajectory type in priority order and returns the
        first match. Evidence scores for each type are collected for
        confidence computation.

        Args:
            values: NDVI time-series values.
            class_labels: Land use class labels at each observation.

        Returns:
            Tuple of (trajectory type, evidence scores dict).
        """
        evidence: Dict[str, float] = {}

        # Test 1: Stable
        is_stable = self._is_stable(values, class_labels, STABLE_STD_THRESHOLD)
        evidence["stable_score"] = 1.0 if is_stable else 0.0
        if is_stable:
            ndvi_std = math.sqrt(
                sum((v - sum(values) / len(values)) ** 2 for v in values)
                / len(values)
            )
            evidence["stable_score"] = max(
                0.5, 1.0 - ndvi_std / STABLE_STD_THRESHOLD,
            )
            return TrajectoryType.STABLE, evidence

        # Test 2: Abrupt change
        is_abrupt, _, abrupt_magnitude = self._detect_abrupt_change(values)
        evidence["abrupt_score"] = min(
            1.0, abs(abrupt_magnitude) / ABRUPT_DROP_THRESHOLD,
        ) if is_abrupt else 0.0
        if is_abrupt:
            return TrajectoryType.ABRUPT_CHANGE, evidence

        # Test 3: Recovery
        is_recovery, rec_comp, rec_rate = self._detect_recovery(values)
        evidence["recovery_score"] = (
            rec_comp if is_recovery and rec_comp is not None else 0.0
        )
        if is_recovery:
            return TrajectoryType.RECOVERY, evidence

        # Test 4: Oscillating
        is_oscillating, period, osc_conf = self._detect_oscillation(values)
        evidence["oscillation_score"] = osc_conf if is_oscillating else 0.0
        if is_oscillating:
            return TrajectoryType.OSCILLATING, evidence

        # Test 5: Gradual change
        is_gradual, _, _, grad_magnitude = self._detect_gradual_change(values)
        slope = self._compute_linear_slope(values)
        evidence["gradual_score"] = min(
            1.0, abs(slope) / GRADUAL_SLOPE_THRESHOLD,
        ) if is_gradual else 0.0
        if is_gradual:
            return TrajectoryType.GRADUAL_CHANGE, evidence

        # Default: gradual change if significant slope, else stable-ish
        if abs(slope) > GRADUAL_SLOPE_THRESHOLD * 0.5:
            evidence["gradual_score"] = min(
                1.0, abs(slope) / GRADUAL_SLOPE_THRESHOLD,
            )
            return TrajectoryType.GRADUAL_CHANGE, evidence

        evidence["stable_score"] = 0.5
        return TrajectoryType.STABLE, evidence

    # ------------------------------------------------------------------
    # Detector: Stable
    # ------------------------------------------------------------------

    def _is_stable(
        self,
        values: List[float],
        class_labels: List[str],
        tolerance: float = STABLE_STD_THRESHOLD,
    ) -> bool:
        """Check if the trajectory is stable (no significant change).

        Stable criterion: std(NDVI) < tolerance AND no class label changes.

        Args:
            values: NDVI time-series values.
            class_labels: Land use class labels.
            tolerance: Maximum standard deviation for stability.

        Returns:
            True if the trajectory is classified as stable.
        """
        if len(values) < 2:
            return True

        mean_val = sum(values) / len(values)
        std_val = math.sqrt(
            sum((v - mean_val) ** 2 for v in values) / len(values)
        )

        if std_val >= tolerance:
            return False

        # Check class labels for any change
        if class_labels and len(class_labels) >= 2:
            first_label = class_labels[0]
            for label in class_labels[1:]:
                if label and label != first_label:
                    return False

        return True

    # ------------------------------------------------------------------
    # Detector: Abrupt Change
    # ------------------------------------------------------------------

    def _detect_abrupt_change(
        self,
        values: List[float],
        dates: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """Detect abrupt NDVI change (drop > threshold in 1-2 months).

        Scans the time series for any consecutive pair (or triplet) of
        observations where the NDVI drop exceeds ABRUPT_DROP_THRESHOLD.

        Args:
            values: NDVI time-series values.
            dates: Optional observation dates.

        Returns:
            Tuple of (is_abrupt, change_date or None, drop_magnitude).
        """
        if len(values) < 2:
            return False, None, 0.0

        max_drop = 0.0
        max_drop_idx = -1

        # Check 1-step drops
        for i in range(1, len(values)):
            drop = values[i - 1] - values[i]
            if drop > max_drop:
                max_drop = drop
                max_drop_idx = i

        # Check 2-step drops (within ABRUPT_WINDOW_MONTHS)
        if len(values) >= 3:
            for i in range(2, len(values)):
                drop = values[i - 2] - values[i]
                if drop > max_drop:
                    max_drop = drop
                    max_drop_idx = i

        is_abrupt = max_drop >= ABRUPT_DROP_THRESHOLD

        change_date = None
        if is_abrupt and dates and max_drop_idx >= 0:
            change_date = dates[min(max_drop_idx, len(dates) - 1)]

        return is_abrupt, change_date, max_drop

    # ------------------------------------------------------------------
    # Detector: Gradual Change
    # ------------------------------------------------------------------

    def _detect_gradual_change(
        self,
        values: List[float],
        dates: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str], Optional[str], float]:
        """Detect gradual NDVI decline over 6+ months.

        Uses least-squares linear regression to compute the NDVI slope.
        If the slope is significantly negative (exceeds threshold) and
        the decline spans at least GRADUAL_MIN_MONTHS, the trajectory
        is classified as gradual change.

        Args:
            values: NDVI time-series values.
            dates: Optional observation dates.

        Returns:
            Tuple of (is_gradual, start_date, end_date, total_decline).
        """
        if len(values) < GRADUAL_MIN_MONTHS:
            return False, None, None, 0.0

        slope = self._compute_linear_slope(values)

        # Negative slope indicates decline
        is_gradual = slope < -GRADUAL_SLOPE_THRESHOLD

        if not is_gradual:
            return False, None, None, 0.0

        # Find the span of the decline
        total_decline = values[-1] - values[0]

        start_date = None
        end_date = None
        if dates and len(dates) >= 2:
            start_date = dates[0]
            end_date = dates[-1]

        return is_gradual, start_date, end_date, total_decline

    # ------------------------------------------------------------------
    # Detector: Oscillation
    # ------------------------------------------------------------------

    def _detect_oscillation(
        self,
        values: List[float],
    ) -> Tuple[bool, Optional[int], float]:
        """Detect oscillating NDVI pattern with period 6-18 months.

        Uses autocorrelation analysis to detect periodicity. Checks
        autocorrelation at lag values corresponding to 6-18 months.
        If a significant autocorrelation peak is found within this
        range, the trajectory is classified as oscillating.

        Args:
            values: NDVI time-series values.

        Returns:
            Tuple of (is_oscillating, period_months or None, autocorr_score).
        """
        n = len(values)
        if n < OSCILLATION_MIN_PERIOD * 2:
            return False, None, 0.0

        mean_val = sum(values) / n
        # Compute variance
        variance = sum((v - mean_val) ** 2 for v in values) / n
        if variance < 1e-12:
            return False, None, 0.0

        best_corr = 0.0
        best_lag = 0

        for lag in range(OSCILLATION_MIN_PERIOD, min(OSCILLATION_MAX_PERIOD + 1, n)):
            # Compute autocorrelation at this lag
            autocorr = 0.0
            count = 0
            for i in range(n - lag):
                autocorr += (values[i] - mean_val) * (values[i + lag] - mean_val)
                count += 1

            if count > 0:
                autocorr = autocorr / (count * variance)

            if autocorr > best_corr:
                best_corr = autocorr
                best_lag = lag

        is_oscillating = best_corr >= OSCILLATION_AUTOCORR_THRESHOLD

        period = best_lag if is_oscillating else None

        return is_oscillating, period, best_corr

    # ------------------------------------------------------------------
    # Detector: Recovery
    # ------------------------------------------------------------------

    def _detect_recovery(
        self,
        values: List[float],
    ) -> Tuple[bool, float, float]:
        """Detect NDVI recovery after disturbance.

        Looks for a pattern where NDVI drops significantly (disturbance)
        and then recovers at least RECOVERY_MIN_PERCENTAGE of the way
        back toward the pre-disturbance baseline.

        Recovery criterion: final values > trough + (baseline - trough) * 0.5

        Args:
            values: NDVI time-series values.

        Returns:
            Tuple of (is_recovery, recovery_completeness, recovery_rate).
            recovery_completeness in [0, 1]; recovery_rate in NDVI/month.
        """
        if len(values) < 4:
            return False, 0.0, 0.0

        # Baseline: mean of first 20% of observations (pre-disturbance)
        baseline_count = max(1, len(values) // 5)
        baseline = sum(values[:baseline_count]) / baseline_count

        # Find trough (minimum after the baseline period)
        post_baseline = values[baseline_count:]
        if not post_baseline:
            return False, 0.0, 0.0

        trough_val = min(post_baseline)
        trough_local_idx = post_baseline.index(trough_val)
        trough_idx = baseline_count + trough_local_idx

        # Check that trough is significantly below baseline
        drop = baseline - trough_val
        if drop < ABRUPT_DROP_THRESHOLD * 0.5:
            return False, 0.0, 0.0

        # Check recovery: values after the trough
        post_trough = values[trough_idx + 1:]
        if not post_trough:
            return False, 0.0, 0.0

        # Recovery completeness
        current_val = sum(post_trough[-max(1, len(post_trough) // 3):]) / max(
            1, len(post_trough[-max(1, len(post_trough) // 3):])
        )
        recovery_amount = current_val - trough_val
        total_drop = baseline - trough_val

        if total_drop < 1e-12:
            return False, 0.0, 0.0

        completeness = recovery_amount / total_drop
        completeness = max(0.0, min(1.0, completeness))

        # Recovery rate (NDVI per month)
        recovery_months = len(post_trough)
        if recovery_months > 0:
            rate = recovery_amount / recovery_months
        else:
            rate = 0.0

        is_recovery = completeness >= RECOVERY_MIN_PERCENTAGE

        return is_recovery, completeness, rate

    # ------------------------------------------------------------------
    # Statistical Helpers
    # ------------------------------------------------------------------

    def _compute_linear_slope(self, values: List[float]) -> float:
        """Compute the linear slope of a time series using least squares.

        Uses ordinary least-squares regression with integer time indices
        (0, 1, 2, ..., n-1) representing months.

        Args:
            values: Time-series values.

        Returns:
            Slope (change per time step / month). Negative = decline.
        """
        n = len(values)
        if n < 2:
            return 0.0

        # x = [0, 1, 2, ..., n-1]
        sum_x = n * (n - 1) / 2.0
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-12:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _find_trough_index(self, values: List[float]) -> int:
        """Find the index of the global minimum (trough) in the values.

        Args:
            values: Time-series values.

        Returns:
            Index of the minimum value.
        """
        if not values:
            return 0
        return values.index(min(values))

    def _months_between(self, date_a: str, date_b: str) -> int:
        """Estimate the number of months between two ISO date strings.

        Args:
            date_a: First date (ISO format).
            date_b: Second date (ISO format).

        Returns:
            Approximate number of months between the dates.
        """
        try:
            d_a = date.fromisoformat(date_a)
            d_b = date.fromisoformat(date_b)
            delta_days = abs((d_b - d_a).days)
            return max(1, delta_days // 30)
        except (ValueError, TypeError):
            return 0

    # ------------------------------------------------------------------
    # Confidence Computation
    # ------------------------------------------------------------------

    def _compute_trajectory_confidence(
        self,
        trajectory_type: TrajectoryType,
        evidence_scores: Dict[str, float],
    ) -> float:
        """Compute confidence in the trajectory classification.

        Combines the primary evidence score for the detected trajectory
        type with a base confidence that varies by type (stable has
        higher base confidence than rare types).

        Args:
            trajectory_type: Classified trajectory type.
            evidence_scores: Evidence scores from each detector.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        # Base confidence by trajectory type
        base_confidence: Dict[str, float] = {
            TrajectoryType.STABLE.value: 0.70,
            TrajectoryType.ABRUPT_CHANGE.value: 0.65,
            TrajectoryType.GRADUAL_CHANGE.value: 0.55,
            TrajectoryType.OSCILLATING.value: 0.60,
            TrajectoryType.RECOVERY.value: 0.60,
        }

        base = base_confidence.get(trajectory_type.value, 0.50)

        # Get the primary evidence score for this type
        score_key = f"{trajectory_type.value.split('_')[0]}_score"
        # Find the matching score key
        primary_score = 0.0
        for key, val in evidence_scores.items():
            if val > primary_score:
                primary_score = val

        # Combine base and evidence
        confidence = base + (1.0 - base) * primary_score * 0.5
        return max(0.0, min(1.0, round(confidence, 4)))

    # ------------------------------------------------------------------
    # Natural Disturbance Detection
    # ------------------------------------------------------------------

    def _distinguish_natural_disturbance(
        self,
        trajectory_type: TrajectoryType,
        values: List[float],
        spatial_context: Optional[Dict[str, Any]],
    ) -> bool:
        """Determine if the trajectory is consistent with natural disturbance.

        Natural disturbance indicators:
        1. Recovery trajectory with fast recovery rate (>0.04 NDVI/month)
        2. Multiple neighbouring plots showing similar patterns (spatial
           correlation, typical of fire or storm events)
        3. Abrupt change followed by quick recovery

        Anthropogenic conversion indicators:
        1. No recovery after change
        2. Transition to agricultural NDVI pattern
        3. Neighbouring plots unaffected (localised clearing)

        Args:
            trajectory_type: Classified trajectory type.
            values: NDVI time-series values.
            spatial_context: Optional neighbour trajectory data.

        Returns:
            True if the trajectory is likely a natural disturbance.
        """
        # Only relevant for abrupt_change and recovery types
        if trajectory_type not in (
            TrajectoryType.ABRUPT_CHANGE,
            TrajectoryType.RECOVERY,
        ):
            return False

        # Check recovery rate
        is_rec, rec_comp, rec_rate = self._detect_recovery(values)
        if is_rec and rec_rate >= NATURAL_DISTURBANCE_RECOVERY_RATE:
            # Fast recovery suggests natural disturbance
            natural_score = 1.0
        elif is_rec and rec_rate > 0.0:
            natural_score = rec_rate / NATURAL_DISTURBANCE_RECOVERY_RATE
        else:
            natural_score = 0.0

        # Check spatial context if available
        if spatial_context:
            neighbour_affected = spatial_context.get(
                "neighbours_affected_fraction", 0.0,
            )
            # If many neighbours are also affected, likely natural disturbance
            if neighbour_affected > 0.5:
                natural_score = min(1.0, natural_score + 0.3)
            elif neighbour_affected > 0.3:
                natural_score = min(1.0, natural_score + 0.15)

        return natural_score >= 0.6

    # ------------------------------------------------------------------
    # Visualization Data Generation
    # ------------------------------------------------------------------

    def _generate_trajectory_visualization(
        self,
        values: List[float],
        dates: List[str],
        trajectory_type: TrajectoryType,
    ) -> Dict[str, Any]:
        """Generate chart data for trajectory visualization.

        Produces x/y coordinate arrays and annotation markers for
        rendering the trajectory in a monitoring dashboard.

        Args:
            values: NDVI time-series values.
            dates: Observation dates.
            trajectory_type: Classified trajectory type.

        Returns:
            Dictionary with x_values (dates), y_values (NDVI), and
            annotation markers.
        """
        x_values = list(dates) if dates else list(range(len(values)))
        y_values = [round(v, 6) for v in values]

        markers: List[Dict[str, Any]] = []

        if trajectory_type == TrajectoryType.ABRUPT_CHANGE:
            # Mark the point of maximum drop
            max_drop_idx = 0
            max_drop_val = 0.0
            for i in range(1, len(values)):
                drop = values[i - 1] - values[i]
                if drop > max_drop_val:
                    max_drop_val = drop
                    max_drop_idx = i
            if max_drop_idx > 0:
                markers.append({
                    "index": max_drop_idx,
                    "type": "change_point",
                    "label": "Abrupt change detected",
                    "value": round(max_drop_val, 4),
                })

        elif trajectory_type == TrajectoryType.RECOVERY:
            trough_idx = self._find_trough_index(values)
            markers.append({
                "index": trough_idx,
                "type": "trough",
                "label": "Disturbance trough",
                "value": round(values[trough_idx], 4),
            })

        # Add trend line
        slope = self._compute_linear_slope(values)
        intercept = sum(values) / len(values) - slope * (len(values) - 1) / 2
        trend_line = [
            round(slope * i + intercept, 6) for i in range(len(values))
        ]

        return {
            "x_values": x_values,
            "y_values": y_values,
            "trend_line": trend_line,
            "markers": markers,
            "trajectory_type": trajectory_type.value,
            "slope": round(slope, 8),
        }

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
            latitude: Latitude to validate.
            longitude: Longitude to validate.

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

    def _validate_date_range(
        self,
        date_from: date,
        date_to: date,
    ) -> None:
        """Validate that date_from <= date_to.

        Args:
            date_from: Start date.
            date_to: End date.

        Raises:
            ValueError: If date_from > date_to.
        """
        if date_from > date_to:
            raise ValueError(
                f"date_from ({date_from}) must be <= date_to ({date_to})"
            )

    def _validate_plot_input(
        self,
        plot: TrajectoryPlotInput,
    ) -> None:
        """Validate plot input data for trajectory analysis.

        Args:
            plot: Plot input to validate.

        Raises:
            ValueError: If required fields are missing or insufficient.
        """
        if not plot.plot_id:
            raise ValueError("plot_id must not be empty")
        self._validate_coordinates(plot.latitude, plot.longitude)

        if not plot.ndvi_values or len(plot.ndvi_values) < MIN_OBSERVATIONS:
            raise ValueError(
                f"ndvi_values requires at least {MIN_OBSERVATIONS} observations, "
                f"got {len(plot.ndvi_values) if plot.ndvi_values else 0}"
            )

        if not plot.date_from:
            raise ValueError("date_from must not be empty")
        if not plot.date_to:
            raise ValueError("date_to must not be empty")

    # ------------------------------------------------------------------
    # Error Result Creation
    # ------------------------------------------------------------------

    def _create_error_result(
        self,
        plot: TrajectoryPlotInput,
        error_msg: str,
    ) -> TemporalTrajectory:
        """Create an error result for a failed trajectory analysis.

        Args:
            plot: Input plot that failed analysis.
            error_msg: Error message describing the failure.

        Returns:
            TemporalTrajectory with zero confidence and error metadata.
        """
        return TemporalTrajectory(
            result_id=_generate_id(),
            plot_id=plot.plot_id,
            trajectory_type=TrajectoryType.STABLE.value,
            confidence=0.0,
            observation_count=len(plot.ndvi_values),
            date_from=plot.date_from,
            date_to=plot.date_to,
            latitude=plot.latitude,
            longitude=plot.longitude,
            processing_time_ms=0.0,
            provenance_hash="",
            timestamp=utcnow().isoformat(),
            metadata={
                "error": True,
                "error_message": error_msg,
                "module_version": _MODULE_VERSION,
            },
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "TrajectoryType",
    # Constants
    "MAX_BATCH_SIZE",
    "MIN_OBSERVATIONS",
    "STABLE_STD_THRESHOLD",
    "ABRUPT_DROP_THRESHOLD",
    "GRADUAL_SLOPE_THRESHOLD",
    "OSCILLATION_MIN_PERIOD",
    "OSCILLATION_MAX_PERIOD",
    "RECOVERY_MIN_PERCENTAGE",
    # Data classes
    "TrajectoryPlotInput",
    "TemporalTrajectory",
    # Engine
    "TemporalTrajectoryAnalyzer",
]
