"""
GL-016 Waterguard Conductivity Divergence Detector

This module provides deterministic detection of divergence between measured
Cycles of Concentration (CoC) and predicted CoC from the thermal model.
Divergence indicates potential issues such as contamination, treatment
chemical overfeed, or sensor faults.

All calculations use statistical methods (NO generative AI) for zero-hallucination
compliance. Provenance tracking via SHA-256 hashes ensures complete audit trails.

Example:
    >>> config = DivergenceDetectorConfig(tolerance_percent=5.0)
    >>> detector = ConductivityDivergenceDetector(config)
    >>> event = detector.detect_coc_divergence(measured_coc=4.2, predicted_coc=3.8)
    >>> if event:
    ...     print(f"Divergence: {event.divergence_percent:.1f}%")

Author: GreenLang Waterguard Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import logging
import math

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class DivergenceCause(Enum):
    """Possible causes of CoC model divergence."""

    CONTAMINATION = "contamination"
    TREATMENT_OVERFEED = "treatment_overfeed"
    TREATMENT_UNDERFEED = "treatment_underfeed"
    SENSOR_FAULT = "sensor_fault"
    MAKEUP_WATER_CHANGE = "makeup_water_change"
    BLOWDOWN_CONTROL_ERROR = "blowdown_control_error"
    PROCESS_LEAK = "process_leak"
    UNKNOWN = "unknown"


class DivergenceSeverity(Enum):
    """Severity levels for divergence events."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DivergenceDirection(Enum):
    """Direction of divergence from predicted value."""

    HIGHER = "higher"
    LOWER = "lower"
    WITHIN_TOLERANCE = "within_tolerance"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DivergenceDetectorConfig:
    """
    Configuration for conductivity divergence detection.

    Attributes:
        tolerance_percent: Acceptable deviation before flagging (default 5%)
        warning_percent: Threshold for warning severity (default 10%)
        critical_percent: Threshold for critical severity (default 20%)
        min_sustained_minutes: Duration for sustained divergence (default 15)
        baseline_window_hours: Historical window for baseline (default 24)
        conductivity_sensor_accuracy: Expected sensor accuracy in uS/cm
        enable_cause_analysis: Whether to analyze likely causes
        enable_trend_detection: Whether to detect divergence trends
    """

    tolerance_percent: float = 5.0
    warning_percent: float = 10.0
    critical_percent: float = 20.0
    min_sustained_minutes: int = 15
    baseline_window_hours: int = 24
    conductivity_sensor_accuracy: float = 10.0  # uS/cm
    enable_cause_analysis: bool = True
    enable_trend_detection: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tolerance_percent <= 0:
            raise ValueError("tolerance_percent must be positive")
        if self.warning_percent <= self.tolerance_percent:
            raise ValueError("warning_percent must exceed tolerance_percent")
        if self.critical_percent <= self.warning_percent:
            raise ValueError("critical_percent must exceed warning_percent")
        if self.min_sustained_minutes < 1:
            raise ValueError("min_sustained_minutes must be at least 1")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConductivityReading:
    """
    A single conductivity measurement.

    Attributes:
        timestamp: When measurement was taken
        measured_conductivity: Actual sensor reading (uS/cm)
        predicted_conductivity: Model-predicted value (uS/cm)
        makeup_conductivity: Makeup water conductivity (uS/cm)
        blowdown_conductivity: Blowdown water conductivity (uS/cm)
        temperature: Water temperature for compensation (C)
    """

    timestamp: datetime
    measured_conductivity: float
    predicted_conductivity: float
    makeup_conductivity: Optional[float] = None
    blowdown_conductivity: Optional[float] = None
    temperature: Optional[float] = None


@dataclass
class ContextualData:
    """
    Contextual data for cause analysis.

    Attributes:
        treatment_dosing_rate: Current chemical dosing rate (L/hr)
        expected_dosing_rate: Expected dosing rate (L/hr)
        blowdown_valve_position: Valve position (0-100%)
        makeup_flow_rate: Makeup water flow (m3/hr)
        process_return_flow: Process return flow (m3/hr)
        recent_sensor_calibration: Time since last calibration
        sensor_drift_detected: Whether drift was detected
    """

    treatment_dosing_rate: Optional[float] = None
    expected_dosing_rate: Optional[float] = None
    blowdown_valve_position: Optional[float] = None
    makeup_flow_rate: Optional[float] = None
    process_return_flow: Optional[float] = None
    recent_sensor_calibration: Optional[datetime] = None
    sensor_drift_detected: bool = False


@dataclass
class DivergenceEvent:
    """
    A detected divergence between measured and predicted CoC.

    Attributes:
        timestamp: When divergence was detected
        measured_coc: Actual measured cycles of concentration
        predicted_coc: Model-predicted cycles of concentration
        divergence_percent: Percentage divergence from prediction
        direction: Whether measured is higher or lower than predicted
        severity: Event severity level
        likely_cause: Most probable cause of divergence
        cause_confidence: Confidence in cause determination (0-1)
        secondary_causes: Other possible causes with probabilities
        recommended_actions: Suggested corrective actions
        duration_minutes: How long divergence has persisted
        provenance_hash: SHA-256 hash for audit trail
    """

    timestamp: datetime
    measured_coc: float
    predicted_coc: float
    divergence_percent: float
    direction: DivergenceDirection
    severity: DivergenceSeverity
    likely_cause: DivergenceCause
    cause_confidence: float
    secondary_causes: Dict[DivergenceCause, float] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    duration_minutes: int = 0
    provenance_hash: str = ""

    def __post_init__(self):
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        data_str = (
            f"{self.timestamp.isoformat()}"
            f"{self.measured_coc:.6f}"
            f"{self.predicted_coc:.6f}"
            f"{self.divergence_percent:.6f}"
            f"{self.direction.value}"
            f"{self.severity.value}"
            f"{self.likely_cause.value}"
            f"{self.cause_confidence:.6f}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class DivergenceTrend:
    """
    Trend analysis for divergence over time.

    Attributes:
        start_time: When trend began
        end_time: Current time
        average_divergence: Mean divergence over period
        divergence_slope: Rate of change (%/hour)
        is_increasing: Whether divergence is growing
        trend_confidence: Statistical confidence in trend
    """

    start_time: datetime
    end_time: datetime
    average_divergence: float
    divergence_slope: float
    is_increasing: bool
    trend_confidence: float


# =============================================================================
# Main Detector Class
# =============================================================================

class ConductivityDivergenceDetector:
    """
    Detects divergence between measured and predicted Cycles of Concentration.

    This detector uses deterministic statistical methods to identify when
    measured conductivity-derived CoC values deviate significantly from
    thermal model predictions. All calculations are auditable with SHA-256
    provenance tracking.

    ZERO-HALLUCINATION COMPLIANCE:
    - All numeric calculations use Python arithmetic only
    - No generative AI models used for threshold decisions
    - Cause determination uses rule-based logic with confidence scores
    - Complete provenance tracking for regulatory audit

    Attributes:
        config: Detector configuration
        history: Recent divergence readings for trend analysis
        sustained_divergence_start: When current divergence began

    Example:
        >>> config = DivergenceDetectorConfig(tolerance_percent=5.0)
        >>> detector = ConductivityDivergenceDetector(config)
        >>> event = detector.detect_coc_divergence(4.2, 3.8)
        >>> if event:
        ...     print(f"Cause: {event.likely_cause.value}")
    """

    def __init__(self, config: Optional[DivergenceDetectorConfig] = None):
        """
        Initialize the conductivity divergence detector.

        Args:
            config: Detector configuration. Uses defaults if not provided.
        """
        self.config = config or DivergenceDetectorConfig()
        self.history: List[Tuple[datetime, float]] = []  # (timestamp, divergence_%)
        self.sustained_divergence_start: Optional[datetime] = None
        self._last_event: Optional[DivergenceEvent] = None

        logger.info(
            f"ConductivityDivergenceDetector initialized with "
            f"tolerance={self.config.tolerance_percent}%"
        )

    def detect_coc_divergence(
        self,
        measured_coc: float,
        predicted_coc: float,
        context: Optional[ContextualData] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[DivergenceEvent]:
        """
        Detect divergence between measured and predicted CoC.

        This is the primary detection method. It calculates the percentage
        divergence and determines severity, likely causes, and recommended
        actions based on configurable thresholds.

        Args:
            measured_coc: Actual measured cycles of concentration
            predicted_coc: Model-predicted cycles of concentration
            context: Optional contextual data for cause analysis
            timestamp: Event timestamp (defaults to now)

        Returns:
            DivergenceEvent if divergence exceeds tolerance, None otherwise

        Raises:
            ValueError: If CoC values are non-positive
        """
        if measured_coc <= 0 or predicted_coc <= 0:
            raise ValueError("CoC values must be positive")

        timestamp = timestamp or datetime.now()

        # Calculate divergence percentage
        divergence_percent = self._calculate_divergence_percent(
            measured_coc, predicted_coc
        )

        # Determine direction
        direction = self._determine_direction(measured_coc, predicted_coc)

        # Update history for trend analysis
        self._update_history(timestamp, divergence_percent)

        # Check if within tolerance
        if abs(divergence_percent) <= self.config.tolerance_percent:
            self.sustained_divergence_start = None
            logger.debug(
                f"CoC divergence {divergence_percent:.1f}% within tolerance "
                f"({self.config.tolerance_percent}%)"
            )
            return None

        # Track sustained divergence
        if self.sustained_divergence_start is None:
            self.sustained_divergence_start = timestamp

        duration_minutes = int(
            (timestamp - self.sustained_divergence_start).total_seconds() / 60
        )

        # Determine severity
        severity = self._determine_severity(abs(divergence_percent))

        # Analyze likely cause
        likely_cause, cause_confidence, secondary_causes = self._analyze_cause(
            measured_coc, predicted_coc, direction, context
        )

        # Generate recommended actions
        recommended_actions = self._generate_recommendations(
            likely_cause, severity, direction, context
        )

        # Create event
        event = DivergenceEvent(
            timestamp=timestamp,
            measured_coc=measured_coc,
            predicted_coc=predicted_coc,
            divergence_percent=divergence_percent,
            direction=direction,
            severity=severity,
            likely_cause=likely_cause,
            cause_confidence=cause_confidence,
            secondary_causes=secondary_causes,
            recommended_actions=recommended_actions,
            duration_minutes=duration_minutes
        )

        self._last_event = event

        logger.warning(
            f"CoC divergence detected: {divergence_percent:.1f}% "
            f"({severity.value}), likely cause: {likely_cause.value}"
        )

        return event

    def detect_from_conductivity(
        self,
        reading: ConductivityReading,
        context: Optional[ContextualData] = None
    ) -> Optional[DivergenceEvent]:
        """
        Detect divergence from raw conductivity readings.

        Converts conductivity values to CoC before detecting divergence.

        Args:
            reading: Conductivity measurement with measured and predicted values
            context: Optional contextual data for cause analysis

        Returns:
            DivergenceEvent if divergence exceeds tolerance, None otherwise
        """
        if reading.makeup_conductivity is None or reading.makeup_conductivity <= 0:
            logger.warning("Cannot calculate CoC without makeup water conductivity")
            return None

        # Calculate CoC from conductivity ratio
        measured_coc = reading.measured_conductivity / reading.makeup_conductivity
        predicted_coc = reading.predicted_conductivity / reading.makeup_conductivity

        return self.detect_coc_divergence(
            measured_coc, predicted_coc, context, reading.timestamp
        )

    def detect_sustained_divergence(
        self,
        readings: List[ConductivityReading],
        context: Optional[ContextualData] = None
    ) -> Optional[DivergenceEvent]:
        """
        Detect sustained divergence over multiple readings.

        A sustained divergence is confirmed when divergence exceeds tolerance
        for the configured minimum duration.

        Args:
            readings: List of conductivity readings over time
            context: Optional contextual data for cause analysis

        Returns:
            DivergenceEvent if sustained divergence detected, None otherwise
        """
        if len(readings) < 2:
            logger.debug("Insufficient readings for sustained divergence detection")
            return None

        # Sort by timestamp
        sorted_readings = sorted(readings, key=lambda r: r.timestamp)

        # Check each reading
        consecutive_divergence_start = None
        last_event = None

        for reading in sorted_readings:
            event = self.detect_from_conductivity(reading, context)

            if event:
                if consecutive_divergence_start is None:
                    consecutive_divergence_start = reading.timestamp
                last_event = event
            else:
                consecutive_divergence_start = None

        # Check if sustained long enough
        if consecutive_divergence_start and last_event:
            duration = (
                sorted_readings[-1].timestamp - consecutive_divergence_start
            ).total_seconds() / 60

            if duration >= self.config.min_sustained_minutes:
                last_event.duration_minutes = int(duration)
                logger.warning(
                    f"Sustained divergence detected for {duration:.0f} minutes"
                )
                return last_event

        return None

    def analyze_trend(
        self,
        window_hours: Optional[int] = None
    ) -> Optional[DivergenceTrend]:
        """
        Analyze divergence trend over time.

        Uses linear regression to determine if divergence is increasing
        or decreasing over the specified window.

        Args:
            window_hours: Analysis window in hours (defaults to config value)

        Returns:
            DivergenceTrend with slope and direction, or None if insufficient data
        """
        if not self.config.enable_trend_detection:
            return None

        window_hours = window_hours or self.config.baseline_window_hours

        if len(self.history) < 3:
            logger.debug("Insufficient history for trend analysis")
            return None

        # Filter to window
        cutoff = datetime.now()
        cutoff_start = datetime(
            cutoff.year, cutoff.month, cutoff.day,
            cutoff.hour - window_hours, cutoff.minute
        )

        window_data = [
            (ts, div) for ts, div in self.history
            if ts >= cutoff_start
        ]

        if len(window_data) < 3:
            return None

        # Calculate linear regression slope
        n = len(window_data)
        first_ts = window_data[0][0]

        # Convert timestamps to hours since first reading
        x_values = [
            (ts - first_ts).total_seconds() / 3600
            for ts, _ in window_data
        ]
        y_values = [div for _, div in window_data]

        # Calculate slope using least squares
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n

        numerator = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(x_values, y_values)
        )
        denominator = sum((x - mean_x) ** 2 for x in x_values)

        if denominator == 0:
            return None

        slope = numerator / denominator  # % per hour

        # Calculate R-squared for confidence
        y_pred = [mean_y + slope * (x - mean_x) for x in x_values]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred))
        ss_tot = sum((y - mean_y) ** 2 for y in y_values)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        trend = DivergenceTrend(
            start_time=window_data[0][0],
            end_time=window_data[-1][0],
            average_divergence=mean_y,
            divergence_slope=slope,
            is_increasing=slope > 0,
            trend_confidence=max(0, min(1, r_squared))
        )

        logger.info(
            f"Divergence trend: slope={slope:.2f}%/hr, "
            f"R-squared={r_squared:.3f}"
        )

        return trend

    def _calculate_divergence_percent(
        self,
        measured: float,
        predicted: float
    ) -> float:
        """
        Calculate percentage divergence.

        Positive values indicate measured > predicted.
        Negative values indicate measured < predicted.
        """
        return ((measured - predicted) / predicted) * 100.0

    def _determine_direction(
        self,
        measured: float,
        predicted: float
    ) -> DivergenceDirection:
        """Determine divergence direction."""
        divergence = measured - predicted
        tolerance_abs = predicted * (self.config.tolerance_percent / 100.0)

        if abs(divergence) <= tolerance_abs:
            return DivergenceDirection.WITHIN_TOLERANCE
        elif divergence > 0:
            return DivergenceDirection.HIGHER
        else:
            return DivergenceDirection.LOWER

    def _determine_severity(self, divergence_percent: float) -> DivergenceSeverity:
        """Determine severity based on divergence magnitude."""
        if divergence_percent >= self.config.critical_percent:
            return DivergenceSeverity.CRITICAL
        elif divergence_percent >= self.config.warning_percent:
            return DivergenceSeverity.WARNING
        else:
            return DivergenceSeverity.INFO

    def _analyze_cause(
        self,
        measured_coc: float,
        predicted_coc: float,
        direction: DivergenceDirection,
        context: Optional[ContextualData]
    ) -> Tuple[DivergenceCause, float, Dict[DivergenceCause, float]]:
        """
        Analyze likely cause of divergence using rule-based logic.

        Returns tuple of (primary_cause, confidence, secondary_causes).
        """
        if not self.config.enable_cause_analysis:
            return DivergenceCause.UNKNOWN, 0.5, {}

        cause_scores: Dict[DivergenceCause, float] = {}

        # Rule-based cause determination
        if direction == DivergenceDirection.HIGHER:
            # Measured CoC higher than predicted
            cause_scores[DivergenceCause.BLOWDOWN_CONTROL_ERROR] = 0.3
            cause_scores[DivergenceCause.CONTAMINATION] = 0.25
            cause_scores[DivergenceCause.MAKEUP_WATER_CHANGE] = 0.2
            cause_scores[DivergenceCause.SENSOR_FAULT] = 0.15

            # Adjust based on context
            if context:
                if context.sensor_drift_detected:
                    cause_scores[DivergenceCause.SENSOR_FAULT] += 0.4

                if context.blowdown_valve_position is not None:
                    if context.blowdown_valve_position < 10:
                        cause_scores[DivergenceCause.BLOWDOWN_CONTROL_ERROR] += 0.3

                if context.process_return_flow and context.process_return_flow > 0:
                    cause_scores[DivergenceCause.PROCESS_LEAK] = 0.35

        else:  # LOWER
            # Measured CoC lower than predicted
            cause_scores[DivergenceCause.TREATMENT_OVERFEED] = 0.3
            cause_scores[DivergenceCause.BLOWDOWN_CONTROL_ERROR] = 0.25
            cause_scores[DivergenceCause.SENSOR_FAULT] = 0.2

            # Adjust based on context
            if context:
                if context.sensor_drift_detected:
                    cause_scores[DivergenceCause.SENSOR_FAULT] += 0.4

                if (context.treatment_dosing_rate and
                    context.expected_dosing_rate and
                    context.treatment_dosing_rate > context.expected_dosing_rate * 1.2):
                    cause_scores[DivergenceCause.TREATMENT_OVERFEED] += 0.35

                if context.blowdown_valve_position is not None:
                    if context.blowdown_valve_position > 50:
                        cause_scores[DivergenceCause.BLOWDOWN_CONTROL_ERROR] += 0.3

        # Normalize scores
        total_score = sum(cause_scores.values())
        if total_score > 0:
            cause_scores = {k: v / total_score for k, v in cause_scores.items()}

        # Find primary cause
        if cause_scores:
            primary_cause = max(cause_scores, key=cause_scores.get)
            confidence = cause_scores[primary_cause]
            secondary_causes = {
                k: v for k, v in cause_scores.items()
                if k != primary_cause and v > 0.1
            }
        else:
            primary_cause = DivergenceCause.UNKNOWN
            confidence = 0.5
            secondary_causes = {}

        return primary_cause, confidence, secondary_causes

    def _generate_recommendations(
        self,
        cause: DivergenceCause,
        severity: DivergenceSeverity,
        direction: DivergenceDirection,
        context: Optional[ContextualData]
    ) -> List[str]:
        """Generate recommended actions based on cause analysis."""
        recommendations = []

        # Cause-specific recommendations
        if cause == DivergenceCause.SENSOR_FAULT:
            recommendations.append("Verify conductivity sensor calibration")
            recommendations.append("Check sensor probe for fouling or damage")
            recommendations.append("Compare with backup sensor if available")

        elif cause == DivergenceCause.BLOWDOWN_CONTROL_ERROR:
            recommendations.append("Check blowdown valve operation")
            recommendations.append("Verify blowdown controller setpoints")
            if direction == DivergenceDirection.HIGHER:
                recommendations.append("Increase blowdown rate if safe")
            else:
                recommendations.append("Decrease blowdown rate")

        elif cause == DivergenceCause.TREATMENT_OVERFEED:
            recommendations.append("Check chemical dosing pump calibration")
            recommendations.append("Verify dosing controller settings")
            recommendations.append("Check for dosing pump malfunction")

        elif cause == DivergenceCause.CONTAMINATION:
            recommendations.append("Check for process water ingress")
            recommendations.append("Inspect cooling tower basin for debris")
            recommendations.append("Sample and analyze makeup water")

        elif cause == DivergenceCause.MAKEUP_WATER_CHANGE:
            recommendations.append("Sample and analyze makeup water quality")
            recommendations.append("Check makeup water source")
            recommendations.append("Update model with new makeup water TDS")

        elif cause == DivergenceCause.PROCESS_LEAK:
            recommendations.append("Inspect heat exchangers for leaks")
            recommendations.append("Check process water return flows")
            recommendations.append("Perform leak detection test")

        # Severity-based recommendations
        if severity == DivergenceSeverity.CRITICAL:
            recommendations.insert(0, "URGENT: Investigate immediately")
            recommendations.append("Consider manual blowdown control")

        return recommendations

    def _update_history(self, timestamp: datetime, divergence: float) -> None:
        """Update history buffer for trend analysis."""
        self.history.append((timestamp, divergence))

        # Trim to window
        cutoff = datetime.now()
        window_seconds = self.config.baseline_window_hours * 3600
        self.history = [
            (ts, div) for ts, div in self.history
            if (cutoff - ts).total_seconds() < window_seconds
        ]

    def get_last_event(self) -> Optional[DivergenceEvent]:
        """Get the most recent divergence event."""
        return self._last_event

    def reset(self) -> None:
        """Reset detector state."""
        self.history.clear()
        self.sustained_divergence_start = None
        self._last_event = None
        logger.info("ConductivityDivergenceDetector reset")


# =============================================================================
# Factory Function
# =============================================================================

def create_divergence_detector(
    tolerance_percent: float = 5.0,
    warning_percent: float = 10.0,
    critical_percent: float = 20.0,
    **kwargs
) -> ConductivityDivergenceDetector:
    """
    Factory function to create a configured divergence detector.

    Args:
        tolerance_percent: Acceptable deviation before flagging
        warning_percent: Threshold for warning severity
        critical_percent: Threshold for critical severity
        **kwargs: Additional configuration options

    Returns:
        Configured ConductivityDivergenceDetector instance
    """
    config = DivergenceDetectorConfig(
        tolerance_percent=tolerance_percent,
        warning_percent=warning_percent,
        critical_percent=critical_percent,
        **kwargs
    )
    return ConductivityDivergenceDetector(config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "ConductivityDivergenceDetector",
    # Configuration
    "DivergenceDetectorConfig",
    # Data classes
    "ConductivityReading",
    "ContextualData",
    "DivergenceEvent",
    "DivergenceTrend",
    # Enums
    "DivergenceCause",
    "DivergenceSeverity",
    "DivergenceDirection",
    # Factory
    "create_divergence_detector",
]
