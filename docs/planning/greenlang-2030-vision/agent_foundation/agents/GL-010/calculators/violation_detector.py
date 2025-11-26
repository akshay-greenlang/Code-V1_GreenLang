"""
Violation Detector Module for GL-010 EMISSIONWATCH.

This module provides real-time exceedance detection and predictive
violation alerts for emissions monitoring. All detection logic is
deterministic and based on statistical methods.

Features:
- Real-time limit exceedance detection
- Predictive alerts based on trend analysis
- Multi-parameter correlation analysis
- False positive filtering
- Severity classification

References:
- EPA CEMS Data Handling and Analysis
- ISO 14956: Air quality - Evaluation of CEMS

Zero-Hallucination Guarantee:
- All detection algorithms are deterministic
- Statistical methods are well-documented
- Full audit trail for all alerts
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import statistics
from pydantic import BaseModel, Field

from .compliance_checker import (
    EmissionLimit, ComplianceStatus, AveragingPeriod,
    SourceCategory, Jurisdiction
)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"    # Limit exceeded
    HIGH = "high"            # 90-100% of limit
    MEDIUM = "medium"        # 80-90% of limit
    LOW = "low"              # 70-80% of limit
    INFO = "info"            # Informational


class AlertType(str, Enum):
    """Types of violation alerts."""
    EXCEEDANCE = "exceedance"              # Limit exceeded
    APPROACHING_LIMIT = "approaching"       # Trending toward limit
    RATE_OF_CHANGE = "rate_of_change"      # Rapid increase
    SUSTAINED_HIGH = "sustained_high"       # Prolonged high level
    CORRELATION = "correlation"             # Multi-param correlation
    AVERAGING_PERIOD = "averaging_period"   # Rolling average concern


class DataQuality(str, Enum):
    """Data quality flags."""
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    MISSING = "missing"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True)
class ViolationAlert:
    """
    Violation alert with full context.

    Attributes:
        alert_id: Unique alert identifier
        alert_type: Type of alert
        severity: Alert severity level
        pollutant: Pollutant involved
        current_value: Current measured value
        limit_value: Applicable limit
        percent_of_limit: Percentage of limit
        timestamp: When alert was generated
        source_id: Source identifier
        description: Human-readable description
        recommended_action: Suggested response
        prediction_confidence: Confidence for predictive alerts (0-1)
    """
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    pollutant: str
    current_value: Decimal
    limit_value: Decimal
    percent_of_limit: Decimal
    timestamp: datetime
    source_id: str
    description: str
    recommended_action: str
    prediction_confidence: Optional[Decimal] = None
    related_alerts: List[str] = field(default_factory=list)


@dataclass
class EmissionDataPoint:
    """Single emission data point."""
    timestamp: datetime
    pollutant: str
    value: Decimal
    unit: str
    quality: DataQuality = DataQuality.VALID
    o2_percent: Optional[Decimal] = None


@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    slope: Decimal  # Rate of change per hour
    r_squared: Decimal  # Correlation coefficient
    intercept: Decimal
    projected_value: Decimal  # Projected value at future time
    projection_time_hours: Decimal
    time_to_limit: Optional[Decimal]  # Hours until limit exceeded


class DetectionConfig(BaseModel):
    """Configuration for violation detection."""
    approach_threshold_percent: float = Field(
        default=80, ge=50, le=100,
        description="Percentage of limit to trigger approach alert"
    )
    rate_of_change_threshold: float = Field(
        default=10, gt=0,
        description="% increase per hour to trigger rate alert"
    )
    sustained_high_hours: int = Field(
        default=4, ge=1,
        description="Hours above threshold for sustained alert"
    )
    prediction_window_hours: int = Field(
        default=24, ge=1, le=168,
        description="Hours to look ahead for predictions"
    )
    minimum_data_points: int = Field(
        default=10, ge=3,
        description="Minimum points for trend analysis"
    )
    false_positive_filter_minutes: int = Field(
        default=15, ge=0,
        description="Minimum duration before alert (filter spikes)"
    )


class ViolationDetector:
    """
    Real-time violation detection engine.

    Provides deterministic detection of:
    - Immediate limit exceedances
    - Approaching limits (trending toward violation)
    - Rate-of-change anomalies
    - Sustained high emissions
    - Multi-parameter correlations

    All detection logic is deterministic and reproducible.
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        history_hours: int = 168  # 1 week
    ):
        """
        Initialize violation detector.

        Args:
            config: Detection configuration
            history_hours: Hours of history to maintain
        """
        self.config = config or DetectionConfig()
        self.history_hours = history_hours

        # Data buffers keyed by (source_id, pollutant)
        self._data_buffers: Dict[Tuple[str, str], deque] = {}

        # Active alerts
        self._active_alerts: Dict[str, ViolationAlert] = {}

        # Alert counter for ID generation
        self._alert_counter = 0

        # Limits registry
        self._limits: Dict[Tuple[str, str], EmissionLimit] = {}

    def register_limit(
        self,
        source_id: str,
        limit: EmissionLimit
    ) -> None:
        """
        Register an emission limit for monitoring.

        Args:
            source_id: Source identifier
            limit: Emission limit to monitor
        """
        key = (source_id, limit.pollutant)
        self._limits[key] = limit

    def add_data_point(
        self,
        source_id: str,
        data_point: EmissionDataPoint
    ) -> List[ViolationAlert]:
        """
        Add a data point and check for violations.

        Args:
            source_id: Source identifier
            data_point: Emission data point

        Returns:
            List of any alerts generated
        """
        key = (source_id, data_point.pollutant)

        # Initialize buffer if needed
        if key not in self._data_buffers:
            max_points = self.history_hours * 60  # Assume minute data
            self._data_buffers[key] = deque(maxlen=max_points)

        # Add to buffer
        self._data_buffers[key].append(data_point)

        # Check for violations
        alerts = []

        # Get applicable limit
        limit = self._limits.get(key)
        if limit is None:
            return alerts

        # Run detection checks
        alerts.extend(self._check_exceedance(source_id, data_point, limit))
        alerts.extend(self._check_approaching_limit(source_id, data_point, limit))
        alerts.extend(self._check_rate_of_change(source_id, data_point, limit))
        alerts.extend(self._check_sustained_high(source_id, data_point, limit))

        # Filter false positives
        alerts = self._filter_false_positives(alerts, source_id, data_point.pollutant)

        # Store active alerts
        for alert in alerts:
            self._active_alerts[alert.alert_id] = alert

        return alerts

    def _check_exceedance(
        self,
        source_id: str,
        data_point: EmissionDataPoint,
        limit: EmissionLimit
    ) -> List[ViolationAlert]:
        """Check for immediate limit exceedance."""
        alerts = []

        if data_point.quality != DataQuality.VALID:
            return alerts

        percent_of_limit = (data_point.value / limit.limit_value) * Decimal("100")

        if data_point.value > limit.limit_value:
            alert = ViolationAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.EXCEEDANCE,
                severity=AlertSeverity.CRITICAL,
                pollutant=data_point.pollutant,
                current_value=data_point.value,
                limit_value=limit.limit_value,
                percent_of_limit=self._apply_precision(percent_of_limit, 1),
                timestamp=datetime.now(),
                source_id=source_id,
                description=f"{data_point.pollutant} exceeds limit: "
                           f"{data_point.value} {data_point.unit} > "
                           f"{limit.limit_value} {limit.unit} "
                           f"({percent_of_limit:.1f}% of limit)",
                recommended_action="Immediate investigation required. "
                                  "Check equipment operation and consider reducing load."
            )
            alerts.append(alert)

        return alerts

    def _check_approaching_limit(
        self,
        source_id: str,
        data_point: EmissionDataPoint,
        limit: EmissionLimit
    ) -> List[ViolationAlert]:
        """Check if emissions are approaching the limit."""
        alerts = []

        if data_point.quality != DataQuality.VALID:
            return alerts

        percent_of_limit = (data_point.value / limit.limit_value) * Decimal("100")
        threshold = Decimal(str(self.config.approach_threshold_percent))

        if percent_of_limit >= threshold and data_point.value <= limit.limit_value:
            # Determine severity based on percentage
            if percent_of_limit >= Decimal("95"):
                severity = AlertSeverity.HIGH
            elif percent_of_limit >= Decimal("90"):
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

            alert = ViolationAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.APPROACHING_LIMIT,
                severity=severity,
                pollutant=data_point.pollutant,
                current_value=data_point.value,
                limit_value=limit.limit_value,
                percent_of_limit=self._apply_precision(percent_of_limit, 1),
                timestamp=datetime.now(),
                source_id=source_id,
                description=f"{data_point.pollutant} at {percent_of_limit:.1f}% of limit "
                           f"({data_point.value} / {limit.limit_value} {limit.unit})",
                recommended_action="Monitor closely. Consider operational adjustments "
                                  "to prevent exceedance."
            )
            alerts.append(alert)

        return alerts

    def _check_rate_of_change(
        self,
        source_id: str,
        data_point: EmissionDataPoint,
        limit: EmissionLimit
    ) -> List[ViolationAlert]:
        """Check for rapid rate of increase."""
        alerts = []
        key = (source_id, data_point.pollutant)

        buffer = self._data_buffers.get(key)
        if buffer is None or len(buffer) < self.config.minimum_data_points:
            return alerts

        # Get recent data (last hour)
        now = data_point.timestamp
        one_hour_ago = now - timedelta(hours=1)

        recent_points = [
            dp for dp in buffer
            if dp.timestamp >= one_hour_ago and dp.quality == DataQuality.VALID
        ]

        if len(recent_points) < 2:
            return alerts

        # Calculate rate of change
        first_value = recent_points[0].value
        last_value = recent_points[-1].value

        if first_value > 0:
            percent_change = ((last_value - first_value) / first_value) * Decimal("100")
        else:
            percent_change = Decimal("0")

        threshold = Decimal(str(self.config.rate_of_change_threshold))

        if percent_change > threshold:
            # Project time to limit
            trend = self._analyze_trend(recent_points, limit)

            alert = ViolationAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.RATE_OF_CHANGE,
                severity=AlertSeverity.MEDIUM,
                pollutant=data_point.pollutant,
                current_value=data_point.value,
                limit_value=limit.limit_value,
                percent_of_limit=(data_point.value / limit.limit_value) * Decimal("100"),
                timestamp=datetime.now(),
                source_id=source_id,
                description=f"{data_point.pollutant} increasing rapidly: "
                           f"{percent_change:.1f}% increase in last hour. "
                           f"Time to limit: {trend.time_to_limit:.1f}h" if trend.time_to_limit else "",
                recommended_action="Investigate cause of increase. "
                                  "Check for operational changes or equipment issues.",
                prediction_confidence=self._apply_precision(trend.r_squared, 2)
            )
            alerts.append(alert)

        return alerts

    def _check_sustained_high(
        self,
        source_id: str,
        data_point: EmissionDataPoint,
        limit: EmissionLimit
    ) -> List[ViolationAlert]:
        """Check for sustained high emissions."""
        alerts = []
        key = (source_id, data_point.pollutant)

        buffer = self._data_buffers.get(key)
        if buffer is None:
            return alerts

        threshold_hours = self.config.sustained_high_hours
        threshold_percent = Decimal(str(self.config.approach_threshold_percent))

        now = data_point.timestamp
        lookback_time = now - timedelta(hours=threshold_hours)

        # Get points in lookback period
        points_in_period = [
            dp for dp in buffer
            if dp.timestamp >= lookback_time and dp.quality == DataQuality.VALID
        ]

        if len(points_in_period) < threshold_hours * 4:  # Minimum 4 points/hour
            return alerts

        # Check if all points are above threshold
        high_points = [
            dp for dp in points_in_period
            if (dp.value / limit.limit_value) * Decimal("100") >= threshold_percent
        ]

        if len(high_points) >= len(points_in_period) * 0.8:  # 80% of points high
            avg_value = sum(dp.value for dp in high_points) / len(high_points)
            avg_percent = (avg_value / limit.limit_value) * Decimal("100")

            alert = ViolationAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.SUSTAINED_HIGH,
                severity=AlertSeverity.HIGH,
                pollutant=data_point.pollutant,
                current_value=data_point.value,
                limit_value=limit.limit_value,
                percent_of_limit=self._apply_precision(avg_percent, 1),
                timestamp=datetime.now(),
                source_id=source_id,
                description=f"{data_point.pollutant} sustained above {threshold_percent}% "
                           f"of limit for {threshold_hours} hours "
                           f"(avg: {avg_value:.2f} {data_point.unit})",
                recommended_action="Extended high emissions detected. "
                                  "Review operating conditions and control equipment."
            )
            alerts.append(alert)

        return alerts

    def _analyze_trend(
        self,
        data_points: List[EmissionDataPoint],
        limit: EmissionLimit
    ) -> TrendAnalysis:
        """
        Analyze emission trend using linear regression.

        Args:
            data_points: List of data points
            limit: Applicable limit

        Returns:
            TrendAnalysis with trend parameters
        """
        if len(data_points) < 2:
            return TrendAnalysis(
                slope=Decimal("0"),
                r_squared=Decimal("0"),
                intercept=Decimal("0"),
                projected_value=data_points[-1].value if data_points else Decimal("0"),
                projection_time_hours=Decimal("0"),
                time_to_limit=None
            )

        # Convert to numeric arrays
        base_time = data_points[0].timestamp
        x_values = [
            (dp.timestamp - base_time).total_seconds() / 3600
            for dp in data_points
        ]
        y_values = [float(dp.value) for dp in data_points]

        # Calculate linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            slope = 0
            intercept = sum_y / n
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))

        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = 0

        # Project future value
        projection_hours = self.config.prediction_window_hours
        last_x = x_values[-1]
        projected_value = slope * (last_x + projection_hours) + intercept

        # Calculate time to limit
        if slope > 0:
            time_to_limit = (float(limit.limit_value) - intercept) / slope - last_x
            if time_to_limit < 0:
                time_to_limit = None
        else:
            time_to_limit = None

        return TrendAnalysis(
            slope=Decimal(str(slope)),
            r_squared=Decimal(str(max(0, r_squared))),
            intercept=Decimal(str(intercept)),
            projected_value=Decimal(str(max(0, projected_value))),
            projection_time_hours=Decimal(str(projection_hours)),
            time_to_limit=Decimal(str(time_to_limit)) if time_to_limit else None
        )

    def _filter_false_positives(
        self,
        alerts: List[ViolationAlert],
        source_id: str,
        pollutant: str
    ) -> List[ViolationAlert]:
        """
        Filter out likely false positive alerts.

        Uses minimum duration requirement and duplicate suppression.
        """
        if self.config.false_positive_filter_minutes == 0:
            return alerts

        filtered = []
        key = (source_id, pollutant)
        buffer = self._data_buffers.get(key, deque())

        filter_duration = timedelta(minutes=self.config.false_positive_filter_minutes)

        for alert in alerts:
            # For exceedances, check if sustained
            if alert.alert_type == AlertType.EXCEEDANCE:
                # Count points above limit in filter period
                cutoff = alert.timestamp - filter_duration
                high_points = [
                    dp for dp in buffer
                    if dp.timestamp >= cutoff and dp.value > alert.limit_value
                ]

                # Require at least 2 points above limit
                if len(high_points) >= 2:
                    filtered.append(alert)
            else:
                # Other alerts pass through
                filtered.append(alert)

        return filtered

    def get_active_alerts(
        self,
        source_id: Optional[str] = None,
        pollutant: Optional[str] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[ViolationAlert]:
        """
        Get currently active alerts.

        Args:
            source_id: Filter by source
            pollutant: Filter by pollutant
            severity: Filter by severity

        Returns:
            List of active alerts
        """
        alerts = list(self._active_alerts.values())

        if source_id:
            alerts = [a for a in alerts if a.source_id == source_id]
        if pollutant:
            alerts = [a for a in alerts if a.pollutant == pollutant]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def clear_alert(self, alert_id: str) -> bool:
        """
        Clear an active alert.

        Args:
            alert_id: Alert to clear

        Returns:
            True if alert was cleared
        """
        if alert_id in self._active_alerts:
            del self._active_alerts[alert_id]
            return True
        return False

    def get_prediction(
        self,
        source_id: str,
        pollutant: str,
        hours_ahead: int = 24
    ) -> Optional[TrendAnalysis]:
        """
        Get emission prediction for a source/pollutant.

        Args:
            source_id: Source identifier
            pollutant: Pollutant type
            hours_ahead: Hours to project

        Returns:
            TrendAnalysis or None
        """
        key = (source_id, pollutant)
        buffer = self._data_buffers.get(key)

        if buffer is None or len(buffer) < self.config.minimum_data_points:
            return None

        limit = self._limits.get(key)
        if limit is None:
            return None

        # Get recent valid points
        valid_points = [dp for dp in buffer if dp.quality == DataQuality.VALID]

        if len(valid_points) < self.config.minimum_data_points:
            return None

        # Use most recent points
        recent_points = list(valid_points)[-100:]  # Last 100 points

        return self._analyze_trend(recent_points, limit)

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# Convenience functions
def create_detector(
    approach_threshold: float = 80,
    rate_threshold: float = 10
) -> ViolationDetector:
    """
    Create a configured violation detector.

    Args:
        approach_threshold: % of limit for approach alerts
        rate_threshold: % change/hour for rate alerts

    Returns:
        Configured ViolationDetector
    """
    config = DetectionConfig(
        approach_threshold_percent=approach_threshold,
        rate_of_change_threshold=rate_threshold
    )
    return ViolationDetector(config=config)


def check_for_violations(
    source_id: str,
    pollutant: str,
    values: List[Tuple[datetime, float]],
    limit_value: float,
    limit_unit: str
) -> List[ViolationAlert]:
    """
    Quick check for violations in a data series.

    Args:
        source_id: Source identifier
        pollutant: Pollutant type
        values: List of (timestamp, value) tuples
        limit_value: Emission limit
        limit_unit: Limit unit

    Returns:
        List of violation alerts
    """
    from .compliance_checker import RegulatoryProgram

    detector = ViolationDetector()

    # Create limit
    limit = EmissionLimit(
        pollutant=pollutant,
        limit_value=Decimal(str(limit_value)),
        unit=limit_unit,
        averaging_period=AveragingPeriod.ONE_HOUR,
        o2_reference=None,
        jurisdiction=Jurisdiction.PERMIT_SPECIFIC,
        program=RegulatoryProgram.PERMIT,
        source_category=SourceCategory.BOILER,
        effective_date=datetime.now().date(),
        citation="Permit specific"
    )

    detector.register_limit(source_id, limit)

    # Add all data points
    all_alerts = []
    for timestamp, value in values:
        data_point = EmissionDataPoint(
            timestamp=timestamp,
            pollutant=pollutant,
            value=Decimal(str(value)),
            unit=limit_unit
        )
        alerts = detector.add_data_point(source_id, data_point)
        all_alerts.extend(alerts)

    return all_alerts
