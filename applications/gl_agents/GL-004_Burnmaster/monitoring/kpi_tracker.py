"""
GL-004 BURNMASTER KPI Tracker Module

This module provides Key Performance Indicator (KPI) tracking for combustion
optimization operations, including fuel intensity, emissions tracking,
stability scores, and turndown achievement monitoring.

Example:
    >>> tracker = KPITracker()
    >>> tracker.track_fuel_intensity(fuel=1000.0, duty=5000.0)
    >>> tracker.track_emissions(nox=25.0, co=15.0)
    >>> report = tracker.generate_kpi_report(period)
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import statistics
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TrendDirection(str, Enum):
    """Direction of KPI trend."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class KPIStatus(str, Enum):
    """KPI achievement status."""
    EXCEEDING = "EXCEEDING"
    ON_TARGET = "ON_TARGET"
    BELOW_TARGET = "BELOW_TARGET"
    CRITICAL = "CRITICAL"


class KPICategory(str, Enum):
    """Category of KPI."""
    EFFICIENCY = "EFFICIENCY"
    EMISSIONS = "EMISSIONS"
    RELIABILITY = "RELIABILITY"
    SAFETY = "SAFETY"
    OPERATIONAL = "OPERATIONAL"


# =============================================================================
# DATA MODELS
# =============================================================================

class DateRange(BaseModel):
    """Date range for KPI queries."""

    start: datetime = Field(..., description="Start of date range")
    end: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="End of date range"
    )


class KPIDataPoint(BaseModel):
    """A single KPI data point."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Point timestamp"
    )
    value: float = Field(..., description="KPI value")
    unit_id: Optional[str] = Field(None, description="Unit identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional data")


class KPITarget(BaseModel):
    """Target values for a KPI."""

    kpi_name: str = Field(..., description="KPI name")
    target_value: float = Field(..., description="Target value")
    warning_threshold: Optional[float] = Field(
        None, description="Warning threshold"
    )
    critical_threshold: Optional[float] = Field(
        None, description="Critical threshold"
    )
    direction: str = Field(
        default="lower_is_better",
        description="lower_is_better or higher_is_better"
    )


class TrendAnalysis(BaseModel):
    """Analysis of KPI trend over time."""

    kpi_name: str = Field(..., description="KPI name")
    period: DateRange = Field(..., description="Analysis period")

    # Trend direction
    direction: TrendDirection = Field(..., description="Trend direction")
    change_percent: float = Field(..., description="Percentage change over period")

    # Statistics
    start_value: Optional[float] = Field(None, description="Value at start")
    end_value: Optional[float] = Field(None, description="Value at end")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    avg_value: Optional[float] = Field(None, description="Average value")
    std_dev: Optional[float] = Field(None, description="Standard deviation")

    # Regression
    slope: Optional[float] = Field(None, description="Linear regression slope")
    r_squared: Optional[float] = Field(None, description="R-squared coefficient")

    # Points
    data_points: int = Field(default=0, description="Number of data points")

    # Confidence
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in trend"
    )


class KPISummary(BaseModel):
    """Summary of a single KPI."""

    kpi_name: str = Field(..., description="KPI name")
    category: KPICategory = Field(..., description="KPI category")
    description: str = Field(default="", description="KPI description")
    unit: str = Field(default="", description="Unit of measurement")

    # Current state
    current_value: float = Field(..., description="Current value")
    previous_value: Optional[float] = Field(None, description="Previous period value")
    target_value: Optional[float] = Field(None, description="Target value")

    # Status
    status: KPIStatus = Field(..., description="KPI status")
    trend: TrendDirection = Field(..., description="Trend direction")
    change_percent: float = Field(default=0.0, description="Change from previous")

    # Period statistics
    period_min: Optional[float] = Field(None, description="Period minimum")
    period_max: Optional[float] = Field(None, description="Period maximum")
    period_avg: Optional[float] = Field(None, description="Period average")

    # Metadata
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time"
    )


class KPIReport(BaseModel):
    """Comprehensive KPI report for a period."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation time"
    )
    period: DateRange = Field(..., description="Report period")

    # Overall performance
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall KPI score"
    )
    overall_status: KPIStatus = Field(..., description="Overall status")
    overall_trend: TrendDirection = Field(..., description="Overall trend")

    # KPI summaries by category
    efficiency_kpis: List[KPISummary] = Field(
        default_factory=list, description="Efficiency KPIs"
    )
    emissions_kpis: List[KPISummary] = Field(
        default_factory=list, description="Emissions KPIs"
    )
    reliability_kpis: List[KPISummary] = Field(
        default_factory=list, description="Reliability KPIs"
    )
    safety_kpis: List[KPISummary] = Field(
        default_factory=list, description="Safety KPIs"
    )
    operational_kpis: List[KPISummary] = Field(
        default_factory=list, description="Operational KPIs"
    )

    # Trend analyses
    trend_analyses: List[TrendAnalysis] = Field(
        default_factory=list, description="Detailed trend analyses"
    )

    # Key highlights
    exceeding_targets: List[str] = Field(
        default_factory=list, description="KPIs exceeding targets"
    )
    below_targets: List[str] = Field(
        default_factory=list, description="KPIs below targets"
    )
    critical_kpis: List[str] = Field(
        default_factory=list, description="KPIs in critical state"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    # Audit
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


# =============================================================================
# KPI TRACKER
# =============================================================================

class KPITracker:
    """
    Comprehensive KPI tracking for combustion optimization.

    Tracks key performance indicators including fuel intensity, emissions,
    stability scores, and turndown achievement. Provides trend analysis
    and reporting capabilities.

    Attributes:
        unit_id: Default unit identifier
        targets: KPI targets dictionary

    Example:
        >>> tracker = KPITracker(unit_id="BNR-001")
        >>> tracker.track_fuel_intensity(fuel=1000.0, duty=5000.0)
        >>> report = tracker.generate_kpi_report(period)
    """

    # Standard KPI definitions
    KPI_DEFINITIONS = {
        'fuel_intensity': {
            'category': KPICategory.EFFICIENCY,
            'unit': 'MMBtu/unit_duty',
            'description': 'Fuel consumption per unit of thermal duty',
            'direction': 'lower_is_better',
        },
        'nox_emissions': {
            'category': KPICategory.EMISSIONS,
            'unit': 'ppm',
            'description': 'NOx emissions concentration',
            'direction': 'lower_is_better',
        },
        'co_emissions': {
            'category': KPICategory.EMISSIONS,
            'unit': 'ppm',
            'description': 'CO emissions concentration',
            'direction': 'lower_is_better',
        },
        'stability_score': {
            'category': KPICategory.RELIABILITY,
            'unit': 'score (0-1)',
            'description': 'Flame stability score',
            'direction': 'higher_is_better',
        },
        'turndown_ratio': {
            'category': KPICategory.OPERATIONAL,
            'unit': 'ratio',
            'description': 'Achieved turndown ratio',
            'direction': 'higher_is_better',
        },
        'combustion_efficiency': {
            'category': KPICategory.EFFICIENCY,
            'unit': '%',
            'description': 'Combustion efficiency percentage',
            'direction': 'higher_is_better',
        },
        'availability': {
            'category': KPICategory.RELIABILITY,
            'unit': '%',
            'description': 'Burner availability percentage',
            'direction': 'higher_is_better',
        },
        'safety_incidents': {
            'category': KPICategory.SAFETY,
            'unit': 'count',
            'description': 'Safety incidents count',
            'direction': 'lower_is_better',
        },
    }

    def __init__(self, unit_id: Optional[str] = None):
        """
        Initialize the KPITracker.

        Args:
            unit_id: Default unit identifier for tracking
        """
        self.unit_id = unit_id
        self._kpi_data: Dict[str, List[KPIDataPoint]] = {}
        self._targets: Dict[str, KPITarget] = {}

        # Initialize data storage for all defined KPIs
        for kpi_name in self.KPI_DEFINITIONS:
            self._kpi_data[kpi_name] = []

        logger.info(f"KPITracker initialized for unit: {unit_id}")

    def track_fuel_intensity(
        self,
        fuel: float,
        duty: float,
        unit_id: Optional[str] = None
    ) -> None:
        """
        Track fuel intensity (fuel consumption per unit duty).

        Args:
            fuel: Fuel consumption (MMBtu or similar)
            duty: Thermal duty output
            unit_id: Optional unit identifier
        """
        if duty <= 0:
            logger.warning("Cannot calculate fuel intensity with zero duty")
            return

        intensity = fuel / duty
        self._record_kpi('fuel_intensity', intensity, unit_id)
        logger.debug(f"Recorded fuel intensity: {intensity:.4f}")

    def track_emissions(
        self,
        nox: float,
        co: float,
        unit_id: Optional[str] = None
    ) -> None:
        """
        Track emissions (NOx and CO).

        Args:
            nox: NOx concentration in ppm
            co: CO concentration in ppm
            unit_id: Optional unit identifier
        """
        self._record_kpi('nox_emissions', nox, unit_id)
        self._record_kpi('co_emissions', co, unit_id)
        logger.debug(f"Recorded emissions: NOx={nox}ppm, CO={co}ppm")

    def track_stability_score(
        self,
        score: float,
        unit_id: Optional[str] = None
    ) -> None:
        """
        Track flame stability score.

        Args:
            score: Stability score (0-1)
            unit_id: Optional unit identifier
        """
        score = max(0.0, min(1.0, score))  # Clamp to 0-1
        self._record_kpi('stability_score', score, unit_id)
        logger.debug(f"Recorded stability score: {score:.3f}")

    def track_turndown_achieved(
        self,
        turndown_ratio: float,
        unit_id: Optional[str] = None
    ) -> None:
        """
        Track achieved turndown ratio.

        Args:
            turndown_ratio: Achieved turndown ratio (e.g., 10:1 = 10.0)
            unit_id: Optional unit identifier
        """
        self._record_kpi('turndown_ratio', turndown_ratio, unit_id)
        logger.debug(f"Recorded turndown ratio: {turndown_ratio}:1")

    def track_combustion_efficiency(
        self,
        efficiency: float,
        unit_id: Optional[str] = None
    ) -> None:
        """
        Track combustion efficiency.

        Args:
            efficiency: Efficiency percentage (0-100)
            unit_id: Optional unit identifier
        """
        efficiency = max(0.0, min(100.0, efficiency))
        self._record_kpi('combustion_efficiency', efficiency, unit_id)
        logger.debug(f"Recorded combustion efficiency: {efficiency:.1f}%")

    def track_availability(
        self,
        availability: float,
        unit_id: Optional[str] = None
    ) -> None:
        """
        Track burner availability.

        Args:
            availability: Availability percentage (0-100)
            unit_id: Optional unit identifier
        """
        availability = max(0.0, min(100.0, availability))
        self._record_kpi('availability', availability, unit_id)
        logger.debug(f"Recorded availability: {availability:.1f}%")

    def track_safety_incident(
        self,
        incident_type: str,
        unit_id: Optional[str] = None
    ) -> None:
        """
        Track a safety incident.

        Args:
            incident_type: Type of incident
            unit_id: Optional unit identifier
        """
        # For safety incidents, we track cumulative count
        current = self._get_latest_value('safety_incidents') or 0
        self._record_kpi(
            'safety_incidents',
            current + 1,
            unit_id,
            metadata={'incident_type': incident_type}
        )
        logger.warning(f"Recorded safety incident: {incident_type}")

    def _record_kpi(
        self,
        kpi_name: str,
        value: float,
        unit_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a KPI data point."""
        if kpi_name not in self._kpi_data:
            self._kpi_data[kpi_name] = []

        point = KPIDataPoint(
            value=value,
            unit_id=unit_id or self.unit_id,
            metadata=metadata or {}
        )
        self._kpi_data[kpi_name].append(point)

        # Limit history size
        if len(self._kpi_data[kpi_name]) > 10000:
            self._kpi_data[kpi_name] = self._kpi_data[kpi_name][-10000:]

    def _get_latest_value(self, kpi_name: str) -> Optional[float]:
        """Get the most recent value for a KPI."""
        if kpi_name in self._kpi_data and self._kpi_data[kpi_name]:
            return self._kpi_data[kpi_name][-1].value
        return None

    def set_target(
        self,
        kpi_name: str,
        target: float,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None
    ) -> None:
        """
        Set target values for a KPI.

        Args:
            kpi_name: KPI name
            target: Target value
            warning_threshold: Warning threshold
            critical_threshold: Critical threshold
        """
        definition = self.KPI_DEFINITIONS.get(kpi_name, {})
        direction = definition.get('direction', 'lower_is_better')

        self._targets[kpi_name] = KPITarget(
            kpi_name=kpi_name,
            target_value=target,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            direction=direction
        )
        logger.info(f"Set target for {kpi_name}: {target}")

    def compute_kpi_trends(
        self,
        kpi: str,
        period: DateRange
    ) -> TrendAnalysis:
        """
        Compute trend analysis for a KPI over a period.

        Args:
            kpi: KPI name
            period: Date range for analysis

        Returns:
            TrendAnalysis with trend details
        """
        # Filter data points within period
        points = [
            p for p in self._kpi_data.get(kpi, [])
            if period.start <= p.timestamp <= period.end
        ]

        if len(points) < 2:
            return TrendAnalysis(
                kpi_name=kpi,
                period=period,
                direction=TrendDirection.INSUFFICIENT_DATA,
                change_percent=0.0,
                data_points=len(points),
                confidence=0.0
            )

        # Sort by timestamp
        points = sorted(points, key=lambda p: p.timestamp)
        values = [p.value for p in points]

        # Calculate basic statistics
        start_value = values[0]
        end_value = values[-1]
        min_value = min(values)
        max_value = max(values)
        avg_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0

        # Calculate percentage change
        if start_value != 0:
            change_percent = ((end_value - start_value) / abs(start_value)) * 100
        else:
            change_percent = 0.0 if end_value == 0 else 100.0

        # Calculate linear regression for trend
        slope, r_squared = self._linear_regression(points)

        # Determine trend direction
        definition = self.KPI_DEFINITIONS.get(kpi, {})
        direction_pref = definition.get('direction', 'lower_is_better')

        if abs(change_percent) < 2.0:  # Less than 2% change
            direction = TrendDirection.STABLE
        elif direction_pref == 'lower_is_better':
            direction = TrendDirection.IMPROVING if change_percent < 0 else TrendDirection.DEGRADING
        else:
            direction = TrendDirection.IMPROVING if change_percent > 0 else TrendDirection.DEGRADING

        # Calculate confidence based on R-squared and data points
        confidence = min(1.0, (r_squared or 0) * (len(points) / 100))

        analysis = TrendAnalysis(
            kpi_name=kpi,
            period=period,
            direction=direction,
            change_percent=change_percent,
            start_value=start_value,
            end_value=end_value,
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            std_dev=std_dev,
            slope=slope,
            r_squared=r_squared,
            data_points=len(points),
            confidence=confidence
        )

        logger.info(
            f"Trend analysis for {kpi}: direction={direction.value}, "
            f"change={change_percent:.1f}%"
        )

        return analysis

    def _linear_regression(
        self,
        points: List[KPIDataPoint]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Perform simple linear regression on data points."""
        if len(points) < 2:
            return None, None

        n = len(points)
        # Use timestamp as x (seconds since first point)
        first_ts = points[0].timestamp.timestamp()
        x = [(p.timestamp.timestamp() - first_ts) for p in points]
        y = [p.value for p in points]

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)

        # Calculate slope
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return None, None

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        if ss_tot == 0:
            r_squared = 1.0 if slope == 0 else 0.0
        else:
            intercept = (sum_y - slope * sum_x) / n
            ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
            r_squared = 1 - (ss_res / ss_tot)

        return slope, max(0, r_squared)

    def generate_kpi_report(self, period: DateRange) -> KPIReport:
        """
        Generate comprehensive KPI report for a period.

        Args:
            period: Date range for the report

        Returns:
            KPIReport with all KPI summaries and analyses
        """
        efficiency_kpis = []
        emissions_kpis = []
        reliability_kpis = []
        safety_kpis = []
        operational_kpis = []
        trend_analyses = []

        exceeding = []
        below = []
        critical = []

        kpi_scores = []

        for kpi_name, definition in self.KPI_DEFINITIONS.items():
            # Get trend analysis
            trend = self.compute_kpi_trends(kpi_name, period)
            trend_analyses.append(trend)

            # Get current and target values
            current_value = trend.end_value or 0.0
            target = self._targets.get(kpi_name)

            # Determine status
            status = self._determine_status(kpi_name, current_value, target)

            # Track achievement
            if status == KPIStatus.EXCEEDING:
                exceeding.append(kpi_name)
                kpi_scores.append(100)
            elif status == KPIStatus.ON_TARGET:
                kpi_scores.append(90)
            elif status == KPIStatus.BELOW_TARGET:
                below.append(kpi_name)
                kpi_scores.append(60)
            else:
                critical.append(kpi_name)
                kpi_scores.append(30)

            # Create summary
            summary = KPISummary(
                kpi_name=kpi_name,
                category=definition['category'],
                description=definition['description'],
                unit=definition['unit'],
                current_value=current_value,
                previous_value=trend.start_value,
                target_value=target.target_value if target else None,
                status=status,
                trend=trend.direction,
                change_percent=trend.change_percent,
                period_min=trend.min_value,
                period_max=trend.max_value,
                period_avg=trend.avg_value,
            )

            # Add to appropriate category
            category = definition['category']
            if category == KPICategory.EFFICIENCY:
                efficiency_kpis.append(summary)
            elif category == KPICategory.EMISSIONS:
                emissions_kpis.append(summary)
            elif category == KPICategory.RELIABILITY:
                reliability_kpis.append(summary)
            elif category == KPICategory.SAFETY:
                safety_kpis.append(summary)
            else:
                operational_kpis.append(summary)

        # Calculate overall score
        overall_score = statistics.mean(kpi_scores) if kpi_scores else 0.0

        # Determine overall status
        if critical:
            overall_status = KPIStatus.CRITICAL
        elif below:
            overall_status = KPIStatus.BELOW_TARGET
        elif exceeding:
            overall_status = KPIStatus.EXCEEDING
        else:
            overall_status = KPIStatus.ON_TARGET

        # Determine overall trend
        improving = sum(1 for t in trend_analyses if t.direction == TrendDirection.IMPROVING)
        degrading = sum(1 for t in trend_analyses if t.direction == TrendDirection.DEGRADING)

        if improving > degrading * 2:
            overall_trend = TrendDirection.IMPROVING
        elif degrading > improving * 2:
            overall_trend = TrendDirection.DEGRADING
        else:
            overall_trend = TrendDirection.STABLE

        # Generate recommendations
        recommendations = self._generate_recommendations(
            efficiency_kpis + emissions_kpis + reliability_kpis + safety_kpis + operational_kpis
        )

        report = KPIReport(
            period=period,
            overall_score=overall_score,
            overall_status=overall_status,
            overall_trend=overall_trend,
            efficiency_kpis=efficiency_kpis,
            emissions_kpis=emissions_kpis,
            reliability_kpis=reliability_kpis,
            safety_kpis=safety_kpis,
            operational_kpis=operational_kpis,
            trend_analyses=trend_analyses,
            exceeding_targets=exceeding,
            below_targets=below,
            critical_kpis=critical,
            recommendations=recommendations,
        )

        # Compute provenance hash
        report.provenance_hash = self._compute_provenance(report)

        logger.info(
            f"Generated KPI report: score={overall_score:.1f}, "
            f"status={overall_status.value}"
        )

        return report

    def _determine_status(
        self,
        kpi_name: str,
        value: float,
        target: Optional[KPITarget]
    ) -> KPIStatus:
        """Determine KPI status based on value and target."""
        if not target:
            return KPIStatus.ON_TARGET  # No target = assume OK

        direction = target.direction

        if direction == 'lower_is_better':
            if target.critical_threshold and value >= target.critical_threshold:
                return KPIStatus.CRITICAL
            if target.warning_threshold and value >= target.warning_threshold:
                return KPIStatus.BELOW_TARGET
            if value <= target.target_value:
                return KPIStatus.EXCEEDING
            return KPIStatus.ON_TARGET
        else:  # higher_is_better
            if target.critical_threshold and value <= target.critical_threshold:
                return KPIStatus.CRITICAL
            if target.warning_threshold and value <= target.warning_threshold:
                return KPIStatus.BELOW_TARGET
            if value >= target.target_value:
                return KPIStatus.EXCEEDING
            return KPIStatus.ON_TARGET

    def _generate_recommendations(
        self,
        summaries: List[KPISummary]
    ) -> List[str]:
        """Generate recommendations based on KPI summaries."""
        recommendations = []

        for summary in summaries:
            if summary.status == KPIStatus.CRITICAL:
                if 'emissions' in summary.kpi_name:
                    recommendations.append(
                        f"URGENT: {summary.kpi_name} is critical. "
                        "Review combustion tuning and consider maintenance."
                    )
                elif 'stability' in summary.kpi_name:
                    recommendations.append(
                        f"URGENT: {summary.kpi_name} is critical. "
                        "Check burner components and flame sensors."
                    )
                else:
                    recommendations.append(
                        f"URGENT: {summary.kpi_name} requires immediate attention."
                    )
            elif summary.status == KPIStatus.BELOW_TARGET:
                if summary.trend == TrendDirection.DEGRADING:
                    recommendations.append(
                        f"{summary.kpi_name} is below target and degrading. "
                        "Investigate root cause."
                    )

        # Add general recommendations
        if not recommendations:
            recommendations.append("All KPIs within acceptable ranges. Continue monitoring.")

        return recommendations

    def _compute_provenance(self, report: KPIReport) -> str:
        """Compute SHA-256 provenance hash for audit."""
        content = report.json(exclude={'provenance_hash'})
        return hashlib.sha256(content.encode()).hexdigest()

    def get_kpi_summary(self, kpi_name: str) -> Optional[Dict[str, Any]]:
        """Get current summary for a single KPI."""
        if kpi_name not in self._kpi_data:
            return None

        data = self._kpi_data[kpi_name]
        if not data:
            return None

        values = [p.value for p in data[-100:]]  # Last 100 points

        return {
            'kpi_name': kpi_name,
            'current_value': values[-1] if values else None,
            'count': len(data),
            'min': min(values) if values else None,
            'max': max(values) if values else None,
            'avg': statistics.mean(values) if values else None,
        }
