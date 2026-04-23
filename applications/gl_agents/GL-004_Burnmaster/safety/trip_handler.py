"""
TripHandler - Detects trip precursors and handles trip events.

This module detects early warning signs of potential trips and coordinates
responses to prevent unnecessary shutdowns while maintaining safety.

CRITICAL: The optimizer NEVER bypasses trips - it only detects precursors
and recommends preventive actions within safe operating bounds.

Example:
    >>> handler = TripHandler(unit_id="BLR-001")
    >>> precursor = handler.detect_trip_precursor(signals)
    >>> if precursor.risk_level > 0.7:
    ...     response = handler.execute_pre_trip_response(precursor)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class TripType(str, Enum):
    """Type of trip condition."""
    FLAME_FAILURE = "flame_failure"
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_TEMP = "high_temp"
    LOW_WATER = "low_water"
    HIGH_CO = "high_co"
    LOW_O2 = "low_o2"
    DRAFT_LOSS = "draft_loss"
    COMBUSTION_OSCILLATION = "combustion_oscillation"
    FUEL_PRESSURE = "fuel_pressure"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Risk level of trip precursor."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ResponseStatus(str, Enum):
    """Status of pre-trip response."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class TripPrecursor(BaseModel):
    """Detected precursor to potential trip."""
    unit_id: str = Field(..., description="Unit identifier")
    trip_type: TripType = Field(..., description="Type of potential trip")
    risk_level: RiskLevel = Field(..., description="Current risk level")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    trigger_signals: Dict[str, float] = Field(default_factory=dict)
    trend_direction: str = Field(..., description="Signal trend: rising, falling, stable")
    time_to_trip_estimate: Optional[float] = Field(None, description="Estimated seconds to trip")
    contributing_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    detection_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class ResponseResult(BaseModel):
    """Result of pre-trip response execution."""
    unit_id: str = Field(..., description="Unit identifier")
    precursor: TripPrecursor = Field(..., description="The precursor responded to")
    status: ResponseStatus = Field(..., description="Response status")
    actions_taken: List[str] = Field(default_factory=list)
    actions_blocked: List[str] = Field(default_factory=list)
    new_risk_score: float = Field(..., ge=0, le=1, description="Risk score after response")
    risk_reduction: float = Field(..., description="Risk reduction achieved")
    observe_only_mode: bool = Field(default=False, description="Switched to observe-only")
    execution_time_ms: float = Field(..., description="Response execution time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class TripEvent(BaseModel):
    """Actual trip event that occurred."""
    event_id: str = Field(..., description="Unique event ID")
    unit_id: str = Field(..., description="Unit identifier")
    trip_type: TripType = Field(..., description="Type of trip")
    trigger_value: float = Field(..., description="Value that triggered trip")
    trip_setpoint: float = Field(..., description="Trip setpoint")
    precursor_detected: bool = Field(default=False, description="Was precursor detected?")
    precursor_warning_time: Optional[float] = Field(None, description="Warning time before trip")
    root_cause: Optional[str] = Field(None, description="Identified root cause")
    sequence_of_events: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class TripAnalysis(BaseModel):
    """Analysis of trip event sequence."""
    unit_id: str = Field(..., description="Unit identifier")
    analysis_period: str = Field(..., description="Period analyzed")
    total_trips: int = Field(..., description="Total trips in period")
    trips_by_type: Dict[str, int] = Field(default_factory=dict)
    precursor_detection_rate: float = Field(..., description="% trips with precursor detected")
    average_warning_time: Optional[float] = Field(None, description="Avg warning time seconds")
    common_root_causes: List[str] = Field(default_factory=list)
    pattern_identified: bool = Field(default=False)
    pattern_description: Optional[str] = Field(None)
    reliability_score: float = Field(..., ge=0, le=100, description="System reliability score")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class PreventionRecommendation(BaseModel):
    """Recommendation for trip prevention."""
    unit_id: str = Field(..., description="Unit identifier")
    based_on_analysis: TripAnalysis = Field(..., description="Analysis this is based on")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    priority_order: List[str] = Field(default_factory=list)
    expected_trip_reduction: float = Field(..., description="Expected % reduction in trips")
    implementation_effort: str = Field(..., description="low, medium, high")
    roi_estimate: Optional[str] = Field(None, description="Return on investment estimate")
    safety_impact: str = Field(..., description="Impact on safety: positive, neutral, negative")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class TripHandler:
    """
    TripHandler detects trip precursors and coordinates preventive responses.

    CRITICAL SAFETY INVARIANT:
    - NEVER bypasses actual trips
    - ONLY detects precursors and recommends safe preventive actions
    - All recommendations stay within validated safety envelope
    - Automatically switches to observe-only on safety concern

    Attributes:
        unit_id: Identifier for the combustion unit
        trip_history: Historical trip events
        precursor_history: Historical precursor detections

    Example:
        >>> handler = TripHandler(unit_id="BLR-001")
        >>> signals = {"flame_signal": 25, "o2": 2.1, "co": 180}
        >>> precursor = handler.detect_trip_precursor(signals)
        >>> if precursor.risk_score > 0.5:
        ...     result = handler.execute_pre_trip_response(precursor)
    """

    # Trip thresholds and warning margins
    THRESHOLDS = {
        TripType.FLAME_FAILURE: {"trip": 15.0, "warning": 25.0},
        TripType.HIGH_CO: {"trip": 400.0, "warning": 250.0},
        TripType.LOW_O2: {"trip": 1.5, "warning": 2.5},
        TripType.HIGH_PRESSURE: {"trip": 150.0, "warning": 140.0},
        TripType.LOW_PRESSURE: {"trip": 50.0, "warning": 60.0},
        TripType.HIGH_TEMP: {"trip": 1000.0, "warning": 950.0},
        TripType.DRAFT_LOSS: {"trip": -0.02, "warning": -0.05},
    }

    def __init__(self, unit_id: str):
        """Initialize TripHandler for a specific unit."""
        self.unit_id = unit_id
        self.trip_history: List[TripEvent] = []
        self.precursor_history: List[TripPrecursor] = []
        self._signal_history: Dict[str, List[float]] = {}
        self._creation_time = datetime.utcnow()
        logger.info(f"TripHandler initialized for unit {unit_id}")

    def detect_trip_precursor(self, signals: Dict[str, float]) -> TripPrecursor:
        """
        Detect precursors to potential trip conditions.

        Args:
            signals: Current signal values (flame_signal, o2, co, etc.)

        Returns:
            TripPrecursor with risk assessment
        """
        # Store signals for trend analysis
        self._update_signal_history(signals)

        # Analyze each potential trip type
        precursors = []

        # Check flame signal
        if 'flame_signal' in signals:
            flame_precursor = self._analyze_flame_precursor(signals['flame_signal'])
            if flame_precursor:
                precursors.append(flame_precursor)

        # Check CO levels
        if 'co' in signals:
            co_precursor = self._analyze_co_precursor(signals['co'])
            if co_precursor:
                precursors.append(co_precursor)

        # Check O2 levels
        if 'o2' in signals:
            o2_precursor = self._analyze_o2_precursor(signals['o2'])
            if o2_precursor:
                precursors.append(o2_precursor)

        # Check pressure
        if 'pressure' in signals:
            pressure_precursor = self._analyze_pressure_precursor(signals['pressure'])
            if pressure_precursor:
                precursors.append(pressure_precursor)

        # Check draft
        if 'draft' in signals:
            draft_precursor = self._analyze_draft_precursor(signals['draft'])
            if draft_precursor:
                precursors.append(draft_precursor)

        # Return highest risk precursor or create low-risk default
        if precursors:
            highest_risk = max(precursors, key=lambda p: p.risk_score)
            self.precursor_history.append(highest_risk)
            return highest_risk

        # No significant precursors detected
        provenance_hash = hashlib.sha256(
            f"no_precursor_{self.unit_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return TripPrecursor(
            unit_id=self.unit_id,
            trip_type=TripType.UNKNOWN,
            risk_level=RiskLevel.LOW,
            risk_score=0.0,
            trigger_signals=signals,
            trend_direction="stable",
            contributing_factors=[],
            recommended_actions=[],
            provenance_hash=provenance_hash
        )

    def execute_pre_trip_response(self, precursor: TripPrecursor) -> ResponseResult:
        """
        Execute pre-trip preventive response within safe bounds.

        CRITICAL: All actions are within validated safety envelope.
        Never recommends actions that push toward instability.

        Args:
            precursor: The detected precursor to respond to

        Returns:
            ResponseResult with actions taken
        """
        start_time = datetime.utcnow()
        actions_taken = []
        actions_blocked = []

        # Determine response based on trip type
        if precursor.trip_type == TripType.FLAME_FAILURE:
            actions_taken, actions_blocked = self._respond_flame_precursor(precursor)
        elif precursor.trip_type == TripType.HIGH_CO:
            actions_taken, actions_blocked = self._respond_co_precursor(precursor)
        elif precursor.trip_type == TripType.LOW_O2:
            actions_taken, actions_blocked = self._respond_o2_precursor(precursor)
        elif precursor.trip_type == TripType.DRAFT_LOSS:
            actions_taken, actions_blocked = self._respond_draft_precursor(precursor)
        else:
            actions_taken = ["Monitor closely"]
            actions_blocked = []

        # Calculate new risk score (simulated reduction)
        risk_reduction = 0.2 if actions_taken else 0.0
        new_risk_score = max(0, precursor.risk_score - risk_reduction)

        # Determine response status
        if actions_blocked and not actions_taken:
            status = ResponseStatus.FAILED
        elif actions_blocked and actions_taken:
            status = ResponseStatus.PARTIAL
        elif actions_taken:
            status = ResponseStatus.SUCCESS
        else:
            status = ResponseStatus.NOT_APPLICABLE

        # Switch to observe-only if high risk remains
        observe_only = new_risk_score > 0.7

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        provenance_hash = hashlib.sha256(
            f"{precursor.json()}{status}{new_risk_score}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        result = ResponseResult(
            unit_id=self.unit_id,
            precursor=precursor,
            status=status,
            actions_taken=actions_taken,
            actions_blocked=actions_blocked,
            new_risk_score=new_risk_score,
            risk_reduction=risk_reduction,
            observe_only_mode=observe_only,
            execution_time_ms=execution_time,
            provenance_hash=provenance_hash
        )

        if observe_only:
            logger.warning(f"Pre-trip response: switching to OBSERVE-ONLY mode, risk={new_risk_score:.2f}")
        else:
            logger.info(f"Pre-trip response executed: {status.value}, risk reduced to {new_risk_score:.2f}")

        return result

    def log_trip_event(self, trip: TripEvent) -> None:
        """
        Log an actual trip event for analysis.

        Args:
            trip: Trip event to log
        """
        # Validate event ID
        if not trip.event_id:
            trip.event_id = hashlib.sha256(
                f"{trip.unit_id}_{trip.trip_type}_{trip.timestamp.isoformat()}".encode()
            ).hexdigest()[:16]

        # Check if precursor was detected
        recent_precursors = [
            p for p in self.precursor_history
            if p.trip_type == trip.trip_type
            and (trip.timestamp - p.detection_timestamp).total_seconds() < 300
        ]

        if recent_precursors:
            trip.precursor_detected = True
            earliest = min(recent_precursors, key=lambda p: p.detection_timestamp)
            trip.precursor_warning_time = (
                trip.timestamp - earliest.detection_timestamp
            ).total_seconds()

        self.trip_history.append(trip)
        logger.warning(f"Trip event logged: {trip.trip_type.value} at {trip.timestamp}")

    def analyze_trip_sequence(self, events: List[TripEvent]) -> TripAnalysis:
        """
        Analyze sequence of trip events for patterns.

        Args:
            events: List of trip events to analyze

        Returns:
            TripAnalysis with patterns and insights
        """
        if not events:
            provenance_hash = hashlib.sha256(
                f"no_events_{self.unit_id}_{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
            return TripAnalysis(
                unit_id=self.unit_id,
                analysis_period="N/A",
                total_trips=0,
                trips_by_type={},
                precursor_detection_rate=0,
                reliability_score=100,
                provenance_hash=provenance_hash
            )

        # Calculate analysis period
        timestamps = [e.timestamp for e in events]
        period_start = min(timestamps)
        period_end = max(timestamps)
        analysis_period = f"{period_start.isoformat()} to {period_end.isoformat()}"

        # Count by type
        trips_by_type: Dict[str, int] = {}
        for event in events:
            type_name = event.trip_type.value
            trips_by_type[type_name] = trips_by_type.get(type_name, 0) + 1

        # Precursor detection rate
        detected_count = sum(1 for e in events if e.precursor_detected)
        detection_rate = (detected_count / len(events) * 100) if events else 0

        # Average warning time
        warning_times = [
            e.precursor_warning_time for e in events
            if e.precursor_warning_time is not None
        ]
        avg_warning_time = statistics.mean(warning_times) if warning_times else None

        # Identify common root causes
        root_causes = [e.root_cause for e in events if e.root_cause]
        common_causes = list(set(root_causes))[:5]

        # Pattern detection (simplified)
        pattern_identified = False
        pattern_description = None
        if len(events) >= 5:
            # Check for repeated trip types
            most_common_type = max(trips_by_type, key=trips_by_type.get)
            if trips_by_type[most_common_type] >= len(events) * 0.5:
                pattern_identified = True
                pattern_description = f"Repeated {most_common_type} trips ({trips_by_type[most_common_type]}/{len(events)})"

        # Calculate reliability score
        hours_in_period = (period_end - period_start).total_seconds() / 3600
        trips_per_hour = len(events) / max(hours_in_period, 1)
        reliability_score = max(0, 100 - (trips_per_hour * 10))

        provenance_hash = hashlib.sha256(
            f"{self.unit_id}_{len(events)}_{detection_rate}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return TripAnalysis(
            unit_id=self.unit_id,
            analysis_period=analysis_period,
            total_trips=len(events),
            trips_by_type=trips_by_type,
            precursor_detection_rate=detection_rate,
            average_warning_time=avg_warning_time,
            common_root_causes=common_causes,
            pattern_identified=pattern_identified,
            pattern_description=pattern_description,
            reliability_score=reliability_score,
            provenance_hash=provenance_hash
        )

    def recommend_trip_prevention(self, analysis: TripAnalysis) -> PreventionRecommendation:
        """
        Generate trip prevention recommendations based on analysis.

        CRITICAL: All recommendations maintain or improve safety.

        Args:
            analysis: Trip analysis to base recommendations on

        Returns:
            PreventionRecommendation with actionable items
        """
        recommendations = []
        priority_order = []

        # Recommendations based on most common trip types
        for trip_type, count in sorted(
            analysis.trips_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if trip_type == "flame_failure":
                recommendations.append({
                    "action": "Increase minimum flame signal threshold",
                    "description": "Raise warning threshold to detect instability earlier",
                    "expected_impact": "Reduce flame failure trips by 30%",
                    "safety_impact": "positive"
                })
                priority_order.append("flame_monitoring")
            elif trip_type == "high_co":
                recommendations.append({
                    "action": "Improve O2 control loop tuning",
                    "description": "Faster response to O2 deviations prevents CO spikes",
                    "expected_impact": "Reduce CO trips by 40%",
                    "safety_impact": "positive"
                })
                priority_order.append("o2_control_tuning")
            elif trip_type == "low_o2":
                recommendations.append({
                    "action": "Increase minimum O2 setpoint margin",
                    "description": "Operate further from minimum O2 limit",
                    "expected_impact": "Reduce low O2 trips by 50%",
                    "safety_impact": "positive"
                })
                priority_order.append("o2_margin_increase")

        # General recommendations
        if analysis.precursor_detection_rate < 70:
            recommendations.append({
                "action": "Enhance precursor detection algorithms",
                "description": "Improve early warning detection capability",
                "expected_impact": f"Increase detection rate from {analysis.precursor_detection_rate:.0f}% to 80%",
                "safety_impact": "positive"
            })
            priority_order.append("detection_enhancement")

        # Expected trip reduction
        if analysis.total_trips > 0:
            expected_reduction = min(50, len(recommendations) * 15)
        else:
            expected_reduction = 0

        # Implementation effort
        if len(recommendations) <= 2:
            effort = "low"
        elif len(recommendations) <= 4:
            effort = "medium"
        else:
            effort = "high"

        provenance_hash = hashlib.sha256(
            f"{analysis.json()}{len(recommendations)}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return PreventionRecommendation(
            unit_id=self.unit_id,
            based_on_analysis=analysis,
            recommendations=recommendations,
            priority_order=priority_order,
            expected_trip_reduction=expected_reduction,
            implementation_effort=effort,
            safety_impact="positive",
            provenance_hash=provenance_hash
        )

    def _update_signal_history(self, signals: Dict[str, float]) -> None:
        """Update signal history for trend analysis."""
        for signal_name, value in signals.items():
            if signal_name not in self._signal_history:
                self._signal_history[signal_name] = []
            self._signal_history[signal_name].append(value)
            # Keep last 100 values
            if len(self._signal_history[signal_name]) > 100:
                self._signal_history[signal_name].pop(0)

    def _analyze_flame_precursor(self, flame_signal: float) -> Optional[TripPrecursor]:
        """Analyze flame signal for trip precursor."""
        thresholds = self.THRESHOLDS[TripType.FLAME_FAILURE]
        trip_point = thresholds["trip"]
        warning_point = thresholds["warning"]

        if flame_signal >= warning_point:
            return None

        # Calculate risk score
        if flame_signal < trip_point:
            risk_score = 1.0
            risk_level = RiskLevel.CRITICAL
        elif flame_signal < (warning_point - trip_point) * 0.3 + trip_point:
            risk_score = 0.8
            risk_level = RiskLevel.HIGH
        elif flame_signal < (warning_point - trip_point) * 0.6 + trip_point:
            risk_score = 0.5
            risk_level = RiskLevel.MODERATE
        else:
            risk_score = 0.3
            risk_level = RiskLevel.LOW

        # Trend analysis
        trend = self._calculate_trend('flame_signal')

        # Estimate time to trip
        if trend == "falling" and risk_score > 0.3:
            time_estimate = max(5, (flame_signal - trip_point) * 2)
        else:
            time_estimate = None

        provenance_hash = hashlib.sha256(
            f"flame_{flame_signal}_{risk_score}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return TripPrecursor(
            unit_id=self.unit_id,
            trip_type=TripType.FLAME_FAILURE,
            risk_level=risk_level,
            risk_score=risk_score,
            trigger_signals={"flame_signal": flame_signal},
            trend_direction=trend,
            time_to_trip_estimate=time_estimate,
            contributing_factors=["Low flame intensity"],
            recommended_actions=[
                "Verify burner fuel supply",
                "Check combustion air flow",
                "Reduce load if unstable"
            ],
            provenance_hash=provenance_hash
        )

    def _analyze_co_precursor(self, co: float) -> Optional[TripPrecursor]:
        """Analyze CO for trip precursor."""
        thresholds = self.THRESHOLDS[TripType.HIGH_CO]
        trip_point = thresholds["trip"]
        warning_point = thresholds["warning"]

        if co <= warning_point:
            return None

        risk_score = min(1.0, (co - warning_point) / (trip_point - warning_point))
        risk_level = (
            RiskLevel.CRITICAL if risk_score >= 0.9 else
            RiskLevel.HIGH if risk_score >= 0.6 else
            RiskLevel.MODERATE
        )

        trend = self._calculate_trend('co')

        provenance_hash = hashlib.sha256(
            f"co_{co}_{risk_score}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return TripPrecursor(
            unit_id=self.unit_id,
            trip_type=TripType.HIGH_CO,
            risk_level=risk_level,
            risk_score=risk_score,
            trigger_signals={"co": co},
            trend_direction=trend,
            contributing_factors=["Incomplete combustion", "Low excess air"],
            recommended_actions=[
                "Increase excess air",
                "Verify fuel quality",
                "Check burner alignment"
            ],
            provenance_hash=provenance_hash
        )

    def _analyze_o2_precursor(self, o2: float) -> Optional[TripPrecursor]:
        """Analyze O2 for low O2 trip precursor."""
        thresholds = self.THRESHOLDS[TripType.LOW_O2]
        trip_point = thresholds["trip"]
        warning_point = thresholds["warning"]

        if o2 >= warning_point:
            return None

        risk_score = min(1.0, (warning_point - o2) / (warning_point - trip_point))
        risk_level = (
            RiskLevel.CRITICAL if risk_score >= 0.9 else
            RiskLevel.HIGH if risk_score >= 0.6 else
            RiskLevel.MODERATE
        )

        trend = self._calculate_trend('o2')

        provenance_hash = hashlib.sha256(
            f"o2_{o2}_{risk_score}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return TripPrecursor(
            unit_id=self.unit_id,
            trip_type=TripType.LOW_O2,
            risk_level=risk_level,
            risk_score=risk_score,
            trigger_signals={"o2": o2},
            trend_direction=trend,
            contributing_factors=["Insufficient combustion air"],
            recommended_actions=[
                "Increase air flow",
                "Reduce fuel flow",
                "Check air damper position"
            ],
            provenance_hash=provenance_hash
        )

    def _analyze_pressure_precursor(self, pressure: float) -> Optional[TripPrecursor]:
        """Analyze pressure for trip precursor."""
        high_thresholds = self.THRESHOLDS[TripType.HIGH_PRESSURE]
        low_thresholds = self.THRESHOLDS[TripType.LOW_PRESSURE]

        # Check high pressure
        if pressure > high_thresholds["warning"]:
            risk_score = min(1.0, (pressure - high_thresholds["warning"]) /
                           (high_thresholds["trip"] - high_thresholds["warning"]))
            trip_type = TripType.HIGH_PRESSURE
            actions = ["Reduce firing rate", "Check steam demand"]
        # Check low pressure
        elif pressure < low_thresholds["warning"]:
            risk_score = min(1.0, (low_thresholds["warning"] - pressure) /
                           (low_thresholds["warning"] - low_thresholds["trip"]))
            trip_type = TripType.LOW_PRESSURE
            actions = ["Increase firing rate", "Check feedwater flow"]
        else:
            return None

        risk_level = (
            RiskLevel.CRITICAL if risk_score >= 0.9 else
            RiskLevel.HIGH if risk_score >= 0.6 else
            RiskLevel.MODERATE
        )

        trend = self._calculate_trend('pressure')

        provenance_hash = hashlib.sha256(
            f"pressure_{pressure}_{risk_score}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return TripPrecursor(
            unit_id=self.unit_id,
            trip_type=trip_type,
            risk_level=risk_level,
            risk_score=risk_score,
            trigger_signals={"pressure": pressure},
            trend_direction=trend,
            contributing_factors=["Pressure deviation"],
            recommended_actions=actions,
            provenance_hash=provenance_hash
        )

    def _analyze_draft_precursor(self, draft: float) -> Optional[TripPrecursor]:
        """Analyze draft for trip precursor."""
        thresholds = self.THRESHOLDS[TripType.DRAFT_LOSS]
        trip_point = thresholds["trip"]
        warning_point = thresholds["warning"]

        if draft <= warning_point:  # Draft is negative, so <= is correct
            return None

        risk_score = min(1.0, (draft - warning_point) / (trip_point - warning_point))
        risk_level = (
            RiskLevel.CRITICAL if risk_score >= 0.9 else
            RiskLevel.HIGH if risk_score >= 0.6 else
            RiskLevel.MODERATE
        )

        trend = self._calculate_trend('draft')

        provenance_hash = hashlib.sha256(
            f"draft_{draft}_{risk_score}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return TripPrecursor(
            unit_id=self.unit_id,
            trip_type=TripType.DRAFT_LOSS,
            risk_level=risk_level,
            risk_score=risk_score,
            trigger_signals={"draft": draft},
            trend_direction=trend,
            contributing_factors=["Insufficient furnace draft"],
            recommended_actions=[
                "Increase ID fan speed",
                "Check for duct blockage"
            ],
            provenance_hash=provenance_hash
        )

    def _calculate_trend(self, signal_name: str) -> str:
        """Calculate signal trend from history."""
        if signal_name not in self._signal_history:
            return "stable"

        history = self._signal_history[signal_name]
        if len(history) < 5:
            return "stable"

        recent = history[-5:]
        avg_diff = sum(recent[i+1] - recent[i] for i in range(len(recent)-1)) / (len(recent)-1)

        if avg_diff > 0.1:
            return "rising"
        elif avg_diff < -0.1:
            return "falling"
        return "stable"

    def _respond_flame_precursor(
        self,
        precursor: TripPrecursor
    ) -> tuple[List[str], List[str]]:
        """Generate response to flame precursor."""
        actions_taken = []
        actions_blocked = []

        if precursor.risk_score < 0.5:
            actions_taken.append("Increased monitoring frequency")
        elif precursor.risk_score < 0.8:
            actions_taken.append("Recommend reduce load to stabilize flame")
            actions_taken.append("Alert operator to flame condition")
        else:
            actions_taken.append("Alert operator: CRITICAL flame condition")
            actions_blocked.append("Load increase blocked - flame unstable")

        return actions_taken, actions_blocked

    def _respond_co_precursor(
        self,
        precursor: TripPrecursor
    ) -> tuple[List[str], List[str]]:
        """Generate response to CO precursor."""
        actions_taken = []
        actions_blocked = []

        if precursor.risk_score < 0.5:
            actions_taken.append("Increased O2 setpoint slightly")
        elif precursor.risk_score < 0.8:
            actions_taken.append("Recommend increase excess air")
            actions_taken.append("Alert operator to CO elevation")
        else:
            actions_taken.append("Alert operator: HIGH CO condition")
            actions_blocked.append("O2 reduction blocked - CO elevated")

        return actions_taken, actions_blocked

    def _respond_o2_precursor(
        self,
        precursor: TripPrecursor
    ) -> tuple[List[str], List[str]]:
        """Generate response to low O2 precursor."""
        actions_taken = []
        actions_blocked = []

        if precursor.risk_score < 0.5:
            actions_taken.append("Recommend slight air increase")
        elif precursor.risk_score < 0.8:
            actions_taken.append("Recommend reduce fuel or increase air")
        else:
            actions_taken.append("Alert operator: LOW O2 condition")
            actions_blocked.append("Fuel increase blocked - low O2")
            actions_blocked.append("Air reduction blocked - low O2")

        return actions_taken, actions_blocked

    def _respond_draft_precursor(
        self,
        precursor: TripPrecursor
    ) -> tuple[List[str], List[str]]:
        """Generate response to draft precursor."""
        actions_taken = []
        actions_blocked = []

        if precursor.risk_score < 0.5:
            actions_taken.append("Monitor draft closely")
        elif precursor.risk_score < 0.8:
            actions_taken.append("Recommend increase ID fan speed")
        else:
            actions_taken.append("Alert operator: DRAFT LOSS condition")
            actions_blocked.append("Firing rate increase blocked - low draft")

        return actions_taken, actions_blocked
