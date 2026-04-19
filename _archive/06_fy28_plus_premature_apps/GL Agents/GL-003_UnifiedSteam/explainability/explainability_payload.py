"""
GL-003 UNIFIEDSTEAM - Explainability Payload

Defines the API payload structure for explainability responses.
This is the standard format returned by the explainability API endpoints.

Structure includes:
- Primary drivers (ranked) with directionality (+/- effect)
- Supporting evidence (signals + time window)
- Physics constraints and safety envelope checks
- Confidence score and uncertainty band
- Suggested verification steps
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)


class DriverDirection(Enum):
    """Direction of driver effect."""
    POSITIVE = "positive"  # Increases the predicted outcome
    NEGATIVE = "negative"  # Decreases the predicted outcome
    NEUTRAL = "neutral"  # No significant effect


class EvidenceType(Enum):
    """Type of supporting evidence."""
    SIGNAL = "signal"  # Sensor/measurement signal
    CALCULATION = "calculation"  # Derived/calculated value
    MODEL_OUTPUT = "model_output"  # ML model prediction
    HISTORICAL = "historical"  # Historical comparison
    RULE_BASED = "rule_based"  # Business rule evaluation


class ConstraintStatus(Enum):
    """Status of a physics constraint."""
    SATISFIED = "satisfied"  # Within limits
    BINDING = "binding"  # At limit
    VIOLATED = "violated"  # Outside limits (should trigger alarm)
    NOT_APPLICABLE = "not_applicable"  # Does not apply to this case


class SafetyCheckResult(Enum):
    """Result of safety envelope check."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_CHECKED = "not_checked"


class VerificationPriority(Enum):
    """Priority of verification step."""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


@dataclass
class PrimaryDriver:
    """
    A primary driver contributing to the recommendation.

    Drivers are ranked by impact magnitude.
    """
    driver_id: str
    driver_name: str
    driver_description: str

    # Current state
    current_value: float
    unit: str
    normal_range: Tuple[float, float]

    # Impact
    direction: DriverDirection
    impact_magnitude: float  # 0-1 scale
    impact_rank: int  # 1 = highest impact

    # Context
    deviation_from_normal: float  # Percentage or absolute
    trend: str  # "increasing", "decreasing", "stable"

    # Source
    source_signal: str  # Tag name or calculation reference
    source_type: EvidenceType

    def to_dict(self) -> Dict:
        return {
            "driver_id": self.driver_id,
            "driver_name": self.driver_name,
            "driver_description": self.driver_description,
            "current_value": self.current_value,
            "unit": self.unit,
            "normal_range": list(self.normal_range),
            "direction": self.direction.value,
            "impact_magnitude": self.impact_magnitude,
            "impact_rank": self.impact_rank,
            "deviation_from_normal": self.deviation_from_normal,
            "trend": self.trend,
            "source_signal": self.source_signal,
            "source_type": self.source_type.value,
        }


@dataclass
class SupportingEvidence:
    """
    Supporting evidence for the recommendation.

    Includes signal data and time windows.
    """
    evidence_id: str
    evidence_type: EvidenceType
    description: str

    # Signal information
    signal_name: str
    signal_value: Union[float, str, List[float]]
    signal_unit: str

    # Time window
    time_window_start: datetime
    time_window_end: datetime
    sample_count: int

    # Quality
    data_quality: str  # "good", "questionable", "bad"
    confidence: float  # 0-1

    # Statistical summary (for time series)
    mean_value: Optional[float] = None
    std_deviation: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "description": self.description,
            "signal_name": self.signal_name,
            "signal_value": self.signal_value,
            "signal_unit": self.signal_unit,
            "time_window_start": self.time_window_start.isoformat(),
            "time_window_end": self.time_window_end.isoformat(),
            "sample_count": self.sample_count,
            "data_quality": self.data_quality,
            "confidence": self.confidence,
            "mean_value": self.mean_value,
            "std_deviation": self.std_deviation,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


@dataclass
class PhysicsConstraint:
    """
    A physics constraint evaluated for the recommendation.
    """
    constraint_id: str
    constraint_name: str
    constraint_description: str
    constraint_type: str  # "mass_balance", "energy_balance", "pressure", "temperature"

    # Limit information
    limit_type: str  # "min", "max", "equality"
    limit_value: float
    current_value: float
    unit: str

    # Status
    status: ConstraintStatus
    margin_percent: float  # Distance from limit as percentage

    # Impact if violated
    violation_consequence: str
    mitigation_options: List[str]

    def to_dict(self) -> Dict:
        return {
            "constraint_id": self.constraint_id,
            "constraint_name": self.constraint_name,
            "constraint_description": self.constraint_description,
            "constraint_type": self.constraint_type,
            "limit_type": self.limit_type,
            "limit_value": self.limit_value,
            "current_value": self.current_value,
            "unit": self.unit,
            "status": self.status.value,
            "margin_percent": self.margin_percent,
            "violation_consequence": self.violation_consequence,
            "mitigation_options": self.mitigation_options,
        }


@dataclass
class SafetyEnvelopeCheck:
    """
    Safety envelope verification result.
    """
    check_id: str
    check_name: str
    check_description: str

    # Result
    result: SafetyCheckResult
    result_details: str

    # Checked values
    checked_parameter: str
    parameter_value: float
    safety_limit: float
    unit: str

    # Margin
    margin_to_limit: float
    margin_percent: float

    # Alarm status
    alarm_active: bool
    alarm_level: Optional[str] = None  # "low", "high", "hihi", etc.

    def to_dict(self) -> Dict:
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "check_description": self.check_description,
            "result": self.result.value,
            "result_details": self.result_details,
            "checked_parameter": self.checked_parameter,
            "parameter_value": self.parameter_value,
            "safety_limit": self.safety_limit,
            "unit": self.unit,
            "margin_to_limit": self.margin_to_limit,
            "margin_percent": self.margin_percent,
            "alarm_active": self.alarm_active,
            "alarm_level": self.alarm_level,
        }


@dataclass
class VerificationStep:
    """
    A suggested verification step for the recommendation.
    """
    step_id: str
    step_number: int
    priority: VerificationPriority

    # Step details
    action: str
    description: str
    expected_outcome: str

    # What to check
    parameters_to_verify: List[str]
    expected_values: Dict[str, Tuple[float, float]]  # Parameter -> (min, max)

    # Timing
    timing: str  # "immediate", "within_5min", "within_1hr", etc.
    estimated_duration_minutes: int

    # Dependencies
    depends_on_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "priority": self.priority.value,
            "action": self.action,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "parameters_to_verify": self.parameters_to_verify,
            "expected_values": {k: list(v) for k, v in self.expected_values.items()},
            "timing": self.timing,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "depends_on_steps": self.depends_on_steps,
        }


@dataclass
class ConfidenceScore:
    """
    Detailed confidence scoring with breakdown.
    """
    overall_confidence: float  # 0-1

    # Confidence components
    data_quality_score: float
    model_confidence: float
    physics_validation_score: float
    historical_accuracy_score: float

    # Uncertainty
    uncertainty_lower: float  # Lower bound of prediction
    uncertainty_upper: float  # Upper bound of prediction
    uncertainty_type: str  # "confidence_interval", "prediction_interval"

    # Factors affecting confidence
    confidence_boosters: List[str]  # Factors increasing confidence
    confidence_reducers: List[str]  # Factors decreasing confidence

    def to_dict(self) -> Dict:
        return {
            "overall_confidence": self.overall_confidence,
            "data_quality_score": self.data_quality_score,
            "model_confidence": self.model_confidence,
            "physics_validation_score": self.physics_validation_score,
            "historical_accuracy_score": self.historical_accuracy_score,
            "uncertainty_lower": self.uncertainty_lower,
            "uncertainty_upper": self.uncertainty_upper,
            "uncertainty_type": self.uncertainty_type,
            "confidence_boosters": self.confidence_boosters,
            "confidence_reducers": self.confidence_reducers,
        }


@dataclass
class ExplainabilityPayload:
    """
    Complete explainability payload for API response.

    This is the standard format returned by explainability endpoints.
    """
    # Identification
    payload_id: str
    recommendation_id: str
    timestamp: datetime
    affected_asset: str
    asset_type: str

    # Version and metadata
    api_version: str = "1.0"
    agent_id: str = "GL-003"

    # Primary drivers (ranked)
    primary_drivers: List[PrimaryDriver] = field(default_factory=list)

    # Supporting evidence
    supporting_evidence: List[SupportingEvidence] = field(default_factory=list)

    # Physics constraints
    physics_constraints: List[PhysicsConstraint] = field(default_factory=list)

    # Safety checks
    safety_envelope_checks: List[SafetyEnvelopeCheck] = field(default_factory=list)

    # Confidence
    confidence_score: Optional[ConfidenceScore] = None

    # Verification steps
    suggested_verification_steps: List[VerificationStep] = field(default_factory=list)

    # Narrative explanations
    summary_explanation: str = ""
    detailed_explanation: str = ""
    engineering_explanation: str = ""

    # Audit trail
    calculation_trace_id: Optional[str] = None
    model_version: Optional[str] = None
    data_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "payload_id": self.payload_id,
            "recommendation_id": self.recommendation_id,
            "timestamp": self.timestamp.isoformat(),
            "affected_asset": self.affected_asset,
            "asset_type": self.asset_type,
            "api_version": self.api_version,
            "agent_id": self.agent_id,
            "primary_drivers": [d.to_dict() for d in self.primary_drivers],
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "physics_constraints": [c.to_dict() for c in self.physics_constraints],
            "safety_envelope_checks": [s.to_dict() for s in self.safety_envelope_checks],
            "confidence_score": self.confidence_score.to_dict() if self.confidence_score else None,
            "suggested_verification_steps": [s.to_dict() for s in self.suggested_verification_steps],
            "summary_explanation": self.summary_explanation,
            "detailed_explanation": self.detailed_explanation,
            "engineering_explanation": self.engineering_explanation,
            "calculation_trace_id": self.calculation_trace_id,
            "model_version": self.model_version,
            "data_timestamp": self.data_timestamp.isoformat() if self.data_timestamp else None,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)


class ExplainabilityPayloadBuilder:
    """
    Builder for constructing ExplainabilityPayload objects.

    Provides a fluent interface for building payloads.
    """

    def __init__(
        self,
        recommendation_id: str,
        affected_asset: str,
        asset_type: str,
    ) -> None:
        self._payload_id = str(uuid.uuid4())
        self._recommendation_id = recommendation_id
        self._affected_asset = affected_asset
        self._asset_type = asset_type
        self._timestamp = datetime.now(timezone.utc)

        self._drivers: List[PrimaryDriver] = []
        self._evidence: List[SupportingEvidence] = []
        self._constraints: List[PhysicsConstraint] = []
        self._safety_checks: List[SafetyEnvelopeCheck] = []
        self._verification_steps: List[VerificationStep] = []
        self._confidence: Optional[ConfidenceScore] = None

        self._summary = ""
        self._detailed = ""
        self._engineering = ""
        self._trace_id: Optional[str] = None
        self._model_version: Optional[str] = None
        self._data_timestamp: Optional[datetime] = None

    def add_driver(
        self,
        name: str,
        description: str,
        current_value: float,
        unit: str,
        direction: DriverDirection,
        impact_magnitude: float,
        normal_range: Tuple[float, float],
        source_signal: str,
        source_type: EvidenceType = EvidenceType.SIGNAL,
        trend: str = "stable",
    ) -> "ExplainabilityPayloadBuilder":
        """Add a primary driver."""
        driver = PrimaryDriver(
            driver_id=str(uuid.uuid4())[:8],
            driver_name=name,
            driver_description=description,
            current_value=current_value,
            unit=unit,
            normal_range=normal_range,
            direction=direction,
            impact_magnitude=impact_magnitude,
            impact_rank=len(self._drivers) + 1,
            deviation_from_normal=self._calculate_deviation(
                current_value, normal_range
            ),
            trend=trend,
            source_signal=source_signal,
            source_type=source_type,
        )
        self._drivers.append(driver)
        return self

    def _calculate_deviation(
        self,
        value: float,
        normal_range: Tuple[float, float],
    ) -> float:
        """Calculate deviation from normal range."""
        low, high = normal_range
        midpoint = (low + high) / 2
        if midpoint != 0:
            return (value - midpoint) / midpoint * 100
        return 0.0

    def add_evidence(
        self,
        description: str,
        signal_name: str,
        signal_value: Union[float, str, List[float]],
        signal_unit: str,
        time_window_minutes: int,
        evidence_type: EvidenceType = EvidenceType.SIGNAL,
        confidence: float = 0.9,
        data_quality: str = "good",
    ) -> "ExplainabilityPayloadBuilder":
        """Add supporting evidence."""
        now = datetime.now(timezone.utc)
        evidence = SupportingEvidence(
            evidence_id=str(uuid.uuid4())[:8],
            evidence_type=evidence_type,
            description=description,
            signal_name=signal_name,
            signal_value=signal_value,
            signal_unit=signal_unit,
            time_window_start=datetime.fromtimestamp(
                now.timestamp() - time_window_minutes * 60, tz=timezone.utc
            ),
            time_window_end=now,
            sample_count=time_window_minutes,  # Assume 1 sample per minute
            data_quality=data_quality,
            confidence=confidence,
        )
        self._evidence.append(evidence)
        return self

    def add_constraint(
        self,
        name: str,
        description: str,
        constraint_type: str,
        limit_type: str,
        limit_value: float,
        current_value: float,
        unit: str,
        violation_consequence: str = "",
        mitigation_options: Optional[List[str]] = None,
    ) -> "ExplainabilityPayloadBuilder":
        """Add a physics constraint."""
        # Determine status
        margin_percent = abs(limit_value - current_value) / abs(limit_value) * 100 if limit_value != 0 else 100

        if limit_type == "max":
            if current_value > limit_value:
                status = ConstraintStatus.VIOLATED
            elif margin_percent < 5:
                status = ConstraintStatus.BINDING
            else:
                status = ConstraintStatus.SATISFIED
        elif limit_type == "min":
            if current_value < limit_value:
                status = ConstraintStatus.VIOLATED
            elif margin_percent < 5:
                status = ConstraintStatus.BINDING
            else:
                status = ConstraintStatus.SATISFIED
        else:
            status = ConstraintStatus.SATISFIED

        constraint = PhysicsConstraint(
            constraint_id=str(uuid.uuid4())[:8],
            constraint_name=name,
            constraint_description=description,
            constraint_type=constraint_type,
            limit_type=limit_type,
            limit_value=limit_value,
            current_value=current_value,
            unit=unit,
            status=status,
            margin_percent=margin_percent,
            violation_consequence=violation_consequence,
            mitigation_options=mitigation_options or [],
        )
        self._constraints.append(constraint)
        return self

    def add_safety_check(
        self,
        name: str,
        description: str,
        parameter: str,
        parameter_value: float,
        safety_limit: float,
        unit: str,
        result: SafetyCheckResult,
        alarm_active: bool = False,
        alarm_level: Optional[str] = None,
    ) -> "ExplainabilityPayloadBuilder":
        """Add a safety envelope check."""
        margin = safety_limit - parameter_value
        margin_percent = margin / safety_limit * 100 if safety_limit != 0 else 100

        check = SafetyEnvelopeCheck(
            check_id=str(uuid.uuid4())[:8],
            check_name=name,
            check_description=description,
            result=result,
            result_details=f"Parameter at {parameter_value:.1f} vs limit {safety_limit:.1f}",
            checked_parameter=parameter,
            parameter_value=parameter_value,
            safety_limit=safety_limit,
            unit=unit,
            margin_to_limit=margin,
            margin_percent=margin_percent,
            alarm_active=alarm_active,
            alarm_level=alarm_level,
        )
        self._safety_checks.append(check)
        return self

    def add_verification_step(
        self,
        action: str,
        description: str,
        expected_outcome: str,
        parameters: List[str],
        expected_values: Dict[str, Tuple[float, float]],
        priority: VerificationPriority = VerificationPriority.RECOMMENDED,
        timing: str = "within_5min",
        duration_minutes: int = 5,
    ) -> "ExplainabilityPayloadBuilder":
        """Add a verification step."""
        step = VerificationStep(
            step_id=str(uuid.uuid4())[:8],
            step_number=len(self._verification_steps) + 1,
            priority=priority,
            action=action,
            description=description,
            expected_outcome=expected_outcome,
            parameters_to_verify=parameters,
            expected_values=expected_values,
            timing=timing,
            estimated_duration_minutes=duration_minutes,
        )
        self._verification_steps.append(step)
        return self

    def set_confidence(
        self,
        overall: float,
        data_quality: float = 0.9,
        model_confidence: float = 0.85,
        physics_validation: float = 0.9,
        historical_accuracy: float = 0.8,
        uncertainty_band: float = 0.1,
        boosters: Optional[List[str]] = None,
        reducers: Optional[List[str]] = None,
    ) -> "ExplainabilityPayloadBuilder":
        """Set confidence score."""
        self._confidence = ConfidenceScore(
            overall_confidence=overall,
            data_quality_score=data_quality,
            model_confidence=model_confidence,
            physics_validation_score=physics_validation,
            historical_accuracy_score=historical_accuracy,
            uncertainty_lower=overall - uncertainty_band,
            uncertainty_upper=min(1.0, overall + uncertainty_band),
            uncertainty_type="confidence_interval",
            confidence_boosters=boosters or [],
            confidence_reducers=reducers or [],
        )
        return self

    def set_explanations(
        self,
        summary: str,
        detailed: str = "",
        engineering: str = "",
    ) -> "ExplainabilityPayloadBuilder":
        """Set narrative explanations."""
        self._summary = summary
        self._detailed = detailed
        self._engineering = engineering
        return self

    def set_audit_info(
        self,
        trace_id: Optional[str] = None,
        model_version: Optional[str] = None,
        data_timestamp: Optional[datetime] = None,
    ) -> "ExplainabilityPayloadBuilder":
        """Set audit trail information."""
        self._trace_id = trace_id
        self._model_version = model_version
        self._data_timestamp = data_timestamp
        return self

    def build(self) -> ExplainabilityPayload:
        """Build the final payload."""
        # Sort drivers by impact
        self._drivers.sort(key=lambda d: d.impact_magnitude, reverse=True)
        for i, driver in enumerate(self._drivers):
            driver.impact_rank = i + 1

        payload = ExplainabilityPayload(
            payload_id=self._payload_id,
            recommendation_id=self._recommendation_id,
            timestamp=self._timestamp,
            affected_asset=self._affected_asset,
            asset_type=self._asset_type,
            primary_drivers=self._drivers,
            supporting_evidence=self._evidence,
            physics_constraints=self._constraints,
            safety_envelope_checks=self._safety_checks,
            confidence_score=self._confidence,
            suggested_verification_steps=self._verification_steps,
            summary_explanation=self._summary,
            detailed_explanation=self._detailed,
            engineering_explanation=self._engineering,
            calculation_trace_id=self._trace_id,
            model_version=self._model_version,
            data_timestamp=self._data_timestamp,
        )

        logger.info(f"Built explainability payload: {self._payload_id}")
        return payload


def create_sample_payload() -> ExplainabilityPayload:
    """Create a sample payload for testing/documentation."""
    builder = ExplainabilityPayloadBuilder(
        recommendation_id="REC-2024-001",
        affected_asset="Trap-A-101",
        asset_type="steam_trap",
    )

    payload = (
        builder
        .add_driver(
            name="Temperature Differential",
            description="Temperature drop across steam trap",
            current_value=185.0,
            unit="F",
            direction=DriverDirection.POSITIVE,
            impact_magnitude=0.35,
            normal_range=(100.0, 180.0),
            source_signal="TT-A-101-DIFF",
            trend="increasing",
        )
        .add_driver(
            name="Subcooling",
            description="Condensate subcooling below saturation",
            current_value=2.5,
            unit="F",
            direction=DriverDirection.NEGATIVE,
            impact_magnitude=0.25,
            normal_range=(5.0, 15.0),
            source_signal="TT-A-101-OUT",
            trend="decreasing",
        )
        .add_evidence(
            description="Temperature trend over past hour",
            signal_name="TT-A-101-IN",
            signal_value=365.0,
            signal_unit="F",
            time_window_minutes=60,
        )
        .add_constraint(
            name="Maximum Outlet Temperature",
            description="Outlet temperature must not exceed condensate system rating",
            constraint_type="temperature",
            limit_type="max",
            limit_value=220.0,
            current_value=180.0,
            unit="F",
            violation_consequence="Condensate flash in return system",
        )
        .add_safety_check(
            name="Trap Inlet Pressure",
            description="Inlet pressure vs maximum rated pressure",
            parameter="PT-A-101",
            parameter_value=150.0,
            safety_limit=300.0,
            unit="psig",
            result=SafetyCheckResult.PASS,
        )
        .add_verification_step(
            action="Inspect trap with ultrasonic tester",
            description="Use ultrasonic leak detector to verify trap operation",
            expected_outcome="Cycling detected (thermostatic) or intermittent discharge (inverted bucket)",
            parameters=["trap_cycling", "discharge_pattern"],
            expected_values={"trap_cycling": (0.5, 5.0)},
            priority=VerificationPriority.REQUIRED,
        )
        .set_confidence(
            overall=0.82,
            boosters=["Recent calibration of temperature sensors", "Consistent historical pattern"],
            reducers=["Limited subcooling data points"],
        )
        .set_explanations(
            summary="Steam trap Trap-A-101 showing signs of potential blow-through based on reduced subcooling and elevated temperature differential.",
            detailed="Analysis indicates the temperature differential across the trap has increased beyond normal operating range while subcooling has decreased, suggesting live steam may be passing through the trap.",
            engineering="Temperature drop across trap increased to 185 F (normal: 100-180 F) while condensate subcooling dropped to 2.5 F (normal: 5-15 F). This pattern is consistent with partial steam blow-through. Recommend ultrasonic inspection to confirm.",
        )
        .build()
    )

    return payload
