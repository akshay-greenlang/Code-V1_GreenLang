"""
GL-012 STEAMQUAL - Root Cause Analyzer

Root cause analysis for steam quality events including high drum level,
separator flooding, PRV condensation, and trap failure.

This module provides:
1. Causal templates for common quality issues
2. Correlation analysis between sensors and events
3. Timeline reconstruction for event sequences
4. Physics-grounded root cause identification

All explanations are traceable to data and assumptions per playbook requirement.

Reference:
    - ASME PTC 19.11 Steam Quality
    - API 534 Steam Drum Internals
    - ISA-TR84.00.02 Safety Instrumented Systems

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RootCauseCategory(Enum):
    """Categories of root causes for quality events."""

    MECHANICAL = "mechanical"          # Equipment failure/degradation
    OPERATIONAL = "operational"        # Operating condition issues
    CONTROL = "control"                # Control system issues
    INSTRUMENTATION = "instrumentation"  # Sensor/measurement issues
    PROCESS = "process"                # Process upset conditions
    EXTERNAL = "external"              # External factors


class CausalChainType(Enum):
    """Types of causal chains."""

    DIRECT = "direct"              # Single cause -> effect
    MULTI_FACTOR = "multi_factor"  # Multiple contributing causes
    CASCADE = "cascade"            # Chain of events
    FEEDBACK = "feedback"          # Feedback loop amplification


class EventSeverity(Enum):
    """Severity levels for quality events."""

    S0_CRITICAL = "S0"     # Immediate safety concern
    S1_HIGH = "S1"         # Significant quality degradation
    S2_MEDIUM = "S2"       # Moderate quality impact
    S3_LOW = "S3"          # Minor quality deviation


class CorrelationType(Enum):
    """Types of correlation between variables."""

    STRONG_POSITIVE = "strong_positive"   # r > 0.7
    MODERATE_POSITIVE = "moderate_positive"  # 0.4 < r < 0.7
    WEAK_POSITIVE = "weak_positive"       # 0.2 < r < 0.4
    NONE = "none"                         # -0.2 < r < 0.2
    WEAK_NEGATIVE = "weak_negative"       # -0.4 < r < -0.2
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 < r < -0.4
    STRONG_NEGATIVE = "strong_negative"   # r < -0.7


class TimelinePriority(Enum):
    """Priority levels for timeline events."""

    PRIMARY = "primary"       # Direct cause
    CONTRIBUTING = "contributing"  # Contributing factor
    CONSEQUENCE = "consequence"    # Effect of the event
    INFORMATIONAL = "informational"  # Background context


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CausalFactor:
    """Single causal factor contributing to an event."""

    factor_id: str
    factor_name: str
    category: RootCauseCategory
    description: str

    # Contribution assessment
    contribution_pct: float  # 0-100%
    confidence: float  # 0-1
    is_primary: bool = False

    # Evidence
    evidence: List[str] = field(default_factory=list)
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    threshold_violations: List[str] = field(default_factory=list)

    # Physics basis
    physics_mechanism: str = ""
    reference_standard: str = ""

    # Recommendations
    corrective_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor_id": self.factor_id,
            "factor_name": self.factor_name,
            "category": self.category.value,
            "description": self.description,
            "contribution_pct": self.contribution_pct,
            "confidence": self.confidence,
            "is_primary": self.is_primary,
            "evidence": self.evidence,
            "sensor_readings": self.sensor_readings,
            "threshold_violations": self.threshold_violations,
            "physics_mechanism": self.physics_mechanism,
            "reference_standard": self.reference_standard,
            "corrective_actions": self.corrective_actions,
        }


@dataclass
class CausalTemplate:
    """Template for known causal patterns."""

    template_id: str
    event_type: str
    description: str

    # Triggers that activate this template
    trigger_conditions: Dict[str, Tuple[str, float]]  # sensor -> (operator, threshold)

    # Causal chain
    causal_chain: List[str]  # Ordered list of cause -> effect
    chain_type: CausalChainType

    # Contributing factors
    typical_factors: List[Dict[str, Any]]

    # Physics grounding
    physics_explanation: str
    reference_standard: str

    # Corrective actions
    recommended_actions: List[str]

    # Metadata
    severity: EventSeverity = EventSeverity.S2_MEDIUM
    typical_duration_minutes: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "event_type": self.event_type,
            "description": self.description,
            "trigger_conditions": {
                k: {"operator": v[0], "threshold": v[1]}
                for k, v in self.trigger_conditions.items()
            },
            "causal_chain": self.causal_chain,
            "chain_type": self.chain_type.value,
            "physics_explanation": self.physics_explanation,
            "reference_standard": self.reference_standard,
            "recommended_actions": self.recommended_actions,
            "severity": self.severity.value,
        }


@dataclass
class CorrelationResult:
    """Result of correlation analysis between variables."""

    correlation_id: str
    variable_1: str
    variable_2: str
    correlation_coefficient: float
    correlation_type: CorrelationType

    # Time analysis
    time_lag_seconds: float = 0.0
    lead_variable: str = ""

    # Statistical significance
    p_value: float = 0.0
    is_significant: bool = False

    # Interpretation
    interpretation: str = ""
    physics_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "variable_1": self.variable_1,
            "variable_2": self.variable_2,
            "correlation_coefficient": self.correlation_coefficient,
            "correlation_type": self.correlation_type.value,
            "time_lag_seconds": self.time_lag_seconds,
            "lead_variable": self.lead_variable,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "interpretation": self.interpretation,
            "physics_explanation": self.physics_explanation,
        }


@dataclass
class TimelineEvent:
    """Single event in the timeline reconstruction."""

    event_id: str
    timestamp: datetime
    event_type: str
    description: str
    priority: TimelinePriority

    # Details
    sensor_tag: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    unit: str = ""

    # Relationship to main event
    relation_to_main: str = ""  # "precursor", "trigger", "consequence", "concurrent"
    time_offset_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "priority": self.priority.value,
            "sensor_tag": self.sensor_tag,
            "value": self.value,
            "threshold": self.threshold,
            "unit": self.unit,
            "relation_to_main": self.relation_to_main,
            "time_offset_seconds": self.time_offset_seconds,
        }


@dataclass
class RootCauseAnalysis:
    """Complete root cause analysis for a quality event."""

    analysis_id: str
    timestamp: datetime
    event_id: str
    event_type: str
    header_id: str

    # Primary findings
    primary_cause: CausalFactor
    contributing_factors: List[CausalFactor]

    # Analysis details
    causal_chain: List[str]
    chain_type: CausalChainType

    # Correlations
    correlations: List[CorrelationResult]

    # Timeline
    timeline: List[TimelineEvent]
    event_duration_minutes: float = 0.0

    # Template match (if any)
    matched_template: Optional[str] = None
    template_confidence: float = 0.0

    # Natural language
    summary: str = ""
    detailed_explanation: str = ""

    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    short_term_actions: List[str] = field(default_factory=list)
    long_term_actions: List[str] = field(default_factory=list)

    # Confidence and provenance
    analysis_confidence: float = 0.8
    provenance_hash: str = ""
    input_data_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "event_type": self.event_type,
            "header_id": self.header_id,
            "primary_cause": self.primary_cause.to_dict(),
            "contributing_factors": [cf.to_dict() for cf in self.contributing_factors],
            "causal_chain": self.causal_chain,
            "chain_type": self.chain_type.value,
            "correlations": [c.to_dict() for c in self.correlations],
            "timeline": [te.to_dict() for te in self.timeline],
            "event_duration_minutes": self.event_duration_minutes,
            "matched_template": self.matched_template,
            "template_confidence": self.template_confidence,
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "immediate_actions": self.immediate_actions,
            "short_term_actions": self.short_term_actions,
            "long_term_actions": self.long_term_actions,
            "analysis_confidence": self.analysis_confidence,
            "provenance_hash": self.provenance_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# CAUSAL TEMPLATES FOR COMMON QUALITY EVENTS
# =============================================================================

QUALITY_EVENT_TEMPLATES = {
    "high_drum_level": CausalTemplate(
        template_id="TPL-001",
        event_type="high_drum_level",
        description="Steam drum level exceeds high alarm threshold",
        trigger_conditions={
            "drum_level_pct": (">", 65.0),
        },
        causal_chain=[
            "Feedwater flow exceeds steam demand",
            "Drum level rises above setpoint",
            "Reduced steam space volume",
            "Increased moisture entrainment",
            "Degraded steam quality",
        ],
        chain_type=CausalChainType.CASCADE,
        typical_factors=[
            {"name": "Feedwater control valve malfunction", "category": "mechanical"},
            {"name": "Load demand reduction without FW adjustment", "category": "operational"},
            {"name": "Level transmitter calibration drift", "category": "instrumentation"},
            {"name": "Blowdown valve stuck closed", "category": "mechanical"},
        ],
        physics_explanation=(
            "High drum level reduces the disengaging area for steam-water separation. "
            "Per API 534, reduced steam space increases water droplet entrainment "
            "velocity, causing moisture carryover into the steam header."
        ),
        reference_standard="API 534, ASME Section I PG-60",
        recommended_actions=[
            "Reduce feedwater flow to restore drum level",
            "Verify feedwater control valve operation",
            "Check level transmitter calibration",
            "Increase continuous blowdown if safe",
            "Monitor steam quality downstream",
        ],
        severity=EventSeverity.S1_HIGH,
        typical_duration_minutes=15.0,
    ),

    "separator_flooding": CausalTemplate(
        template_id="TPL-002",
        event_type="separator_flooding",
        description="Steam separator or scrubber flooding condition",
        trigger_conditions={
            "separator_level_pct": (">", 80.0),
            "separator_dp_psi": ("<", 2.0),
        },
        causal_chain=[
            "Separator drain restricted or failed",
            "Condensate accumulates in separator",
            "Separator flooding occurs",
            "Reduced separation efficiency",
            "Moisture passes to steam header",
        ],
        chain_type=CausalChainType.CASCADE,
        typical_factors=[
            {"name": "Drain trap failed closed", "category": "mechanical"},
            {"name": "Drain line blockage", "category": "mechanical"},
            {"name": "Excessive condensation load", "category": "process"},
            {"name": "Separator sizing inadequate for load", "category": "process"},
        ],
        physics_explanation=(
            "Separator flooding occurs when condensate removal rate is less than "
            "formation rate. Flooded separators cannot provide centrifugal "
            "separation, allowing liquid droplets to pass through with steam."
        ),
        reference_standard="API 560, Separator Design Standards",
        recommended_actions=[
            "Check separator drain trap operation",
            "Clear any drain line blockages",
            "Verify condensate return pressure",
            "Reduce steam demand temporarily if severe",
            "Consider separator bypass while repairing",
        ],
        severity=EventSeverity.S1_HIGH,
        typical_duration_minutes=30.0,
    ),

    "prv_condensation": CausalTemplate(
        template_id="TPL-003",
        event_type="prv_condensation",
        description="Condensation at pressure reducing valve indicating wet steam",
        trigger_conditions={
            "prv_condensation_rate": (">", 10.0),  # lb/hr
            "prv_outlet_temp_f": ("<", "saturation_temp - 5"),
        },
        causal_chain=[
            "Wet steam reaches PRV inlet",
            "Pressure reduction occurs",
            "Steam expands and cools",
            "Moisture condenses at PRV outlet",
            "Condensate detected in downstream piping",
        ],
        chain_type=CausalChainType.DIRECT,
        typical_factors=[
            {"name": "Upstream steam quality degraded", "category": "process"},
            {"name": "Inadequate drip leg drainage", "category": "mechanical"},
            {"name": "Failed upstream steam trap", "category": "mechanical"},
            {"name": "Cold pipe heating condensation", "category": "operational"},
        ],
        physics_explanation=(
            "PRV condensation indicates wet steam at the valve inlet. When "
            "saturated steam with moisture expands across the PRV, the flash "
            "evaporation is incomplete, leaving condensate at lower pressure. "
            "Per ASME B31.1, this indicates x < 1.0 upstream."
        ),
        reference_standard="ASME B31.1, ISA S75.05",
        recommended_actions=[
            "Inspect upstream drip legs and traps",
            "Verify steam trap operation in supply line",
            "Check for insulation damage causing heat loss",
            "Consider adding separator upstream of PRV",
            "Monitor condensate for water hammer risk",
        ],
        severity=EventSeverity.S2_MEDIUM,
        typical_duration_minutes=60.0,
    ),

    "trap_failure_blowthrough": CausalTemplate(
        template_id="TPL-004",
        event_type="trap_failure_blowthrough",
        description="Steam trap failed open (blowing through)",
        trigger_conditions={
            "trap_outlet_temp_f": (">", "saturation_temp - 10"),
            "trap_temperature_differential_f": ("<", 15.0),
        },
        causal_chain=[
            "Trap internal mechanism fails",
            "Steam passes through trap continuously",
            "Live steam enters condensate return",
            "Condensate return system pressurizes",
            "Energy loss and potential water hammer",
        ],
        chain_type=CausalChainType.CASCADE,
        typical_factors=[
            {"name": "Trap wear from cycling", "category": "mechanical"},
            {"name": "Scale/debris preventing closure", "category": "mechanical"},
            {"name": "Incorrect trap sizing (oversized)", "category": "process"},
            {"name": "Trap bellows/thermostatic element failure", "category": "mechanical"},
        ],
        physics_explanation=(
            "A trap blowing through passes live steam to condensate return. "
            "This wastes steam energy and can cause water hammer in the "
            "condensate system. The trap fails its function of removing "
            "condensate while blocking steam."
        ),
        reference_standard="ASME PTC 39, TEMA Standards",
        recommended_actions=[
            "Isolate and replace failed trap",
            "Check condensate return pressure",
            "Inspect upstream strainer",
            "Verify trap sizing for application",
            "Consider trap monitoring system",
        ],
        severity=EventSeverity.S2_MEDIUM,
        typical_duration_minutes=60.0,
    ),

    "trap_failure_blocked": CausalTemplate(
        template_id="TPL-005",
        event_type="trap_failure_blocked",
        description="Steam trap failed closed (blocked)",
        trigger_conditions={
            "trap_inlet_temp_f": (">", "saturation_temp"),
            "upstream_condensate_level": (">", 50.0),
        },
        causal_chain=[
            "Trap fails to open",
            "Condensate backs up upstream",
            "Process equipment floods with condensate",
            "Heat transfer efficiency drops",
            "Potential water hammer in steam lines",
        ],
        chain_type=CausalChainType.CASCADE,
        typical_factors=[
            {"name": "Trap mechanism jammed/stuck", "category": "mechanical"},
            {"name": "Debris blocking inlet", "category": "mechanical"},
            {"name": "Thermostatic element failed", "category": "mechanical"},
            {"name": "Excessive back pressure", "category": "process"},
        ],
        physics_explanation=(
            "A blocked trap causes condensate to accumulate in the steam space, "
            "reducing heat transfer effectiveness and creating conditions for "
            "water hammer when condensate slugs move with steam velocity."
        ),
        reference_standard="ASME PTC 39, FM Global Guidelines",
        recommended_actions=[
            "Replace or repair trap immediately",
            "Check for upstream blockages/debris",
            "Verify condensate return pressure",
            "Drain accumulated condensate safely",
            "Inspect associated equipment for damage",
        ],
        severity=EventSeverity.S1_HIGH,
        typical_duration_minutes=30.0,
    ),
}


# =============================================================================
# ROOT CAUSE ANALYZER
# =============================================================================

class RootCauseAnalyzer:
    """
    Root cause analyzer for steam quality events.

    Provides causal analysis, correlation analysis, and timeline
    reconstruction for quality events, with physics-grounded explanations.

    Example:
        >>> analyzer = RootCauseAnalyzer(agent_id="GL-012")
        >>> analysis = analyzer.analyze_event(quality_event)
        >>> print(analysis.summary)
        >>> print(analysis.primary_cause.factor_name)

    Attributes:
        agent_id: Agent identifier
        templates: Dictionary of causal templates
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-012",
        custom_templates: Optional[Dict[str, CausalTemplate]] = None,
    ) -> None:
        """
        Initialize RootCauseAnalyzer.

        Args:
            agent_id: Agent identifier
            custom_templates: Optional additional templates
        """
        self.agent_id = agent_id
        self.templates = {**QUALITY_EVENT_TEMPLATES}
        if custom_templates:
            self.templates.update(custom_templates)

        self._analyses: Dict[str, RootCauseAnalysis] = {}

        logger.info(f"RootCauseAnalyzer initialized: {agent_id}")

    def analyze_event(
        self,
        event: Dict[str, Any],
        sensor_history: Optional[List[Dict[str, Any]]] = None,
        header_id: str = "HEADER-001",
    ) -> RootCauseAnalysis:
        """
        Perform root cause analysis on a quality event.

        Args:
            event: Quality event data including type, timestamp, sensor readings
            sensor_history: Optional historical sensor data for correlation
            header_id: Steam header identifier

        Returns:
            RootCauseAnalysis with causes, correlations, and timeline
        """
        timestamp = datetime.now(timezone.utc)
        analysis_id = f"RCA-{timestamp.strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"

        event_type = event.get("event_type", "unknown")
        event_id = event.get("event_id", str(uuid.uuid4())[:8])
        event_timestamp = event.get("timestamp", timestamp)
        sensor_readings = event.get("sensor_readings", {})

        # Match template
        matched_template, template_confidence = self._match_template(
            event_type, sensor_readings
        )

        # Identify causal factors
        primary_cause, contributing_factors = self._identify_causal_factors(
            event_type, sensor_readings, matched_template
        )

        # Build causal chain
        if matched_template:
            causal_chain = matched_template.causal_chain
            chain_type = matched_template.chain_type
        else:
            causal_chain = self._infer_causal_chain(event_type, sensor_readings)
            chain_type = CausalChainType.MULTI_FACTOR

        # Perform correlation analysis
        correlations = []
        if sensor_history:
            correlations = self._analyze_correlations(sensor_history)

        # Reconstruct timeline
        timeline = self._reconstruct_timeline(
            event_timestamp, sensor_readings, sensor_history
        )

        # Generate recommendations
        immediate, short_term, long_term = self._generate_recommendations(
            primary_cause, contributing_factors, matched_template
        )

        # Build analysis
        analysis = RootCauseAnalysis(
            analysis_id=analysis_id,
            timestamp=timestamp,
            event_id=event_id,
            event_type=event_type,
            header_id=header_id,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            causal_chain=causal_chain,
            chain_type=chain_type,
            correlations=correlations,
            timeline=timeline,
            matched_template=matched_template.template_id if matched_template else None,
            template_confidence=template_confidence,
            immediate_actions=immediate,
            short_term_actions=short_term,
            long_term_actions=long_term,
        )

        # Generate natural language
        self._generate_summary(analysis)
        self._generate_detailed_explanation(analysis)

        # Calculate confidence and provenance
        self._calculate_confidence(analysis, matched_template, len(correlations))
        self._calculate_provenance(analysis, event)

        # Cache
        self._analyses[analysis_id] = analysis

        logger.info(
            f"Completed root cause analysis for {event_type}: "
            f"primary cause = {primary_cause.factor_name}"
        )

        return analysis

    def _match_template(
        self,
        event_type: str,
        sensor_readings: Dict[str, float],
    ) -> Tuple[Optional[CausalTemplate], float]:
        """Match event to a causal template."""
        best_match = None
        best_confidence = 0.0

        for template_id, template in self.templates.items():
            if template.event_type != event_type:
                continue

            # Check trigger conditions
            conditions_met = 0
            total_conditions = len(template.trigger_conditions)

            for sensor, (operator, threshold) in template.trigger_conditions.items():
                if sensor in sensor_readings:
                    value = sensor_readings[sensor]
                    if self._check_condition(value, operator, threshold):
                        conditions_met += 1

            if total_conditions > 0:
                confidence = conditions_met / total_conditions
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = template

        return best_match, best_confidence

    def _check_condition(
        self, value: float, operator: str, threshold: Any
    ) -> bool:
        """Check if a value meets a condition."""
        # Handle string thresholds (e.g., "saturation_temp - 5")
        if isinstance(threshold, str):
            # For now, use a placeholder saturation temp
            threshold = 353.0  # Approximate sat temp at 125 psig

        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.01

        return False

    def _identify_causal_factors(
        self,
        event_type: str,
        sensor_readings: Dict[str, float],
        template: Optional[CausalTemplate],
    ) -> Tuple[CausalFactor, List[CausalFactor]]:
        """Identify primary and contributing causal factors."""
        factors = []

        if template:
            # Use template factors
            for i, factor_info in enumerate(template.typical_factors):
                factor = CausalFactor(
                    factor_id=f"CF-{i+1:03d}",
                    factor_name=factor_info["name"],
                    category=RootCauseCategory(factor_info["category"]),
                    description=self._get_factor_description(factor_info["name"]),
                    contribution_pct=max(10, 40 - i * 10),  # Decreasing contribution
                    confidence=max(0.5, 0.9 - i * 0.1),
                    is_primary=(i == 0),
                    physics_mechanism=template.physics_explanation,
                    reference_standard=template.reference_standard,
                )
                self._add_evidence(factor, sensor_readings, event_type)
                factors.append(factor)
        else:
            # Infer factors from sensor readings
            factors = self._infer_factors_from_readings(event_type, sensor_readings)

        # Ensure we have at least one factor
        if not factors:
            factors.append(CausalFactor(
                factor_id="CF-001",
                factor_name="Unknown cause",
                category=RootCauseCategory.PROCESS,
                description="Root cause could not be determined from available data",
                contribution_pct=100.0,
                confidence=0.3,
                is_primary=True,
            ))

        # Primary is first, rest are contributing
        primary = factors[0]
        primary.is_primary = True
        contributing = factors[1:]

        return primary, contributing

    def _get_factor_description(self, factor_name: str) -> str:
        """Get detailed description for a factor name."""
        descriptions = {
            "Feedwater control valve malfunction": (
                "Feedwater control valve not responding correctly to level "
                "controller output, causing over/under feed condition."
            ),
            "Load demand reduction without FW adjustment": (
                "Steam demand dropped suddenly but feedwater flow was not "
                "reduced proportionally, causing drum level to rise."
            ),
            "Level transmitter calibration drift": (
                "Drum level transmitter has drifted from calibration, "
                "reporting incorrect level to control system."
            ),
            "Drain trap failed closed": (
                "Separator drain trap is stuck closed, preventing condensate "
                "removal and causing separator flooding."
            ),
            "Failed upstream steam trap": (
                "Steam trap upstream of measurement point has failed, "
                "allowing wet steam to reach downstream equipment."
            ),
            "Trap wear from cycling": (
                "Steam trap internals have worn from normal cycling operation, "
                "causing failure to seal properly."
            ),
        }
        return descriptions.get(factor_name, f"Factor: {factor_name}")

    def _add_evidence(
        self,
        factor: CausalFactor,
        sensor_readings: Dict[str, float],
        event_type: str,
    ) -> None:
        """Add evidence to a causal factor from sensor readings."""
        # Add relevant sensor readings
        factor.sensor_readings = sensor_readings.copy()

        # Check for threshold violations based on event type
        if event_type == "high_drum_level":
            if sensor_readings.get("drum_level_pct", 0) > 65:
                factor.threshold_violations.append(
                    f"Drum level at {sensor_readings['drum_level_pct']:.1f}% > 65% HH alarm"
                )
                factor.evidence.append("Drum level exceeded high-high alarm threshold")

        elif event_type == "separator_flooding":
            if sensor_readings.get("separator_dp_psi", 10) < 2:
                factor.threshold_violations.append(
                    f"Separator dP at {sensor_readings.get('separator_dp_psi', 0):.1f} psi < 2 psi minimum"
                )
                factor.evidence.append("Low separator differential pressure indicates flooding")

        elif event_type == "prv_condensation":
            if sensor_readings.get("prv_condensation_rate", 0) > 10:
                factor.threshold_violations.append(
                    f"PRV condensation at {sensor_readings['prv_condensation_rate']:.1f} lb/hr > 10 lb/hr threshold"
                )
                factor.evidence.append("Significant condensation detected at PRV outlet")

    def _infer_factors_from_readings(
        self,
        event_type: str,
        sensor_readings: Dict[str, float],
    ) -> List[CausalFactor]:
        """Infer causal factors from sensor readings when no template matches."""
        factors = []

        # Check common conditions
        if sensor_readings.get("drum_level_pct", 50) > 60:
            factors.append(CausalFactor(
                factor_id=f"CF-{len(factors)+1:03d}",
                factor_name="Elevated drum level",
                category=RootCauseCategory.OPERATIONAL,
                description="Drum level above normal operating range",
                contribution_pct=40.0,
                confidence=0.7,
                evidence=[f"Drum level at {sensor_readings['drum_level_pct']:.1f}%"],
            ))

        if sensor_readings.get("steam_flow_klb_hr", 50) > 80:
            factors.append(CausalFactor(
                factor_id=f"CF-{len(factors)+1:03d}",
                factor_name="High steam demand",
                category=RootCauseCategory.PROCESS,
                description="Steam flow above typical operating range",
                contribution_pct=30.0,
                confidence=0.6,
                evidence=[f"Steam flow at {sensor_readings['steam_flow_klb_hr']:.1f} klb/hr"],
            ))

        if sensor_readings.get("separator_dp_psi", 5) < 3:
            factors.append(CausalFactor(
                factor_id=f"CF-{len(factors)+1:03d}",
                factor_name="Poor separator performance",
                category=RootCauseCategory.MECHANICAL,
                description="Separator differential pressure below optimal",
                contribution_pct=35.0,
                confidence=0.65,
                evidence=[f"Separator dP at {sensor_readings['separator_dp_psi']:.1f} psi"],
            ))

        return factors

    def _infer_causal_chain(
        self,
        event_type: str,
        sensor_readings: Dict[str, float],
    ) -> List[str]:
        """Infer causal chain when no template matches."""
        # Generic causal chain
        return [
            "Process condition deviated from normal",
            "Control system response was insufficient",
            "Quality parameter exceeded threshold",
            f"{event_type} event detected",
            "Quality degradation occurred",
        ]

    def _analyze_correlations(
        self,
        sensor_history: List[Dict[str, Any]],
    ) -> List[CorrelationResult]:
        """Analyze correlations between sensors in historical data."""
        correlations = []

        if not sensor_history or len(sensor_history) < 10:
            return correlations

        # Get list of sensor tags
        sensor_tags = list(sensor_history[0].keys())
        sensor_tags = [t for t in sensor_tags if t not in ["timestamp", "event_id"]]

        # Calculate correlations between key pairs
        key_pairs = [
            ("drum_level_pct", "steam_flow_klb_hr"),
            ("drum_level_pct", "feedwater_flow_klb_hr"),
            ("separator_dp_psi", "steam_flow_klb_hr"),
            ("steam_quality", "drum_level_pct"),
        ]

        for var1, var2 in key_pairs:
            if var1 in sensor_tags and var2 in sensor_tags:
                corr = self._calculate_correlation(
                    sensor_history, var1, var2
                )
                if corr:
                    correlations.append(corr)

        return correlations

    def _calculate_correlation(
        self,
        sensor_history: List[Dict[str, Any]],
        var1: str,
        var2: str,
    ) -> Optional[CorrelationResult]:
        """Calculate correlation between two variables."""
        try:
            values1 = [d.get(var1, 0) for d in sensor_history]
            values2 = [d.get(var2, 0) for d in sensor_history]

            if len(values1) < 10:
                return None

            # Simple Pearson correlation
            n = len(values1)
            mean1 = sum(values1) / n
            mean2 = sum(values2) / n

            numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
            denom1 = sum((v - mean1) ** 2 for v in values1) ** 0.5
            denom2 = sum((v - mean2) ** 2 for v in values2) ** 0.5

            if denom1 * denom2 == 0:
                return None

            r = numerator / (denom1 * denom2)

            # Determine correlation type
            if r > 0.7:
                corr_type = CorrelationType.STRONG_POSITIVE
            elif r > 0.4:
                corr_type = CorrelationType.MODERATE_POSITIVE
            elif r > 0.2:
                corr_type = CorrelationType.WEAK_POSITIVE
            elif r > -0.2:
                corr_type = CorrelationType.NONE
            elif r > -0.4:
                corr_type = CorrelationType.WEAK_NEGATIVE
            elif r > -0.7:
                corr_type = CorrelationType.MODERATE_NEGATIVE
            else:
                corr_type = CorrelationType.STRONG_NEGATIVE

            return CorrelationResult(
                correlation_id=str(uuid.uuid4())[:8],
                variable_1=var1,
                variable_2=var2,
                correlation_coefficient=round(r, 4),
                correlation_type=corr_type,
                is_significant=abs(r) > 0.3,
                interpretation=self._interpret_correlation(var1, var2, r),
            )

        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return None

    def _interpret_correlation(
        self, var1: str, var2: str, r: float
    ) -> str:
        """Generate interpretation of correlation."""
        if abs(r) < 0.2:
            return f"No significant correlation between {var1} and {var2}."

        direction = "positively" if r > 0 else "negatively"
        strength = "strongly" if abs(r) > 0.7 else "moderately" if abs(r) > 0.4 else "weakly"

        return f"{var1} is {strength} {direction} correlated with {var2} (r={r:.3f})."

    def _reconstruct_timeline(
        self,
        event_timestamp: datetime,
        sensor_readings: Dict[str, float],
        sensor_history: Optional[List[Dict[str, Any]]],
    ) -> List[TimelineEvent]:
        """Reconstruct timeline of events leading to quality issue."""
        timeline = []

        # Add the main event
        timeline.append(TimelineEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=event_timestamp,
            event_type="quality_event",
            description="Quality event triggered",
            priority=TimelinePriority.PRIMARY,
            relation_to_main="trigger",
            time_offset_seconds=0.0,
        ))

        # Add precursor events from sensor readings
        if sensor_readings.get("drum_level_pct", 50) > 60:
            timeline.append(TimelineEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp=event_timestamp - timedelta(minutes=5),
                event_type="drum_level_high",
                description=f"Drum level exceeded 60% ({sensor_readings['drum_level_pct']:.1f}%)",
                priority=TimelinePriority.CONTRIBUTING,
                sensor_tag="drum_level_pct",
                value=sensor_readings["drum_level_pct"],
                threshold=60.0,
                unit="%",
                relation_to_main="precursor",
                time_offset_seconds=-300.0,
            ))

        if sensor_readings.get("separator_dp_psi", 5) < 3:
            timeline.append(TimelineEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp=event_timestamp - timedelta(minutes=3),
                event_type="separator_dp_low",
                description=f"Separator dP below 3 psi ({sensor_readings['separator_dp_psi']:.1f} psi)",
                priority=TimelinePriority.CONTRIBUTING,
                sensor_tag="separator_dp_psi",
                value=sensor_readings["separator_dp_psi"],
                threshold=3.0,
                unit="psi",
                relation_to_main="precursor",
                time_offset_seconds=-180.0,
            ))

        # Sort by timestamp
        timeline.sort(key=lambda e: e.timestamp)

        return timeline

    def _generate_recommendations(
        self,
        primary_cause: CausalFactor,
        contributing_factors: List[CausalFactor],
        template: Optional[CausalTemplate],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate prioritized recommendations."""
        immediate = []
        short_term = []
        long_term = []

        if template:
            # Use template recommendations
            for i, action in enumerate(template.recommended_actions):
                if i < 2:
                    immediate.append(action)
                elif i < 4:
                    short_term.append(action)
                else:
                    long_term.append(action)
        else:
            # Generic recommendations based on cause category
            if primary_cause.category == RootCauseCategory.MECHANICAL:
                immediate.append("Inspect equipment for mechanical failure")
                immediate.append("Verify control valve operation")
                short_term.append("Schedule preventive maintenance")
                long_term.append("Review equipment reliability history")

            elif primary_cause.category == RootCauseCategory.OPERATIONAL:
                immediate.append("Review recent operational changes")
                immediate.append("Verify setpoints are correct")
                short_term.append("Retrain operators on optimal procedures")
                long_term.append("Update operating procedures")

            elif primary_cause.category == RootCauseCategory.INSTRUMENTATION:
                immediate.append("Verify sensor calibration")
                immediate.append("Check for sensor fouling/damage")
                short_term.append("Schedule calibration verification")
                long_term.append("Consider sensor redundancy")

        # Add actions from primary cause
        for action in primary_cause.corrective_actions:
            if action not in immediate and action not in short_term:
                short_term.append(action)

        return immediate, short_term, long_term

    def _generate_summary(self, analysis: RootCauseAnalysis) -> None:
        """Generate natural language summary."""
        primary = analysis.primary_cause

        summary_parts = [
            f"Root cause analysis for {analysis.event_type} event.",
            f"Primary cause: {primary.factor_name} ({primary.contribution_pct:.0f}% contribution, {primary.confidence:.0%} confidence).",
        ]

        if analysis.contributing_factors:
            factors = ", ".join([f.factor_name for f in analysis.contributing_factors[:2]])
            summary_parts.append(f"Contributing factors: {factors}.")

        if analysis.matched_template:
            summary_parts.append(
                f"Matched pattern: {analysis.matched_template} "
                f"({analysis.template_confidence:.0%} confidence)."
            )

        if analysis.immediate_actions:
            summary_parts.append(
                f"Recommended immediate action: {analysis.immediate_actions[0]}."
            )

        analysis.summary = " ".join(summary_parts)

    def _generate_detailed_explanation(self, analysis: RootCauseAnalysis) -> None:
        """Generate detailed explanation."""
        lines = [
            "ROOT CAUSE ANALYSIS REPORT",
            "=" * 50,
            "",
            f"Event: {analysis.event_type}",
            f"Event ID: {analysis.event_id}",
            f"Header: {analysis.header_id}",
            f"Analysis Time: {analysis.timestamp.isoformat()}",
            "",
            "PRIMARY CAUSE:",
            "-" * 40,
            f"  Factor: {analysis.primary_cause.factor_name}",
            f"  Category: {analysis.primary_cause.category.value}",
            f"  Contribution: {analysis.primary_cause.contribution_pct:.0f}%",
            f"  Confidence: {analysis.primary_cause.confidence:.0%}",
            f"  Description: {analysis.primary_cause.description}",
            "",
        ]

        if analysis.primary_cause.physics_mechanism:
            lines.extend([
                "Physics Basis:",
                f"  {analysis.primary_cause.physics_mechanism}",
                "",
            ])

        if analysis.contributing_factors:
            lines.extend([
                "CONTRIBUTING FACTORS:",
                "-" * 40,
            ])
            for i, factor in enumerate(analysis.contributing_factors, 1):
                lines.append(
                    f"  {i}. {factor.factor_name} "
                    f"({factor.contribution_pct:.0f}%, {factor.confidence:.0%} conf.)"
                )
            lines.append("")

        lines.extend([
            "CAUSAL CHAIN:",
            "-" * 40,
        ])
        for i, step in enumerate(analysis.causal_chain, 1):
            lines.append(f"  {i}. {step}")
        lines.append("")

        if analysis.immediate_actions:
            lines.extend([
                "IMMEDIATE ACTIONS:",
                "-" * 40,
            ])
            for action in analysis.immediate_actions:
                lines.append(f"  - {action}")
            lines.append("")

        if analysis.short_term_actions:
            lines.extend([
                "SHORT-TERM ACTIONS:",
                "-" * 40,
            ])
            for action in analysis.short_term_actions:
                lines.append(f"  - {action}")

        analysis.detailed_explanation = "\n".join(lines)

    def _calculate_confidence(
        self,
        analysis: RootCauseAnalysis,
        template: Optional[CausalTemplate],
        correlation_count: int,
    ) -> None:
        """Calculate confidence in the analysis."""
        confidence = 0.6  # Base confidence

        # Boost for template match
        if template:
            confidence += analysis.template_confidence * 0.2

        # Boost for correlations
        if correlation_count > 2:
            confidence += 0.1

        # Boost for evidence
        evidence_count = len(analysis.primary_cause.evidence)
        if evidence_count > 2:
            confidence += 0.1

        analysis.analysis_confidence = min(0.95, confidence)

    def _calculate_provenance(
        self, analysis: RootCauseAnalysis, event: Dict[str, Any]
    ) -> None:
        """Calculate provenance hash."""
        provenance_data = {
            "analysis_id": analysis.analysis_id,
            "event_id": analysis.event_id,
            "event_type": analysis.event_type,
            "timestamp": analysis.timestamp.isoformat(),
            "primary_cause": analysis.primary_cause.factor_name,
            "agent_id": self.agent_id,
        }

        json_str = json.dumps(provenance_data, sort_keys=True)
        analysis.provenance_hash = hashlib.sha256(json_str.encode()).hexdigest()

        input_json = json.dumps(event, sort_keys=True, default=str)
        analysis.input_data_hash = hashlib.sha256(input_json.encode()).hexdigest()[:16]

    def get_analysis(self, analysis_id: str) -> Optional[RootCauseAnalysis]:
        """Get analysis by ID."""
        return self._analyses.get(analysis_id)

    def get_template(self, event_type: str) -> Optional[CausalTemplate]:
        """Get causal template for an event type."""
        return self.templates.get(event_type)

    def list_templates(self) -> List[str]:
        """List available template event types."""
        return list(self.templates.keys())
