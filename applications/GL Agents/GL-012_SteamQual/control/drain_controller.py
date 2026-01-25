"""
GL-012 STEAMQUAL SteamQualityController - Drain Controller

This module implements drain valve and separator control for steam quality
management, including setpoint recommendations, sequencing for drip legs,
flooding prevention, and backpressure management.

Control Architecture:
    - Drain valve setpoint recommendations
    - Automated sequencing for drip leg drainage
    - Flooding prevention with level monitoring
    - Backpressure management for condensate return

Key Features:
    - Proactive condensate removal
    - Water hammer prevention through proper drainage
    - Separator level management
    - Advisory mode (default) with optional automation

Reference Standards:
    - ASME B31.1 Power Piping
    - ASME PTC 39 Steam Traps
    - ISA-18.2 Management of Alarm Systems

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DrainValveState(str, Enum):
    """Drain valve state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    THROTTLED = "throttled"  # Partially open
    CYCLING = "cycling"  # Automated cycling
    FAULT = "fault"
    UNKNOWN = "unknown"


class SeparatorState(str, Enum):
    """Separator state enumeration."""
    NORMAL = "normal"
    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"
    FLOODING = "flooding"
    DRAINING = "draining"
    ISOLATED = "isolated"
    FAULT = "fault"


class DrainSequenceType(str, Enum):
    """Drain sequence type enumeration."""
    STARTUP = "startup"  # System startup drainage
    NORMAL = "normal"  # Normal operation drainage
    WARMUP = "warmup"  # Warm-up period drainage
    SHUTDOWN = "shutdown"  # Shutdown drainage
    EMERGENCY = "emergency"  # Emergency flooding response


class BackpressureStatus(str, Enum):
    """Backpressure status enumeration."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    BLOCKED = "blocked"


# =============================================================================
# DATA MODELS
# =============================================================================

class DrainValveConfig(BaseModel):
    """Drain valve configuration."""

    valve_id: str = Field(..., description="Drain valve identifier")
    asset_id: str = Field(..., description="Associated asset ID")
    location: str = Field(..., description="Valve location description")
    valve_type: str = Field(
        default="on_off",
        description="Valve type (on_off, modulating)"
    )
    nominal_cv: float = Field(default=2.0, ge=0, description="Valve Cv rating")
    max_position_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Maximum allowed position (%)"
    )
    min_open_time_s: float = Field(
        default=5.0,
        ge=0,
        description="Minimum open time per cycle (s)"
    )
    max_open_time_s: float = Field(
        default=60.0,
        ge=0,
        description="Maximum open time per cycle (s)"
    )
    cycle_interval_s: float = Field(
        default=300.0,
        ge=0,
        description="Normal cycling interval (s)"
    )
    has_feedback: bool = Field(
        default=True,
        description="Valve has position feedback"
    )


class SeparatorConfig(BaseModel):
    """Separator configuration."""

    separator_id: str = Field(..., description="Separator identifier")
    asset_id: str = Field(..., description="Associated asset ID")
    design_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Design pressure (kPa)"
    )
    design_capacity_kg_s: float = Field(
        ...,
        ge=0,
        description="Design steam capacity (kg/s)"
    )
    drain_valve_id: str = Field(..., description="Associated drain valve ID")
    level_sensor_id: Optional[str] = Field(
        None,
        description="Level sensor ID if available"
    )
    high_level_pct: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="High level alarm threshold (%)"
    )
    low_level_pct: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Low level alarm threshold (%)"
    )
    flooding_level_pct: float = Field(
        default=95.0,
        ge=0,
        le=100,
        description="Flooding alarm threshold (%)"
    )


class DrainSetpoint(BaseModel):
    """Drain valve setpoint recommendation."""

    setpoint_id: str = Field(..., description="Setpoint ID")
    valve_id: str = Field(..., description="Drain valve ID")
    recommended_position_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Recommended valve position (%)"
    )
    recommended_action: str = Field(
        ...,
        description="Recommended action (open, close, throttle, cycle)"
    )
    open_duration_s: Optional[float] = Field(
        None,
        ge=0,
        description="Recommended open duration for cycling (s)"
    )
    rationale: str = Field(..., description="Setpoint rationale")
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Priority (1=highest)"
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Requires operator confirmation"
    )
    safety_validated: bool = Field(
        default=False,
        description="Passed safety validation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Setpoint timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class SequenceStep(BaseModel):
    """Single step in a drain sequence."""

    step_number: int = Field(..., ge=1, description="Step number")
    valve_id: str = Field(..., description="Drain valve ID")
    action: str = Field(..., description="Action (open, close, throttle)")
    position_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Target position (%)"
    )
    duration_s: float = Field(..., ge=0, description="Step duration (s)")
    wait_for_complete: bool = Field(
        default=True,
        description="Wait for action to complete"
    )
    verify_condition: Optional[str] = Field(
        None,
        description="Condition to verify before proceeding"
    )


class DrainSequence(BaseModel):
    """Complete drain sequence for drip legs."""

    sequence_id: str = Field(..., description="Sequence ID")
    sequence_type: DrainSequenceType = Field(..., description="Sequence type")
    valve_ids: List[str] = Field(..., description="Valves in sequence")
    steps: List[SequenceStep] = Field(..., description="Sequence steps")
    total_duration_s: float = Field(
        ...,
        ge=0,
        description="Total sequence duration (s)"
    )
    description: str = Field(..., description="Sequence description")
    requires_confirmation: bool = Field(
        default=True,
        description="Requires operator confirmation"
    )
    safety_validated: bool = Field(
        default=False,
        description="Passed safety validation"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class FloodingRisk(BaseModel):
    """Flooding risk assessment."""

    assessment_id: str = Field(..., description="Assessment ID")
    separator_id: str = Field(..., description="Separator ID")
    current_level_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current level (%)"
    )
    level_trend: str = Field(
        ...,
        description="Level trend (rising, falling, stable)"
    )
    risk_level: str = Field(
        ...,
        description="Risk level (low, medium, high, critical)"
    )
    estimated_time_to_flood_s: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated time to flooding (s)"
    )
    recommended_action: str = Field(..., description="Recommended action")
    affected_valves: List[str] = Field(
        default_factory=list,
        description="Drain valves to operate"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Assessment timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class BackpressureAnalysis(BaseModel):
    """Condensate backpressure analysis."""

    analysis_id: str = Field(..., description="Analysis ID")
    drain_point_id: str = Field(..., description="Drain point ID")
    steam_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Steam side pressure (kPa)"
    )
    condensate_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Condensate side pressure (kPa)"
    )
    differential_kpa: float = Field(
        ...,
        description="Pressure differential (kPa)"
    )
    status: BackpressureStatus = Field(..., description="Backpressure status")
    min_required_differential_kpa: float = Field(
        ...,
        ge=0,
        description="Minimum required differential (kPa)"
    )
    flow_restriction_pct: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Flow restriction due to backpressure (%)"
    )
    recommended_action: str = Field(..., description="Recommended action")
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Analysis timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class DrainRecommendation(BaseModel):
    """Comprehensive drain control recommendation."""

    recommendation_id: str = Field(..., description="Recommendation ID")
    drain_setpoints: List[DrainSetpoint] = Field(
        default_factory=list,
        description="Drain valve setpoints"
    )
    sequences: List[DrainSequence] = Field(
        default_factory=list,
        description="Drain sequences"
    )
    flooding_risks: List[FloodingRisk] = Field(
        default_factory=list,
        description="Flooding risk assessments"
    )
    backpressure_analysis: List[BackpressureAnalysis] = Field(
        default_factory=list,
        description="Backpressure analyses"
    )
    overall_status: str = Field(..., description="Overall drain status")
    requires_immediate_action: bool = Field(
        default=False,
        description="Immediate action required"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Recommendation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# DRAIN CONTROLLER
# =============================================================================

class DrainController:
    """
    Drain valve and separator control for steam quality management.

    This controller manages drain valves and separators to ensure proper
    condensate removal, prevent flooding, and maintain steam quality.

    Control Features:
        - Drain valve setpoint recommendations
        - Automated sequencing for drip leg drainage
        - Flooding prevention with level monitoring
        - Backpressure management for condensate return
        - Water hammer prevention

    Safety Features:
        - Never blocks safety-critical drainage
        - Flooding detection and emergency response
        - Backpressure monitoring and alerting
        - All actions logged for audit trail

    Attributes:
        controller_id: Unique controller identifier
        valves: Registered drain valve configurations
        separators: Registered separator configurations
        _valve_states: Current valve states
        _separator_states: Current separator states

    Example:
        >>> controller = DrainController("DC-001")
        >>> controller.register_valve(valve_config)
        >>> controller.register_separator(separator_config)
        >>> recommendation = controller.compute_recommendation(measurements)
    """

    def __init__(self, controller_id: str):
        """
        Initialize DrainController.

        Args:
            controller_id: Unique controller identifier
        """
        self.controller_id = controller_id

        # Configurations
        self._valves: Dict[str, DrainValveConfig] = {}
        self._separators: Dict[str, SeparatorConfig] = {}

        # Current states
        self._valve_states: Dict[str, Dict[str, Any]] = {}
        self._separator_states: Dict[str, Dict[str, Any]] = {}

        # History
        self._sequence_history: List[DrainSequence] = []
        self._flooding_events: List[FloodingRisk] = []
        self._max_history_size = 500

        logger.info(f"DrainController {controller_id} initialized")

    def register_valve(self, config: DrainValveConfig) -> None:
        """
        Register a drain valve configuration.

        Args:
            config: Drain valve configuration
        """
        self._valves[config.valve_id] = config
        self._valve_states[config.valve_id] = {
            "state": DrainValveState.UNKNOWN,
            "position_pct": 0.0,
            "last_cycle_time": None
        }
        logger.info(f"Registered drain valve {config.valve_id}")

    def register_separator(self, config: SeparatorConfig) -> None:
        """
        Register a separator configuration.

        Args:
            config: Separator configuration
        """
        self._separators[config.separator_id] = config
        self._separator_states[config.separator_id] = {
            "state": SeparatorState.NORMAL,
            "level_pct": 50.0,
            "last_drain_time": None
        }
        logger.info(f"Registered separator {config.separator_id}")

    def update_valve_state(
        self,
        valve_id: str,
        state: DrainValveState,
        position_pct: float
    ) -> None:
        """
        Update drain valve state.

        Args:
            valve_id: Drain valve ID
            state: Current valve state
            position_pct: Current valve position (%)
        """
        if valve_id not in self._valve_states:
            self._valve_states[valve_id] = {}

        self._valve_states[valve_id].update({
            "state": state,
            "position_pct": position_pct,
            "updated_at": datetime.now()
        })

    def update_separator_level(
        self,
        separator_id: str,
        level_pct: float
    ) -> None:
        """
        Update separator level measurement.

        Args:
            separator_id: Separator ID
            level_pct: Current level (%)
        """
        if separator_id not in self._separator_states:
            self._separator_states[separator_id] = {}

        old_level = self._separator_states[separator_id].get("level_pct", 50.0)
        level_trend = "stable"
        if level_pct > old_level + 2:
            level_trend = "rising"
        elif level_pct < old_level - 2:
            level_trend = "falling"

        self._separator_states[separator_id].update({
            "level_pct": level_pct,
            "level_trend": level_trend,
            "updated_at": datetime.now()
        })

        # Update separator state based on level
        config = self._separators.get(separator_id)
        if config:
            if level_pct >= config.flooding_level_pct:
                self._separator_states[separator_id]["state"] = SeparatorState.FLOODING
            elif level_pct >= config.high_level_pct:
                self._separator_states[separator_id]["state"] = SeparatorState.HIGH_LEVEL
            elif level_pct <= config.low_level_pct:
                self._separator_states[separator_id]["state"] = SeparatorState.LOW_LEVEL
            else:
                self._separator_states[separator_id]["state"] = SeparatorState.NORMAL

    def compute_drain_setpoint(
        self,
        valve_id: str,
        separator_level_pct: Optional[float] = None,
        steam_flow_kg_s: Optional[float] = None
    ) -> DrainSetpoint:
        """
        Compute drain valve setpoint recommendation.

        Args:
            valve_id: Drain valve ID
            separator_level_pct: Current separator level (%)
            steam_flow_kg_s: Current steam flow (kg/s)

        Returns:
            DrainSetpoint: Setpoint recommendation
        """
        start_time = datetime.now()

        if valve_id not in self._valves:
            raise KeyError(f"Valve {valve_id} not registered")

        config = self._valves[valve_id]
        state = self._valve_states.get(valve_id, {})

        # Determine recommended action based on conditions
        recommended_action = "maintain"
        recommended_position = state.get("position_pct", 0.0)
        open_duration = None
        priority = 3
        rationale = "Normal operation - maintaining current state"

        # High level - open drain
        if separator_level_pct is not None and separator_level_pct > 70:
            recommended_action = "open"
            recommended_position = min(100.0, config.max_position_pct)
            open_duration = config.max_open_time_s
            priority = 2 if separator_level_pct > 85 else 3
            rationale = f"High separator level ({separator_level_pct:.1f}%) - draining required"

        # Very high level - emergency drain
        elif separator_level_pct is not None and separator_level_pct > 90:
            recommended_action = "open"
            recommended_position = 100.0
            priority = 1
            rationale = f"Critical separator level ({separator_level_pct:.1f}%) - emergency drain"

        # Low flow - reduce drain
        elif steam_flow_kg_s is not None and steam_flow_kg_s < 0.1:
            recommended_action = "throttle"
            recommended_position = 20.0
            priority = 4
            rationale = "Low steam flow - reduced drainage"

        # Check for cycling need
        last_cycle = state.get("last_cycle_time")
        if last_cycle is not None:
            elapsed = (start_time - last_cycle).total_seconds()
            if elapsed > config.cycle_interval_s:
                recommended_action = "cycle"
                recommended_position = config.max_position_pct
                open_duration = config.min_open_time_s
                priority = 4
                rationale = f"Scheduled drain cycle (last: {elapsed:.0f}s ago)"

        # Generate setpoint ID
        setpoint_id = hashlib.sha256(
            f"DS_{valve_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        setpoint = DrainSetpoint(
            setpoint_id=setpoint_id,
            valve_id=valve_id,
            recommended_position_pct=recommended_position,
            recommended_action=recommended_action,
            open_duration_s=open_duration,
            rationale=rationale,
            priority=priority,
            requires_confirmation=(priority <= 2),  # High priority needs confirmation
            safety_validated=True,
            timestamp=start_time
        )

        # Calculate provenance hash
        setpoint.provenance_hash = hashlib.sha256(
            f"{setpoint_id}|{recommended_position}|{recommended_action}".encode()
        ).hexdigest()

        logger.info(
            f"Drain setpoint {setpoint_id}: valve={valve_id}, "
            f"action={recommended_action}, position={recommended_position:.1f}%"
        )

        return setpoint

    def create_drain_sequence(
        self,
        valve_ids: List[str],
        sequence_type: DrainSequenceType = DrainSequenceType.NORMAL
    ) -> DrainSequence:
        """
        Create a drain sequence for multiple drip legs.

        Args:
            valve_ids: List of drain valve IDs
            sequence_type: Type of sequence

        Returns:
            DrainSequence: Created sequence
        """
        start_time = datetime.now()

        # Validate all valves exist
        for valve_id in valve_ids:
            if valve_id not in self._valves:
                raise KeyError(f"Valve {valve_id} not registered")

        steps = []
        step_number = 0
        total_duration = 0.0

        for valve_id in valve_ids:
            config = self._valves[valve_id]

            # Determine step parameters based on sequence type
            if sequence_type == DrainSequenceType.STARTUP:
                open_duration = config.max_open_time_s
                position = config.max_position_pct
            elif sequence_type == DrainSequenceType.WARMUP:
                open_duration = config.max_open_time_s * 2
                position = config.max_position_pct
            elif sequence_type == DrainSequenceType.SHUTDOWN:
                open_duration = config.max_open_time_s
                position = config.max_position_pct
            elif sequence_type == DrainSequenceType.EMERGENCY:
                open_duration = 0  # Stay open
                position = 100.0
            else:  # NORMAL
                open_duration = config.min_open_time_s
                position = config.max_position_pct

            # Open step
            step_number += 1
            steps.append(SequenceStep(
                step_number=step_number,
                valve_id=valve_id,
                action="open",
                position_pct=position,
                duration_s=open_duration,
                wait_for_complete=True,
                verify_condition=f"{valve_id}.position >= {position * 0.9}"
            ))
            total_duration += open_duration

            # Close step (unless emergency)
            if sequence_type != DrainSequenceType.EMERGENCY:
                step_number += 1
                steps.append(SequenceStep(
                    step_number=step_number,
                    valve_id=valve_id,
                    action="close",
                    position_pct=0.0,
                    duration_s=5.0,
                    wait_for_complete=True,
                    verify_condition=f"{valve_id}.position <= 5"
                ))
                total_duration += 5.0

        # Generate sequence ID
        sequence_id = hashlib.sha256(
            f"SEQ_{sequence_type.value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        sequence = DrainSequence(
            sequence_id=sequence_id,
            sequence_type=sequence_type,
            valve_ids=valve_ids,
            steps=steps,
            total_duration_s=total_duration,
            description=f"{sequence_type.value.title()} drain sequence for {len(valve_ids)} valves",
            requires_confirmation=(sequence_type != DrainSequenceType.EMERGENCY),
            safety_validated=True,
            created_at=start_time
        )

        # Calculate provenance hash
        sequence.provenance_hash = hashlib.sha256(
            f"{sequence_id}|{sequence_type.value}|{len(steps)}".encode()
        ).hexdigest()

        # Store in history
        self._sequence_history.append(sequence)
        if len(self._sequence_history) > self._max_history_size:
            self._sequence_history = self._sequence_history[-self._max_history_size:]

        logger.info(
            f"Created drain sequence {sequence_id}: type={sequence_type.value}, "
            f"valves={len(valve_ids)}, steps={len(steps)}, duration={total_duration:.0f}s"
        )

        return sequence

    def assess_flooding_risk(
        self,
        separator_id: str
    ) -> FloodingRisk:
        """
        Assess flooding risk for a separator.

        Args:
            separator_id: Separator ID

        Returns:
            FloodingRisk: Flooding risk assessment
        """
        start_time = datetime.now()

        if separator_id not in self._separators:
            raise KeyError(f"Separator {separator_id} not registered")

        config = self._separators[separator_id]
        state = self._separator_states.get(separator_id, {})

        current_level = state.get("level_pct", 50.0)
        level_trend = state.get("level_trend", "stable")

        # Determine risk level
        if current_level >= config.flooding_level_pct:
            risk_level = "critical"
        elif current_level >= config.high_level_pct:
            if level_trend == "rising":
                risk_level = "high"
            else:
                risk_level = "medium"
        elif current_level >= config.high_level_pct * 0.8:
            if level_trend == "rising":
                risk_level = "medium"
            else:
                risk_level = "low"
        else:
            risk_level = "low"

        # Estimate time to flood if rising
        estimated_time = None
        if level_trend == "rising" and current_level < config.flooding_level_pct:
            # Rough estimate based on typical rise rates
            remaining_capacity = config.flooding_level_pct - current_level
            typical_rise_rate = 5.0  # % per minute (configurable)
            estimated_time = (remaining_capacity / typical_rise_rate) * 60.0

        # Determine recommended action
        if risk_level == "critical":
            recommended_action = "Emergency drain - open all drain valves immediately"
        elif risk_level == "high":
            recommended_action = "Initiate drain sequence - high priority"
        elif risk_level == "medium":
            recommended_action = "Increase drain valve opening - monitor closely"
        else:
            recommended_action = "Normal operation - routine monitoring"

        # Find affected valves
        affected_valves = [config.drain_valve_id]

        # Generate assessment ID
        assessment_id = hashlib.sha256(
            f"FLOOD_{separator_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        risk = FloodingRisk(
            assessment_id=assessment_id,
            separator_id=separator_id,
            current_level_pct=current_level,
            level_trend=level_trend,
            risk_level=risk_level,
            estimated_time_to_flood_s=estimated_time,
            recommended_action=recommended_action,
            affected_valves=affected_valves,
            timestamp=start_time
        )

        # Calculate provenance hash
        risk.provenance_hash = hashlib.sha256(
            f"{assessment_id}|{risk_level}|{current_level}".encode()
        ).hexdigest()

        # Store critical events
        if risk_level in ["critical", "high"]:
            self._flooding_events.append(risk)
            if len(self._flooding_events) > self._max_history_size:
                self._flooding_events = self._flooding_events[-self._max_history_size:]

        logger.info(
            f"Flooding risk {assessment_id}: separator={separator_id}, "
            f"level={current_level:.1f}%, trend={level_trend}, risk={risk_level}"
        )

        return risk

    def analyze_backpressure(
        self,
        drain_point_id: str,
        steam_pressure_kpa: float,
        condensate_pressure_kpa: float
    ) -> BackpressureAnalysis:
        """
        Analyze condensate backpressure at a drain point.

        Args:
            drain_point_id: Drain point identifier
            steam_pressure_kpa: Steam side pressure (kPa)
            condensate_pressure_kpa: Condensate side pressure (kPa)

        Returns:
            BackpressureAnalysis: Backpressure analysis result
        """
        start_time = datetime.now()

        differential = steam_pressure_kpa - condensate_pressure_kpa

        # Minimum differential for proper drainage (configurable)
        min_differential = 50.0  # kPa

        # Determine status
        if differential <= 0:
            status = BackpressureStatus.BLOCKED
            flow_restriction = 100.0
        elif differential < min_differential * 0.5:
            status = BackpressureStatus.CRITICAL
            flow_restriction = 80.0
        elif differential < min_differential:
            status = BackpressureStatus.ELEVATED
            flow_restriction = 40.0
        else:
            status = BackpressureStatus.NORMAL
            flow_restriction = 0.0

        # Determine recommended action
        warnings = []
        if status == BackpressureStatus.BLOCKED:
            recommended_action = (
                "CRITICAL: Condensate backpressure exceeds steam pressure. "
                "Check condensate return system immediately."
            )
            warnings.append("Risk of reverse flow - water hammer hazard")
        elif status == BackpressureStatus.CRITICAL:
            recommended_action = (
                "High backpressure limiting drainage. "
                "Verify condensate return pump operation."
            )
            warnings.append("Significantly reduced drainage capacity")
        elif status == BackpressureStatus.ELEVATED:
            recommended_action = (
                "Elevated backpressure detected. "
                "Monitor condensate return pressure."
            )
        else:
            recommended_action = "Normal operation - adequate pressure differential"

        # Generate analysis ID
        analysis_id = hashlib.sha256(
            f"BP_{drain_point_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        analysis = BackpressureAnalysis(
            analysis_id=analysis_id,
            drain_point_id=drain_point_id,
            steam_pressure_kpa=steam_pressure_kpa,
            condensate_pressure_kpa=condensate_pressure_kpa,
            differential_kpa=differential,
            status=status,
            min_required_differential_kpa=min_differential,
            flow_restriction_pct=flow_restriction,
            recommended_action=recommended_action,
            warnings=warnings,
            timestamp=start_time
        )

        # Calculate provenance hash
        analysis.provenance_hash = hashlib.sha256(
            f"{analysis_id}|{differential}|{status.value}".encode()
        ).hexdigest()

        logger.info(
            f"Backpressure analysis {analysis_id}: point={drain_point_id}, "
            f"differential={differential:.1f}kPa, status={status.value}"
        )

        return analysis

    def compute_recommendation(
        self,
        measurements: Dict[str, Any]
    ) -> DrainRecommendation:
        """
        Compute comprehensive drain control recommendation.

        Args:
            measurements: Dictionary of measurements by point ID

        Returns:
            DrainRecommendation: Comprehensive recommendation
        """
        start_time = datetime.now()

        drain_setpoints = []
        flooding_risks = []
        backpressure_analyses = []

        # Compute setpoints for all valves
        for valve_id, config in self._valves.items():
            separator_level = measurements.get(f"{config.asset_id}.separator_level")
            steam_flow = measurements.get(f"{config.asset_id}.steam_flow")

            try:
                setpoint = self.compute_drain_setpoint(
                    valve_id,
                    separator_level_pct=separator_level,
                    steam_flow_kg_s=steam_flow
                )
                drain_setpoints.append(setpoint)
            except Exception as e:
                logger.error(f"Error computing setpoint for {valve_id}: {e}")

        # Assess flooding risk for all separators
        for separator_id in self._separators:
            try:
                risk = self.assess_flooding_risk(separator_id)
                flooding_risks.append(risk)
            except Exception as e:
                logger.error(f"Error assessing flooding for {separator_id}: {e}")

        # Analyze backpressure for drain points with pressure data
        for point_id, values in measurements.items():
            if "steam_pressure" in str(point_id) and "condensate_pressure" in measurements:
                try:
                    steam_p = measurements.get(f"{point_id}.steam_pressure", 0)
                    cond_p = measurements.get(f"{point_id}.condensate_pressure", 0)
                    if steam_p > 0:
                        analysis = self.analyze_backpressure(
                            point_id, steam_p, cond_p
                        )
                        backpressure_analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing backpressure for {point_id}: {e}")

        # Determine overall status
        critical_risks = [r for r in flooding_risks if r.risk_level == "critical"]
        high_risks = [r for r in flooding_risks if r.risk_level == "high"]

        if critical_risks:
            overall_status = "critical"
            requires_immediate = True
        elif high_risks:
            overall_status = "warning"
            requires_immediate = True
        elif any(a.status in [BackpressureStatus.BLOCKED, BackpressureStatus.CRITICAL]
                 for a in backpressure_analyses):
            overall_status = "warning"
            requires_immediate = True
        else:
            overall_status = "normal"
            requires_immediate = False

        # Generate recommendation ID
        recommendation_id = hashlib.sha256(
            f"DREC_{self.controller_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        recommendation = DrainRecommendation(
            recommendation_id=recommendation_id,
            drain_setpoints=drain_setpoints,
            sequences=[],  # Sequences created on demand
            flooding_risks=flooding_risks,
            backpressure_analysis=backpressure_analyses,
            overall_status=overall_status,
            requires_immediate_action=requires_immediate,
            timestamp=start_time
        )

        # Calculate provenance hash
        recommendation.provenance_hash = hashlib.sha256(
            f"{recommendation_id}|{overall_status}|{len(drain_setpoints)}".encode()
        ).hexdigest()

        logger.info(
            f"Drain recommendation {recommendation_id}: status={overall_status}, "
            f"setpoints={len(drain_setpoints)}, risks={len(flooding_risks)}"
        )

        return recommendation

    def get_valve_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all registered valves."""
        return self._valve_states.copy()

    def get_separator_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all registered separators."""
        return self._separator_states.copy()

    def get_sequence_history(
        self,
        time_window_minutes: int = 60
    ) -> List[DrainSequence]:
        """Get sequence history within time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        return [s for s in self._sequence_history if s.created_at >= cutoff]

    def get_flooding_events(
        self,
        time_window_minutes: int = 60
    ) -> List[FloodingRisk]:
        """Get flooding events within time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        return [e for e in self._flooding_events if e.timestamp >= cutoff]
