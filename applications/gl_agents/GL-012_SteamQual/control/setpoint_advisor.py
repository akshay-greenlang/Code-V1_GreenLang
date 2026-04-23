"""
GL-012 STEAMQUAL SteamQualityController - Setpoint Advisor

This module implements setpoint recommendations for steam quality control,
including desuperheater/attemperator bounds, header pressure/PRV setpoints,
ramp rate limits, and minimum superheat margin policies.

Control Architecture:
    - Desuperheater/attemperator setpoint bounds
    - Header pressure and PRV setpoint management
    - Ramp rate limits during load changes
    - Minimum superheat margin policy enforcement
    - Advisory mode (default) - recommendations only

Key Features:
    - Dynamic setpoint bounds based on operating conditions
    - Load change guidance with rate limiting
    - Superheat margin protection
    - Cross-asset setpoint coordination

Reference Standards:
    - ISA-18.2 Management of Alarm Systems
    - ASME B31.1 Power Piping
    - IEC 61511 Functional Safety

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

class SetpointType(str, Enum):
    """Setpoint type enumeration."""
    DESUPERHEATER_TEMP = "desuperheater_temp"
    DESUPERHEATER_SPRAY = "desuperheater_spray"
    ATTEMPERATOR_TEMP = "attemperator_temp"
    HEADER_PRESSURE = "header_pressure"
    PRV_DOWNSTREAM = "prv_downstream"
    SEPARATOR_LEVEL = "separator_level"
    SUPERHEAT_MARGIN = "superheat_margin"


class RampRatePolicy(str, Enum):
    """Ramp rate policy enumeration."""
    CONSERVATIVE = "conservative"  # Slow changes, maximum protection
    NORMAL = "normal"  # Balanced approach
    AGGRESSIVE = "aggressive"  # Fast changes, higher risk
    EMERGENCY = "emergency"  # No rate limiting (emergency only)


class SuperheatPolicy(str, Enum):
    """Superheat margin policy enumeration."""
    STRICT = "strict"  # Large margins, conservative
    STANDARD = "standard"  # Industry standard margins
    RELAXED = "relaxed"  # Tighter margins, efficiency focus
    OVERRIDE = "override"  # Manual override (requires authorization)


# =============================================================================
# DATA MODELS
# =============================================================================

class DesuperheaterBounds(BaseModel):
    """Desuperheater/attemperator setpoint bounds."""

    equipment_id: str = Field(..., description="Equipment identifier")
    min_outlet_temp_c: float = Field(
        ...,
        description="Minimum outlet temperature (C)"
    )
    max_outlet_temp_c: float = Field(
        ...,
        description="Maximum outlet temperature (C)"
    )
    target_outlet_temp_c: float = Field(
        ...,
        description="Target outlet temperature (C)"
    )
    min_spray_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Minimum spray valve position (%)"
    )
    max_spray_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Maximum spray valve position (%)"
    )
    saturation_temp_c: float = Field(
        ...,
        description="Current saturation temperature (C)"
    )
    superheat_margin_c: float = Field(
        ...,
        ge=0,
        description="Required superheat margin (C)"
    )
    effective_min_temp_c: float = Field(
        ...,
        description="Effective minimum temp (saturation + margin)"
    )
    constraints_active: bool = Field(
        default=True,
        description="Constraints are active"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Bounds timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class HeaderPressureBounds(BaseModel):
    """Header pressure and PRV setpoint bounds."""

    header_id: str = Field(..., description="Header identifier")
    min_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Minimum header pressure (kPa)"
    )
    max_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Maximum header pressure (kPa)"
    )
    target_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Target header pressure (kPa)"
    )
    prv_setpoint_kpa: float = Field(
        ...,
        ge=0,
        description="PRV downstream setpoint (kPa)"
    )
    safety_relief_kpa: float = Field(
        ...,
        ge=0,
        description="Safety relief valve setting (kPa)"
    )
    min_differential_kpa: float = Field(
        default=50.0,
        ge=0,
        description="Minimum PRV differential (kPa)"
    )
    alarm_low_kpa: float = Field(
        ...,
        ge=0,
        description="Low pressure alarm (kPa)"
    )
    alarm_high_kpa: float = Field(
        ...,
        ge=0,
        description="High pressure alarm (kPa)"
    )
    constraints_active: bool = Field(
        default=True,
        description="Constraints are active"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Bounds timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class RampRateLimits(BaseModel):
    """Ramp rate limits for load changes."""

    policy: RampRatePolicy = Field(..., description="Active ramp rate policy")
    temp_rate_c_per_min: float = Field(
        ...,
        ge=0,
        description="Temperature change rate (C/min)"
    )
    pressure_rate_kpa_per_min: float = Field(
        ...,
        ge=0,
        description="Pressure change rate (kPa/min)"
    )
    spray_rate_pct_per_s: float = Field(
        ...,
        ge=0,
        description="Spray valve change rate (%/s)"
    )
    load_rate_pct_per_min: float = Field(
        ...,
        ge=0,
        description="Load change rate (%/min)"
    )
    cooldown_period_s: float = Field(
        default=60.0,
        ge=0,
        description="Cooldown period between changes (s)"
    )
    description: str = Field(
        default="",
        description="Policy description"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Limits timestamp"
    )


class SuperheatMarginPolicy(BaseModel):
    """Minimum superheat margin policy."""

    policy: SuperheatPolicy = Field(..., description="Active superheat policy")
    min_superheat_c: float = Field(
        ...,
        ge=0,
        description="Minimum superheat margin (C)"
    )
    warning_threshold_c: float = Field(
        ...,
        ge=0,
        description="Warning threshold (C above minimum)"
    )
    critical_threshold_c: float = Field(
        ...,
        ge=0,
        description="Critical threshold (C below minimum)"
    )
    measurement_points: List[str] = Field(
        default_factory=list,
        description="Measurement points covered"
    )
    enforcement_mode: str = Field(
        default="advisory",
        description="Enforcement mode (advisory, blocking)"
    )
    override_authorization: Optional[str] = Field(
        None,
        description="Authorization ID if override active"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Policy timestamp"
    )


class SetpointRecommendation(BaseModel):
    """Setpoint recommendation from advisor."""

    recommendation_id: str = Field(..., description="Recommendation ID")
    setpoint_type: SetpointType = Field(..., description="Setpoint type")
    equipment_id: str = Field(..., description="Equipment identifier")
    current_value: float = Field(..., description="Current setpoint value")
    recommended_value: float = Field(..., description="Recommended value")
    unit: str = Field(..., description="Engineering unit")
    min_allowed: float = Field(..., description="Minimum allowed value")
    max_allowed: float = Field(..., description="Maximum allowed value")
    rationale: str = Field(..., description="Recommendation rationale")
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
    ramp_rate_applied: bool = Field(
        default=False,
        description="Ramp rate limiting was applied"
    )
    safety_validated: bool = Field(
        default=False,
        description="Passed safety validation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Recommendation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class LoadChangeGuidance(BaseModel):
    """Guidance for load changes with setpoint coordination."""

    guidance_id: str = Field(..., description="Guidance ID")
    current_load_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current load (%)"
    )
    target_load_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Target load (%)"
    )
    direction: str = Field(
        ...,
        description="Load change direction (increase, decrease)"
    )
    estimated_duration_min: float = Field(
        ...,
        ge=0,
        description="Estimated duration (minutes)"
    )
    ramp_rate_limits: RampRateLimits = Field(
        ...,
        description="Active ramp rate limits"
    )
    setpoint_sequence: List[SetpointRecommendation] = Field(
        default_factory=list,
        description="Recommended setpoint sequence"
    )
    critical_checkpoints: List[str] = Field(
        default_factory=list,
        description="Critical checkpoints during change"
    )
    abort_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions that should abort the change"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Guidance timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class SetpointValidation(BaseModel):
    """Validation result for a proposed setpoint."""

    validation_id: str = Field(..., description="Validation ID")
    setpoint_type: SetpointType = Field(..., description="Setpoint type")
    equipment_id: str = Field(..., description="Equipment identifier")
    proposed_value: float = Field(..., description="Proposed value")
    is_valid: bool = Field(..., description="Validation passed")
    violations: List[str] = Field(
        default_factory=list,
        description="Constraint violations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    suggested_value: Optional[float] = Field(
        None,
        description="Suggested alternative value"
    )
    affected_constraints: List[str] = Field(
        default_factory=list,
        description="Constraints that were checked"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Validation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# SETPOINT ADVISOR
# =============================================================================

class SetpointAdvisor:
    """
    Setpoint recommendations for steam quality control.

    This advisor provides setpoint recommendations and validation for
    desuperheaters, attemperators, header pressures, and PRVs while
    enforcing ramp rate limits and superheat margin policies.

    Control Features:
        - Dynamic setpoint bounds based on operating conditions
        - Load change guidance with coordinated setpoints
        - Ramp rate limiting for thermal protection
        - Superheat margin policy enforcement

    Safety Features:
        - Never recommends setpoints below saturation + margin
        - Rate limiting prevents thermal shock
        - All recommendations are advisory (require confirmation)
        - Complete audit trail for all recommendations

    Attributes:
        advisor_id: Unique advisor identifier
        ramp_policy: Active ramp rate policy
        superheat_policy: Active superheat margin policy
        _equipment_bounds: Current equipment bounds
        _recommendation_history: History of recommendations

    Example:
        >>> advisor = SetpointAdvisor("SA-001")
        >>> advisor.register_desuperheater(equipment_id, config)
        >>> bounds = advisor.compute_desuperheater_bounds(equipment_id, state)
        >>> recommendation = advisor.recommend_setpoint(equipment_id, target_temp)
    """

    def __init__(
        self,
        advisor_id: str,
        ramp_policy: RampRatePolicy = RampRatePolicy.NORMAL,
        superheat_policy: SuperheatPolicy = SuperheatPolicy.STANDARD
    ):
        """
        Initialize SetpointAdvisor.

        Args:
            advisor_id: Unique advisor identifier
            ramp_policy: Initial ramp rate policy
            superheat_policy: Initial superheat margin policy
        """
        self.advisor_id = advisor_id
        self.ramp_policy = ramp_policy
        self.superheat_policy = superheat_policy

        # Equipment configurations
        self._desuperheaters: Dict[str, Dict[str, Any]] = {}
        self._headers: Dict[str, Dict[str, Any]] = {}

        # Current bounds and policies
        self._equipment_bounds: Dict[str, Any] = {}
        self._active_ramp_limits: Optional[RampRateLimits] = None
        self._active_superheat_policy: Optional[SuperheatMarginPolicy] = None

        # History
        self._recommendation_history: List[SetpointRecommendation] = []
        self._last_setpoint_times: Dict[str, datetime] = {}
        self._max_history_size = 1000

        # Initialize policies
        self._update_ramp_limits()
        self._update_superheat_policy()

        logger.info(
            f"SetpointAdvisor {advisor_id} initialized with "
            f"ramp_policy={ramp_policy.value}, superheat_policy={superheat_policy.value}"
        )

    def register_desuperheater(
        self,
        equipment_id: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Register a desuperheater/attemperator.

        Args:
            equipment_id: Equipment identifier
            config: Equipment configuration
        """
        self._desuperheaters[equipment_id] = config
        logger.info(f"Registered desuperheater {equipment_id}")

    def register_header(
        self,
        header_id: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Register a steam header.

        Args:
            header_id: Header identifier
            config: Header configuration
        """
        self._headers[header_id] = config
        logger.info(f"Registered header {header_id}")

    def compute_desuperheater_bounds(
        self,
        equipment_id: str,
        current_state: Dict[str, float]
    ) -> DesuperheaterBounds:
        """
        Compute desuperheater setpoint bounds.

        Args:
            equipment_id: Desuperheater equipment ID
            current_state: Current state including:
                - pressure_kpa: Current pressure
                - inlet_temp_c: Inlet temperature
                - saturation_temp_c: Saturation temperature

        Returns:
            DesuperheaterBounds: Computed setpoint bounds
        """
        start_time = datetime.now()

        if equipment_id not in self._desuperheaters:
            raise KeyError(f"Desuperheater {equipment_id} not registered")

        config = self._desuperheaters[equipment_id]

        # Get current conditions
        pressure_kpa = current_state.get("pressure_kpa", 1000.0)
        inlet_temp_c = current_state.get("inlet_temp_c", 400.0)
        saturation_temp_c = current_state.get(
            "saturation_temp_c",
            self._calculate_saturation_temp(pressure_kpa)
        )

        # Get superheat margin from policy
        superheat_margin = self._get_superheat_margin()

        # Calculate effective minimum temperature
        effective_min_temp = saturation_temp_c + superheat_margin

        # Get configured limits
        design_min = config.get("min_outlet_temp_c", effective_min_temp)
        design_max = config.get("max_outlet_temp_c", inlet_temp_c)
        design_target = config.get("target_outlet_temp_c", (design_min + design_max) / 2)

        # Apply safety constraints
        min_temp = max(effective_min_temp, design_min)
        max_temp = min(inlet_temp_c - 5.0, design_max)  # 5C below inlet
        target_temp = max(min_temp + 10, min(max_temp - 10, design_target))

        bounds = DesuperheaterBounds(
            equipment_id=equipment_id,
            min_outlet_temp_c=min_temp,
            max_outlet_temp_c=max_temp,
            target_outlet_temp_c=target_temp,
            min_spray_pct=config.get("min_spray_pct", 0.0),
            max_spray_pct=config.get("max_spray_pct", 100.0),
            saturation_temp_c=saturation_temp_c,
            superheat_margin_c=superheat_margin,
            effective_min_temp_c=effective_min_temp,
            constraints_active=True,
            timestamp=start_time
        )

        # Calculate provenance hash
        bounds.provenance_hash = hashlib.sha256(
            f"{equipment_id}|{min_temp}|{max_temp}|{superheat_margin}".encode()
        ).hexdigest()

        # Store bounds
        self._equipment_bounds[equipment_id] = bounds

        logger.info(
            f"Desuperheater bounds {equipment_id}: "
            f"temp=[{min_temp:.1f}, {max_temp:.1f}]C, "
            f"target={target_temp:.1f}C, margin={superheat_margin:.1f}C"
        )

        return bounds

    def compute_header_pressure_bounds(
        self,
        header_id: str,
        current_state: Dict[str, float]
    ) -> HeaderPressureBounds:
        """
        Compute header pressure and PRV setpoint bounds.

        Args:
            header_id: Header identifier
            current_state: Current state including:
                - upstream_pressure_kpa: Upstream pressure
                - downstream_demand_kg_s: Downstream demand

        Returns:
            HeaderPressureBounds: Computed pressure bounds
        """
        start_time = datetime.now()

        if header_id not in self._headers:
            raise KeyError(f"Header {header_id} not registered")

        config = self._headers[header_id]

        # Get configured limits
        min_pressure = config.get("min_pressure_kpa", 500.0)
        max_pressure = config.get("max_pressure_kpa", 1500.0)
        target_pressure = config.get("target_pressure_kpa", (min_pressure + max_pressure) / 2)
        safety_relief = config.get("safety_relief_kpa", max_pressure * 1.1)

        # Get upstream pressure for PRV calculations
        upstream_pressure = current_state.get("upstream_pressure_kpa", max_pressure * 1.2)

        # Calculate PRV setpoint with minimum differential
        min_differential = config.get("min_differential_kpa", 50.0)
        prv_setpoint = min(target_pressure, upstream_pressure - min_differential)

        # Calculate alarm thresholds
        alarm_low = min_pressure * 1.05
        alarm_high = max_pressure * 0.95

        bounds = HeaderPressureBounds(
            header_id=header_id,
            min_pressure_kpa=min_pressure,
            max_pressure_kpa=max_pressure,
            target_pressure_kpa=target_pressure,
            prv_setpoint_kpa=prv_setpoint,
            safety_relief_kpa=safety_relief,
            min_differential_kpa=min_differential,
            alarm_low_kpa=alarm_low,
            alarm_high_kpa=alarm_high,
            constraints_active=True,
            timestamp=start_time
        )

        # Calculate provenance hash
        bounds.provenance_hash = hashlib.sha256(
            f"{header_id}|{min_pressure}|{max_pressure}|{prv_setpoint}".encode()
        ).hexdigest()

        # Store bounds
        self._equipment_bounds[header_id] = bounds

        logger.info(
            f"Header bounds {header_id}: "
            f"pressure=[{min_pressure:.0f}, {max_pressure:.0f}]kPa, "
            f"PRV={prv_setpoint:.0f}kPa"
        )

        return bounds

    def recommend_setpoint(
        self,
        equipment_id: str,
        setpoint_type: SetpointType,
        target_value: float,
        current_value: float
    ) -> SetpointRecommendation:
        """
        Generate setpoint recommendation with validation.

        Args:
            equipment_id: Equipment identifier
            setpoint_type: Type of setpoint
            target_value: Desired target value
            current_value: Current setpoint value

        Returns:
            SetpointRecommendation: Validated recommendation
        """
        start_time = datetime.now()

        # Get bounds for equipment
        bounds = self._equipment_bounds.get(equipment_id)
        if bounds is None:
            raise ValueError(f"No bounds computed for {equipment_id}")

        # Determine unit and limits based on type
        if setpoint_type in [SetpointType.DESUPERHEATER_TEMP, SetpointType.ATTEMPERATOR_TEMP]:
            unit = "C"
            min_allowed = bounds.min_outlet_temp_c
            max_allowed = bounds.max_outlet_temp_c
        elif setpoint_type == SetpointType.DESUPERHEATER_SPRAY:
            unit = "%"
            min_allowed = bounds.min_spray_pct
            max_allowed = bounds.max_spray_pct
        elif setpoint_type in [SetpointType.HEADER_PRESSURE, SetpointType.PRV_DOWNSTREAM]:
            unit = "kPa"
            min_allowed = bounds.min_pressure_kpa
            max_allowed = bounds.max_pressure_kpa
        else:
            unit = ""
            min_allowed = 0
            max_allowed = 100

        # Apply constraints to target
        constrained_value = max(min_allowed, min(max_allowed, target_value))

        # Apply ramp rate limiting
        ramp_limited = False
        if self._active_ramp_limits:
            constrained_value, ramp_limited = self._apply_ramp_limit(
                current_value,
                constrained_value,
                setpoint_type,
                equipment_id
            )

        # Validate and generate warnings
        warnings = []
        safety_validated = True

        if target_value != constrained_value:
            warnings.append(
                f"Target {target_value:.2f} constrained to {constrained_value:.2f}"
            )

        if ramp_limited:
            warnings.append("Ramp rate limiting applied")

        # Check superheat margin for temperature setpoints
        if setpoint_type in [SetpointType.DESUPERHEATER_TEMP, SetpointType.ATTEMPERATOR_TEMP]:
            if hasattr(bounds, 'effective_min_temp_c'):
                if constrained_value < bounds.effective_min_temp_c + 5:
                    warnings.append(
                        f"Approaching superheat margin limit "
                        f"(min: {bounds.effective_min_temp_c:.1f}C)"
                    )

        # Determine priority
        change_magnitude = abs(constrained_value - current_value)
        if change_magnitude > 20:
            priority = 2
        elif change_magnitude > 5:
            priority = 3
        else:
            priority = 4

        # Generate rationale
        rationale = self._generate_setpoint_rationale(
            setpoint_type, current_value, constrained_value, target_value
        )

        # Generate recommendation ID
        recommendation_id = hashlib.sha256(
            f"REC_{equipment_id}_{setpoint_type.value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        recommendation = SetpointRecommendation(
            recommendation_id=recommendation_id,
            setpoint_type=setpoint_type,
            equipment_id=equipment_id,
            current_value=current_value,
            recommended_value=constrained_value,
            unit=unit,
            min_allowed=min_allowed,
            max_allowed=max_allowed,
            rationale=rationale,
            priority=priority,
            requires_confirmation=True,  # Advisory mode
            ramp_rate_applied=ramp_limited,
            safety_validated=safety_validated,
            warnings=warnings,
            timestamp=start_time
        )

        # Calculate provenance hash
        recommendation.provenance_hash = hashlib.sha256(
            f"{recommendation_id}|{constrained_value}|{safety_validated}".encode()
        ).hexdigest()

        # Store in history
        self._recommendation_history.append(recommendation)
        self._last_setpoint_times[equipment_id] = start_time
        if len(self._recommendation_history) > self._max_history_size:
            self._recommendation_history = self._recommendation_history[-self._max_history_size:]

        logger.info(
            f"Setpoint recommendation {recommendation_id}: "
            f"{equipment_id}.{setpoint_type.value} = {constrained_value:.2f}{unit} "
            f"(from {current_value:.2f}, target {target_value:.2f})"
        )

        return recommendation

    def generate_load_change_guidance(
        self,
        current_load_pct: float,
        target_load_pct: float,
        affected_equipment: List[str]
    ) -> LoadChangeGuidance:
        """
        Generate guidance for a load change with coordinated setpoints.

        Args:
            current_load_pct: Current load percentage
            target_load_pct: Target load percentage
            affected_equipment: List of affected equipment IDs

        Returns:
            LoadChangeGuidance: Comprehensive load change guidance
        """
        start_time = datetime.now()

        direction = "increase" if target_load_pct > current_load_pct else "decrease"
        load_change_pct = abs(target_load_pct - current_load_pct)

        # Get active ramp limits
        ramp_limits = self._active_ramp_limits or self._get_default_ramp_limits()

        # Calculate estimated duration
        estimated_duration = load_change_pct / ramp_limits.load_rate_pct_per_min

        # Generate setpoint sequence
        setpoint_sequence = []
        for equipment_id in affected_equipment:
            bounds = self._equipment_bounds.get(equipment_id)
            if bounds is None:
                continue

            # For load increase: gradually increase setpoints
            # For load decrease: gradually decrease setpoints
            if hasattr(bounds, 'target_outlet_temp_c'):
                # Desuperheater
                current_temp = bounds.target_outlet_temp_c
                if direction == "increase":
                    target_temp = bounds.max_outlet_temp_c * 0.9
                else:
                    target_temp = bounds.min_outlet_temp_c * 1.1

                rec = self.recommend_setpoint(
                    equipment_id,
                    SetpointType.DESUPERHEATER_TEMP,
                    target_temp,
                    current_temp
                )
                setpoint_sequence.append(rec)

        # Generate critical checkpoints
        critical_checkpoints = [
            f"Verify superheat margin at {current_load_pct + load_change_pct * 0.25:.0f}% load",
            f"Check separator levels at {current_load_pct + load_change_pct * 0.5:.0f}% load",
            f"Confirm stable operation at {current_load_pct + load_change_pct * 0.75:.0f}% load",
        ]

        # Define abort conditions
        abort_conditions = [
            "Superheat margin below minimum threshold",
            "Separator level above high alarm",
            "Any critical alarm active",
            "Steam quality below acceptable limit"
        ]

        # Generate guidance ID
        guidance_id = hashlib.sha256(
            f"GUIDE_{current_load_pct}_{target_load_pct}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        guidance = LoadChangeGuidance(
            guidance_id=guidance_id,
            current_load_pct=current_load_pct,
            target_load_pct=target_load_pct,
            direction=direction,
            estimated_duration_min=estimated_duration,
            ramp_rate_limits=ramp_limits,
            setpoint_sequence=setpoint_sequence,
            critical_checkpoints=critical_checkpoints,
            abort_conditions=abort_conditions,
            timestamp=start_time
        )

        # Calculate provenance hash
        guidance.provenance_hash = hashlib.sha256(
            f"{guidance_id}|{direction}|{estimated_duration}".encode()
        ).hexdigest()

        logger.info(
            f"Load change guidance {guidance_id}: "
            f"{current_load_pct:.0f}% -> {target_load_pct:.0f}% ({direction}), "
            f"duration={estimated_duration:.1f}min, setpoints={len(setpoint_sequence)}"
        )

        return guidance

    def validate_setpoint(
        self,
        equipment_id: str,
        setpoint_type: SetpointType,
        proposed_value: float
    ) -> SetpointValidation:
        """
        Validate a proposed setpoint against all constraints.

        Args:
            equipment_id: Equipment identifier
            setpoint_type: Type of setpoint
            proposed_value: Proposed setpoint value

        Returns:
            SetpointValidation: Validation result
        """
        start_time = datetime.now()

        violations = []
        warnings = []
        affected_constraints = []
        suggested_value = None
        is_valid = True

        # Get bounds for equipment
        bounds = self._equipment_bounds.get(equipment_id)

        if bounds is None:
            violations.append(f"No bounds defined for {equipment_id}")
            is_valid = False
        else:
            # Check type-specific constraints
            if setpoint_type in [SetpointType.DESUPERHEATER_TEMP, SetpointType.ATTEMPERATOR_TEMP]:
                affected_constraints.append("min_outlet_temp")
                affected_constraints.append("max_outlet_temp")
                affected_constraints.append("superheat_margin")

                if proposed_value < bounds.min_outlet_temp_c:
                    violations.append(
                        f"Below minimum temperature {bounds.min_outlet_temp_c:.1f}C"
                    )
                    is_valid = False
                    suggested_value = bounds.min_outlet_temp_c

                if proposed_value > bounds.max_outlet_temp_c:
                    violations.append(
                        f"Above maximum temperature {bounds.max_outlet_temp_c:.1f}C"
                    )
                    is_valid = False
                    suggested_value = bounds.max_outlet_temp_c

                if proposed_value < bounds.effective_min_temp_c:
                    violations.append(
                        f"Below effective minimum (saturation + margin): "
                        f"{bounds.effective_min_temp_c:.1f}C"
                    )
                    is_valid = False
                    suggested_value = bounds.effective_min_temp_c

                # Warning for approaching limits
                if is_valid and proposed_value < bounds.effective_min_temp_c + 10:
                    warnings.append("Approaching superheat margin limit")

            elif setpoint_type in [SetpointType.HEADER_PRESSURE, SetpointType.PRV_DOWNSTREAM]:
                affected_constraints.append("min_pressure")
                affected_constraints.append("max_pressure")

                if proposed_value < bounds.min_pressure_kpa:
                    violations.append(
                        f"Below minimum pressure {bounds.min_pressure_kpa:.0f}kPa"
                    )
                    is_valid = False
                    suggested_value = bounds.min_pressure_kpa

                if proposed_value > bounds.max_pressure_kpa:
                    violations.append(
                        f"Above maximum pressure {bounds.max_pressure_kpa:.0f}kPa"
                    )
                    is_valid = False
                    suggested_value = bounds.max_pressure_kpa

        # Generate validation ID
        validation_id = hashlib.sha256(
            f"VAL_{equipment_id}_{proposed_value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        validation = SetpointValidation(
            validation_id=validation_id,
            setpoint_type=setpoint_type,
            equipment_id=equipment_id,
            proposed_value=proposed_value,
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            suggested_value=suggested_value,
            affected_constraints=affected_constraints,
            timestamp=start_time
        )

        # Calculate provenance hash
        validation.provenance_hash = hashlib.sha256(
            f"{validation_id}|{is_valid}|{len(violations)}".encode()
        ).hexdigest()

        logger.info(
            f"Setpoint validation {validation_id}: "
            f"{equipment_id}.{setpoint_type.value}={proposed_value:.2f}, "
            f"valid={is_valid}, violations={len(violations)}"
        )

        return validation

    def set_ramp_policy(self, policy: RampRatePolicy) -> None:
        """Set the active ramp rate policy."""
        self.ramp_policy = policy
        self._update_ramp_limits()
        logger.info(f"Ramp rate policy set to {policy.value}")

    def set_superheat_policy(self, policy: SuperheatPolicy) -> None:
        """Set the active superheat margin policy."""
        self.superheat_policy = policy
        self._update_superheat_policy()
        logger.info(f"Superheat policy set to {policy.value}")

    def get_active_ramp_limits(self) -> RampRateLimits:
        """Get currently active ramp rate limits."""
        return self._active_ramp_limits or self._get_default_ramp_limits()

    def get_active_superheat_policy(self) -> SuperheatMarginPolicy:
        """Get currently active superheat margin policy."""
        return self._active_superheat_policy or SuperheatMarginPolicy(
            policy=SuperheatPolicy.STANDARD,
            min_superheat_c=15.0,
            warning_threshold_c=5.0,
            critical_threshold_c=3.0,
            enforcement_mode="advisory"
        )

    def get_recommendation_history(
        self,
        time_window_minutes: int = 60
    ) -> List[SetpointRecommendation]:
        """Get recommendation history within time window."""
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        return [r for r in self._recommendation_history if r.timestamp >= cutoff]

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _update_ramp_limits(self) -> None:
        """Update ramp rate limits based on policy."""
        if self.ramp_policy == RampRatePolicy.CONSERVATIVE:
            self._active_ramp_limits = RampRateLimits(
                policy=self.ramp_policy,
                temp_rate_c_per_min=2.0,
                pressure_rate_kpa_per_min=20.0,
                spray_rate_pct_per_s=1.0,
                load_rate_pct_per_min=2.0,
                cooldown_period_s=120.0,
                description="Conservative ramp rates for maximum protection"
            )
        elif self.ramp_policy == RampRatePolicy.AGGRESSIVE:
            self._active_ramp_limits = RampRateLimits(
                policy=self.ramp_policy,
                temp_rate_c_per_min=8.0,
                pressure_rate_kpa_per_min=80.0,
                spray_rate_pct_per_s=4.0,
                load_rate_pct_per_min=8.0,
                cooldown_period_s=30.0,
                description="Aggressive ramp rates for fast response"
            )
        elif self.ramp_policy == RampRatePolicy.EMERGENCY:
            self._active_ramp_limits = RampRateLimits(
                policy=self.ramp_policy,
                temp_rate_c_per_min=float('inf'),
                pressure_rate_kpa_per_min=float('inf'),
                spray_rate_pct_per_s=float('inf'),
                load_rate_pct_per_min=float('inf'),
                cooldown_period_s=0.0,
                description="Emergency mode - no rate limiting"
            )
        else:  # NORMAL
            self._active_ramp_limits = RampRateLimits(
                policy=self.ramp_policy,
                temp_rate_c_per_min=5.0,
                pressure_rate_kpa_per_min=50.0,
                spray_rate_pct_per_s=2.0,
                load_rate_pct_per_min=5.0,
                cooldown_period_s=60.0,
                description="Normal ramp rates for balanced operation"
            )

    def _update_superheat_policy(self) -> None:
        """Update superheat margin policy."""
        if self.superheat_policy == SuperheatPolicy.STRICT:
            self._active_superheat_policy = SuperheatMarginPolicy(
                policy=self.superheat_policy,
                min_superheat_c=25.0,
                warning_threshold_c=10.0,
                critical_threshold_c=5.0,
                enforcement_mode="advisory"
            )
        elif self.superheat_policy == SuperheatPolicy.RELAXED:
            self._active_superheat_policy = SuperheatMarginPolicy(
                policy=self.superheat_policy,
                min_superheat_c=10.0,
                warning_threshold_c=3.0,
                critical_threshold_c=2.0,
                enforcement_mode="advisory"
            )
        elif self.superheat_policy == SuperheatPolicy.OVERRIDE:
            self._active_superheat_policy = SuperheatMarginPolicy(
                policy=self.superheat_policy,
                min_superheat_c=5.0,
                warning_threshold_c=2.0,
                critical_threshold_c=1.0,
                enforcement_mode="advisory"
            )
        else:  # STANDARD
            self._active_superheat_policy = SuperheatMarginPolicy(
                policy=self.superheat_policy,
                min_superheat_c=15.0,
                warning_threshold_c=5.0,
                critical_threshold_c=3.0,
                enforcement_mode="advisory"
            )

    def _get_superheat_margin(self) -> float:
        """Get current superheat margin from policy."""
        if self._active_superheat_policy:
            return self._active_superheat_policy.min_superheat_c
        return 15.0  # Default

    def _get_default_ramp_limits(self) -> RampRateLimits:
        """Get default ramp rate limits."""
        return RampRateLimits(
            policy=RampRatePolicy.NORMAL,
            temp_rate_c_per_min=5.0,
            pressure_rate_kpa_per_min=50.0,
            spray_rate_pct_per_s=2.0,
            load_rate_pct_per_min=5.0,
            cooldown_period_s=60.0,
            description="Default normal ramp rates"
        )

    def _apply_ramp_limit(
        self,
        current_value: float,
        target_value: float,
        setpoint_type: SetpointType,
        equipment_id: str
    ) -> Tuple[float, bool]:
        """Apply ramp rate limiting to a setpoint change."""
        if not self._active_ramp_limits:
            return target_value, False

        # Get time since last change
        last_time = self._last_setpoint_times.get(equipment_id)
        if last_time is None:
            time_delta_min = 1.0  # Assume 1 minute if no history
        else:
            time_delta_min = (datetime.now() - last_time).total_seconds() / 60.0

        # Determine applicable rate limit
        if setpoint_type in [SetpointType.DESUPERHEATER_TEMP, SetpointType.ATTEMPERATOR_TEMP]:
            rate_limit = self._active_ramp_limits.temp_rate_c_per_min
        elif setpoint_type in [SetpointType.HEADER_PRESSURE, SetpointType.PRV_DOWNSTREAM]:
            rate_limit = self._active_ramp_limits.pressure_rate_kpa_per_min
        elif setpoint_type == SetpointType.DESUPERHEATER_SPRAY:
            rate_limit = self._active_ramp_limits.spray_rate_pct_per_s * 60  # Convert to per min
        else:
            rate_limit = float('inf')

        # Calculate maximum allowed change
        max_change = rate_limit * time_delta_min

        # Apply limit
        change = target_value - current_value
        if abs(change) <= max_change:
            return target_value, False

        if change > 0:
            limited_value = current_value + max_change
        else:
            limited_value = current_value - max_change

        return limited_value, True

    def _calculate_saturation_temp(self, pressure_kpa: float) -> float:
        """Calculate saturation temperature from pressure."""
        # Simplified approximation
        if pressure_kpa <= 0:
            return 100.0

        pressure_bar = pressure_kpa / 100.0
        saturation_temp_c = 100 + 30 * (pressure_bar - 1) ** 0.5

        return saturation_temp_c

    def _generate_setpoint_rationale(
        self,
        setpoint_type: SetpointType,
        current: float,
        recommended: float,
        target: float
    ) -> str:
        """Generate rationale for setpoint recommendation."""
        change = recommended - current
        direction = "increase" if change > 0 else "decrease"

        rationale = f"Recommend {direction} of {abs(change):.2f} "

        if setpoint_type in [SetpointType.DESUPERHEATER_TEMP, SetpointType.ATTEMPERATOR_TEMP]:
            rationale += "C for temperature control. "
        elif setpoint_type in [SetpointType.HEADER_PRESSURE, SetpointType.PRV_DOWNSTREAM]:
            rationale += "kPa for pressure control. "
        else:
            rationale += "for setpoint control. "

        if recommended != target:
            rationale += f"Target {target:.2f} was constrained to {recommended:.2f} "
            rationale += "due to operational limits."

        return rationale
