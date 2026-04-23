"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - PRV Controller

This module implements Pressure Reducing Valve (PRV) setpoint management
for steam header pressure control with demand forecasting and constraint enforcement.

Control Architecture:
    - Downstream pressure setpoint calculation
    - Demand-based setpoint optimization
    - Pressure constraint enforcement
    - Advisory generation for operator guidance

Reference Standards:
    - ASME B31.1 Power Piping
    - ISA-75.01 Control Valve Sizing
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

class PRVState(str, Enum):
    """PRV operational state enumeration."""
    NORMAL = "normal"
    REDUCING = "reducing"
    RELIEVING = "relieving"
    ISOLATED = "isolated"
    FAULT = "fault"


class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    REQUIRES_REVIEW = "requires_review"


class AdvisoryType(str, Enum):
    """Advisory type enumeration."""
    SETPOINT_CHANGE = "setpoint_change"
    CONSTRAINT_APPROACH = "constraint_approach"
    DEMAND_FORECAST = "demand_forecast"
    MAINTENANCE_RECOMMENDATION = "maintenance_recommendation"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"


class ConstraintType(str, Enum):
    """Constraint type enumeration."""
    PRESSURE_MIN = "pressure_min"
    PRESSURE_MAX = "pressure_max"
    RATE_OF_CHANGE = "rate_of_change"
    DOWNSTREAM_DEMAND = "downstream_demand"
    UPSTREAM_AVAILABILITY = "upstream_availability"


# =============================================================================
# DATA MODELS
# =============================================================================

class PRVConfiguration(BaseModel):
    """PRV configuration parameters."""

    prv_id: str = Field(..., description="PRV equipment ID")
    upstream_header_id: str = Field(
        ...,
        description="Upstream header identifier"
    )
    downstream_header_id: str = Field(
        ...,
        description="Downstream header identifier"
    )
    design_inlet_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Design inlet pressure (kPa)"
    )
    design_outlet_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Design outlet pressure (kPa)"
    )
    max_flow_capacity_kg_s: float = Field(
        ...,
        ge=0,
        description="Maximum flow capacity (kg/s)"
    )
    cv_rating: float = Field(..., ge=0, description="Valve Cv rating")
    response_time_s: float = Field(
        default=5.0,
        ge=0,
        description="Valve response time (s)"
    )


class PRVOperatingState(BaseModel):
    """Current PRV operating state."""

    prv_id: str = Field(..., description="PRV equipment ID")
    state: PRVState = Field(..., description="Current PRV state")
    inlet_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Inlet pressure (kPa)"
    )
    outlet_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Outlet pressure (kPa)"
    )
    outlet_setpoint_kpa: float = Field(
        ...,
        ge=0,
        description="Outlet pressure setpoint (kPa)"
    )
    valve_position_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Valve position (%)"
    )
    flow_rate_kg_s: float = Field(..., ge=0, description="Flow rate (kg/s)")
    inlet_temperature_c: float = Field(
        ...,
        description="Inlet temperature (C)"
    )
    outlet_temperature_c: float = Field(
        ...,
        description="Outlet temperature (C)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="State timestamp"
    )


class PressureConstraints(BaseModel):
    """Pressure control constraints."""

    min_downstream_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Minimum downstream pressure (kPa)"
    )
    max_downstream_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Maximum downstream pressure (kPa)"
    )
    max_pressure_rate_kpa_per_min: float = Field(
        default=50.0,
        ge=0,
        description="Maximum pressure change rate (kPa/min)"
    )
    min_differential_pressure_kpa: float = Field(
        default=50.0,
        ge=0,
        description="Minimum pressure differential for stable operation"
    )
    alarm_low_kpa: float = Field(
        ...,
        ge=0,
        description="Low pressure alarm threshold (kPa)"
    )
    alarm_high_kpa: float = Field(
        ...,
        ge=0,
        description="High pressure alarm threshold (kPa)"
    )


class DemandForecast(BaseModel):
    """Steam demand forecast."""

    forecast_id: str = Field(..., description="Forecast ID")
    header_id: str = Field(..., description="Header identifier")
    forecast_horizon_minutes: int = Field(
        default=60,
        ge=1,
        description="Forecast horizon (minutes)"
    )
    predicted_demand_kg_s: List[float] = Field(
        ...,
        description="Predicted demand for each time step"
    )
    time_step_minutes: int = Field(
        default=5,
        description="Time step between predictions (minutes)"
    )
    confidence_level: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="Forecast confidence level"
    )
    peak_demand_kg_s: float = Field(
        ...,
        ge=0,
        description="Peak predicted demand (kg/s)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Forecast generation timestamp"
    )


class Setpoint(BaseModel):
    """PRV setpoint with metadata."""

    setpoint_id: str = Field(..., description="Setpoint ID")
    value_kpa: float = Field(..., ge=0, description="Setpoint value (kPa)")
    source: str = Field(
        ...,
        description="Setpoint source (operator, optimizer, cascade)"
    )
    effective_time: datetime = Field(
        default_factory=datetime.now,
        description="Effective time"
    )
    expiration_time: Optional[datetime] = Field(
        None,
        description="Expiration time (if temporary)"
    )
    reason: str = Field(default="", description="Reason for setpoint")
    authorization_id: Optional[str] = Field(
        None,
        description="Authorization ID"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ValidationResult(BaseModel):
    """Setpoint validation result."""

    validation_id: str = Field(..., description="Validation ID")
    status: ValidationStatus = Field(..., description="Validation status")
    current_setpoint_kpa: float = Field(
        ...,
        description="Current setpoint (kPa)"
    )
    proposed_setpoint_kpa: float = Field(
        ...,
        description="Proposed setpoint (kPa)"
    )
    constraint_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Constraint check results"
    )
    violations: List[str] = Field(
        default_factory=list,
        description="Constraint violations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    recommended_value_kpa: Optional[float] = Field(
        None,
        description="Recommended value if proposed is invalid"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Validation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class PRVAdvisory(BaseModel):
    """PRV operational advisory."""

    advisory_id: str = Field(..., description="Advisory ID")
    advisory_type: AdvisoryType = Field(..., description="Advisory type")
    prv_id: str = Field(..., description="PRV equipment ID")
    title: str = Field(..., description="Advisory title")
    description: str = Field(..., description="Detailed description")
    current_value: float = Field(
        ...,
        description="Current value (context-dependent)"
    )
    recommended_value: float = Field(
        ...,
        description="Recommended value"
    )
    unit: str = Field(..., description="Value unit")
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
    expected_benefit: str = Field(
        default="",
        description="Expected benefit description"
    )
    safety_validated: bool = Field(
        default=False,
        description="Validated against safety constraints"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Advisory timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# PRV CONTROLLER
# =============================================================================

class PRVController:
    """
    Pressure Reducing Valve (PRV) setpoint management.

    This controller manages PRV setpoints for downstream pressure control,
    incorporating demand forecasting and constraint enforcement.

    Control Features:
        - Demand-based setpoint calculation
        - Pressure constraint enforcement
        - Rate-of-change limiting
        - Advisory generation for operators

    Safety Features:
        - Never exceeds pressure limits
        - Minimum differential pressure enforcement
        - Alarm threshold monitoring
        - All changes logged for audit trail

    Attributes:
        config: PRV configuration
        constraints: Pressure constraints
        _current_setpoint: Current active setpoint
        _setpoint_history: History of setpoint changes

    Example:
        >>> controller = PRVController(config, constraints)
        >>> setpoint = controller.compute_downstream_setpoint(
        ...     demand_forecast=forecast,
        ...     header_pressure=1200.0
        ... )
    """

    def __init__(
        self,
        config: PRVConfiguration,
        constraints: PressureConstraints
    ):
        """
        Initialize PRVController.

        Args:
            config: PRV configuration parameters
            constraints: Pressure control constraints
        """
        self.config = config
        self.constraints = constraints
        self._current_setpoint: Optional[Setpoint] = None
        self._setpoint_history: List[Setpoint] = []
        self._max_history_size = 500
        self._last_update_time = datetime.now()

        logger.info(
            f"PRVController initialized for {config.prv_id}: "
            f"design outlet={config.design_outlet_pressure_kpa}kPa, "
            f"constraints=[{constraints.min_downstream_pressure_kpa}, "
            f"{constraints.max_downstream_pressure_kpa}]kPa"
        )

    def compute_downstream_setpoint(
        self,
        demand_forecast: DemandForecast,
        header_pressure_kpa: float,
        current_state: Optional[PRVOperatingState] = None
    ) -> Setpoint:
        """
        Compute downstream pressure setpoint based on demand forecast.

        This method calculates optimal downstream setpoint considering
        predicted demand, header pressure, and operational constraints.

        Args:
            demand_forecast: Steam demand forecast
            header_pressure_kpa: Current header pressure (kPa)
            current_state: Current PRV operating state

        Returns:
            Setpoint: Computed setpoint with metadata
        """
        start_time = datetime.now()

        # Step 1: Determine base setpoint from demand
        base_setpoint = self._calculate_demand_based_setpoint(
            demand_forecast, header_pressure_kpa
        )

        # Step 2: Apply constraints
        constrained_setpoint = self._apply_pressure_constraints(
            base_setpoint, header_pressure_kpa
        )

        # Step 3: Apply rate limiting if we have a current setpoint
        if self._current_setpoint is not None:
            time_delta = (start_time - self._last_update_time).total_seconds() / 60.0
            max_change = self.constraints.max_pressure_rate_kpa_per_min * time_delta

            if abs(constrained_setpoint - self._current_setpoint.value_kpa) > max_change:
                if constrained_setpoint > self._current_setpoint.value_kpa:
                    constrained_setpoint = self._current_setpoint.value_kpa + max_change
                else:
                    constrained_setpoint = self._current_setpoint.value_kpa - max_change

                logger.info(
                    f"Setpoint rate limited: target={base_setpoint:.1f}kPa, "
                    f"limited={constrained_setpoint:.1f}kPa"
                )

        # Generate setpoint ID
        setpoint_id = hashlib.sha256(
            f"PRV_{self.config.prv_id}_{constrained_setpoint}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Create setpoint
        setpoint = Setpoint(
            setpoint_id=setpoint_id,
            value_kpa=constrained_setpoint,
            source="optimizer",
            effective_time=datetime.now(),
            reason=f"Demand forecast peak: {demand_forecast.peak_demand_kg_s:.2f} kg/s"
        )

        # Calculate provenance hash
        setpoint.provenance_hash = hashlib.sha256(
            f"{setpoint_id}|{constrained_setpoint}|{header_pressure_kpa}".encode()
        ).hexdigest()

        # Update internal state
        self._current_setpoint = setpoint
        self._add_to_history(setpoint)
        self._last_update_time = start_time

        logger.info(
            f"Computed setpoint {setpoint_id}: {constrained_setpoint:.1f}kPa "
            f"for header pressure {header_pressure_kpa:.1f}kPa"
        )

        return setpoint

    def validate_setpoint_change(
        self,
        current_kpa: float,
        proposed_kpa: float,
        constraints: Optional[PressureConstraints] = None
    ) -> ValidationResult:
        """
        Validate a proposed setpoint change against constraints.

        This method performs comprehensive validation of setpoint changes
        to ensure safe operation within all defined constraints.

        Args:
            current_kpa: Current setpoint value (kPa)
            proposed_kpa: Proposed setpoint value (kPa)
            constraints: Override constraints (uses instance if None)

        Returns:
            ValidationResult: Validation result with details
        """
        start_time = datetime.now()
        constraints = constraints or self.constraints

        violations = []
        warnings = []
        constraint_checks = {}

        # Check 1: Minimum pressure constraint
        if proposed_kpa < constraints.min_downstream_pressure_kpa:
            violations.append(
                f"Proposed {proposed_kpa:.1f}kPa below minimum "
                f"{constraints.min_downstream_pressure_kpa:.1f}kPa"
            )
            constraint_checks["min_pressure"] = False
        else:
            constraint_checks["min_pressure"] = True

        # Check 2: Maximum pressure constraint
        if proposed_kpa > constraints.max_downstream_pressure_kpa:
            violations.append(
                f"Proposed {proposed_kpa:.1f}kPa above maximum "
                f"{constraints.max_downstream_pressure_kpa:.1f}kPa"
            )
            constraint_checks["max_pressure"] = False
        else:
            constraint_checks["max_pressure"] = True

        # Check 3: Rate of change constraint
        time_delta = (start_time - self._last_update_time).total_seconds() / 60.0
        if time_delta > 0:
            actual_rate = abs(proposed_kpa - current_kpa) / time_delta
            if actual_rate > constraints.max_pressure_rate_kpa_per_min:
                violations.append(
                    f"Rate of change {actual_rate:.1f}kPa/min exceeds maximum "
                    f"{constraints.max_pressure_rate_kpa_per_min:.1f}kPa/min"
                )
                constraint_checks["rate_of_change"] = False
            else:
                constraint_checks["rate_of_change"] = True

        # Check 4: Alarm threshold proximity
        if proposed_kpa < constraints.alarm_low_kpa * 1.1:
            warnings.append(
                f"Proposed {proposed_kpa:.1f}kPa approaching low alarm "
                f"{constraints.alarm_low_kpa:.1f}kPa"
            )
        if proposed_kpa > constraints.alarm_high_kpa * 0.9:
            warnings.append(
                f"Proposed {proposed_kpa:.1f}kPa approaching high alarm "
                f"{constraints.alarm_high_kpa:.1f}kPa"
            )

        # Determine validation status
        if violations:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        # Calculate recommended value if invalid
        recommended_value = None
        if status == ValidationStatus.INVALID:
            recommended_value = max(
                constraints.min_downstream_pressure_kpa,
                min(constraints.max_downstream_pressure_kpa, proposed_kpa)
            )

        # Generate validation ID
        validation_id = hashlib.sha256(
            f"VAL_{current_kpa}_{proposed_kpa}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = ValidationResult(
            validation_id=validation_id,
            status=status,
            current_setpoint_kpa=current_kpa,
            proposed_setpoint_kpa=proposed_kpa,
            constraint_checks=constraint_checks,
            violations=violations,
            warnings=warnings,
            recommended_value_kpa=recommended_value,
            timestamp=datetime.now()
        )

        # Calculate provenance hash
        result.provenance_hash = hashlib.sha256(
            f"{validation_id}|{status.value}|{len(violations)}".encode()
        ).hexdigest()

        logger.info(
            f"Validation {validation_id}: status={status.value}, "
            f"violations={len(violations)}, warnings={len(warnings)}"
        )

        return result

    def generate_advisory(
        self,
        proposed_change: Dict[str, Any]
    ) -> PRVAdvisory:
        """
        Generate operational advisory for proposed PRV change.

        This method creates detailed advisories for operators regarding
        proposed setpoint changes, including expected benefits and risks.

        Args:
            proposed_change: Dictionary containing:
                - proposed_setpoint_kpa: Proposed setpoint value
                - reason: Reason for change
                - expected_benefit: Expected benefit description

        Returns:
            PRVAdvisory: Advisory for operator review
        """
        start_time = datetime.now()

        proposed_kpa = proposed_change.get("proposed_setpoint_kpa", 0)
        reason = proposed_change.get("reason", "")
        expected_benefit = proposed_change.get("expected_benefit", "")

        # Get current setpoint value
        current_kpa = (
            self._current_setpoint.value_kpa
            if self._current_setpoint
            else self.config.design_outlet_pressure_kpa
        )

        # Validate the proposed change
        validation = self.validate_setpoint_change(current_kpa, proposed_kpa)

        # Determine advisory type
        if abs(proposed_kpa - current_kpa) > 10:
            advisory_type = AdvisoryType.SETPOINT_CHANGE
        elif validation.warnings:
            advisory_type = AdvisoryType.CONSTRAINT_APPROACH
        else:
            advisory_type = AdvisoryType.OPTIMIZATION_OPPORTUNITY

        # Generate advisory ID
        advisory_id = hashlib.sha256(
            f"ADV_{self.config.prv_id}_{proposed_kpa}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Determine priority based on validation status
        if validation.status == ValidationStatus.INVALID:
            priority = 2
            safety_validated = False
        elif validation.status == ValidationStatus.WARNING:
            priority = 3
            safety_validated = True
        else:
            priority = 4
            safety_validated = True

        # Build description
        description = self._build_advisory_description(
            current_kpa, proposed_kpa, validation, reason
        )

        advisory = PRVAdvisory(
            advisory_id=advisory_id,
            advisory_type=advisory_type,
            prv_id=self.config.prv_id,
            title=f"PRV Setpoint Change: {current_kpa:.0f} -> {proposed_kpa:.0f} kPa",
            description=description,
            current_value=current_kpa,
            recommended_value=proposed_kpa if safety_validated else (
                validation.recommended_value_kpa or current_kpa
            ),
            unit="kPa",
            priority=priority,
            requires_confirmation=True,
            expected_benefit=expected_benefit,
            safety_validated=safety_validated,
            timestamp=datetime.now()
        )

        # Calculate provenance hash
        advisory.provenance_hash = hashlib.sha256(
            f"{advisory_id}|{advisory_type.value}|{safety_validated}".encode()
        ).hexdigest()

        logger.info(
            f"Advisory {advisory_id}: type={advisory_type.value}, "
            f"priority={priority}, safety_validated={safety_validated}"
        )

        return advisory

    def get_current_setpoint(self) -> Optional[Setpoint]:
        """Get current active setpoint."""
        return self._current_setpoint

    def get_setpoint_history(
        self,
        time_window_minutes: int = 60
    ) -> List[Setpoint]:
        """
        Get setpoint history within time window.

        Args:
            time_window_minutes: Time window in minutes

        Returns:
            List of setpoints within the time window
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        return [
            sp for sp in self._setpoint_history
            if sp.effective_time >= cutoff_time
        ]

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_demand_based_setpoint(
        self,
        demand_forecast: DemandForecast,
        header_pressure_kpa: float
    ) -> float:
        """
        Calculate setpoint based on demand forecast.

        Args:
            demand_forecast: Steam demand forecast
            header_pressure_kpa: Current header pressure

        Returns:
            float: Calculated setpoint (kPa)
        """
        # Base setpoint from design value
        base_setpoint = self.config.design_outlet_pressure_kpa

        # Adjust based on demand ratio
        design_flow = self.config.max_flow_capacity_kg_s
        demand_ratio = demand_forecast.peak_demand_kg_s / design_flow if design_flow > 0 else 0

        # Higher demand may require slightly higher setpoint for margin
        # This is a simplified model - production would use detailed process models
        demand_adjustment = 0.0
        if demand_ratio > 0.8:
            # High demand - increase setpoint for headroom
            demand_adjustment = (demand_ratio - 0.8) * 20.0
        elif demand_ratio < 0.3:
            # Low demand - can operate at lower setpoint
            demand_adjustment = -(0.3 - demand_ratio) * 15.0

        calculated_setpoint = base_setpoint + demand_adjustment

        logger.debug(
            f"Demand-based setpoint: base={base_setpoint:.1f}, "
            f"adjustment={demand_adjustment:.1f}, result={calculated_setpoint:.1f}"
        )

        return calculated_setpoint

    def _apply_pressure_constraints(
        self,
        setpoint_kpa: float,
        header_pressure_kpa: float
    ) -> float:
        """
        Apply pressure constraints to setpoint.

        Args:
            setpoint_kpa: Unconstrained setpoint
            header_pressure_kpa: Current header pressure

        Returns:
            float: Constrained setpoint (kPa)
        """
        # Apply min/max constraints
        constrained = max(
            self.constraints.min_downstream_pressure_kpa,
            min(self.constraints.max_downstream_pressure_kpa, setpoint_kpa)
        )

        # Ensure minimum differential pressure
        max_allowed = header_pressure_kpa - self.constraints.min_differential_pressure_kpa
        if constrained > max_allowed:
            constrained = max_allowed
            logger.info(
                f"Setpoint limited by differential pressure: {constrained:.1f}kPa"
            )

        return constrained

    def _build_advisory_description(
        self,
        current_kpa: float,
        proposed_kpa: float,
        validation: ValidationResult,
        reason: str
    ) -> str:
        """Build detailed advisory description."""
        lines = []

        change_direction = "increase" if proposed_kpa > current_kpa else "decrease"
        change_magnitude = abs(proposed_kpa - current_kpa)

        lines.append(
            f"Proposed {change_direction} of {change_magnitude:.1f} kPa "
            f"from {current_kpa:.1f} to {proposed_kpa:.1f} kPa."
        )

        if reason:
            lines.append(f"Reason: {reason}")

        if validation.violations:
            lines.append("\nConstraint Violations:")
            for v in validation.violations:
                lines.append(f"  - {v}")

        if validation.warnings:
            lines.append("\nWarnings:")
            for w in validation.warnings:
                lines.append(f"  - {w}")

        if validation.recommended_value_kpa:
            lines.append(
                f"\nRecommended value: {validation.recommended_value_kpa:.1f} kPa"
            )

        return "\n".join(lines)

    def _add_to_history(self, setpoint: Setpoint) -> None:
        """Add setpoint to history with size limit."""
        self._setpoint_history.append(setpoint)
        if len(self._setpoint_history) > self._max_history_size:
            self._setpoint_history = self._setpoint_history[-self._max_history_size:]
