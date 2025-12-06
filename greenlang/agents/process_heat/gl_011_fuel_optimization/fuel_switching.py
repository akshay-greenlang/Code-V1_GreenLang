"""
GL-011 FUELCRAFT - Fuel Switching Controller

This module provides automated fuel switching logic with economic trigger
points, equipment transition management, and safety interlocks.

Features:
    - Economic trigger analysis for fuel switching
    - Equipment transition state machine
    - Safety interlock management during transitions
    - Purge cycle coordination
    - Operator confirmation workflows
    - Switch history and analytics

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_switching import (
    ...     FuelSwitchingController,
    ...     SwitchingInput,
    ... )
    >>>
    >>> controller = FuelSwitchingController(config)
    >>> result = controller.evaluate_switch(input_data)
    >>> if result.recommended:
    ...     print(f"Switch to {result.recommended_fuel}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    SwitchingConfig,
    SwitchingMode,
    FuelType,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelPrice,
    SwitchingRecommendation,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SwitchingState(Enum):
    """Fuel switching state machine states."""
    IDLE = auto()
    EVALUATING = auto()
    PENDING_APPROVAL = auto()
    PREPARING = auto()
    PURGING = auto()
    TRANSITIONING = auto()
    STABILIZING = auto()
    COMPLETE = auto()
    ABORTED = auto()
    FAILED = auto()


class TriggerType(Enum):
    """Types of switching triggers."""
    ECONOMIC = "economic"
    SUPPLY = "supply"
    EMISSION_LIMIT = "emission_limit"
    EQUIPMENT = "equipment"
    OPERATOR = "operator"
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"


# Safety interlock flags
SAFETY_INTERLOCKS = {
    "flame_proven": "Flame detector signal present",
    "fuel_pressure_ok": "Fuel supply pressure in range",
    "purge_complete": "Pre/post purge cycle completed",
    "no_active_alarms": "No active safety alarms",
    "combustion_air_ok": "Combustion air flow sufficient",
    "draft_ok": "Furnace draft in range",
    "operator_present": "Operator acknowledgment received",
}


# =============================================================================
# DATA MODELS
# =============================================================================

class SwitchingInput(BaseModel):
    """Input for fuel switching evaluation."""

    # Current state
    current_fuel: str = Field(..., description="Current fuel type")
    current_cost_usd_mmbtu: float = Field(
        ...,
        ge=0,
        description="Current fuel cost ($/MMBTU)"
    )
    current_heat_input_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Current heat input (MMBTU/hr)"
    )

    # Alternative fuels
    available_fuels: List[str] = Field(
        ...,
        description="Available alternative fuels"
    )
    fuel_prices: Dict[str, FuelPrice] = Field(
        ...,
        description="Prices for available fuels"
    )

    # Equipment status
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_online: bool = Field(default=True, description="Equipment online")
    current_load_pct: float = Field(
        default=100.0,
        ge=0,
        le=120,
        description="Current load percentage"
    )

    # Timing
    time_on_current_fuel_hours: float = Field(
        default=0.0,
        ge=0,
        description="Hours on current fuel"
    )
    switches_today: int = Field(
        default=0,
        ge=0,
        description="Number of switches today"
    )

    # Safety status
    safety_interlocks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Safety interlock status"
    )

    # Constraints
    max_transition_time_minutes: Optional[int] = Field(
        default=None,
        description="Maximum allowed transition time"
    )
    required_availability_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Required equipment availability"
    )


class SwitchingOutput(BaseModel):
    """Output from fuel switching evaluation."""

    # Recommendation
    recommended: bool = Field(..., description="Switch recommended")
    current_fuel: str = Field(..., description="Current fuel")
    recommended_fuel: str = Field(..., description="Recommended fuel")
    trigger_type: TriggerType = Field(..., description="Trigger type")
    trigger_reason: str = Field(..., description="Trigger reason")

    # Economic analysis
    current_cost_usd_hr: float = Field(..., description="Current hourly cost")
    recommended_cost_usd_hr: float = Field(..., description="Recommended hourly cost")
    savings_usd_hr: float = Field(..., description="Potential savings ($/hr)")
    savings_pct: float = Field(..., description="Savings percentage")
    payback_hours: Optional[float] = Field(
        default=None,
        description="Payback period (hours)"
    )
    annual_savings_usd: float = Field(
        default=0.0,
        description="Projected annual savings"
    )

    # Transition details
    transition_time_minutes: int = Field(..., description="Transition time")
    requires_purge: bool = Field(..., description="Requires purge cycle")
    requires_load_reduction: bool = Field(
        default=False,
        description="Requires load reduction"
    )
    efficiency_impact_pct: float = Field(
        default=0.0,
        description="Efficiency change during transition"
    )

    # Safety
    safety_status: str = Field(default="ok", description="Safety status")
    safety_checks_passed: bool = Field(..., description="All safety checks passed")
    safety_warnings: List[str] = Field(
        default_factory=list,
        description="Safety warnings"
    )
    blocked_interlocks: List[str] = Field(
        default_factory=list,
        description="Blocked safety interlocks"
    )

    # Approval
    requires_operator_approval: bool = Field(..., description="Needs operator approval")
    approval_timeout: datetime = Field(..., description="Approval timeout")

    # State
    switching_state: SwitchingState = Field(
        default=SwitchingState.IDLE,
        description="Current switching state"
    )

    # Provenance
    evaluation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Evaluation identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Evaluation timestamp"
    )
    provenance_hash: str = Field(..., description="Calculation provenance")

    class Config:
        use_enum_values = True


class SwitchingTrigger(BaseModel):
    """Fuel switching trigger event."""

    trigger_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Trigger identifier"
    )
    trigger_type: TriggerType = Field(..., description="Trigger type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Trigger timestamp"
    )

    current_fuel: str = Field(..., description="Current fuel")
    target_fuel: str = Field(..., description="Target fuel")

    # Trigger details
    price_differential_pct: Optional[float] = Field(
        default=None,
        description="Price differential (%)"
    )
    estimated_savings_usd_hr: Optional[float] = Field(
        default=None,
        description="Estimated savings ($/hr)"
    )

    # Status
    active: bool = Field(default=True, description="Trigger active")
    executed: bool = Field(default=False, description="Switch executed")
    execution_time: Optional[datetime] = Field(
        default=None,
        description="Execution timestamp"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# SWITCH HISTORY
# =============================================================================

@dataclass
class SwitchRecord:
    """Record of a fuel switch event."""
    switch_id: str
    timestamp: datetime
    from_fuel: str
    to_fuel: str
    trigger_type: TriggerType
    duration_minutes: float
    success: bool
    savings_realized_usd: float = 0.0
    notes: str = ""


class SwitchHistory:
    """Maintains history of fuel switching events."""

    def __init__(self, max_records: int = 1000) -> None:
        """Initialize switch history."""
        self._records: List[SwitchRecord] = []
        self._max_records = max_records

    def add_record(self, record: SwitchRecord) -> None:
        """Add a switch record."""
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

    def get_switches_today(self) -> int:
        """Get number of switches today."""
        today = datetime.now(timezone.utc).date()
        return sum(
            1 for r in self._records
            if r.timestamp.date() == today
        )

    def get_time_on_fuel(self, fuel: str) -> float:
        """Get hours on specified fuel since last switch."""
        if not self._records:
            return 0.0

        last_record = self._records[-1]
        if last_record.to_fuel == fuel:
            elapsed = (datetime.now(timezone.utc) - last_record.timestamp)
            return elapsed.total_seconds() / 3600

        return 0.0

    def get_success_rate(self, days: int = 30) -> float:
        """Get switch success rate over period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [r for r in self._records if r.timestamp >= cutoff]

        if not recent:
            return 1.0

        successes = sum(1 for r in recent if r.success)
        return successes / len(recent)

    def get_total_savings(self, days: int = 30) -> float:
        """Get total savings from switches over period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return sum(
            r.savings_realized_usd
            for r in self._records
            if r.timestamp >= cutoff and r.success
        )


# =============================================================================
# FUEL SWITCHING CONTROLLER
# =============================================================================

class FuelSwitchingController:
    """
    Automated fuel switching controller.

    This controller manages fuel switching decisions including economic
    analysis, safety interlocks, and transition state management.

    Features:
        - Economic trigger evaluation
        - Safety interlock verification
        - Transition state machine
        - Operator approval workflow
        - Switch history and analytics

    Example:
        >>> controller = FuelSwitchingController(config)
        >>> result = controller.evaluate_switch(input_data)
        >>> if result.recommended:
        ...     if controller.can_execute():
        ...         controller.execute_switch()
    """

    def __init__(self, config: SwitchingConfig) -> None:
        """
        Initialize the fuel switching controller.

        Args:
            config: Switching configuration
        """
        self.config = config
        self._state = SwitchingState.IDLE
        self._history = SwitchHistory()
        self._pending_switch: Optional[SwitchingOutput] = None
        self._active_triggers: List[SwitchingTrigger] = []
        self._evaluation_count = 0

        logger.info(
            f"FuelSwitchingController initialized "
            f"(mode: {config.mode}, trigger: {config.price_differential_trigger_pct}%)"
        )

    @property
    def state(self) -> SwitchingState:
        """Get current switching state."""
        return self._state

    @property
    def is_idle(self) -> bool:
        """Check if controller is idle."""
        return self._state == SwitchingState.IDLE

    def evaluate_switch(
        self,
        input_data: SwitchingInput,
    ) -> SwitchingOutput:
        """
        Evaluate whether a fuel switch is recommended.

        Analyzes economic benefit, timing constraints, and safety
        conditions to determine if switching is advisable.

        Args:
            input_data: Current operating conditions

        Returns:
            SwitchingOutput with recommendation and analysis
        """
        self._evaluation_count += 1
        self._state = SwitchingState.EVALUATING

        logger.debug(
            f"Evaluating switch from {input_data.current_fuel}, "
            f"{len(input_data.available_fuels)} alternatives"
        )

        # Check if switching is enabled
        if self.config.mode == SwitchingMode.DISABLED:
            return self._create_no_switch_output(
                input_data,
                "Fuel switching disabled"
            )

        # Check timing constraints
        timing_check = self._check_timing_constraints(input_data)
        if not timing_check[0]:
            return self._create_no_switch_output(input_data, timing_check[1])

        # Check safety interlocks
        safety_check = self._check_safety_interlocks(input_data)
        if not safety_check[0]:
            output = self._create_no_switch_output(input_data, "Safety check failed")
            output.safety_checks_passed = False
            output.blocked_interlocks = safety_check[2]
            return output

        # Find best alternative fuel
        best_alternative = self._find_best_alternative(input_data)
        if best_alternative is None:
            return self._create_no_switch_output(
                input_data,
                "No better alternative found"
            )

        target_fuel, target_price, savings = best_alternative

        # Check economic trigger
        savings_pct = (savings / input_data.current_cost_usd_mmbtu) * 100
        if savings_pct < self.config.price_differential_trigger_pct:
            return self._create_no_switch_output(
                input_data,
                f"Savings {savings_pct:.1f}% below trigger {self.config.price_differential_trigger_pct}%"
            )

        # Check minimum savings
        hourly_savings = savings * input_data.current_heat_input_mmbtu_hr
        if hourly_savings < self.config.min_savings_usd_hr:
            return self._create_no_switch_output(
                input_data,
                f"Hourly savings ${hourly_savings:.2f} below minimum ${self.config.min_savings_usd_hr}"
            )

        # Calculate transition costs
        transition_cost = self._calculate_transition_cost(
            input_data.current_fuel,
            target_fuel,
            input_data.current_heat_input_mmbtu_hr
        )

        # Calculate payback
        payback_hours = transition_cost / hourly_savings if hourly_savings > 0 else float("inf")

        if payback_hours > self.config.payback_period_hours:
            return self._create_no_switch_output(
                input_data,
                f"Payback {payback_hours:.1f}h exceeds maximum {self.config.payback_period_hours}h"
            )

        # Create recommendation
        now = datetime.now(timezone.utc)
        current_hourly = input_data.current_cost_usd_mmbtu * input_data.current_heat_input_mmbtu_hr
        target_hourly = target_price * input_data.current_heat_input_mmbtu_hr

        # Calculate annual savings
        annual_savings = hourly_savings * 8760 * 0.9  # 90% availability

        # Determine if approval is required
        requires_approval = (
            self.config.mode == SwitchingMode.SEMI_AUTOMATIC or
            self.config.operator_confirmation_required
        )

        # Create provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            target_fuel,
            {"savings": savings, "payback": payback_hours}
        )

        output = SwitchingOutput(
            recommended=True,
            current_fuel=input_data.current_fuel,
            recommended_fuel=target_fuel,
            trigger_type=TriggerType.ECONOMIC,
            trigger_reason=f"Price savings of {savings_pct:.1f}%",
            current_cost_usd_hr=round(current_hourly, 2),
            recommended_cost_usd_hr=round(target_hourly, 2),
            savings_usd_hr=round(hourly_savings, 2),
            savings_pct=round(savings_pct, 2),
            payback_hours=round(payback_hours, 2),
            annual_savings_usd=round(annual_savings, 0),
            transition_time_minutes=self.config.transition_duration_minutes,
            requires_purge=self.config.require_purge,
            safety_checks_passed=True,
            requires_operator_approval=requires_approval,
            approval_timeout=now + timedelta(
                minutes=self.config.confirmation_timeout_minutes
            ),
            switching_state=SwitchingState.PENDING_APPROVAL if requires_approval else SwitchingState.PREPARING,
            provenance_hash=provenance_hash,
        )

        # Store pending switch
        self._pending_switch = output

        # Update state
        if requires_approval:
            self._state = SwitchingState.PENDING_APPROVAL
        else:
            self._state = SwitchingState.PREPARING

        # Create trigger
        trigger = SwitchingTrigger(
            trigger_type=TriggerType.ECONOMIC,
            current_fuel=input_data.current_fuel,
            target_fuel=target_fuel,
            price_differential_pct=round(savings_pct, 2),
            estimated_savings_usd_hr=round(hourly_savings, 2),
        )
        self._active_triggers.append(trigger)

        return output

    def approve_switch(self, operator_id: str) -> bool:
        """
        Approve pending switch (operator confirmation).

        Args:
            operator_id: Operator identifier

        Returns:
            True if approval successful
        """
        if self._state != SwitchingState.PENDING_APPROVAL:
            logger.warning("No pending switch to approve")
            return False

        if self._pending_switch is None:
            logger.warning("No pending switch data")
            return False

        # Check timeout
        if datetime.now(timezone.utc) > self._pending_switch.approval_timeout:
            logger.warning("Switch approval timeout expired")
            self._state = SwitchingState.IDLE
            self._pending_switch = None
            return False

        logger.info(f"Switch approved by operator {operator_id}")
        self._state = SwitchingState.PREPARING
        return True

    def can_execute(self) -> bool:
        """Check if switch can be executed."""
        return self._state == SwitchingState.PREPARING and self._pending_switch is not None

    def execute_switch(self) -> bool:
        """
        Execute the pending fuel switch.

        Returns:
            True if switch initiated successfully
        """
        if not self.can_execute():
            logger.warning("Cannot execute switch in current state")
            return False

        logger.info(
            f"Executing switch: {self._pending_switch.current_fuel} -> "
            f"{self._pending_switch.recommended_fuel}"
        )

        # Start purge if required
        if self._pending_switch.requires_purge:
            self._state = SwitchingState.PURGING
            # In real implementation, would coordinate with BMS
        else:
            self._state = SwitchingState.TRANSITIONING

        return True

    def complete_switch(self, success: bool, notes: str = "") -> None:
        """
        Mark switch as complete.

        Args:
            success: Whether switch was successful
            notes: Optional notes
        """
        if self._pending_switch is None:
            return

        # Record in history
        record = SwitchRecord(
            switch_id=self._pending_switch.evaluation_id,
            timestamp=datetime.now(timezone.utc),
            from_fuel=self._pending_switch.current_fuel,
            to_fuel=self._pending_switch.recommended_fuel,
            trigger_type=TriggerType(self._pending_switch.trigger_type),
            duration_minutes=self.config.transition_duration_minutes,
            success=success,
            savings_realized_usd=self._pending_switch.savings_usd_hr if success else 0.0,
            notes=notes,
        )
        self._history.add_record(record)

        # Update trigger
        for trigger in self._active_triggers:
            if trigger.target_fuel == self._pending_switch.recommended_fuel:
                trigger.executed = success
                trigger.execution_time = datetime.now(timezone.utc)
                trigger.active = False

        # Reset state
        self._state = SwitchingState.COMPLETE if success else SwitchingState.FAILED
        self._pending_switch = None

        logger.info(
            f"Switch completed: {'success' if success else 'failed'} - {notes}"
        )

    def abort_switch(self, reason: str = "") -> None:
        """
        Abort pending switch.

        Args:
            reason: Abort reason
        """
        logger.warning(f"Aborting switch: {reason}")
        self._state = SwitchingState.ABORTED
        self._pending_switch = None

    def reset(self) -> None:
        """Reset controller to idle state."""
        self._state = SwitchingState.IDLE
        self._pending_switch = None
        self._active_triggers = [t for t in self._active_triggers if t.active]

    def _check_timing_constraints(
        self,
        input_data: SwitchingInput,
    ) -> Tuple[bool, str]:
        """Check timing constraints for switching."""
        # Check minimum run time on current fuel
        if input_data.time_on_current_fuel_hours < self.config.min_run_time_hours:
            return (
                False,
                f"Minimum run time not met: "
                f"{input_data.time_on_current_fuel_hours:.1f}h < "
                f"{self.config.min_run_time_hours}h"
            )

        # Check daily switch limit
        if input_data.switches_today >= self.config.max_switches_per_day:
            return (
                False,
                f"Daily switch limit reached: {input_data.switches_today}"
            )

        return (True, "")

    def _check_safety_interlocks(
        self,
        input_data: SwitchingInput,
    ) -> Tuple[bool, str, List[str]]:
        """Check safety interlocks for switching."""
        if not self.config.safety_interlock_enabled:
            return (True, "", [])

        blocked = []
        warnings = []

        # Check required interlocks
        required_interlocks = [
            "flame_proven",
            "fuel_pressure_ok",
            "no_active_alarms",
        ]

        for interlock in required_interlocks:
            status = input_data.safety_interlocks.get(interlock, False)
            if not status:
                blocked.append(interlock)

        if blocked:
            return (
                False,
                f"Blocked interlocks: {', '.join(blocked)}",
                blocked
            )

        return (True, "", [])

    def _find_best_alternative(
        self,
        input_data: SwitchingInput,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Find best alternative fuel based on cost.

        Returns:
            Tuple of (fuel_type, price, savings) or None
        """
        current_price = input_data.current_cost_usd_mmbtu
        best_fuel = None
        best_price = current_price
        best_savings = 0.0

        for fuel in input_data.available_fuels:
            if fuel == input_data.current_fuel:
                continue

            if fuel not in input_data.fuel_prices:
                continue

            price = input_data.fuel_prices[fuel].total_price

            if price < best_price:
                best_fuel = fuel
                best_price = price
                best_savings = current_price - price

        if best_fuel is None:
            return None

        return (best_fuel, best_price, best_savings)

    def _calculate_transition_cost(
        self,
        from_fuel: str,
        to_fuel: str,
        heat_input_mmbtu_hr: float,
    ) -> float:
        """Calculate transition cost."""
        # Lost production during transition
        transition_hours = self.config.transition_duration_minutes / 60.0

        # Assume 50% efficiency loss during transition
        efficiency_loss_cost = heat_input_mmbtu_hr * transition_hours * 5.0  # $/MMBTU

        # Purge cost if required
        purge_cost = 0.0
        if self.config.require_purge:
            purge_hours = self.config.purge_duration_seconds / 3600.0
            purge_cost = heat_input_mmbtu_hr * purge_hours * 2.0

        return efficiency_loss_cost + purge_cost

    def _create_no_switch_output(
        self,
        input_data: SwitchingInput,
        reason: str,
    ) -> SwitchingOutput:
        """Create output for no switch recommendation."""
        now = datetime.now(timezone.utc)
        current_hourly = (
            input_data.current_cost_usd_mmbtu *
            input_data.current_heat_input_mmbtu_hr
        )

        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            input_data.current_fuel,
            {"recommended": False, "reason": reason}
        )

        self._state = SwitchingState.IDLE

        return SwitchingOutput(
            recommended=False,
            current_fuel=input_data.current_fuel,
            recommended_fuel=input_data.current_fuel,
            trigger_type=TriggerType.ECONOMIC,
            trigger_reason=reason,
            current_cost_usd_hr=round(current_hourly, 2),
            recommended_cost_usd_hr=round(current_hourly, 2),
            savings_usd_hr=0.0,
            savings_pct=0.0,
            transition_time_minutes=0,
            requires_purge=False,
            safety_checks_passed=True,
            requires_operator_approval=False,
            approval_timeout=now,
            switching_state=SwitchingState.IDLE,
            provenance_hash=provenance_hash,
        )

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        target_fuel: str,
        result: Dict[str, Any],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        data = {
            "controller": "FuelSwitchingController",
            "target_fuel": target_fuel,
            "result": result,
        }

        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def evaluation_count(self) -> int:
        """Get total evaluation count."""
        return self._evaluation_count

    @property
    def switches_today(self) -> int:
        """Get number of switches today."""
        return self._history.get_switches_today()

    @property
    def success_rate(self) -> float:
        """Get switch success rate."""
        return self._history.get_success_rate()

    @property
    def total_savings(self) -> float:
        """Get total savings from switches."""
        return self._history.get_total_savings()
