# -*- coding: utf-8 -*-
"""
TransferConstraints - Transfer and blending safety for GL-011 FuelCraft.

This module implements transfer safety validators including rate limits, minimum
stable flow constraints, interlock management, and transfer sequence validation.

Reference Standards:
    - API 2610: Design, Construction, Operation, Maintenance of Terminal Facilities
    - IEC 61511: Functional Safety - Safety Instrumented Systems
    - NFPA 30: Flammable and Combustible Liquids Code

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class TransferAction(str, Enum):
    """Actions for transfer safety control."""
    ALLOW = "allow"
    BLOCK = "block"
    REDUCE_RATE = "reduce_rate"
    EMERGENCY_STOP = "emergency_stop"


class InterlockState(str, Enum):
    """State of a safety interlock."""
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"
    BYPASSED = "bypassed"
    FAILED = "failed"


class ValveState(str, Enum):
    """State of a valve."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_OPEN = "partial"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"


class PumpState(str, Enum):
    """State of a pump."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    FAILED = "failed"


class TransferViolation(BaseModel):
    """Record of a transfer constraint violation."""
    constraint_id: str = Field(...)
    constraint_name: str = Field(...)
    transfer_id: str = Field(...)
    parameter: str = Field(...)
    actual_value: float = Field(...)
    limit_value: float = Field(...)
    limit_type: str = Field(...)
    action: TransferAction = Field(...)
    message: str = Field(...)
    reference_standard: Optional[str] = Field(None)


class TransferValidationResult(BaseModel):
    """Result of transfer constraint validation."""
    validation_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    transfer_id: str = Field(...)
    is_allowed: bool = Field(...)
    max_allowed_rate_m3h: Optional[float] = Field(None)
    violations: List[TransferViolation] = Field(default_factory=list)
    required_actions: List[TransferAction] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class InterlockStatus(BaseModel):
    """Status of a safety interlock."""
    interlock_id: str = Field(...)
    state: InterlockState = Field(...)
    last_checked: datetime = Field(...)
    is_healthy: bool = Field(...)
    bypass_reason: Optional[str] = Field(None)
    bypass_authorized_by: Optional[str] = Field(None)


class TransferRate(BaseModel):
    """Transfer rate configuration."""
    min_rate_m3h: float = Field(..., ge=0)
    max_rate_m3h: float = Field(..., ge=0)
    ramp_rate_m3h_min: float = Field(10.0, ge=0)
    surge_limit_m3h: Optional[float] = Field(None)


class RateLimitValidator:
    """Transfer rate limit validator."""

    def __init__(self, default_max_rate_m3h: float = 500.0, default_min_rate_m3h: float = 50.0):
        """Initialize with default rate limits."""
        self._default_max = default_max_rate_m3h
        self._default_min = default_min_rate_m3h
        self._line_limits: Dict[str, TransferRate] = {}
        logger.info(f"RateLimitValidator initialized: max={default_max_rate_m3h} m3/h")

    def register_line(self, line_id: str, limits: TransferRate) -> None:
        """Register rate limits for a transfer line."""
        self._line_limits[line_id] = limits
        logger.info(f"Registered rate limits for line {line_id}")

    def validate(
        self, current_rate_m3h: float, line_id: str, transfer_id: str
    ) -> TransferValidationResult:
        """Validate transfer rate against line limits."""
        validation_id = hashlib.sha256(
            f"rate|{current_rate_m3h}|{line_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        limits = self._line_limits.get(line_id)
        max_rate = limits.max_rate_m3h if limits else self._default_max
        min_rate = limits.min_rate_m3h if limits else self._default_min

        violations = []
        actions = []

        if current_rate_m3h > max_rate:
            violations.append(TransferViolation(
                constraint_id="RATE_MAX", constraint_name="Maximum Transfer Rate",
                transfer_id=transfer_id, parameter="rate_m3h",
                actual_value=current_rate_m3h, limit_value=max_rate,
                limit_type="max", action=TransferAction.REDUCE_RATE,
                message=f"Rate {current_rate_m3h} m3/h exceeds maximum {max_rate} m3/h",
                reference_standard="API 2610"
            ))
            actions.append(TransferAction.REDUCE_RATE)

        if limits and limits.surge_limit_m3h and current_rate_m3h > limits.surge_limit_m3h:
            violations.append(TransferViolation(
                constraint_id="RATE_SURGE", constraint_name="Surge Limit",
                transfer_id=transfer_id, parameter="rate_m3h",
                actual_value=current_rate_m3h, limit_value=limits.surge_limit_m3h,
                limit_type="max", action=TransferAction.EMERGENCY_STOP,
                message=f"Rate {current_rate_m3h} m3/h exceeds surge limit {limits.surge_limit_m3h} m3/h",
                reference_standard="API 2610 Surge Protection"
            ))
            actions.append(TransferAction.EMERGENCY_STOP)

        is_allowed = not any(a in [TransferAction.BLOCK, TransferAction.EMERGENCY_STOP] for a in actions)

        return TransferValidationResult(
            validation_id=validation_id, transfer_id=transfer_id,
            is_allowed=is_allowed, max_allowed_rate_m3h=max_rate,
            violations=violations, required_actions=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "rate": current_rate_m3h, "line": line_id}, sort_keys=True).encode()
            ).hexdigest()
        )


class MinStableFlowConstraint:
    """Minimum stable flow constraint to prevent pump cavitation."""

    def __init__(self, min_stable_pct: float = 30.0):
        """Initialize with minimum stable flow percentage."""
        self._min_stable_pct = min_stable_pct
        logger.info(f"MinStableFlowConstraint initialized: min_stable={min_stable_pct}%")

    def validate(
        self, current_rate_m3h: float, design_rate_m3h: float, transfer_id: str
    ) -> TransferValidationResult:
        """Validate flow is above minimum stable point."""
        validation_id = hashlib.sha256(
            f"minflow|{current_rate_m3h}|{design_rate_m3h}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        if design_rate_m3h > 0:
            current_pct = (current_rate_m3h / design_rate_m3h) * 100.0
            if current_pct > 0 and current_pct < self._min_stable_pct:
                violations.append(TransferViolation(
                    constraint_id="FLOW_MIN_STABLE", constraint_name="Minimum Stable Flow",
                    transfer_id=transfer_id, parameter="flow_pct",
                    actual_value=current_pct, limit_value=self._min_stable_pct,
                    limit_type="min", action=TransferAction.BLOCK,
                    message=f"Flow at {current_pct:.1f}% of design is below stable minimum {self._min_stable_pct}%",
                    reference_standard="API 610 Pump Curves"
                ))
                actions.append(TransferAction.BLOCK)

        is_allowed = TransferAction.BLOCK not in actions

        return TransferValidationResult(
            validation_id=validation_id, transfer_id=transfer_id,
            is_allowed=is_allowed, violations=violations, required_actions=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "rate": current_rate_m3h}, sort_keys=True).encode()
            ).hexdigest()
        )


class InterlockManager:
    """Safety interlock manager for valves and pumps."""

    def __init__(self):
        """Initialize interlock manager."""
        self._interlocks: Dict[str, InterlockStatus] = {}
        self._valve_states: Dict[str, ValveState] = {}
        self._pump_states: Dict[str, PumpState] = {}
        self._interlock_requirements: Dict[str, List[str]] = {}
        logger.info("InterlockManager initialized")

    def register_interlock(
        self, interlock_id: str, required_for: List[str], initial_state: InterlockState = InterlockState.ENGAGED
    ) -> None:
        """Register a safety interlock."""
        self._interlocks[interlock_id] = InterlockStatus(
            interlock_id=interlock_id, state=initial_state,
            last_checked=datetime.now(timezone.utc),
            is_healthy=initial_state == InterlockState.ENGAGED
        )
        for equipment_id in required_for:
            if equipment_id not in self._interlock_requirements:
                self._interlock_requirements[equipment_id] = []
            self._interlock_requirements[equipment_id].append(interlock_id)
        logger.info(f"Registered interlock {interlock_id} for {required_for}")

    def update_valve_state(self, valve_id: str, state: ValveState) -> None:
        """Update valve state."""
        self._valve_states[valve_id] = state
        logger.debug(f"Valve {valve_id} state updated to {state}")

    def update_pump_state(self, pump_id: str, state: PumpState) -> None:
        """Update pump state."""
        self._pump_states[pump_id] = state
        logger.debug(f"Pump {pump_id} state updated to {state}")

    def bypass_interlock(
        self, interlock_id: str, reason: str, authorized_by: str
    ) -> bool:
        """Bypass an interlock with authorization."""
        if interlock_id not in self._interlocks:
            logger.error(f"Interlock {interlock_id} not found")
            return False

        self._interlocks[interlock_id] = InterlockStatus(
            interlock_id=interlock_id, state=InterlockState.BYPASSED,
            last_checked=datetime.now(timezone.utc), is_healthy=False,
            bypass_reason=reason, bypass_authorized_by=authorized_by
        )
        logger.warning(f"[SAFETY] Interlock {interlock_id} BYPASSED by {authorized_by}: {reason}")
        return True

    def check_transfer_interlocks(
        self, source_tank: str, dest_tank: str, line_id: str, transfer_id: str
    ) -> TransferValidationResult:
        """Check all interlocks for a transfer operation."""
        validation_id = hashlib.sha256(
            f"interlock|{source_tank}|{dest_tank}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        equipment_ids = [source_tank, dest_tank, line_id]

        for equip_id in equipment_ids:
            required_interlocks = self._interlock_requirements.get(equip_id, [])
            for il_id in required_interlocks:
                status = self._interlocks.get(il_id)
                if status is None:
                    violations.append(TransferViolation(
                        constraint_id=f"IL_{il_id}", constraint_name=f"Interlock {il_id}",
                        transfer_id=transfer_id, parameter="interlock_state",
                        actual_value=0.0, limit_value=1.0, limit_type="min",
                        action=TransferAction.BLOCK,
                        message=f"Required interlock {il_id} not found",
                        reference_standard="IEC 61511"
                    ))
                    actions.append(TransferAction.BLOCK)
                elif status.state == InterlockState.FAILED:
                    violations.append(TransferViolation(
                        constraint_id=f"IL_{il_id}", constraint_name=f"Interlock {il_id}",
                        transfer_id=transfer_id, parameter="interlock_state",
                        actual_value=0.0, limit_value=1.0, limit_type="min",
                        action=TransferAction.EMERGENCY_STOP,
                        message=f"CRITICAL: Interlock {il_id} in FAILED state",
                        reference_standard="IEC 61511"
                    ))
                    actions.append(TransferAction.EMERGENCY_STOP)
                elif status.state == InterlockState.BYPASSED:
                    violations.append(TransferViolation(
                        constraint_id=f"IL_{il_id}", constraint_name=f"Interlock {il_id}",
                        transfer_id=transfer_id, parameter="interlock_state",
                        actual_value=0.5, limit_value=1.0, limit_type="min",
                        action=TransferAction.ALLOW,
                        message=f"WARNING: Interlock {il_id} bypassed by {status.bypass_authorized_by}",
                        reference_standard="IEC 61511"
                    ))

        is_allowed = not any(a in [TransferAction.BLOCK, TransferAction.EMERGENCY_STOP] for a in actions)

        return TransferValidationResult(
            validation_id=validation_id, transfer_id=transfer_id,
            is_allowed=is_allowed, violations=violations, required_actions=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({
                    "id": validation_id, "source": source_tank, "dest": dest_tank, "line": line_id
                }, sort_keys=True).encode()
            ).hexdigest()
        )

    def get_valve_state(self, valve_id: str) -> Optional[ValveState]:
        """Get current valve state."""
        return self._valve_states.get(valve_id)

    def get_pump_state(self, pump_id: str) -> Optional[PumpState]:
        """Get current pump state."""
        return self._pump_states.get(pump_id)


class TransferSequenceValidator:
    """Validates transfer operation sequences for safe execution."""

    def __init__(self, interlock_manager: InterlockManager):
        """Initialize with interlock manager."""
        self._interlock_manager = interlock_manager
        self._active_transfers: Dict[str, Dict[str, Any]] = {}
        logger.info("TransferSequenceValidator initialized")

    def validate_start_sequence(
        self,
        transfer_id: str,
        source_tank: str,
        dest_tank: str,
        line_id: str,
        source_valve: str,
        dest_valve: str,
        pump_id: str
    ) -> TransferValidationResult:
        """Validate transfer start sequence."""
        validation_id = hashlib.sha256(
            f"start|{transfer_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        source_valve_state = self._interlock_manager.get_valve_state(source_valve)
        dest_valve_state = self._interlock_manager.get_valve_state(dest_valve)
        pump_state = self._interlock_manager.get_pump_state(pump_id)

        if source_valve_state in [ValveState.FAILED_CLOSED, ValveState.FAILED_OPEN]:
            violations.append(TransferViolation(
                constraint_id="SEQ_SRC_VALVE", constraint_name="Source Valve Status",
                transfer_id=transfer_id, parameter="valve_state",
                actual_value=0.0, limit_value=1.0, limit_type="min",
                action=TransferAction.BLOCK,
                message=f"Source valve {source_valve} in failed state",
                reference_standard="API 2610"
            ))
            actions.append(TransferAction.BLOCK)

        if dest_valve_state in [ValveState.FAILED_CLOSED, ValveState.FAILED_OPEN]:
            violations.append(TransferViolation(
                constraint_id="SEQ_DST_VALVE", constraint_name="Destination Valve Status",
                transfer_id=transfer_id, parameter="valve_state",
                actual_value=0.0, limit_value=1.0, limit_type="min",
                action=TransferAction.BLOCK,
                message=f"Destination valve {dest_valve} in failed state",
                reference_standard="API 2610"
            ))
            actions.append(TransferAction.BLOCK)

        if pump_state == PumpState.FAILED:
            violations.append(TransferViolation(
                constraint_id="SEQ_PUMP", constraint_name="Pump Status",
                transfer_id=transfer_id, parameter="pump_state",
                actual_value=0.0, limit_value=1.0, limit_type="min",
                action=TransferAction.BLOCK,
                message=f"Pump {pump_id} in failed state",
                reference_standard="API 610"
            ))
            actions.append(TransferAction.BLOCK)

        interlock_result = self._interlock_manager.check_transfer_interlocks(
            source_tank, dest_tank, line_id, transfer_id
        )
        violations.extend(interlock_result.violations)
        actions.extend(interlock_result.required_actions)

        is_allowed = not any(a in [TransferAction.BLOCK, TransferAction.EMERGENCY_STOP] for a in actions)

        if is_allowed:
            self._active_transfers[transfer_id] = {
                "source": source_tank, "dest": dest_tank, "line": line_id,
                "started_at": datetime.now(timezone.utc)
            }

        return TransferValidationResult(
            validation_id=validation_id, transfer_id=transfer_id,
            is_allowed=is_allowed, violations=violations, required_actions=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({
                    "id": validation_id, "transfer": transfer_id, "allowed": is_allowed
                }, sort_keys=True).encode()
            ).hexdigest()
        )

    def validate_stop_sequence(self, transfer_id: str) -> TransferValidationResult:
        """Validate transfer stop sequence."""
        validation_id = hashlib.sha256(
            f"stop|{transfer_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        if transfer_id in self._active_transfers:
            del self._active_transfers[transfer_id]
            logger.info(f"Transfer {transfer_id} completed and removed from active list")

        return TransferValidationResult(
            validation_id=validation_id, transfer_id=transfer_id,
            is_allowed=True, violations=[], required_actions=[],
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "transfer": transfer_id, "action": "stop"}, sort_keys=True).encode()
            ).hexdigest()
        )


class TransferConstraintValidator:
    """Unified transfer constraint validator."""

    def __init__(self):
        """Initialize all transfer validators."""
        self.rate_validator = RateLimitValidator()
        self.min_flow = MinStableFlowConstraint()
        self.interlock_manager = InterlockManager()
        self.sequence_validator = TransferSequenceValidator(self.interlock_manager)
        logger.info("TransferConstraintValidator initialized")

    def validate_transfer(
        self,
        transfer_id: str,
        source_tank: str,
        dest_tank: str,
        line_id: str,
        proposed_rate_m3h: float,
        design_rate_m3h: float = 500.0
    ) -> TransferValidationResult:
        """
        Validate all transfer constraints.

        FAIL-CLOSED: Any BLOCK or EMERGENCY_STOP fails entire validation.
        """
        validation_id = hashlib.sha256(
            f"transfer|{transfer_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        all_violations: List[TransferViolation] = []
        all_actions: List[TransferAction] = []

        rate_result = self.rate_validator.validate(proposed_rate_m3h, line_id, transfer_id)
        all_violations.extend(rate_result.violations)
        all_actions.extend(rate_result.required_actions)

        if proposed_rate_m3h > 0:
            flow_result = self.min_flow.validate(proposed_rate_m3h, design_rate_m3h, transfer_id)
            all_violations.extend(flow_result.violations)
            all_actions.extend(flow_result.required_actions)

        interlock_result = self.interlock_manager.check_transfer_interlocks(
            source_tank, dest_tank, line_id, transfer_id
        )
        all_violations.extend(interlock_result.violations)
        all_actions.extend(interlock_result.required_actions)

        is_allowed = not any(a in [TransferAction.BLOCK, TransferAction.EMERGENCY_STOP] for a in all_actions)

        max_allowed_rate = rate_result.max_allowed_rate_m3h

        return TransferValidationResult(
            validation_id=validation_id, transfer_id=transfer_id,
            is_allowed=is_allowed, max_allowed_rate_m3h=max_allowed_rate,
            violations=all_violations, required_actions=list(set(all_actions)),
            provenance_hash=hashlib.sha256(
                json.dumps({
                    "id": validation_id, "transfer": transfer_id,
                    "violations": len(all_violations), "allowed": is_allowed
                }, sort_keys=True).encode()
            ).hexdigest()
        )
