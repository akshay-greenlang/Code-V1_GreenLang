# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Operating Modes Module

Multi-mode operation management: Recommend, Supervised, Autonomous, Fallback.
Includes approval workflows and safety gate integration.

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import hashlib
import logging
import threading

logger = logging.getLogger(__name__)


class OperatingMode(str, Enum):
    """System operating mode enumeration."""
    RECOMMEND_ONLY = "recommend_only"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"
    FALLBACK = "fallback"
    EMERGENCY_STOP = "emergency_stop"


class SafetyGateStatus(str, Enum):
    """Safety gate status."""
    OPEN = "open"
    CLOSED = "closed"
    BYPASSED = "bypassed"
    FAULT = "fault"


class ApprovalWorkflow(BaseModel):
    """Mode transition approval workflow."""
    workflow_id: str = Field(...)
    from_mode: OperatingMode
    to_mode: OperatingMode
    requester: str = Field(...)
    approver: Optional[str] = Field(default=None)
    request_time: datetime = Field(default_factory=datetime.now)
    approval_time: Optional[datetime] = Field(default=None)
    status: str = Field(default="pending")
    reason: str = Field(default="")


class ModeConfiguration(BaseModel):
    """Configuration for each operating mode."""
    mode: OperatingMode
    allowed_actions: List[str] = Field(default_factory=list)
    requires_operator_approval: bool = Field(default=True)
    max_autonomy_duration_hours: float = Field(default=24.0)
    safety_gates_required: List[str] = Field(default_factory=list)
    fallback_on_comm_loss: bool = Field(default=True)


class ModeTransitionRequest(BaseModel):
    """Request for mode transition."""
    target_mode: OperatingMode
    requester: str = Field(...)
    reason: str = Field(default="")
    override_safety: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.now)


class ModeTransitionResult(BaseModel):
    """Result of mode transition attempt."""
    success: bool
    previous_mode: OperatingMode
    current_mode: OperatingMode
    message: str = Field(default="")
    approval_required: bool = Field(default=False)
    workflow_id: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)
    provenance_hash: str = Field(...)


class OperatingModeManager:
    """Operating mode manager with approval workflows."""

    def __init__(self, config: List[ModeConfiguration] = None, alert_callback: Optional[Callable] = None):
        self._lock = threading.RLock()
        self._current_mode = OperatingMode.RECOMMEND_ONLY
        self._mode_configs = {c.mode: c for c in (config or [])}
        self._safety_gates: Dict[str, SafetyGateStatus] = {}
        self._pending_workflows: Dict[str, ApprovalWorkflow] = {}
        self._alert_callback = alert_callback
        self._mode_start_time = datetime.now()
        logger.info(f"OperatingModeManager initialized: mode={self._current_mode.value}")

    def get_current_mode(self) -> OperatingMode:
        return self._current_mode

    def request_mode_transition(self, request: ModeTransitionRequest) -> ModeTransitionResult:
        """Request a mode transition."""
        with self._lock:
            prev_mode = self._current_mode
            target = request.target_mode

            # Check if transition is allowed
            if not self._is_transition_allowed(prev_mode, target):
                return ModeTransitionResult(
                    success=False,
                    previous_mode=prev_mode,
                    current_mode=prev_mode,
                    message=f"Transition from {prev_mode.value} to {target.value} not allowed",
                    provenance_hash=self._calc_provenance(prev_mode, target, False)
                )

            # Check safety gates
            config = self._mode_configs.get(target)
            if config and not request.override_safety:
                for gate_id in config.safety_gates_required:
                    status = self._safety_gates.get(gate_id, SafetyGateStatus.CLOSED)
                    if status != SafetyGateStatus.OPEN:
                        return ModeTransitionResult(
                            success=False,
                            previous_mode=prev_mode,
                            current_mode=prev_mode,
                            message=f"Safety gate {gate_id} not open",
                            provenance_hash=self._calc_provenance(prev_mode, target, False)
                        )

            # Check if approval required
            if config and config.requires_operator_approval and target in [OperatingMode.AUTONOMOUS]:
                workflow = ApprovalWorkflow(
                    workflow_id=hashlib.sha256(f"{target}:{datetime.now()}".encode()).hexdigest()[:16],
                    from_mode=prev_mode,
                    to_mode=target,
                    requester=request.requester,
                    reason=request.reason
                )
                self._pending_workflows[workflow.workflow_id] = workflow
                return ModeTransitionResult(
                    success=False,
                    previous_mode=prev_mode,
                    current_mode=prev_mode,
                    message="Approval required",
                    approval_required=True,
                    workflow_id=workflow.workflow_id,
                    provenance_hash=self._calc_provenance(prev_mode, target, False)
                )

            # Execute transition
            self._current_mode = target
            self._mode_start_time = datetime.now()
            logger.info(f"Mode transition: {prev_mode.value} -> {target.value}")

            return ModeTransitionResult(
                success=True,
                previous_mode=prev_mode,
                current_mode=target,
                message="Transition successful",
                provenance_hash=self._calc_provenance(prev_mode, target, True)
            )

    def approve_transition(self, workflow_id: str, approver: str) -> ModeTransitionResult:
        """Approve a pending mode transition."""
        with self._lock:
            workflow = self._pending_workflows.get(workflow_id)
            if not workflow:
                return ModeTransitionResult(
                    success=False,
                    previous_mode=self._current_mode,
                    current_mode=self._current_mode,
                    message="Workflow not found",
                    provenance_hash=self._calc_provenance(self._current_mode, self._current_mode, False)
                )

            prev_mode = self._current_mode
            workflow.approver = approver
            workflow.approval_time = datetime.now()
            workflow.status = "approved"

            self._current_mode = workflow.to_mode
            self._mode_start_time = datetime.now()
            del self._pending_workflows[workflow_id]

            logger.info(f"Mode transition approved: {prev_mode.value} -> {workflow.to_mode.value}")

            return ModeTransitionResult(
                success=True,
                previous_mode=prev_mode,
                current_mode=workflow.to_mode,
                message="Transition approved and executed",
                provenance_hash=self._calc_provenance(prev_mode, workflow.to_mode, True)
            )

    def set_safety_gate(self, gate_id: str, status: SafetyGateStatus) -> None:
        """Set safety gate status."""
        with self._lock:
            self._safety_gates[gate_id] = status
            logger.info(f"Safety gate {gate_id} set to {status.value}")

    def get_safety_gate(self, gate_id: str) -> SafetyGateStatus:
        """Get safety gate status."""
        return self._safety_gates.get(gate_id, SafetyGateStatus.CLOSED)

    def force_fallback(self, reason: str) -> ModeTransitionResult:
        """Force transition to fallback mode."""
        with self._lock:
            prev_mode = self._current_mode
            self._current_mode = OperatingMode.FALLBACK
            self._mode_start_time = datetime.now()
            logger.warning(f"Forced fallback from {prev_mode.value}: {reason}")

            return ModeTransitionResult(
                success=True,
                previous_mode=prev_mode,
                current_mode=OperatingMode.FALLBACK,
                message=f"Forced fallback: {reason}",
                provenance_hash=self._calc_provenance(prev_mode, OperatingMode.FALLBACK, True)
            )

    def _is_transition_allowed(self, from_mode: OperatingMode, to_mode: OperatingMode) -> bool:
        """Check if mode transition is allowed."""
        # Define allowed transitions
        allowed = {
            OperatingMode.RECOMMEND_ONLY: [OperatingMode.SUPERVISED, OperatingMode.FALLBACK, OperatingMode.EMERGENCY_STOP],
            OperatingMode.SUPERVISED: [OperatingMode.RECOMMEND_ONLY, OperatingMode.AUTONOMOUS, OperatingMode.FALLBACK, OperatingMode.EMERGENCY_STOP],
            OperatingMode.AUTONOMOUS: [OperatingMode.SUPERVISED, OperatingMode.RECOMMEND_ONLY, OperatingMode.FALLBACK, OperatingMode.EMERGENCY_STOP],
            OperatingMode.FALLBACK: [OperatingMode.RECOMMEND_ONLY, OperatingMode.SUPERVISED],
            OperatingMode.EMERGENCY_STOP: [OperatingMode.RECOMMEND_ONLY],
        }
        return to_mode in allowed.get(from_mode, [])

    def _calc_provenance(self, from_mode: OperatingMode, to_mode: OperatingMode, success: bool) -> str:
        data = {"from": from_mode.value, "to": to_mode.value, "success": success, "ts": datetime.now().isoformat()}
        return hashlib.sha256(str(data).encode()).hexdigest()
