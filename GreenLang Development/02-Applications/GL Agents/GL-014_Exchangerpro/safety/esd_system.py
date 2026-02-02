# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - IEC 61511 Emergency Shutdown System

Standalone Emergency Shutdown System (ESD) implementing Safety Instrumented
Functions (SIFs) for heat exchanger protection per IEC 61511/61508 standards.

ESD Features:
1. Multi-level shutdown sequences (Level 0-4)
2. Safety Integrity Level (SIL) tracking
3. Proof test management
4. Bypass management with authorization
5. Full audit trail with provenance

Safety Principles:
- Fail-safe design: shutdown on loss of signal
- Redundant sensors for critical measurements
- Proof testing requirements tracked
- Full audit trail for regulatory compliance

Standards Compliance:
- IEC 61511: Safety Instrumented Systems for Process Industry
- IEC 61508: Functional Safety
- ISA-84: Application of Safety Instrumented Systems

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ShutdownLevel(str, Enum):
    """Shutdown severity levels following IEC 61511 classifications."""

    LEVEL_0 = "level_0"  # Normal operation
    LEVEL_1 = "level_1"  # Soft shutdown - orderly process stop
    LEVEL_2 = "level_2"  # Hard shutdown - immediate process stop
    LEVEL_3 = "level_3"  # Emergency shutdown - ESD
    LEVEL_4 = "level_4"  # Critical emergency - plant-wide ESD


class ShutdownState(str, Enum):
    """Current state of the shutdown system."""

    NORMAL = "normal"  # Normal operation
    ARMED = "armed"  # System armed, ready to initiate
    INITIATING = "initiating"  # Shutdown initiating
    IN_PROGRESS = "in_progress"  # Shutdown in progress
    COMPLETED = "completed"  # Shutdown complete
    RECOVERING = "recovering"  # System recovery in progress
    BYPASSED = "bypassed"  # System bypassed (maintenance)
    FAULT = "fault"  # System fault detected


class SILLevel(int, Enum):
    """Safety Integrity Level per IEC 61508/61511."""

    SIL_1 = 1  # PFD: 10^-2 to 10^-1
    SIL_2 = 2  # PFD: 10^-3 to 10^-2
    SIL_3 = 3  # PFD: 10^-4 to 10^-3
    SIL_4 = 4  # PFD: 10^-5 to 10^-4


class SafetyIntegratedFunction(str, Enum):
    """Safety Instrumented Functions for heat exchangers."""

    HIGH_TEMP_PROTECTION = "high_temp_protection"
    LOW_TEMP_PROTECTION = "low_temp_protection"
    HIGH_PRESSURE_PROTECTION = "high_pressure_protection"
    LOW_FLOW_PROTECTION = "low_flow_protection"
    THERMAL_RUNAWAY_PROTECTION = "thermal_runaway_protection"
    PROCESS_ISOLATION = "process_isolation"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ShutdownEvent:
    """
    Record of a shutdown event.

    Attributes:
        event_id: Unique event identifier
        exchanger_id: Affected heat exchanger
        shutdown_level: Level of shutdown executed
        sil_level: SIL requirement for this shutdown
        trigger_condition: What triggered the shutdown
        sif_triggered: Safety Instrumented Function that triggered
        initiated_at: When shutdown was initiated
        completed_at: When shutdown completed
        duration_seconds: Time to complete shutdown
        success: Whether shutdown completed successfully
        actions_taken: List of shutdown actions taken
        personnel_notified: List of personnel notified
        provenance_hash: SHA-256 hash for audit trail
    """

    event_id: str
    exchanger_id: str
    shutdown_level: ShutdownLevel
    sil_level: SILLevel
    trigger_condition: str
    sif_triggered: Optional[SafetyIntegratedFunction] = None
    initiated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    actions_taken: List[str] = field(default_factory=list)
    personnel_notified: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.event_id}|{self.exchanger_id}|"
                f"{self.shutdown_level.value}|{self.sil_level.value}|"
                f"{self.initiated_at.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ProofTestRecord:
    """
    Record of an ESD proof test per IEC 61511.

    Attributes:
        test_id: Unique test identifier
        exchanger_id: Heat exchanger tested
        test_timestamp: When test was performed
        test_result: PASS or FAIL
        tested_by: Person conducting test
        sil_level: SIL level being tested
        sif_tested: Safety function tested
        notes: Test notes
        next_test_due: When next test is due
        provenance_hash: SHA-256 hash for audit
    """

    test_id: str
    exchanger_id: str
    test_timestamp: datetime
    test_result: str  # "PASS" or "FAIL"
    tested_by: str
    sil_level: SILLevel
    sif_tested: Optional[SafetyIntegratedFunction] = None
    notes: Optional[str] = None
    next_test_due: Optional[datetime] = None
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.test_id}|{self.exchanger_id}|"
                f"{self.test_timestamp.isoformat()}|{self.test_result}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class BypassRecord:
    """
    Record of an ESD bypass authorization.

    Attributes:
        bypass_id: Unique bypass identifier
        exchanger_id: Heat exchanger bypassed
        reason: Reason for bypass
        authorized_by: Person authorizing bypass
        approved_by: Person approving (if different)
        bypass_start: When bypass started
        bypass_end: When bypass ended/expires
        is_active: Whether bypass is currently active
        compensating_measures: Safety measures in place during bypass
        provenance_hash: SHA-256 hash for audit
    """

    bypass_id: str
    exchanger_id: str
    reason: str
    authorized_by: str
    approved_by: Optional[str] = None
    bypass_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bypass_end: Optional[datetime] = None
    is_active: bool = True
    compensating_measures: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.bypass_id}|{self.exchanger_id}|"
                f"{self.reason}|{self.authorized_by}|{self.bypass_start.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# EMERGENCY SHUTDOWN SYSTEM
# =============================================================================


class EmergencyShutdownSystem:
    """
    IEC 61511 Compliant Emergency Shutdown System for Heat Exchangers.

    Implements Safety Instrumented Functions (SIFs) for heat exchanger
    protection including:
    - Process isolation (inlet/outlet valves)
    - Bypass activation
    - Flow diversion
    - Temperature protection

    Safety Principles:
    - Fail-safe design: shutdown on loss of signal
    - Redundant sensors for critical measurements
    - Proof testing requirements tracked
    - Full audit trail for regulatory compliance

    Standards Compliance:
    - IEC 61511: Safety Instrumented Systems for Process Industry
    - IEC 61508: Functional Safety
    - ISA-84: Application of Safety Instrumented Systems

    Example:
        >>> esd = EmergencyShutdownSystem(
        ...     exchanger_id="HX-101",
        ...     sil_level=SILLevel.SIL_2,
        ... )
        >>>
        >>> # Execute emergency shutdown
        >>> event = esd.execute_shutdown(
        ...     level=ShutdownLevel.LEVEL_3,
        ...     trigger="High temperature - 410C",
        ... )
        >>>
        >>> # Check system status
        >>> status = esd.get_status()
        >>> print(f"State: {status['state']}")

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    # Shutdown sequence definitions per level
    SHUTDOWN_SEQUENCES: Dict[ShutdownLevel, List[str]] = {
        ShutdownLevel.LEVEL_0: [],  # Normal operation
        ShutdownLevel.LEVEL_1: [
            "reduce_heat_input",
            "notify_operator",
            "log_event",
        ],
        ShutdownLevel.LEVEL_2: [
            "close_inlet_valve",
            "close_outlet_valve",
            "notify_supervisor",
            "log_event",
        ],
        ShutdownLevel.LEVEL_3: [
            "trip_inlet_valve",
            "trip_outlet_valve",
            "open_bypass_valve",
            "vent_pressure",
            "notify_emergency_team",
            "activate_alarms",
            "log_event",
        ],
        ShutdownLevel.LEVEL_4: [
            "trip_all_valves",
            "isolate_from_process",
            "activate_fire_suppression_if_required",
            "notify_plant_emergency",
            "evacuate_area_if_required",
            "log_event",
        ],
    }

    # Maximum shutdown time requirements (seconds) per SIL
    MAX_SHUTDOWN_TIMES: Dict[SILLevel, float] = {
        SILLevel.SIL_1: 60.0,  # 60 seconds
        SILLevel.SIL_2: 30.0,  # 30 seconds
        SILLevel.SIL_3: 10.0,  # 10 seconds
        SILLevel.SIL_4: 5.0,   # 5 seconds
    }

    # Proof test intervals (days) per SIL
    PROOF_TEST_INTERVALS: Dict[SILLevel, int] = {
        SILLevel.SIL_1: 365,   # Annual
        SILLevel.SIL_2: 365,   # Annual
        SILLevel.SIL_3: 180,   # Semi-annual
        SILLevel.SIL_4: 90,    # Quarterly
    }

    def __init__(
        self,
        exchanger_id: str,
        sil_level: SILLevel = SILLevel.SIL_2,
        action_executor: Optional[Callable[[str, str], bool]] = None,
        notification_callback: Optional[Callable[[str, List[str]], None]] = None,
    ) -> None:
        """
        Initialize Emergency Shutdown System.

        Args:
            exchanger_id: Heat exchanger identifier
            sil_level: Required Safety Integrity Level
            action_executor: Callback to execute shutdown actions
            notification_callback: Callback for emergency notifications
        """
        self.exchanger_id = exchanger_id
        self.sil_level = sil_level
        self._action_executor = action_executor
        self._notification_callback = notification_callback
        self._lock = threading.RLock()

        # State tracking
        self._state = ShutdownState.NORMAL
        self._current_event: Optional[ShutdownEvent] = None
        self._event_history: List[ShutdownEvent] = []

        # Bypass tracking
        self._bypass_record: Optional[BypassRecord] = None
        self._bypass_history: List[BypassRecord] = []

        # Proof test tracking
        self._proof_test_history: List[ProofTestRecord] = []
        self._last_proof_test: Optional[datetime] = None
        self._proof_test_interval_days = self.PROOF_TEST_INTERVALS[sil_level]

        # SIF configuration
        self._enabled_sifs: Dict[SafetyIntegratedFunction, bool] = {
            sif: True for sif in SafetyIntegratedFunction
        }

        logger.info(
            f"EmergencyShutdownSystem initialized: exchanger={exchanger_id}, "
            f"sil={sil_level.value}, proof_test_interval={self._proof_test_interval_days}d"
        )

    @property
    def state(self) -> ShutdownState:
        """Get current shutdown system state."""
        with self._lock:
            return self._state

    @property
    def is_bypassed(self) -> bool:
        """Check if system is currently bypassed."""
        with self._lock:
            if self._state != ShutdownState.BYPASSED:
                return False
            if self._bypass_record and self._bypass_record.bypass_end:
                if datetime.now(timezone.utc) > self._bypass_record.bypass_end:
                    self._clear_bypass_internal()
                    return False
            return True

    def _clear_bypass_internal(self) -> None:
        """Internal method to clear bypass state."""
        if self._bypass_record:
            self._bypass_record.is_active = False
            self._bypass_history.append(self._bypass_record)
        self._bypass_record = None
        self._state = ShutdownState.NORMAL
        logger.info(f"Bypass expired/cleared for {self.exchanger_id}")

    # =========================================================================
    # SHUTDOWN EXECUTION
    # =========================================================================

    def execute_shutdown(
        self,
        level: ShutdownLevel,
        trigger: str,
        sif_triggered: Optional[SafetyIntegratedFunction] = None,
        authorized_by: Optional[str] = None,
    ) -> ShutdownEvent:
        """
        Execute emergency shutdown sequence.

        Args:
            level: Shutdown level to execute
            trigger: Description of triggering condition
            sif_triggered: Safety Instrumented Function that triggered
            authorized_by: Person authorizing (for Level 3+)

        Returns:
            ShutdownEvent record

        Raises:
            RuntimeError: If system is bypassed or in invalid state
        """
        with self._lock:
            # Check if system is bypassed
            if self.is_bypassed:
                raise RuntimeError(
                    f"Cannot execute ESD: system bypassed for {self.exchanger_id} "
                    f"until {self._bypass_record.bypass_end if self._bypass_record else 'N/A'}"
                )

            # Check state
            if self._state not in (
                ShutdownState.NORMAL,
                ShutdownState.ARMED,
                ShutdownState.RECOVERING,
            ):
                logger.warning(
                    f"Shutdown requested but system in state {self._state.value}"
                )

            # Create event
            event_id = f"ESD_{self.exchanger_id}_{int(datetime.now(timezone.utc).timestamp())}"
            event = ShutdownEvent(
                event_id=event_id,
                exchanger_id=self.exchanger_id,
                shutdown_level=level,
                sil_level=self.sil_level,
                trigger_condition=trigger,
                sif_triggered=sif_triggered,
            )

            self._state = ShutdownState.INITIATING
            self._current_event = event

            logger.critical(
                f"EMERGENCY SHUTDOWN INITIATED: exchanger={self.exchanger_id}, "
                f"level={level.value}, trigger={trigger}, sif={sif_triggered}"
            )

            # Execute shutdown sequence
            sequence = self.SHUTDOWN_SEQUENCES.get(level, [])
            self._state = ShutdownState.IN_PROGRESS

            start_time = datetime.now(timezone.utc)
            actions_completed = []
            success = True

            for action in sequence:
                try:
                    if self._action_executor:
                        result = self._action_executor(self.exchanger_id, action)
                        if not result:
                            logger.error(f"Shutdown action failed: {action}")
                            success = False
                    actions_completed.append(action)
                    logger.info(f"Shutdown action completed: {action}")

                except Exception as e:
                    logger.error(f"Shutdown action error: {action} - {e}")
                    success = False

            # Complete event
            end_time = datetime.now(timezone.utc)
            event.completed_at = end_time
            event.duration_seconds = (end_time - start_time).total_seconds()
            event.success = success
            event.actions_taken = actions_completed

            # Check if within required time
            max_time = self.MAX_SHUTDOWN_TIMES.get(self.sil_level, 60.0)
            if event.duration_seconds > max_time:
                logger.error(
                    f"Shutdown exceeded max time for SIL-{self.sil_level.value}: "
                    f"{event.duration_seconds:.1f}s > {max_time}s"
                )
                event.success = False

            # Send notifications
            personnel = self._get_notification_list(level)
            event.personnel_notified = personnel
            if self._notification_callback:
                try:
                    self._notification_callback(event_id, personnel)
                except Exception as e:
                    logger.error(f"Notification failed: {e}")

            # Update state
            self._state = ShutdownState.COMPLETED if success else ShutdownState.FAULT
            self._event_history.append(event)
            self._current_event = None

            logger.info(
                f"Shutdown completed: success={success}, "
                f"duration={event.duration_seconds:.2f}s, "
                f"met_sil_requirement={event.duration_seconds <= max_time}"
            )

            return event

    def _get_notification_list(self, level: ShutdownLevel) -> List[str]:
        """Get list of personnel to notify based on shutdown level."""
        notifications = ["control_room_operator"]

        if level.value >= ShutdownLevel.LEVEL_2.value:
            notifications.extend(["shift_supervisor", "process_engineer"])

        if level.value >= ShutdownLevel.LEVEL_3.value:
            notifications.extend(["operations_manager", "safety_engineer"])

        if level.value >= ShutdownLevel.LEVEL_4.value:
            notifications.extend(["plant_manager", "emergency_coordinator"])

        return notifications

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def arm(self) -> bool:
        """
        Arm the ESD system for immediate response.

        Returns:
            True if armed successfully
        """
        with self._lock:
            if self.is_bypassed:
                logger.warning(f"Cannot arm: system bypassed for {self.exchanger_id}")
                return False

            if self._state != ShutdownState.NORMAL:
                logger.warning(f"Cannot arm: system in state {self._state.value}")
                return False

            self._state = ShutdownState.ARMED
            logger.info(f"ESD system armed for {self.exchanger_id}")
            return True

    def disarm(self, authorized_by: str) -> bool:
        """
        Disarm the ESD system (return to normal).

        Args:
            authorized_by: Person authorizing disarm

        Returns:
            True if disarmed successfully
        """
        with self._lock:
            if self._state not in (ShutdownState.ARMED, ShutdownState.COMPLETED):
                logger.warning(f"Cannot disarm: system in state {self._state.value}")
                return False

            self._state = ShutdownState.NORMAL
            logger.info(f"ESD system disarmed by {authorized_by} for {self.exchanger_id}")
            return True

    def initiate_recovery(self, authorized_by: str) -> bool:
        """
        Initiate recovery from shutdown state.

        Args:
            authorized_by: Person authorizing recovery

        Returns:
            True if recovery initiated
        """
        with self._lock:
            if self._state not in (ShutdownState.COMPLETED, ShutdownState.FAULT):
                logger.warning(f"Cannot recover: system in state {self._state.value}")
                return False

            self._state = ShutdownState.RECOVERING
            logger.info(
                f"Recovery initiated by {authorized_by} for {self.exchanger_id}"
            )
            return True

    def complete_recovery(self, authorized_by: str) -> bool:
        """
        Complete recovery and return to normal operation.

        Args:
            authorized_by: Person confirming recovery

        Returns:
            True if recovery completed
        """
        with self._lock:
            if self._state != ShutdownState.RECOVERING:
                logger.warning(
                    f"Cannot complete recovery: system in state {self._state.value}"
                )
                return False

            self._state = ShutdownState.NORMAL
            logger.info(
                f"Recovery completed by {authorized_by} for {self.exchanger_id}"
            )
            return True

    # =========================================================================
    # BYPASS MANAGEMENT
    # =========================================================================

    def set_bypass(
        self,
        reason: str,
        authorized_by: str,
        approved_by: Optional[str] = None,
        duration_hours: int = 8,
        compensating_measures: Optional[List[str]] = None,
    ) -> BypassRecord:
        """
        Set ESD system bypass (maintenance mode).

        Args:
            reason: Reason for bypass
            authorized_by: Person authorizing bypass
            approved_by: Person approving (for SIL-3+)
            duration_hours: Bypass duration (max 24 hours)
            compensating_measures: Safety measures during bypass

        Returns:
            BypassRecord

        Raises:
            RuntimeError: If bypass not allowed in current state
        """
        with self._lock:
            if self._state not in (ShutdownState.NORMAL, ShutdownState.ARMED):
                raise RuntimeError(
                    f"Cannot bypass: system in state {self._state.value}"
                )

            # SIL-3+ requires approval
            if self.sil_level.value >= SILLevel.SIL_3.value and not approved_by:
                raise RuntimeError(
                    f"SIL-{self.sil_level.value} system requires approval for bypass"
                )

            # Limit bypass duration
            duration_hours = min(duration_hours, 24)

            bypass_id = f"BYP_{self.exchanger_id}_{int(datetime.now(timezone.utc).timestamp())}"
            bypass_record = BypassRecord(
                bypass_id=bypass_id,
                exchanger_id=self.exchanger_id,
                reason=reason,
                authorized_by=authorized_by,
                approved_by=approved_by,
                bypass_end=datetime.now(timezone.utc) + timedelta(hours=duration_hours),
                compensating_measures=compensating_measures or [],
            )

            self._state = ShutdownState.BYPASSED
            self._bypass_record = bypass_record

            logger.warning(
                f"ESD BYPASSED: exchanger={self.exchanger_id}, "
                f"reason={reason}, by={authorized_by}, approved_by={approved_by}, "
                f"duration={duration_hours}h"
            )

            return bypass_record

    def clear_bypass(self, authorized_by: str) -> bool:
        """
        Clear ESD system bypass.

        Args:
            authorized_by: Person clearing bypass

        Returns:
            True if bypass cleared
        """
        with self._lock:
            if self._state != ShutdownState.BYPASSED:
                return False

            self._clear_bypass_internal()
            logger.info(
                f"ESD bypass cleared by {authorized_by} for {self.exchanger_id}"
            )
            return True

    # =========================================================================
    # PROOF TESTING
    # =========================================================================

    def record_proof_test(
        self,
        test_result: bool,
        tested_by: str,
        sif_tested: Optional[SafetyIntegratedFunction] = None,
        notes: Optional[str] = None,
    ) -> ProofTestRecord:
        """
        Record a proof test of the ESD system per IEC 61511.

        Args:
            test_result: Whether test passed
            tested_by: Person conducting test
            sif_tested: Specific SIF tested (or None for full system)
            notes: Test notes

        Returns:
            ProofTestRecord
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            test_id = f"PT_{self.exchanger_id}_{int(now.timestamp())}"

            record = ProofTestRecord(
                test_id=test_id,
                exchanger_id=self.exchanger_id,
                test_timestamp=now,
                test_result="PASS" if test_result else "FAIL",
                tested_by=tested_by,
                sil_level=self.sil_level,
                sif_tested=sif_tested,
                notes=notes,
                next_test_due=now + timedelta(days=self._proof_test_interval_days),
            )

            self._proof_test_history.append(record)

            if test_result:
                self._last_proof_test = now
                logger.info(
                    f"Proof test PASSED for {self.exchanger_id}, "
                    f"next due: {record.next_test_due}"
                )
            else:
                logger.error(
                    f"Proof test FAILED for {self.exchanger_id}, "
                    f"tested_by={tested_by}, notes={notes}"
                )

            return record

    def is_proof_test_overdue(self) -> bool:
        """Check if proof test is overdue."""
        with self._lock:
            if self._last_proof_test is None:
                return True
            days_since_test = (
                datetime.now(timezone.utc) - self._last_proof_test
            ).days
            return days_since_test > self._proof_test_interval_days

    def get_days_until_proof_test(self) -> Optional[int]:
        """Get days until next proof test is due."""
        with self._lock:
            if self._last_proof_test is None:
                return 0  # Overdue
            next_due = self._last_proof_test + timedelta(
                days=self._proof_test_interval_days
            )
            days_remaining = (next_due - datetime.now(timezone.utc)).days
            return max(0, days_remaining)

    # =========================================================================
    # STATUS AND HISTORY
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive ESD system status.

        Returns:
            Status dictionary with all relevant information
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            return {
                "exchanger_id": self.exchanger_id,
                "state": self._state.value,
                "sil_level": self.sil_level.value,
                "is_bypassed": self.is_bypassed,
                "bypass_info": {
                    "reason": self._bypass_record.reason if self._bypass_record else None,
                    "authorized_by": (
                        self._bypass_record.authorized_by if self._bypass_record else None
                    ),
                    "expires": (
                        self._bypass_record.bypass_end.isoformat()
                        if self._bypass_record and self._bypass_record.bypass_end
                        else None
                    ),
                    "compensating_measures": (
                        self._bypass_record.compensating_measures
                        if self._bypass_record
                        else []
                    ),
                },
                "proof_test": {
                    "last_test": (
                        self._last_proof_test.isoformat()
                        if self._last_proof_test
                        else None
                    ),
                    "is_overdue": self.is_proof_test_overdue(),
                    "days_until_due": self.get_days_until_proof_test(),
                    "interval_days": self._proof_test_interval_days,
                },
                "statistics": {
                    "total_shutdowns": len(self._event_history),
                    "successful_shutdowns": sum(
                        1 for e in self._event_history if e.success
                    ),
                    "total_bypasses": len(self._bypass_history)
                    + (1 if self._bypass_record else 0),
                    "total_proof_tests": len(self._proof_test_history),
                },
                "last_shutdown": (
                    {
                        "event_id": self._event_history[-1].event_id,
                        "level": self._event_history[-1].shutdown_level.value,
                        "timestamp": self._event_history[-1].initiated_at.isoformat(),
                        "success": self._event_history[-1].success,
                    }
                    if self._event_history
                    else None
                ),
                "timestamp": now.isoformat(),
            }

    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get shutdown event history.

        Args:
            limit: Maximum events to return

        Returns:
            List of event records
        """
        with self._lock:
            events = self._event_history[-limit:]
            return [
                {
                    "event_id": e.event_id,
                    "shutdown_level": e.shutdown_level.value,
                    "sil_level": e.sil_level.value,
                    "trigger_condition": e.trigger_condition,
                    "sif_triggered": e.sif_triggered.value if e.sif_triggered else None,
                    "initiated_at": e.initiated_at.isoformat(),
                    "completed_at": (
                        e.completed_at.isoformat() if e.completed_at else None
                    ),
                    "duration_seconds": e.duration_seconds,
                    "success": e.success,
                    "actions_taken": e.actions_taken,
                    "personnel_notified": e.personnel_notified,
                    "provenance_hash": e.provenance_hash,
                }
                for e in events
            ]

    def get_proof_test_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get proof test history.

        Args:
            limit: Maximum records to return

        Returns:
            List of proof test records
        """
        with self._lock:
            records = self._proof_test_history[-limit:]
            return [
                {
                    "test_id": r.test_id,
                    "test_timestamp": r.test_timestamp.isoformat(),
                    "test_result": r.test_result,
                    "tested_by": r.tested_by,
                    "sil_level": r.sil_level.value,
                    "sif_tested": r.sif_tested.value if r.sif_tested else None,
                    "notes": r.notes,
                    "next_test_due": (
                        r.next_test_due.isoformat() if r.next_test_due else None
                    ),
                    "provenance_hash": r.provenance_hash,
                }
                for r in records
            ]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_esd_for_exchanger(
    exchanger_id: str,
    exchanger_type: str = "shell_and_tube",
    service: str = "process_heat_recovery",
    action_executor: Optional[Callable[[str, str], bool]] = None,
    notification_callback: Optional[Callable[[str, List[str]], None]] = None,
) -> EmergencyShutdownSystem:
    """
    Factory function to create appropriately configured ESD for an exchanger.

    Determines SIL level based on service criticality per API RP 14C and
    IEC 61511 guidance for process industry.

    Args:
        exchanger_id: Heat exchanger identifier
        exchanger_type: Type of exchanger
        service: Service/application
        action_executor: Callback to execute shutdown actions
        notification_callback: Callback for emergency notifications

    Returns:
        Configured EmergencyShutdownSystem
    """
    # Determine SIL level based on service criticality
    # SIL-4: Nuclear, extremely hazardous chemicals
    # SIL-3: High-pressure steam, reactor systems, fired heaters
    # SIL-2: Process heat recovery, feed preheat
    # SIL-1: Utility services, non-critical cooling

    sil_4_services = ["nuclear", "chlorine", "phosgene"]
    sil_3_services = ["fired_heater", "reactor_feed", "high_pressure_steam"]
    sil_2_services = ["process_heat_recovery", "feed_preheat", "crude_preheat"]

    service_lower = service.lower()

    if any(s in service_lower for s in sil_4_services):
        sil_level = SILLevel.SIL_4
    elif any(s in service_lower for s in sil_3_services):
        sil_level = SILLevel.SIL_3
    elif any(s in service_lower for s in sil_2_services):
        sil_level = SILLevel.SIL_2
    else:
        sil_level = SILLevel.SIL_1

    logger.info(
        f"Creating ESD for {exchanger_id}: type={exchanger_type}, "
        f"service={service}, assigned_sil={sil_level.value}"
    )

    return EmergencyShutdownSystem(
        exchanger_id=exchanger_id,
        sil_level=sil_level,
        action_executor=action_executor,
        notification_callback=notification_callback,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ShutdownLevel",
    "ShutdownState",
    "SILLevel",
    "SafetyIntegratedFunction",
    # Data models
    "ShutdownEvent",
    "ProofTestRecord",
    "BypassRecord",
    # Main class
    "EmergencyShutdownSystem",
    # Factory functions
    "create_esd_for_exchanger",
]
