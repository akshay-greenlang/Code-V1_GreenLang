"""
HardwiredInterlock - Hardwired Interlock Integration for ESD Systems

This module implements hardwired interlock status monitoring and management
for Emergency Shutdown Systems per IEC 61511-1 Clause 11.6. Hardwired
interlocks provide direct hardware-based safety functions that operate
independently of software logic.

Key features:
- Digital input mapping for interlock signals
- Interlock bypass management
- Status reporting and logging
- Interlock testing interface
- Complete audit trail with provenance

Reference: IEC 61511-1 Clause 11.6, IEC 61508-2

Example:
    >>> from greenlang.safety.esd.hardwired_interlock import HardwiredInterlockManager
    >>> manager = HardwiredInterlockManager(system_id="ESD-001")
    >>> status = manager.get_interlock_status("IL-001")
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class InterlockState(str, Enum):
    """Interlock operational states."""

    HEALTHY = "healthy"  # Normal operation, interlock active
    TRIPPED = "tripped"  # Interlock has tripped
    BYPASSED = "bypassed"  # Interlock bypassed
    FAULT = "fault"  # Hardware fault detected
    UNKNOWN = "unknown"  # Status unknown
    TESTING = "testing"  # Under test


class InterlockType(str, Enum):
    """Types of hardwired interlocks."""

    PROCESS_SHUTDOWN = "process_shutdown"  # Process variable triggered
    EQUIPMENT_PROTECTION = "equipment_protection"  # Equipment protection
    FIRE_GAS = "fire_gas"  # Fire and gas detection
    MANUAL_SHUTDOWN = "manual_shutdown"  # Manual pushbutton
    EXTERNAL = "external"  # External system interlock
    PERMISSIVE = "permissive"  # Startup permissive


class SignalType(str, Enum):
    """Digital signal types."""

    NORMALLY_OPEN = "normally_open"  # Contact open in safe state
    NORMALLY_CLOSED = "normally_closed"  # Contact closed in safe state
    FAIL_SAFE = "fail_safe"  # De-energize to trip


class InterlockSignal(BaseModel):
    """Digital input signal mapping for interlock."""

    signal_id: str = Field(
        ...,
        description="Signal identifier (e.g., DI-001)"
    )
    signal_name: str = Field(
        ...,
        description="Signal description"
    )
    signal_type: SignalType = Field(
        default=SignalType.NORMALLY_CLOSED,
        description="Signal type"
    )
    io_address: str = Field(
        ...,
        description="I/O address (e.g., I:1/0)"
    )
    current_value: bool = Field(
        default=False,
        description="Current digital value"
    )
    last_change_time: Optional[datetime] = Field(
        None,
        description="Last value change timestamp"
    )
    debounce_ms: int = Field(
        default=50,
        ge=0,
        description="Debounce time (ms)"
    )
    inverted: bool = Field(
        default=False,
        description="Invert logic"
    )


class InterlockDefinition(BaseModel):
    """Hardwired interlock definition."""

    interlock_id: str = Field(
        ...,
        description="Interlock identifier"
    )
    name: str = Field(
        ...,
        description="Interlock name"
    )
    description: str = Field(
        default="",
        description="Detailed description"
    )
    interlock_type: InterlockType = Field(
        ...,
        description="Type of interlock"
    )
    sil_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="SIL level (0=non-SIS)"
    )
    input_signals: List[InterlockSignal] = Field(
        default_factory=list,
        description="Input signals for interlock"
    )
    voting_logic: str = Field(
        default="1oo1",
        description="Voting logic (e.g., 1oo1, 2oo3)"
    )
    trip_action: str = Field(
        ...,
        description="Action on trip"
    )
    affected_equipment: List[str] = Field(
        default_factory=list,
        description="Equipment affected by this interlock"
    )
    response_time_ms: float = Field(
        default=100.0,
        gt=0,
        description="Expected response time (ms)"
    )
    bypass_allowed: bool = Field(
        default=False,
        description="Can this interlock be bypassed"
    )
    test_interval_days: int = Field(
        default=365,
        gt=0,
        description="Test interval (days)"
    )
    last_test_date: Optional[datetime] = Field(
        None,
        description="Last test date"
    )


class InterlockStatus(BaseModel):
    """Current interlock status report."""

    interlock_id: str = Field(
        ...,
        description="Interlock identifier"
    )
    state: InterlockState = Field(
        ...,
        description="Current state"
    )
    is_tripped: bool = Field(
        default=False,
        description="Is interlock currently tripped"
    )
    is_bypassed: bool = Field(
        default=False,
        description="Is interlock bypassed"
    )
    bypass_id: Optional[str] = Field(
        None,
        description="Active bypass ID"
    )
    signal_states: Dict[str, bool] = Field(
        default_factory=dict,
        description="Current state of all input signals"
    )
    trip_time: Optional[datetime] = Field(
        None,
        description="Time of last trip"
    )
    trip_signal: Optional[str] = Field(
        None,
        description="Signal that caused trip"
    )
    trip_count: int = Field(
        default=0,
        description="Total trip count"
    )
    fault_code: Optional[str] = Field(
        None,
        description="Active fault code"
    )
    fault_description: Optional[str] = Field(
        None,
        description="Fault description"
    )
    test_overdue: bool = Field(
        default=False,
        description="Is proof test overdue"
    )
    next_test_due: Optional[datetime] = Field(
        None,
        description="Next test due date"
    )
    uptime_hours: float = Field(
        default=0.0,
        description="Hours since last trip"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Status timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class InterlockBypass(BaseModel):
    """Interlock bypass record."""

    bypass_id: str = Field(
        default_factory=lambda: f"ILBYP-{uuid.uuid4().hex[:8].upper()}",
        description="Bypass identifier"
    )
    interlock_id: str = Field(
        ...,
        description="Interlock being bypassed"
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Reason for bypass"
    )
    requested_by: str = Field(
        ...,
        description="Requester"
    )
    authorized_by: str = Field(
        ...,
        description="Authorizer"
    )
    activated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Activation time"
    )
    expires_at: datetime = Field(
        ...,
        description="Expiration time"
    )
    deactivated_at: Optional[datetime] = Field(
        None,
        description="Deactivation time"
    )
    is_active: bool = Field(
        default=True,
        description="Is bypass active"
    )
    compensating_measures: List[str] = Field(
        default_factory=list,
        description="Compensating measures in place"
    )
    work_permit_ref: Optional[str] = Field(
        None,
        description="Work permit reference"
    )
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Audit trail"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash"
    )


class InterlockTestResult(BaseModel):
    """Result of interlock test."""

    test_id: str = Field(
        default_factory=lambda: f"ILTEST-{uuid.uuid4().hex[:8].upper()}",
        description="Test identifier"
    )
    interlock_id: str = Field(
        ...,
        description="Interlock tested"
    )
    test_type: str = Field(
        default="functional",
        description="Type of test (functional, trip, response)"
    )
    test_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Test date"
    )
    tester: str = Field(
        ...,
        description="Person conducting test"
    )
    witness: Optional[str] = Field(
        None,
        description="Witness (if required)"
    )
    passed: bool = Field(
        ...,
        description="Did test pass"
    )
    measured_response_ms: Optional[float] = Field(
        None,
        description="Measured response time (ms)"
    )
    required_response_ms: float = Field(
        default=1000.0,
        description="Required response time (ms)"
    )
    signal_tests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual signal test results"
    )
    trip_verified: bool = Field(
        default=False,
        description="Was trip action verified"
    )
    notes: str = Field(
        default="",
        description="Test notes"
    )
    anomalies: List[str] = Field(
        default_factory=list,
        description="Anomalies detected"
    )
    next_test_due: datetime = Field(
        ...,
        description="Next test due date"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash"
    )


class HardwiredInterlockManager:
    """
    Hardwired Interlock Manager.

    Manages hardwired interlock status monitoring, bypass management,
    and testing for Emergency Shutdown Systems per IEC 61511.

    Key responsibilities:
    - Monitor interlock input signals
    - Manage interlock bypasses
    - Track test schedules
    - Generate status reports
    - Maintain audit trails

    The manager follows IEC 61511 principles:
    - Hardwired interlocks are highest priority
    - Complete traceability
    - Fail-safe operation

    Attributes:
        system_id: ESD system identifier
        interlocks: Registered interlock definitions
        bypasses: Active bypass records

    Example:
        >>> manager = HardwiredInterlockManager(system_id="ESD-001")
        >>> manager.register_interlock(interlock_def)
        >>> status = manager.get_interlock_status("IL-001")
    """

    def __init__(
        self,
        system_id: str,
        signal_reader: Optional[Callable[[str], bool]] = None,
        trip_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize HardwiredInterlockManager.

        Args:
            system_id: ESD system identifier
            signal_reader: Callback to read digital signals
            trip_callback: Callback when interlock trips
        """
        self.system_id = system_id
        self.signal_reader = signal_reader or self._default_signal_reader
        self.trip_callback = trip_callback or self._default_trip_callback

        self.interlocks: Dict[str, InterlockDefinition] = {}
        self.interlock_states: Dict[str, InterlockState] = {}
        self.bypasses: Dict[str, InterlockBypass] = {}
        self.test_history: Dict[str, List[InterlockTestResult]] = {}
        self.trip_history: List[Dict[str, Any]] = []

        # Runtime state
        self._last_signal_values: Dict[str, Dict[str, bool]] = {}
        self._trip_times: Dict[str, datetime] = {}
        self._trip_counts: Dict[str, int] = {}

        logger.info(f"HardwiredInterlockManager initialized: {system_id}")

    def register_interlock(
        self,
        definition: InterlockDefinition
    ) -> None:
        """
        Register a hardwired interlock.

        Args:
            definition: Interlock definition
        """
        self.interlocks[definition.interlock_id] = definition
        self.interlock_states[definition.interlock_id] = InterlockState.HEALTHY
        self._last_signal_values[definition.interlock_id] = {}
        self._trip_counts[definition.interlock_id] = 0

        logger.info(
            f"Registered interlock {definition.interlock_id}: "
            f"{definition.name}"
        )

    def update_signal_value(
        self,
        interlock_id: str,
        signal_id: str,
        value: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Update a signal value and check for trip condition.

        Args:
            interlock_id: Interlock identifier
            signal_id: Signal identifier
            value: New signal value

        Returns:
            Trip event if interlock tripped, None otherwise
        """
        if interlock_id not in self.interlocks:
            logger.warning(f"Unknown interlock: {interlock_id}")
            return None

        definition = self.interlocks[interlock_id]

        # Find signal definition
        signal_def = None
        for sig in definition.input_signals:
            if sig.signal_id == signal_id:
                signal_def = sig
                break

        if not signal_def:
            logger.warning(f"Unknown signal {signal_id} for {interlock_id}")
            return None

        # Store previous value
        prev_value = self._last_signal_values.get(
            interlock_id, {}
        ).get(signal_id, None)

        # Update signal value
        signal_def.current_value = value
        if interlock_id not in self._last_signal_values:
            self._last_signal_values[interlock_id] = {}
        self._last_signal_values[interlock_id][signal_id] = value

        if prev_value != value:
            signal_def.last_change_time = datetime.utcnow()
            logger.debug(
                f"Signal {signal_id} changed: {prev_value} -> {value}"
            )

        # Check for trip condition
        if self._check_trip_condition(interlock_id, definition):
            return self._handle_trip(interlock_id, signal_id)

        return None

    def read_all_signals(
        self,
        interlock_id: str
    ) -> Dict[str, bool]:
        """
        Read all signals for an interlock.

        Args:
            interlock_id: Interlock identifier

        Returns:
            Dictionary of signal values
        """
        if interlock_id not in self.interlocks:
            return {}

        definition = self.interlocks[interlock_id]
        values = {}

        for signal in definition.input_signals:
            try:
                value = self.signal_reader(signal.io_address)
                if signal.inverted:
                    value = not value
                values[signal.signal_id] = value
                self.update_signal_value(interlock_id, signal.signal_id, value)
            except Exception as e:
                logger.error(
                    f"Error reading signal {signal.signal_id}: {e}"
                )
                values[signal.signal_id] = False

        return values

    def get_interlock_status(
        self,
        interlock_id: str
    ) -> Optional[InterlockStatus]:
        """
        Get current status of an interlock.

        Args:
            interlock_id: Interlock identifier

        Returns:
            InterlockStatus or None if not found
        """
        if interlock_id not in self.interlocks:
            return None

        definition = self.interlocks[interlock_id]
        state = self.interlock_states.get(interlock_id, InterlockState.UNKNOWN)

        # Check bypass status
        is_bypassed = False
        bypass_id = None
        for bypass in self.bypasses.values():
            if (bypass.interlock_id == interlock_id and
                bypass.is_active and
                datetime.utcnow() < bypass.expires_at):
                is_bypassed = True
                bypass_id = bypass.bypass_id
                break

        # Check test status
        test_overdue = False
        next_test_due = None
        if definition.last_test_date:
            next_test_due = definition.last_test_date + timedelta(
                days=definition.test_interval_days
            )
            test_overdue = datetime.utcnow() > next_test_due

        # Get signal states
        signal_states = {}
        for signal in definition.input_signals:
            signal_states[signal.signal_id] = signal.current_value

        # Calculate uptime
        trip_time = self._trip_times.get(interlock_id)
        uptime_hours = 0.0
        if trip_time:
            uptime_hours = (
                datetime.utcnow() - trip_time
            ).total_seconds() / 3600

        status = InterlockStatus(
            interlock_id=interlock_id,
            state=state,
            is_tripped=state == InterlockState.TRIPPED,
            is_bypassed=is_bypassed,
            bypass_id=bypass_id,
            signal_states=signal_states,
            trip_time=trip_time,
            trip_count=self._trip_counts.get(interlock_id, 0),
            test_overdue=test_overdue,
            next_test_due=next_test_due,
            uptime_hours=uptime_hours,
        )

        status.provenance_hash = self._calculate_provenance(status)
        return status

    def activate_bypass(
        self,
        interlock_id: str,
        reason: str,
        requested_by: str,
        authorized_by: str,
        duration_hours: float,
        compensating_measures: List[str],
        work_permit_ref: Optional[str] = None
    ) -> InterlockBypass:
        """
        Activate a bypass for an interlock.

        Args:
            interlock_id: Interlock to bypass
            reason: Bypass reason
            requested_by: Requester name
            authorized_by: Authorizer name
            duration_hours: Bypass duration
            compensating_measures: Compensating measures
            work_permit_ref: Work permit reference

        Returns:
            InterlockBypass record

        Raises:
            ValueError: If bypass not allowed
        """
        if interlock_id not in self.interlocks:
            raise ValueError(f"Unknown interlock: {interlock_id}")

        definition = self.interlocks[interlock_id]

        if not definition.bypass_allowed:
            raise ValueError(
                f"Bypass not allowed for {interlock_id}"
            )

        # Check for existing active bypass
        for bypass in self.bypasses.values():
            if (bypass.interlock_id == interlock_id and
                bypass.is_active):
                raise ValueError(
                    f"Active bypass already exists: {bypass.bypass_id}"
                )

        bypass = InterlockBypass(
            interlock_id=interlock_id,
            reason=reason,
            requested_by=requested_by,
            authorized_by=authorized_by,
            expires_at=datetime.utcnow() + timedelta(hours=duration_hours),
            compensating_measures=compensating_measures,
            work_permit_ref=work_permit_ref,
            audit_trail=[{
                "action": "activated",
                "timestamp": datetime.utcnow().isoformat(),
                "user": authorized_by,
                "details": {
                    "duration_hours": duration_hours,
                    "reason": reason,
                }
            }]
        )

        bypass.provenance_hash = hashlib.sha256(
            f"{bypass.bypass_id}|{bypass.interlock_id}|{bypass.activated_at.isoformat()}".encode()
        ).hexdigest()

        self.bypasses[bypass.bypass_id] = bypass
        self.interlock_states[interlock_id] = InterlockState.BYPASSED

        logger.warning(
            f"BYPASS ACTIVATED: {interlock_id} by {authorized_by} "
            f"(expires: {bypass.expires_at.isoformat()})"
        )

        return bypass

    def deactivate_bypass(
        self,
        bypass_id: str,
        deactivated_by: str
    ) -> InterlockBypass:
        """
        Deactivate a bypass.

        Args:
            bypass_id: Bypass to deactivate
            deactivated_by: Person deactivating

        Returns:
            Updated InterlockBypass

        Raises:
            ValueError: If bypass not found
        """
        if bypass_id not in self.bypasses:
            raise ValueError(f"Bypass not found: {bypass_id}")

        bypass = self.bypasses[bypass_id]
        bypass.is_active = False
        bypass.deactivated_at = datetime.utcnow()

        bypass.audit_trail.append({
            "action": "deactivated",
            "timestamp": datetime.utcnow().isoformat(),
            "user": deactivated_by,
        })

        # Restore interlock state
        self.interlock_states[bypass.interlock_id] = InterlockState.HEALTHY

        logger.info(
            f"Bypass {bypass_id} deactivated by {deactivated_by}"
        )

        return bypass

    def check_expired_bypasses(self) -> List[InterlockBypass]:
        """
        Check and expire bypasses that have exceeded time limit.

        Returns:
            List of expired bypasses
        """
        now = datetime.utcnow()
        expired = []

        for bypass in self.bypasses.values():
            if bypass.is_active and now > bypass.expires_at:
                bypass.is_active = False
                bypass.deactivated_at = now

                bypass.audit_trail.append({
                    "action": "expired",
                    "timestamp": now.isoformat(),
                    "details": {
                        "scheduled_expiry": bypass.expires_at.isoformat(),
                    }
                })

                self.interlock_states[bypass.interlock_id] = InterlockState.HEALTHY
                expired.append(bypass)

                logger.warning(
                    f"BYPASS EXPIRED: {bypass.bypass_id} "
                    f"for {bypass.interlock_id}"
                )

        return expired

    def run_interlock_test(
        self,
        interlock_id: str,
        tester: str,
        witness: Optional[str] = None,
        simulate_trip: bool = False
    ) -> InterlockTestResult:
        """
        Run a functional test on an interlock.

        Args:
            interlock_id: Interlock to test
            tester: Person conducting test
            witness: Witness (if required)
            simulate_trip: Whether to simulate trip

        Returns:
            InterlockTestResult

        Raises:
            ValueError: If interlock not found
        """
        if interlock_id not in self.interlocks:
            raise ValueError(f"Unknown interlock: {interlock_id}")

        definition = self.interlocks[interlock_id]

        logger.info(
            f"Starting interlock test: {interlock_id} by {tester}"
        )

        # Set state to testing
        prev_state = self.interlock_states.get(interlock_id)
        self.interlock_states[interlock_id] = InterlockState.TESTING

        signal_tests = []
        all_passed = True
        anomalies = []

        # Test each input signal
        for signal in definition.input_signals:
            try:
                value = self.signal_reader(signal.io_address)
                expected = (
                    signal.signal_type == SignalType.NORMALLY_CLOSED
                )
                if signal.inverted:
                    expected = not expected

                passed = value == expected

                signal_tests.append({
                    "signal_id": signal.signal_id,
                    "signal_name": signal.signal_name,
                    "value": value,
                    "expected": expected,
                    "passed": passed,
                })

                if not passed:
                    all_passed = False
                    anomalies.append(
                        f"Signal {signal.signal_id} unexpected value: "
                        f"{value} (expected {expected})"
                    )

            except Exception as e:
                signal_tests.append({
                    "signal_id": signal.signal_id,
                    "signal_name": signal.signal_name,
                    "error": str(e),
                    "passed": False,
                })
                all_passed = False
                anomalies.append(f"Signal {signal.signal_id} read error: {e}")

        # Measure response time if simulating trip
        measured_response_ms = None
        trip_verified = False

        if simulate_trip and all_passed:
            import time
            start_time = time.time()

            # Simulate trip signal
            for signal in definition.input_signals:
                trip_value = (
                    signal.signal_type != SignalType.NORMALLY_CLOSED
                )
                self.update_signal_value(
                    interlock_id,
                    signal.signal_id,
                    trip_value
                )

            end_time = time.time()
            measured_response_ms = (end_time - start_time) * 1000
            trip_verified = True

            # Reset signals to normal
            for signal in definition.input_signals:
                normal_value = (
                    signal.signal_type == SignalType.NORMALLY_CLOSED
                )
                self.update_signal_value(
                    interlock_id,
                    signal.signal_id,
                    normal_value
                )

        # Calculate next test due
        next_test_due = datetime.utcnow() + timedelta(
            days=definition.test_interval_days
        )

        # Update definition
        definition.last_test_date = datetime.utcnow()

        # Restore state
        self.interlock_states[interlock_id] = prev_state or InterlockState.HEALTHY

        result = InterlockTestResult(
            interlock_id=interlock_id,
            tester=tester,
            witness=witness,
            passed=all_passed,
            measured_response_ms=measured_response_ms,
            required_response_ms=definition.response_time_ms,
            signal_tests=signal_tests,
            trip_verified=trip_verified,
            anomalies=anomalies,
            next_test_due=next_test_due,
        )

        result.provenance_hash = hashlib.sha256(
            f"{result.test_id}|{interlock_id}|{result.passed}|{result.test_date.isoformat()}".encode()
        ).hexdigest()

        # Store in history
        if interlock_id not in self.test_history:
            self.test_history[interlock_id] = []
        self.test_history[interlock_id].append(result)

        logger.info(
            f"Interlock test {result.test_id}: "
            f"{'PASS' if all_passed else 'FAIL'}"
        )

        return result

    def reset_tripped_interlock(
        self,
        interlock_id: str,
        reset_by: str,
        reason: str = "Normal reset"
    ) -> Dict[str, Any]:
        """
        Reset a tripped interlock.

        Args:
            interlock_id: Interlock to reset
            reset_by: Person authorizing reset
            reason: Reset reason

        Returns:
            Reset result dictionary
        """
        if interlock_id not in self.interlocks:
            raise ValueError(f"Unknown interlock: {interlock_id}")

        if self.interlock_states.get(interlock_id) != InterlockState.TRIPPED:
            return {
                "status": "not_tripped",
                "message": "Interlock is not in tripped state"
            }

        self.interlock_states[interlock_id] = InterlockState.HEALTHY

        logger.info(
            f"Interlock {interlock_id} reset by {reset_by}: {reason}"
        )

        return {
            "status": "success",
            "interlock_id": interlock_id,
            "reset_by": reset_by,
            "reset_time": datetime.utcnow().isoformat(),
            "reason": reason,
        }

    def get_all_interlocks_status(self) -> Dict[str, Any]:
        """
        Get status of all registered interlocks.

        Returns:
            Summary report of all interlocks
        """
        statuses = []
        for interlock_id in self.interlocks:
            status = self.get_interlock_status(interlock_id)
            if status:
                statuses.append(status)

        healthy_count = sum(
            1 for s in statuses if s.state == InterlockState.HEALTHY
        )
        tripped_count = sum(
            1 for s in statuses if s.state == InterlockState.TRIPPED
        )
        bypassed_count = sum(
            1 for s in statuses if s.state == InterlockState.BYPASSED
        )
        fault_count = sum(
            1 for s in statuses if s.state == InterlockState.FAULT
        )
        test_overdue_count = sum(
            1 for s in statuses if s.test_overdue
        )

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "total_interlocks": len(self.interlocks),
            "healthy_count": healthy_count,
            "tripped_count": tripped_count,
            "bypassed_count": bypassed_count,
            "fault_count": fault_count,
            "test_overdue_count": test_overdue_count,
            "interlocks": [
                {
                    "interlock_id": s.interlock_id,
                    "state": s.state.value,
                    "is_tripped": s.is_tripped,
                    "is_bypassed": s.is_bypassed,
                    "trip_count": s.trip_count,
                    "test_overdue": s.test_overdue,
                }
                for s in statuses
            ],
            "provenance_hash": hashlib.sha256(
                f"{datetime.utcnow().isoformat()}|{len(statuses)}|{healthy_count}".encode()
            ).hexdigest()
        }

    def _check_trip_condition(
        self,
        interlock_id: str,
        definition: InterlockDefinition
    ) -> bool:
        """Check if trip condition is met based on voting logic."""
        # Check if bypassed
        for bypass in self.bypasses.values():
            if (bypass.interlock_id == interlock_id and
                bypass.is_active and
                datetime.utcnow() < bypass.expires_at):
                return False

        # Parse voting logic (e.g., "1oo2", "2oo3")
        voting = definition.voting_logic
        try:
            required, total = voting.split("oo")
            required = int(required)
        except (ValueError, IndexError):
            required = 1

        # Count tripped signals
        tripped_count = 0
        for signal in definition.input_signals:
            trip_value = signal.signal_type != SignalType.NORMALLY_CLOSED
            if signal.inverted:
                trip_value = not trip_value
            if signal.current_value == trip_value:
                tripped_count += 1

        return tripped_count >= required

    def _handle_trip(
        self,
        interlock_id: str,
        trip_signal: str
    ) -> Dict[str, Any]:
        """Handle interlock trip condition."""
        now = datetime.utcnow()

        self.interlock_states[interlock_id] = InterlockState.TRIPPED
        self._trip_times[interlock_id] = now
        self._trip_counts[interlock_id] = (
            self._trip_counts.get(interlock_id, 0) + 1
        )

        definition = self.interlocks[interlock_id]

        trip_event = {
            "event_id": f"TRIP-{uuid.uuid4().hex[:8].upper()}",
            "interlock_id": interlock_id,
            "interlock_name": definition.name,
            "trip_signal": trip_signal,
            "trip_time": now.isoformat(),
            "trip_action": definition.trip_action,
            "affected_equipment": definition.affected_equipment,
            "trip_count": self._trip_counts[interlock_id],
        }

        self.trip_history.append(trip_event)

        logger.critical(
            f"INTERLOCK TRIP: {interlock_id} ({definition.name}) "
            f"by signal {trip_signal}"
        )

        # Execute trip callback
        self.trip_callback(interlock_id, definition.trip_action)

        return trip_event

    def _default_signal_reader(self, io_address: str) -> bool:
        """Default signal reader (simulation)."""
        # In production, this would read from PLC/DCS
        return True  # Normal state

    def _default_trip_callback(
        self,
        interlock_id: str,
        action: str
    ) -> None:
        """Default trip callback."""
        logger.warning(
            f"TRIP CALLBACK: {interlock_id} -> {action}"
        )

    def _calculate_provenance(self, status: InterlockStatus) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{status.interlock_id}|"
            f"{status.state.value}|"
            f"{status.is_tripped}|"
            f"{status.trip_count}|"
            f"{status.timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
