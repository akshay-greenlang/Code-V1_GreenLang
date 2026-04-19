"""
GL-004 BURNMASTER - Write Controller

Centralized write control with mandatory safety checks, mode verification,
envelope validation, and comprehensive audit logging.

ALL writes to DCS MUST go through this controller. No direct writes allowed.

Safety Checks Required for ALL Writes:
    1. Mode Check - Unit must be in permissive mode
    2. Safety Check - BMS must report safe status
    3. Envelope Check - Value must be within safety envelope
    4. Audit Log - All writes logged with full context

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class WriteRequestStatus(str, Enum):
    """Status of write request."""
    PENDING = "pending"
    VALIDATING = "validating"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class WriteRejectReason(str, Enum):
    """Reasons for write rejection."""
    MODE_NOT_PERMISSIVE = "mode_not_permissive"
    SAFETY_CHECK_FAILED = "safety_check_failed"
    ENVELOPE_VIOLATION = "envelope_violation"
    INVALID_TAG = "invalid_tag"
    INVALID_VALUE = "invalid_value"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    VERIFICATION_FAILED = "verification_failed"
    TIMEOUT = "timeout"
    DCS_ERROR = "dcs_error"
    BMS_UNSAFE = "bms_unsafe"


class WriteType(str, Enum):
    """Types of write operations."""
    SETPOINT = "setpoint"
    OUTPUT = "output"
    MODE_CHANGE = "mode_change"
    PARAMETER = "parameter"


@dataclass
class WriteRequest:
    """Write request with full context."""
    request_id: str
    tag: str
    value: float
    unit_id: str
    write_type: WriteType = WriteType.SETPOINT
    user_id: str = "system"
    session_id: str = ""
    reason: str = ""
    correlation_id: Optional[str] = None
    priority: int = 5  # 1-10, higher = more urgent
    timeout_seconds: float = 30.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


@dataclass
class ValidationResult:
    """Result of write validation checks."""
    is_valid: bool
    request_id: str
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    reject_reason: Optional[WriteRejectReason] = None
    details: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "request_id": self.request_id,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "reject_reason": self.reject_reason.value if self.reject_reason else None,
        }


@dataclass
class WriteResult:
    """Result of write execution."""
    success: bool
    request_id: str
    tag: str
    requested_value: float
    actual_value: Optional[float] = None
    status: WriteRequestStatus = WriteRequestStatus.COMPLETED
    reject_reason: Optional[WriteRejectReason] = None
    audit_entry_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    verified: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "request_id": self.request_id,
            "tag": self.tag,
            "requested_value": self.requested_value,
            "actual_value": self.actual_value,
            "status": self.status.value,
            "reject_reason": self.reject_reason.value if self.reject_reason else None,
            "audit_entry_id": self.audit_entry_id,
            "verified": self.verified,
        }


@dataclass
class RollbackResult:
    """Result of write rollback."""
    success: bool
    request_id: str
    tag: str
    original_value: Optional[float]
    rollback_value: Optional[float]
    message: str = ""


@dataclass
class SafetyEnvelope:
    """Safety envelope for a tag."""
    tag: str
    min_value: float
    max_value: float
    max_rate_of_change: Optional[float] = None  # per second
    engineering_units: str = ""


class WriteController:
    """
    Centralized write controller for combustion optimization.

    ALL writes to DCS MUST go through this controller.

    Safety Sequence for Every Write:
    1. Mode Check - Verify unit is in permissive mode
    2. Safety Check - Verify BMS reports safe status
    3. Envelope Check - Verify value within safety limits
    4. Execute Write - Send to DCS
    5. Verify Write - Read back and confirm
    6. Audit Log - Record full transaction

    If any check fails, the write is rejected and logged.
    """

    def __init__(
        self,
        dcs_connector=None,
        bms_interface=None,
        audit_logger=None,
        safety_envelopes: Optional[Dict[str, SafetyEnvelope]] = None,
    ):
        """
        Initialize write controller.

        Args:
            dcs_connector: DCS connector instance
            bms_interface: BMS interface instance (read-only)
            audit_logger: Audit logger instance
            safety_envelopes: Dict of tag to SafetyEnvelope
        """
        self._dcs = dcs_connector
        self._bms = bms_interface
        self._audit_logger = audit_logger
        self._envelopes: Dict[str, SafetyEnvelope] = safety_envelopes or {}

        # Track pending and recent writes
        self._pending_writes: Dict[str, WriteRequest] = {}
        self._write_history: List[WriteResult] = []
        self._last_values: Dict[str, float] = {}  # For rollback

        # Rate limiting
        self._write_timestamps: Dict[str, List[datetime]] = {}
        self._max_writes_per_minute: int = 10

        # Statistics
        self._stats = {
            "total_requests": 0,
            "approved": 0,
            "rejected": 0,
            "completed": 0,
            "failed": 0,
            "rolled_back": 0,
        }

        self._lock = asyncio.Lock()
        logger.info("WriteController initialized")

    def register_envelope(self, envelope: SafetyEnvelope) -> None:
        """Register safety envelope for a tag."""
        self._envelopes[envelope.tag] = envelope
        logger.debug(f"Registered safety envelope for {envelope.tag}")

    async def request_write(self, request: WriteRequest) -> WriteResult:
        """
        Request a write operation with full safety checks.

        This is the ONLY entry point for all DCS writes.

        Args:
            request: WriteRequest with tag, value, and context

        Returns:
            WriteResult with success status and audit trail
        """
        async with self._lock:
            self._stats["total_requests"] += 1
            self._pending_writes[request.request_id] = request

            start_time = datetime.now(timezone.utc)

            try:
                # Step 1: Validate the write request
                validation = await self.validate_write_request(request)

                if not validation.is_valid:
                    self._stats["rejected"] += 1
                    result = WriteResult(
                        success=False,
                        request_id=request.request_id,
                        tag=request.tag,
                        requested_value=request.value,
                        status=WriteRequestStatus.REJECTED,
                        reject_reason=validation.reject_reason,
                        error_message=f"Validation failed: {validation.checks_failed}",
                        started_at=start_time,
                        completed_at=datetime.now(timezone.utc),
                    )

                    # Log rejection
                    await self._log_write_attempt(request, result, validation)
                    return result

                # Step 2: Execute the write
                result = await self.execute_write(request, validation)

                if result.success:
                    # Step 3: Verify the write
                    verification = await self.verify_write_success(request)
                    result.verified = verification.success

                    if not verification.success:
                        logger.warning(f"Write verification failed for {request.tag}")
                        # Don't fail the write, but log the verification issue

                # Record in history
                self._write_history.append(result)
                if len(self._write_history) > 1000:
                    self._write_history = self._write_history[-500:]

                return result

            except Exception as e:
                self._stats["failed"] += 1
                logger.error(f"Write request failed: {e}")

                result = WriteResult(
                    success=False,
                    request_id=request.request_id,
                    tag=request.tag,
                    requested_value=request.value,
                    status=WriteRequestStatus.FAILED,
                    error_message=str(e),
                    started_at=start_time,
                    completed_at=datetime.now(timezone.utc),
                )

                await self._log_write_attempt(request, result, None)
                return result

            finally:
                # Clean up pending
                if request.request_id in self._pending_writes:
                    del self._pending_writes[request.request_id]

    async def validate_write_request(self, request: WriteRequest) -> ValidationResult:
        """
        Validate write request against all safety checks.

        Checks performed:
        1. Rate limit check
        2. Mode permissive check (DCS)
        3. Safety check (BMS)
        4. Envelope check (safety limits)
        5. Value sanity check

        Args:
            request: WriteRequest to validate

        Returns:
            ValidationResult with pass/fail status
        """
        checks_passed = []
        checks_failed = []
        reject_reason = None
        details = {}

        # Check 1: Rate limiting
        if not self._check_rate_limit(request.tag):
            checks_failed.append("rate_limit")
            reject_reason = WriteRejectReason.RATE_LIMIT_EXCEEDED
            details["rate_limit"] = f"Max {self._max_writes_per_minute} writes/min exceeded"
        else:
            checks_passed.append("rate_limit")

        # Check 2: Mode permissive (DCS)
        if self._dcs and reject_reason is None:
            try:
                perm_status = await self._dcs.check_mode_permissive(request.unit_id)
                if perm_status.is_permissive:
                    checks_passed.append("mode_permissive")
                else:
                    checks_failed.append("mode_permissive")
                    reject_reason = WriteRejectReason.MODE_NOT_PERMISSIVE
                    details["mode"] = perm_status.blocking_conditions
            except Exception as e:
                checks_failed.append("mode_permissive")
                reject_reason = WriteRejectReason.DCS_ERROR
                details["mode_error"] = str(e)
        else:
            checks_passed.append("mode_permissive")  # Simulated pass

        # Check 3: Safety check (BMS)
        if self._bms and reject_reason is None:
            try:
                safety = await self._bms.check_safe_to_write(request.unit_id)
                if safety.is_safe:
                    checks_passed.append("bms_safety")
                else:
                    checks_failed.append("bms_safety")
                    reject_reason = WriteRejectReason.BMS_UNSAFE
                    details["bms"] = safety.blocking_conditions
            except Exception as e:
                checks_failed.append("bms_safety")
                reject_reason = WriteRejectReason.SAFETY_CHECK_FAILED
                details["bms_error"] = str(e)
        else:
            checks_passed.append("bms_safety")  # Simulated pass

        # Check 4: Envelope check
        if reject_reason is None:
            envelope = self._envelopes.get(request.tag)
            if envelope:
                if request.value < envelope.min_value:
                    checks_failed.append("envelope_min")
                    reject_reason = WriteRejectReason.ENVELOPE_VIOLATION
                    details["envelope"] = f"Value {request.value} below min {envelope.min_value}"
                elif request.value > envelope.max_value:
                    checks_failed.append("envelope_max")
                    reject_reason = WriteRejectReason.ENVELOPE_VIOLATION
                    details["envelope"] = f"Value {request.value} above max {envelope.max_value}"
                else:
                    checks_passed.append("envelope")

                # Check rate of change
                if envelope.max_rate_of_change and request.tag in self._last_values:
                    last_value = self._last_values[request.tag]
                    change = abs(request.value - last_value)
                    if change > envelope.max_rate_of_change:
                        checks_failed.append("rate_of_change")
                        reject_reason = WriteRejectReason.ENVELOPE_VIOLATION
                        details["rate_of_change"] = f"Change {change} exceeds max {envelope.max_rate_of_change}"
                    else:
                        checks_passed.append("rate_of_change")
            else:
                checks_passed.append("envelope")  # No envelope defined

        # Check 5: Value sanity
        if reject_reason is None:
            if not isinstance(request.value, (int, float)):
                checks_failed.append("value_type")
                reject_reason = WriteRejectReason.INVALID_VALUE
            elif request.value != request.value:  # NaN check
                checks_failed.append("value_nan")
                reject_reason = WriteRejectReason.INVALID_VALUE
            else:
                checks_passed.append("value_sanity")

        is_valid = reject_reason is None

        return ValidationResult(
            is_valid=is_valid,
            request_id=request.request_id,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            reject_reason=reject_reason,
            details=details,
        )

    async def execute_write(
        self,
        request: WriteRequest,
        validation: ValidationResult,
    ) -> WriteResult:
        """
        Execute validated write to DCS.

        Args:
            request: Validated WriteRequest
            validation: Successful ValidationResult

        Returns:
            WriteResult with execution status
        """
        start_time = datetime.now(timezone.utc)

        # Store current value for potential rollback
        if self._dcs:
            try:
                current = await self._dcs.read_tag(request.tag)
                self._last_values[request.tag] = float(current.value)
            except Exception:
                pass  # Continue without rollback capability

        # Generate audit ID
        audit_id = hashlib.sha256(
            f"{request.request_id}|{request.tag}|{request.value}|{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        try:
            if self._dcs:
                # Create audit context
                from .dcs_connector import AuditContext
                audit_context = AuditContext(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    reason=request.reason,
                    correlation_id=request.correlation_id,
                )

                # Execute write
                dcs_result = await self._dcs.write_tag(
                    request.tag,
                    request.value,
                    audit_context,
                )

                if dcs_result.success:
                    self._stats["completed"] += 1
                    self._record_write_timestamp(request.tag)

                    result = WriteResult(
                        success=True,
                        request_id=request.request_id,
                        tag=request.tag,
                        requested_value=request.value,
                        actual_value=dcs_result.actual_value,
                        status=WriteRequestStatus.COMPLETED,
                        audit_entry_id=audit_id,
                        execution_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                        started_at=start_time,
                        completed_at=datetime.now(timezone.utc),
                    )
                else:
                    self._stats["failed"] += 1
                    result = WriteResult(
                        success=False,
                        request_id=request.request_id,
                        tag=request.tag,
                        requested_value=request.value,
                        status=WriteRequestStatus.FAILED,
                        reject_reason=WriteRejectReason.DCS_ERROR,
                        error_message=dcs_result.error_message,
                        audit_entry_id=audit_id,
                        started_at=start_time,
                        completed_at=datetime.now(timezone.utc),
                    )
            else:
                # Simulation mode
                self._stats["completed"] += 1
                self._record_write_timestamp(request.tag)
                self._last_values[request.tag] = request.value

                result = WriteResult(
                    success=True,
                    request_id=request.request_id,
                    tag=request.tag,
                    requested_value=request.value,
                    actual_value=request.value,
                    status=WriteRequestStatus.COMPLETED,
                    audit_entry_id=audit_id,
                    execution_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                    started_at=start_time,
                    completed_at=datetime.now(timezone.utc),
                )

            # Log to audit logger
            await self._log_write_attempt(request, result, validation)

            return result

        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Write execution failed: {e}")

            return WriteResult(
                success=False,
                request_id=request.request_id,
                tag=request.tag,
                requested_value=request.value,
                status=WriteRequestStatus.FAILED,
                error_message=str(e),
                audit_entry_id=audit_id,
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
            )

    async def verify_write_success(
        self,
        request: WriteRequest,
        tolerance_pct: float = 1.0,
    ) -> ValidationResult:
        """
        Verify write was successful by reading back value.

        Args:
            request: Original WriteRequest
            tolerance_pct: Acceptable deviation percentage

        Returns:
            ValidationResult indicating verification status
        """
        checks_passed = []
        checks_failed = []
        details = {}

        if self._dcs:
            try:
                readback = await self._dcs.read_tag(request.tag)
                actual_value = float(readback.value)
                expected_value = request.value

                if expected_value != 0:
                    deviation_pct = abs(actual_value - expected_value) / abs(expected_value) * 100
                else:
                    deviation_pct = abs(actual_value - expected_value) * 100

                if deviation_pct <= tolerance_pct:
                    checks_passed.append("readback_verification")
                    details["deviation_pct"] = deviation_pct
                else:
                    checks_failed.append("readback_verification")
                    details["deviation_pct"] = deviation_pct
                    details["expected"] = expected_value
                    details["actual"] = actual_value

            except Exception as e:
                checks_failed.append("readback_verification")
                details["error"] = str(e)
        else:
            # Simulation mode - assume success
            checks_passed.append("readback_verification")

        return ValidationResult(
            is_valid=len(checks_failed) == 0,
            request_id=request.request_id,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            details=details,
        )

    async def rollback_write(self, request_id: str) -> RollbackResult:
        """
        Rollback a write to previous value.

        Args:
            request_id: ID of write request to rollback

        Returns:
            RollbackResult with rollback status
        """
        # Find the write in history
        write_result = None
        for result in reversed(self._write_history):
            if result.request_id == request_id:
                write_result = result
                break

        if not write_result:
            return RollbackResult(
                success=False,
                request_id=request_id,
                tag="",
                original_value=None,
                rollback_value=None,
                message=f"Write request {request_id} not found in history",
            )

        tag = write_result.tag
        original_value = self._last_values.get(tag)

        if original_value is None:
            return RollbackResult(
                success=False,
                request_id=request_id,
                tag=tag,
                original_value=None,
                rollback_value=None,
                message=f"No original value stored for rollback of {tag}",
            )

        # Create rollback write request
        rollback_request = WriteRequest(
            request_id=str(uuid.uuid4()),
            tag=tag,
            value=original_value,
            unit_id=tag.split(".")[0] if "." in tag else "UNIT1",
            write_type=WriteType.SETPOINT,
            user_id="system",
            reason=f"Rollback of write {request_id}",
            correlation_id=request_id,
        )

        # Execute rollback
        result = await self.request_write(rollback_request)

        if result.success:
            self._stats["rolled_back"] += 1

        return RollbackResult(
            success=result.success,
            request_id=request_id,
            tag=tag,
            original_value=write_result.requested_value,
            rollback_value=original_value,
            message="Rollback successful" if result.success else f"Rollback failed: {result.error_message}",
        )

    def _check_rate_limit(self, tag: str) -> bool:
        """Check if write is within rate limit."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=1)

        if tag not in self._write_timestamps:
            self._write_timestamps[tag] = []

        # Remove old timestamps
        self._write_timestamps[tag] = [
            ts for ts in self._write_timestamps[tag]
            if ts > cutoff
        ]

        return len(self._write_timestamps[tag]) < self._max_writes_per_minute

    def _record_write_timestamp(self, tag: str) -> None:
        """Record write timestamp for rate limiting."""
        now = datetime.now(timezone.utc)
        if tag not in self._write_timestamps:
            self._write_timestamps[tag] = []
        self._write_timestamps[tag].append(now)

    async def _log_write_attempt(
        self,
        request: WriteRequest,
        result: WriteResult,
        validation: Optional[ValidationResult],
    ) -> None:
        """Log write attempt to audit logger."""
        if self._audit_logger:
            try:
                self._audit_logger.log_setpoint_change(
                    tag=request.tag,
                    old_value=self._last_values.get(request.tag),
                    new_value=request.value if result.success else None,
                    source="WRITE_CONTROLLER",
                    user_id=request.user_id,
                    reason=request.reason,
                    success=result.success,
                    validation_result=validation.to_dict() if validation else None,
                )
            except Exception as e:
                logger.error(f"Failed to log write attempt: {e}")

    def get_pending_writes(self) -> List[WriteRequest]:
        """Get list of pending write requests."""
        return list(self._pending_writes.values())

    def get_write_history(self, limit: int = 100) -> List[WriteResult]:
        """Get recent write history."""
        return self._write_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get write controller statistics."""
        return {
            **self._stats,
            "pending_writes": len(self._pending_writes),
            "registered_envelopes": len(self._envelopes),
            "tracked_tags": len(self._last_values),
        }
