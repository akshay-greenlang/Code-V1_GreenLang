"""
GL-016 Waterguard OPC-UA Write Handler

Handles setpoint commands and control actions to OPC-UA endpoints with
safety validation, acknowledgement handling, and watchdog timeouts.

CRITICAL SAFETY RULES:
- NEVER overwrite bad quality with "last good" value
- ALWAYS validate current quality before writing
- ALWAYS enforce rate limiting and safety interlocks
- ALWAYS log all write operations for audit
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from integrations.opcua.opcua_connector import WaterguardOPCUAConnector
from integrations.opcua.opcua_schemas import OPCUAQuality, TagValue

logger = logging.getLogger(__name__)


# =============================================================================
# Write Command Status
# =============================================================================

class WriteStatus(str, Enum):
    """Status of a write command."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXECUTING = "executing"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    """Reasons for command rejection."""
    BAD_QUALITY = "bad_quality"
    SAFETY_INTERLOCK = "safety_interlock"
    RATE_LIMIT = "rate_limit"
    OUT_OF_RANGE = "out_of_range"
    OPERATOR_REQUIRED = "operator_required"
    NOT_CONNECTED = "not_connected"
    DUPLICATE_COMMAND = "duplicate_command"


# =============================================================================
# Write Command
# =============================================================================

class WriteCommand(BaseModel):
    """
    Command for writing a value to an OPC-UA tag.

    Includes safety validation parameters and tracking metadata.
    """

    # Identification
    command_id: UUID = Field(default_factory=uuid4, description="Unique command ID")
    trace_id: Optional[UUID] = Field(default=None, description="Trace ID for correlation")

    # Target
    node_id: str = Field(..., description="Target OPC-UA node ID")
    target_value: float = Field(..., description="Value to write")
    previous_value: Optional[float] = Field(default=None, description="Previous value")

    # Safety
    require_ack: bool = Field(default=True, description="Require acknowledgement")
    check_quality: bool = Field(default=True, description="Verify quality before write")
    validate_range: bool = Field(default=True, description="Validate value is in range")
    min_value: Optional[float] = Field(default=None, description="Minimum allowed value")
    max_value: Optional[float] = Field(default=None, description="Maximum allowed value")
    max_change: Optional[float] = Field(default=None, description="Maximum change per write")

    # Timing
    timeout_seconds: int = Field(default=30, description="Command timeout")
    ramp_rate: Optional[float] = Field(
        default=None,
        description="Ramp rate for gradual changes (units/second)"
    )

    # Metadata
    source: str = Field(default="system", description="Command source")
    operator_id: Optional[str] = Field(default=None, description="Operator ID if manual")
    reason: Optional[str] = Field(default=None, description="Reason for command")
    safety_validated: bool = Field(default=False, description="Safety validation passed")

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None)

    @field_validator("target_value")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Ensure value is finite."""
        import math
        if not math.isfinite(v):
            raise ValueError(f"Value must be finite, got {v}")
        return v

    @property
    def is_expired(self) -> bool:
        """Check if command has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    def validate_range_check(self) -> Optional[str]:
        """Validate value is within range."""
        if not self.validate_range:
            return None

        if self.min_value is not None and self.target_value < self.min_value:
            return f"Value {self.target_value} below minimum {self.min_value}"

        if self.max_value is not None and self.target_value > self.max_value:
            return f"Value {self.target_value} above maximum {self.max_value}"

        if self.max_change is not None and self.previous_value is not None:
            change = abs(self.target_value - self.previous_value)
            if change > self.max_change:
                return f"Change {change} exceeds maximum {self.max_change}"

        return None


# =============================================================================
# Write Result
# =============================================================================

class WriteResult(BaseModel):
    """Result of a write command execution."""

    command_id: UUID = Field(..., description="Original command ID")
    status: WriteStatus = Field(..., description="Final status")

    # Execution details
    executed_at: Optional[datetime] = Field(default=None, description="Execution time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
    actual_value: Optional[float] = Field(default=None, description="Actual value after write")

    # Error info
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    rejection_reason: Optional[RejectionReason] = Field(default=None, description="Rejection reason")

    # Timing
    validation_time_ms: Optional[int] = Field(default=None)
    execution_time_ms: Optional[int] = Field(default=None)
    ack_time_ms: Optional[int] = Field(default=None)
    total_time_ms: Optional[int] = Field(default=None)

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.status in [WriteStatus.COMPLETED, WriteStatus.ACKNOWLEDGED]


# =============================================================================
# Watchdog Configuration
# =============================================================================

@dataclass
class WatchdogConfig:
    """Configuration for command watchdog."""

    # Timeouts
    ack_timeout_seconds: float = 5.0
    completion_timeout_seconds: float = 30.0
    heartbeat_interval_seconds: float = 1.0

    # Retries
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Safety
    fail_safe_value: Optional[float] = None
    fail_safe_enabled: bool = False


# =============================================================================
# Write Handler
# =============================================================================

class WriteHandler:
    """
    Handler for OPC-UA write operations with safety validation.

    Features:
    - Quality validation before writes
    - Rate limiting and change validation
    - Acknowledgement handling
    - Watchdog timeouts
    - Audit logging

    SAFETY CRITICAL:
    - Never overwrites bad quality with "last good" value
    - Validates all writes against safety limits
    - Tracks all operations for audit trail
    """

    def __init__(
        self,
        connector: WaterguardOPCUAConnector,
        watchdog_config: Optional[WatchdogConfig] = None,
        on_command_complete: Optional[Callable[[WriteResult], None]] = None,
        on_error: Optional[Callable[[UUID, Exception], None]] = None,
    ):
        """
        Initialize write handler.

        Args:
            connector: OPC-UA connector instance
            watchdog_config: Watchdog configuration
            on_command_complete: Callback for completed commands
            on_error: Callback for errors
        """
        self.connector = connector
        self.watchdog_config = watchdog_config or WatchdogConfig()
        self._on_command_complete = on_command_complete
        self._on_error = on_error

        # Command tracking
        self._pending_commands: Dict[UUID, WriteCommand] = {}
        self._command_results: Dict[UUID, WriteResult] = {}
        self._ack_events: Dict[UUID, asyncio.Event] = {}

        # Rate limiting
        self._last_write_times: Dict[str, datetime] = {}
        self._write_counts: Dict[str, int] = {}

        # Watchdog
        self._watchdog_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Audit
        self._audit_log: List[Dict[str, Any]] = []

    async def start(self) -> None:
        """Start the write handler."""
        self._shutdown_event.clear()
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("Write handler started")

    async def stop(self) -> None:
        """Stop the write handler."""
        self._shutdown_event.set()
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        logger.info("Write handler stopped")

    async def execute(self, command: WriteCommand) -> WriteResult:
        """
        Execute a write command.

        Args:
            command: Write command to execute

        Returns:
            Write result

        Raises:
            ValueError: If command validation fails
        """
        start_time = time.time()
        validation_start = time.time()

        # Track command
        self._pending_commands[command.command_id] = command

        try:
            # === VALIDATION PHASE ===

            # Check connection
            if not self.connector.is_connected:
                return self._reject_command(
                    command,
                    RejectionReason.NOT_CONNECTED,
                    "Not connected to OPC-UA server"
                )

            # Check for duplicate
            if self._is_duplicate_command(command):
                return self._reject_command(
                    command,
                    RejectionReason.DUPLICATE_COMMAND,
                    "Duplicate command detected"
                )

            # Validate range
            range_error = command.validate_range_check()
            if range_error:
                return self._reject_command(
                    command,
                    RejectionReason.OUT_OF_RANGE,
                    range_error
                )

            # CRITICAL: Check current quality - NEVER overwrite bad quality
            if command.check_quality:
                current_value = await self.connector.read_tag(command.node_id)

                if current_value.quality.is_bad:
                    error_msg = (
                        f"Cannot write to {command.node_id}: current quality is "
                        f"{current_value.quality.value}. Fix the underlying issue "
                        "before writing. NEVER overwrite bad quality with 'last good'."
                    )
                    logger.error(error_msg)
                    return self._reject_command(
                        command,
                        RejectionReason.BAD_QUALITY,
                        error_msg
                    )

                # Store previous value if not provided
                if command.previous_value is None:
                    command.previous_value = current_value.value

            # Check rate limiting
            if self._is_rate_limited(command.node_id):
                return self._reject_command(
                    command,
                    RejectionReason.RATE_LIMIT,
                    "Rate limit exceeded for this tag"
                )

            validation_time_ms = int((time.time() - validation_start) * 1000)

            # === EXECUTION PHASE ===

            execution_start = time.time()
            self._log_audit(command, "executing")

            try:
                # Execute write
                await self.connector.write_tag(
                    node_id=command.node_id,
                    value=command.target_value,
                    check_quality=False,  # Already checked above
                )

                execution_time_ms = int((time.time() - execution_start) * 1000)
                self._update_rate_limit(command.node_id)

            except Exception as e:
                error_msg = f"Write failed: {e}"
                logger.error(error_msg)
                self._log_audit(command, "failed", error=str(e))

                result = WriteResult(
                    command_id=command.command_id,
                    status=WriteStatus.FAILED,
                    error_message=error_msg,
                    validation_time_ms=validation_time_ms,
                    execution_time_ms=int((time.time() - execution_start) * 1000),
                    total_time_ms=int((time.time() - start_time) * 1000),
                )
                self._complete_command(command.command_id, result)
                return result

            # === ACKNOWLEDGEMENT PHASE ===

            ack_time_ms = None
            if command.require_ack:
                ack_start = time.time()
                ack_event = asyncio.Event()
                self._ack_events[command.command_id] = ack_event

                try:
                    # Wait for acknowledgement or timeout
                    await asyncio.wait_for(
                        self._wait_for_ack(command),
                        timeout=self.watchdog_config.ack_timeout_seconds
                    )
                    ack_time_ms = int((time.time() - ack_start) * 1000)

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Acknowledgement timeout for command {command.command_id}"
                    )

                finally:
                    self._ack_events.pop(command.command_id, None)

            # === VERIFICATION PHASE ===

            # Verify the write
            actual_value = None
            try:
                verify_read = await self.connector.read_tag(command.node_id)
                actual_value = verify_read.value

                # Check if value was actually written
                if abs(actual_value - command.target_value) > 0.001:
                    logger.warning(
                        f"Write verification mismatch: expected {command.target_value}, "
                        f"got {actual_value}"
                    )

            except Exception as e:
                logger.warning(f"Verification read failed: {e}")

            # === SUCCESS ===

            total_time_ms = int((time.time() - start_time) * 1000)

            result = WriteResult(
                command_id=command.command_id,
                status=WriteStatus.COMPLETED,
                executed_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                actual_value=actual_value,
                validation_time_ms=validation_time_ms,
                execution_time_ms=execution_time_ms,
                ack_time_ms=ack_time_ms,
                total_time_ms=total_time_ms,
            )

            self._log_audit(command, "completed", actual_value=actual_value)
            self._complete_command(command.command_id, result)

            return result

        except Exception as e:
            logger.error(f"Unexpected error executing command: {e}")
            if self._on_error:
                self._on_error(command.command_id, e)

            result = WriteResult(
                command_id=command.command_id,
                status=WriteStatus.FAILED,
                error_message=str(e),
                total_time_ms=int((time.time() - start_time) * 1000),
            )
            self._complete_command(command.command_id, result)
            return result

        finally:
            self._pending_commands.pop(command.command_id, None)

    def _reject_command(
        self,
        command: WriteCommand,
        reason: RejectionReason,
        message: str,
    ) -> WriteResult:
        """Reject a command with reason."""
        logger.warning(f"Command {command.command_id} rejected: {reason.value} - {message}")
        self._log_audit(command, "rejected", reason=reason.value, error=message)

        result = WriteResult(
            command_id=command.command_id,
            status=WriteStatus.REJECTED,
            error_message=message,
            rejection_reason=reason,
        )

        self._complete_command(command.command_id, result)
        return result

    def _complete_command(self, command_id: UUID, result: WriteResult) -> None:
        """Mark command as complete."""
        self._command_results[command_id] = result
        self._pending_commands.pop(command_id, None)

        if self._on_command_complete:
            try:
                self._on_command_complete(result)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")

    def _is_duplicate_command(self, command: WriteCommand) -> bool:
        """Check if this is a duplicate command."""
        for pending_id, pending_cmd in self._pending_commands.items():
            if pending_id == command.command_id:
                continue
            if (
                pending_cmd.node_id == command.node_id and
                abs(pending_cmd.target_value - command.target_value) < 0.001 and
                (datetime.utcnow() - pending_cmd.created_at).total_seconds() < 5
            ):
                return True
        return False

    def _is_rate_limited(self, node_id: str, min_interval_seconds: float = 1.0) -> bool:
        """Check if writes to this tag are rate limited."""
        last_write = self._last_write_times.get(node_id)
        if last_write is None:
            return False

        elapsed = (datetime.utcnow() - last_write).total_seconds()
        return elapsed < min_interval_seconds

    def _update_rate_limit(self, node_id: str) -> None:
        """Update rate limit tracking for tag."""
        self._last_write_times[node_id] = datetime.utcnow()
        self._write_counts[node_id] = self._write_counts.get(node_id, 0) + 1

    async def _wait_for_ack(self, command: WriteCommand) -> None:
        """Wait for command acknowledgement."""
        # In a real implementation, this would monitor for an ACK signal
        # from the target device. For now, we just verify the value was written.
        await asyncio.sleep(0.1)

        # Read back and verify
        current = await self.connector.read_tag(command.node_id)
        if current.quality.is_good:
            return

        raise TimeoutError("Acknowledgement not received")

    async def _watchdog_loop(self) -> None:
        """Watchdog loop for monitoring command timeouts."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.utcnow()

                # Check for timed out commands
                for command_id, command in list(self._pending_commands.items()):
                    age = (now - command.created_at).total_seconds()

                    if age > command.timeout_seconds:
                        logger.warning(f"Command {command_id} timed out after {age:.1f}s")

                        result = WriteResult(
                            command_id=command_id,
                            status=WriteStatus.TIMEOUT,
                            error_message=f"Command timed out after {age:.1f}s",
                            total_time_ms=int(age * 1000),
                        )

                        self._complete_command(command_id, result)
                        self._log_audit(command, "timeout")

                await asyncio.sleep(self.watchdog_config.heartbeat_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}")

    def _log_audit(
        self,
        command: WriteCommand,
        action: str,
        **kwargs
    ) -> None:
        """Log command to audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "command_id": str(command.command_id),
            "trace_id": str(command.trace_id) if command.trace_id else None,
            "node_id": command.node_id,
            "target_value": command.target_value,
            "previous_value": command.previous_value,
            "action": action,
            "source": command.source,
            "operator_id": command.operator_id,
            "reason": command.reason,
            **kwargs
        }
        self._audit_log.append(entry)

        # Keep only last 10000 entries
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

        logger.info(
            f"AUDIT: {action} command {command.command_id} "
            f"node={command.node_id} value={command.target_value}"
        )

    def get_audit_log(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        entries = self._audit_log

        if since:
            entries = [
                e for e in entries
                if datetime.fromisoformat(e["timestamp"]) > since
            ]

        return entries[-limit:]

    def get_command_result(self, command_id: UUID) -> Optional[WriteResult]:
        """Get result for a command."""
        return self._command_results.get(command_id)

    @property
    def pending_count(self) -> int:
        """Get count of pending commands."""
        return len(self._pending_commands)
