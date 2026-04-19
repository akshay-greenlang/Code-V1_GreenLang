# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Command Handler Module

OPC-UA command execution with handshaking, watchdog timeout, retry with backoff,
and comprehensive audit logging.

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
import time

logger = logging.getLogger(__name__)


class CommandStatus(str, Enum):
    """Command execution status."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


class HandshakeState(str, Enum):
    """OPC-UA handshake state."""
    IDLE = "idle"
    COMMAND_SENT = "command_sent"
    AWAITING_ACK = "awaiting_ack"
    ACKNOWLEDGED = "acknowledged"
    AWAITING_COMPLETE = "awaiting_complete"
    COMPLETE = "complete"
    ERROR = "error"


class AuditLogEntry(BaseModel):
    """Audit log entry for command execution."""
    entry_id: str = Field(...)
    command_id: str = Field(...)
    action: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    user: str = Field(default="system")
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(...)


class CommandConfig(BaseModel):
    """Command handler configuration."""
    handshake_timeout_seconds: float = Field(default=5.0, ge=0.1)
    execution_timeout_seconds: float = Field(default=30.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0)
    initial_retry_delay_seconds: float = Field(default=1.0, ge=0.1)
    watchdog_interval_seconds: float = Field(default=1.0, ge=0.1)
    enable_audit_logging: bool = Field(default=True)


class CommandRequest(BaseModel):
    """Command request to be executed."""
    command_id: str = Field(...)
    tag_name: str = Field(...)
    value: Any = Field(...)
    command_type: str = Field(default="write")
    priority: int = Field(default=5, ge=1, le=10)
    requester: str = Field(default="system")
    timestamp: datetime = Field(default_factory=datetime.now)


class CommandResponse(BaseModel):
    """Response from command execution."""
    command_id: str = Field(...)
    status: CommandStatus = Field(...)
    handshake_state: HandshakeState = Field(...)
    attempts: int = Field(default=0)
    start_time: datetime = Field(...)
    end_time: Optional[datetime] = Field(default=None)
    duration_ms: float = Field(default=0.0)
    error_message: Optional[str] = Field(default=None)
    provenance_hash: str = Field(...)


class CommandHandler:
    """OPC-UA command handler with handshaking and retry logic."""

    def __init__(self, config: CommandConfig, write_callback: Optional[Callable] = None, alert_callback: Optional[Callable] = None):
        self.config = config
        self._write_callback = write_callback
        self._alert_callback = alert_callback
        self._lock = threading.RLock()
        self._pending_commands: Dict[str, CommandRequest] = {}
        self._command_states: Dict[str, CommandResponse] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._watchdog_active = False
        logger.info("CommandHandler initialized")

    def execute_command(self, request: CommandRequest) -> CommandResponse:
        """Execute a command with handshaking and retry."""
        with self._lock:
            self._pending_commands[request.command_id] = request
            start_time = datetime.now()

            response = CommandResponse(
                command_id=request.command_id,
                status=CommandStatus.PENDING,
                handshake_state=HandshakeState.IDLE,
                start_time=start_time,
                provenance_hash=self._calc_provenance(request, CommandStatus.PENDING)
            )
            self._command_states[request.command_id] = response
            self._log_audit(request.command_id, "command_initiated", {"tag": request.tag_name, "value": request.value})

        # Execute with retries
        for attempt in range(self.config.max_retries + 1):
            result = self._execute_single_attempt(request, attempt)
            if result.status == CommandStatus.COMPLETED:
                return result
            if result.status == CommandStatus.REJECTED:
                return result

            # Wait before retry
            if attempt < self.config.max_retries:
                delay = self.config.initial_retry_delay_seconds * (self.config.retry_backoff_factor ** attempt)
                logger.info(f"Command {request.command_id} retry {attempt + 1} in {delay:.1f}s")
                time.sleep(delay)

        # All retries exhausted
        with self._lock:
            response = self._command_states[request.command_id]
            response.status = CommandStatus.FAILED
            response.error_message = "Max retries exhausted"
            response.end_time = datetime.now()
            response.duration_ms = (response.end_time - response.start_time).total_seconds() * 1000
            self._log_audit(request.command_id, "command_failed", {"reason": "max_retries"})
            return response

    def _execute_single_attempt(self, request: CommandRequest, attempt: int) -> CommandResponse:
        """Execute a single command attempt."""
        with self._lock:
            response = self._command_states[request.command_id]
            response.attempts = attempt + 1
            response.handshake_state = HandshakeState.COMMAND_SENT
            response.status = CommandStatus.SENT

        # Send command via callback
        if self._write_callback:
            try:
                result = self._write_callback(request.tag_name, request.value)
                if not result:
                    with self._lock:
                        response.status = CommandStatus.REJECTED
                        response.error_message = "Write callback returned False"
                        return response
            except Exception as e:
                logger.error(f"Write callback error: {e}")
                with self._lock:
                    response.status = CommandStatus.FAILED
                    response.error_message = str(e)
                    return response

        # Wait for handshake
        with self._lock:
            response.handshake_state = HandshakeState.AWAITING_ACK

        ack_received = self._wait_for_acknowledgment(request.command_id)
        if not ack_received:
            with self._lock:
                response.status = CommandStatus.TIMEOUT
                response.error_message = "Handshake timeout"
                return response

        # Wait for completion
        with self._lock:
            response.handshake_state = HandshakeState.AWAITING_COMPLETE

        completed = self._wait_for_completion(request.command_id)
        if not completed:
            with self._lock:
                response.status = CommandStatus.TIMEOUT
                response.error_message = "Execution timeout"
                return response

        # Success
        with self._lock:
            response.status = CommandStatus.COMPLETED
            response.handshake_state = HandshakeState.COMPLETE
            response.end_time = datetime.now()
            response.duration_ms = (response.end_time - response.start_time).total_seconds() * 1000
            response.provenance_hash = self._calc_provenance(request, CommandStatus.COMPLETED)
            self._log_audit(request.command_id, "command_completed", {"duration_ms": response.duration_ms})

            del self._pending_commands[request.command_id]
            return response

    def acknowledge_command(self, command_id: str) -> bool:
        """Acknowledge a pending command (called by external system)."""
        with self._lock:
            response = self._command_states.get(command_id)
            if response and response.handshake_state == HandshakeState.AWAITING_ACK:
                response.handshake_state = HandshakeState.ACKNOWLEDGED
                self._log_audit(command_id, "command_acknowledged", {})
                return True
            return False

    def complete_command(self, command_id: str, success: bool = True) -> bool:
        """Mark a command as complete (called by external system)."""
        with self._lock:
            response = self._command_states.get(command_id)
            if response and response.handshake_state == HandshakeState.AWAITING_COMPLETE:
                response.handshake_state = HandshakeState.COMPLETE
                response.status = CommandStatus.COMPLETED if success else CommandStatus.FAILED
                return True
            return False

    def get_command_status(self, command_id: str) -> Optional[CommandResponse]:
        """Get status of a command."""
        return self._command_states.get(command_id)

    def get_pending_commands(self) -> List[CommandRequest]:
        """Get all pending commands."""
        with self._lock:
            return list(self._pending_commands.values())

    def get_audit_log(self, command_id: Optional[str] = None, hours: float = 24.0) -> List[AuditLogEntry]:
        """Get audit log entries."""
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            log = [e for e in self._audit_log if e.timestamp > cutoff]
            if command_id:
                log = [e for e in log if e.command_id == command_id]
            return log

    def _wait_for_acknowledgment(self, command_id: str) -> bool:
        """Wait for command acknowledgment."""
        deadline = datetime.now() + timedelta(seconds=self.config.handshake_timeout_seconds)
        while datetime.now() < deadline:
            with self._lock:
                response = self._command_states.get(command_id)
                if response and response.handshake_state == HandshakeState.ACKNOWLEDGED:
                    return True
            time.sleep(self.config.watchdog_interval_seconds)
        return False

    def _wait_for_completion(self, command_id: str) -> bool:
        """Wait for command completion."""
        deadline = datetime.now() + timedelta(seconds=self.config.execution_timeout_seconds)
        while datetime.now() < deadline:
            with self._lock:
                response = self._command_states.get(command_id)
                if response and response.handshake_state == HandshakeState.COMPLETE:
                    return True
            time.sleep(self.config.watchdog_interval_seconds)
        return False

    def _log_audit(self, command_id: str, action: str, details: Dict[str, Any]) -> None:
        """Log an audit entry."""
        if not self.config.enable_audit_logging:
            return

        entry = AuditLogEntry(
            entry_id=hashlib.sha256(f"{command_id}:{action}:{datetime.now()}".encode()).hexdigest()[:16],
            command_id=command_id,
            action=action,
            details=details,
            provenance_hash=hashlib.sha256(f"{command_id}:{action}:{details}".encode()).hexdigest()
        )
        self._audit_log.append(entry)

        # Trim old entries
        cutoff = datetime.now() - timedelta(hours=168)
        self._audit_log = [e for e in self._audit_log if e.timestamp > cutoff]

    def _calc_provenance(self, request: CommandRequest, status: CommandStatus) -> str:
        data = {"cmd": request.command_id, "tag": request.tag_name, "status": status.value, "ts": datetime.now().isoformat()}
        return hashlib.sha256(str(data).encode()).hexdigest()
