"""
ESDInterface - Emergency Shutdown System Interface

This module defines the interface specification for Emergency Shutdown
Systems per IEC 61511. The ESD system is the highest priority safety
system, responsible for bringing the process to a safe state during
emergency conditions.

Key requirements:
- ESD has highest priority over all other control systems
- Response time typically <1 second
- De-energize-to-trip operation
- Manual initiation capability

Reference: IEC 61511-1 Clause 11, API RP 14C

Example:
    >>> from greenlang.safety.esd.esd_interface import ESDInterface
    >>> esd = ESDInterface(system_id="ESD-001")
    >>> esd.initiate_shutdown(level=1)
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, field_validator
from enum import Enum, IntEnum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ESDLevel(IntEnum):
    """Emergency Shutdown levels per API RP 14C / IEC 61511."""

    LEVEL_0 = 0  # Total facility shutdown (most severe)
    LEVEL_1 = 1  # Process area shutdown
    LEVEL_2 = 2  # Unit shutdown
    LEVEL_3 = 3  # Equipment shutdown (least severe)


class ESDState(str, Enum):
    """ESD system states."""

    NORMAL = "normal"  # Normal operation
    PRE_ALARM = "pre_alarm"  # Pre-alarm condition
    SHUTDOWN = "shutdown"  # Shutdown in progress
    SAFE_STATE = "safe_state"  # Safe state achieved
    FAULT = "fault"  # System fault
    BYPASS = "bypass"  # System bypassed
    TEST = "test"  # Test mode


class ESDCommandType(str, Enum):
    """Types of ESD commands."""

    INITIATE = "initiate"  # Initiate shutdown
    RESET = "reset"  # Reset after shutdown
    HOLD = "hold"  # Hold current state
    BYPASS_ENABLE = "bypass_enable"  # Enable bypass
    BYPASS_DISABLE = "bypass_disable"  # Disable bypass
    TEST_INITIATE = "test_initiate"  # Start test
    TEST_COMPLETE = "test_complete"  # Complete test


class ESDCommand(BaseModel):
    """ESD command specification."""

    command_id: str = Field(
        default_factory=lambda: f"CMD-{uuid.uuid4().hex[:8].upper()}",
        description="Command identifier"
    )
    command_type: ESDCommandType = Field(
        ...,
        description="Type of command"
    )
    shutdown_level: Optional[ESDLevel] = Field(
        None,
        description="Shutdown level (for INITIATE commands)"
    )
    source: str = Field(
        ...,
        description="Command source (manual, automatic, external)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Command timestamp"
    )
    operator_id: Optional[str] = Field(
        None,
        description="Operator ID for manual commands"
    )
    reason: str = Field(
        default="",
        description="Reason for command"
    )
    priority: int = Field(
        default=0,
        ge=0,
        description="Command priority (0=highest)"
    )


class ESDStatus(BaseModel):
    """ESD system status report."""

    system_id: str = Field(
        ...,
        description="ESD system identifier"
    )
    state: ESDState = Field(
        ...,
        description="Current system state"
    )
    current_level: Optional[ESDLevel] = Field(
        None,
        description="Current shutdown level (if shutdown)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Status timestamp"
    )
    healthy: bool = Field(
        default=True,
        description="System health status"
    )
    bypassed: bool = Field(
        default=False,
        description="Any bypasses active"
    )
    bypass_count: int = Field(
        default=0,
        description="Number of active bypasses"
    )
    fault_count: int = Field(
        default=0,
        description="Number of active faults"
    )
    last_trip_time: Optional[datetime] = Field(
        None,
        description="Last trip timestamp"
    )
    last_trip_level: Optional[ESDLevel] = Field(
        None,
        description="Last trip level"
    )
    last_trip_source: Optional[str] = Field(
        None,
        description="Source of last trip"
    )
    trip_count_total: int = Field(
        default=0,
        description="Total trip count"
    )
    uptime_hours: float = Field(
        default=0.0,
        description="System uptime (hours)"
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


class ESDInterface:
    """
    Emergency Shutdown System Interface.

    Provides the interface for interacting with ESD systems
    per IEC 61511. This is typically the highest-priority
    safety system in a process facility.

    Features:
    - Multi-level shutdown support
    - Manual and automatic initiation
    - Bypass management
    - Response time tracking
    - Complete audit trail

    The interface follows IEC 61511 principles:
    - ESD commands have highest priority
    - De-energize-to-trip operation
    - Fail-safe design

    Attributes:
        system_id: ESD system identifier
        state: Current system state
        shutdown_handlers: Registered shutdown handlers

    Example:
        >>> esd = ESDInterface(system_id="ESD-001")
        >>> result = esd.initiate_shutdown(ESDLevel.LEVEL_1, source="manual")
    """

    def __init__(
        self,
        system_id: str,
        shutdown_handler: Optional[Callable[[ESDLevel], bool]] = None
    ):
        """
        Initialize ESDInterface.

        Args:
            system_id: ESD system identifier
            shutdown_handler: Callback for shutdown execution
        """
        self.system_id = system_id
        self.shutdown_handler = shutdown_handler or self._default_shutdown_handler
        self.state = ESDState.NORMAL
        self.current_level: Optional[ESDLevel] = None
        self.command_history: List[ESDCommand] = []
        self.trip_count = 0
        self.last_trip_time: Optional[datetime] = None
        self.last_trip_level: Optional[ESDLevel] = None
        self.last_trip_source: Optional[str] = None
        self.active_bypasses: Dict[str, Any] = {}
        self.faults: List[Dict[str, Any]] = []
        self.start_time = datetime.utcnow()

        logger.info(f"ESDInterface initialized: {system_id}")

    def initiate_shutdown(
        self,
        level: ESDLevel,
        source: str = "automatic",
        operator_id: Optional[str] = None,
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Initiate emergency shutdown.

        Args:
            level: Shutdown level (0-3)
            source: Command source
            operator_id: Operator ID (for manual commands)
            reason: Reason for shutdown

        Returns:
            Result dictionary with status

        Raises:
            ValueError: If shutdown cannot be initiated
        """
        initiated_at = datetime.utcnow()

        logger.critical(
            f"ESD INITIATE Level {level.value} by {source}: {reason}"
        )

        # Create command record
        command = ESDCommand(
            command_type=ESDCommandType.INITIATE,
            shutdown_level=level,
            source=source,
            operator_id=operator_id,
            reason=reason,
            priority=0,  # ESD has highest priority
        )
        self.command_history.append(command)

        # Check for higher level shutdown already in progress
        if (self.state == ESDState.SHUTDOWN and
            self.current_level is not None and
            self.current_level.value <= level.value):
            logger.warning(
                f"Higher level shutdown (Level {self.current_level.value}) "
                f"already in progress"
            )
            return {
                "status": "already_active",
                "message": f"Level {self.current_level.value} shutdown active",
                "command_id": command.command_id,
            }

        # Check bypass status
        if self.active_bypasses:
            logger.warning(
                f"ESD has {len(self.active_bypasses)} active bypasses"
            )

        # Update state
        self.state = ESDState.SHUTDOWN
        self.current_level = level

        # Execute shutdown
        try:
            success = self.shutdown_handler(level)

            if success:
                self.state = ESDState.SAFE_STATE
                self.trip_count += 1
                self.last_trip_time = initiated_at
                self.last_trip_level = level
                self.last_trip_source = source

                completed_at = datetime.utcnow()
                duration_ms = (completed_at - initiated_at).total_seconds() * 1000

                logger.critical(
                    f"ESD Level {level.value} COMPLETE in {duration_ms:.0f}ms"
                )

                return {
                    "status": "success",
                    "command_id": command.command_id,
                    "level": level.value,
                    "initiated_at": initiated_at.isoformat(),
                    "completed_at": completed_at.isoformat(),
                    "duration_ms": duration_ms,
                    "trip_count": self.trip_count,
                }

            else:
                self.state = ESDState.FAULT
                logger.error(f"ESD Level {level.value} FAILED to execute")

                return {
                    "status": "failed",
                    "command_id": command.command_id,
                    "level": level.value,
                    "error": "Shutdown execution failed",
                }

        except Exception as e:
            self.state = ESDState.FAULT
            logger.error(f"ESD Level {level.value} ERROR: {e}", exc_info=True)

            return {
                "status": "error",
                "command_id": command.command_id,
                "level": level.value,
                "error": str(e),
            }

    def reset(
        self,
        operator_id: str,
        reason: str = "Normal reset"
    ) -> Dict[str, Any]:
        """
        Reset ESD system after shutdown.

        Args:
            operator_id: Operator authorizing reset
            reason: Reset reason

        Returns:
            Result dictionary
        """
        logger.info(f"ESD RESET requested by {operator_id}: {reason}")

        if self.state not in [ESDState.SAFE_STATE, ESDState.SHUTDOWN]:
            return {
                "status": "invalid_state",
                "message": f"Cannot reset from state: {self.state.value}",
            }

        # Create command record
        command = ESDCommand(
            command_type=ESDCommandType.RESET,
            source="manual",
            operator_id=operator_id,
            reason=reason,
        )
        self.command_history.append(command)

        previous_level = self.current_level

        # Reset state
        self.state = ESDState.NORMAL
        self.current_level = None

        logger.info(f"ESD RESET complete from Level {previous_level}")

        return {
            "status": "success",
            "command_id": command.command_id,
            "previous_level": previous_level.value if previous_level else None,
            "reset_by": operator_id,
        }

    def get_status(self) -> ESDStatus:
        """
        Get current ESD system status.

        Returns:
            ESDStatus object
        """
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600.0

        status = ESDStatus(
            system_id=self.system_id,
            state=self.state,
            current_level=self.current_level,
            healthy=self.state != ESDState.FAULT,
            bypassed=len(self.active_bypasses) > 0,
            bypass_count=len(self.active_bypasses),
            fault_count=len(self.faults),
            last_trip_time=self.last_trip_time,
            last_trip_level=self.last_trip_level,
            last_trip_source=self.last_trip_source,
            trip_count_total=self.trip_count,
            uptime_hours=uptime,
        )

        # Calculate provenance hash
        status.provenance_hash = self._calculate_provenance(status)

        return status

    def register_fault(
        self,
        fault_code: str,
        fault_description: str,
        severity: str = "warning"
    ) -> None:
        """
        Register a system fault.

        Args:
            fault_code: Fault code
            fault_description: Fault description
            severity: Severity (info, warning, critical)
        """
        fault = {
            "fault_code": fault_code,
            "description": fault_description,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.faults.append(fault)

        if severity == "critical":
            self.state = ESDState.FAULT

        logger.warning(f"ESD Fault registered: {fault_code} - {fault_description}")

    def clear_fault(self, fault_code: str) -> bool:
        """
        Clear a registered fault.

        Args:
            fault_code: Fault code to clear

        Returns:
            True if fault was cleared
        """
        initial_count = len(self.faults)
        self.faults = [f for f in self.faults if f["fault_code"] != fault_code]

        if len(self.faults) == 0 and self.state == ESDState.FAULT:
            self.state = ESDState.NORMAL

        return len(self.faults) < initial_count

    def add_bypass(
        self,
        bypass_id: str,
        description: str,
        expires_at: datetime
    ) -> None:
        """Add an active bypass."""
        self.active_bypasses[bypass_id] = {
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        logger.warning(f"ESD Bypass added: {bypass_id}")

    def remove_bypass(self, bypass_id: str) -> bool:
        """Remove an active bypass."""
        if bypass_id in self.active_bypasses:
            del self.active_bypasses[bypass_id]
            logger.info(f"ESD Bypass removed: {bypass_id}")
            return True
        return False

    def get_command_history(
        self,
        limit: int = 100
    ) -> List[ESDCommand]:
        """Get command history."""
        return self.command_history[-limit:]

    def _default_shutdown_handler(self, level: ESDLevel) -> bool:
        """Default shutdown handler (simulation)."""
        logger.info(f"Executing default shutdown handler for Level {level.value}")
        # In production, this would trigger actual shutdown actions
        return True

    def _calculate_provenance(self, status: ESDStatus) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{status.system_id}|"
            f"{status.state.value}|"
            f"{status.trip_count_total}|"
            f"{status.timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
