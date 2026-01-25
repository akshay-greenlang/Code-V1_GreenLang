"""
PriorityManager - ESD > DCS > Agent Priority Management

This module implements command priority management ensuring that
Emergency Shutdown (ESD) commands always have highest priority
over DCS (Distributed Control System) and Agent commands.

Priority hierarchy:
1. ESD (Emergency Shutdown) - Highest
2. SIS (Safety Instrumented System)
3. DCS (Distributed Control System)
4. APC (Advanced Process Control)
5. Agent (AI/Optimization Agents)
6. Manual - Lowest

Reference: IEC 61511-1 Clause 11.3, ISA 84

Example:
    >>> from greenlang.safety.esd.priority_manager import PriorityManager
    >>> manager = PriorityManager()
    >>> result = manager.evaluate_command(command)
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum, IntEnum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class CommandPriority(IntEnum):
    """Command priority levels (lower = higher priority)."""

    ESD = 0  # Emergency Shutdown - Highest priority
    SIS = 10  # Safety Instrumented System
    MANUAL_EMERGENCY = 20  # Manual emergency actions
    DCS_INTERLOCK = 30  # DCS interlocks
    DCS_CONTROL = 40  # DCS control actions
    APC = 50  # Advanced Process Control
    AGENT = 60  # AI/Agent commands
    MANUAL_NORMAL = 70  # Normal manual operations
    ADVISORY = 80  # Advisory only


class CommandSource(str, Enum):
    """Source system for commands."""

    ESD_AUTOMATIC = "esd_automatic"
    ESD_MANUAL = "esd_manual"
    SIS = "sis"
    DCS_INTERLOCK = "dcs_interlock"
    DCS_PID = "dcs_pid"
    DCS_SEQUENCE = "dcs_sequence"
    APC = "apc"
    AGENT_OPTIMIZATION = "agent_optimization"
    AGENT_SAFETY = "agent_safety"
    OPERATOR = "operator"
    EXTERNAL = "external"


class CommandDecision(str, Enum):
    """Decision outcome for command evaluation."""

    ALLOW = "allow"  # Command allowed
    BLOCK = "block"  # Command blocked
    DEFER = "defer"  # Command deferred
    OVERRIDE = "override"  # Command overrides previous


class Command(BaseModel):
    """Command specification for priority evaluation."""

    command_id: str = Field(
        default_factory=lambda: f"CMD-{uuid.uuid4().hex[:8].upper()}",
        description="Command identifier"
    )
    source: CommandSource = Field(
        ...,
        description="Command source"
    )
    target_equipment: str = Field(
        ...,
        description="Target equipment tag"
    )
    action: str = Field(
        ...,
        description="Requested action"
    )
    value: Optional[Any] = Field(
        None,
        description="Command value"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Command timestamp"
    )
    expiry_ms: Optional[float] = Field(
        None,
        description="Command expiry time (ms)"
    )
    is_safety_critical: bool = Field(
        default=False,
        description="Is this a safety-critical command"
    )


class PriorityResult(BaseModel):
    """Result of priority evaluation."""

    command_id: str = Field(
        ...,
        description="Evaluated command ID"
    )
    decision: CommandDecision = Field(
        ...,
        description="Decision outcome"
    )
    assigned_priority: CommandPriority = Field(
        ...,
        description="Assigned priority level"
    )
    blocking_command_id: Optional[str] = Field(
        None,
        description="ID of blocking command (if blocked)"
    )
    blocking_reason: Optional[str] = Field(
        None,
        description="Reason for blocking"
    )
    effective_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When decision becomes effective"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="When decision expires"
    )
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Decision audit trail"
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


class PriorityManager:
    """
    Command Priority Manager.

    Ensures proper command priority hierarchy with ESD having
    absolute highest priority. Prevents lower-priority systems
    from interfering with safety actions.

    Priority rules:
    1. ESD commands ALWAYS execute (cannot be blocked)
    2. SIS commands blocked only by ESD
    3. DCS commands blocked by ESD and SIS
    4. Agent commands blocked by all safety systems

    The manager follows IEC 61511 principles:
    - Safety commands have priority
    - Clear audit trail
    - Deterministic behavior

    Attributes:
        active_commands: Currently active commands by equipment
        priority_map: Mapping of sources to priorities

    Example:
        >>> manager = PriorityManager()
        >>> result = manager.evaluate_command(command)
        >>> if result.decision == CommandDecision.ALLOW:
        ...     execute_command(command)
    """

    # Source to priority mapping
    PRIORITY_MAP: Dict[CommandSource, CommandPriority] = {
        CommandSource.ESD_AUTOMATIC: CommandPriority.ESD,
        CommandSource.ESD_MANUAL: CommandPriority.ESD,
        CommandSource.SIS: CommandPriority.SIS,
        CommandSource.DCS_INTERLOCK: CommandPriority.DCS_INTERLOCK,
        CommandSource.DCS_PID: CommandPriority.DCS_CONTROL,
        CommandSource.DCS_SEQUENCE: CommandPriority.DCS_CONTROL,
        CommandSource.APC: CommandPriority.APC,
        CommandSource.AGENT_OPTIMIZATION: CommandPriority.AGENT,
        CommandSource.AGENT_SAFETY: CommandPriority.AGENT,  # Still lower than real safety
        CommandSource.OPERATOR: CommandPriority.MANUAL_NORMAL,
        CommandSource.EXTERNAL: CommandPriority.ADVISORY,
    }

    def __init__(self):
        """Initialize PriorityManager."""
        self.active_commands: Dict[str, List[Command]] = {}
        self.evaluation_history: List[PriorityResult] = []
        logger.info("PriorityManager initialized")

    def evaluate_command(
        self,
        command: Command
    ) -> PriorityResult:
        """
        Evaluate a command against current active commands.

        Args:
            command: Command to evaluate

        Returns:
            PriorityResult with decision
        """
        logger.debug(
            f"Evaluating command {command.command_id} from {command.source.value} "
            f"for {command.target_equipment}"
        )

        # Determine priority
        priority = self._get_priority(command)

        # ESD commands are NEVER blocked
        if priority == CommandPriority.ESD:
            result = self._allow_command(command, priority)
            result.audit_trail.append({
                "action": "esd_priority_override",
                "message": "ESD command has absolute priority",
                "timestamp": datetime.utcnow().isoformat(),
            })
            logger.warning(
                f"ESD command {command.command_id} ALLOWED - absolute priority"
            )
            return result

        # Check for higher priority active commands
        active = self.active_commands.get(command.target_equipment, [])
        blocking_cmd = None

        for active_cmd in active:
            active_priority = self._get_priority(active_cmd)
            if active_priority < priority:  # Lower number = higher priority
                blocking_cmd = active_cmd
                break

        if blocking_cmd:
            result = self._block_command(
                command,
                priority,
                blocking_cmd,
                f"Blocked by higher priority {blocking_cmd.source.value} command"
            )
            logger.info(
                f"Command {command.command_id} BLOCKED by {blocking_cmd.command_id}"
            )
            return result

        # No blocking - allow command
        result = self._allow_command(command, priority)
        logger.debug(f"Command {command.command_id} ALLOWED")
        return result

    def register_active_command(
        self,
        command: Command
    ) -> None:
        """
        Register a command as active.

        Args:
            command: Command to register
        """
        equipment = command.target_equipment
        if equipment not in self.active_commands:
            self.active_commands[equipment] = []

        self.active_commands[equipment].append(command)

        logger.debug(
            f"Registered active command {command.command_id} "
            f"for {equipment}"
        )

    def deregister_command(
        self,
        command_id: str,
        equipment: Optional[str] = None
    ) -> bool:
        """
        Remove a command from active list.

        Args:
            command_id: Command to remove
            equipment: Equipment (if known)

        Returns:
            True if command was found and removed
        """
        if equipment:
            equipments = [equipment]
        else:
            equipments = list(self.active_commands.keys())

        for eq in equipments:
            if eq in self.active_commands:
                initial = len(self.active_commands[eq])
                self.active_commands[eq] = [
                    c for c in self.active_commands[eq]
                    if c.command_id != command_id
                ]
                if len(self.active_commands[eq]) < initial:
                    logger.debug(f"Deregistered command {command_id}")
                    return True

        return False

    def clear_expired_commands(self) -> int:
        """
        Clear expired commands from all equipment.

        Returns:
            Number of commands cleared
        """
        now = datetime.utcnow()
        cleared = 0

        for equipment in list(self.active_commands.keys()):
            initial = len(self.active_commands[equipment])

            self.active_commands[equipment] = [
                c for c in self.active_commands[equipment]
                if not self._is_expired(c, now)
            ]

            cleared += initial - len(self.active_commands[equipment])

            if not self.active_commands[equipment]:
                del self.active_commands[equipment]

        if cleared > 0:
            logger.debug(f"Cleared {cleared} expired commands")

        return cleared

    def can_agent_override(
        self,
        equipment: str,
        safety_context: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if an agent can override current commands for equipment.

        Agents can typically only override other agent commands and
        some APC commands. They CANNOT override ESD, SIS, or DCS interlocks.

        Args:
            equipment: Equipment to check
            safety_context: Is agent operating in safety context

        Returns:
            Tuple of (can_override, reason)
        """
        active = self.active_commands.get(equipment, [])

        if not active:
            return True, "No active commands"

        for cmd in active:
            priority = self._get_priority(cmd)

            # Agents cannot override safety systems
            if priority <= CommandPriority.DCS_INTERLOCK:
                return False, f"Cannot override {cmd.source.value} (safety priority)"

            # Agents cannot override DCS control unless in maintenance
            if priority <= CommandPriority.DCS_CONTROL:
                return False, f"Cannot override DCS control ({cmd.source.value})"

        return True, "Can override lower priority commands"

    def get_command_hierarchy(
        self,
        equipment: str
    ) -> List[Dict[str, Any]]:
        """
        Get current command hierarchy for equipment.

        Args:
            equipment: Equipment tag

        Returns:
            List of active commands sorted by priority
        """
        active = self.active_commands.get(equipment, [])

        hierarchy = []
        for cmd in active:
            priority = self._get_priority(cmd)
            hierarchy.append({
                "command_id": cmd.command_id,
                "source": cmd.source.value,
                "priority": priority.value,
                "priority_name": priority.name,
                "action": cmd.action,
                "timestamp": cmd.timestamp.isoformat(),
            })

        # Sort by priority (lowest number = highest priority)
        hierarchy.sort(key=lambda x: x["priority"])

        return hierarchy

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get priority manager statistics.

        Returns:
            Statistics dictionary
        """
        total_active = sum(
            len(cmds) for cmds in self.active_commands.values()
        )

        by_source = {}
        for cmds in self.active_commands.values():
            for cmd in cmds:
                source = cmd.source.value
                by_source[source] = by_source.get(source, 0) + 1

        decisions = {}
        for result in self.evaluation_history[-1000:]:
            dec = result.decision.value
            decisions[dec] = decisions.get(dec, 0) + 1

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_active_commands": total_active,
            "equipment_count": len(self.active_commands),
            "commands_by_source": by_source,
            "recent_decisions": decisions,
            "evaluations_total": len(self.evaluation_history),
        }

    def _get_priority(self, command: Command) -> CommandPriority:
        """Get priority for a command."""
        return self.PRIORITY_MAP.get(
            command.source,
            CommandPriority.ADVISORY
        )

    def _is_expired(self, command: Command, now: datetime) -> bool:
        """Check if command is expired."""
        if command.expiry_ms is None:
            return False

        from datetime import timedelta
        expiry_time = command.timestamp + timedelta(milliseconds=command.expiry_ms)
        return now > expiry_time

    def _allow_command(
        self,
        command: Command,
        priority: CommandPriority
    ) -> PriorityResult:
        """Create allow result."""
        result = PriorityResult(
            command_id=command.command_id,
            decision=CommandDecision.ALLOW,
            assigned_priority=priority,
            audit_trail=[{
                "action": "evaluated",
                "decision": "allow",
                "priority": priority.name,
                "timestamp": datetime.utcnow().isoformat(),
            }]
        )

        result.provenance_hash = self._calculate_provenance(result)
        self.evaluation_history.append(result)

        return result

    def _block_command(
        self,
        command: Command,
        priority: CommandPriority,
        blocking_cmd: Command,
        reason: str
    ) -> PriorityResult:
        """Create block result."""
        result = PriorityResult(
            command_id=command.command_id,
            decision=CommandDecision.BLOCK,
            assigned_priority=priority,
            blocking_command_id=blocking_cmd.command_id,
            blocking_reason=reason,
            audit_trail=[{
                "action": "evaluated",
                "decision": "block",
                "priority": priority.name,
                "blocking_source": blocking_cmd.source.value,
                "blocking_priority": self._get_priority(blocking_cmd).name,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }]
        )

        result.provenance_hash = self._calculate_provenance(result)
        self.evaluation_history.append(result)

        return result

    def _calculate_provenance(self, result: PriorityResult) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.command_id}|"
            f"{result.decision.value}|"
            f"{result.assigned_priority.name}|"
            f"{result.effective_at.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
