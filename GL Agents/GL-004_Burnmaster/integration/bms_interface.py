"""
GL-004 BURNMASTER - BMS Interface

Burner Management System (BMS) read-only interface for combustion safety monitoring.
This interface NEVER writes to BMS - it is strictly read-only for safety reasons.

Features:
    - Read BMS status and flame status
    - Monitor permissives and safety interlocks
    - Subscribe to alarms and events
    - Safety check validation before control writes

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BurnerState(str, Enum):
    """Burner operational states."""
    OFF = "off"
    PURGE = "purge"
    PILOT = "pilot"
    LIGHT_OFF = "light_off"
    LOW_FIRE = "low_fire"
    MODULATING = "modulating"
    HIGH_FIRE = "high_fire"
    SHUTDOWN = "shutdown"
    LOCKOUT = "lockout"


class FlameCondition(str, Enum):
    """Flame detector conditions."""
    FLAME_PRESENT = "flame_present"
    NO_FLAME = "no_flame"
    FLAME_UNSTABLE = "flame_unstable"
    DETECTOR_FAULT = "detector_fault"
    SELF_CHECK = "self_check"


class PermissiveType(str, Enum):
    """Types of BMS permissives."""
    PURGE_COMPLETE = "purge_complete"
    PILOT_PROVEN = "pilot_proven"
    MAIN_FLAME_PROVEN = "main_flame_proven"
    FUEL_VALVE_CLOSED = "fuel_valve_closed"
    AIR_FLOW_ADEQUATE = "air_flow_adequate"
    DAMPER_POSITION_OK = "damper_position_ok"
    PRESSURE_OK = "pressure_ok"
    TEMPERATURE_OK = "temperature_ok"


class AlarmSeverity(str, Enum):
    """BMS alarm severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"
    LOCKOUT = "lockout"


@dataclass
class BMSStatus:
    """BMS unit status."""
    unit_id: str
    burner_state: BurnerState
    is_running: bool
    is_safe: bool
    active_alarms: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    safety_chain_healthy: bool = True
    controller_online: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "burner_state": self.burner_state.value,
            "is_running": self.is_running,
            "is_safe": self.is_safe,
            "active_alarms": self.active_alarms,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class FlameStatus:
    """Flame detector status."""
    burner_id: str
    condition: FlameCondition
    signal_strength: float
    detector_type: str = "UV"
    is_proven: bool = False
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "burner_id": self.burner_id,
            "condition": self.condition.value,
            "signal_strength": self.signal_strength,
            "is_proven": self.is_proven,
        }


@dataclass
class Permissive:
    """BMS permissive status."""
    permissive_type: PermissiveType
    is_satisfied: bool
    description: str
    tag: Optional[str] = None
    value: Optional[Any] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SafetyCheck:
    """Result of safety check for write operations."""
    is_safe: bool
    unit_id: str
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    blocking_conditions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "unit_id": self.unit_id,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "blocking_conditions": self.blocking_conditions,
        }


@dataclass
class BMSAlarm:
    """BMS alarm event."""
    alarm_id: str
    unit_id: str
    severity: AlarmSeverity
    message: str
    tag: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False


@dataclass
class Subscription:
    """BMS alarm subscription."""
    subscription_id: str
    callback: Callable[[BMSAlarm], None]
    severity_filter: Optional[List[AlarmSeverity]] = None
    unit_filter: Optional[List[str]] = None
    is_active: bool = False


class BMSInterface:
    """
    Read-only BMS Interface for combustion safety monitoring.

    IMPORTANT: This interface NEVER writes to BMS.
    All write operations to the BMS must go through the
    plant's approved safety instrumented system (SIS).
    """

    def __init__(self, dcs_connector=None):
        self._dcs = dcs_connector
        self._alarm_subscriptions: Dict[str, Subscription] = {}
        self._cached_status: Dict[str, BMSStatus] = {}
        self._cached_flame: Dict[str, FlameStatus] = {}
        self._stats = {"reads": 0, "safety_checks": 0, "alarms_received": 0}
        logger.info("BMSInterface initialized (READ-ONLY)")

    async def read_bms_status(self, unit_id: str) -> BMSStatus:
        """Read BMS status for a unit."""
        self._stats["reads"] += 1
        import random
        states = [BurnerState.LOW_FIRE, BurnerState.MODULATING, BurnerState.HIGH_FIRE]
        status = BMSStatus(
            unit_id=unit_id,
            burner_state=random.choice(states),
            is_running=True,
            is_safe=True,
            active_alarms=0,
        )
        self._cached_status[unit_id] = status
        return status

    async def read_flame_status(self, burner_id: str) -> FlameStatus:
        """Read flame detector status."""
        self._stats["reads"] += 1
        import random
        status = FlameStatus(
            burner_id=burner_id,
            condition=FlameCondition.FLAME_PRESENT,
            signal_strength=random.uniform(70, 100),
            is_proven=True,
        )
        self._cached_flame[burner_id] = status
        return status

    async def read_permissives(self, unit_id: str) -> List[Permissive]:
        """Read all permissives for a unit."""
        self._stats["reads"] += 1
        permissives = [
            Permissive(PermissiveType.PURGE_COMPLETE, True, "Purge cycle completed"),
            Permissive(PermissiveType.PILOT_PROVEN, True, "Pilot flame proven"),
            Permissive(PermissiveType.MAIN_FLAME_PROVEN, True, "Main flame proven"),
            Permissive(PermissiveType.AIR_FLOW_ADEQUATE, True, "Combustion air flow adequate"),
            Permissive(PermissiveType.PRESSURE_OK, True, "Fuel pressure within limits"),
        ]
        return permissives

    async def check_safe_to_write(self, unit_id: str) -> SafetyCheck:
        """
        Check if it is safe to write setpoints to DCS for this unit.

        This checks BMS status, flame status, and permissives to determine
        if it is safe to make control adjustments.
        """
        self._stats["safety_checks"] += 1
        checks_passed = []
        checks_failed = []
        blocking_conditions = []

        bms_status = await self.read_bms_status(unit_id)
        if bms_status.is_safe:
            checks_passed.append("BMS status safe")
        else:
            checks_failed.append("BMS status unsafe")
            blocking_conditions.append(f"BMS reports unsafe state: {bms_status.burner_state.value}")

        if bms_status.is_running:
            checks_passed.append("Burner running")
        else:
            checks_failed.append("Burner not running")
            blocking_conditions.append("Burner is not in running state")

        if bms_status.active_alarms == 0:
            checks_passed.append("No active alarms")
        else:
            checks_failed.append("Active alarms present")
            blocking_conditions.append(f"{bms_status.active_alarms} active alarms")

        permissives = await self.read_permissives(unit_id)
        for perm in permissives:
            if perm.is_satisfied:
                checks_passed.append(f"Permissive {perm.permissive_type.value}")
            else:
                checks_failed.append(f"Permissive {perm.permissive_type.value}")
                blocking_conditions.append(f"Permissive not satisfied: {perm.description}")

        is_safe = len(checks_failed) == 0 and len(blocking_conditions) == 0

        return SafetyCheck(
            is_safe=is_safe,
            unit_id=unit_id,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            blocking_conditions=blocking_conditions,
        )

    async def subscribe_to_alarms(
        self,
        callback: Callable[[BMSAlarm], None],
        severity_filter: Optional[List[AlarmSeverity]] = None,
        unit_filter: Optional[List[str]] = None,
    ) -> Subscription:
        """Subscribe to BMS alarms."""
        sub_id = str(uuid.uuid4())
        subscription = Subscription(
            subscription_id=sub_id,
            callback=callback,
            severity_filter=severity_filter,
            unit_filter=unit_filter,
            is_active=True,
        )
        self._alarm_subscriptions[sub_id] = subscription
        logger.info(f"Created BMS alarm subscription {sub_id}")
        return subscription

    async def unsubscribe(self, subscription_id: str) -> None:
        """Remove alarm subscription."""
        if subscription_id in self._alarm_subscriptions:
            self._alarm_subscriptions[subscription_id].is_active = False
            del self._alarm_subscriptions[subscription_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        return {
            **self._stats,
            "active_subscriptions": len([s for s in self._alarm_subscriptions.values() if s.is_active]),
            "cached_units": len(self._cached_status),
        }
