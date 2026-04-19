# -*- coding: utf-8 -*-
"""
GL-003 UNIFIEDSTEAM - Velocity Limiter for Steam Trap Setpoint Protection

Rate-of-change limiting for steam trap system setpoints to prevent:
- Water hammer from rapid pressure changes
- Thermal shock from rapid temperature transitions
- Equipment stress from rapid load changes

Reference Standards:
    - ASME B31.1 Power Piping
    - ASME PTC 39 Steam Traps
    - NFPA 85 Boiler Systems
    - IEC 61511 Functional Safety

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VelocityLimitStatus(str, Enum):
    ALLOWED = "allowed"
    CLAMPED = "clamped"
    BLOCKED = "blocked"


class SetpointDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    UNCHANGED = "unchanged"


class SafetyMode(str, Enum):
    NORMAL = "normal"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class SteamTrapVelocityLimit:
    """Velocity limit for steam trap parameters."""
    parameter_name: str
    max_rate_increase: Decimal
    max_rate_decrease: Decimal
    unit: str
    min_value: Decimal
    max_value: Decimal
    deadband: Decimal
    startup_factor: Decimal = Decimal("0.5")
    description: str = ""


DEFAULT_STEAM_TRAP_LIMITS: Dict[str, SteamTrapVelocityLimit] = {
    "header_pressure_setpoint": SteamTrapVelocityLimit(
        parameter_name="header_pressure_setpoint",
        max_rate_increase=Decimal("0.5"),
        max_rate_decrease=Decimal("0.8"),
        unit="psig",
        min_value=Decimal("10.0"),
        max_value=Decimal("400.0"),
        deadband=Decimal("0.5"),
        description="Header pressure setpoint per ASME B31.1",
    ),
    "temperature_setpoint": SteamTrapVelocityLimit(
        parameter_name="temperature_setpoint",
        max_rate_increase=Decimal("1.5"),
        max_rate_decrease=Decimal("2.0"),
        unit="°F",
        min_value=Decimal("150.0"),
        max_value=Decimal("500.0"),
        deadband=Decimal("0.5"),
        description="Temperature setpoint for thermal stress prevention",
    ),
    "drain_valve_position": SteamTrapVelocityLimit(
        parameter_name="drain_valve_position",
        max_rate_increase=Decimal("5.0"),
        max_rate_decrease=Decimal("5.0"),
        unit="%",
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("0.5"),
        description="Drain valve position rate",
    ),
    "condensate_flow_setpoint": SteamTrapVelocityLimit(
        parameter_name="condensate_flow_setpoint",
        max_rate_increase=Decimal("3.0"),
        max_rate_decrease=Decimal("5.0"),
        unit="%",
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("0.2"),
        description="Condensate flow setpoint",
    ),
    "superheat_setpoint": SteamTrapVelocityLimit(
        parameter_name="superheat_setpoint",
        max_rate_increase=Decimal("0.5"),
        max_rate_decrease=Decimal("0.5"),
        unit="°F",
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("0.2"),
        description="Superheat setpoint",
    ),
}


class VelocityLimitResult(BaseModel):
    """Result of velocity limiting."""
    parameter_name: str
    current_value: float
    requested_value: float
    allowed_value: float
    status: VelocityLimitStatus
    direction: SetpointDirection
    requested_rate: float
    max_rate: float
    actual_rate: float
    clamped_amount: float = 0.0
    safety_mode: SafetyMode = SafetyMode.NORMAL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.provenance_hash:
            content = (
                f"{self.parameter_name}|{self.current_value}|{self.requested_value}|"
                f"{self.allowed_value}|{self.status.value}|{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


class SteamTrapVelocityLimiter:
    """
    Rate-of-change limiter for steam trap setpoints.

    Implements fail-safe velocity limiting to prevent water hammer,
    thermal shock, and equipment stress.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        custom_limits: Optional[Dict[str, SteamTrapVelocityLimit]] = None,
        default_safety_mode: SafetyMode = SafetyMode.NORMAL,
        precision: int = 3,
    ) -> None:
        self._limits = dict(DEFAULT_STEAM_TRAP_LIMITS)
        if custom_limits:
            self._limits.update(custom_limits)

        self._safety_mode = default_safety_mode
        self._precision = precision
        self._quantize_str = "0." + "0" * precision
        self._lock = threading.Lock()

        self._stats = {
            "total_requests": 0,
            "clamped_requests": 0,
            "allowed_requests": 0,
        }

        logger.info("SteamTrapVelocityLimiter initialized with %d limits", len(self._limits))

    @property
    def safety_mode(self) -> SafetyMode:
        return self._safety_mode

    @safety_mode.setter
    def safety_mode(self, mode: SafetyMode) -> None:
        self._safety_mode = mode

    def _quantize(self, value: Decimal) -> Decimal:
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _get_effective_rate(
        self,
        limit: SteamTrapVelocityLimit,
        direction: SetpointDirection,
    ) -> Decimal:
        if direction == SetpointDirection.INCREASING:
            base_rate = limit.max_rate_increase
        elif direction == SetpointDirection.DECREASING:
            base_rate = limit.max_rate_decrease
        else:
            return Decimal("0")

        factor = {
            SafetyMode.NORMAL: Decimal("1.0"),
            SafetyMode.STARTUP: limit.startup_factor,
            SafetyMode.SHUTDOWN: Decimal("0.7"),
            SafetyMode.EMERGENCY: Decimal("1.5"),
        }[self._safety_mode]

        return self._quantize(base_rate * factor)

    def limit_setpoint_change(
        self,
        parameter: str,
        current_value: float,
        requested_value: float,
        dt_seconds: float = 1.0,
        safety_mode_override: Optional[SafetyMode] = None,
    ) -> VelocityLimitResult:
        """Apply velocity limiting to setpoint change."""
        with self._lock:
            self._stats["total_requests"] += 1

            mode = safety_mode_override or self._safety_mode

            if parameter in self._limits:
                limit = self._limits[parameter]
            else:
                limit = SteamTrapVelocityLimit(
                    parameter_name=parameter,
                    max_rate_increase=Decimal("1.0"),
                    max_rate_decrease=Decimal("1.0"),
                    unit="",
                    min_value=Decimal("-1000000"),
                    max_value=Decimal("1000000"),
                    deadband=Decimal("0.01"),
                )

            current = Decimal(str(current_value))
            requested = Decimal(str(requested_value))
            dt = Decimal(str(max(dt_seconds, 0.001)))

            change = requested - current
            requested_rate = abs(change) / dt

            if abs(change) <= limit.deadband:
                self._stats["allowed_requests"] += 1
                return VelocityLimitResult(
                    parameter_name=parameter,
                    current_value=float(current),
                    requested_value=float(requested),
                    allowed_value=float(requested),
                    status=VelocityLimitStatus.ALLOWED,
                    direction=SetpointDirection.UNCHANGED,
                    requested_rate=float(requested_rate),
                    max_rate=0.0,
                    actual_rate=0.0,
                    safety_mode=mode,
                )

            direction = (
                SetpointDirection.INCREASING if change > 0
                else SetpointDirection.DECREASING
            )

            max_rate = self._get_effective_rate(limit, direction)
            max_change = max_rate * dt

            if requested_rate <= max_rate:
                allowed = requested
                status = VelocityLimitStatus.ALLOWED
                actual_rate = requested_rate
                clamped = Decimal("0")
                self._stats["allowed_requests"] += 1
            else:
                if direction == SetpointDirection.INCREASING:
                    allowed = current + max_change
                else:
                    allowed = current - max_change

                status = VelocityLimitStatus.CLAMPED
                actual_rate = max_rate
                clamped = abs(requested - allowed)
                self._stats["clamped_requests"] += 1

            allowed = max(limit.min_value, min(limit.max_value, allowed))
            allowed = self._quantize(allowed)

            return VelocityLimitResult(
                parameter_name=parameter,
                current_value=float(current),
                requested_value=float(requested),
                allowed_value=float(allowed),
                status=status,
                direction=direction,
                requested_rate=float(self._quantize(requested_rate)),
                max_rate=float(max_rate),
                actual_rate=float(self._quantize(actual_rate)),
                clamped_amount=float(self._quantize(clamped)),
                safety_mode=mode,
            )

    def get_statistics(self) -> Dict[str, Any]:
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "clamp_rate": self._stats["clamped_requests"] / total if total > 0 else 0.0,
            "current_safety_mode": self._safety_mode.value,
        }


__all__ = [
    "VelocityLimitStatus",
    "SetpointDirection",
    "SafetyMode",
    "SteamTrapVelocityLimit",
    "VelocityLimitResult",
    "SteamTrapVelocityLimiter",
    "DEFAULT_STEAM_TRAP_LIMITS",
]
