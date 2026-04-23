# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE - Velocity Limiter for Combustion Analytics Setpoints

Rate-of-change limiting for combustion sensor and analytics setpoints.

Reference: NFPA 85, NFPA 86, IEC 61511
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations
import hashlib, logging, threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, Optional
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
class CombusenseVelocityLimit:
    parameter_name: str
    max_rate_increase: Decimal
    max_rate_decrease: Decimal
    unit: str
    min_value: Decimal
    max_value: Decimal
    deadband: Decimal
    startup_factor: Decimal = Decimal("0.5")
    description: str = ""

DEFAULT_COMBUSENSE_LIMITS: Dict[str, CombusenseVelocityLimit] = {
    "o2_setpoint": CombusenseVelocityLimit(
        parameter_name="o2_setpoint",
        max_rate_increase=Decimal("0.3"),
        max_rate_decrease=Decimal("0.3"),
        unit="%O2",
        min_value=Decimal("1.0"),
        max_value=Decimal("15.0"),
        deadband=Decimal("0.05"),
        description="O2 setpoint per NFPA 85",
    ),
    "co_alarm_threshold": CombusenseVelocityLimit(
        parameter_name="co_alarm_threshold",
        max_rate_increase=Decimal("10.0"),
        max_rate_decrease=Decimal("10.0"),
        unit="ppm",
        min_value=Decimal("50.0"),
        max_value=Decimal("1000.0"),
        deadband=Decimal("5.0"),
        description="CO alarm threshold",
    ),
    "nox_setpoint": CombusenseVelocityLimit(
        parameter_name="nox_setpoint",
        max_rate_increase=Decimal("5.0"),
        max_rate_decrease=Decimal("5.0"),
        unit="ppm",
        min_value=Decimal("10.0"),
        max_value=Decimal("500.0"),
        deadband=Decimal("2.0"),
        description="NOx setpoint",
    ),
    "flame_intensity_threshold": CombusenseVelocityLimit(
        parameter_name="flame_intensity_threshold",
        max_rate_increase=Decimal("2.0"),
        max_rate_decrease=Decimal("2.0"),
        unit="%",
        min_value=Decimal("10.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("1.0"),
        description="Flame intensity threshold",
    ),
    "combustion_efficiency_target": CombusenseVelocityLimit(
        parameter_name="combustion_efficiency_target",
        max_rate_increase=Decimal("0.5"),
        max_rate_decrease=Decimal("0.5"),
        unit="%",
        min_value=Decimal("70.0"),
        max_value=Decimal("99.0"),
        deadband=Decimal("0.2"),
        description="Combustion efficiency target",
    ),
}

class VelocityLimitResult(BaseModel):
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
            content = f"{self.parameter_name}|{self.current_value}|{self.allowed_value}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

class CombusenseVelocityLimiter:
    VERSION = "1.0.0"

    def __init__(self, custom_limits: Optional[Dict[str, CombusenseVelocityLimit]] = None,
                 default_safety_mode: SafetyMode = SafetyMode.NORMAL, precision: int = 3):
        self._limits = dict(DEFAULT_COMBUSENSE_LIMITS)
        if custom_limits:
            self._limits.update(custom_limits)
        self._safety_mode = default_safety_mode
        self._precision = precision
        self._quantize_str = "0." + "0" * precision
        self._lock = threading.Lock()
        self._stats = {"total_requests": 0, "clamped_requests": 0, "allowed_requests": 0}
        logger.info("CombusenseVelocityLimiter initialized with %d limits", len(self._limits))

    @property
    def safety_mode(self) -> SafetyMode:
        return self._safety_mode

    @safety_mode.setter
    def safety_mode(self, mode: SafetyMode) -> None:
        self._safety_mode = mode

    def _quantize(self, value: Decimal) -> Decimal:
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _get_effective_rate(self, limit: CombusenseVelocityLimit, direction: SetpointDirection) -> Decimal:
        if direction == SetpointDirection.INCREASING:
            base_rate = limit.max_rate_increase
        elif direction == SetpointDirection.DECREASING:
            base_rate = limit.max_rate_decrease
        else:
            return Decimal("0")
        factor = {SafetyMode.NORMAL: Decimal("1.0"), SafetyMode.STARTUP: limit.startup_factor,
                  SafetyMode.SHUTDOWN: Decimal("0.7"), SafetyMode.EMERGENCY: Decimal("1.5")}[self._safety_mode]
        return self._quantize(base_rate * factor)

    def limit_setpoint_change(self, parameter: str, current_value: float, requested_value: float,
                               dt_seconds: float = 1.0, safety_mode_override: Optional[SafetyMode] = None) -> VelocityLimitResult:
        with self._lock:
            self._stats["total_requests"] += 1
            mode = safety_mode_override or self._safety_mode
            limit = self._limits.get(parameter, CombusenseVelocityLimit(
                parameter_name=parameter, max_rate_increase=Decimal("1.0"), max_rate_decrease=Decimal("1.0"),
                unit="", min_value=Decimal("-1000000"), max_value=Decimal("1000000"), deadband=Decimal("0.01")))

            current = Decimal(str(current_value))
            requested = Decimal(str(requested_value))
            dt = Decimal(str(max(dt_seconds, 0.001)))
            change = requested - current
            requested_rate = abs(change) / dt

            if abs(change) <= limit.deadband:
                self._stats["allowed_requests"] += 1
                return VelocityLimitResult(parameter_name=parameter, current_value=float(current),
                    requested_value=float(requested), allowed_value=float(requested),
                    status=VelocityLimitStatus.ALLOWED, direction=SetpointDirection.UNCHANGED,
                    requested_rate=float(requested_rate), max_rate=0.0, actual_rate=0.0, safety_mode=mode)

            direction = SetpointDirection.INCREASING if change > 0 else SetpointDirection.DECREASING
            max_rate = self._get_effective_rate(limit, direction)
            max_change = max_rate * dt

            if requested_rate <= max_rate:
                allowed, status, actual_rate, clamped = requested, VelocityLimitStatus.ALLOWED, requested_rate, Decimal("0")
                self._stats["allowed_requests"] += 1
            else:
                allowed = current + max_change if direction == SetpointDirection.INCREASING else current - max_change
                status, actual_rate, clamped = VelocityLimitStatus.CLAMPED, max_rate, abs(requested - allowed)
                self._stats["clamped_requests"] += 1

            allowed = max(limit.min_value, min(limit.max_value, self._quantize(allowed)))
            return VelocityLimitResult(parameter_name=parameter, current_value=float(current),
                requested_value=float(requested), allowed_value=float(allowed), status=status,
                direction=direction, requested_rate=float(self._quantize(requested_rate)),
                max_rate=float(max_rate), actual_rate=float(self._quantize(actual_rate)),
                clamped_amount=float(self._quantize(clamped)), safety_mode=mode)

    def get_statistics(self) -> Dict[str, Any]:
        total = self._stats["total_requests"]
        return {**self._stats, "clamp_rate": self._stats["clamped_requests"] / total if total > 0 else 0.0,
                "current_safety_mode": self._safety_mode.value}

__all__ = ["VelocityLimitStatus", "SetpointDirection", "SafetyMode", "CombusenseVelocityLimit",
           "VelocityLimitResult", "CombusenseVelocityLimiter", "DEFAULT_COMBUSENSE_LIMITS"]
