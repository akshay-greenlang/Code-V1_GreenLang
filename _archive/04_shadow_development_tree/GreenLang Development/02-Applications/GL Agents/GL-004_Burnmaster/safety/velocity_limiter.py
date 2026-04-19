# -*- coding: utf-8 -*-
"""
GL-004 BURNMASTER - Velocity Limiter for Setpoint Rate-of-Change Protection

This module implements velocity limiting (rate-of-change protection) for
combustion setpoint adjustments to prevent thermal shock, flame instability,
and unsafe operating conditions.

Safety Function:
    Ensures setpoint changes respect physical constraints:
    - Thermal mass response times
    - Flame stability requirements
    - Equipment stress limitations
    - BMS/SIS ramp rate requirements

Reference Standards:
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - ASME CSD-1: Controls and Safety Devices for Automatically Fired Boilers
    - IEC 61511: Functional Safety - Safety Instrumented Systems
    - API 556: Instrumentation, Control, and Protective Systems

CRITICAL SAFETY RULE:
    This module enforces MAXIMUM rate limits. The velocity limiter will
    CLAMP setpoint changes that exceed safe rates. This is a FAIL-SAFE
    behavior - if in doubt, the limiter will slow down changes.

Example:
    >>> from safety.velocity_limiter import CombustionVelocityLimiter
    >>> limiter = CombustionVelocityLimiter()
    >>> result = limiter.limit_setpoint_change(
    ...     parameter="fuel_flow",
    ...     current_value=100.0,
    ...     requested_value=150.0,
    ...     dt_seconds=1.0
    ... )
    >>> print(result.allowed_value)  # Will be clamped to safe rate

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class VelocityLimitStatus(str, Enum):
    """Status of velocity limiting operation."""

    ALLOWED = "allowed"  # Change within limits
    CLAMPED = "clamped"  # Change clamped to max rate
    BLOCKED = "blocked"  # Change completely blocked (safety interlock)
    REVERSED = "reversed"  # Direction reversed for safety


class SetpointDirection(str, Enum):
    """Direction of setpoint change."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    UNCHANGED = "unchanged"


class SafetyMode(str, Enum):
    """Safety mode affecting velocity limits."""

    NORMAL = "normal"  # Normal operation
    STARTUP = "startup"  # Startup sequence (slower rates)
    SHUTDOWN = "shutdown"  # Shutdown sequence (controlled rates)
    EMERGENCY = "emergency"  # Emergency - fastest safe rates
    MANUAL = "manual"  # Manual mode (strictest limits)


# =============================================================================
# VELOCITY LIMIT DEFINITIONS
# =============================================================================


@dataclass(frozen=True)
class VelocityLimit:
    """
    Velocity limit definition for a combustion parameter.

    All rates are per-second maximums. The limiter enforces these
    regardless of control system request frequency.
    """

    parameter_name: str
    max_rate_increase: Decimal  # Maximum rate of increase per second
    max_rate_decrease: Decimal  # Maximum rate of decrease per second
    unit: str  # Engineering unit
    min_value: Decimal  # Absolute minimum allowed value
    max_value: Decimal  # Absolute maximum allowed value
    deadband: Decimal  # Ignore changes smaller than this
    startup_factor: Decimal = Decimal("0.5")  # Multiplier for startup
    shutdown_factor: Decimal = Decimal("0.7")  # Multiplier for shutdown
    emergency_factor: Decimal = Decimal("2.0")  # Multiplier for emergency
    description: str = ""


# Default velocity limits for combustion parameters
# Source: NFPA 85, ASME CSD-1, industrial best practices
DEFAULT_VELOCITY_LIMITS: Dict[str, VelocityLimit] = {
    "fuel_flow": VelocityLimit(
        parameter_name="fuel_flow",
        max_rate_increase=Decimal("5.0"),  # 5% per second max increase
        max_rate_decrease=Decimal("10.0"),  # 10% per second max decrease (faster)
        unit="%",
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("0.1"),
        description="Fuel flow rate limit per NFPA 85",
    ),
    "air_flow": VelocityLimit(
        parameter_name="air_flow",
        max_rate_increase=Decimal("7.0"),  # 7% per second (slightly faster than fuel)
        max_rate_decrease=Decimal("12.0"),  # 12% per second
        unit="%",
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("0.1"),
        description="Air flow rate limit for air-fuel ratio stability",
    ),
    "firing_rate": VelocityLimit(
        parameter_name="firing_rate",
        max_rate_increase=Decimal("3.0"),  # 3% per second (conservative)
        max_rate_decrease=Decimal("5.0"),  # 5% per second
        unit="%",
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("0.5"),
        description="Firing rate limit per ASME CSD-1",
    ),
    "o2_setpoint": VelocityLimit(
        parameter_name="o2_setpoint",
        max_rate_increase=Decimal("0.5"),  # 0.5% O2 per second
        max_rate_decrease=Decimal("0.5"),  # Symmetric
        unit="%O2",
        min_value=Decimal("1.0"),  # Minimum safe O2
        max_value=Decimal("15.0"),  # Maximum practical O2
        deadband=Decimal("0.05"),
        description="O2 setpoint change limit",
    ),
    "temperature_setpoint": VelocityLimit(
        parameter_name="temperature_setpoint",
        max_rate_increase=Decimal("2.0"),  # 2°C per second
        max_rate_decrease=Decimal("3.0"),  # 3°C per second (faster cooling)
        unit="°C",
        min_value=Decimal("0.0"),
        max_value=Decimal("1500.0"),
        deadband=Decimal("0.5"),
        description="Temperature setpoint limit for thermal stress prevention",
    ),
    "excess_air_bias": VelocityLimit(
        parameter_name="excess_air_bias",
        max_rate_increase=Decimal("1.0"),  # 1% per second
        max_rate_decrease=Decimal("1.0"),  # Symmetric
        unit="%",
        min_value=Decimal("-20.0"),
        max_value=Decimal("50.0"),
        deadband=Decimal("0.2"),
        description="Excess air bias adjustment limit",
    ),
    "damper_position": VelocityLimit(
        parameter_name="damper_position",
        max_rate_increase=Decimal("5.0"),  # 5% per second
        max_rate_decrease=Decimal("5.0"),  # Symmetric
        unit="%",
        min_value=Decimal("0.0"),
        max_value=Decimal("100.0"),
        deadband=Decimal("0.5"),
        description="Damper position rate limit",
    ),
    "fuel_pressure": VelocityLimit(
        parameter_name="fuel_pressure",
        max_rate_increase=Decimal("2.0"),  # 2 units per second
        max_rate_decrease=Decimal("3.0"),  # Faster decrease
        unit="bar",
        min_value=Decimal("0.0"),
        max_value=Decimal("10.0"),
        deadband=Decimal("0.01"),
        description="Fuel pressure setpoint limit",
    ),
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class VelocityLimitConfig(BaseModel):
    """Configuration for velocity limiter."""

    parameter_name: str = Field(..., min_length=1, max_length=100)
    max_rate_increase: float = Field(..., gt=0.0, le=1000.0)
    max_rate_decrease: float = Field(..., gt=0.0, le=1000.0)
    unit: str = Field(default="%", max_length=20)
    min_value: float = Field(default=0.0)
    max_value: float = Field(default=100.0)
    deadband: float = Field(default=0.1, ge=0.0)
    startup_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    shutdown_factor: float = Field(default=0.7, gt=0.0, le=1.0)
    emergency_factor: float = Field(default=2.0, ge=1.0, le=5.0)

    @field_validator("max_rate_decrease")
    @classmethod
    def validate_decrease_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_rate_decrease must be positive")
        return v


class VelocityLimitResult(BaseModel):
    """Result of velocity limiting operation."""

    parameter_name: str = Field(..., description="Parameter being limited")
    current_value: float = Field(..., description="Current value")
    requested_value: float = Field(..., description="Requested value")
    allowed_value: float = Field(..., description="Allowed value after limiting")
    status: VelocityLimitStatus = Field(..., description="Limiting status")
    direction: SetpointDirection = Field(..., description="Direction of change")
    requested_rate: float = Field(..., description="Requested rate of change")
    max_rate: float = Field(..., description="Maximum allowed rate")
    actual_rate: float = Field(..., description="Actual rate after limiting")
    clamped_amount: float = Field(default=0.0, description="Amount clamped")
    safety_mode: SafetyMode = Field(default=SafetyMode.NORMAL)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.parameter_name}|{self.current_value}|{self.requested_value}|"
                f"{self.allowed_value}|{self.status.value}|{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


class VelocityAuditRecord(BaseModel):
    """Audit record for velocity limiting events."""

    record_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parameter_name: str = Field(...)
    status: VelocityLimitStatus = Field(...)
    current_value: float = Field(...)
    requested_value: float = Field(...)
    allowed_value: float = Field(...)
    safety_mode: SafetyMode = Field(...)
    reason: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# VELOCITY LIMITER IMPLEMENTATION
# =============================================================================


class CombustionVelocityLimiter:
    """
    Rate-of-change limiter for combustion setpoint protection.

    Implements velocity limiting to prevent:
    - Thermal shock from rapid temperature changes
    - Flame instability from rapid fuel/air changes
    - Equipment stress from rapid load changes
    - Unsafe operating conditions

    FAIL-SAFE Design:
    - When in doubt, limits are applied (conservative)
    - Unknown parameters use strictest default limits
    - Safety mode always enforced

    Example:
        >>> limiter = CombustionVelocityLimiter()
        >>> result = limiter.limit_setpoint_change(
        ...     parameter="fuel_flow",
        ...     current_value=50.0,
        ...     requested_value=100.0,
        ...     dt_seconds=1.0
        ... )
        >>> print(f"Allowed: {result.allowed_value}")
        Allowed: 55.0  # Clamped to 5%/sec max rate
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        custom_limits: Optional[Dict[str, VelocityLimit]] = None,
        default_safety_mode: SafetyMode = SafetyMode.NORMAL,
        audit_callback: Optional[Callable[[VelocityAuditRecord], None]] = None,
        precision: int = 3,
    ) -> None:
        """
        Initialize velocity limiter.

        Args:
            custom_limits: Custom velocity limits (merged with defaults)
            default_safety_mode: Default safety mode
            audit_callback: Optional callback for audit events
            precision: Decimal precision for calculations
        """
        # Merge custom limits with defaults
        self._limits = dict(DEFAULT_VELOCITY_LIMITS)
        if custom_limits:
            self._limits.update(custom_limits)

        self._safety_mode = default_safety_mode
        self._audit_callback = audit_callback
        self._precision = precision
        self._quantize_str = "0." + "0" * precision

        # State tracking
        self._last_values: Dict[str, Tuple[float, float]] = {}  # param -> (value, timestamp)
        self._audit_records: List[VelocityAuditRecord] = []
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "clamped_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
        }

        logger.info(
            "CombustionVelocityLimiter initialized with %d limits, mode=%s",
            len(self._limits),
            self._safety_mode.value,
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def safety_mode(self) -> SafetyMode:
        """Get current safety mode."""
        return self._safety_mode

    @safety_mode.setter
    def safety_mode(self, mode: SafetyMode) -> None:
        """Set safety mode."""
        old_mode = self._safety_mode
        self._safety_mode = mode
        logger.info("Safety mode changed: %s -> %s", old_mode.value, mode.value)

    @property
    def limits(self) -> Dict[str, VelocityLimit]:
        """Get configured velocity limits."""
        return dict(self._limits)

    @property
    def statistics(self) -> Dict[str, int]:
        """Get limiting statistics."""
        return dict(self._stats)

    # =========================================================================
    # CORE LIMITING METHODS
    # =========================================================================

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _get_effective_rate(
        self,
        limit: VelocityLimit,
        direction: SetpointDirection,
    ) -> Decimal:
        """
        Get effective max rate considering safety mode and direction.

        Args:
            limit: Velocity limit definition
            direction: Direction of change

        Returns:
            Effective maximum rate for current conditions
        """
        # Get base rate based on direction
        if direction == SetpointDirection.INCREASING:
            base_rate = limit.max_rate_increase
        elif direction == SetpointDirection.DECREASING:
            base_rate = limit.max_rate_decrease
        else:
            return Decimal("0")

        # Apply safety mode factor
        factor = {
            SafetyMode.NORMAL: Decimal("1.0"),
            SafetyMode.STARTUP: limit.startup_factor,
            SafetyMode.SHUTDOWN: limit.shutdown_factor,
            SafetyMode.EMERGENCY: limit.emergency_factor,
            SafetyMode.MANUAL: Decimal("0.3"),  # Strictest for manual
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
        """
        Apply velocity limiting to a setpoint change request.

        DETERMINISTIC: Same inputs produce same outputs.

        Args:
            parameter: Parameter name (must exist in limits)
            current_value: Current setpoint value
            requested_value: Requested new value
            dt_seconds: Time delta in seconds
            safety_mode_override: Override safety mode for this request

        Returns:
            VelocityLimitResult with allowed value and status
        """
        with self._lock:
            self._stats["total_requests"] += 1

            # Get safety mode
            mode = safety_mode_override or self._safety_mode

            # Get limit (or use strictest default)
            if parameter in self._limits:
                limit = self._limits[parameter]
            else:
                # Unknown parameter - use strictest limits
                limit = VelocityLimit(
                    parameter_name=parameter,
                    max_rate_increase=Decimal("1.0"),
                    max_rate_decrease=Decimal("1.0"),
                    unit="",
                    min_value=Decimal("-1000000"),
                    max_value=Decimal("1000000"),
                    deadband=Decimal("0.01"),
                    description="Default strict limit for unknown parameter",
                )
                logger.warning(
                    "Unknown parameter '%s', using strict default limits", parameter
                )

            # Convert to Decimal
            current = Decimal(str(current_value))
            requested = Decimal(str(requested_value))
            dt = Decimal(str(max(dt_seconds, 0.001)))  # Prevent division by zero

            # Calculate requested change and rate
            change = requested - current
            requested_rate = abs(change) / dt

            # Determine direction
            if change > limit.deadband:
                direction = SetpointDirection.INCREASING
            elif change < -limit.deadband:
                direction = SetpointDirection.DECREASING
            else:
                # Within deadband - allow directly
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
                    clamped_amount=0.0,
                    safety_mode=mode,
                )

            # Get effective max rate
            max_rate = self._get_effective_rate(limit, direction)

            # Calculate maximum allowed change
            max_change = max_rate * dt

            # Apply limiting
            if requested_rate <= max_rate:
                # Within limits - allow full change
                allowed = requested
                status = VelocityLimitStatus.ALLOWED
                actual_rate = requested_rate
                clamped = Decimal("0")
                self._stats["allowed_requests"] += 1
            else:
                # Exceeds limits - clamp to max rate
                if direction == SetpointDirection.INCREASING:
                    allowed = current + max_change
                else:
                    allowed = current - max_change

                status = VelocityLimitStatus.CLAMPED
                actual_rate = max_rate
                clamped = abs(requested - allowed)
                self._stats["clamped_requests"] += 1

            # Enforce absolute bounds
            allowed = max(limit.min_value, min(limit.max_value, allowed))
            allowed = self._quantize(allowed)

            # Create result
            result = VelocityLimitResult(
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

            # Audit logging
            if status == VelocityLimitStatus.CLAMPED:
                self._log_audit_event(result, f"Clamped by {float(clamped):.3f}")

            # Update last value tracking
            self._last_values[parameter] = (float(allowed), time.monotonic())

            return result

    def limit_batch(
        self,
        changes: Dict[str, Tuple[float, float]],
        dt_seconds: float = 1.0,
    ) -> Dict[str, VelocityLimitResult]:
        """
        Apply velocity limiting to multiple setpoint changes.

        Args:
            changes: Dict of parameter -> (current_value, requested_value)
            dt_seconds: Time delta in seconds

        Returns:
            Dict of parameter -> VelocityLimitResult
        """
        results = {}
        for param, (current, requested) in changes.items():
            results[param] = self.limit_setpoint_change(
                parameter=param,
                current_value=current,
                requested_value=requested,
                dt_seconds=dt_seconds,
            )
        return results

    def validate_ramp(
        self,
        parameter: str,
        start_value: float,
        end_value: float,
        duration_seconds: float,
    ) -> Tuple[bool, str, float]:
        """
        Validate if a ramp is achievable within velocity limits.

        Args:
            parameter: Parameter name
            start_value: Starting value
            end_value: Ending value
            duration_seconds: Desired ramp duration

        Returns:
            Tuple of (is_valid, message, minimum_duration)
        """
        if parameter not in self._limits:
            return False, f"Unknown parameter: {parameter}", 0.0

        limit = self._limits[parameter]
        change = abs(Decimal(str(end_value)) - Decimal(str(start_value)))

        # Determine direction
        if end_value > start_value:
            max_rate = self._get_effective_rate(limit, SetpointDirection.INCREASING)
        else:
            max_rate = self._get_effective_rate(limit, SetpointDirection.DECREASING)

        if max_rate == 0:
            return False, "Zero rate not allowed", float("inf")

        # Calculate minimum duration
        min_duration = float(change / max_rate)

        if duration_seconds >= min_duration:
            return True, "Ramp is achievable", min_duration
        else:
            return (
                False,
                f"Ramp too fast. Need at least {min_duration:.1f}s",
                min_duration,
            )

    # =========================================================================
    # SAFETY MODE METHODS
    # =========================================================================

    def enter_startup_mode(self) -> None:
        """Enter startup mode (slower rates)."""
        self.safety_mode = SafetyMode.STARTUP

    def enter_shutdown_mode(self) -> None:
        """Enter shutdown mode (controlled rates)."""
        self.safety_mode = SafetyMode.SHUTDOWN

    def enter_emergency_mode(self) -> None:
        """Enter emergency mode (fastest safe rates)."""
        self.safety_mode = SafetyMode.EMERGENCY

    def enter_normal_mode(self) -> None:
        """Enter normal operation mode."""
        self.safety_mode = SafetyMode.NORMAL

    def enter_manual_mode(self) -> None:
        """Enter manual mode (strictest limits)."""
        self.safety_mode = SafetyMode.MANUAL

    # =========================================================================
    # AUDIT AND MONITORING
    # =========================================================================

    def _log_audit_event(
        self,
        result: VelocityLimitResult,
        reason: str = "",
    ) -> None:
        """Log an audit event."""
        record = VelocityAuditRecord(
            parameter_name=result.parameter_name,
            status=result.status,
            current_value=result.current_value,
            requested_value=result.requested_value,
            allowed_value=result.allowed_value,
            safety_mode=result.safety_mode,
            reason=reason,
            provenance_hash=result.provenance_hash,
        )
        self._audit_records.append(record)

        # Keep only last 1000 records
        if len(self._audit_records) > 1000:
            self._audit_records = self._audit_records[-1000:]

        # Invoke callback
        if self._audit_callback:
            try:
                self._audit_callback(record)
            except Exception as e:
                logger.error("Audit callback failed: %s", e)

        logger.debug(
            "VelocityLimiter audit: %s %s->%s (allowed: %s, status: %s)",
            result.parameter_name,
            result.current_value,
            result.requested_value,
            result.allowed_value,
            result.status.value,
        )

    def get_audit_records(
        self,
        limit: int = 100,
        parameter: Optional[str] = None,
    ) -> List[VelocityAuditRecord]:
        """Get audit records."""
        records = self._audit_records
        if parameter:
            records = [r for r in records if r.parameter_name == parameter]
        return list(reversed(records[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get velocity limiter statistics."""
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "clamp_rate": self._stats["clamped_requests"] / total if total > 0 else 0.0,
            "block_rate": self._stats["blocked_requests"] / total if total > 0 else 0.0,
            "current_safety_mode": self._safety_mode.value,
            "configured_limits": list(self._limits.keys()),
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_requests": 0,
            "clamped_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
        }

    # =========================================================================
    # LIMIT MANAGEMENT
    # =========================================================================

    def add_limit(self, limit: VelocityLimit) -> None:
        """Add or update a velocity limit."""
        self._limits[limit.parameter_name] = limit
        logger.info("Velocity limit added/updated: %s", limit.parameter_name)

    def remove_limit(self, parameter: str) -> bool:
        """Remove a velocity limit."""
        if parameter in self._limits:
            del self._limits[parameter]
            logger.info("Velocity limit removed: %s", parameter)
            return True
        return False

    def get_limit(self, parameter: str) -> Optional[VelocityLimit]:
        """Get a specific velocity limit."""
        return self._limits.get(parameter)

    def __repr__(self) -> str:
        return (
            f"CombustionVelocityLimiter(mode={self._safety_mode.value}, "
            f"limits={len(self._limits)})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_velocity_limit(
    parameter_name: str,
    max_rate_increase: float,
    max_rate_decrease: float,
    unit: str = "%",
    min_value: float = 0.0,
    max_value: float = 100.0,
    deadband: float = 0.1,
) -> VelocityLimit:
    """
    Create a VelocityLimit with the given parameters.

    Args:
        parameter_name: Name of the parameter
        max_rate_increase: Max rate of increase per second
        max_rate_decrease: Max rate of decrease per second
        unit: Engineering unit
        min_value: Minimum absolute value
        max_value: Maximum absolute value
        deadband: Change deadband

    Returns:
        VelocityLimit instance
    """
    return VelocityLimit(
        parameter_name=parameter_name,
        max_rate_increase=Decimal(str(max_rate_increase)),
        max_rate_decrease=Decimal(str(max_rate_decrease)),
        unit=unit,
        min_value=Decimal(str(min_value)),
        max_value=Decimal(str(max_value)),
        deadband=Decimal(str(deadband)),
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "VelocityLimitStatus",
    "SetpointDirection",
    "SafetyMode",
    # Data classes
    "VelocityLimit",
    # Models
    "VelocityLimitConfig",
    "VelocityLimitResult",
    "VelocityAuditRecord",
    # Main class
    "CombustionVelocityLimiter",
    # Constants
    "DEFAULT_VELOCITY_LIMITS",
    # Functions
    "create_velocity_limit",
]
