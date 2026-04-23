# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Safety Module

This package provides safety-critical components for combustion control:
- Circuit breaker for external system integration
- Safety envelope validation
- IEC 61511 compliant safety functions
- Sensor failure mode handling
- Redundant sensor voting (2oo3, 1oo2D)
- Combustion safeguards and BMS/SIS integration

Author: GL-BackendDeveloper
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CallResult,
    CircuitBreakerRegistry,
    circuit_breaker_registry
)

from .velocity_limiter import (
    VelocityLimitStatus,
    SetpointDirection,
    SafetyMode,
    CombusenseVelocityLimit,
    VelocityLimitResult,
    CombusenseVelocityLimiter,
    DEFAULT_COMBUSENSE_LIMITS,
)

from .sensor_failure_handling import (
    FailureMode,
    FailureAction,
    SensorHealth,
    SensorReading,
    SensorFailureHandler,
)

from .redundant_sensor_voting import (
    VotingArchitecture,
    VoteResult,
    SensorInput,
    VotingOutput,
    RedundantSensorVoter,
)

from .combustion_safeguards import (
    InterlockType,
    SafetyState,
    TripReason,
    SafeguardAction,
    InterlockDefinition,
    CombustionSafeguardSystem,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CallResult",
    "CircuitBreakerRegistry",
    "circuit_breaker_registry",
    # Velocity Limiter
    "VelocityLimitStatus",
    "SetpointDirection",
    "SafetyMode",
    "CombusenseVelocityLimit",
    "VelocityLimitResult",
    "CombusenseVelocityLimiter",
    "DEFAULT_COMBUSENSE_LIMITS",
    # Sensor Failure Handling
    "FailureMode",
    "FailureAction",
    "SensorHealth",
    "SensorReading",
    "SensorFailureHandler",
    # Redundant Sensor Voting
    "VotingArchitecture",
    "VoteResult",
    "SensorInput",
    "VotingOutput",
    "RedundantSensorVoter",
    # Combustion Safeguards
    "InterlockType",
    "SafetyState",
    "TripReason",
    "SafeguardAction",
    "InterlockDefinition",
    "CombustionSafeguardSystem",
]
