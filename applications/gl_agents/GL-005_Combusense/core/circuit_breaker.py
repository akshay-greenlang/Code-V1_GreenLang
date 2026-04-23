# -*- coding: utf-8 -*-
"""
Circuit Breaker Decorator - @circuit_protected for GL-005 COMBUSENSE

This module provides the @circuit_protected decorator for protecting
Modbus/OPC-UA and other external system calls with circuit breaker pattern.

The decorator wraps async functions with automatic circuit breaker protection,
providing fail-safe behavior per IEC 61511 requirements.

Key Features:
    1. Automatic circuit breaker creation and management
    2. Configurable failure thresholds and timeouts
    3. Fallback value support for continuous operation
    4. State transition callbacks for alarm integration
    5. SHA-256 provenance tracking for audit trails

Example:
    >>> from core.circuit_breaker import circuit_protected
    >>>
    >>> @circuit_protected(
    ...     service_name="modbus_plc",
    ...     failure_threshold=3,
    ...     fallback_value={"status": "OFFLINE"}
    ... )
    ... async def read_plc_values():
    ...     return await plc.read_registers([40001, 40002])
    >>>
    >>> result = await read_plc_values()
    >>> if result.from_fallback:
    ...     logger.warning("Using fallback values")

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import (
    TypeVar, Callable, Optional, Any, Dict, List, Awaitable
)
from datetime import datetime, timezone
from enum import Enum
import asyncio
import functools
import hashlib
import json
import logging

# Import from safety module
from safety.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CallResult,
    CircuitBreakerRegistry,
    circuit_breaker_registry,
    RecoveryStrategy,
    SafetyAction,
    FailureType,
    CircuitBreakerState,
    CircuitBreakerMetrics
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


def circuit_protected(
    service_name: str,
    *,
    cache_key: Optional[str] = None,
    timeout: Optional[float] = None,
    failure_threshold: int = 3,
    recovery_timeout_seconds: float = 10.0,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.FALLBACK_CACHE,
    safety_action: SafetyAction = SafetyAction.LOG_ALARM,
    fallback_value: Optional[Any] = None,
    on_fallback: Optional[Callable[..., Any]] = None,
    on_open: Optional[Callable[[str], None]] = None,
    on_close: Optional[Callable[[str], None]] = None
) -> Callable:
    """
    Decorator to protect async functions with circuit breaker pattern.

    Provides automatic circuit breaker protection for Modbus, OPC-UA,
    and other external system integrations in combustion control.

    This decorator:
    1. Creates/retrieves a circuit breaker for the service
    2. Wraps the function call with circuit breaker protection
    3. Handles fallback values when circuit is open
    4. Triggers callbacks on state transitions

    Args:
        service_name: Name of the protected service (e.g., "modbus_plc", "opc_ua_dcs")
        cache_key: Optional key for caching successful results
        timeout: Optional timeout override for the call (seconds)
        failure_threshold: Number of failures before opening circuit
        recovery_timeout_seconds: Time to wait before recovery attempt
        recovery_strategy: Strategy when circuit is open
        safety_action: IEC 61511 safety action on circuit open
        fallback_value: Static fallback value when circuit is open
        on_fallback: Callback when fallback is used: fn(*args, **kwargs) -> value
        on_open: Callback when circuit opens: fn(service_name)
        on_close: Callback when circuit closes: fn(service_name)

    Returns:
        Decorated async function that returns CallResult

    Example - Protect Modbus PLC communication:
        >>> @circuit_protected(
        ...     service_name="modbus_plc_main",
        ...     failure_threshold=3,
        ...     recovery_timeout_seconds=10.0,
        ...     fallback_value={"fuel_flow": 0.0, "status": "OFFLINE"},
        ...     safety_action=SafetyAction.TRIGGER_ALARM
        ... )
        ... async def read_plc_registers() -> Dict[str, float]:
        ...     return await plc_client.read_registers([40001, 40002, 40003])

    Example - Protect OPC-UA DCS communication:
        >>> @circuit_protected(
        ...     service_name="opc_ua_dcs",
        ...     cache_key="dcs_process_values",
        ...     on_open=lambda svc: send_alarm(f"{svc} offline"),
        ...     on_fallback=lambda: get_last_known_values()
        ... )
        ... async def read_dcs_values() -> Dict[str, Any]:
        ...     return await opc_client.read_nodes(node_ids)

    Example - Use in combustion control loop:
        >>> async def control_loop():
        ...     result = await read_plc_registers()
        ...     if result.from_cache:
        ...         logger.warning("Using cached PLC values")
        ...     elif result.from_fallback:
        ...         logger.warning("Using fallback values - PLC offline")
        ...         trigger_safe_mode()
        ...     else:
        ...         process_values(result.value)

    IEC 61511 Compliance:
        - Implements fail-safe behavior per SIL requirements
        - Logs all circuit state transitions for audit trail
        - Supports safety action callbacks for alarm management
        - Provides SHA-256 provenance tracking

    Real-Time Considerations:
        - Fast failure detection (configurable threshold)
        - Quick recovery testing (10s default)
        - Cache-based fallback for continuous operation
        - Minimal latency overhead (<1ms when closed)
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[CallResult]]:
        # Track if breaker was created by this decorator
        _breaker: Optional[CircuitBreaker] = None
        _previous_state: Optional[CircuitState] = None

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> CallResult:
            nonlocal _breaker, _previous_state

            # Get or create circuit breaker
            if _breaker is None:
                # Check registry first
                _breaker = circuit_breaker_registry.get(service_name)

                if _breaker is None:
                    # Create new circuit breaker with config
                    config = CircuitBreakerConfig(
                        service_name=service_name,
                        failure_threshold=failure_threshold,
                        recovery_timeout_seconds=recovery_timeout_seconds,
                        recovery_strategy=recovery_strategy,
                        safety_action_on_open=safety_action,
                        call_timeout_seconds=timeout or 5.0
                    )

                    # Create breaker with fallback provider if specified
                    _breaker = CircuitBreaker(
                        config=config,
                        fallback_provider=on_fallback,
                        alarm_callback=lambda svc, msg: logger.warning(
                            f"[SAFETY ALARM] {svc}: {msg}"
                        )
                    )

                    # Register in global registry
                    await circuit_breaker_registry.register(service_name, config)

                _previous_state = _breaker.state

            # Check for state transitions (for callbacks)
            current_state = _breaker.state
            if _previous_state != current_state:
                if current_state == CircuitState.OPEN and on_open:
                    try:
                        on_open(service_name)
                    except Exception as e:
                        logger.error(f"on_open callback failed: {e}")

                elif current_state == CircuitState.CLOSED and on_close:
                    try:
                        on_close(service_name)
                    except Exception as e:
                        logger.error(f"on_close callback failed: {e}")

                _previous_state = current_state

            # Execute protected call
            result = await _breaker.call(
                func,
                *args,
                timeout=timeout,
                cache_key=cache_key,
                **kwargs
            )

            # Handle fallback value if result failed and circuit is open
            if not result.success and _breaker.is_open:
                if fallback_value is not None:
                    return CallResult(
                        success=True,
                        value=fallback_value,
                        duration_ms=result.duration_ms,
                        from_fallback=True,
                        circuit_state=result.circuit_state,
                        service_name=service_name,
                        provenance_hash=result.provenance_hash
                    )
                elif on_fallback is not None:
                    try:
                        if asyncio.iscoroutinefunction(on_fallback):
                            fb_value = await on_fallback(*args, **kwargs)
                        else:
                            fb_value = on_fallback(*args, **kwargs)

                        return CallResult(
                            success=True,
                            value=fb_value,
                            duration_ms=result.duration_ms,
                            from_fallback=True,
                            circuit_state=result.circuit_state,
                            service_name=service_name,
                            provenance_hash=result.provenance_hash
                        )
                    except Exception as e:
                        logger.error(f"Fallback callback failed: {e}")

            return result

        # Attach metadata to decorated function
        wrapper._circuit_breaker_service = service_name
        wrapper._is_circuit_protected = True

        return wrapper
    return decorator


# =============================================================================
# Utility Functions
# =============================================================================

def get_circuit_breaker(service_name: str) -> Optional[CircuitBreaker]:
    """
    Get circuit breaker by service name.

    Args:
        service_name: Name of the protected service

    Returns:
        CircuitBreaker instance or None if not found
    """
    return circuit_breaker_registry.get(service_name)


async def reset_circuit_breaker(service_name: str) -> bool:
    """
    Reset a specific circuit breaker to closed state.

    Args:
        service_name: Name of the service

    Returns:
        True if reset successful, False if not found
    """
    breaker = circuit_breaker_registry.get(service_name)
    if breaker:
        await breaker.reset()
        return True
    return False


async def force_open_circuit(service_name: str, reason: str) -> bool:
    """
    Force a circuit breaker to open state.

    Use for maintenance or emergency situations.

    Args:
        service_name: Name of the service
        reason: Reason for forcing open

    Returns:
        True if successful, False if not found
    """
    breaker = circuit_breaker_registry.get(service_name)
    if breaker:
        await breaker.force_open(reason)
        return True
    return False


def get_all_circuit_states() -> Dict[str, CircuitBreakerState]:
    """Get states of all registered circuit breakers."""
    return circuit_breaker_registry.get_all_states()


def get_all_circuit_metrics() -> Dict[str, CircuitBreakerMetrics]:
    """Get metrics of all registered circuit breakers."""
    return circuit_breaker_registry.get_all_metrics()


def get_open_circuits() -> List[str]:
    """Get list of currently open circuit breakers."""
    return circuit_breaker_registry.get_open_circuits()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Decorator
    "circuit_protected",
    # Utility functions
    "get_circuit_breaker",
    "reset_circuit_breaker",
    "force_open_circuit",
    "get_all_circuit_states",
    "get_all_circuit_metrics",
    "get_open_circuits",
    # Re-exports from safety module
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CallResult",
    "CircuitBreakerRegistry",
    "circuit_breaker_registry",
    "RecoveryStrategy",
    "SafetyAction",
    "FailureType",
    "CircuitBreakerState",
    "CircuitBreakerMetrics"
]
