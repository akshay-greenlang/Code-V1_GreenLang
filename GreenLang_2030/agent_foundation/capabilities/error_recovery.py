"""
Error Recovery and Resilience Framework - Comprehensive error handling.

This module implements production-ready error recovery and resilience patterns including:
- Retry with exponential backoff and jitter
- Circuit breaker pattern
- Fallback strategies
- Compensation patterns
- Graceful degradation
- Health checking

Example:
    >>> framework = ResilienceFramework(config)
    >>> result = await framework.execute_with_resilience(func, params)
    >>> health = framework.get_health_status()
"""

import asyncio
import hashlib
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator
import numpy as np

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Types of errors for categorization."""

    TRANSIENT = "transient"    # Temporary, retry possible
    PERMANENT = "permanent"    # Persistent, retry futile
    PARTIAL = "partial"       # Partial success/failure
    TIMEOUT = "timeout"       # Exceeded time limit
    RESOURCE = "resource"     # Resource exhaustion
    VALIDATION = "validation" # Input/output validation
    LOGIC = "logic"          # Business logic error


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered


class RetryStrategy(str, Enum):
    """Retry strategies."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


class FallbackStrategy(str, Enum):
    """Fallback strategies."""

    CACHE = "cache"
    DEFAULT = "default"
    ALTERNATIVE = "alternative"
    DEGRADE = "degrade"
    NONE = "none"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    error_id: str
    error_type: ErrorType
    error_message: str
    timestamp: datetime
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0


@dataclass
class RecoveryAction:
    """Action taken for error recovery."""

    action_id: str
    strategy: str
    timestamp: datetime
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RetryPolicy:
    """Retry policy with backoff and jitter."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize retry policy."""
        self.config = config or {}
        self.strategy = RetryStrategy(
            config.get("strategy", RetryStrategy.EXPONENTIAL_BACKOFF)
        )
        self.max_attempts = config.get("max_attempts", 3)
        self.base_delay = config.get("base_delay_seconds", 1.0)
        self.max_delay = config.get("max_delay_seconds", 30.0)
        self.jitter = config.get("jitter", True)
        self.timeout = config.get("timeout_seconds", 30.0)

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_async(func, *args, **kwargs),
                    timeout=self.timeout
                )
                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} timed out")

            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)

                # Don't retry permanent errors
                if error_type == ErrorType.PERMANENT:
                    raise

                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_attempts} failed: {str(e)}"
                )

            # Wait before retry (except on last attempt)
            if attempt < self.max_attempts - 1:
                delay = self._calculate_delay(attempt)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        # All retries exhausted
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(
                self.base_delay * (2 ** attempt),
                self.max_delay
            )

        elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(
                self.base_delay * (attempt + 1),
                self.max_delay
            )

        elif self.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.base_delay

        else:  # IMMEDIATE
            delay = 0

        # Add jitter if enabled
        if self.jitter and delay > 0:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type."""
        error_str = str(error).lower()

        if isinstance(error, TimeoutError) or "timeout" in error_str:
            return ErrorType.TIMEOUT

        if "connection" in error_str or "network" in error_str:
            return ErrorType.TRANSIENT

        if "memory" in error_str or "resource" in error_str:
            return ErrorType.RESOURCE

        if "validation" in error_str or "invalid" in error_str:
            return ErrorType.PERMANENT

        # Default to transient for retry
        return ErrorType.TRANSIENT


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize circuit breaker."""
        self.config = config or {}
        self.failure_threshold = config.get("failure_threshold", 5)
        self.timeout_seconds = config.get("timeout_seconds", 60)
        self.half_open_attempts = config.get("half_open_attempts", 2)

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_change_time = datetime.now()

        # Metrics
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        self.total_calls += 1

        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            # Execute function
            result = await self._execute_async(func, *args, **kwargs)

            # Record success
            self._on_success()

            return result

        except Exception as e:
            # Record failure
            self._on_failure()

            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if not self.last_failure_time:
            return True

        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout_seconds

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        logger.info("Circuit breaker transitioning to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.state_change_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0

    def _on_success(self) -> None:
        """Handle successful execution."""
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            # Check if we should close the circuit
            if self.success_count >= self.half_open_attempts:
                self._transition_to_closed()
        else:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._transition_to_open()

        elif self.state == CircuitState.CLOSED:
            # Check if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        logger.info("Circuit breaker transitioning to CLOSED")
        self.state = CircuitState.CLOSED
        self.state_change_time = datetime.now()
        self.failure_count = 0
        self.success_count = 0

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        logger.warning("Circuit breaker transitioning to OPEN")
        self.state = CircuitState.OPEN
        self.state_change_time = datetime.now()

    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "failure_rate": (
                self.total_failures / self.total_calls
                if self.total_calls > 0 else 0
            ),
            "state_duration_seconds": (
                datetime.now() - self.state_change_time
            ).total_seconds()
        }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._transition_to_closed()
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0


class FallbackHandler:
    """Handle fallback strategies for failures."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize fallback handler."""
        self.config = config or {}
        self.cache: Dict[str, Any] = {}
        self.default_values: Dict[str, Any] = {}
        self.alternative_functions: Dict[str, Callable] = {}

    async def execute_with_fallback(
        self,
        func: Callable,
        fallback_strategy: FallbackStrategy,
        cache_key: Optional[str] = None,
        default_value: Optional[Any] = None,
        alternative_func: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with fallback on failure."""
        try:
            # Try primary function
            result = await self._execute_async(func, *args, **kwargs)

            # Cache successful result if caching enabled
            if cache_key and fallback_strategy == FallbackStrategy.CACHE:
                self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.warning(f"Primary function failed: {str(e)}, using fallback")

            # Apply fallback strategy
            if fallback_strategy == FallbackStrategy.CACHE:
                return self._fallback_to_cache(cache_key)

            elif fallback_strategy == FallbackStrategy.DEFAULT:
                return self._fallback_to_default(cache_key, default_value)

            elif fallback_strategy == FallbackStrategy.ALTERNATIVE:
                return await self._fallback_to_alternative(
                    alternative_func,
                    *args,
                    **kwargs
                )

            elif fallback_strategy == FallbackStrategy.DEGRADE:
                return self._fallback_to_degraded()

            else:
                # No fallback, re-raise
                raise

    def _fallback_to_cache(self, cache_key: Optional[str]) -> Any:
        """Fallback to cached value."""
        if cache_key and cache_key in self.cache:
            logger.info(f"Using cached value for {cache_key}")
            return self.cache[cache_key]
        raise Exception("No cached value available")

    def _fallback_to_default(
        self,
        cache_key: Optional[str],
        default_value: Optional[Any]
    ) -> Any:
        """Fallback to default value."""
        if default_value is not None:
            logger.info("Using default value")
            return default_value

        if cache_key and cache_key in self.default_values:
            logger.info(f"Using registered default value for {cache_key}")
            return self.default_values[cache_key]

        raise Exception("No default value available")

    async def _fallback_to_alternative(
        self,
        alternative_func: Optional[Callable],
        *args,
        **kwargs
    ) -> Any:
        """Fallback to alternative function."""
        if alternative_func:
            logger.info("Using alternative function")
            return await self._execute_async(alternative_func, *args, **kwargs)

        raise Exception("No alternative function available")

    def _fallback_to_degraded(self) -> Any:
        """Fallback to degraded service."""
        logger.warning("Service degraded, returning limited functionality")
        return {
            "status": "degraded",
            "message": "Service operating with limited functionality"
        }

    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    def register_default(self, key: str, value: Any) -> None:
        """Register default value for a key."""
        self.default_values[key] = value

    def register_alternative(self, key: str, func: Callable) -> None:
        """Register alternative function for a key."""
        self.alternative_functions[key] = func

    def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear cache."""
        if key:
            self.cache.pop(key, None)
        else:
            self.cache.clear()


class CompensationHandler:
    """Handle compensating transactions for rollback."""

    def __init__(self):
        """Initialize compensation handler."""
        self.compensation_stack: List[Tuple[str, Callable, tuple, dict]] = []
        self.executed_operations: List[Dict[str, Any]] = []

    async def execute_with_compensation(
        self,
        operation: Callable,
        compensation: Callable,
        operation_id: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with compensation capability."""
        try:
            # Execute operation
            result = await self._execute_async(operation, *args, **kwargs)

            # Record operation and its compensation
            self.compensation_stack.append(
                (operation_id, compensation, args, kwargs)
            )
            self.executed_operations.append({
                "id": operation_id,
                "timestamp": datetime.now(),
                "status": "completed"
            })

            return result

        except Exception as e:
            logger.error(f"Operation {operation_id} failed: {str(e)}")

            # Trigger compensation
            await self.compensate()

            raise

    async def compensate(self) -> None:
        """Execute compensation for all completed operations."""
        logger.info(f"Compensating {len(self.compensation_stack)} operations")

        # Execute compensations in reverse order
        while self.compensation_stack:
            operation_id, compensation, args, kwargs = self.compensation_stack.pop()

            try:
                await self._execute_async(compensation, *args, **kwargs)
                logger.info(f"Compensated operation {operation_id}")

            except Exception as e:
                logger.error(
                    f"Compensation failed for {operation_id}: {str(e)}"
                )

        self.executed_operations.clear()

    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    def clear(self) -> None:
        """Clear compensation stack."""
        self.compensation_stack.clear()
        self.executed_operations.clear()


class ErrorHandler:
    """Comprehensive error handler."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize error handler."""
        self.config = config or {}
        self.error_history: deque = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.recovery_actions: List[RecoveryAction] = []

    def record_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> ErrorRecord:
        """Record error occurrence."""
        error_type = self._classify_error(error)

        record = ErrorRecord(
            error_id=self._generate_error_id(),
            error_type=error_type,
            error_message=str(error),
            timestamp=datetime.now(),
            context=context
        )

        self.error_history.append(record)
        self.error_counts[error_type] += 1

        return record

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type."""
        error_str = str(error).lower()

        if isinstance(error, TimeoutError) or "timeout" in error_str:
            return ErrorType.TIMEOUT

        if "connection" in error_str or "network" in error_str:
            return ErrorType.TRANSIENT

        if "memory" in error_str or "resource" in error_str:
            return ErrorType.RESOURCE

        if "validation" in error_str or "invalid" in error_str:
            return ErrorType.VALIDATION

        if "partial" in error_str or "incomplete" in error_str:
            return ErrorType.PARTIAL

        # Check for known permanent error patterns
        permanent_keywords = ["not found", "does not exist", "invalid config"]
        if any(keyword in error_str for keyword in permanent_keywords):
            return ErrorType.PERMANENT

        return ErrorType.TRANSIENT

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_strategy: Optional[str] = None
    ) -> Optional[Any]:
        """Handle error with appropriate recovery strategy."""
        # Record error
        error_record = self.record_error(error, context)

        # Determine recovery strategy
        if not recovery_strategy:
            recovery_strategy = self._select_recovery_strategy(error_record)

        # Execute recovery
        try:
            recovery_result = await self._execute_recovery(
                error_record,
                recovery_strategy,
                context
            )

            # Record successful recovery
            action = RecoveryAction(
                action_id=self._generate_action_id(),
                strategy=recovery_strategy,
                timestamp=datetime.now(),
                success=True,
                details={"result": str(recovery_result)}
            )
            self.recovery_actions.append(action)

            error_record.recovery_attempted = True
            error_record.recovery_successful = True

            return recovery_result

        except Exception as recovery_error:
            logger.error(f"Recovery failed: {str(recovery_error)}")

            # Record failed recovery
            action = RecoveryAction(
                action_id=self._generate_action_id(),
                strategy=recovery_strategy,
                timestamp=datetime.now(),
                success=False,
                details={"error": str(recovery_error)}
            )
            self.recovery_actions.append(action)

            error_record.recovery_attempted = True
            error_record.recovery_successful = False

            return None

    def _select_recovery_strategy(self, error_record: ErrorRecord) -> str:
        """Select appropriate recovery strategy."""
        error_type = error_record.error_type

        if error_type == ErrorType.TRANSIENT:
            return "retry"
        elif error_type == ErrorType.TIMEOUT:
            return "retry_with_timeout_increase"
        elif error_type == ErrorType.RESOURCE:
            return "wait_and_retry"
        elif error_type == ErrorType.PARTIAL:
            return "resume_partial"
        elif error_type == ErrorType.VALIDATION:
            return "validate_and_fix"
        else:
            return "fallback"

    async def _execute_recovery(
        self,
        error_record: ErrorRecord,
        strategy: str,
        context: Dict[str, Any]
    ) -> Any:
        """Execute recovery strategy."""
        if strategy == "retry":
            return await self._retry_recovery(error_record, context)

        elif strategy == "retry_with_timeout_increase":
            return await self._retry_with_increased_timeout(error_record, context)

        elif strategy == "wait_and_retry":
            return await self._wait_and_retry(error_record, context)

        elif strategy == "resume_partial":
            return await self._resume_partial(error_record, context)

        elif strategy == "validate_and_fix":
            return await self._validate_and_fix(error_record, context)

        elif strategy == "fallback":
            return await self._fallback_recovery(error_record, context)

        else:
            raise Exception(f"Unknown recovery strategy: {strategy}")

    async def _retry_recovery(
        self,
        error_record: ErrorRecord,
        context: Dict[str, Any]
    ) -> Any:
        """Retry the failed operation."""
        logger.info(f"Retrying operation for error {error_record.error_id}")

        # Get original function from context
        if "function" in context:
            func = context["function"]
            args = context.get("args", ())
            kwargs = context.get("kwargs", {})

            # Simple retry (in production, use RetryPolicy)
            await asyncio.sleep(1)
            return await self._execute_async(func, *args, **kwargs)

        raise Exception("No function to retry")

    async def _retry_with_increased_timeout(
        self,
        error_record: ErrorRecord,
        context: Dict[str, Any]
    ) -> Any:
        """Retry with increased timeout."""
        logger.info("Retrying with increased timeout")

        if "function" in context:
            func = context["function"]
            args = context.get("args", ())
            kwargs = context.get("kwargs", {})

            # Increase timeout
            current_timeout = context.get("timeout", 30)
            new_timeout = current_timeout * 1.5

            return await asyncio.wait_for(
                self._execute_async(func, *args, **kwargs),
                timeout=new_timeout
            )

        raise Exception("No function to retry")

    async def _wait_and_retry(
        self,
        error_record: ErrorRecord,
        context: Dict[str, Any]
    ) -> Any:
        """Wait for resources and retry."""
        logger.info("Waiting for resources before retry")

        # Wait for resource availability
        await asyncio.sleep(5)

        return await self._retry_recovery(error_record, context)

    async def _resume_partial(
        self,
        error_record: ErrorRecord,
        context: Dict[str, Any]
    ) -> Any:
        """Resume from partial completion."""
        logger.info("Resuming from partial completion")

        # Get checkpoint or partial result
        partial_result = context.get("partial_result", {})

        # Resume from where we left off
        # Implementation depends on specific use case

        return {"status": "resumed", "partial": partial_result}

    async def _validate_and_fix(
        self,
        error_record: ErrorRecord,
        context: Dict[str, Any]
    ) -> Any:
        """Validate and fix input data."""
        logger.info("Validating and fixing input data")

        # Get input data
        input_data = context.get("input_data", {})

        # Attempt to fix validation errors
        fixed_data = self._fix_validation_errors(input_data, error_record)

        # Retry with fixed data
        if "function" in context:
            func = context["function"]
            return await self._execute_async(func, fixed_data)

        return fixed_data

    async def _fallback_recovery(
        self,
        error_record: ErrorRecord,
        context: Dict[str, Any]
    ) -> Any:
        """Fallback to safe default."""
        logger.info("Using fallback recovery")

        # Return safe default
        return {
            "status": "fallback",
            "message": "Using default recovery",
            "error_type": error_record.error_type
        }

    def _fix_validation_errors(
        self,
        data: Dict[str, Any],
        error_record: ErrorRecord
    ) -> Dict[str, Any]:
        """Attempt to fix validation errors in data."""
        fixed_data = data.copy()

        # Simple fixes for common validation errors
        error_msg = error_record.error_message.lower()

        if "required" in error_msg:
            # Add missing required fields with defaults
            # Implementation depends on schema
            pass

        if "type" in error_msg or "invalid" in error_msg:
            # Fix type mismatches
            # Implementation depends on expected types
            pass

        return fixed_data

    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function asynchronously."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = len(self.error_history)

        if total_errors == 0:
            return {"total_errors": 0}

        # Count by type
        type_counts = dict(self.error_counts)

        # Recovery statistics
        recovery_attempted = sum(
            1 for r in self.error_history if r.recovery_attempted
        )
        recovery_successful = sum(
            1 for r in self.error_history if r.recovery_successful
        )

        # Recent errors (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = sum(
            1 for r in self.error_history
            if r.timestamp > one_hour_ago
        )

        return {
            "total_errors": total_errors,
            "error_by_type": type_counts,
            "recovery_attempted": recovery_attempted,
            "recovery_successful": recovery_successful,
            "recovery_rate": (
                recovery_successful / recovery_attempted
                if recovery_attempted > 0 else 0
            ),
            "recent_errors_1h": recent_errors,
            "recent_error_rate": recent_errors / 60  # Per minute
        }

    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        timestamp = datetime.now().isoformat()
        return f"err_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"

    def _generate_action_id(self) -> str:
        """Generate unique action ID."""
        timestamp = datetime.now().isoformat()
        return f"act_{hashlib.sha256(timestamp.encode()).hexdigest()[:8]}"


class RecoveryStrategy:
    """Recovery strategy selector and executor."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize recovery strategy."""
        self.config = config or {}
        self.retry_policy = RetryPolicy(config.get("retry", {}))
        self.circuit_breaker = CircuitBreaker(config.get("circuit_breaker", {}))
        self.fallback_handler = FallbackHandler(config.get("fallback", {}))

    async def execute_with_recovery(
        self,
        func: Callable,
        recovery_options: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute function with comprehensive recovery."""
        # Determine recovery strategy
        use_retry = recovery_options.get("retry", True)
        use_circuit_breaker = recovery_options.get("circuit_breaker", True)
        use_fallback = recovery_options.get("fallback", True)

        # Execute with selected recovery mechanisms
        try:
            if use_circuit_breaker:
                # Execute through circuit breaker
                async def circuit_wrapped():
                    if use_retry:
                        return await self.retry_policy.execute_with_retry(
                            func, *args, **kwargs
                        )
                    else:
                        return await func(*args, **kwargs)

                return await self.circuit_breaker.execute(circuit_wrapped)

            elif use_retry:
                return await self.retry_policy.execute_with_retry(
                    func, *args, **kwargs
                )

            else:
                return await func(*args, **kwargs)

        except Exception as e:
            if use_fallback:
                # Try fallback
                fallback_strategy = recovery_options.get(
                    "fallback_strategy",
                    FallbackStrategy.DEFAULT
                )

                return await self.fallback_handler.execute_with_fallback(
                    func,
                    fallback_strategy,
                    cache_key=recovery_options.get("cache_key"),
                    default_value=recovery_options.get("default_value"),
                    alternative_func=recovery_options.get("alternative_func"),
                    *args,
                    **kwargs
                )
            else:
                raise


class HealthChecker:
    """Health checking for services and components."""

    def __init__(self):
        """Initialize health checker."""
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}

    def register_health_check(
        self,
        component: str,
        check_func: Callable
    ) -> None:
        """Register health check for component."""
        self.health_checks[component] = check_func

    async def check_health(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Check health of component(s)."""
        if component:
            # Check specific component
            if component in self.health_checks:
                status = await self._perform_health_check(
                    component,
                    self.health_checks[component]
                )
                self.health_status[component] = status
                return {component: status}
            else:
                return {component: {"status": "unknown"}}

        # Check all components
        results = {}
        for comp, check_func in self.health_checks.items():
            status = await self._perform_health_check(comp, check_func)
            self.health_status[comp] = status
            results[comp] = status

        return results

    async def _perform_health_check(
        self,
        component: str,
        check_func: Callable
    ) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if asyncio.iscoroutinefunction(check_func):
                healthy = await check_func()
            else:
                healthy = check_func()

            return {
                "status": "healthy" if healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "details": {}
            }

        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        if not self.health_status:
            return {"status": "unknown", "components": {}}

        healthy_count = sum(
            1 for status in self.health_status.values()
            if status.get("status") == "healthy"
        )

        total_count = len(self.health_status)
        overall_healthy = healthy_count == total_count

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "healthy_components": healthy_count,
            "total_components": total_count,
            "health_percentage": (healthy_count / total_count * 100)
            if total_count > 0 else 0,
            "components": self.health_status
        }


class ResilienceFramework:
    """Main resilience framework integrating all patterns."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize resilience framework."""
        self.config = config or {}
        self.error_handler = ErrorHandler(config)
        self.recovery_strategy = RecoveryStrategy(config)
        self.compensation_handler = CompensationHandler()
        self.health_checker = HealthChecker()

        # Patterns
        self.retry_policy = self.recovery_strategy.retry_policy
        self.circuit_breaker = self.recovery_strategy.circuit_breaker
        self.fallback_handler = self.recovery_strategy.fallback_handler

        # Metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "recovered_executions": 0
        }

    async def execute_with_resilience(
        self,
        func: Callable,
        resilience_options: Dict[str, Any] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with full resilience capabilities."""
        self.metrics["total_executions"] += 1
        options = resilience_options or {}

        try:
            # Execute with recovery
            result = await self.recovery_strategy.execute_with_recovery(
                func,
                options,
                *args,
                **kwargs
            )

            self.metrics["successful_executions"] += 1
            return result

        except Exception as e:
            self.metrics["failed_executions"] += 1

            # Record error
            context = {
                "function": func,
                "args": args,
                "kwargs": kwargs,
                **options
            }

            # Attempt recovery
            recovery_result = await self.error_handler.handle_error(
                e,
                context
            )

            if recovery_result is not None:
                self.metrics["recovered_executions"] += 1
                return recovery_result

            raise

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        return self.health_checker.get_overall_health()

    def get_metrics(self) -> Dict[str, Any]:
        """Get resilience metrics."""
        metrics = self.metrics.copy()

        # Add derived metrics
        if metrics["total_executions"] > 0:
            metrics["success_rate"] = (
                metrics["successful_executions"] / metrics["total_executions"]
            )
            metrics["failure_rate"] = (
                metrics["failed_executions"] / metrics["total_executions"]
            )
            metrics["recovery_rate"] = (
                metrics["recovered_executions"] / metrics["failed_executions"]
                if metrics["failed_executions"] > 0 else 0
            )

        # Add component metrics
        metrics["circuit_breaker"] = self.circuit_breaker.get_state()
        metrics["error_statistics"] = self.error_handler.get_error_statistics()

        return metrics

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state."""
        self.circuit_breaker.reset()

    def clear_fallback_cache(self) -> None:
        """Clear fallback cache."""
        self.fallback_handler.clear_cache()