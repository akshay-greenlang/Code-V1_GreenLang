# -*- coding: utf-8 -*-
"""
SafetyMonitor - Safety oversight for agent orchestration.

This module implements safety monitoring, constraint validation, circuit breakers,
and operational guardrails for safe agent execution in industrial systems.

Example:
    >>> monitor = SafetyMonitor(config=SafetyConfig())
    >>> await monitor.validate_operation(operation)
    >>> if monitor.is_safe_to_proceed():
    ...     await execute_operation()

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SafetyLevel(str, Enum):
    """Safety levels for operations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConstraintType(str, Enum):
    """Types of safety constraints."""

    THRESHOLD = "threshold"
    RATE_LIMIT = "rate_limit"
    DEPENDENCY = "dependency"
    PERMISSION = "permission"
    RESOURCE = "resource"
    TIMING = "timing"
    COMPLIANCE = "compliance"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class ViolationSeverity(str, Enum):
    """Severity of safety violations."""

    CRITICAL = "critical"  # Immediate stop required
    MAJOR = "major"  # Action required, may continue
    MINOR = "minor"  # Warning, continue with caution
    WARNING = "warning"  # Informational warning


@dataclass
class SafetyConstraint:
    """
    Defines a safety constraint for operations.

    Attributes:
        constraint_id: Unique constraint identifier
        name: Constraint name
        constraint_type: Type of constraint
        condition: Condition expression or function name
        threshold: Threshold value for threshold constraints
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        rate_limit: Rate limit (calls per window)
        rate_window_seconds: Rate limit window
        level: Safety level
        enabled: Whether constraint is active
        metadata: Additional constraint metadata
    """

    name: str
    constraint_type: ConstraintType
    constraint_id: str = field(default_factory=lambda: f"constraint-{id(object())}")
    condition: Optional[str] = None
    threshold: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    rate_limit: Optional[int] = None
    rate_window_seconds: float = 60.0
    level: SafetyLevel = SafetyLevel.MEDIUM
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert constraint to dictionary."""
        return {
            "constraint_id": self.constraint_id,
            "name": self.name,
            "constraint_type": self.constraint_type.value,
            "condition": self.condition,
            "threshold": self.threshold,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "rate_limit": self.rate_limit,
            "rate_window_seconds": self.rate_window_seconds,
            "level": self.level.value,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


@dataclass
class SafetyViolation:
    """
    Represents a safety constraint violation.

    Attributes:
        violation_id: Unique violation identifier
        constraint_id: ID of violated constraint
        constraint_name: Name of violated constraint
        severity: Violation severity
        message: Violation description
        actual_value: Actual value that caused violation
        expected_value: Expected/allowed value
        timestamp: When violation occurred
        operation_id: ID of the operation that caused violation
        agent_id: Agent that caused violation
        resolved: Whether violation has been resolved
        metadata: Additional violation metadata
    """

    constraint_id: str
    constraint_name: str
    severity: ViolationSeverity
    message: str
    violation_id: str = field(default_factory=lambda: f"violation-{id(object())}")
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    operation_id: Optional[str] = None
    agent_id: Optional[str] = None
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "violation_id": self.violation_id,
            "constraint_id": self.constraint_id,
            "constraint_name": self.constraint_name,
            "severity": self.severity.value,
            "message": self.message,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
            "timestamp": self.timestamp,
            "operation_id": self.operation_id,
            "agent_id": self.agent_id,
            "resolved": self.resolved,
            "metadata": self.metadata,
        }


@dataclass
class OperationContext:
    """
    Context for an operation being validated.

    Attributes:
        operation_id: Unique operation identifier
        operation_type: Type of operation
        agent_id: Agent performing operation
        parameters: Operation parameters
        timestamp: When operation was initiated
        metadata: Additional context metadata
    """

    operation_type: str
    agent_id: str
    parameters: Dict[str, Any]
    operation_id: str = field(default_factory=lambda: f"op-{id(object())}")
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "agent_id": self.agent_id,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """
    Result of safety validation.

    Attributes:
        is_safe: Whether operation is safe to proceed
        violations: List of violations found
        warnings: List of warnings
        checked_constraints: Number of constraints checked
        validation_time_ms: Time taken for validation
        metadata: Additional result metadata
    """

    is_safe: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_constraints: int = 0
    validation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_safe": self.is_safe,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
            "checked_constraints": self.checked_constraints,
            "validation_time_ms": round(self.validation_time_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Attributes:
        name: Circuit breaker name
        state: Current circuit state
        failure_count: Number of failures
        success_count: Number of successes (in half-open)
        last_failure_time: Last failure timestamp
        last_success_time: Last success timestamp
        failure_threshold: Failures before opening
        success_threshold: Successes to close (half-open)
        timeout_seconds: Time before trying half-open
        metadata: Additional metadata
    """

    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[str] = None
    last_success_time: Optional[str] = None
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_success(self) -> None:
        """Record a successful call."""
        self.last_success_time = datetime.now(timezone.utc).isoformat()

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit {self.name} closed after recovery")

        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.last_failure_time = datetime.now(timezone.utc).isoformat()
        self.failure_count += 1

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit {self.name} opened after half-open failure")

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit {self.name} opened after {self.failure_count} failures"
                )

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                last_fail = datetime.fromisoformat(
                    self.last_failure_time.replace("Z", "+00:00")
                )
                elapsed = (datetime.now(timezone.utc) - last_fail).total_seconds()
                if elapsed >= self.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit {self.name} entering half-open state")
                    return True
            return False

        # Half-open: allow limited calls
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit breaker to dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class RateLimitBucket:
    """
    Token bucket for rate limiting.

    Attributes:
        key: Rate limit key
        tokens: Current token count
        max_tokens: Maximum tokens
        refill_rate: Tokens per second
        last_refill: Last refill timestamp
    """

    key: str
    max_tokens: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Initialize tokens to max."""
        if self.tokens == 0.0:
            self.tokens = float(self.max_tokens)

    def try_consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill = now


@dataclass
class SafetyConfig:
    """
    Configuration for SafetyMonitor.

    Attributes:
        enable_circuit_breakers: Enable circuit breakers
        enable_rate_limiting: Enable rate limiting
        enable_constraint_validation: Enable constraint validation
        default_failure_threshold: Default circuit breaker threshold
        default_timeout_seconds: Default circuit timeout
        default_rate_limit: Default rate limit (calls per minute)
        violation_log_level: Log level for violations
        halt_on_critical: Halt on critical violations
        metrics_enabled: Enable metrics collection
    """

    enable_circuit_breakers: bool = True
    enable_rate_limiting: bool = True
    enable_constraint_validation: bool = True
    default_failure_threshold: int = 5
    default_timeout_seconds: float = 60.0
    default_rate_limit: int = 100  # per minute
    violation_log_level: str = "WARNING"
    halt_on_critical: bool = True
    metrics_enabled: bool = True


@dataclass
class SafetyMetrics:
    """Metrics for safety monitoring."""

    validations_performed: int = 0
    validations_passed: int = 0
    validations_failed: int = 0
    violations_total: int = 0
    violations_critical: int = 0
    violations_major: int = 0
    violations_minor: int = 0
    rate_limits_triggered: int = 0
    circuit_breakers_opened: int = 0
    operations_blocked: int = 0
    avg_validation_time_ms: float = 0.0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "validations_performed": self.validations_performed,
            "validations_passed": self.validations_passed,
            "validations_failed": self.validations_failed,
            "violations_total": self.violations_total,
            "violations_critical": self.violations_critical,
            "violations_major": self.violations_major,
            "violations_minor": self.violations_minor,
            "rate_limits_triggered": self.rate_limits_triggered,
            "circuit_breakers_opened": self.circuit_breakers_opened,
            "operations_blocked": self.operations_blocked,
            "avg_validation_time_ms": round(self.avg_validation_time_ms, 2),
            "last_updated": self.last_updated,
        }


# Type alias for constraint validators
ConstraintValidator = Callable[[OperationContext, SafetyConstraint], Optional[SafetyViolation]]


class SafetyMonitor:
    """
    Safety monitor for agent orchestration.

    Provides:
    - Constraint validation
    - Rate limiting
    - Circuit breakers
    - Violation tracking
    - Safety metrics

    Example:
        >>> config = SafetyConfig(halt_on_critical=True)
        >>> monitor = SafetyMonitor(config)
        >>>
        >>> # Add constraints
        >>> monitor.add_constraint(SafetyConstraint(
        ...     name="max_temperature",
        ...     constraint_type=ConstraintType.THRESHOLD,
        ...     max_value=500.0,
        ...     level=SafetyLevel.CRITICAL
        ... ))
        >>>
        >>> # Validate operation
        >>> context = OperationContext(
        ...     operation_type="thermal_update",
        ...     agent_id="thermal-agent",
        ...     parameters={"temperature": 450}
        ... )
        >>> result = await monitor.validate_operation(context)
        >>> if result.is_safe:
        ...     await execute_operation()
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        """
        Initialize SafetyMonitor.

        Args:
            config: Configuration options
        """
        self.config = config or SafetyConfig()
        self._constraints: Dict[str, SafetyConstraint] = {}
        self._violations: List[SafetyViolation] = []
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._rate_limiters: Dict[str, RateLimitBucket] = {}
        self._custom_validators: Dict[str, ConstraintValidator] = {}
        self._metrics = SafetyMetrics()
        self._validation_times: List[float] = []
        self._lock = asyncio.Lock()
        self._halted = False

        # Register default validators
        self._register_default_validators()

        logger.info("SafetyMonitor initialized")

    def _register_default_validators(self) -> None:
        """Register default constraint validators."""
        self._custom_validators["threshold"] = self._validate_threshold
        self._custom_validators["rate_limit"] = self._validate_rate_limit
        self._custom_validators["resource"] = self._validate_resource

    def add_constraint(self, constraint: SafetyConstraint) -> None:
        """
        Add a safety constraint.

        Args:
            constraint: Constraint to add
        """
        self._constraints[constraint.constraint_id] = constraint
        logger.debug(f"Added constraint: {constraint.name}")

    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Remove a safety constraint.

        Args:
            constraint_id: Constraint to remove

        Returns:
            True if constraint was removed
        """
        if constraint_id in self._constraints:
            del self._constraints[constraint_id]
            return True
        return False

    def add_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
    ) -> CircuitBreaker:
        """
        Add a circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening
            timeout_seconds: Timeout before half-open

        Returns:
            Created circuit breaker
        """
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds,
        )
        self._circuit_breakers[name] = breaker
        return breaker

    def add_rate_limiter(
        self,
        key: str,
        max_requests: int,
        window_seconds: float = 60.0,
    ) -> RateLimitBucket:
        """
        Add a rate limiter.

        Args:
            key: Rate limiter key
            max_requests: Maximum requests in window
            window_seconds: Time window

        Returns:
            Created rate limiter
        """
        bucket = RateLimitBucket(
            key=key,
            max_tokens=max_requests,
            refill_rate=max_requests / window_seconds,
        )
        self._rate_limiters[key] = bucket
        return bucket

    def register_validator(
        self,
        constraint_type: str,
        validator: ConstraintValidator,
    ) -> None:
        """
        Register a custom constraint validator.

        Args:
            constraint_type: Type of constraint
            validator: Validation function
        """
        self._custom_validators[constraint_type] = validator

    async def validate_operation(
        self,
        context: OperationContext,
    ) -> ValidationResult:
        """
        Validate an operation against all constraints.

        Args:
            context: Operation context

        Returns:
            Validation result
        """
        start_time = time.perf_counter()
        violations: List[SafetyViolation] = []
        warnings: List[str] = []
        checked = 0

        # Check if halted
        if self._halted:
            return ValidationResult(
                is_safe=False,
                violations=[
                    SafetyViolation(
                        constraint_id="system",
                        constraint_name="System Halt",
                        severity=ViolationSeverity.CRITICAL,
                        message="System is halted due to previous critical violation",
                    )
                ],
                checked_constraints=0,
            )

        # Check circuit breakers
        if self.config.enable_circuit_breakers:
            breaker_key = f"{context.agent_id}:{context.operation_type}"
            if breaker_key in self._circuit_breakers:
                breaker = self._circuit_breakers[breaker_key]
                if not breaker.can_execute():
                    self._metrics.operations_blocked += 1
                    return ValidationResult(
                        is_safe=False,
                        violations=[
                            SafetyViolation(
                                constraint_id="circuit_breaker",
                                constraint_name=f"Circuit Breaker: {breaker_key}",
                                severity=ViolationSeverity.MAJOR,
                                message="Circuit breaker is open",
                                metadata=breaker.to_dict(),
                            )
                        ],
                        checked_constraints=1,
                    )

        # Check rate limits
        if self.config.enable_rate_limiting:
            rate_key = f"{context.agent_id}:{context.operation_type}"
            if rate_key in self._rate_limiters:
                bucket = self._rate_limiters[rate_key]
                if not bucket.try_consume():
                    self._metrics.rate_limits_triggered += 1
                    violations.append(
                        SafetyViolation(
                            constraint_id="rate_limit",
                            constraint_name=f"Rate Limit: {rate_key}",
                            severity=ViolationSeverity.MAJOR,
                            message="Rate limit exceeded",
                            operation_id=context.operation_id,
                            agent_id=context.agent_id,
                        )
                    )

        # Check constraints
        if self.config.enable_constraint_validation:
            for constraint in self._constraints.values():
                if not constraint.enabled:
                    continue

                checked += 1
                violation = await self._check_constraint(context, constraint)
                if violation:
                    violations.append(violation)
                    self._record_violation(violation)

        # Calculate result
        validation_time = (time.perf_counter() - start_time) * 1000
        self._update_validation_metrics(validation_time, len(violations) == 0)

        # Determine if safe
        has_critical = any(
            v.severity == ViolationSeverity.CRITICAL for v in violations
        )
        has_major = any(
            v.severity == ViolationSeverity.MAJOR for v in violations
        )

        is_safe = not has_critical and not has_major

        # Handle critical violations
        if has_critical and self.config.halt_on_critical:
            self._halted = True
            logger.critical("SAFETY HALT: Critical violation detected")

        return ValidationResult(
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            checked_constraints=checked,
            validation_time_ms=validation_time,
        )

    async def _check_constraint(
        self,
        context: OperationContext,
        constraint: SafetyConstraint,
    ) -> Optional[SafetyViolation]:
        """
        Check a single constraint.

        Args:
            context: Operation context
            constraint: Constraint to check

        Returns:
            Violation if constraint is violated, None otherwise
        """
        validator_key = constraint.constraint_type.value
        validator = self._custom_validators.get(validator_key)

        if validator:
            return validator(context, constraint)

        return None

    def _validate_threshold(
        self,
        context: OperationContext,
        constraint: SafetyConstraint,
    ) -> Optional[SafetyViolation]:
        """Validate threshold constraint."""
        # Get value from parameters
        param_name = constraint.metadata.get("parameter", constraint.name.lower())
        value = context.parameters.get(param_name)

        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        # Check bounds
        if constraint.max_value is not None and value > constraint.max_value:
            severity = (
                ViolationSeverity.CRITICAL
                if constraint.level == SafetyLevel.CRITICAL
                else ViolationSeverity.MAJOR
            )
            return SafetyViolation(
                constraint_id=constraint.constraint_id,
                constraint_name=constraint.name,
                severity=severity,
                message=f"{param_name} ({value}) exceeds maximum ({constraint.max_value})",
                actual_value=value,
                expected_value=f"<= {constraint.max_value}",
                operation_id=context.operation_id,
                agent_id=context.agent_id,
            )

        if constraint.min_value is not None and value < constraint.min_value:
            severity = (
                ViolationSeverity.CRITICAL
                if constraint.level == SafetyLevel.CRITICAL
                else ViolationSeverity.MAJOR
            )
            return SafetyViolation(
                constraint_id=constraint.constraint_id,
                constraint_name=constraint.name,
                severity=severity,
                message=f"{param_name} ({value}) below minimum ({constraint.min_value})",
                actual_value=value,
                expected_value=f">= {constraint.min_value}",
                operation_id=context.operation_id,
                agent_id=context.agent_id,
            )

        return None

    def _validate_rate_limit(
        self,
        context: OperationContext,
        constraint: SafetyConstraint,
    ) -> Optional[SafetyViolation]:
        """Validate rate limit constraint."""
        if not constraint.rate_limit:
            return None

        key = f"{constraint.constraint_id}:{context.agent_id}"

        if key not in self._rate_limiters:
            self._rate_limiters[key] = RateLimitBucket(
                key=key,
                max_tokens=constraint.rate_limit,
                refill_rate=constraint.rate_limit / constraint.rate_window_seconds,
            )

        bucket = self._rate_limiters[key]
        if not bucket.try_consume():
            return SafetyViolation(
                constraint_id=constraint.constraint_id,
                constraint_name=constraint.name,
                severity=ViolationSeverity.MAJOR,
                message=f"Rate limit exceeded: {constraint.rate_limit} per {constraint.rate_window_seconds}s",
                operation_id=context.operation_id,
                agent_id=context.agent_id,
            )

        return None

    def _validate_resource(
        self,
        context: OperationContext,
        constraint: SafetyConstraint,
    ) -> Optional[SafetyViolation]:
        """Validate resource constraint."""
        required_resource = constraint.metadata.get("resource")
        if not required_resource:
            return None

        available_resources = context.metadata.get("available_resources", [])

        if required_resource not in available_resources:
            return SafetyViolation(
                constraint_id=constraint.constraint_id,
                constraint_name=constraint.name,
                severity=ViolationSeverity.MAJOR,
                message=f"Required resource not available: {required_resource}",
                operation_id=context.operation_id,
                agent_id=context.agent_id,
            )

        return None

    def _record_violation(self, violation: SafetyViolation) -> None:
        """Record a violation."""
        self._violations.append(violation)
        self._metrics.violations_total += 1

        if violation.severity == ViolationSeverity.CRITICAL:
            self._metrics.violations_critical += 1
            logger.critical(f"CRITICAL VIOLATION: {violation.message}")
        elif violation.severity == ViolationSeverity.MAJOR:
            self._metrics.violations_major += 1
            logger.error(f"MAJOR VIOLATION: {violation.message}")
        else:
            self._metrics.violations_minor += 1
            logger.warning(f"MINOR VIOLATION: {violation.message}")

    def _update_validation_metrics(
        self,
        validation_time_ms: float,
        passed: bool,
    ) -> None:
        """Update validation metrics."""
        self._metrics.validations_performed += 1

        if passed:
            self._metrics.validations_passed += 1
        else:
            self._metrics.validations_failed += 1

        self._validation_times.append(validation_time_ms)
        if len(self._validation_times) > 1000:
            self._validation_times = self._validation_times[-1000:]

        self._metrics.avg_validation_time_ms = (
            sum(self._validation_times) / len(self._validation_times)
        )
        self._metrics.last_updated = datetime.now(timezone.utc).isoformat()

    def record_success(self, agent_id: str, operation_type: str) -> None:
        """
        Record a successful operation for circuit breaker.

        Args:
            agent_id: Agent ID
            operation_type: Operation type
        """
        breaker_key = f"{agent_id}:{operation_type}"
        if breaker_key in self._circuit_breakers:
            self._circuit_breakers[breaker_key].record_success()

    def record_failure(self, agent_id: str, operation_type: str) -> None:
        """
        Record a failed operation for circuit breaker.

        Args:
            agent_id: Agent ID
            operation_type: Operation type
        """
        breaker_key = f"{agent_id}:{operation_type}"
        if breaker_key in self._circuit_breakers:
            breaker = self._circuit_breakers[breaker_key]
            old_state = breaker.state
            breaker.record_failure()
            if old_state != CircuitState.OPEN and breaker.state == CircuitState.OPEN:
                self._metrics.circuit_breakers_opened += 1

    def is_halted(self) -> bool:
        """Check if system is halted."""
        return self._halted

    def reset_halt(self) -> None:
        """Reset halt state."""
        self._halted = False
        logger.info("Safety halt reset")

    def is_safe_to_proceed(self) -> bool:
        """Check if it's safe to proceed with operations."""
        return not self._halted

    def get_violations(
        self,
        severity: Optional[ViolationSeverity] = None,
        resolved: Optional[bool] = None,
    ) -> List[SafetyViolation]:
        """
        Get violations with optional filtering.

        Args:
            severity: Filter by severity
            resolved: Filter by resolved status

        Returns:
            List of violations
        """
        violations = self._violations

        if severity:
            violations = [v for v in violations if v.severity == severity]

        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]

        return violations

    def resolve_violation(self, violation_id: str) -> bool:
        """
        Mark a violation as resolved.

        Args:
            violation_id: Violation to resolve

        Returns:
            True if violation was found and resolved
        """
        for violation in self._violations:
            if violation.violation_id == violation_id:
                violation.resolved = True
                return True
        return False

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.to_dict()
            for name, breaker in self._circuit_breakers.items()
        }

    def get_metrics(self) -> SafetyMetrics:
        """Get current safety metrics."""
        return self._metrics

    def get_constraints(self) -> Dict[str, SafetyConstraint]:
        """Get all constraints."""
        return self._constraints.copy()


# Factory function
def create_safety_monitor(
    halt_on_critical: bool = True,
    enable_rate_limiting: bool = True,
    enable_circuit_breakers: bool = True,
) -> SafetyMonitor:
    """
    Create a safety monitor with common configurations.

    Args:
        halt_on_critical: Halt on critical violations
        enable_rate_limiting: Enable rate limiting
        enable_circuit_breakers: Enable circuit breakers

    Returns:
        Configured SafetyMonitor instance
    """
    config = SafetyConfig(
        halt_on_critical=halt_on_critical,
        enable_rate_limiting=enable_rate_limiting,
        enable_circuit_breakers=enable_circuit_breakers,
    )
    return SafetyMonitor(config)


__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "ConstraintType",
    "ConstraintValidator",
    "OperationContext",
    "RateLimitBucket",
    "SafetyConfig",
    "SafetyConstraint",
    "SafetyLevel",
    "SafetyMetrics",
    "SafetyMonitor",
    "SafetyViolation",
    "ValidationResult",
    "ViolationSeverity",
    "create_safety_monitor",
]
