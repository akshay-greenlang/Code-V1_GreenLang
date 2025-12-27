"""
GreenLang Framework - Guardrails Integration Module

Provides reusable guardrails integration for all GreenLang agents through:
- @with_guardrails decorator for wrapping agent functions
- Input validation, output checking, and action gating
- Provenance tracking for all violations
- Configurable guardrail profiles for different agent types

Based on:
- OWASP LLM Top 10 (2025)
- NIST AI RMF 1.0
- EU AI Act High-Risk System Requirements

Example:
    >>> from greenlang.shared.guardrails_integration import with_guardrails, GuardrailProfile
    >>>
    >>> @with_guardrails(profile=GuardrailProfile.STRICT)
    ... def calculate_emissions(data: dict) -> dict:
    ...     return {"emissions": data["value"] * 2.5}
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    Awaitable,
)
import asyncio
import hashlib
import json
import logging
import time
import uuid


# Import from advanced guardrails module
from ..advanced.guardrails import (
    GuardrailType,
    ViolationSeverity,
    ActionType,
    GuardrailViolation,
    GuardrailResult,
    Guardrail,
    PromptInjectionGuardrail,
    DataLeakageGuardrail,
    ActionGate,
    SafetyEnvelopeGuardrail,
    PolicyEnforcementGuardrail,
    GuardrailOrchestrator,
    GREENLANG_GUARDRAILS,
    create_default_orchestrator,
)


logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class GuardrailProfile(Enum):
    """
    Predefined guardrail profiles for different security levels.

    MINIMAL: Basic input validation only
    STANDARD: Input + output validation
    STRICT: Full validation with action gating
    INDUSTRIAL: Strict + physical safety constraints
    REGULATORY: Full compliance mode with all guardrails
    """
    MINIMAL = auto()
    STANDARD = auto()
    STRICT = auto()
    INDUSTRIAL = auto()
    REGULATORY = auto()


class GuardrailMode(Enum):
    """
    Guardrail enforcement modes.

    ENFORCE: Block on violations (default)
    WARN: Log violations but continue execution
    AUDIT: Log all checks for compliance auditing
    """
    ENFORCE = auto()
    WARN = auto()
    AUDIT = auto()


@dataclass
class ViolationRecord:
    """
    Complete record of a guardrail violation with provenance.

    Provides SHA-256 hashing for audit trail and regulatory compliance.
    """
    record_id: str
    violation: GuardrailViolation
    agent_id: str
    function_name: str
    input_hash: str
    output_hash: Optional[str]
    timestamp: datetime
    execution_context: Dict[str, Any]
    provenance_hash: str

    def __post_init__(self):
        """Generate record_id if not provided."""
        if not self.record_id:
            self.record_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "violation": self.violation.to_dict(),
            "agent_id": self.agent_id,
            "function_name": self.function_name,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "timestamp": self.timestamp.isoformat(),
            "execution_context": self.execution_context,
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ViolationRecord":
        """Create ViolationRecord from dictionary."""
        violation_data = data["violation"]
        violation = GuardrailViolation(
            violation_id=violation_data["violation_id"],
            guardrail_name=violation_data["guardrail_name"],
            guardrail_type=GuardrailType[violation_data["guardrail_type"]],
            severity=ViolationSeverity[violation_data["severity"]],
            message=violation_data["message"],
            context=violation_data.get("context", {}),
            timestamp=datetime.fromisoformat(violation_data["timestamp"]),
            agent_id=violation_data.get("agent_id", ""),
            remediation=violation_data.get("remediation", ""),
        )
        return cls(
            record_id=data["record_id"],
            violation=violation,
            agent_id=data["agent_id"],
            function_name=data["function_name"],
            input_hash=data["input_hash"],
            output_hash=data.get("output_hash"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            execution_context=data.get("execution_context", {}),
            provenance_hash=data["provenance_hash"],
        )


@dataclass
class GuardrailExecutionResult:
    """
    Result of a guarded function execution.

    Contains both the function result and guardrail check information.
    """
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    input_check: Optional[GuardrailResult] = None
    output_check: Optional[GuardrailResult] = None
    action_check: Optional[GuardrailResult] = None
    violation_records: List[ViolationRecord] = field(default_factory=list)
    execution_time_ms: float = 0.0
    provenance_hash: str = ""

    @property
    def has_violations(self) -> bool:
        """Check if any violations occurred."""
        return len(self.violation_records) > 0

    @property
    def blocking_violations(self) -> List[ViolationRecord]:
        """Get all blocking violations."""
        return [
            r for r in self.violation_records
            if r.violation.severity == ViolationSeverity.BLOCKING
        ]

    @property
    def critical_violations(self) -> List[ViolationRecord]:
        """Get all critical or blocking violations."""
        return [
            r for r in self.violation_records
            if r.violation.severity in (
                ViolationSeverity.CRITICAL,
                ViolationSeverity.BLOCKING
            )
        ]


class ViolationLogger:
    """
    Logger for guardrail violations with provenance tracking.

    Maintains an audit trail of all violations with SHA-256 hashes
    for regulatory compliance.
    """

    def __init__(
        self,
        agent_id: str = "unknown",
        max_history: int = 10000,
    ):
        """
        Initialize violation logger.

        Args:
            agent_id: Agent identifier for provenance tracking
            max_history: Maximum number of records to keep in memory
        """
        self.agent_id = agent_id
        self.max_history = max_history
        self._records: List[ViolationRecord] = []
        self._violation_counts: Dict[str, int] = {}

    def log_violation(
        self,
        violation: GuardrailViolation,
        function_name: str,
        input_data: Any,
        output_data: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ViolationRecord:
        """
        Log a guardrail violation with provenance tracking.

        Args:
            violation: The guardrail violation
            function_name: Name of the function that triggered the violation
            input_data: Input data to the function
            output_data: Output data from the function (if available)
            context: Additional execution context

        Returns:
            ViolationRecord with provenance hash
        """
        timestamp = datetime.now(timezone.utc)

        # Compute hashes
        input_hash = self._compute_hash(input_data)
        output_hash = self._compute_hash(output_data) if output_data else None

        # Compute provenance hash
        provenance_data = {
            "violation_id": violation.violation_id,
            "agent_id": self.agent_id,
            "function_name": function_name,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = self._compute_hash(provenance_data)

        # Create record
        record = ViolationRecord(
            record_id=str(uuid.uuid4()),
            violation=violation,
            agent_id=self.agent_id,
            function_name=function_name,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=timestamp,
            execution_context=context or {},
            provenance_hash=provenance_hash,
        )

        # Store record
        self._records.append(record)
        if len(self._records) > self.max_history:
            self._records = self._records[-self.max_history:]

        # Update counts
        key = f"{violation.guardrail_name}:{violation.severity.name}"
        self._violation_counts[key] = self._violation_counts.get(key, 0) + 1

        # Log the violation
        log_level = self._get_log_level(violation.severity)
        logger.log(
            log_level,
            f"Guardrail violation [{violation.guardrail_name}]: {violation.message}",
            extra={
                "violation_id": violation.violation_id,
                "provenance_hash": provenance_hash,
                "agent_id": self.agent_id,
                "function_name": function_name,
            }
        )

        return record

    def get_records(
        self,
        guardrail_name: Optional[str] = None,
        severity: Optional[ViolationSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ViolationRecord]:
        """
        Get violation records with optional filtering.

        Args:
            guardrail_name: Filter by guardrail name
            severity: Filter by severity level
            start_time: Filter records after this time
            end_time: Filter records before this time
            limit: Maximum number of records to return

        Returns:
            List of matching ViolationRecords
        """
        records = self._records

        if guardrail_name:
            records = [r for r in records if r.violation.guardrail_name == guardrail_name]

        if severity:
            records = [r for r in records if r.violation.severity == severity]

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]

        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        return records[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all violations.

        Returns:
            Dictionary with violation counts and statistics
        """
        severity_counts = {}
        guardrail_counts = {}

        for record in self._records:
            sev = record.violation.severity.name
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

            name = record.violation.guardrail_name
            guardrail_counts[name] = guardrail_counts.get(name, 0) + 1

        return {
            "total_violations": len(self._records),
            "by_severity": severity_counts,
            "by_guardrail": guardrail_counts,
            "agent_id": self.agent_id,
        }

    def export_records(self, format: str = "json") -> str:
        """
        Export all records for compliance reporting.

        Args:
            format: Export format (currently only "json" supported)

        Returns:
            JSON string of all records
        """
        records_data = [r.to_dict() for r in self._records]
        return json.dumps(records_data, indent=2, default=str)

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        if data is None:
            return hashlib.sha256(b"null").hexdigest()

        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            json_str = str(data)

        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _get_log_level(self, severity: ViolationSeverity) -> int:
        """Get logging level for severity."""
        level_map = {
            ViolationSeverity.INFO: logging.INFO,
            ViolationSeverity.WARNING: logging.WARNING,
            ViolationSeverity.ERROR: logging.ERROR,
            ViolationSeverity.CRITICAL: logging.CRITICAL,
            ViolationSeverity.BLOCKING: logging.CRITICAL,
        }
        return level_map.get(severity, logging.WARNING)


class GuardrailsIntegration:
    """
    Main integration class for applying guardrails to agent functions.

    Provides a unified interface for:
    - Input validation
    - Output checking
    - Action gating
    - Provenance tracking

    Example:
        >>> integration = GuardrailsIntegration(agent_id="GL-006")
        >>> result = integration.execute_guarded(my_function, input_data)
        >>> if result.success:
        ...     print(result.result)
    """

    def __init__(
        self,
        agent_id: str,
        orchestrator: Optional[GuardrailOrchestrator] = None,
        profile: GuardrailProfile = GuardrailProfile.STANDARD,
        mode: GuardrailMode = GuardrailMode.ENFORCE,
    ):
        """
        Initialize guardrails integration.

        Args:
            agent_id: Agent identifier for provenance tracking
            orchestrator: Custom guardrail orchestrator (uses default if None)
            profile: Guardrail profile to use
            mode: Enforcement mode
        """
        self.agent_id = agent_id
        self.profile = profile
        self.mode = mode
        self.orchestrator = orchestrator or self._create_orchestrator_for_profile(profile)
        self.violation_logger = ViolationLogger(agent_id=agent_id)

    def _create_orchestrator_for_profile(
        self,
        profile: GuardrailProfile
    ) -> GuardrailOrchestrator:
        """Create an orchestrator configured for the given profile."""
        orchestrator = GuardrailOrchestrator()

        if profile == GuardrailProfile.MINIMAL:
            # Basic input validation only
            orchestrator.add_input_guardrail(PromptInjectionGuardrail())

        elif profile == GuardrailProfile.STANDARD:
            # Input + output validation
            orchestrator.add_input_guardrail(PromptInjectionGuardrail())
            orchestrator.add_output_guardrail(DataLeakageGuardrail())

        elif profile == GuardrailProfile.STRICT:
            # Full validation with action gating
            orchestrator.add_input_guardrail(PromptInjectionGuardrail())
            orchestrator.add_output_guardrail(DataLeakageGuardrail())
            orchestrator.add_action_guardrail(ActionGate(
                allowed_actions={
                    ActionType.READ,
                    ActionType.RECOMMEND,
                    ActionType.OPTIMIZE,
                },
                max_actions_per_minute=60,
            ))

        elif profile == GuardrailProfile.INDUSTRIAL:
            # Strict + physical safety constraints
            orchestrator.add_input_guardrail(PromptInjectionGuardrail())
            orchestrator.add_output_guardrail(DataLeakageGuardrail())
            orchestrator.add_action_guardrail(ActionGate(
                allowed_actions={
                    ActionType.READ,
                    ActionType.RECOMMEND,
                    ActionType.OPTIMIZE,
                },
                max_actions_per_minute=60,
                require_confirmation_for={ActionType.CONTROL, ActionType.EXECUTE},
            ))
            orchestrator.add_action_guardrail(SafetyEnvelopeGuardrail())

        elif profile == GuardrailProfile.REGULATORY:
            # Full compliance mode with all guardrails
            orchestrator.add_input_guardrail(PromptInjectionGuardrail())
            orchestrator.add_output_guardrail(DataLeakageGuardrail())
            orchestrator.add_action_guardrail(ActionGate(
                allowed_actions={
                    ActionType.READ,
                    ActionType.RECOMMEND,
                },
                max_actions_per_minute=30,
                require_confirmation_for={
                    ActionType.WRITE,
                    ActionType.CONTROL,
                    ActionType.EXECUTE,
                    ActionType.OPTIMIZE,
                },
            ))
            orchestrator.add_action_guardrail(SafetyEnvelopeGuardrail())
            orchestrator.add_action_guardrail(PolicyEnforcementGuardrail())

        return orchestrator

    def check_input(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Check input data against all input guardrails.

        Args:
            input_data: Input data to check
            context: Additional context for evaluation

        Returns:
            GuardrailResult with any violations
        """
        return self.orchestrator.check_input(input_data, context or {})

    def check_output(
        self,
        output_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Check output data against all output guardrails.

        Args:
            output_data: Output data to check
            context: Additional context for evaluation

        Returns:
            GuardrailResult with any violations
        """
        return self.orchestrator.check_output(output_data, context or {})

    def check_action(
        self,
        action_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """
        Check action against all action guardrails.

        Args:
            action_data: Action data to check
            context: Additional context with action_type

        Returns:
            GuardrailResult with any violations
        """
        return self.orchestrator.check_action(action_data, context or {})

    def execute_guarded(
        self,
        func: Callable[..., T],
        *args: Any,
        function_name: Optional[str] = None,
        action_type: Optional[ActionType] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GuardrailExecutionResult:
        """
        Execute a function with full guardrail protection.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to function
            function_name: Name for logging (defaults to func.__name__)
            action_type: Type of action for action gating
            context: Additional execution context
            **kwargs: Keyword arguments to pass to function

        Returns:
            GuardrailExecutionResult with result and violation information
        """
        start_time = time.time()
        func_name = function_name or getattr(func, '__name__', 'unknown')
        ctx = context or {}

        # Prepare input data for checking
        input_data = {"args": args, "kwargs": kwargs}

        violation_records: List[ViolationRecord] = []

        # Step 1: Check input
        input_result = self.check_input(input_data, ctx)
        for violation in input_result.violations:
            record = self.violation_logger.log_violation(
                violation=violation,
                function_name=func_name,
                input_data=input_data,
                context=ctx,
            )
            violation_records.append(record)

        # In ENFORCE mode, block on input violations
        if self.mode == GuardrailMode.ENFORCE and input_result.has_blocking_violation:
            return GuardrailExecutionResult(
                success=False,
                error=GuardrailViolationError(
                    f"Input blocked by guardrails: {input_result.violations[0].message}"
                ),
                input_check=input_result,
                violation_records=violation_records,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 2: Check action (if action_type provided)
        action_result = None
        if action_type:
            action_ctx = {**ctx, "action_type": action_type}
            action_result = self.check_action(input_data, action_ctx)
            for violation in action_result.violations:
                record = self.violation_logger.log_violation(
                    violation=violation,
                    function_name=func_name,
                    input_data=input_data,
                    context=action_ctx,
                )
                violation_records.append(record)

            # In ENFORCE mode, block on action violations
            if self.mode == GuardrailMode.ENFORCE and action_result.has_blocking_violation:
                return GuardrailExecutionResult(
                    success=False,
                    error=GuardrailViolationError(
                        f"Action blocked by guardrails: {action_result.violations[0].message}"
                    ),
                    input_check=input_result,
                    action_check=action_result,
                    violation_records=violation_records,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        # Step 3: Execute function
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Function execution failed: {func_name}",
                exc_info=True,
                extra={"agent_id": self.agent_id}
            )
            return GuardrailExecutionResult(
                success=False,
                error=e,
                input_check=input_result,
                action_check=action_result,
                violation_records=violation_records,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 4: Check output
        output_result = self.check_output(result, ctx)
        for violation in output_result.violations:
            record = self.violation_logger.log_violation(
                violation=violation,
                function_name=func_name,
                input_data=input_data,
                output_data=result,
                context=ctx,
            )
            violation_records.append(record)

        # In ENFORCE mode, block on output violations
        if self.mode == GuardrailMode.ENFORCE and output_result.has_blocking_violation:
            return GuardrailExecutionResult(
                success=False,
                result=None,  # Suppress result on output violation
                error=GuardrailViolationError(
                    f"Output blocked by guardrails: {output_result.violations[0].message}"
                ),
                input_check=input_result,
                output_check=output_result,
                action_check=action_result,
                violation_records=violation_records,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Compute provenance hash
        execution_time_ms = (time.time() - start_time) * 1000
        provenance_data = {
            "agent_id": self.agent_id,
            "function_name": func_name,
            "input_hash": self.violation_logger._compute_hash(input_data),
            "output_hash": self.violation_logger._compute_hash(result),
            "execution_time_ms": execution_time_ms,
            "violation_count": len(violation_records),
        }
        provenance_hash = self.violation_logger._compute_hash(provenance_data)

        return GuardrailExecutionResult(
            success=True,
            result=result,
            input_check=input_result,
            output_check=output_result,
            action_check=action_result,
            violation_records=violation_records,
            execution_time_ms=execution_time_ms,
            provenance_hash=provenance_hash,
        )

    async def execute_guarded_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        function_name: Optional[str] = None,
        action_type: Optional[ActionType] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GuardrailExecutionResult:
        """
        Execute an async function with full guardrail protection.

        Args:
            func: Async function to execute
            *args: Positional arguments to pass to function
            function_name: Name for logging (defaults to func.__name__)
            action_type: Type of action for action gating
            context: Additional execution context
            **kwargs: Keyword arguments to pass to function

        Returns:
            GuardrailExecutionResult with result and violation information
        """
        start_time = time.time()
        func_name = function_name or getattr(func, '__name__', 'unknown')
        ctx = context or {}

        # Prepare input data for checking
        input_data = {"args": args, "kwargs": kwargs}

        violation_records: List[ViolationRecord] = []

        # Step 1: Check input
        input_result = self.check_input(input_data, ctx)
        for violation in input_result.violations:
            record = self.violation_logger.log_violation(
                violation=violation,
                function_name=func_name,
                input_data=input_data,
                context=ctx,
            )
            violation_records.append(record)

        if self.mode == GuardrailMode.ENFORCE and input_result.has_blocking_violation:
            return GuardrailExecutionResult(
                success=False,
                error=GuardrailViolationError(
                    f"Input blocked by guardrails: {input_result.violations[0].message}"
                ),
                input_check=input_result,
                violation_records=violation_records,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 2: Check action
        action_result = None
        if action_type:
            action_ctx = {**ctx, "action_type": action_type}
            action_result = self.check_action(input_data, action_ctx)
            for violation in action_result.violations:
                record = self.violation_logger.log_violation(
                    violation=violation,
                    function_name=func_name,
                    input_data=input_data,
                    context=action_ctx,
                )
                violation_records.append(record)

            if self.mode == GuardrailMode.ENFORCE and action_result.has_blocking_violation:
                return GuardrailExecutionResult(
                    success=False,
                    error=GuardrailViolationError(
                        f"Action blocked by guardrails: {action_result.violations[0].message}"
                    ),
                    input_check=input_result,
                    action_check=action_result,
                    violation_records=violation_records,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        # Step 3: Execute async function
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Async function execution failed: {func_name}",
                exc_info=True,
                extra={"agent_id": self.agent_id}
            )
            return GuardrailExecutionResult(
                success=False,
                error=e,
                input_check=input_result,
                action_check=action_result,
                violation_records=violation_records,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Step 4: Check output
        output_result = self.check_output(result, ctx)
        for violation in output_result.violations:
            record = self.violation_logger.log_violation(
                violation=violation,
                function_name=func_name,
                input_data=input_data,
                output_data=result,
                context=ctx,
            )
            violation_records.append(record)

        if self.mode == GuardrailMode.ENFORCE and output_result.has_blocking_violation:
            return GuardrailExecutionResult(
                success=False,
                result=None,
                error=GuardrailViolationError(
                    f"Output blocked by guardrails: {output_result.violations[0].message}"
                ),
                input_check=input_result,
                output_check=output_result,
                action_check=action_result,
                violation_records=violation_records,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        execution_time_ms = (time.time() - start_time) * 1000
        provenance_data = {
            "agent_id": self.agent_id,
            "function_name": func_name,
            "input_hash": self.violation_logger._compute_hash(input_data),
            "output_hash": self.violation_logger._compute_hash(result),
            "execution_time_ms": execution_time_ms,
            "violation_count": len(violation_records),
        }
        provenance_hash = self.violation_logger._compute_hash(provenance_data)

        return GuardrailExecutionResult(
            success=True,
            result=result,
            input_check=input_result,
            output_check=output_result,
            action_check=action_result,
            violation_records=violation_records,
            execution_time_ms=execution_time_ms,
            provenance_hash=provenance_hash,
        )


class GuardrailViolationError(Exception):
    """Exception raised when a guardrail violation blocks execution."""

    def __init__(
        self,
        message: str,
        violations: Optional[List[GuardrailViolation]] = None
    ):
        super().__init__(message)
        self.violations = violations or []


# Global integration instances per agent
_integrations: Dict[str, GuardrailsIntegration] = {}


def get_integration(
    agent_id: str,
    profile: GuardrailProfile = GuardrailProfile.STANDARD,
    mode: GuardrailMode = GuardrailMode.ENFORCE,
) -> GuardrailsIntegration:
    """
    Get or create a GuardrailsIntegration for an agent.

    Args:
        agent_id: Agent identifier
        profile: Guardrail profile to use
        mode: Enforcement mode

    Returns:
        GuardrailsIntegration instance
    """
    key = f"{agent_id}:{profile.name}:{mode.name}"
    if key not in _integrations:
        _integrations[key] = GuardrailsIntegration(
            agent_id=agent_id,
            profile=profile,
            mode=mode,
        )
    return _integrations[key]


def with_guardrails(
    agent_id: Optional[str] = None,
    profile: GuardrailProfile = GuardrailProfile.STANDARD,
    mode: GuardrailMode = GuardrailMode.ENFORCE,
    action_type: Optional[ActionType] = None,
    return_full_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator for wrapping functions with guardrail protection.

    Provides input validation, output checking, action gating, and
    provenance tracking for any function.

    Args:
        agent_id: Agent identifier (uses function module if not provided)
        profile: Guardrail profile to use
        mode: Enforcement mode (ENFORCE, WARN, or AUDIT)
        action_type: Type of action for action gating
        return_full_result: If True, return GuardrailExecutionResult instead of just result

    Returns:
        Decorated function

    Example:
        >>> @with_guardrails(agent_id="GL-006", profile=GuardrailProfile.STRICT)
        ... def calculate_emissions(data: dict) -> dict:
        ...     return {"emissions": data["value"] * 2.5}
        ...
        >>> result = calculate_emissions({"value": 100})
        >>> # Returns {"emissions": 250.0}

        >>> @with_guardrails(agent_id="GL-006", return_full_result=True)
        ... def risky_operation(data: dict) -> dict:
        ...     return {"result": data["value"]}
        ...
        >>> execution_result = risky_operation({"value": 100})
        >>> print(execution_result.provenance_hash)
    """
    def decorator(func: F) -> F:
        # Determine agent_id
        aid = agent_id or getattr(func, '__module__', 'unknown').split('.')[-1]

        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                integration = get_integration(aid, profile, mode)
                execution_result = await integration.execute_guarded_async(
                    func,
                    *args,
                    function_name=func.__name__,
                    action_type=action_type,
                    **kwargs,
                )

                if return_full_result:
                    return execution_result

                if not execution_result.success:
                    if execution_result.error:
                        raise execution_result.error
                    raise GuardrailViolationError("Execution blocked by guardrails")

                return execution_result.result

            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                integration = get_integration(aid, profile, mode)
                execution_result = integration.execute_guarded(
                    func,
                    *args,
                    function_name=func.__name__,
                    action_type=action_type,
                    **kwargs,
                )

                if return_full_result:
                    return execution_result

                if not execution_result.success:
                    if execution_result.error:
                        raise execution_result.error
                    raise GuardrailViolationError("Execution blocked by guardrails")

                return execution_result.result

            return sync_wrapper  # type: ignore

    return decorator


def validate_input(
    agent_id: str = "unknown",
    profile: GuardrailProfile = GuardrailProfile.STANDARD,
) -> Callable[[F], F]:
    """
    Decorator for input validation only (no output checking).

    Lighter weight than @with_guardrails for functions that only need
    input validation.

    Args:
        agent_id: Agent identifier
        profile: Guardrail profile to use

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            integration = get_integration(agent_id, profile, GuardrailMode.ENFORCE)

            # Check input only
            input_data = {"args": args, "kwargs": kwargs}
            input_result = integration.check_input(input_data, {})

            if input_result.has_blocking_violation:
                raise GuardrailViolationError(
                    f"Input blocked: {input_result.violations[0].message}",
                    violations=input_result.violations,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_output(
    agent_id: str = "unknown",
    profile: GuardrailProfile = GuardrailProfile.STANDARD,
) -> Callable[[F], F]:
    """
    Decorator for output validation only (no input checking).

    Lighter weight than @with_guardrails for functions that only need
    output validation.

    Args:
        agent_id: Agent identifier
        profile: Guardrail profile to use

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            integration = get_integration(agent_id, profile, GuardrailMode.ENFORCE)
            output_result = integration.check_output(result, {})

            if output_result.has_blocking_violation:
                raise GuardrailViolationError(
                    f"Output blocked: {output_result.violations[0].message}",
                    violations=output_result.violations,
                )

            return result

        return wrapper  # type: ignore

    return decorator


# Re-export commonly used types from advanced.guardrails
__all__ = [
    # Core classes
    "GuardrailsIntegration",
    "GuardrailExecutionResult",
    "ViolationRecord",
    "ViolationLogger",
    "GuardrailViolationError",
    # Enums
    "GuardrailProfile",
    "GuardrailMode",
    # Decorators
    "with_guardrails",
    "validate_input",
    "validate_output",
    # Utility functions
    "get_integration",
    # Re-exports from advanced.guardrails
    "GuardrailType",
    "ViolationSeverity",
    "ActionType",
    "GuardrailViolation",
    "GuardrailResult",
    "Guardrail",
    "GuardrailOrchestrator",
    "GREENLANG_GUARDRAILS",
]
